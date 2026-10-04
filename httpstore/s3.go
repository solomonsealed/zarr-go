package httpstore

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	zarr "github.com/solomonsealed/zarr-go"
)

var (
	_ zarr.WritableStore = (*S3)(nil)
	_ zarr.ListableStore = (*S3)(nil)
)

// S3Config describes a Zarr hierarchy stored under a prefix of an
// S3-compatible bucket (AWS S3, Cloudflare R2, MinIO, GCS interoperability,
// Tigris, ...).
type S3Config struct {
	Bucket string
	// Prefix is the key prefix of the hierarchy's root within the bucket.
	Prefix string
	// Region defaults to "us-east-1"; use "auto" for Cloudflare R2.
	Region string
	// Endpoint overrides the AWS endpoint, e.g.
	// "https://<account>.r2.cloudflarestorage.com" or
	// "http://localhost:9000". Custom endpoints are addressed path-style.
	Endpoint string
	// ForcePathStyle addresses AWS buckets as s3.<region>.amazonaws.com/<bucket>.
	ForcePathStyle bool
	// Anonymous sends unsigned requests (public buckets); so does an empty
	// AccessKeyID.
	Anonymous       bool
	AccessKeyID     string
	SecretAccessKey string
	SessionToken    string
}

// S3 is a zarr store backed by an S3-compatible object store, signing
// requests with AWS Signature Version 4. It can read (with ranged
// requests), write, delete and list.
type S3 struct {
	cfg    S3Config
	client *http.Client
	now    func() time.Time
}

// NewS3 returns a store for cfg; a nil client means http.DefaultClient.
func NewS3(cfg S3Config, client *http.Client) *S3 {
	if client == nil {
		client = http.DefaultClient
	}
	return &S3{cfg: cfg, client: client, now: time.Now}
}

type s3Request struct {
	method string
	url    string
	header map[string]string
	body   []byte
}

const emptySHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

func (s *S3) region() string {
	if s.cfg.Region == "" {
		return "us-east-1"
	}
	return s.cfg.Region
}

// s3EscapePath percent-encodes every byte except RFC 3986 unreserved
// characters and '/', as SigV4 canonical URIs require.
func s3EscapePath(p string) string {
	var b strings.Builder
	for i := 0; i < len(p); i++ {
		ch := p[i]
		if ch == '/' || ch == '-' || ch == '_' || ch == '.' || ch == '~' ||
			('A' <= ch && ch <= 'Z') || ('a' <= ch && ch <= 'z') || ('0' <= ch && ch <= '9') {
			b.WriteByte(ch)
		} else {
			fmt.Fprintf(&b, "%%%02X", ch)
		}
	}
	return b.String()
}

func s3EscapeQuery(s string) string { return strings.ReplaceAll(s3EscapePath(s), "/", "%2F") }

// target returns scheme://host, the host, and the escaped path of an
// object key (relative to the bucket).
func (s *S3) target(key string) (base, host, path string, err error) {
	if s.cfg.Endpoint != "" {
		u, err := url.Parse(strings.TrimRight(s.cfg.Endpoint, "/"))
		if err != nil {
			return "", "", "", fmt.Errorf("httpstore: invalid S3 endpoint %q: %w", s.cfg.Endpoint, err)
		}
		basePath := strings.TrimRight(u.Path, "/")
		if !strings.HasPrefix(u.Host, s.cfg.Bucket+".") {
			basePath += "/" + s.cfg.Bucket
		}
		return u.Scheme + "://" + u.Host, u.Host, s3EscapePath(basePath + "/" + key), nil
	}
	if s.cfg.ForcePathStyle {
		host = "s3." + s.region() + ".amazonaws.com"
		return "https://" + host, host, s3EscapePath("/" + s.cfg.Bucket + "/" + key), nil
	}
	host = s.cfg.Bucket + ".s3." + s.region() + ".amazonaws.com"
	return "https://" + host, host, s3EscapePath("/" + key), nil
}

func (s *S3) fullKey(key string) string {
	p := strings.Trim(s.cfg.Prefix, "/")
	if p == "" {
		return key
	}
	if key == "" {
		return p
	}
	return p + "/" + key
}

func (s *S3) request(method, key, query string, header map[string]string, body []byte) (*s3Request, error) {
	base, host, path, err := s.target(key)
	if err != nil {
		return nil, err
	}
	if key == "" && !strings.HasSuffix(path, "/") {
		path += "/"
	}
	u := base + path
	if query != "" {
		u += "?" + query
	}
	if header == nil {
		header = map[string]string{}
	}
	r := &s3Request{method: method, url: u, header: header, body: body}
	hash := emptySHA256
	if body != nil {
		sum := sha256.Sum256(body)
		hash = hex.EncodeToString(sum[:])
	}
	s.sign(r, host, path, query, hash, s.now())
	return r, nil
}

// sign adds SigV4 headers to r.
func (s *S3) sign(r *s3Request, host, path, query, payloadHash string, now time.Time) {
	c := s.cfg
	if c.Anonymous || c.AccessKeyID == "" {
		return
	}
	now = now.UTC()
	amzDate := now.Format("20060102T150405Z")
	date := amzDate[:8]
	r.header["x-amz-date"] = amzDate
	r.header["x-amz-content-sha256"] = payloadHash
	signed := map[string]string{"host": host, "x-amz-date": amzDate, "x-amz-content-sha256": payloadHash}
	if c.SessionToken != "" {
		r.header["x-amz-security-token"] = c.SessionToken
		signed["x-amz-security-token"] = c.SessionToken
	}
	names := make([]string, 0, len(signed))
	for k := range signed {
		names = append(names, k)
	}
	sort.Strings(names)
	var canonHeaders strings.Builder
	for _, k := range names {
		canonHeaders.WriteString(k + ":" + strings.TrimSpace(signed[k]) + "\n")
	}
	signedHeaders := strings.Join(names, ";")
	canonical := strings.Join([]string{r.method, path, query, canonHeaders.String(), signedHeaders, payloadHash}, "\n")
	scope := date + "/" + s.region() + "/s3/aws4_request"
	sum := sha256.Sum256([]byte(canonical))
	toSign := "AWS4-HMAC-SHA256\n" + amzDate + "\n" + scope + "\n" + hex.EncodeToString(sum[:])
	key := hmacSHA256([]byte("AWS4"+c.SecretAccessKey), date)
	key = hmacSHA256(key, s.region())
	key = hmacSHA256(key, "s3")
	key = hmacSHA256(key, "aws4_request")
	r.header["authorization"] = "AWS4-HMAC-SHA256 Credential=" + c.AccessKeyID + "/" + scope +
		", SignedHeaders=" + signedHeaders + ", Signature=" + hex.EncodeToString(hmacSHA256(key, toSign))
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

func (s *S3) do(ctx context.Context, r *s3Request) (int, []byte, error) {
	var body io.Reader
	if r.body != nil {
		body = bytes.NewReader(r.body)
	}
	req, err := http.NewRequestWithContext(ctx, r.method, r.url, body)
	if err != nil {
		return 0, nil, err
	}
	for k, v := range r.header {
		req.Header.Set(k, v)
	}
	if r.body != nil {
		req.ContentLength = int64(len(r.body))
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	return resp.StatusCode, data, err
}

func s3Error(op, key string, status int, body []byte) error {
	msg := strings.TrimSpace(string(body))
	if len(msg) > 300 {
		msg = msg[:300] + "..."
	}
	return fmt.Errorf("httpstore: S3 %s %s: status %d: %s", op, key, status, msg)
}

func (s *S3) Get(ctx context.Context, key string, r zarr.ByteRange) ([]byte, bool, error) {
	h := map[string]string{}
	switch {
	case r.IsFull():
	case r.Offset < 0:
		h["range"] = "bytes=" + strconv.FormatInt(r.Offset, 10)
	case r.Length == 0:
		h["range"] = "bytes=" + strconv.FormatInt(r.Offset, 10) + "-"
	default:
		h["range"] = "bytes=" + strconv.FormatInt(r.Offset, 10) + "-" + strconv.FormatInt(r.Offset+r.Length-1, 10)
	}
	req, err := s.request("GET", s.fullKey(key), "", h, nil)
	if err != nil {
		return nil, false, err
	}
	status, body, err := s.do(ctx, req)
	if err != nil {
		return nil, false, err
	}
	switch status {
	case http.StatusOK:
		b, err := r.Slice(body) // the server ignored the range
		return b, true, err
	case http.StatusPartialContent:
		return body, true, nil
	case http.StatusNotFound:
		return nil, false, nil
	}
	return nil, false, s3Error("GET", key, status, body)
}

func (s *S3) Set(ctx context.Context, key string, value []byte) error {
	if value == nil {
		value = []byte{}
	}
	req, err := s.request("PUT", s.fullKey(key), "", map[string]string{"content-type": "application/octet-stream"}, value)
	if err != nil {
		return err
	}
	status, body, err := s.do(ctx, req)
	if err != nil {
		return err
	}
	if status != http.StatusOK && status != http.StatusCreated && status != http.StatusNoContent {
		return s3Error("PUT", key, status, body)
	}
	return nil
}

func (s *S3) Delete(ctx context.Context, key string) error {
	req, err := s.request("DELETE", s.fullKey(key), "", nil, nil)
	if err != nil {
		return err
	}
	status, body, err := s.do(ctx, req)
	if err != nil {
		return err
	}
	if status != http.StatusOK && status != http.StatusNoContent && status != http.StatusNotFound {
		return s3Error("DELETE", key, status, body)
	}
	return nil
}

// list runs ListObjectsV2 over all pages.
func (s *S3) list(ctx context.Context, prefix, delimiter string) (keys, prefixes []string, err error) {
	full := s.fullKey(prefix)
	if prefix == "" && full != "" {
		full += "/"
	}
	token := ""
	for {
		q := map[string]string{"list-type": "2", "prefix": full}
		if delimiter != "" {
			q["delimiter"] = delimiter
		}
		if token != "" {
			q["continuation-token"] = token
		}
		names := make([]string, 0, len(q))
		for k := range q {
			names = append(names, k)
		}
		sort.Strings(names)
		parts := make([]string, len(names))
		for i, k := range names {
			parts[i] = s3EscapeQuery(k) + "=" + s3EscapeQuery(q[k])
		}
		req, err := s.request("GET", "", strings.Join(parts, "&"), nil, nil)
		if err != nil {
			return nil, nil, err
		}
		status, body, err := s.do(ctx, req)
		if err != nil {
			return nil, nil, err
		}
		if status != http.StatusOK {
			return nil, nil, s3Error("LIST", prefix, status, body)
		}
		var res struct {
			Contents []struct {
				Key string `xml:"Key"`
			} `xml:"Contents"`
			CommonPrefixes []struct {
				Prefix string `xml:"Prefix"`
			} `xml:"CommonPrefixes"`
			IsTruncated           bool   `xml:"IsTruncated"`
			NextContinuationToken string `xml:"NextContinuationToken"`
		}
		if err := xml.Unmarshal(body, &res); err != nil {
			return nil, nil, fmt.Errorf("httpstore: S3 list response: %w", err)
		}
		root := strings.Trim(s.cfg.Prefix, "/")
		strip := func(k string) string {
			if root == "" {
				return k
			}
			return strings.TrimPrefix(k, root+"/")
		}
		for _, c := range res.Contents {
			keys = append(keys, strip(c.Key))
		}
		for _, p := range res.CommonPrefixes {
			prefixes = append(prefixes, strip(p.Prefix))
		}
		if !res.IsTruncated || res.NextContinuationToken == "" {
			break
		}
		token = res.NextContinuationToken
	}
	return keys, prefixes, nil
}

func (s *S3) List(ctx context.Context, prefix string) ([]string, error) {
	keys, _, err := s.list(ctx, prefix, "")
	var out []string
	for _, k := range keys {
		if strings.HasPrefix(k, prefix) {
			out = append(out, k)
		}
	}
	slices.Sort(out)
	return out, err
}

func (s *S3) ListDir(ctx context.Context, prefix string) ([]string, error) {
	keys, prefixes, err := s.list(ctx, prefix, "/")
	if err != nil {
		return nil, err
	}
	var out []string
	for _, k := range append(keys, prefixes...) {
		if rest, ok := strings.CutPrefix(k, prefix); ok && rest != "" {
			out = append(out, strings.TrimSuffix(rest, "/"))
		}
	}
	slices.Sort(out)
	return slices.Compact(out), nil
}
