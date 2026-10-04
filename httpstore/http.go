// Package httpstore provides Zarr stores over HTTP: plain HTTP(S) with
// Range requests (read-only), and S3-compatible object stores (AWS S3,
// Cloudflare R2, MinIO, GCS interoperability, ...) with SigV4 signing.
//
// They live outside the zarr package so that programs that do not need
// them, such as WebAssembly builds using other HTTP clients, do not link
// net/http.
package httpstore

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"

	zarr "github.com/solomonsealed/zarr-go"
)

var _ zarr.Store = (*HTTP)(nil)

// HTTP reads a Zarr hierarchy over HTTP(S), with Range requests for
// partial reads. It cannot list keys, so groups need consolidated
// metadata to enumerate their members.
type HTTP struct {
	base   string
	client *http.Client
	// Header is added to every request (e.g. Authorization).
	Header http.Header

	mu sync.Mutex
}

// New returns a store reading keys below baseURL. A nil client means
// http.DefaultClient.
func New(baseURL string, client *http.Client) (*HTTP, error) {
	u, err := url.Parse(baseURL)
	if err != nil {
		return nil, err
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("httpstore: needs an http(s) URL, not %q", baseURL)
	}
	if client == nil {
		client = http.DefaultClient
	}
	return &HTTP{base: strings.TrimSuffix(baseURL, "/"), client: client, Header: http.Header{}}, nil
}

func (s *HTTP) url(key string) string {
	parts := strings.Split(key, "/")
	for i, p := range parts {
		parts[i] = url.PathEscape(p)
	}
	return s.base + "/" + strings.Join(parts, "/")
}

func (s *HTTP) Get(ctx context.Context, key string, r zarr.ByteRange) ([]byte, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, s.url(key), nil)
	if err != nil {
		return nil, false, err
	}
	for k, v := range s.Header {
		req.Header[k] = v
	}
	switch {
	case r.IsFull():
	case r.Offset < 0:
		req.Header.Set("Range", fmt.Sprintf("bytes=%d", r.Offset))
	case r.Length == 0:
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", r.Offset))
	default:
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-%d", r.Offset, r.Offset+r.Length-1))
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, false, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusNotFound, http.StatusGone:
		io.Copy(io.Discard, resp.Body)
		return nil, false, nil
	case http.StatusOK:
		data, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, true, err
		}
		// The server ignored the range.
		b, err := r.Slice(data)
		return b, true, err
	case http.StatusPartialContent:
		data, err := io.ReadAll(resp.Body)
		return data, true, err
	}
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
	return nil, false, &HTTPError{URL: req.URL.String(), Status: resp.StatusCode, Body: string(bytes.TrimSpace(body))}
}

// HTTPError is an unexpected HTTP response.
type HTTPError struct {
	URL    string
	Status int
	Body   string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("httpstore: GET %s: HTTP %d %s", e.URL, e.Status, e.Body)
}
