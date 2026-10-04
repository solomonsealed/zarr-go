package httpstore

import (
	"context"
	"encoding/xml"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	zarr "github.com/solomonsealed/zarr-go"
)

// storeContract exercises a writable, listable store.
func storeContract(t *testing.T, s zarr.Store) {
	t.Helper()
	ctx := context.Background()
	ws := s.(zarr.WritableStore)
	if _, ok, err := s.Get(ctx, "missing", zarr.ByteRange{}); ok || err != nil {
		t.Fatalf("missing key: ok=%v err=%v", ok, err)
	}
	for k, v := range map[string]string{"zarr.json": "{}", "a/zarr.json": "{}", "a/c/0/1": "0123456789", "b/x": ""} {
		if err := ws.Set(ctx, k, []byte(v)); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		r    zarr.ByteRange
		want string
	}{{zarr.ByteRange{}, "0123456789"}, {zarr.ByteRange{Offset: 3, Length: 4}, "3456"}, {zarr.ByteRange{Offset: 7}, "789"}, {zarr.ByteRange{Offset: -2}, "89"}} {
		got, ok, err := s.Get(ctx, "a/c/0/1", tc.r)
		if err != nil || !ok || string(got) != tc.want {
			t.Errorf("Get(%v) = %q, %v, %v", tc.r, got, ok, err)
		}
	}
	ls := s.(zarr.ListableStore)
	keys, err := ls.List(ctx, "a/")
	if err != nil || !slices.Equal(keys, []string{"a/c/0/1", "a/zarr.json"}) {
		t.Errorf("List = %v, %v", keys, err)
	}
	names, err := ls.ListDir(ctx, "")
	if err != nil || !slices.Equal(names, []string{"a", "b", "zarr.json"}) {
		t.Errorf("ListDir root = %v, %v", names, err)
	}
	if err := ws.Delete(ctx, "a/c/0/1"); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.Get(ctx, "a/c/0/1", zarr.ByteRange{}); ok {
		t.Error("deleted key still present")
	}
}

func TestHTTPStore(t *testing.T) {
	ctx := context.Background()
	mem := zarr.NewMemoryStore()
	a, _ := zarr.CreateArray(ctx, mem, "arr", zarr.ArraySpec{Shape: []uint64{6}, ChunkShape: []uint64{2}, ShardShape: []uint64{6}, DataType: "uint8", Compressors: []zarr.CodecSpec{}}, nil)
	d, _ := zarr.FromSlice([]uint64{6}, []uint8{1, 2, 3, 4, 5, 6})
	a.Set(ctx, d)
	if _, err := zarr.ConsolidateMetadata(ctx, mem, ""); err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var ranges []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.URL.Path, "/root/")
		mu.Lock()
		ranges = append(ranges, key+" "+r.Header.Get("Range"))
		mu.Unlock()
		v, ok, _ := mem.Get(r.Context(), key, zarr.ByteRange{})
		if !ok {
			http.NotFound(w, r)
			return
		}
		http.ServeContent(w, r, key, time.Time{}, strings.NewReader(string(v)))
	}))
	defer srv.Close()
	hs, err := New(srv.URL+"/root", nil)
	if err != nil {
		t.Fatal(err)
	}
	g, err := zarr.OpenGroup(ctx, hs, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !g.Consolidated() {
		t.Fatal("consolidated metadata not used")
	}
	arr, err := g.OpenArray(ctx, "arr")
	if err != nil {
		t.Fatal(err)
	}
	ranges = nil
	got, err := arr.Get(ctx, zarr.Index(3))
	if err != nil || got.Data[0] != 4 {
		t.Fatalf("got %v, %v", got, err)
	}
	// One suffix read for the shard index, one ranged read for the chunk.
	if len(ranges) != 2 || !strings.HasSuffix(ranges[0], "bytes=-52") || !strings.HasPrefix(ranges[1], "arr/c/0 bytes=") {
		t.Errorf("requests = %q", ranges)
	}
}

// fakeS3 is a minimal S3 API: GET (with Range), PUT, DELETE, ListObjectsV2.
type fakeS3 struct {
	mu   sync.Mutex
	objs map[string][]byte
}

func (f *fakeS3) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	path := strings.TrimPrefix(r.URL.Path, "/bucket/")
	if r.URL.Path == "/bucket/" && r.URL.Query().Get("list-type") == "2" {
		prefix, delim := r.URL.Query().Get("prefix"), r.URL.Query().Get("delimiter")
		type content struct{ Key string }
		type cp struct{ Prefix string }
		var res struct {
			XMLName        xml.Name  `xml:"ListBucketResult"`
			Contents       []content `xml:"Contents"`
			CommonPrefixes []cp      `xml:"CommonPrefixes"`
		}
		var keys []string
		for k := range f.objs {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		seen := map[string]bool{}
		for _, k := range keys {
			rest, ok := strings.CutPrefix(k, prefix)
			if !ok {
				continue
			}
			if delim != "" {
				if i := strings.Index(rest, delim); i >= 0 {
					p := prefix + rest[:i+1]
					if !seen[p] {
						seen[p] = true
						res.CommonPrefixes = append(res.CommonPrefixes, cp{p})
					}
					continue
				}
			}
			res.Contents = append(res.Contents, content{k})
		}
		xml.NewEncoder(w).Encode(res)
		return
	}
	if r.Header.Get("authorization") == "" {
		http.Error(w, "unsigned", http.StatusForbidden)
		return
	}
	switch r.Method {
	case "GET":
		v, ok := f.objs[path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if rg := r.Header.Get("range"); strings.HasPrefix(rg, "bytes=-") {
			n, _ := strconv.Atoi(rg[len("bytes=-"):])
			w.WriteHeader(http.StatusPartialContent)
			w.Write(v[len(v)-n:])
			return
		}
		http.ServeContent(w, r, path, time.Time{}, strings.NewReader(string(v)))
	case "PUT":
		b, _ := io.ReadAll(r.Body)
		f.objs[path] = b
	case "DELETE":
		delete(f.objs, path)
		w.WriteHeader(http.StatusNoContent)
	}
}

func TestS3Store(t *testing.T) {
	srv := httptest.NewServer(&fakeS3{objs: map[string][]byte{}})
	defer srv.Close()
	s := NewS3(S3Config{Bucket: "bucket", Prefix: "data/zarr", Endpoint: srv.URL, AccessKeyID: "AK", SecretAccessKey: "SK"}, nil)
	storeContract(t, s)
}

// The AWS documentation's SigV4 example (GET Bucket, List Objects).
func TestS3Signature(t *testing.T) {
	s := NewS3(S3Config{Bucket: "examplebucket", AccessKeyID: "AKIAIOSFODNN7EXAMPLE", SecretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}, nil)
	r := &s3Request{method: "GET", header: map[string]string{}}
	s.sign(r, "examplebucket.s3.amazonaws.com", "/", "max-keys=2&prefix=J", emptySHA256, time.Date(2013, 5, 24, 0, 0, 0, 0, time.UTC))
	want := "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, " +
		"SignedHeaders=host;x-amz-content-sha256;x-amz-date, " +
		"Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"
	if got := r.header["authorization"]; got != want {
		t.Fatalf("authorization =\n%s\nwant\n%s", got, want)
	}
	for _, tc := range []struct {
		cfg  S3Config
		want string
	}{
		{S3Config{Bucket: "b", Region: "eu-west-1"}, "https://b.s3.eu-west-1.amazonaws.com/p/k%20x"},
		{S3Config{Bucket: "b", ForcePathStyle: true}, "https://s3.us-east-1.amazonaws.com/b/p/k%20x"},
		{S3Config{Bucket: "b", Endpoint: "https://acct.r2.cloudflarestorage.com"}, "https://acct.r2.cloudflarestorage.com/b/p/k%20x"},
	} {
		base, _, path, _ := NewS3(tc.cfg, nil).target("p/k x")
		if base+path != tc.want {
			t.Errorf("%+v: %s, want %s", tc.cfg, base+path, tc.want)
		}
	}
}
