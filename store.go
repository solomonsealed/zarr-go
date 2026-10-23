package zarr

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
)

// ByteRange selects part of a stored value. The zero value selects all of it.
type ByteRange struct {
	// Offset is the first byte. A negative Offset selects the last
	// -Offset bytes (a suffix), and Length is then ignored.
	Offset int64
	// Length is the number of bytes from Offset; 0 means to the end.
	Length int64
}

// IsFull reports whether r selects the whole value.
func (r ByteRange) IsFull() bool { return r.Offset == 0 && r.Length == 0 }

func (r ByteRange) String() string {
	switch {
	case r.IsFull():
		return "all"
	case r.Offset < 0:
		return fmt.Sprintf("last %d bytes", -r.Offset)
	case r.Length == 0:
		return fmt.Sprintf("bytes %d-", r.Offset)
	}
	return fmt.Sprintf("bytes %d-%d", r.Offset, r.Offset+r.Length-1)
}

// resolve returns [start, end) of the range in a value of the given size.
func (r ByteRange) resolve(size int64) (int64, int64, error) {
	b, err := r.bounds(size)
	return b[0], b[1], err
}

func (r ByteRange) bounds(size int64) ([2]int64, error) {
	if r.Offset < 0 {
		if -r.Offset > size {
			return [2]int64{}, fmt.Errorf("zarr: suffix of %d bytes from a value of %d bytes", -r.Offset, size)
		}
		return [2]int64{size + r.Offset, size}, nil
	}
	end := size
	if r.Length > 0 {
		end = r.Offset + r.Length
	}
	if r.Offset > size || end > size || end < r.Offset {
		return [2]int64{}, fmt.Errorf("zarr: %v outside a value of %d bytes", r, size)
	}
	return [2]int64{r.Offset, end}, nil
}

// Slice returns the bytes of r in data, or an error if out of range.
func (r ByteRange) Slice(data []byte) ([]byte, error) {
	s, e, err := r.resolve(int64(len(data)))
	if err != nil {
		return nil, err
	}
	return data[s:e], nil
}

// Store holds a Zarr hierarchy as keys and values. Keys are "/"-separated
// paths without a leading slash ("zarr.json", "a/b/c/0/1"). Stores must be
// safe for concurrent use; arrays never modify the bytes Get returns.
type Store interface {
	// Get returns the bytes of key in r. ok is false if the key does not
	// exist.
	Get(ctx context.Context, key string, r ByteRange) (data []byte, ok bool, err error)
}

// WritableStore is a Store that can be modified.
type WritableStore interface {
	Store
	Set(ctx context.Context, key string, value []byte) error
	// Delete removes a key; deleting a missing key is not an error.
	Delete(ctx context.Context, key string) error
}

// ListableStore is a Store that can enumerate its keys.
type ListableStore interface {
	Store
	// List returns all keys starting with prefix, sorted.
	List(ctx context.Context, prefix string) ([]string, error)
	// ListDir returns the names directly below prefix (a key prefix
	// ending in "/", or "" for the root): keys and "directories", sorted.
	ListDir(ctx context.Context, prefix string) ([]string, error)
}

// ErrReadOnly is returned when writing through a read-only store or array.
var ErrReadOnly = errors.New("zarr: read-only")

// joinKey joins path components, ignoring empty ones.
func joinKey(parts ...string) string {
	var out []string
	for _, p := range parts {
		p = strings.Trim(p, "/")
		if p != "" {
			out = append(out, p)
		}
	}
	return strings.Join(out, "/")
}

// normalizePath cleans a node path: no leading or trailing slashes.
func normalizePath(p string) (string, error) {
	p = strings.Trim(p, "/")
	if p == "" {
		return "", nil
	}
	for _, part := range strings.Split(p, "/") {
		if part == "" || part == "." || part == ".." {
			return "", fmt.Errorf("zarr: invalid node path %q", p)
		}
	}
	return p, nil
}

// listDirFromList implements ListDir on top of List.
func listDirFromList(keys []string, prefix string) []string {
	seen := map[string]bool{}
	var out []string
	for _, k := range keys {
		rest, ok := strings.CutPrefix(k, prefix)
		if !ok || rest == "" {
			continue
		}
		name, _, _ := strings.Cut(rest, "/")
		if !seen[name] {
			seen[name] = true
			out = append(out, name)
		}
	}
	slices.Sort(out)
	return out
}

// ---------------------------------------------------------------------------
// MemoryStore

// MemoryStore is an in-memory store, safe for concurrent use.
type MemoryStore struct {
	mu   sync.RWMutex
	data map[string][]byte
}

// NewMemoryStore returns an empty in-memory store.
func NewMemoryStore() *MemoryStore { return &MemoryStore{data: map[string][]byte{}} }

func (m *MemoryStore) Get(_ context.Context, key string, r ByteRange) ([]byte, bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	v, ok := m.data[key]
	if !ok {
		return nil, false, nil
	}
	b, err := r.Slice(v)
	if err != nil {
		return nil, true, err
	}
	return slices.Clone(b), true, nil
}

func (m *MemoryStore) Set(_ context.Context, key string, value []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[key] = slices.Clone(value)
	return nil
}

func (m *MemoryStore) Delete(_ context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.data, key)
	return nil
}

func (m *MemoryStore) List(_ context.Context, prefix string) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []string
	for k := range m.data {
		if strings.HasPrefix(k, prefix) {
			out = append(out, k)
		}
	}
	slices.Sort(out)
	return out, nil
}

func (m *MemoryStore) ListDir(ctx context.Context, prefix string) ([]string, error) {
	keys, err := m.List(ctx, prefix)
	if err != nil {
		return nil, err
	}
	return listDirFromList(keys, prefix), nil
}

// ---------------------------------------------------------------------------
// Chunks from a store

// ChunkSource supplies the encoded bytes of an array's chunks (for sharded
// arrays: of its shards), addressed by chunk grid coordinates. Arrays call
// it from several goroutines at once, so it must be safe for concurrent
// use. Arrays never modify the bytes it returns.
type ChunkSource interface {
	// GetChunk returns the bytes in r of the chunk at coords. ok is false
	// if the chunk was never written, in which case readers use the fill
	// value.
	GetChunk(ctx context.Context, coords []uint64, r ByteRange) (data []byte, ok bool, err error)
}

// ChunkWriter is a ChunkSource that also stores chunks; writing an array
// needs one.
type ChunkWriter interface {
	ChunkSource
	// SetChunk stores the encoded bytes of a chunk.
	SetChunk(ctx context.Context, coords []uint64, data []byte) error
	// DeleteChunk removes a chunk, so readers see the fill value.
	DeleteChunk(ctx context.Context, coords []uint64) error
}

// StoreChunks returns a ChunkSource (a ChunkWriter if store is writable)
// for the array at path whose chunk keys follow enc.
func StoreChunks(store Store, path string, enc ChunkKeyEncoding) ChunkSource {
	sc := storeChunks{store: store, path: path, enc: enc}
	if _, ok := store.(WritableStore); ok {
		return writableStoreChunks{sc}
	}
	return sc
}

type storeChunks struct {
	store Store
	path  string
	enc   ChunkKeyEncoding
}

func (s storeChunks) key(coords []uint64) (string, error) {
	k := s.enc.Key(coords)
	if k == "" {
		return "", fmt.Errorf("zarr: chunk key encoding %q has no key for chunk %v", s.enc.Name, coords)
	}
	return joinKey(s.path, k), nil
}

func (s storeChunks) GetChunk(ctx context.Context, coords []uint64, r ByteRange) ([]byte, bool, error) {
	k, err := s.key(coords)
	if err != nil {
		return nil, false, err
	}
	return s.store.Get(ctx, k, r)
}

type writableStoreChunks struct{ storeChunks }

func (s writableStoreChunks) SetChunk(ctx context.Context, coords []uint64, data []byte) error {
	k, err := s.key(coords)
	if err != nil {
		return err
	}
	return s.store.(WritableStore).Set(ctx, k, data)
}

func (s writableStoreChunks) DeleteChunk(ctx context.Context, coords []uint64) error {
	k, err := s.key(coords)
	if err != nil {
		return err
	}
	return s.store.(WritableStore).Delete(ctx, k)
}
