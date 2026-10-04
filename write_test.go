package zarr

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
)

// counter is a bytes codec that counts encodes and decodes, to check how
// much work partial writes and reads do.
type counter struct{ enc, dec *atomic.Int64 }

func (c counter) Encode(in []byte) ([]byte, error) { c.enc.Add(1); return in, nil }
func (c counter) Decode(in []byte) ([]byte, error) { c.dec.Add(1); return in, nil }

var encodes, decodes atomic.Int64

func init() {
	RegisterCodec("test.counter", func(json.RawMessage, ChunkSpec) (any, error) {
		return counter{&encodes, &decodes}, nil
	})
}

// recordingStore counts reads and the bytes they return.
type recordingStore struct {
	*MemoryStore
	mu     sync.Mutex
	gets   int
	ranged int
	bytes  int
}

func (s *recordingStore) Get(ctx context.Context, key string, r ByteRange) ([]byte, bool, error) {
	b, ok, err := s.MemoryStore.Get(ctx, key, r)
	s.mu.Lock()
	s.gets++
	if !r.IsFull() {
		s.ranged++
	}
	s.bytes += len(b)
	s.mu.Unlock()
	return b, ok, err
}

func (s *recordingStore) reset() { s.mu.Lock(); s.gets, s.ranged, s.bytes = 0, 0, 0; s.mu.Unlock() }

// A small write to a sharded array decodes and re-encodes only the inner
// chunks it touches; reads fetch only the inner chunks they need.
func TestPartialShardIO(t *testing.T) {
	ctx := context.Background()
	st := &recordingStore{MemoryStore: NewMemoryStore()}
	a, err := CreateArray(ctx, st, "a", ArraySpec{Shape: []uint64{64, 64}, ChunkShape: []uint64{8, 8}, ShardShape: []uint64{64, 64},
		DataType: "float32", Compressors: []CodecSpec{CodecJSON("test.counter", "")}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	vals := make([]float32, 64*64)
	for i := range vals {
		vals[i] = float32(i)
	}
	data, _ := FromSlice([]uint64{64, 64}, vals)
	if err := a.Set(ctx, data); err != nil {
		t.Fatal(err)
	}
	if e, d := encodes.Swap(0), decodes.Swap(0); e != 64 || d != 0 {
		t.Fatalf("full write: %d encodes, %d decodes; want 64, 0", e, d)
	}
	before, _, _ := st.MemoryStore.Get(ctx, "a/c/0/0", ByteRange{})

	one, _ := FromSlice([]uint64{}, []float32{-1})
	if err := a.Set(ctx, one, Index(10), Index(20)); err != nil {
		t.Fatal(err)
	}
	if e, d := encodes.Swap(0), decodes.Swap(0); e != 1 || d != 1 {
		t.Fatalf("one-element write: %d encodes, %d decodes; want 1, 1", e, d)
	}
	// Untouched inner chunks keep their bytes (in Morton order, moved).
	after, _, _ := st.MemoryStore.Get(ctx, "a/c/0/0", ByteRange{})
	if len(after) != len(before) || bytes.Equal(after, before) {
		t.Fatalf("shard sizes %d -> %d", len(before), len(after))
	}

	st.reset()
	got, err := a.Get(ctx, Slice(7, 9), Slice(15, 17))
	if err != nil {
		t.Fatal(err)
	}
	if v, _ := got.Values(); !slices.Equal(v.([]float32), []float32{463, 464, 527, 528}) {
		t.Fatalf("read back %v", v)
	}
	// The index, then the 4 inner chunks the region overlaps in one
	// coalesced request.
	if st.ranged != 2 || decodes.Swap(0) != 4 {
		t.Errorf("partial read: %d ranged reads of %d", st.ranged, st.gets)
	}
	// A whole-shard write with all fill deletes the shard.
	zeros, _ := FromSlice([]uint64{64, 64}, make([]float32, 64*64))
	if err := a.Set(ctx, zeros); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := st.MemoryStore.Get(ctx, "a/c/0/0", ByteRange{}); ok {
		t.Error("all-fill shard was not deleted")
	}
}

func TestWriteDeletesFillChunks(t *testing.T) {
	ctx := context.Background()
	st := NewMemoryStore()
	arr, err := CreateArray(ctx, st, "a", ArraySpec{Shape: []uint64{4}, ChunkShape: []uint64{2}, DataType: "float64", FillValue: math.NaN()}, nil)
	if err != nil {
		t.Fatal(err)
	}
	nan, _ := FromSlice([]uint64{2}, []float64{math.NaN(), math.NaN()})
	vals, _ := FromSlice([]uint64{4}, []float64{1, 2, 3, 4})
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	chunks := func() []string { k, _ := st.List(ctx, "a/c/"); return k }
	must(arr.Write(ctx, []uint64{0}, nan))
	if len(chunks()) != 0 {
		t.Fatalf("chunks %v", chunks())
	}
	must(arr.Write(ctx, []uint64{0}, vals))
	if len(chunks()) != 2 {
		t.Fatalf("chunks %v", chunks())
	}
	must(arr.Write(ctx, []uint64{2}, nan))
	if !slices.Equal(chunks(), []string{"a/c/0"}) {
		t.Errorf("all-fill chunk was not deleted: %v", chunks())
	}
	// With WriteEmptyChunks, fill chunks are stored.
	arr2, _ := OpenArray(ctx, st, "a", &Options{WriteEmptyChunks: true})
	must(arr2.Write(ctx, []uint64{2}, nan))
	if len(chunks()) != 2 {
		t.Errorf("WriteEmptyChunks: chunks %v", chunks())
	}
}

// Every selection kind reads what a reference implementation computes
// element by element, and writes go where reads find them.
func TestSelectionsAgainstReference(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(9))
	for _, spec := range []ArraySpec{
		{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, DataType: "int32"},
		{Shape: []uint64{9, 7}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "int32", Compressors: []CodecSpec{}},
		{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, ChunkEdges: [][]uint64{{1, 3, 5}, nil}, DataType: "int32"},
		{ZarrFormat: 2, Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, DataType: "<i4", Order: "F"},
	} {
		a, err := CreateArray(ctx, NewMemoryStore(), "a", spec, nil)
		if err != nil {
			t.Fatal(err)
		}
		ref := make([]int32, 63)
		for i := range ref {
			ref[i] = int32(i)
		}
		d, _ := FromSlice([]uint64{9, 7}, ref)
		if err := a.Set(ctx, d); err != nil {
			t.Fatal(err)
		}
		at := func(i, j int64) int32 { return ref[i*7+j] }
		for range 200 {
			r0, r1 := randDim(rng, 9), randDim(rng, 7)
			got, err := a.Get(ctx, r0.sel, r1.sel)
			if err != nil {
				t.Fatalf("%v %v: %v", r0.sel, r1.sel, err)
			}
			var want []int32
			for _, i := range r0.idx {
				for _, j := range r1.idx {
					want = append(want, at(i, j))
				}
			}
			v, _ := got.Values()
			if !slices.Equal(v.([]int32), want) {
				t.Fatalf("Get(%v, %v) = %v, want %v", r0.sel, r1.sel, v, want)
			}
		}
		pts, _ := a.GetCoordinates(ctx, []int64{8, 0, -1, 3}, []int64{6, 0, 0, -2})
		if v, _ := pts.Values(); !slices.Equal(v.([]int32), []int32{62, 0, 56, 26}) {
			t.Errorf("points = %v", v)
		}
		// Write through a mask and read it back with the same mask.
		mask := make([]bool, 63)
		for i := range mask {
			mask[i] = i%5 == 0
		}
		m, _ := FromSlice([]uint64{9, 7}, mask)
		neg := make([]int32, 13)
		for i := range neg {
			neg[i] = -int32(i)
		}
		nd, _ := FromSlice([]uint64{13}, neg)
		if err := a.SetMask(ctx, nd, m); err != nil {
			t.Fatal(err)
		}
		back, _ := a.GetMask(ctx, m)
		if v, _ := back.Values(); !slices.Equal(v.([]int32), neg) {
			t.Errorf("mask round trip = %v", v)
		}
	}
}

type dimRef struct {
	sel DimSelector
	idx []int64
}

func randDim(rng *rand.Rand, n int64) dimRef {
	switch rng.Intn(4) {
	case 0:
		i := rng.Int63n(n)
		return dimRef{Index(i - n*int64(rng.Intn(2))), []int64{i}}
	case 1:
		start, stop, step := rng.Int63n(n+1), rng.Int63n(n+1), 1+rng.Int63n(3)
		var idx []int64
		for i := start; i < stop; i += step {
			idx = append(idx, i)
		}
		return dimRef{SliceStep(start, stop, step), idx}
	case 2:
		var idx []int64
		for range rng.Intn(int(n) + 2) {
			idx = append(idx, rng.Int63n(n))
		}
		return dimRef{IndicesSelector(idx), idx}
	}
	m := make([]bool, n)
	var idx []int64
	for i := range m {
		if m[i] = rng.Intn(2) == 0; m[i] {
			idx = append(idx, int64(i))
		}
	}
	return dimRef{MaskSelector(m), idx}
}

func TestResizeAndAppend(t *testing.T) {
	ctx := context.Background()
	st := NewMemoryStore()
	a, _ := CreateArray(ctx, st, "a", ArraySpec{Shape: []uint64{4, 4}, ChunkShape: []uint64{2, 2}, DataType: "uint8"}, nil)
	d, _ := FromSlice([]uint64{4, 4}, slices.Repeat([]uint8{1}, 16))
	a.Set(ctx, d)
	if err := a.Resize(ctx, []uint64{3, 1}); err != nil {
		t.Fatal(err)
	}
	keys, _ := st.List(ctx, "a/c/")
	if !slices.Equal(keys, []string{"a/c/0/0", "a/c/1/0"}) {
		t.Errorf("after shrinking: %v", keys)
	}
	reopened, _ := OpenArray(ctx, st, "a", nil)
	if !slices.Equal(reopened.Shape(), []uint64{3, 1}) {
		t.Errorf("shape %v", reopened.Shape())
	}
	row, _ := FromSlice([]uint64{2, 1}, []uint8{7, 8})
	if err := a.Append(ctx, row, 0); err != nil {
		t.Fatal(err)
	}
	all, _ := a.ReadAll(ctx)
	if !slices.Equal(all.Data, []uint8{1, 1, 1, 7, 8}) || !slices.Equal(all.Shape, []uint64{5, 1}) {
		t.Errorf("after append: %v %v", all.Shape, all.Data)
	}
}

func TestGroupsAndConsolidation(t *testing.T) {
	ctx := context.Background()
	for _, format := range []int{2, 3} {
		st := NewMemoryStore()
		root, err := CreateGroup(ctx, st, "", GroupSpec{ZarrFormat: format, Attributes: map[string]any{"title": "t"}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := root.CreateArray(ctx, "x/y", ArraySpec{Shape: []uint64{3}, DataType: "int8"}, nil); err != nil {
			t.Fatal(err)
		}
		if _, err := root.CreateArray(ctx, "x/y", ArraySpec{Shape: []uint64{3}, DataType: "int8"}, nil); err == nil {
			t.Error("created over an existing array")
		}
		var walked []string
		root.Walk(ctx, func(rel string, n Node) error { walked = append(walked, fmt.Sprintf("%s:%T", rel, n)); return nil })
		if !slices.Equal(walked, []string{"x:*zarr.Group", "x/y:*zarr.Array"}) {
			t.Errorf("v%d walk = %v", format, walked)
		}
		if _, err := ConsolidateMetadata(ctx, st, ""); err != nil {
			t.Fatal(err)
		}
		g, _ := OpenGroup(ctx, st, "", nil)
		if !g.Consolidated() {
			t.Fatalf("v%d: not consolidated", format)
		}
		arr, err := g.OpenArray(ctx, "x/y")
		if err != nil || arr.ZarrFormat() != format {
			t.Errorf("v%d: consolidated lookup: %v", format, err)
		}
		if err := g.SetAttributes(ctx, map[string]any{"title": "new"}); err != nil {
			t.Fatal(err)
		}
		g2, _ := OpenGroup(ctx, st, "", nil)
		if g2.Attributes()["title"] != "new" {
			t.Errorf("v%d: attributes %v", format, g2.Attributes())
		}
	}
}

// NewArray serves arrays from any ChunkSource, as Icechunk does. Arrays
// use their source from several goroutines at once.
type mapChunks struct {
	mu sync.Mutex
	m  map[string][]byte
}

func (s *mapChunks) GetChunk(_ context.Context, c []uint64, r ByteRange) ([]byte, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.m[fmt.Sprint(c)]
	if !ok {
		return nil, false, nil
	}
	b, err := r.Slice(v)
	return b, true, err
}
func (s *mapChunks) SetChunk(_ context.Context, c []uint64, d []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.m == nil {
		s.m = map[string][]byte{}
	}
	s.m[fmt.Sprint(c)] = d
	return nil
}
func (s *mapChunks) DeleteChunk(_ context.Context, c []uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.m, fmt.Sprint(c))
	return nil
}

func TestNewArrayWithChunkSource(t *testing.T) {
	ctx := context.Background()
	doc, err := ArraySpec{Shape: []uint64{6}, ChunkShape: []uint64{2}, ShardShape: []uint64{4}, DataType: "int16"}.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	src := &mapChunks{}
	a, err := NewArray(doc, src, nil)
	if err != nil {
		t.Fatal(err)
	}
	d, _ := FromSlice([]uint64{6}, []int16{1, 2, 3, 4, 5, 6})
	if err := a.Set(ctx, d); err != nil {
		t.Fatal(err)
	}
	if len(src.m) != 2 {
		t.Errorf("%d shards", len(src.m))
	}
	got, _ := a.Get(ctx, SliceStep(1, 6, 2))
	if v, _ := got.Values(); !slices.Equal(v.([]int16), []int16{2, 4, 6}) {
		t.Errorf("values %v", v)
	}
}

// frozenChunks returns the same buffer on every read and fails the test
// if anything modifies it (as a caching ChunkSource would see).
type frozenChunks struct {
	t    *testing.T
	data map[string][]byte
	orig map[string][]byte
	mu   sync.Mutex
}

func (f *frozenChunks) GetChunk(_ context.Context, c []uint64, r ByteRange) ([]byte, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	v, ok := f.data[fmt.Sprint(c)]
	if !ok {
		return nil, false, nil
	}
	b, err := r.Slice(v)
	return b, true, err
}

func (f *frozenChunks) SetChunk(_ context.Context, c []uint64, d []byte) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.data[fmt.Sprint(c)], f.orig[fmt.Sprint(c)] = d, bytes.Clone(d)
	return nil
}

func (f *frozenChunks) DeleteChunk(_ context.Context, c []uint64) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.data, fmt.Sprint(c))
	delete(f.orig, fmt.Sprint(c))
	return nil
}

func (f *frozenChunks) check() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for k, v := range f.data {
		if !bytes.Equal(v, f.orig[k]) {
			f.t.Fatalf("chunk %s was modified in place", k)
		}
	}
}

// Arrays never modify the bytes a ChunkSource returns, even when chunks
// are uncompressed and writes are partial.
func TestSourceBytesAreNotModified(t *testing.T) {
	ctx := context.Background()
	for _, spec := range []ArraySpec{
		{Shape: []uint64{8}, ChunkShape: []uint64{4}, DataType: "int32", Compressors: []CodecSpec{}},
		{Shape: []uint64{8}, ChunkShape: []uint64{2}, ShardShape: []uint64{8}, DataType: "int32", Compressors: []CodecSpec{}},
		{Shape: []uint64{8}, ChunkShape: []uint64{4}, DataType: "int32", Compressors: []CodecSpec{Crc32c()}},
	} {
		doc, err := spec.Metadata()
		if err != nil {
			t.Fatal(err)
		}
		src := &frozenChunks{t: t, data: map[string][]byte{}, orig: map[string][]byte{}}
		a, err := NewArray(doc, src, nil)
		if err != nil {
			t.Fatal(err)
		}
		d, _ := FromSlice([]uint64{8}, []int32{1, 2, 3, 4, 5, 6, 7, 8})
		if err := a.Set(ctx, d); err != nil {
			t.Fatal(err)
		}
		one, _ := FromSlice([]uint64{}, []int32{-1})
		if err := a.Set(ctx, one, Index(1)); err != nil {
			t.Fatal(err)
		}
		c, _ := a.ReadChunk(ctx, []uint64{0})
		c.Data[0] = 99
		src.check()
		if shape, _ := a.SelectionShape(Index(1), Ellipsis); len(shape) != 0 {
			t.Errorf("SelectionShape = %v", shape)
		}
	}
}
