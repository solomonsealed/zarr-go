package zarr

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

// allocated returns the bytes f allocates.
func allocated(f func()) uint64 {
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	f()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

func rectilinearDoc(shape, chunkShapes string) []byte {
	return []byte(`{"zarr_format":3,"node_type":"array","shape":` + shape + `,"data_type":"int8",
		"chunk_grid":{"name":"rectilinear","configuration":{"kind":"inline","chunk_shapes":` + chunkShapes + `}},
		"chunk_key_encoding":{"name":"default"},"fill_value":0,"codecs":[{"name":"bytes"}]}`)
}

// Rectilinear chunk edges must cover the array (as zarr-python requires);
// reading past them used to loop forever.
func TestRectilinearEdgesMustCoverShape(t *testing.T) {
	if _, err := NewArray(rectilinearDoc("[10]", "[[3, 3]]"), &mapChunks{}, nil); err == nil {
		t.Error("edges summing to 6 accepted for a dimension of 10")
	}
	// Edges past the end (left by shrinking) are fine.
	a, err := NewArray(rectilinearDoc("[10]", "[[3, 3, [2, 3]]]"), &mapChunks{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Get(context.Background(), SliceStep(0, 10, 4)); err != nil {
		t.Error(err)
	}
	// Run lengths can declare more edges than memory holds, or overflow.
	for _, edges := range []string{"[[[1, 1099511627776]]]", "[[9223372036854775808, 9223372036854775808]]"} {
		var err error
		if n := allocated(func() { _, err = NewArray(rectilinearDoc("[10]", edges), &mapChunks{}, nil) }); err == nil || n > 1<<20 {
			t.Errorf("edges %s: %v, %d bytes allocated", edges, err, n)
		}
	}
}

// Shapes whose sizes overflow are refused when the array is opened,
// instead of wrapping around to small buffers.
func TestOversizedShapesAreRefused(t *testing.T) {
	doc := func(shape, chunks, codecs string) []byte {
		return []byte(`{"zarr_format":3,"node_type":"array","shape":` + shape + `,"data_type":"int8",
			"chunk_grid":{"name":"regular","configuration":{"chunk_shape":` + chunks + `}},
			"chunk_key_encoding":{"name":"default"},"fill_value":0,"codecs":` + codecs + `}`)
	}
	bytesCodec := `[{"name":"bytes"}]`
	for _, d := range [][]byte{
		doc("[8589934592, 2147483648]", "[8589934592, 2147483648]", bytesCodec), // 2^64 elements
		doc("[18446744073709551615]", "[1]", bytesCodec),
		doc("[1152921504606846976]", "[1152921504606846976]", `[{"name":"sharding_indexed","configuration":{"chunk_shape":[1],
			"codecs":[{"name":"bytes"}],"index_codecs":[{"name":"bytes","configuration":{"endian":"little"}}]}}]`), // a 2^64-byte index
		[]byte(`{"zarr_format":2,"shape":[18446744073709551615],"chunks":[1],"dtype":"|i1","fill_value":0,
			"order":"C","filters":null,"compressor":null}`),
	} {
		if _, err := NewArray(d, &mapChunks{}, nil); err == nil {
			t.Errorf("accepted %s", d)
		}
	}
	// A wrapped-around shape no longer validates.
	n := &NDArray{Shape: []uint64{1 << 33, 1 << 31}, DataType: MustDataType("int8")}
	if _, err := n.Values(); err == nil {
		t.Error("Values of an array of 2^64 elements without data succeeded")
	}
}

// Selections too large to hold are an error, not a hang.
func TestHugeSelectionIsAnError(t *testing.T) {
	doc, err := ArraySpec{Shape: []uint64{1 << 40, 1 << 40}, ChunkShape: []uint64{1 << 10, 1 << 10}, DataType: "int8"}.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	a, err := NewArray(doc, &mapChunks{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Get(context.Background()); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Errorf("Get of 2^80 elements: %v", err)
	}
	one, _ := FromSlice([]uint64{}, []int8{1})
	if err := a.Set(context.Background(), one); err == nil || !strings.Contains(err.Error(), "too large") {
		t.Errorf("Set of 2^80 elements: %v", err)
	}
}

// Size headers in chunks are checked against the bytes that follow them
// before anything that size is allocated.
func TestChunkSizeHeadersAreBounded(t *testing.T) {
	// A 17-byte zstd frame declaring 15 GiB of content.
	frame := []byte{0x28, 0xb5, 0x2f, 0xfd, 0xc0, 0x00}
	frame = binary.LittleEndian.AppendUint64(frame, 15<<30)
	frame = append(frame, 0x01, 0x00, 0x00)
	// The same frame after a valid one.
	two := append(zstdFrame(t, []byte("abc")), frame...)
	// numcodecs' LZ4 header declaring 2 GiB.
	lz4 := []byte{0, 0, 0, 0x80, 0x00}
	for name, f := range map[string]func() error{
		"zstd":         func() error { _, err := unzstd(frame); return err },
		"zstd, second": func() error { _, err := unzstd(two); return err },
		"lz4":          func() error { _, err := unlz4Numcodecs(lz4); return err },
	} {
		var err error
		if n := allocated(func() { err = f() }); err == nil || n > 1<<20 {
			t.Errorf("%s: %v, %d MiB allocated", name, err, n>>20)
		}
	}
	// A vlen chunk declaring 2^24 strings in 8 bytes.
	doc, err := ArraySpec{Shape: []uint64{1 << 24}, DataType: "string", Compressors: []CodecSpec{}}.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	a, err := NewArray(doc, &mapChunks{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	vlen := binary.LittleEndian.AppendUint32(nil, 1<<24)
	vlen = binary.LittleEndian.AppendUint32(vlen, 0)
	_, spec := a.chunkSpec([]uint64{0})
	if n := allocated(func() { _, err = a.pipe.decode(context.Background(), vlen, spec) }); err == nil || n > 1<<20 {
		t.Errorf("vlen: %v, %d MiB allocated", err, n>>20)
	}
	// Highly compressible data still decodes.
	zeros := make([]byte, 64<<20)
	z := zstdFrame(t, zeros)
	if out, err := unzstd(z); err != nil || len(out) != len(zeros) {
		t.Errorf("%d bytes of zstd: %d bytes, %v", len(z), len(out), err)
	}
}

func zstdFrame(t *testing.T, b []byte) []byte {
	t.Helper()
	z, err := zstdBytes(b, 19, true)
	if err != nil {
		t.Fatal(err)
	}
	return z
}

// Floats at 2^63 and 2^64 are out of range for int64 and uint64: their
// float64 bounds round up to them. Converting them was implementation
// defined (different on amd64 and arm64).
func TestCastValueAtIntegerLimits(t *testing.T) {
	ctx := context.Background()
	const two63, two64 = 9223372036854775808.0, 18446744073709551616.0
	for _, tc := range []struct {
		to, oor string
		in      []float64
		want    any // []int64, []uint64, or nil for an error
	}{
		{"int64", "", []float64{two63}, nil},
		{"int64", "", []float64{-two63, two63 - 1024}, []int64{math.MinInt64, math.MaxInt64 - 1023}},
		{"int64", "clamp", []float64{two63, -two63 - 4096}, []int64{math.MaxInt64, math.MinInt64}},
		{"int64", "wrap", []float64{two63, 1.5 * two64, -two63 - 4096, -1}, []int64{math.MinInt64, math.MinInt64, math.MaxInt64 - 4095, -1}},
		{"uint64", "", []float64{two64}, nil},
		{"uint64", "clamp", []float64{two64, -1}, []uint64{math.MaxUint64, 0}},
		{"uint64", "wrap", []float64{two64, -1, two64 + 16384, -3 * two64}, []uint64{0, math.MaxUint64, 16384, 0}},
		{"int8", "wrap", []float64{300, -129, 1e10}, []int64{44, 127, 0}},
	} {
		in, _ := FromSlice([]uint64{uint64(len(tc.in))}, tc.in)
		spec := ChunkSpec{Shape: in.Shape, DataType: in.DataType}
		c, err := NewCodec(CastValue(tc.to, "", tc.oor), spec)
		if err != nil {
			t.Fatal(err)
		}
		out, err := c.(ArrayToArrayCodec).Encode(ctx, in, spec)
		if tc.want == nil {
			if err == nil {
				t.Errorf("%v to %s accepted", tc.in, tc.to)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v to %s (%s): %v", tc.in, tc.to, tc.oor, err)
			continue
		}
		var got any
		if tc.to == "uint64" {
			got, _ = Convert[uint64](out)
		} else {
			got, _ = Convert[int64](out)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%v to %s (%s) = %v, want %v", tc.in, tc.to, tc.oor, got, tc.want)
		}
	}
	// FromValues checks the same bounds.
	for _, tc := range []struct {
		dt string
		v  float64
		ok bool
	}{{"int64", two63, false}, {"int64", -two63, true}, {"uint64", two64, false}, {"uint64", two64 / 2, true}, {"int8", 128, false}, {"int8", -128, true}} {
		if _, err := FromValues([]uint64{1}, MustDataType(tc.dt), []float64{tc.v}); (err == nil) != tc.ok {
			t.Errorf("FromValues(%s, %v): %v", tc.dt, tc.v, err)
		}
	}
}

// Element sizes and datetime scales beyond NumPy's limits (2^31-1) are
// refused; a negative or huge size made parsing panic allocating the fill
// value.
func TestDataTypeSizesAreBounded(t *testing.T) {
	v3 := func(dt string) string {
		return `{"zarr_format":3,"node_type":"array","shape":[1],"data_type":` + dt + `,
			"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[1]}},
			"chunk_key_encoding":{"name":"default"},"fill_value":0,"codecs":[{"name":"bytes"}]}`
	}
	v2 := func(dt string) string {
		return `{"zarr_format":2,"shape":[1],"chunks":[1],"dtype":` + dt + `,"fill_value":null,"order":"C","filters":null,"compressor":null}`
	}
	for _, doc := range []string{
		v3(`{"name":"raw_bytes","configuration":{"length_bytes":4611686018427387904}}`),
		v3(`{"name":"fixed_length_utf32","configuration":{"length_bytes":2147483648}}`),
		v3(`"r800000000000000000"`),
		v3(`{"name":"struct","configuration":{"fields":[{"name":"a","data_type":{"name":"raw_bytes","configuration":{"length_bytes":2147483647}}},
			{"name":"b","data_type":{"name":"raw_bytes","configuration":{"length_bytes":2147483647}}}]}}`),
		v3(`{"name":"numpy.timedelta64","configuration":{"unit":"h","scale_factor":4611686018427387904}}`),
		v3(`{"name":"numpy.datetime64","configuration":{"unit":"s","scale_factor":1.5}}`),
		v2(`"|S-5"`),
		v2(`"<U-1"`),
		v2(`"|V4611686018427387904"`),
		v2(`"<U1073741824"`),
		v2(`"<m8[4294967296s]"`),
		v2(`[["a", "|V2147483647"], ["b", "|V2147483647"]]`),
	} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panic: %v", doc, r)
				}
			}()
			if _, err := ParseMetadata([]byte(doc)); err == nil {
				t.Errorf("accepted %s", doc)
			}
		}()
	}
	// The same NumPy type strings in codec configurations.
	for _, dt := range []string{"<U-1", "|S-8", "<V2147483648"} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("astype %s: panic: %v", dt, r)
				}
			}()
			doc := []byte(`{"zarr_format":3,"node_type":"array","shape":[2],"data_type":"float32",
				"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},"chunk_key_encoding":{"name":"default"},
				"fill_value":0,"codecs":[{"name":"numcodecs.astype","configuration":{"encode_dtype":"` + dt + `"}},{"name":"bytes"}]}`)
			if _, err := NewArray(doc, &mapChunks{}, nil); err == nil {
				t.Errorf("astype %s accepted", dt)
			}
		}()
	}
}

// ReadChunk turns panics into errors, as the other reads do.
func TestPanicInReadChunkIsAnError(t *testing.T) {
	doc := []byte(`{"zarr_format":3,"node_type":"array","shape":[4],"data_type":"int8",
		"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},
		"chunk_key_encoding":{"name":"default"},"fill_value":0,"codecs":[{"name":"bytes"}]}`)
	arr, err := NewArray(doc, panickySource{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := arr.ReadChunk(context.Background(), []uint64{1}); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("err = %v, want the panic as an error", err)
	}
}

// NDArrays whose data does not match their shape are errors, not panics.
func TestInconsistentNDArrays(t *testing.T) {
	ctx := context.Background()
	a, err := CreateArray(ctx, NewMemoryStore(), "a", ArraySpec{Shape: []uint64{2, 2}, DataType: "int8"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	short := func(dt DataType) *NDArray { return &NDArray{Shape: []uint64{2, 2}, DataType: dt, Data: []byte{1}} }
	st := Struct(StructField{Name: "a", Type: MustDataType("int8")})
	for name, f := range map[string]func() error{
		"GetMask":  func() error { _, err := a.GetMask(ctx, short(MustDataType("bool"))); return err },
		"SetMask":  func() error { return a.SetMask(ctx, short(MustDataType("int8")), short(MustDataType("bool"))) },
		"Field":    func() error { _, err := short(st).Field("a"); return err },
		"Times":    func() error { _, err := short(Datetime64("s", 1)).Times(); return err },
		"Float64s": func() error { _, err := short(MustDataType("int8")).Float64s(); return err },
		"FillValue": func() error {
			_, err := (ArraySpec{Shape: []uint64{2}, DataType: "int16", FillValue: &NDArray{Shape: []uint64{}, DataType: MustDataType("int16"), Data: []byte{1}}}).Metadata()
			return err
		},
	} {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panic: %v", name, r)
				}
			}()
			if err := f(); err == nil {
				t.Errorf("%s accepted inconsistent data", name)
			}
		}()
	}
}

// Concurrent calls on one array, each writing and reading back its own
// shards and reading the others', under the race detector.
func TestConcurrentUse(t *testing.T) {
	ctx := context.Background()
	for _, spec := range []ArraySpec{
		{Shape: []uint64{8, 30}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "int32"},
		{Shape: []uint64{8, 30}, ChunkShape: []uint64{4, 6}, ShardShape: []uint64{8, 6}, DataType: "string",
			Codecs: []CodecSpec{Sharding([]uint64{2, 3}, []CodecSpec{VlenUTF8()}, nil, "start")}},
	} {
		a, err := CreateArray(ctx, NewMemoryStore(), "a", spec, &CreateOptions{Options: Options{Concurrency: 8}})
		if err != nil {
			t.Fatal(err)
		}
		dt := a.DataType()
		const workers = 5 // one column of shards each
		done := make(chan error, workers)
		for w := range workers {
			go func() {
				lo := int64(6 * w)
				for i := range int64(20) {
					// Repeated indices: the last value wins.
					v := valuesND(dt, []uint64{3, 2}, []int64{i, i, i, i, i, i})
					if err := a.Set(ctx, v, Indices(1, 5, 5), Indices(lo+1, lo+1)); err != nil {
						done <- err
						return
					}
					if err := a.SetCoordinates(ctx, valuesND(dt, []uint64{2}, []int64{i, i}), []int64{0, 7}, []int64{lo, lo + 5}); err != nil {
						done <- err
						return
					}
					got, err := a.Get(ctx, Indices(0, 1, 5, 7, 5), Indices(lo, lo+1, lo+5, lo+1))
					if err != nil {
						done <- err
						return
					}
					// (Not ndValues: t.Fatal cannot be called here.)
					want := valuesND(dt, []uint64{}, []int64{i})
					for _, k := range []int{0, 5, 14, 19} {
						if dt.Variable() && got.Strings[k] != want.Strings[0] || !dt.Variable() && string(got.Data[4*k:4*k+4]) != string(want.Data) {
							done <- fmt.Errorf("element %d of %v %v after writing %d", k, got.Data, got.Strings, i)
							return
						}
					}
					// Shards other workers write to: a read concurrent with
					// a write to the same shard may fail.
					a.Get(ctx, SliceStep(0, End, 3), Indices(0, 29, 13, 13))
					a.GetCoordinates(ctx, []int64{0, 7, 3}, []int64{2, 29, 13})
				}
				done <- nil
			}()
		}
		for range workers {
			if err := <-done; err != nil {
				t.Fatal(err)
			}
		}
		for w := range int64(workers) {
			lo := 6 * w
			got, err := a.GetCoordinates(ctx, []int64{0, 7, 1, 5}, []int64{lo, lo + 5, lo + 1, lo + 1})
			if err != nil {
				t.Fatal(err)
			}
			if v := ndValues(t, got); !slices.Equal(v, []int64{19, 19, 19, 19}) {
				t.Errorf("%s: worker %d wrote %v", dt.Name, w, v)
			}
		}
	}
}

// numcodecs.shuffle with a huge elementsize from the metadata used to loop
// that many times over an empty buffer.
func TestShuffleOfEmptyBuffer(t *testing.T) {
	c, err := NewCodec(Codec("numcodecs.shuffle", map[string]any{"elementsize": int64(1) << 40}), ChunkSpec{Shape: []uint64{1}, DataType: MustDataType("uint8")})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := c.(BytesCodec).Decode(nil); done <- err }()
	select {
	case err := <-done:
		if err != nil {
			t.Error(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("shuffling an empty buffer hangs")
	}
}

// A .zmetadata entry with only .zattrs is not a member: MemberNames
// listed it, and Members then failed to open it. (zarr-python refuses to
// open such a group at all.)
func TestConsolidatedV2OrphanAttributes(t *testing.T) {
	ctx := context.Background()
	st := NewMemoryStore()
	st.Set(ctx, ".zgroup", []byte(`{"zarr_format": 2}`))
	st.Set(ctx, ".zmetadata", []byte(`{"metadata": {".zgroup": {"zarr_format": 2}, "x/.zattrs": {"a": 1},
		"y/.zgroup": {"zarr_format": 2}}, "zarr_consolidated_format": 1}`))
	g, err := OpenGroup(ctx, st, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	names, err := g.MemberNames(ctx)
	if err != nil || !slices.Equal(names, []string{"y"}) {
		t.Errorf("MemberNames = %v, %v", names, err)
	}
	if _, err := g.Members(ctx); err != nil {
		t.Error(err)
	}
}

// emptyMissing returns empty, non-nil bytes for missing chunks.
type emptyMissing struct{ *mapChunks }

func (s emptyMissing) GetChunk(ctx context.Context, c []uint64, r ByteRange) ([]byte, bool, error) {
	b, ok, err := s.mapChunks.GetChunk(ctx, c, r)
	if !ok {
		b = []byte{}
	}
	return b, ok, err
}

// A partial write to a missing shard ignores whatever bytes the source
// returned with ok false.
func TestShardWriteToMissingShard(t *testing.T) {
	doc, err := ArraySpec{Shape: []uint64{4}, ChunkShape: []uint64{2}, ShardShape: []uint64{4}, DataType: "int8"}.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	a, err := NewArray(doc, emptyMissing{&mapChunks{}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	one, _ := FromSlice([]uint64{}, []int8{1})
	if err := a.Set(context.Background(), one, Index(1)); err != nil {
		t.Error(err)
	}
}
