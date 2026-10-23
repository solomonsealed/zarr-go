package zarr

import (
	"context"
	"fmt"
	"math/rand"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"sync"
	"testing"
)

// fuzzInput turns fuzzer bytes into choices; it returns zeros once the
// bytes run out, so every input is a valid program.
type fuzzInput struct{ b []byte }

func (r *fuzzInput) byte() byte {
	if len(r.b) == 0 {
		return 0
	}
	c := r.b[0]
	r.b = r.b[1:]
	return c
}

// intn returns a number in [0, n), n > 0.
func (r *fuzzInput) intn(n int) int {
	if n <= 1 {
		return 0
	}
	v := int(r.byte())
	if n > 256 {
		v = v<<8 | int(r.byte())
	}
	return v % n
}

// rangeIn returns a number in [lo, hi].
func (r *fuzzInput) rangeIn(lo, hi int64) int64 { return lo + int64(r.intn(int(hi-lo+1))) }

// ---------------------------------------------------------------------------
// Selections against a reference

// selectionSpecs are the arrays FuzzSelection runs on. Values are the
// integers of a reference array, stored as int32 or as their decimal
// strings; the fill value is -1.
var selectionSpecs = []ArraySpec{
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, DataType: "int32"},
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "int32", Compressors: []CodecSpec{}},
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 6}, ShardShape: []uint64{8, 6}, DataType: "int32",
		Codecs: []CodecSpec{Sharding([]uint64{2, 3}, []CodecSpec{BytesLE(), Crc32c()}, nil, "start")}},
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, ChunkEdges: [][]uint64{{1, 3, 5}, nil}, DataType: "int32"},
	{Shape: []uint64{12, 7}, ChunkShape: []uint64{4, 3}, ChunkEdges: [][]uint64{{2, 1, 6, 3}, {3, 4}}, DataType: "int32"}, // see resizedSpecs
	{ZarrFormat: 2, Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, DataType: ">i4", Order: "F"},
	{Shape: []uint64{5, 6, 7}, ChunkShape: []uint64{2, 4, 3}, DataType: "int32", Filters: []CodecSpec{Transpose([]int{2, 0, 1})}, Serializer: ptr(Bytes("big"))},
	{Shape: []uint64{5, 6, 7}, ChunkShape: []uint64{1, 2, 3}, ShardShape: []uint64{2, 4, 6}, DataType: "int32", IndexLocation: "start"},
	{Shape: []uint64{50}, ChunkShape: []uint64{3}, ShardShape: []uint64{12}, DataType: "int32"},
	{Shape: []uint64{50}, ChunkShape: []uint64{64}, DataType: "int32"},
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, DataType: "string"},
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 3}, DataType: "string"},
	{Shape: []uint64{}, DataType: "int32"},
	{Shape: []uint64{0, 4}, ChunkShape: []uint64{2, 2}, DataType: "int32"},
	// Shards with codecs around them are decoded and encoded whole.
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 6}, DataType: "int32",
		Codecs: []CodecSpec{Sharding([]uint64{2, 3}, []CodecSpec{Transpose([]int{1, 0}), BytesLE()}, nil, "end"), Crc32c()}},
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 6}, DataType: "int32",
		Codecs: []CodecSpec{Transpose([]int{1, 0}), Sharding([]uint64{3, 2}, []CodecSpec{BytesLE()}, nil, "start")}},
	{ZarrFormat: 2, Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, DataType: "|O", Order: "F"},
	{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, ChunkEdges: [][]uint64{{2, 2, 5}, {1, 6}}, DataType: "string"},
}

// resizedSpecs are shrunk before use, leaving chunk edges past the end.
var resizedSpecs = map[int][]uint64{4: {9, 5}}

func ptr[T any](v T) *T { return &v }

// refArray is the reference: values in C order, and the chunk grid's
// starts per dimension.
type refArray struct {
	shape  []uint64
	vals   []int64
	starts [][]uint64 // chunk (shard) boundaries, the last >= shape
}

func newRefArray(a *Array) *refArray {
	r := &refArray{shape: a.Shape(), vals: slices.Repeat([]int64{-1}, int(numElements(a.Shape())))}
	grid := a.ChunkGridShape()
	r.starts = make([][]uint64, len(r.shape))
	for d := range r.shape {
		r.starts[d] = []uint64{0}
		for c := uint64(0); c < grid[d]; c++ {
			s, n := a.meta.grid.span(d, c)
			r.starts[d] = append(r.starts[d], s+n)
		}
	}
	return r
}

func (r *refArray) strides() []uint64 { return stridesOf(r.shape) }

func valuesND(dt DataType, shape []uint64, vals []int64) *NDArray {
	if dt.Variable() {
		strs := make([]string, len(vals))
		for i, v := range vals {
			strs[i] = strconv.FormatInt(v, 10)
		}
		n, err := FromStrings(shape, strs)
		if err != nil {
			panic(err)
		}
		return n
	}
	n, err := FromValues(shape, dt, vals)
	if err != nil {
		panic(err)
	}
	return n
}

func ndValues(t *testing.T, n *NDArray) []int64 {
	t.Helper()
	if n.DataType.Variable() {
		out := make([]int64, len(n.Strings))
		for i, s := range n.Strings {
			v, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				t.Fatalf("element %d is %q", i, s)
			}
			out[i] = v
		}
		return out
	}
	v, err := Convert[int64](n)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

// dimChoice is a selector with the positions it selects.
type dimChoice struct {
	sel     DimSelector
	idx     []uint64
	squeeze bool
	invalid bool
}

func randSelector(in *fuzzInput, n uint64) dimChoice {
	ni := int64(n)
	switch in.intn(16) {
	case 0, 1, 2:
		if n == 0 {
			return dimChoice{sel: Index(0), invalid: true}
		}
		i := in.rangeIn(-ni, ni-1)
		return dimChoice{sel: Index(i), idx: []uint64{uint64((i + ni) % ni)}, squeeze: true}
	case 3, 4, 5, 6, 7:
		start, stop := in.rangeIn(-ni-2, ni+2), in.rangeIn(-ni-2, ni+2)
		if in.intn(4) == 0 {
			stop = End
		}
		step := in.rangeIn(1, max(ni, 1)+1)
		lo, hi := pyClamp(start, ni), pyClamp(stop, ni)
		var idx []uint64
		for i := lo; i < hi; i += step {
			idx = append(idx, uint64(i))
		}
		return dimChoice{sel: SliceStep(start, stop, step), idx: idx}
	case 8, 9, 10:
		var sel []int64
		var idx []uint64
		if n > 0 {
			for range in.intn(int(n) + 3) {
				i := in.rangeIn(-ni, ni-1)
				sel = append(sel, i)
				idx = append(idx, uint64((i+ni)%ni))
			}
		}
		return dimChoice{sel: IndicesSelector(sel), idx: idx}
	case 11, 12:
		m := make([]bool, n)
		var idx []uint64
		for i := range m {
			if m[i] = in.intn(2) == 1; m[i] {
				idx = append(idx, uint64(i))
			}
		}
		return dimChoice{sel: MaskSelector(m), idx: idx}
	case 13:
		switch in.intn(5) {
		case 0:
			return dimChoice{sel: Index(ni), invalid: true}
		case 1:
			return dimChoice{sel: Indices(0, -ni-1), invalid: true}
		case 2:
			return dimChoice{sel: Mask(make([]bool, n+1)...), invalid: true}
		case 3:
			return dimChoice{sel: SliceStep(0, 1, 0), invalid: true}
		}
		return dimChoice{sel: SliceStep(0, 1, -1), invalid: true}
	}
	idx := make([]uint64, n)
	for i := range idx {
		idx[i] = uint64(i)
	}
	return dimChoice{sel: All(), idx: idx}
}

// pyClamp resolves a slice bound like Python's slice.indices.
func pyClamp(v, n int64) int64 {
	if v == End {
		return n
	}
	if v < 0 {
		v += n
	}
	return min(max(v, 0), n)
}

// randSelection returns selectors for every dimension, sometimes with an
// Ellipsis or with trailing All() selectors left out.
func randSelection(in *fuzzInput, shape []uint64) ([]DimSelector, []dimChoice) {
	nd := len(shape)
	choices := make([]dimChoice, nd)
	sel := make([]DimSelector, nd)
	for d := range choices {
		choices[d] = randSelector(in, shape[d])
		sel[d] = choices[d].sel
	}
	switch in.intn(4) {
	case 0: // an Ellipsis standing for some All() dimensions
		p := in.intn(nd + 1)
		c := 0
		for p+c < nd && in.intn(2) == 0 {
			c++
		}
		for d := p; d < p+c; d++ {
			choices[d] = randSelector(&fuzzInput{[]byte{15}}, shape[d])
		}
		sel = slices.Concat(sel[:p], []DimSelector{Ellipsis}, sel[p+c:])
	case 1: // trailing selectors left out
		for len(sel) > 0 && in.intn(2) == 0 {
			d := len(sel) - 1
			choices[d] = randSelector(&fuzzInput{[]byte{15}}, shape[d])
			sel = sel[:d]
		}
	}
	return sel, choices
}

// positions lists the linear positions an orthogonal selection visits, in
// C order, and the result shape.
func (r *refArray) positions(choices []dimChoice) ([]uint64, []uint64) {
	var shape []uint64
	counts := make([]uint64, len(choices))
	for d, c := range choices {
		counts[d] = uint64(len(c.idx))
		if !c.squeeze {
			shape = append(shape, counts[d])
		}
	}
	st := r.strides()
	var out []uint64
	forEachIndex(counts, func(_ uint64, k []uint64) bool {
		var p uint64
		for d := range k {
			p += choices[d].idx[k[d]] * st[d]
		}
		out = append(out, p)
		return true
	})
	if shape == nil {
		shape = []uint64{}
	}
	return out, shape
}

// randData returns values to write to a selection of the given shape:
// the full shape, a single value or a shape that broadcasts to it.
func randData(in *fuzzInput, shape []uint64, next *int64) ([]uint64, []int64, []int64) {
	dshape := slices.Clone(shape)
	switch in.intn(4) {
	case 0:
		dshape = []uint64{}
	case 1:
		k := in.intn(len(dshape) + 1)
		dshape = dshape[k:]
		for d := range dshape {
			if in.intn(3) == 0 {
				dshape[d] = 1
			}
		}
	}
	vals := make([]int64, numElements(dshape))
	for i := range vals {
		*next++
		vals[i] = *next
	}
	// Broadcast to shape for the reference.
	full := make([]int64, numElements(shape))
	off := len(shape) - len(dshape)
	dst := stridesOf(dshape)
	forEachIndex(shape, func(i uint64, k []uint64) bool {
		var s uint64
		for d := range dshape {
			if dshape[d] != 1 {
				s += k[off+d] * dst[d]
			}
		}
		full[i] = vals[s]
		return true
	})
	return dshape, vals, full
}

// blockChoice picks a block selection and its element ranges.
func (r *refArray) randBlocks(in *fuzzInput) ([]DimSelector, []dimChoice, bool) {
	sel := make([]DimSelector, len(r.shape))
	choices := make([]dimChoice, len(r.shape))
	for d, n := range r.shape {
		st := r.starts[d]
		nc := int64(len(st) - 1)
		if nc == 0 {
			return nil, nil, false
		}
		var lo, hi int64
		if in.intn(2) == 0 {
			i := in.rangeIn(-nc, nc-1)
			sel[d] = Index(i)
			lo = (i + nc) % nc
			hi = lo + 1
		} else {
			lo = in.rangeIn(0, nc-1)
			hi = in.rangeIn(lo, nc)
			sel[d] = Slice(lo, hi)
		}
		for p := st[lo]; p < min(st[hi], n); p++ {
			choices[d].idx = append(choices[d].idx, p)
		}
	}
	return sel, choices, true
}

func checkSelections(t *testing.T, specIndex int, ops []byte) {
	ctx := context.Background()
	specIndex %= len(selectionSpecs)
	spec := selectionSpecs[specIndex]
	spec.FillValue = int32(-1)
	if spec.DataType == "string" || spec.DataType == "|O" {
		spec.FillValue = "-1"
	}
	store := NewMemoryStore()
	in := &fuzzInput{ops}
	a, err := CreateArray(ctx, store, "a", spec, &CreateOptions{Options: Options{Concurrency: 1 + in.intn(4), WriteEmptyChunks: in.intn(4) == 0}})
	if err != nil {
		t.Fatal(err)
	}
	if shape := resizedSpecs[specIndex]; shape != nil {
		if err := a.Resize(ctx, shape); err != nil {
			t.Fatal(err)
		}
	}
	ref := newRefArray(a)
	dt := a.DataType()
	var next int64
	check := func(what string, got *NDArray, err error, wantShape []uint64, pos []uint64) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		if !slices.Equal(got.Shape, wantShape) {
			t.Fatalf("%s: shape %v, want %v", what, got.Shape, wantShape)
		}
		vals := ndValues(t, got)
		for i, p := range pos {
			if vals[i] != ref.vals[p] {
				t.Fatalf("%s: element %d is %d, want %d (position %d)", what, i, vals[i], ref.vals[p], p)
			}
		}
	}
	for op := 0; len(in.b) > 0 && op < 64; op++ {
		switch in.intn(9) {
		case 0, 1: // orthogonal read
			sel, choices := randSelection(in, ref.shape)
			got, err := a.Get(ctx, sel...)
			if slices.ContainsFunc(choices, func(c dimChoice) bool { return c.invalid }) {
				if err == nil {
					t.Fatalf("Get(%v) accepted an invalid selection", sel)
				}
				continue
			}
			pos, shape := ref.positions(choices)
			check(fmt.Sprintf("Get(%v)", sel), got, err, shape, pos)
		case 2, 3: // orthogonal write
			sel, choices := randSelection(in, ref.shape)
			invalid := slices.ContainsFunc(choices, func(c dimChoice) bool { return c.invalid })
			var pos, shape []uint64
			if !invalid {
				pos, shape = ref.positions(choices)
			}
			dshape, vals, full := randData(in, shape, &next)
			err := a.Set(ctx, valuesND(dt, dshape, vals), sel...)
			if invalid {
				if err == nil {
					t.Fatalf("Set(%v) accepted an invalid selection", sel)
				}
				continue
			}
			if err != nil {
				t.Fatalf("Set(%v) with data of shape %v: %v", sel, dshape, err)
			}
			for i, p := range pos {
				ref.vals[p] = full[i]
			}
		case 4: // points
			if len(ref.shape) == 0 || numElements(ref.shape) == 0 {
				continue
			}
			npts := in.intn(12)
			coords := make([][]int64, len(ref.shape))
			pos := make([]uint64, npts)
			st := ref.strides()
			for i := range npts {
				for d, n := range ref.shape {
					c := in.rangeIn(-int64(n), int64(n)-1)
					coords[d] = append(coords[d], c)
					pos[i] += uint64((c+int64(n))%int64(n)) * st[d]
				}
			}
			if npts == 0 {
				for d := range coords {
					coords[d] = []int64{}
				}
			}
			if in.intn(2) == 0 {
				got, err := a.GetCoordinates(ctx, coords...)
				check(fmt.Sprintf("GetCoordinates(%v)", coords), got, err, []uint64{uint64(npts)}, pos)
				continue
			}
			dshape, vals, full := randData(in, []uint64{uint64(npts)}, &next)
			if err := a.SetCoordinates(ctx, valuesND(dt, dshape, vals), coords...); err != nil {
				t.Fatalf("SetCoordinates(%v): %v", coords, err)
			}
			for i, p := range pos {
				ref.vals[p] = full[i]
			}
		case 5: // mask
			m := make([]bool, len(ref.vals))
			var pos []uint64
			for i := range m {
				if m[i] = in.intn(3) == 0; m[i] {
					pos = append(pos, uint64(i))
				}
			}
			mask, _ := FromValues(ref.shape, MustDataType("bool"), m)
			if len(ref.shape) == 0 {
				continue // coordinate selections need a dimension
			}
			if in.intn(2) == 0 {
				got, err := a.GetMask(ctx, mask)
				check("GetMask", got, err, []uint64{uint64(len(pos))}, pos)
				continue
			}
			dshape, vals, full := randData(in, []uint64{uint64(len(pos))}, &next)
			if err := a.SetMask(ctx, valuesND(dt, dshape, vals), mask); err != nil {
				t.Fatalf("SetMask: %v", err)
			}
			for i, p := range pos {
				ref.vals[p] = full[i]
			}
		case 6: // blocks
			sel, choices, ok := ref.randBlocks(in)
			if !ok {
				continue
			}
			pos, shape := ref.positions(choices)
			if in.intn(2) == 0 {
				got, err := a.GetBlocks(ctx, sel...)
				check(fmt.Sprintf("GetBlocks(%v)", sel), got, err, shape, pos)
				continue
			}
			dshape, vals, full := randData(in, shape, &next)
			if err := a.SetBlocks(ctx, valuesND(dt, dshape, vals), sel...); err != nil {
				t.Fatalf("SetBlocks(%v): %v", sel, err)
			}
			for i, p := range pos {
				ref.vals[p] = full[i]
			}
		case 7: // a whole chunk
			grid := a.ChunkGridShape()
			if numElements(grid) == 0 {
				continue
			}
			coords := make([]uint64, len(grid))
			for d := range grid {
				coords[d] = uint64(in.intn(int(grid[d])))
			}
			c, err := a.ReadChunk(ctx, coords)
			if err != nil {
				t.Fatalf("ReadChunk(%v): %v", coords, err)
			}
			vals := ndValues(t, c)
			st := ref.strides()
			forEachIndex(c.Shape, func(i uint64, k []uint64) bool {
				var p uint64
				for d := range k {
					x := ref.starts[d][coords[d]] + k[d]
					if x >= ref.shape[d] {
						return true
					}
					p += x * st[d]
				}
				if vals[i] != ref.vals[p] {
					t.Fatalf("ReadChunk(%v): element %v is %d, want %d", coords, k, vals[i], ref.vals[p])
				}
				return true
			})
		case 8: // everything, through a newly opened array
			b, err := OpenArray(ctx, store, "a", nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := b.ReadAll(ctx)
			all := make([]uint64, len(ref.vals))
			for i := range all {
				all[i] = uint64(i)
			}
			check("ReadAll", got, err, ref.shape, all)
		}
	}
}

func FuzzSelection(f *testing.F) {
	rng := rand.New(rand.NewSource(1))
	for i := range 3 * len(selectionSpecs) {
		ops := make([]byte, 600)
		rng.Read(ops)
		f.Add(uint8(i), ops)
	}
	f.Fuzz(func(t *testing.T, spec uint8, ops []byte) {
		checkSelections(t, int(spec), ops)
	})
}

// ---------------------------------------------------------------------------
// Decoding arbitrary chunk bytes

// decodeSpecs cover the codecs and data types chunk bytes can be decoded
// with.
var decodeSpecs = []ArraySpec{
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int16", Compressors: []CodecSpec{}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int32", Serializer: ptr(Bytes("big")), Compressors: []CodecSpec{Crc32c()}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "float32"},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "float64", Compressors: []CodecSpec{Gzip(1)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int32", Compressors: []CodecSpec{Blosc("lz4", 5, "shuffle")}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "uint16", Compressors: []CodecSpec{Codec("numcodecs.lz4", nil), Codec("numcodecs.crc32", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "string"},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "variable_length_bytes", Compressors: []CodecSpec{}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "bool", Filters: []CodecSpec{Codec("numcodecs.packbits", nil)}, Compressors: []CodecSpec{}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int32", Filters: []CodecSpec{Codec("numcodecs.delta", map[string]any{"dtype": "<i4"})},
		Compressors: []CodecSpec{Codec("numcodecs.shuffle", nil), Codec("numcodecs.zlib", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "int16", Compressors: []CodecSpec{}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "float32", IndexLocation: "start"},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 6}, ShardShape: []uint64{8, 6}, DataType: "int16",
		Codecs: []CodecSpec{Sharding([]uint64{2, 3}, []CodecSpec{BytesLE()}, []CodecSpec{BytesLE()}, "start")}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "string"},
	{Shape: []uint64{3, 4, 5}, ChunkShape: []uint64{2, 3, 4}, DataType: "uint8", Filters: []CodecSpec{Transpose([]int{1, 2, 0})}, Compressors: []CodecSpec{Codec("numcodecs.bz2", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "float64", Filters: []CodecSpec{Codec("numcodecs.fixedscaleoffset", map[string]any{"offset": 1, "scale": 10, "astype": "<i2"})}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int16", Filters: []CodecSpec{ScaleOffset(2, 3), CastValue("int8", "", "clamp")}},
	{ZarrFormat: 2, Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "<f8", Compressors: []CodecSpec{Codec("blosc", map[string]any{"cname": "zstd", "clevel": 3, "shuffle": 2})}},
	{ZarrFormat: 2, Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "|O", Type: DataType{}, Order: "F"},
	{ZarrFormat: 2, Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "<U3", Filters: []CodecSpec{Codec("categorize", map[string]any{"labels": []string{"a", "bc"}, "dtype": "<U3"})}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int64", Compressors: []CodecSpec{Codec("numcodecs.lzma", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int8", Compressors: []CodecSpec{Codec("numcodecs.fletcher32", nil), Codec("numcodecs.adler32", nil), Codec("numcodecs.jenkins_lookup3", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, Type: Struct(StructField{Name: "a", Type: MustDataType("int16")}, StructField{Name: "b", Type: MustDataType("float64")}), Compressors: []CodecSpec{}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "int4", Compressors: []CodecSpec{}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "float8_e4m3fn", Compressors: []CodecSpec{Codec("numcodecs.base64", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "float32", Filters: []CodecSpec{Codec("numcodecs.bitround", map[string]any{"keepbits": 5})}, Compressors: []CodecSpec{Codec("numcodecs.zstd", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "float64", Filters: []CodecSpec{Codec("numcodecs.astype", map[string]any{"encode_dtype": "<f4"})}, Compressors: []CodecSpec{Codec("numcodecs.gzip", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "uint32", IndexCodecs: []CodecSpec{Bytes("big"), Codec("numcodecs.crc32", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, Type: FixedLengthUTF32(2), Compressors: []CodecSpec{Codec("numcodecs.shuffle", map[string]any{"elementsize": 4})}},
	{ZarrFormat: 2, Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: ">i8", Order: "F", Compressors: []CodecSpec{Codec("zlib", nil)}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{4, 3}, DataType: "bool", Compressors: []CodecSpec{Blosc("blosclz", 9, "bitshuffle")}},
	{Shape: []uint64{6, 5}, ChunkShape: []uint64{2, 3}, ShardShape: []uint64{4, 6}, DataType: "int16", Compressors: []CodecSpec{Codec("numcodecs.lz4", nil)}},
}

// fuzzChunks serves one chunk's bytes for every chunk and keeps writes.
type fuzzChunks struct {
	mu   sync.Mutex
	data []byte
}

func (s *fuzzChunks) GetChunk(_ context.Context, _ []uint64, r ByteRange) ([]byte, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.data == nil {
		return nil, false, nil
	}
	b, err := r.Slice(s.data)
	return slices.Clone(b), true, err
}

func (s *fuzzChunks) SetChunk(_ context.Context, _ []uint64, d []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = slices.Clone(d)
	return nil
}

func (s *fuzzChunks) DeleteChunk(context.Context, []uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = nil
	return nil
}

// decodeDocs returns the metadata of each of decodeSpecs and a valid first
// chunk.
var decodeDocs = sync.OnceValue(func() (out [][2][]byte) {
	for _, spec := range decodeSpecs {
		docs, err := spec.Documents()
		if err != nil {
			panic(fmt.Sprintf("%+v: %v", spec, err))
		}
		doc := docs["zarr.json"]
		if doc == nil {
			doc = docs[".zarray"]
		}
		src := &fuzzChunks{}
		a, err := NewArray(doc, src, nil)
		if err != nil {
			panic(err)
		}
		n := int(numElements(a.Shape()))
		var data *NDArray
		switch dt := a.DataType(); {
		case dt.Variable() || dt.Kind == KindFixedString:
			strs := make([]string, n)
			for i := range strs {
				strs[i] = []string{"a", "bc", ""}[i%3]
			}
			data, err = FromValues(a.Shape(), dt, strs)
		case dt.Kind == KindStruct:
			vals := make([]map[string]any, n)
			for i := range vals {
				vals[i] = map[string]any{"a": i, "b": float64(i) / 4}
			}
			data, err = FromValues(a.Shape(), dt, vals)
		default:
			vals := make([]int64, n)
			for i := range vals {
				vals[i] = int64(i % 7)
			}
			data, err = FromValues(a.Shape(), dt, vals)
		}
		if err != nil {
			panic(err)
		}
		if err := a.Set(context.Background(), data); err != nil {
			panic(err)
		}
		out = append(out, [2][]byte{doc, src.data})
	}
	return out
})

func FuzzDecodeChunk(f *testing.F) {
	for i, d := range decodeDocs() {
		f.Add(uint8(i), d[1])
		if len(d[1]) > 8 {
			// A truncated and a corrupted copy.
			f.Add(uint8(i), d[1][:len(d[1])/2])
			c := slices.Clone(d[1])
			c[len(c)/3] ^= 0x5a
			f.Add(uint8(i), c)
		}
	}
	// Headers claiming huge sizes: a zstd frame of 15 GiB, an LZ4 block
	// of 2 GiB, a vlen chunk of 2^32-1 items.
	f.Add(uint8(2), []byte{0x28, 0xb5, 0x2f, 0xfd, 0xc0, 0x00, 0, 0, 0, 0xc0, 3, 0, 0, 0, 1, 0, 0})
	f.Add(uint8(5), []byte{0, 0, 0, 0x80, 0, 0, 0, 0, 0})
	f.Add(uint8(7), []byte{0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0})
	f.Fuzz(func(t *testing.T, which uint8, chunk []byte) {
		ctx := context.Background()
		doc := decodeDocs()[int(which)%len(decodeSpecs)][0]
		src := &fuzzChunks{data: slices.Clone(chunk)}
		a, err := NewArray(doc, src, nil)
		if err != nil {
			t.Fatal(err)
		}
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		zeros := make([]uint64, a.NDim())
		_, spec := a.chunkSpec(zeros)
		// Not through the pool, which turns panics into errors.
		a.pipe.decode(ctx, chunk, spec)
		a.ReadChunk(ctx, zeros)
		sel := make([]DimSelector, a.NDim())
		for d := range sel {
			sel[d] = Index(0)
		}
		a.Get(ctx, sel...)
		if len(sel) > 0 {
			sel[len(sel)-1] = SliceStep(1, End, 2)
		}
		a.Get(ctx, sel...)
		a.Set(ctx, a.FillValue(), sel...)
		runtime.ReadMemStats(&after)
		if n := after.TotalAlloc - before.TotalAlloc; n > 1<<30 {
			t.Fatalf("decoding %d bytes allocated %d MB", len(chunk), n>>20)
		}
	})
}

// ---------------------------------------------------------------------------
// Parsing arbitrary metadata

func metadataSeeds() [][]byte {
	var out [][]byte
	specs := slices.Concat(selectionSpecs, decodeSpecs, []ArraySpec{
		{Shape: []uint64{4}, Type: Datetime64("ms", 10), FillValue: "NaT"},
		{ZarrFormat: 2, Shape: []uint64{4}, DataType: "<m8[3s]"},
		{Shape: []uint64{4, 4}, ChunkShape: []uint64{2, 2}, Type: NullTerminatedBytes(3), DimensionNames: []string{"x", ""}, Attributes: map[string]any{"a": []any{1, 2.5, "s"}}},
		{Shape: []uint64{4}, DataType: "complex64", FillValue: complex(1, -1)},
		{Shape: []uint64{4}, DataType: "float64", FillValue: "NaN", ChunkKeyEncoding: ChunkKeyEncoding{Name: "v2", Separator: "/"}},
		{ZarrFormat: 2, Shape: []uint64{4}, Type: Struct(StructField{Name: "a", Type: MustDataType("int8")}, StructField{Name: "b", Type: RawBytes(2)})},
		{Shape: []uint64{4, 3}, ChunkShape: []uint64{2, 2}, DataType: "int8", ChunkKeyEncoding: ChunkKeyEncoding{Name: "template", Template: "k/{-1}_{*:03}", Separator: "."}},
	})
	for _, spec := range specs {
		docs, err := spec.Documents()
		if err != nil {
			panic(err)
		}
		for _, name := range []string{"zarr.json", ".zarray"} {
			if d := docs[name]; d != nil {
				out = append(out, d)
			}
		}
	}
	// Invalid documents the repairs read, and group extra fields.
	out = append(out,
		[]byte(v2ArrayDoc(`[6]`, `[false]`)),
		[]byte(v3ArrayDoc(`[4, 20]`, regularGridJSON(`[2, [5.0, 10, 5]]`), `[{"name": "bytes"}]`)),
		[]byte(v3ArrayDoc(`[9]`, regularGridJSON(`[0]`), sharded3(`[true]`))),
		[]byte(v3ArrayDoc(`[6]`, `{"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": [[[2.0, 2], true, 1]]}}`, `[{"name": "bytes"}]`)),
		[]byte(`{"ext": {"must_understand": false}, "zarr_format": 3, "node_type": "group", "attributes": {}}`),
	)
	// Groups, with consolidated metadata.
	ctx := context.Background()
	for _, format := range []int{2, 3} {
		st := NewMemoryStore()
		root, _ := CreateGroup(ctx, st, "", GroupSpec{ZarrFormat: format, Attributes: map[string]any{"t": "x"}}, nil)
		root.CreateArray(ctx, "g/a", ArraySpec{Shape: []uint64{3}, ChunkShape: []uint64{2}, DataType: "int16"}, nil)
		ConsolidateMetadata(ctx, st, "")
		for _, k := range []string{"zarr.json", ".zmetadata"} {
			if d, ok, _ := st.Get(ctx, k, ByteRange{}); ok {
				out = append(out, d)
			}
		}
	}
	return out
}

// useArray reads and writes a little of an array whose chunks are small
// (reading allocates whole chunks).
func useArray(ctx context.Context, a *Array) {
	a.FillValue().Values()
	if numElements(a.meta.largestChunk()) > 1<<12 || numElements(a.ChunkGridShape()) == 0 {
		return
	}
	zeros := make([]uint64, a.NDim())
	a.ReadChunk(ctx, zeros)
	_, spec := a.chunkSpec(zeros)
	if enc, err := a.pipe.encode(ctx, newFilled(spec.Shape, a.DataType(), a.meta.fill), spec); err == nil {
		a.pipe.decode(ctx, enc, spec)
	}
	sel := make([]DimSelector, a.NDim())
	for d := range sel {
		sel[d] = Index(0)
	}
	a.Get(ctx, sel...)
	a.Set(ctx, a.FillValue(), sel...)
	if a.NDim() <= 4 {
		// A few elements spread over the whole array.
		for d, n := range a.Shape() {
			sel[d] = SliceStep(0, End, int64(max(n/5, 1)))
		}
		a.Get(ctx, sel...)
	}
	if a.NDim() > 0 {
		a.GetCoordinates(ctx, slices.Repeat([][]int64{{0, -1}}, a.NDim())...)
		a.GetBlocks(ctx, Index(-1))
	}
}

// bigElement matches element sizes from a megabyte to NumPy's limit of 2
// GiB: they are legal, and parsing allocates a fill value that size.
// (Larger ones are refused without allocating.)
var bigElement = regexp.MustCompile(`length_bytes"\s*:\s*\d{7,10}\b|"[<>|=]?[SUVr]\d{7,10}\b`)

func FuzzParseMetadata(f *testing.F) {
	for _, d := range metadataSeeds() {
		f.Add(d)
	}
	f.Fuzz(func(t *testing.T, doc []byte) {
		if bigElement.Match(doc) {
			return
		}
		ctx := context.Background()
		n := allocated(func() {
			if m, err := ParseMetadata(doc); err == nil {
				m.Document()
				m.AttributesDocument()
				m.GridShape()
				if a, err := NewArrayFromMetadata(m, &fuzzChunks{}, nil); err == nil {
					useArray(ctx, a)
				}
			}
			Attributes(doc)
			for _, name := range []string{"zarr.json", ".zarray", ".zgroup"} {
				st := NewMemoryStore()
				st.Set(ctx, name, doc)
				if name == ".zgroup" {
					st.Set(ctx, ".zgroup", []byte(`{"zarr_format": 2}`))
					st.Set(ctx, ".zattrs", doc)
					st.Set(ctx, ".zmetadata", doc)
				}
				node, err := Open(ctx, st, "", nil)
				if err != nil {
					continue
				}
				switch n := node.(type) {
				case *Array:
					useArray(ctx, n)
				case *Group:
					n.Walk(ctx, func(_ string, m Node) error {
						if a, ok := m.(*Array); ok {
							useArray(ctx, a)
						}
						return nil
					})
					ConsolidateMetadata(ctx, st, "")
				}
			}
		})
		if n > 256<<20 {
			t.Fatalf("%d MiB allocated", n>>20)
		}
	})
}

// FuzzChunkKeyTemplate checks template chunk key encodings: parsing any
// template fails cleanly or gives keys for every chunk of the arrays it
// suits, which parse back to their coordinates.
func FuzzChunkKeyTemplate(f *testing.F) {
	for _, ex := range templateExamples {
		f.Add(ex.template, ex.separator, uint64(11), uint64(22), uint64(33))
	}
	f.Add("{1:020}-{0}_{*:05}", "/", uint64(0), uint64(1<<63), uint64(7))
	f.Fuzz(func(t *testing.T, template, sep string, a, b, c uint64) {
		coords := []uint64{a, b, c}
		for nd := 0; nd <= len(coords); nd++ {
			e := ChunkKeyEncoding{Name: "template", Template: template, Separator: sep}
			if e.check(nd) != nil {
				continue
			}
			key := e.Key(coords[:nd])
			if key == "" {
				t.Fatalf("%q (%q): no key for %v", template, sep, coords[:nd])
			}
			if got, ok := e.Parse(key, nd); ok && e.Key(got) != key {
				t.Fatalf("%q: %q parsed as %v", template, key, got)
			}
		}
	})
}
