package zarr

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/bits"
	"slices"
	"sort"
)

// shardingCodec implements the sharding_indexed array-to-bytes codec: a
// shard holds a grid of inner chunks, each encoded with the inner codecs,
// plus an index of (offset, nbytes) pairs at its start or end.
type shardingCodec struct {
	inner        []uint64
	innerPipe    *pipeline
	indexPipe    *pipeline
	indexAtStart bool
	elem         ChunkSpec // data type and fill value of the shard's elements
	// writeEmpty keeps inner chunks that hold only the fill value.
	writeEmpty bool
}

func init() {
	RegisterCodec("sharding_indexed", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		return newShardingCodec(cfg, in)
	})
}

var indexType = DataType{Name: "uint64", Kind: KindUint, Size: 8}

func newShardingCodec(raw json.RawMessage, in ChunkSpec) (*shardingCodec, error) {
	var cfg struct {
		ChunkShape    []uint64    `json:"chunk_shape"`
		Codecs        []CodecSpec `json:"codecs"`
		IndexCodecs   []CodecSpec `json:"index_codecs"`
		IndexLocation string      `json:"index_location"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("zarr: sharding_indexed: %w", err)
	}
	if len(cfg.ChunkShape) != len(in.Shape) && in.Shape != nil {
		return nil, fmt.Errorf("zarr: sharding_indexed chunk_shape %v does not match %d dimensions", cfg.ChunkShape, len(in.Shape))
	}
	for i, c := range cfg.ChunkShape {
		if c == 0 {
			return nil, fmt.Errorf("zarr: sharding_indexed chunk_shape contains 0")
		}
		if in.Shape != nil && in.Shape[i]%c != 0 {
			return nil, fmt.Errorf("zarr: shard shape %v is not a multiple of inner chunk shape %v", in.Shape, cfg.ChunkShape)
		}
	}
	if in.Shape != nil {
		grid := make([]uint64, len(in.Shape))
		for i := range grid {
			grid[i] = in.Shape[i] / cfg.ChunkShape[i]
		}
		if !fitsInMemory(append(grid, 2), indexType) {
			return nil, fmt.Errorf("zarr: shards of %v inner chunks are too large", grid)
		}
	}
	switch cfg.IndexLocation {
	case "", "end", "start":
	default:
		return nil, fmt.Errorf("zarr: invalid sharding index_location %q", cfg.IndexLocation)
	}
	s := &shardingCodec{inner: cfg.ChunkShape, indexAtStart: cfg.IndexLocation == "start", elem: in}
	var err error
	innerSpec := in.withShape(cfg.ChunkShape)
	if s.innerPipe, err = buildPipeline(cfg.Codecs, innerSpec, false); err != nil {
		return nil, fmt.Errorf("zarr: sharding_indexed inner codecs: %w", err)
	}
	if cfg.IndexCodecs == nil {
		cfg.IndexCodecs = []CodecSpec{{Name: "bytes", Configuration: json.RawMessage(`{"endian":"little"}`)}, {Name: "crc32c"}}
	}
	idxSpec := ChunkSpec{Shape: append(slices.Clone(cfg.ChunkShape), 2), DataType: indexType, FillValue: fillValue{fixed: make([]byte, 8)}.ndarray(indexType)}
	if s.indexPipe, err = buildPipeline(cfg.IndexCodecs, idxSpec, false); err != nil {
		return nil, fmt.Errorf("zarr: sharding_indexed index codecs: %w", err)
	}
	if _, ok := s.indexPipe.encodedSize(idxSpec); !ok {
		return nil, fmt.Errorf("%w: sharding index codecs without a fixed encoded size (%s)", ErrUnsupported, s.indexPipe)
	}
	return s, nil
}

// gridShape returns the number of inner chunks per dimension of a shard.
func (s *shardingCodec) gridShape(shard []uint64) ([]uint64, error) {
	if len(shard) != len(s.inner) {
		return nil, fmt.Errorf("zarr: shard has %d dimensions, inner chunk_shape has %d", len(shard), len(s.inner))
	}
	g := make([]uint64, len(shard))
	for i := range shard {
		if shard[i]%s.inner[i] != 0 {
			return nil, fmt.Errorf("zarr: shard shape %v is not a multiple of inner chunk shape %v", shard, s.inner)
		}
		g[i] = shard[i] / s.inner[i]
	}
	return g, nil
}

func (s *shardingCodec) indexSpec(grid []uint64) ChunkSpec {
	return ChunkSpec{Shape: append(slices.Clone(grid), 2), DataType: indexType, FillValue: fillValue{fixed: make([]byte, 8)}.ndarray(indexType)}
}

func (s *shardingCodec) indexSize(grid []uint64) int64 {
	n, _ := s.indexPipe.encodedSize(s.indexSpec(grid))
	return int64(n)
}

// shardIndex holds the decoded (offset, nbytes) pairs; missing chunks are
// marked with offset = nbytes = 2^64-1.
type shardIndex []uint64

func (ix shardIndex) entry(i uint64) (off, n uint64, ok bool) {
	off, n = ix[2*i], ix[2*i+1]
	return off, n, !(off == math.MaxUint64 && n == math.MaxUint64)
}

func (s *shardingCodec) decodeIndex(ctx context.Context, raw []byte, grid []uint64) (shardIndex, error) {
	a, err := s.indexPipe.decode(ctx, raw, s.indexSpec(grid))
	if err != nil {
		return nil, fmt.Errorf("zarr: shard index: %w", err)
	}
	ix := make(shardIndex, len(a.Data)/8)
	for i := range ix {
		ix[i] = binary.LittleEndian.Uint64(a.Data[8*i:])
	}
	return ix, nil
}

// indexRange returns the byte range of the index in a shard.
func (s *shardingCodec) indexRange(grid []uint64) ByteRange {
	n := s.indexSize(grid)
	if s.indexAtStart {
		return ByteRange{Offset: 0, Length: n}
	}
	return ByteRange{Offset: -n}
}

// Decode decodes a complete shard.
func (s *shardingCodec) Decode(ctx context.Context, in []byte, spec ChunkSpec) (*NDArray, error) {
	out := newFilled(spec.Shape, spec.DataType, spec.fill())
	all := fullProjection(spec.Shape)
	src := bytesSource(in)
	if err := s.readPartial(ctx, src, spec, all, out, stridesOf(spec.Shape), 1); err != nil {
		return nil, err
	}
	return out, nil
}

// Encode writes a complete shard. Inner chunks that hold only the fill
// value are omitted unless writeEmpty is set.
func (s *shardingCodec) Encode(ctx context.Context, a *NDArray, spec ChunkSpec) ([]byte, error) {
	grid, err := s.gridShape(spec.Shape)
	if err != nil {
		return nil, err
	}
	chunks := make([][]byte, numElements(grid))
	present := make([]bool, len(chunks))
	innerSpec := spec.withShape(s.inner)
	var firstErr error
	forEachIndex(grid, func(i uint64, coords []uint64) bool {
		inner := newFilled(s.inner, spec.DataType, spec.fill())
		lo := make([]uint64, len(coords))
		for d := range coords {
			lo[d] = coords[d] * s.inner[d]
		}
		copyBox(inner, make([]uint64, len(coords)), a, lo, s.inner)
		if !s.writeEmpty && isFill(inner, spec.fill()) {
			return true
		}
		enc, err := s.innerPipe.encode(ctx, inner, innerSpec)
		if err != nil {
			firstErr = err
			return false
		}
		chunks[i], present[i] = enc, true
		return true
	})
	if firstErr != nil {
		return nil, firstErr
	}
	return s.assemble(ctx, grid, chunks, present)
}

// assemble lays out encoded inner chunks in Morton order, as zarr-python
// does, and adds the index. It returns nil if no chunk is present.
func (s *shardingCodec) assemble(ctx context.Context, grid []uint64, chunks [][]byte, present []bool) ([]byte, error) {
	n := len(chunks)
	index := make([]byte, 16*n)
	idxSize := int(s.indexSize(grid))
	pos := 0
	if s.indexAtStart {
		pos = idxSize
	}
	var body []byte
	any := false
	strides := stridesOf(grid)
	for _, coords := range mortonOrder(grid) {
		var i uint64
		for d, c := range coords {
			i += c * strides[d]
		}
		if !present[i] || len(chunks[i]) == 0 {
			binary.LittleEndian.PutUint64(index[16*i:], math.MaxUint64)
			binary.LittleEndian.PutUint64(index[16*i+8:], math.MaxUint64)
			continue
		}
		any = true
		binary.LittleEndian.PutUint64(index[16*i:], uint64(pos))
		binary.LittleEndian.PutUint64(index[16*i+8:], uint64(len(chunks[i])))
		body = append(body, chunks[i]...)
		pos += len(chunks[i])
	}
	if !any {
		return nil, nil
	}
	ixSpec := s.indexSpec(grid)
	encIndex, err := s.indexPipe.encode(ctx, &NDArray{Shape: ixSpec.Shape, DataType: indexType, Data: index}, ixSpec)
	if err != nil {
		return nil, err
	}
	if len(encIndex) != idxSize {
		return nil, fmt.Errorf("zarr: shard index encoded to %d bytes, expected %d", len(encIndex), idxSize)
	}
	if s.indexAtStart {
		return append(encIndex, body...), nil
	}
	return append(body, encIndex...), nil
}

// mortonOrder lists the coordinates of grid in zarr-python's "compressed"
// Morton order: bits interleaved with dimension 0 least significant, and
// dimensions dropping out once their coordinates have no more bits.
func mortonOrder(grid []uint64) [][]uint64 {
	nd := len(grid)
	nbits := make([]int, nd)
	maxBits, total := 0, 0
	for d, n := range grid {
		if n > 1 {
			nbits[d] = bits.Len64(n - 1)
		}
		maxBits = max(maxBits, nbits[d])
		total += nbits[d]
	}
	count := numElements(grid)
	out := make([][]uint64, 0, count)
	decode := func(z uint64) []uint64 {
		c := make([]uint64, nd)
		in := 0
		for b := 0; b < maxBits; b++ {
			for d := 0; d < nd; d++ {
				if b < nbits[d] {
					c[d] |= (z >> in & 1) << b
					in++
				}
			}
		}
		return c
	}
	if total < 63 && uint64(1)<<total <= 4*count {
		for z := uint64(0); z < 1<<total; z++ {
			c := decode(z)
			ok := true
			for d := range c {
				if c[d] >= grid[d] {
					ok = false
					break
				}
			}
			if ok {
				out = append(out, c)
			}
		}
		return out
	}
	// Sparse case: sort all coordinates by their Morton code.
	type item struct {
		code []uint64 // little-endian multiword code
		c    []uint64
	}
	items := make([]item, 0, count)
	words := (total + 63) / 64
	forEachIndex(grid, func(_ uint64, coords []uint64) bool {
		code := make([]uint64, max(words, 1))
		in := 0
		for b := 0; b < maxBits; b++ {
			for d := 0; d < nd; d++ {
				if b < nbits[d] {
					code[in/64] |= (coords[d] >> b & 1) << (in % 64)
					in++
				}
			}
		}
		items = append(items, item{code, slices.Clone(coords)})
		return true
	})
	sort.SliceStable(items, func(a, b int) bool {
		x, y := items[a].code, items[b].code
		for w := len(x) - 1; w >= 0; w-- {
			if x[w] != y[w] {
				return x[w] < y[w]
			}
		}
		return false
	})
	for _, it := range items {
		out = append(out, it.c)
	}
	return out
}

// ---------------------------------------------------------------------------
// Partial reads

// byteSource reads a range of a stored object; ok is false if it is absent.
type byteSource func(ctx context.Context, r ByteRange) (data []byte, ok bool, err error)

func bytesSource(b []byte) byteSource {
	return func(_ context.Context, r ByteRange) ([]byte, bool, error) {
		off, end, err := r.resolve(int64(len(b)))
		if err != nil {
			return nil, false, err
		}
		return b[off:end], true, nil
	}
}

// subSource is the byte range [off, off+n) of src.
func subSource(src byteSource, off, n int64) byteSource {
	return func(ctx context.Context, r ByteRange) ([]byte, bool, error) {
		o, e, err := r.resolve(n)
		if err != nil {
			return nil, false, err
		}
		return src(ctx, ByteRange{Offset: off + o, Length: e - o})
	}
}

// readPartial copies the projected part of a shard into out, reading only
// the index and the inner chunks the projection touches. out is
// pre-filled with the fill value.
func (s *shardingCodec) readPartial(ctx context.Context, src byteSource, spec ChunkSpec, proj projection, out *NDArray, outStrides []uint64, concurrency int) error {
	grid, err := s.gridShape(spec.Shape)
	if err != nil {
		return err
	}
	type innerTask struct {
		linear uint64
		proj   projection
	}
	var tasks []innerTask
	strides := stridesOf(grid)
	proj.split(s.inner, func(coords []uint64, p projection) bool {
		var i uint64
		for d, c := range coords {
			i += c * strides[d]
		}
		tasks = append(tasks, innerTask{i, p})
		return true
	})
	if len(tasks) == 0 {
		return nil
	}
	if uint64(len(tasks)) == numElements(grid) {
		// Every inner chunk is needed: fetch the shard at once.
		data, ok, err := src(ctx, ByteRange{})
		if err != nil || !ok {
			return err
		}
		src = bytesSource(data)
	}
	raw, ok, err := src(ctx, s.indexRange(grid))
	if err != nil || !ok {
		return err
	}
	ix, err := s.decodeIndex(ctx, raw, grid)
	if err != nil {
		return err
	}
	innerSpec := spec.withShape(s.inner)
	nested := s.innerPipe.sharding()
	type need struct {
		proj   projection
		off, n uint64
	}
	var needs []need
	for _, t := range tasks {
		off, n, ok := ix.entry(t.linear)
		if !ok {
			continue
		}
		if off > math.MaxInt64 || n > math.MaxInt64-off {
			return fmt.Errorf("zarr: shard index entry out of range")
		}
		needs = append(needs, need{t.proj, off, n})
	}
	p := newPool(ctx, concurrency)
	decode := func(ctx context.Context, data []byte, proj projection) error {
		c, err := s.innerPipe.decode(ctx, data, innerSpec)
		if err != nil {
			return fmt.Errorf("zarr: decoding inner chunk: %w", err)
		}
		gather(out, outStrides, c, proj)
		return nil
	}
	if nested != nil {
		for _, nd := range needs {
			p.do(func() error {
				return nested.readPartial(p.ctx, subSource(src, int64(nd.off), int64(nd.n)), innerSpec, nd.proj, out, outStrides, 1)
			})
		}
		return p.wait()
	}
	// Coalesce nearby inner chunks into one request, as zarr-python does
	// (gaps up to 1 MiB, requests up to 16 MiB).
	const maxGap, maxSpan = 1 << 20, 16 << 20
	sort.Slice(needs, func(i, j int) bool { return needs[i].off < needs[j].off })
	for i := 0; i < len(needs); {
		j, end := i+1, needs[i].off+needs[i].n
		for j < len(needs) && needs[j].off <= end+maxGap && needs[j].off+needs[j].n-needs[i].off <= maxSpan {
			end = max(end, needs[j].off+needs[j].n)
			j++
		}
		group, start := needs[i:j], needs[i].off
		p.do(func() error {
			data, ok, err := src(p.ctx, ByteRange{Offset: int64(start), Length: int64(end - start)})
			if err != nil || !ok {
				return err
			}
			if uint64(len(data)) != end-start {
				return fmt.Errorf("zarr: inner chunk range [%d, %d) outside the shard", start, end)
			}
			for _, nd := range group {
				if err := decode(p.ctx, data[nd.off-start:nd.off-start+nd.n], nd.proj); err != nil {
					return err
				}
			}
			return nil
		})
		i = j
	}
	return p.wait()
}

// ---------------------------------------------------------------------------
// Partial writes

// writePartial applies a write to a shard: inner chunks the projection
// touches are merged (decoding existing data only when partly covered) and
// re-encoded; all others keep their encoded bytes. existing is the current
// shard (nil if absent). It returns the new shard, or nil if it holds no
// inner chunks.
func (s *shardingCodec) writePartial(ctx context.Context, existing []byte, spec ChunkSpec, proj projection, data *NDArray, dataStrides []uint64) ([]byte, error) {
	grid, err := s.gridShape(spec.Shape)
	if err != nil {
		return nil, err
	}
	n := numElements(grid)
	chunks := make([][]byte, n)
	present := make([]bool, n)
	if existing != nil {
		r, err := s.indexRange(grid).bounds(int64(len(existing)))
		if err != nil {
			return nil, err
		}
		ix, err := s.decodeIndex(ctx, existing[r[0]:r[1]], grid)
		if err != nil {
			return nil, err
		}
		for i := uint64(0); i < n; i++ {
			off, l, ok := ix.entry(i)
			if !ok {
				continue
			}
			if off > uint64(len(existing)) || l > uint64(len(existing))-off {
				return nil, fmt.Errorf("zarr: shard index entry out of range")
			}
			chunks[i], present[i] = existing[off:off+l], true
		}
	}
	innerSpec := spec.withShape(s.inner)
	fill := spec.fill()
	var firstErr error
	strides := stridesOf(grid)
	proj.split(s.inner, func(coords []uint64, p projection) bool {
		var i uint64
		for d, x := range coords {
			i += x * strides[d]
		}
		var c *NDArray
		if p.covers(s.inner) || !present[i] {
			c = newFilled(s.inner, spec.DataType, fill)
		} else if c, err = s.innerPipe.decode(ctx, chunks[i], innerSpec); err != nil {
			firstErr = fmt.Errorf("zarr: decoding inner chunk: %w", err)
			return false
		}
		scatter(c, data, dataStrides, p)
		if !s.writeEmpty && isFill(c, fill) {
			chunks[i], present[i] = nil, false
			return true
		}
		enc, err := s.innerPipe.encode(ctx, c, innerSpec)
		if err != nil {
			firstErr = err
			return false
		}
		chunks[i], present[i] = enc, true
		return true
	})
	if firstErr != nil {
		return nil, firstErr
	}
	return s.assemble(ctx, grid, chunks, present)
}

// fullProjection selects every element of a chunk, in order.
func fullProjection(shape []uint64) projection {
	p := make(orthoProj, len(shape))
	for d, n := range shape {
		p[d] = dimProj{start: 0, step: 1, n: n, out: 0}
	}
	return p
}

// copyBox copies the box of the given shape at srcOff in src to dstOff in dst.
func copyBox(dst *NDArray, dstOff []uint64, src *NDArray, srcOff []uint64, box []uint64) {
	nd := len(box)
	p := make(orthoProj, nd)
	clipped := make([]uint64, nd)
	for d := range box {
		// Clip to both arrays.
		b := box[d]
		if srcOff[d] >= src.Shape[d] || dstOff[d] >= dst.Shape[d] {
			return
		}
		b = min(b, src.Shape[d]-srcOff[d], dst.Shape[d]-dstOff[d])
		clipped[d] = b
		p[d] = dimProj{start: srcOff[d], step: 1, n: b, out: dstOff[d]}
	}
	gather(dst, stridesOf(dst.Shape), src, p)
}
