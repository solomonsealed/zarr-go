package zarr

import (
	"context"
	"fmt"
	"slices"
	"sync"
)

// Options tunes array access.
type Options struct {
	// Concurrency bounds parallel chunk reads and writes per call. Zero
	// means 16.
	Concurrency int
	// WriteEmptyChunks stores chunks that hold only the fill value instead
	// of deleting them (zarr-python's write_empty_chunks).
	WriteEmptyChunks bool
}

// Array is a Zarr array (format 2 or 3). It is safe for concurrent use,
// except Resize, Append and SetAttributes, which change the metadata.
// Concurrent writes to the same chunk are not coordinated, and a read of
// part of a shard that is being rewritten (index first, then inner chunks)
// may fail or return inconsistent data.
type Array struct {
	meta        *Metadata
	src         ChunkSource
	pipe        *pipeline
	elem        ChunkSpec // data type and fill value of chunks
	concurrency int
	writeEmpty  bool

	// For arrays opened from a store.
	store Store
	path  string
	mu    sync.Mutex // serializes metadata updates
}

// NewArray opens an array from its metadata document (zarr.json, or a v2
// .zarray) and a source of its chunks. It is the low-level entry point
// for storage systems that address chunks by coordinates, like Icechunk;
// OpenArray reads arrays from a Store.
func NewArray(doc []byte, src ChunkSource, opts *Options) (*Array, error) {
	meta, err := ParseMetadata(doc)
	if err != nil {
		return nil, err
	}
	return NewArrayFromMetadata(meta, src, opts)
}

// NewArrayFromMetadata opens an array from parsed metadata and a chunk source.
func NewArrayFromMetadata(meta *Metadata, src ChunkSource, opts *Options) (*Array, error) {
	a := &Array{meta: meta, src: src, concurrency: 16}
	if opts != nil {
		if opts.Concurrency > 0 {
			a.concurrency = opts.Concurrency
		}
		a.writeEmpty = opts.WriteEmptyChunks
	}
	a.elem = ChunkSpec{Shape: meta.nominalChunkShape(), DataType: meta.DataType, FillValue: meta.fill.ndarray(meta.DataType), bigEndian: meta.BigEndian}
	var err error
	if a.pipe, err = chunkPipeline(meta, a.elem); err != nil {
		return nil, err
	}
	if sc := a.pipe.sharding(); sc != nil {
		sc.setWriteEmpty(a.writeEmpty)
	}
	return a, nil
}

// chunkPipeline builds the codec pipeline of an array's chunks.
func chunkPipeline(meta *Metadata, elem ChunkSpec) (*pipeline, error) {
	if c := meta.largestChunk(); !fitsInMemory(c, meta.DataType) {
		return nil, fmt.Errorf("zarr: chunks of shape %v are too large", c)
	}
	if meta.ZarrFormat != 2 {
		return buildPipeline(meta.Codecs, elem, false)
	}
	var specs []CodecSpec
	if meta.Order == "F" && len(meta.Shape) > 1 {
		order := make([]int, len(meta.Shape))
		for i := range order {
			order[i] = len(order) - 1 - i
		}
		specs = append(specs, Transpose(order))
	}
	specs = append(specs, meta.Filters...)
	if meta.Compressor != nil {
		specs = append(specs, *meta.Compressor)
	}
	return buildPipeline(specs, elem, true)
}

func (s *shardingCodec) setWriteEmpty(v bool) {
	s.writeEmpty = v
	if n := s.innerPipe.sharding(); n != nil {
		n.setWriteEmpty(v)
	}
}

// Metadata returns the parsed metadata.
func (a *Array) Metadata() *Metadata { return a.meta }

// Path returns the array's path in its store ("" for the root or for
// arrays opened with NewArray).
func (a *Array) Path() string { return a.path }

// ZarrFormat returns 2 or 3.
func (a *Array) ZarrFormat() int { return a.meta.ZarrFormat }

// Shape returns the array shape.
func (a *Array) Shape() []uint64 { return slices.Clone(a.meta.Shape) }

// NDim returns the number of dimensions.
func (a *Array) NDim() int { return len(a.meta.Shape) }

// DataType returns the element type.
func (a *Array) DataType() DataType { return a.meta.DataType }

// FillValue returns the fill value as a 0-d array.
func (a *Array) FillValue() *NDArray { return a.meta.FillValueArray() }

// Attributes returns the user attributes.
func (a *Array) Attributes() map[string]any { return a.meta.Attributes }

// DimensionNames returns the dimension names ("" for unnamed dimensions).
func (a *Array) DimensionNames() []string {
	out := make([]string, len(a.meta.Shape))
	for i, n := range a.meta.DimensionNames {
		if n != nil && i < len(out) {
			out[i] = *n
		}
	}
	return out
}

// ChunkGridShape returns the number of chunks (shards, for sharded
// arrays) along each dimension.
func (a *Array) ChunkGridShape() []uint64 { return a.meta.GridShape() }

// Chunks returns the shape of the chunks read and written as a unit: the
// inner chunk shape for sharded arrays (zarr-python's Array.chunks); nil
// for rectilinear grids.
func (a *Array) Chunks() []uint64 {
	if sc := a.pipe.sharding(); sc != nil {
		return slices.Clone(sc.inner)
	}
	return slices.Clone(a.meta.ChunkShape)
}

// Shards returns the shard shape of a sharded array, else nil.
func (a *Array) Shards() []uint64 {
	if a.pipe.sharding() != nil {
		return slices.Clone(a.meta.ChunkShape)
	}
	return nil
}

func (a *Array) chunkSpec(coords []uint64) (start []uint64, spec ChunkSpec) {
	start, shape := a.meta.ChunkBounds(coords)
	return start, a.elem.withShape(shape)
}

func (a *Array) chunkBytes(coords []uint64) byteSource {
	c := slices.Clone(coords)
	return func(ctx context.Context, r ByteRange) ([]byte, bool, error) {
		return a.src.GetChunk(ctx, c, r)
	}
}

func (a *Array) checkCoords(coords []uint64) error {
	if len(coords) != len(a.meta.Shape) {
		return fmt.Errorf("zarr: chunk coordinates %v do not match %d dimensions", coords, len(a.meta.Shape))
	}
	for d, c := range coords {
		if c >= a.meta.grid.numChunks(d, a.meta.Shape[d]) {
			return fmt.Errorf("zarr: chunk coordinates %v out of range", coords)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Reading

// Get reads an orthogonal selection, one selector per dimension (fewer
// selectors select all of the remaining dimensions). Index selectors drop
// their dimension, so selecting a single element returns a 0-d array.
func (a *Array) Get(ctx context.Context, sel ...DimSelector) (*NDArray, error) {
	ds, err := normalizeSelection(sel, a.meta.Shape)
	if err != nil {
		return nil, err
	}
	shape := make([]uint64, len(ds))
	var squeezed []uint64
	for d, s := range ds {
		shape[d] = s.len()
		if !s.squeeze {
			squeezed = append(squeezed, s.len())
		}
	}
	if !fitsInMemory(shape, a.meta.DataType) {
		return nil, fmt.Errorf("zarr: selection of shape %v is too large", squeezed)
	}
	out := newFilled(shape, a.meta.DataType, a.meta.fill)
	if err := a.readTasks(ctx, orthoTasks(a.meta.grid, ds), out, stridesOf(shape)); err != nil {
		return nil, err
	}
	if squeezed == nil {
		squeezed = []uint64{}
	}
	out.Shape = squeezed
	return out, nil
}

// SelectionShape returns the shape Get would return for sel, without
// reading anything (e.g. to bound the size of a request).
func (a *Array) SelectionShape(sel ...DimSelector) ([]uint64, error) {
	ds, err := normalizeSelection(sel, a.meta.Shape)
	if err != nil {
		return nil, err
	}
	shape := []uint64{}
	for _, s := range ds {
		if !s.squeeze {
			shape = append(shape, s.len())
		}
	}
	return shape, nil
}

// Read reads the region starting at start with extent count.
func (a *Array) Read(ctx context.Context, start, count []uint64) (*NDArray, error) {
	sel, err := a.regionSelection(start, count)
	if err != nil {
		return nil, err
	}
	return a.Get(ctx, sel...)
}

// ReadAll reads the whole array.
func (a *Array) ReadAll(ctx context.Context) (*NDArray, error) { return a.Get(ctx) }

func (a *Array) regionSelection(start, count []uint64) ([]DimSelector, error) {
	nd := len(a.meta.Shape)
	if len(start) != nd || len(count) != nd {
		return nil, fmt.Errorf("zarr: region has %d/%d dimensions, array has %d", len(start), len(count), nd)
	}
	sel := make([]DimSelector, nd)
	for d := range start {
		if start[d]+count[d] > a.meta.Shape[d] || start[d]+count[d] < start[d] {
			return nil, fmt.Errorf("zarr: region [%d, %d) out of bounds for dimension %d of length %d", start[d], start[d]+count[d], d, a.meta.Shape[d])
		}
		sel[d] = SliceSelector{int64(start[d]), int64(start[d] + count[d]), 1}
	}
	return sel, nil
}

// GetCoordinates reads individual points (zarr-python's
// get_coordinate_selection / vindex): coords holds one index list per
// dimension, all of the same length (a list of length 1 is broadcast).
// The result is 1-D, in the order of the points.
func (a *Array) GetCoordinates(ctx context.Context, coords ...[]int64) (*NDArray, error) {
	points, err := a.points(coords)
	if err != nil {
		return nil, err
	}
	out := newFilled([]uint64{uint64(len(points))}, a.meta.DataType, a.meta.fill)
	if err := a.readTasks(ctx, pointTasks(a.meta.grid, a.NDim(), points), out, []uint64{1}); err != nil {
		return nil, err
	}
	return out, nil
}

// GetMask reads the elements where mask (a bool array of the array's
// shape) is true, in C order (zarr-python's get_mask_selection).
func (a *Array) GetMask(ctx context.Context, mask *NDArray) (*NDArray, error) {
	coords, err := a.maskCoords(mask)
	if err != nil {
		return nil, err
	}
	return a.GetCoordinates(ctx, coords...)
}

// GetBlocks reads whole chunks (shards, for sharded arrays) selected by
// chunk index: Index or Slice (step 1) per dimension (zarr-python's
// get_block_selection). No dimension is dropped.
func (a *Array) GetBlocks(ctx context.Context, sel ...DimSelector) (*NDArray, error) {
	s, err := a.blockSelection(sel)
	if err != nil {
		return nil, err
	}
	return a.Get(ctx, s...)
}

// ReadChunk decodes one complete chunk (shard, for sharded arrays),
// including any part beyond the array's edge. Missing chunks are returned
// filled with the fill value.
func (a *Array) ReadChunk(ctx context.Context, coords []uint64) (*NDArray, error) {
	if err := a.checkCoords(coords); err != nil {
		return nil, err
	}
	_, spec := a.chunkSpec(coords)
	var c *NDArray
	// In a pool, like other reads, so that a panic becomes an error.
	p := newPool(ctx, 1)
	p.do(func() error {
		data, ok, err := a.src.GetChunk(p.ctx, coords, ByteRange{})
		if err != nil {
			return err
		}
		if !ok {
			c = newFilled(spec.Shape, a.meta.DataType, a.meta.fill)
			return nil
		}
		if c, err = a.pipe.decode(p.ctx, data, spec); err != nil {
			return fmt.Errorf("zarr: decoding chunk %v: %w", coords, err)
		}
		return nil
	})
	if err := p.wait(); err != nil {
		return nil, err
	}
	return c, nil
}

// readTasks reads the selected parts of chunks into out.
func (a *Array) readTasks(ctx context.Context, tasks []chunkTask, out *NDArray, outStrides []uint64) error {
	if len(tasks) == 0 {
		return nil
	}
	p := newPool(ctx, a.concurrency)
	sc := a.pipe.sharding()
	for _, t := range tasks {
		p.do(func() error {
			_, spec := a.chunkSpec(t.coords)
			if sc != nil {
				// Separate pool inside: waiting on the outer one from a
				// task could deadlock.
				if err := sc.readPartial(p.ctx, a.chunkBytes(t.coords), spec, t.proj, out, outStrides, a.concurrency); err != nil {
					return fmt.Errorf("zarr: reading shard %v: %w", t.coords, err)
				}
				return nil
			}
			data, ok, err := a.src.GetChunk(p.ctx, t.coords, ByteRange{})
			if err != nil || !ok {
				return err
			}
			c, err := a.pipe.decode(p.ctx, data, spec)
			if err != nil {
				return fmt.Errorf("zarr: decoding chunk %v: %w", t.coords, err)
			}
			gather(out, outStrides, c, t.proj)
			return nil
		})
	}
	return p.wait()
}

// points validates coordinate lists and transposes them into points.
func (a *Array) points(coords [][]int64) ([][]uint64, error) {
	nd := a.NDim()
	if len(coords) != nd {
		return nil, fmt.Errorf("zarr: coordinate selection has %d index lists for %d dimensions", len(coords), nd)
	}
	n := 1
	if nd == 0 {
		return nil, fmt.Errorf("zarr: coordinate selection on a 0-d array")
	}
	for _, c := range coords {
		if len(c) != 1 {
			if n != 1 && len(c) != n {
				return nil, fmt.Errorf("zarr: coordinate lists of lengths %d and %d cannot be broadcast", n, len(c))
			}
			n = len(c)
		}
	}
	if len(coords[0]) == 0 {
		n = 0
	}
	for _, c := range coords {
		if len(c) == 0 {
			n = 0
		}
	}
	points := make([][]uint64, n)
	for i := range points {
		pt := make([]uint64, nd)
		for d, c := range coords {
			v := c[0]
			if len(c) > 1 {
				v = c[i]
			}
			var err error
			if pt[d], err = wrapIndex(v, a.meta.Shape[d], d); err != nil {
				return nil, err
			}
		}
		points[i] = pt
	}
	return points, nil
}

func (a *Array) maskCoords(mask *NDArray) ([][]int64, error) {
	if mask.DataType.Kind != KindBool || !slices.Equal(mask.Shape, a.meta.Shape) {
		return nil, fmt.Errorf("zarr: mask must be a bool array of shape %v", a.meta.Shape)
	}
	if err := mask.validate(); err != nil {
		return nil, err
	}
	nd := a.NDim()
	coords := make([][]int64, nd)
	for d := range coords {
		coords[d] = []int64{}
	}
	forEachIndex(a.meta.Shape, func(i uint64, idx []uint64) bool {
		if mask.Data[i] != 0 {
			for d := range idx {
				coords[d] = append(coords[d], int64(idx[d]))
			}
		}
		return true
	})
	if nd > 0 && len(coords[0]) == 0 {
		return coords, nil
	}
	return coords, nil
}

// blockSelection converts a selection over the chunk grid to one over
// the array, with zarr-python's rules (BlockIndexer): block indices wrap
// once, slices become element ranges with Python slice semantics, and a
// range must start inside the array.
func (a *Array) blockSelection(sel []DimSelector) ([]DimSelector, error) {
	grid := a.meta.GridShape()
	nd := len(grid)
	var expanded []DimSelector
	ell := false
	for i, s := range sel {
		if _, ok := s.(ellipsisSelector); ok {
			if ell {
				return nil, fmt.Errorf("zarr: more than one Ellipsis in a selection")
			}
			ell = true
			fill := nd - (len(sel) - 1)
			if fill < 0 {
				return nil, fmt.Errorf("zarr: block selection has too many items")
			}
			expanded = append(expanded, sel[:i]...)
			for range fill {
				expanded = append(expanded, All())
			}
			expanded = append(expanded, sel[i+1:]...)
		}
	}
	if !ell {
		expanded = slices.Clone(sel)
	}
	if len(expanded) > nd {
		return nil, fmt.Errorf("zarr: block selection has %d items for %d dimensions", len(expanded), nd)
	}
	out := make([]DimSelector, nd)
	for d := 0; d < nd; d++ {
		nc, dimLen := int64(grid[d]), int64(a.meta.Shape[d])
		var s DimSelector = All()
		if d < len(expanded) {
			s = expanded[d]
		}
		var start, stop int64
		switch x := s.(type) {
		case IndexSelector:
			i := int64(x)
			if i < 0 {
				i += nc
			}
			if i < 0 || i >= nc {
				return nil, fmt.Errorf("zarr: block index %d out of bounds for dimension %d with %d chunk(s)", int64(x), d, nc)
			}
			cs, size := a.meta.grid.span(d, uint64(i))
			start, stop = int64(cs), int64(cs+size)
		case SliceSelector:
			if x.Step != 1 {
				return nil, fmt.Errorf("zarr: block selections only support step 1")
			}
			lo, hi := x.Start, x.Stop
			if hi == End {
				hi = nc
			}
			if lo < 0 {
				lo += nc
			}
			if hi < 0 {
				hi += nc
			}
			start, stop = a.blockOffset(d, lo, nc, dimLen), a.blockOffset(d, hi, nc, dimLen)
		default:
			return nil, fmt.Errorf("zarr: block selections take Index or Slice selectors, not %T", s)
		}
		if start >= dimLen || start < 0 {
			return nil, &BoundsError{Index: start, Dim: d, Length: uint64(dimLen)}
		}
		out[d] = SliceSelector{start, stop, 1}
	}
	return out, nil
}

// blockOffset is the element offset of block i (zarr-python's
// chunk_offset, used even for out-of-range or negative i).
func (a *Array) blockOffset(d int, i, nc, dimLen int64) int64 {
	if i >= nc {
		return dimLen
	}
	switch g := a.meta.grid.(type) {
	case regularGrid:
		return i * int64(g[d])
	case *rectilinearGrid:
		if g.step[d] > 0 {
			return i * int64(g.step[d])
		}
		// zarr-python treats identical edges as a fixed chunk size.
		if e, ok := g.uniformEdge(d, uint64(dimLen)); ok {
			return i * int64(e)
		}
		if i <= 0 {
			return 0
		}
		return int64(g.starts[d][i])
	}
	return 0
}

// ---------------------------------------------------------------------------
// Bounded parallelism

// pool runs tasks with bounded concurrency and remembers the first error.
// The first failure cancels the pool's context, stopping pending tasks.
type pool struct {
	ctx    context.Context
	cancel context.CancelFunc
	sem    chan struct{}
	wg     sync.WaitGroup
	once   sync.Once
	err    error
}

func newPool(ctx context.Context, n int) *pool {
	ctx, cancel := context.WithCancel(ctx)
	return &pool{ctx: ctx, cancel: cancel, sem: make(chan struct{}, max(n, 1))}
}

func (p *pool) do(task func() error) {
	select {
	case p.sem <- struct{}{}:
	case <-p.ctx.Done():
		p.fail(p.ctx.Err())
		return
	}
	p.wg.Add(1)
	go func() {
		defer func() { <-p.sem; p.wg.Done() }()
		// A panic here would take down the whole program (in a Worker: the
		// isolate serving every in-flight request).
		defer func() {
			if r := recover(); r != nil {
				p.fail(fmt.Errorf("zarr: panic while processing a chunk: %v", r))
			}
		}()
		if p.ctx.Err() != nil {
			return
		}
		if err := task(); err != nil {
			p.fail(err)
		}
	}()
}

func (p *pool) fail(err error) {
	p.once.Do(func() {
		p.err = err
		p.cancel()
	})
}

func (p *pool) wait() error {
	p.wg.Wait()
	p.cancel()
	return p.err
}
