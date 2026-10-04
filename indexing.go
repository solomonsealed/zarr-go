package zarr

import (
	"fmt"
	"math"
	"slices"
	"sort"
)

// Selections follow zarr-python (and so NumPy) indexing rules:
//
//   - Get/Set take one DimSelector per dimension ("orthogonal indexing",
//     zarr-python's get_orthogonal_selection / oindex, which includes basic
//     indexing): Index drops the dimension, Slice selects a range with a
//     positive step, Indices an arbitrary list of positions (any order,
//     repeats allowed when reading) and Mask a boolean mask of the
//     dimension's length. Missing trailing selectors select everything and
//     Ellipsis stands for as many All() as needed.
//   - GetCoordinates/SetCoordinates select individual points
//     (get_coordinate_selection / vindex).
//   - GetMask/SetMask select the elements where a boolean array of the
//     array's shape is true (get_mask_selection).
//   - GetBlocks/SetBlocks select whole chunks by chunk index
//     (get_block_selection / blocks).
//
// Negative indices count from the end of the dimension.

// DimSelector selects positions along one dimension.
type DimSelector interface{ dimSelector() }

// SliceSelector selects Start, Start+Step, ... up to (excluding) Stop, with
// Python's slice semantics: negative Start/Stop count from the end, and
// both are clamped to the dimension. Step must be positive.
type SliceSelector struct{ Start, Stop, Step int64 }

// IndexSelector selects one position and drops the dimension.
type IndexSelector int64

// IndicesSelector selects a list of positions, in the given order.
type IndicesSelector []int64

// MaskSelector selects the positions where the mask is true; its length
// must equal the dimension's.
type MaskSelector []bool

type ellipsisSelector struct{}

func (SliceSelector) dimSelector()    {}
func (IndexSelector) dimSelector()    {}
func (IndicesSelector) dimSelector()  {}
func (MaskSelector) dimSelector()     {}
func (ellipsisSelector) dimSelector() {}

// End is a Stop value meaning "to the end of the dimension".
const End = math.MaxInt64

// Ellipsis stands for All() on as many dimensions as needed.
var Ellipsis DimSelector = ellipsisSelector{}

// All selects a whole dimension (":").
func All() DimSelector { return SliceSelector{0, End, 1} }

// Slice selects [start, stop) ("start:stop").
func Slice(start, stop int64) DimSelector { return SliceSelector{start, stop, 1} }

// SliceStep selects start, start+step, ... below stop ("start:stop:step").
func SliceStep(start, stop, step int64) DimSelector { return SliceSelector{start, stop, step} }

// Index selects one position and drops the dimension.
func Index(i int64) DimSelector { return IndexSelector(i) }

// Indices selects the given positions in order (an integer array index).
func Indices(idx ...int64) DimSelector { return IndicesSelector(idx) }

// Mask selects the positions where mask is true (a boolean array index).
func Mask(mask ...bool) DimSelector { return MaskSelector(mask) }

// BoundsError reports an index outside an array or chunk grid.
type BoundsError struct {
	Index  int64
	Dim    int
	Length uint64
}

func (e *BoundsError) Error() string {
	return fmt.Sprintf("zarr: index %d out of bounds for dimension %d of length %d", e.Index, e.Dim, e.Length)
}

// ---------------------------------------------------------------------------
// Normalized per-dimension selections

// dimSel is a selection along one dimension, normalized against its length.
type dimSel struct {
	// Strided form (list == nil): positions start + i*step, i < count.
	start, step, count uint64
	// List form: positions in output order.
	list    []uint64
	squeeze bool
}

func (s dimSel) len() uint64 {
	if s.list != nil {
		return uint64(len(s.list))
	}
	return s.count
}

func (s dimSel) at(i uint64) uint64 {
	if s.list != nil {
		return s.list[i]
	}
	return s.start + i*s.step
}

func wrapIndex(i int64, n uint64, dim int) (uint64, error) {
	v := i
	if v < 0 {
		v += int64(n)
	}
	if v < 0 || uint64(v) >= n {
		return 0, &BoundsError{Index: i, Dim: dim, Length: n}
	}
	return uint64(v), nil
}

// normalizeSlice resolves a slice against length n like Python's
// slice.indices for a positive step.
func normalizeSlice(s SliceSelector, n uint64) (dimSel, error) {
	if s.Step < 1 {
		if s.Step == 0 {
			return dimSel{}, fmt.Errorf("zarr: slice step cannot be zero")
		}
		return dimSel{}, fmt.Errorf("%w: negative slice steps (as in zarr-python)", ErrUnsupported)
	}
	clamp := func(v int64) uint64 {
		if v < 0 {
			v += int64(n)
			if v < 0 {
				return 0
			}
		}
		return min(uint64(v), n)
	}
	start, stop := clamp(s.Start), clamp(s.Stop)
	if s.Stop == End {
		stop = n
	}
	var count uint64
	if stop > start {
		count = (stop - start + uint64(s.Step) - 1) / uint64(s.Step)
	}
	return dimSel{start: start, step: uint64(s.Step), count: count}, nil
}

func normalizeDim(sel DimSelector, n uint64, dim int) (dimSel, error) {
	switch s := sel.(type) {
	case SliceSelector:
		return normalizeSlice(s, n)
	case IndexSelector:
		v, err := wrapIndex(int64(s), n, dim)
		return dimSel{start: v, step: 1, count: 1, squeeze: true}, err
	case IndicesSelector:
		list := make([]uint64, len(s))
		for i, x := range s {
			v, err := wrapIndex(x, n, dim)
			if err != nil {
				return dimSel{}, err
			}
			list[i] = v
		}
		return dimSel{list: list}, nil
	case MaskSelector:
		if uint64(len(s)) != n {
			return dimSel{}, fmt.Errorf("zarr: boolean mask of length %d for dimension %d of length %d", len(s), dim, n)
		}
		list := []uint64{}
		for i, b := range s {
			if b {
				list = append(list, uint64(i))
			}
		}
		return dimSel{list: list}, nil
	case nil:
		return normalizeSlice(SliceSelector{0, End, 1}, n)
	}
	return dimSel{}, fmt.Errorf("zarr: unsupported selector %T", sel)
}

// normalizeSelection expands Ellipsis and missing trailing selectors.
func normalizeSelection(sel []DimSelector, shape []uint64) ([]dimSel, error) {
	nd := len(shape)
	expanded := make([]DimSelector, 0, nd)
	ell := -1
	for i, s := range sel {
		if _, ok := s.(ellipsisSelector); ok {
			if ell >= 0 {
				return nil, fmt.Errorf("zarr: more than one Ellipsis in a selection")
			}
			ell = i
		}
	}
	if ell >= 0 {
		fill := nd - (len(sel) - 1)
		if fill < 0 {
			return nil, fmt.Errorf("zarr: selection has %d items for %d dimensions", len(sel)-1, nd)
		}
		expanded = append(expanded, sel[:ell]...)
		for range fill {
			expanded = append(expanded, All())
		}
		expanded = append(expanded, sel[ell+1:]...)
	} else {
		if len(sel) > nd {
			return nil, fmt.Errorf("zarr: selection has %d items for %d dimensions", len(sel), nd)
		}
		expanded = append(expanded, sel...)
		for len(expanded) < nd {
			expanded = append(expanded, All())
		}
	}
	out := make([]dimSel, nd)
	for d, s := range expanded {
		var err error
		if out[d], err = normalizeDim(s, shape[d], d); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Projections: the part of a selection that falls in one chunk

// projection maps selected positions inside a chunk to positions in the
// selection's output.
type projection interface {
	// split divides the projection among the chunks of the given shape
	// it touches (the inner chunks of a shard): fn receives each one's
	// grid coordinates, in C order, and the projection relative to it,
	// until it returns false.
	split(chunk []uint64, fn func(coords []uint64, p projection) bool)
	// covers reports whether writing the projection replaces every
	// position of a chunk of the given shape (zarr-python's
	// is_complete_chunk), so the chunk's previous content is not needed.
	covers(shape []uint64) bool
	// walk visits runs: n elements at chunk linear offsets c, c+cstep, ...
	// and output linear offsets o, o+ostep, ... for a chunk of the given
	// shape and output strides.
	walk(chunkShape, outStrides []uint64, fn func(c, cstep, o, ostep, n uint64))
}

// dimProj maps positions along one dimension of a chunk to output
// positions: either strided (chunk start+i*step -> out+i) or explicit.
type dimProj struct {
	start, step, n, out uint64
	in, outp            []uint64 // explicit form when in != nil
}

func (p dimProj) len() uint64 {
	if p.in != nil {
		return uint64(len(p.in))
	}
	return p.n
}

func (p dimProj) at(i uint64) (c, o uint64) {
	if p.in != nil {
		return p.in[i], p.outp[i]
	}
	return p.start + i*p.step, p.out + i
}

// covers reports whether the projection writes all of [0, size), by
// zarr-python's rule (is_complete_chunk): a step-1 range from 0 to the
// end. Explicit position lists never count, even if they cover everything.
func (p dimProj) covers(size uint64) bool {
	return p.in == nil && p.start == 0 && p.step == 1 && p.n >= size
}

type orthoProj []dimProj

func (p orthoProj) split(chunk []uint64, fn func([]uint64, projection) bool) {
	nd := len(p)
	idx := make([][]uint64, nd)
	projs := make([][]dimProj, nd)
	counts := make([]uint64, nd)
	for d, dp := range p {
		idx[d], projs[d] = dimChunks(regularGrid(chunk), d, dimSel{start: dp.start, step: dp.step, count: dp.n, list: dp.in})
		// dimChunks numbers outputs from 0; map them to dp's.
		for k := range projs[d] {
			q := &projs[d][k]
			if dp.in != nil {
				for j, o := range q.outp {
					q.outp[j] = dp.outp[o]
				}
			} else {
				q.out += dp.out
			}
		}
		counts[d] = uint64(len(idx[d]))
	}
	forEachIndex(counts, func(_ uint64, k []uint64) bool {
		coords := make([]uint64, nd)
		q := make(orthoProj, nd)
		for d := range k {
			coords[d], q[d] = idx[d][k[d]], projs[d][k[d]]
		}
		return fn(coords, q)
	})
}

func (p orthoProj) covers(shape []uint64) bool {
	for d := range p {
		if !p[d].covers(shape[d]) {
			return false
		}
	}
	return true
}

func (p orthoProj) walk(chunkShape, outStrides []uint64, fn func(c, cstep, o, ostep, n uint64)) {
	nd := len(p)
	if nd == 0 {
		fn(0, 1, 0, 1, 1)
		return
	}
	for _, d := range p {
		if d.len() == 0 {
			return
		}
	}
	cst := stridesOf(chunkShape)
	last := p[nd-1]
	idx := make([]uint64, nd-1)
	for {
		var cb, ob uint64
		for d := 0; d < nd-1; d++ {
			c, o := p[d].at(idx[d])
			cb += c * cst[d]
			ob += o * outStrides[d]
		}
		if last.in == nil {
			fn(cb+last.start*cst[nd-1], last.step*cst[nd-1], ob+last.out*outStrides[nd-1], outStrides[nd-1], last.n)
		} else {
			for i := range last.in {
				fn(cb+last.in[i]*cst[nd-1], 1, ob+last.outp[i]*outStrides[nd-1], 1, 1)
			}
		}
		d := nd - 2
		for ; d >= 0; d-- {
			idx[d]++
			if idx[d] < p[d].len() {
				break
			}
			idx[d] = 0
		}
		if d < 0 {
			return
		}
	}
}

// pointProj is a list of points inside a chunk (coordinate selection).
type pointProj struct {
	nd     int
	coords []uint64 // nd coordinates per point, relative to the chunk
	out    []uint64 // linear output position per point
}

func (p *pointProj) split(chunk []uint64, fn func([]uint64, projection) bool) {
	points := make([][]uint64, len(p.out))
	for i := range points {
		points[i] = p.coords[i*p.nd : (i+1)*p.nd]
	}
	for _, t := range pointTasks(regularGrid(chunk), p.nd, points) {
		q := t.proj.(*pointProj)
		for j, o := range q.out {
			q.out[j] = p.out[o]
		}
		if !fn(t.coords, q) {
			return
		}
	}
}

// covers is false: zarr-python never treats point selections as
// covering a chunk.
func (p *pointProj) covers([]uint64) bool { return false }

func (p *pointProj) walk(chunkShape, outStrides []uint64, fn func(c, cstep, o, ostep, n uint64)) {
	st := stridesOf(chunkShape)
	for i := range p.out {
		var lin uint64
		for d := 0; d < p.nd; d++ {
			lin += p.coords[i*p.nd+d] * st[d]
		}
		fn(lin, 1, p.out[i], 1, 1)
	}
}

// gather copies the projected elements of chunk c into out (whose
// strides are outStrides; for point selections out is 1-D).
func gather(out *NDArray, outStrides []uint64, c *NDArray, p projection) {
	es := uint64(out.DataType.Size)
	p.walk(c.Shape, outStrides, func(ci, cstep, oi, ostep, n uint64) {
		if out.Strings != nil {
			for k := uint64(0); k < n; k++ {
				out.Strings[oi+k*ostep] = c.Strings[ci+k*cstep]
			}
			return
		}
		if cstep == 1 && ostep == 1 {
			copy(out.Data[oi*es:(oi+n)*es], c.Data[ci*es:(ci+n)*es])
			return
		}
		for k := uint64(0); k < n; k++ {
			o, s := (oi+k*ostep)*es, (ci+k*cstep)*es
			copy(out.Data[o:o+es], c.Data[s:s+es])
		}
	})
}

// scatter copies data (the selection's values, strides dataStrides) into
// the projected positions of chunk c.
func scatter(c *NDArray, data *NDArray, dataStrides []uint64, p projection) {
	es := uint64(c.DataType.Size)
	p.walk(c.Shape, dataStrides, func(ci, cstep, oi, ostep, n uint64) {
		if c.Strings != nil {
			for k := uint64(0); k < n; k++ {
				c.Strings[ci+k*cstep] = data.Strings[oi+k*ostep]
			}
			return
		}
		if cstep == 1 && ostep == 1 {
			copy(c.Data[ci*es:(ci+n)*es], data.Data[oi*es:(oi+n)*es])
			return
		}
		for k := uint64(0); k < n; k++ {
			d, s := (ci+k*cstep)*es, (oi+k*ostep)*es
			copy(c.Data[d:d+es], data.Data[s:s+es])
		}
	})
}

// ---------------------------------------------------------------------------
// Splitting selections by chunk

// chunkTask is the part of a selection that falls in one chunk.
type chunkTask struct {
	coords []uint64
	proj   projection
}

// dimChunks splits a dimension's selection by chunk: for each chunk index
// touched, the projection of the selection onto it.
func dimChunks(g chunkGrid, dim int, s dimSel) ([]uint64, []dimProj) {
	var idx []uint64
	var projs []dimProj
	n := s.len()
	if n == 0 {
		return nil, nil
	}
	if s.list == nil {
		// Strided: walk chunk by chunk.
		i := uint64(0)
		for i < n {
			pos := s.start + i*s.step
			ci := g.index(dim, pos)
			cstart, csize := g.span(dim, ci)
			cend := cstart + csize
			last := min(n-1, (cend-1-s.start)/s.step)
			idx = append(idx, ci)
			projs = append(projs, dimProj{start: pos - cstart, step: s.step, n: last - i + 1, out: i})
			i = last + 1
		}
		return idx, projs
	}
	// Explicit positions: group by chunk.
	byChunk := map[uint64]*dimProj{}
	for i, pos := range s.list {
		ci := g.index(dim, pos)
		cstart, _ := g.span(dim, ci)
		p, ok := byChunk[ci]
		if !ok {
			p = &dimProj{in: []uint64{}, outp: []uint64{}}
			byChunk[ci] = p
			idx = append(idx, ci)
		}
		p.in = append(p.in, pos-cstart)
		p.outp = append(p.outp, uint64(i))
	}
	slices.Sort(idx)
	for _, ci := range idx {
		projs = append(projs, *byChunk[ci])
	}
	return idx, projs
}

// orthoTasks splits an orthogonal selection into per-chunk tasks.
func orthoTasks(g chunkGrid, sel []dimSel) []chunkTask {
	nd := len(sel)
	idx := make([][]uint64, nd)
	projs := make([][]dimProj, nd)
	for d := range sel {
		idx[d], projs[d] = dimChunks(g, d, sel[d])
		if len(idx[d]) == 0 {
			return nil
		}
	}
	var tasks []chunkTask
	counts := make([]uint64, nd)
	for d := range counts {
		counts[d] = uint64(len(idx[d]))
	}
	forEachIndex(counts, func(_ uint64, k []uint64) bool {
		t := chunkTask{coords: make([]uint64, nd)}
		p := make(orthoProj, nd)
		for d := range k {
			t.coords[d] = idx[d][k[d]]
			p[d] = projs[d][k[d]]
		}
		t.proj = p
		tasks = append(tasks, t)
		return true
	})
	return tasks
}

// pointTasks splits a coordinate selection (points[i] holds the nd
// coordinates of point i) into per-chunk tasks, in chunk order.
func pointTasks(g chunkGrid, nd int, points [][]uint64) []chunkTask {
	type entry struct {
		key    []uint64
		coords []uint64
		out    uint64
	}
	entries := make([]entry, len(points))
	for i, pt := range points {
		e := entry{key: make([]uint64, nd), coords: make([]uint64, nd), out: uint64(i)}
		for d := range pt {
			e.key[d] = g.index(d, pt[d])
			cs, _ := g.span(d, e.key[d])
			e.coords[d] = pt[d] - cs
		}
		entries[i] = e
	}
	sort.SliceStable(entries, func(a, b int) bool { return slices.Compare(entries[a].key, entries[b].key) < 0 })
	var tasks []chunkTask
	for i := 0; i < len(entries); {
		j := i
		p := &pointProj{nd: nd}
		for j < len(entries) && slices.Equal(entries[j].key, entries[i].key) {
			p.coords = append(p.coords, entries[j].coords...)
			p.out = append(p.out, entries[j].out)
			j++
		}
		tasks = append(tasks, chunkTask{coords: entries[i].key, proj: p})
		i = j
	}
	return tasks
}

// forEachIndex visits every multi-index in grid in C order with its linear
// index, stopping early when fn returns false.
func forEachIndex(grid []uint64, fn func(linear uint64, coords []uint64) bool) {
	total := numElements(grid)
	coords := make([]uint64, len(grid))
	for i := uint64(0); i < total; i++ {
		if !fn(i, coords) {
			return
		}
		for d := len(grid) - 1; d >= 0; d-- {
			coords[d]++
			if coords[d] < grid[d] {
				break
			}
			coords[d] = 0
		}
	}
}
