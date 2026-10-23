package zarr

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
)

// Set writes data to an orthogonal selection (see Get). data must have
// the selection's shape (without the dimensions Index selectors drop), or
// a shape that broadcasts to it, such as a single element. Chunks only
// partly covered are read, updated and rewritten; chunks that end up
// holding only the fill value are deleted unless Options.WriteEmptyChunks
// is set, as zarr-python does. For sharded arrays only the inner chunks
// the selection touches are decoded and re-encoded.
func (a *Array) Set(ctx context.Context, data *NDArray, sel ...DimSelector) error {
	ds, err := normalizeSelection(sel, a.meta.Shape)
	if err != nil {
		return err
	}
	full := make([]uint64, len(ds))
	var squeezed []uint64
	for d, s := range ds {
		full[d] = s.len()
		if !s.squeeze {
			squeezed = append(squeezed, s.len())
		}
	}
	vals, err := a.prepareData(data, squeezed, full)
	if err != nil {
		return err
	}
	return a.writeTasks(ctx, orthoTasks(a.meta.grid, ds), vals, stridesOf(full))
}

// Write stores data at the region starting at start (data.Shape is the
// region's extent).
func (a *Array) Write(ctx context.Context, start []uint64, data *NDArray) error {
	sel, err := a.regionSelection(start, data.Shape)
	if err != nil {
		return err
	}
	return a.Set(ctx, data, sel...)
}

// SetCoordinates writes individual points (see GetCoordinates). data is
// 1-D with one value per point, or a single value. With repeated points
// the last value wins.
func (a *Array) SetCoordinates(ctx context.Context, data *NDArray, coords ...[]int64) error {
	points, err := a.points(coords)
	if err != nil {
		return err
	}
	n := uint64(len(points))
	vals, err := a.prepareData(data, []uint64{n}, []uint64{n})
	if err != nil {
		return err
	}
	return a.writeTasks(ctx, pointTasks(a.meta.grid, a.NDim(), points), vals, []uint64{1})
}

// SetMask writes the elements where mask is true (see GetMask).
func (a *Array) SetMask(ctx context.Context, data *NDArray, mask *NDArray) error {
	coords, err := a.maskCoords(mask)
	if err != nil {
		return err
	}
	return a.SetCoordinates(ctx, data, coords...)
}

// SetBlocks writes whole chunks selected by chunk index (see GetBlocks).
func (a *Array) SetBlocks(ctx context.Context, data *NDArray, sel ...DimSelector) error {
	s, err := a.blockSelection(sel)
	if err != nil {
		return err
	}
	return a.Set(ctx, data, s...)
}

// prepareData checks data against the selection (shape: the result shape
// of the selection, full: with dropped dimensions as 1) and broadcasts it
// to full.
func (a *Array) prepareData(data *NDArray, shape, full []uint64) (*NDArray, error) {
	if data == nil {
		return nil, fmt.Errorf("zarr: no data to write")
	}
	if !data.DataType.Equal(a.meta.DataType) {
		return nil, fmt.Errorf("zarr: writing %s data to a %s array", data.DataType.Name, a.meta.DataType.Name)
	}
	if err := data.validate(); err != nil {
		return nil, err
	}
	if shape == nil {
		shape = []uint64{}
	}
	if slices.Equal(data.Shape, shape) || slices.Equal(data.Shape, full) {
		return &NDArray{Shape: full, DataType: data.DataType, Data: data.Data, Strings: data.Strings}, nil
	}
	if !fitsInMemory(shape, data.DataType) {
		return nil, fmt.Errorf("zarr: selection of shape %v is too large", shape)
	}
	b, err := broadcastTo(data, shape)
	if err != nil {
		return nil, err
	}
	b.Shape = full
	return b, nil
}

// broadcastTo expands data to shape with NumPy's broadcasting rules.
func broadcastTo(data *NDArray, shape []uint64) (*NDArray, error) {
	if len(data.Shape) > len(shape) {
		return nil, fmt.Errorf("zarr: data of shape %v does not match selection of shape %v", data.Shape, shape)
	}
	off := len(shape) - len(data.Shape)
	src := stridesOf(data.Shape)
	strides := make([]uint64, len(shape)) // 0 for broadcast dimensions
	for d := range data.Shape {
		switch data.Shape[d] {
		case shape[off+d]:
			strides[off+d] = src[d]
		case 1:
		default:
			return nil, fmt.Errorf("zarr: data of shape %v cannot be broadcast to selection of shape %v", data.Shape, shape)
		}
	}
	out := newFilled(shape, data.DataType, fillValue{})
	es := uint64(data.DataType.Size)
	forEachIndex(shape, func(i uint64, idx []uint64) bool {
		var s uint64
		for d, x := range idx {
			s += x * strides[d]
		}
		if out.Strings != nil {
			out.Strings[i] = data.Strings[s]
		} else {
			copy(out.Data[i*es:(i+1)*es], data.Data[s*es:(s+1)*es])
		}
		return true
	})
	return out, nil
}

// writeTasks applies a write chunk by chunk.
func (a *Array) writeTasks(ctx context.Context, tasks []chunkTask, data *NDArray, dataStrides []uint64) error {
	w, ok := a.src.(ChunkWriter)
	if !ok {
		return fmt.Errorf("%w: array chunks are not writable", ErrReadOnly)
	}
	if len(tasks) == 0 {
		return nil
	}
	if err := a.storeRepair(ctx); err != nil {
		return err
	}
	p := newPool(ctx, a.concurrency)
	for _, t := range tasks {
		p.do(func() error { return a.writeChunk(p.ctx, w, t, data, dataStrides) })
	}
	return p.wait()
}

// storeRepair stores the repaired metadata of an array read from a stored
// document whose repair moves chunks (see Metadata.RepairPending), before
// chunks are written under it, as zarr-python does. The stored document is
// read again, since another writer may have replaced it: if that one lays
// out chunks differently, the write fails and nothing is stored; if it
// needs no repair, it is kept.
func (a *Array) storeRepair(ctx context.Context) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.meta.stored == nil {
		return nil
	}
	ws, ok := a.store.(WritableStore)
	if !ok {
		return nil // NewArray: the caller stores the metadata
	}
	names := []string{"zarr.json"}
	if a.meta.ZarrFormat == 2 {
		names = []string{".zarray", ".zattrs"}
	}
	stored := map[string][]byte{}
	for _, n := range names {
		b, ok, err := a.store.Get(ctx, joinKey(a.path, n), ByteRange{})
		if err != nil {
			return err
		}
		if ok {
			stored[n] = b
		}
	}
	if doc, ok := stored[names[0]]; ok {
		var cur *Metadata
		var err error
		if a.meta.ZarrFormat == 2 {
			cur, err = ParseMetadataV2(doc, stored[".zattrs"])
		} else {
			cur, err = parseV3Metadata(doc)
		}
		if err != nil {
			return fmt.Errorf("zarr: array %q: %w", a.path, err)
		}
		if chunkLayout(cur) != chunkLayout(a.meta) {
			return fmt.Errorf("zarr: array %q: the metadata stored has changed since the array was opened; reopen the array to write to it (nothing was stored)", a.path)
		}
		if cur.stored != nil {
			docs, err := cur.documents()
			if err != nil {
				return err
			}
			for n, b := range docs {
				if old, ok := stored[n]; ok && sameJSON(old, b) {
					delete(docs, n)
				}
			}
			if err := setDocs(ctx, ws, a.path, docs); err != nil {
				return err
			}
		}
	}
	a.meta.stored = nil
	return nil
}

// chunkLayout describes how an array's chunks are laid out: its chunk grid
// and, if sharded, the inner chunk shape.
func chunkLayout(m *Metadata) string {
	g, _ := marshalPython(m.grid.json(), -1)
	for _, c := range m.Codecs {
		if c.Name == "sharding_indexed" {
			var cfg struct {
				ChunkShape json.RawMessage `json:"chunk_shape"`
			}
			json.Unmarshal(c.Configuration, &cfg)
			return string(g) + " " + string(cfg.ChunkShape)
		}
	}
	return string(g)
}

// sameJSON reports whether two JSON documents hold the same value.
func sameJSON(a, b []byte) bool {
	var x, y any
	if json.Unmarshal(pyCompat(a), &x) != nil || json.Unmarshal(pyCompat(b), &y) != nil {
		return false
	}
	return reflect.DeepEqual(x, y)
}

func (a *Array) writeChunk(ctx context.Context, w ChunkWriter, t chunkTask, data *NDArray, dataStrides []uint64) error {
	cStart, spec := a.chunkSpec(t.coords)
	if sc := a.pipe.sharding(); sc != nil {
		// A write covering every inner chunk needs no read (edge shards
		// never qualify: their out-of-bounds inner chunks are not covered).
		var existing []byte
		if !t.proj.covers(spec.Shape) {
			raw, ok, err := w.GetChunk(ctx, t.coords, ByteRange{})
			if err != nil {
				return err
			}
			if ok {
				existing = raw
				if existing == nil {
					existing = []byte{} // stored but empty: truncated, not missing
				}
			}
		}
		shard, err := sc.writePartial(ctx, existing, spec, t.proj, data, dataStrides)
		if err != nil {
			return fmt.Errorf("zarr: writing shard %v: %w", t.coords, err)
		}
		if shard == nil {
			return w.DeleteChunk(ctx, t.coords)
		}
		return w.SetChunk(ctx, t.coords, shard)
	}
	// Covered means the write replaces the chunk's in-bounds part.
	inBounds := make([]uint64, len(cStart))
	for d := range cStart {
		inBounds[d] = min(spec.Shape[d], a.meta.Shape[d]-cStart[d])
	}
	var chunk *NDArray
	if t.proj.covers(inBounds) {
		chunk = newFilled(spec.Shape, a.meta.DataType, a.meta.fill)
	} else {
		raw, ok, err := w.GetChunk(ctx, t.coords, ByteRange{})
		if err != nil {
			return err
		}
		if !ok {
			chunk = newFilled(spec.Shape, a.meta.DataType, a.meta.fill)
		} else if chunk, err = a.pipe.decode(ctx, raw, spec); err != nil {
			return fmt.Errorf("zarr: decoding chunk %v: %w", t.coords, err)
		}
	}
	scatter(chunk, data, dataStrides, t.proj)
	if !a.writeEmpty && isFill(chunk, a.meta.fill) {
		// Delete even a chunk that was never written, as zarr-python does
		// (versioned stores like Icechunk record the deletion).
		return w.DeleteChunk(ctx, t.coords)
	}
	enc, err := a.pipe.encode(ctx, chunk, spec)
	if err != nil {
		return fmt.Errorf("zarr: encoding chunk %v: %w", t.coords, err)
	}
	return w.SetChunk(ctx, t.coords, enc)
}

// isFill reports whether every element equals the fill value, by
// zarr-python's rules: a float fill of ±0 compares bit patterns (so -0.0
// differs from 0.0), other float and complex fills compare numerically
// with every NaN equal to a NaN fill, structs compare field values with
// NaN never equal, and everything else compares bytes.
func isFill(n *NDArray, fill fillValue) bool {
	if n.Strings != nil || n.DataType.Variable() {
		for _, s := range n.Strings {
			if s != fill.str {
				return false
			}
		}
		return true
	}
	es := len(fill.fixed)
	if es == 0 {
		return true
	}
	eq := elementEqual(n.DataType, fill.fixed)
	for i := 0; i+es <= len(n.Data); i += es {
		if !eq(n.Data[i : i+es]) {
			return false
		}
	}
	return true
}

// elementEqual returns a predicate comparing one element to fill.
func elementEqual(dt DataType, fill []byte) func([]byte) bool {
	switch dt.Kind {
	case KindFloat:
		f := floatAt(dt, fill)
		if f == 0 {
			return func(b []byte) bool { return bytes.Equal(b, fill) }
		}
		return func(b []byte) bool {
			if bytes.Equal(b, fill) {
				return true
			}
			v := floatAt(dt, b)
			return v == f || (math.IsNaN(v) && math.IsNaN(f))
		}
	case KindComplex:
		h := dt.Size / 2
		fr, fi := floatAt(dt, fill), floatAt(dt, fill[h:])
		fnan := math.IsNaN(fr) || math.IsNaN(fi)
		return func(b []byte) bool {
			if bytes.Equal(b, fill) {
				return true
			}
			r, i := floatAt(dt, b), floatAt(dt, b[h:])
			if math.IsNaN(r) || math.IsNaN(i) {
				return fnan
			}
			return r == fr && i == fi
		}
	case KindStruct:
		eqs := make([]func([]byte) bool, len(dt.Fields))
		for k, f := range dt.Fields {
			eqs[k] = structFieldEqual(f.Type, fill[f.Offset:f.Offset+f.Type.Size])
		}
		return func(b []byte) bool {
			for k, f := range dt.Fields {
				if !eqs[k](b[f.Offset : f.Offset+f.Type.Size]) {
					return false
				}
			}
			return true
		}
	}
	return func(b []byte) bool { return bytes.Equal(b, fill) }
}

// structFieldEqual compares struct fields as numpy's structured equality
// does: numerically, NaN never equal.
func structFieldEqual(dt DataType, fill []byte) func([]byte) bool {
	switch dt.Kind {
	case KindFloat:
		f := floatAt(dt, fill)
		return func(b []byte) bool { return floatAt(dt, b) == f }
	case KindComplex:
		h := dt.Size / 2
		fr, fi := floatAt(dt, fill), floatAt(dt, fill[h:])
		return func(b []byte) bool { return floatAt(dt, b) == fr && floatAt(dt, b[h:]) == fi }
	case KindStruct:
		return elementEqual(dt, fill)
	}
	return func(b []byte) bool { return bytes.Equal(b, fill) }
}

// ---------------------------------------------------------------------------
// Resizing

// Resize changes the array's shape. Chunks that lie wholly outside the
// new shape are deleted; chunks straddling the new edge keep their data,
// as in zarr-python. The array must have been opened from a writable
// store (for arrays from NewArray, update the metadata yourself).
func (a *Array) Resize(ctx context.Context, shape []uint64) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(shape) != a.NDim() {
		return fmt.Errorf("zarr: cannot resize %d dimensions to %v", a.NDim(), shape)
	}
	ws, ok := a.store.(WritableStore)
	if !ok {
		return fmt.Errorf("%w: resizing needs a writable store", ErrReadOnly)
	}
	w, _ := a.src.(ChunkWriter)
	oldGrid := a.meta.GridShape()
	newMeta := *a.meta
	newMeta.doc = slices.Clone(a.meta.doc)
	newMeta.setShape(shape)
	newGrid := newMeta.GridShape()
	// Encoded before any chunk is deleted, as zarr-python does: metadata
	// that cannot be stored then fails with the store untouched.
	docs, err := newMeta.documents()
	if err != nil {
		return err
	}
	shrinks := false
	for d := range oldGrid {
		if newGrid[d] < oldGrid[d] {
			shrinks = true
		}
	}
	if shrinks && w != nil {
		p := newPool(ctx, a.concurrency)
		forEachIndex(oldGrid, func(_ uint64, c []uint64) bool {
			for d := range c {
				if c[d] >= newGrid[d] {
					coords := slices.Clone(c)
					p.do(func() error { return w.DeleteChunk(p.ctx, coords) })
					break
				}
			}
			return true
		})
		if err := p.wait(); err != nil {
			return err
		}
	}
	if err := setDocs(ctx, ws, a.path, docs); err != nil {
		return err
	}
	newMeta.stored = nil // the repair, if any, is stored
	*a.meta = newMeta
	return nil
}

// Append grows the array along axis and writes data at the end, as
// zarr-python's Array.append does.
func (a *Array) Append(ctx context.Context, data *NDArray, axis int) error {
	shape := a.Shape()
	if axis < 0 {
		axis += len(shape)
	}
	if axis < 0 || axis >= len(shape) || len(data.Shape) != len(shape) {
		return fmt.Errorf("zarr: cannot append data of shape %v along axis %d to shape %v", data.Shape, axis, shape)
	}
	for d := range shape {
		if d != axis && data.Shape[d] != shape[d] {
			return fmt.Errorf("zarr: shape of data to append %v is not compatible with the array's %v", data.Shape, shape)
		}
	}
	start := make([]uint64, len(shape))
	start[axis] = shape[axis]
	shape[axis] += data.Shape[axis]
	if err := a.Resize(ctx, shape); err != nil {
		return err
	}
	return a.Write(ctx, start, data)
}
