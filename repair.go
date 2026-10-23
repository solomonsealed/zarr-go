package zarr

import (
	"encoding/json"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Repairs of invalid array metadata that older software stored, read as
// zarr-python (3.4.1 and later, zarr.core.metadata.repair) reads them:
//
//   - A regular chunk size of 0 or JSON false (Zarr v2 "chunks", Zarr v3
//     "regular" chunk_shape), which zarr-python wrote for arrays created
//     with a zero-length axis until 3.4 and from 2.18.7 to 3.2.1 for an
//     explicit chunks=(0,) or False, is read as 1, or the inner chunk size
//     for sharded arrays. Such arrays hold no data.
//   - JSON true, which zarr-python 3.0 and 3.2 wrote for chunks=True, is
//     read as 1: in regular and rectilinear grids and in the inner chunk
//     shape of any sharding codec.
//   - A "regular" chunk_shape mixing chunk sizes with lists of edge lengths
//     ([2, [5, 10, 5]], written by zarr-python 3.2.0 and 3.2.1) is read as
//     the rectilinear grid it describes.
//   - Rectilinear edge lengths stored as integral JSON floats ([[4.0, 2]],
//     as zarr-python 3.2 wrote them) are read as integers.
//
// Anything else invalid is left for the metadata parser to reject. A
// repair that reads a chunk size as another size "moves chunks": chunks
// written under the repaired metadata are not where a reader of the stored
// document looks for them, so the array stores the repaired metadata before
// it writes chunks (see Metadata.RepairPending).

// repairArrayDocument applies the repairs for a Zarr format to a stored
// array document. It returns the repaired document (doc itself if nothing
// changed), whether anything changed, and whether a repair moves chunks.
// doc is not modified.
func repairArrayDocument(doc jsonObject, format int) (out jsonObject, changed, movesChunks bool) {
	out = doc
	var repairs []func(jsonObject) (jsonObject, bool, bool)
	if format == 2 {
		repairs = append(repairs, repairChunkSizesV2)
	} else {
		// The inner chunk shape is read first: it gives the unit of the
		// outer chunk shape. Edge lengths are read last, after any repair
		// that yields a rectilinear chunk grid.
		repairs = append(repairs, repairInnerChunkSizesV3, repairChunkSizesV3, repairEdgeLengthsV3)
	}
	for _, r := range repairs {
		if d, ok, moves := r(out); ok {
			out, changed, movesChunks = d, true, movesChunks || moves
		}
	}
	return out, changed, movesChunks
}

// jsonInt classifies a parsed JSON value as Python's json module reads
// it: an integer (a number without fraction or exponent), or a boolean,
// which Python treats as an integer. sign is the integer's sign.
func jsonInt(v any) (sign int, isBool, ok bool) {
	switch x := v.(type) {
	case bool:
		if x {
			return 1, true, true
		}
		return 0, true, true
	case json.Number:
		s := string(x)
		if strings.ContainsAny(s, ".eE") {
			return 0, false, false
		}
		neg := strings.HasPrefix(s, "-")
		if strings.TrimLeft(strings.TrimPrefix(s, "-"), "0") == "" {
			return 0, false, true
		}
		if neg {
			return -1, false, true
		}
		return 1, false, true
	}
	return 0, false, false
}

// isPyIntList reports whether v is a JSON array of integers (false and
// true count).
func isPyIntList(v any) bool {
	l, ok := v.([]any)
	if !ok {
		return false
	}
	for _, e := range l {
		if _, _, ok := jsonInt(e); !ok {
			return false
		}
	}
	return true
}

var jsonOne = json.Number("1")

// axisSpan is what is known of the length of an axis whose chunk size is
// read: nothing (the inner chunk shape of a sharding codec), or whether it
// is 0.
type axisSpan struct{ known, zero bool }

// readChunkSize reads one entry of a stored regular chunk shape: an
// integer >= 1 is kept, true is read as 1, and 0 or false as unit (the
// smallest chunk edge length the axis can have: 1, or the inner chunk
// size of a shard) where the axis length is known. A flat list is kept as
// the chunk edge lengths of its axis, which only a rectilinear grid can
// declare. ok is false if the entry cannot be read.
func readChunkSize(size any, span axisSpan, unit any) (edge any, moves, ok bool) {
	if size == true {
		return jsonOne, false, true
	}
	if sign, _, isInt := jsonInt(size); isInt {
		switch {
		case sign > 0:
			return size, false, true
		case sign == 0 && span.known:
			return unit, true, true
		}
		return nil, false, false
	}
	if l, isList := size.([]any); isList {
		for _, e := range l {
			if _, nested := e.([]any); nested {
				return nil, false, false
			}
		}
		return size, false, true
	}
	return nil, false, false
}

// readChunkShape reads a stored regular chunk shape entry by entry (see
// readChunkSize) for axes of the given spans, whose chunks are multiples of
// units (1 where not given). changed reports whether any entry was read as
// another value; ok is false if the shape cannot be read.
func readChunkShape(stored any, spans []axisSpan, units []any) (edges []any, changed, moves, ok bool) {
	l, isList := stored.([]any)
	if !isList || len(l) != len(spans) {
		return nil, false, false, false
	}
	edges = make([]any, len(l))
	for i, size := range l {
		var unit any = jsonOne
		if i < len(units) {
			unit = units[i]
		}
		edge, m, ok := readChunkSize(size, spans[i], unit)
		if !ok {
			return nil, false, false, false
		}
		edges[i] = edge
		// Only true and a 0 read as the unit are read as other values.
		changed = changed || size == true || m
		moves = moves || m
	}
	return edges, changed, moves, true
}

func spansOf(shape []any) []axisSpan {
	spans := make([]axisSpan, len(shape))
	for i, n := range shape {
		sign, _, _ := jsonInt(n)
		spans[i] = axisSpan{known: true, zero: sign == 0}
	}
	return spans
}

func repairChunkSizesV2(doc jsonObject) (jsonObject, bool, bool) {
	shape, _ := doc.get("shape")
	if !isPyIntList(shape) {
		return nil, false, false
	}
	chunks, _ := doc.get("chunks")
	edges, changed, moves, ok := readChunkShape(chunks, spansOf(shape.([]any)), nil)
	if !ok || !changed {
		return nil, false, false
	}
	out := slices.Clone(doc)
	out.set("chunks", edges)
	return out, true, moves
}

// readShardingCodec reads the inner chunk shape of a sharding codec, and
// those of the sharding codecs nested in its codecs, as chunk shapes with
// no known axis lengths: only true is read as another value (1), which
// moves no chunks.
func readShardingCodec(codec any) (any, bool) {
	obj, ok := codec.(jsonObject)
	if !ok {
		return nil, false
	}
	if name, _ := obj.get("name"); name != "sharding_indexed" {
		return nil, false
	}
	v, _ := obj.get("configuration")
	cfg, ok := v.(jsonObject)
	if !ok {
		return nil, false
	}
	newCfg := slices.Clone(cfg)
	changed := false
	if stored, ok := cfg.get("chunk_shape"); ok {
		if l, isList := stored.([]any); isList {
			if edges, ch, _, ok := readChunkShape(l, make([]axisSpan, len(l)), nil); ok && ch {
				newCfg.set("chunk_shape", edges)
				changed = true
			}
		}
	}
	if inner, ok := cfg.get("codecs"); ok {
		if l, isList := inner.([]any); isList {
			if codecs, ch := readCodecs(l); ch {
				newCfg.set("codecs", codecs)
				changed = true
			}
		}
	}
	if !changed {
		return nil, false
	}
	out := slices.Clone(obj)
	out.set("configuration", newCfg)
	return out, true
}

func readCodecs(codecs []any) ([]any, bool) {
	var out []any
	for i, c := range codecs {
		if r, ok := readShardingCodec(c); ok {
			if out == nil {
				out = slices.Clone(codecs)
			}
			out[i] = r
		}
	}
	return out, out != nil
}

func repairInnerChunkSizesV3(doc jsonObject) (jsonObject, bool, bool) {
	stored, _ := doc.get("codecs")
	l, ok := stored.([]any)
	if !ok {
		return nil, false, false
	}
	codecs, changed := readCodecs(l)
	if !changed {
		return nil, false, false
	}
	out := slices.Clone(doc)
	out.set("codecs", codecs)
	return out, true, false
}

// innerChunkShape returns the inner chunk shape of the first sharding
// codec of a Zarr v3 document (nil if it has none); ok is false if its
// inner chunk sizes are not all integers of at least 1.
func innerChunkShape(doc jsonObject) (units []any, ok bool) {
	stored, _ := doc.get("codecs")
	codecs, _ := stored.([]any)
	for _, c := range codecs {
		obj, isObj := c.(jsonObject)
		if !isObj {
			continue
		}
		if name, _ := obj.get("name"); name != "sharding_indexed" {
			continue
		}
		v, _ := obj.get("configuration")
		cfg, isObj := v.(jsonObject)
		if !isObj {
			continue
		}
		inner, has := cfg.get("chunk_shape")
		if !has {
			continue
		}
		if !isPyIntList(inner) {
			return nil, false
		}
		for _, e := range inner.([]any) {
			if sign, _, _ := jsonInt(e); sign < 1 {
				return nil, false
			}
		}
		return inner.([]any), true
	}
	return nil, true
}

func repairChunkSizesV3(doc jsonObject) (jsonObject, bool, bool) {
	g, _ := doc.get("chunk_grid")
	grid, ok := g.(jsonObject)
	shape, _ := doc.get("shape")
	if !ok || !isPyIntList(shape) {
		return nil, false, false
	}
	if name, _ := grid.get("name"); name != "regular" {
		return nil, false, false
	}
	units, ok := innerChunkShape(doc)
	if !ok {
		return nil, false, false
	}
	c, _ := grid.get("configuration")
	cfg, ok := c.(jsonObject)
	if !ok {
		return nil, false, false
	}
	stored, _ := cfg.get("chunk_shape")
	edges, changed, moves, ok := readChunkShape(stored, spansOf(shape.([]any)), units)
	if !ok {
		return nil, false, false
	}
	lists := 0
	for _, e := range edges {
		if _, isList := e.([]any); isList {
			lists++
		}
	}
	out := slices.Clone(doc)
	switch {
	case lists == 0:
		if !changed {
			return nil, false, false
		}
		newCfg := slices.Clone(cfg)
		newCfg.set("chunk_shape", edges)
		newGrid := slices.Clone(grid)
		newGrid.set("configuration", newCfg)
		out.set("chunk_grid", newGrid)
		return out, true, moves
	case lists == len(edges):
		// Only a mix of chunk sizes and edge lists was ever stored in a
		// regular grid.
		return nil, false, false
	}
	// Only zarr-python 3.2.x reads the stored grid, so the array stores the
	// rectilinear grid before it writes chunks.
	out.set("chunk_grid", jsonObject{{"name", "rectilinear"}, {"configuration", jsonObject{{"kind", "inline"}, {"chunk_shapes", edges}}}})
	return out, true, true
}

// readEdgeLength reads one stored edge length of a rectilinear grid: true
// as 1, and an integral float of at least 1 (4.0) as the integer it equals.
func readEdgeLength(edge any) (any, bool) {
	if edge == true {
		return jsonOne, true
	}
	n, ok := edge.(json.Number)
	if !ok || !strings.ContainsAny(string(n), ".eE") {
		return edge, false
	}
	f, err := n.Float64()
	// Larger values are no chunk size; the parser refuses them.
	if err != nil || f < 1 || f >= 1<<64 || f != math.Trunc(f) {
		return edge, false
	}
	return json.Number(strconv.FormatUint(uint64(f), 10)), true
}

func repairEdgeLengthsV3(doc jsonObject) (jsonObject, bool, bool) {
	g, _ := doc.get("chunk_grid")
	grid, ok := g.(jsonObject)
	if !ok {
		return nil, false, false
	}
	if name, _ := grid.get("name"); name != "rectilinear" {
		return nil, false, false
	}
	c, _ := grid.get("configuration")
	cfg, ok := c.(jsonObject)
	if !ok {
		return nil, false, false
	}
	s, _ := cfg.get("chunk_shapes")
	stored, ok := s.([]any)
	if !ok {
		return nil, false, false
	}
	changed := false
	read := make([]any, len(stored))
	for d, axis := range stored {
		entries, isList := axis.([]any)
		if !isList {
			read[d] = axis // a bare chunk size is kept as stored
			continue
		}
		out := make([]any, len(entries))
		for i, e := range entries {
			if pair, isPair := e.([]any); isPair && len(pair) == 2 {
				size, ch := readEdgeLength(pair[0])
				out[i] = []any{size, pair[1]}
				changed = changed || ch
				continue
			}
			var ch bool
			out[i], ch = readEdgeLength(e)
			changed = changed || ch
		}
		read[d] = out
	}
	if !changed {
		return nil, false, false
	}
	newCfg := slices.Clone(cfg)
	newCfg.set("chunk_shapes", read)
	newGrid := slices.Clone(grid)
	newGrid.set("configuration", newCfg)
	out := slices.Clone(doc)
	out.set("chunk_grid", newGrid)
	return out, true, false
}
