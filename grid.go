package zarr

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// chunkGrid maps array positions to chunks along each dimension.
type chunkGrid interface {
	// numChunks returns the number of chunks along dim for an array of length n.
	numChunks(dim int, n uint64) uint64
	// span returns the start position and length of chunk idx along dim.
	span(dim int, idx uint64) (start, size uint64)
	// index returns the chunk containing position pos along dim.
	index(dim int, pos uint64) uint64
	// json renders the chunk_grid entry of zarr.json.
	json() any
	// resized returns the grid for a new array shape.
	resized(old, shape []uint64) chunkGrid
}

type regularGrid []uint64

func (g regularGrid) numChunks(dim int, n uint64) uint64 {
	return (n + g[dim] - 1) / g[dim]
}
func (g regularGrid) span(dim int, idx uint64) (uint64, uint64) { return idx * g[dim], g[dim] }
func (g regularGrid) index(dim int, pos uint64) uint64          { return pos / g[dim] }
func (g regularGrid) json() any {
	return jsonObject{{"name", "regular"}, {"configuration", jsonObject{{"chunk_shape", []uint64(g)}}}}
}
func (g regularGrid) resized(_, _ []uint64) chunkGrid { return g }

// rectilinearGrid has explicit chunk edges per dimension; a dimension with
// no explicit edges repeats a regular step.
type rectilinearGrid struct {
	step   []uint64   // >0 for regular-step dimensions
	starts [][]uint64 // cumulative starts (len = nchunks+1) for explicit dims
}

// newRectilinearGrid builds a grid from explicit edge lengths per
// dimension; dimensions without edges repeat the regular step steps[d].
func newRectilinearGrid(steps []uint64, chunks [][]uint64) (*rectilinearGrid, error) {
	g := &rectilinearGrid{step: make([]uint64, len(chunks)), starts: make([][]uint64, len(chunks))}
	for d, edges := range chunks {
		if len(edges) == 0 {
			if d >= len(steps) || steps[d] == 0 {
				return nil, fmt.Errorf("zarr: rectilinear dimension %d has neither edges nor a chunk size", d)
			}
			g.step[d] = steps[d]
			continue
		}
		for _, e := range edges {
			if e == 0 {
				return nil, fmt.Errorf("zarr: rectilinear chunk edge 0 in dimension %d", d)
			}
		}
		starts := make([]uint64, len(edges)+1)
		for i, e := range edges {
			starts[i+1] = starts[i] + e
		}
		g.starts[d] = starts
	}
	return g, nil
}

func (g *rectilinearGrid) numChunks(dim int, n uint64) uint64 {
	if s := g.step[dim]; s > 0 {
		return (n + s - 1) / s
	}
	st := g.starts[dim]
	if n == 0 {
		return 0
	}
	for i := 1; i < len(st); i++ {
		if st[i] >= n {
			return uint64(i)
		}
	}
	return uint64(len(st) - 1)
}

func (g *rectilinearGrid) span(dim int, idx uint64) (uint64, uint64) {
	if s := g.step[dim]; s > 0 {
		return idx * s, s
	}
	st := g.starts[dim]
	if idx+1 >= uint64(len(st)) {
		return st[len(st)-1], 0
	}
	return st[idx], st[idx+1] - st[idx]
}

func (g *rectilinearGrid) index(dim int, pos uint64) uint64 {
	if s := g.step[dim]; s > 0 {
		return pos / s
	}
	st := g.starts[dim]
	lo, hi := 0, len(st)-2
	for lo < hi {
		mid := (lo + hi + 1) / 2
		if st[mid] <= pos {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	return uint64(lo)
}

func (g *rectilinearGrid) edges(dim int) []uint64 {
	st := g.starts[dim]
	out := make([]uint64, len(st)-1)
	for i := range out {
		out[i] = st[i+1] - st[i]
	}
	return out
}

// uniformEdge reports whether dimension d's explicit edges are all equal
// and cover extent, which zarr-python handles as a regular dimension.
func (g *rectilinearGrid) uniformEdge(d int, extent uint64) (uint64, bool) {
	edges := g.edges(d)
	if len(edges) == 0 {
		return 0, false
	}
	for _, e := range edges {
		if e != edges[0] {
			return 0, false
		}
	}
	sum := g.starts[d][len(g.starts[d])-1]
	if sum == extent || uint64(len(edges)) == (extent+edges[0]-1)/edges[0] {
		return edges[0], true
	}
	return 0, false
}

// json renders the grid as zarr-python does: bare ints for regular
// dimensions, and edge lists run-length encoded when that is shorter.
func (g *rectilinearGrid) json() any {
	dims := make([]any, len(g.step))
	for d := range dims {
		if g.step[d] > 0 {
			dims[d] = g.step[d]
			continue
		}
		edges := g.edges(d)
		var rle []any
		for i := 0; i < len(edges); {
			j := i
			for j < len(edges) && edges[j] == edges[i] {
				j++
			}
			if j-i > 1 {
				rle = append(rle, []any{edges[i], j - i})
			} else {
				rle = append(rle, edges[i])
			}
			i = j
		}
		if len(rle) < len(edges) {
			dims[d] = rle
		} else {
			list := make([]any, len(edges))
			for i, e := range edges {
				list[i] = e
			}
			dims[d] = list
		}
	}
	return jsonObject{{"name", "rectilinear"}, {"configuration", jsonObject{{"kind", "inline"}, {"chunk_shapes", dims}}}}
}

// resized appends an edge to explicit dimensions that no longer cover the
// array, as zarr-python does.
func (g *rectilinearGrid) resized(_, shape []uint64) chunkGrid {
	ng := &rectilinearGrid{step: append([]uint64{}, g.step...), starts: make([][]uint64, len(g.starts))}
	for d, st := range g.starts {
		ng.starts[d] = append([]uint64{}, st...)
		if g.step[d] == 0 && shape[d] > st[len(st)-1] {
			ng.starts[d] = append(ng.starts[d], shape[d])
		}
	}
	return ng
}

// maxRectilinearEdges bounds the chunk edges of a rectilinear grid (all
// dimensions together), which run-length encoding lets a few bytes of
// metadata declare: 32 MiB of offsets, far more than any real grid.
const maxRectilinearEdges = 1 << 22

func parseChunkGrid(raw json.RawMessage, shape []uint64) (chunkGrid, []uint64, error) {
	ndim := len(shape)
	var g struct {
		Name          string `json:"name"`
		Configuration struct {
			ChunkShape  []uint64          `json:"chunk_shape"`
			Kind        string            `json:"kind"`
			ChunkShapes []json.RawMessage `json:"chunk_shapes"`
		} `json:"configuration"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		return nil, nil, fmt.Errorf("zarr: invalid chunk_grid: %w", err)
	}
	switch g.Name {
	case "regular":
		cs := g.Configuration.ChunkShape
		if len(cs) != ndim {
			return nil, nil, fmt.Errorf("zarr: chunk_shape has %d dimensions, array has %d", len(cs), ndim)
		}
		for _, c := range cs {
			if c == 0 {
				return nil, nil, fmt.Errorf("zarr: chunk_shape contains 0")
			}
		}
		return regularGrid(cs), cs, nil
	case "rectilinear":
		if k := g.Configuration.Kind; k != "inline" && k != "" {
			return nil, nil, fmt.Errorf("%w: rectilinear chunk grid kind %q", ErrUnsupported, k)
		}
		shapes := g.Configuration.ChunkShapes
		if len(shapes) != ndim {
			return nil, nil, fmt.Errorf("zarr: chunk_shapes has %d dimensions, array has %d", len(shapes), ndim)
		}
		grid := &rectilinearGrid{step: make([]uint64, ndim), starts: make([][]uint64, ndim)}
		edges := uint64(0)
		for d, spec := range shapes {
			var step uint64
			if json.Unmarshal(spec, &step) == nil {
				if step == 0 {
					return nil, nil, fmt.Errorf("zarr: rectilinear step 0")
				}
				grid.step[d] = step
				continue
			}
			var entries []json.RawMessage
			if err := json.Unmarshal(spec, &entries); err != nil || len(entries) == 0 {
				return nil, nil, fmt.Errorf("zarr: invalid rectilinear chunk_shapes entry %s", spec)
			}
			starts := []uint64{0}
			for _, e := range entries {
				edge, count := uint64(0), uint64(1)
				if json.Unmarshal(e, &edge) != nil || edge == 0 {
					var rle []uint64
					if json.Unmarshal(e, &rle) != nil || len(rle) != 2 || rle[0] == 0 || rle[1] == 0 {
						return nil, nil, fmt.Errorf("zarr: invalid rectilinear edge %s", e)
					}
					edge, count = rle[0], rle[1]
				}
				if count > maxRectilinearEdges-edges {
					return nil, nil, fmt.Errorf("%w: rectilinear grid with more than %d chunk edges", ErrUnsupported, maxRectilinearEdges)
				}
				edges += count
				for range count {
					end := starts[len(starts)-1] + edge
					if end < edge {
						return nil, nil, fmt.Errorf("zarr: rectilinear chunk edges of dimension %d overflow", d)
					}
					starts = append(starts, end)
				}
			}
			// Edges may extend past the array (after shrinking it), but must
			// cover it, as zarr-python requires.
			if starts[len(starts)-1] < shape[d] {
				return nil, nil, fmt.Errorf("zarr: rectilinear chunk edges of dimension %d sum to %d, less than its length %d", d, starts[len(starts)-1], shape[d])
			}
			grid.starts[d] = starts
		}
		return grid, nil, nil
	}
	return nil, nil, fmt.Errorf("%w: chunk grid %q", ErrUnsupported, g.Name)
}

// ---------------------------------------------------------------------------
// Chunk key encodings

// ChunkKeyEncoding maps chunk grid coordinates to store keys.
type ChunkKeyEncoding struct {
	// Name is "default" ("c/1/2") or "v2" ("1.2").
	Name string
	// Separator is "/" or "."; empty means the encoding's default.
	Separator string
}

func (e ChunkKeyEncoding) separator() string {
	if e.Separator != "" {
		return e.Separator
	}
	if e.Name == "v2" {
		return "."
	}
	return "/"
}

// Key returns the key of the chunk at coords, relative to the array.
func (e ChunkKeyEncoding) Key(coords []uint64) string {
	parts := make([]string, len(coords))
	for i, c := range coords {
		parts[i] = strconv.FormatUint(c, 10)
	}
	if e.Name == "v2" {
		if len(parts) == 0 {
			return "0"
		}
		return strings.Join(parts, e.separator())
	}
	return strings.Join(append([]string{"c"}, parts...), e.separator())
}

// Parse returns the chunk coordinates of a key, or false if the key is not
// a chunk key of an array with ndim dimensions.
func (e ChunkKeyEncoding) Parse(key string, ndim int) ([]uint64, bool) {
	var parts []string
	if e.Name == "v2" {
		if ndim == 0 {
			return []uint64{}, key == "0"
		}
		parts = strings.Split(key, e.separator())
	} else {
		parts = strings.Split(key, e.separator())
		if len(parts) == 0 || parts[0] != "c" {
			return nil, false
		}
		parts = parts[1:]
	}
	if len(parts) != ndim {
		return nil, false
	}
	out := make([]uint64, ndim)
	for i, p := range parts {
		v, err := strconv.ParseUint(p, 10, 64)
		if err != nil || strconv.FormatUint(v, 10) != p {
			return nil, false
		}
		out[i] = v
	}
	return out, true
}

func (e ChunkKeyEncoding) json() any {
	return jsonObject{{"name", e.Name}, {"configuration", jsonObject{{"separator", e.separator()}}}}
}
