package consistency

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

var (
	randomCases = flag.Int("zarr.cases", 60, "number of random cases in TestRandomCases")
	randomSeed  = flag.Int64("zarr.seed", 1, "seed of TestRandomCases")
)

// TestRandomCases combines data types, codec chains, sharding, grids,
// shapes, fill values and both formats at random, and runs each through
// the standard exercise. Use -zarr.cases and -zarr.seed for longer runs.
func TestRandomCases(t *testing.T) {
	rng := rand.New(rand.NewSource(*randomSeed))
	n := *randomCases
	if testing.Short() {
		n = min(n, 15)
	}
	for i := 0; i < n; i++ {
		c := randomCase(rng)
		c.Name = fmt.Sprintf("%03d-%s", i, c.Name)
		run(t, c, 6, 6, rng.Intn(3) == 0)
	}
}

func pick[T any](rng *rand.Rand, v ...T) T { return v[rng.Intn(len(v))] }

func randomCase(rng *rand.Rand) Case {
	nd := pick(rng, 0, 1, 1, 2, 2, 2, 3, 3, 4)
	shape, chunks := make([]uint64, nd), make([]uint64, nd)
	for d := range shape {
		shape[d] = uint64(pick(rng, 0, 1, 2, 5, 7, 8, 13, 16, 30))
		if nd > 2 && shape[d] > 8 {
			shape[d] = 8
		}
		chunks[d] = uint64(1 + rng.Intn(int(max(shape[d], 1))+2))
	}
	c := Case{Shape: shape, Chunks: chunks}
	var parts []string
	if rng.Intn(4) == 0 && nd > 0 {
		c.ZarrFormat = 2
		d := pick(rng, dtypes...)
		for d.v2 == "" {
			d = pick(rng, dtypes...)
		}
		c.DType = raw(d.v2)
		if d.v2 == `"|O"` {
			c.StringType = "string"
		}
		c.Order = pick(rng, "C", "F")
		if len(d.fills) > 0 && rng.Intn(2) == 0 {
			c.FillValue = raw(pick(rng, d.fills...))
		}
		comp := pick(rng, "", `{"id": "zlib", "level": 1}`, `{"id": "bz2", "level": 3}`, `{"id": "zstd", "level": 1}`, `{"id": "crc32"}`, `{"id": "lz4"}`, `{"id": "blosc", "cname": "zstd", "clevel": 2, "shuffle": 2}`)
		if comp == "" {
			c.Compressors = compressors()
		} else {
			c.Compressors = compressors(comp)
		}
		parts = append(parts, "v2", strings.Trim(d.v2, `"<>|`), c.Order, codecName(comp))
		return named(c, parts)
	}
	c.ZarrFormat = 3
	d := pick(rng, dtypes...)
	// zarr-python 3.4 cannot write 0-d variable-length bytes arrays.
	for nd == 0 && strings.Contains(d.v3, "variable_length_bytes") {
		d = pick(rng, dtypes...)
	}
	c.DataType = raw(d.v3)
	if len(d.fills) > 0 && rng.Intn(2) == 0 {
		c.FillValue = raw(pick(rng, d.fills...))
	}
	parts = append(parts, "v3", shortType(d.v3))
	variable := strings.Contains(d.v3, `"string"`) || strings.Contains(d.v3, "variable_length")
	if !variable && rng.Intn(4) == 0 && nd > 0 {
		c.Serializer = raw(`{"name": "bytes", "configuration": {"endian": "big"}}`)
		parts = append(parts, "big")
	}
	if rng.Intn(4) == 0 && nd > 1 {
		perm := rng.Perm(nd)
		b, _ := json.Marshal(perm)
		c.Filters = rawList(fmt.Sprintf(`{"name": "transpose", "configuration": {"order": %s}}`, b))
		parts = append(parts, "T")
	}
	comp := pick(rng, "none", "none", "default", `{"name": "crc32c"}`, `{"name": "gzip", "configuration": {"level": 1}}`, `{"name": "numcodecs.bz2", "configuration": {"level": 1}}`, `{"name": "numcodecs.lzma", "configuration": {}}`, `{"name": "blosc", "configuration": {"cname": "lz4"}}`, `{"name": "numcodecs.crc32", "configuration": {}}`)
	switch comp {
	case "none":
		c.Compressors = compressors()
	case "default":
	default:
		c.Compressors = compressors(comp)
	}
	parts = append(parts, codecName(comp))
	if nd > 0 && rng.Intn(3) == 0 {
		shards := make([]uint64, nd)
		for i := range shards {
			shards[i] = chunks[i] * uint64(1+rng.Intn(3))
		}
		c.Shards = shards
		c.IndexLocation = pick(rng, "end", "start")
		parts = append(parts, "sharded-"+c.IndexLocation)
	} else if nd > 0 && rng.Intn(4) == 0 {
		c.ChunkEdges = make([][]uint64, nd)
		for i := range c.ChunkEdges {
			if rng.Intn(2) == 0 || shape[i] == 0 {
				continue
			}
			var edges []uint64
			for left := shape[i]; left > 0; {
				e := min(left, uint64(1+rng.Intn(4)))
				edges = append(edges, e)
				left -= e
			}
			c.ChunkEdges[i] = edges
		}
		parts = append(parts, "rectilinear")
	}
	// (zarr-python 3.4 fails partial writes of variable-length bytes with
	// write_empty_chunks.)
	if rng.Intn(5) == 0 && !strings.Contains(d.v3, "variable_length_bytes") {
		c.WriteEmptyChunks = true
		parts = append(parts, "write-empty")
	}
	if nd > 0 && rng.Intn(6) == 0 {
		c.ChunkKeyEncoding = raw(randomTemplate(rng, nd))
		parts = append(parts, "template")
	}
	return named(c, parts)
}

// randomTemplate returns a template chunk key encoding for nd dimensions:
// some coordinates named by index (positive or negative), the others by
// {*}, with optional zero padding. Literals separate the expressions, so
// keys are unambiguous.
func randomTemplate(rng *rand.Rand, nd int) string {
	pad := func() string {
		if rng.Intn(3) == 0 {
			return fmt.Sprintf(":0%d", 1+rng.Intn(4))
		}
		return ""
	}
	perm := rng.Perm(nd)
	named := rng.Intn(nd + 1)
	var exprs []string
	for _, d := range perm[:named] {
		i := d
		if rng.Intn(2) == 0 {
			i = d - nd
		}
		exprs = append(exprs, fmt.Sprintf("{%d%s}", i, pad()))
	}
	cfg := map[string]string{}
	if named < nd {
		at := rng.Intn(len(exprs) + 1)
		exprs = append(exprs[:at], append([]string{"{*" + pad() + "}"}, exprs[at:]...)...)
		cfg["separator"] = pick(rng, "/", ".", "_", "-")
	}
	var b strings.Builder
	b.WriteString(pick(rng, "", "c/", "chunks.", "k-"))
	for i, e := range exprs {
		if i > 0 {
			b.WriteString(pick(rng, "/", ".", "_", "-x"))
		}
		b.WriteString(e)
	}
	b.WriteString(pick(rng, "", ".bin", "_end"))
	cfg["format"] = b.String()
	out, _ := json.Marshal(map[string]any{"name": "template", "configuration": cfg})
	return string(out)
}

func named(c Case, parts []string) Case {
	c.Name = strings.Join(append(parts, fmt.Sprint(c.Shape)), "-")
	return c
}

func codecName(s string) string {
	if s == "" {
		return "none"
	}
	var obj struct{ Name, ID string }
	if json.Unmarshal([]byte(s), &obj) == nil && obj.Name+obj.ID != "" {
		return obj.Name + obj.ID
	}
	return s
}

func shortType(v3 string) string {
	var name string
	if json.Unmarshal([]byte(v3), &name) == nil {
		return name
	}
	var obj struct{ Name string }
	json.Unmarshal([]byte(v3), &obj)
	return obj.Name
}
