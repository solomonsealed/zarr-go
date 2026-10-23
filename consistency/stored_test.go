package consistency

import (
	"bytes"
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

// storedCase is metadata another program stored: older zarr-python
// releases, netCDF-C, Zarr.jl, jzarr, zarrs.
type storedCase struct {
	name string
	// docs are the array's documents by name (".zarray", ".zattrs",
	// "zarr.json").
	docs map[string]string
	// refused: both implementations must refuse to open the array.
	refused bool
	// resize, if set, is applied before the writes (arrays stored with a
	// zero-length axis).
	resize []uint64
	// compressed: chunks are compared by value, not byte for byte.
	compressed bool
}

func v2Doc(shape, chunks, dtype, fill, extra string) string {
	return `{"zarr_format": 2, "shape": ` + shape + `, "chunks": ` + chunks + `, "dtype": ` + dtype + `, "fill_value": ` + fill +
		`, "order": "C", "filters": null, "compressor": null` + extra + `}`
}

// v3Doc is a zarr.json as zarr-python 3 writes it (the arrays whose
// metadata needs repairs were written by its earlier releases).
func v3Doc(shape, dataType, grid, fill, codecs, extra string) string {
	return `{"shape": ` + shape + `, "data_type": ` + dataType + `, "chunk_grid": ` + grid +
		`, "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": ` + fill +
		`, "codecs": ` + codecs + `, "attributes": {}, "zarr_format": 3, "node_type": "array", "storage_transformers": []` + extra + `}`
}

func regular(shape string) string {
	return `{"name": "regular", "configuration": {"chunk_shape": ` + shape + `}}`
}

func rectilinear(shapes string) string {
	return `{"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": ` + shapes + `}}`
}

const (
	bytesLE = `[{"name": "bytes", "configuration": {"endian": "little"}}]`
	// zarr-python 2.x wrote .zarray with sorted keys and an indent of 4.
	zarr2Doc = `{
    "chunks": [
        0
    ],
    "compressor": {
        "blocksize": 0,
        "clevel": 5,
        "cname": "lz4",
        "id": "blosc",
        "shuffle": 1
    },
    "dimension_separator": ".",
    "dtype": "<f8",
    "fill_value": "NaN",
    "filters": null,
    "order": "C",
    "shape": [
        0
    ],
    "zarr_format": 2
}`
)

func sharded(inner, innerCodecs, indexCodecs string) string {
	return `[{"name": "sharding_indexed", "configuration": {"chunk_shape": ` + inner + `, "codecs": ` + innerCodecs + `, "index_codecs": ` + indexCodecs + `, "index_location": "end"}}]`
}

const crcIndex = `[{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}]`

func storedCases() []storedCase {
	v2 := func(name, doc string) storedCase {
		return storedCase{name: name, docs: map[string]string{".zarray": doc}}
	}
	v3 := func(name, doc string) storedCase {
		return storedCase{name: name, docs: map[string]string{"zarr.json": doc}}
	}
	refused := func(c storedCase) storedCase { c.refused = true; return c }
	grown := func(c storedCase, shape ...uint64) storedCase { c.resize = shape; return c }
	return []storedCase{
		// Chunk sizes of 0, false and true (zarr-python 3.4.1 repairs).
		grown(v2("v2-chunks-0-empty", v2Doc(`[0]`, `[0]`, `"<i2"`, `0`, ``)), 5),
		v2("v2-chunks-0", v2Doc(`[6]`, `[0]`, `"<i2"`, `0`, ``)),
		v2("v2-chunks-false", v2Doc(`[4, 3]`, `[false, 2]`, `"<u4"`, `7`, ``)),
		v2("v2-chunks-true", v2Doc(`[3, 4]`, `[true, 2]`, `"<i8"`, `0`, ``)),
		grown(storedCase{name: "v2-zarr2", docs: map[string]string{".zarray": zarr2Doc, ".zattrs": `{"title": "x"}`}}, 7),
		grown(v3("v3-chunks-0-empty", v3Doc(`[0, 3]`, `"int16"`, regular(`[0, 2]`), `0`, bytesLE, ``)), 5, 3),
		v3("v3-chunks-0", v3Doc(`[6]`, `"float32"`, regular(`[0]`), `"NaN"`, bytesLE, ``)),
		v3("v3-chunks-false", v3Doc(`[6]`, `"float32"`, regular(`[false]`), `0.0`, bytesLE, ``)),
		v3("v3-chunks-true", v3Doc(`[4, 3]`, `"int8"`, regular(`[true, 2]`), `0`, `[{"name": "bytes"}]`, ``)),
		grown(v3("v3-sharded-0-empty", v3Doc(`[0]`, `"int16"`, regular(`[0]`), `0`, sharded(`[2]`, bytesLE, crcIndex), ``)), 9),
		v3("v3-sharded-0", v3Doc(`[9]`, `"int16"`, regular(`[0]`), `0`, sharded(`[3]`, bytesLE, crcIndex), ``)),
		v3("v3-sharded-inner-true", v3Doc(`[4]`, `"int16"`, regular(`[4]`), `0`, sharded(`[true]`, bytesLE, crcIndex), ``)),
		v3("v3-mixed-grid", v3Doc(`[4, 20]`, `"int32"`, regular(`[2, [5, 10, 5]]`), `0`, bytesLE, ``)),
		v3("v3-mixed-grid-floats", v3Doc(`[4, 20]`, `"int32"`, regular(`[2, [5.0, 10.0, 5.0]]`), `0`, bytesLE, ``)),
		v3("v3-rectilinear-floats", v3Doc(`[6]`, `"uint8"`, rectilinear(`[[[2.0, 2], true, 1]]`), `0`, `[{"name": "bytes"}]`, ``)),
		v3("v3-rectilinear-explicit", v3Doc(`[6, 4]`, `"uint8"`, rectilinear(`[[2, 2, 2], 2]`), `0`, `[{"name": "bytes"}]`, ``)),
		refused(v3("v3-chunks-float", v3Doc(`[6]`, `"int8"`, regular(`[2.0]`), `0`, `[{"name": "bytes"}]`, ``))),
		refused(v3("v3-chunks-negative", v3Doc(`[6]`, `"int8"`, regular(`[-1]`), `0`, `[{"name": "bytes"}]`, ``))),
		refused(v3("v3-chunks-string", v3Doc(`[6]`, `"int8"`, regular(`["2"]`), `0`, `[{"name": "bytes"}]`, ``))),
		refused(v3("v3-chunks-all-lists", v3Doc(`[6]`, `"int8"`, regular(`[[2, 4]]`), `0`, `[{"name": "bytes"}]`, ``))),
		refused(v3("v3-sharded-inner-0", v3Doc(`[4]`, `"int16"`, regular(`[4]`), `0`, sharded(`[0]`, bytesLE, crcIndex), ``))),
		refused(v3("v3-sharded-inner-false", v3Doc(`[4]`, `"int16"`, regular(`[4]`), `0`, sharded(`[false]`, bytesLE, crcIndex), ``))),
		// Sharding validation (zarr-python 3.4.1).
		refused(v3("v3-index-compressed", v3Doc(`[8]`, `"int16"`, regular(`[8]`), `0`, sharded(`[4]`, bytesLE, `[{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd", "configuration": {"level": 0, "checksum": false}}]`), ``))),
		refused(v3("v3-nested-indivisible", v3Doc(`[8]`, `"int16"`, regular(`[8]`), `0`, sharded(`[4]`, sharded(`[3]`, bytesLE, crcIndex), crcIndex), ``))),
		// Zarr v2 data types written with a byte order that does not apply
		// (netCDF-C, Zarr.jl, jzarr), and time units.
		v2("v2-dtype-<u1", v2Doc(`[5]`, `[2]`, `"<u1"`, `0`, ``)),
		v2("v2-dtype->i1", v2Doc(`[5]`, `[2]`, `">i1"`, `-1`, ``)),
		v2("v2-dtype-<b1", v2Doc(`[5]`, `[2]`, `"<b1"`, `false`, ``)),
		v2("v2-dtype->V3", v2Doc(`[5]`, `[2]`, `">V3"`, `"AAAA"`, ``)),
		v2("v2-dtype-struct", v2Doc(`[5]`, `[2]`, `[["a", "<i1"], ["b", ">u1"], ["c", "<V2"]]`, `"AAAAAA=="`, ``)),
		v2("v2-dtype-μs", v2Doc(`[5]`, `[2]`, `"<M8[3μs]"`, `null`, ``)),
		// Little-endian: big-endian generic times trip a zarr-python issue
		// (TestKnownPythonIssues).
		v2("v2-dtype-generic-scale", v2Doc(`[5]`, `[2]`, `"<m8[2generic]"`, `null`, ``)),
		v2("v2-dtype-big-scaled", v2Doc(`[5]`, `[2]`, `">M8[2s]"`, `null`, ``)),
		v2("v2-extra-key", v2Doc(`[5]`, `[2]`, `"<i4"`, `0`, `, "extra": 1`)),
		{name: "v2-zstd", compressed: true, docs: map[string]string{".zarray": `{"zarr_format": 2, "shape": [5], "chunks": [2], "dtype": "<f4", "fill_value": 0, "order": "C", "filters": null, "compressor": {"id": "zstd", "level": 1}}`}},
		v3("v3-dtype-μs", v3Doc(`[5]`, `{"name": "numpy.datetime64", "configuration": {"unit": "μs", "scale_factor": 3}}`, regular(`[2]`), `"NaT"`, bytesLE, ``)),
		v3("v3-dtype-generic-scale", v3Doc(`[5]`, `{"name": "numpy.timedelta64", "configuration": {"unit": "generic", "scale_factor": 2}}`, regular(`[2]`), `"NaT"`, bytesLE, ``)),
		v3("v3-dtype-struct-μs", v3Doc(`[5]`, `{"name": "struct", "configuration": {"fields": [{"name": "t", "data_type": {"name": "numpy.datetime64", "configuration": {"unit": "μs", "scale_factor": 1}}}]}}`, regular(`[2]`), `"AAAAAAAAAAA="`, bytesLE, ``)),
		// Extra fields.
		v3("v3-extra-field", v3Doc(`[5]`, `"int8"`, regular(`[2]`), `0`, `[{"name": "bytes"}]`, `, "ext": {"must_understand": false, "x": [1, 2]}`)),
		refused(v3("v3-extra-field-not-object", v3Doc(`[5]`, `"int8"`, regular(`[2]`), `0`, `[{"name": "bytes"}]`, `, "ext": 1`))),
		refused(v3("v3-extra-field-must-understand", v3Doc(`[5]`, `"int8"`, regular(`[2]`), `0`, `[{"name": "bytes"}]`, `, "ext": {"must_understand": true}`))),
	}
}

// TestStoredDocuments opens array metadata other programs stored with both
// implementations. Both must refuse the same documents, and read the
// others alike (zarr-python's repairs of invalid ones included): the same
// documents to write back, the same repairs awaiting storage, and the same
// stores after writes, resizes and attribute updates.
func TestStoredDocuments(t *testing.T) {
	for _, sc := range storedCases() {
		t.Run(sc.name, func(t *testing.T) {
			o := pythonOracle(t)
			h := newHarness(t, o, Case{Parity: ptrBool(!sc.compressed)}, seedOf(t.Name()))
			for _, dir := range []string{h.pyDir, h.goDir} {
				for name, doc := range sc.docs {
					p := filepath.Join(dir, "arr", name)
					must(t, os.MkdirAll(filepath.Dir(p), 0o755))
					must(t, os.WriteFile(p, []byte(doc), 0o644))
				}
			}
			if !h.adopt(sc.refused) {
				return
			}
			if sc.resize != nil {
				h.resize(sc.resize)
			}
			fill := h.goArr.FillValue()
			h.write(selection{Kind: "orthogonal"}, genData(h.rng, h.dt, h.goArr.Shape(), fill, 0.2))
			h.crossRead("after full write")
			sel := randomSelection(h.rng, "basic", h.goArr.Shape(), h.goArr.ChunkGridShape())
			if cur, err := sel.get(h.ctx, h.goArr); err == nil {
				h.write(sel, genData(h.rng, h.dt, cur.Shape, fill, 0.2))
			}
			h.setAttributes(map[string]any{"updated": true}) // sorts after existing keys
			if shape := h.goArr.Shape(); len(shape) > 0 {
				grown := append([]uint64{}, shape...)
				grown[0] += 3
				h.resize(grown)
			}
		})
	}
}

func ptrBool(v bool) *bool { return &v }

// adopt opens the array both stores hold: both must refuse it (if
// refused) or read it alike. It returns whether it was opened.
func (h *harness) adopt(refused bool) bool {
	t := h.t
	t.Helper()
	var py struct {
		Documents     map[string]string `json:"documents"`
		RepairPending bool              `json:"repair_pending"`
	}
	pyErr := h.o.call("documents", map[string]any{"store": h.pyDir, "path": "arr"}, &py)
	a, goErr := zarr.OpenArray(h.ctx, zarr.NewLocalStore(h.goDir), "arr", nil)
	if refused || pyErr != nil || goErr != nil {
		if pyErr != nil && goErr != nil && refused {
			h.report.Errors++
			t.Logf("both refused: python %v; go %v", pyErr, goErr)
			return false
		}
		t.Fatalf("open: python %v; go %v (expected both to %s)", pyErr, goErr, map[bool]string{true: "refuse", false: "open"}[refused])
	}
	h.goArr, h.dt = a, a.DataType()
	m := a.Metadata()
	goDocs := map[string][]byte{}
	if m.ZarrFormat == 2 {
		goDocs[".zarray"], _ = m.Document()
		goDocs[".zattrs"], _ = m.AttributesDocument()
	} else {
		goDocs["zarr.json"], _ = m.Document()
	}
	for name, enc := range py.Documents {
		want, _ := base64.StdEncoding.DecodeString(enc)
		h.report.Comparisons++
		if !bytes.Equal(want, goDocs[name]) {
			t.Fatalf("%s to write back differs:\n%s", name, lineDiff(string(want), string(goDocs[name])))
		}
	}
	if py.RepairPending != m.RepairPending() {
		t.Fatalf("repair pending: python %v, go %v", py.RepairPending, m.RepairPending())
	}
	h.crossRead("open")
	return true
}

// setAttributes replaces the array's attributes with both implementations
// (which start empty, so zarr-python's update_attributes replaces them too).
func (h *harness) setAttributes(attrs map[string]any) {
	t := h.t
	t.Helper()
	h.report.Ops++
	merged := map[string]any{}
	for k, v := range h.goArr.Attributes() {
		merged[k] = v
	}
	for k, v := range attrs {
		merged[k] = v
	}
	pyErr := h.o.call("update_attributes", map[string]any{"store": h.pyDir, "path": "arr", "attributes": attrs}, nil)
	goErr := h.goArr.SetAttributes(h.ctx, merged)
	if pyErr != nil || goErr != nil {
		t.Fatalf("set attributes: python %v; go %v", pyErr, goErr)
	}
	h.compareStores("after setting attributes")
}

// TestTruncatedShards checks that both implementations refuse shards too
// short to hold their index, empty ones included, on every read and
// partial write (zarr-python 3.4.1).
func TestTruncatedShards(t *testing.T) {
	for _, loc := range []string{"end", "start"} {
		for _, size := range []int{0, 7} {
			c := Case{Name: "truncated", ZarrFormat: 3, Shape: []uint64{8}, Chunks: []uint64{2}, Shards: []uint64{4}, IndexLocation: loc, DataType: raw(`"int16"`), Compressors: compressors()}
			t.Run(loc+"-"+strings.Repeat("x", size), func(t *testing.T) {
				o := pythonOracle(t)
				h := newHarness(t, o, c, seedOf(t.Name()))
				if !h.create() {
					t.Fatal("both refused the case")
				}
				h.write(selection{Kind: "orthogonal"}, genData(h.rng, h.dt, h.goArr.Shape(), h.goArr.FillValue(), 0))
				for _, dir := range []string{h.pyDir, h.goDir} {
					must(t, os.WriteFile(filepath.Join(dir, "arr", "c", "0"), make([]byte, size), 0o644))
				}
				before := h.report.Errors
				all := selection{Kind: "orthogonal"}
				part := selection{Kind: "orthogonal", Items: []dimItem{slc(ptr(1), ptr(2), nil)}}
				h.read(all)
				h.read(part)
				one, _ := zarr.FromSlice([]uint64{1}, []int16{5})
				h.write(part, one)
				if got := h.report.Errors - before; got != 3 {
					t.Fatalf("%d of 3 operations on the truncated shard were refused by both", got)
				}
			})
		}
	}
}

// TestGroupExtraFields checks Zarr v3 group documents with extra fields:
// objects with "must_understand": false are kept and written back (also in
// consolidated metadata), others refused, as zarr-python 3.4.1 does.
func TestGroupExtraFields(t *testing.T) {
	ctx := context.Background()
	for _, c := range []struct {
		name, doc string
		refused   bool
	}{
		{"kept", `{"ext": {"must_understand": false, "x": 1}, "zarr_format": 3, "node_type": "group", "attributes": {"a": 1}}`, false},
		{"not-object", `{"zarr_format": 3, "node_type": "group", "attributes": {}, "ext": [1]}`, true},
		{"must-understand", `{"zarr_format": 3, "node_type": "group", "attributes": {}, "ext": {"must_understand": true}}`, true},
	} {
		t.Run(c.name, func(t *testing.T) {
			o := pythonOracle(t)
			py, gd := t.TempDir(), t.TempDir()
			for _, dir := range []string{py, gd} {
				must(t, os.WriteFile(filepath.Join(dir, "zarr.json"), []byte(c.doc), 0o644))
			}
			var pyDocs struct {
				Documents map[string]string `json:"documents"`
			}
			pyErr := o.call("documents", map[string]any{"store": py, "path": ""}, &pyDocs)
			g, goErr := zarr.OpenGroup(ctx, zarr.NewLocalStore(gd), "", nil)
			if c.refused {
				if pyErr == nil || goErr == nil {
					t.Fatalf("expected both to refuse: python %v; go %v", pyErr, goErr)
				}
				return
			}
			if pyErr != nil || goErr != nil {
				t.Fatalf("open: python %v; go %v", pyErr, goErr)
			}
			must(t, o.call("update_attributes", map[string]any{"store": py, "path": "", "attributes": map[string]any{"b": 2}}, nil))
			must(t, g.SetAttributes(ctx, map[string]any{"a": 1, "b": 2}))
			compareDirs(t, py, gd, "after setting attributes")
			ac := hierarchyCase(Case{Shape: []uint64{3}, Chunks: []uint64{2}}, 3)
			must(t, o.call("create", map[string]any{"store": py, "path": "x", "case": ac}, nil))
			spec, err := ac.goSpec()
			must(t, err)
			_, err = zarr.CreateArray(ctx, zarr.NewLocalStore(gd), "x", spec, nil)
			must(t, err)
			must(t, o.call("consolidate", map[string]any{"store": py}, nil))
			_, err = zarr.ConsolidateMetadata(ctx, zarr.NewLocalStore(gd), "")
			must(t, err)
			compareDirs(t, py, gd, "consolidated")
		})
	}
}

// TestConsolidatedRepairs consolidates a hierarchy holding an array whose
// stored metadata needs a repair that moves chunks: the consolidated
// metadata keeps the document as stored until the array stores its repair,
// as in zarr-python.
func TestConsolidatedRepairs(t *testing.T) {
	ctx := context.Background()
	for _, format := range []int{2, 3} {
		t.Run(map[int]string{2: "v2", 3: "v3"}[format], func(t *testing.T) {
			o := pythonOracle(t)
			py, gd := t.TempDir(), t.TempDir()
			name, doc := ".zarray", v2Doc(`[6]`, `[0]`, `"<i2"`, `0`, ``)
			if format == 3 {
				name, doc = "zarr.json", v3Doc(`[6]`, `"int16"`, regular(`[0]`), `0`, bytesLE, ``)
			}
			for _, dir := range []string{py, gd} {
				must(t, o.call("create_group", map[string]any{"store": dir, "zarr_format": format}, nil))
				must(t, os.MkdirAll(filepath.Join(dir, "x"), 0o755))
				must(t, os.WriteFile(filepath.Join(dir, "x", name), []byte(doc), 0o644))
			}
			must(t, o.call("consolidate", map[string]any{"store": py}, nil))
			_, err := zarr.ConsolidateMetadata(ctx, zarr.NewLocalStore(gd), "")
			must(t, err)
			compareDirs(t, py, gd, "consolidated")
		})
	}
}
