package zarr

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"testing"
)

// The behaviour of zarr-python 3.4.1 these tests pin down is also checked
// against zarr-python itself in consistency/stored_test.go.

func v3ArrayDoc(shape, grid, codecs string) string {
	return `{"shape": ` + shape + `, "data_type": "int16", "chunk_grid": ` + grid + `, "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": 0, "codecs": ` + codecs + `, "attributes": {}, "zarr_format": 3, "node_type": "array", "storage_transformers": []}`
}

func v2ArrayDoc(shape, chunks string) string {
	return `{"zarr_format": 2, "shape": ` + shape + `, "chunks": ` + chunks + `, "dtype": "<i2", "fill_value": 0, "order": "C", "filters": null, "compressor": null}`
}

const shardCodecs = `[{"name": "sharding_indexed", "configuration": {"chunk_shape": %s, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}], "index_location": "end"}}]`

func sharded3(inner string) string { return strings.Replace(shardCodecs, "%s", inner, 1) }

func regularGridJSON(cs string) string {
	return `{"name": "regular", "configuration": {"chunk_shape": ` + cs + `}}`
}

func TestRepairInvalidChunkShapes(t *testing.T) {
	le := `[{"name": "bytes", "configuration": {"endian": "little"}}]`
	cases := []struct {
		name, doc string
		chunks    []uint64 // Array.Chunks
		grid      []uint64 // ChunkGridShape
		pending   bool
		wrote     string // in the compact document written back
	}{
		{"v2 0", v2ArrayDoc(`[6]`, `[0]`), []uint64{1}, []uint64{6}, true, `"chunks": [1]`},
		{"v2 false", v2ArrayDoc(`[4, 3]`, `[false, 2]`), []uint64{1, 2}, []uint64{4, 2}, true, `"chunks": [1, 2]`},
		{"v2 true", v2ArrayDoc(`[3, 4]`, `[true, 2]`), []uint64{1, 2}, []uint64{3, 2}, false, `"chunks": [1, 2]`},
		{"v3 0 empty", v3ArrayDoc(`[0]`, regularGridJSON(`[0]`), le), []uint64{1}, []uint64{0}, true, `"chunk_shape": [1]`},
		{"v3 true", v3ArrayDoc(`[4]`, regularGridJSON(`[true]`), le), []uint64{1}, []uint64{4}, false, `"chunk_shape": [1]`},
		{"v3 sharded 0", v3ArrayDoc(`[9]`, regularGridJSON(`[0]`), sharded3(`[3]`)), []uint64{3}, []uint64{3}, true, `"chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [3]}}`},
		{"v3 inner true", v3ArrayDoc(`[4]`, regularGridJSON(`[4]`), sharded3(`[true]`)), []uint64{1}, []uint64{1}, false, `"sharding_indexed", "configuration": {"chunk_shape": [1]`},
		{"v3 mixed", v3ArrayDoc(`[4, 20]`, regularGridJSON(`[2, [5.0, 10, 5]]`), le), nil, []uint64{2, 3}, true, `"chunk_grid": {"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": [2, [5, 10, 5]]}}`},
		{"v3 rectilinear floats", v3ArrayDoc(`[6]`, `{"name": "rectilinear", "configuration": {"kind": "inline", "chunk_shapes": [[[2.0, 2], true, 1]]}}`, le), nil, []uint64{4}, false, `"chunk_shapes": [[[2, 2], [1, 2]]]`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a, err := NewArray([]byte(c.doc), &mapChunks{}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if got := a.Chunks(); !slices.Equal(got, c.chunks) {
				t.Errorf("chunks %v, want %v", got, c.chunks)
			}
			if got := a.ChunkGridShape(); !slices.Equal(got, c.grid) {
				t.Errorf("grid %v, want %v", got, c.grid)
			}
			if got := a.Metadata().RepairPending(); got != c.pending {
				t.Errorf("repair pending %v, want %v", got, c.pending)
			}
			doc, _ := marshalPython(a.Metadata().doc, -1)
			if !strings.Contains(string(doc), c.wrote) {
				t.Errorf("document written back:\n%s\nlacks %s", doc, c.wrote)
			}
		})
	}
	for _, doc := range []string{
		v3ArrayDoc(`[6]`, regularGridJSON(`[2.0]`), le),
		v3ArrayDoc(`[6]`, regularGridJSON(`[-1]`), le),
		v3ArrayDoc(`[6]`, regularGridJSON(`["2"]`), le),
		v3ArrayDoc(`[6]`, regularGridJSON(`[[2, 4]]`), le), // only edge lists
		v3ArrayDoc(`[4]`, regularGridJSON(`[4]`), sharded3(`[0]`)),
		v3ArrayDoc(`[4]`, regularGridJSON(`[4]`), sharded3(`[false]`)),
		v2ArrayDoc(`[4]`, `[2.0]`),
		v2ArrayDoc(`[4]`, `[[2, 2]]`),
	} {
		if _, err := NewArray([]byte(doc), &mapChunks{}, nil); err == nil {
			t.Errorf("accepted %s", doc)
		}
	}
}

// An array zarr-python 2 wrote (sorted keys, indent 4) with chunks of 0 is
// written back as zarr-python 3.4.1 writes it.
func TestRepairZarr2Document(t *testing.T) {
	stored := "{\n    \"chunks\": [\n        0\n    ],\n    \"compressor\": {\n        \"blocksize\": 0,\n        \"clevel\": 5,\n        \"cname\": \"lz4\",\n        \"id\": \"blosc\",\n        \"shuffle\": 1\n    },\n    \"dimension_separator\": \".\",\n    \"dtype\": \"<u1\",\n    \"fill_value\": 0,\n    \"filters\": null,\n    \"order\": \"C\",\n    \"shape\": [\n        0\n    ],\n    \"zarr_format\": 2\n}"
	// zarr-python 3.4.1's metadata.to_buffer_dict()[".zarray"].
	want := "{\n  \"shape\": [\n    0\n  ],\n  \"chunks\": [\n    1\n  ],\n  \"dtype\": \"|u1\",\n  \"fill_value\": 0,\n  \"order\": \"C\",\n  \"filters\": null,\n  \"dimension_separator\": \".\",\n  \"compressor\": {\n    \"id\": \"blosc\",\n    \"cname\": \"lz4\",\n    \"clevel\": 5,\n    \"shuffle\": 1,\n    \"blocksize\": 0\n  },\n  \"zarr_format\": 2\n}"
	m, err := ParseMetadataV2([]byte(stored), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := m.Document(); string(got) != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
	if !m.RepairPending() {
		t.Error("no repair pending")
	}
}

func TestRepairStoredBeforeFirstWrite(t *testing.T) {
	ctx := context.Background()
	open := func(t *testing.T, s *MemoryStore) *Array {
		t.Helper()
		a, err := OpenArray(ctx, s, "a", nil)
		if err != nil {
			t.Fatal(err)
		}
		return a
	}
	one, _ := FromSlice([]uint64{1}, []int16{7})

	t.Run("stored", func(t *testing.T) {
		s := NewMemoryStore()
		s.Set(ctx, "a/.zarray", []byte(v2ArrayDoc(`[6]`, `[0]`)))
		a := open(t, s)
		if err := a.Set(ctx, one, Slice(2, 2)); err != nil { // empty selection: nothing stored
			t.Fatal(err)
		}
		if _, ok, _ := s.Get(ctx, "a/.zattrs", ByteRange{}); ok {
			t.Fatal("an empty write stored metadata")
		}
		if err := a.Set(ctx, one, Index(1)); err != nil {
			t.Fatal(err)
		}
		doc, _, _ := s.Get(ctx, "a/.zarray", ByteRange{})
		m, err := ParseMetadataV2(doc, nil)
		if err != nil || m.RepairPending() || !slices.Equal(m.ChunkShape, []uint64{1}) {
			t.Fatalf("stored metadata %s: %v", doc, err)
		}
		// The attributes document is stored too, as zarr-python does.
		if _, ok, _ := s.Get(ctx, "a/.zattrs", ByteRange{}); !ok {
			t.Error("no .zattrs stored")
		}
		if _, ok, _ := s.Get(ctx, "a/1", ByteRange{}); !ok || a.Metadata().RepairPending() {
			t.Error("chunk not written, or repair still pending")
		}
	})
	t.Run("replaced by another layout", func(t *testing.T) {
		s := NewMemoryStore()
		s.Set(ctx, "a/zarr.json", []byte(v3ArrayDoc(`[6]`, regularGridJSON(`[0]`), `[{"name": "bytes", "configuration": {"endian": "little"}}]`)))
		a := open(t, s)
		valid := v3ArrayDoc(`[6]`, regularGridJSON(`[2]`), `[{"name": "bytes", "configuration": {"endian": "little"}}]`)
		s.Set(ctx, "a/zarr.json", []byte(valid))
		if err := a.Set(ctx, one, Index(1)); err == nil || !strings.Contains(err.Error(), "reopen") {
			t.Fatalf("got %v", err)
		}
		keys, _ := s.List(ctx, "")
		if doc, _, _ := s.Get(ctx, "a/zarr.json", ByteRange{}); len(keys) != 1 || string(doc) != valid {
			t.Errorf("stored %v", keys)
		}
	})
	t.Run("replaced by a valid document", func(t *testing.T) {
		s := NewMemoryStore()
		s.Set(ctx, "a/.zarray", []byte(v2ArrayDoc(`[6]`, `[0]`)))
		a := open(t, s)
		valid := v2ArrayDoc(`[6]`, `[1]`)
		s.Set(ctx, "a/.zarray", []byte(valid))
		if err := a.Set(ctx, one, Index(1)); err != nil {
			t.Fatal(err)
		}
		if doc, _, _ := s.Get(ctx, "a/.zarray", ByteRange{}); string(doc) != valid {
			t.Errorf("valid metadata replaced by\n%s", doc)
		}
	})
}

func TestConsolidatedMetadataKeepsStoredDocument(t *testing.T) {
	ctx := context.Background()
	s := NewMemoryStore()
	if _, err := CreateGroup(ctx, s, "", GroupSpec{}, nil); err != nil {
		t.Fatal(err)
	}
	s.Set(ctx, "x/zarr.json", []byte(v3ArrayDoc(`[6]`, regularGridJSON(`[0]`), `[{"name": "bytes", "configuration": {"endian": "little"}}]`)))
	s.Set(ctx, "y/zarr.json", []byte(v3ArrayDoc(`[6]`, regularGridJSON(`[true]`), `[{"name": "bytes", "configuration": {"endian": "little"}}]`)))
	g, err := ConsolidateMetadata(ctx, s, "")
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		CM struct {
			Metadata map[string]struct {
				Grid struct {
					Configuration struct {
						ChunkShape []any `json:"chunk_shape"`
					} `json:"configuration"`
				} `json:"chunk_grid"`
			} `json:"metadata"`
		} `json:"consolidated_metadata"`
	}
	b, _, _ := s.Get(ctx, "zarr.json", ByteRange{})
	if err := json.Unmarshal(b, &doc); err != nil {
		t.Fatal(err)
	}
	// The repair that moves chunks is left for the array to store; the
	// one that respells a value is consolidated.
	if x := doc.CM.Metadata["x"].Grid.Configuration.ChunkShape; len(x) != 1 || x[0] != float64(0) {
		t.Errorf("x: %v", x)
	}
	if y := doc.CM.Metadata["y"].Grid.Configuration.ChunkShape; len(y) != 1 || y[0] != float64(1) {
		t.Errorf("y: %v", y)
	}
	a, err := g.OpenArray(ctx, "x")
	if err != nil || !slices.Equal(a.Chunks(), []uint64{1}) {
		t.Fatalf("%v %v", a, err)
	}
}

func TestCanonicalDataTypes(t *testing.T) {
	for stored, want := range map[string]string{
		`"<u1"`: `"|u1"`, `">i1"`: `"|i1"`, `"<b1"`: `"|b1"`, `">V3"`: `"|V3"`,
		`[["a", "<i1"], ["b", ">u1"]]`: `[["a", "|i1"], ["b", "|u1"]]`,
		`"<M8[3μs]"`:                   `"<M8[3us]"`,
		`">m8[2generic]"`:              `">m8[2generic]"`,
		`"<M8[1ns]"`:                   `"<M8[ns]"`,
	} {
		m, err := ParseMetadataV2([]byte(`{"zarr_format": 2, "shape": [2], "chunks": [2], "dtype": `+stored+`, "fill_value": null, "order": "C", "filters": null, "compressor": null}`), nil)
		if err != nil {
			t.Errorf("%s: %v", stored, err)
			continue
		}
		dt, _ := m.doc.get("dtype")
		if got, _ := marshalPython(dt, -1); string(got) != want {
			t.Errorf("%s written back as %s, want %s", stored, got, want)
		}
	}
	if d := Datetime64("μs", 3); d.Unit != "us" {
		t.Errorf("unit %q", d.Unit)
	}
	if got, _ := Timedelta64("generic", 2).v2JSON(false); got != "<m8[2generic]" {
		t.Errorf("got %v", got)
	}
	m, err := ParseMetadata([]byte(`{"shape": [2], "data_type": {"name": "struct", "configuration": {"fields": [{"name": "t", "data_type": {"name": "numpy.datetime64", "configuration": {"unit": "μs", "scale_factor": 1}}}]}}, "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}}, "chunk_key_encoding": {"name": "default"}, "fill_value": "AAAAAAAAAAA=", "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "zarr_format": 3, "node_type": "array"}`))
	if err != nil {
		t.Fatal(err)
	}
	doc, _ := m.Document()
	if strings.Contains(string(doc), "\\u03bcs") || !strings.Contains(string(doc), `"unit": "us"`) || !strings.Contains(string(doc), `"t": 0`) {
		t.Errorf("written back as\n%s", doc)
	}
}

func TestGroupExtraFields(t *testing.T) {
	ctx := context.Background()
	s := NewMemoryStore()
	s.Set(ctx, "zarr.json", []byte(`{"ext": {"must_understand": false, "x": 1}, "zarr_format": 3, "node_type": "group", "attributes": {"a": 1}}`))
	g, err := OpenGroup(ctx, s, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := g.SetAttributes(ctx, map[string]any{"a": 1, "b": 2}); err != nil {
		t.Fatal(err)
	}
	// What zarr-python 3.4.1 stores after g.attrs["b"] = 2.
	want := "{\n  \"attributes\": {\n    \"a\": 1,\n    \"b\": 2\n  },\n  \"zarr_format\": 3,\n  \"node_type\": \"group\",\n  \"ext\": {\n    \"must_understand\": false,\n    \"x\": 1\n  }\n}"
	if got, _, _ := s.Get(ctx, "zarr.json", ByteRange{}); string(got) != want {
		t.Errorf("got\n%s", got)
	}
	if _, err := ConsolidateMetadata(ctx, s, ""); err != nil {
		t.Fatal(err)
	}
	if got, _, _ := s.Get(ctx, "zarr.json", ByteRange{}); !strings.HasSuffix(string(got), "\"node_type\": \"group\",\n  \"ext\": {\n    \"must_understand\": false,\n    \"x\": 1\n  }\n}") {
		t.Errorf("consolidated:\n%s", got)
	}
	for _, extra := range []string{`1`, `[1]`, `{"must_understand": true}`, `{"x": 1}`} {
		s.Set(ctx, "zarr.json", []byte(`{"zarr_format": 3, "node_type": "group", "attributes": {}, "ext": `+extra+`}`))
		if _, err := OpenGroup(ctx, s, "", nil); !errors.Is(err, ErrUnsupported) {
			t.Errorf("extra field %s: %v", extra, err)
		}
	}
}

// Overwriting a node with one that cannot be stored leaves it intact.
func TestCreateGroupValidatesBeforeOverwriting(t *testing.T) {
	ctx := context.Background()
	s := NewMemoryStore()
	if _, err := CreateGroup(ctx, s, "g", GroupSpec{Attributes: map[string]any{"keep": true}}, nil); err != nil {
		t.Fatal(err)
	}
	_, err := CreateGroup(ctx, s, "g", GroupSpec{Attributes: map[string]any{"bad": make(chan int)}}, &CreateOptions{Overwrite: true})
	if err == nil {
		t.Fatal("unstorable attributes accepted")
	}
	g, err := OpenGroup(ctx, s, "g", nil)
	if err != nil || g.Attributes()["keep"] != true {
		t.Fatalf("existing group lost: %v", err)
	}
}

// A shard too short to hold its index is an error on every path, also
// when empty (zarr-python 3.4.1).
func TestTruncatedShards(t *testing.T) {
	ctx := context.Background()
	for _, loc := range []string{"end", "start"} {
		s := NewMemoryStore()
		a, err := CreateArray(ctx, s, "a", ArraySpec{Shape: []uint64{8}, ChunkShape: []uint64{2}, ShardShape: []uint64{4}, IndexLocation: loc, DataType: "int16", Compressors: []CodecSpec{}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		v, _ := FromSlice([]uint64{8}, []int16{1, 2, 3, 4, 5, 6, 7, 8})
		if err := a.Set(ctx, v); err != nil {
			t.Fatal(err)
		}
		for _, size := range []int{0, 7} {
			s.Set(ctx, "a/c/0", make([]byte, size))
			one, _ := FromSlice([]uint64{1}, []int16{9})
			if _, err := a.ReadAll(ctx); err == nil {
				t.Errorf("%s/%d: full read succeeded", loc, size)
			}
			if _, err := a.Get(ctx, Index(1)); err == nil {
				t.Errorf("%s/%d: partial read succeeded", loc, size)
			}
			if err := a.Set(ctx, one, Index(1)); err == nil {
				t.Errorf("%s/%d: partial write succeeded", loc, size)
			}
		}
	}
}

// Explicit chunk edges declare a varying dimension even when uniform, so
// block slices clamp as in zarr-python 3.4.1.
func TestBlockSelectionUniformEdges(t *testing.T) {
	a, err := NewArray(rectilinearDoc("[7, 1]", "[[3, 4], [1]]"), &mapChunks{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := a.GetBlocks(context.Background(), Slice(0, 1), Slice(-3, 3))
	if err != nil || !slices.Equal(got.Shape, []uint64{3, 1}) {
		t.Fatalf("%v %v", got, err)
	}
	got, err = a.GetBlocks(context.Background(), Slice(0, 1), Slice(-7, -3))
	if err != nil || !slices.Equal(got.Shape, []uint64{3, 0}) {
		t.Fatalf("%v %v", got, err)
	}
}
