package zarr

import (
	"context"
	"slices"
	"strings"
	"testing"
)

// The examples of the zarr-extensions specification
// (chunk-key-encodings/template/examples.tsv).
var templateExamples = []struct{ template, separator, coords, encoded string }{
	{"c/{*}", "/", "11,22,33", "c/11/22/33"},
	{"{*}", ".", "11,22,33", "11.22.33"},
	{"prefix/{*}", "/", "11,22,33", "prefix/11/22/33"},
	{"{*}.suffix", "_", "11,22,33", "11_22_33.suffix"},
	{"{2}/{0}/{1}", ".", "11,22,33", "33/11/22"},
	{"{-1}/{*}", "/", "11,22,33", "33/11/22"},
	{"{*:04}", "/", "11,22,33", "0011/0022/0033"},
	{"{*:02}", "/", "111,2,33", "111/02/33"},
	{"a{{b{*}}}c", ".", "11,22,33", "a{b11.22.33}c"},
}

func TestTemplateExamples(t *testing.T) {
	for _, ex := range templateExamples {
		var coords []uint64
		for _, c := range strings.Split(ex.coords, ",") {
			var v uint64
			for _, r := range c {
				v = 10*v + uint64(r-'0')
			}
			coords = append(coords, v)
		}
		e := ChunkKeyEncoding{Name: "template", Template: ex.template, Separator: ex.separator}
		if err := e.check(len(coords)); err != nil {
			t.Errorf("%s: %v", ex.template, err)
			continue
		}
		if got := e.Key(coords); got != ex.encoded {
			t.Errorf("%s %v: key %q, want %q", ex.template, coords, got, ex.encoded)
		}
		if got, ok := e.Parse(ex.encoded, len(coords)); !ok || !slices.Equal(got, coords) {
			t.Errorf("%s: parsing %q gave %v %v", ex.template, ex.encoded, got, ok)
		}
	}
	e := ChunkKeyEncoding{Name: "template", Template: "{*:04}", Separator: "/"}
	for _, key := range []string{"011/0022/0033", "0011/0022", "0011/0022/0033/0044", "0011/0022/x"} {
		if c, ok := e.Parse(key, 3); ok {
			t.Errorf("parsed %q as %v", key, c)
		}
	}
}

func TestTemplateValidation(t *testing.T) {
	for _, c := range []struct {
		template, separator string
		ndim                int
	}{
		{"chunks", "", 1},       // no expression
		{"/{0}", "", 1},         // begins with /
		{"a/../{0}", "", 1},     // ..
		{"a//{0}", "", 1},       // //
		{`a\{0}`, "", 1},        // backslash
		{"zarr.json{0}", "", 1}, // zarr.json
		{"{0", "", 1},           // unmatched {
		{"{0}}", "", 1},         // unmatched }
		{"{x}", "", 1},          // not an index
		{"{0:4}", "", 1},        // padding without 0
		{"{0:0}", "", 1},        // padding without width
		{"{-0}", "", 1},         // outside the grammar
		{"{*}/{*}", "/", 2},     // two catch-alls
		{"{0}/{0}", "", 2},      // an index twice
		{"{0}/{-2}", "", 2},     // the same coordinate twice
		{"{*}", "", 2},          // catch-all without a separator
		{"{0}", "", 2},          // a coordinate left out
		{"{0}/{1}", "", 1},      // index past the end
		{"{-2}", "", 1},         // index before the start
		{"{0}/{*}", "/", 1},     // catch-all with nothing left
		{"{*}", ".", 0},         // 0-d: nothing to insert
		{"{0:0999999}", "", 1},  // padding beyond the limit
		{"{99999999999999999999}", "", 1},
	} {
		e := ChunkKeyEncoding{Name: "template", Template: c.template, Separator: c.separator}
		if err := e.check(c.ndim); err == nil {
			t.Errorf("%q (%q) accepted for %d dimensions", c.template, c.separator, c.ndim)
		}
		if k := e.Key(make([]uint64, c.ndim)); k != "" {
			t.Errorf("%q: key %q", c.template, k)
		}
	}
}

func TestTemplateArrays(t *testing.T) {
	ctx := context.Background()
	s := NewMemoryStore()
	spec := ArraySpec{Shape: []uint64{4, 6, 5}, ChunkShape: []uint64{2, 3, 5}, DataType: "int16", Compressors: []CodecSpec{},
		ChunkKeyEncoding: ChunkKeyEncoding{Name: "template", Template: "t/{-1}/{*:02}.bin", Separator: "_"}}
	a, err := CreateArray(ctx, s, "a", spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	doc, _, _ := s.Get(ctx, "a/zarr.json", ByteRange{})
	if !strings.Contains(string(doc), "\"chunk_key_encoding\": {\n    \"name\": \"template\",\n    \"configuration\": {\n      \"format\": \"t/{-1}/{*:02}.bin\",\n      \"separator\": \"_\"\n    }\n  }") {
		t.Errorf("metadata:\n%s", doc)
	}
	v, _ := FromSlice([]uint64{4, 6, 5}, make([]int16, 120))
	for i := range v.Data {
		v.Data[i] = byte(i)
	}
	if err := a.Set(ctx, v); err != nil {
		t.Fatal(err)
	}
	keys, _ := s.List(ctx, "a/t/")
	want := []string{"a/t/0/00_00.bin", "a/t/0/00_01.bin", "a/t/0/01_00.bin", "a/t/0/01_01.bin"}
	if !slices.Equal(keys, want) {
		t.Errorf("keys %v", keys)
	}
	b, err := OpenArray(ctx, s, "a", nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.ReadAll(ctx)
	if err != nil || !slices.Equal(got.Data, v.Data) {
		t.Fatalf("read back: %v", err)
	}

	// Templates must suit the array, and Zarr v2 has its own encoding.
	for _, bad := range []ArraySpec{
		{Shape: []uint64{4, 6}, DataType: "int8", ChunkKeyEncoding: ChunkKeyEncoding{Name: "template", Template: "{0}"}},
		{Shape: []uint64{4}, DataType: "int8", ChunkKeyEncoding: ChunkKeyEncoding{Name: "template", Template: "{*}"}},
		{ZarrFormat: 2, Shape: []uint64{4}, DataType: "<i2", ChunkKeyEncoding: ChunkKeyEncoding{Name: "template", Template: "{0}"}},
	} {
		if _, err := bad.Metadata(); err == nil {
			t.Errorf("accepted %+v", bad.ChunkKeyEncoding)
		}
	}
	doc3 := func(cke string) []byte {
		return []byte(`{"zarr_format": 3, "node_type": "array", "shape": [4], "data_type": "int8", "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}}, "chunk_key_encoding": ` + cke + `, "fill_value": 0, "codecs": [{"name": "bytes"}]}`)
	}
	for _, cke := range []string{
		`{"name": "template", "configuration": {"format": "{0}", "extra": 1}}`,
		`{"name": "template", "configuration": {"separator": "/"}}`,
		`{"name": "template"}`,
		`{"name": "template", "configuration": {"format": 3}}`,
	} {
		if _, err := ParseMetadata(doc3(cke)); err == nil {
			t.Errorf("accepted %s", cke)
		}
	}
	if m, err := ParseMetadata(doc3(`{"name": "template", "configuration": {"format": "x{0}", "separator": null}}`)); err != nil || m.ChunkKey([]uint64{3}) != "x3" {
		t.Errorf("null separator: %v", err)
	}
}
