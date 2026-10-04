package zarr

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestParseSelection(t *testing.T) {
	shape := []uint64{10, 20, 30}
	for _, tc := range []struct {
		sel          string
		start, count []uint64
		squeeze      []bool
	}{
		{"", []uint64{0, 0, 0}, []uint64{10, 20, 30}, []bool{false, false, false}},
		{"2:5", []uint64{2, 0, 0}, []uint64{3, 20, 30}, []bool{false, false, false}},
		{"3, :, -5:", []uint64{3, 0, 25}, []uint64{1, 20, 5}, []bool{true, false, false}},
		{"[:4, 19]", []uint64{0, 19, 0}, []uint64{4, 1, 30}, []bool{false, true, false}},
		{"8:100", []uint64{8, 0, 0}, []uint64{2, 20, 30}, []bool{false, false, false}},
		{"-1", []uint64{9, 0, 0}, []uint64{1, 20, 30}, []bool{true, false, false}},
		{"..., 4", []uint64{0, 0, 4}, []uint64{10, 20, 1}, []bool{false, false, true}},
	} {
		s, c, q, err := ParseSelection(tc.sel, shape)
		if err != nil || !slices.Equal(s, tc.start) || !slices.Equal(c, tc.count) || !slices.Equal(q, tc.squeeze) {
			t.Errorf("ParseSelection(%q) = %v %v %v %v", tc.sel, s, c, q, err)
		}
	}
	for _, bad := range []string{"10", "1,2,3,4", "1:5:3", "x", "[1,2]"} {
		if _, _, _, err := ParseSelection(bad, shape); err == nil {
			t.Errorf("ParseSelection(%q) accepted", bad)
		}
	}
}

func TestParseSelector(t *testing.T) {
	sel, err := ParseSelector("::2, [3, -1, 3], [true, false], ..., -2")
	if err != nil {
		t.Fatal(err)
	}
	want := []DimSelector{SliceSelector{0, End, 2}, IndicesSelector{3, -1, 3}, MaskSelector{true, false}, Ellipsis, IndexSelector(-2)}
	if !reflect.DeepEqual(sel, want) {
		t.Errorf("got %#v", sel)
	}
	for _, bad := range []string{"1:2:3:4", "[1, true]", "[1", "a"} {
		if _, err := ParseSelector(bad); err == nil {
			t.Errorf("ParseSelector(%q) accepted", bad)
		}
	}
}

func TestHalfFloat(t *testing.T) {
	for _, f := range []float32{0, 1, -2.5, 65504, 6.103515625e-05, 5.960464477539063e-08} {
		if got := halfToFloat32(float32ToHalf(f)); got != f {
			t.Errorf("half round trip %v -> %v", f, got)
		}
	}
	if !math.IsNaN(float64(halfToFloat32(0x7e00))) || !math.IsInf(float64(halfToFloat32(0xfc00)), -1) {
		t.Error("half NaN/Inf")
	}
	for v, want := range map[float64]uint16{0.7: 0x399a, 65520: 0x7c00, 65519: 0x7bff, -0.1: 0xae66, 1e-8: 0} {
		if got := float64ToHalf(v); got != want {
			t.Errorf("float64ToHalf(%v) = %#04x, want %#04x", v, got, want)
		}
	}
}

func TestPythonFloatRepr(t *testing.T) {
	for f, want := range map[float64]string{0: "0.0", 1: "1.0", 0.1: "0.1", 1e16: "1e+16", 1234567890123456: "1234567890123456.0",
		1e-5: "1e-05", 0.0001: "0.0001", 1.5e300: "1.5e+300", 5e-324: "5e-324", 123.456: "123.456", -2.5: "-2.5"} {
		if got := pythonFloat(f); got != want {
			t.Errorf("pythonFloat(%v) = %s, want %s", f, got, want)
		}
	}
	if got := pythonFloat(math.Copysign(0, -1)); got != "-0.0" {
		t.Errorf("-0.0 -> %s", got)
	}
	b, _ := marshalPython(jsonObject{{"a", []any{1, "é😀\n"}}, {"b", jsonObject{}}}, 2)
	if want := "{\n  \"a\": [\n    1,\n    \"\\u00e9\\ud83d\\ude00\\n\"\n  ],\n  \"b\": {}\n}"; string(b) != want {
		t.Errorf("marshalPython = %s", b)
	}
}

func TestNDArrayJSON(t *testing.T) {
	n, _ := FromSlice([]uint64{3}, []float64{0, math.NaN(), 2.5})
	b, err := json.Marshal(n)
	if err != nil || string(b) != `{"shape":[3],"dtype":"float64","data":[0,null,2.5]}` {
		t.Errorf("json = %s, %v", b, err)
	}
}

// A panicking chunk source becomes an error instead of crashing the program.
type panickySource struct{}

func (panickySource) GetChunk(context.Context, []uint64, ByteRange) ([]byte, bool, error) {
	panic("boom")
}

func TestPanicInChunkReadIsAnError(t *testing.T) {
	doc := []byte(`{"zarr_format":3,"node_type":"array","shape":[4],"data_type":"int8",
		"chunk_grid":{"name":"regular","configuration":{"chunk_shape":[2]}},
		"chunk_key_encoding":{"name":"default"},"fill_value":0,"codecs":[{"name":"bytes"}]}`)
	arr, err := NewArray(doc, panickySource{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := arr.ReadAll(context.Background()); err == nil || !strings.Contains(err.Error(), "boom") {
		t.Errorf("err = %v, want the panic as an error", err)
	}
}

// Attributes keep integers exact, as Python does, and accept Python's
// bare NaN.
func TestAttributeNumbers(t *testing.T) {
	attrs, err := Attributes([]byte(`{"attributes": {"big": 4611686018427387904, "neg": -5, "f": 1.5,
		"e": 1e3, "u": 18446744073709551615, "huge": 123456789012345678901234567890, "list": [1, {"x": 2.0}], "nan": NaN}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(attrs["nan"].(float64)) {
		t.Errorf("nan = %v", attrs["nan"])
	}
	delete(attrs, "nan")
	want := map[string]any{
		"big": int64(4611686018427387904), "neg": int64(-5), "f": 1.5, "e": 1000.0,
		"u": uint64(18446744073709551615), "huge": json.Number("123456789012345678901234567890"),
		"list": []any{int64(1), map[string]any{"x": 2.0}},
	}
	if !reflect.DeepEqual(attrs, want) {
		t.Errorf("attributes = %#v", attrs)
	}
}

type celsius float32

func TestFromValues(t *testing.T) {
	// Named types.
	n, err := FromSlice([]uint64{2}, []celsius{1.5, -2})
	if err != nil || n.DataType.Name != "float32" {
		t.Fatalf("FromSlice(named) = %v, %v", n, err)
	}
	// float16 and bfloat16 from float64.
	for _, name := range []string{"float16", "bfloat16", "float8_e4m3fn"} {
		n, err := FromValues([]uint64{3}, MustDataType(name), []float64{1, -0.5, 2})
		if err != nil {
			t.Fatal(err)
		}
		vals, _ := n.Values()
		if !reflect.DeepEqual(vals, []float32{1, -0.5, 2}) {
			t.Errorf("%s values = %v", name, vals)
		}
	}
	// Sub-byte integers keep the low bits.
	n, _ = FromValues([]uint64{3}, MustDataType("int4"), []int{-1, 7, -8})
	if !slices.Equal(n.Data, []byte{0x0f, 0x07, 0x08}) {
		t.Errorf("int4 data = % x", n.Data)
	}
	if vals, _ := n.Values(); !reflect.DeepEqual(vals, []int8{-1, 7, -8}) {
		t.Errorf("int4 values = %v", vals)
	}
	if _, err := FromValues([]uint64{1}, MustDataType("int4"), []int{8}); err == nil {
		t.Error("int4 accepted 8")
	}
	// Datetimes from time.Time, timedeltas from Duration.
	ts := time.Date(2020, 1, 2, 3, 4, 5, 0, time.UTC)
	n, err = FromValues([]uint64{2}, Datetime64("s", 1), []time.Time{ts, {}})
	if err != nil {
		t.Fatal(err)
	}
	if vals, _ := n.Values(); !reflect.DeepEqual(vals, []int64{ts.Unix(), math.MinInt64}) {
		t.Errorf("datetime values = %v", vals)
	}
	times, _ := n.Times()
	if !times[0].Equal(ts) || !times[1].IsZero() {
		t.Errorf("Times = %v", times)
	}
	n, err = FromValues([]uint64{1}, Timedelta64("ms", 10), []time.Duration{50 * time.Millisecond})
	if err != nil || n.Data[0] != 5 {
		t.Errorf("timedelta = % x, %v", n.Data, err)
	}
	// Fixed-length strings and bytes, structs.
	n, _ = FromValues([]uint64{2}, FixedLengthUTF32(3), []string{"héé", "a"})
	if vals, _ := n.Values(); !reflect.DeepEqual(vals, []string{"héé", "a"}) {
		t.Errorf("utf32 values = %v", vals)
	}
	if _, err := FromValues([]uint64{1}, FixedLengthUTF32(2), []string{"abc"}); err == nil {
		t.Error("too long string accepted")
	}
	st := Struct(StructField{Name: "a", Type: MustDataType("int16")}, StructField{Name: "b", Type: MustDataType("float32")})
	n, err = FromValues([]uint64{2}, st, []map[string]any{{"a": 3, "b": 1.5}, {"b": -1}})
	if err != nil {
		t.Fatal(err)
	}
	f, _ := n.Field("a")
	if vals, _ := f.Values(); !reflect.DeepEqual(vals, []int16{3, 0}) {
		t.Errorf("field a = %v", vals)
	}
	vals, _ := n.Values()
	if got := vals.([]map[string]any)[0]["b"]; got != float32(1.5) {
		t.Errorf("struct values = %v", vals)
	}
}

func TestDataTypeJSON(t *testing.T) {
	for _, s := range []string{`"bool"`, `"int4"`, `"bfloat16"`, `"complex_float8_e4m3fn"`, `"r16"`,
		`{"name": "numpy.datetime64", "configuration": {"unit": "ns", "scale_factor": 1}}`,
		`{"name": "struct", "configuration": {"fields": [{"name": "a", "data_type": "int8"}]}}`} {
		dt, err := ParseDataTypeJSON(json.RawMessage(s))
		if err != nil {
			t.Fatalf("%s: %v", s, err)
		}
		b, _ := marshalPython(dt.v3JSON(), -1)
		if string(b) != s {
			t.Errorf("%s rendered as %s", s, b)
		}
	}
	if _, err := ParseDataType("float7"); err == nil {
		t.Error("float7 accepted")
	}
	for _, s := range []string{`"<f8"`, `">i2"`, `"|b1"`, `"<U4"`, `"|S3"`, `"|V2"`, `"<M8[10ms]"`, `[["a", "<i4"], ["b", "|u1"]]`} {
		dt, big, err := parseV2DataType(json.RawMessage(s), "")
		if err != nil {
			t.Fatalf("%s: %v", s, err)
		}
		v, _ := dt.v2JSON(big)
		b, _ := marshalPython(v, -1)
		if string(b) != s {
			t.Errorf("%s rendered as %s", s, b)
		}
	}
}

func TestMortonOrder(t *testing.T) {
	got := mortonOrder([]uint64{3, 2})
	want := [][]uint64{{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 0}, {2, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("3x2 = %v", got)
	}
	got = mortonOrder([]uint64{2, 3, 2})
	want = [][]uint64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {0, 2, 0}, {1, 2, 0}, {0, 2, 1}, {1, 2, 1}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("2x3x2 = %v", got)
	}
	// The sparse path (sorting codes) agrees with the dense one.
	for _, g := range [][]uint64{{100, 1}, {1, 9}, {33}, {5, 7, 3}} {
		o := mortonOrder(g)
		if uint64(len(o)) != numElements(g) {
			t.Errorf("%v: %d coordinates", g, len(o))
		}
	}
}
