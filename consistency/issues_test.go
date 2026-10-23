package consistency

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

// TestKnownPythonIssues reproduces the zarr-python 3.4 / numcodecs 0.17
// (and zarr-template) issues the consistency tests ran into and work
// around, and checks that
// Go handles each case correctly. If an issue no longer occurs upstream,
// the test fails so the workaround can be removed and the case covered.
func TestKnownPythonIssues(t *testing.T) {
	o := pythonOracle(t)
	ctx := context.Background()
	issues := []struct {
		name, what string
		goCheck    func(t *testing.T)
	}{
		{"vlen_bytes_single_write", "writing one element of a variable_length_bytes array raises TypeError", func(t *testing.T) {
			a := newArray(t, zarr.ArraySpec{Shape: []uint64{3}, DataType: "variable_length_bytes"})
			v, _ := zarr.FromValues([]uint64{}, a.DataType(), [][]byte{[]byte("x")})
			must(t, a.Set(ctx, v, zarr.Index(0)))
			got, _ := a.Get(ctx, zarr.Index(0))
			if got.Strings[0] != "x" {
				t.Fatalf("got %q", got.Strings[0])
			}
		}},
		{"vlen_bytes_nested", "reading one element of a sharded variable_length_bytes array returns it inside nested 0-d object arrays", func(t *testing.T) {
			a := newArray(t, zarr.ArraySpec{Shape: []uint64{2}, ChunkShape: []uint64{1}, ShardShape: []uint64{2}, DataType: "variable_length_bytes"})
			v, _ := zarr.FromValues([]uint64{2}, a.DataType(), []string{"a\x00", "b"})
			must(t, a.Set(ctx, v))
			got, _ := a.Get(ctx, zarr.Index(0))
			if len(got.Shape) != 0 || got.Strings[0] != "a\x00" {
				t.Fatalf("got %v %q", got.Shape, got.Strings)
			}
		}},
		{"generic_time_big_endian", "chunks of a big-endian generic timedelta64 array (\">m8[2generic]\") that hold only NaT are stored instead of deleted", func(t *testing.T) {
			s := zarr.NewMemoryStore()
			a, err := zarr.CreateArray(ctx, s, "a", zarr.ArraySpec{ZarrFormat: 2, Shape: []uint64{4}, ChunkShape: []uint64{2}, DataType: ">m8[2generic]",
				FillValue: json.RawMessage("null"), Compressors: []zarr.CodecSpec{}}, nil)
			must(t, err)
			v, _ := zarr.FromValues([]uint64{4}, a.DataType(), []int64{1, 2, math.MinInt64, math.MinInt64})
			must(t, a.Set(ctx, v))
			if _, ok, _ := s.Get(ctx, "a/1", zarr.ByteRange{}); ok {
				t.Fatal("go stored a chunk holding only NaT")
			}
		}},
		{"template_dimensions", "zarr-template 0.1.0 accepts a chunk key template that does not name every coordinate (\"{0}\" on a 2-d array), giving several chunks one key", func(t *testing.T) {
			_, err := zarr.CreateArray(ctx, zarr.NewMemoryStore(), "a", zarr.ArraySpec{Shape: []uint64{2, 4}, ChunkShape: []uint64{1, 2}, DataType: "int8",
				ChunkKeyEncoding: zarr.ChunkKeyEncoding{Name: "template", Template: "{0}"}}, nil)
			if err == nil {
				t.Fatal("go accepted the template")
			}
		}},
		{"quantize_astype", "numcodecs.quantize with an astype other than its dtype cannot be read back (the v3 wrapper does not propagate the type)", func(t *testing.T) {
			a := newArray(t, zarr.ArraySpec{Shape: []uint64{4}, DataType: "float32", Compressors: []zarr.CodecSpec{},
				Filters: []zarr.CodecSpec{zarr.CodecJSON("numcodecs.quantize", `{"digits": 1, "dtype": "<f4", "astype": "<f2"}`)}})
			v, _ := zarr.FromSlice([]uint64{4}, []float32{0, 1.25, 2, 3})
			must(t, a.Set(ctx, v))
			got, _ := a.ReadAll(ctx)
			vals, _ := got.Values()
			if vals.([]float32)[1] != 1.25 {
				t.Fatalf("got %v", vals)
			}
		}},
		{"bitround_f_order", "numcodecs.BitRound on Zarr v2 F-order arrays scrambles the data (its copy() is C-ordered)", func(t *testing.T) {
			a := newArray(t, zarr.ArraySpec{ZarrFormat: 2, Shape: []uint64{4, 3}, DataType: "<f4", Order: "F", Compressors: []zarr.CodecSpec{},
				Filters: []zarr.CodecSpec{zarr.CodecJSON("bitround", `{"keepbits": 10}`)}})
			x := make([]float32, 12)
			for i := range x {
				x[i] = float32(i)
			}
			v, _ := zarr.FromSlice([]uint64{4, 3}, x)
			must(t, a.Set(ctx, v))
			got, _ := a.ReadAll(ctx)
			if !bytes.Equal(got.Data, v.Data) {
				t.Fatalf("round trip changed the data")
			}
		}},
		{"open_array_config", "zarr.open_array ignores its config argument (write_empty_chunks stays False); Array.with_config works", nil},
		{"ellipsis_with_array", "an Ellipsis after an integer array in an orthogonal selection raises ValueError (replace_ellipsis compares with ==)", func(t *testing.T) {
			a := newArray(t, zarr.ArraySpec{Shape: []uint64{3, 3}, DataType: "int8"})
			got, err := a.Get(ctx, zarr.Indices(0, 2), zarr.Ellipsis)
			if err != nil || len(got.Shape) != 2 || got.Shape[0] != 2 {
				t.Fatalf("got %v, %v", got, err)
			}
		}},
		{"empty_buffers", "numcodecs LZ4, Zstd and Blosc cannot decode their own output for empty buffers, and Fletcher32 cannot encode one", func(t *testing.T) {
			for _, spec := range []string{`{"name": "numcodecs.lz4"}`, `{"name": "numcodecs.zstd"}`, `{"name": "numcodecs.blosc"}`, `{"name": "numcodecs.fletcher32"}`} {
				var cs zarr.CodecSpec
				must(t, cs.UnmarshalJSON([]byte(spec)))
				c, err := zarr.NewCodec(cs, zarr.ChunkSpec{DataType: zarr.MustDataType("uint8")})
				must(t, err)
				bc := c.(zarr.BytesToBytesCodec)
				enc, err := bc.Encode(nil)
				must(t, err)
				if dec, err := bc.Decode(enc); err != nil || len(dec) != 0 {
					t.Fatalf("%s: %v", spec, err)
				}
			}
		}},
	}
	for _, is := range issues {
		t.Run(is.name, func(t *testing.T) {
			var res struct {
				Occurs bool   `json:"occurs"`
				Detail string `json:"detail"`
			}
			if err := o.call("known_issue", map[string]any{"name": is.name, "store": t.TempDir()}, &res); err != nil {
				t.Fatal(err)
			}
			if !res.Occurs {
				t.Errorf("fixed upstream? %s no longer occurs; remove its workaround in the tests", is.what)
			} else {
				t.Logf("zarr-python: %s (%s)", is.what, res.Detail)
			}
			if is.goCheck != nil {
				is.goCheck(t)
			}
		})
	}
}

func newArray(t *testing.T, spec zarr.ArraySpec) *zarr.Array {
	t.Helper()
	a, err := zarr.CreateArray(context.Background(), zarr.NewMemoryStore(), "a", spec, nil)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}
