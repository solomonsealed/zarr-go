package consistency

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

// TestTensorStore checks the zarr-extensions data types zarr-python lacks
// against TensorStore's zarr3 driver: TensorStore reads arrays Go wrote,
// Go reads arrays TensorStore wrote, and the chunks are identical.
func TestTensorStore(t *testing.T) {
	o := pythonOracle(t)
	if err := o.call("ts_read", map[string]any{"store": t.TempDir()}, nil); err != nil && err.(*pyError).Type == "ModuleNotFoundError" {
		t.Skip("tensorstore is not installed")
	}
	types := []string{"bfloat16", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e4m3b11fnuz", "float8_e5m2", "float8_e5m2fnuz", "float8_e3m4", "float8_e8m0fnu", "float4_e2m1fn", "int4", "int2"}
	ctx := context.Background()
	rng := rand.New(rand.NewSource(5))
	for _, name := range types {
		for _, sharded := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/sharded=%v", name, sharded), func(t *testing.T) {
				dt := zarr.MustDataType(name)
				spec := zarr.ArraySpec{Shape: []uint64{9, 7}, ChunkShape: []uint64{4, 3}, Type: dt, Compressors: []zarr.CodecSpec{}}
				if sharded {
					spec.ShardShape = []uint64{8, 6}
				}
				goDir, tsDir := t.TempDir(), t.TempDir()
				ga, err := zarr.CreateArray(ctx, zarr.NewLocalStore(goDir), "a", spec, nil)
				if err != nil {
					t.Fatal(err)
				}
				ta, err := zarr.CreateArray(ctx, zarr.NewLocalStore(tsDir), "a", spec, nil)
				if err != nil {
					t.Fatal(err)
				}
				data := &zarr.NDArray{Shape: spec.Shape, DataType: dt, Data: make([]byte, 63*dt.Size)}
				rng.Read(data.Data)
				// Canonicalize: sub-byte types keep their value in the low bits.
				if dt.Bits != 0 {
					for i := range data.Data {
						data.Data[i] &= 1<<dt.Bits - 1
					}
				}
				if err := ga.Write(ctx, []uint64{0, 0}, data); err != nil {
					t.Fatal(err)
				}
				if err := o.call("ts_write", map[string]any{"store": tsDir, "path": "a", "data": toWire(data)}, nil); err != nil {
					t.Fatal(err)
				}
				// Inner chunk order within shards is up to the writer
				// (TensorStore's is not zarr-python's Morton order), so
				// sharded arrays are compared by reading each other's.
				if !sharded {
					compareDirs(t, tsDir, goDir, "after writing")
				}
				var w wireArray
				if err := o.call("ts_read", map[string]any{"store": goDir, "path": "a"}, &w); err != nil {
					t.Fatal(err)
				}
				got, _ := fromWire(w, dt)
				if err := sameArray(got, data); err != nil {
					t.Fatalf("tensorstore reading go's array: %v", err)
				}
				back, err := ta.ReadAll(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if err := sameArray(back, data); err != nil {
					t.Fatalf("go reading tensorstore's array: %v", err)
				}
			})
		}
	}
}
