package zarr_test

import (
	"context"
	"fmt"
	"log"
	"time"

	zarr "github.com/solomonsealed/zarr-go"
)

func Example() {
	ctx := context.Background()
	store := zarr.NewMemoryStore() // or NewLocalStore(dir), NewS3Store(...), NewHTTPStore(...)

	arr, err := zarr.CreateArray(ctx, store, "temperature", zarr.ArraySpec{
		Shape:          []uint64{4, 6},
		ChunkShape:     []uint64{2, 3},
		DataType:       "float32",
		FillValue:      "NaN",
		DimensionNames: []string{"y", "x"},
		Attributes:     map[string]any{"units": "K"},
	}, nil)
	if err != nil {
		log.Fatal(err)
	}
	vals := make([]float32, 24)
	for i := range vals {
		vals[i] = float32(i)
	}
	data, _ := zarr.FromSlice([]uint64{4, 6}, vals)
	if err := arr.Set(ctx, data); err != nil {
		log.Fatal(err)
	}

	// Every other row, columns 5 and 0.
	sel, _ := arr.Get(ctx, zarr.SliceStep(0, zarr.End, 2), zarr.Indices(5, 0))
	v, _ := sel.Values()
	fmt.Println(sel.Shape, v)
	// Output: [2 2] [5 0 17 12]
}

func ExampleArray_GetCoordinates() {
	ctx := context.Background()
	arr, _ := zarr.CreateArray(ctx, zarr.NewMemoryStore(), "a", zarr.ArraySpec{Shape: []uint64{3, 3}, DataType: "int16"}, nil)
	d, _ := zarr.FromSlice([]uint64{3, 3}, []int16{0, 1, 2, 3, 4, 5, 6, 7, 8})
	arr.Set(ctx, d)

	// Points (0, 2), (2, 0) and (-1, -1).
	pts, _ := arr.GetCoordinates(ctx, []int64{0, 2, -1}, []int64{2, 0, -1})
	v, _ := pts.Values()
	fmt.Println(v)
	// Output: [2 6 8]
}

func ExampleArray_Set_sharded() {
	ctx := context.Background()
	// 8x8 chunks stored in 64x64 shards. Writing one element re-encodes
	// only the inner chunk holding it.
	arr, _ := zarr.CreateArray(ctx, zarr.NewMemoryStore(), "a", zarr.ArraySpec{
		Shape: []uint64{128, 128}, ChunkShape: []uint64{8, 8}, ShardShape: []uint64{64, 64}, DataType: "uint8",
	}, nil)
	one, _ := zarr.FromSlice([]uint64{}, []uint8{42})
	arr.Set(ctx, one, zarr.Index(70), zarr.Index(3))
	got, _ := arr.Get(ctx, zarr.Slice(69, 72), zarr.Index(3))
	v, _ := got.Values()
	fmt.Println(arr.Chunks(), arr.Shards(), v)
	// Output: [8 8] [64 64] [0 42 0]
}

func ExampleFromValues() {
	// Types without a Go equivalent are built from Go values.
	t0 := time.Date(2024, 3, 1, 12, 0, 0, 0, time.UTC)
	times, _ := zarr.FromValues([]uint64{2}, zarr.Datetime64("s", 1), []time.Time{t0, t0.Add(time.Hour)})
	half, _ := zarr.FromValues([]uint64{3}, zarr.MustDataType("bfloat16"), []float64{1, 0.1, -2})
	v, _ := half.Values()
	fmt.Println(times.Data[:8], v)
	// Output: [64 195 225 101 0 0 0 0] [1 0.100097656 -2]
}

func ExampleOpenGroup() {
	ctx := context.Background()
	store := zarr.NewMemoryStore()
	root, _ := zarr.CreateGroup(ctx, store, "", zarr.GroupSpec{Attributes: map[string]any{"title": "demo"}}, nil)
	root.CreateArray(ctx, "obs/temp", zarr.ArraySpec{Shape: []uint64{10}, DataType: "float64"}, nil)
	zarr.ConsolidateMetadata(ctx, store, "")

	g, _ := zarr.OpenGroup(ctx, store, "", nil)
	g.Walk(ctx, func(path string, n zarr.Node) error {
		fmt.Printf("%s %T\n", path, n)
		return nil
	})
	fmt.Println(g.Consolidated())
	// Output:
	// obs *zarr.Group
	// obs/temp *zarr.Array
	// true
}

func ExampleArraySpec_Metadata() {
	doc, _ := zarr.ArraySpec{ZarrFormat: 2, Shape: []uint64{4}, ChunkShape: []uint64{2}, DataType: ">i2",
		Compressors: []zarr.CodecSpec{zarr.CodecJSON("zlib", `{"level": 3}`)}}.Metadata()
	fmt.Println(string(doc))
	// Output:
	// {
	//   "shape": [
	//     4
	//   ],
	//   "chunks": [
	//     2
	//   ],
	//   "dtype": ">i2",
	//   "fill_value": 0,
	//   "order": "C",
	//   "filters": null,
	//   "dimension_separator": ".",
	//   "compressor": {
	//     "id": "zlib",
	//     "level": 3
	//   },
	//   "zarr_format": 2
	// }
}
