# zarr-go

A pure-Go implementation of [Zarr](https://zarr.dev), formats 2 and 3: arrays
and groups, every data type and codec zarr-python 3 supports plus the
zarr-extensions ones, sharding with partial reads and writes, and NumPy-style
selections. It has no cgo, and builds for `GOOS=js GOARCH=wasm` and
`wasip1`.

It is checked against zarr-python (3.4.1), both ways: the same operations
through both implementations produce byte-identical metadata, the same
stored keys, byte-identical chunks wherever the codecs are deterministic,
and the same values on read. See [consistency](consistency/README.md).

```sh
go get github.com/solomonsealed/zarr-go
```

```go
import zarr "github.com/solomonsealed/zarr-go"
```

## Quick start

```go
ctx := context.Background()
store := zarr.NewLocalStore("data.zarr") // or httpstore.NewS3(...), zarr.NewMemoryStore(), ...

arr, err := zarr.CreateArray(ctx, store, "temperature", zarr.ArraySpec{
	Shape:      []uint64{365, 720, 1440},
	ChunkShape: []uint64{1, 180, 180},
	ShardShape: []uint64{30, 720, 1440}, // optional
	DataType:   "float32",
	FillValue:  "NaN",
	DimensionNames: []string{"time", "lat", "lon"},
	Attributes: map[string]any{"units": "K"},
}, nil)

// Write a region (data is an *NDArray; see FromSlice and FromValues).
err = arr.Write(ctx, []uint64{0, 0, 0}, day)

// Read every 4th latitude of day 10, and two longitudes.
sub, err := arr.Get(ctx, zarr.Index(10), zarr.SliceStep(0, zarr.End, 4), zarr.Indices(0, 720))
vals, _ := sub.Values() // []float32

// Open an existing array or group (format 2 or 3).
arr, err = zarr.OpenArray(ctx, store, "temperature", nil)
root, err := zarr.OpenGroup(ctx, store, "", nil)
```

## Features

**Formats.** Zarr v3 (`zarr.json`) and v2 (`.zarray`, `.zattrs`, `.zgroup`),
including v2 `order: "F"`, `dimension_separator`, big-endian dtypes, `null`
fill values and object arrays with `vlen-utf8`/`vlen-bytes`. Metadata is
written exactly as zarr-python writes it, key order and formatting included.

**Data types.** `bool`, `int8`–`int64`, `uint8`–`uint64`, `float16/32/64`,
`complex64/128`, `string`, `variable_length_bytes`, `fixed_length_utf32`,
`null_terminated_bytes`, `raw_bytes`, `rN`, `numpy.datetime64` and
`numpy.timedelta64` (any unit and scale factor), `struct` (and the legacy
`structured`), and the zarr-extensions types `bfloat16`, `float8_*`,
`float6_*`, `float4_e2m1fn`, `int2`, `int4`, `uint2`, `uint4` and
`complex_*`. Narrow float conversions match ml_dtypes bit for bit.

**Codecs** (all encode and decode):

| | |
|---|---|
| Array to array | `transpose`, `scale_offset`, `cast_value`, `numcodecs.delta`, `fixedscaleoffset`, `quantize`, `astype`, `packbits`, `bitround`, and `categorize` (v2) |
| Array to bytes | `bytes` (both endians), `vlen-utf8`, `vlen-bytes`, `sharding_indexed` |
| Bytes to bytes | `zstd`, `gzip`, `blosc` (blosclz, lz4, lz4hc, snappy, zlib, zstd; byte and bit shuffle), `crc32c`, and `numcodecs.zlib`, `gzip`, `bz2`, `lzma`, `lz4`, `zstd`, `blosc`, `shuffle`, `crc32`, `crc32c`, `adler32`, `fletcher32`, `jenkins_lookup3`, `base64` |

Blosc (blosclz, lz4, lz4hc), LZ4 and bz2 are ports of the C code numcodecs
uses and produce identical bytes. zstd, gzip, zlib and lzma interoperate with
the C libraries but compress differently. Add codecs of any kind with
`RegisterCodec`.

**Sharding.** Reads fetch the shard index with one ranged request, then only
the inner chunks a selection needs, coalescing nearby ones into one request
as zarr-python does. Writes decode and re-encode only the inner chunks they
touch, copy the others' bytes, and lay the shard out in zarr-python's Morton
order. Nested sharding, `index_location: "start"` and any fixed-size index
codecs work.

**Selections.** Every zarr-python selection kind, for reading and writing:

| zarr-python | Go |
|---|---|
| `a[1, 2:8:2, :]`, `a.oindex[[3, 0], mask]` | `a.Get(ctx, zarr.Index(1), zarr.SliceStep(2, 8, 2), zarr.All())`, `a.Get(ctx, zarr.Indices(3, 0), zarr.Mask(m...))` |
| `a.vindex[[0, 2], [1, 3]]` | `a.GetCoordinates(ctx, []int64{0, 2}, []int64{1, 3})` |
| `a.vindex[mask_array]` | `a.GetMask(ctx, mask)` |
| `a.blocks[1, :2]` | `a.GetBlocks(ctx, zarr.Index(1), zarr.Slice(0, 2))` |
| `a[...] = v` | `a.Set(ctx, v)`, `SetCoordinates`, `SetMask`, `SetBlocks` |

Negative indices, steps, unordered and repeated indices, `Ellipsis` and
NumPy broadcasting of written values work as in zarr-python.
`ParseSelector` parses the same syntax from strings (`"::2, [3, -1], ..."`).

**Writes** follow zarr-python: partly covered chunks are read, merged and
rewritten; chunks that end up holding only the fill value are deleted (or
kept, with `Options.WriteEmptyChunks`); `Resize` and `Append` delete chunks
outside the new shape.

**Chunk grids and keys.** Regular and rectilinear (variable chunk sizes)
grids. The `default` and `v2` chunk key encodings, and the zarr-extensions
[`template`](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/template)
encoding, which builds keys from a template such as `"{-1}/{*:03}.tif"`
(`ChunkKeyEncoding{Name: "template", Template: ..., Separator: "_"}`).

**Metadata other programs wrote.** Invalid metadata that zarr-python 3.4.1
reads leniently is read the same way: chunk sizes stored as 0 or `false` (by
zarr-python 2.18–3.3 for arrays with a zero-length axis) as 1, or the inner
chunk size when sharded; `true` as 1; float rectilinear edges as integers;
and "regular" grids listing chunk edges as rectilinear. When a repair moves
chunks, `Metadata.RepairPending` reports it, and the array stores the
repaired metadata before its first write. Zarr v2 dtypes such as `"<u1"`
(netCDF-C, Zarr.jl, jzarr) are read and written back as `"|u1"`, as
zarr-python does.

**Hierarchies.** `CreateGroup`, `OpenGroup`, `Group.Members`, `Walk`,
attributes, extra fields of Zarr v3 groups (objects with
`"must_understand": false`, kept and written back), and consolidated
metadata in both zarr-python formats (`ConsolidateMetadata`; opening a group
uses it automatically). Creating a node over an existing one with
`Overwrite` validates the new node before deleting the old one.

**Stores.** `MemoryStore`, `LocalStore` (zarr-python's directory layout,
atomic writes), `ZipStore`, and `FSStore` for any `io/fs.FS` such as
`embed.FS`; in the `httpstore` package, `httpstore.NewS3` (AWS, R2, MinIO,
GCS interoperability; SigV4; read, write and list) and `httpstore.New`
(plain HTTP, read-only, with Range requests). They are separate so that
programs not using them, such as WebAssembly builds with their own HTTP
client, do not link `net/http`. A `Store` is a small interface: `Get` with an
optional byte range (including suffix ranges), plus `Set`/`Delete` and
`List`/`ListDir` for writable and listable stores.

## Values

`NDArray` holds an array in C order: fixed-size elements little-endian in
`Data`, variable-length ones in `Strings`. `Values` returns a typed slice
(`[]float32`, `[]int64`, `[]string`, `[]map[string]any` for structs, ...),
`Convert[T]` converts numbers, `Field` extracts a struct field and `Times`
converts datetimes. To build arrays, `FromSlice` takes any Go numeric slice
(named types such as `type Celsius float32` included) and `FromValues` covers
every data type, e.g. `[]time.Time` for datetimes or `[]float64` for bfloat16.

## Other storage systems

`NewArray(doc, src, opts)` opens an array from its metadata document and a
`ChunkSource` that returns chunks by grid coordinates (with byte ranges, for
partial shard reads); a `ChunkWriter` makes it writable. This is how
[icechunk-go](https://github.com/solomonsealed/icechunk-go) serves arrays from
Icechunk repositories.

## Limitations

- zstd, gzip, zlib and lzma output differs from the C libraries'
  (interoperable, verified both ways), so chunks using them are not
  byte-identical with zarr-python's.
- The numcodecs `pcodec` and `zfpy` codecs, and v2 object codecs other than
  `vlen-utf8`/`vlen-bytes` (`json2`, `msgpack2`, `pickle`, `vlen-array`), are
  not built in; `RegisterCodec` can add them. (numcodecs 0.17.1 refuses to
  decode `pickle` by default, as it can run arbitrary code.)
- Zarr v3 metadata is written back as it was stored, with the changes made
  to it and the normalizations zarr-python 3.4.1 applies to the fill value,
  `μs` time units and rectilinear grids; zarr-python re-renders all of it,
  so documents other programs wrote in another key order or with other codec
  defaults may be rewritten differently. Zarr v2 `.zarray` documents are
  rewritten as zarr-python writes them.
- Concurrent writes to the same chunk are not coordinated, as in zarr-python.
- Attributes written from Go maps have sorted keys.
- `ZipStore` keeps written values in memory until `Close`.
