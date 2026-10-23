# Consistency with zarr-python

The tests in this directory check that this package and
[zarr-python](https://github.com/zarr-developers/zarr-python) (3.4.1, with
numcodecs 0.17.1; see `requirements.txt`) agree, in both directions, on
everything they both support. They drive zarr-python through
`oracle.py`, a small JSON-lines server, apply the same operations with both
implementations, and require that:

- **metadata is byte-identical**: `zarr.json`, `.zarray`, `.zattrs`, `.zgroup`
  and consolidated metadata, as zarr-python formats them;
- **stores hold the same keys** after every operation, so the same chunks
  exist and the same ones were deleted;
- **chunks are byte-identical** wherever the codecs are deterministic,
  including blosc (blosclz, lz4, lz4hc), LZ4 and bz2. With zstd, gzip, zlib
  and lzma, whose Go implementations compress differently, each
  implementation must decode the other's chunks to the same values;
- **reads agree**: Go and zarr-python read the same values from both stores,
  for whole arrays and for random selections of every kind (basic,
  orthogonal, coordinate, mask and block), and both refuse the same invalid
  ones.

## Running

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r consistency/requirements.txt
ZARR_PYTHON=.venv/bin/python go test ./consistency/
```

A run takes about a minute and ends with a summary line, e.g.

```
consistency: 506 array cases, 7704 operations, 63767 comparisons (31030 chunks byte-identical, 11540 chunks decoded and compared), 544 operations both refused
```

`TestRandomCases` takes `-zarr.cases=N` and `-zarr.seed=S` for longer runs.

Without `ZARR_PYTHON`, the live tests skip and `TestFixtures` and
`TestHierarchyFixtures` check Go against recordings of zarr-python's stores
and answers in `testdata/fixtures` (401 arrays and four hierarchies, about
2 MB). Re-record them after changing a case:

```sh
ZARR_PYTHON=.venv/bin/python go test ./consistency/ -run TestFixtures -zarr.record
```

## What is covered

| Test | What it compares |
|---|---|
| `TestDataTypesV3`, `TestDataTypesV2` | Every data type, with interesting fill values (NaN, -0.0, ±Infinity, int64 and uint64 extremes, NaT, non-ASCII strings, structs); v2 also in Fortran order, big-endian and with `null` fill values. |
| `TestCodecsV3`, `TestCodecsV2` | Every codec as zarr-python's `create_array` takes it: `bytes` (both endians, structs, UTF-32), `transpose`, `gzip`, `zstd`, `crc32c`, `blosc` (all compressors and shuffles), the `numcodecs.*` compressors, checksums and filters, `scale_offset` and `cast_value`; in v2, numcodecs filters and compressors including `categorize`. |
| `TestSharding` | Morton order of inner chunks, edge shards, `index_location`, custom index codecs, nested sharding, codecs outside sharding, `write_empty_chunks`, and partial shard writes. |
| `TestGridsAndKeys`, `TestShapes` | Rectilinear grids, chunk key encodings (including the zarr-extensions `template` encoding, through the zarr-template package), 0-d arrays, empty dimensions, chunks larger than the array, dimension names, attributes. |
| `TestRandomCases` | Random combinations of all of the above. |
| `TestStoredDocuments` | Metadata other programs stored, opened by both: the invalid chunk shapes zarr-python 3.4.1 repairs (sizes of 0, `false` and `true`, float edges, "regular" grids listing edges; written by zarr-python 2.18–3.3), v2 data types spelled with a byte order that does not apply (netCDF-C, Zarr.jl, jzarr), `μs` and scaled generic time units, extra fields, and documents both must refuse (invalid chunk sizes, compressed shard indexes, indivisible nested shards). Both must write back the same documents and agree after writes, resizes and attribute updates. |
| `TestTruncatedShards`, `TestGroupExtraFields`, `TestConsolidatedRepairs` | Shards too short for their index are refused on every path; group extra fields are kept, written back and consolidated, or refused; consolidated metadata keeps a document as stored until its repair is stored. |
| `TestHierarchy` | Groups, implicit parents, attributes and consolidated metadata (v2 and v3), and member listings. |
| `TestCodecParity` | 201 codec configurations on buffers from 0 bytes to 300 KB: byte-identical output where deterministic, and each side decoding the other's output. |
| `TestNarrowFloats` | bfloat16, float8, float6 and float4 conversions against ml_dtypes: every bit pattern decoded, and values around every representable number and halfway point encoded. |
| `TestTensorStore` | The zarr-extensions data types zarr-python lacks (bfloat16, float8, float4, int2, int4), against TensorStore's zarr3 driver. |
| `TestKnownPythonIssues` | The zarr-python and numcodecs issues below: it reproduces each one, checks Go's behaviour, and fails if an issue is fixed upstream so its workaround can be removed. |

## Issues found in zarr-python 3.4 / numcodecs 0.17

The comparison works around these; Go handles each case correctly.

- Writing a single element of a `variable_length_bytes` array raises
  `TypeError: Expected bytes, got numpy.ndarray`.
- Reading a single element of a sharded `variable_length_bytes` array
  returns it inside nested 0-d object arrays.
- Chunks of a big-endian generic `timedelta64` array (`">m8[2generic]"`)
  that hold only NaT are stored instead of deleted.
- `numcodecs.quantize` with an `astype` other than its `dtype` cannot be read
  back: the v3 wrapper does not pass the new type to the next codec.
- `numcodecs.bitround` on Zarr v2 arrays with `order="F"` scrambles the data,
  because `BitRound` copies the array in C order.
- `zarr.open_array(..., config=...)` ignores `write_empty_chunks`;
  `Array.with_config` works.
- An `Ellipsis` after an integer array in an orthogonal selection raises
  `ValueError` (`replace_ellipsis` compares items with `==`).
- numcodecs' LZ4, Zstd and Blosc cannot decode their own output for an empty
  buffer, and Fletcher32 cannot encode one.
- `cast_value` (cast-value-rs 0.4.2) treats a float of exactly 2^63 (2^64)
  as within int64 (uint64) range; Go reports it as out of range unless
  `out_of_range` is `clamp`. The tests use values away from those limits.
- Writes with repeated integer indices have no defined winner: zarr-python
  orders them with an unstable `argsort`. The tests write unique indices.
- zarr-template 0.1.0 accepts a template that does not name every chunk
  coordinate (`"{0}"` on a 2-d array), so several chunks share a key; the
  specification requires an error, which Go gives.

zarr-python 3.4.1 fixed one issue listed for 3.4.0: reading a single
element of a `variable_length_bytes` array no longer drops trailing NUL
bytes.

Two differences are by design: TensorStore orders inner chunks within a shard
differently from zarr-python (the spec leaves the order open), so sharded
arrays are compared with it by value; and multi-threaded blosc in numcodecs
is not deterministic, which zarr-python avoids by disabling blosc threads.
