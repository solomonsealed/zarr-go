package consistency

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"testing"
)

func raw(s string) json.RawMessage { return json.RawMessage(s) }

func seedOf(name string) int64 {
	h := fnv.New64a()
	h.Write([]byte(name))
	return int64(h.Sum64() >> 1)
}

// run runs a case as a subtest with the standard exercise.
func run(t *testing.T, c Case, writes, reads int, resize bool) {
	t.Run(c.Name, func(t *testing.T) {
		o := pythonOracle(t)
		h := newHarness(t, o, c, seedOf(t.Name()))
		h.exercise(writes, reads, resize)
	})
}

// dtypeCase pairs a Zarr v3 data type with its NumPy (Zarr v2) spelling
// and interesting fill values (v3 JSON).
type dtypeCase struct {
	v3    string
	v2    string
	fills []string
}

var dtypes = []dtypeCase{
	{`"bool"`, `"|b1"`, []string{`true`}},
	{`"int8"`, `"|i1"`, []string{`-128`, `5`}},
	{`"int16"`, `"<i2"`, []string{`-32768`}},
	{`"int32"`, `"<i4"`, []string{`2147483647`}},
	{`"int64"`, `"<i8"`, []string{`-9223372036854775808`, `9007199254740993`}},
	{`"uint8"`, `"|u1"`, []string{`255`}},
	{`"uint16"`, `"<u2"`, []string{`65535`}},
	{`"uint32"`, `"<u4"`, []string{`4000000000`}},
	{`"uint64"`, `"<u8"`, []string{`18446744073709551615`}},
	{`"float16"`, `"<f2"`, []string{`"NaN"`, `-0.0`, `0.1`, `"Infinity"`}},
	{`"float32"`, `"<f4"`, []string{`"NaN"`, `-0.0`, `0.1`, `"-Infinity"`, `1e+30`}},
	{`"float64"`, `"<f8"`, []string{`"NaN"`, `-0.0`, `5e-324`, `1e+300`, `"Infinity"`}},
	{`"complex64"`, `"<c8"`, []string{`["NaN", 1.0]`, `[-0.0, 0.0]`}},
	{`"complex128"`, `"<c16"`, []string{`[1.5, "-Infinity"]`}},
	{`"string"`, `"|O"`, []string{`"fill"`, `"é😀"`}},
	// A non-empty fill for variable-length bytes breaks partial writes in
	// zarr-python 3.4 (TypeError: Expected bytes, got numpy.ndarray); see
	// TestKnownPythonIssues.
	{`"variable_length_bytes"`, ``, nil},
	{`{"name": "fixed_length_utf32", "configuration": {"length_bytes": 20}}`, `"<U5"`, []string{`"héllo"`}},
	{`{"name": "null_terminated_bytes", "configuration": {"length_bytes": 6}}`, `"|S6"`, []string{`"YWIAYw=="`}},
	{`{"name": "raw_bytes", "configuration": {"length_bytes": 5}}`, `"|V5"`, []string{`"AQIDBAU="`}},
	{`{"name": "numpy.datetime64", "configuration": {"unit": "ns", "scale_factor": 1}}`, `"<M8[ns]"`, []string{`1577836800000000000`}},
	{`{"name": "numpy.datetime64", "configuration": {"unit": "s", "scale_factor": 10}}`, `"<M8[10s]"`, []string{`0`}},
	{`{"name": "numpy.timedelta64", "configuration": {"unit": "ms", "scale_factor": 1}}`, `"<m8[ms]"`, []string{`-5`}},
	{`{"name": "struct", "configuration": {"fields": [{"name": "a", "data_type": "int32"}, {"name": "b", "data_type": "float64"}]}}`, `[["a", "<i4"], ["b", "<f8"]]`, []string{`{"a": 1, "b": "NaN"}`}},
	{`{"name": "struct", "configuration": {"fields": [{"name": "x", "data_type": "uint8"}, {"name": "s", "data_type": {"name": "null_terminated_bytes", "configuration": {"length_bytes": 3}}}, {"name": "c", "data_type": "complex64"}]}}`, `[["x", "|u1"], ["s", "|S3"], ["c", "<c8"]]`, nil},
}

// dataTypeCasesV3: every data type with default codecs and uncompressed,
// a 3-d shape with edge chunks, and each interesting fill value.
func dataTypeCasesV3() []Case {
	var out []Case
	for _, d := range dtypes {
		fills := append([]string{""}, d.fills...)
		for _, f := range fills {
			c := Case{Name: fmt.Sprintf("%s/fill=%s", d.v3, f), ZarrFormat: 3, Shape: []uint64{7, 5, 3}, Chunks: []uint64{3, 2, 2}, DataType: raw(d.v3)}
			if f != "" {
				c.FillValue = raw(f)
			}
			out = append(out, c)
			// Uncompressed, so chunks must match byte for byte.
			c.Name += "/nocompress"
			c.Compressors = compressors()
			out = append(out, c)
		}
	}
	return out
}

func TestDataTypesV3(t *testing.T) {
	for _, c := range dataTypeCasesV3() {
		run(t, c, 8, 10, c.Compressors == nil)
	}
}

// dataTypeCasesV2: the same in Zarr v2, including big-endian dtypes,
// Fortran order and null fill values.
func dataTypeCasesV2() []Case {
	var out []Case
	for _, d := range dtypes {
		if d.v2 == "" {
			continue
		}
		for _, order := range []string{"C", "F"} {
			c := Case{Name: fmt.Sprintf("%s/%s", d.v2, order), ZarrFormat: 2, Shape: []uint64{7, 5, 3}, Chunks: []uint64{3, 2, 2}, DType: raw(d.v2), Order: order, Compressors: compressors()}
			if d.v2 == `"|O"` {
				c.StringType = "string"
			}
			out = append(out, c)
			for _, f := range d.fills {
				c2 := c
				c2.Name += "/fill=" + f
				c2.FillValue = raw(f)
				out = append(out, c2)
			}
		}
		// Big-endian variants.
		var s string
		if json.Unmarshal(raw(d.v2), &s) == nil && len(s) > 2 && s[0] == '<' {
			out = append(out, Case{Name: ">" + s[1:], ZarrFormat: 2, Shape: []uint64{9, 4}, Chunks: []uint64{4, 3}, DType: raw(`">` + s[1:] + `"`), Compressors: compressors()})
		}
	}
	// Only whole-chunk writes: zarr-python 3.4 fails partial writes of
	// variable-length bytes (see TestKnownPythonIssues).
	out = append(out,
		Case{Name: "vlen-bytes", ZarrFormat: 2, Shape: []uint64{6}, Chunks: []uint64{6}, DType: raw(`"|O"`), StringType: "bytes", Compressors: compressors()},
		Case{Name: "null-fill", ZarrFormat: 2, Shape: []uint64{6, 6}, Chunks: []uint64{4, 4}, DType: raw(`"<f4"`), FillValue: raw(`null`), Compressors: compressors()},
		Case{Name: "null-fill-datetime", ZarrFormat: 2, Shape: []uint64{6}, Chunks: []uint64{4}, DType: raw(`"<M8[s]"`), FillValue: raw(`null`), Compressors: compressors()})
	return out
}

func TestDataTypesV2(t *testing.T) {
	for _, c := range dataTypeCasesV2() {
		writes := 6
		if c.Name == "vlen-bytes" {
			writes = 0
		}
		run(t, c, writes, 6, c.FillValue == nil && c.Name != "vlen-bytes")
	}
}

// codecCase is a codec chain to test on a data type.
type codecCase struct {
	name        string
	dtype       string
	filters     []string
	serializer  string
	compressors []string // nil: none
	fill        string
	safe        bool
}

var codecCases = []codecCase{
	{name: "bytes-big", dtype: `"float64"`, serializer: `{"name": "bytes", "configuration": {"endian": "big"}}`},
	{name: "bytes-big-complex", dtype: `"complex128"`, serializer: `{"name": "bytes", "configuration": {"endian": "big"}}`},
	{name: "bytes-big-struct", dtype: `{"name": "struct", "configuration": {"fields": [{"name": "a", "data_type": "int16"}, {"name": "b", "data_type": "float32"}]}}`, serializer: `{"name": "bytes", "configuration": {"endian": "big"}}`},
	{name: "bytes-big-utf32", dtype: `{"name": "fixed_length_utf32", "configuration": {"length_bytes": 12}}`, serializer: `{"name": "bytes", "configuration": {"endian": "big"}}`},
	{name: "gzip", dtype: `"float64"`, compressors: []string{`{"name": "gzip", "configuration": {"level": 5}}`}},
	{name: "zstd-checksum", dtype: `"int32"`, compressors: []string{`{"name": "zstd", "configuration": {"level": 3, "checksum": true}}`}},
	{name: "crc32c", dtype: `"uint16"`, compressors: []string{`{"name": "crc32c"}`}},
	{name: "zstd-crc32c", dtype: `"float32"`, compressors: []string{`{"name": "zstd", "configuration": {"level": 1, "checksum": false}}`, `{"name": "crc32c"}`}},
	{name: "transpose", dtype: `"int16"`, filters: []string{`{"name": "transpose", "configuration": {"order": [2, 0, 1]}}`}},
	{name: "transpose-strings", dtype: `"string"`, filters: []string{`{"name": "transpose", "configuration": {"order": [1, 2, 0]}}`}},
	{name: "transpose-twice", dtype: `"float32"`, filters: []string{`{"name": "transpose", "configuration": {"order": [2, 1, 0]}}`, `{"name": "transpose", "configuration": {"order": [1, 0, 2]}}`}},
	{name: "vlen-gzip", dtype: `"string"`, compressors: []string{`{"name": "gzip", "configuration": {"level": 1}}`}},
	// Blosc: zarr-python fills in typesize and shuffle.
	{name: "blosc-lz4", dtype: `"float32"`, compressors: []string{`{"name": "blosc", "configuration": {"cname": "lz4", "clevel": 5, "shuffle": "shuffle"}}`}},
	{name: "blosc-lz4hc-bit", dtype: `"float64"`, compressors: []string{`{"name": "blosc", "configuration": {"cname": "lz4hc", "clevel": 9, "shuffle": "bitshuffle"}}`}},
	{name: "blosc-blosclz", dtype: `"int16"`, compressors: []string{`{"name": "blosc", "configuration": {"cname": "blosclz", "clevel": 3}}`}},
	{name: "blosc-zstd", dtype: `"float64"`, compressors: []string{`{"name": "blosc", "configuration": {"cname": "zstd", "clevel": 3, "shuffle": "bitshuffle"}}`}},
	{name: "blosc-zlib", dtype: `"int16"`, compressors: []string{`{"name": "blosc", "configuration": {"cname": "zlib", "clevel": 1, "shuffle": "noshuffle"}}`}},
	{name: "blosc-uint8", dtype: `"uint8"`, compressors: []string{`{"name": "blosc", "configuration": {"cname": "lz4"}}`}},
	{name: "blosc-blocksize", dtype: `"float32"`, compressors: []string{`{"name": "blosc", "configuration": {"cname": "lz4", "clevel": 1, "shuffle": "shuffle", "blocksize": 64}}`}},
	// numcodecs compressors and checksums.
	{name: "numcodecs.zlib", dtype: `"float64"`, compressors: []string{`{"name": "numcodecs.zlib", "configuration": {"level": 3}}`}},
	{name: "numcodecs.gzip", dtype: `"float64"`, compressors: []string{`{"name": "numcodecs.gzip", "configuration": {"level": 1}}`}},
	{name: "numcodecs.bz2", dtype: `"float64"`, compressors: []string{`{"name": "numcodecs.bz2", "configuration": {"level": 9}}`}},
	{name: "numcodecs.bz2-strings", dtype: `"string"`, compressors: []string{`{"name": "numcodecs.bz2", "configuration": {}}`}},
	{name: "numcodecs.lzma", dtype: `"int32"`, compressors: []string{`{"name": "numcodecs.lzma", "configuration": {}}`}},
	{name: "numcodecs.lzma-preset", dtype: `"int32"`, compressors: []string{`{"name": "numcodecs.lzma", "configuration": {"preset": 1}}`}},
	{name: "numcodecs.lz4", dtype: `"float32"`, compressors: []string{`{"name": "numcodecs.lz4", "configuration": {}}`}},
	{name: "numcodecs.zstd", dtype: `"float32"`, compressors: []string{`{"name": "numcodecs.zstd", "configuration": {"level": 1}}`}},
	{name: "numcodecs.blosc", dtype: `"float64"`, compressors: []string{`{"name": "numcodecs.blosc", "configuration": {"cname": "zstd", "clevel": 3, "shuffle": 2}}`}},
	{name: "numcodecs.shuffle", dtype: `"float64"`, compressors: []string{`{"name": "numcodecs.shuffle", "configuration": {}}`}},
	{name: "numcodecs.shuffle-zlib", dtype: `"int32"`, compressors: []string{`{"name": "numcodecs.shuffle", "configuration": {"elementsize": 4}}`, `{"name": "numcodecs.zlib", "configuration": {"level": 1}}`}},
	{name: "numcodecs.checksums", dtype: `"int16"`, compressors: []string{`{"name": "numcodecs.crc32", "configuration": {}}`, `{"name": "numcodecs.crc32c", "configuration": {}}`, `{"name": "numcodecs.adler32", "configuration": {"location": "end"}}`, `{"name": "numcodecs.fletcher32", "configuration": {}}`}},
	{name: "numcodecs.crc32-end", dtype: `"float32"`, compressors: []string{`{"name": "numcodecs.crc32", "configuration": {"location": "end"}}`}},
	// numcodecs filters.
	{name: "numcodecs.delta", dtype: `"float64"`, filters: []string{`{"name": "numcodecs.delta", "configuration": {"dtype": "<f8"}}`}, safe: true},
	{name: "numcodecs.delta-astype", dtype: `"int32"`, filters: []string{`{"name": "numcodecs.delta", "configuration": {"dtype": "<i4", "astype": "<i2"}}`}, safe: true},
	{name: "numcodecs.delta-int8", dtype: `"int8"`, filters: []string{`{"name": "numcodecs.delta", "configuration": {"dtype": "|i1"}}`}},
	{name: "numcodecs.fixedscaleoffset", dtype: `"float64"`, filters: []string{`{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 1000, "scale": 10, "dtype": "<f8", "astype": "<i4"}}`}, safe: true},
	{name: "numcodecs.fixedscaleoffset-f32", dtype: `"float32"`, filters: []string{`{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 0.5, "scale": 4.0}}`}, safe: true},
	{name: "numcodecs.fixedscaleoffset-int", dtype: `"int32"`, filters: []string{`{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 7, "scale": 3, "dtype": "<i4", "astype": "<i4"}}`}, safe: true},
	{name: "numcodecs.quantize", dtype: `"float64"`, filters: []string{`{"name": "numcodecs.quantize", "configuration": {"digits": 3}}`}, safe: true},
	// (zarr-python 3.4 cannot read Quantize with an astype that differs
	// from dtype; see TestKnownPythonIssues.)
	{name: "numcodecs.quantize-f32", dtype: `"float32"`, filters: []string{`{"name": "numcodecs.quantize", "configuration": {"digits": 1, "dtype": "<f4"}}`}, safe: true},
	{name: "numcodecs.astype", dtype: `"float64"`, filters: []string{`{"name": "numcodecs.astype", "configuration": {"encode_dtype": "<f4"}}`}, safe: true},
	{name: "numcodecs.astype-int", dtype: `"int64"`, filters: []string{`{"name": "numcodecs.astype", "configuration": {"encode_dtype": "<i2", "decode_dtype": "<i8"}}`}, safe: true},
	{name: "numcodecs.packbits", dtype: `"bool"`, filters: []string{`{"name": "numcodecs.packbits", "configuration": {}}`}},
	{name: "numcodecs.bitround", dtype: `"float32"`, filters: []string{`{"name": "numcodecs.bitround", "configuration": {"keepbits": 7}}`}},
	{name: "numcodecs.bitround-f64", dtype: `"float64"`, filters: []string{`{"name": "numcodecs.bitround", "configuration": {"keepbits": 0}}`}},
	{name: "numcodecs.bitround-f16", dtype: `"float16"`, filters: []string{`{"name": "numcodecs.bitround", "configuration": {"keepbits": 3}}`}},
	// scale_offset and cast_value.
	{name: "scale_offset", dtype: `"float64"`, filters: []string{`{"name": "scale_offset", "configuration": {"offset": 1.0, "scale": 10}}`}},
	{name: "scale_offset-f32", dtype: `"float32"`, filters: []string{`{"name": "scale_offset", "configuration": {"offset": "NaN"}}`}},
	{name: "scale_offset-int", dtype: `"int16"`, filters: []string{`{"name": "scale_offset", "configuration": {"offset": 5, "scale": 3}}`}, fill: `5`, safe: true},
	{name: "scale_offset-identity", dtype: `"uint32"`, filters: []string{`{"name": "scale_offset"}`}},
	{name: "cast_value-uint8", dtype: `"float32"`, filters: []string{`{"name": "scale_offset", "configuration": {"offset": 1.0, "scale": 10}}`, `{"name": "cast_value", "configuration": {"data_type": "uint8", "out_of_range": "clamp", "scalar_map": {"encode": [["NaN", 255]], "decode": [[255, "NaN"]]}}}`}, fill: `"NaN"`},
	{name: "cast_value-f16", dtype: `"float64"`, filters: []string{`{"name": "cast_value", "configuration": {"data_type": "float16", "out_of_range": "clamp"}}`}},
	{name: "cast_value-int8-wrap", dtype: `"int32"`, filters: []string{`{"name": "cast_value", "configuration": {"data_type": "int8", "out_of_range": "wrap"}}`}},
	{name: "cast_value-round", dtype: `"float64"`, filters: []string{`{"name": "cast_value", "configuration": {"data_type": "int16", "rounding": "towards-zero", "out_of_range": "clamp", "scalar_map": {"encode": [["NaN", 0], ["Infinity", 32767]]}}}`}},
	{name: "cast_value-widen", dtype: `"uint8"`, filters: []string{`{"name": "cast_value", "configuration": {"data_type": "float32"}}`}, serializer: `{"name": "bytes", "configuration": {"endian": "little"}}`},
}

func codecCasesV3() []Case {
	var out []Case
	for _, cc := range codecCases {
		c := Case{Name: cc.name, ZarrFormat: 3, Shape: []uint64{9, 7, 4}, Chunks: []uint64{4, 3, 4}, DataType: raw(cc.dtype), Safe: cc.safe}
		c.Filters = rawList(cc.filters...)
		if cc.serializer != "" {
			c.Serializer = raw(cc.serializer)
		}
		c.Compressors = compressors(cc.compressors...)
		if cc.fill != "" {
			c.FillValue = raw(cc.fill)
		}
		out = append(out, c)
	}
	return out
}

// TestCodecsV3 runs each codec chain through the standard exercise.
func TestCodecsV3(t *testing.T) {
	for _, c := range codecCasesV3() {
		run(t, c, 6, 6, true)
	}
}

// codecCasesV2: numcodecs filters and compressors in Zarr v2 metadata
// (zarr-python writes their get_config(), defaults included).
func codecCasesV2() []Case {
	type v2case struct {
		name        string
		dtype       string
		filters     []string
		compressors []string // nil: no compressor
		safe        bool
	}
	cases := []v2case{
		{name: "zlib", dtype: `"<f8"`, compressors: []string{`{"id": "zlib", "level": 3}`}},
		{name: "gzip", dtype: `"<i4"`, compressors: []string{`{"id": "gzip"}`}},
		{name: "bz2", dtype: `"<f4"`, compressors: []string{`{"id": "bz2", "level": 5}`}},
		{name: "lzma", dtype: `"<u2"`, compressors: []string{`{"id": "lzma"}`}},
		{name: "lz4", dtype: `"<f8"`, compressors: []string{`{"id": "lz4"}`}},
		{name: "zstd", dtype: `"<f8"`, compressors: []string{`{"id": "zstd", "level": 2}`}},
		{name: "zstd-checksum", dtype: `"<f8"`, compressors: []string{`{"id": "zstd", "level": 1, "checksum": true}`}},
		{name: "blosc", dtype: `"<f8"`, compressors: []string{`{"id": "blosc"}`}},
		{name: "blosc-zstd-bit", dtype: `"<i2"`, compressors: []string{`{"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}`}},
		{name: "blosc-blosclz-noshuffle", dtype: `"|u1"`, compressors: []string{`{"id": "blosc", "cname": "blosclz", "clevel": 9, "shuffle": 0}`}},
		{name: "blosc-auto", dtype: `"|i1"`, compressors: []string{`{"id": "blosc", "cname": "lz4hc", "clevel": 4, "shuffle": -1}`}},
		{name: "shuffle-zlib", dtype: `"<f4"`, filters: []string{`{"id": "shuffle", "elementsize": 4}`}, compressors: []string{`{"id": "zlib"}`}},
		{name: "checksums", dtype: `"<i8"`, filters: []string{`{"id": "crc32"}`, `{"id": "adler32", "location": "end"}`, `{"id": "fletcher32"}`}, compressors: []string{`{"id": "crc32c"}`}},
		{name: "delta", dtype: `"<i4"`, filters: []string{`{"id": "delta", "dtype": "<i4"}`}, compressors: []string{`{"id": "zlib", "level": 1}`}, safe: true},
		{name: "delta-astype", dtype: `"<i8"`, filters: []string{`{"id": "delta", "dtype": "i8", "astype": "i2"}`}, safe: true},
		{name: "delta-float", dtype: `"<f4"`, filters: []string{`{"id": "delta", "dtype": "float32"}`}, safe: true},
		{name: "fixedscaleoffset", dtype: `"<f8"`, filters: []string{`{"id": "fixedscaleoffset", "offset": 1000, "scale": 10, "dtype": "<f8", "astype": "<u2"}`}, safe: true},
		{name: "fixedscaleoffset-f2", dtype: `"<f2"`, filters: []string{`{"id": "fixedscaleoffset", "offset": 0.25, "scale": 3.0, "dtype": "<f2"}`}, safe: true},
		{name: "quantize", dtype: `"<f8"`, filters: []string{`{"id": "quantize", "digits": 2, "dtype": "<f8"}`}, safe: true},
		{name: "quantize-astype", dtype: `"<f8"`, filters: []string{`{"id": "quantize", "digits": 1, "dtype": "<f8", "astype": "<f4"}`}, safe: true},
		{name: "astype", dtype: `"<f8"`, filters: []string{`{"id": "astype", "encode_dtype": "<f4", "decode_dtype": "<f8"}`}, safe: true},
		{name: "packbits", dtype: `"|b1"`, filters: []string{`{"id": "packbits"}`}},
		// (C order only: zarr-python 3.4 scrambles F-order chunks through
		// BitRound; see TestKnownPythonIssues.)
		{name: "bitround", dtype: `"<f4"`, filters: []string{`{"id": "bitround", "keepbits": 4}`}},
		{name: "categorize", dtype: `"<U3"`, filters: []string{`{"id": "categorize", "labels": ["ab", "c", "xyz"], "dtype": "<U3", "astype": "|u1"}`}},
		{name: "big-endian-delta", dtype: `">i4"`, filters: []string{`{"id": "delta", "dtype": ">i4"}`}, safe: true},
		{name: "strings-zlib", dtype: `"|O"`, compressors: []string{`{"id": "zlib", "level": 1}`}},
	}
	var out []Case
	for _, v := range cases {
		for _, order := range []string{"C", "F"} {
			if order == "F" && v.name == "bitround" {
				continue
			}
			c := Case{Name: v.name + "/" + order, ZarrFormat: 2, Shape: []uint64{9, 7, 4}, Chunks: []uint64{4, 3, 4}, DType: raw(v.dtype), Order: order, Safe: v.safe}
			if v.dtype == `"|O"` {
				c.StringType = "string"
			}
			c.Filters = rawList(v.filters...)
			c.Compressors = compressors(v.compressors...)
			out = append(out, c)
		}
	}
	return out
}

func TestCodecsV2(t *testing.T) {
	for _, c := range codecCasesV2() {
		run(t, c, 5, 5, c.Order == "C")
	}
}

// shardingCases: shard layouts (Morton order, edge shards, index location
// and codecs, nesting) and partial shard writes, chunk for chunk.
func shardingCases() []Case {
	none := compressors()
	cases := []Case{
		{Name: "default", Shape: []uint64{10, 9}, Chunks: []uint64{2, 3}, Shards: []uint64{4, 6}, DataType: raw(`"float32"`)},
		{Name: "uncompressed", Shape: []uint64{10, 9}, Chunks: []uint64{2, 3}, Shards: []uint64{4, 6}, DataType: raw(`"int16"`), Compressors: none},
		{Name: "index-start", Shape: []uint64{10, 9}, Chunks: []uint64{2, 3}, Shards: []uint64{4, 6}, IndexLocation: "start", DataType: raw(`"uint8"`), Compressors: none},
		{Name: "morton-3d", Shape: []uint64{6, 4, 10}, Chunks: []uint64{2, 2, 2}, Shards: []uint64{6, 4, 10}, DataType: raw(`"int32"`), Compressors: none},
		{Name: "morton-1d", Shape: []uint64{30}, Chunks: []uint64{1}, Shards: []uint64{12}, DataType: raw(`"float64"`), Compressors: none},
		{Name: "smaller-than-shard", Shape: []uint64{3, 5}, Chunks: []uint64{2, 2}, Shards: []uint64{8, 8}, DataType: raw(`"float32"`), FillValue: raw(`-1.0`), Compressors: none},
		{Name: "strings", Shape: []uint64{7, 6}, Chunks: []uint64{2, 2}, Shards: []uint64{4, 4}, DataType: raw(`"string"`), Compressors: none},
		{Name: "transpose-big-endian", Shape: []uint64{8, 8, 3}, Chunks: []uint64{2, 4, 3}, Shards: []uint64{4, 8, 3}, DataType: raw(`"float64"`),
			Filters: rawList(`{"name": "transpose", "configuration": {"order": [2, 1, 0]}}`), Serializer: raw(`{"name": "bytes", "configuration": {"endian": "big"}}`), Compressors: compressors(`{"name": "crc32c"}`)},
		{Name: "write-empty", Shape: []uint64{8, 8}, Chunks: []uint64{2, 2}, Shards: []uint64{4, 4}, DataType: raw(`"int8"`), Compressors: none, WriteEmptyChunks: true},
		{Name: "nested", Shape: []uint64{16, 12}, Chunks: []uint64{8, 12}, DataType: raw(`"uint16"`), Compressors: none,
			Serializer: raw(`{"name": "sharding_indexed", "configuration": {"chunk_shape": [4, 6], "codecs": [{"name": "sharding_indexed", "configuration": {"chunk_shape": [2, 3], "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}], "index_location": "start"}}], "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}], "index_location": "end"}}`)},
		{Name: "index-codecs-big", Shape: []uint64{9, 9}, Chunks: []uint64{6, 6}, DataType: raw(`"int32"`), Compressors: none,
			Serializer: raw(`{"name": "sharding_indexed", "configuration": {"chunk_shape": [3, 2], "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_codecs": [{"name": "bytes", "configuration": {"endian": "big"}}], "index_location": "end"}}`)},
		{Name: "index-codecs-numcodecs", Shape: []uint64{9, 9}, Chunks: []uint64{6, 6}, DataType: raw(`"int32"`), Compressors: none,
			Serializer: raw(`{"name": "sharding_indexed", "configuration": {"chunk_shape": [3, 3], "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "numcodecs.crc32", "configuration": {}}], "index_location": "start"}}`)},
		{Name: "index-codecs-transpose", Shape: []uint64{9, 9}, Chunks: []uint64{6, 6}, DataType: raw(`"int32"`), Compressors: none,
			Serializer: raw(`{"name": "sharding_indexed", "configuration": {"chunk_shape": [2, 3], "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_codecs": [{"name": "transpose", "configuration": {"order": [2, 0, 1]}}, {"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}], "index_location": "end"}}`)},
		{Name: "inner-compressors", Shape: []uint64{10, 10}, Chunks: []uint64{5, 2}, Shards: []uint64{10, 4}, DataType: raw(`"float64"`), Compressors: compressors(`{"name": "numcodecs.bz2", "configuration": {"level": 1}}`, `{"name": "crc32c"}`)},
		{Name: "outer-compressor", Shape: []uint64{8, 8}, Chunks: []uint64{4, 4}, DataType: raw(`"int16"`), Compressors: compressors(`{"name": "crc32c"}`),
			Serializer: raw(`{"name": "sharding_indexed", "configuration": {"chunk_shape": [2, 2], "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}], "index_location": "end"}}`)},
	}
	for i := range cases {
		cases[i].ZarrFormat = 3
	}
	return cases
}

func TestSharding(t *testing.T) {
	for _, c := range shardingCases() {
		run(t, c, 10, 8, true)
	}
}

// gridCases: rectilinear chunk grids and chunk key encodings.
func gridCases() []Case {
	none := compressors()
	cases := []Case{
		{Name: "rectilinear", ZarrFormat: 3, Shape: []uint64{10, 7}, Chunks: []uint64{3, 2}, ChunkEdges: [][]uint64{{3, 3, 4}, {2, 2, 2, 1}}, DataType: raw(`"int32"`), Compressors: none},
		{Name: "rectilinear-step", ZarrFormat: 3, Shape: []uint64{10, 7}, Chunks: []uint64{3, 2}, ChunkEdges: [][]uint64{{1, 5, 4}, nil}, DataType: raw(`"float32"`), Compressors: none},
		{Name: "rectilinear-beyond", ZarrFormat: 3, Shape: []uint64{10}, Chunks: []uint64{4}, ChunkEdges: [][]uint64{{4, 4, 4, 4}}, DataType: raw(`"float64"`), Compressors: none},
		{Name: "key-dot", ZarrFormat: 3, Shape: []uint64{5, 5}, Chunks: []uint64{2, 2}, DataType: raw(`"int8"`), ChunkKeyEncoding: raw(`{"name": "default", "configuration": {"separator": "."}}`), Compressors: none},
		{Name: "key-v2-slash", ZarrFormat: 3, Shape: []uint64{5, 5}, Chunks: []uint64{2, 2}, DataType: raw(`"int8"`), ChunkKeyEncoding: raw(`{"name": "v2", "configuration": {"separator": "/"}}`), Compressors: none},
		{Name: "key-v2-dot", ZarrFormat: 3, Shape: []uint64{5, 5}, Chunks: []uint64{2, 2}, DataType: raw(`"int8"`), ChunkKeyEncoding: raw(`{"name": "v2", "configuration": {"separator": "."}}`), Compressors: none},
		{Name: "v2-slash", ZarrFormat: 2, Shape: []uint64{5, 5}, Chunks: []uint64{2, 2}, DType: raw(`"<i4"`), ChunkKeyEncoding: raw(`{"name": "v2", "configuration": {"separator": "/"}}`), Compressors: none},
		// The zarr-extensions template encoding (zarr-python via zarr-template).
		{Name: "key-template", ZarrFormat: 3, Shape: []uint64{5, 6, 4}, Chunks: []uint64{2, 2, 3}, DataType: raw(`"int16"`), ChunkKeyEncoding: raw(`{"name": "template", "configuration": {"format": "{2}/c{0}_{1:02}.bin"}}`), Compressors: none},
		{Name: "key-template-catchall", ZarrFormat: 3, Shape: []uint64{5, 6, 7}, Chunks: []uint64{2, 4, 3}, DataType: raw(`"float32"`), ChunkKeyEncoding: raw(`{"name": "template", "configuration": {"format": "data/{-1}/{*:03}", "separator": "."}}`)},
		{Name: "key-template-braces", ZarrFormat: 3, Shape: []uint64{9}, Chunks: []uint64{2}, DataType: raw(`"uint8"`), ChunkKeyEncoding: raw(`{"name": "template", "configuration": {"format": "a{{b{*}}}c", "separator": "."}}`), Compressors: none},
		{Name: "key-template-sharded", ZarrFormat: 3, Shape: []uint64{8, 8}, Chunks: []uint64{2, 2}, Shards: []uint64{4, 4}, DataType: raw(`"int32"`), ChunkKeyEncoding: raw(`{"name": "template", "configuration": {"format": "shard-{*}", "separator": "-"}}`), Compressors: none},
	}
	return cases
}

func TestGridsAndKeys(t *testing.T) {
	for _, c := range gridCases() {
		run(t, c, 8, 8, true)
	}
}

// shapeCases: 0-d arrays, empty dimensions, chunks larger than the array,
// one-element chunks, many dimensions, names and attributes.
func shapeCases() []Case {
	none := compressors()
	attrs := map[string]any{"units": "m", "nested": map[string]any{"a": []any{1, 2.5, nil, "é"}}, "big": int64(9007199254740993), "flag": true}
	cases := []Case{
		{Name: "0d", ZarrFormat: 3, Shape: []uint64{}, Chunks: []uint64{}, DataType: raw(`"float64"`), Compressors: none},
		{Name: "0d-v2", ZarrFormat: 2, Shape: []uint64{}, Chunks: []uint64{}, DType: raw(`"<i2"`), Compressors: none},
		{Name: "0d-sharded", ZarrFormat: 3, Shape: []uint64{}, Chunks: []uint64{}, Shards: []uint64{}, DataType: raw(`"int16"`), Compressors: none},
		{Name: "empty-dim", ZarrFormat: 3, Shape: []uint64{0, 5}, Chunks: []uint64{2, 2}, DataType: raw(`"int32"`), Compressors: none},
		{Name: "big-chunks", ZarrFormat: 3, Shape: []uint64{3, 4}, Chunks: []uint64{10, 10}, DataType: raw(`"uint32"`), Compressors: none},
		{Name: "unit-chunks", ZarrFormat: 3, Shape: []uint64{5, 4}, Chunks: []uint64{1, 1}, DataType: raw(`"float16"`), Compressors: none},
		{Name: "4d", ZarrFormat: 3, Shape: []uint64{3, 4, 5, 2}, Chunks: []uint64{2, 3, 2, 1}, DataType: raw(`"complex64"`), Compressors: none},
		{Name: "1d-long", ZarrFormat: 3, Shape: []uint64{500}, Chunks: []uint64{7}, DataType: raw(`"int64"`), Compressors: none},
		{Name: "names-attrs", ZarrFormat: 3, Shape: []uint64{4, 3}, Chunks: []uint64{2, 2}, DataType: raw(`"int8"`), DimensionNames: []any{"y", nil}, Attributes: attrs, Compressors: none},
		{Name: "attrs-v2", ZarrFormat: 2, Shape: []uint64{4, 3}, Chunks: []uint64{2, 2}, DType: raw(`"|i1"`), Attributes: attrs, Compressors: none},
	}
	return cases
}

func TestShapes(t *testing.T) {
	for _, c := range shapeCases() {
		run(t, c, 8, 8, len(c.Shape) > 0)
	}
}
