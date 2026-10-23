package zarr

import (
	"encoding/json"
	"fmt"
	"slices"
)

// ArraySpec describes a new array. Its Metadata method renders the
// metadata zarr-python writes for the same arguments to
// zarr.create_array, byte for byte (attributes are written with sorted
// keys, since Go maps have no order).
type ArraySpec struct {
	// ZarrFormat is 3 (the default) or 2.
	ZarrFormat int
	Shape      []uint64
	// ChunkShape is the chunk shape (the inner chunk shape when sharded);
	// nil means one chunk for the whole array.
	ChunkShape []uint64
	// ChunkEdges, if set, makes a rectilinear chunk grid (Zarr v3):
	// explicit chunk lengths per dimension, or nil to repeat
	// ChunkShape[d]. zarr-python needs the array.rectilinear_chunks option
	// to read these.
	ChunkEdges [][]uint64
	// DataType is a Zarr v3 data type name such as "float32" or "string",
	// or for Zarr v2 a NumPy type string such as ">i4". Type is used
	// instead when DataType is empty (for struct and parametrized types).
	DataType string
	Type     DataType
	// FillValue defaults to the type's zero value (false, 0, 0.0, "", NaT).
	// Go values of the element type are accepted, as are the JSON
	// spellings "NaN", "Infinity", "-Infinity" and hex bit patterns, a
	// json.RawMessage, or a 0-d *NDArray.
	FillValue any

	// The codecs, as zarr-python's create_array takes them: Filters
	// (array-to-array codecs, or Zarr v2 filters), Serializer (Zarr v3
	// array-to-bytes codec; nil for the default bytes or vlen codec) and
	// Compressors (bytes-to-bytes codecs; nil means zstd level 0, an empty
	// non-nil slice none; Zarr v2 takes at most one). When sharded, these
	// are the inner chunk codecs.
	Filters     []CodecSpec
	Serializer  *CodecSpec
	Compressors []CodecSpec
	// Codecs, if set, is the complete Zarr v3 codec list (inner codecs
	// when sharded) and replaces Filters, Serializer and Compressors.
	Codecs []CodecSpec

	// ShardShape, if set, stores chunks in shards of this shape (a multiple
	// of ChunkShape) with the sharding_indexed codec.
	ShardShape []uint64
	// IndexLocation is "end" (the default) or "start".
	IndexLocation string
	// IndexCodecs default to bytes (little endian) + crc32c.
	IndexCodecs []CodecSpec

	// ChunkKeyEncoding defaults to "default" with "/" (Zarr v3) or "v2"
	// with "." (Zarr v2).
	ChunkKeyEncoding ChunkKeyEncoding
	// Order is the Zarr v2 memory layout of chunks, "C" (default) or "F".
	Order string
	// DimensionNames are optional (Zarr v3); "" leaves a dimension unnamed.
	DimensionNames []string
	Attributes     map[string]any
}

// Common Zarr v3 codecs, with the JSON zarr-python writes.

// Bytes is the bytes serializer; endian "" omits the configuration
// (as zarr-python does for single-byte types).
func Bytes(endian string) CodecSpec {
	if endian == "" {
		return CodecSpec{Name: "bytes"}
	}
	return CodecJSON("bytes", fmt.Sprintf(`{"endian": %q}`, endian))
}

// BytesLE is the little-endian bytes serializer.
func BytesLE() CodecSpec { return Bytes("little") }

// Zstd is the zstd compressor without checksum.
func Zstd(level int) CodecSpec { return ZstdCodec(level, false) }

// ZstdCodec is the zstd compressor.
func ZstdCodec(level int, checksum bool) CodecSpec {
	return CodecJSON("zstd", fmt.Sprintf(`{"level": %d, "checksum": %t}`, level, checksum))
}

// Gzip is the gzip compressor.
func Gzip(level int) CodecSpec { return CodecJSON("gzip", fmt.Sprintf(`{"level": %d}`, level)) }

// Crc32c appends a CRC32C checksum (no configuration, as zarr-python writes it).
func Crc32c() CodecSpec { return CodecSpec{Name: "crc32c"} }

// Blosc configures blosc: cname "blosclz", "lz4", "lz4hc", "snappy",
// "zlib" or "zstd"; shuffle "noshuffle", "shuffle" or "bitshuffle" ("" to
// pick from the type size like zarr-python). The typesize is filled in
// from the data type.
func Blosc(cname string, clevel int, shuffle string) CodecSpec {
	cfg := jsonObject{{"cname", cname}, {"clevel", clevel}}
	if shuffle != "" {
		cfg = append(cfg, jsonMember{"shuffle", shuffle})
	}
	cfg = append(cfg, jsonMember{"blocksize", 0})
	b, _ := marshalPython(cfg, -1)
	return CodecSpec{Name: "blosc", Configuration: b}
}

// Transpose reorders dimensions in storage.
func Transpose(order []int) CodecSpec {
	b, _ := marshalPython(jsonObject{{"order", order}}, -1)
	return CodecSpec{Name: "transpose", Configuration: b}
}

// VlenUTF8 is the serializer of "string" arrays.
func VlenUTF8() CodecSpec { return CodecJSON("vlen-utf8", "{}") }

// VlenBytes is the serializer of variable-length bytes arrays.
func VlenBytes() CodecSpec { return CodecJSON("vlen-bytes", "{}") }

// ScaleOffset is the scale_offset codec: stored = (value - offset) * scale.
// offset and scale are numbers or JSON spellings ("NaN", hex).
func ScaleOffset(offset, scale any) CodecSpec {
	cfg := jsonObject{}
	if !isNumber(offset, 0) {
		cfg = append(cfg, jsonMember{"offset", offset})
	}
	if !isNumber(scale, 1) {
		cfg = append(cfg, jsonMember{"scale", scale})
	}
	if len(cfg) == 0 {
		return CodecSpec{Name: "scale_offset"}
	}
	b, _ := marshalPython(cfg, -1)
	return CodecSpec{Name: "scale_offset", Configuration: b}
}

func isNumber(v any, want float64) bool {
	switch x := v.(type) {
	case int:
		return float64(x) == want
	case int64:
		return float64(x) == want
	case float64:
		return x == want
	}
	return false
}

// CastValue is the cast_value codec converting values to dataType.
// rounding "" means "nearest-even"; outOfRange is "", "clamp" or "wrap".
func CastValue(dataType, rounding, outOfRange string) CodecSpec {
	cfg := jsonObject{{"data_type", dataType}}
	if rounding != "" && rounding != "nearest-even" {
		cfg = append(cfg, jsonMember{"rounding", rounding})
	}
	if outOfRange != "" {
		cfg = append(cfg, jsonMember{"out_of_range", outOfRange})
	}
	b, _ := marshalPython(cfg, -1)
	return CodecSpec{Name: "cast_value", Configuration: b}
}

// Sharding returns a sharding_indexed codec.
func Sharding(chunkShape []uint64, codecs, indexCodecs []CodecSpec, indexLocation string) CodecSpec {
	if indexCodecs == nil {
		indexCodecs = []CodecSpec{BytesLE(), Crc32c()}
	}
	if indexLocation == "" {
		indexLocation = "end"
	}
	cfg := jsonObject{{"chunk_shape", chunkShape}, {"codecs", codecsJSON(codecs)}, {"index_codecs", codecsJSON(indexCodecs)}, {"index_location", indexLocation}}
	b, _ := marshalPython(cfg, -1)
	return CodecSpec{Name: "sharding_indexed", Configuration: b}
}

func codecsJSON(cs []CodecSpec) []any {
	out := make([]any, len(cs))
	for i, c := range cs {
		out[i] = c.json()
	}
	return out
}

// resolveType returns the spec's data type and (Zarr v2) byte order.
func (s ArraySpec) resolveType(format int) (DataType, bool, error) {
	if s.DataType == "" {
		if s.Type.Kind == 0 {
			return DataType{}, false, fmt.Errorf("zarr: ArraySpec needs a DataType")
		}
		return s.Type, false, nil
	}
	if format == 2 {
		raw, _ := json.Marshal(s.DataType)
		if dt, big, err := parseV2DataType(raw, ""); err == nil && s.DataType != "bool" {
			return dt, big, nil
		}
	}
	dt, err := ParseDataType(s.DataType)
	return dt, false, err
}

// defaultSerializer is the array-to-bytes codec zarr-python picks for dt.
func defaultSerializer(dt DataType) CodecSpec {
	switch dt.Kind {
	case KindString:
		return VlenUTF8()
	case KindBytes:
		return VlenBytes()
	}
	if needsEndian(dt) {
		return BytesLE()
	}
	return Bytes("")
}

// needsEndian reports whether byte order applies to dt.
func needsEndian(dt DataType) bool { return len(dt.swapUnits()) > 0 }

// Metadata renders zarr.json (Zarr v3) or .zarray (Zarr v2), validated by
// parsing it back.
func (s ArraySpec) Metadata() ([]byte, error) {
	m, err := s.build()
	if err != nil {
		return nil, err
	}
	return m.Document()
}

// Documents renders all of the array's metadata files by name: zarr.json,
// or .zarray and .zattrs.
func (s ArraySpec) Documents() (map[string][]byte, error) {
	m, err := s.build()
	if err != nil {
		return nil, err
	}
	return m.documents()
}

// build renders the metadata and parses it back.
func (s ArraySpec) build() (*Metadata, error) {
	format := s.ZarrFormat
	if format == 0 {
		format = 3
	}
	if format != 2 && format != 3 {
		return nil, fmt.Errorf("zarr: invalid zarr format %d", format)
	}
	dt, big, err := s.resolveType(format)
	if err != nil {
		return nil, err
	}
	shape := s.Shape
	if shape == nil {
		shape = []uint64{}
	}
	chunks := s.ChunkShape
	if chunks == nil {
		chunks = make([]uint64, len(shape))
		for i, n := range shape {
			chunks[i] = max(n, 1)
		}
	}
	if len(chunks) != len(shape) {
		return nil, fmt.Errorf("zarr: chunk shape %v does not match shape %v", chunks, shape)
	}
	fill, err := fillFromGo(dt, s.FillValue)
	if err != nil {
		return nil, err
	}
	if s.FillValue == nil && (dt.Kind == KindDatetime || dt.Kind == KindTimedelta) {
		fill, _ = parseFillValue(json.RawMessage(`"NaT"`), dt)
	}
	attrs := attrsJSON(s.Attributes)
	if format == 2 {
		return s.buildV2(dt, big, shape, chunks, fill, attrs)
	}

	codecs := s.Codecs
	if codecs == nil {
		codecs = slices.Clone(s.Filters)
		ser := defaultSerializer(dt)
		if s.Serializer != nil {
			ser = *s.Serializer
		}
		codecs = append(codecs, ser)
		if s.Compressors == nil {
			codecs = append(codecs, Zstd(0))
		} else {
			codecs = append(codecs, s.Compressors...)
		}
	}
	elem := ChunkSpec{Shape: chunks, DataType: dt, FillValue: fill.ndarray(dt)}
	if codecs, err = evolveCodecs(codecs, elem); err != nil {
		return nil, err
	}
	var grid chunkGrid = regularGrid(chunks)
	if s.ShardShape != nil && len(shape) > 0 { // zarr-python ignores shards for 0-d arrays
		if len(s.ShardShape) != len(shape) {
			return nil, fmt.Errorf("zarr: shard shape %v does not match shape %v", s.ShardShape, shape)
		}
		if s.ChunkEdges != nil {
			return nil, fmt.Errorf("zarr: sharding with a rectilinear grid is not supported")
		}
		idx := s.IndexCodecs
		if idx == nil {
			idx = []CodecSpec{BytesLE(), Crc32c()}
		}
		codecs = []CodecSpec{Sharding(chunks, codecs, idx, s.IndexLocation)}
		grid = regularGrid(s.ShardShape)
	}
	explicit := false
	for _, e := range s.ChunkEdges {
		explicit = explicit || e != nil
	}
	if explicit { // all-regular dimensions make a regular grid, as in zarr-python
		if len(s.ChunkEdges) != len(shape) {
			return nil, fmt.Errorf("zarr: chunk edges for %d dimensions, array has %d", len(s.ChunkEdges), len(shape))
		}
		g, err := newRectilinearGrid(chunks, s.ChunkEdges)
		if err != nil {
			return nil, err
		}
		for d, edges := range s.ChunkEdges {
			if edges == nil {
				continue
			}
			if sum := g.starts[d][len(g.starts[d])-1]; sum != shape[d] {
				// zarr-python requires an exact cover when creating.
				return nil, fmt.Errorf("zarr: chunk edges of dimension %d sum to %d, not the dimension's length %d", d, sum, shape[d])
			}
		}
		grid = g
	}
	kenc := s.ChunkKeyEncoding
	if kenc.Name == "" {
		kenc.Name = "default"
	}
	kenc.Separator = kenc.separator()
	if err := kenc.check(len(shape)); err != nil {
		return nil, err
	}
	doc := jsonObject{
		{"shape", shape},
		{"data_type", dt.v3JSON()},
		{"chunk_grid", grid.json()},
		{"chunk_key_encoding", kenc.json()},
		{"fill_value", fillJSON(dt, fill, 3)},
		{"codecs", codecsJSON(codecs)},
		{"attributes", attrs},
	}
	if s.DimensionNames != nil {
		if len(s.DimensionNames) != len(shape) {
			return nil, fmt.Errorf("zarr: %d dimension names for %d dimensions", len(s.DimensionNames), len(shape))
		}
		names := make([]any, len(s.DimensionNames))
		for i, n := range s.DimensionNames {
			if n != "" {
				names[i] = n
			}
		}
		doc = append(doc, jsonMember{"dimension_names", names})
	}
	doc = append(doc, jsonMember{"zarr_format", 3}, jsonMember{"node_type", "array"}, jsonMember{"storage_transformers", []any{}})
	out, err := marshalPython(doc, 2)
	if err != nil {
		return nil, err
	}
	m, err := parseV3Metadata(out)
	if err != nil {
		return nil, fmt.Errorf("zarr: invalid array spec: %w", err)
	}
	if _, err := chunkPipeline(m, ChunkSpec{Shape: m.nominalChunkShape(), DataType: m.DataType, FillValue: m.fill.ndarray(m.DataType)}); err != nil {
		return nil, fmt.Errorf("zarr: invalid array spec: %w", err)
	}
	return m, nil
}

func (s ArraySpec) buildV2(dt DataType, big bool, shape, chunks []uint64, fill fillValue, attrs any) (*Metadata, error) {
	var fillJ any = fillJSON(dt, fill, 2)
	if raw, ok := s.FillValue.(json.RawMessage); ok && string(raw) == "null" {
		fillJ = nil // an explicit null: readers use the type's default
	}
	if s.ShardShape != nil || s.DimensionNames != nil || s.Serializer != nil || s.Codecs != nil || s.ChunkEdges != nil {
		return nil, fmt.Errorf("zarr: shards, dimension names, serializers, codec lists and rectilinear grids need Zarr v3")
	}
	if s.ChunkKeyEncoding.Name == "template" {
		return nil, fmt.Errorf("zarr: the template chunk key encoding needs Zarr v3")
	}
	dtJSON, err := dt.v2JSON(big)
	if err != nil {
		return nil, err
	}
	filters := slices.Clone(s.Filters)
	switch dt.Kind {
	case KindString:
		filters = append([]CodecSpec{{Name: "vlen-utf8"}}, filters...)
	case KindBytes:
		filters = append([]CodecSpec{{Name: "vlen-bytes"}}, filters...)
	}
	var filtersJSON any
	if len(filters) > 0 {
		fs := make([]any, len(filters))
		for i, f := range filters {
			if fs[i], err = numcodecJSON(f); err != nil {
				return nil, err
			}
		}
		filtersJSON = fs
	}
	var compressor any
	switch {
	case s.Compressors == nil:
		compressor = jsonObject{{"id", "zstd"}, {"level", 0}}
	case len(s.Compressors) == 1:
		if compressor, err = numcodecJSON(s.Compressors[0]); err != nil {
			return nil, err
		}
	case len(s.Compressors) > 1:
		return nil, fmt.Errorf("zarr: Zarr v2 arrays take at most one compressor")
	}
	order := s.Order
	if order == "" {
		order = "C"
	}
	sep := "."
	if s.ChunkKeyEncoding.Separator != "" {
		sep = s.ChunkKeyEncoding.Separator
	}
	doc := jsonObject{
		{"shape", shape},
		{"chunks", chunks},
		{"dtype", dtJSON},
		{"fill_value", fillJ},
		{"order", order},
		{"filters", filtersJSON},
		{"dimension_separator", sep},
		{"compressor", compressor},
		{"zarr_format", 2},
	}
	zarray, err := marshalPython(doc, 2)
	if err != nil {
		return nil, err
	}
	zattrs, err := marshalPython(attrs, 2)
	if err != nil {
		return nil, err
	}
	m, err := ParseMetadataV2(zarray, zattrs)
	if err != nil {
		return nil, fmt.Errorf("zarr: invalid array spec: %w", err)
	}
	if _, err := chunkPipeline(m, ChunkSpec{Shape: m.ChunkShape, DataType: m.DataType, FillValue: m.fill.ndarray(m.DataType), bigEndian: m.BigEndian}); err != nil {
		return nil, fmt.Errorf("zarr: invalid array spec: %w", err)
	}
	return m, nil
}

// numcodecJSON renders a Zarr v2 filter or compressor as zarr-python
// writes it: numcodecs' get_config(), with defaults filled in and dtypes
// normalized ("f8" becomes "<f8").
func numcodecJSON(c CodecSpec) (any, error) {
	cfg := jsonObject{}
	if len(c.Configuration) > 0 {
		p, err := parseOrdered(c.Configuration)
		if err != nil {
			return nil, err
		}
		o, ok := p.(jsonObject)
		if !ok {
			return nil, fmt.Errorf("zarr: codec %q configuration is not an object", c.Name)
		}
		cfg = o
	}
	cfg.del("id")
	get := func(k string, dflt any) any {
		if v, ok := cfg.get(k); ok {
			return v
		}
		return dflt
	}
	dtype := func(k string, dflt any) (any, error) {
		v := get(k, dflt)
		s, ok := v.(string)
		if !ok {
			return v, nil
		}
		dt, big, err := parseNumpyDType(s)
		if err != nil {
			return nil, fmt.Errorf("zarr: codec %q: %w", c.Name, err)
		}
		return dt.v2JSON(big)
	}
	out := jsonObject{{"id", c.Name}}
	add := func(k string, v any) { out = append(out, jsonMember{k, v}) }
	var err error
	switch c.Name {
	case "zlib", "gzip", "bz2":
		add("level", get("level", 1))
	case "lzma":
		add("format", get("format", 1))
		add("check", get("check", -1))
		add("preset", get("preset", nil))
		add("filters", get("filters", nil))
	case "zstd":
		add("level", get("level", 0))
		// zarr-python drops checksum when false.
		if cs := get("checksum", false); cs != false {
			add("checksum", cs)
		}
	case "blosc":
		add("cname", get("cname", "lz4"))
		add("clevel", get("clevel", 5))
		add("shuffle", get("shuffle", 1))
		add("blocksize", get("blocksize", 0))
	case "lz4":
		add("acceleration", get("acceleration", 1))
	case "shuffle":
		add("elementsize", get("elementsize", 4))
	case "jenkins_lookup3":
		add("initval", get("initval", 0))
		add("prefix", get("prefix", nil))
	case "delta", "quantize", "fixedscaleoffset", "categorize":
		switch c.Name {
		case "quantize":
			add("digits", get("digits", nil))
		case "fixedscaleoffset":
			add("scale", get("scale", nil))
			add("offset", get("offset", nil))
		case "categorize":
			add("labels", get("labels", []any{}))
		}
		d, err := dtype("dtype", nil)
		if err != nil {
			return nil, err
		}
		add("dtype", d)
		dflt := d
		if c.Name == "categorize" {
			dflt = "|u1"
		}
		a, err := dtype("astype", dflt)
		if err != nil {
			return nil, err
		}
		add("astype", a)
	case "astype":
		for _, k := range []string{"encode_dtype", "decode_dtype"} {
			v, err := dtype(k, nil)
			if err != nil {
				return nil, err
			}
			add(k, v)
		}
	default:
		out = append(out, cfg...)
	}
	return out, err
}

// evolveCodecs fills in the settings zarr-python derives from the data
// type when it creates an array (its codecs' evolve_from_array_spec).
func evolveCodecs(codecs []CodecSpec, elem ChunkSpec) ([]CodecSpec, error) {
	out := make([]CodecSpec, len(codecs))
	cur := elem
	afterSerializer := false
	for i, c := range codecs {
		cfg := jsonObject{}
		if len(c.Configuration) > 0 {
			p, err := parseOrdered(c.Configuration)
			if err != nil {
				return nil, fmt.Errorf("zarr: codec %q: %w", c.Name, err)
			}
			if o, ok := p.(jsonObject); ok {
				cfg = o
			}
		}
		changed := false
		set := func(k string, v any) {
			if _, ok := cfg.get(k); !ok {
				cfg = append(cfg, jsonMember{k, v})
				changed = true
			}
		}
		switch c.Name {
		case "bytes":
			if !needsEndian(cur.DataType) && len(cfg) > 0 {
				cfg.del("endian")
				changed = true
			}
		case "blosc":
			ts := cur.DataType.Size
			if afterSerializer {
				ts = max(cur.DataType.Size, 1)
			}
			// zarr-python's BloscCodec.to_dict order.
			shuffle, hasShuffle := cfg.get("shuffle")
			if !hasShuffle || shuffle == nil {
				shuffle = "shuffle"
				if ts == 1 {
					shuffle = "bitshuffle"
				}
			}
			get := func(k string, d any) any {
				if v, ok := cfg.get(k); ok && v != nil {
					return v
				}
				return d
			}
			cfg = jsonObject{{"typesize", get("typesize", ts)}, {"cname", get("cname", "zstd")}, {"clevel", get("clevel", 5)}, {"shuffle", shuffle}, {"blocksize", get("blocksize", 0)}}
			changed = true
		case "numcodecs.shuffle":
			set("elementsize", cur.DataType.Size)
		case "numcodecs.fixedscaleoffset", "numcodecs.quantize":
			set("dtype", numpyName(cur.DataType))
		case "numcodecs.astype":
			set("decode_dtype", numpyName(cur.DataType))
		}
		if changed {
			b, err := marshalPython(cfg, -1)
			if err != nil {
				return nil, err
			}
			c.Configuration = b
			if c.Name == "bytes" && len(cfg) == 0 {
				c.Configuration = nil
			}
		}
		out[i] = c
		// Track the data type through array-to-array codecs.
		if !afterSerializer {
			codec, err := newCodec(c, cur, false)
			if err != nil {
				return nil, err
			}
			switch x := codec.(type) {
			case ArrayToArrayCodec:
				if cur, err = x.Resolve(cur); err != nil {
					return nil, err
				}
			case ArrayToBytesCodec:
				afterSerializer = true
			}
		}
	}
	return out, nil
}

// numpyName is str(np.dtype) for numeric types ("float64", "int16").
func numpyName(dt DataType) string { return dt.Name }
