// Package zarr reads and writes Zarr arrays and groups (formats 2 and 3)
// in pure Go.
//
// It covers the data types, codecs, chunk grids and selections of
// zarr-python 3 and is checked against it, both ways, byte for byte where
// the codecs are deterministic: see the consistency directory.
//
// Arrays and groups live in a Store: memory, local files, zip archives, any
// io/fs.FS, S3 and HTTP (package httpstore), or your own implementation.
// Lower down, NewArray opens an array from its metadata and a ChunkSource,
// which is how the icechunk-go package serves arrays from Icechunk
// repositories.
package zarr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"
)

// Metadata is a parsed array metadata document: zarr.json (Zarr v3) or
// .zarray plus .zattrs (Zarr v2).
type Metadata struct {
	ZarrFormat int
	NodeType   string
	Shape      []uint64
	DataType   DataType
	// ChunkShape is the regular chunk shape (the shard shape for sharded
	// arrays); nil for rectilinear grids.
	ChunkShape       []uint64
	ChunkKeyEncoding ChunkKeyEncoding
	FillValue        json.RawMessage
	// Codecs is the Zarr v3 codec chain.
	Codecs         []CodecSpec
	Attributes     map[string]any
	DimensionNames []*string

	// Zarr v2: memory order of chunks ("C" or "F"), filters and compressor
	// (Name is the numcodecs id, Configuration its other settings) and
	// the byte order of the dtype.
	Order      string
	Filters    []CodecSpec
	Compressor *CodecSpec
	BigEndian  bool

	grid chunkGrid
	fill fillValue
	// doc is the document (zarr.json or .zarray) with its key order and
	// any extra fields, re-rendered when the metadata changes.
	doc      jsonObject
	rawAttrs any // ordered attributes as parsed
	// stored is the document as stored when reading it needed a repair
	// that moves chunks (see repair.go), until the repair is stored.
	stored jsonObject
}

// RepairPending reports whether the metadata was read from an invalid
// stored document whose repair moves chunks: a chunk size of 0 or false,
// read as 1 (or the inner chunk size of a shard), or a "regular" grid
// listing chunk edges, read as rectilinear. zarr-python 3.4.1 reads such
// documents the same way. Chunks written under the repaired metadata are
// not where readers of the stored document look for them, so the repaired
// Document must be stored before chunks are written: arrays opened from a
// Store do this on their first write; callers writing through NewArray's
// ChunkWriter must do it themselves.
func (m *Metadata) RepairPending() bool { return m.stored != nil }

// ParseMetadata parses an array's zarr.json, or a Zarr v2 .zarray (whose
// attributes then live in .zattrs; see ParseMetadataV2).
func ParseMetadata(doc []byte) (*Metadata, error) {
	var probe struct {
		ZarrFormat int `json:"zarr_format"`
	}
	if err := json.Unmarshal(pyCompat(doc), &probe); err != nil {
		return nil, fmt.Errorf("zarr: invalid metadata: %w", err)
	}
	if probe.ZarrFormat == 2 {
		return ParseMetadataV2(doc, nil)
	}
	return parseV3Metadata(doc)
}

type rawV3 struct {
	ZarrFormat          int             `json:"zarr_format"`
	NodeType            string          `json:"node_type"`
	Shape               []uint64        `json:"shape"`
	DataType            json.RawMessage `json:"data_type"`
	ChunkGrid           json.RawMessage `json:"chunk_grid"`
	ChunkKeyEncoding    json.RawMessage `json:"chunk_key_encoding"`
	FillValue           json.RawMessage `json:"fill_value"`
	Codecs              []CodecSpec     `json:"codecs"`
	Attributes          json.RawMessage `json:"attributes"`
	DimensionNames      []*string       `json:"dimension_names"`
	StorageTransformers []CodecSpec     `json:"storage_transformers"`
}

// orderedDocument parses an array metadata document keeping key order,
// and applies the repairs of invalid documents (see repair.go). If a
// repair changed it, repaired is the re-rendered document (else nil), and
// stored is the document as stored if the repair moves chunks.
func orderedDocument(doc []byte, format int) (ordered jsonObject, repaired []byte, stored jsonObject, err error) {
	v, err := parseOrdered(doc)
	if err != nil {
		return nil, nil, nil, err
	}
	obj, ok := v.(jsonObject)
	if !ok {
		return nil, nil, nil, fmt.Errorf("not a JSON object")
	}
	fixed, changed, moves := repairArrayDocument(obj, format)
	if !changed {
		return obj, nil, nil, nil
	}
	if repaired, err = marshalPython(fixed, -1); err != nil {
		return nil, nil, nil, err
	}
	if moves {
		stored = obj
	}
	return fixed, repaired, stored, nil
}

func parseV3Metadata(doc []byte) (*Metadata, error) {
	var raw rawV3
	if err := json.Unmarshal(pyCompat(doc), &raw); err != nil {
		return nil, fmt.Errorf("zarr: invalid zarr.json: %w", err)
	}
	if raw.ZarrFormat != 3 {
		return nil, fmt.Errorf("%w: zarr_format %d", ErrUnsupported, raw.ZarrFormat)
	}
	if raw.NodeType != "array" {
		return nil, fmt.Errorf("zarr: node_type is %q, not an array", raw.NodeType)
	}
	if len(raw.StorageTransformers) > 0 {
		return nil, fmt.Errorf("%w: storage transformer %q", ErrUnsupported, raw.StorageTransformers[0].Name)
	}
	ordered, repaired, stored, err := orderedDocument(doc, 3)
	if err != nil {
		return nil, fmt.Errorf("zarr: invalid zarr.json: %w", err)
	}
	if repaired != nil {
		raw = rawV3{}
		if err := json.Unmarshal(pyCompat(repaired), &raw); err != nil {
			return nil, fmt.Errorf("zarr: invalid zarr.json: %w", err)
		}
	}
	m := &Metadata{
		ZarrFormat:     3,
		NodeType:       raw.NodeType,
		Shape:          raw.Shape,
		FillValue:      raw.FillValue,
		Codecs:         raw.Codecs,
		DimensionNames: raw.DimensionNames,
		doc:            ordered,
		stored:         stored,
	}
	// Extensions the reader must understand are refused.
	for _, mem := range m.doc {
		switch mem.Key {
		case "zarr_format", "node_type", "shape", "data_type", "chunk_grid", "chunk_key_encoding", "fill_value",
			"codecs", "attributes", "dimension_names", "storage_transformers":
			continue
		}
		if obj, ok := mem.Value.(jsonObject); ok {
			if mu, _ := obj.get("must_understand"); mu == false {
				continue
			}
		}
		return nil, fmt.Errorf("%w: metadata field %q", ErrUnsupported, mem.Key)
	}
	if m.Attributes, err = decodeAttributes(raw.Attributes); err != nil {
		return nil, err
	}
	m.rawAttrs, _ = m.doc.get("attributes")
	if m.Shape == nil {
		m.Shape = []uint64{}
	}
	if err := checkShape(m.Shape); err != nil {
		return nil, err
	}
	if m.DimensionNames != nil && len(m.DimensionNames) != len(m.Shape) {
		return nil, fmt.Errorf("zarr: %d dimension_names for %d dimensions", len(m.DimensionNames), len(m.Shape))
	}
	if m.DataType, err = parseDataType(raw.DataType); err != nil {
		return nil, err
	}
	if dt, ok := m.doc.get("data_type"); ok {
		if canon, changed := canonicalTimeUnits(dt); changed {
			m.doc.set("data_type", canon) // as zarr-python writes it back
		}
	}
	if m.grid, m.ChunkShape, err = parseChunkGrid(raw.ChunkGrid, m.Shape); err != nil {
		return nil, err
	}
	if _, ok := m.grid.(*rectilinearGrid); ok {
		// zarr-python writes rectilinear grids back in its run-length form.
		m.doc.set("chunk_grid", m.grid.json())
	}
	if m.ChunkKeyEncoding, err = parseChunkKeyEncoding(raw.ChunkKeyEncoding, len(m.Shape)); err != nil {
		return nil, err
	}
	if len(raw.FillValue) == 0 {
		return nil, fmt.Errorf("zarr: zarr.json has no fill_value")
	}
	if m.fill, err = parseFillValue(raw.FillValue, m.DataType); err != nil {
		return nil, err
	}
	// zarr-python writes the fill value back from what it read ("NaT" as
	// its integer, a struct's base64 bytes as an object of its fields).
	m.doc.set("fill_value", fillJSON(m.DataType, m.fill, 3))
	if len(m.Codecs) == 0 {
		return nil, fmt.Errorf("zarr: zarr.json has no codecs")
	}
	return m, nil
}

func parseChunkKeyEncoding(raw json.RawMessage, ndim int) (ChunkKeyEncoding, error) {
	if len(raw) == 0 {
		return ChunkKeyEncoding{Name: "default", Separator: "/"}, nil
	}
	var e struct {
		Name          string          `json:"name"`
		Configuration json.RawMessage `json:"configuration"`
	}
	if json.Unmarshal(raw, &e.Name) != nil {
		if err := json.Unmarshal(raw, &e); err != nil {
			return ChunkKeyEncoding{}, fmt.Errorf("zarr: invalid chunk_key_encoding %s", raw)
		}
	}
	var k ChunkKeyEncoding
	switch e.Name {
	case "default", "v2":
		var cfg struct {
			Separator string `json:"separator"`
		}
		if err := parseConfig(e.Configuration, &cfg); err != nil {
			return k, fmt.Errorf("zarr: invalid chunk_key_encoding %s", raw)
		}
		k = ChunkKeyEncoding{Name: e.Name, Separator: cfg.Separator}
		k.Separator = k.separator()
	case "template":
		var cfg map[string]json.RawMessage
		if err := json.Unmarshal(e.Configuration, &cfg); err != nil || cfg == nil {
			return k, fmt.Errorf("zarr: invalid chunk_key_encoding %s", raw)
		}
		k.Name = e.Name
		for key, v := range cfg {
			var err error
			switch key {
			case "format":
				err = json.Unmarshal(v, &k.Template)
			case "separator":
				if string(v) != "null" {
					err = json.Unmarshal(v, &k.Separator)
				}
			default:
				err = fmt.Errorf("unknown field %q", key)
			}
			if err != nil {
				return k, fmt.Errorf("zarr: invalid template chunk_key_encoding %s: %v", raw, err)
			}
		}
		if _, ok := cfg["format"]; !ok {
			return k, fmt.Errorf("zarr: template chunk_key_encoding without a format: %s", raw)
		}
	default:
		return ChunkKeyEncoding{}, fmt.Errorf("%w: chunk key encoding %q", ErrUnsupported, e.Name)
	}
	return k, k.check(ndim)
}

// ParseMetadataV2 parses a Zarr v2 array: its .zarray document and its
// .zattrs document (nil if absent).
func ParseMetadataV2(zarray, zattrs []byte) (*Metadata, error) {
	var raw struct {
		ZarrFormat         int               `json:"zarr_format"`
		Shape              []uint64          `json:"shape"`
		Chunks             []uint64          `json:"chunks"`
		DType              json.RawMessage   `json:"dtype"`
		FillValue          json.RawMessage   `json:"fill_value"`
		Order              string            `json:"order"`
		Filters            []json.RawMessage `json:"filters"`
		Compressor         json.RawMessage   `json:"compressor"`
		DimensionSeparator string            `json:"dimension_separator"`
	}
	// Repaired first: stored chunk sizes may be JSON booleans.
	ordered, repaired, stored, err := orderedDocument(zarray, 2)
	if err != nil {
		return nil, fmt.Errorf("zarr: invalid .zarray: %w", err)
	}
	if repaired != nil {
		zarray = repaired
	}
	if err := json.Unmarshal(zarray, &raw); err != nil {
		return nil, fmt.Errorf("zarr: invalid .zarray: %w", err)
	}
	if raw.ZarrFormat != 2 {
		return nil, fmt.Errorf("zarr: .zarray has zarr_format %d", raw.ZarrFormat)
	}
	m := &Metadata{ZarrFormat: 2, NodeType: "array", Shape: raw.Shape, ChunkShape: raw.Chunks, FillValue: raw.FillValue, Order: raw.Order, doc: ordered, stored: stored}
	if m.Shape == nil {
		m.Shape = []uint64{}
	}
	if m.ChunkShape == nil {
		m.ChunkShape = []uint64{}
	}
	if err := checkShape(m.Shape); err != nil {
		return nil, err
	}
	if len(m.ChunkShape) != len(m.Shape) {
		return nil, fmt.Errorf("zarr: chunks %v do not match shape %v", m.ChunkShape, m.Shape)
	}
	for _, c := range m.ChunkShape {
		if c == 0 {
			return nil, fmt.Errorf("zarr: chunks contain 0")
		}
	}
	m.grid = regularGrid(m.ChunkShape)
	if m.Order == "" {
		m.Order = "C"
	}
	if m.Order != "C" && m.Order != "F" {
		return nil, fmt.Errorf("zarr: invalid order %q", m.Order)
	}
	sep := raw.DimensionSeparator
	if sep == "" {
		sep = "."
	}
	if sep != "." && sep != "/" {
		return nil, fmt.Errorf("zarr: invalid dimension_separator %q", sep)
	}
	m.ChunkKeyEncoding = ChunkKeyEncoding{Name: "v2", Separator: sep}
	parseNumcodec := func(r json.RawMessage) (CodecSpec, error) {
		var obj map[string]json.RawMessage
		if err := json.Unmarshal(r, &obj); err != nil {
			return CodecSpec{}, fmt.Errorf("zarr: invalid codec %s", r)
		}
		var id string
		if err := json.Unmarshal(obj["id"], &id); err != nil || id == "" {
			return CodecSpec{}, fmt.Errorf("zarr: codec without id: %s", r)
		}
		o, err := parseOrdered(r)
		if err != nil {
			return CodecSpec{}, err
		}
		cfg := o.(jsonObject)
		cfg.del("id")
		b, _ := marshalPython(cfg, -1)
		return CodecSpec{Name: id, Configuration: b}, nil
	}
	objectCodec := ""
	for _, f := range raw.Filters {
		c, err := parseNumcodec(f)
		if err != nil {
			return nil, err
		}
		m.Filters = append(m.Filters, c)
		if objectCodec == "" && isObjectCodec(c.Name) {
			objectCodec = c.Name
		}
	}
	if len(raw.Compressor) > 0 && string(raw.Compressor) != "null" {
		c, err := parseNumcodec(raw.Compressor)
		if err != nil {
			return nil, err
		}
		m.Compressor = &c
		if objectCodec == "" && isObjectCodec(c.Name) {
			objectCodec = c.Name
		}
	}
	if m.DataType, m.BigEndian, err = parseV2DataType(raw.DType, objectCodec); err != nil {
		return nil, err
	}
	if m.fill, err = parseV2Fill(raw.FillValue, m.DataType); err != nil {
		return nil, err
	}
	if doc, err := m.v2Document(len(raw.FillValue) == 0 || string(raw.FillValue) == "null"); err == nil {
		m.doc = doc
	}
	if zattrs != nil {
		if m.Attributes, err = decodeAttributes(pyCompat(zattrs)); err != nil {
			return nil, err
		}
		m.rawAttrs, _ = parseOrdered(zattrs)
	} else {
		m.Attributes = map[string]any{}
	}
	return m, nil
}

// v2Document renders the .zarray as zarr-python writes back what it read
// (ArrayV2Metadata.to_dict): its keys in order, unknown keys dropped, codec
// configurations as numcodecs' get_config() gives them, and the dtype in
// NumPy's spelling: "|" for single-byte and raw types stored with "<" or
// ">" (as netCDF-C, Zarr.jl and jzarr write them), "us" for "μs".
func (m *Metadata) v2Document(nullFill bool) (jsonObject, error) {
	dt, err := m.DataType.v2JSON(m.BigEndian)
	if err != nil {
		return nil, err
	}
	var fill any
	if !nullFill {
		fill = fillJSON(m.DataType, m.fill, 2)
	}
	var filters any
	if len(m.Filters) > 0 {
		fs := make([]any, len(m.Filters))
		for i, f := range m.Filters {
			if fs[i], err = numcodecJSON(f); err != nil {
				return nil, err
			}
		}
		filters = fs
	}
	var compressor any
	if m.Compressor != nil {
		if compressor, err = numcodecJSON(*m.Compressor); err != nil {
			return nil, err
		}
	}
	return jsonObject{
		{"shape", m.Shape},
		{"chunks", m.ChunkShape},
		{"dtype", dt},
		{"fill_value", fill},
		{"order", m.Order},
		{"filters", filters},
		{"dimension_separator", m.ChunkKeyEncoding.Separator},
		{"compressor", compressor},
		{"zarr_format", 2},
	}, nil
}

func isObjectCodec(id string) bool {
	switch id {
	case "vlen-utf8", "vlen-bytes", "vlen-array", "json2", "msgpack2", "pickle", "categorize":
		return true
	}
	return false
}

// parseV2Fill decodes a v2 fill value; null means the type's default (NaT
// for datetimes), as zarr-python reads it.
func parseV2Fill(raw json.RawMessage, d DataType) (fillValue, error) {
	if len(raw) == 0 || string(raw) == "null" {
		if d.Kind == KindDatetime || d.Kind == KindTimedelta {
			return parseFillValue(json.RawMessage(`"NaT"`), d)
		}
		return parseFillValue(nil, d)
	}
	var s string
	if json.Unmarshal(raw, &s) == nil && strings.HasPrefix(s, "0x") && d.Kind == KindFloat {
		return fillValue{}, fmt.Errorf("zarr: hex fill_value %q is not valid in Zarr v2", s)
	}
	return parseFillValue(raw, d)
}

// Attributes extracts the attributes of a zarr.json document (array or
// group), with numbers decoded as decodeAttributes does.
func Attributes(doc []byte) (map[string]any, error) {
	var raw struct {
		Attributes json.RawMessage `json:"attributes"`
	}
	if err := json.Unmarshal(pyCompat(doc), &raw); err != nil {
		return nil, fmt.Errorf("zarr: invalid zarr.json: %w", err)
	}
	return decodeAttributes(raw.Attributes)
}

// decodeAttributes decodes user attributes keeping integers exact, as
// zarr-python (Python's json) does: integers become int64, or uint64 above
// int64's range, or json.Number beyond 64 bits; other numbers float64.
// Python's bare NaN and Infinity tokens are accepted.
func decodeAttributes(raw json.RawMessage) (map[string]any, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return map[string]any{}, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var attrs map[string]any
	if err := dec.Decode(&attrs); err != nil {
		fixed, ok := pythonNonFinite(raw)
		if !ok {
			return nil, fmt.Errorf("zarr: invalid attributes: %w", err)
		}
		return decodeAttributes(fixed)
	}
	if attrs == nil {
		return map[string]any{}, nil
	}
	return exactNumbers(attrs).(map[string]any), nil
}

func exactNumbers(v any) any {
	switch x := v.(type) {
	case map[string]any:
		for k, e := range x {
			x[k] = exactNumbers(e)
		}
	case []any:
		for i, e := range x {
			x[i] = exactNumbers(e)
		}
	case json.Number:
		s := x.String()
		if !strings.ContainsAny(s, ".eE") {
			if i, err := strconv.ParseInt(s, 10, 64); err == nil {
				return i
			}
			if u, err := strconv.ParseUint(s, 10, 64); err == nil {
				return u
			}
			return x
		}
		f, _ := x.Float64()
		return f
	case string:
		switch x {
		case nanToken:
			return nanValue()
		case infToken:
			return infValue(1)
		case negInfToken:
			return infValue(-1)
		}
	}
	return v
}

// ---------------------------------------------------------------------------
// Rendering

// Document returns the zarr.json (Zarr v3) or .zarray (Zarr v2) document,
// formatted as zarr-python writes it.
func (m *Metadata) Document() ([]byte, error) {
	return marshalPython(m.doc, 2)
}

// AttributesDocument returns the .zattrs document of a Zarr v2 array.
func (m *Metadata) AttributesDocument() ([]byte, error) {
	if m.rawAttrs != nil {
		return marshalPython(m.rawAttrs, 2)
	}
	return marshalPython(attrsJSON(m.Attributes), 2)
}

// documents returns the files that store the metadata, by name.
func (m *Metadata) documents() (map[string][]byte, error) {
	doc, err := m.Document()
	if err != nil {
		return nil, err
	}
	if m.ZarrFormat == 2 {
		attrs, err := m.AttributesDocument()
		if err != nil {
			return nil, err
		}
		return map[string][]byte{".zarray": doc, ".zattrs": attrs}, nil
	}
	return map[string][]byte{"zarr.json": doc}, nil
}

// setAttributes replaces the attributes.
func (m *Metadata) setAttributes(attrs map[string]any) error {
	if attrs == nil {
		attrs = map[string]any{}
	}
	raw, err := marshalPython(attrsJSON(attrs), -1)
	if err != nil {
		return err
	}
	if m.Attributes, err = decodeAttributes(raw); err != nil {
		return err
	}
	m.rawAttrs = attrsJSON(attrs)
	if m.ZarrFormat == 3 {
		m.doc.set("attributes", m.rawAttrs)
	}
	return nil
}

// setShape changes the shape (resizing).
func (m *Metadata) setShape(shape []uint64) {
	old := m.Shape
	m.Shape = slices.Clone(shape)
	m.grid = m.grid.resized(old, shape)
	m.doc.set("shape", m.Shape)
	if _, ok := m.grid.(*rectilinearGrid); ok {
		m.doc.set("chunk_grid", m.grid.json())
	}
}

// attrsJSON prepares attributes for writing: maps get sorted keys (Go maps
// have no order) and non-finite floats Python's bare tokens.
func attrsJSON(attrs map[string]any) any {
	if attrs == nil {
		return jsonObject{}
	}
	return attrsValue(attrs)
}

func attrsValue(v any) any {
	switch x := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		obj := make(jsonObject, len(keys))
		for i, k := range keys {
			obj[i] = jsonMember{k, attrsValue(x[k])}
		}
		return obj
	case []any:
		out := make([]any, len(x))
		for i, e := range x {
			out[i] = attrsValue(e)
		}
		return out
	}
	return v
}

// GridShape returns the number of chunks along each dimension.
func (m *Metadata) GridShape() []uint64 {
	out := make([]uint64, len(m.Shape))
	for d, n := range m.Shape {
		out[d] = m.grid.numChunks(d, n)
	}
	return out
}

// ChunkKey returns the chunk's store key relative to the array, following
// the array's chunk key encoding (e.g. "c/0/1", or "0.1" for Zarr v2).
func (m *Metadata) ChunkKey(coords []uint64) string { return m.ChunkKeyEncoding.Key(coords) }

// ChunkBounds returns the start position and shape of the chunk at coords
// (the full chunk, which may extend past the array's edge).
func (m *Metadata) ChunkBounds(coords []uint64) (start, shape []uint64) {
	start, shape = make([]uint64, len(coords)), make([]uint64, len(coords))
	for d, c := range coords {
		start[d], shape[d] = m.grid.span(d, c)
	}
	return start, shape
}

// nominalChunkShape is the shape of chunk 0, standing in for the chunk
// shape of rectilinear grids when codecs are set up.
func (m *Metadata) nominalChunkShape() []uint64 {
	if m.ChunkShape != nil {
		return m.ChunkShape
	}
	out := make([]uint64, len(m.Shape))
	for d := range out {
		_, out[d] = m.grid.span(d, 0)
		out[d] = max(out[d], 1)
	}
	return out
}

// largestChunk is the shape of the largest chunk (shard) along each
// dimension.
func (m *Metadata) largestChunk() []uint64 {
	g, ok := m.grid.(*rectilinearGrid)
	if !ok {
		return m.nominalChunkShape()
	}
	out := m.nominalChunkShape()
	for d := range out {
		if g.step[d] > 0 {
			continue
		}
		for c, nc := uint64(1), g.numChunks(d, m.Shape[d]); c < nc; c++ {
			_, n := g.span(d, c)
			out[d] = max(out[d], n)
		}
	}
	return out
}

// checkShape refuses lengths beyond int64's range, which NumPy and the
// selection arithmetic cannot index.
func checkShape(shape []uint64) error {
	for _, n := range shape {
		if n > math.MaxInt64 {
			return fmt.Errorf("zarr: shape %v is too large", shape)
		}
	}
	return nil
}

// FillValueArray returns the fill value as a 0-d array.
func (m *Metadata) FillValueArray() *NDArray { return m.fill.ndarray(m.DataType) }

// canonicalTimeUnits respells the microsecond unit "μs" as "us" in a Zarr
// v3 data_type (including struct fields), as zarr-python writes it.
func canonicalTimeUnits(v any) (any, bool) {
	obj, ok := v.(jsonObject)
	if !ok {
		return v, false
	}
	name, _ := obj.get("name")
	c, _ := obj.get("configuration")
	cfg, ok := c.(jsonObject)
	if !ok {
		return v, false
	}
	switch name {
	case "numpy.datetime64", "numpy.timedelta64":
		if u, _ := cfg.get("unit"); u != "μs" {
			return v, false
		}
		cfg = slices.Clone(cfg)
		cfg.set("unit", "us")
	case "struct", "structured":
		f, _ := cfg.get("fields")
		fields, ok := f.([]any)
		if !ok {
			return v, false
		}
		var out []any
		for i, field := range fields {
			var nf any
			switch x := field.(type) {
			case jsonObject:
				dt, _ := x.get("data_type")
				if ndt, changed := canonicalTimeUnits(dt); changed {
					y := slices.Clone(x)
					y.set("data_type", ndt)
					nf = y
				}
			case []any:
				if len(x) == 2 {
					if ndt, changed := canonicalTimeUnits(x[1]); changed {
						nf = []any{x[0], ndt}
					}
				}
			}
			if nf != nil {
				if out == nil {
					out = slices.Clone(fields)
				}
				out[i] = nf
			}
		}
		if out == nil {
			return v, false
		}
		cfg = slices.Clone(cfg)
		cfg.set("fields", out)
	default:
		return v, false
	}
	obj = slices.Clone(obj)
	obj.set("configuration", cfg)
	return obj, true
}
