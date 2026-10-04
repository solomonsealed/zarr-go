package zarr

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// CodecSpec is one entry of a codecs list: a codec name and its JSON
// configuration. For Zarr v2 filters and compressors, Name is the
// numcodecs id and Configuration the whole numcodecs config object.
type CodecSpec struct {
	Name          string
	Configuration json.RawMessage
}

func (c *CodecSpec) UnmarshalJSON(b []byte) error {
	var name string
	if json.Unmarshal(b, &name) == nil {
		c.Name, c.Configuration = name, nil
		return nil
	}
	var obj struct {
		Name          string          `json:"name"`
		Configuration json.RawMessage `json:"configuration"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return err
	}
	c.Name, c.Configuration = obj.Name, obj.Configuration
	return nil
}

func (c CodecSpec) MarshalJSON() ([]byte, error) { return marshalPython(c.json(), -1) }

func (c CodecSpec) json() any {
	obj := jsonObject{{"name", c.Name}}
	if len(c.Configuration) > 0 {
		obj = append(obj, jsonMember{"configuration", c.Configuration})
	}
	return obj
}

// Codec builds a CodecSpec from a name and configuration. A nil config is
// written as {} (zarr-python requires a configuration for numcodecs.*).
// Configuration keys are written in sorted order; use CodecJSON to control
// the exact JSON.
func Codec(name string, config map[string]any) CodecSpec {
	if config == nil {
		config = map[string]any{}
	}
	c := CodecSpec{Name: name}
	c.Configuration, _ = marshalPython(config, -1)
	return c
}

// CodecJSON builds a CodecSpec from a name and raw JSON configuration
// (nil for none).
func CodecJSON(name string, config string) CodecSpec {
	c := CodecSpec{Name: name}
	if config != "" {
		c.Configuration = json.RawMessage(config)
	}
	return c
}

// ChunkSpec describes chunk data at one stage of a codec pipeline.
type ChunkSpec struct {
	Shape    []uint64
	DataType DataType
	// FillValue is a 0-d array holding the fill value at this stage.
	FillValue *NDArray

	bigEndian bool // Zarr v2: the dtype's stored byte order
	v2        bool // building a Zarr v2 pipeline
}

func (s ChunkSpec) fill() fillValue { return fillOf(s.FillValue) }

func (s ChunkSpec) withShape(shape []uint64) ChunkSpec {
	s.Shape = shape
	return s
}

// ArrayToArrayCodec transforms an array into another array (transpose,
// delta, scale_offset, ...).
type ArrayToArrayCodec interface {
	// Resolve returns the shape, data type and fill value of the encoded
	// representation of chunks described by in.
	Resolve(in ChunkSpec) (ChunkSpec, error)
	// Encode transforms a chunk described by in.
	Encode(ctx context.Context, a *NDArray, in ChunkSpec) (*NDArray, error)
	// Decode reverses Encode, producing a chunk described by in.
	Decode(ctx context.Context, a *NDArray, in ChunkSpec) (*NDArray, error)
}

// ArrayToBytesCodec serializes an array (bytes, vlen-utf8,
// sharding_indexed, ...). Each pipeline has exactly one.
type ArrayToBytesCodec interface {
	Encode(ctx context.Context, a *NDArray, in ChunkSpec) ([]byte, error)
	Decode(ctx context.Context, b []byte, in ChunkSpec) (*NDArray, error)
}

// BytesCodec is a bytes-to-bytes codec (compressor, checksum, ...). Codecs
// that can also encode implement BytesEncoder.
type BytesCodec interface {
	Decode(in []byte) ([]byte, error)
}

// BytesEncoder is implemented by bytes-to-bytes codecs that can encode,
// which writing arrays requires.
type BytesEncoder interface {
	Encode(in []byte) ([]byte, error)
}

// BytesToBytesCodec is a bytes-to-bytes codec that can encode and decode.
type BytesToBytesCodec interface {
	BytesCodec
	BytesEncoder
}

// EncodedSizer is implemented by codecs whose output size depends only on
// their input size (checksums, ...). Codecs in a sharding index pipeline
// must have a fixed size.
type EncodedSizer interface {
	EncodedSize(inputSize int) int
}

// BytesCodecFunc adapts a decoding function to BytesCodec.
type BytesCodecFunc func(in []byte) ([]byte, error)

// Decode implements BytesCodec.
func (f BytesCodecFunc) Decode(in []byte) ([]byte, error) { return f(in) }

// CodecFactory builds a codec from its JSON configuration. in describes
// the chunks the codec receives when encoding (its data type matters for
// e.g. blosc's typesize). It returns an ArrayToArrayCodec,
// ArrayToBytesCodec or BytesCodec.
type CodecFactory func(config json.RawMessage, in ChunkSpec) (any, error)

// BytesCodecFactory builds a bytes-to-bytes codec from its JSON
// configuration. elemSize is the size in bytes of the array's elements at
// that pipeline stage.
type BytesCodecFactory func(config json.RawMessage, elemSize int) (BytesCodec, error)

var (
	registryMu sync.RWMutex
	registry   = map[string]CodecFactory{}
)

// RegisterCodec makes a codec of any kind available by name, replacing
// any existing codec with that name. Names of numcodecs codecs used by
// Zarr v2 arrays are looked up as "numcodecs.<id>", then "<id>".
func RegisterCodec(name string, f CodecFactory) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registry[name] = f
}

// RegisterBytesCodec makes a bytes-to-bytes codec available by name.
func RegisterBytesCodec(name string, f BytesCodecFactory) {
	RegisterCodec(name, func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		return f(cfg, in.DataType.Size)
	})
}

func lookupCodec(name string) (CodecFactory, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	f, ok := registry[name]
	return f, ok
}

// newCodec instantiates a codec; v2 looks up numcodecs ids.
func newCodec(s CodecSpec, in ChunkSpec, v2 bool) (any, error) {
	in.v2 = v2
	f, ok := lookupCodec(s.Name)
	if v2 {
		if g, ok2 := lookupCodec("numcodecs." + s.Name); ok2 {
			f, ok = g, true
		}
	}
	if !ok {
		return nil, fmt.Errorf("%w: codec %q", ErrUnsupported, s.Name)
	}
	c, err := f(s.Configuration, in)
	if err != nil {
		return nil, fmt.Errorf("zarr: codec %q: %w", s.Name, err)
	}
	switch c.(type) {
	case ArrayToArrayCodec, ArrayToBytesCodec, BytesCodec:
		return c, nil
	}
	return nil, fmt.Errorf("zarr: codec %q has an unknown kind %T", s.Name, c)
}

// NewCodec instantiates a registered codec for chunks described by in. It
// returns an ArrayToArrayCodec, ArrayToBytesCodec or BytesCodec.
func NewCodec(spec CodecSpec, in ChunkSpec) (any, error) {
	if in.FillValue == nil && !in.DataType.Variable() && in.DataType.Kind != 0 {
		in.FillValue = fillValue{fixed: make([]byte, in.DataType.Size)}.ndarray(in.DataType)
	}
	return newCodec(spec, in, false)
}

func parseConfig(cfg json.RawMessage, v any) error {
	if len(cfg) == 0 || string(cfg) == "null" {
		return nil
	}
	return json.Unmarshal(cfg, v)
}

// ---------------------------------------------------------------------------
// Pipelines

// pipeline is a codec chain: array-to-array codecs, one array-to-bytes
// codec and bytes-to-bytes codecs.
type pipeline struct {
	aa []ArrayToArrayCodec
	ab ArrayToBytesCodec
	bb []BytesCodec
	// specs are the codecs' names, for error messages.
	names []string
}

// buildPipeline instantiates and orders codecs for chunks described by in.
// For Zarr v2 (v2 true), names are numcodecs ids and the bytes codec is
// implied; array codecs after bytes codecs work on the bytes reinterpreted
// as the current data type, as numcodecs does.
func buildPipeline(specs []CodecSpec, in ChunkSpec, v2 bool) (*pipeline, error) {
	p := &pipeline{}
	cur := in
	for _, s := range specs {
		c, err := newCodec(s, cur, v2)
		if err != nil {
			return nil, err
		}
		switch c := c.(type) {
		case ArrayToArrayCodec:
			if p.ab != nil {
				return nil, fmt.Errorf("zarr: array-to-array codec %q after the array-to-bytes codec", s.Name)
			}
			p.aa = append(p.aa, c)
			if cur, err = c.Resolve(cur); err != nil {
				return nil, fmt.Errorf("zarr: codec %q: %w", s.Name, err)
			}
		case ArrayToBytesCodec:
			if p.ab != nil {
				return nil, fmt.Errorf("zarr: more than one array-to-bytes codec (%q)", s.Name)
			}
			if len(p.bb) > 0 {
				return nil, fmt.Errorf("zarr: array-to-bytes codec %q after a bytes-to-bytes codec", s.Name)
			}
			p.ab = c
		case BytesCodec:
			if p.ab == nil {
				// The bytes codec is implied when missing (single-byte
				// types written by older tools, and Zarr v2).
				if p.ab, err = newBytesSerializer(cur, v2BigEndian(p.aa, in, v2)); err != nil {
					return nil, err
				}
				p.names = append(p.names, "bytes")
			}
			p.bb = append(p.bb, c)
		}
		p.names = append(p.names, s.Name)
	}
	if p.ab == nil {
		ab, err := newBytesSerializer(cur, v2BigEndian(p.aa, in, v2))
		if err != nil {
			return nil, err
		}
		p.ab = ab
		p.names = append(p.names, "bytes")
	}
	return p, nil
}

// byteOrderer is implemented by Zarr v2 filters whose output data type
// names a byte order (numcodecs "astype" strings such as ">i2").
type byteOrderer interface {
	outputBigEndian() (big, known bool)
}

// v2BigEndian returns the byte order of the implied bytes codec of a v2
// pipeline: that of the last filter naming one, else of the array's dtype.
func v2BigEndian(aa []ArrayToArrayCodec, in ChunkSpec, v2 bool) bool {
	if !v2 {
		return false
	}
	for i := len(aa) - 1; i >= 0; i-- {
		if bo, ok := aa[i].(byteOrderer); ok {
			if big, known := bo.outputBigEndian(); known {
				return big
			}
		}
	}
	return in.bigEndian
}

// resolve returns the chunk spec entering each array-to-array codec and,
// last, the array-to-bytes codec.
func (p *pipeline) resolve(in ChunkSpec) ([]ChunkSpec, error) {
	specs := make([]ChunkSpec, len(p.aa)+1)
	specs[0] = in
	for i, c := range p.aa {
		var err error
		if specs[i+1], err = c.Resolve(specs[i]); err != nil {
			return nil, err
		}
	}
	return specs, nil
}

func (p *pipeline) decode(ctx context.Context, data []byte, in ChunkSpec) (*NDArray, error) {
	var err error
	for i := len(p.bb) - 1; i >= 0; i-- {
		if data, err = p.bb[i].Decode(data); err != nil {
			return nil, err
		}
	}
	specs, err := p.resolve(in)
	if err != nil {
		return nil, err
	}
	a, err := p.ab.Decode(ctx, data, specs[len(p.aa)])
	if err != nil {
		return nil, err
	}
	for i := len(p.aa) - 1; i >= 0; i-- {
		if a, err = p.aa[i].Decode(ctx, a, specs[i]); err != nil {
			return nil, err
		}
	}
	if err := a.validate(); err != nil {
		return nil, fmt.Errorf("zarr: decoded chunk: %w", err)
	}
	a.Shape, a.DataType = in.Shape, in.DataType
	return a, nil
}

func (p *pipeline) encode(ctx context.Context, a *NDArray, in ChunkSpec) ([]byte, error) {
	specs, err := p.resolve(in)
	if err != nil {
		return nil, err
	}
	for i, c := range p.aa {
		if a, err = c.Encode(ctx, a, specs[i]); err != nil {
			return nil, err
		}
	}
	out, err := p.ab.Encode(ctx, a, specs[len(p.aa)])
	if err != nil {
		return nil, err
	}
	for i, c := range p.bb {
		enc, ok := c.(BytesEncoder)
		if !ok {
			return nil, fmt.Errorf("%w: encoding with codec %q", ErrUnsupported, p.names[len(p.aa)+1+i])
		}
		if out, err = enc.Encode(out); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// encodedSize returns the encoded size of chunks described by in, if it
// depends only on their shape.
func (p *pipeline) encodedSize(in ChunkSpec) (int, bool) {
	specs, err := p.resolve(in)
	if err != nil {
		return 0, false
	}
	last := specs[len(specs)-1]
	sz, ok := p.ab.(interface {
		encodedSize(ChunkSpec) (int, bool)
	})
	if !ok {
		return 0, false
	}
	n, ok := sz.encodedSize(last)
	if !ok {
		return 0, false
	}
	for _, c := range p.bb {
		s, ok := c.(EncodedSizer)
		if !ok {
			return 0, false
		}
		n = s.EncodedSize(n)
	}
	return n, true
}

// sharding returns the sharding codec if it is the pipeline's only codec,
// in which case shards can be read and written partially.
func (p *pipeline) sharding() *shardingCodec {
	if sc, ok := p.ab.(*shardingCodec); ok && len(p.aa) == 0 && len(p.bb) == 0 {
		return sc
	}
	return nil
}

func (p *pipeline) String() string { return strings.Join(p.names, ", ") }
