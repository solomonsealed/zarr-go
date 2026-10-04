package zarr

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// Array-to-array codecs: numcodecs filters (delta, fixedscaleoffset,
// quantize, astype, packbits, bitround, categorize) and the Zarr v3
// scale_offset and cast_value codecs. Arithmetic follows NumPy 2: Python
// numbers in configurations are "weak" (an operation on a float32 array
// stays float32), float16 operations compute in float32 and round, and
// integer arithmetic wraps.

// parseNumpyDType parses a NumPy dtype as numcodecs configurations write
// it: "<i4", "i4", "|u1", "float64", "bool", "<U5", "|O".
func parseNumpyDType(s string) (DataType, bool, error) {
	if dt, err := ParseDataType(s); err == nil {
		return dt, false, nil
	}
	raw, _ := json.Marshal(s)
	return parseV2DataType(raw, "vlen-utf8")
}

// numericValue reads element i of a real numeric or bool array as float64
// (floats) or as int64/uint64 bits (integers).
type numeric struct {
	isFloat  bool
	unsigned bool
	f        float64
	i        int64
	u        uint64
}

func readNumeric(dt DataType, b []byte) numeric {
	switch dt.Kind {
	case KindFloat:
		return numeric{isFloat: true, f: floatAt(dt, b)}
	case KindUint:
		return numeric{unsigned: true, u: uintAt(b, dt.Size, dt.Bits)}
	case KindBool:
		return numeric{unsigned: true, u: uint64(b[0] & 1)}
	}
	return numeric{i: intAt(b, dt.Size, dt.Bits)}
}

func (n numeric) float() float64 {
	switch {
	case n.isFloat:
		return n.f
	case n.unsigned:
		return float64(n.u)
	}
	return float64(n.i)
}

func (n numeric) bits() uint64 {
	if n.unsigned {
		return n.u
	}
	return uint64(n.i)
}

// writeNumeric stores n as dt with NumPy's astype rules: integers wrap,
// floats truncate towards zero when converted to integers, and narrow
// floats round to nearest even.
func writeNumeric(dt DataType, b []byte, n numeric) {
	switch dt.Kind {
	case KindFloat:
		putFloat(b, dt, roundTo(dt, n.float()))
	case KindBool:
		if n.float() != 0 || (!n.isFloat && n.bits() != 0) {
			b[0] = 1
		} else {
			b[0] = 0
		}
	default:
		var v uint64
		if n.isFloat {
			f := math.Trunc(n.f)
			if f >= 0 && f >= math.MaxInt64 {
				v = uint64(f)
			} else {
				v = uint64(int64(f))
			}
		} else {
			v = n.bits()
		}
		if dt.Bits != 0 {
			v &= 1<<dt.Bits - 1
		}
		putUint(b[:dt.Size], v)
	}
}

// roundTo rounds f to the precision of float type dt.
func roundTo(dt DataType, f float64) float64 {
	if ff := dt.floatFormat(); ff != nil {
		return ff.decode(ff.encode(f))
	}
	if dt.Size == 4 {
		return float64(float32(f))
	}
	return f
}

// floatOp applies op in the arithmetic of float type dt (float16 and
// narrower compute in float32 and round, as NumPy does).
func floatOp(dt DataType, op func(a, b float64) float64, a, b float64) float64 {
	switch {
	case dt.Size == 8:
		return op(a, b)
	case dt.Size == 4 && dt.floatFormat() == nil:
		return float64(float32(op(a, b)))
	default:
		return roundTo(dt, float64(float32(op(a, b))))
	}
}

func sub(a, b float64) float64 { return a - b }
func add(a, b float64) float64 { return a + b }
func mul(a, b float64) float64 { return a * b }
func div(a, b float64) float64 { return a / b }

func elementCount(a *NDArray) int { return int(numElements(a.Shape)) }

// mapArray converts every element of a (type from) into a new array of
// type to and shape.
func mapArray(a *NDArray, from, to DataType, shape []uint64, f func(numeric) numeric) *NDArray {
	n := elementCount(a)
	out := &NDArray{Shape: shape, DataType: to, Data: make([]byte, n*to.Size)}
	for i := 0; i < n; i++ {
		writeNumeric(to, out.Data[i*to.Size:], f(readNumeric(from, a.Data[i*from.Size:])))
	}
	return out
}

// ---------------------------------------------------------------------------
// numcodecs filters

// numcodecsFilter is a shape-preserving filter with an optional change of
// data type (astype).
type numcodecsFilter struct {
	dtype, astype  DataType
	astypeBig      bool
	astypeBigKnown bool
	encode, decode func(a *NDArray, in, out DataType) (*NDArray, error)
	requireFloat   bool
	resolveAstype  bool // the encoded type is astype (delta, fixedscaleoffset, quantize, astype)
}

func (f *numcodecsFilter) Resolve(in ChunkSpec) (ChunkSpec, error) {
	if f.requireFloat && in.DataType.Kind != KindFloat {
		return in, fmt.Errorf("needs floating-point data, not %s", in.DataType.Name)
	}
	if in.DataType.Variable() {
		return in, fmt.Errorf("does not support %s data", in.DataType.Name)
	}
	out := in
	if f.resolveAstype {
		out.DataType = f.astype
		// The fill value through the filter.
		if in.FillValue != nil {
			enc, err := f.encode(in.FillValue, in.DataType, f.astype)
			if err == nil {
				out.FillValue = enc
				out.FillValue.Shape = []uint64{}
			}
		}
	}
	return out, nil
}

func (f *numcodecsFilter) outputBigEndian() (bool, bool) { return f.astypeBig, f.astypeBigKnown }

func (f *numcodecsFilter) Encode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	out, err := f.encode(a, in.DataType, f.astype)
	if err != nil {
		return nil, err
	}
	out.Shape = in.Shape
	return out, nil
}

func (f *numcodecsFilter) Decode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	out, err := f.decode(a, f.astype, in.DataType)
	if err != nil {
		return nil, err
	}
	out.Shape = in.Shape
	return out, nil
}

func init() {
	// parse reads dtype and astype ("astype" defaults to dtype; dtype
	// defaults to the chunk's data type).
	parse := func(in ChunkSpec, dtypeStr, astypeStr string) (*numcodecsFilter, error) {
		f := &numcodecsFilter{dtype: in.DataType, astype: in.DataType, astypeBig: in.bigEndian}
		if dtypeStr != "" {
			dt, _, err := parseNumpyDType(dtypeStr)
			if err != nil {
				return nil, err
			}
			if in.DataType.Kind != 0 && dt.Size != in.DataType.Size {
				return nil, fmt.Errorf("dtype %s does not match the %s data", dtypeStr, in.DataType.Name)
			}
			f.dtype, f.astype = dt, dt
		}
		if astypeStr != "" {
			dt, big, err := parseNumpyDType(astypeStr)
			if err != nil {
				return nil, err
			}
			f.astype, f.astypeBig = dt, big
			f.astypeBigKnown = strings.HasPrefix(astypeStr, "<") || strings.HasPrefix(astypeStr, ">")
		}
		return f, nil
	}

	RegisterCodec("numcodecs.delta", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct{ DType, AsType string }
		if err := parseConfig(cfg, &struct {
			DType  *string `json:"dtype"`
			AsType *string `json:"astype"`
		}{&c.DType, &c.AsType}); err != nil {
			return nil, err
		}
		f, err := parse(in, c.DType, c.AsType)
		if err != nil {
			return nil, err
		}
		f.resolveAstype = true
		f.encode = func(a *NDArray, from, to DataType) (*NDArray, error) {
			n := elementCount(a)
			out := &NDArray{Shape: a.Shape, DataType: to, Data: make([]byte, n*to.Size)}
			var prev numeric
			for i := 0; i < n; i++ {
				cur := readNumeric(from, a.Data[i*from.Size:])
				d := cur
				if i > 0 {
					if from.Kind == KindFloat {
						d = numeric{isFloat: true, f: floatOp(from, sub, cur.f, prev.f)}
					} else {
						d = numeric{unsigned: cur.unsigned, i: cur.i - prev.i, u: cur.u - prev.u}
					}
				}
				// The difference lives in dtype (wrapping), then is cast.
				tmp := make([]byte, from.Size)
				writeNumeric(from, tmp, d)
				writeNumeric(to, out.Data[i*to.Size:], readNumeric(from, tmp))
				prev = cur
			}
			return out, nil
		}
		f.decode = func(a *NDArray, from, to DataType) (*NDArray, error) {
			// np.cumsum(enc, out=dec): accumulate in astype (integers in
			// int64), store as dtype.
			n := elementCount(a)
			out := &NDArray{Shape: a.Shape, DataType: to, Data: make([]byte, n*to.Size)}
			var accF float64
			var accI uint64
			for i := 0; i < n; i++ {
				v := readNumeric(from, a.Data[i*from.Size:])
				if from.Kind == KindFloat {
					if i == 0 {
						accF = v.f
					} else {
						accF = floatOp(from, add, accF, v.f)
					}
					writeNumeric(to, out.Data[i*to.Size:], numeric{isFloat: true, f: accF})
				} else {
					accI += v.bits()
					writeNumeric(to, out.Data[i*to.Size:], numeric{i: int64(accI), u: accI, unsigned: to.Kind == KindUint})
				}
			}
			return out, nil
		}
		return f, nil
	})

	RegisterCodec("numcodecs.fixedscaleoffset", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Offset json.Number `json:"offset"`
			Scale  json.Number `json:"scale"`
			DType  string      `json:"dtype"`
			AsType string      `json:"astype"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		f, err := parse(in, c.DType, c.AsType)
		if err != nil {
			return nil, err
		}
		f.resolveAstype = true
		offInt, offIsInt := pyInt(c.Offset)
		scaleInt, scaleIsInt := pyInt(c.Scale)
		off, _ := strconv.ParseFloat(string(c.Offset), 64)
		scale, _ := strconv.ParseFloat(string(c.Scale), 64)
		if c.Offset == "" {
			off, offInt, offIsInt = 0, 0, true
		}
		if c.Scale == "" {
			scale, scaleInt, scaleIsInt = 1, 1, true
		}
		f.encode = func(a *NDArray, from, to DataType) (*NDArray, error) {
			return mapArray(a, from, to, a.Shape, func(v numeric) numeric {
				switch {
				case from.Kind == KindFloat:
					x := floatOp(from, mul, floatOp(from, sub, v.f, off), scale)
					return numeric{isFloat: true, f: roundTo(from, math.RoundToEven(x))}
				case offIsInt && scaleIsInt:
					// Integer arithmetic in dtype, wrapping.
					tmp := make([]byte, from.Size)
					writeNumeric(from, tmp, numeric{i: (v.i - offInt) * scaleInt, u: (v.u - uint64(offInt)) * uint64(scaleInt), unsigned: v.unsigned})
					return readNumeric(from, tmp)
				case offIsInt:
					tmp := make([]byte, from.Size)
					writeNumeric(from, tmp, numeric{i: v.i - offInt, u: v.u - uint64(offInt), unsigned: v.unsigned})
					return numeric{isFloat: true, f: math.RoundToEven(readNumeric(from, tmp).float() * scale)}
				}
				return numeric{isFloat: true, f: math.RoundToEven((v.float() - off) * scale)}
			}), nil
		}
		f.decode = func(a *NDArray, from, to DataType) (*NDArray, error) {
			return mapArray(a, from, to, a.Shape, func(v numeric) numeric {
				if from.Kind == KindFloat {
					return numeric{isFloat: true, f: floatOp(from, add, floatOp(from, div, v.f, scale), off)}
				}
				return numeric{isFloat: true, f: v.float()/scale + off}
			}), nil
		}
		return f, nil
	})

	RegisterCodec("numcodecs.quantize", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Digits int    `json:"digits"`
			DType  string `json:"dtype"`
			AsType string `json:"astype"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		f, err := parse(in, c.DType, c.AsType)
		if err != nil {
			return nil, err
		}
		if f.dtype.Kind != KindFloat || f.astype.Kind != KindFloat {
			return nil, fmt.Errorf("quantize supports only floating point data")
		}
		f.resolveAstype, f.requireFloat = true, true
		precision := math.Pow(10, -float64(c.Digits))
		exp := math.Log10(precision)
		if exp < 0 {
			exp = math.Floor(exp)
		} else {
			exp = math.Ceil(exp)
		}
		bits := math.Ceil(math.Log2(math.Pow(10, -exp)))
		scale := math.Pow(2, bits)
		f.encode = func(a *NDArray, from, to DataType) (*NDArray, error) {
			return mapArray(a, from, to, a.Shape, func(v numeric) numeric {
				x := roundTo(from, math.RoundToEven(floatOp(from, mul, scale, v.f)))
				return numeric{isFloat: true, f: floatOp(from, div, x, scale)}
			}), nil
		}
		f.decode = func(a *NDArray, from, to DataType) (*NDArray, error) {
			return mapArray(a, from, to, a.Shape, func(v numeric) numeric { return v }), nil
		}
		return f, nil
	})

	RegisterCodec("numcodecs.astype", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Encode string `json:"encode_dtype"`
			Decode string `json:"decode_dtype"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		if c.Encode == "" {
			return nil, fmt.Errorf("astype needs encode_dtype")
		}
		f, err := parse(in, c.Decode, c.Encode)
		if err != nil {
			return nil, err
		}
		f.resolveAstype = true
		conv := func(a *NDArray, from, to DataType) (*NDArray, error) {
			return mapArray(a, from, to, a.Shape, func(v numeric) numeric { return v }), nil
		}
		f.encode, f.decode = conv, conv
		return f, nil
	})

	RegisterCodec("numcodecs.bitround", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Keepbits int `json:"keepbits"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		if c.Keepbits < 0 {
			return nil, fmt.Errorf("keepbits must be zero or positive")
		}
		return bitRound{keepbits: c.Keepbits}, nil
	})

	RegisterCodec("numcodecs.packbits", func(json.RawMessage, ChunkSpec) (any, error) { return packBits{}, nil })

	RegisterCodec("numcodecs.categorize", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Labels []string `json:"labels"`
			DType  string   `json:"dtype"`
			AsType string   `json:"astype"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		if c.AsType == "" {
			c.AsType = "|u1"
		}
		at, big, err := parseNumpyDType(c.AsType)
		if err != nil {
			return nil, err
		}
		if at.Kind != KindInt && at.Kind != KindUint {
			return nil, fmt.Errorf("categorize astype must be an integer type")
		}
		return &categorize{labels: c.Labels, astype: at, big: big}, nil
	})

	RegisterCodec("scale_offset", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Offset json.RawMessage `json:"offset"`
			Scale  json.RawMessage `json:"scale"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		return newScaleOffset(c.Offset, c.Scale, in.DataType)
	})

	RegisterCodec("cast_value", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		return newCastValue(cfg, in.DataType)
	})
}

// pyInt reports whether a JSON number is an integer literal (a Python int).
func pyInt(n json.Number) (int64, bool) {
	if n == "" || strings.ContainsAny(string(n), ".eE") {
		return 0, false
	}
	v, err := n.Int64()
	return v, err == nil
}

// ---------------------------------------------------------------------------
// bitround

// bitRound is numcodecs.bitround: lossy rounding of float mantissas to
// keepbits bits (decoding is the identity).
type bitRound struct{ keepbits int }

func (b bitRound) Resolve(in ChunkSpec) (ChunkSpec, error) {
	if in.DataType.Kind != KindFloat || in.DataType.floatFormat() != nil && in.DataType.Name != "float16" {
		return in, fmt.Errorf("bitround needs float16/32/64 data, not %s", in.DataType.Name)
	}
	return in, nil
}

func (b bitRound) Decode(_ context.Context, a *NDArray, _ ChunkSpec) (*NDArray, error) { return a, nil }

func (b bitRound) Encode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	size := in.DataType.Size
	bits := map[int]int{2: 10, 4: 23, 8: 52}[size]
	if b.keepbits > bits {
		return nil, fmt.Errorf("zarr: bitround keepbits %d too large for %s", b.keepbits, in.DataType.Name)
	}
	if b.keepbits == bits {
		return a, nil
	}
	maskbits := uint(bits - b.keepbits)
	mask := ^uint64(0) << maskbits
	half := uint64(1)<<(maskbits-1) - 1
	out := &NDArray{Shape: a.Shape, DataType: a.DataType, Data: make([]byte, len(a.Data))}
	for i := 0; i+size <= len(a.Data); i += size {
		v := getUint(a.Data[i:i+size], size)
		v += (v>>maskbits)&1 + half
		putUint(out.Data[i:i+size], v&mask)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// packbits

// packBits is numcodecs.packbits: bools packed 8 per byte (most
// significant bit first) after a byte counting the padding bits.
type packBits struct{}

func (packBits) Resolve(in ChunkSpec) (ChunkSpec, error) {
	if in.DataType.Kind != KindBool {
		return in, fmt.Errorf("packbits needs bool data, not %s", in.DataType.Name)
	}
	n := numElements(in.Shape)
	u8 := MustDataType("uint8")
	return ChunkSpec{Shape: []uint64{1 + (n+7)/8}, DataType: u8, FillValue: fillValue{fixed: []byte{0}}.ndarray(u8)}, nil
}

func (packBits) Encode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	n := len(a.Data)
	out := make([]byte, 1+(n+7)/8)
	if n%8 != 0 {
		out[0] = byte(8 - n%8)
	}
	for i, b := range a.Data {
		if b != 0 {
			out[1+i/8] |= 0x80 >> (i % 8)
		}
	}
	return &NDArray{Shape: []uint64{uint64(len(out))}, DataType: MustDataType("uint8"), Data: out}, nil
}

func (packBits) Decode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	if len(a.Data) < 1 {
		return nil, fmt.Errorf("zarr: empty packbits buffer")
	}
	n := 8*(len(a.Data)-1) - int(a.Data[0])
	if want := int(numElements(in.Shape)); n != want || int(a.Data[0]) > 7 {
		return nil, fmt.Errorf("zarr: packbits buffer holds %d bits, expected %d", n, want)
	}
	out := make([]byte, n)
	for i := range out {
		if a.Data[1+i/8]&(0x80>>(i%8)) != 0 {
			out[i] = 1
		}
	}
	return &NDArray{Shape: in.Shape, DataType: in.DataType, Data: out}, nil
}

// ---------------------------------------------------------------------------
// categorize (Zarr v2 only)

type categorize struct {
	labels []string
	astype DataType
	big    bool
}

func (c *categorize) outputBigEndian() (bool, bool) { return c.big, true }

func (c *categorize) Resolve(in ChunkSpec) (ChunkSpec, error) {
	if in.DataType.Kind != KindString && in.DataType.Kind != KindFixedString {
		return in, fmt.Errorf("categorize needs string data, not %s", in.DataType.Name)
	}
	out := in
	out.DataType = c.astype
	out.FillValue = fillValue{fixed: make([]byte, c.astype.Size)}.ndarray(c.astype)
	return out, nil
}

func (c *categorize) Encode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	vals, err := a.Values()
	if err != nil {
		return nil, err
	}
	strs := vals.([]string)
	out := &NDArray{Shape: in.Shape, DataType: c.astype, Data: make([]byte, len(strs)*c.astype.Size)}
	for i, s := range strs {
		for k, l := range c.labels {
			if s == l {
				putUint(out.Data[i*c.astype.Size:(i+1)*c.astype.Size], uint64(k+1))
			}
		}
	}
	return out, nil
}

func (c *categorize) Decode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	n := elementCount(a)
	strs := make([]string, n)
	for i := range strs {
		k := uintAt(a.Data[i*c.astype.Size:], c.astype.Size, 0)
		if k >= 1 && k <= uint64(len(c.labels)) {
			strs[i] = c.labels[k-1]
		}
	}
	return FromValues(in.Shape, in.DataType, strs)
}

// ---------------------------------------------------------------------------
// scale_offset

// scaleOffset encodes (x - offset) * scale and decodes x / scale + offset
// in the array's own data type; integer overflow and inexact integer
// division are errors.
type scaleOffset struct {
	dt            DataType
	offset, scale numeric
}

func newScaleOffset(offRaw, scaleRaw json.RawMessage, dt DataType) (*scaleOffset, error) {
	if dt.Kind != KindInt && dt.Kind != KindUint && dt.Kind != KindFloat {
		return nil, fmt.Errorf("scale_offset supports only integer and floating-point data, not %s", dt.Name)
	}
	parse := func(raw json.RawMessage, dflt string) (numeric, error) {
		if len(raw) == 0 {
			raw = json.RawMessage(dflt)
		}
		// Strings that look like integers are accepted for integer types.
		var s string
		if json.Unmarshal(raw, &s) == nil && dt.Kind != KindFloat {
			if _, err := strconv.ParseInt(s, 10, 64); err == nil {
				raw = json.RawMessage(s)
			}
		}
		fv, err := parseFillValue(raw, dt)
		if err != nil {
			return numeric{}, fmt.Errorf("scale_offset value %s is not representable in %s", raw, dt.Name)
		}
		return readNumeric(dt, fv.fixed), nil
	}
	off, err := parse(offRaw, "0")
	if err != nil {
		return nil, err
	}
	sc, err := parse(scaleRaw, "1")
	if err != nil {
		return nil, err
	}
	if sc.float() == 0 && (sc.isFloat || sc.bits() == 0) {
		return nil, fmt.Errorf("scale_offset scale must be non-zero")
	}
	return &scaleOffset{dt: dt, offset: off, scale: sc}, nil
}

func (s *scaleOffset) Resolve(in ChunkSpec) (ChunkSpec, error) {
	out := in
	if in.FillValue != nil {
		f, err := s.Encode(context.Background(), &NDArray{Shape: []uint64{1}, DataType: in.DataType, Data: in.FillValue.Data}, in.withShape([]uint64{1}))
		if err != nil {
			return in, fmt.Errorf("scale_offset fill value: %w", err)
		}
		f.Shape = []uint64{}
		out.FillValue = f
	}
	return out, nil
}

// intRange returns the range of an integer type as big-enough float64s
// and exact bounds.
func intBounds(dt DataType) (lo int64, hi uint64) {
	bits := dt.Bits
	if bits == 0 {
		bits = 8 * dt.Size
	}
	if dt.Kind == KindUint {
		return 0, math.MaxUint64 >> (64 - bits)
	}
	return -1 << (bits - 1), uint64(1)<<(bits-1) - 1
}

func (s *scaleOffset) Encode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	dt := in.DataType
	n := elementCount(a)
	out := &NDArray{Shape: a.Shape, DataType: dt, Data: make([]byte, len(a.Data))}
	for i := 0; i < n; i++ {
		v := readNumeric(dt, a.Data[i*dt.Size:])
		var r numeric
		if dt.Kind == KindFloat {
			r = numeric{isFloat: true, f: floatOp(dt, mul, floatOp(dt, sub, v.f, s.offset.f), s.scale.f)}
		} else {
			x, ok := checkedIntOp(dt, v, s.offset, s.scale, true)
			if !ok {
				lo, hi := intBounds(dt)
				return nil, fmt.Errorf("zarr: scale_offset produced a value outside the range of %s [%d, %d]", dt.Name, lo, hi)
			}
			r = x
		}
		writeNumeric(dt, out.Data[i*dt.Size:], r)
	}
	return out, nil
}

func (s *scaleOffset) Decode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	dt := in.DataType
	n := elementCount(a)
	out := &NDArray{Shape: a.Shape, DataType: dt, Data: make([]byte, len(a.Data))}
	for i := 0; i < n; i++ {
		v := readNumeric(dt, a.Data[i*dt.Size:])
		var r numeric
		if dt.Kind == KindFloat {
			r = numeric{isFloat: true, f: floatOp(dt, add, floatOp(dt, div, v.f, s.scale.f), s.offset.f)}
		} else {
			x, ok := checkedIntOp(dt, v, s.offset, s.scale, false)
			if !ok {
				return nil, fmt.Errorf("zarr: scale_offset decode of %v is not exact or out of range for %s", v.float(), dt.Name)
			}
			r = x
		}
		writeNumeric(dt, out.Data[i*dt.Size:], r)
	}
	return out, nil
}

// checkedIntOp computes (v - off) * scale (encode) or v / scale + off
// (decode, requiring exact division) exactly, failing if the result is
// outside dt.
func checkedIntOp(dt DataType, v, off, scale numeric, encode bool) (numeric, bool) {
	toBig := func(n numeric) *big.Int {
		if n.unsigned {
			return new(big.Int).SetUint64(n.u)
		}
		return big.NewInt(n.i)
	}
	x, o, k := toBig(v), toBig(off), toBig(scale)
	if encode {
		x.Sub(x, o).Mul(x, k)
	} else {
		var m big.Int
		x.QuoRem(x, k, &m)
		if m.Sign() != 0 {
			return numeric{}, false
		}
		x.Add(x, o)
	}
	lo, hi := intBounds(dt)
	if x.Cmp(big.NewInt(lo)) < 0 || x.Cmp(new(big.Int).SetUint64(hi)) > 0 {
		return numeric{}, false
	}
	if x.Sign() < 0 {
		return numeric{i: x.Int64()}, true
	}
	return numeric{unsigned: true, u: x.Uint64()}, true
}

// ---------------------------------------------------------------------------
// cast_value

type castValue struct {
	from, to   DataType
	rounding   roundingMode
	outOfRange string // "", "clamp", "wrap"
	encodeMap  []scalarPair
	decodeMap  []scalarPair
}

type scalarPair struct{ key, val json.RawMessage }

var castTypes = map[string]bool{
	"int2": true, "int4": true, "int8": true, "int16": true, "int32": true, "int64": true,
	"uint2": true, "uint4": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
	"float4_e2m1fn": true, "float6_e2m3fn": true, "float6_e3m2fn": true, "float8_e3m4": true,
	"float8_e4m3": true, "float8_e4m3b11fnuz": true, "float8_e4m3fnuz": true, "float8_e5m2": true,
	"float8_e5m2fnuz": true, "float8_e8m0fnu": true, "bfloat16": true, "float16": true, "float32": true, "float64": true,
}

func newCastValue(cfg json.RawMessage, from DataType) (*castValue, error) {
	var c struct {
		DataType   json.RawMessage `json:"data_type"`
		Rounding   string          `json:"rounding"`
		OutOfRange *string         `json:"out_of_range"`
		ScalarMap  struct {
			Encode [][2]json.RawMessage `json:"encode"`
			Decode [][2]json.RawMessage `json:"decode"`
		} `json:"scalar_map"`
	}
	if err := parseConfig(cfg, &c); err != nil {
		return nil, err
	}
	to, err := parseDataType(c.DataType)
	if err != nil {
		return nil, err
	}
	if !castTypes[to.Name] {
		return nil, fmt.Errorf("cast_value supports only integer and floating-point types, not %s", to.Name)
	}
	if from.Kind != 0 && !castTypes[from.Name] {
		return nil, fmt.Errorf("cast_value supports only integer and floating-point types, not %s", from.Name)
	}
	cv := &castValue{from: from, to: to}
	switch c.Rounding {
	case "", "nearest-even":
	case "towards-zero":
		cv.rounding = roundTowardsZero
	case "towards-positive":
		cv.rounding = roundTowardsPositive
	case "towards-negative":
		cv.rounding = roundTowardsNegative
	case "nearest-away":
		cv.rounding = roundNearestAway
	default:
		return nil, fmt.Errorf("invalid cast_value rounding %q", c.Rounding)
	}
	if c.OutOfRange != nil {
		cv.outOfRange = *c.OutOfRange
		switch cv.outOfRange {
		case "clamp":
		case "wrap":
			if to.Kind == KindFloat {
				return nil, fmt.Errorf("out_of_range \"wrap\" is only valid for integer target types")
			}
		default:
			return nil, fmt.Errorf("invalid cast_value out_of_range %q", cv.outOfRange)
		}
	}
	for _, p := range c.ScalarMap.Encode {
		cv.encodeMap = append(cv.encodeMap, scalarPair{p[0], p[1]})
	}
	for _, p := range c.ScalarMap.Decode {
		cv.decodeMap = append(cv.decodeMap, scalarPair{p[0], p[1]})
	}
	return cv, nil
}

func (c *castValue) Resolve(in ChunkSpec) (ChunkSpec, error) {
	out := in
	out.DataType = c.to
	if in.FillValue != nil {
		f, err := c.convert(&NDArray{Shape: []uint64{1}, DataType: in.DataType, Data: in.FillValue.Data}, in.DataType, c.to, c.encodeMap)
		if err != nil {
			return in, fmt.Errorf("cast_value fill value: %w", err)
		}
		f.Shape = []uint64{}
		out.FillValue = f
	} else {
		out.FillValue = fillValue{fixed: make([]byte, c.to.Size)}.ndarray(c.to)
	}
	return out, nil
}

func (c *castValue) Encode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	out, err := c.convert(a, in.DataType, c.to, c.encodeMap)
	if err != nil {
		return nil, err
	}
	out.Shape = a.Shape
	return out, nil
}

func (c *castValue) Decode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	out, err := c.convert(a, c.to, in.DataType, c.decodeMap)
	if err != nil {
		return nil, err
	}
	out.Shape = in.Shape
	return out, nil
}

// convert casts every element from one type to another.
func (c *castValue) convert(a *NDArray, from, to DataType, smap []scalarPair) (*NDArray, error) {
	type mapping struct {
		key   numeric
		isNaN bool
		val   []byte
	}
	var maps []mapping
	for _, p := range smap {
		kf, err := parseFillValue(p.key, from)
		if err != nil {
			return nil, fmt.Errorf("cast_value scalar_map key %s: %w", p.key, err)
		}
		vf, err := parseFillValue(p.val, to)
		if err != nil {
			return nil, fmt.Errorf("cast_value scalar_map value %s: %w", p.val, err)
		}
		k := readNumeric(from, kf.fixed)
		maps = append(maps, mapping{key: k, isNaN: k.isFloat && math.IsNaN(k.f), val: vf.fixed})
	}
	n := elementCount(a)
	out := &NDArray{Shape: a.Shape, DataType: to, Data: make([]byte, n*to.Size)}
	for i := 0; i < n; i++ {
		v := readNumeric(from, a.Data[i*from.Size:])
		dst := out.Data[i*to.Size : (i+1)*to.Size]
		mapped := false
		for _, m := range maps {
			if (m.isNaN && v.isFloat && math.IsNaN(v.f)) || (!m.isNaN && numericEqual(v, m.key)) {
				copy(dst, m.val)
				mapped = true
				break
			}
		}
		if mapped {
			continue
		}
		if err := castOne(dst, v, to, c.rounding, c.outOfRange); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func numericEqual(a, b numeric) bool {
	if a.isFloat || b.isFloat {
		return a.float() == b.float()
	}
	if a.unsigned != b.unsigned {
		if a.unsigned {
			return b.i >= 0 && uint64(b.i) == a.u
		}
		return a.i >= 0 && uint64(a.i) == b.u
	}
	return a.bits() == b.bits()
}

// castOne converts one value with the given rounding and out-of-range rule.
func castOne(dst []byte, v numeric, to DataType, mode roundingMode, oor string) error {
	if to.Kind == KindFloat {
		f := v.float()
		var r float64
		switch ff := to.floatFormat(); {
		case ff != nil:
			b, _ := ff.encodeMode(f, mode, false)
			putUint(dst, b)
			r = ff.decode(b)
		case to.Size == 4:
			r32 := roundFloat32(f, mode)
			binary.LittleEndian.PutUint32(dst, math.Float32bits(r32))
			r = float64(r32)
		default:
			binary.LittleEndian.PutUint64(dst, math.Float64bits(f))
			r = f
		}
		// Finite values that overflow are errors unless out_of_range is
		// set, in which case they become Inf as NumPy's astype does.
		if oor == "" && !math.IsInf(f, 0) && !math.IsNaN(f) && (math.IsInf(r, 0) || math.IsNaN(r)) {
			return fmt.Errorf("zarr: cast_value: value %v out of range for %s", f, to.Name)
		}
		return nil
	}
	// Integer target.
	lo, hi := intBounds(to)
	if v.isFloat {
		if math.IsNaN(v.f) {
			return fmt.Errorf("zarr: cast_value: NaN cannot be cast to %s without a scalar_map entry", to.Name)
		}
		r := roundWith(v.f, mode)
		switch {
		// float64(hi) rounds up to 2^63 and 2^64 for int64 and uint64,
		// which are out of range; hi+1 is exact either way.
		case r < float64(lo) || r >= float64(hi)+1:
			switch oor {
			case "clamp":
				if r < float64(lo) {
					putInt(dst, lo)
				} else {
					putUint(dst, hi)
				}
				return nil
			case "wrap":
				if math.IsInf(r, 0) {
					return fmt.Errorf("zarr: cast_value: %v cannot be wrapped into %s", r, to.Name)
				}
				// |m| < 2^64 converts exactly; negative values wrap
				// as two's complement.
				m := math.Mod(r, math.Ldexp(1, intBitsOf(to)))
				u := uint64(math.Abs(m))
				if m < 0 {
					u = -u
				}
				writeNumeric(to, dst, numeric{unsigned: true, u: u})
				return nil
			}
			return fmt.Errorf("zarr: cast_value: value %v out of range [%d, %d] of %s", r, lo, hi, to.Name)
		}
		if r < 0 {
			writeNumeric(to, dst, numeric{i: int64(r)})
		} else {
			writeNumeric(to, dst, numeric{unsigned: true, u: uint64(r)})
		}
		return nil
	}
	// Integer to integer.
	inRange := false
	if v.unsigned {
		inRange = v.u <= hi
	} else {
		inRange = v.i >= lo && (v.i < 0 || uint64(v.i) <= hi)
	}
	if !inRange {
		switch oor {
		case "clamp":
			if !v.unsigned && v.i < lo {
				putInt(dst, lo)
			} else {
				putUint(dst, hi)
			}
			return nil
		case "wrap":
			writeNumeric(to, dst, v)
			return nil
		}
		return fmt.Errorf("zarr: cast_value: value %v out of range [%d, %d] of %s", v.float(), lo, hi, to.Name)
	}
	writeNumeric(to, dst, v)
	return nil
}

func intBitsOf(dt DataType) int {
	if dt.Bits != 0 {
		return dt.Bits
	}
	return 8 * dt.Size
}

// roundFloat32 converts to float32 with the given rounding.
func roundFloat32(f float64, mode roundingMode) float32 {
	r := float32(f)
	if mode == roundNearestEven || math.IsNaN(f) || math.IsInf(f, 0) || float64(r) == f {
		return r
	}
	// Step to the neighbour in the requested direction.
	up := math.Nextafter32(r, float32(math.Inf(1)))
	down := math.Nextafter32(r, float32(math.Inf(-1)))
	lo, hi := r, r
	if float64(r) < f {
		hi = up
	} else {
		lo = down
	}
	switch mode {
	case roundTowardsZero:
		if f > 0 {
			return lo
		}
		return hi
	case roundTowardsPositive:
		return hi
	case roundTowardsNegative:
		return lo
	case roundNearestAway:
		dl, dh := f-float64(lo), float64(hi)-f
		if dl < dh {
			return lo
		}
		if dh < dl || f > 0 {
			return hi
		}
		return lo
	}
	return r
}
