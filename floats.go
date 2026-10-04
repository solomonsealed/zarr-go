package zarr

import "math"

// floatFormat describes a binary floating-point format narrower than
// float32: IEEE half, bfloat16 and the ML formats of the zarr-extensions
// registry (float8_*, float6_*, float4_*), with the same conversion rules
// as ml_dtypes.
type floatFormat struct {
	bits     int // significant bits; the value sits in the low bits of size bytes
	expBits  int
	mantBits int
	bias     int
	signed   bool
	kind     nanKind
}

type nanKind uint8

const (
	// nanIEEE: the all-ones exponent holds ±Inf (zero mantissa) and NaNs.
	nanIEEE nanKind = iota
	// nanFN ("finite"): no Inf; only all-ones exponent and mantissa is NaN.
	nanFN
	// nanFNUZ: no Inf, no negative zero; the negative-zero pattern is NaN.
	nanFNUZ
	// nanNone: neither Inf nor NaN (float6_*, float4_*).
	nanNone
	// nanE8M0: unsigned powers of two; 0xff is NaN (float8_e8m0fnu).
	nanE8M0
)

var floatFormats = map[string]*floatFormat{
	"float16":            {16, 5, 10, 15, true, nanIEEE},
	"bfloat16":           {16, 8, 7, 127, true, nanIEEE},
	"float8_e3m4":        {8, 3, 4, 3, true, nanIEEE},
	"float8_e4m3":        {8, 4, 3, 7, true, nanIEEE},
	"float8_e4m3fn":      {8, 4, 3, 7, true, nanFN},
	"float8_e4m3fnuz":    {8, 4, 3, 8, true, nanFNUZ},
	"float8_e4m3b11fnuz": {8, 4, 3, 11, true, nanFNUZ},
	"float8_e5m2":        {8, 5, 2, 15, true, nanIEEE},
	"float8_e5m2fnuz":    {8, 5, 2, 16, true, nanFNUZ},
	"float8_e8m0fnu":     {8, 8, 0, 127, false, nanE8M0},
	"float6_e2m3fn":      {6, 2, 3, 1, true, nanNone},
	"float6_e3m2fn":      {6, 3, 2, 3, true, nanNone},
	"float4_e2m1fn":      {4, 2, 1, 1, true, nanNone},
}

func (f *floatFormat) size() int { return (f.bits + 7) / 8 }

func (f *floatFormat) signBit() uint64 {
	if !f.signed {
		return 0
	}
	return 1 << (f.bits - 1)
}

func (f *floatFormat) expMax() uint64  { return 1<<f.expBits - 1 }
func (f *floatFormat) mantMax() uint64 { return 1<<f.mantBits - 1 }

// decode converts the format's bits to float64 (exactly).
func (f *floatFormat) decode(b uint64) float64 {
	b &= 1<<f.bits - 1
	if f.kind == nanE8M0 {
		if b == 0xff {
			return math.NaN()
		}
		return math.Ldexp(1, int(b)-127)
	}
	neg := b&f.signBit() != 0
	exp := (b >> f.mantBits) & f.expMax()
	mant := b & f.mantMax()
	var v float64
	switch {
	case f.kind == nanFNUZ && b == f.signBit():
		return math.NaN()
	case f.kind == nanIEEE && exp == f.expMax():
		if mant != 0 {
			return f.nan64(neg, mant)
		}
		v = math.Inf(1)
	case f.kind == nanFN && exp == f.expMax() && mant == f.mantMax():
		return f.nan64(neg, mant)
	case exp == 0:
		v = math.Ldexp(float64(mant), 1-f.bias-f.mantBits)
	default:
		v = math.Ldexp(float64(mant|1<<f.mantBits), int(exp)-f.bias-f.mantBits)
	}
	if neg {
		v = -v
	}
	return v
}

// nan64 widens a NaN to float64 keeping its sign and payload, as NumPy's
// conversions do.
func (f *floatFormat) nan64(neg bool, mant uint64) float64 {
	b := uint64(0x7ff0000000000000) | mant<<(52-f.mantBits)
	if b&0x000fffffffffffff == 0 {
		b |= 0x0008000000000000
	}
	if neg {
		b |= 1 << 63
	}
	return math.Float64frombits(b)
}

// maxFinite returns the bit pattern (without sign) of the largest finite value.
func (f *floatFormat) maxFinite() uint64 {
	switch f.kind {
	case nanIEEE:
		return (f.expMax()-1)<<f.mantBits | f.mantMax()
	case nanFN:
		return f.expMax()<<f.mantBits | (f.mantMax() - 1)
	}
	return f.expMax()<<f.mantBits | f.mantMax()
}

// nan returns the format's NaN for a NaN input with the given sign, as
// ml_dtypes converts it.
func (f *floatFormat) nan(neg bool) uint64 {
	var sign uint64
	if neg {
		sign = f.signBit()
	}
	switch f.kind {
	case nanIEEE:
		return sign | f.expMax()<<f.mantBits | 1<<(f.mantBits-1)
	case nanFN:
		return sign | f.expMax()<<f.mantBits | f.mantMax()
	case nanFNUZ:
		return f.signBit()
	case nanE8M0:
		return 0xff
	}
	// No NaN: ml_dtypes yields a zero whose sign bit is the inverse of the
	// NaN's.
	if neg {
		return 0
	}
	return f.signBit()
}

// Rounding modes (cast_value's names).
type roundingMode uint8

const (
	roundNearestEven roundingMode = iota
	roundTowardsZero
	roundTowardsPositive
	roundTowardsNegative
	roundNearestAway
)

func roundWith(x float64, mode roundingMode) float64 {
	switch mode {
	case roundTowardsZero:
		return math.Trunc(x)
	case roundTowardsPositive:
		return math.Ceil(x)
	case roundTowardsNegative:
		return math.Floor(x)
	case roundNearestAway:
		return math.Round(x)
	}
	return math.RoundToEven(x)
}

// encode converts v to the format with round-to-nearest-even, as NumPy
// (float16) and ml_dtypes (the others) do: overflow becomes Inf where the
// format has one, NaN for the "fn" and "fnuz" formats, and saturates for
// formats without either. ml_dtypes converts float64 through float32, so
// values a hair past a halfway point round as the halfway point does.
func (f *floatFormat) encode(v float64) uint64 {
	if f != halfFormat && !math.IsNaN(v) {
		v = float64(float32(v))
	}
	b, _ := f.encodeMode(v, roundNearestEven, false)
	return b
}

// encodeMode converts v with the given rounding. With clamp, out-of-range
// finite values (and infinities, for formats without Inf) saturate to the
// largest finite value. ok is false if v was out of range without clamp.
func (f *floatFormat) encodeMode(v float64, mode roundingMode, clamp bool) (bits uint64, ok bool) {
	neg := math.Signbit(v)
	if math.IsNaN(v) {
		if f.kind == nanIEEE {
			// Keep the payload's top bits, quieted, as NumPy does.
			mant := math.Float64bits(v) >> (52 - f.mantBits) & f.mantMax()
			mant |= 1 << (f.mantBits - 1)
			var sign uint64
			if neg {
				sign = f.signBit()
			}
			return sign | f.expMax()<<f.mantBits | mant, true
		}
		return f.nan(neg), true
	}
	var sign uint64
	if neg && f.signed {
		sign = f.signBit()
	}
	if f.kind == nanE8M0 {
		return f.encodeE8M0(v, mode, clamp)
	}
	if !f.signed && neg {
		return f.nan(false), false
	}
	a := math.Abs(v)
	// Directed rounding works on magnitudes: towards ±Inf depends on sign.
	m := mode
	if neg {
		switch mode {
		case roundTowardsPositive:
			m = roundTowardsNegative
		case roundTowardsNegative:
			m = roundTowardsPositive
		}
	}
	overflow := func() (uint64, bool) {
		if clamp || f.kind == nanNone {
			return sign | f.maxFinite(), clamp || math.IsInf(v, 0) || f.kind == nanNone
		}
		switch f.kind {
		case nanIEEE:
			return sign | f.expMax()<<f.mantBits, true
		case nanFN:
			return sign | f.expMax()<<f.mantBits | f.mantMax(), true
		}
		return f.signBit(), true // fnuz NaN
	}
	if math.IsInf(a, 1) {
		if f.kind == nanIEEE && !clamp {
			return sign | f.expMax()<<f.mantBits, true
		}
		return overflow()
	}
	if a == 0 {
		if f.kind == nanFNUZ {
			return 0, true
		}
		return sign, true
	}
	minExp := 1 - f.bias // exponent of the smallest normal
	_, e := math.Frexp(a)
	e-- // a in [2^e, 2^(e+1))
	var exp, mant uint64
	if e < minExp {
		q := roundWith(math.Ldexp(a, f.mantBits-minExp), m)
		mant = uint64(q)
		if mant > f.mantMax() { // rounded up to the smallest normal
			exp, mant = 1, 0
		}
	} else {
		q := roundWith(math.Ldexp(a, f.mantBits-e), m)
		if q >= math.Ldexp(1, f.mantBits+1) {
			q /= 2
			e++
		}
		if e+f.bias > int(f.expMax()) {
			return overflow()
		}
		exp, mant = uint64(e+f.bias), uint64(q)-1<<f.mantBits
	}
	r := exp<<f.mantBits | mant
	if r > f.maxFinite() {
		return overflow()
	}
	if r == 0 && f.kind == nanFNUZ {
		return 0, true
	}
	return sign | r, true
}

func (f *floatFormat) encodeE8M0(v float64, mode roundingMode, clamp bool) (uint64, bool) {
	if v <= 0 || math.IsInf(v, 1) {
		if clamp && v > 0 {
			return 0xfe, true
		}
		if clamp && v <= 0 {
			return 0, true
		}
		return 0xff, false
	}
	frac, e := math.Frexp(v) // v = frac*2^e, frac in [0.5,1)
	e--
	m := frac * 2 // in [1,2)
	switch mode {
	case roundTowardsPositive, roundNearestAway:
		if m > 1 && mode == roundTowardsPositive {
			e++
		} else if mode == roundNearestAway && m >= 1.5 {
			e++
		}
	case roundNearestEven:
		// ml_dtypes rounds the halfway point up.
		if m >= 1.5 {
			e++
		}
	}
	if e+127 > 254 {
		if clamp {
			return 0xfe, true
		}
		return 0xff, false
	}
	if e+127 < 0 {
		if clamp {
			return 0, true
		}
		return 0, true
	}
	return uint64(e + 127), true
}

// Half precision shortcuts.
var (
	halfFormat     = floatFormats["float16"]
	bfloat16Format = floatFormats["bfloat16"]
)

func halfToFloat32(h uint16) float32 { return float32(halfFormat.decode(uint64(h))) }

// float64ToHalf converts to IEEE half precision, rounding to nearest even
// like numpy.
func float64ToHalf(f float64) uint16 { return uint16(halfFormat.encode(f)) }

func float32ToHalf(f float32) uint16 { return float64ToHalf(float64(f)) }
