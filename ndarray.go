package zarr

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/bits"
	"reflect"
	"strings"
	"time"
	"unicode/utf8"
)

// NDArray is an N-dimensional array in C (row-major) order.
type NDArray struct {
	Shape    []uint64
	DataType DataType
	// Data holds fixed-size elements, little-endian, DataType.Size bytes
	// each (see DataType for the layout of each kind).
	Data []byte
	// Strings holds the elements of variable-length (string/bytes) types.
	Strings []string
}

func numElements(shape []uint64) uint64 {
	n := uint64(1)
	for _, s := range shape {
		n *= s
	}
	return n
}

// fitsInMemory reports whether an array of the given shape and type has
// a byte size that fits in an int, so that numElements and the sizes
// derived from it cannot overflow.
func fitsInMemory(shape []uint64, dt DataType) bool {
	n := uint64(max(dt.Size, 1))
	if dt.Variable() {
		n = 16 // a string header
	}
	for _, s := range shape {
		hi, lo := bits.Mul64(n, s)
		if hi != 0 || lo > math.MaxInt {
			return false
		}
		n = lo
	}
	return true
}

// fillValue is one element in the in-memory representation.
type fillValue struct {
	fixed []byte // fixed-size types
	str   string // variable-length types
}

func (f fillValue) ndarray(dt DataType) *NDArray {
	n := &NDArray{Shape: []uint64{}, DataType: dt}
	if dt.Variable() {
		n.Strings = []string{f.str}
	} else {
		n.Data = append([]byte(nil), f.fixed...)
	}
	return n
}

func fillOf(n *NDArray) fillValue {
	if n == nil {
		return fillValue{}
	}
	if n.DataType.Variable() {
		if len(n.Strings) == 0 {
			return fillValue{}
		}
		return fillValue{str: n.Strings[0]}
	}
	return fillValue{fixed: n.Data[:n.DataType.Size]}
}

func newFilled(shape []uint64, dt DataType, fill fillValue) *NDArray {
	n := numElements(shape)
	out := &NDArray{Shape: append([]uint64{}, shape...), DataType: dt}
	if dt.Variable() {
		out.Strings = make([]string, n)
		if fill.str != "" {
			for i := range out.Strings {
				out.Strings[i] = fill.str
			}
		}
		return out
	}
	out.Data = make([]byte, n*uint64(dt.Size))
	fillBytes(out.Data, fill.fixed)
	return out
}

// fillBytes repeats elem over buf.
func fillBytes(buf, elem []byte) {
	allZero := true
	for _, b := range elem {
		if b != 0 {
			allZero = false
			break
		}
	}
	if allZero || len(buf) == 0 {
		clear(buf)
		return
	}
	copy(buf, elem)
	// Doubling copy fills the buffer in O(log n) copy calls.
	for filled := len(elem); filled < len(buf); filled *= 2 {
		copy(buf[filled:], buf[:filled])
	}
}

// Len returns the number of elements.
func (n *NDArray) Len() int { return int(numElements(n.Shape)) }

// Clone returns a deep copy.
func (n *NDArray) Clone() *NDArray {
	return &NDArray{
		Shape:    append([]uint64{}, n.Shape...),
		DataType: n.DataType,
		Data:     append([]byte(nil), n.Data...),
		Strings:  append([]string(nil), n.Strings...),
	}
}

// Reshape returns the array viewed with another shape of the same size
// (the data is shared).
func (n *NDArray) Reshape(shape []uint64) (*NDArray, error) {
	if numElements(shape) != numElements(n.Shape) {
		return nil, fmt.Errorf("zarr: cannot reshape %v to %v", n.Shape, shape)
	}
	return &NDArray{Shape: append([]uint64{}, shape...), DataType: n.DataType, Data: n.Data, Strings: n.Strings}, nil
}

// validate checks that the data matches the shape and type.
func (n *NDArray) validate() error {
	if !fitsInMemory(n.Shape, n.DataType) {
		return fmt.Errorf("zarr: shape %v of %s is too large", n.Shape, n.DataType.Name)
	}
	want := numElements(n.Shape)
	if n.DataType.Variable() {
		if uint64(len(n.Strings)) != want {
			return fmt.Errorf("zarr: %d strings for shape %v", len(n.Strings), n.Shape)
		}
		return nil
	}
	if uint64(len(n.Data)) != want*uint64(n.DataType.Size) {
		return fmt.Errorf("zarr: %d bytes for shape %v of %s (%d-byte elements)", len(n.Data), n.Shape, n.DataType.Name, n.DataType.Size)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Reading values

func getUint(b []byte, size int) uint64 {
	var v uint64
	for i := size - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v
}

func putUint(b []byte, v uint64) {
	for i := range b {
		b[i] = byte(v >> (8 * i))
	}
}

func putInt(b []byte, v int64) { putUint(b, uint64(v)) }

// intAt reads a signed integer of the given size and significant bits.
func intAt(b []byte, size, bits int) int64 {
	v := getUint(b, size)
	if bits == 0 {
		bits = 8 * size
	}
	shift := 64 - bits
	return int64(v<<shift) >> shift
}

func uintAt(b []byte, size, bits int) uint64 {
	v := getUint(b, size)
	if bits != 0 {
		v &= 1<<bits - 1
	}
	return v
}

// floatAt reads one float component of type dt (a float type, or the
// component type of a complex type).
func floatAt(dt DataType, b []byte) float64 {
	size := dt.componentSize()
	if f := dt.floatFormat(); f != nil {
		return f.decode(getUint(b, size))
	}
	if size == 4 {
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b)))
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(b))
}

// Values returns the elements as a typed slice:
//
//   - bool: []bool
//   - int8 … int64: []int8 … []int64; int2 and int4: []int8
//   - uint8 … uint64: []uint8 … []uint64; uint2 and uint4: []uint8
//   - float32 and all narrower floats (float16, bfloat16, float8_*, …): []float32
//   - float64: []float64
//   - complex64 and narrower complex types: []complex64; complex128: []complex128
//   - numpy.datetime64 / timedelta64: []int64 (counts of the unit; see Times)
//   - string, fixed_length_utf32: []string
//   - bytes (variable length): []string holding the raw bytes
//   - raw and fixed-length bytes: [][]byte (null_terminated_bytes without
//     trailing NULs)
//   - struct: []map[string]any with each field's value as above
func (n *NDArray) Values() (any, error) {
	if err := n.validate(); err != nil {
		return nil, err
	}
	dt, d, cnt := n.DataType, n.Data, n.Len()
	es := dt.Size
	switch dt.Kind {
	case KindBool:
		out := make([]bool, cnt)
		for i := range out {
			out[i] = d[i] != 0
		}
		return out, nil
	case KindInt, KindDatetime, KindTimedelta:
		switch {
		case es == 1:
			out := make([]int8, cnt)
			for i := range out {
				out[i] = int8(intAt(d[i:], 1, dt.Bits))
			}
			return out, nil
		case es == 2:
			out := make([]int16, cnt)
			for i := range out {
				out[i] = int16(binary.LittleEndian.Uint16(d[2*i:]))
			}
			return out, nil
		case es == 4:
			out := make([]int32, cnt)
			for i := range out {
				out[i] = int32(binary.LittleEndian.Uint32(d[4*i:]))
			}
			return out, nil
		case es == 8:
			out := make([]int64, cnt)
			for i := range out {
				out[i] = int64(binary.LittleEndian.Uint64(d[8*i:]))
			}
			return out, nil
		}
	case KindUint:
		switch es {
		case 1:
			out := make([]uint8, cnt)
			for i := range out {
				out[i] = uint8(uintAt(d[i:], 1, dt.Bits))
			}
			return out, nil
		case 2:
			out := make([]uint16, cnt)
			for i := range out {
				out[i] = binary.LittleEndian.Uint16(d[2*i:])
			}
			return out, nil
		case 4:
			out := make([]uint32, cnt)
			for i := range out {
				out[i] = binary.LittleEndian.Uint32(d[4*i:])
			}
			return out, nil
		case 8:
			out := make([]uint64, cnt)
			for i := range out {
				out[i] = binary.LittleEndian.Uint64(d[8*i:])
			}
			return out, nil
		}
	case KindFloat:
		if dt.Name == "float64" {
			out := make([]float64, cnt)
			for i := range out {
				out[i] = math.Float64frombits(binary.LittleEndian.Uint64(d[8*i:]))
			}
			return out, nil
		}
		out := make([]float32, cnt)
		for i := range out {
			out[i] = float32(floatAt(dt, d[i*es:]))
		}
		return out, nil
	case KindComplex:
		h := es / 2
		if dt.Size == 16 {
			out := make([]complex128, cnt)
			for i := range out {
				out[i] = complex(floatAt(dt, d[i*es:]), floatAt(dt, d[i*es+h:]))
			}
			return out, nil
		}
		out := make([]complex64, cnt)
		for i := range out {
			out[i] = complex(float32(floatAt(dt, d[i*es:])), float32(floatAt(dt, d[i*es+h:])))
		}
		return out, nil
	case KindString, KindBytes:
		return append([]string(nil), n.Strings...), nil
	case KindFixedString:
		out := make([]string, cnt)
		for i := range out {
			out[i] = utf32String(d[i*es : (i+1)*es])
		}
		return out, nil
	case KindRaw, KindFixedBytes:
		out := make([][]byte, cnt)
		for i := range out {
			b := d[i*es : (i+1)*es]
			if dt.Name == "null_terminated_bytes" {
				b = bytes.TrimRight(b, "\x00")
			}
			out[i] = append([]byte(nil), b...)
		}
		return out, nil
	case KindStruct:
		fields := make([]any, len(dt.Fields))
		for j, f := range dt.Fields {
			fa, err := n.Field(f.Name)
			if err != nil {
				return nil, err
			}
			if fields[j], err = fa.Values(); err != nil {
				return nil, err
			}
		}
		out := make([]map[string]any, cnt)
		for i := range out {
			m := make(map[string]any, len(dt.Fields))
			for j, f := range dt.Fields {
				m[f.Name] = reflect.ValueOf(fields[j]).Index(i).Interface()
			}
			out[i] = m
		}
		return out, nil
	}
	return nil, fmt.Errorf("%w: values of %s", ErrUnsupported, dt.Name)
}

func utf32String(b []byte) string {
	var sb strings.Builder
	for j := 0; j+4 <= len(b); j += 4 {
		r := binary.LittleEndian.Uint32(b[j:])
		if r == 0 {
			// numpy strips trailing NULs only; embedded ones are kept.
			rest := b[j:]
			if bytes.Count(rest, []byte{0}) == len(rest) {
				break
			}
		}
		sb.WriteRune(rune(r))
	}
	return sb.String()
}

// Field returns one field of a struct array as an array of its own.
func (n *NDArray) Field(name string) (*NDArray, error) {
	f, ok := n.DataType.Field(name)
	if !ok {
		return nil, fmt.Errorf("zarr: %s has no field %q", n.DataType.Name, name)
	}
	if err := n.validate(); err != nil {
		return nil, err
	}
	cnt, es, fs := n.Len(), n.DataType.Size, f.Type.Size
	out := &NDArray{Shape: append([]uint64{}, n.Shape...), DataType: f.Type, Data: make([]byte, cnt*fs)}
	for i := 0; i < cnt; i++ {
		copy(out.Data[i*fs:(i+1)*fs], n.Data[i*es+f.Offset:])
	}
	return out, nil
}

// Times returns the elements of a datetime64 array as times (UTC);
// NaT becomes the zero time. Calendar units ("Y", "M") are supported.
func (n *NDArray) Times() ([]time.Time, error) {
	if n.DataType.Kind != KindDatetime {
		return nil, fmt.Errorf("zarr: Times on a %s array", n.DataType.Name)
	}
	vals, err := n.Values()
	if err != nil {
		return nil, err
	}
	out := make([]time.Time, n.Len())
	for i, v := range vals.([]int64) {
		if v == math.MinInt64 {
			continue
		}
		t, err := datetimeToTime(v, n.DataType)
		if err != nil {
			return nil, err
		}
		out[i] = t
	}
	return out, nil
}

var unitDurations = map[string]time.Duration{
	"W": 7 * 24 * time.Hour, "D": 24 * time.Hour, "h": time.Hour, "m": time.Minute, "s": time.Second,
	"ms": time.Millisecond, "us": time.Microsecond, "μs": time.Microsecond, "ns": time.Nanosecond,
}

func datetimeToTime(v int64, dt DataType) (time.Time, error) {
	v *= int64(dt.ScaleFactor)
	epoch := time.Unix(0, 0).UTC()
	switch dt.Unit {
	case "Y":
		return epoch.AddDate(int(v), 0, 0), nil
	case "M":
		return epoch.AddDate(0, int(v), 0), nil
	case "ps", "fs", "as":
		div := map[string]int64{"ps": 1e3, "fs": 1e6, "as": 1e9}[dt.Unit]
		return time.Unix(0, v/div).UTC(), nil
	}
	d, ok := unitDurations[dt.Unit]
	if !ok {
		return time.Time{}, fmt.Errorf("%w: datetime unit %q", ErrUnsupported, dt.Unit)
	}
	if d >= time.Second {
		return time.Unix(v*int64(d/time.Second), 0).UTC(), nil
	}
	per := int64(time.Second / d)
	sec, rem := v/per, v%per
	if rem < 0 {
		sec, rem = sec-1, rem+per
	}
	return time.Unix(sec, rem*int64(d)).UTC(), nil
}

func timeToDatetime(t time.Time, dt DataType) (int64, error) {
	if t.IsZero() {
		return math.MinInt64, nil
	}
	t = t.UTC()
	var v int64
	switch dt.Unit {
	case "Y":
		v = int64(t.Year() - 1970)
	case "M":
		v = int64(t.Year()-1970)*12 + int64(t.Month()-1)
	case "ps", "fs", "as":
		mul := map[string]int64{"ps": 1e3, "fs": 1e6, "as": 1e9}[dt.Unit]
		v = t.UnixNano() * mul
	default:
		d, ok := unitDurations[dt.Unit]
		if !ok {
			return 0, fmt.Errorf("%w: datetime unit %q", ErrUnsupported, dt.Unit)
		}
		if d >= time.Second {
			v = t.Unix() / int64(d/time.Second)
			if t.Unix()%int64(d/time.Second) < 0 {
				v--
			}
		} else {
			per := int64(time.Second / d)
			v = t.Unix()*per + int64(t.Nanosecond())/int64(d)
		}
	}
	return v / int64(dt.ScaleFactor), nil
}

// Float64s returns the elements of a real numeric or bool array converted to
// float64.
func (n *NDArray) Float64s() ([]float64, error) {
	if err := n.validate(); err != nil {
		return nil, err
	}
	out := make([]float64, n.Len())
	for i := range out {
		v, err := n.float64At(i)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

func (n *NDArray) float64At(i int) (float64, error) {
	dt := n.DataType
	b := n.Data[i*dt.Size:]
	switch dt.Kind {
	case KindBool:
		if b[0] != 0 {
			return 1, nil
		}
		return 0, nil
	case KindInt, KindDatetime, KindTimedelta:
		return float64(intAt(b, dt.Size, dt.Bits)), nil
	case KindUint:
		return float64(uintAt(b, dt.Size, dt.Bits)), nil
	case KindFloat:
		return floatAt(dt, b), nil
	}
	return 0, fmt.Errorf("%w: cannot convert %s to float64", ErrUnsupported, dt.Name)
}

// Number is the set of Go numeric types Convert can produce.
type Number interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~int | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uint | ~float32 | ~float64
}

// Convert returns the elements of a real numeric or bool array converted to
// T with Go conversion semantics. Integer sources convert exactly (no
// round-trip through float64); float to integer conversion of NaN or
// out-of-range values is implementation-defined, as in Go.
func Convert[T Number](n *NDArray) ([]T, error) {
	if err := n.validate(); err != nil {
		return nil, err
	}
	dt := n.DataType
	out := make([]T, n.Len())
	switch dt.Kind {
	case KindInt, KindDatetime, KindTimedelta:
		for i := range out {
			out[i] = T(intAt(n.Data[i*dt.Size:], dt.Size, dt.Bits))
		}
	case KindUint:
		for i := range out {
			out[i] = T(uintAt(n.Data[i*dt.Size:], dt.Size, dt.Bits))
		}
	default:
		for i := range out {
			v, err := n.float64At(i)
			if err != nil {
				return nil, err
			}
			out[i] = T(v)
		}
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// Building arrays

// Scalar is the set of Go element types FromSlice accepts.
type Scalar interface {
	~bool | ~int8 | ~int16 | ~int32 | ~int64 | ~int | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uint |
		~float32 | ~float64 | ~complex64 | ~complex128
}

// FromSlice wraps Go values (C order) as an NDArray of the matching Zarr
// data type: bool, int8 … uint64, float32, float64, complex64, complex128
// (int and uint map to int64 and uint64). Named types such as
// "type Celsius float32" are accepted. Use FromValues for other data types.
func FromSlice[T Scalar](shape []uint64, values []T) (*NDArray, error) {
	var name string
	switch reflect.TypeFor[T]().Kind() {
	case reflect.Bool:
		name = "bool"
	case reflect.Int8:
		name = "int8"
	case reflect.Int16:
		name = "int16"
	case reflect.Int32:
		name = "int32"
	case reflect.Int64, reflect.Int:
		name = "int64"
	case reflect.Uint8:
		name = "uint8"
	case reflect.Uint16:
		name = "uint16"
	case reflect.Uint32:
		name = "uint32"
	case reflect.Uint64, reflect.Uint:
		name = "uint64"
	case reflect.Float32:
		name = "float32"
	case reflect.Float64:
		name = "float64"
	case reflect.Complex64:
		name = "complex64"
	case reflect.Complex128:
		name = "complex128"
	}
	return FromValues(shape, MustDataType(name), values)
}

// FromStrings wraps strings (C order) as an NDArray of the "string" type.
func FromStrings(shape []uint64, values []string) (*NDArray, error) {
	return FromValues(shape, DataType{Name: "string", Kind: KindString}, values)
}

// FromValues builds an NDArray of type dt from Go values in C order. It
// accepts, per data type:
//
//   - numeric and bool types: a slice of any Go numeric or bool type
//     (including named types); values must be representable (integers in
//     range and integral, or rounded to nearest-even for narrow floats)
//   - complex types: []complex64 or []complex128 (or real slices)
//   - numpy.datetime64: []int64 counts or []time.Time (zero time is NaT);
//     numpy.timedelta64: []int64 counts or []time.Duration
//   - string, fixed_length_utf32: []string; null_terminated_bytes,
//     raw_bytes, rN, bytes: [][]byte or []string
//   - struct: []map[string]any (missing fields are zero) or [][]byte of
//     packed elements
func FromValues(shape []uint64, dt DataType, values any) (*NDArray, error) {
	rv := reflect.ValueOf(values)
	if rv.Kind() != reflect.Slice {
		return nil, fmt.Errorf("zarr: FromValues needs a slice, got %T", values)
	}
	cnt := rv.Len()
	if uint64(cnt) != numElements(shape) {
		return nil, fmt.Errorf("zarr: %d values for shape %v", cnt, shape)
	}
	out := &NDArray{Shape: append([]uint64{}, shape...), DataType: dt}
	if dt.Variable() {
		out.Strings = make([]string, cnt)
		for i := range cnt {
			s, err := stringValue(rv.Index(i))
			if err != nil {
				return nil, err
			}
			if dt.Kind == KindString && !utf8.ValidString(s) {
				return nil, fmt.Errorf("zarr: string %q is not valid UTF-8", s)
			}
			out.Strings[i] = s
		}
		return out, nil
	}
	out.Data = make([]byte, cnt*dt.Size)
	for i := range cnt {
		if err := setElement(out.Data[i*dt.Size:(i+1)*dt.Size], dt, rv.Index(i)); err != nil {
			return nil, fmt.Errorf("zarr: element %d: %w", i, err)
		}
	}
	return out, nil
}

func stringValue(v reflect.Value) (string, error) {
	switch {
	case v.Kind() == reflect.String:
		return v.String(), nil
	case v.Kind() == reflect.Slice && v.Type().Elem().Kind() == reflect.Uint8:
		return string(v.Bytes()), nil
	case v.Kind() == reflect.Interface && !v.IsNil():
		return stringValue(v.Elem())
	}
	return "", fmt.Errorf("zarr: %s is not a string or []byte", v.Type())
}

// setElement encodes one Go value as an element of dt into b.
func setElement(b []byte, dt DataType, v reflect.Value) error {
	if v.Kind() == reflect.Interface {
		if v.IsNil() {
			clear(b)
			return nil
		}
		v = v.Elem()
	}
	switch dt.Kind {
	case KindBool:
		if v.Kind() == reflect.Bool {
			if v.Bool() {
				b[0] = 1
			} else {
				b[0] = 0
			}
			return nil
		}
		f, ok := numberOf(v)
		if !ok {
			return fmt.Errorf("cannot store %s as bool", v.Type())
		}
		b[0] = 0
		if f != 0 {
			b[0] = 1
		}
		return nil
	case KindInt, KindUint:
		return setInteger(b, dt, v)
	case KindDatetime, KindTimedelta:
		switch x := v.Interface().(type) {
		case time.Time:
			n, err := timeToDatetime(x, dt)
			if err != nil {
				return err
			}
			putInt(b, n)
			return nil
		case time.Duration:
			n, err := durationToCount(x, dt)
			if err != nil {
				return err
			}
			putInt(b, n)
			return nil
		}
		return setInteger(b, MustDataType("int64"), v)
	case KindFloat:
		f, ok := numberOf(v)
		if !ok {
			return fmt.Errorf("cannot store %s as %s", v.Type(), dt.Name)
		}
		putFloat(b, dt, f)
		return nil
	case KindComplex:
		var c complex128
		switch v.Kind() {
		case reflect.Complex64, reflect.Complex128:
			c = v.Complex()
		default:
			f, ok := numberOf(v)
			if !ok {
				return fmt.Errorf("cannot store %s as %s", v.Type(), dt.Name)
			}
			c = complex(f, 0)
		}
		h := dt.Size / 2
		putFloat(b[:h], dt, real(c))
		putFloat(b[h:], dt, imag(c))
		return nil
	case KindFixedString:
		s, err := stringValue(v)
		if err != nil {
			return err
		}
		clear(b)
		i := 0
		for _, r := range s {
			if i+4 > len(b) {
				return fmt.Errorf("string %q longer than %d characters", s, len(b)/4)
			}
			binary.LittleEndian.PutUint32(b[i:], uint32(r))
			i += 4
		}
		return nil
	case KindRaw, KindFixedBytes:
		s, err := stringValue(v)
		if err != nil {
			return err
		}
		if len(s) > len(b) {
			return fmt.Errorf("%d bytes do not fit %s of %d bytes", len(s), dt.Name, len(b))
		}
		clear(b)
		copy(b, s)
		return nil
	case KindStruct:
		if s, err := stringValue(v); err == nil {
			if len(s) != len(b) {
				return fmt.Errorf("packed struct element has %d bytes, want %d", len(s), len(b))
			}
			copy(b, s)
			return nil
		}
		if v.Kind() != reflect.Map || v.Type().Key().Kind() != reflect.String {
			return fmt.Errorf("cannot store %s as a struct", v.Type())
		}
		clear(b)
		for _, k := range v.MapKeys() {
			f, ok := dt.Field(k.String())
			if !ok {
				return fmt.Errorf("struct has no field %q", k.String())
			}
			if err := setElement(b[f.Offset:f.Offset+f.Type.Size], f.Type, v.MapIndex(k)); err != nil {
				return fmt.Errorf("field %q: %w", f.Name, err)
			}
		}
		return nil
	}
	return fmt.Errorf("%w: values of %s", ErrUnsupported, dt.Name)
}

func durationToCount(d time.Duration, dt DataType) (int64, error) {
	unit, ok := unitDurations[dt.Unit]
	if !ok {
		return 0, fmt.Errorf("%w: timedelta unit %q from a Duration", ErrUnsupported, dt.Unit)
	}
	step := unit * time.Duration(dt.ScaleFactor)
	if d%step != 0 {
		return 0, fmt.Errorf("duration %v is not a multiple of %d%s", d, dt.ScaleFactor, dt.Unit)
	}
	return int64(d / step), nil
}

func numberOf(v reflect.Value) (float64, bool) {
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return float64(v.Uint()), true
	case reflect.Float32, reflect.Float64:
		return v.Float(), true
	case reflect.Bool:
		if v.Bool() {
			return 1, true
		}
		return 0, true
	}
	return 0, false
}

// setInteger stores an integer value, checking its range.
func setInteger(b []byte, dt DataType, v reflect.Value) error {
	bits := dt.Bits
	if bits == 0 {
		bits = 8 * dt.Size
	}
	var lo, hi int64
	var uhi uint64
	if dt.Kind == KindUint || dt.Kind == KindBool {
		uhi = math.MaxUint64 >> (64 - bits)
	} else {
		lo, hi = -1<<(bits-1), 1<<(bits-1)-1
	}
	store := func(u uint64) {
		if dt.Bits != 0 {
			u &= 1<<dt.Bits - 1
		}
		putUint(b, u)
	}
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		x := v.Int()
		if dt.Kind == KindUint {
			if x < 0 || uint64(x) > uhi {
				return fmt.Errorf("%d out of range for %s", x, dt.Name)
			}
		} else if x < lo || x > hi {
			return fmt.Errorf("%d out of range for %s", x, dt.Name)
		}
		store(uint64(x))
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		x := v.Uint()
		if (dt.Kind == KindUint && x > uhi) || (dt.Kind != KindUint && x > uint64(hi)) {
			return fmt.Errorf("%d out of range for %s", x, dt.Name)
		}
		store(x)
		return nil
	case reflect.Bool:
		store(map[bool]uint64{false: 0, true: 1}[v.Bool()])
		return nil
	case reflect.Float32, reflect.Float64:
		f := v.Float()
		if f != math.Trunc(f) || math.IsNaN(f) {
			return fmt.Errorf("%v is not an integer", f)
		}
		// The upper bounds as float64 round up to 2^64 and 2^63, which
		// are out of range; adding 1 keeps them exact.
		if dt.Kind == KindUint {
			if f < 0 || f >= float64(uhi)+1 {
				return fmt.Errorf("%v out of range for %s", f, dt.Name)
			}
			store(uint64(f))
		} else {
			if f < float64(lo) || f >= float64(hi)+1 {
				return fmt.Errorf("%v out of range for %s", f, dt.Name)
			}
			store(uint64(int64(f)))
		}
		return nil
	}
	return fmt.Errorf("cannot store %s as %s", v.Type(), dt.Name)
}

// putFloat stores f as one float component of dt.
func putFloat(b []byte, dt DataType, f float64) {
	size := dt.componentSize()
	if ff := dt.floatFormat(); ff != nil {
		putUint(b[:size], ff.encode(f))
		return
	}
	if size == 4 {
		binary.LittleEndian.PutUint32(b, math.Float32bits(float32(f)))
		return
	}
	binary.LittleEndian.PutUint64(b, math.Float64bits(f))
}

// ---------------------------------------------------------------------------
// JSON

// MarshalJSON encodes {"shape": [...], "dtype": "...", "data": [...]}
// with data flattened in C order. NaN and infinite floats, which JSON
// cannot represent, are encoded as null; complex numbers as [re, im];
// raw/fixed bytes as base64 strings; datetimes as integer counts.
func (n *NDArray) MarshalJSON() ([]byte, error) {
	vals, err := n.Values()
	if err != nil {
		return nil, err
	}
	data, err := jsonValues(vals)
	if err != nil {
		return nil, err
	}
	shape := n.Shape
	if shape == nil {
		shape = []uint64{}
	}
	return json.Marshal(struct {
		Shape []uint64 `json:"shape"`
		DType string   `json:"dtype"`
		Data  any      `json:"data"`
	}{shape, n.DataType.Name, data})
}

func jsonValues(vals any) (any, error) {
	switch v := vals.(type) {
	case []float32:
		return floatsJSON(len(v), func(i int) float64 { return float64(v[i]) }), nil
	case []float64:
		return floatsJSON(len(v), func(i int) float64 { return v[i] }), nil
	case []complex64:
		out := make([][2]any, len(v))
		for i, c := range v {
			out[i] = [2]any{floatJSON(float64(real(c))), floatJSON(float64(imag(c)))}
		}
		return out, nil
	case []complex128:
		out := make([][2]any, len(v))
		for i, c := range v {
			out[i] = [2]any{floatJSON(real(c)), floatJSON(imag(c))}
		}
		return out, nil
	case []uint8:
		// Avoid encoding/json's base64 for []byte.
		out := make([]uint16, len(v))
		for i, b := range v {
			out[i] = uint16(b)
		}
		return out, nil
	case [][]byte:
		out := make([]string, len(v))
		for i, b := range v {
			out[i] = base64.StdEncoding.EncodeToString(b)
		}
		return out, nil
	case []map[string]any:
		out := make([]map[string]any, len(v))
		for i, m := range v {
			o := make(map[string]any, len(m))
			for k, x := range m {
				switch y := x.(type) {
				case float32:
					o[k] = floatJSON(float64(y))
				case float64:
					o[k] = floatJSON(y)
				case complex64:
					o[k] = [2]any{floatJSON(float64(real(y))), floatJSON(float64(imag(y)))}
				case complex128:
					o[k] = [2]any{floatJSON(real(y)), floatJSON(imag(y))}
				default:
					o[k] = x
				}
			}
			out[i] = o
		}
		return out, nil
	}
	return vals, nil
}

func floatJSON(f float64) any {
	if math.IsNaN(f) || math.IsInf(f, 0) {
		return nil
	}
	return f
}

func floatsJSON(n int, at func(int) float64) []any {
	out := make([]any, n)
	for i := range out {
		out[i] = floatJSON(at(i))
	}
	return out
}
