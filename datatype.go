package zarr

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ErrUnsupported is returned for metadata features this package cannot decode.
var ErrUnsupported = errors.New("zarr: unsupported")

// Kind classifies data types.
type Kind uint8

const (
	KindBool Kind = iota + 1
	KindInt
	KindUint
	// KindFloat covers float16/32/64, bfloat16 and the float8/6/4 formats.
	KindFloat
	// KindComplex covers complex64/128 and the complex_<float> types.
	KindComplex
	// KindRaw is an opaque fixed-size type ("r8", "r16", ...).
	KindRaw
	// KindString is variable-length UTF-8 ("string").
	KindString
	// KindBytes is variable-length binary ("bytes", "variable_length_bytes").
	KindBytes
	// KindDatetime / KindTimedelta are numpy datetime64 / timedelta64,
	// stored as int64 counts of ScaleFactor × Unit.
	KindDatetime
	KindTimedelta
	// KindFixedString is numpy's fixed-length UTF-32 ("fixed_length_utf32").
	KindFixedString
	// KindFixedBytes is numpy's fixed-length bytes ("null_terminated_bytes",
	// "raw_bytes").
	KindFixedBytes
	// KindStruct is a record of named fields ("struct", "structured").
	KindStruct
)

// DataType describes an array's element type.
//
// In memory (NDArray.Data) elements are stored packed and little-endian:
// numbers in their native width, sub-byte types (int4, float4_e2m1fn, ...)
// in the low bits of one byte, complex numbers as real then imaginary part,
// fixed-length strings as UTF-32 code units and structs as their fields in
// order. Variable-length types use NDArray.Strings instead.
type DataType struct {
	// Name is the Zarr v3 data type name, e.g. "float32" or "numpy.datetime64".
	Name string
	Kind Kind
	// Size is the element size in bytes; 0 for variable-length types.
	Size int
	// Bits is the number of significant bits for sub-byte types (int2,
	// int4, uint2, uint4, float6_*, float4_*); 0 otherwise.
	Bits int
	// Unit and ScaleFactor give the datetime/timedelta resolution
	// (e.g. "ns" and 1, or "s" and 10 for numpy's "M8[10s]").
	Unit        string
	ScaleFactor int
	// Fields lists the members of a struct type.
	Fields []StructField
	// Configuration is the raw data type configuration, if any.
	Configuration map[string]any
}

// StructField is one member of a struct data type.
type StructField struct {
	Name string
	Type DataType
	// Offset is the field's byte offset within an element.
	Offset int
}

// Variable reports whether elements have variable length.
func (d DataType) Variable() bool { return d.Kind == KindString || d.Kind == KindBytes }

func (d DataType) String() string { return d.Name }

// Equal reports whether two data types are the same.
func (d DataType) Equal(o DataType) bool {
	if d.Name != o.Name || d.Size != o.Size || d.Unit != o.Unit || d.ScaleFactor != o.ScaleFactor || len(d.Fields) != len(o.Fields) {
		return false
	}
	for i := range d.Fields {
		if d.Fields[i].Name != o.Fields[i].Name || !d.Fields[i].Type.Equal(o.Fields[i].Type) {
			return false
		}
	}
	return true
}

// floatFormat returns the conversion rules of a narrow float type (or of
// the components of a narrow complex type); nil for float32/float64.
func (d DataType) floatFormat() *floatFormat {
	switch d.Kind {
	case KindFloat:
		return floatFormats[d.Name]
	case KindComplex:
		return floatFormats[strings.TrimPrefix(d.Name, "complex_")]
	}
	return nil
}

// componentSize is the size of one float of a float or complex type.
func (d DataType) componentSize() int {
	if d.Kind == KindComplex {
		return d.Size / 2
	}
	return d.Size
}

// swapUnits returns the byte ranges (offset, size) within an element that
// are reversed to change byte order; nil if byte order does not apply.
func (d DataType) swapUnits() [][2]int {
	switch d.Kind {
	case KindInt, KindUint, KindFloat, KindDatetime, KindTimedelta:
		if d.Size > 1 {
			return [][2]int{{0, d.Size}}
		}
	case KindComplex:
		if h := d.Size / 2; h > 1 {
			return [][2]int{{0, h}, {h, h}}
		}
	case KindFixedString:
		units := make([][2]int, d.Size/4)
		for i := range units {
			units[i] = [2]int{4 * i, 4}
		}
		return units
	case KindStruct:
		var units [][2]int
		for _, f := range d.Fields {
			for _, u := range f.Type.swapUnits() {
				units = append(units, [2]int{f.Offset + u[0], u[1]})
			}
		}
		return units
	}
	return nil
}

// byteSwap returns data (whole elements of d) with the byte order of every
// multi-byte number reversed; data itself is not modified.
func byteSwap(d DataType, data []byte) []byte {
	units := d.swapUnits()
	if len(units) == 0 || d.Size == 0 {
		return data
	}
	out := make([]byte, len(data))
	copy(out, data)
	for e := 0; e+d.Size <= len(out); e += d.Size {
		for _, u := range units {
			b := out[e+u[0] : e+u[0]+u[1]]
			for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
				b[i], b[j] = b[j], b[i]
			}
		}
	}
	return out
}

// ParseDataType parses a Zarr v3 data type name such as "float32".
func ParseDataType(name string) (DataType, error) {
	raw, _ := json.Marshal(name)
	return parseDataType(raw)
}

// ParseDataTypeJSON parses a Zarr v3 data_type value: a name or an object
// with "name" and "configuration".
func ParseDataTypeJSON(raw json.RawMessage) (DataType, error) { return parseDataType(raw) }

var subByte = map[string]int{"int2": 2, "int4": 4, "uint2": 2, "uint4": 4}

func parseDataType(raw json.RawMessage) (DataType, error) {
	var name string
	var cfg map[string]any
	if err := json.Unmarshal(raw, &name); err != nil {
		var obj struct {
			Name          string         `json:"name"`
			Configuration map[string]any `json:"configuration"`
		}
		if err := json.Unmarshal(raw, &obj); err != nil {
			return DataType{}, fmt.Errorf("zarr: invalid data_type %s", raw)
		}
		name, cfg = obj.Name, obj.Configuration
	}
	d := DataType{Name: name, Configuration: cfg}
	switch name {
	case "bool":
		d.Kind, d.Size = KindBool, 1
	case "int8", "int16", "int32", "int64":
		d.Kind = KindInt
		d.Size, _ = strconv.Atoi(name[3:])
		d.Size /= 8
	case "uint8", "uint16", "uint32", "uint64":
		d.Kind = KindUint
		d.Size, _ = strconv.Atoi(name[4:])
		d.Size /= 8
	case "int2", "int4":
		d.Kind, d.Size, d.Bits = KindInt, 1, subByte[name]
	case "uint2", "uint4":
		d.Kind, d.Size, d.Bits = KindUint, 1, subByte[name]
	case "float32", "float64":
		d.Kind = KindFloat
		d.Size, _ = strconv.Atoi(name[5:])
		d.Size /= 8
	case "complex64", "complex128":
		d.Kind = KindComplex
		d.Size, _ = strconv.Atoi(name[7:])
		d.Size /= 8
	case "complex_float32":
		d.Kind, d.Size = KindComplex, 8
	case "complex_float64":
		d.Kind, d.Size = KindComplex, 16
	case "string":
		d.Kind = KindString
	case "bytes", "variable_length_bytes":
		d.Kind = KindBytes
	case "numpy.datetime64", "numpy.timedelta64":
		d.Kind, d.Size = KindDatetime, 8
		if name == "numpy.timedelta64" {
			d.Kind = KindTimedelta
		}
		unit, _ := cfg["unit"].(string)
		d.Unit = timeUnit(unit)
		if d.Unit == "" {
			return d, fmt.Errorf("zarr: data type %s needs configuration.unit", name)
		}
		d.ScaleFactor = 1
		if f, ok := cfg["scale_factor"].(float64); ok {
			d.ScaleFactor = 0 // invalid, unless an integer NumPy accepts
			if f == math.Trunc(f) && f >= 1 && f <= math.MaxInt32 {
				d.ScaleFactor = int(f)
			}
		}
		if !validTimeUnit(d.Unit) || d.ScaleFactor < 1 {
			return d, fmt.Errorf("zarr: invalid %s unit %q / scale_factor %d", name, d.Unit, d.ScaleFactor)
		}
	case "fixed_length_utf32", "null_terminated_bytes", "raw_bytes":
		d.Kind = KindFixedBytes
		if name == "fixed_length_utf32" {
			d.Kind = KindFixedString
		}
		f, ok := cfg["length_bytes"].(float64)
		// NumPy's limit on element sizes is 2^31-1 bytes.
		if !ok || f < 0 || f > math.MaxInt32 || f != float64(int(f)) || (name == "fixed_length_utf32" && int(f)%4 != 0) {
			return d, fmt.Errorf("zarr: data type %s needs a valid configuration.length_bytes", name)
		}
		d.Size = int(f)
	case "struct", "structured":
		d.Kind = KindStruct
		if err := d.parseStructFields(raw); err != nil {
			return d, err
		}
	default:
		if f, ok := floatFormats[name]; ok {
			d.Kind, d.Size = KindFloat, f.size()
			if f.bits < 8 {
				d.Bits = f.bits
			}
			return d, nil
		}
		if f, ok := floatFormats[strings.TrimPrefix(name, "complex_")]; ok && strings.HasPrefix(name, "complex_") {
			if f.bits < 8 {
				d.Bits = f.bits
			}
			d.Kind, d.Size = KindComplex, 2*f.size()
			return d, nil
		}
		if strings.HasPrefix(name, "r") {
			if bits, err := strconv.Atoi(name[1:]); err == nil && bits > 0 && bits%8 == 0 && bits/8 <= math.MaxInt32 && name[1] != '0' {
				d.Kind, d.Size = KindRaw, bits/8
				return d, nil
			}
		}
		return d, fmt.Errorf("%w: data type %q", ErrUnsupported, name)
	}
	return d, nil
}

func (d *DataType) parseStructFields(raw json.RawMessage) error {
	var obj struct {
		Configuration struct {
			Fields []json.RawMessage `json:"fields"`
		} `json:"configuration"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil || len(obj.Configuration.Fields) == 0 {
		return fmt.Errorf("zarr: %s data type needs configuration.fields", d.Name)
	}
	seen := map[string]bool{}
	for _, f := range obj.Configuration.Fields {
		var name string
		var dtRaw json.RawMessage
		var canonical struct {
			Name     *string         `json:"name"`
			DataType json.RawMessage `json:"data_type"`
		}
		var legacy []json.RawMessage
		switch {
		case json.Unmarshal(f, &canonical) == nil && canonical.Name != nil:
			name, dtRaw = *canonical.Name, canonical.DataType
		case json.Unmarshal(f, &legacy) == nil && len(legacy) == 2 && json.Unmarshal(legacy[0], &name) == nil:
			dtRaw = legacy[1]
		default:
			return fmt.Errorf("zarr: invalid struct field %s", f)
		}
		ft, err := parseDataType(dtRaw)
		if err != nil {
			return fmt.Errorf("zarr: struct field %q: %w", name, err)
		}
		if ft.Variable() {
			return fmt.Errorf("%w: variable-length struct field %q", ErrUnsupported, name)
		}
		if seen[name] {
			return fmt.Errorf("zarr: duplicate struct field %q", name)
		}
		seen[name] = true
		if ft.Size > math.MaxInt32-d.Size { // NumPy's limit
			return fmt.Errorf("%w: struct of more than %d bytes", ErrUnsupported, math.MaxInt32)
		}
		d.Fields = append(d.Fields, StructField{Name: name, Type: ft, Offset: d.Size})
		d.Size += ft.Size
	}
	return nil
}

// timeUnit spells the microsecond unit as NumPy does ("us"; "μs" is
// accepted as input), as zarr-python 3.4.1 does.
func timeUnit(u string) string {
	if u == "μs" {
		return "us"
	}
	return u
}

func validTimeUnit(u string) bool {
	switch u {
	case "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as", "generic":
		return true
	}
	return false
}

// Struct returns a struct data type with the given fields (offsets are
// computed, packed as numpy does without alignment).
func Struct(fields ...StructField) DataType {
	d := DataType{Name: "struct", Kind: KindStruct}
	for _, f := range fields {
		f.Offset = d.Size
		d.Fields = append(d.Fields, f)
		d.Size += f.Type.Size
	}
	return d
}

// Field returns the struct field with the given name.
func (d DataType) Field(name string) (StructField, bool) {
	for _, f := range d.Fields {
		if f.Name == name {
			return f, true
		}
	}
	return StructField{}, false
}

// Datetime64 returns numpy.datetime64 with the given unit ("s", "ns", "D", ...).
func Datetime64(unit string, scaleFactor int) DataType {
	return DataType{Name: "numpy.datetime64", Kind: KindDatetime, Size: 8, Unit: timeUnit(unit), ScaleFactor: max(scaleFactor, 1)}
}

// Timedelta64 returns numpy.timedelta64 with the given unit.
func Timedelta64(unit string, scaleFactor int) DataType {
	return DataType{Name: "numpy.timedelta64", Kind: KindTimedelta, Size: 8, Unit: timeUnit(unit), ScaleFactor: max(scaleFactor, 1)}
}

// FixedLengthUTF32 returns numpy's fixed-length unicode type ("U<n>") of n characters.
func FixedLengthUTF32(chars int) DataType {
	return DataType{Name: "fixed_length_utf32", Kind: KindFixedString, Size: 4 * chars}
}

// NullTerminatedBytes returns numpy's fixed-length bytes type ("S<n>").
func NullTerminatedBytes(n int) DataType {
	return DataType{Name: "null_terminated_bytes", Kind: KindFixedBytes, Size: n}
}

// RawBytes returns numpy's opaque void type ("V<n>").
func RawBytes(n int) DataType {
	return DataType{Name: "raw_bytes", Kind: KindFixedBytes, Size: n}
}

// MustDataType is ParseDataType that panics on error, for package-level
// variables and tests.
func MustDataType(name string) DataType {
	d, err := ParseDataType(name)
	if err != nil {
		panic(err)
	}
	return d
}

// v3JSON renders the data type as zarr-python writes it in zarr.json.
func (d DataType) v3JSON() any {
	switch d.Kind {
	case KindDatetime, KindTimedelta:
		return jsonObject{{"name", d.Name}, {"configuration", jsonObject{{"unit", timeUnit(d.Unit)}, {"scale_factor", d.ScaleFactor}}}}
	case KindFixedString, KindFixedBytes:
		return jsonObject{{"name", d.Name}, {"configuration", jsonObject{{"length_bytes", d.Size}}}}
	case KindStruct:
		fields := make([]any, len(d.Fields))
		for i, f := range d.Fields {
			if d.Name == "structured" {
				fields[i] = []any{f.Name, f.Type.v3JSON()}
			} else {
				fields[i] = jsonObject{{"name", f.Name}, {"data_type", f.Type.v3JSON()}}
			}
		}
		return jsonObject{{"name", d.Name}, {"configuration", jsonObject{{"fields", fields}}}}
	case KindBytes:
		if d.Name == "bytes" {
			return "variable_length_bytes"
		}
	}
	return d.Name
}

// ---------------------------------------------------------------------------
// Zarr v2 (numpy) data type strings

// parseV2Time parses NumPy datetime64/timedelta64 type strings: "<M8[ns]",
// ">m8[10s]", or "<M8" (the generic unit).
func parseV2Time(s string) (DataType, bool) {
	if len(s) < 3 || (s[1] != 'M' && s[1] != 'm') || s[2] != '8' {
		return DataType{}, false
	}
	unit, scale := "generic", 1
	if rest := s[3:]; rest != "" {
		if len(rest) < 3 || rest[0] != '[' || rest[len(rest)-1] != ']' {
			return DataType{}, false
		}
		inner := rest[1 : len(rest)-1]
		i := 0
		for i < len(inner) && inner[i] >= '0' && inner[i] <= '9' {
			i++
		}
		if i > 0 {
			n, err := strconv.Atoi(inner[:i])
			if err != nil || n < 1 || n > math.MaxInt32 {
				return DataType{}, false
			}
			scale = n
		}
		unit = inner[i:]
	}
	d := Datetime64(unit, scale)
	if s[1] == 'm' {
		d = Timedelta64(unit, scale)
	}
	return d, validTimeUnit(unit)
}

// parseV2DataType parses a .zarray "dtype": a numpy type string such as
// "<f8" or a list of [name, type] pairs. objectCodec is the id of the
// object codec among the filters ("vlen-utf8", "vlen-bytes"), needed for
// "|O". bigEndian reports the stored byte order.
func parseV2DataType(raw json.RawMessage, objectCodec string) (d DataType, bigEndian bool, err error) {
	var fields [][]json.RawMessage
	if json.Unmarshal(raw, &fields) == nil {
		d = DataType{Name: "struct", Kind: KindStruct}
		anyBig, anyLittle := false, false
		for _, f := range fields {
			var name string
			if len(f) < 2 || json.Unmarshal(f[0], &name) != nil {
				return d, false, fmt.Errorf("zarr: invalid structured dtype %s", raw)
			}
			if len(f) > 2 {
				return d, false, fmt.Errorf("%w: structured dtype with sub-array field %q", ErrUnsupported, name)
			}
			ft, big, err := parseV2DataType(f[1], "")
			if err != nil {
				return d, false, err
			}
			if ft.Size > 1 && len(ft.swapUnits()) > 0 {
				anyBig = anyBig || big
				anyLittle = anyLittle || !big
			}
			if ft.Size > math.MaxInt32-d.Size { // NumPy's limit
				return d, false, fmt.Errorf("%w: structured dtype of more than %d bytes", ErrUnsupported, math.MaxInt32)
			}
			d.Fields = append(d.Fields, StructField{Name: name, Type: ft, Offset: d.Size})
			d.Size += ft.Size
		}
		if anyBig && anyLittle {
			return d, false, fmt.Errorf("%w: structured dtype mixing byte orders", ErrUnsupported)
		}
		return d, anyBig, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil || len(s) < 2 {
		return d, false, fmt.Errorf("zarr: invalid dtype %s", raw)
	}
	order, code := s[0], s[1:]
	if order != '<' && order != '>' && order != '|' && order != '=' {
		order, code = '|', s
	}
	bigEndian = order == '>'
	if strings.ContainsAny(code[:1], "Mm") {
		d, ok := parseV2Time(string(order) + code)
		if !ok {
			return d, false, fmt.Errorf("zarr: invalid dtype %q", s)
		}
		return d, bigEndian, nil
	}
	if code == "O" {
		switch objectCodec {
		case "vlen-utf8", "categorize", "":
			return DataType{Name: "string", Kind: KindString}, false, nil
		case "vlen-bytes":
			return DataType{Name: "variable_length_bytes", Kind: KindBytes}, false, nil
		}
		return d, false, fmt.Errorf("%w: object dtype with object codec %q", ErrUnsupported, objectCodec)
	}
	if len(code) < 2 {
		return d, false, fmt.Errorf("zarr: invalid dtype %q", s)
	}
	n, err := strconv.Atoi(code[1:])
	// NumPy's limit on element sizes is 2^31-1 bytes.
	if err != nil || n < 0 || n > math.MaxInt32 || (code[0] == 'U' && n > math.MaxInt32/4) {
		return d, false, fmt.Errorf("zarr: invalid dtype %q", s)
	}
	var name string
	switch code[0] {
	case 'b':
		name = "bool"
	case 'i':
		name = fmt.Sprintf("int%d", 8*n)
	case 'u':
		name = fmt.Sprintf("uint%d", 8*n)
	case 'f':
		name = fmt.Sprintf("float%d", 8*n)
	case 'c':
		name = fmt.Sprintf("complex%d", 8*n)
	case 'U':
		return FixedLengthUTF32(n), bigEndian, nil
	case 'S':
		return NullTerminatedBytes(n), false, nil
	case 'V':
		return RawBytes(n), false, nil
	default:
		return d, false, fmt.Errorf("%w: dtype %q", ErrUnsupported, s)
	}
	d, err = ParseDataType(name)
	if err != nil || d.Size != n || (code[0] == 'b' && n != 1) {
		return d, false, fmt.Errorf("%w: dtype %q", ErrUnsupported, s)
	}
	return d, bigEndian, nil
}

// v2JSON renders the data type as a .zarray dtype, as zarr-python writes it.
func (d DataType) v2JSON(bigEndian bool) (any, error) {
	order := "<"
	if bigEndian {
		order = ">"
	}
	if d.Size == 1 || d.Kind == KindFixedBytes || d.Kind == KindRaw {
		order = "|"
	}
	switch d.Kind {
	case KindBool:
		return "|b1", nil
	case KindInt, KindUint, KindFloat, KindComplex:
		if d.Bits != 0 || d.floatFormat() != nil && d.Name != "float16" {
			return nil, fmt.Errorf("%w: data type %s in Zarr v2", ErrUnsupported, d.Name)
		}
		code := map[Kind]string{KindInt: "i", KindUint: "u", KindFloat: "f", KindComplex: "c"}[d.Kind]
		return fmt.Sprintf("%s%s%d", order, code, d.Size), nil
	case KindDatetime, KindTimedelta:
		code := "M"
		if d.Kind == KindTimedelta {
			code = "m"
		}
		scale := ""
		if d.ScaleFactor > 1 {
			scale = strconv.Itoa(d.ScaleFactor)
		}
		if d.Unit == "generic" {
			// NumPy's dtype.str drops a generic unit's scale factor, so
			// zarr-python (3.4.1) spells a scaled one out: "<M8[2generic]".
			if scale == "" {
				return fmt.Sprintf("%s%s8", order, code), nil
			}
			return fmt.Sprintf("%s%s8[%sgeneric]", order, code, scale), nil
		}
		return fmt.Sprintf("%s%s8[%s%s]", order, code, scale, timeUnit(d.Unit)), nil
	case KindFixedString:
		return fmt.Sprintf("%sU%d", order, d.Size/4), nil
	case KindFixedBytes:
		if d.Name == "raw_bytes" {
			return fmt.Sprintf("|V%d", d.Size), nil
		}
		return fmt.Sprintf("|S%d", d.Size), nil
	case KindRaw:
		return fmt.Sprintf("|V%d", d.Size), nil
	case KindString, KindBytes:
		return "|O", nil
	case KindStruct:
		fields := make([]any, len(d.Fields))
		for i, f := range d.Fields {
			ft, err := f.Type.v2JSON(bigEndian)
			if err != nil {
				return nil, err
			}
			fields[i] = []any{f.Name, ft}
		}
		return fields, nil
	}
	return nil, fmt.Errorf("%w: data type %s in Zarr v2", ErrUnsupported, d.Name)
}
