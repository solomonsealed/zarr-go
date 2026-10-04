package zarr

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// parseFillValue decodes a JSON fill value (Zarr v3 rules, accepting the
// v2 spellings too) into one element of the in-memory representation.
func parseFillValue(raw json.RawMessage, d DataType) (fillValue, error) {
	if len(raw) == 0 || string(raw) == "null" {
		if d.Variable() {
			return fillValue{}, nil
		}
		return fillValue{fixed: zeroValue(d)}, nil
	}
	bad := func() (fillValue, error) {
		return fillValue{}, fmt.Errorf("zarr: invalid fill_value %s for %s", raw, d.Name)
	}
	out := make([]byte, d.Size)
	switch d.Kind {
	case KindBool:
		var b bool
		if err := json.Unmarshal(raw, &b); err != nil {
			var n float64
			if json.Unmarshal(raw, &n) != nil {
				return bad()
			}
			b = n != 0
		}
		if b {
			out[0] = 1
		}
	case KindInt, KindUint, KindDatetime, KindTimedelta:
		var s string
		if json.Unmarshal(raw, &s) == nil {
			if s == "NaT" && (d.Kind == KindDatetime || d.Kind == KindTimedelta) {
				putInt(out, math.MinInt64)
				break
			}
			if bits, ok := parseHexBits(s, d.Size); ok {
				putUint(out, bits)
				break
			}
			return bad()
		}
		n, ok := new(big.Float).SetPrec(200).SetString(strings.TrimSpace(string(raw)))
		if !ok {
			return bad()
		}
		i, acc := n.Int(nil)
		if acc != big.Exact {
			return bad()
		}
		bits := d.Bits
		if bits == 0 {
			bits = 8 * d.Size
		}
		if d.Kind == KindUint {
			if i.Sign() < 0 || i.BitLen() > bits {
				return bad()
			}
			putUint(out, i.Uint64())
		} else {
			lim := new(big.Int).Lsh(big.NewInt(1), uint(bits-1))
			if i.Cmp(lim) >= 0 || i.Cmp(new(big.Int).Neg(lim)) < 0 {
				return bad()
			}
			v := i.Int64()
			if d.Bits != 0 {
				v &= 1<<d.Bits - 1
			}
			putInt(out, v)
		}
	case KindFloat:
		bits, ok := parseFloatFill(raw, d)
		if !ok {
			return bad()
		}
		putUint(out, bits)
	case KindComplex:
		var parts []json.RawMessage
		if json.Unmarshal(raw, &parts) != nil || len(parts) != 2 {
			return bad()
		}
		h := d.Size / 2
		for i, p := range parts {
			bits, ok := parseFloatFill(p, d)
			if !ok {
				return bad()
			}
			putUint(out[i*h:(i+1)*h], bits)
		}
	case KindRaw, KindFixedBytes:
		b, ok := parseBytesFill(raw)
		if !ok || len(b) > len(out) {
			return bad()
		}
		copy(out, b)
	case KindFixedString:
		var s string
		if json.Unmarshal(raw, &s) != nil {
			return bad()
		}
		i := 0
		for _, r := range s {
			if i+4 > len(out) {
				return bad()
			}
			binary.LittleEndian.PutUint32(out[i:], uint32(r))
			i += 4
		}
	case KindString:
		var s string
		if json.Unmarshal(raw, &s) != nil {
			return bad()
		}
		return fillValue{str: s}, nil
	case KindBytes:
		b, ok := parseBytesFill(raw)
		if !ok {
			return bad()
		}
		return fillValue{str: string(b)}, nil
	case KindStruct:
		var fields map[string]json.RawMessage
		if json.Unmarshal(raw, &fields) == nil {
			for name, v := range fields {
				f, ok := d.Field(name)
				if !ok {
					return bad()
				}
				fv, err := parseFillValue(v, f.Type)
				if err != nil {
					return fillValue{}, err
				}
				copy(out[f.Offset:], fv.fixed)
			}
			break
		}
		b, ok := parseBytesFill(raw)
		if !ok || len(b) != len(out) {
			return bad()
		}
		copy(out, b)
	}
	return fillValue{fixed: out}, nil
}

// zeroValue is NumPy's cast of 0 to d: zero bytes, except that text
// fields of structs become "0".
func zeroValue(d DataType) []byte {
	out := make([]byte, d.Size)
	if d.Kind == KindStruct {
		for _, f := range d.Fields {
			switch f.Type.Kind {
			case KindStruct:
				copy(out[f.Offset:], zeroValue(f.Type))
			case KindFixedString:
				if f.Type.Size >= 4 {
					out[f.Offset] = '0'
				}
			case KindFixedBytes:
				if f.Type.Name == "null_terminated_bytes" && f.Type.Size > 0 {
					out[f.Offset] = '0'
				}
			}
		}
	}
	return out
}

// parseBytesFill accepts base64 strings (zarr-python) and integer lists
// (the v3 spec for rN types).
func parseBytesFill(raw json.RawMessage) ([]byte, bool) {
	var ints []int
	if json.Unmarshal(raw, &ints) == nil {
		b := make([]byte, len(ints))
		for i, v := range ints {
			if v < 0 || v > 255 {
				return nil, false
			}
			b[i] = byte(v)
		}
		return b, true
	}
	var s string
	if json.Unmarshal(raw, &s) != nil {
		return nil, false
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, false
	}
	return b, true
}

func parseHexBits(s string, size int) (uint64, bool) {
	if !strings.HasPrefix(s, "0x") || len(s) <= 2 || len(s) > 2+2*size {
		return 0, false
	}
	u, err := strconv.ParseUint(s[2:], 16, 64)
	return u, err == nil
}

// canonicalNaN returns the quiet NaN numpy produces for "NaN".
func canonicalNaN(d DataType) uint64 {
	switch d.componentSize() {
	case 4:
		if d.floatFormat() == nil {
			return 0x7fc00000
		}
	case 8:
		return 0x7ff8000000000000
	}
	return d.floatFormat().nan(false)
}

// parseFloatFill returns the bits of one float component of d.
func parseFloatFill(raw json.RawMessage, d DataType) (uint64, bool) {
	size := d.componentSize()
	var f float64
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "NaN":
			return canonicalNaN(d), true
		case "Infinity":
			f = math.Inf(1)
		case "-Infinity":
			f = math.Inf(-1)
		default:
			return parseHexBits(s, size)
		}
	} else if json.Unmarshal(raw, &f) != nil {
		return 0, false
	}
	if ff := d.floatFormat(); ff != nil {
		return ff.encode(f), true
	}
	if size == 4 {
		return uint64(math.Float32bits(float32(f))), true
	}
	return math.Float64bits(f), true
}

// ---------------------------------------------------------------------------
// Rendering

// floatFillJSON renders a float component as zarr-python does.
func floatFillJSON(f float64) any {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	}
	return f
}

// fillJSON renders the in-memory fill value as zarr.json's fill_value
// (format 3) or .zarray's (format 2).
func fillJSON(d DataType, f fillValue, format int) any {
	switch d.Kind {
	case KindBool:
		return f.fixed[0] != 0
	case KindInt, KindDatetime, KindTimedelta:
		return intAt(f.fixed, d.Size, d.Bits)
	case KindUint:
		return uintAt(f.fixed, d.Size, d.Bits)
	case KindFloat:
		return floatFillJSON(floatAt(d, f.fixed))
	case KindComplex:
		h := d.Size / 2
		return []any{floatFillJSON(floatAt(d, f.fixed)), floatFillJSON(floatAt(d, f.fixed[h:]))}
	case KindRaw:
		out := make([]any, len(f.fixed))
		for i, b := range f.fixed {
			out[i] = int(b)
		}
		return out
	case KindFixedBytes:
		b := f.fixed
		if d.Name == "null_terminated_bytes" {
			// A numpy bytes_ value: trailing NULs are not part of it.
			b = bytes.TrimRight(b, "\x00")
		}
		return base64.StdEncoding.EncodeToString(b)
	case KindFixedString:
		return utf32String(f.fixed)
	case KindString:
		return f.str
	case KindBytes:
		return base64.StdEncoding.EncodeToString([]byte(f.str))
	case KindStruct:
		if format == 2 || d.Name == "structured" {
			return base64.StdEncoding.EncodeToString(f.fixed)
		}
		obj := jsonObject{}
		for _, fl := range d.Fields {
			obj = append(obj, jsonMember{fl.Name, fillJSON(fl.Type, fillValue{fixed: f.fixed[fl.Offset : fl.Offset+fl.Type.Size]}, format)})
		}
		return obj
	}
	return nil
}

// fillFromGo converts a Go fill value (ArraySpec.FillValue) to the
// in-memory representation. nil means the type's zero value.
func fillFromGo(d DataType, v any) (fillValue, error) {
	if v == nil {
		return parseFillValue(nil, d)
	}
	switch x := v.(type) {
	case json.RawMessage:
		return parseFillValue(x, d)
	case *NDArray:
		if !x.DataType.Equal(d) || x.Len() != 1 {
			return fillValue{}, fmt.Errorf("zarr: fill value array of %s does not fit %s", x.DataType.Name, d.Name)
		}
		if err := x.validate(); err != nil {
			return fillValue{}, err
		}
		return fillOf(x), nil
	case string:
		// Strings are values for string types, and the JSON spellings
		// ("NaN", "Infinity", "0x7fc00000", "NaT", base64) otherwise.
		if d.Kind == KindString {
			return fillValue{str: x}, nil
		}
		if d.Kind == KindBytes || d.Kind == KindFixedBytes || d.Kind == KindRaw || d.Kind == KindStruct {
			if d.Kind == KindBytes {
				return fillValue{str: x}, nil
			}
			if len(x) > d.Size {
				return fillValue{}, fmt.Errorf("zarr: fill value of %d bytes does not fit %s", len(x), d.Name)
			}
			b := make([]byte, d.Size)
			copy(b, x)
			return fillValue{fixed: b}, nil
		}
		raw, _ := json.Marshal(x)
		return parseFillValue(raw, d)
	case []byte:
		return fillFromGo(d, string(x))
	}
	n, err := FromValues([]uint64{1}, d, []any{v})
	if err != nil {
		return fillValue{}, fmt.Errorf("zarr: fill value %v (%T) does not fit %s: %w", v, v, d.Name, err)
	}
	fv := fillOf(n)
	// Go's math.NaN() has a payload numpy's NaN lacks; store the canonical
	// NaN so the fill value reads back as zarr-python writes it.
	if (d.Kind == KindFloat || d.Kind == KindComplex) && d.floatFormat() == nil {
		cs := d.componentSize()
		for p := 0; p+cs <= len(fv.fixed); p += cs {
			if math.IsNaN(floatAt(d, fv.fixed[p:])) {
				putUint(fv.fixed[p:p+cs], canonicalNaN(d)|uint64(fv.fixed[p+cs-1]&0x80)<<(8*(cs-1)))
			}
		}
	}
	return fv, nil
}
