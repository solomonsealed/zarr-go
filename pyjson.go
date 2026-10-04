package zarr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Metadata documents are written the way zarr-python writes them
// (json.dumps(doc, indent=2)): keys in insertion order, non-ASCII escaped,
// floats in Python's repr, so the same array definition produces the same
// bytes from either implementation.

// jsonObject is a JSON object that keeps its key order.
type jsonObject []jsonMember

type jsonMember struct {
	Key   string
	Value any
}

// get returns the value of key.
func (o jsonObject) get(key string) (any, bool) {
	for _, m := range o {
		if m.Key == key {
			return m.Value, true
		}
	}
	return nil, false
}

// set replaces key's value, or appends it.
func (o *jsonObject) set(key string, v any) {
	for i, m := range *o {
		if m.Key == key {
			(*o)[i].Value = v
			return
		}
	}
	*o = append(*o, jsonMember{key, v})
}

func (o *jsonObject) del(key string) {
	*o = slices.DeleteFunc(*o, func(m jsonMember) bool { return m.Key == key })
}

// MarshalJSON lets jsonObject values be passed to encoding/json too.
func (o jsonObject) MarshalJSON() ([]byte, error) { return marshalPython(o, -1) }

// parseOrdered decodes JSON keeping object key order (as jsonObject) and
// numbers exact (as json.Number). Python's bare NaN, Infinity and
// -Infinity become float64 values.
func parseOrdered(raw []byte) (any, error) {
	v, err := parseOrderedStrict(raw)
	if err == nil {
		return v, nil
	}
	fixed, ok := pythonNonFinite(raw)
	if !ok {
		return nil, err
	}
	v, err2 := parseOrderedStrict(fixed)
	if err2 != nil {
		return nil, err
	}
	return replaceNonFinite(v), nil
}

func parseOrderedStrict(raw []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	v, err := parseOrderedValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err == nil {
		return nil, fmt.Errorf("zarr: trailing data after JSON value")
	}
	return v, nil
}

// Markers for Python's non-finite float tokens while decoding.
const (
	nanToken    = "\x00zarr:NaN"
	infToken    = "\x00zarr:Infinity"
	negInfToken = "\x00zarr:-Infinity"
)

func nanValue() float64         { return math.NaN() }
func infValue(sign int) float64 { return math.Inf(sign) }

// pyCompat returns doc with Python's bare NaN/Infinity tokens replaced by
// marker strings (which decodeAttributes turns back into floats), so
// encoding/json can decode documents zarr-python wrote.
func pyCompat(doc []byte) []byte {
	if json.Valid(doc) {
		return doc
	}
	if fixed, ok := pythonNonFinite(doc); ok {
		return fixed
	}
	return doc
}

// pythonNonFinite replaces bare NaN, Infinity and -Infinity tokens
// (which Python's json writes for non-finite floats) by marker strings.
func pythonNonFinite(raw []byte) ([]byte, bool) {
	var out bytes.Buffer
	found := false
	inString := false
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		if inString {
			out.WriteByte(c)
			if c == '\\' && i+1 < len(raw) {
				i++
				out.WriteByte(raw[i])
			} else if c == '"' {
				inString = false
			}
			continue
		}
		switch {
		case c == '"':
			inString = true
			out.WriteByte(c)
		case bytes.HasPrefix(raw[i:], []byte("NaN")):
			out.Write(jsonString(nanToken))
			i += 2
			found = true
		case bytes.HasPrefix(raw[i:], []byte("Infinity")):
			out.Write(jsonString(infToken))
			i += 7
			found = true
		case bytes.HasPrefix(raw[i:], []byte("-Infinity")):
			out.Write(jsonString(negInfToken))
			i += 8
			found = true
		default:
			out.WriteByte(c)
		}
	}
	return out.Bytes(), found
}

func jsonString(s string) []byte {
	b, _ := json.Marshal(s)
	return b
}

func replaceNonFinite(v any) any {
	switch x := v.(type) {
	case jsonObject:
		for i := range x {
			x[i].Value = replaceNonFinite(x[i].Value)
		}
	case []any:
		for i := range x {
			x[i] = replaceNonFinite(x[i])
		}
	case string:
		switch x {
		case nanToken:
			return math.NaN()
		case infToken:
			return math.Inf(1)
		case negInfToken:
			return math.Inf(-1)
		}
	}
	return v
}

func parseOrderedValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			obj := jsonObject{}
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				k, _ := kt.(string)
				v, err := parseOrderedValue(dec)
				if err != nil {
					return nil, err
				}
				obj.set(k, v)
			}
			_, err := dec.Token()
			return obj, err
		case '[':
			arr := []any{}
			for dec.More() {
				v, err := parseOrderedValue(dec)
				if err != nil {
					return nil, err
				}
				arr = append(arr, v)
			}
			_, err := dec.Token()
			return arr, err
		}
		return nil, fmt.Errorf("zarr: unexpected JSON delimiter %v", t)
	}
	return tok, nil
}

// marshalPython encodes v like Python's json.dumps(v, indent=indent);
// indent < 0 means indent=None (compact, ", " and ": " separators).
func marshalPython(v any, indent int) ([]byte, error) {
	var buf bytes.Buffer
	if err := writePython(&buf, v, indent, 0); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func writePython(buf *bytes.Buffer, v any, indent, depth int) error {
	newline := func(d int) {
		if indent >= 0 {
			buf.WriteByte('\n')
			buf.WriteString(strings.Repeat(" ", indent*d))
		}
	}
	sep := ", "
	if indent >= 0 {
		sep = ","
	}
	switch x := v.(type) {
	case nil:
		buf.WriteString("null")
		return nil
	case bool:
		if x {
			buf.WriteString("true")
		} else {
			buf.WriteString("false")
		}
		return nil
	case string:
		writePythonString(buf, x)
		return nil
	case json.Number:
		s := x.String()
		if strings.ContainsAny(s, ".eE") {
			f, err := x.Float64()
			if err != nil && !math.IsInf(f, 0) {
				return err
			}
			buf.WriteString(pythonFloat(f))
		} else {
			buf.WriteString(strings.TrimPrefix(s, "+"))
		}
		return nil
	case float64:
		buf.WriteString(pythonFloat(x))
		return nil
	case float32:
		buf.WriteString(pythonFloat(float64(x)))
		return nil
	case int:
		buf.WriteString(strconv.Itoa(x))
		return nil
	case int64:
		buf.WriteString(strconv.FormatInt(x, 10))
		return nil
	case uint64:
		buf.WriteString(strconv.FormatUint(x, 10))
		return nil
	case json.RawMessage:
		p, err := parseOrdered(x)
		if err != nil {
			return err
		}
		return writePython(buf, p, indent, depth)
	case jsonObject:
		if len(x) == 0 {
			buf.WriteString("{}")
			return nil
		}
		buf.WriteByte('{')
		for i, m := range x {
			if i > 0 {
				buf.WriteString(sep)
			}
			newline(depth + 1)
			writePythonString(buf, m.Key)
			buf.WriteString(": ")
			if err := writePython(buf, m.Value, indent, depth+1); err != nil {
				return err
			}
		}
		newline(depth)
		buf.WriteByte('}')
		return nil
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		obj := make(jsonObject, len(keys))
		for i, k := range keys {
			obj[i] = jsonMember{k, x[k]}
		}
		return writePython(buf, obj, indent, depth)
	case []any:
		if len(x) == 0 {
			buf.WriteString("[]")
			return nil
		}
		buf.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				buf.WriteString(sep)
			}
			newline(depth + 1)
			if err := writePython(buf, e, indent, depth+1); err != nil {
				return err
			}
		}
		newline(depth)
		buf.WriteByte(']')
		return nil
	case json.Marshaler:
		raw, err := x.MarshalJSON()
		if err != nil {
			return err
		}
		return writePython(buf, json.RawMessage(raw), indent, depth)
	}
	// Other slices, maps and numbers via reflection.
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Slice, reflect.Array:
		if rv.Kind() == reflect.Slice && rv.IsNil() {
			buf.WriteString("null")
			return nil
		}
		items := make([]any, rv.Len())
		for i := range items {
			items[i] = rv.Index(i).Interface()
		}
		return writePython(buf, items, indent, depth)
	case reflect.Map:
		if rv.Type().Key().Kind() != reflect.String {
			break
		}
		m := make(map[string]any, rv.Len())
		for it := rv.MapRange(); it.Next(); {
			m[it.Key().String()] = it.Value().Interface()
		}
		return writePython(buf, m, indent, depth)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		buf.WriteString(strconv.FormatInt(rv.Int(), 10))
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		buf.WriteString(strconv.FormatUint(rv.Uint(), 10))
		return nil
	case reflect.Float32, reflect.Float64:
		buf.WriteString(pythonFloat(rv.Float()))
		return nil
	case reflect.Bool:
		return writePython(buf, rv.Bool(), indent, depth)
	case reflect.String:
		writePythonString(buf, rv.String())
		return nil
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			buf.WriteString("null")
			return nil
		}
		return writePython(buf, rv.Elem().Interface(), indent, depth)
	}
	// Structs and anything else: through encoding/json.
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return writePython(buf, json.RawMessage(raw), indent, depth)
}

// writePythonString writes s as json.dumps does with ensure_ascii=True.
func writePythonString(buf *bytes.Buffer, s string) {
	buf.WriteByte('"')
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		i += size
		switch {
		case r == '"':
			buf.WriteString(`\"`)
		case r == '\\':
			buf.WriteString(`\\`)
		case r == '\n':
			buf.WriteString(`\n`)
		case r == '\r':
			buf.WriteString(`\r`)
		case r == '\t':
			buf.WriteString(`\t`)
		case r == '\b':
			buf.WriteString(`\b`)
		case r == '\f':
			buf.WriteString(`\f`)
		case r < 0x20 || (r >= 0x7f && r < 0x10000):
			if r == 0x7f {
				buf.WriteByte(0x7f) // DEL is ASCII; Python leaves it as is
				continue
			}
			fmt.Fprintf(buf, `\u%04x`, r)
		case r >= 0x10000:
			r -= 0x10000
			fmt.Fprintf(buf, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
		default:
			buf.WriteRune(r)
		}
	}
	buf.WriteByte('"')
}

// pythonFloat formats f as Python's float.__repr__ (and json.dumps).
func pythonFloat(f float64) string {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	}
	// Shortest round-trip digits and decimal exponent.
	e := strconv.FormatFloat(f, 'e', -1, 64) // e.g. "-1.2345e+06"
	neg := strings.HasPrefix(e, "-")
	e = strings.TrimPrefix(e, "-")
	mant, expStr, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expStr)
	digits := strings.Replace(mant, ".", "", 1)
	var s string
	if exp < -4 || exp >= 16 {
		s = digits[:1]
		if len(digits) > 1 {
			s += "." + digits[1:]
		}
		sign := "+"
		if exp < 0 {
			sign, exp = "-", -exp
		}
		s += fmt.Sprintf("e%s%02d", sign, exp)
	} else if exp < 0 {
		s = "0." + strings.Repeat("0", -exp-1) + digits
	} else if len(digits) <= exp+1 {
		s = digits + strings.Repeat("0", exp+1-len(digits)) + ".0"
	} else {
		s = digits[:exp+1] + "." + digits[exp+1:]
	}
	if neg {
		s = "-" + s
	}
	return s
}
