package zarr

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// The "template" chunk key encoding of zarr-extensions
// (chunk-key-encodings/template): keys are made from a template string
// whose braced expressions insert chunk coordinates. {0} is the first
// coordinate, {-1} the last, and {*} every coordinate not named elsewhere,
// joined by the configuration's separator; ":0N" after the index zero-pads
// to N digits ("{1:04}"). "{{" and "}}" are literal braces.
//
//	{"name": "template", "configuration": {"format": "c/{*}.chunk", "separator": "/"}}
//
// The template is checked against the array's dimensionality when its
// metadata is parsed, so every chunk has a key.

// keyTemplate is a parsed template string.
type keyTemplate struct {
	parts    []templatePart
	catchAll bool
	nIndex   int // index expressions
}

type templatePart struct {
	literal string
	expr    bool
	all     bool // {*}
	index   int  // for index expressions
	pad     int
}

// maxKeyPad bounds zero padding, which a few bytes of metadata could
// otherwise make arbitrarily long (uint64 values have at most 20 digits).
const maxKeyPad = 255

var templateExpr = regexp.MustCompile(`^(\*|-?[0-9]+)(?::0([0-9]+))?$`)

// Parsed templates, for the keys of every chunk. The cache is bounded:
// metadata from anywhere can name templates, and beyond the bound they are
// parsed on each use.
var (
	templateCache  sync.Map // template string -> *keyTemplate or error
	templateCached atomic.Int32
)

const maxCachedTemplates = 256

// parseKeyTemplate parses and checks a template string.
func parseKeyTemplate(s string) (*keyTemplate, error) {
	v, ok := templateCache.Load(s)
	if !ok {
		t, err := parseKeyTemplateUncached(s)
		if v = t; err != nil {
			v = err
		}
		if templateCached.Load() < maxCachedTemplates {
			if _, loaded := templateCache.LoadOrStore(s, v); !loaded {
				templateCached.Add(1)
			}
		}
	}
	if err, isErr := v.(error); isErr {
		return nil, err
	}
	return v.(*keyTemplate), nil
}

func parseKeyTemplateUncached(s string) (*keyTemplate, error) {
	bad := func(format string, args ...any) error {
		return fmt.Errorf("zarr: invalid chunk key template %q: %s", s, fmt.Sprintf(format, args...))
	}
	t := &keyTemplate{}
	seen := map[int]bool{}
	var lit strings.Builder
	flush := func() error {
		l := lit.String()
		lit.Reset()
		if l == "" {
			return nil
		}
		for _, sub := range []string{"..", `\`, "\x00", "//", "zarr.json"} {
			if strings.Contains(l, sub) {
				return bad("contains %q", sub)
			}
		}
		t.parts = append(t.parts, templatePart{literal: l})
		return nil
	}
	if strings.HasPrefix(s, "/") {
		return nil, bad("begins with /")
	}
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '{' && i+1 < len(s) && s[i+1] == '{', c == '}' && i+1 < len(s) && s[i+1] == '}':
			lit.WriteByte(c)
			i++
		case c == '}':
			return nil, bad("unmatched }")
		case c == '{':
			end := strings.IndexByte(s[i:], '}')
			if end < 0 {
				return nil, bad("unmatched {")
			}
			m := templateExpr.FindStringSubmatch(s[i+1 : i+end])
			// "-0" is outside the grammar (and implementations read it
			// differently).
			if m == nil || strings.HasPrefix(m[1], "-") && strings.TrimLeft(m[1][1:], "0") == "" {
				return nil, bad("invalid expression {%s}", s[i+1:i+end])
			}
			p := templatePart{expr: true}
			if m[2] != "" {
				pad, err := strconv.Atoi(m[2])
				if err != nil || pad > maxKeyPad {
					return nil, bad("invalid padding in {%s}", s[i+1:i+end])
				}
				p.pad = pad
			}
			if m[1] == "*" {
				if t.catchAll {
					return nil, bad("more than one {*}")
				}
				p.all, t.catchAll = true, true
			} else {
				n, err := strconv.Atoi(m[1])
				if err != nil {
					return nil, bad("invalid index in {%s}", s[i+1:i+end])
				}
				if seen[n] {
					return nil, bad("index %d used more than once", n)
				}
				seen[n] = true
				p.index = n
				t.nIndex++
			}
			if err := flush(); err != nil {
				return nil, err
			}
			t.parts = append(t.parts, p)
			i += end
		default:
			lit.WriteByte(c)
		}
	}
	if err := flush(); err != nil {
		return nil, err
	}
	if t.nIndex == 0 && !t.catchAll {
		return nil, bad("no braced expression")
	}
	return t, nil
}

// layout resolves the template for arrays of ndim dimensions: the
// coordinate each index expression names (pos, by part; -1 for others) and
// the coordinates {*} takes. Each index must name a different coordinate,
// every coordinate must be named once, and {*} must take at least one.
func (t *keyTemplate) layout(ndim int, separator string) (pos, rest []int, err error) {
	if t.catchAll && separator == "" {
		return nil, nil, fmt.Errorf("zarr: chunk key template with {*} needs a separator")
	}
	pos = make([]int, len(t.parts))
	named := make([]bool, ndim)
	for i, p := range t.parts {
		pos[i] = -1
		if !p.expr || p.all {
			continue
		}
		d := p.index
		if d < 0 {
			d += ndim
		}
		if d < 0 || d >= ndim {
			return nil, nil, fmt.Errorf("zarr: chunk key template index %d out of range for %d dimensions", p.index, ndim)
		}
		if named[d] {
			return nil, nil, fmt.Errorf("zarr: chunk key template names dimension %d more than once", d)
		}
		named[d], pos[i] = true, d
	}
	for d := range ndim {
		if !named[d] {
			rest = append(rest, d)
		}
	}
	switch {
	case !t.catchAll && len(rest) > 0:
		return nil, nil, fmt.Errorf("zarr: chunk key template has %d expressions for %d dimensions", t.nIndex, ndim)
	case t.catchAll && len(rest) == 0:
		return nil, nil, fmt.Errorf("zarr: chunk key template's {*} has no dimensions left of %d", ndim)
	}
	return pos, rest, nil
}

func padded(v uint64, pad int) string {
	s := strconv.FormatUint(v, 10)
	if len(s) < pad {
		s = strings.Repeat("0", pad-len(s)) + s
	}
	return s
}

// key renders the key of the chunk at coords; ok is false if the template
// does not suit len(coords) dimensions.
func (t *keyTemplate) key(coords []uint64, separator string) (string, bool) {
	pos, rest, err := t.layout(len(coords), separator)
	if err != nil {
		return "", false
	}
	var b strings.Builder
	for i, p := range t.parts {
		switch {
		case !p.expr:
			b.WriteString(p.literal)
		case p.all:
			for j, d := range rest {
				if j > 0 {
					b.WriteString(separator)
				}
				b.WriteString(padded(coords[d], p.pad))
			}
		default:
			b.WriteString(padded(coords[pos[i]], p.pad))
		}
	}
	return b.String(), true
}

// parse returns the coordinates whose key is key. For templates whose
// expressions are not separated by literals, keys can be ambiguous; parse
// then returns one of the coordinates that have the key.
func (t *keyTemplate) parse(key string, ndim int, separator string) ([]uint64, bool) {
	pos, rest, err := t.layout(ndim, separator)
	if err != nil {
		return nil, false
	}
	var re strings.Builder
	re.WriteString("^")
	for _, p := range t.parts {
		switch {
		case !p.expr:
			re.WriteString(regexp.QuoteMeta(p.literal))
		case p.all:
			re.WriteString(`([0-9]+(?:` + regexp.QuoteMeta(separator) + `[0-9]+){` + strconv.Itoa(len(rest)-1) + `})`)
		default:
			re.WriteString(`([0-9]+)`)
		}
	}
	re.WriteString("$")
	rx, err := regexp.Compile(re.String())
	if err != nil {
		return nil, false
	}
	m := rx.FindStringSubmatch(key)
	if m == nil {
		return nil, false
	}
	coords := make([]uint64, ndim)
	g := 1
	for i, p := range t.parts {
		if !p.expr {
			continue
		}
		vals, dims := []string{m[g]}, []int{pos[i]}
		if p.all {
			vals, dims = strings.Split(m[g], separator), rest
		}
		g++
		if len(vals) != len(dims) {
			return nil, false
		}
		for j, v := range vals {
			n, err := strconv.ParseUint(v, 10, 64)
			if err != nil {
				return nil, false
			}
			coords[dims[j]] = n
		}
	}
	// Only the canonical spelling is a key (no extra zero padding).
	if k, ok := t.key(coords, separator); !ok || k != key {
		return nil, false
	}
	return coords, true
}
