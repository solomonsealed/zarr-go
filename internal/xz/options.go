package xz

import (
	"encoding/json"
	"math"
	"sort"
	"strconv"
)

// lzmaOptions are the LZMA1/LZMA2 parameters of a filter specifier
// (liblzma's lzma_options_lzma).
type lzmaOptions struct {
	dictSize   uint32
	lc, lp, pb uint32
	mode       uint32
	niceLen    uint32
	mf         uint32
	depth      uint32
}

// chainFilter is a parsed filter of a chain.
type chainFilter struct {
	id    uint64
	lzma  lzmaOptions // FilterLZMA1, FilterLZMA2
	dist  int         // FilterDelta: 1..256
	start uint32      // BCJ filters: start_offset
}

// presetOptions mirrors liblzma's lzma_lzma_preset.
func presetOptions(preset uint32) (lzmaOptions, error) {
	level := preset & 0x1F
	flags := preset &^ 0x1F
	if level > 9 || flags&^PresetExtreme != 0 {
		return lzmaOptions{}, errorf(ErrUnsupported, "invalid compression preset: %d", preset)
	}
	dictPow2 := [10]uint8{18, 20, 21, 22, 22, 23, 23, 24, 25, 26}
	o := lzmaOptions{dictSize: 1 << dictPow2[level], lc: 3, lp: 0, pb: 2}
	if level <= 3 {
		o.mode = ModeFast
		o.mf = MFHC4
		if level == 0 {
			o.mf = MFHC3
		}
		o.niceLen = 273
		if level <= 1 {
			o.niceLen = 128
		}
		o.depth = [4]uint32{4, 8, 24, 48}[level]
	} else {
		o.mode = ModeNormal
		o.mf = MFBT4
		switch level {
		case 4:
			o.niceLen = 16
		case 5:
			o.niceLen = 32
		default:
			o.niceLen = 64
		}
		o.depth = 0
	}
	if flags&PresetExtreme != 0 {
		o.mode = ModeNormal
		o.mf = MFBT4
		if level == 3 || level == 5 {
			o.niceLen = 192
			o.depth = 0
		} else {
			o.niceLen = 273
			o.depth = 512
		}
	}
	return o, nil
}

func isBCJ(id uint64) bool {
	switch id {
	case FilterX86, FilterPowerPC, FilterIA64, FilterARM, FilterARMThumb, FilterSPARC, FilterARM64:
		return true
	}
	return false
}

func filterName(id uint64) string {
	switch id {
	case FilterLZMA1:
		return "LZMA1"
	case FilterLZMA2:
		return "LZMA2"
	case FilterDelta:
		return "delta"
	case FilterX86:
		return "x86"
	case FilterPowerPC:
		return "PowerPC"
	case FilterIA64:
		return "IA-64"
	case FilterARM:
		return "ARM"
	case FilterARMThumb:
		return "ARM-Thumb"
	case FilterSPARC:
		return "SPARC"
	case FilterARM64:
		return "ARM64"
	case FilterRISCV:
		return "RISC-V"
	}
	return "0x" + strconv.FormatUint(id, 16)
}

// parseChain converts Python-style filter specifiers into a chain. forEncode
// enables the validation liblzma does when initializing an encoder (the
// decoder ignores the encoder-only options).
func parseChain(filters []Filter, forEncode bool) ([]chainFilter, error) {
	if len(filters) > 4 {
		return nil, errorf(ErrUnsupported, "too many filters - liblzma supports a maximum of 4")
	}
	chain := make([]chainFilter, len(filters))
	for i, f := range filters {
		c, err := parseFilter(f, forEncode)
		if err != nil {
			return nil, err
		}
		chain[i] = c
	}
	return chain, nil
}

func parseFilter(f Filter, forEncode bool) (chainFilter, error) {
	c := chainFilter{id: f.ID}
	// Keys are processed in sorted order so that errors are deterministic.
	keys := make([]string, 0, len(f.Options))
	for k := range f.Options {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	get := func(k string) (uint32, error) {
		v, err := toUint32(f.Options[k])
		if err != nil {
			return 0, errorf(ErrUnsupported, "%s filter option %q: %v", filterName(f.ID), k, err)
		}
		return v, nil
	}
	badKey := func(k string) error {
		return errorf(ErrUnsupported, "invalid filter specifier for %s filter: unknown option %q", filterName(f.ID), k)
	}

	switch {
	case f.ID == FilterLZMA1 || f.ID == FilterLZMA2:
		preset := uint32(PresetDefault)
		if _, ok := f.Options["preset"]; ok {
			p, err := get("preset")
			if err != nil {
				return c, err
			}
			preset = p
		}
		o, err := presetOptions(preset)
		if err != nil {
			return c, err
		}
		for _, k := range keys {
			var dst *uint32
			switch k {
			case "preset":
				continue
			case "dict_size":
				dst = &o.dictSize
			case "lc":
				dst = &o.lc
			case "lp":
				dst = &o.lp
			case "pb":
				dst = &o.pb
			case "mode":
				dst = &o.mode
			case "nice_len":
				dst = &o.niceLen
			case "mf":
				dst = &o.mf
			case "depth":
				dst = &o.depth
			default:
				return c, badKey(k)
			}
			v, err := get(k)
			if err != nil {
				return c, err
			}
			*dst = v
		}
		if f.ID == FilterLZMA1 || forEncode {
			// liblzma's is_lclppb_valid (the LZMA2 decoder ignores lc/lp/pb).
			if o.lc > 4 || o.lp > 4 || o.lc+o.lp > 4 || o.pb > 4 {
				return c, errorf(ErrUnsupported, "%s: invalid lc/lp/pb (%d/%d/%d): need lc+lp <= 4 and pb <= 4", filterName(f.ID), o.lc, o.lp, o.pb)
			}
		}
		if forEncode {
			if o.dictSize < 4096 || o.dictSize > MaxDictSize {
				return c, errorf(ErrUnsupported, "%s: dict_size %d out of range [4096, %d]", filterName(f.ID), o.dictSize, MaxDictSize)
			}
			if o.mode != ModeFast && o.mode != ModeNormal {
				return c, errorf(ErrUnsupported, "%s: invalid mode %d", filterName(f.ID), o.mode)
			}
			switch o.mf {
			case MFHC3, MFHC4, MFBT2, MFBT3, MFBT4:
			default:
				return c, errorf(ErrUnsupported, "%s: invalid match finder %d", filterName(f.ID), o.mf)
			}
			if o.niceLen < 2 || o.niceLen > 273 {
				return c, errorf(ErrUnsupported, "%s: nice_len %d out of range [2, 273]", filterName(f.ID), o.niceLen)
			}
		} else if o.dictSize > MaxDictSize {
			return c, errorf(ErrMemLimit, "%s: dict_size %d exceeds %d", filterName(f.ID), o.dictSize, MaxDictSize)
		}
		c.lzma = o

	case f.ID == FilterDelta:
		dist := uint32(1)
		for _, k := range keys {
			if k != "dist" {
				return c, badKey(k)
			}
			v, err := get(k)
			if err != nil {
				return c, err
			}
			dist = v
		}
		if dist < 1 || dist > 256 {
			return c, errorf(ErrUnsupported, "delta: dist %d out of range [1, 256]", dist)
		}
		c.dist = int(dist)

	case isBCJ(f.ID):
		for _, k := range keys {
			if k != "start_offset" {
				return c, badKey(k)
			}
			v, err := get(k)
			if err != nil {
				return c, err
			}
			c.start = v
		}

	default:
		return c, errorf(ErrUnsupported, "invalid filter ID: %s", filterName(f.ID))
	}
	return c, nil
}

// validateXZChain applies liblzma's chain rules for .xz: 1-4 filters, the
// last (and only the last) being LZMA2.
func validateXZChain(chain []chainFilter) error {
	if len(chain) == 0 || len(chain) > 4 {
		return errorf(ErrUnsupported, "invalid filter chain length %d", len(chain))
	}
	for i, c := range chain {
		last := i == len(chain)-1
		if last != (c.id == FilterLZMA2) || c.id == FilterLZMA1 {
			return errorf(ErrUnsupported, "invalid filter chain for FORMAT_XZ: %s at position %d", filterName(c.id), i)
		}
	}
	return nil
}

// validateRawChain: like .xz but the last filter may also be LZMA1.
func validateRawChain(chain []chainFilter) error {
	if len(chain) == 0 || len(chain) > 4 {
		return errorf(ErrUnsupported, "invalid filter chain length %d", len(chain))
	}
	for i, c := range chain {
		last := i == len(chain)-1
		isLZMA := c.id == FilterLZMA1 || c.id == FilterLZMA2
		if last != isLZMA {
			return errorf(ErrUnsupported, "invalid filter chain: %s at position %d", filterName(c.id), i)
		}
	}
	return nil
}

// toUint32 converts a decoded-JSON or Go numeric value to uint32, as Python's
// uint32 converter does (rejecting negative, too large and non-integral
// values).
func toUint32(v any) (uint32, error) {
	u, err := toUint64(v)
	if err != nil {
		return 0, err
	}
	if u > math.MaxUint32 {
		return 0, errValueRange
	}
	return uint32(u), nil
}

var (
	errValueType  = errorf(ErrUnsupported, "value is not an integer")
	errValueRange = errorf(ErrUnsupported, "value out of range")
)

func toUint64(v any) (uint64, error) {
	signed := func(x int64) (uint64, error) {
		if x < 0 {
			return 0, errValueRange
		}
		return uint64(x), nil
	}
	switch x := v.(type) {
	case int:
		return signed(int64(x))
	case int8:
		return signed(int64(x))
	case int16:
		return signed(int64(x))
	case int32:
		return signed(int64(x))
	case int64:
		return signed(x)
	case uint:
		return uint64(x), nil
	case uint8:
		return uint64(x), nil
	case uint16:
		return uint64(x), nil
	case uint32:
		return uint64(x), nil
	case uint64:
		return x, nil
	case float32:
		return floatToUint64(float64(x))
	case float64:
		return floatToUint64(x)
	case json.Number:
		if u, err := strconv.ParseUint(string(x), 10, 64); err == nil {
			return u, nil
		}
		if i, err := strconv.ParseInt(string(x), 10, 64); err == nil {
			return signed(i)
		}
		f, err := x.Float64()
		if err != nil {
			return 0, errValueType
		}
		return floatToUint64(f)
	}
	return 0, errValueType
}

func floatToUint64(f float64) (uint64, error) {
	if f != math.Trunc(f) || math.IsInf(f, 0) || math.IsNaN(f) {
		return 0, errValueType
	}
	if f < 0 || f >= 1<<64 {
		return 0, errValueRange
	}
	return uint64(f), nil
}

// FilterFromMap converts a Python-style filter specifier, e.g. as decoded
// from numcodecs JSON ({"id": 3, "dist": 4}), into a Filter. The "id" entry
// is required.
//
// encoding/json decodes numbers into float64, which cannot represent
// FilterLZMA1 (2^62+1) exactly; the float64 value 2^62 is therefore taken to
// mean FilterLZMA1 (no filter has ID 2^62). Decoding with UseNumber avoids
// the ambiguity altogether.
func FilterFromMap(m map[string]any) (Filter, error) {
	raw, ok := m["id"]
	if !ok {
		return Filter{}, errorf(ErrUnsupported, "filter specifier must have an \"id\" entry")
	}
	var id uint64
	if f, isFloat := raw.(float64); isFloat && f == float64(FilterLZMA1) {
		id = FilterLZMA1
	} else {
		var err error
		if id, err = toUint64(raw); err != nil {
			return Filter{}, errorf(ErrUnsupported, "filter \"id\": %v", err)
		}
	}
	f := Filter{ID: id, Options: make(map[string]any, len(m)-1)}
	for k, v := range m {
		if k != "id" {
			f.Options[k] = v
		}
	}
	return f, nil
}

// Map returns the Python-style filter specifier for f (the inverse of
// FilterFromMap).
func (f Filter) Map() map[string]any {
	m := make(map[string]any, len(f.Options)+1)
	for k, v := range f.Options {
		m[k] = v
	}
	m["id"] = f.ID
	return m
}
