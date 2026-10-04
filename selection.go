package zarr

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseSelector parses a NumPy-style selection, as written between the
// brackets of arr[...]: comma-separated items, each an index ("3", "-1"),
// a slice ("2:10", "::2", "-5:"), an integer list ("[0, 4, 2]"), a boolean
// list ("[true, false, true]") or "...". A selection wrapped as a whole in
// brackets ("[:4, 19]") is unwrapped. An empty string selects everything.
func ParseSelector(sel string) ([]DimSelector, error) {
	sel = strings.TrimSpace(sel)
	if inner, ok := strings.CutPrefix(sel, "["); ok && strings.HasSuffix(sel, "]") && matchingBracket(sel) == len(sel)-1 {
		inner = strings.TrimSuffix(inner, "]")
		if strings.ContainsAny(inner, ":[") || strings.Contains(inner, "...") {
			sel = strings.TrimSpace(inner)
		}
	}
	if sel == "" {
		return nil, nil
	}
	items, err := splitTopLevel(sel)
	if err != nil {
		return nil, err
	}
	out := make([]DimSelector, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		switch {
		case item == "...":
			out = append(out, Ellipsis)
		case strings.HasPrefix(item, "["):
			if !strings.HasSuffix(item, "]") {
				return nil, fmt.Errorf("zarr: invalid list %q", item)
			}
			body := strings.TrimSpace(item[1 : len(item)-1])
			var parts []string
			if body != "" {
				parts = strings.Split(body, ",")
			}
			bools, ints := []bool{}, []int64{}
			for _, p := range parts {
				p = strings.TrimSpace(p)
				switch strings.ToLower(p) {
				case "true":
					bools = append(bools, true)
					continue
				case "false":
					bools = append(bools, false)
					continue
				}
				v, err := strconv.ParseInt(p, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("zarr: invalid index %q", p)
				}
				ints = append(ints, v)
			}
			if len(bools) > 0 && len(ints) > 0 {
				return nil, fmt.Errorf("zarr: list %q mixes integers and booleans", item)
			}
			if len(bools) > 0 {
				out = append(out, MaskSelector(bools))
			} else {
				out = append(out, IndicesSelector(ints))
			}
		case strings.Contains(item, ":"):
			parts := strings.Split(item, ":")
			if len(parts) > 3 {
				return nil, fmt.Errorf("zarr: invalid slice %q", item)
			}
			s := SliceSelector{0, End, 1}
			vals := []*int64{&s.Start, &s.Stop, &s.Step}
			for i, p := range parts {
				p = strings.TrimSpace(p)
				if p == "" {
					continue
				}
				v, err := strconv.ParseInt(p, 10, 64)
				if err != nil {
					return nil, fmt.Errorf("zarr: invalid slice %q", item)
				}
				*vals[i] = v
			}
			out = append(out, s)
		default:
			v, err := strconv.ParseInt(item, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("zarr: invalid index %q", item)
			}
			out = append(out, IndexSelector(v))
		}
	}
	return out, nil
}

func matchingBracket(s string) int {
	depth := 0
	for i, c := range s {
		switch c {
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// splitTopLevel splits on commas outside brackets.
func splitTopLevel(s string) ([]string, error) {
	var out []string
	depth, start := 0, 0
	for i, c := range s {
		switch c {
		case '[':
			depth++
		case ']':
			depth--
			if depth < 0 {
				return nil, fmt.Errorf("zarr: unbalanced brackets in %q", s)
			}
		case ',':
			if depth == 0 {
				out = append(out, s[start:i])
				start = i + 1
			}
		}
	}
	if depth != 0 {
		return nil, fmt.Errorf("zarr: unbalanced brackets in %q", s)
	}
	out = append(out, s[start:])
	return out, nil
}

// ParseSelection parses a basic selection of indices and step-1 slices
// (see ParseSelector) into a region: its start, extent, and which
// dimensions were selected by a single index.
func ParseSelection(sel string, shape []uint64) (start, count []uint64, squeeze []bool, err error) {
	items, err := ParseSelector(sel)
	if err != nil {
		return nil, nil, nil, err
	}
	ds, err := normalizeSelection(items, shape)
	if err != nil {
		return nil, nil, nil, err
	}
	nd := len(shape)
	start, count, squeeze = make([]uint64, nd), make([]uint64, nd), make([]bool, nd)
	for d, s := range ds {
		if s.list != nil || (s.step != 1 && s.count > 1) {
			return nil, nil, nil, fmt.Errorf("zarr: selection %q is not a region (steps and lists need Get)", sel)
		}
		start[d], count[d], squeeze[d] = s.start, s.count, s.squeeze
	}
	return start, count, squeeze, nil
}
