package consistency

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"slices"

	zarr "github.com/solomonsealed/zarr-go"
)

// wireArray carries an array between Go and the oracle: fixed-size
// elements as base64 of the canonical little-endian layout, strings and
// variable-length bytes as lists.
type wireArray struct {
	Shape   []uint64 `json:"shape"`
	Data    *string  `json:"data,omitempty"`
	Strings []string `json:"strings,omitempty"`
	Bytes   []string `json:"bytes,omitempty"`
}

func toWire(n *zarr.NDArray) wireArray {
	w := wireArray{Shape: n.Shape}
	if w.Shape == nil {
		w.Shape = []uint64{}
	}
	switch n.DataType.Kind {
	case zarr.KindString:
		w.Strings = append([]string{}, n.Strings...)
	case zarr.KindBytes:
		w.Bytes = make([]string, len(n.Strings))
		for i, s := range n.Strings {
			w.Bytes[i] = base64.StdEncoding.EncodeToString([]byte(s))
		}
	default:
		s := base64.StdEncoding.EncodeToString(n.Data)
		w.Data = &s
	}
	return w
}

func fromWire(w wireArray, dt zarr.DataType) (*zarr.NDArray, error) {
	n := &zarr.NDArray{Shape: w.Shape, DataType: dt}
	if n.Shape == nil {
		n.Shape = []uint64{}
	}
	switch {
	case w.Data != nil:
		b, err := base64.StdEncoding.DecodeString(*w.Data)
		if err != nil {
			return nil, err
		}
		n.Data = b
		if dt.Variable() {
			return nil, fmt.Errorf("fixed-size data for %s", dt.Name)
		}
	case w.Bytes != nil:
		n.Strings = make([]string, len(w.Bytes))
		for i, s := range w.Bytes {
			b, err := base64.StdEncoding.DecodeString(s)
			if err != nil {
				return nil, err
			}
			n.Strings[i] = string(b)
		}
	default:
		n.Strings = w.Strings
		if n.Strings == nil {
			n.Strings = []string{}
		}
		if !dt.Variable() {
			if len(n.Strings) != 0 {
				return nil, fmt.Errorf("strings for %s", dt.Name)
			}
			n.Strings, n.Data = nil, []byte{}
		}
	}
	return n, nil
}

// sameArray reports whether two arrays have the same shape and elements
// (bit for bit).
func sameArray(a, b *zarr.NDArray) error {
	as, bs := a.Shape, b.Shape
	if as == nil {
		as = []uint64{}
	}
	if bs == nil {
		bs = []uint64{}
	}
	if !slices.Equal(as, bs) {
		return fmt.Errorf("shape %v vs %v", as, bs)
	}
	if a.DataType.Variable() {
		if len(a.Strings) != len(b.Strings) {
			return fmt.Errorf("%d vs %d strings", len(a.Strings), len(b.Strings))
		}
		for i := range a.Strings {
			if a.Strings[i] != b.Strings[i] {
				return fmt.Errorf("element %d: %q vs %q", i, a.Strings[i], b.Strings[i])
			}
		}
		return nil
	}
	if !bytes.Equal(a.Data, b.Data) {
		es := a.DataType.Size
		for i := 0; i+es <= len(a.Data) && i+es <= len(b.Data); i += es {
			if !bytes.Equal(a.Data[i:i+es], b.Data[i:i+es]) {
				return fmt.Errorf("element %d: % x vs % x (%d vs %d bytes)", i/es, a.Data[i:i+es], b.Data[i:i+es], len(a.Data), len(b.Data))
			}
		}
		return fmt.Errorf("%d vs %d bytes", len(a.Data), len(b.Data))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Selections

// dimItem is one per-dimension selector on the wire.
type dimItem struct {
	Index    *int64   `json:"index,omitempty"`
	Slice    []*int64 `json:"slice,omitempty"` // start, stop, step; null = None
	Ints     []int64  `json:"ints,omitempty"`
	Bools    []bool   `json:"bools,omitempty"`
	Ellipsis bool     `json:"ellipsis,omitempty"`
	isInts   bool
	isBools  bool
}

func (d dimItem) MarshalJSON() ([]byte, error) {
	switch {
	case d.Index != nil:
		return marshal(map[string]any{"index": *d.Index})
	case d.Slice != nil:
		return marshal(map[string]any{"slice": d.Slice})
	case d.isInts:
		ints := d.Ints
		if ints == nil {
			ints = []int64{}
		}
		return marshal(map[string]any{"ints": ints})
	case d.isBools:
		b := d.Bools
		if b == nil {
			b = []bool{}
		}
		return marshal(map[string]any{"bools": b})
	}
	return marshal(map[string]any{"ellipsis": true})
}

func (d *dimItem) UnmarshalJSON(b []byte) error {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return err
	}
	*d = dimItem{}
	switch {
	case m["index"] != nil:
		d.Index = new(int64)
		return json.Unmarshal(m["index"], d.Index)
	case m["slice"] != nil:
		return json.Unmarshal(m["slice"], &d.Slice)
	case m["ints"] != nil:
		d.isInts = true
		return json.Unmarshal(m["ints"], &d.Ints)
	case m["bools"] != nil:
		d.isBools = true
		return json.Unmarshal(m["bools"], &d.Bools)
	}
	d.Ellipsis = true
	return nil
}

func (d dimItem) goSelector() zarr.DimSelector {
	switch {
	case d.Index != nil:
		return zarr.Index(*d.Index)
	case d.Slice != nil:
		s := zarr.SliceSelector{Start: 0, Stop: zarr.End, Step: 1}
		if d.Slice[0] != nil {
			s.Start = *d.Slice[0]
		}
		if d.Slice[1] != nil {
			s.Stop = *d.Slice[1]
		}
		if d.Slice[2] != nil {
			s.Step = *d.Slice[2]
		}
		return s
	case d.isInts:
		return zarr.IndicesSelector(append([]int64{}, d.Ints...))
	case d.isBools:
		return zarr.MaskSelector(append([]bool{}, d.Bools...))
	}
	return zarr.Ellipsis
}

func (d dimItem) String() string {
	switch {
	case d.Index != nil:
		return fmt.Sprint(*d.Index)
	case d.Slice != nil:
		p := func(v *int64) string {
			if v == nil {
				return ""
			}
			return fmt.Sprint(*v)
		}
		return p(d.Slice[0]) + ":" + p(d.Slice[1]) + ":" + p(d.Slice[2])
	case d.isInts:
		return fmt.Sprint(d.Ints)
	case d.isBools:
		return fmt.Sprintf("mask%v", d.Bools)
	}
	return "..."
}

func idx(i int64) dimItem { return dimItem{Index: &i} }

func slc(start, stop, step *int64) dimItem { return dimItem{Slice: []*int64{start, stop, step}} }

func ints(v ...int64) dimItem { return dimItem{Ints: v, isInts: true} }

func bools(v ...bool) dimItem { return dimItem{Bools: v, isBools: true} }

func ptr(v int64) *int64 { return &v }

// selection is a selection on the wire.
type selection struct {
	Kind   string     `json:"kind"` // orthogonal, basic, coordinate, mask, block
	Items  []dimItem  `json:"items,omitempty"`
	Coords [][]int64  `json:"coords,omitempty"`
	Mask   *wireArray `json:"mask,omitempty"`
	mask   *zarr.NDArray
}

func (s selection) String() string {
	switch s.Kind {
	case "coordinate":
		return fmt.Sprintf("vindex%v", s.Coords)
	case "mask":
		return "mask"
	}
	return fmt.Sprintf("%s%v", s.Kind, s.Items)
}

func (s selection) goSelectors() []zarr.DimSelector {
	out := make([]zarr.DimSelector, len(s.Items))
	for i, it := range s.Items {
		out[i] = it.goSelector()
	}
	return out
}

func (s selection) get(ctx context.Context, a *zarr.Array) (*zarr.NDArray, error) {
	switch s.Kind {
	case "orthogonal", "basic":
		return a.Get(ctx, s.goSelectors()...)
	case "block":
		return a.GetBlocks(ctx, s.goSelectors()...)
	case "coordinate":
		return a.GetCoordinates(ctx, s.Coords...)
	case "mask":
		return a.GetMask(ctx, s.mask)
	}
	return nil, fmt.Errorf("unknown selection kind %q", s.Kind)
}

func (s selection) set(ctx context.Context, a *zarr.Array, data *zarr.NDArray) error {
	switch s.Kind {
	case "orthogonal", "basic":
		return a.Set(ctx, data, s.goSelectors()...)
	case "block":
		return a.SetBlocks(ctx, data, s.goSelectors()...)
	case "coordinate":
		return a.SetCoordinates(ctx, data, s.Coords...)
	case "mask":
		return a.SetMask(ctx, data, s.mask)
	}
	return fmt.Errorf("unknown selection kind %q", s.Kind)
}

func withMask(m *zarr.NDArray) selection {
	w := toWire(m)
	return selection{Kind: "mask", Mask: &w, mask: m}
}
