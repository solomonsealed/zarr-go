package zarr

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"slices"
	"unicode/utf8"
)

// Core Zarr v3 codecs: bytes, vlen-utf8 / vlen-bytes and transpose.

func init() {
	RegisterCodec("bytes", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Endian *string `json:"endian"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		big := false
		if c.Endian != nil {
			switch *c.Endian {
			case "big":
				big = true
			case "little":
			default:
				return nil, fmt.Errorf("invalid endian %q", *c.Endian)
			}
		}
		return newBytesSerializer(in, big)
	})
	vlen := func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		if !in.DataType.Variable() {
			return nil, fmt.Errorf("vlen codecs need a variable-length data type, not %s", in.DataType.Name)
		}
		return vlenCodec{}, nil
	}
	for _, name := range []string{"vlen-utf8", "vlen-bytes", "numcodecs.vlen-utf8", "numcodecs.vlen-bytes"} {
		RegisterCodec(name, vlen)
	}
	RegisterCodec("transpose", func(cfg json.RawMessage, in ChunkSpec) (any, error) {
		var c struct {
			Order json.RawMessage `json:"order"`
		}
		if err := json.Unmarshal(cfg, &c); err != nil {
			return nil, err
		}
		var order []int
		var named string
		if json.Unmarshal(c.Order, &order) != nil {
			if json.Unmarshal(c.Order, &named) != nil || (named != "C" && named != "F") {
				return nil, fmt.Errorf("invalid transpose order %s", c.Order)
			}
		}
		t := &transposeCodec{order: order, fortran: named == "F"}
		if in.Shape != nil {
			if _, err := t.perm(len(in.Shape)); err != nil {
				return nil, err
			}
		}
		return t, nil
	})
}

// ---------------------------------------------------------------------------
// bytes

type bytesCodec struct{ big bool }

func newBytesSerializer(in ChunkSpec, big bool) (ArrayToBytesCodec, error) {
	if in.DataType.Variable() {
		return nil, fmt.Errorf("zarr: the bytes codec cannot encode variable-length %s data", in.DataType.Name)
	}
	return &bytesCodec{big: big}, nil
}

func (b *bytesCodec) Decode(_ context.Context, in []byte, spec ChunkSpec) (*NDArray, error) {
	want := numElements(spec.Shape) * uint64(spec.DataType.Size)
	if uint64(len(in)) != want {
		return nil, fmt.Errorf("zarr: decoded chunk has %d bytes, expected %d", len(in), want)
	}
	if b.big {
		in = byteSwap(spec.DataType, in) // a copy
	} else {
		// The input may be a stored value (a cache entry, a checksum
		// codec's sub-slice): never hand it out as mutable array data.
		in = slices.Clone(in)
	}
	return &NDArray{Shape: spec.Shape, DataType: spec.DataType, Data: in}, nil
}

func (b *bytesCodec) Encode(_ context.Context, a *NDArray, spec ChunkSpec) ([]byte, error) {
	if b.big {
		return byteSwap(spec.DataType, a.Data), nil
	}
	return a.Data, nil
}

func (b *bytesCodec) encodedSize(spec ChunkSpec) (int, bool) {
	return int(numElements(spec.Shape)) * spec.DataType.Size, true
}

// ---------------------------------------------------------------------------
// vlen-utf8 / vlen-bytes: a uint32 item count followed by (uint32 length,
// bytes) per item, all little-endian.

type vlenCodec struct{}

func (vlenCodec) Decode(_ context.Context, in []byte, spec ChunkSpec) (*NDArray, error) {
	if len(in) < 4 {
		return nil, fmt.Errorf("zarr: vlen chunk too short")
	}
	n := uint64(binary.LittleEndian.Uint32(in))
	if want := numElements(spec.Shape); n != want {
		return nil, fmt.Errorf("zarr: vlen chunk has %d items, expected %d", n, want)
	}
	if n > uint64(len(in)-4)/4 { // each item has a length
		return nil, fmt.Errorf("zarr: truncated vlen chunk")
	}
	strs := make([]string, n)
	pos := 4
	for i := range strs {
		if pos+4 > len(in) {
			return nil, fmt.Errorf("zarr: truncated vlen chunk")
		}
		l := int(binary.LittleEndian.Uint32(in[pos:]))
		pos += 4
		if l < 0 || l > len(in)-pos {
			return nil, fmt.Errorf("zarr: truncated vlen chunk")
		}
		strs[i] = string(in[pos : pos+l])
		pos += l
	}
	return &NDArray{Shape: spec.Shape, DataType: spec.DataType, Strings: strs}, nil
}

func (vlenCodec) Encode(_ context.Context, a *NDArray, spec ChunkSpec) ([]byte, error) {
	size := 4
	for _, s := range a.Strings {
		size += 4 + len(s)
	}
	out := make([]byte, 0, size)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(a.Strings)))
	for _, s := range a.Strings {
		if spec.DataType.Kind == KindString && !utf8.ValidString(s) {
			return nil, fmt.Errorf("zarr: string %q is not valid UTF-8", s)
		}
		out = binary.LittleEndian.AppendUint32(out, uint32(len(s)))
		out = append(out, s...)
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// transpose

type transposeCodec struct {
	order   []int
	fortran bool
}

func (t *transposeCodec) perm(n int) ([]int, error) {
	if t.order == nil {
		p := make([]int, n)
		for i := range p {
			p[i] = i
			if t.fortran {
				p[i] = n - 1 - i
			}
		}
		return p, nil
	}
	if len(t.order) != n {
		return nil, fmt.Errorf("zarr: transpose order has %d entries for %d dimensions", len(t.order), n)
	}
	seen := make([]bool, n)
	for _, o := range t.order {
		if o < 0 || o >= n || seen[o] {
			return nil, fmt.Errorf("zarr: invalid transpose order %v", t.order)
		}
		seen[o] = true
	}
	return t.order, nil
}

func (t *transposeCodec) Resolve(in ChunkSpec) (ChunkSpec, error) {
	p, err := t.perm(len(in.Shape))
	if err != nil {
		return in, err
	}
	out := make([]uint64, len(in.Shape))
	for i, o := range p {
		out[i] = in.Shape[o]
	}
	return in.withShape(out), nil
}

func (t *transposeCodec) Encode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	p, err := t.perm(len(a.Shape))
	if err != nil {
		return nil, err
	}
	return permute(a, p), nil
}

func (t *transposeCodec) Decode(_ context.Context, a *NDArray, in ChunkSpec) (*NDArray, error) {
	p, err := t.perm(len(in.Shape))
	if err != nil {
		return nil, err
	}
	inv := make([]int, len(p))
	for i, o := range p {
		inv[o] = i
	}
	out := permute(&NDArray{Shape: a.Shape, DataType: in.DataType, Data: a.Data, Strings: a.Strings}, inv)
	return out, nil
}

// permute returns a transposed so that output dimension i is input
// dimension p[i] (numpy's a.transpose(p)).
func permute(a *NDArray, p []int) *NDArray {
	n := len(a.Shape)
	shape := make([]uint64, n)
	for i, o := range p {
		shape[i] = a.Shape[o]
	}
	out := &NDArray{Shape: shape, DataType: a.DataType}
	total := numElements(shape)
	variable := a.DataType.Variable()
	if variable {
		out.Strings = make([]string, total)
	} else {
		out.Data = make([]byte, len(a.Data))
	}
	inStride := stridesOf(a.Shape)
	// Walking the output in C order, output dim i advances input dim p[i].
	step := make([]uint64, n)
	for i, o := range p {
		step[i] = inStride[o]
	}
	es := uint64(a.DataType.Size)
	idx := make([]uint64, n)
	var src uint64
	for dst := uint64(0); dst < total; dst++ {
		if variable {
			out.Strings[dst] = a.Strings[src]
		} else {
			copy(out.Data[dst*es:(dst+1)*es], a.Data[src*es:(src+1)*es])
		}
		for d := n - 1; d >= 0; d-- {
			idx[d]++
			src += step[d]
			if idx[d] < shape[d] {
				break
			}
			src -= step[d] * idx[d]
			idx[d] = 0
		}
	}
	return out
}

func stridesOf(shape []uint64) []uint64 {
	st := make([]uint64, len(shape))
	acc := uint64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		st[i] = acc
		acc *= shape[i]
	}
	return st
}
