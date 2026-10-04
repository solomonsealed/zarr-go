package lz4

import (
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// Deterministic test data, identical to gen() in testdata/gen.py.

type splitmix struct{ s uint64 }

func (r *splitmix) next() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

func (r *splitmix) intn(n int) int { return int(r.next() % uint64(n)) }

func (r *splitmix) bytes(n int) []byte {
	out := make([]byte, 0, n+8)
	for len(out) < n {
		out = binary.LittleEndian.AppendUint64(out, r.next())
	}
	return out[:n]
}

var genWords = []string{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"zarr", "blosc", "chunk", "array", "data", "compress", "shuffle", "\n"}

var genMixedKinds = []string{"zero", "rand", "text", "runs", "walk32", "f64", "periodic"}

func genData(kind string, n int, seed uint64) []byte {
	r := &splitmix{seed}
	if n == 0 {
		return []byte{}
	}
	switch kind {
	case "zero":
		return make([]byte, n)
	case "rand":
		return r.bytes(n)
	case "text":
		var out []byte
		for len(out) < n {
			w := genWords[r.intn(len(genWords))]
			out = append(out, w...)
			if w != "\n" {
				out = append(out, ' ')
			}
		}
		return out[:n]
	case "runs":
		var out []byte
		for len(out) < n {
			v := r.next()
			for range 1 + (v>>8)%64 {
				out = append(out, byte(v))
			}
		}
		return out[:n]
	case "sparse":
		out := make([]byte, n)
		for range n / 100 {
			v := r.next()
			out[v%uint64(n)] = byte(v >> 32)
		}
		return out
	case "periodic":
		p := 1 + r.intn(300)
		pat := r.bytes(p)
		out := make([]byte, 0, n+p)
		for len(out) < n {
			out = append(out, pat...)
		}
		return out[:n]
	case "walk8", "walk16", "walk32", "walk64":
		bitsz, _ := strconv.Atoi(kind[4:])
		size := bitsz / 8
		m := n/size + 1
		v := int64(r.next() & 0xFFFF)
		steps := r.bytes(m)
		out := make([]byte, 0, m*size)
		for _, b := range steps {
			v += int64(b%7) - 3
			out = binary.LittleEndian.AppendUint64(out, uint64(v))[:len(out)+size]
		}
		return out[:n]
	case "f64":
		m := n/8 + 1
		b := r.bytes(m)
		out := make([]byte, 0, m*8)
		for i := range m {
			v := int64(i)*3 + int64(b[i]%4)
			out = binary.LittleEndian.AppendUint64(out, math.Float64bits(float64(v)/8))
		}
		return out[:n]
	case "f32":
		m := n/4 + 1
		b := r.bytes(m)
		out := make([]byte, 0, m*4)
		v := int64(1000)
		for i := range m {
			v += int64(b[i]%9) - 4
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(float32(v)/16))
		}
		return out[:n]
	case "mixed":
		var out []byte
		for len(out) < n {
			k := genMixedKinds[r.intn(len(genMixedKinds))]
			seglen := 1 + r.intn(65536)
			out = append(out, genData(k, seglen, r.next())...)
		}
		return out[:n]
	}
	panic("unknown data kind " + kind)
}

// genSpec parses "kind:n:seed".
func genSpec(spec string) ([]byte, error) {
	parts := strings.Split(spec, ":")
	if len(parts) != 3 {
		return nil, fmt.Errorf("bad spec %q", spec)
	}
	n, err := strconv.Atoi(parts[1])
	if err != nil {
		return nil, err
	}
	seed, err := strconv.ParseUint(parts[2], 10, 64)
	if err != nil {
		return nil, err
	}
	return genData(parts[0], n, seed), nil
}
