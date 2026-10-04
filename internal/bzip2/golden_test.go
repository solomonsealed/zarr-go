package bzip2

import (
	"bytes"
	"compress/bzip2"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Golden vectors produced by testdata/gen.py from Python's bz2 module
// (libbzip2 1.0.8). Inputs are regenerated here from the same generator specs.

type genSpec struct {
	Kind   string    `json:"kind"`
	Byte   int       `json:"byte"`
	N      int       `json:"n"`
	Hex    string    `json:"hex"`
	Seed   uint64    `json:"seed"`
	K      uint64    `json:"k"`
	Base   uint64    `json:"base"`
	MaxRun uint64    `json:"maxrun"`
	Scale  float64   `json:"scale"`
	Value  float64   `json:"value"`
	Start  int64     `json:"start"`
	Step   int64     `json:"step"`
	Amp    uint64    `json:"amp"`
	Parts  []genSpec `json:"parts"`
	Unit   *genSpec  `json:"unit"`
	File   string    `json:"file"`
}

type vector struct {
	Name      string  `json:"name"`
	Gen       genSpec `json:"gen"`
	Level     int     `json:"level"`
	InLen     int     `json:"in_len"`
	InSHA256  string  `json:"in_sha256"`
	OutLen    int     `json:"out_len"`
	OutSHA256 string  `json:"out_sha256"`
	OutB64    string  `json:"out_b64"`
}

type vectorFile struct {
	Python   string   `json:"python"`
	LibBzip2 string   `json:"libbzip2"`
	Vectors  []vector `json:"vectors"`
}

func loadVectors(t testing.TB) vectorFile {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var vf vectorFile
	if err := json.Unmarshal(b, &vf); err != nil {
		t.Fatal(err)
	}
	if len(vf.Vectors) == 0 {
		t.Fatal("no vectors")
	}
	return vf
}

// splitMix64 matches SplitMix64 in gen.py.
type splitMix64 struct{ s uint64 }

func (r *splitMix64) next() uint64 {
	r.s += 0x9E3779B97F4A7C15
	z := r.s
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
	z = (z ^ (z >> 27)) * 0x94D049BB133111EB
	return z ^ (z >> 31)
}

var words = strings.Fields(
	"the of and to in is that it for as with was on be by at this are from or " +
		"an have not which but all were they one their has more been can would " +
		"data array chunk zarr compression block sort huffman table stream bytes " +
		"float value index shape dtype codec level store group metadata")

// generate materialises a generator spec; it must match generate() in gen.py.
func generate(t testing.TB, s genSpec) []byte {
	switch s.Kind {
	case "empty":
		return []byte{}
	case "repeat":
		return bytes.Repeat([]byte{byte(s.Byte)}, s.N)
	case "pattern":
		p, err := hex.DecodeString(s.Hex)
		if err != nil {
			t.Fatal(err)
		}
		return bytes.Repeat(p, s.N/len(p)+1)[:s.N]
	case "random":
		r := splitMix64{s.Seed}
		out := make([]byte, (s.N+7)/8*8)
		for i := 0; i < len(out); i += 8 {
			binary.LittleEndian.PutUint64(out[i:], r.next())
		}
		return out[:s.N]
	case "alpha":
		r := splitMix64{s.Seed}
		out := make([]byte, s.N)
		for i := range out {
			out[i] = byte(s.Base + r.next()%s.K)
		}
		return out
	case "geom":
		r := splitMix64{s.Seed}
		out := make([]byte, s.N)
		for i := range out {
			out[i] = byte(s.Base + uint64(bits.TrailingZeros64(r.next())))
		}
		return out
	case "norun":
		r := splitMix64{s.Seed}
		out := make([]byte, s.N)
		prev := -1
		for i := range out {
			var c int
			if prev < 0 {
				c = int(r.next() % 256)
			} else {
				c = int(r.next() % 255)
				if c >= prev {
					c++
				}
			}
			out[i] = byte(c)
			prev = c
		}
		return out
	case "runs":
		r := splitMix64{s.Seed}
		out := make([]byte, 0, s.N+int(s.MaxRun))
		for len(out) < s.N {
			c := byte(s.Base + r.next()%s.K)
			n := 1 + r.next()%s.MaxRun
			for j := uint64(0); j < n; j++ {
				out = append(out, c)
			}
		}
		return out[:s.N]
	case "text":
		r := splitMix64{s.Seed}
		out := make([]byte, 0, s.N+16)
		for len(out) < s.N {
			out = append(out, words[r.next()%uint64(len(words))]...)
			switch r.next() % 16 {
			case 0:
				out = append(out, ".\n"...)
			case 1:
				out = append(out, ", "...)
			default:
				out = append(out, ' ')
			}
		}
		return out[:s.N]
	case "f64ramp":
		out := make([]byte, 8*s.N)
		for i := 0; i < s.N; i++ {
			v := float64(float64(i) * s.Scale)
			binary.LittleEndian.PutUint64(out[8*i:], math.Float64bits(v))
		}
		return out
	case "f64const":
		out := make([]byte, 8*s.N)
		for i := 0; i < s.N; i++ {
			binary.LittleEndian.PutUint64(out[8*i:], math.Float64bits(s.Value))
		}
		return out
	case "f32walk":
		r := splitMix64{s.Seed}
		out := make([]byte, 4*s.N)
		x := int64(0)
		for i := 0; i < s.N; i++ {
			x += int64(r.next()%7) - 3
			v := float32(float32(x) / 8)
			binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(v))
		}
		return out
	case "i32ramp":
		out := make([]byte, 4*s.N)
		for i := 0; i < s.N; i++ {
			binary.LittleEndian.PutUint32(out[4*i:], uint32(int32(s.Start+s.Step*int64(i))))
		}
		return out
	case "u16noise":
		r := splitMix64{s.Seed}
		out := make([]byte, 2*s.N)
		for i := 0; i < s.N; i++ {
			binary.LittleEndian.PutUint16(out[2*i:], uint16(s.Base+r.next()%s.Amp))
		}
		return out
	case "tile":
		u := generate(t, *s.Unit)
		return bytes.Repeat(u, s.N/len(u)+1)[:s.N]
	case "concat":
		var out []byte
		for _, p := range s.Parts {
			out = append(out, generate(t, p)...)
		}
		return out
	case "golden":
		// The input is recovered from the expected output itself.
		comp, err := os.ReadFile(filepath.Join("testdata", s.File))
		if err != nil {
			t.Fatal(err)
		}
		out, err := io.ReadAll(bzip2.NewReader(bytes.NewReader(comp)))
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	t.Fatalf("unknown generator kind %q", s.Kind)
	return nil
}

func sha256Hex(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func decompress(t testing.TB, comp []byte) []byte {
	t.Helper()
	out, err := io.ReadAll(bzip2.NewReader(bytes.NewReader(comp)))
	if err != nil {
		t.Fatalf("compress/bzip2: %v", err)
	}
	return out
}

func TestGolden(t *testing.T) {
	vf := loadVectors(t)
	t.Logf("%d vectors (Python %s, libbzip2 %s)", len(vf.Vectors), vf.Python, vf.LibBzip2)
	for _, v := range vf.Vectors {
		t.Run(v.Name, func(t *testing.T) {
			if testing.Short() && v.InLen > 1<<20 {
				t.Skip("large input in -short mode")
			}
			t.Parallel()
			in := generate(t, v.Gen)
			if len(in) != v.InLen || sha256Hex(in) != v.InSHA256 {
				t.Fatalf("generated input mismatch: len %d (want %d)", len(in), v.InLen)
			}
			got, err := Compress(in, v.Level)
			if err != nil {
				t.Fatal(err)
			}
			if v.OutB64 != "" {
				want, err := base64.StdEncoding.DecodeString(v.OutB64)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("output differs from Python at byte %d:\n got %x\nwant %x", firstDiff(got, want), got, want)
				}
			}
			if v.Gen.Kind == "golden" {
				want, err := os.ReadFile(filepath.Join("testdata", v.Gen.File))
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("output differs from Python at byte %d (got %d bytes, want %d)", firstDiff(got, want), len(got), len(want))
				}
			}
			if len(got) != v.OutLen || sha256Hex(got) != v.OutSHA256 {
				t.Fatalf("output differs from Python: got %d bytes sha256 %s, want %d bytes sha256 %s",
					len(got), sha256Hex(got), v.OutLen, v.OutSHA256)
			}
			// Every output must decode with the standard library.
			if dec := decompress(t, got); !bytes.Equal(dec, in) {
				t.Fatalf("compress/bzip2 round trip mismatch")
			}
		})
	}
}

// TestRoundTrip checks that compress/bzip2 decodes our output for a spread
// of generated inputs at every level (independently of the golden data).
func TestRoundTrip(t *testing.T) {
	r := splitMix64{42}
	kinds := []func(n int) []byte{
		func(n int) []byte { return generate(t, genSpec{Kind: "random", Seed: r.next(), N: n}) },
		func(n int) []byte { return generate(t, genSpec{Kind: "text", Seed: r.next(), N: n}) },
		func(n int) []byte {
			return generate(t, genSpec{Kind: "runs", Seed: r.next(), N: n, K: 1 + r.next()%8, MaxRun: 1 + r.next()%300})
		},
		func(n int) []byte { return generate(t, genSpec{Kind: "pattern", Hex: "616263", N: n}) },
		func(n int) []byte { return generate(t, genSpec{Kind: "f32walk", Seed: r.next(), N: n / 4}) },
	}
	for i := 0; i < 60; i++ {
		n := int(r.next() % 50000)
		if i%10 == 0 {
			n = int(r.next() % 400000)
		}
		in := kinds[i%len(kinds)](n)
		level := 1 + i%9
		got, err := Compress(in, level)
		if err != nil {
			t.Fatal(err)
		}
		if dec := decompress(t, got); !bytes.Equal(dec, in) {
			t.Fatalf("case %d (level %d, %d bytes): round trip mismatch", i, level, len(in))
		}
	}
}

func TestInvalidLevel(t *testing.T) {
	for _, level := range []int{-1, 0, 10, 100} {
		if _, err := Compress([]byte("x"), level); err == nil {
			t.Errorf("level %d: expected error", level)
		}
	}
}

func TestDoesNotModifyInput(t *testing.T) {
	in := generate(t, genSpec{Kind: "text", Seed: 1, N: 50000})
	orig := bytes.Clone(in)
	if _, err := Compress(in, 9); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(in, orig) {
		t.Fatal("Compress modified its input")
	}
}

func FuzzCompress(f *testing.F) {
	f.Add([]byte{}, uint8(9))
	f.Add([]byte("a"), uint8(1))
	f.Add([]byte("aaaa"), uint8(9))
	f.Add(bytes.Repeat([]byte("ab"), 5000), uint8(9))
	f.Add(bytes.Repeat([]byte{0}, 70000), uint8(1))
	f.Add([]byte("hello, world\nhello, world\n"), uint8(5))
	f.Fuzz(func(t *testing.T, in []byte, lv uint8) {
		level := 1 + int(lv)%9
		got, err := Compress(in, level)
		if err != nil {
			t.Fatal(err)
		}
		dec, err := io.ReadAll(bzip2.NewReader(bytes.NewReader(got)))
		if err != nil {
			t.Fatalf("compress/bzip2: %v", err)
		}
		if !bytes.Equal(dec, in) {
			t.Fatal("round trip mismatch")
		}
	})
}

func benchCompress(b *testing.B, in []byte, level int) {
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Compress(in, level); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCompress(b *testing.B) {
	const n = 4 << 20
	inputs := []struct {
		name string
		spec genSpec
	}{
		{"f32walk", genSpec{Kind: "f32walk", Seed: 1, N: n / 4}},
		{"f64ramp", genSpec{Kind: "f64ramp", Scale: 0.001, N: n / 8}},
		{"text", genSpec{Kind: "text", Seed: 1, N: n}},
		{"random", genSpec{Kind: "random", Seed: 1, N: n}},
		{"zeros", genSpec{Kind: "repeat", N: n}},
		{"abab", genSpec{Kind: "pattern", Hex: "6162", N: n}},
	}
	for _, in := range inputs {
		data := generate(b, in.spec)
		b.Run(fmt.Sprintf("%s-4MB-l9", in.name), func(b *testing.B) { benchCompress(b, data, 9) })
	}
}
