package lz4

import (
	"bytes"
	"fmt"
	"testing"
)

func roundTripSpecs() []string {
	var specs []string
	seed := 1000
	for _, n := range []int{0, 1, 3, 12, 13, 64, 300, 4096, 65546, 65547, 200001} {
		for _, k := range []string{"zero", "rand", "text", "runs", "sparse", "periodic", "walk16", "f64", "mixed"} {
			specs = append(specs, fmt.Sprintf("%s:%d:%d", k, n, seed))
			seed++
		}
	}
	return specs
}

func TestRoundTrip(t *testing.T) {
	for _, spec := range roundTripSpecs() {
		src, err := genSpec(spec)
		if err != nil {
			t.Fatal(err)
		}
		check := func(name string, comp []byte) {
			t.Helper()
			if len(comp) > CompressBound(len(src)) {
				t.Errorf("%s %s: %d bytes exceeds bound", spec, name, len(comp))
			}
			dec := make([]byte, len(src))
			if err := Decompress(comp, dec); err != nil {
				t.Fatalf("%s %s: %v", spec, name, err)
			}
			if !bytes.Equal(dec, src) {
				t.Fatalf("%s %s: round trip mismatch", spec, name)
			}
		}
		for _, a := range []int{1, 2, 10, 1000} {
			check(fmt.Sprintf("fast%d", a), CompressFast(src, a))
		}
		levels := []int{1, 2, 3, 5, 9, 10, 11, 12}
		if len(src) > 100000 && testing.Short() {
			levels = []int{2, 9}
		}
		for _, l := range levels {
			check(fmt.Sprintf("hc%d", l), CompressHC(src, l))
		}
	}
}

// TestLimitedOutput checks that the *Into variants never write past
// len(dst), and that whenever they succeed the output is identical to the
// unlimited output.
func TestLimitedOutput(t *testing.T) {
	for _, spec := range []string{"text:5000:1", "walk32:70000:2", "rand:3000:3", "runs:20000:4"} {
		src, _ := genSpec(spec)
		full := CompressFast(src, 1)
		fullHC := CompressHC(src, 9)
		for _, c := range []int{0, 1, len(full) - 1, len(full), len(full) + 10, len(src) - 1, len(src)} {
			if c < 0 {
				continue
			}
			buf := make([]byte, c+16)
			for i := range buf {
				buf[i] = 0xAA
			}
			n := CompressFastInto(buf[:c], src, 1)
			if n > 0 && !bytes.Equal(buf[:n], full) {
				t.Errorf("%s cap %d: limited output differs", spec, c)
			}
			for _, b := range buf[c:] {
				if b != 0xAA {
					t.Fatalf("%s cap %d: wrote past capacity", spec, c)
				}
			}
			n = CompressHCInto(buf[:c], src, 9)
			if n > 0 && !bytes.Equal(buf[:n], fullHC) {
				t.Errorf("%s cap %d: limited HC output differs", spec, c)
			}
			for _, b := range buf[c:] {
				if b != 0xAA {
					t.Fatalf("%s cap %d: HC wrote past capacity", spec, c)
				}
			}
		}
	}
}

func TestDecompressCorrupt(t *testing.T) {
	src, _ := genSpec("text:3000:7")
	comp := CompressHC(src, 9)
	dst := make([]byte, len(src))
	// Every truncation must fail cleanly.
	for i := range len(comp) {
		if err := Decompress(comp[:i], dst); err == nil {
			t.Errorf("truncation to %d bytes accepted", i)
		}
	}
	// Wrong output sizes must fail.
	for _, n := range []int{0, 1, len(src) - 1, len(src) + 1} {
		if err := Decompress(comp, make([]byte, n)); err == nil {
			t.Errorf("output size %d accepted", n)
		}
	}
	// Bit flips must not panic.
	for i := range comp {
		for bit := range 8 {
			c := bytes.Clone(comp)
			c[i] ^= 1 << bit
			_ = Decompress(c, dst)
		}
	}
	// Special cases of LZ4_decompress_safe.
	if err := Decompress([]byte{0}, nil); err != nil {
		t.Errorf("empty block: %v", err)
	}
	if err := Decompress(nil, nil); err == nil {
		t.Error("empty input with empty output accepted")
	}
	if err := Decompress(nil, make([]byte, 1)); err == nil {
		t.Error("empty input accepted")
	}
	// A zero offset is rejected.
	bad := []byte{0x10, 'a', 0, 0, 0x50, 'a', 'b', 'c', 'd', 'e'}
	if err := Decompress(bad, make([]byte, 1+4+5)); err == nil {
		t.Error("zero offset accepted")
	}
}

func FuzzDecompress(f *testing.F) {
	for _, spec := range []string{"text:200:1", "runs:500:2", "zero:100:3", "walk32:400:4"} {
		src, _ := genSpec(spec)
		f.Add(CompressFast(src, 1), uint16(len(src)))
		f.Add(CompressHC(src, 9), uint16(len(src)))
		f.Add(CompressHC(src, 12), uint16(len(src)))
	}
	f.Add([]byte{0}, uint16(0))
	f.Add([]byte{0xF0, 0xFF, 0xFF}, uint16(1000))
	f.Fuzz(func(t *testing.T, comp []byte, n uint16) {
		dst := make([]byte, n)
		if err := Decompress(comp, dst); err != nil {
			return
		}
		// Whatever decoded successfully must re-encode and decode to the
		// same bytes.
		re := CompressFast(dst, 1)
		dec := make([]byte, len(dst))
		if err := Decompress(re, dec); err != nil || !bytes.Equal(dec, dst) {
			t.Fatalf("re-encoded output does not round trip: %v", err)
		}
	})
}

func FuzzCompress(f *testing.F) {
	f.Add([]byte("hello hello hello hello hello hello"), uint8(1), uint8(9))
	f.Add(bytes.Repeat([]byte{1, 2, 3, 4}, 100), uint8(3), uint8(12))
	f.Add([]byte{}, uint8(1), uint8(2))
	f.Fuzz(func(t *testing.T, src []byte, accel, level uint8) {
		for _, comp := range [][]byte{CompressFast(src, int(accel)), CompressHC(src, int(level%13))} {
			if len(comp) > CompressBound(len(src)) {
				t.Fatalf("output %d exceeds bound %d", len(comp), CompressBound(len(src)))
			}
			dec := make([]byte, len(src))
			if err := Decompress(comp, dec); err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(dec, src) {
				t.Fatal("round trip mismatch")
			}
		}
	})
}
