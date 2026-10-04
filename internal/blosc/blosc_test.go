package blosc

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"testing"
)

// naiveBitShuffle is the definition of bitshuffle: output row 8*j+k holds
// bit k of byte j of every element, element i at bit i%8 of byte i/8.
func naiveBitShuffle(typesize int, src []byte) []byte {
	n := len(src) / typesize
	dst := make([]byte, len(src))
	rowBytes := n / 8
	for i := range n {
		for j := range typesize {
			v := src[i*typesize+j]
			for k := range 8 {
				if v&(1<<k) != 0 {
					dst[(j*8+k)*rowBytes+i/8] |= 1 << (i % 8)
				}
			}
		}
	}
	copy(dst[n*typesize:], src[n*typesize:])
	return dst
}

func TestShuffle(t *testing.T) {
	for _, ts := range []int{1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 33, 255} {
		for _, n := range []int{0, 1, 7, 8, 64, 100, 1000, 4096 + 5} {
			src, _ := genSpec(fmt.Sprintf("rand:%d:%d", n, ts*1000+n))
			got := make([]byte, n)
			shuffle(ts, src, got)
			want := make([]byte, n)
			q := n / ts
			for i := range q {
				for j := range ts {
					want[j*q+i] = src[i*ts+j]
				}
			}
			copy(want[q*ts:], src[q*ts:])
			if !bytes.Equal(got, want) {
				t.Fatalf("shuffle ts=%d n=%d mismatch", ts, n)
			}
			back := make([]byte, n)
			unshuffle(ts, got, back)
			if !bytes.Equal(back, src) {
				t.Fatalf("unshuffle ts=%d n=%d mismatch", ts, n)
			}

			bs := make([]byte, n)
			bitshuffle(ts, src, bs, make([]byte, n))
			if q%8 == 0 {
				if want := naiveBitShuffle(ts, src); !bytes.Equal(bs, want) {
					t.Fatalf("bitshuffle ts=%d n=%d mismatch", ts, n)
				}
			} else if !bytes.Equal(bs, src) {
				t.Fatalf("bitshuffle ts=%d n=%d: c-blosc 1.x copies blocks of %d elements", ts, n, q)
			}
			back = make([]byte, n)
			bitunshuffle(ts, bs, back, 2, make([]byte, n))
			if !bytes.Equal(back, src) {
				t.Fatalf("bitunshuffle ts=%d n=%d mismatch", ts, n)
			}
		}
	}
}

// TestBitunshuffleV3 checks the c-blosc2 (format version >= 3) layout:
// the largest multiple of 8 elements is transposed, the rest stored as is.
func TestBitunshuffleV3(t *testing.T) {
	for _, ts := range []int{1, 2, 4, 8, 3} {
		src, _ := genSpec(fmt.Sprintf("walk32:%d:%d", 21*ts+2, ts))
		m := (len(src) / ts) &^ 7
		enc := append(naiveBitShuffle(ts, src[:m*ts]), src[m*ts:]...)
		got := make([]byte, len(src))
		bitunshuffle(ts, enc, got, 3, make([]byte, len(src)))
		if !bytes.Equal(got, src) {
			t.Errorf("typesize %d: version 3 bitunshuffle mismatch", ts)
		}
	}
}

func TestComputeBlocksize(t *testing.T) {
	for _, tc := range []struct {
		code, clevel, typesize, nbytes, forced, want int
	}{
		{codeLZ4, 5, 4, 0, 0, 1},
		{codeLZ4, 5, 8, 7, 0, 1},
		{codeLZ4, 5, 4, 1000, 0, 1000},
		{codeLZ4, 5, 4, 1002, 0, 1000},     // multiple of typesize
		{codeLZ4, 5, 4, 100000, 0, 100000}, // split: 128K*4 = 512K, clamped to nbytes
		{codeLZ4, 5, 4, 1 << 22, 0, 512 << 10},
		{codeLZ4, 5, 32, 1 << 22, 0, 128 << 10}, // typesize > 16: no split
		{codeLZ4HC, 9, 32, 1 << 22, 0, 1 << 20},
		{codeZstd, 9, 4, 1 << 22, 0, 1 << 20},
		{codeZstd, 0, 4, 1 << 22, 0, 16 << 10},
		{codeBlosclz, 1, 2, 1 << 22, 0, 64 << 10},
		{codeBlosclz, 1, 4, 1 << 22, 4096, 64 << 10}, // forced blocksizes are enlarged for splits too
		{codeBlosclz, 1, 32, 1 << 22, 4096, 4096},
		{codeBlosclz, 1, 32, 1 << 22, 5, 128},
		{codeBlosclz, 1, 32, 1 << 22, -1, 128},
		{codeBlosclz, 1, 3, 1 << 22, 0, 65535}, // 16K*3 -> 64K, rounded to a multiple of 3
		{codeBlosclz, 9, 16, 1 << 22, 0, 1 << 20},
	} {
		if got := computeBlocksize(tc.code, tc.clevel, tc.typesize, tc.nbytes, tc.forced); got != tc.want {
			t.Errorf("computeBlocksize(%+v) = %d", tc, got)
		}
	}
}

func TestLevelMappings(t *testing.T) {
	want := []int{-1, 1, 3, 5, 7, 9, 11, 13, 20, 22}
	for clevel, w := range want {
		if clevel > 0 && ZstdLevel(clevel) != w {
			t.Errorf("ZstdLevel(%d) = %d, want %d", clevel, ZstdLevel(clevel), w)
		}
	}
	if ZlibLevel(7) != 7 || LZ4Acceleration(9) != 1 || LZ4Acceleration(1) != 9 {
		t.Error("level mappings")
	}
}

func TestOptionErrors(t *testing.T) {
	src := make([]byte, 1000)
	for _, o := range []Options{
		{Compressor: "lzma", Level: 5},
		{Compressor: "", Level: 5},
		{Compressor: "lz4", Level: -1},
		{Compressor: "lz4", Level: 10},
		{Compressor: "lz4", Level: 5, Shuffle: 3},
		{Compressor: "lz4", Level: 5, Shuffle: -2},
		{Compressor: "lz4", Level: 5, TypeSize: -4},
	} {
		if _, err := Compress(src, o); err == nil {
			t.Errorf("%+v accepted", o)
		}
	}
	// TypeSize 0 means 1; AutoShuffle picks bit shuffle for 1-byte types.
	a, _ := Compress(src, Options{Compressor: "lz4", Level: 5, Shuffle: AutoShuffle})
	b, _ := Compress(src, Options{Compressor: "lz4", Level: 5, Shuffle: BitShuffle, TypeSize: 1})
	if !bytes.Equal(a, b) {
		t.Error("TypeSize 0 / AutoShuffle")
	}
	// Type sizes above 255 are treated as 1, but AutoShuffle still uses the
	// item size as numcodecs does.
	a, _ = Compress(src, Options{Compressor: "lz4", Level: 5, Shuffle: AutoShuffle, TypeSize: 300})
	b, _ = Compress(src, Options{Compressor: "lz4", Level: 5, Shuffle: ByteShuffle, TypeSize: 1})
	if !bytes.Equal(a, b) {
		t.Error("TypeSize > 255")
	}
}

func TestEmpty(t *testing.T) {
	for _, c := range []string{"blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"} {
		out, err := Compress(nil, Options{Compressor: c, Level: 5, Shuffle: 1, TypeSize: 4})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != 16 || out[2]&flagMemcpyed == 0 {
			t.Errorf("%s: empty frame %x", c, out)
		}
		dec, err := Decompress(out)
		if err != nil || len(dec) != 0 {
			t.Errorf("%s: decode empty frame: %v %v", c, dec, err)
		}
	}
}

// TestRoundTrip covers all codecs (including snappy, which numcodecs lacks)
// over shuffle modes, type sizes and block sizes.
func TestRoundTrip(t *testing.T) {
	specs := []string{"walk32:100003:1", "f64:65536:2", "text:5000:3", "rand:3000:4",
		"zero:200:5", "sparse:130:6", "mixed:150000:7", "runs:127:8"}
	for _, spec := range specs {
		src, _ := genSpec(spec)
		for _, c := range []string{"blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"} {
			for _, clevel := range []int{0, 1, 4, 9} {
				for _, sh := range []int{0, 1, 2} {
					for _, ts := range []int{1, 2, 4, 8, 11, 16, 17} {
						o := Options{Compressor: c, Level: clevel, Shuffle: sh, TypeSize: ts}
						out, err := Compress(src, o)
						if err != nil {
							t.Fatalf("%s %+v: %v", spec, o, err)
						}
						if len(out) > len(src)+16 {
							t.Fatalf("%s %+v: %d bytes", spec, o, len(out))
						}
						dec, err := Decompress(out)
						if err != nil {
							t.Fatalf("%s %+v: %v", spec, o, err)
						}
						if !bytes.Equal(dec, src) {
							t.Fatalf("%s %+v: round trip mismatch", spec, o)
						}
					}
				}
			}
		}
	}
}

// TestSnappyLayout checks the frame c-blosc would write with snappy: format
// 2 in the flags, split blocks, and a snappy_max_compressed_length output
// budget (a split is only compressed if that much room is left).
func TestSnappyLayout(t *testing.T) {
	src, _ := genSpec("walk32:100000:11")
	out, err := Compress(src, Options{Compressor: "snappy", Level: 5, Shuffle: 1, TypeSize: 4})
	if err != nil {
		t.Fatal(err)
	}
	if out[1] != 1 || out[2] != 2<<5|flagShuffle || out[3] != 4 {
		t.Errorf("header %x", out[:4])
	}
	if bs := binary.LittleEndian.Uint32(out[8:]); bs != 100000 {
		t.Errorf("blocksize %d", bs)
	}
	// One block split in 4 streams.
	off := int(binary.LittleEndian.Uint32(out[16:]))
	if off != 20 {
		t.Fatalf("first block at %d", off)
	}
	compressed := 0
	for range 4 {
		csize := int(binary.LittleEndian.Uint32(out[off:]))
		// Like c-blosc, a snappy split may even be larger than its 25000
		// input bytes as long as it fits snappy_max_compressed_length.
		if csize != 25000 {
			compressed++
		}
		off += 4 + csize
	}
	if compressed == 0 {
		t.Error("no split compressed")
	}
	if off != len(out) {
		t.Errorf("frame ends at %d, have %d", off, len(out))
	}
	// Barely compressible data cannot reserve 32+n+n/6 bytes and is stored.
	rnd, _ := genSpec("rand:4096:12")
	out, _ = Compress(rnd, Options{Compressor: "snappy", Level: 5, Shuffle: 0, TypeSize: 1})
	if out[2]&flagMemcpyed == 0 {
		t.Errorf("random data not stored: flags %x", out[2])
	}
}

func TestSizesAndDecompressInto(t *testing.T) {
	src, _ := genSpec("text:70000:21")
	out, _ := Compress(src, Options{Compressor: "lz4", Level: 5, Shuffle: 1, TypeSize: 2})
	nbytes, cbytes, blocksize, err := Sizes(out)
	if err != nil || nbytes != len(src) || cbytes != len(out) || blocksize != len(src)-len(src)%2 {
		t.Errorf("Sizes = %d %d %d %v", nbytes, cbytes, blocksize, err)
	}
	if _, err := DecompressInto(make([]byte, len(src)-1), out); err == nil {
		t.Error("short destination accepted")
	}
	dst := make([]byte, len(src)+10)
	n, err := DecompressInto(dst, out)
	if err != nil || n != len(src) || !bytes.Equal(dst[:n], src) {
		t.Errorf("DecompressInto: %d %v", n, err)
	}
	// Trailing bytes after the frame are ignored.
	dec, err := Decompress(append(bytes.Clone(out), 1, 2, 3))
	if err != nil || !bytes.Equal(dec, src) {
		t.Errorf("trailing data: %v", err)
	}
}

func corruptionSeeds() [][]byte {
	var seeds [][]byte
	for i, c := range []string{"blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"} {
		for _, spec := range []string{"walk32:3000:31", "text:700:32", "runs:200:33"} {
			src, _ := genSpec(spec)
			for _, sh := range []int{0, 1, 2} {
				out, _ := Compress(src, Options{Compressor: c, Level: 1 + i, Shuffle: sh, TypeSize: 4, BlockSize: 512})
				seeds = append(seeds, out)
			}
		}
	}
	return seeds
}

func TestDecompressCorrupt(t *testing.T) {
	for _, frame := range corruptionSeeds() {
		for i := range len(frame) {
			if _, err := Decompress(frame[:i]); err == nil && i < len(frame) {
				t.Errorf("truncation to %d of %d bytes accepted", i, len(frame))
			}
		}
		for i := range frame {
			for _, x := range []byte{1, 0x80, 0xFF} {
				c := bytes.Clone(frame)
				c[i] ^= x
				_, _ = Decompress(c) // must not panic
			}
		}
	}
	for _, bad := range [][]byte{
		nil,
		make([]byte, 15),
		{2, 1, 0x21, 4, 0x10, 0, 0, 0, 0x10, 0, 0, 0, 0x30, 0, 0, 0},          // truncated block table
		{2, 1, 0x23, 4, 0x10, 0, 0, 0, 0x10, 0, 0, 0, 0x30, 0, 0, 0},          // memcpyed size mismatch
		{2, 1, 0xA1, 4, 0x10, 0, 0, 0, 0x10, 0, 0, 0, 0x10, 0, 0, 0},          // unknown format 5
		{2, 1, 0x29, 4, 0x10, 0, 0, 0, 0x10, 0, 0, 0, 0x30, 0, 0, 0},          // reserved flag
		{2, 1, 0x21, 4, 0x10, 0, 0, 0, 0x20, 0, 0, 0, 0x30, 0, 0, 0},          // blocksize > nbytes
		{2, 1, 0x21, 0, 0x10, 0, 0, 0, 0x10, 0, 0, 0, 0x30, 0, 0, 0},          // typesize 0
		{2, 1, 0x21, 4, 0xFF, 0xFF, 0xFF, 0xFF, 0x10, 0, 0, 0, 0x30, 0, 0, 0}, // negative size
		{5, 1, 0x05, 4, 0x10, 0, 0, 0, 0x10, 0, 0, 0, 0x30, 0, 0, 0},          // c-blosc2 extended header
		// A huge claimed size with a tiny lz4 split.
		{2, 1, 0x30, 1, 0, 0, 0, 0x40, 0, 0, 0, 0x40, 25, 0, 0, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0},
		// Two 128-byte lz4 blocks sharing one split.
		{2, 1, 0x30, 1, 0, 1, 0, 0, 0x80, 0, 0, 0, 0x1D, 0, 0, 0, 24, 0, 0, 0, 24, 0, 0, 0, 1, 0, 0, 0, 0},
	} {
		if _, err := Decompress(bad); err == nil {
			t.Errorf("%x accepted", bad)
		}
	}
}

func FuzzDecompress(f *testing.F) {
	for _, s := range corruptionSeeds() {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, frame []byte) {
		if n, _, _, err := Sizes(frame); err == nil && n > 1<<22 {
			return // keep allocations small; sizes are bounded by parseFrame anyway
		}
		out, err := Decompress(frame)
		if err != nil {
			return
		}
		n, _, _, _ := Sizes(frame)
		if len(out) != n {
			t.Fatalf("decoded %d bytes, header says %d", len(out), n)
		}
	})
}

func FuzzRoundTrip(f *testing.F) {
	f.Add([]byte("hello hello hello hello"), uint8(1), uint8(5), uint8(1), uint8(4), uint16(0))
	f.Add(bytes.Repeat([]byte{0, 1, 2, 3, 4, 5, 6, 7}, 100), uint8(0), uint8(9), uint8(2), uint8(8), uint16(130))
	f.Fuzz(func(t *testing.T, src []byte, codec, level, sh, ts uint8, bs uint16) {
		names := []string{"blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"}
		o := Options{Compressor: names[int(codec)%len(names)], Level: int(level % 10),
			Shuffle: int(sh%4) - 1, TypeSize: int(ts), BlockSize: int(bs)}
		out, err := Compress(src, o)
		if err != nil {
			t.Fatal(err)
		}
		dec, err := Decompress(out)
		if err != nil {
			t.Fatalf("%+v: %v", o, err)
		}
		if !bytes.Equal(dec, src) {
			t.Fatalf("%+v: round trip mismatch", o)
		}
	})
}

// TestConcurrent exercises the pooled codec state from several goroutines
// (run with -race).
func TestConcurrent(t *testing.T) {
	src, _ := genSpec("walk32:200000:41")
	done := make(chan error)
	for g := range 8 {
		go func() {
			var err error
			for i := range 6 {
				c := []string{"blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"}[(g+i)%6]
				o := Options{Compressor: c, Level: 1 + (g+i)%9, Shuffle: i % 3, TypeSize: 4, BlockSize: 16384}
				var out, dec []byte
				if out, err = Compress(src, o); err != nil {
					break
				}
				if dec, err = Decompress(out); err != nil {
					break
				}
				if !bytes.Equal(dec, src) {
					err = fmt.Errorf("%+v: mismatch", o)
					break
				}
			}
			done <- err
		}()
	}
	for range 8 {
		if err := <-done; err != nil {
			t.Error(err)
		}
	}
}
