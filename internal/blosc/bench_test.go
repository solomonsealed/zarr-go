package blosc

import (
	"fmt"
	"testing"
)

var benchCases = []struct {
	spec     string
	typesize int
}{
	{"f64:8388608:1", 8},
	{"walk32:4194304:2", 4},
	{"text:2097152:3", 1},
}

func BenchmarkCompress(b *testing.B) {
	for _, bc := range benchCases {
		src, _ := genSpec(bc.spec)
		for _, c := range []string{"blosclz", "lz4", "lz4hc", "zlib", "zstd"} {
			for _, sh := range []int{1, 2} {
				o := Options{Compressor: c, Level: 5, Shuffle: sh, TypeSize: bc.typesize}
				b.Run(fmt.Sprintf("%s/%s/shuffle%d", bc.spec, c, sh), func(b *testing.B) {
					b.SetBytes(int64(len(src)))
					for b.Loop() {
						if _, err := Compress(src, o); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

func BenchmarkDecompress(b *testing.B) {
	for _, bc := range benchCases {
		src, _ := genSpec(bc.spec)
		for _, c := range []string{"blosclz", "lz4", "lz4hc", "zlib", "zstd"} {
			for _, sh := range []int{1, 2} {
				frame, _ := Compress(src, Options{Compressor: c, Level: 5, Shuffle: sh, TypeSize: bc.typesize})
				dst := make([]byte, len(src))
				b.Run(fmt.Sprintf("%s/%s/shuffle%d", bc.spec, c, sh), func(b *testing.B) {
					b.SetBytes(int64(len(src)))
					for b.Loop() {
						if _, err := DecompressInto(dst, frame); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		}
	}
}

func BenchmarkShuffle(b *testing.B) {
	for _, ts := range []int{1, 2, 4, 8, 16} {
		src, _ := genSpec("walk32:262144:9")
		dst := make([]byte, len(src))
		back := make([]byte, len(src))
		scratch := make([]byte, len(src))
		b.Run(fmt.Sprintf("byte/%d", ts), func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for b.Loop() {
				shuffle(ts, src, dst)
			}
		})
		b.Run(fmt.Sprintf("unbyte/%d", ts), func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for b.Loop() {
				unshuffle(ts, dst, back)
			}
		})
		b.Run(fmt.Sprintf("bit/%d", ts), func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for b.Loop() {
				bitshuffle(ts, src, dst, scratch)
			}
		})
		b.Run(fmt.Sprintf("unbit/%d", ts), func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for b.Loop() {
				bitunshuffle(ts, dst, back, 2, scratch)
			}
		})
	}
}
