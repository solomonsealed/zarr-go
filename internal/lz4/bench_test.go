package lz4

import (
	"fmt"
	"testing"
)

var benchSpecs = []string{"text:1048579:1", "walk32:1048579:2", "rand:1048579:3", "mixed:262151:4"}

func BenchmarkCompress(b *testing.B) {
	for _, spec := range benchSpecs {
		src, _ := genSpec(spec)
		dst := make([]byte, CompressBound(len(src)))
		b.Run(spec+"/fast1", func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for b.Loop() {
				CompressFastInto(dst, src, 1)
			}
		})
		for _, level := range []int{2, 4, 9, 10, 12} {
			b.Run(fmt.Sprintf("%s/hc%d", spec, level), func(b *testing.B) {
				b.SetBytes(int64(len(src)))
				for b.Loop() {
					CompressHCInto(dst, src, level)
				}
			})
		}
	}
}

func BenchmarkDecompress(b *testing.B) {
	for _, spec := range benchSpecs {
		src, _ := genSpec(spec)
		comp := CompressHC(src, 9)
		dst := make([]byte, len(src))
		b.Run(spec, func(b *testing.B) {
			b.SetBytes(int64(len(src)))
			for b.Loop() {
				if err := Decompress(comp, dst); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
