package blosc

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestExport writes Go-compressed frames for every codec numcodecs supports
// into $BLOSC_EXPORT_DIR, to be checked with
//
//	python internal/blosc/testdata/gen.py verify $BLOSC_EXPORT_DIR
//
// which decodes them with numcodecs and compares with the regenerated data.
// It is skipped when the variable is not set.
func TestExport(t *testing.T) {
	dir := os.Getenv("BLOSC_EXPORT_DIR")
	if dir == "" {
		t.Skip("BLOSC_EXPORT_DIR not set")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	specs := []string{"walk32:100000:501", "f64:300000:502", "text:20000:503", "rand:5000:504",
		"mixed:1048583:505", "sparse:777:506", "zero:0:507", "runs:100:508", "walk16:65537:509"}
	n := 0
	for _, spec := range specs {
		src, err := genSpec(spec)
		if err != nil {
			t.Fatal(err)
		}
		for _, cname := range []string{"blosclz", "lz4", "lz4hc", "zlib", "zstd"} {
			for _, clevel := range []int{0, 1, 5, 9} {
				for _, sh := range []int{0, 1, 2} {
					for _, ts := range []int{1, 2, 4, 8, 13, 300} {
						for _, bs := range []int{0, 1000, 65536} {
							if (ts == 13 || ts == 300) && bs != 0 {
								continue
							}
							o := Options{Compressor: cname, Level: clevel, Shuffle: sh, TypeSize: ts, BlockSize: bs}
							out, err := Compress(src, o)
							if err != nil {
								t.Fatal(err)
							}
							name := fmt.Sprintf("%s_%d_%d_%d_%d__%s.blosc", cname, clevel, sh, ts, bs,
								strings.ReplaceAll(spec, ":", "_"))
							if err := os.WriteFile(filepath.Join(dir, name), out, 0o644); err != nil {
								t.Fatal(err)
							}
							n++
						}
					}
				}
			}
		}
	}
	t.Logf("wrote %d frames to %s", n, dir)
}
