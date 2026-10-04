package lz4

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestGolden checks byte-identity with the C LZ4 1.10.0 library bundled in
// numcodecs 0.17.0, using the vectors written by testdata/gen.py.
func TestGolden(t *testing.T) {
	f, err := os.Open("testdata/golden.txt.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(zr)
	sc.Buffer(nil, 1<<20)
	cache := map[string][]byte{}
	lines, calls := 0, 0
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		fields := strings.Split(line, "\t")
		if len(fields) != 5 {
			t.Fatalf("bad golden line %q", line)
		}
		fn, spec, param, results, want := fields[0], fields[1], fields[2], fields[3], fields[4]
		if parts := strings.Split(spec, ":"); testing.Short() && len(parts) == 3 {
			if n, _ := strconv.Atoi(parts[1]); n > 70000 {
				continue
			}
		}
		src, ok := cache[spec]
		if !ok {
			if src, err = genSpec(spec); err != nil {
				t.Fatal(err)
			}
			cache = map[string][]byte{spec: src} // keep only the current input
		}
		p, _ := strconv.Atoi(param)
		h := sha256.New()
		var got []string
		for _, item := range strings.Split(results, ",") {
			key, _, _ := strings.Cut(item, "=")
			k, err := strconv.Atoi(key)
			if err != nil {
				t.Fatalf("bad item %q in %q", item, line)
			}
			var out []byte
			switch fn {
			case "fast":
				out = CompressFast(src, k)
			case "hc":
				out = CompressHC(src, k)
			case "fastcap":
				dst := make([]byte, k)
				out = dst[:CompressFastInto(dst, src, p)]
			case "hccap":
				dst := make([]byte, k)
				out = dst[:CompressHCInto(dst, src, p)]
			default:
				t.Fatalf("unknown function %q", fn)
			}
			calls++
			got = append(got, fmt.Sprintf("%s=%d", key, len(out)))
			h.Write(binary.LittleEndian.AppendUint32(nil, uint32(len(out))))
			h.Write(out)
			// Every successful output must decode back to the input.
			if len(out) > 0 {
				dec := make([]byte, len(src))
				if err := Decompress(out, dec); err != nil || string(dec) != string(src) {
					t.Errorf("%s %s %s: output does not round trip: %v", fn, spec, item, err)
				}
			}
		}
		lines++
		if g := strings.Join(got, ","); g != results {
			t.Errorf("%s %s param=%s:\n got  %s\n want %s", fn, spec, param, g, results)
			continue
		}
		if g := hex.EncodeToString(h.Sum(nil))[:len(want)]; g != want {
			t.Errorf("%s %s param=%s: sizes match but bytes differ", fn, spec, param)
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	t.Logf("checked %d golden groups (%d compressor calls)", lines, calls)
}
