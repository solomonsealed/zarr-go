package blosc

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

type goldenVector struct {
	line     string
	opts     Options
	spec     string
	n        int
	size     int
	digest   string
	exact    bool // byte-identity (blosclz, lz4, lz4hc) vs header-only
	implicit bool // numcodecs derived typesize from the array item size
}

func readGolden(t *testing.T) []goldenVector {
	t.Helper()
	f, err := os.Open("testdata/golden.txt.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	var vs []goldenVector
	sc := bufio.NewScanner(zr)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		fl := strings.Split(line, "\t")
		if len(fl) != 8 {
			t.Fatalf("bad golden line %q", line)
		}
		v := goldenVector{line: line, spec: fl[5], digest: fl[7]}
		v.opts.Compressor = fl[0]
		v.opts.Level, _ = strconv.Atoi(fl[1])
		v.opts.Shuffle, _ = strconv.Atoi(fl[2])
		ts := fl[3]
		if strings.HasPrefix(ts, "i") {
			v.implicit = true
			ts = ts[1:]
		}
		v.opts.TypeSize, _ = strconv.Atoi(ts)
		v.opts.BlockSize, _ = strconv.Atoi(fl[4])
		v.size, _ = strconv.Atoi(fl[6])
		v.n, _ = strconv.Atoi(strings.Split(v.spec, ":")[1])
		v.exact = !strings.HasPrefix(v.digest, "hdr:")
		vs = append(vs, v)
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	return vs
}

// TestGolden checks Compress against numcodecs 0.17.0 / c-blosc 1.21.7.dev
// (testdata/gen.py): byte-identical frames for blosclz, lz4 and lz4hc, the
// same header (and therefore block and split layout) for zlib and zstd. Every
// frame must also decode back to its input.
func TestGolden(t *testing.T) {
	vs := readGolden(t)
	var spec string
	var src []byte
	exact, layout, memcpyDiff := 0, 0, 0
	for _, v := range vs {
		if testing.Short() && v.n > 600000 {
			continue
		}
		if v.spec != spec {
			var err error
			if src, err = genSpec(v.spec); err != nil {
				t.Fatal(err)
			}
			spec = v.spec
		}
		out, err := Compress(src, v.opts)
		if err != nil {
			t.Errorf("%s: %v", v.line, err)
			continue
		}
		if v.exact {
			exact++
			if len(out) != v.size {
				t.Errorf("%s: got %d bytes, want %d", v.line, len(out), v.size)
			} else if h := sha256.Sum256(out); hex.EncodeToString(h[:])[:len(v.digest)] != v.digest {
				t.Errorf("%s: same size, different bytes", v.line)
			}
		} else {
			layout++
			want, _ := hex.DecodeString(strings.TrimPrefix(v.digest, "hdr:"))
			got := bytes.Clone(out[:12])
			// The memcpy fallback depends on how well the codec compresses,
			// which legitimately differs between implementations.
			if got[2]&flagMemcpyed != want[2]&flagMemcpyed {
				memcpyDiff++
				t.Logf("%s: memcpy flag differs (got %d bytes, numcodecs %d)", v.line, len(out), v.size)
				got[2] ^= flagMemcpyed
			}
			if !bytes.Equal(got, want) {
				t.Errorf("%s: header %x, want %x", v.line, got, want)
			}
		}
		dec, err := Decompress(out)
		if err != nil {
			t.Errorf("%s: decompress: %v", v.line, err)
		} else if !bytes.Equal(dec, src) {
			t.Errorf("%s: round trip mismatch", v.line)
		}
	}
	t.Logf("%d byte-identical vectors, %d layout vectors (%d with a different memcpy decision)", exact, layout, memcpyDiff)
}

// TestNumcodecsFrames decodes complete frames written by numcodecs: zlib
// and zstd payloads from the C libraries, and multi-threaded frames whose
// blocks are stored out of order.
func TestNumcodecsFrames(t *testing.T) {
	files, err := filepath.Glob("testdata/frames/*.blosc")
	if err != nil || len(files) == 0 {
		t.Fatalf("no frames: %v", err)
	}
	for _, path := range files {
		name := strings.TrimSuffix(filepath.Base(path), ".blosc")
		parts := strings.Split(name, "_")
		spec := strings.Join(parts[len(parts)-3:], ":")
		want, err := genSpec(spec)
		if err != nil {
			t.Fatal(err)
		}
		frame, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		got, err := Decompress(frame)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s: decoded data differs", name)
		}
	}
}
