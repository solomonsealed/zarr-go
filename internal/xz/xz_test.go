package xz

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"hash/crc32"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	ulzma "github.com/ulikunitz/xz/lzma"
)

func lz2(opts map[string]any) Filter { return Filter{ID: FilterLZMA2, Options: opts} }
func lz1(opts map[string]any) Filter { return Filter{ID: FilterLZMA1, Options: opts} }
func delta(dist int) Filter          { return Filter{ID: FilterDelta, Options: map[string]any{"dist": dist}} }
func bcj(id uint64, off int) Filter {
	return Filter{ID: id, Options: map[string]any{"start_offset": off}}
}
func mustNotPanicErr(t *testing.T, err error) {
	t.Helper()
	var pe *panicError
	if errors.As(err, &pe) {
		t.Fatalf("panic inside the LZMA decoder: %v", err)
	}
}

func testData(n int, seed int64) []byte {
	r := rand.New(rand.NewSource(seed))
	words := []string{"zarr ", "chunk ", "array ", "lzma ", "delta ", "0.125 ", "\n", "float32 ", "shape "}
	var b bytes.Buffer
	for b.Len() < n {
		if r.Intn(10) == 0 {
			b.WriteByte(byte(r.Intn(256)))
		} else {
			b.WriteString(words[r.Intn(len(words))])
		}
	}
	return b.Bytes()[:n]
}

func TestRoundTripMatrix(t *testing.T) {
	inputs := map[string][]byte{
		"empty": {},
		"one":   {42},
		"text":  testData(100000, 1),
		"rand":  func() []byte { b := make([]byte, 70000); rand.New(rand.NewSource(2)).Read(b); return b }(),
		"zeros": make([]byte, 300000),
	}
	type cfg struct {
		name    string
		format  int
		check   int
		preset  int
		filters []Filter
	}
	cfgs := []cfg{
		{"xz", FormatXZ, -1, -1, nil},
		{"xz-none", FormatXZ, CheckNone, 0, nil},
		{"xz-crc32", FormatXZ, CheckCRC32, 1, nil},
		{"xz-sha256", FormatXZ, CheckSHA256, 9 | PresetExtreme, nil},
		{"xz-delta", FormatXZ, -1, -1, []Filter{delta(4), lz2(map[string]any{"preset": 1})}},
		{"xz-x86-delta", FormatXZ, CheckCRC32, -1, []Filter{bcj(FilterX86, 0), delta(2), lz2(nil)}},
		{"xz-arm64", FormatXZ, -1, -1, []Filter{bcj(FilterARM64, 64), lz2(nil)}},
		{"xz-lclppb", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"lc": 0, "lp": 4, "pb": 4, "dict_size": 4096})}},
		{"alone", FormatAlone, -1, -1, nil},
		{"alone-custom", FormatAlone, CheckNone, -1, []Filter{lz1(map[string]any{"lc": 1, "lp": 3, "pb": 0, "dict_size": 5000})}},
		{"raw-lzma2", FormatRaw, -1, -1, []Filter{lz2(nil)}},
		{"raw-lzma1", FormatRaw, -1, -1, []Filter{lz1(map[string]any{"preset": 2})}},
		{"raw-sparc-lzma1", FormatRaw, -1, -1, []Filter{bcj(FilterSPARC, 8), lz1(nil)}},
		{"raw-delta-lzma2", FormatRaw, -1, -1, []Filter{delta(256), lz2(map[string]any{"dict_size": 1 << 16})}},
		{"raw-lzma2-dict4k", FormatRaw, -1, -1, []Filter{lz2(map[string]any{"dict_size": 4096})}},
		{"raw-lzma2-dict5000", FormatRaw, -1, -1, []Filter{lz2(map[string]any{"dict_size": 5000, "lc": 4})}},
	}
	for _, c := range cfgs {
		for name, in := range inputs {
			t.Run(c.name+"/"+name, func(t *testing.T) {
				comp, err := Compress(in, c.format, c.check, c.preset, c.filters)
				if err != nil {
					t.Fatal(err)
				}
				var df []Filter
				if c.format == FormatRaw {
					df = c.filters
				}
				got, err := Decompress(comp, c.format, df)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, in) {
					t.Fatalf("mismatch at %d", firstDiff(got, in))
				}
				if c.format != FormatRaw {
					if got, err := Decompress(comp, FormatAuto, nil); err != nil || !bytes.Equal(got, in) {
						t.Fatalf("FormatAuto: %v", err)
					}
				}
			})
		}
	}
}

// A >2 MiB input forces several LZMA2 chunks (each at most 2 MiB
// uncompressed / 64 KiB compressed).
func TestRoundTripLarge(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	in := testData(5<<20, 7)
	comp, err := Compress(in, FormatXZ, -1, 6, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := Decompress(comp, FormatXZ, nil)
	if err != nil || !bytes.Equal(got, in) {
		t.Fatalf("err=%v equal=%v", err, bytes.Equal(got, in))
	}
	t.Logf("%d -> %d bytes", len(in), len(comp))
}

func TestXZLayout(t *testing.T) {
	in := testData(5000, 3)
	comp, err := Compress(in, FormatXZ, -1, -1, []Filter{delta(4), lz2(nil)})
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(comp[:6], streamMagic) || comp[7] != CheckCRC64 {
		t.Fatalf("bad stream header % x", comp[:12])
	}
	// Block header: both sizes present, two filters.
	if flags := comp[13]; flags != 0xC1 {
		t.Fatalf("block flags 0x%02x, want 0xc1", flags)
	}
	// LZMA2 dictionary declared in the header is shrunk to the input size.
	hsize := (int(comp[12]) + 1) * 4
	hdr := comp[12 : 12+hsize]
	i := bytes.Index(hdr, []byte{0x21, 0x01})
	if i < 0 {
		t.Fatal("no LZMA2 filter flags")
	}
	if d, _ := lzma2DictSize(hdr[i+2]); d != 6144 { // 5000 rounded up to 2^n or 3*2^(n-1)
		t.Fatalf("declared dictionary %d, want 6144", d)
	}
	// Empty input: no blocks, 32 bytes like liblzma.
	empty, _ := Compress(nil, FormatXZ, -1, -1, nil)
	py := "fd377a585a000004e6d6b446000000001cdf44211fb6f37d010000000004595a"
	if got := hexString(empty); got != py {
		t.Fatalf("empty .xz = %s, want %s", got, py)
	}
}

func hexString(b []byte) string {
	const digits = "0123456789abcdef"
	out := make([]byte, 0, 2*len(b))
	for _, c := range b {
		out = append(out, digits[c>>4], digits[c&15])
	}
	return string(out)
}

func TestCompressArgErrors(t *testing.T) {
	in := []byte("hello")
	for _, c := range []struct {
		name    string
		format  int
		check   int
		preset  int
		filters []Filter
		want    error
	}{
		{"format auto", FormatAuto, -1, -1, nil, ErrUnsupported},
		{"format 7", 7, -1, -1, nil, ErrUnsupported},
		{"check with alone", FormatAlone, CheckCRC32, -1, nil, ErrUnsupported},
		{"check with raw", FormatRaw, CheckCRC64, -1, []Filter{lz2(nil)}, ErrUnsupported},
		{"preset and filters", FormatXZ, -1, 3, []Filter{lz2(nil)}, ErrUnsupported},
		{"preset 10", FormatXZ, -1, 10, nil, ErrUnsupported},
		{"preset flag", FormatXZ, -1, 6 | 0x100, nil, ErrUnsupported},
		{"preset negative", FormatXZ, -1, -2, nil, ErrUnsupported},
		{"check 2", FormatXZ, 2, -1, nil, ErrUnsupported},
		{"check 16", FormatXZ, 16, -1, nil, ErrUnsupported},
		{"raw without filters", FormatRaw, -1, -1, nil, ErrUnsupported},
		{"raw with preset only", FormatRaw, -1, 6, nil, ErrUnsupported},
		{"xz with lzma1", FormatXZ, -1, -1, []Filter{lz1(nil)}, ErrUnsupported},
		{"xz lzma2 not last", FormatXZ, -1, -1, []Filter{lz2(nil), delta(1)}, ErrUnsupported},
		{"xz no lzma", FormatXZ, -1, -1, []Filter{delta(1)}, ErrUnsupported},
		{"too many filters", FormatXZ, -1, -1, []Filter{delta(1), delta(1), delta(1), delta(1), lz2(nil)}, ErrUnsupported},
		{"alone with lzma2", FormatAlone, -1, -1, []Filter{lz2(nil)}, ErrUnsupported},
		{"alone with delta", FormatAlone, -1, -1, []Filter{delta(1), lz1(nil)}, ErrUnsupported},
		{"dict too small", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"dict_size": 4000})}, ErrUnsupported},
		{"dict too large", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"dict_size": 1<<31 + 0})}, ErrUnsupported},
		{"lc+lp", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"lc": 4, "lp": 1})}, ErrUnsupported},
		{"pb", FormatRaw, -1, -1, []Filter{lz1(map[string]any{"pb": 5})}, ErrUnsupported},
		{"bad mode", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"mode": 3})}, ErrUnsupported},
		{"bad mf", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"mf": 5})}, ErrUnsupported},
		{"bad nice_len", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"nice_len": 300})}, ErrUnsupported},
		{"unknown key", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"foo": 1})}, ErrUnsupported},
		{"delta start_offset", FormatXZ, -1, -1, []Filter{{ID: FilterDelta, Options: map[string]any{"start_offset": 4}}, lz2(nil)}, ErrUnsupported},
		{"delta dist 0", FormatXZ, -1, -1, []Filter{delta(0), lz2(nil)}, ErrUnsupported},
		{"delta dist 257", FormatXZ, -1, -1, []Filter{delta(257), lz2(nil)}, ErrUnsupported},
		{"negative option", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"lc": -1})}, ErrUnsupported},
		{"fractional option", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"lc": 1.5})}, ErrUnsupported},
		{"string option", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"lc": "1"})}, ErrUnsupported},
		{"unknown filter", FormatXZ, -1, -1, []Filter{{ID: 0x77}, lz2(nil)}, ErrUnsupported},
		{"riscv", FormatXZ, -1, -1, []Filter{{ID: FilterRISCV}, lz2(nil)}, ErrUnsupported},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := Compress(in, c.format, c.check, c.preset, c.filters)
			if !errors.Is(err, c.want) {
				t.Fatalf("err = %v, want %v", err, c.want)
			}
		})
	}
}

func TestDecompressArgs(t *testing.T) {
	xzData, _ := Compress([]byte("hello"), FormatXZ, -1, -1, nil)
	if _, err := Decompress(xzData, 5, nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("format 5: %v", err)
	}
	if _, err := Decompress(xzData, FormatRaw, nil); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("raw without filters: %v", err)
	}
	if _, err := Decompress(xzData, FormatRaw, []Filter{delta(1)}); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("raw chain without LZMA: %v", err)
	}
	// Filters are ignored for self-describing formats (Python raises; see
	// the Decompress doc comment).
	got, err := Decompress(xzData, FormatXZ, []Filter{delta(4), lz2(map[string]any{"preset": 1})})
	if err != nil || string(got) != "hello" {
		t.Fatalf("xz with filters: %q %v", got, err)
	}
	if _, err := Decompress(nil, FormatXZ, nil); !errors.Is(err, ErrTruncated) {
		t.Fatalf("empty input: %v", err)
	}
	if _, err := Decompress([]byte("not xz at all, really"), FormatXZ, nil); !errors.Is(err, ErrFormat) {
		t.Fatalf("not xz: %v", err)
	}
	if _, err := Decompress(xzData, FormatRaw, []Filter{lz2(map[string]any{"dict_size": 1 << 31})}); !errors.Is(err, ErrMemLimit) {
		t.Fatalf("raw dict 2 GiB: %v", err)
	}
}

func TestPresets(t *testing.T) {
	want := []uint32{1 << 18, 1 << 20, 1 << 21, 1 << 22, 1 << 22, 1 << 23, 1 << 23, 1 << 24, 1 << 25, 1 << 26}
	for level, d := range want {
		for _, p := range []uint32{uint32(level), uint32(level) | PresetExtreme} {
			o, err := presetOptions(p)
			if err != nil || o.dictSize != d || o.lc != 3 || o.lp != 0 || o.pb != 2 {
				t.Fatalf("preset %#x: %+v %v", p, o, err)
			}
		}
	}
	if _, err := presetOptions(10); err == nil {
		t.Fatal("preset 10 accepted")
	}
}

func TestFilterFromMap(t *testing.T) {
	// As encoding/json decodes numcodecs metadata (float64), including the
	// LZMA1 ID that float64 cannot represent exactly.
	var specs []map[string]any
	js := `[{"id": 3, "dist": 4}, {"id": 4611686018427387905, "preset": 1, "lc": 0}]`
	if err := json.Unmarshal([]byte(js), &specs); err != nil {
		t.Fatal(err)
	}
	f0, err := FilterFromMap(specs[0])
	if err != nil || f0.ID != FilterDelta {
		t.Fatalf("%+v %v", f0, err)
	}
	f1, err := FilterFromMap(specs[1])
	if err != nil || f1.ID != FilterLZMA1 {
		t.Fatalf("%+v %v", f1, err)
	}
	comp, err := Compress([]byte("abcabcabc"), FormatRaw, -1, -1, []Filter{f0, f1})
	if err != nil {
		t.Fatal(err)
	}
	if got, err := Decompress(comp, FormatRaw, []Filter{f0, f1}); err != nil || string(got) != "abcabcabc" {
		t.Fatalf("%q %v", got, err)
	}
	// json.Number path.
	dec := json.NewDecoder(bytes.NewReader([]byte(js)))
	dec.UseNumber()
	specs = nil
	if err := dec.Decode(&specs); err != nil {
		t.Fatal(err)
	}
	if f, err := FilterFromMap(specs[1]); err != nil || f.ID != FilterLZMA1 {
		t.Fatalf("%+v %v", f, err)
	}
	if _, err := FilterFromMap(map[string]any{"dist": 1}); err == nil {
		t.Fatal("missing id accepted")
	}
	m := f0.Map()
	if m["id"] != FilterDelta || m["dist"] != float64(4) {
		t.Fatalf("Map() = %v", m)
	}
}

// heapGrowth runs f and returns the bytes allocated during it.
func heapGrowth(f func()) uint64 {
	var a, b runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&a)
	f()
	runtime.ReadMemStats(&b)
	return b.TotalAlloc - a.TotalAlloc
}

// Small inputs that declare huge dictionaries must not allocate them.
func TestMemoryBound(t *testing.T) {
	in := []byte("tiny payload, tiny payload, tiny payload")

	// .lzma with a 1 GiB dictionary in the header.
	alone, err := Compress(in, FormatAlone, -1, -1, nil)
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(alone[1:5], 1<<30)
	var got []byte
	if n := heapGrowth(func() { got, err = Decompress(alone, FormatAlone, nil) }); n > 16<<20 {
		t.Errorf(".lzma with 1 GiB dictionary allocated %d bytes", n)
	}
	if err != nil || !bytes.Equal(got, in) {
		t.Fatalf("%v", err)
	}
	// Dictionaries above MaxDictSize are rejected.
	binary.LittleEndian.PutUint32(alone[1:5], 0xFFFFFFFF)
	if _, err := Decompress(alone, FormatAlone, nil); !errors.Is(err, ErrMemLimit) {
		t.Fatalf("4 GiB .lzma dictionary: %v", err)
	}

	// .xz whose LZMA2 property says 1.5 GiB, with no sizes in the header.
	x := buildXZ(t, in, 37, false)
	if n := heapGrowth(func() { got, err = Decompress(x, FormatXZ, nil) }); n > 16<<20 {
		t.Errorf(".xz with 1.5 GiB dictionary allocated %d bytes", n)
	}
	if err != nil || !bytes.Equal(got, in) {
		t.Fatalf("%v", err)
	}
	if _, err := Decompress(buildXZ(t, in, 40, false), FormatXZ, nil); !errors.Is(err, ErrMemLimit) {
		t.Fatalf("4 GiB LZMA2 dictionary: %v", err)
	}

	// Raw LZMA2 with preset 9 (64 MiB dictionary) on a small input.
	raw, _ := Compress(in, FormatRaw, -1, -1, []Filter{lz2(map[string]any{"preset": 9})})
	if n := heapGrowth(func() { got, err = Decompress(raw, FormatRaw, []Filter{lz2(map[string]any{"preset": 9})}) }); n > 16<<20 {
		t.Errorf("raw preset 9 allocated %d bytes", n)
	}
	if err != nil || !bytes.Equal(got, in) {
		t.Fatalf("%v", err)
	}
	// Compressing a small input with preset 9 must not allocate the 64 MiB
	// dictionary either.
	if n := heapGrowth(func() { _, err = Compress(in, FormatXZ, -1, 9|PresetExtreme, nil) }); n > 16<<20 || err != nil {
		t.Errorf("compress preset 9 allocated %d bytes (%v)", n, err)
	}
}

// TestFarMatchRetry: without a known size, decoding first tries a
// dictionary of max(64 KiB, 16*len(input)) and must transparently retry with
// the full one when a match reaches further back. Also checks the .xz index
// read-ahead (sizes absent from block headers) with the same data.
func TestFarMatchRetry(t *testing.T) {
	x := make([]byte, 100<<10)
	rand.New(rand.NewSource(5)).Read(x)
	in := append(append(append([]byte{}, x...), make([]byte, 4<<20)...), x...)
	for _, c := range []struct {
		format  int
		filters []Filter
	}{
		{FormatRaw, []Filter{lz2(nil)}},
		{FormatRaw, []Filter{lz1(nil)}},
		{FormatAlone, nil},
		{FormatXZ, nil},
	} {
		comp, err := Compress(in, c.format, -1, -1, c.filters)
		if err != nil {
			t.Fatal(err)
		}
		if first, _ := dictCaps(8<<20, unknownSizes, len(comp)); first >= len(in)-len(x) {
			t.Fatalf("test data does not force a retry: first cap %d, input %d", first, len(comp))
		}
		got, err := Decompress(comp, c.format, c.filters)
		if err != nil || !bytes.Equal(got, in) {
			t.Fatalf("format %d: %v", c.format, err)
		}
	}
	// The same as a liblzma-style .xz (no sizes in the block header): the
	// index read-ahead provides the exact block size.
	comp, _ := Compress(in, FormatXZ, CheckCRC32, -1, nil)
	stripped := stripBlockSizes(t, comp)
	if hints := xzIndexHints(stripped); len(hints[0]) != 1 || hints[0][0].uncompressed != uint64(len(in)) {
		t.Fatalf("index hints = %v", hints)
	}
	got, err := Decompress(stripped, FormatXZ, nil)
	if err != nil || !bytes.Equal(got, in) {
		t.Fatalf("stripped: %v", err)
	}
}

// stripBlockSizes rewrites a single-block .xz stream made by Compress
// (CRC32 check, LZMA2 only) without sizes in the block header, as liblzma
// writes it.
func stripBlockSizes(t *testing.T, comp []byte) []byte {
	t.Helper()
	hsize := (int(comp[12]) + 1) * 4
	csize, n1, _ := readVLI(comp[14:])
	usize, n2, _ := readVLI(comp[14+n1:])
	filt := comp[14+n1+n2 : 14+n1+n2+3]
	payload := comp[12+hsize : 12+hsize+int(csize)]
	sum := comp[12+hsize+int(csize)+(4-int(csize)%4)%4:][:4]
	h := append([]byte{0, 0}, filt...)
	for (len(h)+4)%4 != 0 {
		h = append(h, 0)
	}
	h[0] = byte((len(h)+4)/4 - 1)
	h = binary.LittleEndian.AppendUint32(h, crc32.ChecksumIEEE(h))
	out := append(append([]byte{}, comp[:12]...), h...)
	out = append(out, payload...)
	for len(out)%4 != 0 {
		out = append(out, 0)
	}
	out = append(out, sum...)
	idx := appendVLI(appendVLI(appendVLI([]byte{0}, 1), uint64(len(h)+len(payload)+4)), usize)
	for len(idx)%4 != 0 {
		idx = append(idx, 0)
	}
	idx = binary.LittleEndian.AppendUint32(idx, crc32.ChecksumIEEE(idx))
	out = append(out, idx...)
	var footer [12]byte
	binary.LittleEndian.PutUint32(footer[4:8], uint32(len(idx)/4-1))
	footer[9] = comp[7]
	binary.LittleEndian.PutUint32(footer[0:4], crc32.ChecksumIEEE(footer[4:10]))
	copy(footer[10:], "YZ")
	return append(out, footer[:]...)
}

// buildXZ makes a single-block .xz stream with the given LZMA2 dictionary
// code; sizes controls whether the block header records the sizes.
func buildXZ(t *testing.T, in []byte, dictCode byte, sizes bool) []byte {
	t.Helper()
	comp, err := Compress(in, FormatXZ, CheckCRC32, -1, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Re-encode the block header with the requested properties.
	hsize := (int(comp[12]) + 1) * 4
	payloadStart := 12 + hsize
	csize, n1, _ := readVLI(comp[14:])
	usize, _, _ := readVLI(comp[14+n1:])
	payload := comp[payloadStart : payloadStart+int(csize)]
	h := []byte{0, 0}
	if sizes {
		h[1] = 0xC0
		h = appendVLI(h, csize)
		h = appendVLI(h, usize)
	}
	h = append(h, 0x21, 1, dictCode)
	for (len(h)+4)%4 != 0 {
		h = append(h, 0)
	}
	h[0] = byte((len(h)+4)/4 - 1)
	h = binary.LittleEndian.AppendUint32(h, crc32.ChecksumIEEE(h))
	out := append([]byte{}, comp[:12]...)
	out = append(out, h...)
	out = append(out, payload...)
	for i := len(payload); i%4 != 0; i++ {
		out = append(out, 0)
	}
	out = binary.LittleEndian.AppendUint32(out, crc32.ChecksumIEEE(in))
	idx := []byte{0}
	idx = appendVLI(idx, 1)
	idx = appendVLI(idx, uint64(len(h)+len(payload)+4))
	idx = appendVLI(idx, usize)
	for len(idx)%4 != 0 {
		idx = append(idx, 0)
	}
	idx = binary.LittleEndian.AppendUint32(idx, crc32.ChecksumIEEE(idx))
	out = append(out, idx...)
	var footer [12]byte
	binary.LittleEndian.PutUint32(footer[4:8], uint32(len(idx)/4-1))
	footer[9] = CheckCRC32
	binary.LittleEndian.PutUint32(footer[0:4], crc32.ChecksumIEEE(footer[4:10]))
	copy(footer[10:], "YZ")
	return append(out, footer[:]...)
}

func TestAutoAfterXZ(t *testing.T) {
	// FormatAuto: a .lzma stream whose first four bytes are zero (lc=lp=pb=0,
	// 16 MiB dictionary) following an .xz stream is not stream padding.
	a := testData(3000, 21)
	b := testData(2000, 22)
	xa, _ := Compress(a, FormatXZ, -1, -1, nil)
	lb, err := Compress(b, FormatAlone, -1, -1, []Filter{lz1(map[string]any{"lc": 0, "lp": 0, "pb": 0, "dict_size": 1 << 24})})
	if err != nil {
		t.Fatal(err)
	}
	binary.LittleEndian.PutUint32(lb[1:5], 1<<24) // as liblzma would write it
	if !bytes.Equal(lb[:4], []byte{0, 0, 0, 0}) {
		t.Fatalf(".lzma header % x", lb[:5])
	}
	got, err := Decompress(append(append([]byte{}, xa...), lb...), FormatAuto, nil)
	if err != nil || !bytes.Equal(got, append(append([]byte{}, a...), b...)) {
		t.Fatalf("err=%v len=%d", err, len(got))
	}
	// Stream padding before another .xz stream, and trailing zeros.
	xb, _ := Compress(b, FormatXZ, CheckSHA256, -1, nil)
	in := append(append(append(append([]byte{}, xa...), 0, 0, 0, 0), xb...), 0, 0, 0)
	got, err = Decompress(in, FormatXZ, nil)
	if err != nil || !bytes.Equal(got, append(append([]byte{}, a...), b...)) {
		t.Fatalf("padding: err=%v len=%d", err, len(got))
	}
	// Padding that is not a multiple of four: like Python, the rest is not
	// a valid stream and is ignored.
	in = append(append(append([]byte{}, xa...), 0, 0, 0), xb...)
	if got, err = Decompress(in, FormatXZ, nil); err != nil || !bytes.Equal(got, a) {
		t.Fatalf("bad padding: err=%v len=%d", err, len(got))
	}
	if !pickyDictSize(0) || !pickyDictSize(1<<24) || !pickyDictSize(3<<20) || pickyDictSize(5000) {
		t.Fatal("pickyDictSize")
	}
}

// .lzma files with the size in the header and no end marker (as written by
// e.g. 7-Zip) decode; a wrong size is an error.
func TestAloneKnownSizeNoEOPM(t *testing.T) {
	for _, n := range []int{1, 1000, 300000} {
		in := testData(n, int64(n))
		comp := knownSizeAlone(t, in)
		got, err := Decompress(comp, FormatAlone, nil)
		if err != nil || !bytes.Equal(got, in) {
			t.Fatalf("n=%d: %v", n, err)
		}
		if got, err := Decompress(comp, FormatAuto, nil); err != nil || !bytes.Equal(got, in) {
			t.Fatalf("n=%d auto: %v", n, err)
		}
		binary.LittleEndian.PutUint64(comp[5:13], uint64(n+1))
		if _, err := Decompress(comp, FormatAlone, nil); err == nil {
			t.Fatalf("n=%d: wrong size accepted", n)
		}
		binary.LittleEndian.PutUint64(comp[5:13], uint64(n-1))
		if _, err := Decompress(comp, FormatAlone, nil); err == nil && n > 1 {
			t.Fatalf("n=%d: short size accepted", n)
		}
	}
}

func knownSizeAlone(t testing.TB, in []byte) []byte {
	var buf bytes.Buffer
	w, err := ulzma.WriterConfig{DictCap: 1 << 20, SizeInHeader: true, Size: int64(len(in))}.NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	w.Write(in)
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestDecompressNoAlias(t *testing.T) {
	// Decompress must not modify its input, and Compress must not modify
	// the data (filters run on a copy).
	in := testData(20000, 9)
	orig := append([]byte(nil), in...)
	comp, err := Compress(in, FormatXZ, -1, -1, []Filter{bcj(FilterX86, 0), delta(3), lz2(nil)})
	if err != nil || !bytes.Equal(in, orig) {
		t.Fatalf("Compress modified its input (%v)", err)
	}
	compOrig := append([]byte(nil), comp...)
	if _, err := Decompress(comp, FormatXZ, nil); err != nil || !bytes.Equal(comp, compOrig) {
		t.Fatalf("Decompress modified its input (%v)", err)
	}
}

// Every truncation of every golden vector must fail cleanly (or, for
// prefixes ending exactly at a stream boundary, succeed), never panic.
func TestTruncationsAndBitFlips(t *testing.T) {
	g := loadGolden(t)
	r := rand.New(rand.NewSource(1))
	for _, c := range g.Cases {
		data := g.data(t, c)
		filters := toFilters(t, c.Filters)
		step := 1
		if len(data) > 300 {
			step = len(data) / 150
		}
		for i := 0; i < len(data); i += step {
			_, err := Decompress(data[:i], c.Format, filters)
			mustNotPanicErr(t, err)
		}
		for k := 0; k < 100 && len(data) > 0; k++ {
			b := append([]byte(nil), data...)
			b[r.Intn(len(b))] ^= byte(1 << r.Intn(8))
			_, err := Decompress(b, c.Format, filters)
			mustNotPanicErr(t, err)
			_, err = Decompress(b, FormatAuto, nil)
			mustNotPanicErr(t, err)
		}
	}
}

var fuzzRawChains = [][]Filter{
	{lz2(nil)},
	{lz1(nil)},
	{lz1(map[string]any{"lc": 0, "lp": 4, "pb": 4})},
	{delta(4), lz2(nil)},
	{bcj(FilterX86, 0), lz2(nil)},
	{bcj(FilterIA64, 16), delta(2), lz1(nil)},
}

func FuzzDecompress(f *testing.F) {
	vecs, _ := filepath.Glob(filepath.Join("testdata", "vectors", "*.bin"))
	for _, p := range vecs {
		b, err := os.ReadFile(p)
		if err == nil {
			f.Add(b, uint8(0))
			f.Add(b, uint8(1))
		}
	}
	f.Add([]byte{}, uint8(2))
	f.Fuzz(func(t *testing.T, data []byte, sel uint8) {
		format := int(sel % 4)
		var filters []Filter
		if format == FormatRaw {
			filters = fuzzRawChains[int(sel/4)%len(fuzzRawChains)]
		}
		out, err := Decompress(data, format, filters)
		var pe *panicError
		if errors.As(err, &pe) {
			t.Fatalf("panic inside the LZMA decoder: %v", err)
		}
		if err == nil && format == FormatXZ && len(out) > 0 {
			// Valid input: re-encoding round-trips.
			comp, err := Compress(out, FormatXZ, -1, 0, nil)
			if err != nil {
				t.Fatal(err)
			}
			back, err := Decompress(comp, FormatXZ, nil)
			if err != nil || !bytes.Equal(back, out) {
				t.Fatalf("re-encode round trip failed: %v", err)
			}
		}
	})
}

// benchInput is a 1 MiB float32 array (a noisy sine), a typical zarr chunk.
func benchInput() []byte {
	r := rand.New(rand.NewSource(1))
	b := make([]byte, 4*262144)
	for i := 0; i < 262144; i++ {
		v := float32(math.Round(1000*math.Sin(float64(i)/500)+r.NormFloat64()*3) / 8)
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
	return b
}

// BenchmarkDecompressXZ decodes a 1 MiB chunk laid out as liblzma writes it
// (no sizes in the block header; preset 6, 8 MiB dictionary declared).
func BenchmarkDecompressXZ(b *testing.B) {
	in := benchInput()
	comp, _ := Compress(in, FormatXZ, CheckCRC32, -1, nil)
	comp = stripBlockSizes(&testing.T{}, comp)
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := Decompress(comp, FormatXZ, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCompressXZ(b *testing.B) {
	in := benchInput()
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := Compress(in, FormatXZ, -1, -1, nil); err != nil {
			b.Fatal(err)
		}
	}
}
