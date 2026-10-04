package xz

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"testing"
)

// Live cross-checks against Python's lzma module (liblzma) and numcodecs.
// They run only when ZARR_PYTHON names a Python interpreter, e.g.
//
//	ZARR_PYTHON=$PWD/.venv/bin/python go test ./internal/xz/ -run Python -v
//
// The numcodecs checks additionally need numcodecs importable.

const pyHelper = `
import sys, json, lzma, base64
reqs = json.load(sys.stdin)
res = []
for r in reqs:
    try:
        data = base64.b64decode(r["data"])
        op = r["op"]
        if op == "decompress":
            out = lzma.decompress(data, format=r["format"], filters=r.get("filters"))
        elif op == "compress":
            out = lzma.compress(data, format=r["format"], check=r["check"],
                                preset=r.get("preset"), filters=r.get("filters"))
        elif op == "numcodecs_decode":
            from numcodecs import LZMA
            codec = LZMA(format=r["format"], check=r["check"], preset=r.get("preset"), filters=r.get("filters"))
            out = bytes(codec.decode(data))
        else:
            raise ValueError("bad op " + op)
        res.append({"ok": True, "data": base64.b64encode(out).decode()})
    except Exception as e:
        res.append({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
json.dump(res, sys.stdout)
`

type pyReq struct {
	Op      string           `json:"op"`
	Data    string           `json:"data"`
	Format  int              `json:"format"`
	Check   int              `json:"check"`
	Preset  *int             `json:"preset"`
	Filters []map[string]any `json:"filters"`
}

type pyRes struct {
	OK    bool   `json:"ok"`
	Data  string `json:"data"`
	Error string `json:"error"`
}

func pythonPath(t *testing.T) string {
	p := os.Getenv("ZARR_PYTHON")
	if p == "" {
		t.Skip("set ZARR_PYTHON to a Python interpreter to run the live liblzma cross-checks")
	}
	return p
}

func runPython(t *testing.T, py string, reqs []pyReq) []pyRes {
	t.Helper()
	in, err := json.Marshal(reqs)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(py, "-c", pyHelper)
	cmd.Dir = t.TempDir() // never run from inside site-packages
	cmd.Stdin = bytes.NewReader(in)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("python: %v\n%s", err, stderr.String())
	}
	var res []pyRes
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("python output: %v", err)
	}
	if len(res) != len(reqs) {
		t.Fatalf("python returned %d results for %d requests", len(res), len(reqs))
	}
	return res
}

func filterMaps(fs []Filter) []map[string]any {
	if fs == nil {
		return nil
	}
	ms := make([]map[string]any, len(fs))
	for i, f := range fs {
		ms[i] = f.Map()
	}
	return ms
}

func intPtr(i int) *int { return &i }

// liveInputs are the inputs for the live checks: text, numeric arrays,
// incompressible data, BCJ-heavy data and a >2 MiB input (several LZMA2
// chunks).
func liveInputs() map[string][]byte {
	r := rand.New(rand.NewSource(11))
	f32 := make([]byte, 4*200000)
	for i := 0; i < 200000; i++ {
		binary.LittleEndian.PutUint32(f32[4*i:], math.Float32bits(float32(math.Sin(float64(i)/300)*1e3+float64(i%17))))
	}
	rnd := make([]byte, 150000)
	r.Read(rnd)
	branchy := make([]byte, 400000)
	for i := range branchy {
		switch k := r.Intn(16); {
		case k < 3:
			branchy[i] = 0xE8
		case k < 4:
			branchy[i] = 0xE9
		case k < 7:
			branchy[i] = 0x00
		case k < 9:
			branchy[i] = 0xFF
		case k < 10:
			branchy[i] = 0xEB
		case k < 11:
			branchy[i] = 0x48 | byte(r.Intn(4))
		case k < 12:
			branchy[i] = 0x94 + byte(r.Intn(4))
		default:
			branchy[i] = byte(r.Intn(256))
		}
	}
	mixed := append(append(append(testData(1<<20, 12), rnd...), make([]byte, 1<<20)...), testData(1<<20, 13)...)
	return map[string][]byte{
		"empty":   {},
		"one":     {7},
		"text":    testData(120000, 14),
		"f32":     f32,
		"rand":    rnd,
		"branchy": branchy,
		"mixed":   mixed,
	}
}

type liveCfg struct {
	name    string
	format  int
	check   int
	preset  int
	filters []Filter
	inputs  []string
}

func liveConfigs() []liveCfg {
	small := []string{"empty", "one", "text", "rand", "f32"}
	var cs []liveCfg
	add := func(name string, format, check, preset int, filters []Filter, inputs ...string) {
		if inputs == nil {
			inputs = small
		}
		cs = append(cs, liveCfg{name, format, check, preset, filters, inputs})
	}
	add("xz-default", FormatXZ, -1, -1, nil, "empty", "one", "text", "rand", "f32", "mixed")
	for _, chk := range []int{CheckNone, CheckCRC32, CheckCRC64, CheckSHA256} {
		add(fmt.Sprintf("xz-check%d", chk), FormatXZ, chk, -1, nil)
	}
	for lvl := 0; lvl <= 9; lvl++ {
		add(fmt.Sprintf("xz-preset%d", lvl), FormatXZ, -1, lvl, nil, "text", "f32")
		add(fmt.Sprintf("xz-preset%de", lvl), FormatXZ, -1, lvl|PresetExtreme, nil, "text")
	}
	for _, d := range []int{1, 2, 4, 8, 256} {
		add(fmt.Sprintf("xz-delta%d", d), FormatXZ, -1, -1, []Filter{delta(d), lz2(map[string]any{"preset": 1})})
	}
	for _, b := range []struct {
		name string
		id   uint64
	}{{"x86", FilterX86}, {"powerpc", FilterPowerPC}, {"ia64", FilterIA64}, {"arm", FilterARM},
		{"armthumb", FilterARMThumb}, {"sparc", FilterSPARC}, {"arm64", FilterARM64}} {
		add("xz-"+b.name, FormatXZ, -1, -1, []Filter{bcj(b.id, 0), lz2(nil)}, "branchy", "text", "one")
		add("xz-"+b.name+"-off", FormatXZ, CheckCRC32, -1, []Filter{bcj(b.id, 1<<20), lz2(nil)}, "branchy")
		if b.id != FilterARM64 { // Python's lzma module has no ARM64 filter specifier
			add("raw-"+b.name, FormatRaw, -1, -1, []Filter{bcj(b.id, 0), lz2(nil)}, "branchy")
			add("raw-"+b.name+"-lzma1", FormatRaw, -1, -1, []Filter{bcj(b.id, 4096), lz1(nil)}, "branchy")
		}
	}
	add("xz-chain3", FormatXZ, CheckSHA256, -1, []Filter{bcj(FilterX86, 0), delta(4), lz2(map[string]any{"preset": 9 | PresetExtreme})}, "branchy", "f32")
	add("xz-lclppb", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"lc": 0, "lp": 4, "pb": 0, "dict_size": 1 << 16})})
	add("xz-dict4k", FormatXZ, -1, -1, []Filter{lz2(map[string]any{"dict_size": 4096})}, "text", "rand", "mixed")
	add("alone-default", FormatAlone, -1, -1, nil, "empty", "one", "text", "rand", "f32", "mixed")
	add("alone-preset0", FormatAlone, CheckNone, 0, nil)
	add("alone-preset9e", FormatAlone, -1, 9|PresetExtreme, nil, "text")
	add("alone-custom", FormatAlone, -1, -1, []Filter{lz1(map[string]any{"lc": 1, "lp": 3, "pb": 1, "dict_size": 5000})})
	add("raw-lzma2", FormatRaw, -1, -1, []Filter{lz2(nil)}, "empty", "one", "text", "rand", "f32", "mixed")
	add("raw-lzma2-dict4k", FormatRaw, -1, -1, []Filter{lz2(map[string]any{"dict_size": 4096})}, "text", "rand")
	add("raw-lzma2-lc4pb4", FormatRaw, -1, -1, []Filter{lz2(map[string]any{"lc": 4, "lp": 0, "pb": 4, "preset": 3})})
	add("raw-lzma1", FormatRaw, -1, -1, []Filter{lz1(nil)}, "empty", "one", "text", "rand", "f32")
	add("raw-lzma1-lc0lp4pb4", FormatRaw, -1, -1, []Filter{lz1(map[string]any{"lc": 0, "lp": 4, "pb": 4, "dict_size": 1 << 16})})
	add("raw-lzma1-dict5000", FormatRaw, -1, -1, []Filter{lz1(map[string]any{"dict_size": 5000})}, "text", "rand")
	add("raw-delta-lzma1", FormatRaw, -1, -1, []Filter{delta(4), lz1(map[string]any{"lc": 0, "lp": 2, "pb": 2})}, "f32")
	add("raw-delta-lzma2", FormatRaw, -1, -1, []Filter{delta(2), lz2(nil)}, "f32")
	return cs
}

// TestPythonDecodesGo: everything Compress produces is decoded by Python's
// lzma.decompress (and, for numcodecs-style configurations, by the
// numcodecs LZMA codec) to the original data.
func TestPythonDecodesGo(t *testing.T) {
	py := pythonPath(t)
	inputs := liveInputs()
	type item struct {
		name string
		in   []byte
	}
	var reqs []pyReq
	var items []item
	for _, c := range liveConfigs() {
		for _, inName := range c.inputs {
			in := inputs[inName]
			comp, err := Compress(in, c.format, c.check, c.preset, c.filters)
			if err != nil {
				t.Fatalf("%s/%s: %v", c.name, inName, err)
			}
			var df []map[string]any
			if c.format == FormatRaw {
				df = filterMaps(c.filters)
			}
			enc := base64.StdEncoding.EncodeToString(comp)
			reqs = append(reqs, pyReq{Op: "decompress", Data: enc, Format: c.format, Filters: df})
			items = append(items, item{c.name + "/" + inName, in})
			// numcodecs decodes with its own configuration; it can't decode
			// .xz/.lzma when filters are configured (see Decompress).
			if c.format == FormatRaw || c.filters == nil {
				var preset *int
				if c.preset != -1 {
					preset = intPtr(c.preset)
				}
				reqs = append(reqs, pyReq{Op: "numcodecs_decode", Data: enc, Format: c.format, Check: c.check,
					Preset: preset, Filters: filterMaps(c.filters)})
				items = append(items, item{c.name + "/" + inName + "/numcodecs", in})
			}
		}
	}
	// .lzma with the size in the header and no end marker (7-Zip style).
	for _, inName := range []string{"one", "text", "rand", "f32"} {
		enc := base64.StdEncoding.EncodeToString(knownSizeAlone(t, inputs[inName]))
		for _, f := range []int{FormatAlone, FormatAuto} {
			reqs = append(reqs, pyReq{Op: "decompress", Data: enc, Format: f})
			items = append(items, item{fmt.Sprintf("alone-known-size/%s/format%d", inName, f), inputs[inName]})
		}
	}
	// And the exact arguments of every Python-generated golden vector.
	g := loadGolden(t)
	for _, c := range g.Cases {
		if c.Compress == nil || len(c.Expect) != 1 {
			continue
		}
		in := g.Inputs[c.Expect[0]]
		comp, err := compressGolden(t, in, c.Compress)
		if err != nil {
			t.Fatalf("golden %s: %v", c.Name, err)
		}
		reqs = append(reqs, pyReq{Op: "decompress", Data: base64.StdEncoding.EncodeToString(comp),
			Format: c.Format, Filters: c.Filters})
		items = append(items, item{"golden/" + c.Name, in})
	}

	res := runPython(t, py, reqs)
	failed := 0
	for i, r := range res {
		if !r.OK {
			failed++
			t.Errorf("%s: Python failed: %s", items[i].name, r.Error)
			continue
		}
		got, _ := base64.StdEncoding.DecodeString(r.Data)
		if !bytes.Equal(got, items[i].in) {
			failed++
			t.Errorf("%s: Python output differs (got %d bytes, want %d)", items[i].name, len(got), len(items[i].in))
		}
	}
	t.Logf("%d Go-compressed streams checked with Python, %d failures", len(reqs), failed)
}

// TestPythonEncodesGoDecodes: Python compresses the live inputs with many
// configurations and this package decodes them. For raw [BCJ/delta, LZMA2]
// streams it also decodes with LZMA2 only and compares liblzma's filtered
// bytes with this package's filter encoder, byte for byte, on large
// branch-heavy random data.
func TestPythonEncodesGoDecodes(t *testing.T) {
	py := pythonPath(t)
	inputs := liveInputs()
	type item struct {
		name    string
		in      []byte
		format  int
		filters []Filter
	}
	var reqs []pyReq
	var items []item
	for _, c := range liveConfigs() {
		if c.format == FormatXZ && len(c.filters) > 0 && c.filters[0].ID == FilterARM64 {
			continue // Python's lzma module cannot encode ARM64
		}
		for _, inName := range c.inputs {
			var preset *int
			if c.preset != -1 {
				preset = intPtr(c.preset)
			}
			reqs = append(reqs, pyReq{Op: "compress", Data: base64.StdEncoding.EncodeToString(inputs[inName]),
				Format: c.format, Check: c.check, Preset: preset, Filters: filterMaps(c.filters)})
			var df []Filter
			if c.format == FormatRaw {
				df = c.filters
			}
			items = append(items, item{c.name + "/" + inName, inputs[inName], c.format, df})
		}
	}
	res := runPython(t, py, reqs)
	exact := 0
	for i, r := range res {
		it := items[i]
		if !r.OK {
			t.Errorf("%s: Python compress failed: %s", it.name, r.Error)
			continue
		}
		comp, _ := base64.StdEncoding.DecodeString(r.Data)
		got, err := Decompress(comp, it.format, it.filters)
		if err != nil {
			t.Errorf("%s: %v", it.name, err)
			continue
		}
		if !bytes.Equal(got, it.in) {
			t.Errorf("%s: output differs at %d", it.name, firstDiff(got, it.in))
			continue
		}
		if it.format != FormatRaw {
			if got, err := Decompress(comp, FormatAuto, nil); err != nil || !bytes.Equal(got, it.in) {
				t.Errorf("%s: FormatAuto: %v", it.name, err)
			}
		}
		if it.format == FormatRaw && len(it.filters) == 2 {
			lib, err := Decompress(comp, FormatRaw, it.filters[1:])
			if err != nil {
				t.Errorf("%s: LZMA-only decode: %v", it.name, err)
				continue
			}
			chain, _ := parseChain(it.filters, true)
			if ours := applyEncodeFilters(it.in, chain[:1]); !bytes.Equal(ours, lib) {
				t.Errorf("%s: filter output differs from liblzma's at %d", it.name, firstDiff(ours, lib))
			}
			exact++
		}
	}
	t.Logf("%d Python-compressed streams decoded; %d filter outputs compared byte for byte", len(reqs), exact)
}
