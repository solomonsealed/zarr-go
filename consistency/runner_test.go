package consistency

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	zarr "github.com/solomonsealed/zarr-go"
)

func marshal(v any) ([]byte, error) { return json.Marshal(v) }

// Case describes an array in terms both implementations understand: the
// arguments of zarr-python's create_array. Codecs are JSON objects in the
// order zarr-python receives them.
type Case struct {
	Name          string     `json:"-"`
	ZarrFormat    int        `json:"zarr_format"`
	Shape         []uint64   `json:"shape"`
	Chunks        []uint64   `json:"chunks"`
	ChunkEdges    [][]uint64 `json:"chunk_edges,omitempty"`
	Shards        []uint64   `json:"shards,omitempty"`
	IndexLocation string     `json:"index_location,omitempty"`
	// DataType is the Zarr v3 data_type JSON; DType the Zarr v2 dtype.
	DataType   json.RawMessage `json:"data_type,omitempty"`
	DType      json.RawMessage `json:"dtype,omitempty"`
	StringType string          `json:"string_type,omitempty"` // v2 "|O": "string" or "bytes"
	FillValue  json.RawMessage `json:"fill_value,omitempty"`
	// Filters, Serializer and Compressors are Zarr v3 codec objects, or
	// numcodecs configs ({"id": ...}) for Zarr v2. Compressors nil means
	// the default; an empty list means none.
	Filters          []json.RawMessage  `json:"filters,omitempty"`
	Serializer       json.RawMessage    `json:"serializer,omitempty"`
	Compressors      *[]json.RawMessage `json:"compressors,omitempty"`
	DimensionNames   []any              `json:"dimension_names,omitempty"`
	Attributes       map[string]any     `json:"attributes,omitempty"`
	ChunkKeyEncoding json.RawMessage    `json:"chunk_key_encoding,omitempty"`
	Order            string             `json:"order,omitempty"`
	WriteEmptyChunks bool               `json:"write_empty_chunks,omitempty"`
	// Parity reports whether chunk bytes must be identical; it is computed
	// from the codecs unless set.
	Parity *bool `json:"-"`
	// Safe restricts generated values to small finite numbers, for lossy
	// codecs whose out-of-range behaviour NumPy leaves undefined.
	Safe bool `json:"-"`
}

func rawList(objs ...string) []json.RawMessage {
	out := make([]json.RawMessage, len(objs))
	for i, o := range objs {
		out[i] = json.RawMessage(o)
	}
	return out
}

func compressors(objs ...string) *[]json.RawMessage {
	l := rawList(objs...)
	return &l
}

// goSpec converts the case to an ArraySpec.
func (c Case) goSpec() (zarr.ArraySpec, error) {
	s := zarr.ArraySpec{
		ZarrFormat:    c.ZarrFormat,
		Shape:         c.Shape,
		ChunkShape:    c.Chunks,
		ChunkEdges:    c.ChunkEdges,
		ShardShape:    c.Shards,
		IndexLocation: c.IndexLocation,
		Order:         c.Order,
		Attributes:    c.Attributes,
	}
	if c.FillValue != nil {
		s.FillValue = c.FillValue
	}
	codec := func(raw json.RawMessage) (zarr.CodecSpec, error) {
		var cs zarr.CodecSpec
		if c.ZarrFormat == 2 {
			var obj map[string]json.RawMessage
			if err := json.Unmarshal(raw, &obj); err != nil {
				return cs, err
			}
			json.Unmarshal(obj["id"], &cs.Name)
			cs.Configuration = raw
			return cs, nil
		}
		err := json.Unmarshal(raw, &cs)
		return cs, err
	}
	for _, f := range c.Filters {
		cs, err := codec(f)
		if err != nil {
			return s, err
		}
		s.Filters = append(s.Filters, cs)
	}
	if c.Serializer != nil {
		cs, err := codec(c.Serializer)
		if err != nil {
			return s, err
		}
		s.Serializer = &cs
	}
	if c.Compressors != nil {
		s.Compressors = []zarr.CodecSpec{}
		for _, f := range *c.Compressors {
			cs, err := codec(f)
			if err != nil {
				return s, err
			}
			s.Compressors = append(s.Compressors, cs)
		}
	}
	if c.ZarrFormat == 2 {
		var dt string
		if err := json.Unmarshal(c.DType, &dt); err == nil {
			s.DataType = dt
			if dt == "|O" {
				s.DataType = "string"
				if c.StringType == "bytes" {
					s.DataType = "variable_length_bytes"
				}
			}
		} else {
			// Structured: [[name, dtype], ...] as the v2 dtype.
			t, err := zarr.ParseMetadataV2([]byte(fmt.Sprintf(`{"zarr_format":2,"shape":[1],"chunks":[1],"dtype":%s,"fill_value":null,"order":"C","filters":null,"compressor":null}`, c.DType)), nil)
			if err != nil {
				return s, err
			}
			s.Type = t.DataType
		}
	} else {
		dt, err := zarr.ParseDataTypeJSON(c.DataType)
		if err != nil {
			return s, err
		}
		s.Type = dt
		// Keep zarr-python's spelling of the type (e.g. "structured").
	}
	for _, n := range c.DimensionNames {
		if n == nil {
			s.DimensionNames = append(s.DimensionNames, "")
		} else {
			s.DimensionNames = append(s.DimensionNames, n.(string))
		}
	}
	if c.ChunkKeyEncoding != nil {
		var e struct {
			Name          string `json:"name"`
			Configuration struct {
				Separator string `json:"separator"`
				Format    string `json:"format"`
			} `json:"configuration"`
		}
		if err := json.Unmarshal(c.ChunkKeyEncoding, &e); err != nil {
			return s, err
		}
		s.ChunkKeyEncoding = zarr.ChunkKeyEncoding{Name: e.Name, Separator: e.Configuration.Separator, Template: e.Configuration.Format}
	}
	return s, nil
}

// Codecs whose output legitimately differs between the implementations
// (other compressor implementations, or gzip's timestamp). Chunks written
// through them are compared by decoding instead of byte for byte.
var nonDeterministic = map[string]bool{
	"zstd": true, "gzip": true, "numcodecs.zstd": true, "numcodecs.gzip": true,
	"numcodecs.zlib": true, "numcodecs.lzma": true, "zlib": true, "lzma": true,
}

// Compressors made byte-identical to the C libraries numcodecs uses
// (blosc's zstd and zlib payloads come from other implementations).
var (
	bloscParity = map[string]bool{"blosclz": true, "lz4": true, "lz4hc": true}
	lz4Parity   = true
)

func (c Case) parity() bool {
	if c.Parity != nil {
		return *c.Parity
	}
	if c.Compressors == nil {
		return false // the default is zstd
	}
	objs := append(append([]json.RawMessage{}, c.Filters...), *c.Compressors...)
	if c.Serializer != nil {
		objs = append(objs, c.Serializer)
	}
	for _, o := range objs {
		var spec struct {
			Name   string         `json:"name"`
			ID     string         `json:"id"`
			Config map[string]any `json:"configuration"`
			Cname  string         `json:"cname"`
		}
		json.Unmarshal(o, &spec)
		name := spec.Name + spec.ID
		if nonDeterministic[name] {
			return false
		}
		switch name {
		case "blosc", "numcodecs.blosc":
			cname, _ := spec.Config["cname"].(string)
			if spec.Cname != "" {
				cname = spec.Cname
			}
			if cname == "" {
				cname = map[bool]string{true: "zstd", false: "lz4"}[name == "blosc" && c.ZarrFormat != 2]
			}
			if !bloscParity[cname] {
				return false
			}
		case "numcodecs.lz4", "lz4":
			if !lz4Parity {
				return false
			}
		}
	}
	return true
}

// ---------------------------------------------------------------------------
// Running a case

type harness struct {
	t      *testing.T
	o      *oracle
	c      Case
	ctx    context.Context
	pyDir  string
	goDir  string
	goArr  *zarr.Array
	dt     zarr.DataType
	rng    *rand.Rand
	report *caseReport
}

// caseReport counts what was compared.
type caseReport struct {
	Name        string
	Ops         int
	Comparisons int
	ByteChunks  int // chunks compared byte for byte
	ValueChunks int // chunks compared after decoding
	Errors      int // operations both sides rejected
}

var (
	reportMu sync.Mutex
	reports  []*caseReport
)

func newHarness(t *testing.T, o *oracle, c Case, seed int64) *harness {
	h := &harness{t: t, o: o, c: c, ctx: context.Background(), pyDir: t.TempDir(), goDir: t.TempDir(), rng: rand.New(rand.NewSource(seed)), report: &caseReport{Name: t.Name()}}
	reportMu.Lock()
	reports = append(reports, h.report)
	reportMu.Unlock()
	return h
}

func (h *harness) pyReq(extra map[string]any) map[string]any {
	req := map[string]any{"store": h.pyDir, "path": "arr", "write_empty_chunks": h.c.WriteEmptyChunks}
	for k, v := range extra {
		req[k] = v
	}
	return req
}

// create makes the array with both implementations and compares the
// metadata. It returns false if both refused the case.
func (h *harness) create() bool {
	t := h.t
	t.Helper()
	pyErr := h.o.call("create", h.pyReq(map[string]any{"case": h.c}), nil)
	spec, err := h.c.goSpec()
	var goErr error
	if err == nil {
		h.goArr, goErr = zarr.CreateArray(h.ctx, zarr.NewLocalStore(h.goDir), "arr", spec, &zarr.CreateOptions{Options: zarr.Options{WriteEmptyChunks: h.c.WriteEmptyChunks}})
	} else {
		goErr = err
	}
	if pyErr != nil || goErr != nil {
		if pyErr != nil && goErr != nil {
			h.report.Errors++
			t.Logf("both refused the case: python %v; go %v", pyErr, goErr)
			return false
		}
		t.Fatalf("create: python %v; go %v", pyErr, goErr)
	}
	h.dt = h.goArr.DataType()
	h.compareStores("create")
	return true
}

// files returns the files of a store directory.
func files(t *testing.T, dir string) map[string][]byte {
	out := map[string][]byte{}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		b, err := os.ReadFile(p)
		out[filepath.ToSlash(rel)] = b
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func isMetadataKey(k string) bool {
	switch filepath.Base(k) {
	case "zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata":
		return true
	}
	return false
}

// compareStores requires identical keys and metadata, and identical or
// equivalent chunks.
func (h *harness) compareStores(when string) {
	t := h.t
	t.Helper()
	py, gs := files(t, h.pyDir), files(t, h.goDir)
	var pk, gk []string
	for k := range py {
		pk = append(pk, k)
	}
	for k := range gs {
		gk = append(gk, k)
	}
	slices.Sort(pk)
	slices.Sort(gk)
	if !slices.Equal(pk, gk) {
		t.Fatalf("%s: keys differ\npython: %v\ngo:     %v", when, pk, gk)
	}
	var pyArr *zarr.Array
	for _, k := range pk {
		h.report.Comparisons++
		if isMetadataKey(k) {
			if !bytes.Equal(py[k], gs[k]) {
				t.Fatalf("%s: %s differs:\n%s", when, k, lineDiff(string(py[k]), string(gs[k])))
			}
			continue
		}
		if h.c.parity() {
			h.report.ByteChunks++
			if !bytes.Equal(py[k], gs[k]) {
				t.Fatalf("%s: chunk %s differs (%d vs %d bytes)\npython: % x\ngo:     % x", when, k, len(py[k]), len(gs[k]), head(py[k]), head(gs[k]))
			}
			continue
		}
		// Different compressor output: both must decode to the same chunk.
		h.report.ValueChunks++
		if pyArr == nil {
			var err error
			if pyArr, err = zarr.OpenArray(h.ctx, zarr.NewLocalStore(h.pyDir), "arr", nil); err != nil {
				t.Fatalf("%s: go opening python's array: %v", when, err)
			}
		}
		rel := strings.TrimPrefix(k, "arr/")
		coords, ok := h.goArr.Metadata().ChunkKeyEncoding.Parse(rel, h.goArr.NDim())
		if !ok {
			t.Fatalf("%s: unexpected key %s", when, k)
		}
		pc, err := pyArr.ReadChunk(h.ctx, coords)
		if err != nil {
			t.Fatalf("%s: go decoding python's chunk %s: %v", when, k, err)
		}
		gc, err := h.goArr.ReadChunk(h.ctx, coords)
		if err != nil {
			t.Fatalf("%s: go decoding its chunk %s: %v", when, k, err)
		}
		if err := sameArray(pc, gc); err != nil {
			t.Fatalf("%s: chunk %s decodes differently: %v", when, k, err)
		}
	}
}

// lineDiff shows the first differing lines with some context.
func lineDiff(a, b string) string {
	al, bl := strings.Split(a, "\n"), strings.Split(b, "\n")
	i := 0
	for i < len(al) && i < len(bl) && al[i] == bl[i] {
		i++
	}
	lo := max(0, i-4)
	var sb strings.Builder
	for k := lo; k < i; k++ {
		fmt.Fprintf(&sb, "      %s\n", al[k])
	}
	for k := i; k < min(len(al), i+6); k++ {
		fmt.Fprintf(&sb, "  py: %s\n", al[k])
	}
	for k := i; k < min(len(bl), i+6); k++ {
		fmt.Fprintf(&sb, "  go: %s\n", bl[k])
	}
	return sb.String()
}

func head(b []byte) []byte {
	if len(b) > 64 {
		return b[:64]
	}
	return b
}

// crossRead reads both stores with both implementations.
func (h *harness) crossRead(when string) {
	t := h.t
	t.Helper()
	var pp, pg wireArray
	if err := h.o.call("read_all", map[string]any{"store": h.pyDir, "path": "arr"}, &pp); err != nil {
		t.Fatalf("%s: python reading its array: %v", when, err)
	}
	if err := h.o.call("read_all", map[string]any{"store": h.goDir, "path": "arr"}, &pg); err != nil {
		t.Fatalf("%s: python reading go's array: %v", when, err)
	}
	pyP, err := fromWire(pp, h.dt)
	if err != nil {
		t.Fatal(err)
	}
	pyG, err := fromWire(pg, h.dt)
	if err != nil {
		t.Fatal(err)
	}
	goG, err := h.goArr.ReadAll(h.ctx)
	if err != nil {
		t.Fatalf("%s: go reading its array: %v", when, err)
	}
	other, err := zarr.OpenArray(h.ctx, zarr.NewLocalStore(h.pyDir), "arr", nil)
	if err != nil {
		t.Fatalf("%s: go opening python's array: %v", when, err)
	}
	goP, err := other.ReadAll(h.ctx)
	if err != nil {
		t.Fatalf("%s: go reading python's array: %v", when, err)
	}
	h.report.Comparisons += 3
	for _, c := range []struct {
		name string
		a    *zarr.NDArray
	}{{"python reading go's store", pyG}, {"go reading python's store", goP}, {"go reading its store", goG}} {
		if err := sameArray(pyP, c.a); err != nil {
			t.Fatalf("%s: %s differs from python reading its store: %v", when, c.name, err)
		}
	}
}

// write applies a write with both implementations.
func (h *harness) write(sel selection, data *zarr.NDArray) {
	t := h.t
	t.Helper()
	h.report.Ops++
	pyErr := h.o.call("write", h.pyReq(map[string]any{"selection": sel, "data": toWire(data)}), nil)
	goErr := sel.set(h.ctx, h.goArr, data)
	if pyErr != nil || goErr != nil {
		if pyErr != nil && goErr != nil {
			h.report.Errors++
			return
		}
		t.Fatalf("write %v: python %v; go %v", sel, pyErr, goErr)
	}
	h.compareStores("after write " + sel.String())
}

// read compares a selection read by both implementations from both stores.
func (h *harness) read(sel selection) {
	t := h.t
	t.Helper()
	h.report.Ops++
	var pw wireArray
	pyErr := h.o.call("read", h.pyReq(map[string]any{"selection": sel}), &pw)
	g, goErr := sel.get(h.ctx, h.goArr)
	if pyErr != nil || goErr != nil {
		if pyErr != nil && goErr != nil {
			h.report.Errors++
			return
		}
		t.Fatalf("read %v: python %v; go %v", sel, pyErr, goErr)
	}
	p, err := fromWire(pw, h.dt)
	if err != nil {
		t.Fatal(err)
	}
	h.report.Comparisons++
	if err := sameArray(p, g); err != nil {
		t.Fatalf("read %v: %v", sel, err)
	}
}

func (h *harness) resize(shape []uint64) {
	t := h.t
	t.Helper()
	h.report.Ops++
	pyErr := h.o.call("resize", h.pyReq(map[string]any{"shape": shape}), nil)
	goErr := h.goArr.Resize(h.ctx, shape)
	if pyErr != nil || goErr != nil {
		if pyErr != nil && goErr != nil {
			h.report.Errors++
			return
		}
		t.Fatalf("resize %v: python %v; go %v", shape, pyErr, goErr)
	}
	h.compareStores(fmt.Sprintf("after resize %v", shape))
	h.crossRead(fmt.Sprintf("after resize %v", shape))
}

// ---------------------------------------------------------------------------
// Data

// genData returns random values of the array's type; with probability
// pFill each element is the fill value.
func genData(rng *rand.Rand, dt zarr.DataType, shape []uint64, fill *zarr.NDArray, pFill float64) *zarr.NDArray {
	return genDataSafe(rng, dt, shape, fill, pFill, false)
}

func genDataSafe(rng *rand.Rand, dt zarr.DataType, shape []uint64, fill *zarr.NDArray, pFill float64, safe bool) *zarr.NDArray {
	n := 1
	for _, s := range shape {
		n *= int(s)
	}
	out := &zarr.NDArray{Shape: append([]uint64{}, shape...), DataType: dt}
	if dt.Variable() {
		out.Strings = make([]string, n)
		for i := range out.Strings {
			if rng.Float64() < pFill {
				out.Strings[i] = fill.Strings[0]
				continue
			}
			out.Strings[i] = randString(rng, dt.Kind == zarr.KindBytes)
		}
		return out
	}
	out.Data = make([]byte, n*dt.Size)
	for i := 0; i < n; i++ {
		e := out.Data[i*dt.Size : (i+1)*dt.Size]
		if rng.Float64() < pFill {
			copy(e, fill.Data)
			continue
		}
		if safe && (dt.Kind == zarr.KindFloat || dt.Kind == zarr.KindInt || dt.Kind == zarr.KindUint) {
			v, _ := zarr.FromValues([]uint64{1}, dt, []float64{float64(rng.Intn(100))})
			if dt.Kind == zarr.KindFloat {
				v, _ = zarr.FromValues([]uint64{1}, dt, []float64{math.Round(rng.NormFloat64()*1000) / 64})
			}
			copy(e, v.Data)
			continue
		}
		genElement(rng, dt, e)
	}
	return out
}

var sampleRunes = []rune("abcxyzAZ09 _-éüßøπЖ中文😀∑ ")

func randString(rng *rand.Rand, raw bool) string {
	n := rng.Intn(8)
	if rng.Intn(6) == 0 {
		n = 0
	}
	if raw {
		b := make([]byte, n)
		rng.Read(b)
		return string(b)
	}
	var sb strings.Builder
	for range n {
		sb.WriteRune(sampleRunes[rng.Intn(len(sampleRunes))])
	}
	return sb.String()
}

func genElement(rng *rand.Rand, dt zarr.DataType, e []byte) {
	switch dt.Kind {
	case zarr.KindBool:
		e[0] = byte(rng.Intn(2))
	case zarr.KindInt, zarr.KindUint:
		rng.Read(e)
		if rng.Intn(3) == 0 { // small values compress and hit edge cases
			clear(e)
			e[0] = byte(rng.Intn(5))
			if dt.Kind == zarr.KindInt && rng.Intn(2) == 0 {
				for i := range e {
					e[i] = ^e[i]
				}
			}
		}
	case zarr.KindFloat:
		genFloat(rng, dt, e)
	case zarr.KindComplex:
		h := dt.Size / 2
		half := dt
		half.Size = h
		half.Kind = zarr.KindFloat
		half.Name = map[int]string{4: "float32", 8: "float64"}[h]
		genFloat(rng, half, e[:h])
		genFloat(rng, half, e[h:])
	case zarr.KindDatetime, zarr.KindTimedelta:
		v := rng.Int63n(4e12) - 2e12
		if rng.Intn(8) == 0 {
			v = math.MinInt64 // NaT
		}
		binary.LittleEndian.PutUint64(e, uint64(v))
	case zarr.KindFixedString:
		clear(e)
		n := rng.Intn(len(e)/4 + 1)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(e[4*i:], uint32(sampleRunes[rng.Intn(len(sampleRunes))]))
		}
	case zarr.KindFixedBytes, zarr.KindRaw:
		rng.Read(e)
		if dt.Name == "null_terminated_bytes" {
			n := rng.Intn(len(e) + 1)
			clear(e[n:])
			for i := 0; i < n; i++ {
				if e[i] == 0 {
					e[i] = 1
				}
			}
		}
	case zarr.KindStruct:
		for _, f := range dt.Fields {
			genElement(rng, f.Type, e[f.Offset:f.Offset+f.Type.Size])
		}
	}
}

func genFloat(rng *rand.Rand, dt zarr.DataType, e []byte) {
	var f float64
	switch r := rng.Intn(20); {
	case r == 0:
		f = math.NaN()
	case r == 1:
		f = math.Inf(1 - 2*rng.Intn(2))
	case r == 2:
		f = math.Copysign(0, -1)
	case r == 3:
		f = 0
	case r < 8:
		f = float64(rng.Intn(200) - 100)
	default:
		f = rng.NormFloat64() * math.Pow(10, float64(rng.Intn(6)-3))
	}
	v, _ := zarr.FromValues([]uint64{1}, dt, []float64{f})
	copy(e, v.Data)
	if math.IsNaN(f) {
		// numpy's NaN, not Go's (which has a payload).
		switch dt.Size {
		case 2:
			binary.LittleEndian.PutUint16(e, 0x7e00)
		case 4:
			binary.LittleEndian.PutUint32(e, 0x7fc00000)
		case 8:
			binary.LittleEndian.PutUint64(e, 0x7ff8000000000000)
		}
	}
}

// ---------------------------------------------------------------------------
// Selections

func randomSelection(rng *rand.Rand, kind string, shape, grid []uint64) selection {
	nd := len(shape)
	switch kind {
	case "coordinate":
		k := rng.Intn(12)
		coords := make([][]int64, nd)
		for d := range coords {
			coords[d] = make([]int64, k)
			for i := range coords[d] {
				n := int64(shape[d])
				if n == 0 {
					return selection{Kind: "orthogonal"}
				}
				coords[d][i] = rng.Int63n(2*n) - n
			}
		}
		return selection{Kind: "coordinate", Coords: coords}
	case "mask":
		n := 1
		for _, s := range shape {
			n *= int(s)
		}
		m := make([]bool, n)
		density := rng.Float64()
		for i := range m {
			m[i] = rng.Float64() < density
		}
		arr, _ := zarr.FromSlice(shape, m)
		return withMask(arr)
	case "block":
		items := make([]dimItem, nd)
		for d := range items {
			nc := int64(grid[d])
			if nc > 0 && rng.Intn(2) == 0 {
				items[d] = idx(rng.Int63n(2*nc) - nc)
			} else {
				items[d] = slc(randBound(rng, nc), randBound(rng, nc), nil)
			}
		}
		return selection{Kind: "block", Items: items}
	}
	items := make([]dimItem, nd)
	for d := range items {
		n := int64(shape[d])
		choice := rng.Intn(4)
		if kind == "basic" {
			choice = rng.Intn(2)
		}
		if n == 0 {
			choice = 1
		}
		switch choice {
		case 0:
			items[d] = idx(rng.Int63n(2*n) - n)
		case 1:
			var step *int64
			if rng.Intn(2) == 0 {
				step = ptr(int64(1 + rng.Intn(3)))
			}
			items[d] = slc(randBound(rng, n), randBound(rng, n), step)
		case 2:
			k := rng.Intn(int(2*n) + 1)
			v := make([]int64, k)
			for i := range v {
				v[i] = rng.Int63n(2*n) - n
			}
			items[d] = ints(v...)
		case 3:
			m := make([]bool, n)
			for i := range m {
				m[i] = rng.Intn(2) == 0
			}
			items[d] = bools(m...)
		}
	}
	// Trailing dimensions may be left out, or replaced by an Ellipsis
	// (not after array items: zarr-python's replace_ellipsis compares
	// items with ==, which fails on arrays).
	hasArrays := false
	for _, it := range items {
		hasArrays = hasArrays || it.isInts || it.isBools
	}
	if nd > 0 && rng.Intn(5) == 0 {
		cut := rng.Intn(nd + 1)
		if rng.Intn(2) == 0 && !hasArrays {
			items = append(items[:cut], dimItem{Ellipsis: true})
		} else {
			items = items[:cut]
		}
	}
	return selection{Kind: kind, Items: items}
}

func randBound(rng *rand.Rand, n int64) *int64 {
	if rng.Intn(4) == 0 {
		return nil
	}
	v := rng.Int63n(2*n+5) - n - 2
	return &v
}

// uniquePoints drops repeated positions from a write selection:
// zarr-python orders them with an unstable argsort, so which repeated
// write wins is undefined there.
func uniquePoints(sel selection, shape []uint64) selection {
	wrap := func(v int64, d int) int64 {
		if v < 0 {
			v += int64(shape[d])
		}
		return v
	}
	if sel.Kind == "orthogonal" || sel.Kind == "basic" {
		d := 0
		for i, it := range sel.Items {
			if it.Ellipsis {
				d += len(shape) - (len(sel.Items) - 1)
				continue
			}
			if it.isInts {
				seen := map[int64]bool{}
				v := []int64{}
				for _, x := range it.Ints {
					if !seen[wrap(x, d)] {
						seen[wrap(x, d)] = true
						v = append(v, x)
					}
				}
				sel.Items[i] = ints(v...)
			}
			d++
		}
	}
	if sel.Kind != "coordinate" || len(sel.Coords) == 0 {
		return sel
	}
	seen := map[string]bool{}
	out := make([][]int64, len(sel.Coords))
	for i := range sel.Coords[0] {
		key := ""
		for d := range sel.Coords {
			key += fmt.Sprint(wrap(sel.Coords[d][i], d), ",")
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		for d := range sel.Coords {
			out[d] = append(out[d], sel.Coords[d][i])
		}
	}
	for d := range out {
		if out[d] == nil {
			out[d] = []int64{}
		}
	}
	sel.Coords = out
	return sel
}

// exercise runs a standard sequence: a full write, random partial writes
// of every selection kind (some all-fill, deleting chunks), random reads,
// and for resizable cases a shrink and regrow.
func (h *harness) exercise(writes, reads int, resize bool) {
	t := h.t
	t.Helper()
	if !h.create() {
		return
	}
	fill := h.goArr.FillValue()
	shape := h.goArr.Shape()
	h.write(selection{Kind: "orthogonal"}, genDataSafe(h.rng, h.dt, shape, fill, 0.1, h.c.Safe))
	h.crossRead("after full write")
	kinds := []string{"orthogonal", "basic", "coordinate", "mask", "block"}
	for i := 0; i < writes; i++ {
		sel := uniquePoints(randomSelection(h.rng, kinds[i%len(kinds)], h.goArr.Shape(), h.goArr.ChunkGridShape()), h.goArr.Shape())
		cur, err := sel.get(h.ctx, h.goArr)
		if err != nil {
			h.read(sel) // both should refuse it
			continue
		}
		if h.dt.Kind == zarr.KindBytes && len(cur.Shape) == 0 {
			continue // zarr-python 3.4 cannot write one variable-length bytes element
		}
		pFill := 0.1
		if h.rng.Intn(4) == 0 {
			pFill = 1 // delete chunks
		}
		h.write(sel, genDataSafe(h.rng, h.dt, cur.Shape, fill, pFill, h.c.Safe))
	}
	h.crossRead("after partial writes")
	for i := 0; i < reads; i++ {
		h.read(randomSelection(h.rng, kinds[i%len(kinds)], h.goArr.Shape(), h.goArr.ChunkGridShape()))
	}
	if resize && len(shape) > 0 {
		small := slices.Clone(shape)
		for d := range small {
			small[d] = small[d] / 2
		}
		h.resize(small)
		h.resize(shape)
		sel := randomSelection(h.rng, "basic", shape, h.goArr.ChunkGridShape())
		if cur, err := sel.get(h.ctx, h.goArr); err == nil && !(h.dt.Kind == zarr.KindBytes && len(cur.Shape) == 0) {
			h.write(sel, genDataSafe(h.rng, h.dt, cur.Shape, fill, 0, h.c.Safe))
			h.crossRead("after regrow and write")
		}
	}
}

var _ = errors.New
var _ = utf8.ValidString
