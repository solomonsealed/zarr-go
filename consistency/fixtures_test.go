package consistency

import (
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"io/fs"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

// Recorded fixtures let the comparison with zarr-python run without
// Python: zarr-python's stores, the operations that produced them, and its
// answers to reads are committed under testdata/fixtures, and
// TestFixtures checks Go against them. Regenerate them with
//
//	ZARR_PYTHON=.venv/bin/python go test ./consistency -run TestFixtures -zarr.record

var record = flag.Bool("zarr.record", false, "re-record testdata/fixtures with zarr-python")

const fixtureDir = "testdata/fixtures"

// fixture is one recorded case.
type fixture struct {
	Case   Case        `json:"case"`
	Name   string      `json:"name"`
	Writes []fxWrite   `json:"writes"`
	Reads  []fxRead    `json:"reads"`
	All    wireArray   `json:"all"`
	Errors []selection `json:"errors,omitempty"` // selections zarr-python refused
}

type fxWrite struct {
	Sel  selection `json:"selection"`
	Data wireArray `json:"data"`
}

type fxRead struct {
	Sel    selection `json:"selection"`
	Result wireArray `json:"result"`
}

// fixtureCases are the recorded cases: every list above, on smaller arrays.
func fixtureCases() []Case {
	var out []Case
	add := func(group string, cs []Case) {
		for _, c := range cs {
			c.Name = group + "/" + c.Name
			out = append(out, c)
		}
	}
	add("dtypes-v3", dataTypeCasesV3())
	add("dtypes-v2", dataTypeCasesV2())
	add("codecs-v3", codecCasesV3())
	add("codecs-v2", codecCasesV2())
	add("sharding", shardingCases())
	add("grids", gridCases())
	add("shapes", shapeCases())
	return out
}

var unsafeName = regexp.MustCompile(`[^A-Za-z0-9._=-]+`)

func fixtureName(c Case) string {
	h := fnv.New32a()
	h.Write([]byte(c.Name))
	return fmt.Sprintf("%s-%06x", strings.Trim(unsafeName.ReplaceAllString(c.Name, "_"), "_"), h.Sum32()&0xffffff)
}

// zipDir packs a store directory into dir.zip (deflated) and removes it.
func zipDir(dir string) error {
	f, err := os.Create(dir + ".zip")
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	err = filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		w, err := zw.Create(filepath.ToSlash(rel))
		if err != nil {
			return err
		}
		b, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		_, err = w.Write(b)
		return err
	})
	if err != nil {
		return err
	}
	if err := zw.Close(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.RemoveAll(dir)
}

// unzipStore extracts a recorded store into a temporary directory.
func unzipStore(t *testing.T, name string) string {
	t.Helper()
	zs, err := zarr.OpenZipStore(filepath.Join(fixtureDir, name+".zip"))
	if err != nil {
		t.Fatal(err)
	}
	defer zs.Close()
	dir := t.TempDir()
	keys, _ := zs.List(context.Background(), "")
	for _, k := range keys {
		v, _, err := zs.Get(context.Background(), k, zarr.ByteRange{})
		if err != nil {
			t.Fatal(err)
		}
		p := filepath.Join(dir, filepath.FromSlash(k))
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, v, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// TestFixtures checks Go against the recorded zarr-python fixtures (or
// records them with -zarr.record).
func TestFixtures(t *testing.T) {
	if *record {
		recordFixtures(t)
		return
	}
	ctx := context.Background()
	manifest := readManifest(t)
	if len(manifest) == 0 {
		t.Skip("no recorded fixtures")
	}
	for _, fx := range manifest {
		t.Run(fx.Name, func(t *testing.T) {
			pyDir := unzipStore(t, fx.Name)
			goDir := t.TempDir()
			spec, err := fx.Case.goSpec()
			if err != nil {
				t.Fatal(err)
			}
			opts := &zarr.CreateOptions{Options: zarr.Options{WriteEmptyChunks: fx.Case.WriteEmptyChunks}}
			ga, err := zarr.CreateArray(ctx, zarr.NewLocalStore(goDir), "arr", spec, opts)
			if err != nil {
				t.Fatal(err)
			}
			dt := ga.DataType()
			for _, w := range fx.Writes {
				data, err := fromWire(w.Data, dt)
				if err != nil {
					t.Fatal(err)
				}
				w.Sel.restoreMask()
				if err := w.Sel.set(ctx, ga, data); err != nil {
					t.Fatalf("write %v: %v", w.Sel, err)
				}
			}
			// Same files: metadata byte for byte, chunks byte for byte where
			// the codecs are deterministic, else decoding to the same values.
			h := &harness{t: t, c: fx.Case, ctx: ctx, pyDir: pyDir, goDir: goDir, goArr: ga, dt: dt, report: &caseReport{}}
			h.compareStores("replay")
			// Go reads zarr-python's store as zarr-python did.
			pa, err := zarr.OpenArray(ctx, zarr.NewLocalStoreReadOnly(pyDir), "arr", nil)
			if err != nil {
				t.Fatal(err)
			}
			all, err := pa.ReadAll(ctx)
			if err != nil {
				t.Fatal(err)
			}
			want, _ := fromWire(fx.All, dt)
			if err := sameArray(want, all); err != nil {
				t.Fatalf("reading zarr-python's array: %v", err)
			}
			for _, r := range fx.Reads {
				r.Sel.restoreMask()
				got, err := r.Sel.get(ctx, pa)
				if err != nil {
					t.Fatalf("read %v: %v", r.Sel, err)
				}
				want, _ := fromWire(r.Result, dt)
				if dt.Kind == zarr.KindBytes && len(got.Shape) == 0 {
					got.Strings[0] = strings.TrimRight(got.Strings[0], "\x00")
				}
				if err := sameArray(want, got); err != nil {
					t.Fatalf("read %v: %v", r.Sel, err)
				}
			}
			for _, sel := range fx.Errors {
				sel.restoreMask()
				if _, err := sel.get(ctx, pa); err == nil {
					t.Errorf("read %v: zarr-python refused it, go did not", sel)
				}
			}
		})
	}
}

func (s *selection) restoreMask() {
	if s.Kind == "mask" && s.Mask != nil && s.mask == nil {
		s.mask, _ = fromWire(*s.Mask, zarr.MustDataType("bool"))
	}
}

func readManifest(t *testing.T) []fixture {
	f, err := os.Open(filepath.Join(fixtureDir, "manifest.json.gz"))
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	var m []fixture
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber() // attributes keep their integers
	if err := dec.Decode(&m); err != nil {
		t.Fatal(err)
	}
	return m
}

// recordFixtures runs every fixture case through zarr-python and records
// the stores, the writes, the reads and their results.
func recordFixtures(t *testing.T) {
	o := pythonOracle(t)
	ctx := context.Background()
	if err := os.RemoveAll(fixtureDir); err != nil {
		t.Fatal(err)
	}
	var manifest []fixture
	for _, c := range fixtureCases() {
		name := fixtureName(c)
		dir, err := filepath.Abs(filepath.Join(fixtureDir, name))
		if err != nil {
			t.Fatal(err)
		}
		rng := rand.New(rand.NewSource(seedOf(name)))
		if err := o.call("create", map[string]any{"store": dir, "path": "arr", "case": c}, nil); err != nil {
			t.Logf("%s: zarr-python refuses the case: %v", name, err)
			os.RemoveAll(dir)
			continue
		}
		// A scratch Go array computes selection shapes and fill values.
		spec, err := c.goSpec()
		if err != nil {
			t.Fatal(err)
		}
		ga, err := zarr.CreateArray(ctx, zarr.NewMemoryStore(), "arr", spec, nil)
		if err != nil {
			t.Fatal(err)
		}
		fx := fixture{Case: c, Name: name}
		dt, fill, shape := ga.DataType(), ga.FillValue(), ga.Shape()
		write := func(sel selection, data *zarr.NDArray) {
			req := map[string]any{"store": dir, "path": "arr", "selection": sel, "data": toWire(data), "write_empty_chunks": c.WriteEmptyChunks}
			if err := o.call("write", req, nil); err != nil {
				t.Fatalf("%s: write %v: %v", name, sel, err)
			}
			if err := sel.set(ctx, ga, data); err != nil {
				t.Fatalf("%s: go write %v: %v", name, sel, err)
			}
			fx.Writes = append(fx.Writes, fxWrite{sel, toWire(data)})
		}
		write(selection{Kind: "orthogonal"}, genDataSafe(rng, dt, shape, fill, 0.1, c.Safe))
		for i, kind := range []string{"orthogonal", "coordinate", "block"} {
			if c.Name == "dtypes-v2/vlen-bytes" {
				break
			}
			sel := uniquePoints(randomSelection(rng, kind, shape, ga.ChunkGridShape()), shape)
			cur, err := sel.get(ctx, ga)
			if err != nil || (dt.Kind == zarr.KindBytes && len(cur.Shape) == 0) {
				continue
			}
			pFill := 0.1
			if i == 2 {
				pFill = 1 // deletes chunks
			}
			write(sel, genDataSafe(rng, dt, cur.Shape, fill, pFill, c.Safe))
		}
		kinds := []string{"orthogonal", "basic", "coordinate", "mask", "block"}
		for i := 0; i < 6; i++ {
			sel := randomSelection(rng, kinds[i%len(kinds)], shape, ga.ChunkGridShape())
			var res wireArray
			if err := o.call("read", map[string]any{"store": dir, "path": "arr", "selection": sel}, &res); err != nil {
				fx.Errors = append(fx.Errors, sel)
				continue
			}
			fx.Reads = append(fx.Reads, fxRead{sel, res})
		}
		if err := o.call("read_all", map[string]any{"store": dir, "path": "arr"}, &fx.All); err != nil {
			t.Fatal(err)
		}
		if err := zipDir(dir); err != nil {
			t.Fatal(err)
		}
		manifest = append(manifest, fx)
	}
	recordHierarchies(t, o)
	slices.SortFunc(manifest, func(a, b fixture) int { return strings.Compare(a.Name, b.Name) })
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	zw, _ := gzip.NewWriterLevel(&buf, gzip.BestCompression)
	zw.Write(b)
	zw.Close()
	if err := os.WriteFile(filepath.Join(fixtureDir, "manifest.json.gz"), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	var total int64
	filepath.Walk(fixtureDir, func(_ string, fi os.FileInfo, _ error) error {
		if fi != nil && !fi.IsDir() {
			total += fi.Size()
		}
		return nil
	})
	t.Logf("recorded %d fixtures, %d KB", len(manifest), total/1024)
}
