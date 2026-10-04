package consistency

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

var hierarchyGroups = []struct {
	path  string
	attrs map[string]any
}{
	{"", map[string]any{"title": "root", "n": 1}},
	{"A", map[string]any{"b": 2}},
	{"A/sub", nil},
	{"b", map[string]any{"é": "ü", "list": []any{1, "x", nil, 2.5}}},
}

var hierarchyArrays = []struct {
	path string
	c    Case
}{
	{"A/x", Case{Shape: []uint64{4}, Chunks: []uint64{2}, Attributes: map[string]any{"units": "K"}}},
	{"A/sub/y", Case{Shape: []uint64{2, 3}, Chunks: []uint64{2, 2}}},
	{"z", Case{Shape: []uint64{}, Chunks: []uint64{}}},
	{"implicit/deep/w", Case{Shape: []uint64{5}, Chunks: []uint64{5}}}, // parents are created
}

func hierarchyCase(c Case, format int) Case {
	c.ZarrFormat = format
	if format == 3 {
		c.DataType = raw(`"int16"`)
	} else {
		c.DType = raw(`"<i2"`)
	}
	c.Compressors = compressors()
	return c
}

// buildHierarchy creates the test hierarchy with Go (o == nil) or zarr-python.
func buildHierarchy(t *testing.T, o *oracle, dir string, format int) {
	t.Helper()
	ctx := context.Background()
	gs := zarr.NewLocalStore(dir)
	for _, g := range hierarchyGroups {
		if o != nil {
			if err := o.call("create_group", map[string]any{"store": dir, "path": g.path, "zarr_format": format, "attributes": g.attrs}, nil); err != nil {
				t.Fatal(err)
			}
		} else if _, err := zarr.CreateGroup(ctx, gs, g.path, zarr.GroupSpec{ZarrFormat: format, Attributes: g.attrs}, &zarr.CreateOptions{Overwrite: true}); err != nil {
			t.Fatal(err)
		}
	}
	for _, a := range hierarchyArrays {
		c := hierarchyCase(a.c, format)
		if o != nil {
			if err := o.call("create", map[string]any{"store": dir, "path": a.path, "case": c}, nil); err != nil {
				t.Fatal(err)
			}
			continue
		}
		spec, err := c.goSpec()
		if err != nil {
			t.Fatal(err)
		}
		if _, err := zarr.CreateArray(ctx, gs, a.path, spec, nil); err != nil {
			t.Fatal(err)
		}
	}
}

// TestHierarchy builds the same hierarchy with both implementations
// (groups, nested arrays, implicit parents, attributes), consolidates its
// metadata, and compares every file and both views of its members.
func TestHierarchy(t *testing.T) {
	for _, format := range []int{3, 2} {
		t.Run(fmt.Sprintf("v%d", format), func(t *testing.T) {
			o := pythonOracle(t)
			ctx := context.Background()
			py, gd := t.TempDir(), t.TempDir()
			buildHierarchy(t, o, py, format)
			buildHierarchy(t, nil, gd, format)
			compareDirs(t, py, gd, "hierarchy")
			compareMembers(t, o, py, gd, false)

			if err := o.call("consolidate", map[string]any{"store": py}, nil); err != nil {
				t.Fatal(err)
			}
			if _, err := zarr.ConsolidateMetadata(ctx, zarr.NewLocalStore(gd), ""); err != nil {
				t.Fatal(err)
			}
			compareDirs(t, py, gd, "consolidated")
			compareMembers(t, o, py, gd, true)
		})
	}
}

func compareDirs(t *testing.T, a, b, when string) {
	t.Helper()
	fa, fb := files(t, a), files(t, b)
	var ka, kb []string
	for k := range fa {
		ka = append(ka, k)
	}
	for k := range fb {
		kb = append(kb, k)
	}
	slices.Sort(ka)
	slices.Sort(kb)
	if !slices.Equal(ka, kb) {
		t.Fatalf("%s: files differ\npython: %v\ngo:     %v", when, ka, kb)
	}
	for _, k := range ka {
		if string(fa[k]) != string(fb[k]) {
			t.Fatalf("%s: %s differs:\n%s", when, k, lineDiff(string(fa[k]), string(fb[k])))
		}
	}
}

type member struct {
	Type       string         `json:"type"`
	Attributes map[string]any `json:"attributes"`
}

type listing struct {
	Attributes map[string]any    `json:"attributes"`
	Members    map[string]member `json:"members"`
}

func norm(v any) string { b, _ := json.Marshal(v); return string(b) }

// pyListing lists a hierarchy with zarr-python.
func pyListing(t *testing.T, o *oracle, dir string, consolidated bool) string {
	t.Helper()
	var l listing
	if err := o.call("members", map[string]any{"store": dir, "use_consolidated": consolidated}, &l); err != nil {
		t.Fatalf("python listing %s: %v", dir, err)
	}
	return norm(l)
}

// goListing lists a hierarchy with Go.
func goListing(t *testing.T, dir string, consolidated bool) string {
	t.Helper()
	root, err := zarr.OpenGroup(context.Background(), zarr.NewLocalStore(dir), "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if root.Consolidated() != consolidated {
		t.Fatalf("go: consolidated = %v", root.Consolidated())
	}
	g := listing{Attributes: root.Attributes(), Members: map[string]member{}}
	err = root.Walk(context.Background(), func(rel string, n zarr.Node) error {
		typ := "group"
		if _, ok := n.(*zarr.Array); ok {
			typ = "array"
		}
		g.Members[rel] = member{Type: typ, Attributes: n.Attributes()}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return norm(g)
}

// compareMembers lists the hierarchy with zarr-python (on both stores)
// and with Go (on both stores).
func compareMembers(t *testing.T, o *oracle, py, gd string, consolidated bool) {
	t.Helper()
	want := pyListing(t, o, py, consolidated)
	if got := pyListing(t, o, gd, consolidated); got != want {
		t.Fatalf("python lists the stores differently:\n%s\n%s", want, got)
	}
	for _, dir := range []string{py, gd} {
		if got := goListing(t, dir, consolidated); got != want {
			t.Fatalf("go lists %s differently:\npython: %s\ngo:     %s", dir, want, got)
		}
	}
}

// TestHierarchyFixtures replays the recorded hierarchies without Python.
func TestHierarchyFixtures(t *testing.T) {
	b, err := os.ReadFile(filepath.Join(fixtureDir, "hierarchy.json"))
	if os.IsNotExist(err) || *record {
		t.Skip("no recorded hierarchies")
	}
	if err != nil {
		t.Fatal(err)
	}
	var listings map[string]string
	if err := json.Unmarshal(b, &listings); err != nil {
		t.Fatal(err)
	}
	for _, format := range []int{3, 2} {
		t.Run(fmt.Sprintf("v%d", format), func(t *testing.T) {
			gd := t.TempDir()
			buildHierarchy(t, nil, gd, format)
			name := fmt.Sprintf("hierarchy-v%d", format)
			compareDirs(t, unzipStore(t, name), gd, "hierarchy")
			if got := goListing(t, gd, false); got != listings[name] {
				t.Fatalf("listing:\npython: %s\ngo:     %s", listings[name], got)
			}
			if _, err := zarr.ConsolidateMetadata(context.Background(), zarr.NewLocalStore(gd), ""); err != nil {
				t.Fatal(err)
			}
			compareDirs(t, unzipStore(t, name+"-consolidated"), gd, "consolidated")
			if got := goListing(t, gd, true); got != listings[name+"-consolidated"] {
				t.Fatalf("consolidated listing:\npython: %s\ngo:     %s", listings[name+"-consolidated"], got)
			}
		})
	}
}

// recordHierarchies records zarr-python's hierarchies and listings.
func recordHierarchies(t *testing.T, o *oracle) {
	listings := map[string]string{}
	for _, format := range []int{3, 2} {
		name := fmt.Sprintf("hierarchy-v%d", format)
		dir, _ := filepath.Abs(filepath.Join(fixtureDir, name))
		buildHierarchy(t, o, dir, format)
		listings[name] = pyListing(t, o, dir, false)
		cdir := dir + "-consolidated"
		if err := os.CopyFS(cdir, os.DirFS(dir)); err != nil {
			t.Fatal(err)
		}
		if err := o.call("consolidate", map[string]any{"store": cdir}, nil); err != nil {
			t.Fatal(err)
		}
		listings[name+"-consolidated"] = pyListing(t, o, cdir, true)
		for _, d := range []string{dir, cdir} {
			if err := zipDir(d); err != nil {
				t.Fatal(err)
			}
		}
	}
	b, _ := json.MarshalIndent(listings, "", "  ")
	if err := os.WriteFile(filepath.Join(fixtureDir, "hierarchy.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}
