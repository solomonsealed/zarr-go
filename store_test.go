package zarr

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// storeContract exercises the Store interfaces.
func storeContract(t *testing.T, s Store) {
	t.Helper()
	ctx := context.Background()
	ws, writable := s.(WritableStore)
	if !writable {
		t.Fatal("store is not writable")
	}
	if _, ok, err := s.Get(ctx, "missing", ByteRange{}); ok || err != nil {
		t.Fatalf("missing key: ok=%v err=%v", ok, err)
	}
	for k, v := range map[string]string{"zarr.json": "{}", "a/zarr.json": "{}", "a/c/0/1": "0123456789", "b/x": ""} {
		if err := ws.Set(ctx, k, []byte(v)); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		r    ByteRange
		want string
	}{{ByteRange{}, "0123456789"}, {ByteRange{Offset: 3, Length: 4}, "3456"}, {ByteRange{Offset: 7}, "789"}, {ByteRange{Offset: -2}, "89"}} {
		got, ok, err := s.Get(ctx, "a/c/0/1", tc.r)
		if err != nil || !ok || string(got) != tc.want {
			t.Errorf("Get(%v) = %q, %v, %v", tc.r, got, ok, err)
		}
	}
	if got, ok, err := s.Get(ctx, "b/x", ByteRange{}); !ok || err != nil || len(got) != 0 {
		t.Errorf("empty value: %q %v %v", got, ok, err)
	}
	if ls, ok := s.(ListableStore); ok {
		keys, err := ls.List(ctx, "a/")
		if err != nil || !slices.Equal(keys, []string{"a/c/0/1", "a/zarr.json"}) {
			t.Errorf("List = %v, %v", keys, err)
		}
		names, err := ls.ListDir(ctx, "")
		if err != nil || !slices.Equal(names, []string{"a", "b", "zarr.json"}) {
			t.Errorf("ListDir root = %v, %v", names, err)
		}
		names, err = ls.ListDir(ctx, "a/")
		if err != nil || !slices.Equal(names, []string{"c", "zarr.json"}) {
			t.Errorf("ListDir a/ = %v, %v", names, err)
		}
	}
	if err := ws.Delete(ctx, "a/c/0/1"); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := s.Get(ctx, "a/c/0/1", ByteRange{}); ok {
		t.Error("deleted key still present")
	}
	if err := ws.Delete(ctx, "never"); err != nil {
		t.Errorf("deleting a missing key: %v", err)
	}
}

func TestMemoryStore(t *testing.T) { storeContract(t, NewMemoryStore()) }

func TestLocalStore(t *testing.T) {
	dir := t.TempDir()
	storeContract(t, NewLocalStore(dir))
	s := NewLocalStore(dir)
	if _, _, err := s.Get(context.Background(), "../x", ByteRange{}); err == nil {
		t.Error("escaping key accepted")
	}
	if err := NewLocalStoreReadOnly(dir).Set(context.Background(), "k", nil); err == nil {
		t.Error("read-only store wrote")
	}
}

func TestFSAndZipStores(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	zpath := filepath.Join(dir, "a.zip")
	z, err := CreateZipStore(zpath)
	if err != nil {
		t.Fatal(err)
	}
	a, err := CreateArray(ctx, z, "g/arr", ArraySpec{Shape: []uint64{5, 4}, ChunkShape: []uint64{2, 3}, DataType: "int32"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	vals := make([]int32, 20)
	for i := range vals {
		vals[i] = int32(i * i)
	}
	data, _ := FromSlice([]uint64{5, 4}, vals)
	if err := a.Set(ctx, data); err != nil {
		t.Fatal(err)
	}
	if err := z.Close(); err != nil {
		t.Fatal(err)
	}
	zr, err := OpenZipStore(zpath)
	if err != nil {
		t.Fatal(err)
	}
	defer zr.Close()
	check := func(s Store) {
		t.Helper()
		arr, err := OpenArray(ctx, s, "g/arr", nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := arr.Get(ctx, SliceStep(0, 5, 2), Indices(3, 0))
		if err != nil {
			t.Fatal(err)
		}
		v, _ := got.Values()
		if !slices.Equal(v.([]int32), []int32{9, 0, 121, 64, 361, 256}) {
			t.Errorf("values = %v", v)
		}
		g, err := OpenGroup(ctx, s, "", nil)
		if err != nil {
			t.Fatal(err)
		}
		names, err := g.MemberNames(ctx)
		if err != nil || !slices.Equal(names, []string{"g"}) {
			t.Errorf("members = %v, %v", names, err)
		}
	}
	check(zr)
	// Unpack and read through os.DirFS.
	keys, _ := zr.List(ctx, "")
	for _, k := range keys {
		v, _, _ := zr.Get(ctx, k, ByteRange{})
		os.MkdirAll(filepath.Join(dir, "x", filepath.Dir(k)), 0o755)
		os.WriteFile(filepath.Join(dir, "x", k), v, 0o644)
	}
	check(NewFSStore(os.DirFS(filepath.Join(dir, "x"))))
}
