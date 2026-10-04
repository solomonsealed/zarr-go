package zarr

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// LocalStore stores keys as files below a directory, the layout
// zarr-python's LocalStore uses. Writes go to a temporary file that is
// then renamed, so readers never see partial values.
type LocalStore struct {
	root     string
	readOnly bool
}

// NewLocalStore returns a store rooted at dir (created on first write).
func NewLocalStore(dir string) *LocalStore { return &LocalStore{root: dir} }

// NewLocalStoreReadOnly returns a store rooted at dir that refuses writes.
func NewLocalStoreReadOnly(dir string) *LocalStore { return &LocalStore{root: dir, readOnly: true} }

// Root returns the store's directory.
func (s *LocalStore) Root() string { return s.root }

func (s *LocalStore) file(key string) (string, error) {
	if key == "" {
		return s.root, nil
	}
	if strings.HasPrefix(key, "/") || strings.Contains(key, "\\") {
		return "", fmt.Errorf("zarr: invalid key %q", key)
	}
	for _, part := range strings.Split(key, "/") {
		if part == "" || part == "." || part == ".." {
			return "", fmt.Errorf("zarr: invalid key %q", key)
		}
	}
	return filepath.Join(s.root, filepath.FromSlash(key)), nil
}

func (s *LocalStore) Get(_ context.Context, key string, r ByteRange) ([]byte, bool, error) {
	name, err := s.file(key)
	if err != nil {
		return nil, false, err
	}
	f, err := os.Open(name)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrInvalid) {
			return nil, false, nil
		}
		return nil, false, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	if fi.IsDir() {
		return nil, false, nil
	}
	start, end, err := r.resolve(fi.Size())
	if err != nil {
		return nil, true, err
	}
	buf := make([]byte, end-start)
	if _, err := f.ReadAt(buf, start); err != nil && err != io.EOF {
		return nil, true, err
	}
	return buf, true, nil
}

func (s *LocalStore) Set(_ context.Context, key string, value []byte) error {
	if s.readOnly {
		return ErrReadOnly
	}
	name, err := s.file(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(name), "."+filepath.Base(name)+".tmp*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(value); err != nil {
		tmp.Close()
		os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	if err := os.Rename(tmp.Name(), name); err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return nil
}

// Delete removes a key; deleting a directory removes everything below it,
// as zarr-python's LocalStore does.
func (s *LocalStore) Delete(_ context.Context, key string) error {
	if s.readOnly {
		return ErrReadOnly
	}
	name, err := s.file(key)
	if err != nil {
		return err
	}
	fi, err := os.Stat(name)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if fi.IsDir() {
		return os.RemoveAll(name)
	}
	if err := os.Remove(name); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	return nil
}

func (s *LocalStore) List(_ context.Context, prefix string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(s.root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(s.root, p)
		if err != nil {
			return err
		}
		key := filepath.ToSlash(rel)
		if isTempName(d.Name()) {
			return nil
		}
		if strings.HasPrefix(key, prefix) {
			out = append(out, key)
		}
		return nil
	})
	slices.Sort(out)
	return out, err
}

func isTempName(name string) bool {
	return strings.HasPrefix(name, ".") && strings.Contains(name, ".tmp")
}

func (s *LocalStore) ListDir(_ context.Context, prefix string) ([]string, error) {
	dir, err := s.file(strings.TrimSuffix(prefix, "/"))
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if !isTempName(e.Name()) {
			out = append(out, e.Name())
		}
	}
	slices.Sort(out)
	return out, nil
}

// ---------------------------------------------------------------------------
// FSStore

// FSStore is a read-only store over an io/fs.FS, such as os.DirFS or an
// embed.FS holding a Zarr hierarchy.
type FSStore struct{ fsys fs.FS }

// NewFSStore returns a store reading from fsys.
func NewFSStore(fsys fs.FS) *FSStore { return &FSStore{fsys: fsys} }

func (s *FSStore) Get(_ context.Context, key string, r ByteRange) ([]byte, bool, error) {
	if !fs.ValidPath(key) {
		return nil, false, nil
	}
	f, err := s.fsys.Open(key)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	if fi.IsDir() {
		return nil, false, nil
	}
	if ra, ok := f.(io.ReaderAt); ok && !r.IsFull() {
		start, end, err := r.resolve(fi.Size())
		if err != nil {
			return nil, true, err
		}
		buf := make([]byte, end-start)
		if _, err := ra.ReadAt(buf, start); err != nil && err != io.EOF {
			return nil, true, err
		}
		return buf, true, nil
	}
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, true, err
	}
	b, err := r.Slice(data)
	return b, true, err
}

func (s *FSStore) List(_ context.Context, prefix string) ([]string, error) {
	var out []string
	err := fs.WalkDir(s.fsys, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasPrefix(p, prefix) {
			out = append(out, p)
		}
		return nil
	})
	slices.Sort(out)
	return out, err
}

func (s *FSStore) ListDir(_ context.Context, prefix string) ([]string, error) {
	dir := strings.TrimSuffix(prefix, "/")
	if dir == "" {
		dir = "."
	}
	entries, err := fs.ReadDir(s.fsys, dir)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	out := make([]string, len(entries))
	for i, e := range entries {
		out[i] = e.Name()
	}
	slices.Sort(out)
	return out, nil
}
