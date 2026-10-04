package zarr

import (
	"archive/zip"
	"context"
	"io"
	"os"
	"slices"
	"strings"
)

// ---------------------------------------------------------------------------
// ZipStore

// ZipStore reads a Zarr hierarchy from a zip archive, as zarr-python's
// ZipStore writes it, or builds one (CreateZipStore).
type ZipStore struct {
	ra     io.ReaderAt
	closer io.Closer
	files  map[string]*zip.File

	// Writing: values are kept in memory and the archive is written on
	// Close.
	mem *MemoryStore
	out io.WriteCloser
}

// OpenZipStore opens a zip archive for reading.
func OpenZipStore(path string) (*ZipStore, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	s, err := NewZipStoreReader(f, fi.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	s.closer = f
	return s, nil
}

// NewZipStoreReader reads a zip archive from r.
func NewZipStoreReader(r io.ReaderAt, size int64) (*ZipStore, error) {
	zr, err := zip.NewReader(r, size)
	if err != nil {
		return nil, err
	}
	s := &ZipStore{ra: r, files: map[string]*zip.File{}}
	for _, f := range zr.File {
		if !strings.HasSuffix(f.Name, "/") {
			s.files[strings.TrimPrefix(f.Name, "/")] = f // later duplicates win
		}
	}
	return s, nil
}

// CreateZipStore returns a writable store that writes a zip archive
// (entries stored uncompressed, like zarr-python's default) to path when
// closed.
func CreateZipStore(path string) (*ZipStore, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &ZipStore{mem: NewMemoryStore(), out: f}, nil
}

func (s *ZipStore) Get(ctx context.Context, key string, r ByteRange) ([]byte, bool, error) {
	if s.mem != nil {
		return s.mem.Get(ctx, key, r)
	}
	f, ok := s.files[key]
	if !ok {
		return nil, false, nil
	}
	if f.Method == zip.Store && !r.IsFull() {
		off, err := f.DataOffset()
		if err != nil {
			return nil, true, err
		}
		start, end, err := r.resolve(int64(f.UncompressedSize64))
		if err != nil {
			return nil, true, err
		}
		buf := make([]byte, end-start)
		if _, err := s.ra.ReadAt(buf, off+start); err != nil && err != io.EOF {
			return nil, true, err
		}
		return buf, true, nil
	}
	rc, err := f.Open()
	if err != nil {
		return nil, true, err
	}
	defer rc.Close()
	data, err := io.ReadAll(rc)
	if err != nil {
		return nil, true, err
	}
	b, err := r.Slice(data)
	return b, true, err
}

func (s *ZipStore) Set(ctx context.Context, key string, value []byte) error {
	if s.mem == nil {
		return ErrReadOnly
	}
	return s.mem.Set(ctx, key, value)
}

func (s *ZipStore) Delete(ctx context.Context, key string) error {
	if s.mem == nil {
		return ErrReadOnly
	}
	return s.mem.Delete(ctx, key)
}

func (s *ZipStore) List(ctx context.Context, prefix string) ([]string, error) {
	if s.mem != nil {
		return s.mem.List(ctx, prefix)
	}
	var out []string
	for k := range s.files {
		if strings.HasPrefix(k, prefix) {
			out = append(out, k)
		}
	}
	slices.Sort(out)
	return out, nil
}

func (s *ZipStore) ListDir(ctx context.Context, prefix string) ([]string, error) {
	keys, err := s.List(ctx, prefix)
	if err != nil {
		return nil, err
	}
	return listDirFromList(keys, prefix), nil
}

// Close writes the archive (for stores from CreateZipStore) and releases
// the file.
func (s *ZipStore) Close() error {
	if s.mem != nil && s.out != nil {
		zw := zip.NewWriter(s.out)
		keys, _ := s.mem.List(context.Background(), "")
		for _, k := range keys {
			v, _, _ := s.mem.Get(context.Background(), k, ByteRange{})
			w, err := zw.CreateHeader(&zip.FileHeader{Name: k, Method: zip.Store})
			if err != nil {
				return err
			}
			if _, err := w.Write(v); err != nil {
				return err
			}
		}
		if err := zw.Close(); err != nil {
			return err
		}
		err := s.out.Close()
		s.out = nil
		return err
	}
	if s.closer != nil {
		return s.closer.Close()
	}
	return nil
}
