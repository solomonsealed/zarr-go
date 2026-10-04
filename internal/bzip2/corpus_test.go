package bzip2

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
)

// TestExternalCorpus compares Compress against an externally generated corpus
// of (level, input, expected output) records, e.g. one produced by Python's
// bz2 module. It is skipped unless BZIP2_CORPUS names one or more corpus
// files (separated by the OS path list separator).
//
// Record format (little-endian): u8 level, u32 len, input, u32 len, output.
func TestExternalCorpus(t *testing.T) {
	paths := os.Getenv("BZIP2_CORPUS")
	if paths == "" {
		t.Skip("BZIP2_CORPUS not set")
	}
	for _, path := range filepath.SplitList(paths) {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		r := bufio.NewReaderSize(f, 1<<20)
		n, failed := 0, 0
		var inBytes int64
		for {
			level, in, want, err := readRecord(r)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				t.Fatalf("%s: record %d: %v", path, n, err)
			}
			got, err := Compress(in, level)
			if err != nil {
				t.Fatalf("%s: record %d: %v", path, n, err)
			}
			if !bytes.Equal(got, want) {
				failed++
				if failed <= 20 {
					t.Errorf("%s: record %d (level %d, %d bytes in): output differs (got %d bytes, want %d; first diff at %d)",
						path, n, level, len(in), len(got), len(want), firstDiff(got, want))
				}
			}
			n++
			inBytes += int64(len(in))
		}
		f.Close()
		t.Logf("%s: %d records, %d input bytes, %d mismatches", path, n, inBytes, failed)
	}
}

func readRecord(r io.Reader) (level int, in, out []byte, err error) {
	var lv [1]byte
	if _, err = io.ReadFull(r, lv[:]); err != nil {
		return
	}
	level = int(lv[0])
	readBlob := func() ([]byte, error) {
		var l [4]byte
		if _, err := io.ReadFull(r, l[:]); err != nil {
			return nil, io.ErrUnexpectedEOF
		}
		b := make([]byte, binary.LittleEndian.Uint32(l[:]))
		if _, err := io.ReadFull(r, b); err != nil {
			return nil, io.ErrUnexpectedEOF
		}
		return b, nil
	}
	if in, err = readBlob(); err != nil {
		return
	}
	out, err = readBlob()
	return
}

func firstDiff(a, b []byte) int {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
