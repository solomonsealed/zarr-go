package blosc

import (
	"bytes"
	"compress/zlib"
	"fmt"
	"io"
	"sync"

	"github.com/klauspost/compress/snappy"
	"github.com/klauspost/compress/zstd"
)

// ---------------------------------------------------------------------------
// Snappy (raw block format, as snappy_compress / snappy_uncompress).

// snappyMaxCompressedLength is snappy_max_compressed_length.
func snappyMaxCompressedLength(n int) int { return 32 + n + n/6 }

// snappyCompress mirrors snappy_wrap_compress: snappy_compress refuses to
// run (SNAPPY_BUFFER_TOO_SMALL) unless the output buffer can hold
// snappy_max_compressed_length bytes.
func snappyCompress(in, dst []byte) int {
	if len(dst) < snappyMaxCompressedLength(len(in)) {
		return 0
	}
	var enc []byte
	if len(dst) >= snappy.MaxEncodedLen(len(in)) {
		enc = snappy.Encode(dst, in) // in place
	} else {
		enc = snappy.Encode(nil, in)
	}
	if len(enc) > len(dst) {
		return 0
	}
	if len(enc) > 0 && &enc[0] != &dst[0] {
		copy(dst, enc)
	}
	return len(enc)
}

func snappyDecompress(src, dst []byte) error {
	if n, err := snappy.DecodedLen(src); err != nil || n != len(dst) {
		return fmt.Errorf("%w: snappy split decodes to %d bytes, want %d (%v)", ErrCorrupt, n, len(dst), err)
	}
	out, err := snappy.Decode(dst, src)
	if err != nil {
		return fmt.Errorf("%w: snappy: %v", ErrCorrupt, err)
	}
	if len(out) != len(dst) {
		return fmt.Errorf("%w: snappy split decoded to %d bytes, want %d", ErrCorrupt, len(out), len(dst))
	}
	return nil
}

// ---------------------------------------------------------------------------
// Zlib (compress2 / uncompress).

var (
	zlibWriters [11]sync.Pool // levels 0-9; index 10 is zlib.HuffmanOnly
	zlibBufs    = sync.Pool{New: func() any { return new(bytes.Buffer) }}
)

// zlibEncode writes the zlib stream of in at the given level into buf.
func zlibEncode(buf *bytes.Buffer, in []byte, level int) error {
	buf.Reset()
	i := level
	if level == zlib.HuffmanOnly {
		i = 10
	}
	pooled := i >= 0 && i < len(zlibWriters)
	var w *zlib.Writer
	if pooled {
		if v := zlibWriters[i].Get(); v != nil {
			w = v.(*zlib.Writer)
			w.Reset(buf)
		}
	}
	if w == nil {
		var err error
		if w, err = zlib.NewWriterLevel(buf, level); err != nil {
			return fmt.Errorf("blosc: zlib: %w", err)
		}
	}
	if _, err := w.Write(in); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	if pooled {
		zlibWriters[i].Put(w)
	}
	return nil
}

// zlibCompress mirrors zlib_wrap_compress (compress2 into a buffer of
// len(dst) bytes, 0 if it does not fit).
func zlibCompress(in, dst []byte, level int) (int, error) {
	buf := zlibBufs.Get().(*bytes.Buffer)
	defer zlibBufs.Put(buf)
	if err := zlibEncode(buf, in, level); err != nil {
		return 0, err
	}
	if buf.Len() >= len(in) {
		// Go's deflate emits stored blocks for match-less data at low
		// levels, where C zlib still Huffman-codes the literals (e.g. byte
		// planes of shuffled numeric data). Retry Huffman-only before
		// letting blosc store the split uncompressed.
		huff := zlibBufs.Get().(*bytes.Buffer)
		defer zlibBufs.Put(huff)
		if err := zlibEncode(huff, in, zlib.HuffmanOnly); err != nil {
			return 0, err
		}
		if huff.Len() < buf.Len() {
			buf = huff
		}
	}
	if buf.Len() > len(dst) {
		return 0, nil // Z_BUF_ERROR in compress2
	}
	return copy(dst, buf.Bytes()), nil
}

type zlibReader struct {
	br bytes.Reader
	zr io.ReadCloser
}

var zlibReaders sync.Pool

func zlibDecompress(src, dst []byte) error {
	r, _ := zlibReaders.Get().(*zlibReader)
	if r == nil {
		r = new(zlibReader)
	}
	r.br.Reset(src)
	var err error
	if r.zr == nil {
		r.zr, err = zlib.NewReader(&r.br)
	} else {
		err = r.zr.(zlib.Resetter).Reset(&r.br, nil)
	}
	if err != nil {
		r.zr = nil
		zlibReaders.Put(r)
		return fmt.Errorf("%w: zlib: %v", ErrCorrupt, err)
	}
	defer zlibReaders.Put(r)
	if _, err := io.ReadFull(r.zr, dst); err != nil {
		return fmt.Errorf("%w: zlib split shorter than %d bytes: %v", ErrCorrupt, len(dst), err)
	}
	// The stream must end here (and its checksum must verify).
	var one [1]byte
	if n, err := r.zr.Read(one[:]); n != 0 || err != io.EOF {
		if err == nil || err == io.EOF {
			return fmt.Errorf("%w: zlib split longer than %d bytes", ErrCorrupt, len(dst))
		}
		return fmt.Errorf("%w: zlib: %v", ErrCorrupt, err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Zstd (ZSTD_compress / ZSTD_decompress).

var zstdEncoders [23]struct {
	once sync.Once
	enc  *zstd.Encoder
	err  error
}

func zstdEncoder(level int) (*zstd.Encoder, error) {
	level = min(max(level, 1), len(zstdEncoders)-1)
	e := &zstdEncoders[level]
	e.once.Do(func() {
		e.enc, e.err = zstd.NewWriter(nil,
			zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(level)),
			zstd.WithEncoderCRC(false), // ZSTD_compress writes no checksum
			// Like C zstd, entropy-code the literals of blocks without
			// matches (klauspost skips this below SpeedBetterCompression,
			// which makes e.g. shuffled text incompressible).
			zstd.WithAllLitEntropyCompression(true),
			zstd.WithEncoderConcurrency(1))
	})
	return e.enc, e.err
}

func zstdCompress(in, dst []byte, level int) (int, error) {
	enc, err := zstdEncoder(level)
	if err != nil {
		return 0, fmt.Errorf("blosc: zstd: %w", err)
	}
	out := enc.EncodeAll(in, dst[:0:len(dst)])
	if len(out) > len(dst) {
		return 0, nil // dstSize_tooSmall
	}
	if len(out) > 0 && &out[0] != &dst[0] {
		copy(dst, out)
	}
	return len(out), nil
}

var zstdDec struct {
	once sync.Once
	dec  *zstd.Decoder
	err  error
}

func zstdDecompress(src, dst []byte) error {
	zstdDec.once.Do(func() {
		zstdDec.dec, zstdDec.err = zstd.NewReader(nil,
			zstd.WithDecoderConcurrency(0),
			zstd.WithDecodeAllCapLimit(true)) // never decode more than cap(dst)
	})
	if zstdDec.err != nil {
		return fmt.Errorf("blosc: zstd: %w", zstdDec.err)
	}
	out, err := zstdDec.dec.DecodeAll(src, dst[:0:len(dst)])
	if err != nil {
		return fmt.Errorf("%w: zstd: %v", ErrCorrupt, err)
	}
	if len(out) != len(dst) {
		return fmt.Errorf("%w: zstd split decoded to %d bytes, want %d", ErrCorrupt, len(out), len(dst))
	}
	if &out[0] != &dst[0] {
		copy(dst, out)
	}
	return nil
}

// plausibleSplit reports whether a compressed split could decode to n
// bytes, using generous upper bounds on each format's expansion ratio. It
// lets Decompress reject frames whose header claims an absurd size before
// allocating the output.
func plausibleSplit(format int, split []byte, n int) bool {
	c, m := int64(len(split)), int64(n) // int64: no overflow on 32-bit platforms
	switch format {
	case formatBlosclz, formatLZ4:
		// A long match adds at most 255 bytes per input byte.
		return m <= 256*c+64
	case formatSnappy:
		dl, err := snappy.DecodedLen(split)
		return err == nil && dl == n
	case formatZlib:
		// Deflate cannot exceed ~1032:1.
		return m <= 1100*c+1024
	case formatZstd:
		// A 4-byte RLE block expands to at most 128 KiB.
		return m <= 32768*c
	}
	return false
}
