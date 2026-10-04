package xz

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/ulikunitz/xz/lzma"
)

// maxRatio bounds how much output one byte of LZMA input can produce: with
// probabilities saturated at 2017/2048 the cheapest symbol (a 273-byte rep0
// match, 14 binary decisions) costs ~0.31 bits, i.e. ~7100 output bytes per
// input byte. Dictionaries never need to be larger than
// maxRatio*len(input), which bounds memory for small hostile inputs that
// declare huge dictionaries. The factor 32768 leaves a wide safety margin.
const maxRatio = 1 << 15

const minDictCap = 4096

// lzma2BufSize is the encoder lookahead buffer for LZMA2: an uncompressed
// chunk holds at most 64 KiB + 2 bytes, so this guarantees the copy-back.
const lzma2BufSize = 1<<16 + 1<<12

// sizes describes what is known about the output size of an LZMA stream.
type sizes struct {
	known int64 // exact size from the stream itself (header), or -1
	hint  int64 // expected size from elsewhere (the .xz index), or -1
}

var unknownSizes = sizes{-1, -1}

// dictCaps returns the dictionary capacity to try first and the largest one
// that can be needed. A dictionary never needs to exceed the output size, so
// full is the declared size capped by the known size and by
// maxRatio*len(input). With only an index hint, both are the hinted size.
// With nothing known, first is a modest multiple of the input size; a
// too-small dictionary can only make decoding fail (a match distance beyond
// it is rejected), never produce wrong output, so callers retry with full
// after such a failure.
func dictCaps(declared uint32, sz sizes, inputLen int) (first, full int) {
	clamp := func(c int64) int {
		if c < minDictCap {
			c = minDictCap
		}
		return int(c)
	}
	f := int64(declared)
	if sz.known >= 0 && sz.known < f {
		f = sz.known
	}
	if b := int64(inputLen) * maxRatio; b < f {
		f = b
	}
	full = clamp(f)
	switch {
	case sz.known >= 0:
		return full, full
	case sz.hint >= 0:
		// The index size is exact for any valid stream, and the index is
		// validated after decoding, so no retry: a block that needs more
		// is corrupt either way.
		first = clamp(min(f, sz.hint))
		return first, first
	default:
		first = clamp(min(f, max(1<<16, 16*int64(inputLen))))
	}
	return first, full
}

// outputHint is the output size to preallocate: the expected size, but
// not more than max(1 MiB, 64*inputLen) so that a lying header or index
// cannot force a huge allocation up front (larger outputs grow normally).
func outputHint(sz sizes, inputLen int) int {
	h := sz.known
	if h < 0 {
		h = sz.hint
	}
	if h < 0 {
		return 0
	}
	return int(min(h, max(1<<20, 64*int64(inputLen))))
}

// withRetry runs decode with the first dictionary capacity and, if that
// fails because a match reached beyond the dictionary (the only way a
// too-small dictionary shows up in github.com/ulikunitz/xz/lzma: its
// decoderDict.writeMatch rejects the distance), again with the full one.
// TestFarMatchRetry covers this path.
func withRetry(out []byte, first, full int, decode func(out []byte, dictCap int) ([]byte, int, error)) ([]byte, int, error) {
	start := len(out)
	res, n, err := decode(out, first)
	if err != nil && first < full && errors.Is(err, ErrCorrupt) && strings.Contains(err.Error(), "distance out of range") {
		res, n, err = decode(res[:start], full)
	}
	return res, n, err
}

// panicError wraps a panic raised inside the third-party LZMA code so that
// corrupt input can never crash the caller.
type panicError struct{ v any }

func (p *panicError) Error() string {
	return fmt.Sprintf("xz: internal error in LZMA decoder: %v", p.v)
}
func (p *panicError) Unwrap() error { return ErrCorrupt }

// mapLZMAError classifies errors from the ulikunitz decoder.
func mapLZMAError(err error) error {
	switch {
	case err == nil:
		return nil
	case errors.Is(err, io.ErrUnexpectedEOF), errors.Is(err, io.EOF):
		return ErrTruncated
	case errors.Is(err, ErrCorrupt), errors.Is(err, ErrTruncated), errors.Is(err, ErrMemLimit):
		return err
	}
	return errorf(ErrCorrupt, "%v", err)
}

// readAll drains r into out. limit >= 0 is the exact number of bytes the
// stream must produce (more is an error, reported as soon as it happens).
func readAll(out []byte, r io.Reader, limit int64, hint int) (res []byte, err error) {
	defer func() {
		if p := recover(); p != nil {
			res, err = out, &panicError{p}
		}
	}()
	start := len(out)
	if hint > 0 && cap(out)-len(out) < hint {
		grown := make([]byte, len(out), len(out)+hint)
		copy(grown, out)
		out = grown
	}
	for {
		if cap(out)-len(out) < 4096 {
			out = append(out, make([]byte, 4096+len(out)/2)...)[:len(out)]
		}
		buf := out[len(out):cap(out)]
		if limit >= 0 {
			// Ask for at most one byte beyond the limit.
			if rem := limit - int64(len(out)-start) + 1; int64(len(buf)) > rem {
				buf = buf[:rem]
			}
		}
		n, rerr := r.Read(buf)
		out = out[:len(out)+n]
		if limit >= 0 && int64(len(out)-start) > limit {
			return out, errorf(ErrCorrupt, "uncompressed data exceeds the declared size %d", limit)
		}
		if rerr == io.EOF {
			return out, nil
		}
		if rerr != nil {
			return out, mapLZMAError(rerr)
		}
		if n == 0 {
			return out, errorf(ErrCorrupt, "LZMA decoder made no progress")
		}
	}
}

// decodeLZMA2 decodes an LZMA2 chunk sequence (up to and including its end
// marker) from the start of in, appending the output to out. It returns the
// number of input bytes consumed.
func decodeLZMA2(out, in []byte, dictSize uint32, sz sizes) ([]byte, int, error) {
	if dictSize > MaxDictSize {
		return out, 0, errorf(ErrMemLimit, "LZMA2 dictionary size %d exceeds %d", dictSize, uint32(MaxDictSize))
	}
	if len(in) == 0 {
		return out, 0, ErrTruncated
	}
	first, full := dictCaps(dictSize, sz, len(in))
	return withRetry(out, first, full, func(out []byte, dictCap int) (res []byte, n int, err error) {
		start := len(out)
		br := bytes.NewReader(in)
		var r *lzma.Reader2
		func() {
			defer func() {
				if p := recover(); p != nil {
					err = &panicError{p}
				}
			}()
			r, err = lzma.Reader2Config{DictCap: dictCap}.NewReader2(br)
		}()
		if err != nil {
			return out, 0, mapLZMAError(err)
		}
		out, err = readAll(out, r, sz.known, outputHint(sz, len(in)))
		if err != nil {
			return out, 0, err
		}
		if !r.EOS() {
			return out, 0, ErrTruncated
		}
		if sz.known >= 0 && int64(len(out)-start) != sz.known {
			return out, 0, errorf(ErrCorrupt, "uncompressed size %d does not match the declared %d", len(out)-start, sz.known)
		}
		return out, len(in) - br.Len(), nil
	})
}

// lzmaProps converts and validates lc/lp/pb.
func lzmaProps(o lzmaOptions) (lzma.Properties, error) {
	if o.lc > 4 || o.lp > 4 || o.lc+o.lp > 4 || o.pb > 4 {
		return lzma.Properties{}, errorf(ErrUnsupported, "invalid lc/lp/pb (%d/%d/%d)", o.lc, o.lp, o.pb)
	}
	return lzma.Properties{LC: int(o.lc), LP: int(o.lp), PB: int(o.pb)}, nil
}

// decodeLZMA1 decodes a headerless LZMA1 stream from the start of in,
// appending to out. size < 0 means unknown (an end marker is required);
// otherwise the stream ends after size bytes (an end marker may follow).
// It returns the number of input bytes consumed.
func decodeLZMA1(out, in []byte, o lzmaOptions, size int64) (res []byte, n int, err error) {
	props, err := lzmaProps(o)
	if err != nil {
		return out, 0, err
	}
	if o.dictSize > MaxDictSize {
		return out, 0, errorf(ErrMemLimit, "LZMA dictionary size %d exceeds %d", o.dictSize, uint32(MaxDictSize))
	}
	if size == 0 {
		// The ulikunitz decoder always decodes one symbol before checking
		// the size; liblzma accepts a finished range coder (or an end
		// marker) here, so handle that case directly.
		if len(in) < 5 {
			return out, 0, ErrTruncated
		}
		if in[0] != 0 {
			return out, 0, errorf(ErrCorrupt, "invalid LZMA range coder start")
		}
		if in[1]|in[2]|in[3]|in[4] == 0 {
			return out, 5, nil
		}
		// Otherwise only an immediate end marker is acceptable.
		start := len(out)
		out, n, err = decodeLZMA1(out, in, o, -1)
		if err == nil && len(out) != start {
			return out[:start], 0, errorf(ErrCorrupt, "data beyond the declared size 0")
		}
		return out, n, err
	}
	sz := sizes{known: size, hint: -1}
	first, full := dictCaps(o.dictSize, sz, len(in))
	return withRetry(out, first, full, func(out []byte, dictCap int) (res []byte, n int, err error) {
		pr := &prefixReader{data: in}
		pr.hdr[0] = props.Code()
		binary.LittleEndian.PutUint32(pr.hdr[1:5], uint32(dictCap))
		if size < 0 {
			binary.LittleEndian.PutUint64(pr.hdr[5:13], ^uint64(0))
		} else {
			binary.LittleEndian.PutUint64(pr.hdr[5:13], uint64(size))
		}
		var r *lzma.Reader
		func() {
			defer func() {
				if p := recover(); p != nil {
					err = &panicError{p}
				}
			}()
			r, err = lzma.ReaderConfig{DictCap: dictCap}.NewReader(pr)
		}()
		if err != nil {
			return out, 0, mapLZMAError(err)
		}
		start := len(out)
		out, err = readAll(out, r, size, outputHint(sz, len(in)))
		if err != nil {
			return out, 0, err
		}
		if size >= 0 && int64(len(out)-start) != size {
			return out, 0, errorf(ErrCorrupt, "uncompressed size %d does not match the declared %d", len(out)-start, size)
		}
		return out, pr.pos, nil
	})
}

// prefixReader serves a synthesized 13-byte .lzma header followed by data,
// without copying data. It implements io.ByteReader so that the LZMA
// decoder reads exactly the bytes it needs; pos is the number of data bytes
// consumed.
type prefixReader struct {
	hdr  [lzma.HeaderLen]byte
	hpos int
	data []byte
	pos  int
}

func (r *prefixReader) Read(p []byte) (int, error) {
	n := 0
	if r.hpos < len(r.hdr) {
		n = copy(p, r.hdr[r.hpos:])
		r.hpos += n
	}
	if n < len(p) {
		k := copy(p[n:], r.data[r.pos:])
		r.pos += k
		n += k
	}
	if n == 0 && len(p) > 0 {
		return 0, io.EOF
	}
	return n, nil
}

func (r *prefixReader) ReadByte() (byte, error) {
	if r.hpos < len(r.hdr) {
		r.hpos++
		return r.hdr[r.hpos-1], nil
	}
	if r.pos >= len(r.data) {
		return 0, io.EOF
	}
	r.pos++
	return r.data[r.pos-1], nil
}

// encodeDictCap returns the dictionary capacity for the encoder: the
// requested size shrunk to the input size (matches can never reach further
// back than the start of the input), but at least 4 KiB.
func encodeDictCap(requested uint32, n int) int {
	c := int64(requested)
	if int64(n) < c {
		c = int64(n)
	}
	if c < minDictCap {
		c = minDictCap
	}
	return int(c)
}

// encodeLZMA2 appends an LZMA2 chunk sequence (with end marker) to out and
// returns the dictionary size the decoder needs.
func encodeLZMA2(out, data []byte, o lzmaOptions) ([]byte, uint32, error) {
	props, err := lzmaProps(o)
	if err != nil {
		return out, 0, err
	}
	dictCap := encodeDictCap(o.dictSize, len(data))
	buf := bytes.NewBuffer(out)
	// When a chunk turns out incompressible, Writer2 re-reads its input
	// (up to ~64 KiB) from the encoder's ring buffer, which holds
	// DictCap+BufSize bytes; with small dictionaries the default 4 KiB
	// BufSize is too small for that ("insufficient space"). BufSize does
	// not affect match distances, which stay within DictCap.
	cfg := lzma.Writer2Config{Properties: &props, DictCap: dictCap, BufSize: lzma2BufSize, Matcher: lzma.HashTable4}
	w, err := cfg.NewWriter2(buf)
	if err != nil {
		return out, 0, errorf(ErrUnsupported, "%v", err)
	}
	if _, err = w.Write(data); err != nil {
		return out, 0, err
	}
	if err = w.Close(); err != nil {
		return out, 0, err
	}
	return buf.Bytes(), uint32(dictCap), nil
}

// roundDictSize rounds up to the next 2^n or 2^n + 2^(n-1), as liblzma
// does for the .lzma header and as the LZMA2 property byte requires.
func roundDictSize(d uint32) uint32 {
	x := d - 1
	x |= x >> 2
	x |= x >> 3
	x |= x >> 4
	x |= x >> 8
	x |= x >> 16
	if x != 0xFFFFFFFF {
		x++
	}
	return x
}

// encodeLZMA1 returns an LZMA1 stream with end marker; with header it is a
// complete .lzma file (unknown size in the header, as liblzma writes).
func encodeLZMA1(data []byte, o lzmaOptions, header bool) ([]byte, error) {
	props, err := lzmaProps(o)
	if err != nil {
		return nil, err
	}
	dictCap := encodeDictCap(o.dictSize, len(data))
	var buf bytes.Buffer
	buf.Grow(len(data)/2 + 64)
	w, err := lzma.WriterConfig{Properties: &props, DictCap: dictCap, Matcher: lzma.HashTable4, EOSMarker: true}.NewWriter(&buf)
	if err != nil {
		return nil, errorf(ErrUnsupported, "%v", err)
	}
	if _, err = w.Write(data); err != nil {
		return nil, err
	}
	if err = w.Close(); err != nil {
		return nil, err
	}
	b := buf.Bytes()
	if !header {
		return b[lzma.HeaderLen:], nil
	}
	binary.LittleEndian.PutUint32(b[1:5], roundDictSize(uint32(dictCap)))
	binary.LittleEndian.PutUint64(b[5:13], ^uint64(0))
	return b, nil
}

// pickyDictSize is liblzma's .lzma auto-detection test: the dictionary
// size must be 2^n or 2^n + 2^(n-1). (Like liblzma's, it also accepts 0,
// through unsigned wrap-around.)
func pickyDictSize(d uint32) bool {
	x := d - 1
	x |= x >> 2
	x |= x >> 3
	x |= x >> 4
	x |= x >> 8
	x |= x >> 16
	return x+1 == d
}

func encodeAlone(data []byte, o lzmaOptions) ([]byte, error) {
	return encodeLZMA1(data, o, true)
}

// decodeAlone decodes one .lzma stream from the start of in. picky applies
// the extra header sanity checks liblzma uses when auto-detecting the format.
func decodeAlone(out, in []byte, picky bool) ([]byte, int, error) {
	// Header fields are validated as soon as they are available, like
	// liblzma, so that trailing garbage is classified like Python does.
	if len(in) < 1 {
		return out, 0, ErrTruncated
	}
	p := in[0]
	if p > (4*5+4)*9+8 {
		return out, 0, errorf(ErrFormat, ".lzma properties byte 0x%02x", p)
	}
	o := lzmaOptions{pb: uint32(p) / 45, lp: uint32(p) % 45 / 9, lc: uint32(p) % 9}
	if o.lc+o.lp > 4 {
		return out, 0, errorf(ErrFormat, ".lzma properties byte 0x%02x (lc+lp > 4)", p)
	}
	if len(in) < 5 {
		return out, 0, ErrTruncated
	}
	o.dictSize = binary.LittleEndian.Uint32(in[1:5])
	if picky && o.dictSize != 0xFFFFFFFF && !pickyDictSize(o.dictSize) {
		return out, 0, errorf(ErrFormat, ".lzma dictionary size %d", o.dictSize)
	}
	if len(in) < 13 {
		return out, 0, ErrTruncated
	}
	size := int64(-1)
	if s := binary.LittleEndian.Uint64(in[5:13]); s != ^uint64(0) {
		if picky && s >= 1<<38 {
			return out, 0, errorf(ErrFormat, ".lzma uncompressed size %d", s)
		}
		if s >= 1<<62 {
			return out, 0, errorf(ErrUnsupported, ".lzma uncompressed size %d", s)
		}
		size = int64(s)
	}
	if o.dictSize > MaxDictSize {
		return out, 0, errorf(ErrMemLimit, ".lzma dictionary size %d exceeds %d", o.dictSize, uint32(MaxDictSize))
	}
	out, n, err := decodeLZMA1(out, in[13:], o, size)
	if err != nil {
		return out, 0, err
	}
	return out, 13 + n, nil
}
