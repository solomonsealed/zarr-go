package xz

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"hash/crc32"
	"hash/crc64"
)

// The .xz container format, per "The .xz File Format" version 1.2.1.

var (
	streamMagic = []byte{0xFD, '7', 'z', 'X', 'Z', 0x00}
	footerMagic = []byte{'Y', 'Z'}
	crc64Table  = crc64.MakeTable(crc64.ECMA)
)

const (
	streamHeaderSize = 12
	vliMaxBytes      = 9
	maxBlockHeader   = 1024
)

// checkSize returns the size of the check field for a check ID (0-15).
func checkSize(id byte) int {
	return [16]int{0, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64}[id&0x0F]
}

// checkSupported reports whether a check ID can be computed (and therefore
// used for encoding). Unsupported IDs are skipped when decoding, as liblzma
// does by default.
func checkSupported(id int) bool {
	switch id {
	case CheckNone, CheckCRC32, CheckCRC64, CheckSHA256:
		return true
	}
	return false
}

// computeCheck returns the check field for data, or nil if the check ID is
// not supported (or is CheckNone).
func computeCheck(id byte, data []byte) []byte {
	switch id {
	case CheckCRC32:
		return binary.LittleEndian.AppendUint32(nil, crc32.ChecksumIEEE(data))
	case CheckCRC64:
		return binary.LittleEndian.AppendUint64(nil, crc64.Checksum(data, crc64Table))
	case CheckSHA256:
		s := sha256.Sum256(data)
		return s[:]
	}
	return nil
}

func appendVLI(b []byte, x uint64) []byte {
	for x >= 0x80 {
		b = append(b, byte(x)|0x80)
		x >>= 7
	}
	return append(b, byte(x))
}

// readVLI decodes a variable-length integer (at most 9 bytes, minimal
// encoding, value < 2^63).
func readVLI(b []byte) (uint64, int, error) {
	var x uint64
	for i := 0; i < vliMaxBytes; i++ {
		if i >= len(b) {
			return 0, 0, ErrTruncated
		}
		c := b[i]
		x |= uint64(c&0x7F) << (7 * i)
		if c&0x80 == 0 {
			if c == 0 && i > 0 {
				return 0, 0, errorf(ErrCorrupt, "non-minimal variable-length integer")
			}
			return x, i + 1, nil
		}
	}
	return 0, 0, errorf(ErrCorrupt, "variable-length integer too long")
}

// lzma2DictSize decodes the LZMA2 dictionary-size property byte.
func lzma2DictSize(b byte) (uint32, error) {
	if b > 40 {
		return 0, errorf(ErrUnsupported, "invalid LZMA2 dictionary size code %d", b)
	}
	if b == 40 {
		return 0xFFFFFFFF, nil
	}
	return (2 | uint32(b)&1) << (b/2 + 11), nil
}

// lzma2DictCode returns the smallest property code whose size is >= d.
func lzma2DictCode(d uint32) byte {
	for c := byte(0); c < 40; c++ {
		if s, _ := lzma2DictSize(c); s >= d {
			return c
		}
	}
	return 40
}

// filterFromProps builds a chain entry from .xz filter flags.
func filterFromProps(id uint64, props []byte) (chainFilter, error) {
	c := chainFilter{id: id}
	switch {
	case id == FilterLZMA2:
		if len(props) != 1 {
			return c, errorf(ErrUnsupported, "LZMA2 properties size %d", len(props))
		}
		d, err := lzma2DictSize(props[0])
		if err != nil {
			return c, err
		}
		if d > MaxDictSize {
			return c, errorf(ErrMemLimit, "LZMA2 dictionary size %d exceeds %d", d, uint32(MaxDictSize))
		}
		c.lzma.dictSize = d
	case id == FilterDelta:
		if len(props) != 1 {
			return c, errorf(ErrUnsupported, "delta properties size %d", len(props))
		}
		c.dist = int(props[0]) + 1
	case isBCJ(id):
		switch len(props) {
		case 0:
		case 4:
			c.start = binary.LittleEndian.Uint32(props)
		default:
			return c, errorf(ErrUnsupported, "%s properties size %d", filterName(id), len(props))
		}
	case id >= 1<<62:
		return c, errorf(ErrCorrupt, "reserved filter ID 0x%x in .xz block header", id)
	default:
		return c, errorf(ErrUnsupported, "unsupported filter %s", filterName(id))
	}
	return c, nil
}

type indexRecord struct {
	unpadded, uncompressed uint64
}

// decodeXZStream decodes one .xz stream from the start of in, appending the
// output to out. It returns the number of bytes consumed (through the stream
// footer; stream padding is left to the caller). hint, if not nil, is the
// stream's index as read ahead by xzIndexHints; it is only used to size
// buffers (the real index is still validated against the decoded blocks).
func decodeXZStream(out, in []byte, hint []indexRecord) ([]byte, int, error) {
	if len(in) < streamHeaderSize {
		return out, 0, ErrTruncated
	}
	if !bytes.Equal(in[:6], streamMagic) {
		return out, 0, errorf(ErrFormat, "not an .xz stream")
	}
	flags := in[6:8]
	if crc32.ChecksumIEEE(flags) != binary.LittleEndian.Uint32(in[8:12]) {
		return out, 0, errorf(ErrCorrupt, "stream header CRC32 mismatch")
	}
	if flags[0] != 0 || flags[1]&0xF0 != 0 {
		return out, 0, errorf(ErrUnsupported, "unsupported stream flags 0x%02x%02x", flags[0], flags[1])
	}
	check := flags[1]
	pos := streamHeaderSize
	var records []indexRecord
	for {
		if pos >= len(in) {
			return out, 0, ErrTruncated
		}
		if in[pos] == 0x00 { // index indicator
			break
		}
		var (
			rec indexRecord
			n   int
			err error
		)
		blockHint := int64(-1)
		if i := len(records); i < len(hint) && hint[i].uncompressed < 1<<62 {
			blockHint = int64(hint[i].uncompressed)
		}
		out, rec, n, err = decodeBlock(out, in[pos:], check, blockHint)
		if err != nil {
			return out, 0, err
		}
		records = append(records, rec)
		pos += n
	}
	idxLen, err := decodeIndex(in[pos:], records)
	if err != nil {
		return out, 0, err
	}
	pos += idxLen
	if len(in)-pos < streamHeaderSize {
		return out, 0, ErrTruncated
	}
	footer := in[pos : pos+streamHeaderSize]
	if !bytes.Equal(footer[10:12], footerMagic) {
		return out, 0, errorf(ErrCorrupt, "bad stream footer magic")
	}
	if crc32.ChecksumIEEE(footer[4:10]) != binary.LittleEndian.Uint32(footer[0:4]) {
		return out, 0, errorf(ErrCorrupt, "stream footer CRC32 mismatch")
	}
	if backward := (uint64(binary.LittleEndian.Uint32(footer[4:8])) + 1) * 4; backward != uint64(idxLen) {
		return out, 0, errorf(ErrCorrupt, "stream footer backward size %d does not match index size %d", backward, idxLen)
	}
	if !bytes.Equal(footer[8:10], flags) {
		return out, 0, errorf(ErrCorrupt, "stream footer flags do not match the header")
	}
	return out, pos + streamHeaderSize, nil
}

// decodeBlock decodes the block at the start of in (whose first byte is the
// non-zero block header size byte).
func decodeBlock(out, in []byte, check byte, sizeHint int64) ([]byte, indexRecord, int, error) {
	var rec indexRecord
	hsize := (int(in[0]) + 1) * 4
	if len(in) < hsize {
		return out, rec, 0, ErrTruncated
	}
	hdr := in[:hsize]
	end := hsize - 4
	if crc32.ChecksumIEEE(hdr[:end]) != binary.LittleEndian.Uint32(hdr[end:]) {
		return out, rec, 0, errorf(ErrCorrupt, "block header CRC32 mismatch")
	}
	flags := hdr[1]
	if flags&0x3C != 0 {
		return out, rec, 0, errorf(ErrUnsupported, "unsupported block flags 0x%02x", flags)
	}
	p := 2
	field := func(what string) (uint64, error) {
		v, n, err := readVLI(hdr[p:end])
		if err != nil {
			return 0, errorf(ErrCorrupt, "block header %s: %v", what, err)
		}
		p += n
		return v, nil
	}
	csize, usize := int64(-1), int64(-1)
	if flags&0x40 != 0 {
		v, err := field("compressed size")
		if err != nil {
			return out, rec, 0, err
		}
		if v == 0 {
			return out, rec, 0, errorf(ErrCorrupt, "block compressed size is zero")
		}
		csize = int64(v)
	}
	if flags&0x80 != 0 {
		v, err := field("uncompressed size")
		if err != nil {
			return out, rec, 0, err
		}
		usize = int64(v)
	}
	chain := make([]chainFilter, int(flags&3)+1)
	for i := range chain {
		id, err := field("filter ID")
		if err != nil {
			return out, rec, 0, err
		}
		psize, err := field("filter properties size")
		if err != nil {
			return out, rec, 0, err
		}
		if psize > uint64(end-p) {
			return out, rec, 0, errorf(ErrCorrupt, "block header filter properties overflow the header")
		}
		if chain[i], err = filterFromProps(id, hdr[p:p+int(psize)]); err != nil {
			return out, rec, 0, err
		}
		p += int(psize)
	}
	for _, b := range hdr[p:end] {
		if b != 0 {
			return out, rec, 0, errorf(ErrUnsupported, "non-zero block header padding")
		}
	}
	if err := validateXZChain(chain); err != nil {
		return out, rec, 0, err
	}

	data := in[hsize:]
	if csize >= 0 {
		if csize > int64(len(data)) {
			return out, rec, 0, ErrTruncated
		}
		data = data[:csize]
	}
	outStart := len(out)
	out, consumed, err := decodeLZMA2(out, data, chain[len(chain)-1].lzma.dictSize, sizes{known: usize, hint: sizeHint})
	if err != nil {
		if csize >= 0 && errors.Is(err, ErrTruncated) {
			err = errorf(ErrCorrupt, "LZMA2 data ends before the declared compressed size")
		}
		return out, rec, 0, err
	}
	if csize >= 0 && int64(consumed) != csize {
		return out, rec, 0, errorf(ErrCorrupt, "compressed size %d does not match the declared %d", consumed, csize)
	}
	block := out[outStart:]
	applyDecodeFilters(block, chain[:len(chain)-1])

	q := hsize + consumed
	pad := (4 - consumed%4) % 4
	cs := checkSize(check)
	if len(in)-q < pad+cs {
		return out, rec, 0, ErrTruncated
	}
	for _, b := range in[q : q+pad] {
		if b != 0 {
			return out, rec, 0, errorf(ErrCorrupt, "non-zero block padding")
		}
	}
	q += pad
	if want := computeCheck(check, block); want != nil && !bytes.Equal(want, in[q:q+cs]) {
		return out, rec, 0, errorf(ErrCorrupt, "integrity check (type %d) mismatch", check)
	}
	q += cs
	rec = indexRecord{unpadded: uint64(hsize + consumed + cs), uncompressed: uint64(len(block))}
	return out, rec, q, nil
}

// parseIndex parses the index at the start of in (whose first byte is the
// zero index indicator) and returns its records and size.
func parseIndex(in []byte) ([]indexRecord, int, error) {
	p := 1 // index indicator
	vli := func(what string) (uint64, error) {
		v, n, err := readVLI(in[p:])
		if err != nil {
			if errors.Is(err, ErrTruncated) {
				return 0, err
			}
			return 0, errorf(ErrCorrupt, "index %s: %v", what, err)
		}
		p += n
		return v, nil
	}
	count, err := vli("record count")
	if err != nil {
		return nil, 0, err
	}
	// Each record takes at least two bytes; don't trust count beyond that.
	if count > uint64(len(in)-p)/2 {
		if uint64(len(in)) < 1<<20 {
			return nil, 0, ErrTruncated
		}
		return nil, 0, errorf(ErrCorrupt, "index record count %d too large", count)
	}
	records := make([]indexRecord, count)
	for i := range records {
		if records[i].unpadded, err = vli("unpadded size"); err != nil {
			return nil, 0, err
		}
		if records[i].uncompressed, err = vli("uncompressed size"); err != nil {
			return nil, 0, err
		}
	}
	pad := (4 - p%4) % 4
	if len(in)-p < pad+4 {
		return nil, 0, ErrTruncated
	}
	for _, b := range in[p : p+pad] {
		if b != 0 {
			return nil, 0, errorf(ErrCorrupt, "non-zero index padding")
		}
	}
	p += pad
	if crc32.ChecksumIEEE(in[:p]) != binary.LittleEndian.Uint32(in[p:p+4]) {
		return nil, 0, errorf(ErrCorrupt, "index CRC32 mismatch")
	}
	return records, p + 4, nil
}

// decodeIndex validates the index at the start of in against the decoded
// blocks and returns its size.
func decodeIndex(in []byte, blocks []indexRecord) (int, error) {
	// A record count that disagrees with the blocks is corruption even if
	// the index is also truncated, so check it first.
	if count, _, err := readVLI(in[1:]); err == nil && count != uint64(len(blocks)) {
		return 0, errorf(ErrCorrupt, "index has %d records but the stream has %d blocks", count, len(blocks))
	}
	records, n, err := parseIndex(in)
	if err != nil {
		return 0, err
	}
	for i, r := range records {
		if r != blocks[i] {
			return 0, errorf(ErrCorrupt, "index record %d (%d, %d) does not match block (%d, %d)",
				i, r.unpadded, r.uncompressed, blocks[i].unpadded, blocks[i].uncompressed)
		}
	}
	return n, nil
}

// xzIndexHints reads the indexes of the .xz streams in data backwards from
// the end (as xz --list does), returning each stream's records keyed by the
// offset of its stream header. Decoding then knows every block's
// uncompressed size in advance, even though liblzma (and so Python) does not
// record sizes in block headers, and can size dictionaries and output
// buffers exactly. The scan stops at the first thing that doesn't parse
// (e.g. trailing garbage); streams before that point simply get no hints.
func xzIndexHints(data []byte) map[int][]indexRecord {
	hints := map[int][]indexRecord{}
	end := len(data)
	for {
		for end >= 4 && binary.LittleEndian.Uint32(data[end-4:end]) == 0 {
			end -= 4 // stream padding
		}
		if end < 2*streamHeaderSize+8 {
			return hints
		}
		footer := data[end-streamHeaderSize : end]
		if !bytes.Equal(footer[10:12], footerMagic) ||
			crc32.ChecksumIEEE(footer[4:10]) != binary.LittleEndian.Uint32(footer[0:4]) {
			return hints
		}
		backward := (int64(binary.LittleEndian.Uint32(footer[4:8])) + 1) * 4
		idxStart := int64(end-streamHeaderSize) - backward
		if idxStart < streamHeaderSize || data[idxStart] != 0 {
			return hints
		}
		records, n, err := parseIndex(data[idxStart : end-streamHeaderSize])
		if err != nil || int64(n) != backward {
			return hints
		}
		blocks := int64(0)
		for _, r := range records {
			blocks += int64((r.unpadded + 3) &^ 3)
			if r.unpadded > 1<<40 || blocks > idxStart {
				return hints
			}
		}
		start := idxStart - blocks - streamHeaderSize
		if start < 0 {
			return hints
		}
		h := data[start : start+streamHeaderSize]
		if !bytes.Equal(h[:6], streamMagic) || !bytes.Equal(h[6:8], footer[8:10]) ||
			crc32.ChecksumIEEE(h[6:8]) != binary.LittleEndian.Uint32(h[8:12]) {
			return hints
		}
		hints[int(start)] = records
		end = int(start)
	}
}

// encodeXZ writes a single-stream .xz file with one block (none for empty
// input) whose header records both sizes.
func encodeXZ(data []byte, chain []chainFilter, check byte) ([]byte, error) {
	out := make([]byte, 0, len(data)/3+256)
	out = append(out, streamMagic...)
	flags := []byte{0, check}
	out = append(out, flags...)
	out = binary.LittleEndian.AppendUint32(out, crc32.ChecksumIEEE(flags))

	var records []indexRecord
	if len(data) > 0 {
		last := chain[len(chain)-1]
		filtered := applyEncodeFilters(data, chain[:len(chain)-1])
		comp, dict, err := encodeLZMA2(nil, filtered, last.lzma)
		if err != nil {
			return nil, err
		}
		hdr, err := blockHeader(chain, dict, uint64(len(comp)), uint64(len(data)))
		if err != nil {
			return nil, err
		}
		out = append(out, hdr...)
		out = append(out, comp...)
		for i := len(comp); i%4 != 0; i++ {
			out = append(out, 0)
		}
		sum := computeCheck(check, data)
		out = append(out, sum...)
		records = append(records, indexRecord{
			unpadded:     uint64(len(hdr) + len(comp) + len(sum)),
			uncompressed: uint64(len(data)),
		})
	}

	idxStart := len(out)
	out = append(out, 0)
	out = appendVLI(out, uint64(len(records)))
	for _, r := range records {
		out = appendVLI(out, r.unpadded)
		out = appendVLI(out, r.uncompressed)
	}
	for (len(out)-idxStart)%4 != 0 {
		out = append(out, 0)
	}
	out = binary.LittleEndian.AppendUint32(out, crc32.ChecksumIEEE(out[idxStart:]))
	idxLen := len(out) - idxStart

	var footer [streamHeaderSize]byte
	binary.LittleEndian.PutUint32(footer[4:8], uint32(idxLen/4-1))
	copy(footer[8:10], flags)
	binary.LittleEndian.PutUint32(footer[0:4], crc32.ChecksumIEEE(footer[4:10]))
	copy(footer[10:12], footerMagic)
	return append(out, footer[:]...), nil
}

func blockHeader(chain []chainFilter, dict uint32, csize, usize uint64) ([]byte, error) {
	h := []byte{0, byte(len(chain)-1) | 0x40 | 0x80}
	h = appendVLI(h, csize)
	h = appendVLI(h, usize)
	for _, f := range chain {
		h = appendVLI(h, f.id)
		switch {
		case f.id == FilterLZMA2:
			h = append(h, 1, lzma2DictCode(dict))
		case f.id == FilterDelta:
			h = append(h, 1, byte(f.dist-1))
		case isBCJ(f.id):
			if f.start == 0 {
				h = append(h, 0)
			} else {
				h = append(h, 4)
				h = binary.LittleEndian.AppendUint32(h, f.start)
			}
		default:
			return nil, errorf(ErrUnsupported, "filter %s cannot be used in .xz", filterName(f.id))
		}
	}
	for (len(h)+4)%4 != 0 {
		h = append(h, 0)
	}
	if len(h)+4 > maxBlockHeader {
		return nil, errorf(ErrUnsupported, "block header too large")
	}
	h[0] = byte((len(h)+4)/4 - 1)
	return binary.LittleEndian.AppendUint32(h, crc32.ChecksumIEEE(h)), nil
}
