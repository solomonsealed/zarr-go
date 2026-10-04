// Package blosc reads and writes c-blosc 1.x frames (the format used by the
// numcodecs/zarr "blosc" codec).
//
// Compress is a port of the compression path of c-blosc 1.21.7.dev (commit
// 04c06fb7339d3df3600b1862077b8699b30a0e1d, bundled by numcodecs 0.17.0):
// blosc_compress_ctx with one thread, compute_blocksize, the
// BLOSC_FORWARD_COMPAT_SPLIT split mode, per-block byte/bit shuffle, the
// incompressible-split and memcpy fallbacks and the BloscLZ, LZ4 and LZ4HC
// codecs, so that for those codecs its output is byte-identical to
//
//	numcodecs.Blosc(cname, clevel, shuffle, blocksize, typesize).encode(data)
//
// as computed by numcodecs in single-threaded mode (numcodecs.blosc.use_threads
// = False, or blosc.set_nthreads(1)). With several threads c-blosc writes
// the compressed blocks of multi-block buffers in completion order, so its
// output is not deterministic (but is still read by Decompress).
//
// The zlib and zstd codecs produce the same frame layout (header, block size,
// splits) but their payloads come from Go's compress/zlib and
// github.com/klauspost/compress/zstd instead of the C libraries, so the bytes
// differ; both directions decode fine. Snappy uses the raw snappy block
// format via github.com/klauspost/compress/snappy.
//
// The package is pure Go (no cgo, no unsafe) and builds for js/wasm.
package blosc

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/solomonsealed/zarr-go/internal/lz4"
)

// Frame and library constants (blosc.h, blosc.c).
const (
	headerLen     = 16 // BLOSC_MIN_HEADER_LENGTH == BLOSC_MAX_OVERHEAD
	versionFormat = 2  // BLOSC_VERSION_FORMAT written by c-blosc 1.x
	versionLZ     = 1  // BLOSC_*_VERSION_FORMAT, the same for every codec

	flagShuffle    = 0x1  // BLOSC_DOSHUFFLE
	flagMemcpyed   = 0x2  // BLOSC_MEMCPYED
	flagBitShuffle = 0x4  // BLOSC_DOBITSHUFFLE
	flagReserved   = 0x8  // "compressor flags from the future"
	flagDontSplit  = 0x10 // blocks are not split into typesize streams

	minBufferSize = 128      // MIN_BUFFERSIZE
	maxSplits     = 16       // MAX_SPLITS
	l1            = 32 << 10 // L1
	maxTypeSize   = 255      // BLOSC_MAX_TYPESIZE

	// MaxBufferSize is BLOSC_MAX_BUFFERSIZE, the largest input Compress
	// accepts.
	MaxBufferSize = math.MaxInt32 - headerLen
	maxBlockSize  = (math.MaxInt32 - maxTypeSize*4) / 3 // BLOSC_MAX_BLOCKSIZE
)

// Compressor codes (BLOSC_BLOSCLZ ... BLOSC_ZSTD).
const (
	codeBlosclz = iota
	codeLZ4
	codeLZ4HC
	codeSnappy
	codeZlib
	codeZstd
)

// Compressor formats stored in bits 5-7 of the flags (BLOSC_*_FORMAT).
const (
	formatBlosclz = 0
	formatLZ4     = 1 // lz4 and lz4hc
	formatSnappy  = 2
	formatZlib    = 3
	formatZstd    = 4
)

var compCodes = map[string]int{
	"blosclz": codeBlosclz,
	"lz4":     codeLZ4,
	"lz4hc":   codeLZ4HC,
	"snappy":  codeSnappy,
	"zlib":    codeZlib,
	"zstd":    codeZstd,
}

var compFormats = [...]int{
	codeBlosclz: formatBlosclz,
	codeLZ4:     formatLZ4,
	codeLZ4HC:   formatLZ4,
	codeSnappy:  formatSnappy,
	codeZlib:    formatZlib,
	codeZstd:    formatZstd,
}

// Shuffle modes.
const (
	NoShuffle   = 0
	ByteShuffle = 1
	BitShuffle  = 2
	// AutoShuffle is numcodecs' AUTOSHUFFLE (-1): bit shuffle when the type
	// size is 1, byte shuffle otherwise.
	AutoShuffle = -1
)

// Options configures Compress. The fields mirror the arguments of
// numcodecs.blosc.compress / blosc_compress_ctx.
type Options struct {
	// Compressor is one of "blosclz", "lz4", "lz4hc", "snappy", "zlib",
	// "zstd".
	Compressor string
	// Level is the compression level, 0-9. Level 0 stores the data
	// uncompressed.
	Level int
	// Shuffle is NoShuffle (0), ByteShuffle (1), BitShuffle (2) or
	// AutoShuffle (-1).
	Shuffle int
	// TypeSize is the element size in bytes. numcodecs uses the item size
	// of the input array (1 for bytes) unless the codec was created with
	// an explicit typesize; zarr v3's BloscCodec passes the data type size.
	// 0 means 1. Values above 255 make c-blosc treat the buffer as a byte
	// stream (typesize 1).
	TypeSize int
	// BlockSize forces the block size (c-blosc clamps it to [128,
	// BLOSC_MAX_BLOCKSIZE] and may still enlarge it for split codecs);
	// 0 selects it automatically.
	BlockSize int
}

var (
	// ErrCorrupt is returned by Decompress for malformed frames.
	ErrCorrupt = errors.New("blosc: corrupt frame")
	// ErrUnsupported is returned for frames using unsupported features.
	ErrUnsupported = errors.New("blosc: unsupported frame")
)

// ZstdLevel returns the zstd compression level c-blosc uses for a blosc
// clevel (zstd_wrap_compress): 2*clevel-1 for clevel < 9, ZSTD_maxCLevel()
// (22) for 9, except that clevel 8 maps to 22-2 = 20. That is
// 1→1, 2→3, 3→5, 4→7, 5→9, 6→11, 7→13, 8→20, 9→22. Compress maps the
// result onto github.com/klauspost/compress/zstd with
// zstd.EncoderLevelFromZstd (1: fastest, 3-5: default, 7-9: better,
// ≥11: best).
func ZstdLevel(clevel int) int {
	const zstdMaxCLevel = 22
	if clevel >= 9 {
		return zstdMaxCLevel
	}
	if clevel == 8 {
		return zstdMaxCLevel - 2
	}
	return clevel*2 - 1
}

// ZlibLevel returns the zlib compression level c-blosc uses for a blosc
// clevel: it passes clevel through unchanged (zlib_wrap_compress calls
// compress2 with clevel).
func ZlibLevel(clevel int) int { return clevel }

// LZ4Acceleration returns the LZ4_compress_fast acceleration c-blosc uses
// for the lz4 codec (get_accel): 10 - clevel. lz4hc uses clevel directly
// as the LZ4HC level.
func LZ4Acceleration(clevel int) int { return 10 - clevel }

// splitBlock is split_block() in the default BLOSC_FORWARD_COMPAT_SPLIT mode
// (numcodecs never changes it): every codec but zstd splits blocks into
// typesize streams when typesize <= 16 and a stream would hold at least 128
// elements.
func splitBlock(code, typesize, blocksize int) bool {
	return code != codeZstd && typesize <= maxSplits && blocksize/typesize >= minBufferSize
}

// hcr reports whether a codec is meant for high compression ratios (HCR).
func hcr(code int) bool { return code == codeLZ4HC || code == codeZlib || code == codeZstd }

// computeBlocksize is compute_blocksize().
func computeBlocksize(code, clevel, typesize, nbytes, forced int) int {
	if nbytes < typesize {
		return 1 // protection against very small buffers
	}
	blocksize := nbytes
	if forced != 0 {
		blocksize = min(max(forced, minBufferSize), maxBlockSize)
	} else if nbytes >= l1 {
		blocksize = l1
		if hcr(code) {
			blocksize *= 2
		}
		switch clevel {
		case 0:
			blocksize /= 4
		case 1:
			blocksize /= 2
		case 2:
		case 3:
			blocksize *= 2
		case 4, 5:
			blocksize *= 4
		case 6, 7, 8:
			blocksize *= 8
		case 9:
			blocksize *= 8
			if hcr(code) {
				blocksize *= 2
			}
		}
	}
	// Enlarge the blocksize for splittable codecs.
	if clevel > 0 && splitBlock(code, typesize, blocksize) {
		blocksize = min(blocksize, 1<<18) * typesize
		blocksize = min(max(blocksize, 1<<16), 1<<20)
	}
	blocksize = min(blocksize, nbytes)
	if blocksize > typesize {
		blocksize = blocksize / typesize * typesize
	}
	return blocksize
}

// Compress compresses src into a c-blosc 1.x frame, reproducing
// blosc_compress_ctx(clevel, shuffle, typesize, len(src), src, dest,
// len(src)+16, compressor, blocksize, 1) as called by numcodecs.
func Compress(src []byte, o Options) ([]byte, error) {
	code, ok := compCodes[o.Compressor]
	if !ok {
		return nil, fmt.Errorf("blosc: unknown compressor %q", o.Compressor)
	}
	clevel := o.Level
	if clevel < 0 || clevel > 9 {
		return nil, fmt.Errorf("blosc: compression level %d out of range [0, 9]", clevel)
	}
	typesize := o.TypeSize
	if typesize == 0 {
		typesize = 1
	}
	if typesize < 0 {
		return nil, fmt.Errorf("blosc: invalid typesize %d", o.TypeSize)
	}
	doshuffle := o.Shuffle
	if doshuffle == AutoShuffle {
		// numcodecs resolves AUTOSHUFFLE with the unclamped item size.
		doshuffle = ByteShuffle
		if typesize == 1 {
			doshuffle = BitShuffle
		}
	}
	if doshuffle < NoShuffle || doshuffle > BitShuffle {
		return nil, fmt.Errorf("blosc: invalid shuffle mode %d", o.Shuffle)
	}
	nbytes := len(src)
	if nbytes > MaxBufferSize {
		return nil, fmt.Errorf("blosc: input of %d bytes exceeds %d", nbytes, MaxBufferSize)
	}
	if typesize > maxTypeSize {
		typesize = 1 // treat the buffer as a byte stream
	}

	// initialize_context_compression.
	c := &cctx{
		code:     code,
		clevel:   clevel,
		typesize: typesize,
		src:      src,
		dest:     make([]byte, nbytes+headerLen), // numcodecs' destination size
	}
	c.blocksize = computeBlocksize(code, clevel, typesize, nbytes, o.BlockSize)
	c.nblocks = nbytes / c.blocksize
	c.leftover = nbytes % c.blocksize
	if c.leftover > 0 {
		c.nblocks++
	}

	// write_compression_header.
	d := c.dest
	d[0] = versionFormat
	d[1] = versionLZ
	d[3] = byte(typesize)
	binary.LittleEndian.PutUint32(d[4:], uint32(nbytes))
	binary.LittleEndian.PutUint32(d[8:], uint32(c.blocksize))
	var flags byte
	if clevel == 0 || nbytes < minBufferSize {
		flags |= flagMemcpyed // plain copy
	}
	switch doshuffle {
	case ByteShuffle:
		flags |= flagShuffle
	case BitShuffle:
		flags |= flagBitShuffle
	}
	if !splitBlock(code, typesize, c.blocksize) {
		flags |= flagDontSplit
	}
	flags |= byte(compFormats[code]) << 5
	c.flags = flags

	// blosc_compress_context.
	var ntbytes int
	if flags&flagMemcpyed == 0 {
		var err error
		if ntbytes, err = c.serial(); err != nil {
			return nil, err
		}
	}
	if ntbytes == 0 {
		// Incompressible (or clevel 0 / tiny buffer): store a plain copy.
		c.flags |= flagMemcpyed
		copy(d[headerLen:], src)
		ntbytes = headerLen + nbytes
	}
	d[2] = c.flags
	binary.LittleEndian.PutUint32(d[12:], uint32(ntbytes))
	if ntbytes < len(d)-len(d)/8 {
		// Do not pin the len(src)-sized work buffer.
		return append([]byte(nil), d[:ntbytes]...), nil
	}
	return d[:ntbytes:ntbytes], nil
}

// cctx holds the compression state (struct blosc_context).
type cctx struct {
	code, clevel, typesize int
	flags                  byte
	src, dest              []byte
	blocksize              int
	nblocks, leftover      int
	tmp, tmp2              []byte // shuffle output, bitshuffle scratch
}

// serial is serial_blosc() for compression. It returns the total frame size,
// or 0 if the data does not compress into len(src)+16 bytes.
func (c *cctx) serial() (int, error) {
	ntbytes := headerLen + 4*c.nblocks
	for j := range c.nblocks {
		binary.LittleEndian.PutUint32(c.dest[headerLen+4*j:], uint32(ntbytes))
		bsize := c.blocksize
		leftoverblock := false
		if j == c.nblocks-1 && c.leftover > 0 {
			bsize = c.leftover
			leftoverblock = true
		}
		start := j * c.blocksize
		cbytes, err := c.block(c.src[start:start+bsize], leftoverblock, ntbytes)
		if err != nil {
			return 0, err
		}
		if cbytes == 0 {
			return 0, nil // incompressible data
		}
		ntbytes += cbytes
	}
	return ntbytes, nil
}

// block is blosc_c(): shuffle and compress one block into c.dest at offset
// ntbytes. It returns the number of bytes written, or 0 if the block does not
// fit (the whole buffer is then stored uncompressed).
func (c *cctx) block(src []byte, leftoverblock bool, ntbytes int) (int, error) {
	bsize := len(src)
	maxbytes := len(c.dest)
	dontSplit := c.flags&flagDontSplit != 0
	typesize := c.typesize
	data := src
	if c.flags&flagShuffle != 0 && typesize > 1 {
		data = c.scratch(bsize)
		shuffle(typesize, src, data)
	} else if c.flags&flagBitShuffle != 0 && bsize >= typesize {
		data = c.scratch(bsize)
		if cap(c.tmp2) < bsize {
			c.tmp2 = make([]byte, c.blocksize)
		}
		bitshuffle(typesize, src, data, c.tmp2[:bsize])
	}

	nsplits := 1
	if !dontSplit && !leftoverblock {
		nsplits = typesize
	}
	neblock := bsize / nsplits
	ctbytes := 0
	dpos := ntbytes
	for j := range nsplits {
		dpos += 4
		ntbytes += 4
		ctbytes += 4
		maxout := neblock
		if c.code == codeSnappy {
			maxout = snappyMaxCompressedLength(neblock)
		}
		if ntbytes+maxout > maxbytes {
			maxout = maxbytes - ntbytes // avoid buffer overrun
			if maxout <= 0 {
				return 0, nil // non-compressible block
			}
		}
		in := data[j*neblock : (j+1)*neblock]
		cbytes, err := c.compressSplit(in, c.dest[dpos:dpos+maxout], !dontSplit)
		if err != nil {
			return 0, err
		}
		if cbytes < 0 || cbytes > maxout {
			return 0, fmt.Errorf("blosc: internal error: compressor returned %d bytes (max %d)", cbytes, maxout)
		}
		if cbytes == 0 || cbytes == neblock {
			// The compressor has been unable to compress data at all:
			// store the split as is (a split whose size equals neblock is
			// read back verbatim).
			if ntbytes+neblock > maxbytes {
				return 0, nil
			}
			copy(c.dest[dpos:], in)
			cbytes = neblock
		}
		binary.LittleEndian.PutUint32(c.dest[dpos-4:], uint32(cbytes))
		dpos += cbytes
		ntbytes += cbytes
		ctbytes += cbytes
	}
	return ctbytes, nil
}

func (c *cctx) scratch(n int) []byte {
	if cap(c.tmp) < n {
		c.tmp = make([]byte, c.blocksize)
	}
	return c.tmp[:n]
}

// compressSplit compresses one split into dst (len(dst) is the C maxout)
// and returns the compressed size, 0 meaning "did not compress".
func (c *cctx) compressSplit(in, dst []byte, split bool) (int, error) {
	switch c.code {
	case codeBlosclz:
		return blosclzCompress(c.clevel, in, dst, split), nil
	case codeLZ4:
		return lz4.CompressFastInto(dst, in, LZ4Acceleration(c.clevel)), nil
	case codeLZ4HC:
		return lz4.CompressHCInto(dst, in, c.clevel), nil
	case codeSnappy:
		return snappyCompress(in, dst), nil
	case codeZlib:
		return zlibCompress(in, dst, ZlibLevel(c.clevel))
	case codeZstd:
		return zstdCompress(in, dst, ZstdLevel(c.clevel))
	}
	return 0, fmt.Errorf("blosc: unknown compressor code %d", c.code)
}

// Sizes returns the uncompressed size, the frame size and the block size
// recorded in the header of a blosc frame (blosc_cbuffer_sizes), without
// validating the rest of the frame.
func Sizes(src []byte) (nbytes, cbytes, blocksize int, err error) {
	if len(src) < headerLen {
		return 0, 0, 0, fmt.Errorf("%w: %d bytes is shorter than the header", ErrCorrupt, len(src))
	}
	nbytes = int(int32(binary.LittleEndian.Uint32(src[4:])))
	blocksize = int(int32(binary.LittleEndian.Uint32(src[8:])))
	cbytes = int(int32(binary.LittleEndian.Uint32(src[12:])))
	return nbytes, cbytes, blocksize, nil
}

// Decompress decodes a c-blosc 1.x frame. Bytes after the frame's recorded
// compressed size are ignored. It returns an error, never panics, on
// corrupt input.
func Decompress(src []byte) ([]byte, error) {
	f, err := parseFrame(src)
	if err != nil {
		return nil, err
	}
	out := make([]byte, f.nbytes)
	if err := f.decode(out); err != nil {
		return nil, err
	}
	return out, nil
}

// DecompressInto decodes a c-blosc 1.x frame into dst, which must have room
// for the uncompressed size (see Sizes), and returns that size.
func DecompressInto(dst, src []byte) (int, error) {
	f, err := parseFrame(src)
	if err != nil {
		return 0, err
	}
	if len(dst) < f.nbytes {
		return 0, fmt.Errorf("blosc: destination of %d bytes is too small for %d", len(dst), f.nbytes)
	}
	if err := f.decode(dst[:f.nbytes]); err != nil {
		return 0, err
	}
	return f.nbytes, nil
}

type frame struct {
	src               []byte // truncated to cbytes
	version, flags    byte
	typesize          int
	nbytes, blocksize int
	nblocks, leftover int
	format            int
	dontSplit         bool
}

// parseFrame validates the header (blosc_run_decompression_with_context)
// and the block/split table, so that the output can be allocated safely.
func parseFrame(src []byte) (*frame, error) {
	nbytes, cbytes, blocksize, err := Sizes(src)
	if err != nil {
		return nil, err
	}
	f := &frame{version: src[0], flags: src[2], typesize: int(src[3]), nbytes: nbytes, blocksize: blocksize}
	if nbytes == 0 {
		return f, nil // empty buffer
	}
	if nbytes < 0 {
		return nil, fmt.Errorf("%w: negative size", ErrCorrupt)
	}
	if f.version == 0 {
		return nil, fmt.Errorf("%w: version 0", ErrCorrupt)
	}
	if f.version >= 3 && f.flags&(flagShuffle|flagBitShuffle) == flagShuffle|flagBitShuffle {
		return nil, fmt.Errorf("%w: c-blosc2 extended header", ErrUnsupported)
	}
	if blocksize <= 0 || blocksize > nbytes || blocksize > maxBlockSize || f.typesize == 0 {
		return nil, fmt.Errorf("%w: invalid block size %d or type size %d", ErrCorrupt, blocksize, f.typesize)
	}
	if f.flags&flagReserved != 0 {
		return nil, fmt.Errorf("%w: reserved flag set", ErrUnsupported)
	}
	if cbytes < headerLen || cbytes > len(src) {
		return nil, fmt.Errorf("%w: header claims %d compressed bytes, have %d", ErrCorrupt, cbytes, len(src))
	}
	f.src = src[:cbytes]
	f.nblocks = nbytes / blocksize
	f.leftover = nbytes % blocksize
	if f.leftover > 0 {
		f.nblocks++
	}
	if f.flags&flagMemcpyed != 0 {
		if nbytes+headerLen != cbytes {
			return nil, fmt.Errorf("%w: stored frame size mismatch", ErrCorrupt)
		}
		return f, nil
	}
	f.format = int(f.flags >> 5)
	if f.format > formatZstd {
		return nil, fmt.Errorf("%w: compressor format %d", ErrUnsupported, f.format)
	}
	if f.nblocks > (cbytes-headerLen)/4 {
		return nil, fmt.Errorf("%w: truncated block table", ErrCorrupt)
	}
	f.dontSplit = f.flags&flagDontSplit != 0

	// Walk all splits up front: offsets must be in range, each split must
	// be able to expand to its size, and (as in every frame c-blosc writes,
	// serially or with threads) the splits must not share bytes, so that
	// their total size fits in the frame. Together this bounds the output
	// size by the input size before anything is allocated.
	room := cbytes - headerLen - 4*f.nblocks
	for j := range f.nblocks {
		bsize, leftoverblock := f.blockSize(j)
		nsplits, neblock, err := f.splits(bsize, leftoverblock)
		if err != nil {
			return nil, err
		}
		off := int(int32(binary.LittleEndian.Uint32(f.src[headerLen+4*j:])))
		for range nsplits {
			csize, next, err := f.split(off)
			if err != nil {
				return nil, err
			}
			if csize != neblock && !plausibleSplit(f.format, f.src[off+4:next], neblock) {
				return nil, fmt.Errorf("%w: split of %d bytes cannot expand to %d", ErrCorrupt, csize, neblock)
			}
			if room -= 4 + csize; room < 0 {
				return nil, fmt.Errorf("%w: splits overlap", ErrCorrupt)
			}
			off = next
		}
	}
	return f, nil
}

func (f *frame) blockSize(j int) (bsize int, leftoverblock bool) {
	if j == f.nblocks-1 && f.leftover > 0 {
		return f.leftover, true
	}
	return f.blocksize, false
}

// splits returns the number and size of the splits of a block (blosc_d):
// besides the flag, the typesize/blocksize condition keeps compatibility
// with frames written before the split flag existed.
func (f *frame) splits(bsize int, leftoverblock bool) (nsplits, neblock int, err error) {
	nsplits = 1
	if !f.dontSplit && f.typesize <= maxSplits && bsize/f.typesize >= minBufferSize && !leftoverblock {
		nsplits = f.typesize
	}
	neblock = bsize / nsplits
	if neblock*nsplits != bsize {
		return 0, 0, fmt.Errorf("%w: block of %d bytes cannot be split in %d", ErrCorrupt, bsize, nsplits)
	}
	return nsplits, neblock, nil
}

// split reads the split header at offset off, returning the compressed
// size and the offset of the next split.
func (f *frame) split(off int) (csize, next int, err error) {
	cbytes := len(f.src)
	if off < 0 || off > cbytes-4 {
		return 0, 0, fmt.Errorf("%w: split offset %d out of range", ErrCorrupt, off)
	}
	csize = int(int32(binary.LittleEndian.Uint32(f.src[off:])))
	off += 4
	if csize < 0 || csize > cbytes-off {
		return 0, 0, fmt.Errorf("%w: split size %d out of range", ErrCorrupt, csize)
	}
	return csize, off + csize, nil
}

// decode decompresses the frame into out (len(out) == f.nbytes).
func (f *frame) decode(out []byte) error {
	if f.nbytes == 0 {
		return nil
	}
	if f.flags&flagMemcpyed != 0 {
		copy(out, f.src[headerLen:])
		return nil
	}
	var tmp, tmp2 []byte
	for j := range f.nblocks {
		bsize, leftoverblock := f.blockSize(j)
		dest := out[j*f.blocksize : j*f.blocksize+bsize]
		doShuffle := f.flags&flagShuffle != 0 && f.typesize > 1
		doBitShuffle := f.flags&flagBitShuffle != 0 && bsize >= f.typesize
		target := dest
		if doShuffle || doBitShuffle {
			if tmp == nil {
				tmp = make([]byte, f.blocksize)
			}
			target = tmp[:bsize]
		}
		nsplits, neblock, err := f.splits(bsize, leftoverblock)
		if err != nil {
			return err
		}
		off := int(int32(binary.LittleEndian.Uint32(f.src[headerLen+4*j:])))
		for s := range nsplits {
			csize, next, err := f.split(off)
			if err != nil {
				return err
			}
			in := f.src[off+4 : next]
			part := target[s*neblock : (s+1)*neblock]
			if csize == neblock {
				copy(part, in)
			} else if err := decodeSplit(f.format, in, part); err != nil {
				return err
			}
			off = next
		}
		switch {
		case doShuffle:
			unshuffle(f.typesize, target, dest)
		case doBitShuffle:
			if tmp2 == nil {
				tmp2 = make([]byte, f.blocksize)
			}
			bitunshuffle(f.typesize, target, dest, f.version, tmp2[:bsize])
		}
	}
	return nil
}

// decodeSplit decodes one compressed split into exactly len(dst) bytes.
func decodeSplit(format int, src, dst []byte) error {
	switch format {
	case formatBlosclz:
		if n := blosclzDecompress(src, dst); n != len(dst) {
			return fmt.Errorf("%w: blosclz split decoded to %d bytes, want %d", ErrCorrupt, n, len(dst))
		}
		return nil
	case formatLZ4:
		if err := lz4.Decompress(src, dst); err != nil {
			return fmt.Errorf("%w: %v", ErrCorrupt, err)
		}
		return nil
	case formatSnappy:
		return snappyDecompress(src, dst)
	case formatZlib:
		return zlibDecompress(src, dst)
	case formatZstd:
		return zstdDecompress(src, dst)
	}
	return fmt.Errorf("%w: compressor format %d", ErrUnsupported, format)
}
