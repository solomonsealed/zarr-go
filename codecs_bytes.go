package zarr

import (
	"bytes"
	"compress/bzip2"
	"compress/gzip"
	"compress/zlib"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/adler32"
	"hash/crc32"
	"io"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// Bytes-to-bytes codecs: compressors and checksums, as Zarr v3 core codecs
// (gzip, zstd, blosc, crc32c) and numcodecs codecs ("numcodecs.<id>" in
// Zarr v3, "<id>" in Zarr v2).

// bytesFuncs is a bytes-to-bytes codec built from functions.
type bytesFuncs struct {
	name string
	dec  func([]byte) ([]byte, error)
	enc  func([]byte) ([]byte, error)
	// overhead, if >= 0, is the fixed number of bytes encoding adds.
	overhead int
}

func (c bytesFuncs) Decode(in []byte) ([]byte, error) { return c.dec(in) }

func (c bytesFuncs) Encode(in []byte) ([]byte, error) {
	if c.enc == nil {
		return nil, fmt.Errorf("%w: encoding with codec %q", ErrUnsupported, c.name)
	}
	return c.enc(in)
}

// sizedBytesFuncs is a bytesFuncs whose encoded size is fixed.
type sizedBytesFuncs struct{ bytesFuncs }

func (c sizedBytesFuncs) EncodedSize(n int) int { return n + c.overhead }

func registerBytes(names []string, f func(cfg json.RawMessage, in ChunkSpec, name string) (BytesCodec, error)) {
	for _, name := range names {
		RegisterCodec(name, func(cfg json.RawMessage, in ChunkSpec) (any, error) { return f(cfg, in, name) })
	}
}

func init() {
	registerBytes([]string{"gzip", "numcodecs.gzip"}, func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
		c := struct {
			Level int `json:"level"`
		}{Level: 5}
		if name != "gzip" {
			c.Level = 1 // numcodecs' default
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		if c.Level < 0 || c.Level > 9 {
			return nil, fmt.Errorf("gzip level %d out of range", c.Level)
		}
		return bytesFuncs{name: name, dec: gunzip, enc: func(in []byte) ([]byte, error) { return gzipBytes(in, c.Level) }}, nil
	})
	registerBytes([]string{"numcodecs.zlib"}, func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
		c := struct {
			Level int `json:"level"`
		}{Level: 1}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		return bytesFuncs{name: name, dec: unzlib, enc: func(in []byte) ([]byte, error) { return zlibBytes(in, c.Level) }}, nil
	})
	registerBytes([]string{"zstd", "numcodecs.zstd"}, func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
		var c struct {
			Level    int  `json:"level"`
			Checksum bool `json:"checksum"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		return bytesFuncs{name: name, dec: unzstd, enc: func(in []byte) ([]byte, error) { return zstdBytes(in, c.Level, c.Checksum) }}, nil
	})
	registerBytes([]string{"blosc", "numcodecs.blosc"}, func(cfg json.RawMessage, in ChunkSpec, name string) (BytesCodec, error) {
		// numcodecs takes the type size from the buffer it is given:
		// zarr-python's v3 wrapper passes bytes (size 1), Zarr v2 the
		// array. The core codec's configuration has it.
		size := in.DataType.Size
		if name == "numcodecs.blosc" && !in.v2 {
			size = 1
		}
		o, err := parseBloscConfig(cfg, size)
		if err != nil {
			return nil, err
		}
		return bytesFuncs{name: name, dec: bloscDecompress, enc: func(b []byte) ([]byte, error) { return bloscCompress(b, o) }}, nil
	})
	registerBytes([]string{"numcodecs.bz2"}, func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
		c := struct {
			Level int `json:"level"`
		}{Level: 1}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		if c.Level < 1 || c.Level > 9 {
			return nil, fmt.Errorf("bz2 level %d out of range", c.Level)
		}
		return bytesFuncs{name: name, dec: unbzip2, enc: func(in []byte) ([]byte, error) { return bz2Compress(in, c.Level) }}, nil
	})
	registerBytes([]string{"numcodecs.lzma"}, func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
		c := struct {
			Format  int             `json:"format"`
			Check   int             `json:"check"`
			Preset  *int            `json:"preset"`
			Filters json.RawMessage `json:"filters"`
		}{Format: 1, Check: -1}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		filters, err := parseLZMAFilters(c.Filters)
		if err != nil {
			return nil, err
		}
		preset := -1
		if c.Preset != nil {
			preset = *c.Preset
		}
		return bytesFuncs{name: name,
			dec: func(in []byte) ([]byte, error) { return lzmaDecompress(in, c.Format, filters) },
			enc: func(in []byte) ([]byte, error) { return lzmaCompress(in, c.Format, c.Check, preset, filters) },
		}, nil
	})
	registerBytes([]string{"numcodecs.lz4"}, func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
		c := struct {
			Acceleration int `json:"acceleration"`
		}{Acceleration: 1}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		return bytesFuncs{name: name, dec: unlz4Numcodecs, enc: func(in []byte) ([]byte, error) {
			out := binary.LittleEndian.AppendUint32(nil, uint32(len(in)))
			return append(out, lz4CompressFast(in, c.Acceleration)...), nil
		}}, nil
	})
	registerBytes([]string{"numcodecs.base64"}, func(json.RawMessage, ChunkSpec, string) (BytesCodec, error) {
		return bytesFuncs{name: "base64",
			dec: func(in []byte) ([]byte, error) {
				out := make([]byte, base64.StdEncoding.DecodedLen(len(in)))
				n, err := base64.StdEncoding.Decode(out, in)
				return out[:n], err
			},
			enc: func(in []byte) ([]byte, error) {
				out := make([]byte, base64.StdEncoding.EncodedLen(len(in)))
				base64.StdEncoding.Encode(out, in)
				return out, nil
			}}, nil
	})
	registerBytes([]string{"numcodecs.shuffle"}, func(cfg json.RawMessage, in ChunkSpec, name string) (BytesCodec, error) {
		var c struct {
			ElementSize *int `json:"elementsize"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		size := in.DataType.Size // zarr-python fills in the item size
		if c.ElementSize != nil {
			size = *c.ElementSize
		}
		apply := func(f func(int, []byte, []byte)) func([]byte) ([]byte, error) {
			return func(in []byte) ([]byte, error) {
				if size <= 1 || len(in) == 0 {
					return in, nil
				}
				if len(in)%size != 0 {
					return nil, fmt.Errorf("zarr: shuffle buffer of %d bytes is not a multiple of elementsize %d", len(in), size)
				}
				out := make([]byte, len(in))
				f(size, in, out)
				return out, nil
			}
		}
		return sizedBytesFuncs{bytesFuncs{name: name, dec: apply(unshuffle), enc: apply(shuffle)}}, nil
	})

	// Checksums. numcodecs' default location is "start", except "end" for
	// CRC32C; fletcher32 and jenkins_lookup3 always append.
	crc32cSum := func(b []byte) uint32 { return crc32.Checksum(b, castagnoli) }
	checksum := func(defaultLocation string, sum func([]byte) uint32) func(json.RawMessage, ChunkSpec, string) (BytesCodec, error) {
		return func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
			var c struct {
				Location string `json:"location"`
			}
			if err := parseConfig(cfg, &c); err != nil {
				return nil, err
			}
			if c.Location == "" {
				c.Location = defaultLocation
			}
			if c.Location != "start" && c.Location != "end" {
				return nil, fmt.Errorf("invalid checksum location %q", c.Location)
			}
			return sizedBytesFuncs{bytesFuncs{name: name, overhead: 4,
				dec: func(in []byte) ([]byte, error) { return stripChecksum(in, c.Location, name, sum) },
				enc: func(in []byte) ([]byte, error) { return addChecksum(in, c.Location, sum), nil }}}, nil
		}
	}
	registerBytes([]string{"crc32c"}, func(cfg json.RawMessage, in ChunkSpec, name string) (BytesCodec, error) {
		return checksum("end", crc32cSum)(nil, in, name)
	})
	registerBytes([]string{"numcodecs.crc32c"}, checksum("end", crc32cSum))
	registerBytes([]string{"numcodecs.crc32"}, checksum("start", crc32.ChecksumIEEE))
	registerBytes([]string{"numcodecs.adler32"}, checksum("start", adler32.Checksum))
	registerBytes([]string{"numcodecs.fletcher32"}, func(json.RawMessage, ChunkSpec, string) (BytesCodec, error) {
		return checksum("end", fletcher32)(nil, ChunkSpec{}, "fletcher32")
	})
	registerBytes([]string{"numcodecs.jenkins_lookup3"}, func(cfg json.RawMessage, _ ChunkSpec, name string) (BytesCodec, error) {
		var c struct {
			InitVal uint32          `json:"initval"`
			Prefix  json.RawMessage `json:"prefix"`
		}
		if err := parseConfig(cfg, &c); err != nil {
			return nil, err
		}
		var prefix []byte
		if len(c.Prefix) > 0 && string(c.Prefix) != "null" {
			var ok bool
			if prefix, ok = parseBytesFill(c.Prefix); !ok {
				return nil, fmt.Errorf("invalid jenkins_lookup3 prefix %s", c.Prefix)
			}
		}
		sum := func(b []byte) uint32 {
			if prefix != nil {
				b = append(append([]byte(nil), prefix...), b...)
			}
			return jenkinsLookup3(b, c.InitVal)
		}
		return checksum("end", sum)(nil, ChunkSpec{}, name)
	})
}

var castagnoli = crc32.MakeTable(crc32.Castagnoli)

func stripChecksum(in []byte, location, kind string, sum func([]byte) uint32) ([]byte, error) {
	if len(in) < 4 {
		return nil, fmt.Errorf("zarr: %s buffer too short", kind)
	}
	var data, stored []byte
	if location == "end" {
		data, stored = in[:len(in)-4], in[len(in)-4:]
	} else {
		data, stored = in[4:], in[:4]
	}
	if got, want := sum(data), binary.LittleEndian.Uint32(stored); got != want {
		return nil, fmt.Errorf("zarr: %s checksum mismatch (stored %08x, computed %08x)", kind, want, got)
	}
	return data, nil
}

func addChecksum(in []byte, location string, sum func([]byte) uint32) []byte {
	cs := binary.LittleEndian.AppendUint32(nil, sum(in))
	if location == "end" {
		return append(append(make([]byte, 0, len(in)+4), in...), cs...)
	}
	return append(cs, in...)
}

// fletcher32 is numcodecs' Fletcher-32, HDF5's variant: big-endian 16-bit
// words, a trailing odd byte as the high byte of a final word.
func fletcher32(data []byte) uint32 {
	var sum1, sum2 uint32
	words := len(data) / 2
	for i := 0; i < words; {
		n := min(360, words-i)
		for ; n > 0; n-- {
			sum1 += uint32(binary.BigEndian.Uint16(data[2*i:]))
			sum2 += sum1
			i++
		}
		sum1 = (sum1 & 0xffff) + (sum1 >> 16)
		sum2 = (sum2 & 0xffff) + (sum2 >> 16)
	}
	if len(data)%2 == 1 {
		sum1 += uint32(data[len(data)-1]) << 8
		sum2 += sum1
		sum1 = (sum1 & 0xffff) + (sum1 >> 16)
		sum2 = (sum2 & 0xffff) + (sum2 >> 16)
	}
	sum1 = (sum1 & 0xffff) + (sum1 >> 16)
	sum2 = (sum2 & 0xffff) + (sum2 >> 16)
	return sum2<<16 | sum1
}

// jenkinsLookup3 is Bob Jenkins' hashlittle (lookup3.c) as HDF5 and
// numcodecs use it.
func jenkinsLookup3(k []byte, initval uint32) uint32 {
	rot := func(x uint32, n uint) uint32 { return x<<n | x>>(32-n) }
	a := 0xdeadbeef + uint32(len(k)) + initval
	b, c := a, a
	for len(k) > 12 {
		a += uint32(k[0]) | uint32(k[1])<<8 | uint32(k[2])<<16 | uint32(k[3])<<24
		b += uint32(k[4]) | uint32(k[5])<<8 | uint32(k[6])<<16 | uint32(k[7])<<24
		c += uint32(k[8]) | uint32(k[9])<<8 | uint32(k[10])<<16 | uint32(k[11])<<24
		// mix
		a -= c
		a ^= rot(c, 4)
		c += b
		b -= a
		b ^= rot(a, 6)
		a += c
		c -= b
		c ^= rot(b, 8)
		b += a
		a -= c
		a ^= rot(c, 16)
		c += b
		b -= a
		b ^= rot(a, 19)
		a += c
		c -= b
		c ^= rot(b, 4)
		b += a
		k = k[12:]
	}
	if len(k) == 0 {
		return c
	}
	var tail [12]byte
	copy(tail[:], k)
	a += uint32(tail[0]) | uint32(tail[1])<<8 | uint32(tail[2])<<16 | uint32(tail[3])<<24
	b += uint32(tail[4]) | uint32(tail[5])<<8 | uint32(tail[6])<<16 | uint32(tail[7])<<24
	c += uint32(tail[8]) | uint32(tail[9])<<8 | uint32(tail[10])<<16 | uint32(tail[11])<<24
	// final
	c ^= b
	c -= rot(b, 14)
	a ^= c
	a -= rot(c, 11)
	b ^= a
	b -= rot(a, 25)
	c ^= b
	c -= rot(b, 16)
	a ^= c
	a -= rot(c, 4)
	b ^= a
	b -= rot(a, 14)
	c ^= b
	c -= rot(b, 24)
	return c
}

func gunzip(in []byte) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(in))
	if err != nil {
		return nil, fmt.Errorf("zarr: gzip: %w", err)
	}
	defer r.Close()
	return io.ReadAll(r)
}

func unzlib(in []byte) ([]byte, error) {
	r, err := zlib.NewReader(bytes.NewReader(in))
	if err != nil {
		return nil, fmt.Errorf("zarr: zlib: %w", err)
	}
	defer r.Close()
	return io.ReadAll(r)
}

func unbzip2(in []byte) ([]byte, error) {
	out, err := io.ReadAll(bzip2.NewReader(bytes.NewReader(in)))
	if err != nil {
		return nil, fmt.Errorf("zarr: bz2: %w", err)
	}
	return out, nil
}

func gzipBytes(in []byte, level int) ([]byte, error) {
	var buf bytes.Buffer
	w, err := gzip.NewWriterLevel(&buf, level)
	if err != nil {
		return nil, fmt.Errorf("zarr: gzip: %w", err)
	}
	w.Write(in)
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func zlibBytes(in []byte, level int) ([]byte, error) {
	var buf bytes.Buffer
	w, err := zlib.NewWriterLevel(&buf, min(max(level, -1), 9))
	if err != nil {
		return nil, fmt.Errorf("zarr: zlib: %w", err)
	}
	w.Write(in)
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

var (
	zstdOnce sync.Once
	zstdDec  *zstd.Decoder
	zstdErr  error
)

func unzstd(in []byte) ([]byte, error) {
	zstdOnce.Do(func() {
		zstdDec, zstdErr = zstd.NewReader(nil, zstd.WithDecoderConcurrency(0), zstd.WithDecoderMaxMemory(1<<34))
	})
	if zstdErr != nil {
		return nil, zstdErr
	}
	if err := checkZstdSizes(in); err != nil {
		return nil, err
	}
	out, err := zstdDec.DecodeAll(in, nil)
	if err != nil {
		return nil, fmt.Errorf("zarr: zstd: %w", err)
	}
	return out, nil
}

// checkZstdSizes refuses zstd frames that declare more content than
// their bytes can hold (the decoder allocates the declared size up
// front): a block decodes to at most 128 KiB and takes at least 3 bytes.
// Anything else wrong with the frames is left for the decoder to report.
func checkZstdSizes(in []byte) error {
	limit := uint64(len(in)/3+1) << 17
	var total uint64
	for len(in) > 0 {
		var h zstd.Header
		rest, err := h.DecodeAndStrip(in)
		if err != nil {
			return nil
		}
		if h.Skippable {
			if uint64(h.SkippableSize) > uint64(len(rest)) {
				return nil
			}
			in = rest[h.SkippableSize:]
			continue
		}
		if h.FrameContentSize > limit-total {
			return fmt.Errorf("zarr: zstd: frame content size %d exceeds what %d bytes can hold", h.FrameContentSize, len(in))
		}
		total += h.FrameContentSize
		for last := false; !last; {
			if len(rest) < 3 {
				return nil
			}
			bh := uint32(rest[0]) | uint32(rest[1])<<8 | uint32(rest[2])<<16
			size := int(bh >> 3)
			if bh>>1&3 == 1 { // RLE: one byte, repeated
				size = 1
			}
			if len(rest) < 3+size {
				return nil
			}
			rest, last = rest[3+size:], bh&1 != 0
		}
		if h.HasCheckSum {
			if len(rest) < 4 {
				return nil
			}
			rest = rest[4:]
		}
		in = rest
	}
	return nil
}

type zstdKey struct {
	level    int
	checksum bool
}

var zstdEncoders sync.Map // zstdKey -> *zstd.Encoder

// zstdBytes compresses one frame. Level follows zstd's numbering (0 means
// the default, 3; negative levels are the fastest); checksum adds the
// frame checksum.
func zstdBytes(in []byte, level int, checksum bool) ([]byte, error) {
	k := zstdKey{level, checksum}
	e, ok := zstdEncoders.Load(k)
	if !ok {
		lvl := level
		if lvl == 0 {
			lvl = 3
		}
		if lvl < 1 {
			lvl = 1
		}
		enc, err := zstd.NewWriter(nil,
			zstd.WithEncoderLevel(zstd.EncoderLevelFromZstd(lvl)),
			zstd.WithEncoderCRC(checksum),
			zstd.WithEncoderConcurrency(1),
			zstd.WithZeroFrames(true))
		if err != nil {
			return nil, err
		}
		e, _ = zstdEncoders.LoadOrStore(k, enc)
	}
	return e.(*zstd.Encoder).EncodeAll(in, nil), nil
}

// unlz4Numcodecs decodes numcodecs' LZ4 framing: a little-endian uint32
// uncompressed size followed by one LZ4 block.
func unlz4Numcodecs(in []byte) ([]byte, error) {
	if len(in) < 4 {
		return nil, fmt.Errorf("zarr: lz4 buffer too short")
	}
	n := binary.LittleEndian.Uint32(in)
	// An LZ4 block decodes to at most 255 bytes per input byte (plus a
	// few literals).
	if n > 1<<31 || uint64(n) > 255*uint64(len(in))+16 {
		return nil, fmt.Errorf("zarr: lz4 header claims %d bytes", n)
	}
	out := make([]byte, n)
	if err := lz4Decompress(in[4:], out); err != nil {
		return nil, err
	}
	return out, nil
}

// shuffle groups byte j of every element together (the inverse of unshuffle).
func shuffle(typesize int, src, dst []byte) {
	n := len(src) / typesize
	for j := 0; j < typesize; j++ {
		row := dst[j*n : (j+1)*n]
		for i := range row {
			row[i] = src[i*typesize+j]
		}
	}
	copy(dst[n*typesize:], src[n*typesize:])
}

// unshuffle reverses shuffle. Trailing bytes that do not form a whole
// element are copied.
func unshuffle(typesize int, src, dst []byte) {
	n := len(src) / typesize
	for j := 0; j < typesize; j++ {
		row := src[j*n : (j+1)*n]
		for i, b := range row {
			dst[i*typesize+j] = b
		}
	}
	copy(dst[n*typesize:], src[n*typesize:])
}
