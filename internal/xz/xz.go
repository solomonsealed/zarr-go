// Package xz implements the subset of liblzma (as exposed by Python's lzma
// module) that the numcodecs "lzma" codec needs: lzma.compress and
// lzma.decompress for the .xz, .lzma ("alone") and raw container formats,
// with LZMA1/LZMA2 plus the delta and BCJ filters.
//
// The LZMA/LZMA2 entropy coding is delegated to github.com/ulikunitz/xz/lzma;
// the .xz/.lzma containers, integrity checks, filter chains, the delta filter
// and the BCJ filters are implemented here. Everything is pure Go.
//
// Output is not byte-identical to liblzma (the LZMA encoder differs), but it
// is decodable by liblzma/Python for the same format and filter chain, and
// everything Python's lzma.compress produces decodes here.
package xz

import (
	"errors"
	"fmt"
)

// Container formats, with the numeric values of Python's lzma.FORMAT_*.
const (
	FormatAuto  = 0 // decode only: .xz or .lzma, detected per stream
	FormatXZ    = 1
	FormatAlone = 2 // legacy .lzma
	FormatRaw   = 3 // no container; the filter chain must be supplied
)

// Integrity check IDs, with the numeric values of Python's lzma.CHECK_*.
const (
	CheckNone    = 0
	CheckCRC32   = 1
	CheckCRC64   = 4
	CheckSHA256  = 10
	CheckIDMax   = 15
	CheckUnknown = 16
)

// Filter IDs, with the numeric values of Python's lzma.FILTER_*.
// FilterARM64 and FilterRISCV are liblzma >= 5.4 / 5.6 IDs that Python's
// lzma module does not name; ARM64 is supported here, RISC-V is not.
const (
	FilterLZMA1    uint64 = 0x4000000000000001
	FilterLZMA2    uint64 = 0x21
	FilterDelta    uint64 = 0x03
	FilterX86      uint64 = 0x04
	FilterPowerPC  uint64 = 0x05
	FilterIA64     uint64 = 0x06
	FilterARM      uint64 = 0x07
	FilterARMThumb uint64 = 0x08
	FilterSPARC    uint64 = 0x09
	FilterARM64    uint64 = 0x0A
	FilterRISCV    uint64 = 0x0B
)

// Encoder tuning constants (Python's lzma.MODE_*, MF_*), accepted in LZMA
// filter options for compatibility. They are validated but otherwise do not
// influence this package's encoder.
const (
	ModeFast   = 1
	ModeNormal = 2
	MFHC3      = 0x03
	MFHC4      = 0x04
	MFBT2      = 0x12
	MFBT3      = 0x13
	MFBT4      = 0x14
)

// Presets (Python's lzma.PRESET_*). A preset is a level 0-9, optionally OR-ed
// with PresetExtreme. PresetExtreme is untyped so that 9|PresetExtreme can be
// passed as an int on 64-bit platforms (including js/wasm).
const (
	PresetDefault = 6
	PresetExtreme = 0x80000000
)

// MaxDictSize is the largest dictionary accepted, for encoding (as in
// liblzma) and for decoding (a memory bound; liblzma itself would accept up to
// 4 GiB when no memory limit is set). Decoding never allocates more
// dictionary than the data can need, so typical memory use is far lower.
const MaxDictSize = 1<<30 + 1<<29 // 1.5 GiB

// Errors. Errors returned by this package wrap one of these.
var (
	// ErrTruncated: the input ended before the end of a stream.
	ErrTruncated = errors.New("xz: compressed data ended before the end-of-stream marker was reached")
	// ErrFormat: the input is not in the requested container format.
	ErrFormat = errors.New("xz: input format not recognized")
	// ErrCorrupt: the input is corrupt (bad check, bad index, bad LZMA data...).
	ErrCorrupt = errors.New("xz: corrupt input data")
	// ErrUnsupported: the input or the arguments use unsupported options.
	ErrUnsupported = errors.New("xz: invalid or unsupported options")
	// ErrMemLimit: the input requires a dictionary larger than MaxDictSize.
	ErrMemLimit = errors.New("xz: dictionary size exceeds memory limit")
)

func errorf(base error, format string, args ...any) error {
	return fmt.Errorf("%w: "+format, append([]any{base}, args...)...)
}

// Filter is one entry of a filter chain. It mirrors a Python lzma filter
// specifier dict: ID is the "id" entry and Options holds the remaining
// entries, e.g. {"preset": 1, "dict_size": 1 << 20, "lc": 3, "lp": 0,
// "pb": 2, "mode": 2, "nice_len": 64, "mf": 0x14, "depth": 0} for LZMA1/2,
// {"dist": 4} for delta and {"start_offset": 0} for BCJ filters.
// Option values may be any Go integer type, an integral float64 or a
// json.Number.
type Filter struct {
	ID      uint64
	Options map[string]any
}

// Decompress decodes data like Python's lzma.decompress(data, format=format,
// filters=filters).
//
// format is FormatAuto (.xz or .lzma, detected per stream), FormatXZ,
// FormatAlone or FormatRaw. filters is required for FormatRaw; for the other
// formats the container describes its own filters and filters is ignored
// (Python raises in that case, which makes numcodecs unable to read data it
// wrote with format=FORMAT_XZ and a filter chain; ignoring it lets such data
// be read).
//
// Like Python, concatenated streams are decoded and their outputs joined, and
// trailing data after at least one complete stream is ignored if it is not a
// valid stream (but is an error if it is a truncated one). Unlike Python,
// .xz stream padding (a multiple of four zero bytes) followed by another .xz
// stream is skipped and that stream decoded, as the .xz specification and
// the xz tool require (Python stops at the padding), and any run of zero
// bytes at the very end is accepted (Python rejects fewer than 12).
//
// Memory: dictionaries larger than MaxDictSize are rejected (ErrMemLimit);
// otherwise the dictionary allocated is the declared one capped by the
// output size when it is known (block headers, the .xz index, which is read
// ahead from the end of the input, or the .lzma header) or else by what the
// input can expand to, so small inputs never allocate large dictionaries.
// A declared uncompressed size that the data does not match is an error.
func Decompress(data []byte, format int, filters []Filter) ([]byte, error) {
	var chain []chainFilter
	switch format {
	case FormatAuto, FormatXZ, FormatAlone:
	case FormatRaw:
		if len(filters) == 0 {
			return nil, errorf(ErrUnsupported, "must specify filters for FORMAT_RAW")
		}
		var err error
		if chain, err = parseChain(filters, false); err != nil {
			return nil, err
		}
		if err = validateRawChain(chain); err != nil {
			return nil, err
		}
	default:
		return nil, errorf(ErrUnsupported, "invalid container format: %d", format)
	}
	return decompressLoop(data, format, chain)
}

// Compress encodes data like Python's lzma.compress(data, format=format,
// check=check, preset=preset, filters=filters).
//
// format is FormatXZ, FormatAlone or FormatRaw. check is one of the Check*
// constants or -1 for the default (CRC64 for .xz, none otherwise); only .xz
// supports checks. preset is -1 (Python's None, meaning PresetDefault) or a
// level 0-9 optionally OR-ed with PresetExtreme; it may not be combined with
// filters. filters (nil meaning Python's None) gives an explicit chain: for
// .xz up to three delta/BCJ filters followed by LZMA2; for .lzma exactly one
// LZMA1 filter; for raw, delta/BCJ filters followed by LZMA1 or LZMA2.
//
// The .xz output has one block (none for empty input) whose header records
// the compressed and uncompressed sizes, followed by the index and footer.
// The dictionary actually used (and declared in the headers) is the requested
// one shrunk to the input size, which only lowers decoder memory use.
func Compress(data []byte, format int, check int, preset int, filters []Filter) ([]byte, error) {
	switch format {
	case FormatXZ, FormatAlone, FormatRaw:
	default:
		return nil, errorf(ErrUnsupported, "invalid container format: %d", format)
	}
	if format != FormatXZ && check != -1 && check != CheckNone {
		return nil, errorf(ErrUnsupported, "integrity checks are only supported by FORMAT_XZ")
	}
	if preset != -1 && len(filters) > 0 {
		return nil, errorf(ErrUnsupported, "cannot specify both preset and filter chain")
	}
	var presetOpts lzmaOptions
	if len(filters) == 0 {
		if preset == -1 {
			preset = PresetDefault
		}
		if preset < 0 || uint64(preset) > 0xFFFFFFFF {
			return nil, errorf(ErrUnsupported, "invalid compression preset: %d", preset)
		}
		var err error
		if presetOpts, err = presetOptions(uint32(preset)); err != nil {
			return nil, err
		}
	}

	switch format {
	case FormatXZ:
		if check == -1 {
			check = CheckCRC64
		}
		if !checkSupported(check) {
			return nil, errorf(ErrUnsupported, "unsupported integrity check: %d", check)
		}
		chain := []chainFilter{{id: FilterLZMA2, lzma: presetOpts}}
		if len(filters) > 0 {
			var err error
			if chain, err = parseChain(filters, true); err != nil {
				return nil, err
			}
		}
		if err := validateXZChain(chain); err != nil {
			return nil, err
		}
		return encodeXZ(data, chain, byte(check))

	case FormatAlone:
		opts := presetOpts
		if len(filters) > 0 {
			chain, err := parseChain(filters, true)
			if err != nil {
				return nil, err
			}
			if len(chain) != 1 || chain[0].id != FilterLZMA1 {
				return nil, errorf(ErrUnsupported, "invalid filter chain for FORMAT_ALONE - must be a single LZMA1 filter")
			}
			opts = chain[0].lzma
		}
		return encodeAlone(data, opts)

	default: // FormatRaw
		if len(filters) == 0 {
			return nil, errorf(ErrUnsupported, "must specify filters for FORMAT_RAW")
		}
		chain, err := parseChain(filters, true)
		if err != nil {
			return nil, err
		}
		if err = validateRawChain(chain); err != nil {
			return nil, err
		}
		return encodeRaw(data, chain)
	}
}

// decompressLoop mirrors Python's lzma.decompress: decode streams one after
// another; after the first, a stream that fails with anything but truncation
// (or a memory-limit error) ends decoding and the rest is ignored.
func decompressLoop(data []byte, format int, chain []chainFilter) ([]byte, error) {
	var (
		out   []byte
		hints map[int][]indexRecord
	)
	pos := 0
	for first := true; ; first = false {
		rest := data[pos:]
		var (
			n    int
			err  error
			isXZ bool
		)
		start := len(out)
		switch {
		case format == FormatRaw:
			out, n, err = decodeRaw(out, rest, chain)
		case format == FormatXZ, format == FormatAuto && len(rest) > 0 && rest[0] == 0xFD:
			isXZ = true
			if hints == nil {
				hints = xzIndexHints(data)
			}
			out, n, err = decodeXZStream(out, rest, hints[pos])
		default: // FormatAlone, or FormatAuto with a non-.xz first byte
			out, n, err = decodeAlone(out, rest, format == FormatAuto)
		}
		if err != nil {
			if first || errors.Is(err, ErrTruncated) || errors.Is(err, ErrMemLimit) {
				return nil, err
			}
			return out[:start], nil
		}
		if n <= 0 { // cannot happen: every stream has a non-empty encoding
			return nil, errorf(ErrCorrupt, "empty stream")
		}
		pos += n
		if isXZ {
			// Stream padding: a multiple of four zero bytes between .xz
			// streams (skipped only if another .xz stream follows, so that
			// in FormatAuto a .lzma stream starting with zero bytes is not
			// eaten), or any number of zero bytes at the end.
			k := 0
			for pos+k < len(data) && data[pos+k] == 0 {
				k++
			}
			if pos+k == len(data) || (k%4 == 0 && data[pos+k] == 0xFD) {
				pos += k
			}
		}
		if pos >= len(data) {
			if out == nil {
				out = []byte{}
			}
			return out, nil
		}
	}
}

// decodeRaw decodes one raw stream with the given chain, appending to out.
func decodeRaw(out, in []byte, chain []chainFilter) ([]byte, int, error) {
	start := len(out)
	last := chain[len(chain)-1]
	var (
		n   int
		err error
	)
	if last.id == FilterLZMA1 {
		out, n, err = decodeLZMA1(out, in, last.lzma, -1)
	} else {
		out, n, err = decodeLZMA2(out, in, last.lzma.dictSize, unknownSizes)
	}
	if err != nil {
		return out, 0, err
	}
	applyDecodeFilters(out[start:], chain[:len(chain)-1])
	return out, n, nil
}

func encodeRaw(data []byte, chain []chainFilter) ([]byte, error) {
	filtered := applyEncodeFilters(data, chain[:len(chain)-1])
	last := chain[len(chain)-1]
	if last.id == FilterLZMA1 {
		return encodeLZMA1(filtered, last.lzma, false)
	}
	out, _, err := encodeLZMA2(nil, filtered, last.lzma)
	return out, err
}
