package zarr

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/solomonsealed/zarr-go/internal/blosc"
	"github.com/solomonsealed/zarr-go/internal/bzip2"
	"github.com/solomonsealed/zarr-go/internal/lz4"
	"github.com/solomonsealed/zarr-go/internal/xz"
)

// Compressors implemented in internal packages: blosc, LZ4 and bzip2
// produce the same bytes as the C libraries numcodecs uses; LZMA
// interoperates with liblzma.

// parseBloscConfig reads a blosc configuration: the Zarr v3 codec's
// ("shuffle" as a name) or numcodecs' ("shuffle" as 0, 1, 2 or -1 for
// automatic). elemSize is the type size when the configuration has none.
func parseBloscConfig(cfg json.RawMessage, elemSize int) (blosc.Options, error) {
	var c struct {
		Cname     string          `json:"cname"`
		Clevel    *int            `json:"clevel"`
		Shuffle   json.RawMessage `json:"shuffle"`
		Typesize  int             `json:"typesize"`
		Blocksize int             `json:"blocksize"`
	}
	if err := parseConfig(cfg, &c); err != nil {
		return blosc.Options{}, err
	}
	o := blosc.Options{Compressor: c.Cname, Level: 5, TypeSize: c.Typesize, BlockSize: c.Blocksize, Shuffle: blosc.ByteShuffle}
	if o.Compressor == "" {
		o.Compressor = "lz4"
	}
	switch o.Compressor {
	case "blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd":
	default:
		return o, fmt.Errorf("unknown blosc compressor %q", o.Compressor)
	}
	if c.Clevel != nil {
		o.Level = *c.Clevel
	}
	if o.Level < 0 || o.Level > 9 {
		return o, fmt.Errorf("blosc clevel %d out of range", o.Level)
	}
	if o.TypeSize <= 0 {
		o.TypeSize = max(elemSize, 1)
	}
	var s string
	var n int
	switch {
	case len(c.Shuffle) == 0 || string(c.Shuffle) == "null":
	case json.Unmarshal(c.Shuffle, &s) == nil:
		switch s {
		case "noshuffle":
			o.Shuffle = blosc.NoShuffle
		case "shuffle":
			o.Shuffle = blosc.ByteShuffle
		case "bitshuffle":
			o.Shuffle = blosc.BitShuffle
		default:
			return o, fmt.Errorf("invalid blosc shuffle %q", s)
		}
	case json.Unmarshal(c.Shuffle, &n) == nil:
		if n < -1 || n > 2 {
			return o, fmt.Errorf("invalid blosc shuffle %d", n)
		}
		o.Shuffle = n
	default:
		return o, fmt.Errorf("invalid blosc shuffle %s", c.Shuffle)
	}
	return o, nil
}

func bloscCompress(in []byte, o blosc.Options) ([]byte, error) { return blosc.Compress(in, o) }

func bloscDecompress(in []byte) ([]byte, error) {
	out, err := blosc.Decompress(in)
	if err != nil {
		return nil, fmt.Errorf("zarr: blosc: %w", err)
	}
	return out, nil
}

func lz4Decompress(src, dst []byte) error {
	if err := lz4.Decompress(src, dst); err != nil {
		return fmt.Errorf("zarr: lz4: %w", err)
	}
	return nil
}

func lz4CompressFast(src []byte, acceleration int) []byte { return lz4.CompressFast(src, acceleration) }

func bz2Compress(in []byte, level int) ([]byte, error) { return bzip2.Compress(in, level) }

type lzmaFilter = xz.Filter

func parseLZMAFilters(raw json.RawMessage) ([]lzmaFilter, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var fs []map[string]any
	if err := dec.Decode(&fs); err != nil {
		return nil, err
	}
	out := make([]lzmaFilter, len(fs))
	for i, f := range fs {
		var err error
		if out[i], err = xz.FilterFromMap(f); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func lzmaDecompress(in []byte, format int, filters []lzmaFilter) ([]byte, error) {
	out, err := xz.Decompress(in, format, filters)
	if err != nil {
		return nil, fmt.Errorf("zarr: lzma: %w", err)
	}
	return out, nil
}

func lzmaCompress(in []byte, format, check, preset int, filters []lzmaFilter) ([]byte, error) {
	return xz.Compress(in, format, check, preset, filters)
}
