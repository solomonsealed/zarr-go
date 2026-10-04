package consistency

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

// parityInputs returns test buffers: empty and tiny ones, zeros, random
// bytes, smooth float arrays, small integers and text, around the sizes
// where compressors change strategy.
func parityInputs(rng *rand.Rand, quick bool) map[string][]byte {
	out := map[string][]byte{"empty": {}, "one": {7}}
	sizes := []int{3, 16, 100, 1000, 4096, 65536 + 3, 300000}
	if quick {
		sizes = []int{3, 100, 4096, 70000}
	}
	for _, n := range sizes {
		out[fmt.Sprint("zeros-", n)] = make([]byte, n)
		r := make([]byte, n)
		rng.Read(r)
		out[fmt.Sprint("random-", n)] = r
		f := make([]byte, n/8*8)
		x := 0.0
		for i := 0; i+8 <= len(f); i += 8 {
			x += rng.NormFloat64()
			binary.LittleEndian.PutUint64(f[i:], math.Float64bits(math.Round(x*16)/16))
		}
		out[fmt.Sprint("smooth-f64-", n)] = f
		s := make([]byte, n/4*4)
		for i := 0; i+4 <= len(s); i += 4 {
			binary.LittleEndian.PutUint32(s[i:], uint32(rng.Intn(50)))
		}
		out[fmt.Sprint("small-int-", n)] = s
		var tb bytes.Buffer
		for tb.Len() < n {
			fmt.Fprintf(&tb, "zarr chunk %d of array %s; ", rng.Intn(1000), []string{"temperature", "pressure", "wind"}[rng.Intn(3)])
		}
		out[fmt.Sprint("text-", n)] = tb.Bytes()[:n]
	}
	return out
}

type parityCodec struct {
	name       string
	v3         string // Zarr v3 codec JSON for Go
	py         string // numcodecs config for the oracle
	dtype      string // NumPy dtype giving the type size
	exact      bool   // Go must produce the same bytes
	noPyDecode bool
}

var (
	quirkMu sync.Mutex
	quirks  = map[string]bool{}
)

// noteQuirk records a numcodecs limitation met by the parity test.
func noteQuirk(t *testing.T, codec, input, what string) {
	key := codec + ": " + what
	quirkMu.Lock()
	defer quirkMu.Unlock()
	if !quirks[key] {
		quirks[key] = true
		t.Logf("numcodecs quirk (%s input): %s", input, key)
	}
}

// TestCodecParity compares Go's codecs with numcodecs on raw buffers:
// identical output where the codec is deterministic, and each side
// decoding the other's output everywhere.
func TestCodecParity(t *testing.T) {
	o := pythonOracle(t)
	rng := rand.New(rand.NewSource(1))
	inputs := parityInputs(rng, testing.Short())
	codecs := []parityCodec{
		{name: "crc32", v3: `{"name": "numcodecs.crc32"}`, py: `{"id": "crc32"}`, exact: true},
		{name: "crc32-end", v3: `{"name": "numcodecs.crc32", "configuration": {"location": "end"}}`, py: `{"id": "crc32", "location": "end"}`, exact: true},
		{name: "crc32c", v3: `{"name": "numcodecs.crc32c"}`, py: `{"id": "crc32c"}`, exact: true},
		{name: "adler32", v3: `{"name": "numcodecs.adler32"}`, py: `{"id": "adler32"}`, exact: true},
		{name: "fletcher32", v3: `{"name": "numcodecs.fletcher32"}`, py: `{"id": "fletcher32"}`, exact: true},
		{name: "jenkins_lookup3", v3: `{"name": "numcodecs.jenkins_lookup3"}`, py: `{"id": "jenkins_lookup3"}`, exact: true},
		{name: "jenkins_lookup3-initval", v3: `{"name": "numcodecs.jenkins_lookup3", "configuration": {"initval": 12345}}`, py: `{"id": "jenkins_lookup3", "initval": 12345}`, exact: true},
		{name: "shuffle", v3: `{"name": "numcodecs.shuffle", "configuration": {"elementsize": 4}}`, py: `{"id": "shuffle", "elementsize": 4}`, exact: true},
		{name: "base64", v3: `{"name": "numcodecs.base64"}`, py: `{"id": "base64"}`, exact: true},
		{name: "zlib", v3: `{"name": "numcodecs.zlib", "configuration": {"level": 6}}`, py: `{"id": "zlib", "level": 6}`},
		{name: "gzip", v3: `{"name": "numcodecs.gzip", "configuration": {"level": 1}}`, py: `{"id": "gzip", "level": 1}`},
		{name: "zstd", v3: `{"name": "numcodecs.zstd", "configuration": {"level": 3}}`, py: `{"id": "zstd", "level": 3}`},
		{name: "zstd-checksum", v3: `{"name": "numcodecs.zstd", "configuration": {"level": -1, "checksum": true}}`, py: `{"id": "zstd", "level": -1, "checksum": true}`},
		{name: "lzma", v3: `{"name": "numcodecs.lzma"}`, py: `{"id": "lzma"}`},
		{name: "lzma-alone", v3: `{"name": "numcodecs.lzma", "configuration": {"format": 2, "preset": 1}}`, py: `{"id": "lzma", "format": 2, "preset": 1}`},
		{name: "lzma-raw-delta", v3: `{"name": "numcodecs.lzma", "configuration": {"format": 3, "filters": [{"id": 3, "dist": 4}, {"id": 33, "preset": 1}]}}`, py: `{"id": "lzma", "format": 3, "filters": [{"id": 3, "dist": 4}, {"id": 33, "preset": 1}]}`},
		{name: "lz4", v3: `{"name": "numcodecs.lz4"}`, py: `{"id": "lz4"}`, exact: lz4Parity},
		{name: "lz4-accel", v3: `{"name": "numcodecs.lz4", "configuration": {"acceleration": 8}}`, py: `{"id": "lz4", "acceleration": 8}`, exact: lz4Parity},
	}
	for level := 1; level <= 9; level += 4 {
		codecs = append(codecs, parityCodec{name: fmt.Sprint("bz2-", level), v3: fmt.Sprintf(`{"name": "numcodecs.bz2", "configuration": {"level": %d}}`, level), py: fmt.Sprintf(`{"id": "bz2", "level": %d}`, level), exact: true})
	}
	for _, cname := range []string{"lz4", "lz4hc", "blosclz", "zstd", "zlib"} {
		for _, shuf := range []int{0, 1, 2} {
			for _, ts := range []int{1, 4, 8} {
				for _, clevel := range []int{0, 1, 5, 9} {
					if testing.Short() && clevel != 5 {
						continue
					}
					codecs = append(codecs, parityCodec{
						name:  fmt.Sprintf("blosc-%s-s%d-t%d-c%d", cname, shuf, ts, clevel),
						v3:    fmt.Sprintf(`{"name": "numcodecs.blosc", "configuration": {"cname": %q, "clevel": %d, "shuffle": %d, "typesize": %d}}`, cname, clevel, shuf, ts),
						py:    fmt.Sprintf(`{"id": "blosc", "cname": %q, "clevel": %d, "shuffle": %d}`, cname, clevel, shuf),
						dtype: fmt.Sprintf("|V%d", ts),
						exact: bloscParity[cname] || clevel == 0,
					})
				}
			}
		}
	}
	exact, decoded := 0, 0
	for _, c := range codecs {
		t.Run(c.name, func(t *testing.T) {
			var spec zarr.CodecSpec
			if err := json.Unmarshal([]byte(c.v3), &spec); err != nil {
				t.Fatal(err)
			}
			codec, err := zarr.NewCodec(spec, zarr.ChunkSpec{DataType: zarr.MustDataType("uint8")})
			if err != nil {
				t.Fatal(err)
			}
			bc := codec.(zarr.BytesToBytesCodec)
			for name, in := range inputs {
				if c.dtype != "" {
					var ts int
					fmt.Sscanf(c.dtype, "|V%d", &ts)
					in = in[:len(in)/ts*ts]
				}
				goEnc, goErr := bc.Encode(in)
				var pyEnc string
				req := map[string]any{"codec": json.RawMessage(c.py), "data": base64.StdEncoding.EncodeToString(in)}
				if c.dtype != "" {
					req["dtype"] = c.dtype
				}
				pyErr := o.call("codec_encode", req, &pyEnc)
				switch {
				case goErr != nil && pyErr != nil:
					continue // both refuse (e.g. shuffle of a partial element)
				case pyErr != nil:
					// numcodecs cannot encode it (Fletcher32 of an empty buffer).
					noteQuirk(t, c.name, name, "python cannot encode: "+pyErr.Error())
					continue
				case goErr != nil:
					t.Fatalf("%s: go encode: %v", name, goErr)
				}
				pb, _ := base64.StdEncoding.DecodeString(pyEnc)
				if c.exact {
					if !bytes.Equal(goEnc, pb) {
						t.Fatalf("%s (%d bytes): output differs: go %d bytes, python %d\ngo: % x\npy: % x", name, len(in), len(goEnc), len(pb), head(goEnc), head(pb))
					}
					exact++
				}
				// Each decodes the other's output.
				dec, err := bc.Decode(pb)
				if err != nil || !bytes.Equal(dec, in) {
					t.Fatalf("%s: go decoding python's output: %v (%d vs %d bytes)", name, err, len(dec), len(in))
				}
				if !c.noPyDecode {
					var back string
					if err := o.call("codec_decode", map[string]any{"codec": json.RawMessage(c.py), "data": base64.StdEncoding.EncodeToString(goEnc)}, &back); err != nil {
						// Fine if numcodecs cannot decode its own output either
						// (LZ4, Zstd and Blosc with empty buffers).
						if o.call("codec_decode", map[string]any{"codec": json.RawMessage(c.py), "data": pyEnc}, nil) != nil {
							noteQuirk(t, c.name, name, "python cannot decode its own output: "+err.Error())
							continue
						}
						t.Fatalf("%s: python decoding go's output: %v", name, err)
					}
					if bb, _ := base64.StdEncoding.DecodeString(back); !bytes.Equal(bb, in) {
						t.Fatalf("%s: python decoded go's output wrongly", name)
					}
				}
				decoded++
			}
		})
	}
	t.Logf("%d codec configurations, %d inputs: %d outputs byte-identical, %d cross-decoded", len(codecs), len(inputs), exact, decoded)
}
