package consistency

import (
	"encoding/base64"
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	zarr "github.com/solomonsealed/zarr-go"
)

// narrowFloats are the float types NumPy lacks; ml_dtypes defines their
// conversions, which zarr-extensions refers to.
var narrowFloats = []string{
	"float16", "bfloat16", "float8_e3m4", "float8_e4m3", "float8_e4m3fn", "float8_e4m3fnuz",
	"float8_e4m3b11fnuz", "float8_e5m2", "float8_e5m2fnuz", "float8_e8m0fnu",
	"float6_e2m3fn", "float6_e3m2fn", "float4_e2m1fn",
}

func f64Bytes(v []float64) string {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(x))
	}
	return base64.StdEncoding.EncodeToString(b)
}

// TestNarrowFloats compares Go's conversions of every narrow float type
// with ml_dtypes: decoding every bit pattern, and encoding float64 values
// around every representable value, halfway points, special values,
// overflow and underflow.
func TestNarrowFloats(t *testing.T) {
	o := pythonOracle(t)
	rng := rand.New(rand.NewSource(3))
	for _, name := range narrowFloats {
		t.Run(name, func(t *testing.T) {
			dt := zarr.MustDataType(name)
			bitsOf := uint(8 * dt.Size)
			if dt.Bits != 0 {
				bitsOf = uint(dt.Bits)
			}
			// Every bit pattern, decoded.
			n := 1 << bitsOf
			patterns := &zarr.NDArray{Shape: []uint64{uint64(n)}, DataType: dt, Data: make([]byte, n*dt.Size)}
			for i := 0; i < n; i++ {
				if dt.Size == 2 {
					binary.LittleEndian.PutUint16(patterns.Data[2*i:], uint16(i))
				} else {
					patterns.Data[i] = byte(i)
				}
			}
			decoded, err := zarr.Convert[float64](patterns)
			if err != nil {
				t.Fatal(err)
			}
			// Values to encode: each representable value and its
			// neighbours, midpoints between consecutive values, specials and
			// random magnitudes.
			var vals []float64
			finite := []float64{}
			for _, v := range decoded {
				if !math.IsNaN(v) && !math.IsInf(v, 0) {
					finite = append(finite, v)
				}
			}
			for _, v := range finite {
				vals = append(vals, v, math.Nextafter(v, math.Inf(1)), math.Nextafter(v, math.Inf(-1)))
			}
			sorted := append([]float64{}, finite...)
			sortFloats(sorted)
			for i := 1; i < len(sorted); i++ {
				mid := sorted[i-1] + (sorted[i]-sorted[i-1])/2
				vals = append(vals, mid, math.Nextafter(mid, math.Inf(1)), math.Nextafter(mid, math.Inf(-1)))
			}
			vals = append(vals, math.NaN(), math.Copysign(math.NaN(), -1), math.Inf(1), math.Inf(-1), 0, math.Copysign(0, -1),
				1e300, -1e300, 5e-324, -5e-324, math.MaxFloat32, math.SmallestNonzeroFloat32)
			for i := 0; i < 2000; i++ {
				vals = append(vals, rng.NormFloat64()*math.Pow(2, float64(rng.Intn(80)-40)))
			}
			var res struct{ Bits, Back string }
			// Decoding: ml_dtypes' value of every pattern.
			if err := o.call("ml_dtypes_decode", map[string]any{"dtype": name, "bits": base64.StdEncoding.EncodeToString(patterns.Data)}, &res); err != nil {
				t.Fatal(err)
			}
			back, _ := base64.StdEncoding.DecodeString(res.Back)
			for i, v := range decoded {
				want := math.Float64frombits(binary.LittleEndian.Uint64(back[8*i:]))
				if math.Float64bits(v) != math.Float64bits(want) && !(math.IsNaN(v) && math.IsNaN(want)) {
					t.Fatalf("decoding %#x: go %v, ml_dtypes %v", i, v, want)
				}
			}
			// Encoding.
			if err := o.call("ml_dtypes", map[string]any{"dtype": name, "values": f64Bytes(vals)}, &res); err != nil {
				t.Fatal(err)
			}
			want, _ := base64.StdEncoding.DecodeString(res.Bits)
			got, err := zarr.FromValues([]uint64{uint64(len(vals))}, dt, vals)
			if err != nil {
				t.Fatal(err)
			}
			bad := 0
			for i, v := range vals {
				g, w := got.Data[i*dt.Size:(i+1)*dt.Size], want[i*dt.Size:(i+1)*dt.Size]
				if string(g) != string(w) {
					if bad < 10 {
						t.Errorf("encoding %v (%#x): go % x, ml_dtypes % x", v, math.Float64bits(v), g, w)
					}
					bad++
				}
			}
			if bad > 0 {
				t.Fatalf("%d of %d values encode differently", bad, len(vals))
			}
			t.Logf("%d patterns decoded, %d values encoded identically", n, len(vals))
		})
	}
}

func sortFloats(v []float64) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j] < v[j-1]; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}
