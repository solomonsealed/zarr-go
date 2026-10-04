package blosc

import "encoding/binary"

// Byte shuffle and bit shuffle as implemented by c-blosc 1.21
// (shuffle-generic.h, bitshuffle-generic.c and the dispatchers in shuffle.c).
// The SSE2/AVX2 variants produce the same bytes as the generic code.

// shuffle is blosc_internal_shuffle: byte j of every element is gathered
// into row j. Trailing bytes that do not form a whole element are copied
// unchanged.
func shuffle(typesize int, src, dst []byte) {
	n := len(src) / typesize
	switch typesize {
	case 1:
		copy(dst, src)
		return
	case 2:
		d0, d1 := dst[:n], dst[n:2*n]
		for i := range n {
			s := src[2*i : 2*i+2]
			d0[i], d1[i] = s[0], s[1]
		}
	case 4:
		d0, d1, d2, d3 := dst[:n], dst[n:2*n], dst[2*n:3*n], dst[3*n:4*n]
		for i := range n {
			s := src[4*i : 4*i+4]
			d0[i], d1[i], d2[i], d3[i] = s[0], s[1], s[2], s[3]
		}
	case 8:
		d0, d1, d2, d3 := dst[:n], dst[n:2*n], dst[2*n:3*n], dst[3*n:4*n]
		d4, d5, d6, d7 := dst[4*n:5*n], dst[5*n:6*n], dst[6*n:7*n], dst[7*n:8*n]
		for i := range d0 {
			s := src[8*i : 8*i+8]
			d0[i], d1[i], d2[i], d3[i] = s[0], s[1], s[2], s[3]
			d4[i], d5[i], d6[i], d7[i] = s[4], s[5], s[6], s[7]
		}
	default:
		for j := range typesize {
			row := dst[j*n : (j+1)*n]
			for i := range row {
				row[i] = src[i*typesize+j]
			}
		}
	}
	copy(dst[n*typesize:len(src)], src[n*typesize:])
}

// unshuffle is blosc_internal_unshuffle, the inverse of shuffle.
func unshuffle(typesize int, src, dst []byte) {
	n := len(src) / typesize
	switch typesize {
	case 1:
		copy(dst, src)
		return
	case 2:
		s0, s1 := src[:n], src[n:2*n]
		for i := range n {
			d := dst[2*i : 2*i+2]
			d[0], d[1] = s0[i], s1[i]
		}
	case 4:
		s0, s1, s2, s3 := src[:n], src[n:2*n], src[2*n:3*n], src[3*n:4*n]
		for i := range n {
			d := dst[4*i : 4*i+4]
			d[0], d[1], d[2], d[3] = s0[i], s1[i], s2[i], s3[i]
		}
	case 8:
		s0, s1, s2, s3 := src[:n], src[n:2*n], src[2*n:3*n], src[3*n:4*n]
		s4, s5, s6, s7 := src[4*n:5*n], src[5*n:6*n], src[6*n:7*n], src[7*n:8*n]
		for i := range s0 {
			d := dst[8*i : 8*i+8]
			d[0], d[1], d[2], d[3] = s0[i], s1[i], s2[i], s3[i]
			d[4], d[5], d[6], d[7] = s4[i], s5[i], s6[i], s7[i]
		}
	default:
		for j := range typesize {
			row := src[j*n : (j+1)*n]
			for i, b := range row {
				dst[i*typesize+j] = b
			}
		}
	}
	copy(dst[n*typesize:len(src)], src[n*typesize:])
}

// transBit8x8 is TRANS_BIT_8X8: transpose the 8x8 bit matrix held in x
// (byte m of the little-endian word is row m). It is its own inverse.
func transBit8x8(x uint64) uint64 {
	t := (x ^ (x >> 7)) & 0x00AA00AA00AA00AA
	x = x ^ t ^ (t << 7)
	t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC
	x = x ^ t ^ (t << 14)
	t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0
	x = x ^ t ^ (t << 28)
	return x
}

// bitTranspose is bshuf_trans_bit_elem for n elements (n a multiple of 8):
// output row 8*j+k (n/8 bytes) holds bit k of byte j of every element,
// element i at bit i%8 of byte i/8. Like the C code it first gathers byte j
// of every element (a byte shuffle, into scratch), then transposes 8x8 bit
// blocks of each byte row.
func bitTranspose(typesize, n int, src, dst, scratch []byte) {
	rows := src[:n*typesize]
	if typesize > 1 {
		rows = scratch[:n*typesize]
		shuffle(typesize, src[:n*typesize], rows)
	}
	nb := n / 8
	for j := range typesize {
		row := rows[j*n : (j+1)*n]
		out := dst[j*8*nb : (j+1)*8*nb]
		o0, o1, o2, o3 := out[0:nb], out[nb:2*nb], out[2*nb:3*nb], out[3*nb:4*nb]
		o4, o5, o6, o7 := out[4*nb:5*nb], out[5*nb:6*nb], out[6*nb:7*nb], out[7*nb:8*nb]
		for r := range o0 {
			x := transBit8x8(binary.LittleEndian.Uint64(row[8*r:]))
			o0[r], o1[r], o2[r], o3[r] = byte(x), byte(x>>8), byte(x>>16), byte(x>>24)
			o4[r], o5[r], o6[r], o7[r] = byte(x>>32), byte(x>>40), byte(x>>48), byte(x>>56)
		}
	}
}

// bitUntranspose is bshuf_untrans_bit_elem, the inverse of bitTranspose.
func bitUntranspose(typesize, n int, src, dst, scratch []byte) {
	rows := dst[:n*typesize]
	if typesize > 1 {
		rows = scratch[:n*typesize]
	}
	nb := n / 8
	for j := range typesize {
		in := src[j*8*nb : (j+1)*8*nb]
		i0, i1, i2, i3 := in[0:nb], in[nb:2*nb], in[2*nb:3*nb], in[3*nb:4*nb]
		i4, i5, i6, i7 := in[4*nb:5*nb], in[5*nb:6*nb], in[6*nb:7*nb], in[7*nb:8*nb]
		row := rows[j*n : (j+1)*n]
		for r := range i0 {
			x := uint64(i0[r]) | uint64(i1[r])<<8 | uint64(i2[r])<<16 | uint64(i3[r])<<24 |
				uint64(i4[r])<<32 | uint64(i5[r])<<40 | uint64(i6[r])<<48 | uint64(i7[r])<<56
			binary.LittleEndian.PutUint64(row[8*r:], transBit8x8(x))
		}
	}
	if typesize > 1 {
		unshuffle(typesize, rows, dst[:n*typesize])
	}
}

// bitshuffle is blosc_internal_bitshuffle (c-blosc 1.x): when the number of
// whole elements is a multiple of 8 they are bit-transposed and the trailing
// bytes copied; otherwise the whole block is copied unchanged. scratch must
// hold len(src) bytes.
func bitshuffle(typesize int, src, dst, scratch []byte) {
	n := len(src) / typesize
	if n%8 != 0 {
		copy(dst, src)
		return
	}
	bitTranspose(typesize, n, src, dst, scratch)
	copy(dst[n*typesize:len(src)], src[n*typesize:])
}

// bitunshuffle reverses bitshuffle. Blosc format version 2 (c-blosc 1.x)
// stores blocks whose element count is not a multiple of 8 unchanged; later
// format versions (c-blosc2) transpose the largest multiple of 8 elements
// and store the remainder verbatim. scratch must hold len(src) bytes.
func bitunshuffle(typesize int, src, dst []byte, version byte, scratch []byte) {
	n := len(src) / typesize
	if n%8 != 0 {
		if version <= 2 {
			copy(dst, src)
			return
		}
		n -= n % 8
	}
	bitUntranspose(typesize, n, src, dst, scratch)
	copy(dst[n*typesize:len(src)], src[n*typesize:])
}
