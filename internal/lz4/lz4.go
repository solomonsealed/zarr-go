// Package lz4 implements the LZ4 block format: a bounds-checked decoder and
// faithful ports of the reference compressors from LZ4 1.10.0 (the version
// bundled by c-blosc 1.21.7 / numcodecs 0.17.0):
//
//   - CompressFast ports LZ4_compress_fast (lz4.c, LZ4_compress_generic with
//     the byU16 hash table below LZ4_64Klimit and the 64-bit byU32 table
//     above it).
//   - CompressHC ports LZ4_compress_HC (lz4hc.c: LZ4MID_compress for levels
//     1-2, LZ4HC_compress_hashChain for 3-9 and LZ4HC_compress_optimal for
//     10-12).
//
// The compressors produce output byte-identical to the C library as built on
// 64-bit little-endian platforms (x86-64, arm64), which is what numcodecs
// ships, and they do so on every GOARCH (including wasm). The *Into variants
// take an output buffer whose length is the destination capacity and
// reproduce the C "limitedOutput" behaviour, including its conservative
// overflow checks: they return 0 when the C function would return 0 (output
// does not fit). c-blosc relies on this, as it compresses every split into a
// buffer no larger than the split.
//
// All functions are safe for concurrent use.
//
// The package is pure Go (no cgo, no unsafe) so it also builds for
// GOOS=js GOARCH=wasm.
package lz4

import (
	"encoding/binary"
	"errors"
	"math/bits"
)

// Format constants (lz4.c "Common Constants").
const (
	minMatch     = 4
	lastLiterals = 5  // the last 5 bytes of a block are always literals
	mfLimit      = 12 // no match may start within the last 12 bytes
	minLength    = mfLimit + 1
	mlBits       = 4
	mlMask       = 1<<mlBits - 1
	runMask      = 1<<(8-mlBits) - 1
	distanceMax  = 65535

	// MaxInputSize is LZ4_MAX_INPUT_SIZE.
	MaxInputSize = 0x7E000000
)

// ErrCorrupt is returned by Decompress for malformed input.
var ErrCorrupt = errors.New("lz4: corrupt block")

// CompressBound returns LZ4_compressBound(n): the maximum compressed size of
// an n-byte input, or 0 if n is negative or larger than MaxInputSize.
func CompressBound(n int) int {
	if n < 0 || n > MaxInputSize {
		return 0
	}
	return n + n/255 + 16
}

func read16(b []byte, i int) uint16 { return binary.LittleEndian.Uint16(b[i:]) }
func read32(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[i:]) }
func read64(b []byte, i int) uint64 { return binary.LittleEndian.Uint64(b[i:]) }

// count is LZ4_count: the number of equal bytes at b[in:] and b[match:],
// stopping at inLimit (exclusive bound for in).
func count(b []byte, in, match, inLimit int) int {
	start := in
	for in < inLimit-7 {
		if diff := read64(b, match) ^ read64(b, in); diff != 0 {
			return in - start + bits.TrailingZeros64(diff)>>3
		}
		in += 8
		match += 8
	}
	for in < inLimit && b[match] == b[in] {
		in++
		match++
	}
	return in - start
}

// copyMatch copies length bytes from dst[m:] to dst[op:] (m < op) with the
// forward, byte-by-byte semantics of LZ77 matches, so that offsets shorter
// than the length repeat the pattern. Overlapping copies double the copied
// prefix each round.
func copyMatch(dst []byte, op, m, length int) {
	end := op + length
	for op < end {
		op += copy(dst[op:end], dst[m:op])
	}
}

// Decompress decodes the raw LZ4 block src into dst, which must be exactly
// the decompressed size. It mirrors the acceptance rules of
// LZ4_decompress_safe (the last sequence holds only literals that end the
// input exactly, matches may not end within the last 5 output bytes, and
// literals ending within the last 12 output bytes must be the last
// sequence), additionally rejecting zero offsets. It never reads or writes
// out of bounds and never panics.
func Decompress(src, dst []byte) error {
	oend := len(dst)
	if oend == 0 {
		if len(src) == 1 && src[0] == 0 {
			return nil
		}
		return ErrCorrupt
	}
	iend := len(src)
	ip, op := 0, 0
	for {
		if ip >= iend {
			return ErrCorrupt
		}
		token := src[ip]
		ip++

		// Literals.
		length := int(token >> mlBits)
		if length == runMask {
			for {
				if ip >= iend {
					return ErrCorrupt
				}
				s := src[ip]
				ip++
				length += int(s)
				if length > iend { // cannot be satisfied by the input
					return ErrCorrupt
				}
				if s != 255 {
					break
				}
			}
		}
		if op+length > oend-mfLimit || ip+length > iend-(2+1+lastLiterals) {
			// Must be the last sequence: literals end the input exactly.
			if ip+length != iend || op+length > oend {
				return ErrCorrupt
			}
			copy(dst[op:], src[ip:iend])
			op += length
			if op != oend {
				return ErrCorrupt
			}
			return nil
		}
		copy(dst[op:op+length], src[ip:ip+length])
		ip += length
		op += length

		// Match.
		offset := int(src[ip]) | int(src[ip+1])<<8
		ip += 2
		length = int(token & mlMask)
		if length == mlMask {
			for {
				if ip >= iend-lastLiterals+1 {
					return ErrCorrupt
				}
				s := src[ip]
				ip++
				length += int(s)
				if length > oend {
					return ErrCorrupt
				}
				if s != 255 {
					break
				}
			}
		}
		length += minMatch
		if offset == 0 || offset > op || op+length > oend-lastLiterals {
			return ErrCorrupt
		}
		copyMatch(dst, op, op-offset, length)
		op += length
	}
}
