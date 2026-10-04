package blosc

import "encoding/binary"

// Port of blosclz.c from c-blosc 1.21.7.dev (commit 04c06fb, the version
// bundled by numcodecs 0.17.0). BloscLZ is a FastLZ derivative; the
// compressor first estimates the compression ratio of a part of the buffer
// ("entropy probing", get_cratio) and gives up early on data that would not
// compress well enough for the requested level.

const (
	blzMaxCopy        = 32
	blzMaxDistance    = 8191
	blzMaxFarDistance = 65535 + blzMaxDistance - 1
	blzHashLog        = 14
	blzHashLog2       = 12
)

func blzRead32(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[i:]) }

func blzHash(seq uint32, log uint) uint32 { return (seq * 2654435761) >> (32 - log) }

// blzGetRun is get_run: x is the byte before ip; it advances ip while the
// bytes at ref equal x, stopping at ipBound. It returns the position of the
// first mismatch (not one past it).
func blzGetRun(b []byte, ip, ipBound, ref int) int {
	x := b[ip-1]
	v := uint64(x) * 0x0101010101010101
	for ip < ipBound-8 {
		if binary.LittleEndian.Uint64(b[ref:]) != v {
			for b[ref] == x {
				ref++
				ip++
			}
			return ip
		}
		ip += 8
		ref += 8
	}
	for ip < ipBound && b[ref] == x {
		ref++
		ip++
	}
	return ip
}

// blzGetMatch is get_match (and get_match_16, which gives identical
// results): it returns one past the first position where b[ip] and b[ref]
// differ, or ipBound if they agree up to it.
func blzGetMatch(b []byte, ip, ipBound, ref int) int {
	for ip < ipBound-8 {
		if binary.LittleEndian.Uint64(b[ref:]) != binary.LittleEndian.Uint64(b[ip:]) {
			for {
				eq := b[ref] == b[ip]
				ref++
				ip++
				if !eq {
					return ip
				}
			}
		}
		ip += 8
		ref += 8
	}
	for ip < ipBound {
		eq := b[ref] == b[ip]
		ref++
		ip++
		if !eq {
			break
		}
	}
	return ip
}

func blzGetRunOrMatch(b []byte, ip, ipBound, ref int, run bool) int {
	if run {
		return blzGetRun(b, ip, ipBound, ref)
	}
	return blzGetMatch(b, ip, ipBound, ref)
}

// blzCratio is get_cratio: a guess of the compression ratio of the first
// min(maxlen, 4096) bytes of ibase, simulating the encoder's output size.
func blzCratio(ibase []byte, maxlen, minlen, ipshift int) float64 {
	const hashlen = 1 << blzHashLog2
	var htab [hashlen]uint16
	limit := min(maxlen, hashlen) // uint16_t in C; maxlen >= 0
	ipBound := limit - 1
	ipLimit := limit - 12
	oc := 5 // we start with literal copy
	copied := 4
	ip := 0
	literal := func(anchor int) {
		oc++
		ip = anchor + 1
		copied++
		if copied == blzMaxCopy {
			copied = 0
			oc++
		}
	}
	for ip < ipLimit {
		anchor := ip
		seq := blzRead32(ibase, ip)
		hval := blzHash(seq, blzHashLog2)
		ref := int(htab[hval])
		distance := anchor - ref
		htab[hval] = uint16(anchor)
		if distance == 0 || distance >= blzMaxFarDistance {
			literal(anchor)
			continue
		}
		if blzRead32(ibase, ref) != blzRead32(ibase, ip) {
			literal(anchor)
			continue
		}
		ref += 4
		ip = anchor + 4
		distance-- // distance is biased
		ip = blzGetRunOrMatch(ibase, ip, ipBound, ref, distance == 0)
		ip -= ipshift
		length := ip - anchor
		if length < minlen {
			literal(anchor)
			continue
		}
		if copied == 0 {
			oc--
		}
		copied = 0
		if length >= 7 {
			oc += (length-7)/255 + 1
		}
		if distance < blzMaxDistance {
			oc += 2
		} else {
			oc += 4
		}
		// Update the hash at match boundary.
		seq = blzRead32(ibase, ip)
		htab[blzHash(seq, blzHashLog2)] = uint16(ip)
		ip += 2
		oc++ // assuming literal copy
	}
	return float64(ip) / float64(oc)
}

var blzCratioThreshold = [10]float64{0, 2, 1.5, 1.2, 1.2, 1.2, 1.2, 1.15, 1.1, 1.0}

var blzHashLogs = [10]uint{0, blzHashLog - 2, blzHashLog - 1, blzHashLog, blzHashLog,
	blzHashLog, blzHashLog, blzHashLog, blzHashLog, blzHashLog}

// blosclzCompress is blosclz_compress(clevel, src, len(src), dst,
// maxout=len(dst), splitBlock). It returns the compressed size or 0 if the
// data is not worth compressing or does not fit. clevel must be 1..9.
func blosclzCompress(clevel int, src, dst []byte, splitBlock bool) int {
	length := len(src)
	maxout := len(dst)

	// Experiments say that checking 1/4 of the buffer is enough to figure
	// out the approximate cratio; probe at the end of the buffer.
	maxlen := length / 4
	shift := length - maxlen
	cratio := blzCratio(src[shift:], maxlen, 3, 3)
	if cratio < blzCratioThreshold[clevel] {
		return 0
	}

	ipshift, minlen := 4, 4
	if !splitBlock || cratio < 4 {
		ipshift, minlen = 3, 3
	}
	hashlog := blzHashLogs[clevel]

	if length < 16 || maxout < 66 {
		return 0
	}
	ipBound := length - 1
	ipLimit := length - 12

	var htabArr [1 << blzHashLog]uint32
	htab := htabArr[:1<<hashlog]

	// We start with literal copy.
	op := 0
	copied := 4
	dst[0] = blzMaxCopy - 1
	copy(dst[1:5], src[:4])
	op = 5
	ip := 4

	// literal is the LITERAL macro: copy src[anchor] and advance.
	literal := func(anchor int) bool {
		if op+2 > maxout {
			return false
		}
		dst[op] = src[anchor]
		op++
		ip = anchor + 1
		copied++
		if copied == blzMaxCopy {
			copied = 0
			dst[op] = blzMaxCopy - 1
			op++
		}
		return true
	}

	for ip < ipLimit {
		anchor := ip
		seq := blzRead32(src, ip)
		hval := blzHash(seq, hashlog)
		ref := int(htab[hval])
		distance := anchor - ref
		htab[hval] = uint32(anchor)

		if distance == 0 || distance >= blzMaxFarDistance {
			if !literal(anchor) {
				return 0
			}
			continue
		}
		// Is this a match? Check the first 4 bytes.
		if blzRead32(src, ref) != blzRead32(src, ip) {
			if !literal(anchor) {
				return 0
			}
			continue
		}
		ref += 4
		ip = anchor + 4
		distance-- // distance is biased
		// Get runs or matches; zero distance means a run.
		ip = blzGetRunOrMatch(src, ip, ipBound, ref, distance == 0)
		// Length is biased, '1' means a match of 3 bytes.
		ip -= ipshift
		length := ip - anchor
		// Encoding short lengths is expensive during decompression.
		if length < minlen || (length <= 5 && distance >= blzMaxDistance) {
			if !literal(anchor) {
				return 0
			}
			continue
		}

		if copied != 0 {
			dst[op-copied-1] = byte(copied - 1) // copy is biased, '0' means 1 byte copy
		} else {
			op-- // back, to overwrite the copy count
		}
		copied = 0

		// Encode the match (MATCH_SHORT, MATCH_LONG, MATCH_SHORT_FAR,
		// MATCH_LONG_FAR).
		if distance < blzMaxDistance {
			if length < 7 {
				if op+2 > maxout {
					return 0
				}
				dst[op] = byte(length<<5 + distance>>8)
				dst[op+1] = byte(distance)
				op += 2
			} else {
				if op+1 > maxout {
					return 0
				}
				dst[op] = byte(7<<5 + distance>>8)
				op++
				for length -= 7; length >= 255; length -= 255 {
					if op+1 > maxout {
						return 0
					}
					dst[op] = 255
					op++
				}
				if op+2 > maxout {
					return 0
				}
				dst[op] = byte(length)
				dst[op+1] = byte(distance)
				op += 2
			}
		} else {
			// Far away, but not yet in the another galaxy...
			distance -= blzMaxDistance
			if length < 7 {
				if op+4 > maxout {
					return 0
				}
				dst[op] = byte(length<<5 + 31)
				dst[op+1] = 255
				dst[op+2] = byte(distance >> 8)
				dst[op+3] = byte(distance)
				op += 4
			} else {
				if op+1 > maxout {
					return 0
				}
				dst[op] = 7<<5 + 31
				op++
				for length -= 7; length >= 255; length -= 255 {
					if op+1 > maxout {
						return 0
					}
					dst[op] = 255
					op++
				}
				if op+4 > maxout {
					return 0
				}
				dst[op] = byte(length)
				dst[op+1] = 255
				dst[op+2] = byte(distance >> 8)
				dst[op+3] = byte(distance)
				op += 4
			}
		}

		// Update the hash at match boundary.
		seq = blzRead32(src, ip)
		htab[blzHash(seq, hashlog)] = uint32(ip)
		ip++
		if clevel == 9 {
			// In some situations, including a second hash proves to be
			// useful; c-blosc only does it at clevel 9. Note: it hashes
			// the 3 bytes after ip (seq >> 8), not a fresh 4-byte read.
			seq >>= 8
			htab[blzHash(seq, hashlog)] = uint32(ip)
		}
		ip++

		if op+1 > maxout {
			return 0
		}
		dst[op] = blzMaxCopy - 1 // assuming literal copy
		op++
	}

	// Left-over as literal copy.
	for ip <= ipBound {
		if op+2 > maxout {
			return 0
		}
		dst[op] = src[ip]
		op++
		ip++
		copied++
		if copied == blzMaxCopy {
			copied = 0
			dst[op] = blzMaxCopy - 1
			op++
		}
	}

	// If we have copied something, adjust the copy length.
	if copied != 0 {
		dst[op-copied-1] = byte(copied - 1)
	} else {
		op--
	}

	// Marker for blosclz.
	dst[0] |= 1 << 5
	return op
}

// blosclzDecompress is blosclz_decompress: it decodes src into dst and
// returns the number of bytes written, or 0 on malformed input, with the
// same acceptance rules as the C function (in particular, a stream that ends
// right after a match header stops before copying that match). Unlike the
// C code it never reads or writes out of bounds.
func blosclzDecompress(src, dst []byte) int {
	ipLimit := len(src)
	opLimit := len(dst)
	if ipLimit == 0 {
		return 0
	}
	ip, op := 0, 0
	ctrl := int(src[ip]) & 31
	ip++
	for {
		if ctrl >= 32 {
			// Match.
			length := ctrl>>5 - 1
			ofs := (ctrl & 31) << 8
			if length == 7-1 {
				for {
					if ip+1 >= ipLimit {
						return 0
					}
					code := int(src[ip])
					ip++
					length += code
					if code != 255 {
						break
					}
				}
			} else if ip+1 >= ipLimit {
				return 0
			}
			code := int(src[ip])
			ip++
			length += 3
			ref := op - ofs - code
			// Match from 16-bit distance.
			if code == 255 && ofs == 31<<8 {
				if ip+1 >= ipLimit {
					return 0
				}
				ofs = int(src[ip])<<8 | int(src[ip+1])
				ip += 2
				ref = op - ofs - blzMaxDistance
			}
			if op+length > opLimit {
				return 0
			}
			if ref-1 < 0 {
				return 0
			}
			if ip >= ipLimit {
				break
			}
			ctrl = int(src[ip])
			ip++
			ref--
			// Forward LZ77 copy (memset for runs, copy_match/wild_copy in
			// C): overlapping copies double the copied prefix each round.
			for end := op + length; op < end; {
				op += copy(dst[op:end], dst[ref:op])
			}
		} else {
			// Literal run.
			ctrl++
			if op+ctrl > opLimit || ip+ctrl > ipLimit {
				return 0
			}
			copy(dst[op:op+ctrl], src[ip:ip+ctrl])
			op += ctrl
			ip += ctrl
			if ip >= ipLimit {
				break
			}
			ctrl = int(src[ip])
			ip++
		}
	}
	return op
}
