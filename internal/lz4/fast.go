package lz4

// Port of LZ4_compress_fast / LZ4_compress_generic_validated (lz4.c, LZ4
// 1.10.0) for a freshly initialised state (noDict, noDictIssue,
// currentOffset 0), on a 64-bit little-endian platform.

const (
	hashLog         = 12                     // LZ4_HASHLOG = LZ4_MEMORY_USAGE(14) - 2
	limit64K        = 64<<10 + (mfLimit - 1) // LZ4_64Klimit
	skipTrigger     = 6                      // LZ4_skipTrigger
	accelerationMax = 65537                  // LZ4_ACCELERATION_MAX
	prime4bytes     = 2654435761             // used by LZ4_hash4
	prime5bytes     = 889523592379           // used by LZ4_hash5
	u16TableSize    = 1 << (hashLog + 1)     // byU16 table entries
	u32TableSize    = 1 << hashLog           // byU32 table entries
)

// CompressFast compresses src into a new LZ4 block exactly as
// LZ4_compress_fast(src, dst, len(src), LZ4_compressBound(len(src)),
// acceleration) does. Acceleration values below 1 mean 1 and values above
// 65537 are clamped, as in C. It returns nil if len(src) > MaxInputSize.
func CompressFast(src []byte, acceleration int) []byte {
	bound := CompressBound(len(src))
	if bound == 0 {
		return nil
	}
	dst := make([]byte, bound)
	return shrink(dst[:CompressFastInto(dst, src, acceleration)])
}

// shrink returns b, copied into a right-sized slice when its backing array
// is much larger (so that callers do not pin a worst-case buffer).
func shrink(b []byte) []byte {
	if len(b) < cap(b)-cap(b)/8 {
		return append([]byte(nil), b...)
	}
	return b
}

// CompressFastInto is LZ4_compress_fast(src, dst, len(src), len(dst),
// acceleration): it compresses src into dst and returns the compressed size,
// or 0 if the result does not fit. When len(dst) < CompressBound(len(src))
// the C "limitedOutput" mode is used, whose overflow checks are conservative
// (it may give up although the block would have fit); this function gives up
// in exactly the same situations.
func CompressFastInto(dst, src []byte, acceleration int) int {
	if acceleration < 1 {
		acceleration = 1
	}
	if acceleration > accelerationMax {
		acceleration = accelerationMax
	}
	if len(src) > MaxInputSize {
		return 0
	}
	limited := len(dst) < CompressBound(len(src))
	if len(src) == 0 {
		// LZ4_compress_generic: an empty input is a single zero token.
		if limited && len(dst) <= 0 {
			return 0
		}
		dst[0] = 0
		return 1
	}
	if len(src) < limit64K {
		var table [u16TableSize]uint16
		return compressGeneric(dst, src, acceleration, limited, table[:], nil)
	}
	var table [u32TableSize]uint32
	return compressGeneric(dst, src, acceleration, limited, nil, table[:])
}

// compressGeneric is LZ4_compress_generic_validated. Exactly one of t16
// (tableType byU16, LZ4_hash4 on 4 bytes) and t32 (byU32, LZ4_hash5 on the
// low 5 bytes of an 8-byte read) is non-nil. Positions are stored as indexes
// from the start of src (base == source because currentOffset == 0).
func compressGeneric(dst, src []byte, acceleration int, limited bool, t16 []uint16, t32 []uint32) int {
	u16 := t16 != nil
	hash := func(p int) uint32 {
		if u16 {
			return (read32(src, p) * prime4bytes) >> (32 - (hashLog + 1))
		}
		return uint32(((read64(src, p) << 24) * prime5bytes) >> (64 - hashLog))
	}
	get := func(h uint32) int {
		if u16 {
			return int(t16[h])
		}
		return int(t32[h])
	}
	put := func(h uint32, idx int) {
		if u16 {
			t16[h] = uint16(idx)
		} else {
			t32[h] = uint32(idx)
		}
	}

	olimit := len(dst)
	iend := len(src)
	mflimitPlusOne := iend - mfLimit + 1
	matchlimit := iend - lastLiterals
	anchor, op := 0, 0

	if iend >= minLength {
		// First byte.
		put(hash(0), 0)
		ip := 1
		forwardH := hash(ip)

	mainLoop:
		for {
			var match int

			// Find a match.
			{
				forwardIP := ip
				step := 1
				searchMatchNb := acceleration << skipTrigger
				for {
					h := forwardH
					current := forwardIP
					matchIndex := get(h)
					ip = forwardIP
					forwardIP += step
					step = searchMatchNb >> skipTrigger
					searchMatchNb++

					if forwardIP > mflimitPlusOne {
						break mainLoop // to last literals
					}

					match = matchIndex
					forwardH = hash(forwardIP)
					put(h, current)

					// byU16 inputs are < 64 KiB, so only byU32 needs the
					// distance check.
					if !u16 && matchIndex+distanceMax < current {
						continue
					}
					if read32(src, match) == read32(src, ip) {
						break
					}
				}
			}

			// Catch up.
			if match > 0 && src[ip-1] == src[match-1] {
				for {
					ip--
					match--
					if !(ip > anchor && match > 0 && src[ip-1] == src[match-1]) {
						break
					}
				}
			}

			// Encode literals.
			token := op
			{
				litLength := ip - anchor
				op++
				if limited && op+litLength+(2+1+lastLiterals)+litLength/255 > olimit {
					return 0
				}
				if litLength >= runMask {
					l := litLength - runMask
					dst[token] = runMask << mlBits
					for ; l >= 255; l -= 255 {
						dst[op] = 255
						op++
					}
					dst[op] = byte(l)
					op++
				} else {
					dst[token] = byte(litLength << mlBits)
				}
				op += copy(dst[op:], src[anchor:ip])
			}

			// _next_match loop: encode a match, then try the next position
			// immediately (a sequence with zero literals).
			for {
				off := ip - match
				dst[op] = byte(off)
				dst[op+1] = byte(off >> 8)
				op += 2

				matchCode := count(src, ip+minMatch, match+minMatch, matchlimit)
				ip += matchCode + minMatch
				if limited && op+(1+lastLiterals)+(matchCode+240)/255 > olimit {
					return 0
				}
				if matchCode >= mlMask {
					dst[token] += mlMask
					matchCode -= mlMask
					for ; matchCode >= 255; matchCode -= 255 {
						dst[op] = 255
						op++
					}
					dst[op] = byte(matchCode)
					op++
				} else {
					dst[token] += byte(matchCode)
				}

				anchor = ip

				// Test end of chunk.
				if ip >= mflimitPlusOne {
					break mainLoop
				}

				// Fill table.
				put(hash(ip-2), ip-2)

				// Test next position.
				h := hash(ip)
				current := ip
				matchIndex := get(h)
				match = matchIndex
				put(h, current)
				if (u16 || matchIndex+distanceMax >= current) && read32(src, match) == read32(src, ip) {
					token = op
					op++
					dst[token] = 0
					continue
				}
				break
			}

			// Prepare next loop.
			ip++
			forwardH = hash(ip)
		}
	}

	// Encode last literals.
	lastRun := iend - anchor
	if limited && op+lastRun+1+(lastRun+255-runMask)/255 > olimit {
		return 0
	}
	if lastRun >= runMask {
		acc := lastRun - runMask
		dst[op] = runMask << mlBits
		op++
		for ; acc >= 255; acc -= 255 {
			dst[op] = 255
			op++
		}
		dst[op] = byte(acc)
		op++
	} else {
		dst[op] = byte(lastRun << mlBits)
		op++
	}
	op += copy(dst[op:], src[anchor:])
	return op
}
