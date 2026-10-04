package lz4

import (
	"math/bits"
	"sync"
)

// Port of LZ4_compress_HC (lz4hc.c, LZ4 1.10.0) for a freshly initialised
// state without dictionary (noDictCtx, favorDecSpeed off), on a 64-bit
// little-endian platform.
//
// LZ4_initStreamHC zeroes the whole state (so, unlike LZ4HC_clearTables, the
// chain table starts as zeros) and LZ4HC_init_internal then places the
// first input byte at index 64 KiB: dictLimit == lowLimit == nextToUpdate ==
// hcBase. Index i therefore designates src[i-hcBase]. Since there is no
// dictionary, every candidate below hcBase is rejected by the distance
// checks and the external-dictionary branches of the C code are unreachable;
// they are omitted.

const (
	hcHashLog     = 15 // LZ4HC_HASH_LOG
	hcHashSize    = 1 << hcHashLog
	hcChainSize   = 1 << 16 // LZ4HC_MAXD
	hcBase        = 64 << 10
	midHashLog    = hcHashLog - 1 // LZ4MID_HASHLOG
	midHashSize   = 1 << midHashLog
	optNum        = 1 << 12 // LZ4_OPT_NUM
	trailingLits  = 3       // TRAILING_LITERALS
	optimalML     = mlMask - 1 + minMatch
	hcClevelMax   = 12
	hcClevelDef   = 9
	hcClevelOptMn = 10
)

type hcStrategy int

const (
	stratMid hcStrategy = iota
	stratHC
	stratOpt
)

// clTable is k_clTable: strategy, nbSearches and targetLength per level.
var clTable = [hcClevelMax + 1]struct {
	strat        hcStrategy
	nbSearches   int
	targetLength int
}{
	{stratMid, 2, 16}, // 0, unused
	{stratMid, 2, 16}, // 1
	{stratMid, 2, 16}, // 2
	{stratHC, 4, 16},  // 3
	{stratHC, 8, 16},  // 4
	{stratHC, 16, 16}, // 5
	{stratHC, 32, 16}, // 6
	{stratHC, 64, 16}, // 7
	{stratHC, 128, 16},
	{stratHC, 256, 16}, // 9
	{stratOpt, 96, 64}, // 10
	{stratOpt, 512, 128},
	{stratOpt, 16384, optNum}, // 12
}

type hcMatch struct{ len, off, back int }

type hcCtx struct {
	hashTable    [hcHashSize]uint32
	chainTable   [hcChainSize]uint16
	nextToUpdate uint32
	src          []byte
	opt          []hcOpt
}

type hcOpt struct{ price, off, mlen, litlen int }

var hcPool = sync.Pool{New: func() any { return new(hcCtx) }}

// CompressHC compresses src into a new LZ4 block exactly as
// LZ4_compress_HC(src, dst, len(src), LZ4_compressBound(len(src)), level)
// does. Levels below 1 mean 9 (LZ4HC_CLEVEL_DEFAULT) and levels above 12 are
// clamped to 12. It returns nil if len(src) > MaxInputSize.
func CompressHC(src []byte, level int) []byte {
	bound := CompressBound(len(src))
	if bound == 0 {
		return nil
	}
	dst := make([]byte, bound)
	return shrink(dst[:CompressHCInto(dst, src, level)])
}

// CompressHCInto is LZ4_compress_HC(src, dst, len(src), len(dst), level):
// it returns the compressed size, or 0 when the C function would fail to fit
// the output into len(dst) bytes (limitedOutput mode, used when
// len(dst) < CompressBound(len(src)), with the same conservative checks).
func CompressHCInto(dst, src []byte, level int) int {
	if len(src) > MaxInputSize {
		return 0
	}
	if level < 1 {
		level = hcClevelDef
	}
	if level > hcClevelMax {
		level = hcClevelMax
	}
	limited := len(dst) < CompressBound(len(src))

	c := hcPool.Get().(*hcCtx)
	defer func() {
		c.src = nil
		hcPool.Put(c)
	}()
	// LZ4_initStreamHC + LZ4HC_init_internal.
	clear(c.hashTable[:])
	clear(c.chainTable[:])
	c.nextToUpdate = hcBase
	c.src = src

	p := clTable[level]
	switch p.strat {
	case stratMid:
		return c.compressMid(dst, limited)
	case stratHC:
		return c.compressHashChain(dst, p.nbSearches, limited)
	default:
		return c.compressOptimal(dst, p.nbSearches, p.targetLength, limited, level >= hcClevelMax)
	}
}

func hcHash(v uint32) uint32 { return (v * prime4bytes) >> (32 - hcHashLog) }

// encodeSequence is LZ4HC_encodeSequence. It writes one sequence (literals
// src[*anchor:*ip], then a match of matchLength at offset) and advances ip
// and anchor past the match. It reports false on a (conservative) output
// overflow in limited mode, in which case only *op may have moved.
func encodeSequence(dst, src []byte, ip, op, anchor *int, matchLength, offset int, limited bool, oend int) bool {
	token := *op
	o := *op + 1
	length := *ip - *anchor
	if limited && o+length/255+length+(2+1+lastLiterals) > oend {
		*op = o
		return false
	}
	if length >= runMask {
		l := length - runMask
		dst[token] = runMask << mlBits
		for ; l >= 255; l -= 255 {
			dst[o] = 255
			o++
		}
		dst[o] = byte(l)
		o++
	} else {
		dst[token] = byte(length << mlBits)
	}
	o += copy(dst[o:], src[*anchor:*ip])

	dst[o] = byte(offset)
	dst[o+1] = byte(offset >> 8)
	o += 2

	length = matchLength - minMatch
	if limited && o+length/255+(1+lastLiterals) > oend {
		*op = o
		return false
	}
	if length >= mlMask {
		dst[token] += mlMask
		length -= mlMask
		for ; length >= 510; length -= 510 {
			dst[o] = 255
			dst[o+1] = 255
			o += 2
		}
		if length >= 255 {
			length -= 255
			dst[o] = 255
			o++
		}
		dst[o] = byte(length)
		o++
	} else {
		dst[token] += byte(length)
	}
	*op = o
	*ip += matchLength
	*anchor = *ip
	return true
}

// lastLiteralsHC encodes the final literal run as the _last_literals blocks
// of the HC compressors do (limitedOutput or notLimited only).
func lastLiteralsHC(dst, src []byte, op, anchor int, limited bool) int {
	lastRunSize := len(src) - anchor
	llAdd := (lastRunSize + 255 - runMask) / 255
	if limited && op+1+llAdd+lastRunSize > len(dst) {
		return 0
	}
	if lastRunSize >= runMask {
		acc := lastRunSize - runMask
		dst[op] = runMask << mlBits
		op++
		for ; acc >= 255; acc -= 255 {
			dst[op] = 255
			op++
		}
		dst[op] = byte(acc)
		op++
	} else {
		dst[op] = byte(lastRunSize << mlBits)
		op++
	}
	op += copy(dst[op:], src[anchor:])
	return op
}

// ---------------------------------------------------------------------------
// Level 1-2: LZ4MID_compress.

func midHash4(v uint32) uint32 { return (v * prime4bytes) >> (32 - midHashLog) }

// midHash7 is LZ4MID_hash7: it hashes the low 56 bits of a little-endian
// 8-byte read.
func midHash7(v uint64) uint32 {
	return uint32(((v << (64 - 56)) * 58295818150454627) >> (64 - midHashLog))
}

func (c *hcCtx) compressMid(dst []byte, limited bool) int {
	src := c.src
	hash4Table := c.hashTable[:midHashSize]
	hash8Table := c.hashTable[midHashSize : 2*midHashSize]
	addPos8 := func(p int, idx uint32) { hash8Table[midHash7(read64(src, p))] = idx }
	addPos4 := func(p int, idx uint32) { hash4Table[midHash4(read32(src, p))] = idx }

	iend := len(src)
	mflimit := iend - mfLimit
	matchlimit := iend - lastLiterals
	ilimit := iend - 8 // LZ4MID_HASHSIZE
	ilimitIdx := uint32(ilimit) + hcBase
	ip, anchor, op := 0, 0, 0
	oend := len(dst)
	const prefixIdx = hcBase

	if iend >= minLength {
		for ip <= mflimit {
			ipIndex := uint32(ip) + hcBase
			var matchLength int
			var matchDistance uint32
			found := false

			// Search long match.
			{
				h8 := midHash7(read64(src, ip))
				pos8 := hash8Table[h8]
				hash8Table[h8] = ipIndex
				if ipIndex-pos8 <= distanceMax && pos8 >= prefixIdx {
					matchLength = count(src, ip, int(pos8-hcBase), matchlimit)
					if matchLength >= minMatch {
						matchDistance = ipIndex - pos8
						found = true
					}
				}
			}
			// Search short match.
			if !found {
				h4 := midHash4(read32(src, ip))
				pos4 := hash4Table[h4]
				hash4Table[h4] = ipIndex
				if ipIndex-pos4 <= distanceMax && pos4 >= prefixIdx {
					matchLength = count(src, ip, int(pos4-hcBase), matchlimit)
					if matchLength >= minMatch {
						// Short match found, let's just check ip+1 for longer.
						h8 := midHash7(read64(src, ip+1))
						pos8 := hash8Table[h8]
						m2Distance := ipIndex + 1 - pos8
						matchDistance = ipIndex - pos4
						if m2Distance <= distanceMax && pos8 >= prefixIdx && ip < mflimit {
							ml2 := count(src, ip+1, int(pos8-hcBase), matchlimit)
							if ml2 > matchLength {
								hash8Table[h8] = ipIndex + 1
								ip++
								matchLength = ml2
								matchDistance = m2Distance
							}
						}
						found = true
					}
				}
			}
			if !found {
				ip += 1 + (ip-anchor)>>9 // skip faster over incompressible data
				continue
			}

			// _lz4mid_encode_sequence: catch back.
			for ip > anchor && uint32(ip) > matchDistance && src[ip-1] == src[ip-int(matchDistance)-1] {
				ip--
				matchLength++
			}

			// Fill table with beginning of match. Note: the C code uses the
			// index computed at the top of the loop (ipIndex), not the index
			// of the possibly moved ip.
			addPos8(ip+1, ipIndex+1)
			addPos8(ip+2, ipIndex+2)
			addPos4(ip+1, ipIndex+1)

			if !encodeSequence(dst, src, &ip, &op, &anchor, matchLength, int(matchDistance), limited, oend) {
				return 0 // _lz4mid_dest_overflow with limitedOutput
			}

			// Fill table with end of match.
			endMatchIdx := uint32(ip) + hcBase
			posM2 := endMatchIdx - 2
			if posM2 < ilimitIdx {
				if ip > 5 {
					addPos8(ip-5, endMatchIdx-5)
				}
				addPos8(ip-3, endMatchIdx-3)
				addPos8(ip-2, endMatchIdx-2)
				addPos4(ip-2, endMatchIdx-2)
				addPos4(ip-1, endMatchIdx-1)
			}
		}
	}
	return lastLiteralsHC(dst, src, op, anchor, limited)
}

// ---------------------------------------------------------------------------
// Hash chain match finder.

// insert is LZ4HC_Insert: update the chains up to ip (excluded).
func (c *hcCtx) insert(ip int) {
	src := c.src
	target := uint32(ip) + hcBase
	for idx := c.nextToUpdate; idx < target; idx++ {
		h := hcHash(read32(src, int(idx-hcBase)))
		delta := idx - c.hashTable[h]
		if delta > distanceMax {
			delta = distanceMax
		}
		c.chainTable[uint16(idx)] = uint16(delta)
		c.hashTable[h] = idx
	}
	c.nextToUpdate = target
}

// countBack is LZ4HC_countBack: minus the number of equal bytes before ip
// and match, not going below iMin/mMin.
func countBack(src []byte, ip, match, iMin, mMin int) int {
	back := 0
	lo := max(iMin-ip, mMin-match)
	for back-lo > 3 {
		if v := read32(src, ip+back-4) ^ read32(src, match+back-4); v != 0 {
			return back - bits.LeadingZeros32(v)>>3
		}
		back -= 4
	}
	for back > lo && src[ip+back-1] == src[match+back-1] {
		back--
	}
	return back
}

// countPattern is LZ4HC_countPattern: the number of bytes from ip matching
// the repeated 4-byte pattern (read as little endian), up to iEnd.
func countPattern(src []byte, ip, iEnd int, pattern32 uint32) int {
	start := ip
	pattern := uint64(pattern32) | uint64(pattern32)<<32
	for ip < iEnd-7 {
		if diff := read64(src, ip) ^ pattern; diff != 0 {
			ip += bits.TrailingZeros64(diff) >> 3
			return ip - start
		}
		ip += 8
	}
	pb := pattern
	for ip < iEnd && src[ip] == byte(pb) {
		ip++
		pb >>= 8
	}
	return ip - start
}

// reverseCountPattern is LZ4HC_reverseCountPattern.
func reverseCountPattern(src []byte, ip, iLow int, pattern uint32) int {
	start := ip
	for ip >= iLow+4 {
		if read32(src, ip-4) != pattern {
			break
		}
		ip -= 4
	}
	for b := 3; b >= 0 && ip > iLow; b-- {
		if src[ip-1] != byte(pattern>>(8*b)) {
			break
		}
		ip--
	}
	return start - ip
}

// protectDictEnd is LZ4HC_protectDictEnd (unsigned arithmetic).
func protectDictEnd(dictLimit, matchIndex uint32) bool {
	return dictLimit-1-matchIndex >= 3
}

type repeatState int

const (
	repUntested repeatState = iota
	repNot
	repConfirmed
)

// insertAndGetWiderMatch is LZ4HC_InsertAndGetWiderMatch (noDictCtx,
// favorCompressionRatio).
func (c *hcCtx) insertAndGetWiderMatch(ip, iLowLimit, iHighLimit, longest, maxNbAttempts int, patternAnalysis, chainSwap bool) hcMatch {
	src := c.src
	chainTable := &c.chainTable
	const prefixIdx = hcBase
	ipIndex := uint32(ip) + hcBase
	withinStartDistance := hcBase+(distanceMax+1) > ipIndex
	lowestMatchIndex := uint32(hcBase)
	if !withinStartDistance {
		lowestMatchIndex = ipIndex - distanceMax
	}
	lookBackLength := ip - iLowLimit
	nbAttempts := maxNbAttempts
	matchChainPos := uint32(0)
	pattern := read32(src, ip)
	repeat := repUntested
	srcPatternLength := 0
	offset, sBack := 0, 0

	// First match.
	c.insert(ip)
	matchIndex := c.hashTable[hcHash(pattern)]

	for matchIndex >= lowestMatchIndex && nbAttempts > 0 {
		matchLength := 0
		nbAttempts--
		// matchIndex >= lowestMatchIndex >= prefixIdx: within the prefix.
		matchPtr := int(matchIndex - hcBase)
		if read16(src, iLowLimit+longest-1) == read16(src, matchPtr-lookBackLength+longest-1) {
			if read32(src, matchPtr) == pattern {
				back := 0
				if lookBackLength != 0 {
					back = countBack(src, ip, matchPtr, iLowLimit, 0)
				}
				matchLength = minMatch + count(src, ip+minMatch, matchPtr+minMatch, iHighLimit)
				matchLength -= back
				if matchLength > longest {
					longest = matchLength
					offset = int(ipIndex - matchIndex)
					sBack = back
				}
			}
		}

		if chainSwap && matchLength == longest { // better match => select a better chain
			if matchIndex+uint32(longest) <= ipIndex {
				const kTrigger = 4
				distanceToNextMatch := uint32(1)
				end := longest - minMatch + 1
				step := 1
				accel := 1 << kTrigger
				for pos := 0; pos < end; pos += step {
					candidateDist := uint32(chainTable[uint16(matchIndex+uint32(pos))])
					step = accel >> kTrigger
					accel++
					if candidateDist > distanceToNextMatch {
						distanceToNextMatch = candidateDist
						matchChainPos = uint32(pos)
						accel = 1 << kTrigger
					}
				}
				if distanceToNextMatch > 1 {
					if distanceToNextMatch > matchIndex {
						break // avoid overflow
					}
					matchIndex -= distanceToNextMatch
					continue
				}
			}
		}

		distNextMatch := uint32(chainTable[uint16(matchIndex)])
		if patternAnalysis && distNextMatch == 1 && matchChainPos == 0 {
			matchCandidateIdx := matchIndex - 1
			// May be a repeated pattern.
			if repeat == repUntested {
				if pattern&0xFFFF == pattern>>16 && pattern&0xFF == pattern>>24 {
					repeat = repConfirmed
					srcPatternLength = countPattern(src, ip+4, iHighLimit, pattern) + 4
				} else {
					repeat = repNot
				}
			}
			if repeat == repConfirmed && matchCandidateIdx >= lowestMatchIndex && protectDictEnd(prefixIdx, matchCandidateIdx) {
				mPtr := int(matchCandidateIdx - hcBase)
				if read32(src, mPtr) == pattern { // good candidate
					forwardPatternLength := countPattern(src, mPtr+4, iHighLimit, pattern) + 4
					backLength := reverseCountPattern(src, mPtr, 0, pattern)
					// Limit backLength not go further than lowestMatchIndex.
					backLength = int(matchCandidateIdx - max(matchCandidateIdx-uint32(backLength), lowestMatchIndex))
					currentSegmentLength := backLength + forwardPatternLength
					if currentSegmentLength >= srcPatternLength && forwardPatternLength <= srcPatternLength {
						// Best position, full pattern, might be followed by more match.
						newMatchIndex := matchCandidateIdx + uint32(forwardPatternLength) - uint32(srcPatternLength)
						if protectDictEnd(prefixIdx, newMatchIndex) {
							matchIndex = newMatchIndex
						} else {
							matchIndex = prefixIdx
						}
					} else {
						// Farthest position in current segment.
						newMatchIndex := matchCandidateIdx - uint32(backLength)
						if !protectDictEnd(prefixIdx, newMatchIndex) {
							matchIndex = prefixIdx
						} else {
							matchIndex = newMatchIndex
							if lookBackLength == 0 { // no back possible
								maxML := min(currentSegmentLength, srcPatternLength)
								if longest < maxML {
									if uint32(ip)+hcBase-matchIndex > distanceMax {
										break
									}
									longest = maxML
									offset = int(ipIndex - matchIndex)
								}
								distToNextPattern := uint32(chainTable[uint16(matchIndex)])
								if distToNextPattern > matchIndex {
									break // avoid overflow
								}
								matchIndex -= distToNextPattern
							}
						}
					}
					continue
				}
			}
		}

		// Follow current chain.
		matchIndex -= uint32(chainTable[uint16(matchIndex+matchChainPos)])
	}
	return hcMatch{len: longest, off: offset, back: sBack}
}

// ---------------------------------------------------------------------------
// Levels 3-9: LZ4HC_compress_hashChain.

func (c *hcCtx) compressHashChain(dst []byte, maxNbAttempts int, limited bool) int {
	src := c.src
	patternAnalysis := maxNbAttempts > 128 // levels 9+
	iend := len(src)
	mflimit := iend - mfLimit
	matchlimit := iend - lastLiterals
	oend := len(dst)
	ip, anchor, op := 0, 0, 0

	var (
		start0, start2, start3 int
		m0, m1, m2, m3         hcMatch
	)

	if iend < minLength {
		goto lastLits
	}

	for ip <= mflimit {
		m1 = c.insertAndGetWiderMatch(ip, ip, matchlimit, minMatch-1, maxNbAttempts, patternAnalysis, false)
		if m1.len < minMatch {
			ip++
			continue
		}

		// Saved, in case we would skip too much.
		start0 = ip
		m0 = m1

	search2:
		if ip+m1.len <= mflimit {
			start2 = ip + m1.len - 2
			m2 = c.insertAndGetWiderMatch(start2, ip, matchlimit, m1.len, maxNbAttempts, patternAnalysis, false)
			start2 += m2.back
		} else {
			m2 = hcMatch{}
		}

		if m2.len <= m1.len { // no better match => encode ML1 immediately
			if !encodeSequence(dst, src, &ip, &op, &anchor, m1.len, m1.off, limited, oend) {
				return 0
			}
			continue
		}

		if start0 < ip { // first match was skipped at least once
			if start2 < ip+m0.len { // squeezing ML1 between ML0 (original ML1) and ML2
				ip = start0
				m1 = m0 // restore initial Match1
			}
		}

		// Here, start0 == ip.
		if start2-ip < 3 { // first match too small: removed
			ip = start2
			m1 = m2
			goto search2
		}

	search3:
		if start2-ip < optimalML {
			newML := m1.len
			if newML > optimalML {
				newML = optimalML
			}
			if ip+newML > start2+m2.len-minMatch {
				newML = start2 - ip + m2.len - minMatch
			}
			if correction := newML - (start2 - ip); correction > 0 {
				start2 += correction
				m2.len -= correction
			}
		}

		if start2+m2.len <= mflimit {
			start3 = start2 + m2.len - 3
			m3 = c.insertAndGetWiderMatch(start3, start2, matchlimit, m2.len, maxNbAttempts, patternAnalysis, false)
			start3 += m3.back
		} else {
			m3 = hcMatch{}
		}

		if m3.len <= m2.len { // no better match => encode ML1 and ML2
			if start2 < ip+m1.len {
				m1.len = start2 - ip
			}
			if !encodeSequence(dst, src, &ip, &op, &anchor, m1.len, m1.off, limited, oend) {
				return 0
			}
			ip = start2
			if !encodeSequence(dst, src, &ip, &op, &anchor, m2.len, m2.off, limited, oend) {
				return 0
			}
			continue
		}

		if start3 < ip+m1.len+3 { // not enough space for match 2: remove it
			if start3 >= ip+m1.len { // can write Seq1 immediately ==> Seq2 is removed, so Seq3 becomes Seq1
				if start2 < ip+m1.len {
					correction := ip + m1.len - start2
					start2 += correction
					m2.len -= correction
					if m2.len < minMatch {
						start2 = start3
						m2 = m3
					}
				}
				if !encodeSequence(dst, src, &ip, &op, &anchor, m1.len, m1.off, limited, oend) {
					return 0
				}
				ip = start3
				m1 = m3
				start0 = start2
				m0 = m2
				goto search2
			}
			start2 = start3
			m2 = m3
			goto search3
		}

		// OK, now we have 3 ascending matches; let's write the first one ML1.
		if start2 < ip+m1.len {
			if start2-ip < optimalML {
				if m1.len > optimalML {
					m1.len = optimalML
				}
				if ip+m1.len > start2+m2.len-minMatch {
					m1.len = start2 - ip + m2.len - minMatch
				}
				if correction := m1.len - (start2 - ip); correction > 0 {
					start2 += correction
					m2.len -= correction
				}
			} else {
				m1.len = start2 - ip
			}
		}
		if !encodeSequence(dst, src, &ip, &op, &anchor, m1.len, m1.off, limited, oend) {
			return 0
		}

		// ML2 becomes ML1.
		ip = start2
		m1 = m2
		// ML3 becomes ML2.
		start2 = start3
		m2 = m3
		// Let's find a new ML3.
		goto search3
	}

lastLits:
	return lastLiteralsHC(dst, src, op, anchor, limited)
}

// ---------------------------------------------------------------------------
// Levels 10-12: LZ4HC_compress_optimal.

func literalsPrice(litlen int) int {
	price := litlen
	if litlen >= runMask {
		price += 1 + (litlen-runMask)/255
	}
	return price
}

func sequencePrice(litlen, mlen int) int {
	price := 1 + 2 + literalsPrice(litlen) // token + 16-bit offset
	if mlen >= mlMask+minMatch {
		price += 1 + (mlen-(mlMask+minMatch))/255
	}
	return price
}

// findLongerMatch is LZ4HC_FindLongerMatch (favorCompressionRatio).
func (c *hcCtx) findLongerMatch(ip, iHighLimit, minLen, nbSearches int) hcMatch {
	md := c.insertAndGetWiderMatch(ip, ip, iHighLimit, minLen, nbSearches, true, true)
	if md.len <= minLen {
		return hcMatch{}
	}
	return md
}

func (c *hcCtx) compressOptimal(dst []byte, nbSearches, sufficientLen int, limited, fullUpdate bool) int {
	src := c.src
	if c.opt == nil {
		c.opt = make([]hcOpt, optNum+trailingLits)
	}
	opt := c.opt
	iend := len(src)
	mflimit := iend - mfLimit
	matchlimit := iend - lastLiterals
	oend := len(dst)
	ip, anchor, op := 0, 0, 0

	if sufficientLen >= optNum {
		sufficientLen = optNum - 1
	}

	for ip <= mflimit {
		llen := ip - anchor
		var bestMlen, bestOff, cur, lastMatchPos int

		firstMatch := c.findLongerMatch(ip, matchlimit, minMatch-1, nbSearches)
		if firstMatch.len == 0 {
			ip++
			continue
		}

		if firstMatch.len > sufficientLen {
			// Good enough solution: immediate encoding.
			if !encodeSequence(dst, src, &ip, &op, &anchor, firstMatch.len, firstMatch.off, limited, oend) {
				return 0
			}
			continue
		}

		// Set prices for first positions (literals).
		for rPos := range minMatch {
			opt[rPos] = hcOpt{price: literalsPrice(llen + rPos), off: 0, mlen: 1, litlen: llen + rPos}
		}
		// Set prices using initial match.
		for mlen := minMatch; mlen <= firstMatch.len; mlen++ {
			opt[mlen] = hcOpt{price: sequencePrice(llen, mlen), off: firstMatch.off, mlen: mlen, litlen: llen}
		}
		lastMatchPos = firstMatch.len
		for addLit := 1; addLit <= trailingLits; addLit++ {
			opt[lastMatchPos+addLit] = hcOpt{price: opt[lastMatchPos].price + literalsPrice(addLit), off: 0, mlen: 1, litlen: addLit}
		}

		// Check further positions.
		encodeNow := false
		for cur = 1; cur < lastMatchPos; cur++ {
			curPtr := ip + cur
			if curPtr > mflimit {
				break
			}
			if fullUpdate {
				// Not useful to search here if next position has same (or
				// lower) cost, unless cost rises sharply after.
				if opt[cur+1].price <= opt[cur].price && opt[cur+minMatch].price < opt[cur].price+3 {
					continue
				}
			} else if opt[cur+1].price <= opt[cur].price {
				continue
			}

			var newMatch hcMatch
			if fullUpdate {
				newMatch = c.findLongerMatch(curPtr, matchlimit, minMatch-1, nbSearches)
			} else {
				newMatch = c.findLongerMatch(curPtr, matchlimit, lastMatchPos-cur, nbSearches)
			}
			if newMatch.len == 0 {
				continue
			}

			if newMatch.len > sufficientLen || newMatch.len+cur >= optNum {
				// Immediate encoding.
				bestMlen = newMatch.len
				bestOff = newMatch.off
				lastMatchPos = cur + 1
				encodeNow = true
				break
			}

			// Before match: set price with literals at beginning.
			{
				baseLitlen := opt[cur].litlen
				for litlen := 1; litlen < minMatch; litlen++ {
					price := opt[cur].price - literalsPrice(baseLitlen) + literalsPrice(baseLitlen+litlen)
					pos := cur + litlen
					if price < opt[pos].price {
						opt[pos] = hcOpt{price: price, off: 0, mlen: 1, litlen: baseLitlen + litlen}
					}
				}
			}

			// Set prices using match at position = cur.
			{
				matchML := newMatch.len
				for ml := minMatch; ml <= matchML; ml++ {
					pos := cur + ml
					var price, ll int
					if opt[cur].mlen == 1 {
						ll = opt[cur].litlen
						if cur > ll {
							price = opt[cur-ll].price
						}
						price += sequencePrice(ll, ml)
					} else {
						ll = 0
						price = opt[cur].price + sequencePrice(0, ml)
					}
					if pos > lastMatchPos+trailingLits || price <= opt[pos].price {
						if ml == matchML && lastMatchPos < pos {
							lastMatchPos = pos
						}
						opt[pos] = hcOpt{price: price, off: newMatch.off, mlen: ml, litlen: ll}
					}
				}
			}
			// Complete following positions with literals.
			for addLit := 1; addLit <= trailingLits; addLit++ {
				opt[lastMatchPos+addLit] = hcOpt{price: opt[lastMatchPos].price + literalsPrice(addLit), off: 0, mlen: 1, litlen: addLit}
			}
		}

		if !encodeNow {
			bestMlen = opt[lastMatchPos].mlen
			bestOff = opt[lastMatchPos].off
			cur = lastMatchPos - bestMlen
		}

		// encode: reverse traversal, looking for the shortest path.
		{
			candidatePos := cur
			selectedMatchLength := bestMlen
			selectedOffset := bestOff
			for {
				nextMatchLength := opt[candidatePos].mlen // can be 1, means literal
				nextOffset := opt[candidatePos].off
				opt[candidatePos].mlen = selectedMatchLength
				opt[candidatePos].off = selectedOffset
				selectedMatchLength = nextMatchLength
				selectedOffset = nextOffset
				if nextMatchLength > candidatePos {
					break // last match elected, first match to encode
				}
				candidatePos -= nextMatchLength
			}
		}

		// Encode all recorded sequences in order.
		for rPos := 0; rPos < lastMatchPos; {
			ml := opt[rPos].mlen
			offset := opt[rPos].off
			if ml == 1 { // literal
				ip++
				rPos++
				continue
			}
			rPos += ml
			if !encodeSequence(dst, src, &ip, &op, &anchor, ml, offset, limited, oend) {
				return 0
			}
		}
	}
	return lastLiteralsHC(dst, src, op, anchor, limited)
}
