// Package bzip2 implements a pure-Go bzip2 compressor whose output is
// byte-for-byte identical to libbzip2 1.0.8, as produced by Python's
// bz2.compress(data, compresslevel).
//
// The implementation is a faithful port of libbzip2's compression path
// (bzlib.c, compress.c, blocksort.c, huffman.c): the initial run-length
// encoding, block sizing, the main/fallback block sort with the default work
// factor of 30, MTF/RLE2 coding, Huffman table selection and the bit-level
// stream format. Decompression is left to the standard library's
// compress/bzip2 package.
package bzip2

import (
	"bytes"
	"fmt"
)

// Constants from bzlib_private.h.
const (
	nRadix     = 2
	nQSort     = 12
	nShell     = 18
	nOvershoot = nRadix + nQSort + nShell + 2 // BZ_N_OVERSHOOT

	maxAlphaSize = 258 // BZ_MAX_ALPHA_SIZE
	nGroupsMax   = 6   // BZ_N_GROUPS
	groupSize    = 50  // BZ_G_SIZE
	nIters       = 4   // BZ_N_ITERS
	maxSelectors = 2 + 900000/groupSize

	runA = 0 // BZ_RUNA
	runB = 1 // BZ_RUNB

	lesserICost  = 0  // BZ_LESSER_ICOST
	greaterICost = 15 // BZ_GREATER_ICOST

	// workFactor is the value Python's BZ2Compressor (and the bzip2 tool)
	// end up using: BZ2_bzCompressInit maps workFactor 0 to 30.
	workFactor = 30
)

// Compress compresses data into a complete bzip2 stream using a block size of
// level*100k bytes (level must be in 1..9). The result is identical to
// Python's bz2.compress(data, compresslevel=level) with libbzip2 1.0.8.
func Compress(data []byte, level int) ([]byte, error) {
	if level < 1 || level > 9 {
		return nil, fmt.Errorf("bzip2: invalid compression level %d (must be 1-9)", level)
	}
	e := newEncoder(level, len(data))
	e.compress(data)
	out := e.bw.out
	if cap(out)-len(out) > len(out)/4+4096 {
		out = bytes.Clone(out) // don't pin a mostly unused buffer
	}
	return out, nil
}

// encoder holds the state of libbzip2's EState that matters for output.
type encoder struct {
	blockSize100k int
	nblockMAX     int32

	// Initial run-length encoder state (state_in_ch / state_in_len).
	// stateInCh == 256 means "no pending run".
	stateInCh  uint32
	stateInLen int32

	// Current block.
	block       []byte // nblock bytes plus nOvershoot bytes of slack
	nblock      int32
	blockCRC    uint32
	combinedCRC uint32
	inUse       [256]bool
	blockNo     int

	// Block sorting.
	ptr      []uint32 // sorted rotation order (arr1)
	quadrant []uint16 // mainSort's quadrant cache
	ftab     []uint32 // mainSort's 2-byte radix table / fallbackSort's bhtab
	eclass   []uint32 // fallbackSort's equivalence classes
	budget   int32
	origPtr  int32

	// MTF / Huffman stage.
	mtfv        []uint16
	nMTF        int32
	nInUse      int32
	unseqToSeq  [256]byte
	mtfFreq     [maxAlphaSize]int32
	selector    []byte
	selectorMtf []byte
	len         [nGroupsMax][maxAlphaSize]byte
	code        [nGroupsMax][maxAlphaSize]int32
	rfreq       [nGroupsMax][maxAlphaSize]int32

	bw bitWriter
}

func newEncoder(level, inputLen int) *encoder {
	e := &encoder{
		blockSize100k: level,
		nblockMAX:     int32(100000*level - 19),
	}
	// libbzip2 allocates room for 100000*level symbols. A block never holds
	// more than nblockMAX+4 symbols, and the initial RLE expands input by at
	// most 5/4, so small inputs get correspondingly small buffers.
	n := 100000 * level
	if m := inputLen + inputLen/4 + 8; m < n {
		n = m
	}
	e.block = make([]byte, n+nOvershoot)
	e.ptr = make([]uint32, n)
	e.mtfv = make([]uint16, n+1)
	nSel := n/groupSize + 2 // at most maxSelectors, since n <= 900000
	e.selector = make([]byte, nSel)
	e.selectorMtf = make([]byte, nSel)
	e.bw.out = make([]byte, 0, min(inputLen/4, 1<<20)+64)
	return e
}

// compress mirrors how Python drives libbzip2: one BZ2_bzCompress(BZ_RUN)
// pass over all of the input followed by BZ2_bzCompress(BZ_FINISH).
//
// Note the RUN/FINISH split matters in one corner case: if a block becomes
// full exactly when the input runs out, BZ_RUN compresses that block as a
// non-final block and the pending run (state_in_ch) ends up in a small extra
// final block, whereas a single BZ_FINISH call (BZ2_bzBuffToBuffCompress)
// would flush the pending run into the full block.
func (e *encoder) compress(data []byte) {
	e.initRL()
	e.prepareNewBlock()
	for i := 0; i < len(data); {
		ch := uint32(data[i])
		if ch == e.stateInCh && e.stateInLen < 255 {
			// Fast path: ADD_CHAR_TO_BLOCK would just increment
			// state_in_len, which leaves nblock (and so the block-full
			// check) unchanged.
			limit := min(int(255-e.stateInLen), len(data)-i)
			n := 1
			for n < limit && data[i+n] == data[i] {
				n++
			}
			e.stateInLen += int32(n)
			i += n
			continue
		}
		e.addChar(ch)
		i++
		if e.nblock >= e.nblockMAX {
			e.compressBlock(false)
			e.prepareNewBlock()
		}
	}
	e.flushRL()
	e.compressBlock(true)
}

func (e *encoder) prepareNewBlock() {
	e.nblock = 0
	e.blockCRC = 0xffffffff
	e.inUse = [256]bool{}
	e.blockNo++
}

func (e *encoder) initRL() {
	e.stateInCh = 256
	e.stateInLen = 0
}

func (e *encoder) flushRL() {
	if e.stateInCh < 256 {
		e.addPairToBlock()
	}
	e.initRL()
}

func (e *encoder) addPairToBlock() {
	ch := byte(e.stateInCh)
	for i := int32(0); i < e.stateInLen; i++ {
		e.blockCRC = e.blockCRC<<8 ^ crcTable[byte(e.blockCRC>>24)^ch]
	}
	e.inUse[ch] = true
	b := e.block[e.nblock:]
	switch e.stateInLen {
	case 1:
		b[0] = ch
		e.nblock++
	case 2:
		b[0], b[1] = ch, ch
		e.nblock += 2
	case 3:
		b[0], b[1], b[2] = ch, ch, ch
		e.nblock += 3
	default:
		e.inUse[e.stateInLen-4] = true
		b[0], b[1], b[2], b[3] = ch, ch, ch, ch
		b[4] = byte(e.stateInLen - 4)
		e.nblock += 5
	}
}

// addChar is ADD_CHAR_TO_BLOCK.
func (e *encoder) addChar(ch uint32) {
	if ch != e.stateInCh && e.stateInLen == 1 {
		c := byte(e.stateInCh)
		e.blockCRC = e.blockCRC<<8 ^ crcTable[byte(e.blockCRC>>24)^c]
		e.inUse[c] = true
		e.block[e.nblock] = c
		e.nblock++
		e.stateInCh = ch
	} else if ch != e.stateInCh || e.stateInLen == 255 {
		if e.stateInCh < 256 {
			e.addPairToBlock()
		}
		e.stateInCh = ch
		e.stateInLen = 1
	} else {
		// Not reached from compress, which extends runs itself.
		e.stateInLen++
	}
}

// compressBlock is BZ2_compressBlock.
func (e *encoder) compressBlock(isLastBlock bool) {
	if e.nblock > 0 {
		e.blockCRC = ^e.blockCRC
		e.combinedCRC = (e.combinedCRC<<1 | e.combinedCRC>>31) ^ e.blockCRC
		e.blockSort()
	}

	if e.blockNo == 1 {
		e.bw.writeBits(8, 'B')
		e.bw.writeBits(8, 'Z')
		e.bw.writeBits(8, 'h')
		e.bw.writeBits(8, uint32('0'+e.blockSize100k))
	}

	if e.nblock > 0 {
		e.bw.writeBits(24, 0x314159)
		e.bw.writeBits(24, 0x265359)
		e.bw.writeBits(32, e.blockCRC)
		e.bw.writeBits(1, 0) // never randomised
		e.bw.writeBits(24, uint32(e.origPtr))
		e.generateMTFValues()
		e.sendMTFValues()
	}

	if isLastBlock {
		e.bw.writeBits(24, 0x177245)
		e.bw.writeBits(24, 0x385090)
		e.bw.writeBits(32, e.combinedCRC)
		e.bw.flush()
	}
}

// bitWriter is libbzip2's bsW/bsFinishWrite, buffering bits MSB-first.
type bitWriter struct {
	out  []byte
	buf  uint64 // pending bits, left-aligned
	live uint   // number of pending bits (< 32 between calls)
}

func (w *bitWriter) writeBits(n uint, v uint32) {
	w.buf |= uint64(v) << (64 - w.live - n)
	w.live += n
	if w.live >= 32 {
		w.out = append(w.out, byte(w.buf>>56), byte(w.buf>>48), byte(w.buf>>40), byte(w.buf>>32))
		w.buf <<= 32
		w.live -= 32
	}
}

func (w *bitWriter) flush() {
	for w.live > 0 {
		w.out = append(w.out, byte(w.buf>>56))
		w.buf <<= 8
		if w.live < 8 {
			w.live = 0
		} else {
			w.live -= 8
		}
	}
}

// crcTable is BZ2_crc32Table: the MSB-first CRC-32 table for polynomial
// 0x04c11db7.
var crcTable = func() (t [256]uint32) {
	for i := range t {
		c := uint32(i) << 24
		for k := 0; k < 8; k++ {
			if c&0x80000000 != 0 {
				c = c<<1 ^ 0x04c11db7
			} else {
				c <<= 1
			}
		}
		t[i] = c
	}
	return
}()

func (e *encoder) makeMaps() {
	e.nInUse = 0
	for i := 0; i < 256; i++ {
		if e.inUse[i] {
			e.unseqToSeq[i] = byte(e.nInUse)
			e.nInUse++
		}
	}
}

// generateMTFValues computes the MTF/RLE2 symbol stream from the sorted
// block.
func (e *encoder) generateMTFValues() {
	var yy [256]byte
	ptr := e.ptr
	block := e.block
	mtfv := e.mtfv
	nblock := e.nblock

	e.makeMaps()
	eob := e.nInUse + 1
	for i := int32(0); i <= eob; i++ {
		e.mtfFreq[i] = 0
	}

	wr := int32(0)
	zPend := int32(0)
	for i := int32(0); i < e.nInUse; i++ {
		yy[i] = byte(i)
	}

	flushZeros := func() {
		zPend--
		for {
			if zPend&1 != 0 {
				mtfv[wr] = runB
				e.mtfFreq[runB]++
			} else {
				mtfv[wr] = runA
				e.mtfFreq[runA]++
			}
			wr++
			if zPend < 2 {
				break
			}
			zPend = (zPend - 2) / 2
		}
		zPend = 0
	}

	for i := int32(0); i < nblock; i++ {
		j := int32(ptr[i]) - 1
		if j < 0 {
			j += nblock
		}
		llI := e.unseqToSeq[block[j]]

		if yy[0] == llI {
			zPend++
			continue
		}
		if zPend > 0 {
			flushZeros()
		}
		// Move llI to the front. Equivalent to libbzip2's shifting loop:
		// yy[0:k] moves up by one and llI (found at index k >= 1) goes to 0.
		k := bytes.IndexByte(yy[1:e.nInUse], llI) + 1
		copy(yy[1:k+1], yy[:k])
		yy[0] = llI
		mtfv[wr] = uint16(k + 1)
		wr++
		e.mtfFreq[k+1]++
	}

	if zPend > 0 {
		flushZeros()
	}

	mtfv[wr] = uint16(eob)
	wr++
	e.mtfFreq[eob]++

	e.nMTF = wr
}

// sendMTFValues chooses the Huffman tables and emits the block body.
func (e *encoder) sendMTFValues() {
	var cost [nGroupsMax]uint16
	var fave [nGroupsMax]int32
	mtfv := e.mtfv[:e.nMTF]
	nMTF := e.nMTF

	alphaSize := int(e.nInUse + 2)
	for t := 0; t < nGroupsMax; t++ {
		for v := 0; v < alphaSize; v++ {
			e.len[t][v] = greaterICost
		}
	}

	// Decide how many coding tables to use.
	var nGroups int
	switch {
	case nMTF < 200:
		nGroups = 2
	case nMTF < 600:
		nGroups = 3
	case nMTF < 1200:
		nGroups = 4
	case nMTF < 2400:
		nGroups = 5
	default:
		nGroups = 6
	}

	// Generate an initial set of coding tables.
	{
		nPart := nGroups
		remF := nMTF
		gs := 0
		for nPart > 0 {
			tFreq := remF / int32(nPart)
			ge := gs - 1
			aFreq := int32(0)
			for aFreq < tFreq && ge < alphaSize-1 {
				ge++
				aFreq += e.mtfFreq[ge]
			}
			if ge > gs && nPart != nGroups && nPart != 1 && (nGroups-nPart)%2 == 1 {
				aFreq -= e.mtfFreq[ge]
				ge--
			}
			for v := 0; v < alphaSize; v++ {
				if v >= gs && v <= ge {
					e.len[nPart-1][v] = lesserICost
				} else {
					e.len[nPart-1][v] = greaterICost
				}
			}
			nPart--
			gs = ge + 1
			remF -= aFreq
		}
	}

	// Iterate BZ_N_ITERS times to improve the tables.
	//
	// Group costs are at most 50*17 = 850, so the per-table costs of a
	// symbol are packed into 10-bit lanes of one uint64 (the C code does the
	// same with 16-bit lanes for the 6-table case).
	var lenPack [maxAlphaSize]uint64
	nSelectors := 0
	for iter := 0; iter < nIters; iter++ {
		for t := 0; t < nGroups; t++ {
			fave[t] = 0
			for v := 0; v < alphaSize; v++ {
				e.rfreq[t][v] = 0
			}
		}
		for v := 0; v < alphaSize; v++ {
			var p uint64
			for t := 0; t < nGroups; t++ {
				p |= uint64(e.len[t][v]) << (10 * t)
			}
			lenPack[v] = p
		}

		nSelectors = 0
		for gs := 0; gs < int(nMTF); gs += groupSize {
			ge := gs + groupSize - 1
			if ge >= int(nMTF) {
				ge = int(nMTF) - 1
			}
			// Calculate the cost of this group as coded by each table.
			var sum uint64
			for _, icv := range mtfv[gs : ge+1] {
				sum += lenPack[icv]
			}
			for t := 0; t < nGroups; t++ {
				cost[t] = uint16(sum>>(10*t)) & 0x3ff
			}
			// Find the coding table which is best for this group.
			bc := int32(999999999)
			bt := -1
			for t := 0; t < nGroups; t++ {
				if int32(cost[t]) < bc {
					bc = int32(cost[t])
					bt = t
				}
			}
			fave[bt]++
			e.selector[nSelectors] = byte(bt)
			nSelectors++
			// Increment the symbol frequencies for the selected table.
			rf := &e.rfreq[bt]
			for _, icv := range mtfv[gs : ge+1] {
				rf[icv]++
			}
		}

		// Recompute the tables based on the accumulated frequencies.
		for t := 0; t < nGroups; t++ {
			hbMakeCodeLengths(e.len[t][:], e.rfreq[t][:], alphaSize, 17)
		}
	}

	if nGroups >= 8 || nSelectors >= 32768 || nSelectors > maxSelectors {
		panic("bzip2: internal error 3003")
	}

	// Compute MTF values for the selectors.
	{
		var pos [nGroupsMax]byte
		for i := 0; i < nGroups; i++ {
			pos[i] = byte(i)
		}
		for i := 0; i < nSelectors; i++ {
			llI := e.selector[i]
			j := 0
			tmp := pos[j]
			for llI != tmp {
				j++
				tmp, pos[j] = pos[j], tmp
			}
			pos[0] = tmp
			e.selectorMtf[i] = byte(j)
		}
	}

	// Assign actual codes for the tables.
	for t := 0; t < nGroups; t++ {
		minLen := 32
		maxLen := 0
		for i := 0; i < alphaSize; i++ {
			l := int(e.len[t][i])
			if l > maxLen {
				maxLen = l
			}
			if l < minLen {
				minLen = l
			}
		}
		if maxLen > 17 || minLen < 1 {
			panic("bzip2: internal error 3004")
		}
		hbAssignCodes(e.code[t][:], e.len[t][:], minLen, maxLen, alphaSize)
	}

	bw := &e.bw

	// Transmit the mapping table.
	{
		var inUse16 [16]bool
		for i := 0; i < 16; i++ {
			for j := 0; j < 16; j++ {
				if e.inUse[i*16+j] {
					inUse16[i] = true
				}
			}
		}
		for i := 0; i < 16; i++ {
			if inUse16[i] {
				bw.writeBits(1, 1)
			} else {
				bw.writeBits(1, 0)
			}
		}
		for i := 0; i < 16; i++ {
			if !inUse16[i] {
				continue
			}
			for j := 0; j < 16; j++ {
				if e.inUse[i*16+j] {
					bw.writeBits(1, 1)
				} else {
					bw.writeBits(1, 0)
				}
			}
		}
	}

	// Now the selectors.
	bw.writeBits(3, uint32(nGroups))
	bw.writeBits(15, uint32(nSelectors))
	for i := 0; i < nSelectors; i++ {
		for j := 0; j < int(e.selectorMtf[i]); j++ {
			bw.writeBits(1, 1)
		}
		bw.writeBits(1, 0)
	}

	// Now the coding tables.
	for t := 0; t < nGroups; t++ {
		curr := int(e.len[t][0])
		bw.writeBits(5, uint32(curr))
		for i := 0; i < alphaSize; i++ {
			l := int(e.len[t][i])
			for curr < l {
				bw.writeBits(2, 2)
				curr++
			}
			for curr > l {
				bw.writeBits(2, 3)
				curr--
			}
			bw.writeBits(1, 0)
		}
	}

	// And finally, the block data proper.
	selCtr := 0
	for gs := 0; gs < int(nMTF); gs += groupSize {
		ge := gs + groupSize - 1
		if ge >= int(nMTF) {
			ge = int(nMTF) - 1
		}
		sel := e.selector[selCtr]
		lens := &e.len[sel]
		codes := &e.code[sel]
		for _, v := range mtfv[gs : ge+1] {
			bw.writeBits(uint(lens[v]), uint32(codes[v]))
		}
		selCtr++
	}
	if selCtr != nSelectors {
		panic("bzip2: internal error 3007")
	}
}
