package bzip2

import (
	"encoding/binary"
	"math/bits"
)

// Port of libbzip2 1.0.8 blocksort.c.
//
// The sorted order of distinct rotations is unique, but for periodic blocks
// (identical rotations) the position of rotation 0 among its equals -- and
// hence origPtr -- depends on the algorithm. Despite the remark in
// BZ2_blockSort, mainSort and fallbackSort do order such ties differently
// (mainSort can finish on periodic blocks by resolving ties through the
// quadrant cache), so to be byte-identical with libbzip2 both sorts and the
// work budget that chooses between them are ported step for step. The golden
// vectors named period-* fail if either is changed.

// blockSort is BZ2_blockSort. On return e.ptr[0:nblock] holds the sorted
// rotation order and e.origPtr the position of rotation 0.
func (e *encoder) blockSort() {
	nblock := e.nblock
	if nblock < 10000 {
		e.fallbackSort()
	} else {
		if len(e.quadrant) < int(nblock)+nOvershoot {
			e.quadrant = make([]uint16, len(e.block))
		}
		if len(e.ftab) < 65537 {
			e.ftab = make([]uint32, 65537)
		}
		wfact := int32(workFactor)
		budgetInit := nblock * ((wfact - 1) / 3)
		e.budget = budgetInit
		e.mainSort()
		if e.budget < 0 {
			e.fallbackSort()
		}
	}

	e.origPtr = -1
	for i := int32(0); i < nblock; i++ {
		if e.ptr[i] == 0 {
			e.origPtr = i
			break
		}
	}
	if e.origPtr == -1 {
		panic("bzip2: internal error 1003")
	}
}

/*---------------------------------------------*/
/*--- Fallback O(N log(N)^2) sorting        ---*/
/*--- algorithm, for repetitive blocks      ---*/
/*---------------------------------------------*/

func fallbackSimpleSort(fmap, eclass []uint32, lo, hi int32) {
	if lo == hi {
		return
	}
	if hi-lo > 3 {
		for i := hi - 4; i >= lo; i-- {
			tmp := fmap[i]
			ecTmp := eclass[tmp]
			j := i + 4
			for ; j <= hi && ecTmp > eclass[fmap[j]]; j += 4 {
				fmap[j-4] = fmap[j]
			}
			fmap[j-4] = tmp
		}
	}
	for i := hi - 1; i >= lo; i-- {
		tmp := fmap[i]
		ecTmp := eclass[tmp]
		j := i + 1
		for ; j <= hi && ecTmp > eclass[fmap[j]]; j++ {
			fmap[j-1] = fmap[j]
		}
		fmap[j-1] = tmp
	}
}

const (
	fallbackQSortSmallThresh = 10
	fallbackQSortStackSize   = 100
)

func fallbackQSort3(fmap, eclass []uint32, loSt, hiSt int32) {
	var stackLo, stackHi [fallbackQSortStackSize]int32
	r := uint32(0)

	sp := 0
	stackLo[sp], stackHi[sp] = loSt, hiSt
	sp++

	for sp > 0 {
		if sp >= fallbackQSortStackSize-1 {
			panic("bzip2: internal error 1004")
		}
		sp--
		lo, hi := stackLo[sp], stackHi[sp]
		if hi-lo < fallbackQSortSmallThresh {
			fallbackSimpleSort(fmap, eclass, lo, hi)
			continue
		}

		// Random partitioning, as in libbzip2 (Sedgewick's constants).
		r = ((r * 7621) + 1) % 32768
		var med uint32
		switch r % 3 {
		case 0:
			med = eclass[fmap[lo]]
		case 1:
			med = eclass[fmap[(lo+hi)>>1]]
		default:
			med = eclass[fmap[hi]]
		}

		unLo, ltLo := lo, lo
		unHi, gtHi := hi, hi

		for {
			for unLo <= unHi {
				n := int32(eclass[fmap[unLo]]) - int32(med)
				if n == 0 {
					fmap[unLo], fmap[ltLo] = fmap[ltLo], fmap[unLo]
					ltLo++
					unLo++
					continue
				}
				if n > 0 {
					break
				}
				unLo++
			}
			for unLo <= unHi {
				n := int32(eclass[fmap[unHi]]) - int32(med)
				if n == 0 {
					fmap[unHi], fmap[gtHi] = fmap[gtHi], fmap[unHi]
					gtHi--
					unHi--
					continue
				}
				if n < 0 {
					break
				}
				unHi--
			}
			if unLo > unHi {
				break
			}
			fmap[unLo], fmap[unHi] = fmap[unHi], fmap[unLo]
			unLo++
			unHi--
		}

		if gtHi < ltLo {
			continue
		}

		n := min(ltLo-lo, unLo-ltLo)
		vswap(fmap, lo, unLo-n, n)
		m := min(hi-gtHi, gtHi-unHi)
		vswap(fmap, unLo, hi-m+1, m)

		n = lo + unLo - ltLo - 1
		m = hi - (gtHi - unHi) + 1

		if n-lo > hi-m {
			stackLo[sp], stackHi[sp] = lo, n
			sp++
			stackLo[sp], stackHi[sp] = m, hi
			sp++
		} else {
			stackLo[sp], stackHi[sp] = m, hi
			sp++
			stackLo[sp], stackHi[sp] = lo, n
			sp++
		}
	}
}

func vswap(p []uint32, p1, p2, n int32) {
	for n > 0 {
		p[p1], p[p2] = p[p2], p[p1]
		p1++
		p2++
		n--
	}
}

// fallbackSort sorts the rotations of e.block[0:nblock] into e.ptr using
// libbzip2's exponential radix sort (Manber-Myers style prefix doubling).
//
// In C, eclass aliases the block storage and the block is reconstructed at
// the end; here eclass is a separate array, so the block is left intact.
func (e *encoder) fallbackSort() {
	nblock := e.nblock
	if len(e.eclass) < int(nblock) {
		e.eclass = make([]uint32, len(e.ptr))
	}
	nBhtab := 2 + nblock/32
	if len(e.ftab) < int(nBhtab)+1 {
		e.ftab = make([]uint32, nBhtab+1)
	}
	fmap := e.ptr[:nblock]
	eclass := e.eclass[:nblock]
	bhtab := e.ftab
	block := e.block[:nblock]

	var ftab [257]int32

	// Initial 1-char radix sort to generate initial fmap and initial BH bits.
	for _, c := range block {
		ftab[c]++
	}
	for i := 1; i < 257; i++ {
		ftab[i] += ftab[i-1]
	}
	for i, c := range block {
		k := ftab[c] - 1
		ftab[c] = k
		fmap[k] = uint32(i)
	}

	// The sentinel bits may land one word past nBhtab; C leaves the rest of
	// that word unspecified but never reads it.
	clear(bhtab[:nBhtab+1])
	setBH := func(zz int32) { bhtab[zz>>5] |= 1 << (uint32(zz) & 31) }
	clearBH := func(zz int32) { bhtab[zz>>5] &^= 1 << (uint32(zz) & 31) }
	for i := 0; i < 256; i++ {
		setBH(ftab[i])
	}

	// Set sentinel bits for block-end detection.
	for i := int32(0); i < 32; i++ {
		setBH(nblock + 2*i)
		clearBH(nblock + 2*i + 1)
	}

	// The log(N) loop.
	H := int32(1)
	for {
		j := int32(0)
		for i := int32(0); i < nblock; i++ {
			if bhtab[i>>5]&(1<<(uint32(i)&31)) != 0 {
				j = i
			}
			k := int32(fmap[i]) - H
			if k < 0 {
				k += nblock
			}
			eclass[k] = uint32(j)
		}

		nNotDone := int32(0)
		r := int32(-1)
		for {
			// Find the next non-singleton bucket.
			k := r + 1
			for isSetBH(bhtab, k) && k&31 != 0 {
				k++
			}
			if isSetBH(bhtab, k) {
				for bhtab[k>>5] == 0xffffffff {
					k += 32
				}
				for isSetBH(bhtab, k) {
					k++
				}
			}
			l := k - 1
			if l >= nblock {
				break
			}
			for !isSetBH(bhtab, k) && k&31 != 0 {
				k++
			}
			if !isSetBH(bhtab, k) {
				for bhtab[k>>5] == 0 {
					k += 32
				}
				for !isSetBH(bhtab, k) {
					k++
				}
			}
			r = k - 1
			if r >= nblock {
				break
			}

			// Now [l, r] bracket current bucket.
			if r > l {
				nNotDone += r - l + 1
				fallbackQSort3(fmap, eclass, l, r)

				// Scan bucket and generate header bits.
				cc := int32(-1)
				for i := l; i <= r; i++ {
					cc1 := int32(eclass[fmap[i]])
					if cc != cc1 {
						setBH(i)
						cc = cc1
					}
				}
			}
		}

		H *= 2
		if H > nblock || nNotDone == 0 {
			break
		}
	}
}

func isSetBH(bhtab []uint32, zz int32) bool {
	return bhtab[zz>>5]&(1<<(uint32(zz)&31)) != 0
}

/*---------------------------------------------*/
/*--- The main, O(N^2 log(N)) sorting       ---*/
/*--- algorithm.  Faster for "normal"       ---*/
/*--- non-repetitive blocks.                ---*/
/*---------------------------------------------*/

// mainGtU reports whether the rotation starting at i1 sorts after the one
// starting at i2, charging the work budget for long comparisons exactly as
// libbzip2 does.
func mainGtU(i1, i2 uint32, block []byte, quadrant []uint16, nblock uint32, budget *int32) bool {
	// The first 12 bytes are compared without consulting the quadrants.
	// Comparing big-endian words is the same as comparing bytes in order.
	if a, b := binary.BigEndian.Uint64(block[i1:]), binary.BigEndian.Uint64(block[i2:]); a != b {
		return a > b
	}
	if a, b := binary.BigEndian.Uint32(block[i1+8:]), binary.BigEndian.Uint32(block[i2+8:]); a != b {
		return a > b
	}
	i1 += 12
	i2 += 12

	k := int32(nblock) + 8
	for {
		// Eight positions; at each, the block byte is compared before the
		// quadrant value. p is the first position whose block bytes differ
		// (8 if none), so quadrants before p decide first.
		a, b := binary.BigEndian.Uint64(block[i1:]), binary.BigEndian.Uint64(block[i2:])
		p := uint32(8)
		if a != b {
			p = uint32(bits.LeadingZeros64(a^b)) / 8
		}
		q1 := quadrant[i1 : i1+8]
		q2 := quadrant[i2 : i2+8]
		for n := uint32(0); n < p; n++ {
			if s1, s2 := q1[n], q2[n]; s1 != s2 {
				return s1 > s2
			}
		}
		if a != b {
			return a > b
		}
		i1 += 8
		i2 += 8
		if i1 >= nblock {
			i1 -= nblock
		}
		if i2 >= nblock {
			i2 -= nblock
		}
		k -= 8
		*budget--
		if k < 0 {
			break
		}
	}
	return false
}

// Knuth's increments seem to work better than Incerpi-Sedgewick here.
var incs = [14]int32{1, 4, 13, 40, 121, 364, 1093, 3280,
	9841, 29524, 88573, 265720, 797161, 2391484}

func (e *encoder) mainSimpleSort(lo, hi, d int32) {
	ptr := e.ptr
	block := e.block
	quadrant := e.quadrant
	nblock := uint32(e.nblock)
	ud := uint32(d)

	bigN := hi - lo + 1
	if bigN < 2 {
		return
	}

	hp := 0
	for incs[hp] < bigN {
		hp++
	}
	hp--

	for ; hp >= 0; hp-- {
		h := incs[hp]
		i := lo + h
	insertions:
		for {
			// Three insertions per budget check, as in the C code.
			for c := 0; c < 3; c++ {
				if i > hi {
					break insertions
				}
				v := ptr[i]
				j := i
				for mainGtU(ptr[j-h]+ud, v+ud, block, quadrant, nblock, &e.budget) {
					ptr[j] = ptr[j-h]
					j = j - h
					if j <= lo+h-1 {
						break
					}
				}
				ptr[j] = v
				i++
			}
			if e.budget < 0 {
				return
			}
		}
	}
}

func mmed3(a, b, c byte) byte {
	if a > b {
		a, b = b, a
	}
	if b > c {
		b = c
		if a > b {
			b = a
		}
	}
	return b
}

const (
	mainQSortSmallThresh = 20
	mainQSortDepthThresh = nRadix + nQSort
	mainQSortStackSize   = 100
)

// mainQSort3 is libbzip2's multikey (Bentley-Sedgewick) 3-way quicksort.
func (e *encoder) mainQSort3(loSt, hiSt, dSt int32) {
	ptr := e.ptr
	block := e.block

	var stackLo, stackHi, stackD [mainQSortStackSize]int32
	var nextLo, nextHi, nextD [3]int32

	sp := 0
	push := func(lz, hz, dz int32) {
		stackLo[sp], stackHi[sp], stackD[sp] = lz, hz, dz
		sp++
	}
	push(loSt, hiSt, dSt)

	for sp > 0 {
		if sp >= mainQSortStackSize-2 {
			panic("bzip2: internal error 1001")
		}
		sp--
		lo, hi, d := stackLo[sp], stackHi[sp], stackD[sp]
		if hi-lo < mainQSortSmallThresh || d > mainQSortDepthThresh {
			e.mainSimpleSort(lo, hi, d)
			if e.budget < 0 {
				return
			}
			continue
		}
		ud := uint32(d)

		med := int32(mmed3(block[ptr[lo]+ud], block[ptr[hi]+ud], block[ptr[(lo+hi)>>1]+ud]))

		unLo, ltLo := lo, lo
		unHi, gtHi := hi, hi

		for {
			for unLo <= unHi {
				n := int32(block[ptr[unLo]+ud]) - med
				if n == 0 {
					ptr[unLo], ptr[ltLo] = ptr[ltLo], ptr[unLo]
					ltLo++
					unLo++
					continue
				}
				if n > 0 {
					break
				}
				unLo++
			}
			for unLo <= unHi {
				n := int32(block[ptr[unHi]+ud]) - med
				if n == 0 {
					ptr[unHi], ptr[gtHi] = ptr[gtHi], ptr[unHi]
					gtHi--
					unHi--
					continue
				}
				if n < 0 {
					break
				}
				unHi--
			}
			if unLo > unHi {
				break
			}
			ptr[unLo], ptr[unHi] = ptr[unHi], ptr[unLo]
			unLo++
			unHi--
		}

		if gtHi < ltLo {
			push(lo, hi, d+1)
			continue
		}

		n := min(ltLo-lo, unLo-ltLo)
		vswap(ptr, lo, unLo-n, n)
		m := min(hi-gtHi, gtHi-unHi)
		vswap(ptr, unLo, hi-m+1, m)

		n = lo + unLo - ltLo - 1
		m = hi - (gtHi - unHi) + 1

		nextLo[0], nextHi[0], nextD[0] = lo, n, d
		nextLo[1], nextHi[1], nextD[1] = m, hi, d
		nextLo[2], nextHi[2], nextD[2] = n+1, m-1, d+1

		swapNext := func(a, b int) {
			nextLo[a], nextLo[b] = nextLo[b], nextLo[a]
			nextHi[a], nextHi[b] = nextHi[b], nextHi[a]
			nextD[a], nextD[b] = nextD[b], nextD[a]
		}
		size := func(a int) int32 { return nextHi[a] - nextLo[a] }
		if size(0) < size(1) {
			swapNext(0, 1)
		}
		if size(1) < size(2) {
			swapNext(1, 2)
		}
		if size(0) < size(1) {
			swapNext(0, 1)
		}

		push(nextLo[0], nextHi[0], nextD[0])
		push(nextLo[1], nextHi[1], nextD[1])
		push(nextLo[2], nextHi[2], nextD[2])
	}
}

const (
	setMask   = 1 << 21
	clearMask = ^uint32(setMask)
)

// mainSort is libbzip2's main sorting algorithm: a 2-byte radix sort
// followed by sorting the small buckets with mainQSort3 and deriving the
// rest by Seward's copying trick. If the work budget runs out (e.budget < 0
// on return) the result is discarded in favour of fallbackSort.
func (e *encoder) mainSort() {
	ptr := e.ptr
	block := e.block
	quadrant := e.quadrant
	ftab := e.ftab[:65537]
	nblock := e.nblock

	var runningOrder [256]int32
	var bigDone [256]bool
	var copyStart, copyEnd [256]int32

	// Set up the 2-byte frequency table.
	clear(ftab)
	j := int32(block[0]) << 8
	for i := nblock - 1; i >= 0; i-- {
		quadrant[i] = 0
		j = (j >> 8) | int32(block[i])<<8
		ftab[j]++
	}

	for i := int32(0); i < nOvershoot; i++ {
		block[nblock+i] = block[i]
		quadrant[nblock+i] = 0
	}

	// Complete the initial radix sort.
	for i := 1; i <= 65536; i++ {
		ftab[i] += ftab[i-1]
	}

	s := uint16(block[0]) << 8
	for i := nblock - 1; i >= 0; i-- {
		s = (s >> 8) | uint16(block[i])<<8
		j := ftab[s] - 1
		ftab[s] = j
		ptr[j] = uint32(i)
	}

	// Now ftab contains the first loc of every small bucket. Calculate the
	// running order, from smallest to largest big bucket.
	for i := 0; i <= 255; i++ {
		bigDone[i] = false
		runningOrder[i] = int32(i)
	}

	bigFreq := func(b int32) uint32 { return ftab[(b+1)<<8] - ftab[b<<8] }
	{
		h := int32(1)
		for {
			h = 3*h + 1
			if h > 256 {
				break
			}
		}
		for {
			h = h / 3
			for i := h; i <= 255; i++ {
				vv := runningOrder[i]
				j := i
				for bigFreq(runningOrder[j-h]) > bigFreq(vv) {
					runningOrder[j] = runningOrder[j-h]
					j = j - h
					if j <= h-1 {
						break
					}
				}
				runningOrder[j] = vv
			}
			if h == 1 {
				break
			}
		}
	}

	// The main sorting loop.
	for i := 0; i <= 255; i++ {
		// Process big buckets, starting with the least full.
		ss := runningOrder[i]

		// Step 1: complete the big bucket [ss] by quicksorting any unsorted
		// small buckets [ss, j], for j != ss.
		for j := int32(0); j <= 255; j++ {
			if j != ss {
				sb := (ss << 8) + j
				if ftab[sb]&setMask == 0 {
					lo := int32(ftab[sb] & clearMask)
					hi := int32(ftab[sb+1]&clearMask) - 1
					if hi > lo {
						e.mainQSort3(lo, hi, nRadix)
						if e.budget < 0 {
							return
						}
					}
				}
				ftab[sb] |= setMask
			}
		}

		if bigDone[ss] {
			panic("bzip2: internal error 1006")
		}

		// Step 2: scan this big bucket [ss] so as to synthesise the sorted
		// order for small buckets [t, ss] for all t, including [ss, ss].
		for j := int32(0); j <= 255; j++ {
			copyStart[j] = int32(ftab[(j<<8)+ss] & clearMask)
			copyEnd[j] = int32(ftab[(j<<8)+ss+1]&clearMask) - 1
		}
		for j := int32(ftab[ss<<8] & clearMask); j < copyStart[ss]; j++ {
			k := int32(ptr[j]) - 1
			if k < 0 {
				k += nblock
			}
			c1 := block[k]
			if !bigDone[c1] {
				ptr[copyStart[c1]] = uint32(k)
				copyStart[c1]++
			}
		}
		for j := int32(ftab[(ss+1)<<8]&clearMask) - 1; j > copyEnd[ss]; j-- {
			k := int32(ptr[j]) - 1
			if k < 0 {
				k += nblock
			}
			c1 := block[k]
			if !bigDone[c1] {
				ptr[copyEnd[c1]] = uint32(k)
				copyEnd[c1]--
			}
		}

		if !((copyStart[ss]-1 == copyEnd[ss]) || (copyStart[ss] == 0 && copyEnd[ss] == nblock-1)) {
			panic("bzip2: internal error 1007")
		}

		for j := int32(0); j <= 255; j++ {
			ftab[(j<<8)+ss] |= setMask
		}

		// Step 3: the [ss] big bucket is now done. Record this fact, and
		// update the quadrant descriptors (including the overshoot area).
		bigDone[ss] = true

		if i < 255 {
			bbStart := int32(ftab[ss<<8] & clearMask)
			bbSize := int32(ftab[(ss+1)<<8]&clearMask) - bbStart
			shifts := uint(0)

			for (bbSize >> shifts) > 65534 {
				shifts++
			}

			for j := bbSize - 1; j >= 0; j-- {
				a2update := ptr[bbStart+j]
				qVal := uint16(j >> shifts)
				quadrant[a2update] = qVal
				if a2update < nOvershoot {
					quadrant[a2update+uint32(nblock)] = qVal
				}
			}
		}
	}
}
