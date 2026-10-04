package bzip2

// hbMakeCodeLengths is BZ2_hbMakeCodeLengths: it builds Huffman code lengths
// for freq[0:alphaSize] into length, limited to maxLen bits by repeatedly
// flattening the frequencies. Ties are broken exactly as in libbzip2 (heap
// order plus the tree-depth tiebreak packed into the low byte of each
// weight).
func hbMakeCodeLengths(length []byte, freq []int32, alphaSize, maxLen int) {
	// Nodes and heap entries run from 1. Entry 0 for both the heap and nodes
	// is a sentinel.
	var heap [maxAlphaSize + 2]int32
	var weight [maxAlphaSize * 2]int32
	var parent [maxAlphaSize * 2]int32

	for i := 0; i < alphaSize; i++ {
		f := freq[i]
		if f == 0 {
			f = 1
		}
		weight[i+1] = f << 8
	}

	for {
		nNodes := int32(alphaSize)
		nHeap := 0

		heap[0] = 0
		weight[0] = 0
		parent[0] = -2

		upHeap := func(z int) {
			zz := z
			tmp := heap[zz]
			for weight[tmp] < weight[heap[zz>>1]] {
				heap[zz] = heap[zz>>1]
				zz >>= 1
			}
			heap[zz] = tmp
		}
		downHeap := func(z int) {
			zz := z
			tmp := heap[zz]
			for {
				yy := zz << 1
				if yy > nHeap {
					break
				}
				if yy < nHeap && weight[heap[yy+1]] < weight[heap[yy]] {
					yy++
				}
				if weight[tmp] < weight[heap[yy]] {
					break
				}
				heap[zz] = heap[yy]
				zz = yy
			}
			heap[zz] = tmp
		}

		for i := 1; i <= alphaSize; i++ {
			parent[i] = -1
			nHeap++
			heap[nHeap] = int32(i)
			upHeap(nHeap)
		}

		for nHeap > 1 {
			n1 := heap[1]
			heap[1] = heap[nHeap]
			nHeap--
			downHeap(1)
			n2 := heap[1]
			heap[1] = heap[nHeap]
			nHeap--
			downHeap(1)
			nNodes++
			parent[n1] = nNodes
			parent[n2] = nNodes
			w1, w2 := weight[n1], weight[n2]
			d1, d2 := w1&0xff, w2&0xff
			weight[nNodes] = (w1&^0xff + w2&^0xff) | (1 + max(d1, d2))
			parent[nNodes] = -1
			nHeap++
			heap[nHeap] = nNodes
			upHeap(nHeap)
		}

		tooLong := false
		for i := 1; i <= alphaSize; i++ {
			j := 0
			k := int32(i)
			for parent[k] >= 0 {
				k = parent[k]
				j++
			}
			length[i-1] = byte(j)
			if j > maxLen {
				tooLong = true
			}
		}

		if !tooLong {
			break
		}

		for i := 1; i <= alphaSize; i++ {
			j := weight[i] >> 8
			j = 1 + (j / 2)
			weight[i] = j << 8
		}
	}
}

// hbAssignCodes is BZ2_hbAssignCodes: canonical code assignment in order of
// increasing length, then symbol index.
func hbAssignCodes(code []int32, length []byte, minLen, maxLen, alphaSize int) {
	vec := int32(0)
	for n := minLen; n <= maxLen; n++ {
		for i := 0; i < alphaSize; i++ {
			if int(length[i]) == n {
				code[i] = vec
				vec++
			}
		}
		vec <<= 1
	}
}
