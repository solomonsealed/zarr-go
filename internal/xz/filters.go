package xz

import "encoding/binary"

// The delta and BCJ ("branch/call/jump") filters, ported from liblzma's
// src/liblzma/delta and src/liblzma/simple. Each BCJ function converts as
// much of buf as it can, starting at stream position pos, and returns the
// number of bytes processed; the unprocessed tail (a partial instruction at
// the very end) is passed through unchanged, as liblzma does at the end of a
// stream. Processing a whole stream in one call is equivalent to liblzma's
// incremental processing because each filter is a left-to-right scan whose
// only cross-call state is carried explicitly (x86) or absent.

// applyEncodeFilters runs the non-last filters of a chain in order on a copy
// of data (data itself is never modified).
func applyEncodeFilters(data []byte, chain []chainFilter) []byte {
	if len(chain) == 0 {
		return data
	}
	buf := make([]byte, len(data))
	copy(buf, data)
	for _, f := range chain {
		applyFilter(f, buf, true)
	}
	return buf
}

// applyDecodeFilters undoes the non-last filters of a chain, in place.
func applyDecodeFilters(buf []byte, chain []chainFilter) {
	for i := len(chain) - 1; i >= 0; i-- {
		applyFilter(chain[i], buf, false)
	}
}

func applyFilter(f chainFilter, buf []byte, encode bool) {
	switch f.id {
	case FilterDelta:
		if encode {
			deltaEncode(buf, f.dist)
		} else {
			deltaDecode(buf, f.dist)
		}
	case FilterX86:
		var s x86State
		s.init()
		s.code(f.start, encode, buf)
	case FilterPowerPC:
		powerpcCode(f.start, encode, buf)
	case FilterIA64:
		ia64Code(f.start, encode, buf)
	case FilterARM:
		armCode(f.start, encode, buf)
	case FilterARMThumb:
		armthumbCode(f.start, encode, buf)
	case FilterSPARC:
		sparcCode(f.start, encode, buf)
	case FilterARM64:
		arm64Code(f.start, encode, buf)
	}
}

// deltaEncode: out[i] = in[i] - in[i-dist] (bytes before the start are 0).
func deltaEncode(buf []byte, dist int) {
	for i := len(buf) - 1; i >= dist; i-- {
		buf[i] -= buf[i-dist]
	}
}

// deltaDecode: out[i] = in[i] + out[i-dist].
func deltaDecode(buf []byte, dist int) {
	for i := dist; i < len(buf); i++ {
		buf[i] += buf[i-dist]
	}
}

// x86State is the cross-call state of the x86 filter.
type x86State struct {
	prevMask uint32
	prevPos  uint32
}

func (s *x86State) init() {
	s.prevMask = 0
	s.prevPos = ^uint32(4) // (uint32_t)(-5)
}

func test86MSByte(b byte) bool { return b == 0x00 || b == 0xFF }

func (s *x86State) code(nowPos uint32, encode bool, buf []byte) int {
	maskToAllowed := [8]bool{true, true, true, false, true, false, false, false}
	maskToBitNumber := [8]uint32{0, 1, 2, 2, 3, 3, 3, 3}

	prevMask := s.prevMask
	prevPos := s.prevPos
	if len(buf) < 5 {
		return 0
	}
	if nowPos-prevPos > 5 {
		prevPos = nowPos - 5
	}
	limit := len(buf) - 5
	i := 0
	for i <= limit {
		b := buf[i]
		if b != 0xE8 && b != 0xE9 {
			i++
			continue
		}
		offset := nowPos + uint32(i) - prevPos
		prevPos = nowPos + uint32(i)
		if offset > 5 {
			prevMask = 0
		} else {
			for k := uint32(0); k < offset; k++ {
				prevMask &= 0x77
				prevMask <<= 1
			}
		}
		b = buf[i+4]
		if test86MSByte(b) && maskToAllowed[(prevMask>>1)&0x7] && (prevMask>>1) < 0x10 {
			src := uint32(b)<<24 | uint32(buf[i+3])<<16 | uint32(buf[i+2])<<8 | uint32(buf[i+1])
			var dest uint32
			for {
				if encode {
					dest = src + (nowPos + uint32(i) + 5)
				} else {
					dest = src - (nowPos + uint32(i) + 5)
				}
				if prevMask == 0 {
					break
				}
				idx := maskToBitNumber[prevMask>>1]
				b = byte(dest >> (24 - idx*8))
				if !test86MSByte(b) {
					break
				}
				src = dest ^ (1<<(32-idx*8) - 1)
			}
			buf[i+4] = byte(^(((dest >> 24) & 1) - 1))
			buf[i+3] = byte(dest >> 16)
			buf[i+2] = byte(dest >> 8)
			buf[i+1] = byte(dest)
			i += 5
			prevMask = 0
		} else {
			i++
			prevMask |= 1
			if test86MSByte(b) {
				prevMask |= 0x10
			}
		}
	}
	s.prevMask = prevMask
	s.prevPos = prevPos
	return i
}

func powerpcCode(nowPos uint32, encode bool, buf []byte) int {
	size := len(buf) &^ 3
	i := 0
	for ; i < size; i += 4 {
		// PowerPC branch: 6 bits opcode 18, 24 bits offset, AA=0, LK=1.
		if buf[i]>>2 != 0x12 || buf[i+3]&3 != 1 {
			continue
		}
		src := (uint32(buf[i])&3)<<24 | uint32(buf[i+1])<<16 | uint32(buf[i+2])<<8 | uint32(buf[i+3])&^3
		var dest uint32
		if encode {
			dest = nowPos + uint32(i) + src
		} else {
			dest = src - (nowPos + uint32(i))
		}
		buf[i] = 0x48 | byte((dest>>24)&0x03)
		buf[i+1] = byte(dest >> 16)
		buf[i+2] = byte(dest >> 8)
		buf[i+3] = buf[i+3]&0x03 | byte(dest)
	}
	return i
}

func armCode(nowPos uint32, encode bool, buf []byte) int {
	size := len(buf) &^ 3
	i := 0
	for ; i < size; i += 4 {
		if buf[i+3] != 0xEB { // BL with condition "always"
			continue
		}
		src := (uint32(buf[i+2])<<16 | uint32(buf[i+1])<<8 | uint32(buf[i])) << 2
		var dest uint32
		if encode {
			dest = nowPos + uint32(i) + 8 + src
		} else {
			dest = src - (nowPos + uint32(i) + 8)
		}
		dest >>= 2
		buf[i+2] = byte(dest >> 16)
		buf[i+1] = byte(dest >> 8)
		buf[i] = byte(dest)
	}
	return i
}

func armthumbCode(nowPos uint32, encode bool, buf []byte) int {
	if len(buf) < 4 {
		return 0
	}
	size := len(buf) - 4
	i := 0
	for ; i <= size; i += 2 {
		if buf[i+1]&0xF8 != 0xF0 || buf[i+3]&0xF8 != 0xF8 {
			continue
		}
		src := ((uint32(buf[i+1])&7)<<19 | uint32(buf[i])<<11 | (uint32(buf[i+3])&7)<<8 | uint32(buf[i+2])) << 1
		var dest uint32
		if encode {
			dest = nowPos + uint32(i) + 4 + src
		} else {
			dest = src - (nowPos + uint32(i) + 4)
		}
		dest >>= 1
		buf[i+1] = 0xF0 | byte((dest>>19)&0x7)
		buf[i] = byte(dest >> 11)
		buf[i+3] = 0xF8 | byte((dest>>8)&0x7)
		buf[i+2] = byte(dest)
		i += 2
	}
	return i
}

func sparcCode(nowPos uint32, encode bool, buf []byte) int {
	size := len(buf) &^ 3
	i := 0
	for ; i < size; i += 4 {
		if !(buf[i] == 0x40 && buf[i+1]&0xC0 == 0x00) && !(buf[i] == 0x7F && buf[i+1]&0xC0 == 0xC0) {
			continue
		}
		src := binary.BigEndian.Uint32(buf[i:]) << 2
		var dest uint32
		if encode {
			dest = nowPos + uint32(i) + src
		} else {
			dest = src - (nowPos + uint32(i))
		}
		dest >>= 2
		dest = ((0-((dest>>22)&1))<<22)&0x3FFFFFFF | dest&0x3FFFFF | 0x40000000
		binary.BigEndian.PutUint32(buf[i:], dest)
	}
	return i
}

func ia64Code(nowPos uint32, encode bool, buf []byte) int {
	branchTable := [32]uint32{
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0,
		4, 4, 6, 6, 0, 0, 7, 7,
		4, 4, 0, 0, 4, 4, 0, 0,
	}
	i := 0
	for ; i+16 <= len(buf); i += 16 {
		mask := branchTable[buf[i]&0x1F]
		bitPos := uint32(5)
		for slot := 0; slot < 3; slot, bitPos = slot+1, bitPos+41 {
			if (mask>>slot)&1 == 0 {
				continue
			}
			bytePos := int(bitPos >> 3)
			bitRes := bitPos & 7
			var instruction uint64
			for j := 0; j < 6; j++ {
				instruction |= uint64(buf[i+j+bytePos]) << (8 * j)
			}
			instNorm := instruction >> bitRes
			if (instNorm>>37)&0xF != 0x5 || (instNorm>>9)&0x7 != 0 {
				continue
			}
			src := uint32((instNorm >> 13) & 0xFFFFF)
			src |= uint32((instNorm>>36)&1) << 20
			src <<= 4
			var dest uint32
			if encode {
				dest = nowPos + uint32(i) + src
			} else {
				dest = src - (nowPos + uint32(i))
			}
			dest >>= 4
			instNorm &^= uint64(0x8FFFFF) << 13
			instNorm |= uint64(dest&0xFFFFF) << 13
			instNorm |= uint64(dest&0x100000) << (36 - 20)
			instruction &= (1 << bitRes) - 1
			instruction |= instNorm << bitRes
			for j := 0; j < 6; j++ {
				buf[i+j+bytePos] = byte(instruction >> (8 * j))
			}
		}
	}
	return i
}

func arm64Code(nowPos uint32, encode bool, buf []byte) int {
	i := 0
	for ; i+4 <= len(buf); i += 4 {
		pc := nowPos + uint32(i)
		instr := binary.LittleEndian.Uint32(buf[i:])
		if instr>>26 == 0x25 {
			// BL: the full 26-bit immediate is converted.
			src := instr
			instr = 0x94000000
			pc >>= 2
			if !encode {
				pc = 0 - pc
			}
			instr |= (src + pc) & 0x03FFFFFF
			binary.LittleEndian.PutUint32(buf[i:], instr)
		} else if instr&0x9F000000 == 0x90000000 {
			// ADRP: only values within +/-512 MiB are converted.
			src := (instr>>29)&3 | (instr>>3)&0x001FFFFC
			if (src+0x00020000)&0x001C0000 != 0 {
				continue
			}
			instr &= 0x9000001F
			pc >>= 12
			if !encode {
				pc = 0 - pc
			}
			dest := src + pc
			instr |= (dest & 3) << 29
			instr |= (dest & 0x0003FFFC) << 3
			instr |= (0 - (dest & 0x00020000)) & 0x00E00000
			binary.LittleEndian.PutUint32(buf[i:], instr)
		}
	}
	return i
}
