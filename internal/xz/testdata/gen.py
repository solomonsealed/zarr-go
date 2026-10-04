#!/usr/bin/env python3
"""Generate the golden test vectors for internal/xz.

    python gen.py

Writes inputs/*.bin, vectors/*.bin and manifest.json next to this script.
Every vector is produced by Python's lzma module (liblzma) or built by hand
below, and is checked against lzma.decompress before it is written, so the
expectations in manifest.json are liblzma's behaviour. The xz command-line
tool, if available, is used for the ARM64 BCJ vectors (Python's lzma module
cannot encode ARM64) and to confirm the stream-padding vector (on which
Python's lzma.decompress deliberately differs, see below).
"""

import hashlib
import json
import lzma
import math
import os
import random
import shutil
import struct
import subprocess
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
INPUTS = os.path.join(HERE, "inputs")
VECTORS = os.path.join(HERE, "vectors")

L1, L2 = lzma.FILTER_LZMA1, lzma.FILTER_LZMA2
DELTA = lzma.FILTER_DELTA
BCJ = {
    "x86": lzma.FILTER_X86,
    "powerpc": lzma.FILTER_POWERPC,
    "ia64": lzma.FILTER_IA64,
    "arm": lzma.FILTER_ARM,
    "armthumb": lzma.FILTER_ARMTHUMB,
    "sparc": lzma.FILTER_SPARC,
}
FILTER_ARM64 = 0x0A

# --------------------------------------------------------------------------
# Inputs


def words_text(seed, n):
    rng = random.Random(seed)
    words = ("the of and to in is was that for on with as by at from this be "
             "array chunk zarr compression filter stream block index dictionary "
             "literal match distance range coder float integer shape dtype "
             "metadata attribute group store codec lzma delta branch").split()
    out = []
    size = 0
    while size < n:
        line = " ".join(rng.choice(words) for _ in range(rng.randint(4, 14)))
        line = line.capitalize() + ".\n"
        out.append(line)
        size += len(line)
    return "".join(out).encode()[:n]


def float32_series(n):
    vals = [round(math.sin(i / 37.0) * 1000.0 + i * 0.25) / 4.0 for i in range(n)]
    return struct.pack("<%df" % n, *vals)


def int16_series(n):
    return struct.pack("<%dh" % n, *[int(3000 * math.sin(i / 50.0)) for i in range(n)])


def x86_code(seed, n):
    rng = random.Random(seed)
    ops = [b"\x55", b"\x48\x89\xe5", b"\x48\x83\xec\x10", b"\x89\x7d\xfc",
           b"\x8b\x45\xfc", b"\xc3", b"\x90", b"\x31\xc0", b"\x5d",
           b"\x48\x8b\x45\xf8", b"\x0f\xb6\xc0", b"\x74\x05", b"\x75\xf0"]
    out = bytearray()
    while len(out) < n:
        r = rng.random()
        if r < 0.14:   # call rel32
            out += b"\xe8" + struct.pack("<i", rng.randint(-60000, 60000))
        elif r < 0.18:  # jmp rel32
            out += b"\xe9" + struct.pack("<i", rng.randint(-4000, 4000))
        elif r < 0.24:  # E8/E9 bytes close together (exercises prev_mask)
            out += bytes(rng.choice([0xE8, 0xE9, 0x00, 0xFF, 0x12])
                         for _ in range(rng.randint(2, 7)))
        elif r < 0.27:
            out += bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 6)))
        else:
            out += rng.choice(ops)
    return bytes(out[:n])


def words_be(seed, n, special, p):
    rng = random.Random(seed)
    common = [0x7C0802A6, 0x9421FFF0, 0x38600000, 0x4E800020, 0x60000000,
              0x80010014, 0x7C6802A6, 0x3C600000]
    out = bytearray()
    while len(out) < n:
        if rng.random() < p:
            out += struct.pack(">I", special(rng))
        else:
            out += struct.pack(">I", rng.choice(common))
    return bytes(out[:n])


def words_le(seed, n, special, p, common):
    rng = random.Random(seed)
    out = bytearray()
    while len(out) < n:
        if rng.random() < p:
            out += struct.pack("<I", special(rng))
        else:
            out += struct.pack("<I", rng.choice(common))
    return bytes(out[:n])


def ppc_code(seed, n):
    # bl: opcode 18, 24-bit word offset, AA=0, LK=1
    return words_be(seed, n, lambda r: (18 << 26) | (r.randint(0, 1 << 22) << 2) | 1, 0.3)


def sparc_code(seed, n):
    def call(r):
        d = r.randint(-(1 << 20), 1 << 20) & 0x3FFFFFFF
        return 0x40000000 | d
    return words_be(seed, n, call, 0.3)


def arm_code(seed, n):
    common = [0xE92D4800, 0xE28DB004, 0xE24DD008, 0xE3A03000, 0xE1A00003,
              0xE8BD8800, 0xE59F3010, 0xE5933000]
    return words_le(seed, n, lambda r: 0xEB000000 | r.randint(0, 0xFFFFFF), 0.3, common)


def arm64_code(seed, n):
    common = [0xA9BF7BFD, 0x910003FD, 0xA8C17BFD, 0xD65F03C0, 0x52800000,
              0xF9400BE0, 0xB9001FE0, 0xD503201F]

    def special(r):
        if r.random() < 0.6:
            return (0x25 << 26) | r.getrandbits(26)  # BL
        imm = r.randint(-(1 << 17), 1 << 17) & 0x1FFFFF  # ADRP, in range
        return 0x90000000 | ((imm & 3) << 29) | (((imm >> 2) & 0x7FFFF) << 5) | r.randint(0, 30)
    return words_le(seed, n, special, 0.35, common)


def armthumb_code(seed, n):
    rng = random.Random(seed)
    common = [0xB580, 0xAF00, 0x2300, 0x4618, 0xBD80, 0x6878, 0x3301, 0x46BD]
    out = bytearray()
    while len(out) < n:
        if rng.random() < 0.25:
            off = rng.getrandbits(22)
            out += struct.pack("<HH", 0xF000 | ((off >> 11) & 0x7FF), 0xF800 | (off & 0x7FF))
        else:
            out += struct.pack("<H", rng.choice(common))
    return bytes(out[:n])


def ia64_code(seed, n):
    rng = random.Random(seed)
    branchy = [0x10, 0x11, 0x12, 0x13, 0x16, 0x17, 0x18, 0x19, 0x1C, 0x1D]
    out = bytearray()
    while len(out) < n:
        tmpl = rng.choice(branchy) if rng.random() < 0.7 else rng.getrandbits(5)
        bundle = tmpl
        for slot in range(3):
            ins = rng.getrandbits(41)
            if rng.random() < 0.6:
                ins &= ~(0xF << 37)
                ins |= 0x5 << 37
                ins &= ~(0x7 << 9)
            bundle |= ins << (5 + 41 * slot)
        out += bundle.to_bytes(16, "little")
    return bytes(out[:n])


def make_inputs():
    return {
        "empty": b"",
        "text": words_text(1, 6000),
        "text3k": words_text(2, 3000),
        "f32": float32_series(1024),
        "i16": int16_series(1024),
        "rand": random.Random(3).randbytes(1024),
        "x86": x86_code(4, 3000),
        "powerpc": ppc_code(5, 3072),
        "ia64": ia64_code(6, 2048),
        "arm": arm_code(7, 3072),
        "armthumb": armthumb_code(8, 3072),
        "sparc": sparc_code(9, 3072),
        "arm64": arm64_code(10, 3072),
    }


# --------------------------------------------------------------------------
# Hand-built .xz containers


def crc32le(b):
    return struct.pack("<I", zlib.crc32(b) & 0xFFFFFFFF)


_CRC64_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0xC96C5795D7870F42 if _c & 1 else _c >> 1
    _CRC64_TABLE.append(_c)


def crc64(b):
    c = 0xFFFFFFFFFFFFFFFF
    for x in b:
        c = _CRC64_TABLE[(c ^ x) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFFFFFFFFFF


CHECK_SIZES = [0, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64]


def check_field(check, data):
    if check == lzma.CHECK_NONE:
        return b""
    if check == lzma.CHECK_CRC32:
        return crc32le(data)
    if check == lzma.CHECK_CRC64:
        return struct.pack("<Q", crc64(data))
    if check == lzma.CHECK_SHA256:
        return hashlib.sha256(data).digest()
    # Unsupported check types: liblzma skips them, any content will do.
    return bytes((7 * i + 1) & 0xFF for i in range(CHECK_SIZES[check]))


def vli(x):
    out = bytearray()
    while x >= 0x80:
        out.append((x & 0x7F) | 0x80)
        x >>= 7
    out.append(x)
    return bytes(out)


def stream_header(check):
    flags = bytes([0, check])
    return b"\xfd7zXZ\x00" + flags + crc32le(flags)


def stream_footer(check, index_size, flags=None):
    flags = flags if flags is not None else bytes([0, check])
    body = struct.pack("<I", index_size // 4 - 1) + flags
    return crc32le(body) + body + b"YZ"


def build_block(data, chain, check, sizes=(True, True), extra_pad=0,
                csize_delta=0, usize_delta=0, reserved_flag=False,
                block_pad=b"\0\0\0"):
    payload = lzma.compress(data, format=lzma.FORMAT_RAW, filters=chain)
    flags = len(chain) - 1
    body = bytearray()
    if sizes[0]:
        flags |= 0x40
        body += vli(len(payload) + csize_delta)
    if sizes[1]:
        flags |= 0x80
        body += vli(len(data) + usize_delta)
    if reserved_flag:
        flags |= 0x04
    for f in chain:
        props = lzma._encode_filter_properties(f)
        body += vli(f["id"]) + vli(len(props)) + props
    hdr = bytearray([0, flags]) + body
    while (len(hdr) + 4) % 4:
        hdr.append(0)
    hdr += b"\0" * extra_pad
    hdr[0] = (len(hdr) + 4) // 4 - 1
    hdr = bytes(hdr) + crc32le(bytes(hdr))
    pad = block_pad[: (-len(payload)) % 4]
    chk = check_field(check, data)
    block = hdr + payload + pad + chk
    return block, (len(hdr) + len(payload) + len(chk), len(data))


def build_index(records):
    idx = bytearray(b"\0") + vli(len(records))
    for unpadded, usize in records:
        idx += vli(unpadded) + vli(usize)
    while len(idx) % 4:
        idx.append(0)
    return bytes(idx) + crc32le(bytes(idx))


def build_stream(blocks, check, index_records=None, footer_flags=None, backward_delta=0):
    out = bytearray(stream_header(check))
    records = []
    for data, chain, kw in blocks:
        b, rec = build_block(data, chain, check, **kw)
        out += b
        records.append(rec)
    idx = build_index(index_records if index_records is not None else records)
    out += idx
    out += stream_footer(check, len(idx) + 4 * backward_delta, footer_flags)
    return bytes(out)


# --------------------------------------------------------------------------
# Cases

cases = []
_files = {}


def py_decompress(data, fmt, filters):
    return lzma.decompress(data, format=fmt, filters=filters)


def add(name, data, fmt, expect, inputs, filters=None, compress=None, error=False,
        source="python", note=None, python_expect=None):
    """Record a vector after checking liblzma's behaviour on it."""
    assert name not in _files, name
    want = b"".join(inputs[k] for k in expect)
    if error:
        try:
            py_decompress(data, fmt, filters)
        except lzma.LZMAError:
            pass
        else:
            raise AssertionError("%s: Python decoded a vector expected to fail" % name)
    else:
        got = py_decompress(data, fmt, filters)
        pw = want if python_expect is None else b"".join(inputs[k] for k in python_expect)
        assert got == pw, name
    auto_ok = False
    if not error and fmt in (lzma.FORMAT_XZ, lzma.FORMAT_ALONE) and python_expect is None:
        try:
            auto_ok = py_decompress(data, lzma.FORMAT_AUTO, None) == want
        except lzma.LZMAError:
            pass
    path = name + ".bin"
    _files[name] = path
    with open(os.path.join(VECTORS, path), "wb") as f:
        f.write(data)
    c = {"name": name, "file": path, "format": fmt, "filters": filters,
         "expect": expect, "error": error, "auto_ok": auto_ok, "source": source}
    if compress is not None:
        c["compress"] = compress
    if note:
        c["note"] = note
    if python_expect is not None:
        c["python_expect"] = python_expect
    cases.append(c)


def py_case(name, inputs, inp, fmt=lzma.FORMAT_XZ, check=-1, preset=None, filters=None):
    data = lzma.compress(inputs[inp], format=fmt, check=check, preset=preset, filters=filters)
    add(name, data, fmt, [inp], inputs,
        filters=filters if fmt == lzma.FORMAT_RAW else None,
        compress={"format": fmt, "check": check, "preset": preset, "filters": filters})
    return data


def flip(data, i):
    b = bytearray(data)
    b[i] ^= 0x01
    return bytes(b)


hb_src = "handbuilt"


def main():
    os.makedirs(INPUTS, exist_ok=True)
    os.makedirs(VECTORS, exist_ok=True)
    for d in (INPUTS, VECTORS):
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    inputs = make_inputs()
    for k, v in inputs.items():
        with open(os.path.join(INPUTS, k + ".bin"), "wb") as f:
            f.write(v)

    # Formats x inputs, default options.
    for inp in ("empty", "text", "f32", "rand"):
        py_case("xz_%s" % inp, inputs, inp)
        py_case("alone_%s" % inp, inputs, inp, fmt=lzma.FORMAT_ALONE)
        py_case("raw_lzma2_%s" % inp, inputs, inp, fmt=lzma.FORMAT_RAW, filters=[{"id": L2}])
        py_case("raw_lzma1_%s" % inp, inputs, inp, fmt=lzma.FORMAT_RAW, filters=[{"id": L1}])

    # Every check type.
    for check, cname in ((lzma.CHECK_NONE, "none"), (lzma.CHECK_CRC32, "crc32"),
                         (lzma.CHECK_CRC64, "crc64"), (lzma.CHECK_SHA256, "sha256")):
        py_case("xz_check_%s" % cname, inputs, "text", check=check)

    # Presets 0-9, plain and extreme.
    for level in range(10):
        py_case("xz_preset%d" % level, inputs, "text3k", preset=level)
        py_case("xz_preset%de" % level, inputs, "text3k", preset=level | lzma.PRESET_EXTREME)
    py_case("alone_preset0", inputs, "text3k", fmt=lzma.FORMAT_ALONE, preset=0)
    py_case("alone_preset9e", inputs, "text3k", fmt=lzma.FORMAT_ALONE, preset=9 | lzma.PRESET_EXTREME)

    # Delta filter chains.
    for dist in (1, 2, 4, 8):
        py_case("xz_delta%d_f32" % dist, inputs, "f32",
                filters=[{"id": DELTA, "dist": dist}, {"id": L2, "preset": 6}])
    py_case("xz_delta2_i16", inputs, "i16", filters=[{"id": DELTA, "dist": 2}, {"id": L2, "preset": 1}])
    py_case("raw_delta4_lzma2", inputs, "f32", fmt=lzma.FORMAT_RAW,
            filters=[{"id": DELTA, "dist": 4}, {"id": L2}])
    py_case("raw_delta4_lzma1", inputs, "f32", fmt=lzma.FORMAT_RAW,
            filters=[{"id": DELTA, "dist": 4}, {"id": L1, "lc": 0, "lp": 2, "pb": 2}])
    py_case("xz_delta256", inputs, "text", filters=[{"id": DELTA, "dist": 256}, {"id": L2}])

    # BCJ filter chains: .xz and raw (raw vectors let the Go tests compare
    # the BCJ-encoded bytes with liblzma's directly).
    for bname, bid in BCJ.items():
        py_case("xz_%s" % bname, inputs, bname, filters=[{"id": bid}, {"id": L2}])
        py_case("raw_%s_lzma2" % bname, inputs, bname, fmt=lzma.FORMAT_RAW,
                filters=[{"id": bid}, {"id": L2}])
        py_case("raw_%s_off" % bname, inputs, bname, fmt=lzma.FORMAT_RAW,
                filters=[{"id": bid, "start_offset": 0x1234 * 4}, {"id": L2}])
    py_case("xz_x86_off", inputs, "x86", filters=[{"id": lzma.FILTER_X86, "start_offset": 0x4000}, {"id": L2}])
    py_case("xz_x86_delta", inputs, "x86",
            filters=[{"id": lzma.FILTER_X86}, {"id": DELTA, "dist": 3}, {"id": L2, "preset": 9}])

    # LZMA1/LZMA2 with custom lc/lp/pb/dict_size.
    for lc, lp, pb in ((0, 0, 0), (4, 0, 4), (0, 4, 0), (1, 3, 1), (2, 2, 3)):
        py_case("raw_lzma1_lc%dlp%dpb%d" % (lc, lp, pb), inputs, "i16", fmt=lzma.FORMAT_RAW,
                filters=[{"id": L1, "lc": lc, "lp": lp, "pb": pb, "dict_size": 1 << 16}])
    for lc, lp, pb in ((0, 4, 0), (4, 0, 4)):
        py_case("raw_lzma2_lc%dlp%dpb%d" % (lc, lp, pb), inputs, "i16", fmt=lzma.FORMAT_RAW,
                filters=[{"id": L2, "lc": lc, "lp": lp, "pb": pb, "dict_size": 1 << 16}])
    py_case("alone_lzma1_custom", inputs, "i16", fmt=lzma.FORMAT_ALONE,
            filters=[{"id": L1, "lc": 0, "lp": 2, "pb": 0, "dict_size": 5000}])
    py_case("raw_lzma1_preset1", inputs, "text", fmt=lzma.FORMAT_RAW, filters=[{"id": L1, "preset": 1}])

    # .lzma with a known size in the header (as 7-Zip and older tools write
    # it), followed by an end marker, and the size-0 forms.
    known = bytearray(lzma.compress(inputs["text3k"], format=lzma.FORMAT_ALONE))
    known[5:13] = struct.pack("<Q", len(inputs["text3k"]))
    add("alone_known_size_eopm", bytes(known), lzma.FORMAT_ALONE, ["text3k"], inputs, source=hb_src)
    add("alone_size0_rc", bytes.fromhex("5d0000800000000000000000000000000000"), lzma.FORMAT_ALONE,
        ["empty"], inputs, source=hb_src)
    add("alone_size0_eopm", bytes.fromhex("5d0000800000000000000000000083fffbffffc0000000"),
        lzma.FORMAT_ALONE, ["empty"], inputs, source=hb_src)
    bad = bytearray(known)
    bad[5:13] = struct.pack("<Q", len(inputs["text3k"]) + 1)
    add("alone_known_size_wrong", bytes(bad), lzma.FORMAT_ALONE, [], inputs, error=True, source=hb_src)

    # Concatenated streams (Python's lzma.decompress loop semantics).
    xa = lzma.compress(inputs["text3k"])
    xb = lzma.compress(inputs["rand"], check=lzma.CHECK_SHA256)
    aa = lzma.compress(inputs["text3k"], format=lzma.FORMAT_ALONE)
    ab = lzma.compress(inputs["i16"], format=lzma.FORMAT_ALONE, preset=1)
    add("multi_xz_xz", xa + xb, lzma.FORMAT_XZ, ["text3k", "rand"], inputs)
    add("multi_xz_empty_xz", xa + lzma.compress(b"") + xb, lzma.FORMAT_XZ, ["text3k", "rand"], inputs)
    add("multi_alone_alone", aa + ab, lzma.FORMAT_ALONE, ["text3k", "i16"], inputs)
    add("multi_auto_mixed", xa + ab + xb, lzma.FORMAT_AUTO, ["text3k", "i16", "rand"], inputs)
    add("multi_xz_then_alone", xa + ab, lzma.FORMAT_XZ, ["text3k"], inputs,
        note="FORMAT_XZ: a following .lzma stream is not a valid stream and is ignored")
    r2 = [{"id": L2}]
    add("multi_raw_lzma2", lzma.compress(inputs["text3k"], format=lzma.FORMAT_RAW, filters=r2)
        + lzma.compress(inputs["rand"], format=lzma.FORMAT_RAW, filters=r2),
        lzma.FORMAT_RAW, ["text3k", "rand"], inputs, filters=r2)
    r1 = [{"id": L1}]
    add("multi_raw_lzma1", lzma.compress(inputs["text3k"], format=lzma.FORMAT_RAW, filters=r1)
        + lzma.compress(inputs["i16"], format=lzma.FORMAT_RAW, filters=r1),
        lzma.FORMAT_RAW, ["text3k", "i16"], inputs, filters=r1)
    add("trailing_garbage", xa + b"garbage12345", lzma.FORMAT_XZ, ["text3k"], inputs,
        note="trailing data that is not a stream is ignored, as in Python")
    add("trailing_garbage_auto", xa + b"garbage", lzma.FORMAT_AUTO, ["text3k"], inputs)
    add("trailing_truncated_stream", xa + xb[:40], lzma.FORMAT_XZ, [], inputs, error=True,
        note="a truncated following stream is an error, as in Python")

    # Hand-built .xz streams: multiple blocks, sizes present/absent, header
    # padding, unsupported check types, 3-filter chains, empty blocks.
    t = inputs["text"]
    lz2 = [{"id": L2, "preset": 6}]
    hb = "handbuilt"
    add("hb_blocks_sizes", build_stream([(t[:2000], lz2, {}), (t[2000:4000], lz2, {}), (t[4000:], lz2, {})],
                                        lzma.CHECK_CRC32), lzma.FORMAT_XZ, ["text"], inputs, source=hb)
    add("hb_blocks_nosizes", build_stream([(t[:2000], lz2, {"sizes": (False, False)}),
                                           (t[2000:4000], lz2, {"sizes": (True, False)}),
                                           (t[4000:], lz2, {"sizes": (False, True)})],
                                          lzma.CHECK_CRC64), lzma.FORMAT_XZ, ["text"], inputs, source=hb)
    add("hb_header_padding", build_stream([(t, lz2, {"extra_pad": 12})], lzma.CHECK_SHA256),
        lzma.FORMAT_XZ, ["text"], inputs, source=hb)
    for chk in (2, 5, 7, 13):
        add("hb_check%d" % chk, build_stream([(t, lz2, {"sizes": (False, False)})], chk),
            lzma.FORMAT_XZ, ["text"], inputs, source=hb,
            note="check type %d is not supported by liblzma and is skipped" % chk)
    x = inputs["x86"]
    chain3 = [{"id": lzma.FILTER_X86, "start_offset": 16}, {"id": DELTA, "dist": 2}, {"id": L2, "preset": 2}]
    add("hb_chain3", build_stream([(x, chain3, {}), (x, chain3, {"sizes": (False, False)})], lzma.CHECK_CRC32),
        lzma.FORMAT_XZ, ["x86", "x86"], inputs, source=hb)
    add("hb_empty_block", build_stream([(b"", lz2, {}), (t, lz2, {}), (b"", lz2, {"sizes": (False, False)})],
                                       lzma.CHECK_CRC64), lzma.FORMAT_XZ, ["text"], inputs, source=hb)
    s1 = build_stream([(inputs["text3k"], lz2, {})], lzma.CHECK_CRC32)
    s2 = build_stream([(inputs["rand"], lz2, {})], lzma.CHECK_NONE)
    padded = s1 + b"\0" * 4 + s2 + b"\0" * 8
    add("hb_stream_padding", padded, lzma.FORMAT_XZ, ["text3k", "rand"], inputs, source=hb,
        python_expect=["text3k"],
        note="xz spec / xz tool: padding between streams is skipped and the next "
             "stream decoded; Python's lzma.decompress stops at the padding")
    if shutil.which("xz"):
        got = subprocess.run(["xz", "-dc"], input=padded, capture_output=True, check=True).stdout
        assert got == inputs["text3k"] + inputs["rand"], "xz tool disagrees on stream padding"

    # Negative cases (liblzma rejects all of them).
    one = [(t, lz2, {})]
    assert len(xa) - 12 - 8 - 1 > 12 + 12
    add("bad_check_crc64", flip(xa, len(xa) - 12 - 8 - 1), lzma.FORMAT_XZ, [], inputs, error=True)
    add("bad_index_crc", flip(xa, len(xa) - 12 - 1), lzma.FORMAT_XZ, [], inputs, error=True)
    add("bad_block_header_crc", flip(xa, 12 + 5), lzma.FORMAT_XZ, [], inputs, error=True)
    add("bad_stream_header_crc", flip(xa, 9), lzma.FORMAT_XZ, [], inputs, error=True)
    add("bad_lzma2_data", flip(xa, 12 + 12 + 40), lzma.FORMAT_XZ, [], inputs, error=True)
    add("truncated_xz", xa[:-1], lzma.FORMAT_XZ, [], inputs, error=True)
    add("truncated_xz_header", xa[:8], lzma.FORMAT_XZ, [], inputs, error=True)
    add("truncated_alone", aa[:-3], lzma.FORMAT_ALONE, [], inputs, error=True)
    rl2 = lzma.compress(t, format=lzma.FORMAT_RAW, filters=r2)
    add("truncated_raw_lzma2", rl2[:-1], lzma.FORMAT_RAW, [], inputs, filters=r2, error=True,
        note="LZMA2 end marker missing")
    rl1 = lzma.compress(t, format=lzma.FORMAT_RAW, filters=r1)
    add("truncated_raw_lzma1", rl1[:-2], lzma.FORMAT_RAW, [], inputs, filters=r1, error=True)
    add("empty_input_xz", b"", lzma.FORMAT_XZ, [], inputs, error=True)
    add("empty_input_raw", b"", lzma.FORMAT_RAW, [], inputs, filters=r2, error=True)
    add("not_xz", b"definitely not an xz stream", lzma.FORMAT_XZ, [], inputs, error=True)
    add("hb_bad_usize", build_stream([(t, lz2, {"usize_delta": 1})], lzma.CHECK_CRC32),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    add("hb_bad_usize_small", build_stream([(t, lz2, {"usize_delta": -1})], lzma.CHECK_CRC32),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    add("hb_bad_csize", build_stream([(t, lz2, {"csize_delta": -1})], lzma.CHECK_CRC32),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    add("hb_index_count", build_stream(one, lzma.CHECK_CRC32, index_records=[(100, 8000), (12, 1)]),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    _, rec = build_block(t, lz2, lzma.CHECK_CRC32)
    add("hb_index_usize", build_stream(one, lzma.CHECK_CRC32, index_records=[(rec[0], rec[1] + 1)]),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    add("hb_index_unpadded", build_stream(one, lzma.CHECK_CRC32, index_records=[(rec[0] + 4, rec[1])]),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    add("hb_footer_flags", build_stream(one, lzma.CHECK_CRC32, footer_flags=bytes([0, lzma.CHECK_CRC64])),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    add("hb_backward_size", build_stream(one, lzma.CHECK_CRC32, backward_delta=1),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    add("hb_reserved_flag", build_stream([(t, lz2, {"reserved_flag": True})], lzma.CHECK_CRC32),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)
    n = 4000
    while len(lzma.compress(t[:n], format=lzma.FORMAT_RAW, filters=lz2)) % 4 == 0:
        n += 1
    add("hb_block_padding", build_stream([(t[:n], lz2, {"block_pad": b"\1\1\1"})], lzma.CHECK_CRC32),
        lzma.FORMAT_XZ, [], inputs, error=True, source=hb)

    # ARM64 BCJ via the xz tool (liblzma >= 5.4 decodes it; Python's lzma
    # module cannot encode it).
    if shutil.which("xz"):
        for name, extra in (("xz_arm64", []), ("xz_arm64_off", ["--arm64=start=4096"])):
            args = ["xz", "-c", "--format=xz", "--check=crc32"] + (extra or ["--arm64"]) + ["--lzma2=preset=6"]
            data = subprocess.run(args, input=inputs["arm64"], capture_output=True, check=True).stdout
            add(name, data, lzma.FORMAT_XZ, ["arm64"], inputs, source="xz-cli")
        # Raw ARM64 can't be decoded by Python's lzma module (it has no filter
        # specifier for ARM64), so this one is checked with the xz tool; the
        # Go tests also decode it with just LZMA2 to get liblzma's
        # ARM64-encoded bytes for a byte-exact comparison.
        data = subprocess.run(["xz", "-c", "--format=raw", "--arm64", "--lzma2=preset=6"],
                              input=inputs["arm64"], capture_output=True, check=True).stdout
        got = subprocess.run(["xz", "-dc", "--format=raw", "--arm64", "--lzma2=preset=6"],
                             input=data, capture_output=True, check=True).stdout
        assert got == inputs["arm64"]
        with open(os.path.join(VECTORS, "raw_arm64_lzma2.bin"), "wb") as f:
            f.write(data)
        cases.append({"name": "raw_arm64_lzma2", "file": "raw_arm64_lzma2.bin", "format": 3,
                      "filters": [{"id": FILTER_ARM64}, {"id": L2}], "expect": ["arm64"],
                      "error": False, "auto_ok": False, "source": "xz-cli"})
        # Multi-threaded xz: several blocks with sizes in the headers.
        data = subprocess.run(["xz", "-c", "-T2", "--block-size=2000", "--check=sha256", "-1"],
                              input=inputs["text"], capture_output=True, check=True).stdout
        add("xzcli_mt_blocks", data, lzma.FORMAT_XZ, ["text"], inputs, source="xz-cli")
        data = subprocess.run(["xz", "-c", "-T1", "--block-size=3000", "--check=crc32", "-6"],
                              input=inputs["f32"], capture_output=True, check=True).stdout
        add("xzcli_st_blocks", data, lzma.FORMAT_XZ, ["f32"], inputs, source="xz-cli")

    manifest = {
        "generator": "gen.py",
        "python_lzma": "liblzma via Python %s" % ".".join(map(str, __import__("sys").version_info[:3])),
        "inputs": sorted(inputs),
        "cases": cases,
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        # One case per line keeps the file small and diffs readable.
        head = json.dumps({k: v for k, v in manifest.items() if k != "cases"})
        f.write(head[:-1] + ', "cases": [\n')
        f.write(",\n".join(json.dumps(c) for c in cases))
        f.write("\n]}\n")
    total = sum(os.path.getsize(os.path.join(d, f)) for d in (INPUTS, VECTORS) for f in os.listdir(d))
    print("%d cases, %d bytes of vectors and inputs" % (len(cases), total))


if __name__ == "__main__":
    main()
