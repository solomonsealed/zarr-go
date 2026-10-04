#!/usr/bin/env python3
"""Generate golden vectors for internal/lz4 from the LZ4 library bundled
with numcodecs (numcodecs 0.17.0 vendors LZ4 1.10.0 via c-blosc).

Usage (from any directory; needs numpy and numcodecs):

    python internal/lz4/testdata/gen.py

Writes golden.txt.gz next to this script. The Go test (golden_test.go)
regenerates every input with an identical pseudo-random generator (see
testdata_test.go), runs the Go port and compares.

Line format (tab separated), one line per input and function:

    func  spec  param  results  sha

func      "fast":    numcodecs.LZ4(acceleration=a).encode(data) without its
                     4-byte length prefix, i.e. LZ4_compress_fast with
                     capacity LZ4_compressBound(n); results lists a=size.
          "hc":      LZ4_compress_HC(src, dst, n, LZ4_compressBound(n), level);
                     results lists level=size.
          "fastcap": LZ4_compress_fast(src, dst, n, cap, param) with smaller
                     capacities (limitedOutput mode); results lists cap=ret.
          "hccap":   LZ4_compress_HC(src, dst, n, cap, param), likewise.
spec      data generator "kind:n:seed" (see gen()).
param     acceleration or level for the *cap functions, "-" otherwise.
results   comma separated key=return value (0 = did not fit).
sha       first 20 hex digits of sha256 over, for each result in order,
          the return value as 4 little-endian bytes followed by the output.
"""

import ctypes
import gzip
import hashlib
import os
import sys

import numpy as np
import numcodecs
import numcodecs.blosc
import numcodecs.lz4

MASK = (1 << 64) - 1


class Rand:
    """splitmix64"""

    def __init__(self, seed):
        self.s = seed & MASK

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def intn(self, n):
        return self.next() % n

    def bytes(self, n):
        out = bytearray()
        while len(out) < n:
            out += self.next().to_bytes(8, "little")
        return bytes(out[:n])


WORDS = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "zarr", "blosc", "chunk", "array", "data", "compress", "shuffle", "\n"]
MIXED_KINDS = ["zero", "rand", "text", "runs", "walk32", "f64", "periodic"]


def gen(kind, n, seed):
    """Deterministic test data; must match testdata_test.go exactly."""
    r = Rand(seed)
    if n == 0:
        return b""
    if kind == "zero":
        return bytes(n)
    if kind == "rand":
        return r.bytes(n)
    if kind == "text":
        out = bytearray()
        while len(out) < n:
            w = WORDS[r.intn(len(WORDS))]
            out += w.encode() if w == "\n" else (w + " ").encode()
        return bytes(out[:n])
    if kind == "runs":
        out = bytearray()
        while len(out) < n:
            v = r.next()
            out += bytes([v & 0xFF]) * (1 + ((v >> 8) % 64))
        return bytes(out[:n])
    if kind == "sparse":
        out = bytearray(n)
        for _ in range(n // 100):
            v = r.next()
            out[v % n] = (v >> 32) & 0xFF
        return bytes(out)
    if kind == "periodic":
        p = 1 + r.intn(300)
        pat = r.bytes(p)
        return (pat * (n // p + 1))[:n]
    if kind in ("walk8", "walk16", "walk32", "walk64"):
        size = int(kind[4:]) // 8
        m = n // size + 1
        start = r.next() & 0xFFFF
        steps = np.frombuffer(r.bytes(m), dtype=np.uint8).astype(np.int64) % 7 - 3
        vals = start + np.cumsum(steps)
        dt = {1: "<u1", 2: "<u2", 4: "<u4", 8: "<u8"}[size]
        return (vals & ((1 << (8 * size)) - 1) if size < 8 else vals).astype(dt).tobytes()[:n]
    if kind == "f64":
        m = n // 8 + 1
        b = np.frombuffer(r.bytes(m), dtype=np.uint8).astype(np.int64)
        v = np.arange(m, dtype=np.int64) * 3 + b % 4
        return (v.astype(np.float64) / 8).astype("<f8").tobytes()[:n]
    if kind == "f32":
        m = n // 4 + 1
        b = np.frombuffer(r.bytes(m), dtype=np.uint8).astype(np.int64)
        v = 1000 + np.cumsum(b % 9 - 4)
        return (v.astype(np.float32) / np.float32(16)).astype("<f4").tobytes()[:n]
    if kind == "mixed":
        out = bytearray()
        while len(out) < n:
            k = MIXED_KINDS[r.intn(len(MIXED_KINDS))]
            seglen = 1 + r.intn(65536)
            out += gen(k, seglen, r.next())
        return bytes(out[:n])
    raise ValueError(kind)


_cache = {}


def data(spec):
    if spec not in _cache:
        kind, n, seed = spec.split(":")
        _cache[spec] = gen(kind, int(n), int(seed))
    return _cache[spec]


class Group:
    def __init__(self):
        self.h = hashlib.sha256()
        self.items = []

    def add(self, key, res, out):
        self.items.append(f"{key}={res}")
        self.h.update(res.to_bytes(4, "little", signed=True))
        self.h.update(out)

    def line(self, fn, spec, param):
        return f"{fn}\t{spec}\t{param}\t{','.join(self.items)}\t{self.h.hexdigest()[:20]}"


# numcodecs' lz4 extension exports lz4.c only; LZ4_compress_HC (lz4hc.c from
# the same vendored LZ4 1.10.0) is linked into the blosc extension.
lib = ctypes.CDLL(numcodecs.lz4.__file__)
hclib = ctypes.CDLL(numcodecs.blosc.__file__)
lib.LZ4_compress_fast.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.LZ4_compress_fast.restype = ctypes.c_int
hclib.LZ4_compress_HC.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
hclib.LZ4_compress_HC.restype = ctypes.c_int
lib.LZ4_compressBound.argtypes = [ctypes.c_int]
lib.LZ4_compressBound.restype = ctypes.c_int


def call(fn, src, cap, param):
    buf = ctypes.create_string_buffer(max(cap, 1))
    n = fn(src, buf, len(src), cap, param)
    return n, buf.raw[:n]


def main():
    lines = []

    small = [0, 1, 2, 4, 5, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20, 24, 31, 32, 33, 40, 63, 64, 65, 100]
    medium = [127, 128, 200, 255, 256, 257, 511, 1000, 1024, 4096, 4099, 10000, 65535, 65536]
    boundary = [65546, 65547, 65548, 70000]  # LZ4_64Klimit = 65547: byU16 -> byU32
    large = [100000, 262144 + 7, 1048576 + 3]
    kinds = ["zero", "rand", "text", "runs", "sparse", "periodic", "walk8", "walk16",
             "walk32", "walk64", "f64", "f32", "mixed"]

    specs = []
    seed = 1
    for n in small:
        for k in ["rand", "text", "zero", "periodic"]:
            specs.append(f"{k}:{n}:{seed}")
            seed += 1
    for n in medium + boundary:
        for k in kinds:
            specs.append(f"{k}:{n}:{seed}")
            seed += 1
    for n in large:
        for k in ["rand", "text", "runs", "walk32", "f64", "mixed", "zero"]:
            specs.append(f"{k}:{n}:{seed}")
            seed += 1

    accels = [1, 2, 3, 5, 9, 17, 100, 65537, 100000, 0, -5]
    for spec in specs:
        src = data(spec)
        n = len(src)
        bound = lib.LZ4_compressBound(n)
        big = n > 300000

        g = Group()
        for a in (accels if n <= 70000 else [1, 4, 9, 65537]):
            enc = numcodecs.LZ4(acceleration=a).encode(src)
            assert int.from_bytes(enc[:4], "little") == n
            g.add(a, len(enc) - 4, enc[4:])
        lines.append(g.line("fast", spec, "-"))

        g = Group()
        for level in (range(-1, 14) if not big else [1, 2, 3, 6, 9]):
            if n > 70000 and level >= 11:
                continue
            res, out = call(hclib.LZ4_compress_HC, src, bound, level)
            g.add(level, res, out)
        lines.append(g.line("hc", spec, "-"))

        if 0 < n <= 70000:
            # limitedOutput mode: capacities around the compressed size and
            # around the input size (what blosc uses).
            for a in [1, 4]:
                full, _ = call(lib.LZ4_compress_fast, src, bound, a)
                g = Group()
                for cap in sorted({max(c, 0) for c in [n, n - 1, full, full - 1, full + 1, full + 7,
                                                       full + 8, full - 8, n * 3 // 4, 1, 0]}):
                    res, out = call(lib.LZ4_compress_fast, src, cap, a)
                    g.add(cap, res, out)
                lines.append(g.line("fastcap", spec, a))
            for level in [2, 4, 9, 10, 12]:
                full, _ = call(hclib.LZ4_compress_HC, src, bound, level)
                g = Group()
                for cap in sorted({max(c, 0) for c in [n, n - 1, full, full - 1, full + 1, full + 6,
                                                       full + 8, full - 8, 0]}):
                    res, out = call(hclib.LZ4_compress_HC, src, cap, level)
                    g.add(cap, res, out)
                lines.append(g.line("hccap", spec, level))

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden.txt.gz")
    text = (f"# generated by gen.py with numcodecs {numcodecs.__version__}, "
            f"LZ4 {numcodecs.lz4.VERSION_STRING}\n" + "\n".join(lines) + "\n")
    with open(path, "wb") as f:
        f.write(gzip.compress(text.encode(), mtime=0))
    print(f"wrote {len(lines)} vectors to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
