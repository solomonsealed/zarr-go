#!/usr/bin/env python3
"""Golden vectors for internal/blosc, produced with numcodecs 0.17.0 (which
bundles c-blosc 1.21.7.dev, commit 04c06fb7339d3df3600b1862077b8699b30a0e1d).

Usage (needs numpy and numcodecs):

    python internal/blosc/testdata/gen.py            # regenerate testdata
    python internal/blosc/testdata/gen.py verify DIR # check Go-written frames

Generation writes, next to this script:

golden.txt.gz   one line per numcodecs.Blosc(...).encode() call:

                    cname clevel shuffle typesize blocksize spec size digest

                For blosclz, lz4 and lz4hc, digest is the first 20 hex digits
                of the sha256 of the frame (the Go port must be byte-identical).
                For zlib and zstd it is "hdr:" + the first 12 header bytes in
                hex (version, versionlz, flags, typesize, nbytes, blocksize):
                the frame layout must match, the payloads legitimately
                differ. spec is a data generator "kind:n:seed" (see gen(),
                mirrored by testdata_test.go). typesize is passed explicitly
                (Blosc(typesize=...)), except "i<k>" which encodes a numpy
                array of itemsize k with typesize=None (numcodecs then uses
                the item size).

frames/*.blosc  complete numcodecs frames (zlib, zstd, and multi-threaded
                frames whose blocks are stored out of order) that Go must
                decode; the file name holds the parameters.

numcodecs is run with blosc.use_threads = False (blosc_compress_ctx with one
thread): with several threads c-blosc writes the blocks of multi-block
buffers in completion order, so its output is not deterministic.

"verify DIR" decodes every DIR/*.blosc written by the Go test
(BLOSC_EXPORT_DIR=DIR go test -run TestExport ./internal/blosc/) with
numcodecs and compares it with the regenerated input; blosclz, lz4 and lz4hc
frames must moreover be byte-identical to numcodecs' own encoding.
"""

import gzip
import hashlib
import os
import sys

import numpy as np
import numcodecs
from numcodecs import blosc

MASK = (1 << 64) - 1
HERE = os.path.dirname(os.path.abspath(__file__))


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
        if len(_cache) > 64:
            _cache.clear()
        kind, n, seed = spec.split(":")
        _cache[spec] = gen(kind, int(n), int(seed))
    return _cache[spec]


def encode(src, cname, clevel, shuffle, typesize, blocksize):
    """numcodecs.Blosc(...).encode(); typesize "i<k>" passes a numpy array of
    itemsize k (n must be a multiple of k) and lets numcodecs use its item
    size."""
    if isinstance(typesize, str):
        k = int(typesize[1:])
        arr = np.frombuffer(src, dtype=f"V{k}") if k not in (1, 2, 4, 8) else \
            np.frombuffer(src, dtype={1: "u1", 2: "<u2", 4: "<f4", 8: "<f8"}[k])
        return numcodecs.Blosc(cname, clevel, shuffle, blocksize).encode(arr)
    return numcodecs.Blosc(cname, clevel, shuffle, blocksize, typesize=typesize).encode(
        np.frombuffer(src, dtype=np.uint8))


KINDS = ["zero", "rand", "text", "runs", "sparse", "periodic", "walk8", "walk16",
         "walk32", "walk64", "f64", "f32", "mixed"]
TYPESIZES = [1, 1, 2, 2, 4, 4, 4, 8, 8, 8, 3, 5, 7, 12, 16, 17, 24, 32, 33, 64, 100,
             128, 255, 256, 1000, "i2", "i4", "i8", "i16"]
BLOCKSIZES = [0] * 10 + [1, 64, 127, 128, 129, 256, 1000, 4096, 4099, 8192, 16384,
                         32768, 65536, 65537, 131072, 262144, 1048576, 4194304, -1]


def specs():
    small = [0, 1, 2, 7, 15, 16, 17, 100, 127, 128, 129, 130, 200, 255, 256, 257, 500,
             1000, 1023, 1024, 2049, 4095, 4096, 4097, 8192, 10000]
    medium = [16383, 32767, 32768, 32769, 50000, 65535, 65536, 65537, 100000, 131072,
              200003, 262144, 262147, 500000]
    large = [1 << 20, (1 << 20) + 5, (2 << 20) + 3, (3 << 20) + 11]
    out = []
    seed = 1
    i = 0
    for sizes, per in ((small, 4), (medium, 4), (large, 3)):
        for n in sizes:
            for _ in range(per):
                out.append(f"{KINDS[i % len(KINDS)]}:{n}:{seed}")
                i += 1
                seed += 1
    return out


def pick_typesize(r, n):
    while True:
        ts = TYPESIZES[r.intn(len(TYPESIZES))]
        if isinstance(ts, str) and n % int(ts[1:]) != 0:
            continue
        return ts


def main():
    lines = []
    frames = {}
    r = Rand(12345)
    blosc.use_threads = False

    def add(cname, clevel, shuffle, ts, bs, spec):
        src = data(spec)
        enc = encode(src, cname, clevel, shuffle, ts, bs)
        if cname in ("zlib", "zstd"):
            digest = "hdr:" + enc[:12].hex()
        else:
            digest = hashlib.sha256(enc).hexdigest()[:20]
        lines.append(f"{cname}\t{clevel}\t{shuffle}\t{ts}\t{bs}\t{spec}\t{len(enc)}\t{digest}")
        return enc

    for spec in specs():
        n = int(spec.split(":")[1])
        k = 30 if n <= 10000 else 18 if n <= 500000 else 5
        for _ in range(k):
            cname = ["blosclz", "lz4", "lz4hc"][r.intn(3)]
            add(cname, r.intn(10), r.intn(4) - 1, pick_typesize(r, n),
                BLOCKSIZES[r.intn(len(BLOCKSIZES))], spec)
        for _ in range(2 if n <= 500000 else 1):
            cname = ["zlib", "zstd"][r.intn(2)]
            add(cname, r.intn(10), r.intn(4) - 1, pick_typesize(r, n),
                BLOCKSIZES[r.intn(len(BLOCKSIZES))], spec)

    # Exhaustive codec x level x shuffle grid on a few inputs.
    for spec, ts in [("walk32:100000:9001", 4), ("f64:65536:9002", 8), ("text:5000:9003", 1),
                     ("mixed:300000:9004", 2), ("walk16:40000:9005", 2), ("sparse:20000:9006", 16)]:
        for cname in ["blosclz", "lz4", "lz4hc", "zlib", "zstd"]:
            for clevel in range(10):
                for shuffle in [0, 1, 2]:
                    add(cname, clevel, shuffle, ts, 0, spec)

    # Complete frames for Go to decode.
    for cname, clevel, shuffle, ts, bs, spec in [
        ("zstd", 1, 1, 4, 0, "walk32:20000:7001"),
        ("zstd", 5, 2, 8, 0, "f64:70000:7002"),
        ("zstd", 9, 0, 1, 4096, "text:9000:7003"),
        ("zstd", 3, 1, 2, 0, "rand:3000:7004"),
        ("zlib", 1, 1, 4, 0, "walk32:20000:7005"),
        ("zlib", 6, 2, 2, 0, "walk16:50000:7006"),
        ("zlib", 9, 0, 1, 1000, "text:9000:7007"),
        ("zlib", 4, 1, 16, 0, "sparse:30000:7008"),
    ]:
        enc = encode(data(spec), cname, clevel, shuffle, ts, bs)
        frames[f"{cname}_{clevel}_{shuffle}_{ts}_{bs}_{spec.replace(':', '_')}"] = enc

    # Multi-threaded frames: blocks may be stored out of order.
    blosc.use_threads = True
    blosc.set_nthreads(4)
    for cname, clevel, shuffle, ts, spec in [
        ("blosclz", 5, 1, 4, "walk32:300000:7101"),
        ("lz4", 9, 2, 8, "f64:300000:7102"),
        ("zstd", 3, 1, 2, "walk16:300000:7103"),
    ]:
        src = data(spec)
        for attempt in range(500):
            enc = encode(src, cname, clevel, shuffle, ts, 16384)
            starts = np.frombuffer(enc[16:16 + 4 * ((len(src) + 16383) // 16384)], "<u4")
            if np.any(np.diff(starts.astype(np.int64)) < 0):
                break
        frames[f"threads_{cname}_{clevel}_{shuffle}_{ts}_16384_{spec.replace(':', '_')}"] = enc

    path = os.path.join(HERE, "golden.txt.gz")
    text = (f"# generated by gen.py with numcodecs {numcodecs.__version__}, "
            f"c-blosc {blosc.VERSION_STRING}\n" + "\n".join(lines) + "\n")
    with open(path, "wb") as f:
        f.write(gzip.compress(text.encode(), mtime=0))
    fdir = os.path.join(HERE, "frames")
    os.makedirs(fdir, exist_ok=True)
    for name in os.listdir(fdir):
        os.remove(os.path.join(fdir, name))
    for name, enc in frames.items():
        with open(os.path.join(fdir, name + ".blosc"), "wb") as f:
            f.write(enc)
    print(f"wrote {len(lines)} vectors to {path} and {len(frames)} frames", file=sys.stderr)


def verify(d):
    """Decode Go-written frames named
    <cname>_<clevel>_<shuffle>_<typesize>_<blocksize>__<kind>_<n>_<seed>.blosc."""
    blosc.use_threads = False
    ok = bad = same = 0
    for name in sorted(os.listdir(d)):
        if not name.endswith(".blosc"):
            continue
        spec = name[:-len(".blosc")].split("__")[-1].replace("_", ":")
        with open(os.path.join(d, name), "rb") as f:
            enc = f.read()
        if not data(spec):
            # numcodecs cannot decode empty frames, not even its own
            # ("error during blosc decompression: 0"); compare headers.
            cname, clevel, shuffle, ts, bs = name.split("__")[0].split("_")
            ref = numcodecs.Blosc(cname, int(clevel), int(shuffle), int(bs), typesize=int(ts)).encode(b"")
            if enc != ref:
                print(f"FAIL {name}: empty frame {enc.hex()} != numcodecs {ref.hex()}")
                bad += 1
            else:
                ok += 1
            continue
        try:
            dec = bytes(numcodecs.Blosc().decode(enc))
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
            bad += 1
            continue
        if dec != data(spec):
            print(f"FAIL {name}: decoded data differs")
            bad += 1
            continue
        ok += 1
        cname, clevel, shuffle, ts, bs = name.split("__")[0].split("_")
        if cname in ("blosclz", "lz4", "lz4hc"):
            ref = encode(data(spec), cname, int(clevel), int(shuffle), int(ts), int(bs))
            if enc != ref:
                print(f"FAIL {name}: not byte-identical to numcodecs")
                bad += 1
            else:
                same += 1
    print(f"{ok} frames decoded correctly ({same} also byte-identical to numcodecs), {bad} failures")
    return bad == 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        sys.exit(0 if verify(sys.argv[2]) else 1)
    main()
