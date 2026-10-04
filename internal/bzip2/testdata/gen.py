#!/usr/bin/env python3
"""Generate golden test vectors for internal/bzip2 using Python's bz2 module.

The reference is libbzip2 1.0.8 as driven by bz2.compress(data, compresslevel)
(BZ2_bzCompressInit(level, 0, workFactor=0 -> 30), one BZ_RUN pass, then
BZ_FINISH).

To keep the committed data small, most inputs are described by a generator
spec that golden_test.go regenerates bit-for-bit in Go; only the length and
SHA-256 of input and expected output are stored (plus the full output, base64,
when it is small). Inputs that cannot be regenerated exactly in Go (numpy sin /
random-normal data) are stored as their expected .bz2 output in golden/; the
Go test recovers the input by decompressing it with compress/bzip2.

Usage (from any directory; writes next to this script):

    python3 gen.py

Requires numpy. Re-running must reproduce vectors.json and golden/*.bz2
byte-for-byte (given the same libbzip2).
"""

import base64
import bz2
import ctypes
import hashlib
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")
INLINE_MAX = 2048  # store outputs up to this size in full (base64)

MASK64 = (1 << 64) - 1


class SplitMix64:
    """splitmix64; golden_test.go has an identical implementation."""

    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)


WORDS = (
    "the of and to in is that it for as with was on be by at this are from or "
    "an have not which but all were they one their has more been can would "
    "data array chunk zarr compression block sort huffman table stream bytes "
    "float value index shape dtype codec level store group metadata"
).split()


def generate(spec):
    """Materialise a generator spec. Must match generate() in golden_test.go."""
    kind = spec["kind"]
    if kind == "empty":
        return b""
    if kind == "repeat":
        return bytes([spec["byte"]]) * spec["n"]
    if kind == "pattern":
        p = bytes.fromhex(spec["hex"])
        n = spec["n"]
        return (p * (n // len(p) + 1))[:n]
    if kind == "random":
        # Little-endian splitmix64 outputs, truncated to n bytes.
        r = SplitMix64(spec["seed"])
        n = spec["n"]
        words = (n + 7) // 8
        return b"".join(struct.pack("<Q", r.next()) for _ in range(words))[:n]
    if kind == "alpha":
        # n bytes drawn from [base, base+k) (mod 256).
        r = SplitMix64(spec["seed"])
        base, k = spec["base"], spec["k"]
        return bytes((base + r.next() % k) & 0xFF for _ in range(spec["n"]))
    if kind == "geom":
        # Geometrically distributed bytes: base + trailing zeros of a random
        # 64-bit word (64 for zero). Skewed enough to force libbzip2's
        # Huffman length limiting (maxLen 17).
        r = SplitMix64(spec["seed"])
        out = bytearray()
        for _ in range(spec["n"]):
            v = r.next()
            tz = (v & -v).bit_length() - 1 if v else 64
            out.append((spec["base"] + tz) & 0xFF)
        return bytes(out)
    if kind == "norun":
        # Random bytes where no byte equals its predecessor (so the initial
        # RLE never fires and block boundaries are predictable).
        r = SplitMix64(spec["seed"])
        out = bytearray()
        prev = -1
        for _ in range(spec["n"]):
            if prev < 0:
                c = r.next() % 256
            else:
                c = r.next() % 255
                if c >= prev:
                    c += 1
            out.append(c)
            prev = c
        return bytes(out)
    if kind == "runs":
        # Runs of random length 1..maxrun of bytes from [base, base+k)
        # (mod 256).
        r = SplitMix64(spec["seed"])
        base, k, maxrun, n = spec["base"], spec["k"], spec["maxrun"], spec["n"]
        out = bytearray()
        while len(out) < n:
            c = (base + r.next() % k) & 0xFF
            ln = 1 + r.next() % maxrun
            out += bytes([c]) * ln
        return bytes(out[:n])
    if kind == "text":
        r = SplitMix64(spec["seed"])
        n = spec["n"]
        out = bytearray()
        while len(out) < n:
            out += WORDS[r.next() % len(WORDS)].encode()
            s = r.next() % 16
            out += b".\n" if s == 0 else (b", " if s == 1 else b" ")
        return bytes(out[:n])
    if kind == "f64ramp":
        # float64(i) * scale, little-endian (one IEEE multiply: exact in Go).
        scale = spec["scale"]
        return (np.arange(spec["n"], dtype="<f8") * scale).astype("<f8").tobytes()
    if kind == "f64const":
        return np.full(spec["n"], spec["value"], dtype="<f8").tobytes()
    if kind == "f32walk":
        # float32(x)/8 for a random walk x with steps in [-3, 3].
        r = SplitMix64(spec["seed"])
        x = 0
        vals = []
        for _ in range(spec["n"]):
            x += r.next() % 7 - 3
            vals.append(x)
        return (np.array(vals, dtype="<f4") / np.float32(8)).astype("<f4").tobytes()
    if kind == "i32ramp":
        a = spec["start"] + spec["step"] * np.arange(spec["n"], dtype=np.int64)
        return a.astype("<i4").tobytes()
    if kind == "u16noise":
        r = SplitMix64(spec["seed"])
        base, amp = spec["base"], spec["amp"]
        return np.array(
            [base + r.next() % amp for _ in range(spec["n"])], dtype="<u2"
        ).tobytes()
    if kind == "tile":
        # The output of the "unit" spec repeated and truncated to n bytes.
        u = generate(spec["unit"])
        n = spec["n"]
        return (u * (n // len(u) + 1))[:n]
    if kind == "concat":
        return b"".join(generate(p) for p in spec["parts"])
    raise ValueError(f"unknown generator kind {kind!r}")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def vectors():
    """Yield (name, spec, level) for generator-described vectors."""
    # Empty input at every level.
    for lv in range(1, 10):
        yield f"empty-l{lv}", {"kind": "empty"}, lv

    # Single bytes.
    for b in (0x00, 0x61, 0xFF):
        yield f"byte-{b:02x}", {"kind": "repeat", "byte": b, "n": 1}, 9
    for lv in (1, 5):
        yield f"byte-61-l{lv}", {"kind": "repeat", "byte": 0x61, "n": 1}, lv

    # Runs of one byte value (initial RLE: 1-3 literal, 4..255 -> 4 + count).
    for n in (2, 3, 4, 5, 6, 7, 8, 9, 254, 255, 256, 257, 258, 259, 260,
              509, 510, 511, 512, 1000, 1020, 1021, 4000, 65536, 100000):
        yield f"run-61-{n}", {"kind": "repeat", "byte": 0x61, "n": n}, 9
    for n in (3, 4, 5, 255, 256, 1000):
        yield f"run-00-{n}", {"kind": "repeat", "byte": 0x00, "n": n}, 9
        yield f"run-ff-{n}", {"kind": "repeat", "byte": 0xFF, "n": n}, 9

    # Random bytes (main sort for >= 10000 symbols, fallback sort below).
    for n in (1, 2, 3, 10, 100, 255, 256, 1000, 4096, 9999, 10000, 10001,
              65536, 100000, 250000):
        yield f"random-{n}", {"kind": "random", "seed": n, "n": n}, 9
    # Sizes around the nGroups thresholds (nMTF < 200/600/1200/2400).
    for n in (150, 198, 199, 300, 598, 599, 900, 1198, 1199, 1500, 2000, 2398, 2399, 3000):
        yield f"random-{n}", {"kind": "random", "seed": n, "n": n}, 9
    yield "random-1M-l9", {"kind": "random", "seed": 7, "n": 1_000_000}, 9
    yield "random-1M-l1", {"kind": "random", "seed": 7, "n": 1_000_000}, 1
    yield "random-300k-l2", {"kind": "random", "seed": 8, "n": 300_000}, 2

    # Small alphabets.
    for k in (2, 3, 4, 16):
        for n in (5000, 50000):
            yield f"alpha{k}-{n}", {"kind": "alpha", "seed": k * 1000 + n,
                                    "n": n, "k": k, "base": 0x41}, 9
    yield "alpha2-500k-l1", {"kind": "alpha", "seed": 99, "n": 500_000,
                             "k": 2, "base": 0}, 1

    # Skewed symbol statistics (Huffman code length limiting).
    for n, lv in ((20000, 9), (200_000, 9), (900_000, 9), (300_000, 1)):
        yield f"geom-{n}-l{lv}", {"kind": "geom", "seed": n, "n": n, "base": 0x20}, lv
    yield "geom-alpha-mix", {"kind": "concat", "parts": [
        {"kind": "geom", "seed": 77, "n": 100_000, "base": 0},
        {"kind": "alpha", "seed": 78, "n": 100_000, "k": 200, "base": 50},
    ]}, 9

    # Random runs (exercise RLE, including runs spanning block boundaries).
    for lv, n, maxrun, k in ((9, 20000, 10, 4), (9, 300_000, 300, 8),
                             (1, 300_000, 300, 8), (1, 1_000_000, 600, 3),
                             (3, 800_000, 20, 256)):
        yield f"runs-{n}-{maxrun}-{k}-l{lv}", {
            "kind": "runs", "seed": n + maxrun, "n": n, "maxrun": maxrun,
            "k": k, "base": 0x30}, lv

    # Text.
    for n in (100, 1000, 20000, 200_000):
        yield f"text-{n}", {"kind": "text", "seed": n, "n": n}, 9
    yield "text-2M-l9", {"kind": "text", "seed": 2, "n": 2_000_000}, 9
    yield "text-2M-l1", {"kind": "text", "seed": 2, "n": 2_000_000}, 1

    # Numeric arrays, as zarr chunks look.
    yield "f64ramp-1000", {"kind": "f64ramp", "n": 1000, "scale": 0.001}, 9
    yield "f64ramp-128k-0.001", {"kind": "f64ramp", "n": 131072, "scale": 0.001}, 9
    yield "f64ramp-128k-0.5", {"kind": "f64ramp", "n": 131072, "scale": 0.5}, 9
    yield "f64ramp-128k-l1", {"kind": "f64ramp", "n": 131072, "scale": 0.1}, 1
    yield "f64const-1.0-1000", {"kind": "f64const", "n": 1000, "value": 1.0}, 9
    yield "f64const-1.0-100k", {"kind": "f64const", "n": 100000, "value": 1.0}, 9
    yield "f64const-pi-200k-l1", {"kind": "f64const", "n": 200000,
                                  "value": 3.141592653589793}, 1
    yield "f32walk-256k-l9", {"kind": "f32walk", "seed": 1, "n": 262144}, 9
    yield "f32walk-256k-l1", {"kind": "f32walk", "seed": 1, "n": 262144}, 1
    yield "f32walk-4096", {"kind": "f32walk", "seed": 2, "n": 4096}, 9
    yield "i32ramp-250k", {"kind": "i32ramp", "n": 250000, "start": -1000, "step": 3}, 9
    yield "u16noise-500k-l4", {"kind": "u16noise", "seed": 5, "n": 500000,
                               "base": 1000, "amp": 37}, 4

    # All-zero data (after RLE: repeating 00 00 00 00 fb units).
    yield "zeros-1MiB-l1", {"kind": "repeat", "byte": 0, "n": 1 << 20}, 1
    yield "zeros-1MiB-l9", {"kind": "repeat", "byte": 0, "n": 1 << 20}, 9
    yield "zeros-255x4000", {"kind": "repeat", "byte": 0, "n": 255 * 4000}, 9
    yield "zeros-10MiB-l1", {"kind": "repeat", "byte": 0, "n": 10 << 20}, 1
    yield "zeros-10MiB-l9", {"kind": "repeat", "byte": 0, "n": 10 << 20}, 9

    # Periodic data: identical rotations, so origPtr depends on how
    # libbzip2's sorts order ties (fallbackSort directly below 10000
    # symbols, mainSort exhausting its budget and falling back above).
    for n in (2, 3, 4, 100, 101, 9998, 9999, 10000, 10001, 20000, 100000,
              899981, 899982):
        yield f"ab-{n}", {"kind": "pattern", "hex": b"ab".hex(), "n": n}, 9
    yield "ab-2M-l9", {"kind": "pattern", "hex": b"ab".hex(), "n": 2_000_000}, 9
    yield "ab-1M-l1", {"kind": "pattern", "hex": b"ab".hex(), "n": 1_000_000}, 1
    for n in (3000, 3001, 30000, 30001, 300000):
        yield f"abc-{n}", {"kind": "pattern", "hex": b"abc".hex(), "n": n}, 9
    yield "abcd-40000", {"kind": "pattern", "hex": b"abcd".hex(), "n": 40000}, 9
    yield "abcdefg-70000", {"kind": "pattern", "hex": b"abcdefg".hex(), "n": 70000}, 9
    yield "aaaab-50000", {"kind": "pattern", "hex": b"aaaab".hex(), "n": 50000}, 9
    yield "aaaaab-6000", {"kind": "pattern", "hex": b"aaaaab".hex(), "n": 6000}, 9
    yield "a00-30000", {"kind": "pattern", "hex": "6100", "n": 30000}, 9
    p100 = {"kind": "random", "seed": 100, "n": 100}
    yield "period100-100000", {"kind": "tile", "unit": p100, "n": 100000}, 9
    yield "period100-99999", {"kind": "tile", "unit": p100, "n": 99999}, 9
    p1000 = {"kind": "text", "seed": 1000, "n": 1000}
    yield "period1000-500000", {"kind": "tile", "unit": p1000, "n": 500000}, 9
    yield "period1000-500000-l1", {"kind": "tile", "unit": p1000, "n": 500000}, 1
    p5000 = {"kind": "random", "seed": 5000, "n": 5000}
    yield "period5000-20000", {"kind": "tile", "unit": p5000, "n": 20000}, 9
    # Large periods with few repetitions: mainSort can then finish within its
    # budget (ties are mostly resolved through the quadrant cache), so the
    # tie order comes from mainSort rather than fallbackSort.
    for kind, p, m in (("random", 5000, 2), ("random", 5000, 3), ("random", 2500, 4),
                       ("random", 3000, 5), ("random", 10000, 2), ("random", 1000, 10),
                       ("random", 2000, 6), ("random", 12345, 3), ("random", 40000, 2),
                       ("text", 7000, 3), ("text", 20000, 2), ("text", 1500, 8),
                       ("alpha", 6000, 2), ("f32walk", 4000, 3)):
        if kind == "alpha":
            unit = {"kind": "alpha", "seed": p, "n": p, "k": 4, "base": 0x61}
        elif kind == "f32walk":
            unit = {"kind": "f32walk", "seed": p, "n": p // 4}
        else:
            unit = {"kind": kind, "seed": p + m, "n": p}
        for extra in (0, 1):
            yield f"period-{kind}{p}x{m}+{extra}", {
                "kind": "tile", "unit": unit, "n": p * m + extra}, 9
    yield "f64-1.0-pattern-800k", {"kind": "pattern", "hex": "000000000000f03f",
                                   "n": 800000}, 9

    # Block boundaries. With "norun" input every byte but the pending one is
    # in the block, so a block (nblockMAX = 100000*level - 19 symbols) fills
    # exactly after nblockMAX+1 input bytes.
    for n in (99980, 99981, 99982, 99983, 99984, 199964, 199965, 199966):
        yield f"norun-{n}-l1", {"kind": "norun", "seed": 1, "n": n}, 1
    for n in (899981, 899982, 899983):
        yield f"norun-{n}-l9", {"kind": "norun", "seed": 9, "n": n}, 9
    # A run crossing the level-1 block boundary.
    for pre in (99976, 99977, 99978, 99979, 99980, 99981):
        for run in (3, 4, 5, 6, 300):
            yield f"boundary-run-{pre}-{run}-l1", {"kind": "concat", "parts": [
                {"kind": "norun", "seed": 3, "n": pre},
                {"kind": "repeat", "byte": 0x5A, "n": run},
            ]}, 1

    # Every level, on input spanning several blocks at the low levels.
    mixed = {"kind": "concat", "parts": [
        {"kind": "text", "seed": 11, "n": 400_000},
        {"kind": "f32walk", "seed": 12, "n": 100_000},
        {"kind": "random", "seed": 13, "n": 300_000},
        {"kind": "repeat", "byte": 0, "n": 100_000},
        {"kind": "pattern", "hex": b"xy".hex(), "n": 50_000},
        {"kind": "runs", "seed": 14, "n": 200_000, "maxrun": 40, "k": 5, "base": 0x61},
    ]}
    for lv in range(1, 10):
        yield f"mixed-1.45M-l{lv}", mixed, lv
        yield f"hello-l{lv}", {"kind": "pattern", "hex": b"hello, world\n".hex(),
                               "n": 13 * 3}, lv


def golden_inputs():
    """Inputs stored only via their expected .bz2 output (numpy-derived)."""
    x = np.arange(8192)
    yield "np-sin-f32-8192", np.sin(x * 0.01).astype("<f4").tobytes(), 9
    yield "np-sin-f64-2048-l1", np.sin(x[:2048] * 0.05).astype("<f8").tobytes(), 1
    rng = np.random.default_rng(1234)
    yield "np-randn-f4-4096", rng.standard_normal(4096).astype("<f4").tobytes(), 9
    yield "np-linspace-f8-4000", np.linspace(-1, 1, 4000).astype("<f8").tobytes(), 9


def main():
    lib = ctypes.CDLL(__import__("_bz2").__file__)
    lib.BZ2_bzlibVersion.restype = ctypes.c_char_p
    libver = lib.BZ2_bzlibVersion().decode()

    os.makedirs(GOLDEN_DIR, exist_ok=True)
    out_vectors = []
    names = set()
    cache = {}
    for name, spec, level in vectors():
        assert name not in names, name
        names.add(name)
        key = json.dumps(spec, sort_keys=True)
        if key not in cache:
            cache = {key: generate(spec)}  # consecutive reuse only
        data = cache[key]
        comp = bz2.compress(data, level)
        assert bz2.decompress(comp) == data
        v = {
            "name": name,
            "gen": spec,
            "level": level,
            "in_len": len(data),
            "in_sha256": sha(data),
            "out_len": len(comp),
            "out_sha256": sha(comp),
        }
        if len(comp) <= INLINE_MAX:
            v["out_b64"] = base64.b64encode(comp).decode()
        out_vectors.append(v)

    for name, data, level in golden_inputs():
        assert name not in names, name
        names.add(name)
        comp = bz2.compress(data, level)
        fn = name + ".bz2"
        with open(os.path.join(GOLDEN_DIR, fn), "wb") as f:
            f.write(comp)
        out_vectors.append({
            "name": name,
            "gen": {"kind": "golden", "file": "golden/" + fn},
            "level": level,
            "in_len": len(data),
            "in_sha256": sha(data),
            "out_len": len(comp),
            "out_sha256": sha(comp),
        })

    doc = {
        "comment": "Generated by gen.py; do not edit. Expected outputs are "
                   "Python bz2.compress(data, level).",
        "python": sys.version.split()[0],
        "libbzip2": libver,
        "vectors": out_vectors,
    }
    with open(os.path.join(HERE, "vectors.json"), "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    total = sum(v["in_len"] for v in out_vectors)
    print(f"{len(out_vectors)} vectors, {total} input bytes, libbzip2 {libver}")


if __name__ == "__main__":
    main()
