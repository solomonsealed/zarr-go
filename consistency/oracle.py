"""zarr-python oracle for the Go consistency tests.

Reads JSON requests, one per line, from stdin and answers each with one
JSON line on stdout: {"ok": ...result...} or {"error": "...", "type": "..."}.
Binary data travels base64-encoded; array elements in the canonical
little-endian layout the Go package uses (see consistency/README.md).

Run by the Go tests (ZARR_PYTHON=.venv/bin/python go test ./consistency);
it can also be driven by hand:

    echo '{"op": "versions"}' | .venv/bin/python consistency/oracle.py
"""

import base64
import hashlib
import json
import os
import sys
import traceback
import warnings

import numpy as np

warnings.simplefilter("ignore")

import numcodecs  # noqa: E402
import zarr  # noqa: E402
from zarr.core.dtype import get_data_type_from_json  # noqa: E402
from zarr.storage import LocalStore  # noqa: E402

zarr.config.set({"array.rectilinear_chunks": True})


def b64(b: bytes) -> str:
    return base64.b64encode(bytes(b)).decode()


def unb64(s: str) -> bytes:
    return base64.b64decode(s)


# ---------------------------------------------------------------------------
# Data types and the wire format


def le(dt: np.dtype) -> np.dtype:
    """The little-endian (canonical) version of a numpy dtype."""
    if dt.fields is not None:
        return np.dtype([(name, le(dt.fields[name][0])) for name in dt.names])
    if dt.kind in "biufcmMU" and dt.itemsize > 1:
        return dt.newbyteorder("<")
    return dt


def is_variable(dt: np.dtype) -> bool:
    return dt.kind in "OT"


def to_wire(arr, as_bytes=None) -> dict:
    arr = np.asarray(arr)
    out = {"shape": list(arr.shape)}
    if is_variable(arr.dtype):
        flat = [x[()] if isinstance(x, np.ndarray) else x for x in arr.reshape(-1)]
        if as_bytes is None:
            as_bytes = any(isinstance(x, (bytes, bytearray, memoryview)) for x in flat)
        if as_bytes:
            out["bytes"] = [b64(bytes(x)) for x in flat]
        else:
            out["strings"] = [str(x) for x in flat]
        return out
    a = np.ascontiguousarray(arr.astype(le(arr.dtype), copy=False))
    if a.dtype.kind in "mM":
        a = a.view("<i8")
    out["data"] = b64(a.tobytes())
    return out


def from_wire(w: dict, dt: np.dtype):
    shape = tuple(w["shape"])
    if "strings" in w:
        if dt.kind == "T":
            return np.array(w["strings"], dtype=dt).reshape(shape)
        return np.array(w["strings"], dtype=object).reshape(shape)
    if "bytes" in w:
        a = np.empty(len(w["bytes"]), dtype=object)
        for i, x in enumerate(w["bytes"]):
            a[i] = unb64(x)
        return a.reshape(shape)
    if "data" not in w:  # an empty list of strings or bytes
        if dt.kind == "T":
            return np.array([], dtype=dt).reshape(shape)
        if dt.kind == "O":
            return np.empty(shape, dtype=object)
        raise ValueError("no data")
    raw = unb64(w["data"])
    a = np.frombuffer(raw, dtype=le(dt)).reshape(shape)
    return a.astype(dt)


def fill_scalar(zdtype, fill_json, fmt):
    return zdtype.from_json_scalar(fill_json, zarr_format=fmt)


# ---------------------------------------------------------------------------
# Selections


def dim_item(x):
    if "index" in x:
        return int(x["index"])
    if "slice" in x:
        s = x["slice"]
        return slice(s[0], s[1], s[2])
    if "ints" in x:
        return np.array(x["ints"], dtype=np.int64)
    if "bools" in x:
        return np.array(x["bools"], dtype=bool)
    if "ellipsis" in x:
        return Ellipsis
    raise ValueError(f"bad selector {x}")


def selection(sel: dict):
    kind = sel["kind"]
    if kind in ("orthogonal", "basic", "block"):
        return kind, tuple(dim_item(x) for x in sel.get("items") or [])
    if kind == "coordinate":
        return kind, tuple(np.array(c, dtype=np.int64) for c in sel["coords"])
    if kind == "mask":
        return kind, from_wire(sel["mask"], np.dtype(bool))
    raise ValueError(f"bad selection kind {kind}")


def get(arr, sel):
    kind, s = selection(sel)
    if kind == "orthogonal":
        res = arr.get_orthogonal_selection(s)
    elif kind == "basic":
        res = arr.get_basic_selection(s)
    elif kind == "block":
        res = arr.get_block_selection(s)
    elif kind == "coordinate":
        res = arr.get_coordinate_selection(s)
    else:
        res = arr.get_mask_selection(s)
    # Scalars (basic selections of one element) lose the dtype's padding.
    return np.asarray(res, dtype=arr.dtype)


def put(arr, sel, value):
    kind, s = selection(sel)
    if np.ndim(value) == 0:
        value = value[()]
    if kind == "orthogonal":
        arr.set_orthogonal_selection(s, value)
    elif kind == "basic":
        arr.set_basic_selection(s, value)
    elif kind == "block":
        arr.set_block_selection(s, value)
    elif kind == "coordinate":
        arr.set_coordinate_selection(s, value)
    else:
        arr.set_mask_selection(s, value)


# ---------------------------------------------------------------------------
# Operations


def open_array(req):
    arr = zarr.open_array(LocalStore(req["store"]), path=req.get("path", ""), mode="r+")
    # open_array ignores its config argument in zarr 3.4; with_config works.
    return arr.with_config({"write_empty_chunks": bool(req.get("write_empty_chunks", False))})


def op_versions(req):
    import ml_dtypes

    try:
        import cast_value_rs  # noqa: F401
        from importlib.metadata import version

        cv = version("cast-value-rs")
    except Exception:
        cv = None
    return {
        "zarr": zarr.__version__,
        "numcodecs": numcodecs.__version__,
        "numpy": np.__version__,
        "ml_dtypes": ml_dtypes.__version__,
        "cast_value_rs": cv,
        "python": sys.version.split()[0],
    }


def op_create(req):
    c = req["case"]
    fmt = c.get("zarr_format", 3)
    store = LocalStore(req["store"])
    kw = {}
    if fmt == 3:
        zdtype = get_data_type_from_json(c["data_type"], zarr_format=3)
        kw["dtype"] = zdtype
        if "fill_value" in c:
            kw["fill_value"] = fill_scalar(zdtype, c["fill_value"], 3)
        if c.get("filters") is not None:
            kw["filters"] = c["filters"]
        if c.get("serializer") is not None:
            kw["serializer"] = c["serializer"]
        if c.get("compressors") is not None:
            kw["compressors"] = c["compressors"]
        if c.get("shards") is not None:
            kw["shards"] = {"shape": tuple(c["shards"]), "index_location": c.get("index_location", "end")}
        if c.get("dimension_names") is not None:
            kw["dimension_names"] = c["dimension_names"]
        if c.get("chunk_key_encoding") is not None:
            kw["chunk_key_encoding"] = c["chunk_key_encoding"]
    else:
        np_dtype = np.dtype(c["dtype"]) if not isinstance(c["dtype"], list) else np.dtype([tuple(f) for f in c["dtype"]])
        if c["dtype"] == "|O" or c.get("string_type"):
            kind = c.get("string_type", "string")
            from zarr.core.dtype import VariableLengthBytes, VariableLengthUTF8

            kw["dtype"] = VariableLengthUTF8() if kind == "string" else VariableLengthBytes()
            zdtype = kw["dtype"]
        else:
            kw["dtype"] = np_dtype
            zdtype = get_data_type_from_json({"name": c["dtype"], "object_codec_id": None}, zarr_format=2)
        if "fill_value" in c:
            fv = c["fill_value"]
            kw["fill_value"] = None if fv is None else fill_scalar(zdtype, fv, 2)
        if c.get("filters") is not None:
            kw["filters"] = [numcodecs.get_codec(dict(f)) for f in c["filters"]]
        if c.get("compressors") is not None:
            kw["compressors"] = [numcodecs.get_codec(dict(f)) for f in c["compressors"]] or None
        if c.get("order"):
            kw["order"] = c["order"]
        if c.get("chunk_key_encoding") is not None:
            kw["chunk_key_encoding"] = c["chunk_key_encoding"]
    chunks = c.get("chunk_edges") or c["chunks"]
    if c.get("chunk_edges"):
        chunks = [e if e is not None else c["chunks"][i] for i, e in enumerate(c["chunk_edges"])]
    zarr.create_array(
        store,
        name=req.get("path") or None,
        shape=tuple(c["shape"]),
        chunks=chunks if c.get("chunk_edges") else tuple(chunks),
        zarr_format=fmt,
        attributes=c.get("attributes"),
        config={"write_empty_chunks": bool(c.get("write_empty_chunks", False))},
        **kw,
    )
    return {}


def op_write(req):
    arr = open_array(req)
    put(arr, req["selection"], from_wire(req["data"], arr.dtype))
    return {}


def is_bytes_array(arr) -> bool:
    from zarr.core.dtype import VariableLengthBytes

    return isinstance(arr.metadata.dtype if hasattr(arr.metadata, "dtype") else arr.metadata.data_type, VariableLengthBytes)


def op_read(req):
    arr = open_array(req)
    return to_wire(get(arr, req["selection"]), is_bytes_array(arr))


def op_read_all(req):
    arr = zarr.open_array(LocalStore(req["store"]), path=req.get("path", ""), mode="r")
    return to_wire(np.asarray(arr[...], dtype=arr.dtype), is_bytes_array(arr))  # 0-d reads are scalars


def op_resize(req):
    arr = open_array(req)
    arr.resize(tuple(req["shape"]))
    return {}


def op_append(req):
    arr = open_array(req)
    arr.append(from_wire(req["data"], arr.dtype), axis=req.get("axis", 0))
    return {}


def op_snapshot(req):
    root = req["store"]
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            key = os.path.relpath(p, root).replace(os.sep, "/")
            with open(p, "rb") as fh:
                out[key] = b64(fh.read())
    return out


def op_create_group(req):
    zarr.create_group(
        LocalStore(req["store"]),
        path=req.get("path") or None,
        zarr_format=req.get("zarr_format", 3),
        attributes=req.get("attributes"),
        overwrite=True,
    )
    return {}


def op_consolidate(req):
    zarr.consolidate_metadata(LocalStore(req["store"]), path=req.get("path") or None)
    return {}


def op_members(req):
    g = zarr.open_group(LocalStore(req["store"]), path=req.get("path") or None, mode="r", use_consolidated=req.get("use_consolidated"))
    out = {}
    for name, node in g.members(max_depth=None):
        out[name] = {"type": "array" if isinstance(node, zarr.Array) else "group", "attributes": dict(node.attrs)}
    return {"attributes": dict(g.attrs), "members": out}


def numcodec(c):
    if "id" in c:
        return numcodecs.get_codec(dict(c))
    name = c["name"].removeprefix("numcodecs.")
    return numcodecs.get_codec({"id": name, **(c.get("configuration") or {})})


def op_codec_encode(req):
    codec = numcodec(req["codec"])
    data = unb64(req["data"])
    if req.get("dtype"):
        data = np.frombuffer(data, dtype=np.dtype(req["dtype"]))
    out = codec.encode(data)
    return b64(np.asarray(out).tobytes() if not isinstance(out, (bytes, bytearray)) else out)


def op_codec_decode(req):
    codec = numcodec(req["codec"])
    out = codec.decode(unb64(req["data"]))
    if out is None:
        raise RuntimeError("codec returned no data")
    return b64(np.asarray(out).tobytes() if not isinstance(out, (bytes, bytearray)) else out)


def op_ml_dtypes(req):
    """Convert float64 values to an ml_dtypes type and back (reference)."""
    import ml_dtypes

    dt = np.dtype(getattr(ml_dtypes, req["dtype"])) if hasattr(ml_dtypes, req["dtype"]) else np.dtype(req["dtype"])
    vals = np.frombuffer(unb64(req["values"]), dtype="<f8")
    enc = vals.astype(dt)
    raw = enc.view(np.uint8 if dt.itemsize == 1 else np.uint16)
    dec = enc.astype(np.float64)
    return {"bits": b64(raw.astype("<u2" if dt.itemsize == 2 else "u1").tobytes()), "back": b64(dec.astype("<f8").tobytes())}


def op_ml_dtypes_decode(req):
    """Decode bit patterns of an ml_dtypes type to float64 (reference)."""
    import ml_dtypes

    name = req["dtype"]
    dt = np.dtype(getattr(ml_dtypes, name)) if hasattr(ml_dtypes, name) else np.dtype(name)
    raw = np.frombuffer(unb64(req["bits"]), dtype="<u2" if dt.itemsize == 2 else "u1")
    vals = raw.view(dt) if dt.itemsize == raw.dtype.itemsize else raw.astype(dt)
    return {"bits": "", "back": b64(vals.astype(np.float64).astype("<f8").tobytes())}


def ts_open(req):
    import tensorstore as ts

    return ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": os.path.join(req["store"], req.get("path", ""))}}, open=True).result()


def op_ts_write(req):
    """Write a whole array with TensorStore (extension data types)."""
    t = ts_open(req)
    dt = t.dtype.numpy_dtype
    w = req["data"]
    a = np.frombuffer(unb64(w["data"]), dtype=dt).reshape(tuple(w["shape"]))
    t.write(a).result()
    return {}


def op_ts_read(req):
    t = ts_open(req)
    a = np.ascontiguousarray(t.read().result())
    return {"shape": list(a.shape), "data": b64(a.tobytes())}


def op_known_issue(req):
    """Reproduce a zarr-python/numcodecs issue; returns whether it still occurs."""
    from zarr.core.dtype import VariableLengthBytes

    name, store = req["name"], LocalStore(req["store"])
    if name == "vlen_bytes_single_write":
        a = zarr.create_array(store, name="a", shape=(3,), chunks=(3,), dtype=VariableLengthBytes())
        try:
            a[0] = b"x"
            return {"occurs": False}
        except TypeError as e:
            return {"occurs": True, "detail": str(e)}
    if name == "vlen_bytes_single_read":
        a = zarr.create_array(store, name="a", shape=(2,), chunks=(2,), dtype=VariableLengthBytes())
        a[:] = np.array([b"a\x00", b"b"], dtype=object)
        v = a.get_basic_selection(0).item()
        return {"occurs": v != b"a\x00", "detail": repr(v)}
    if name == "quantize_astype":
        from zarr.codecs.numcodecs import Quantize

        a = zarr.create_array(store, name="a", shape=(4,), chunks=(4,), dtype="f4", filters=[Quantize(digits=1, dtype="<f4", astype="<f2")], compressors=None)
        a[:] = np.arange(4, dtype="f4")
        try:
            a[:]
            return {"occurs": False}
        except ValueError as e:
            return {"occurs": True, "detail": str(e)}
    if name == "bitround_f_order":
        a = zarr.create_array(store, name="a", shape=(4, 3), chunks=(4, 3), dtype="<f4", zarr_format=2, order="F", filters=[numcodecs.BitRound(keepbits=10)], compressors=None)
        x = np.arange(12, dtype="<f4").reshape(4, 3)
        a[:] = x
        return {"occurs": not np.array_equal(a[:], x), "detail": str(a[:].ravel().tolist())}
    if name == "open_array_config":
        zarr.create_array(store, name="a", shape=(4,), chunks=(2,), dtype="i1")
        a = zarr.open_array(store, path="a", config={"write_empty_chunks": True})
        return {"occurs": not a.config.write_empty_chunks}
    if name == "ellipsis_with_array":
        a = zarr.create_array(store, name="a", shape=(3, 3), chunks=(3, 3), dtype="i1")
        try:
            a.get_orthogonal_selection((np.array([0, 2]), Ellipsis))
            return {"occurs": False}
        except ValueError as e:
            return {"occurs": True, "detail": str(e)}
    if name == "empty_buffers":
        failing = []
        for c in [numcodecs.LZ4(), numcodecs.Zstd(), numcodecs.Blosc()]:
            try:
                c.decode(c.encode(b""))
            except Exception:
                failing.append(type(c).__name__)
        try:
            numcodecs.Fletcher32().encode(b"")
        except Exception:
            failing.append("Fletcher32")
        return {"occurs": bool(failing), "detail": ", ".join(failing)}
    raise ValueError(f"unknown issue {name}")


OPS = {k[3:]: v for k, v in globals().items() if k.startswith("op_")}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("op") == "exit":
                break
            res = {"ok": OPS[req["op"]](req)}
        except Exception as e:  # report every failure to the caller
            res = {"error": f"{type(e).__name__}: {e}", "type": type(e).__name__, "trace": traceback.format_exc(limit=4)}
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
