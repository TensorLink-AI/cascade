"""W6p (F3): the GiftEvalPretrain ``target`` shape is measured per record
batch. ``combine_chunks()`` on a whole large multivariate column
(``cmip6_1850``: C=53, 8192 series, L=7300, 9 batches) overflows the int32
list offsets (found building the mirror, 2026-11-25); the per-chunk measure
must give the SAME arrays, so the index bytes validators bind do not move.
Contract: README.md ("Hippius mirror").
"""

from __future__ import annotations

import gzip
import importlib

import numpy as np
import pytest

from . import fixtures as fx

pa = pytest.importorskip("pyarrow")


def _idx():
    return importlib.import_module("cascade.eval.lane_d_public_index")


def _whole_column_shape(col):
    """The measure before the fix (whole-column ``combine_chunks()``), kept
    verbatim as the reference the per-chunk measure must reproduce."""
    import pyarrow.compute as pc

    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    outer = pc.list_value_length(col).to_numpy(zero_copy_only=False).astype(np.int64)
    if not pa.types.is_list(col.type.value_type) and not pa.types.is_large_list(
            col.type.value_type):
        return np.ones_like(outer), outer
    inner = pc.list_value_length(pc.list_flatten(col)).to_numpy(
        zero_copy_only=False).astype(np.int64)
    lengths = np.zeros_like(outer)
    starts = np.concatenate([[0], np.cumsum(outer)[:-1]]).astype(np.int64)
    for i, (s, c) in enumerate(zip(starts, outer, strict=True)):
        if c == 0:
            continue
        seg = inner[s:s + c]
        if seg.min() != seg.max():
            raise ValueError(f"row {i}: channels of unequal length {sorted(set(seg.tolist()))}")
        lengths[i] = seg[0]
    return outer, lengths


def _null_chunk(n_series: int, channels: int, length: int):
    """``fixed_size_list<list<null>>[C]`` (the real folder's outer type) of
    ``n_series`` rows of ``C`` channels of ``length``: the inner values are a
    NullArray (no buffers), so a column of > 2**31 inner values costs only
    its offsets (a few MB)."""
    off = pa.array([k * length for k in range(n_series * channels + 1)], type=pa.int32())
    inner = pa.ListArray.from_arrays(off, pa.nulls(n_series * channels * length))
    return pa.FixedSizeListArray.from_arrays(inner, channels)


def test_list_shape_survives_int32_offset_overflow():
    C, L = 53, 7300
    n = 2**31 // (C * L) // 2 + 1          # > 2**30 inner values per chunk
    col = pa.chunked_array([_null_chunk(n, C, L), _null_chunk(n, C, L),
                            _null_chunk(3, C, 10)])
    assert sum(len(c.values.values) for c in col.chunks) > 2**31 - 1
    with pytest.raises(pa.ArrowInvalid, match="offset overflow"):
        col.combine_chunks()                 # the pre-fix failure, reproduced
    try:
        ch, ln = _idx()._list_shape(col)
    except pa.ArrowInvalid as e:
        raise AssertionError(f"_list_shape concatenated the whole column: {e}") from e
    assert ch.dtype == ln.dtype == np.int64
    assert ch.tolist() == [C] * (2 * n + 3)
    assert ln.tolist() == [L] * (2 * n) + [10] * 3


def _chunked(values, ty, sizes):
    arr = pa.array(values, type=ty)
    out, at = [], 0
    for s in sizes:
        out.append(arr.slice(at, s))
        at += s
    return pa.chunked_array(out, type=ty)


@pytest.mark.parametrize("ty,values", [
    (pa.list_(pa.float32()), [[1.0] * k for k in (3, 0, 5, 2, 7, 1, 4)]),
    (pa.list_(pa.list_(pa.float32())),
     [[[1.0] * 4] * 3, [], [[2.0] * 6] * 2, [[0.5] * 1] * 5, [[1.0] * 9], [[3.0] * 2] * 2,
      [[1.0] * 3] * 4]),
    (pa.list_(pa.list_(pa.float32()), 2),
     [[[1.0] * 4] * 2, [[2.0] * 6] * 2, [[0.5]] * 2, [[1.0] * 9] * 2, [[3.0] * 2] * 2,
      [[1.0] * 3] * 2, [[]] * 2]),
])
@pytest.mark.parametrize("sizes", [(7,), (3, 4), (1, 0, 2, 4), (0, 7, 0)])
def test_list_shape_matches_the_whole_column_measure(ty, values, sizes):
    idx = _idx()
    col = _chunked(values, ty, sizes)
    want = _whole_column_shape(col)
    got = idx._list_shape(col)
    for w, g in zip(want, got, strict=True):
        assert g.dtype == w.dtype and g.tolist() == w.tolist()
    flat = idx._list_shape(col.combine_chunks())
    for w, g in zip(want, flat, strict=True):
        assert g.tolist() == w.tolist()


def test_list_shape_reports_the_column_row_of_unequal_channels():
    ty = pa.list_(pa.list_(pa.float32()))
    col = _chunked([[[1.0] * 2] * 2, [[1.0] * 2] * 2, [[1.0], [1.0, 2.0]]], ty, (2, 1))
    with pytest.raises(ValueError, match=r"row 2: channels of unequal length \[1, 2\]"):
        _idx()._list_shape(col)


def test_gift_index_bytes_match_the_whole_column_build(tmp_path, monkeypatch):
    """The fixture mirror (arrow files of several record batches, uni- and
    multivariate) indexes to the same bytes with the per-chunk measure as
    with the pre-fix one."""
    idx = _idx()
    root, ds = fx.gift_mirror(tmp_path)
    t = idx._open_arrow(root / "multi_traffic" / "data-00000-of-00001.arrow")
    assert t.column("target").num_chunks > 1
    new = idx.build_index(root, source=fx.GIFT, datasets=ds, out_dir=tmp_path / "new")
    monkeypatch.setattr(idx, "_list_shape", _whole_column_shape)
    old = idx.build_index(root, source=fx.GIFT, datasets=ds, out_dir=tmp_path / "old")
    assert new == old
    for name in (idx.INDEX_FILE, idx.MANIFEST_FILE):
        assert (tmp_path / "new" / name).read_bytes() == (tmp_path / "old" / name).read_bytes()
    lines = gzip.decompress((tmp_path / "new" / idx.INDEX_FILE).read_bytes()).splitlines()
    assert len(lines) == new["n_series"] == 37
