"""W5k (owner 2026-11-24, issue #894): ``read_public_windows`` takes the
series from a ``reader(row)`` (the eval host's Hippius mirror) instead of a
local mirror, and calls it ONLY for the drawn series; the windows are the
ones the local read builds. Contract: README.md "Hippius mirror reader".
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from .f3_fixture import fx

GIFT, RMISC = "gift_eval_pretrain", "rmisc"


def _pub():
    return importlib.import_module("cascade.validator.lane_d_public")


@pytest.mark.parametrize("source", [GIFT, RMISC])
def test_a_reader_replaces_the_local_mirror_for_the_drawn_series_only(tmp_path, source):
    pytest.importorskip("pyarrow")
    pub = _pub()
    from cascade.eval.lane_d_public_index import iter_index, read_series

    b = fx.build(tmp_path, source)
    d = pub.draw_public(b["index"], source=source, boundary_block_hash=fx.BLOCK_HASH,
                        horizon=64, n_windows=6)
    asked = []

    def reader(row):
        asked.append(row["ordinal"])
        return read_series(b["root"], row, source=source)

    ws, dr = pub.read_public_windows(d, index_dir=b["index"], reader=reader,
                                     block_hash=fx.BLOCK_HASH, max_channels=32)
    lw, ld = pub.read_public_windows(d, index_dir=b["index"], data_root=b["root"],
                                     block_hash=fx.BLOCK_HASH, max_channels=32)
    assert sorted(set(asked)) == sorted({w["ordinal"] for w in d["windows"]})
    assert len(set(asked)) < sum(1 for _ in iter_index(b["index"]))
    assert dr == ld and [w["window_id"] for w in ws] == [w["window_id"] for w in lw]
    for x, y in zip(ws, lw, strict=True):
        assert np.array_equal(x["history"], y["history"]) and np.array_equal(x["target"],
                                                                             y["target"])


def test_neither_a_reader_nor_a_data_root_is_refused(tmp_path):
    pub = _pub()
    with pytest.raises(ValueError, match="data_root or a reader"):
        pub.read_public_windows({"source": GIFT, "windows": []}, index_dir=tmp_path,
                                block_hash=None, max_channels=1)
