"""W8c (plan rev6.18 A3, F4): the live settlement's public half is wired.

At ``[lane_d] public_weight > 0`` the inputs source draws the public sources
from the box's ``public_indexes`` and the boundary block hash, asks the eval
host for every needed digest's per-window public scores (the host reads the
drawn series from its own public mirror: ``public`` request), and returns
them as ``LaneDInputs.public_indexes`` / ``public_scores``, which W8b passes
to ``settle_lane_d``. Fail closed: no indexes, a host failure or drops that
differ between digests are ``public_unavailable``; a miner's own fault is
NaN on every public window (the settlement closes it). Contract: README.md
"Public half (rev6.18 F4)".
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.accept.w7a import helpers as w7a
from tests.accept.w8b import helpers as w8b

from . import helpers as h


def _pub():
    from tests.accept.w7a import pub

    assert hasattr(pub, "index_only"), (
        "F4 (rev6.18 A3): tests/accept/w7a/pub.py has no index-only public fixture")
    return pub


def _codes():
    m = h.inputs_mod()
    assert "public_unavailable" in m.INPUTS_ERROR_CODES, (
        "F4: the inputs source has no public_unavailable code (the public half is unwired)")
    return m


class PublicHost(h.EvalHost):
    """The fake eval host plus the F4 ``public_scores`` request."""

    def __init__(self, fx, indexes, *, fault=None, drop_for=None):
        super().__init__(fx)
        self.indexes = indexes
        self.public_calls: list[dict] = []
        self.fault = fault
        self.drop_for = drop_for

    def public_scores(self, digest, *, context, horizon, draws, block_hash, max_channels):
        self.public_calls.append({"digest": digest, "context": context, "horizon": horizon,
                                  "draws": draws, "block_hash": block_hash,
                                  "max_channels": max_channels})
        if digest == self.fault:
            raise h.forecasts_mod().LaneDForecastError("forecast_malformed", "w8c public")
        dropped = {}
        if self.drop_for == digest:
            d0 = draws[0]
            dropped = {d0["source"]: [{"window_id": d0["windows"][0]["window_id"],
                                       "reason": "public_no_scale"}]}
        got = _pub().public_scores(self.indexes, block_hash, [digest], horizon=horizon,
                                   dropped=dropped)
        return {"scores": got["scores"][digest], "dropped": got["dropped"]}


def _block_hash(block: int) -> str:
    """The boundary hash W8b's step reads too (one chain)."""
    return w8b.block_hash(block)


def _rig(tmp_path, **host):
    _codes()
    pub = _pub()
    rig = h.Rig(tmp_path)
    rig.cfg = replace(rig.cfg, lane_d=w7a.lane_d_cfg(public_weight=0.5, window_ratio_floor=0.5))
    rig.reader.block_hash = _block_hash
    indexes = pub.index_only(tmp_path / "public")
    rig.eval_host = PublicHost(rig.fx, indexes, **host)
    rig.source = rig.build(public_indexes=indexes)
    return rig, indexes


def test_the_source_returns_the_public_half_for_every_needed_digest(tmp_path):
    from cascade.validator.lane_d_public import draw_public

    rig, indexes = _rig(tmp_path)
    got = rig.source(h.B1)
    assert got.public_indexes is not None and sorted(got.public_indexes) == sorted(indexes)
    asked = {c["digest"] for c in rig.eval_host.public_calls}
    assert asked == set(got.forecasts), "every needed digest is scored on the public half"
    bh = _block_hash(h.B1)
    want = [draw_public(indexes[s], source=s, boundary_block_hash=bh, horizon=h.HORIZON)
            for s in sorted(indexes)]
    for c in rig.eval_host.public_calls:
        assert c["block_hash"] == bh and c["draws"] == want, "the host gets the box's draw"
    for dg, per in got.public_scores["scores"].items():
        n = sum(len(per[s]) for s in per)
        assert n == sum(len(d["windows"]) for d in want), dg


def test_the_settlement_runs_on_the_public_half(tmp_path):
    """W8b's step passes the public half through to settle_lane_d."""
    rig, _ = _rig(tmp_path)
    step = rig.step()
    s = step(h.B1)
    assert s is not None and "public" in s.report, "the settled report has no public half"
    assert s.report["params"]["public_weight"] == 0.5
    assert all("public" in v["halves"] for v in s.report["verdicts"])


def test_no_public_indexes_is_public_unavailable(tmp_path):
    m = _codes()
    rig, _ = _rig(tmp_path)
    rig.source = rig.build()
    with pytest.raises(m.LaneDInputsError) as e:
        rig.source(h.B1)
    h.assert_inputs_error(e.value, "public_unavailable")


def test_a_miners_public_fault_is_nan_on_every_public_window(tmp_path):
    rig0 = h.Rig(tmp_path / "probe")
    x = rig0.fx["x"]
    rig, _ = _rig(tmp_path / "r", fault=x)
    got = rig.source(h.B1)
    rows = [r for per in got.public_scores["scores"][x].values() for r in per]
    assert rows and all(np.isnan(r["crps"]) and np.isnan(r["mase"]) for r in rows)


def test_a_bars_public_fault_fails_closed(tmp_path):
    m = _codes()
    rig0 = h.Rig(tmp_path / "probe")
    rig, _ = _rig(tmp_path / "r", fault=rig0.fx["house"]["digest"])
    with pytest.raises(m.LaneDInputsError) as e:
        rig.source(h.B1)
    h.assert_inputs_error(e.value, "public_unavailable")


def test_drops_that_differ_between_digests_fail_closed(tmp_path):
    m = _codes()
    rig0 = h.Rig(tmp_path / "probe")
    rig, _ = _rig(tmp_path / "r", drop_for=rig0.fx["x"])
    with pytest.raises(m.LaneDInputsError) as e:
        rig.source(h.B1)
    h.assert_inputs_error(e.value, "public_unavailable")


def test_the_eval_host_scores_the_drawn_public_windows(tmp_path, monkeypatch):
    """The eval host's end: it reads the drawn series from its mirror (here a
    stand-in for the pyarrow reader), forecasts them at the entry's context
    and scores each window with the lane D per-window API."""
    fc = h.forecasts_mod()
    assert hasattr(fc, "public_scores_entry"), (
        "F4: cascade.eval.lane_d_forecasts has no public_scores_entry (the host's public end)")
    from cascade.eval.lane_d_scorer import score_window
    from cascade.validator import lane_d_public

    rng = np.random.default_rng(5)
    wins = [{"window_id": f"rmisc:s{i}:200:h8", "series_id": f"s{i}", "source": "rmisc",
             "family": "fam", "freq": "H", "seasonal_period": 24,
             "history": rng.random(300) + 1.0, "target": rng.random(8) + 1.0, "channels": None}
            for i in range(3)]

    def fake_read(draw, *, index_dir, data_root, block_hash, max_channels):
        assert str(index_dir).endswith("rmisc/index") and str(data_root).endswith("rmisc/data")
        return list(wins), [{"window_id": "rmisc:gone:0:h8", "reason": "public_short_history"}]

    monkeypatch.setattr(lane_d_public, "read_public_windows", fake_read)
    q = np.sort(rng.random((3, 8, 9)) + 1.0, axis=-1)
    seen = {}

    def forecast(ckpt, histories, *, context, horizon, contexts, device):
        seen["lens"] = [np.shape(x)[-1] for x in histories]
        return q

    class Cache:
        def get(self, digest):
            return tmp_path

    got = fc.public_scores_entry(Cache(), "d" * 64, [{"source": "rmisc", "windows": []}],
                                 public_root=tmp_path, context=128, horizon=8,
                                 contexts=(128,), block_hash=None, max_channels=32,
                                 forecast=forecast)
    assert seen["lens"] == [128, 128, 128], "each history is cut to the context"
    assert got["dropped"] == {"rmisc": [{"window_id": "rmisc:gone:0:h8",
                                         "reason": "public_short_history"}]}
    rows = got["scores"]["rmisc"]
    assert [r["window_id"] for r in rows] == [w["window_id"] for w in wins]
    for r, w, qq in zip(rows, wins, q, strict=True):
        c, m_ = score_window(qq, w["target"], w["history"], 24)
        assert np.isclose(r["crps"], c) and np.isclose(r["mase"], m_)
