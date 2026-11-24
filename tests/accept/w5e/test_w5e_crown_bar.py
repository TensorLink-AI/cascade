"""rev6.23 (owner 2026-11-24): the lane D crown bar decays with KING_D's
tenure. A first verdict clears iff its combined LCB >= the bar in force at
the settlement: ``d_margin_start`` at tenure 0, falling linearly to
``d_margin`` over ``d_margin_warmup_settlements`` lane D settlements (the
house holding the crown counts as tenure). The report records the tenure and
the bar applied; the state carries the tenure into the next settlement.

Every state comes from real settlements (the W5e fixtures)."""

from __future__ import annotations

import pytest

from .helpers import (
    T_X,
    boundary,
    entry,
    house,
    lane_d_cfg,
    model_forecasts,
    read_state,
    settle,
    settle_crowning,
    snapshot,
    uids,
    verdict_of,
)


def _need_rev623() -> None:
    """rev6.23 is built: ``LaneDConfig`` has ``court_weights`` and
    ``crown_bar`` (on main both are absent: an assertion, the right failure)."""
    from cascade.shared.config.lane_d import LaneDConfig

    assert "court_weights" in LaneDConfig.__dataclass_fields__ and \
        callable(getattr(LaneDConfig, "crown_bar", None)), \
        "plan rev6.23 is not built: LaneDConfig has no court_weights / crown_bar"


def _fc(recs, models: dict) -> dict:
    return {dg: model_forecasts(recs, q, name) for dg, (q, name) in models.items()}


def _x_at(tmp_path, n_before: int, cfg):
    """``n_before`` settlements with no entry (the house holds KING_D), then
    X judged at the next one; returns ``(x, state path, X's settlement)``."""
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1)
    fc = _fc(recs1, {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")})
    for n in range(1, n_before + 1):
        s = settle(cfg, boundary(n), pool1, [], h0, {}, {}, path)
        assert s.report is not None and s.report["judged"] == []
    s = settle_crowning(cfg, boundary(n_before + 1), pool1, [x], h0, fc, uids(x), path,
                        candidates=[x["digest"]], references=[h0["digest"]],
                        tag=f"cb{n_before}")
    return x, path, s


def test_the_first_settlement_applies_d_margin_start_at_tenure_0(tmp_path):
    _need_rev623()
    cfg = lane_d_cfg()
    assert (cfg.d_margin_start, cfg.d_margin, cfg.d_margin_warmup_settlements) == \
        (0.03, 0.01, 32), "the rev6.23 defaults (owner 2026-11-24)"
    _, _, s = _x_at(tmp_path, 0, cfg)
    assert s.report["crown_bar"] == {"tenure": 0, "margin": 0.03}


@pytest.mark.parametrize("n_before", [1, 3])
def test_the_house_holding_the_crown_counts_as_tenure(tmp_path, n_before):
    _need_rev623()
    cfg = lane_d_cfg()
    _, path, s = _x_at(tmp_path, n_before, cfg)
    want = 0.03 - (0.03 - 0.01) * n_before / 32
    assert s.report["crown_bar"]["tenure"] == n_before
    assert s.report["crown_bar"]["margin"] == pytest.approx(want, abs=1e-15)


def test_the_bar_is_flat_at_d_margin_after_the_warmup(tmp_path):
    _need_rev623()
    cfg = lane_d_cfg(d_margin_warmup_settlements=2)
    _, _, s = _x_at(tmp_path, 3, cfg)
    assert s.report["crown_bar"] == {"tenure": 3, "margin": 0.01}


def test_an_lcb_between_the_floor_and_the_start_bar_clears_only_after_tenure(tmp_path):
    """The same X, the same windows' quality: refused at tenure 0 by
    ``d_margin_start``, crowned at tenure 1 (warmup 1, so the bar is
    ``d_margin``) where its LCB clears the floor."""
    _need_rev623()
    _, _, probe0 = _x_at(tmp_path / "p0", 0, lane_d_cfg())
    _, _, probe1 = _x_at(tmp_path / "p1", 1, lane_d_cfg())
    lcb0 = verdict_of(probe0.report, entry("x", reveal_ts=T_X)["digest"])["lcb"]
    lcb1 = verdict_of(probe1.report, entry("x", reveal_ts=T_X)["digest"])["lcb"]
    lo, hi = min(lcb0, lcb1), max(lcb0, lcb1)
    assert 0.0 < lo <= hi < 1.0, f"fixture: X's LCBs {lcb0}, {lcb1} are not inside (0, 1)"
    cfg = lane_d_cfg(d_margin=lo / 2, d_margin_start=(hi + 1.0) / 2,
                     d_margin_warmup_settlements=1)
    x, path, s0 = _x_at(tmp_path / "t0", 0, cfg)
    v0 = verdict_of(s0.report, x["digest"])
    assert s0.report["crown_bar"] == {"tenure": 0, "margin": cfg.d_margin_start}
    assert v0["lcb"] >= cfg.d_margin and v0["cleared"] is False and v0["outcome"] == "lost"
    assert read_state(path)["king"]["house"] is True
    x, path, s1 = _x_at(tmp_path / "t1", 1, cfg)
    v1 = verdict_of(s1.report, x["digest"])
    assert s1.report["crown_bar"] == {"tenure": 1, "margin": cfg.d_margin}
    assert v1["cleared"] is True and v1["outcome"] == "crowned"
    assert read_state(path)["king"]["digest"] == x["digest"]


def test_a_crown_restarts_the_tenure_in_state(tmp_path):
    _need_rev623()
    cfg = lane_d_cfg()
    x, path, s = _x_at(tmp_path, 2, cfg)
    assert verdict_of(s.report, x["digest"])["outcome"] == "crowned"
    # the next settlement is the new KING_D's first after its crowning
    assert read_state(path)["king_tenure"] == 1
    pool1, _ = snapshot(tmp_path, 1)
    s2 = settle(cfg, boundary(4), pool1, [], house(), {}, uids(x), path)
    assert s2.report["crown_bar"]["tenure"] == 1
    assert read_state(path)["king_tenure"] == 2


def test_no_crown_advances_the_tenure_by_one(tmp_path):
    _need_rev623()
    cfg = lane_d_cfg()
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, _ = snapshot(tmp_path, 1)
    for n in (1, 2, 3):
        s = settle(cfg, boundary(n), pool1, [], house(), {}, {}, path)
        assert s.report["crown_bar"]["tenure"] == n - 1
        assert read_state(path)["king_tenure"] == n


@pytest.mark.parametrize("over", [
    {"d_margin": 0.04},                         # end > start
    {"d_margin_start": 0.005},                  # start < end
    {"d_margin": 0.0},                          # end <= 0
    {"d_margin_warmup_settlements": 0},         # warmup < 1
])
def test_the_loader_refuses_a_bad_bar(over):
    _need_rev623()
    with pytest.raises(ValueError):
        lane_d_cfg(**over)
