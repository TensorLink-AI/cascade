"""W8 through the settlement: T1 (house only, 2 eras) and the report half of
invariant 9 (armed with ``share = 0`` => identical vector AND a valid report).

The settlement is the W1.5 box-side ``settle_lane_d`` (synthetic eval-host
forecasts, the W1.5 ``pool_d/1`` fixture builders); its ``pay`` record feeds
``lane_parts`` exactly as the validator does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .helpers import (
    ARM_BLOCK,
    BURN_UID,
    ERA,
    LANE_A_COURT,
    REL,
    SETTLEMENT,
    as_bytes,
    normalised,
    paid_uids,
    parts,
    vector,
)

# T1: two settlements in two different eras after arming.
T1_BOUNDARIES = (ARM_BLOCK + SETTLEMENT, ARM_BLOCK + ERA + SETTLEMENT)

LANE_A_FILES = {"cascade_state.json": b'{"king": "lane-a-king", "king_since_block": 9000000}',
                "validator_state.db": b"SQLite format 3\x00 lane A verdict rows"}


def _w15():
    from tests.accept.w1_5 import helpers

    return helpers


def _house_only_chain(tmp: Path):
    """Two ``pool_d`` snapshots and the house's forecasts; no miner entry."""
    h = _w15()
    house = h.text_digest("w8 house")
    rng = np.random.default_rng(8)
    pools, forecasts = {}, {}
    for snap in (1, 2):
        series = h.make_series(snap, 80 + snap)
        targets, fc = {}, {house: {}}
        for s in series:
            sd = float(np.std(s["history"]))
            target = np.resize(s["history"][-24:], h.HORIZON) + rng.normal(0, 0.3, h.HORIZON)
            targets[s["series_id"]] = target
            fc[house][h.window_id(s["series_id"], s["target_ts"])] = h.quantiles_around(
                target, 0.3 * sd)
        pools[snap] = h.write_pool_d(tmp / f"pool_d_{snap}", snap, series, targets)
        forecasts[snap] = fc
    return {"house": {"digest": house, "reveal_ts": h.HOUSE_T}, "pools": pools,
            "forecasts": forecasts, "signer": h.make_signer(tmp / "key")}


def _state_dir(tmp: Path) -> Path:
    d = tmp / "state"
    d.mkdir(parents=True)
    for name, data in LANE_A_FILES.items():
        (d / name).write_bytes(data)
    return d


def _settle(chain, cfg, state_path, boundary, snap):
    from cascade.validator.lane_d import settle_lane_d

    return settle_lane_d(cfg=cfg, boundary=boundary, pool_dir=chain["pools"][snap], entries=[],
                         house=dict(chain["house"]), forecasts=chain["forecasts"][snap],
                         uid_of={}, state_path=state_path, signer=chain["signer"])


def _pay_parts(pay, cfg):
    return parts(d_cfg=cfg if pay is not None else None,
                 d_court=tuple(pay["court_uids"]) if pay is not None else ())


def _assert_report_shape(report, cfg, boundary):
    assert report is not None, "an armed settlement produced no lane D report"
    assert report["schema"] == "lane-d-report/1"
    assert report["boundary"] == boundary
    assert report["params"]["share"] == cfg.share
    sha = report["rows_sha256"]
    assert isinstance(sha, str) and len(sha) == 64 and int(sha, 16) >= 0


def _assert_report_verifies(report, cfg, signer):
    """sr25519 verification needs bittensor >= 11 (``bittensor.sp_core``),
    which uv.lock pins; an older local SDK skips only this step."""
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.checks import PASS  # noqa: I001
    from cascade.audit.lane_d import check_lane_d
    from cascade.validator.lane_d_report import verify_report

    assert verify_report(report, signer.ss58_address)
    res = check_lane_d(report, cfg, signer_hotkey=signer.ss58_address)
    assert res.status == PASS, f"check_lane_d: {res.status} {res.detail}"


def test_t1_house_only_two_eras_slot_burns(tmp_path):
    """T1: KING_D = house; its slot burns; lane A verdicts and court unchanged."""
    cfg = _w15().lane_d_cfg()
    chain = _house_only_chain(tmp_path / "chain")
    state_dir = _state_dir(tmp_path)
    a0, _ = parts()
    for snap, boundary in enumerate(T1_BOUNDARIES, start=1):
        s = _settle(chain, cfg, state_dir / "lane_d_state.json", boundary, snap)
        assert s.pay is not None and s.pay["share"] == 0.25
        assert s.pay["court_uids"] == [None], "the house holds KING_D: one burning slot"
        _assert_report_shape(s.report, cfg, boundary)
        a, b = _pay_parts(s.pay, cfg)
        vec = vector(a, b)
        assert vec[BURN_UID] == pytest.approx(0.25, rel=REL)          # D21: the slot burns
        assert sum(b) == pytest.approx(0.25, rel=REL) and b[BURN_UID] == pytest.approx(0.25)
        assert paid_uids(vec) == list(LANE_A_COURT)                   # the house is unpaid
        assert paid_uids(a) == paid_uids(a0)
        assert normalised(a) == pytest.approx(normalised(a0), rel=REL)
        assert sum(a) == pytest.approx(0.75, rel=REL)
    for name, data in LANE_A_FILES.items():
        assert (state_dir / name).read_bytes() == data, f"lane A state {name} changed"


def test_inv9_share_zero_vector_armed_nonempty_lane_d_effect(tmp_path):
    """Armed with ``share = 0``: the vector is byte-identical to unarmed, and
    a valid signed lane D report is still produced."""
    h = _w15()
    cfg0, unarmed = h.lane_d_cfg(share=0.0), h.lane_d_cfg(from_block=0)
    chain = _house_only_chain(tmp_path / "chain")
    s_un = _settle(chain, unarmed, _state_dir(tmp_path / "u") / "lane_d_state.json",
                   T1_BOUNDARIES[0], 1)
    assert s_un.report is None and s_un.pay is None
    a_un, _ = _pay_parts(s_un.pay, unarmed)
    today = vector(a_un, None)
    for snap, boundary in enumerate(T1_BOUNDARIES, start=1):
        s = _settle(chain, cfg0, _state_dir(tmp_path / f"z{snap}") / "lane_d_state.json",
                    boundary, snap)
        assert s.pay is not None and s.pay["share"] == 0.0
        _assert_report_shape(s.report, cfg0, boundary)
        a, b = _pay_parts(s.pay, cfg0)
        assert as_bytes(vector(a, b)) == as_bytes(today)
    _assert_report_verifies(s.report, cfg0, chain["signer"])
