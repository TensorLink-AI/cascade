"""W8b satellite path (rev6.18 B1, owner 2026-11-23): "A failed lane D check
must burn only lane D's share. [...] Done when a deliberately broken lane D
report on a satellite leaves lane A, B and C weights unchanged, with a test."

The satellite (`cascade.validator.satellite`) replays one main-validator
receipt at a settlement lane D pays (lane D carved from lane A). The lane D
report it reads is signed by the main validator but broken (B3 burn check
`ld_rows_sha256`). The pushed vector must equal the recorded one on every
uid but lane D's court slot and `burn_uid`, byte for byte, with exactly
lane D's share moved to `burn_uid`; the satellite never burns the whole
vector for it.

On main the satellite reads no lane D record, so every lane D settlement
fails `weights` / `weights-derive` and the whole vector burns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

pytest.importorskip("bittensor.sp_core")

LANE_D_UID = 3          # x3_hk: lane D's KING_D in the broken report's court
SHARE = 0.25


def _cfg(tmp_path):
    from cascade.shared.config import load_chain_config, parse_chain_config
    from tests.unit.predecision_main import predecision_path
    from tests.unit.test_satellite_replay import BOUNDARIES, MAIN_KP, TRAINER_KP

    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = json.loads(json.dumps(load_chain_config(predecision_path(tmp_path)).raw))
    sets = {
        "manifest": {"trainer_hotkey": TRAINER_KP.ss58_address, "validator_hotkey": ""},
        "netconfig": {"owner_hotkey": MAIN_KP.ss58_address},
        "training": {"expected_gpu": "", "train_image_digest": ""},
        "generator": {"corpus_target_points": 0, "min_channels": 1, "sandbox_strict": False},
        "v2": {"launch_at_utc": "", "commit_open_at_utc": "", "commit_from_block": 0,
               "image_decoupled_from_block": 0, "prelaunch_burn_readiness": ""},
        "round": {"commit_floor_block": 0, "dedup_mode": "off", "epoch_blocks_prev": 0,
                  "epoch_activation_block": 0},
        "scoring": {"win_margin_start_prev": 0.0, "margin_activation_block": 0,
                    "forfeit_from_block": 0, "margin_v2_from_block": 0},
    }
    for table, kv in sets.items():
        raw.setdefault(table, {}).update(kv)
    cfg = parse_chain_config(raw)
    lane_d = replace(cfg.lane_d, from_block=BOUNDARIES[0] - 1, share=SHARE,
                     takes_from="lane_a")
    assert lane_d.active(BOUNDARIES[0])
    return replace(cfg, lane_d=lane_d)


def _producer_vector(cfg, reward_uids, n):
    """Lane A's court in lane A's pool (1 - share), lane D's court
    `[LANE_D_UID]` in lane D's share (W8's emission)."""
    from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts

    a, b = lane_parts(list(reward_uids), n, king_decay=cfg.scoring.king_decay,
                      burn_uid=cfg.scoring.burn_uid, lane_d_cfg=cfg.lane_d,
                      d_court_uids=[LANE_D_UID])
    return emission_vector(a, b, EmissionPolicy(), burn_uid=cfg.scoring.burn_uid,
                           lane_a_uids=list(reward_uids))


def _broken_report(cfg, boundary: int) -> dict:
    """A lane D report the main validator signed whose rows do not hash to
    `rows_sha256` (B3 `ld_rows_sha256`: burn)."""
    from cascade.audit.lane_d import _canon, _expected_params
    from tests.unit.test_satellite_replay import MAIN_KP

    pool = "ab" * 32
    report = {
        "schema": "lane-d-report/1", "boundary": boundary, "scoring_method": "quantile_head",
        "params": _expected_params(cfg.lane_d), "pool": {"sha256": pool},
        "seed": f"lane-d/1:{boundary}:{pool}", "rows": [], "rows_sha256": "0" * 64,
        "pay": {"share": SHARE, "court": [{"digest": "d" * 64, "hotkey": "x3_hk",
                                           "uid": LANE_D_UID, "burn_reason": None}]},
        "signer": MAIN_KP.ss58_address,
    }
    report["signature"] = MAIN_KP.sign(_canon(report)).hex()
    return report


def _publish_lane_d(store, cfg, boundary: int) -> None:
    from cascade.audit.lane_d import _canon

    report = _broken_report(cfg, boundary)
    body = hashlib.sha256(_canon({k: v for k, v in report.items() if k != "signature"}))
    rec = {"schema": "lane-d-receipt/1", "boundary": boundary,
           "prior_state_sha256": "1" * 64, "posterior_state_sha256": "2" * 64,
           "report_sha256": body.hexdigest()}
    store.put_text(f"lane-d/reports/lane_d_report_{boundary}.json", json.dumps(report))
    store.put_text(f"lane-d/receipts/lane_d_receipt_{boundary}.json", json.dumps(rec))


@pytest.fixture
def offline(monkeypatch):
    import cascade.audit.main as audit_main

    def _offline(cfg, key):
        raise RuntimeError("offline")
    monkeypatch.setattr(audit_main, "_fetch_text", _offline)


def test_a_broken_lane_d_report_burns_only_lane_d_share(tmp_path, offline, caplog):
    from tests.unit.test_satellite_replay import (
        BOUNDARIES,
        HOTKEYS,
        DictStore,
        FakeClient,
        _runner,
        make_settlement,
    )

    cfg = _cfg(tmp_path / "cfg")
    n = len(HOTKEYS)
    burn = int(cfg.scoring.burn_uid)
    assert burn != LANE_D_UID
    store = DictStore()

    def lane_d_weights(rc):
        return replace(rc, weights=tuple(_producer_vector(cfg, rc.reward_uids, n)))

    receipt, _ = make_settlement(cfg, store, 0, tamper=lane_d_weights)
    recorded = [float(w) for w in receipt.weights]
    assert recorded[LANE_D_UID] == pytest.approx(SHARE)
    _publish_lane_d(store, cfg, BOUNDARIES[0])

    runner = _runner(cfg, store, tmp_path / "sat")
    client = FakeClient(BOUNDARIES[0] + 300)
    with caplog.at_level("WARNING", logger="cascade.satellite"):
        doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc        # never a whole-vector burn
    assert doc["last_verified_round"] == receipt.round_id
    pushed = client.pushed[-1]
    for u in range(n):
        if u == LANE_D_UID:
            assert pushed[u] == pytest.approx(recorded[u] - SHARE, abs=1e-12)
        elif u == burn:
            assert pushed[u] == pytest.approx(recorded[u] + SHARE, abs=1e-12)
        else:
            assert pushed[u] == recorded[u], f"uid {u}: {pushed[u]} != recorded {recorded[u]}"
    assert sum(pushed) == pytest.approx(1.0)
    assert "lane_d_check_burn" in caplog.text and "ld_rows_sha256" in caplog.text


def test_a_missing_lane_d_record_burns_only_lane_d_share(tmp_path, offline):
    """No lane D report published at a lane D settlement
    (`lane_d_record_missing`): lane D's share burns, lanes A/B/C as recorded."""
    from tests.unit.test_satellite_replay import (
        BOUNDARIES,
        HOTKEYS,
        DictStore,
        FakeClient,
        _runner,
        make_settlement,
    )

    cfg = _cfg(tmp_path / "cfg")
    n = len(HOTKEYS)
    burn = int(cfg.scoring.burn_uid)
    store = DictStore()
    receipt, _ = make_settlement(
        cfg, store, 0,
        tamper=lambda rc: replace(rc, weights=tuple(_producer_vector(cfg, rc.reward_uids, n))))
    recorded = [float(w) for w in receipt.weights]
    runner = _runner(cfg, store, tmp_path / "sat")
    client = FakeClient(BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    pushed = client.pushed[-1]
    for u in range(n):
        if u not in (LANE_D_UID, burn):
            assert pushed[u] == recorded[u], u
    assert pushed[burn] == pytest.approx(recorded[burn] + SHARE, abs=1e-12)
