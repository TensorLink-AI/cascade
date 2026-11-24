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


# ── #913: lane D paying a miner court (KING_D + a prior king) ───────────────
#
# Testnet b274626d (2026-11-24): once lane D paid a real hotkey, every
# satellite burned its WHOLE vector at the push ("reward-uid: uid 3 is
# registered to …, which the receipt does not pay"): the payee attribution
# ignored lane D's court. The court below is KING_D (uid 3) and one prior
# king (uid 2) at the [lane_d] court_weights in force at the settlement.

PRIOR_UID = 2


def _court_cfg(tmp_path):
    """`_cfg` with no public half (the hand-built report carries none)."""
    cfg = _cfg(tmp_path)
    return replace(cfg, lane_d=replace(cfg.lane_d, public_weight=0.0))


def _paying_report(cfg, boundary: int, court_hotkeys=("x3_hk", "x2_hk")) -> dict:
    """A lane D report the main validator signed at a settlement that judges
    nothing (a miner holds KING_D): every burn-class check passes and the
    paid court is KING_D, then the prior king, by recency."""
    from cascade.audit.lane_d import _canon, _expected_params, crown_bar_of
    from tests.unit.test_satellite_replay import MAIN_KP

    pool = "ab" * 32
    king = {"digest": "d" * 64, "hotkey": court_hotkeys[0], "house": False, "reveal_ts": 1000}
    prior = {"digest": "e" * 64, "hotkey": court_hotkeys[1], "house": False, "reveal_ts": 900}
    report = {
        "schema": "lane-d-report/1", "boundary": boundary, "scoring_method": "quantile_head",
        "params": _expected_params(cfg.lane_d), "pool": {"sha256": pool},
        "seed": f"lane-d/1:{boundary}:{pool}", "rows": [],
        "rows_sha256": hashlib.sha256(_canon([])).hexdigest(),
        "windows": [], "entries": [], "cohorts": [], "verdicts": [], "judged": [],
        "house": {"digest": "f" * 64, "reveal_ts": 0},
        "king_before": king, "king_after": king, "crowned": None, "prior_kings": [prior],
        "crown_bar": {"tenure": 2, "margin": crown_bar_of(cfg.lane_d, 2)},
        "pay": {"share": SHARE, "court": [
            {"digest": king["digest"], "hotkey": king["hotkey"], "uid": LANE_D_UID,
             "burn_reason": None},
            {"digest": prior["digest"], "hotkey": prior["hotkey"], "uid": PRIOR_UID,
             "burn_reason": None}]},
        "signer": MAIN_KP.ss58_address,
    }
    report["signature"] = MAIN_KP.sign(_canon(report)).hex()
    return report


def _publish_report(store, report: dict) -> None:
    from cascade.audit.lane_d import _canon

    b = int(report["boundary"])
    body = hashlib.sha256(_canon({k: v for k, v in report.items() if k != "signature"}))
    rec = {"schema": "lane-d-receipt/1", "boundary": b,
           "prior_state_sha256": "1" * 64, "posterior_state_sha256": "2" * 64,
           "report_sha256": body.hexdigest()}
    store.put_text(f"lane-d/reports/lane_d_report_{b}.json", json.dumps(report))
    store.put_text(f"lane-d/receipts/lane_d_receipt_{b}.json", json.dumps(rec))


def _court_vector(cfg, reward_uids, n, court):
    from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts

    a, b = lane_parts(list(reward_uids), n, king_decay=cfg.scoring.king_decay,
                      burn_uid=cfg.scoring.burn_uid, lane_d_cfg=cfg.lane_d,
                      d_court_uids=list(court))
    return emission_vector(a, b, EmissionPolicy(), burn_uid=cfg.scoring.burn_uid,
                           lane_a_uids=list(reward_uids))


def _court_settlement(tmp_path, store):
    from tests.unit.test_satellite_replay import BOUNDARIES, HOTKEYS, make_settlement

    cfg = _court_cfg(tmp_path / "cfg")
    n = len(HOTKEYS)
    court = [LANE_D_UID, PRIOR_UID]
    receipt, _ = make_settlement(
        cfg, store, 0,
        tamper=lambda rc: replace(rc, weights=tuple(_court_vector(cfg, rc.reward_uids, n,
                                                                  court))))
    _publish_report(store, _paying_report(cfg, BOUNDARIES[0]))
    return cfg, receipt


def test_913_a_lane_d_court_paying_king_d_and_a_prior_king_is_followed(tmp_path, offline):
    """KING_D + a prior king paid at the court weights in force: the
    satellite verifies the signed court, attributes its payees and sets the
    recorded vector exactly; nothing burns."""
    from tests.unit.test_satellite_replay import (
        BOUNDARIES,
        DictStore,
        FakeClient,
        _runner,
    )

    store = DictStore()
    cfg, receipt = _court_settlement(tmp_path, store)
    recorded = [float(w) for w in receipt.weights]
    w = [float(x) for x in cfg.lane_d.court_weights[:2]]
    assert recorded[LANE_D_UID] == pytest.approx(SHARE * w[0] / sum(w))
    assert recorded[PRIOR_UID] == pytest.approx(SHARE * w[1] / sum(w))
    runner = _runner(cfg, store, tmp_path / "sat")
    client = FakeClient(BOUNDARIES[0] + 300)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    assert doc["mode"] != "burning", doc
    assert client.pushed[-1] == recorded, "the satellite did not set the lane D court"
    assert runner.state.attribution["lane_d"] == {str(LANE_D_UID): "x3_hk",
                                                  str(PRIOR_UID): "x2_hk"}
    # restart: the persisted state carries the court attribution
    again = _runner(cfg, store, tmp_path / "sat")
    client2 = FakeClient(BOUNDARIES[0] + 400)
    again.push(client2, now_block=client2.block)
    assert client2.pushed[-1] == recorded


@pytest.mark.parametrize("bad_uid", [LANE_D_UID, PRIOR_UID])
def test_913_a_lane_d_payee_registered_elsewhere_burns_only_lane_d_share(tmp_path, offline,
                                                                         caplog, bad_uid):
    """The satellite's metagraph registers a lane D court uid to a hotkey
    the signed court does not name: lane D's share (and only it) burns;
    lanes A/B/C are set exactly as recorded."""
    from tests.unit.test_satellite_replay import (
        BOUNDARIES,
        HOTKEYS,
        DictStore,
        FakeClient,
        _runner,
    )

    store = DictStore()
    cfg, receipt = _court_settlement(tmp_path, store)
    recorded = [float(w) for w in receipt.weights]
    burn = int(cfg.scoring.burn_uid)
    hotkeys = list(HOTKEYS)
    hotkeys[bad_uid] = "mallory_hk"
    runner = _runner(cfg, store, tmp_path / "sat")
    client = FakeClient(BOUNDARIES[0] + 300, hotkeys=hotkeys)
    with caplog.at_level("WARNING", logger="cascade.satellite"):
        doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc       # never a whole-vector burn
    pushed = client.pushed[-1]
    d_part = {LANE_D_UID: recorded[LANE_D_UID], PRIOR_UID: recorded[PRIOR_UID]}
    for u in range(len(HOTKEYS)):
        if u in d_part:
            assert pushed[u] == pytest.approx(0.0, abs=1e-12), u
        elif u == burn:
            assert pushed[u] == pytest.approx(recorded[u] + sum(d_part.values()), abs=1e-12)
        else:
            assert pushed[u] == recorded[u], f"uid {u}: lane A/B/C moved"
    assert sum(pushed) == pytest.approx(1.0)
    assert "lane_d_check_burn" in caplog.text and "reward-uid" in caplog.text


def test_913_a_lane_a_payee_registered_elsewhere_still_burns_the_whole_vector(tmp_path,
                                                                              offline):
    """Scoping covers lane D's payees only: a lane A payee mismatch at the
    same settlement still burns the whole vector (unchanged)."""
    from tests.unit.test_satellite_replay import (
        BOUNDARIES,
        HOTKEYS,
        DictStore,
        FakeClient,
        _runner,
    )

    store = DictStore()
    cfg, receipt = _court_settlement(tmp_path, store)
    burn = int(cfg.scoring.burn_uid)
    lane_a = [u for u in receipt.reward_uids if u != burn]
    assert lane_a, "fixture: lane A pays a uid other than burn_uid"
    hotkeys = list(HOTKEYS)
    hotkeys[lane_a[0]] = "mallory_hk"
    runner = _runner(cfg, store, tmp_path / "sat")
    client = FakeClient(BOUNDARIES[0] + 300, hotkeys=hotkeys)
    doc = runner.poll(client)
    assert doc["last_failing_check"] == "reward-uid", doc
    pushed = client.pushed[-1]
    assert pushed[burn] == pytest.approx(1.0)
