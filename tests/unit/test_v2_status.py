"""``status/v2.json`` (cascade.shared.v2_status): the dashboards' unsigned view
of the v2 competition — go-live state, the lane-B driver state, the king's
routing health — and the lane-B columns a receipt adds to its index row."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.shared import v2_status
from cascade.shared.config import LaneBConfig
from cascade.shared.hippius import StorageError
from cascade.shared.v2_status import (
    V2_STATUS_KEY,
    build_v2_status,
    driver_king_load,
    emission_split,
    golive,
    king_routing_health,
    payout_terms,
    publish_trainer_v2_status,
    publish_v2_status,
    v2_from_block,
    v2_index_fields,
)

DOMAINS = ("energy", "retail")
PTR = "metro-v1:trained:hippius:cascade/ckpt-r9-king@sha256:" + "a" * 64
PTR_PREV = "metro-v1:trained:hippius:cascade/ckpt-r8-king@sha256:" + "b" * 64


@pytest.fixture(autouse=True)
def _fresh_rate_limit():
    v2_status._LAST.clear()
    yield
    v2_status._LAST.clear()


def _cfg(from_block=0, **lb):
    lane_b = LaneBConfig(from_block=from_block, domains=DOMAINS if from_block or lb else (), **lb)
    return SimpleNamespace(lane_b=lane_b, subnet=SimpleNamespace(netuid=91),
                           v2=SimpleNamespace(from_block=from_block))


def _driver_state():
    sel = {1: [3, 4, 5], 2: [7, 8]}
    sub = {"hotkey": "hk-open", "uid": 12, "gen_ref": "ns/g@sha256:" + "c" * 64,
           "domain": "energy", "commit_block": 1000}
    return {
        "queue": json.dumps({
            "pending": [{"hotkey": "hk-wait", "uid": 13, "gen_ref": "ns/w@sha256:" + "d" * 64,
                         "domain": "retail", "commit_block": 1100}],
            "open": [{"sub": sub, "selection": {str(k): v for k, v in sel.items()},
                      "base_king_ref": PTR, "base_era": 4, "exclude": {}}],
        }),
        "finished": [],
        "pending_merges": [{"rebased_pointer": "metro-v1:trained:hippius:cascade/"
                                               "ckpt-r77-expert-retail-u9@sha256:" + "e" * 64,
                            "king_pointer": PTR, "selection": {"1": [9]}, "round_id": "77"}],
        "lineage_tokens": 150_000_000_000,
        "routing": {"era": 5, "block": 18000, "king": PTR, "prev_king": PTR_PREV,
                    "domains": {"energy": {"overlap": [0.9, 0.7], "jaccard": [0.8, 0.75],
                                           "tokens": 512},
                                "retail": {"overlap": [0.85, 0.95], "jaccard": [0.6, 0.9],
                                           "tokens": 400}},
                    "open": False, "reason": "lineage 1.5e+11 < 2e+11 tokens"},
        "routing_history": [
            {"era": e, "block": 3600 * e, "king": PTR, "prev_king": PTR_PREV,
             "lineage_tokens": e * 10, "domains": {"energy": {"overlap": [0.5, 0.6],
                                                             "jaccard": [0.4, 0.5]}}}
            for e in range(1, 4)],
        "gate_opened_block": 0,
        "outcomes": [{"round_id": "76", "hotkey": "hk-x", "uid": 3, "domain": "energy",
                      "outcome": "routing_stale", "detail": {"jaccard": [0.2]}}],
        "merged_inits": {"5": {"pointer": PTR, "base": PTR_PREV, "merges": [
            {"rebased_pointer": "metro-v1:trained:hippius:cascade/ckpt-r70-expert-energy-u2@sha256:"
                                + "f" * 64, "round_id": "70", "selection": {"1": [1]}}]}},
    }


# ── go-live ──────────────────────────────────────────────────────────────────


def test_golive_states():
    assert golive(_cfg(0), 500) == {"from_block": 0, "state": "unannounced", "blocks_to_go": None}
    assert golive(_cfg(1000), 400) == {"from_block": 1000, "state": "scheduled", "blocks_to_go": 600}
    assert golive(_cfg(1000), 1000)["state"] == "live"


def test_v2_section_wins_over_lane_b():
    cfg = _cfg(1001)
    del cfg.v2
    cfg.training = SimpleNamespace(v2_from_block=1000)    # [v2] from_block via the loader
    assert v2_from_block(cfg) == 1000
    cfg.v2 = SimpleNamespace(from_block=2000)
    assert v2_from_block(cfg) == 2000
    cfg = _cfg(0)
    cfg.training = SimpleNamespace(v2_from_block=0)       # 0 placeholder on mainnet
    assert golive(cfg, 5000)["state"] == "unannounced"


def test_lane_b_launch_does_not_stand_in_for_lane_a():
    cfg = _cfg(5001)
    cfg.v2.from_block = 1000
    doc = build_v2_status(cfg, block=2000)
    assert doc["golive"]["from_block"] == 1000
    assert doc["lane_b"]["from_block"] == 5000 and not doc["lane_b"]["active"]
    del cfg.v2
    assert v2_from_block(cfg) == 0


def test_payout_terms_read_either_denomination():
    t = payout_terms(LaneBConfig(domains=DOMAINS, half_life_rounds=1.0, floor=0.1,
                                 win_ttl_eras=28))
    # the gain-tiered full-pay period (owner 2026-10-20) replaces "full"
    assert t == {"full_pay_tiers": [[0.05, 10], [0.02, 5], [0.01, 2]], "half_life": 1.0,
                 "floor": 0.1, "ttl": 28.0, "unit": "eras", "share": 0.5, "slots": 2,
                 "remainder": "burn"}

    @dataclass(frozen=True)
    class SettleLB:                    # the original build: settlements, no TTL
        domains: tuple = DOMAINS
        half_life_rounds: float = 1.0
        floor: float = 0.1
        share: float = 0.5
        remainder: str = "burn"

    t = payout_terms(SettleLB())
    assert t["unit"] == "settlements" and t["ttl"] is None and t["full_pay_tiers"] is None

    @dataclass(frozen=True)
    class EraLB:                       # the go-live build: eras + a win TTL
        domains: tuple = DOMAINS
        full_pay_tiers: tuple = ((0.1, 2),)
        half_life_eras: float = 1.0
        floor: float = 0.1
        win_ttl_eras: int = 8
        share: float = 0.5
        remainder: str = "burn"

    t = payout_terms(EraLB())
    assert t["unit"] == "eras" and t["full_pay_tiers"] == [[0.1, 2]] and t["ttl"] == 8


# ── routing health ───────────────────────────────────────────────────────────


def test_king_routing_health_balanced_and_dead():
    E = 256
    load = np.full((2, E), 10.0)
    load[1, :64] = 0                        # a quarter of layer 1's experts never routed
    h = king_routing_health(load, bins=32, layers=[1, 5])
    assert h["experts"] == E and h["bins"] == 32
    a, b = h["layers"]
    assert a["layer"] == 1 and a["max_over_mean"] == 1.0 and a["cv"] == 0.0
    assert a["dead"] == 0 and a["entropy"] == pytest.approx(1.0)
    assert a["heat"] == [1.0] * 32
    assert b["layer"] == 5 and b["dead"] == 64 and b["entropy"] < 1.0
    assert b["heat"][:8] == [0.0] * 8 and b["heat"][8] == pytest.approx(4 / 3, abs=1e-3)
    assert b["max_over_mean"] == pytest.approx(4 / 3, abs=1e-3)


def test_king_routing_health_empty_and_zero_layer():
    assert king_routing_health(np.zeros((0, 0))) == {}
    h = king_routing_health(np.zeros((1, 8)), bins=4)
    assert h["layers"][0]["dead"] == 8 and h["layers"][0]["max_over_mean"] is None


def test_driver_king_load_sums_the_kings_cached_probes():
    probes = {(PTR, "energy", "d1"): SimpleNamespace(affinity=np.ones((2, 4))),
              (PTR, "retail", "d2"): SimpleNamespace(affinity=2 * np.ones((2, 4))),
              (PTR_PREV, "energy", "d1"): SimpleNamespace(affinity=100 * np.ones((2, 4)))}
    drv = SimpleNamespace(cfg=_cfg(1), _probes=probes)
    load, n = driver_king_load(drv, PTR)
    assert n == 2 and load.tolist() == [[3.0] * 4] * 2
    assert driver_king_load(drv, "") == (None, 0)


# ── the document ─────────────────────────────────────────────────────────────


def test_build_v2_status_from_driver_state():
    doc = build_v2_status(_cfg(1000), block=20000, state=_driver_state(), era_index=5,
                          king_pointer=PTR, health={"experts": 256}, health_domains=2,
                          as_of="2026-10-10T00:00:00+00:00")
    assert doc["schema"] == 1 and doc["golive"]["state"] == "live" and doc["netuid"] == 91
    lb = doc["lane_b"]
    assert lb["active"] and lb["domains"] == list(DOMAINS)
    g = lb["gate"]
    assert not g["open"] and g["lineage_tokens"] == 150_000_000_000
    assert g["open_tokens"] == 200_000_000_000 and g["reason"].startswith("lineage")
    assert g["latest"]["domains"]["energy"]["overlap"] == [0.9, 0.7]
    assert [h["era"] for h in g["history"]] == [1, 2, 3]
    assert g["history"][0]["overlap"] == [0.5, 0.6]
    e, r = lb["per_domain"]["energy"], lb["per_domain"]["retail"]
    assert e["open"][0]["hotkey"] == "hk-open" and e["open"][0]["per_layer"] == {"1": 3, "2": 2}
    assert e["selection"]["rows"] == {"1": [3, 4, 5], "2": [7, 8]}
    assert e["dropped"][0]["outcome"] == "routing_stale"
    assert r["pending"][0]["hotkey"] == "hk-wait" and r["pending_merge"] is True
    assert lb["merged_total"] == 1 and lb["merges"][0]["domains"] == ["energy"]
    assert doc["king"]["era"] == 5 and "2 domain" in doc["king"]["health_source"]
    json.dumps(doc, allow_nan=False)        # strict JSON


def test_build_v2_status_before_go_live_is_well_formed():
    doc = build_v2_status(_cfg(0), block=10, state=None)
    assert doc["golive"]["state"] == "unannounced"
    assert doc["lane_b"]["active"] is False and doc["lane_b"]["per_domain"] == {}
    assert doc["lane_b"]["gate"]["lineage_tokens"] == 0
    json.dumps(doc, allow_nan=False)


def test_history_is_downsampled():
    st = _driver_state()
    st["routing_history"] = [{"era": i, "block": i, "lineage_tokens": i,
                              "domains": {"energy": {"overlap": [0.5], "jaccard": [0.5]}}}
                             for i in range(1000)]
    hist = build_v2_status(_cfg(1), block=5, state=st)["lane_b"]["gate"]["history"]
    assert len(hist) <= 96 and hist[0]["era"] == 0 and hist[-1]["era"] == 999


# ── receipt index columns ────────────────────────────────────────────────────


def test_index_fields_absent_on_pre_v2_receipts():
    assert v2_index_fields(SimpleNamespace(lane_b=None, lane_b_pay=None)) == {}


def test_index_fields_and_exact_emission_split():
    # court = uids 1,2 (lane A 0.5 total), two domain slots of 0.25: energy
    # pays uid 7 in full, retail pays uid 8 at 0.4; the rest burns to uid 0
    weights = [0.15, 0.3, 0.2, 0, 0, 0, 0, 0.25, 0.1]
    pay = {"block": 9000, "epoch_blocks": 900, "wins": [
        {"domain": "energy", "hotkey": "a", "uid": 7, "merge_block": 8100, "age": 1, "multiplier": 1.0},
        {"domain": "retail", "hotkey": "b", "uid": 8, "merge_block": 3600, "age": 6, "multiplier": 0.4}]}
    rows = ({"hotkey": "a", "domain": "energy", "stage": "verdict", "passed": True,
             "reason": "merge", "target_lcb": 0.031, "guard_lcbs": {"retail": -0.002},
             "public_lcbs": {"gift_crps": 0.001}},
            {"hotkey": "c", "domain": "retail", "stage": "verdict", "passed": False,
             "reason": "regression on domain energy: LCB -0.0200 < -0.01", "target_lcb": 0.02,
             "guard_lcbs": {"energy": -0.02}, "public_lcbs": {}},
            {"hotkey": "d", "domain": "retail", "stage": "stale", "passed": False,
             "reason": "rebased onto x", "target_lcb": None, "guard_lcbs": {}, "public_lcbs": {}})
    rec = SimpleNamespace(lane_b=rows, lane_b_pay=pay, weights=tuple(weights), reward_uids=(1, 2))
    f = v2_index_fields(rec)
    assert f["lane_b_pay"] == pay
    v = f["lane_b_verdicts"]
    assert v[0]["passed"] and v[0]["worst_guard_lcb"] == -0.002 and v[0]["failed_guard"] is None
    assert v[1]["failed_guard"] == "energy" and v[2]["stage"] == "stale"
    assert f["emission_split"] == {"lane_a": 0.5, "lane_b": 0.35, "burn": 0.15}


def test_emission_split_overlapping_uid_needs_a_slot():
    # the only winner also sits in the court: no exact decomposition…
    assert emission_split([0.5, 0.5], [1], {"wins": [{"uid": 1, "multiplier": 1.0}]}) is None
    # …unless the pay record carries the share/slot terms
    s = emission_split([0.25, 0.75], [1], {"share": 0.5, "slots": 2,
                                           "wins": [{"uid": 1, "multiplier": 1.0}]})
    assert s == {"lane_a": 0.5, "lane_b": 0.25, "burn": 0.25}


# ── publish ──────────────────────────────────────────────────────────────────


class _Store:
    def __init__(self, fail_acl=False):
        self.fail_acl, self.writes = fail_acl, []

    def put_text(self, key, text, *, content_type="", acl=None):
        if acl and self.fail_acl:
            raise StorageError("acl unsupported")
        self.writes.append((key, json.loads(text), acl))


def test_publish_is_public_read_with_fallback():
    s = _Store()
    assert publish_v2_status(s, {"schema": 1}) == V2_STATUS_KEY
    assert s.writes == [(V2_STATUS_KEY, {"schema": 1}, "public-read")]
    s = _Store(fail_acl=True)
    publish_v2_status(s, {"schema": 1})
    assert s.writes[0][2] is None


def _sched(store, cfg=None, drv=None):
    king = SimpleNamespace(trained_pointer=PTR)
    cur = SimpleNamespace(index=5, king=lambda: king)
    return SimpleNamespace(cfg=cfg or _cfg(0), lane_b=drv, state=SimpleNamespace(current=cur),
                           r=SimpleNamespace(manifest_store=lambda: store))


def test_trainer_hook_rate_limits_and_skips_unchanged():
    store, t = _Store(), [0.0]
    sch = _sched(store)
    assert publish_trainer_v2_status(sch, 100, clock=lambda: t[0])
    assert store.writes[0][1]["golive"]["state"] == "unannounced"
    t[0] = 10
    assert not publish_trainer_v2_status(sch, 101, clock=lambda: t[0])   # inside the interval
    t[0] = 500
    assert publish_trainer_v2_status(sch, 102, clock=lambda: t[0])       # block moved
    t[0] = 1000
    sch.cfg = _cfg(5000)
    assert publish_trainer_v2_status(sch, 102, clock=lambda: t[0])
    assert store.writes[-1][1]["golive"] == {"from_block": 5000, "state": "scheduled",
                                             "blocks_to_go": 4898}


def test_trainer_hook_never_raises():
    class Boom:
        def put_text(self, *a, **k):
            raise RuntimeError("s3 down")

    assert publish_trainer_v2_status(_sched(Boom()), 1, clock=lambda: 0.0) is False


def test_unchanged_state_still_refreshes_heartbeat(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from cascade.miner.live_status import freshness

    store, t = _Store(), [0.0]
    sch = _sched(store)
    base = datetime(2026, 10, 22, tzinfo=UTC)
    monkeypatch.setattr(v2_status, "_now", lambda: (base + timedelta(seconds=t[0])).isoformat())
    for stamp in (0, 121, 242, 363):
        t[0] = stamp
        assert publish_trainer_v2_status(sch, 100, clock=lambda: t[0])
        doc = store.writes[-1][1]
        assert freshness(doc, now=base + timedelta(seconds=stamp), max_age=180)["fresh"]
    assert len(store.writes) == 4


def test_trainer_publishes_current_merged_init_and_running_policy_before_settlement():
    store = _Store()
    sch = _sched(store)
    cur = sch.state.current
    cur.start_block = 18000
    cur.warm_start_ckpt = "metro-v1:trained:hippius:cascade/merged-init@sha256:" + "b" * 64
    cur.warm_start_size = "moe-28m"
    cur.generation, cur.member_index = 3, 1
    sch.cfg.generation = SimpleNamespace(enabled=True, window_seconds=5400, vcpus=32,
                                         max_parallel_jobs=2)
    assert publish_trainer_v2_status(sch, 18001, clock=lambda: 0.0)
    doc = store.writes[-1][1]
    assert doc["warm_start"]["pointer"] == cur.warm_start_ckpt != PTR
    assert doc["warm_start"]["generation"] == 3
    assert doc["warm_start"]["era_start_block"] == 18000
    assert doc["policy"]["generation"]["window_seconds"] == 5400


def test_trainer_hook_reads_driver_state_and_king_health():
    store = _Store()
    st = _driver_state()
    probes = {(PTR, d, "x"): SimpleNamespace(affinity=np.ones((7, 256))) for d in DOMAINS}
    drv = SimpleNamespace(cfg=_cfg(1), st=st, _probes=probes)
    sch = _sched(store, cfg=_cfg(1), drv=drv)
    assert publish_trainer_v2_status(sch, 20000, clock=lambda: 0.0, allow_probe=False)
    doc = store.writes[0][1]
    assert doc["lane_b"]["gate"]["lineage_tokens"] == 150_000_000_000
    h = doc["king"]["routing_health"]
    assert len(h["layers"]) == 7 and h["layers"][0]["max_over_mean"] == 1.0
    assert [x["layer"] for x in h["layers"]] == doc["model"]["moe_blocks"]
