"""Lane C eval-offload robustness (W-CMP, owner 2026-11-27 "fix all those
things"). The MarketCast offload pod, the provisioner's mkt stage, the
trainer's report wait and the lane-C intake:

1. the pod-free prechecks (snapshot, block hash) run BEFORE the lease, so a
   missing ``pool_mkt`` snapshot never rents a pod;
2. a forecast run longer than ``mkt_lease_hours`` keeps its pod while the
   validator heartbeats, up to ``mkt_lease_max_hours``;
3. the validator's ``--mkt-host-wait`` covers the provisioner's worst-case
   boot (a shorter wait released the request mid-boot);
4. the mkt stage falls back RTX 3090 → RTX 4090 → RTX A6000, each capped;
5. a round with no MarketCast entries is not awaited, and only a CONFIRMED
   404 ages the wait (a Hippius read error never turns a report "absent");
6. recording "absent" for a round WITH entries is an ERROR (the line
   cascade-ops' ``mkt_report_absent`` pattern matches);
7. ``deploy/restart_provisioner.sh`` never stops the MarketCast provisioner;
8. lane C refuses a payer bundle without a Lium key.

No pods, no chain, no network, no torch."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
T0 = 1_800_000_000.0


# ── 1. prechecks before the lease ────────────────────────────────────────────


def _publisher(tmp_path, *, snapshot=True, bhash=True):
    from cascade.shared.hippius import ObjectNotFound
    from cascade.validator.marketcast import MarketCastMixin

    leased: list = []

    class _Lease:
        def __init__(self, path):
            leased.append(path)

        def acquire(self, rid, wait_s):
            return "HOST"

        def touch(self, rid):
            pass

        def release(self):
            pass

    class _Sess:
        def __init__(self, host):
            pass

        def start(self):
            pass

        def close(self):
            pass

    class _Store:
        def get_text(self, key):
            raise ObjectNotFound(key)

    class V(MarketCastMixin):
        pass

    def _hash(b):
        if not bhash:
            raise ConnectionError("rpc down")
        return "0x" + "00" * 32

    v = V()
    v.mkt_eval_hosts = tmp_path / "mkt_hosts.toml"
    v.mkt_lease_fn, v.mkt_session_fn = _Lease, _Sess
    v._mkt_store_obj = _Store()
    v.mkt_snapshot_fn = lambda block: (SimpleNamespace(), "k", "s") if snapshot else None
    v.mkt_block_hash_fn = _hash
    published: list = []
    v._mkt_publish_round = lambda item, *, holders=None: published.append(item["round_id"]) or True
    return v, leased, published


def test_w_cmp_missing_snapshot_never_leases_a_pod(tmp_path):
    v, leased, published = _publisher(tmp_path, snapshot=False)
    assert v._mkt_publish_remote([{"round_id": "r1", "block": 10, "candidates": [{}]}], {}) == 0
    assert leased == [], "a missing pool_mkt snapshot rented a pod"
    assert published == []
    # a ready round still leases and publishes
    v, leased, published = _publisher(tmp_path)
    assert v._mkt_publish_remote([{"round_id": "r1", "block": 10, "candidates": [{}]}], {}) == 1
    assert len(leased) == 1 and published == ["r1"]


def test_w_cmp_unreadable_block_hash_never_leases_a_pod(tmp_path):
    v, leased, _ = _publisher(tmp_path, bhash=False)
    assert v._mkt_publish_remote([{"round_id": "r1", "block": 10, "candidates": [{}]}], {}) == 0
    assert leased == [], "an unreadable block hash rented a pod"


# ── 2. lease extension while the heartbeat is fresh ──────────────────────────


def _mkt_loop(tmp_path, *, clock=None, prov=None, policy=None):
    from cascade.provision.policy import ProvisionPolicy, StagePolicy
    from tests.unit.test_provision_loop import Clock, FakeProvider, FakeStore, make_loop

    policy = policy or ProvisionPolicy(
        max_spend_per_round=5.0,
        mkt=StagePolicy(sku="NVIDIA GeForce RTX 3090", market_sku="RTX3090", gpus_per_pod=1,
                        max_pods=1, providers=("lium",), max_price_hr=0.4))
    prov = prov or FakeProvider("lium", price=0.2)
    loop = make_loop(tmp_path, providers={"lium": prov}, policy=policy,
                     clock=clock or Clock(T0), store=FakeStore())
    loop.mkt_hosts_path = tmp_path / "mkt_hosts.toml"
    return loop, prov


def _touch(loop, age_s=0.0):
    from cascade.validator.mkt_offload import request_path_for

    p = request_path_for(loop.mkt_hosts_path)
    p.write_text("{}", encoding="utf-8")
    t = loop.clock() - age_s
    os.utime(p, (t, t))


def _cycle(loop):
    loop.run_once()
    t = getattr(loop, "_mkt_thread", None)
    if t is not None:
        t.join(timeout=30)


def test_w_cmp_a_heartbeat_keeps_the_pod_past_the_base_lease(tmp_path):
    from tests.unit.test_provision_loop import Clock

    clock = Clock(T0)
    loop, prov = _mkt_loop(tmp_path, clock=clock)
    _touch(loop)
    _cycle(loop)
    assert len(prov.launched) == 1
    clock.t += 3 * 3600 + 60                       # past mkt_lease_hours = 3
    _touch(loop)                                   # the validator is still computing
    loop.run_once()
    assert prov.terminated == [], "a >3 h forecast run lost its pod mid-run"
    clock.t += 3 * 3600                            # past mkt_lease_max_hours = 6
    _touch(loop)
    loop.run_once()
    assert prov.terminated == prov.launched


def test_w_cmp_lease_max_parses_round_trips_and_is_bounded():
    from cascade.provision.core import ProvisionError
    from cascade.provision.main import build_policy

    text = textwrap.dedent("""
        [provisioner]
        max_spend_per_round = 5
        mkt_lease_hours = 2
        mkt_lease_max_hours = 7.5
        [provisioner.mkt]
        sku = "NVIDIA GeForce RTX 3090"
        market_sku = "RTX3090"
        max_price_hr = 0.4
    """)
    pol = build_policy(tomllib.loads(text))
    assert hasattr(pol, "mkt_lease_max_hours"), "mkt_lease_max_hours is not parsed"
    assert (pol.mkt_lease_hours, pol.mkt_lease_max_hours) == (2.0, 7.5)
    assert pol.mkt_ttl_hours(heartbeat_fresh=True) == 7.5
    assert pol.mkt_ttl_hours(heartbeat_fresh=False) == 2.0
    default = build_policy(tomllib.loads(text.replace("mkt_lease_max_hours = 7.5\n", "")))
    assert default.mkt_lease_max_hours == 6.0
    with pytest.raises(ProvisionError, match="mkt_lease_max_hours"):
        build_policy(tomllib.loads(text.replace("7.5", "1")))       # below the base lease
    with pytest.raises(ProvisionError, match="mkt_lease_max_hours"):
        build_policy(tomllib.loads(text.replace("7.5", "25")))


def test_w_cmp_the_budget_prices_the_extended_lease(tmp_path):
    from cascade.provision.policy import ProvisionPolicy, StagePolicy

    pol = ProvisionPolicy(
        max_spend_per_round=2.0,                   # 0.4 × 3 h fits, 0.4 × 6 h does not
        mkt=StagePolicy(sku="NVIDIA GeForce RTX 3090", market_sku="RTX3090", gpus_per_pod=1,
                        max_pods=1, providers=("lium",), max_price_hr=0.4,
                        candidates=()))
    from tests.unit.test_provision_loop import FakeProvider

    loop, prov = _mkt_loop(tmp_path, policy=pol, prov=FakeProvider("lium", price=0.4))
    _touch(loop)
    _cycle(loop)
    assert prov.launched == []


# ── 3. the host wait covers the worst-case boot ──────────────────────────────


def test_w_cmp_mkt_host_wait_covers_the_worst_case_boot():
    from cascade.provision.core import DEFAULT_READY_TIMEOUT
    from cascade.provision.policy import MKT_BOOT_STEPS_S, MKT_BOOT_WORST_S
    from cascade.validator.main import _build_parser as build_parser
    from cascade.validator.marketcast import MarketCastMixin

    assert MKT_BOOT_STEPS_S[0] == DEFAULT_READY_TIMEOUT
    # ready + 300 s ssh probe + 900 s key auth + gate, with one replacement
    assert MKT_BOOT_WORST_S >= 2 * (DEFAULT_READY_TIMEOUT + 300.0 + 900.0)
    args = build_parser().parse_args(["--mkt-eval-hosts", "/x/mkt_hosts.toml"])
    assert args.mkt_host_wait >= MKT_BOOT_WORST_S, (
        f"--mkt-host-wait {args.mkt_host_wait:.0f}s < worst-case boot {MKT_BOOT_WORST_S:.0f}s")
    assert MarketCastMixin.mkt_host_wait_s >= MKT_BOOT_WORST_S


# ── 4. the SKU ladder ────────────────────────────────────────────────────────


def _mkt_raw(**mkt):
    return {"provisioner": {"max_spend_per_round": 5,
                            "mkt": {"sku": "NVIDIA GeForce RTX 3090", "market_sku": "RTX3090",
                                    "max_price_hr": 0.4, "providers": ["lium"], **mkt}}}


def test_w_cmp_mkt_stage_ships_a_4090_a6000_ladder_with_caps():
    from cascade.provision.main import build_policy

    pol = build_policy(_mkt_raw())
    rungs = [(c.sku, c.marketplace_sku, c.max_price_hr) for c in pol.mkt.sku_candidates]
    assert rungs[0] == ("NVIDIA GeForce RTX 3090", "RTX3090", 0.4)
    assert [r[1] for r in rungs[1:]] == ["RTX4090", "A6000"], rungs
    assert rungs[1][0] == "NVIDIA GeForce RTX 4090" and rungs[2][0] == "NVIDIA RTX A6000"
    for _sku, _m, cap in rungs:
        # every rung's extended lease fits the deployed $5 cap
        assert 0 < cap * pol.mkt_lease_max_hours <= 5.0
    assert build_policy(_mkt_raw(candidate=[])).mkt.candidates == ()          # opt-out
    own = build_policy(_mkt_raw(candidate=[{"sku": "NVIDIA L4", "market_sku": "L4",
                                            "max_price_hr": 0.5}]))
    assert [c.sku for c in own.mkt.candidates] == ["NVIDIA L4"]               # replaced


def test_w_cmp_a_3090_drought_rents_the_next_rung(tmp_path):
    from cascade.provision.main import build_policy
    from tests.unit.test_provision_loop import FakeProvider

    class Prov(FakeProvider):
        def available(self, sku, count, *, gpus=1):
            return sku != "RTX3090"

    prov = Prov("lium", price=0.5)
    loop, _ = _mkt_loop(tmp_path, policy=build_policy(_mkt_raw()), prov=prov)
    _touch(loop)
    _cycle(loop)
    assert len(prov.launched) == 1
    from cascade.provision.state import load_state

    (inst,) = load_state(tmp_path / "state.json").instances
    assert inst.stage == "mkt" and "4090" in str(inst.sku)


# ── 5 + 6. the trainer's report wait ─────────────────────────────────────────


FROM = 9000
REF = "alfa/recipe@sha256:" + "1" * 64


def _driver(tmp_path, read):
    from cascade.funding.queue import LaneCQueue
    from cascade.shared.config.lane_c import LaneCConfig
    from cascade.shared.config.marketcast import parse_marketcast
    from cascade.trainer.lane_c_driver import LaneCDriver

    mc = replace(parse_marketcast({
        "suites": ["returns"], "assets": {"crypto": ["BTC"]},
        "frequencies": {"returns": ["1m"]}, "horizons": {"returns": ["1m"]},
        "max_steps": 1000}), from_block=FROM, report_timeout_settlements=2)
    cfg = SimpleNamespace(lane_c=LaneCConfig(from_block=FROM, targets=("mkt:returns",),
                                             legs_per_target=1), marketcast=mc)
    published: list = []

    class Ops:
        q = LaneCQueue(tmp_path / "q.json", clock=lambda: 1_000_000.0)

        def queue(self):
            return self.q

        def sign(self, m):
            return replace(m, signature="ab" * 32)

        def publish(self, key, text):
            published.append((key, text))

        def read_mkt_report(self, rid):
            return read(rid)

    return LaneCDriver(cfg, Ops(), tmp_path / "drv.json"), published


def _finish(drv, hk, n, target="mkt:returns"):
    from cascade.shared.manifest import format_trained_pointer

    drv.st.finished.append({
        "leg": {"hotkey": hk, "uid": n, "target": target, "ref": REF, "entry_ref": "",
                "started_block": FROM + n},
        "trained_pointer": format_trained_pointer(f"cascade/ckpt@sha256:{n:064x}"),
        "train_block": FROM + n})


def _settle(drv, i):
    from cascade.shared.manifest import format_trained_pointer

    return drv.settle(round_id=f"r{i}", block=FROM + i * 7200, era_index=1,
                      base_pointer=format_trained_pointer("cascade/ckpt@sha256:" + "a" * 64))


def test_w_cmp_a_round_without_marketcast_entries_is_not_awaited(tmp_path):
    reads: list = []
    drv, published = _driver(tmp_path, lambda rid: reads.append(rid))
    for i in (1, 2, 3):
        assert _settle(drv, i) == ""
    assert drv.st.mkt_await == [], "a round with no MarketCast entries is awaited"
    assert published == [] and reads == []


def test_w_cmp_a_round_with_entries_is_awaited_and_its_successor_too(tmp_path):
    drv, _ = _driver(tmp_path, lambda rid: None)
    _finish(drv, "ALFA", 1)
    assert _settle(drv, 1)
    assert [(a["rid"], a.get("entries")) for a in drv.st.mkt_await] == [("r1", 1)]
    _settle(drv, 2)                    # no entry, but r1 still awaited: r2 may carry
    assert [a["rid"] for a in drv.st.mkt_await] == ["r1", "r2"]


def test_w_cmp_a_read_error_never_ages_the_wait(tmp_path):
    from cascade.shared.hippius import StorageError

    def _down(rid):
        raise StorageError("s3_get_failed: 503 Service Unavailable")

    drv, published = _driver(tmp_path, _down)
    _finish(drv, "ALFA", 1)
    _settle(drv, 1)
    for i in range(2, 12):             # ten settlements of Hippius outage
        _settle(drv, i)
    from cascade.shared.lane_c_manifest import load_lane_c_manifest

    for _key, text in published:
        assert "absent" not in load_lane_c_manifest(text).mkt_reports.values(), (
            "a read error turned an existing report absent")
    assert next(a for a in drv.st.mkt_await if a["rid"] == "r1")["age"] == 0


def test_w_cmp_confirmed_misses_record_absent_with_an_error_alert(tmp_path, caplog):
    drv, published = _driver(tmp_path, lambda rid: None)        # confirmed 404s
    _finish(drv, "ALFA", 1)
    _settle(drv, 1)
    with caplog.at_level(logging.WARNING, logger="cascade.trainer.lane_c_driver"):
        _settle(drv, 2)
        _settle(drv, 3)
    from cascade.shared.lane_c_manifest import load_lane_c_manifest

    assert load_lane_c_manifest(published[-1][1]).mkt_reports == {"r1": "absent"}
    errs = [r for r in caplog.records if r.levelno >= logging.ERROR
            and "r1" in r.getMessage()]
    assert errs, "the first absent of a round with entries was not an ERROR"
    # the cascade-ops follower's mkt_report_absent pattern still matches the line
    pat = re.compile(r"lane C: MarketCast report of (?P<round>\S+) absent after "
                     r"(?P<n>\d+) settlement")
    assert pat.search(errs[0].getMessage())


# ── 7. the fleet restart script spares the MarketCast provisioner ────────────


def test_w_cmp_restart_provisioner_spares_the_mkt_provisioner(tmp_path):
    script = (ROOT / "deploy" / "restart_provisioner.sh").read_text(encoding="utf-8")
    fn = re.search(r"^collect_pids\(\) \{.*?^\}", script, re.S | re.M).group(0)
    pats = re.search(r"^PATTERNS=\(.*?^\)", script, re.S | re.M).group(0)
    excl = re.search(r"^EXCLUDE_RE=.*$", script, re.M)
    probe = tmp_path / "probe.sh"
    probe.write_text("#!/bin/bash\n" + pats + "\n" + (excl.group(0) if excl else "") + "\n"
                     + fn + "\ncollect_pids\n", encoding="utf-8")
    sleeper = "import time; time.sleep(60)"
    mkt = subprocess.Popen([sys.executable, "-c", sleeper, "-m", "cascade.provision.main",
                            "--config", "/root/cascade/provision.mkt.toml"])
    fleet = subprocess.Popen([sys.executable, "-c", sleeper, "-m", "cascade.provision.main",
                              "--config", "/root/cascade/provision.toml"])
    try:
        out = subprocess.run(["bash", str(probe)], capture_output=True, text=True,
                             timeout=30).stdout.split()
    finally:
        mkt.kill()
        fleet.kill()
    assert str(fleet.pid) in out
    assert str(mkt.pid) not in out, "restart_provisioner.sh would stop the mkt provisioner"


# ── 8. lane C refuses a bundle without a Lium key ────────────────────────────


def _intake_headers(tmp_path, gpu="1xRTX4090", key="sk"):
    from cascade.miner.lane_c import build_lane_c_headers
    from tests.unit.test_miner_chosen_gpu import HK, _clock, _intake, _sign
    from tests.unit.test_miner_chosen_gpu import REF as GREF
    from tests.unit.test_mkt_gpus import CAT

    intake = _intake(tmp_path, lane_c=True)
    intake.mkt_gpu_catalog = CAT
    h = build_lane_c_headers("lane-c-fund", HK, "mkt:returns", GREF, key, _sign,
                             now=_clock, gpu=gpu)
    return intake, h


def test_w_cmp_lane_c_refuses_a_bundle_without_a_lium_key(tmp_path):
    from cascade.funding.mkt_gpus import LIUM_ONLY
    from cascade.funding.providers import PayerCredential, encode_credentials

    bundle = encode_credentials([PayerCredential("shadeform", "sf")])
    intake, h = _intake_headers(tmp_path, key=bundle)
    status, body = intake.lane_c_fund(h)
    assert (status, body.get("code")) == (400, LIUM_ONLY), body
    assert "Lium" in body["message"]
    # a bundle WITH a Lium key still funds (the trainer rents on that key)
    both = encode_credentials([PayerCredential("shadeform", "sf"), PayerCredential("lium", "lk")])
    intake2, h2 = _intake_headers(tmp_path / "b", key=both)
    assert intake2.lane_c_fund(h2)[0] == 202
