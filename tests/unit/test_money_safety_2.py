"""Money safety II (the 2026-11-03 money-leak audit, owner: "close all avenues").

One test group per audit avenue: the operator runway guard and the spend
booking of every operator allocation, the queue schema, the priced keep, the
first-step watchdog, post-rent rejects, price caps, the early waste alert,
backoff + create intents + payer TTLs, the operator rent path's doomed check,
the pre-corpus would-rent verdict, the idle-king reaper and the operator
inventory. Every new knob has a loader round trip (integral floats accepted).
"""
from __future__ import annotations

import dataclasses
import json
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.funding.miner_spend import MAKE_GOOD, WASTE, MinerSpend
from cascade.funding.operator_approvals import OperatorApprovals
from cascade.funding.queue import (
    INFRA_BACKOFF_BASE_SECONDS,
    QUEUE_SCHEMA_VERSION,
    FundedQueue,
    QueueSchemaTooNew,
)
from cascade.provision.core import (
    OperatorRentsHeld,
    PodAddress,
    executor_gpu_count,
    executor_price_per_hour,
)
from cascade.provision.state import PodInstance
from cascade.shared.config import RoundConfig, load_chain_config
from cascade.shared.config.money import MONEY_KNOBS
from cascade.trainer import money_guard, money_watch
from cascade.trainer.faults import _FundedLegSkip as _Skip

REPO_TOML = Path(__file__).resolve().parents[2] / "chain.toml"
HK = "5DpyoY8pxp5xCXvErnuUDDygBT6rXQcTEBYJHBYo4rpPpGRT"
REF = "vault/direct@sha256:" + "a" * 64
NEW_KNOBS = ("harvest_keep_cost_fraction", "operator_runway_alert_hours",
             "operator_standing_pods_usd_per_hour", "operator_daily_cap_scope",
             "miner_waste_alert_usd", "king_pod_ttl_margin_seconds", "king_idle_reap_checks",
             "king_idle_check_seconds", "operator_inventory_interval_seconds",
             "operator_inventory_allow", "operator_inventory_host_files",
             "harvest_max_bytes_factor", "ops_operator_pool_enabled",
             "receipt_watch_validator_hotkey")


# ── knobs: defaults + loader parse + round trip ─────────────────────────────


def _edited(tmp_path, round_lines: str) -> Path:
    src = REPO_TOML.read_text(encoding="utf-8")
    anchor = "funded_operator_fallback = false\n"
    assert src.count(anchor) == 1
    p = tmp_path / "chain.toml"
    p.write_text(src.replace(anchor, anchor + round_lines, 1), encoding="utf-8")
    return p


def test_new_knobs_are_registered_with_their_defaults():
    rnd = RoundConfig()
    for name in NEW_KNOBS:
        assert name in MONEY_KNOBS, name
        assert getattr(rnd, name) == MONEY_KNOBS[name][0], name
    assert rnd.harvest_keep_max_seconds == 10800 and rnd.harvest_keep_cost_fraction == 0.5
    assert rnd.miner_waste_alert_usd == 15.0 and rnd.operator_runway_alert_hours == 6.0
    assert rnd.operator_daily_cap_scope == "all"
    assert rnd.funded_ready_timeout_seconds == 420.0


def test_new_knobs_round_trip_through_the_loader_integral_floats_included(tmp_path):
    p = _edited(tmp_path,
                "harvest_keep_max_seconds = 7200.0\nharvest_keep_cost_fraction = 0.25\n"
                "operator_runway_alert_hours = 8\nminer_waste_alert_usd = 20\n"
                'operator_daily_cap_scope = "miner_legs"\n'
                "king_pod_ttl_margin_seconds = 3600.0\nking_idle_reap_checks = 3.0\n"
                "king_idle_check_seconds = 600\noperator_inventory_interval_seconds = 1800.0\n"
                'operator_inventory_allow = ["paracast-bw"]\n'
                'operator_inventory_host_files = ["/root/cascade/eval_hosts.toml"]\n'
                "progress_first_step_seconds = 2700\n"
                "harvest_max_bytes_factor = 2\nops_operator_pool_enabled = true\n"
                'receipt_watch_validator_hotkey = " 5Val "\n'
                "operator_standing_pods_usd_per_hour = "
                "{ valeval-blackwell2 = 1.6, targon-eval = 2 }\n")
    r = load_chain_config(p).round
    assert (r.harvest_keep_max_seconds, r.harvest_keep_cost_fraction) == (7200, 0.25)
    assert (r.operator_runway_alert_hours, r.miner_waste_alert_usd) == (8.0, 20.0)
    assert r.operator_daily_cap_scope == "miner_legs"
    assert (r.king_pod_ttl_margin_seconds, r.king_idle_reap_checks,
            r.king_idle_check_seconds, r.operator_inventory_interval_seconds) == (3600, 3, 600,
                                                                                   1800)
    assert r.operator_inventory_allow == ("paracast-bw",)
    assert r.operator_inventory_host_files == ("/root/cascade/eval_hosts.toml",)
    assert r.operator_standing_pods_usd_per_hour == (("targon-eval", 2.0),
                                                     ("valeval-blackwell2", 1.6))
    assert r.progress_first_step_seconds == 2700
    assert (r.harvest_max_bytes_factor, r.ops_operator_pool_enabled,
            r.receipt_watch_validator_hotkey) == (2.0, True, "5Val")


@pytest.mark.parametrize("line", ['operator_daily_cap_scope = "king"',
                                  "harvest_keep_cost_fraction = 1.5",
                                  "king_idle_reap_checks = 1.5",
                                  "miner_waste_alert_usd = -1",
                                  'operator_inventory_allow = "x"',
                                  'ops_operator_pool_enabled = "yes"',
                                  "receipt_watch_validator_hotkey = 5"])
def test_bad_new_knobs_fail_loud(tmp_path, line):
    with pytest.raises(ValueError, match=line.split(" ")[0]):
        load_chain_config(_edited(tmp_path, line + "\n"))


# ── shared fakes ─────────────────────────────────────────────────────────────


def _runner(tmp_path, *, balance=1000.0, ledger=(), **round_kw):
    round_kw.setdefault("operator_max_price_per_hour", 20.0)
    round_kw.setdefault("funded_ready_timeout_seconds", 0.0)
    rnd = RoundConfig(funded_mode="required", funded_pod_sku="8xH100", epoch_blocks=900,
                      era_settlements=4, **round_kw)
    r = SimpleNamespace(cfg=SimpleNamespace(round=rnd, subnet=SimpleNamespace(netuid=91)),
                        work_root=tmp_path, _funded_leg_failures={})
    r._leg_wall_seconds = lambda sku=None: 3 * 3600.0
    r._funded_skus_for_rent = lambda: ("8xH100",)
    r._operator_balance_reader = lambda: balance
    rows = list(ledger)
    r._load_funded_ledger = lambda: rows
    r.rows = rows

    def record(hk, msg, *, miner_fault, error_class, burn):
        r._funded_leg_failures[hk] = (msg, miner_fault, error_class, burn)

    r._record_funded_failure = record
    return r


def _pod(pid, payer=""):
    return PodInstance(provider="lium", instance_id=pid, stage="funded",
                       rented_at_iso="2026-11-03T00:00:00Z", payer_hotkey=payer)


class _Prov:
    name = "lium"

    def __init__(self, ids=("pod-0",), price=None):
        self.ids, self.price, self.launched = list(ids), price, 0

    def launch(self, spec):
        self.launched += 1
        return list(self.ids)

    def price_of(self, pod_id):
        return self.price


KING = "cascade-n91-777-funded-king"


# ── 1. the operator RUNWAY guard + booking every operator allocation ────────


def test_committed_spend_counts_live_operator_pods_and_standing_pods(tmp_path):
    r = _runner(tmp_path, ledger=[_pod(KING + "-0"), _pod("cascade-n91-777-funded-abc-0",
                                                           payer=HK)],
                operator_standing_pods_usd_per_hour=(("valeval", 2.0),))
    money_guard.OperatorSpend.for_runner(r).charge("", 120.0, kind="king", pod_id=KING + "-0",
                                                   rate=10.0, hours=12.0, now=1000.0)
    c = money_guard.committed_spend(r, now=1000.0 + 2 * 3600)
    # the king: 10 $/h x 10 h left; the payer pod is the miner's; valeval 2 x 6 h
    assert c["committed_usd"] == pytest.approx(100.0 + 12.0)
    assert c["burn_usd_per_hour"] == pytest.approx(12.0)
    rw = money_guard.operator_runway(r, balance=300.0, now=1000.0 + 2 * 3600)
    assert rw["runway_hours"] == pytest.approx((300.0 - 112.0 - 50.0) / 12.0, abs=0.01)
    assert not rw["held"]
    assert money_guard.operator_runway(r, balance=150.0, now=1000.0 + 2 * 3600)["held"]


def test_runway_alert_fires_once_and_clears(tmp_path):
    r = _runner(tmp_path, operator_standing_pods_usd_per_hour=(("valeval", 10.0),))
    money_guard.runway_tick(r, 150.0)                 # (150 - 60 - 50) / 10 = 4 h < 6 h
    codes = [c["code"] for c in money_guard.health_conditions(r, {})]
    assert codes.count("operator_runway_low") == 1
    money_guard.runway_tick(r, 150.0)
    assert [c["code"] for c in money_guard.health_conditions(r, {})].count(
        "operator_runway_low") == 1
    money_guard.runway_tick(r, 5000.0)
    assert "operator_runway_low" not in {c["code"] for c in money_guard.health_conditions(r, {})}
    assert json.loads((tmp_path / money_guard.RUNWAY_FILE).read_text())["balance_usd"] == 5000.0


def test_no_new_rents_refuses_the_king_too_and_clears_on_recovery(tmp_path):
    r = _runner(tmp_path, balance=100.0, operator_daily_spend_cap_usd=0.0)
    prov = money_guard.guard_operator_provider(r, _Prov([KING + "-0"]), enforce=False)
    spec = SimpleNamespace(name_prefix=KING)
    with pytest.raises(OperatorRentsHeld, match="NO NEW OPERATOR RENTS"):
        prov.launch(spec)                         # 100 - 0 - 20 x 12 h < 50
    assert prov._provider.launched == 0           # nothing created
    assert "operator_rents_held" in {c["code"] for c in money_guard.health_conditions(r, {})}
    # distinct from the 402 breaker: the breaker stays closed
    assert money_guard.OperatorBreaker.for_runner(r).open_record() is None
    r._operator_balance_reader = lambda: 10_000.0
    assert prov.launch(spec) == [KING + "-0"]


def test_every_operator_launch_is_booked_and_a_lemon_settles_to_its_real_cost(tmp_path):
    r = _runner(tmp_path)
    leg = "cascade-n91-777-funded-abc-op"
    prov = money_guard.guard_operator_provider(r, _Prov([leg + "-0"], price=8.0))
    prov.launch(SimpleNamespace(name_prefix=leg))
    spend = money_guard.OperatorSpend.for_runner(r)
    row = spend.row_for(leg + "-0")
    assert row["kind"] == "operator_leg" and row["rate"] == 8.0
    assert row["usd"] == pytest.approx(8.0 * 3.0)            # landed price x leg hours
    settled = spend.settle(leg + "-0", now=row["at"] + 1800)
    assert settled == pytest.approx(4.0)                      # half an hour at $8/h
    # an approved miner leg's usable pod RELABELS the booking — never twice
    money_guard._leg_local(r).operator_approval = {
        "queue": "funded_queue.json", "hotkey": HK, "ref": REF, "attempt_id": "",
        "cost": 30.0, "hours": 3.0, "kind": "make_good", "book": True}
    prov.launch(SimpleNamespace(name_prefix=leg))
    money_guard.note_operator_launch(r, HK, landed_rate=8.0, pod_id=leg + "-0")
    rows = [x for x in spend.rows() if x.get("pod_id") == leg + "-0"]
    assert len(rows) == 2 and rows[-1]["kind"] == "make_good" and rows[-1]["hotkey"] == HK


def test_daily_cap_counts_the_king_and_cpu_vms_unless_scoped_to_miner_legs(tmp_path):
    r = _runner(tmp_path, balance=10_000.0, operator_daily_spend_cap_usd=250.0)
    king = money_guard.guard_operator_provider(r, _Prov([KING + "-0"]), enforce=False)
    # the king is COUNTED but never refused by the cap (the runway guards it)
    assert king.launch(SimpleNamespace(name_prefix=KING)) == [KING + "-0"]     # $240
    assert money_guard.OperatorSpend.for_runner(r).total() == pytest.approx(240.0)
    leg = "cascade-n91-777-funded-abc-op"
    legs = money_guard.guard_operator_provider(r, _Prov([leg + "-0"]))
    with pytest.raises(OperatorRentsHeld, match="daily cap"):
        legs.launch(SimpleNamespace(name_prefix=leg))           # 240 + 60 > 250
    gen = SimpleNamespace(hotkey=HK, ref=REF)
    assert "daily spend cap" in money_guard.would_rent_refusal(r, gen, operator=True)[0]
    r.cfg.round = dataclasses.replace(r.cfg.round, operator_daily_cap_scope="miner_legs")
    assert money_guard.OperatorSpend.for_runner(r).total(scope="miner_legs") == 0.0
    assert legs.launch(SimpleNamespace(name_prefix=leg)) == [leg + "-0"]


def test_cpu_vms_are_booked_and_held_with_every_operator_rent(tmp_path):
    r = _runner(tmp_path, balance=10_000.0)

    class _Cpu:
        name = "shadeform"

        def __init__(self):
            self.kw, self.gone = None, []

        def launch(self, shape, *, name, ssh_pubkey, image=None, ttl_seconds=0.0):
            self.kw = {"ttl_seconds": ttl_seconds}
            return "vm-1"

        def terminate(self, vm_id):
            self.gone.append(vm_id)

    inner = _Cpu()
    cpu = money_guard.BookedCpuProvider(r, inner)
    assert cpu.name == "shadeform"
    assert cpu.launch(SimpleNamespace(price_per_hour=1.0), name="cgen-x", ssh_pubkey="k",
                      ttl_seconds=7200) == "vm-1"
    assert inner.kw == {"ttl_seconds": 7200}
    row = money_guard.OperatorSpend.for_runner(r).row_for("vm-1")
    assert row["kind"] == "cpu_vm" and row["usd"] == pytest.approx(2.0)
    cpu.terminate("vm-1")
    assert inner.gone == ["vm-1"]
    r._operator_balance_reader = lambda: 10.0
    with pytest.raises(OperatorRentsHeld):
        cpu.launch(SimpleNamespace(price_per_hour=1.0), name="cgen-y", ssh_pubkey="k")


# ── 2. queue schema safety ───────────────────────────────────────────────────


def _queue(tmp_path, **kw):
    q = FundedQueue(tmp_path / "funded_queue.json", **kw)
    q.add(HK, REF, 10)
    return q


def test_unknown_entry_and_header_keys_survive_load_and_save(tmp_path):
    q = _queue(tmp_path)
    doc = json.loads(q.path.read_text())
    assert doc["schema_version"] == QUEUE_SCHEMA_VERSION
    doc["entries"][0]["future_field"] = {"x": 1}
    doc["published_by"] = "intake-v9"
    q.path.write_text(json.dumps(doc))
    assert q.get(HK).extra == {"future_field": {"x": 1}}
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    q.requeue(HK, error="sold out", error_class="no_capacity", burn_attempt=False)
    doc = json.loads(q.path.read_text())
    assert doc["entries"][0]["future_field"] == {"x": 1}
    assert doc["published_by"] == "intake-v9"
    assert q.get(HK) == q.get(HK)                 # extra never breaks equality


def test_a_newer_schema_on_disk_refuses_the_save(tmp_path, caplog):
    q = _queue(tmp_path)
    doc = json.loads(q.path.read_text())
    doc["schema_version"] = QUEUE_SCHEMA_VERSION + 1
    q.path.write_text(json.dumps(doc))
    before = q.path.read_text()
    with pytest.raises(QueueSchemaTooNew):
        q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    assert q.path.read_text() == before          # nothing rewritten
    assert any(rec.levelname == "CRITICAL" for rec in caplog.records)


# ── 3. the priced keep, R2 loud, the lost-run retrain is operator-billed ────


def test_keep_budget_is_half_the_trained_run_and_capped(tmp_path):
    from cascade.trainer.harvest_guard import keep_budget_seconds

    r = _runner(tmp_path)
    assert keep_budget_seconds(r) == pytest.approx(0.5 * 3 * 3600)
    assert keep_budget_seconds(r, {"trained_seconds": 10 * 3600}) == 10800   # the 3 h cap
    r.cfg.round = RoundConfig(harvest_keep_cost_fraction=0.0)
    assert keep_budget_seconds(r) == 0.0


def test_a_kept_pod_is_released_after_the_priced_keep(tmp_path, monkeypatch):
    from cascade.trainer import harvest_guard
    from cascade.trainer.harvest_guard import HarvestDiskFault, handle_our_side_fault

    r = _runner(tmp_path)
    gen = SimpleNamespace(hotkey=HK, ref=REF, uid=1)
    t = [1_000_000.0]
    monkeypatch.setattr(harvest_guard.time, "time", lambda: t[0])
    exc = HarvestDiskFault("ENOSPC")
    assert handle_our_side_fault(r, gen, exc, pod_id="p-0", payer=True, trained=True,
                                 released=False)
    t[0] += 0.5 * 3 * 3600 - 60
    assert handle_our_side_fault(r, gen, exc, pod_id="p-0", payer=True, trained=True,
                                 released=False)
    t[0] += 120
    assert not handle_our_side_fault(r, gen, exc, pod_id="p-0", payer=True, trained=True,
                                     released=False)
    assert "miner_billed_loss" in {c["code"] for c in money_guard.health_conditions(r, {})}


def test_no_r2_is_said_loudly_before_the_pod_is_kept(tmp_path, caplog):
    from cascade.trainer.harvest_guard import HarvestDiskFault, guarded_harvest

    r = _runner(tmp_path, harvest_disk_retries=0, expected_checkpoint_gb=0.0,
                min_free_disk_gb=0.0)

    def attempt():
        raise OSError(28, "No space left on device")

    with pytest.raises(HarvestDiskFault):
        guarded_harvest(r, tmp_path / "dest", attempt, label="hk/job", r2_fallback=None)
    assert "harvest_no_r2" in {c["code"] for c in money_guard.health_conditions(r, {})}
    assert any("R2 is NOT configured" in rec.getMessage() for rec in caplog.records)


# ── 4. the first-step watchdog with progress_first_step_seconds = 2700 ──────


def test_first_step_watchdog_fires_at_2700_s():
    from cascade.trainer.cost_guards import FIRST_STEP_ABORT, leg_watchdog

    t = [0.0]
    r = SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig(progress_first_step_seconds=2700)))
    w = leg_watchdog(r, gen=SimpleNamespace(hotkey=HK), pod=SimpleNamespace(instance_id="p-0"),
                     contract=SimpleNamespace(training_steps=1000, max_train_seconds=3600))
    assert w is not None
    w.clock = lambda: t[0]
    assert w(None) is None
    t[0] = 2699.0
    assert w({}) is None
    t[0] = 2700.0
    v = w({})
    assert v and FIRST_STEP_ABORT in v and w.kind == "first_step"


def _aborting(msg):
    from cascade.trainer.remote import LegAborted

    class D:
        def dispatch(self, host, **kw):
            raise LegAborted(msg, returncode=254)

    return D()


def test_a_host_hang_is_an_unburned_operator_fault_and_quarantines_the_host(monkeypatch):
    from cascade.trainer import host_preflight, runner_funded_legs
    from cascade.trainer.cost_guards import FIRST_STEP_ABORT
    from cascade.trainer.faults import _OperatorFault, classify_leg_exception

    quarantined = []
    monkeypatch.setattr(host_preflight, "quarantine_host_ip",
                        lambda ip, why: quarantined.append(ip))
    host = SimpleNamespace(host="1.2.3.4")
    with pytest.raises(runner_funded_legs.HostHangFault) as ei:
        runner_funded_legs._dispatch_watched(_aborting(f"{FIRST_STEP_ABORT}: x"), host,
                                             corpus_files=["s0"], hotkey=HK)
    assert quarantined == ["1.2.3.4"] and isinstance(ei.value, _OperatorFault)
    # on-pod generation: still unburned, but the host is not blamed
    with pytest.raises(_OperatorFault) as ei2:
        runner_funded_legs._dispatch_watched(_aborting(f"{FIRST_STEP_ABORT}: x"), host,
                                             hotkey=HK)
    assert not isinstance(ei2.value, runner_funded_legs.HostHangFault)
    assert quarantined == ["1.2.3.4"]
    r = SimpleNamespace()
    gen = SimpleNamespace(hotkey=HK, ref=REF)
    seeds = SimpleNamespace(base_seed=7)
    # a generator fault before step 1 (the worker's rc 3) stays the miner's
    assert classify_leg_exception(r, gen, seeds, RuntimeError("corpus"), rc=3, text="",
                                  stalled_before=True, shards_leg=False)[1] == "generator"


# ── 5. post-rent rejects ─────────────────────────────────────────────────────


def test_a_short_pod_quarantines_its_host_ip(monkeypatch):
    from cascade.provision import core
    from cascade.trainer import remote
    from cascade.trainer.runner_funded_pods import _FundedPodMixin

    seen = []
    monkeypatch.setattr(core, "record_host_quarantine",
                        lambda ip, why, **kw: seen.append((ip, kw.get("host_fault"))))
    monkeypatch.setattr(remote, "probe_gpu_count", lambda host: (6, ""))
    why = _FundedPodMixin._host_gpu_shortfall(SimpleNamespace(), SimpleNamespace(host="9.9.9.9"),
                                              "8xH200", "pod x")
    assert why.startswith("short pod") and seen == [("9.9.9.9", True)]


def test_listings_are_filtered_on_the_exact_and_free_gpu_count(monkeypatch):
    from cascade.provision import core

    prov = core.LiumProvider()
    rows = [{"id": "a", "gpu_count": 8, "price_per_hour": 20.0},
            {"id": "b", "price_per_hour": 20.0},                          # count unknown
            {"id": "c", "gpu_count": 8, "available_gpu_count": 6, "price_per_hour": 20.0},
            {"id": "d", "gpu_count": 6, "price_per_hour": 15.0}]
    monkeypatch.setattr(prov, "_cli", lambda args: SimpleNamespace(stdout=json.dumps(rows)))
    monkeypatch.setattr(prov, "_steer_hosts", lambda sku, shaped: shaped)
    monkeypatch.setattr(core, "cooling_executors", lambda: set())
    assert [e["id"] for e in prov._list_executors("8xH200")] == ["a"]


def test_ready_timeout_default_is_420():
    assert RoundConfig().funded_ready_timeout_seconds == 420.0


def test_post_rent_rejects_are_operator_make_good_not_miner_waste(tmp_path):
    from cascade.trainer import miner_spend_guard as g

    r = _runner(tmp_path)
    q = _queue(tmp_path)
    r._funded_queue = lambda: q
    g.spend_start(r, HK, "lemon-0", rate=36.0)
    led = MinerSpend.beside(q.path)
    with led._locked() as recs:                   # it lived 10 minutes
        recs[led._key(HK, REF)]["pods"][0]["start"] -= 600
        led._save(recs)
    g.spend_end(r, HK, "lemon-0", outcome=MAKE_GOOD, reason="post-rent reject: short pod")
    s = led.summary(HK, ref=REF, funded_at=q.get(HK).funded_at)
    assert s.waste_usd == 0.0 and s.make_good_usd == pytest.approx(6.0, abs=0.1)
    owed = [c for c in money_guard.health_conditions(r, {}) if c["code"] == "miner_make_good_owed"]
    assert len(owed) == 1 and owed[0]["make_good_usd"] == pytest.approx(6.0, abs=0.1)


# ── 6. price caps ────────────────────────────────────────────────────────────


def test_a_per_gpu_listing_is_priced_per_pod_never_per_gpu():
    assert executor_price_per_hour({"price_per_hour": 30.0, "price_per_gpu_hour": 4.0}) == 30.0
    assert executor_price_per_hour({"price_per_gpu_hour": 4.0, "gpu_count": 8}) == 32.0
    assert executor_price_per_hour({"price_per_gpu_hour": 4.0}) is None       # unpriced
    assert executor_price_per_hour({"price_per_gpu_hour": 4.0}, gpu_count=8) == 32.0
    assert executor_gpu_count({"machine": {"gpu_count": 8}}) == 8


def test_a_chosen_gpu_without_max_price_takes_the_operator_cap():
    from cascade.funding.gpu_choice import (
        GPU_HEADER,
        GPU_MAX_PRICE_HEADER,
        GpuCatalog,
        parse_gpu_headers,
    )

    cat = GpuCatalog(skus=("8xH100", "8xH200"), default_max_price_per_hour=18.0)
    choice, _seg, err = parse_gpu_headers({GPU_HEADER: "8xH200"}, cat)
    assert err is None and choice.max_price_per_hour == 18.0
    choice, _seg, err = parse_gpu_headers({GPU_HEADER: "8xH200", GPU_MAX_PRICE_HEADER: "0"}, cat)
    assert err is None and choice.max_price_per_hour == 0.0                 # explicit: uncapped
    cfg = SimpleNamespace(round=RoundConfig(funded_max_price_per_hour=17.0),
                          throne_contracts=lambda: [])
    assert GpuCatalog.from_config(cfg).default_max_price_per_hour == 17.0


def test_the_allow_fallback_path_uses_the_lower_of_the_miner_and_operator_caps():
    from cascade.trainer.runner_capacity import _FundedCapacityMixin

    r = SimpleNamespace(_funded_price_caps=lambda: {"max_price_per_hour": 20.0,
                                                    "max_leg_cost_usd": 100.0})
    f = _FundedCapacityMixin._fallback_price_caps
    assert f(r, SimpleNamespace(max_price_per_hour=12.0))["max_price_per_hour"] == 12.0
    assert f(r, SimpleNamespace(max_price_per_hour=30.0))["max_price_per_hour"] == 20.0
    assert f(r, SimpleNamespace(max_price_per_hour=0.0))["max_price_per_hour"] == 20.0
    assert f(r, SimpleNamespace(max_price_per_hour=12.0))["max_leg_cost_usd"] == 100.0


# ── 7. the early waste alert, leaked records kept open ──────────────────────


def test_the_first_infra_loss_over_15_files_the_make_good_once(tmp_path):
    from cascade.trainer import miner_spend_guard as g

    r = _runner(tmp_path)
    q = _queue(tmp_path)
    r._funded_queue = lambda: q
    led = MinerSpend.beside(q.path)
    for pid, minutes in (("p1-0", 10), ("p2-0", 30)):
        g.spend_start(r, HK, pid, rate=36.0)
        with led._locked() as recs:
            for p in recs[led._key(HK, REF)]["pods"]:
                if p["pod_id"] == pid:
                    p["start"] -= minutes * 60
            led._save(recs)
        g.spend_end(r, HK, pid, outcome=WASTE, reason="lost")
        pend = OperatorApprovals.for_work_root(tmp_path).pending()
        if pid == "p1-0":
            assert pend == []                       # $6 < $15
    (req,) = OperatorApprovals.for_work_root(tmp_path).pending()
    assert req["kind"] == "spend_cap_make_good" and req["miner_lost_usd"] >= 15.0
    OperatorApprovals.for_work_root(tmp_path).clear("funded_queue.json", HK)
    g.spend_start(r, HK, "p3-0", rate=36.0)
    g.spend_end(r, HK, "p3-0", outcome=WASTE, reason="lost")
    assert OperatorApprovals.for_work_root(tmp_path).pending() == []        # filed ONCE


# ── 8. backoff, create intents, payer TTLs ──────────────────────────────────


def test_an_unburned_infra_requeue_backs_off(tmp_path):
    t = [1000.0]
    q = _queue(tmp_path, clock=lambda: t[0])
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    q.requeue(HK, error="create_unresolved: prior rental is not resolved", error_class="infra",
              burn_attempt=False)
    e = q.get(HK)
    assert e.not_before == 1000.0 + INFRA_BACKOFF_BASE_SECONDS and e.attempts == 0
    assert not q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=2)
    # market waits keep their cadence
    t[0] = e.not_before + 1
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=3)
    q.requeue(HK, error="sold out", error_class="no_capacity", burn_attempt=False)
    assert q.get(HK).not_before == 0.0


def test_a_confirmed_gone_pod_resolves_its_create_intents():
    from cascade.funding.providers import begin_create, create_intents
    from cascade.provision.funded import resolve_scope_intents

    vault = SimpleNamespace(dir=None)
    begin_create(vault, "tok-a", scope="cascade-n91-1-funded-abc", provider="lium")
    begin_create(vault, "tok-b", scope="cascade-n91-1-funded-abc", provider="lium")
    begin_create(vault, "tok-c", scope="cascade-n91-1-funded-zzz", provider="lium")
    assert resolve_scope_intents(vault, "cascade-n91-1-funded-abc") == 2
    assert create_intents(vault, "cascade-n91-1-funded-abc") == []
    assert len(create_intents(vault, "cascade-n91-1-funded-zzz")) == 1


def test_payer_keys_are_refreshed_while_they_have_ledger_rows_or_intents(tmp_path, monkeypatch):
    from cascade.funding import providers

    refreshed, leases = [], []
    vault = SimpleNamespace(dir=None, refresh=lambda hk: refreshed.append(hk) or True)
    providers.begin_create(vault, "tok-x", scope="cascade-n91-1-funded-orphan", provider="lium")
    monkeypatch.setattr(providers, "lease_records",
                        lambda v, hotkey=None: iter([("tok-x", "hkX", None)]))
    monkeypatch.setattr(providers, "_lease_vault",
                        lambda v: SimpleNamespace(refresh=lambda t: leases.append(t)))
    r = SimpleNamespace(_payer_vault=lambda: vault,
                        _load_funded_ledger=lambda: [_pod("cascade-n91-1-funded-abc-0",
                                                          payer="hkA")])
    assert money_watch.refresh_payer_ttls(r) == 2
    assert sorted(refreshed) == ["hkA", "hkX"] and leases == ["tok-x"]


# ── 9 / 10. the operator rent path: doomed check; the pre-corpus verdict ────


def test_would_rent_refuses_without_side_effects(tmp_path, monkeypatch):
    from cascade.trainer import cost_guards

    r = _runner(tmp_path)
    gen = SimpleNamespace(hotkey=HK, ref=REF)
    assert money_guard.would_rent_refusal(r, gen, operator=True) == ("", "")
    money_guard.OperatorBreaker.for_runner(r).trip("402")
    why, cls = money_guard.would_rent_refusal(r, gen, operator=True)
    assert "circuit breaker" in why and cls == "operator_blocked"
    assert money_guard.would_rent_refusal(r, gen, operator=False) == ("", "")
    (tmp_path / money_guard.BREAKER_FILE).unlink()
    r._operator_balance_reader = lambda: 60.0
    why, cls = money_guard.would_rent_refusal(r, gen, operator=True)
    assert "no new operator rents" in why and cls == "operator_blocked"
    monkeypatch.setattr(cost_guards, "doomed_leg_reason", lambda runner, skus: "doomed leg: x")
    assert money_guard.would_rent_refusal(r, gen, operator=False) == ("doomed leg: x",
                                                                       "no_capacity")
    assert money_guard.OperatorSpend.for_runner(r).rows() == []             # nothing booked


def test_the_operator_rent_path_checks_doomed_legs():
    import inspect

    from cascade.trainer.runner_operator_pods import _OperatorPodMixin

    src = inspect.getsource(_OperatorPodMixin._rent_operator_leg_host)
    assert "doomed_leg_reason" in src
    src = inspect.getsource(_OperatorPodMixin._run_operator_rented_leg)
    assert src.index("would_rent_refusal") < src.index("leg_corpus(")


# ── 11. the idle-king reaper ─────────────────────────────────────────────────


def _reaper_runner(tmp_path, keeps):
    in_use = KING + "-0"
    backup = "cascade-n91-777-funded-king-replica-1-0"
    stale = "cascade-n91-555-funded-king-0"
    r = _runner(tmp_path, ledger=[_pod(in_use), _pod(backup), _pod(stale)],
                king_idle_reap_checks=2, king_idle_check_seconds=900)
    r.__dict__["_funded_king_host"] = SimpleNamespace(allocation_id=in_use)
    r.__dict__["_rolling_sched"] = SimpleNamespace(state=SimpleNamespace(
        current=SimpleNamespace(base_seed=777), next=None))
    r._operator_pod_provider = lambda pod: object()
    r._sweep_keeps_pod = lambda adapter, pid, *, what: keeps(pid)
    r.torn = []
    r._teardown_operator_pod = lambda pod: r.torn.append(pod.instance_id)
    return r, in_use, backup, stale


def test_the_reaper_never_touches_the_king_in_use_and_reaps_idle_backups(tmp_path):
    r, in_use, backup, stale = _reaper_runner(tmp_path, keeps=lambda pid: False)
    assert {p.instance_id for p in money_watch.reap_candidates(r)} == {backup, stale}
    assert money_watch.king_reaper_tick(r, now=1000.0) == []         # idle 1/2
    assert money_watch.king_reaper_tick(r, now=1500.0) == []         # not due yet
    assert sorted(money_watch.king_reaper_tick(r, now=1900.0)) == sorted([backup, stale])
    assert in_use not in r.torn
    assert "king_pod_reaped" in {c["code"] for c in money_guard.health_conditions(r, {})}


def test_a_busy_pod_resets_its_idle_count(tmp_path):
    state = {"busy": False}
    r, _in_use, backup, stale = _reaper_runner(
        tmp_path, keeps=lambda pid: pid == backup and state["busy"] or pid == stale)
    money_watch.king_reaper_tick(r, now=1000.0)
    state["busy"] = True
    money_watch.king_reaper_tick(r, now=2000.0)
    state["busy"] = False
    assert money_watch.king_reaper_tick(r, now=3000.0) == []          # count restarted
    assert money_watch.king_reaper_tick(r, now=4000.0) == [backup]


def test_the_king_ttl_is_one_era_plus_the_margin():
    from cascade.trainer.runner_operator_pods import king_pod_ttl_seconds

    r = SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig(
        epoch_blocks=900, era_settlements=4, king_pod_ttl_margin_seconds=3600)))
    assert king_pod_ttl_seconds(r) == 900 * 4 * 12 + 3600
    r.cfg.round = RoundConfig()
    assert king_pod_ttl_seconds(r) == 0.0


# ── 12. the operator-account inventory ──────────────────────────────────────


def test_inventory_alerts_once_per_unknown_pod_and_never_deletes(tmp_path):
    hosts = tmp_path / "eval_hosts.toml"
    hosts.write_text('[[host]]\nname = "valeval"\nhost = "86.125.172.23"\n')
    r = _runner(tmp_path, ledger=[_pod("cascade-n91-777-funded-abc-op-0")],
                operator_inventory_allow=("paracast-ok",),
                operator_inventory_host_files=(str(hosts),))
    rows = [{"name": "cascade-n91-777-funded-abc-op-0", "status": "RUNNING"},
            {"name": "eager-wolf-e5", "status": "RUNNING",
             "ssh_cmd": "ssh root@86.125.172.23 -p 41020"},
            {"name": "paracast-ok", "status": "RUNNING"},
            {"name": "paracast-bw", "status": "RUNNING", "price_per_hour": 3.2,
             "ssh_cmd": "ssh root@5.5.5.5 -p 22"}]
    calls = []

    def lister():
        calls.append(1)
        return rows

    got = money_watch.inventory_tick(r, now=10_000.0, lister=lister)
    assert [u["name"] for u in got] == ["paracast-bw"] and got[0]["usd_per_hour"] == 3.2
    unknown = [c for c in money_guard.health_conditions(r, {})
               if c["code"] == "operator_unknown_pod"]
    assert len(unknown) == 1 and "paracast-bw" in unknown[0]["message"]
    assert money_watch.inventory_tick(r, now=10_100.0, lister=lister) == []   # not due
    assert calls == [1]


# ── 13. an unreachable pod last seen finished-unharvested is kept ───────────


def test_an_unreachable_pod_last_seen_finished_is_kept_for_the_priced_keep(tmp_path):
    from cascade.trainer.remote import RemoteHost
    from cascade.trainer.runner_funded_pods import _FundedPodMixin

    probe_ok = "__CASCADE_POD_PROBE__"
    answers = {"out": types.SimpleNamespace(
        returncode=0, stdout=f"FINISHED:60:_train_work/_dispatch/x\n{probe_ok}\n", stderr="")}

    class R(_FundedPodMixin):
        def __init__(self):
            self.cfg = SimpleNamespace(round=RoundConfig(funded_mode="required",
                                                         funded_pods="rent"),
                                       subnet=SimpleNamespace(netuid=91))
            self.work_root = tmp_path
            self._funded_ledger_lock = threading.Lock()
            self._sweep_ssh_runner = lambda argv, timeout, stdin=None: answers["out"]
            self.t = [2_000_000.0]
            self._sweep_clock = lambda: self.t[0]

        def _funded_pod_profile(self):
            return RemoteHost(name="profile", host="127.0.0.1", user="root", key_path="/k",
                              workdir="/root/cascade", stage="final", profile_only=True)

        def _leg_wall_seconds(self, sku=None):
            return 4 * 3600.0

    class A:
        def get_ip(self, pod_id):
            return PodAddress(ip="10.0.0.9", ssh_port=2222)

    r = R()
    t0 = r.t[0]
    assert r._sweep_keeps_pod(A(), "pod-f", what="t")                  # finished, unharvested
    answers["out"] = types.SimpleNamespace(returncode=255, stdout="", stderr="refused")
    assert r._sweep_keeps_pod(A(), "pod-f", what="t")                  # unreachable: the grace
    r.t[0] = t0 + r.SWEEP_UNREACHABLE_GRACE_SECONDS + 10               # past the 30 min grace
    assert r._sweep_keeps_pod(A(), "pod-f", what="t")                  # priced keep: 0.5 x 4 h
    r.t[0] = t0 + 2 * 3600 + 20
    assert not r._sweep_keeps_pod(A(), "pod-f", what="t")              # the keep is spent


# ── the operator CLI resolves the work root independent of the CWD ──────────


def test_operator_cli_resolves_the_work_root_from_anywhere(tmp_path, monkeypatch, capsys):
    from cascade.trainer import operator_legs_cli as olc

    deploy = (tmp_path / "cascade").resolve()
    (deploy / "_train_work").mkdir(parents=True)
    (deploy / "chain.toml").write_text("")
    elsewhere = tmp_path / "home"
    elsewhere.mkdir()
    # resolution order: --work-root, --chain-toml, ./chain.toml, $CASCADE_ROOT, the deploy dir
    assert olc.resolve_work_root(deploy / "x")[0] == (deploy / "x").resolve()
    assert olc.resolve_work_root(chain_toml=deploy / "chain.toml")[0] == deploy / "_train_work"
    assert olc.resolve_work_root(cwd=deploy, env={})[0] == deploy / "_train_work"
    assert olc.resolve_work_root(cwd=elsewhere, env={"CASCADE_ROOT": str(deploy)})[0] == \
        deploy / "_train_work"
    assert olc.resolve_work_root(cwd=elsewhere, env={}, deploy_root=deploy)[0] == \
        deploy / "_train_work"
    # run from another directory: the pending leg under the deployment is found
    q = FundedQueue(deploy / "_train_work" / "funded_queue.json")
    q.add(HK, REF, 10)
    OperatorApprovals.for_work_root(deploy / "_train_work").request(
        "funded_queue.json", HK, ref=REF, attempt_id="", queue_path=str(q.path),
        kind="make_good", funded_at=q.get(HK).funded_at)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("CASCADE_ROOT", str(deploy))
    assert olc.main(["list-operator-legs"]) == 0
    out = capsys.readouterr().out
    assert f"queue: {deploy / '_train_work' / 'funded_queue.json'}" in out
    assert f"AWAITING  {HK}" in out
    assert olc.main(["approve-operator-leg", HK]) == 0
    assert "APPROVED" in capsys.readouterr().out


def test_operator_cli_fails_loud_when_the_queue_is_missing(tmp_path, capsys):
    from cascade.trainer import operator_legs_cli as olc

    (tmp_path / "wr").mkdir()
    for argv in (["list-operator-legs"], ["approve-operator-leg", HK],
                 ["deny-operator-leg", HK], ["resume-operator-leg", HK]):
        assert olc.main([*argv, "--work-root", str(tmp_path / "wr")]) == 2
        err = capsys.readouterr().err
        assert "no funding queue at" in err and str(tmp_path / "wr") in err
    assert olc.main(["rent-status", "--work-root", str(tmp_path / "missing")]) == 2
    assert olc.main(["rent-status", "--work-root", str(tmp_path / "wr")]) == 0


# ══ second audit (rc19 = main 1cc20cb) ══════════════════════════════════════


# ── A: a miner-caused payer-pod loss is never our fault ─────────────────────


def _payer_runner(tmp_path, *, status="RUNNING", balance=50.0):
    vault = SimpleNamespace(get=lambda hk: "sk-payer")

    class _P:
        def _pod(self, pod_id):
            return None if status is None else {"status": status}

    return SimpleNamespace(_payer_vault=lambda: vault, _payer_pod_provider=lambda v, pod: _P(),
                           _payer_balance_reader=lambda key: balance)


def test_payer_side_evidence_proves_a_miner_cause():
    from cascade.trainer.runner_funded_legs import payer_loss_evidence

    pod = _pod("p-0", payer=HK)
    assert payer_loss_evidence(_payer_runner(None), pod)["miner"] is False
    assert payer_loss_evidence(_payer_runner(None, balance=0.0), pod)["miner"] is True
    assert payer_loss_evidence(_payer_runner(None, status="STOPPED"), pod)["miner"] is True
    ev = payer_loss_evidence(_payer_runner(None, status=None), pod)       # absent: not proof
    assert ev["pod_status"] == "absent" and ev["miner"] is False


def test_a_watchdog_abort_with_a_foreign_gpu_hog_is_the_miners(monkeypatch):
    from cascade.trainer import host_preflight, remote, runner_funded_legs
    from cascade.trainer.cost_guards import THROUGHPUT_ABORT

    quarantined = []
    monkeypatch.setattr(host_preflight, "quarantine_host_ip",
                        lambda ip, why: quarantined.append(ip))
    monkeypatch.setattr(remote, "probe_foreign_gpu_procs",
                        lambda host: (["4242:python hog.py"], ""))
    with pytest.raises(runner_funded_legs.PayerPodInterference, match="hog.py"):
        runner_funded_legs._dispatch_watched(_aborting(f"{THROUGHPUT_ABORT}: slow"),
                                             SimpleNamespace(host="1.2.3.4"), payer_pod=True,
                                             hotkey=HK)
    assert quarantined == []
    monkeypatch.setattr(remote, "probe_foreign_gpu_procs", lambda host: ([], ""))
    from cascade.trainer.faults import _OperatorFault

    with pytest.raises(_OperatorFault):
        runner_funded_legs._dispatch_watched(_aborting(f"{THROUGHPUT_ABORT}: slow"),
                                             SimpleNamespace(host="1.2.3.4"), payer_pod=True,
                                             hotkey=HK)
    assert quarantined == ["1.2.3.4"]


def test_foreign_gpu_processes_are_told_from_ours():
    from cascade.trainer.remote import parse_foreign_gpu_procs

    out = ("GPU:11:python -m cascade.trainer.worker --role challenger\n"
           "GPU:12:/usr/bin/python3 /root/miner/hog.py\n__CASCADE_GPU_PROCS__\n")
    assert parse_foreign_gpu_procs(out) == ["12:/usr/bin/python3 /root/miner/hog.py"]


def test_an_approval_request_carries_the_failure_and_the_evidence(tmp_path):
    r = _runner(tmp_path)
    q = _queue(tmp_path)
    r._funded_queue = lambda: q
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    q.requeue(HK, error="pod lost mid-run", error_class="infra", burn_attempt=False)
    assert q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=2) or True
    money_guard._leg_local(r).loss_evidence = {"pod_status": "absent", "miner": False}
    with pytest.raises(_Skip):
        money_guard.operator_leg_gate(r, SimpleNamespace(hotkey=HK, ref=REF, uid=7),
                                      kind="make_good", rent=True)
    (req,) = OperatorApprovals.for_work_root(tmp_path).pending()
    assert req["last_error"] == "pod lost mid-run" and req["error_class"] == "infra"
    assert req["evidence"] == {"pod_status": "absent", "miner": False}


# ── B: a bounded, contained ssh harvest ─────────────────────────────────────


def test_the_harvest_dir_must_be_the_workers_layout():
    from cascade.trainer.faults import CheckpointUnverifiable
    from cascade.trainer.loop import _check_harvest_dir, harvest_max_bytes

    _check_harvest_dir("_train_work/777/toto2-4m/challenger/checkpoint")
    _check_harvest_dir("/root/cascade/_train_work/mkt-5abc-12")
    for bad in ("/", "/root", "_train_work", "_train_work/../../etc", ""):
        with pytest.raises(CheckpointUnverifiable):
            _check_harvest_dir(bad)
    assert harvest_max_bytes(RoundConfig(expected_checkpoint_gb=15.0)) == int(15 * 1.2 * 2 ** 30)
    assert harvest_max_bytes(RoundConfig(harvest_max_bytes_factor=0.0)) is None


# ── C: withdraw of a running leg ─────────────────────────────────────────────


def test_withdraw_of_an_in_flight_entry_keeps_its_key_and_never_resurrects(tmp_path):
    from tests.unit.test_funding_intake import HK as IHK
    from tests.unit.test_funding_intake import REF as IREF
    from tests.unit.test_funding_intake import make_intake

    intake, _ = make_intake(tmp_path)
    intake.fund({"X-Miner-Hotkey": IHK, "X-Commit-Ref": IREF, "X-Lium-Api-Key": "sk"})
    assert intake.queue.mark_in_flight(IHK, IREF, target_boundary=1, era_index=1,
                                       started_block=1)
    status, body = intake.withdraw({"X-Miner-Hotkey": IHK, "X-Commit-Ref": IREF})
    assert (status, body["code"]) == (409, "not_queued")
    assert intake.vault.get(IHK) == "sk" and intake.queue.get(IHK).status == "in_flight"
    q = _queue(tmp_path / "w")
    assert q.withdraw(HK)
    assert not q.requeue(HK, error="x", error_class="infra", burn_attempt=False)
    assert q.get(HK).status == "withdrawn"
    r = _runner(tmp_path / "w")
    r._funded_queue = lambda: q
    with pytest.raises(_Skip):
        money_guard.operator_leg_gate(r, SimpleNamespace(hotkey=HK, ref=REF, uid=1),
                                      kind="expired_key", rent=True)
    assert OperatorApprovals.for_work_root(tmp_path / "w").pending() == []


# ── D: our validator's rejected receipt is seen ─────────────────────────────


def test_a_rejected_receipt_of_our_settlement_raises_one_alert(tmp_path):
    r = _runner(tmp_path, receipt_watch_validator_hotkey="5Val")
    money_watch.note_published_settlement(r, SimpleNamespace(
        round_id="r-9", entries=[SimpleNamespace(miner_hotkey=HK)]))
    doc = {"round_id": "r-9", "status": "rejected", "reject_reason": "era_init_mismatch: x"}
    got = money_watch.rejected_receipt_tick(r, now=1000.0, reader=lambda: json.dumps(doc))
    assert got["hotkeys"] == [HK]
    (a,) = [c for c in money_guard.health_conditions(r, {}) if c["code"] == "settlement_rejected"]
    assert "era_init_mismatch" in a["message"]
    other = {"round_id": "theirs", "status": "rejected"}
    assert money_watch.rejected_receipt_tick(r, now=2000.0,
                                             reader=lambda: json.dumps(other)) is None


# ── G: approvals have a default cost ceiling ────────────────────────────────


def test_an_approval_defaults_to_a_ceiling_and_a_dearer_landing_needs_reapproval(tmp_path):
    store = OperatorApprovals.for_work_root(tmp_path)
    store.request("funded_queue.json", HK, ref=REF, attempt_id="a1", projected_cost_usd=40.0,
                  kind="make_good")
    rec = store.decide(HK, approve=True)
    assert rec["max_cost_usd"] == pytest.approx(50.0)
    r = _runner(tmp_path)
    money_guard._leg_local(r).operator_approval = {
        "queue": "funded_queue.json", "hotkey": HK, "ref": REF, "attempt_id": "a1",
        "cost": 40.0, "hours": 3.0, "kind": "make_good", "book": True, "reservation": ""}
    assert money_guard.note_operator_launch(r, HK, landed_rate=20.0, pod_id="p-0") is False
    (req,) = store.pending()
    assert req["projected_cost_usd"] == pytest.approx(60.0)


# ── H: the daily cap is reserved atomically over a rolling 24 h ─────────────


def test_reservations_are_atomic_and_the_window_rolls(tmp_path):
    spend = money_guard.OperatorSpend(tmp_path / "operator_spend.json")
    a = spend.reserve_if_room("hkA", 60.0, cap=100.0, kind="reservation", now=1000.0)
    assert a and spend.reserve_if_room("hkB", 60.0, cap=100.0, kind="reservation",
                                       now=1000.0) is None
    # the same leg re-gated REPLACES its reservation
    assert spend.reserve_if_room("hkA", 70.0, cap=100.0, kind="reservation", now=1001.0)
    assert spend.total(now=1002.0) == pytest.approx(70.0)
    assert spend.total(now=1001.0 + spend.RESERVATION_TTL_SECONDS + 1) == 0.0     # expired
    spend.charge("", 50.0, kind="operator_leg", pod_id="p-0", rate=10.0, hours=5.0, now=2000.0)
    assert spend.total(now=2000.0 + 23 * 3600) == pytest.approx(50.0)
    assert spend.total(now=2000.0 + 25 * 3600) == 0.0                              # rolled off
    assert spend.settle("p-0", now=2000.0 + 8 * 3600, true_up=True) == pytest.approx(80.0)


# ── I: a live operator pod is never torn down on one failed probe ───────────


def test_adoption_retries_and_never_kills_a_pod_with_a_live_worker():
    import inspect

    from cascade.trainer.runner_operator_pods import _OperatorPodMixin

    src = inspect.getsource(_OperatorPodMixin._rent_operator_leg_host)
    assert "ADOPT_PROBE_TRIES" in src
    assert src.index("_operator_pod_worker_alive") < src.index('_lemon("adopted operator leg')
    assert _OperatorPodMixin.ADOPT_PROBE_TRIES >= 3


# ── J: mismatch evidence copies are reclaimable when old / many ─────────────


def test_evidence_copies_are_kept_young_and_few(tmp_path):
    import os

    from cascade.shared.disk_space import EVIDENCE_KEEP_SECONDS, _evidence_protected

    seed = tmp_path / "777"
    seed.mkdir()
    proofs = []
    for i in range(3):
        pf = seed / f"job{i}.unverifiable.json"
        pf.write_text("{}")
        os.utime(pf, (1000.0 + i, 1000.0 + i))
        proofs.append(pf)
    now = 1010.0
    assert not _evidence_protected(tmp_path, proofs[0], now=now)       # beyond the newest 2
    assert _evidence_protected(tmp_path, proofs[2], now=now)
    assert not _evidence_protected(tmp_path, proofs[2], now=1002.0 + EVIDENCE_KEEP_SECONDS + 1)


# ── K: an unfunded rotated king key is refused ──────────────────────────────


def test_an_unfunded_king_key_does_not_lift_the_slash(tmp_path, monkeypatch):
    import requests

    from cascade.trainer.runner_rolling import _RollingMixin

    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(ok=True))
    vault = SimpleNamespace(get_shared=lambda hk: "sk-king")
    r = SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig(funded_max_price_per_hour=20.0)),
                        _payer_vault=lambda: vault, _leg_wall_seconds=lambda sku=None: 4 * 3600.0,
                        _payer_balance_reader=lambda key: 10.0, work_root=tmp_path)
    assert _RollingMixin._rolling_king_key_ok(r, HK) is False
    assert "king_key_unfunded" in {c["code"] for c in money_guard.health_conditions(r, {})}
    r._payer_balance_reader = lambda key: 500.0
    assert _RollingMixin._rolling_king_key_ok(r, HK) is True


# ── L: a stuck teardown is priced and alerted ───────────────────────────────


def test_a_stuck_teardown_raises_one_priced_alert(tmp_path):
    r = _runner(tmp_path, ledger=[_pod("cascade-n91-1-funded-abc-op-0")])
    money_watch.note_teardown(r, "cascade-n91-1-funded-abc-op-0", ok=False, now=1000.0)
    assert money_watch.stuck_teardown_tick(r, now=1000.0 + 600) == []
    assert money_watch.stuck_teardown_tick(r, now=1000.0 + 1800) == [
        "cascade-n91-1-funded-abc-op-0"]
    (a,) = [c for c in money_guard.health_conditions(r, {}) if c["code"] == "teardown_stuck"]
    assert a["accrued_usd"] == pytest.approx(20.0 * 0.5)
    money_watch.note_teardown(r, "cascade-n91-1-funded-abc-op-0", ok=True)
    assert "teardown_stuck" not in {c["code"] for c in money_guard.health_conditions(r, {})}


# ── M / E: knob traps ───────────────────────────────────────────────────────


def test_caps_never_switch_the_operator_pool_on():
    from cascade.provision.operator_pool import enabled

    def r(**kw):
        return SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig(**kw)))

    assert not enabled(r(ops_operator_daily_cap_usd=100.0, ops_operator_leg_cap_usd=50.0))
    assert enabled(r(ops_operator_pool_enabled=True))
    assert enabled(r(ops_operator_providers=("lium", "shadeform")))


def test_an_approval_on_file_allows_the_operator_rent_whatever_the_fallback_flag(tmp_path,
                                                                                monkeypatch):
    from cascade.trainer.runner_operator_pods import _OperatorPodMixin

    r = _runner(tmp_path, funded_pods="rent", funded_operator_fallback=False,
                ops_paid_fallback_after_seconds=3600)
    q = _queue(tmp_path)
    r._funded_queue = lambda: q
    allowed = _OperatorPodMixin._operator_rent_allowed
    assert allowed(r, HK) is False          # no approval, fallback off, paid fallback scoped
    store = OperatorApprovals.for_work_root(tmp_path)
    store.request("funded_queue.json", HK, ref=REF, attempt_id="", kind="make_good",
                  funded_at=q.get(HK).funded_at)
    store.decide(HK, approve=True)
    assert allowed(r, HK) is True           # an approved make-good never strands
    r.cfg.round = dataclasses.replace(r.cfg.round, funded_operator_fallback=True)
    r._leg_local_obj = SimpleNamespace(operator_rent_ok=True)
    assert allowed(r, "someone-else") is True   # the flag only lets a leg PROPOSE


def test_the_cpu_reaper_covers_providers_dropped_from_the_config(tmp_path):
    from cascade.shared.config.generation import GenerationConfig
    from cascade.trainer.runner_generation import _ledger_only_providers

    ledger = tmp_path / "vms.json"
    ledger.write_text(json.dumps([{"provider": "shadeform", "name": "cgen-1", "vm_id": "v"}]))
    got = _ledger_only_providers(ledger, [], GenerationConfig())
    assert [p.name for p in got] == ["shadeform"]
    assert _ledger_only_providers(ledger, got, GenerationConfig()) == []


# ── #418 follow-up: the recent rate judges a long harvest ───────────────────


def test_a_fast_start_then_trickle_harvest_is_stopped_on_its_recent_rate():
    from cascade.trainer.remote import (
        HARVEST_RATE_WINDOW_SECONDS,
        _RateWindow,
        harvest_total_breach,
    )

    w = _RateWindow()
    assert w.add(0.0, 0) is None
    w.add(100.0, 10_000_000_000)                                   # a fast start
    rate = w.add(100.0 + HARVEST_RATE_WINDOW_SECONDS + 1, 10_000_100_000)   # then a trickle
    assert rate is not None and rate < 1_000_000
    assert harvest_total_breach(4000.0, 10_000_100_000, 3600, recent_rate=rate)
    assert harvest_total_breach(4000.0, 10_000_100_000, 3600) is None      # the average hid it


# ── F: the per-coldkey limit bounds operator-paid work only ─────────────────


def test_the_coldkey_limit_never_holds_a_miner_key_leg(tmp_path):
    from cascade.trainer import admission as A

    r = SimpleNamespace(cfg=SimpleNamespace(round=RoundConfig(ops_max_active_per_coldkey=1)),
                        work_root=tmp_path, _record_funded_failure=lambda *a, **k: None)
    A.remember_owners(r, [SimpleNamespace(hotkey=h, coldkey="same", commit_block=1)
                          for h in ("hk1", "hk2")])
    billed = {"hk1": SimpleNamespace(operator_billed=True),
              "hk2": SimpleNamespace(operator_billed=False)}
    r._funded_queue = lambda: SimpleNamespace(get=billed.get)
    r._rolling_sched = SimpleNamespace(_threads={"hk1": 1}, lane_b=None, lane_c=None)
    assert not A.owner_busy(r, "hk2")                 # a miner-key leg is never held
    billed["hk2"] = SimpleNamespace(operator_billed=True)
    assert A.owner_busy(r, "hk2")                     # two operator-paid legs of one owner
