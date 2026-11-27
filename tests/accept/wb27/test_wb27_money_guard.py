"""Owner 2026-11-27 ("fix all those things"): money/pod safety fixes found in
the money-guard audit. No torch, no network, no provider.

1. The runway counted only the funded ledger + the hand-kept
   ``operator_standing_pods_usd_per_hour`` table ($2.09/h) while the operator
   account burned ~$15/h (owner and eval pods on no ledger). Every pod the
   operator account lists that is not on the ledger now burns at its listed
   $/h; the table is only the fallback while no fresh listing is known.
2. The operator-leg projection labelled with the first type, priced at the
   cheapest offer of ANY type and timed at the fastest type's wall; the
   landed ceiling check used the projection's (fastest) hours. Now: per type
   (its price x its wall), the max over the allowed (pinned) types, and the
   landed type's own wall at launch.
3. Make-good debt lived only in an in-memory alert: now a persistent
   owed→paid ledger beside the spend record, in the trainer status.
4. Lemons on the miner's key had no per-submission cap: ``[round]
   miner_make_good_cap_usd`` (default $25), then operator-billed and
   approval-gated.
5. Every operator leg was logged "operator-billed (payer key expired)".
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import helpers as h

HK = h.HK
REF = h.REF
KING = "cascade-n91-777-funded-king"
NOW = 1_800_000_000.0


def _mg():
    return importlib.import_module("cascade.trainer.money_guard")


def _runner(tmp_path, *, balance=1000.0, ledger=(), **round_kw):
    from cascade.shared.config import RoundConfig

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

    def record(hk, msg, *, miner_fault, error_class, burn):
        r._funded_leg_failures[hk] = (msg, miner_fault, error_class, burn)

    r._record_funded_failure = record
    return r


def _pod(pid, payer=""):
    from cascade.provision.state import PodInstance

    return PodInstance(provider="lium", instance_id=pid, stage="funded",
                       rented_at_iso="2026-11-03T00:00:00Z", payer_hotkey=payer)


def _row(name, price, status="RUNNING"):
    return {"name": name, "huid": f"{name}-huid", "id": f"id-{name}", "status": status,
            "price_per_hour": price}


def _need_listing(mg):
    assert hasattr(mg, "note_operator_listing") and hasattr(mg, "operator_listing"), \
        "the runway guard never sees the operator account's pod listing"


# ── 1. the runway counts every pod the operator account lists ───────────────

def test_runway_burn_counts_every_listed_pod_not_on_the_ledger(tmp_path):
    with h.torch_blocked():
        mg = _mg()
    _need_listing(mg)
    r = _runner(tmp_path, ledger=[_pod(KING + "-0"), _pod("cascade-n91-777-funded-abc-0",
                                                           payer=HK)],
                operator_standing_pods_usd_per_hour=(("valeval-blackwell2", 0.89),))
    mg.OperatorSpend.for_runner(r).charge("", 120.0, kind="king", pod_id=KING + "-0",
                                          rate=10.0, hours=12.0, now=NOW)
    mg.note_operator_listing(r, [
        _row(KING + "-0", 10.0),                     # on the ledger: counted once, there
        _row("main-pod", 11.12),                     # the owner's pod: on no ledger
        _row("valeval-blackwell4", 1.48),            # the eval pod the table forgot
        _row("mystery", None),                       # unpriced: the ceiling ($20/h)
        _row("gone-pod", 9.0, status="TERMINATED"),  # not billing
    ], now=NOW + 2 * 3600)
    c = mg.committed_spend(r, now=NOW + 2 * 3600)
    assert c["burn_source"] == "listing"
    assert c["burn_usd_per_hour"] == pytest.approx(10.0 + 11.12 + 1.48 + 20.0)
    horizon = r.cfg.round.operator_runway_alert_hours
    assert c["committed_usd"] == pytest.approx(10.0 * 10 + (11.12 + 1.48 + 20.0) * horizon,
                                               abs=0.01)
    names = {p["pod"] for p in c["pods"]}
    assert {"main-pod", "valeval-blackwell4", "mystery"} <= names
    assert "valeval-blackwell2" not in names and "gone-pod" not in names


def test_the_standing_table_is_only_the_fallback_without_a_fresh_listing(tmp_path):
    with h.torch_blocked():
        mg = _mg()
    _need_listing(mg)
    r = _runner(tmp_path, operator_standing_pods_usd_per_hour=(("valeval", 2.0),))
    c = mg.committed_spend(r, now=NOW)
    assert c["burn_source"] == "table" and c["burn_usd_per_hour"] == pytest.approx(2.0)
    mg.note_operator_listing(r, [_row("main-pod", 11.0)], now=NOW)
    assert mg.committed_spend(r, now=NOW + 60)["burn_usd_per_hour"] == pytest.approx(11.0)
    stale = NOW + mg.OPERATOR_LISTING_MAX_AGE_SECONDS + 1
    c = mg.committed_spend(r, now=stale)
    assert c["burn_source"] == "table" and c["burn_usd_per_hour"] == pytest.approx(2.0)


def test_runway_low_fires_on_the_real_burn(tmp_path):
    with h.torch_blocked():
        mg = _mg()
    _need_listing(mg)
    r = _runner(tmp_path, operator_standing_pods_usd_per_hour=(("valeval", 0.5),),
                operator_balance_reserve_usd=50.0, operator_runway_alert_hours=6.0)
    mg.runway_tick(r, 200.0)                       # the stale table: (200-3-50)/0.5 h
    assert "operator_runway_low" not in {c["code"] for c in mg.health_conditions(r, {})}
    mg.note_operator_listing(r, [_row("main-pod", 11.12), _row("valeval", 0.5),
                                 _row("valeval-blackwell4", 1.48), _row("rt2-pod", 1.9)])
    rw = mg.runway_tick(r, 200.0)
    assert rw["burn_usd_per_hour"] == pytest.approx(15.0)
    assert rw["hours_to_reserve"] == pytest.approx(150.0 / 15.0, abs=0.01)
    low = [c for c in mg.health_conditions(r, {}) if c["code"] == "operator_runway_low"]
    assert len(low) == 1 and low[0]["severity"] == "critical"
    assert "15.00/h" in low[0]["message"]
    saved = json.loads((tmp_path / mg.RUNWAY_FILE).read_text())
    assert saved["burn_source"] == "listing" and saved["hours_to_reserve"] == pytest.approx(10.0)


def test_the_money_watch_keeps_the_account_listing_for_the_runway(tmp_path):
    with h.torch_blocked():
        mg = _mg()
        mw = importlib.import_module("cascade.trainer.money_watch")
    _need_listing(mg)
    assert hasattr(mw, "listing_tick"), "nothing refreshes the operator listing"
    r = _runner(tmp_path, operator_inventory_interval_seconds=3600)
    rows = [_row("main-pod", 11.12), _row("valpriv-4090", 0.6)]
    assert mw.listing_tick(r, now=NOW, lister=lambda: rows) == 2
    assert mw.listing_tick(r, now=NOW + 10, lister=lambda: rows) is None      # not due
    assert mg.committed_spend(r, now=NOW + 20)["burn_usd_per_hour"] == pytest.approx(11.72)
    # the hourly inventory's own listing refreshes it too
    r2 = _runner(tmp_path / "b", operator_inventory_interval_seconds=3600)
    (tmp_path / "b").mkdir()
    mw.inventory_tick(r2, now=NOW, lister=lambda: rows)
    assert mg.operator_listing(r2, now=NOW + 1) is not None
    # inventory off ⇒ no listing (the table is the runway's source)
    r3 = _runner(tmp_path, operator_inventory_interval_seconds=0)
    assert mw.listing_tick(r3, now=NOW, lister=lambda: rows) is None


# ── 2. the projection is per GPU type, the max over the allowed types ───────

WALLS = {"8xRTX4090": 10 * 3600.0, "8xH200": 3.4 * 3600.0}


def _proj_runner(tmp_path, **kw):
    r = _runner(tmp_path, operator_max_price_per_hour=30.0, **kw)
    r._funded_skus_for_rent = lambda: ("8xRTX4090", "8xH200")
    r._leg_wall_seconds = lambda sku=None: WALLS[sku] if sku else min(WALLS.values())
    r._operator_offer_rates = lambda skus: {"8xRTX4090": 4.0, "8xH200": 22.0}
    r._operator_offer_rate = lambda skus: 4.0          # the old seam: the cheapest
    return r


def test_projection_is_each_types_price_times_its_own_wall_max_over_types(tmp_path):
    with h.torch_blocked():
        mg = _mg()
    r = _proj_runner(tmp_path)
    sku, rate, hours, cost = mg.projected_operator_leg_cost(r, live_price=True)
    # 8xRTX4090: 4 x 10 h = 40; 8xH200: 22 x 3.4 h = 74.8 — never 4 x 3.4 h
    assert (sku, rate) == ("8xH200", 22.0)
    assert hours == pytest.approx(3.4) and cost == pytest.approx(74.8)
    # no live price: the ceiling on every type, so the SLOWEST type's wall
    sku, rate, hours, cost = mg.projected_operator_leg_cost(r, live_price=False)
    assert (sku, rate, hours) == ("8xRTX4090", 30.0, pytest.approx(10.0))
    assert cost == pytest.approx(300.0)


def test_projection_uses_the_approvals_pinned_types(tmp_path, monkeypatch):
    with h.torch_blocked():
        mg = _mg()
    assert "hotkey" in inspect.signature(mg.projected_operator_leg_cost).parameters, \
        "the projection ignores the approval's pinned GPU types"
    r = _proj_runner(tmp_path)
    monkeypatch.setattr(mg, "operator_pinned_skus", lambda runner, hk, queue=None:
                        ("8xRTX4090",) if hk == HK else ())
    sku, rate, hours, cost = mg.projected_operator_leg_cost(r, live_price=True, hotkey=HK)
    assert (sku, rate, cost) == ("8xRTX4090", 4.0, pytest.approx(40.0))
    assert mg.projected_operator_leg_cost(r, live_price=True, hotkey="other")[0] == "8xH200"


def test_the_landed_ceiling_uses_the_landed_types_wall(tmp_path):
    with h.torch_blocked():
        mg = _mg()
        from cascade.funding.operator_approvals import OperatorApprovals
    assert "landed_sku" in inspect.signature(mg.note_operator_launch).parameters, \
        "the landed ceiling check times every type at the projection's hours"
    r = _proj_runner(tmp_path)
    store = OperatorApprovals.for_work_root(tmp_path)
    assert store.request("funded_queue.json", HK, ref=REF, attempt_id="a1", kind="make_good",
                         projected_cost_usd=30.0)
    store.decide(HK, approve=True, attempt_id="a1", max_cost_usd=35.0)
    a = {"queue": "funded_queue.json", "hotkey": HK, "ref": REF, "attempt_id": "a1",
         "cost": 30.0, "hours": 3.4, "kind": "make_good", "book": False}
    mg._leg_local(r).operator_approval = dict(a)
    # landed on the 8xRTX4090 at $4/h: 4 x 10 h = $40 > $35 (4 x 3.4 h = $13.6 slipped)
    assert mg.note_operator_launch(r, HK, landed_rate=4.0, pod_id="p", landed_sku="8xRTX4090") \
        is False
    assert store.get("funded_queue.json", HK)["state"] != "launched"
    mg._leg_local(r).operator_approval = dict(a)
    assert mg.note_operator_launch(r, HK, landed_rate=4.0, pod_id="p", landed_sku="8xH200")


def test_the_operator_pod_launch_passes_its_landed_type():
    src = (h.REPO / "cascade" / "trainer" / "runner_operator_pods.py").read_text("utf-8")
    assert "landed_sku=sku" in src, "the operator leg launch never names its landed type"


# ── 3. the persistent make-good owed→paid ledger ─────────────────────────────

def _ledger_mod():
    assert importlib.util.find_spec("cascade.funding.make_good_ledger") is not None, \
        "no persistent make-good ledger"
    return importlib.import_module("cascade.funding.make_good_ledger")


class _Clock:
    def __init__(self, t=NOW):
        self.t = t

    def __call__(self):
        return self.t


def test_a_make_good_pod_is_owed_persistently_and_survives_a_refund(tmp_path):
    m = _ledger_mod()
    from cascade.funding.miner_spend import MAKE_GOOD, MinerSpend

    clock = _Clock()
    spend = MinerSpend(tmp_path / "miner_spend.json", clock=clock)
    spend.start(HK, ref=REF, funded_at=1.0, pod_id="pod-a", rate=8.0, new_rent=True)
    clock.t += 1800                                    # half an hour at $8/h
    spend.end(HK, "pod-a", outcome=MAKE_GOOD, reason="lemon")
    led = m.MakeGoodLedger.for_work_root(tmp_path)
    v = led.view()
    assert v["owed_usd"] == pytest.approx(4.0) and v["outstanding_usd"] == pytest.approx(4.0)
    assert v["miners"][0]["hotkey"] == HK and v["miners"][0]["pods"] == 1
    # a re-fund (new funded_at) replaces the spend record; the debt stays
    spend.start(HK, ref=REF, funded_at=2.0, pod_id="pod-b", rate=8.0, new_rent=True)
    assert spend.summary(HK, ref=REF, funded_at=2.0).make_good_usd == 0.0
    assert m.MakeGoodLedger.for_work_root(tmp_path).view()["owed_usd"] == pytest.approx(4.0)
    # a waste close is the miner's waste, never our debt
    clock.t += 600
    spend.end(HK, "pod-b", outcome="waste")
    assert led.view()["owed_usd"] == pytest.approx(4.0)
    # re-recording the same rent is idempotent
    owed, _paid = led.rows(HK)
    assert not led.record_owed(HK, rent_id=owed[0]["rent_id"], usd=4.0)
    assert led.view()["owed_usd"] == pytest.approx(4.0)


def test_payments_settle_the_outstanding_debt(tmp_path):
    m = _ledger_mod()
    led = m.MakeGoodLedger.for_work_root(tmp_path)
    assert led.record_owed(HK, rent_id="r1", usd=10.0, pod_id="p1", reason="lemon")
    assert led.record_owed("other", rent_id="r2", usd=3.0)
    assert led.mark_paid(HK, 4.0, note="tao refund") == pytest.approx(4.0)
    v = led.view()
    assert (v["owed_usd"], v["paid_usd"], v["outstanding_usd"]) == (13.0, 4.0, 9.0)
    assert v["miners"][0]["hotkey"] == HK and v["miners"][0]["outstanding_usd"] == 6.0
    assert led.mark_paid(HK, note="rest") == pytest.approx(6.0)        # all outstanding
    assert led.mark_paid(HK, note="nothing left") == 0.0
    # atomic JSON on disk; an unreadable file is never overwritten
    doc = json.loads((tmp_path / m.MAKE_GOOD_LEDGER_FILE).read_text())
    assert set(doc["owed"]) == {"r1", "r2"} and len(doc["paid"]) == 2
    (tmp_path / m.MAKE_GOOD_LEDGER_FILE).write_text("{broken")
    with pytest.raises(m.MakeGoodUnknown):
        led.record_owed(HK, rent_id="r3", usd=1.0)
    assert (tmp_path / m.MAKE_GOOD_LEDGER_FILE).read_text() == "{broken"


def test_the_trainer_status_carries_the_make_good_debt(tmp_path):
    m = _ledger_mod()
    vs = importlib.import_module("cascade.shared.v2_status")
    assert hasattr(vs, "make_good_view"), "the trainer status has no make-good view"
    m.MakeGoodLedger.for_work_root(tmp_path).record_owed(HK, rent_id="r1", usd=7.5)
    v = vs.make_good_view(tmp_path)
    assert v["outstanding_usd"] == 7.5
    assert v["miners"] == [{"hotkey": HK, "owed_usd": 7.5, "paid_usd": 0.0,
                            "outstanding_usd": 7.5, "pods": 1}]
    assert vs.make_good_view(None) is None
    src = inspect.getsource(vs.publish_trainer_v2_status)
    assert 'doc["make_good"] = make_good_view(' in src


def test_owner_cli_lists_and_records_payments(tmp_path, capsys):
    _ledger_mod()
    cli = importlib.import_module("cascade.trainer.operator_legs_cli")
    assert {"make-good-ledger", "make-good-paid"} <= set(cli.COMMANDS)
    from cascade.funding.make_good_ledger import MakeGoodLedger

    MakeGoodLedger.for_work_root(tmp_path).record_owed(HK, rent_id="r1", usd=12.0)
    assert cli.main(["make-good-paid", HK, "--usd", "5", "--note", "paid in TAO",
                     "--work-root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert cli.main(["make-good-ledger", "--json", "--work-root", str(tmp_path)]) == 0
    v = json.loads(capsys.readouterr().out)
    assert v["outstanding_usd"] == 7.0 and v["paid_usd"] == 5.0


# ── 4. the per-submission make-good cap on the miner's key ───────────────────

def test_the_make_good_cap_knob_parses_and_round_trips(tmp_path):
    from cascade.shared.config import RoundConfig, load_chain_config

    assert hasattr(RoundConfig(), "miner_make_good_cap_usd"), "no miner_make_good_cap_usd knob"
    assert RoundConfig().miner_make_good_cap_usd == 25.0
    assert load_chain_config(h.CHAIN_TOML).round.miner_make_good_cap_usd == 25.0
    src = h.CHAIN_TOML.read_text(encoding="utf-8")
    assert src.count("\n[round]\n") == 1
    p = tmp_path / "chain.toml"
    p.write_text(src.replace("\n[round]\n", "\n[round]\nminer_make_good_cap_usd = 40\n", 1))
    assert load_chain_config(p).round.miner_make_good_cap_usd == 40.0
    p.write_text(src.replace("\n[round]\n", "\n[round]\nminer_make_good_cap_usd = -1\n", 1))
    with pytest.raises(ValueError, match="miner_make_good_cap_usd"):
        load_chain_config(p)


class _Queue:
    def __init__(self, path, entry):
        self.path, self.entry = Path(path), entry

    def get(self, hk):
        return self.entry if hk == self.entry.hotkey else None

    def set_operator_billed(self, hk, flag):
        self.entry.operator_billed = bool(flag)
        return True


def _capped_runner(tmp_path, owed_usd, **kw):
    from cascade.funding.miner_spend import MAKE_GOOD, MinerSpend

    r = _runner(tmp_path, **kw)
    entry = SimpleNamespace(hotkey=HK, ref=REF, funded_at=1.0, operator_billed=False,
                            operator_faults=0, attempt_id="a1")
    q = _Queue(tmp_path / "funded_queue.json", entry)
    r._funded_queue = lambda: q
    clock = _Clock()
    spend = MinerSpend.beside(q.path, clock=clock)
    spend.start(HK, ref=REF, funded_at=1.0, pod_id="lemon-0", rate=owed_usd, new_rent=True)
    clock.t += 3600
    spend.end(HK, "lemon-0", outcome=MAKE_GOOD, reason="lemon")
    return r, entry


def test_at_the_cap_the_leg_goes_operator_billed_and_requeues_unburned(tmp_path):
    with h.torch_blocked():
        g = importlib.import_module("cascade.trainer.miner_spend_guard")
        from cascade.trainer.faults import _FundedLegSkip
    assert hasattr(g, "make_good_cap_check"), "no per-submission make-good cap"
    r, entry = _capped_runner(tmp_path, 26.0)
    with pytest.raises(_FundedLegSkip):
        g.make_good_cap_check(r, SimpleNamespace(hotkey=HK, ref=REF, uid=7))
    assert entry.operator_billed is True
    _msg, miner_fault, cls, burn = r._funded_leg_failures[HK]
    assert (miner_fault, burn) == (False, False) and cls == "operator_transient"
    mg = _mg()
    assert "miner_make_good_cap" in {c["code"] for c in mg.health_conditions(r, {})}
    # under the cap (or cap off): nothing happens
    r2, e2 = _capped_runner(tmp_path / "u", 20.0)
    g.make_good_cap_check(r2, SimpleNamespace(hotkey=HK, ref=REF, uid=7))
    r3, e3 = _capped_runner(tmp_path / "off", 99.0, miner_make_good_cap_usd=0.0)
    g.make_good_cap_check(r3, SimpleNamespace(hotkey=HK, ref=REF, uid=7))
    assert not e2.operator_billed and not e3.operator_billed


def test_a_lemon_past_the_cap_rents_no_replacement_even_with_operator_capacity(tmp_path):
    with h.torch_blocked():
        g = importlib.import_module("cascade.trainer.miner_spend_guard")
        from cascade.trainer.runner_funded_pods import _FundedPodMixin
    assert hasattr(g, "make_good_over_cap"), "no per-submission make-good cap"
    r, entry = _capped_runner(tmp_path, 30.0)
    r._operator_recovery_available = lambda hk=None: True
    r.OPERATOR_FAULT_RETRIES = 3
    assert _FundedPodMixin._count_funded_loss(r, HK, more=True) is False
    assert entry.operator_billed is True
    # under the cap the old rule holds: a replacement on the miner's key
    r2, e2 = _capped_runner(tmp_path / "u", 10.0)
    r2._operator_recovery_available = lambda hk=None: True
    r2.OPERATOR_FAULT_RETRIES = 3
    assert _FundedPodMixin._count_funded_loss(r2, HK, more=True) is True
    assert e2.operator_billed is False


def test_the_rent_path_checks_the_cap_before_every_miner_key_rent():
    src = (h.REPO / "cascade" / "trainer" / "runner_funded_pods.py").read_text("utf-8")
    assert "make_good_cap_check(self, gen)" in src
    # the small-loss rerun and the no-operator-capacity retry never undo it
    assert src.count("make_good_over_cap(self, gen.hotkey)") >= 2


# ── 5. the operator leg log names the real reason ────────────────────────────

def test_the_operator_leg_log_names_its_real_reason(tmp_path):
    with h.torch_blocked():
        mg = _mg()
    assert hasattr(mg, "operator_billing_reason"), "operator legs log a fixed reason"
    r = _runner(tmp_path)
    mg._leg_local(r).operator_approval = {"kind": "make_good", "lost": 12.0}
    assert "lost on our side" in mg.operator_billing_reason(r)
    mg._leg_local(r).operator_approval = {"kind": "expired_key"}
    assert mg.operator_billing_reason(r) == "payer key expired"
    mg._leg_local(r).operator_approval = {"kind": "spend_cap_make_good"}
    assert "waste cap" in mg.operator_billing_reason(r)
    assert mg.operator_billing_reason(r, king_backup=True) == "king backup pod"
    src = (h.REPO / "cascade" / "trainer" / "runner_operator_pods.py").read_text("utf-8")
    assert '"operator-billed (payer key expired)"' not in src
