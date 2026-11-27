"""Owner 2026-11-26 22:45Z: the 5GMzsP… lane B make-good must run on 8xH100 or
8xH200 ONLY, never 8xRTX4090 (it had rented the cheapest measured type, an
8xRTX4090). An operator approval now pins GPU types
(``approve-operator-leg <hk> --sku 8xH100,8xH200``): the operator-billed leg
rents exactly those (cheapest available first among them), an unmeasured
pinned type included (wall-bounded, throughput judge off), and the approval
ceiling binds to the pinned types' projection, not the cheapest measured
type's. No pin = unchanged. No torch."""

from __future__ import annotations

import ast
import importlib
import threading
from pathlib import Path
from types import SimpleNamespace

from . import helpers as h
from .test_wb27_watchdog import H200_RATE, _ops

GEN = h.REF
HK = "alice"
Q = "lane_b_queue.json"
FUNDED_AT = 1795652216.0
PIN = ("8xH100", "8xH200")


def _approvals():
    m = importlib.import_module("cascade.funding.operator_approvals")
    assert hasattr(m.OperatorApprovals, "pinned_skus") and hasattr(m, "normalize_skus"), \
        "operator approvals carry no GPU-type pin (skus)"
    assert hasattr(m.OperatorApprovals, "bind_ceiling"), \
        "operator approvals cannot bind a pinned approval's ceiling"
    return m


def _store(tmp_path):
    return _approvals().OperatorApprovals(Path(tmp_path) / "operator_approvals.json")


def _pending(store, *, cost=40.0):
    # the request as filed today: projected on the cheapest MEASURED type
    assert store.request(Q, HK, ref=GEN, attempt_id="a1", kind="make_good",
                         funded_at=FUNDED_AT, sku="8xRTX4090", rate_per_hour=4.0,
                         expected_hours=10.0, projected_cost_usd=cost)


# ── the store ────────────────────────────────────────────────────────────────

def test_an_approval_stores_and_keeps_its_gpu_pin(tmp_path):
    store = _store(tmp_path)
    _pending(store)
    rec = store.decide(HK, approve=True, attempt_id="a1", queue_name=Q,
                       skus="8xH100, 8xH200,8xh100")
    assert rec["skus"] == ["8xH100", "8xH200"]
    # no --max-cost: never the cheapest-measured projection x 1.25 ($50)
    assert rec["max_cost_usd"] is None and rec["ceiling_basis"] == "pinned_skus"
    assert store.pinned_skus(Q, HK, ref=f"energy|{GEN}", funded_at=FUNDED_AT) == PIN
    assert store.pinned_skus(Q, HK, ref=GEN, funded_at=FUNDED_AT + 1) == ()
    # the ceiling binds once, to the pinned projection
    assert store.bind_ceiling(Q, HK, cost_usd=200.0) == 250.0
    assert store.bind_ceiling(Q, HK, cost_usd=999.0) == 250.0
    # withdrawn at the landed cost and re-filed: the pin survives request()
    assert store.require_reapproval(Q, HK, cost_usd=300.0, reason="landed over")
    store.request(Q, HK, ref=GEN, attempt_id="a1", kind="make_good", funded_at=FUNDED_AT,
                  projected_cost_usd=300.0)
    assert store.get(Q, HK)["skus"] == ["8xH100", "8xH200"]
    rec = store.decide(HK, approve=True, queue_name=Q)
    assert rec["skus"] == ["8xH100", "8xH200"]


def test_no_pin_is_unchanged(tmp_path):
    store = _store(tmp_path)
    _pending(store)
    rec = store.decide(HK, approve=True, attempt_id="a1", queue_name=Q)
    assert "skus" not in rec and rec["max_cost_usd"] == 50.0      # 40 x 1.25, as today
    assert store.pinned_skus(Q, HK, ref=GEN, funded_at=FUNDED_AT) == ()
    assert store.bind_ceiling(Q, HK, cost_usd=999.0) == 50.0      # never rebinds
    # an explicit --max-cost with a pin is the owner's ceiling
    store2 = _store(tmp_path / "b")
    _pending(store2)
    rec = store2.decide(HK, approve=True, queue_name=Q, max_cost_usd=120.0, skus=PIN)
    assert rec["max_cost_usd"] == 120.0 and rec["ceiling_basis"] == "owner"


def test_the_cli_takes_sku():
    _approvals()
    with h.torch_blocked():
        cli = importlib.import_module("cascade.trainer.operator_legs_cli")
    args = cli._parser().parse_args(["approve-operator-leg", HK, "--sku", "8xH100,8xH200"])
    assert args.sku == "8xH100,8xH200"


# ── the lane B driver: the rent scope ────────────────────────────────────────

class _Queue:
    def __init__(self, path, *, billed):
        self.path = Path(path)
        self.entry = SimpleNamespace(hotkey=HK, ref=f"energy|{GEN}", funded_at=FUNDED_AT,
                                     operator_billed=billed, attempt_id="a1",
                                     status="in_flight")

    def get(self, hotkey):
        return self.entry if hotkey == HK else None


def _dispatch(cfg, tmp_path, monkeypatch, *, billed=True, pin=PIN, funding=None):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg

    m = _approvals()
    seen: list = []
    ops, r = _ops(cfg, tmp_path, monkeypatch, seen)
    import cascade.funding.lane_b_gpus as lbg

    # 8xRTX4090 is measured and the cheapest; 8xH100 is unmeasured
    rates = {"8xH200": H200_RATE, "8xRTX4090": 0.1366}
    gpus = SimpleNamespace(
        rent_skus=lambda: ("8xRTX4090", "8xH200"),
        rate=lambda sku: (SimpleNamespace(steps_per_s=rates[sku]) if sku in rates else None))
    monkeypatch.setattr(lbg, "lane_b_gpus_from_env", lambda cfg, **kw: gpus)
    queue = _Queue(Path(tmp_path) / Q, billed=billed)
    monkeypatch.setattr(type(ops), "funded_queue", lambda self: queue)
    store = m.OperatorApprovals.for_work_root(tmp_path)
    _pending(store)
    store.decide(HK, approve=True, attempt_id="a1", queue_name=Q, skus=pin)
    scopes: list = []
    inner = type(r)._run_funded_leg

    def run(self, disp, gen, seeds, block, contract, suffix, **kw):
        scopes.append((getattr(self._leg_local_obj, "lane_b_rent_skus", None),
                       tuple(self._funded_skus_for_rent())))
        return inner(self, disp, gen, seeds, block, contract, suffix, **kw)

    monkeypatch.setattr(type(r), "_run_funded_leg", run)
    sub = LaneBSubmission(HK, 7, GEN, "energy", 5, targets=["energy"])
    leg = OpenLeg(sub=sub, selection={0: [1, 2]}, base_king_ref="kingref", base_era=3,
                  funding=dict(funding or {"entry_ref": "x", "gpu": "8xRTX4090"}))
    ev = threading.Event()
    ops.dispatch_leg(leg, 3, 7300, lambda *a, **k: ev.set())
    assert ev.wait(5)
    (_contract, judged), = seen
    return scopes[0], judged


def test_a_pinned_make_good_rents_only_the_pinned_types(cfg, tmp_path, monkeypatch):
    (scope, rentable), judged = _dispatch(cfg, tmp_path, monkeypatch)
    assert scope == PIN and rentable == PIN, \
        f"the pinned make-good may rent {rentable}, not only {PIN}"
    assert "8xRTX4090" not in rentable                 # cheaper, measured, never rented
    # 8xH100 (unmeasured) rents too: wall-bounded, throughput judge off for it
    assert judged is not None and judged[1] is None


def test_a_pinned_fixed_steps_leg_is_wall_bounded(cfg, tmp_path, monkeypatch):
    (scope, _), judged = _dispatch(cfg, tmp_path, monkeypatch, funding={
        "entry_ref": "x", "gpu": "8xRTX4090", "train_hours": 3.0, "leg_steps": 599,
        "leg_guard_seconds": 13603})
    assert scope == PIN and judged is not None


def test_no_pin_or_miner_billed_is_unchanged(cfg, tmp_path, monkeypatch):
    _approvals()
    (scope, _), _ = _dispatch(cfg, tmp_path / "a", monkeypatch, pin=None)
    assert scope == ("8xRTX4090", "8xH200")            # lane B's measured list, as today
    (scope, _), _ = _dispatch(cfg, tmp_path / "b", monkeypatch, billed=False)
    assert scope == ("8xRTX4090", "8xH200")            # a miner-billed leg ignores it


# ── the gate: projection and ceiling on the pinned types ─────────────────────

def test_the_gate_prices_and_binds_the_ceiling_on_the_pinned_types(cfg, tmp_path,
                                                                   monkeypatch):
    m = _approvals()
    import cascade.trainer.money_guard as mg
    from cascade.funding.lane_b_gpus import rent_scope
    from cascade.trainer.loop import TrainerRunner

    r = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path, trainer_spec="m:C")
    queue = _Queue(Path(tmp_path) / Q, billed=True)
    r.__dict__["_funded_queue"] = lambda: queue
    asked: list = []

    def offer(skus):
        asked.append(tuple(skus))
        return 4.0 if "8xRTX4090" in skus else 22.0

    r.__dict__["_operator_offer_rate"] = offer
    monkeypatch.setattr(mg, "operator_rent_preflight", lambda runner, gen, cost: None)
    monkeypatch.setattr(mg, "operator_max_price_per_hour", lambda rnd: 0.0, raising=False)
    import cascade.provision.operator_pool as op
    monkeypatch.setattr(op, "operator_max_price_per_hour", lambda rnd: 0.0)
    store = m.OperatorApprovals.for_work_root(tmp_path)
    _pending(store)                                    # $40 on the 8xRTX4090
    store.decide(HK, approve=True, attempt_id="a1", queue_name=Q, skus=PIN)
    gen = SimpleNamespace(hotkey=HK, ref=GEN, uid=7)
    gpus = SimpleNamespace(rent_skus=lambda: ("8xRTX4090", "8xH200"))
    pinned = mg.operator_pinned_skus(r, HK, queue=queue)
    assert pinned == PIN
    with rent_scope(r, gpus, only=pinned):
        mg.operator_leg_gate(r, gen, kind="make_good", rent=True)   # approved: no skip
    assert asked and all(s == PIN for s in asked), asked
    rec = store.get(Q, HK)
    proj = rec["ceiling_projection_usd"]
    assert proj > 40.0 and rec["max_cost_usd"] == round(proj * 1.25, 2)


# ── lane C (MarketCast) shares the rule ──────────────────────────────────────

def test_the_marketcast_leg_rents_the_pinned_types():
    _approvals()
    src = (h.REPO / "cascade" / "trainer" / "marketcast_runner.py").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "train_mkt_leg")
    body = ast.unparse(fn)
    assert "operator_pinned_skus(r, gen.hotkey, queue=ops.queue())" in body, \
        "a MarketCast make-good ignores the approval's GPU pin"
    assert "mkt_rent_scope(r, rent_skus)" in body
