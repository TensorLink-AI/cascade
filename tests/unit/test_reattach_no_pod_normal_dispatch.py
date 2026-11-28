"""A re-attach with no ledgered pod is a NORMAL dispatch (2026-11-28).

Incident: lane B funded leg 5H63pHtyHgkf was re-dispatched after a trainer
restart WITH a persisted re-attach record, so the whole leg body ran under
``leg_reattach.reattach_only()``: the CPU-tier corpus generation ran in full
(~1 h, Shadeform cpu_large) and only then the rent path refused ("re-attach
refused: the leg's pod is not on the funded ledger") and requeued the leg.

Now the decision is made BEFORE any expensive stage:

* lane B / lane C: no ledgered pod (payer or operator) of the leg ⇒ the leg
  runs as a normal dispatch (same leg, same in-flight attempt), not attach-only;
  a ledgered pod (or an unknown ledger) ⇒ attach-only exactly as before (never
  a fresh rent of a pod name that might be live);
* the funded path itself refuses an attach-only leg with no ledgered pod
  before its corpus (the backstop for any other attach-only caller).

Torch-free: fake runners and ops only; nothing touches a pod, the chain or a key.
"""

from __future__ import annotations

import threading
import types

from cascade.trainer import leg_reattach

HK = "5H63pHtyHgkfAbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
SEED = 77
NETUID = 91


class _Ledgered(Exception):
    """Raised by the fake runner at the first contract read: the leg body ran
    that far; the test reads the mode it ran in."""


def _runner(ledger, *, operator_prefix=True):
    """A fake runner: current era + seeds, a funded ledger of ``ledger`` pod
    names (``None`` = unreadable), and a contract read that records whether
    the leg body runs attach-only (then stops the body)."""
    from cascade.provision.funded import funded_pod_name

    seen: list[bool] = []
    era = types.SimpleNamespace(start_block=100, index=2)

    def throne_contracts_at(block):
        seen.append(leg_reattach.attach_only())
        raise _Ledgered("stop after the mode is known")

    def load_ledger():
        if ledger is None:
            raise OSError("ledger unreadable")
        return [types.SimpleNamespace(instance_id=n) for n in ledger]

    r = types.SimpleNamespace(
        cfg=types.SimpleNamespace(lane_b=types.SimpleNamespace(leg_funding="miner"),
                                  subnet=types.SimpleNamespace(netuid=NETUID),
                                  throne_contracts_at=throne_contracts_at),
        _rolling_sched=types.SimpleNamespace(state=types.SimpleNamespace(current=era)),
        _rolling_seeds=lambda e: types.SimpleNamespace(base_seed=SEED),
        _load_funded_ledger=load_ledger)
    if operator_prefix:
        r._operator_leg_pod_prefix = lambda rid, hk: f"cascade-n{NETUID}-{rid}-op-{hk[:12]}"
    payer = funded_pod_name(str(SEED), HK, NETUID) + "-0"
    operator = f"cascade-n{NETUID}-{SEED}-op-{HK[:12]}-0"
    return r, seen, payer, operator


def _probe(ops, leg):
    """The ops' ledger verdict (``None`` when the ops has no probe)."""
    fn = getattr(ops, "leg_pod_ledgered", None)
    return fn(leg) if callable(fn) else None


def _reattach_b(runner):
    from cascade.trainer.expert_lane import LaneBSubmission, OpenLeg
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    ops = RunnerLaneBOps(runner, client=None, window_source=None)
    leg = OpenLeg(sub=LaneBSubmission(HK, 7, "g", "energy", 5), selection={0: [1]},
                  base_king_ref="k", base_era=2, funding={"entry_ref": "e", "started_block": 9})
    got, fired = [], threading.Event()

    def done(leg, ptr, *a, failure=None, **kw):
        got.append((ptr, failure))
        fired.set()

    ops.reattach_leg(leg, 2, 9, done, {"base_seed": str(SEED), "era_index": 2,
                                       "dispatch_block": 9})
    assert fired.wait(10), "the leg thread never reported"
    return ops, leg, got[0]


def test_lane_b_reattach_without_ledgered_pod_runs_a_normal_dispatch():
    runner, seen, _payer, _op = _runner([])
    ops, leg, (_ptr, failure) = _reattach_b(runner)
    # the leg body ran in NORMAL mode: its generation and a fresh rent are
    # allowed (the money gates apply) — never attach-only generate-then-refuse
    assert seen == [False]
    assert not leg_reattach.is_refusal(failure)
    assert _probe(ops, leg) is False


def test_lane_b_reattach_with_ledgered_payer_or_operator_pod_stays_attach_only():
    for which in ("payer", "operator"):
        runner, seen, payer, op = _runner([])
        name = payer if which == "payer" else op
        runner._load_funded_ledger = lambda n=name: [types.SimpleNamespace(instance_id=n)]
        ops, leg, _ = _reattach_b(runner)
        assert seen == [True], which          # adopted as today, never a fresh rent
        assert _probe(ops, leg) is True, which


def test_lane_b_unknown_ledger_stays_attach_only():
    # an unreadable ledger, or a runner without the operator name helper,
    # could hide a live pod: attach-only (the rent refusal) decides
    for runner, seen, *_ in (_runner(None), _runner([], operator_prefix=False)):
        ops, leg, _ = _reattach_b(runner)
        assert seen == [True]
        assert _probe(ops, leg) is True


def test_lane_b_another_leg_on_the_ledger_is_not_this_legs_pod():
    runner, seen, payer, _ = _runner([])
    other = ["cascade-n91-77-funded-someoneelse-0", payer.replace(f"-{SEED}-", "-76-")]
    runner._load_funded_ledger = lambda: [types.SimpleNamespace(instance_id=n) for n in other]
    ops, leg, _ = _reattach_b(runner)
    assert seen == [False] and _probe(ops, leg) is False


# ── lane C ───────────────────────────────────────────────────────────────────


def _lane_c(tmp, ledgered):
    from cascade.trainer.lane_c_driver import LaneCDriver

    modes: list[bool] = []
    q = types.SimpleNamespace(requeued=[], in_flight=lambda: [],
                              requeue=lambda hk, **kw: q.requeued.append((hk, kw)) or True)

    def fetch_recipe(ref, target, **kw):
        modes.append(leg_reattach.attach_only())
        raise RuntimeError("stop after the mode is known")

    ops = types.SimpleNamespace(queue=lambda: q, supports_reattach=True,
                                spawn=lambda fn, name: fn(), fetch_recipe=fetch_recipe)
    if ledgered is not None:
        ops.leg_pod_ledgered = lambda leg: ledgered
    cfg = types.SimpleNamespace(round=types.SimpleNamespace(dispatch_reattach_grace_seconds=900))
    drv = LaneCDriver(cfg, ops, tmp / "lane_c_driver.json")
    leg = {"hotkey": HK, "uid": 3, "target": "mkt:market", "ref": "r", "entry_ref": "e",
           "commit_block": 1, "base_pointer": "p", "base_era": 2, "started_block": 10,
           "attempt_id": "a1", "train_time": 1.0}
    drv.st.open.append(leg)
    return drv, leg, modes


def test_lane_c_reattach_without_ledgered_pod_runs_normally(tmp_path):
    drv, leg, modes = _lane_c(tmp_path, ledgered=False)
    drv._run_leg(leg, reattach=True)
    assert modes == [False]


def test_lane_c_reattach_with_ledgered_pod_or_no_probe_stays_attach_only(tmp_path):
    for i, ledgered in enumerate((True, None)):
        d = tmp_path / str(i)
        d.mkdir()
        drv, leg, modes = _lane_c(d, ledgered=ledgered)
        drv._run_leg(leg, reattach=True)
        assert modes == [True], ledgered


def test_runner_lane_c_ops_reads_the_ledger_for_both_pod_names():
    from cascade.trainer.lane_c_driver import RunnerLaneCOps

    runner, _seen, payer, op = _runner([])
    ops = RunnerLaneCOps(runner, client=None)
    leg = {"hotkey": HK}
    assert _probe(ops, leg) is False
    for name in (payer, op):
        runner._load_funded_ledger = lambda n=name: [types.SimpleNamespace(instance_id=n)]
        assert _probe(ops, leg) is True
    runner._rolling_sched = None              # no era yet: unknown ⇒ attach-only
    assert _probe(ops, leg) is True


# ── the funded path backstop ────────────────────────────────────────────────


def test_funded_path_refuses_attach_only_without_ledgered_pod_before_the_corpus(monkeypatch):
    import cascade.trainer.leg_pins as leg_pins
    import cascade.trainer.runner_funded_legs as rfl
    from cascade.trainer.faults import _FundedLegSkip

    corpus_calls: list = []
    monkeypatch.setattr(rfl, "leg_corpus", lambda *a, **k: corpus_calls.append(a) or {})
    monkeypatch.setattr(rfl, "_reupload_pending", lambda *a, **k: None)
    monkeypatch.setattr(rfl, "_pre_corpus_money_gates", lambda *a, **k: None)
    monkeypatch.setattr(leg_pins, "contract_runs_after", lambda cfg, c: False)
    rents: list = []
    failures: list = []
    runner, _seen, payer, _op = _runner([])
    runner._funded_leg_failures = {}
    runner._record_funded_failure = lambda hk, msg, **kw: failures.append((hk, msg, kw))

    def rent(*a, **k):
        rents.append(a)
        raise RuntimeError("rent reached")

    runner._rent_funded_host = rent
    gen = types.SimpleNamespace(hotkey=HK, uid=7, ref="g")
    seeds = types.SimpleNamespace(base_seed=SEED)
    contract = types.SimpleNamespace(corpus_mode="shards", arch_preset="p")

    def run():
        try:
            rfl._FundedLegMixin._run_funded_leg_scoped(
                runner, None, gen, seeds, 9, contract, "-x", warm_start_ref=None,
                role="expert", bench=False)
        except _FundedLegSkip:
            return "skipped"
        except Exception as e:  # noqa: BLE001 — base: the body ran on
            return f"ran on: {e}"
        return "returned"

    with leg_reattach.reattach_only():
        outcome = run()
    assert outcome == "skipped"
    assert corpus_calls == [] and rents == []          # nothing generated, nothing rented
    assert failures and leg_reattach.is_refusal(failures[0][1])
    assert failures[0][2]["error_class"] == "no_capacity" and failures[0][2]["burn"] is False

    # a ledgered pod: attach-only proceeds as today (corpus, then the adopt)
    runner._load_funded_ledger = lambda: [types.SimpleNamespace(instance_id=payer)]
    with leg_reattach.reattach_only():
        outcome = run()
    assert outcome == "ran on: rent reached" and len(corpus_calls) == 1
    # outside attach-only the backstop never fires (a normal leg)
    runner._load_funded_ledger = lambda: []
    assert run() == "ran on: rent reached" and len(corpus_calls) == 2
