"""WB24 (owner 2026-11-24, "is the miner CLI up to date?"): ``cascade lane-b
test`` — the miner's LOCAL preview of whether a lane B leg would win — runs
the SAME judge the validators run, :func:`cascade.eval.expert_lane.
judge_expert_slots` (#854: every slot tested; declared and undeclared max-T
families at ``gain_margin`` / ``undeclared_gain_margin``; CRPS or MASE with
the other non-inferior) plus WB26's guard 4 + anchor guard on suite slots
(#882). Without a local anchor the output says that suite slots were
previewed without the anchor guard. Each slot is printed in plain English.

The verdict is compared byte-for-byte with a direct ``judge_expert_slots``
call built the way ``cascade.validator.expert_lane.judge_entry`` builds it,
so the preview cannot drift from consensus again. Torch is blocked; window
scores and bench rows are W-S2 / WB26 stand-ins.

Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.accept.w_s2 import helpers as ws2
from tests.accept.wb26 import helpers as wb26

from . import helpers as h

SEED = 7


def _spec(fn):
    """A missing behaviour fails as an AssertionError (never an import,
    attribute, key or value error), so BASE fails for the right reason."""
    import functools

    @functools.wraps(fn)
    def run(*a, **kw):
        try:
            return fn(*a, **kw)
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001 - every other failure is the spec failing
            raise AssertionError(f"{type(e).__name__}: {e}") from e

    return run


def _dump(tv) -> str:
    return json.dumps(asdict(tv), sort_keys=True, default=repr)


def _rows(gift=1.0):
    return {"before": wb26.bench_rows(), "candidate": wb26.bench_rows({("gift", "crps"): gift}),
            "anchor": wb26.bench_rows()}


def _validator_verdict(lb, before, after, rows, declared, anchor_scores):
    """``judge_entry``'s call, spelled out from the validator's own pieces."""
    from cascade.eval.expert_lane import judge_expert_slots
    from cascade.eval.public_guards import pair_public_rows, sentinel_pairing_kw
    from cascade.validator.expert_lane import _anchor_input, _arrays, window_horizon

    k, c = _arrays(before), _arrays(after)
    public = pair_public_rows(rows["before"], rows["candidate"], min_items=lb.min_public_items,
                              **sentinel_pairing_kw(rows["before"], rows["candidate"],
                                                    lb.require_public, None)).public
    hts = tuple(lb.horizon_target_names)
    vrows = {"king": rows["before"], "candidate": rows["candidate"], "anchor": rows["anchor"]}
    return judge_expert_slots(
        slots=lb.targets, declared=declared, domains=k["domains"],
        king_qloss=k["qloss"], king_abs_target=k["abs"], king_mase=k["mase"],
        chal_qloss=c["qloss"], chal_abs_target=c["abs"], chal_mase=c["mase"],
        clusters=k["clusters"], public=public, params=lb.verdict_params(), seed=str(SEED),
        horizons=[window_horizon(sid) for sid, _ in k["ids"]] if hts else None,
        horizon_targets=hts,
        anchor=lambda: _anchor_input(vrows, k, "anchor-ptr", lambda: anchor_scores, lb, None))


@_spec
def test_preview_is_byte_identical_to_the_validator_judge():
    from cascade.miner.lane_tests import judge_slots_preview

    with h.torch_blocked():
        lb = wb26.lane_b_cfg()
        before = ws2.window_scores({})
        after = ws2.window_scores({"energy": 0.85})
        rows = _rows(gift=0.9)
        got, missing = judge_slots_preview(lb, declared=("energy",), before=before, after=after,
                                           public_rows=rows, anchor=before,
                                           anchor_label="anchor-ptr", seed=SEED)
        want = _validator_verdict(lb, before, after, rows, ("energy",), before)
    assert missing == ""
    assert _dump(got) == _dump(want)
    # every configured slot is tested, declared at gain_margin, the rest stricter
    assert got.targets == tuple(lb.targets) and got.declared == ("energy",)
    assert got.verdicts["energy"].declared and got.verdicts["energy"].bar == lb.gain_margin
    for t in lb.targets:
        if t != "energy":
            assert not got.verdicts[t].declared
            assert got.verdicts[t].bar == lb.undeclared_gain_margin
    assert got.verdicts["energy"].passed, got.verdicts["energy"].reason


@_spec
def test_plain_english_lines_name_slot_family_bar_gains_and_outcome():
    from cascade.miner.lane_tests import describe_slots, judge_slots_preview

    with h.torch_blocked():
        lb = wb26.lane_b_cfg()
        before = ws2.window_scores({})
        tv, _ = judge_slots_preview(lb, declared=("energy",), before=before,
                                    after=ws2.window_scores({"energy": 0.85}),
                                    public_rows=_rows(), anchor=before, anchor_label="a", seed=SEED)
    lines = describe_slots(tv, anchor_checked=True)
    energy = next(line for line in lines if line.strip().startswith("energy "))
    assert "[declared, bar +1.00%]" in energy
    assert "CRPS gain" in energy and "MASE gain" in energy and "LCB" in energy
    assert "CLEARED on" in energy
    web = next(line for line in lines if line.strip().startswith("web "))
    assert "[undeclared, bar +2.00%]" in web and "not cleared" in web
    assert not any("WITHOUT the anchor guard" in line for line in lines)


@_spec
def test_suite_slots_without_anchor_are_flagged_not_silently_passed():
    from cascade.eval.expert_lane import ANCHOR_MISSING
    from cascade.miner.lane_tests import describe_slots, judge_slots_preview

    with h.torch_blocked():
        lb = wb26.lane_b_cfg()
        before = ws2.window_scores({})
        after = ws2.window_scores({"energy": 0.85})
        tv, _ = judge_slots_preview(lb, declared=("suite:gift",), before=before, after=after,
                                    public_rows=_rows(gift=0.8), anchor=None, seed=SEED)
        with_anchor, _ = judge_slots_preview(lb, declared=("suite:gift",), before=before, after=after,
                                             public_rows=_rows(gift=0.8), anchor=before,
                                             anchor_label="anchor-ptr", seed=SEED)
    gift = tv.verdicts["suite:gift"]
    assert not gift.passed and gift.reason.startswith(ANCHOR_MISSING), gift.reason
    lines = describe_slots(tv, anchor_checked=False)
    assert any("suite slots are previewed WITHOUT the anchor guard" in line for line in lines)
    gl = next(line for line in lines if line.strip().startswith("suite:gift "))
    assert "anchor guard was not run locally" in gl
    # with the anchor scored locally the real guard 5 runs (recorded, never anchor_missing)
    ga = with_anchor.verdicts["suite:gift"]
    assert not ga.reason.startswith(ANCHOR_MISSING), ga.reason
    assert ga.suite_guards.get("anchor_pointer") == "anchor-ptr"


@_spec
def test_lane_b_test_runs_the_validator_judge_and_prints_slots(tmp_path, monkeypatch):
    import cascade.eval.expert_lane as el
    from cascade.miner import lane_tests

    calls = []
    real = el.judge_expert_slots

    def spy(**kw):
        calls.append(kw)
        return real(**kw)

    def no_old_judge(**kw):
        if not kw.get("guards_only"):
            raise AssertionError("lane-b test called the old per-target judge")
        return real_targets(**kw)

    real_targets = el.judge_expert_targets
    monkeypatch.setattr(el, "judge_expert_slots", spy)
    monkeypatch.setattr(el, "judge_expert_targets", no_old_judge)
    base = tmp_path / "base"
    base.mkdir()
    toml = h.write_toml(tmp_path / "chain.toml")
    cfg = h.load_cfg(toml)
    cfg = replace(cfg, lane_b=replace(cfg.lane_b, require_public=(), min_target_windows=20,
                                      min_target_clusters=5, min_guard_windows=20, bootstrap_B=200))
    monkeypatch.setattr("cascade.miner.verify.verify_repo",
                        lambda *a, **kw: SimpleNamespace(ok=True, render=lambda: ""))
    monkeypatch.setattr("cascade.miner.score._resolve_warm_start",
                        lambda cfg, spec, cache_dir: (base, f"local {spec}"))
    monkeypatch.setattr("cascade.validator.pool.window_source_from_dir", lambda *a, **kw: object())
    monkeypatch.setattr("cascade.validator.expert_lane.affinity_contexts", lambda *a, **kw: [np.ones(64)])
    monkeypatch.setattr("cascade.validator.expert_lane.verdict_windows", lambda *a, **kw: [object()] * 5)
    monkeypatch.setattr("cascade.trainer.expert_lane.probe_target_routing", lambda *a:
                        SimpleNamespace(select=lambda c: {0: [1, 3]}))

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def series(self): return iter([np.ones(64)])

    monkeypatch.setattr("cascade.miner.local_corpus.local_round_stream", lambda *a, **kw: Stream())

    class Trainer:
        def __init__(self, device=None): pass

        def train(self, stream, contract, **kw):
            return SimpleNamespace(local_dir=Path(kw["out_dir"]), train_seconds=1.0, metrics={})

    monkeypatch.setattr("cascade.trainer.toto2_trainer.Toto2Trainer", Trainer)

    def scores(path, windows, **kw):
        return ws2.window_scores({} if Path(path) == base else {"energy": 0.85})

    monkeypatch.setattr("cascade.validator.evaluator.evaluate_checkpoint", scores)
    args = SimpleNamespace(repo_dir=tmp_path, base=str(base), at_block=None, targets="energy",
                           pool_dir=tmp_path, work_dir=tmp_path / "work", device="cpu", seed=SEED,
                           train_hours=1e-5, public_project=None, anchor=None)
    with h.torch_blocked():
        result = lane_tests.test_lane_b(args, cfg)
    # the qualification call (acceptance's marginal judge may also be spied,
    # depending on import order; its seed is "local:<seed>|marginal")
    calls = [c for c in calls if c["seed"] == str(SEED)]
    assert len(calls) == 1
    assert tuple(calls[0]["slots"]) == tuple(cfg.lane_b.targets)
    assert tuple(calls[0]["declared"]) == ("energy",)
    assert "energy" in result["won_slots"]
    assert result["qualification"]["verdicts"]["energy"]["passed"]
    report = "\n".join(result["slot_report"])
    assert "energy [declared" in report and "CLEARED" in report
    assert any("without the anchor guard" in x for x in result["limitations"])
    assert result["directional_only"] and not result["live_reward_guaranteed"]


@_spec
def test_lane_b_test_help_offers_anchor():
    import argparse

    from cascade.miner.cli import build_parser

    p = build_parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    lane_b = sub.choices["lane-b"]
    inner = next(a for a in lane_b._actions if isinstance(a, argparse._SubParsersAction))
    help_text = inner.choices["test"].format_help()
    assert "--anchor" in help_text and "anchor guard" in help_text
