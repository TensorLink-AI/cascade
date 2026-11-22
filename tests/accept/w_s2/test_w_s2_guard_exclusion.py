"""W-S2: lane B guard/target exclusion for the era's §16 sentinel (plan §3.0,
§16 rule 2, D33 / D35; T-row TS4).

The W-S2 row, verbatim:

    | W-S2 | **(rev4: now in §3)** lane B guard/target exclusion for the
    sentinel | — | W-S1 | ci | `scripts/laned/accept W-S2` (TS4) |

TS4, verbatim (§4):

    | TS4 | **(rev4: now in §4)** Lane B entry targeting the sentinel |
    Deferred (D35) |

§16 rule 2: "**Lane B:** it's neither a guard nor a target that era (D35)."
Rule 3: "**The other suites remain full regression guards** in every lane."
D35 (§17.4 register): decided — "A sentinel suite is not a lane B target or
guard that era".

The TARGET half (TS4: a suite-target entry on the sentinel is deferred) was
built by WB18 and is re-checked here. The GUARD half is new: while a suite is
the era's sentinel it is not one of lane B's public-benchmark guards (the
``<suite>_crps`` / ``<suite>_mase`` keys the consensus judge builds from the
trainer-signed item rows), nor a ``[lane_b] require_public`` requirement, in
every place lane B computes those guards: the validators' judge, the
trainer's self-judge and settlement, and the warm-start acceptance.
"""

from __future__ import annotations

import json
import re

import pytest

from . import helpers as h


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _one(judgements: list):
    assert len(judgements) == 1, judgements
    return judgements[0]


def _public_keys(j) -> set[str]:
    assert j.verdict is not None, f"the entry was not judged to a verdict: {j.reason!r}"
    return set(j.verdict.public_lcbs) | {k for k in j.verdict.unguarded
                                         if k.endswith(tuple(f"_{m}" for m in h.METRICS))}


# ── the plan text ────────────────────────────────────────────────────────────


def test_plan_rows_pinned():
    text = h.PLAN.read_text(encoding="utf-8")
    assert ("| W-S2 | **(rev4: now in §3)** lane B guard/target exclusion for the sentinel "
            "| — | W-S1 | ci | `scripts/laned/accept W-S2` (TS4) |") in text
    assert ("| TS4 | **(rev4: now in §4)** Lane B entry targeting the sentinel | "
            "Deferred (D35) |") in text
    assert "| TS4 | W-S2, WB18 |" in text
    assert ("| D35 | decided | A sentinel suite is not a lane B target or guard that era "
            "| S | owner (default), 2026-11-21 |") in text
    flat = _flat(text)
    for line in (
        "**Lane B:** it's neither a guard nor a target that era (D35).",
        "**The other suites remain full regression guards** in every lane.",
        "**WP W-S2:** the lane B guard/target exclusion (with the DEC-CA-0071 amendment).",
        "**TS4:** a lane B entry targeting the sentinel suite is deferred (D35).",
    ):
        assert line in flat, line


# ── 1. the consensus judge (cascade.validator.expert_lane) ───────────────────


def test_sentinel_only_regression_is_not_guarded(tmp_path):
    """A candidate 30% worse on the sentinel suite only, and better on the
    other three, merges in an era whose sentinel is that suite."""
    cfg = h.lane_b_cfg()
    em = h.expert_manifest(h.entry(h.bench_rows(h.SENTINEL_ONLY)))
    j = _one(h.judge(em, cfg, tmp_path, sentinel=h.SENTINEL))
    assert j.passed and j.reason == "merge", (
        f"the sentinel {h.SENTINEL!r} is not a lane B guard that era (D35); got "
        f"passed={j.passed} reason={j.reason!r}")
    keys = _public_keys(j)
    assert not keys & set(h.guard_keys(h.SENTINEL)), (
        f"no {h.SENTINEL}_* public guard is built or listed unguarded; got {sorted(keys)}")
    want = {k for s in h.SUITES if s != h.SENTINEL for k in h.guard_keys(s)}
    assert set(j.verdict.public_lcbs) == want, sorted(j.verdict.public_lcbs)


def test_other_suites_remain_full_guards(tmp_path):
    """§16 rule 3: a regression on a non-sentinel suite still blocks."""
    cfg = h.lane_b_cfg()
    rows = dict(h.SENTINEL_ONLY, gift=1.30)
    j = _one(h.judge(h.expert_manifest(h.entry(h.bench_rows(rows))), cfg, tmp_path,
                     sentinel=h.SENTINEL))
    assert not j.passed
    assert j.reason.startswith("regression on gift_"), j.reason


def test_domain_guards_unaffected(tmp_path):
    """The sentinel drops public-suite guards only; a regression on another
    pool domain still blocks."""
    cfg = h.lane_b_cfg()
    em = h.expert_manifest(h.entry(h.bench_rows(h.CLEAN)))
    j = _one(h.judge(em, cfg, tmp_path, cand_effect={"energy": 0.85, "web": 1.30},
                     sentinel=h.SENTINEL))
    assert not j.passed and "regression on domain web" in j.reason, j.reason


def test_other_guard_numbers_unchanged(tmp_path):
    """Dropping the sentinel's keys leaves every other guard's LCB (and the
    target LCB) byte-identical to the unarmed judgement of the same rows."""
    cfg = h.lane_b_cfg()
    em = h.expert_manifest(h.entry(h.bench_rows(h.CLEAN)))
    base = _one(h.judge(em, cfg, tmp_path))
    armed = _one(h.judge(em, cfg, tmp_path, sentinel=h.SENTINEL))
    assert base.passed and armed.passed
    assert armed.verdict.target_lcb == base.verdict.target_lcb
    assert armed.verdict.guard_lcbs == base.verdict.guard_lcbs
    assert armed.verdict.public_lcbs == {
        k: v for k, v in base.verdict.public_lcbs.items()
        if k not in h.guard_keys(h.SENTINEL)}


def test_required_sentinel_suite_is_not_required(tmp_path):
    """``require_public`` naming the sentinel: its rows may be missing that
    era; a missing NON-sentinel required suite still refuses."""
    cfg = h.lane_b_cfg(require_public=("gift", h.SENTINEL))
    no_sentinel = h.bench_rows(h.CLEAN, absent=(h.SENTINEL,))
    j = _one(h.judge(h.expert_manifest(h.entry(no_sentinel)), cfg, tmp_path,
                     sentinel=h.SENTINEL))
    assert j.passed and j.reason == "merge", (
        f"the sentinel is not a required lane B guard that era; got {j.reason!r}")
    no_gift = h.bench_rows(h.CLEAN, absent=("gift",))
    j = _one(h.judge(h.expert_manifest(h.entry(no_gift)), cfg, tmp_path, sentinel=h.SENTINEL))
    assert not j.passed and "public guard missing" in j.reason and "gift" in j.reason, j.reason


def test_rotation_restores_the_guard(tmp_path):
    """Next era another suite is the sentinel: the old one is a guard again
    (its regression blocks) and the new one is excluded."""
    cfg = h.lane_b_cfg()
    em = h.expert_manifest(h.entry(h.bench_rows(h.SENTINEL_ONLY)))
    j = _one(h.judge(em, cfg, tmp_path, sentinel="gift"))
    assert not j.passed and j.reason.startswith(f"regression on {h.SENTINEL}_"), j.reason
    assert not set(j.verdict.public_lcbs) & set(h.guard_keys("gift")), j.verdict.public_lcbs


def test_sentinel_none_is_the_unarmed_call(tmp_path):
    """``sentinel=None`` is exactly today's judgement (receipt rows too)."""
    cfg = h.lane_b_cfg(require_public=("gift", h.SENTINEL))
    for rows in (h.bench_rows(h.SENTINEL_ONLY), h.bench_rows(h.CLEAN, absent=(h.SENTINEL,))):
        em = h.expert_manifest(h.entry(rows))
        assert h.record(h.judge(em, cfg, tmp_path, sentinel=None)) == h.record(
            h.judge(em, cfg, tmp_path))


# ── 2. the warm-start acceptance (cascade.validator.merge_acceptance) ────────


def test_acceptance_excludes_the_sentinel():
    cfg = h.lane_b_cfg(require_public=("gift", h.SENTINEL))
    wins, detail = h.acceptance(cfg, h.SENTINEL_ONLY, sentinel=h.SENTINEL)
    assert [w.domain for w in wins] == ["energy"], detail
    wins, detail = h.acceptance(cfg, h.CLEAN, absent=(h.SENTINEL,), sentinel=h.SENTINEL)
    assert [w.domain for w in wins] == ["energy"], detail
    wins, detail = h.acceptance(cfg, dict(h.SENTINEL_ONLY, boom=1.30), sentinel=h.SENTINEL)
    assert wins == [] and str(detail["reason"]).startswith("regression on boom_"), detail


# ── 3. the validator wiring (cascade.validator.lane_b) ───────────────────────


def test_validator_judges_with_the_recorded_sentinel(tmp_path):
    """The settlement's W-S1 draw (the runner's recorded sentinel for that
    round) is the suite the validator's lane B judge leaves out."""
    lb = h.lane_b_cfg()
    cfg = h.chain_cfg(lb, sentinel_from_block=h.ERA_LEN)
    em = h.expert_manifest(h.entry(h.bench_rows(h.SENTINEL_ONLY)))
    r = h.validator_round(tmp_path, cfg, em, sentinel_record=h.sentinel_rec(h.SENTINEL))
    j = _one(r.judgements)
    assert j.passed, (f"the validator judges lane B without the era's sentinel "
                      f"{h.SENTINEL!r} as a guard; got {j.reason!r}")
    row = json.loads(json.dumps(r.rows, default=str))[0]
    assert not set(row.get("public_lcbs", {})) & set(h.guard_keys(h.SENTINEL)), row
    rot = h.validator_round(tmp_path, cfg, em, sentinel_record=h.sentinel_rec("gift"))
    assert not _one(rot.judgements).passed, "the rotated-out suite is a guard again"


# ── 4. the trainer (cascade.trainer.lane_b_driver) ───────────────────────────


def _judge_sentinel(call: dict):
    if "sentinel" in call["kw"]:
        return call["kw"]["sentinel"]
    return call["args"][3] if len(call["args"]) > 3 else None


def test_trainer_self_judge_uses_the_settlement_sentinel(tmp_path):
    """settle() self-judges with the sentinel of the era the settlement
    belongs to (``settlement_era_start``: block 1000 for the boundary 2000),
    drawn by W-S1 from that block's hash — with or without suite targets."""
    sc = h.trainer_scenario(tmp_path, sentinel_from_block=h.ERA_LEN)
    h.settle_ok(sc, tmp_path)
    assert len(sc.ops.judge_calls) == 1, sc.ops.judge_calls
    assert _judge_sentinel(sc.ops.judge_calls[0]) == h.SENTINEL, (
        f"ops.judge(manifest, round_id, block, sentinel=...) carries the era's sentinel "
        f"{h.SENTINEL!r}; got {sc.ops.judge_calls[0]['kw']!r}")
    assert set(sc.ops.hash_reads) == {h.SETTLE_ERA_START}, sc.ops.hash_reads


def test_trainer_sentinel_follows_the_block_hash(tmp_path):
    sc = h.trainer_scenario(tmp_path, sentinel_from_block=h.ERA_LEN, era1_sentinel="boom")
    h.settle_ok(sc, tmp_path)
    assert _judge_sentinel(sc.ops.judge_calls[0]) == "boom", sc.ops.judge_calls


def test_trainer_settle_does_not_require_the_sentinel(tmp_path):
    """The king's bench rows may lack the sentinel suite that era even when
    ``require_public`` names it; a missing non-sentinel suite still holds."""
    rows = h.bench_rows({}, absent=(h.SENTINEL,))
    sc = h.trainer_scenario(tmp_path / "a", sentinel_from_block=h.ERA_LEN, king_rows=rows)
    h.settle_ok(sc, tmp_path / "a")
    assert sc.ops.published, "the settlement published its expert manifest"
    rows = h.bench_rows({}, absent=("gift",))
    sc = h.trainer_scenario(tmp_path / "b", sentinel_from_block=h.ERA_LEN, king_rows=rows)
    with pytest.raises(OSError, match="gift"):
        h.settle(sc, tmp_path / "b")


def test_runner_ops_judge_takes_the_sentinel():
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    assert h.has_param(RunnerLaneBOps.judge, "sentinel"), (
        "RunnerLaneBOps.judge forwards the era's sentinel to judge_manifest")
    import inspect

    assert inspect.signature(RunnerLaneBOps.judge).parameters["sentinel"].default is None


# ── 5. the target half (TS4, built by WB18) still holds ──────────────────────


def test_ts4_sentinel_target_still_deferred(tmp_path):
    from tests.accept.wb18 import helpers as wb18

    sc = wb18.Scenario(tmp_path, max_defer=2)
    sc.set_sentinel(1, h.SENTINEL)
    sc.fund(wb18.HK_A, 10, f"suite:{h.SENTINEL}")
    sc.open_gate()
    assert wb18.HK_A not in sc.open_legs()
    assert wb18.deferred_for(sc.status(), wb18.HK_A)["deferred"] == "sentinel_target"


# ── invariant 1 (paired per invariant 10) ────────────────────────────────────


def test_inv1_judge_unchanged_when_unarmed(tmp_path):
    """No sentinel: every suite guards, as today."""
    cfg = h.lane_b_cfg(require_public=("gift", h.SENTINEL))
    j = _one(h.judge(h.expert_manifest(h.entry(h.bench_rows(h.SENTINEL_ONLY))), cfg, tmp_path))
    assert not j.passed and j.reason.startswith(f"regression on {h.SENTINEL}_"), j.reason
    assert set(j.verdict.public_lcbs) == {k for s in h.SUITES for k in h.guard_keys(s)}
    em = h.expert_manifest(h.entry(h.bench_rows(h.CLEAN, absent=(h.SENTINEL,))))
    j = _one(h.judge(em, cfg, tmp_path))
    assert not j.passed and "public guard missing" in j.reason, j.reason


def test_inv1_judge_armed_nonempty_lane_d_effect(tmp_path):
    cfg = h.lane_b_cfg(require_public=("gift", h.SENTINEL))
    j = _one(h.judge(h.expert_manifest(h.entry(h.bench_rows(h.SENTINEL_ONLY))), cfg, tmp_path,
                     sentinel=h.SENTINEL))
    assert j.passed, j.reason


def test_inv1_acceptance_unchanged_when_unarmed():
    cfg = h.lane_b_cfg(require_public=("gift", h.SENTINEL))
    wins, detail = h.acceptance(cfg, h.SENTINEL_ONLY)
    assert wins == [] and str(detail["reason"]).startswith(f"regression on {h.SENTINEL}_")
    wins, detail = h.acceptance(cfg, h.CLEAN)
    assert [w.domain for w in wins] == ["energy"], detail


def test_inv1_acceptance_armed_nonempty_lane_d_effect():
    cfg = h.lane_b_cfg(require_public=("gift", h.SENTINEL))
    wins, detail = h.acceptance(cfg, h.SENTINEL_ONLY, sentinel=h.SENTINEL)
    assert [w.domain for w in wins] == ["energy"], detail


def test_inv1_validator_unchanged_when_unarmed(tmp_path):
    """No recorded draw (``[v2] sentinel_from_block = 0``): today's verdict."""
    lb = h.lane_b_cfg()
    cfg = h.chain_cfg(lb, sentinel_from_block=0)
    em = h.expert_manifest(h.entry(h.bench_rows(h.SENTINEL_ONLY)))
    r = h.validator_round(tmp_path, cfg, em, sentinel_record=None)
    j = _one(r.judgements)
    assert not j.passed and j.reason.startswith(f"regression on {h.SENTINEL}_"), j.reason
    assert set(j.verdict.public_lcbs) == {k for s in h.SUITES for k in h.guard_keys(s)}


def test_inv1_validator_armed_nonempty_lane_d_effect(tmp_path):
    lb = h.lane_b_cfg()
    cfg = h.chain_cfg(lb, sentinel_from_block=h.ERA_LEN)
    em = h.expert_manifest(h.entry(h.bench_rows(h.SENTINEL_ONLY)))
    r = h.validator_round(tmp_path, cfg, em, sentinel_record=h.sentinel_rec(h.SENTINEL))
    assert _one(r.judgements).passed, _one(r.judgements).reason


def test_inv1_trainer_unchanged_when_unarmed(tmp_path):
    """``sentinel_from_block = 0``: no hash read, ``ops.judge`` called exactly
    as today, and a king without a required suite still holds settlement."""
    sc = h.trainer_scenario(tmp_path / "a", sentinel_from_block=0)
    h.settle(sc, tmp_path / "a")
    assert sc.ops.hash_reads == []
    call = sc.ops.judge_calls[0]
    assert call["kw"] == {} and len(call["args"]) == 3, call
    assert call["args"][1:] == (str(h.SETTLE_BLOCK), h.SETTLE_BLOCK)
    rows = h.bench_rows({}, absent=(h.SENTINEL,))
    sc = h.trainer_scenario(tmp_path / "b", sentinel_from_block=0, king_rows=rows)
    with pytest.raises(OSError, match=h.SENTINEL):
        h.settle(sc, tmp_path / "b")


def test_inv1_trainer_armed_nonempty_lane_d_effect(tmp_path):
    sc = h.trainer_scenario(tmp_path, sentinel_from_block=h.ERA_LEN,
                            king_rows=h.bench_rows({}, absent=(h.SENTINEL,)))
    h.settle_ok(sc, tmp_path)
    assert sc.ops.published and _judge_sentinel(sc.ops.judge_calls[0]) == h.SENTINEL
