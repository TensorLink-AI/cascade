"""Validator side of lane B (DEC-CA-0050): judge a settlement's expert
manifest and fold the winners into the lane-B consensus state.

Per entry, in order — the first failure rejects the entry:

1. ``stale``     the leg must be rebased onto THIS validator's current king
                 (``ExpertManifest.king_pointer`` == the king pointer it holds);
2. ``selection`` only when a re-derivation hook is passed (tests, tools):
                 the expert rows must equal the ESFT selection on the king +
                 affinity slice + recorded ``exclude``. The live validator
                 passes NO hook — the rows are TRUSTED trainer output
                 (DEC-CA-0050): the slice's pool snapshot and block are not
                 in the manifest, the rows were planned on the leg's BASE
                 king (not this settlement's), and an exact match over
                 float32 CPU routing across validator hardware would fork on
                 a tie at the ESFT cut. What binds a leg is stages 3-4: it
                 may change nothing but its declared rows, and those rows
                 must win the verdict and clear every guard;
3. ``merge``     the rebased checkpoint differs from the king ONLY on its
                 selected rows (``moe_model.merge_expert_slices``) — and, from
                 ``[lane_b] merge_guard_from_block`` (``strict``), holds no
                 non-finite value on them;
4. ``verdict``   gain on the target domain + every non-inferiority guard
                 (``eval.expert_lane.judge_expert_leg``) on the round's windows,
                 public guards from the trainer-signed per-item rows.

Passing entries become ``DomainWin``s at the settlement block; with several
on one domain the best target LCB wins (ties: earliest commit block — never
UID). Pure given the injected I/O callables, so cascade-audit replays it.

DEC-CA-0059 multi-target entries (``ExpertEntry.targets``): the selection is
the union of each declared target's re-derivation; each target is judged at
``alpha / k`` with the guards run ONCE (any guard failure blocks every
target); each won target is its own ``DomainWin`` (its own slot and decay
clock), all naming the same judged checkpoint — merged into the king once.
A single-target entry is judged, recorded and reported byte-for-byte as
before.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eval.expert_lane import ExpertVerdict, TargetsVerdict, judge_expert_targets
from ..shared.expert_manifest import ExpertEntry, ExpertManifest
from ..shared.lane_b import DomainWin, LaneBState


def verdict_windows(cfg, window_source, round_seed, *, block):
    """The round's verdict windows — the ONE draw rule shared by the validator
    (``ValidatorRunner._verdict_windows``) and the trainer's lane-B self-judge,
    so both score the identical set: the scored horizon ladder when active at
    ``block``, else the single-horizon rotating draw."""
    from ..shared.era import v2_settlement_active
    from .windows import ladder_windows_for_round, scored_ladder

    budget_fn = getattr(cfg, "eval_windows_at", None)
    budget = budget_fn(block) if budget_fn else cfg.eval.n_windows
    horizons = scored_ladder(cfg.eval, block)
    if not horizons:
        draw = window_source.windows_for_round
        if getattr(getattr(cfg, "v2", None), "eval_n_windows", 0) and v2_settlement_active(cfg, block):
            draw = getattr(window_source, "windows_for_verdict", draw)
        return draw(round_seed, budget, block=block)
    return ladder_windows_for_round(
        window_source, horizons=horizons, n_windows=budget,
        context_length=cfg.eval.context_length, round_seed=round_seed, block=block,
    )


AFFINITY_SEED = "lane-b-affinity"


def affinity_contexts(cfg, window_source, domain: str, *, block) -> list:
    """The lane-B affinity slice: the CONTEXTS (inputs only, never targets) of
    the first ``affinity_windows`` pool windows of ``domain`` in a fixed-seed
    draw at ``block`` — a pure function of the pool snapshot, so the trainer's
    selection and a validator's re-derivation read the same inputs. A
    ``horizon:<h>`` target (DEC-CA-0059) takes the first ``affinity_windows``
    windows of ANY domain (its routing is read over the masked forecast
    patches, not the context)."""
    from ..shared.config.v2 import HORIZON_TARGET_PREFIX

    any_domain = str(domain).startswith(HORIZON_TARGET_PREFIX)
    n = max(cfg.lane_b.affinity_windows * 8, cfg.eval.n_windows)
    ws = window_source.windows_for_round(AFFINITY_SEED, n, block=block)
    out = []
    for w in ws:
        if any_domain or (w.metadata or {}).get("domain") == domain:
            ctx = np.asarray(w.history, dtype=np.float64)
            out.append(ctx[0] if ctx.ndim == 2 else ctx)
            if len(out) >= cfg.lane_b.affinity_windows:
                break
    return out


@dataclass
class ExpertJudgement:
    entry: ExpertEntry
    passed: bool
    stage: str                 # "stale" | "selection" | "merge" | "verdict"
    reason: str
    verdict: ExpertVerdict | None = None
    king_pointer: str = ""
    # DEC-CA-0059: the per-target verdicts of a judged entry (one target ⇒
    # ``{domain: verdict}``); ``verdict`` is the first declared target's
    targets: TargetsVerdict | None = None

    def target_verdicts(self) -> dict[str, ExpertVerdict]:
        if self.targets is not None:
            return dict(self.targets.verdicts)
        return {} if self.verdict is None else {self.entry.domain: self.verdict}


_LADDER_ID = None


def window_horizon(series_id: str) -> int | None:
    """The scored horizon of a ladder window (``h<h>-s<pool index>``, the id
    ``windows.horizon_draw`` stamps), else None."""
    import re

    global _LADDER_ID
    if _LADDER_ID is None:
        _LADDER_ID = re.compile(r"^h([0-9]+)-s[0-9]+$")
    m = _LADDER_ID.match(str(series_id))
    return int(m.group(1)) if m else None


def _arrays(scores) -> dict:
    return {
        "qloss": np.asarray([s.qloss_per_q for s in scores], dtype=np.float64),
        "abs": np.asarray([s.abs_target for s in scores], dtype=np.float64),
        "mase": np.asarray([s.mase for s in scores], dtype=np.float64),
        "domains": [s.domain or "" for s in scores],
        "clusters": [s.source if s.source is not None else s.series_id for s in scores],
        "ids": [(s.series_id, s.channel) for s in scores],
    }


def judge_entry(entry: ExpertEntry, *, manifest: ExpertManifest, king_pointer: str,
                king_scores, evaluate: Callable[[str], list], fetch_dir: Callable[[str], Path],
                cfg, seed: str, recompute_selection: Callable | None = None,
                strict: bool = False) -> ExpertJudgement:
    if manifest.king_pointer != king_pointer:
        return ExpertJudgement(entry, False, "stale",
                               f"rebased onto {manifest.king_pointer}, king is {king_pointer}")
    declared = entry.declared
    if (any(t not in _targets(cfg) for t in declared)
            or len(declared) > int(getattr(cfg, "max_domains_per_leg", 1) or 1)):
        return ExpertJudgement(entry, False, "targets",
                               f"declared targets {list(declared)} are not 1..max_domains_per_leg"
                               f" configured lane-B targets")
    if recompute_selection is not None:
        from ..trainer.expert_lane import union_selection

        # DEC-CA-0059: the union of every declared target's re-derivation
        want = union_selection(recompute_selection(king_pointer, t, entry.exclude)
                               for t in declared)
        if {int(k): sorted(v) for k, v in want.items()} != entry.selection:
            return ExpertJudgement(entry, False, "selection",
                                   "expert rows differ from the ESFT re-derivation")
    # A FETCH failure propagates (a transient the caller retries): folding it
    # into a "merge" rejection would make one validator's network blip a
    # permanent verdict its peers never reached.
    kd, cd = fetch_dir(king_pointer), fetch_dir(entry.rebased_pointer)
    try:
        from safetensors.torch import load_file

        from ..trainer.expert_lane import _moe_names, check_finite_rows
        from ..trainer.moe_model import merge_expert_slices

        cand = load_file(str(cd / "weights.safetensors"))
        names = _moe_names(kd)
        merge_expert_slices(load_file(str(kd / "weights.safetensors")), cand,
                            entry.selection, names)
        if strict:
            check_finite_rows(cand, entry.selection, names)
    except Exception as e:  # noqa: BLE001 — any failure here refuses the entry
        return ExpertJudgement(entry, False, "merge", f"{type(e).__name__}: {e}")

    k = _arrays(king_scores)
    c = _arrays(evaluate(entry.rebased_pointer))
    if k["ids"] != c["ids"]:
        return ExpertJudgement(entry, False, "verdict", "king/candidate windows not paired")
    public = None
    rows = entry.public_rows or {}
    if rows or cfg.require_public:
        from ..eval.public_guards import MissingPublicGuard, pair_public_rows

        try:
            public = pair_public_rows(rows.get("king"), rows.get("candidate"),
                                      require=cfg.require_public,
                                      min_items=cfg.min_public_items).public
        except MissingPublicGuard as e:
            return ExpertJudgement(entry, False, "verdict", f"public guard missing: {e}")
    hts = tuple(getattr(cfg, "horizon_target_names", ()) or ())
    tv = judge_expert_targets(
        targets=declared, domains=k["domains"],
        king_qloss=k["qloss"], king_abs_target=k["abs"], king_mase=k["mase"],
        chal_qloss=c["qloss"], chal_abs_target=c["abs"], chal_mase=c["mase"],
        clusters=k["clusters"], public=public, params=cfg.verdict_params(),
        seed=f"{seed}|{entry.miner_hotkey}|{entry.domain}",
        horizons=[window_horizon(sid) for sid, _ in k["ids"]] if hts else None,
        horizon_targets=hts,
    )
    v = tv.verdicts[declared[0]]
    if len(declared) == 1:
        return ExpertJudgement(entry, v.passed, "verdict", v.reason, v, king_pointer=king_pointer)
    won = tv.won
    reason = (f"won {won}" if won else (tv.guard_reason or "no target won"))
    return ExpertJudgement(entry, bool(won), "verdict", reason, v, king_pointer=king_pointer,
                           targets=tv)


def _targets(cfg) -> tuple[str, ...]:
    return tuple(getattr(cfg, "targets", None) or cfg.domains)


def judge_manifest(manifest: ExpertManifest, **kw) -> list[ExpertJudgement]:
    """Judge every entry whose leading target is configured (an entry for an
    unconfigured one is ignored, as before); a multi-target entry declaring
    an unconfigured target, or more than ``max_domains_per_leg``, is
    rejected at stage ``targets``. ``strict`` (keyword, default False) arms
    the non-finite refusal of stage ``merge`` — callers pass
    ``cfg.merge_guard_at_settlement(settlement block)``."""
    return [judge_entry(e, manifest=manifest, **kw) for e in manifest.entries
            if e.domain in _targets(kw["cfg"])]


def apply_judgements(state: LaneBState, judgements: list[ExpertJudgement], *, round_id: str,
                     settle_block: int, tiers=None, qualify_only: bool = False) -> list[DomainWin]:
    """Record one win per domain (best target LCB, ties to the earliest commit
    block, then hotkey). Returns the wins recorded. Each win is tiered by its
    OWN target LCB (DEC-CA-0059: per target of a multi-target leg) under
    ``tiers`` — ``[lane_b] full_pay_tiers`` in force at the settlement (None =
    the shipped defaults) — and the tier is recorded on it."""
    from ..eval.expert_lane import full_pay
    from ..shared.config.v2 import FULL_PAY_TIERS

    tiers = FULL_PAY_TIERS if tiers is None else tiers
    best: dict[str, tuple] = {}
    for j in judgements:
        if not j.passed:
            continue
        # DEC-CA-0059: every target the entry won competes for its own slot
        for d, v in j.target_verdicts().items():
            if j.targets is not None and not v.passed:
                continue                  # a multi-target leg's unwon target
            key = (-v.target_lcb, j.entry.commit_block, j.entry.miner_hotkey)
            cur = best.get(d)
            if cur is None or key < cur[0]:
                best[d] = (key, j, v)
    wins = []
    for d in sorted(best):
        _, j, v = best[d]
        w = DomainWin(domain=d, hotkey=j.entry.miner_hotkey, uid=j.entry.miner_uid,
                      round_id=round_id, merge_block=int(settle_block),
                      gen_ref=j.entry.gen_ref, target_lcb=float(v.target_lcb),
                      rebased_pointer=j.entry.rebased_pointer, king_pointer=j.king_pointer,
                      selection={str(k): list(v) for k, v in j.entry.selection.items()},
                      **({"commit_block": j.entry.commit_block, "targets": j.entry.declared,
                          "qualified_block": int(settle_block)} if qualify_only else {})
                      ).with_full_pay(full_pay(float(v.target_lcb), tiers))
        if qualify_only:
            state.qualify(w)
        else:
            state.record(w)
        wins.append(w)
    return wins


def judgements_record(judgements: list[ExpertJudgement]) -> list[dict]:
    """The receipt's ``lane_b`` verdict rows (drop-when-empty at the caller).
    A single-target entry is one row, exactly as before; a multi-target entry
    (DEC-CA-0059) is one row PER declared target, each carrying the leg's
    ``targets``, the per-target gain level ``alpha`` (= alpha / k), its
    ``target_gain`` / ``inconclusive`` and the leg's ``guard_reason`` — what
    ``cascade-audit`` needs to re-derive every per-target verdict."""
    out = []
    for j in judgements:
        declared = j.entry.declared
        if len(declared) > 1:
            out.extend(_multi_rows(j, declared))
            continue
        v = j.verdict
        out.append({
            "hotkey": j.entry.miner_hotkey, "domain": j.entry.domain, "stage": j.stage,
            "passed": j.passed, "reason": j.reason,
            "target_lcb": None if v is None else v.target_lcb,
            "guard_lcbs": {} if v is None else v.guard_lcbs,
            "public_lcbs": {} if v is None else v.public_lcbs,
        })
    return out


def _num(x):
    return None if x is None or (isinstance(x, float) and x != x) else float(x)


def _multi_rows(j: ExpertJudgement, declared: tuple[str, ...]) -> list[dict]:
    tv = j.targets
    rows = []
    for t in declared:
        v = None if tv is None else tv.verdicts.get(t)
        rows.append({
            "hotkey": j.entry.miner_hotkey, "domain": t, "stage": j.stage,
            "passed": bool(v is not None and v.passed),
            "reason": j.reason if v is None else v.reason,
            "target_lcb": None if v is None else v.target_lcb,
            "guard_lcbs": {} if v is None else v.guard_lcbs,
            "public_lcbs": {} if v is None else v.public_lcbs,
            "targets": list(declared), "leg": j.entry.rebased_pointer,
            "alpha": None if tv is None else float(tv.alpha),
            "target_gain": None if v is None else _num(v.target_gain),
            "target_n": None if v is None else int(v.target_n),
            "target_clusters": None if v is None else int(v.target_clusters),
            "inconclusive": "" if v is None else v.inconclusive,
            "guard_reason": "" if tv is None else tv.guard_reason,
        })
    return rows
