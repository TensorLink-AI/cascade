"""Shared spec and helpers for the WB26 acceptance tests: lane B suite-target
judging (DEC-CA-0071 §2-§3, Amendment rev6.19; plan §3 WB26 row, rev6.19 (G)).

The rule under test (rev6.19 (G), owner 2026-11-24; full-pay input,
orchestrator 2026-11-24):

- a suite target ``suite:<s>`` wins when CRPS OR MASE clears
  ``LCB_{alpha/(2k)} >= gain_margin_public`` (0.01) on the whole suite's
  paired items with a strictly positive point gain;
- the metric that did not clear must be non-inferior, ``LCB_alpha >=
  -tau_public``;
- fewer than ``min_public_items`` paired items is ``inconclusive: items``;
- the win's ``target_lcb`` (the ``full_pay`` tier input) is the SMALLER
  cleared LCB;
- every existing guard holds, plus guard 4 (whole-pool point gain >= 0) and
  the anchor guard (domain / horizon / public guards against the anchor king
  at ``tau_anchor_domain`` / ``tau_anchor_public``, default 0.0); no anchor
  is ``anchor_missing`` (fail-closed).

Everything runs the REAL code: ``judge_expert_targets`` (eval), the
validator's ``judge_manifest`` / ``judgements_record`` and
``LaneBMixin._lane_b_judge_experts``, the real ``[lane_b]`` parser and the
audit's lane B row check. Stand-ins are W-S2's GPU boundary (the torch
weight I/O and byte-check merge) and the window scores / bench rows.

No module in WB26's PATHS is imported at module level.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from tests.accept.w_s2 import helpers as ws2

REPO = Path(__file__).resolve().parents[3]
SUITES = ("gift", "boom", "time", "fev")
METRICS = ("crps", "mase")
DOMAINS = ws2.DOMAINS
N_ITEMS = 40
B = 400
ALPHA = 0.05
SEED = "wb26"
GAIN_MARGIN_PUBLIC = 0.01
TAU_PUBLIC = 0.01

ANCHOR_PTR_NAME = "wb26-anchor"
pointer = ws2.pointer
KING_PTR = ws2.pointer(ws2.KING_PTR_NAME)
CAND_PTR = ws2.pointer(ws2.CAND_PTR_NAME)
ANCHOR_PTR = ws2.pointer(ANCHOR_PTR_NAME)


def has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def need_anchor_param():
    from cascade.eval.expert_lane import judge_expert_targets

    assert has_param(judge_expert_targets, "anchor"), (
        "judge_expert_targets takes the anchor king's scores as keyword `anchor` "
        "(WB26, DEC-CA-0071 §3 guard 5)")


# ── synthetic inputs (eval level) ────────────────────────────────────────────


def items(seed: int, n: int = N_ITEMS) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.2, 2.0, size=n)


def public(factors: dict | None = None, *, n: dict | None = None) -> dict:
    """Paired ``{<s>_<m>: (king, chal)}``: ``factors[(s, m)]`` multiplies the
    candidate's items (1.0 = the king's numbers); ``n[s]`` trims a suite."""
    factors, n = factors or {}, n or {}
    out = {}
    for i, s in enumerate(SUITES):
        for j, m in enumerate(METRICS):
            k = items(100 + 10 * i + j, int(n.get(s, N_ITEMS)))
            f = float(factors.get((s, m), 1.0))
            out[f"{s}_{m}"] = (list(k), list(k * f))
    return out


def pool(effect: dict | None = None):
    """Per-window arrays from W-S2's window scores: ``effect[domain]``
    multiplies the candidate's losses on that domain."""
    k = ws2.window_scores({})
    c = ws2.window_scores(effect or {})

    def arr(scores):
        return (np.asarray([s.qloss_per_q for s in scores]),
                np.asarray([s.abs_target for s in scores]),
                np.asarray([s.mase for s in scores]))

    doms = [s.domain for s in k]
    clusters = [s.source for s in k]
    return doms, clusters, arr(k), arr(c)


def anchor_input(pub: dict, *, pool_effect: dict | None = None,
                 public_factors: dict | None = None, drop: tuple[str, ...] = ()) -> dict:
    """The anchor king: by default identical to the king (so the anchor
    guard re-states the king guards at tau 0). ``pool_effect[d]`` scales the
    anchor's losses on domain d (< 1 = the anchor is better); public
    ``public_factors[(s, m)]`` scales the anchor's items."""
    _, _, (kq, ka, km), _ = pool()
    eff = pool_effect or {}
    f = np.asarray([float(eff.get(s.domain, 1.0)) for s in ws2.window_scores({})])
    pf = public_factors or {}
    apub = {}
    for name, (k, c) in pub.items():
        if name in drop:
            continue
        s, m = name.split("_")
        apub[name] = (list(np.asarray(k) * float(pf.get((s, m), 1.0))), list(c))
    return {"pointer": ANCHOR_PTR, "qloss": kq * f[:, None], "abs": ka, "mase": km * f,
            "public": apub}


def params(**over):
    from cascade.eval.expert_lane import ExpertLaneParams

    base = {"alpha": ALPHA, "B": B, "min_target_windows": 20, "min_guard_windows": 20,
            "min_public_items": 10}
    return ExpertLaneParams(**{**base, **over})


def judge(targets, *, factors=None, pool_effect=None, n=None, anchor="same",
          anchor_kw=None, horizons=None, horizon_targets=(), **over):
    """Run the real ``judge_expert_targets``; ``anchor="same"`` builds the
    default anchor (= the king), ``None`` passes no anchor."""
    from cascade.eval.expert_lane import judge_expert_targets

    need_anchor_param()
    doms, clusters, (kq, ka, km), (cq, ca, cm) = pool(pool_effect)
    pub = public(factors, n=n)
    if anchor == "same":
        anchor = anchor_input(pub, **(anchor_kw or {}))
    return judge_expert_targets(
        targets=list(targets), domains=doms,
        king_qloss=kq, king_abs_target=ka, king_mase=km,
        chal_qloss=cq, chal_abs_target=ca, chal_mase=cm,
        clusters=clusters, public=pub, params=params(**over), seed=SEED,
        horizons=horizons, horizon_targets=horizon_targets, anchor=anchor)


def suite_rec(v) -> dict:
    rec = getattr(v, "suite", None)
    assert isinstance(rec, dict) and rec, (
        f"a conclusive suite-target verdict records its gain test in `suite` "
        f"(cleared_metric, crps_lcb, mase_lcb, ...); got {rec!r} (reason {v.reason!r})")
    return rec


# ── validator level ──────────────────────────────────────────────────────────


def bench_rows(factors: dict | None = None, *, absent=()) -> dict:
    """Item rows per suite with per-metric factors ``factors[(s, m)]``."""
    factors = factors or {}
    out = {}
    for i, s in enumerate(SUITES):
        if s in absent:
            out[s] = {"status": "skipped", "rows": []}
            continue
        kc, km = items(100 + 10 * i), items(101 + 10 * i)
        fc, fm = float(factors.get((s, "crps"), 1.0)), float(factors.get((s, "mase"), 1.0))
        out[s] = {"status": "ok", "rows": [
            {"item": f"{s}/item-{j:03d}", "crps": float(kc[j]) * fc, "mase": float(km[j]) * fm}
            for j in range(N_ITEMS)]}
    return out


def lane_b_cfg(**over):
    from cascade.shared.config import parse_lane_b

    raw = {"from_block": 1, "domains": list(DOMAINS), "open_tokens": 100,
           "routing_jaccard": 0.7, "bootstrap_B": B, "min_target_windows": 20,
           "min_guard_windows": 20, "require_public": list(SUITES),
           "suite_targets": True, "contamination_hashset":
           "sha256:" + hashlib.sha256(b"wb26 contamination hash set").hexdigest()}
    raw.update(over)
    return parse_lane_b(raw)


def entry(domain: str, cand: dict, *, anchor: dict | None = None, uid: int = 11,
          hk: str = ws2.HK, rebased: str | None = None):
    from cascade.shared.expert_manifest import ExpertEntry

    rows = {"king": bench_rows({}), "candidate": cand}
    if anchor is not None:
        rows["anchor"] = anchor
    return ExpertEntry(miner_hotkey=hk, miner_uid=uid, gen_ref=ws2.GEN, domain=domain,
                       selection={0: [0, 1, 2, 3]}, exclude={},
                       base_king_pointer=KING_PTR, rebased_pointer=rebased or CAND_PTR,
                       corpus_digest="corpus", train_block=1500, commit_block=900,
                       public_rows=rows)


def judge_manifest(em, cfg, tmp: Path, *, anchor_pointer="set", cand_effect=None,
                   anchor_effect=None, calls: list | None = None):
    """The real validator ``judge_manifest``; ``evaluate`` serves the king,
    the candidate and the anchor and logs every pointer it scores."""
    from cascade.validator.expert_lane import judge_entry, judge_manifest

    assert has_param(judge_manifest, "anchor_pointer") or has_param(judge_entry,
                                                                    "anchor_pointer"), (
        "judge_manifest / judge_entry take the anchor king as keyword `anchor_pointer` (WB26)")
    king = ws2.window_scores({})
    cand = ws2.window_scores(cand_effect or {})
    anc = ws2.window_scores(anchor_effect or {})
    log = calls if calls is not None else []

    def evaluate(p):
        log.append(p)
        return {CAND_PTR: cand, ANCHOR_PTR: anc}.get(p, king)

    kw = {} if anchor_pointer is None else {
        "anchor_pointer": ANCHOR_PTR if anchor_pointer == "set" else anchor_pointer}
    with ws2.torch_blocked(), ws2.gpu_boundary():
        return judge_manifest(em, king_pointer=em.king_pointer, king_scores=king,
                              evaluate=evaluate, fetch_dir=lambda p: Path(tmp), cfg=cfg,
                              seed="wb26", **kw)


def record(judgements) -> list:
    from cascade.validator.expert_lane import judgements_record

    return json.loads(json.dumps(judgements_record(judgements), sort_keys=True, default=str))


def validator_round(tmp: Path, cfg, em, *, warm_start_ckpt: str = "", warm_start_base: str = "",
                    merges: tuple = ()):
    """The real ``LaneBMixin._lane_b_judge_experts`` for ``em`` on a runner
    whose storage / windows / evaluator are stand-ins; the training manifest
    carries the era's warm-start fields. Returns (round, evaluated pointers)."""
    from cascade.shared.manifest import TrainedEntry, TrainingManifest
    from cascade.validator.lane_b import LaneBMixin

    king_scores, cand_scores = ws2.window_scores({}), ws2.window_scores(ws2.GAIN_EFFECT)
    text = em.to_json()
    seen: list[str] = []

    class Runner(LaneBMixin):
        def __init__(self) -> None:
            self.cfg = ws2.chain_cfg(cfg, sentinel_from_block=0)
            self.verify_signatures = False
            self._lane_b_window_source = object()

        def _epoch_start_block(self, manifest) -> int:
            return (int(manifest.created_block) // ws2.ERA_LEN) * ws2.ERA_LEN

        def _lane_b_read(self, key: str) -> str:
            return text

        def _lane_b_sentinel_kw(self, manifest) -> dict:
            return {}

        def _verdict_windows(self, source, base_seed, *, block=None) -> list:
            return []

        def _training_for(self, manifest):
            return None

        def _evaluate(self, e, windows, training, *, block=None) -> list:
            seen.append(e.trained_pointer)
            return cand_scores if e.trained_pointer == CAND_PTR else king_scores

        def _fetch_checkpoint_dir(self, e) -> Path:
            return Path(tmp)

    king = TrainedEntry(miner_hotkey="5King", miner_uid=1, role="king", gen_ref=ws2.GEN,
                        trained_pointer=KING_PTR, corpus_digest="k", train_block=1500)
    manifest = TrainingManifest(round_id=str(em.round_id), created_block=ws2.SETTLE_BLOCK,
                                contract_digest="c", base_arch_digest="b", eval_dataset="d",
                                entries=[king], expert_manifest_sha256=em.digest(),
                                warm_start_ckpt=warm_start_ckpt, warm_start_base=warm_start_base,
                                warm_start_merges=merges)
    with ws2.torch_blocked(), ws2.gpu_boundary():
        out = Runner()._lane_b_judge_experts(manifest)
    return out, seen


# ── trainer level ────────────────────────────────────────────────────────────


def trainer_scenario(tmp: Path, *, target: str = "suite:gift"):
    """WB18's real intake / funding queue / driver (via W-S2's scenario
    shape) with one funded entry for ``target`` trained and finished in era
    1, ready to settle at block 2000; the §16 sentinel is off. ``ops.fetch``
    / ``ops.bench_rows`` log what they read; the anchor checkpoint has its
    own directory and its own (king-identical) rows."""
    from dataclasses import replace

    from tests.accept.wb18 import helpers as wb18

    sc = wb18.Scenario(Path(tmp), suite_targets=True, sentinel_from_block=0)
    sc.cfg.lane_b = replace(sc.cfg.lane_b, require_public=SUITES)
    ops = sc.ops
    ops.judge_calls, ops.published, ops.fetched, ops.benched = [], [], [], []
    king_rows = bench_rows({})
    anchor_dir = Path(tmp) / "anchor"

    def fetch(p):
        ops.fetched.append(p)
        return anchor_dir if p == ANCHOR_PTR else Path(tmp) / "ckpt"

    def bench_rows_(ckpt_dir):
        ops.benched.append(Path(ckpt_dir))
        if Path(ckpt_dir) in (Path(tmp) / "ckpt", anchor_dir):
            return king_rows
        return bench_rows({("gift", "crps"): 0.95})

    def judge_(manifest, round_id, block, *args, **kw):
        ops.judge_calls.append({"manifest": manifest, "kw": dict(kw)})
        return []

    ops.fetch, ops.bench_rows, ops.judge = fetch, bench_rows_, judge_
    ops.upload = lambda ckpt_dir, repo: ws2.pointer(repo)
    ops.publish_expert_manifest = ops.published.append
    sc.fund(wb18.HK_E, 5, target)
    sc.driver.add_lineage_tokens(1000)
    assert sc.tick(0, ws2.TRAINER_PREV_KING) == []
    sc.tick(1, ws2.TRAINER_KING)
    assert ops.dispatched, "fixture: the funded entry did not start"
    for leg, done in ops.dispatched:
        done(leg, ws2.pointer(f"leg-{leg.sub.hotkey[7:9]}"), "corpus", 1500, "cpu")
    assert sc.driver.st.finished, "fixture: the finished leg was not recorded"
    return sc


def trainer_settle(sc, tmp: Path, **kw):
    from cascade.trainer.lane_b_driver import LaneBDriver

    if "anchor_pointer" in kw:
        assert has_param(LaneBDriver.settle, "anchor_pointer"), (
            "LaneBDriver.settle takes the era's anchor king as keyword `anchor_pointer` (WB26)")
    with ws2.torch_blocked(), ws2.gpu_boundary(), ws2.stub_settle_leg(tmp):
        return sc.driver.settle(round_id=str(ws2.SETTLE_BLOCK), block=ws2.SETTLE_BLOCK,
                                era_index=1, king_pointer=ws2.TRAINER_KING,
                                work_dir=Path(tmp) / "work", **kw)
