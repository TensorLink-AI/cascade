"""Shared spec and helpers for the W-S2 acceptance tests: lane B guard/target
exclusion for the era's §16 sentinel (D33 / D35; TS4).

Everything here comes from docs/plans/LANE_D_PLAN.md: the W-S2 row (§3.0),
§16 rule 2 ("Lane B: it's neither a guard nor a target that era (D35)"),
D35 (§6 and the §17.4 register), TS4 and the build rules (§3.1). WB18
already built the TARGET half (a suite-target entry on the sentinel is
deferred, ``deferred: sentinel_target``); these tests pin the GUARD half and
re-check that the target half still holds.

The scenarios run the REAL lane B judging code without torch:

- the consensus judge: ``cascade.validator.expert_lane.judge_manifest`` /
  ``judge_entry`` over a real ``ExpertManifest`` and real ``WindowScore``
  lists, with the public guards built by the real ``pair_public_rows`` and
  judged by the real ``judge_expert_targets``;
- the warm-start acceptance: the real
  ``cascade.validator.merge_acceptance.judge_acceptance``;
- the validator: the real ``LaneBMixin._lane_b_judge_experts`` on a runner
  whose storage, window source and evaluator are stand-ins, holding W-S1's
  recorded draw for the settlement (``ValidatorRunner._sentinel_for``);
- the trainer: the real ``LaneBDriver.tick`` / ``settle`` (WB18's scenario:
  real intake, funding queue and driver).

Stand-ins, all for the GPU boundary or external services (build rule 2):

- ``gpu_boundary``: ``safetensors.torch.load_file`` and
  ``moe_model.merge_expert_slices`` (the byte-check merge of a candidate's
  expert rows, torch tensors) plus ``expert_lane._moe_names`` (reads the
  MoE config through the torch model module), and the trainer's
  ``settle_leg`` (the torch rebase of a finished leg);
- the window scores (the eval forward), the checkpoint store, the chain's
  block hashes and the trainer's ``ops.judge`` (WB18's ``Ops`` plus storage).

No module in W-S2's PATHS is imported at module level, so a missing name is
reported inside the test that needs it.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import inspect
import json
import sys
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"

# §16 rule 1 default ``sentinel_suites`` (the four bench suites; the lane B
# public guard keys are ``<suite>_crps`` / ``<suite>_mase``).
SUITES = ("gift", "boom", "time", "fev")
METRICS = ("crps", "mase")
SENTINEL = "time"

ERA_LEN = 1000
SETTLE_BLOCK = 2 * ERA_LEN          # the settlement that closes era 1 (starts at 1000)
SETTLE_ERA_START = ERA_LEN

DOMAINS = ("energy", "web")
N_PER_DOMAIN = 120
N_ITEMS = 40

HK = "5HotkeyGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
GEN = "miner/gen-gg@sha256:" + hashlib.sha256(b"w-s2 gen").hexdigest()


def pointer(name: str) -> str:
    from cascade.shared.manifest import format_trained_pointer

    return format_trained_pointer(f"cascade/{name}@sha256:"
                                  + hashlib.sha256(f"w-s2 {name}".encode()).hexdigest())


KING_PTR_NAME = "w-s2-king"
CAND_PTR_NAME = "w-s2-cand"
# the trainer scenario's era-0 / era-1 kings (trained pointers, as settle needs)
TRAINER_PREV_KING = ("metro-v1:trained:hippius:cascade/w-s2-prev@sha256:"
                     + hashlib.sha256(b"w-s2 prev king").hexdigest())
TRAINER_KING = ("metro-v1:trained:hippius:cascade/w-s2-tking@sha256:"
                + hashlib.sha256(b"w-s2 trainer king").hexdigest())


def guard_keys(suite: str) -> tuple[str, ...]:
    return tuple(f"{suite}_{m}" for m in METRICS)


# ── torch guard ──────────────────────────────────────────────────────────────


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(
                f"W-S2: '{name}' must not be imported (the sentinel exclusion is guard "
                "selection, not model code; the merge is the GPU-boundary stand-in)", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


# ── the GPU boundary ─────────────────────────────────────────────────────────


class MergeError(Exception):
    """The stand-in moe_model's merge refusal type."""


@contextmanager
def gpu_boundary():
    """Replace the torch-only weight I/O of ``judge_entry`` (load the king and
    candidate tensors, byte-check the candidate's expert rows against the
    king) with a pass-through; everything after it (the paired verdict) is
    the real code."""
    from cascade.trainer import expert_lane as tel

    st = types.ModuleType("safetensors.torch")
    st.load_file = lambda path: {}
    moe = types.ModuleType("cascade.trainer.moe_model")
    moe.merge_expert_slices = lambda king, cand, selection, names: cand
    moe.MergeError = MergeError
    saved = {k: sys.modules.get(k) for k in ("safetensors.torch", "cascade.trainer.moe_model")}
    real_names = tel._moe_names
    sys.modules["safetensors.torch"] = st
    sys.modules["cascade.trainer.moe_model"] = moe
    tel._moe_names = lambda ckpt_dir: [("blocks.0.ffn.w_in", "blocks.0.ffn.w_out")]
    try:
        yield
    finally:
        tel._moe_names = real_names
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ── the scenario: one domain leg with public guards ──────────────────────────


def lane_b_cfg(*, require_public=(), suite_targets: bool = False):
    """``[lane_b]`` through the real parser: two domains, small floors so the
    fixture's 120 windows per domain are conclusive."""
    from cascade.shared.config import parse_lane_b

    raw = {"from_block": 1, "domains": list(DOMAINS), "open_tokens": 100,
           "routing_jaccard": 0.7, "bootstrap_B": 300, "min_target_windows": 20,
           "min_guard_windows": 20, "require_public": list(require_public)}
    if suite_targets:
        raw.update({"suite_targets": True, "contamination_hashset":
                    "sha256:" + hashlib.sha256(b"w-s2 contamination hash set").hexdigest()})
    return parse_lane_b(raw)


def window_scores(effect: dict[str, float]) -> list:
    """King-paired window scores: ``effect[domain]`` multiplies the loss of
    every window of that domain (1.0 = the king's own numbers)."""
    from cascade.eval.scoring import WindowScore

    rng = np.random.default_rng(20261122)
    out = []
    for d in DOMAINS:
        f = float(effect.get(d, 1.0))
        for i in range(N_PER_DOMAIN):
            q = rng.uniform(0.1, 1.0, size=9)
            out.append(WindowScore(series_id=f"{d}-w{i}", mase=float(rng.uniform(0.5, 1.5)) * f,
                                   qloss_per_q=q * f, abs_target=float(rng.uniform(5.0, 10.0)),
                                   domain=d, source=f"{d}-feed-{i % 30}"))
    return out


KING_SCORES_EFFECT: dict[str, float] = {}
GAIN_EFFECT = {"energy": 0.85}       # the leg's target improves, web unchanged


def bench_rows(factors: dict[str, float], *, absent=()) -> dict:
    """Item rows (``item_rows_from_report`` shape) per suite; ``factors[s]``
    multiplies the suite's CRPS and MASE (1.0 = the king's numbers). A suite
    in ``absent`` is reported ``skipped`` with no rows."""
    rng = np.random.default_rng(7)
    out = {}
    for s in SUITES:
        base = rng.uniform(0.2, 2.0, size=(N_ITEMS, 2))
        if s in absent:
            out[s] = {"status": "skipped", "rows": []}
            continue
        f = float(factors.get(s, 1.0))
        out[s] = {"status": "ok", "rows": [
            {"item": f"{s}/item-{i:03d}", "crps": float(base[i, 0]) * f,
             "mase": float(base[i, 1]) * f} for i in range(N_ITEMS)]}
    return out


KING_ROWS = bench_rows({})
# the candidate: a hair better on every suite, but 30% worse on the sentinel
SENTINEL_ONLY = {"gift": 0.99, "boom": 0.99, "time": 1.30, "fev": 0.99}
CLEAN = {"gift": 0.99, "boom": 0.99, "time": 0.99, "fev": 0.99}


def entry(cand_rows: dict | None, *, king_rows: dict | None = None, domain: str = "energy"):
    from cascade.shared.expert_manifest import ExpertEntry

    public = {} if cand_rows is None else {"king": king_rows or KING_ROWS,
                                           "candidate": cand_rows}
    return ExpertEntry(miner_hotkey=HK, miner_uid=11, gen_ref=GEN, domain=domain,
                       selection={0: [0, 1, 2, 3]}, exclude={},
                       base_king_pointer=pointer(KING_PTR_NAME),
                       rebased_pointer=pointer(CAND_PTR_NAME), corpus_digest="corpus",
                       train_block=1500, commit_block=900, public_rows=public)


def expert_manifest(*entries, round_id: str = str(SETTLE_BLOCK)):
    from cascade.shared.expert_manifest import ExpertManifest

    return ExpertManifest(round_id=round_id, created_block=SETTLE_BLOCK,
                          king_pointer=pointer(KING_PTR_NAME), lineage_tokens=1000,
                          entries=tuple(entries))


def has_param(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def judge(manifest, cfg, tmp: Path, *, cand_effect=None, **kw) -> list:
    """The real consensus judge of ``manifest`` (``judge_manifest``); ``kw``
    passes ``sentinel=`` when the test sets it (checked to exist first, so a
    missing keyword is an assertion, not a TypeError)."""
    from cascade.validator.expert_lane import judge_entry, judge_manifest

    if "sentinel" in kw:
        assert has_param(judge_manifest, "sentinel") or has_param(judge_entry, "sentinel"), (
            "judge_manifest / judge_entry take the era's sentinel suite as keyword "
            "`sentinel` (W-S2: §16 rule 2, D35 — the sentinel is not a lane B guard that era)")
    king = window_scores(KING_SCORES_EFFECT)
    cand = window_scores(GAIN_EFFECT if cand_effect is None else cand_effect)
    cand_ptr = pointer(CAND_PTR_NAME)
    with torch_blocked(), gpu_boundary():
        return judge_manifest(manifest, king_pointer=manifest.king_pointer, king_scores=king,
                              evaluate=lambda p: cand if p == cand_ptr else king,
                              fetch_dir=lambda p: Path(tmp), cfg=cfg, seed="w-s2", **kw)


def record(judgements: list) -> str:
    from cascade.validator.expert_lane import judgements_record

    return json.dumps(judgements_record(judgements), sort_keys=True, default=str)


# ── the warm-start acceptance (pure) ─────────────────────────────────────────


def acceptance(cfg, cand_factors: dict[str, float], *, absent=(), **kw):
    """The real ``judge_acceptance`` of one qualified energy win: the
    candidate's loss is 0.85× on energy against both the accepted-so-far
    checkpoint and the original init; ``cand_factors`` shape its public
    rows (both baselines carry the king's rows)."""
    from cascade.shared.lane_b import DomainWin
    from cascade.validator.merge_acceptance import judge_acceptance

    if "sentinel" in kw:
        assert has_param(judge_acceptance, "sentinel"), (
            "judge_acceptance takes the era's sentinel suite as keyword `sentinel` "
            "(W-S2: the warm-start acceptance's public guards are lane B guards too)")
    win = DomainWin(domain="energy", hotkey=HK, uid=11, round_id=str(SETTLE_BLOCK),
                    merge_block=SETTLE_BLOCK, gen_ref=GEN, target_lcb=0.05,
                    rebased_pointer=pointer(CAND_PTR_NAME), king_pointer=pointer(KING_PTR_NAME),
                    selection={0: [0, 1, 2, 3]}, commit_block=900)
    base = window_scores({})
    cand = window_scores(GAIN_EFFECT)
    rows = {"before": KING_ROWS, "anchor": KING_ROWS,
            "candidate": bench_rows(cand_factors, absent=absent)}
    with torch_blocked():
        return judge_acceptance([win], anchor_scores=base, before_scores=base,
                                candidate_scores=cand, public_rows=rows, cfg=cfg,
                                seed="w-s2", **kw)


# ── chain helpers ────────────────────────────────────────────────────────────


def hash_drawing(suite: str, era: int) -> str:
    """A ``0x`` + 64-hex block hash whose W-S1 draw over ``SUITES`` is ``suite``."""
    from cascade.shared.sentinel import sentinel_for_era

    for i in range(10_000):
        h = "0x" + hashlib.sha256(f"w-s2 era {era} try {i}".encode()).hexdigest()
        if sentinel_for_era(h, SUITES) == suite:
            return h
    raise AssertionError(f"no block hash draws {suite!r}")


def chain_cfg(lb, *, sentinel_from_block: int):
    from cascade.shared.config.rounds import RoundConfig

    v2 = SimpleNamespace(sentinel_from_block=int(sentinel_from_block),
                         sentinel_suites=SUITES, sentinel_ratchet=True, sentinel_tau=0.01)
    return SimpleNamespace(lane_b=lb, v2=v2,
                           round=RoundConfig(epoch_blocks=ERA_LEN, era_settlements=1))


# ── the validator: LaneBMixin._lane_b_judge_experts ──────────────────────────


def validator_round(tmp: Path, cfg, em, *, sentinel_record: dict | None):
    """Run the real ``_lane_b_judge_experts`` for settlement ``em.round_id``
    on a runner that holds W-S1's recorded draw (``_settlement_sentinel``;
    ``None`` = unarmed). Storage, windows and the evaluator are stand-ins."""
    from cascade.shared.manifest import TrainedEntry, TrainingManifest
    from cascade.validator.lane_b import LaneBMixin
    from cascade.validator.loop import ValidatorRunner

    king_ptr, cand_ptr = pointer(KING_PTR_NAME), pointer(CAND_PTR_NAME)
    king_scores, cand_scores = window_scores({}), window_scores(GAIN_EFFECT)
    text = em.to_json()

    class Runner(LaneBMixin):
        _sentinel_for = ValidatorRunner._sentinel_for

        def __init__(self) -> None:
            self.cfg = cfg
            self.verify_signatures = False
            self._lane_b_window_source = object()
            self._settlement_sentinel = (None if sentinel_record is None
                                         else (str(em.round_id), sentinel_record))
            self.reads: list[str] = []

        def _epoch_start_block(self, manifest) -> int:
            return (int(manifest.created_block) // ERA_LEN) * ERA_LEN

        def _lane_b_read(self, key: str) -> str:
            self.reads.append(key)
            return text

        def _verdict_windows(self, source, base_seed, *, block=None) -> list:
            return []

        def _training_for(self, manifest):
            return None

        def _evaluate(self, e, windows, training, *, block=None) -> list:
            return cand_scores if e.trained_pointer == cand_ptr else king_scores

        def _fetch_checkpoint_dir(self, e) -> Path:
            return Path(tmp)

    king = TrainedEntry(miner_hotkey="5King", miner_uid=1, role="king", gen_ref=GEN,
                        trained_pointer=king_ptr, corpus_digest="k", train_block=1500)
    manifest = TrainingManifest(round_id=str(em.round_id), created_block=SETTLE_BLOCK,
                                contract_digest="c", base_arch_digest="b", eval_dataset="d",
                                entries=[king], expert_manifest_sha256=em.digest())
    runner = Runner()
    with torch_blocked(), gpu_boundary():
        return runner._lane_b_judge_experts(manifest)


def sentinel_rec(suite: str) -> dict:
    return {"suite": suite, "block": SETTLE_ERA_START, "block_hash": hash_drawing(suite, 1)}


# ── the trainer: LaneBDriver.settle (WB18's scenario + storage) ──────────────


def trainer_scenario(tmp: Path, *, sentinel_from_block: int, require_public=("gift", "time"),
                     suite_targets: bool = False, king_rows: dict | None = None,
                     era1_sentinel: str = SENTINEL):
    """WB18's real intake / funding queue / driver with one funded domain
    entry (energy) trained and finished in era 1, ready to settle at block
    2000 (the boundary closing era 1, whose sentinel is drawn from block
    1000's hash). ``king_rows`` are the bench rows ``ops.bench_rows`` reads
    for the king (candidates read ``CLEAN`` rows)."""
    from tests.accept.wb18 import helpers as wb18

    sc = wb18.Scenario(Path(tmp), suite_targets=suite_targets,
                       sentinel_from_block=sentinel_from_block)
    sc.cfg.lane_b = replace(sc.cfg.lane_b, require_public=tuple(require_public))
    ops = sc.ops
    ops.judge_calls = []
    ops.published = []
    ops.uploaded = []
    ops.king_rows = KING_ROWS if king_rows is None else king_rows
    ops.king_dir = Path(tmp) / "ckpt"

    def bench_rows_(ckpt_dir):
        return ops.king_rows if Path(ckpt_dir) == ops.king_dir else bench_rows(CLEAN)

    def upload(ckpt_dir, repo):
        ops.uploaded.append(repo)
        return pointer(repo)

    def judge_(manifest, round_id, block, *args, **kw):
        ops.judge_calls.append({"args": (manifest, round_id, block) + tuple(args), "kw": dict(kw)})
        return []

    ops.bench_rows = bench_rows_
    ops.upload = upload
    ops.publish_expert_manifest = ops.published.append
    ops.judge = judge_
    for era, suite in ((1, era1_sentinel), (2, "gift")):
        sc.set_sentinel(era, suite)
    sc.fund(wb18.HK_E, 5, "energy")
    sc.driver.add_lineage_tokens(1000)
    assert sc.tick(0, TRAINER_PREV_KING) == []
    sc.tick(1, TRAINER_KING)
    assert sc.driver.st.gate_opened_block > 0, sc.driver.st.routing
    assert ops.dispatched, "fixture: the funded energy entry did not start"
    for leg, done in ops.dispatched:
        done(leg, pointer(f"leg-{leg.sub.hotkey[7:9]}"), "corpus", 1500, "cpu")
    assert sc.driver.st.finished, "fixture: the finished leg was not recorded"
    return sc


@contextmanager
def stub_settle_leg(tmp: Path):
    """The torch rebase of a finished leg onto the current king."""
    from cascade.trainer import lane_b_driver

    real = lane_b_driver.settle_leg

    def fake(king_dir, base_dir, leg_dir, selection, out, *, refuse_nonfinite=False):
        Path(out).mkdir(parents=True, exist_ok=True)
        return Path(out)

    lane_b_driver.settle_leg = fake
    try:
        yield
    finally:
        lane_b_driver.settle_leg = real


def settle(sc, tmp: Path):
    with torch_blocked(), gpu_boundary(), stub_settle_leg(tmp):
        return sc.driver.settle(round_id=str(SETTLE_BLOCK), block=SETTLE_BLOCK, era_index=1,
                                king_pointer=TRAINER_KING, work_dir=Path(tmp) / "work")


def settle_ok(sc, tmp: Path):
    """``settle`` where the contract says the settlement goes through: a
    hold (``OSError``, e.g. a required public suite missing) is a failed
    expectation, reported as such."""
    try:
        return settle(sc, tmp)
    except OSError as e:
        raise AssertionError(f"the settlement held instead of settling: {e}") from e
