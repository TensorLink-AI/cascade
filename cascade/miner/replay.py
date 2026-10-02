"""Round replay — score a generator on a PAST round, against that round's king.

Every scored round is reproducible from public artifacts:

* the signed **receipt** carries the round's seeds (``generation_seed`` /
  ``training_seed``), the eval draw seed (``base_seed``), the epoch block, the
  verdict window ids, and the king's signed per-window scores
  (``entry_scores``);
* its embedded **manifest** carries the warm-start init the round trained from
  (``warm_start_ckpt``) and the pinned pool snapshot (``eval_pool_sha256``);
* the **revealed snapshot** (``cascade-pool reveal``, ~48h after it retires) is
  the exact pool those windows were cut from.

A replay trains ONLY the candidate — same init, same seeds, same contract as the
round's legs — scores it on the exact verdict windows, and pairs it with the
king's receipt scores under the round's own recorded KOTH params. The king is
never retrained: its side of the duel is the validators' own signed numbers.

The window draw is self-verifying: the rebuilt windows must reproduce the
receipt's ``window_ids`` exactly, or the replay refuses to score.

Still directional, not a verdict: the candidate trains on YOUR hardware/software
while the king's numbers come from the operator's leg, and the replay judges the
candidate as the round's only challenger (no cohort correction). A full-budget
replay of the king's own generator against its receipt measures that gap.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from .score import train_and_evaluate

log = logging.getLogger("cascade.miner.replay")

# Marker the reveal writes into each snapshot folder (cascade.pool.reveal).
SHA_MARKER = "POOL_SHA256"


class ReplayError(RuntimeError):
    """The round cannot be replayed faithfully (missing data or a mismatch)."""


def _norm_digest(d: str) -> str:
    d = (d or "").strip().lower()
    return d.split(":", 1)[1] if d.startswith("sha256:") else d


@dataclass(frozen=True)
class ReplayRound:
    """Everything needed to re-run one round's duel for a new challenger."""

    receipt: object               # RoundReceipt
    snapshot_dir: Path
    windows: list                 # EvalWindow, the round's exact verdict windows
    seeds: object                 # RoundSeeds — the round's training seeds
    init_pointer: str             # manifest warm_start_ckpt ("" = random init)
    contract_block: int | None    # era start (DEC-CA-0047), None = base contract
    size: str                     # the king entry's size tag
    king_scores: list             # WindowScore, paired with ``windows``
    baseline_scores: list | None  # the init's scores (increment / init gate), if recorded
    king_gpu: str                 # GPU the king's leg ran on (match it for fidelity)
    contract: object = None       # the round's own training contract (from its manifest)
    contract_match: bool = True   # rebuilt contract hashes to the manifest's digest
    contract_overrides: dict = field(default_factory=dict)   # fields taken from the round


@dataclass(frozen=True)
class ReplayResult:
    round_id: str
    geomean: float                # candidate, lower is better
    king_geomean: float           # the king's receipt score on the same windows
    n_windows: int
    full_budget: bool             # False ⇒ no verdict (a short leg is not comparable)
    verdict: object | None        # eval.koth.RoundResult when full_budget
    corpus_digest: str
    train_seconds: float
    init_label: str
    king_gpu: str
    scores: list = field(default_factory=list, repr=False)   # candidate WindowScores


def load_receipt_spec(spec: str, cfg, *, validator_hotkey: str = ""):
    """``spec`` is a local receipt JSON path, a round id, or ``latest``."""
    from ..shared.receipt import load_receipt

    p = Path(spec)
    if p.is_file():
        return load_receipt(p.read_text(encoding="utf-8"))
    from ..audit.main import fetch_receipt_text

    round_id = None if spec == "latest" else spec
    return load_receipt(fetch_receipt_text(cfg, round_id, validator_hotkey))


def find_snapshot_dir(root: Path | str, pool_sha256: str) -> Path:
    """The revealed snapshot folder whose ``POOL_SHA256`` is ``pool_sha256``.

    ``root`` may be that folder itself, a ``snapshots/`` dir of revealed
    folders, or a dataset checkout containing ``snapshots/``."""
    want = _norm_digest(pool_sha256)
    if not want:
        raise ReplayError("the round pins no eval-pool sha256; cannot locate its snapshot")
    root = Path(root)
    candidates = [root, *sorted(root.glob("*")), *sorted(root.glob("snapshots/*"))]
    for d in candidates:
        marker = d / SHA_MARKER
        if marker.is_file() and _norm_digest(marker.read_text(encoding="utf-8")) == want:
            return d
    raise ReplayError(
        f"no revealed snapshot with POOL_SHA256={want[:16]}… under {root} "
        "(reveals lag ~48h; fetch the newest folders of the eval-pool dataset)")


def _primary_king(cfg, receipt, manifest):
    """``(size, king entry, king EntryScores)`` for the round's primary size."""
    kings = {e.size: e for e in manifest.entries_for_role("king")}
    primary = cfg.training.primary_size.arch_preset
    size = next((s for s in (primary, "") if s in kings), None)
    if size is None:
        if len(kings) != 1:
            raise ReplayError(f"cannot pick the primary king size from {sorted(kings)}")
        size = next(iter(kings))
    entry = kings[size]
    rec = next((r for r in receipt.entry_scores
                if r.role == "king" and r.size == size and r.hotkey == entry.miner_hotkey), None)
    if rec is None:
        raise ReplayError(f"receipt carries no king entry_scores for size {size!r}")
    return size, entry, rec


def receipt_king_scores(cfg, receipt_text: str) -> list:
    """The king's signed per-window scores (primary size) from a receipt."""
    from ..shared.receipt import load_receipt

    receipt = load_receipt(receipt_text)
    _, _, rec = _primary_king(cfg, receipt, receipt.load_embedded_manifest())
    return [w.to_score() for w in rec.scores]


def verdict_windows(cfg, snapshot_dir: Path, base_seed: int, block: int) -> list:
    """The round's verdict windows, drawn exactly as the validator draws them
    (``ValidatorLoop._verdict_windows``): the scored horizon ladder when active
    at ``block``, else the single-horizon rotating draw."""
    from ..validator.pool import window_source_from_dir
    from ..validator.windows import ladder_windows_for_round, scored_ladder

    src = window_source_from_dir(Path(snapshot_dir), cfg, label=f"replay={snapshot_dir}")
    horizons = scored_ladder(cfg.eval, block)
    if not horizons:
        return src.windows_for_round(base_seed, cfg.eval.n_windows, block=block)
    return ladder_windows_for_round(
        src, horizons=horizons, n_windows=cfg.eval.n_windows,
        context_length=cfg.eval.context_length, round_seed=base_seed, block=block,
    )


def round_contract(cfg, manifest, contract_block: int | None):
    """The training contract the round ACTUALLY ran under.

    The manifest embeds its signed ``contract_body``; the operator's deployed
    config can differ from this checkout's chain.toml (observed: live rounds on
    ``points+mv20`` billing and a newer worker image while the repo still read
    ``series_points``). Scalar fields of the body override the local contract
    at the round's block. Returns ``(contract, digest_matches, overrides)``."""
    from dataclasses import fields as dc_fields

    from ..shared.manifest import contract_digest

    local = cfg.training.at_block(contract_block)
    body = manifest.contract_body if isinstance(manifest.contract_body, dict) else {}
    names = {f.name for f in dc_fields(local)}
    over = {k: v for k, v in body.items()
            if k in names and isinstance(v, (str, int, float, bool))
            and getattr(local, k) != v}
    contract = replace(local, **over) if over else local
    match = bool(manifest.contract_digest) and contract_digest(contract) == manifest.contract_digest
    if over:
        log.warning("round contract differs from local chain.toml: %s",
                    {k: f"{getattr(local, k)!r} -> {v!r}" for k, v in over.items()})
    return contract, match, over


def load_replay_round(cfg, receipt, snapshot_root: Path | str) -> ReplayRound:
    """Resolve a scored receipt + revealed snapshots into a :class:`ReplayRound`.

    Raises :class:`ReplayError` when the round is not replayable: not scored, no
    king scores, snapshot missing, or rebuilt windows that do not match the
    receipt's ``window_ids``."""
    from ..audit.checks import _baseline_pooled
    from ..trainer.contract import RoundSeeds

    if receipt.status != "scored" or receipt.eval_context is None:
        raise ReplayError(f"round {receipt.round_id} was not scored ({receipt.status})")
    manifest = receipt.load_embedded_manifest()
    size, king_entry, rec = _primary_king(cfg, receipt, manifest)
    king_scores = [w.to_score() for w in rec.scores]

    pool_sha = manifest.eval_pool_sha256 or receipt.eval_context.pool_digest
    snap = find_snapshot_dir(snapshot_root, pool_sha)
    block = int(receipt.epoch_start_block)
    windows = verdict_windows(cfg, snap, int(receipt.base_seed), block)
    got = tuple(w.series_id for w in windows)
    if got != tuple(receipt.eval_context.window_ids):
        raise ReplayError(
            f"rebuilt {len(got)} windows do not reproduce the receipt's "
            f"{len(receipt.eval_context.window_ids)} window_ids — the local chain.toml "
            "draw rules differ from the round's (update the image / checkout)")

    seeds = RoundSeeds(base_seed=int(receipt.base_seed),
                       generation_seed=int(receipt.generation_seed),
                       training_seed=int(receipt.training_seed))
    contract_block = int(receipt.era_start_block) or None
    contract, match, over = round_contract(cfg, manifest, contract_block)
    if isinstance(manifest.contract_body, dict) and not match:
        # Never train under a contract the round did not run: a body field this
        # checkout cannot express would silently change the leg.
        raise ReplayError(f"round {receipt.round_id}: the rebuilt contract does not hash to "
                          "the manifest's contract_digest (checkout too old for this round?)")
    return ReplayRound(
        receipt=receipt, snapshot_dir=snap, windows=windows, seeds=seeds,
        init_pointer=manifest.warm_start_ckpt or "",
        contract_block=contract_block,
        size=size, king_scores=king_scores,
        baseline_scores=_baseline_pooled(receipt, [size]),
        king_gpu=king_entry.gpu_name,
        contract=contract, contract_match=match, contract_overrides=over,
    )


def judge(rr: ReplayRound, chal_scores: list, cfg):
    """Judge ``chal_scores`` against the round's king under the round's recorded
    params, as ``cascade-audit`` replays a single-challenger verdict: a round
    without baseline rows falls back to LEVEL through the same
    ``cfg.judged_level_params`` the validator uses (never a v2 increment bar
    read as a level bar)."""
    from ..audit.checks import _bootstrap_seed
    from ..eval.koth import KothParams, evaluate_round

    v = rr.receipt.verdict
    if v is None:
        raise ReplayError("receipt has no verdict to take the round's params from")
    params = KothParams(**v.params)
    increment = params.margin_mode == "increment" and rr.baseline_scores is not None
    gate_on = str(getattr(params, "init_gate_mode", "off") or "off") != "off"
    block = int(rr.receipt.epoch_start_block)
    params = (replace(params, margin_mode="increment") if increment
              else cfg.judged_level_params(params, block))
    return evaluate_round(
        rr.king_scores, chal_scores, params,
        seed=_bootstrap_seed(v.bootstrap_seed),
        king_tenure_rounds=v.king_tenure_rounds,
        baseline_scores=(rr.baseline_scores
                         if increment or (gate_on and rr.baseline_scores is not None)
                         else None),
    )


def score_replay(
    repo_dir: Path | str,
    cfg,
    rr: ReplayRound,
    *,
    train_hours: float | None = None,
    device: str = "cpu",
    cache_dir: Path | str = "./_score_work",
    trainer_spec: str = "cascade.trainer.toto2_trainer:Toto2Trainer",
    seed_salt: int = 0,
    keep_checkpoint: Path | None = None,
    use_sandbox: bool = False,
) -> ReplayResult:
    """Train ``repo_dir`` as a challenger of the replayed round and score it.

    ``train_hours=None`` trains the round's FULL contract and judges the result
    against the king's receipt scores. A shorter budget still scores on the
    round's exact windows but returns no verdict: a short leg against a
    full-budget king measures the budget, not the data.

    ``seed_salt != 0`` trains under salted seeds instead of the round's (the
    windows stay the round's) — the gauntlet's noise calibration. A salted leg
    is never judged against the receipt: the king trained under the round's
    seeds. ``keep_checkpoint`` keeps the trained checkpoint at that path."""
    from ..eval.scoring import global_geomean
    from .score import _resolve_warm_start

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    contract = (rr.contract if rr.contract is not None
                else cfg.training.at_block(rr.contract_block)).primary_size
    full = train_hours is None
    seeds = salted_seeds(rr.seeds, seed_salt, cfg)
    if not full:
        contract = contract.for_hours(
            train_hours,
            guard_factor=cfg.round.heat_guard_factor,
            guard_floor_seconds=cfg.round.heat_guard_floor_seconds,
        )
    ws_dir, init_label = _resolve_warm_start(cfg, rr.init_pointer or None, cache_dir=cache)
    run = train_and_evaluate(
        Path(repo_dir), cfg, contract=contract, token_budget=contract.train_tokens,
        seeds=seeds, windows=rr.windows, warm_start_dir=ws_dir, init_label=init_label,
        device=device, cache=cache, trainer_spec=trainer_spec,
        hours_label="full contract" if full else f"{train_hours:.3g}h",
        keep_dir=keep_checkpoint, use_sandbox=use_sandbox,
    )
    if len(run.scores) != len(rr.king_scores):
        raise ReplayError(f"candidate produced {len(run.scores)} scores vs the king's "
                          f"{len(rr.king_scores)}; cannot pair")
    return ReplayResult(
        round_id=str(rr.receipt.round_id),
        geomean=global_geomean(run.scores),
        king_geomean=global_geomean(rr.king_scores),
        n_windows=len(run.scores),
        full_budget=full,
        verdict=judge(rr, run.scores, cfg) if full and not seed_salt else None,
        corpus_digest=run.corpus_digest,
        train_seconds=run.train_seconds,
        init_label=init_label,
        king_gpu=rr.king_gpu,
        scores=run.scores,
    )


def salted_seeds(seeds, salt: int, cfg):
    """The round's seeds (``salt == 0``) or a deterministic salted pair."""
    if not salt:
        return seeds
    from ..trainer.contract import RoundSeeds

    mixed = int.from_bytes(hashlib.blake2b(
        f"{seeds.base_seed}:{int(salt)}".encode(), digest_size=8).digest(), "big") >> 1
    return RoundSeeds.derive(mixed, cfg.training)
