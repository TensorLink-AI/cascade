"""Local generator scoring — the fast miner iteration loop.

`cascade score` closes the loop the on-chain round can't: train the fixed model
on your generator's data at a cheap local budget (``[round] heat_train_hours``:
minutes, not the ~3h leg) and score it on a pool you control, entirely offline — no chain, no TAO,
no ~12h round. It reuses the exact pieces the trainer/validator use
(``open_round_stream`` → the reference ``BaseTrainer`` → ``evaluate_checkpoint``
→ ``global_geomean``), so the number tracks the metric the duel judges.

The score is **directional, not the verdict**: you score on a public/sample pool,
while the validator scores on its private rotating pool. By default this trains
from random init; live rounds train from the promoted cascade warm-start once a
generation is live, so pass ``--warm-start live`` (the current era init from
fresh trainer status, with its matching effective training contract) or an explicit ``repo@digest`` /
local checkpoint dir to train the way the live round does — same starting
weights AND the warm-started recipe (``warm_lr_scale``, wsd warmup-once). Use it
to hill-climb locally, then let the real eval rank you. Pair it with
``cascade fetch king`` to score the reigning king the same way and compare.

Needs the ``[train]`` extra (torch) and, ideally, a GPU — the cheap budget keeps
it to minutes.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("cascade.miner.score")


@dataclass(frozen=True)
class ScoreResult:
    geomean: float          # geomean(CRPS/MWSQL, MASE) — LOWER is better
    n_windows: int
    corpus_digest: str
    n_series: int
    train_seconds: float
    pool_label: str
    init_label: str = "random init"
    contract_digest: str = ""
    train_tokens: int = 0


def _load_pool_windows(cfg, *, pool_dir, pool_ref, n_windows, seed, cache_dir):
    """Resolve the scoring pool to a list of EvalWindow.

    Precedence: an explicit local ``pool_dir`` (your own held-out .npy/.npz), then
    a Hub ``pool_ref`` (a pinned public pool), else an offline **synthetic sample**
    (deterministic, clearly directional — swap in real data for a meaningful
    signal). Returns ``(windows, label)``.
    """
    from ..validator.pool import window_source_from_dir
    from ..validator.windows import build_windows_from_series

    if pool_dir is not None:
        src = window_source_from_dir(Path(pool_dir), cfg, label=f"dir={pool_dir}")
        return src.windows_for_round(seed, n_windows), f"dir:{pool_dir}"

    if pool_ref:
        from ..shared.hippius import HubConfig, HubRef, fetch_from_hub

        dest = Path(cache_dir) / "score-pool" / HubRef.parse(pool_ref).digest.replace(":", "-")
        fetch_from_hub(pool_ref, dest, HubConfig.from_storage(cfg.storage))
        src = window_source_from_dir(dest, cfg, label=f"ref={pool_ref}")
        return src.windows_for_round(seed, n_windows), f"ref:{pool_ref}"

    # Offline synthetic sample — no network, deterministic in as_of.
    import datetime as dt

    from ..pool.source import HarvestContext
    from ..pool.sources.synthetic import SyntheticSource

    ctx = HarvestContext(
        as_of=dt.date(2026, 1, 1),
        context_length=cfg.eval.context_length,
        horizon=cfg.eval.horizon,
        max_series=max(n_windows, 128),
    )
    series, ids = [], []
    for hs in SyntheticSource(n_series=max(n_windows, 128)).harvest(lambda *_a, **_k: None, ctx):
        series.append(hs.values)
        ids.append(hs.series_id)
    windows = build_windows_from_series(
        series, context_length=cfg.eval.context_length, horizon=cfg.eval.horizon, id_prefix=""
    )
    return windows[:n_windows], "synthetic-sample (directional only — use --pool-dir for real data)"


LIVE_WARM_START = "live"


def resolve_live_warm_start(cfg) -> str | None:
    """Require the same current contract/init pair as every live scoring entry point."""
    from .training_context import live_context

    return live_context(cfg)[1]


def _resolve_warm_start(cfg, warm_start, *, cache_dir) -> tuple[Path | None, str]:
    """Resolve ``warm_start`` to a local checkpoint dir for the trainer.

    ``None``/empty ⇒ random init. ``"live"`` ⇒ the current era init from
    fresh trainer/chain status with its effective contract; unavailable evidence
    never silently means random init. A ``metro-v1:trained:hippius:…`` pointer or
    a bare ``repo@digest`` is fetched from the Hub into ``cache_dir``; anything
    else must be an existing local directory holding the checkpoint weights.
    Returns ``(dir_or_None, label)``.
    """
    if warm_start is None or str(warm_start).strip() == "":
        return None, "random init"
    spec = str(warm_start).strip()

    if spec.lower() == LIVE_WARM_START:
        init = resolve_live_warm_start(cfg)
        if not init:
            return None, "random init (live init has no checkpoint)"
        spec = str(init)

    from ..shared.hippius import HubConfig, HubRef, fetch_from_hub, is_hub_ref
    from ..shared.manifest import parse_trained_pointer

    ref = parse_trained_pointer(spec) or (spec if is_hub_ref(spec) else None)
    if ref is not None:
        dest = Path(cache_dir) / "warm-start" / HubRef.parse(ref).digest.replace(":", "-")
        fetch_from_hub(ref, dest, HubConfig.from_storage(cfg.storage))
        return dest, f"warm-start {ref}"

    local = Path(spec)
    if local.is_dir():
        return local, f"warm-start dir:{local}"
    raise ValueError(
        f"--warm-start {spec!r} is neither 'live', a Hub ref (repo@digest / trained "
        "pointer), nor an existing checkpoint directory"
    )


def score_generator(
    repo_dir: Path | str,
    cfg,
    *,
    pool_dir: Path | str | None = None,
    pool_ref: str = "",
    train_hours: float | None = None,
    n_windows: int | None = None,
    device: str = "cpu",
    seed: int = 0,
    cache_dir: Path | str = "./_score_work",
    trainer_spec: str = "cascade.trainer.toto2_trainer:Toto2Trainer",
    warm_start: str | Path | None = None,
    at_block: int | None = None,
) -> ScoreResult:
    """Train the fixed model on ``repo_dir``'s corpus at a cheap local budget and
    score it on the resolved pool. Returns a :class:`ScoreResult` (lower geomean better).

    ``train_hours`` defaults to ``[round] heat_train_hours`` (the cheap local
    budget); ``n_windows`` to ``[round] heat_n_windows``. Reuses the round's
    contract at the primary size, scaled down to that budget.

    ``warm_start`` (see :func:`_resolve_warm_start`) replaces the seeded random
    init with a promoted cascade checkpoint — ``"live"`` for the current era init — and switches the trainer to the warm-started recipe, so
    the run mirrors a live warm-started round instead of a from-scratch one.
    """
    from ..eval.scoring import global_geomean
    from ..shared.manifest import contract_digest
    from ..trainer.contract import RoundSeeds
    from ..trainer.main import _load_trainer
    from ..validator.evaluator import evaluate_checkpoint
    from .local_corpus import local_round_stream
    from .training_context import configured_context, live_context

    if str(warm_start).lower() == LIVE_WARM_START:
        cfg, warm_start, _ = live_context(cfg)
    else:
        cfg = configured_context(cfg, at_block)

    repo = Path(repo_dir)
    hours = train_hours if train_hours is not None else cfg.round.heat_train_hours
    # for_hours scales the wall-clock cap down with the budget (same contract +
    # [round] guard knobs), so a bug that stalls your generator fails in
    # minutes locally, and a slow generator shows deadline truncation here
    # proportional to what it would suffer on a real leg.
    contract = cfg.screen_contract().for_hours(           # primary/screen size
        hours,
        guard_factor=cfg.round.heat_guard_factor,
        guard_floor_seconds=cfg.round.heat_guard_floor_seconds,
    )
    token_budget = contract.train_tokens
    n_win = n_windows if n_windows is not None else min(cfg.round.heat_n_windows, cfg.eval.n_windows)
    seeds = RoundSeeds.derive(seed, cfg.training)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    windows, pool_label = _load_pool_windows(
        cfg, pool_dir=pool_dir, pool_ref=pool_ref, n_windows=n_win, seed=seed, cache_dir=cache
    )
    if not windows:
        raise ValueError("scoring pool produced no windows (check --pool-dir / --pool contents)")

    ws_dir, init_label = _resolve_warm_start(cfg, warm_start, cache_dir=cache)
    train_kwargs = {"warm_start_dir": ws_dir} if ws_dir is not None else {}

    base_trainer = _load_trainer(trainer_spec)
    if hasattr(base_trainer, "device"):
        base_trainer.device = device
    with tempfile.TemporaryDirectory(dir=cache, prefix="ckpt-") as td:
        out_dir = Path(td)
        log.info("training on %s at %.3gh (%s point-passes) from %s …",
                 repo.name, hours, f"{token_budget:,}", init_label)
        with local_round_stream(
            contract.corpus_mode, repo, seeds.generation_seed, cfg.generator,
            generation=cfg.generation,
            token_budget=token_budget, use_sandbox=False,      # local, trusted-own-code path
            blocked=cfg.static_guard.blocked,
            seed_mix=int(getattr(contract, "gen_seed_mix", 1) or 1),
            budget_denomination=getattr(contract, "budget_denomination", "points"),
            corpus_epochs=int(getattr(contract, "corpus_epochs", 0) or 0),
            **({"continuation_protocol": contract.continuation_protocol} if getattr(contract, "continuation_protocol", "") else {}),
        ) as rs:
            result = base_trainer.train(
                rs.series(), contract,
                training_seed=seeds.training_seed, token_budget=token_budget, out_dir=out_dir,
                **train_kwargs,
            )
            corpus_digest, n_series = rs.digest, rs.n_series
        scores = evaluate_checkpoint(
            result.local_dir, windows, num_samples=cfg.eval.num_samples, device=device
        )

    return ScoreResult(
        geomean=global_geomean(scores),
        n_windows=len(scores),
        corpus_digest=corpus_digest,
        n_series=n_series,
        train_seconds=result.train_seconds,
        pool_label=pool_label,
        init_label=init_label,
        contract_digest=contract_digest(contract),
        train_tokens=token_budget,
    )
