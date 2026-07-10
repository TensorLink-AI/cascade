"""Local generator scoring — the fast miner iteration loop.

`cascade score` closes the loop the on-chain round can't: train the fixed model
on your generator's data at the cheap **heat** budget (minutes, not the ~3h
final) and score it on a pool you control, entirely offline — no chain, no TAO,
no ~30-minute wait. It reuses the exact pieces the trainer/validator use
(``open_round_stream`` → the reference ``BaseTrainer`` → ``evaluate_checkpoint``
→ ``global_geomean``), so the number tracks how the heat screener would rank you.

The score is **directional, not the verdict**: you score on a public/sample pool,
while the validator scores on its private rotating pool. Use it to hill-climb
locally, then let the real eval rank you. Pair it with ``cascade fetch king`` to
score the reigning king the same way and compare.

Needs the ``[train]`` extra (torch) and, ideally, a GPU — the heat budget keeps
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
) -> ScoreResult:
    """Train the fixed model on ``repo_dir``'s corpus at the heat budget and score
    it on the resolved pool. Returns a :class:`ScoreResult` (lower geomean better).

    ``train_hours`` defaults to ``[round] heat_train_hours`` (the cheap screen);
    ``n_windows`` to ``[round] heat_n_windows``. Reuses the round's contract at the
    screen size, so the number mirrors the heat screener's ranking.
    """
    from ..eval.scoring import global_geomean
    from ..trainer.contract import RoundSeeds
    from ..trainer.main import _load_trainer
    from ..trainer.stream import open_round_stream
    from ..validator.evaluator import evaluate_checkpoint

    repo = Path(repo_dir)
    contract = cfg.screen_contract()                      # primary/screen size
    hours = train_hours if train_hours is not None else cfg.round.heat_train_hours
    token_budget = contract.tokens_for_hours(hours)
    n_win = n_windows if n_windows is not None else min(cfg.round.heat_n_windows, cfg.eval.n_windows)
    seeds = RoundSeeds.derive(seed, cfg.training)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    windows, pool_label = _load_pool_windows(
        cfg, pool_dir=pool_dir, pool_ref=pool_ref, n_windows=n_win, seed=seed, cache_dir=cache
    )
    if not windows:
        raise ValueError("scoring pool produced no windows (check --pool-dir / --pool contents)")

    base_trainer = _load_trainer(trainer_spec)
    with tempfile.TemporaryDirectory(dir=cache, prefix="ckpt-") as td:
        out_dir = Path(td)
        log.info("training on %s at %.3gh (%s point-passes) …", repo.name, hours, f"{token_budget:,}")
        with open_round_stream(
            contract.corpus_mode, repo, seeds.generation_seed, cfg.generator,
            token_budget=token_budget, use_sandbox=False,      # local, trusted-own-code path
            blocked=cfg.static_guard.blocked,
        ) as rs:
            result = base_trainer.train(
                rs.series(), contract,
                training_seed=seeds.training_seed, token_budget=token_budget, out_dir=out_dir,
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
    )
