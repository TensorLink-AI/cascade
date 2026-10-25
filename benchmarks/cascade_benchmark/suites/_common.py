"""Shared gluonts scoring for the GIFT-Eval and BOOM runners.

Both expose datasets through gift-eval's ``Dataset`` class, so scoring one
config is identical and matches gift-eval's reference runner
(``notebooks/naive.ipynb``): the same ``evaluate_model`` call with seasonality
from the dataset frequency. We return the per-config MASE / MAE / CRPS; the
cross-dataset aggregation (Seasonal-Naive normalized shifted-geomean) lives in
``cascade_benchmark.aggregate`` so it matches the official leaderboards.
"""

from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field

# 9-level grid — matches GIFT-Eval and cascade's DEFAULT_QUANTILE_LEVELS.
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def describe_exception(exc: BaseException, limit: int = 300) -> str:
    """``"<ExcType>: <message>"`` clipped to ``limit`` chars — the skip reason
    a report carries for a config that produced no row (never a traceback:
    reports are published, and a traceback can quote data paths)."""
    msg = str(exc).strip().splitlines()
    text = f"{type(exc).__name__}: {msg[0] if msg else ''}".strip()
    return text[:limit]


def build_dataset_or_reason(name: str, term: str, *, storage_env_var: str = "GIFT_EVAL"):
    """Construct a gift-eval ``Dataset`` following the reference runner's
    ``to_univariate`` rule (multivariate series split into univariate, matching
    the leaderboard). Returns ``(dataset, "")`` or ``(None, reason)`` — the
    reason is what the sweep records for the config it could not load, so a
    partial sweep can never masquerade as the whole benchmark.
    """
    from gift_eval.data import Dataset

    try:
        probe = Dataset(name=name, term=term, to_univariate=False, storage_env_var=storage_env_var)
        to_univariate = probe.target_dim != 1
        ds = Dataset(
            name=name, term=term, to_univariate=to_univariate, storage_env_var=storage_env_var
        )
        return ds, ""
    except Exception as e:  # noqa: BLE001 — unknown/invalid (name, term) combo → skip
        return None, describe_exception(e)


def score_dataset(
    ds, checkpoint_dir: str, *, num_samples: int, device: str, batch_size: int = 64
) -> dict:
    """Score the checkpoint on one ``Dataset`` and return ``{MASE, MAE, CRPS}``.

    The ``evaluate_model`` call mirrors gift-eval's reference runner exactly
    (kwargs + seasonality), so the per-config numbers are leaderboard-faithful.
    Column names ``MASE[0.5]`` / ``MAE[0.5]`` / ``mean_weighted_sum_quantile_loss``
    are verified against ``naive.ipynb``. ``batch_size`` is the number of series
    the predictor forwards through the checkpoint at once (quantile-head path).
    """
    from gluonts.ev.metrics import MAE, MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    from ..predictor import CheckpointPredictor

    season_length = get_seasonality(ds.freq)
    predictor = CheckpointPredictor(
        checkpoint_dir,
        prediction_length=ds.prediction_length,
        num_samples=num_samples,
        device=device,
        batch_size=batch_size,
    )
    res = evaluate_model(
        predictor,
        test_data=ds.test_data,
        metrics=[MASE(), MAE(), MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILE_LEVELS))],
        batch_size=512,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    return {
        "MASE": float(res["MASE[0.5]"][0]),
        "MAE": float(res["MAE[0.5]"][0]),
        "CRPS": float(res["mean_weighted_sum_quantile_loss"][0]),
    }


# ── work units (the sweep scheduler's contract, ``cascade_benchmark.parallel``) ──
#
# Every suite is split into independent UNITS — one GIFT-Eval config, one BOOM
# dataset, one TIME task, one FEV task — and exposes three functions:
#
#   plan(*, max_series)            -> SuitePlan | SuiteResult  (parent, checkpoint-free)
#   run_unit(ctx, unit, ckpts, **) -> [outcome per checkpoint]  (any process)
#   assemble(plan, outcomes)       -> SuiteResult              (parent, canonical order)
#
# An outcome is a small picklable dict holding ``"row"`` (scored), ``"skip"``
# (the skip record a sequential sweep writes) or ``"abort"`` (a failure that
# ended the old per-suite loop — ``{"status", "detail"}``), plus suite extras.
# ``run_unit`` never raises. ``assemble`` walks the outcomes in UNIT order and
# stops at the first abort, exactly where the sequential loop stopped — so the
# report is byte-identical whether the units ran in one process, in N worker
# processes in any completion order, or once for several checkpoints together.


@dataclass
class SuitePlan:
    """One suite's work list. ``units`` are picklable payloads in canonical
    order; ``ctx`` is shipped to every worker; ``local`` stays in the parent
    (assembly-only state such as the vendored baselines)."""

    suite: str
    units: list = field(default_factory=list)
    ctx: dict = field(default_factory=dict)
    local: dict = field(default_factory=dict)


def abort(status: str, detail: str) -> dict:
    return {"abort": {"status": status, "detail": detail}}


def abort_from_exc(exc: BaseException, *, import_detail: str | None = None) -> dict:
    """The outcome of the old suite-level ``except`` clauses: an ImportError is
    ``skipped`` (``import_detail`` + the error), anything else ``error`` with a
    3-frame traceback. Call from inside the ``except`` block."""
    if import_detail is not None and isinstance(exc, ImportError):
        return abort("skipped", f"{import_detail}: {exc}")
    return abort("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}")


def until_abort(outcomes: list[dict]) -> tuple[list[dict], dict | None]:
    """``(outcomes before the first abort, that abort or None)``."""
    for i, o in enumerate(outcomes):
        if "abort" in o:
            return outcomes[:i], o["abort"]
    return list(outcomes), None


def run_plan(module, plan, checkpoints: list[str], *, num_samples: int, device: str,
             batch_size: int) -> list:
    """Run an already-built plan's units in THIS process, in order, for every
    checkpoint — a checkpoint whose unit aborts gets no further units (the old
    loop's early exit). Returns one ``SuiteResult`` per checkpoint."""
    if not isinstance(plan, SuitePlan):
        return [copy.deepcopy(plan) for _ in checkpoints]
    outs: list[list[dict]] = [[] for _ in checkpoints]
    live = list(range(len(checkpoints)))
    for unit in plan.units:
        if not live:
            break
        got = module.run_unit(plan.ctx, unit, [checkpoints[i] for i in live],
                              num_samples=num_samples, device=device, batch_size=batch_size)
        for i, o in zip(live, got, strict=True):
            outs[i].append(o)
        live = [i for i in live if "abort" not in outs[i][-1]]
    return [module.assemble(plan, o) for o in outs]


def run_sequential(module, checkpoints: list[str], *, num_samples: int,
                   max_series: int | None, device: str, batch_size: int) -> list:
    """``plan`` + :func:`run_plan` — one suite, in this process."""
    return run_plan(module, module.plan(max_series=max_series), checkpoints,
                    num_samples=num_samples, device=device, batch_size=batch_size)
