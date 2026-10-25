"""MarketCast recipe (DEC-CA-0064) — the miner's declarative fine-tune recipe
for the pinned Timestral base, validated against ``[marketcast]`` bounds.

A MarketCast submission is (a) this recipe (``marketcast.json`` at the root
of the committed tree) PLUS (b) an OPTIONAL miner-designed generator tree
(the lane-A / lane-C ``generate()`` contract, run sandboxed) whose synthetic
series are mixed with the real market data at ``synthetic_fraction``
(``<= [marketcast] max_synthetic_fraction``; > 0 needs the generator).

Pure and deterministic, mirroring :func:`cascade.shared.lane_c.parse_recipe`
(unknown keys refused at every level, every bound checked). The lane-C
``recipe.json`` keys are untouched — this is a separate file and parser.

Recipe keys (all required at the top level):

* ``target``            ``mkt:<suite>`` — the lane-C slot the entry competes for;
* ``data``              ``assets`` {class: [symbols ⊆ config]}, ``class_weights``
                        {class: w ≥ 0, sum > 0}, ``frequencies`` (⊆ the config's),
                        ``start`` / ``end`` ISO dates — ``end`` at least
                        ``embargo_days`` before the submission (forward-only eval);
* ``context_length``    ``[patch, max_context_length]`` (≤ 32768);
* ``horizon_mix``       {horizon: w ≥ 0, sum > 0}, horizons of the target suite;
* ``covariates``        ⊆ ``[marketcast] covariates`` (inputs, never targets);
* ``augmentations``     [{``name`` ∈ allow-list, ``p`` ∈ [0, 1], ``params``
                        {str: finite number}}], at most ``MAX_AUGMENTATIONS``;
* ``objective``         ``suite_weights`` {suite: w ≥ 0; target's > 0},
                        optional ``quantile_weights`` (one ≥ 0 per base knot,
                        sum > 0), optional ``coverage_penalty`` ∈ [0, 10];
* ``optimizer``         ``lr`` (0, max_lr], ``schedule``, ``warmup_fraction``
                        [0, 0.2], ``steps`` [1, max_steps], ``batch``
                        [1, max_batch], ``weight_decay`` [0, 1], ``clip`` (0, 10],
                        optional ``ema_decay`` 0 or [0.9, 1);
* ``trainable``         ``scope`` full | last_n_blocks (+ ``n_blocks`` 1..12) |
                        frozen_embeddings | lora (+ ``lora_rank`` 1..max_lora_rank);
* ``synthetic_fraction`` [0, max_synthetic_fraction].
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .config.lane_c import RECIPE_SCHEDULES, check_target
from .config.marketcast import TIMESTRAL_BLOCKS, TIMESTRAL_PATCH, TRAINABLE_SCOPES

MARKETCAST_RECIPE_FILE = "marketcast.json"
TOP_KEYS = ("target", "data", "context_length", "horizon_mix", "covariates", "augmentations",
            "objective", "optimizer", "trainable", "synthetic_fraction")
DATA_KEYS = ("assets", "class_weights", "frequencies", "start", "end")
OBJECTIVE_KEYS = ("suite_weights", "quantile_weights", "coverage_penalty")
OPTIMIZER_KEYS = ("lr", "schedule", "warmup_fraction", "steps", "batch", "weight_decay", "clip",
                  "ema_decay")
OPTIMIZER_REQUIRED = OPTIMIZER_KEYS[:-1]
AUG_KEYS = ("name", "p", "params")
TRAINABLE_KEYS = ("scope", "n_blocks", "lora_rank")
MAX_AUGMENTATIONS = 8
MAX_WARMUP_FRACTION = 0.2
MAX_COVERAGE_PENALTY = 10.0
MAX_CLIP = 10.0


@dataclass(frozen=True)
class MarketCastRecipe:
    target: str
    suite: str
    assets: tuple[tuple[str, tuple[str, ...]], ...]
    class_weights: tuple[tuple[str, float], ...]
    frequencies: tuple[str, ...]
    start: str
    end: str
    context_length: int
    horizon_mix: tuple[tuple[str, float], ...]
    covariates: tuple[str, ...]
    augmentations: tuple[tuple[str, float, tuple[tuple[str, float], ...]], ...]
    suite_weights: tuple[tuple[str, float], ...]
    quantile_weights: tuple[float, ...]
    coverage_penalty: float
    lr: float
    schedule: str
    warmup_fraction: float
    steps: int
    batch: int
    weight_decay: float
    clip: float
    ema_decay: float
    scope: str
    n_blocks: int
    lora_rank: int
    synthetic_fraction: float


def _obj(v: Any, where: str, keys: tuple[str, ...], required: tuple[str, ...]) -> dict:
    if not isinstance(v, dict):
        raise ValueError(f"marketcast recipe {where} must be an object")
    extra = sorted(set(v) - set(keys))
    if extra:
        raise ValueError(f"marketcast recipe {where} has unknown keys {extra}; "
                         f"allowed {list(keys)}")
    missing = [k for k in required if k not in v]
    if missing:
        raise ValueError(f"marketcast recipe {where} is missing {missing}")
    return v


def _f(v: Any, where: str, lo: float, hi: float, *, lo_open: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"marketcast recipe {where} must be a finite number; got {v!r}")
    x = float(v)
    if (x <= lo if lo_open else x < lo) or x > hi:
        raise ValueError(f"marketcast recipe {where} {x} must be in "
                         f"{'(' if lo_open else '['}{lo}, {hi}]")
    return x


def _i(v: Any, where: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise ValueError(f"marketcast recipe {where} must be an integer in [{lo}, {hi}]; "
                         f"got {v!r}")
    return int(v)


def _labels(v: Any, where: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(v, str) or not isinstance(v, (list, tuple)) or not all(
            isinstance(x, str) for x in v):
        raise ValueError(f"marketcast recipe {where} must be a list of strings")
    if len(set(v)) != len(v):
        raise ValueError(f"marketcast recipe {where} must be unique")
    bad = [x for x in v if x not in allowed]
    if bad:
        raise ValueError(f"marketcast recipe {where} {bad} not allowed; allowed {list(allowed)}")
    return tuple(v)


def _weights(v: Any, where: str, allowed: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(v, dict) or not v:
        raise ValueError(f"marketcast recipe {where} must be a non-empty object")
    bad = [k for k in v if k not in allowed]
    if bad:
        raise ValueError(f"marketcast recipe {where} keys {bad} not allowed; "
                         f"allowed {list(allowed)}")
    out = {k: _f(w, f"{where}.{k}", 0.0, 1e6) for k, w in v.items()}
    if sum(out.values()) <= 0.0:
        raise ValueError(f"marketcast recipe {where} weights must sum > 0")
    return out


def _iso(v: Any, where: str) -> date:
    if not isinstance(v, str):
        raise ValueError(f"marketcast recipe {where} must be an ISO date YYYY-MM-DD")
    try:
        return date.fromisoformat(v)
    except ValueError as e:
        raise ValueError(f"marketcast recipe {where} {v!r} is not an ISO date") from e


def parse_marketcast_recipe(obj: Any, cfg: Any, *, target: str | None = None,
                            submission_time: float | None = None,
                            has_generator: bool = False) -> MarketCastRecipe:
    """Validate a ``marketcast.json`` payload (dict or JSON text) against
    ``cfg`` (a :class:`~cascade.shared.config.marketcast.MarketCastConfig`).

    ``target`` (the committed target) must equal the recipe's when given.
    ``submission_time`` (unix seconds, the submission's block time) enforces
    the embargo: ``data.end`` must be at least ``embargo_days`` before it.
    ``has_generator`` says the submission ships a generator tree;
    ``synthetic_fraction > 0`` without one is refused. Raises ``ValueError``."""
    if isinstance(obj, (str, bytes, bytearray)):
        try:
            obj = json.loads(obj)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"marketcast recipe is not valid JSON: {e}") from e
    top = _obj(obj, "", TOP_KEYS, TOP_KEYS)
    suites = tuple(cfg.suites)
    spec = check_target(str(top["target"]), mkt_suites=suites)
    if not spec.startswith("mkt:"):
        raise ValueError(f"marketcast recipe target {spec!r} must be mkt:<suite>")
    suite = spec.split(":", 1)[1]
    if target is not None and check_target(str(target), mkt_suites=suites) != spec:
        raise ValueError(f"marketcast recipe target {spec!r} != committed target {target!r}")

    from .config.marketcast import duration_minutes

    data = _obj(top["data"], "data", DATA_KEYS, DATA_KEYS)
    cfg_assets = cfg.assets_map
    if not isinstance(data["assets"], dict) or not data["assets"]:
        raise ValueError("marketcast recipe data.assets must be a non-empty object")
    assets: dict[str, tuple[str, ...]] = {}
    for cls, syms in data["assets"].items():
        if cls not in cfg_assets:
            raise ValueError(f"marketcast recipe data.assets class {cls!r} is not configured")
        ss = _labels(syms, f"data.assets.{cls}", tuple(cfg_assets[cls]))
        if not ss:
            raise ValueError(f"marketcast recipe data.assets.{cls} is empty")
        assets[cls] = ss
    cw = _weights(data["class_weights"], "data.class_weights", tuple(assets))
    all_freqs = tuple(sorted({f for fs in cfg.frequencies_map.values() for f in fs},
                             key=duration_minutes))
    freqs = _labels(data["frequencies"], "data.frequencies", all_freqs)
    if not freqs:
        raise ValueError("marketcast recipe data.frequencies is empty")
    start, end = _iso(data["start"], "data.start"), _iso(data["end"], "data.end")
    if end <= start:
        raise ValueError(f"marketcast recipe data.end {end} must be after data.start {start}")
    if submission_time is not None:
        sub_day = datetime.fromtimestamp(float(submission_time), tz=UTC).date()
        limit = sub_day - timedelta(days=int(cfg.embargo_days))
        if end > limit:
            raise ValueError(f"marketcast recipe data.end {end} is inside the "
                             f"{cfg.embargo_days}-day embargo (must be <= {limit})")

    ctx = _i(top["context_length"], "context_length", TIMESTRAL_PATCH,
             int(cfg.max_context_length))
    hmix = _weights(top["horizon_mix"], "horizon_mix", tuple(cfg.horizons_map.get(suite, ())))
    covs = _labels(top["covariates"], "covariates", tuple(cfg.covariates))

    augs_raw = top["augmentations"]
    if not isinstance(augs_raw, list) or len(augs_raw) > MAX_AUGMENTATIONS:
        raise ValueError(f"marketcast recipe augmentations must be a list of at most "
                         f"{MAX_AUGMENTATIONS}")
    augs = []
    for j, a in enumerate(augs_raw):
        a = _obj(a, f"augmentations[{j}]", AUG_KEYS, ("name", "p"))
        name = a["name"]
        if name not in tuple(cfg.augmentations):
            raise ValueError(f"marketcast recipe augmentation {name!r} not allowed; "
                             f"allowed {list(cfg.augmentations)}")
        p = _f(a["p"], f"augmentations[{j}].p", 0.0, 1.0)
        params = a.get("params", {})
        if not isinstance(params, dict) or not all(isinstance(k, str) for k in params):
            raise ValueError(f"marketcast recipe augmentations[{j}].params must be an object")
        pp = tuple(sorted((k, _f(v, f"augmentations[{j}].params.{k}", -1e6, 1e6))
                          for k, v in params.items()))
        augs.append((str(name), p, pp))
    if len({a[0] for a in augs}) != len(augs):
        raise ValueError("marketcast recipe augmentations must name each transform once")

    ob = _obj(top["objective"], "objective", OBJECTIVE_KEYS, ("suite_weights",))
    sw = _weights(ob["suite_weights"], "objective.suite_weights", suites)
    if sw.get(suite, 0.0) <= 0.0:
        raise ValueError(f"marketcast recipe objective.suite_weights must weight the target "
                         f"suite {suite!r} > 0")
    knots = tuple(cfg.quantile_knots)
    qw_raw = ob.get("quantile_weights")
    if qw_raw is None:
        qw = tuple(1.0 for _ in knots)
    else:
        if not isinstance(qw_raw, list) or len(qw_raw) != len(knots):
            raise ValueError(f"marketcast recipe objective.quantile_weights must list one "
                             f"weight per base knot ({len(knots)})")
        qw = tuple(_f(w, f"objective.quantile_weights[{j}]", 0.0, 1e6)
                   for j, w in enumerate(qw_raw))
        if sum(qw) <= 0.0:
            raise ValueError("marketcast recipe objective.quantile_weights must sum > 0")
    cpen = _f(ob.get("coverage_penalty", 0.0), "objective.coverage_penalty", 0.0,
              MAX_COVERAGE_PENALTY)

    op = _obj(top["optimizer"], "optimizer", OPTIMIZER_KEYS, OPTIMIZER_REQUIRED)
    lr = _f(op["lr"], "optimizer.lr", 0.0, float(cfg.max_lr), lo_open=True)
    sched = op["schedule"]
    if sched not in RECIPE_SCHEDULES:
        raise ValueError(f"marketcast recipe optimizer.schedule {sched!r} must be one of "
                         f"{RECIPE_SCHEDULES}")
    wf = _f(op["warmup_fraction"], "optimizer.warmup_fraction", 0.0, MAX_WARMUP_FRACTION)
    steps = _i(op["steps"], "optimizer.steps", 1, int(cfg.max_steps))
    batch = _i(op["batch"], "optimizer.batch", 1, int(cfg.max_batch))
    wd = _f(op["weight_decay"], "optimizer.weight_decay", 0.0, 1.0)
    clip = _f(op["clip"], "optimizer.clip", 0.0, MAX_CLIP, lo_open=True)
    ema = _f(op.get("ema_decay", 0.0), "optimizer.ema_decay", 0.0, 1.0)
    if ema != 0.0 and not 0.9 <= ema < 1.0:
        raise ValueError(f"marketcast recipe optimizer.ema_decay {ema} must be 0 (off) or "
                         "in [0.9, 1)")

    tr = _obj(top["trainable"], "trainable", TRAINABLE_KEYS, ("scope",))
    scope = tr["scope"]
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"marketcast recipe trainable.scope {scope!r} must be one of "
                         f"{TRAINABLE_SCOPES}")
    n_blocks = lora_rank = 0
    if scope == "last_n_blocks":
        n_blocks = _i(tr.get("n_blocks"), "trainable.n_blocks", 1, TIMESTRAL_BLOCKS)
    elif "n_blocks" in tr:
        raise ValueError("marketcast recipe trainable.n_blocks is only for last_n_blocks")
    if scope == "lora":
        lora_rank = _i(tr.get("lora_rank"), "trainable.lora_rank", 1, int(cfg.max_lora_rank))
    elif "lora_rank" in tr:
        raise ValueError("marketcast recipe trainable.lora_rank is only for lora")

    syn = _f(top["synthetic_fraction"], "synthetic_fraction", 0.0,
             float(cfg.max_synthetic_fraction))
    if syn > 0.0 and not has_generator:
        raise ValueError("marketcast recipe synthetic_fraction > 0 needs a generator tree")

    return MarketCastRecipe(
        target=spec, suite=suite, assets=tuple(sorted(assets.items())),
        class_weights=tuple(sorted(cw.items())), frequencies=freqs,
        start=start.isoformat(), end=end.isoformat(), context_length=ctx,
        horizon_mix=tuple(sorted(hmix.items())), covariates=covs, augmentations=tuple(augs),
        suite_weights=tuple(sorted(sw.items())), quantile_weights=qw, coverage_penalty=cpen,
        lr=lr, schedule=str(sched), warmup_fraction=wf, steps=steps, batch=batch,
        weight_decay=wd, clip=clip, ema_decay=ema, scope=str(scope), n_blocks=n_blocks,
        lora_rank=lora_rank, synthetic_fraction=syn,
    )


# ── the fine-tune's marketcast_meta.json (harvest + validator, one rule) ────

META_MAX_BYTES = 64 * 1024
META_REQUIRED = ("base_ref", "base_sha256", "target", "covariates", "data_digest")


def meta_problem(meta: Any, cfg: Any, *, size: int | None = None) -> str:
    """Why a fine-tune's ``marketcast_meta.json`` (parsed) is unusable — ""
    when fine. Checked at harvest (``cascade.trainer.marketcast_runner``) AND
    again by the validator inside its fault path, so a malformed meta is the
    miner's deterministic ``fault``, never a hold: the required fields, the
    pinned base, an enabled ``mkt:`` target, covariate names unique and ⊆ BOTH
    ``[marketcast] covariates`` and the builders
    (:data:`cascade.eval.marketcast.PAST_COVARIATES` / ``KNOWN_COVARIATES``),
    and the size (≤ :data:`META_MAX_BYTES`)."""
    if size is not None and int(size) > META_MAX_BYTES:
        return f"marketcast_meta.json is {size} bytes > {META_MAX_BYTES}"
    if not isinstance(meta, dict):
        return "marketcast_meta.json is not a JSON object"
    missing = [k for k in META_REQUIRED if k not in meta]
    if missing:
        return f"marketcast_meta.json lacks {missing}"
    if (str(meta["base_ref"]) != cfg.base_ref
            or str(meta["base_sha256"]).lower() != cfg.base_sha256):
        return "marketcast_meta.json names a base other than the pinned [marketcast] base"
    t = str(meta["target"])
    if not t.startswith("mkt:") or t.split(":", 1)[1] not in tuple(cfg.suites):
        return f"marketcast_meta.json target {t!r} is not an enabled MarketCast suite"
    return covariates_problem(meta["covariates"], cfg)


def covariates_problem(names: Any, cfg: Any) -> str:
    """Why a fine-tune's covariate list is unusable ("" when fine)."""
    from ..eval.marketcast import KNOWN_COVARIATES, PAST_COVARIATES

    if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)) or not all(
            isinstance(x, str) for x in names):
        return f"covariates must be a list of names; got {names!r}"
    if len(set(names)) != len(names):
        return f"covariates {list(names)} repeat a name"
    known = set(PAST_COVARIATES) | set(KNOWN_COVARIATES)
    bad = sorted(set(names) - (set(cfg.covariates) & known))
    if bad:
        return f"covariates {bad} are not allowed (allowed {sorted(set(cfg.covariates) & known)})"
    return ""


__all__ = ["MARKETCAST_RECIPE_FILE", "META_MAX_BYTES", "MarketCastRecipe", "covariates_problem",
           "meta_problem", "parse_marketcast_recipe"]
