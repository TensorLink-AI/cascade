"""``[marketcast]`` — MarketCast, a lane-C fine-tuning eval on a pinned
Timestral-28M base (DEC-CA-0064). Built; shipped OFF (``from_block = 0``).

MarketCast judges fine-tunes of a pinned third-party base model
(``mistralai/Timestral-28M``, ``base_arch = "timestral"``) paired against the
untouched base on forward-only market windows, one lane-C target slot per
suite (``mkt:<suite>``). The pure scoring pieces live in
:mod:`cascade.eval.marketcast`; the base-model interface in
:mod:`cascade.eval.timestral_adapter`.

Why a sibling ``[marketcast]`` table and not ``[lane_c.marketcast]``:

* ``[lane_c]`` is on the DEC-CA-0057 overlay allow-list with EVERY key
  (``ALLOWED["lane_c"] = None``), so a nested sub-table there would be
  overlay-settable. MarketCast pins a new base architecture, its checkpoint
  digest and a data contract — release-only terms — so it lives in its own
  table. Only its listed numeric verdict, payout and liveness controls are
  overlay-settable; base pins, activation, universe and data contract remain
  release-only (``ALLOWED["marketcast"]``).
* :class:`~cascade.shared.config.lane_c.LaneCConfig` stays byte-identical
  (its fields and defaults are consensus terms replayed by the audit).

What MarketCast shares with lane C is the TARGET GRAMMAR (``mkt:<suite>`` is
a lane-C target, accepted only for the suites :attr:`MarketCastConfig.
target_suites` admits — none while inert) and the payout / TTL machinery.

Activation: ``from_block`` > 0 arms MarketCast at that era boundary. The
table-local arming checks run here (a pinned ``base_ref`` commit, a
``base_sha256``, and a non-empty data source with a built adapter for every
asset class that lists assets); the cross-table checks (era boundary,
``>= [v2] from_block``, ``>= [lane_c] from_block``, ``[v2] bench_source =
"validator"`` in force, ``[netconfig] owner_hotkey`` set) run in the loader
(:func:`check_marketcast_arming`).

Data sources (owner 2026-10-19): Hyperliquid is PRIMARY wherever it lists the
asset — main-dex perps for crypto, the ``xyz`` builder-deployed perps for
FX, metals, commodities, indices / ETFs and mega-caps (1m ``candleSnapshot``
+ hourly ``fundingHistory``, public, no key). Binance spot 1m klines are the
crypto CROSS-CHECK. dTAO alpha is read from the subnet pool state with the
pinned bittensor SDK. A ``data_sources`` value is ``"<primary>[+<cross-check>...]"``;
``""`` = no source chosen, which refuses arming for a class that lists assets.
The asset lists are the universe Hyperliquid listed on 2026-10-19
(``POST https://api.hyperliquid.xyz/info`` ``{"type":"meta"}`` and
``{"type":"meta","dex":"xyz"}``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

_log = logging.getLogger("cascade.config.marketcast")

# The suites (DEC-CA-0064 §Suites; ``ohlcv`` / ``high`` / ``low`` added
# 2026-11-10, owner). With ``composite = false`` a suite is its own lane-C
# target ``mkt:<suite>``; with ``composite = true`` the ONE target
# ``mkt:market`` (:data:`COMPOSITE_TARGET`) is judged over every enabled suite.
SUITES: tuple[str, ...] = ("returns", "vol", "range", "funding", "liquidity", "alpha",
                           "ohlcv", "high", "low")
# The code-level (absent-table) default: the original launch set. The
# SHIPPED chain.toml enables the 2026-11-10 composite launch set instead
# (:data:`COMPOSITE_LAUNCH_SUITES`).
DEFAULT_SUITES: tuple[str, ...] = ("returns", "vol")
# Owner 2026-11-10 ("a model good at forecasting markets"): one composite
# target over these suites; funding stays off (its next rate is published).
COMPOSITE_LAUNCH_SUITES: tuple[str, ...] = ("ohlcv", "returns", "vol", "high", "low", "range",
                                            "liquidity")
COMPOSITE_TARGET = "market"
# The composite's suite weights (owner 2026-11-18, "a balance that produces a
# model good at forecasting"): each of four properties of a market forecast
# counts once instead of once per suite that re-measures it — location
# (returns + ohlcv.c) 0.30, dispersion (vol + range) 0.30, extremes (high +
# low + ohlcv.h / ohlcv.l) 0.25, activity (liquidity + ohlcv.v) 0.15; ohlcv's
# 0.20 is its four scored fields at 0.05 each. Renormalised over the enabled
# suites; funding / alpha (off at launch) weigh like a single-property suite.
COMPOSITE_SUITE_WEIGHTS: dict[str, float] = {
    "returns": 0.25, "vol": 0.18, "range": 0.12, "high": 0.075, "low": 0.075,
    "liquidity": 0.10, "ohlcv": 0.20, "funding": 0.10, "alpha": 0.10,
}
# No-regression guards on the composite (owner 2026-11-18): CRPS stays the one
# score; each guard is a second proper loss on the same windows that the
# fine-tune may not be CONFIDENTLY worse than the base on (UCB >= -guard_floor), so a CRPS
# win bought by reshaping the bands at the cost of the point forecast, the
# variance forecast or the tails fails.
#   mase  — |median - y| / the window's CRPS scale (the seasonal-naive MAE:
#           exactly MASE), on the level-like suites (a returns median is ~0);
#   qlike — Patton's QLIKE of the quantile-implied mean variance on vol (the
#           loss that ranks vol forecasts right on a noisy realised proxy);
#   tail  — 2 x mean pinball at TAIL_SCORE_LEVELS / scale, the quantile
#           (VaR) score of the tails, on the calibration-gated suites.
GUARD_METRICS: tuple[str, ...] = ("mase", "qlike", "tail")
GUARD_SUITES: dict[str, tuple[str, ...]] = {
    "mase": ("ohlcv", "vol", "range", "high", "low", "liquidity"),
    "qlike": ("vol",),
    "tail": ("returns", "vol"),
}
TAIL_SCORE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.95, 0.99)
# ``ohlcv`` forecasts the actual next-bar levels of these fields, one series
# each (window suite label ``ohlcv.<field>``), scored with scale-normalised
# CRPS; ``o`` is carried with ``ohlcv_open_weight`` (default 0 = not scored:
# a Hyperliquid perp's open is the previous close).
OHLCV_FIELDS: tuple[str, ...] = ("o", "h", "l", "c", "v")
OHLCV_FIELD_NAMES: dict[str, str] = {"o": "open", "h": "high", "l": "low", "c": "close",
                                     "v": "volume"}
# Native Hyperliquid candle intervals a snapshot may carry as their own bank
# besides the 1-minute bars and the hourly (coarse) bank.
NATIVE_INTERVALS: tuple[str, ...] = ("5m", "15m")
# Calibration is a GATE (AND-ed into the verdict) on these suites only: the
# central-interval coverage gate and the tail hit-rate gate (red-team F1b,
# 2026-10-20 — vol was ungated, so collapsed log-RV quantiles passed).
COVERAGE_GATE_SUITES: tuple[str, ...] = ("returns", "vol")
ASSET_CLASSES: tuple[str, ...] = (
    "crypto", "fx", "metals", "commodities", "equity_etf", "alpha",
)
BASE_ARCHS: tuple[str, ...] = ("timestral",)

# Timestral-28M facts, read 2026-10-19 from the Hub at the pinned commit
# (https://huggingface.co/api/models/mistralai/Timestral-28M, ``sha``; the
# tree endpoint gives model.safetensors' LFS oid = its sha256). config.json:
# patch_size 32, d_model 384 (the README says "embedding dimension 256"),
# num_heads 6, mlp_hidden_dim 1024, block_schedule of 12 layers (9 "time" +
# 3 "time_variate"), block_size 32 patches (= 1024 steps per pass),
# context_length 32768, 21 quantile knots.
TIMESTRAL_REPO = "mistralai/Timestral-28M"
TIMESTRAL_REVISION = "180698bbf5a50dda3516aa6419798437fb03a430"
TIMESTRAL_SHA256 = "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d"
TIMESTRAL_PINNED_REF = f"{TIMESTRAL_REPO}@{TIMESTRAL_REVISION}"
TIMESTRAL_KNOTS: tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
)
TIMESTRAL_MAX_CONTEXT = 32768
# a recipe's leg_hours ceiling: a week (the cap a recipe may ask for is
# [marketcast] max_leg_hours; the verdict never waits on the leg's length)
MAX_LEG_HOURS_CEILING = 168.0
# Owner 2026-11-24: a MarketCast leg fine-tunes Timestral-28M on ONE GPU (the
# worker is one process on cuda:0), so a miner may rent ANY Lium GPU type.
# ``[marketcast] funded_pod_skus`` is the RECOMMENDED list (no warning; the
# default leg rents the cheapest fitting one); any other well-formed type is
# accepted at the miner's own risk (cascade.funding.mkt_gpus). Trainer and
# intake policy only: no validator, satellite or audit path reads it.
_SKU_RE = re.compile(r"^(?:[1-9]\d?x)?[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")
CRPS_SCALES = ("level", "seasonal_naive")
# Owner 2026-11-24 (DEC-CA-0074): a verdict within 13 hours of submission,
# judged on windows drawn from the embargo. Code bounds of its overlay knobs.
VERDICT_MAX_HOURS = 13
VERDICT_WAIT_MAX_BLOCKS = 3900            # 13 h of 12 s blocks
BACKTEST_WINDOWS_PER_CELL_MAX = 256
EMBARGO_RULE = "embargo-v1"               # the signed draw rule name (rows + receipts)
DEFAULT_FORWARD_HORIZONS = ("1mx60", "5mx36")
TIMESTRAL_MAX_HORIZON = 1024      # per forward pass; longer autoregresses on the median
TIMESTRAL_PATCH = 32

# Assets per class: the universe Hyperliquid LISTED on 2026-10-19 (main-dex
# perps for crypto; ``xyz`` dex perps for the rest, symbols without the
# ``xyz:`` prefix). FX: Hyperliquid lists only EUR, JPY and GBP (vs USD) — the
# other 25 of the 28 majors/crosses need another source (owner open item:
# Pyth Hermes polling, Chainlink on-chain FX as a cross-check). Alpha: dTAO
# subnet netuids, ``SN<netuid>``.
DEFAULT_ASSETS: dict[str, tuple[str, ...]] = {
    "crypto": ("BTC", "ETH", "SOL", "XRP", "HYPE", "BNB", "DOGE", "ADA", "AVAX", "LINK",
               "SUI", "TRX", "LTC", "DOT", "BCH", "NEAR", "APT", "ARB", "OP", "TAO", "ENA",
               "AAVE", "UNI", "INJ", "TIA", "SEI", "WIF", "JUP"),
    "fx": ("EUR", "JPY", "GBP"),
    "metals": ("GOLD", "SILVER", "PLATINUM", "PALLADIUM"),
    "commodities": ("CL", "BRENTOIL", "NATGAS", "COPPER"),
    "equity_etf": ("SP500", "XYZ100", "JP225", "KR200", "SMH", "XLE", "TLT", "XBI", "EWJ",
                   "EWY", "EWZ", "EWT", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
                   "TSLA"),
    "alpha": ("SN1", "SN8", "SN19", "SN50", "SN64", "SN88", "SN91", "SN123"),
}
# Which asset classes feed which suite (funding needs perps; alpha is its own
# class and suite).
PERP_CLASSES: tuple[str, ...] = ("crypto", "fx", "metals", "commodities", "equity_etf")
SUITE_CLASSES: dict[str, tuple[str, ...]] = {
    "returns": PERP_CLASSES, "vol": PERP_CLASSES, "range": PERP_CLASSES,
    "funding": PERP_CLASSES, "liquidity": PERP_CLASSES, "alpha": ("alpha",),
    "ohlcv": PERP_CLASSES, "high": PERP_CLASSES, "low": PERP_CLASSES,
}
# Series sampling intervals per suite (what the base sees as one step).
DEFAULT_FREQUENCIES: dict[str, tuple[str, ...]] = {
    "returns": ("1m", "5m", "1h", "1d"),
    "vol": ("5m", "15m", "1h", "1d"),
    "range": ("1h", "4h", "1d"),
    "funding": ("8h",),
    "liquidity": ("1h", "1d"),
    "alpha": ("1h", "1d"),
    "ohlcv": ("5m", "1h", "1d"),
    "high": ("1h", "4h", "1d"),
    "low": ("1h", "4h", "1d"),
}
# Forecast horizons per suite (durations). ``5d`` was dropped from returns
# (review 2026-10-19): a 6-day verdict wait cannot hold two aligned 5-day
# windows (:func:`scorability_problems`), and raising the wait for it would
# delay every verdict of the suite.
DEFAULT_HORIZONS: dict[str, tuple[str, ...]] = {
    "returns": ("1m", "5m", "1h", "4h", "1d"),
    "vol": ("5m", "15m", "1h", "1d"),
    "range": ("1h", "4h", "1d"),
    "funding": ("8h",),
    "liquidity": ("1h", "1d"),
    "alpha": ("1h", "1d"),
    "ohlcv": ("5m", "1h", "4h", "1d"),
    "high": ("1h", "4h", "1d"),
    "low": ("1h", "4h", "1d"),
}
# Recipe allow-lists. Covariates are INPUTS, never targets (DEC-CA-0064
# §Covariates); each has a builder in :func:`cascade.eval.marketcast.
# build_covariates`. ``open_interest`` and ``macro_events`` are NOT allowed:
# Hyperliquid serves no historical OI and no event calendar is pinned.
DEFAULT_COVARIATES: tuple[str, ...] = ("cross_asset", "funding", "volume", "calendar", "session",
                                       # owner 2026-11-18: cross markets, order flow, events
                                       "cross_eth", "cross_equity", "cross_gold", "cross_fx",
                                       "open_interest", "spread", "events")
# Each has a pure seeded transform in :mod:`cascade.trainer.marketcast_leg`.
DEFAULT_AUGMENTATIONS: tuple[str, ...] = ("scale", "jitter", "crop", "time_warp", "mixup")
TRAINABLE_SCOPES: tuple[str, ...] = ("full", "last_n_blocks", "frozen_embeddings", "lora")
TIMESTRAL_BLOCKS = 12             # len(config.json block_schedule) at the pinned commit
# Data sources: ids, the classes each may serve, and which have an adapter in
# this release (:mod:`cascade.pool.marketcast_feeds`). ``pyth_hermes`` /
# ``chainlink`` are the owner's FX candidates — recognised, NOT built, so a
# class naming one cannot be armed.
DATA_SOURCE_CLASSES: dict[str, tuple[str, ...]] = {
    "hyperliquid": ("crypto",),
    "hyperliquid_xyz": ("fx", "metals", "commodities", "equity_etf"),
    "binance_spot": ("crypto",),
    "bittensor_alpha": ("alpha",),
    "pyth_hermes": ("fx", "metals", "commodities", "equity_etf"),
    "chainlink": ("fx", "metals", "commodities"),
}
IMPLEMENTED_SOURCES: tuple[str, ...] = (
    "hyperliquid", "hyperliquid_xyz", "binance_spot", "bittensor_alpha")
# Per class: primary first, then cross-checks; "" = no source chosen (refuses
# arming for a class that lists assets).
DEFAULT_DATA_SOURCES: dict[str, str] = {
    "crypto": "hyperliquid+binance_spot",
    "fx": "hyperliquid_xyz",
    "metals": "hyperliquid_xyz",
    "commodities": "hyperliquid_xyz",
    "equity_etf": "hyperliquid_xyz",
    "alpha": "bittensor_alpha",
}
# Bittensor block time (the snapshot's block -> unix-time anchor).
BLOCK_SECONDS = 12.0
# The fewest finite context steps a window needs (:func:`cascade.eval.
# marketcast.cut_windows`).
MIN_CONTEXT = 32
# Banks (review 2026-10-19; native intervals 2026-11-10). Hyperliquid serves
# only the most recent ~5000 candles per interval (measured 2026-10-19: 1m
# ~3.6 days, 5m ~17 days, 15m ~52 days, 1h ~208 days, 4h ~833 days). A series
# at frequency ``f`` is cut from the COARSEST bank whose bar interval divides
# ``f`` (:meth:`MarketCastConfig.bank_minutes`): the 1-minute bars
# (``snapshot_days``), a native bank of ``native_intervals`` (5m / 15m,
# ``native_history_days``) or the hourly coarse bank (``coarse_history_days``).
# Every suite value is a function of the block's OHLCV candle (open = first
# bar's open, high / low = max / min, close = last bar's close, volume = sum),
# so cutting it from a finer bank gives the same value — the bank is only a
# history-depth choice. Realized vol is Garman–Klass on that candle (2026-11-10,
# owner: computable from native candles, so no 1-minute history is needed).
COARSE_BAR_MINUTES = 60
COARSE_MIN_FREQ_MINUTES = 60

_DUR_RE = re.compile(r"^([1-9][0-9]{0,4})(m|h|d)$")
# a multi-step horizon ``<bar>x<steps>`` (owner 2026-11-18): ``steps`` bars of
# ``<bar>`` ahead, e.g. ``1mx60`` = 60 one-minute bars, ``1dx7`` = a week of days
_PATH_RE = re.compile(r"^([1-9][0-9]{0,4}[mhd])x([1-9][0-9]{0,3})$")
_DUR_MIN = {"m": 1, "h": 60, "d": 1440}
_LABEL_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")
_ASSET_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,24}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESERVED_POOL_PREFIXES = ("pool", "pool_c", "pool_d")


def horizon_path(spec: str) -> tuple[str, int] | None:
    """``"1mx60"`` → ``("1m", 60)``: a multi-step horizon's bar and step
    count; None for a plain duration horizon."""
    m = _PATH_RE.match(str(spec))
    return (m.group(1), int(m.group(2))) if m else None


def duration_minutes(spec: str) -> int:
    """``"5m"`` / ``"4h"`` / ``"1d"`` → minutes; a multi-step horizon
    ``"<bar>x<steps>"`` → ``bar × steps`` minutes. Raises ``ValueError``."""
    p = horizon_path(spec)
    if p is not None:
        return duration_minutes(p[0]) * p[1]
    m = _DUR_RE.match(str(spec))
    if not m:
        raise ValueError(f"duration {spec!r} must be <int><m|h|d>, e.g. '5m', '4h', '1d'")
    return int(m.group(1)) * _DUR_MIN[m.group(2)]


def coarse_frequency(suite: str, freq_minutes: int) -> bool:
    """Whether ``suite`` at ``freq_minutes`` is cut from the snapshot's
    coarse (hourly) bank: every whole-hour frequency (``suite`` is kept for
    the call sites; since the Garman–Klass vol of 2026-11-10 no suite needs
    finer bars inside a block)."""
    f = int(freq_minutes)
    return f >= COARSE_MIN_FREQ_MINUTES and f % COARSE_MIN_FREQ_MINUTES == 0


def base_suite(label: str) -> str:
    """A window's suite label → its configured suite (``"ohlcv.c"`` →
    ``"ohlcv"``)."""
    return str(label).split(".", 1)[0]


def split_base_ref(ref: str) -> tuple[str, str]:
    """``"owner/name@<40-hex commit>"`` → ``(repo, revision)``; a bare
    ``"owner/name"`` is the UNPINNED placeholder (revision ``""``). A branch
    or tag name is refused — the base must be pinned to a commit."""
    repo, sep, rev = str(ref).partition("@")
    if not _REPO_RE.match(repo):
        raise ValueError(f"[marketcast] base_ref {ref!r}: repo must be <owner>/<name>")
    if sep and not _SHA1_RE.match(rev):
        raise ValueError(f"[marketcast] base_ref {ref!r}: revision must be a 40-hex commit "
                         "sha (never a branch or tag)")
    return repo, rev


@dataclass(frozen=True)
class MarketCastConfig:
    """``[marketcast]`` (DEC-CA-0064). Mapping-valued keys are stored as
    sorted ``((key, value), ...)`` tuples so the dataclass stays hashable;
    read them through the ``*_map`` properties."""

    # Activation: 0 = off; an era-boundary block arms it (loader checks).
    continuation_protocol: str = ""

    from_block: int = 0
    # Containment kill switch (owner 2026-11-24): DERIVED, never a
    # [marketcast] key — the loader copies ``[lane_c] until_block`` here so a
    # closed lane C also stops MarketCast judging (0 = no end).
    until_block: int = 0
    # The pinned base (release-only: a new architecture).
    base_ref: str = TIMESTRAL_PINNED_REF
    base_sha256: str = TIMESTRAL_SHA256
    base_arch: str = "timestral"
    quantile_knots: tuple[float, ...] = TIMESTRAL_KNOTS
    # What is judged.
    suites: tuple[str, ...] = DEFAULT_SUITES
    assets: tuple[tuple[str, tuple[str, ...]], ...] = tuple(sorted(DEFAULT_ASSETS.items()))
    frequencies: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
        sorted(DEFAULT_FREQUENCIES.items()))
    horizons: tuple[tuple[str, tuple[str, ...]], ...] = tuple(sorted(DEFAULT_HORIZONS.items()))
    data_sources: tuple[tuple[str, str], ...] = tuple(sorted(DEFAULT_DATA_SOURCES.items()))
    # Calibration gate (COVERAGE_GATE_SUITES): per horizon, a gated level fails
    # when its class-weighted empirical coverage is below nominal − tolerance
    # or above nominal + coverage_tolerance_over (each widened to
    # calibration_z × its standard error when the counts are thin).
    coverage_levels: tuple[float, ...] = (0.5, 0.8, 0.98)
    coverage_tolerance: float = 0.05
    # Verdict (lane-C shape; never a zero margin).
    alpha: float = 0.05
    bootstrap_B: int = 10_000
    gain_margin: float = 0.01
    # (``full_pay_tiers`` is retired, DEC-CA-0073: an mkt: win's full-pay
    # duration is lane C's gain-scaled rule, ramped from this gain_margin.)
    # review M10 (2026-10-19): clusters are asset class × time block, so the
    # floors count independent evidence, not windows of one regime
    min_windows: int = 200
    min_clusters: int = 8           # distinct (asset class, time block) clusters
    # Separate private pool of forward-only windows.
    pool_prefix: str = "pool_mkt"
    # Recipe bounds (:func:`cascade.shared.marketcast_recipe.parse_marketcast_recipe`).
    max_context_length: int = TIMESTRAL_MAX_CONTEXT
    covariates: tuple[str, ...] = DEFAULT_COVARIATES
    augmentations: tuple[str, ...] = DEFAULT_AUGMENTATIONS
    embargo_days: int = 30          # training data must end this long before submission
    # owner 2026-11-16: a recipe may train on the generator alone (1.0) or on
    # real data alone (0.0); at 1.0 its data.assets may be empty
    max_synthetic_fraction: float = 1.0
    # owner 2026-11-16: the recipe's own leg_hours (wall of its fine-tune leg,
    # miner-paid) is capped here; a recipe without it trains [lane_c] leg_hours
    max_leg_hours: float = 12.0
    max_lr: float = 1e-3
    max_steps: int = 100_000
    max_batch: int = 512
    max_lora_rank: int = 64
    # Window cutting / verdict timing (validator + audit).
    eval_context_length: int = 4096       # context steps both base and fine-tune see
    windows_per_cell: int = 48            # FORWARD windows per (asset, horizon), spread evenly
    start_margin_seconds: int = 3600      # forward windows start this long after dispatch
    verdict_wait_blocks: int = 3900       # an inconclusive entry waits this long (<= 13 h)
    snapshot_days: int = 21               # realized bars an eval snapshot carries
    # Review fixes 2026-10-19 (DEC-CA-0064 §Review fixes):
    coarse_history_days: int = 180        # hourly bars kept for >= 1h series (HL serves ~208 d)
    report_timeout_settlements: int = 8   # trainer records "absent" after this many settlements
    max_infra_deferrals: int = 8          # main-validator compute failures before an infra drop
    # Review fixes 2026-10-19, second batch (DEC-CA-0064 §Review fixes II):
    cluster_block_hours: int = 6          # bootstrap time block (clusters = class × block)
    min_time_blocks: int = 8              # distinct time blocks before a verdict (else wait)
    min_bar_coverage: float = 0.8         # share of live bars in the 24 h before a window
    min_dollar_volume_24h: float = 1_000_000.0   # USD traded in the 24 h before (0 = off)
    coverage_tolerance_over: float = 0.10  # gated levels also fail ABOVE nominal + this
    # per-window CRPS divisor (owner 2026-11-17, "do gift eval style"):
    # "level" = ohlcv ÷ last price, other suites in their units;
    # "seasonal_naive" = every window ÷ the GIFT-Eval/MASE seasonal-naive error of
    # its own context, and the training loss on the same scale (the default)
    crps_scale: str = "seasonal_naive"
    # trainer-side relay of the bar recording to training pods (release-only):
    # "<bucket>/<prefix>/" on the [storage] S3 endpoint, read with the
    # HIPPIUS_S3_* credentials; "" = off (the pod fetches the API itself)
    bar_relay: str = ""
    # the worker image MarketCast legs boot (trainer-only, release-only): a
    # "<repo>:<tag>@sha256:<64 hex>" ref with its worker_code_fingerprint
    # (scripts/worker_code_fingerprint.py), both or neither. Set, every
    # MarketCast leg rents and code-checks on these pins while every other leg
    # keeps [round] funded_pod_image / worker_code_fingerprint; "" = the fleet's.
    pod_image: str = ""
    worker_code_fingerprint: str = ""
    # the leg splits each recipe batch into micro-batches of at most this many
    # context steps (batch x context_length) and accumulates their gradients
    # (release-only): any recipe the bounds admit fits a 16 GB card (measured
    # 2026-11-17: full fine-tune 2^19 steps ≈ 9-12 GiB on an RTX 3060)
    micro_batch_tokens: int = 524_288
    coverage_gate_levels: tuple[float, ...] = (0.5, 0.8)   # 98 % is report-only
    # Red-team fixes 2026-10-20 (DEC-CA-0064 §Red-team fixes):
    # F1 — the gain is a geomean over (asset class × horizon) of the ratio of
    # MEAN losses pooled over the class's time blocks (the bootstrap resamples
    # class × time-block clusters); a cell (class × block × horizon) whose
    # base mean loss is ≤ zero_scale_frac × its horizon's median cell is
    # masked (recorded); each ratio is clipped to [1/cell_ratio_cap, cap].
    zero_scale_frac: float = 1e-3
    cell_ratio_cap: float = 4.0
    # F1b — the tail gate: the fine-tune's hit rate beyond each tail level
    # (below q_l for l < 0.5, above q_l for l > 0.5) must sit within
    # tail_tolerance of min(l, 1 − l), per horizon AND pooled over horizons.
    tail_levels: tuple[float, ...] = (0.05, 0.95)
    tail_tolerance: float = 0.025
    # F1b/F2 — every calibration tolerance is max(tolerance, calibration_z ×
    # the class-weighted standard error at the nominal rate), so a thin cell
    # never false-fails a calibrated forecaster.
    calibration_z: float = 3.0
    # F4 — behavioural near-duplicate: mean |Δq| between a candidate's and a
    # senior entry's forecasts on shared windows, over the base's 5–95 %
    # width, below this ⇒ duplicate (earliest commit holds).
    near_dup_tol: float = 0.02
    # F6 — trainer policy: open MarketCast legs per coldkey.
    max_open_legs_per_coldkey: int = 1
    # Owner 2026-11-24: the RECOMMENDED GPU types a MarketCast leg rents (the
    # default leg: cheapest fitting of these; any other Lium type is the
    # miner's own risk). () = [round] funded_pod_skus (pre-2026-11-24).
    funded_pod_skus: tuple[str, ...] = ()
    # Owner 2026-11-10 — one composite target (``mkt:market``) over every
    # enabled suite: the gain is the geomean over suites of each suite's
    # (class × horizon)-balanced gain, one cluster bootstrap for all; an entry
    # that is CONFIDENTLY worse on ANY suite (its UCB below ``-composite_suite_floor``) fails
    # (no-regression floor). ``false`` = the per-suite ``mkt:<suite>`` slots.
    composite: bool = False
    composite_suite_floor: float = 0.0
    # ohlcv: the scoring weight of the open field (the other fields weigh 1)
    ohlcv_open_weight: float = 0.0
    # the composite's per-suite weights (COMPOSITE_SUITE_WEIGHTS) and its
    # no-regression guards (GUARD_METRICS; () = none)
    composite_suite_weights: tuple[tuple[str, float], ...] = tuple(
        sorted(COMPOSITE_SUITE_WEIGHTS.items()))
    guards: tuple[str, ...] = GUARD_METRICS
    guard_floor: float = 0.0
    # Owner 2026-11-18: the composite must also beat the strong simple
    # baselines (cascade.eval.marketcast_baselines: random walk + EWMA / FHS,
    # seasonal AR(1)) — composite LCB >= 0, no suite confidently worse
    baseline_gate: bool = True
    # Owner 2026-11-18: the pinned SCHEDULED-event calendar the ``events``
    # covariate reads (unix instants of high-impact releases: FOMC, CPI,
    # NFP, …; known in advance, so a known-future input). Maintained in the
    # chain config from the official calendars; empty = the rows are 0 / 1.
    event_times: tuple[float, ...] = ()
    # Native candle banks (owner 2026-11-10) and the days each carries; empty =
    # the 1-minute + hourly banks only (the pre-2026-11-10 layout).
    native_intervals: tuple[str, ...] = ()
    native_history_days: tuple[tuple[str, int], ...] = (("15m", 50), ("5m", 21))
    # Owner 2026-11-24 (DEC-CA-0074): every horizon is judged on windows
    # DRAWN from the embargo — the ``embargo_eval_days`` before the leg's
    # dispatch, which no training leg reads (a recipe's data ends >=
    # ``embargo_days`` before its commit) — ``backtest_windows_per_cell`` per
    # (asset, series, horizon), ranked by the round block's hash (unknown
    # while the leg trains); judge and confirm draw from the two halves of a
    # fixed public window partition. The ``forward_horizons`` also take up to
    # ``windows_per_cell`` forward windows already closed in the snapshot
    # (start > dispatch + start_margin_seconds, end <= dispatch +
    # verdict_wait_blocks), never waited for.
    embargo_eval_days: int = 28
    backtest_windows_per_cell: int = 48
    forward_horizons: tuple[str, ...] = ()   # parse default: DEFAULT_FORWARD_HORIZONS enabled
    # DEC-CA-0075 (W-CMP): MarketCast as a declarative lane-C competition
    # spec (``[lane_c.specs.mkt]``, :mod:`cascade.shared.competition_spec`).
    # The pool's signer ("" / "owner" = the main validator's hotkey, today's
    # rule), its schema ("" = legacy snapshot OR standard pool, today; a
    # standard-pool version = that version only), the standard-pool tags a
    # series must carry ({} = every series), the rule menu picks, and the
    # typed refusal of the spec ("" = accepted; set ⇒ MarketCast is inert at
    # every block and its share burns).
    pool_signer: str = ""
    pool_schema: str = ""
    pool_select: tuple[tuple[str, str], ...] = ()
    # W-CMP2: metric channel name -> the pool's channel name ({} = same names)
    pool_channels: tuple[tuple[str, str], ...] = ()
    test_family: str = "bonferroni"
    pay_curve: str = "fixed_term"
    refused: str = ""
    # DEC-CA-0075 (W-CMP2): a lane-C spec INSTANCE on the marketcast runner —
    # its classes are its standard pool's ``class`` tags and every enabled
    # suite reads every class (no market feed, no SUITE_CLASSES table)
    open_classes: bool = False

    def active(self, block: int | None) -> bool:
        """Whether MarketCast judges a settlement at ``block`` (never while
        its competition spec is refused, DEC-CA-0075)."""
        return bool(self.from_block > 0 and block is not None and int(block) >= self.from_block
                    and (self.until_block <= 0 or int(block) <= self.until_block)
                    and not self.refused)

    @property
    def wait_blocks(self) -> int:
        """The inconclusive wait in force: ``verdict_wait_blocks`` capped at
        :data:`VERDICT_WAIT_MAX_BLOCKS` (owner 2026-11-24: <= 13 h)."""
        return min(int(self.verdict_wait_blocks), VERDICT_WAIT_MAX_BLOCKS)

    @property
    def base_pointer(self) -> str:
        """The identity MarketCast slots and pending entries are judged
        against (the lane-C ``base_pointer`` of an ``mkt:`` win): the pinned
        base, which never moves — so :func:`cascade.shared.lane_c.win_beats`
        always falls through to the LCB comparison."""
        return mkt_base_pointer(self)

    @property
    def target_names(self) -> tuple[str, ...]:
        """The ``mkt:<name>`` lane-C target names this config defines:
        ``("market",)`` when ``composite``, else one per suite."""
        return (COMPOSITE_TARGET,) if self.composite else tuple(self.suites)

    @property
    def target_suites(self) -> tuple[str, ...]:
        """The target names whose ``mkt:<name>`` lane-C targets are admitted
        — none while MarketCast is inert."""
        return self.target_names if self.from_block > 0 else ()

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(f"mkt:{s}" for s in self.target_names)

    def suites_of(self, name: str) -> tuple[str, ...]:
        """The suites a target name is judged over (the composite: every
        enabled suite)."""
        return tuple(self.suites) if name == COMPOSITE_TARGET else (str(name),)

    def series_labels(self, suite: str) -> tuple[str, ...]:
        """The window suite labels of one suite: ``ohlcv`` → one per scored
        field (``ohlcv.<field>``; the open only with ``ohlcv_open_weight``
        > 0), every other suite itself."""
        if suite != "ohlcv":
            return (suite,)
        return tuple(f"ohlcv.{f}" for f in OHLCV_FIELDS
                     if f != "o" or float(self.ohlcv_open_weight) > 0)

    @property
    def suite_weight_map(self) -> dict[str, float]:
        """The composite's weight of each ENABLED suite, summing to 1."""
        w = dict(self.composite_suite_weights)
        tot = sum(w[s] for s in self.suites)
        return {s: w[s] / tot for s in self.suites}

    def guards_of(self, label: str) -> tuple[str, ...]:
        """The composite guards a window of suite label ``label`` carries."""
        if not self.composite:
            return ()
        s = base_suite(label)
        return tuple(g for g in self.guards if s in GUARD_SUITES[g])

    @property
    def native_history_map(self) -> dict[str, int]:
        return dict(self.native_history_days)

    def bank_minutes(self, freq_minutes: int) -> int:
        """The bar interval (minutes) a series at ``freq_minutes`` is cut
        from: the coarsest of 1m, the ``native_intervals`` and the hourly
        bank that divides it."""
        f = int(freq_minutes)
        if coarse_frequency("", f):
            return COARSE_BAR_MINUTES
        best = 1
        for iv in self.native_intervals:
            m = duration_minutes(iv)
            if f % m == 0 and m > best:
                best = m
        return best

    def bank_days(self, bar_minutes: int) -> int:
        """Days of history the bank of ``bar_minutes`` carries."""
        bm = int(bar_minutes)
        if bm == COARSE_BAR_MINUTES:
            return int(self.coarse_history_days)
        if bm == 1:
            return int(self.snapshot_days)
        for iv, d in self.native_history_days:
            if duration_minutes(iv) == bm:
                return int(d)
        return int(self.snapshot_days)

    @property
    def assets_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.assets)

    @property
    def frequencies_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.frequencies)

    @property
    def horizons_map(self) -> dict[str, tuple[str, ...]]:
        return dict(self.horizons)

    @property
    def data_sources_map(self) -> dict[str, str]:
        return dict(self.data_sources)

    def sources_for(self, cls: str) -> tuple[str, ...]:
        """The class's source ids, primary first ("" ⇒ ``()``)."""
        return split_sources(self.data_sources_map.get(cls, ""))

    def suite_classes(self, suite: str) -> tuple[str, ...]:
        """The asset classes ``suite`` reads (:data:`SUITE_CLASSES`; a spec
        instance, ``open_classes``: every configured class)."""
        if self.open_classes:
            return tuple(c for c, _a in self.assets)
        return tuple(SUITE_CLASSES.get(suite, ()))

    def suite_assets(self, suite: str) -> list[tuple[str, str]]:
        """``[(class, symbol), ...]`` a suite is judged over (sorted)."""
        am = self.assets_map
        return sorted((c, a) for c in self.suite_classes(suite) for a in am.get(c, ()))

    @property
    def armed_classes(self) -> tuple[str, ...]:
        """The classes an enabled suite reads that list at least one asset."""
        am = self.assets_map
        return tuple(sorted({c for s in self.suites for c in self.suite_classes(s)
                             if am.get(c)}))


def mkt_base_pointer(mc) -> str:
    """``"<base_ref>|<base_sha256>"`` — see :attr:`MarketCastConfig.base_pointer`."""
    return f"{mc.base_ref}|{mc.base_sha256}"


def split_sources(spec: str) -> tuple[str, ...]:
    """``"hyperliquid+binance_spot"`` → ``("hyperliquid", "binance_spot")``."""
    return tuple(x for x in str(spec or "").split("+") if x)


def arming_problems(mc: MarketCastConfig) -> list[str]:
    """The table-local reasons ``mc`` cannot be armed (empty = armable)."""
    out: list[str] = []
    _repo, rev = split_base_ref(mc.base_ref)
    if not rev:
        out.append(f"base_ref {mc.base_ref!r} is not pinned to a commit (<repo>@<40-hex>)")
    if not mc.base_sha256:
        out.append("base_sha256 is unset (the base checkpoint is not pinned)")
    for cls in mc.armed_classes:
        srcs = mc.sources_for(cls)
        if not srcs:
            out.append(f"data_sources.{cls} is empty (an owner choice) but assets.{cls} "
                       "lists assets — set a source or empty the class")
            continue
        unbuilt = [x for x in srcs if x not in IMPLEMENTED_SOURCES]
        if unbuilt:
            out.append(f"data_sources.{cls} names {unbuilt}: no adapter in this release")
    return out + draw_problems(mc)


def scorability_problems(mc: MarketCastConfig) -> list[str]:
    """Why a configured (suite, horizon) could never yield a window (empty =
    every one is scorable). Per enabled suite and horizon, at the frequency
    it is forecast at (:func:`cascade.eval.marketcast.window_geometry`):

    * the history of the bank the series is cut from
      (:meth:`MarketCastConfig.bank_days`) holds at least ``MIN_CONTEXT +
      steps`` blocks.

    Whether the DEC-CA-0074 embargo draw can hold the windows is an arming
    check (:func:`draw_problems`)."""
    from ...eval.marketcast import window_geometry

    out: list[str] = []
    fm, hm = mc.frequencies_map, mc.horizons_map
    for s in mc.suites:
        for h in hm.get(s, ()):
            try:
                f, steps = window_geometry(fm.get(s, ()), h)
            except ValueError as e:
                out.append(f"{s}/{h}: {e}")
                continue
            days = mc.bank_days(mc.bank_minutes(f))
            blocks = int(days) * 1440 // int(f)
            if blocks < MIN_CONTEXT + steps:
                out.append(f"{s}/{h} at {f}m: {days} days of history hold {blocks} blocks < "
                           f"{MIN_CONTEXT} context + {steps} target")
    return out


def draw_problems(mc: MarketCastConfig) -> list[str]:
    """DEC-CA-0074: why the embargo draw could never judge (empty = it can;
    an ARMING check, :func:`arming_problems`): an embargo of >= 2 days (the
    draw reads only days no leg reads); per enabled (suite, horizon) the
    ``embargo_eval_days`` span holds an aligned window (window origins sit
    on the absolute UTC grid, so the first may start up to one horizon after
    the span opens: ``span >= 2 x horizon``) and a ``forward_horizons``
    horizon one in the forward span (``verdict_wait_blocks`` (capped) x
    BLOCK_SECONDS - start_margin_seconds); the span holds ``min_time_blocks``
    distinct cluster blocks."""
    out: list[str] = []
    if int(mc.embargo_days) < 2:
        out.append(f"embargo_days={mc.embargo_days}: the embargo draw needs >= 2 days no leg "
                   f"reads")
    span = int(mc.embargo_eval_days) * 86400
    fwd = int(mc.wait_blocks) * BLOCK_SECONDS - int(mc.start_margin_seconds)
    for s in mc.suites:
        for h in mc.horizons_map.get(s, ()):
            hs = 60 * duration_minutes(h)
            if span < 2 * hs:
                out.append(f"{s}/{h}: the {mc.embargo_eval_days}-day embargo eval span holds "
                           f"no aligned window (needs >= {2 * hs}s)")
            if h in tuple(mc.forward_horizons) and fwd < 2 * hs:
                out.append(f"{s}/{h}: a forward horizon must fit the forward span ({fwd:.0f}s "
                           f"= verdict_wait_blocks × {BLOCK_SECONDS:g} − start_margin_seconds; "
                           f"needs >= {2 * hs}s)")
    tb_s = int(mc.min_time_blocks) * int(mc.cluster_block_hours) * 3600
    if span < tb_s:
        out.append(f"the embargo eval span ({span}s) is shorter than min_time_blocks × "
                   f"cluster_block_hours ({tb_s}s): no verdict could ever conclude")
    return out


def _num(raw, name: str, *, integer: bool = False):
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or (
            integer and not isinstance(raw, int)):
        raise ValueError(f"[marketcast] {name} must be {'an integer' if integer else 'a number'}; "
                         f"got {raw!r}")
    return int(raw) if integer else float(raw)


def _str_list(raw, name: str) -> tuple[str, ...]:
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)) or not all(
            isinstance(x, str) for x in raw):
        raise ValueError(f"[marketcast] {name} must be a list of strings; got {raw!r}")
    return tuple(raw)


def _per_suite(raw, name: str, default: dict, suites: tuple[str, ...]):
    if not isinstance(raw, dict):
        raise ValueError(f"[marketcast] {name} must be a table suite -> [durations]; got {raw!r}")
    out: dict[str, tuple[str, ...]] = {}
    for suite, durs in raw.items():
        if suite not in SUITES:
            raise ValueError(f"[marketcast] {name} names unknown suite {suite!r}; "
                             f"suites are {SUITES}")
        ds = _str_list(durs, f"{name}.{suite}")
        if not ds or len(set(ds)) != len(ds):
            raise ValueError(f"[marketcast] {name}.{suite} must be a non-empty unique list")
        for d in ds:
            duration_minutes(d)
        out[suite] = ds
    missing = [s for s in suites if s not in out]
    if missing:
        raise ValueError(f"[marketcast] {name} is missing enabled suites {missing}")
    return out


def _funded_pod_skus(v: object) -> tuple[str, ...]:
    """``[marketcast] funded_pod_skus``: a list of unique GPU types (``"H100"``
    or a layout ``"<n>x<model>"``, n 1..99); ``[]`` / absent = ``()`` (the
    trainer then rents from ``[round] funded_pod_skus``)."""
    if v is None or v == () or v == []:
        return ()
    if isinstance(v, (str, bytes)) or not isinstance(v, (list, tuple)):
        raise ValueError(f"[marketcast] funded_pod_skus must be a list of GPU types; got {v!r}")
    out: list[str] = []
    for x in v:
        if not isinstance(x, str) or not _SKU_RE.match(x.strip()):
            raise ValueError(f"[marketcast] funded_pod_skus entry {x!r} is not a GPU type "
                             "('H100' or '<n>x<model>', letters/digits/._-)")
        out.append(x.strip())
    if len({s.lower() for s in out}) != len(out):
        raise ValueError(f"[marketcast] funded_pod_skus must be unique; got {v!r}")
    return tuple(out)


def is_gpu_type(sku: object) -> bool:
    """Whether ``sku`` is a well-formed GPU type (``"H100"``, ``"1xRTX4090"``)."""
    return isinstance(sku, str) and bool(_SKU_RE.match(sku.strip()))


def _parse_spec_fields(mc: dict) -> dict:
    """DEC-CA-0075: the spec-compiled keys (``pool_signer``, ``pool_schema``,
    ``pool_select``, ``test_family``, ``pay_curve``, ``refused``); absent =
    today's rule."""
    from ..competition_spec import POOL_SCHEMAS
    from ..rules_menu import check_choice

    out: dict = {}
    for k in ("pool_signer", "pool_schema", "refused"):
        v = mc.get(k, "")
        if not isinstance(v, str):
            raise ValueError(f"[marketcast] {k} must be a string; got {v!r}")
        out[k] = "" if (k == "pool_signer" and v == "owner") else v
    if out["pool_schema"] not in POOL_SCHEMAS:
        raise ValueError(f"[marketcast] pool_schema must be one of {POOL_SCHEMAS}; "
                         f"got {out['pool_schema']!r}")
    sel = mc.get("pool_select", {})
    if not isinstance(sel, dict) or not all(isinstance(a, str) and isinstance(b, str) and b
                                            for a, b in sel.items()):
        raise ValueError(f"[marketcast] pool_select must be a table tag -> value; got {sel!r}")
    out["pool_select"] = tuple(sorted(sel.items()))
    chn = mc.get("pool_channels", {})
    if not isinstance(chn, dict) or not all(isinstance(a, str) and isinstance(b, str) and b
                                            for a, b in chn.items()):
        raise ValueError(f"[marketcast] pool_channels must be a table channel -> pool "
                         f"channel; got {chn!r}")
    out["pool_channels"] = tuple(sorted(chn.items()))
    out["test_family"], out["pay_curve"] = check_choice(
        "lane_c", test_family=mc.get("test_family", "bonferroni"),
        pay_curve=mc.get("pay_curve", "fixed_term"))
    return out


_OPEN_CLASS_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


def parse_marketcast(raw: object, *, open_classes: bool = False) -> MarketCastConfig:
    """Parse + validate ``[marketcast]`` (absent ⇒ the inert defaults).
    Refuses arming (``from_block`` > 0) and malformed placeholders."""
    mc = dict(raw) if isinstance(raw, dict) else {}
    fb = mc.get("from_block", 0)
    if isinstance(fb, bool) or not isinstance(fb, int) or fb < 0:
        raise ValueError(f"[marketcast] from_block must be an integer >= 0; got {fb!r}")
    base_ref = str(mc.get("base_ref", TIMESTRAL_PINNED_REF))
    split_base_ref(base_ref)
    sha = str(mc.get("base_sha256", TIMESTRAL_SHA256)).lower()
    if sha and not _SHA256_RE.match(sha):
        raise ValueError(f"[marketcast] base_sha256 must be '' (TBD) or 64 hex; got {sha!r}")
    arch = str(mc.get("base_arch", "timestral"))
    if arch not in BASE_ARCHS:
        raise ValueError(f"[marketcast] base_arch must be one of {BASE_ARCHS}; got {arch!r}")
    knots_raw = mc.get("quantile_knots", TIMESTRAL_KNOTS)
    if isinstance(knots_raw, (str, bytes)) or not isinstance(knots_raw, (list, tuple)):
        raise ValueError(f"[marketcast] quantile_knots must be a list; got {knots_raw!r}")
    knots = tuple(_num(k, "quantile_knots") for k in knots_raw)
    if (len(knots) < 3 or not all(0.0 < k < 1.0 for k in knots)
            or any(b <= a for a, b in zip(knots, knots[1:], strict=False))
            or 0.5 not in knots):
        raise ValueError("[marketcast] quantile_knots must be >= 3 strictly increasing levels "
                         f"in (0, 1) including the median 0.5; got {list(knots)}")
    suites = _str_list(mc.get("suites", DEFAULT_SUITES), "suites")
    if not suites or len(set(suites)) != len(suites) or not set(suites) <= set(SUITES):
        raise ValueError(f"[marketcast] suites must be a non-empty unique subset of {SUITES}; "
                         f"got {list(suites)}")
    assets_raw = mc.get("assets", DEFAULT_ASSETS)
    if not isinstance(assets_raw, dict):
        raise ValueError(f"[marketcast] assets must be a table class -> [symbols]; "
                         f"got {assets_raw!r}")
    assets: dict[str, tuple[str, ...]] = {}
    for cls, syms in assets_raw.items():
        if open_classes:
            # DEC-CA-0075 (W-CMP2): a spec instance's classes are the standard
            # pool's ``class`` tags (any lower-case label); no feed reads them
            if not isinstance(cls, str) or not _OPEN_CLASS_RE.match(cls):
                raise ValueError(f"[marketcast] assets class {cls!r} must be [a-z][a-z0-9_]{{0,31}}")
        elif cls not in ASSET_CLASSES:
            raise ValueError(f"[marketcast] assets names unknown class {cls!r}; classes are "
                             f"{ASSET_CLASSES}")
        ss = _str_list(syms, f"assets.{cls}")
        if len(set(ss)) != len(ss) or not all(_ASSET_RE.match(s) for s in ss):
            raise ValueError(f"[marketcast] assets.{cls} must be unique symbols "
                             "[A-Za-z0-9_.-]{1,24}")
        assets[cls] = ss
    freqs = _per_suite(mc.get("frequencies", DEFAULT_FREQUENCIES), "frequencies",
                       DEFAULT_FREQUENCIES, suites)
    hzs = _per_suite(mc.get("horizons", DEFAULT_HORIZONS), "horizons", DEFAULT_HORIZONS, suites)
    paths = sorted({f for fs in freqs.values() for f in fs if horizon_path(f) is not None})
    if paths:
        raise ValueError(f"[marketcast] frequencies {paths} must be bars, not <bar>x<steps> paths")
    for s in suites:
        lo = min(duration_minutes(f) for f in freqs[s])
        short = [h for h in hzs[s] if duration_minutes(h) < lo]
        if short:
            raise ValueError(f"[marketcast] horizons.{s} {short} are shorter than the suite's "
                             f"finest frequency")
    src_raw = mc.get("data_sources", DEFAULT_DATA_SOURCES)
    if not isinstance(src_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in src_raw.items()):
        raise ValueError(f"[marketcast] data_sources must be a table class -> source; "
                         f"got {src_raw!r}")
    bad = [k for k in src_raw if k not in ASSET_CLASSES]
    if bad and not open_classes:
        raise ValueError(f"[marketcast] data_sources names unknown classes {bad}")
    sources = {**DEFAULT_DATA_SOURCES, **src_raw}
    for cls, spec in sources.items():
        ids = split_sources(spec)
        if len(set(ids)) != len(ids) or (spec and not ids):
            raise ValueError(f"[marketcast] data_sources.{cls} {spec!r} must be '' or "
                             "'<source>[+<source>...]' without repeats")
        for x in ids:
            if x not in DATA_SOURCE_CLASSES:
                raise ValueError(f"[marketcast] data_sources.{cls}: unknown source {x!r}; "
                                 f"known {sorted(DATA_SOURCE_CLASSES)}")
            if cls not in DATA_SOURCE_CLASSES[x]:
                raise ValueError(f"[marketcast] data_sources.{cls}: source {x!r} serves "
                                 f"{DATA_SOURCE_CLASSES[x]}, not {cls!r}")
    cov_raw = mc.get("coverage_levels", (0.5, 0.8, 0.98))
    if isinstance(cov_raw, (str, bytes)) or not isinstance(cov_raw, (list, tuple)):
        raise ValueError(f"[marketcast] coverage_levels must be a list; got {cov_raw!r}")
    cov = tuple(_num(c, "coverage_levels") for c in cov_raw)
    if not cov or len(set(cov)) != len(cov) or not all(0.0 < c < 1.0 for c in cov):
        raise ValueError(f"[marketcast] coverage_levels must be unique levels in (0, 1); "
                         f"got {list(cov)}")
    for c in cov:
        lo_q, hi_q = (1.0 - c) / 2.0, (1.0 + c) / 2.0
        if lo_q < knots[0] - 1e-12 or hi_q > knots[-1] + 1e-12:
            raise ValueError(f"[marketcast] coverage level {c} needs quantiles {lo_q:g}/{hi_q:g} "
                             f"outside the base's knots [{knots[0]}, {knots[-1]}]")
    tol = _num(mc.get("coverage_tolerance", 0.05), "coverage_tolerance")
    if not 0.0 < tol < 0.5:
        raise ValueError(f"[marketcast] coverage_tolerance must be in (0, 0.5); got {tol}")
    alpha = _num(mc.get("alpha", 0.05), "alpha")
    if not 0.0 < alpha < 0.5:
        raise ValueError(f"[marketcast] alpha must be in (0, 0.5); got {alpha}")
    B = _num(mc.get("bootstrap_B", 10_000), "bootstrap_B", integer=True)
    if B < 100:
        raise ValueError(f"[marketcast] bootstrap_B must be >= 100; got {B}")
    margin = _num(mc.get("gain_margin", 0.01), "gain_margin")
    if not 0.0 < margin < 1.0:
        raise ValueError(f"[marketcast] gain_margin must be in (0, 1) — a zero bar lets an "
                         f"untouched base win ~alpha of the time; got {margin}")
    mw = _num(mc.get("min_windows", 200), "min_windows", integer=True)
    mcl = _num(mc.get("min_clusters", 8), "min_clusters", integer=True)
    if mw < 1 or not 1 <= mcl <= mw:
        raise ValueError("[marketcast] need min_windows >= 1 and 1 <= min_clusters <= "
                         "min_windows")
    pp = str(mc.get("pool_prefix", "pool_mkt"))
    if not _LABEL_RE.match(pp) or pp in RESERVED_POOL_PREFIXES:
        raise ValueError(f"[marketcast] pool_prefix {pp!r} must match [a-z0-9_-]{{1,32}} and "
                         f"differ from {RESERVED_POOL_PREFIXES}")
    mcl_ctx = _num(mc.get("max_context_length", TIMESTRAL_MAX_CONTEXT), "max_context_length",
                   integer=True)
    if not TIMESTRAL_PATCH <= mcl_ctx <= TIMESTRAL_MAX_CONTEXT:
        raise ValueError(f"[marketcast] max_context_length must be in [{TIMESTRAL_PATCH}, "
                         f"{TIMESTRAL_MAX_CONTEXT}]; got {mcl_ctx}")
    covs = _str_list(mc.get("covariates", DEFAULT_COVARIATES), "covariates")
    augs = _str_list(mc.get("augmentations", DEFAULT_AUGMENTATIONS), "augmentations")
    for name, xs in (("covariates", covs), ("augmentations", augs)):
        if len(set(xs)) != len(xs) or not all(_LABEL_RE.match(x) for x in xs):
            raise ValueError(f"[marketcast] {name} must be unique labels [a-z0-9_-]{{1,32}}")
    emb = _num(mc.get("embargo_days", 30), "embargo_days", integer=True)
    if emb < 0:
        raise ValueError(f"[marketcast] embargo_days must be >= 0; got {emb}")
    syn = _num(mc.get("max_synthetic_fraction", 1.0), "max_synthetic_fraction")
    if not 0.0 <= syn <= 1.0:
        raise ValueError(f"[marketcast] max_synthetic_fraction must be in [0, 1]; got {syn}")
    mlh = _num(mc.get("max_leg_hours", 12.0), "max_leg_hours")
    if not 0.0 < mlh <= MAX_LEG_HOURS_CEILING:
        raise ValueError(f"[marketcast] max_leg_hours must be in (0, {MAX_LEG_HOURS_CEILING:g}]; "
                         f"got {mlh}")
    mlr = _num(mc.get("max_lr", 1e-3), "max_lr")
    msteps = _num(mc.get("max_steps", 100_000), "max_steps", integer=True)
    mbatch = _num(mc.get("max_batch", 512), "max_batch", integer=True)
    mrank = _num(mc.get("max_lora_rank", 64), "max_lora_rank", integer=True)
    if not 0.0 < mlr <= 1.0 or msteps < 1 or mbatch < 1 or mrank < 1:
        raise ValueError("[marketcast] need 0 < max_lr <= 1 and max_steps, max_batch, "
                         "max_lora_rank >= 1")
    ectx = _num(mc.get("eval_context_length", 4096), "eval_context_length", integer=True)
    if not TIMESTRAL_PATCH <= ectx <= mcl_ctx:
        raise ValueError(f"[marketcast] eval_context_length must be in [{TIMESTRAL_PATCH}, "
                         f"max_context_length]; got {ectx}")
    wpc = _num(mc.get("windows_per_cell", 48), "windows_per_cell", integer=True)
    marg = _num(mc.get("start_margin_seconds", 3600), "start_margin_seconds", integer=True)
    wait = _num(mc.get("verdict_wait_blocks", 3900), "verdict_wait_blocks", integer=True)
    sdays = _num(mc.get("snapshot_days", 21), "snapshot_days", integer=True)
    if wpc < 1 or marg < 0 or wait < 0 or sdays < 1:
        raise ValueError("[marketcast] need windows_per_cell >= 1, start_margin_seconds >= 0, "
                         "verdict_wait_blocks >= 0 and snapshot_days >= 1")
    if wait > VERDICT_WAIT_MAX_BLOCKS:
        # owner 2026-11-24 (DEC-CA-0074): a verdict within 13 h — a larger
        # value (an older deployed file, an overlay) is capped in code
        _log.warning("[marketcast] verdict_wait_blocks=%d is capped at %d (a verdict within "
                     "%d h, DEC-CA-0074)", wait, VERDICT_WAIT_MAX_BLOCKS, VERDICT_MAX_HOURS)
    cdays = _num(mc.get("coarse_history_days", 180), "coarse_history_days", integer=True)
    rto = _num(mc.get("report_timeout_settlements", 8), "report_timeout_settlements",
               integer=True)
    mid = _num(mc.get("max_infra_deferrals", 8), "max_infra_deferrals", integer=True)
    if cdays < sdays or rto < 1 or mid < 0:
        raise ValueError("[marketcast] need coarse_history_days >= snapshot_days, "
                         "report_timeout_settlements >= 1 and max_infra_deferrals >= 0")
    cbh = _num(mc.get("cluster_block_hours", 6), "cluster_block_hours", integer=True)
    mtb = _num(mc.get("min_time_blocks", 8), "min_time_blocks", integer=True)
    if not 1 <= cbh <= 24 or 24 % cbh or mtb < 1:
        raise ValueError("[marketcast] need cluster_block_hours in 1..24 dividing 24 and "
                         "min_time_blocks >= 1")
    if "min_target_abs_bp" in mc:
        raise ValueError("[marketcast] min_target_abs_bp was removed (red-team F3, 2026-10-20): "
                         "a filter on the REALISED target conditions the test set on the "
                         "outcome; zero-scale cells are masked by zero_scale_frac instead")
    mbc = _num(mc.get("min_bar_coverage", 0.8), "min_bar_coverage")
    mdv = _num(mc.get("min_dollar_volume_24h", 1_000_000.0), "min_dollar_volume_24h")
    if not 0.0 <= mbc <= 1.0 or mdv < 0:
        raise ValueError("[marketcast] need min_bar_coverage in [0, 1] and "
                         "min_dollar_volume_24h >= 0")
    tov = _num(mc.get("coverage_tolerance_over", 0.10), "coverage_tolerance_over")
    crps_scale = str(mc.get("crps_scale", "seasonal_naive"))
    bar_relay = str(mc.get("bar_relay", "") or "")
    mbt = _num(mc.get("micro_batch_tokens", 524_288), "micro_batch_tokens", integer=True)
    if mbt < 1:
        raise ValueError(f"[marketcast] micro_batch_tokens must be >= 1; got {mbt}")
    pod_image = str(mc.get("pod_image", "") or "")
    wcf = str(mc.get("worker_code_fingerprint", "") or "").lower()
    if bool(pod_image) != bool(wcf):
        raise ValueError("[marketcast] pod_image and worker_code_fingerprint are set together "
                         "(both or neither)")
    if pod_image and not re.search(r"@sha256:[0-9a-f]{64}$", pod_image):
        raise ValueError(f"[marketcast] pod_image must be pinned by digest "
                         f"('<repo>:<tag>@sha256:<64 hex>'); got {pod_image!r}")
    if wcf and not re.fullmatch(r"[0-9a-f]{64}", wcf):
        raise ValueError(f"[marketcast] worker_code_fingerprint must be 64 hex; got {wcf!r}")
    if bar_relay and ("/" not in bar_relay.strip("/") or not bar_relay.endswith("/")):
        raise ValueError(f"[marketcast] bar_relay must be '<bucket>/<prefix>/'; got {bar_relay!r}")
    if crps_scale not in CRPS_SCALES:
        raise ValueError(f"[marketcast] crps_scale must be one of {CRPS_SCALES}; got {crps_scale!r}")
    if not 0.0 < tov < 0.5:
        raise ValueError(f"[marketcast] coverage_tolerance_over must be in (0, 0.5); got {tov}")
    cgl_raw = mc.get("coverage_gate_levels", (0.5, 0.8))
    if isinstance(cgl_raw, (str, bytes)) or not isinstance(cgl_raw, (list, tuple)):
        raise ValueError(f"[marketcast] coverage_gate_levels must be a list; got {cgl_raw!r}")
    cgl = tuple(_num(c, "coverage_gate_levels") for c in cgl_raw)
    if not set(cgl) <= set(cov) or len(set(cgl)) != len(cgl):
        raise ValueError(f"[marketcast] coverage_gate_levels {list(cgl)} must be unique levels "
                         f"of coverage_levels {list(cov)}")
    zsf = _num(mc.get("zero_scale_frac", 1e-3), "zero_scale_frac")
    cap = _num(mc.get("cell_ratio_cap", 4.0), "cell_ratio_cap")
    if not 0.0 <= zsf < 1.0 or not cap > 1.0:
        raise ValueError("[marketcast] need zero_scale_frac in [0, 1) and cell_ratio_cap > 1")
    tl_raw = mc.get("tail_levels", (0.05, 0.95))
    if isinstance(tl_raw, (str, bytes)) or not isinstance(tl_raw, (list, tuple)):
        raise ValueError(f"[marketcast] tail_levels must be a list; got {tl_raw!r}")
    tls = tuple(_num(x, "tail_levels") for x in tl_raw)
    if len(set(tls)) != len(tls) or not all(x in knots and x != 0.5 for x in tls):
        raise ValueError(f"[marketcast] tail_levels {list(tls)} must be unique knots of "
                         f"quantile_knots other than the median")
    ttol = _num(mc.get("tail_tolerance", 0.025), "tail_tolerance")
    cz = _num(mc.get("calibration_z", 3.0), "calibration_z")
    if not 0.0 < ttol < 0.5 or cz < 0:
        raise ValueError("[marketcast] need tail_tolerance in (0, 0.5) and calibration_z >= 0")
    ndt = _num(mc.get("near_dup_tol", 0.02), "near_dup_tol")
    if not 0.0 <= ndt < 1.0:
        raise ValueError(f"[marketcast] near_dup_tol must be in [0, 1); got {ndt}")
    molc = _num(mc.get("max_open_legs_per_coldkey", 1), "max_open_legs_per_coldkey",
                integer=True)
    if molc < 1:
        raise ValueError("[marketcast] max_open_legs_per_coldkey must be >= 1")
    mkt_skus = _funded_pod_skus(mc.get("funded_pod_skus", ()))
    comp = mc.get("composite", False)
    if not isinstance(comp, bool):
        raise ValueError(f"[marketcast] composite must be true or false; got {comp!r}")
    floor = _num(mc.get("composite_suite_floor", 0.0), "composite_suite_floor")
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"[marketcast] composite_suite_floor must be in [0, 1); got {floor}")
    ow = _num(mc.get("ohlcv_open_weight", 0.0), "ohlcv_open_weight")
    if not 0.0 <= ow <= 1.0:
        raise ValueError(f"[marketcast] ohlcv_open_weight must be in [0, 1]; got {ow}")
    sw_raw = mc.get("composite_suite_weights", COMPOSITE_SUITE_WEIGHTS)
    if not isinstance(sw_raw, dict):
        raise ValueError(f"[marketcast] composite_suite_weights must be a table suite -> "
                         f"weight; got {sw_raw!r}")
    sw = dict(COMPOSITE_SUITE_WEIGHTS)
    for s_, w_ in sw_raw.items():
        if s_ not in SUITES:
            raise ValueError(f"[marketcast] composite_suite_weights names unknown suite {s_!r}")
        sw[s_] = _num(w_, f"composite_suite_weights.{s_}")
        if not sw[s_] > 0:
            raise ValueError(f"[marketcast] composite_suite_weights.{s_} must be > 0")
    gd = _str_list(mc.get("guards", GUARD_METRICS), "guards")
    if len(set(gd)) != len(gd) or not set(gd) <= set(GUARD_METRICS):
        raise ValueError(f"[marketcast] guards must be unique members of {GUARD_METRICS}; "
                         f"got {list(gd)}")
    if mc.get("gap_horizons"):
        # retired (owner 2026-11-24, DEC-CA-0074): every horizon is drawn from
        # the embargo; an older file still naming it loads, the key does nothing
        _log.warning("[marketcast] gap_horizons is retired (DEC-CA-0074): every horizon is "
                     "drawn from the embargo; the key is ignored")
    fed = _num(mc.get("embargo_eval_days", min(28, max(1, emb - 1))), "embargo_eval_days",
               integer=True)
    bwpc = _num(mc.get("backtest_windows_per_cell", 48), "backtest_windows_per_cell",
                integer=True)
    if not 1 <= fed <= max(1, emb - 1):
        raise ValueError(f"[marketcast] embargo_eval_days must be in [1, embargo_days - 1] "
                         f"(the draw stays inside the embargo no leg reads); got {fed}")
    if not 1 <= bwpc <= BACKTEST_WINDOWS_PER_CELL_MAX:
        raise ValueError(f"[marketcast] backtest_windows_per_cell must be in "
                         f"[1, {BACKTEST_WINDOWS_PER_CELL_MAX}]; got {bwpc}")
    # absent = the default short paths this config enables (owner 2026-11-24:
    # 1m and 5m fit the 13 h without a wait); listed = exactly those
    fh = (_str_list(mc["forward_horizons"], "forward_horizons") if "forward_horizons" in mc
          else tuple(h_ for h_ in DEFAULT_FORWARD_HORIZONS
                     if any(h_ in hzs.get(s_, ()) for s_ in suites)))
    if len(set(fh)) != len(fh):
        raise ValueError(f"[marketcast] forward_horizons must be unique; got {list(fh)}")
    for h_ in fh:
        if not any(h_ in hzs.get(s_, ()) for s_ in suites):
            raise ValueError(f"[marketcast] forward_horizons {h_!r} is no enabled suite's "
                             f"horizon")
    ev_raw = mc.get("event_times", ())
    if not isinstance(ev_raw, (list, tuple)):
        raise ValueError(f"[marketcast] event_times must be a list of ISO UTC instants; got "
                         f"{ev_raw!r}")
    evs = []
    for x in ev_raw:
        try:
            dt = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"[marketcast] event_times {x!r} is not an ISO instant") from e
        if dt.tzinfo is None:
            raise ValueError(f"[marketcast] event_times {x!r} needs a UTC offset (…Z)")
        evs.append(dt.timestamp())
    bgate = mc.get("baseline_gate", True)
    if not isinstance(bgate, bool):
        raise ValueError(f"[marketcast] baseline_gate must be true or false; got {bgate!r}")
    gfl = _num(mc.get("guard_floor", 0.0), "guard_floor")
    if not 0.0 <= gfl < 1.0:
        raise ValueError(f"[marketcast] guard_floor must be in [0, 1); got {gfl}")
    if "tail" in gd and not set(TAIL_SCORE_LEVELS) <= set(knots):
        raise ValueError(f"[marketcast] the tail guard needs {list(TAIL_SCORE_LEVELS)} among "
                         f"quantile_knots")
    nat = _str_list(mc.get("native_intervals", ()), "native_intervals")
    if len(set(nat)) != len(nat) or not set(nat) <= set(NATIVE_INTERVALS):
        raise ValueError(f"[marketcast] native_intervals must be unique members of "
                         f"{NATIVE_INTERVALS}; got {list(nat)}")
    nhd_raw = mc.get("native_history_days", {"5m": 21, "15m": 50})
    if not isinstance(nhd_raw, dict):
        raise ValueError(f"[marketcast] native_history_days must be a table interval -> days; "
                         f"got {nhd_raw!r}")
    nhd = {"5m": 21, "15m": 50}
    for iv, d in nhd_raw.items():
        if iv not in NATIVE_INTERVALS:
            raise ValueError(f"[marketcast] native_history_days names {iv!r}; intervals are "
                             f"{NATIVE_INTERVALS}")
        nhd[iv] = _num(d, f"native_history_days.{iv}", integer=True)
        if nhd[iv] < 1:
            raise ValueError(f"[marketcast] native_history_days.{iv} must be >= 1")
    if comp and len(suites) < 2:
        raise ValueError("[marketcast] composite needs at least two suites (one suite is its "
                         "own mkt:<suite> target)")
    from .training import _continuation_protocol
    cfg_out = MarketCastConfig(
        continuation_protocol=_continuation_protocol(mc.get("continuation_protocol", "")),
        max_context_length=mcl_ctx, covariates=covs, augmentations=augs, embargo_days=emb,
        max_synthetic_fraction=syn, max_leg_hours=mlh, max_lr=mlr, max_steps=msteps, max_batch=mbatch,
        max_lora_rank=mrank,
        from_block=fb, base_ref=base_ref, base_sha256=sha, base_arch=arch,
        quantile_knots=knots, suites=suites,
        assets=tuple(sorted(assets.items())),
        frequencies=tuple(sorted(freqs.items())),
        horizons=tuple(sorted(hzs.items())),
        data_sources=tuple(sorted(sources.items())),
        coverage_levels=cov, coverage_tolerance=tol, alpha=alpha, bootstrap_B=B,
        gain_margin=margin, min_windows=mw, min_clusters=mcl, pool_prefix=pp,
        eval_context_length=ectx, windows_per_cell=wpc, start_margin_seconds=marg,
        verdict_wait_blocks=wait, snapshot_days=sdays, coarse_history_days=cdays,
        report_timeout_settlements=rto, max_infra_deferrals=mid,
        cluster_block_hours=cbh, min_time_blocks=mtb,
        min_bar_coverage=mbc, min_dollar_volume_24h=mdv, coverage_tolerance_over=tov, crps_scale=crps_scale, bar_relay=bar_relay,
        pod_image=pod_image, worker_code_fingerprint=wcf,
        micro_batch_tokens=mbt,
        coverage_gate_levels=cgl, zero_scale_frac=zsf, cell_ratio_cap=cap, tail_levels=tls,
        tail_tolerance=ttol, calibration_z=cz, near_dup_tol=ndt,
        max_open_legs_per_coldkey=molc, funded_pod_skus=mkt_skus,
        composite=comp, composite_suite_floor=floor, ohlcv_open_weight=ow,
        composite_suite_weights=tuple(sorted(sw.items())), guards=gd, guard_floor=gfl,
        baseline_gate=bgate, event_times=tuple(sorted(evs)),
        native_intervals=nat, native_history_days=tuple(sorted(nhd.items())),
        embargo_eval_days=fed, backtest_windows_per_cell=bwpc, forward_horizons=fh,
        **_parse_spec_fields(mc), open_classes=bool(open_classes),
    )
    unscorable = scorability_problems(cfg_out)
    if unscorable:
        raise ValueError("[marketcast] unscorable (suite, horizon): " + "; ".join(unscorable))
    if fb:
        bad = arming_problems(cfg_out)
        if bad:
            raise ValueError(f"[marketcast] from_block={fb} cannot arm MarketCast: "
                             + "; ".join(bad))
    return cfg_out


def check_marketcast_arming(mc: MarketCastConfig, *, era_blocks: int, v2_from_block: int,
                            lane_c_from_block: int, validator_bench_from: int,
                            owner_hotkey: str) -> None:
    """The cross-table arming checks (the loader calls this when
    ``from_block > 0``): an era boundary at or after the v2 go-live and lane
    C's own block, with the DEC-CA-0063 main-validator bench source in force
    by then (the main validator's GPU scores MarketCast) and a named main
    validator (``[netconfig] owner_hotkey``: it signs the snapshot and the
    rows). Raises ``ValueError``."""
    fb = int(mc.from_block)
    if not fb:
        return
    probs = []
    if era_blocks > 0 and fb % int(era_blocks):
        probs.append(f"must be an era boundary (a multiple of {era_blocks})")
    if not v2_from_block or fb < int(v2_from_block):
        probs.append(f"must be >= [v2] from_block ({v2_from_block or 'unset'})")
    if not lane_c_from_block or fb < int(lane_c_from_block):
        probs.append(f"must be >= [lane_c] from_block ({lane_c_from_block or 'unset'})")
    if not validator_bench_from or fb < int(validator_bench_from):
        probs.append("needs [v2] bench_source = \"validator\" in force at or before it "
                     f"(bench_source_from_block={validator_bench_from or 'unset'})")
    if not str(owner_hotkey or ""):
        probs.append("needs [netconfig] owner_hotkey (the main validator signs the snapshot "
                     "and the rows)")
    if probs:
        raise ValueError(f"[marketcast] from_block={fb}: " + "; ".join(probs))


__all__ = [
    "ASSET_CLASSES", "BASE_ARCHS", "DEFAULT_AUGMENTATIONS", "DEFAULT_COVARIATES",
    "TIMESTRAL_BLOCKS", "TRAINABLE_SCOPES", "COVERAGE_GATE_SUITES", "DEFAULT_SUITES",
    "MarketCastConfig", "SUITES", "TIMESTRAL_KNOTS", "TIMESTRAL_MAX_CONTEXT",
    "TIMESTRAL_MAX_HORIZON", "TIMESTRAL_PATCH", "TIMESTRAL_REPO", "duration_minutes",
    "parse_marketcast", "split_base_ref", "BLOCK_SECONDS", "DATA_SOURCE_CLASSES",
    "IMPLEMENTED_SOURCES", "PERP_CLASSES", "SUITE_CLASSES", "TIMESTRAL_PINNED_REF",
    "TIMESTRAL_REVISION", "TIMESTRAL_SHA256", "arming_problems", "check_marketcast_arming",
    "split_sources", "COARSE_BAR_MINUTES", "MIN_CONTEXT", "coarse_frequency",
    "mkt_base_pointer", "scorability_problems", "COMPOSITE_LAUNCH_SUITES", "COMPOSITE_TARGET",
    "NATIVE_INTERVALS", "OHLCV_FIELDS", "OHLCV_FIELD_NAMES", "base_suite",
    "horizon_path", "COMPOSITE_SUITE_WEIGHTS", "GUARD_METRICS", "GUARD_SUITES", "TAIL_SCORE_LEVELS",
    "BACKTEST_WINDOWS_PER_CELL_MAX", "draw_problems", "is_gpu_type", "EMBARGO_RULE", "VERDICT_MAX_HOURS", "VERDICT_WAIT_MAX_BLOCKS",
]
