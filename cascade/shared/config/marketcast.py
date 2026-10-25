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

import re
from dataclasses import dataclass

from .v2 import FULL_PAY_TIERS, parse_full_pay_tiers

# The six suites (DEC-CA-0064 §Suites). A suite is the lane-C target
# ``mkt:<suite>``.
SUITES: tuple[str, ...] = ("returns", "vol", "range", "funding", "liquidity", "alpha")
# Recommended launch set (research 2026-10-19): returns + vol.
DEFAULT_SUITES: tuple[str, ...] = ("returns", "vol")
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
}
# Series sampling intervals per suite (what the base sees as one step).
DEFAULT_FREQUENCIES: dict[str, tuple[str, ...]] = {
    "returns": ("1m", "5m", "1h", "1d"),
    "vol": ("5m", "15m", "1h", "1d"),
    "range": ("1h", "4h", "1d"),
    "funding": ("8h",),
    "liquidity": ("1h", "1d"),
    "alpha": ("1h", "1d"),
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
}
# Recipe allow-lists. Covariates are INPUTS, never targets (DEC-CA-0064
# §Covariates); each has a builder in :func:`cascade.eval.marketcast.
# build_covariates`. ``open_interest`` and ``macro_events`` are NOT allowed:
# Hyperliquid serves no historical OI and no event calendar is pinned.
DEFAULT_COVARIATES: tuple[str, ...] = ("cross_asset", "funding", "volume", "calendar", "session")
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
# Coarse bank (review 2026-10-19): series at >= 1h frequencies are cut from
# hourly bars kept for ``coarse_history_days`` (Hyperliquid serves only the
# most recent ~5000 candles per interval — measured 2026-10-19: 1m ~3.6 days,
# 5m ~17 days, 15m ~52 days, 1h ~208 days, 4h ~833 days), so a daily series
# has enough history to be scored. Realized vol needs >= 2 returns inside a
# block, so it reads the coarse bank only at >= 1d (24 hourly returns a day).
COARSE_BAR_MINUTES = 60
COARSE_MIN_FREQ_MINUTES = 60
COARSE_MIN_FREQ_MINUTES_VOL = 1440

_DUR_RE = re.compile(r"^([1-9][0-9]{0,4})(m|h|d)$")
_DUR_MIN = {"m": 1, "h": 60, "d": 1440}
_LABEL_RE = re.compile(r"^[a-z0-9_\-]{1,32}$")
_ASSET_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,24}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RESERVED_POOL_PREFIXES = ("pool", "pool_c", "pool_d")


def duration_minutes(spec: str) -> int:
    """``"5m"`` / ``"4h"`` / ``"1d"`` → minutes. Raises ``ValueError``."""
    m = _DUR_RE.match(str(spec))
    if not m:
        raise ValueError(f"duration {spec!r} must be <int><m|h|d>, e.g. '5m', '4h', '1d'")
    return int(m.group(1)) * _DUR_MIN[m.group(2)]


def coarse_frequency(suite: str, freq_minutes: int) -> bool:
    """Whether ``suite`` at ``freq_minutes`` is cut from the snapshot's
    coarse (hourly) bank rather than its 1-minute bars."""
    lo = COARSE_MIN_FREQ_MINUTES_VOL if suite == "vol" else COARSE_MIN_FREQ_MINUTES
    return int(freq_minutes) >= lo


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
    # Gain-tiered full-pay period of an ``mkt:`` win (owner 2026-10-20), on
    # lane C's slot TTL and decay: the ``full_eras`` of the highest threshold
    # its confirmed gain LCB meets. Its OWN tiers, not [lane_c]'s: the
    # MarketCast LCB is a horizon-equal clustered geomean gain on one suite's
    # loss (CRPS-bp / QLIKE / log range …) vs the pinned base, not lane C's
    # cluster-bootstrapped sqrt(WQL · MASE) gain vs the king — the same number
    # is not the same evidence. Release-only like the rest of [marketcast].
    # Uncalibrated placeholders (lane B / C's defaults).
    full_pay_tiers: tuple[tuple[float, int], ...] = FULL_PAY_TIERS
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
    max_synthetic_fraction: float = 0.5
    max_lr: float = 1e-3
    max_steps: int = 100_000
    max_batch: int = 512
    max_lora_rank: int = 64
    # Window cutting / verdict timing (validator + audit).
    eval_context_length: int = 4096       # context steps both base and fine-tune see
    windows_per_cell: int = 48            # windows per (asset, horizon), spread evenly
    start_margin_seconds: int = 3600      # windows start this long after the train block
    verdict_wait_blocks: int = 43_200     # an entry waits for windows this long (6 days)
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

    def active(self, block: int | None) -> bool:
        """Whether MarketCast judges a settlement at ``block``."""
        return bool(self.from_block > 0 and block is not None and int(block) >= self.from_block)

    @property
    def base_pointer(self) -> str:
        """The identity MarketCast slots and pending entries are judged
        against (the lane-C ``base_pointer`` of an ``mkt:`` win): the pinned
        base, which never moves — so :func:`cascade.shared.lane_c.win_beats`
        always falls through to the LCB comparison."""
        return mkt_base_pointer(self)

    @property
    def target_suites(self) -> tuple[str, ...]:
        """The suites whose ``mkt:<suite>`` lane-C targets are admitted —
        none while MarketCast is inert."""
        return tuple(self.suites) if self.from_block > 0 else ()

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(f"mkt:{s}" for s in self.suites)

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

    def suite_assets(self, suite: str) -> list[tuple[str, str]]:
        """``[(class, symbol), ...]`` a suite is judged over (sorted)."""
        am = self.assets_map
        return sorted((c, a) for c in SUITE_CLASSES.get(suite, ()) for a in am.get(c, ()))

    @property
    def armed_classes(self) -> tuple[str, ...]:
        """The classes an enabled suite reads that list at least one asset."""
        am = self.assets_map
        return tuple(sorted({c for s in self.suites for c in SUITE_CLASSES.get(s, ())
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
    return out


def scorability_problems(mc: MarketCastConfig) -> list[str]:
    """Why a configured (suite, horizon) could never yield a window (empty =
    every one is scorable). Per enabled suite and horizon, at the frequency
    it is forecast at (:func:`cascade.eval.marketcast.window_geometry`):

    * the snapshot's history at that frequency — ``coarse_history_days``
      for a coarse-bank series, else ``snapshot_days`` — holds at least
      ``MIN_CONTEXT + steps`` blocks;
    * the forward wait ``verdict_wait_blocks × BLOCK_SECONDS −
      start_margin_seconds`` holds at least TWO non-overlapping windows at
      the worst grid alignment (window origins sit on the absolute UTC grid,
      so the first may start up to one horizon after ``after``: ``wait ≥ 3 ×
      horizon``)."""
    from ...eval.marketcast import window_geometry

    out: list[str] = []
    wait_s = int(mc.verdict_wait_blocks) * BLOCK_SECONDS - int(mc.start_margin_seconds)
    fm, hm = mc.frequencies_map, mc.horizons_map
    for s in mc.suites:
        for h in hm.get(s, ()):
            try:
                f, steps = window_geometry(fm.get(s, ()), h)
            except ValueError as e:
                out.append(f"{s}/{h}: {e}")
                continue
            days = mc.coarse_history_days if coarse_frequency(s, f) else mc.snapshot_days
            blocks = int(days) * 1440 // int(f)
            if blocks < MIN_CONTEXT + steps:
                out.append(f"{s}/{h} at {f}m: {days} days of history hold {blocks} blocks < "
                           f"{MIN_CONTEXT} context + {steps} target")
            hs = 60 * duration_minutes(h)
            if wait_s < 3 * hs:
                out.append(f"{s}/{h}: the forward wait ({wait_s:.0f}s = verdict_wait_blocks × "
                           f"{BLOCK_SECONDS:g} − start_margin_seconds) holds fewer than 2 aligned "
                           f"windows (needs >= {3 * hs}s)")
    tb_s = int(mc.min_time_blocks) * int(mc.cluster_block_hours) * 3600
    if wait_s < tb_s:
        out.append(f"the forward wait ({wait_s:.0f}s) is shorter than min_time_blocks × "
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


def parse_marketcast(raw: object) -> MarketCastConfig:
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
        if cls not in ASSET_CLASSES:
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
    if bad:
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
    # the TTL bound (lane C's win_ttl_eras) is checked by the loader
    fpt = parse_full_pay_tiers(mc.get("full_pay_tiers", FULL_PAY_TIERS), section="marketcast",
                               win_ttl_eras=0)
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
    syn = _num(mc.get("max_synthetic_fraction", 0.5), "max_synthetic_fraction")
    if not 0.0 <= syn <= 1.0:
        raise ValueError(f"[marketcast] max_synthetic_fraction must be in [0, 1]; got {syn}")
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
    wait = _num(mc.get("verdict_wait_blocks", 43_200), "verdict_wait_blocks", integer=True)
    sdays = _num(mc.get("snapshot_days", 21), "snapshot_days", integer=True)
    if wpc < 1 or marg < 0 or wait < 0 or sdays < 1:
        raise ValueError("[marketcast] need windows_per_cell >= 1, start_margin_seconds >= 0, "
                         "verdict_wait_blocks >= 0 and snapshot_days >= 1")
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
    from .training import _continuation_protocol
    cfg_out = MarketCastConfig(
        continuation_protocol=_continuation_protocol(mc.get("continuation_protocol", "")),
        max_context_length=mcl_ctx, covariates=covs, augmentations=augs, embargo_days=emb,
        max_synthetic_fraction=syn, max_lr=mlr, max_steps=msteps, max_batch=mbatch,
        max_lora_rank=mrank,
        from_block=fb, base_ref=base_ref, base_sha256=sha, base_arch=arch,
        quantile_knots=knots, suites=suites,
        assets=tuple(sorted(assets.items())),
        frequencies=tuple(sorted(freqs.items())),
        horizons=tuple(sorted(hzs.items())),
        data_sources=tuple(sorted(sources.items())),
        coverage_levels=cov, coverage_tolerance=tol, alpha=alpha, bootstrap_B=B,
        gain_margin=margin, full_pay_tiers=fpt, min_windows=mw, min_clusters=mcl, pool_prefix=pp,
        eval_context_length=ectx, windows_per_cell=wpc, start_margin_seconds=marg,
        verdict_wait_blocks=wait, snapshot_days=sdays, coarse_history_days=cdays,
        report_timeout_settlements=rto, max_infra_deferrals=mid,
        cluster_block_hours=cbh, min_time_blocks=mtb,
        min_bar_coverage=mbc, min_dollar_volume_24h=mdv, coverage_tolerance_over=tov,
        coverage_gate_levels=cgl, zero_scale_frac=zsf, cell_ratio_cap=cap, tail_levels=tls,
        tail_tolerance=ttol, calibration_z=cz, near_dup_tol=ndt,
        max_open_legs_per_coldkey=molc,
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
    "mkt_base_pointer", "scorability_problems",
]
