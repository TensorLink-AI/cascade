"""Lane B's own GPU list (WB22; LANE_D_PLAN §12b.6 D42 item 1, TB23 / TB24).

``[lane_b] funded_pod_skus`` is separate from ``[round] funded_pod_skus``: a
type listed for lane B never becomes a lane A type. Lane B legs choose (and
rent) only from it, and only a type with a measured expert-only step rate in
WB19's table (``evidence/wb19.json``, ``laned_wb19_table/1``) — a listed type
with no rate is refused at the intake with :data:`UNMEASURED` and never
rented. The table is trainer-side operational data: it is checked against
the pinned train image (a repin makes it stale) and never feeds consensus.

The rent path reads lane B's list through :func:`rent_scope`, a thread-local
on the runner, so lane A legs (other threads, no scope) are unchanged.

:func:`exact_rederive_ok` is the audit pin: a byte-exact re-derivation of a
lane B leg is valid only on the recorded GPU model AND the recorded GPU
count (``gpu_n``; a different count changes the reduction order, WB25).

Stdlib only: importing it never imports torch.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .gpu_choice import GpuCatalog, GpuChoice
from .step_rates import StepRate, StepRateTableError, load_step_rate_table, rate_for

log = logging.getLogger("cascade.funding")

__all__ = [
    "DEFAULT_STEP_RATES",
    "STEP_RATES_ENV",
    "UNMEASURED",
    "LaneBGpus",
    "exact_rederive_ok",
    "lane_b_gpus",
    "lane_b_gpus_from_env",
    "lane_b_image_pin",
    "load_lane_b_rates",
    "rent_scope",
    "step_rates_path",
]

STEP_RATES_ENV = "CASCADE_LANE_B_STEP_RATES"
DEFAULT_STEP_RATES = "evidence/wb19.json"
UNMEASURED = "gpu_rate_unmeasured"

_REPO = Path(__file__).resolve().parents[2]
_SHA = re.compile(r"sha256:[0-9a-f]{64}")


def step_rates_path(env=None) -> Path:
    """The step-rate table: ``$CASCADE_LANE_B_STEP_RATES``, else WB19's
    committed evidence (``<repo>/evidence/wb19.json``)."""
    env = os.environ if env is None else env
    p = str(env.get(STEP_RATES_ENV, "") or "").strip()
    return Path(p) if p else _REPO / DEFAULT_STEP_RATES


def lane_b_image_pin(cfg, block: int | None = None) -> str:
    """``sha256:<64>`` of the train image of the contract at ``block``
    (default: ``[v2] from_block``) — the image a lane B leg runs."""
    b = int(cfg.training.v2_from_block) if block is None else int(block)
    digest = str(cfg.throne_contracts_at(b)[0].train_image_digest or "")
    m = _SHA.search(digest)
    if m is None:
        raise ValueError(f"the contract at block {b} pins no sha256 train image "
                         f"({digest!r})")
    return m.group(0)


def load_lane_b_rates(path, *, image_digest: str) -> dict[str, StepRate]:
    """``{sku: StepRate}``; a missing file is ``{}`` (nothing measured).
    A malformed, failed or stale table raises :class:`StepRateTableError`."""
    p = Path(path)
    if not p.is_file():
        return {}
    return load_step_rate_table(p, image_digest=image_digest)


@dataclass(frozen=True)
class LaneBGpus:
    """Lane B's GPU types and their measured rates. ``rates`` None = no
    table wired (dev / unit construction only): the rate gate is off."""

    skus: tuple[str, ...]
    rates: dict | None
    default_max_price_per_hour: float = 0.0

    @property
    def catalog(self) -> GpuCatalog:
        """What a lane B fund's GPU headers are checked against."""
        return GpuCatalog(skus=tuple(self.skus), walls=(), wall_cap_seconds=0.0,
                          default_max_price_per_hour=float(self.default_max_price_per_hour))

    def rate(self, sku: str) -> StepRate | None:
        return None if self.rates is None else rate_for(self.rates, sku)

    def measured(self, sku: str) -> bool:
        return self.rates is None or self.rate(sku) is not None

    def rent_skus(self) -> tuple[str, ...]:
        """The types a lane B leg may rent: the measured ones, in list order."""
        return tuple(s for s in self.skus if self.measured(s))

    def refusal(self, choice: GpuChoice) -> dict | None:
        """The typed refusal of a fund whose leg could not rent, else None."""
        gpu = str(getattr(choice, "gpu", "") or "")
        if gpu and not self.measured(gpu):
            return {"code": UNMEASURED,
                    "message": f"{gpu} is in [lane_b] funded_pod_skus but has no measured "
                               "expert-only step rate (WB19) on the pinned image yet — "
                               "choose a measured type: "
                               + (", ".join(self.rent_skus()) or "none measured yet")}
        if not gpu and not self.rent_skus():
            return {"code": UNMEASURED,
                    "message": "no [lane_b] funded_pod_skus type has a measured "
                               "expert-only step rate (WB19) on the pinned image yet — "
                               "lane B legs cannot rent until one is measured"}
        return None


def lane_b_gpus(cfg, rates: dict | None) -> LaneBGpus:
    return LaneBGpus(skus=tuple(cfg.lane_b.funded_pod_skus or ()), rates=rates,
                     default_max_price_per_hour=float(
                         getattr(cfg.round, "funded_max_price_per_hour", 0.0) or 0.0))


def lane_b_gpus_from_env(cfg, *, env=None, block: int | None = None) -> LaneBGpus:
    """:func:`lane_b_gpus` with the table at :func:`step_rates_path`. A table
    that cannot be used is logged and fails closed (nothing measured)."""
    path = step_rates_path(env)
    try:
        rates = load_lane_b_rates(path, image_digest=lane_b_image_pin(cfg, block))
    except (StepRateTableError, ValueError) as e:
        log.error("lane B step-rate table %s unusable (%s): every lane B GPU type is "
                  "treated as unmeasured", path, e)
        rates = {}
    return lane_b_gpus(cfg, rates)


_UNSET = object()


@contextmanager
def rent_scope(runner, gpus: LaneBGpus, *, only: tuple[str, ...] = ()):
    """While the block runs on this thread, the runner's rent path
    (``runner_capacity``) rents from lane B's measured list only — or, with
    ``only`` (owner 2026-11-24: an own-risk GPU type the miner chose, no WB19
    rate), exactly those types."""
    loc = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    prev = getattr(loc, "lane_b_rent_skus", _UNSET)
    loc.lane_b_rent_skus = tuple(only) if only else tuple(gpus.rent_skus())
    try:
        yield loc
    finally:
        if prev is _UNSET:
            del loc.lane_b_rent_skus
        else:
            loc.lane_b_rent_skus = prev


def exact_rederive_ok(*, recorded_gpu_name: str, recorded_gpu_n: int,
                      local_gpu_name: str, local_gpu_n: int) -> bool:
    """May a byte-exact audit re-derivation of a lane B leg run here? Only on
    the recorded GPU model and the recorded GPU count. An unrecorded count
    (``gpu_n`` ≤ 0) is a pre-WB25 one-GPU leg."""
    want_n = int(recorded_gpu_n) if int(recorded_gpu_n) > 0 else 1
    return (bool(recorded_gpu_name) and recorded_gpu_name == local_gpu_name
            and int(local_gpu_n) == want_n)
