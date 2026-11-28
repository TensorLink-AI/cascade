"""One pre-rent feasibility check for every paid leg (retro 2026-11-28 #5).

Every paid step (a lane A leg, a lane B expert leg or judge pod, a lane C
MarketCast fine-tune) used to find its own infeasibility AFTER money was
spent: a recipe that cannot fit the GPU (#989, 2026-11-27 RTX 4090 OOM), a
data range that cannot cut a sample (#990, 30 min of paid CPU prep), a GPU
whose architecture the pinned torch has no kernels for (2026-11-28 01:44:
``CUDA error: no kernel image is available`` on a Blackwell sm_120 judge pod
under torch 2.4.1+cu124). This module is the ONE place those checks live:

* :func:`gpu_arch_refusal` — the GPU's compute capability vs the pinned
  torch build's CUDA arch list (:data:`TORCH_CUDA_ARCH`, recorded data: the
  box never imports torch);
* :func:`mkt_vram_refusal` — a MarketCast recipe's memory estimate
  (:mod:`cascade.funding.mkt_fit`) vs the GPU;
* :func:`lane_b_vram_refusal` — a lane B expert leg's context vs the GPU's
  measured memory (WB19 step-rate table, ``lane_b_levers.context_fits``);
* :func:`geometry_refusal` — a MarketCast recipe's data range vs the
  recorded bar history (:mod:`cascade.funding.mkt_geom`);
* :func:`image_refusal` / :func:`walls_refusal` — the pinned image is
  qualified and its walls measured (the reasons come from
  ``leg_pins.after_image_refusal`` / ``cost_guards.measured_wall_refusal``);
* :func:`disk_refusal` — free disk on the target vs the pull plus headroom.

:func:`feasibility` runs every applicable check over one :class:`LegPlan`
and returns the first :class:`Refusal` (or None). A refusal carries a typed
``code``, the miner-visible ``message`` and its ``actor``: ``"miner"`` (the
miner's own choice cannot work — the entry is released unburned, nothing
rented, fix and fund again) or ``"operator"`` (our config / image / disk —
requeued unburned with a backoff, no cost to the miner). The intake calls it
at fund time (free refusal) and the trainer again right before each rent.

Stdlib only: the intake and the box are torch-free.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

__all__ = [
    "GPU_COMPUTE_CAPABILITY",
    "INFEASIBLE_CLASS",
    "PINNED_TORCH",
    "TORCH_CUDA_ARCH",
    "LegPlan",
    "Refusal",
    "arch_supported",
    "disk_refusal",
    "failure_message",
    "feasibility",
    "geometry_refusal",
    "gpu_arch_refusal",
    "gpu_compute_capability",
    "image_refusal",
    "lane_b_vram_refusal",
    "mkt_vram_refusal",
    "parse_compute_cap",
    "public_text",
    "walls_refusal",
]

# The queue error class of a MINER-side refusal recorded at rent time:
# miner_fault (the entry is released, like ``auth``), never burned, nothing
# rented. An operator-side refusal is recorded as ``operator_blocked``.
INFEASIBLE_CLASS = "infeasible"
FAILURE_PREFIX = "feasibility"

# ── GPU architecture vs the pinned torch ────────────────────────────────────

# The pinned torch build (uv.lock ``torch 2.4.1+cu124``; pyproject
# ``torch==2.4.1`` from the cu124 index). The reproducibility contract: it is
# never bumped to fit a GPU — the GPU is chosen to fit it.
PINNED_TORCH = "2.4.1+cu124"

# ``torch.cuda.get_arch_list()`` of each pinned build, recorded where the
# image is built (``python -c "import torch; print(torch.cuda.get_arch_list())"``
# on the pod — never on the box). Cubins only: the 2.4.1 cu124 wheel ships
# no PTX a newer major could JIT, which is why sm_120 failed with "no kernel
# image" instead of running slowly. A repin of torch MUST add its list here
# (the accept test pins uv.lock's torch version to a key of this table).
TORCH_CUDA_ARCH: dict[str, tuple[str, ...]] = {
    "2.4.1+cu124": ("sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"),
}

# Compute capability (major, minor) per GPU base type; keys normalised like
# :func:`cascade.funding.mkt_fit.gpu_memory_gib` ("8xH100" -> "H100").
GPU_COMPUTE_CAPABILITY: dict[str, tuple[int, int]] = {
    # Volta / Turing
    "V100": (7, 0), "V10032GB": (7, 0), "T4": (7, 5),
    # Ampere
    "A100": (8, 0), "A10080GB": (8, 0), "A100SXM": (8, 0), "A100SXM4": (8, 0),
    "A100PCIE": (8, 0), "A10040GB": (8, 0), "A800": (8, 0), "A30": (8, 0),
    "A10": (8, 6), "A10G": (8, 6), "A16": (8, 6), "A40": (8, 6), "RTXA4000": (8, 6),
    "RTXA4500": (8, 6), "RTXA5000": (8, 6), "RTXA6000": (8, 6), "A6000": (8, 6),
    "RTX3090": (8, 6), "RTX3090TI": (8, 6),
    # Ada
    "L4": (8, 9), "L40": (8, 9), "L40S": (8, 9), "RTX4090": (8, 9), "RTX4090D": (8, 9),
    "RTX4080": (8, 9), "RTX4080SUPER": (8, 9), "RTX4070TI": (8, 9), "RTX6000ADA": (8, 9),
    "RTX6000": (8, 9),
    # Hopper
    "H100": (9, 0), "H100SXM": (9, 0), "H100PCIE": (9, 0), "H100NVL": (9, 0),
    "H800": (9, 0), "H200": (9, 0), "H200NVL": (9, 0), "GH200": (9, 0),
    # Blackwell (none runs under torch 2.4.1+cu124)
    "B200": (10, 0), "GB200": (10, 0), "B300": (10, 3), "RTX5090": (12, 0),
    "RTX5080": (12, 0), "RTXPRO6000": (12, 0),
}

_LAYOUT = re.compile(r"^\s*\d+\s*[xX×]\s*")
_SM = re.compile(r"^sm_(\d+)$")


def _base_type(gpu: str) -> str:
    base = _LAYOUT.sub("", str(gpu or "")).upper()
    base = re.sub(r"^(NVIDIA|GEFORCE)", "", re.sub(r"[^A-Z0-9]", "", base))
    return base.removeprefix("GEFORCE")


def gpu_compute_capability(gpu: str) -> tuple[int, int] | None:
    """``(major, minor)`` of a GPU type or layout, None when unknown."""
    return GPU_COMPUTE_CAPABILITY.get(_base_type(gpu))


def parse_compute_cap(text: str) -> tuple[int, int] | None:
    """``nvidia-smi --query-gpu=compute_cap`` output ("12.0") -> (12, 0)."""
    lines = str(text or "").strip().splitlines()
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", lines[0]) if lines else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def _sm(arch: str) -> tuple[int, int] | None:
    m = _SM.match(str(arch).strip())
    if not m:
        return None
    n = m.group(1)
    return int(n[:-1]), int(n[-1])


def arch_supported(cap: tuple[int, int], arch_list: Iterable[str]) -> bool:
    """A device runs a build when the build ships a cubin of the SAME major
    and a minor at or below the device's (CUDA binary compatibility: sm_86
    code runs on sm_89; nothing of major 9 runs on major 10 or 12)."""
    major, minor = cap
    for a in arch_list:
        sm = _sm(a)
        if sm is not None and sm[0] == major and sm[1] <= minor:
            return True
    return False


# ── the refusal ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Refusal:
    code: str                 # typed refusal code (the intake's 400 ``code``)
    message: str              # miner-visible, actionable
    check: str                # which check refused (arch / vram / geometry / ...)
    actor: str                # "miner" (their choice cannot work) | "operator"
    detail: dict = field(default_factory=dict)

    def as_body(self) -> dict:
        """The intake's refusal body (``code`` + ``message`` + detail)."""
        return {**self.detail, "code": self.code, "message": self.message}

    @property
    def miner_side(self) -> bool:
        return self.actor == "miner"


def failure_message(r: Refusal) -> str:
    """The rent-time verdict text (``feasibility <code>: <message>``) the
    queue stores; :func:`public_text` maps it back by ``code`` only."""
    return f"{FAILURE_PREFIX} {r.code}: {r.message}"


GPU_ARCH_UNSUPPORTED = "gpu_arch_unsupported"
NO_SUPPORTED_GPU = "no_supported_gpu"
GPU_MEMORY = "recipe_exceeds_gpu_memory"
CONTEXT_UNFIT = "context_length_unfit"
DATA_RANGE = "recipe_data_range_uncuttable"
IMAGE_UNQUALIFIED = "image_unqualified"
WALLS_UNMEASURED = "walls_unmeasured"
DISK_SHORT = "target_disk_short"

# Miner-visible text per code (the #996 whitelist rule: a template, never the
# trainer's prose). ``public_reason`` uses it for ``infeasible`` and for an
# ``operator_blocked`` verdict whose text starts ``feasibility <code>:``.
_PUBLIC = {
    GPU_ARCH_UNSUPPORTED: "refused before renting: the GPU type you chose cannot run the "
                          "pinned torch build (no kernels for its architecture): choose "
                          "another --gpu and fund again (nothing rented, no attempt charged)",
    GPU_MEMORY: "refused before renting: your recipe does not fit the GPU's memory: reduce "
                "batch/context/LoRA rank or choose a larger GPU and fund again (nothing "
                "rented, no attempt charged)",
    CONTEXT_UNFIT: "refused before renting: context_length does not fit the chosen GPU's "
                   "measured memory: choose a larger GPU and fund again (nothing rented)",
    DATA_RANGE: "refused before renting: your recipe's data range cannot cut its training "
                "samples: move data.start later (`cascade lane-c data` shows the recorded "
                "history) and fund a new ref (nothing rented)",
    NO_SUPPORTED_GPU: "waiting: our side failed a pre-rent check (no GPU type in the "
                      "operator's list runs the pinned torch); nothing rented, no cost to you",
    IMAGE_UNQUALIFIED: "waiting: our side failed a pre-rent check (the worker image is not "
                       "qualified yet); nothing rented, no cost to you",
    WALLS_UNMEASURED: "waiting: our side failed a pre-rent check (no measured leg wall for "
                      "the pinned image); nothing rented, no cost to you",
    DISK_SHORT: "waiting: our side failed a pre-rent check (not enough disk on the target "
                "host); nothing rented, no cost to you",
}
_OPERATOR_GENERIC = ("waiting: our side failed a pre-rent check; nothing rented, no cost "
                     "to you")
_MINER_GENERIC = ("refused before renting: this leg cannot run as funded; fix it and fund "
                  "again (nothing rented, no attempt charged)")
_FAILURE_RE = re.compile(r"^\s*" + FAILURE_PREFIX + r" ([a-z_]{1,48}):")


def public_text(error: str, *, miner_side: bool) -> tuple[str, str] | None:
    """``(code, text)`` for a stored rent-time feasibility verdict, None when
    ``error`` is not one. Only the code is read from the text."""
    m = _FAILURE_RE.match(error or "")
    if not m:
        return None
    code = m.group(1)
    if code in _PUBLIC:
        return code, _PUBLIC[code]
    return ("infeasible", _MINER_GENERIC) if miner_side else ("feasibility_operator",
                                                              _OPERATOR_GENERIC)


# ── the checks ──────────────────────────────────────────────────────────────


def gpu_arch_refusal(gpu: str = "", *, candidates: Iterable[str] = (),
                     compute_cap: tuple[int, int] | None = None, chosen: bool = True,
                     torch_pin: str = PINNED_TORCH,
                     arch_list: Iterable[str] | None = None) -> Refusal | None:
    """The arch check. ``gpu`` with ``chosen`` = the miner's own choice
    (refused as a MINER refusal); ``chosen=False`` with ``candidates`` = the
    operator's default list: refused (OPERATOR) only when NO candidate runs
    the build. ``compute_cap`` (an observed ``nvidia-smi`` value) overrides
    the type table; an unknown type is not refused (the pod-side check sees
    the real device). ``arch_list`` (image metadata) overrides
    :data:`TORCH_CUDA_ARCH`."""
    archs = tuple(arch_list) if arch_list is not None else TORCH_CUDA_ARCH.get(torch_pin)
    if not archs:
        return None
    if chosen and (gpu or compute_cap is not None):
        cap = compute_cap if compute_cap is not None else gpu_compute_capability(gpu)
        if cap is None or arch_supported(cap, archs):
            return None
        what = gpu or "this GPU"
        return Refusal(
            GPU_ARCH_UNSUPPORTED,
            f"{what} is compute capability sm_{cap[0]}{cap[1]}, and the pinned torch "
            f"{torch_pin} ships kernels for {', '.join(archs)} only (\"no kernel image is "
            "available\"): choose another GPU type (Ampere / Ada / Hopper, e.g. A100, L40S, "
            "RTX4090, H100)", "arch", "miner",
            {"gpu": gpu, "compute_capability": f"{cap[0]}.{cap[1]}", "torch": torch_pin,
             "arch_list": list(archs)})
    cands = tuple(candidates)
    if not cands:
        return None
    caps = [gpu_compute_capability(g) for g in cands]
    if any(c is None or arch_supported(c, archs) for c in caps):
        return None
    return Refusal(
        NO_SUPPORTED_GPU,
        f"none of the operator's GPU types ({', '.join(cands)}) runs the pinned torch "
        f"{torch_pin} ({', '.join(archs)})", "arch", "operator",
        {"gpus": list(cands), "torch": torch_pin})


def mkt_vram_refusal(recipe, gpu: str, recommended: Iterable[str], *,
                     horizons: Iterable[str] = (), micro_batch_tokens: int | None = None
                     ) -> Refusal | None:
    """#989's memory fit as a refusal (MINER side: the recipe is theirs).
    ``gpu`` "" = every recommended type the default rent may pick."""
    from .mkt_fit import DEFAULT_MICRO_BATCH_TOKENS, fit_verdict

    v = fit_verdict(recipe, gpu, tuple(recommended), horizons=horizons,
                    micro_batch_tokens=int(micro_batch_tokens or DEFAULT_MICRO_BATCH_TOKENS))
    if v is None or v.fits is not False:
        return None
    return Refusal(GPU_MEMORY, v.message(), "vram", "miner",
                   {"estimate_gib": v.estimate_gib, "gpu": v.gpu, "gpu_gib": v.gpu_gib,
                    "fitting_gpus": list(v.fitting)})


def lane_b_vram_refusal(rate, context_length: int, *, own_risk: bool = False
                        ) -> Refusal | None:
    """A lane B expert leg's memory: the default 4096 context is what every
    measured GPU was measured at (WB19), so it always fits; a longer context
    must fit the chosen type's MEASURED memory (``lane_b_levers.context_fits``:
    peak scaled by the context, under ``MEMORY_HEADROOM``). An own-risk
    (unmeasured) type runs at the miner's risk and is not refused."""
    from ..shared import lane_b_levers as lv

    if int(context_length) == lv.DEFAULT_CONTEXT or own_risk:
        return None
    if rate is not None and lv.context_fits(rate, int(context_length)):
        return None
    return Refusal(CONTEXT_UNFIT, f"context_length {int(context_length)} does not fit the "
                   "chosen GPU's measured memory (choose a GPU)", "vram", "miner",
                   {"context_length": int(context_length)})


def geometry_refusal(recipe, mc, index: Mapping | None) -> Refusal | None:
    """#990's sample geometry as a refusal (MINER side)."""
    if not index:
        return None
    from .mkt_geom import geom_verdict

    v = geom_verdict(recipe, mc, index)
    if v is None or not v.refused:
        return None
    return Refusal(DATA_RANGE, v.message(), "geometry", "miner",
                   {"risk": v.risk, "suite": v.suite, "horizon": v.horizon,
                    "timeframe": v.bar, "history_start": v.bar_first,
                    "fits_from_start": v.fix_start})


def image_refusal(reason: str) -> Refusal | None:
    """The image-qualification reason (``leg_pins.after_image_refusal``)."""
    return (Refusal(IMAGE_UNQUALIFIED, str(reason), "image", "operator") if reason
            else None)


def walls_refusal(reason: str) -> Refusal | None:
    """The measured-wall reason (``cost_guards.measured_wall_refusal``)."""
    return (Refusal(WALLS_UNMEASURED, str(reason), "walls", "operator") if reason
            else None)


def disk_refusal(free_bytes: int | None, need_bytes: int | None, *,
                 headroom_bytes: int = 0, where: str = "target") -> Refusal | None:
    """Free disk on the target vs the pull plus headroom (OPERATOR side:
    hosts and disks are ours). Unknown free / need: not refused."""
    if free_bytes is None or need_bytes is None:
        return None
    need = int(need_bytes) + int(headroom_bytes)
    if int(free_bytes) >= need:
        return None
    return Refusal(DISK_SHORT, f"{where} has {int(free_bytes) / 1e9:.1f} GB free and the "
                   f"step needs {need / 1e9:.1f} GB (pull {int(need_bytes) / 1e9:.1f} GB + "
                   f"headroom {int(headroom_bytes) / 1e9:.1f} GB)", "disk", "operator",
                   {"free_bytes": int(free_bytes), "need_bytes": need})


# ── one entry for a whole leg ───────────────────────────────────────────────


@dataclass(frozen=True)
class LegPlan:
    """Everything a paid leg's checks read; a field left at its default skips
    that check. ``lane``: ``"a"`` | ``"b"`` | ``"c"`` (informational)."""

    lane: str = "a"
    gpu: str = ""                          # the miner-chosen type ("" = default list)
    candidates: tuple[str, ...] = ()       # the default rent list (no choice)
    compute_cap: tuple[int, int] | None = None
    torch_pin: str = PINNED_TORCH
    arch_list: tuple[str, ...] | None = None
    image_reason: str = ""                 # leg_pins.after_image_refusal
    walls_reason: str = ""                 # cost_guards.measured_wall_refusal
    mkt_recipe: object = None              # a marketcast.json object (lane C)
    mkt_cfg: object = None
    mkt_history: Mapping | None = None
    mkt_horizons: tuple[str, ...] = ()
    micro_batch_tokens: int | None = None
    lane_b_rate: object = None
    lane_b_context: int | None = None
    lane_b_own_risk: bool = False
    disk_free_bytes: int | None = None
    disk_need_bytes: int | None = None
    disk_headroom_bytes: int = 0


def refusals(plan: LegPlan) -> list[Refusal]:
    """Every refusal of ``plan`` (operator-side first: ours to fix before a
    miner is told anything about their choice)."""
    out: list[Refusal | None] = [
        image_refusal(plan.image_reason),
        walls_refusal(plan.walls_reason),
        disk_refusal(plan.disk_free_bytes, plan.disk_need_bytes,
                     headroom_bytes=plan.disk_headroom_bytes),
        gpu_arch_refusal(plan.gpu, candidates=plan.candidates, compute_cap=plan.compute_cap,
                         chosen=bool(plan.gpu or plan.compute_cap is not None),
                         torch_pin=plan.torch_pin, arch_list=plan.arch_list),
    ]
    if plan.mkt_recipe is not None:
        out.append(mkt_vram_refusal(plan.mkt_recipe, plan.gpu, plan.candidates,
                                    horizons=plan.mkt_horizons,
                                    micro_batch_tokens=plan.micro_batch_tokens))
        if plan.mkt_cfg is not None:
            out.append(geometry_refusal(plan.mkt_recipe, plan.mkt_cfg, plan.mkt_history))
    if plan.lane_b_context is not None:
        out.append(lane_b_vram_refusal(plan.lane_b_rate, plan.lane_b_context,
                                       own_risk=plan.lane_b_own_risk))
    return [r for r in out if r is not None]


def feasibility(plan: LegPlan) -> Refusal | None:
    """The first refusal of ``plan``, None when the leg may be paid for."""
    got = refusals(plan)
    return got[0] if got else None
