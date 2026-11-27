"""MarketCast fit check (2026-11-27): will a recipe's fine-tune fit the GPU?

A MarketCast leg fine-tunes Timestral-28M in ONE worker process on
``cuda:0`` (fp32, ``cascade.trainer.marketcast_leg.finetune`` over
``cascade.eval.timestral_adapter.forward_block``). A recipe may ask for
``batch`` <= 512, ``context_length`` <= 32768 and ``lora_rank`` <= 64, which
is far more than a 24 GB card holds: on 2026-11-27 a funded leg (batch 16,
context 4096, LoRA rank 32, ten covariates) died with ``torch.OutOfMemoryError``
on a 1xRTX4090 and the trainer re-rented the same type for the same
deterministic OOM. The trainer now ends such an entry (``faults.GPU_OOM_CLASS``);
this module lets the intake refuse it before anything is rented.

The estimate (derived from the adapter and the pinned library's layers):

* rows per sample ``V = 1 + Σ rows(covariate)`` (each past covariate one
  row; ``calendar`` 4, ``session`` 3, ``events`` 2 known rows —
  ``cascade.eval.marketcast.build_covariates``);
* samples per forward ``m = min(batch, max(1, micro_batch_tokens // context))``
  (the leg's gradient accumulation), so ``R = m · V`` packed rows;
* patches per row ``n = ceil(context / 32) + hp`` with ``hp`` the largest
  horizon's patch count; with covariates the library runs the encoder AND
  one forecast token per patch (``_predict_window``): ``S = 2n`` tokens per
  row, else ``S = n`` (``_predict_context_only``);
* saved activations, fp32, per token: 15 attention sublayers (9 ``time`` +
  3 ``time_variate`` layers × 2) at ``16·d + 3·r`` floats (norm, q/k/v and
  their RoPE halves, the attention output, the sigmoid gate, a rank-``r``
  LoRA intermediate on each of wQKV / gate / wO) and 12 SwiGLU MLPs at
  ``2·d + 4·h + 2·r`` (d = 384, h = 1024, ``r`` = the LoRA rank, 0 for the
  other scopes);
* the attention masks each SDPA call keeps as a float bias: per time
  attention ``R · 3n²`` (two-stream; ``R·(L² + hp·n)`` without covariates),
  per variate attention ``S · R²`` (it runs across every row of the batch);
* plus a fixed 1.5 GiB (CUDA context, weights, optimizer state), times a
  10 % margin on the activations.

Calibration (the 2026-11-27 OOM): the failed allocation was 312 MiB (the
allocator's 2 MiB rounding of a (tokens, 1152) fp32 wQKV LoRA output),
i.e. 70,543..70,997 tokens; the formula gives R·S = 272 · 260 = 70,720 at
that step (hp = 2) — the token count is exact. PyTorch held 22.15 GiB at
that point of the FORWARD pass; the formula puts that ~7.5 of the 12
layers in and the whole recipe at ~47 GiB with margin — over the 23.5 GiB
card, inside an 80 GB A100 / H100. The estimate is a planning bound, not a
measurement: it is deliberately conservative (no activation is assumed
freed early).

Stdlib only: the intake is torch-free.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "FIXED_GIB",
    "GPU_MEMORY_GIB",
    "MARGIN",
    "FitVerdict",
    "estimate_peak_gib",
    "fit_verdict",
    "gpu_memory_gib",
    "recipe_shape",
]

# Timestral-28M (config.json at the pinned commit; shared.config.marketcast)
D_MODEL = 384
MLP_HIDDEN = 1024
N_LAYERS = 12
N_ATTN_SUBLAYERS = 15          # 9 "time" + 3 "time_variate" x 2
N_TIME_ATTN = 12
N_VARIATE_ATTN = 3
PATCH = 32
BLOCK_PATCHES = 32             # one decode block (1024 steps)
FP32 = 4
KNOWN_ROWS = {"calendar": 4, "session": 3, "events": 2}
DEFAULT_MICRO_BATCH_TOKENS = 524_288
MARGIN = 1.10                  # on the activation estimate
FIXED_GIB = 1.5                # CUDA context + weights + optimizer state
GIB = float(1 << 30)

# Per-GPU memory, GiB, as ``torch.cuda`` reports the device's total
# capacity (the 2026-11-27 RTX 4090 said 23.52 GiB). Keys are the base type
# upper-cased with every non-alphanumeric dropped ("1xRTX4090" -> "RTX4090").
# A100 / A800 without a size are the 80 GB parts (the provision tables rent
# A100-SXM4-80GB); "...40GB" names are the 40 GB parts.
GPU_MEMORY_GIB: dict[str, float] = {
    "RTX3090": 23.6, "RTX3090TI": 23.6, "RTX4090": 23.5, "RTX4090D": 23.5,
    "RTX5090": 31.4, "RTX4080": 15.6, "RTX4080SUPER": 15.6, "RTX4070TI": 11.7,
    "RTXA4000": 15.7, "RTXA4500": 19.7, "RTXA5000": 23.6, "RTXA6000": 47.5, "A6000": 47.5,
    "RTX6000ADA": 47.5, "RTX6000": 47.5, "RTXPRO6000": 94.9, "L4": 22.0, "A10": 22.0,
    "A10G": 22.0, "A16": 14.6, "T4": 14.6, "V100": 15.8, "V10032GB": 31.7, "A30": 23.5,
    "A40": 44.4, "L40": 44.4, "L40S": 44.4, "A100": 79.2, "A10080GB": 79.2,
    "A100SXM": 79.2, "A100SXM4": 79.2, "A100PCIE": 79.2, "A10040GB": 39.4, "A800": 79.2,
    "H100": 79.1, "H100SXM": 79.1, "H100PCIE": 79.1, "H100NVL": 93.1, "H800": 79.1,
    "H200": 139.7, "H200NVL": 139.7, "GH200": 94.5, "B200": 178.4, "B300": 268.0,
}

_LAYOUT = re.compile(r"^\s*\d+\s*[xX×]\s*")


def gpu_memory_gib(gpu: str) -> float | None:
    """Per-GPU memory of a GPU type or layout (``"8xH100"`` -> one H100),
    None when the type is not in :data:`GPU_MEMORY_GIB`."""
    base = _LAYOUT.sub("", str(gpu or "")).upper()
    base = re.sub(r"^(NVIDIA|GEFORCE)", "", re.sub(r"[^A-Z0-9]", "", base))
    base = base.removeprefix("GEFORCE")
    return GPU_MEMORY_GIB.get(base)


@dataclass(frozen=True)
class RecipeShape:
    batch: int
    context: int
    lora_rank: int
    rows: int                  # V: target + covariate rows per sample
    horizon_patches: int


def _int(v, lo: int, hi: int) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        return None
    return v


def _horizon_steps(label: str, freqs: tuple[str, ...]) -> int:
    from ..shared.config.marketcast import duration_minutes, horizon_path

    p = horizon_path(label)
    if p is not None:
        return int(p[1])
    h = duration_minutes(label)
    fit = []
    for f in freqs:
        try:
            m = duration_minutes(f)
        except ValueError:
            continue
        if 0 < m <= h and h % m == 0:
            fit.append(m)
    return h // max(fit) if fit else h       # no frequency: 1-minute bars (conservative)


def recipe_shape(obj, *, horizons: Iterable[str] = ()) -> RecipeShape | None:
    """The memory-relevant shape of a ``marketcast.json`` object (read
    leniently: None when a field is missing or out of range — the trainer's
    full parse refuses such a recipe anyway). ``horizons`` adds the
    configured horizon labels the leg may also train (other weighted suites)."""
    if not isinstance(obj, Mapping):
        return None
    op = obj.get("optimizer")
    tr = obj.get("trainable")
    if not isinstance(op, Mapping) or not isinstance(tr, Mapping):
        return None
    batch = _int(op.get("batch"), 1, 1 << 20)
    ctx = _int(obj.get("context_length"), 1, 1 << 20)
    if batch is None or ctx is None:
        return None
    rank = 0
    if tr.get("scope") == "lora":
        rank = _int(tr.get("lora_rank"), 1, 1 << 12)
        if rank is None:
            return None
    covs = obj.get("covariates", [])
    if not isinstance(covs, list) or not all(isinstance(c, str) for c in covs):
        return None
    rows = 1 + sum(KNOWN_ROWS.get(c, 1) for c in set(covs))
    data = obj.get("data") if isinstance(obj.get("data"), Mapping) else {}
    freqs = tuple(str(f) for f in (data.get("frequencies") or ()) if isinstance(f, str))
    mix = obj.get("horizon_mix")
    labels = [str(k) for k in (mix.keys() if isinstance(mix, Mapping) else ())]
    labels += [str(h) for h in horizons]
    hp = 1
    for lab in labels:
        try:
            steps = _horizon_steps(lab, freqs)
        except ValueError:
            continue
        hp = max(hp, min(BLOCK_PATCHES, math.ceil(steps / PATCH)))
    return RecipeShape(batch, ctx, rank, rows, hp)


def estimate_peak_gib(shape: RecipeShape, *,
                      micro_batch_tokens: int = DEFAULT_MICRO_BATCH_TOKENS) -> float:
    """Estimated peak GPU memory (GiB) of one training step, with
    :data:`MARGIN` on the activations and :data:`FIXED_GIB` added."""
    m = min(shape.batch, max(1, int(micro_batch_tokens) // max(1, shape.context)))
    r_rows = m * shape.rows
    ll = math.ceil(shape.context / PATCH)
    n = ll + shape.horizon_patches
    cov = shape.rows > 1
    s = 2 * n if cov else n
    r = shape.lora_rank
    per_token = (N_ATTN_SUBLAYERS * (16 * D_MODEL + 3 * r)
                 + N_LAYERS * (2 * D_MODEL + 4 * MLP_HIDDEN + 2 * r))
    act = FP32 * r_rows * s * per_token
    time_bias = r_rows * (3 * n * n if cov else ll * ll + shape.horizon_patches * n)
    var_bias = s * r_rows * r_rows if cov else 0
    masks = FP32 * (N_TIME_ATTN * time_bias + N_VARIATE_ATTN * var_bias)
    return round((act + masks) * MARGIN / GIB + FIXED_GIB, 1)


@dataclass(frozen=True)
class FitVerdict:
    estimate_gib: float
    gpu: str                   # the type checked ("" = none chosen)
    gpu_gib: float | None      # its memory (None = unknown type)
    fits: bool | None          # None = unknown type
    too_small: tuple[str, ...]  # recommended types below the estimate
    fitting: tuple[str, ...]   # recommended types at or above it

    def message(self) -> str:
        fit = ", ".join(self.fitting) or "none of the recommended types"
        if self.gpu and self.gpu_gib is not None:
            where = f"{self.gpu} has {self.gpu_gib:g} GiB per GPU"
        else:
            where = ("without --gpu the leg may rent " + ", ".join(self.too_small)
                     + " (" + ", ".join(f"{gpu_memory_gib(g):g} GiB" for g in self.too_small)
                     + ")")
        return (f"your recipe needs an estimated {self.estimate_gib:g} GiB of GPU memory "
                f"(batch, context_length, covariates and LoRA rank; the leg trains on ONE "
                f"GPU) and {where}: it would run out of GPU memory, and an out-of-memory "
                f"leg ends the entry with no retry: reduce batch/context/LoRA rank or "
                f"choose a larger GPU ({fit}, or any Lium type with at least "
                f"{self.estimate_gib:g} GiB per GPU at your own risk)")


def fit_verdict(obj, gpu: str, recommended: Iterable[str], *, horizons: Iterable[str] = (),
                micro_batch_tokens: int = DEFAULT_MICRO_BATCH_TOKENS) -> FitVerdict | None:
    """The fit of recipe ``obj`` on ``gpu`` ("" = no choice: every
    ``recommended`` type the default rent may pick), None when the recipe's
    shape cannot be read."""
    shape = recipe_shape(obj, horizons=horizons)
    if shape is None:
        return None
    est = estimate_peak_gib(shape, micro_batch_tokens=micro_batch_tokens)
    rec = tuple(recommended)
    known = [(g, gpu_memory_gib(g)) for g in rec]
    small = tuple(g for g, mem in known if mem is not None and mem < est)
    fitting = tuple(g for g, mem in known if mem is not None and mem >= est)
    if gpu:
        mem = gpu_memory_gib(gpu)
        return FitVerdict(est, gpu, mem, None if mem is None else mem >= est, small, fitting)
    return FitVerdict(est, "", None, not small, small, fitting)
