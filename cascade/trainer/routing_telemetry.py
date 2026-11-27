"""MoE routing telemetry for the training log (observability only).

Per optimizer step the pyramid trainer already holds, for every MoE layer, the
step's integer routing counts (``micro_batch_pass`` returns them; they are the
same tensors ``update_bias`` consumes, summed over micro-batches and — on a
gang — over ranks). This module turns those counts into a handful of scalar
health numbers that ride the existing per-step log channel (the S3 LogSink, the
live wandb mirror and the progress file):

``routing/L{block}/entropy``            normalised load entropy H(f) / ln E, in [0, 1]
``routing/L{block}/entropy_nats``       H(f) in nats
``routing/L{block}/dead_experts``       experts that received no token this step
``routing/L{block}/starved_experts``    experts below 10% of the mean load
``routing/L{block}/load_max_over_mean`` hottest expert / mean load (1.0 = balanced)
``routing/L{block}/load_min_over_mean`` coldest expert / mean load
``routing/L{block}/load_max_over_min``  hottest / coldest (omitted while one is dead)
``routing/L{block}/load_p10|p50|p90_over_mean``  load-fraction quantiles / mean
``routing/L{block}/aux_loss``           the layer's router auxiliary loss, only when
                                        the forward left one on ``last_aux``
``routing/L{block}/bias_absmax``        |route_bias| max (loss-free balancing state)

plus cross-layer summaries under ``routing/`` (``dead_experts``, ``dead_frac``,
``entropy_min``, ``entropy_mean``, ``load_max_over_mean``, ``aux_loss_mean``).

The per-expert fraction vector itself is summarised as quantiles rather than
logged expert by expert: the 120M preset has 2,688 routed experts, far too many
scalar series for a dashboard. Per-token routing entropy needs the router's
score matrix, which the forward does not retain; it is deliberately not
recomputed here (that would add work to the locked forward).

Numerics contract: everything here READS tensors the training step already
produced. It runs after the optimizer step, under ``torch.no_grad()``, on
``.detach()``-ed copies moved to the host — no RNG draw, no collective, no
write to any tensor the training graph uses. Any failure warns ONCE and turns
the telemetry off for the rest of the run; it can never abort training.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

log = logging.getLogger("cascade.trainer.routing")

# Same cadence as the pyramid trainer's per-step log line (step 1, then every
# 50 steps). ``[telemetry] routing_every_steps`` overrides it; 0 disables.
ROUTING_EVERY_DEFAULT = 50
_STARVED_FRACTION_OF_MEAN = 0.1


def _host_list(values) -> list[float]:
    """A detached host copy of a 1-D count vector (tensor or sequence)."""
    if hasattr(values, "detach"):
        values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        values = values.reshape(-1).tolist()
    return [float(v) for v in values]


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted, non-empty list."""
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def layer_routing_stats(counts) -> dict:
    """Health scalars for ONE layer's routing counts (length E >= 1).

    An all-zero vector (no tokens routed) yields ``dead_experts == E`` and no
    ratio fields. Pure: returns plain Python floats/ints only."""
    c = _host_list(counts)
    E = len(c)
    if E == 0:
        return {}
    total = sum(c)
    dead = sum(1 for v in c if v <= 0)
    out: dict = {"dead_experts": dead, "experts": E}
    if total <= 0:
        return out
    mean = total / E
    f = [v / total for v in c]
    h = max(0.0, -sum(p * math.log(p) for p in f if p > 0))
    out["entropy_nats"] = h
    out["entropy"] = h / math.log(E) if E > 1 else 1.0
    out["starved_experts"] = sum(1 for v in c if v < _STARVED_FRACTION_OF_MEAN * mean)
    hi, lo = max(c), min(c)
    out["load_max_over_mean"] = hi / mean
    out["load_min_over_mean"] = lo / mean
    if lo > 0:
        out["load_max_over_min"] = hi / lo
    rel = sorted(v / mean for v in c)
    for q, name in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
        out[f"load_{name}_over_mean"] = _quantile(rel, q)
    return out


def _scalar(t) -> float | None:
    """``float(t)`` for a detached scalar tensor/number; ``None`` if absent."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach()
        if hasattr(t, "numel") and t.numel() != 1:
            return None
        t = t.float().cpu().item() if hasattr(t, "cpu") else t.item()
    v = float(t)
    return v if math.isfinite(v) else None


def routing_record(loads: Sequence, *, block_ids: Sequence[int] | None = None,
                   aux: Sequence | None = None, bias: Sequence | None = None,
                   prefix: str = "routing") -> dict:
    """Flat ``{prefix/L{block}/<stat>: scalar}`` dict over every MoE layer,
    plus cross-layer summaries under ``{prefix}/``. ``loads[i]`` is layer i's
    count vector; ``block_ids[i]`` its block index (defaults to ``i``);
    ``aux[i]`` / ``bias[i]`` the layer's aux-loss scalar / route-bias vector
    (either may be ``None``). Pure: plain scalars out, inputs untouched."""
    ids = list(block_ids) if block_ids is not None else list(range(len(loads)))
    if len(ids) != len(loads):
        raise ValueError("block_ids must match loads")
    out: dict = {}
    entropies: list[float] = []
    aux_vals: list[float] = []
    dead_total = experts_total = 0
    worst = 0.0
    for i, (block, load) in enumerate(zip(ids, loads, strict=True)):
        stats = layer_routing_stats(load)
        if not stats:
            continue
        key = f"{prefix}/L{int(block)}"
        experts_total += stats.pop("experts")
        dead_total += stats["dead_experts"]
        if "entropy" in stats:
            entropies.append(stats["entropy"])
        worst = max(worst, stats.get("load_max_over_mean", 0.0))
        for name, value in stats.items():
            out[f"{key}/{name}"] = value
        a = _scalar(aux[i]) if aux is not None and i < len(aux) else None
        if a is not None:
            out[f"{key}/aux_loss"] = a
            aux_vals.append(a)
        if bias is not None and i < len(bias) and bias[i] is not None:
            b = _host_list(bias[i])
            if b:
                out[f"{key}/bias_absmax"] = max(abs(v) for v in b)
    if experts_total:
        out[f"{prefix}/dead_experts"] = dead_total
        out[f"{prefix}/dead_frac"] = dead_total / experts_total
        out[f"{prefix}/load_max_over_mean"] = worst
    if entropies:
        out[f"{prefix}/entropy_min"] = min(entropies)
        out[f"{prefix}/entropy_mean"] = sum(entropies) / len(entropies)
    if aux_vals:
        out[f"{prefix}/aux_loss_mean"] = sum(aux_vals) / len(aux_vals)
    return out


class RoutingTelemetry:
    """Cadence + failure guard around :func:`routing_record` for one run.

    ``every`` <= 0 disables it. :meth:`emit` never raises: the first failure
    is logged as a warning and the telemetry switches itself off."""

    def __init__(self, every: int = ROUTING_EVERY_DEFAULT):
        try:
            self.every = max(0, int(every))
        except (TypeError, ValueError):
            self.every = ROUTING_EVERY_DEFAULT
        self._failed = False

    def due(self, step: int) -> bool:
        return (self.every > 0 and not self._failed
                and (step == 1 or step % self.every == 0))

    def emit(self, logger, *, step: int, tokens: int, model, loads) -> None:
        """Log one ``{"event": "routing", ...}`` record when ``step`` is due.
        ``model`` exposes ``moe_layers()`` and ``cfg.moe_blocks`` (the pyramid
        model); ``loads`` is the step's per-layer count list."""
        if logger is None or not self.due(step):
            return
        try:
            import torch

            with torch.no_grad():
                layers = list(model.moe_layers())
                blocks = list(getattr(model.cfg, "moe_blocks", range(len(layers))))
                if len(blocks) != len(layers):
                    blocks = list(range(len(layers)))
                rec = routing_record(
                    list(loads), block_ids=blocks,
                    aux=[getattr(layer, "last_aux", None) for layer in layers],
                    bias=[getattr(layer, "route_bias", None) for layer in layers])
            if rec:
                logger({"event": "routing", "step": int(step), "tokens": int(tokens), **rec})
        except Exception as e:  # noqa: BLE001 — telemetry must never abort training
            self._failed = True
            log.warning("routing telemetry disabled for this run after an error: %s", e)
