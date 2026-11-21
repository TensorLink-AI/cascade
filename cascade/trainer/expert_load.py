"""Log-only routed-expert load telemetry for the cascade_pyramid trainer.

The training loop already computes, every optimizer step, each MoE layer's
global per-expert assignment counts (the ``load`` the loss-free balancing
bias consumes) — and then discards them. This module accumulates those counts
so a leg's log says whether routing is balanced *on the training stream*:

* per 50-step window (on the ``step`` log line), and
* per phase — trunk (constant LR) vs cooldown (the WSD decay) — in ``done``.

Per layer it reports ``eff`` (effective experts = exp of the load entropy;
E when perfectly balanced, 1 when collapsed), ``dead`` (experts with zero
assignments over the window) and ``max_mean`` (busiest expert's load over the
mean). 2026-11-21: the king's routing probe found ~28 effective of 512 with
>200 dead on probe inputs; this separates a training-time balancing failure
from a train→eval distribution shift.

Strictly read-only: it never touches the model, the bias, RNG or any tensor
training reads. Accumulation stays on the counts' device (no per-step host
sync); a summary is computed only when a window or phase is read.
"""

from __future__ import annotations

import math

import torch


def load_stats(counts: torch.Tensor) -> dict:
    """``{"eff", "dead", "max_mean"}`` for one layer's per-expert counts."""
    c = counts.detach().to("cpu", torch.float64)
    total = float(c.sum())
    n = int(c.numel())
    if total <= 0 or n == 0:
        return {"eff": 0.0, "dead": n, "max_mean": 0.0}
    p = c / total
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum())
    return {"eff": round(math.exp(entropy), 2), "dead": int((c == 0).sum()),
            "max_mean": round(float(c.max()) / (total / n), 2)}


class ExpertLoadTelemetry:
    """Accumulates per-layer expert counts by window and by WSD phase."""

    def __init__(self, trunk_steps: int):
        self.trunk_steps = int(trunk_steps)
        self._window: list[torch.Tensor] | None = None
        self._phase: dict[str, list[torch.Tensor]] = {}

    @staticmethod
    def _add(acc: list[torch.Tensor] | None, load) -> list[torch.Tensor]:
        loads = [t.detach() for t in load]
        if acc is None:
            return [t.to(torch.int64, copy=True) for t in loads]
        for a, t in zip(acc, loads, strict=True):
            a += t.to(torch.int64)
        return acc

    def record(self, step: int, load) -> None:
        """``step``: the 0-based step these counts belong to; ``load``: one
        count tensor per MoE layer (as returned by ``micro_batch_pass``)."""
        if not load:
            return
        phase = "trunk" if step < self.trunk_steps else "cooldown"
        self._window = self._add(self._window, load)
        self._phase[phase] = self._add(self._phase.get(phase), load)

    def window(self) -> list[dict] | None:
        """Stats since the previous call (then resets), or ``None``."""
        if self._window is None:
            return None
        out = [load_stats(c) for c in self._window]
        self._window = None
        return out

    def summary(self) -> dict | None:
        """``{"trunk": [...per layer], "cooldown": [...]}`` for ``done``."""
        if not self._phase:
            return None
        return {ph: [load_stats(c) for c in acc] for ph, acc in self._phase.items()}
