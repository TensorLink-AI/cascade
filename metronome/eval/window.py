"""EvalWindow — one held-out forecasting task.

Numpy-only by design: the eval/scoring path carries no torch dependency so the
statistics are unit-testable in a minimal environment. The validator's
evaluator (which loads the trained model and needs torch) converts model output
to numpy sample arrays and feeds them through :mod:`.scoring`.

A window is a single real-world series split into ``history`` (the context the
model sees) and ``target`` (the held-out continuation it must forecast). The
eval set is the *same* for the king's model and the challenger's model — that
shared, fixed, real-world set is what makes the KOTH comparison a controlled
measurement of data quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvalWindow:
    """One context/target split of a held-out series.

    History and target carry a leading channel axis so the eval set is
    multivariate-ready, matching the Toto2 backbone metronome trains. A 1-D
    ``(L,)`` array passed in is promoted to a single channel ``(1, L)``; today
    every window is univariate (``n_channels == 1``) and scoring runs per
    channel, so turning on multivariate eval later needs no container change.

    Attributes:
        series_id: stable identifier, used for logging and pairing.
        history: shape ``(C, L)`` — the context the model conditions on.
        target: shape ``(C, H)`` — the held-out continuation to forecast.
        metadata: free-form; ``freq`` / ``seasonal_period`` drive MASE
            seasonality (see :func:`metronome.eval.scoring`).
    """

    series_id: str
    history: np.ndarray
    target: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Promote 1-D (L,) → (1, L) so the container always carries channels.
        if self.history.ndim == 1:
            object.__setattr__(self, "history", self.history[None, :])
        if self.target.ndim == 1:
            object.__setattr__(self, "target", self.target[None, :])
        if self.history.ndim != 2:
            raise ValueError(f"history must be 1-D (L,) or 2-D (C, L); got {self.history.shape}")
        if self.target.ndim != 2:
            raise ValueError(f"target must be 1-D (H,) or 2-D (C, H); got {self.target.shape}")
        if self.history.shape[0] != self.target.shape[0]:
            raise ValueError(
                f"history/target channel mismatch: {self.history.shape[0]} vs "
                f"{self.target.shape[0]}"
            )

    @property
    def n_channels(self) -> int:
        return int(self.target.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.target.shape[-1])
