# Vendored and trimmed from TempoPFN (Apache-2.0). See repo-root NOTICE.
#
# Original carried a torch-based BatchTimeSeriesContainer used by the training
# pipeline; the cascade base generator only needs the numpy TimeSeriesContainer
# emitted by generate_batch(), so the torch dependency and the batch/history
# container are dropped here to keep the vendored surface minimal.
from dataclasses import dataclass

import numpy as np

from tempo_gen.data.frequency import Frequency


@dataclass
class TimeSeriesContainer:
    """
    Container for a batch of time series data without explicit history/future split.

    Attributes:
        values: np.ndarray of shape [batch_size, seq_len] (univariate) or
            [batch_size, seq_len, num_channels] (multivariate).
        start: List[np.datetime64], length == batch_size.
        frequency: List[Frequency], length == batch_size.
    """

    values: np.ndarray
    start: list[np.datetime64]
    frequency: list[Frequency]

    def __post_init__(self):
        if not isinstance(self.values, np.ndarray):
            raise TypeError("values must be a np.ndarray")
        if not isinstance(self.start, list) or not all(isinstance(x, np.datetime64) for x in self.start):
            raise TypeError("start must be a List[np.datetime64]")
        if not isinstance(self.frequency, list) or not all(isinstance(x, Frequency) for x in self.frequency):
            raise TypeError("frequency must be a List[Frequency]")

        if len(self.values.shape) < 2 or len(self.values.shape) > 3:
            raise ValueError(
                "values must have 2 or 3 dimensions "
                "[batch_size, seq_len] or [batch_size, seq_len, num_channels], "
                f"got shape {self.values.shape}"
            )

        batch_size = self.values.shape[0]
        if len(self.start) != batch_size:
            raise ValueError(f"Length of start ({len(self.start)}) must match batch_size ({batch_size})")
        if len(self.frequency) != batch_size:
            raise ValueError(f"Length of frequency ({len(self.frequency)}) must match batch_size ({batch_size})")

    @property
    def batch_size(self) -> int:
        return self.values.shape[0]

    @property
    def seq_length(self) -> int:
        return self.values.shape[1]

    @property
    def num_channels(self) -> int:
        return self.values.shape[2] if len(self.values.shape) == 3 else 1
