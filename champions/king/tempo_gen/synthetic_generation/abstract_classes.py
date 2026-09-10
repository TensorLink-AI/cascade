# Vendored from TempoPFN (Apache-2.0). See repo-root NOTICE.
#
# DETERMINISM FIX (cascade): the upstream ``_set_random_seeds`` derived a
# per-generator offset with the builtin ``hash(self.__class__.__name__)``, whose
# value is randomised per process via PYTHONHASHSEED — so two audit runs in
# separate processes produced DIFFERENT corpora. cascade requires the corpus to
# be a pure function of (seed, n_series), audited across processes. We replace the
# builtin hash with ``zlib.crc32`` (a stable, process-independent checksum).
import zlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from tempo_gen.data.containers import TimeSeriesContainer
from tempo_gen.data.frequency import (
    select_safe_random_frequency,
    select_safe_start_date,
)
from tempo_gen.synthetic_generation.generator_params import GeneratorParams


class AbstractTimeSeriesGenerator(ABC):
    """
    Abstract base class for synthetic time series generators.
    """

    @abstractmethod
    def generate_time_series(self, random_seed: int | None = None) -> np.ndarray:
        """
        Generate synthetic time series data.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Time series values of shape (length,) for univariate or
            (length, num_channels) for multivariate time series.
        """
        pass


class GeneratorWrapper:
    """
    Unified base class for all generator wrappers, using a GeneratorParams dataclass
    for configuration. Provides parameter sampling, validation, and batch formatting utilities.
    """

    def __init__(self, params: GeneratorParams):
        """
        Initialize the GeneratorWrapper with a GeneratorParams dataclass.

        Parameters
        ----------
        params : GeneratorParams
            Dataclass instance containing all generator configuration parameters.
        """
        self.params = params
        self._set_random_seeds(self.params.global_seed)

    @staticmethod
    def _stable_name_hash(name: str) -> int:
        # Process-independent replacement for builtin hash(): zlib.crc32 returns
        # the same 32-bit value in every interpreter regardless of PYTHONHASHSEED.
        return zlib.crc32(name.encode("utf-8"))

    def _set_random_seeds(self, seed: int) -> None:
        # For parameter sampling, we want different generators to get different
        # parameter sequences even from the same base seed — derive a per-class
        # offset from a STABLE hash of the class name (see module header).
        param_seed = (int(seed) + self._stable_name_hash(self.__class__.__name__)) % 2**31
        self.rng = np.random.default_rng(param_seed)

        # Seed the global numpy (legacy) and torch RNGs too: a few vendored
        # generators (anomalies, spikes) and forecast_pfn's spike augmentation use
        # the global np.random.* stream, so it must be reseeded deterministically
        # before every batch. torch is seeded per the cascade determinism rules.
        np.random.seed(int(seed) % 2**31)
        torch.manual_seed(int(seed))

    def _sample_parameters(self, batch_size: int) -> dict[str, Any]:
        """
        Sample parameters with total_length fixed and history_length calculated.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing sampled parameter values where
            history_length = total_length - future_length.
        """

        # Select a suitable frequency based on the total length
        frequency = [select_safe_random_frequency(self.params.length, self.rng) for _ in range(batch_size)]
        start = [select_safe_start_date(self.params.length, frequency[i], self.rng) for i in range(batch_size)]

        return {
            "frequency": frequency,
            "start": start,
        }

    @abstractmethod
    def generate_batch(self, batch_size: int, seed: int | None = None, **kwargs) -> TimeSeriesContainer:
        raise NotImplementedError("Subclasses must implement generate_batch()")
