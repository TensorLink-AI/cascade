"""Versioned direct-MoE contracts. No production architecture or policy defaults."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

SUITES = ("GIFT", "FEV", "TIME", "BOOM")
POLICIES = (
    "architecture", "initial_generator", "bootstrap_payer", "baseline_payer",
    "founding_count", "founding_deadline", "value_budget", "public_guard",
    "private_gain_rule", "funding_deadline", "outage_deadline", "hardware_profiles",
    "continuation_reward", "royalty", "pause_payout", "reservation_policy",
    "specialist_window", "edited_optimizer_policy", "repeated_look_policy",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class DirectConfig:
    version: int = 1
    from_block: int = 0
    settings: dict | None = None

    def unresolved(self):
        s = self.settings or {}
        return [key for key in POLICIES if key not in s or s[key] is None]

    def validate(self):
        if self.version != 1 or self.from_block < 0:
            raise ValueError("unsupported direct-MoE version/activation")
        if self.from_block:
            missing = self.unresolved()
            if missing:
                raise ValueError("unresolved direct-MoE policies: " + ", ".join(missing))
            # A policy-filled file is not a validated deployment contract.
            raise ValueError("direct-MoE live activation unavailable: worker/provider qualification required")


def parse_direct_config(raw):
    unknown = set(raw) - {"version", "from_block", "settings"}
    if unknown:
        raise ValueError(f"unknown direct_moe fields: {sorted(unknown)}")
    cfg = DirectConfig(**raw)
    cfg.validate()
    return cfg


@dataclass(frozen=True)
class BatchSchedule:
    context: int = 4096
    channels: int = 8
    thresholds: tuple[int, ...] = (150_000_000_000, 400_000_000_000)
    batches: tuple[int, ...] = (64, 128, 256)
    horizon: int = 1_000_000_000_000
    warmup: int = 10_000_000_000
    # Values includes missing/padded input slots; unused tail slots are masked.
    accounting: str = "scalar_input_slots_tail_masked"

    def __post_init__(self):
        if (min(self.context, self.channels, self.horizon) <= 0
                or not 0 <= self.warmup <= self.horizon
                or len(self.batches) != len(self.thresholds) + 1
                or tuple(sorted(set(self.thresholds))) != self.thresholds
                or any(x <= 0 for x in (*self.thresholds, *self.batches))
                or self.accounting != "scalar_input_slots_tail_masked"):
            raise ValueError("invalid lifetime batch schedule")

    def batch(self, exposure):
        if exposure < 0:
            raise ValueError("negative exposure")
        return self.batches[sum(exposure >= t for t in self.thresholds)]

    def steps(self, exposure, budget, *, world_size=8, micro_batch=1):
        """Exact step plan; partial threshold/tail steps never claim overshot values.

        valid_values is a global prefix in rank-major/microbatch-major order.
        The worker must mask the suffix and normalise gradients by valid values.
        """
        if min(budget, world_size, micro_batch) <= 0 or exposure < 0:
            raise ValueError("invalid budget/topology")
        if exposure + budget > self.horizon:
            raise ValueError("lineage horizon exceeded")
        used = 0
        while used < budget:
            position = exposure + used
            batch = self.batch(position)
            if batch % (world_size * micro_batch):
                raise ValueError("global batch incompatible with accumulation")
            capacity = batch * self.context * self.channels
            edge = next((t for t in self.thresholds if t > position), self.horizon)
            n = min(capacity, budget - used, edge - position)
            yield {"exposure": position, "valid_values": n, "global_batch": batch,
                   "accumulation": batch // (world_size * micro_batch),
                   "capacity_values": capacity,
                   "lr_scale": min(1.0, position / self.warmup) if self.warmup else 1.0}
            used += n


@dataclass(frozen=True)
class Cohort:
    anchor: str
    exposure: int
    budget: int
    generator_contract: str
    optimizer_contract: str
    schedule: BatchSchedule
    seed: int
    architecture: str
    resource_contract: str

    def __post_init__(self):
        if self.exposure < 0 or self.budget <= 0 or not all((self.anchor,
                self.generator_contract, self.optimizer_contract, self.architecture,
                self.resource_contract)):
            raise ValueError("incomplete cohort")

    @property
    def key(self):
        # Evaluation panel intentionally excluded: changed windows rescore endpoints.
        return digest(asdict(self))

    def cache_key(self, generator):
        return digest({"cohort": self.key, "generator": generator})


def finite(value, *, minimum=0):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError("invalid finite nonnegative value")
    return value
