"""Era arithmetic for rolling intake + the era king (DEC-CA-0043).

Vocabulary
----------
* **Settlement** — one manifest + verdict on the epoch grid (what the codebase
  calls a *round*; ``round_id`` stays the settlement boundary's seed).
* **Era** — ``era_settlements`` consecutive settlements sharing ONE set of
  training seeds, ONE init and ONE cached king checkpoint.

Everything here is pure block arithmetic so the trainer, every validator and
the audit derive the SAME era for a block with zero trainer discretion:

* ``era_index    = start_block // (epoch_blocks × era_settlements)``
* era boundaries are grid boundaries (an era is a whole number of epochs);
* era seeds derive from the PREVIOUS era's start block hash —
  ``seed_block = start_block − era_length`` — so the era is known one era
  ahead and the king's leg (and late-era challengers) pre-train under the
  next era's seeds.

Block gates (release-then-activate, all equal to one ROLLOVER block, checked
by ``load_chain_config``):

* ``[round] rolling_from_block`` — trainer policy: rolling intake,
  cross-boundary legs, settlement manifests.
* ``[scoring] era_king_from_block`` — CONSENSUS: validators accept an
  era-seeded king entry and verify the era envelope.
* ``[scoring] tenure_blocks_from_block`` — CONSENSUS: tenure and ripeness
  re-denominated to blocks (``margin_warmup_blocks`` /
  ``cascade_reign_blocks``) so a faster grid does not shorten the decay and
  promotion clocks in wall-time.

Before the gate every function returns the pre-rollover answer (``active``
false, ``None`` eras) so callers stay bit-identical to the ungated code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import RoundConfig, ScoringConfig, effective_epoch_blocks


@dataclass(frozen=True)
class EraSpec:
    """The era a settlement belongs to, as stamped in the manifest.

    ``generation`` / ``member_index`` name the init: era ``n`` trains from
    ``members_gen(generation)[member_index]`` with ``member_index = n % k``
    and ``generation`` the latest promotion whose ``effective_era <= n``
    (0 = no generation live ⇒ random init from the era's training seed).
    """

    index: int
    start_block: int
    seed_block: int
    generation: int = 0
    member_index: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "index": int(self.index),
            "start_block": int(self.start_block),
            "seed_block": int(self.seed_block),
            "generation": int(self.generation),
            "member_index": int(self.member_index),
        }

    @classmethod
    def from_json(cls, obj: Any) -> EraSpec:
        if not isinstance(obj, dict):
            raise ValueError("era must be an object")
        try:
            return cls(
                index=int(obj["index"]),
                start_block=int(obj["start_block"]),
                seed_block=int(obj["seed_block"]),
                generation=int(obj.get("generation", 0) or 0),
                member_index=int(obj.get("member_index", 0) or 0),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"era stamp malformed: {e}") from e


def rolling_active(round_cfg: RoundConfig, block: int | None) -> bool:
    """Trainer-policy gate: rolling intake from ``rolling_from_block``."""
    gate = int(getattr(round_cfg, "rolling_from_block", 0) or 0)
    return gate > 0 and block is not None and int(block) >= gate


def era_king_active(scoring: ScoringConfig, block: int | None) -> bool:
    """CONSENSUS gate: the validator verifies the era envelope for a settlement
    whose epoch boundary is ``>= era_king_from_block``."""
    gate = int(getattr(scoring, "era_king_from_block", 0) or 0)
    return gate > 0 and block is not None and int(block) >= gate


def tenure_blocks_active(scoring: ScoringConfig, block: int | None) -> bool:
    """CONSENSUS gate: tenure and ripeness counted in BLOCKS from
    ``tenure_blocks_from_block`` (``margin_warmup_blocks`` /
    ``cascade_reign_blocks`` in force)."""
    gate = int(getattr(scoring, "tenure_blocks_from_block", 0) or 0)
    return gate > 0 and block is not None and int(block) >= gate


def era_length_blocks(round_cfg: RoundConfig, block: int) -> int:
    """Blocks per era in force at ``block``: the grid length there times
    ``era_settlements`` (``era_settlements <= 0`` ⇒ one settlement per era —
    the degenerate "every round is its own era")."""
    n = max(1, int(getattr(round_cfg, "era_settlements", 0) or 0))
    return int(effective_epoch_blocks(round_cfg, block)) * n


def era_index_of(round_cfg: RoundConfig, block: int) -> int:
    return int(block) // era_length_blocks(round_cfg, block)


def era_start_block(round_cfg: RoundConfig, block: int) -> int:
    length = era_length_blocks(round_cfg, block)
    return (int(block) // length) * length


def era_for_block(round_cfg: RoundConfig, block: int) -> EraSpec:
    """The era containing ``block`` (generation/member unresolved — the
    caller fills them from the promotion ledger)."""
    length = era_length_blocks(round_cfg, block)
    index = int(block) // length
    start = index * length
    return EraSpec(index=index, start_block=start,
                   seed_block=max(0, start - length))


def next_era_start(round_cfg: RoundConfig, block: int) -> int:
    """First era boundary strictly after ``block``."""
    length = era_length_blocks(round_cfg, block)
    return (int(block) // length + 1) * length


def min_effective_era(round_cfg: RoundConfig, created_block: int) -> int:
    """Earliest era a promotion record created at ``created_block`` may take
    effect: the first era boundary at least ONE FULL ERA after creation.

    A record created inside era ``e`` (blocks ``[e·L, (e+1)·L)``) becomes
    effective no earlier than era ``e + 2`` — the boundary at ``(e+1)·L``
    starts the notice era, the one at ``(e+2)·L`` is the first that had a
    whole era of notice. Validators reject earlier claims, so every pre-train
    window knows both its init and its seeds.
    """
    return era_index_of(round_cfg, created_block) + 2


def member_index_for_era(era_index: int, k: int) -> int:
    """DEC-CA-0013's ``epoch_index % k`` rotation, re-keyed on the era."""
    return int(era_index) % max(1, int(k))


def effective_margin_warmup_rounds(round_cfg: RoundConfig, scoring: ScoringConfig,
                                   block: int | None) -> int:
    """``margin_warmup_rounds`` in force at ``block``: from
    ``tenure_blocks_from_block`` the warm-up is ``margin_warmup_blocks``
    expressed in settlements of the grid in force at ``block``, so a 4× faster
    grid keeps the same wall-time decay; before it (or with the blocks knob
    unset) the shipped ``margin_warmup_rounds``."""
    blocks = int(getattr(scoring, "margin_warmup_blocks", 0) or 0)
    if blocks > 0 and tenure_blocks_active(scoring, block):
        return max(1, blocks // max(1, int(effective_epoch_blocks(round_cfg, int(block)))))
    return int(scoring.margin_warmup_rounds)


def effective_reign_threshold_rounds(round_cfg: RoundConfig, scoring: ScoringConfig,
                                     block: int | None) -> int:
    """Cascade ripeness threshold in ROUNDS in force at ``block``: from
    ``tenure_blocks_from_block`` it is ``cascade_reign_blocks`` on the grid
    in force at ``block``; before it the shipped ``cascade_reign_rounds``
    (dataclass field ``cascade_reign_days``)."""
    blocks = int(getattr(scoring, "cascade_reign_blocks", 0) or 0)
    if blocks > 0 and tenure_blocks_active(scoring, block):
        return max(1, blocks // max(1, int(effective_epoch_blocks(round_cfg, int(block)))))
    return int(scoring.cascade_reign_days)


def tenure_rounds_at(round_cfg: RoundConfig, scoring: ScoringConfig, *,
                     block: int | None, tenure_rounds: int,
                     king_since_block: int | None) -> int:
    """The king's tenure as the margin schedule counts it for a settlement at
    epoch boundary ``block``.

    Pre-gate: the recorded ``tenure_rounds`` counter (rounds survived).
    From ``tenure_blocks_from_block``: blocks reigned divided by the grid in
    force at ``block`` — a king crowned on the 3600 grid keeps its wall-time
    tenure across a switch to 900 instead of being reset to a quarter. A
    legacy king with no recorded crowning block is anchored at
    ``block − tenure_rounds × grid_before_the_gate``; that anchor is a pure
    function of the counter every validator already agreed on.
    """
    if block is None or not tenure_blocks_active(scoring, block):
        return int(tenure_rounds)
    b = int(block)
    grid = max(1, int(effective_epoch_blocks(round_cfg, b)))
    if king_since_block is None:
        gate = int(scoring.tenure_blocks_from_block)
        prev_grid = max(1, int(effective_epoch_blocks(round_cfg, max(0, gate - 1))))
        king_since_block = b - int(tenure_rounds) * prev_grid
    return max(0, (b - int(king_since_block)) // grid)
