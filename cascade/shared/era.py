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

from collections.abc import Iterable
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


def forfeit_successor(scoring: ScoringConfig, block: int | None) -> str:
    """CONSENSUS (DEC-CA-0048): the hotkey crowned in place of a forfeited
    king at ``block`` — ``[scoring] forfeit_successor_hotkey`` once the
    forfeiture is in force and the successor is not itself listed; "" =
    fall back to the court (most recent eligible former king, else vacant)."""
    if not forfeited_hotkeys(scoring, block):
        return ""
    succ = str(getattr(scoring, "forfeit_successor_hotkey", "") or "").strip()
    return "" if not succ or succ in forfeited_hotkeys(scoring, block) else succ


def forfeited_hotkeys(scoring: ScoringConfig, block: int | None) -> frozenset[str]:
    """CONSENSUS gate (DEC-CA-0048): the forfeit list in force for a settlement
    whose boundary is ``block``; empty before ``forfeit_from_block`` or when the
    list is empty. ``None`` (unknown height) ⇒ empty."""
    hks = tuple(getattr(scoring, "forfeit_hotkeys", ()) or ())
    gate = int(getattr(scoring, "forfeit_from_block", 0) or 0)
    if not hks or gate <= 0 or block is None or int(block) < gate:
        return frozenset()
    return frozenset(hks)


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


def settlement_era(round_cfg: RoundConfig, boundary: int) -> EraSpec:
    """The era a SETTLEMENT at epoch boundary ``boundary`` belongs to: the
    era containing ``boundary − 1``. An era's settlements are the boundaries
    in ``(start, start + length]`` — its LAST settlement is the next era's
    start block, so a leg that finishes just before the era rolls is still
    judged against this era's king, and the next era's first settlement is
    one grid step after its start. That makes the intake seamless: a leg
    that cannot clear ``start + length`` is exactly one whose start falls in
    the next era's pre-train window (``wall + margin`` before its start).
    Generation/member unresolved, like :func:`era_for_block`."""
    return era_for_block(round_cfg, max(0, int(boundary) - 1))


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


def effective_resync_cap_rounds(round_cfg: RoundConfig, scoring: ScoringConfig,
                                block: int | None) -> int:
    """King-resync safety-valve cap in ROUNDS in force at ``block``: from
    ``tenure_blocks_from_block`` it is ``king_resync_max_blocks`` on the grid
    in force there (the valve keeps its wall-time length across the grid
    switch — counted in settlements alone it tripped 4× sooner); before it,
    ``king_resync_max_rounds``. ``<= 0`` disables the valve either way."""
    blocks = int(getattr(scoring, "king_resync_max_blocks", 0) or 0)
    if blocks > 0 and tenure_blocks_active(scoring, block):
        return max(1, blocks // max(1, int(effective_epoch_blocks(round_cfg, int(block)))))
    return int(scoring.king_resync_max_rounds)


def legacy_king_anchor(round_cfg: RoundConfig, scoring: ScoringConfig, *,
                       block: int, tenure_rounds: int) -> int:
    """CONSENSUS: the crowning block imputed to a king that reigned before
    ``tenure_blocks_from_block`` (no recorded ``king_since_block``):
    ``block − tenure_rounds × grid_before_the_gate`` — its ``tenure_rounds``
    old-grid rounds laid out backwards from the settlement at ``block``. A
    pure function of the counter every validator already agreed on.

    It must be computed ONCE — at the king's first settlement past the gate —
    and persisted as ``king_since_block``. Re-deriving it at every settlement
    is wrong: ``tenure_rounds`` keeps counting NEW-grid settlements, so the
    imputed anchor slid back a whole old-grid round per settlement and the
    tenure grew ``old_grid / new_grid`` (4× on 3600→900) per settlement
    instead of 1 — the margin decayed four times faster than DEC-CA-0043
    intends."""
    gate = int(scoring.tenure_blocks_from_block)
    prev_grid = max(1, int(effective_epoch_blocks(round_cfg, max(0, gate - 1))))
    return int(block) - int(tenure_rounds) * prev_grid


def tenure_rounds_at(round_cfg: RoundConfig, scoring: ScoringConfig, *,
                     block: int | None, tenure_rounds: int,
                     king_since_block: int | None) -> int:
    """The king's tenure as the margin schedule counts it for a settlement at
    epoch boundary ``block``.

    Pre-gate: the recorded ``tenure_rounds`` counter (rounds survived).
    From ``tenure_blocks_from_block``: blocks reigned since
    ``king_since_block`` divided by the grid in force at ``block`` — a king
    crowned on the 3600 grid keeps its wall-time tenure across a switch to
    900 instead of being reset to a quarter. A legacy king with no recorded
    crowning block is anchored by :func:`legacy_king_anchor` for THIS call;
    the validator persists that anchor at the first post-gate settlement
    (``ChampionState.king_since_block``) so it is never re-imputed from a
    counter that has moved on.
    """
    if block is None or not tenure_blocks_active(scoring, block):
        return int(tenure_rounds)
    b = int(block)
    grid = max(1, int(effective_epoch_blocks(round_cfg, b)))
    if king_since_block is None:
        king_since_block = legacy_king_anchor(round_cfg, scoring, block=b,
                                             tenure_rounds=tenure_rounds)
    return max(0, (b - int(king_since_block)) // grid)


# ── cascade v2 (DEC-CA-0050): one typed activation block ─────────────────────


def settlement_contract_block(round_cfg: RoundConfig, boundary: int) -> int:
    """The block a SETTLEMENT's training contract resolves at: the start of
    the era it belongs to (:func:`settlement_era`). Pure block arithmetic —
    the trainer (``--contract-block``), every validator and the audit derive
    the same contract for a settlement without trusting its era stamp."""
    return int(settlement_era(round_cfg, int(boundary)).start_block)


def v2_settlement_active(cfg: Any, boundary: int | None) -> bool:
    """CONSENSUS: whether a settlement at epoch boundary ``boundary`` judges a
    v2 era (its era starts at or after ``[v2] from_block``). The settlement AT
    the v2 block belongs to the last pre-v2 era, so it is NOT v2."""
    v2 = getattr(cfg, "v2", None)
    if v2 is None or not int(getattr(v2, "from_block", 0) or 0) or boundary is None:
        return False
    return bool(v2.active(settlement_contract_block(cfg.round, int(boundary))))


def bench_gate_mode_at(cfg: Any, boundary: int | None) -> str:
    """CONSENSUS: the DEC-CA-0054 bench gate mode for a settlement at
    ``boundary`` — ``[v2] bench_gate_mode`` on a v2 settlement when set, else
    ``[scoring] bench_gate_mode``."""
    v2_mode = str(getattr(getattr(cfg, "v2", None), "bench_gate_mode", "off") or "off")
    if v2_mode != "off" and v2_settlement_active(cfg, boundary):
        return v2_mode
    return str(getattr(cfg.scoring, "bench_gate_mode", "off") or "off")


def bench_gate_configured(cfg: Any) -> bool:
    """Whether the bench gate can apply at any block (trainer-side: publish
    per-item rows and hold unbenched legs only then)."""
    v2_mode = str(getattr(getattr(cfg, "v2", None), "bench_gate_mode", "off") or "off")
    return (str(getattr(cfg.scoring, "bench_gate_mode", "off") or "off") != "off"
            or (v2_mode != "off" and int(getattr(cfg.v2, "from_block", 0) or 0) > 0))


# The DEC-CA-0061 rule's name (``cascade.eval.bench_gate.RULE_V2``; kept as a
# literal here so this module imports nothing from cascade.eval).
BENCH_RULE_V1, BENCH_RULE_V2 = "dec-ca-0054", "avg_noninf_winrate"


def bench_gate_v2_from(cfg: Any) -> int:
    """``[v2] bench_gate_v2_from_block`` (DEC-CA-0061), 0 = never."""
    return int(getattr(getattr(cfg, "v2", None), "bench_gate_v2_from_block", 0) or 0)


def bench_gate_rule_for_era(cfg: Any, era_start: int | None) -> str:
    """CONSENSUS: the bench gate rule for every settlement of the era
    starting at ``era_start`` — DEC-CA-0061's "avg_noninf_winrate" once that
    start is at or after ``[v2] bench_gate_v2_from_block``, else
    "dec-ca-0054"."""
    n = bench_gate_v2_from(cfg)
    if n > 0 and era_start is not None and int(era_start) >= n:
        return BENCH_RULE_V2
    return BENCH_RULE_V1


def bench_gate_rule_at(cfg: Any, boundary: int | None) -> str:
    """CONSENSUS: the bench gate rule for a settlement at ``boundary``
    (:func:`bench_gate_rule_for_era` of its era start, the same arithmetic as
    :func:`v2_settlement_active`, so the settlement AT the block keeps the old
    rule). Stamped in ``KothParams.bench_gate_rule``."""
    if boundary is None or not bench_gate_v2_from(cfg):
        return BENCH_RULE_V1
    return bench_gate_rule_for_era(cfg, settlement_contract_block(cfg.round, int(boundary)))


def bench_gate_v2_configured(cfg: Any) -> bool:
    """Whether the DEC-CA-0061 rule can apply at any block (trainer-side: add
    FEV to the bench sweep only then)."""
    return bench_gate_v2_from(cfg) > 0 and bench_gate_configured(cfg)


def v2_first_era(cfg: Any) -> int | None:
    """Index of the first v2 era (``[v2] from_block`` is an era boundary), or
    ``None`` when v2 is not configured."""
    n = int(getattr(getattr(cfg, "v2", None), "from_block", 0) or 0)
    return era_index_of(cfg.round, n) if n > 0 else None


def v2_first_warm_era(cfg: Any) -> int | None:
    """CONSENSUS: the first era a v2-lineage promotion may initialise, or
    ``None`` when v2 is not configured. A generation fired at the v2 block
    takes effect two eras later (``min_effective_era``); ``[v2]
    random_init_eras`` holds every v2 era before ``v2_first_era + N`` at
    random init, so the lineage explores from scratch for N eras before any
    warm start."""
    first = v2_first_era(cfg)
    if first is None:
        return None
    hold = int(getattr(getattr(cfg, "v2", None), "random_init_eras", 0) or 0)
    return first + max(2, hold)


def v2_reign_floor_block(cfg: Any) -> int:
    """CONSENSUS: the block the promotion reign clock may count from in v2 —
    the start of era ``v2_first_era + random_init_eras``. A reign that began
    earlier counts from here, so the first v2 promotion needs the random-init
    window PLUS a ripe reign after it (a dethrone restarts the clock as
    usual). 0 when v2 is not configured or the window is 0."""
    v2 = getattr(cfg, "v2", None)
    n = int(getattr(v2, "from_block", 0) or 0)
    hold = int(getattr(v2, "random_init_eras", 0) or 0)
    if n <= 0 or hold <= 0:
        return 0
    return n + hold * era_length_blocks(cfg.round, n)


def v2_generation_usable(cfg: Any, era_start: int, effective_era: int) -> bool:
    """CONSENSUS (DEC-CA-0050 lineage reset): may a promotion generation
    taking effect at ``effective_era`` initialise the era starting at
    ``era_start``? Always outside v2. In a v2 era only a generation from the
    v2 lineage that takes effect at or after :func:`v2_first_warm_era`
    (``v2_first_era + max(2, random_init_eras)``). Anything older is a toto2
    generation — a toto2 checkpoint cannot initialise the MoE — or falls in
    the random-init window, and the era trains from random init (generation
    0). The trainer's ledger, every validator's ``_check_era_init`` and the
    audit apply this one rule."""
    warm = v2_first_warm_era(cfg)
    if warm is None or not cfg.v2.active(int(era_start)):
        return True
    return int(effective_era) >= warm


def v2_era_active(cfg: Any, era_start: int | None) -> bool:
    """Whether the era starting at ``era_start`` is a v2 era (tolerates a
    config without ``[v2]`` — always False)."""
    v2 = getattr(cfg, "v2", None)
    return bool(v2 is not None and era_start is not None and v2.active(int(era_start)))


def v2_seed_epoch_start(cfg: Any, era_index: int, warm_eras: Iterable[int]) -> int | None:
    """CONSENSUS (DEC-CA-0055): the first era of the v2 SEED EPOCH containing
    ``era_index``, or ``None`` pre-v2. A seed epoch runs from one warm start
    to the next: it begins at the first v2 era (random init) and at each
    v2-lineage promotion's ``effective_era`` (``warm_eras``), and every era in
    it shares ONE training seed set and ONE init. The king trains once per
    epoch — once per warm start — and a challenger that dethrones inside the
    epoch trained with the same seeds and init, so its checkpoint is the
    king's leg for the rest of it."""
    first = v2_first_era(cfg)
    if first is None or int(era_index) < first:
        return None
    return max([first] + [int(e) for e in warm_eras if first < int(e) <= int(era_index)])


def v2_era_seed_block(cfg: Any, era: EraSpec, warm_eras: Iterable[int]) -> int:
    """CONSENSUS: the seed block of ``era`` — its seed epoch's FIRST era's
    seed block (the block one era before that era starts), else the era's own
    ``seed_block``. The v2 grid is fixed from ``[v2] from_block``, so the era
    length there is the length of every v2 era."""
    start = v2_seed_epoch_start(cfg, int(era.index), warm_eras)
    if start is None or start == int(era.index):
        return int(era.seed_block)
    length = era_length_blocks(cfg.round, int(era.start_block))
    return max(0, (start - 1) * length)


def v2_check_due(cfg: Any, *, boundary: int, reign_start_block: int | None) -> bool:
    """CONSENSUS (DEC-CA-0055): is the king check due at the settlement at
    ``boundary``? Due once a promotion fired here would take effect
    (``min_effective_era``) at least ``[v2] reign_warm_start_blocks`` after
    the reign began — the reign counted from no earlier than
    :func:`v2_reign_floor_block` — so a passing check's warm start lands a
    fixed time into the reign (3 days by default; the era of notice is
    absorbed, not added). False when the reign is unanchored or the key is 0."""
    need = int(getattr(getattr(cfg, "v2", None), "reign_warm_start_blocks", 0) or 0)
    if need <= 0 or reign_start_block is None:
        return False
    start = max(int(reign_start_block), v2_reign_floor_block(cfg))
    eff_era = min_effective_era(cfg.round, int(boundary))
    eff_block = eff_era * era_length_blocks(cfg.round, int(boundary))
    return eff_block - start >= need


# ── DEC-CA-0063: lane-A benches on the main validator's GPU ─────────────────


def bench_source_from(cfg: Any) -> int:
    """``[v2] bench_source_from_block`` when ``[v2] bench_source =
    "validator"``, else 0 (never)."""
    v2 = getattr(cfg, "v2", None)
    if str(getattr(v2, "bench_source", "trainer") or "trainer") != "validator":
        return 0
    return int(getattr(v2, "bench_source_from_block", 0) or 0)


def validator_bench_era(cfg: Any, era_start: int | None) -> bool:
    """CONSENSUS: whether the era starting at ``era_start`` takes its lane-A
    bench rows from the main validator (DEC-CA-0063) instead of the trainer.
    Trainer-side: no lane-A challenger / king sweeps for that era's legs."""
    n = bench_source_from(cfg)
    return bool(n > 0 and era_start is not None and int(era_start) >= n)


def validator_bench_at(cfg: Any, boundary: int | None) -> bool:
    """CONSENSUS: :func:`validator_bench_era` of the era a settlement at
    ``boundary`` belongs to (the settlement AT the block keeps the trainer
    report, exactly like :func:`bench_gate_rule_at`)."""
    if boundary is None or not bench_source_from(cfg):
        return False
    return validator_bench_era(cfg, settlement_contract_block(cfg.round, int(boundary)))


def validator_bench_configured(cfg: Any) -> bool:
    """Whether a validator bench source applies at any block."""
    return bench_source_from(cfg) > 0
