"""The validator's half of a lane-C competition: what
:class:`~cascade.validator.lane_c.LaneCMixin` asks of a competition kind when
it judges and folds a settlement (the trainer's half:
:mod:`cascade.trainer.lane_c_competitions`).

The mixin owns what every competition shares — reading and verifying the
signed lane-C manifest bound by the lane-A settlement, the catch-up walk in
chain order, the receipt rows, the slot fold, the payout split by
competition weight. A competition supplies:

* :meth:`ValidatorCompetition.judge` — the settlement's
  ``(plan, judgements)`` for its own targets, from the verified manifest
  (``cm``; None when the settlement bound none or it failed verification).
  Deterministic given the bytes and ``st``; a READ failure raises
  :class:`~cascade.validator.lane_c.LaneCTransient` (the settlement holds).
  Called at EVERY lane-C settlement (also before the kind starts or after it
  ends, so it can return ``ended`` rows); judge nothing while not started.
* :meth:`~ValidatorCompetition.fold` — apply ``plan`` to ``st``; the wins
  that took a slot. The default folds the judgements through the generic
  pending → confirm rule (:func:`cascade.validator.lane_c.apply_lane_c`):
  judge-phase rows pick a pending best passer per target, the next
  settlement's confirm-phase row takes the slot. A competition that keeps
  its own state (MarketCast's round backlog) overrides it; per-kind state
  lives in ``st.comp[<kind>]`` (omitted when empty).

The judgement rows go on the receipt and the auditor replays them with the
generic DEC-CA-0056 rules unless the kind ships its own audit hook (the
MarketCast replay is :mod:`cascade.audit.marketcast_reproduce`).

The MAIN validator may publish an arbiter report per round (MarketCast's
rows); the trainer records it in ``reports[<kind>]`` of a later manifest,
and the kind's ``judge`` applies a round exactly when a manifest records it.
Report publishing and the optional GPU offload
(:mod:`cascade.validator.mkt_offload`) are the kind's own code.
"""

from __future__ import annotations

from ..shared.config.lane_c import MKT_KIND, kind_of


class ValidatorCompetition:
    kind: str = ""
    # True: :meth:`fold` = the generic pending/confirm fold of the judgements
    generic_fold: bool = True

    def judge(self, runner, manifest, cm, *, rid: str, block: int, era_index: int, st,
              commitments_fn) -> tuple[object, list]:
        raise NotImplementedError

    def fold(self, st, plan, judgements: list, *, round_id: str, settle_block: int,
             era_index: int) -> list:
        return []


class MarketCastValidator(ValidatorCompetition):
    """DEC-CA-0064 MarketCast (:class:`cascade.validator.marketcast.MarketCastMixin`):
    rows on the main validator, rounds applied from trainer-arbitrated
    reports, its own backlog fold."""

    kind = MKT_KIND
    generic_fold = False

    def judge(self, runner, manifest, cm, *, rid, block, era_index, st, commitments_fn):
        return runner._mkt_settle(manifest, cm, rid=rid, block=block, era_index=era_index,
                                  st=st, commitments_fn=commitments_fn)

    def fold(self, st, plan, judgements, *, round_id, settle_block, era_index):
        from .marketcast import fold_mkt_plan

        return fold_mkt_plan(st, plan)   # labels computed at judge time


VALIDATOR_COMPETITIONS: dict[str, ValidatorCompetition] = {}


def register_validator_competition(comp: ValidatorCompetition) -> ValidatorCompetition:
    from ..shared.config.lane_c import COMPETITION_KINDS

    k = str(comp.kind)
    if k not in COMPETITION_KINDS:
        raise ValueError(f"lane-C competition {k!r}: register_competition_kind it first")
    if k in VALIDATOR_COMPETITIONS:
        raise ValueError(f"lane-C competition {k!r} already has a validator half")
    VALIDATOR_COMPETITIONS[k] = comp
    return comp


def validator_competition(target_or_kind: str) -> ValidatorCompetition | None:
    s = str(target_or_kind or "")
    return VALIDATOR_COMPETITIONS.get(kind_of(s) if ":" in s else s)


def judging_order() -> list[ValidatorCompetition]:
    """MarketCast first (its rows lead the receipt, as before), then every
    other kind by name — a fixed order every validator shares."""
    return sorted(VALIDATOR_COMPETITIONS.values(), key=lambda c: (c.kind != MKT_KIND, c.kind))


register_validator_competition(MarketCastValidator())

__all__ = ["VALIDATOR_COMPETITIONS", "MarketCastValidator", "ValidatorCompetition",
           "judging_order", "register_validator_competition", "validator_competition"]
