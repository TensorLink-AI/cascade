"""The trainer's half of a lane-C competition: everything
:class:`~cascade.trainer.lane_c_driver.LaneCDriver` asks of a competition
kind, behind one object per kind (registered here, looked up by a target's
``<kind>:`` prefix).

The driver owns what every competition shares — the queue, the per-target
leg slots, one leg per hotkey across lanes, the end-before-leg check, the
signed settlement manifest and its write-ahead. A competition supplies:

* :meth:`TrainerCompetition.live` — whether it accepts legs at a block;
* :meth:`~TrainerCompetition.admitted_values` — the target values a
  commitment may name (``parse_finetune_commit`` / ``check_target``);
* :meth:`~TrainerCompetition.read_recipe` — the miner's recipe tree → a
  :class:`~cascade.trainer.finetune.LegRecipe` (a malformed one raises
  :class:`~cascade.trainer.finetune.RecipeError`: miner fault);
* :meth:`~TrainerCompetition.leg_hours` — the recipe's own wall (None = the
  ``[lane_c]`` default), re-checked against the competition's end before any
  rent;
* :meth:`~TrainerCompetition.train_leg` — one leg on the MINER's key → the
  ``finished`` record (``trained_pointer``, ``train_block``, …, and the
  entry's ``competition`` dict, which rides the signed manifest);
* optionally an ARBITER report (:meth:`~TrainerCompetition.report_key`): the
  main validator publishes one per settlement; the trainer's next manifest
  records its sha256, or ``"absent"`` after
  :meth:`~TrainerCompetition.report_timeout` settlements, under
  ``reports[<kind>]``.

MarketCast (DEC-CA-0064) is :class:`MarketCastTrainer`; its wire names
(``marketcast`` on an entry, ``mkt_reports`` / ``mkt_await``) predate this
interface and are kept byte-identical. Every other kind uses the generic
``competition`` / ``reports`` / ``awaits`` fields. A new kind is code (it
ships with a release, like its validator half in
:mod:`cascade.validator.lane_c_competitions`); configuring it is
``[lane_c.competitions.<kind>]``. ``docs/LANE_C_COMPETITIONS.md`` walks
through adding one.
"""

from __future__ import annotations

from pathlib import Path

from ..shared.config.lane_c import MKT_KIND, kind_of


class TrainerCompetition:
    """Base class: override what the kind needs. The defaults describe a
    competition with no arbiter report and the ``[lane_c]`` leg wall."""

    kind: str = ""

    def live(self, cfg, block: int) -> bool:
        """Whether a leg of this kind may be DISPATCHED at ``block`` (the
        competition's own start/end is checked by the driver too)."""
        raise NotImplementedError

    def admitted_values(self, cfg) -> tuple[str, ...]:
        """The target values a commitment may name now (() = whatever the
        kind's ``parse_value`` accepts)."""
        return ()

    def coldkey_cap(self, cfg) -> int:
        """Open legs one coldkey may hold at once (red-team F6)."""
        return 1

    def read_recipe(self, ops, tree: Path, *, target: str, submission_time: float | None,
                    train_time: float | None):
        raise NotImplementedError

    def leg_hours(self, cfg, recipe) -> float | None:
        return None

    def train_leg(self, ops, leg: dict, recipe) -> dict:
        raise NotImplementedError

    # ── the optional arbiter report ──

    def report_key(self, round_id: str) -> str | None:
        """Store key of the main validator's report for ``round_id`` (None =
        this kind has no arbiter report)."""
        return None

    def report_timeout(self, cfg) -> int:
        return 8

    def report_digest(self, text: str) -> str:
        import hashlib

        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def on_report(self, driver, text: str) -> None:
        """Called once per newly recorded report (e.g. requeue entries the
        validator gave up on for infra). Best effort."""


class MarketCastTrainer(TrainerCompetition):
    """DEC-CA-0064 MarketCast: Timestral fine-tunes on the miner's key."""

    kind = MKT_KIND

    def live(self, cfg, block: int) -> bool:
        mc = getattr(cfg, "marketcast", None)
        return mc is not None and bool(mc.active(block))

    def admitted_values(self, cfg) -> tuple[str, ...]:
        return tuple(getattr(getattr(cfg, "marketcast", None), "target_suites", ()) or ())

    def coldkey_cap(self, cfg) -> int:
        return int(getattr(cfg.marketcast, "max_open_legs_per_coldkey", 1))

    def read_recipe(self, ops, tree: Path, *, target: str, submission_time: float | None,
                    train_time: float | None):
        # marketcast.json (+ an optional generator tree); the embargo against
        # the commit time, the pod clipped to train_time (review M12)
        from .marketcast_runner import load_mkt_lineage, read_mkt_recipe

        return read_mkt_recipe(tree, ops.r.cfg, target=target,
                               submission_time=submission_time, train_time=train_time,
                               lineage=load_mkt_lineage(ops.r.work_root))

    def leg_hours(self, cfg, recipe) -> float | None:
        # the recipe's own wall (up to [marketcast] max_leg_hours)
        if not getattr(recipe, "leg_hours", 0.0):
            return None
        from .marketcast_runner import mkt_leg_hours

        return mkt_leg_hours(cfg, float(recipe.leg_hours))

    def train_leg(self, ops, leg: dict, recipe) -> dict:
        from .marketcast_runner import train_mkt_leg

        return train_mkt_leg(ops, leg, recipe)

    # MarketCast's arbiter keeps its own driver path (``mkt_reports`` /
    # ``mkt_await``, byte-identical): :meth:`LaneCDriver._mkt_arbiters`.


TRAINER_COMPETITIONS: dict[str, TrainerCompetition] = {}


def register_trainer_competition(comp: TrainerCompetition) -> TrainerCompetition:
    from ..shared.config.lane_c import COMPETITION_KINDS

    k = str(comp.kind)
    if k not in COMPETITION_KINDS:
        raise ValueError(f"lane-C competition {k!r}: register_competition_kind it first")
    if k in TRAINER_COMPETITIONS:
        raise ValueError(f"lane-C competition {k!r} already has a trainer half")
    TRAINER_COMPETITIONS[k] = comp
    return comp


def trainer_competition(target_or_kind: str) -> TrainerCompetition | None:
    """The trainer half of ``target``'s (or ``kind``'s) competition (None =
    no trainer support: the target is refused)."""
    s = str(target_or_kind or "")
    return TRAINER_COMPETITIONS.get(kind_of(s) if ":" in s else s)


register_trainer_competition(MarketCastTrainer())

__all__ = ["TRAINER_COMPETITIONS", "MarketCastTrainer", "TrainerCompetition",
           "register_trainer_competition", "trainer_competition"]
