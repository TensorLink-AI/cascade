"""``[lane_d]`` — the distillation / real-data lane (DEC-CA-0062), WIRED BUT
NOT IMPLEMENTED.

The table parses and lands on :attr:`ChainConfig.lane_d` so every surface
(config, overlays, emission, status) has a place to read it from, but the
lane cannot be armed: a ``from_block`` other than 0 is refused at load until
the DEC's open questions are answered and the trainer / validator halves
exist. Every key below is a placeholder default the owner fills in.

Interfaces to fill (DEC-CA-0062 "Wiring points"): emission (a lane-D share
cut from ``takes_from``), a signed lane-D manifest bound into the lane-A
manifest (like ``lane_c_digest``), a validator mixin, a trainer driver, an
intake route, a pool prefix and the audit check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

LANE_D_TAKES_FROM = ("lane_a", "lane_b", "lane_c")
LANE_D_REMAINDERS = ("burn",)


@dataclass(frozen=True)
class LaneDConfig:
    # Activation: 0 = off. Refused > 0 until the lane is implemented.
    from_block: int = 0
    # Emissions: ``share`` of the total, carved from ``takes_from``.
    share: float = 0.0
    takes_from: str = "lane_b"
    remainder: str = "burn"
    # What is competed on (TBD): distillation targets, the teacher, and the
    # owner-pinned real corpus (DEC-CA-0028 ``real_corpus_ref`` pattern).
    targets: tuple[str, ...] = ()
    teacher_ref: str = ""
    # Owner 2026-10-19: the teacher is the Ephemeris time-series foundation-
    # model API (hosted zero-shot panel). ``teacher_mode`` = its routing mode
    # ("route" | "ensemble" | "explicit"); ``teacher_models`` names the panel
    # members for "explicit". Validators never call it (DEC-CA-0062).
    teacher_api: str = "ephemeris"
    teacher_mode: str = "route"
    teacher_models: tuple[str, ...] = ()
    real_corpus_ref: str = ""
    # The miner's recipe surface (owner 2026-10-19, details TBD): data is
    # SELECTED, not generated — from the real corpus and/or prior training
    # corpora (DEC-CA-0053 datasets) — plus distillation, augmentation, the
    # experts to train and a longer context (up to ``max_context_length``).
    data_sources: tuple[str, ...] = ("real", "prior_corpus")
    augmentations: tuple[str, ...] = ()
    expert_choice: str = "miner"
    max_context_length: int = 32768
    # Eval pool (a separate private pool, like MarketCast's ``pool_mkt``).
    pool_prefix: str = "pool_d"
    # Verdict (placeholders mirroring lane C; never a zero margin).
    alpha: float = 0.05
    gain_margin: float = 0.01
    # Payout schedule (placeholders mirroring lane B / C).
    full_rounds: int = 5
    half_life_rounds: float = 1.0
    floor: float = 0.1
    win_ttl_eras: int = 28
    # Free-form owner notes carried through for the dashboards.
    notes: dict = field(default_factory=dict)

    def active(self, boundary: int | None) -> bool:
        """Never active while the lane is unimplemented (``from_block`` is
        load-checked to 0)."""
        return False


def parse_lane_d(raw: object) -> LaneDConfig:
    """Parse ``[lane_d]``. Refuses arming (``from_block`` > 0) and malformed
    placeholders; everything else is carried as written for later use."""
    ld = raw if isinstance(raw, dict) else {}
    fb = ld.get("from_block", 0)
    if isinstance(fb, bool) or not isinstance(fb, int) or fb < 0:
        raise ValueError(f"[lane_d] from_block must be an integer >= 0; got {fb!r}")
    if fb:
        raise ValueError("[lane_d] from_block > 0: lane D (DEC-CA-0062, distillation / real "
                         "data) is wired but not implemented — fill in the DEC and ship the "
                         "trainer + validator halves before arming it")
    share = ld.get("share", 0.0)
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0.0 <= share < 1.0:
        raise ValueError(f"[lane_d] share must be in [0, 1); got {share!r}")
    takes = str(ld.get("takes_from", "lane_b"))
    if takes not in LANE_D_TAKES_FROM:
        raise ValueError(f"[lane_d] takes_from must be one of {LANE_D_TAKES_FROM}; got {takes!r}")
    rem = str(ld.get("remainder", "burn"))
    if rem not in LANE_D_REMAINDERS:
        raise ValueError(f"[lane_d] remainder must be one of {LANE_D_REMAINDERS}; got {rem!r}")
    targets = ld.get("targets", ())
    if not isinstance(targets, (list, tuple)) or not all(isinstance(t, str) for t in targets):
        raise ValueError(f"[lane_d] targets must be a list of strings; got {targets!r}")
    srcs = ld.get("data_sources", ("real", "prior_corpus"))
    if (not isinstance(srcs, (list, tuple)) or not srcs
            or not set(srcs) <= {"real", "prior_corpus"}):
        raise ValueError(f"[lane_d] data_sources must be a non-empty subset of "
                         f"['real', 'prior_corpus']; got {srcs!r}")
    augs = ld.get("augmentations", ())
    if not isinstance(augs, (list, tuple)) or not all(isinstance(a, str) for a in augs):
        raise ValueError(f"[lane_d] augmentations must be a list of strings; got {augs!r}")
    ec = str(ld.get("expert_choice", "miner"))
    if ec not in ("miner", "affinity", "all"):
        raise ValueError(f"[lane_d] expert_choice must be miner|affinity|all; got {ec!r}")
    mcl = ld.get("max_context_length", 32768)
    if isinstance(mcl, bool) or not isinstance(mcl, int) or not 0 < mcl <= 32768:
        raise ValueError(f"[lane_d] max_context_length must be in (0, 32768]; got {mcl!r}")
    mode = str(ld.get("teacher_mode", "route"))
    if mode not in ("route", "ensemble", "explicit"):
        raise ValueError(f"[lane_d] teacher_mode must be route|ensemble|explicit; got {mode!r}")
    models = ld.get("teacher_models", ())
    if not isinstance(models, (list, tuple)) or not all(isinstance(m, str) for m in models):
        raise ValueError(f"[lane_d] teacher_models must be a list of strings; got {models!r}")
    if mode == "explicit" and not models:
        raise ValueError("[lane_d] teacher_mode = 'explicit' needs teacher_models")
    margin = float(ld.get("gain_margin", 0.01))
    if not 0.0 < margin < 1.0:
        raise ValueError(f"[lane_d] gain_margin must be in (0, 1); got {margin}")
    notes = ld.get("notes", {})
    return LaneDConfig(
        from_block=0, share=float(share), takes_from=takes, remainder=rem,
        targets=tuple(targets), teacher_ref=str(ld.get("teacher_ref", "")),
        teacher_api=str(ld.get("teacher_api", "ephemeris")), teacher_mode=mode,
        teacher_models=tuple(models),
        real_corpus_ref=str(ld.get("real_corpus_ref", "")),
        data_sources=tuple(srcs), augmentations=tuple(augs), expert_choice=ec,
        max_context_length=int(mcl),
        pool_prefix=str(ld.get("pool_prefix", "pool_d")),
        alpha=float(ld.get("alpha", 0.05)), gain_margin=margin,
        full_rounds=int(ld.get("full_rounds", 5)),
        half_life_rounds=float(ld.get("half_life_rounds", 1.0)),
        floor=float(ld.get("floor", 0.1)), win_ttl_eras=int(ld.get("win_ttl_eras", 28)),
        notes=dict(notes) if isinstance(notes, dict) else {},
    )


__all__ = ["LANE_D_REMAINDERS", "LANE_D_TAKES_FROM", "LaneDConfig", "parse_lane_d"]
