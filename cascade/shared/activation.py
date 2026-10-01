"""Stake-weighted activation of the DEC-CA-0043 rollover (DEC-CA-0045).

The rollover block today is typed into ``chain.toml`` and every validator
has to upgrade before it. This module lets the fleet DECIDE the block on
chain instead:

1. **Signal.** A validator on the new release writes a plain on-chain
   commitment from its hotkey — ``cascade-ready:1:<feature>:0:0`` — once, at
   startup (:func:`format_signal`). Miners never see it: the trainer's field
   reads the revealed store and :meth:`ChainClient.poll_commitments` drops
   the reserved prefix.
2. **Tally.** At every boundary of the grid in force, every node reads the
   metagraph and the plain commitments AS OF that boundary block and sums
   the stake of the permit-holding validators that have signalled
   (:func:`tally`). Same block, same chain state, same number everywhere.
3. **Lock-in.** The first boundary where the signed share reaches
   ``[activation] threshold`` locks in. The rollover is the NEXT boundary
   after it (:func:`activation_block_for`): the round where the count
   crosses finishes on the old rules, the next one starts on the new.
   Lock-in is one-way — a node that has seen it persists the block
   (:class:`ActivationStore`) and never re-evaluates, so stake drifting back
   under the line the next day changes nothing.
4. **Agreement.** A validator that has locked in rewrites its note as
   ``cascade-ready:1:<feature>:<lock_block>:<activation_block>``. A node that
   restarts, joins late, or cannot read an old block adopts the block that
   validators holding ``threshold`` of eligible stake all name
   (:func:`agreed_activation`) — no archive node, no trainer needed. A node
   NEVER counts from a later chain view when the boundary read fails: two
   nodes counting different states is the fork this exists to prevent, so
   it waits for the notes instead.
5. **Apply.** :func:`apply_activation` rewrites the loaded config so every
   DEC-CA-0043 key names the resolved block (rolling intake, era king,
   tenure in blocks, the grid switch, the increment-unit max-T) — exactly
   what the owner would have typed, checked by the loader's own rule. The
   typed-in keys always win (the owner override).

The validator stamps the resolved block on every receipt from then on
(``activation_block``, drop-when-default), so the audit replays each round
under the block the fleet actually decided.

**Secondary features (DEC-CA-0048).** A hotkey holds ONE plain commitment,
so any later decision rides in the same note as extra ``<feature>:<lock>:
<act>`` segments after the primary one: ``cascade-ready:1:<f1>:<l1>:<a1>:
<f2>:<l2>:<a2>``. Each segment is tallied, locked in and persisted on its
own (:func:`resolve_activation` takes a :class:`FeatureSpec`); the primary
segment keeps today's exact bytes, so a note without a second feature is
byte-identical to a v1 note. A parser from before this module carried
segments reads an extended note as malformed and counts it as NOT signed —
acceptable only because every feature that rides here is itself a consensus
change every validator must install. The king forfeiture
(``[scoring] forfeit_hotkeys`` with ``forfeit_from_block = 0``) is the first
such feature: its name is ``forfeit-<sha256(sorted hotkeys)[:8]>`` so a
changed list is a fresh vote, and lock-in writes the resolved block into
``forfeit_from_block`` (:func:`apply_forfeit_activation`). A typed-in
``forfeit_from_block`` is the owner override, exactly as for the rollover.
The DEC-CA-0049 dethrone bar v2 is the second such feature
(``margin-v2-<sha256(start|end|blocks)[:8]>``): lock-in writes the rollover
boundary into ``[scoring] margin_v2_from_block``
(:func:`apply_margin_v2_activation`); a typed block is the owner override.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import (
    ChainConfig,
    check_rollover_alignment,
    effective_epoch_blocks,
    margin_v2_configured,
)

log = logging.getLogger("cascade.activation")

SIGNAL_PREFIX = "cascade-ready:"
SIGNAL_VERSION = 1
# A boundary older than this many blocks at the head is treated as PRUNED
# when its as-of read fails (public finney endpoints discard state after
# ~256 blocks); a younger failure is a transient and is retried as-is.
PRUNED_AFTER_BLOCKS = 300


# ── the on-chain note ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReadySignal:
    """One validator's parsed note. ``lock_block`` / ``activation_block`` are
    0 until that validator has itself seen lock-in."""

    feature: str
    lock_block: int = 0
    activation_block: int = 0


def _segment(feature: str, lock_block: int, activation_block: int) -> str:
    feat = str(feature or "").strip()
    if not feat or ":" in feat or any(c.isspace() for c in feat):
        raise ValueError(f"feature must be a non-empty token without ':' or spaces; got {feat!r}")
    return f"{feat}:{int(lock_block)}:{int(activation_block)}"


def format_signal(feature: str, *, lock_block: int = 0, activation_block: int = 0) -> str:
    """``cascade-ready:1:<feature>:<lock_block>:<activation_block>``."""
    return f"{SIGNAL_PREFIX}{SIGNAL_VERSION}:{_segment(feature, lock_block, activation_block)}"


def format_signals(primary: ReadySignal, *extras: ReadySignal) -> str:
    """The primary segment (today's exact note) followed by one segment per
    secondary feature: ``cascade-ready:1:<f1>:<l1>:<a1>[:<f2>:<l2>:<a2>…]``.
    Feature names must be distinct."""
    sigs = (primary, *extras)
    names = [x.feature for x in sigs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature in note: {names}")
    body = ":".join(_segment(x.feature, x.lock_block, x.activation_block) for x in sigs)
    return f"{SIGNAL_PREFIX}{SIGNAL_VERSION}:{body}"


def is_signal_payload(payload: object) -> bool:
    return isinstance(payload, str) and payload.startswith(SIGNAL_PREFIX)


def _parse_segment(feat: str, lock: str, act: str) -> ReadySignal | None:
    try:
        lock_b, act_b = int(lock), int(act)
    except ValueError:
        return None
    if not feat or lock_b < 0 or act_b < 0 or (act_b and not lock_b) or (lock_b and act_b <= lock_b):
        return None
    return ReadySignal(feature=feat, lock_block=lock_b, activation_block=act_b)


def parse_signals(payload: object) -> tuple[ReadySignal, ...] | None:
    """Every segment of a note, primary first; ``None`` for anything that is
    not a well-formed v1 note (a future version, a generator pointer, a
    malformed or duplicated segment, garbage). A whole note stands or falls
    together: one bad segment and no segment counts."""
    if not is_signal_payload(payload):
        return None
    parts = str(payload)[len(SIGNAL_PREFIX):].split(":")
    if len(parts) < 4 or (len(parts) - 1) % 3:
        return None
    try:
        if int(parts[0]) != SIGNAL_VERSION:
            return None
    except ValueError:
        return None
    out: list[ReadySignal] = []
    for i in range(1, len(parts), 3):
        sig = _parse_segment(parts[i], parts[i + 1], parts[i + 2])
        if sig is None:
            return None
        out.append(sig)
    if len({x.feature for x in out}) != len(out):
        return None
    return tuple(out)


def parse_signal(payload: object) -> ReadySignal | None:
    """The PRIMARY segment of a note (``None`` when the note is malformed) —
    the pre-segment reading, kept for callers that only know one feature."""
    sigs = parse_signals(payload)
    return sigs[0] if sigs else None


def signal_for(payload: object, feature: str) -> ReadySignal | None:
    """The segment of ``payload`` that names ``feature``, else ``None``."""
    sigs = parse_signals(payload)
    if not sigs:
        return None
    for sig in sigs:
        if sig.feature == feature:
            return sig
    return None


# ── the tally ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ValidatorStake:
    """A metagraph row as the tally sees it (``ChainClient.validator_stakes``)."""

    hotkey: str
    stake: float
    permit: bool
    last_update: int = 0


@dataclass(frozen=True)
class Tally:
    feature: str
    block: int                       # the boundary the chain was read AS OF
    threshold: float
    signed_stake: float
    total_stake: float
    signed: tuple[str, ...]          # eligible hotkeys that have signalled
    eligible: tuple[str, ...]        # every hotkey counted in total_stake

    @property
    def ratio(self) -> float:
        return self.signed_stake / self.total_stake if self.total_stake > 0 else 0.0

    @property
    def locked(self) -> bool:
        return self.total_stake > 0 and self.ratio >= self.threshold

    def to_json(self) -> dict:
        d = asdict(self)
        d["ratio"] = round(self.ratio, 6)
        d["locked"] = self.locked
        d["n_signed"] = len(self.signed)
        d["n_eligible"] = len(self.eligible)
        return d


def eligible_validators(
    validators: list[ValidatorStake] | tuple[ValidatorStake, ...], *, block: int,
    dormant_after_blocks: int = 0,
) -> list[ValidatorStake]:
    """Permit holders with positive stake; with ``dormant_after_blocks`` set,
    only those that set weights within that many blocks of ``block``."""
    out = []
    for v in validators:
        if not v.permit or not (v.stake > 0):
            continue
        if dormant_after_blocks > 0 and int(block) - int(v.last_update) > int(dormant_after_blocks):
            continue
        out.append(v)
    return out


def tally(
    feature: str,
    validators: list[ValidatorStake] | tuple[ValidatorStake, ...],
    signals: dict[str, str],
    *,
    threshold: float,
    block: int,
    dormant_after_blocks: int = 0,
) -> Tally:
    """Sum the eligible stake behind ``feature`` at ``block``.

    ``signals`` maps hotkey → raw plain-commitment payload (every hotkey's,
    the tally parses and filters). A note for another feature, or a
    malformed one, counts as not signed.
    """
    elig = eligible_validators(validators, block=block, dormant_after_blocks=dormant_after_blocks)
    signed: list[str] = []
    signed_stake = 0.0
    total = 0.0
    for v in elig:
        total += float(v.stake)
        sig = signal_for(signals.get(v.hotkey), feature)
        if sig is not None:
            signed.append(v.hotkey)
            signed_stake += float(v.stake)
    return Tally(
        feature=str(feature), block=int(block), threshold=float(threshold),
        signed_stake=signed_stake, total_stake=total,
        signed=tuple(sorted(signed)), eligible=tuple(sorted(v.hotkey for v in elig)),
    )


def valid_pair(round_cfg: Any, lock_block: int, activation_block: int) -> bool:
    """A note's ``(lock, act)`` is admissible only when ``lock`` is a boundary
    of the grid in force there and ``act`` is exactly the boundary after it —
    the one pair :func:`resolve_activation` could ever have produced. A pair
    that fails this is a typo or a bug on the signalling validator and must
    never be adopted: :func:`apply_activation` would refuse it on every poll
    and the node would stop tallying for good."""
    lock, act = int(lock_block), int(activation_block)
    if lock <= 0 or act <= lock:
        return False
    if lock % int(effective_epoch_blocks(round_cfg, lock)):
        return False
    return act == activation_block_for(round_cfg, lock)


def named_locks(
    feature: str,
    validators: list[ValidatorStake] | tuple[ValidatorStake, ...],
    signals: dict[str, str],
    *,
    round_cfg: Any = None,
) -> dict[tuple[int, int], float]:
    """Stake behind each admissible ``(lock, act)`` pair named in the notes
    of ``validators`` (already filtered for eligibility by the caller)."""
    by_block: dict[tuple[int, int], float] = {}
    for v in validators:
        sig = signal_for(signals.get(v.hotkey), feature)
        if sig is None or not sig.activation_block:
            continue
        if round_cfg is not None and not valid_pair(round_cfg, sig.lock_block,
                                                    sig.activation_block):
            log.warning("activation: ignoring an inadmissible note from %s (%r)",
                        v.hotkey, signals.get(v.hotkey))
            continue
        key = (sig.lock_block, sig.activation_block)
        by_block[key] = by_block.get(key, 0.0) + float(v.stake)
    return by_block


def agreed_activation(
    feature: str,
    validators: list[ValidatorStake] | tuple[ValidatorStake, ...],
    signals: dict[str, str],
    *,
    threshold: float,
    block: int,
    dormant_after_blocks: int = 0,
    round_cfg: Any = None,
) -> tuple[int, int] | None:
    """``(lock_block, activation_block)`` that validators holding ``threshold``
    of the eligible stake all name in their notes, else ``None``.

    This is how a node that missed the lock-in boundary (restart, late
    join, no archive node) catches up without re-reading old chain state:
    the decision is carried by the validators that made it. With
    ``round_cfg`` only admissible pairs count (:func:`valid_pair`).
    """
    elig = eligible_validators(validators, block=block, dormant_after_blocks=dormant_after_blocks)
    total = sum(float(v.stake) for v in elig)
    if total <= 0:
        return None
    by_block = named_locks(feature, elig, signals, round_cfg=round_cfg)
    if not by_block:
        return None
    key, stake = max(by_block.items(), key=lambda kv: (kv[1], -kv[0][1]))
    return key if stake / total >= float(threshold) else None


# ── block arithmetic ─────────────────────────────────────────────────────────


def latest_boundary(round_cfg: Any, block: int) -> int:
    """The last boundary at or before ``block`` on the grid in force there."""
    eb = int(effective_epoch_blocks(round_cfg, int(block)))
    return (int(block) // eb) * eb


def next_boundary(round_cfg: Any, boundary: int) -> int:
    """The boundary after ``boundary`` on the grid in force there."""
    return int(boundary) + int(effective_epoch_blocks(round_cfg, int(boundary)))


def activation_block_for(round_cfg: Any, lock_block: int) -> int:
    """The rollover for a lock-in observed at boundary ``lock_block``: the NEXT
    boundary on the grid in force at ``lock_block``. The round in which the
    count crossed finishes on the old rules; the one after starts on the
    new — the clean restart point."""
    eb = int(effective_epoch_blocks(round_cfg, int(lock_block)))
    return (int(lock_block) // eb + 1) * eb


# ── applying it to the loaded config ─────────────────────────────────────────


def configured_rollover(cfg: ChainConfig) -> int:
    """The rollover TYPED into chain.toml (0 = none). The loader guarantees
    every DEC-CA-0043 key names this one block. A block written by
    :func:`apply_activation` is not "typed" (``activation.resolved_block``
    marks it), so the resolver keeps treating it as the fleet's decision."""
    if int(getattr(cfg.activation, "resolved_block", 0) or 0):
        return 0
    return int(getattr(cfg.round, "rolling_from_block", 0) or 0)


def resolved_rollover(cfg: ChainConfig) -> int:
    """The rollover this config runs with from validator signals (0 = none
    applied / typed in)."""
    return int(getattr(cfg.activation, "resolved_block", 0) or 0)


@dataclass(frozen=True)
class FeatureSpec:
    """One feature the fleet decides on chain. ``typed_block`` > 0 is the
    owner override: the feature is not signalled, tallied or resolved."""
    name: str
    typed_block: int = 0

    @property
    def decided_on_chain(self) -> bool:
        return bool(self.name) and not int(self.typed_block)


def primary_feature(cfg: ChainConfig) -> FeatureSpec:
    """The DEC-CA-0043 rollover feature (``[activation] feature``)."""
    return FeatureSpec(name=str(cfg.activation.feature or ""), typed_block=configured_rollover(cfg))


def forfeit_feature_name(hotkeys, successor: str = "") -> str:
    """``forfeit-<sha256("<sorted hotkeys>|<successor>")[:8]>`` — the list
    AND the successor are the feature, so an edited list or a different
    successor is a fresh vote and two validators shipping different
    forfeitures never count each other."""
    hks = sorted({str(h).strip() for h in (hotkeys or ()) if str(h).strip()})
    if not hks:
        return ""
    body = ",".join(hks) + "|" + str(successor or "").strip()
    return "forfeit-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:8]


def configured_forfeit_block(cfg: ChainConfig) -> int:
    """The forfeiture block TYPED into chain.toml (0 = none / decided on chain).
    A block written by :func:`apply_forfeit_activation` is not typed
    (``activation.resolved_forfeit_block`` marks it)."""
    if int(getattr(cfg.activation, "resolved_forfeit_block", 0) or 0):
        return 0
    return int(getattr(cfg.scoring, "forfeit_from_block", 0) or 0)


def resolved_forfeit_block(cfg: ChainConfig) -> int:
    return int(getattr(cfg.activation, "resolved_forfeit_block", 0) or 0)


def forfeit_feature(cfg: ChainConfig) -> FeatureSpec | None:
    """The DEC-CA-0048 forfeiture as a feature the fleet decides: present
    when ``[activation]`` is on and ``[scoring] forfeit_hotkeys`` is
    non-empty; ``typed_block`` = the typed ``forfeit_from_block`` (the
    override — then nothing is signalled). ``None`` = no forfeiture in the
    config, or activation off (a typed block is then the only way)."""
    if not cfg.activation.enabled:
        return None
    name = forfeit_feature_name(getattr(cfg.scoring, "forfeit_hotkeys", ()) or (),
                                getattr(cfg.scoring, "forfeit_successor_hotkey", "") or "")
    if not name:
        return None
    return FeatureSpec(name=name, typed_block=configured_forfeit_block(cfg))


def forfeit_block_for(cfg: ChainConfig, activation_block: int) -> int:
    """The forfeiture block for a lock-in whose rollover would be
    ``activation_block``: that boundary itself — the settlement right after
    the one where the count crossed (owner 2026-09-28: the changeover follows
    the validators' upgrades as closely as the grid allows, no era of
    notice). A forfeiture therefore lands mid-era: the validators crown the
    successor there and hold (no duel judged) while the trainer retrains the
    running era's king leg for the successor (``_forfeit_switch`` judges an
    era by its LAST settlement); settlements after that leg lands are judged
    normally."""
    return int(activation_block)


def apply_forfeit_activation(cfg: ChainConfig, block: int) -> ChainConfig:
    """The config the owner would have typed for a forfeiture decided at
    rollover ``block``: ``[scoring] forfeit_from_block`` = that boundary
    (:func:`forfeit_block_for`). A typed block is returned unchanged (the
    owner override); a block already applied is idempotent; a different
    block after lock-in raises (one-way)."""
    block = int(block or 0)
    if block <= 0 or configured_forfeit_block(cfg):
        return cfg
    if not (getattr(cfg.scoring, "forfeit_hotkeys", ()) or ()):
        return cfg
    have = resolved_forfeit_block(cfg)
    if have == block:
        return cfg
    if have:
        raise ValueError(f"forfeiture {have} already applied; a lock-in is one-way and cannot "
                         f"move it to {block}")
    if block % int(effective_epoch_blocks(cfg.round, block - 1)):
        raise ValueError(f"forfeiture block {block} is not a settlement boundary")
    gate = forfeit_block_for(cfg, block)
    return replace(cfg, scoring=replace(cfg.scoring, forfeit_from_block=gate),
                   activation=replace(cfg.activation, resolved_forfeit_block=block))


def margin_v2_feature_name(start: float, end: float, warmup_blocks: int) -> str:
    """``margin-v2-<sha256("<start>|<end>|<blocks>")[:8]>`` — the values ARE
    the feature, so a changed bar is a fresh vote and two validators shipping
    different bars never count each other. ``""`` when the bar is unset."""
    if not (float(start) > 0.0 and float(end) > 0.0 and int(warmup_blocks) > 0):
        return ""
    body = f"{float(start)!r}|{float(end)!r}|{int(warmup_blocks)}"
    return "margin-v2-" + hashlib.sha256(body.encode("utf-8")).hexdigest()[:8]


def configured_margin_v2_block(cfg: ChainConfig) -> int:
    """The margin-v2 block TYPED into chain.toml (0 = none / decided on
    chain). A block written by :func:`apply_margin_v2_activation` is not typed
    (``activation.resolved_margin_v2_block`` marks it)."""
    if int(getattr(cfg.activation, "resolved_margin_v2_block", 0) or 0):
        return 0
    return int(getattr(cfg.scoring, "margin_v2_from_block", 0) or 0)


def resolved_margin_v2_block(cfg: ChainConfig) -> int:
    return int(getattr(cfg.activation, "resolved_margin_v2_block", 0) or 0)


def margin_v2_feature(cfg: ChainConfig) -> FeatureSpec | None:
    """The DEC-CA-0049 dethrone bar as a feature the fleet decides: present
    when ``[activation]`` is on and the v2 bar is defined; ``typed_block`` =
    the typed ``margin_v2_from_block`` (the override — then nothing is
    signalled). ``None`` = no v2 bar in the config, or activation off."""
    if not cfg.activation.enabled or not margin_v2_configured(cfg.scoring):
        return None
    sc = cfg.scoring
    name = margin_v2_feature_name(sc.win_margin_start_v2, sc.win_margin_end_v2,
                                  sc.margin_warmup_blocks_v2)
    if not name:
        return None
    return FeatureSpec(name=name, typed_block=configured_margin_v2_block(cfg))


def apply_margin_v2_activation(cfg: ChainConfig, block: int) -> ChainConfig:
    """The config the owner would have typed for a v2 bar decided at rollover
    ``block``: ``[scoring] margin_v2_from_block`` = that boundary — the one
    AFTER the lock-in boundary, like the DEC-CA-0043 rollover, so the round in
    which the count crossed is judged on the old bar. A typed block is
    returned unchanged (the owner override); a block already applied is
    idempotent; a different block after lock-in raises (one-way)."""
    block = int(block or 0)
    if block <= 0 or configured_margin_v2_block(cfg):
        return cfg
    if not margin_v2_configured(cfg.scoring):
        return cfg
    have = resolved_margin_v2_block(cfg)
    if have == block:
        return cfg
    if have:
        raise ValueError(f"margin v2 {have} already applied; a lock-in is one-way and cannot "
                         f"move it to {block}")
    if block % int(effective_epoch_blocks(cfg.round, block - 1)):
        raise ValueError(f"margin v2 block {block} is not a settlement boundary")
    return replace(cfg, scoring=replace(cfg.scoring, margin_v2_from_block=block),
                   activation=replace(cfg.activation, resolved_margin_v2_block=block))


def apply_activation(cfg: ChainConfig, block: int) -> ChainConfig:
    """The config the owner would have typed for a rollover at ``block``.

    Every DEC-CA-0043 key that is 0 becomes ``block``: ``rolling_from_block``,
    ``era_king_from_block``, ``tenure_blocks_from_block``, the grid switch
    (``epoch_blocks_prev`` = the loaded grid, ``epoch_blocks`` =
    ``[activation] epoch_blocks_after`` or the loaded grid,
    ``epoch_activation_block`` = ``block``) and — because the era king is
    judged under it — ``cohort_maxt_increment_from_block`` and
    ``cohort_maxt_from_block`` when still 0. A config that already names a
    rollover is returned unchanged (the owner override). The result passes
    the loader's own alignment check or ``ValueError`` is raised — a
    runtime rollover can never reach a state the loader would refuse.
    """
    block = int(block or 0)
    if block <= 0 or configured_rollover(cfg):
        return cfg
    if resolved_rollover(cfg) == block:
        return cfg                                   # already applied
    if resolved_rollover(cfg):
        raise ValueError(
            f"rollover {resolved_rollover(cfg)} already applied; a lock-in is one-way "
            f"and cannot move it to {block}")
    r, s = cfg.round, cfg.scoring
    grid_before = int(effective_epoch_blocks(r, block - 1)) if block > 0 else int(r.epoch_blocks)
    grid_after = int(cfg.activation.epoch_blocks_after or grid_before)
    if int(r.epoch_activation_block) and int(r.epoch_activation_block) != block:
        raise ValueError(
            f"[round] epoch_activation_block={r.epoch_activation_block} is already "
            f"scheduled; a resolved rollover at {block} cannot move it")
    if int(r.era_settlements or 0) < 1:
        raise ValueError(
            "[round] era_settlements must be >= 1 for a rollover (the loader refuses a "
            "typed-in one without it; a resolved one is refused the same way)")
    new_round = replace(
        r,
        rolling_from_block=block,
        epoch_blocks=grid_after,
        epoch_blocks_prev=grid_before,
        epoch_activation_block=block,
    )
    new_scoring = replace(
        s,
        era_king_from_block=block,
        tenure_blocks_from_block=block,
        cohort_maxt_from_block=int(s.cohort_maxt_from_block or block),
        cohort_maxt_increment_from_block=int(s.cohort_maxt_increment_from_block or block),
    )
    check_rollover_alignment(
        rolling_from_block=new_round.rolling_from_block,
        era_king_from_block=new_scoring.era_king_from_block,
        tenure_blocks_from_block=new_scoring.tenure_blocks_from_block,
        epoch_blocks=new_round.epoch_blocks,
        epoch_blocks_prev=new_round.epoch_blocks_prev,
        epoch_activation_block=new_round.epoch_activation_block,
        era_settlements=new_round.era_settlements,
        cohort_maxt_from_block=new_scoring.cohort_maxt_from_block,
        cohort_maxt_increment_from_block=new_scoring.cohort_maxt_increment_from_block,
        funded_pods=new_round.funded_pods,
        funded_king_rent=new_round.funded_king_rent,
    )
    return replace(cfg, round=new_round, scoring=new_scoring,
                   activation=replace(cfg.activation, resolved_block=block))


# ── persistence ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ActivationRecord:
    """What a node has decided so far. ``activation_block`` 0 = not locked in;
    ``last_checked_boundary`` = the last boundary this node tallied (so a
    poll loop tallies each boundary once). ``source`` says where the block
    came from: ``config`` / ``tally`` / ``signals`` / ``state``."""

    feature: str = ""
    lock_block: int = 0
    activation_block: int = 0
    last_checked_boundary: int = 0
    source: str = ""

    @property
    def locked(self) -> bool:
        return self.activation_block > 0


class ActivationStore:
    """A small JSON file beside the service's state (best-effort I/O: a
    missing or unreadable file reads as a blank record)."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path is not None else None

    def load(self) -> ActivationRecord:
        if self.path is None or not self.path.exists():
            return ActivationRecord()
        try:
            obj = json.loads(self.path.read_text(encoding="utf-8"))
            return ActivationRecord(
                feature=str(obj.get("feature", "") or ""),
                lock_block=int(obj.get("lock_block", 0) or 0),
                activation_block=int(obj.get("activation_block", 0) or 0),
                last_checked_boundary=int(obj.get("last_checked_boundary", 0) or 0),
                source=str(obj.get("source", "") or ""),
            )
        except (OSError, ValueError, TypeError) as e:
            log.warning("activation state %s unreadable (%s); starting blank", self.path, e)
            return ActivationRecord()

    def save(self, rec: ActivationRecord) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(asdict(rec), indent=2, sort_keys=True),
                                 encoding="utf-8")
        except OSError as e:
            log.warning("activation state %s not written: %s", self.path, e)


# ── the resolver ─────────────────────────────────────────────────────────────


@dataclass
class Resolution:
    """One resolver pass. ``record`` is the (possibly advanced) record to
    persist; ``tally`` the boundary tally when one was taken this pass."""

    record: ActivationRecord
    tally: Tally | None = None
    changed: bool = False


def _read_chain(client: Any, block: int | None) -> tuple[list[ValidatorStake], dict[str, str]]:
    validators = list(client.validator_stakes(block=block))
    signals = dict(client.read_plain_commitments(block=block))
    return validators, signals


def resolve_activation(
    cfg: ChainConfig, client: Any, *, now_block: int, record: ActivationRecord,
    feature: FeatureSpec | None = None,
) -> Resolution:
    """Advance ``record`` one step against the chain.

    Order: the typed-in rollover (nothing to resolve); an already locked-in
    record (one-way); the validators' own notes (a ``threshold``-stake
    agreement on one block); else the tally at the latest boundary at or
    before ``now_block`` — once per boundary — locking in when it crosses.
    Never raises on a chain read failure: the pass is a no-op and the
    caller retries next poll.
    """
    ac = cfg.activation
    spec = feature if feature is not None else primary_feature(cfg)
    feature = spec.name
    if not feature:
        return Resolution(record=record)
    typed = int(spec.typed_block)
    if typed:
        if record.activation_block != typed or record.source != "config":
            return Resolution(record=replace(record, feature=feature, activation_block=typed,
                                             source="config"), changed=True)
        return Resolution(record=record)
    if record.feature and record.feature != feature:
        record = ActivationRecord()                  # another feature's decision: blank
    if record.source == "config":
        # The typed-in rollover was withdrawn from chain.toml: a typed block
        # was never the chain's decision, so the record must not stay
        # "locked" on it — the chain decides again from here.
        log.warning("activation: typed-in block %d for %s withdrawn from the config; "
                    "resolving from validator signals again", record.activation_block, feature)
        record = ActivationRecord()
        return Resolution(record=record, changed=True)
    if record.locked:
        return Resolution(record=record)

    # Boundaries are tallied IN ORDER: the next one after the last this node
    # counted (or the latest one for a fresh record). A boundary this node
    # could not read is retried, never skipped for a later one — skipping
    # would let a node that was down across the lock-in boundary count a
    # later boundary and name a different rollover (the fork this exists to
    # prevent). A node stranded on a pruned endpoint relies on the notes.
    latest = latest_boundary(cfg.round, int(now_block))
    if record.last_checked_boundary:
        boundary = next_boundary(cfg.round, record.last_checked_boundary)
    else:
        boundary = latest
    if boundary > latest:
        return Resolution(record=record)             # nothing new; no chain read

    # A new boundary (or a fresh record). Notes first: the validators that
    # already locked in carry the decision, and a decision made elsewhere
    # beats this node's own count.
    try:
        live_validators, live_signals = _read_chain(client, None)
    except Exception as e:  # noqa: BLE001 — chain flake: retry next poll
        log.warning("activation: chain read failed (%s); retrying next poll", e)
        return Resolution(record=record)
    validators, signals = live_validators, live_signals
    agreed = agreed_activation(feature, validators, signals, threshold=ac.threshold,
                               block=int(now_block), dormant_after_blocks=ac.dormant_after_blocks,
                               round_cfg=cfg.round)
    if agreed is not None:
        lock, act = agreed
        rec = replace(record, feature=feature, lock_block=int(lock), activation_block=int(act),
                      last_checked_boundary=max(record.last_checked_boundary, int(lock)),
                      source="signals")
        log.info("activation: %s locked in at block %d per validator notes — rollover at %d",
                 feature, lock, act)
        return Resolution(record=rec, changed=True)

    # The boundary itself, read AS OF the boundary block — the only read
    # that gives every node the same stake. A node whose endpoint cannot
    # serve that block does NOT count from a later view (two nodes counting
    # different states is exactly the fork this exists to prevent): it
    # retries THIS boundary next poll and, failing that, adopts the fleet's
    # decision from the notes.
    try:
        validators, signals = _read_chain(client, boundary)
    except Exception as e:  # noqa: BLE001
        if int(now_block) - boundary >= PRUNED_AFTER_BLOCKS and not named_locks(
                feature, eligible_validators(live_validators, block=int(now_block),
                                             dormant_after_blocks=ac.dormant_after_blocks),
                live_signals, round_cfg=cfg.round):
            # The boundary is gone from this endpoint's state AND no
            # validator's note names a lock-in: nobody crossed there (a
            # locked-in validator rewrites its note within a poll), so the
            # boundary is skipped rather than retried forever. A node that
            # restarted across a boundary on a pruned endpoint would
            # otherwise never count again; a fresh record whose latest
            # boundary is already old would never count at all.
            log.warning("activation: boundary %d is unreadable here (%s) and no validator "
                        "names a lock-in — skipping it; the next boundary is counted "
                        "as of its own block", boundary, e)
            return Resolution(record=replace(record, feature=feature,
                                             last_checked_boundary=boundary), changed=True)
        log.warning("activation: as-of read at boundary %d failed (%s); not counting from "
                    "a later view — retrying this boundary next poll (%d behind the latest)",
                    boundary, e, latest - boundary)
        return Resolution(record=record)
    t = tally(feature, validators, signals, threshold=ac.threshold, block=boundary,
              dormant_after_blocks=ac.dormant_after_blocks)
    if not t.eligible:
        # An empty metagraph is a failed read wearing a 0/0 tally (the SDK
        # returns no neurons on a falsy runtime result). Counting it would
        # consume the boundary while peers read real state there.
        log.warning("activation: no eligible validator at boundary %d — treating as a "
                    "failed read; retrying this boundary next poll", boundary)
        return Resolution(record=record)
    rec = replace(record, feature=feature, last_checked_boundary=boundary)
    if t.locked:
        act = activation_block_for(cfg.round, boundary)
        # The signers this node just counted may have locked in EARLIER
        # (their notes name the block); when validators holding ``threshold``
        # of the signed stake name one earlier admissible pair, that is the
        # fleet's decision and this node's later boundary is not.
        earlier = _earlier_lock_named_by_signers(cfg, feature, t, validators, signals,
                                                 before=boundary)
        if earlier is not None:
            lock, act = earlier
            rec = replace(rec, lock_block=int(lock), activation_block=int(act), source="signals")
            log.info("activation: %s crossed at boundary %d but the signers already locked "
                     "in at %d — adopting their rollover at %d", feature, boundary, lock, act)
            return Resolution(record=rec, tally=t, changed=True)
        rec = replace(rec, lock_block=boundary, activation_block=act, source="tally")
        log.info("activation: %s LOCKED IN at boundary %d (%.1f%% of eligible stake, %d/%d "
                 "validators) — rollover at block %d", feature, boundary, 100 * t.ratio,
                 len(t.signed), len(t.eligible), act)
    else:
        log.info("activation: %s at boundary %d: %.1f%% of eligible stake signed (%d/%d "
                 "validators, threshold %.0f%%) — not yet", feature, boundary, 100 * t.ratio,
                 len(t.signed), len(t.eligible), 100 * ac.threshold)
    return Resolution(record=rec, tally=t, changed=True)


def _earlier_lock_named_by_signers(
    cfg: ChainConfig, feature: str, t: Tally, validators: list[ValidatorStake],
    signals: dict[str, str], *, before: int,
) -> tuple[int, int] | None:
    """The earliest admissible ``(lock, act)`` with ``lock < before`` named by
    validators holding ``threshold`` of the SIGNED stake in ``t``."""
    if t.signed_stake <= 0:
        return None
    signed = {hk for hk in t.signed}
    named = named_locks(feature, [v for v in validators if v.hotkey in signed], signals,
                        round_cfg=cfg.round)
    named = {k: s for k, s in named.items() if k[0] < int(before)}
    if not named:
        return None
    key, stake = min(named.items(), key=lambda kv: (kv[0][0], -kv[1]))
    return key if stake / t.signed_stake >= float(cfg.activation.threshold) else None


def record_for(cfg: ChainConfig, record: ActivationRecord,
               feature: FeatureSpec | None = None) -> ActivationRecord:
    """``record`` if it belongs to ``feature`` (default: this config's primary
    feature), else a blank one — a persisted decision for a renamed feature
    (or an edited forfeit list) must never arm anything."""
    name = feature.name if feature is not None else str(cfg.activation.feature or "")
    if record.feature and record.feature != name:
        log.warning("activation: ignoring a persisted record for feature %r (config runs %r)",
                    record.feature, name)
        return ActivationRecord()
    return record


class ActivationWatcher:
    """Per-tick resolution for a service without its own hook (the
    provisioner): holds the record, resolves once per boundary, and hands
    back the armed config when a lock-in applies. ``store_path`` None ⇒
    nothing is written (the provisioner re-resolves from the notes on a
    restart instead of sharing the trainer's record)."""

    def __init__(self, cfg: ChainConfig, *, store_path: Path | str | None = None) -> None:
        self.cfg = cfg
        self.store = ActivationStore(store_path)
        self.record = record_for(cfg, self.store.load())
        self.tally: Tally | None = None

    def tick(self, client: Any, block: int) -> ChainConfig | None:
        """Returns the newly armed config when this tick applied a lock-in,
        else ``None``. Never raises."""
        if not self.cfg.activation.enabled or configured_rollover(self.cfg):
            return None
        try:
            res = resolve_activation(self.cfg, client, now_block=int(block), record=self.record)
            if res.tally is not None:
                self.tally = res.tally
            if res.changed:
                self.record = res.record
                self.store.save(res.record)
            if self.record.locked and self.record.source != "config":
                new = apply_activation(self.cfg, self.record.activation_block)
                if new is not self.cfg:
                    self.cfg = new
                    log.warning("activation: DEC-CA-0043 rollover ARMED at block %d (via %s)",
                                self.record.activation_block, self.record.source)
                    return new
        except Exception as e:  # noqa: BLE001
            log.warning("activation step failed (%s); retrying next tick", e)
        return None


def startup_activation(
    cfg: ChainConfig, client: Any, *, store_path: Path | str | None,
) -> ChainConfig:
    """One-shot startup resolution: one watcher tick at the current block.
    Best-effort — any failure returns ``cfg`` unchanged."""
    if not cfg.activation.enabled or configured_rollover(cfg):
        return cfg
    try:
        now_block = int(client.current_block())
    except Exception as e:  # noqa: BLE001
        log.warning("activation: startup resolution skipped (%s); running the typed-in "
                    "config", e)
        return cfg
    return ActivationWatcher(cfg, store_path=store_path).tick(client, now_block) or cfg


def apply_receipt_activation(cfg: ChainConfig, receipt: Any) -> ChainConfig:
    """For the audit: the config a receipt was judged under — its recorded
    ``activation_block`` applied when the loaded config names no rollover
    and runs the feature. A receipt without the field (pre-DEC-CA-0045, or
    a typed-in rollover) replays under the loaded config as before. A block
    the loaded config cannot apply leaves it unchanged; the ``activation``
    check reports why."""
    block = int(getattr(receipt, "activation_block", 0) or 0)
    if block and not configured_rollover(cfg) and cfg.activation.enabled:
        try:
            cfg = apply_activation(cfg, block)
        except ValueError as e:
            log.warning("activation: receipt block %d not applied to the audit config (%s)",
                        block, e)
    # DEC-CA-0049: a receipt judged under a fleet-decided v2 bar records the
    # block; replay under it (a typed block in the loaded config wins).
    mv2 = int(getattr(receipt, "margin_v2_block", 0) or 0)
    if mv2 and not configured_margin_v2_block(cfg) and margin_v2_configured(cfg.scoring):
        try:
            cfg = apply_margin_v2_activation(cfg, mv2)
        except ValueError as e:
            log.warning("activation: receipt margin-v2 block %d not applied to the audit "
                        "config (%s)", mv2, e)
    return cfg


def _segment_for(spec: FeatureSpec, record: ActivationRecord | None) -> ReadySignal:
    if record is not None and record.locked and record.lock_block:
        return ReadySignal(spec.name, lock_block=record.lock_block,
                           activation_block=record.activation_block)
    return ReadySignal(spec.name)


def own_signal_payload(cfg: ChainConfig, record: ActivationRecord,
                       forfeit_record: ActivationRecord | None = None,
                       margin_v2_record: ActivationRecord | None = None) -> str | None:
    """The note this node should have on chain right now (``None`` when
    signalling is off): plain readiness until lock-in, then the agreed block
    — per segment. With no forfeiture decided on chain the note is exactly
    the single-segment v1 note; a typed-in rollover AND no forfeiture vote
    makes it inert (``None``)."""
    if not cfg.activation.enabled:
        return None
    primary = primary_feature(cfg)
    forfeit = forfeit_feature(cfg)
    extras: list[ReadySignal] = []
    if forfeit is not None and forfeit.decided_on_chain:
        extras.append(_segment_for(forfeit, forfeit_record))
    mv2 = margin_v2_feature(cfg)
    if mv2 is not None and mv2.decided_on_chain:
        extras.append(_segment_for(mv2, margin_v2_record))
    if not primary.decided_on_chain and not extras:
        return None                                  # typed-in rollover: the note is inert
    head = _segment_for(primary, record) if primary.decided_on_chain else ReadySignal(primary.name)
    return format_signals(head, *extras)


def ensure_signal(client: Any, cfg: ChainConfig, record: ActivationRecord, *,
                  hotkey: str, current: dict[str, str] | None = None,
                  forfeit_record: ActivationRecord | None = None,
                  margin_v2_record: ActivationRecord | None = None) -> bool:
    """Write this validator's note unless the chain already carries it.
    Returns True when a write happened. Never raises (a failed write is
    retried on the next call)."""
    want = own_signal_payload(cfg, record, forfeit_record, margin_v2_record)
    if want is None:
        return False
    try:
        have = current if current is not None else dict(client.read_plain_commitments())
        if have.get(hotkey) == want:
            return False
        client.set_plain_commitment(want)
        log.info("activation: signalled %s on chain", want)
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("activation: signal write failed (%s); retrying later", e)
        return False


def summary(cfg: ChainConfig, record: ActivationRecord, t: Tally | None,
            forfeit_record: ActivationRecord | None = None,
            forfeit_tally: Tally | None = None,
            margin_v2_record: ActivationRecord | None = None,
            margin_v2_tally: Tally | None = None) -> dict:
    """Presentational block for ``status/chain.json`` / dashboards."""
    out: dict = {
        "feature": cfg.activation.feature,
        "threshold": float(cfg.activation.threshold),
        "lock_block": int(record.lock_block),
        "activation_block": int(record.activation_block),
        "source": record.source,
    }
    if t is not None:
        out["tally"] = t.to_json()
    forfeit = forfeit_feature(cfg)
    if forfeit is not None:
        rec = forfeit_record or ActivationRecord()
        fo: dict = {
            "feature": forfeit.name,
            "hotkeys": len(getattr(cfg.scoring, "forfeit_hotkeys", ()) or ()),
            "typed_block": int(forfeit.typed_block),
            "lock_block": int(rec.lock_block),
            "activation_block": int(rec.activation_block),
            "source": rec.source,
        }
        if forfeit_tally is not None:
            fo["tally"] = forfeit_tally.to_json()
        out["forfeit"] = fo
    mv2 = margin_v2_feature(cfg)
    if mv2 is not None:
        rec = margin_v2_record or ActivationRecord()
        mo: dict = {
            "feature": mv2.name,
            "win_margin_start_v2": float(cfg.scoring.win_margin_start_v2),
            "win_margin_end_v2": float(cfg.scoring.win_margin_end_v2),
            "margin_warmup_blocks_v2": int(cfg.scoring.margin_warmup_blocks_v2),
            "typed_block": int(mv2.typed_block),
            "lock_block": int(rec.lock_block),
            "activation_block": int(rec.activation_block),
            "source": rec.source,
        }
        if margin_v2_tally is not None:
            mo["tally"] = margin_v2_tally.to_json()
        out["margin_v2"] = mo
    return out
