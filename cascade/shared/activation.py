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
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import ChainConfig, check_rollover_alignment, effective_epoch_blocks

log = logging.getLogger("cascade.activation")

SIGNAL_PREFIX = "cascade-ready:"
SIGNAL_VERSION = 1


# ── the on-chain note ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReadySignal:
    """One validator's parsed note. ``lock_block`` / ``activation_block`` are
    0 until that validator has itself seen lock-in."""

    feature: str
    lock_block: int = 0
    activation_block: int = 0


def format_signal(feature: str, *, lock_block: int = 0, activation_block: int = 0) -> str:
    """``cascade-ready:1:<feature>:<lock_block>:<activation_block>``."""
    feat = str(feature or "").strip()
    if not feat or ":" in feat or any(c.isspace() for c in feat):
        raise ValueError(f"feature must be a non-empty token without ':' or spaces; got {feat!r}")
    return f"{SIGNAL_PREFIX}{SIGNAL_VERSION}:{feat}:{int(lock_block)}:{int(activation_block)}"


def is_signal_payload(payload: object) -> bool:
    return isinstance(payload, str) and payload.startswith(SIGNAL_PREFIX)


def parse_signal(payload: object) -> ReadySignal | None:
    """Parse a note; ``None`` for anything that is not a well-formed v1 note
    (a future version, a generator pointer, garbage)."""
    if not is_signal_payload(payload):
        return None
    parts = str(payload)[len(SIGNAL_PREFIX):].split(":")
    if len(parts) != 4:
        return None
    ver, feat, lock, act = parts
    try:
        if int(ver) != SIGNAL_VERSION:
            return None
        lock_b, act_b = int(lock), int(act)
    except ValueError:
        return None
    if not feat or lock_b < 0 or act_b < 0 or (act_b and not lock_b) or (lock_b and act_b <= lock_b):
        return None
    return ReadySignal(feature=feat, lock_block=lock_b, activation_block=act_b)


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
        sig = parse_signal(signals.get(v.hotkey))
        if sig is not None and sig.feature == feature:
            signed.append(v.hotkey)
            signed_stake += float(v.stake)
    return Tally(
        feature=str(feature), block=int(block), threshold=float(threshold),
        signed_stake=signed_stake, total_stake=total,
        signed=tuple(sorted(signed)), eligible=tuple(sorted(v.hotkey for v in elig)),
    )


def agreed_activation(
    feature: str,
    validators: list[ValidatorStake] | tuple[ValidatorStake, ...],
    signals: dict[str, str],
    *,
    threshold: float,
    block: int,
    dormant_after_blocks: int = 0,
) -> tuple[int, int] | None:
    """``(lock_block, activation_block)`` that validators holding ``threshold``
    of the eligible stake all name in their notes, else ``None``.

    This is how a node that missed the lock-in boundary (restart, late
    join, no archive node) catches up without re-reading old chain state:
    the decision is carried by the validators that made it.
    """
    elig = eligible_validators(validators, block=block, dormant_after_blocks=dormant_after_blocks)
    total = sum(float(v.stake) for v in elig)
    if total <= 0:
        return None
    by_block: dict[tuple[int, int], float] = {}
    for v in elig:
        sig = parse_signal(signals.get(v.hotkey))
        if sig is None or sig.feature != feature or not sig.activation_block:
            continue
        key = (sig.lock_block, sig.activation_block)
        by_block[key] = by_block.get(key, 0.0) + float(v.stake)
    if not by_block:
        return None
    key, stake = max(by_block.items(), key=lambda kv: (kv[1], -kv[0][1]))
    return key if stake / total >= float(threshold) else None


# ── block arithmetic ─────────────────────────────────────────────────────────


def latest_boundary(round_cfg: Any, block: int) -> int:
    """The last boundary at or before ``block`` on the grid in force there."""
    eb = int(effective_epoch_blocks(round_cfg, int(block)))
    return (int(block) // eb) * eb


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
    feature = ac.feature
    if not feature:
        return Resolution(record=record)
    typed = configured_rollover(cfg)
    if typed:
        if record.activation_block != typed or record.source != "config":
            return Resolution(record=replace(record, feature=feature, activation_block=typed,
                                             source="config"), changed=True)
        return Resolution(record=record)
    if record.feature and record.feature != feature:
        record = ActivationRecord()                  # another feature's decision: blank
    if record.locked:
        return Resolution(record=record)

    boundary = latest_boundary(cfg.round, int(now_block))
    if boundary <= record.last_checked_boundary:
        return Resolution(record=record)             # this boundary is done; no chain read

    # A new boundary (or a fresh record). Notes first: the validators that
    # already locked in carry the decision, and a decision made elsewhere
    # beats this node's own count.
    try:
        validators, signals = _read_chain(client, None)
    except Exception as e:  # noqa: BLE001 — chain flake: retry next poll
        log.warning("activation: chain read failed (%s); retrying next poll", e)
        return Resolution(record=record)
    agreed = agreed_activation(feature, validators, signals, threshold=ac.threshold,
                               block=int(now_block), dormant_after_blocks=ac.dormant_after_blocks)
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
    # retries next poll and, failing that, adopts the fleet's decision from
    # the notes at the next boundary.
    try:
        validators, signals = _read_chain(client, boundary)
    except Exception as e:  # noqa: BLE001
        log.warning("activation: as-of read at boundary %d failed (%s); not counting from "
                    "a later view — retrying next poll", boundary, e)
        return Resolution(record=record)
    t = tally(feature, validators, signals, threshold=ac.threshold, block=boundary,
              dormant_after_blocks=ac.dormant_after_blocks)
    rec = replace(record, feature=feature, last_checked_boundary=boundary)
    if t.locked:
        act = activation_block_for(cfg.round, boundary)
        rec = replace(rec, lock_block=boundary, activation_block=act, source="tally")
        log.info("activation: %s LOCKED IN at boundary %d (%.1f%% of eligible stake, %d/%d "
                 "validators) — rollover at block %d", feature, boundary, 100 * t.ratio,
                 len(t.signed), len(t.eligible), act)
    else:
        log.info("activation: %s at boundary %d: %.1f%% of eligible stake signed (%d/%d "
                 "validators, threshold %.0f%%) — not yet", feature, boundary, 100 * t.ratio,
                 len(t.signed), len(t.eligible), 100 * ac.threshold)
    return Resolution(record=rec, tally=t, changed=True)


def record_for(cfg: ChainConfig, record: ActivationRecord) -> ActivationRecord:
    """``record`` if it belongs to this config's feature, else a blank one —
    a persisted decision for a renamed feature must never arm anything."""
    if record.feature and record.feature != cfg.activation.feature:
        log.warning("activation: ignoring a persisted record for feature %r (config runs %r)",
                    record.feature, cfg.activation.feature)
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
    if not block or configured_rollover(cfg) or not cfg.activation.enabled:
        return cfg
    try:
        return apply_activation(cfg, block)
    except ValueError as e:
        log.warning("activation: receipt block %d not applied to the audit config (%s)",
                    block, e)
        return cfg


def own_signal_payload(cfg: ChainConfig, record: ActivationRecord) -> str | None:
    """The note this node should have on chain right now (``None`` when
    signalling is off): plain readiness until lock-in, then the agreed block."""
    if not cfg.activation.enabled:
        return None
    if record.locked and record.lock_block:
        return format_signal(cfg.activation.feature, lock_block=record.lock_block,
                             activation_block=record.activation_block)
    return format_signal(cfg.activation.feature)


def ensure_signal(client: Any, cfg: ChainConfig, record: ActivationRecord, *,
                  hotkey: str, current: dict[str, str] | None = None) -> bool:
    """Write this validator's note unless the chain already carries it.
    Returns True when a write happened. Never raises (a failed write is
    retried on the next call)."""
    want = own_signal_payload(cfg, record)
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


def summary(cfg: ChainConfig, record: ActivationRecord, t: Tally | None) -> dict:
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
    return out
