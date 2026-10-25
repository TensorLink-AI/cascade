"""Lane B (DEC-CA-0050) consensus state: the live domain winners and the
combined lane-A + lane-B weight vector.

Pure and deterministic — every validator derives the same state from the
same receipts, and ``cascade-audit`` replays it. Payout age is counted in
ERAS (``era_blocks`` = the grid in force × ``era_settlements``), from the
settlement that merged the win (age 0 = the first era-length after it), so a
win pays in full for its RECORDED ``full_eras`` eras, then decays, and stops
paying at ``win_ttl_eras`` (its slot burns; the merged weights stay in the
king).

Gain-tiered full-pay period (owner 2026-10-20): ``full_eras`` is chosen AT
THE WIN from the win's target gain LCB (``target_lcb``) by ``[lane_b]
full_pay_tiers`` (:func:`cascade.eval.expert_lane.full_pay`) and recorded on
the :class:`DomainWin` (``tier`` / ``full_eras`` / ``below_floor``) — never
re-evaluated, so a later change to the tiers touches only later wins. A
multi-target leg's wins are tiered per target, each by its own target LCB.

Slots (DEC-CA-0059) are the ``[lane_b]`` TARGETS — the domains, then one
``horizon:<h>`` per scored horizon (none on a chain without a scored ladder,
where the slot list is exactly ``domains``). ``DomainWin.domain`` names the
target. A multi-target leg that won several targets holds several wins, each
on its own slot and decay clock, all naming one judged checkpoint (one
merge).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from ..eval.expert_lane import FullPay, LaneBWin, reward_multiplier, split_emissions
from .chain import decayed_share_vector


@dataclass(frozen=True)
class DomainWin:
    domain: str
    hotkey: str
    uid: int
    round_id: str
    merge_block: int          # the settlement boundary block that merged it
    gen_ref: str = ""
    target_lcb: float = 0.0
    # merge provenance: the judged checkpoint, the king it was rebased onto,
    # and its expert rows — the delta (rebased − king) on those rows is what
    # the next era's init carries, re-derivable by every validator
    rebased_pointer: str = ""
    king_pointer: str = ""
    selection: dict = field(default_factory=dict)
    # gain-tiered full-pay period, fixed at the win from ``target_lcb``
    # (:func:`cascade.eval.expert_lane.full_pay`); None = untiered (refused
    # by the payout — every recorded win carries its tier)
    tier: float | None = None
    full_eras: int | None = None
    below_floor: bool = False
    commit_block: int = 0
    targets: tuple[str, ...] = ()
    qualified_block: int = 0

    def __post_init__(self):
        object.__setattr__(self, "targets", tuple(self.targets))

    def with_full_pay(self, fp: FullPay) -> DomainWin:
        from dataclasses import replace

        return replace(self, tier=float(fp.tier), full_eras=int(fp.full_eras),
                       below_floor=bool(fp.below_floor))

    def full_pay_record(self) -> dict:
        """``{gain_lcb, tier, full_eras, below_floor}`` — what the payout
        record and the settlement state stamp."""
        return {"gain_lcb": float(self.target_lcb), "tier": self.tier,
                "full_eras": self.full_eras, "below_floor": bool(self.below_floor)}


def win_full_eras(win) -> int:
    """A recorded win's full-pay period; a win without one is refused (the
    payout is computed only from recorded fields)."""
    n = getattr(win, "full_eras", None)
    if n is None:
        raise ValueError(f"win {getattr(win, 'hotkey', '?')!r} carries no full-pay tier "
                         "(full_eras)")
    return int(n)


def merge_identity(m) -> tuple:
    """A merge record's identity — ``(rebased_pointer, king_pointer, rows)``:
    what makes two records the SAME delta (DEC-CA-0059 merge-once)."""
    get = m.get if isinstance(m, dict) else (lambda k, d=None: getattr(m, k, d))
    sel = get("selection", {}) or {}
    return (str(get("rebased_pointer", "") or ""), str(get("king_pointer", "") or ""),
            tuple(sorted((int(k), tuple(sorted(int(x) for x in v))) for k, v in sel.items())))


def merge_identities(merges) -> set[tuple]:
    """The distinct merge identities in ``merges`` (a list of merge records
    or ``DomainWin``s)."""
    return {merge_identity(m) for m in merges}


HISTORY_KEEP = 64
APPLIED_KEEP = 4096


@dataclass
class LaneBState:
    wins: dict[str, DomainWin] = field(default_factory=dict)
    # every recorded win (bounded) — merge provenance outlives a slot takeover
    history: list[DomainWin] = field(default_factory=list)
    # settlements whose expert manifest has been APPLIED (round id -> block):
    # applying is idempotent, so a crash between persisting this state and
    # the lane-A marker re-processes the round without double-recording.
    applied: dict[str, int] = field(default_factory=dict)
    # the last settled lane-B boundary block — the payout decay clock a
    # restarted validator resumes from (never the head floored)
    settled_block: int | None = None
    # the last settlement folded IN CHAIN ORDER (direct or catch-up): a
    # settlement whose ``prev_round_id`` differs starts a catch-up walk
    last_round: str | None = None
    # Individual wins qualify a delta; they do not replace the paid holder.
    qualifications: list[DomainWin] = field(default_factory=list)
    # Durable merge-once clock, including after a paid holder is replaced.
    accepted: dict[str, int] = field(default_factory=dict)

    def qualify(self, win: DomainWin) -> None:
        if win not in self.qualifications:
            self.qualifications.append(win)

    def qualified_for(self, merge: dict) -> list[DomainWin]:
        identity = merge_identity(merge)
        return [w for w in self.qualifications
                if merge_identity(w) == identity and w.round_id == str(merge.get("round_id", ""))]

    def accept(self, merge: dict, wins: list[DomainWin], block: int) -> None:
        key = json.dumps(merge_identity(merge), separators=(",", ":"))
        if key in self.accepted:
            return
        from dataclasses import replace

        for win in wins:
            accepted = replace(win, merge_block=int(block))
            current = self.wins.get(win.domain)
            if current is None or current.merge_block <= int(block):
                self.wins[win.domain] = accepted
            self.record(accepted)
        self.accepted[key] = int(block)

    def record(self, win: DomainWin) -> None:
        """A newer merged win on a domain takes its slot (the old win stops
        earning — owner: the cap is per win). An OLDER win (an out-of-order
        catch-up or an on-demand era-init judge) enters the history only, so
        the folded state does not depend on the order settlements arrive."""
        cur = self.wins.get(win.domain)
        if cur is None or (cur.merge_block, cur.round_id) <= (win.merge_block, win.round_id):
            self.wins[win.domain] = win
        if win not in self.history:
            self.history.append(win)
            self.history.sort(key=lambda w: (w.merge_block, w.round_id, w.domain))
            del self.history[:-HISTORY_KEEP]

    def mark_applied(self, round_id: str, block: int, *, in_chain: bool = True) -> None:
        """Record ``round_id`` (settled at ``block``) as applied. ``in_chain``
        (the direct path and catch-up, oldest first) also advances the chain
        position; an on-demand judge does not."""
        self.applied[str(round_id)] = int(block)
        if len(self.applied) > APPLIED_KEEP:
            for rid, _ in sorted(self.applied.items(), key=lambda kv: kv[1])[
                    : len(self.applied) - APPLIED_KEEP]:
                del self.applied[rid]
        if self.settled_block is None or int(block) > self.settled_block:
            self.settled_block = int(block)
        if in_chain:
            self.last_round = str(round_id)

    def has_merge(self, rebased_pointer: str, king_pointer: str, selection: dict) -> bool:
        sel = {str(k): sorted(int(x) for x in v) for k, v in selection.items()}
        return any(w.rebased_pointer == rebased_pointer and w.king_pointer == king_pointer
                   and {str(k): sorted(v) for k, v in w.selection.items()} == sel
                   for w in self.history)

    def eras_since(self, win: DomainWin, block: int, era_blocks: int) -> int:
        """Whole eras between the merging settlement and ``block``."""
        if era_blocks <= 0:
            raise ValueError("era_blocks must be > 0")
        return max(0, (int(block) - int(win.merge_block)) // int(era_blocks))

    def to_json(self) -> str:
        return json.dumps({"wins": {d: asdict(w) for d, w in sorted(self.wins.items())},
                           "history": [asdict(w) for w in self.history],
                           "applied": dict(sorted(self.applied.items())),
                           "settled_block": self.settled_block,
                           "last_round": self.last_round,
                           "qualifications": [asdict(w) for w in self.qualifications],
                           "accepted": dict(sorted(self.accepted.items()))}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> LaneBState:
        """Parse a persisted state. An EMPTY or torn file raises (the caller
        rebuilds by catch-up) — never a silent empty state."""
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("lane-B state is not a JSON object")
        sb = obj.get("settled_block")
        return cls(wins={d: DomainWin(**w) for d, w in obj.get("wins", {}).items()},
                   history=[DomainWin(**w) for w in obj.get("history", [])],
                   applied={str(k): int(v) for k, v in (obj.get("applied") or {}).items()},
                   settled_block=None if sb is None else int(sb),
                   last_round=obj.get("last_round"),
                   qualifications=[DomainWin(**w) for w in obj.get("qualifications", [])],
                   accepted={str(k): int(v) for k, v in obj.get("accepted", {}).items()})


def save_state_atomic(state: LaneBState, path) -> None:
    """tmp + fsync + rename (+ directory fsync): a crash leaves either the
    old file or the new one, never a torn one."""
    import os
    from pathlib import Path

    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    data = state.to_json().encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    try:
        fd = os.open(str(p.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def win_expired(age_eras: int, cfg) -> bool:
    """A win at ``age_eras`` has reached ``[lane_b] win_ttl_eras`` (0 = never)."""
    ttl = int(getattr(cfg, "win_ttl_eras", 0) or 0)
    return ttl > 0 and int(age_eras) >= ttl


def lane_b_slots(cfg) -> tuple[str, ...]:
    """The lane-B emission slots (DEC-CA-0059): ``cfg.targets`` (domains +
    horizon targets), or ``cfg.domains`` for a config without targets."""
    return tuple(getattr(cfg, "targets", None) or cfg.domains)


def lane_b_payouts(state: LaneBState, *, block: int, era_blocks: int, cfg, n_uids: int,
                   uid_of: dict[str, int] | None = None) -> list[dict]:
    """The PAID live wins at ``block`` — what :func:`combined_weights` pays and
    the receipt's ``lane_b_pay`` records: ``{domain, hotkey, uid,
    merge_block, age, multiplier, gain_lcb, tier, full_eras, below_floor}``
    per configured domain (``age`` in eras; the full-pay fields as RECORDED
    on the win),
    deregistered or out-of-range winners and EXPIRED wins (``age >=
    win_ttl_eras``) dropped — an unpaid slot burns."""
    out = []
    for d in lane_b_slots(cfg):
        w = state.wins.get(d)
        if w is None:
            continue
        if int(block) < int(w.merge_block):
            continue
        uid = w.uid if uid_of is None else uid_of.get(w.hotkey)
        if uid is None or not 0 <= int(uid) < n_uids:
            continue
        age = state.eras_since(w, block, era_blocks)
        if win_expired(age, cfg):
            continue
        m = reward_multiplier(age, full_eras=win_full_eras(w),
                              half_life_rounds=cfg.half_life_rounds, floor=cfg.floor)
        out.append({"domain": d, "hotkey": w.hotkey, "uid": int(uid),
                    "merge_block": int(w.merge_block), "age": int(age), "multiplier": m,
                    **w.full_pay_record()})
    return out


def combined_weights(lane_a_uids: list[int], n_uids: int, *, king_decay: float, burn_uid: int,
                     state: LaneBState, block: int, era_blocks: int, cfg,
                     uid_of: dict[str, int] | None = None) -> list[float]:
    """The weight vector a validator sets once lane B is active: lane A's
    court vector (unchanged rule) scaled to ``1 − share``, each live target
    win paid ``share / n_targets × multiplier``, the unpaid lane-B remainder
    burned (or returned to lane A). Before ``cfg.from_block`` this is exactly
    ``decayed_share_vector`` — callers may rely on that."""
    lane_a_vec = decayed_share_vector(lane_a_uids, n_uids, decay=king_decay, burn_uid=burn_uid)
    if not cfg.active(block):
        return lane_a_vec
    lane_a = {u: w for u, w in enumerate(lane_a_vec) if w > 0}
    wins = [LaneBWin(uid=p["uid"], domain=p["domain"], multiplier=p["multiplier"])
            for p in lane_b_payouts(state, block=block, era_blocks=era_blocks, cfg=cfg,
                                    n_uids=n_uids, uid_of=uid_of)]
    shares = split_emissions(lane_a, wins, share_b=cfg.share,
                             n_domain_slots=max(1, len(lane_b_slots(cfg))),
                             burn_uid=burn_uid, remainder=cfg.remainder)
    vec = [0.0] * n_uids
    for u, w in shares.items():
        vec[u] += w
    return vec
