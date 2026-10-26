"""Lane C (DEC-CA-0056) consensus state: the per-target fine-tune slot
holders, their payouts and the recipe rule. MarketCast-only lane C (owner
2026-10-20): every target is a forward-only competition's (today
``mkt:<suite>``, DEC-CA-0064); the generic DEC-CA-0056 window selection over
the public-feed ``pool_c`` is retired.

Pure and deterministic — every validator derives the same state from the same
settlements, and ``cascade-audit`` replays the payouts. Mirrors
:mod:`cascade.shared.lane_b`: payout age is counted in ERAS (``era_blocks``
= the grid in force × ``era_settlements``) from the settlement that judged the
win (age 0 = the first era-length after it); a win pays in full for its
RECORDED ``full_eras`` eras, then decays to ``floor``, and stops paying at
``win_ttl_eras`` (its slot burns). Gain-tiered full-pay period (owner
2026-10-20): ``full_eras`` is chosen at the confirmed win from its gain LCB by
``[lane_c] full_pay_tiers`` — ``[marketcast] full_pay_tiers`` for an ``mkt:``
target (:func:`target_full_pay`) — and recorded on the :class:`TargetWin`
(``tier`` / ``full_eras`` / ``below_floor``), never re-evaluated. A confirmed win takes a target's slot
per :func:`win_beats` (a strictly higher gain LCB on the same king, or any
win on a newer king). A winning fine-tune is published and paid but NEVER becomes the
pretraining init — lane A's lineage is untouched.

A win is CONFIRMED, never immediate: a target's best passer at one
settlement (at ``alpha / k`` over that target's ``k`` judged entries) is
held in ``pending`` and wins only if it passes again, at ``alpha`` against
the same king, on the NEXT lane-C settlement's fresh windows (the win block
is that settlement's). An entry inconclusive for lack of distinct clusters
is ``deferred`` once to the next settlement with the same king. Pending
entries ride the state (snapshot / rollback / catch-up) exactly like slots.

MarketCast (DEC-CA-0064) ``mkt:<suite>`` targets use the same slots and
pending set with one difference: their windows are forward-only market
windows that close over time, so an entry inconclusive for lack of windows
or clusters WAITS (re-deferred, or a confirm kept pending) while the
settlement is within ``[marketcast] verdict_wait_blocks`` of the block its
wait started; their base is the pinned Timestral checkpoint, not the king,
so a king change never makes them stale.

MarketCast review fixes (2026-10-19): an ``mkt:`` win's ``base_pointer`` is
the pinned base's identity (``[marketcast] base_pointer``), so
:func:`win_beats` always compares LCBs for it; MarketCast entries in flight
live in ``mkt_backlog`` (one item per MarketCast round, applied strictly in
round order once a trainer-signed lane-C manifest records the round's report
— see :mod:`cascade.validator.lane_c`) and ``mkt_seen`` is the exact-identity
registry (checkpoint sha256 → the earliest-committed entry). Both are
omitted from the persisted state when empty.

Emission: lane C's share comes from ``takes_from`` (lane A on mainnet,
lane B under the legacy/testnet split). Mainnet stages: A 100, then A 75 /
B 25, then A 50 / B 25 / C 25. Lane C is cut into per-target slots (unpaid
burns). Before ``from_block`` nothing changes.

Concurrent competitions (owner 2026-10-20, :func:`lane_c_split`): lane C's
share is split by competition WEIGHT (normalised over every CONFIGURED
competition, so one starting or ending never changes another's pay), then
equally among that competition's targets: a target's slot is ``share ×
weight / Σweight / n_targets(competition)`` × its multiplier. A competition
is ``pending`` before its start (its part burns), ``active`` while judged,
``winding_down`` after its ``until_block`` while ``wind_down =
"finish_full_pay"`` and within ``wind_down_max_eras`` eras of the end (a held
win is paid in full while inside its own full-pay period, else its slot
burns — no decay tail), then ``ended`` (its part burns). With the shipped
config (MarketCast only, weight 1) every slot is ``share / n_targets`` —
exactly the single-competition rule.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..eval.expert_lane import FullPay, full_pay, reward_multiplier
from .config.lane_c import (
    COMPETITION_KINDS,
    COMPETITION_STATES,
    RECIPE_SCHEDULES,
    check_target,
    is_mkt_target,
)

# ── recipe (miner side; validated by the trainer, one rule for both) ────────

RECIPE_FILE = "recipe.json"
RECIPE_KEYS = ("target", "lr_scale", "steps_fraction", "warmup_fraction", "schedule")
MAX_WARMUP_FRACTION = 0.2


@dataclass(frozen=True)
class Recipe:
    """A miner's fine-tune recipe (``recipe.json`` at the root of the
    committed generator tree). ``lr_scale`` multiplies the leg's base LR;
    ``steps_fraction`` is the fraction of the leg's step budget actually run;
    ``warmup_fraction`` of those steps warm up linearly; ``schedule`` shapes
    the LR after warmup."""
    target: str
    lr_scale: float
    steps_fraction: float
    warmup_fraction: float
    schedule: str


def _f(obj: dict, k: str) -> float:
    v = obj.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"recipe {k} must be a finite number; got {v!r}")
    return float(v)


def parse_recipe(obj: Any, cfg: Any, *, target: str | None = None) -> Recipe:
    """Validate a ``recipe.json`` payload (a dict, or its JSON text) against
    ``[lane_c]`` (``cfg``: a :class:`~cascade.shared.config.lane_c.LaneCConfig`).
    ``target`` (the committed target) must equal the recipe's when given.
    Raises ``ValueError`` on any violation; unknown keys are refused."""
    if isinstance(obj, (str, bytes, bytearray)):
        try:
            obj = json.loads(obj)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"recipe is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("recipe must be a JSON object")
    extra = sorted(set(obj) - set(RECIPE_KEYS))
    if extra:
        raise ValueError(f"recipe has unknown keys {extra}; allowed {list(RECIPE_KEYS)}")
    missing = [k for k in RECIPE_KEYS if k not in obj]
    if missing:
        raise ValueError(f"recipe is missing {missing}")
    spec = check_target(str(obj["target"]))
    if spec not in tuple(getattr(cfg, "targets", ()) or ()):
        raise ValueError(f"recipe target {spec!r} is not a [lane_c] target")
    if target is not None and check_target(str(target)) != spec:
        raise ValueError(f"recipe target {spec!r} != committed target {target!r}")
    lr = _f(obj, "lr_scale")
    if not 0.0 < lr <= float(cfg.max_lr_scale):
        raise ValueError(f"recipe lr_scale {lr} must be in (0, {cfg.max_lr_scale}]")
    sf = _f(obj, "steps_fraction")
    if not 0.0 < sf <= float(cfg.max_steps_fraction):
        raise ValueError(f"recipe steps_fraction {sf} must be in (0, {cfg.max_steps_fraction}]")
    wf = _f(obj, "warmup_fraction")
    if not 0.0 <= wf <= MAX_WARMUP_FRACTION:
        raise ValueError(f"recipe warmup_fraction {wf} must be in [0, {MAX_WARMUP_FRACTION}]")
    sched = obj.get("schedule")
    if sched not in RECIPE_SCHEDULES:
        raise ValueError(f"recipe schedule {sched!r} must be one of {RECIPE_SCHEDULES}")
    return Recipe(target=spec, lr_scale=lr, steps_fraction=sf, warmup_fraction=wf,
                  schedule=str(sched))


# ── consensus state ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TargetWin:
    target: str
    hotkey: str
    uid: int
    round_id: str
    win_block: int            # the settlement boundary block that judged it
    win_era: int = 0
    trained_pointer: str = ""
    recipe_ref: str = ""
    base_pointer: str = ""
    gain_lcb: float = 0.0
    # gain-tiered full-pay period, fixed at the confirmed win from
    # ``gain_lcb`` (:func:`target_full_pay`); None = untiered (refused by the
    # payout — every recorded win carries its tier)
    tier: float | None = None
    full_eras: int | None = None
    below_floor: bool = False

    def full_pay_record(self) -> dict:
        """``{gain_lcb, tier, full_eras, below_floor}`` — what the payout
        record and the settlement state stamp."""
        return {"gain_lcb": float(self.gain_lcb), "tier": self.tier,
                "full_eras": self.full_eras, "below_floor": bool(self.below_floor)}


def full_pay_tiers_for(target: str, cfg, mc=None) -> tuple:
    """The full-pay tiers a win on ``target`` is tiered by: ``[marketcast]
    full_pay_tiers`` (``mc``) for an ``mkt:`` target, else ``[lane_c]``'s
    (``cfg``)."""
    if is_mkt_target(target):
        if mc is None:
            raise ValueError(f"an {target!r} win needs [marketcast] for its full-pay tiers")
        return tuple(mc.full_pay_tiers)
    return tuple(cfg.full_pay_tiers)


def target_full_pay(target: str, gain_lcb: float | None, cfg, mc=None) -> FullPay:
    """CONSENSUS: the full-pay period of a confirmed win on ``target`` with
    gain LCB ``gain_lcb`` (:func:`cascade.eval.expert_lane.full_pay` over
    :func:`full_pay_tiers_for`)."""
    return full_pay(gain_lcb, full_pay_tiers_for(target, cfg, mc))


PENDING_KINDS = ("confirm", "deferred")


@dataclass(frozen=True)
class PendingEntry:
    """A lane-C entry carried to the next settlement with the same king:
    ``"confirm"`` (passed as its target's best; re-scored at ``alpha`` on the
    next settlement's windows — pass ⇒ the win) or ``"deferred"`` (too few
    distinct clusters; re-judged once as a candidate of the next
    settlement). ``entry`` is the signed :class:`~cascade.shared.
    lane_c_manifest.LaneCEntry` as a dict; ``round_id`` / ``block`` name the
    settlement that first judged it."""
    target: str
    kind: str
    entry: dict
    base_pointer: str
    round_id: str
    block: int
    gain: float | None = None
    gain_lcb: float | None = None
    # DEC-CA-0064 MarketCast entries only: {"origin_block", "fresh_after"} —
    # the block its wait started and (a confirm) the instant its fresh
    # windows must start after. Omitted from the persisted state when empty.
    mkt: dict = field(default_factory=dict)

    @property
    def hotkey(self) -> str:
        return str(self.entry.get("miner_hotkey", ""))

    def sort_key(self) -> tuple:
        return (self.target, self.kind, int(self.block), self.round_id, self.hotkey,
                str(self.entry.get("trained_pointer", "")))


HISTORY_KEEP = 64
APPLIED_KEEP = 4096


def win_beats(new: TargetWin, cur: TargetWin) -> bool:
    """The lane-C slot rule (DEC-CA-0056): does ``new`` take ``cur``'s slot?

    * different ``base_pointer`` (fine-tunes of different kings): the LATER
      win — by ``(win_block, round_id)`` — holds; a holder trained on an older
      king no longer defends (kings only move forward, so the later win is the
      one on the current king);
    * same ``base_pointer``: only a STRICTLY higher recorded ``gain_lcb`` vs
      the king takes the slot (and so restarts the payout clock); on an exact
      tie the EARLIER win holds. No special case for the holder's own hotkey.

    The LCBs compared come from different settlements' windows: a
    conservative point comparison of recorded values, not a paired test.
    Pure in the four fields, hence order-independent."""
    newer = (int(new.win_block), str(new.round_id)) > (int(cur.win_block), str(cur.round_id))
    if new.base_pointer != cur.base_pointer:
        return newer
    if float(new.gain_lcb) != float(cur.gain_lcb):
        return float(new.gain_lcb) > float(cur.gain_lcb)
    return not newer and (int(new.win_block), str(new.round_id)) != (int(cur.win_block),
                                                                     str(cur.round_id))


@dataclass
class LaneCState:
    slots: dict[str, TargetWin] = field(default_factory=dict)
    history: list[TargetWin] = field(default_factory=list)
    # settlements whose lane-C manifest has been APPLIED (round id -> block):
    # idempotent per round id (a crash between this state and the lane-A
    # marker re-processes the round without double-recording)
    applied: dict[str, int] = field(default_factory=dict)
    # the last settled lane-C boundary — the payout clock a restart resumes
    settled_block: int | None = None
    # the last settlement folded IN CHAIN ORDER (direct or catch-up)
    last_round: str | None = None
    # entries awaiting confirmation / deferred (resolved at the next lane-C
    # settlement that has a king)
    pending: list[PendingEntry] = field(default_factory=list)
    # the receipt rows of ``last_round`` as folded: a crash-replay of that
    # round returns them verbatim (re-judging against the post-fold state
    # would see a different pending set)
    last_rows: list | None = None
    # DEC-CA-0064: MarketCast rounds awaiting their report, oldest first:
    # {"round_id", "block", "created_block", "candidates": [...], "arbiter":
    # "" | sha256 | "absent", "age"} (cascade.validator.lane_c)
    mkt_backlog: list = field(default_factory=list)
    # DEC-CA-0064 exact-identity dedup: checkpoint sha256 -> {"hotkey",
    # "pointer", "target", "commit_block", "train_block"} of the earliest commit
    mkt_seen: dict = field(default_factory=dict)

    def set_pending(self, pending: list[PendingEntry]) -> None:
        for p in pending:
            if p.kind not in PENDING_KINDS:
                raise ValueError(f"unknown pending kind {p.kind!r}")
        self.pending = sorted(pending, key=PendingEntry.sort_key)

    def record(self, win: TargetWin) -> bool:
        """Offer a confirmed win to its target's slot (:func:`win_beats`);
        True when it takes the slot. Every offered win enters the history.
        The rule is a pure pairwise preference on ``(base_pointer, gain_lcb,
        win_block, round_id)``, so the fold is order-independent (an
        out-of-order catch-up lands on the same holder)."""
        cur = self.slots.get(win.target)
        took = cur is None or win_beats(win, cur)
        if took:
            self.slots[win.target] = win
        if win not in self.history:
            self.history.append(win)
            self.history.sort(key=lambda w: (w.win_block, w.round_id, w.target))
            del self.history[:-HISTORY_KEEP]
        return took

    def mark_applied(self, round_id: str, block: int, *, in_chain: bool = True) -> None:
        self.applied[str(round_id)] = int(block)
        if len(self.applied) > APPLIED_KEEP:
            for rid, _ in sorted(self.applied.items(), key=lambda kv: kv[1])[
                    : len(self.applied) - APPLIED_KEEP]:
                del self.applied[rid]
        if self.settled_block is None or int(block) > self.settled_block:
            self.settled_block = int(block)
        if in_chain:
            self.last_round = str(round_id)

    @staticmethod
    def eras_since(win: TargetWin, block: int, era_blocks: int) -> int:
        if era_blocks <= 0:
            raise ValueError("era_blocks must be > 0")
        return max(0, (int(block) - int(win.win_block)) // int(era_blocks))

    def to_json(self) -> str:
        d = {"slots": {t: asdict(w) for t, w in sorted(self.slots.items())},
             "history": [asdict(w) for w in self.history],
             "applied": dict(sorted(self.applied.items())),
             "settled_block": self.settled_block,
             "last_round": self.last_round,
             "pending": [_pending_dict(p) for p in self.pending],
             "last_rows": self.last_rows}
        if self.mkt_backlog:
            d["mkt_backlog"] = self.mkt_backlog
        if self.mkt_seen:
            d["mkt_seen"] = self.mkt_seen
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> LaneCState:
        """Parse a persisted state; an empty / torn file RAISES (the caller
        rebuilds by catch-up) — never a silent empty state."""
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("lane-C state is not a JSON object")
        sb = obj.get("settled_block")
        return cls(slots={t: TargetWin(**w) for t, w in (obj.get("slots") or {}).items()},
                   history=[TargetWin(**w) for w in obj.get("history", [])],
                   applied={str(k): int(v) for k, v in (obj.get("applied") or {}).items()},
                   settled_block=None if sb is None else int(sb),
                   last_round=obj.get("last_round"),
                   pending=[PendingEntry(**p) for p in obj.get("pending") or ()],
                   last_rows=obj.get("last_rows"),
                   mkt_backlog=list(obj.get("mkt_backlog") or ()),
                   mkt_seen=dict(obj.get("mkt_seen") or {}))


def _pending_dict(p: PendingEntry) -> dict:
    d = asdict(p)
    if not d.get("mkt"):
        d.pop("mkt", None)
    return d


def save_state_atomic(state: LaneCState, path) -> None:
    from .atomic import write_text_atomic

    write_text_atomic(path, state.to_json())


def load_state(path) -> LaneCState | None:
    """The persisted state, a fresh one when absent, or None when the file is
    torn / unreadable (the caller discards it loudly and rebuilds)."""
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return LaneCState()
    try:
        return LaneCState.from_json(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


# ── payouts and emission ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class LaneCWin:
    """One PAID slot of a weight vector (``share / n_targets × multiplier``)."""
    uid: int
    target: str
    multiplier: float


def win_expired(age_eras: int, cfg) -> bool:
    ttl = int(getattr(cfg, "win_ttl_eras", 0) or 0)
    return ttl > 0 and int(age_eras) >= ttl


def target_multiplier(age_eras: int, cfg, full_eras: int | None) -> float:
    """A lane-C win's multiplier at ``age_eras`` from its RECORDED
    ``full_eras`` (None = an untiered win: refused) on ``[lane_c]``'s decay."""
    if full_eras is None:
        raise ValueError("lane-C win carries no full-pay tier (full_eras)")
    return reward_multiplier(int(age_eras), full_eras=int(full_eras),
                             half_life_rounds=cfg.half_life_rounds, floor=cfg.floor)


# ── concurrent competitions (owner 2026-10-20) ───────────────────────────────


def competition_from_block(comp, mc=None) -> int:
    """The start block of ``comp``: ``[marketcast] from_block`` (``mc``) for a
    kind whose start lives there (``mkt``), else its own ``from_block``."""
    ck = COMPETITION_KINDS.get(comp.kind)
    if ck is not None and ck.from_block_table == "marketcast":
        return int(getattr(mc, "from_block", 0) or 0) if mc is not None else 0
    return int(comp.from_block)


def competition_started(comp, block: int | None, mc=None) -> bool:
    """Whether ``comp`` has started at settlement ``block`` (``mkt``:
    :meth:`MarketCastConfig.active`, exactly the pre-competitions rule)."""
    ck = COMPETITION_KINDS.get(comp.kind)
    if ck is not None and ck.from_block_table == "marketcast":
        return mc is not None and bool(mc.active(block))
    fb = int(comp.from_block)
    return fb > 0 and block is not None and int(block) >= fb


def competition_past_end(comp, block: int | None) -> bool:
    """Whether settlement ``block`` lies after ``comp``'s ``until_block`` (the
    settlement AT it is the last one judged)."""
    u = int(comp.until_block)
    return u > 0 and block is not None and int(block) > u


def _in_wind_down(comp, block: int | None, era_blocks: int) -> bool:
    """``block`` lies after ``comp``'s end and inside its wind-down window
    (``finish_full_pay``, fewer than ``wind_down_max_eras`` eras since it)."""
    if not competition_past_end(comp, block) or comp.wind_down != "finish_full_pay":
        return False
    eb = max(1, int(era_blocks))
    return (int(block) - int(comp.until_block)) // eb < int(comp.wind_down_max_eras)


def competition_state(comp, block: int | None, era_blocks: int, mc=None) -> str:
    """CONSENSUS: ``comp``'s state at settlement ``block`` —
    :data:`~cascade.shared.config.lane_c.COMPETITION_STATES`. Pure in the
    config: ``winding_down`` is the window after the end (``finish_full_pay``,
    fewer than ``wind_down_max_eras`` eras since ``until_block``) whether or
    not a held win still pays in it."""
    if not competition_started(comp, block, mc):
        return "pending"
    if not competition_past_end(comp, block):
        return "active"
    return "winding_down" if _in_wind_down(comp, block, era_blocks) else "ended"


def wind_down_pays(comp, win, block: int, era_blocks: int) -> bool:
    """Whether a held win of a competition past its end is paid at ``block``:
    inside the wind-down window AND still inside its own recorded full-pay
    period (``age < full_eras``). No decay tail: outside either, nothing."""
    if comp is None or win is None or getattr(win, "full_eras", None) is None:
        return False
    age = max(0, (int(block) - int(win.win_block)) // max(1, int(era_blocks)))
    return _in_wind_down(comp, block, era_blocks) and age < int(win.full_eras)


def competition_closed(cfg, target: str, block: int | None) -> str | None:
    """The ADMISSION refusal for a new ``target`` entry at chain ``block``
    (intake / commit; None = open): its competition's ``until_block`` has
    been reached."""
    comp = cfg.competition_of(target)
    if comp is None or not int(comp.until_block) or block is None:
        return None
    if int(block) >= int(comp.until_block):
        return (f"lane-C competition {comp.kind!r} ended at block {comp.until_block} — "
                f"no new {target!r} entries are admitted")
    return None


def active_targets(cfg, block: int | None, mc=None) -> tuple[str, ...]:
    """The lane-C targets JUDGED at a settlement ``block`` (review L9;
    concurrent competitions): those whose competition has started and not
    passed its ``until_block`` — an ``mkt:<suite>`` one only while
    ``[marketcast]`` (``mc``) is active there."""
    out = []
    for t in cfg.targets:
        comp = cfg.competition_of(t)
        if (comp is not None and competition_started(comp, block, mc)
                and not competition_past_end(comp, block)):
            out.append(t)
    return tuple(out)


def lane_c_split(cfg, block: int, era_blocks: int, mc=None) -> tuple[list[dict], dict]:
    """CONSENSUS: ``(competitions, slot_frac)`` at settlement ``block``.

    ``competitions``: per configured competition (kind order) its config
    (:meth:`LaneCCompetition.record`, ``from_block`` resolved), ``frac`` =
    ``weight / Σweight`` over EVERY configured competition, ``state``
    (:func:`competition_state`), ``targets`` and ``slot_frac`` (= ``frac /
    n_targets`` while ``active`` / ``winding_down``, else 0 — its part
    burns). ``slot_frac``: ``{target: fraction of lane C's share}`` for every
    slot-bearing target. Its values sum to <= 1; the rest of the share burns."""
    comps = cfg.competition_list
    total = sum(float(c.weight) for c in comps)
    recs: list[dict] = []
    slot: dict[str, float] = {}
    for c in comps:
        frac = float(c.weight) / total if total > 0 else 0.0
        state = competition_state(c, block, era_blocks, mc)
        tg = cfg.competition_targets(c.kind)
        paying = tg if state in ("active", "winding_down") else ()
        per = frac / len(paying) if paying else 0.0
        for t in paying:
            slot[t] = per
        recs.append({**c.record(), "from_block": competition_from_block(c, mc), "frac": frac,
                     "state": state, "targets": list(tg), "slot_frac": per})
    assert all(r["state"] in COMPETITION_STATES for r in recs)
    return recs, slot


def lane_c_payouts(state: LaneCState, *, block: int, era_blocks: int, cfg, n_uids: int,
                   uid_of: dict[str, int] | None = None,
                   targets: tuple[str, ...] | None = None,
                   winding: frozenset[str] | set[str] = frozenset()) -> list[dict]:
    """The PAID live slots at ``block`` — what the vector pays and the
    receipt's ``lane_c_pay`` records: ``{target, hotkey, uid, win_block, age,
    multiplier, gain_lcb, tier, full_eras, below_floor}`` per paying target
    (``targets``, default every configured one; in that order);
    deregistered / out-of-range holders and EXPIRED wins are dropped (their
    slot burns). A target in ``winding`` (its competition is winding down)
    pays multiplier 1.0 while :func:`wind_down_pays`, else nothing, and its
    row carries ``"wind_down": True``."""
    out = []
    for t in (cfg.targets if targets is None else targets):
        w = state.slots.get(t)
        if w is None:
            continue
        if int(block) < int(w.win_block):
            continue
        uid = w.uid if uid_of is None else uid_of.get(w.hotkey)
        if uid is None or not 0 <= int(uid) < n_uids:
            continue
        age = state.eras_since(w, block, era_blocks)
        if win_expired(age, cfg):
            continue
        row = {"target": t, "hotkey": w.hotkey, "uid": int(uid),
               "win_block": int(w.win_block), "age": int(age)}
        if t in winding:
            if not wind_down_pays(cfg.competition_of(t), w, block, era_blocks):
                continue
            row["multiplier"] = 1.0
            row["wind_down"] = True
        else:
            row["multiplier"] = target_multiplier(age, cfg, w.full_eras)
        out.append({**row, **w.full_pay_record()})
    return out


def pay_wins(pay: dict | None) -> list[LaneCWin]:
    """The :class:`LaneCWin` list of a ``lane_c_pay`` record."""
    return [LaneCWin(uid=int(w["uid"]), target=str(w["target"]),
                     multiplier=float(w["multiplier"])) for w in (pay or {}).get("wins", ())]


def lane_pool_shares(lane_b_cfg, lane_c_cfg=None) -> tuple[float, float]:
    """``(B, C)`` allocations; pass C only when it is active. A gets the rest.

    Mainnet takes C from A, preserving B's allocation. ``takes_from=lane_b``
    retains the original split for existing configurations and receipts.
    """
    share_b = float(getattr(lane_b_cfg, "share", 0.0) or 0.0) if lane_b_cfg is not None else 0.0
    if lane_c_cfg is None:
        return share_b, 0.0
    if lane_c_cfg.takes_from == "lane_a":
        return share_b, min(float(lane_c_cfg.share), max(0.0, 1.0 - share_b))
    if lane_b_cfg is None:
        raise ValueError("lane C taken from lane B cannot run without lane B")
    share_c = min(float(lane_c_cfg.share), share_b)
    return share_b - share_c, share_c


def lane_c_shares(lane_b_cfg, lane_c_cfg, block: int | None) -> tuple[float, float]:
    """``(B, C)`` allocations at settlement ``block``; A receives the rest."""
    active_c = lane_c_cfg if lane_c_cfg is not None and lane_c_cfg.active(block) else None
    active_b = lane_b_cfg if lane_b_cfg is not None and lane_b_cfg.active(block) else None
    return lane_pool_shares(active_b, active_c)


def lane_c_vector(n_uids: int, *, cfg, wins: list[LaneCWin], burn_uid: int,
                  share: float | None = None,
                  slot_frac: dict[str, float] | None = None) -> list[float]:
    """Lane C's part of the vector: each paid slot ``share × slot_frac[target]
    × multiplier`` on its holder, the unpaid remainder on ``burn_uid``. Sums
    to ``share`` (default ``cfg.share``). ``slot_frac`` = the settlement's
    per-target fractions (:func:`lane_c_split`, the ``lane_c_pay`` record's);
    default an equal split over every configured target."""
    s = float(cfg.share if share is None else share)
    if slot_frac is None:
        n = max(1, len(cfg.targets))
        slot_frac = {t: 1.0 / n for t in cfg.targets}
    out = [0.0] * n_uids
    seen: set[str] = set()
    paid = 0.0
    for w in wins:
        if w.target in seen:
            raise ValueError(f"two paid lane-C wins for target {w.target!r}")
        if w.target not in slot_frac:
            raise ValueError(f"lane-C win for {w.target!r}, which holds no slot here")
        seen.add(w.target)
        amt = s * float(slot_frac[w.target]) * min(1.0, max(0.0, float(w.multiplier)))
        out[int(w.uid)] += amt
        paid += amt
    left = s - paid
    if left > 0:
        out[burn_uid] += left
    return out


def lane_c_pay_record(state: LaneCState, *, block: int, era_blocks: int, cfg, n_uids: int,
                      uid_of: dict[str, int] | None = None, mc=None) -> dict:
    """The receipt's ``lane_c_pay`` record: ``{"block", "era_blocks",
    "share", "slots", "slot_frac", "competitions", "wins": [...]}`` —
    ``competitions`` / ``slot_frac`` are :func:`lane_c_split`'s, ``slots``
    counts the slot-bearing targets and the paid ones are among them."""
    comps, slot = lane_c_split(cfg, int(block), int(era_blocks), mc)
    paying = tuple(t for t in cfg.targets if t in slot)
    winding = frozenset(t for c in comps if c["state"] == "winding_down" for t in c["targets"])
    return {"block": int(block), "era_blocks": int(era_blocks),
            "share": float(cfg.share), "slots": len(paying),
            "slot_frac": {t: slot[t] for t in paying}, "competitions": comps,
            "wins": lane_c_payouts(state, block=block, era_blocks=era_blocks, cfg=cfg,
                                   n_uids=n_uids, uid_of=uid_of, targets=paying,
                                   winding=winding)}


__all__ = [
    "PENDING_KINDS", "LaneCState", "LaneCWin", "MAX_WARMUP_FRACTION", "PendingEntry",
    "RECIPE_FILE", "RECIPE_KEYS", "Recipe",
    "TargetWin", "active_targets", "competition_closed", "competition_from_block",
    "competition_past_end", "competition_started", "competition_state", "full_pay_tiers_for",
    "lane_c_pay_record", "lane_c_payouts", "lane_c_shares", "lane_c_split", "lane_c_vector",
    "lane_pool_shares",
    "load_state", "parse_recipe", "pay_wins", "save_state_atomic", "target_full_pay",
    "target_multiplier", "wind_down_pays", "win_beats", "win_expired",
]
