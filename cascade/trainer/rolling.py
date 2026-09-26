"""Rolling intake + era king (DEC-CA-0043) — the trainer side.

From ``[round] rolling_from_block`` the trainer stops running boundary-
synchronous rounds. Instead:

* **Challengers train the moment they are funded.** Every tick the funded
  queue is drained in seniority order into legs that start NOW (up to
  ``funded_field_cap`` in flight), each targeting the earliest settlement
  boundary its wall + publish margin can clear. A leg that would land in the
  NEXT era waits for that era's pre-train window (wall + margin before the
  boundary) so it trains under the right seeds and init — a miner never pays
  for a leg that lands on the wrong init.
* **Verdicts stay batched.** Every epoch boundary is a *settlement*: ONE
  manifest carrying the era king's entry plus every challenger harvested,
  verified and benched since the last settlement (same era). Nothing
  finished ⇒ no manifest. Manifests are hash-chained (``prev_round_id``).
* **The king's leg is trained once per era and cached.** An era is
  ``era_settlements`` boundaries sharing one set of seeds (the previous era's
  start block hash), one init (``members_gen(g)[era % k]``) and one king
  checkpoint. The next era's king pre-trains during the last wall + margin
  of the current era. A dethrone does not end the era: the winner's
  checkpoint — trained under the era's seeds and init with the full budget —
  becomes the era king at zero cost.
* **The king is derived from the validators' receipts**, never from the
  metagraph (incentive lags a dethrone longer than a settlement); the
  metagraph is only a lagging cross-check that logs on disagreement.
* **Bench at completion.** A payer pod benches its checkpoint the moment the
  leg finishes (harvest → ingest-verify → bench → tear down); the top-N
  operator re-bench runs on the era king's pod. A payer pod is never held to
  a settlement boundary.

The policy lives in :class:`RollingScheduler`; everything that touches a pod
or a bucket goes through :class:`LegOps` so the policy is unit-testable with
a fake. State (``era_state.json`` under the work root) survives restarts:
era index, cached king pointer + bench, the pre-trained next-era king, the
generation ledger, finished-but-unsettled legs; in-flight legs live in the
funded queue (``in_flight`` status with their target boundary, era and
start block).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..shared.era import (
    EraSpec,
    era_for_block,
    era_length_blocks,
    member_index_for_era,
    min_effective_era,
    next_era_start,
    settlement_era,
)
from ..shared.manifest import TrainedEntry
from .remote import error_tail

log = logging.getLogger(__name__)

BLOCK_SECONDS = 12.0
# A booted king pod gets ONE retry after a non-transport failure (a Hippius
# read stall is not necessarily the pod's fault); the next failure — or any
# transport failure at once — rotates it: host quarantined, pod torn down,
# the next tick re-rents elsewhere. Without this the leg looped into the
# same bad network path every tick for hours (2026-09-21 review).
KING_SAME_POD_RETRIES = 1
STATE_FILE = "era_state.json"


# ── persisted state ──────────────────────────────────────────────────────────


@dataclass
class EraState:
    """One era as the trainer tracks it."""

    index: int
    start_block: int
    seed_block: int
    base_seed: int                    # block_seed(seed_block): the era's training seeds
    generation: int = 0
    member_index: int = 0
    warm_start_ckpt: str = ""
    warm_start_size: str = ""
    king_hotkey: str = ""
    king_uid: int = -1
    king_ref: str = ""
    king_entry: dict | None = None    # TrainedEntry (asdict), role "king"
    king_init: str = ""               # the init king_entry trained from (must == warm_start_ckpt)
    king_bench: dict | None = None    # BenchScores (asdict)
    king_bench_published: bool = False
    king_leg_failed: str = ""         # last king-leg failure (retry pending)
    king_leg_failures: int = 0        # consecutive failures on the CURRENT king pod

    def spec(self) -> EraSpec:
        return EraSpec(index=self.index, start_block=self.start_block,
                       seed_block=self.seed_block, generation=self.generation,
                       member_index=self.member_index)

    def king(self) -> TrainedEntry | None:
        return _entry_from_json(self.king_entry) if self.king_entry else None


@dataclass
class FinishedLeg:
    """A challenger leg that returned, verified, benched (or not), waiting
    for its settlement."""

    hotkey: str
    uid: int
    ref: str
    era_index: int
    started_block: int
    reveal_block: int
    entry: dict                        # TrainedEntry (asdict)
    bench: dict | None = None          # operator-verified BenchScores, else None
    label: str = ""


@dataclass
class RollingState:
    current: EraState | None = None
    next: EraState | None = None
    # generation ledger: gen -> {"members": [[ckpt, size], ...], "effective_era": n}
    generations: dict[str, dict] = field(default_factory=dict)
    finished: list[FinishedLeg] = field(default_factory=list)
    published: list[FinishedLeg] = field(default_factory=list)   # this era's settled legs
    last_published_round_id: str = ""
    last_settled_boundary: int = 0
    rents: list[dict] = field(default_factory=list)              # roster: seniority at rent


def _entry_from_json(obj: dict) -> TrainedEntry:
    from ..shared.manifest import _bench_from_json

    return TrainedEntry(
        miner_hotkey=str(obj["miner_hotkey"]), miner_uid=int(obj["miner_uid"]),
        role=str(obj["role"]), gen_ref=str(obj["gen_ref"]),
        trained_pointer=str(obj["trained_pointer"]),
        corpus_digest=str(obj["corpus_digest"]), train_block=int(obj["train_block"]),
        gpu_name=str(obj.get("gpu_name", "") or ""), size=str(obj.get("size", "") or ""),
        bench_scores=_bench_from_json(obj.get("bench_scores")),
        duel_rank=int(obj.get("duel_rank", 0) or 0),
    )


def _entry_to_json(entry: TrainedEntry) -> dict:
    return dataclasses.asdict(entry)


def load_state(path: Path) -> RollingState:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return RollingState()
    except Exception as e:  # noqa: BLE001 — a torn file must not sink the trainer
        log.error("rolling: era state %s unreadable (%s); starting fresh", path, e)
        return RollingState()

    def _era(obj):
        return EraState(**obj) if obj else None

    return RollingState(
        current=_era(raw.get("current")),
        next=_era(raw.get("next")),
        generations={str(k): dict(v) for k, v in (raw.get("generations") or {}).items()},
        finished=[FinishedLeg(**x) for x in raw.get("finished", [])],
        published=[FinishedLeg(**x) for x in raw.get("published", [])],
        last_published_round_id=str(raw.get("last_published_round_id", "") or ""),
        last_settled_boundary=int(raw.get("last_settled_boundary", 0) or 0),
        rents=list(raw.get("rents", [])),
    )


def save_state(path: Path, state: RollingState) -> None:
    body = {
        "current": dataclasses.asdict(state.current) if state.current else None,
        "next": dataclasses.asdict(state.next) if state.next else None,
        "generations": state.generations,
        "finished": [dataclasses.asdict(x) for x in state.finished],
        "published": [dataclasses.asdict(x) for x in state.published],
        "last_published_round_id": state.last_published_round_id,
        "last_settled_boundary": state.last_settled_boundary,
        "rents": state.rents,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


# ── pure policy ──────────────────────────────────────────────────────────────


def king_pod_should_rotate(exc: BaseException, failures: int) -> bool:
    """Rotate the era king's pod after ``failures`` consecutive king-leg
    failures on it: at once when the failure is a transport one (the pod is
    unreachable), else once the same-pod retry budget is spent."""
    from .loop import _transport_failure

    return _transport_failure(exc) or int(failures) > KING_SAME_POD_RETRIES


def wall_of_block(block: int, *, now: float, block_now: int) -> float:
    """Wall-clock estimate for chain height ``block``."""
    return now + (int(block) - int(block_now)) * BLOCK_SECONDS


def target_boundary(round_cfg, *, block_now: int, now: float, wall_seconds: float,
                    margin_seconds: float, start_at: float | None = None) -> int:
    """Earliest epoch boundary ``B`` with ``start + wall + margin < wall(B)``
    for a leg starting at ``start_at`` (default now)."""
    from ..shared.config import effective_epoch_blocks

    start = now if start_at is None else start_at
    b = int(block_now)
    eb = int(effective_epoch_blocks(round_cfg, b))
    boundary = (b // eb + 1) * eb
    while wall_of_block(boundary, now=now, block_now=block_now) <= start + wall_seconds + margin_seconds:
        eb = int(effective_epoch_blocks(round_cfg, boundary))
        boundary += eb
    return boundary


@dataclass(frozen=True)
class Admission:
    """Where a leg admitted now would land."""

    target_boundary: int
    era_index: int
    start_after: float          # wall-clock the leg may start (now, or the pre-train window)
    cross_era: bool


def admit(round_cfg, *, block_now: int, now: float, wall_seconds: float,
          margin_seconds: float, current_era: int) -> Admission:
    """Cross-era admission rule (DEC-CA-0043 Init §5): a leg whose wall +
    margin cannot clear the current era's last settlement is not started
    under this era's seeds; it starts under the next era's, no earlier than
    that era's pre-train window (wall + margin before its first boundary).
    Within-era slips settle at the next boundary."""
    b = target_boundary(round_cfg, block_now=block_now, now=now,
                        wall_seconds=wall_seconds, margin_seconds=margin_seconds)
    # The settlement at boundary B belongs to the era containing B − 1: an
    # era's last settlement is the next era's start block (settlement_era).
    era = settlement_era(round_cfg, b)
    if era.index <= current_era:
        return Admission(target_boundary=b, era_index=era.index, start_after=now,
                         cross_era=False)
    # Lands in a later era: start in the pre-train window of that era.
    era_start = era.start_block
    window_open = wall_of_block(era_start, now=now, block_now=block_now) - wall_seconds - margin_seconds
    start = max(now, window_open)
    b2 = target_boundary(round_cfg, block_now=block_now, now=now, wall_seconds=wall_seconds,
                         margin_seconds=margin_seconds, start_at=start)
    era2 = settlement_era(round_cfg, b2)
    return Admission(target_boundary=b2, era_index=era2.index, start_after=start,
                     cross_era=True)


def king_pretrain_open(round_cfg, *, block_now: int, now: float, wall_seconds: float,
                       margin_seconds: float) -> bool:
    """Whether the next era's king leg should already be training: inside
    the last ``wall + margin`` of the current era."""
    nxt = next_era_start(round_cfg, block_now)
    return now >= wall_of_block(nxt, now=now, block_now=block_now) - wall_seconds - margin_seconds


def era_window(round_cfg, era) -> tuple[int, int]:
    """``[lo, hi)``: the train blocks an entry settled under ``era`` may carry
    — the validator's ``era_entry_out_of_window`` check (from the era's seed
    block to its end). One entry outside it rejects the whole settlement."""
    start = int(era.start_block)
    return int(era.seed_block), start + era_length_blocks(round_cfg, start)


def resolve_generation(generations: dict[str, dict], era_index: int) -> tuple[int, list]:
    """``(generation, members)`` for era ``era_index``: the latest generation
    whose ``effective_era <= era_index``; ``(0, [])`` = random init."""
    best_gen, best_members = 0, []
    for g, rec in generations.items():
        gi = int(g)
        if int(rec.get("effective_era", 0)) <= int(era_index) and gi > best_gen:
            best_gen, best_members = gi, list(rec.get("members") or [])
    return best_gen, best_members


def era_init(generations: dict[str, dict], era_index: int) -> tuple[int, int, str, str]:
    """``(generation, member_index, checkpoint, size)`` for the era."""
    gen, members = resolve_generation(generations, era_index)
    if gen == 0 or not members:
        return 0, 0, "", ""
    idx = member_index_for_era(era_index, len(members))
    ckpt, size = members[idx][0], members[idx][1]
    return gen, idx, str(ckpt), str(size)


# ── pod / bucket operations (swappable) ──────────────────────────────────────


class LegOps:
    """Everything the scheduler needs from the outside world, on the live
    :class:`~cascade.trainer.loop.TrainerRunner`. Tests replace this."""

    def __init__(self, runner) -> None:
        self.r = runner

    # chain / storage
    def block_seed(self, client, block: int) -> int:
        return int(client.block_seed(int(block)))

    def commitments(self, client) -> list:
        return client.poll_commitments(include_history=True)

    def receipt_king(self) -> str | None:
        return self.r._receipt_king()

    def latest_round_id(self) -> str:
        """``round_id`` of the bucket's ``latest.json`` — the chain root the
        first settlement links to (the last legacy round, or our own last
        settlement after a lost state file); "" when unreadable."""
        return self.r._rolling_latest_round_id()

    def promotion_effective_era(self, generation: int) -> int | None:
        """``effective_era`` of the PUBLISHED ``promotions/gen-<n>.json`` (0 =
        a pre-era record, live immediately) — the source validators install
        from, so a lost ledger rebuilds to the same eras. None = unreadable."""
        return self.r._rolling_promotion_effective_era(generation)

    def metagraph_king(self, client) -> str | None:
        try:
            return client.highest_incentive_hotkey()
        except Exception:  # noqa: BLE001 — cross-check only
            return None

    # legs
    def train_challenger(self, gen, era: EraState, block: int, *, end_wall: float) -> TrainedEntry:
        return self.r._rolling_train_challenger(gen, era, block, end_wall=end_wall)

    def train_king(self, gen, era: EraState, block: int, *, end_wall: float) -> TrainedEntry:
        return self.r._rolling_train_king(gen, era, block, end_wall=end_wall)

    def cached_leg(self, era: EraState, role: str, gen) -> TrainedEntry | None:
        contract = self.r.cfg.throne_contracts()[0]
        suffix = "" if role == "king" else f"-u{gen.uid}"
        # The era's init is part of the job: a record from the same seed but
        # another init (the legacy rotation, an edited state) is never reused.
        init = self.r._rolling_warm_start_ref(era, contract) or ""
        return self.r._load_completed_leg(round_id=era.base_seed, contract=contract,
                                          role=role, hotkey=gen.hotkey,
                                          gen_ref=gen.ref, suffix=suffix,
                                          warm_start_ckpt=init)

    def discard_cached_leg(self, era: EraState, role: str, gen) -> None:
        """Drop the persisted record of a leg that must be retrained (an
        entry trained outside the era window would otherwise be reused from
        the record on every re-admission)."""
        contract = self.r.cfg.throne_contracts()[0]
        suffix = "" if role == "king" else f"-u{gen.uid}"
        self.r._discard_completed_leg(round_id=era.base_seed, contract=contract,
                                      role=role, hotkey=gen.hotkey, suffix=suffix)

    def bench_challenger(self, entry: TrainedEntry, king: TrainedEntry | None,
                         era: EraState) -> dict | None:
        return self.r._rolling_bench_challenger(entry, king, era)

    def bench_king(self, entry: TrainedEntry, era: EraState) -> dict | None:
        return self.r._rolling_bench_king(entry, era)

    def leg_failure(self, hotkey: str) -> tuple[str, bool, str, bool]:
        return self.r._funded_leg_failures.get(
            hotkey, ("challenger leg failed before dispatch", False, "infra", True))

    def teardown_kept_pod(self, hotkey: str) -> None:
        self.r._teardown_kept_funded_pod(hotkey)

    def retire_king_pod(self, era: EraState) -> None:
        self.r._rolling_retire_king_pod(era)

    def rotate_king_pod(self, era: EraState, reason: str) -> None:
        self.r._rolling_rotate_king_pod(era, reason)

    # publication
    def publish_manifest(self, manifest) -> None:
        self.r.publish(manifest)

    def publish_bench(self, round_id: str, created_block: int, entries: list) -> object | None:
        return self.r._rolling_publish_bench(round_id, created_block, entries)

    def publish_roster(self, round_id: str, roster: dict) -> None:
        self.r._rolling_publish_roster(round_id, roster)

    def promotion_boundary(self, king_hotkey: str, epoch_start: int, round_id: str,
                           effective_era: int) -> None:
        self.r._promotion_boundary_step(king_hotkey, epoch_start, round_id,
                                        effective_era=effective_era)

    def promotion_generations(self) -> tuple[int, list, int]:
        """``(generation, members[(ckpt, size)], effective_era)`` the engine
        currently holds (effective_era 0 = unknown/immediate)."""
        p = self.r.promotion
        if p is None:
            return 0, [], 0
        members = [(m.checkpoint_id, m.size) for m in getattr(p, "members", ())]
        pending = getattr(p, "pending_record", None)
        eff = int(getattr(pending, "effective_era", 0) or 0) if pending is not None else 0
        return int(getattr(p, "generation", 0) or 0), members, eff

    def record_bench_candidates(self, manifest, report) -> None:
        p = self.r.promotion
        if p is not None and report is not None:
            try:
                p.record_bench(manifest, report)
            except Exception as e:  # noqa: BLE001
                log.warning("rolling: promotion candidate recording failed (ignored): %s", e)

    def publish_champion(self, gen, round_id: str) -> None:
        self.r._maybe_publish_champion(gen, round_id)

    # queue / burn / dedup
    def queue(self):
        return self.r._funded_queue()

    def burned(self) -> set[str]:
        from .loop import _load_seen_hotkeys

        if not self.r.cfg.round.one_submission_per_hotkey:
            return set()
        return set(_load_seen_hotkeys(self.r._submissions_path()))

    def burn(self, gens: list) -> None:
        self.r._burn_hotkeys(gens)

    def vault_owned(self, gens: list) -> list:
        return self.r._verify_vault_ownership(gens)

    def dedup_registry(self):
        return self.r._rolling_dedup_registry()

    def sweep_pods(self, *, keep_round_ids: tuple[str, ...], keep_payers: set[str]) -> None:
        self.r._reconcile_funded_pods(keep_round_ids=keep_round_ids, keep_payers=keep_payers)

    def wall_seconds(self) -> float:
        return float(self.r._leg_wall_seconds(None))

    def margin_seconds(self) -> float:
        return float(self.r.FUNDED_PUBLISH_MARGIN_SECONDS)

    def fitting_skus(self) -> tuple[str, ...]:
        try:
            skus = self.r._funded_skus_for_rent()
            return tuple(self.r._skus_fitting_now(skus) or skus)
        except Exception:  # noqa: BLE001
            return ()


# ── the scheduler ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedGen:
    hotkey: str
    uid: int
    ref: str
    reveal_block: int = 0


class RollingScheduler:
    """One instance per trainer process; :meth:`tick` runs every poll."""

    def __init__(self, runner, ops: LegOps | None = None, *, state_path: Path | None = None,
                 clock=time.time) -> None:
        self.r = runner
        self.ops = ops or LegOps(runner)
        self.cfg = runner.cfg
        self.clock = clock
        self.state_path = state_path or (Path(runner.work_root) / STATE_FILE)
        self.state = load_state(self.state_path)
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}     # hotkey -> leg thread
        self._king_threads: dict[int, threading.Thread] = {}  # era index -> king leg thread
        self._started = False

    # ── persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        with self._lock:
            save_state(self.state_path, self.state)

    # ── tick ─────────────────────────────────────────────────────────────────

    def tick(self, client, block: int) -> None:
        from ..shared.config import effective_epoch_blocks

        block = int(block)
        now = self.clock()
        eb = int(effective_epoch_blocks(self.cfg.round, block))
        epoch_start = (block // eb) * eb
        era = era_for_block(self.cfg.round, block)
        self._sync_generations()
        if not self._started:
            self._startup(client, block)
            self._started = True
        # Settle BEFORE rolling: the boundary that starts era n+1 is era n's
        # last settlement (settlement_era), judged against era n's king.
        self._settle(client, block, epoch_start)
        self._roll_era(client, era, block)
        # Boundaries of the era just started (a tick that skipped the era
        # boundary lands here with them pending): settle them now, not next
        # grid step.
        self._settle(client, block, epoch_start)
        self._adopt_dethrone(client)
        self._maybe_king_legs(client, block, now)
        self._intake(client, block, now)

    # ── startup / restart ────────────────────────────────────────────────────

    def _startup(self, client, block: int) -> None:
        """Restart re-entry: keep the pods of in-flight legs and both era king
        pods, re-attach every in-flight leg (a persisted completed leg is
        reused, never retrained), sweep everything else."""
        keep_ids = tuple(str(e.base_seed) for e in (self.state.current, self.state.next) if e)
        self._seed_chain_root()
        queue = self.ops.queue()
        flights = queue.in_flight() if queue is not None else []
        self.ops.sweep_pods(keep_round_ids=keep_ids,
                            keep_payers={e.hotkey for e in flights})
        done = {(f.hotkey, f.ref) for f in self.state.finished}
        for e in flights:
            if (e.hotkey, e.ref) in done:
                continue                      # finished before the restart; settles normally
            era = self._era_state_for(e.era_index)
            if era is None and int(e.era_index) <= 0 and self.state.current is not None:
                # A leg stamped with no era (legacy carry-over dispatched at the
                # rollover tick, 2026-09-24) belongs to the era that is running
                # now — re-attach it there rather than requeue (a requeue rents
                # a second pod while the first keeps training).
                era = self.state.current
                log.warning("rolling: in-flight leg %s carries no era index — re-attached "
                            "under the current era %d", e.hotkey[:12], era.index)
            if era is None:
                log.warning("rolling: in-flight leg %s targets unknown era %d — requeued",
                            e.hotkey[:12], e.era_index)
                queue.requeue(e.hotkey, error="trainer restarted; era state lost",
                              error_class="infra", burn_attempt=False)
                continue
            gen = ResolvedGen(e.hotkey, self._uid_of(client, e.hotkey), e.ref, e.reveal_block)
            started, target = self._flight_stamp(e, era, block, queue)
            self._launch_leg(gen, era, started, target, e.label, resumed=True)

    def _flight_stamp(self, e, era: EraState, block: int, queue) -> tuple[int, int]:
        """The (train block, target boundary) a re-attached leg runs under:
        the queue's stamp where it is sound, else re-derived — the block now
        for a start outside ``era``'s window, the first settlement of ``era``
        the wall clears (its last one at worst) for a target outside it — and
        the entry re-stamped so the next restart reads a sound one.
        2026-09-24: a requeue zeroes era/target/start; a leg re-attached from
        such a stamp published ``train_block 0`` and every validator rejected
        the settlement whole (``era_entry_out_of_window``)."""
        lo, hi = era_window(self.cfg.round, era)
        started, target = int(e.started_block), int(e.target_boundary)
        started_ok = lo <= started < hi
        target_ok = target > 0 and settlement_era(self.cfg.round, target).index == era.index
        if started_ok and target_ok and int(e.era_index) == era.index:
            return started, target
        if not started_ok:
            started = int(block) if lo <= int(block) < hi else lo
        if not target_ok:
            now = self.clock()
            target = target_boundary(self.cfg.round, block_now=block, now=now,
                                     wall_seconds=self.ops.wall_seconds(),
                                     margin_seconds=self.ops.margin_seconds())
            if settlement_era(self.cfg.round, target).index != era.index:
                target = hi                       # the era's last settlement
        log.warning("rolling: in-flight leg %s carried stamp era %d / block %d / target %d, "
                    "not sound for era %d (window [%d, %d)) — re-attached at block %d "
                    "toward boundary %d", e.hotkey[:12], e.era_index, e.started_block,
                    e.target_boundary, era.index, lo, hi, started, target)
        if queue is not None and not queue.restamp_flight(
                e.hotkey, target_boundary=target, era_index=era.index, started_block=started):
            log.warning("rolling: %s's flight could not be re-stamped in the queue",
                        e.hotkey[:12])
        return started, target

    def _seed_chain_root(self) -> bool:
        """The first settlement chains to the bucket's ``latest.json`` (the
        last legacy round — every validator's ``last_handled_round_id``);
        a state file that never published seeds ``last_published_round_id``
        from it. An empty root is never published against (see _settle)."""
        if self.state.last_published_round_id:
            return True
        root = self.ops.latest_round_id()
        if not root:
            return False
        with self._lock:
            self.state.last_published_round_id = str(root)
            self._save()
        log.info("rolling: manifest chain root seeded from latest.json (round %s)", root)
        return True

    def _uid_of(self, client, hotkey: str) -> int:
        try:
            uid = client.uid_for_hotkey(hotkey)
            return -1 if uid is None else int(uid)
        except Exception:  # noqa: BLE001
            return -1

    def _era_state_for(self, index: int) -> EraState | None:
        for e in (self.state.current, self.state.next):
            if e is not None and e.index == int(index):
                return e
        return None

    # ── generations ──────────────────────────────────────────────────────────

    def _sync_generations(self) -> None:
        """Learn generations from the promotion engine: the live one (seeded
        as effective immediately when the ledger is empty) and a fired record
        with its ``effective_era``."""
        gen, members, eff = self.ops.promotion_generations()
        if gen <= 0 or not members:
            return
        key = str(gen)
        if key in self.state.generations:
            return
        eff = int(eff or 0)
        known = eff > 0
        if not known:
            # The engine clears its pending record at publish, so after a lost
            # ledger only the PUBLISHED record (what validators install from)
            # says when this generation takes effect.
            published = self.ops.promotion_effective_era(gen)
            if published is not None:
                eff, known = int(published), True
        with self._lock:
            if key in self.state.generations:
                return
            if known:
                effective = eff
            elif not self.state.generations:
                # First sight of the engine with no record readable: the live
                # generation was installed before any era (a fired-but-not-
                # yet-effective one is still the engine's pending record,
                # which carries its era) — live now.
                effective = 0
                log.warning("rolling: promotion record gen=%d unreadable with an empty "
                            "ledger — assuming the generation is already live", gen)
            else:
                # Never guess an era: the entry is skipped and the record
                # re-read next tick (a guess would be written once and never
                # revisited — the trainer then trains the generation on the
                # wrong era while validators install it on the published one).
                log.warning("rolling: promotion record gen=%d unreadable; its era stays "
                            "unresolved until the record can be read", gen)
                return
            self.state.generations[key] = {
                "members": [[c, s] for c, s in members], "effective_era": int(effective)}
            self._save()
            log.info("rolling: generation %d effective from era %d (%d member(s))",
                     gen, effective, len(members))

    # ── eras ─────────────────────────────────────────────────────────────────

    def _new_era_state(self, client, era: EraSpec) -> EraState:
        gen, idx, ckpt, size = era_init(self.state.generations, era.index)
        return EraState(index=era.index, start_block=era.start_block,
                        seed_block=era.seed_block,
                        base_seed=self.ops.block_seed(client, era.seed_block),
                        generation=gen, member_index=idx,
                        warm_start_ckpt=ckpt, warm_start_size=size)

    def _roll_era(self, client, era: EraSpec, block: int) -> None:
        with self._lock:
            cur = self.state.current
            if cur is not None and cur.index == era.index:
                return
            king_hotkey = self._king_hotkey(client) or (cur.king_hotkey if cur else "")
            nxt = self.state.next
            if nxt is not None and nxt.index == era.index and nxt.king_hotkey == king_hotkey:
                new = nxt
            else:
                if nxt is not None and nxt.index == era.index:
                    log.warning("rolling: pre-trained era %d king %s is no longer the king "
                                "(%s) — its leg is discarded", era.index,
                                nxt.king_hotkey[:12], king_hotkey[:12])
                    self.ops.retire_king_pod(nxt)
                new = self._new_era_state(client, era)
                new.king_hotkey = king_hotkey
            if cur is not None:
                self.ops.retire_king_pod(cur)
                # Legs finished under the old era but never settled (no king
                # leg all era) are judged never against another era's king.
                stale = [f for f in self.state.finished if f.era_index < era.index]
                for f in stale:
                    log.warning("rolling: %s's leg from era %d never settled — requeued "
                                "unburned", f.hotkey[:12], f.era_index)
                    q = self.ops.queue()
                    if q is not None:
                        q.requeue(f.hotkey, error="era ended without a settlement",
                                  error_class="no_capacity", burn_attempt=False)
                self.state.finished = [f for f in self.state.finished
                                       if f.era_index >= era.index]
                self.state.published = []
            self.state.current = new
            self.state.next = None
            self._save()
            log.info("rolling: era %d started at block %d (seed block %d, generation %d "
                     "member %d, king %s%s)", era.index, era.start_block, era.seed_block,
                     new.generation, new.member_index, (new.king_hotkey or "?")[:12],
                     " — cached king leg" if new.king_entry else "")

    def _king_hotkey(self, client) -> str | None:
        """The king per the validators' receipts; the metagraph only cross-
        checks (it lags a dethrone longer than a settlement)."""
        receipt = self.ops.receipt_king()
        chain = self.ops.metagraph_king(client)
        if receipt and chain and receipt != chain:
            log.info("rolling: receipt king %s != metagraph king %s (incentive lag; "
                     "the receipt decides)", receipt[:12], chain[:12])
        return receipt or chain

    # ── king legs ────────────────────────────────────────────────────────────

    def _maybe_king_legs(self, client, block: int, now: float) -> None:
        cur = self.state.current
        if cur is None:
            return
        if cur.king_entry is None and cur.king_hotkey and cur.index not in self._king_threads:
            self._launch_king_leg(client, cur, block, now)
        wall, margin = self.ops.wall_seconds(), self.ops.margin_seconds()
        if (self.state.next is None
                and king_pretrain_open(self.cfg.round, block_now=block, now=now,
                                       wall_seconds=wall, margin_seconds=margin)):
            nxt_spec = era_for_block(self.cfg.round, next_era_start(self.cfg.round, block))
            with self._lock:
                nxt = self._new_era_state(client, nxt_spec)
                nxt.king_hotkey = cur.king_hotkey
                nxt.king_uid, nxt.king_ref = cur.king_uid, cur.king_ref
                self.state.next = nxt
                self._save()
            log.info("rolling: pre-train window for era %d open — king leg starts",
                     nxt.index)
            self._launch_king_leg(client, nxt, block, now)
        elif (self.state.next is not None and self.state.next.king_entry is None
              and self.state.next.index not in self._king_threads):
            self._launch_king_leg(client, self.state.next, block, now)

    def king_end_wall(self, era: EraState, *, block: int, now: float) -> float:
        """The king leg's own target: the wall-clock of ``era``'s LAST
        settlement (its end block). Like a challenger's target boundary this
        bounds the rent wait — a sold-out marketplace is waited out up to
        the king's latest safe start (end − wall − margin), not given up on
        at the first empty listing (2026-09-21 testnet: era 13427/13428
        king legs failed instantly with "no RTX4090 capacity before the
        round's latest safe start" because the king path had no target and
        the deadline helper fell back to "now")."""
        end_block = int(era.start_block) + era_length_blocks(self.cfg.round, int(era.start_block))
        return wall_of_block(end_block, now=now, block_now=block)

    def _launch_king_leg(self, client, era: EraState, block: int, now: float) -> None:
        if not era.king_hotkey:
            return
        if not era.king_ref:
            ref = self._current_ref(client, era.king_hotkey, block)
            if not ref:
                log.warning("rolling: king %s has no revealed generator; era %d waits",
                            era.king_hotkey[:12], era.index)
                return
            era.king_ref = ref
            era.king_uid = self._uid_of(client, era.king_hotkey)
            self._save()
        gen = ResolvedGen(era.king_hotkey, era.king_uid, era.king_ref)
        cached = self.ops.cached_leg(era, "king", gen)
        if cached is not None:
            with self._lock:
                era.king_entry = _entry_to_json(replace(cached, role="king"))
                era.king_init = era.warm_start_ckpt     # cached_leg matched on it
                self._save()
            log.info("rolling: era %d king leg already complete (cached) — reused", era.index)
            return

        end_wall = self.king_end_wall(era, block=block, now=now)

        def _run() -> None:
            try:
                entry = self.ops.train_king(gen, era, block, end_wall=end_wall)
                entry = replace(entry, role="king")
                with self._lock:
                    era.king_entry = _entry_to_json(entry)
                    era.king_init = era.warm_start_ckpt
                    era.king_leg_failed = ""
                    era.king_leg_failures = 0
                    self._save()
                log.info("rolling: era %d king leg complete: %s", era.index,
                         entry.trained_pointer)
                bench = self.ops.bench_king(entry, era)
                with self._lock:
                    era.king_bench = bench
                    self._save()
            except Exception as e:  # noqa: BLE001 — retried next tick; never kills the loop
                log.error("rolling: era %d king leg FAILED: %s", era.index, e)
                with self._lock:
                    era.king_leg_failed = error_tail(e, 300)
                    era.king_leg_failures += 1
                    failures = era.king_leg_failures
                    self._save()
                if king_pod_should_rotate(e, failures):
                    reason = (f"era {era.index} king leg failed {failures}x on this pod: "
                              f"{error_tail(e, 160)}")
                    log.warning("rolling: era %d king pod ROTATES after %d failure(s) — "
                                "host quarantined, next tick re-rents elsewhere",
                                era.index, failures)
                    try:
                        self.ops.rotate_king_pod(era, reason)
                    except Exception as e2:  # noqa: BLE001 — the retry still re-rents
                        log.error("rolling: era %d king pod rotation failed: %s",
                                  era.index, e2)
                    with self._lock:
                        era.king_leg_failures = 0
                        self._save()
            finally:
                self._king_threads.pop(era.index, None)

        t = threading.Thread(target=_run, name=f"king-era{era.index}", daemon=True)
        self._king_threads[era.index] = t
        t.start()

    def _current_ref(self, client, hotkey: str, block: int) -> str | None:
        from ..validator.loop import ref_as_of

        try:
            return ref_as_of(self.ops.commitments(client), hotkey, int(block))
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: commitments unavailable (%s)", e)
            return None

    # ── intake ───────────────────────────────────────────────────────────────

    def _intake(self, client, block: int, now: float) -> None:
        queue = self.ops.queue()
        cur = self.state.current
        if queue is None or cur is None:
            return
        from ..funding.queue import select_field

        try:
            history = self.ops.commitments(client)
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: commitments unavailable (%s); intake waits", e)
            return
        from ..validator.loop import ref_as_of

        # Pending reveals promote from chain truth (the intake's backstop).
        queue.promote_pending(lambda hk, ref: self._reveal_block(history, hk, ref))
        queue.expire_stale()
        cap = int(self.cfg.round.funded_field_cap or 0) or max(1, int(self.cfg.round.finalist_cap))
        in_flight = len(queue.in_flight())
        burned = self.ops.burned()
        wall, margin = self.ops.wall_seconds(), self.ops.margin_seconds()
        held: list[str] = []
        not_jumps: set[str] = set()       # seniors that could not start this pass
        for entry in select_field(queue.entries(), cap=0):
            if in_flight >= cap:
                held.append(entry.hotkey)
                not_jumps.add(entry.hotkey)
                continue
            if entry.hotkey in burned:
                queue.fail(entry.hotkey, error="hotkey already used its one lifetime "
                           "submission", error_class="burned", expect_ref=entry.ref)
                continue
            revealed = ref_as_of(history, entry.hotkey, block)
            if revealed is None:
                held.append(entry.hotkey)          # funded, not yet revealed: waits
                not_jumps.add(entry.hotkey)
                continue
            if revealed != entry.ref:
                queue.fail(entry.hotkey, error=f"funded ref {entry.ref} no longer matches "
                           f"the revealed {revealed} — fund the new ref",
                           error_class="ref_mismatch", expect_ref=entry.ref)
                continue
            adm = admit(self.cfg.round, block_now=block, now=now, wall_seconds=wall,
                        margin_seconds=margin, current_era=cur.index)
            if adm.start_after > now:
                held.append(entry.hotkey)          # waits for the pre-train window
                not_jumps.add(entry.hotkey)
                continue
            era = self._era_state_for(adm.era_index)
            if era is None:
                if adm.era_index == cur.index + 1:
                    nxt = era_for_block(self.cfg.round, next_era_start(self.cfg.round, block))
                    with self._lock:
                        era = self._new_era_state(client, nxt)
                        era.king_hotkey, era.king_uid, era.king_ref = (
                            cur.king_hotkey, cur.king_uid, cur.king_ref)
                        self.state.next = era
                        self._save()
                else:
                    held.append(entry.hotkey)
                    not_jumps.add(entry.hotkey)
                    continue
            gen = ResolvedGen(entry.hotkey, self._uid_of(client, entry.hotkey), entry.ref,
                              entry.reveal_block)
            if self.ops.vault_owned([gen]) != [gen] and not self._vault_owned_ok(gen):
                queue.fail(entry.hotkey, error="vault ref not owned by this hotkey",
                           error_class="ref_mismatch", expect_ref=entry.ref)
                continue
            dup = self._dedup_check(gen, era, history)
            if dup is not None:
                not_jumps.add(entry.hotkey)
                continue
            if not queue.mark_in_flight(entry.hotkey, entry.ref,
                                        target_boundary=adm.target_boundary,
                                        era_index=adm.era_index, started_block=block):
                not_jumps.add(entry.hotkey)
                continue
            self._record_rent(entry, adm, queue, not_jumps)
            in_flight += 1
            self._launch_leg(gen, era, block, adm.target_boundary, entry.label)
        if held:
            queue.touch(held)

    def _vault_owned_ok(self, gen) -> bool:
        from ..funding.store import parse_vault_ref

        return parse_vault_ref(gen.ref) is None

    @staticmethod
    def _reveal_block(history: list, hotkey: str, ref: str) -> int | None:
        from ..interface.validation import parse_commit

        for c in history:
            if c.hotkey != hotkey:
                continue
            parsed = parse_commit(c.payload)
            if parsed is not None and parsed.ref == ref:
                return int(c.commit_block)
        return None

    def _record_rent(self, entry, adm: Admission, queue, not_jumps: set[str]) -> None:
        """Seniority evidence for the roster: at this rent, which more-senior
        queued entries (earlier reveal) that COULD have started were passed
        over. A senior that could not start this pass — unrevealed, waiting
        for its pre-train window, held at the cap, dropped by dedup — is not
        a jump (DEC-CA-0043 "first pick of fitting executors"); the drain is
        in reveal order among the startable, so this list is empty unless the
        order was broken."""
        from ..funding.queue import select_field

        ahead = [e.hotkey for e in select_field(queue.entries(), cap=0)
                 if e.status == "queued" and e.hotkey not in not_jumps
                 and (e.reveal_block, e.hotkey) < (entry.reveal_block, entry.hotkey)]
        with self._lock:
            self.state.rents.append({
                "hotkey": entry.hotkey, "reveal_block": int(entry.reveal_block),
                "skus": list(self.ops.fitting_skus()),
                "target_boundary": int(adm.target_boundary), "era_index": int(adm.era_index),
                "passed_over": ahead, "at": self.clock()})
            self._save()

    def _dedup_check(self, gen, era: EraState, history: list) -> str | None:
        """Persistent exact-identity dedup (DEC-CA-0008 tiers) over the queue,
        in-flight legs, the era king and published champions; earliest
        COMMIT wins. Returns the matched hotkey when ``gen`` is dropped."""
        reg = self.ops.dedup_registry()
        if reg is None:
            return None
        try:
            verdict = reg.admit(gen, era, history)
        except Exception as e:  # noqa: BLE001 — fail open like the boundary screen
            log.warning("rolling: dedup registry failed for %s (%s); admitted unscreened",
                        gen.hotkey[:12], e)
            return None
        if verdict is None:
            return None
        matched, tier, enforce = verdict
        if not enforce:
            log.info("rolling: dedup shadow — %s duplicates %s (%s); admitted",
                     gen.hotkey[:12], matched[:12], tier)
            return None
        q = self.ops.queue()
        if q is not None:
            what = ("contains an earlier private submission's module"
                    if tier == "private_copy" else "duplicates")
            q.fail(gen.hotkey, error=f"generator {what} {matched} ({tier}); the "
                   "earliest commit keeps the entry", error_class="duplicate",
                   expect_ref=gen.ref)
        self.ops.burn([gen])
        log.warning("rolling: %s dropped — duplicates %s (%s)", gen.hotkey[:12],
                    matched[:12], tier)
        return matched

    # ── legs ─────────────────────────────────────────────────────────────────

    def _launch_leg(self, gen, era: EraState, block: int, target: int, label: str,
                    *, resumed: bool = False) -> None:
        if gen.hotkey in self._threads:
            return
        queue = self.ops.queue()
        end_wall = wall_of_block(target, now=self.clock(), block_now=block)

        def _finish(entry: TrainedEntry) -> None:
            king = era.king()
            bench = None
            try:
                bench = self.ops.bench_challenger(entry, king, era)
            except Exception as e:  # noqa: BLE001 — bench is telemetry, never the leg
                log.warning("rolling: bench for %s failed (%s)", gen.hotkey[:12], e)
            finally:
                try:
                    self.ops.teardown_kept_pod(gen.hotkey)
                except Exception as e:  # noqa: BLE001
                    log.error("rolling: kept pod teardown for %s failed: %s",
                              gen.hotkey[:12], e)
            with self._lock:
                self.state.finished.append(FinishedLeg(
                    hotkey=gen.hotkey, uid=gen.uid, ref=gen.ref, era_index=era.index,
                    started_block=block, reveal_block=gen.reveal_block,
                    entry=_entry_to_json(entry), bench=bench, label=label))
                self._save()
            log.info("rolling: %s's leg finished (era %d, target boundary %d)%s",
                     gen.hotkey[:12], era.index, target,
                     "" if bench else " — no bench numbers")

        def _run() -> None:
            try:
                cached = self.ops.cached_leg(era, "challenger", gen)
                if cached is not None:
                    log.info("rolling: %s's leg already complete (persisted) — reused",
                             gen.hotkey[:12])
                    _finish(cached)
                    return
                entry = self.ops.train_challenger(gen, era, block, end_wall=end_wall)
                _finish(entry)
            except Exception as e:  # noqa: BLE001 — settle the leg from its outcome
                msg, miner_fault, error_class, burn = self.ops.leg_failure(gen.hotkey)
                if queue is not None:
                    if miner_fault:
                        queue.fail(gen.hotkey, error=msg or str(e), error_class=error_class,
                                   expect_ref=gen.ref)
                        if error_class in ("generator", "tamper"):
                            self.ops.burn([gen])
                    else:
                        queue.requeue(gen.hotkey, error=msg or str(e),
                                      error_class=error_class, burn_attempt=burn)
                log.warning("rolling: %s's leg failed [%s]: %s", gen.hotkey[:12],
                            error_class, error_tail(msg or str(e), 200))
            finally:
                self._threads.pop(gen.hotkey, None)

        t = threading.Thread(target=_run, name=f"leg-{gen.hotkey[:12]}", daemon=True)
        self._threads[gen.hotkey] = t
        t.start()
        log.info("rolling: %s's leg %s (era %d, target boundary %d)", gen.hotkey[:12],
                 "re-attached" if resumed else "dispatched", era.index, target)

    # ── dethrone adoption ────────────────────────────────────────────────────

    def _adopt_dethrone(self, client) -> None:
        cur = self.state.current
        if cur is None:
            return
        king = self._king_hotkey(client)
        if not king or king == cur.king_hotkey:
            return
        winner = next((f for f in reversed(self.state.published) if f.hotkey == king), None)
        with self._lock:
            if winner is None:
                log.warning("rolling: receipts name %s king but no settled leg of this era "
                            "is theirs — era %d king leg restarts for them",
                            king[:12], cur.index)
                cur.king_hotkey, cur.king_uid, cur.king_ref = king, -1, ""
                cur.king_entry, cur.king_bench, cur.king_bench_published = None, None, False
                cur.king_init = ""
            else:
                entry = replace(_entry_from_json(winner.entry), role="king")
                cur.king_hotkey, cur.king_uid, cur.king_ref = king, winner.uid, winner.ref
                cur.king_entry = _entry_to_json(entry)
                cur.king_init = cur.warm_start_ckpt     # a settled leg of THIS era
                cur.king_bench = winner.bench
                cur.king_bench_published = winner.bench is not None
                log.info("rolling: DETHRONE — %s's checkpoint %s adopted as era %d king "
                         "(era continues)", king[:12], entry.trained_pointer, cur.index)
            if self.state.next is not None and self.state.next.king_hotkey != king:
                self.ops.retire_king_pod(self.state.next)
                self.state.next = None
            self._save()
        self.r._rolling_note_king_host(cur)

    # ── settlement ───────────────────────────────────────────────────────────

    def _settle(self, client, block: int, epoch_start: int) -> None:
        """Every boundary since the last settled one, OLDEST FIRST. A trainer
        outage across a boundary publishes that settlement late, stamped with
        the boundary it belongs to (``created_block`` = the boundary —
        validators floor it to the grid, so it resolves to the era the legs
        trained under) instead of throwing the finished legs away and
        re-billing their payers."""
        from ..shared.config import effective_epoch_blocks

        cur = self.state.current
        if cur is None:
            return
        start = int(cur.start_block)
        era_end = start + int(era_length_blocks(self.cfg.round, start))
        b = int(self.state.last_settled_boundary)
        if b < start:
            # Never settled under this era: the first boundary stepped is the
            # era's start itself (the reign clock ticks there; it is the
            # previous era's last settlement, nothing of ours).
            b = start - int(effective_epoch_blocks(self.cfg.round, max(0, start - 1)))
        while True:
            b += int(effective_epoch_blocks(self.cfg.round, b))
            if b > int(epoch_start) or b > era_end:
                break              # a later era's boundary settles after the roll
            if b < int(epoch_start):
                log.warning("rolling: boundary %d passed while the trainer was away — "
                            "settling it late (block %d)", b, block)
            self._settle_boundary(client, b, created_block=(b if b < int(epoch_start)
                                                            else int(block)))

    def _settle_boundary(self, client, epoch_start: int, *, created_block: int) -> None:
        cur = self.state.current
        if cur is None or epoch_start <= self.state.last_settled_boundary:
            return
        base_seed = self.ops.block_seed(client, epoch_start)
        round_id = str(base_seed)
        # The reign clock ticks at EVERY boundary, settlement or not.
        try:
            self.ops.promotion_boundary(cur.king_hotkey, epoch_start, round_id,
                                        min_effective_era(self.cfg.round, epoch_start))
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: promotion step failed at %d: %s", epoch_start, e)
        self._sync_generations()
        if settlement_era(self.cfg.round, epoch_start).index != cur.index:
            # A boundary before this era's first settlement (the rollover
            # boundary itself): nothing of ours to settle there.
            self.state.last_settled_boundary = epoch_start
            self._save()
            return
        with self._lock:
            ready = [f for f in self.state.finished if f.era_index == cur.index]
            king = cur.king()
            if king is not None and cur.king_init != cur.warm_start_ckpt:
                # The regulator: a king entry must have trained from the era's
                # init — the same one every challenger in the manifest trained
                # from — or the duel is not like-for-like. Drop it and let the
                # king leg retrain (next tick, init-checked cache); the
                # finished legs wait. Nothing is published this boundary.
                log.error("rolling: boundary %d — era %d king entry %s trained from init "
                          "%r but the era's init is %r; dropped, king leg retrains, %d "
                          "finished leg(s) wait, nothing published", epoch_start, cur.index,
                          king.trained_pointer, cur.king_init or "<random init>",
                          cur.warm_start_ckpt or "<random init>", len(ready))
                cur.king_entry, cur.king_bench, cur.king_bench_published = None, None, False
                cur.king_init = ""
                king = None
            if king is None:
                log.info("rolling: boundary %d — era %d has no king leg yet; %d finished "
                         "leg(s) wait, nothing published", epoch_start, cur.index, len(ready))
                self.state.last_settled_boundary = epoch_start
                self._save()
                return
            if not ready:
                log.info("rolling: boundary %d — nothing finished since the last "
                         "settlement; no manifest", epoch_start)
                self.state.last_settled_boundary = epoch_start
                self._save()
                return
        lo, hi = era_window(self.cfg.round, cur)
        if not (lo <= int(king.train_block) < hi):
            # Validators reject a settlement whole for ONE entry outside the
            # window; a king entry outside it can only come from a stamp bug
            # or a reused record of another era — retrain it, publish nothing.
            log.error("rolling: boundary %d — era %d's king entry was trained at block %d, "
                      "outside the era window [%d, %d); king entry dropped, the king leg "
                      "retrains; %d finished leg(s) wait, nothing published",
                      epoch_start, cur.index, int(king.train_block), lo, hi, len(ready))
            with self._lock:
                cur.king_entry, cur.king_bench, cur.king_bench_published = None, None, False
                self.state.last_settled_boundary = epoch_start
                self._save()
            self.ops.discard_cached_leg(cur, "king", ResolvedGen(cur.king_hotkey, cur.king_uid,
                                                                 cur.king_ref))
            return
        bad = [f for f in ready if not (lo <= int(f.entry.get("train_block", 0) or 0) < hi)]
        if bad:
            queue = self.ops.queue()
            for f in bad:
                tb = int(f.entry.get("train_block", 0) or 0)
                log.error("rolling: boundary %d — %s's leg was trained at block %d, outside "
                          "era %d's window [%d, %d): a trainer fault, never the miner's — "
                          "dropped before publication, record discarded, requeued unburned "
                          "(retrains)", epoch_start, f.hotkey[:12], tb, cur.index, lo, hi)
                self.ops.discard_cached_leg(cur, "challenger",
                                            ResolvedGen(f.hotkey, f.uid, f.ref, f.reveal_block))
                if queue is not None:
                    queue.requeue(f.hotkey, error=f"trained at block {tb}, outside era "
                                  f"{cur.index}'s window [{lo}, {hi}) — trainer fault, retrained",
                                  error_class="infra", burn_attempt=False)
            with self._lock:
                self.state.finished = [f for f in self.state.finished if f not in bad]
                ready = [f for f in ready if f not in bad]
                if not ready:
                    self.state.last_settled_boundary = epoch_start
                self._save()
            if not ready:
                return
        if not self._seed_chain_root():
            # An unchained manifest is rejected by every validator AFTER the
            # legs are marked done and burned — never publish one. The legs
            # wait for the next boundary; the root is retried every tick.
            log.error("rolling: boundary %d — manifest chain root unknown (latest.json "
                      "unreadable); %d finished leg(s) wait, nothing published",
                      epoch_start, len(ready))
            return
        with self._lock:
            manifest = self._build_manifest(cur, king, ready, round_id, created_block,
                                            pin_block=epoch_start)
        self.ops.publish_manifest(manifest)
        bench_entries = self._bench_entries(cur, king, ready)
        report = None
        if bench_entries:
            report = self.ops.publish_bench(round_id, created_block, bench_entries)
            if report is not None:
                cur.king_bench_published = cur.king_bench_published or cur.king_bench is not None
        self.ops.record_bench_candidates(manifest, report)
        queue = self.ops.queue()
        gens = []
        for f in ready:
            if queue is not None:
                queue.mark_done(f.hotkey)
            gens.append(ResolvedGen(f.hotkey, f.uid, f.ref, f.reveal_block))
        if gens:
            self.ops.burn(gens)
        with self._lock:
            self.state.finished = [f for f in self.state.finished if f not in ready]
            self.state.published.extend(ready)
            self.state.last_published_round_id = round_id
            self.state.last_settled_boundary = epoch_start
            roster = self._roster(round_id, ready, queue)
            self.state.rents = []
            self._save()
        self.ops.publish_roster(round_id, roster)
        try:
            self.ops.publish_champion(ResolvedGen(cur.king_hotkey, cur.king_uid, cur.king_ref),
                                      round_id)
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: champion publish failed (ignored): %s", e)
        log.info("rolling: SETTLEMENT %s at boundary %d — era %d king %s + %d challenger(s)",
                 round_id, epoch_start, cur.index, cur.king_hotkey[:12], len(ready))

    def _build_manifest(self, era: EraState, king: TrainedEntry, ready: list[FinishedLeg],
                        round_id: str, block: int, *, pin_block: int):
        from ..shared.manifest import TrainingManifest, contract_digest, contract_payload

        ordered = sorted(ready, key=lambda f: (f.reveal_block, f.hotkey))
        entries = [king]
        for i, f in enumerate(ordered):
            entries.append(replace(_entry_from_json(f.entry), role="challenger",
                                   duel_rank=(i if len(ordered) > 1 else 0)))
        pool_key, pool_sha = "", ""
        fn = getattr(self.r, "pool_provenance_fn", None)
        if fn is not None:
            try:
                # The pin is the daily snapshot at the SETTLEMENT boundary —
                # the block validators verify it at (an era can span a
                # snapshot's effective block; the era start would then fail
                # the pin on every settlement after the crossing).
                pool_key, pool_sha = fn(int(round_id), int(pin_block))
            except Exception as e:  # noqa: BLE001
                log.warning("rolling: eval-pool pin unavailable (%s)", e)
        return TrainingManifest(
            round_id=round_id, created_block=int(block),
            contract_digest=contract_digest(self.cfg.training),
            base_arch_digest=self.cfg.training.base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset, entries=entries,
            eval_pool_key=str(pool_key or ""), eval_pool_sha256=str(pool_sha or ""),
            warm_start_ckpt=era.warm_start_ckpt, warm_start_size=era.warm_start_size,
            contract_body=contract_payload(self.cfg.training),
            era=era.spec().to_json(),
            prev_round_id=self.state.last_published_round_id,
        )

    def _bench_entries(self, era: EraState, king: TrainedEntry, ready: list[FinishedLeg]) -> list:
        """``(entry, BenchScores)`` pairs for the settlement's bench report:
        the king's numbers once per era (its first report), every benched
        challenger every time."""
        from ..shared.manifest import BenchScores

        out = []
        if era.king_bench is not None and not era.king_bench_published:
            out.append((king, BenchScores(**era.king_bench)))
        for f in ready:
            if f.bench is not None:
                out.append((_entry_from_json(f.entry), BenchScores(**f.bench)))
        return out

    def _roster(self, round_id: str, ready: list[FinishedLeg], queue) -> dict:
        from ..funding.queue import select_field

        entries = queue.entries() if queue is not None else []
        seated = [{"hotkey": f.hotkey, "ref": f.ref, "reveal_block": f.reveal_block,
                   "label": f.label, "started_block": f.started_block}
                  for f in sorted(ready, key=lambda f: (f.reveal_block, f.hotkey))]
        return {
            "round_id": round_id, "mode": "rolling",
            "seated": seated,
            "in_flight": [{"hotkey": e.hotkey, "reveal_block": e.reveal_block,
                           "target_boundary": e.target_boundary, "era_index": e.era_index,
                           "label": e.label} for e in entries if e.status == "in_flight"],
            "waiting": [{"hotkey": e.hotkey, "reveal_block": e.reveal_block, "label": e.label}
                        for e in select_field(entries, cap=0)],
            "terminal": [{"hotkey": e.hotkey, "error_class": e.last_error_class}
                         for e in entries if e.status == "failed"],
            "rents": list(self.state.rents),
        }
