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

# ``bench_challenger`` returns ``{VERIFY_PENDING: payer_scores}`` when the
# payer-pod numbers exist but the operator's verification cannot run yet.
VERIFY_PENDING = "_verify_pending"
# Verification attempts that RAN and failed (push/sweep error, no scores)
# before a queued entry is given up; waiting for a king leg/pod costs none.
MAX_VERIFY_TRIES = 3
# Settlement bench reports kept for late additions (4 eras of 4 settlements).
KEEP_BENCH_REPORTS = 16

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
class LegWalls:
    """Measured wall per generator ref (`<work_root>/leg_walls.json`): what a
    leg of this ref actually took last time, dispatch → finish. The SKU wall
    table is a per-GPU estimate; a CPU-bound generator ran 1.9× it on a 4090
    (2026-09-28: a 5 h leg landed after its era ended and was lost). Admission
    fits a KNOWN ref on ``max(sku wall, measured × 1.1)``; an unseen ref keeps
    the SKU estimate. Best-effort file; a read/write error never touches a leg."""

    SAFETY = 1.10

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _load(self) -> dict:
        try:
            return dict(json.loads(self.path.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            return {}

    def record(self, ref: str, seconds: float, *, sku: str = "") -> None:
        try:
            d = self._load()
            d[str(ref)] = {"seconds": float(seconds), "sku": str(sku or ""), "at": time.time()}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(d, sort_keys=True, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except Exception as e:  # noqa: BLE001
            log.warning("leg walls: could not record %s (%s)", str(ref)[-16:], e)

    def estimate(self, ref: str) -> float | None:
        v = self._load().get(str(ref))
        try:
            return float(v["seconds"]) * self.SAFETY if v else None
        except Exception:  # noqa: BLE001
            return None

    def fit_wall(self, ref: str, sku_wall: float) -> float:
        est = self.estimate(ref)
        return max(float(sku_wall), est) if est is not None else float(sku_wall)


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
    # Deferred operator verification of payer-pod benches: {pointer, hotkey,
    # entry, payer, round_id ("" until the leg settles), tries}. A leg whose
    # verification cannot run when it finishes (no era king leg yet, king pod
    # busy/gone) waits here instead of losing its bench numbers.
    verify_queue: list[dict] = field(default_factory=list)
    # The signed bench report of each recent settlement, so a late-verified
    # number is ADDED to its own round's report (validators join bench numbers
    # on (round_id, trained_pointer) and re-probe that report): {round_id:
    # {"created_block": n, "pairs": [[entry, scores], ...]}}.
    bench_reports: dict[str, dict] = field(default_factory=dict)


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
        verify_queue=[dict(x) for x in raw.get("verify_queue", [])],
        bench_reports={str(k): dict(v) for k, v in (raw.get("bench_reports") or {}).items()},
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
        "verify_queue": state.verify_queue,
        "bench_reports": state.bench_reports,
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

    def _era_contract(self, era: EraState):
        """The contract a leg of ``era`` trains and is persisted under — the
        one effective at the era's start (DEC-CA-0047 scheduled switch), the
        same ``_rolling_train_*`` persist with. Looking a record up under the
        base contract misses every leg trained after a switch (2026-09-28
        17:54: era 2546's landed king leg was retrained instead of reused)."""
        return self.r.cfg.throne_contracts_at(int(era.start_block))[0]

    def cached_leg(self, era: EraState, role: str, gen) -> TrainedEntry | None:
        contract = self._era_contract(era)
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
        contract = self._era_contract(era)
        suffix = "" if role == "king" else f"-u{gen.uid}"
        self.r._discard_completed_leg(round_id=era.base_seed, contract=contract,
                                      role=role, hotkey=gen.hotkey, suffix=suffix)

    def bench_challenger(self, entry: TrainedEntry, king: TrainedEntry | None,
                         era: EraState) -> dict | None:
        return self.r._rolling_bench_challenger(entry, king, era)

    def bench_king(self, entry: TrainedEntry, era: EraState) -> dict | None:
        return self.r._rolling_bench_king(entry, era)

    def verify_bench(self, entry: TrainedEntry, payer: dict, king: TrainedEntry,
                     era: EraState) -> tuple[str, dict | None]:
        return self.r._rolling_verify_payer(entry, payer, king, era)

    def leg_failure(self, hotkey: str) -> tuple[str, bool, str, bool]:
        return self.r._funded_leg_failures.get(
            hotkey, ("challenger leg failed before dispatch", False, "infra", True))

    def teardown_kept_pod(self, hotkey: str) -> None:
        self.r._teardown_kept_funded_pod(hotkey)

    def retire_king_pod(self, era: EraState) -> None:
        self.r._rolling_retire_king_pod(era)

    def release_idle_king_pod(self, era: EraState) -> None:
        """Same teardown as :meth:`retire_king_pod` (bench-lock aware), kept
        separate so an idle release is never read as a throne/king change."""
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
        self._verify_thread: threading.Thread | None = None   # one operator verification at a time
        self._idle_released: set[int] = set()   # eras whose idle king pod was released early
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
        self._requeue_stale_finished()
        # Boundaries of the era just started (a tick that skipped the era
        # boundary lands here with them pending): settle them now, not next
        # grid step.
        self._settle(client, block, epoch_start)
        self._adopt_dethrone(client)
        self._maybe_king_legs(client, block, now)
        self._drain_verifies()
        self._release_idle_king_pods()
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
        done = {(f.hotkey, f.ref) for f in self.state.finished}
        dead = self._dead_era_flights(client, flights, done, block)
        self.ops.sweep_pods(keep_round_ids=keep_ids,
                            keep_payers={e.hotkey for e in flights} - set(dead))
        for e in flights:
            if (e.hotkey, e.ref) in done:
                continue                      # finished before the restart; settles normally
            if e.hotkey in dead:
                era_idx, era_end = dead[e.hotkey]
                # Its era ended while the trainer was away and no checkpoint
                # landed: the leg can never be judged (a leg that finishes
                # after its era is requeued at finish anyway), so re-attaching
                # only bills the payer for a second run — or a fresh pod when
                # the first is gone (2026-09-28 13:06: two era-2545 legs were
                # re-rented for a dead era, one of them then burned for a NaN
                # on the new host). Pod released by the sweep above; requeue
                # unburned now.
                log.warning("rolling: in-flight leg %s belongs to era %d, which ended at "
                            "block %d (now %d) — it can never be judged; pod released, "
                            "requeued unburned", e.hotkey[:12], era_idx, era_end, int(block))
                queue.requeue(e.hotkey, error="trainer restarted after the leg's era ended",
                              error_class="no_capacity", burn_attempt=False)
                continue
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

    def _dead_era_flights(self, client, flights, done: set, block: int) -> dict[str, tuple[int, int]]:
        """In-flight legs whose era's last boundary is already behind ``block``
        and that left no persisted checkpoint: ``hotkey -> (era index, era
        end block)``. They are requeued unburned at restore instead of being
        re-attached (or re-rented)."""
        dead: dict[str, tuple[int, int]] = {}
        for e in flights:
            if (e.hotkey, e.ref) in done:
                continue
            era = self._era_state_for(e.era_index)
            if era is None:
                continue
            start = int(era.start_block)
            era_end = start + int(era_length_blocks(self.cfg.round, start))
            if int(block) < era_end:
                continue
            gen = ResolvedGen(e.hotkey, self._uid_of(client, e.hotkey), e.ref, e.reveal_block)
            if self.ops.cached_leg(era, "challenger", gen) is not None:
                continue                      # a landed checkpoint settles late, never retrains
            dead[e.hotkey] = (int(era.index), era_end)
        return dead

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
                self._requeue_stale_finished(era.index)
                self.state.published = []
            self.state.current = new
            self.state.next = None
            self._save()
            log.info("rolling: era %d started at block %d (seed block %d, generation %d "
                     "member %d, king %s%s)", era.index, era.start_block, era.seed_block,
                     new.generation, new.member_index, (new.king_hotkey or "?")[:12],
                     " — cached king leg" if new.king_entry else "")

    def _requeue_stale_finished(self, era_index: int | None = None) -> None:
        """Requeue (unburned) every finished-but-unsettled leg of an era older
        than ``era_index`` (default: the current era) and drop it from
        ``state.finished``: a dead era has no settlement left, so the leg can
        never be judged. Called at the rollover and on every tick — a leg
        recorded under a dead era by any other path (2026-09-28 22:24: three
        legs whose bench outlived their era) is swept up at once, not at the
        next rollover 12 h later."""
        if era_index is None:
            if self.state.current is None:
                return
            era_index = int(self.state.current.index)
        with self._lock:
            stale = [f for f in self.state.finished if f.era_index < int(era_index)]
            if not stale:
                return
            self.state.finished = [f for f in self.state.finished
                                   if f.era_index >= int(era_index)]
        q = self.ops.queue()
        for f in stale:
            log.warning("rolling: %s's leg from era %d never settled — requeued "
                        "unburned", f.hotkey[:12], f.era_index)
            if q is not None:
                q.requeue(f.hotkey, error="era ended without a settlement",
                          error_class="no_capacity", burn_attempt=False)
        self._save()

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

    def _drop_foreign_king_entry(self, era: EraState | None) -> bool:
        """A king entry is only ever the era king's own leg. A dethrone adopted
        while the previous king's leg was still training (``_adopt_dethrone``
        cannot stop a running thread) used to let that leg land as the NEW
        king's entry: the settlement then published the old king's checkpoint
        under a crowned challenger, the validators held on king_resyncing,
        and the new king's leg never trained that era (2026-09-29 era 2549).
        Drop it so the king leg retrains for the real king. Returns True when
        an entry was dropped. Caller need not hold the lock."""
        if era is None or not era.king_entry or not era.king_hotkey:
            return False
        entry = _entry_from_json(era.king_entry)
        if entry.miner_hotkey == era.king_hotkey:
            return False
        with self._lock:
            log.error("rolling: era %d king entry %s belongs to %s but the era king is %s "
                      "— dropped, the king leg retrains for %s", era.index,
                      entry.trained_pointer, entry.miner_hotkey[:12], era.king_hotkey[:12],
                      era.king_hotkey[:12])
            era.king_entry, era.king_bench, era.king_bench_published = None, None, False
            era.king_init = ""
            self._save()
        return True

    def _maybe_king_legs(self, client, block: int, now: float) -> None:
        cur = self.state.current
        if cur is None:
            return
        self._drop_foreign_king_entry(cur)
        self._drop_foreign_king_entry(self.state.next)
        if (cur.king_entry is not None and cur.index not in self._king_threads
                and self._forfeit_switch(cur)):      # armed mid-reign: the running era switches too
            self.ops.retire_king_pod(cur)
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

    def _forfeit_gate_block(self, era: EraState) -> int:
        """The block ``era`` is judged at for a forfeiture: its LAST settlement
        (the era end). A forfeiture that lands anywhere inside the era hands
        the throne over for the rest of it — the validators crown the
        successor at that boundary and hold until this king leg lands."""
        return int(era.start_block) + int(era_length_blocks(self.cfg.round, int(era.start_block)))

    def _forfeit_switch(self, era: EraState) -> bool:
        """DEC-CA-0048: hand ``era``'s throne to the named successor when any
        of its settlements is judged under a forfeiture that lists its king.
        Returns True when the era's king changed (its king leg must be
        (re)trained). With no successor named the era trains NO king leg
        and waits for the validators' receipts (a vacant throne)."""
        from ..shared.era import forfeit_successor, forfeited_hotkeys

        scoring = self.r.cfg.scoring
        gate_block = self._forfeit_gate_block(era)
        forfeited = forfeited_hotkeys(scoring, gate_block)
        if not era.king_hotkey or era.king_hotkey not in forfeited:
            return False
        succ = forfeit_successor(scoring, gate_block)
        if not succ:
            log.warning("rolling: era %d king %s is FORFEITED from block %s and no successor "
                        "is named — no king leg; waiting for the validators' receipts",
                        era.index, era.king_hotkey[:12], scoring.forfeit_from_block)
            return False
        with self._lock:
            log.warning("rolling: era %d king %s is FORFEITED from block %s — throne passes to "
                        "the named successor %s; its king leg trains for this era",
                        era.index, era.king_hotkey[:12], scoring.forfeit_from_block, succ[:12])
            era.king_hotkey, era.king_uid, era.king_ref = succ, -1, ""
            era.king_entry, era.king_bench, era.king_bench_published = None, None, False
            era.king_init = ""
            self._save()
        return True

    def _launch_king_leg(self, client, era: EraState, block: int, now: float) -> None:
        from ..shared.era import forfeited_hotkeys

        self._forfeit_switch(era)
        if era.king_hotkey and era.king_hotkey in forfeited_hotkeys(
                self.r.cfg.scoring, self._forfeit_gate_block(era)):
            return                                   # forfeited, no successor: vacant
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
            if era is self.state.current:
                self._publish_champion_now(era)   # a crowned king with no leg here yet
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
                if era.king_hotkey != gen.hotkey:
                    # The throne moved while this leg trained: it is not the
                    # era king's leg any more — discard, never adopt or bench it.
                    log.warning("rolling: era %d king leg for %s landed after the throne "
                                "passed to %s — discarded (%s)", era.index, gen.hotkey[:12],
                                (era.king_hotkey or "<vacant>")[:12], entry.trained_pointer)
                    return
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

    def _release_idle_king_pods(self) -> None:
        """Release an era's operator king pod the moment it has nothing to do.

        The pod used to live for the whole era (about 15 h at about $1.30/h on an
        H100) only to re-bench payer checkpoints that mostly never came
        (2026-10-03: idle 11.5 h after its king leg and bench). Released when:
        the era's king leg is in, its bench thread is done, no verification
        is running or queued. A verification queued later waits ("wait" costs
        no attempt) for the next king pod — the next era's pod is rented for
        its own king leg anyway — and its numbers republish into their round's
        report then. Never under a running sweep (``retire_king_pod`` defers
        on the bench lock). Idempotent per process; a restart re-checks once."""
        with self._lock:
            if self._verify_thread is not None and self._verify_thread.is_alive():
                return
            if self.state.verify_queue:
                return
            eras = [e for e in (self.state.current, self.state.next) if e is not None]
            idle = [e for e in eras
                    if e.king_entry is not None
                    and e.index not in self._king_threads
                    and e.index not in self._idle_released]
            for e in idle:
                self._idle_released.add(e.index)
        for e in idle:
            log.info("rolling: era %d king pod released — king leg and bench done, nothing "
                     "to verify (a later verification waits for the next king pod)", e.index)
            try:
                self.ops.release_idle_king_pod(e)
            except Exception as ex:  # noqa: BLE001 — the rollover teardown still runs
                log.warning("rolling: early release of era %d king pod failed: %s", e.index, ex)

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
            # fit on what THIS ref measured last time (max with the SKU wall) and
            # leave room for the post-round bench before the boundary
            entry_wall = self._leg_walls().fit_wall(entry.ref, wall)
            adm = admit(self.cfg.round, block_now=block, now=now, wall_seconds=entry_wall,
                        margin_seconds=margin + self._bench_margin(), current_era=cur.index)
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
            if self._blocked_check(gen, block):
                continue
            if self._packed_source_check(gen, block):
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

    def _blocked_check(self, gen, block: int | None = None) -> bool:
        """``[round] blocked_hotkeys`` (DEC-CA-0048): refused at the door, never
        rents a pod. Terminal ``failed`` [blocked], no burn — the fee question is
        policy, not code. Returns True when ``gen`` was dropped."""
        rnd = self.r.cfg.round
        blocked = rnd.blocked_at(block) if hasattr(rnd, "blocked_at") else frozenset()
        if gen.hotkey not in blocked:
            return False
        q = self.ops.queue()
        if q is not None:
            q.fail(gen.hotkey, error="hotkey is on the operator's admission denylist "
                   "([round] blocked_hotkeys)", error_class="blocked", expect_ref=gen.ref)
        log.warning("rolling: %s refused — blocked_hotkeys", gen.hotkey[:12])
        return True

    def _packed_source_check(self, gen, block: int | None = None) -> bool:
        """``[static_guard] packed_sources = "reject"``: a generator that ships
        Python inside a string constant is refused at the door (entry failed
        [generator], fee burns like any other miner-fault rejection). Only at
        admission — the per-leg preflight keeps scanning, so a seated king's
        legs are untouched. Returns True when ``gen`` was dropped."""
        sg = self.r.cfg.static_guard
        mode = (sg.packed_sources_at(block) if hasattr(sg, "packed_sources_at")
                else getattr(sg, "packed_sources", "scan"))
        if mode != "reject":
            return False
        reg = self.ops.dedup_registry()
        if reg is None or not hasattr(reg, "packed_source_verdict"):
            return False
        try:
            res = reg.packed_source_verdict(gen.ref)
        except Exception as e:  # noqa: BLE001 — fail open like the dedup screen
            log.warning("rolling: packed-source check failed for %s (%s); admitted unscreened",
                        gen.hotkey[:12], e)
            return False
        if res is None or res.ok or not (res.reason or "").startswith("packed_source"):
            return False
        q = self.ops.queue()
        if q is not None:
            q.fail(gen.hotkey, error=f"generator ships Python inside a string constant "
                   f"({res.file or 'generator.py'}, {res.reason}); every module must be a "
                   ".py file in the repo", error_class="generator", expect_ref=gen.ref)
        self.ops.burn([gen])
        log.warning("rolling: %s dropped — packed source in %s (%s)", gen.hotkey[:12],
                    res.file or "generator.py", res.reason)
        return True

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
                    if tier == "private_copy"
                    else "packs, or is packed inside, an earlier submission's code as"
                    if tier.startswith("embedded_") else "duplicates")
            q.fail(gen.hotkey, error=f"generator {what} {matched} ({tier}); the "
                   "earliest commit keeps the entry", error_class="duplicate",
                   expect_ref=gen.ref)
        self.ops.burn([gen])
        log.warning("rolling: %s dropped — duplicates %s (%s)", gen.hotkey[:12],
                    matched[:12], tier)
        return matched

    def _leg_walls(self) -> LegWalls:
        return LegWalls(Path(self.r.work_root) / "leg_walls.json")

    def _bench_margin(self) -> float:
        return float(getattr(self.cfg.round, "funded_bench_margin_seconds", 0) or 0)

    # ── legs ─────────────────────────────────────────────────────────────────

    def _launch_leg(self, gen, era: EraState, block: int, target: int, label: str,
                    *, resumed: bool = False) -> None:
        if gen.hotkey in self._threads:
            return
        queue = self.ops.queue()
        end_wall = wall_of_block(target, now=self.clock(), block_now=block)
        t_launch = self.clock()

        def _finish(entry: TrainedEntry) -> None:
            self._leg_walls().record(gen.ref, self.clock() - t_launch,
                                     sku=str(getattr(entry, "gpu_name", "") or ""))
            with self._lock:
                rolled = self.state.current is not None and era.index < self.state.current.index
            if rolled:
                # Its era has no settlement left: the leg can never be judged
                # (a later era trains from a different init). Tell the miner NOW,
                # not at the next rollover, and never record an orphan.
                log.warning("rolling: %s's leg finished after era %d ended — requeued "
                            "unburned at finish (era now %d)", gen.hotkey[:12], era.index,
                            self.state.current.index)
                try:
                    self.ops.teardown_kept_pod(gen.hotkey)
                except Exception as e:  # noqa: BLE001
                    log.error("rolling: kept pod teardown for %s failed: %s", gen.hotkey[:12], e)
                if queue is not None:
                    queue.requeue(gen.hotkey, error="leg finished after its era ended",
                                  error_class="no_capacity", burn_attempt=False)
                return
            # Record the leg the moment TRAINING ends: the bench below is
            # telemetry (an hour and more on a payer pod) and must never hold
            # the settlement. Before 2026-10-02 the leg was filed only after its
            # bench, so a bench that outlived the era (09-28 22:24, 10-02 22:03
            # uid 156) stranded a trained leg and the miner paid for a second
            # one. Late numbers join the settlement's bench report instead.
            with self._lock:
                rolled = (self.state.current is not None
                          and era.index < self.state.current.index)
                if not rolled:
                    self.state.finished.append(FinishedLeg(
                        hotkey=gen.hotkey, uid=gen.uid, ref=gen.ref, era_index=era.index,
                        started_block=block, reveal_block=gen.reveal_block,
                        entry=_entry_to_json(entry), bench=None, label=label))
                    self._save()
            if rolled:
                log.warning("rolling: %s's leg finished after era %d ended — requeued "
                            "unburned at finish (era now %d)", gen.hotkey[:12], era.index,
                            self.state.current.index)
                try:
                    self.ops.teardown_kept_pod(gen.hotkey)
                except Exception as e:  # noqa: BLE001
                    log.error("rolling: kept pod teardown for %s failed: %s", gen.hotkey[:12], e)
                if queue is not None:
                    queue.requeue(gen.hotkey, error="leg finished after its era ended",
                                  error_class="no_capacity", burn_attempt=False)
                return
            log.info("rolling: %s's leg finished (era %d, target boundary %d) — bench runs "
                     "alongside; the settlement does not wait for it",
                     gen.hotkey[:12], era.index, target)
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
            if bench is None:
                log.info("rolling: %s's bench produced no numbers", gen.hotkey[:12])
                return
            self._attach_bench(entry, bench)

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
        from ..shared.era import forfeited_hotkeys

        if king in forfeited_hotkeys(self.r.cfg.scoring, self._forfeit_gate_block(cur)):
            # DEC-CA-0048: the last scored receipt predates the crowning and
            # the on-chain incentive lags it a tempo, so both can still name
            # the FORFEITED king. That is not a dethrone — the validators
            # crowned the named successor at the gate — so the era's throne
            # never goes back (2026-09-28 13:08: era 2546 flipped to the
            # forfeited king for one tick and its king leg "restarted"; had
            # the successor's leg already landed it would have been thrown
            # away). ``_forfeit_switch`` keeps the era on the successor.
            log.debug("rolling: receipts/incentive still name forfeited %s king — stale, "
                      "ignored (era %d stays with %s)", king[:12], cur.index,
                      (cur.king_hotkey or "<vacant>")[:12])
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
        self._publish_champion_now(cur)

    def _publish_champion_now(self, era: EraState) -> None:
        """Run the champion-publication policy for ``era``'s king as soon as
        its ref is known — at dethrone adoption (or once the new king's ref
        resolves) — instead of waiting for the next settlement that carries a
        manifest, which kept a crowned king's code private for hours. The
        publisher is idempotent per (king, round), so the settlement's own
        call afterwards is a no-op. Best-effort: never blocks the loop."""
        if not era.king_hotkey or not era.king_ref:
            return
        try:
            self.ops.publish_champion(ResolvedGen(era.king_hotkey, era.king_uid, era.king_ref),
                                      str(self.state.last_published_round_id or ""))
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: champion publish at dethrone failed (ignored; the next "
                        "settlement retries): %s", e)

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
                                                            else int(block)),
                                  now_block=block)

    def _late_under_forfeited_king(self, cur: EraState, epoch_start: int,
                                   now_block: int | None) -> bool:
        """DEC-CA-0048 on the block clock: a boundary of an era whose king is
        forfeited by ``now_block`` but NOT at the era's own gate (a late
        settlement of an era that ended before the forfeiture) can never be
        judged — the validators crowned the successor at the gate and reject
        a manifest carrying the old king whole (``king_resyncing``), after the
        trainer marked its legs done and burned (2026-09-28 13:06: the late
        settlement of 9165600, two paid legs lost). Publish nothing; requeue
        every finished leg unburned so it retrains against the successor.
        Returns True when the boundary was closed this way."""
        from ..shared.era import forfeited_hotkeys

        scoring = self.r.cfg.scoring
        king_hk = cur.king_hotkey
        if (now_block is None or not king_hk
                or king_hk not in forfeited_hotkeys(scoring, int(now_block))
                or king_hk in forfeited_hotkeys(scoring, self._forfeit_gate_block(cur))):
            return False                      # not forfeited, or the era switches itself
        with self._lock:
            ready = [f for f in self.state.finished if f.era_index == cur.index]
        queue = self.ops.queue()
        log.warning("rolling: boundary %d — era %d king %s is forfeited at block %d, after "
                    "the era's last settlement: the validators crowned the successor and "
                    "would reject this manifest whole; nothing published, %d finished "
                    "leg(s) requeued unburned", epoch_start, cur.index, king_hk[:12],
                    int(now_block), len(ready))
        for f in ready:
            self.ops.discard_cached_leg(cur, "challenger",
                                        ResolvedGen(f.hotkey, f.uid, f.ref, f.reveal_block))
            if queue is not None:
                queue.requeue(f.hotkey, error=f"era {cur.index} king forfeited before its "
                              f"settlement at {epoch_start} could be judged — retrained "
                              f"against the successor", error_class="no_capacity",
                              burn_attempt=False)
        with self._lock:
            self.state.finished = [f for f in self.state.finished if f not in ready]
            self.state.last_settled_boundary = epoch_start
            self._save()
        return True

    def _settle_boundary(self, client, epoch_start: int, *, created_block: int,
                         now_block: int | None = None) -> None:
        cur = self.state.current
        if cur is None or epoch_start <= self.state.last_settled_boundary:
            return
        if self._late_under_forfeited_king(cur, epoch_start, now_block):
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
        self._drop_foreign_king_entry(cur)
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
        self._record_bench_report(round_id, created_block, bench_entries, ready,
                                  str(getattr(manifest, "warm_start_ckpt", "") or ""))
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
            contract_digest=contract_digest(self.cfg.training.at_block(int(era.start_block))),
            base_arch_digest=self.cfg.training.base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset, entries=entries,
            eval_pool_key=str(pool_key or ""), eval_pool_sha256=str(pool_sha or ""),
            warm_start_ckpt=era.warm_start_ckpt, warm_start_size=era.warm_start_size,
            contract_body=contract_payload(self.cfg.training.at_block(int(era.start_block))),
            era=era.spec().to_json(),
            prev_round_id=self.state.last_published_round_id,
        )

    # ── deferred bench verification ──────────────────────────────────────────

    def _queue_verify(self, entry: TrainedEntry, payer: dict) -> None:
        """Park ``entry``'s payer-pod numbers until the operator can verify
        them. Caller holds the lock."""
        if any(q["pointer"] == entry.trained_pointer for q in self.state.verify_queue):
            return
        reason = str(payer.get("_reason", "") or "")
        payer = {k: v for k, v in payer.items() if not str(k).startswith("_")}
        self.state.verify_queue.append({
            "pointer": entry.trained_pointer, "hotkey": entry.miner_hotkey,
            "entry": _entry_to_json(entry), "payer": payer, "round_id": "", "tries": 0,
        })
        log.info("rolling: %s's payer-pod bench queued for operator verification%s",
                 entry.miner_hotkey[:12], f" ({reason})" if reason else "")

    def _record_bench_report(self, round_id: str, created_block: int, bench_entries: list,
                             ready: list[FinishedLeg], warm_start_ckpt: str = "") -> None:
        """Remember what this settlement's bench report carries, and stamp the
        settled legs' queued verifications with it: a late-verified number is
        added to THIS report (validators join on (round_id, pointer))."""
        import dataclasses as _dc

        pointers = {_entry_from_json(f.entry).trained_pointer for f in ready}
        with self._lock:
            self.state.bench_reports[str(round_id)] = {
                "created_block": int(created_block),
                "pairs": [[_entry_to_json(e), _dc.asdict(b)] for e, b in bench_entries],
                # Every leg this settlement judged (benched or not) and the init it
                # trained from: a bench that lands AFTER the settlement finds its
                # report here, and the republish can feed promotion candidates.
                "pointers": sorted(pointers),
                "warm_start_ckpt": str(warm_start_ckpt or ""),
            }
            by_age = sorted(self.state.bench_reports,
                            key=lambda r: int(self.state.bench_reports[r].get("created_block", 0)))
            for old in by_age[:-KEEP_BENCH_REPORTS]:      # oldest first (state JSON is key-sorted)
                self.state.bench_reports.pop(old, None)
            for q in self.state.verify_queue:
                if q["pointer"] in pointers and not q.get("round_id"):
                    q["round_id"] = str(round_id)
            self._save()

    def _drain_verifies(self) -> None:
        """Run the next queued verification — one at a time, on the current
        era's king pod — once that era's king leg exists. Queue hygiene first:
        an unsettled entry whose leg was requeued (era ended) has no report to
        join and is dropped; so is one whose round report aged out."""
        if self._verify_thread is not None and self._verify_thread.is_alive():
            return
        with self._lock:
            finished = {_entry_from_json(f.entry).trained_pointer for f in self.state.finished}
            keep = []
            for q in self.state.verify_queue:
                if not q.get("round_id") and q["pointer"] not in finished:
                    log.warning("rolling: queued bench verification for %s dropped — its leg "
                                "never settled (requeued)", q["hotkey"][:12])
                elif q.get("round_id") and q["round_id"] not in self.state.bench_reports:
                    log.warning("rolling: queued bench verification for %s dropped — round "
                                "%s's report aged out", q["hotkey"][:12], q["round_id"])
                else:
                    keep.append(q)
            if len(keep) != len(self.state.verify_queue):
                self.state.verify_queue = keep
                self._save()
            cur = self.state.current
            king = cur.king() if cur is not None else None
            if not keep or king is None:
                return
            item = dict(keep[0])

        def _run() -> None:
            entry = _entry_from_json(item["entry"])
            try:
                status, scores = self.ops.verify_bench(entry, dict(item["payer"]), king, cur)
            except Exception as e:  # noqa: BLE001 — counts as a failed attempt
                status, scores = "failed", {"error": str(e)[:200]}
            self._apply_verify(item["pointer"], status, scores)

        t = threading.Thread(target=_run, name=f"verify-{item['hotkey'][:12]}", daemon=True)
        self._verify_thread = t
        t.start()

    def _apply_verify(self, pointer: str, status: str, scores: dict | None) -> None:
        republish = None
        with self._lock:
            q = next((x for x in self.state.verify_queue if x["pointer"] == pointer), None)
            if q is None:
                return
            if status == "wait":
                return                                    # king pod not usable yet
            if status == "failed":
                q["tries"] = int(q.get("tries", 0)) + 1
                if q["tries"] < MAX_VERIFY_TRIES:
                    log.warning("rolling: bench verification for %s failed (attempt %d/%d): %s",
                                q["hotkey"][:12], q["tries"], MAX_VERIFY_TRIES,
                                (scores or {}).get("error", ""))
                    self._save()
                    return
                log.warning("rolling: bench verification for %s failed %d times — dropped",
                            q["hotkey"][:12], q["tries"])
            elif status == "forged":
                log.error("rolling: %s's payer-pod bench did not reproduce on the operator's "
                          "pod (TAMPER) — numbers dropped", q["hotkey"][:12])
            self.state.verify_queue = [x for x in self.state.verify_queue if x is not q]
            if status == "ok" and scores is not None:
                republish = self._join_bench_locked(pointer, q["entry"], dict(scores),
                                                    q.get("round_id") or "")
                if republish == "leg":
                    log.info("rolling: %s's bench verified — joins its settlement report",
                             q["hotkey"][:12])
                    republish = None
            self._save()
        if republish is not None:
            self._republish(pointer, *republish)

    # ── bench that lands after its leg was recorded ─────────────────────────

    def _round_of_pointer(self, pointer: str) -> str:
        """Settlement round whose report covers ``pointer`` ("" = none kept).
        Caller holds the lock."""
        for rid, rep in self.state.bench_reports.items():
            if pointer in rep.get("pointers", ()) or any(
                    p[0].get("trained_pointer") == pointer for p in rep.get("pairs", ())):
                return str(rid)
        return ""

    def _join_bench_locked(self, pointer: str, entry_json: dict, scores: dict,
                           round_id: str):
        """Attach ``scores`` to the leg they belong to. Returns "leg" when the
        leg is still unsettled (numbers ride its own settlement's report), a
        ``(round_id, created_block, pairs, warm_start_ckpt)`` republish tuple
        when the leg already settled, or None when no report can take them.
        Caller holds the lock."""
        leg = next((f for f in self.state.finished
                    if _entry_from_json(f.entry).trained_pointer == pointer), None)
        if leg is not None:
            leg.bench = dict(scores)
            return "leg"
        rid = round_id or self._round_of_pointer(pointer)
        rep = self.state.bench_reports.get(rid) if rid else None
        if rep is None:
            return None
        rep["pairs"] = [p for p in rep["pairs"] if p[0].get("trained_pointer") != pointer]
        rep["pairs"].append([entry_json, dict(scores)])
        return (rid, int(rep["created_block"]), [list(p) for p in rep["pairs"]],
                str(rep.get("warm_start_ckpt", "") or ""))

    def _republish(self, pointer: str, rid: str, cblock: int, pairs: list,
                   warm_start_ckpt: str) -> None:
        from types import SimpleNamespace

        from ..shared.manifest import BenchScores

        typed = [(_entry_from_json(e), BenchScores(**b)) for e, b in pairs]
        try:
            report = self.ops.publish_bench(rid, cblock, typed)
            log.info("rolling: round %s bench report republished with %s's numbers "
                     "(%d entries)", rid, pointer[-12:], len(typed))
        except Exception as e:  # noqa: BLE001 — telemetry; the numbers stay recorded
            log.warning("rolling: republishing round %s's bench report failed: %s", rid, e)
            return
        # A late bench is still a promotion candidate (record_bench is idempotent
        # by pointer and reads only round_id + warm_start_ckpt from the manifest).
        self.ops.record_bench_candidates(
            SimpleNamespace(round_id=rid, warm_start_ckpt=warm_start_ckpt), report)

    def _attach_bench(self, entry: TrainedEntry, bench: dict) -> None:
        """Route a challenger's bench result, whenever it lands: verification
        pending → the verify queue (stamped with the settled round, if any);
        numbers → the unsettled leg, or a republish of the settled round's
        report; neither available → logged and dropped (telemetry only)."""
        pointer = entry.trained_pointer
        republish = None
        with self._lock:
            pending = bench.get(VERIFY_PENDING) if isinstance(bench, dict) else None
            if pending is not None:
                # Payer numbers the operator could not verify yet: queue them —
                # the numbers land once verified, never unverified.
                self._queue_verify(entry, pending)
                rid = self._round_of_pointer(pointer)
                for q in self.state.verify_queue:
                    if q["pointer"] == pointer and rid and not q.get("round_id"):
                        q["round_id"] = rid
                self._save()
                return
            republish = self._join_bench_locked(pointer, _entry_to_json(entry), dict(bench), "")
            self._save()
        if republish == "leg":
            log.info("rolling: %s's bench landed before its settlement — rides its report",
                     entry.miner_hotkey[:12])
        elif republish is None:
            log.warning("rolling: %s's bench landed but no settlement report can take it "
                        "(report aged out) — numbers dropped", entry.miner_hotkey[:12])
        else:
            self._republish(pointer, *republish)

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
