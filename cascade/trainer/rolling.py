"""Rolling intake + era king (DEC-CA-0043) — the trainer's scheduler.

Once the rollover is active (``[round] rolling_from_block``, typed in or
decided on chain via DEC-CA-0045) this is how the trainer works:

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
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..shared.atomic import read_text_with_backup, write_text_atomic
from ..shared.config import BLOCK_SECONDS
from ..shared.era import (
    EraSpec,
    bench_gate_configured,
    bench_gate_mode_at,
    era_for_block,
    era_length_blocks,
    member_index_for_era,
    min_effective_era,
    next_era_start,
    settlement_era,
    v2_era_active,
    v2_era_seed_block,
    v2_first_warm_era,
    v2_generation_usable,
    v2_seed_epoch_start,
    validator_bench_at,
    validator_bench_era,
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

# A booted king pod gets ONE retry after a non-transport failure (a Hippius
# read stall is not necessarily the pod's fault); the next failure — or any
# transport failure at once — rotates it: host quarantined, pod torn down,
# the next tick re-rents elsewhere, instead of looping into the same bad
# network path every tick.
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
    # DEC-CA-0050: a lane-B merged init — the promotion member it was built
    # from and the merge records (declared in the manifest; validators rebuild)
    warm_start_base: str = ""
    warm_start_merges: list = field(default_factory=list)
    king_hotkey: str = ""
    king_uid: int = -1
    king_ref: str = ""
    king_entry: dict | None = None    # TrainedEntry (asdict), role "king"
    king_init: str = ""               # the init king_entry trained from (must == warm_start_ckpt)
    king_bench: dict | None = None    # BenchScores (asdict)
    king_bench_published: bool = False
    king_leg_failed: str = ""         # last king-leg failure (retry pending)
    king_leg_failures: int = 0        # consecutive failures on the CURRENT king pod
    king_leg_failures_total: int = 0  # since last success, across pod rotations
    # DEC-CA-0055 seed epochs (v2 only; -1 = the era is its own epoch, every
    # pre-v2 era): the first era of the warm-start epoch this era belongs to.
    # Every era of an epoch shares ``seed_block``/``base_seed`` and the init,
    # so the king leg trains once per epoch and is carried across its eras.
    epoch_start: int = -1
    # DEC-CA-0055 item 6: who paid for the king entry's leg — "king" (the
    # king's own payer-vault key), "operator" (the fallback), "" (pre-v2 /
    # unknown). Stamped on the manifest's king entry (``funded_by``).
    king_funded_by: str = ""
    # DEC-CA-0055: the CURRENT init's BenchScores (the "init" bench-report
    # entry), benched once per seed epoch; tries cap the retry loop.
    init_bench: dict | None = None
    init_bench_tries: int = 0
    # the PUBLISHED random-init checkpoint the init bench scored ("" when the
    # epoch is warm-started: the init is ``warm_start_ckpt`` itself)
    init_pointer: str = ""

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
    table is a per-GPU estimate, and a CPU-bound generator can run far past
    it (landing after its era ended, and being lost). Admission
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
            write_text_atomic(self.path, json.dumps(d, sort_keys=True, indent=1))
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
    # sha256 of the exact text published for ``last_published_round_id``
    # (``dump_manifest``, UTF-8) — the next manifest's ``prev_manifest_sha256``
    # ("" = unknown: the field is omitted, validators leave it unchecked)
    last_published_manifest_sha256: str = ""
    # Write-ahead of a settlement publish: {boundary, round_id, created_block,
    # manifest (the signed dump_manifest text), pointers (the settled legs),
    # bench_first, bench ({pairs, rows_for} | None)}. Persisted BEFORE the
    # first byte goes out and cleared in the same save that commits the
    # boundary, so a crash in between republishes EXACTLY this manifest (and
    # report) under its round id instead of re-deriving a different one.
    pending_publish: dict | None = None
    # DEC-CA-0063 arbiter: validator-sourced settlements whose main-validator
    # bench report the trainer has not recorded yet — {round_id: later
    # settlements built without it}. Each later manifest records a round
    # once (``validator_bench_reports``: its sha256, or "absent" at ``[v2]
    # bench_report_timeout_settlements``) and drops it here.
    vbench_pending: dict[str, int] = field(default_factory=dict)
    # DEC-CA-0063 item 9: how many times each settled leg (trained pointer)
    # was carried into a later settlement because the main validator's sweep
    # cap BUDGET-DEFERRED it (bounded by VBENCH_MAX_CARRIES, inside its era).
    vbench_carries: dict[str, int] = field(default_factory=dict)
    # DEC-CA-0055: {king hotkey: sha256 of the payer-vault key its king leg
    # failed on ("" = the vault held none)} — only a ROTATED key lifts the
    # slash, across trainer restarts too (dropped from the file when empty).
    king_failed_keys: dict[str, str] = field(default_factory=dict)


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
        funded_by=str(obj.get("funded_by", "") or ""),
        init_pointer=obj.get("init_pointer"),
        training_contract_digest=str(obj.get("training_contract_digest", "") or ""),
    )


def _entry_to_json(entry: TrainedEntry) -> dict:
    return dataclasses.asdict(entry)


def _with_funded_by(entry: TrainedEntry, funded_by: str) -> TrainedEntry:
    """``entry`` with ``funded_by`` (DEC-CA-0055 manifest field) set — a
    no-op on a build whose TrainedEntry has no such field yet."""
    if not any(f.name == "funded_by" for f in dataclasses.fields(entry)):
        return entry
    if str(getattr(entry, "funded_by", "") or "") == str(funded_by or ""):
        return entry
    return replace(entry, funded_by=str(funded_by or ""))


def _known_fields(cls, obj: dict, what: str):
    """``cls(**obj)`` over the fields ``cls`` knows: a key an older/newer
    build wrote is dropped (logged), never a TypeError that keeps the
    scheduler from building."""
    known = {f.name for f in dataclasses.fields(cls)}
    extra = sorted(k for k in obj if k not in known)
    if extra:
        log.warning("rolling: era state %s carries unknown key(s) %s — dropped", what, extra)
    return cls(**{k: v for k, v in obj.items() if k in known})


def load_state(path: Path) -> RollingState:
    raw, src = read_text_with_backup(path, json.loads)
    if raw is None:
        if src == "unreadable":     # a torn file (and no good backup) must not sink the trainer
            log.error("rolling: era state %s and its backup unreadable; starting fresh", path)
        return RollingState()
    if src == "backup":
        log.warning("rolling: era state %s unreadable — restored from its .bak", path)

    def _era(obj):
        return _known_fields(EraState, obj, "era") if obj else None

    pending = raw.get("pending_publish")
    return RollingState(
        current=_era(raw.get("current")),
        next=_era(raw.get("next")),
        generations={str(k): dict(v) for k, v in (raw.get("generations") or {}).items()},
        finished=[_known_fields(FinishedLeg, x, "finished leg") for x in raw.get("finished", [])],
        published=[_known_fields(FinishedLeg, x, "published leg")
                   for x in raw.get("published", [])],
        last_published_round_id=str(raw.get("last_published_round_id", "") or ""),
        last_settled_boundary=int(raw.get("last_settled_boundary", 0) or 0),
        rents=list(raw.get("rents", [])),
        verify_queue=[dict(x) for x in raw.get("verify_queue", [])],
        bench_reports={str(k): dict(v) for k, v in (raw.get("bench_reports") or {}).items()},
        last_published_manifest_sha256=str(raw.get("last_published_manifest_sha256", "") or ""),
        pending_publish=dict(pending) if isinstance(pending, dict) else None,
        vbench_pending={str(k): int(v) for k, v in (raw.get("vbench_pending") or {}).items()},
        vbench_carries={str(k): int(v) for k, v in (raw.get("vbench_carries") or {}).items()},
        king_failed_keys={str(k): str(v or "")
                          for k, v in (raw.get("king_failed_keys") or {}).items()},
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
        "last_published_manifest_sha256": state.last_published_manifest_sha256,
        "pending_publish": state.pending_publish,
    }
    if state.vbench_pending:
        body["vbench_pending"] = state.vbench_pending
    if state.vbench_carries:
        body["vbench_carries"] = state.vbench_carries
    if state.king_failed_keys:
        body["king_failed_keys"] = state.king_failed_keys
    write_text_atomic(path, json.dumps(body, sort_keys=True, indent=2), backup=True)


# ── pure policy ──────────────────────────────────────────────────────────────


def king_pod_should_rotate(exc: BaseException, failures: int) -> bool:
    """Rotate the era king's pod after ``failures`` consecutive king-leg
    failures on it: at once when the failure is a transport one (the pod is
    unreachable), else once the same-pod retry budget is spent."""
    from ..funding.faults import classify_rent_failure
    from .loop import _transport_failure

    if classify_rent_failure(str(exc)) in {"auth", "rate_limited"}:
        return False
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


def resolve_generation(generations: dict[str, dict], era_index: int,
                       usable=None) -> tuple[int, list]:
    """``(generation, members)`` for era ``era_index``: the latest generation
    whose ``effective_era <= era_index``; ``(0, [])`` = random init.
    ``usable(effective_era)`` (DEC-CA-0050 v2 lineage reset,
    :func:`cascade.shared.era.v2_generation_usable`) turns a generation that
    may not initialise this era into random init — the same rule every
    validator's era gate applies."""
    best_gen, best_members, best_eff = 0, [], 0
    for g, rec in generations.items():
        gi = int(g)
        if int(rec.get("effective_era", 0)) <= int(era_index) and gi > best_gen:
            best_gen, best_members = gi, list(rec.get("members") or [])
            best_eff = int(rec.get("effective_era", 0))
    if best_gen and usable is not None and not usable(best_eff):
        return 0, []
    return best_gen, best_members


def era_init(generations: dict[str, dict], era_index: int,
             usable=None) -> tuple[int, int, str, str]:
    """``(generation, member_index, checkpoint, size)`` for the era."""
    gen, members = resolve_generation(generations, era_index, usable)
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
    def chain_read(self, fn):
        """``fn()`` under the runner's hard chain-read deadline
        (``_with_deadline`` / ``chain_read_timeout_s``, the one the block
        read uses): a hung websocket raises ``TimeoutError`` and the tick
        skips that step instead of wedging. No runner helper ⇒ plain call."""
        r = getattr(self, "r", None)
        with_deadline = getattr(r, "_with_deadline", None)
        if not callable(with_deadline):
            return fn()
        return with_deadline(fn, float(getattr(r, "chain_read_timeout_s", 60.0) or 60.0))

    def block_seed(self, client, block: int) -> int:
        return int(self.chain_read(lambda: client.block_seed(int(block))))

    def commitments(self, client) -> list:
        from .admission import remember_owners

        return remember_owners(self.r, self.chain_read(lambda: client.poll_commitments(include_history=True)))

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
            cfg = getattr(self.r, "cfg", None)
            burn = (int(cfg.scoring.burn_uid),) if cfg is not None else ()
            return self.chain_read(lambda: client.highest_incentive_hotkey(exclude_uids=burn))
        except Exception:  # noqa: BLE001 — cross-check only (a timeout too)
            return None

    # legs
    def train_challenger(self, gen, era: EraState, block: int, *, end_wall: float) -> TrainedEntry:
        from .admission import owner_admission
        from .progress import progress_scope, report_progress

        with owner_admission(self.r, gen.hotkey, deadline=end_wall - self._era_contract(era).max_train_seconds), progress_scope(self.r, gen):
            entry = self.r._rolling_train_challenger(gen, era, block, end_wall=end_wall)
            report_progress(self.r, "awaiting_settlement")
            return entry

    def train_king(self, gen, era: EraState, block: int, *, end_wall: float) -> TrainedEntry:
        from .runner_funded_legs import funded_leg_scope

        with funded_leg_scope(self.r, None, self._era_contract(era), king_leg=True):
            return self.r._rolling_train_king(gen, era, block, end_wall=end_wall)

    def _era_contract(self, era: EraState):
        """The contract a leg of ``era`` trains and is persisted under — the
        one effective at the era's start (DEC-CA-0047 scheduled switch), the
        same ``_rolling_train_*`` persist with. Looking a record up under the
        base contract would miss every leg trained after a switch."""
        return self.r.cfg.throne_contracts_at(int(era.start_block))[0]

    def cached_leg(self, era: EraState, role: str, gen) -> TrainedEntry | None:
        contract = self._era_contract(era)
        suffix = "" if role == "king" else f"-u{gen.uid}"
        # The era's init is part of the job: a record from the same seed but
        # another init (another member, an edited state) is never reused.
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
        # No recorded verdict: per-leg verdicts are cleared at every leg
        # start, so a path that fails without recording one (the operator-
        # lane fallback, a dispatcher/config error) is OURS, never the
        # miner's — operator class: no burned attempt, the queue backs off.
        from ..funding.queue import OPERATOR_FAULT_CLASS

        return self.r._funded_leg_failures.get(
            hotkey, ("challenger leg failed without a recorded verdict (operator side)",
                     False, OPERATOR_FAULT_CLASS, False))

    def king_funding(self, era: EraState, gen) -> str:
        """DEC-CA-0055: who paid for ``era``'s persisted king leg ("king" /
        "operator" / "" unknown), from its completed-leg record."""
        fn = getattr(getattr(self, "r", None), "_rolling_king_funding", None)
        return str(fn(era, gen) or "") if callable(fn) else ""

    def bench_init(self, era: EraState) -> dict | None:
        return self.r._rolling_bench_init(era)

    def carry_king_pod(self, old: EraState, new: EraState) -> None:
        """Hand ``old``'s operator king pod to ``new`` (same seed epoch: same
        pod name prefix) instead of tearing it down."""
        fn = getattr(getattr(self, "r", None), "_rolling_carry_king_pod", None)
        if callable(fn):
            fn(old, new)

    def king_key_ok(self, hotkey: str) -> bool:
        """DEC-CA-0055: a slashed king's rotated key now works."""
        fn = getattr(getattr(self, "r", None), "_rolling_king_key_ok", None)
        try:
            return bool(fn(hotkey)) if callable(fn) else False
        except Exception:  # noqa: BLE001 — a probe never sinks a roll
            return False

    def refresh_king_key(self, hotkey: str) -> bool:
        """Re-stamp the king's payer-vault TTL (DEC-CA-0055 item 6)."""
        fn = getattr(getattr(self, "r", None), "_rolling_refresh_king_key", None)
        return bool(fn(hotkey)) if callable(fn) else False

    def teardown_kept_pod(self, hotkey: str) -> None:
        self.r._teardown_kept_funded_pod(hotkey)

    def retire_king_pod(self, era: EraState) -> None:
        self.r._rolling_retire_king_pod(era)

    def king_pod_can_bench(self, era: EraState) -> bool:
        return self.r._rolling_king_pod_can_bench(era)

    def release_idle_king_pod(self, era: EraState) -> None:
        """Same teardown as :meth:`retire_king_pod` (bench-lock aware), kept
        separate so an idle release is never read as a throne/king change."""
        self.r._rolling_retire_king_pod(era)

    def rotate_king_pod(self, era: EraState, reason: str) -> None:
        self.r._rolling_rotate_king_pod(era, reason)

    # publication
    def sign_manifest(self, manifest):
        """The manifest signed with the trainer hotkey (unchanged without a
        wallet). The scheduler signs ONCE and persists the signed text before
        publishing, so a retried publish sends the same bytes (sr25519
        signatures are randomised: re-signing would not)."""
        wallet = getattr(getattr(self, "r", None), "wallet", None)
        if wallet is None or manifest.signature is not None:
            return manifest
        from ..shared.manifest import sign_manifest

        return sign_manifest(manifest, wallet)

    def publish_manifest(self, manifest) -> None:
        """Publish ``manifest`` byte-exactly as ``dump_manifest`` renders it —
        an already-signed manifest is never re-signed."""
        if manifest.signature is None:
            self.r.publish(manifest)          # no wallet: warns, publishes the same dump
            return
        from ..shared.hippius import publish_manifest
        from ..shared.manifest import dump_manifest

        key = publish_manifest(self.r.manifest_store(), dump_manifest(manifest),
                               manifest.round_id)
        log.info("published manifest round=%s entries=%d signed=True → %s",
                 manifest.round_id, len(manifest.entries), key)
        self.r._publish_training_summary(manifest)

    def publish_bench(self, round_id: str, created_block: int, entries: list, *,
                      rows_for: frozenset | None = None,
                      init: dict | None = None) -> object | None:
        kw = {"init": init} if init else {}
        return self.r._rolling_publish_bench(round_id, created_block, entries,
                                             rows_for=rows_for, **kw)

    def read_text(self, key: str) -> str:
        """The text at ``key`` in the manifest bucket (``ObjectNotFound`` when
        absent) — the main validator's bench report the next manifest records
        (DEC-CA-0063)."""
        return self.r.manifest_store().get_text(key)

    def publish_text(self, key: str, text: str) -> None:
        """Put ``text`` at ``key`` in the manifest bucket (a lane-C manifest
        republished ahead of the lane-A manifest that binds it)."""
        self.r.manifest_store().put_text(key, text, content_type="application/json")

    def publish_roster(self, round_id: str, roster: dict) -> None:
        self.r._rolling_publish_roster(round_id, roster)

    def promotion_boundary(self, king_hotkey: str, epoch_start: int, round_id: str,
                           effective_era: int) -> None:
        self.r._promotion_boundary_step(king_hotkey, epoch_start, round_id,
                                        effective_era=effective_era)

    def promotion_generations(self) -> tuple[int, list, int]:
        """``(generation, members[(ckpt, size)], effective_era)`` the engine
        currently holds (effective_era 0 = unknown/immediate).

        In the v2 lineage the engine's generation enters the era ledger only
        once its published record's notice is bound into the settlement chain
        in time — judged by the validators' own
        :meth:`~cascade.validator.king_check.KingCheckMixin._v2_notice_reason`
        at ``latest.json``. A record whose notice is FINALLY late
        (:data:`~cascade.validator.king_check.NOTICE_LATE`) is never
        effective: ``(0, [], 0)`` for good (cached), the ledger keeps the
        previous generation as the init — the exact state every validator
        holds after skipping that generation — and the next king check fires
        the following generation on the same rule. Not announced YET, or
        unreadable evidence, holds (``StorageError``) without a verdict."""
        p = self.r.promotion
        if p is None:
            return 0, [], 0
        generation = int(getattr(p, "generation", 0) or 0)
        members = [(m.checkpoint_id, m.size) for m in getattr(p, "members", ())]
        if not generation or not members:
            return 0, [], 0
        pending = getattr(p, "pending_record", None)
        eff = int(getattr(pending, "effective_era", 0) or 0) if pending is not None else 0
        warm = v2_first_warm_era(self.r.cfg)
        if warm is not None:
            # Fire-time state is not permission to train: a failed/delayed
            # publication may never have entered the settlement chain.
            from types import SimpleNamespace

            from ..shared.hippius import StorageError, manifest_round_key
            from ..shared.manifest import load_manifest
            from ..shared.promotion import (
                load_promotion_record,
                promotion_record_key,
                verify_promotion_record_signature,
            )
            from ..validator.king_check import NOTICE_LATE, KingCheckMixin

            try:
                if pending is not None:
                    return 0, [], 0
                cache_key = (id(self.r.cfg), generation, tuple(members))
                cached = getattr(self, "_promotion_notice_cache", None)
                if cached is not None and cached[0] == cache_key:
                    if cached[1] is None:
                        return 0, [], 0          # judged never effective (NOTICE_LATE)
                    return generation, members, cached[1]
                store = self.r.manifest_store()
                record = load_promotion_record(store.get_text(promotion_record_key(generation)))
                eff = int(record.effective_era)
                if eff >= warm:
                    if (record.generation != generation
                            or [(m.checkpoint_id, m.size) for m in record.members] != members
                            or not verify_promotion_record_signature(record, self.r.cfg.manifest.trainer_hotkey)):
                        return 0, [], 0

                    def load(rid):
                        return load_manifest(store.get_text(manifest_round_key(rid)))

                    probe = SimpleNamespace(cfg=self.r.cfg, verify_signatures=True,
                        _promotion_check_manifest=load(self.latest_round_id()),
                        _lane_b_load_settlement=load)
                    reason = KingCheckMixin._v2_notice_reason(probe, record)
                    if reason is not None:
                        log.error("rolling: promotion %d not installed: %s", generation, reason)
                        if reason == NOTICE_LATE:
                            # final for every node: the generation is skipped,
                            # the previous init stays — the next king check
                            # fires the following generation
                            self._promotion_notice_cache = (cache_key, None)
                        return 0, [], 0
                    self._promotion_notice_cache = (cache_key, eff)
            except Exception as e:  # noqa: BLE001 — keep the last verified generation
                log.warning("rolling: promotion publication/notice unavailable: %s", e)
                raise StorageError(f"promotion publication/notice unavailable: {e}") from e
        return generation, members, eff

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

    def burned(self, block: int) -> set[str]:
        """Hotkeys that already used this round's one submission (a round is
        one settlement step of the grid in force)."""
        from ..shared.config import effective_epoch_blocks
        from .runner_common import _load_burns

        rnd = self.r.cfg.round
        if not rnd.one_submission_per_hotkey:
            return set()

        def round_start(b: int) -> int:
            eb = max(1, int(effective_epoch_blocks(rnd, int(b))))
            return (int(b) // eb) * eb

        cur = round_start(block)
        return {h for h, b in _load_burns(self.r._submissions_path()).items()
                if b >= 0 and round_start(b) == cur}

    def burn(self, gens: list, block: int) -> None:
        self.r._burn_hotkeys(gens, block)

    def vault_owned(self, gens: list) -> list:
        return self.r._verify_vault_ownership(gens)

    def dedup_registry(self):
        return self.r._rolling_dedup_registry()

    def sweep_pods(self, *, keep_round_ids: tuple[str, ...], keep_payers: set[str]) -> None:
        self.r._reconcile_funded_pods(keep_round_ids=keep_round_ids, keep_payers=keep_payers)

    def wall_seconds(self) -> float:
        # A shards leg first spends the CPU tier's lead (DEC-CA-0053) before
        # its GPU wall starts, so admission and the king's pre-train window
        # budget both. The GPU rent's own fit check runs after generation and
        # keeps the bare wall.
        lead = getattr(self.r, "_generation_lead_seconds", None)
        return float(self.r._leg_wall_seconds(None)) + (float(lead()) if callable(lead) else 0.0)

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
        self._king_rebench_at: dict[int, float] = {}  # era index -> last king re-bench try
        self._king_thread_seed: dict[int, int] = {}   # era index -> base seed of its king LEG
        self._init_threads: dict[int, threading.Thread] = {}  # era index -> init bench
        self._init_bench_at: dict[int, float] = {}    # era index -> last init bench try
        self._verify_thread: threading.Thread | None = None   # one operator verification at a time
        self._idle_released: set[int] = set()   # eras whose idle king pod was released early
        self._started = False
        self._reconciled_at = 0.0               # last periodic in-flight/pod reconcile
        # The ONE background lane-B settlement (DEC-CA-0050): {"rid", "thread",
        # "started", "digest", "error"} — off the tick thread, still ordered
        # before its lane-A manifest (see _lane_b_settle).
        self._lane_b_slot: dict | None = None
        # DEC-CA-0050 lane-B driver (None = lane B not wired in this process)
        self.lane_b = getattr(runner, "lane_b_driver", None)
        # DEC-CA-0056 lane-C driver (None = lane C not wired in this process)
        self.lane_c = getattr(runner, "lane_c_driver", None)

    # ── persistence ──────────────────────────────────────────────────────────

    def _save(self) -> None:
        with self._lock:
            save_state(self.state_path, self.state)

    # ── tick ─────────────────────────────────────────────────────────────────

    # Periodic in-flight / pod reconcile (T4): the startup re-entry, repeated
    # while the process lives, so an orphaned flight or a leaked pod is not
    # left until the next restart.
    RECONCILE_SECONDS = 1800.0

    def tick(self, client, block: int) -> None:
        from ..shared.config import effective_epoch_blocks

        block = int(block)
        now = self.clock()
        eb = int(effective_epoch_blocks(self.cfg.round, block))
        epoch_start = (block // eb) * eb
        era = era_for_block(self.cfg.round, block)
        self._sync_generations()
        self._reconcile_prepared_inits()
        if not self._started:
            self._startup(client, block)
            self._started = True
            self._reconciled_at = now
        # Settle BEFORE rolling: the boundary that starts era n+1 is era n's
        # last settlement (settlement_era), judged against era n's king. A
        # settlement still waiting (lane B settling, a chain read timed out)
        # holds the roll too — never roll past an unsettled boundary.
        if (self._chain_step("settlement", self._settle, client, block, epoch_start)
                and self._chain_step("era roll", self._roll_era, client, era, block)):
            self._requeue_stale_finished()
            # Boundaries of the era just started (a tick that skipped the era
            # boundary lands here with them pending): settle them now, not
            # next grid step.
            self._chain_step("settlement", self._settle, client, block, epoch_start)
        # While a lane-B settlement is in flight the era king is frozen: the
        # expert manifest it publishes is rebased onto (and binds) that king,
        # so the lane-A manifest that binds it must carry the same one.
        frozen = self._lane_b_pinned()
        if not frozen:
            self._chain_step("dethrone adoption", self._adopt_dethrone, client)
        self._chain_step("king legs", self._maybe_king_legs, client, block, now, frozen)
        self._drain_verifies()
        self._release_idle_king_pods()
        self._chain_step("intake", self._intake, client, block, now)
        self._chain_step("reconcile", self._maybe_reconcile, client, block, now)
        self._lane_b_tick(block)
        self._lane_c_tick(block)
        from ..shared.v2_status import publish_trainer_v2_status

        publish_trainer_v2_status(self, block)   # status/v2.json for the dashboards; never raises
        if self.lane_b is not None and hasattr(self.lane_b, "status"):
            from .lane_b_driver import publish_lane_b_status

            publish_lane_b_status(self, block)   # status/lane_b.json; never raises
        if self.lane_c is not None:
            from .lane_c_driver import publish_lane_c_status

            publish_lane_c_status(self, block)   # status/lane_c.json; never raises

    def _chain_step(self, name: str, fn, *args) -> bool:
        """Run one tick step; a chain read that hit its deadline
        (``TimeoutError``, LegOps.chain_read) skips the step this tick — logged,
        retried next poll — instead of failing the whole tick. Any other
        error propagates as before. Returns False when the step was skipped
        or reported it is still waiting (returned False)."""
        from .merge_acceptance import MergeAcceptancePending

        try:
            return fn(*args) is not False
        except MergeAcceptancePending as e:
            log.info("rolling: %s waits for lane-B acceptance (%s)", name, e)
            return False
        except TimeoutError as e:
            log.warning("rolling: %s skipped this tick — chain read timed out (%s)", name,
                        e or "deadline")
            return False

    # ── lane B (DEC-CA-0050) ─────────────────────────────────────────────────

    def _lane_b_tick(self, block: int) -> None:
        drv, cur = self.lane_b, self.state.current
        if drv is None or cur is None or not self.cfg.lane_b.active(block):
            return
        king = cur.king()
        if king is None:
            return                       # legs anchor on the era king's checkpoint
        try:
            drv.tick(block, cur.index, king.trained_pointer)
        except Exception as e:  # noqa: BLE001 — lane B never stalls lane A
            log.warning("rolling: lane B tick failed: %s", e)

    # Lane-B settlement bounds. The lane-A manifest binds the expert
    # manifest's digest, so lane B must settle FIRST (DEC-CA-0050) — it cannot
    # run after or alongside its own lane-A publish. It runs in ONE background
    # slot instead of on the tick thread: each tick waits at most
    # LANE_B_SETTLE_WAIT_SECONDS for it, then the lane-A boundary waits
    # (nothing published, the roll held) while intake, king legs and
    # verifications carry on. Past LANE_B_SETTLE_MAX_SECONDS the settlement
    # is abandoned: the driver commits nothing for that round after it (its
    # legs stay finished for the next settlement) and lane A binds whatever
    # the driver had already committed — never a digest it could still change.
    LANE_B_SETTLE_WAIT_SECONDS = 30.0
    LANE_B_SETTLE_MAX_SECONDS = 3600.0

    def _lane_b_settle(self, round_id: str, block: int, cur: EraState, king) -> str | None:
        """Settle lane B for this boundary BEFORE the lane-A manifest is built
        and published; returns the expert manifest digest the lane-A manifest
        binds ("" = no lane-B legs this round — and on any failure, which
        keeps the finished legs for the next settlement and binds nothing),
        or None while the background settlement is still running (the
        lane-A boundary waits for the next tick)."""
        drv = self.lane_b
        if drv is None or king is None or not self.cfg.lane_b.active(block):
            return ""
        rid = str(round_id)
        slot = self._lane_b_slot
        if slot is not None and slot["rid"] != rid:
            # A settlement of another round is still in the slot (its boundary
            # closed without a publish): stop it from committing, and never run
            # two settlements over the same finished legs at once.
            if slot["thread"].is_alive():
                self._lane_b_abandon(drv, slot["rid"])
                slot["abandoned"] = True
                log.warning("rolling: lane B settlement of %s still running — %s waits "
                            "for it to exit", slot["rid"], rid)
                return None
            slot = None
        if slot is None:
            slot = {"rid": rid, "started": self.clock(), "digest": "", "error": None,
                    "king": king.trained_pointer, "boundary": int(block)}

            def _run(slot=slot) -> None:
                try:
                    drv.settle(round_id=rid, block=int(block), era_index=cur.index,
                               king_pointer=king.trained_pointer,
                               work_dir=Path(self.r.work_root) / "lane_b")
                    slot["digest"] = drv.expert_manifest_digest(rid)
                except Exception as e:  # noqa: BLE001 — lane B never stalls lane A
                    slot["error"] = e

            slot["thread"] = threading.Thread(target=_run, name=f"laneb-settle-{rid}",
                                              daemon=True)
            self._lane_b_slot = slot
            slot["thread"].start()
        slot["thread"].join(self.LANE_B_SETTLE_WAIT_SECONDS)
        if slot["thread"].is_alive():
            waited = self.clock() - float(slot["started"])
            if waited < self.LANE_B_SETTLE_MAX_SECONDS:
                log.info("rolling: lane B settlement of %s still running (%.0fs) — the "
                         "lane-A settlement waits for it", rid, waited)
                return None
            digest = self._lane_b_abandon(drv, rid)
            slot["abandoned"] = True
            log.warning("rolling: lane B settlement of %s exceeded %.0fs — abandoned; lane A "
                        "binds %s", rid, self.LANE_B_SETTLE_MAX_SECONDS,
                        digest or "nothing (its legs wait for the next settlement)")
            return digest
        self._lane_b_slot = None
        if slot["king"] != king.trained_pointer:
            # cannot happen while the king is frozen (tick); never silent
            log.error("rolling: lane B settled %s against king %s but the lane-A manifest "
                      "carries %s", rid, slot["king"], king.trained_pointer)
        if slot["error"] is not None:
            log.warning("rolling: lane B settlement failed at %s: %s", rid, slot["error"])
            return ""
        return str(slot["digest"] or "")

    def _lane_b_pinned(self) -> bool:
        """Whether a lane-B settlement still pins the current era king: one is
        in the slot, not abandoned, and its boundary is still open. A slot
        whose boundary closed some other way (the settlement withheld, the
        king entry dropped) is let go — abandoned first if it still runs — so
        it can never keep the king frozen."""
        slot = self._lane_b_slot
        if slot is None or slot.get("abandoned"):
            return False
        if int(slot["boundary"]) > int(self.state.last_settled_boundary):
            return True
        if slot["thread"].is_alive():
            self._lane_b_abandon(self.lane_b, slot["rid"])
            slot["abandoned"] = True
        else:
            self._lane_b_slot = None
        return False

    @staticmethod
    def _lane_b_abandon(drv, rid: str) -> str:
        """Tell the driver ``rid`` is abandoned (it commits nothing for it from
        now on) and return the digest it had ALREADY committed ("" = none)."""
        abandon = getattr(drv, "abandon", None)
        try:
            return str((abandon(rid) if callable(abandon)
                        else drv.expert_manifest_digest(rid)) or "")
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: lane B abandon of %s failed: %s", rid, e)
            return ""

    # ── lane C (DEC-CA-0056) ─────────────────────────────────────────────────

    def _lane_c_tick(self, block: int) -> None:
        drv, cur = self.lane_c, self.state.current
        if drv is None or cur is None or not self.cfg.lane_c.era_active(int(cur.start_block)):
            return
        king = cur.king()
        if king is None:
            return                       # legs fine-tune the era king's checkpoint
        try:
            drv.tick(block, cur.index, king.trained_pointer, int(cur.start_block))
        except TimeoutError as e:
            log.warning("rolling: lane C tick skipped — chain read timed out (%s)", e)
        except Exception as e:  # noqa: BLE001 — lane C never stalls lane A
            log.warning("rolling: lane C tick failed: %s", e)

    def _lane_c_settle(self, round_id: str, block: int, created_block: int, cur: EraState,
                       king) -> tuple[str, dict | None]:
        """Settle lane C for this boundary BEFORE the lane-A manifest is built:
        ``(digest, published)`` — the sha256 of the lane-C manifest text the
        lane-A manifest binds (``lane_c_digest``; "" = nothing settled, and on
        ANY failure: its legs wait for the next settlement and nothing is
        bound) and ``{key, text}`` of it for the lane-A write-ahead (a replay
        republishes the identical bytes first). Synchronous: no evaluation
        runs here, only a sign and one put."""
        drv = self.lane_c
        if drv is None or king is None or not self.cfg.lane_c.active(block):
            return "", None
        try:
            digest = drv.settle(round_id=str(round_id), block=int(block),
                                era_index=cur.index, base_pointer=king.trained_pointer,
                                created_block=int(created_block),
                                base_round_id=str(round_id), pool_block=int(block))
        except Exception as e:  # noqa: BLE001 — lane C never stalls lane A
            log.warning("rolling: lane C settlement failed at %s (its legs wait; nothing "
                        "bound): %s", round_id, e)
            return "", None
        return str(digest or ""), (drv.published(str(round_id)) if digest else None)

    def _lane_c_live(self) -> set[str]:
        drv = self.lane_c
        if drv is None:
            return set()
        try:
            return set(drv.live_hotkeys())
        except Exception:  # noqa: BLE001
            return set()

    def _lane_b_live(self) -> set[str]:
        """Hotkeys with a miner-funded lane-B leg running (DEC-CA-0059 item 8):
        their payer pods survive the sweep and they start no lane-A leg."""
        drv = self.lane_b
        live = getattr(drv, "live_hotkeys", None)
        if not callable(live):
            return set()
        try:
            return set(live())
        except Exception:  # noqa: BLE001
            return set()

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
        self._reattach_flights(client, block, queue, flights, done, dead)

    def _maybe_reconcile(self, client, block: int, now: float) -> None:
        """Every ``RECONCILE_SECONDS``: the startup re-entry for a LIVE
        process. An in-flight queue entry with no leg thread here (its thread
        died without settling the entry) is re-attached — or requeued
        unburned when its era is over — and the pod sweep runs again.

        The sweep can never take a pod a live leg owns: it keeps both era
        king pods (current + next, by name) and every payer / operator-rented
        leg pod of a hotkey that is in flight in the queue OR has a leg
        thread here (a settled leg's thread still runs its payer-pod bench
        after ``mark_done``). Ledger rows and payer pods of anything else are
        leftovers. The sweep and every leg launch run on the tick thread, so
        no leg can be admitted between computing the keep-set and the sweep."""
        if now - self._reconciled_at < self.RECONCILE_SECONDS:
            return
        self._reconciled_at = now
        self._refresh_king_key(self.state.current)
        queue = self.ops.queue()
        if queue is None:
            return
        try:
            flights = queue.in_flight()
            done = {(f.hotkey, f.ref) for f in self.state.finished}
            orphans = [e for e in flights
                       if e.hotkey not in self._threads and (e.hotkey, e.ref) not in done]
            dead = self._dead_era_flights(client, orphans, done, block)
            if orphans:
                log.warning("rolling: %d in-flight leg(s) with no leg thread (%s) — "
                            "re-attached / requeued", len(orphans),
                            ", ".join(e.hotkey[:12] for e in orphans))
                self._reattach_flights(client, block, queue, orphans, done, dead)
            keep_ids = tuple(str(e.base_seed)
                             for e in (self.state.current, self.state.next) if e)
            keep = ({e.hotkey for e in queue.in_flight()} | set(self._threads)) - set(dead)
            # DEC-CA-0056 / DEC-CA-0059: a running lane-C / lane-B leg's payer
            # pod is never swept
            keep |= self._lane_c_live() | self._lane_b_live()
            self.ops.sweep_pods(keep_round_ids=keep_ids, keep_payers=keep)
        except TimeoutError:
            raise
        except Exception as e:  # noqa: BLE001 — housekeeping never sinks a tick
            log.warning("rolling: periodic reconcile failed (retried in %.0fs): %s",
                        self.RECONCILE_SECONDS, e)

    def _reattach_flights(self, client, block: int, queue, flights, done: set,
                          dead: dict) -> None:
        for e in flights:
            if (e.hotkey, e.ref) in done:
                continue                      # finished before the restart; settles normally
            if e.hotkey in dead:
                era_idx, era_end = dead[e.hotkey]
                # Its era ended while the trainer was away and no checkpoint
                # landed: the leg can never be judged (a leg that finishes
                # after its era is requeued at finish anyway), so re-attaching
                # only bills the payer for a second run — or a fresh pod when
                # the first is gone. Pod released by the sweep above; requeue
                # unburned now.
                log.warning("rolling: in-flight leg %s belongs to era %d, which ended at "
                            "block %d (now %d) — it can never be judged; pod released, "
                            "requeued unburned", e.hotkey[:12], era_idx, era_end, int(block))
                queue.requeue(e.hotkey, error="trainer restarted after the leg's era ended",
                              error_class="no_capacity", burn_attempt=False)
                continue
            era = self._era_state_for(e.era_index)
            if era is None and int(e.era_index) <= 0 and self.state.current is not None:
                # A leg stamped with no era (dispatched at the rollover tick)
                # belongs to the era that is running now — re-attach it there
                # rather than requeue (a requeue rents a second pod while the
                # first keeps training).
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
            if v2_era_active(self.cfg, era.start_block) and (
                    e.training_init != era.warm_start_ckpt
                    or e.training_contract_digest != self._contract_identity(era)):
                queue.requeue(e.hotkey, error="in-flight init/contract is missing or incompatible after restart",
                              error_class="infra", burn_attempt=False, operator_billed=True)
                continue
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
        the entry re-stamped so the next restart reads a sound one. A requeue
        zeroes era/target/start, and a leg re-attached from such a stamp would
        publish ``train_block 0`` — every validator rejects that settlement
        whole (``era_entry_out_of_window``)."""
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

    def _contract_identity(self, era):
        from ..shared.manifest import contract_digest

        return contract_digest(self.cfg.training.at_block(int(era.start_block)))

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
            self.state.last_published_manifest_sha256 = ""   # its bytes are not ours to vouch for
            self._save()
        log.info("rolling: manifest chain root seeded from latest.json (round %s)", root)
        return True

    def _uid_of(self, client, hotkey: str) -> int:
        try:
            uid = self.ops.chain_read(lambda: client.uid_for_hotkey(hotkey))
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
            self._reconcile_prepared_inits()
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
        self._reconcile_prepared_inits()

    def _reconcile_prepared_inits(self):
        # A notice/record read can recover after the next era was prepared.
        # Reuse neither its seeds nor its checkpoint under a different init.
        for name in ("current", "next"):
            era = getattr(self.state, name)
            if era is None or not v2_era_active(self.cfg, era.start_block):
                continue
            want_gen, want_idx, want_init, _ = era_init(
                self.state.generations, era.index,
                lambda eff, era=era: v2_generation_usable(self.cfg, era.start_block, eff))
            if (era.generation, era.member_index, era.warm_start_base or era.warm_start_ckpt) != (
                    want_gen, want_idx, want_init):
                self._discard_prepared_era(name, "verified promotion changed the era init")

    # ── eras ─────────────────────────────────────────────────────────────────

    def _warm_eras(self) -> list[int]:
        """Effective eras of the v2-lineage generations in the ledger — the
        warm starts that open a DEC-CA-0055 seed epoch. Empty pre-v2."""
        warm = v2_first_warm_era(self.cfg)
        if warm is None or not getattr(getattr(self.cfg, "v2", None), "king_ladder", False):
            return []
        return sorted({int(rec.get("effective_era", 0) or 0)
                       for rec in self.state.generations.values()
                       if int(rec.get("effective_era", 0) or 0) >= int(warm)})

    def _epoch_of(self, era_index: int) -> int | None:
        """First era of the seed epoch containing ``era_index`` (``None``
        pre-v2, or without ``[v2] king_ladder``)."""
        if not getattr(getattr(self.cfg, "v2", None), "king_ladder", False):
            return None
        return v2_seed_epoch_start(self.cfg, int(era_index), self._warm_eras())

    @staticmethod
    def _in_epoch_tail(era: EraState | None) -> bool:
        """``era`` is a LATER era of its seed epoch: no init change, no king
        leg of its own (the epoch's leg is carried)."""
        return era is not None and int(era.epoch_start) >= 0 and int(era.index) != int(
            era.epoch_start)

    @staticmethod
    def _same_epoch(a: EraState | None, b: EraState | None) -> bool:
        return (a is not None and b is not None and int(a.epoch_start) >= 0
                and int(a.epoch_start) == int(b.epoch_start)
                and int(a.base_seed) == int(b.base_seed))

    def _new_era_state(self, client, era: EraSpec) -> EraState:
        gen, idx, ckpt, size = era_init(
            self.state.generations, era.index,
            lambda eff: v2_generation_usable(self.cfg, era.start_block, eff))
        # DEC-CA-0055 seed epochs: every era of a warm-start epoch trains under
        # the epoch's FIRST era's seed block (stamped as the era's seed_block:
        # the train window and the era's training seeds both derive from it).
        warm = self._warm_eras()
        epoch = self._epoch_of(era.index)
        seed_block = (int(v2_era_seed_block(self.cfg, era, warm)) if epoch is not None
                      else int(era.seed_block))
        base, merges = "", []
        if (self.lane_b is not None and ckpt and self.cfg.lane_b.active(era.start_block)):
            from .merge_acceptance import MergeAcceptancePending

            try:
                if epoch is not None and int(epoch) != int(era.index):
                    # Lane-B merges change the init only at a warm start: a
                    # later era of the epoch reuses the epoch's merged init
                    # (if any) and never applies a merge pending since.
                    merged, merges = self.lane_b.init_for(
                        int(epoch), ckpt, Path(self.r.work_root) / "lane_b",
                        apply_pending=False)
                else:
                    prepare = getattr(self.lane_b, "prepare_init_for", self.lane_b.init_for)
                    merged, merges = prepare(
                        era.index, ckpt, Path(self.r.work_root) / "lane_b",
                        era_start=int(era.start_block))
                if merges:
                    base, ckpt = ckpt, merged
            except MergeAcceptancePending:
                raise
            except Exception as e:  # noqa: BLE001 — fall back to the plain member
                log.warning("rolling: lane B merged init for era %d failed (%s); "
                            "training from the member", era.index, e)
                base, merges = "", []
        return EraState(index=era.index, start_block=era.start_block,
                        seed_block=seed_block,
                        base_seed=self.ops.block_seed(client, seed_block),
                        generation=gen, member_index=idx,
                        warm_start_ckpt=ckpt, warm_start_size=size,
                        warm_start_base=base, warm_start_merges=list(merges),
                        epoch_start=(-1 if epoch is None else int(epoch)))

    def _carry_epoch_king(self, src: EraState, dst: EraState) -> bool:
        """Copy ``src``'s king entry (and the epoch's init bench) into a later
        era ``dst`` of the SAME seed epoch: same seeds, same init, so the
        epoch's one king leg — or the dethroner's entry that replaced it — is
        ``dst``'s king leg. Returns True when a king entry was carried. Caller
        holds the lock."""
        if not self._same_epoch(src, dst) or src.warm_start_ckpt != dst.warm_start_ckpt:
            return False
        if dst.init_bench is None and src.init_bench is not None:
            dst.init_bench, dst.init_bench_tries = dict(src.init_bench), src.init_bench_tries
            dst.init_pointer = src.init_pointer
        if not src.king_entry or src.king_hotkey != dst.king_hotkey or dst.king_entry:
            return False
        if src.king_init != src.warm_start_ckpt:
            return False
        dst.king_entry = dict(src.king_entry)
        dst.king_init = dst.warm_start_ckpt
        dst.king_uid, dst.king_ref = src.king_uid, src.king_ref
        dst.king_bench = dict(src.king_bench) if src.king_bench else None
        dst.king_bench_published = False       # rides dst's first report
        dst.king_funded_by = src.king_funded_by
        if dst.king_funded_by == "operator" and self.ops.king_key_ok(dst.king_hotkey):
            # the king rotated to a working key: the slash lifts from this era
            dst.king_funded_by = "king"
            log.info("rolling: era %d king %s fixed its key — slash lifted", dst.index,
                     dst.king_hotkey[:12])
        log.info("rolling: era %d carries seed-epoch %d's king leg %s (no retrain)",
                 dst.index, dst.epoch_start, _entry_from_json(dst.king_entry).trained_pointer)
        return True

    def update_config(self, cfg) -> None:
        """A dated launch can invalidate an era prepared under the old contract."""
        old = self.cfg
        self.cfg = cfg
        for name in ("current", "next"):
            era = getattr(self.state, name)
            if (era is not None and v2_era_active(cfg, era.start_block)
                    and not v2_era_active(old, era.start_block)):
                self._discard_prepared_era(name, "training contract changed at v2 launch")

    def _discard_prepared_era(self, name, reason):
        with self._lock:
            era = getattr(self.state, name)
            if era is None:
                return
            # Running callbacks retain this object and must discard their result.
            era._superseded = True
            setattr(self.state, name, None)
            stale = [leg for leg in self.state.finished if leg.era_index == era.index]
            self.state.finished = [leg for leg in self.state.finished if leg.era_index != era.index]
            self._save()
        self.ops.retire_king_pod(era)
        log.warning("rolling: prepared era %d discarded: %s", era.index, reason)
        queue = self.ops.queue()
        if queue is not None:
            for leg in stale:
                queue.requeue(leg.hotkey, error=reason,
                              error_class="infra", burn_attempt=False, operator_billed=True)

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
                    if not self._same_epoch(nxt, cur):   # a shared epoch pod stays
                        self.ops.retire_king_pod(nxt)
                new = self._new_era_state(client, era)
                new.king_hotkey = king_hotkey
            same_epoch = self._same_epoch(cur, new)
            if same_epoch:
                # DEC-CA-0055: the epoch's king leg (or the dethroner's entry
                # that replaced it) is this era's king leg — no retrain.
                self._carry_epoch_king(cur, new)
            if cur is not None:
                if (self.lane_b is not None and cur.king_entry and not same_epoch
                        and v2_era_active(self.cfg, int(cur.start_block))):
                    from ..shared.manifest import contract_digest

                    entry = _entry_from_json(cur.king_entry)
                    self.lane_b.record_lineage_epoch(
                        f"{cur.epoch_start}:{cur.seed_block}:{cur.base_seed}", {
                        "pointer": entry.trained_pointer, "gen_ref": entry.gen_ref,
                        "corpus_digest": entry.corpus_digest,
                        "contract_digest": contract_digest(
                            self.cfg.throne_contracts_at(int(cur.start_block))[0]),
                        "warm_start_ref": cur.warm_start_ckpt or "",
                        "token_budget": self.cfg.throne_contracts_at(
                            int(cur.start_block))[0].train_tokens,
                    })
                if same_epoch:
                    # same pod-name prefix (shared base seed): hand the pod over
                    # — a teardown by prefix would also take the new era's
                    self.ops.carry_king_pod(cur, new)
                else:
                    self.ops.retire_king_pod(cur)
                # Legs finished under the old era but never settled (no king
                # leg all era) are judged never against another era's king.
                self._requeue_stale_finished(era.index)
                if not same_epoch:
                    # inside a seed epoch the settled legs stay: a dethroner
                    # crowned at the old era's last settlement is adopted from
                    # here (same seeds and init)
                    self.state.published = []
                    self.state.vbench_carries = {}
            self.state.current = new
            self.state.next = None
            self._save()
            log.info("rolling: era %d started at block %d (seed block %d, generation %d "
                     "member %d, king %s%s)", era.index, era.start_block, new.seed_block,
                     new.generation, new.member_index, (new.king_hotkey or "?")[:12],
                     " — cached king leg" if new.king_entry else "")
        self._refresh_king_key(new)

    def _requeue_stale_finished(self, era_index: int | None = None) -> None:
        """Requeue (unburned) every finished-but-unsettled leg of an era older
        than ``era_index`` (default: the current era) and drop it from
        ``state.finished``: a dead era has no settlement left, so the leg can
        never be judged. Called at the rollover and on every tick — a leg
        recorded under a dead era by any other path (a bench that outlived
        its era) is swept up at once, not at the next rollover."""
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
                          error_class="no_capacity", burn_attempt=False, operator_billed=True)
        self._save()

    def _refresh_king_key(self, era: EraState | None) -> None:
        """DEC-CA-0055 item 6: the king's own payer-vault key funds its leg,
        but the king trains once per seed epoch (days) while a vault entry
        lives ``funded_entry_ttl_hours``. Re-stamp its TTL at every era roll
        and periodic reconcile while it is king in a v2 era, so the key is not
        purged mid-reign. Best-effort; never raises."""
        if era is None or not era.king_hotkey or not self._ladder_era(era):
            return
        try:
            if self.ops.refresh_king_key(era.king_hotkey):
                log.debug("rolling: king %s payer key TTL re-stamped", era.king_hotkey[:12])
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: king key TTL re-stamp failed (ignored): %s", e)

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
        and the new king's leg never trained that era. Drop it so the king leg retrains for the real king. Returns True when
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

    def _maybe_king_legs(self, client, block: int, now: float,
                         freeze_current: bool = False) -> None:
        cur = self.state.current
        if cur is None:
            return
        if not freeze_current:       # (a lane-B settlement pins the current king)
            self._drop_foreign_king_entry(cur)
            if (cur.king_entry is not None and cur.index not in self._king_threads
                    and self._forfeit_switch(cur)):  # armed mid-reign: the running era switches too
                self.ops.retire_king_pod(cur)
            if (cur.king_entry is None and cur.king_hotkey
                    and cur.index not in self._king_threads):
                self._launch_king_leg(client, cur, block, now)
        self._drop_foreign_king_entry(self.state.next)
        self._maybe_rebench_king(cur, now)
        self._maybe_bench_init(cur, now)
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
                if self._in_epoch_tail(nxt):
                    self._carry_epoch_king(cur, nxt)
                self._save()
            if nxt.king_entry is None:
                log.info("rolling: pre-train window for era %d open — king leg starts",
                         nxt.index)
                self._launch_king_leg(client, nxt, block, now)
        elif (self.state.next is not None and self.state.next.king_entry is None
              and self.state.next.index not in self._king_threads):
            self._launch_king_leg(client, self.state.next, block, now)

    KING_REBENCH_SECONDS = 600.0

    def _maybe_rebench_king(self, era: EraState, now: float) -> None:
        """LEGACY-ONLY (trainer bench source, DEC-CA-0054): inert for an era
        whose lane-A rows come from the main validator (DEC-CA-0063) — kept
        for testnet / any ``bench_source = "trainer"`` era. Under the three-suite bench gate (DEC-CA-0054) every challenger is
        paired with the king's rows, so a king entry without a bench (the
        sweep failed, no bench host, or a cached leg reused without one) is
        re-benched on a later tick instead of leaving the era uncomputable."""
        if not bench_gate_configured(self.cfg) or validator_bench_era(
                self.cfg, int(era.start_block)):
            return          # (DEC-CA-0063: the main validator benches the king)
        if era.king_entry is None or era.king_bench is not None:
            return
        if era.index in self._king_threads:
            return
        if now - self._king_rebench_at.get(era.index, 0.0) < self.KING_REBENCH_SECONDS:
            return
        self._king_rebench_at[era.index] = now
        entry = era.king()

        def _run():
            try:
                bench = self.ops.bench_king(entry, era)
                with self._lock:
                    if era.king_entry is not None and era.king().trained_pointer == \
                            entry.trained_pointer:
                        era.king_bench = bench
                        self._save()
                log.info("rolling: era %d king re-bench %s", era.index,
                         "landed" if bench is not None else "came back empty (retried)")
            except Exception as e:  # noqa: BLE001 — retried after KING_REBENCH_SECONDS
                log.warning("rolling: era %d king re-bench failed: %s", era.index, e)
            finally:
                self._king_threads.pop(era.index, None)

        t = threading.Thread(target=_run, name=f"king-bench-era{era.index}", daemon=True)
        self._king_threads[era.index] = t
        t.start()

    def king_end_wall(self, era: EraState, *, block: int, now: float) -> float:
        """The king leg's own target: the wall-clock of ``era``'s LAST
        settlement (its end block). Like a challenger's target boundary this
        bounds the rent wait — a sold-out marketplace is waited out up to
        the king's latest safe start (end − wall − margin), not given up on
        at the first empty listing."""
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
        preflight = getattr(self.r, "_v2_baseline_ready", None)
        if preflight is not None and not preflight(gen, int(era.start_block)):
            return
        if self._in_epoch_tail(era):
            # DEC-CA-0055: the king trains once per seed epoch. A later era
            # takes the epoch's leg from a sibling era state, or waits while
            # that leg is still training; it trains only when no epoch leg
            # exists for this king at all (a forfeiture successor, lost state).
            with self._lock:
                for other in (self.state.current, self.state.next):
                    if other is not None and other is not era:
                        self._carry_epoch_king(other, era)
                if era.king_entry:
                    self._save()
                    return
            if self._epoch_king_busy(era):
                return
        cached = self.ops.cached_leg(era, "king", gen)
        if cached is not None:
            funded = self.ops.king_funding(era, gen) if era.epoch_start >= 0 else ""
            with self._lock:
                era.king_entry = _entry_to_json(replace(cached, role="king"))
                era.king_init = era.warm_start_ckpt     # cached_leg matched on it
                era.king_funded_by = funded
                self._save()
            log.info("rolling: era %d king leg already complete (cached) — reused", era.index)
            return
        if self._in_epoch_tail(era):
            log.warning("rolling: era %d (seed epoch %d) has no epoch king leg for %s — "
                        "training one under the epoch's seeds and init", era.index,
                        era.epoch_start, era.king_hotkey[:12])

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
                funded = self.ops.king_funding(era, gen) if era.epoch_start >= 0 else ""
                with self._lock:
                    era.king_entry = _entry_to_json(entry)
                    era.king_init = era.warm_start_ckpt
                    era.king_funded_by = funded
                    era.king_leg_failed = ""
                    era.king_leg_failures = 0
                    era.king_leg_failures_total = 0
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
                    from ..funding.faults import classify_rent_failure

                    if classify_rent_failure(str(e)) not in {"auth", "rate_limited"}:
                        era.king_leg_failures += 1
                        era.king_leg_failures_total += 1
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
                self._king_thread_seed.pop(era.index, None)

        t = threading.Thread(target=_run, name=f"king-era{era.index}", daemon=True)
        self._king_threads[era.index] = t
        self._king_thread_seed[era.index] = int(era.base_seed)
        t.start()

    def _epoch_king_busy(self, era: EraState) -> bool:
        """Another era's king LEG of the same seed epoch is still training —
        its persisted record becomes ``era``'s king leg once it lands."""
        for idx, seed in list(self._king_thread_seed.items()):
            t = self._king_threads.get(idx)
            if (idx != era.index and int(seed) == int(era.base_seed)
                    and t is not None and t.is_alive()):
                return True
        return False

    # ── the init bench (DEC-CA-0055) ─────────────────────────────────────────

    INIT_BENCH_SECONDS = 600.0
    INIT_BENCH_MAX_TRIES = 3

    def _init_bench_owed(self, era: EraState) -> bool:
        """A v2 seed epoch's init still needs its bench (the "init" entry of
        the signed bench report the king check compares against)."""
        return (int(era.epoch_start) >= 0 and self._ladder_era(era)
                and era.init_bench is None
                and int(era.init_bench_tries) < self.INIT_BENCH_MAX_TRIES)

    def _ladder_era(self, era: EraState) -> bool:
        """A v2 era under the DEC-CA-0055 king ladder (``[v2] king_ladder``):
        king-only promotions, the init bench, the king's own key."""
        return bool(v2_era_active(self.cfg, int(era.start_block))
                    and getattr(getattr(self.cfg, "v2", None), "king_ladder", False))

    def _maybe_bench_init(self, era: EraState | None, now: float) -> None:
        """Bench the CURRENT init once per seed epoch on the era's operator
        king pod (after the king leg: the pod exists then). A later era of the
        epoch carries the numbers (``_carry_epoch_king``)."""
        if era is None or era.king_entry is None or not self._init_bench_owed(era):
            return
        if era.index in self._king_threads or era.index in self._init_threads:
            return
        if now - self._init_bench_at.get(era.index, 0.0) < self.INIT_BENCH_SECONDS:
            return
        self._init_bench_at[era.index] = now

        def _run():
            try:
                bench = self.ops.bench_init(era)
                with self._lock:
                    if bench is not None:
                        bench = dict(bench)
                        era.init_pointer = str(bench.pop("_pointer", "") or "")
                        era.init_bench = bench
                    else:
                        era.init_bench_tries += 1
                    self._save()
                log.info("rolling: era %d init bench %s", era.index,
                         "landed" if bench is not None else "came back empty")
            except Exception as e:  # noqa: BLE001 — retried after INIT_BENCH_SECONDS
                with self._lock:
                    era.init_bench_tries += 1
                    self._save()
                log.warning("rolling: era %d init bench failed: %s", era.index, e)
            finally:
                self._init_threads.pop(era.index, None)

        t = threading.Thread(target=_run, name=f"init-bench-era{era.index}", daemon=True)
        self._init_threads[era.index] = t
        t.start()

    def _can_bench(self, era: EraState) -> bool:
        try:
            return bool(self.ops.king_pod_can_bench(era))
        except Exception:  # noqa: BLE001 — unknown ⇒ keep the pod for its queue
            return True

    def _release_idle_king_pods(self) -> None:
        """Release an era's operator king pod the moment it has nothing to do.

        Holding it for the whole era bills an idle GPU for payer re-benches
        that mostly never come. Released when:
        the era's king leg is in, its bench thread is done, no verification
        is running or queued. A verification queued later waits ("wait" costs
        no attempt) for the next king pod — the next era's pod is rented for
        its own king leg anyway — and its numbers republish into their round's
        report then. A pod that cannot bench at all (driver too old) is
        released even with verifications queued — they wait for the next pod.
        Never under a running sweep (``retire_king_pod`` defers on the bench
        lock). Idempotent per process; a restart re-checks once."""
        with self._lock:
            if self._verify_thread is not None and self._verify_thread.is_alive():
                return
            queued = bool(self.state.verify_queue)
            eras = [e for e in (self.state.current, self.state.next) if e is not None]
            idle = [e for e in eras
                    if e.king_entry is not None
                    and e.index not in self._king_threads
                    and e.index not in self._init_threads
                    and e.index not in self._idle_released
                    and not self._king_bench_owed(e)
                    and not self._init_bench_owed(e)]
        if queued:
            # Queued verifications keep the pod — unless it cannot bench them at
            # all (driver too old): then they wait for the next king pod instead
            # of holding an idle GPU.
            idle = [e for e in idle if not self._can_bench(e)]
        with self._lock:
            for e in idle:
                self._idle_released.add(e.index)
        for e in idle:
            log.info("rolling: era %d king pod released — king leg and bench done, %s "
                     "(a later verification waits for the next king pod)", e.index,
                     "its pod cannot bench the queued verifications" if queued
                     else "nothing to verify")
            try:
                self.ops.release_idle_king_pod(e)
            except Exception as ex:  # noqa: BLE001 — the rollover teardown still runs
                log.warning("rolling: early release of era %d king pod failed: %s", e.index, ex)

    def _king_bench_owed(self, era: EraState) -> bool:
        """Under the three-suite bench gate (DEC-CA-0054) an era whose king
        has no bench yet still needs its pod: every challenger pairs with the
        king's rows, and a released pod also drops the host mapping the
        re-bench (``_maybe_rebench_king``) benches on. The gap between two
        re-bench tries registers no thread, so "no king thread" alone is not
        "nothing to do"."""
        if era.king_bench is not None or not bench_gate_configured(self.cfg):
            return False
        if validator_bench_era(self.cfg, int(era.start_block)):
            return False            # DEC-CA-0063: no trainer king bench to owe
        start = int(era.start_block)
        end = start + int(era_length_blocks(self.cfg.round, start))
        return any(bench_gate_mode_at(self.cfg, b) != "off" for b in (start + 1, end))

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
        # The v2 release's opening window reserves paid A entries for v2.
        # Do not dispatch them under the still-live v1 contract, even if the
        # operator installed this release days before the scheduled opening.
        if ((self.cfg.v2.commit_open_at_utc or self.cfg.v2.commit_from_block)
                and (not self.cfg.v2.active(block) or not self.cfg.v2.commit_from_block)):
            queue.hold_for_launch()
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
        from ..shared.calendar_launch import effective_commit_floor

        floor = effective_commit_floor(self.cfg, block)
        if floor:
            queue.rebase_preopening_reveals(floor, lambda hk, ref: self._reveal_block(history, hk, ref, block))
        queue.expire_stale()
        cap = int(self.cfg.round.funded_field_cap or 0) or max(1, int(self.cfg.round.finalist_cap))
        in_flight = len(queue.in_flight())
        burned = self.ops.burned(block)
        wall, margin = self.ops.wall_seconds(), self.ops.margin_seconds()
        held: list[str] = []
        not_jumps: set[str] = set()       # seniors that could not start this pass
        queue_now = getattr(queue, "clock", None)
        queue_now = float(queue_now()) if callable(queue_now) else now
        for entry in select_field(queue.entries(), cap=0):
            if in_flight >= cap:
                held.append(entry.hotkey)
                not_jumps.add(entry.hotkey)
                continue
            nb = float(getattr(entry, "not_before", 0.0) or 0.0)
            if nb and queue_now < nb:
                # Backing off after an operator-fault requeue: mark_in_flight
                # would refuse it anyway — skip it before admit/dedup run.
                held.append(entry.hotkey)
                not_jumps.add(entry.hotkey)
                continue
            if entry.hotkey in burned:
                queue.fail(entry.hotkey, error="hotkey already used its one submission "
                           "this round; fund again next round", error_class="burned",
                           expect_ref=entry.ref)
                continue
            from .admission import owner_busy

            real_runner = getattr(self.ops, "r", None)
            if real_runner is not None and owner_busy(real_runner, entry.hotkey):
                held.append(entry.hotkey)
                not_jumps.add(entry.hotkey)
                continue
            if entry.hotkey in self._lane_c_live() | self._lane_b_live():
                # DEC-CA-0056 / DEC-CA-0059: one leg per hotkey across lanes
                # (pod names, the harvest dir and the payer key are per
                # hotkey) — waits
                held.append(entry.hotkey)
                not_jumps.add(entry.hotkey)
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
            reveal_block = self._reveal_block(history, entry.hotkey, entry.ref, block)
            if reveal_block is None or reveal_block < floor:
                queue.fail(entry.hotkey, error="This reveal precedes the competition opening; "
                           "make a fresh commitment and fund it. No training attempt was consumed.",
                           error_class="commit_before_open", expect_ref=entry.ref)
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
                        if self._in_epoch_tail(era):
                            self._carry_epoch_king(cur, era)
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
                                        era_index=adm.era_index, started_block=block,
                                        training_init=era.warm_start_ckpt,
                                        training_contract_digest=self._contract_identity(era)):
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
    def _reveal_block(history: list, hotkey: str, ref: str, as_of: int | None = None) -> int | None:
        from ..interface.validation import parse_commit

        best = None
        for c in history:
            if c.hotkey != hotkey or (as_of is not None and int(c.commit_block) > as_of):
                continue
            parsed = parse_commit(c.payload)
            if parsed is not None and parsed.ref == ref:
                value = int(c.commit_block)
                best = value if best is None else max(value, best)
        return best

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
        self.ops.burn([gen], block if block is not None else
                      (self.state.current.start_block if self.state.current else -1))
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
        self.ops.burn([gen], era.start_block)
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
            if getattr(era, "_superseded", False):
                try:
                    self.ops.teardown_kept_pod(gen.hotkey)
                except Exception as e:  # noqa: BLE001
                    log.error("rolling: superseded pod cleanup for %s failed: %s", gen.hotkey[:12], e)
                if queue is not None:
                    queue.requeue(gen.hotkey, error="training contract changed at v2 launch",
                                  error_class="infra", burn_attempt=False, operator_billed=True)
                return
            self._leg_walls().record(gen.ref, self.clock() - t_launch,
                                     sku=str(getattr(entry, "gpu_name", "") or ""))
            with self._lock:
                rolled = (getattr(era, "_superseded", False) or (
                    self.state.current is not None and era.index < self.state.current.index))
            if rolled:
                # Its era has no settlement left: the leg can never be judged
                # (a later era trains from a different init). Tell the miner NOW,
                # not at the next rollover, and never record an orphan.
                log.warning("rolling: %s's leg finished after era %d ended — requeued "
                            "unburned at finish (era now %d)", gen.hotkey[:12], era.index,
                            getattr(self.state.current, "index", era.index))
                try:
                    self.ops.teardown_kept_pod(gen.hotkey)
                except Exception as e:  # noqa: BLE001
                    log.error("rolling: kept pod teardown for %s failed: %s", gen.hotkey[:12], e)
                if queue is not None:
                    queue.requeue(gen.hotkey, error="leg finished after its era ended",
                                  error_class="no_capacity", burn_attempt=False,
                                  operator_billed=True)
                return
            # Record the leg the moment TRAINING ends: the bench below is
            # telemetry (an hour and more on a payer pod) and must never hold
            # the settlement: a bench that outlived the era would strand a
            # trained leg and the miner would pay for a second one. Late
            # numbers join the settlement's bench report instead.
            with self._lock:
                rolled = (getattr(era, "_superseded", False) or (self.state.current is not None
                          and era.index < self.state.current.index))
                if not rolled:
                    self.state.finished.append(FinishedLeg(
                        hotkey=gen.hotkey, uid=gen.uid, ref=gen.ref, era_index=era.index,
                        started_block=block, reveal_block=gen.reveal_block,
                        entry=_entry_to_json(entry), bench=None, label=label))
                    self._save()
            if rolled:
                log.warning("rolling: %s's leg finished after era %d ended — requeued "
                            "unburned at finish (era now %d)", gen.hotkey[:12], era.index,
                            getattr(self.state.current, "index", era.index))
                try:
                    self.ops.teardown_kept_pod(gen.hotkey)
                except Exception as e:  # noqa: BLE001
                    log.error("rolling: kept pod teardown for %s failed: %s", gen.hotkey[:12], e)
                if queue is not None:
                    queue.requeue(gen.hotkey, error="leg finished after its era ended",
                                  error_class="no_capacity", burn_attempt=False,
                                  operator_billed=True)
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
                            self.ops.burn([gen], block)
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
            # never goes back (a flip would restart the king leg and throw
            # away the successor's). ``_forfeit_switch`` keeps the era on the
            # successor.
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
                cur.king_funded_by = ""
            else:
                entry = replace(_entry_from_json(winner.entry), role="king")
                cur.king_hotkey, cur.king_uid, cur.king_ref = king, winner.uid, winner.ref
                cur.king_entry = _entry_to_json(entry)
                cur.king_init = cur.warm_start_ckpt     # a settled leg of THIS era
                cur.king_bench = winner.bench
                cur.king_bench_published = winner.bench is not None
                # The dethroner paid for its own winning leg (DEC-CA-0055 item 6):
                # in a seed epoch that checkpoint IS the king leg.
                cur.king_funded_by = "king" if cur.epoch_start >= 0 else ""
                log.info("rolling: DETHRONE — %s's checkpoint %s adopted as era %d king "
                         "(era continues)", king[:12], entry.trained_pointer, cur.index)
            if self.state.next is not None and self.state.next.king_hotkey != king:
                if not self._same_epoch(self.state.next, cur):   # a shared epoch pod stays
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
            if self._settle_boundary(client, b, created_block=(b if b < int(epoch_start)
                                                               else int(block)),
                                     now_block=block) is False:
                return False       # waits (lane B still settling): later boundaries too
        return True

    def _late_under_forfeited_king(self, cur: EraState, epoch_start: int,
                                   now_block: int | None) -> bool:
        """DEC-CA-0048 on the block clock: a boundary of an era whose king is
        forfeited by ``now_block`` but NOT at the era's own gate (a late
        settlement of an era that ended before the forfeiture) can never be
        judged — the validators crowned the successor at the gate and reject
        a manifest carrying the old king whole (``king_resyncing``), after the
        trainer would have marked its legs done and burned. Publish nothing;
        requeue
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
                              burn_attempt=False, operator_billed=True)
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
        pend = self.state.pending_publish
        if pend is not None:
            # A publish that started before a crash/failure: republish EXACTLY
            # what may already be out (same round id, same bytes), then commit.
            # Nothing is re-derived — more legs may have finished since.
            log.warning("rolling: boundary %d — settlement %s (boundary %s) was being "
                        "published; republishing the same manifest", epoch_start,
                        pend.get("round_id"), pend.get("boundary"))
            self._publish_settlement(cur, pend, replay=True)
            if int(pend.get("boundary", 0) or 0) >= int(epoch_start):
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
        if self.state.current is not cur:
            return  # recovered evidence invalidated this era's prepared init
        if settlement_era(self.cfg.round, epoch_start).index != cur.index:
            # A boundary before this era's first settlement (the rollover
            # boundary itself): nothing of ours to settle there.
            self.state.last_settled_boundary = epoch_start
            self._save()
            return
        self._drop_foreign_king_entry(cur)
        with self._lock:
            ready = [f for f in self.state.finished if f.era_index == cur.index]
            ready = self._hold_unbenched(ready, epoch_start, cur)
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
            if not ready and not self._needs_settlement_progress(epoch_start):
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
                                  error_class="infra", burn_attempt=False, operator_billed=True)
            with self._lock:
                self.state.finished = [f for f in self.state.finished if f not in bad]
                ready = [f for f in ready if f not in bad]
                if not ready and not self._needs_settlement_progress(epoch_start):
                    self.state.last_settled_boundary = epoch_start
                self._save()
            if not ready and not self._needs_settlement_progress(epoch_start):
                return
        if not self._seed_chain_root():
            # An unchained manifest is rejected by every validator AFTER the
            # legs are marked done and burned — never publish one. The legs
            # wait for the next boundary; the root is retried every tick.
            log.error("rolling: boundary %d — manifest chain root unknown (latest.json "
                      "unreadable); %d finished leg(s) wait, nothing published",
                      epoch_start, len(ready))
            return
        # Lane B FIRST (DEC-CA-0050): the expert manifest is published and its
        # digest signed into the lane-A manifest, so a validator polling
        # between the two publishes can never handle this settlement without
        # its lane-B legs (the merge the next era init declares).
        expert_sha = self._lane_b_settle(round_id, epoch_start, cur, king)
        if expert_sha is None:
            return False                     # lane B still settling: this boundary waits
        # Lane C next (DEC-CA-0056), the same binding: its manifest is out and
        # its sha256 signed into the lane-A manifest before that publishes.
        lane_c_sha, lane_c_pub = self._lane_c_settle(round_id, epoch_start, created_block,
                                                     cur, king)
        from ..shared.manifest import dump_manifest

        # DEC-CA-0063: record the main validator's bench reports of earlier
        # validator-sourced settlements (sha256, or "absent" past the timeout)
        vb_texts: dict[str, str] = {}
        vb_records, vb_pending = self._vbench_records(round_id, epoch_start, texts=vb_texts)
        with self._lock:
            # DEC-CA-0063 item 9: legs a report recorded HERE budget-deferred
            # ride this settlement again, including a progress-only settlement.
            carried = self._vbench_carry(cur, ready, vb_texts) if vb_texts else []
            manifest = self._build_manifest(cur, king, ready, round_id, created_block,
                                            pin_block=epoch_start,
                                            expert_manifest_sha256=expert_sha,
                                            lane_c_digest=lane_c_sha,
                                            validator_bench_reports=vb_records,
                                            carried=carried)
        bench_entries = self._bench_entries(cur, king, ready)
        init_bench = self._init_bench_entry(cur)
        # Signed ONCE; the exact text is written ahead of the first publish
        # (see RollingState.pending_publish).
        manifest = self.ops.sign_manifest(manifest)
        pend = {
            "boundary": int(epoch_start), "round_id": round_id,
            "created_block": int(created_block), "manifest": dump_manifest(manifest),
            "pointers": sorted(_entry_from_json(f.entry).trained_pointer for f in ready),
            # Under the three-suite bench gate (DEC-CA-0054) the rows validators
            # pair are in the bench report, so it goes out BEFORE the manifest:
            # a validator that sees the settlement always sees its report, and
            # a failed bench publish withholds the settlement (retried next
            # tick) instead of leaving every validator to judge it without rows.
            # DEC-CA-0063: a validator-sourced settlement's gate rows are the
            # main validator's, so nothing in the trainer report gates it.
            "bench_first": (bench_gate_mode_at(self.cfg, epoch_start) != "off"
                            and not validator_bench_at(self.cfg, epoch_start)),
            "bench": ({"pairs": [[_entry_to_json(e), dataclasses.asdict(b)]
                                 for e, b in bench_entries],
                       **({"init": init_bench} if init_bench else {})}
                      if (bench_entries or init_bench) else None),
        }
        if lane_c_pub:
            # the lane-C manifest this one binds: a replay republishes it first
            pend["lane_c"] = dict(lane_c_pub)
        if vb_pending is not None:
            # committed with the boundary (the records ride the signed text)
            pend["vbench_pending"] = dict(vb_pending)
        if carried:
            pend["carried"] = sorted(_entry_from_json(f.entry).trained_pointer for f in carried)
        with self._lock:
            self.state.pending_publish = pend
            self._save()
        self._publish_settlement(cur, pend, manifest=manifest)

    def _publish_settlement(self, cur: EraState, pend: dict, *, manifest=None,
                            replay: bool = False) -> None:
        """Publish the written-ahead settlement ``pend`` (bench report and
        manifest, in its order), then commit the boundary and clear the
        write-ahead in one save. A replay republishes the persisted text and
        the persisted report pairs (rows limited to the pointers it carries);
        a failure anywhere leaves ``pending_publish`` for the next tick."""
        from ..shared.manifest import BenchScores, dump_manifest, load_manifest

        round_id = str(pend["round_id"])
        epoch_start = int(pend["boundary"])
        created_block = int(pend["created_block"])
        text = str(pend["manifest"])
        if manifest is None:
            manifest = load_manifest(text)
        published_text = dump_manifest(manifest)
        if published_text != text:
            log.error("rolling: settlement %s's written-ahead manifest does not round-trip "
                      "byte-exactly — republishing the re-rendered text", round_id)
        pointers = set(pend.get("pointers") or ())
        carried_ptrs = set(pend.get("carried") or ())
        with self._lock:
            ready = [f for f in self.state.finished
                     if _entry_from_json(f.entry).trained_pointer in pointers]
            carried = [f for f in self.state.published
                       if _entry_from_json(f.entry).trained_pointer in carried_ptrs]
        bench = pend.get("bench") or {}
        bench_entries = [(_entry_from_json(e), BenchScores(**b))
                         for e, b in bench.get("pairs") or ()]
        init_bench = bench.get("init") or None
        kw = ({"rows_for": frozenset(e.trained_pointer for e, _ in bench_entries)}
              if replay else {})
        if init_bench:
            kw["init"] = init_bench
        report = None
        lane_c = pend.get("lane_c") or None
        if replay and lane_c:
            # DEC-CA-0056: the identical lane-C bytes this manifest binds go out
            # (again) BEFORE it — a crash between the two never leaves the
            # binding manifest without its lane-C manifest
            self.ops.publish_text(str(lane_c["key"]), str(lane_c["text"]))
        bench_first = bool(pend.get("bench_first"))
        if bench_first and (bench_entries or init_bench):
            report = self.ops.publish_bench(round_id, created_block, bench_entries, **kw)
        self.ops.publish_manifest(manifest)
        if not bench_first and (bench_entries or init_bench):
            report = self.ops.publish_bench(round_id, created_block, bench_entries, **kw)
        if report is not None:
            cur.king_bench_published = cur.king_bench_published or cur.king_bench is not None
        self._record_bench_report(round_id, created_block, bench_entries, ready,
                                  str(getattr(manifest, "warm_start_ckpt", "") or ""),
                                  init=init_bench)
        self.ops.record_bench_candidates(manifest, report)
        queue = self.ops.queue()
        for f in ready:
            if queue is not None:
                queue.mark_done(f.hotkey, expect_ref=f.ref,
                                expect_started_block=f.started_block)
            # The burn lands in the round the leg STARTED in: the hotkey may
            # fund again from the next round on.
            self.ops.burn([ResolvedGen(f.hotkey, f.uid, f.ref, f.reveal_block)],
                          int(f.started_block))
        with self._lock:
            self.state.finished = [f for f in self.state.finished if f not in ready]
            self.state.published.extend(ready)
            self.state.last_published_round_id = round_id
            self.state.last_published_manifest_sha256 = hashlib.sha256(
                published_text.encode("utf-8")).hexdigest()
            self.state.last_settled_boundary = max(int(self.state.last_settled_boundary),
                                                   epoch_start)
            self.state.pending_publish = None
            if "vbench_pending" in pend:
                self.state.vbench_pending = {str(k): int(v)
                                             for k, v in pend["vbench_pending"].items()}
            for p in sorted(carried_ptrs):
                # already marked done / burned / published at their own
                # settlement: only the carry count moves
                self.state.vbench_carries[p] = int(self.state.vbench_carries.get(p, 0)) + 1
            roster = self._roster(round_id, ready, queue, carried=carried)
            self.state.rents = []
            self._save()
        self.ops.publish_roster(round_id, roster)
        try:
            self.ops.publish_champion(ResolvedGen(cur.king_hotkey, cur.king_uid, cur.king_ref),
                                      round_id)
        except Exception as e:  # noqa: BLE001
            log.warning("rolling: champion publish failed (ignored): %s", e)
        log.info("rolling: SETTLEMENT %s at boundary %d — era %d king %s + %d challenger(s)%s",
                 round_id, epoch_start, cur.index, cur.king_hotkey[:12], len(ready),
                 f" + {len(carried)} carried (budget-deferred)" if carried else "")

    def _needs_settlement_progress(self, boundary: int) -> bool:
        """Existing king-only envelopes advance auxiliary lanes and arbiters.

        Once a lane or validator-sourced benches are active, a quiet lane A
        must not stop reports, confirmations, acceptance or reward clocks.
        """
        return (self.cfg.lane_b.active(boundary)
                or self.cfg.lane_c.active(boundary)
                or validator_bench_at(self.cfg, boundary)
                or bool(self.state.vbench_pending))

    def _build_manifest(self, era: EraState, king: TrainedEntry, ready: list[FinishedLeg],
                        round_id: str, block: int, *, pin_block: int,
                        expert_manifest_sha256: str = "", lane_c_digest: str = "",
                        validator_bench_reports: dict | None = None,
                        carried: list[FinishedLeg] | tuple = ()):
        from ..shared.manifest import TrainingManifest, contract_digest, contract_payload

        ordered = sorted([*ready, *carried], key=lambda f: (f.reveal_block, f.hotkey))
        # DEC-CA-0055 item 6: the king entry names who paid for its leg (v2
        # eras only; "" everywhere else, dropped from the signed bytes).
        # every ladder-era king entry names its payer; unknown ⇒ the operator
        # unknown (a lost completed-leg record) is never a slash: only a key
        # the trainer SAW fail is "operator"
        funded = ((era.king_funded_by or "operator_fault") if self._ladder_era(era) else "")
        entries = [_with_funded_by(king, funded)]
        for i, f in enumerate(ordered):
            entries.append(_with_funded_by(
                replace(_entry_from_json(f.entry), role="challenger",
                        duel_rank=(i if len(ordered) > 1 else 0)), ""))
        contract = self.cfg.training.at_block(int(era.start_block))
        if v2_era_active(self.cfg, era.start_block):
            from ..shared.hippius import StorageError

            for entry in entries:
                if (entry.init_pointer != era.warm_start_ckpt
                        or entry.training_contract_digest != contract_digest(contract)):
                    raise StorageError(
                        "refusing v2 manifest: entry lacks the scheduled init/contract identity; "
                        "check worker release and requeue incompatible work")
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
            base_arch_digest=self.cfg.training.at_block(int(era.start_block)).base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset, entries=entries,
            eval_pool_key=str(pool_key or ""), eval_pool_sha256=str(pool_sha or ""),
            warm_start_ckpt=era.warm_start_ckpt, warm_start_size=era.warm_start_size,
            warm_start_base=era.warm_start_base if era.warm_start_merges else "",
            warm_start_merges=tuple(era.warm_start_merges),
            expert_manifest_sha256=str(expert_manifest_sha256 or ""),
            # DEC-CA-0056: passed only when set (a build without the field, and
            # every pre-lane-C manifest, is byte-identical)
            **({"lane_c_digest": str(lane_c_digest)} if lane_c_digest else {}),
            # DEC-CA-0063: passed only when something is recorded (every other
            # manifest is byte-identical)
            **({"validator_bench_reports": dict(validator_bench_reports)}
               if validator_bench_reports else {}),
            contract_body=contract_payload(self.cfg.training.at_block(int(era.start_block))),
            promotion_notices=(self.r._promotion_notices(int(block))
                               if hasattr(self.r, "_promotion_notices") else None),
            era=era.spec().to_json(),
            prev_round_id=self.state.last_published_round_id,
            # the hash link to that manifest's exact published bytes ("" when
            # unknown — a seeded chain root or a pre-upgrade state file).
            # Rollout-gated: validators must run code that knows the field
            # before it is signed into a manifest (``[manifest]
            # hash_link_from_block``, 0 = never emitted; the sha is tracked
            # regardless).
            prev_manifest_sha256=(self.state.last_published_manifest_sha256
                                  if self._hash_link_active(pin_block) else ""),
        )

    def _vbench_carry(self, cur: EraState, ready: list[FinishedLeg],
                      texts: dict[str, str]) -> list[FinishedLeg]:
        """DEC-CA-0063 item 9: the settled legs of era ``cur`` that a report
        recorded by THIS manifest (``texts`` = round id → the exact text
        recorded) lists ``budget_deferred`` — carried into this settlement so
        every validator judges them again (private pool and bench gate), at
        most :data:`~cascade.shared.bench_report.VBENCH_MAX_CARRIES` times
        each. Only the trainer's own published legs of this era can ride (a
        listed pointer that is not one is ignored), never one whose hotkey
        has a fresh leg here or is the king's. Caller holds the lock."""
        from ..shared.bench_report import VBENCH_MAX_CARRIES, load_bench_report

        by_ptr = {_entry_from_json(f.entry).trained_pointer: f for f in self.state.published
                  if int(f.era_index) == int(cur.index)}
        taken = {f.hotkey for f in ready} | {str(cur.king_hotkey or "")}
        out: list[FinishedLeg] = []
        for rid in sorted(texts, key=lambda r: int(r) if str(r).isdigit() else 0):
            try:
                report = load_bench_report(texts[rid])
            except (ValueError, KeyError, TypeError):
                continue
            for d in report.budget_deferred:
                ptr = str(d.get("trained_pointer", ""))
                f = by_ptr.get(ptr)
                if f is None or f.hotkey in taken:
                    continue
                n = int(self.state.vbench_carries.get(ptr, 0))
                if n >= VBENCH_MAX_CARRIES:
                    log.warning("rolling: %s's leg was budget-deferred again (report %s) "
                                "after %d carr%s — not carried (skipped, never a loss)",
                                f.hotkey[:12], rid, n, "y" if n == 1 else "ies")
                    continue
                log.info("rolling: carrying %s's budget-deferred leg (report %s, carry %d/%d)",
                         f.hotkey[:12], rid, n + 1, VBENCH_MAX_CARRIES)
                out.append(f)
                taken.add(f.hotkey)
        return out

    def _vbench_records(self, round_id: str, epoch_start: int, *,
                        texts: dict | None = None) -> tuple[dict, dict | None]:
        """DEC-CA-0063 arbiter: ``(records, pending after)`` for the manifest
        of settlement ``round_id``. For every earlier validator-sourced
        settlement still pending, read the main validator's
        ``bench_v/round-<id>.json``: present ⇒ record its sha256; absent ⇒
        count one more settlement, and record "absent" once ``[v2]
        bench_report_timeout_settlements`` are reached; a READ failure
        (store down) neither counts nor records. This settlement itself is
        added when its era takes its lane-A rows from the validator.
        ``texts`` (optional) receives round id → the exact text recorded.
        ``(_, None)`` = nothing to track (the state is left untouched)."""
        from ..shared.bench_report import validator_bench_report_key
        from ..shared.hippius import ObjectNotFound

        with self._lock:
            pending = dict(self.state.vbench_pending)
        own = validator_bench_at(self.cfg, int(epoch_start))
        if not pending and not own:
            return {}, None
        timeout = max(1, int(getattr(self.cfg.v2, "bench_report_timeout_settlements", 8)))
        records: dict[str, str] = {}
        after: dict[str, int] = {}
        reader = getattr(self.ops, "read_text", None)
        for rid, waited in sorted(pending.items(), key=lambda kv: int(kv[0])):
            if str(rid) == str(round_id):
                continue
            try:
                if reader is None:
                    raise ObjectNotFound(rid)
                text = reader(validator_bench_report_key(str(rid)))
            except ObjectNotFound:
                if int(waited) + 1 >= timeout:
                    records[str(rid)] = "absent"
                    log.warning("rolling: settlement %s's validator bench report is still "
                                "absent after %d settlement(s) — recorded ABSENT (its "
                                "clearers are infra)", rid, int(waited) + 1)
                else:
                    after[str(rid)] = int(waited) + 1
                continue
            except Exception as e:  # noqa: BLE001 — store down: retried next settlement
                log.warning("rolling: validator bench report of %s unreadable (%s); not "
                            "counted", rid, e)
                after[str(rid)] = int(waited)
                continue
            records[str(rid)] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if texts is not None:
                texts[str(rid)] = text
        if own:
            after[str(round_id)] = 0
        if records:
            log.info("rolling: settlement %s records %d validator bench report(s): %s",
                     round_id, len(records),
                     ", ".join(f"{k}={v[:12]}" for k, v in sorted(records.items())))
        return records, after

    def _hash_link_active(self, boundary: int) -> bool:
        gate = int(getattr(getattr(self.cfg, "manifest", None), "hash_link_from_block", 0)
                   or 0)
        v2 = getattr(self.cfg, "v2", None)
        return (gate > 0 and int(boundary) >= gate) or bool(v2 and v2.active(int(boundary)))

    # ── deferred bench verification ──────────────────────────────────────────
    # LEGACY-ONLY for lane A (trainer bench source): from ``[v2]
    # bench_source_from_block`` the trainer runs no challenger sweep, so no
    # payer-pod number is queued for verification (DEC-CA-0063 item 7).

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
                             ready: list[FinishedLeg], warm_start_ckpt: str = "",
                             *, init: dict | None = None) -> None:
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
                # The pointers whose gate rows the FIRST publish carried: a
                # republish never adds rows for anyone else (DEC-CA-0054), so a
                # validator judging either version reaches the same verdict.
                "rows_pointers": sorted(e.trained_pointer for e, _ in bench_entries),
                # DEC-CA-0055: the "init" entry this report carried (a republish
                # keeps it)
                **({"init": dict(init)} if init else {}),
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
                str(rep.get("warm_start_ckpt", "") or ""),
                frozenset(rep.get("rows_pointers", ())))

    def _republish(self, pointer: str, rid: str, cblock: int, pairs: list,
                   warm_start_ckpt: str, rows_for: frozenset = frozenset()) -> None:
        from types import SimpleNamespace

        from ..shared.manifest import BenchScores

        typed = [(_entry_from_json(e), BenchScores(**b)) for e, b in pairs]
        with self._lock:            # the DEC-CA-0055 init entry the report carried
            init = (self.state.bench_reports.get(str(rid)) or {}).get("init") or None
        try:
            # A late entry rides for promotion only: its gate rows are never
            # added after the settlement was published (validators that judged
            # the first version saw none).
            report = self.ops.publish_bench(rid, cblock, typed, rows_for=rows_for,
                                            **({"init": init} if init else {}))
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

    def _hold_unbenched(self, ready: list[FinishedLeg], epoch_start: int,
                        era: EraState | None = None) -> list[FinishedLeg]:
        """LEGACY-ONLY (trainer bench source): the lane-A bench hold.
        With the three-suite bench gate on (DEC-CA-0054) a leg settles only
        once its operator-verified bench has landed, so validators can pair it
        with the king — except at the era's LAST settlement, where everything
        settles (a leg still unbenched is then uncomputable for the gate and
        cannot be crowned under ``enforce``). Gate off ⇒ unchanged."""
        if bench_gate_mode_at(self.cfg, epoch_start) == "off":
            return ready
        if validator_bench_at(self.cfg, epoch_start):
            return ready         # DEC-CA-0063: the main validator benches winners itself
        if epoch_start == next_era_start(self.cfg.round, max(0, epoch_start - 1)):
            return ready
        if era is not None and era.king_entry is not None and era.king_bench is None:
            # Every challenger pairs with the king's rows: without them the
            # whole settlement would be uncomputable. Wait for the king's bench
            # (re-tried by _maybe_rebench_king), same last-settlement rule.
            if ready:
                log.info("rolling: boundary %d — %d leg(s) wait for the era king's bench "
                         "(bench gate on)", epoch_start, len(ready))
            return []
        held = [f for f in ready if f.bench is None]
        if held:
            log.info("rolling: boundary %d — %d leg(s) wait for their bench before "
                     "settling (bench gate on): %s", epoch_start, len(held),
                     ", ".join(f.hotkey[:12] for f in held))
        return [f for f in ready if f.bench is not None]

    def _bench_entries(self, era: EraState, king: TrainedEntry, ready: list[FinishedLeg]) -> list:
        """``(entry, BenchScores)`` pairs for the settlement's bench report:
        the king's numbers once per era (its first report), every benched
        challenger every time."""
        from ..shared.manifest import BenchScores

        out = []
        # The three-suite bench gate (DEC-CA-0054) pairs every challenger with
        # the king inside ONE signed report, so the king rides every report
        # while the gate is on (record_bench is idempotent by pointer).
        gate_on = bench_gate_configured(self.cfg)
        # DEC-CA-0055: the king check reads the king's AND the init's numbers
        # from the previous settlement's report — the king rides every one
        every = gate_on or self._ladder_era(era)
        if era.king_bench is not None and (every or not era.king_bench_published):
            out.append((king, BenchScores(**era.king_bench)))
        for f in ready:
            if f.bench is not None:
                out.append((_entry_from_json(f.entry), BenchScores(**f.bench)))
        return out

    def _init_bench_entry(self, era: EraState) -> dict | None:
        """The "init" bench-report entry for a v2 settlement of ``era``
        (DEC-CA-0055): ``{"pointer", "size", "scores"}`` — the init's
        checkpoint pointer ("" = the epoch's random init) and its BenchScores
        — once the epoch's init bench has landed; it rides EVERY report of the
        epoch so the king check finds it next to the king's numbers. None
        pre-v2 or before the bench lands."""
        if int(era.epoch_start) < 0 or era.init_bench is None:
            return None
        if not self._ladder_era(era):
            return None
        pointer = str(era.warm_start_ckpt or era.init_pointer or "")
        if not pointer:
            return None          # validators score the init: it must be fetchable
        size = era.warm_start_size or self.cfg.throne_contracts_at(
            int(era.start_block))[0].arch_preset
        return {"pointer": pointer, "size": str(size or ""), "scores": dict(era.init_bench)}

    def _roster(self, round_id: str, ready: list[FinishedLeg], queue, *,
                carried: list[FinishedLeg] | tuple = ()) -> dict:
        from ..funding.queue import select_field

        entries = queue.entries() if queue is not None else []
        seated = [{"hotkey": f.hotkey, "ref": f.ref, "reveal_block": f.reveal_block,
                   "label": f.label, "started_block": f.started_block}
                  for f in sorted(ready, key=lambda f: (f.reveal_block, f.hotkey))]
        # DEC-CA-0063 item 9: a carried (budget-deferred) leg sits in the
        # manifest again; it was seated at its own settlement
        seated += [{"hotkey": f.hotkey, "ref": f.ref, "reveal_block": f.reveal_block,
                    "label": f.label, "started_block": f.started_block, "carried": True}
                   for f in sorted(carried, key=lambda f: (f.reveal_block, f.hotkey))]
        return {
            "round_id": round_id, "mode": "rolling",
            "seated": seated,
            "in_flight": [{"hotkey": e.hotkey, "reveal_block": e.reveal_block,
                           "target_boundary": e.target_boundary, "era_index": e.era_index,
                           "label": e.label, "gpu": getattr(e, "gpu", "")}
                          for e in entries if e.status == "in_flight"],
            "waiting": [{"hotkey": e.hotkey, "reveal_block": e.reveal_block, "label": e.label,
                         "gpu": getattr(e, "gpu", "")}
                        for e in select_field(entries, cap=0)],
            "terminal": [{"hotkey": e.hotkey, "error_class": e.last_error_class}
                         for e in entries if e.status == "failed"],
            "rents": list(self.state.rents),
        }
