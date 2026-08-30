"""Trainer service loop — the owner-operated training round.

Each round the trainer:

1. Resolves on-chain generator commitments to ``(hotkey, uid, ref)``.
2. Identifies the reigning king (the highest-incentive UID on the metagraph in
   live mode; a caller-supplied hotkey offline) and selects challengers.
3. For the king and each challenger, under one shared :class:`RoundSeeds`:
   fetches the generator from the Hippius Hub registry by ref, builds the corpus,
   trains a fresh base model via the owner's :class:`BaseTrainer` (streaming
   per-step metrics to Hippius S3), and uploads the checkpoint to the registry.
4. Assembles a :class:`TrainingManifest`, signs it with the trainer hotkey, and
   (live) publishes it to the Hippius S3 manifest bucket for validators.

The pure planning + assembly logic is testable without GPUs, a chain, or
Hippius; the GPU / registry / S3 / chain calls are isolated in
:meth:`TrainerRunner.train_one`, :meth:`TrainerRunner.publish`, and the live
:meth:`TrainerRunner.run_forever`.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

from ..interface.validation import check_repo_size, parse_commit
from ..shared.chain import Commitment
from ..shared.config import ChainConfig, TrainingContractConfig, effective_epoch_blocks
from ..shared.hippius import (
    HubConfig,
    LogSink,
    ObjectNotFound,
    S3Config,
    S3Store,
    StorageError,
    fetch_from_hub,
    generator_archive_key,
    manifest_round_key,
    publish_manifest,
    upload_dir_to_hub_or_hf,
)
from ..shared.manifest import (
    BenchScores,
    HeatEntrant,
    HeatResult,
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    contract_payload,
    dump_manifest,
    format_trained_pointer,
    parse_trained_pointer,
    sign_manifest,
)
from .contract import BaseTrainer, RoundSeeds, TrainResult, assert_train_image
from .corpus import CorpusError, build_round_corpus
from .host_probe import host_snapshot, host_summary_line
from .stream import open_round_stream
from .wandb_sink import open_wandb_run

if TYPE_CHECKING:  # keeps the eval stack out of the trainer's import graph
    from ..eval.scoring import WindowScore

# Screens one heat checkpoint: given the trained heat-model directory, the
# generator that produced its corpus, the round's base seed (so the screening
# window slice can rotate per round), and the round's epoch-boundary block (so a
# daily-snapshot pool selects the SAME snapshot the validator will judge the
# final on — not whatever is newest), return a heat score (LOWER is better, e.g.
# geomean(CRPS, MASE) on the held-out windows). Injected so the trainer's
# screening stays a testable boundary — the default wiring (torch evaluator +
# eval pool) is attached in cascade.trainer.main.
#
# Returning the per-window ``list[WindowScore]`` instead of the scalar is
# preferred: the ranking is identical (the runner reduces with global_geomean),
# and the per-window scores additionally (a) feed the shadow selection
# diagnostics in cascade.eval.heat — P(best) per entrant and the
# leader-vs-runner-up LCB, which say whether the screen was decisive — and
# (b) yield the raw CRPS/MASE components, global_components(scores), published
# on the heat standings for miner transparency. A scalar-only screener still
# works; the round then carries neither diagnostics nor per-entrant components.
ScreenFn = Callable[
    [Path, "ResolvedGenerator", int, int | None], "float | list[WindowScore]"
]

# Scores the INCREMENTAL tie run-off windows for one heat checkpoint
# (DEC-CA-0012): windows ``[start, stop)`` of the round's seeded window
# permutation, on CPU, against the heat checkpoint still on local disk. The
# round's window selection is a seeded permutation PREFIX, so the heat's slice
# is a strict prefix of the run-off's larger slice for the same round seed —
# the caller concatenates the returned scores onto the heat's and pairing
# holds by construction (``joint_bag_geomeans`` raises on mismatched
# ``abs_target``). Args mirror ScreenFn plus ``(start, stop)``. Injected so
# the run-off stays a testable boundary; the default wiring (same pool source
# as the screener) is attached in cascade.trainer.main.
RunoffFn = Callable[
    [Path, "ResolvedGenerator", int, int | None, int, int], "list[WindowScore]"
]

# Scores one final-duel checkpoint on the public suites (GIFT-Eval / BOOM /
# TIME) for Cascade, given its local checkpoint dir. Returns the six-number
# BenchScores the trainer publishes in the round's signed bench report
# (cascade.shared.bench_report) AFTER the manifest is out, or None when the
# sidecar could not produce a complete set (best-effort — a miss just leaves that
# checkpoint out of the report). Injected so the trainer's Cascade eval stays a
# testable boundary; the default wiring (fetch + benchmark sidecar) is attached in
# cascade.trainer.main.
BenchEvalFn = Callable[[Path], "BenchScores | None"]

log = logging.getLogger("cascade.trainer")

# Corpus-seed salt for the bench-anneal leg (DEC-CA-0030): the anneal resumes
# a canonical checkpoint on FRESH data (base_seed ^ salt) so the decay pass
# never re-fits the round's exact training draw, and the salted seed keys the
# leg's _train_work dir + checkpoint repo away from every canonical name.
# Value matches the 2026-08-23 offline calibration runs, so their published
# -anneal-u<uid> artifacts stay reproducible under the production leg.
BENCH_ANNEAL_SALT = 0xA11EA1A11EA1

# Dedup probe stage: concurrent sandboxes, and the floor under the per-draw
# wall clock derived from [round] dedup_probe_budget_seconds (a budget so tight
# that no generator could start would fail the whole field, not screen it).
_PROBE_WORKERS = 4
_PROBE_MIN_DRAW_SECONDS = 30
# CorpusError messages that pin the fault on the MINER's generator. Anything
# else — sandbox_crashed, sandbox_isolation_unavailable, missing container
# runtime/image, digest transit mismatch — is our infrastructure, and under
# dedup_probe_mode = "enforce" a misattributed infra fault burns the entire
# field (probe drops burn hotkeys). Unknown messages therefore fail OPEN.
_PROBE_MINER_FAULT_PREFIXES = (
    "generator is non-deterministic",   # _probe_digest's own verdict
    "generator_",       # timeout / stalled / import / construct / output_rejected …
    "missing generator.py",
    "submission_too_large",
    "repo_layout",
    "repo_too_large",
    "blocked_import",
)


def _http_status_in_chain(exc: BaseException | None) -> int | None:
    """First HTTP status code found walking an exception's cause chain."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if isinstance(status, int):
            return status
        exc = exc.__cause__ or exc.__context__
    return None


def _pctl(vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of a non-empty list (pure; no numpy here)."""
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def host_spread_part(entries: list[dict]) -> str:
    """Fleet-uniformity clause for the round roll-up, from host-probe facts.

    The operator's one-line answer to "was this round's field screened on one
    class of hardware?". ``spread`` is max/min of the calibration bench — the
    number to watch, because it bounds how much of a round's throughput dispersion
    the pods could account for on their own; at 1.0x the fleet is uniform and any
    remaining spread is the generators. ``pods``/``machines`` count distinct
    ``host_id``/``host_boot_id``: fewer machines than pods means co-tenancy, which
    is a slow-run explanation no per-pod field can show.

    Empty string when no run reported a bench, so the roll-up degrades to exactly
    its previous text rather than printing a hollow clause.
    """
    benches = [
        float(e["host_bench_tokens_per_s"]) for e in entries
        if isinstance(e, dict) and "host_bench_tokens_per_s" in e
    ]
    if not benches:
        return ""
    lo, hi = min(benches), max(benches)
    pods = {e.get("host_id") for e in entries if isinstance(e, dict) and e.get("host_id")}
    machines = {
        e.get("host_boot_id") for e in entries
        if isinstance(e, dict) and e.get("host_boot_id")
    }
    skus = {
        e.get("host_gpu_name") for e in entries
        if isinstance(e, dict) and e.get("host_gpu_name")
    }
    part = (
        f"host_bench min={lo:,.0f} p50={_pctl(benches, 0.5):,.0f} max={hi:,.0f} "
        f"spread={hi / max(lo, 1e-9):.2f}x ({len(benches)} benched, "
        f"{len(pods)} pods"
    )
    if machines:
        part += f", {len(machines)} machines"
    if skus:
        part += f", {len(skus)} skus"
    return part + ")"


def telemetry_rollup_line(
    round_id: int | str, heat_metrics: list[dict], final_metrics: list[dict]
) -> str:
    """One-line per-round starvation/deadline/host roll-up from run metrics dicts.

    Pure formatting so the aggregation is unit-testable. Entries without the
    telemetry keys are skipped (a custom BaseTrainer may not emit them; remote
    runs keep their metrics on the pod — see ``_train_checkpoint``), and the
    trailing count says how many runs actually reported, so a silent-majority
    round can't masquerade as a healthy one.

    The host clause (:func:`host_spread_part`) is filtered separately: host facts
    come from the pod probe, not the backend's metrics, so a run can carry them
    while lacking ``deadline_hit`` — and a fleet-uniformity number computed over
    only the metrics-reporting subset would be the wrong denominator.
    """
    heats = [m for m in heat_metrics if isinstance(m, dict) and "deadline_hit" in m]
    finals = [m for m in final_metrics if isinstance(m, dict) and "deadline_hit" in m]
    waits = [float(m["data_wait_frac"]) for m in (*heats, *finals) if "data_wait_frac" in m]
    wait_part = (
        f"data_wait_frac p50={_pctl(waits, 0.5):.3f} p95={_pctl(waits, 0.95):.3f}"
        if waits else "data_wait_frac n/a"
    )
    hit_h = sum(bool(m.get("deadline_hit")) for m in heats)
    hit_f = sum(bool(m.get("deadline_hit")) for m in finals)
    reported = len(heats) + len(finals)
    total = len(heat_metrics) + len(final_metrics)
    host_part = host_spread_part([*heat_metrics, *final_metrics])
    return (
        f"round={round_id} telemetry: deadline_hit {hit_h}/{len(heats)} heats + "
        f"{hit_f}/{len(finals)} finals; {wait_part} ({reported}/{total} runs "
        "reported metrics)" + (f"; {host_part}" if host_part else "")
    )


def _load_seen_hotkeys(path: Path) -> set[str]:
    """Load the persisted 1-hotkey-1-submission burn set (best-effort)."""
    try:
        return {str(h) for h in json.loads(path.read_text(encoding="utf-8"))}
    except FileNotFoundError:
        return set()
    except Exception as e:  # noqa: BLE001
        log.warning("submissions db %s unreadable (%s); starting from empty", path, e)
        return set()


def _save_seen_hotkeys(path: Path, seen: set[str]) -> None:
    """Persist the burn set (best-effort — anti-spam must never abort a round)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(seen)), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("could not persist submissions db to %s: %s", path, e)


# Storage-layer failure fingerprints. A challenger whose run died inside the
# artifact/checkpoint fetch path (registry 401/403, Hippius chunk errors)
# failed at OUR storage boundary, not in its own generator (2026-08-11: a
# ~1-minute registry auth blip 401'd two fetchable artifacts on dispatch AND
# on the instant retry, dropping and burning both challengers). Two uses:
# the dispatch retry backs off before its second attempt (registry blips are
# global — an instant retry on another host hits the same blip), and a heat
# drop that matches is exempt from the submission burn IF the artifact
# fetches cleanly again at heat settle (see _burn_hotkeys). Matching is on
# the remote stderr tail, which miner code can spoof — that only matters for
# the burn exemption, which is therefore capped at one per hotkey lifetime.
_STORAGE_FAILURE_MARKERS = (
    "storageerror", "hippius", "hf_hub_download", "file_download",
    "generator_artifact_unreachable",
)
STORAGE_RETRY_BACKOFF_SECONDS = 45.0
# Cool-down before a re-queued heat challenger is dispatched again. Longer than
# the in-dispatch storage backoff: a re-queue means the dispatch AND its retry
# both died, so whatever ate them (registry brown-out, provider network blip)
# gets time to pass while other lanes keep training.
HEAT_REQUEUE_COOLDOWN_SECONDS = 120.0


def _storage_failure(exc: BaseException) -> bool:
    """True when an exception (or, for remote failures, the stderr tail in its
    message) points at the storage layer rather than the challenger's code."""
    if isinstance(exc, StorageError):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _STORAGE_FAILURE_MARKERS)


def _infra_failure(exc: BaseException) -> bool:
    """True when a heat failure is the INFRASTRUCTURE's fault, not the
    challenger's: the storage layer (Hippius fetch/push) or the SSH transport
    (rc=255 — connection lost, never the remote command's own exit)."""
    return _storage_failure(exc) or getattr(exc, "returncode", None) == 255


def _run_heat_field(run_fn, challengers, *, max_workers: int, requeues: int,
                    cooldown_seconds: float, note_progress, storage_dropped,
                    sleep=time.sleep):
    """Run every challenger through ``run_fn`` concurrently, re-queueing
    infrastructure casualties in the SAME heat.

    A challenger whose dispatch (and its in-dispatch retry) died on an infra
    failure — storage layer or SSH transport — goes back into the pool up to
    ``requeues`` times, after ``cooldown_seconds``; the 2026-08-19 datacenter
    blip terminally dropped three challengers whose retries happened to be the
    runs in flight. Challenger-fault failures (their generator raising, OOM,
    the guard) never re-queue, so there is no free-retry surface for bad code.

    Returns ``(out, terminal_transport_failures)`` — the latter feeds the
    caller's dead-fleet wipeout check, counting only challengers that EXHAUSTED
    their attempts at the transport level. ``storage_dropped`` maps hotkey→ref
    for terminal storage drops (the burn-exemption candidates), matching the
    old inline behaviour.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
    from concurrent.futures import wait as futures_wait

    out = []
    transport_failures = 0
    requeues_left = {c.hotkey: max(0, int(requeues)) for c in challengers}
    total = len(challengers)
    done = 0

    def _cooled_run(c):
        sleep(cooldown_seconds)
        return run_fn(c)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pending = {ex.submit(run_fn, c): c for c in challengers}
        while pending:
            done_futs, _ = futures_wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done_futs:
                c = pending.pop(fut)
                try:
                    out.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    if _infra_failure(e) and requeues_left.get(c.hotkey, 0) > 0:
                        requeues_left[c.hotkey] -= 1
                        log.warning(
                            "heat: challenger %s hit an infrastructure failure (%s); "
                            "re-queueing after %.0fs cool-down (%d re-queue(s) left)",
                            c.hotkey, e, cooldown_seconds, requeues_left[c.hotkey])
                        pending[ex.submit(_cooled_run, c)] = c
                        continue
                    if getattr(e, "returncode", None) == 255:
                        transport_failures += 1
                    if _storage_failure(e):
                        # Candidate for the burn exemption — re-verified at
                        # heat settle (see _burn_hotkeys).
                        storage_dropped[c.hotkey] = c.ref
                    log.warning("heat: challenger %s failed on remote: %s", c.hotkey, e)
                done += 1
                note_progress(done, total)
    return out, transport_failures


def _load_commit_witness(path: Path) -> dict[str, dict]:
    """Load the persisted commit-order witness (best-effort).

    Shape: ``{hotkey: {"pending": block|None, "committed": block|None}}`` —
    ``pending`` is a commit currently sealed on chain, ``committed`` is the
    block of the last commit we watched go from sealed to revealed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(h): {"pending": v.get("pending"), "committed": v.get("committed")}
                for h, v in raw.items() if isinstance(v, dict)}
    except FileNotFoundError:
        return {}
    except Exception as e:  # noqa: BLE001
        log.warning("commit witness %s unreadable (%s); commit order falls back "
                    "to reveal order until it refills", path, e)
        return {}


def _save_commit_witness(path: Path, witness: dict[str, dict]) -> None:
    """Persist the witness (best-effort — evidence collection must never abort
    a round; a lost file degrades the tie-break, it does not break it)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(witness, sort_keys=True), encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        log.warning("could not persist commit witness to %s: %s", path, e)


@dataclass(frozen=True)
class ResolvedGenerator:
    hotkey: str
    uid: int
    ref: str           # generator's Hippius Hub reference (repo@digest)
    reveal_block: int = 0  # block the pointer became publicly readable at


@dataclass(frozen=True)
class RoundPlan:
    king: ResolvedGenerator | None
    challengers: list[ResolvedGenerator]


# Sentinel identity for the genesis baseline king ([round] genesis_generator_ref):
# a fixed, un-earnable floor that is NOT a registered miner. ``GENESIS_KING_UID``
# is -1 — out of range for every metagraph — so the validator's
# ``decayed_share_vector`` drops it and burns to ``burn_uid``; the baseline
# reigns without drawing emission until a real miner dethrones it. The hotkey is
# a reserved string no wallet can hold, so it never collides with a challenger.
GENESIS_KING_HOTKEY = "__genesis_baseline__"
GENESIS_KING_UID = -1


def make_bench_eval_fn(cfg: ChainConfig, *, device: str = "cpu") -> BenchEvalFn:
    """Default Cascade bench evaluator: run the sidecar on a checkpoint dir over
    GIFT-Eval / BOOM / TIME and return the six-number :class:`BenchScores`, or
    ``None`` when the sidecar can't produce a complete set. Wired in trainer.main
    when ``[scoring] cascade_enabled``; the checkpoint fetch is the caller's job
    (``TrainerRunner._bench_duel_checkpoints``)."""

    def _eval(ckpt_dir: Path) -> BenchScores | None:
        from ..eval.benchmarks import extract_bench_scores, run_benchmarks

        ec = cfg.eval
        report = run_benchmarks(
            ckpt_dir,
            project_dir=ec.benchmark_project_dir,
            suites=("gift-eval", "boom", "time"),
            num_samples=ec.benchmark_num_samples or ec.num_samples,
            max_series=ec.cascade_bench_max_series,  # 0 = full battery
            device=device,
        )
        scores = extract_bench_scores(report)
        return BenchScores(**scores) if scores is not None else None

    return _eval


def resolve_commitments(
    commitments: list[Commitment], cutoff_block: int | None = None,
    floor_block: int = 0,
) -> list[ResolvedGenerator]:
    """Parse each commitment's generator pointer, dropping malformed ones.

    A later reveal from the same hotkey wins (miners re-deploy by committing a
    new ref), so we keep the highest ``reveal_block`` per hotkey.

    When ``cutoff_block`` is given (the round's epoch boundary), only commits
    REVEALED STRICTLY BEFORE it are eligible — this is the daily submission
    deadline, and it gates on the reveal block (the chain's revealed store does
    not carry the original commit block). A timelock reveal landing at/after
    the boundary competes in the next round, not this one; because the boundary
    is deterministic every honest party re-derives the identical field. The
    latest-reveal-wins rule applies only among a hotkey's eligible (pre-cutoff)
    reveals.
    """
    best: dict[str, tuple[int, ResolvedGenerator]] = {}
    for c in commitments:
        # The go-live floor: commits from before the official launch block
        # (netuid squatters, rehearsal commits) never compete — applied to
        # EVERY resolution path, king lookup included, so a pre-live commit
        # can neither enter a heat nor hold a throne. Gates on the reveal
        # block, the only block the revealed store carries.
        if floor_block and c.commit_block < floor_block:
            continue
        if cutoff_block is not None and c.commit_block >= cutoff_block:
            continue
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        rg = ResolvedGenerator(
            hotkey=c.hotkey, uid=c.uid, ref=parsed.ref, reveal_block=c.commit_block
        )
        prev = best.get(c.hotkey)
        if prev is None or c.commit_block >= prev[0]:
            best[c.hotkey] = (c.commit_block, rg)
    return [rg for _, rg in best.values()]


def plan_round(
    resolved: list[ResolvedGenerator],
    king_hotkey: str | None,
    *,
    king: ResolvedGenerator | None = None,
    genesis_ref: str | None = None,
) -> RoundPlan:
    """Split the field into the king and the challengers.

    ``king_hotkey`` is the reigning champion. ``king`` is its pre-resolved
    generator (resolved cutoff-exempt by the caller, since the reigning king is
    not a fresh submission); when omitted it is looked up in ``resolved`` by
    hotkey. Only when there is **no champion at all** (genesis) is the lowest-UID
    generator promoted to interim king. A champion that is named but has no
    resolvable commitment is a loud warning, not a silent swap — silently training
    a different king would make the validator reject every round `king_resyncing`.
    Challengers are returned in a stable order (by UID).

    Two cheap anti-duplicate filters run here, before any generator is fetched or
    trained (a round is ~3h of GPU per generator):

    * **duplicate-of-king** — a challenger whose generator ref equals the king's
      (same ``repo@digest``) is byte-identical to the king (the OCI digest is the
      content hash). It can only tie the king, never clear the win margin, so it
      is dropped rather than handed a wasted round. This is the cascade
      analogue of teutonic's ``check_model_copy`` "same repo + same digest →
      instant reject".
    * **same-ref dedup** — if two hotkeys committed the *same* generator ref,
      only the EARLIEST REVEAL is kept (UID breaking ties); the others would be
      identical runs. First-to-publish wins so committing a competitor's
      visible ref (which needs no upload — the ref string is enough) can never
      steal their slot, whatever the copier's UID.
    """
    by_hotkey = {rg.hotkey: rg for rg in resolved}
    if king is None and king_hotkey:
        king = by_hotkey.get(king_hotkey)
    field_ = sorted(resolved, key=lambda r: r.uid)
    if king is None:
        if genesis_ref:
            # Genesis baseline king ([round] genesis_generator_ref): whenever no
            # on-chain champion has a resolvable commitment, train a FIXED
            # baseline generator as the king — an un-earnable floor — rather than
            # promoting a miner. Its sentinel uid (-1) makes the validator burn
            # emission until a real miner dethrones it (see GENESIS_KING_*). This
            # also means a genesis round always has a king to train, instead of
            # aborting "nothing to train" until the first miner resolves.
            king = ResolvedGenerator(
                hotkey=GENESIS_KING_HOTKEY, uid=GENESIS_KING_UID, ref=genesis_ref)
        else:
            if king_hotkey:
                # A champion exists but we couldn't resolve its generator — never
                # silently crown a challenger in its place (that orphans the throne
                # and the validator rejects the round). Genesis (no champion) is the
                # only case where promoting the lowest UID is correct.
                log.warning("reigning king %s has no resolvable commitment; "
                            "falling back to interim king (validator may hold)", king_hotkey[:12])
            king = field_[0] if field_ else None
    king_ref = king.ref if king is not None else None

    # Earliest reveal (UID tiebreak) owns each duplicated ref — never the
    # lowest UID, which would let a low-UID copier take the original's slot.
    ref_owner: dict[str, ResolvedGenerator] = {}
    for rg in field_:
        cur = ref_owner.get(rg.ref)
        if cur is None or (rg.reveal_block, rg.uid) < (cur.reveal_block, cur.uid):
            ref_owner[rg.ref] = rg

    challengers: list[ResolvedGenerator] = []
    for rg in field_:
        if king is not None and rg.hotkey == king.hotkey:
            continue
        if king_ref is not None and rg.ref == king_ref:
            log.info("dropping challenger %s: generator ref is identical to the king", rg.hotkey)
            continue
        if ref_owner[rg.ref].hotkey != rg.hotkey:
            log.info("dropping challenger %s: duplicate of a ref revealed earlier by %s",
                     rg.hotkey, ref_owner[rg.ref].hotkey)
            continue
        challengers.append(rg)
    return RoundPlan(king=king, challengers=challengers)


def _drop_final_content_clones(
    entries: list[TrainedEntry], jobs: list[tuple[ResolvedGenerator, str]]
) -> list[TrainedEntry]:
    """Drop final-stage challenger entries whose corpus is byte-identical to
    another entry's under the round's shared seed.

    The ref-level filters in :func:`plan_round` cannot see a re-upload of
    someone else's generator (same bytes, different repo ⇒ different ref); the
    corpus digest can — identical content under one :class:`RoundSeeds` yields
    an identical corpus. Per size: a challenger matching the KING's digest is
    dropped (it can only tie, never clear the win margin — the content-level
    analogue of plan_round's duplicate-of-king rule), and among challengers
    sharing a digest only the EARLIEST REVEAL survives (UID tiebreak), so a
    margin-window clone cannot ride a copied corpus into the throne decision.
    """
    order = {rg.hotkey: (rg.reveal_block, rg.uid) for rg, _ in jobs}
    king_digest: dict[str, str] = {e.size: e.corpus_digest for e in entries if e.role == "king"}
    best: dict[tuple[str, str], TrainedEntry] = {}
    for e in entries:
        if e.role != "challenger":
            continue
        cur = best.get((e.size, e.corpus_digest))
        if cur is None or order.get(e.miner_hotkey, (1 << 62, 1 << 62)) < order.get(
            cur.miner_hotkey, (1 << 62, 1 << 62)
        ):
            best[(e.size, e.corpus_digest)] = e

    kept: list[TrainedEntry] = []
    for e in entries:
        if e.role != "challenger":
            kept.append(e)
            continue
        if e.corpus_digest == king_digest.get(e.size):
            log.info("final: challenger %s (%s) dropped: corpus identical to the king's",
                     e.miner_hotkey, e.size)
            continue
        if best[(e.size, e.corpus_digest)].miner_hotkey != e.miner_hotkey:
            log.info("final: challenger %s (%s) dropped: corpus identical to an "
                     "earlier-revealed challenger's", e.miner_hotkey, e.size)
            continue
        kept.append(e)
    return kept


class _FinalLanePool(queue.Queue):
    """Lane queue for the FINAL stage that discovers pods rented mid-final.

    ``get()`` (the blocking, no-timeout form the dispatch path uses) refreshes
    membership from a ``refresh_fn`` on a timeout loop: hosts whose NAMES are
    new enter the rotation, so a slot lost at rental (the provisioner's
    replace-once-then-drop) can be topped up by re-renting while the duel is
    already running. Before this, the fleet was snapshotted once at final
    start and a queued job was bound to it forever — r44 2026-08-27: four jobs
    serialized two-at-a-time on the surviving pod while a replacement pod
    would have sat invisible. Removal stays FAILURE-driven (a dead lane fails
    its dispatch and the retry policy moves on): the hosts file is
    authoritative for membership, never for liveness.
    """

    REFRESH_INTERVAL_S = 60.0

    def __init__(self, initial_hosts: list, refresh_fn):
        super().__init__()
        self._refresh_fn = refresh_fn          # () -> list[RemoteHost]; may raise
        self._absorb_lock = threading.Lock()
        self._known: dict[str, object] = {}
        for h in initial_hosts:
            self._known[getattr(h, "name", str(h))] = h
            super().put(h)

    def known_hosts(self) -> list:
        """Current membership (for lane-count geometry) — grows, never shrinks."""
        return list(self._known.values())

    def _absorb_new(self) -> None:
        try:
            fresh = self._refresh_fn()
        except Exception as e:  # noqa: BLE001 — a torn/absent file keeps the last set
            log.debug("final lane pool refresh failed (keeping current set): %s", e)
            return
        with self._absorb_lock:
            for h in fresh:
                name = getattr(h, "name", None)
                if name and name not in self._known:
                    self._known[name] = h
                    super().put(h)
                    log.info("final lane pool: new lane %s joined mid-final", name)

    def get(self, block: bool = True, timeout: float | None = None):
        if not block or timeout is not None:
            return super().get(block=block, timeout=timeout)
        while True:
            self._absorb_new()
            try:
                return super().get(timeout=self.REFRESH_INTERVAL_S)
            except queue.Empty:
                continue


def _final_repo_suffix(
    jobs: list[tuple[ResolvedGenerator, str]],
    gen: ResolvedGenerator,
    role: str,
) -> str:
    """Checkpoint-repo/work-dir disambiguator for a final-stage run.

    A final's checkpoint repo and work dir are keyed ``<seed>-<role>-<size>``,
    which is unique only while at most ONE challenger trains per size. A
    DEC-CA-0012 cohort puts several challengers in the final at the same size,
    so — exactly like the remote heat's ``-heat-u<uid>`` — each cohort
    challenger gets a per-uid suffix and cannot overwrite a peer's checkpoint
    or repo. Empty for the king and for a single-challenger final, so every
    pre-cohort round keeps its exact repo names and ``trained_pointer`` bytes
    (the inert-default bit-identity DEC-CA-0012 requires).
    """
    if role != "challenger":
        return ""
    if sum(1 for _, r in jobs if r == "challenger") <= 1:
        return ""
    return f"-u{gen.uid}"


@dataclass
class TrainerRunner:
    """Owner-operated trainer. ``base_trainer`` is the GPU backend (Protocol).

    Storage is Hippius: generators + checkpoints on the Hub registry (by
    ``repo@digest``), training logs + the manifest on S3.
    """

    cfg: ChainConfig
    base_trainer: BaseTrainer
    work_root: Path
    wallet: object | None = None       # bittensor wallet for signing (live)
    use_sandbox: bool = True           # run generators in the isolated subprocess
    # Heat screener: scores a trained heat checkpoint (lower better) to rank the
    # field down to [round] finalists before the expensive final. None ⇒ no
    # internal screen (the field's natural order is taken). Wired in trainer.main.
    screen_fn: ScreenFn | None = None
    # Tie run-off screener (DEC-CA-0012): scores the incremental windows
    # [start, stop) of the round's seeded window slice for one heat checkpoint,
    # so a statistically tied heat top can be re-scored on a larger eval before
    # GPU lanes are spent on it. Consulted only when [round] max_finalists > 1
    # AND tie_runoff_windows > 0. None ⇒ no run-off (a tied top advances by the
    # heat ranking, capped). Wired in trainer.main off the same pool source as
    # screen_fn — same permutation, so the slices concatenate paired.
    runoff_fn: RunoffFn | None = None
    # Eval-pool pin: ``(base_seed, block) -> (key, sha256)`` provenance of the
    # pool snapshot this round screens on, stamped (and therefore signed) into
    # the manifest so validators verify their own snapshot selection against it
    # rather than trusting the unsigned pool index. None ⇒ manifests go out
    # unpinned (legacy). Wired in trainer.main from the screen pool source.
    pool_provenance_fn: object | None = None
    # Realised round composition: ``(base_seed, block) -> dict | None`` — the
    # jittered mix's post-hoc domain/class breakdown of the round's eval draw
    # (None while the mix is inactive). Attached to the manifest UNSIGNED (like
    # ``heat``) for the public feed. Wired in trainer.main from the same pool
    # source the screen uses.
    composition_fn: object | None = None
    # Cascade: scores a duel checkpoint on GIFT-Eval / BOOM / TIME for the round's
    # POST-PUBLISH signed bench report (cascade.shared.bench_report) — validators
    # read one authoritative signed set per role, so promotion stays consensus-
    # safe. Runs only when [scoring] cascade_enabled, strictly after the manifest
    # is published (publication never waits on a benchmark). None ⇒ no report
    # (rounds simply contribute no bench numbers). Wired in trainer.main.
    bench_eval_fn: BenchEvalFn | None = None
    # Remote (two-device) training: when ``remote_hosts`` is set, each round's
    # king and challenger train on separate SSH GPU pods in parallel (see
    # cascade.trainer.remote). ``trainer_spec`` is the BaseTrainer 'module:Class'
    # the pods run. None ⇒ local sequential training on this box.
    remote_hosts: list | None = None
    trainer_spec: str | None = None
    remote_timeout_seconds: int = 6 * 3600
    # Frozen-block protection for the live loop's chain reads. A bittensor
    # websocket can go quietly stale (serving a ~20-min-old block) or hang
    # without erroring, which makes run_forever re-enter an already-published
    # round. When the height stops advancing for this long (blocks are ~12s, so
    # a multi-minute freeze is anomalous) the substrate connection is rebuilt
    # before the read is trusted — the same guard the provisioner already runs
    # (cascade.provision.loop._current_block). ``chain_clock`` is the monotonic
    # source the freeze timer reads (injected for tests).
    stale_block_after_s: float = 300.0
    chain_clock: Callable[[], float] = time.monotonic
    # Hard deadline for a single chain read (current_block/block_seed): a
    # bittensor websocket call has no client-side timeout and can hang
    # indefinitely, so every read runs under this wall-clock cap (a hung read is
    # treated exactly like a raised one — rebuild and retry). Mirrors the
    # provisioner's _with_deadline(…, 60.0).
    chain_read_timeout_s: float = 60.0
    # Elastic fleet: when ``remote_hosts_path`` is set, run_forever RE-READS the
    # hosts TOML at the start of every round, so a per-round provisioner (rent
    # pods when the field is big, tear down after) changes the fleet without a
    # trainer restart. A missing/empty file ⇒ this round trains locally.
    # ``hosts_wait_seconds`` waits up to that long for the file to appear/fill
    # before falling back — with timed reveals the field is only countable
    # ~reveal_margin_blocks before the boundary, so pods finish booting after
    # the round starts.
    remote_hosts_path: Path | None = None
    hosts_wait_seconds: int = 0
    # Post-round public-benchmark telemetry (GIFT-Eval/BOOM/TIME) of the round's
    # king on the idle pod. LOG-ONLY: validators score rounds exclusively on the
    # private eval pool; this never feeds weights or the throne (see bench_hook).
    bench_plan: object | None = None
    # Cascade duel bench on the REMOTE workers: when set (cascade_enabled +
    # remote_hosts), each duel checkpoint's GIFT-Eval/BOOM/TIME scoring runs on
    # the pod that just trained it — GPU, checkpoint already local — instead of
    # a local-CPU subprocess. The numbers land in the round's post-publish
    # signed bench report. Falls back to the local ``bench_eval_fn`` when there
    # is no remote host.
    cascade_bench_plan: object | None = None
    # Live presentational reporting for the dashboards: the round stage
    # (``status/round.json`` — where the round actually is, instead of a
    # wall-clock estimate that ignores field size) AND the heat standings
    # (``status/heat.json`` + ``heats/`` — published when the heat settles, not
    # when the round's receipt lands; DEC-CA-0011). OFF by default so offline
    # runs and tests never touch storage; trainer.main enables it for the live
    # service. Best-effort everywhere — a publish failure must never disturb a
    # round.
    publish_stage_status: bool = False
    # Cascade warm-start consumption (DEC-CA-0005/0012): path of the
    # promoted-init pointer file (``[validator] warm_start_init_path``), written
    # by the trainer's own promotion engine (below). When the file exists,
    # every run this round initialises from this epoch's member of the promoted
    # set instead of random init, and the manifest records the pin (signed) so
    # validators verify it against the envelope and cascade-audit re-derives
    # from it. None ⇒ random init always (cascade off / pre-warm-start deploy).
    warm_start_path: Path | None = None
    # Cascade promotion engine (cascade.trainer.promotion, DEC-CA-0013): the
    # trainer is the selection authority — the engine tracks the reign, logs
    # benched duel candidates, fires promotions (signed PromotionRecord to the
    # manifest bucket), and writes the pointer file above. None ⇒ the trainer
    # only CONSUMES a pointer file (or trains from random init) and never
    # promotes.
    promotion: object | None = None
    # Restart re-entry escape hatch (--force-rerun-round, Approve-tier): the ONE
    # round_id allowed past the already-published guard, for a legitimate
    # operator-driven re-train/re-publish of a finished round. None ⇒ the guard
    # always applies.
    force_rerun_round: str | None = None
    _hub: HubConfig | None = field(default=None, repr=False)
    _manifest_store: S3Store | None = field(default=None, repr=False)
    _logs_store: S3Store | None = field(default=None, repr=False)
    # Per-round starvation/deadline telemetry collected from every run trained
    # IN THIS PROCESS (local rounds; a remote round's metrics stay on its pods,
    # where each worker logs its own telemetry line). Keyed by stage for the
    # roll-up; reset at every run_round.
    _round_telemetry: dict = field(
        default_factory=lambda: {"heat": [], "final": []}, repr=False
    )
    # Which pod each FINAL run actually landed on, keyed by (role, size,
    # hotkey) — the post-publish bench runs on the pod that holds the
    # checkpoint at its _train_work path, and the dispatch retry means that is
    # not always the round-robin pick. Reset at every run_round.
    _final_role_hosts: dict = field(default_factory=dict, repr=False)
    # Context for stage reporting, set at round start so publish() (which only
    # sees the manifest) and the heat-progress hooks know which round they are
    # reporting for. ``_stage_published_at`` throttles heat-progress writes.
    _stage_ctx: dict | None = field(default=None, repr=False)
    _stage_published_at: float = field(default=0.0, repr=False)
    # Frozen-block tracker (block height, wall-time it last advanced) for
    # _block_with_freeze_guard. Rebuilt naturally on the first read.
    _last_block: int | None = field(default=None, repr=False)
    _block_changed_at: float = field(default=0.0, repr=False)
    # Last king a signed receipt named (see _receipt_king): sticky across
    # transient fetch failures so the reign clock never flaps back to the
    # lagging incentive king mid-dethrone.
    _last_receipt_king: str | None = field(default=None, repr=False)
    # Heat drops whose failure matched the storage layer (hotkey → gen ref),
    # reset each round: candidates for the burn exemption in _burn_hotkeys.
    _storage_dropped: dict = field(default_factory=dict, repr=False)
    # Challengers dropped BEFORE the field was even eligible (today: burned
    # hotkeys re-committing their one used submission), reset each round.
    # ``[{hotkey, uid, reason}]`` — feeds the published heat standings so a
    # skipped miner can read WHY it is not listed (r48: 328 burned re-commits
    # published as an unexplained "0 entrants").
    _round_skipped: list = field(default_factory=list, repr=False)

    # ── storage handles (lazy so offline/tests need no Hippius) ──────────────

    def hub(self) -> HubConfig:
        if self._hub is None:
            self._hub = HubConfig.from_storage(self.cfg.storage)
        return self._hub

    def _hf_ckpt_repo(self, ckpt_repo: str) -> str | None:
        """The HuggingFace **model** repo to mirror a checkpoint to when the Hub is
        down, or ``None`` when no HF fallback is configured (Hub-only).

        Reuses the HF account from ``[storage] hf_backup_repo`` — the same mirror
        the manifest/receipt store already falls back to; its namespace owns the
        model repos too — and keeps the Hub repo's basename, so a fallback ref
        reads ``<hf_ns>/ckpt-r…``. Empty/namespaceless ``hf_backup_repo`` ⇒ no
        checkpoint fallback, matching the manifest store's own gating."""
        backup = self.cfg.storage.hf_backup_repo
        if not backup or "/" not in backup:
            return None
        ns = backup.split("/", 1)[0]
        return f"{ns}/{ckpt_repo.rsplit('/', 1)[-1]}"

    def manifest_store(self):
        # HF-backed when [storage] hf_backup_repo is set, else plain S3 — so the
        # trainer's manifest write survives a Hippius S3 outage (writes to HF).
        if self._manifest_store is None:
            from ..shared.hippius import open_manifest_store

            self._manifest_store = open_manifest_store(self.cfg.storage)
        return self._manifest_store

    def logs_store(self) -> S3Store:
        if self._logs_store is None:
            self._logs_store = S3Store(
                S3Config.from_storage(self.cfg.storage, bucket=self.cfg.storage.logs_bucket)
            )
        return self._logs_store

    # ── anti-spam: commit-order witness (who submitted a generator FIRST) ────

    def _commit_witness_path(self) -> Path:
        p = Path(self.cfg.round.commit_witness_path)
        return p if p.is_absolute() else (self.work_root / p)

    def witness_commits(self, client: object) -> dict[str, dict]:
        """Record which hotkeys have a commit sealed on chain right now.

        Called every poll tick, INCLUDING the ticks that skip round work — the
        window this observes is the epoch, not the round. The chain deletes a
        commit's block when drand reveals it (see
        :meth:`ChainClient.poll_pending_commits`), so evidence of who committed
        first exists only for whoever was watching while it was sealed.

        State machine per hotkey: a sealed commit sets ``pending``; when that
        hotkey's seal disappears (the reveal landed) ``pending`` freezes into
        ``committed``, which is what the duplicate screen orders on. A new seal
        overwrites ``pending`` — a hotkey that re-commits is claiming a new
        submission, and the old commit's priority must not follow it.

        Best-effort throughout: this is evidence collection, and a tick that
        fails just means the tie-break degrades to reveal order.
        """
        poll = getattr(client, "poll_pending_commits", None)
        if poll is None:
            return {}
        try:
            sealed = poll()
        except Exception as e:  # noqa: BLE001 — advisory; never break the loop
            log.warning("commit witness: poll failed (%s)", e)
            sealed = None
        if sealed is None:
            # A failed read says nothing about what is sealed on chain. Running
            # the freeze pass on it would treat every pending commit as just
            # revealed, on no evidence at all.
            return _load_commit_witness(self._commit_witness_path())
        path = self._commit_witness_path()
        witness = _load_commit_witness(path)
        changed = False
        for hotkey, block in sealed.items():
            entry = witness.setdefault(hotkey, {"pending": None, "committed": None})
            if entry.get("pending") != block:
                entry["pending"] = block
                changed = True
                log.info("commit witness: %s sealed a commit at block %s",
                         hotkey, block)
        for hotkey, entry in witness.items():
            pending = entry.get("pending")
            if pending is not None and hotkey not in sealed:
                # The seal is gone: that commit revealed, so its block is now
                # the hotkey's evidence of when it submitted.
                entry["pending"] = None
                entry["committed"] = int(pending)
                changed = True
                log.info("commit witness: %s revealed; commit block %s recorded",
                         hotkey, pending)
        if changed:
            _save_commit_witness(path, witness)
        return witness

    def _commit_priority(self, entrants: list[ResolvedGenerator]) -> dict[str, int]:
        """``{hotkey: ordering block}`` for the duplicate screen's tie-break.

        The witnessed commit block when we have one; otherwise the entrant's
        REVEAL block, which is the same units and always ``>=`` the commit it
        belongs to. That fallback is deliberately not "sort them last": an
        entrant we merely failed to observe must not be expropriated for it,
        and reveal order still puts a copyist behind their victim — copying a
        revealed repo means revealing after it.
        """
        witness = _load_commit_witness(self._commit_witness_path())
        priority: dict[str, int] = {}
        missing, unknown = [], []
        for c in entrants:
            committed = (witness.get(c.hotkey) or {}).get("committed")
            if committed:
                priority[c.hotkey] = int(committed)
            elif c.reveal_block:
                priority[c.hotkey] = int(c.reveal_block)
                missing.append(c.hotkey)
            else:
                # Neither witnessed nor revealed at a known block. Leaving it
                # out sorts it LAST (screen_duplicates' _LAST), with UID
                # breaking ties among the equally-unknown. Mapping it to block
                # 0 instead would sort it FIRST and let an entrant we know
                # nothing about win every collision it appears in.
                unknown.append(c.hotkey)
        if missing:
            log.info("commit witness: no commit block for %d/%d entrant(s) "
                     "(%s%s) — they order on their reveal block",
                     len(missing), len(entrants), ", ".join(missing[:5]),
                     "…" if len(missing) > 5 else "")
        if unknown:
            log.warning("commit witness: no commit AND no reveal block for %s "
                        "— ordered last", ", ".join(unknown))
        return priority

    # ── anti-spam: 1 hotkey = 1 submission (lifetime) ────────────────────────

    def _submissions_path(self) -> Path:
        """Where the burn set is persisted. A relative ``submissions_db_path`` is
        resolved under ``work_root`` (a stable per-deployment dir; per-test tmp)."""
        p = Path(self.cfg.round.submissions_db_path)
        return p if p.is_absolute() else (self.work_root / p)

    def _filter_burned_challengers(
        self, challengers: list[ResolvedGenerator]
    ) -> list[ResolvedGenerator]:
        """Drop challengers whose hotkey already used its one submission.

        Read-only: the survivors are burned by :meth:`_burn_hotkeys` only after
        the heat stage completes. No-op when ``[round] one_submission_per_hotkey``
        is False (testnet). The king is never here (``plan_round`` separates it),
        so the incumbent is exempt.

        Skips are recorded on ``_round_skipped`` (reset here, once per round)
        so the published heat standings can say WHY a committed hotkey is not
        an entrant — the skip is otherwise visible only in this journal.
        """
        self._round_skipped = []
        if not self.cfg.round.one_submission_per_hotkey:
            return challengers
        seen = _load_seen_hotkeys(self._submissions_path())
        for c in challengers:
            if c.hotkey in seen:
                log.info("skipping challenger %s: hotkey already used its 1 submission "
                         "(re-register to resubmit)", c.hotkey)
                self._round_skipped.append({"hotkey": c.hotkey, "uid": c.uid,
                                            "reason": "already_submitted"})
        return [c for c in challengers if c.hotkey not in seen]

    def _burn_hotkeys(self, challengers: list[ResolvedGenerator]) -> None:
        """Burn the challengers that got their shot: 1 hotkey = 1 submission.

        Called AFTER the heat stage completes (not at entry): a round that
        crashes or aborts mid-heat — a pod fleet dying, the trainer restarting —
        must never consume a miner's single lifetime submission without having
        actually screened it. Entrants whose own generator failed to train or
        score DO burn (that was their shot); a round-level failure before this
        point burns no one and the field simply re-enters the retried round.

        Storage-fault exemption: a challenger dropped by a failure that matched
        the storage layer (``_storage_dropped``) is NOT burned when its artifact
        fetches cleanly at settle time — that proves a transient registry fault
        on our boundary, not a broken submission (2026-08-11: a ~1-min 401 blip
        burned two fetchable challengers). The failure text comes from remote
        stderr, which miner code can spoof to dodge its burn, so each hotkey
        gets this exemption ONCE (persisted beside the burn set); a still-dead
        artifact (private/missing repo) burns as before.
        """
        if not self.cfg.round.one_submission_per_hotkey or not challengers:
            return
        exempt: set[str] = set()
        casualties = {c.hotkey: self._storage_dropped[c.hotkey]
                      for c in challengers if c.hotkey in self._storage_dropped}
        if casualties:
            ex_path = self._submissions_path().with_name("trainer_burn_exemptions.json")
            already = _load_seen_hotkeys(ex_path)
            for hk, ref in casualties.items():
                if hk in already:
                    log.warning("burning %s despite a storage-fault drop: its one "
                                "lifetime exemption is used", hk)
                elif self._artifact_fetchable_now(ref):
                    exempt.add(hk)
                    log.warning("not burning %s: dropped by a storage-layer fault but "
                                "%s fetches cleanly at settle (transient registry "
                                "error) — it re-enters next round", hk, ref[:48])
                else:
                    log.info("burning %s: artifact %s still unfetchable at settle "
                             "(miner-side, not a transient fault)", hk, ref[:48])
            if exempt:
                _save_seen_hotkeys(ex_path, already | exempt)
        path = self._submissions_path()
        seen = _load_seen_hotkeys(path)
        _save_seen_hotkeys(path, seen | {c.hotkey for c in challengers
                                         if c.hotkey not in exempt})

    def _artifact_fetchable_now(self, ref: str) -> bool:
        """Re-test a generator fetch from this box (burn-exemption evidence).

        Best-effort and conservative: any failure — including a full storage
        outage — reports unfetchable, so the exemption is only granted on
        positive proof that the artifact serves cleanly."""
        import shutil
        import uuid

        probe_dir = self.work_root / f"_burn_verify-{uuid.uuid4().hex[:8]}"
        try:
            fetch_from_hub(ref, probe_dir, self.hub())
            return True
        except Exception as e:  # noqa: BLE001
            log.info("burn re-verify: %s still failing (%s)", ref[:48], e)
            return False
        finally:
            shutil.rmtree(probe_dir, ignore_errors=True)

    def _archive_generator_tree(self, ref: str, d: Path) -> None:
        """Best-effort audit archive of a fetched generator tree.

        Miners routinely delete their repos post-round (OPSLOG 2026-08-25:
        three deletions in one night broke an anneal and two legs' audit
        replay); the resolve-time fetch is the one guaranteed moment the code
        exists. Tar the tree to the manifest bucket under its ref key — the
        mirror store dual-writes to R2 like every other put. Skip-if-present
        keeps it idempotent across retries; ANY failure is swallowed — the
        archive must never disturb the round it is recording.
        """
        import io
        import tarfile
        try:
            store = self.manifest_store()
            key = generator_archive_key(ref)
            try:
                store.get_bytes(key)
                return  # already archived (tars are KBs; a GET probe is fine)
            except Exception:  # noqa: BLE001 — missing/unreadable ⇒ (re)write
                pass
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                tar.add(str(d), arcname=".")
            store.put_bytes(key, buf.getvalue())
            log.info("archived generator %s (%d bytes)", ref[:60], buf.tell())
        except Exception as e:  # noqa: BLE001 — never sink a round for the archive
            log.warning("generator archive failed for %s (ignored): %s", ref[:60], e)

    # ── anti-spam: content-level duplicate screen (pre-heat) ─────────────────
    # Probe concurrency: sandboxes are subprocesses, each holding up to
    # [generator] max_memory_mb, so this also bounds the stage's peak RSS.

    def _screen_duplicate_entrants(
        self,
        king: ResolvedGenerator | None,
        entrants: list[ResolvedGenerator],
        base_seed: int,
        *,
        static_only: bool = False,
        report: bool = True,
        budget_seconds: int | None = None,
    ) -> list[ResolvedGenerator]:
        """Drop entrants whose repo CONTENT duplicates the king or an earlier
        entrant, before any heat GPU is spent (see :mod:`cascade.interface.dedup`).

        The on-chain same-ref dedup in :func:`plan_round` only catches identical
        ``repo@digest`` pointers; this compares the fetched trees, so
        re-uploads, comment shuffles, and rename-only copies all collapse to
        the earliest-committed original. Enforcement is EXACT-identity only
        (no similarity threshold — gameable by spacing; see
        :mod:`cascade.interface.dedup`). Judgement is pairwise against a
        specific rival, never transitive.

        Fail-open: an infrastructure error (fetch outage, unexpected exception)
        keeps the entrant in the heat rather than eating its slot on a guess —
        except an entrant whose OWN ref does not fetch, which is dropped (it
        could not have trained anyway; per the burn rules that was its shot).
        In ``shadow`` mode verdicts are computed and logged but nothing drops.
        Every verdict lands in ``<work_root>/<round>/dedup_report.json``.

        The whole screen runs under ``[round] dedup_phase_seconds``, or
        ``budget_seconds`` when the caller has a tighter one. Its inputs are
        attacker-chosen (repo bytes, generator runtime), so an unbounded screen
        is a way to stall the round it exists to protect; on expiry the field
        proceeds UNSCREENED, which is the same fail-open direction every other
        error path here takes. ``static_only`` runs the fingerprint tiers
        without the probe (no code execution) — that is what makes the screen
        safe to call from the provisioner's sizing path, which passes its own
        budget because it is itself running under a subprocess timeout.
        """
        mode = (self.cfg.round.dedup_mode or "off").lower()
        if mode not in ("shadow", "enforce") or not entrants:
            return entrants
        import shutil

        fetch_root = self.work_root / f"{base_seed}" / "dedup"
        budget = max(30, int(budget_seconds if budget_seconds is not None
                             else self.cfg.round.dedup_phase_seconds or 0) or 10 ** 9)
        try:
            return self._with_deadline(
                lambda: self._screen_duplicate_entrants_inner(
                    king, entrants, base_seed, mode, fetch_root,
                    static_only=static_only, report=report),
                budget)
        except TimeoutError:
            # The helper thread is abandoned (it dies with the process, as in
            # _with_deadline's other callers) and its fetch tree is removed
            # underneath it — any probe still running there fails harmlessly
            # into a log line, because the round has already moved on.
            log.error("dedup: screen exceeded its %ss budget for round=%s; the "
                      "field proceeds UNSCREENED", budget, base_seed)
            return entrants
        finally:
            shutil.rmtree(fetch_root, ignore_errors=True)

    def _screen_duplicate_entrants_inner(
        self,
        king: ResolvedGenerator | None,
        entrants: list[ResolvedGenerator],
        base_seed: int,
        mode: str,
        fetch_root: Path,
        *,
        static_only: bool = False,
        report: bool = True,
    ) -> list[ResolvedGenerator]:
        from ..interface.dedup import (
            collapse_identical_behavior,
            fingerprint_dir,
            screen_duplicates,
        )

        rnd = self.cfg.round
        fp_kwargs = {"max_tokens": rnd.dedup_max_tokens,
                     "max_text_mb": rnd.dedup_max_text_mb}
        king_fp = None
        king_dir: Path | None = None
        if king is not None:
            try:
                king_dir = Path(fetch_from_hub(king.ref, fetch_root / "king",
                                               hub=self.hub()))
                self._archive_generator_tree(king.ref, king_dir)
                king_fp = fingerprint_dir(king_dir, **fp_kwargs)
            except Exception as e:  # noqa: BLE001 — king trains later regardless
                log.warning("dedup: king repo %s unfetchable (%s); screening "
                            "challengers against each other only", king.ref, e)

        fetch_failed: list[ResolvedGenerator] = []
        unscreened: list[ResolvedGenerator] = []
        oversize: list[ResolvedGenerator] = []
        bulky: list[tuple[ResolvedGenerator, object]] = []
        triples: list[tuple[str, int, object]] = []
        dirs: dict[str, Path] = {}
        for c in entrants:
            try:
                d = Path(fetch_from_hub(c.ref, fetch_root / f"u{c.uid}", hub=self.hub()))
                self._archive_generator_tree(c.ref, d)
                # A repo over [generator] max_repo_mb fails its heat run anyway
                # (build_round_corpus checks the same bound), so there is no
                # reason to spend fingerprint time on it — and skipping it
                # keeps the screen's cost off an input the miner chooses.
                size = check_repo_size(d, self.cfg.generator.max_repo_mb)
                if not size.ok:
                    log.info("dedup[%s]: challenger %s (uid=%s) ref %s is %s — "
                             "not screened; the heat rejects it on its own",
                             mode, c.hotkey, c.uid, c.ref, size.reason)
                    oversize.append(c)
                    continue
                fp = fingerprint_dir(d, **fp_kwargs)
            except StorageError as e:
                # Fault attribution matters: pods fetch --gen-ref themselves,
                # so an orchestrator-side fetch failure does NOT mean the ref
                # cannot train (a flaky orchestrator↔Hub leg has coincided
                # with pods pulling the same refs clean). Drop only when the
                # response pins the fault on the miner — denied or missing
                # (401/403/404). Anything else (transport, 5xx, timeout) fails
                # OPEN: the entrant proceeds to the heat unscreened and the
                # pod re-fetches.
                status = _http_status_in_chain(e)
                if status in (401, 403, 404):
                    log.info("dedup[%s]: challenger %s (uid=%s) ref %s denied/"
                             "missing (HTTP %s)%s", mode, c.hotkey, c.uid, c.ref,
                             status, " — dropped pre-heat" if mode == "enforce"
                             else " — shadow, kept")
                    fetch_failed.append(c)
                else:
                    log.warning("dedup: fetch of %s failed (%s); challenger %s "
                                "(uid=%s) proceeds unscreened", c.ref, e,
                                c.hotkey, c.uid)
                    unscreened.append(c)
                continue
            except Exception as e:  # noqa: BLE001 — repo bytes are attacker-chosen;
                # one repo crafted to crash the fingerprinter must cost only its
                # own screening, never the screen (let alone the round).
                log.warning("dedup: fingerprint of %s failed (%s: %s); challenger "
                            "%s (uid=%s) proceeds unscreened", c.ref,
                            type(e).__name__, e, c.hotkey, c.uid)
                unscreened.append(c)
                continue
            if not fp.scoreable:
                # Not an error, but not normal either: honest submissions run
                # ~100KB / 7-11k tokens. The exact digest tiers still judge
                # this entrant (digests are streamed, never capped); only the
                # config_only delta measurement is unavailable. Bulk that
                # reads as padding belongs in the report where a human can
                # see the pattern build.
                log.warning("dedup: challenger %s (uid=%s) has %d tokens%s — over "
                            "the %d cap; exact digest tiers only",
                            c.hotkey, c.uid, fp.n_tokens,
                            " (files too large to decode in full)"
                            if fp.truncated else "", rnd.dedup_max_tokens)
                bulky.append((c, fp))
            dirs[c.hotkey] = d
            triples.append((c.hotkey, c.uid, fp))

        try:
            result = screen_duplicates(
                triples, king_fp,
                config_only_enforce=rnd.dedup_config_only_enforce,
                priority=self._commit_priority(entrants),
                enforce=(mode == "enforce"),
            )
        except Exception as e:  # noqa: BLE001 — the screen must never sink a round
            log.warning("dedup: screen failed (%s); heat proceeds unscreened", e)
            if mode == "enforce":
                return [c for c in entrants if c not in fetch_failed]
            return entrants

        # ── behavioral probe: determinism + same-process collapse ────────────
        # Probe only the entrants the static tiers kept — no sandbox time on
        # already-dropped copies. Each survivor's generator draws a small
        # corpus TWICE under the shared round seed: two draws that differ
        # violate the determinism contract (`cascade verify` enforces the same
        # rule miner-side); identical probe bytes across two survivors (or vs
        # the king) is the same generative process, whatever the code says.
        # Probe-derived drops are gated on [round] dedup_probe_mode,
        # INDEPENDENTLY of dedup_mode — the static tiers can enforce while
        # the probe observes (probe drops burn hotkeys; ship shadow first).
        probe_mode = (rnd.dedup_probe_mode or "off").lower()
        probe_enforce = probe_mode == "enforce"
        probe_dropped: list[dict] = []
        behavior_dropped: tuple = ()
        probe_n = int(rnd.dedup_probe_series or 0)
        per_draw = 0
        # static_only short-circuits: the sizing path must not even ask about
        # the sandbox, let alone log about it.
        if static_only or (probe_n > 0 and probe_mode != "off"
                           and not self._probe_sandbox_ok()):
            probe_mode, probe_enforce, probe_n = "off", False, 0
        if probe_n > 0 and probe_mode in ("shadow", "enforce"):
            from concurrent.futures import ThreadPoolExecutor

            uid_of = {h: u for h, u, _ in triples}
            survivors = [c for c in entrants
                         if c.hotkey in set(result.kept_hotkeys)
                         and c.hotkey in dirs]
            # The probe gets its OWN wall clock, and the per-draw share is
            # derived from the STAGE budget: the full-corpus budget would let
            # one hostile submission stall the orchestrator for its whole
            # duration per draw, and a fixed per-draw budget still scales the
            # stage with the size of the field (which the field chooses).
            # Worst case is dedup_probe_budget_seconds, not N × per-draw.
            # The sandbox spends REAL wall clock on top of max_generate_seconds
            # (+30s subprocess kill slack, +120s container startup — see
            # sandbox.py/_container.py communicate timeouts), so the stage must
            # fund per_draw + grace per draw or it overruns its budget, trips
            # the outer dedup_phase_seconds deadline, and takes the static
            # verdicts down with it. If the field is too large for the budget
            # to fund even the floor, the probe SKIPS — never the whole screen.
            grace = (120 if self.cfg.generator.sandbox_mode == "container"
                     else 30)
            waves = max(1, -(-(len(survivors) + (1 if king_dir is not None
                                                 else 0)) // _PROBE_WORKERS))
            per_draw = min(int(rnd.dedup_probe_generate_seconds),
                           int(rnd.dedup_probe_budget_seconds) // (waves * 2)
                           - grace)
            if per_draw < _PROBE_MIN_DRAW_SECONDS:
                log.warning(
                    "dedup-probe[%s]: %d survivor(s) (%d wave(s)) cannot fit "
                    "the %ss stage budget once the %ss spawn grace per draw is "
                    "funded — probe SKIPPED this round; static tiers already "
                    "applied", probe_mode, len(survivors), waves,
                    rnd.dedup_probe_budget_seconds, grace)
                probe_mode, probe_enforce, probe_n, per_draw = "off", False, 0, 0
        if probe_n > 0 and probe_mode in ("shadow", "enforce"):
            # corpus_target_points is zeroed: the probe compares SMALL
            # fixed-count draws across entrants, so it must stay count-
            # denominated even when the materialised drain is armed points-
            # denominated (DEC-CA-0031).
            probe_cfg = replace(
                self.cfg.generator, corpus_n_series=probe_n,
                max_generate_seconds=per_draw, corpus_target_points=0)
            gen_seed = RoundSeeds.derive(base_seed, self.cfg.training).generation_seed
            log.info("dedup-probe[%s]: %d survivor(s), %d wave(s), %ss per draw "
                     "(stage budget %ss)", probe_mode, len(survivors), waves,
                     per_draw, rnd.dedup_probe_budget_seconds)

            # Sandboxes are subprocesses, so a small thread pool bounds the
            # worst-case wall clock without stacking rlimits in-process.
            with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
                king_future = (pool.submit(self._probe_digest, king_dir, gen_seed,
                                           probe_cfg, check_determinism=False)
                               if king_dir is not None else None)
                futures = {c.hotkey: pool.submit(self._probe_digest,
                                                 dirs[c.hotkey], gen_seed, probe_cfg)
                           for c in survivors}

                king_digest = None
                if king_future is not None:
                    try:
                        king_digest = king_future.result()
                    except Exception as e:  # noqa: BLE001 — never the challengers' problem
                        log.warning("dedup: king probe failed (%s); behavior tier "
                                    "runs among challengers only", e)

                behaved: list[tuple[str, int, str]] = []
                for c in survivors:  # deterministic order regardless of completion
                    try:
                        digest = futures[c.hotkey].result()
                    except CorpusError as e:
                        if not str(e).startswith(_PROBE_MINER_FAULT_PREFIXES):
                            # Sandbox/infra fault (missing image, no netns,
                            # container crash) — never the miner's problem.
                            log.warning("dedup: probe infra fault for %s (%s); "
                                        "kept unprobed", c.hotkey, e)
                            behaved.append((c.hotkey, uid_of[c.hotkey],
                                            f"\x00unprobed:{c.hotkey}"))
                            continue
                        tier = ("nondeterministic" if "non-deterministic" in str(e)
                                else "probe_failed")
                        log.info("dedup-probe[%s]: challenger %s (uid=%s) %s: %s%s",
                                 probe_mode, c.hotkey, c.uid, tier, e,
                                 "" if probe_enforce else " — shadow, kept")
                        probe_dropped.append({"hotkey": c.hotkey, "uid": c.uid,
                                              "tier": tier, "detail": str(e)[:300]})
                    except Exception as e:  # noqa: BLE001 — infra failure: fail open
                        log.warning("dedup: probe infrastructure error for %s (%s); "
                                    "kept unprobed", c.hotkey, e)
                        behaved.append((c.hotkey, uid_of[c.hotkey],
                                        f"\x00unprobed:{c.hotkey}"))
                    else:
                        behaved.append((c.hotkey, uid_of[c.hotkey], digest))

            behavior_kept, behavior_dropped = collapse_identical_behavior(
                behaved, king_digest)
            for v in behavior_dropped:
                log.info("dedup-probe[%s]: challenger %s (uid=%s) is "
                         "behavior_identical to %s (uid=%s) under the shared "
                         "seed%s", probe_mode, v.hotkey, v.uid, v.matched_hotkey,
                         v.matched_uid, "" if probe_enforce else " — shadow, kept")

        for v in result.dropped:
            log.info("dedup[%s]: challenger %s (uid=%s) is %s of %s (uid=%s)%s",
                     mode, v.hotkey, v.uid, v.tier,
                     v.matched_hotkey, v.matched_uid,
                     "" if mode == "enforce" else " — shadow, kept")
        for v in result.shadow:
            log.info("dedup[label]: challenger %s (uid=%s) vs %s (uid=%s) "
                     "%s (delta=%s) — logged, kept", v.hotkey, v.uid,
                     v.matched_hotkey, v.matched_uid, v.tier, v.abs_delta)

        report_doc = {
            "round_id": str(base_seed),
            "mode": mode,
            "config_only_enforce": rnd.dedup_config_only_enforce,
            "max_tokens": rnd.dedup_max_tokens,
            "max_text_mb": rnd.dedup_max_text_mb,
            "probe_mode": probe_mode,
            "probe_series": probe_n,
            "probe_seconds_per_draw": per_draw,
            "fetch_failed": [{"hotkey": c.hotkey, "uid": c.uid, "ref": c.ref}
                             for c in fetch_failed],
            "unscreened": [{"hotkey": c.hotkey, "uid": c.uid, "ref": c.ref}
                           for c in unscreened],
            "oversize": [{"hotkey": c.hotkey, "uid": c.uid, "ref": c.ref}
                         for c in oversize],
            "over_token_cap": [{"hotkey": c.hotkey, "uid": c.uid,
                                "n_tokens": fp.n_tokens, "truncated": fp.truncated}
                               for c, fp in bulky],
            "dropped": [
                *(vars(v) | {"enforced": mode == "enforce"} for v in result.dropped),
                *(vars(v) | {"enforced": probe_enforce} for v in behavior_dropped),
            ],
            "probe_dropped": [d | {"enforced": probe_enforce} for d in probe_dropped],
            "shadow": [vars(v) for v in result.shadow],
        }
        if report:
            self._write_dedup_report(base_seed, report_doc)

        # Static-tier drops (and denied/missing refs) apply under dedup_mode;
        # probe-derived drops apply under dedup_probe_mode — independent gates.
        # (fetch_failed never entered triples, so enforce excludes them; the
        # unscreened — orchestrator-side fetch faults — stay IN.)
        kept = (set(result.kept_hotkeys) | {c.hotkey for c in unscreened}
                | {c.hotkey for c in oversize}
                if mode == "enforce" else {c.hotkey for c in entrants})
        if probe_enforce:
            kept -= {d["hotkey"] for d in probe_dropped}
            kept -= {v.hotkey for v in behavior_dropped}
        return [c for c in entrants if c.hotkey in kept]

    def _write_dedup_report(self, base_seed: int, doc: dict) -> None:
        """Persist the round's dedup verdicts to disk and to the logs store.

        The logs-store copy is the shadow-mode EVIDENCE this feature exists to
        collect, so it must not depend on orchestrator disk. Verdicts are
        exact-identity tiers (no tunable bar to reverse-engineer), but the
        config_only deltas still profile the field — keep the logs bucket
        private.
        """
        report_json = json.dumps(doc, indent=1)
        try:
            out_dir = self.work_root / f"{base_seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "dedup_report.json").write_text(report_json)
        except OSError as e:
            log.warning("dedup: could not write report for round=%s: %s", base_seed, e)
        try:
            self.logs_store().put_text(
                f"logs/round-{base_seed}/dedup_report.json", report_json,
                content_type="application/json")
        except Exception as e:  # noqa: BLE001 — telemetry only, never sinks a round
            log.warning("dedup: report upload failed for round=%s: %s", base_seed, e)

    def _probe_sandbox_ok(self) -> bool:
        """Whether the probe may execute untrusted generator code here.

        The probe's threat model (DEC-CA-0008) rests entirely on the sandbox
        being KERNEL-enforced, because unlike a pod the orchestrator is not
        disposable: it holds the private eval pool and the trainer's wallet,
        and the subprocess sandbox shares their uid and filesystem. Container
        mode gives that (``--network=none``, ``--cap-drop=ALL``, read-only
        rootfs, only the repo bind-mounted); subprocess mode gives it only with
        ``sandbox_strict``, which refuses to run when netns is unavailable
        instead of degrading to the in-process socket guard — a guard that does
        not survive a C extension or a spawned child.

        Neither ⇒ the probe disables itself. Note that SHADOW mode does not
        make this safe: shadow gates the drops, not the execution.
        """
        g = self.cfg.generator
        if self.use_sandbox and (g.sandbox_mode == "container" or g.sandbox_strict):
            return True
        # ``use_sandbox=False`` runs generator code IN THIS PROCESS — strictly
        # worse than a degradable sandbox, so it needs the opt-in too. (Earlier
        # revisions of this gate treated it as safe because it is the in-process
        # test path; a security gate that returns "fine" for its worst input is
        # wrong even while nothing in production reaches it.)
        posture = "in-process (use_sandbox=false)" if not self.use_sandbox else \
            f"sandbox_mode={g.sandbox_mode!r}, sandbox_strict=false"
        if self.cfg.round.dedup_probe_allow_weak_sandbox:
            log.warning("dedup-probe: running with %s — untrusted generator code "
                        "shares this host with the eval pool and wallet "
                        "(dedup_probe_allow_weak_sandbox=true)", posture)
            return True
        log.error("dedup-probe: DISABLED — the probe executes untrusted "
                  "generator code on the orchestrator and this host offers only "
                  "%s. Set sandbox_mode='container' or sandbox_strict=true in "
                  "[generator], or dedup_probe_allow_weak_sandbox=true to accept "
                  "the risk. Static dedup tiers are unaffected.", posture)
        return False

    def _probe_digest(
        self,
        repo_dir: Path,
        generation_seed: int,
        probe_cfg,
        *,
        check_determinism: bool = True,
    ) -> str:
        """Digest of a small sandbox-drawn corpus under the shared round seed.

        With ``check_determinism`` the corpus is drawn twice and the digests
        must match — the same contract ``cascade verify`` holds miners to,
        enforced here so a generator that seeds from entropy cannot re-roll a
        fresh corpus per run. Raises :class:`CorpusError` on mismatch or on
        any generator failure.

        THREAT MODEL (deliberate): this executes untrusted generator code on
        the ORCHESTRATOR — previously pod-only — via the same hardened path
        the pods use (:func:`build_round_corpus` → ``run_in_sandbox``: netns,
        rlimits, static-guard blocklist; ``[generator] sandbox_mode =
        "container"`` reroutes to the docker/podman sandbox and is honored
        here too). On a production orchestrator set ``sandbox_strict = true``
        (refuse rather than degrade when netns is unavailable) and prefer
        container mode. See DEC-CA-0008.
        """
        first = build_round_corpus(
            repo_dir, generation_seed, probe_cfg, "cache_reuse",
            use_sandbox=self.use_sandbox,
            blocked=self.cfg.static_guard.blocked,
        )
        if check_determinism:
            second = build_round_corpus(
                repo_dir, generation_seed, probe_cfg, "cache_reuse",
                use_sandbox=self.use_sandbox,
                blocked=self.cfg.static_guard.blocked,
            )
            if first.digest != second.digest:
                raise CorpusError(
                    "generator is non-deterministic: two probe draws at the same "
                    "seed produced different corpora")
        return first.digest

    def _mark_heat_complete(
        self,
        base_seed: int,
        screened: list[ResolvedGenerator],
        finalists: list[ResolvedGenerator],
    ) -> None:
        """Drop ``work_root/<round_id>/heat_complete.json`` when the heat settles.

        The teardown signal for an external provisioner: once the field is
        screened, burned, and the finalists chosen, no heat-stage dispatch can
        occur for the rest of the round, so heat-tagged pods are safe to
        terminate while the final still runs (see docs/DEPLOY_PODS.md). Written
        atomically (tmp + rename) so a watcher never reads a torn file; the
        round_id equals the round's base_seed (the work-root subdir key).
        Best-effort: a write failure must never sink a round.
        """
        payload = {
            "round_id": str(base_seed),
            "screened": len(screened),
            "finalists": [c.hotkey for c in finalists],
        }
        try:
            out_dir = self.work_root / f"{base_seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "heat_complete.json.tmp"
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp.replace(out_dir / "heat_complete.json")
        except OSError as e:
            log.warning("could not write heat_complete marker for round=%s: %s",
                        base_seed, e)

    def _settled_finalists(
        self,
        base_seed: int,
        challengers: list[ResolvedGenerator],
    ) -> list[ResolvedGenerator] | None:
        """This round's already-settled finalist set, when its heat completed.

        The round-retry reuse guard (r47, 2026-08-28): a FINAL-stage failure
        sends ``run_forever`` back through :meth:`run_round` for the SAME round
        id — but the heat settle already burned every entrant
        (:meth:`_burn_hotkeys`), so the retry's re-derived eligibility finds
        them all consumed, fields nobody, and the round silently degenerates
        into a king-only walkover that discards the settled heat's finalists
        (r47 needed manual surgery on the submissions db to recover them).
        When ``work_root/<round_id>/heat_complete.json`` records a non-empty
        finalist set for THIS round, the retry reuses it directly instead.

        Scope is exactly the marker's finalist hotkeys for exactly this round:
        the burn set itself is never edited, so those hotkeys stay burned for
        every later round, and non-finalist entrants stay out (they had their
        screening shot in the settled heat).

        Returns the finalists in their settled (heat-rank) order — the order
        :meth:`run_round` stamps ``duel_rank`` from — or ``None`` for the old
        behaviour: no marker (first entry, or a retry from BEFORE the heat
        settled), an unreadable/malformed marker (warn and fall back), a
        marker for a different round, or a legitimately empty finalist set
        (a zero-finalist heat means the king-only walkover is correct).
        """
        path = self.work_root / f"{base_seed}" / "heat_complete.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None  # heat never settled for this round — normal entry
        except (OSError, ValueError) as e:
            log.warning("round=%s: heat_complete marker unreadable (%s); "
                        "re-deriving eligibility from scratch", base_seed, e)
            return None
        hotkeys = raw.get("finalists") if isinstance(raw, dict) else None
        if (not isinstance(raw, dict) or str(raw.get("round_id")) != str(base_seed)
                or not isinstance(hotkeys, list)
                or not all(isinstance(h, str) for h in hotkeys)):
            log.warning("round=%s: heat_complete marker malformed (%r); "
                        "re-deriving eligibility from scratch", base_seed, raw)
            return None
        if not hotkeys:
            return None  # heat settled with zero finalists — walkover is correct
        by_hotkey = {c.hotkey: c for c in challengers}
        matched = [by_hotkey[h] for h in hotkeys if h in by_hotkey]
        missing = [h for h in hotkeys if h not in by_hotkey]
        if missing:
            log.warning("round=%s retry: settled finalist(s) %s no longer resolve "
                        "from the commitment set; reusing the %d that do",
                        base_seed, ", ".join(missing), len(matched))
        if not matched:
            log.warning("round=%s retry: none of the settled finalists resolve; "
                        "re-deriving eligibility from scratch", base_seed)
            return None
        log.warning("round=%s retry: heat already settled — reusing finalist(s) %s "
                    "directly (their hotkeys are already burned; the burn still "
                    "stands for later rounds)",
                    base_seed, ", ".join(c.hotkey for c in matched))
        return matched

    # ── live round-stage reporting (status/round.json, presentational) ───────

    HEAT_PROGRESS_PUBLISH_SECONDS = 300.0

    def _publish_stage(
        self,
        stage: str,
        *,
        heat_done: int | None = None,
        heat_total: int | None = None,
        finalists: int | None = None,
    ) -> None:
        """Best-effort publish of the trainer-reported round stage.

        No-op unless ``publish_stage_status`` is on and a round context was set
        by :meth:`run_round`. Never raises: the doc is presentational (the
        dashboards' live stage strip) and a storage failure must never disturb
        the round it is describing.
        """
        if not self.publish_stage_status or self._stage_ctx is None:
            return
        from datetime import datetime

        from ..shared.chain_status import build_round_status, publish_round_status

        try:
            doc = build_round_status(
                round_id=self._stage_ctx["round_id"],
                epoch_start_block=self._stage_ctx["epoch_start_block"],
                stage=stage,
                as_of=datetime.now(UTC).isoformat(),
                heat_done=heat_done,
                heat_total=heat_total,
                finalists=finalists,
                warm_start=self._stage_ctx.get("warm_start"),
            )
            publish_round_status(self.manifest_store(), doc)
            self._stage_published_at = time.time()
        except Exception as e:  # noqa: BLE001 — presentational, never sinks a round
            log.debug("round-stage publish failed (ignored): %s", e)

    def _note_heat_progress(self, done: int, total: int) -> None:
        """Throttled heat-progress publish from the heat train loops (at most
        one write per :data:`HEAT_PROGRESS_PUBLISH_SECONDS`; the final count
        lands with the ``duel`` transition anyway)."""
        if not self.publish_stage_status or self._stage_ctx is None:
            return
        if time.time() - self._stage_published_at < self.HEAT_PROGRESS_PUBLISH_SECONDS:
            return
        self._publish_stage("heat", heat_done=done, heat_total=total)

    def _publish_heat_standings(
        self,
        heat: HeatResult | None,
        *,
        screened: int,
    ) -> None:
        """Publish the heat standings the moment the heat settles.

        The same standings ride the manifest, but that only reaches the public
        through a validator's receipt — after the duel trained AND was scored,
        hours later, and not at all for a round rejected at a gate. A miner's
        next submission deadline can pass in that gap, so the trainer mirrors
        them here instead (``status/heat.json`` + ``heats/``; see
        :mod:`cascade.shared.heat_status`).

        A no-screen round publishes too, carrying the reason: otherwise the live
        pointer would keep serving the PREVIOUS round's standings as this
        round's. Gated on ``publish_stage_status`` like the stage doc, and
        equally best-effort — presentational, unsigned, never weight-bearing, so
        a storage failure must not disturb the round it describes.
        """
        if not self.publish_stage_status or self._stage_ctx is None:
            return
        from datetime import datetime

        from ..shared.heat_status import (
            build_heat_status,
            publish_heat_status,
            update_heat_index,
        )

        skipped = list(self._round_skipped)
        n = max(0, self.cfg.round.finalist_cap)
        if screened == 0:
            reason = "no eligible challengers entered the round"
            if skipped:
                # An empty field with standing commitments is not silence — say
                # why the committed hotkeys were not eligible (r48: 328 burned
                # re-commits read as an unexplained "0 entrants").
                by_reason: dict[str, int] = {}
                for s in skipped:
                    key = str(s.get("reason", "unknown"))
                    by_reason[key] = by_reason.get(key, 0) + 1
                detail = ", ".join(f"{v} {k}" for k, v in sorted(by_reason.items()))
                reason += f" ({len(skipped)} commitment(s) skipped: {detail})"
        elif self.screen_fn is None:
            reason = "no screener configured; the field advanced by UID order"
        else:
            reason = (f"the field fit within the {n} finalist slot(s) — every entrant "
                      "advanced without spending heat compute")
        try:
            doc = build_heat_status(
                heat,
                round_id=self._stage_ctx["round_id"],
                epoch_start_block=self._stage_ctx["epoch_start_block"],
                as_of=datetime.now(UTC).isoformat(),
                screened=screened,
                netuid=self.cfg.subnet.netuid,
                no_screen_reason="" if heat is not None else reason,
                finalists=n,
                warm_start=self._stage_ctx.get("warm_start"),
                skipped=skipped,
            )
            store = self.manifest_store()
            publish_heat_status(store, doc)
            update_heat_index(store, doc)
            log.info("round=%s: published heat standings (%d entrants) to status/heat.json "
                     "+ heats/round-%s.json", self._stage_ctx["round_id"],
                     len(doc.get("entrants", ())), self._stage_ctx["round_id"])
        except Exception as e:  # noqa: BLE001 — presentational, never sinks a round
            log.warning("heat-standings publish failed (ignored): %s", e)

    # ── per-generator train (GPU + registry + S3 boundary) ───────────────────

    def _train_checkpoint(
        self,
        gen: ResolvedGenerator,
        seeds: RoundSeeds,
        contract: TrainingContractConfig,
        token_budget: int,
        out_dir: Path,
        *,
        log_role: str,
        warm_start_dir: Path | None = None,
    ) -> tuple[TrainResult, str, int, int]:
        """Fetch generator (registry) → build corpus → train into ``out_dir``,
        streaming per-step metrics to S3. No upload — the caller decides whether
        the checkpoint is uploaded (final) or thrown away after screening (heat).

        ``contract`` is the per-size training contract (the base recipe with this
        size's width/depth/digest/throughput); ``token_budget`` is its compute
        budget for this stage. ``warm_start_dir`` is a fetched promoted-init
        checkpoint dir: the run initialises from its weights instead of random
        (Cascade consumption — forwarded to the backend only when set, so
        custom BaseTrainers without the kwarg keep working random-init).
        Returns ``(result, corpus_digest, n_series, total_points)``. Raises on
        any failure.
        """
        gen_dir = out_dir.parent / "generator"
        try:
            fetch_from_hub(gen.ref, gen_dir, self.hub())
        except StorageError as e:
            status = _http_status_in_chain(e)
            if status in (401, 403, 404):
                # The MINER's repo, not our infra: a private or missing artifact
                # is the submitter's fault (Hippius Harbor projects must be
                # public for the trainer to pull them — see docs/MINER.md).
                raise CorpusError(
                    f"generator_artifact_unreachable: HTTP {status} for {gen.ref}"
                ) from e
            raise
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            "round=%s run=%s: fetched generator %s — building corpus + training "
            "(mode=%s, budget=%s point-passes) …",
            seeds.base_seed, log_role, gen.ref[:48],
            contract.corpus_mode, f"{token_budget:,}",
        )

        # Stream per-step metrics to S3 (best-effort: logging must never abort a
        # training run). ``log_role`` carries the size/heat tag so each run's log
        # lands at a distinct key (king-toto2-4m, challenger-toto2-22m, heat-<hk>).
        sink: LogSink | None = None
        try:
            sink = LogSink(self.logs_store(), round_id=str(seeds.base_seed), role=log_role)
        except Exception as e:  # noqa: BLE001
            log.warning("log sink unavailable (continuing without S3 logs): %s", e)
        # Optional live wandb mirror (observability only — the same per-step
        # records, so miners can watch this run train as it occurs). Best-effort:
        # disabled/unavailable ⇒ None, and every wandb call swallows its errors.
        wandb_sink = open_wandb_run(
            self.cfg.wandb,
            round_id=str(seeds.base_seed), role=log_role,
            hotkey=gen.hotkey, uid=gen.uid, size=contract.arch_preset,
            config={"corpus_mode": contract.corpus_mode, "token_budget": token_budget,
                    "contract_digest": contract_digest(contract)},
        )
        emitters = [s for s in (sink, wandb_sink) if s is not None]
        logger = (lambda record: [s.emit(record) for s in emitters]) if emitters else None

        # Host telemetry, taken HERE — after the generator is fetched but before
        # the corpus stream (and therefore the sandbox child) exists, so the
        # calibration bench measures the pod and not the submission competing
        # with it. It cannot eat the compute budget: max_train_seconds anchors at
        # the first training batch, and token_budget is a token count.
        host_facts = self._host_snapshot()
        if host_facts:
            for s in emitters:
                s.emit({"event": "host", "role": log_role, **host_facts})
            log.info(
                "round=%s run=%s host: %s",
                seeds.base_seed, log_role, host_summary_line(host_facts),
            )

        with open_round_stream(
            contract.corpus_mode,
            gen_dir, seeds.generation_seed, self.cfg.generator,
            token_budget=token_budget,
            use_sandbox=self.use_sandbox,
            blocked=self.cfg.static_guard.blocked,
            max_wall_seconds=contract.max_train_seconds,
            seed_mix=int(getattr(contract, "gen_seed_mix", 1) or 1),
        ) as rs:
            result = self.base_trainer.train(
                rs.series(),
                contract,
                training_seed=seeds.training_seed,
                token_budget=token_budget,
                out_dir=out_dir,
                logger=logger,
                **({"warm_start_dir": warm_start_dir} if warm_start_dir is not None else {}),
            )
            corpus_digest, n_series, total_points = rs.digest, rs.n_series, rs.total_points

        # The host facts are repeated on the summary row, not just left on their
        # own "host" record: the whole point is regressing realized throughput on
        # host capability, and that is a one-liner only when both live on the same
        # row. The standalone record is what a run that DIES before the summary
        # leaves behind — on the live wandb mirror and the stderr line, since the
        # S3 blob is only written at flush.
        summary = {"event": "summary", "role": log_role, "corpus_digest": corpus_digest,
                   "n_series": n_series, "total_points": total_points,
                   "train_seconds": result.train_seconds, **host_facts, **result.metrics}
        for s in emitters:
            s.emit(summary)
        if sink is not None:
            try:
                sink.flush()
            except Exception as e:  # noqa: BLE001
                log.warning("failed to flush S3 training logs: %s", e)
        if wandb_sink is not None:
            wandb_sink.finish()

        log.info(
            "round=%s run=%s hotkey=%s mode=%s n=%d points=%d digest=%s",
            seeds.base_seed, log_role, gen.hotkey, contract.corpus_mode,
            n_series, total_points, corpus_digest[:12],
        )
        # One parseable key=value telemetry line per run. TrainResult.metrics
        # never crosses the remote boundary (the worker's receipt is a
        # TrainedEntry, which carries no metrics — and the receipt protocol
        # stays as-is), so this line IS how a remote run's starvation/deadline
        # telemetry reaches the dispatch output: it lands on the worker's
        # stderr, which the orchestrator's SSH dispatch captures. Local runs
        # additionally feed the per-round roll-up (telemetry_rollup_line).
        m = result.metrics or {}
        if "deadline_hit" in m:
            log.info(
                "round=%s run=%s telemetry: deadline_hit=%s tokens_frac=%s "
                "data_wait_s=%s data_wait_frac=%s",
                seeds.base_seed, log_role, m.get("deadline_hit"),
                m.get("tokens_frac"), m.get("data_wait_s"), m.get("data_wait_frac"),
            )
        stage = "heat" if log_role.startswith("heat") else "final"
        self._round_telemetry[stage].append({**host_facts, **m})
        return result, corpus_digest, n_series, total_points

    def _host_snapshot(self) -> dict:
        """Host facts for the run about to start, per ``[telemetry]`` — ``{}`` when
        disabled or wholly unavailable.

        The device comes from the backend when it exposes one: ``BaseTrainer`` is a
        Protocol and only the reference implementation carries a ``.device``, so a
        custom backend benches on whatever torch finds rather than not at all.
        """
        tcfg = getattr(self.cfg, "telemetry", None)
        if tcfg is not None and not getattr(tcfg, "host_probe", True):
            return {}
        run_bench = getattr(tcfg, "host_bench", True) if tcfg is not None else True
        try:
            return host_snapshot(
                device=getattr(self.base_trainer, "device", None),
                run_bench=bool(run_bench),
            )
        except Exception as e:  # noqa: BLE001 — telemetry must never abort a round
            log.warning("host telemetry unavailable (continuing): %s", e)
            return {}

    def train_one(
        self,
        gen: ResolvedGenerator,
        role: str,
        seeds: RoundSeeds,
        block: int,
        *,
        contract: TrainingContractConfig | None = None,
        token_budget: int | None = None,
        repo_suffix: str = "",
        heat: bool = False,
        warm_start_ref: str | None = None,
    ) -> TrainedEntry:
        """Train one generator at one size, upload its checkpoint, return the receipt.

        ``contract`` defaults to the primary (smallest) size; ``token_budget`` to
        that size's full ``train_tokens`` (pass a cheaper budget for a heat screen).
        The checkpoint is uploaded to a size-tagged registry repo
        (``ckpt-r<seed>-<role>-<size><repo_suffix>``) and the entry carries the
        ``size`` tag so the validator can pair king and challenger per size before
        combining their scores. ``repo_suffix`` disambiguates otherwise-identical
        repos (same seed/role/size) so parallel runs — several heat challengers, or
        finalists>1 at one size — never overwrite each other's checkpoint.

        ``heat`` tags this run as a heat SCREEN rather than a final, so its S3 and
        wandb telemetry lands at a distinct ``heat-<hotkey>`` key — matching the
        local heat path (:meth:`_heat_train`). Without it a remote heat and the
        final for the SAME challenger at the primary size share one ``<role>-<size>``
        key, so their S3 logs collide and their wandb runs are indistinguishable
        (and, with the deterministic run id, collapse into one). The receipt's
        ``role`` is unaffected — the manifest still pairs king vs challenger.

        Raises on any failure; the caller decides whether a failed challenger
        simply doesn't qualify (it does) or a failed king aborts the round (it
        does — there's nothing to defend against).
        """
        contract = contract if contract is not None else self.cfg.training.primary_size
        token_budget = token_budget if token_budget is not None else contract.train_tokens
        size = contract.arch_preset
        out_dir = self.work_root / f"{seeds.base_seed}" / size / f"{role}{repo_suffix}" / "checkpoint"
        log_role = f"heat-{gen.hotkey}" if heat else f"{role}-{size}"
        # Retry-without-retrain: if a PRIOR run of this exact job already trained
        # a complete checkpoint here (marker written post-train, pre-upload) and
        # only the upload failed, reuse it instead of burning the budget again.
        # Sound because training is deterministic — same generator/seeds/contract
        # rederives byte-identical weights (r44 2026-08-27: an upload flap after
        # a finished 3h final run triggered a full retrain of the SAME bytes).
        reused = self._reusable_checkpoint(out_dir, contract, gen)
        if reused is not None:
            log.warning(
                "round=%s %s %s: COMPLETE checkpoint already at %s (prior run "
                "trained it; only its upload failed) — skipping straight to "
                "upload", seeds.base_seed, role, gen.hotkey, out_dir,
            )
            corpus_digest = str(reused["corpus_digest"])
            gpu_name = str(reused.get("gpu_name", ""))
        else:
            # Cascade warm-start: fetch the pinned promoted init (content-addressed;
            # the OCI digest verifies the bytes). A fetch failure RAISES — the run
            # must never silently fall back to random init (DEC-CA-0005).
            ws_dir = self._fetch_checkpoint_dir(warm_start_ref) if warm_start_ref else None
            result, corpus_digest, _, _ = self._train_checkpoint(
                gen, seeds, contract, token_budget, out_dir, log_role=log_role,
                warm_start_dir=ws_dir,
            )
            gpu_name = str(result.metrics.get("gpu_name", ""))
            self._write_train_complete_marker(
                out_dir, contract, gen, corpus_digest=corpus_digest, gpu_name=gpu_name,
            )

        ckpt_repo = f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-{role}-{size}{repo_suffix}"
        # Hub is priority-one; mirror to HF only if the Hub is down (keeps a round
        # alive through a Hub outage instead of failing the checkpoint upload).
        up = upload_dir_to_hub_or_hf(
            out_dir, ckpt_repo, self.hub(),
            hf_repo=self._hf_ckpt_repo(ckpt_repo),
        )
        return TrainedEntry(
            miner_hotkey=gen.hotkey,
            miner_uid=gen.uid,
            role=role,
            gen_ref=gen.ref,
            trained_pointer=format_trained_pointer(up.ref.immutable_ref),
            corpus_digest=corpus_digest,
            train_block=block,
            gpu_name=gpu_name,
            size=size,
        )

    # ── retry-without-retrain (the .train_complete marker) ────────────────────

    TRAIN_COMPLETE_MARKER = ".train_complete"

    def _write_train_complete_marker(
        self, out_dir: Path, contract: TrainingContractConfig,
        gen: ResolvedGenerator, *, corpus_digest: str, gpu_name: str,
    ) -> None:
        """Stamp ``out_dir`` as a COMPLETE training product (written after
        training returns, before the upload is attempted) so an upload-failure
        retry can skip the retrain. Atomic (tmp + rename); best-effort — a
        marker miss just costs the old retrain-on-retry behaviour."""
        try:
            payload = json.dumps({
                "contract_digest": contract_digest(contract),
                "gen_ref": gen.ref,
                "corpus_digest": corpus_digest,
                "gpu_name": gpu_name,
            }, sort_keys=True)
            tmp = out_dir / (self.TRAIN_COMPLETE_MARKER + ".tmp")
            tmp.write_text(payload)
            tmp.rename(out_dir / self.TRAIN_COMPLETE_MARKER)
        except Exception as e:  # noqa: BLE001 — never sink a finished run
            log.warning("train-complete marker write failed for %s: %s", out_dir, e)

    def _reusable_checkpoint(
        self, out_dir: Path, contract: TrainingContractConfig, gen: ResolvedGenerator,
    ) -> dict | None:
        """The marker payload iff ``out_dir`` holds a complete checkpoint for
        EXACTLY this job (same contract digest + same generator ref, weights
        present) — else None. ``out_dir`` is already round/size/role-scoped, so
        the digest+ref match is a belt-and-braces identity check, not the only
        one. A torn/stale/mismatched marker reads as None (retrain, the safe
        direction)."""
        marker = out_dir / self.TRAIN_COMPLETE_MARKER
        try:
            if not marker.exists() or not (out_dir / "weights.safetensors").exists():
                return None
            payload = json.loads(marker.read_text())
            if (payload.get("contract_digest") == contract_digest(contract)
                    and payload.get("gen_ref") == gen.ref
                    and payload.get("corpus_digest")):
                return payload
        except Exception as e:  # noqa: BLE001 — unreadable marker ⇒ retrain
            log.warning("train-complete marker unreadable at %s (%s); retraining",
                        out_dir, e)
        return None

    def _load_warm_start(self, *, epoch_index: int | None = None) -> tuple[str, str] | None:
        """This round's promoted init as ``(checkpoint pointer, size)``, or
        ``None`` when no promotion has fired (file absent) or consumption isn't
        wired.

        A multi-member pointer file (DEC-CA-0013) carries the live generation's
        ``members`` list; the round trains from the epoch-rotation member
        (``epoch_index % len(members)``; index ``None`` ⇒ the first member). A
        legacy single-pointer file reads as a one-member set. Validators accept
        any live member, so the rotation itself is trainer policy.

        A pointer file that EXISTS but is unusable (unreadable JSON, missing or
        malformed ``checkpoint_id``) RAISES and aborts the round: once a
        promotion is live, training must never silently fall back to random
        init — the round's reproducibility contract pins the init
        (DEC-CA-0005). ``size`` defaults to the primary arch preset for pointer
        files written before the field existed.

        With the promotion engine wired it is the SINGLE source of the
        allocation policy — a future policy change edits one place, not two.
        But an engine with NO members does not short-circuit: it falls through
        to the pointer file, so an engine whose state was lost while a live
        pointer file survives (readable ⇒ trains from it; unreadable ⇒ raises)
        can never silently turn a live promotion back into random init."""
        if self.promotion is not None:
            picked = self.promotion.init_for_epoch(int(epoch_index or 0))
            if picked is not None:
                ref, size = picked
                if not ref or parse_trained_pointer(ref) is None:
                    raise RuntimeError(
                        f"promotion engine returned no usable checkpoint_id: {ref!r}")
                return ref, size or self.cfg.training.primary_size.arch_preset
        if self.warm_start_path is None:
            return None
        p = Path(self.warm_start_path)
        if not p.is_file():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — a live-but-broken pin must abort, not degrade
            raise RuntimeError(f"warm-start pointer {p} unreadable: {e}") from e
        members = obj.get("members") or []
        if members:
            m = members[int(epoch_index or 0) % len(members)]
            ref = str(m.get("checkpoint_id") or "")
            size = str(m.get("size") or "")
        else:
            ref = str(obj.get("checkpoint_id") or "")
            size = str(obj.get("size") or "")
        if not ref or parse_trained_pointer(ref) is None:
            raise RuntimeError(
                f"warm-start pointer {p} carries no usable checkpoint_id: {ref!r}"
            )
        size = size or self.cfg.training.primary_size.arch_preset
        return ref, size

    def _fetch_checkpoint_dir(self, trained_pointer: str) -> Path:
        """Fetch a just-trained checkpoint from the registry to a local dir (the
        OCI digest self-verifies the bytes). Uniform for local and remote training,
        since every final checkpoint is uploaded to the registry."""
        ref = parse_trained_pointer(trained_pointer)
        if ref is None:
            raise ValueError(f"malformed trained_pointer: {trained_pointer!r}")
        from ..shared.hippius import HubRef

        dest = self.work_root / "_bench_ckpts" / HubRef.parse(ref).digest.replace(":", "-")
        fetch_from_hub(ref, dest, self.hub())
        return dest

    # ── Cascade post-publish duel bench (signed bench report) ────────────────

    def will_run_post_publish_bench(self) -> bool:
        """Whether this trainer benches the duel after publishing — [scoring]
        cascade_enabled plus a wired eval path. Everything bench-related (the
        report, the wandb pair, the provisioner's teardown-hold marker) keys off
        this one predicate so nothing fires while Cascade is off."""
        return bool(self.cfg.scoring.cascade_enabled) and (
            self.bench_eval_fn is not None or self.cascade_bench_plan is not None
        )

    def _bench_pending_path(self, round_id: str) -> Path:
        return self.work_root / str(round_id) / "bench_pending.json"

    def _bench_complete_path(self, round_id: str) -> Path:
        return self.work_root / str(round_id) / "bench_complete.json"

    def _mark_bench_pending(self, round_id: str) -> None:
        """Drop ``work_root/<round_id>/bench_pending.json`` BEFORE the manifest
        publishes — the provisioner's signal to hold the final pod's teardown
        for the post-publish bench (see provision.policy.bench_hold_active).
        Written pre-publish because the teardown trigger IS the manifest: a
        marker written after it races the very sweep it exists to pause.
        Atomic (tmp + rename) and best-effort, like the heat marker."""
        try:
            out_dir = self.work_root / str(round_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "bench_pending.json.tmp"
            tmp.write_text(json.dumps({"round_id": str(round_id)}, sort_keys=True),
                           encoding="utf-8")
            tmp.replace(self._bench_pending_path(round_id))
        except OSError as e:
            log.warning("could not write bench_pending marker for round=%s: %s", round_id, e)

    def _mark_bench_complete(self, round_id: str, *, uploaded: bool) -> None:
        """Release the teardown hold: record the outcome in ``bench_complete.json``
        and remove the pending marker. Runs on success AND failure — a failed
        bench must free the pod immediately, not ride out the hold cap."""
        try:
            out_dir = self.work_root / str(round_id)
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_dir / "bench_complete.json.tmp"
            tmp.write_text(json.dumps({"round_id": str(round_id), "uploaded": bool(uploaded)},
                                      sort_keys=True), encoding="utf-8")
            tmp.replace(self._bench_complete_path(round_id))
            self._bench_pending_path(round_id).unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not write bench_complete marker for round=%s: %s", round_id, e)

    def _release_stale_bench_hold(self, round_id: str) -> None:
        """Free a teardown hold whose bench is never coming.

        A restart between publish and bench completion kills the bench thread
        with the process, but its ``bench_pending.json`` stays armed — and the
        restarted trainer skips the already-published round, so nothing would
        release the provisioner's hold until the cap. Called from the
        restart re-entry skip path; a marker with its completion already
        recorded is left alone."""
        try:
            if (self._bench_pending_path(round_id).is_file()
                    and not self._bench_complete_path(round_id).is_file()):
                log.warning("round=%s: bench_pending marker with no completed bench "
                            "(restart mid-bench?); releasing the teardown hold",
                            round_id)
                self._mark_bench_complete(round_id, uploaded=False)
        except OSError:
            pass

    def run_post_publish_bench(self, manifest: TrainingManifest) -> object | None:
        """Bench BOTH final-duel checkpoints and publish the round's signed bench
        report — strictly AFTER :meth:`publish` (validators are already scoring;
        nothing here can delay or modify the round). Best-effort throughout:
        returns the published :class:`~cascade.shared.bench_report.BenchReport`
        or ``None``, never raises, and always releases the provisioner's
        teardown hold on exit. A no-op (no markers touched) when the bench is
        not armed — the marker lifecycle exists only where the hold does."""
        if not self.will_run_post_publish_bench():
            return None
        # Snapshot the round's final-pod assignments NOW, at thread start —
        # still inside round N. The scratch shadow dispatches HOURS later, by
        # which time run_round may have started round N+1 and repopulated
        # _final_role_hosts with the SAME (role, size, hotkey) key pointing at
        # N+1's live final pod; a late lookup would land an hours-long
        # training run on the pod running the next round's consensus-relevant
        # final.
        final_hosts = dict(self._final_role_hosts)
        report = None
        try:
            try:
                report = self._post_publish_bench(manifest)
            except Exception as e:  # noqa: BLE001 — bench telemetry must never fail a round
                log.warning("round=%s: post-publish bench failed (ignored): %s",
                            manifest.round_id, e)
            # Promotion candidates (DEC-CA-0013): both duel checkpoints' scores
            # feed the engine's candidate log (thread-safe; this runs on the
            # bench thread). Guarded — candidate bookkeeping must never fail
            # the bench. Recorded BEFORE the scratch shadow below: the shadow
            # is an hours-long telemetry leg, and a promotion must never wait
            # on telemetry.
            if report is not None and self.promotion is not None:
                try:
                    self.promotion.record_bench(manifest, report)
                except Exception as e:  # noqa: BLE001
                    log.warning("round=%s: promotion candidate recording failed (ignored): %s",
                                manifest.round_id, e)
            # Shadow scratch control (DEC-CA-0014 Stage 1): every M-th
            # warm-started round, train the king's generator from scratch and
            # publish the labeled telemetry report. INSIDE the try so it runs
            # under the provisioner's teardown hold (it needs the final pod),
            # and after the duel bench + candidate recording so it never
            # delays the reign log's numbers. Never raises; a duel-bench miss
            # doesn't cancel it (the scratch curve alone is still a data
            # point).
            self._run_scratch_shadow(manifest, report, final_hosts=final_hosts)
        finally:
            self._mark_bench_complete(manifest.round_id, uploaded=report is not None)
        return report

    # ── shadow scratch control (DEC-CA-0014 Stage 1) ─────────────────────────

    def _scratch_shadow_due(self, manifest: TrainingManifest) -> bool:
        """Whether this round owes a scratch-shadow leg: ``[telemetry]
        scratch_shadow_every_rounds`` armed, the round actually trained
        warm-started (on a random-init round every run already IS the scratch
        control), the epoch grid lands on the cadence, and a train+bench path
        is wired. Pure telemetry predicate — nothing consensus-visible reads
        it."""
        every = int(getattr(self.cfg.telemetry, "scratch_shadow_every_rounds", 0) or 0)
        if every <= 0:
            return False
        if not manifest.warm_start_ckpt:
            log.info("round=%s: scratch shadow skipped — round trained from "
                     "random init (nothing to contrast)", manifest.round_id)
            return False
        # Same epoch index the warm-start rotation uses, derived from the
        # manifest alone so a restart (or the bench thread outliving the round
        # context) computes the identical cadence.
        eb = effective_epoch_blocks(self.cfg.round, int(manifest.created_block))
        epoch_idx = int(manifest.created_block) // eb
        if epoch_idx % every != 0:
            return False
        remote_ok = bool(self.remote_hosts and self.trainer_spec
                         and self.cascade_bench_plan is not None)
        local_ok = self.bench_eval_fn is not None
        if not (remote_ok or local_ok):
            log.warning("round=%s: scratch shadow due but no train+bench path "
                        "wired (remote hosts+plan or local bench_eval_fn); skipped",
                        manifest.round_id)
            return False
        return True

    def _run_scratch_shadow(self, manifest: TrainingManifest,
                            duel_report: object | None = None,
                            final_hosts: dict | None = None) -> object | None:
        """Train the king's generator FROM SCRATCH under the round's identical
        contract + seeds, bench it, and publish the labeled telemetry report
        (``benchmarks/scratch/round-<id>.json`` + trend index) beside the
        round's signed bench report. DEC-CA-0014 Stage 1.

        TELEMETRY ONLY, enforced structurally: the scratch entry is never a
        manifest entry, never a ``BenchReport`` entry (validators parse that
        schema on the promotion-provenance path), and is never handed to
        ``promotion.record_bench`` — the promotion candidate pool cannot see
        it. Best-effort throughout: returns the published report or ``None``,
        never raises into the bench thread.

        Runs strictly post-publish on the king's (now idle) final pod, under
        the provisioner's bench hold — arming this requires sizing ``[eval]
        bench_hold_max_hours`` for duel bench + a full training leg + one more
        bench (see the DEC-CA-0014 node's Stage-1 budget note).
        """
        try:
            if not self._scratch_shadow_due(manifest):
                return None
            from ..shared.scratch_report import (
                ScratchBenchReport,
                bench_geomean,
                dump_scratch_report,
                publish_scratch_report,
                sign_scratch_report,
                update_scratch_index,
            )

            primary = self.cfg.throne_contracts()[0]
            size = primary.arch_preset
            king = next((e for e in manifest.entries
                         if e.role == "king" and (e.size or size) == size), None)
            if king is None:
                log.warning("round=%s: scratch shadow skipped — no king entry "
                            "at the primary size", manifest.round_id)
                return None
            gen = ResolvedGenerator(hotkey=king.miner_hotkey, uid=king.miner_uid,
                                    ref=king.gen_ref)
            seeds = RoundSeeds.derive(int(manifest.round_id), self.cfg.training)
            log.info("round=%s: scratch shadow leg starting — king generator %s "
                     "from RANDOM INIT (lineage init this round: %s)",
                     manifest.round_id, king.gen_ref[:48], manifest.warm_start_ckpt[:48])

            if self.remote_hosts and self.trainer_spec and self.cascade_bench_plan is not None:
                pointer, scores = self._scratch_shadow_remote(
                    manifest, gen, primary, seeds, final_hosts=final_hosts)
            else:
                pointer, scores = self._scratch_shadow_local(
                    manifest, gen, primary, seeds)
            if scores is None:
                log.warning("round=%s: scratch shadow produced no bench scores; "
                            "no scratch report published", manifest.round_id)
                return None

            king_bench = None
            if duel_report is not None:
                king_bench = duel_report.entry_for("king", size=size)
            report = ScratchBenchReport(
                round_id=manifest.round_id,
                created_block=manifest.created_block,
                gen_ref=king.gen_ref,
                miner_hotkey=king.miner_hotkey,
                miner_uid=king.miner_uid,
                trained_pointer=pointer or "",
                scores=scores,
                warm_start_ckpt=manifest.warm_start_ckpt,
                generation=int(getattr(self.promotion, "generation", 0) or 0),
                king_pointer=(king_bench.trained_pointer if king_bench else ""),
                king_scores=(king_bench.scores if king_bench else None),
            )
            if self.wallet is not None:
                report = sign_scratch_report(report, self.wallet)
            store = self.manifest_store()
            key = publish_scratch_report(store, dump_scratch_report(report),
                                         manifest.round_id)
            try:
                update_scratch_index(store, report)
            except Exception as e:  # noqa: BLE001 — the index is presentational
                log.warning("scratch index update failed (report published): %s", e)
            gap = (f"{bench_geomean(scores) - bench_geomean(king_bench.scores):+.4f}"
                   if king_bench else "n/a (no king bench)")
            log.info("published scratch report round=%s scratch_geomean=%.4f "
                     "lineage_gap=%s signed=%s → s3://%s/%s",
                     manifest.round_id, bench_geomean(scores), gap,
                     report.signature is not None,
                     self.cfg.storage.manifest_bucket, key)
            self._log_scratch_wandb(report)
            return report
        except Exception as e:  # noqa: BLE001 — shadow telemetry must never raise
            log.warning("round=%s: scratch shadow failed (ignored): %s",
                        manifest.round_id, e)
            return None

    def _scratch_shadow_remote(self, manifest: TrainingManifest, gen: ResolvedGenerator,
                               primary, seeds: RoundSeeds,
                               final_hosts: dict | None = None) -> tuple[str | None, object | None]:
        """Remote scratch leg on the king's final pod, through the PINNED
        worker image's existing CLI — no pod-side code change, so this ships
        trainer-side unilaterally (the whole point of Stage 1).

        Dispatch shape, and why: ``role="king"`` with ``repo_suffix="-scratch"``
        keeps the checkpoint repo distinct (``ckpt-r<seed>-king-<size>-scratch``)
        while ``--train-hours <target_train_hours>`` routes through the
        worker's screen path at the FULL budget — ``for_hours(target)`` yields
        the byte-identical token count, and the wall guard
        ``min(max(guard_factor×target, floor), max_train_seconds)`` equals
        ``max_train_seconds`` at the shipped ``heat_guard_factor >= 1.0``. The
        screen path is what gives the run its own telemetry key
        (``heat-<king_hotkey>``, a key a king never otherwise uses) instead of
        colliding with the real king's ``king-<size>`` S3 log + wandb run. Two
        accepted asymmetries, both telemetry-grade: the worker skips
        ``assert_train_image`` on the screen path (the pod was launched from
        the pinned image regardless), and the run's log label says heat. No
        ``warm_start_ref`` ⇒ random init.
        """
        from .bench_hook import run_post_round_benchmark
        from .remote import RemoteDispatcher, pod_lane_count

        size = primary.arch_preset
        # ONLY the snapshot taken at bench-thread start may name the pod: a
        # live _final_role_hosts / _hosts_for("final") lookup this many hours
        # after publish can resolve to the NEXT round's freshly-dispatched
        # final pod (same (role, size, hotkey) key, new fleet) and land a
        # full training leg on top of a consensus-relevant run. No snapshot
        # entry ⇒ skip the shadow this round — a missed telemetry point beats
        # a contended final.
        host = (final_hosts or {}).get(("king", size, gen.hotkey))
        if host is None:
            log.warning("round=%s: scratch shadow skipped — no snapshotted "
                        "king final pod for this round (restart mid-round, or "
                        "a fully local final)", manifest.round_id)
            return None, None
        disp = RemoteDispatcher(
            trainer_spec=self.trainer_spec,
            timeout_seconds=self.remote_timeout_seconds,
            extra_forward_env=self._pod_extra_forward_env(),
        )
        entry = disp.dispatch(
            host, gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey, role="king",
            base_seed=seeds.base_seed, block=int(manifest.created_block),
            arch_preset=size, train_hours=primary.target_train_hours,
            repo_suffix="-scratch", warm_start_ref=None,
            lane_count=pod_lane_count(host, [host]),
        )
        from ..eval.benchmarks import extract_bench_scores

        bench = run_post_round_benchmark(
            host, manifest.round_id, size, self.cascade_bench_plan,
            work_root=self.work_root, role="king-scratch",
        )
        scores = extract_bench_scores(bench) if bench is not None else None
        return entry.trained_pointer, (BenchScores(**scores) if scores else None)

    def _scratch_shadow_local(self, manifest: TrainingManifest, gen: ResolvedGenerator,
                              primary, seeds: RoundSeeds) -> tuple[str | None, object | None]:
        """Local scratch leg (single-box / testnet): train on this box with a
        first-class telemetry label (``scratch-king-<size>``; the local path
        controls ``log_role`` directly, unlike the pinned remote worker),
        upload the checkpoint, bench via ``bench_eval_fn``."""
        size = primary.arch_preset
        out_dir = (self.work_root / f"{seeds.base_seed}" / size
                   / "king-scratch" / "checkpoint")
        result, _, _, _ = self._train_checkpoint(
            gen, seeds, primary, primary.train_tokens, out_dir,
            log_role=f"scratch-king-{size}", warm_start_dir=None,
        )
        pointer = None
        try:
            ckpt_repo = f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-king-{size}-scratch"
            up = upload_dir_to_hub_or_hf(result.local_dir, ckpt_repo, self.hub(),
                                         hf_repo=self._hf_ckpt_repo(ckpt_repo))
            pointer = format_trained_pointer(up.ref.immutable_ref)
        except Exception as e:  # noqa: BLE001 — the bench numbers matter more than the upload
            log.warning("round=%s: scratch checkpoint upload failed (benching "
                        "the local dir anyway): %s", manifest.round_id, e)
        scores = self.bench_eval_fn(result.local_dir) if self.bench_eval_fn else None
        return pointer, scores

    def _log_scratch_wandb(self, report: object) -> None:
        """Mirror the scratch-vs-lineage pair to wandb — same swallow-everything
        contract as every other observability path."""
        try:
            if not getattr(self.cfg.wandb, "enabled", False):
                return
            from ..shared.scratch_report import bench_geomean

            sink = open_wandb_run(
                self.cfg.wandb, round_id=str(report.round_id),
                role="scratch-shadow", hotkey=report.miner_hotkey,
                uid=report.miner_uid, size="",
            )
            if sink is None:
                return
            record = {
                "event": "scratch_shadow", "generation": report.generation,
                "scratch_geomean": bench_geomean(report.scores),
            }
            if report.king_scores is not None:
                king_gm = bench_geomean(report.king_scores)
                record["king_geomean"] = king_gm
                record["gap"] = record["scratch_geomean"] - king_gm
            sink.emit(record)
            sink.finish()
        except Exception as e:  # noqa: BLE001 — wandb must never disturb the shadow
            log.debug("wandb scratch-shadow log failed (continuing): %s", e)

    def _receipt_king(self) -> str | None:
        """The current king per the validators' signed receipt trail, or the
        last king a receipt named when none is readable right now (``None``
        only before any receipt was ever read). This is the prompt dethrone
        signal the reign clock needs: validators reset their clocks at the
        dethrone verdict, while the on-chain incentive (the caller's fallback
        king source) lags it 1-2 epochs. The last-known value is STICKY across
        transient fetch failures — during the lag window a blip that fell back
        to the stale incentive king would flap the engine's king view and
        reset the reign clock twice. Best-effort; never raises."""
        try:
            from ..shared.hippius import RECEIPT_LATEST_KEY, receipt_latest_key
            from ..shared.receipt import load_receipt, verify_receipt_signature

            anchor = self.cfg.manifest.validator_hotkey
            store = self.manifest_store()
            for key in (receipt_latest_key(anchor), RECEIPT_LATEST_KEY):
                try:
                    receipt = load_receipt(store.get_text(key))
                except Exception:  # noqa: BLE001 — absent/unreadable ⇒ next candidate
                    continue
                if anchor and not verify_receipt_signature(receipt, anchor):
                    continue
                v = receipt.verdict
                if receipt.status == "scored" and v is not None and v.king_hotkey:
                    self._last_receipt_king = str(v.king_hotkey)
                    return self._last_receipt_king
        except Exception:  # noqa: BLE001 — the sticky/incentive fallbacks cover a miss
            pass
        return self._last_receipt_king

    def _seed_promotion_reign(self) -> None:
        """Deploy-time backfill (DEC-CA-0013): count the rounds the current
        king has ALREADY reigned before this engine existed.

        Runs once, at the first boundary where the engine has never seen a
        king — after that the engine's own persisted clock owns the count.
        Validators anchored their clocks at the dethrone verdict, possibly
        rounds ago; an engine anchoring "now" would fire the promotion
        ``reign_threshold`` rounds later than the fleet's ripeness check
        requires it to be able to fire. The reign is reconstructed from the
        same signed artifacts validators used: the receipt index's unbroken
        tail of scored rounds held by the current king (``reign_tail``) seeds
        the clock, and each reign round's published trainer-signed bench
        report replays through :meth:`TrainerPromotion.record_bench` to
        rebuild the candidate log — exactly the state the engine would hold
        had it been running since the dethrone. Rounds whose bench report is
        unpublished contribute no candidates (the same gap the validator's
        reign log has). Best-effort: any miss falls back to ``note_round``'s
        anchor-at-this-boundary behaviour."""
        if self.promotion is None or getattr(self.promotion, "king_hotkey", None) is not None:
            return
        if getattr(self.promotion, "reign_start_block", None) is not None:
            return
        try:
            from ..shared.bench_report import (
                bench_report_key,
                load_bench_report,
                verify_bench_report_signature,
            )
            from ..shared.hippius import RECEIPT_INDEX_KEY
            from ..shared.manifest import load_manifest, verify_signature
            from .promotion import reign_tail

            store = self.manifest_store()
            idx = json.loads(store.get_text(RECEIPT_INDEX_KEY))
            tail = reign_tail(idx.get("rounds") or [],
                              self.cfg.manifest.validator_hotkey)
            if tail is None:
                return
            king, start_block, round_ids = tail
            if not self.promotion.seed_reign(king, start_block):
                return
            added = 0
            for rid in round_ids:  # oldest → newest
                try:
                    manifest = load_manifest(
                        store.get_text(manifest_round_key(rid)))
                    report = load_bench_report(
                        store.get_text(bench_report_key(rid)))
                except Exception:  # noqa: BLE001 — no report yet ⇒ no candidates
                    continue
                trainer_hotkey = self.cfg.manifest.trainer_hotkey
                if trainer_hotkey and not verify_bench_report_signature(
                        report, trainer_hotkey):
                    log.warning("promotion backfill: bench report for round=%s "
                                "fails signature vs pinned trainer hotkey; skipped", rid)
                    continue
                if trainer_hotkey and not verify_signature(manifest, trainer_hotkey):
                    log.warning("promotion backfill: manifest for round=%s fails "
                                "signature vs pinned trainer hotkey; skipped", rid)
                    continue
                added += self.promotion.record_bench(manifest, report)
            log.info("promotion: deploy backfill — king %s reign counted from "
                     "block %d (%d round(s), %d candidate(s) recovered)",
                     king[:12], start_block, len(round_ids), added)
        except Exception as e:  # noqa: BLE001 — never sink a round on backfill
            log.warning("promotion: deploy backfill failed (engine anchors at "
                        "this boundary instead): %s", e)

    def _replay_reign_bench_reports(self) -> None:
        """Re-ingest the CURRENT reign's published bench reports at the
        boundary. The candidate pool otherwise only sees reports the
        in-process bench thread publishes — an out-of-band report (a mop-up
        after a failed leg, a report that landed while the trainer was down)
        never yields candidates, and a promotion then fires off an incomplete
        reign (observed 2026-08-18: r26's better leg, 0.64149, missed the
        gen-3 member set because its mop-up report was published externally).

        Cheap and idempotent by construction: ``record_bench`` dedupes on
        ``trained_pointer`` and gates on the live generation, so replaying
        every reign round each boundary is a handful of small S3 reads a day
        that add nothing when nothing is missing. Runs BEFORE
        ``maybe_promote`` so a fire-time selection sees the full reign.
        Same signature discipline as the deploy backfill: both the report
        and the manifest must verify against the pinned trainer hotkey."""
        if self.promotion is None or getattr(self.promotion, "king_hotkey", None) is None:
            return
        try:
            from ..shared.bench_report import (
                bench_report_key,
                load_bench_report,
                verify_bench_report_signature,
            )
            from ..shared.hippius import RECEIPT_INDEX_KEY
            from ..shared.manifest import load_manifest, verify_signature
            from .promotion import reign_tail

            store = self.manifest_store()
            idx = json.loads(store.get_text(RECEIPT_INDEX_KEY))
            tail = reign_tail(idx.get("rounds") or [],
                              self.cfg.manifest.validator_hotkey)
            if tail is None:
                return
            _king, _start_block, round_ids = tail
            trainer_hotkey = self.cfg.manifest.trainer_hotkey
            added = 0
            for rid in round_ids:
                try:
                    manifest = load_manifest(
                        store.get_text(manifest_round_key(rid)))
                    report = load_bench_report(
                        store.get_text(bench_report_key(rid)))
                except Exception:  # noqa: BLE001 — no report yet ⇒ nothing to replay
                    continue
                if trainer_hotkey and not verify_bench_report_signature(
                        report, trainer_hotkey):
                    log.warning("promotion replay: bench report for round=%s "
                                "fails signature vs pinned trainer hotkey; skipped", rid)
                    continue
                if trainer_hotkey and not verify_signature(manifest, trainer_hotkey):
                    log.warning("promotion replay: manifest for round=%s fails "
                                "signature vs pinned trainer hotkey; skipped", rid)
                    continue
                added += self.promotion.record_bench(manifest, report)
            if added:
                log.info("promotion: boundary replay recovered %d candidate(s) "
                         "from published bench report(s)", added)
        except Exception as e:  # noqa: BLE001 — replay must never sink a round
            log.warning("promotion: bench-report replay failed (ignored): %s", e)

    def _flush_pending_promotion(self, round_id: str) -> None:
        """Publish a fired-but-unpublished promotion record, if one is pending.
        Guarded and idempotent — called at the round boundary AND right before
        the manifest publishes, so a transient store outage at fire time heals
        within the same round instead of costing the fleet the whole round."""
        if self.promotion is None:
            return
        try:
            pending = self.promotion.unpublished_record()
            if pending is not None:
                self._publish_promotion_record(pending)
                self.promotion.mark_record_published()
        except Exception as e:  # noqa: BLE001 — stays pending, retried next flush
            log.warning("promotion record publish failed for round=%s "
                        "(stays pending): %s", round_id, e)

    def _publish_promotion_record(self, record: object) -> None:
        """Sign and publish a fired promotion's record to the manifest bucket
        (``promotions/gen-<n>.json`` + locator index) — the declaration
        validators verify the new generation against. Unsigned publishes are
        possible (no wallet: offline runs) but rejected by verifying
        validators, exactly like an unsigned bench report."""
        from ..shared.promotion import (
            dump_promotion_record,
            publish_promotion_record,
            sign_promotion_record,
        )

        if self.wallet is not None:
            record = sign_promotion_record(record, self.wallet)
        else:
            log.warning("publishing an UNSIGNED promotion record (no wallet); "
                        "validators will reject the new generation")
        key = publish_promotion_record(
            self.manifest_store(), dump_promotion_record(record), record.generation)
        log.info("published promotion record generation=%d members=%d signed=%s → s3://%s/%s",
                 record.generation, len(record.members),
                 record.signature is not None, self.cfg.storage.manifest_bucket, key)

    def _post_publish_bench(self, manifest: TrainingManifest) -> object | None:
        from ..shared.bench_report import (
            BenchEntry,
            BenchReport,
            dump_bench_report,
            publish_bench_report,
            sign_bench_report,
        )

        primary = self.cfg.throne_contracts()[0].arch_preset
        # The duel at the primary throne size: the king plus each finalist
        # (legacy size == "" reads as primary). Benched checkpoint-by-checkpoint;
        # a miss drops that entry from the report, never the report itself.
        duel = [e for e in manifest.entries if (e.size or primary) == primary]
        scored = self._bench_duel_checkpoints(duel, manifest.round_id, primary)
        entries = []
        for m_entry in duel:
            scores = scored.get(m_entry.trained_pointer)
            if scores is None:
                log.warning("round=%s: bench produced no complete score set for %s %s; "
                            "report omits it", manifest.round_id, m_entry.role,
                            m_entry.miner_hotkey)
                continue
            entries.append(BenchEntry(
                role=m_entry.role, size=m_entry.size or primary,
                miner_hotkey=m_entry.miner_hotkey, miner_uid=m_entry.miner_uid,
                trained_pointer=m_entry.trained_pointer, scores=scores,
            ))
        if not entries:
            log.warning("round=%s: no duel checkpoint produced bench scores; "
                        "no bench report published", manifest.round_id)
            return None
        report = BenchReport(round_id=manifest.round_id,
                             created_block=manifest.created_block,
                             entries=tuple(entries))
        if self.wallet is not None:
            report = sign_bench_report(report, self.wallet)
        else:
            log.warning("publishing an UNSIGNED bench report (no wallet); "
                        "validators will ignore it")
        key = publish_bench_report(self.manifest_store(), dump_bench_report(report),
                                   manifest.round_id)
        log.info("published bench report round=%s roles=[%s] signed=%s → s3://%s/%s",
                 manifest.round_id, ", ".join(e.role for e in entries),
                 report.signature is not None, self.cfg.storage.manifest_bucket, key)
        self._log_bench_pair_wandb(report)
        return report

    def _bench_duel_checkpoints(
        self, duel: list[TrainedEntry], round_id: str, primary: str
    ) -> dict:
        """Score each duel checkpoint on GIFT-Eval/BOOM/TIME, returning
        ``{trained_pointer: BenchScores}`` (misses simply absent).

        Remote plan wired ⇒ each checkpoint benches on the pod that trained it
        (GPU, checkpoint already at its ``_train_work`` path). Checkpoints on
        DIFFERENT physical pods bench in parallel — a serial king+challenger
        full battery would spend most of the provisioner's teardown hold — but
        launches sharing ONE pod (per-lane host entries with the same address,
        the JIT-final topology) run sequentially: every bench launch pkills any
        running sweep on its pod (PREEMPT_BENCHMARKS), so a parallel same-pod
        launch would murder its sibling mid-battery. Grouping is therefore by
        pod ADDRESS, not host name; each entry still benches through its own
        lane host so it keeps its assigned CUDA device. No remote plan ⇒
        sequential local ``bench_eval_fn`` over registry fetches."""
        from concurrent.futures import ThreadPoolExecutor

        out: dict[str, BenchScores] = {}
        if self.cascade_bench_plan is not None and self.remote_hosts:
            by_pod: dict[str, list[tuple[object, TrainedEntry]]] = {}
            for entry in duel:
                host = self._bench_host_for(entry, primary)
                if host is None:
                    continue
                pod = str(getattr(host, "host", None) or getattr(host, "name", host))
                by_pod.setdefault(pod, []).append((host, entry))

            def _bench_pod_group(pairs: list[tuple[object, TrainedEntry]]) -> None:
                for host, entry in pairs:
                    try:
                        scores = self._remote_bench_scores(host, entry, round_id, primary)
                    except Exception as e:  # noqa: BLE001 — one bad sweep must not lose the rest
                        log.warning("round=%s: remote bench failed for %s %s on %s: %s",
                                    round_id, entry.role, entry.miner_hotkey,
                                    getattr(host, "name", host), e)
                        continue
                    if scores is not None:
                        out[entry.trained_pointer] = scores

            with ThreadPoolExecutor(max_workers=max(1, len(by_pod))) as ex:
                list(ex.map(_bench_pod_group, by_pod.values()))
            return out
        if self.bench_eval_fn is not None:
            for entry in duel:
                try:
                    ckpt = self._fetch_checkpoint_dir(entry.trained_pointer)
                    scores = self.bench_eval_fn(ckpt)
                except Exception as e:  # noqa: BLE001 — one miss must not sink the rest
                    log.warning("round=%s: local bench failed for %s %s: %s",
                                round_id, entry.role, entry.miner_hotkey, e)
                    continue
                if scores is not None:
                    out[entry.trained_pointer] = scores
        return out

    def _bench_host_for(self, entry: TrainedEntry, primary: str) -> object | None:
        """The pod holding ``entry``'s checkpoint: the tracked dispatch host,
        or the round-robin heuristic (king first, challenger next) when a
        restart between duel and bench lost the tracking dict."""
        host = self._final_role_hosts.get(
            (entry.role, entry.size or primary, entry.miner_hotkey))
        if host is not None:
            return host
        hosts = self._hosts_for("final")
        if not hosts:
            return None
        return hosts[0] if entry.role == "king" else hosts[1 % len(hosts)]

    def _remote_bench_scores(
        self, host: object, entry: TrainedEntry, round_id: str, primary: str
    ) -> BenchScores | None:
        """One remote sweep → six numbers, or ``None`` on any miss.

        With ``[telemetry] bench_anneal_fraction`` armed (DEC-CA-0030) the
        sweep scores an ANNEALED copy of the checkpoint — finished-form
        numbers for the signed BenchScores — falling back to the raw
        checkpoint on any anneal-leg failure."""
        from ..eval.benchmarks import extract_bench_scores
        from .bench_hook import run_post_round_benchmark

        bench_round, bench_role = round_id, entry.role
        frac = float(getattr(self.cfg.telemetry, "bench_anneal_fraction", 0.0) or 0.0)
        if frac > 0.0:
            annealed = self._dispatch_bench_anneal(host, entry, round_id, primary, frac)
            if annealed is not None:
                bench_round, bench_role = annealed
        report = run_post_round_benchmark(
            host, bench_round, entry.size or primary, self.cascade_bench_plan,
            work_root=self.work_root, role=bench_role,
        )
        scores = extract_bench_scores(report) if report is not None else None
        return BenchScores(**scores) if scores is not None else None

    def _dispatch_bench_anneal(
        self, host: object, entry: TrainedEntry, round_id: str, primary: str,
        frac: float,
    ) -> tuple[str, str] | None:
        """Run the bench-anneal leg (DEC-CA-0030) for one duel checkpoint on
        the pod that trained it, returning the ``(round_dir, role_dir)`` pair
        naming the annealed checkpoint's ``_train_work`` location for the
        bench sweep — or ``None`` on any failure (the caller benches the raw
        checkpoint instead; log-only telemetry must never lose the sweep).

        The leg is a stock worker run: resume ``entry.trained_pointer``
        (weights + optimizer state) on a FRESH salted corpus for
        ``frac × target_train_hours`` under the pure-decay recipe
        (``--anneal``). The salt keys the work dir + checkpoint repo away
        from every canonical name, so nothing this leg writes can collide
        with a consensus artifact."""
        from .remote import RemoteDispatcher, pod_lane_count

        if not self.trainer_spec:
            log.warning("round=%s: bench-anneal skipped (no trainer_spec — "
                        "benching the raw checkpoint)", round_id)
            return None
        size = entry.size or primary
        contract = self.cfg.training.primary_size
        if size != contract.arch_preset:
            spec = next((sp for sp in self.cfg.training.extra_sizes
                         if sp.arch_preset == size), None)
            if spec is None:
                log.warning("round=%s: bench-anneal skipped for unknown size %r",
                            round_id, size)
                return None
            contract = self.cfg.training.for_size(spec)
        salted = (int(round_id) ^ BENCH_ANNEAL_SALT) & ((1 << 63) - 1)
        suffix = f"-anneal-u{entry.miner_uid}"
        try:
            disp = RemoteDispatcher(
                trainer_spec=self.trainer_spec,
                timeout_seconds=self.remote_timeout_seconds,
                extra_forward_env=self._pod_extra_forward_env(),
            )
            # warm_start_ref makes the leg warm_started=True in the trainer, so
            # under an armed [training] warm_lr_scale (DEC-CA-0035) its cosine
            # decays from base_lr × scale. Exactly right for warm-started duel
            # checkpoints (matches their stable-phase LR); for a generation-
            # start checkpoint (stable phase ran full base_lr) the leg starts
            # one notch low — accepted: telemetry-only, errs conservative, and
            # all legs stay mutually comparable at the same start LR.
            disp.dispatch(
                host, gen_ref=entry.gen_ref, uid=entry.miner_uid,
                hotkey=entry.miner_hotkey, role=entry.role,
                base_seed=salted, block=entry.train_block,
                arch_preset=size, train_hours=contract.target_train_hours * frac,
                repo_suffix=suffix, warm_start_ref=entry.trained_pointer,
                lane_count=pod_lane_count(host, [host]), anneal=True,
            )
        except Exception as e:  # noqa: BLE001 — telemetry: fall back, never raise
            log.warning("round=%s: bench-anneal leg failed for %s %s (benching "
                        "the raw checkpoint instead): %s",
                        round_id, entry.role, entry.miner_hotkey, e)
            return None
        log.info("round=%s: bench-anneal leg done for %s %s (frac=%.2f) — "
                 "benching the annealed copy", round_id, entry.role,
                 entry.miner_hotkey, frac)
        # The worker wrote _train_work/<salted>/<size>/<role><suffix>/checkpoint
        # (dir = role + repo suffix, same layout the scratch shadow benches).
        return str(salted), f"{entry.role}{suffix}"

    def _log_bench_pair_wandb(self, report: object) -> None:
        """Mirror the round's king/challenger bench pair to wandb when
        ``[wandb] enabled`` — observability only, same swallow-everything
        contract as the per-step training mirror."""
        try:
            if not getattr(self.cfg.wandb, "enabled", False):
                return
            entries = list(getattr(report, "entries", ()) or ())
            if not entries:
                return
            anchor = next((e for e in entries if e.role == "king"), entries[0])
            sink = open_wandb_run(
                self.cfg.wandb, round_id=str(report.round_id),
                role=f"bench-{anchor.size}", hotkey=anchor.miner_hotkey,
                uid=anchor.miner_uid, size=anchor.size,
            )
            if sink is None:
                return
            for e in entries:
                s = e.scores
                sink.emit({
                    "event": "cascade_bench", "role": e.role,
                    "miner_hotkey": e.miner_hotkey, "miner_uid": e.miner_uid,
                    "gifteval_crps": s.gifteval_crps, "gifteval_mase": s.gifteval_mase,
                    "boom_crps": s.boom_crps, "boom_mase": s.boom_mase,
                    "time_crps": s.time_crps, "time_mase": s.time_mase,
                })
            sink.finish()
        except Exception as e:  # noqa: BLE001 — wandb must never disturb the bench
            log.debug("wandb bench-pair log failed (continuing): %s", e)

    def launch_post_publish_bench(self, manifest: TrainingManifest) -> object | None:
        """Fire-and-forget wrapper around :meth:`run_post_publish_bench`: a
        daemon thread, so the live loop moves straight on to polling the next
        epoch while validators score the already-published round."""
        import threading

        t = threading.Thread(
            target=self.run_post_publish_bench, args=(manifest,),
            name=f"cascade-bench-{manifest.round_id}", daemon=True,
        )
        t.start()
        log.info("post-publish duel bench launched for round=%s", manifest.round_id)
        return t

    def run_round(
        self,
        commitments: list[Commitment],
        king_hotkey: str | None,
        base_seed: int,
        block: int,
        *,
        cutoff_block: int | None = None,
    ) -> TrainingManifest:
        """Run one daily round and return the assembled (unsigned) manifest.

        Two stages, both under one shared :class:`RoundSeeds` (identical random
        init for the whole round):

        1. **Heat** — every eligible challenger is trained cheaply
           (``[round] heat_train_hours`` on the primary size) and screened; the
           top ``[round] finalists`` advance.
        2. **Final** — the king and the surviving finalists are trained to the
           full ``[training] target_train_hours`` at EVERY configured size
           (primary + ``[[training.sizes]]``). Each (king, challenger) pair is
           tagged with its size so the validator can combine scores across sizes
           into one throne.

        ``cutoff_block`` (the epoch boundary) is the submission deadline: only
        commitments revealed before it are eligible (see
        :func:`resolve_commitments`). Does not publish; see :meth:`publish`.
        Trains locally (sequential) by default, or across ``remote_hosts`` when
        configured. A king failure at any size aborts the round.
        """
        resolved = resolve_commitments(commitments, cutoff_block=cutoff_block,
                                       floor_block=self.cfg.round.commit_floor_block)
        # The reigning king is NOT a new submission — it already holds the throne,
        # so it is exempt from the challenger submission cutoff. Resolve it from the
        # FULL commitment set: a champion that (re-)committed at/after the epoch
        # boundary must still be trained AS king, not silently replaced by a
        # challenger. Training the wrong king makes the validator (whose champion
        # this is) reject the round `king_resyncing` until they re-converge.
        king_rg = None
        if king_hotkey is not None:
            king_rg = next(
                (rg for rg in resolve_commitments(
                    commitments, floor_block=self.cfg.round.commit_floor_block)
                 if rg.hotkey == king_hotkey),
                None,
            )
        plan = plan_round(resolved, king_hotkey, king=king_rg,
                          genesis_ref=self.cfg.round.genesis_generator_ref or None)
        if plan.king is None:
            raise RuntimeError("no resolvable generators on the netuid; nothing to train")

        seeds = RoundSeeds.derive(base_seed, self.cfg.training)
        # Fresh telemetry for this round (see _train_checkpoint / the roll-ups).
        self._round_telemetry = {"heat": [], "final": []}
        self._final_role_hosts = {}
        # The screener keys a daily-snapshot eval pool by the round's epoch
        # boundary. The live loop supplies it as ``cutoff_block``; derive it for
        # direct callers (scripts, operators) so a bucket-backed pool never
        # silently screens on a NEWER snapshot than the validator will judge the
        # final on (``None`` would mean "newest").
        screen_block = cutoff_block
        if screen_block is None:
            epoch_blocks = effective_epoch_blocks(self.cfg.round, block)
            screen_block = (block // epoch_blocks) * epoch_blocks

        # Cascade warm-start: this round's member of the live promoted set —
        # ONE init for every run this round (heat AND final: screening must
        # rank on the same init the final trains at, and the manifest carries a
        # single signed pin). The epoch index drives the rotation across
        # members. Raises on a live-but-broken pointer; None ⇒ random init.
        epoch_idx = int(screen_block) // effective_epoch_blocks(self.cfg.round, int(screen_block))
        warm_start = self._load_warm_start(epoch_index=epoch_idx)
        if warm_start is not None:
            log.info("round=%s warm-start init: %s (size=%s, epoch_index=%d)",
                     base_seed, warm_start[0], warm_start[1], epoch_idx)

        self._storage_dropped.clear()   # per-round: see _burn_hotkeys exemption
        # Round-level retry AFTER this round's heat settled (r47): the settle
        # burned every entrant, so re-deriving eligibility would field nobody
        # and walk the king over the settled finalists. Reuse them instead —
        # the burn filter and the dedup screen are deliberately bypassed for
        # exactly these hotkeys in exactly this round (the settled heat already
        # screened and deduped them); the burn set itself is untouched.
        reused = self._settled_finalists(base_seed, plan.challengers)
        if reused is not None:
            eligible = list(reused)
            screened = list(reused)
        else:
            eligible = self._filter_burned_challengers(plan.challengers)
            # Content-level duplicate screen ([round] dedup_mode): re-uploads and
            # near-copies of the king or a lower-UID challenger lose their heat GPU
            # slot before any pod is dispatched. They stay in ``eligible`` — entering
            # the round as a copy still consumes the one lifetime submission.
            screened = self._screen_duplicate_entrants(plan.king, eligible, base_seed)
        # Stage reporting context for this round; the epoch boundary is the
        # dashboards' join key (they derive it from the same grid).
        ws_info = None
        if warm_start is not None:
            ws_info = {"init_checkpoint": str(warm_start[0]),
                       "size": str(warm_start[1])}
            gen = int(getattr(self.promotion, "generation", 0) or 0)
            if gen:
                ws_info["generation"] = gen
            # The rotation's pick for the NEXT round — a schedule, not a
            # promise: a promotion firing at the boundary replaces the member
            # set, and the next round then trains from the new generation.
            try:
                nxt = self._load_warm_start(epoch_index=epoch_idx + 1)
            except RuntimeError:
                nxt = None
            if nxt is not None:
                ws_info["next_scheduled_init"] = str(nxt[0])
        self._stage_ctx = {"round_id": str(base_seed),
                           "epoch_start_block": int(screen_block),
                           "warm_start": ws_info}
        self._publish_stage("heat", heat_done=0, heat_total=len(screened))
        if reused is not None:
            # Retry after settle: no heat compute is re-spent and no standings
            # are re-derived — the finalists advance exactly as settled.
            finalists, heat = list(reused), None
        else:
            finalists, heat = self._run_heat(screened, seeds, block,
                                             screen_block=screen_block,
                                             warm_start=warm_start)
        # Burn only now, after the heat stage completed: every eligible entrant
        # got its screening attempt (or its pass-through to the final). A crash
        # mid-heat leaves the burn set untouched, so no miner's one lifetime
        # submission is consumed by a round that never judged it. (On the
        # settled-retry path this is an idempotent no-op: the reused finalists
        # were burned at the original settle.)
        self._burn_hotkeys(eligible)
        if reused is None:
            # Heat settled (screened + burned + finalists chosen): signal external
            # watchers (the provisioner) that heat-stage pods are now safe to release.
            self._mark_heat_complete(base_seed, eligible, finalists)
            # Heat feedback goes public NOW, not with the round's receipt: the duel
            # and its validation still have hours to run, and a miner reading its
            # placement needs it before the next submission deadline.
            # (On the settled-retry path both already happened at the original
            # settle; re-publishing here would overwrite the real standings
            # with a no-screen doc.)
            self._publish_heat_standings(heat, screened=len(screened))
        self._publish_stage("duel", heat_done=len(eligible),
                            heat_total=len(eligible), finalists=len(finalists))
        self._log_telemetry_rollup(base_seed)  # heat-stage standings so far
        # JIT final fleets: the marker we just wrote is what tells a
        # stage-phased provisioner (final_rent_on = "heat_complete") to rent
        # the duel pods — re-read hosts and wait for final-capable entries
        # before dispatching, instead of duelling on the round-start snapshot
        # of heat pods that the same marker is tearing down. Instant when the
        # final entries were present all along (the pre-phased fleet shape).
        self._reload_remote_hosts(require_stage="final")
        jobs: list[tuple[ResolvedGenerator, str]] = [(plan.king, "king")]
        jobs += [(c, "challenger") for c in finalists]

        entries = self._train_final(jobs, seeds, block, warm_start=warm_start)
        if len(finalists) > 1:
            # DEC-CA-0012: stamp the advancing cohort's record order — 0-based,
            # best observed heat geomean first. Record order ONLY: the
            # validator sorts on (duel_rank, hotkey), judges the WHOLE cohort,
            # and crowns the best margin-clearer, so this never decides the
            # throne. A single finalist keeps the field default (0, dropped
            # from the canonical body), so those manifests hash exactly as
            # before. Stamped before the content-clone drop below — a dropped
            # clone leaves a non-contiguous rank, which consumers tolerate
            # (they sort, never index).
            order = {c.hotkey: i for i, c in enumerate(finalists)}
            entries = [
                replace(e, duel_rank=order[e.miner_hotkey])
                if e.role == "challenger" and e.miner_hotkey in order else e
                for e in entries
            ]
        entries = _drop_final_content_clones(entries, jobs)
        if not any(e.role == "king" for e in entries):
            raise RuntimeError("king training produced no entry; aborting round")
        self._log_telemetry_rollup(base_seed)  # complete heats + finals picture

        # Cascade's public-benchmark eval deliberately does NOT run here: it
        # runs strictly AFTER publish() — validators must start scoring the
        # duel the moment the manifest lands — and its numbers travel in the
        # round's separate signed bench report (run_post_publish_bench), never
        # inside the manifest entry.

        # Pin the round's eval pool: the provenance of the snapshot the heat
        # screened on (selected at screen_block, same rule the validator uses),
        # so the pin the trainer signs is the pool validators must judge on.
        # Best-effort: a miss just publishes an unpinned (legacy) manifest.
        pool_key, pool_sha = "", ""
        if self.pool_provenance_fn is not None:
            try:
                pool_key, pool_sha = self.pool_provenance_fn(base_seed, screen_block)
            except Exception as e:  # noqa: BLE001 — pinning must never sink a round
                log.warning("eval-pool pin unavailable for round=%s: %s", base_seed, e)

        # Post-hoc realised mix of the round's eval draw (unsigned, like heat).
        # Best-effort: a miss just publishes without the block.
        composition = None
        if self.composition_fn is not None:
            try:
                composition = self.composition_fn(base_seed, screen_block)
            except Exception as e:  # noqa: BLE001 — never sinks a round
                log.warning("round composition unavailable for round=%s: %s", base_seed, e)

        return TrainingManifest(
            round_id=str(base_seed),
            created_block=block,
            contract_digest=contract_digest(self.cfg.training),
            base_arch_digest=self.cfg.training.base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset,
            entries=entries,
            heat=heat,
            composition=composition,
            eval_pool_key=str(pool_key or ""),
            eval_pool_sha256=str(pool_sha or ""),
            warm_start_ckpt=warm_start[0] if warm_start else "",
            warm_start_size=warm_start[1] if warm_start else "",
            # Publish the round's full training contract (DEC-CA-0036). Signed
            # with the rest of the manifest, so the terms a round trained under
            # are pinned to the round itself rather than inferred from whatever
            # chain.toml says whenever someone later reads it.
            contract_body=contract_payload(self.cfg.training),
        )

    def _log_telemetry_rollup(self, base_seed: int) -> None:
        """INFO roll-up of the round's collected run telemetry (skipped when no
        run trained in this process — a fully remote round's metrics live in
        each pod's own telemetry line instead)."""
        heats = self._round_telemetry["heat"]
        finals = self._round_telemetry["final"]
        if heats or finals:
            log.info("%s", telemetry_rollup_line(base_seed, heats, finals))

    def _run_heat(
        self,
        challengers: list[ResolvedGenerator],
        seeds: RoundSeeds,
        block: int,
        *,
        screen_block: int | None = None,
        warm_start: tuple[str, str] | None = None,
    ) -> tuple[list[ResolvedGenerator], HeatResult | None]:
        """Screen the field down to ``[round] finalists`` for the final stage.

        Each challenger is trained for ``[round] heat_train_hours`` on the primary
        (smallest) size and scored by the injected ``screen_fn`` (lower is
        better); the cheapest ``finalists`` advance, ``(reveal_block, uid)``
        breaking ties for determinism (a UID is not a seniority claim — it
        recycles; the reveal block is). When the field already fits within
        the finalist cap, or no
        ``screen_fn`` is wired, the field's natural order (lowest UID first) is
        taken without spending heat compute. A challenger that fails to train or
        screen is dropped (it simply doesn't qualify).

        ``screen_block`` is the round's epoch-boundary block, handed to the
        screener so a daily-snapshot eval pool selects the SAME snapshot the
        validator will judge the final on (``block`` is the current height,
        which could select a snapshot published after the boundary).

        Returns ``(finalists, heat)`` where ``heat`` is the informational
        standings the dashboard shows every entrant (:class:`HeatResult`), or
        ``None`` when no screen actually ran (no compute was spent to rank).

        With ``[round] max_finalists > 1`` the finalist count responds to the
        screen's own decisiveness statistic instead (DEC-CA-0012): a leader the
        screen separated advances alone; a statistically tied top is re-scored
        on a larger eval slice (the tie run-off) and the survivors advance,
        capped — see :meth:`_advance_cohort`. At the shipped default
        (``max_finalists = 1``) that path never runs and the behaviour above is
        bit-identical.
        """
        n = max(0, self.cfg.round.finalists)
        armed = self.cfg.round.max_finalists > 1
        cap = max(0, self.cfg.round.finalist_cap)
        if not challengers or cap == 0:
            return [], None
        if self.screen_fn is None or len(challengers) <= cap:
            if self.screen_fn is None and len(challengers) > cap:
                log.warning("no screen_fn wired; taking %d of %d challengers by UID order",
                            cap, len(challengers))
            return list(challengers[:cap]), None

        # for_hours scales the token budget AND the hard wall-clock cap to the
        # cheap heat budget — the run stops at whichever is reached first, so a
        # stalling generator costs minutes of a heat slot, never the final-scale
        # max_train_seconds.
        rnd = self.cfg.round
        heat_contract = self.cfg.screen_contract().for_hours(
            rnd.heat_train_hours,
            guard_factor=rnd.heat_guard_factor,
            guard_floor_seconds=rnd.heat_guard_floor_seconds,
        )
        heat_tokens = heat_contract.train_tokens
        # Screen from the same init the final will train at (warm-start applies
        # only to the matching size; other sizes keep random init).
        ws_ref = (warm_start[0]
                  if warm_start and warm_start[1] == heat_contract.arch_preset else None)
        trained = self._heat_train(challengers, seeds, block, heat_contract, heat_tokens,
                                   warm_start_ref=ws_ref)
        trained_hotkeys = {c.hotkey for c, _, _ in trained}
        # Content-level first-submitter rule: two challengers whose corpora share
        # a digest under this round's shared seed submitted the same generator
        # CONTENT (a re-upload of someone else's repo escapes the ref-level dedup
        # in plan_round but cannot escape this). Only the earliest reveal is
        # screened; clones would otherwise tie it exactly and could steal its
        # finalist slot on the UID tiebreak.
        by_digest: dict[str, ResolvedGenerator] = {}
        for c, _, digest in trained:
            cur = by_digest.get(digest)
            if cur is None or (c.reveal_block, c.uid) < (cur.reveal_block, cur.uid):
                by_digest[digest] = c
        duplicates = {c.hotkey for c, _, digest in trained if by_digest[digest].hotkey != c.hotkey}
        for hk in duplicates:
            log.info("heat: challenger %s dropped: corpus is byte-identical to an "
                     "earlier-revealed submission", hk)

        scored: list[tuple[float, int, ResolvedGenerator]] = []
        # Per-window scores per entrant, kept only when the screener hands them
        # over — they feed the shadow diagnostics, never the ranking.
        components: dict[str, list] = {}
        # The (crps, mase) aggregates published on the standings, derived from
        # the same per-window scores. (None, None) for a scalar-only screener.
        raw_components: dict[str, tuple[float | None, float | None]] = {}
        for c, ckpt_dir, _ in trained:
            if c.hotkey in duplicates:
                continue
            try:
                raw = self.screen_fn(ckpt_dir, c, seeds.base_seed, screen_block)
                score = self._screen_score(raw)
            except Exception as e:  # noqa: BLE001 — a broken heat entry just doesn't qualify
                log.warning("heat: challenger %s failed to screen: %s", c.hotkey, e)
                continue
            if isinstance(raw, float | int):  # a bare scalar — no raw components
                raw_components[c.hotkey] = (None, None)
            else:  # per-window scores: diagnostics AND the published components
                from ..eval.scoring import global_components

                scores = list(raw)
                components[c.hotkey] = scores
                raw_components[c.hotkey] = global_components(scores)
            log.info("heat: challenger %s score=%.5f", c.hotkey, score)
            scored.append((score, c.uid, c))

        # Lower score better; ties break on (reveal_block, uid) — a UID is not
        # a seniority claim (Bittensor recycles them), the reveal block is
        # (DEC-CA-0012; NOTE-ca-operational-invariants).
        scored.sort(key=lambda t: (t[0], t[2].reveal_block, t[1]))
        # Init-baseline shadow ([round] init_gate_mode = "shadow"): score the
        # very init this heat trained from, on the same slice — the null
        # baseline KOTH otherwise never sees. NEVER shapes advancement (the
        # enforcing gate lives in the validator's duel, [scoring]
        # init_gate_mode); here it is a published standings row only.
        init_score = self._init_baseline(scored, ws_ref, seeds, screen_block)
        diagnostics = self._screen_diagnostics(scored, components, seeds.base_seed)
        if armed:
            ckpt_dirs = {c.hotkey: d for c, d, _ in trained}
            winners = self._advance_cohort(scored, components, diagnostics,
                                           seeds, screen_block, cap, ckpt_dirs)
        else:
            winners = [c for _, _, c in scored[:n]]
        log.info("heat: %d/%d advance to the final: %s",
                 len(winners), len(challengers), [c.hotkey for c in winners])
        heat = self._heat_result(
            challengers, scored, winners, trained_hotkeys, heat_contract.arch_preset,
            len(winners) if armed else n,
            duplicates=duplicates, diagnostics=diagnostics, components=raw_components,
            init_baseline=init_score,
        )
        return winners, heat

    def _init_baseline(
        self,
        scored: list[tuple[float, int, ResolvedGenerator]],
        ws_ref: str | None,
        seeds: RoundSeeds,
        screen_block: int | None,
    ) -> float | None:
        """Score the round's warm-start init on the heat's own slice (shadow).

        Every entrant trained from this checkpoint's lineage branch, so its
        score on the same windows is the round's "did training add value?"
        null baseline. Precisely: what is scored is the init checkpoint's
        SCORED face (``weights.safetensors`` via the wrapper) — the same
        artifact form every entrant is scored on — which under an armed
        finished-form mechanism (fork-anneal / EMA) is not byte-identical to
        the stable branch entrants resume from. Score-vs-score is the
        apples-to-apples comparison; just don't read the row as "the exact
        tensor state training started from". Logged
        and published on the standings; it never changes who advances — the
        enforcing floor is the validator's duel-side gate. Fails OPEN on any
        scoring error: a baseline hiccup must never sink the round.
        """
        rnd = self.cfg.round
        mode = str(rnd.init_gate_mode or "off").lower()
        if mode != "shadow":
            if mode not in ("off", "shadow"):
                log.warning("[round] init_gate_mode=%r not supported at the heat "
                            "(only 'off'/'shadow'; enforcement is [scoring]-side); "
                            "treating as 'off'", rnd.init_gate_mode)
            return None
        if not ws_ref or not scored or self.screen_fn is None:
            return None
        try:
            init_dir = self._fetch_checkpoint_dir(ws_ref)
            score = self._screen_score(
                self.screen_fn(init_dir, None, seeds.base_seed, screen_block))
        except Exception as e:  # noqa: BLE001 — fail open, loudly
            log.error("heat: init-baseline scoring failed (%s: %s); shadow row "
                      "omitted this round", type(e).__name__, e)
            return None
        beat = sum(1 for s, _, _ in scored if s <= score)
        log.info("heat: init baseline score=%.5f — %d/%d entrants beat it",
                 score, beat, len(scored))
        return score

    def _advance_cohort(
        self,
        scored: list[tuple[float, int, ResolvedGenerator]],
        components: dict[str, list],
        diagnostics,
        seeds: RoundSeeds,
        screen_block: int | None,
        cap: int,
        ckpt_dirs: dict[str, Path],
    ) -> list[ResolvedGenerator]:
        """The DEC-CA-0012 advance rule, armed only (``max_finalists > 1``).

        A leader the screen separated from every other entrant (paired LCB > 0
        off the joint bootstrap, :func:`cascade.eval.heat.tied_set`) advances
        ALONE, whatever the cap. A statistically tied top is re-scored on a
        larger eval slice when the run-off is wired (:meth:`_tie_runoff`); the
        survivors advance, capped at ``max_finalists``. With the run-off
        disabled (or unable to finish) the pre-run-off tied set advances,
        capped, in heat-rank order.

        Degrades safely: a scalar-only screener or an unpaired field carries no
        diagnostics, and ``tied_set`` then returns the leader alone — exactly
        the pre-DEC-CA-0012 single-finalist behaviour.
        """
        from ..eval.heat import tied_set

        ranked = [c for _, _, c in scored]
        keys = [c.hotkey for c in ranked]
        by_key = {c.hotkey: c for c in ranked}
        runoff_on = self.runoff_fn is not None and self.cfg.round.tie_runoff_windows > 0
        # The run-off re-scores the WHOLE tied set (CPU minutes, wall-clock
        # capped), so it is derived uncapped; without a run-off the cap applies
        # immediately. GPU spend is bounded by the cap either way.
        tied = tied_set(diagnostics, keys, cap=(len(keys) if runoff_on else cap))
        if len(tied) <= 1:
            log.info("heat: screen separated the leader (%s); it advances alone",
                     keys[0])
            return ranked[:1]
        if runoff_on:
            survivors = self._tie_runoff(tied, by_key, components, seeds,
                                         screen_block, cap, ckpt_dirs)
            if survivors is not None:
                return survivors
        log.info("heat: tied top of %d advances capped at %d (no run-off verdict)",
                 len(tied), cap)
        return [by_key[k] for k in tied[:cap]]

    def _tie_runoff(
        self,
        tied_keys: list[str],
        by_key: dict[str, ResolvedGenerator],
        components: dict[str, list],
        seeds: RoundSeeds,
        screen_block: int | None,
        cap: int,
        ckpt_dirs: dict[str, Path],
    ) -> list[ResolvedGenerator] | None:
        """Re-score the tied heat top on a larger eval slice (DEC-CA-0012).

        CPU-only, on the orchestrator, against heat checkpoints still on local
        disk — no retraining, no GPU rent. Scores only the windows the heat has
        NOT already scored: the round's window selection is a seeded
        permutation prefix, so the heat's slice is a strict prefix of the
        ``[round] tie_runoff_windows`` slice for the same round seed; the
        incremental scores are concatenated onto the heat's and pairing holds
        by construction. The extended field is then re-ranked and the tied set
        re-derived at the same uncorrected bar; the survivors (capped) are
        returned best-first.

        Runs under the ``tie_runoff_phase_seconds`` wall clock. Returns
        ``None`` when the run-off cannot run or finish (no incremental windows,
        clock expiry, a scoring failure) — the caller falls back to the
        pre-run-off tied set, because a screen that cannot finish must not
        sink the round it protects.
        """
        from ..eval.heat import screen_diagnostics, tied_set
        from ..eval.scoring import global_geomean

        rnd = self.cfg.round
        n_heat = len(components[tied_keys[0]])
        n_total = min(rnd.tie_runoff_windows, self.cfg.eval.n_windows)
        if n_total <= n_heat:
            log.info("heat: tie run-off skipped: tie_runoff_windows=%d adds no windows "
                     "beyond the %d the heat already screened", rnd.tie_runoff_windows, n_heat)
            return None
        deadline = time.monotonic() + max(0, rnd.tie_runoff_phase_seconds)
        extended: dict[str, list] = {}
        for key in tied_keys:
            if time.monotonic() >= deadline:
                log.warning("heat: tie run-off wall clock (%ds) expired with %d/%d "
                            "re-scored; falling back to the pre-run-off tied set",
                            rnd.tie_runoff_phase_seconds, len(extended), len(tied_keys))
                return None
            try:
                extra = self.runoff_fn(ckpt_dirs[key], by_key[key], seeds.base_seed,
                                       screen_block, n_heat, n_total)
                extended[key] = list(components[key]) + list(extra)
            except Exception as e:  # noqa: BLE001 — the run-off must never sink a round
                log.warning("heat: tie run-off failed on %s: %s; falling back to the "
                            "pre-run-off tied set", key, e)
                return None
        rescored = [(global_geomean(extended[k]), by_key[k].uid, by_key[k])
                    for k in tied_keys]
        rescored.sort(key=lambda t: (t[0], t[2].reveal_block, t[1]))
        ranked_keys = [c.hotkey for _, _, c in rescored]
        try:
            diag = screen_diagnostics(
                [(k, extended[k]) for k in ranked_keys],
                seed=seeds.base_seed,
                B=self.cfg.scoring.bootstrap_B,
                alpha=self.cfg.scoring.bootstrap_alpha,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("heat: tie run-off diagnostics unavailable: %s; falling back "
                        "to the pre-run-off tied set", e)
            return None
        survivors = tied_set(diag, ranked_keys, cap=cap)
        log.info("heat: tie run-off on %d windows: %d tied → %d advance: %s",
                 n_total, len(tied_keys), len(survivors), survivors)
        return [by_key[k] for k in survivors]

    @staticmethod
    def _screen_score(raw: object) -> float:
        """Reduce a screener's return to the ranking scalar (lower is better).

        A screener may return the scalar directly or the per-window
        ``list[WindowScore]``; the latter is reduced with the same
        ``global_geomean`` the duel reports, so both paths rank identically.
        """
        if isinstance(raw, float | int):
            return float(raw)
        from ..eval.scoring import global_geomean

        return float(global_geomean(list(raw)))  # type: ignore[arg-type]

    def _screen_diagnostics(
        self,
        scored: list[tuple[float, int, ResolvedGenerator]],
        components: dict[str, list],
        base_seed: int,
    ):
        """Shadow selection diagnostics for a settled heat, or None.

        Measures how decisive the screen was — it does NOT choose the finalists;
        ``scored`` is already sorted and the winners already taken. Needs every
        scored entrant to have handed over per-window components (they are all
        screened on one window slice, so they are paired by construction).

        Wrapped: a diagnostic must never cost a round its heat.
        """
        if len(scored) < 2 or any(c.hotkey not in components for _, _, c in scored):
            return None
        try:
            from ..eval.heat import screen_diagnostics

            return screen_diagnostics(
                [(c.hotkey, components[c.hotkey]) for _, _, c in scored],
                seed=base_seed,
                B=self.cfg.scoring.bootstrap_B,
                alpha=self.cfg.scoring.bootstrap_alpha,
            )
        except Exception as e:  # noqa: BLE001 — shadow only; never fails a heat
            log.warning("heat: selection diagnostics unavailable: %s", e)
            return None

    def _heat_result(
        self,
        challengers: list[ResolvedGenerator],
        scored: list[tuple[float, int, ResolvedGenerator]],
        winners: list[ResolvedGenerator],
        trained_hotkeys: set[str],
        screen_size: str,
        finalists: int,
        *,
        duplicates: set[str] = frozenset(),
        diagnostics=None,
        components: dict[str, tuple[float | None, float | None]] | None = None,
        init_baseline: float | None = None,
    ) -> HeatResult:
        """Assemble the informational standings from a completed heat.

        Each scored entrant carries its ``rel_score`` (``score / best``) plus, when
        the screener returned per-window scores, its raw ``crps`` and
        ``mase`` on the round's eval-pool slice — published so miners can see their
        absolute error, not just the relative ranking. Entrants that never produced
        a score are carried too, tagged by how they dropped out: ``duplicate``
        (corpus byte-identical to an earlier reveal), ``failed_train`` (crashed the
        screen budget) or ``failed_screen`` (trained but the scorer raised).

        ``diagnostics`` (a :class:`cascade.eval.heat.HeatDiagnostics`, or None) is
        the shadow measurement of how decisive the screen was. It is recorded
        alongside the standings; it did not influence them.
        """
        comps = components or {}
        advanced = {c.hotkey for c in winners}
        scored_hotkeys = {c.hotkey for _, _, c in scored}
        best = scored[0][0] if scored else None
        p_best = diagnostics.p_best if diagnostics is not None else {}
        entrants: list[HeatEntrant] = []
        for rank, (score, _uid, c) in enumerate(scored, start=1):
            rel = (score / best) if (best is not None and best > 0) else None
            crps, mase = comps.get(c.hotkey, (None, None))
            entrants.append(HeatEntrant(
                uid=c.uid, hotkey=c.hotkey, gen_ref=c.ref,
                status="advanced" if c.hotkey in advanced else "screened",
                rank=rank, rel_score=rel, p_best=p_best.get(c.hotkey),
                crps=crps, mase=mase,
            ))
        for c in challengers:
            if c.hotkey in scored_hotkeys:
                continue
            if c.hotkey in duplicates:
                status = "duplicate"
            else:
                status = "failed_screen" if c.hotkey in trained_hotkeys else "failed_train"
            entrants.append(HeatEntrant(
                uid=c.uid, hotkey=c.hotkey, gen_ref=c.ref, status=status,
            ))
        if diagnostics is not None:
            log.info("heat: screen decisiveness: leader=%s p_best=%.3f lcb_vs_runner_up=%+.4f "
                     "(n_windows=%d n_clusters=%d)",
                     diagnostics.leader_key, p_best.get(diagnostics.leader_key, float("nan")),
                     diagnostics.leader_lcb if diagnostics.leader_lcb is not None else float("nan"),
                     diagnostics.n_windows, diagnostics.n_clusters)
        return HeatResult(
            screen_size=screen_size,
            finalists=finalists,
            entrants=tuple(entrants),
            leader_lcb=(diagnostics.leader_lcb if diagnostics is not None else None),
            n_windows=(diagnostics.n_windows if diagnostics is not None else None),
            n_clusters=(diagnostics.n_clusters if diagnostics is not None else None),
            init_baseline=init_baseline,
        )

    def _heat_train(
        self,
        challengers: list[ResolvedGenerator],
        seeds: RoundSeeds,
        block: int,
        heat_contract: TrainingContractConfig,
        heat_tokens: int,
        *,
        warm_start_ref: str | None = None,
    ) -> list[tuple[ResolvedGenerator, Path, str]]:
        """Train each heat challenger, returning ``[(challenger, local_ckpt_dir,
        corpus_digest)]`` for the ones that trained — the digest feeds the
        content-level duplicate drop in :meth:`_run_heat`. Dispatches to
        ``remote_hosts`` (GPU pods) when configured — the pod trains at the cheap
        heat budget and the checkpoint is fetched back for local screening, so the
        orchestrator (with the wallet) never needs a GPU — else trains locally. A
        failed train drops that challenger (it just doesn't qualify)."""
        if self.remote_hosts:
            return self._heat_train_remote(challengers, seeds, block, heat_contract,
                                           warm_start_ref=warm_start_ref)
        ws_dir = self._fetch_checkpoint_dir(warm_start_ref) if warm_start_ref else None
        out: list[tuple[ResolvedGenerator, Path, str]] = []
        for done, c in enumerate(challengers, start=1):
            out_dir = self.work_root / f"{seeds.base_seed}" / "heat" / c.hotkey / "checkpoint"
            try:
                result, digest, _, _ = self._train_checkpoint(
                    c, seeds, heat_contract, heat_tokens, out_dir, log_role=f"heat-{c.hotkey}",
                    warm_start_dir=ws_dir,
                )
                out.append((c, result.local_dir, digest))
            except Exception as e:  # noqa: BLE001
                log.warning("heat: challenger %s failed to train: %s", c.hotkey, e)
            self._note_heat_progress(done, len(challengers))
        return out

    def _pod_extra_forward_env(self) -> tuple[str, ...]:
        """Env vars every pod dispatch forwards on top of each host's own list.

        When [wandb] is enabled the training runs on the pod, so the POD is where
        ``open_wandb_run`` needs ``WANDB_API_KEY`` — forwarding it here means
        pod-side wandb logs land without the operator having to name the key in
        every host's ``forward_env`` (the silent-no-op that leaves wandb runs with
        no training logs). Only forwarded if actually present in the orchestrator
        env; absent ⇒ the pod's wandb no-ops exactly as before."""
        return ("WANDB_API_KEY",) if getattr(self.cfg.wandb, "enabled", False) else ()

    def _hosts_for(self, stage: str) -> list:
        """The pods serving ``stage`` ("heat" | "final"): hosts tagged with that
        stage or ``"any"``. The cheap-GPU seam — heats can run on a cheaper SKU
        class than the final, because heat checkpoints are trainer-internal
        (screened, discarded, never validated) while the final's king and
        challenger must satisfy the validator's gpu_name pairing. When no host
        matches the stage (e.g. a fleet tagged all-final), every host is used
        with a warning rather than stranding the stage: a heat on final-class
        pods is just pricier, and a final on the remaining pods still pairs
        king/challenger on one list."""
        hosts = self.remote_hosts or []
        matched = [h for h in hosts if getattr(h, "stage", "any") in ("any", stage)]
        if hosts and not matched:
            log.warning("no remote hosts tagged for stage %r; using all %d host(s)",
                        stage, len(hosts))
            return list(hosts)
        return matched

    @staticmethod
    def _dispatch_with_retry(disp, hosts: list, i: int, *, describe: str,
                             used_host: list | None = None, **kw):
        """Dispatch to the round-robin host, retrying ONCE on the next host on
        any failure. Rented pods churn — SSH flaps, reclaimed boxes, slow image
        pulls — and one flaky box must cost a retry, not a challenger's only
        heat slot or (for the king) the entire round. With a single host the
        retry re-uses it, since the failure may be transient rather than the
        box. A second failure propagates to the caller's policy (drop the
        challenger / abort the round).

        This seam also knows the round's full lane fan-out (``hosts``), so it
        computes each pod's lane count here and hands it to the dispatch —
        the pod-side sandbox slices its CPU cores off that geometry (see
        ``remote.pod_lane_count`` / ``sandbox._lane_cpu_slice``). ``used_host``
        (when given) receives the host the dispatch actually SUCCEEDED on —
        the post-publish duel bench must target the pod that holds the
        checkpoint, and a retry moves it off the round-robin pick."""
        from .remote import pod_lane_count

        host = hosts[i % len(hosts)]
        try:
            entry = disp.dispatch(host, lane_count=pod_lane_count(host, hosts), **kw)
        except Exception as e:  # noqa: BLE001 — any dispatch failure is retryable once
            retry_host = hosts[(i + 1) % len(hosts)]
            if _storage_failure(e):
                # Registry blips are global, not per-box: an instant retry on
                # another host hits the same blip. Wait it out first.
                log.warning("%s failed on %s at the storage layer (%s); backing off "
                            "%.0fs before the retry", describe, getattr(host, "name", host),
                            e, STORAGE_RETRY_BACKOFF_SECONDS)
                time.sleep(STORAGE_RETRY_BACKOFF_SECONDS)
            log.warning("%s failed on %s (%s); retrying on %s", describe,
                        getattr(host, "name", host), e, getattr(retry_host, "name", retry_host))
            entry = disp.dispatch(retry_host, lane_count=pod_lane_count(retry_host, hosts), **kw)
            host = retry_host
        if used_host is not None:
            used_host.append(host)
        return entry

    @staticmethod
    def _dispatch_on_free_lane(disp, free_lanes, hosts: list, *, describe: str,
                               used_host: list | None = None, **kw):
        """Dispatch on the next IDLE lane, retrying once on whichever lane is
        free after a failure (a different one whenever one is available).

        Same retry policy as :meth:`_dispatch_with_retry`, but lane occupancy
        is tracked through ``free_lanes`` (a ``queue.Queue`` of hosts) instead
        of a static ``i % n`` pin. The pin double-booked GPUs: a fast-failing
        challenger freed its worker THREAD but not its lane, so the next
        challenger landed on a still-busy GPU while the freed one idled — and
        heats are wall-clock scored, so the co-tenant's throughput (and score)
        halved (2026-07-15). A checked-out lane always returns to the pool,
        success or failure: a lane that failed for a challenger-specific
        reason (import error, OOM) is still good silicon."""
        from .remote import pod_lane_count

        host = free_lanes.get()
        try:
            entry = disp.dispatch(host, lane_count=pod_lane_count(host, hosts), **kw)
        except Exception as e:  # noqa: BLE001 — any dispatch failure is retryable once
            free_lanes.put(host)                 # failed lane rejoins the rotation
            if _storage_failure(e):
                # Same rationale as _dispatch_with_retry: registry blips are
                # global, so wait before re-dispatching. Sleep BEFORE taking
                # the retry lane so no idle GPU is held through the backoff.
                log.warning("%s failed on %s at the storage layer; backing off "
                            "%.0fs before the retry", describe,
                            getattr(host, "name", host), STORAGE_RETRY_BACKOFF_SECONDS)
                time.sleep(STORAGE_RETRY_BACKOFF_SECONDS)
            retry_host = free_lanes.get()        # next idle lane; different when one exists
            log.warning("%s failed on %s (%s); retrying on %s", describe,
                        getattr(host, "name", host), e,
                        getattr(retry_host, "name", retry_host))
            try:
                entry = disp.dispatch(retry_host,
                                      lane_count=pod_lane_count(retry_host, hosts), **kw)
                if used_host is not None:
                    used_host.append(retry_host)
                return entry
            finally:
                free_lanes.put(retry_host)
        else:
            free_lanes.put(host)
            if used_host is not None:
                used_host.append(host)
            return entry

    def _heat_train_remote(
        self,
        challengers: list[ResolvedGenerator],
        seeds: RoundSeeds,
        block: int,
        heat_contract: TrainingContractConfig,
        *,
        warm_start_ref: str | None = None,
    ) -> list[tuple[ResolvedGenerator, Path, str]]:
        """Screen-train the field on the GPU pods: dispatch each challenger to a
        host (round-robin across ``remote_hosts``, in parallel), training at the
        cheap ``[round] heat_train_hours`` on the screen size, then fetch each
        checkpoint back for local screening. Each pushes to a per-challenger repo
        so concurrent heat runs never collide. A challenger that fails to train or
        fetch is dropped. The pod's receipt carries the corpus digest, threaded
        through for the content-level duplicate drop."""
        import queue
        from concurrent.futures import ThreadPoolExecutor

        from .remote import RemoteDispatcher, RemoteDispatchError, probe_host

        if not self.trainer_spec:
            raise RuntimeError("remote heat requires trainer_spec (BaseTrainer 'module:Class')")
        hosts = self._hosts_for("heat")
        # Liveness probe BEFORE fan-out: a dead pod that still answers TCP burns
        # one challenger per dispatch (rc=255) until the whole field is spent, so
        # exclude hosts that don't pass a real SSH echo and fail the stage loudly
        # if none survive (rather than dispatching a field into a dead fleet).
        # Probed CONCURRENTLY so wall-clock is one probe timeout, not N of them.
        if hosts:
            with ThreadPoolExecutor(max_workers=min(len(hosts), 32)) as probe_ex:
                alive = list(probe_ex.map(probe_host, hosts))
            live_hosts = [h for h, ok in zip(hosts, alive, strict=True) if ok]
            if len(live_hosts) != len(hosts):
                log.warning("heat: %d of %d host(s) failed the SSH liveness probe; "
                            "excluding them from dispatch", len(hosts) - len(live_hosts),
                            len(hosts))
            if not live_hosts:
                raise RemoteDispatchError(
                    f"heat stage: all {len(hosts)} host(s) failed the SSH liveness "
                    "probe — refusing to dispatch into a dead fleet")
            hosts = live_hosts
        hub = self.hub()  # pre-init (thread-safe) before the pool
        # Heat dispatches get a TIGHT SSH timeout: the pod-side guard already
        # kills a slow run at the scaled max_train_seconds, so the only thing a
        # long outer timeout buys is a wedged pod (kernel hang, dead network
        # where SSH never returns) holding a heat slot for the full 6h default.
        # Guard + 30min covers fetch/sandbox/upload overheads around training.
        heat_timeout = min(self.remote_timeout_seconds, heat_contract.max_train_seconds + 1800)
        disp = RemoteDispatcher(trainer_spec=self.trainer_spec, timeout_seconds=heat_timeout,
                                extra_forward_env=self._pod_extra_forward_env())

        # Lane pool: dispatch lands on whichever GPU lane is actually idle
        # (see _dispatch_on_free_lane — the old i % n pin double-booked lanes).
        free_lanes: queue.Queue = queue.Queue()
        for h in hosts:
            free_lanes.put(h)

        def _run(c: ResolvedGenerator) -> tuple[ResolvedGenerator, Path, str]:
            entry = self._dispatch_on_free_lane(
                disp, free_lanes, hosts, describe=f"heat challenger {c.hotkey}",
                gen_ref=c.ref, uid=c.uid, hotkey=c.hotkey, role="challenger",
                base_seed=seeds.base_seed, block=block,
                arch_preset=heat_contract.arch_preset,
                train_hours=self.cfg.round.heat_train_hours,
                repo_suffix=f"-heat-u{c.uid}",
                warm_start_ref=warm_start_ref,
            )
            ref = parse_trained_pointer(entry.trained_pointer)
            if ref is None:
                raise RuntimeError(f"malformed trained_pointer: {entry.trained_pointer!r}")
            out_dir = self.work_root / f"{seeds.base_seed}" / "heat" / c.hotkey / "checkpoint"
            fetch_from_hub(ref, out_dir, hub)
            return c, out_dir, entry.corpus_digest

        # +2 executor headroom over the lane count: a re-queued challenger
        # sleeps its cool-down inside a pool thread WITHOUT holding a lane
        # (lane occupancy is the free_lanes queue), so sleepers must not
        # starve dispatchable threads.
        out, transport_failures = _run_heat_field(
            _run, challengers,
            max_workers=max(1, len(hosts)) + 2,
            requeues=self.cfg.round.heat_infra_requeues,
            cooldown_seconds=HEAT_REQUEUE_COOLDOWN_SECONDS,
            note_progress=self._note_heat_progress,
            storage_dropped=self._storage_dropped,
        )
        # A heat where EVERY dispatch died at the transport level (rc=255) is a
        # dead-fleet wipeout, not a screened-out field: refuse to let the caller
        # cache a 0/N heat as complete (a king-only manifest would publish).
        # Raise so the round retries after operator intervention.
        if challengers and not out and transport_failures == len(challengers):
            raise RemoteDispatchError(
                f"heat stage: all {len(challengers)} dispatch(es) failed with transport "
                "errors (rc=255) — refusing to cache a 0/N heat as complete")
        return out

    def _train_final(
        self, jobs: list[tuple[ResolvedGenerator, str]], seeds: RoundSeeds, block: int,
        *, warm_start: tuple[str, str] | None = None,
    ) -> list[TrainedEntry]:
        """Train the final jobs at each throne size, returning all receipts.

        One (king + finalists) pass per size in ``cfg.throne_contracts()`` (the
        ``[round] throne_sizes``); a king failure at any size aborts the round, a
        challenger failure drops only that challenger from that size.
        ``warm_start`` (pointer, size) applies to the matching size's pass only —
        an init trained at one size can't initialise another."""
        if not self.remote_hosts:
            # This box is the runtime for a local final; with remote hosts the
            # check runs on each pod (cascade-train-worker), which is the runtime.
            assert_train_image(self.cfg.training)
        entries: list[TrainedEntry] = []
        for contract in self.cfg.throne_contracts():
            token_budget = contract.train_tokens
            ws_ref = (warm_start[0]
                      if warm_start and warm_start[1] == contract.arch_preset else None)
            if self.remote_hosts:
                entries += self._train_remote(jobs, seeds, block, contract, token_budget,
                                              warm_start_ref=ws_ref)
            else:
                entries += self._train_local(jobs, seeds, block, contract, token_budget,
                                             warm_start_ref=ws_ref)
        return entries

    def _train_local(
        self,
        jobs: list[tuple[ResolvedGenerator, str]],
        seeds: RoundSeeds,
        block: int,
        contract: TrainingContractConfig,
        token_budget: int,
        *,
        warm_start_ref: str | None = None,
    ) -> list[TrainedEntry]:
        """Sequential training on this box for one size: king first (its failure
        aborts the round), then each challenger (a failure just drops it)."""
        entries: list[TrainedEntry] = []
        for gen, role in jobs:
            try:
                entries.append(
                    self.train_one(gen, role, seeds, block,
                                   contract=contract, token_budget=token_budget,
                                   repo_suffix=_final_repo_suffix(jobs, gen, role),
                                   warm_start_ref=warm_start_ref)
                )
            except Exception as e:  # noqa: BLE001
                if role == "king":
                    raise
                log.warning("challenger %s failed to train (%s): %s",
                            gen.hotkey, contract.arch_preset, e)
        return entries

    def _train_remote(
        self,
        jobs: list[tuple[ResolvedGenerator, str]],
        seeds: RoundSeeds,
        block: int,
        contract: TrainingContractConfig,
        token_budget: int,  # noqa: ARG002 — budget travels via chain.toml on the pod
        *,
        warm_start_ref: str | None = None,
    ) -> list[TrainedEntry]:
        """Parallel training across ``remote_hosts`` for one size (king→pod A,
        challenger→pod B over SSH). Equal compute is preserved (fixed token
        budget); audit is tolerance-based on rented hardware. King failure aborts
        the round; a challenger failure drops only that challenger."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from .remote import RemoteDispatcher

        if not self.trainer_spec:
            raise RuntimeError("remote training requires trainer_spec (BaseTrainer 'module:Class')")
        hosts = self._hosts_for("final")
        disp = RemoteDispatcher(
            trainer_spec=self.trainer_spec, timeout_seconds=self.remote_timeout_seconds,
            extra_forward_env=self._pod_extra_forward_env(),
        )

        def _fresh_final_hosts() -> list:
            if self.remote_hosts_path is None:
                return []
            from .remote import load_hosts

            now = load_hosts(self.remote_hosts_path)
            return [h for h in now if getattr(h, "stage", "any") in ("any", "final")]

        # Lane pool over the free-lane dispatch (the heat's anti-double-booking
        # pattern) PLUS mid-final membership refresh: a top-up pod rented after
        # a boot-failure drop joins the rotation while queued jobs wait.
        lane_pool = _FinalLanePool(hosts, _fresh_final_hosts)

        def _run(i: int, gen: ResolvedGenerator, role: str) -> TrainedEntry:  # noqa: ARG001
            used: list = []
            # Cohort finals need per-uid checkpoint repos (see
            # _final_repo_suffix); forwarded only when non-empty so a
            # single-challenger round's dispatch stays byte-identical.
            suffix = _final_repo_suffix(jobs, gen, role)
            entry = self._dispatch_on_free_lane(
                disp, lane_pool, lane_pool.known_hosts(),
                describe=f"final {role} {gen.hotkey}",
                used_host=used,
                gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey,
                role=role, base_seed=seeds.base_seed, block=block,
                arch_preset=contract.arch_preset,
                warm_start_ref=warm_start_ref,
                **({"repo_suffix": suffix} if suffix else {}),
            )
            if used:
                # Post-publish bench target: the pod actually holding this
                # final checkpoint at its _train_work path.
                self._final_role_hosts[(role, contract.arch_preset, gen.hotkey)] = used[-1]
            return entry

        results: list[TrainedEntry | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            futs = {ex.submit(_run, i, gen, role): (i, gen, role)
                    for i, (gen, role) in enumerate(jobs)}
            for fut in as_completed(futs):
                i, gen, role = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:  # noqa: BLE001
                    if role == "king":
                        raise RuntimeError(f"king training failed on remote: {e}") from e
                    log.warning("challenger %s failed on remote (%s): %s",
                                gen.hotkey, contract.arch_preset, e)
        return [r for r in results if r is not None]

    def publish(self, manifest: TrainingManifest) -> None:
        """Sign the manifest with the trainer hotkey and write it to the Hippius
        S3 manifest bucket (``round-<id>.json`` + ``latest.json``)."""
        if self.wallet is not None:
            manifest = sign_manifest(manifest, self.wallet)
        elif manifest.signature is None:
            log.warning("publishing an UNSIGNED manifest (no wallet); validators will reject it")
        key = publish_manifest(self.manifest_store(), dump_manifest(manifest), manifest.round_id)
        log.info(
            "published manifest round=%s entries=%d signed=%s → s3://%s/%s",
            manifest.round_id, len(manifest.entries), manifest.signature is not None,
            self.cfg.storage.manifest_bucket, key,
        )
        self._publish_mix(manifest)
        # The manifest is out: validators take over. Guard on the round context
        # matching so a direct publish() of some other round's manifest never
        # mislabels the live one.
        if self._stage_ctx is not None and self._stage_ctx["round_id"] == manifest.round_id:
            self._publish_stage("validation")
        # The disk half of the restart re-entry guard: remember the published
        # round across a process restart (and through a store too flaky to
        # answer the guard's manifest probe).
        self._persist_last_round(manifest.round_id)

    def _publish_mix(self, manifest: TrainingManifest) -> None:
        """Mirror the manifest's unsigned ``composition`` block to
        ``mix/round-<id>.json`` + ``mix/latest.json``, public-read.

        The dashboard's Eval-mix tab reads these: the manifest itself is not
        anonymously readable, so the presentational copy needs its own public
        key (the copy validators audit stays in the signed manifest). Absent
        on a pre-jitter round — the tab renders the absence, never a
        fabricated mix. Best-effort like the heat standings: a storage
        failure must not disturb the round just published.
        """
        if manifest.composition is None:
            return
        from ..shared.heat_status import _publish_public_json

        doc = {"round_id": str(manifest.round_id), "composition": manifest.composition}
        try:
            store = self.manifest_store()
            for key in (f"mix/round-{manifest.round_id}.json", "mix/latest.json"):
                _publish_public_json(store, key, doc)
            log.info("round=%s: published eval mix to mix/round-%s.json",
                     manifest.round_id, manifest.round_id)
        except Exception as e:  # noqa: BLE001 — presentational, never sinks a round
            log.warning("eval-mix publish failed (ignored): %s", e)

    # ── restart re-entry guard (already-published rounds) ────────────────────

    def _last_round_path(self) -> Path:
        return self.work_root / "last_round.json"

    def _persist_last_round(self, round_id: str) -> None:
        """Record the just-published round in ``work_root/last_round.json``.

        run_forever's in-memory ``last_round`` dies with the process while the
        round_id re-derives from the epoch-start block, so without a persisted
        marker a restarted trainer walks straight back into the round it just
        published. Written atomically (tmp + rename) so a crash never leaves a
        torn marker; best-effort — the manifest-store probe still guards when
        this write fails (and vice versa: the marker covers a store whose
        publish landed but whose reads are flaking).
        """
        try:
            self.work_root.mkdir(parents=True, exist_ok=True)
            tmp = self.work_root / "last_round.json.tmp"
            tmp.write_text(json.dumps({"round_id": str(round_id)}, sort_keys=True),
                           encoding="utf-8")
            tmp.replace(self._last_round_path())
        except OSError as e:
            log.warning("could not persist last_round marker for round=%s: %s", round_id, e)

    def _persisted_last_round(self) -> str | None:
        """The round_id persisted by the last successful publish, else None
        (first round after deploy, wiped work dir, unreadable marker)."""
        try:
            return str(json.loads(self._last_round_path().read_text(encoding="utf-8"))["round_id"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _round_already_published(self, round_id: str) -> bool:
        """True when this round is already finished — the restart re-entry guard.

        A trainer restart mid-epoch forgets ``last_round`` and re-enters the
        round it already completed: on 2026-07-23 the re-entry re-published a
        stale manifest (feeding a full fleet teardown 31s after rent), and on
        2026-07-24 it skipped every challenger as already burned and then sat
        in the pre-duel final-hosts wait over an empty hosts.toml — primed to
        seize the NEXT round's freshly rented final pods and dispatch the stale
        duel onto them. "Finished" is strictly "manifest present at
        ``manifest_round_key(round_id)``" or the persisted post-publish marker
        — never the work dir, which a crash mid-round (pre-publish) also
        leaves behind and which must still resume.

        Fail-open by design: a manifest probe that errors (primary AND backup
        unreadable) proceeds with the round — re-running one finished round is
        recoverable, a trainer deadlocked on blind storage is not. The probe
        reads through :meth:`manifest_store` (primary + R2 fallback), so a
        Hippius outage alone does not blind the guard.
        """
        if self.force_rerun_round is not None and str(self.force_rerun_round) == str(round_id):
            log.warning("force-rerun-round %s: bypassing the already-published guard "
                        "(operator escape hatch)", round_id)
            return False
        if self._persisted_last_round() == str(round_id):
            log.info("round %s already published (persisted last_round marker); "
                     "resuming poll — restart re-entry guard", round_id)
            return True
        try:
            self.manifest_store().get_text(manifest_round_key(str(round_id)))
        except ObjectNotFound:
            return False    # genuinely unpublished — normal round entry
        except Exception as e:  # noqa: BLE001 — fail open rather than deadlock the subnet
            log.warning("already-published probe failed for round=%s (%s); "
                        "proceeding with the round (fail-open)", round_id, e)
            return False
        log.info("round %s already published (manifest present); resuming poll — "
                 "restart re-entry guard", round_id)
        return True

    # ── live loop ────────────────────────────────────────────────────────────

    def _reload_remote_hosts(self, require_stage: str | None = None) -> None:
        """Refresh ``remote_hosts`` from ``remote_hosts_path`` for this round.

        The elastic-fleet seam: a per-round provisioner (sized off the revealed
        field, e.g. ``cascade-trainer --plan-only``) rents pods, health-checks
        them, and writes the hosts TOML; this re-read picks the fleet up without
        a trainer restart. Waits up to ``hosts_wait_seconds`` for the file to
        appear/fill — pods boot after the reveal-margin field count, so the
        round can start before they are ready — then falls back to local
        training rather than holding the round hostage. No-op when no
        ``remote_hosts_path`` is configured (a static ``remote_hosts`` list, or
        purely local training).

        ``require_stage`` waits for a fleet that can SERVE that stage (a host
        tagged with it, or ``"any"``): the mid-round re-read before the final
        dispatch, for provisioners that rent the final fleet just-in-time at
        the heat_complete marker (``final_rent_on = "heat_complete"``) —
        without it the duel would dispatch onto the round-start snapshot,
        i.e. heat pods that are being torn down at that very moment. On
        timeout the last loaded fleet is kept (``_hosts_for`` then applies
        its use-all-hosts fallback), preserving pre-phased behaviour for
        fleets whose final entries were present all along.
        """
        if self.remote_hosts_path is None:
            return
        from .remote import RemoteDispatchError, load_hosts

        deadline = time.time() + max(0, self.hosts_wait_seconds)
        hosts = None
        while True:
            try:
                hosts = load_hosts(self.remote_hosts_path)
            except RemoteDispatchError as e:
                hosts, reason = None, str(e)
            else:
                reason = ""
            serves_stage = hosts and (require_stage is None or any(
                getattr(h, "stage", "any") in ("any", require_stage) for h in hosts))
            if serves_stage:
                if self.remote_hosts is None or [h.name for h in hosts] != [
                    h.name for h in self.remote_hosts
                ]:
                    log.info("round fleet: %d pod(s): %s",
                             len(hosts), ", ".join(h.name for h in hosts))
                self.remote_hosts = hosts
                return
            if time.time() >= deadline:
                if require_stage is not None and hosts:
                    log.warning("no %s-stage hosts appeared within %ds; proceeding "
                                "with the %d host(s) on file", require_stage,
                                self.hosts_wait_seconds, len(hosts))
                    self.remote_hosts = hosts
                else:
                    log.warning("no remote hosts available (%s); training locally this round",
                                reason or str(self.remote_hosts_path))
                    self.remote_hosts = None
                return
            time.sleep(min(15.0, max(1.0, deadline - time.time())))

    @staticmethod
    def _with_deadline(fn, seconds: float):
        """Run ``fn()`` under a HARD wall-clock deadline in a helper thread.

        bittensor's websocket calls have no client-side timeout and can hang
        indefinitely; a loop that must publish once per round cannot block on
        one. A timed-out call leaks its helper thread (it dies with the
        process), the accepted cost. Raises ``TimeoutError`` on deadline.
        Mirrors ``cascade.provision.loop.ProvisionerLoop._with_deadline``."""
        import concurrent.futures

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return ex.submit(fn).result(timeout=seconds)
        finally:
            ex.shutdown(wait=False)

    def _block_with_freeze_guard(self, client: object) -> int:
        """The chain height, rebuilding the connection when it hangs or freezes.

        A bittensor websocket can die quietly in three ways: it keeps answering
        ``current_block()`` with a stale value (~20-min-old block), it raises,
        or it hangs without erroring — and on a stale read the trainer
        re-derives an already-published round and re-enters it. Guard all three,
        exactly like the provisioner's ``_current_block``:

        * the read runs under ``chain_read_timeout_s`` so a hang becomes a
          ``TimeoutError`` instead of a wedged loop;
        * a raised/hung read rebuilds the connection (``client.reconnect()``)
          and retries once — the fresh client's answer is trusted;
        * a height that has not advanced for ``stale_block_after_s`` (blocks are
          ~12s, so a multi-minute freeze is anomalous) rebuilds and re-reads.

        A client without ``reconnect`` (offline fakes) cannot be rebuilt: a
        raise propagates and a freeze just returns the last read, unchanged."""
        now = self.chain_clock()
        reconnect = getattr(client, "reconnect", None)
        try:
            block = int(self._with_deadline(client.current_block, self.chain_read_timeout_s))
        except Exception as e:  # noqa: BLE001 — a dead/hung client is rebuildable
            if reconnect is None:
                raise
            log.warning("chain read failed/hung (%s); rebuilding substrate connection",
                        type(e).__name__)
            reconnect()
            block = int(self._with_deadline(client.current_block, self.chain_read_timeout_s))
            self._block_changed_at = now
        if self._last_block is None or block != self._last_block:
            self._last_block = block
            self._block_changed_at = now
        elif reconnect is not None and now - self._block_changed_at > self.stale_block_after_s:
            log.warning("chain block frozen at %d for %.0fs — rebuilding substrate "
                        "connection (quietly dead websocket?)",
                        block, now - self._block_changed_at)
            reconnect()
            block = int(self._with_deadline(client.current_block, self.chain_read_timeout_s))
            self._last_block = block
            self._block_changed_at = now
        return block

    def run_forever(self, client: object) -> None:  # pragma: no cover
        """Poll → train → publish, once per daily round (epoch).

        A *round* is one ``[round] epoch_blocks`` window (~24h). It is keyed by
        the chain block hash at the EPOCH BOUNDARY (``epoch_start = block //
        epoch_blocks × epoch_blocks``), which is the shared base seed — so the
        whole day's heat and final trainings share one :class:`RoundSeeds`, and
        every honest party re-derives the same seeds and the same eligible field.
        The reigning king is the highest-incentive UID on the metagraph
        (validators own the dethrone decision; the trainer just reads weights).
        """
        poll = self.cfg.manifest.poll_seconds
        last_round: str | None = None
        while True:
            try:
                block = self._block_with_freeze_guard(client)
                # Resolved per tick, NOT hoisted: under a scheduled cadence
                # change ([round] epoch_activation_block) a hoisted value would
                # pin the trainer to the pre-switch length until it restarted,
                # which is exactly the drift the block gate exists to prevent.
                epoch_blocks = effective_epoch_blocks(self.cfg.round, block)
                # BEFORE the round-skip branches: the commit-order evidence this
                # collects is destroyed by the chain at reveal, and most of the
                # window where it exists is on ticks that do no round work.
                self.witness_commits(client)
                epoch = block // epoch_blocks
                epoch_start = epoch * epoch_blocks
                base_seed = client.block_seed(epoch_start)
                round_id = str(base_seed)
                if round_id == last_round:
                    time.sleep(poll)
                    continue
                # A restart forgets last_round while the epoch-start block still
                # derives the same round_id, so probe for a finished round
                # BEFORE any round work — the commitment poll, the hosts wait —
                # and re-skip it exactly like a matching last_round (see
                # _round_already_published for the two incidents this stops).
                if self._round_already_published(round_id):
                    # A restart between publish and bench completion leaves the
                    # bench_pending marker armed with no bench coming (the
                    # thread died with the old process) — release the hold
                    # rather than bill the final pod to the cap.
                    self._release_stale_bench_hold(round_id)
                    last_round = round_id
                    time.sleep(poll)
                    continue
                # Full reveal history: a hotkey whose newest reveal landed at or
                # after this boundary must still field its latest PRE-boundary
                # reveal (resolve_commitments picks it) — latest-only reads made
                # an early next-round re-commit forfeit the current round.
                commitments = client.poll_commitments(include_history=True)
                king_hotkey = client.highest_incentive_hotkey()
                # Cascade promotion (DEC-CA-0013): track the reign at the
                # boundary and fire a promotion BEFORE the round trains, so the
                # signed record is published (and fetchable by validators)
                # before any manifest pins a new-generation member. The reign
                # clock keys off the signed receipt trail's verdict king when
                # available — the on-chain incentive lags a dethrone by 1-2
                # epochs, and a clock still anchored to the deposed reign would
                # fire a promotion every validator judges premature — falling
                # back to the incentive king when no receipt is readable.
                # Guarded: promotion must never sink a round.
                if self.promotion is not None:
                    try:
                        # First boundary of a never-anchored engine: count the
                        # rounds the king already reigned (signed receipt tail
                        # + published bench reports) before the clock ticks —
                        # see _seed_promotion_reign. No-op ever after.
                        self._seed_promotion_reign()
                        self.promotion.note_round(self._receipt_king() or king_hotkey,
                                                  epoch_block=epoch_start)
                        # Out-of-band bench reports (mop-ups, trainer-downtime
                        # publishes) enter the pool here — before maybe_promote,
                        # so a promotion that fires NOW selects from the full
                        # reign, not just what the in-process bench thread saw.
                        self._replay_reign_bench_reports()
                        self.promotion.maybe_promote(
                            epoch_block=epoch_start, round_id=round_id)
                    except Exception as e:  # noqa: BLE001
                        log.warning("promotion step failed for round=%s: %s", round_id, e)
                    # Publish-with-retry: the record survives (persisted) as
                    # pending until the publish lands, so a store outage never
                    # orphans a generation the pointer file already rotates
                    # over. Flushed again right before this round's manifest
                    # publishes (below) — the record must be fetchable before
                    # any validator gates the manifest that pins its member.
                    self._flush_pending_promotion(round_id)
                self._reload_remote_hosts()  # per-round elastic fleet pickup
                log.info("starting round=%s epoch=%d epoch_start=%d king=%s field=%d",
                         round_id, epoch, epoch_start, king_hotkey,
                         len({c.hotkey for c in commitments}))
                manifest = self.run_round(
                    commitments, king_hotkey, base_seed, block, cutoff_block=epoch_start,
                )
                # Cascade duel bench hold: the marker must be on disk BEFORE the
                # manifest publishes — the manifest is the provisioner's final-pod
                # teardown trigger, and a marker written after it races the very
                # sweep it pauses. Never written while cascade_enabled is off.
                will_bench = self.will_run_post_publish_bench()
                if will_bench:
                    self._mark_bench_pending(round_id)
                # Last chance before validators gate this round's manifest: a
                # promotion record whose boundary-time publish failed on a
                # transient outage retries NOW, hours later — otherwise the
                # manifest pins a member of a generation no validator can
                # fetch, and the whole fleet rejects the round.
                self._flush_pending_promotion(round_id)
                self.publish(manifest)
                if will_bench:
                    # Guarded separately (like the telemetry launch below): the
                    # round is published, so a bench failure must not fall
                    # through to the round handler and re-run it next poll.
                    try:
                        self.launch_post_publish_bench(manifest)
                    except Exception as e:  # noqa: BLE001 — telemetry/promotion input only
                        self._mark_bench_complete(round_id, uploaded=False)
                        log.warning("post-publish bench launch failed (ignored): %s", e)
                if self.bench_plan is not None and self.remote_hosts and not will_bench:
                    # Guarded separately: the round is already published, so a
                    # telemetry failure here must not fall through to the round
                    # handler and re-run (re-train + re-publish) it next poll.
                    # Skipped when the cascade bench runs (`not will_bench`
                    # above): both shell out to cascade-benchmark on the final
                    # pod and every launch preempts the previous sweep, so the
                    # two would kill each other — and the cascade bench already
                    # covers the king this telemetry would have sampled.
                    try:
                        from .bench_hook import launch_post_round_benchmark

                        # The final trains king checkpoints for the throne
                        # sizes, which need not include the primary preset.
                        # Prefer a final-class pod (the heat pods may be a
                        # cheaper SKU the benchmark sweep wasn't sized for).
                        launch_post_round_benchmark(
                            self._hosts_for("final")[0], round_id,
                            self.cfg.throne_contracts()[0].arch_preset, self.bench_plan,
                            work_root=self.work_root,
                        )
                    except Exception as e:  # noqa: BLE001 — telemetry only
                        log.warning("post-round benchmark launch failed (ignored): %s", e)
                last_round = round_id
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one round
                log.exception("round failed; retrying after poll interval: %s", e)
            time.sleep(poll)
