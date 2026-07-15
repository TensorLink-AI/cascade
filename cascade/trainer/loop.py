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
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..interface.validation import parse_commit
from ..shared.chain import Commitment
from ..shared.config import ChainConfig, TrainingContractConfig
from ..shared.hippius import (
    HubConfig,
    LogSink,
    S3Config,
    S3Store,
    StorageError,
    fetch_from_hub,
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
    dump_manifest,
    format_trained_pointer,
    parse_trained_pointer,
    sign_manifest,
)
from .contract import BaseTrainer, RoundSeeds, TrainResult, assert_train_image
from .corpus import CorpusError
from .stream import open_round_stream
from .wandb_sink import open_wandb_run

# Screens one heat checkpoint: given the trained heat-model directory, the
# generator that produced its corpus, the round's base seed (so the screening
# window slice can rotate per round), and the round's epoch-boundary block (so a
# daily-snapshot pool selects the SAME snapshot the validator will judge the
# final on — not whatever is newest), return a heat score (LOWER is better, e.g.
# geomean(CRPS, MASE) on the held-out windows). Injected so the trainer's
# screening stays a testable boundary — the default wiring (torch evaluator +
# eval pool) is attached in cascade.trainer.main.
ScreenFn = Callable[[Path, "ResolvedGenerator", int, int | None], float]

# Scores the king's trained checkpoint on the public suites (GIFT-Eval / BOOM /
# TIME) for Cascade, given its local checkpoint dir. Returns the six-number
# BenchScores the trainer stamps onto the king's manifest entry, or None when the
# sidecar could not produce a complete set (best-effort — a miss just leaves the
# king entry without bench_scores). Injected so the trainer's Cascade eval stays a
# testable boundary; the default wiring (fetch + benchmark sidecar) is attached in
# cascade.trainer.main.
BenchEvalFn = Callable[[Path], "BenchScores | None"]

log = logging.getLogger("cascade.trainer")


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


def telemetry_rollup_line(
    round_id: int | str, heat_metrics: list[dict], final_metrics: list[dict]
) -> str:
    """One-line per-round starvation/deadline roll-up from run metrics dicts.

    Pure formatting so the aggregation is unit-testable. Entries without the
    telemetry keys are skipped (a custom BaseTrainer may not emit them; remote
    runs keep their metrics on the pod — see ``_train_checkpoint``), and the
    trailing count says how many runs actually reported, so a silent-majority
    round can't masquerade as a healthy one.
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
    return (
        f"round={round_id} telemetry: deadline_hit {hit_h}/{len(heats)} heats + "
        f"{hit_f}/{len(finals)} finals; {wait_part} ({reported}/{total} runs "
        "reported metrics)"
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
    (``TrainerRunner._stamp_king_bench_scores``)."""

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
        if floor_block and c.reveal_block < floor_block:
            continue
        if cutoff_block is not None and c.reveal_block >= cutoff_block:
            continue
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        rg = ResolvedGenerator(
            hotkey=c.hotkey, uid=c.uid, ref=parsed.ref, reveal_block=c.reveal_block
        )
        prev = best.get(c.hotkey)
        if prev is None or c.reveal_block >= prev[0]:
            best[c.hotkey] = (c.reveal_block, rg)
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
    # Eval-pool pin: ``(base_seed, block) -> (key, sha256)`` provenance of the
    # pool snapshot this round screens on, stamped (and therefore signed) into
    # the manifest so validators verify their own snapshot selection against it
    # rather than trusting the unsigned pool index. None ⇒ manifests go out
    # unpinned (legacy). Wired in trainer.main from the screen pool source.
    pool_provenance_fn: object | None = None
    # Cascade: scores the king's checkpoint on GIFT-Eval / BOOM / TIME and stamps
    # the numbers onto its manifest entry (so validators read one authoritative,
    # signed set — consensus-safe promotion). Runs only when [scoring]
    # cascade_enabled. None ⇒ no stamping (the king entry carries no bench_scores
    # and validators fall back to scoring it themselves). Wired in trainer.main.
    bench_eval_fn: BenchEvalFn | None = None
    # Remote (two-device) training: when ``remote_hosts`` is set, each round's
    # king and challenger train on separate SSH GPU pods in parallel (see
    # cascade.trainer.remote). ``trainer_spec`` is the BaseTrainer 'module:Class'
    # the pods run. None ⇒ local sequential training on this box.
    remote_hosts: list | None = None
    trainer_spec: str | None = None
    remote_timeout_seconds: int = 6 * 3600
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
    # Cascade king bench eval on the REMOTE worker: when set (cascade_enabled +
    # remote_hosts), the king's GIFT-Eval/BOOM/TIME scoring runs on the pod that
    # just trained it — GPU, checkpoint already local — instead of a local-CPU
    # subprocess. The six numbers still go on the signed manifest. Falls back to
    # the local ``bench_eval_fn`` when there is no remote host.
    cascade_bench_plan: object | None = None
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
        """
        if not self.cfg.round.one_submission_per_hotkey:
            return challengers
        seen = _load_seen_hotkeys(self._submissions_path())
        for c in challengers:
            if c.hotkey in seen:
                log.info("skipping challenger %s: hotkey already used its 1 submission "
                         "(re-register to resubmit)", c.hotkey)
        return [c for c in challengers if c.hotkey not in seen]

    def _burn_hotkeys(self, challengers: list[ResolvedGenerator]) -> None:
        """Burn the challengers that got their shot: 1 hotkey = 1 submission.

        Called AFTER the heat stage completes (not at entry): a round that
        crashes or aborts mid-heat — a pod fleet dying, the trainer restarting —
        must never consume a miner's single lifetime submission without having
        actually screened it. Entrants whose own generator failed to train or
        score DO burn (that was their shot); a round-level failure before this
        point burns no one and the field simply re-enters the retried round.
        """
        if not self.cfg.round.one_submission_per_hotkey or not challengers:
            return
        path = self._submissions_path()
        seen = _load_seen_hotkeys(path)
        _save_seen_hotkeys(path, seen | {c.hotkey for c in challengers})

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
    ) -> tuple[TrainResult, str, int, int]:
        """Fetch generator (registry) → build corpus → train into ``out_dir``,
        streaming per-step metrics to S3. No upload — the caller decides whether
        the checkpoint is uploaded (final) or thrown away after screening (heat).

        ``contract`` is the per-size training contract (the base recipe with this
        size's width/depth/digest/throughput); ``token_budget`` is its compute
        budget for this stage. Returns ``(result, corpus_digest, n_series,
        total_points)``. Raises on any failure.
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

        with open_round_stream(
            contract.corpus_mode,
            gen_dir, seeds.generation_seed, self.cfg.generator,
            token_budget=token_budget,
            use_sandbox=self.use_sandbox,
            blocked=self.cfg.static_guard.blocked,
            max_wall_seconds=contract.max_train_seconds,
        ) as rs:
            result = self.base_trainer.train(
                rs.series(),
                contract,
                training_seed=seeds.training_seed,
                token_budget=token_budget,
                out_dir=out_dir,
                logger=logger,
            )
            corpus_digest, n_series, total_points = rs.digest, rs.n_series, rs.total_points

        summary = {"event": "summary", "role": log_role, "corpus_digest": corpus_digest,
                   "n_series": n_series, "total_points": total_points,
                   "train_seconds": result.train_seconds, **result.metrics}
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
        self._round_telemetry[stage].append(dict(m))
        return result, corpus_digest, n_series, total_points

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

        Raises on any failure; the caller decides whether a failed challenger
        simply doesn't qualify (it does) or a failed king aborts the round (it
        does — there's nothing to defend against).
        """
        contract = contract if contract is not None else self.cfg.training.primary_size
        token_budget = token_budget if token_budget is not None else contract.train_tokens
        size = contract.arch_preset
        out_dir = self.work_root / f"{seeds.base_seed}" / size / f"{role}{repo_suffix}" / "checkpoint"
        result, corpus_digest, _, _ = self._train_checkpoint(
            gen, seeds, contract, token_budget, out_dir, log_role=f"{role}-{size}",
        )

        ckpt_repo = f"{self.hub().namespace}/ckpt-r{seeds.base_seed}-{role}-{size}{repo_suffix}"
        # Hub is priority-one; mirror to HF only if the Hub is down (keeps a round
        # alive through a Hub outage instead of failing the checkpoint upload).
        up = upload_dir_to_hub_or_hf(
            result.local_dir, ckpt_repo, self.hub(),
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
            gpu_name=str(result.metrics.get("gpu_name", "")),
            size=size,
        )

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

    def _stamp_king_bench_scores(
        self, entries: list[TrainedEntry], seeds: RoundSeeds
    ) -> list[TrainedEntry]:
        """Score the king's checkpoint on GIFT-Eval / BOOM / TIME and return the
        entries with those numbers stamped onto the king's (primary throne size)
        entry. Best-effort: any failure logs and returns the entries unchanged, so
        a benchmark hiccup never fails a round — validators then fall back to
        scoring the checkpoint themselves."""
        primary = self.cfg.throne_contracts()[0].arch_preset
        king_idx = next(
            (i for i, e in enumerate(entries)
             if e.role == "king" and (e.size == primary or e.size == "")),
            next((i for i, e in enumerate(entries) if e.role == "king"), None),
        )
        if king_idx is None:
            return entries
        king = entries[king_idx]
        arch_preset = king.size or primary
        try:
            if self.cascade_bench_plan is not None and self.remote_hosts:
                # Bench on the pod that just trained the king: GPU, and the
                # checkpoint is already at its _train_work path (no local fetch).
                scores = self._remote_king_bench_scores(str(seeds.base_seed), arch_preset)
            elif self.bench_eval_fn is not None:
                ckpt = self._fetch_checkpoint_dir(king.trained_pointer)
                scores = self.bench_eval_fn(ckpt)
            else:
                scores = None
        except Exception as e:  # noqa: BLE001 — Cascade telemetry must never fail a round
            log.warning("round=%s: king bench eval failed (%s); manifest omits bench_scores",
                        seeds.base_seed, e)
            return entries
        if scores is None:
            log.warning("round=%s: king bench eval produced no complete score set; "
                        "manifest omits bench_scores", seeds.base_seed)
            return entries
        log.info(
            "round=%s: stamped king bench_scores gift(crps=%.5f mase=%.5f) "
            "boom(crps=%.5f mase=%.5f) time(crps=%.5f mase=%.5f)",
            seeds.base_seed, scores.gifteval_crps, scores.gifteval_mase,
            scores.boom_crps, scores.boom_mase, scores.time_crps, scores.time_mase,
        )
        entries[king_idx] = replace(king, bench_scores=scores)
        return entries

    def _remote_king_bench_scores(self, round_id: str, arch_preset: str) -> BenchScores | None:
        """Score the round's king on GIFT-Eval/BOOM/TIME on the pod that trained it
        (GPU; the checkpoint is already at its ``_train_work`` path) and parse the
        six signed numbers. Reuses the post-round-benchmark remote path; best-effort
        — returns None on any miss, so the manifest simply omits ``bench_scores``."""
        from ..eval.benchmarks import extract_bench_scores
        from .bench_hook import run_post_round_benchmark

        host = self.remote_hosts[0]  # the king trains on the first pod (single-worker today)
        report = run_post_round_benchmark(
            host, round_id, arch_preset, self.cascade_bench_plan, work_root=self.work_root,
        )
        scores = extract_bench_scores(report) if report is not None else None
        return BenchScores(**scores) if scores is not None else None

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

        # The screener keys a daily-snapshot eval pool by the round's epoch
        # boundary. The live loop supplies it as ``cutoff_block``; derive it for
        # direct callers (scripts, operators) so a bucket-backed pool never
        # silently screens on a NEWER snapshot than the validator will judge the
        # final on (``None`` would mean "newest").
        screen_block = cutoff_block
        if screen_block is None:
            epoch_blocks = max(1, self.cfg.round.epoch_blocks)
            screen_block = (block // epoch_blocks) * epoch_blocks

        eligible = self._filter_burned_challengers(plan.challengers)
        finalists, heat = self._run_heat(eligible, seeds, block,
                                         screen_block=screen_block)
        # Burn only now, after the heat stage completed: every eligible entrant
        # got its screening attempt (or its pass-through to the final). A crash
        # mid-heat leaves the burn set untouched, so no miner's one lifetime
        # submission is consumed by a round that never judged it.
        self._burn_hotkeys(eligible)
        # Heat settled (screened + burned + finalists chosen): signal external
        # watchers (the provisioner) that heat-stage pods are now safe to release.
        self._mark_heat_complete(base_seed, eligible, finalists)
        self._log_telemetry_rollup(base_seed)  # heat-stage standings so far
        jobs: list[tuple[ResolvedGenerator, str]] = [(plan.king, "king")]
        jobs += [(c, "challenger") for c in finalists]

        entries = self._train_final(jobs, seeds, block)
        entries = _drop_final_content_clones(entries, jobs)
        if not any(e.role == "king" for e in entries):
            raise RuntimeError("king training produced no entry; aborting round")
        self._log_telemetry_rollup(base_seed)  # complete heats + finals picture

        # Cascade: score the king's checkpoint on the public suites and stamp the
        # numbers onto its manifest entry, so every validator promotes off one
        # signed set (see cascade.validator.cascade). Best-effort and gated on
        # [scoring] cascade_enabled; a failure just leaves bench_scores unset.
        if self.cfg.scoring.cascade_enabled and (
            self.bench_eval_fn is not None or self.cascade_bench_plan is not None
        ):
            entries = self._stamp_king_bench_scores(entries, seeds)

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

        return TrainingManifest(
            round_id=str(base_seed),
            created_block=block,
            contract_digest=contract_digest(self.cfg.training),
            base_arch_digest=self.cfg.training.base_arch_digest,
            eval_dataset=self.cfg.eval.eval_dataset,
            entries=entries,
            heat=heat,
            eval_pool_key=str(pool_key or ""),
            eval_pool_sha256=str(pool_sha or ""),
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
    ) -> tuple[list[ResolvedGenerator], HeatResult | None]:
        """Screen the field down to ``[round] finalists`` for the final stage.

        Each challenger is trained for ``[round] heat_train_hours`` on the primary
        (smallest) size and scored by the injected ``screen_fn`` (lower is
        better); the cheapest ``finalists`` advance, UID breaking ties for
        determinism. When the field already fits within ``finalists``, or no
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
        """
        n = max(0, self.cfg.round.finalists)
        if not challengers or n == 0:
            return [], None
        if self.screen_fn is None or len(challengers) <= n:
            if self.screen_fn is None and len(challengers) > n:
                log.warning("no screen_fn wired; taking %d of %d challengers by UID order",
                            n, len(challengers))
            return list(challengers[:n]), None

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
        trained = self._heat_train(challengers, seeds, block, heat_contract, heat_tokens)
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
        for c, ckpt_dir, _ in trained:
            if c.hotkey in duplicates:
                continue
            try:
                score = float(self.screen_fn(ckpt_dir, c, seeds.base_seed, screen_block))
            except Exception as e:  # noqa: BLE001 — a broken heat entry just doesn't qualify
                log.warning("heat: challenger %s failed to screen: %s", c.hotkey, e)
                continue
            log.info("heat: challenger %s score=%.5f", c.hotkey, score)
            scored.append((score, c.uid, c))

        scored.sort(key=lambda t: (t[0], t[1]))  # lower score better; UID tiebreak
        winners = [c for _, _, c in scored[:n]]
        log.info("heat: %d/%d advance to the final: %s",
                 len(winners), len(challengers), [c.hotkey for c in winners])
        heat = self._heat_result(
            challengers, scored, winners, trained_hotkeys, heat_contract.arch_preset, n,
            duplicates=duplicates,
        )
        return winners, heat

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
    ) -> HeatResult:
        """Assemble the informational standings from a completed heat.

        Scores are recorded only *relative to the best entrant* (``score / best``)
        — the raw numbers stay off the public record so the private, per-round
        rotated eval pool can't be reverse-engineered from the heat. Entrants that
        never produced a score are carried too, tagged by how they dropped out:
        ``duplicate`` (corpus byte-identical to an earlier reveal), ``failed_train``
        (crashed the screen budget) or ``failed_screen`` (trained but the scorer
        raised).
        """
        advanced = {c.hotkey for c in winners}
        scored_hotkeys = {c.hotkey for _, _, c in scored}
        best = scored[0][0] if scored else None
        entrants: list[HeatEntrant] = []
        for rank, (score, _uid, c) in enumerate(scored, start=1):
            rel = (score / best) if (best is not None and best > 0) else None
            entrants.append(HeatEntrant(
                uid=c.uid, hotkey=c.hotkey, gen_ref=c.ref,
                status="advanced" if c.hotkey in advanced else "screened",
                rank=rank, rel_score=rel,
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
        return HeatResult(screen_size=screen_size, finalists=finalists, entrants=tuple(entrants))

    def _heat_train(
        self,
        challengers: list[ResolvedGenerator],
        seeds: RoundSeeds,
        block: int,
        heat_contract: TrainingContractConfig,
        heat_tokens: int,
    ) -> list[tuple[ResolvedGenerator, Path, str]]:
        """Train each heat challenger, returning ``[(challenger, local_ckpt_dir,
        corpus_digest)]`` for the ones that trained — the digest feeds the
        content-level duplicate drop in :meth:`_run_heat`. Dispatches to
        ``remote_hosts`` (GPU pods) when configured — the pod trains at the cheap
        heat budget and the checkpoint is fetched back for local screening, so the
        orchestrator (with the wallet) never needs a GPU — else trains locally. A
        failed train drops that challenger (it just doesn't qualify)."""
        if self.remote_hosts:
            return self._heat_train_remote(challengers, seeds, block, heat_contract)
        out: list[tuple[ResolvedGenerator, Path, str]] = []
        for c in challengers:
            out_dir = self.work_root / f"{seeds.base_seed}" / "heat" / c.hotkey / "checkpoint"
            try:
                result, digest, _, _ = self._train_checkpoint(
                    c, seeds, heat_contract, heat_tokens, out_dir, log_role=f"heat-{c.hotkey}",
                )
                out.append((c, result.local_dir, digest))
            except Exception as e:  # noqa: BLE001
                log.warning("heat: challenger %s failed to train: %s", c.hotkey, e)
        return out

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
    def _dispatch_with_retry(disp, hosts: list, i: int, *, describe: str, **kw):
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
        ``remote.pod_lane_count`` / ``sandbox._lane_cpu_slice``)."""
        from .remote import pod_lane_count

        host = hosts[i % len(hosts)]
        try:
            return disp.dispatch(host, lane_count=pod_lane_count(host, hosts), **kw)
        except Exception as e:  # noqa: BLE001 — any dispatch failure is retryable once
            retry_host = hosts[(i + 1) % len(hosts)]
            log.warning("%s failed on %s (%s); retrying on %s", describe,
                        getattr(host, "name", host), e, getattr(retry_host, "name", retry_host))
            return disp.dispatch(retry_host, lane_count=pod_lane_count(retry_host, hosts), **kw)

    def _heat_train_remote(
        self,
        challengers: list[ResolvedGenerator],
        seeds: RoundSeeds,
        block: int,
        heat_contract: TrainingContractConfig,
    ) -> list[tuple[ResolvedGenerator, Path, str]]:
        """Screen-train the field on the GPU pods: dispatch each challenger to a
        host (round-robin across ``remote_hosts``, in parallel), training at the
        cheap ``[round] heat_train_hours`` on the screen size, then fetch each
        checkpoint back for local screening. Each pushes to a per-challenger repo
        so concurrent heat runs never collide. A challenger that fails to train or
        fetch is dropped. The pod's receipt carries the corpus digest, threaded
        through for the content-level duplicate drop."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from .remote import RemoteDispatcher

        if not self.trainer_spec:
            raise RuntimeError("remote heat requires trainer_spec (BaseTrainer 'module:Class')")
        hosts = self._hosts_for("heat")
        hub = self.hub()  # pre-init (thread-safe) before the pool
        # Heat dispatches get a TIGHT SSH timeout: the pod-side guard already
        # kills a slow run at the scaled max_train_seconds, so the only thing a
        # long outer timeout buys is a wedged pod (kernel hang, dead network
        # where SSH never returns) holding a heat slot for the full 6h default.
        # Guard + 30min covers fetch/sandbox/upload overheads around training.
        heat_timeout = min(self.remote_timeout_seconds, heat_contract.max_train_seconds + 1800)
        disp = RemoteDispatcher(trainer_spec=self.trainer_spec, timeout_seconds=heat_timeout)

        def _run(i: int, c: ResolvedGenerator) -> tuple[ResolvedGenerator, Path, str]:
            entry = self._dispatch_with_retry(
                disp, hosts, i, describe=f"heat challenger {c.hotkey}",
                gen_ref=c.ref, uid=c.uid, hotkey=c.hotkey, role="challenger",
                base_seed=seeds.base_seed, block=block,
                arch_preset=heat_contract.arch_preset,
                train_hours=self.cfg.round.heat_train_hours,
                repo_suffix=f"-heat-u{c.uid}",
            )
            ref = parse_trained_pointer(entry.trained_pointer)
            if ref is None:
                raise RuntimeError(f"malformed trained_pointer: {entry.trained_pointer!r}")
            out_dir = self.work_root / f"{seeds.base_seed}" / "heat" / c.hotkey / "checkpoint"
            fetch_from_hub(ref, out_dir, hub)
            return c, out_dir, entry.corpus_digest

        out: list[tuple[ResolvedGenerator, Path, str]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as ex:
            futs = {ex.submit(_run, i, c): c for i, c in enumerate(challengers)}
            for fut in as_completed(futs):
                c = futs[fut]
                try:
                    out.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    log.warning("heat: challenger %s failed on remote: %s", c.hotkey, e)
        return out

    def _train_final(
        self, jobs: list[tuple[ResolvedGenerator, str]], seeds: RoundSeeds, block: int
    ) -> list[TrainedEntry]:
        """Train the final jobs at each throne size, returning all receipts.

        One (king + finalists) pass per size in ``cfg.throne_contracts()`` (the
        ``[round] throne_sizes``); a king failure at any size aborts the round, a
        challenger failure drops only that challenger from that size."""
        if not self.remote_hosts:
            # This box is the runtime for a local final; with remote hosts the
            # check runs on each pod (cascade-train-worker), which is the runtime.
            assert_train_image(self.cfg.training)
        entries: list[TrainedEntry] = []
        for contract in self.cfg.throne_contracts():
            token_budget = contract.train_tokens
            if self.remote_hosts:
                entries += self._train_remote(jobs, seeds, block, contract, token_budget)
            else:
                entries += self._train_local(jobs, seeds, block, contract, token_budget)
        return entries

    def _train_local(
        self,
        jobs: list[tuple[ResolvedGenerator, str]],
        seeds: RoundSeeds,
        block: int,
        contract: TrainingContractConfig,
        token_budget: int,
    ) -> list[TrainedEntry]:
        """Sequential training on this box for one size: king first (its failure
        aborts the round), then each challenger (a failure just drops it)."""
        entries: list[TrainedEntry] = []
        for gen, role in jobs:
            try:
                entries.append(
                    self.train_one(gen, role, seeds, block,
                                   contract=contract, token_budget=token_budget)
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
            trainer_spec=self.trainer_spec, timeout_seconds=self.remote_timeout_seconds
        )

        def _run(i: int, gen: ResolvedGenerator, role: str) -> TrainedEntry:
            return self._dispatch_with_retry(
                disp, hosts, i, describe=f"final {role} {gen.hotkey}",
                gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey,
                role=role, base_seed=seeds.base_seed, block=block,
                arch_preset=contract.arch_preset,
            )

        results: list[TrainedEntry | None] = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=max(1, len(hosts))) as ex:
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

    # ── live loop ────────────────────────────────────────────────────────────

    def _reload_remote_hosts(self) -> None:
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
        """
        if self.remote_hosts_path is None:
            return
        from .remote import RemoteDispatchError, load_hosts

        deadline = time.time() + max(0, self.hosts_wait_seconds)
        while True:
            try:
                hosts = load_hosts(self.remote_hosts_path)
            except RemoteDispatchError as e:
                hosts, reason = None, str(e)
            else:
                reason = ""
            if hosts:
                if self.remote_hosts is None or [h.name for h in hosts] != [
                    h.name for h in self.remote_hosts
                ]:
                    log.info("round fleet: %d pod(s): %s",
                             len(hosts), ", ".join(h.name for h in hosts))
                self.remote_hosts = hosts
                return
            if time.time() >= deadline:
                log.warning("no remote hosts available (%s); training locally this round",
                            reason or str(self.remote_hosts_path))
                self.remote_hosts = None
                return
            time.sleep(min(15.0, max(1.0, deadline - time.time())))

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
        epoch_blocks = max(1, self.cfg.round.epoch_blocks)
        last_round: str | None = None
        while True:
            try:
                block = client.current_block()
                epoch = block // epoch_blocks
                epoch_start = epoch * epoch_blocks
                base_seed = client.block_seed(epoch_start)
                round_id = str(base_seed)
                if round_id == last_round:
                    time.sleep(poll)
                    continue
                commitments = client.poll_commitments()
                king_hotkey = client.highest_incentive_hotkey()
                self._reload_remote_hosts()  # per-round elastic fleet pickup
                log.info("starting round=%s epoch=%d epoch_start=%d king=%s field=%d",
                         round_id, epoch, epoch_start, king_hotkey, len(commitments))
                manifest = self.run_round(
                    commitments, king_hotkey, base_seed, block, cutoff_block=epoch_start,
                )
                self.publish(manifest)
                if self.bench_plan is not None and self.remote_hosts:
                    # Guarded separately: the round is already published, so a
                    # telemetry failure here must not fall through to the round
                    # handler and re-run (re-train + re-publish) it next poll.
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
