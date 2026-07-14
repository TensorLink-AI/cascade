"""Validator loop — manifest → eval → KOTH decision → weights.

The validator never trains. Each round it:

1. Reads the current :class:`TrainingManifest` from the owner dataset repo and
   verifies its signature + that king and challenger share the contract digest
   (the controlled-experiment guarantee).
2. Pulls the king's and challenger's trained checkpoints and scores both on the
   *same* held-out eval windows.
3. Runs the paired-bootstrap KOTH verdict and folds it into the sticky
   champion state (``dethrone_cp`` consecutive wins to take the throne;
   ``dethrone_cp = 1`` makes it single-round).
4. Sets weights: equal share across the current king plus up to
   ``[scoring] reward_prior_kings`` registered prior kings (teutonic-style),
   collapsing to winner-take-all when ``reward_prior_kings = 0``.

The pure orchestration in :meth:`ValidatorRunner.process_round` is testable by
injecting ``evaluate_fn`` and ``windows``; HF + torch + chain are isolated
behind the defaults.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING

from ..eval.koth import RoundResult, evaluate_round
from ..eval.scoring import WindowScore
from ..eval.window import EvalWindow
from ..shared.config import ChainConfig
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    parse_trained_pointer,
    verify_signature,
)
from ..shared.receipt import (
    EntryScores,
    EvalContext,
    Participant,
    RoundReceipt,
    VerdictRecord,
    WindowScoreRecord,
    build_receipt,
)
from . import state as state_mod
from .state import ChampionState, StateTransition

if TYPE_CHECKING:
    from ..trainer.remote import RemoteHost
    from .cascade import CascadeController

log = logging.getLogger("cascade.validator")

# Resolve a trained entry to its per-window scores on the eval set.
EvaluateFn = Callable[[TrainedEntry, list[EvalWindow]], list[WindowScore]]
# Resolve a trained entry to its gift-eval ratio rows for the public-benchmark
# gate: ``{"status", "rows", "revision"}`` (see ``eval.benchmarks.run_gift_rows``)
# or ``None`` when the sidecar produced nothing.
GiftRowsFn = Callable[[TrainedEntry], dict | None]


def participants_from_commitments(commitments: list, cutoff_block: int) -> tuple[Participant, ...]:
    """The round's eligible participant set, for the public receipt.

    Mirrors the trainer's eligibility rule (``trainer.loop.resolve_commitments``):
    parseable generator pointers revealed STRICTLY BEFORE the epoch boundary,
    latest commit per hotkey among the eligible ones — but keeps ``commit_block``
    so an auditor can re-check every entrant against the cutoff. Sorted by UID
    for a deterministic receipt body.
    """
    from ..interface.validation import parse_commit

    best: dict[str, Participant] = {}
    for c in commitments:
        if c.commit_block >= cutoff_block:
            continue
        parsed = parse_commit(c.payload)
        if parsed is None:
            continue
        prev = best.get(c.hotkey)
        if prev is None or c.commit_block >= prev.commit_block:
            best[c.hotkey] = Participant(
                hotkey=c.hotkey, uid=c.uid, gen_ref=parsed.ref, commit_block=c.commit_block
            )
    return tuple(sorted(best.values(), key=lambda p: p.uid))


@dataclass(frozen=True)
class RoundOutcome:
    result: RoundResult
    transition: StateTransition
    # Every per-window score that fed the verdict, one record per evaluated
    # (role, size) entry in evaluation order — threaded out so the live loop can
    # publish them in the round's signed public receipt (cascade.shared.receipt).
    entry_scores: tuple[EntryScores, ...] = ()
    # The king's tenure AT decision time (it set the margin); recorded in the
    # receipt so an auditor can recompute margin_for_tenure without validator state.
    king_tenure_rounds: int = 0


@dataclass
class ValidatorRunner:
    cfg: ChainConfig
    state: ChampionState = field(default_factory=ChampionState)
    evaluate_fn: EvaluateFn | None = None     # injected in tests; defaults to registry+torch
    gift_rows_fn: GiftRowsFn | None = None    # injected in tests; defaults to the sidecar bridge
    cache_dir: Path | None = None
    device: str = "cpu"
    # Optional resolver for the GPU eval-offload pod, called AT EACH offloaded
    # eval (see cascade.validator.eval_offload.make_eval_host_fn): the pod is
    # elastic — rented per round by the provisioner, torn down on the receipt —
    # so it must be re-resolved lazily, not captured at startup. ``None`` (or a
    # call returning ``None``) ⇒ that eval runs on ``device`` locally. The
    # wallet and every consensus decision stay on this box either way; the pod
    # is never used for the private-pool duel.
    eval_host_fn: Callable[[], RemoteHost | None] | None = None
    verify_signatures: bool = True            # gate manifests on the trainer-hotkey signature
    # Cascade — king-reign promotion (see cascade.validator.cascade). When wired,
    # the reign clock is reset on each dethrone, every reigning-king checkpoint is
    # scored (GIFT-Eval + TIME) and logged, and once per round the clock is checked;
    # a fired Cascade vacates the champion throne to re-open the competition from
    # the promoted warm-start init. None ⇒ Cascade is disabled (pure KOTH).
    cascade: CascadeController | None = None

    # ── manifest gating ─────────────────────────────────────────────────────

    def check_manifest(self, manifest: TrainingManifest) -> str | None:
        """Return a rejection reason string, or None if the manifest is usable.

        Enforces (1) the trainer-hotkey signature, (2) the contract-digest match
        (king and challenger trained under the same terms), and (3) that the
        manifest targets our configured base architecture and eval dataset.
        """
        if self.verify_signatures and not verify_signature(manifest, self.cfg.manifest.trainer_hotkey):
            return "signature_invalid"
        want_contract = contract_digest(self.cfg.training)
        if manifest.contract_digest != want_contract:
            return f"contract_digest_mismatch: {manifest.contract_digest} != {want_contract}"
        if manifest.base_arch_digest != self.cfg.training.base_arch_digest:
            return "base_arch_digest_mismatch"
        if manifest.eval_dataset != self.cfg.eval.eval_dataset:
            return "eval_dataset_mismatch"
        gpu_reason = self._check_gpu(manifest)
        if gpu_reason is not None:
            return gpu_reason
        return None

    @staticmethod
    def check_pool_pin(
        manifest: TrainingManifest, window_source: object, *, block: int | None
    ) -> str | None:
        """Verify this round's eval pool against the trainer-signed pin.

        A pinned manifest carries the ``(key, sha256)`` of the snapshot the
        trainer screened on; the validator's own deterministic selection (same
        epoch block, same rule) must resolve to the identical pair. The pin is
        inside the signed body, so pool integrity descends from the trainer
        signature rather than the unsigned ``pool/index.json`` — a poisoned
        index or tampered tar surfaces here as a loud reject instead of scoring
        on attacker-chosen data. Unpinned manifests (older trainers) keep the
        legacy index-trust behaviour. Returns a reject reason or ``None``.
        """
        if not (manifest.eval_pool_key and manifest.eval_pool_sha256):
            return None
        prov_fn = getattr(window_source, "provenance_for_round", None)
        if prov_fn is None:
            return ("pool_pin_unverifiable: manifest pins the eval pool but this "
                    "validator's pool source reports no provenance")
        try:
            key, sha = prov_fn(int(manifest.round_id), block=block)
        except Exception as e:  # noqa: BLE001 — an unreadable pool must reject, not crash
            return f"pool_pin_unverifiable: provenance lookup failed: {e}"
        if not key or not sha:
            return ("pool_pin_unverifiable: manifest pins the eval pool but this "
                    "validator resolved no snapshot for the round")
        if (key, sha) != (manifest.eval_pool_key, manifest.eval_pool_sha256):
            return (f"pool_pin_mismatch: manifest signed {manifest.eval_pool_key}@"
                    f"{manifest.eval_pool_sha256[:12]}…, this validator resolved "
                    f"{key}@{sha[:12]}…")
        return None

    def _check_gpu(self, manifest: TrainingManifest) -> str | None:
        """Matched-hardware gate for byte-exact re-derivation.

        If ``[training] expected_gpu`` is pinned, every entry must report that GPU.
        Otherwise require only that king and challenger ran the same GPU (when both
        report one) — equal compute is already guaranteed by the token budget, but
        a byte-exact audit needs the comparison run on one SKU.
        """
        pinned = self.cfg.training.expected_gpu
        gpus = {e.gpu_name for e in manifest.entries if e.gpu_name}
        if pinned:
            bad = sorted(g for g in gpus if g != pinned)
            if bad or any(not e.gpu_name for e in manifest.entries):
                return f"gpu_mismatch: expected {pinned!r}, manifest has {sorted(gpus)!r}"
        elif len(gpus) > 1:
            return f"gpu_mismatch: king/challenger on different GPUs {sorted(gpus)!r}"
        return None

    # ── per-round decision ──────────────────────────────────────────────────

    def _fetch_checkpoint_dir(self, entry: TrainedEntry) -> Path:
        """Fetch a trained checkpoint from the Hippius Hub registry to a local
        dir and return it. The OCI digest in the ref pins the bytes, so the
        fetch is self-verifying; repeated fetches of the same ref land in the
        same digest-named dir (cheap to reuse)."""
        from ..shared.hippius import HubConfig, HubRef, fetch_from_hub

        ref = parse_trained_pointer(entry.trained_pointer)
        if ref is None:
            raise ValueError(f"malformed trained_pointer: {entry.trained_pointer!r}")
        hub = HubConfig.from_storage(self.cfg.storage)
        dest = Path(self.cache_dir or "./_eval_ckpts") / HubRef.parse(ref).digest.replace(":", "-")
        fetch_from_hub(ref, dest, hub)
        return dest

    def _evaluate(self, entry: TrainedEntry, windows: list[EvalWindow]) -> list[WindowScore]:
        if self.evaluate_fn is not None:
            return self.evaluate_fn(entry, windows)
        # Default path: fetch the checkpoint from the Hippius Hub registry and
        # score it (registry + torch).
        from .evaluator import evaluate_checkpoint

        dest = self._fetch_checkpoint_dir(entry)
        return evaluate_checkpoint(
            dest, windows, num_samples=self.cfg.eval.num_samples, device=self.device
        )

    # ── public-benchmark no-regression gate ─────────────────────────────────

    def _eval_host(self) -> RemoteHost | None:
        """The offload pod for THIS eval — resolved fresh every time so an
        elastic pod (rented per round manifest, torn down on the receipt)
        appears and disappears without a validator restart."""
        return self.eval_host_fn() if self.eval_host_fn is not None else None

    def _gift_rows(self, entry: TrainedEntry) -> dict | None:
        """Gift-eval ratio rows for one entry — injected in tests, else the
        sidecar bridge on the fetched checkpoint dir."""
        if self.gift_rows_fn is not None:
            return self.gift_rows_fn(entry)

        ec = self.cfg.eval
        dest = self._fetch_checkpoint_dir(entry)
        num_samples = ec.gift_gate_num_samples or ec.num_samples
        eval_host = self._eval_host()
        if eval_host is not None:
            # Offload the (heavy, paired) gift-eval to the GPU pod; the paired
            # bootstrap and every consensus decision stay on this box.
            from .eval_offload import gift_rows_via_host

            return gift_rows_via_host(
                eval_host, dest,
                datasets=ec.gift_gate_datasets,
                num_samples=num_samples,
                data_dir=(ec.gift_gate_data_dir or None),
                device="cuda",
                timeout_s=ec.gift_gate_timeout_s,
            )
        from ..eval.benchmarks import run_gift_rows

        return run_gift_rows(
            dest,
            project_dir=ec.benchmark_project_dir,
            datasets=ec.gift_gate_datasets,
            num_samples=num_samples,
            device=self.device,
            data_dir=(ec.gift_gate_data_dir or None),
            timeout_s=ec.gift_gate_timeout_s,
        )

    def _run_gift_gate(
        self,
        result: RoundResult,
        king_entry: TrainedEntry,
        chal_entry: TrainedEntry,
        *,
        seed: int | str,
        round_id: str,
    ) -> RoundResult:
        """Fold the public-benchmark gate into a *winning* round result.

        Scores both sides on gift-eval (via the sidecar bridge) and runs the
        paired no-regression bootstrap. The gate is uncomputable — and the
        round therefore inconclusive under ``enforce`` — when either sidecar run
        fails, gift-eval was skipped/errored, or the two runs scored against
        different pinned data revisions (a consensus-safety check: king and
        challenger must be judged on identical public data).
        """
        from ..eval.gift_gate import evaluate_gift_gate, uncomputable_gate
        from ..eval.koth import apply_gift_gate

        p = self.cfg.koth_params()
        mode = p.gift_gate_mode
        king_run = self._gift_rows(king_entry)
        chal_run = self._gift_rows(chal_entry)
        if (
            king_run is None or chal_run is None
            or king_run.get("status") != "ok" or chal_run.get("status") != "ok"
        ):
            gate = uncomputable_gate(p.gift_gate_tolerance, "gift-eval sidecar unavailable/errored")
        elif king_run.get("revision") != chal_run.get("revision"):
            gate = uncomputable_gate(
                p.gift_gate_tolerance,
                f"data-revision mismatch: king {king_run.get('revision')} != "
                f"chal {chal_run.get('revision')}",
            )
        else:
            gate = evaluate_gift_gate(
                king_run["rows"], chal_run["rows"],
                tolerance=p.gift_gate_tolerance,
                alpha=p.bootstrap_alpha,
                B=p.bootstrap_B,
                seed=seed,
                min_configs=p.gift_gate_min_configs,
            )
        log.info(
            "gift-gate round=%s mode=%s computed=%s passed=%s lcb=%s tol=%.4f "
            "n_configs=%d king_agg=%.5f chal_agg=%.5f%s",
            round_id, mode, gate.computed, gate.passed,
            f"{gate.lcb:.5f}" if gate.computed else "n/a", gate.tolerance,
            gate.n_configs, gate.king_agg, gate.chal_agg,
            "" if gate.computed else f" reason={gate.reason!r}",
        )
        return apply_gift_gate(result, gate, mode=mode)

    def _maybe_run_benchmarks(
        self, manifest: TrainingManifest, outcome: RoundOutcome | None
    ) -> None:  # pragma: no cover — exercised only in the live loop
        """Log public-benchmark numbers for a newly crowned king (log-only).

        Best-effort and strictly off the consensus path: it runs only when a
        challenger just dethroned the king, scores that new king's checkpoint via
        the isolated sidecar, and logs whatever comes back. Any failure is
        swallowed — a benchmark hiccup must never disturb weights or KOTH state.
        """
        ec = self.cfg.eval
        if not ec.run_benchmarks:
            return
        if outcome is None or not outcome.transition.dethroned:
            return
        new_king = manifest.entry_for_role("challenger")
        if new_king is None:
            return
        try:
            from ..eval.benchmarks import format_report, run_benchmarks

            ckpt = self._fetch_checkpoint_dir(new_king)
            report = run_benchmarks(
                ckpt,
                project_dir=ec.benchmark_project_dir,
                suites=ec.benchmark_suites or ("gift-eval", "boom", "time"),
                num_samples=ec.benchmark_num_samples or ec.num_samples,
                max_series=ec.benchmark_max_series,
                device=self.device,
            )
            if report is not None:
                log.info(
                    "benchmarks round=%s king=%s %s",
                    manifest.round_id, self.state.king_hotkey, format_report(report),
                )
        except Exception as e:  # noqa: BLE001 — log-only, never fatal
            log.warning("benchmark hook failed for round=%s: %s", manifest.round_id, e)

    # ── Cascade: king-reign promotion ────────────────────────────────────────

    def _current_king_entry(self, manifest: TrainingManifest) -> TrainedEntry | None:
        """The manifest checkpoint the reigning champion produced this round.

        Cascade times the *validator's champion*, not the manifest's (lagging)
        king role — so the checkpoint to score is the entry whose miner hotkey is
        the champion. Prefers the primary throne size (what the benchmark sidecar
        scores) and falls back to any size that hotkey trained."""
        hk = self.state.king_hotkey
        if hk is None:
            return None
        matches = [e for e in manifest.entries if e.miner_hotkey == hk]
        if not matches:
            return None
        primary = self.cfg.throne_contracts()[0].arch_preset
        return next((e for e in matches if e.size == primary), matches[0])

    @staticmethod
    def _bench_scores_dict(entry: TrainedEntry) -> dict | None:
        """The six Cascade numbers off a manifest entry's trainer-signed
        ``bench_scores`` (GIFT-Eval / BOOM / TIME CRPS+MASE), or ``None`` when the
        entry carries none. This is the authoritative, consensus-safe source: every
        validator reads the identical signed numbers."""
        bs = entry.bench_scores
        if bs is None:
            return None
        return {
            "gifteval_crps": bs.gifteval_crps, "gifteval_mase": bs.gifteval_mase,
            "boom_crps": bs.boom_crps, "boom_mase": bs.boom_mase,
            "time_crps": bs.time_crps, "time_mase": bs.time_mase,
        }

    def _bench_metrics_via_sidecar(self, entry: TrainedEntry) -> dict | None:  # pragma: no cover — sidecar glue
        """Fallback: score one checkpoint on GIFT-Eval, BOOM, and TIME via the
        out-of-process sidecar, returning the six numbers or ``None`` when any suite
        is missing/errored. Used only when the manifest carries no ``bench_scores``
        (e.g. a trainer that predates the Cascade hook). NOTE: independently-run GPU
        sweeps are not bit-reproducible, so this path is not consensus-safe across
        validators — prefer the trainer-signed numbers."""
        from ..eval.benchmarks import extract_bench_scores, run_benchmarks

        ec = self.cfg.eval
        ckpt = self._fetch_checkpoint_dir(entry)
        num_samples = ec.benchmark_num_samples or ec.num_samples
        eval_host = self._eval_host()
        if eval_host is not None:
            # Offload the cascade bench (GIFT-Eval+BOOM+TIME) to the GPU pod,
            # same seam as the gift-eval gate; the wallet stays on this box.
            from .eval_offload import bench_scores_via_host

            metrics = bench_scores_via_host(
                eval_host, ckpt,
                num_samples=num_samples,
                max_series=ec.cascade_bench_max_series,  # 0 = full battery
                data_dir=(ec.gift_gate_data_dir or None),
                device="cuda",
                timeout_s=ec.gift_gate_timeout_s,
            )
        else:
            report = run_benchmarks(
                ckpt,
                project_dir=ec.benchmark_project_dir,
                suites=("gift-eval", "boom", "time"),
                num_samples=num_samples,
                max_series=ec.cascade_bench_max_series,  # 0 = full battery
                device=self.device,
            )
            metrics = extract_bench_scores(report)
        if metrics is None:
            log.warning("cascade: incomplete GIFT-Eval/BOOM/TIME metrics for king checkpoint %s; "
                        "not recording this round", entry.trained_pointer)
        return metrics

    def _record_king_checkpoint(
        self, manifest: TrainingManifest, now: float
    ) -> None:  # pragma: no cover — sidecar glue
        """Add the reigning king's checkpoint to the reign log so a later Cascade
        selection is a lookup, not a re-eval. Prefers the trainer's signed
        ``bench_scores`` on the manifest (consensus-safe); falls back to scoring via
        the local sidecar only when the manifest carries none. Best-effort: a miss
        just means this round's checkpoint isn't a promotion candidate."""
        if self.cascade is None or self.state.king_hotkey is None:
            return
        entry = self._current_king_entry(manifest)
        if entry is None:
            return
        metrics = self._bench_scores_dict(entry) or self._bench_metrics_via_sidecar(entry)
        if metrics is None:
            return
        self.cascade.record_checkpoint(entry.trained_pointer, now=now, **metrics)

    def _cascade_round(
        self, manifest: TrainingManifest, outcome: RoundOutcome | None
    ) -> None:  # pragma: no cover — live-loop glue; the controller is unit-tested
        """One Cascade step, run at the end of a round (after weights/receipts, so
        the outgoing king still earns this round). Resets the reign clock on a
        dethrone, records the reigning king's checkpoint, then checks the clock —
        a fired Cascade vacates the champion throne so the field re-competes from
        the promoted init next round. Fully guarded: Cascade never disturbs KOTH."""
        if self.cascade is None:
            return
        import time

        now = time.time()
        try:
            # Reuse KOTH's dethrone signal to reset the clock (never reimplement it);
            # on genesis, crown the first champion so the reign clock starts ticking.
            if outcome is not None and outcome.transition.dethroned and outcome.transition.new_king_hotkey:
                self.cascade.note_dethrone(outcome.transition.new_king_hotkey, now=now)
            elif self.cascade.state.king_hotkey is None and self.state.king_hotkey is not None:
                self.cascade.note_dethrone(self.state.king_hotkey, now=now)
            self._record_king_checkpoint(manifest, now)
            event = self.cascade.cascade_check(now)
            if event is not None:
                self._apply_cascade(event)
        except Exception as e:  # noqa: BLE001 — Cascade must never disturb a round
            log.warning("cascade step failed for round=%s: %s", manifest.round_id, e)

    def _apply_cascade(self, event: object) -> None:  # pragma: no cover — live-loop glue
        """Vacate the champion throne after a Cascade so the competition re-opens:
        clear the king (tenure/streaks reset) and persist. Next round crowns
        whoever wins from the newly-installed warm-start init."""
        winner = getattr(event, "winner", None)
        old_king = getattr(event, "old_king", None)
        self.state = ChampionState()
        self._persist_state()
        log.info(
            "cascade: champion throne vacated (old king %s); field re-competes from "
            "checkpoint %s next round",
            (old_king or "?")[:12],
            getattr(winner, "checkpoint_id", "?"),
        )

    def process_round(
        self,
        manifest: TrainingManifest,
        windows: list[EvalWindow],
        base_seed: int | str,
    ) -> RoundOutcome | None:
        """Evaluate one manifest against the eval windows and update state.

        A round carries one (king, challenger) pair PER trained size (the primary
        plus any ``[[training.sizes]]``). Each size's pair is scored on the SAME
        windows, then the per-size scores are POOLED — king's across sizes vs
        challenger's across sizes, in identical order — and a single paired
        bootstrap decides ONE throne on the combined score (scaling-aware KOTH).
        Pooling preserves pairing because each size's king and challenger share
        the window ``abs_target``.

        Returns None (king holds, no state change) when the manifest carries no
        size with both a king and a challenger, or fails the contract gate.
        Otherwise returns the round outcome with the (already-applied) transition.
        """
        reason = self.check_manifest(manifest)
        if reason is not None:
            log.warning("rejecting manifest round=%s: %s", manifest.round_id, reason)
            return None

        king_by_size = {e.size: e for e in manifest.entries_for_role("king")}
        chal_by_size = {e.size: e for e in manifest.entries_for_role("challenger")}
        paired_sizes = [s for s in manifest.sizes() if s in king_by_size and s in chal_by_size]
        if not paired_sizes:
            log.info("manifest round=%s has no king/challenger pair; king holds", manifest.round_id)
            return None

        king_scores: list[WindowScore] = []
        chal_scores: list[WindowScore] = []
        score_records: list[EntryScores] = []
        for size in paired_sizes:
            import time as _time

            _t0 = _time.perf_counter()
            ks = self._evaluate(king_by_size[size], windows)
            _t_king = _time.perf_counter() - _t0
            _t1 = _time.perf_counter()
            cs = self._evaluate(chal_by_size[size], windows)
            _t_chal = _time.perf_counter() - _t1
            log.info(
                "round=%s eval-timing size=%s device=%s n_windows=%d num_samples=%d "
                "king=%.1fs challenger=%.1fs total=%.1fs",
                manifest.round_id, size, self.device, len(windows),
                self.cfg.eval.num_samples, _t_king, _t_chal, _t_king + _t_chal,
            )
            king_scores += ks
            chal_scores += cs
            for entry, scores in ((king_by_size[size], ks), (chal_by_size[size], cs)):
                score_records.append(EntryScores(
                    role=entry.role, size=size,
                    hotkey=entry.miner_hotkey, uid=entry.miner_uid,
                    scores=tuple(WindowScoreRecord.from_score(s) for s in scores),
                ))
        # One challenger generator competes at every size, so any size's entry
        # carries its identity for the KOTH state machine.
        chal_entry = chal_by_size[paired_sizes[0]]

        tenure_at_decision = self.state.tenure_rounds
        result = evaluate_round(
            king_scores,
            chal_scores,
            self.cfg.koth_params(),
            seed=base_seed,
            king_tenure_rounds=tenure_at_decision,
        )
        # Public-benchmark no-regression gate: only on a private-pool win, and
        # only when enabled. It can block a dethrone (or, uncomputable, hold the
        # round) but never grant one. Gated on the primary size's checkpoint
        # pair (the pooled decision spans sizes; the gate screens on one).
        if self.cfg.scoring.gift_gate_mode != "off" and result.challenger_wins_round:
            gate_size = (
                self.cfg.training.arch_preset
                if self.cfg.training.arch_preset in king_by_size and
                self.cfg.training.arch_preset in chal_by_size
                else paired_sizes[0]
            )
            result = self._run_gift_gate(
                result, king_by_size[gate_size], chal_by_size[gate_size],
                seed=base_seed, round_id=manifest.round_id,
            )
        transition = state_mod.apply_round(
            self.state,
            challenger_hotkey=chal_entry.miner_hotkey,
            challenger_uid=chal_entry.miner_uid,
            result=result,
            dethrone_cp=self.cfg.scoring.dethrone_cp,
            keep_former_kings=self.cfg.scoring.reward_prior_kings,
        )
        self.state = transition.state
        log.info(
            "round=%s lcb=%.4f margin=%.4f win=%s %s king=%s tenure=%d",
            manifest.round_id, result.lcb, result.margin, result.challenger_wins_round,
            transition.note, self.state.king_hotkey, self.state.tenure_rounds,
        )
        # Shadow diagnostics: never gate the verdict. A rank-based view that
        # disagrees with the LCB, or a per-domain win-rate sign flip, means the
        # pool composition is doing the deciding — alert-worthy, not decisive.
        if result.win_rate is not None:
            log.info(
                "round=%s diag n_clusters=%d win_rate=%.3f wilcoxon_p=%s per_domain=%s",
                manifest.round_id, result.n_clusters, result.win_rate,
                f"{result.wilcoxon_p:.4g}" if result.wilcoxon_p is not None else "n/a",
                {d: f"{wr:.2f}/n{n}" for d, (wr, n) in (result.per_domain_win_rate or {}).items()},
            )
        return RoundOutcome(
            result=result, transition=transition, entry_scores=tuple(score_records),
            king_tenure_rounds=tenure_at_decision,
        )

    def _epoch_start_block(self, manifest: TrainingManifest) -> int:
        """The round's epoch-boundary block: ``created_block`` floored to the
        epoch grid. Monotonic and identical for every validator (from the shared
        manifest), so it is the consensus key for daily eval-pool snapshot
        selection — unlike the round id, which is a block *hash* (non-monotonic).
        """
        epoch_blocks = max(1, self.cfg.round.epoch_blocks)
        return (manifest.created_block // epoch_blocks) * epoch_blocks

    # ── public round receipts ────────────────────────────────────────────────

    def build_round_receipt(
        self,
        manifest: TrainingManifest,
        *,
        base_seed: int,
        epoch_start_block: int,
        epoch_block_hash: str,
        outcome: RoundOutcome | None = None,
        windows: list[EvalWindow] | None = None,
        reject_reason: str | None = None,
        participants: tuple[Participant, ...] = (),
        pool_provenance: tuple[str, str] = ("", ""),
        reward_uids: tuple[int, ...] = (),
        weights: tuple[float, ...] = (),
        validator_hotkey: str = "",
    ) -> RoundReceipt:
        """Assemble the round's public receipt (pure — no I/O, no signing).

        A gated-out manifest — or one with no (king, challenger) pair to score —
        yields a ``rejected`` receipt carrying the reason; a scored round yields
        the full record: chain context, embedded manifest, participant set, the
        eval slice, every per-window score, the verdict, and the weight vector.
        """
        from ..trainer.contract import RoundSeeds

        seeds = RoundSeeds.derive(base_seed, self.cfg.training)
        if reject_reason is not None:
            return build_receipt(
                round_id=manifest.round_id, status="rejected",
                epoch_start_block=epoch_start_block, epoch_block_hash=epoch_block_hash,
                base_seed=base_seed, seeds=seeds, manifest=manifest,
                participants=participants, reject_reason=reject_reason,
                reward_uids=reward_uids, weights=weights,
                validator_hotkey=validator_hotkey,
            )
        if outcome is None or windows is None:
            raise ValueError("a scored receipt needs both outcome and windows")
        eval_context = EvalContext(
            pool_ref=pool_provenance[0],
            pool_digest=pool_provenance[1],
            window_ids=tuple(w.series_id for w in windows),
            n_windows=len(windows),
            num_samples=self.cfg.eval.num_samples,
        )
        verdict = VerdictRecord.from_round(
            outcome.result, outcome.transition,
            params=self.cfg.koth_params(), bootstrap_seed=base_seed,
            king_tenure_rounds=outcome.king_tenure_rounds,
        )
        return build_receipt(
            round_id=manifest.round_id, status="scored",
            epoch_start_block=epoch_start_block, epoch_block_hash=epoch_block_hash,
            base_seed=base_seed, seeds=seeds, manifest=manifest,
            participants=participants, eval_context=eval_context,
            entry_scores=outcome.entry_scores, verdict=verdict,
            reward_uids=reward_uids, weights=weights,
            validator_hotkey=validator_hotkey,
        )

    def _publish_round_receipt(
        self,
        client: object,
        manifest: TrainingManifest,
        base_seed: int,
        *,
        outcome: RoundOutcome | None = None,
        windows: list[EvalWindow] | None = None,
        reject_reason: str | None = None,
        window_source: object = None,
        reward_uids: tuple[int, ...] = (),
        weights: tuple[float, ...] = (),
    ) -> None:  # pragma: no cover — live-loop glue; assembly is unit-tested
        """Gather chain context, sign, and publish the round receipt.

        Best-effort by design: a receipt failure must never disturb weights or
        KOTH state (they are already committed), so every chain lookup degrades
        to an empty field and any publish error is logged and swallowed. The
        audit CLI treats missing context as WARN, not PASS.
        """
        from datetime import datetime

        from ..shared.hippius import (
            open_manifest_store,
            publish_receipt,
            update_receipt_index,
        )
        from ..shared.receipt import dump_receipt, sign_receipt, summarize_receipt

        try:
            epoch_start = self._epoch_start_block(manifest)
            epoch_hash = ""
            current_block: int | None = None
            participants: tuple[Participant, ...] = ()
            try:
                epoch_hash = client.block_hash(epoch_start)
                participants = participants_from_commitments(
                    client.poll_commitments(), cutoff_block=epoch_start
                )
                # Anchor for the dashboard's next-round countdown (best-effort;
                # the client extrapolates block→wall-clock from this + as_of).
                current_block = int(client.current_block())
            except Exception as e:  # noqa: BLE001 — chain context is best-effort
                log.warning("receipt chain context unavailable for round=%s: %s",
                            manifest.round_id, e)
            provenance = ("", "")
            prov_fn = getattr(window_source, "provenance_for_round", None)
            if prov_fn is not None:
                try:
                    # Same epoch block that selected the round's windows, so the
                    # recorded provenance is the pool actually scored.
                    provenance = tuple(prov_fn(base_seed, block=epoch_start))
                except Exception as e:  # noqa: BLE001
                    log.warning("pool provenance unavailable for round=%s: %s",
                                manifest.round_id, e)
            wallet = getattr(client, "wallet", lambda: None)()
            hotkey_ss58 = str(getattr(getattr(wallet, "hotkey", None), "ss58_address", "") or "")

            receipt = self.build_round_receipt(
                manifest,
                base_seed=base_seed,
                epoch_start_block=epoch_start,
                epoch_block_hash=str(epoch_hash),
                outcome=outcome,
                windows=windows,
                reject_reason=reject_reason,
                participants=participants,
                pool_provenance=(provenance[0], provenance[1]),
                reward_uids=reward_uids,
                weights=weights,
                validator_hotkey=hotkey_ss58,
            )
            if wallet is not None:
                receipt = sign_receipt(receipt, wallet)
            else:
                log.warning("publishing an UNSIGNED receipt (no wallet) for round=%s",
                            manifest.round_id)
            store = open_manifest_store(self.cfg.storage)
            key = publish_receipt(store, dump_receipt(receipt), manifest.round_id,
                                  validator_hotkey=hotkey_ss58)
            log.info("published %s receipt round=%s signed=%s → s3://%s/%s",
                     receipt.status, manifest.round_id, receipt.signature is not None,
                     self.cfg.storage.manifest_bucket, key)
            # Refresh the dashboard-facing rolling index (best-effort, and inside
            # the outer guard: a listing convenience must never disturb a round).
            try:
                now_iso = datetime.now(UTC).isoformat(timespec="seconds")
                # Schedule anchor for the "time until next round" countdown. The
                # next round begins at the next epoch boundary; the client turns
                # blocks into wall-clock via block_time_s, extrapolating current_block
                # from `as_of`. Bittensor blocks are ~12s regardless of epoch_blocks
                # (testnet shortens epochs, not block time), so it is a constant.
                chain = {
                    "as_of": now_iso,
                    "current_block": current_block,
                    "epoch_start_block": epoch_start,
                    "epoch_blocks": int(self.cfg.round.epoch_blocks),
                    "block_time_s": 12.0,
                }
                update_receipt_index(
                    store, summarize_receipt(receipt),
                    updated_at=now_iso,
                    subnet={"netuid": self.cfg.subnet.netuid, "name": self.cfg.subnet.name},
                    chain=chain,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("receipt index update failed for round=%s: %s",
                            manifest.round_id, e)
        except Exception as e:  # noqa: BLE001 — receipts must never disturb the round
            log.warning("receipt publication failed for round=%s: %s", manifest.round_id, e)

    # ── live loop ────────────────────────────────────────────────────────────

    def run_forever(self, client: object, *, window_source: object) -> None:  # pragma: no cover
        """Poll the manifest bucket → evaluate → set weights, once per round.

        ``window_source`` is a :class:`cascade.validator.windows.WindowSource`
        (the loaded private pool). Each new manifest's ``round_id`` is the base
        seed; the same seed drives the rotating window slice so every validator
        scores the identical set.
        """
        import time

        from ..shared.hippius import open_manifest_store, read_latest_manifest
        from ..shared.manifest import load_manifest

        store = open_manifest_store(self.cfg.storage)
        poll = self.cfg.manifest.poll_seconds
        last_round: str | None = None
        while True:
            try:
                manifest = load_manifest(read_latest_manifest(store))
                if manifest.round_id != last_round:
                    base_seed = int(manifest.round_id)
                    log.info(
                        "new manifest round=%s entries=%d (%s); gating + scoring …",
                        manifest.round_id, len(manifest.entries),
                        ",".join(f"{e.role}:uid{e.miner_uid}" for e in manifest.entries),
                    )
                    # Gate first so a rejected manifest never moves weights.
                    reason = self.check_manifest(manifest)
                    if reason is None:
                        # Pool-pin gate: the signed snapshot pin must match this
                        # validator's own deterministic selection for the round.
                        reason = self.check_pool_pin(
                            manifest, window_source,
                            block=self._epoch_start_block(manifest),
                        )
                    if reason is not None:
                        log.warning("rejecting manifest round=%s: %s", manifest.round_id, reason)
                        last_round = manifest.round_id
                        # A rejected round still gets a public receipt carrying
                        # the gate's reason — visible, not silently absent.
                        self._publish_round_receipt(
                            client, manifest, base_seed,
                            reject_reason=reason, window_source=window_source,
                        )
                    elif not self.king_synced(manifest):
                        # The trainer trained the OLD king (incentive lags a
                        # dethrone). Hold the KOTH state and keep voting the champion
                        # so incentive migrates and the trainer re-syncs — bounded by
                        # the safety valve (see _resync_step). A public receipt
                        # records why, not a silent skip.
                        last_round = manifest.round_id
                        self.state, reject_reason = self._resync_step(manifest)
                        self._persist_state()
                        reward_uids = self._reward_uids(manifest, None, client)
                        weights_vec = self._apply_weights(client, manifest.round_id, reward_uids)
                        self._publish_round_receipt(
                            client, manifest, base_seed,
                            reject_reason=reject_reason,
                            window_source=window_source,
                            reward_uids=tuple(reward_uids), weights=weights_vec,
                        )
                    else:
                        # The epoch block selects the daily snapshot; base_seed
                        # rotates the window slice within it.
                        windows = window_source.windows_for_round(
                            base_seed, self.cfg.eval.n_windows,
                            block=self._epoch_start_block(manifest),
                        )
                        # process_round mutates the sticky KOTH state atomically (it
                        # raises before any mutation on a transient eval/fetch error,
                        # leaving state untouched for a clean retry). Mark the round
                        # consumed as soon as it returns, so a later weight-set failure
                        # can NEVER re-run it and double-count the streak/tenure.
                        outcome = self.process_round(manifest, windows, base_seed)
                        # Back in sync — clear any accumulated resync holds so a
                        # future desync starts the safety-valve count from zero.
                        if self.state.resync_holds:
                            self.state = replace(self.state, resync_holds=0)
                        last_round = manifest.round_id
                        self._persist_state()
                        reward_uids = self._reward_uids(manifest, outcome, client)
                        weights_vec = self._apply_weights(client, manifest.round_id, reward_uids)
                        # The public receipt — strictly after weights, so it
                        # records what was actually set (empty vector = the
                        # weight extrinsic failed this round).
                        if outcome is not None:
                            self._publish_round_receipt(
                                client, manifest, base_seed,
                                outcome=outcome, windows=windows,
                                window_source=window_source,
                                reward_uids=tuple(reward_uids), weights=weights_vec,
                            )
                        else:
                            # Gated in but nothing to score (no king/challenger
                            # pair at any size): a public record still exists.
                            self._publish_round_receipt(
                                client, manifest, base_seed,
                                reject_reason="no_king_challenger_pair",
                                window_source=window_source,
                                reward_uids=tuple(reward_uids), weights=weights_vec,
                            )
                        # Log-only public benchmarks for a freshly crowned king.
                        # Strictly after weights are decided; never affects them.
                        self._maybe_run_benchmarks(manifest, outcome)
                        # Cascade step — strictly last, so this round's weights and
                        # receipt already recorded the outgoing king. Resets the
                        # reign clock on a dethrone, records the king's checkpoint,
                        # and fires the promotion when the clock is ripe.
                        self._cascade_round(manifest, outcome)
            except Exception as e:  # noqa: BLE001 — a service loop must not die on one round
                log.exception("round processing failed; retrying after poll: %s", e)
            time.sleep(poll)

    @staticmethod
    def _manifest_king_hotkey(manifest: TrainingManifest) -> str | None:
        e = manifest.entry_for_role("king")
        return e.miner_hotkey if e is not None else None

    @staticmethod
    def _manifest_king_uid(manifest: TrainingManifest) -> int | None:
        e = manifest.entry_for_role("king")
        return e.miner_uid if e is not None else None

    def king_synced(self, manifest: TrainingManifest) -> bool:
        """Whether the round's *trained* king matches the validator's champion.

        The trainer picks the king it trains from on-chain incentive, which lags
        the validator's dethrone verdicts (OPEN_QUESTIONS #3). Until the champion
        the validator crowned actually becomes the highest-incentive UID — and so
        the king the trainer trains — the two disagree, and a round trained
        against the *old* king must not have its verdict applied to the *new*
        champion. Synced when the champion is unset (bootstrap) or the trained
        king is the champion.
        """
        if self.state.king_hotkey is None:
            return True
        return self._manifest_king_hotkey(manifest) == self.state.king_hotkey

    def _resync_step(self, manifest: TrainingManifest) -> tuple[ChampionState, str]:
        """Next champion state + receipt reason for a king-resync round.

        Called when the trained king != champion (``king_synced`` is False). By
        default it holds the throne, bumping the consecutive-hold counter, and the
        caller keeps voting the champion so incentive migrates and the trainer
        re-syncs. SAFETY VALVE: once the champion has stayed un-synced for
        ``scoring.king_resync_max_rounds`` consecutive rounds it can never be the
        king the trainer trains (e.g. it has no usable commitment), so holding
        forever would wedge the subnet — the valve abandons it and adopts the
        trainer's trained king (:func:`state.demote_to_trained`), and normal
        scoring resumes next round. ``king_resync_max_rounds <= 0`` disables the
        valve (hold indefinitely). Pure: returns the new state; the caller
        persists, votes, and publishes.
        """
        champ = self.state.king_hotkey
        trained = self._manifest_king_hotkey(manifest)
        holds = self.state.resync_holds + 1
        cap = self.cfg.scoring.king_resync_max_rounds
        if 0 < cap <= holds and trained is not None:
            log.warning(
                "round=%s king_resync SAFETY VALVE: champion %s un-synced %d rounds "
                "(cap=%d) — demoting to trained king %s, resuming normal scoring next round",
                manifest.round_id, (champ or "?")[:12], holds, cap, (trained or "?")[:12],
            )
            return (
                state_mod.demote_to_trained(
                    self.state, trained_hotkey=trained,
                    trained_uid=self._manifest_king_uid(manifest),
                ),
                f"king_resync_demoted: champion {champ} un-synced {holds} rounds "
                f"(cap={cap}); adopted trained king {trained}",
            )
        log.warning(
            "round=%s trainer king %s != champion %s; voting champion to re-sync "
            "incentive, KOTH state held (%d/%s)",
            manifest.round_id, (trained or "?")[:12], (champ or "?")[:12],
            holds, cap if cap > 0 else "∞",
        )
        return (
            replace(self.state, resync_holds=holds),
            f"king_resyncing: champion {champ} != trained king {trained}",
        )

    def _king_uid_to_vote(self, manifest: TrainingManifest, *, client: object | None = None) -> int | None:
        """The UID to put the king's weight on this round.

        The **validator's champion state** is the authority on who holds the
        throne, so vote *that* king every round — not the (lagging) king the
        trainer happened to train. This is what makes a dethrone STICK: the new
        champion keeps the weight, incentive migrates to it, and next round the
        trainer trains it as king (they re-sync). Voting the trained/manifest
        king instead — the old behaviour — reverted a dethrone the moment the
        trainer lagged one round, orphaning the champion. The champion hotkey is
        resolved to its current UID via the metagraph (robust to re-registration);
        the manifest king is used only to bootstrap when there is no champion yet.
        """
        if self.state.king_hotkey is not None:
            if client is not None:
                resolved = client.uid_for_hotkey(self.state.king_hotkey)  # type: ignore[attr-defined]
                if resolved is not None:
                    return resolved
            return self.state.king_uid
        king_entry = manifest.entry_for_role("king")
        return king_entry.miner_uid if king_entry is not None else None

    def _reward_uids(
        self, manifest: TrainingManifest, outcome: RoundOutcome | None, client: object
    ) -> list[int]:
        """UIDs that share this round's weight: the current king plus any
        ``former_kings`` still registered (teutonic-style equal-share payout).

        Returns an empty list when there is no king to vote for at all (no
        champion and no manifest king); the loop hands that to
        ``set_equal_share_weights``, which burns to ``burn_uid`` rather than
        reverting. The list is otherwise deduped/range-checked there too.
        """
        uids: list[int] = []
        king_uid = self._king_uid_to_vote(manifest, client=client)
        if king_uid is not None:
            uids.append(king_uid)
        for hk in self.state.former_kings:
            uid = client.uid_for_hotkey(hk)  # type: ignore[attr-defined]
            if uid is not None:
                uids.append(uid)
        return uids

    def _apply_weights(self, client: object, round_id: str, reward_uids: list[int]) -> tuple[float, ...]:
        """Set the equal-share weight vector on chain; return it (empty on failure).

        Shared by the scored path and the king-resync path. Always sets weights —
        an empty ``reward_uids`` burns to ``burn_uid`` so emission still leaves the
        network. A failed extrinsic is logged and retried next round (the empty
        vector is recorded truthfully in the receipt)."""
        from ..shared.chain import decayed_share_vector

        decay = self.cfg.scoring.king_decay
        try:
            n_uids = client.n_uids()  # type: ignore[attr-defined]
            client.set_equal_share_weights(  # type: ignore[attr-defined]
                reward_uids, n_uids, decay=decay, burn_uid=self.cfg.scoring.burn_uid,
            )
            vec = tuple(decayed_share_vector(
                reward_uids, n_uids, decay=decay, burn_uid=self.cfg.scoring.burn_uid))
            log.info("round=%s weights set: reward_uids=%s (n_uids=%d, burn_uid=%d)",
                     round_id, reward_uids or [self.cfg.scoring.burn_uid], n_uids,
                     self.cfg.scoring.burn_uid)
            return vec
        except Exception as e:  # noqa: BLE001 — retried next round
            log.warning("weight set failed for round=%s (king holds, retried next round): %s",
                        round_id, e)
            return ()

    def _persist_state(self) -> None:  # pragma: no cover
        from . import state as state_mod

        try:
            Path(self.cfg.validator.state_db_path).write_text(
                state_mod.dumps(self.state), encoding="utf-8"
            )
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist validator state: %s", e)


def _load_state(path: str) -> ChampionState:
    """Load persisted champion state from ``state_db_path`` (JSON), or a fresh
    state if the file is absent/unreadable."""
    p = Path(path)
    if not p.is_file():
        return ChampionState()
    try:
        return state_mod.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning("could not load validator state from %s (%s); starting fresh", path, e)
        return ChampionState()


def _warm_start_installer(path: Path) -> Callable[[object], None]:
    """The default Cascade installer: promote the winning checkpoint by writing its
    pointer (and its eval numbers) to ``warm_start_init_path`` — the seam the
    trainer reads to warm-start every subsequent round from. Promotes AS-IS; no
    retrain/fine-tune."""

    def _install(winner: object) -> None:  # pragma: no cover — file glue
        import time

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "checkpoint_id": getattr(winner, "checkpoint_id", None),
                    "score": getattr(winner, "score", None),
                    "gifteval_crps": getattr(winner, "gifteval_crps", None),
                    "gifteval_mase": getattr(winner, "gifteval_mase", None),
                    "time_crps": getattr(winner, "time_crps", None),
                    "time_mase": getattr(winner, "time_mase", None),
                    "installed_at": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        log.info("cascade: warm-start init written to %s (checkpoint %s)",
                 path, getattr(winner, "checkpoint_id", "?"))

    return _install


def _build_cascade(cfg: ChainConfig) -> CascadeController:
    """Construct the Cascade controller from config, restoring the persisted reign
    clock + checkpoint log so it resumes across restarts."""
    from .cascade import CascadeController, load_state

    state_path = Path(cfg.validator.cascade_state_db_path)
    return CascadeController(
        reign_days=cfg.scoring.cascade_reign_days,
        state=load_state(state_path),
        install_fn=_warm_start_installer(Path(cfg.validator.warm_start_init_path)),
        state_path=state_path,
    )


def build_runner(
    *,
    chain_toml: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
    eval_host_fn: Callable[[], RemoteHost | None] | None = None,
) -> ValidatorRunner:
    """Construct a runner from ``chain.toml``, restoring persisted champion
    state. Wallet/chain wiring for live weight-setting is attached by
    ``cascade-validator`` (see main.py). ``eval_host_fn`` (optional) resolves
    the GPU pod to offload heavy evals to — re-invoked per eval, so an elastic
    provisioner-rented pod is picked up lazily; the wallet stays on this box."""
    from ..shared.config import load_chain_config

    cfg = load_chain_config(chain_toml)
    # Cascade is opt-in ([scoring] cascade_enabled); off ⇒ no controller is wired
    # and the runner is pure KOTH.
    cascade = _build_cascade(cfg) if cfg.scoring.cascade_enabled else None
    return ValidatorRunner(
        cfg=cfg, state=_load_state(cfg.validator.state_db_path),
        cache_dir=cache_dir, device=device, cascade=cascade, eval_host_fn=eval_host_fn,
    )
