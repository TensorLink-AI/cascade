"""Checkpoint evaluation, increment baseline and public-benchmark gates."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..eval.koth import RoundResult
from ..eval.scoring import WindowScore
from ..eval.window import EvalWindow
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    parse_trained_pointer,
)
from ..shared.receipt import EntryScores

if TYPE_CHECKING:
    from ..trainer.remote import RemoteHost
    from .loop import RoundOutcome

log = logging.getLogger("cascade.validator")


class EvaluationMixin:
    """Fetch and score checkpoints; the gift-eval gate and benchmark hook."""

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

    def _verdict_windows(
        self, window_source: object, round_seed: int | str, *, block: int | None
    ) -> list[EvalWindow]:
        """The round's verdict windows: the scored horizon ladder when active
        at this round's epoch block (``[eval] scored_horizons`` /
        ``scored_from_block``), else the single-horizon rotating draw. Both
        are deterministic in ``(round_seed, block)``, so every validator
        scores the identical set."""
        from .expert_lane import verdict_windows

        return verdict_windows(self.cfg, window_source, round_seed, block=block)

    def _evaluate(self, entry: TrainedEntry, windows: list[EvalWindow],
                  training=None) -> list[WindowScore]:
        if self.evaluate_fn is not None:
            return self.evaluate_fn(entry, windows)
        # Default path: fetch the checkpoint from the Hippius Hub registry and
        # score it (registry + torch).
        from .evaluator import evaluate_checkpoint

        dest = self._fetch_checkpoint_dir(entry)
        # The entry's contract arms the ingest guard: the checkpoint's code
        # must be this release's, its config the contract's, its weights the
        # pinned model's shape — a funded (miner-pod) checkpoint is untrusted.
        # ``training`` is the settlement's era contract (DEC-CA-0050 v2: the
        # MoE from the v2 block); None = the base [training].
        training = self.cfg.training if training is None else training
        size = getattr(entry, "size", "") or training.arch_preset
        contract = training.contract_for(size)
        private_host_fn = getattr(self, "private_eval_host_fn", None)
        workers = getattr(self, "private_eval_workers", 1)
        if private_host_fn is not None:
            from ..eval.scoring import score_joint_forecaster_on_windows
            from .eval_offload import PrivateEvalUnavailable, forecasts_via_host

            host = private_host_fn()
            if host is None:
                raise PrivateEvalUnavailable("configured private evaluation GPU host is unavailable")
            samples = iter(forecasts_via_host(
                host, dest, windows, contract=contract, num_samples=self.cfg.eval.num_samples,
                workers=workers, timeout_s=self.private_eval_timeout_s))
            return score_joint_forecaster_on_windows(
                lambda *_: next(samples), windows, self.cfg.eval.num_samples)
        return evaluate_checkpoint(
            dest, windows, num_samples=self.cfg.eval.num_samples, device=self.device,
            contract=contract, workers=workers,
        )

    @staticmethod
    def _entry_fault_reason(err: Exception) -> str | None:
        """The verdict reason when ``err`` is a DETERMINISTIC fault of the
        checkpoint under evaluation (``[scoring] entry_fault_from_block``),
        else ``None`` — a transient the settlement retries.

        Deterministic: the scorer's ``ValueError`` (non-finite or wrong-shape
        forecast samples, a malformed trained pointer) and every
        :class:`~cascade.validator.evaluator.EvaluatorError` (tampered
        checkpoint, wrapper import / construct / forward-shape failures) —
        the same bytes fail the same way on every validator. Transient: a
        storage / registry fetch failure (``StorageError``), an out-of-memory
        (even when the wrapper constructor wrapped it), and anything else."""
        from ..shared.hippius import StorageError
        from .evaluator import EvaluatorError

        if isinstance(err, StorageError):
            return None
        if isinstance(err, (EvaluatorError, ValueError)):
            text = f"{type(err).__name__}: {err}"
            if "out of memory" in text.lower() or "outofmemory" in text.lower():
                return None
            return text
        return None

    # Sentinel identity for the increment margin's third reference — the
    # shared warm-start init checkpoint. Not a miner: uid −1 is out of range
    # for every metagraph, and the audit locates its receipt rows by ROLE
    # ("baseline"), not by this name.
    BASELINE_HOTKEY = "__warm_start_init__"

    def _score_increment_baseline(
        self,
        manifest: TrainingManifest,
        paired_sizes: list[str],
        windows: list[EvalWindow],
        score_records: list[EntryScores],
    ) -> list[WindowScore] | None:
        """Score the shared warm-start init on the verdict windows.

        Serves both consumers of the baseline: the increment margin
        (DEC-CA-0027) and the init-baseline floor ([scoring] init_gate_mode).
        Appends the init's rows to the receipt either way (role="baseline").

        Returns ``None`` — judge at "level" / floor not applied — when the
        round has no baseline:
        a random-init round (no ``warm_start_ckpt``; the E2 init-round
        semantics), or a multi-size duel / size mismatch, where a single init
        cannot reference every paired size (weights never cross sizes).
        """
        if not manifest.warm_start_ckpt:
            log.info("round=%s baseline eval on a random-init round: no init "
                     "to score (increment judges at level; init floor skips)",
                     manifest.round_id)
            return None
        if len(paired_sizes) != 1 or manifest.warm_start_size != paired_sizes[0]:
            log.warning(
                "round=%s margin_mode=increment needs a single-size duel at the "
                "warm-start size (paired=%s, warm_start_size=%r); judging at level",
                manifest.round_id, paired_sizes, manifest.warm_start_size)
            return None
        size = paired_sizes[0]
        entry = TrainedEntry(
            miner_hotkey=self.BASELINE_HOTKEY, miner_uid=-1, role="king",
            gen_ref="", trained_pointer=manifest.warm_start_ckpt,
            corpus_digest="", train_block=0, size=size,
        )
        scores = self._evaluate(entry, windows, self._training_for(manifest))
        score_records.append(EntryScores(
            role="baseline", size=size, hotkey=self.BASELINE_HOTKEY, uid=-1,
            scores=self._window_records(scores, manifest),
        ))
        log.info("round=%s increment baseline scored: %d rows from %s",
                 manifest.round_id, len(scores), manifest.warm_start_ckpt)
        return scores

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

    def _bench_gate_side(self, round_id: str, entry: TrainedEntry) -> tuple[dict | None, dict | None]:
        """One checkpoint's ``(item rows, six numbers)`` from the round's
        trainer-signed bench report (DEC-CA-0054) — or, on a settlement whose
        lane-A rows come from the main validator (DEC-CA-0063), from its sweep
        / signed report — or ``(None, None)`` when the
        entry, its rows pointer, or rows matching the signed sha256 are absent.

        A store READ failure (network/auth/5xx) raises
        :class:`~cascade.shared.hippius.StorageError` instead: "could not read"
        is a transient the settlement retries (``process_settlement`` raises
        before any state mutation), never a verdict one validator reaches while
        its peers, reading fine, crown."""
        import dataclasses

        from ..shared.bench_report import load_bench_rows
        from ..shared.hippius import ObjectNotFound, StorageError

        vb = getattr(self, "_vbench", None)
        if vb is not None and vb.get("round_id") == str(round_id):
            # DEC-CA-0063: the main validator's own sweep (winners only), or
            # its signed report on every other validator
            return self._vbench_side(entry, role=str(entry.role or "challenger"))
        report = self._fetch_bench_report(round_id, require=entry.trained_pointer,
                                          raise_unavailable=True)
        e = report.entry_for_pointer(entry.trained_pointer) if report is not None else None
        if e is None or not e.rows_key:
            return None, None
        try:
            text = self._bench_store().get_text(e.rows_key)
        except ObjectNotFound:
            log.warning("bench-gate round=%s rows for %s not found at %s",
                        round_id, entry.miner_hotkey[:12], e.rows_key)
            return None, None
        except Exception as exc:  # noqa: BLE001 — store down / auth / network
            raise StorageError(f"bench_rows_unavailable: round={round_id}: {exc}") from exc
        try:
            rows = load_bench_rows(text, e.rows_sha256)
        except Exception as exc:  # noqa: BLE001 — forged/malformed rows ⇒ uncomputable
            log.warning("bench-gate round=%s rows for %s unusable: %s",
                        round_id, entry.miner_hotkey[:12], exc)
            return None, None
        return rows, dataclasses.asdict(e.scores)

    def _run_bench_gate(
        self,
        result: RoundResult,
        king_entry: TrainedEntry,
        chal_entry: TrainedEntry,
        *,
        seed: int | str,
        round_id: str,
        mode: str,
        rule: str = "dec-ca-0054",
    ) -> RoundResult:
        """Fold the bench gate into a *winning* round result. Under
        DEC-CA-0054 (``rule`` "dec-ca-0054") the challenger may not be
        significantly worse than the king on GIFT-Eval, BOOM or TIME (paired,
        on the trainer-signed rows every validator reads identically) and its
        six-number geomean must be better; under DEC-CA-0061
        ("avg_noninf_winrate") FEV joins and, per benchmark task, the
        suite-equal average ratio must be below 1, no suite's ratio LCB
        (alpha/4) above 1 + tolerance, and the win rate at least
        ``bench_gate_min_win_rate``. A missing side is uncomputable
        (inconclusive under enforce). ``mode`` / ``rule`` are the
        settlement's effective ones (``bench_gate_mode_at`` /
        ``bench_gate_rule_at``)."""
        from ..eval.bench_gate import evaluate_bench_gate, uncomputable_bench_gate
        from ..eval.koth import apply_bench_gate

        p = self.cfg.koth_params()
        thr = {"tolerance": p.bench_gate_tolerance, "min_win_rate": p.bench_gate_min_win_rate}
        self._bench_gate_infra = None
        self._bench_gate_deferred = None
        self._bench_gate_fault = None
        vb = getattr(self, "_vbench", None)
        if vb is not None and vb.get("round_id") == str(round_id):
            return self._run_vbench_gate(result, king_entry, chal_entry, seed=seed,
                                         round_id=round_id, mode=mode, rule=rule)
        king_rows, king_six = self._bench_gate_side(round_id, king_entry)
        chal_rows, chal_six = self._bench_gate_side(round_id, chal_entry)
        if king_rows is None or chal_rows is None:
            gate = uncomputable_bench_gate(
                "signed bench rows missing for "
                + ("king" if king_rows is None else "challenger"), rule=rule, **thr)
        else:
            gate = evaluate_bench_gate(
                king_rows, chal_rows, king_six, chal_six,
                alpha=p.bootstrap_alpha, B=p.bootstrap_B, seed=seed,
                min_items=p.bench_gate_min_items, rule=rule,
                tolerance=p.bench_gate_tolerance, min_win_rate=p.bench_gate_min_win_rate,
            )
        log.info(
            "bench-gate round=%s mode=%s rule=%s chal=%s computed=%s passed=%s ucb=%s n=%s "
            "king_six=%.5f chal_six=%.5f avg_ratio=%.5f worst_lcb=%.5f win_rate=%.4f%s",
            round_id, mode, rule, chal_entry.miner_hotkey[:12], gate.computed,
            gate.passed, {k: round(v, 5) for k, v in gate.suite_ucb.items()}, gate.suite_n,
            gate.king_score, gate.chal_score, gate.avg_ratio, gate.worst_lcb, gate.win_rate,
            f" reason={gate.reason!r}" if gate.reason else "",
        )
        return apply_bench_gate(result, gate, mode=mode)

    def _run_vbench_gate(self, result: RoundResult, king_entry: TrainedEntry,
                         chal_entry: TrainedEntry, *, seed: int | str, round_id: str,
                         mode: str, rule: str) -> RoundResult:
        """The bench gate on a DEC-CA-0063 settlement (the main validator's
        sweep / report). The CHALLENGER is taken first (its cap check reserves
        the king's allowance when the king still needs a sweep): its fresh
        sweep fixes the pinned data revision, and a cached king swept on
        another revision is swept again. An infra side (``_vbench_outcome``
        "infra": a trainer-recorded "absent" report, an infra-failed sweep —
        or a king's checkpoint fault) sets ``self._bench_gate_infra``, a
        budget-deferred challenger ``self._bench_gate_deferred`` — the cohort
        loop then skips this clearer (never a loss) instead of holding the
        round on it. A challenger's checkpoint fault FAILS the gate (computed,
        ``self._bench_gate_fault``). Entries swept on different revisions are
        uncomputable (the GIFT gate's rule)."""
        from ..eval.bench_gate import (
            checkpoint_fault_bench_gate,
            evaluate_bench_gate,
            uncomputable_bench_gate,
        )
        from ..eval.koth import apply_bench_gate

        p = self.cfg.koth_params()
        thr = {"tolerance": p.bench_gate_tolerance, "min_win_rate": p.bench_gate_min_win_rate}
        chal_rows, chal_six = self._vbench_side(chal_entry, role="challenger",
                                                reserve_for=king_entry,
                                                pair_with=(king_entry, "king"))
        out = self._vbench_outcome(chal_entry)
        king_rows = king_six = None
        if out is None:
            king_rows, king_six = self._vbench_side(king_entry, role="king", capped=False)
            if king_rows is not None and chal_rows is not None and self._vbench_cached(king_entry):
                kr, cr = self._vbench_revisions(king_entry), self._vbench_revisions(chal_entry)
                if kr and cr and any(kr[s] != cr[s] for s in set(kr) & set(cr)):
                    log.info("bench-gate round=%s: cached king rows were swept on data "
                             "revisions %s, the challenger on %s — re-sweeping the king",
                             round_id, kr, cr)
                    self._vbench_drop(king_entry.trained_pointer)
                    king_rows, king_six = self._vbench_side(king_entry, role="king",
                                                            capped=False)
            kout = self._vbench_outcome(king_entry)
            if kout is not None:
                # the king's checkpoint is never the clearer's fault
                out = ("infra", self._vbench_infra(king_entry) or kout[1])
        kind, why = out if out is not None else ("", "")
        if kind == "fault":
            self._bench_gate_fault = why
            gate = checkpoint_fault_bench_gate(why, rule=rule, **thr)
        elif kind == "deferred":
            self._bench_gate_deferred = why
            gate = uncomputable_bench_gate(f"validator bench deferred: {why}", rule=rule, **thr)
        elif kind == "infra":
            self._bench_gate_infra = why
            gate = uncomputable_bench_gate(f"validator bench infra: {why}", rule=rule, **thr)
        elif king_rows is None or chal_rows is None:
            gate = uncomputable_bench_gate(
                "signed bench rows missing for "
                + ("king" if king_rows is None else "challenger"), rule=rule, **thr)
        else:
            gate = evaluate_bench_gate(
                king_rows, chal_rows, king_six, chal_six,
                alpha=p.bootstrap_alpha, B=p.bootstrap_B, seed=seed,
                min_items=p.bench_gate_min_items, rule=rule,
                tolerance=p.bench_gate_tolerance, min_win_rate=p.bench_gate_min_win_rate,
                king_revisions=self._vbench_revisions(king_entry),
                chal_revisions=self._vbench_revisions(chal_entry),
            )
        log.info("bench-gate round=%s mode=%s rule=%s source=validator chal=%s computed=%s "
                 "passed=%s outcome=%s%s", round_id, mode, rule, chal_entry.miner_hotkey[:12],
                 gate.computed, gate.passed, f"{kind}:{why}" if kind else None,
                 f" reason={gate.reason!r}" if gate.reason else "")
        return apply_bench_gate(result, gate, mode=mode)

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
        # The crowned challenger, NOT the manifest's first — a multi-finalist
        # cohort (DEC-CA-0012) carries several, and taking the first would
        # benchmark a checkpoint that did not take the throne.
        crowned = outcome.transition.new_king_hotkey
        new_king = next(
            (e for e in manifest.entries_for_role("challenger")
             if e.miner_hotkey == crowned),
            None,
        )
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
