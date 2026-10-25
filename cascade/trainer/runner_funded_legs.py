"""Running a funded leg on its payer pod, and the pod/burn bookkeeping after it (TrainerRunner mixin)."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

from ..funding.queue import OPERATOR_FAULT_CLASS
from .corpus import DIVERGED_MARKER
from .faults import (
    KEY_LOST_REASON,
    STALL_CLASS,
    _FundedDiverged,
    _FundedKeyLost,
    _FundedTamper,
    _OperatorFault,
    classify_funded_worker_failure,
)
from .progress import progress_binding, report_progress
from .runner_common import (
    ResolvedGenerator,
    _add_seen_hotkeys,
    read_pending_upload,
    release_executor_of,
    router_kw,
    write_pending_upload,
)
from .runner_generation import leg_corpus

log = logging.getLogger("cascade.trainer")


@contextmanager
def funded_leg_scope(runner, queue, contract, *, king_leg=False, gen=None):
    """Keep a leg's queue and budget thread-local across rent and harvest."""
    local = runner.__dict__.setdefault("_leg_local_obj", threading.local())
    missing = object()
    saved = {k: getattr(local, k, missing)
             for k in ("funded_queue", "funded_contract", "funded_is_king", "funded_allocated",
                       "funded_progress", "funded_key_digest")}
    local.funded_queue, local.funded_contract = queue, contract
    local.funded_is_king = bool(king_leg)
    local.funded_allocated = False
    local.funded_key_digest = None
    local.funded_progress = progress_binding(runner, gen) if gen is not None and not king_leg else None
    try:
        yield
    except Exception as exc:
        if king_leg and local.funded_key_digest is not None:
            exc.payer_key_digest = local.funded_key_digest
        raise
    finally:
        for key, value in saved.items():
            if value is missing:
                delattr(local, key)
            else:
                setattr(local, key, value)


class _FundedLegMixin:
    """One funded leg end to end, its bench-pod retention and the burn set."""

    def _run_funded_leg(self, disp, gen: ResolvedGenerator, seeds, block: int,
                        contract, suffix: str, *, warm_start_ref: str | None,
                        contract_block: int | None = None, router_slow: bool = False,
                        era_index: int | None = None, role: str = "challenger",
                        finetune_recipe: str | None = None, bench: bool = True,
                        gpu_choice=None, domain: str | None = None,
                        expert_selection: dict | None = None, funded_queue=None,
                        king_leg: bool = False):
        from .admission import FairSlots, owner_admission
        from .faults import _FundedLegSkip

        with funded_leg_scope(self, funded_queue, contract, king_leg=king_leg, gen=gen):
            self._funded_leg_failures.pop(gen.hotkey, None)
            reused = _reupload_pending(self, gen, seeds, contract, suffix, warm_start_ref)
            if reused is not None:
                report_progress(self, "awaiting_settlement")
                return reused
            with owner_admission(self, gen.hotkey, exempt=king_leg):
                gate = self.__dict__.setdefault("_funded_job_slots", FairSlots())
                # A king has its own reserved admission and cannot queue behind
                # miner jobs that themselves yield to its pending GPU rent.
                deadline = None if king_leg else time.time() + 1800.0
                end = getattr(self.__dict__["_leg_local_obj"], "end_wall", None)
                if not king_leg and end is not None:
                    deadline = float(end) - float(contract.max_train_seconds)
                    deadline -= self.FUNDED_PUBLISH_MARGIN_SECONDS
                    if getattr(contract, "corpus_mode", "") == "shards":
                        deadline -= self._generation_lead_seconds()

                def limit():
                    rnd = self.cfg.round
                    return (int(rnd.funded_field_cap or 0) or max(1, int(rnd.finalist_cap)))

                # King priority is enforced again under the rent lock; this
                # admission cap bounds miners preparing/waiting/training globally.
                from contextlib import nullcontext

                admission = nullcontext(True) if king_leg else gate.slot(limit, deadline=deadline)
                report_progress(self, "waiting_admission")
                with admission as acquired:
                    if not acquired:
                        self._record_funded_failure(
                            gen.hotkey, "shared compute admission is full; requeued before rent",
                            miner_fault=False, error_class="no_capacity", burn=False)
                        raise _FundedLegSkip(gen.hotkey)
                    try:
                        from .faults import _FundedOperatorFallback, _FundedOperatorRent

                        try:
                            entry = _FundedLegMixin._run_funded_leg_scoped(
                                self, disp, gen, seeds, block, contract, suffix,
                                warm_start_ref=warm_start_ref, contract_block=contract_block,
                                router_slow=router_slow, era_index=era_index, role=role,
                                finetune_recipe=finetune_recipe, bench=bench, gpu_choice=gpu_choice,
                                domain=domain, expert_selection=expert_selection)
                        except _FundedOperatorFallback as e:
                            if role not in ("expert", "mkt_finetune") or bench:
                                raise
                            if isinstance(e, _FundedOperatorRent):
                                entry = self._run_operator_rented_leg(disp, gen, seeds, block, contract, suffix,
                                    warm_start_ref=warm_start_ref, contract_block=contract_block,
                                    router_slow=router_slow, role=role, domain=domain,
                                    expert_selection=expert_selection, finetune_recipe=finetune_recipe)
                                report_progress(self, "awaiting_settlement")
                                return entry
                            entry = _FundedLegMixin._specialist_operator_lane(
                                self, disp, gen, seeds, block, contract, suffix,
                                warm_start_ref=warm_start_ref, contract_block=contract_block,
                                era_index=era_index, role=role, finetune_recipe=finetune_recipe,
                                domain=domain, expert_selection=expert_selection)
                        report_progress(self, "awaiting_settlement")
                        return entry
                    except Exception:
                        verdict = self._funded_leg_failures.get(gen.hotkey)
                        local = self.__dict__["_leg_local_obj"]
                        if local.funded_allocated and (verdict is None or not verdict[1]):
                            queue = self._funded_queue()
                            if queue is not None:
                                queue.set_operator_billed(gen.hotkey, True)
                        raise

    def _specialist_operator_lane(self, disp, gen, seeds, block, contract, suffix, *,
                                  warm_start_ref, contract_block, era_index, role,
                                  finetune_recipe, domain, expert_selection):
        """B/C recovery uses the shared, configured operator pool, without rent.

        Keep the funded queue/contract scope and admission slot until harvest
        completes. Never silently run a specialist as a full-model challenger.
        """
        from dataclasses import replace

        from ..funding.store import parse_vault_ref
        from .faults import _FundedOperatorFallback

        if not self._operator_fallback_lanes():
            raise _FundedOperatorFallback(gen.hotkey)
        corpus = leg_corpus(self, gen, seeds, contract,
                            **({"era_index": era_index} if era_index is not None else {}))
        digest = parse_vault_ref(gen.ref)
        def prepare(host):
            staged = self._stage_vault_zip_on(host, digest) if digest else host
            return replace(staged, isolated=True, forward_env=())

        runner = self

        class RecoveryDispatcher:
            def dispatch(self, host, **kw):
                # MarketCast requires local-only even on an operator lane.
                # Harvest under THIS specialist's scheduled contract, not the
                # global primary contract the generic operator hook resolves.
                report_progress(runner, "training")
                receipt = disp.dispatch(host, local_checkpoint=True, **kw)
                report_progress(runner, "harvesting")
                return runner._harvest_funded_checkpoint(host, receipt, contract, seeds, suffix)

        pool = self._rolling_lane_pool()
        report_progress(self, "waiting_operator", "operator_recovery")
        deadline = time.time() + 1800.0
        end = getattr(self.__dict__["_leg_local_obj"], "end_wall", None)
        if end is not None:
            deadline = min(deadline, float(end) - float(contract.max_train_seconds)
                           - self.FUNDED_PUBLISH_MARGIN_SECONDS)
        try:
            return self._dispatch_on_free_lane(
                RecoveryDispatcher(), pool, pool.known_hosts(),
                describe=f"{role} operator recovery {gen.hotkey}",
                prepare=prepare, deadline=deadline,
                gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey, role=role,
                base_seed=seeds.base_seed, block=block, arch_preset=contract.arch_preset,
                warm_start_ref=warm_start_ref, contract_block=contract_block,
                repo_suffix=suffix, **corpus,
                **({"finetune_recipe": finetune_recipe} if finetune_recipe else {}),
                **({"domain": domain, "expert_selection": expert_selection}
                   if role == "expert" else {}))
        except Exception as e:
            if isinstance(e, (_FundedTamper, _FundedDiverged)):
                miner_fault = True
                error_class = "tamper" if isinstance(e, _FundedTamper) else "generator"
            else:
                q = self._funded_queue()
                prior = q.get(gen.hotkey) if q is not None else None
                miner_fault, error_class, _ = classify_funded_worker_failure(
                    getattr(e, "returncode", None), str(e),
                    stalled_before=bool(prior is not None
                                        and prior.last_error_class == STALL_CLASS))
                if error_class == "infra":
                    error_class = OPERATOR_FAULT_CLASS
            self._record_funded_failure(gen.hotkey, str(e), miner_fault=miner_fault,
                                        error_class=error_class, burn=False)
            raise

    def _run_funded_leg_scoped(self, disp, gen, seeds, block, contract, suffix, *,
                               warm_start_ref, contract_block=None, router_slow=False,
                               era_index=None, role="challenger", finetune_recipe=None,
                               bench=True, gpu_choice=None, domain=None,
                               expert_selection=None):
        """One funded challenger leg on its payer's own pod: rent → (stage the
        vault ZIP when the ref is one) → dispatch → harvest → teardown —
        immediately on any failure, or (with the payer-pod bench armed) after
        the completion bench that runs where the checkpoint was trained (see
        :meth:`_keep_funded_pod_for_bench`).

        DEC-CA-0056: a lane-C fine-tune leg passes ``role="finetune"`` with
        its ``finetune_recipe`` (JSON) and ``bench=False`` — the same rent /
        identity / runtime / harvest / ingest-verify path, never benched (the
        pod is torn down when the leg returns).

        ``gpu_choice`` (DEC-CA-0060) is the miner's GPU type for this leg; None
        = the hotkey's in-flight lane-A entry decides (see
        :meth:`_rent_funded_host`).

        DEC-CA-0059 item 8: a lane-B domain-expert leg passes ``role="expert"``
        with its ``domain`` label and ``expert_selection`` (the rows it owns)
        and ``bench=False`` — the same payer-pod path, the worker freezing
        everything but those rows."""
        from ..funding.store import parse_vault_ref

        # A new leg: a verdict left by this hotkey's previous leg must never
        # be read back as this one's (leg_failure falls back to its default).
        self._funded_leg_failures.pop(gen.hotkey, None)
        # A previous attempt of THIS leg (same era seeds, ref, size, init)
        # trained and verified fine but lost its upload: re-upload the local
        # copy instead of renting and retraining.
        reused = _reupload_pending(self, gen, seeds, contract, suffix, warm_start_ref)
        if reused is not None:
            return reused
        # DEC-CA-0053: a shards leg's corpus is made on the CPU tier FIRST —
        # the payer's GPU is rented only once there is something to train on.
        try:
            if getattr(contract, "corpus_mode", "") == "shards":
                report_progress(self, "preparing_data")
            corpus = leg_corpus(self, gen, seeds, contract,
                                **({"era_index": era_index} if era_index is not None else {}))
        except Exception as e:  # noqa: BLE001 — classify like a worker failure
            _record_generation_failure(self, gen, e)
            raise
        report_progress(self, "waiting_gpu", "availability_check")
        host, pod = self._rent_funded_host(
            str(seeds.base_seed), gen,
            **({"gpu_choice": gpu_choice} if gpu_choice is not None else {}))
        keep = False
        try:
            # Identity pin, before a single byte of dispatch: the pod that
            # answers to this name must be the one we rented (platform id)
            # and still running. A payer relaunching "their" pod under the
            # same name is TAMPER — their shot, spent — never infra.
            why = self._funded_pod_identity_mismatch(pod)
            if why == KEY_LOST_REASON:
                raise _FundedKeyLost(f"before dispatch: {why}")
            if why:
                raise _FundedTamper(f"before dispatch: {why}")
            # Runtime attestation, before a single byte of dispatch: lium can
            # serve a host-cached stale image under the degraded tag ref (the
            # env digest gate attests the REQUEST there, not the runtime).
            # A stale pod is a provider fault — infra, requeue, never a burn.
            from .remote import probe_worker_runtime
            why = probe_worker_runtime(host)
            if why:
                raise _OperatorFault(f"funded pod runtime rejected: {why}")
            digest = parse_vault_ref(gen.ref)
            if digest is not None:
                host = self._stage_vault_zip_on(host, digest)
            harvest = self._funded_harvest()
            report_progress(self, "training")
            entry = disp.dispatch(
                host, lane_count=1,
                gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey,
                role=role, base_seed=seeds.base_seed, block=block,
                arch_preset=contract.arch_preset, warm_start_ref=warm_start_ref,
                local_checkpoint=harvest, contract_block=contract_block,
                **corpus,
                **router_kw(router_slow),
                **({"repo_suffix": suffix} if suffix else {}),
                **({"finetune_recipe": finetune_recipe} if finetune_recipe else {}),
                **({"domain": domain, "expert_selection": expert_selection}
                   if role == "expert" else {}),
            )
            if getattr(contract, "base_arch", "") == "cascade_moe" and role != "mkt_finetune":
                from ..shared.manifest import contract_digest

                if (getattr(entry, "init_pointer", None) != (warm_start_ref or "")
                        or getattr(entry, "training_contract_digest", "") != contract_digest(contract)):
                    raise _OperatorFault(
                        "worker receipt lacks or disagrees with the dispatched init/contract; "
                        "rebuild and repin the v2 worker image")
            # …and again when the leg returns: the checkpoint this entry
            # points at must have come from the pod we pinned.
            why = self._funded_pod_identity_mismatch(pod)
            if why == KEY_LOST_REASON:
                raise _FundedKeyLost(f"after training: {why}")
            if why:
                raise _FundedTamper(f"after training: {why}")
            # Pod hygiene before the checkpoint leaves the pod: a process of
            # anyone but ours holding the checkpoint dir is tamper; extra
            # login sessions or a remote IDE server on the pod are logged.
            from .remote import probe_pod_hygiene

            hyg = probe_pod_hygiene(
                host, self._pod_checkpoint_dir(entry, seeds, contract, suffix))
            if hyg.foreign_procs:
                raise _FundedTamper(
                    f"pod hygiene: {len(hyg.foreign_procs)} foreign process(es) hold the "
                    f"checkpoint dir — {hyg.foreign_procs[0][:160]}")
            if hyg.error:
                log.warning("funded leg %s: pod hygiene probe inconclusive (%s)",
                            gen.hotkey, hyg.error)
            elif hyg.sessions or hyg.ide_server:
                log.warning("funded leg %s: pod shows %d interactive ssh session(s) besides "
                            "ours%s", gen.hotkey, hyg.sessions,
                            " and a remote IDE server" if hyg.ide_server else "")
            if harvest:
                report_progress(self, "harvesting")
                # Credential-free pod: the checkpoint is still ON the pod.
                # Pull it over the pinned ssh, ingest-verify the local copy,
                # and upload it under the OPERATOR's identity — the entry's
                # pointer is then a checkpoint whose bytes we inspected.
                try:
                    entry = self._harvest_funded_checkpoint(
                        host, entry, contract, seeds, suffix)
                except _OperatorFault:
                    # Upload lost after its retries: bind the kept copy to
                    # this leg's init too, so only the same leg reuses it.
                    rec = read_pending_upload(self.work_root, seeds.base_seed, gen.hotkey)
                    if rec is not None:
                        rec["warm_start_ref"] = warm_start_ref or ""
                        write_pending_upload(self.work_root, seeds.base_seed,
                                             gen.hotkey, rec)
                    raise
            else:
                # Robot mode: the pod pushed it — fetch and run the ingest
                # guard NOW, before it can reach the manifest; validators and
                # the king pod's bench must never see a checkpoint whose
                # code/config/weights deviate from the contract.
                why = self._funded_checkpoint_mismatch(entry, contract)
                if why:
                    raise _FundedTamper(f"checkpoint: {why}")
            if bench and self._keep_funded_pod_for_bench():
                # The checkpoint benches on THIS pod at completion (payer-
                # billed, the miner's own sidecar eval); the scheduler tears
                # it down when the bench ends (_teardown_kept_funded_pod).
                with self._funded_bench_lock:
                    self._funded_bench_pods[gen.hotkey] = pod
                self._final_role_hosts[
                    ("challenger", contract.arch_preset, gen.hotkey)] = host
                keep = True
            return entry
        except _FundedTamper as e:
            self._record_funded_failure(gen.hotkey, f"pod identity: {e}",
                                        miner_fault=True, error_class="tamper",
                                        burn=False)
            raise
        except _FundedKeyLost as e:
            self._record_funded_failure(
                gen.hotkey, f"payer key lost while the pod was live ({e}) — "
                "the pod cannot be re-identified or stopped from here; "
                "re-fund to supply a fresh key",
                miner_fault=True, error_class="auth", burn=False)
            raise
        except _OperatorFault as e:
            # Our side (or the provider's) failed; the miner's pod and run
            # were fine — requeued, never one of their attempts.
            self._record_funded_failure(gen.hotkey, f"operator-side fault: {e}",
                                        miner_fault=False, error_class=OPERATOR_FAULT_CLASS,
                                        burn=False)
            queue = self._funded_queue()
            if queue is not None:
                queue.set_operator_billed(gen.hotkey, True)
            raise
        except _FundedDiverged as e:
            # The same verdict the worker's own rc=3 gives a non-finite loss
            # (check_loss_finite): the miner's corpus, the miner's shot.
            self._record_funded_failure(gen.hotkey, f"{DIVERGED_MARKER}: {e}",
                                        miner_fault=True, error_class="generator",
                                        burn=False)
            raise
        except Exception as e:  # noqa: BLE001 — classify, record, re-raise for the drop
            from .faults import _transport_failure
            from .gpu_health import health_for

            if _transport_failure(e):
                health_for(self).record(pod.sku, f"{pod.provider}:{pod.instance_id}",
                                        kind="lost", reason=str(e))
            rc = getattr(e, "returncode", None)
            text = str(e)
            if "Host key verification failed" in text:
                # The pinned host key did not match: a different container
                # answered at the pod's address mid-leg.
                self._record_funded_failure(
                    gen.hotkey, f"pod identity: ssh host key changed mid-leg "
                    f"({text[-200:]})", miner_fault=True, error_class="tamper",
                    burn=False)
                raise
            queue = self._funded_queue()
            prior = queue.get(gen.hotkey) if queue is not None else None
            miner_fault, error_class, burn = classify_funded_worker_failure(
                rc, text,
                stalled_before=bool(prior is not None
                                    and prior.last_error_class == STALL_CLASS))
            self._record_funded_failure(gen.hotkey, text, miner_fault=miner_fault,
                                        error_class=error_class, burn=burn)
            if not miner_fault:
                queue = self._funded_queue()
                if queue is not None:
                    queue.set_operator_billed(gen.hotkey, True)
            raise
        finally:
            if not keep:
                self._teardown_funded_pod(pod)

    def _keep_funded_pod_for_bench(self) -> bool:
        """Whether a funded leg's pod outlives its leg for the completion
        bench: ``[telemetry] funded_bench`` with Cascade on and a bench plan
        wired. Without it the challenger is never benched — the sweep runs
        where the checkpoint is."""
        return bool(
            getattr(self.cfg.telemetry, "funded_bench", True)
            and self.cfg.scoring.cascade_enabled
            and self.cascade_bench_plan is not None
        )

    def _teardown_kept_funded_pod(self, hotkey: str) -> None:
        """Tear down the pod kept for ``hotkey``'s bench. When nothing is kept
        in memory the payer's ACCOUNT is swept for this payer's funded pods
        instead: a leg re-attached after a restart and settled from its
        persisted checkpoint never passes the funded-leg path that registers
        the kept pod, and must not leave it running on the miner's bill."""
        with self._funded_bench_lock:
            pod = self._funded_bench_pods.pop(hotkey, None)
        if pod is not None:
            self._teardown_funded_pod(pod)
            return
        self._sweep_payer_account(hotkey, why="leg finished with no kept pod on record")

    def _sweep_payer_account(self, hotkey: str, *, why: str) -> list[str]:
        """Terminate every funded pod of THIS payer on their account (their
        vaulted key; names scoped by ``payer_pod_pattern``) and drop their
        ledger rows. Returns the pod names confirmed gone."""
        vault = self._payer_vault()
        if vault is None or not vault.has(hotkey):
            log.warning("funded teardown: %s — no vaulted key for payer %s, cannot sweep "
                        "their account; any pod there bills them until they stop it",
                        why, hotkey[:12])
            return []
        from ..provision.funded import lium_provider_for_key, reconcile_funded

        factory = self.__dict__.get("_funded_provider_factory") or lium_provider_for_key
        killed = reconcile_funded([], vault, netuid=self.cfg.subnet.netuid,
                                  provider_factory=factory, hotkeys=[hotkey])
        for name in killed:
            release_executor_of(self, name)
        with self._funded_ledger_lock:
            self._save_funded_ledger([x for x in self._load_funded_ledger()
                                      if x.payer_hotkey != hotkey])
        log.log(logging.WARNING if killed else logging.INFO,
                "funded teardown: %s — swept payer %s's account: %d pod(s) terminated%s",
                why, hotkey[:12], len(killed), f" ({', '.join(killed)})" if killed else "")
        return killed

    def _burn_hotkeys(self, challengers: list[ResolvedGenerator]) -> None:
        """Burn the challengers that got their shot: 1 hotkey = 1 submission.

        Called when a leg SETTLES (its entry is in a published manifest) or
        its submission is refused for the miner's own fault — never for a leg
        lost to our infrastructure."""
        if not self.cfg.round.one_submission_per_hotkey or not challengers:
            return
        _add_seen_hotkeys(self._submissions_path(), [c.hotkey for c in challengers])


def _record_generation_failure(runner, gen, e: BaseException) -> None:
    """The queue verdict for a leg whose corpus could not be made on the CPU
    tier. A generator fault (rc 3: :class:`GenerationFault`, ``CorpusError``)
    is the miner's, classed like the worker's own rc 3; everything else —
    no CPU capacity, a VM that never booted, a transfer — is the operator-run
    tier's, requeued without burning an attempt."""
    rc = getattr(e, "returncode", None)
    if rc is None and type(e).__name__ == "CorpusError":
        rc = 3
    if rc != 3:
        runner._record_funded_failure(gen.hotkey, f"generation (operator tier): {e}",
                                      miner_fault=False, error_class=OPERATOR_FAULT_CLASS,
                                      burn=False)
        return
    q = runner._funded_queue()
    prior = q.get(gen.hotkey) if q is not None else None
    miner_fault, error_class, burn = classify_funded_worker_failure(
        rc, str(e), stalled_before=bool(prior is not None
                                        and prior.last_error_class == STALL_CLASS))
    runner._record_funded_failure(gen.hotkey, f"generation: {e}",
                                  miner_fault=miner_fault, error_class=error_class,
                                  burn=burn)


def _reupload_pending(runner, gen, seeds, contract, suffix: str, warm_start_ref):
    """The entry of this leg's earlier, verified harvest whose upload failed
    (see :func:`cascade.trainer.loop.upload_harvested_checkpoint`), re-uploaded
    now; ``None`` when there is none for exactly this leg. A failing re-upload
    records an operator fault (no burn) and raises."""
    rec = read_pending_upload(runner.work_root, seeds.base_seed, gen.hotkey)
    if rec is None:
        return None
    if (rec.get("gen_ref") != gen.ref or rec.get("size") != contract.arch_preset
            or rec.get("suffix") != suffix or rec.get("role") != "challenger"
            or rec.get("warm_start_ref") != (warm_start_ref or "")):
        return None
    from .loop import upload_harvested_checkpoint

    log.warning("funded leg %s: re-uploading the checkpoint an earlier attempt of this "
                "leg harvested and verified (its upload failed) — no retrain",
                gen.hotkey[:12])
    try:
        return upload_harvested_checkpoint(runner, rec, base_seed=seeds.base_seed)
    except _OperatorFault as e:
        runner._record_funded_failure(gen.hotkey, f"operator-side fault: {e}",
                                      miner_fault=False, error_class=OPERATOR_FAULT_CLASS,
                                      burn=False)
        raise
