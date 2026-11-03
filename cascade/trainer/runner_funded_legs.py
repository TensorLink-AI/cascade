"""Running a funded leg on its payer pod, and the pod/burn bookkeeping after it (TrainerRunner mixin)."""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from contextlib import contextmanager

from ..funding.queue import OPERATOR_FAULT_CLASS
from .corpus import DIVERGED_MARKER
from .faults import (
    KEY_LOST_REASON,
    STALL_CLASS,
    CheckpointUnverifiable,
    _FundedDiverged,
    _FundedKeyLost,
    _FundedTamper,
    _OperatorFault,
    classify_funded_worker_failure,
    classify_leg_exception,
)
from .harvest_release import release_after_verified_harvest
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



_KING_KEY_LOCK = threading.Lock()


def king_key_legs(runner) -> dict:
    """hotkey -> number of own-key king legs (DEC-CA-0055 item 6) running now,
    with ``"_renting"`` -> those still renting their pod."""
    return runner.__dict__.setdefault("_king_key_legs", {})


@contextmanager
def king_key_leg(runner, hotkey: str):
    """Register a running own-key king leg for the pod-kill guards."""
    legs = king_key_legs(runner)
    with _KING_KEY_LOCK:
        legs[hotkey] = legs.get(hotkey, 0) + 1
    try:
        yield
    finally:
        with _KING_KEY_LOCK:
            legs[hotkey] = legs.get(hotkey, 0) - 1
            if legs[hotkey] <= 0:
                legs.pop(hotkey, None)


@contextmanager
def king_key_renting(runner):
    """An own-key king leg is renting: challengers yield the marketplace."""
    legs = king_key_legs(runner)
    with _KING_KEY_LOCK:
        legs["_renting"] = legs.get("_renting", 0) + 1
    try:
        yield
    finally:
        with _KING_KEY_LOCK:
            legs["_renting"] = legs.get("_renting", 0) - 1
            if legs["_renting"] <= 0:
                legs.pop("_renting", None)

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
                    # The SAME wall admission budgeted (the measured SKU wall,
                    # capped by max_train_seconds): the cap here opened a dead
                    # zone where admission targeted a boundary this deadline had
                    # already closed, so no-corpus legs idled ~1.2 h per cycle
                    # (2026-10-31). The rent's own fit check still refuses a
                    # type that cannot land; the progress watchdog a slow pod.
                    deadline = float(end) - float(self._leg_wall_seconds(None))
                    deadline -= self.FUNDED_PUBLISH_MARGIN_SECONDS
                    # A corpus already generated (earlier attempt / era of the
                    # same seed epoch) needs only its GPU wall, not the CPU lead.
                    if (getattr(contract, "corpus_mode", "") == "shards"
                            and not self._leg_corpus_ready(gen, seeds)):
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
                        late = deadline is not None and time.time() >= deadline
                        self._record_funded_failure(
                            gen.hotkey,
                            ("past the latest safe start for this boundary; requeued before rent"
                             if late else "shared compute admission is full; requeued before rent"),
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
                    except Exception as exc:
                        from .harvest_guard import keeps_pod, local_disk_error

                        verdict = self._funded_leg_failures.get(gen.hotkey)
                        local = self.__dict__["_leg_local_obj"]
                        # A pod KEPT after our disk failed is re-attached on
                        # the miner's own key: never flagged operator-billed.
                        ours_kept = keeps_pod(exc) or local_disk_error(exc) is not None
                        if (local.funded_allocated and not ours_kept
                                and (verdict is None or not verdict[1])):
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
            deadline = min(deadline, float(end) - float(self._leg_wall_seconds(None))
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
                miner_fault, error_class, _ = classify_leg_exception(
                    self, gen, seeds, e, rc=getattr(e, "returncode", None), text=str(e),
                    stalled_before=bool(prior is not None
                                        and prior.last_error_class == STALL_CLASS),
                    shards_leg=bool(corpus.get("corpus_files")))
                if error_class == "infra":
                    error_class = OPERATOR_FAULT_CLASS
            from .faults import record_no_lanes_verdict
            from .runner_rolling import record_lane_breaker_victim

            if not (record_lane_breaker_victim(self, gen.hotkey, e)
                    or record_no_lanes_verdict(self, gen.hotkey, e)):
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
        # Money safety II: the money gates as a side-effect-free "would it
        # rent?" verdict BEFORE the corpus (a CPU VM is operator money): a
        # leg that cannot rent never generates. King legs are exempt (a held
        # king is the era loop's RentsHalted, never a skip it counts).
        if not getattr(self.__dict__.get("_leg_local_obj"), "funded_is_king", False):
            _pre_corpus_money_gates(self, gen, gpu_choice, round_id=str(seeds.base_seed))
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
        # A harvest that failed on OUR side (disk/IO) with the run complete:
        # the pod is NEVER torn down for it (harvest_guard) — the requeued leg
        # re-attaches to its ledgered allocation and harvests again.
        hold_pod = False
        trained = False
        released: list[bool] = []
        # the harvest of a payer pod: the miner controls it (review F7)
        _loc = self.__dict__.setdefault("_leg_local_obj", threading.local())
        _loc.harvest_pod_payer, _loc.leg_trained = True, False

        def release_pod() -> None:
            # Owner 2026-11-01: the harvested copy is verified (on the box, or
            # on R2 with the r2 transport) and nothing benches on this pod —
            # stop its bill before the upload / Hippius push.
            if not released:
                # verified harvest: this pod PRODUCED the result (never waste)
                _spend_result(self, gen.hotkey, pod)
                self._teardown_funded_pod(pod)
                released.append(True)

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
            disp = _with_leg_watchdog(self, disp, gen, pod, contract, host)
            # (a harvest inside dispatch marks the leg trained — harvest_guard)
            entry = _dispatch_watched(
                disp, host, lane_count=1,
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
            trained = True
            if getattr(contract, "base_arch", "") in ("cascade_moe", "cascade_pyramid") and role != "mkt_finetune":
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
                # pointer is then a checkpoint whose bytes we inspected. A pod
                # nothing benches on is released once the copy is verified,
                # before the upload (cascade.trainer.harvest_release).
                early = (contextlib.nullcontext() if _funded_pod_benched_here(self, bench)
                         else release_after_verified_harvest(self, release_pod))
                try:
                    with early:
                        entry = self._harvest_funded_checkpoint(
                            host, entry, contract, seeds, suffix)
                except _OperatorFault:
                    # Upload lost after its retries: bind the kept copy to
                    # this leg's init too, so only the same leg reuses it.
                    rec = read_pending_upload(self.work_root, seeds.base_seed, gen.hotkey,
                                              gen_ref=gen.ref, size=contract.arch_preset, suffix=suffix)
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
            if not released and bench and self._keep_funded_pod_for_bench():
                # The checkpoint benches on THIS pod at completion (payer-
                # billed, the miner's own sidecar eval); the scheduler tears
                # it down when the bench ends (_teardown_kept_funded_pod).
                with self._funded_bench_lock:
                    self._funded_bench_pods[gen.hotkey] = pod
                self._final_role_hosts[
                    ("challenger", contract.arch_preset, gen.hotkey)] = host
                keep = True
            _spend_result(self, gen.hotkey, pod)
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
        except CheckpointUnverifiable as e:
            # Terminal but UNBURNED (review F7): evidence kept, owner decides.
            self._record_funded_failure(gen.hotkey, f"checkpoint: {e}", miner_fault=True,
                                        error_class=CheckpointUnverifiable.error_class,
                                        burn=False)
            raise
        except _OperatorFault as e:
            # Our side (or the provider's) failed; the miner's pod and run
            # were fine — requeued, never one of their attempts.
            from .harvest_guard import handle_our_side_fault, keeps_pod

            self._record_funded_failure(gen.hotkey, f"operator-side fault: {e}",
                                        miner_fault=False, error_class=OPERATOR_FAULT_CLASS,
                                        burn=False)
            # box health (may halt every rent) + the bounded keep (F6)
            if handle_our_side_fault(self, gen, e, pod_id=str(getattr(pod, "instance_id", "") or ""), payer=True,
                                     trained=trained or _harvested_in_dispatch(self),
                                     released=bool(released)):
                # The run is complete on the MINER's pod: the retry re-attaches
                # there (their key, no new rent) — not an operator make-good.
                hold_pod = True
                raise
            if keeps_pod(e):
                raise                  # our harvest failed: never an operator make-good
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
            miner_fault, error_class, burn = classify_leg_exception(
                self, gen, seeds, e, rc=rc, text=text,
                stalled_before=bool(prior is not None
                                    and prior.last_error_class == STALL_CLASS),
                shards_leg=bool(corpus.get("corpus_files")))
            self._record_funded_failure(gen.hotkey, text, miner_fault=miner_fault,
                                        error_class=error_class, burn=burn)
            from .harvest_guard import handle_our_side_fault, keeps_pod, local_disk_error

            if handle_our_side_fault(self, gen, e, pod_id=str(getattr(pod, "instance_id", "") or ""), payer=True,
                                     trained=trained or _harvested_in_dispatch(self),
                                     released=bool(released)):
                # OUR disk failed with the run on the miner's pod: keep it for
                # the re-attach on their key (never an operator make-good).
                hold_pod = True
                raise
            if keeps_pod(e) or local_disk_error(e) is not None:
                raise                  # our side: never an operator make-good
            if not miner_fault:
                queue = self._funded_queue()
                if queue is not None:
                    queue.set_operator_billed(gen.hotkey, True)
            raise
        finally:
            self.__dict__["_leg_local_obj"].harvest_pod_payer = False
            if not keep and not released and not hold_pod:
                verdict = self._funded_leg_failures.get(gen.hotkey)
                if verdict is not None and verdict[1]:
                    # a genuine miner fault: waste of THEIR making (no make-good)
                    from .miner_spend_guard import MINER, spend_end

                    spend_end(self, gen.hotkey, str(getattr(pod, "instance_id", "") or ""),
                              outcome="waste", reason=str(verdict[0])[:120], kind=MINER)
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
        ledger rows. Returns the pod names confirmed gone.

        Never while ANOTHER leg of this hotkey runs an own-key king leg
        (DEC-CA-0055 item 6): the sweep is name-scoped, not leg-scoped, and
        would kill that live king pod (2026-10-31 review)."""
        mine = 1 if getattr(self.__dict__.get("_leg_local_obj"), "funded_is_king", False) else 0
        if king_key_legs(self).get(hotkey, 0) > mine:
            log.warning("funded teardown: %s — payer %s has a king leg running on "
                        "this account; account sweep skipped (its pod is live)",
                        why, hotkey[:12])
            return []
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
        from .miner_spend_guard import spend_end

        for name in killed:
            release_executor_of(self, name)
            spend_end(self, hotkey, name, outcome="waste",       # the account sweep (S1)
                      reason="account sweep without a harvested result")
        with self._funded_ledger_lock:
            self._save_funded_ledger([x for x in self._load_funded_ledger()
                                      if x.payer_hotkey != hotkey])
        log.log(logging.WARNING if killed else logging.INFO,
                "funded teardown: %s — swept payer %s's account: %d pod(s) terminated%s",
                why, hotkey[:12], len(killed), f" ({', '.join(killed)})" if killed else "")
        return killed

    def _burn_hotkeys(self, challengers: list[ResolvedGenerator], block: int = -1) -> None:
        """Burn the challengers that got their shot: 1 hotkey = 1 submission
        per round. ``block`` places the burn in its round (the leg's start
        block at settlement, the admission block on a refusal); the hotkey is
        free again in the next round.

        Called when a leg SETTLES (its entry is in a published manifest) or
        its submission is refused for the miner's own fault — never for a leg
        lost to our infrastructure."""
        if not self.cfg.round.one_submission_per_hotkey or not challengers:
            return
        _add_seen_hotkeys(self._submissions_path(), [c.hotkey for c in challengers], block)


def _leg_allocated(runner, round_id: str, hotkey: str) -> bool:
    """A ledgered pod of THIS leg exists (payer or operator-rented): a
    restart / retry re-attaches to it — no pre-rent gate may strand it."""
    from ..provision.funded import funded_pod_name

    try:
        base = funded_pod_name(str(round_id), hotkey, runner.cfg.subnet.netuid)
        names = {base + "-0", runner._operator_leg_pod_prefix(str(round_id), hotkey) + "-0"}
        return any(p.instance_id in names for p in runner._load_funded_ledger())
    except Exception:  # noqa: BLE001 — unknown: assume allocated (never strand a live leg)
        return True


def _pre_corpus_money_gates(runner, gen, gpu_choice=None, *, round_id: str = "") -> None:
    """Before a payer leg's corpus is generated: the would-rent verdict
    (:func:`~cascade.trainer.money_guard.would_rent_refusal`; the operator
    gates too when the entry is operator-billed) and the miner-key money
    gates (waste cap, the miner's own budget) — so a leg that cannot rent
    pays for no CPU VM. Raises :class:`_FundedLegSkip` (nothing rented,
    unburned) with the verdict recorded."""
    from .faults import _FundedLegSkip
    from .money_guard import would_rent_refusal

    if _leg_allocated(runner, round_id, gen.hotkey):
        return                       # a re-attach: the rent path adopts, never re-gates
    q = runner._funded_queue()
    e = q.get(gen.hotkey) if q is not None else None
    operator = bool(e is not None and getattr(e, "operator_billed", False))
    why, cls = would_rent_refusal(runner, gen, operator=operator)
    if why:
        log.warning("funded leg %s: %s — no corpus generated", gen.hotkey[:12], why)
        runner._record_funded_failure(gen.hotkey, why, miner_fault=False, error_class=cls,
                                      burn=False)
        raise _FundedLegSkip(gen.hotkey)
    if not operator and e is not None:
        from .miner_spend_guard import budget_check, waste_cap_check

        waste_cap_check(runner, gen)                     # pauses at the cap (no rent)
        budget_check(runner, gen, gpu_choice if gpu_choice is not None
                     else getattr(e, "gpu_choice", None))


def _with_leg_watchdog(runner, disp, gen, pod, contract, host=None):
    """``disp`` with the leg's progress watchdog (and periodic teardown-key
    check) attached — a detached :class:`RemoteDispatcher` only; any other
    dispatcher (attached form, test doubles) is returned unchanged. ``host``
    (the leg's RemoteHost) names the pod's IP in the verdict."""
    from dataclasses import replace

    from .cost_guards import check_payer_key, leg_watchdog
    from .remote import RemoteDispatcher

    if not isinstance(disp, RemoteDispatcher) or not disp.detached:
        return disp
    payer = str(getattr(pod, "payer_hotkey", "") or "")

    def key_check():
        check_payer_key(runner, hotkey=gen.hotkey, pod_name=str(pod.instance_id), pod=pod)
        from .miner_spend_guard import spend_touch

        spend_touch(runner, gen.hotkey, str(pod.instance_id))    # seen alive (review S1)

    watch = leg_watchdog(runner, gen=gen, pod=pod, contract=contract,
                         key_check=key_check if payer else None,
                         host=str(getattr(host, "host", "") or ""))
    return disp if watch is None else replace(disp, on_progress=watch)


def _spend_result(runner, hotkey: str, pod) -> None:
    from .miner_spend_guard import RESULT, spend_end

    spend_end(runner, hotkey, str(getattr(pod, "instance_id", "") or ""), outcome=RESULT,
              reason="harvested and verified")


def _harvested_in_dispatch(runner) -> bool:
    """A harvest ran inside this thread's dispatch (the isolated-host hook):
    the run had finished training (harvest_guard marks it)."""
    return bool(getattr(runner.__dict__.get("_leg_local_obj"), "leg_trained", False))


def _dispatch_watched(disp, host, **kw):
    """``disp.dispatch`` with a watchdog abort turned into an operator fault:
    the miner's pod and generator were fine, OUR wall estimate (or the
    host) was not. The host is quarantined for a day: a host that cannot hold
    the leg's rate — or never reached step 1 — is a hardware/fabric fault, and
    re-renting it (2026-11-01: 93.120.231.186, 0.2 steps/s on two legs in a
    row) only repeats the loss."""
    from .cost_guards import FIRST_STEP_ABORT
    from .host_preflight import quarantine_host_ip
    from .remote import LegAborted

    try:
        return disp.dispatch(host, **kw)
    except LegAborted as e:
        # Never a burn: OUR watchdog aborted it (an operator-side fault). A
        # hang before step 1 on a leg whose corpus was made OFF the pod (the
        # CPU tier) can only be the host — NCCL init, the driver — and its
        # IP is quarantined. With on-pod generation the time before step 1
        # includes the miner's generator, so it is not proof of a bad host:
        # still unburned, the host is not blamed (money safety II).
        first_step = FIRST_STEP_ABORT in str(e)
        if not first_step or kw.get("corpus_files"):
            quarantine_host_ip(str(getattr(host, "host", "") or ""), str(e)[-300:])
        else:
            log.warning("leg %s: first-step deadline with on-pod generation — not proof of a "
                        "bad host; requeued unburned, host not quarantined",
                        str(kw.get("hotkey", ""))[:12])
        cls = HostHangFault if first_step and kw.get("corpus_files") else _OperatorFault
        raise cls(str(e)) from e


class HostHangFault(_OperatorFault):
    """The worker never reached step 1 on a pre-generated corpus: the HOST
    hung (NCCL init / a collective / the driver) — an operator-side,
    unburned fault; the host is quarantined."""


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
        # no CPU capacity / a VM that never booted: transient (review F4);
        # OUR disk failing the corpus write still halts every rent (R5)
        from ..funding.queue import OPERATOR_TRANSIENT_CLASS
        from .box_health import note_disk_error

        note_disk_error(getattr(runner, "work_root", None), e, f"corpus of {gen.hotkey[:12]}")

        runner._record_funded_failure(gen.hotkey, f"generation (operator tier): {e}",
                                      miner_fault=False, error_class=OPERATOR_TRANSIENT_CLASS,
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
    rec = read_pending_upload(runner.work_root, seeds.base_seed, gen.hotkey,
                              gen_ref=gen.ref, size=contract.arch_preset, suffix=suffix,
                              role="challenger", warm_start_ref=warm_start_ref or "")
    if rec is None:
        return None
    if (rec.get("gen_ref") != gen.ref or rec.get("size") != contract.arch_preset
            or rec.get("suffix") != suffix or rec.get("role") != "challenger"
            or rec.get("warm_start_ref") != (warm_start_ref or "")):
        return None
    from ..shared.manifest import contract_digest
    from .loop import upload_harvested_checkpoint

    recorded = rec.get("contract_digest")
    if (recorded and recorded != contract_digest(contract)
            or not recorded and getattr(contract, "base_arch", "") == "cascade_pyramid"):
        return None
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
    except _FundedTamper as e:
        # r2 transport: the finite/code check runs at the push, so a retried
        # push can still reach the verdict the ssh harvest reaches up front.
        runner._record_funded_failure(gen.hotkey, f"checkpoint: {e}", miner_fault=True,
                                      error_class="tamper", burn=False)
        raise
    except _FundedDiverged as e:
        runner._record_funded_failure(gen.hotkey, f"{DIVERGED_MARKER}: {e}",
                                      miner_fault=True, error_class="generator", burn=False)
        raise


def _funded_pod_benched_here(runner, bench: bool) -> bool:
    """Whether THIS leg's pod may still be needed after its harvest: the
    trainer benches the checkpoint on it (payer-pod bench armed, the leg
    benchable, the era not validator-benched — DEC-CA-0063). Otherwise the
    pod is released once the harvested copy is verified. A king own-key
    leg's pod is torn down the moment its leg returns either way. No bench
    source on this thread (a caller that does not set it) keeps today's rule."""
    if not (bench and runner._keep_funded_pod_for_bench()):
        return False
    local = runner.__dict__.get("_leg_local_obj")
    if getattr(local, "funded_is_king", False):
        return False
    return not bool(getattr(local, "validator_bench", False))
