"""The CPU generation tier on the trainer side (TrainerRunner mixin,
DEC-CA-0053): a ``shards`` leg's corpus is made BEFORE its GPU pod is rented,
so neither the payer nor the operator pays for an idle GPU while it builds."""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..generation.stage import StageResult

log = logging.getLogger("cascade.trainer")


def leg_corpus(runner, gen, seeds, contract, *, era_index: int | None = None) -> dict:
    """``{"corpus_files": ...}`` dispatch kwargs for a ``shards`` leg, ``{}``
    for every other leg (its dispatch stays byte-identical)."""
    if getattr(contract, "corpus_mode", "") != "shards":
        return {}
    return {"corpus_files": runner._leg_corpus(gen, seeds, contract, era_index=era_index)}


class _GenerationMixin:
    """Runs :class:`cascade.generation.stage.GenerationStage` for a leg."""

    # Boot + bootstrap of the CPU VM, and the digest pass + upload after the
    # window: the tier's time beyond ``[generation] window_seconds``.
    GENERATION_OVERHEAD_SECONDS = 2 * 3600.0
    # A sold-out CPU market: re-probe this long (inside the overhead above)
    # before the leg requeues as an operator-side infra fault.
    GENERATION_CAPACITY_WAIT_SECONDS = 1800.0
    GENERATION_VM_LEDGER = "generation_vms.json"

    def _v2_baseline_ready(self, gen, block):
        if not self.cfg.v2.active(block):
            return True
        from .launch_preflight import ready

        try:
            if ready(self.cfg, gen.ref, self.work_root):
                self._v2_launch_preflight_status = {"state": "ready", "ref": gen.ref}
                return True
        except (OSError, ValueError, KeyError) as exc:
            log.error("v2 launch preflight refused: %s", exc)
            reason = str(exc)
        else:
            reason = "initial v2 king source has no matching full verification"
        self._v2_launch_preflight_status = {"state": "blocked", "ref": gen.ref, "reason": reason}
        log.error("v2 launch blocked before rental: qualify a compliant initial king with "
                  "cascade-trainer --preflight-v2-baseline REF --work-root PATH")
        return False

    def _generation_lead_seconds(self) -> float:
        """How long a leg spends on the CPU tier before its GPU wall starts:
        0 unless a ``shards`` contract can apply — the live one, or the v2 one
        once ``[v2] from_block`` is typed (the eras just before go-live then
        admit a little more conservatively, which is harmless)."""
        tr = self.cfg.training
        v2 = tr.v2_contract
        shards = (tr.corpus_mode == "shards"
                  or (v2 is not None and v2.corpus_mode == "shards"
                      and int(self.cfg.v2.from_block or 0) > 0))
        if not (shards and self.cfg.generation.enabled):
            return 0.0
        return float(self.cfg.generation.window_seconds) + self.GENERATION_OVERHEAD_SECONDS

    def _corpus_store(self):
        store = self.__dict__.get("_corpus_store_obj")
        if store is None:
            from ..shared.hippius import S3Config, S3Store

            bucket = self.cfg.generation.bucket or self.cfg.storage.manifest_bucket
            store = S3Store(S3Config.from_storage(self.cfg.storage, bucket=bucket))
            self.__dict__["_corpus_store_obj"] = store
        return store

    def _generation_stage(self):
        # The king and miners can enter this on their first tick together;
        # they must share one stage/admission gate from the first VM onward.
        lock = self.__dict__.setdefault("_generation_stage_lock", threading.RLock())
        with lock:
            return _GenerationMixin._generation_stage_locked(self)

    def _generation_stage_locked(self):
        stage = self.__dict__.get("_generation_stage_obj")
        if stage is None:
            from ..generation.stage import GenerationStage
            from ..provision.cpu import cpu_providers

            key = self._generation_ssh_key()
            pub = Path(key + ".pub").expanduser()
            if not pub.is_file():
                raise RuntimeError(f"generation tier: no public key at {pub} for the CPU VMs")
            stage = GenerationStage(
                cfg=self.cfg, store=self._corpus_store(),
                providers=cpu_providers(self.cfg.generation),
                ssh_key_path=key, ssh_pubkey=pub.read_text(encoding="utf-8").strip(),
                vm_image=self.cfg.generation.vm_image,
                ledger_path=self._generation_vm_ledger(),
                capacity_wait_s=self.GENERATION_CAPACITY_WAIT_SECONDS)
            self.__dict__["_generation_stage_obj"] = stage
        return stage

    def _generation_vm_ledger(self) -> Path:
        return Path(self.work_root) / self.GENERATION_VM_LEDGER

    def _reap_generation_vms(self) -> list[str]:
        """Terminate CPU VMs a crash or a failed terminate left billing (the VM
        ledger, plus a ``cgen-`` listing where the provider has one). Run at
        startup and periodically; best-effort, never raises."""
        ledger = self._generation_vm_ledger()
        if not (self.cfg.generation.enabled or ledger.is_file()):
            return []
        try:
            from ..generation.stage import reap_generation_vms

            stage = self.__dict__.get("_generation_stage_obj")
            if stage is not None:
                providers = list(stage.providers)
            else:
                from ..provision.cpu import cpu_providers

                providers = cpu_providers(self.cfg.generation)
            return reap_generation_vms(ledger, providers)
        except Exception as e:  # noqa: BLE001 — a reaper must never sink the loop
            log.warning("generation VM reap failed (retried next pass): %s", e)
            return []

    def _generation_ssh_key(self) -> str:
        """The operator's pod ssh key (the ``[[host]]`` profile the trainer
        already rents GPU pods with)."""
        for h in self._load_remote_hosts() or []:
            if getattr(h, "key_path", None):
                return str(Path(h.key_path).expanduser())
        return str(Path("~/.ssh/id_ed25519").expanduser())

    def _load_remote_hosts(self) -> list:
        if getattr(self, "remote_hosts_path", None) is not None:
            from .remote import load_hosts

            try:
                return list(load_hosts(self.remote_hosts_path))
            except Exception:  # noqa: BLE001 — fall back to the static list
                pass
        return list(self.remote_hosts or [])

    def _leg_corpus(self, gen, seeds, contract, *, era_index: int | None = None
                    ) -> dict[str, str] | None:
        """The pod files of a ``shards`` leg (``corpus.json`` + download links),
        or ``None`` when the leg's contract streams as before. Raises
        :class:`~cascade.generation.stage.GenerationFault` (the miner's) or
        :class:`~cascade.generation.stage.GenerationInfraError` (requeue)."""
        if getattr(contract, "corpus_mode", "") != "shards":
            return None
        # One corpus per leg: a rent that falls back (operator lane, operator
        # rent) reuses it instead of repeating the generation window. Cache
        # the corpus metadata, not the expiring download links: retries can
        # happen days later within the same seed epoch.
        cache = self.__dict__.setdefault("_leg_corpus_cache", {})
        key = (gen.hotkey, str(seeds.base_seed), gen.ref,
               self.cfg.generator.min_channels, self.cfg.generator.max_channels)
        # One build per key even when two threads (lane A, lane-B domains) ask
        # at once: the second waits and takes the first's corpus.
        locks = self.__dict__.setdefault("_leg_corpus_locks", {})
        with locks.setdefault(key, threading.Lock()):
            if key not in cache:
                result = self._build_leg_corpus(gen, seeds, era_index)
                if len(cache) >= 256:                 # old legs' entries
                    cache.pop(next(iter(cache)))
                cache[key] = result
            result = cache[key]
        return self._generation_stage().pod_files(result)

    def _build_leg_corpus(self, gen, seeds, era_index: int | None) -> StageResult:
        from ..generation.stage import GenerationInfraError
        from ..shared.hippius import fetch_from_hub

        # The ref is part of every name: a hotkey's second ref in one era gets
        # its own fetch (never the first ref's cached tree), its own dataset
        # prefix and its own VM.
        ref_tag = hashlib.sha256(str(gen.ref).encode()).hexdigest()[:12]
        round_id = f"{seeds.base_seed}-{ref_tag}"
        repo = Path(self.work_root) / "_generation" / f"{gen.hotkey[:16]}-{round_id}"
        try:
            fetch_from_hub(gen.ref, repo, self.hub())
        except Exception as e:  # noqa: BLE001
            # The same split the GPU path makes: a private/missing artifact is
            # the miner's (CorpusError → rc 3), a storage blip is ours.
            from ..shared.hippius import StorageError
            from .corpus import CorpusError

            status = getattr(getattr(e, "response", None), "status_code", None)
            if isinstance(e, StorageError) and status in (401, 403, 404):
                raise CorpusError(f"generator_artifact_unreachable: HTTP {status}") from e
            raise GenerationInfraError(f"generator fetch failed: {e}") from e
        try:
            stage = self._generation_stage()
            return stage.generate(
                round_id=round_id, hotkey=gen.hotkey, uid=gen.uid, era=era_index,
                gen_ref=gen.ref, repo_dir=repo, seed=int(seeds.generation_seed),
                **({"priority": True} if getattr(self.__dict__.get("_leg_local_obj"),
                                                 "funded_is_king", False) else {}))
        finally:
            shutil.rmtree(repo, ignore_errors=True)
