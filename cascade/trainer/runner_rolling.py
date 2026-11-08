"""Rolling intake + era king pod/bucket adapters (TrainerRunner mixin)."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading

from ..shared.bench_report import pointer_only_init_scores
from ..shared.manifest import BenchScores
from .contract import RoundSeeds
from .faults import (
    _FundedOperatorFallback,
    _FundedOperatorRent,
    _FundedTamper,
    _quarantine_lane_host,
    king_key_failed,
)
from .lane_pool import (
    _FinalLanePool,
    host_cannot_bench,
    king_pod_name_prefix,
)
from .runner_common import (
    era_contracts,
    router_kw,
    settlement_contracts,
)
from .runner_generation import leg_corpus

log = logging.getLogger("cascade.trainer")


# How long a rollover teardown waits for a benchmark sweep on the old king pod.
KING_RETIRE_WAIT_S = 3 * 3600


def _validator_benches(cfg, era) -> bool:
    """DEC-CA-0063: ``era``'s lane-A bench rows come from the main validator
    (winners only, its own GPU) — the trainer runs no challenger / king sweep
    and no payer-pod verification for it."""
    from ..shared.era import bench_source_from, validator_bench_era

    if not bench_source_from(cfg):
        return False
    return validator_bench_era(cfg, int(era.start_block))


def _bench_fev_required(cfg, era) -> bool:
    """DEC-CA-0061: whether ``era``'s settlements judge under the four-suite
    rule, so its published (operator) benches must carry FEV (the init's
    too: under that rule the DEC-CA-0055 king check runs the gate's rule on
    the king's and the init's FEV task rows)."""
    from ..shared.era import BENCH_RULE_V2, bench_gate_rule_for_era, bench_gate_v2_configured

    if not bench_gate_v2_configured(cfg):
        return False                # the gate runs at no block: the sweep has no FEV
    return bench_gate_rule_for_era(cfg, int(era.start_block)) == BENCH_RULE_V2


# DEC-CA-0063 eras: the main validator benches the init itself (king check,
# ``_vbench_side(init_e, role="init")``); all it reads from the trainer's
# report is the "init" entry's POINTER — a pointer-only entry (sentinel
# scores, see cascade.shared.bench_report.POINTER_ONLY_INIT_SCORE) that every
# reader defers on. ``_rolling_publish_bench`` keeps it alone in its report.


# How long the pod-side random-init build may take (CPU construction + save of
# the full-size init; minutes in practice).
INIT_BUILD_TIMEOUT_S = 1800

# Run ON THE POD with the pod's own interpreter and chain.toml: the
# random-init checkpoint a v2 leg with ``training_seed`` starts from
# (moe_trainer.build_random_init — the king leg's own construction), refused
# unless the pod resolves exactly the trainer's contract (digest). argv:
# <chain.toml or ""> <contract block> <training seed> <arch preset>
# <contract digest> <out dir>.
_INIT_BUILD_PY = (
    "import sys\n"
    "from pathlib import Path\n"
    "from cascade.shared.config import load_chain_config\n"
    "from cascade.shared.manifest import contract_digest\n"
    "toml, block, seed, preset, digest, out = sys.argv[1:7]\n"
    "c = load_chain_config(toml or None).throne_contracts_at(int(block))[0]\n"
    "got = contract_digest(c)\n"
    "if c.arch_preset != preset or got != digest:\n"
    "    sys.exit('init build refused: pod contract %s/%s != trainer %s/%s'"
    " % (c.arch_preset, got, preset, digest))\n"
    "from cascade.trainer.moe_trainer import build_random_init\n"
    "build_random_init(c, training_seed=int(seed), out_dir=Path(out))\n"
)


def _init_rebuildable(contract) -> bool:
    """Whether ``contract``'s random init has a ``build_random_init``."""
    return str(getattr(contract, "base_arch", "") or "") in ("cascade_moe", "cascade_pyramid")


def build_init_remote_command(host, out_dir: str, *, contract_block: int, training_seed: int,
                              arch_preset: str, digest: str) -> str:
    """The pod-side shell string that builds the epoch's random-init
    checkpoint into ``out_dir`` (wiped first). CPU only (``CUDA_VISIBLE_DEVICES``
    blanked): the GPU is never touched. Pure — safe to unit test.

    The trainer box never builds the init itself (7 GB box: a full-size model
    construction is an OOM risk) — it only harvests the written files."""
    import shlex

    argv = [host.remote_python, "-c", _INIT_BUILD_PY, str(host.chain_toml or ""),
            str(int(contract_block)), str(int(training_seed)), str(arch_preset),
            str(digest), out_dir]
    out = shlex.quote(out_dir)
    return (f"cd {shlex.quote(host.workdir)} && rm -rf {out} && "
            f"CUDA_VISIBLE_DEVICES= {shlex.join(argv)}")


class _KingKeyFailed(Exception):
    """The king's own payer key cannot fund its leg (DEC-CA-0055 item 6) —
    the operator trains it instead and the manifest says so. ``slash`` is
    False when the cause is ours (make-good, capacity hybrid, an operator-
    billed entry): the operator still trains the leg, but the manifest says
    ``operator_fault`` and validators do not slash the king."""

    def __init__(self, msg: str, *, slash: bool = True, key_digest: str | None = None) -> None:
        super().__init__(msg)
        self.slash = slash
        self.key_digest = key_digest


def _king_failed_keys(runner, update=None) -> dict[str, str]:
    """DEC-CA-0055: the king failed-key digests (``RollingState.
    king_failed_keys``, persisted with the era state so a trainer restart
    cannot let the SAME key lift the slash); ``update(d)`` mutates and saves
    them under the scheduler's lock. No scheduler yet (unit harness) ⇒ an
    in-memory dict on the runner."""
    sched = runner.__dict__.get("_rolling_sched")
    if sched is None:
        d = runner.__dict__.setdefault("_king_failed_key", {})
        if update is not None:
            update(d)
        return d
    with sched._lock:
        d = sched.state.king_failed_keys
        if update is not None:
            update(d)
            sched._save()
        return dict(d)


class _RollingMixin:
    """Rolling intake + era king (DEC-CA-0043) pod/bucket adapters."""

    # ── rolling intake + era king (DEC-CA-0043) — pod/bucket adapters ──────

    @property
    def _leg_local(self) -> threading.local:
        loc = self.__dict__.get("_leg_local_obj")
        if loc is None:
            loc = threading.local()
            self.__dict__["_leg_local_obj"] = loc
        return loc

    def _rolling(self):
        sched = self.__dict__.get("_rolling_sched")
        if sched is None:
            from .rolling import RollingScheduler

            sched = RollingScheduler(self)
            self.__dict__["_rolling_sched"] = sched
        return sched

    def _dispatcher(self):
        from pathlib import Path

        from .remote import RemoteDispatcher

        if not self.trainer_spec:
            raise RuntimeError("remote training requires trainer_spec")
        if self.cfg.round.ops_continuation_interval_seconds:
            if not self.detached_dispatch:
                raise ValueError("continuation requires detached dispatch")
            from .continuation import PROTOCOL
            configured = [self.cfg.training, self.cfg.training.v2_contract]
            supported = any(c is not None and getattr(c, 'continuation_protocol', '') == PROTOCOL for c in configured)
            supported = supported or self.cfg.marketcast.continuation_protocol == PROTOCOL
            if not supported:
                raise ValueError('continuation has not been activated in a released training contract')
        return RemoteDispatcher(
            trainer_spec=self.trainer_spec, timeout_seconds=self.remote_timeout_seconds,
            extra_forward_env=self._pod_extra_forward_env(),
            isolated_forward_env=self._pod_isolated_forward_env(),
            harvest=self._harvest_leg,
            evidence_dir=(str(Path(self.work_root) / "_failed_legs")
                          if getattr(self, "work_root", None) is not None else None),
            **({"cancel_event": self._leg_local.king_cancel} if getattr(self._leg_local, "king_cancel", None) is not None else {}),
            **({"continuation_root": str(self.work_root / "_continuations"),
                "continuation_contract": self.cfg.training,
                "continuation_marketcast_protocol": self.cfg.marketcast.continuation_protocol,
                "continuation_interval": self.cfg.round.ops_continuation_interval_seconds,
                "continuation_max_bytes": self.cfg.round.ops_continuation_max_bytes}
               if self.cfg.round.ops_continuation_interval_seconds else {}),
            **self._leg_pins_hook(),
            prefetch_shards=bool(getattr(self.cfg.round, "funded_prefetch_shards", True)),
            **self._dispatch_mode(),
        )

    def _leg_pins_hook(self) -> dict:
        """DEC-CA-0067: the per-leg image hook, only while a repin is armed."""
        from ..shared.era import image_after_armed

        if not image_after_armed(self.cfg):
            return {}
        from .leg_pins import dispatch_host_hook

        return {"host_hook": dispatch_host_hook(self)}

    def _rolling_seeds(self, era):
        return RoundSeeds.derive(int(era.base_seed), self.cfg.training)

    def _era_contract_block(self, era) -> int | None:
        """The ``--contract-block`` a rolling leg carries: the era's start block
        when a contract switch is SCHEDULED, else ``None`` (no flag). A worker
        image that predates the flag rejects it at launch (unrecognized
        argument, exit 2), and the flag changes nothing without a schedule, so
        it is emitted only once the owner has scheduled a switch — by which
        point the flag-aware image must already be pinned."""
        t = self.cfg.training
        if str(getattr(t, "budget_denomination_after", "") or "") and int(
                getattr(t, "budget_denomination_after_block", 0) or 0) > 0:
            return int(era.start_block)
        if int(getattr(t, "v2_from_block", 0) or 0) > 0:
            # DEC-CA-0050: the v2 switch is a scheduled contract change too
            return int(era.start_block)
        return None

    def _era_router_slow(self, era) -> bool:
        """DEC-CA-0050 slow router: every lane-A leg of ``era`` runs the
        router at ``router_lr_scale`` / ``route_bias_rate_scale`` once the
        lane-B routing gate has OPENED — and only when it opened no later
        than the era's SEED block, the earliest block any leg of the era can
        start (its pre-train window), so the king and every challenger of
        the era agree. Off when the era's contract leaves both at 1.0."""
        st = getattr(getattr(self, "lane_b_driver", None), "st", None)
        opened = int(getattr(st, "gate_opened_block", 0) or 0)
        if not opened or opened > int(era.seed_block):
            return False
        c = self.cfg.training.at_block(int(era.start_block))
        return (float(getattr(c, "router_lr_scale", 1.0)) != 1.0
                or float(getattr(c, "route_bias_rate_scale", 1.0)) != 1.0)

    def _rolling_warm_start_ref(self, era, contract) -> str | None:
        return (era.warm_start_ckpt
                if era.warm_start_ckpt and era.warm_start_size == contract.arch_preset
                else None)

    def _rolling_train_challenger(self, gen, era, block: int, *, end_wall: float):
        """One funded challenger leg outside any round: payer pod (or the
        operator-lane fallback), the era's seeds and init, its own target
        boundary as the rent-wait deadline."""
        contract = self.cfg.throne_contracts_at(int(era.start_block))[0]
        from .leg_pins import leg_contract_scope

        with leg_contract_scope(self, contract):
            return self._rolling_train_challenger_in(gen, era, block, contract,
                                                     end_wall=end_wall)

    def _rolling_train_challenger_in(self, gen, era, block: int, contract, *, end_wall: float):
        from ..funding.store import parse_vault_ref

        seeds = self._rolling_seeds(era)
        suffix = f"-u{gen.uid}"
        ws_ref = self._rolling_warm_start_ref(era, contract)
        slow = router_kw(self._era_router_slow(era))
        disp = self._dispatcher()
        self._leg_local.end_wall = float(end_wall)
        # Opt-in for _FundedOperatorRent: an operator-billed leg with no lane
        # on file rents an operator pod instead of dying.
        self._leg_local.operator_rent_ok = True
        # DEC-CA-0063 era: the trainer never benches on the payer pod, so the
        # pod is released as soon as its harvest is verified (harvest_release).
        self._leg_local.validator_bench = _validator_benches(self.cfg, era)
        try:
            try:
                entry = self._run_funded_leg(disp, gen, seeds, int(block), contract, suffix,
                                             warm_start_ref=ws_ref,
                                             contract_block=self._era_contract_block(era),
                                             **slow)
            except _FundedOperatorRent:
                from .progress import report_progress

                report_progress(self, "waiting_operator", "operator_recovery")
                log.warning("rolling: %s — operator-billed (payer key expired / make-good) "
                            "with no operator lane on file: renting an OPERATOR pod",
                            gen.hotkey[:12])
                entry = self._run_operator_rented_leg(
                    disp, gen, seeds, int(block), contract, suffix, warm_start_ref=ws_ref,
                    contract_block=self._era_contract_block(era), **slow)
            except _FundedOperatorFallback:
                from .progress import report_progress

                report_progress(self, "waiting_operator", "operator_recovery")
                log.warning("rolling: %s — no marketplace capacity; running on an "
                            "OPERATOR lane (operator-billed)", gen.hotkey[:12])
                pool = self._rolling_lane_pool()
                used: list = []
                digest = parse_vault_ref(gen.ref)
                def prepare(host):
                    staged = self._stage_vault_zip_on(host, digest) if digest else host
                    report_progress(self, "training")
                    return staged
                try:
                    entry = self._dispatch_on_free_lane(
                        disp, pool, pool.known_hosts(),
                        describe=f"rolling challenger {gen.hotkey}", used_host=used,
                        prepare=prepare, deadline=self._operator_lane_deadline_fn(),
                        gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey,
                        role="challenger", base_seed=seeds.base_seed, block=int(block),
                        arch_preset=contract.arch_preset, warm_start_ref=ws_ref,
                        contract_block=self._era_contract_block(era), **slow,
                        repo_suffix=suffix,
                        **leg_corpus(self, gen, seeds, contract),   # cached
                    )
                except Exception as e:
                    from .faults import record_no_lanes_verdict

                    # No live lane / a broken lane: the LANES failed, not the
                    # leg — no_capacity, no backoff; the next try rents.
                    if not record_lane_breaker_victim(self, gen.hotkey, e):
                        record_no_lanes_verdict(self, gen.hotkey, e)
                    raise
                if used:
                    self._final_role_hosts[("challenger", contract.arch_preset, gen.hotkey)] = used[-1]
        finally:
            self._leg_local.end_wall = None
            self._leg_local.operator_rent_ok = False
            self._leg_local.validator_bench = False
        self._persist_completed_leg(entry, round_id=era.base_seed, contract=contract,
                                    role="challenger", hotkey=gen.hotkey, suffix=suffix,
                                    warm_start_ckpt=ws_ref or "")
        return entry

    def _rolling_final_hosts(self) -> list:
        """The operator FINAL lanes on file NOW (``stage`` ``"any"`` /
        ``"final"``; hosts.toml re-read every call when the trainer runs from
        a file, so a lane hand-rented after startup is seen), profile-only and
        init-only entries excluded."""
        from .remote import is_profile_only

        lanes = [h for h in _hosts_on_file(self)
                 if getattr(h, "stage", "any") in ("any", "final") and not is_profile_only(h)]
        return [_with_deployed_chain_toml(self, h) for h in lanes]

    def _rolling_init_lanes(self) -> list:
        """The ``stage = "init"`` lanes on file NOW: operator-rented hosts that
        exist ONLY to build the seed epoch's random-init checkpoint
        (:meth:`_rolling_publish_init_pointer`). Never a leg target
        (:meth:`_rolling_final_hosts` and the capacity count skip the stage),
        so no miner code runs there and the entry needs no ``isolated`` flag;
        operator-billed by definition — the operator put it on file. Blocked /
        quarantined hosts are left out (logged)."""
        from .host_preflight import host_blocked_reason
        from .remote import is_profile_only

        out = []
        for h in _hosts_on_file(self):
            if getattr(h, "stage", "any") != "init" or is_profile_only(h):
                continue
            try:
                why = host_blocked_reason(self, str(getattr(h, "host", "") or ""))
            except Exception as e:  # noqa: BLE001 — an unreadable blocklist blocks nothing
                log.warning("rolling: init lane %s block check failed (ignored): %s",
                            getattr(h, "name", h), e)
                why = ""
            if why:
                log.warning("rolling: init lane %s left out (%s)", getattr(h, "name", h), why)
                continue
            out.append(_with_deployed_chain_toml(self, h))
        return out

    def _rolling_lane_pool(self):
        """ONE lane pool for every rolling fallback leg (created lazily): lane
        occupancy, the join gate's rejections and the dead-pod drain persist
        across legs. A pool per leg (the first cut) handed every leg the whole
        fleet as free — two concurrent fallbacks could double-book a lane —
        and forgot each leg's dead pod for the next one. Lanes join through
        :meth:`_operator_lane_gate` (ssh, worker runtime, bench floor) from a
        live hosts.toml refresh."""
        pool = self.__dict__.get("_rolling_lane_pool_obj")
        threshold = int(getattr(getattr(getattr(self, "cfg", None), "round", None),
                                "operator_lane_breaker_failures", 0) or 0)
        if pool is None:
            from .host_preflight import host_blocked_reason

            # The initial lanes skip the join gate, but never the blocklist /
            # quarantine: a blocked one is left to the refresh, whose gate
            # refuses it (and re-gates it later, so an unblocked IP returns).
            initial = []
            for h in self._rolling_final_hosts():
                why = host_blocked_reason(self, str(getattr(h, "host", "") or ""))
                if why:
                    log.warning("rolling lane pool: lane %s left out (%s)",
                                getattr(h, "name", h), why)
                    continue
                initial.append(h)
            # sync_from_file: an edited hosts.toml entry reaches dispatch and a
            # removed lane leaves once idle (never a stale host object).
            pool = _FinalLanePool(initial, self._rolling_final_hosts,
                                  gate_fn=self._operator_lane_gate, sync_from_file=True,
                                  breaker_threshold=threshold)
            self.__dict__["_rolling_lane_pool_obj"] = pool
        else:
            pool.breaker_threshold = threshold      # follows the live config
        return pool

    def _operator_funds_genesis_king(self, gen) -> bool:
        """Owner rule 2026-10-31: the genesis king (its ref listed in
        ``[generator] king_min_channels_refs``) is ALWAYS trained on the
        operator's account, never billed to that miner's key. A king that
        re-commits a new ref is no longer the genesis king: DEC-CA-0055 applies."""
        refs = getattr(getattr(self.cfg, "generator", None), "king_min_channels_refs", ()) or ()
        return gen is not None and str(getattr(gen, "ref", "") or "") in set(refs)

    def _king_self_funds(self, era, gen=None) -> bool:
        """DEC-CA-0055 item 6: in a v2 era the king's leg rents on the king's
        own payer-vault key (funded pods armed and a vault configured) —
        except the genesis king, which the operator always pays for."""
        from ..shared.era import v2_era_active

        if self._operator_funds_genesis_king(gen):
            return False
        rnd = self.cfg.round
        return bool(v2_era_active(self.cfg, int(era.start_block))
                    and bool(getattr(self.cfg.v2, "king_ladder", False))
                    and str(getattr(rnd, "funded_pods", "") or "") == "rent"
                    and getattr(rnd, "payer_vault_dir", ""))

    # repo/pod-dir suffix of a king leg run on the king's own key (never
    # collides with the operator king leg or the hotkey's challenger leg)
    KING_KEY_SUFFIX = "-kingkey"

    def _rolling_train_king_own_key(self, gen, era, block: int, *, end_wall: float,
                                    contract, seeds, ws_ref):
        """The king leg on the KING's deposited key — the funded challenger
        path (payer pod, credential-free harvest, ingest verify). Raises
        :class:`_KingKeyFailed` when the key is the problem (absent/expired in
        the vault, rejected, out of funds, revoked mid-leg, tamper on the
        king's own pod): the caller falls back to the operator's key. Any
        other failure re-raises (the king leg retries next tick on the key).
        The payer pod is torn down at once: the king's operator-trusted bench
        and the era's payer verifications run on an operator pod."""
        vault = self._payer_vault()
        observed = (vault.get_shared(gen.hotkey) if vault is not None else None) or ''
        observed_digest = hashlib.sha256(observed.encode()).hexdigest() if observed else ''
        if not observed:
            raise _KingKeyFailed("no key in the payer vault (absent, revoked or expired)",
                                 key_digest=observed_digest)
        vault.refresh(gen.hotkey)
        disp = self._dispatcher()
        self._leg_local.end_wall = float(end_wall)
        from ..funding.gpu_choice import GpuChoice
        from .runner_funded_legs import king_key_leg

        # registered for the pod-kill guards (sweep / account sweep / admission)
        with king_key_leg(self, gen.hotkey):
            try:
                # The king leg rides no fund request: the default GPU path, never
                # a choice made for this hotkey's challenger leg (DEC-CA-0060).
                entry = self._run_funded_leg(
                    disp, gen, seeds, int(block), contract, self.KING_KEY_SUFFIX,
                    king_leg=True,
                    warm_start_ref=ws_ref, contract_block=self._era_contract_block(era),
                    era_index=era.index, gpu_choice=GpuChoice(),
                    **router_kw(self._era_router_slow(era)))
            except _FundedOperatorFallback as e:
                # make-good / capacity hybrid / operator-billed: OUR side — no slash
                raise _KingKeyFailed(f"funded path handed the leg to the operator ({e})",
                                     slash=False) from e
            except _FundedTamper as e:
                raise _KingKeyFailed(f"tamper on the king's own pod: {e}",
                                     key_digest=getattr(e, "payer_key_digest", observed_digest)) from e
            except Exception as e:
                verdict = self._funded_leg_failures.get(gen.hotkey)
                if king_key_failed(verdict, e):
                    raise _KingKeyFailed(f"{(verdict or (str(e),))[0]}",
                                         key_digest=getattr(e, "payer_key_digest", observed_digest)) from e
                raise
            finally:
                self._leg_local.end_wall = None
                # never bills the king past its leg: the bench runs on our pod
                with contextlib.suppress(Exception):
                    self._teardown_kept_funded_pod(gen.hotkey)
                self._final_role_hosts.pop(("challenger", contract.arch_preset, gen.hotkey), None)
        return entry

    def _rolling_king_funding(self, era, gen) -> str:
        """Who paid for ``era``'s persisted king leg ("king" / "operator" /
        "" unknown), read from its completed-leg record."""
        import json

        contract = self.cfg.throne_contracts_at(int(era.start_block))[0]
        path = self._completed_leg_path(era.base_seed, contract.arch_preset, "king",
                                        gen.hotkey, "")
        try:
            return str(json.loads(path.read_text(encoding="utf-8")).get("funded_by", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _note_king_key_failed(self, hotkey: str, *, key_digest: str | None = None) -> None:
        """Remember the digest of the vault key that failed ("" when none),
        so only a ROTATED key can clear the slash (:meth:`_rolling_king_key_ok`)."""
        import hashlib

        get_vault = getattr(self, "_payer_vault", None)
        vault = get_vault() if callable(get_vault) else None
        key = (vault.get_shared(hotkey) if vault is not None else None) or ""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest() if key else ""
        _king_failed_keys(self, lambda d: d.__setitem__(hotkey, digest if key_digest is None else key_digest))

    def _rolling_king_key_ok(self, hotkey: str) -> bool:
        """DEC-CA-0055: has a slashed king fixed its key? True when the vault
        holds a key OTHER than the one that failed (rotated via
        ``/v1/credentials``; the failed digest survives restarts in the era
        state) and Lium accepts it (an authenticated read). The
        king trains once per seed epoch, so this — not a retrain — is what
        lifts the slash for the epoch's later eras.

        Auth only, NOT funds: no Lium balance read is used anywhere in this
        codebase (intake deliberately skips a balance pre-check), so an
        authenticated but unfunded rotated key passes here; the king's next
        own-key leg then fails as out of funds and re-opens a lapse — under
        the escalating slash that is a SECOND lapse (rest of the reign).
        Money safety II: a plain Lium key must ALSO hold the king leg's
        projected cost (``payer_balance_refusal`` via ``GET /users/me``); an
        unreadable balance never blocks."""
        import hashlib

        import requests

        from ..provision.core import LIUM_API_BASE

        vault = self._payer_vault()
        key = (vault.get_shared(hotkey) if vault is not None else None) or ""
        if not key:
            return False
        failed = _king_failed_keys(self).get(hotkey)
        if failed is not None and hashlib.sha256(key.encode("utf-8")).hexdigest() == failed:
            return False
        from ..funding.providers import PREFIX, decode_credentials

        if key.startswith(PREFIX):
            from ..provision.funded import provider_for_key

            try:
                for credential in decode_credentials(key):
                    provider_for_key(credential.provider, credential.api_key).list_tagged("cascade-")
            except Exception:
                return False
            _king_failed_keys(self, lambda d: d.pop(hotkey, None))
            return True
        try:
            r = requests.get(f"{LIUM_API_BASE}/templates", headers={"X-API-KEY": key},
                             timeout=20)
        except requests.RequestException:
            return False
        if not r.ok:
            return False
        # Money safety II (audit K): authenticated is not FUNDED — a rotated
        # key whose balance cannot pay the king's leg fails here (one alert),
        # before a rent that would only open a second lapse. An unreadable
        # balance never blocks (the rent's own 402 still classifies).
        from .money_guard import alert, payer_balance_refusal

        try:
            rnd = self.cfg.round
            rate = max(float(getattr(rnd, "payer_projected_rate_per_hour", 0.0) or 0.0),
                       float(getattr(rnd, "funded_max_price_per_hour", 0.0) or 0.0))
            hours = float(self._leg_wall_seconds()) / 3600.0
            poor = payer_balance_refusal(self, key, hours=hours, rate=rate) if hours else ""
        except Exception as e:  # noqa: BLE001 — unknown: never blocks (the rent's 402 does)
            log.debug("king key balance check unavailable (%s)", e)
            poor = ""
        if poor:
            alert(self, "king_key_unfunded", hotkey[:16],
                  f"king {hotkey[:12]}: rotated key authenticates but is unfunded — {poor}",
                  severity="critical")
            return False
        _king_failed_keys(self, lambda d: d.pop(hotkey, None))
        return True

    def _rolling_refresh_king_key(self, hotkey: str) -> bool:
        """Re-stamp the king's payer-vault TTL while it is king (it trains
        once per seed epoch, which outlives the vault TTL). False when there
        is no vault or no key."""
        if str(getattr(self.cfg.round, "funded_pods", "") or "") != "rent":
            return False
        vault = self._payer_vault()
        if vault is None:
            return False
        return bool(vault.refresh(hotkey))

    def _rolling_train_king(self, gen, era, block: int, *, end_wall: float):
        if not self.cfg.round.ops_king_backup_after_seconds:
            return self._rolling_train_king_single(gen, era, block, end_wall=end_wall)
        if not self.detached_dispatch:
            raise ValueError("king replicas require detached dispatch")
        import dataclasses

        from ..shared.manifest import TrainedEntry, contract_digest
        from .king_hedge import KingHedge
        contract = self.cfg.throne_contracts_at(int(era.start_block))[0]
        seeds = self._rolling_seeds(era)
        ws = self._rolling_warm_start_ref(era, contract)
        binding = {"gen_ref": gen.ref, "hotkey": gen.hotkey, "contract": contract_digest(contract),
                   "init": ws, "seed": seeds.base_seed}
        from .continuation import canonical
        job = hashlib.sha256(canonical(binding)).hexdigest()
        hedge = KingHedge(self.work_root / str(era.base_seed) / "king_replicas" / (job + ".json"), binding)
        primary_prefix = king_pod_name_prefix(self.cfg.subnet.netuid, str(era.base_seed)) + '-replica-' + job[:12]
        backup_prefix = king_pod_name_prefix(self.cfg.subnet.netuid, str(era.base_seed)) + '-backup-' + job[:12]
        hosts = {}
        def primary(cancel):
            self._leg_local.king_cancel = cancel
            self._leg_local.king_hedge = hedge
            self._leg_local.king_primary_prefix = primary_prefix
            self._leg_local.funded_is_king = True
            try:
                entry, funded_by, host = self._rolling_train_king_single(
                    gen, era, block, end_wall=end_wall, persist=False)
                hosts['primary'] = host
                return {'entry': dataclasses.asdict(entry), 'funded_by': funded_by}
            finally:
                if cancel.is_set():
                    # Only this replica's allocation, never the era's winner.
                    prefix = primary_prefix + '-0'
                    for pod in self._load_funded_ledger():
                        if not pod.payer_hotkey and pod.instance_id == prefix:
                            self._teardown_operator_pod(pod)
        def backup(cancel):
            from .leg_pins import leg_contract_scope

            with leg_contract_scope(self, contract):
                return _backup(cancel)

        def _backup(cancel):
            from .runner_generation import leg_corpus
            self._leg_local.king_cancel = cancel
            self._leg_local.funded_is_king = True
            self._leg_local.end_wall = float(end_wall)
            prefix = backup_prefix
            corpus = leg_corpus(self, gen, seeds, contract, era_index=era.index)
            host, pod_id = self._rent_operator_leg_host(str(era.base_seed), gen, name_prefix=prefix)
            try:
                host = self._stage_king_vault(host, gen)
                entry = self._dispatcher().dispatch(host, lane_count=1, gen_ref=gen.ref,
                    uid=gen.uid, hotkey=gen.hotkey, role='king', base_seed=seeds.base_seed,
                    block=int(block), arch_preset=contract.arch_preset, warm_start_ref=ws,
                    contract_block=self._era_contract_block(era), repo_suffix='-backup',
                    **corpus, **router_kw(self._era_router_slow(era)))
                self._refuse_diverged_king(entry, contract)
                if self._king_self_funds(era, gen):
                    from .king_hedge import king_account_status
                    try:
                        status, key_digest = self._with_deadline(
                            lambda: king_account_status(self, gen.hotkey), min(10, self.chain_read_timeout_s))
                    except Exception:
                        status = 'unknown'
                    if status == 'failed':
                        self._note_king_key_failed(gen.hotkey, key_digest=key_digest)
                        hedge.note_key_failure()
                hosts['backup'] = host
                return {'entry': dataclasses.asdict(entry), 'funded_by': 'operator_fault'}
            finally:
                if cancel.is_set():
                    for pod in self._load_funded_ledger():
                        if pod.instance_id == pod_id and not pod.payer_hotkey:
                            self._teardown_operator_pod(pod)
        def cleanup(replica):
            alias = (backup_prefix if replica == 'backup' else primary_prefix) + '-0'
            for pod in self._load_funded_ledger():
                if not pod.payer_hotkey and pod.instance_id == alias:
                    self._teardown_operator_pod(pod)
            with self._funded_king_lock:
                if replica == 'primary' and self._funded_king_host is hosts.get('primary'):
                    self._funded_king_host = None
        result = hedge.run(primary, backup, delay=self.cfg.round.ops_king_backup_after_seconds,
                           deadline=end_wall, cleanup=cleanup)
        entry = TrainedEntry(**result['entry'])
        host = hosts.get(result['replica'])
        if host is not None:
            self._final_role_hosts[("king", contract.arch_preset, gen.hotkey)] = host
            self.__dict__.setdefault("_rolling_king_hosts", {})[era.index] = host
        self._king_rent_done = True
        self._persist_completed_leg(entry, round_id=era.base_seed, contract=contract,
            role="king", hotkey=gen.hotkey, warm_start_ckpt=ws or "", funded_by=result['funded_by'])
        return entry

    def _rolling_train_king_single(self, gen, era, block: int, *, end_wall: float, persist=True):
        """See :meth:`_rolling_train_king_single_in` (DEC-CA-0067: the leg's
        contract names its era pins on this thread)."""
        from .leg_pins import leg_contract_scope

        with leg_contract_scope(self, self.cfg.throne_contracts_at(int(era.start_block))[0]):
            return self._rolling_train_king_single_in(gen, era, block, end_wall=end_wall,
                                                      persist=persist)

    def _rolling_train_king_single_in(self, gen, era, block: int, *, end_wall: float,
                                      persist=True):
        """The era king's leg on the operator's JIT pod (kept for the era: the
        top-N re-bench targets it). ``end_wall`` — the era's last settlement —
        is the leg's own target (thread-local), so the king rent waits out a
        sold-out marketplace up to ITS latest safe start like a challenger
        instead of giving up at the first empty listing.

        In a v2 era (DEC-CA-0055 item 6) the leg first rents on the KING's
        own deposited key; only when that key fails does it fall back to the
        operator's pod. The completed-leg record carries who paid
        (``funded_by``), which the scheduler stamps on the manifest."""
        contract = self.cfg.throne_contracts_at(int(era.start_block))[0]
        seeds = self._rolling_seeds(era)
        ws_ref = self._rolling_warm_start_ref(era, contract)
        # the genesis king is operator-paid by owner rule: our side, never a slash
        funded_by = "operator_fault" if self._operator_funds_genesis_king(gen) else ""
        if self._king_self_funds(era, gen):
            try:
                entry = self._rolling_train_king_own_key(
                    gen, era, block, end_wall=end_wall, contract=contract, seeds=seeds,
                    ws_ref=ws_ref)
            except _KingKeyFailed as e:
                funded_by = "operator" if e.slash else "operator_fault"
                if e.slash:
                    self._note_king_key_failed(gen.hotkey, **({"key_digest": e.key_digest} if e.key_digest is not None else {}))
                    hedge = getattr(self._leg_local, "king_hedge", None)
                    if hedge is not None:
                        hedge.note_key_failure()
                log.warning("rolling: era %d king %s — %s (%s): the OPERATOR trains the "
                            "king leg (funded_by=%s)", era.index, gen.hotkey[:12],
                            "its own key failed" if e.slash else "operator-side fallback",
                            e, funded_by)
            else:
                self._refuse_diverged_king(entry, contract)
                if not persist:
                    return entry, "king", None
                self._persist_completed_leg(entry, round_id=era.base_seed, contract=contract,
                                            role="king", hotkey=gen.hotkey,
                                            warm_start_ckpt=ws_ref or "", funded_by="king")
                log.info("rolling: era %d king leg ran on the king's own key", era.index)
                return entry
        disp = self._dispatcher()
        # DEC-CA-0053: the king's corpus too is made before its GPU rent.
        # The inherited king can arrive here before it has any payer key.
        # CPU admission must still recognize it as king work, just as the
        # own-key funded scope and replica threads do.
        local = self._leg_local
        previous = getattr(local, "funded_is_king", False)
        local.funded_is_king = True
        try:
            corpus = leg_corpus(self, gen, seeds, contract, era_index=era.index)
        finally:
            local.funded_is_king = previous
        # Slot claim + rent as one unit against a bench-only king rent
        # (_rolling_king_bench_host): the single cached king host must never
        # hand one era's pod to the other era's leg.
        with self._king_rent_serial:
            with self._funded_king_lock:
                if self._rolling_king_host_era != era.index:
                    self._funded_king_host = None
                    self._rolling_king_host_era = era.index
            # A new king rent: challengers yield to it again, and a previous
            # attempt's give-up no longer cancels their capacity waits.
            self._king_rent_done = False
            self._funded_wait_abort.clear()
            self._leg_local.end_wall = float(end_wall)
            try:
                host = self._stage_king_vault(self._rent_king_host(str(era.base_seed)), gen)
            finally:
                self._king_rent_done = True
                self._leg_local.end_wall = None
        entry = disp.dispatch(
            host, lane_count=1, gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey,
            role="king", base_seed=seeds.base_seed, block=int(block),
            arch_preset=contract.arch_preset, warm_start_ref=ws_ref,
            contract_block=self._era_contract_block(era), **corpus,
            **router_kw(self._era_router_slow(era)),
        )
        self._refuse_diverged_king(entry, contract)
        if not persist:
            return entry, funded_by, host
        self._final_role_hosts[("king", contract.arch_preset, gen.hotkey)] = host
        self.__dict__.setdefault("_rolling_king_hosts", {})[era.index] = host
        self._persist_completed_leg(entry, round_id=era.base_seed, contract=contract,
                                    role="king", hotkey=gen.hotkey,
                                    warm_start_ckpt=ws_ref or "", funded_by=funded_by)
        return entry

    def _rolling_carry_king_pod(self, old, new) -> None:
        """DEC-CA-0055 seed epoch: ``new`` shares ``old``'s base seed — and so
        its king pod name prefix — and carries its king leg. Hand the operator
        king pod over (mapping + slot) instead of tearing it down by prefix,
        which would also take any pod rented for ``new``."""
        hosts = self.__dict__.setdefault("_rolling_king_hosts", {})
        host = hosts.pop(old.index, None)
        if host is not None:
            hosts.setdefault(new.index, host)
        with self._funded_king_lock:
            if self._rolling_king_host_era == old.index:
                self._rolling_king_host_era = new.index
        log.info("rolling: era %d king pod carried into era %d (same seed epoch)",
                 old.index, new.index)

    def _rolling_note_king_host(self, era) -> None:
        """After a dethrone adoption the operator's era pod benches for the
        NEW king's hotkey."""
        contract = era_contracts(self.cfg, era)[0]
        host = self.__dict__.get("_rolling_king_hosts", {}).get(era.index)
        if host is not None and era.king_hotkey:
            self._final_role_hosts[("king", contract.arch_preset, era.king_hotkey)] = host

    def _rolling_king_pod_can_bench(self, era) -> bool:
        host = self.__dict__.get("_rolling_king_hosts", {}).get(era.index)
        return host is None or not host_cannot_bench(self, host)

    def _rolling_retire_king_pod(self, era) -> None:
        """Tear down an era's operator king pod (era over, or its pre-trained
        king superseded) — but never under a running benchmark sweep, which
        would lose a payer verification mid-run. A busy pod is retired by a
        waiter thread once its sweep releases the pod lock (capped)."""
        from .bench_hook import host_bench_lock, host_busy

        host = self.__dict__.get("_rolling_king_hosts", {}).get(era.index)
        if host is not None and host_busy(host):
            log.info("rolling: era %d king pod busy benchmarking — teardown deferred until "
                     "the sweep ends", era.index)

            def _later() -> None:
                lock = host_bench_lock(host)
                got = lock.acquire(timeout=KING_RETIRE_WAIT_S)
                if got:
                    lock.release()
                else:
                    log.warning("rolling: era %d king pod still busy after %ds — tearing "
                                "down anyway", era.index, KING_RETIRE_WAIT_S)
                self._rolling_retire_king_pod_now(era)

            threading.Thread(target=_later, name=f"retire-king-{era.index}", daemon=True).start()
            return
        self._rolling_retire_king_pod_now(era)

    def _rolling_retire_king_pod_now(self, era) -> None:
        prefix = king_pod_name_prefix(self.cfg.subnet.netuid, str(era.base_seed))
        for pod in self._load_funded_ledger():
            if not pod.payer_hotkey and str(pod.instance_id).startswith(prefix + "-"):
                self._teardown_operator_pod(pod)
        host = self.__dict__.get("_rolling_king_hosts", {}).pop(era.index, None)
        if host is not None:
            # Forget role mappings onto the released pod: a later verification
            # must "wait" for the next king pod, never SSH into a dead one.
            for k in [k for k, h in self._final_role_hosts.items() if h is host]:
                self._final_role_hosts.pop(k, None)
        with self._funded_king_lock:
            if self._rolling_king_host_era == era.index:
                self._funded_king_host = None

    def _rolling_rotate_king_pod(self, era, reason: str) -> None:
        """A booted king pod whose leg keeps failing (or went unreachable) is
        a lemon HOST: quarantine its IP (Lium relists the same machine under
        fresh executor ids) and tear the pod down, so the next king-leg
        attempt re-rents elsewhere instead of looping into the same wall."""
        from ..funding.faults import classify_rent_failure

        if classify_rent_failure(reason) in {"auth", "rate_limited"}:
            log.warning("king pod rotation skipped: account/API failure is not a hardware fault")
            return
        host = self.__dict__.get("_rolling_king_hosts", {}).get(era.index)
        if host is None:
            with self._funded_king_lock:
                if self._rolling_king_host_era == era.index:
                    host = self._funded_king_host
        if host is None:
            return
        from .gpu_health import health_for
        instance_id = getattr(host, 'allocation_id', '')
        pods = [p for p in self._load_funded_ledger() if not p.payer_hotkey]
        if not instance_id:
            matching = [p for p in pods if self._is_king_pod_of(p.instance_id, str(era.base_seed))]
            # A legacy host with two matching replicas is ambiguous. Never
            # turn one failed worker into teardown of every healthy replica.
            instance_id = matching[0].instance_id if len(matching) == 1 else ''
        health_for(self).record(getattr(host, 'sku', ''),
            getattr(host, "allocation_incident", "") or f"king:{instance_id or host.host}", kind='lost', reason=reason)
        _quarantine_lane_host(host, f"king pod {getattr(host, 'name', host)}: {reason}")
        for pod in pods:
            if instance_id and pod.instance_id == instance_id:
                self._teardown_operator_pod(pod)
        hosts = self.__dict__.get('_rolling_king_hosts', {})
        for index, value in list(hosts.items()):
            if value is host:
                hosts.pop(index, None)
        for key, value in list(self._final_role_hosts.items()):
            if value is host:
                self._final_role_hosts.pop(key, None)
        with self._funded_king_lock:
            if self._funded_king_host is host:
                self._funded_king_host = None
        if not instance_id:
            log.warning('failed king host has no unambiguous allocation; retained for operator reconciliation')

    @property
    def _king_rent_serial(self) -> threading.Lock:
        """Serialises the king-host slot claim + rent of a king leg and of a
        bench-only king rent (created lazily; ``dict.setdefault`` is atomic)."""
        return self.__dict__.setdefault("_king_rent_serial_obj", threading.Lock())

    # How long a bench-only king rent waits for a king leg's rent in progress.
    KING_BENCH_RENT_WAIT_S = 60.0

    def _rolling_king_bench_host(self, entry, era, contract):
        """A pod to bench ``era``'s king on when no host is mapped to it — a
        trainer restart (the mapping is in-process only) or a pod released
        before the bench landed. Only while the bench gate is configured
        (DEC-CA-0054: every challenger pairs with the king's rows; without the
        gate the king bench is telemetry and is simply skipped, as before).

        The era's own operator king pod when still mapped, else that pod
        re-rented through the king-leg rent path (``_rent_king_host`` adopts
        the era's pod by name while it is still up — the startup sweep keeps
        it — so a restart normally rents nothing). The rent carries no target
        (no capacity wait) and its give-up never cancels challenger capacity
        waits. The pod is named for the era, so the rollover retires it and
        the sweeps keep it meanwhile. Another era's king pod is never
        borrowed: the next era's may be training its king leg. Returns the
        host with the king checkpoint pushed, or None (the scheduler re-tries
        after KING_REBENCH_SECONDS)."""
        from ..shared.era import bench_gate_configured
        from .bench_hook import push_checkpoint

        if not bench_gate_configured(self.cfg):
            return None
        hosts = self.__dict__.setdefault("_rolling_king_hosts", {})
        host = hosts.get(era.index)
        if host is None:
            if not self._king_rent_serial.acquire(timeout=self.KING_BENCH_RENT_WAIT_S):
                log.warning("rolling: era %d king bench — a king rent is in progress; "
                            "retried later", era.index)
                return None
            try:
                with self._funded_king_lock:
                    held = self._rolling_king_host_era
                    if held is not None and held > era.index and self._funded_king_host:
                        held_later = True
                    else:
                        held_later = False
                        if held != era.index:
                            self._funded_king_host = None
                            self._rolling_king_host_era = era.index
                if held_later:
                    log.warning("rolling: era %d king bench — the king pod slot belongs to "
                                "era %d; retried later", era.index, held)
                    return None
                log.warning("rolling: era %d king %s has NO bench host (trainer restart or "
                            "released pod) and the bench gate needs its rows — re-renting "
                            "the era's operator king pod for the bench only", era.index,
                            entry.miner_hotkey[:12])
                abort = getattr(self, "_funded_wait_abort", None)
                was_set = abort.is_set() if abort is not None else True
                try:
                    from .leg_pins import leg_contract_scope

                    # DEC-CA-0067: the bench pod is rented on THIS era's image
                    # pins (leg_round reads the era contract on this thread)
                    with leg_contract_scope(self, contract):
                        host = self._rent_king_host(str(era.base_seed))
                except Exception as e:  # noqa: BLE001 — retried by the scheduler
                    if abort is not None and not was_set:
                        abort.clear()   # a bench rent's give-up never releases challengers
                    log.warning("rolling: era %d king bench pod rent failed: %s", era.index, e)
                    return None
                hosts[era.index] = host
            finally:
                self._king_rent_serial.release()
        push_checkpoint(host, self._fetch_checkpoint_dir(entry.trained_pointer),
                        str(era.base_seed), contract.arch_preset, "king")
        self._final_role_hosts[("king", contract.arch_preset, entry.miner_hotkey)] = host
        return host

    def _rolling_bench_king(self, entry, era) -> dict | None:
        # LEGACY-ONLY for lane A: a no-op for a validator-sourced era (DEC-CA-0063)
        import dataclasses

        contract = era_contracts(self.cfg, era)[0]
        if self.cascade_bench_plan is None or _validator_benches(self.cfg, era):
            return None
        host = self._final_role_hosts.get(("king", contract.arch_preset, entry.miner_hotkey))
        if host is None:
            host = self._rolling_king_bench_host(entry, era, contract)
        if host is None:
            return None
        why = host_cannot_bench(self, host)
        if why:
            log.warning("rolling: era %d king bench skipped — %s", era.index, why)
            return None
        scores = self._remote_bench_scores(host, entry, str(era.base_seed),
                                           contract.arch_preset, role_dir="king",
                                           require_fev=_bench_fev_required(self.cfg, era),
                                           contract_block=self._era_contract_block(era))
        return dataclasses.asdict(scores) if scores is not None else None

    def _rolling_bench_init(self, era) -> dict | None:
        """The seed epoch's CURRENT init entry (DEC-CA-0055 "init").

        Validator-sourced era (DEC-CA-0063): the trainer benches nothing and
        rents nothing — see :meth:`_rolling_publish_init_pointer`.

        Trainer-sourced (legacy) era: bench the init on the era's operator
        king pod — the member checkpoint when the epoch is warm-started, else
        the random-init model, BUILT ON THAT POD from the epoch's training
        seed (never on the trainer box) and, once benched, harvested and
        published. Operator numbers only (never a payer pod)."""
        import dataclasses

        from ..eval.benchmarks import extract_bench_scores
        from .bench_hook import push_checkpoint, run_post_round_benchmark

        if _validator_benches(self.cfg, era):
            return self._rolling_publish_init_pointer(era)
        if self.cascade_bench_plan is None:
            return None
        king = era.king()
        if king is None:
            return None
        contract = era_contracts(self.cfg, era)[0]
        host = (self._final_role_hosts.get(("king", contract.arch_preset, king.miner_hotkey))
                or self.__dict__.get("_rolling_king_hosts", {}).get(era.index))
        if host is None:
            host = self._rolling_king_bench_host(king, era, contract)
        if host is None or getattr(host, "isolated", False):
            return None
        why = host_cannot_bench(self, host)
        if why:
            log.warning("rolling: era %d init bench skipped — %s", era.index, why)
            return None
        round_id = str(era.base_seed)
        if era.warm_start_ckpt:
            local = self._fetch_checkpoint_dir(era.warm_start_ckpt)
            push_checkpoint(host, local, round_id, contract.arch_preset, "init")
            remote_dir = ""
        else:
            if not _init_rebuildable(contract):
                log.warning("rolling: era %d init bench — random init of base_arch %r is "
                            "not rebuildable; skipped", era.index,
                            getattr(contract, "base_arch", ""))
                return None
            # built straight into the "init" role dir the sweep benches
            remote_dir = self._rolling_build_init_on_pod(host, era, contract)
        report = run_post_round_benchmark(host, round_id, contract.arch_preset,
                                          self.cascade_bench_plan, work_root=self.work_root,
                                          role="init")
        raw = (extract_bench_scores(report, require_fev=_bench_fev_required(self.cfg, era))
               if report is not None else None)
        if raw is None:
            return None
        out = dataclasses.asdict(BenchScores(**raw))
        pointer = str(era.warm_start_ckpt or "")
        if not era.warm_start_ckpt:
            # validators score the init on the pool: the random-init model must
            # be a PUBLISHED, fetchable checkpoint (the init entry's pointer)
            pointer = out["_pointer"] = self._rolling_harvest_publish_init(
                host, remote_dir, era, contract)
        if pointer:
            # DEC-CA-0061: the king check's bench half runs the gate's rule on
            # the init's TASK rows, so they are published like any entry's
            self._stash_bench_rows(pointer, report)
        return out

    def _rolling_publish_init_pointer(self, era) -> dict | None:
        """DEC-CA-0063 era: publish the random-init checkpoint POINTER the main
        validator's king check benches (it sweeps the init on its own GPU).

        No bench, never a rent, never a payer (miner-key) pod, never a pod
        with a leg in flight. The init is built — CPU, contract-digest checked
        on the pod — harvested to the box as plain files and uploaded, on the
        first OPERATOR-billed pod the trainer already holds for the seed
        epoch, in this order:

        1. the era's operator king pod while still mapped (right after a new
           seed epoch's king leg, which trains from this same random init);
        2. an IDLE operator lane (checked out of the lane pool, so no leg can
           take it meanwhile; never the last idle lane);
        3. an INIT-ONLY lane on file (hosts.toml ``stage = "init"``,
           :meth:`_rolling_init_lanes`) — the operator's way to supply a pod
           when the epoch has no operator king pod and no lane pool (2026-11-08:
           an own-key king, every leg on payer pods, pointer parked for days);
        4. otherwise the request is parked: the next operator-billed
           challenger pod of this seed epoch builds it right after its leg's
           harvest, before its release (:meth:`_rolling_init_on_leg_pod`).

        Returns the pointer-only entry ``{six sentinels, "_pointer"}``, or
        ``{"_deferred": why}`` while parked (the scheduler re-polls without
        spending a try), or None (nothing to publish). A failed build raises
        (a try). A warm-started epoch needs nothing."""
        if era.warm_start_ckpt:
            return None
        built = (str(getattr(era, "init_pointer", "") or "")
                 or self.__dict__.get("_init_pointers_built", {}).get(str(era.base_seed), ""))
        if built:
            return {**pointer_only_init_scores(), "_pointer": built}
        if str(era.base_seed) in self.__dict__.get("_init_building", set()):
            return {"_deferred": "being built on an operator leg pod"}
        contract = era_contracts(self.cfg, era)[0]
        if not _init_rebuildable(contract):
            log.warning("rolling: era %d init pointer — random init of base_arch %r is not "
                        "rebuildable; not published", era.index,
                        getattr(contract, "base_arch", ""))
            return None
        king = era.king()
        host = (self._final_role_hosts.get(("king", contract.arch_preset, king.miner_hotkey))
                if king is not None else None)
        host = host or self.__dict__.get("_rolling_king_hosts", {}).get(era.index)
        if host is not None and self._rolling_operator_host(host):
            return self._rolling_init_entry_on(host, era, contract, "operator king pod")
        lane = self._rolling_take_idle_operator_lane()
        if lane is not None:
            pool, lane_host = lane
            try:
                return self._rolling_init_entry_on(lane_host, era, contract,
                                                   "idle operator lane")
            finally:
                pool.put(lane_host)
        for init_lane in self._rolling_init_lanes():
            # the lane trains on the DEPLOYED chain.toml like a rented pod (the
            # on-pod build checks the contract digest against it)
            push = getattr(self, "_push_deployed_chain_toml", None)
            target = push(init_lane) if callable(push) else init_lane
            return self._rolling_init_entry_on(target, era, contract, "init lane")
        self.__dict__.setdefault("_init_pointer_wanted", {})[str(era.base_seed)] = era
        why = ("the era's king pod is a payer (miner-key) pod" if host is not None
               else "no operator king pod (cached/carried or own-key king leg, or released)")
        logged = self.__dict__.setdefault("_init_defer_logged", set())
        if era.index not in logged:
            logged.add(era.index)
            log.warning("rolling: era %d init pointer deferred — %s, no idle operator lane and "
                        "no init lane (hosts.toml stage = \"init\") on file; NOT renting one. "
                        "The next operator-billed challenger pod of seed epoch %s builds it "
                        "after its leg", era.index, why, era.base_seed)
        return {"_deferred": why}

    def _rolling_init_entry_on(self, host, era, contract, what: str) -> dict:
        pointer = self._rolling_harvest_publish_init(
            host, self._rolling_build_init_on_pod(host, era, contract), era, contract)
        self.__dict__.setdefault("_init_pointers_built", {})[str(era.base_seed)] = pointer
        self.__dict__.get("_init_pointer_wanted", {}).pop(str(era.base_seed), None)
        log.info("rolling: era %d random-init checkpoint built on %s %s and published → %s "
                 "(pointer only; the main validator benches it)", era.index, what,
                 getattr(host, "name", host), pointer)
        return {**pointer_only_init_scores(), "_pointer": pointer}

    def _rolling_operator_host(self, host) -> bool:
        """``host`` is OPERATOR-billed: an operator lane on file (not
        isolated), or a credential-free (isolated) pod whose ledger row has no
        payer — never a payer's (miner-key) pod. ``isolated`` alone means
        credential-free, which operator king / leg pods are too. Unknown ⇒
        False."""
        if host is None:
            return False
        if not getattr(host, "isolated", False):
            return True
        aid = str(getattr(host, "allocation_id", "") or "")
        if not aid:
            return False
        try:
            pods = self._load_funded_ledger()
        except Exception:  # noqa: BLE001 — unknown billing ⇒ not ours to use
            return False
        return any(str(p.instance_id) == aid and not str(p.payer_hotkey or "") for p in pods)

    def _rolling_take_idle_operator_lane(self):
        """``(pool, host)``: an operator lane checked OUT of the existing
        rolling lane pool (idle by the pool's own occupancy — a lane in the
        queue has no leg), or None. Never builds the pool (that probes every
        lane) and never takes the LAST idle lane (a fallback leg may need it)."""
        import queue as _queue

        pool = self.__dict__.get("_rolling_lane_pool_obj")
        if pool is None or pool.qsize() < 2:
            return None
        try:
            host = pool.get(block=False)
        except _queue.Empty:
            return None
        if not self._rolling_operator_host(host):
            pool.put(host)
            return None
        return pool, host

    def _rolling_init_on_leg_pod(self, host, base_seed, teardown) -> bool:
        """Called by an operator-billed leg pod's release, right after its
        leg's verified harvest (the worker has exited). When a parked init
        request is for this pod's seed epoch, claim it and build there in a
        background thread that tears the pod down when done (``teardown``);
        returns True then (the caller must NOT tear down). Else False."""
        wanted = self.__dict__.get("_init_pointer_wanted", {})
        key = str(base_seed)
        if key not in wanted or not self._rolling_operator_host(host):
            return False
        era = wanted.pop(key, None)
        if era is None:
            return False
        building = self.__dict__.setdefault("_init_building", set())
        building.add(key)

        def _run() -> None:
            try:
                contract = era_contracts(self.cfg, era)[0]
                self._rolling_init_entry_on(host, era, contract,
                                            "operator challenger pod (after its leg)")
            except Exception as e:  # noqa: BLE001 — re-parked for the next pod
                wanted.setdefault(key, era)
                log.warning("rolling: era %d init build on leg pod %s failed: %s", era.index,
                            getattr(host, "name", host), e)
            finally:
                building.discard(key)
                teardown()

        threading.Thread(target=_run, name=f"init-on-leg-{key[:8]}", daemon=True).start()
        return True

    def _rolling_build_init_on_pod(self, host, era, contract) -> str:
        """Build the epoch's random-init checkpoint ON ``host`` (CPU), at the
        "init" role checkpoint dir; returns that pod-side dir."""
        from ..shared.manifest import contract_digest
        from .bench_hook import role_paths
        from .remote import build_ssh_argv, error_tail, run_ssh

        ckpt, _ = role_paths(host, str(era.base_seed), contract.arch_preset, "init")
        cmd = build_init_remote_command(
            host, ckpt, contract_block=int(era.start_block),
            training_seed=int(self._rolling_seeds(era).training_seed),
            arch_preset=str(contract.arch_preset), digest=contract_digest(contract))
        proc = run_ssh(build_ssh_argv(host, cmd), INIT_BUILD_TIMEOUT_S)
        if proc.returncode != 0:
            raise RuntimeError(f"random-init build on {getattr(host, 'name', host)} failed "
                               f"(rc={proc.returncode}): "
                               f"{error_tail((proc.stderr or '') + (proc.stdout or ''))}")
        return ckpt

    def _rolling_harvest_publish_init(self, host, remote_dir: str, era, contract) -> str:
        """Copy the pod-built init back as plain files (verified file-for-file
        against the pod's own listing — nothing is loaded on the box), upload
        it, drop the box copy. Returns the init's trained pointer."""
        import shutil
        from pathlib import Path

        from .harvest_release import local_file_manifest, manifest_mismatches
        from .remote import harvest_remote_dir, remote_file_manifest

        dest = Path(self.work_root) / "init_publish" / str(era.base_seed)
        try:
            pod = remote_file_manifest(host, remote_dir)
            if not pod:
                raise RuntimeError(f"pod {getattr(host, 'name', host)} lists no init files "
                                   f"under {remote_dir}")
            harvest_remote_dir(host, remote_dir, dest)
            bad = manifest_mismatches(pod, local_file_manifest(dest))
            if bad:
                raise RuntimeError("harvested init differs from the pod's copy: "
                                   + "; ".join(bad[:3]))
            return self._publish_init_checkpoint(dest, era, contract)
        finally:
            shutil.rmtree(dest, ignore_errors=True)

    def _publish_init_checkpoint(self, local, era, contract) -> str:
        """Upload the rebuilt random-init checkpoint under the operator's
        identity → its ``metro-v1:trained:`` pointer."""
        from pathlib import Path

        from ..shared.bench_report import random_init_repo_name
        from ..shared.hippius import upload_dir_to_hub_or_hf
        from ..shared.manifest import format_trained_pointer

        # the basename binds the seed epoch (its seed block): validators check
        # it once [v2] init_epoch_bind_from_block is in force
        repo = random_init_repo_name(era.base_seed, contract.arch_preset, int(era.seed_block))
        full = f"{self.hub().namespace}/{repo}"
        up = upload_dir_to_hub_or_hf(Path(local), full, self.hub(),
                                     hf_repo=self._hf_ckpt_repo(full))
        return format_trained_pointer(up.ref.immutable_ref)

    def _rolling_bench_challenger(self, entry, king, era) -> dict | None:
        """Bench at completion (DEC-CA-0043): on the payer's kept pod, then the
        operator re-bench on the era king's pod; only operator numbers are
        returned (forged sweep ⇒ None). LEGACY-ONLY for lane A: a no-op for a
        validator-sourced era (DEC-CA-0063)."""
        import dataclasses

        if _validator_benches(self.cfg, era):
            # DEC-CA-0063: the main validator benches only private-pool
            # winners on its own GPU — no payer-pod sweep, no re-bench
            log.info("round=%s: %s's leg is not benched by the trainer (validator bench "
                     "source)", era.base_seed, entry.miner_hotkey[:12])
            return None
        contract = era_contracts(self.cfg, era)[0]
        host = self._final_role_hosts.get(("challenger", contract.arch_preset, entry.miner_hotkey))
        if host is None or self.cascade_bench_plan is None:
            return None
        role_dir = f"challenger-u{entry.miner_uid}"
        scores = self._remote_bench_scores(host, entry, str(era.base_seed),
                                           contract.arch_preset, role_dir=role_dir,
                                           require_fev=_bench_fev_required(self.cfg, era),
                                           contract_block=self._era_contract_block(era))
        if scores is None:
            return None
        if not getattr(host, "isolated", False):
            return dataclasses.asdict(scores)              # operator lane: our own numbers
        # A payer's pod: publish only numbers the operator reproduced on its
        # own era king pod. When that cannot happen NOW (no era king leg yet,
        # king pod busy benching, pod gone) the payer numbers are handed back
        # for the scheduler's verification queue instead of being dropped.
        from .bench_hook import host_busy
        from .rolling import VERIFY_PENDING

        payer = dataclasses.asdict(scores)
        if int(getattr(self.cfg.telemetry, "funded_bench_verify_top", 1)) <= 0:
            log.info("round=%s: payer-pod bench for %s unverified (verification off) — "
                     "not published", era.base_seed, entry.miner_hotkey[:12])
            return None
        if king is None:
            return {VERIFY_PENDING: {**payer, "_reason": "era king leg not trained yet"}}
        king_host = self._rolling_verify_host(king, era)
        if king_host is not None and host_busy(king_host):
            return {VERIFY_PENDING: {**payer, "_reason": "king pod busy"}}
        status, ours = self._rolling_verify_payer(entry, payer, king, era)
        if status == "ok":
            return ours
        if status == "forged":
            return None
        return {VERIFY_PENDING: {**payer, "_reason": f"verification {status}"}}

    def _rolling_verify_host(self, king, era):
        contract = era_contracts(self.cfg, era)[0]
        return (self._final_role_hosts.get(("king", contract.arch_preset, king.miner_hotkey))
                or self.__dict__.get("_rolling_king_hosts", {}).get(era.index))

    def _rolling_verify_payer(self, entry, payer: dict, king, era) -> tuple[str, dict | None]:
        """Re-bench ONE payer-benched checkpoint on the era king's operator pod.

        ``("ok", scores)`` — reproduced within tolerance: the OPERATOR's numbers;
        ``("forged", None)`` — the payer's report beats ours by more than the
        tolerance (tamper: dropped); ``("wait", None)`` — no usable king pod
        yet (costs no attempt); ``("failed", {"error"})`` — the re-bench ran
        and produced nothing. Bench runs on one pod are serialized
        (bench_hook host lock), so this never pre-empts another sweep."""
        import dataclasses

        from ..eval.benchmarks import extract_bench_scores
        from ..validator.cascade import cascade_score
        from .bench_hook import push_checkpoint, run_post_round_benchmark

        contract = era_contracts(self.cfg, era)[0]
        primary = contract.arch_preset
        king_host = self._rolling_verify_host(king, era)
        if (king_host is None and self.cascade_bench_plan is not None
                and int(getattr(era, "epoch_start", -1)) >= 0
                and int(era.index) != int(era.epoch_start)):
            # DEC-CA-0055: a later era of a seed epoch rents no king leg pod,
            # so "wait for the next king pod" could wait days — rent the era's
            # operator pod for the bench (bench-gate configured only).
            king_host = self._rolling_king_bench_host(king, era, contract)
        if king_host is None or self.cascade_bench_plan is None:
            return "wait", None
        if host_cannot_bench(self, king_host):
            return "wait", None          # never burn an attempt on a pod that cannot bench

        def _score(d: dict) -> float:
            return cascade_score(d["gifteval_crps"], d["gifteval_mase"], d["boom_crps"],
                                 d["boom_mase"], d["time_crps"], d["time_mase"])

        round_id = str(era.base_seed)
        try:
            local = self._fetch_checkpoint_dir(entry.trained_pointer)
            role_dir = f"challenger-u{entry.miner_uid}-verify"
            push_checkpoint(king_host, local, round_id, primary, role_dir)
            report = run_post_round_benchmark(king_host, round_id, primary,
                                              self.cascade_bench_plan,
                                              work_root=self.work_root, role=role_dir)
            raw = (extract_bench_scores(report, require_fev=_bench_fev_required(self.cfg, era))
                   if report is not None else None)
        except Exception as e:  # noqa: BLE001 — an attempt that ran and failed
            return "failed", {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        if raw is None:
            return "failed", {"error": "operator re-bench produced no scores"}
        ours = dataclasses.asdict(BenchScores(**raw))
        self._stash_bench_rows(entry.trained_pointer, report)       # the operator's sweep
        reported, actual = _score(payer), _score(ours)
        tol = float(getattr(self.cfg.telemetry, "funded_bench_verify_tolerance", 0.02))
        if reported < actual * (1.0 - tol):
            log.error("round=%s: payer-pod bench for %s reported %.5f but the operator's "
                      "re-bench scores %.5f — forged sweep; entry dropped (TAMPER)",
                      round_id, entry.miner_hotkey[:12], reported, actual)
            return "forged", None
        log.info("round=%s: payer-pod bench for %s verified (payer %.5f, operator %.5f) — "
                 "publishing the operator's numbers", round_id, entry.miner_hotkey[:12],
                 reported, actual)
        return "ok", ours

    def _rolling_publish_bench(self, round_id: str, created_block: int, pairs: list, *,
                               rows_for: frozenset | None = None, init: dict | None = None):
        import dataclasses

        from ..shared.bench_report import (
            BenchEntry,
            BenchReport,
            dump_bench_report,
            publish_bench_report,
            sign_bench_report,
        )

        primary = settlement_contracts(self.cfg, int(created_block))[0].arch_preset
        rows = self._bench_rows_registry()
        if rows_for is not None:            # a republish: only the first version's rows
            rows = {k: v for k, v in rows.items() if k in rows_for}
        entries = tuple(BenchEntry(role=e.role, size=e.size or primary,
                                   miner_hotkey=e.miner_hotkey, miner_uid=e.miner_uid,
                                   trained_pointer=e.trained_pointer, scores=s,
                                   rows_key=rows.get(e.trained_pointer, {}).get("key", ""),
                                   rows_sha256=rows.get(e.trained_pointer, {}).get("sha256", ""))
                        for e, s in pairs)
        if init:
            # DEC-CA-0055: the CURRENT init's numbers (role "init"; pointer "" =
            # the seed epoch's random init), what the king check compares the
            # king's six suite means against.
            try:
                ip = str(init.get("pointer") or "")
                ie = BenchEntry(
                    role="init", size=str(init.get("size") or primary), miner_hotkey="",
                    miner_uid=-1, trained_pointer=ip,
                    scores=BenchScores(**init["scores"]),
                    rows_key=rows.get(ip, {}).get("key", "") if ip else "",
                    rows_sha256=rows.get(ip, {}).get("sha256", "") if ip else "")
                if ie.pointer_only:
                    # DEC-CA-0063 pointer-only init: alone in its report, with no
                    # rows, so even a reader that predates the marker defers
                    # before comparing its sentinels — the king check finds no
                    # king entry (and under rule v2 no init rows)
                    ie = dataclasses.replace(ie, rows_key="", rows_sha256="")
                    if entries:
                        log.warning("bench report round=%s: %d scored entr%s dropped next to "
                                    "a pointer-only init entry (validator bench source: "
                                    "the main validator's report carries them)", round_id,
                                    len(entries), "y" if len(entries) == 1 else "ies")
                    entries = ()
                entries = (*entries, ie)
            except (ValueError, KeyError, TypeError) as e:
                log.warning("bench report round=%s: init entry not representable on this "
                            "build (%s) — published without it", round_id, e)
        report = BenchReport(round_id=str(round_id), created_block=int(created_block),
                             entries=entries)
        if self.wallet is not None:
            report = sign_bench_report(report, self.wallet)
        key = publish_bench_report(self.manifest_store(), dump_bench_report(report), round_id)
        log.info("published bench report round=%s roles=[%s] signed=%s → s3://%s/%s",
                 round_id, ", ".join(e.role for e in entries), report.signature is not None,
                 self.cfg.storage.manifest_bucket, key)
        with contextlib.suppress(Exception):   # telemetry only
            self._log_bench_pair_wandb(report)
        return report

    def _rolling_latest_round_id(self) -> str:
        """``round_id`` of the manifest bucket's ``latest.json`` — the chain
        root the first settlement links to. "" when unreadable (the scheduler
        then withholds the settlement rather than publish an unchained one)."""
        from ..shared.hippius import read_latest_manifest
        from ..shared.manifest import load_manifest

        try:
            return str(load_manifest(read_latest_manifest(self.manifest_store())).round_id)
        except Exception as e:  # noqa: BLE001 — absent/unreadable ⇒ unknown root
            log.warning("rolling: latest.json unreadable (%s); manifest chain root unknown", e)
            return ""

    def _rolling_promotion_effective_era(self, generation: int) -> int | None:
        """``effective_era`` of the published ``promotions/gen-<n>.json``
        (0 = a pre-era record); None when the record cannot be read."""
        from ..shared.promotion import load_promotion_record, promotion_record_key

        try:
            rec = load_promotion_record(
                self.manifest_store().get_text(promotion_record_key(int(generation))))
        except Exception as e:  # noqa: BLE001 — not published yet / store down
            log.warning("rolling: promotion record gen=%d unreadable (%s)", generation, e)
            return None
        return int(getattr(rec, "effective_era", 0) or 0)

    def _rolling_publish_roster(self, round_id: str, roster: dict) -> None:
        from ..shared.hippius import publish_public_json

        doc = {"funded_pods": self.cfg.round.funded_pods, **roster}
        try:
            store = self.manifest_store()
            for key in (f"funded/round-{round_id}.json", "funded/latest.json"):
                publish_public_json(store, key, doc)
        except Exception as e:  # noqa: BLE001 — transparency must not sink a settlement
            log.warning("funded roster publish failed (ignored): %s", e)

    def _rolling_dedup_registry(self):
        from .dedup_registry import DedupRegistry

        mode = (self.cfg.round.dedup_mode or "off").lower()
        if mode not in ("shadow", "enforce"):
            return None
        reg = self.__dict__.get("_dedup_registry_obj")
        if reg is None:
            reg = DedupRegistry(self, self.work_root / "dedup_registry.json", mode=mode)
            self.__dict__["_dedup_registry_obj"] = reg
        return reg


def record_lane_breaker_victim(runner, hotkey: str, exc: BaseException) -> bool:
    """A leg that failed on a lane the circuit breaker just took out
    (``exc.lane_breaker``) failed because of that LANE: its verdict is a
    capacity miss — requeued unburned and WITHOUT the operator-fault backoff
    (the queue backs off only ``operator`` class), so the healthy rest of the
    fleet can take it at once. Returns True when it recorded one."""
    if not getattr(exc, "lane_breaker", False):
        return False
    runner._record_funded_failure(
        hotkey, f"LANE BREAKER: the operator lane this leg ran on failed several "
        f"different legs in a row and left the rotation — requeued without backoff "
        f"({str(exc)[-300:]})", miner_fault=False, error_class="no_capacity", burn=False)
    return True


def _hosts_on_file(runner) -> list:
    """Every hosts.toml entry NOW: re-read on each call when the trainer runs
    from a file (so a lane hand-rented after startup is seen), else the
    startup snapshot."""
    from .remote import load_hosts

    if getattr(runner, "remote_hosts_path", None) is not None:
        return list(load_hosts(runner.remote_hosts_path))
    return list(getattr(runner, "remote_hosts", None) or ())


def _with_deployed_chain_toml(runner, host):
    """An operator lane trains on the orchestrator's DEPLOYED chain.toml, the
    same file a rented pod gets (``_push_deployed_chain_toml``; the join gate
    pushes it). Without this a hand-rented lane that boots the image's baked
    snapshot resolves the wrong contract: 2026-11-01 two Targon 8xH200 lanes
    failed every v2 leg ``unknown --arch-preset 'cascade-moe-v2-120m'`` for
    ~1.5 h. A host that names its own ``chain_toml`` keeps it."""
    from dataclasses import replace as dc_replace

    if getattr(host, "chain_toml", None) or not getattr(runner, "deployed_chain_toml", None):
        return host
    return dc_replace(host, chain_toml=f"{host.workdir}/chain.deployed.toml")
