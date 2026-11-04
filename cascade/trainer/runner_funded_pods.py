"""Per-payer funded pods: vault, rent, ledger, identity and ingest guards (TrainerRunner mixin)."""

from __future__ import annotations

import contextlib
import json
import logging
import shlex
import threading
import time
from dataclasses import replace
from pathlib import Path

from ..shared.manifest import (
    TrainedEntry,
    contract_digest,
)
from .faults import (
    KEY_LOST_REASON,
    _FundedDiverged,
    _FundedLegSkip,
    _FundedOperatorFallback,
    _FundedOperatorRent,
    _quarantine_lane_host,
)
from .lane_pool import king_pod_name_prefix
from .remote import error_tail
from .runner_common import (
    ResolvedGenerator,
    claim_executor,
    live_claimed_executors,
    release_executor_of,
)

log = logging.getLogger("cascade.trainer")


def _parse_funded_ledger(text):
    from dataclasses import fields

    from ..provision.state import PodInstance

    raw = json.loads(text)
    if not isinstance(raw, list):
        raise ValueError("funded ledger must be a list")
    known = {f.name for f in fields(PodInstance)}
    return [PodInstance(**{k: v for k, v in d.items() if k in known}) for d in raw]


def _entry_gpu_choice(entry, ref: str):
    """The GPU choice (DEC-CA-0060) of ``entry`` when it is the in-flight
    lane-A entry for ``ref``, else the default ``GpuChoice()`` — a king leg on
    its own key, or a hotkey whose entry moved on, never inherits a choice
    made for another leg."""
    from ..funding.gpu_choice import GpuChoice

    if (entry is None or getattr(entry, "status", "") != "in_flight"
            or getattr(entry, "ref", None) != ref or not getattr(entry, "gpu", "")):
        return GpuChoice()
    return GpuChoice(gpu=str(entry.gpu),
                     allow_fallback=bool(getattr(entry, "gpu_allow_fallback", False)),
                     max_price_per_hour=float(getattr(entry, "gpu_max_price_per_hour", 0.0)
                                              or 0.0))


def _gpu_choice_key(choice) -> tuple:
    """A comparable form of a ``GpuChoice`` (the type case-insensitive; the
    fallback flag and cap only mean something with a type chosen)."""
    gpu = str(getattr(choice, "gpu", "") or "").strip().lower()
    if not gpu:
        return ("", False, 0.0)
    return (gpu, bool(getattr(choice, "allow_fallback", False)),
            float(getattr(choice, "max_price_per_hour", 0.0) or 0.0))


def _changed_gpu_choice(runner, hotkey: str, ref: str, attempt_id: str, current):
    """The miner's NEW GPU choice for a leg waiting for capacity, or None to
    keep ``current`` (incident 2026-11-03: a re-fund switched 8xH200 → 8xH100
    but the waiting leg polled for 8xH200 until its deadline).

    Only the SAME in-flight entry counts — status ``in_flight``, this leg's
    ``ref`` and ``attempt_id`` — and not one whose re-fund queued a new ref
    behind the flight (that choice belongs to the queued submission). Any read
    failure keeps the current choice: a torn queue never ends a leg."""
    from ..funding.gpu_choice import GpuChoice

    try:
        queue = runner._funded_queue()
        entry = queue.get(hotkey) if queue is not None else None
        if (entry is None or getattr(entry, "status", "") != "in_flight"
                or getattr(entry, "ref", None) != ref
                or str(getattr(entry, "attempt_id", "") or "") != attempt_id
                or getattr(entry, "queued_ref", "")):
            return None
        gpu = str(getattr(entry, "gpu", "") or "")
        new = GpuChoice(gpu=gpu,
                        allow_fallback=bool(getattr(entry, "gpu_allow_fallback", False))
                        if gpu else False,
                        max_price_per_hour=float(getattr(entry, "gpu_max_price_per_hour", 0.0)
                                                 or 0.0) if gpu else 0.0)
    except Exception as e:  # noqa: BLE001 — unreadable queue: keep the current choice
        log.debug("funded leg %s: GPU choice re-read failed (%s) — keeping %s",
                  hotkey[:12], e, getattr(current, "gpu", "") or "the default")
        return None
    return None if _gpu_choice_key(new) == _gpu_choice_key(current) else new


def _describe_gpu_choice(choice) -> str:
    gpu = str(getattr(choice, "gpu", "") or "")
    if not gpu:
        return "default (cheapest fitting type)"
    cap = float(getattr(choice, "max_price_per_hour", 0.0) or 0.0)
    return (f"{gpu} (cap {f'${cap:.2f}/h' if cap else 'none'}"
            f"{', fallback allowed' if getattr(choice, 'allow_fallback', False) else ''})")


@contextlib.contextmanager
def allocation_lock(lock):
    """Serialize selection/launch; release once its executor is claimed."""
    lock.acquire()
    held = True

    def release():
        nonlocal held
        if held:
            held = False
            lock.release()

    try:
        yield release
    finally:
        release()


def _kept_pod_record(runner, pod_id: str):
    from .box_health import BoxHealth

    for hk, rec in (BoxHealth.for_runner(runner).state().get("legs") or {}).items():
        if rec.get("pod_id") == pod_id and rec.get("kept"):
            return hk, rec
    return None, None


def _kept_past_budget(runner, pod_id: str, now: float) -> bool:
    """``pod_id`` was kept after OUR harvest fault past its PRICED keep
    (:func:`~cascade.trainer.harvest_guard.keep_budget_seconds`): release it
    (priced alert, run lost)."""
    try:
        from .harvest_guard import keep_budget_seconds

        hk, rec = _kept_pod_record(runner, pod_id)
        if rec is None:
            return False
        limit = keep_budget_seconds(runner, rec)
        if time.time() - float(rec.get("kept_since") or now) < limit:
            return False
        from .box_health import BoxHealth, miner_lost_usd
        from .money_guard import alert

        BoxHealth.for_runner(runner).note_leg(hk, {"kept": False})
        who = "MINER" if rec.get("payer") else "operator"
        r2 = ("the pod → R2 push was tried" if rec.get("r2_tried")
              else "NO pod → R2 push was tried (R2 not configured or not reached)")
        alert(runner, "miner_billed_loss" if rec.get("payer") else "operator_billed_loss",
              hk[:16], f"kept pod {pod_id} released after the priced keep of {limit:.0f} s "
              f"({r2}) — the finished run is lost: ≤ ${miner_lost_usd(runner):.2f} billed to "
              f"the {who}; its retrain is an operator make-good (owner approval)",
              severity="critical")
        log.error("funded sweep: releasing KEPT pod %s of %s — past the %.0f s keep budget",
                  pod_id, hk[:12], limit)
        return True
    except Exception as e:  # noqa: BLE001 — unsure ⇒ the normal sweep rule
        log.warning("funded sweep: kept-pod budget check failed for %s (%s)", pod_id, e)
        return False


class _FundedPodMixin:
    """Per-payer funded pods (DEC-CA-0036, ``[round] funded_pods = "rent"``)."""

    # ── per-payer funded pods (DEC-CA-0036, [round] funded_pods = "rent") ────

    def _payer_vault(self):
        """The payer-key vault shared with cascade-intake, or ``None`` unset."""
        d = self.cfg.round.payer_vault_dir
        if not d:
            return None
        from ..funding.vault import PayerKeyVault

        p = Path(d)
        vault = PayerKeyVault(
            dir=(p if p.is_absolute() else (self.work_root / p)),
            ttl_seconds=self.cfg.round.funded_entry_ttl_hours * 3600.0,
        )
        # The intake wrote these keys from ITS process; this one starts empty —
        # hydrate from the shared directory or every get() comes back None.
        vault.hydrate()
        return vault

    # Env pairs the funded-leg credential resolver reads. Neither is ever
    # forwarded to a pod: the ADMIN pair mints per-pod robots and stays on the
    # orchestrator; the FUNDED pair is the static fallback that IS handed to
    # payer pods (a push-only robot the operator created by hand).
    HUB_ADMIN_ENV = ("CASCADE_HUB_ADMIN_USERNAME", "CASCADE_HUB_ADMIN_PASSWORD")
    FUNDED_HUB_ENV = ("CASCADE_FUNDED_HUB_USERNAME", "CASCADE_FUNDED_HUB_PASSWORD")

    def _hub_robots(self):
        """The Harbor robot minter on a PROJECT-ADMIN Hub login, or ``None``.

        Harbor refuses to let a robot account manage robots, and the
        operator's everyday Hub identity is itself a project robot
        (``robot$cascade+cascade-bot``), so minting needs a real user login:
        ``CASCADE_HUB_ADMIN_USERNAME`` / ``CASCADE_HUB_ADMIN_PASSWORD``. Absent
        that, ``None`` — the caller then tries the static funded robot and
        otherwise fails CLOSED. Nothing here ever forwards the operator's
        login to a payer-controlled pod.
        """
        import base64
        import os

        try:
            from ..funding.robots import HarborRobots
            from ..shared.hippius import HubConfig

            user = os.environ.get(self.HUB_ADMIN_ENV[0], "")
            pw = os.environ.get(self.HUB_ADMIN_ENV[1], "")
            if not (user and pw):
                return None
            if user.startswith("robot$"):
                log.error("%s is a robot account — Harbor forbids robots minting "
                          "robots; set a project-admin USER login", self.HUB_ADMIN_ENV[0])
                return None
            header = "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()
            return HarborRobots(HubConfig.from_storage(self.cfg.storage).registry_url,
                                header)
        except Exception as e:  # noqa: BLE001 — a minter fault is an infra fault
            log.error("Hub robot minter unavailable: %s", e)
            return None

    def _funded_pod_credential(self, round_id: str, hotkey: str):
        """``(env_pairs, robot_id)`` for one payer pod, or ``None`` (fail closed).

        Preference: a per-pod robot minted now (revoked at teardown) → the
        static funded robot from the environment (rotated by hand) → nothing.
        The operator's own Hub login is NEVER an option here.
        """
        import os

        from ..funding.robots import robot_name

        minter = self._hub_robots()
        if minter is not None:
            try:
                cred = minter.create(
                    robot_name(self.cfg.subnet.netuid, str(round_id), hotkey),
                    self.hub().namespace,
                    duration_days=self.cfg.round.funded_robot_duration_days)
                return cred.as_env(), cred.id
            except Exception as e:  # noqa: BLE001 — fall through to the static robot
                log.error("Hub robot mint for %s failed: %s", hotkey, e)
        user = os.environ.get(self.FUNDED_HUB_ENV[0], "")
        pw = os.environ.get(self.FUNDED_HUB_ENV[1], "")
        if user and pw:
            if not user.startswith("robot$"):
                log.error("%s must be a push-only Hub ROBOT (robot$…), not a "
                          "user login — refusing to hand a user login to a "
                          "payer pod", self.FUNDED_HUB_ENV[0])
                return None
            log.info("funded leg for %s uses the static funded robot %s "
                     "(no admin login to mint a per-pod one)", hotkey, user)
            return (("HIPPIUS_HUB_USERNAME", user),
                    ("HIPPIUS_HUB_PASSWORD", pw)), 0
        log.error("no credential for the funded leg of %s: set %s/%s (per-pod "
                  "robots) or %s/%s (a static push-only robot)", hotkey,
                  *self.HUB_ADMIN_ENV, *self.FUNDED_HUB_ENV)
        return None

    def _revoke_robot(self, pod) -> None:
        """Best-effort revoke of a ledgered pod's Hub robot (Harbor expiry is
        the backstop when this fails)."""
        robot_id = int(getattr(pod, "robot_id", 0) or 0)
        if not robot_id:
            return
        minter = self._hub_robots()
        if minter is None:
            log.error("cannot revoke Hub robot id=%d for pod %s (no operator Hub "
                      "login) — it expires on its own", robot_id, pod.instance_id)
            return
        try:
            minter.delete(robot_id)
        except Exception as e:  # noqa: BLE001 — expiry backstop
            log.error("Hub robot id=%d for pod %s NOT revoked (%s) — it expires "
                      "on its own", robot_id, pod.instance_id, e)

    # ── completed legs survive a trainer restart ─────────────────────────────
    # Each finished leg is written to work_root/<seed>/<size>/completed_legs/
    # and a retry of the same job reuses it instead of dispatching (training
    # is deterministic; the checkpoint is already on the Hub) — a restart must
    # never retrain a finished leg and re-bill its payer.

    def _completed_leg_path(self, round_id, size: str, role: str, hotkey: str,
                            suffix: str = "") -> Path:
        return (self.work_root / str(round_id) / (size or "primary") / "completed_legs"
                / f"{role}-{hotkey}{suffix}.json")

    def _persist_completed_leg(self, entry, *, round_id, contract, role: str,
                               hotkey: str, suffix: str = "",
                               warm_start_ckpt: str | None = "",
                               funded_by: str = "") -> None:
        """Best-effort: a write miss only costs the old retrain-on-restart.

        ``warm_start_ckpt`` is the init the leg trained from ("" = random):
        the record is reusable only for a job with the SAME init (see
        :meth:`_load_completed_leg`)."""
        import dataclasses

        try:
            if not isinstance(entry, TrainedEntry):
                return
            path = self._completed_leg_path(round_id, contract.arch_preset, role, hotkey, suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"contract_digest": contract_digest(contract),
                       "warm_start_ckpt": str(warm_start_ckpt or ""),
                       "entry": dataclasses.asdict(entry)}
            if funded_by:
                # DEC-CA-0055: who paid for a v2 king leg ("king" / "operator")
                payload["funded_by"] = str(funded_by)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
        except Exception as e:  # noqa: BLE001
            log.warning("could not persist completed %s leg for %s: %s", role, hotkey, e)

    def _discard_completed_leg(self, *, round_id, contract, role: str, hotkey: str,
                               suffix: str = "") -> bool:
        """Drop the persisted record of a leg that must be retrained (an entry
        trained outside its era window would otherwise be reused from the
        record on every re-admission)."""
        try:
            path = self._completed_leg_path(round_id, contract.arch_preset, role, hotkey, suffix)
            if path.exists():
                path.unlink()
                log.warning("completed %s leg record for %s discarded (%s)", role, hotkey[:12],
                            path.name)
                return True
        except Exception as e:  # noqa: BLE001
            log.warning("could not discard completed %s leg for %s: %s", role, hotkey, e)
        return False

    def _load_completed_leg(self, *, round_id, contract, role: str, hotkey: str,
                            gen_ref: str, suffix: str = "",
                            warm_start_ckpt: str | None = None):
        """The persisted entry for EXACTLY this job (same contract digest, same
        generator ref — and, when ``warm_start_ckpt`` is given, the same init),
        else None.

        The init is part of the job: a seed can recur under a different init
        (another member of the promoted set), and a leg trained from one must
        never be reused for the other. ``None`` skips the check; a record
        without the field counts as random init ("")."""
        from .remote import receipt_to_entry

        path = self._completed_leg_path(round_id, contract.arch_preset, role, hotkey, suffix)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            log.warning("completed-leg record %s unreadable (%s); retraining", path, e)
            return None
        try:
            if str(raw.get("contract_digest")) != contract_digest(contract):
                return None
            entry = receipt_to_entry(dict(raw["entry"]))
        except Exception as e:  # noqa: BLE001
            log.warning("completed-leg record %s malformed (%s); retraining", path, e)
            return None
        if entry.gen_ref != gen_ref or entry.role != role or entry.miner_hotkey != hotkey:
            return None
        if warm_start_ckpt is not None:
            recorded = str(raw.get("warm_start_ckpt", "") or "")
            if recorded != str(warm_start_ckpt or ""):
                log.warning("completed-leg record %s was trained from init %r; this job's "
                            "init is %r — not reusable, retraining", path,
                            recorded or "<random init>", warm_start_ckpt or "<random init>")
                return None
        return entry

    def _push_deployed_chain_toml(self, host):
        """Copy the orchestrator's deployed chain.toml onto a pod the trainer
        rented itself and return the host pinned to it.

        Mirrors the provisioner's post-gate config push for gated lanes. A
        rented pod never sees that hook, so without this it would run its
        image-baked snapshot (pre-release ``train_image_digest`` pin) and the
        worker would refuse the final. No-op (host unchanged) when
        ``deployed_chain_toml`` is unset; a failed copy raises so the leg is
        an infra failure, never a silent run on the wrong contract."""
        import shlex
        import subprocess
        from dataclasses import replace as dc_replace

        src = getattr(self, "deployed_chain_toml", None)
        if not src:
            return host
        from .remote import build_scp_argv, build_ssh_argv

        dest = f"{host.workdir}/chain.deployed.toml"
        runner = getattr(self, "_scp_runner", None) or (
            lambda argv: subprocess.run(argv, check=True, capture_output=True, timeout=120))
        runner(build_ssh_argv(host, f"mkdir -p {shlex.quote(host.workdir)}"))
        runner(build_scp_argv(host, str(src), dest))
        log.info("pushed deployed chain.toml → %s:%s", host.name, dest)
        return dc_replace(host, chain_toml=dest)

    def _funded_pod_profile(self):
        """The pod profile every rented pod mirrors: the first ``final``/``any``
        host on file (a profile-only ``[[host]]`` entry is enough), else the
        first host of any stage. Rented pods must be interchangeable with the
        operator's (same image, paths, ssh identity) — the validator's
        gpu_name pairing and the audit both assume one runtime.

        Like :meth:`_rolling_final_hosts`, hosts.toml is re-read on every call
        when the trainer runs from a file — a profile edited (or first written)
        after startup is mirrored without a restart; an unreadable file falls
        back to the startup snapshot."""
        from .remote import RemoteDispatchError, load_hosts

        hosts: list = []
        if getattr(self, "remote_hosts_path", None) is not None:
            try:
                hosts = list(load_hosts(self.remote_hosts_path))
            except RemoteDispatchError:
                hosts = []
        hosts = hosts or list(self.remote_hosts or [])
        hosts = [h for h in hosts if getattr(h, "stage", "any") in ("any", "final")] or hosts
        if not hosts:
            raise RuntimeError("funded_pods=rent needs at least one operator "
                               "final host to mirror (none configured)")
        return hosts[0]

    def _funded_ledger_path(self) -> Path:
        return self.work_root / "funded_pods.json"

    def _load_funded_ledger(self) -> list:
        from ..shared.atomic import read_text_with_backup

        rows, source = read_text_with_backup(self._funded_ledger_path(), _parse_funded_ledger)
        if source == "absent":
            return []
        if rows is None:
            raise RuntimeError("funded pod ledger unreadable; refusing to overwrite billing evidence")
        if source == "backup":
            log.error("funded pod ledger recovered from backup; inspect original billing evidence")
        return rows

    def _save_funded_ledger(self, pods: list) -> None:
        from dataclasses import asdict

        from ..shared.atomic import read_text_with_backup, write_text_atomic

        # If recovery used the backup, copying the corrupt primary over it
        # would destroy the only good ledger before the new rename completes.
        _, source = read_text_with_backup(self._funded_ledger_path(), _parse_funded_ledger)
        if source == "unreadable":
            raise RuntimeError("funded pod ledger unreadable; refusing to overwrite billing evidence")
        try:
            write_text_atomic(self._funded_ledger_path(),
                              json.dumps([asdict(x) for x in pods], indent=2),
                              backup=source == "main")
        except OSError as e:
            from .box_health import note_disk_error

            note_disk_error(self.work_root, e, "funded pod ledger")   # halts rents (R5)
            raise

    def _ledger_add(self, pod) -> None:
        with self._funded_ledger_lock:
            pods = self._load_funded_ledger()
            pods = [x for x in pods if x.instance_id != pod.instance_id] + [pod]
            self._save_funded_ledger(pods)

    def _ledger_remove(self, instance_id: str) -> None:
        with self._funded_ledger_lock:
            self._save_funded_ledger(
                [x for x in self._load_funded_ledger()
                 if x.instance_id != instance_id])

    def _reconcile_funded_pods(self, *, keep_round_ids: tuple[str, ...] = (),
                               keep_payers: set[str] | frozenset[str] = frozenset()) -> None:
        """Startup sweep: tear down ledgered leftovers, then the per-payer
        orphan sweep (crash between launch and ledger). Best-effort — a payer
        API hiccup must never block the scheduler.

        ``keep_round_ids`` — the eras whose operator king pods stay (a restart
        re-attaches to them, and a complete checkpoint there skips the
        retrain); ``keep_payers`` — hotkeys with a leg in flight, whose payer
        pod or operator-rented leg pod stays for the same reason."""
        if self.cfg.round.funded_pods != "rent":
            return
        from ..provision.funded import reconcile_funded, teardown_funded

        keep_ids = tuple(x for x in keep_round_ids if x)
        try:
            pods = self._load_funded_ledger()
            # Operator-billed ledger pods (payer_hotkey = "") never go through
            # the per-payer path — it would demand a vault key that rightly
            # does not exist — and are swept BEFORE the vault check, so a
            # mis-set vault never leaves operator pods billing.
            for pod in [x for x in pods if not x.payer_hotkey]:
                if any(self._is_king_pod_of(pod.instance_id, rid) for rid in keep_ids):
                    log.info("funded sweep: keeping %s — an era king pod (a restart "
                             "re-attaches to it)", pod.instance_id)
                    continue
                if keep_payers and self._is_operator_leg_pod_of(pod.instance_id, keep_payers):
                    log.info("funded sweep: keeping %s — an operator-billed leg still "
                             "in flight (re-attaches after a restart)", pod.instance_id)
                    continue
                if self._sweep_keeps_pod(self._operator_pod_provider(pod), pod.instance_id,
                                         what="operator ledger pod"):
                    continue
                self._teardown_operator_pod(pod)
            try:
                self._sweep_operator_orphans(keep_ids, keep_payers)
            except Exception as e:  # noqa: BLE001 — never blocks the payer sweep below
                log.warning("funded sweep: operator-account orphan listing failed: %s", e)
            vault = self._payer_vault()
            if vault is None:
                if any(x.payer_hotkey for x in pods):
                    log.error("funded ledger holds payer pods but [round] "
                              "payer_vault_dir is unset — they bill their "
                              "miners until the vault comes back")
                return
            pods = [x for x in pods if x.payer_hotkey and x.payer_hotkey not in keep_payers]
            # A pod whose leg left the queue can still hold a live worker or a
            # finished, unharvested checkpoint (2026-10-31: three restarts cut
            # live legs) — probed per pod; kept pods stay on the ledger.
            kept = [x for x in pods
                    if self._sweep_keeps_pod(self._payer_pod_provider(vault, x), x.instance_id,
                                             what=f"payer {x.payer_hotkey[:12]} ledger pod")]
            kept_ids = {x.instance_id for x in kept}
            pods = [x for x in pods if x.instance_id not in kept_ids]
            for pod in pods:
                self._revoke_robot(pod)
            if pods:
                leftovers = teardown_funded(pods, vault)
                gone = {x.instance_id for x in pods} - {x.instance_id for x in leftovers}
                from .miner_spend_guard import spend_end

                for x in pods:
                    if x.instance_id in gone:      # the sweep's teardown (review S1)
                        spend_end(self, x.payer_hotkey, x.instance_id, outcome="waste",
                                  reason="swept without a harvested result")
                with self._funded_ledger_lock:
                    # Only CONFIRMED-gone rows leave the ledger: a kept pod's
                    # row must survive, or the listing sweep below would read
                    # it as an off-ledger orphan.
                    self._save_funded_ledger(
                        [x for x in self._load_funded_ledger() if x.instance_id not in gone])
                for inst in leftovers:
                    log.error("funded pod %s (payer %s) could not be confirmed "
                              "gone — still billing the miner; kept on the "
                              "ledger for the next sweep",
                              inst.instance_id, inst.payer_hotkey)
            # Orphans of every payer without an in-flight leg.
            killed = reconcile_funded(self._load_funded_ledger(), vault,
                                      netuid=self.cfg.subnet.netuid, exclude=keep_payers,
                                      keep=lambda adapter, pod_id, hotkey: self._sweep_keeps_pod(
                                          adapter, pod_id,
                                          what=f"payer {hotkey[:12]} off-ledger pod"))
            from .miner_spend_guard import close_vanished, spend_end

            for name in killed or ():
                spend_end(self, "", str(name), outcome="waste", reason="off-ledger pod swept")
            close_vanished(self)                     # open records of pods long gone
        except Exception as e:  # noqa: BLE001
            log.warning("funded pod reconcile failed (ignored): %s", e)

    # A rent ledgers its pod right after the create; younger unledgered pods
    # may be that window, never an orphan.
    OPERATOR_ORPHAN_GRACE_SECONDS = 900.0

    def _sweep_operator_orphans(self, keep_ids: tuple[str, ...],
                                keep_payers: set[str] | frozenset[str]) -> None:
        """Tear down THIS deployment's pods on the operator's Lium account that
        no ledger row tracks — listing-based, like the per-payer
        :func:`reconcile_funded`. The ledger sweep alone cannot see a pod that
        left the ledger: 2026-10-29 a released 8xH200 king pod dropped out of
        ``lium ps`` right after ``lium rm`` (so the verified teardown removed
        it from the ledger), came back BROKEN and billed for 6 h. Scoped to
        ``cascade-n<netuid>-…-funded-…`` names; current era king pods and
        in-flight operator legs are kept, as in the ledger sweep."""
        import datetime as _dt

        from ..provision.core import LiumProvider
        from ..provision.funded import terminate_verified

        prefix = f"cascade-n{int(self.cfg.subnet.netuid)}-"
        provider = LiumProvider()
        ledgered = {p.instance_id for p in self._load_funded_ledger()}
        now = _dt.datetime.now(_dt.UTC)
        for row in provider._list_pods():
            name = str(row.get("name") or "")
            if not name.startswith(prefix) or "-funded-" not in name or name in ledgered:
                continue
            if any(self._is_king_pod_of(name, rid) for rid in keep_ids):
                continue
            if keep_payers and self._is_operator_leg_pod_of(name, keep_payers):
                continue
            try:
                created = _dt.datetime.fromisoformat(str(row.get("created_at")))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=_dt.UTC)
                if (now - created).total_seconds() < self.OPERATOR_ORPHAN_GRACE_SECONDS:
                    continue
            except (TypeError, ValueError):
                pass                       # no usable age: an off-ledger pod is still an orphan
            if self._sweep_keeps_pod(provider, name, what="operator off-ledger pod"):
                continue
            gone = terminate_verified(provider, name)
            log.error("funded sweep: operator pod %s (%s) was on no ledger — %s", name,
                      row.get("status") or "?", "terminated" if gone
                      else "terminate NOT confirmed; still billing, retried next sweep")

    # ── the sweep's per-pod guard (2026-10-31 incidents) ─────────────────────
    # Restarts at 20:52, 23:13 and 23:34 swept live funded pods mid-training —
    # once a pod holding a FINISHED but unharvested checkpoint — because the
    # sweep trusted the queue's keep-set alone. The pod itself is now asked.

    # An unreachable pod is kept this long from the first sweep that could not
    # reach it (unreachable ≠ dead: the leg's own poll rides out transport
    # blips the same way). Persisted (sweep_unreachable.json beside the
    # ledger): repeated restarts must not reset the grace forever.
    SWEEP_UNREACHABLE_GRACE_SECONDS = 1800.0
    # A finished exit-0 run with no harvested marker keeps its pod this long
    # after it exited; past it nothing will harvest it (the re-attach pass
    # already ran) and the pod is released, loudly.
    SWEEP_UNHARVESTED_KEEP_SECONDS = 6 * 3600.0
    SWEEP_PROBE_TIMEOUT_SECONDS = 60
    POD_HARVEST_RECORDS_MAX = 512

    def _pod_harvests_path(self) -> Path:
        return Path(self.work_root) / "pod_harvests.json"

    def _pod_harvests(self) -> dict:
        """``"ip:port" -> wall time of the last verified harvest`` from that
        endpoint (persisted beside the funded ledger, so it survives the very
        restart the sweep guards)."""
        cache = self.__dict__.get("_pod_harvest_records")
        if cache is not None:
            return cache
        from ..shared.atomic import read_text_with_backup

        def parse(text):
            raw = json.loads(text)
            return {str(k): float(v) for k, v in dict(raw).items()}

        cache = {}
        try:
            rows, _ = read_text_with_backup(self._pod_harvests_path(), parse)
            cache.update(rows or {})
        except Exception as e:  # noqa: BLE001 — empty = every finished run is kept
            log.warning("pod harvest records unreadable (%s) — finished runs are kept", e)
        self.__dict__["_pod_harvest_records"] = cache
        return cache

    def _record_pod_harvest(self, ip: str, port: int) -> None:
        if not ip or getattr(self, "work_root", None) is None:
            return
        from ..shared.atomic import write_text_atomic

        clock = getattr(self, "_sweep_clock", None) or time.time
        lock = self.__dict__.setdefault("_pod_harvest_lock", threading.Lock())
        with lock:
            recs = self._pod_harvests()
            recs[f"{ip}:{int(port or 22)}"] = float(clock())
            for k in sorted(recs, key=recs.get)[:max(0, len(recs) - self.POD_HARVEST_RECORDS_MAX)]:
                recs.pop(k, None)
            write_text_atomic(self._pod_harvests_path(), json.dumps(recs, indent=0), backup=True)

    SWEEP_UNREACHABLE_FORGET_SECONDS = 7 * 86400.0

    def _sweep_unreachable_path(self) -> Path:
        return Path(self.work_root) / "sweep_unreachable.json"

    def _sweep_unreachable(self) -> dict:
        """``pod id -> wall time the sweep first found it unreachable``,
        loaded once per process from the persisted file."""
        cache = self.__dict__.get("_sweep_unreachable_since")
        if cache is not None:
            return cache
        cache = {}
        if getattr(self, "work_root", None) is not None:
            from ..shared.atomic import read_text_with_backup

            def parse(text):
                return {str(k): float(v) for k, v in dict(json.loads(text)).items()}

            try:
                rows, _ = read_text_with_backup(self._sweep_unreachable_path(), parse)
                cache.update(rows or {})
            except Exception as e:  # noqa: BLE001 — empty = a fresh grace (keeps)
                log.warning("sweep unreachable records unreadable (%s) — graces restart", e)
        self.__dict__["_sweep_unreachable_since"] = cache
        return cache

    def _sweep_unreachable_update(self, pod_id: str, now: float, *, down: bool) -> float:
        """Record (``down``) or clear ``pod_id``'s unreachable-since stamp and
        persist the change; returns the stamp (``now`` when cleared)."""
        lock = self.__dict__.setdefault("_sweep_unreachable_lock", threading.Lock())
        with lock:
            recs = self._sweep_unreachable()
            before = dict(recs)
            if down:
                since = recs.setdefault(str(pod_id), float(now))
            else:
                since = float(now)
                recs.pop(str(pod_id), None)
            for k in [k for k, v in recs.items() if now - v > self.SWEEP_UNREACHABLE_FORGET_SECONDS]:
                recs.pop(k, None)              # long-gone pods never pile up
            if recs != before and getattr(self, "work_root", None) is not None:
                from ..shared.atomic import write_text_atomic

                try:
                    write_text_atomic(self._sweep_unreachable_path(),
                                      json.dumps(recs, indent=0, sort_keys=True), backup=True)
                except OSError as e:
                    log.warning("sweep unreachable records not persisted: %s", e)
            return since

    def _sweep_finished_path(self) -> Path:
        return Path(self.work_root) / "sweep_finished_unharvested.json"

    def _sweep_finished(self, pod_id: str, now: float, *, seen: bool | None) -> float | None:
        """``pod_id``'s persisted "last seen: exited 0, unharvested" stamp
        (money safety II). ``seen`` True records it (first sighting kept),
        False clears it, None only reads. Returns the stamp or None."""
        if getattr(self, "work_root", None) is None:
            return None
        from ..shared.atomic import read_text_with_backup, write_text_atomic

        lock = self.__dict__.setdefault("_sweep_finished_lock", threading.Lock())
        with lock:
            recs = self.__dict__.get("_sweep_finished_recs")
            if recs is None:
                recs = {}
                try:
                    rows, _ = read_text_with_backup(
                        self._sweep_finished_path(),
                        lambda t: {str(k): float(v) for k, v in dict(json.loads(t)).items()})
                    recs.update(rows or {})
                except Exception as e:  # noqa: BLE001 — empty: a fresh record
                    log.debug("sweep finished-run records unreadable (%s)", e)
                self.__dict__["_sweep_finished_recs"] = recs
            before = dict(recs)
            if seen is True:
                recs.setdefault(str(pod_id), float(now))
            elif seen is False:
                recs.pop(str(pod_id), None)
            for k in [k for k, v in recs.items() if now - v > self.SWEEP_UNREACHABLE_FORGET_SECONDS]:
                recs.pop(k, None)
            if recs != before:
                try:
                    write_text_atomic(self._sweep_finished_path(),
                                      json.dumps(recs, indent=0, sort_keys=True), backup=True)
                except OSError as e:
                    log.warning("sweep finished-run records not persisted: %s", e)
            return recs.get(str(pod_id))

    def _operator_pod_provider(self, pod):
        """The provider ``_teardown_operator_pod`` would terminate ``pod`` on."""
        from ..provision.core import LiumProvider
        from ..provision.operator_pool import OperatorPool

        return OperatorPool(self) if pod.provider == OperatorPool.name else LiumProvider()

    def _payer_pod_liveness(self, pod_id: str):
        """``(address | None, definitive)`` for this leg's ledgered payer pod,
        asked through ITS payer provider (bounded retries: a blip is never
        "gone"). Unknown when the pod, its key or the probe is unavailable."""
        from .runner_common import probe_pod_live

        try:
            pod = next((p for p in self._load_funded_ledger() if p.instance_id == pod_id), None)
            vault = self._payer_vault() if pod is not None else None
            provider = self._payer_pod_provider(vault, pod) if vault is not None else None
            live = getattr(provider, "live_pod_address", None)
        except Exception as e:  # noqa: BLE001 — unknown, never "gone"
            log.warning("funded leg: liveness of %s unknown (%s)", pod_id, e)
            return None, False
        if not callable(live):
            return None, False
        addr, definitive = probe_pod_live(live, pod_id,
                                          sleep=getattr(self, "_rent_wait_sleep", None))
        if addr is not None and pod is not None:
            from .miner_spend_guard import spend_touch

            spend_touch(self, pod.payer_hotkey, pod_id)        # seen alive
        return addr, definitive

    def _payer_pod_provider(self, vault, pod):
        """``pod``'s payer-account adapter (its vaulted key), or None."""
        try:
            from ..funding.providers import pod_credential
            from ..provision.funded import lium_provider_for_key, provider_for_key

            cred = pod_credential(vault, pod)
            if cred is None or not cred.api_key:
                return None
            factory = self.__dict__.get("_funded_provider_factory")
            if factory is not None and factory is not lium_provider_for_key:
                return factory(cred.api_key)
            return provider_for_key(pod.provider, cred.api_key)
        except Exception as e:  # noqa: BLE001 — unknown, the guard decides
            log.warning("funded sweep: no payer adapter for %s (%s)", pod.instance_id, e)
            return None

    def _sweep_probe_host(self, ip: str, port: int):
        """A read-only probe identity for a pod the sweep has no pinned key
        for: the rented-pod profile's user/key/workdir, a throwaway
        known_hosts (a reused ip:port must never fail the probe on a stale
        key — that would read as "unreachable")."""
        from .remote import RemoteHost

        profile = self._funded_pod_profile()
        return RemoteHost(
            name="sweep-probe", host=str(ip), port=int(port), user=profile.user,
            key_path=profile.key_path, workdir=profile.workdir,
            ssh_options=("UserKnownHostsFile=/dev/null", "ConnectTimeout=15",
                         "LogLevel=ERROR", *profile.ssh_options))

    def _sweep_keeps_pod(self, adapter, pod_id: str, *, what: str) -> bool:
        """Whether the sweep must KEEP ``pod_id`` — asked of the pod itself:

        - a detached worker still running (run dir pid answers ``kill -0``);
        - a run that finished with exit code 0 and was never harvested, for
          ``SWEEP_UNHARVESTED_KEEP_SECONDS`` after it exited;
        - unreachable (no address, ssh failed) for less than
          ``SWEEP_UNREACHABLE_GRACE_SECONDS`` since first seen so;
        - anything the guard cannot decide (no adapter, no profile, a crash).

        A kept pod costs $/h; a cut pod costs the whole leg — every keep is
        logged at ERROR so an operator sees the bill."""
        clock = getattr(self, "_sweep_clock", None) or time.time
        now = float(clock())
        try:
            if adapter is None:
                log.error("funded sweep: KEEPING %s %s — no provider adapter to locate it "
                          "(cannot verify it is idle)", what, pod_id)
                return True
            try:
                addr = adapter.get_ip(pod_id)
                tagged = getattr(adapter, "list_tagged", None)
                if addr is None and callable(tagged) and str(pod_id).startswith("cascade-"):
                    # A write-ahead name (Shadeform assigns its id after create)
                    for pid in tagged(pod_id):
                        if pid != pod_id:
                            addr = adapter.get_ip(pid)
                            if addr is not None:
                                break
            except Exception as e:  # noqa: BLE001 — a listing failure is "unreachable"
                log.warning("funded sweep: %s %s address lookup failed: %s", what, pod_id, e)
                addr = None
            probe = None
            if addr is not None and getattr(addr, "ip", None):
                from .remote import probe_pod_runs

                host = self._sweep_probe_host(addr.ip, getattr(addr, "ssh_port", 22) or 22)
                probe = probe_pod_runs(host, runner=getattr(self, "_sweep_ssh_runner", None),
                                       timeout=self.SWEEP_PROBE_TIMEOUT_SECONDS)
            if probe is None or not probe.reachable:
                since = self._sweep_unreachable_update(pod_id, now, down=True)
                down = now - since
                why = "no ssh address" if probe is None else (probe.error or "ssh failed")
                if down < self.SWEEP_UNREACHABLE_GRACE_SECONDS:
                    log.error("funded sweep: KEEPING %s %s — unreachable (%s) for %.0fs, "
                              "inside the %.0fs grace (unreachable is not dead)", what, pod_id,
                              why, down, self.SWEEP_UNREACHABLE_GRACE_SECONDS)
                    return True
                # Money safety II: its last observed state was a FINISHED,
                # unharvested run (exit 0): an unreachable pod may still hold
                # the only copy of a paid run — kept until the priced keep
                # (half the run's cost, capped) is spent, never 30 min.
                finished = self._sweep_finished(pod_id, now, seen=None)
                if finished is not None:
                    from .harvest_guard import keep_budget_seconds

                    budget = keep_budget_seconds(self)
                    if now - finished < budget:
                        log.error("funded sweep: KEEPING %s %s — unreachable (%s) but last seen "
                                  "with a FINISHED unharvested run %.0fs ago; kept for the "
                                  "priced keep (%.0fs)", what, pod_id, why, now - finished,
                                  budget)
                        return True
                    self._sweep_finished(pod_id, now, seen=False)
                log.error("funded sweep: %s %s unreachable (%s) for %.0fs, past the %.0fs "
                          "grace — releasing", what, pod_id, why, down,
                          self.SWEEP_UNREACHABLE_GRACE_SECONDS)
                self._sweep_unreachable_update(pod_id, now, down=False)
                return False
            self._sweep_unreachable_update(pod_id, now, down=False)
            if probe.live:
                log.error("funded sweep: KEEPING %s %s — %d live detached worker(s) (%s); "
                          "its leg is not in flight here — check it", what, pod_id,
                          len(probe.live), ", ".join(probe.live))
                return True
            # A finished exit-0 run is harvested iff this endpoint was harvested
            # AFTER the run exited (exit time from the pod's own age, so no
            # clock skew; a reused ip:port's older record never counts).
            harvested_at = self._pod_harvests().get(
                f"{addr.ip}:{int(getattr(addr, 'ssh_port', 22) or 22)}", float("-inf"))
            unharvested = [(d, age) for d, age in probe.finished_ok
                           if harvested_at < now - age]
            fresh = [(d, age) for d, age in unharvested
                     if age < self.SWEEP_UNHARVESTED_KEEP_SECONDS]
            self._sweep_finished(pod_id, now, seen=bool(fresh))
            if fresh and _kept_past_budget(self, pod_id, now):
                # A pod WE kept after our own harvest fault: bounded at
                # [round] harvest_keep_max_seconds, not the 6 h restart grace —
                # it bills its payer (review F6). Released with a priced alert.
                return False
            if fresh:
                log.error("funded sweep: KEEPING %s %s — finished run(s) with exit code 0 "
                          "NOT harvested (%s); harvest it or it is released %.0fs after "
                          "exit", what, pod_id,
                          ", ".join(f"{d} {age:.0f}s ago" for d, age in fresh),
                          self.SWEEP_UNHARVESTED_KEEP_SECONDS)
                return True
            if unharvested:
                log.error("funded sweep: %s %s — unharvested run(s) older than %.0fs (%s); "
                          "releasing", what, pod_id, self.SWEEP_UNHARVESTED_KEEP_SECONDS,
                          ", ".join(d for d, _ in unharvested))
            return False
        except Exception as e:  # noqa: BLE001 — unsure ⇒ keep
            log.error("funded sweep: KEEPING %s %s — guard failed (%s)", what, pod_id, e)
            return True

    def _is_king_pod_of(self, instance_id: str, round_id: str) -> bool:
        prefix = king_pod_name_prefix(self.cfg.subnet.netuid, str(round_id))
        return str(instance_id).startswith(prefix + "-")

    def _record_funded_failure(self, hotkey: str, msg: str, *, miner_fault: bool,
                               error_class: str, burn: bool) -> None:
        self._funded_leg_failures[hotkey] = (error_tail(msg, 500), miner_fault,
                                             error_class, burn)

    def _count_funded_loss(self, hotkey: str, *, more: bool) -> bool:
        """Count one billed allocation that delivered nothing (a lemon, a
        stale or slow pod released before training) as an operator-side loss
        on the entry; return whether the leg may rent a replacement.

        Without operator make-good capacity the leg wrapper counts ONE loss
        per failed leg attempt however many pods it burned, so a systemic
        fault (a broken image, a market of lemons) billed the miner
        ``FUNDED_MAX_STALE_PODS`` pods per attempt — nine before the
        ``OPERATOR_FAULT_RETRIES`` pause. Every replaced pod now counts, and a
        replacement is rented only while the losses stay under the cap, so a
        systemic fault bills at most ``OPERATOR_FAULT_RETRIES`` allocations in
        total. ``more`` is the caller's own verdict on renting again. While a
        replacement is allowed the flag is cleared so the loop continues on
        the miner's key; otherwise it stays set and the wrapper's flagging at
        the leg's end is a no-op (no double count). With operator capacity
        configured the wrapper's accounting is unchanged.
        """
        queue = self._funded_queue()
        if queue is None or self._operator_recovery_available(hotkey):
            return more
        if not queue.set_operator_billed(hotkey, True):
            return more
        faults = int(getattr(queue.get(hotkey), "operator_faults", 0) or 0)
        allowed = bool(more) and faults < self.OPERATOR_FAULT_RETRIES
        if allowed:
            queue.set_operator_billed(hotkey, False)
        else:
            log.warning("funded leg %s: operator-side loss %d/%d — no further pod on the "
                        "miner's key this attempt", hotkey[:12], faults, self.OPERATOR_FAULT_RETRIES)
        return allowed

    def _route_lost_run_to_operator(self, round_id: str, gen) -> None:
        """A finished run of ``gen`` was lost on OUR side (money safety II):
        its retrain is an operator make-good — gated by the owner's approval
        (``make_good`` with the miner's loss priced in the request), then an
        operator lane or an operator-rented pod. Without operator capacity it
        waits (``operator_blocked``): the miner's key is never charged again.
        Always raises (a routing exception or :class:`_FundedLegSkip`)."""
        from ..funding.queue import OPERATOR_BLOCKED_CLASS
        from .box_health import miner_lost_usd
        from .money_guard import alert, gate_or_adopt

        lost = miner_lost_usd(self)
        if not self._operator_recovery_available(gen.hotkey):
            alert(self, "lost_run_needs_operator", gen.hotkey[:16],
                  f"{gen.hotkey[:12]}: a finished run (≤ ${lost:.2f} of the miner's money) was "
                  "lost on our side and its retrain needs OPERATOR capacity (an operator lane "
                  "or funded_operator_fallback) — it waits; the miner's key is not charged",
                  severity="critical")
            self._record_funded_failure(
                gen.hotkey, "your finished run was lost on the operator's side; its retrain "
                "runs on the operator's account once capacity is configured — your key is not "
                "charged again", miner_fault=False, error_class=OPERATOR_BLOCKED_CLASS,
                burn=False)
            raise _FundedLegSkip(gen.hotkey)
        lanes = self._operator_fallback_lanes()
        gate_or_adopt(self, round_id, gen, kind="make_good", rent=not lanes, lost_usd=lost)
        # Approved: from here on the entry is operator-billed (the next try
        # routes through the make-good branch, never the payer's key).
        q = self._funded_queue()
        if q is not None:
            with contextlib.suppress(Exception):
                q.set_operator_billed(gen.hotkey, True)
        from .box_health import BoxHealth

        with contextlib.suppress(Exception):
            BoxHealth.for_runner(self).forget_leg(gen.hotkey)
        log.warning("final challenger %s: retrain after a LOST finished run — operator-billed "
                    "make-good (%s), never on the miner's key", gen.hotkey,
                    "operator lane" if lanes else "operator-rented pod")
        if lanes:
            raise _FundedOperatorFallback(gen.hotkey)
        raise _FundedOperatorRent(gen.hotkey)

    def _rent_funded_host(self, round_id: str, gen: ResolvedGenerator, *, gpu_choice=None):
        """:meth:`_rent_funded_host_unguarded`; an own-key KING rent is
        registered while it lasts so challengers yield it the marketplace
        (``_king_pending``), as they do the operator king's rent."""
        if not getattr(self.__dict__.get("_leg_local_obj"), "funded_is_king", False):
            return _FundedPodMixin._rent_funded_host_unguarded(self, round_id, gen, gpu_choice=gpu_choice)
        from .runner_funded_legs import king_key_renting

        with king_key_renting(self):
            return _FundedPodMixin._rent_funded_host_unguarded(self, round_id, gen, gpu_choice=gpu_choice)

    def _rent_funded_host_unguarded(self, round_id: str, gen: ResolvedGenerator, *,
                                    gpu_choice=None):
        """Rent ``gen``'s leg pod on ITS payer's key → ``(RemoteHost, PodInstance)``.

        Every failure records a verdict (``_funded_leg_failures``) for the
        queue and raises ``_FundedLegSkip`` so the leg is dropped.

        DEC-CA-0060: ``gpu_choice`` (a ``GpuChoice``; None = read it from the
        hotkey's IN-FLIGHT lane-A queue entry for this ref) names the GPU type
        the miner chose. That type is rented under the miner's own $/h cap,
        never the operator caps; with no capacity the leg waits for it until
        its latest safe start and then requeues as a capacity miss (no burn,
        nothing rented) — or, with ``allow_fallback``, rents the default
        cheapest fitting type at once. No choice = the default path,
        unchanged. The token budget and the wall are the contract's either
        way (DEC-CA-0001).
        """
        from ..provision.core import record_host_quarantine
        from ..provision.funded import LEMON_CLASS, rent_funded_pod
        from ..provision.layout import visible_devices
        from .host_preflight import host_blocked_reason, sync_host_blocklist
        from .remote import RemoteHost

        # [round] funded_host_blocklist → every provider listing this rent makes.
        sync_host_blocklist(self)
        queue = self._funded_queue()
        flagged = queue.get(gen.hotkey) if queue is not None else None
        if (flagged is not None and bool(getattr(flagged, "operator_billed", False))
                and not self._operator_recovery_available(gen.hotkey)):
            # No operator capacity is configured, so a make-good can never run.
            # The leg retries on the miner's key with a new pod; a prior
            # allocation not confirmed released still blocks the rent (ledger
            # and create-intent guards), so nothing is billed twice. A cause
            # that keeps recurring (a broken image, our credential setup) must
            # not keep billing the miner: after the cap the entry pauses.
            faults = int(getattr(flagged, "operator_faults", 0) or 0)
            from .miner_spend_guard import capped

            if capped(self, gen):
                # A make-good approved after the miner waste cap: NEVER again
                # on the miner's key — it waits for operator capacity.
                self._record_funded_failure(
                    gen.hotkey, "the operator make-good after the waste cap needs operator "
                    "capacity; your key is not charged again", miner_fault=False,
                    error_class="operator_blocked", burn=False)
                raise _FundedLegSkip(gen.hotkey)
            if faults >= self.OPERATOR_FAULT_RETRIES:
                self._record_funded_failure(
                    gen.hotkey, f"paused after {faults} operator-side failures; the operator "
                    "must fix the cause before this leg rents again (not the miner's fault, "
                    "nothing burned)", miner_fault=False, error_class="operator_blocked",
                    burn=False)
                raise _FundedLegSkip(gen.hotkey)
            log.warning("final challenger %s: operator-side failure %d/%d and no operator "
                        "make-good configured; retrying on the miner's key with a new pod",
                        gen.hotkey, faults, self.OPERATOR_FAULT_RETRIES)
            queue.set_operator_billed(gen.hotkey, False)
            flagged = queue.get(gen.hotkey)
        if flagged is not None and bool(getattr(flagged, "operator_billed", False)):
            # Owner make-good: this entry's leg is on the operator's bill — an
            # operator lane (or an operator-rented pod), never the payer's key;
            # a miss requeues the leg unburned like any lane fault.
            from .money_guard import gate_or_adopt

            lanes = self._operator_fallback_lanes()
            if not lanes and self._operator_rent_allowed(gen.hotkey):
                # Owner 2026-11-02: never on our account without the owner's
                # approval of THIS leg (parks awaiting_operator_approval); a
                # leg already holding its operator pod is adopted (F1).
                gate_or_adopt(self, round_id, gen, kind="make_good", rent=True)
                log.warning("final challenger %s: entry is OPERATOR-BILLED — no operator "
                            "lane on file, renting a pod on the OPERATOR's account",
                            gen.hotkey)
                raise _FundedOperatorRent(gen.hotkey)
            if not lanes:
                self._record_funded_failure(
                    gen.hotkey, "operator-funded recovery is required; no operator capacity "
                    "is configured, so the miner's key will not be charged again",
                    miner_fault=False, error_class="operator_blocked", burn=False)
                raise _FundedLegSkip(gen.hotkey)
            gate_or_adopt(self, round_id, gen, kind="make_good", rent=False)
            log.warning("final challenger %s: entry is OPERATOR-BILLED (owner make-good) — "
                        "running on an operator lane (%d on file), never on the payer's key",
                        gen.hotkey, len(lanes))
            raise _FundedOperatorFallback(gen.hotkey)
        # A lane-A leg's choice lives on its in-flight queue entry and is
        # RE-READ while the leg waits for capacity (a re-fund may change it);
        # an explicit choice (king: the default; lane B / marketcast: their
        # own store) is fixed for the leg.
        reread_choice = gpu_choice is None
        if gpu_choice is None:
            gpu_choice = _entry_gpu_choice(flagged, gen.ref)
        flight_attempt = str(getattr(flagged, "attempt_id", "") or "")
        rnd = self.cfg.round
        from ..provision.funded import funded_pod_name
        from .cost_guards import measured_wall_refusal

        # A ledgered allocation of THIS leg (a restart, a retry after a
        # transport drop) is adopted, not rented: its run is re-attached, so
        # the rent-time guards below (measured walls, doomed start) must not
        # strand it — the progress watchdog judges the re-attached run.
        expected_id = funded_pod_name(str(round_id), gen.hotkey, self.cfg.subnet.netuid) + "-0"
        try:
            prior_alloc = any(p.instance_id == expected_id for p in self._load_funded_ledger())
        except Exception:  # noqa: BLE001 — unreadable ledger: the guards apply
            prior_alloc = False
        why = "" if prior_alloc else measured_wall_refusal(rnd)
        if why:
            # No measured wall for the pinned image: every latest safe start
            # and cost cap below is a guess (2026-10-31: 20x off). Nothing is
            # rented; requeued unburned as OUR fault.
            log.error("funded leg %s: %s", gen.hotkey[:12], why)
            from ..funding.queue import OPERATOR_BLOCKED_CLASS

            self._record_funded_failure(gen.hotkey, why, miner_fault=False,
                                        error_class=OPERATOR_BLOCKED_CLASS, burn=False)
            raise _FundedLegSkip(gen.hotkey)
        # No blind retrain (2026-11-02: every ENOSPC harvest re-rented on the
        # MINER's key and retrained, ~$640 lost). A leg an our-side fault hit
        # AFTER it trained either re-harvests its KEPT pod (adopted below,
        # once the box-health halt is clear) or — the finished run is gone —
        # retrains only with the owner's approval, even on the miner's key.
        from .box_health import BoxHealth, rent_halt_refusal
        from .runner_common import disk_guard_refusal

        lost_leg = BoxHealth.for_runner(self).leg(gen.hotkey) if queue is not None else None
        if lost_leg and (not lost_leg.get("trained") or lost_leg.get("ref", gen.ref) != gen.ref):
            lost_leg = None
        pod_live: bool | None = None
        if prior_alloc:
            # Is this leg's ledgered pod actually live and adoptable? A stale
            # write-ahead row must not bypass the halt / canary / reservation
            # (review N3), and a lost kept pod must be known BEFORE any rent.
            probe = getattr(self, "_payer_pod_liveness", None)
            addr, definitive = probe(expected_id) if callable(probe) else (None, False)
            pod_live = True if addr is not None else (False if definitive else None)
            if pod_live is False:
                log.warning("funded leg %s: ledgered pod %s is not live — a fresh rent, every "
                            "rent guard applies", gen.hotkey[:12], expected_id)
                prior_alloc = False
                # The lost pod's own rent closes at its LAST-SEEN-ALIVE time as
                # infra waste (a host loss) — before the fresh rent of the same
                # pod name can open (2026-11-03 5EvqPCMN).
                from .miner_spend_guard import spend_end

                spend_end(self, gen.hotkey, expected_id, outcome="waste", at_last_seen=True,
                          reason="lost: not live at re-attach (closed at its last-seen time)")
        if lost_leg is not None:
            from ..funding.queue import OPERATOR_DISK_CLASS, OPERATOR_TRANSIENT_CLASS

            if pod_live is True and lost_leg.get("kept"):
                # Re-harvest the kept pod: never held by the halt; it needs room
                # for its OWN checkpoint only (review N2).
                from .harvest_guard import _room
                from .runner_common import HARVEST_DIRNAME

                short = _room(self, Path(self.work_root) / HARVEST_DIRNAME,
                              f"re-harvest of {gen.hotkey[:12]}'s kept pod")
                if short:
                    self._record_funded_failure(
                        gen.hotkey, f"operator disk: no room yet ({short / 1e9:.1f} GB short) "
                        "to re-harvest the kept pod — retried, nothing rented",
                        miner_fault=False, error_class=OPERATOR_DISK_CLASS, burn=False)
                    raise _FundedLegSkip(gen.hotkey)
            elif pod_live is None and prior_alloc:
                # Liveness unknown (a provider blip): neither rent nor park.
                self._record_funded_failure(
                    gen.hotkey, "kept pod liveness unknown (provider blip) — re-attach "
                    "retried, nothing rented", miner_fault=False,
                    error_class=OPERATOR_TRANSIENT_CLASS, burn=False)
                raise _FundedLegSkip(gen.hotkey)
            else:
                # The finished run is gone (or its pod was never kept). Money
                # safety II: the retrain is OUR make-good — OPERATOR-billed
                # (an operator lane or an operator-rented pod), with the
                # owner's approval, NEVER on the miner's key again.
                self._route_lost_run_to_operator(round_id, gen)
        # Disk guard (2026-10-31): never start a NEW leg the box may not have
        # room to finish (harvest, ledgers, pending records); also the box
        # health halt and the pre-rent canary. An adopted allocation is a live
        # leg — it always keeps going.
        # Only a pod confirmed LIVE skips the guards: an unknown liveness (no
        # key / provider / probe) gets every guard too (review R6).
        adopt_live = bool(prior_alloc and pod_live is True)
        is_king = bool(getattr(self.__dict__.get("_leg_local_obj"), "funded_is_king", False))
        if is_king and not adopt_live:
            # An own-key KING leg held by a disk halt: RentsHalted, never a
            # _FundedLegSkip the era loop counts and rotates on (review D2).
            from .box_health import RentsHalted

            held = rent_halt_refusal(self, king=True)
            if held:
                raise RentsHalted(held)
        low = "" if adopt_live else disk_guard_refusal(self)
        if low:
            from ..funding.queue import OPERATOR_DISK_CLASS

            log.error("funded leg %s: %s", gen.hotkey[:12], low)
            self._record_funded_failure(gen.hotkey, low, miner_fault=False,
                                        error_class=OPERATOR_DISK_CLASS, burn=False)
            raise _FundedLegSkip(gen.hotkey)
        vault = self._payer_vault()
        if vault is None:
            self._record_funded_failure(
                gen.hotkey, "funded_pods=rent but [round] payer_vault_dir is "
                "unset (operator config)", miner_fault=False,
                error_class="infra", burn=False)
            raise _FundedLegSkip(gen.hotkey)
        api_key = vault.get(gen.hotkey)
        local = self.__dict__.get('_leg_local_obj')
        if local is not None and getattr(local, 'funded_is_king', False):
            import hashlib
            local.funded_key_digest = hashlib.sha256(api_key.encode()).hexdigest() if api_key else ''
        # A key that survives to a training leg must outlive that leg however
        # stale its intake was (the vault's own contract — PayerKeyVault.
        # refresh): re-stamp the TTL now so the after-training identity check,
        # the teardown and the bench can still act on the payer's account (a
        # key that aged out mid-leg reads as tamper at harvest and leaves the
        # pod unstoppable).
        if api_key and vault.refresh(gen.hotkey):
            log.info("final challenger %s: payer key TTL re-stamped for the leg",
                     gen.hotkey)
        if (api_key and not prior_alloc
                and float(getattr(rnd, "payer_projected_rate_per_hour", 0.0) or 0.0) > 0):
            # A key that cannot pay for the whole leg never starts it: the
            # miner would pay for a pod that dies mid-run (no attempt charged).
            from ..funding.queue import PAYER_BALANCE_CLASS
            from .money_guard import payer_balance_refusal

            # A miner-chosen type is projected at the miner's own cap (an
            # 8xH200 at $30+/h must not be under-projected at the knob) or,
            # uncapped, at the operator's funded ceiling.
            knob = float(getattr(rnd, "payer_projected_rate_per_hour", 0.0) or 0.0)
            chosen_rate = (float(getattr(gpu_choice, "max_price_per_hour", 0.0) or 0.0)
                           or float(getattr(rnd, "funded_max_price_per_hour", 0.0) or 0.0)
                           ) if getattr(gpu_choice, "gpu", "") else 0.0
            poor = payer_balance_refusal(self, api_key, hours=self._leg_wall_seconds() / 3600.0,
                                         rate=max(knob, chosen_rate))
            if poor:
                # nothing rented, requeued with a backoff — never terminal,
                # never a burn, never counted against any cap
                log.warning("funded leg %s: %s", gen.hotkey[:12], poor)
                self._record_funded_failure(gen.hotkey, poor, miner_fault=False,
                                            error_class=PAYER_BALANCE_CLASS, burn=False)
                raise _FundedLegSkip(gen.hotkey)
        if not api_key:
            from .money_guard import gate_or_adopt

            if self._operator_fallback_lanes():
                gate_or_adopt(self, round_id, gen, kind="expired_key", rent=False)
                # The payer's key aged out while their leg kept failing on our
                # side. With the owner-armed hybrid on and operator lanes on
                # file, the leg runs operator-billed rather than dropping the
                # entrant for our faults.
                log.warning("final challenger %s: no vaulted key (TTL expired) — running "
                            "on an OPERATOR lane (operator-billed; "
                            "funded_operator_fallback)", gen.hotkey)
                raise _FundedOperatorFallback(gen.hotkey)
            if self._operator_rent_allowed(gen.hotkey):
                # A queued entrant whose key aged out while the marketplace was
                # sold out is not dropped: with no operator lane on file the leg
                # rents a pod on the OPERATOR's account — ONLY once the owner
                # approved this leg (2026-11-02: an unapproved expired-key rent
                # drained the operator account). The entry is flagged so the
                # approved leg stays on our bill; the miner is never burned.
                gate_or_adopt(self, round_id, gen, kind="expired_key", rent=True)
                log.warning("final challenger %s: no vaulted key (TTL expired) — "
                            "operator-billed (payer key expired): renting a pod on the "
                            "OPERATOR's account", gen.hotkey)
                q = self._funded_queue()
                if q is not None:
                    with contextlib.suppress(Exception):
                        q.set_operator_billed(gen.hotkey, True)
                raise _FundedOperatorRent(gen.hotkey)
            self._record_funded_failure(
                gen.hotkey, "no vaulted key for this hotkey (TTL expired or "
                "never funded here) — re-fund to supply a fresh key",
                miner_fault=True, error_class="auth", burn=False)
            raise _FundedLegSkip(gen.hotkey)
        try:
            profile = self._funded_pod_profile()
            key_path = Path(profile.key_path or "").expanduser()
            ssh_pubkey = (key_path.parent / (key_path.name + ".pub")
                          ).read_text(encoding="utf-8").strip()
            skus = self._funded_skus_for_rent()
            round_sku = skus[0] if skus else ""
            if not round_sku or not rnd.funded_pod_image:
                raise RuntimeError("funded_pods=rent needs [round] "
                                   "funded_pod_sku (or funded_pod_skus) and "
                                   "funded_pod_image")
            # DEC-CA-0060: the miner's chosen type and cap, or None (default).
            chosen = (self._chosen_gpu_rent(gpu_choice, f"funded leg {gen.hotkey[:12]}")
                      if getattr(gpu_choice, "gpu", "") else None)
            if chosen is not None:
                round_sku = chosen[0][0]
        except Exception as e:  # noqa: BLE001 — operator config faults never burn miners
            self._record_funded_failure(gen.hotkey, f"operator-side funded-pod "
                                        f"config fault: {e}", miner_fault=False,
                                        error_class="infra", burn=False)
            raise _FundedLegSkip(gen.hotkey) from e
        # TRUE write-ahead: the pod's name is deterministic (launch appends
        # "-0"), so ledger the INTENT before `lium up` — a crash anywhere in
        # launch/wait/IP leaves the entry for the startup sweep, instead of a
        # pod billing its miner with no record.
        from ..provision.funded import funded_pod_name
        from ..provision.state import PodInstance

        netuid = self.cfg.subnet.netuid
        expected_id = funded_pod_name(str(round_id), gen.hotkey, netuid) + "-0"
        self._ledger_add(PodInstance(
            provider="lium", instance_id=expected_id, stage="funded",
            rented_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sku=round_sku, gpus=1, payer_hotkey=gen.hotkey))
        # One rent at a time (see _funded_rent_lock): later rents must SEE
        # earlier claims, or every concurrent pair picks the market's same
        # top executor and races one create-rate window.
        selected_provider = {"name": "lium", "credential_id": ""}

        def on_provider_selected(provider, credential_id):
            selected_provider.update(name=provider, credential_id=credential_id)
            self._ledger_add(PodInstance(
                provider=provider, instance_id=expected_id, stage="funded",
                rented_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                sku=round_sku, gpus=1, payer_hotkey=gen.hotkey,
                payer_credential_id=credential_id))

        stale_pods = 0
        fell_back = False
        if chosen is not None:
            log.info("funded leg %s: miner-chosen GPU %s (miner cap %s; %s)",
                     gen.hotkey[:12], chosen[0][0],
                     f"${chosen[1]['max_price_per_hour']:.2f}/h"
                     if chosen[1]["max_price_per_hour"] else "none",
                     "falls back to the cheapest type" if gpu_choice.allow_fallback
                     else "waits for this type")
        # Only a leg that started WITH a miner choice follows later changes;
        # king legs and default-path legs are never re-steered.
        reread_choice = bool(reread_choice and not is_king and not adopt_live
                             and getattr(gpu_choice, "gpu", ""))

        def choice_update():
            if not reread_choice:
                return None
            return _changed_gpu_choice(self, gen.hotkey, gen.ref, flight_attempt, gpu_choice)

        while True:
            new_choice = choice_update()
            if new_choice is not None:
                # The miner re-funded with another GPU choice while this leg
                # waited: switch to it before the next rent attempt — its SKU,
                # its cap, and (through the wait / doomed-leg checks below)
                # its own latest safe start; a type that no longer fits the
                # leg's target settlement is never rented.
                log.warning("funded leg %s: miner changed GPU choice %s → %s",
                            gen.hotkey[:12], _describe_gpu_choice(gpu_choice),
                            _describe_gpu_choice(new_choice))
                gpu_choice = new_choice
                chosen = (self._chosen_gpu_rent(gpu_choice, f"funded leg {gen.hotkey[:12]}")
                          if gpu_choice.gpu else None)
                fell_back = False
                if chosen is not None:
                    round_sku = chosen[0][0]
                elif skus:
                    round_sku = skus[0]
            queue = self._funded_queue()
            recovery = queue.get(gen.hotkey) if queue is not None else None
            if recovery is not None and recovery.operator_billed:
                self._ledger_remove(expected_id)
                self._record_funded_failure(
                    gen.hotkey, "a previous allocation failed on the operator side; "
                    "recovery must not rent another GPU on the miner's key",
                    miner_fault=False, error_class="operator_blocked", burn=False)
                raise _FundedLegSkip(gen.hotkey)
            # King first: a challenger never takes the rent lock (or the last
            # GPU) while the round's JIT king still needs a pod.
            if self._yield_to_king(f"funded leg {gen.hotkey[:12]}") is False:
                self._ledger_remove(expected_id)
                self._record_funded_failure(
                    gen.hotkey, "king still needs GPU capacity — challenger requeued",
                    miner_fault=False, error_class="no_capacity", burn=False)
                raise _FundedLegSkip(gen.hotkey)
            if not adopt_live and not is_king:
                # The miner waste cap, before EVERY rent on the miner's key
                # (a lemon re-rent and a retrain included).
                from .miner_spend_guard import budget_check, waste_cap_check

                try:
                    waste_cap_check(self, gen)
                    budget_check(self, gen, gpu_choice)      # the miner's own total budget
                except _FundedLegSkip:
                    self._ledger_remove(expected_id)
                    raise
            if not adopt_live:
                # The full halt check (it may probe) runs BEFORE the rent lock;
                # under the lock only the standing verdict is read (D1).
                halted = rent_halt_refusal(self, king=is_king)
                if halted:
                    self._ledger_remove(expected_id)
                    if is_king:
                        from .box_health import RentsHalted

                        raise RentsHalted(halted)
                    from ..funding.queue import OPERATOR_DISK_CLASS

                    self._record_funded_failure(
                        gen.hotkey, f"operator disk: {halted} — nothing rented",
                        miner_fault=False, error_class=OPERATOR_DISK_CLASS, burn=False)
                    raise _FundedLegSkip(gen.hotkey)
            with allocation_lock(self._funded_rent_lock) as release_allocation:
                if self._king_pending():
                    # The king can become pending while this thread waits
                    # for the rent lock. Do not race its next acquisition.
                    continue
                claimed = live_claimed_executors(self)
                from ..provision.core import scan_ssh_host_key

                allocated = []

                def on_allocated(machine_id, pod_id, *, allocated=allocated,
                                 release_allocation=release_allocation,
                                 gpu_choice=gpu_choice, round_sku=round_sku):
                    from .progress import report_progress

                    report_progress(self, "booting")
                    allocated.append(pod_id)
                    if not is_king:
                        from .miner_spend_guard import fallback_rate, spend_start

                        # a fresh create (a machine id) is a NEW rent record —
                        # never the previous rent of this reused pod name;
                        # an adopted allocation ("" machine) updates its own
                        spend_start(self, gen.hotkey, pod_id, rate=fallback_rate(self, gpu_choice),
                                    new_rent=bool(machine_id),
                                    rent_id=f"{machine_id}@{time.time():.0f}" if machine_id
                                    else "")
                    if selected_provider["credential_id"]:
                        self._ledger_add(PodInstance(
                            provider=selected_provider["name"], instance_id=pod_id, stage="funded",
                            rented_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            sku=round_sku, gpus=1, payer_hotkey=gen.hotkey,
                            payer_credential_id=selected_provider["credential_id"]))
                        if pod_id != expected_id:
                            self._ledger_remove(expected_id)
                    local = self.__dict__.get("_leg_local_obj")
                    if local is not None:
                        local.funded_allocated = True
                    if machine_id:
                        claim_executor(self, machine_id, pod_id)
                        release_allocation()

                if chosen is not None:
                    # The miner's type, the miner's cap (DEC-CA-0060).
                    fit, caps = chosen
                else:
                    # Open market: only the types a leg started now still fits.
                    fit, caps = (self._skus_fitting_now(skus) or skus,
                                 self._fallback_price_caps(gpu_choice) if fell_back
                                 else self._funded_price_caps())
                from .cost_guards import doomed_leg_reason, provider_ttl_seconds
                from .gpu_health import health_for

                doomed = "" if prior_alloc else doomed_leg_reason(self, tuple(fit))
                if doomed:
                    # Every type that could be rented now lands the leg after
                    # the settlement it must make: never start it (the miner
                    # would pay for a leg that can only be requeued at finish).
                    log.error("funded leg %s: %s", gen.hotkey[:12], doomed)
                    self._ledger_remove(expected_id)
                    self._record_funded_failure(gen.hotkey, doomed, miner_fault=False,
                                                error_class="no_capacity", burn=False)
                    raise _FundedLegSkip(gen.hotkey)
                if not adopt_live:
                    # The halt is re-checked right before EVERY launch — a wait
                    # or a lemon re-rent can outlive the leg-start check (N3).
                    # Under the rent lock: the cached verdict only (D1).
                    halted = rent_halt_refusal(self, king=is_king, cached_only=True)
                    if halted:
                        from ..funding.queue import OPERATOR_DISK_CLASS

                        self._ledger_remove(expected_id)
                        if is_king:
                            from .box_health import RentsHalted

                            raise RentsHalted(halted)
                        self._record_funded_failure(
                            gen.hotkey, f"operator disk: {halted} — nothing rented",
                            miner_fault=False, error_class=OPERATOR_DISK_CLASS, burn=False)
                        raise _FundedLegSkip(gen.hotkey)
                result = rent_funded_pod(
                    round_id=str(round_id), hotkey=gen.hotkey, api_key=api_key,
                    sku=fit[0], image=rnd.funded_pod_image,
                    ssh_pubkey=ssh_pubkey, netuid=netuid,
                    ready_timeout=rnd.funded_ready_timeout_seconds,
                    exclude_ids=claimed, host_key_scanner=scan_ssh_host_key,
                    cpu_blocklist=rnd.funded_cpu_blocklist,
                    skus=fit, price_caps=caps,
                    on_allocated=on_allocated, vault=vault,
                    on_provider_selected=on_provider_selected, gpu_health=health_for(self),
                    ttl_seconds=provider_ttl_seconds(self, kind="leg"),
                )
                if not result.ok and not result.leaked_pod:
                    from .miner_spend_guard import MAKE_GOOD, spend_end

                    for pod_id in allocated:
                        release_executor_of(self, pod_id)
                        if pod_id != expected_id:
                            self._ledger_remove(pod_id)
                        # A lemon (the platform never delivered it) is the
                        # host's fault, never the miner's: booked as an
                        # operator make-good (money safety II), not their
                        # waste. Any other failed rent: the miner paid for
                        # nothing — waste.
                        # a lemon or any allocated-then-failed rent (no ssh
                        # key pin, no identity): the host's / our fault — an
                        # operator make-good of its cost, never their waste
                        if getattr(result, "create_refused", False):
                            # The platform refused the create: nothing was ever
                            # allocated or billed — closed at its start, $0
                            # (before any re-price, which would move last-seen).
                            from .miner_spend_guard import INFRA as _INFRA

                            spend_end(self, gen.hotkey, pod_id, outcome=MAKE_GOOD, kind=_INFRA,
                                      at_last_seen=True,
                                      reason=f"create refused by the platform (nothing "
                                             f"allocated): {str(result.error)[:100]}")
                            continue
                        lemon = result.error_class != "auth"
                        landed = getattr(result, "price_per_hour", None)
                        if landed:
                            # the real price, never the flat fallback (S2)
                            from .miner_spend_guard import spend_start

                            spend_start(self, gen.hotkey, pod_id, rate=landed)
                        from .miner_spend_guard import INFRA, MINER

                        spend_end(self, gen.hotkey, pod_id,
                                  outcome=MAKE_GOOD if lemon else "waste",
                                  kind=INFRA if lemon else MINER,
                                  reason=f"{'post-rent failure' if lemon else 'rent failed'}: "
                                         f"{str(result.error)[:120]}")
                if result.ok and result.pod is not None and not is_king:
                    from .miner_spend_guard import fallback_rate, spend_start

                    landed = getattr(result, "price_per_hour", None)
                    spend_start(self, gen.hotkey, result.pod.instance_id,
                                rate=landed or fallback_rate(self, gpu_choice))
                # Review (5EvqPCMN): a failed rent / post-rent reject never
                # proposes a FULL-leg operator make-good — what the miner lost
                # is the reject's own cost, booked make_good with one priced
                # alert (miner_spend_guard.note_make_good); the leg retries on
                # their key under the bad-pod budget and the waste cap.
                if result.ok and result.machine_id:
                    claim_executor(self, result.machine_id,
                                   result.pod.instance_id if result.pod is not None else "")
                if result.ok:
                    local = self.__dict__.get("_leg_local_obj")
                    if local is not None:
                        local.funded_allocated = True
                    # The pod answered SSH — but is it running the PINNED
                    # code? A host serving a stale image under the pinned tag
                    # boots an old worker that echoes our injected digest env.
                    # Hash its files before it ever sees a leg; a mismatch
                    # releases the pod, keeps its executor excluded and rents
                    # again — the miner is never blamed.
                    adopted = bool(getattr(result, "adopted", False))
                    # The pod's REAL host, now that its ssh address is known: a
                    # listing that could not resolve hosts must never train on
                    # a quarantined/blocklisted machine (2026-11-01 05:16,
                    # 38.255.28.18). Torn down at once; its executor stays
                    # excluded (claimed) like any rejected pod.
                    blocked = ("" if adopted or result.address is None
                               else host_blocked_reason(self, result.address.ip))
                    why = blocked or self._funded_pod_code_mismatch(result, profile)
                    # Throughput gate (funded_host_bench_floor): a slow executor
                    # eats the leg's equal wall — released and rented again on
                    # the same bad-pod budget. Only its executor stays excluded
                    # (claimed above): slowness is per card, not per host.
                    # An ADOPTED pod is mid-leg: its GPU is busy with the run
                    # we are about to attach to, so the calibration bench
                    # would read low and "release" the pod, killing the leg. It
                    # passed the gate when it was rented.
                    slow = ("" if (why or adopted)
                            else self._funded_pod_too_slow(result, profile))
                    # Multi-GPU preflight (an N-rank NCCL all_reduce under a
                    # hard timeout): a host that hangs or crawls a collective
                    # passed every other check. Its host is quarantined (the
                    # fabric is per machine, not per executor id) by the
                    # check itself. Adopted pods are mid-leg: skipped.
                    if not (why or slow or adopted):
                        slow = funded_pod_comm_failure(self, result, profile)
                    if why or slow:
                        if result.pod is not None:
                            health_for(self).record(
                                result.pod.sku, f"{result.pod.provider}:{result.pod.instance_id}",
                                kind="stale" if why else "slow", reason=why or slow)
                        queue = self._funded_queue()
                        stale_pods += 1
                        if why and not blocked and result.address is not None:
                            # The stale cache lives on the HOST: every executor
                            # id it lists would boot the same image.
                            record_host_quarantine(result.address.ip, why)
                        why = why or slow
                        log.warning("funded leg %s: pod %s rejected — %s; releasing "
                                    "and renting again (%d stale pod(s) so far)",
                                    gen.hotkey[:12], result.pod.instance_id if result.pod
                                    else "?", why, stale_pods)
                        if str(why).startswith("short pod") and result.machine_id:
                            from ..provision.short_pods import strike

                            strike(executor_id=result.machine_id,
                                   ip=getattr(result.address, "ip", "") or "",
                                   reason=f"{gen.hotkey[:12]}: {why}")
                        if result.pod is not None and not is_king:
                            # OUR post-rent reject (short pod / stale image /
                            # slow host / NCCL) on the miner's key: an
                            # operator make-good, never the miner's waste,
                            # with one priced alert (money safety II).
                            from .miner_spend_guard import MAKE_GOOD, spend_end

                            spend_end(self, gen.hotkey, result.pod.instance_id,
                                      outcome=MAKE_GOOD,
                                      reason=f"post-rent reject: {str(why)[:120]}")
                        self._teardown_funded_pod(result.pod)
                        # A confirmed release permits the bounded replacement
                        # loop. An unconfirmed release must block another rent.
                        if result.pod and any(p.instance_id == result.pod.instance_id
                                              for p in self._load_funded_ledger()):
                            result = replace(result, ok=False, error_class="infra",
                                             burn_attempt=False, leaked_pod=result.pod.instance_id)
                            break
                        # The teardown released its claim; this executor stays
                        # excluded (instance-less claim, aged out as a backstop).
                        claim_executor(self, result.machine_id)
                        # The released pod billed the miner for nothing: one
                        # operator-side loss, and a replacement only inside
                        # both the bad-pod budget and the loss cap.
                        if not self._count_funded_loss(
                                gen.hotkey, more=stale_pods < self.FUNDED_MAX_STALE_PODS):
                            result = replace(
                                result, ok=False,
                                error=f"{stale_pods} consecutive pods booted a stale "
                                      f"worker image ({why})", error_class="infra",
                                burn_attempt=False, pod=None, address=None)
                            break
                        continue
                elif result.error_class == LEMON_CLASS:
                    # The platform delivered nothing usable (never ready): the
                    # host is already quarantined by rent_funded_pod — rent
                    # again elsewhere while the round's latest safe start
                    # allows, on the same bad-pod budget as a stale image.
                    stale_pods += 1
                    now_fn = getattr(self, "_rent_wait_now", None) or time.time
                    if self._count_funded_loss(
                            gen.hotkey, more=(stale_pods < self.FUNDED_MAX_STALE_PODS
                                              and now_fn() < self._funded_rent_wait_deadline())):
                        log.warning("funded leg %s: lemon pod (%s); renting again on "
                                    "another host (%d bad pod(s) so far)",
                                    gen.hotkey[:12], error_tail(result.error, 200), stale_pods)
                        continue
                    result = replace(
                        result, error=f"{stale_pods} bad pod(s), last: {result.error}",
                        error_class="infra", burn_attempt=False)
            if result.ok or result.error_class != "no_capacity":
                break
            if chosen is not None and gpu_choice.allow_fallback:
                # DEC-CA-0060: the miner allowed it — rent the default
                # (cheapest fitting type, operator caps) from now on.
                log.info("funded leg %s: no %s capacity under the miner's cap — falling "
                         "back to the cheapest fitting type (allow_fallback)",
                         gen.hotkey[:12], chosen[0][0])
                chosen = None
                fell_back = True        # min(miner cap, operator caps) from now on
                continue
            # Sold out RIGHT NOW is not a verdict: wait for a GPU (outside the
            # rent lock, so sibling legs that find one proceed) and rent again.
            # Only the round's latest safe start turns this into a requeue.
            from .progress import report_progress

            report_progress(self, "waiting_gpu", "market_capacity")
            from ..provision.payer_pool import payer_capacity
            wait_caps = (chosen[1] if chosen is not None else
                         self._fallback_price_caps(gpu_choice) if fell_back
                         else self._funded_price_caps())
            waited = self._wait_for_funded_capacity(
                chosen[0] if chosen is not None else skus,
                describe=f"funded leg {gen.hotkey[:12]}", hotkey=gen.hotkey,
                capacity_probe=lambda fitting, wait_caps=wait_caps: payer_capacity(api_key, fitting, wait_caps,
                    live_claimed_executors(self), health_for(self)),
                allow_paid_fallback=chosen is None,
                **({"choice_changed": lambda: choice_update() is not None}
                   if reread_choice else {}),
                **({"price_caps": wait_caps} if chosen is not None or fell_back else {}))
            if waited == "gpu_changed":
                continue                  # re-derived at the top of the loop
            if waited in ("operator_rent", "operator"):
                from .money_guard import gate_or_adopt

                self._ledger_remove(expected_id)
                gate_or_adopt(self, round_id, gen, kind=("paid_fallback" if waited == "operator_rent"
                                                         else "operator_lane"),
                              rent=waited == "operator_rent")
            if waited == "operator_rent":
                queue = self._funded_queue()
                if queue is not None:
                    queue.set_operator_billed(gen.hotkey, True)
                raise _FundedOperatorRent(gen.hotkey)
            if waited == "operator":
                # Owner-armed hybrid: this leg runs on an operator lane
                # (operator-billed). The payer's write-ahead row goes (no pod
                # was ever launched for it) and nothing is recorded as a fault.
                self._ledger_remove(expected_id)
                raise _FundedOperatorFallback(gen.hotkey)
            if not waited:
                break
        if not result.ok:
            # rent_funded_pod's own failure path verified-terminated anything
            # it launched; drop the write-ahead entry only when nothing leaked
            # (a leaked pod keeps its ledger row for the sweep).
            if not result.leaked_pod:
                self._ledger_remove(expected_id)
            # A LEAKED pod's spend record stays OPEN (money safety II): it may
            # still bill the miner, so its cost keeps growing until the sweep
            # CONFIRMS it gone (the sweep's teardown closes it as waste).
            msg = result.error
            if chosen is not None and result.error_class == "no_capacity":
                msg = (f"no {chosen[0][0]} capacity (miner-chosen GPU, no fallback) before "
                       f"the latest safe start — requeued, nothing rented: {msg}")
            if result.leaked_pod:
                msg += (f" [LEAKED pod {result.leaked_pod} may still bill this "
                        f"payer — stop it through its GPU provider on your "
                        f"account]")
            self._record_funded_failure(
                gen.hotkey, msg, miner_fault=(result.error_class == "auth"),
                error_class=result.error_class, burn=result.burn_attempt)
            raise _FundedLegSkip(gen.hotkey)
        # Replace the intent row with the real pod record (same instance id).
        self._ledger_add(result.pod)
        if result.pod.instance_id != expected_id:
            self._ledger_remove(expected_id)
        # Least-privilege credential for a payer-controlled box: a per-pod Hub
        # robot (push-only, checkpoint project, revoked at teardown) and
        # NOTHING from the orchestrator's environment (cascade.funding.robots).
        # Fail closed — never fall back to forwarding the operator's login.
        from dataclasses import replace as dc_replace

        if self._funded_harvest():
            # Credential-free pod: the worker trains --local-only and the
            # orchestrator harvests the checkpoint, so NOTHING is minted and
            # nothing at all reaches the pod's environment.
            env_pairs: tuple = ()
            pod = result.pod
        else:
            cred = self._funded_pod_credential(str(round_id), gen.hotkey)
            if cred is None:
                self._teardown_funded_pod(result.pod)
                self._record_funded_failure(
                    gen.hotkey, "operator-side fault: could not mint a per-pod Hub "
                    "credential (the leg never forwards operator logins to a payer "
                    "pod)", miner_fault=False, error_class="infra", burn=False)
                raise _FundedLegSkip(gen.hotkey)
            env_pairs, robot_id = cred
            pod = dc_replace(result.pod, robot_id=robot_id)
            self._ledger_add(pod)
        host = RemoteHost(
            name=f"funded-{gen.hotkey[:12].lower()}",
            host=result.address.ip, port=result.address.ssh_port,
            user=profile.user, key_path=profile.key_path,
            remote_python=profile.remote_python, workdir=profile.workdir,
            # A layout pod ("8xH200") trains on all its GPUs.
            cuda_device=visible_devices(getattr(result.pod, "sku", "") or ""),
            chain_toml=profile.chain_toml,
            forward_env=(), static_env=env_pairs, isolated=True,
            pinned_host_key=result.host_key,
            ssh_options=profile.ssh_options,
            stage="final",
        )
        # The payer's pod boots the image's baked chain.toml (pre-release
        # pin): give it the deployed one before any dispatch.
        host = self._push_deployed_chain_toml(host)
        return host, pod

    def _funded_checkpoint_mismatch(self, entry, contract) -> str | None:
        """Why a funded leg's published checkpoint fails the ingest guard, or
        ``None`` when it passes (cascade.eval.checkpoint_guard: repo-identical
        code, contract config, pinned weight shapes and size). Unfetchable
        counts as a mismatch — an entry we cannot inspect is not published."""
        from ..eval.checkpoint_guard import (
            CheckpointDiverged,
            CheckpointTampered,
            verify_checkpoint,
            verify_weights_finite,
        )
        from ..shared.hippius import HubConfig, HubRef, fetch_from_hub
        from ..shared.manifest import parse_trained_pointer

        pointer = str(getattr(entry, "trained_pointer", "") or "")
        ref = parse_trained_pointer(pointer)
        if ref is None:
            return f"malformed trained_pointer {pointer!r}"
        try:
            dest = (self.work_root / "_funded_ckpt_guard"
                    / HubRef.parse(ref).digest.replace(":", "-"))
            fetch_from_hub(ref, dest, HubConfig.from_storage(self.cfg.storage))
            verify_checkpoint(dest, contract)
            verify_weights_finite(dest)
        except CheckpointTampered as e:
            return str(e)
        except CheckpointDiverged as e:
            # Not a mismatch: the run diverged. Its own class, its own verdict.
            raise _FundedDiverged(str(e)) from e
        except Exception as e:  # noqa: BLE001 — unfetchable ⇒ not publishable
            return f"could not fetch/inspect the checkpoint: {str(e)[:200]}"
        return None

    def _refuse_diverged_king(self, entry, contract) -> None:
        """Abort the final on a king checkpoint whose weights are NaN/inf.

        The king's pod pushes its checkpoint itself; fetch it back and run the
        finiteness check before it can be paired in a manifest — a NaN king
        entry makes every validator's scorer raise and costs the whole round
        its verdict. Unfetchable is an infra failure of the final (we never
        publish what we could not inspect)."""
        from ..eval.checkpoint_guard import CheckpointDiverged, verify_weights_finite
        from ..shared.hippius import HubConfig, HubRef, fetch_from_hub
        from ..shared.manifest import parse_trained_pointer

        pointer = str(getattr(entry, "trained_pointer", "") or "")
        ref = parse_trained_pointer(pointer)
        if ref is None:
            raise RuntimeError(f"king leg returned a malformed trained_pointer {pointer!r}")
        dest = (self.work_root / "_funded_ckpt_guard"
                / HubRef.parse(ref).digest.replace(":", "-"))
        fetch_from_hub(ref, dest, HubConfig.from_storage(self.cfg.storage))
        try:
            verify_weights_finite(dest)
        except CheckpointDiverged as e:
            raise _FundedDiverged(f"king checkpoint: {e}") from e

    def _pod_checkpoint_dir(self, entry, seeds, contract, suffix: str) -> str:
        """Where the leg's checkpoint lives on its pod: the local-train
        receipt says; a pushed entry trained into the worker's default
        ``_train_work`` layout."""
        given = str(getattr(entry, "checkpoint_dir", "") or "")
        if given:
            return given
        return (f"_train_work/{seeds.base_seed}/{contract.arch_preset}/"
                f"challenger{suffix}/checkpoint")

    def _funded_pod_identity_mismatch(self, pod) -> str | None:
        """Why the pod answering to ``pod.instance_id`` is not the one we
        rented, or ``None`` when it is. Names are owner-chosen and reusable;
        the platform's pod id is not — so a payer who relaunched "their" pod
        under the same name shows up here (checked before dispatch and when
        the leg returns; the pinned SSH host key guards the transport in
        between). Unverifiable (no key, API down) counts as a mismatch: we
        never dispatch into a pod we cannot identify."""
        if not getattr(pod, "pod_uid", ""):
            return "no platform identity was pinned at rent"
        vault = self._payer_vault()
        from ..funding.providers import pod_credential

        credential = pod_credential(vault, pod) if vault is not None else None
        if credential is None:
            return KEY_LOST_REASON
        try:
            from ..provision.funded import provider_for_key

            ident = provider_for_key(credential.provider, credential.api_key).pod_identity(pod.instance_id)
        except Exception as e:  # noqa: BLE001 — unverifiable ⇒ mismatch
            return f"identity check failed: {str(e)[:200]}"
        if ident is None:
            return "pod no longer listed on the payer's account"
        if ident.get("id") != pod.pod_uid:
            return (f"pod id changed: rented {pod.pod_uid}, now {ident.get('id')} "
                    "(replaced under the same name)")
        if str(ident.get("status", "")).upper() != "RUNNING":
            return f"pod status is {ident.get('status')!r}, not RUNNING"
        return None

    def _stage_king_vault(self, host, gen):
        """The JIT king pod with the king's vault ZIP staged when the king is a
        PRIVATE (``vault/direct``) submission: such a ref resolves only where
        its ZIP is staged, and the operator-rented king pod never goes through
        the funded/operator-lane staging. Public refs pass through
        unchanged."""
        from ..funding.store import parse_vault_ref

        digest = parse_vault_ref(gen.ref)
        if not digest:
            return host
        log.info("king %s is a private vault submission — staging its ZIP on the king pod",
                 gen.hotkey)
        return self._stage_vault_zip_on(host, digest)

    def _stage_vault_zip_on(self, host, digest_hex: str):
        """Ship ONE vault ZIP to the funded pod; return the host with the
        pod-local ``CASCADE_VAULT_DIR`` pinned into its dispatch env."""
        import shlex
        import subprocess
        from dataclasses import replace as dc_replace

        store = self._submission_store()
        if store is None:
            raise RuntimeError("vault ref selected but submission_vault_dir unset")
        staged = store.stage_for_dispatch(
            digest_hex, self.work_root / "_vault_dispatch" / digest_hex)
        from .remote import build_scp_argv, build_ssh_argv

        pod_dir = f"{host.workdir}/_vault_stage"
        # Same transport policy as every dispatch: a pinned host key stays
        # pinned for the copy (an accept-new scp here would be the one hole
        # in the leg's identity pin).
        subprocess.run(build_ssh_argv(host, f"mkdir -p {shlex.quote(pod_dir)}"),
                       check=True, capture_output=True, timeout=60)
        subprocess.run(build_scp_argv(host, str(staged), f"{pod_dir}/{digest_hex}.zip"),
                       check=True, capture_output=True, timeout=300)
        return dc_replace(host, static_env=(*host.static_env,
                                            ("CASCADE_VAULT_DIR", pod_dir)))

    def _funded_pod_code_mismatch(self, result, profile) -> str:
        """Why a freshly rented funded pod must NOT receive a leg, or ``""``.

        Hashes the pod's ``cascade/**/*.py`` over the pinned ssh identity and
        compares it with ``[round] worker_code_fingerprint``
        (:mod:`cascade.provision.codeprint`). Unpinned ⇒ always ``""``. A
        transport failure is a mismatch too: a pod we cannot verify is a pod
        we do not dispatch to.
        """
        pinned = (self.cfg.round.worker_code_fingerprint or "").strip().lower()
        if not pinned or result.address is None:
            return ""
        from ..provision.codeprint import CHECK_TIMEOUT_SECONDS, remote_code_fingerprint
        from .remote import RemoteHost, build_ssh_argv, run_ssh

        host = RemoteHost(
            name="funded-codecheck", host=result.address.ip, port=result.address.ssh_port,
            user=profile.user, key_path=profile.key_path,
            remote_python=profile.remote_python, workdir=profile.workdir,
            pinned_host_key=result.host_key, ssh_options=profile.ssh_options,
        )

        def _run(argv):
            return run_ssh(build_ssh_argv(host, shlex.join(list(argv))),
                           timeout=CHECK_TIMEOUT_SECONDS)

        try:
            got, n, err = remote_code_fingerprint(_run, workdir=profile.workdir)
        except Exception as e:  # noqa: BLE001 — unverifiable ⇒ unusable
            return f"code fingerprint check errored: {e}"
        if err:
            return f"code fingerprint unavailable: {err}"
        if got != pinned:
            return (f"stale worker image: pod code {got[:12]}… ({n} files) != pinned "
                    f"{pinned[:12]}…")
        return ""

    def _operator_lane_gate(self, host) -> str | None:
        """Why an operator lane may NOT join the final's pool, or None.

        Three checks, cheapest first: the ssh endpoint answers (a refused
        endpoint also quarantines the host); the pod runs THIS release's
        worker (:func:`probe_worker_runtime`); the host clears the bench floor
        (:meth:`_host_bench_below_floor`, off when no floor is set for the
        SKU) — the same guards a funded rent applies, so a hand-rented lane
        can be neither slow nor dead."""
        from .host_preflight import host_blocked_reason, multi_gpu_preflight_failure
        from .remote import build_ssh_argv, probe_worker_runtime, run_ssh

        label = f"lane {getattr(host, 'name', host)}"
        # [round] funded_host_blocklist + the host quarantine file: a host we
        # refuse to rent is refused as a lane too (no ssh needed to say so).
        why = host_blocked_reason(self, str(getattr(host, "host", "") or ""))
        if why:
            return why
        try:
            p = run_ssh(build_ssh_argv(host, "echo LANE_OK"), timeout=45)
        except Exception as e:  # noqa: BLE001
            return f"ssh failed: {e}"
        if p.returncode != 0 or "LANE_OK" not in (p.stdout or ""):
            if p.returncode == 255:
                _quarantine_lane_host(host, f"{label}: ssh refused at lane join")
            return f"ssh rc={p.returncode}: {(p.stderr or '')[-160:]}"
        if getattr(self, "deployed_chain_toml", None):
            # Same config a rented pod gets; the lane's workers run with
            # --chain-toml <workdir>/chain.deployed.toml (_rolling_final_hosts).
            try:
                self._push_deployed_chain_toml(host)
            except Exception as e:  # noqa: BLE001 — a lane without the config must not join
                return f"deployed chain.toml push failed: {e}"
        why = probe_worker_runtime(host)
        if why:
            return why
        rnd = self.cfg.round
        sku = getattr(self, "_funded_round_sku", "") or getattr(rnd, "funded_pod_sku", "")
        if sku:
            why = self._host_bench_below_floor(host, sku, label)
            if why:
                return why
        # Multi-GPU preflight on the lane's own GPUs (its CUDA_VISIBLE_DEVICES,
        # else its SKU's layout); a failing lane's host is quarantined.
        why = multi_gpu_preflight_failure(
            self, host, str(getattr(host, "sku", "") or sku or ""), label)
        if why:
            return why
        return None

    def _host_bench_below_floor(self, host, sku: str, label: str) -> str:
        """Why ``host`` is too slow for this round, or ``""``.

        Runs the fixed calibration bench over ssh (:func:`cascade.trainer.
        remote.probe_host_bench`) and compares its composite tokens/s with
        ``[round] funded_host_bench_floor[sku]``. Off (no floor for the SKU)
        ⇒ no probe at all. A probe that cannot be measured is logged and
        passes: the check exists to keep a slow executor from eating a leg's
        equal wall, and must never itself sink a leg over a transient ssh
        hiccup. Every measured value is logged so the fleet's spread stays
        visible even while the floor is off.
        """
        from ..shared.config import funded_host_bench_floor_for
        from .remote import probe_host_bench

        why = self._host_gpu_shortfall(host, sku, label)
        if why:
            return why
        floor = funded_host_bench_floor_for(
            getattr(self.cfg.round, "funded_host_bench_floor", ()) or (), sku)
        if floor <= 0:
            return ""
        tps, err = probe_host_bench(host)
        if tps is None:
            log.warning("%s: host bench unavailable (%s) — floor %s tokens/s not "
                        "enforced on this pod", label, err, f"{floor:,.0f}")
            return ""
        log.info("%s: host bench %s tokens/s (floor %s, sku %s)",
                 label, f"{tps:,.0f}", f"{floor:,.0f}", sku)
        if tps < floor:
            return (f"slow host: calibration bench {tps:,.0f} tokens/s < floor "
                    f"{floor:,.0f} for {sku}")
        return ""

    def _host_gpu_shortfall(self, host, sku: str, label: str) -> str:
        """Why ``host`` has fewer GPUs than the layout it was rented as, or
        ``""``. Lium has sold an ``8xH200`` executor that exposed 6 (testnet
        2026-10-30): the leg trains on fewer cards than its wall assumes, at
        the full layout price. An unreadable count passes, like the bench."""
        from ..provision.layout import layout_gpus
        from .remote import probe_gpu_count

        want = layout_gpus(sku)
        if want <= 1:
            return ""
        got, err = probe_gpu_count(host)
        if got is None:
            log.warning("%s: gpu count unavailable (%s) — %d-GPU layout not verified",
                        label, err, want)
            return ""
        if got < want:
            why = f"short pod: {got} GPU(s) visible, rented as {sku} ({want})"
            # The missing cards are the HOST's (every executor id it lists
            # sells the same short machine): quarantine its IP, not just the
            # executor id (money safety II; 17 post-rent rejects Oct 30-Nov 3).
            ip = str(getattr(host, "host", "") or "")
            if ip:
                from ..provision.core import record_host_quarantine

                record_host_quarantine(ip, f"{label}: {why}", host_fault=True)
            # The persistent short-pod blacklist is struck ONCE, by the caller
            # that knows the executor id (review N1): the payer reject site,
            # the operator-leg / king lemon paths.
            return why
        return ""

    def _funded_pod_too_slow(self, result, profile) -> str:
        """Why a freshly rented funded pod must NOT receive a leg on throughput
        grounds, or ``""`` — the pre-dispatch half of the host-bench floor
        (see :meth:`_host_bench_below_floor`); same ssh identity pin as the
        code fingerprint check."""
        if result.address is None:
            return ""
        rnd = self.cfg.round
        sku = (getattr(result, "sku", "") or getattr(self, "_funded_round_sku", "")
               or rnd.funded_pod_sku)
        from .remote import RemoteHost

        host = RemoteHost(
            name="funded-hostbench", host=result.address.ip, port=result.address.ssh_port,
            user=profile.user, key_path=profile.key_path,
            remote_python=profile.remote_python, workdir=profile.workdir,
            cuda_device="0", pinned_host_key=result.host_key,
            ssh_options=profile.ssh_options,
        )
        pod_id = result.pod.instance_id if result.pod is not None else "?"
        return self._host_bench_below_floor(host, sku, f"funded pod {pod_id}")

    def _teardown_funded_pod(self, pod) -> None:
        """Tear one funded pod down NOW (leg finished); ledger reflects reality."""
        # A pod torn down without a recorded result produced nothing for the
        # miner: close its spend record as waste (a no-op after a result).
        from .miner_spend_guard import spend_end

        spend_end(self, str(getattr(pod, "payer_hotkey", "") or ""),
                  str(getattr(pod, "instance_id", "") or ""), outcome="waste",
                  reason="torn down without a harvested result")
        # The leg is over: its Hub robot dies first, whatever the pod does.
        self._revoke_robot(pod)
        vault = self._payer_vault()
        if vault is None:
            # Never silent: a miner-billed pod would run with zero operator
            # signal. The ledger entry stays so the sweep retries once the
            # vault is configured again.
            log.error("funded pod %s (payer %s): [round] payer_vault_dir is "
                      "unset — cannot tear it down; it bills the miner until "
                      "the vault is restored", pod.instance_id, pod.payer_hotkey)
            return
        from ..provision.funded import teardown_funded

        try:
            leftovers = teardown_funded([pod], vault)
        except Exception as e:  # noqa: BLE001
            from .money_watch import note_teardown

            note_teardown(self, pod.instance_id, ok=False)
            log.error("funded pod %s teardown crashed (kept on ledger for the "
                      "startup sweep): %s", pod.instance_id, e)
            return
        from .money_watch import note_teardown

        note_teardown(self, pod.instance_id, ok=not leftovers)
        if leftovers:
            log.error("funded pod %s could not be confirmed gone — still "
                      "billing payer %s; kept on ledger", pod.instance_id,
                      pod.payer_hotkey)
        else:
            # Confirmed gone: its executor is free for the next rent.
            release_executor_of(self, pod.instance_id)
            try:
                self._ledger_remove(pod.instance_id)
            except OSError as e:
                # A full disk (ENOSPC, 2026-10-31) after the pod is gone: the
                # row stays and the startup sweep re-confirms it (an empty
                # listing of an observed pod settles as gone, no re-bill).
                log.error("funded pod %s is gone but its ledger row could not be "
                          "removed (%s) — the startup sweep settles it",
                          pod.instance_id, e)


def funded_pod_comm_failure(runner, result, profile) -> str:
    """Why a freshly rented funded pod fails the multi-GPU preflight, or ``""``
    (single-GPU layout / preflight off / no address). Same ssh identity pin as
    the code-fingerprint check; ALL of the layout's GPUs are visible."""
    if getattr(result, "address", None) is None:
        return ""
    from ..provision.layout import visible_devices
    from .host_preflight import multi_gpu_preflight_failure
    from .remote import RemoteHost

    rnd = runner.cfg.round
    sku = (getattr(getattr(result, "pod", None), "sku", "") or getattr(result, "sku", "")
           or getattr(runner, "_funded_round_sku", "") or getattr(rnd, "funded_pod_sku", ""))
    host = RemoteHost(
        name="funded-nccl", host=result.address.ip, port=result.address.ssh_port,
        user=profile.user, key_path=profile.key_path,
        remote_python=profile.remote_python, workdir=profile.workdir,
        cuda_device=visible_devices(sku or ""), pinned_host_key=result.host_key,
        ssh_options=profile.ssh_options, sku=sku or "",
    )
    pod_id = result.pod.instance_id if getattr(result, "pod", None) is not None else "?"
    return multi_gpu_preflight_failure(runner, host, sku or "", f"funded pod {pod_id}")
