"""Durable direct scheduler: provisioning → training → evaluation → settlement.

The scheduler's decisions use the journal. Worker-host dispatch is idempotent per
job/fence, and a replaced process can only publish non-authoritative objects.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from ..provision.direct import Allocation, AllocationPending
from ..shared.direct_moe import canonical, digest
from .evaluation import replay_decision
from .signing import sign


class LocalCPUProvider:
    """Actual existing CPU capacity; only for local/staging validation, no cloud bill."""

    def __init__(self, lease_seconds=3600):
        self.lease_seconds = lease_seconds
        self.reservations = {}

    def lookup(self, key):
        return self.reservations.get(key)

    def reserve(self, key, *, profile, world_size, cap, now):
        if profile != "local-cpu" or world_size > (os.cpu_count() or 1):
            raise ValueError("local CPU capacity/profile mismatch")
        self.reservations.setdefault(
            key, Allocation(key, tuple(range(world_size)), now + self.lease_seconds, profile, 0)
        )
        return self.reservations[key]

    def release(self, key):
        self.reservations.pop(key, None)


class LocalTorchDispatcher:
    """Real torchrun processes on the worker host; survives controller restart."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.children = []

    def directory(self, job, fence):
        return self.root / digest({"job": job, "fence": fence})

    def attest(self, spec):
        if spec["device"] != "cpu":
            raise ValueError("local integration dispatcher cannot attest GPUs")
        import torch

        if torch.__version__.split("+")[0] != "2.4.1":
            raise ValueError("worker PyTorch version differs from pinned release")
        return list(range(spec["world_size"]))

    def ensure_started(self, spec):
        root = self.directory(spec["job"], spec["fence"])
        root.mkdir(exist_ok=True)
        spec = {**spec, "output": str(root / "checkpoints")}
        path = root / "spec.json"
        body = canonical(spec)
        if path.exists() and path.read_text() != body:
            raise ValueError("worker fence reused with different treatment")
        if not path.exists():
            # Controller is the only spec publisher; process lock protects launch.
            path.write_text(body)
        if (root / "done.json").exists():
            return
        with (root / "launch.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            log = (root / "worker.log").open("ab")
            try:
                # Pass the already-held lock into the supervisor. There is no
                # check/unlock/spawn gap in which a second controller can launch.
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "cascade.direct.supervisor",
                        "--spec",
                        str(path),
                        "--run",
                        str(root),
                        "--lock-fd",
                        str(lock.fileno()),
                    ],
                    pass_fds=(lock.fileno(),),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.children.append(child)
            finally:
                log.close()
        self.children[:] = [p for p in self.children if p.poll() is None]

    def poll(self, job, fence):
        root = self.directory(job, fence)
        # Read completion first: a visible marker guarantees the final checkpoint
        # was already published. Completion arriving later waits for the next poll.
        done = (
            json.loads((root / "done.json").read_text()) if (root / "done.json").exists() else None
        )
        manifests = []
        for path in sorted((root / "checkpoints").glob("checkpoint-*.json")):
            manifests.append(json.loads(path.read_text())["manifest"])
        return manifests, done

    def stop(self, job, fence):
        root = self.directory(job, fence)
        if (root / "pid.json").exists() and not (root / "done.json").exists():
            record = json.loads((root / "pid.json").read_text())
            pid = record["pid"]
            with contextlib.suppress(ProcessLookupError, FileNotFoundError):
                start = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[19]
                if start != record["start"]:
                    raise ValueError("worker PID was reused; refuse to signal unrelated process")
                os.killpg(pid, signal.SIGTERM)


class DirectRuntime:
    def __init__(self, controller, dispatcher, evaluator, *, trainer_key=None):
        self.controller, self.dispatcher, self.evaluator = controller, dispatcher, evaluator
        self.trainer_key = trainer_key

    def tick(self, now):
        ctl = self.controller
        outcomes = {}
        for job in list(ctl.ledger.state["jobs"]):
            try:
                j = ctl.ledger.state["jobs"][job]
                if "work" not in j:
                    outcomes[job] = "awaiting_training_contract"
                    continue
                if j["state"] in (
                    "admitted",
                    "retry_wait",
                    "waiting_capacity",
                    "reserving",
                    "releasing",
                ):
                    if now < j["next_retry"]:
                        continue
                    ctl.reserve(job, now=now)
                j = ctl.ledger.state["jobs"][job]
                resume = (
                    j["checkpoint"]
                    if j["state"] == "reserved"
                    else j.get("launch_resume", j["checkpoint"])
                )
                spec = {**j["work"], "fence": j["fence"], "resume": resume}
                if spec["device"] == "cuda":
                    spec["worker_deadline"] = j["lease_until"]
                if j["reservation"] and hasattr(self.dispatcher, "bind_allocation"):
                    self.dispatcher.bind_allocation(
                        job,
                        j["reservation"].get("allocation"),
                        j["reservation"]["profile"],
                        j["reservation"]["key"],
                        provider=j["reservation"]["provider"],
                    )
                if j["state"] == "reserved":
                    try:
                        ready = self.dispatcher.attest(spec)
                    except ValueError:
                        ctl.fault(job, "capacity", now=now)
                        outcomes[job] = "hardware_qualification_failed"
                        continue
                    ctl.launch(job, ready_ranks=ready, now=now)
                j = ctl.ledger.state["jobs"][job]
                if j["state"] in ("running", "checkpointing"):
                    if now >= j["lease_until"]:
                        self.dispatcher.stop(job, j["fence"])
                        ctl.fault(job, "worker", now=now)
                        continue
                    self.dispatcher.ensure_started(spec)
                    manifests, done = self.dispatcher.poll(job, j["fence"])
                    for manifest in manifests:
                        if manifest["consumed"] > ctl.ledger.state["jobs"][job]["consumed"]:
                            ctl.checkpoint(job, manifest, now=now)
                    if done is not None:
                        current = ctl.ledger.state["jobs"][job]
                        if (
                            done["returncode"] != 0
                            or current["consumed"] != current["terms"]["budget"]
                        ):
                            ctl.fault(job, "worker", now=now)
                        else:
                            ctl.release(job, now=now, reason="completed")
                j = ctl.ledger.state["jobs"][job]
                if j["state"] in ("evaluating", "inconclusive") and now >= j["next_retry"]:
                    try:
                        decision = self.evaluator.evaluate(job, ctl.ledger.state)
                        replay_decision(decision)
                    except (OSError, TimeoutError, ValueError, subprocess.SubprocessError) as exc:
                        ctl.fault(job, "evaluation", now=now)
                        outcomes[job] = f"evaluation_retry:{type(exc).__name__}"
                        continue
                    ctl.apply(
                        f"decision:{job}:{digest(decision)}",
                        "decision_evidence",
                        now,
                        job=job,
                        decision=decision,
                    )
                    receipt = decision["public"]
                    ctl.apply(
                        f"evaluate:{job}:{digest(receipt)}",
                        "evaluation",
                        now,
                        job=job,
                        receipt=receipt,
                    )
                outcomes[job] = ctl.ledger.state["jobs"][job]["state"]
            except AllocationPending:
                outcomes[job] = "waiting_provider_reconciliation"
            except (OSError, TimeoutError) as exc:
                # Retain allocation/checkpoint and the original effect intent.
                # Never convert an infrastructure timeout into scientific failure.
                outcomes[job] = f"infrastructure_retry:{type(exc).__name__}"
        return outcomes

    def settle(self, job, *, now):
        ctl = self.controller
        j = ctl.ledger.state["jobs"][job]
        cohort = ctl.ledger.state["cohorts"][j["terms"]["cohort"]]
        winner = cohort["winner"]
        if winner is not None and winner != job:
            reference = digest(ctl.ledger.state["jobs"][winner]["checkpoint"])
            if j["receipt"]["verdict"]["reference"] != reference:
                decision = self.evaluator.evaluate(job, ctl.ledger.state, reference_id=reference)
                replay_decision(decision)
                ctl.apply(
                    f"decision:{job}:{digest(decision)}",
                    "decision_evidence",
                    now,
                    job=job,
                    decision=decision,
                )
                ctl.apply(
                    f"compare:{job}:{digest(decision['public'])}",
                    "comparison",
                    now,
                    job=job,
                    receipt=decision["public"],
                )
        return ctl.settle(job, now=now)

    def merge(self, jobs, *, optimizer_policy, now):
        from ..trainer.direct_model import build_direct_model
        from ..trainer.moe_model import moe_param_names
        from .experts import merge_checkpoint

        ctl = self.controller
        state = ctl.ledger.state
        if not jobs:
            return ctl.merge([], None, None, now=now)
        architecture = state["jobs"][jobs[0]]["work"]["architecture"]
        model = build_direct_model(architecture, device="meta")
        contributions = []
        for job in jobs:
            j = state["jobs"][job]
            if (
                j["state"] != "accepted"
                or j["terms"]["role"] != "expert"
                or j["terms"]["anchor"] != state["anchor"]
            ):
                raise ValueError("expert is not accepted against the current anchor")
            if digest(j["work"]["architecture"]) != digest(architecture):
                raise ValueError("expert architecture differs from merge contract")
            contributions.append(
                (j["checkpoint"], {int(k): v for k, v in j["work"]["selection"].items()})
            )
        base = self.evaluator.manifest(state["anchor"], state)
        manifest = merge_checkpoint(
            ctl.objects,
            base,
            contributions,
            moe_param_names(model),
            optimizer_policy=optimizer_policy,
        )
        pointer = digest(manifest)
        self.evaluator.initial_artifacts[pointer] = manifest
        decision = self.evaluator.evaluate_artifacts(state["anchor"], pointer, state, architecture)
        replay_decision(decision)
        return ctl.apply(
            f"merge:{digest([jobs, pointer, decision])}",
            "merge",
            now,
            jobs=jobs,
            combined=pointer,
            receipt=decision["public"],
            manifest=manifest,
            decision=decision,
        )

    def signed_journal(self):
        if self.trainer_key is None:
            raise ValueError("configured trainer signing key required")
        return sign(self.controller.ledger.export(), self.trainer_key, scope="journal")
