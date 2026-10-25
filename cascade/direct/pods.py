"""SSH dispatch to a rented gang pod, with orchestrator-side harvest to Hippius.

The rolling lane's transport (``cascade.trainer.remote``), applied to direct
jobs: the pod carries NO credentials — not Hippius, not the payer key. The
orchestrator stages every object a job reads into the pod's content store,
starts the worker supervisor detached, polls over SSH, and harvests each
checkpoint (tar under the ``data`` filter) into Hippius before the controller
records it. Every SSH runs against the host key pinned at reservation.

Worker-side state is idempotent per job/fence: ``LocalTorchDispatcher`` holds
the launch flock across supervisor, launcher and ranks, and a repeated
``start`` attaches instead of launching twice.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from ..provision.direct import AllocationPending, ProviderTransient
from ..shared.direct_moe import canonical, digest
from ..trainer.remote import RemoteHost, build_ssh_argv, run_ssh

RUNS_DIR = "direct-runs"
OBJECTS_DIR = "direct-objects"


def object_refs(value):
    """Every ``{"uri", "sha256"}`` reference inside a spec or manifest."""
    if isinstance(value, dict):
        if set(value) >= {"uri", "sha256"} and isinstance(value.get("sha256"), str):
            yield value
        for v in value.values():
            yield from object_refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from object_refs(v)


def rewrite_uris(manifest, prefix):
    """``manifest`` with every object URI moved under ``prefix`` (by digest)."""
    if isinstance(manifest, dict):
        if set(manifest) >= {"uri", "sha256"} and isinstance(manifest.get("sha256"), str):
            return {**manifest, "uri": prefix + manifest["sha256"]}
        return {k: rewrite_uris(v, prefix) for k, v in manifest.items()}
    if isinstance(manifest, list):
        return [rewrite_uris(v, prefix) for v in manifest]
    return manifest


class PodDispatcher:
    def __init__(self, objects, providers, profiles, *, remote_python, workdir, key_path=None,
                 staging_dir=None, runner=run_ssh, tar_runner=None):
        self.objects, self.providers, self.profiles = objects, providers, profiles
        self.python, self.workdir, self.key_path = remote_python, workdir.rstrip("/"), key_path
        self.staging = Path(staging_dir or tempfile.gettempdir())
        self.run, self.tar_runner = runner, tar_runner
        self.bindings = {}
        self.uploaded = set()

    @property
    def runs(self):
        return f"{self.workdir}/{RUNS_DIR}"

    @property
    def store(self):
        return f"{self.workdir}/{OBJECTS_DIR}"

    def bind_allocation(self, job, allocation, profile, reservation, provider=None):
        self.bindings[job] = (allocation, profile, reservation, provider)

    def host(self, job):
        allocation, _, reservation, provider = self.bindings[job]
        address, host_key = self.providers[provider].address(reservation)
        return RemoteHost(name=allocation, host=address.ip, port=address.ssh_port,
                          key_path=self.key_path, remote_python=self.python, workdir=self.workdir,
                          pinned_host_key=host_key, isolated=True)

    def _command(self, verb, *args):
        return (f"cd {shlex.quote(self.workdir)} && {shlex.quote(self.python)} -m cascade.direct.pods "
                + shlex.join([verb, "--runs", self.runs, "--objects", self.store, *args]))

    def _call(self, job, verb, *args, body=None, timeout=180):
        host = self.host(job)
        try:
            p = self.run(build_ssh_argv(host, self._command(verb, *args)), timeout,
                         canonical(body) if body is not None else "")
        except (subprocess.TimeoutExpired, OSError) as e:
            raise ProviderTransient(f"pod {verb}: {e}") from None
        if p.returncode != 0:
            raise ProviderTransient(f"pod {verb} failed (rc={p.returncode}): {(p.stderr or '')[-300:]}")
        lines = [ln for ln in (p.stdout or "").splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None

    def pod_spec(self, spec):
        # Storage is excluded from the training contract digest, so pointing the
        # worker at its pod-local store changes no admitted term.
        return {**spec, "storage": {"kind": "pod", "root": self.store}}

    # ── staging and harvest ──────────────────────────────────────────────────

    def _stage(self, job, refs):
        wanted = sorted({r["sha256"] for r in refs})
        missing = self._call(job, "missing", body=wanted) or []
        if not missing:
            return
        if set(missing) - set(wanted):
            raise ValueError("pod reported objects it was not asked for")
        tmp = Path(tempfile.mkdtemp(dir=self.staging, prefix="direct-stage-"))
        try:
            for sha in missing:
                self.objects.download(self.objects.prefix + sha, tmp / sha)
            host = self.host(job)
            argv = build_ssh_argv(host, f"mkdir -p {shlex.quote(self.store)} && "
                                        f"tar -C {shlex.quote(self.store)} -xf -")
            if self.tar_runner is not None:
                rc = self.tar_runner("push", argv, tmp, missing)
            else:
                proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                with tarfile.open(fileobj=proc.stdin, mode="w|") as tf:
                    for sha in missing:
                        tf.add(tmp / sha, arcname=sha)
                proc.stdin.close()
                rc = proc.wait(timeout=3600)
            if rc != 0:
                raise ProviderTransient(f"staging objects to pod failed (rc={rc})")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _harvest(self, job, manifest):
        shas = sorted({r["sha256"] for r in object_refs(manifest)} - self.uploaded)
        shas = [s for s in shas if not self.objects.exists(self.objects.prefix + s)]
        if shas:
            tmp = Path(tempfile.mkdtemp(dir=self.staging, prefix="direct-harvest-"))
            try:
                host = self.host(job)
                argv = build_ssh_argv(host, f"tar -C {shlex.quote(self.store)} -cf - "
                                            + " ".join(shlex.quote(s) for s in shas))
                if self.tar_runner is not None:
                    rc = self.tar_runner("pull", argv, tmp, shas)
                else:
                    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    try:
                        # The stream comes off a payer-controlled box: the data
                        # filter refuses traversal, links and device nodes.
                        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
                            tf.extractall(tmp, filter="data")
                    except tarfile.TarError as e:
                        proc.kill()
                        raise ProviderTransient(f"refused checkpoint archive: {e}") from None
                    finally:
                        proc.stdout.close()
                    rc = proc.wait(timeout=600)
                if rc != 0:
                    raise ProviderTransient(f"checkpoint harvest failed (rc={rc})")
                for sha in shas:
                    # put_file re-hashes: a corrupted or substituted file fails here.
                    self.objects.put_file(self.objects.prefix + sha, tmp / sha)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        self.uploaded.update(r["sha256"] for r in object_refs(manifest))
        return rewrite_uris(manifest, self.objects.prefix)

    # ── dispatcher protocol (DirectRuntime) ──────────────────────────────────

    def attest(self, spec):
        _, profile, _, _ = self.bindings[spec["job"]]
        expected = self.profiles[profile]
        result = self._call(spec["job"], "attest", body={"architecture": spec["architecture"]})
        if result is None:
            raise AllocationPending("awaiting measured GPU readiness")
        if (
            result["torch"] != "2.4.1"
            or result["count"] != spec["world_size"]
            or len(result["gpus"]) != spec["world_size"]
        ):
            raise ValueError("remote worker software/topology mismatch")
        for gpu in result["gpus"]:
            if expected.gpu_name not in gpu["name"] or gpu["bytes"] < expected.memory_gib * 1024**3:
                raise ValueError("remote worker GPU differs from qualified profile")
        if result["architecture"] != digest(spec["architecture"]):
            raise ValueError("remote worker contract attestation mismatch")
        return list(range(spec["world_size"]))

    def ensure_started(self, spec):
        self._stage(spec["job"], list(object_refs({k: v for k, v in spec.items() if k != "storage"})))
        self._call(spec["job"], "start", body=self.pod_spec(spec))

    def poll(self, job, fence):
        result = self._call(job, "poll", "--job", job, "--fence", str(fence))
        manifests = [self._harvest(job, m) for m in result["manifests"]]
        return sorted(manifests, key=lambda m: m["consumed"]), result["done"]

    def stop(self, job, fence):
        # The provider terminates the pod; fencing is committed by the controller
        # before any replacement becomes authoritative.
        with contextlib.suppress(ProviderTransient, AllocationPending):
            self._call(job, "stop", "--job", job, "--fence", str(fence), timeout=60)


# ── pod-side entry point (runs inside the worker image, no credentials) ─────


def main(argv=None):
    p = argparse.ArgumentParser(prog="cascade.direct.pods")
    p.add_argument("verb", choices=("attest", "missing", "start", "poll", "stop"))
    p.add_argument("--runs", required=True)
    p.add_argument("--objects", required=True)
    p.add_argument("--job")
    p.add_argument("--fence")
    args = p.parse_args(argv)
    from .runtime import LocalTorchDispatcher

    store = Path(args.objects)
    store.mkdir(parents=True, exist_ok=True)
    workers = LocalTorchDispatcher(args.runs)
    raw = sys.stdin.read() if args.verb in ("attest", "missing", "start") else ""
    body = json.loads(raw) if raw.strip() else None
    if args.verb == "attest":
        import torch

        from ..trainer.direct_model import build_direct_model

        build_direct_model(body["architecture"], device="meta")
        out = dict(
            torch=torch.__version__.split("+")[0],
            count=torch.cuda.device_count(),
            gpus=[dict(name=torch.cuda.get_device_name(i),
                       bytes=torch.cuda.get_device_properties(i).total_memory)
                  for i in range(torch.cuda.device_count())],
            architecture=digest(body["architecture"]),
        )
    elif args.verb == "missing":
        out = [sha for sha in body if not (store / sha).is_file()]
    elif args.verb == "start":
        workers.ensure_started(body)
        out = {"started": True}
    elif args.verb == "poll":
        manifests, done = workers.poll(args.job, int(args.fence))
        out = {"manifests": manifests, "done": done}
    else:
        workers.stop(args.job, int(args.fence))
        out = {"stopped": True}
    print(canonical(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
