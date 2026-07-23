"""Remote training dispatch — run king and challenger on separate GPU boxes.

The trainer can run a round's two (or more) trainings on separate rented GPU pods
(Lium, Targon, or any SSH-reachable host) **in parallel**, instead of
sequentially on one local device. Because the compute budget is a fixed
``train_tokens`` count (not wall-clock), splitting across devices keeps king and
challenger on *identical compute*; only byte-exact re-derivation relaxes to
tolerance — rented marketplace hardware varies (see ``chain.toml`` corpus_mode).

Design: the remote unit is a **round-worker**, not a remote ``BaseTrainer``. Each
pod pulls its generator from the Hippius Hub registry by ref, builds the corpus in
its own sandbox, trains, uploads the checkpoint to the registry, and prints a
``TrainedEntry`` receipt (``cascade.trainer.worker``). The orchestrator (which
holds the wallet) collects the receipts and signs + publishes the manifest
locally — **the trainer hotkey never lands on a rented box**. A pod needs
cascade + torch + a GPU + registry/S3 access (seed its env once when you rent
it), not the wallet.

Hosts live in a **trainer-local** file (NOT ``chain.toml`` — that file is public
and shared with miners/validators). Transport is the system ``ssh`` client, so
there is no extra dependency. The command-construction and receipt-parsing
helpers are pure and unit-tested; only :func:`dispatch_train` shells out.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..shared.manifest import TrainedEntry

log = logging.getLogger("cascade.trainer.remote")

# Marker the worker prints immediately before its JSON receipt, so the orchestrator
# can pick the receipt out of arbitrary stdout (banners, framework chatter).
RECEIPT_SENTINEL = "__CASCADE_RECEIPT__"


class RemoteDispatchError(RuntimeError):
    """An SSH dispatch or receipt parse failed."""


HOST_STAGES = ("any", "heat", "final")


@dataclass(frozen=True)
class RemoteHost:
    """One SSH-reachable GPU pod that can run a training worker.

    ``forward_env`` names env vars the orchestrator copies from its own
    environment into the remote command (e.g. registry/S3 credentials) — use it
    only if you have not pre-seeded the pod's env; the bittensor wallet is never
    forwarded. ``cuda_device`` pins ``CUDA_VISIBLE_DEVICES`` on the pod.

    ``stage`` restricts which round stage the pod serves: ``"heat"`` (screen
    trainings only), ``"final"`` (king/finalist trainings only), or ``"any"``
    (both — the default, and the pre-stage behaviour). This is the cheap-GPU
    seam: heats can run on a cheaper SKU class (e.g. A6000) because heat
    checkpoints are trainer-internal — screened and discarded, never validated —
    while the final MUST stay on one SKU (the validator's gpu_name gate pairs
    king and challenger). Keep each stage's pods a single SKU.
    """

    name: str
    host: str
    port: int = 22
    user: str = "root"
    key_path: str | None = None
    remote_python: str = "python"
    workdir: str = "."
    cuda_device: str | None = None
    chain_toml: str | None = None          # path to chain.toml on the pod (if non-default)
    forward_env: tuple[str, ...] = ()
    ssh_options: tuple[str, ...] = ()       # extra raw `-o Key=Value` style flags
    stage: str = "any"                      # "any" | "heat" | "final"


def load_hosts(path: Path | str) -> list[RemoteHost]:
    """Load remote training hosts from a trainer-local TOML file.

    Schema (``[[host]]`` array of tables)::

        [[host]]
        name = "king-box"
        host = "1.2.3.4"
        port = 22
        user = "root"
        key_path = "~/.ssh/lium"
        remote_python = "/root/cascade/.venv/bin/python"
        workdir = "/root/cascade"
        cuda_device = "0"
        forward_env = ["HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY", "HIPPIUS_HUB_TOKEN"]
        stage = "any"       # "heat" | "final" | "any" — which round stage this pod serves
    """
    p = Path(path)
    if not p.is_file():
        raise RemoteDispatchError(f"remote hosts file not found: {p}")
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    entries = raw.get("host", [])
    if not entries:
        raise RemoteDispatchError(f"no [[host]] entries in {p}")
    hosts: list[RemoteHost] = []
    for h in entries:
        stage = str(h.get("stage", "any"))
        if stage not in HOST_STAGES:
            raise RemoteDispatchError(
                f"host {h.get('name', '?')!r}: stage={stage!r} invalid; expected one of {HOST_STAGES}"
            )
        hosts.append(
            RemoteHost(
                name=str(h["name"]),
                host=str(h["host"]),
                port=int(h.get("port", 22)),
                user=str(h.get("user", "root")),
                key_path=h.get("key_path"),
                remote_python=str(h.get("remote_python", "python")),
                workdir=str(h.get("workdir", ".")),
                cuda_device=(str(h["cuda_device"]) if "cuda_device" in h else None),
                chain_toml=h.get("chain_toml"),
                forward_env=tuple(str(x) for x in h.get("forward_env", ())),
                ssh_options=tuple(str(x) for x in h.get("ssh_options", ())),
                stage=stage,
            )
        )
    return hosts


def worker_argv(
    host: RemoteHost,
    *,
    gen_ref: str,
    uid: int,
    hotkey: str,
    role: str,
    base_seed: int,
    block: int,
    trainer_spec: str,
    arch_preset: str | None = None,
    train_hours: float | None = None,
    repo_suffix: str = "",
    warm_start_ref: str | None = None,
) -> list[str]:
    """The ``cascade.trainer.worker`` argv to run on the pod (no env/cd).

    ``arch_preset`` pins which configured size the pod trains (the primary size
    or one of ``[[training.sizes]]``); omitted ⇒ the worker trains the primary
    size, preserving single-size behaviour. ``train_hours`` overrides the compute
    budget (a cheap heat screen); ``repo_suffix`` disambiguates the checkpoint
    repo so parallel same-size runs (heat challengers) don't collide.
    ``warm_start_ref`` is the round's pinned Cascade init (a trained_pointer);
    the pod fetches it from the registry and trains from its weights instead of
    random init — omitted ⇒ random init."""
    argv = [
        host.remote_python, "-m", "cascade.trainer.worker",
        "--gen-ref", gen_ref,
        "--uid", str(int(uid)),
        "--hotkey", hotkey,
        "--role", role,
        "--base-seed", str(int(base_seed)),
        "--block", str(int(block)),
        "--trainer", trainer_spec,
    ]
    if arch_preset:
        argv += ["--arch-preset", arch_preset]
    if train_hours is not None:
        argv += ["--train-hours", repr(float(train_hours))]
    if repo_suffix:
        # `=` form: the suffix starts with '-' (e.g. -heat-u3), which argparse
        # would otherwise mistake for a flag.
        argv.append(f"--repo-suffix={repo_suffix}")
    if warm_start_ref:
        argv += ["--warm-start-ref", warm_start_ref]
    if host.chain_toml:
        argv += ["--chain-toml", host.chain_toml]
    return argv


# Training always wins the GPU: every dispatch first kills any still-running
# post-round benchmark sweep (log-only telemetry, see bench_hook) so a straggler
# can never contend a worker toward its max_train_seconds guard. The pattern is
# anchored to the venv entrypoint path (how the running scorer's cmdline reads)
# with the bracket trick, so it matches neither this dispatch command's own
# shell nor the benchmark *launch* command's shell — only the live scorer. The
# trailing `( |$)` keeps it off the scorer's siblings: without it the substring
# also matches `bin/cascade-benchmark-download`, and every dispatch would kill
# an in-progress (multi-hour) benchmark-data pull on the pod.
PREEMPT_BENCHMARKS = "pkill -f 'bin/cascade[-]benchmark( |$)' 2>/dev/null; "


def pod_lane_count(host: RemoteHost, hosts: list[RemoteHost] | None) -> int:
    """How many worker lanes share ``host``'s physical pod in the active fleet.

    A multi-GPU pod appears in the hosts list as one ``[[host]]`` entry per
    GPU lane — same SSH endpoint, different ``cuda_device`` — so lanes on a
    pod are exactly the entries sharing ``(host, port)``. Only the
    orchestrator can count this fan-out (each lane's pod-side view is a
    masked single device); the count travels to the pod via
    :func:`build_remote_command` so the sandbox can slice cores fairly.
    Endpoint-less host objects (test stubs) count as single-lane.
    """
    key = (getattr(host, "host", None), getattr(host, "port", None))
    if not hosts or key[0] is None:
        return 1
    return sum(
        1 for h in hosts
        if (getattr(h, "host", None), getattr(h, "port", None)) == key
    ) or 1


def build_remote_command(
    host: RemoteHost, argv: list[str], env: dict[str, str],
    *, lane_count: int | None = None,
) -> str:
    """The single shell string ssh runs on the pod: benchmark preemption, then
    ``cd workdir && ENV… argv``.

    Everything is ``shlex.quote``d so credentials/paths with spaces or shell
    metacharacters can't break out.

    ``lane_count`` (the pod's lane fan-out, see :func:`pod_lane_count`) stamps
    ``CASCADE_LANE_INDEX``/``CASCADE_LANE_COUNT`` into the lane's env — the
    seam the sandbox's per-lane CPU fairness reads (``_lane_cpu_slice``). Both
    are OMITTED unless the pod runs >1 lane AND ``cuda_device`` is a single
    device ordinal, so local runs, single-lane pods, and non-lane masks keep
    today's behavior exactly.
    """
    prefix = ""
    full_env = dict(env)
    if host.cuda_device is not None:
        full_env["CUDA_VISIBLE_DEVICES"] = host.cuda_device
        device = str(host.cuda_device).strip()
        if lane_count is not None and lane_count > 1 and device.isdigit():
            full_env["CASCADE_LANE_INDEX"] = device
            full_env["CASCADE_LANE_COUNT"] = str(int(lane_count))
    if full_env:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(full_env.items())) + " "
    return f"{PREEMPT_BENCHMARKS}cd {shlex.quote(host.workdir)} && {prefix}{shlex.join(argv)}"


def build_ssh_argv(host: RemoteHost, remote_command: str) -> list[str]:
    """The local ``ssh`` argv that runs ``remote_command`` on ``host``."""
    argv = ["ssh", "-p", str(host.port), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new"]
    if host.key_path:
        argv += ["-i", str(Path(host.key_path).expanduser())]
    for opt in host.ssh_options:
        argv += ["-o", opt]
    argv += [f"{host.user}@{host.host}", remote_command]
    return argv


def parse_receipt(stdout: str) -> dict:
    """Extract the worker's JSON receipt (the text after :data:`RECEIPT_SENTINEL`).

    The worker sends logs to stderr and the receipt to stdout, but we still scan
    for the sentinel so stray stdout chatter (CUDA banners, etc.) is tolerated.
    """
    for line in reversed(stdout.splitlines()):
        idx = line.find(RECEIPT_SENTINEL)
        if idx >= 0:
            payload = line[idx + len(RECEIPT_SENTINEL):].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError as e:
                raise RemoteDispatchError(f"malformed receipt JSON: {e}") from e
    raise RemoteDispatchError("no receipt sentinel found in worker stdout")


def receipt_to_entry(receipt: dict) -> TrainedEntry:
    """Validate a receipt dict into a :class:`TrainedEntry` (re-runs its checks)."""
    try:
        return TrainedEntry(
            miner_hotkey=str(receipt["miner_hotkey"]),
            miner_uid=int(receipt["miner_uid"]),
            role=str(receipt["role"]),
            gen_ref=str(receipt["gen_ref"]),
            trained_pointer=str(receipt["trained_pointer"]),
            corpus_digest=str(receipt["corpus_digest"]),
            train_block=int(receipt["train_block"]),
            gpu_name=str(receipt.get("gpu_name", "")),
            size=str(receipt.get("size", "")),
        )
    except (KeyError, ValueError) as e:
        raise RemoteDispatchError(f"receipt is not a valid TrainedEntry: {e}") from e


@dataclass
class RemoteDispatcher:
    """Runs a training worker on a remote pod over SSH and returns its receipt."""

    trainer_spec: str
    timeout_seconds: int = 6 * 3600       # generous: a full ~3h training + overhead
    _runner: object = field(default=None, repr=False)  # injectable for tests

    def dispatch(
        self,
        host: RemoteHost,
        *,
        gen_ref: str,
        uid: int,
        hotkey: str,
        role: str,
        base_seed: int,
        block: int,
        arch_preset: str | None = None,
        train_hours: float | None = None,
        repo_suffix: str = "",
        warm_start_ref: str | None = None,
        lane_count: int | None = None,
    ) -> TrainedEntry:
        import os

        argv = worker_argv(
            host, gen_ref=gen_ref, uid=uid, hotkey=hotkey, role=role,
            base_seed=base_seed, block=block, trainer_spec=self.trainer_spec,
            arch_preset=arch_preset, train_hours=train_hours, repo_suffix=repo_suffix,
            warm_start_ref=warm_start_ref,
        )
        env = {k: os.environ[k] for k in host.forward_env if k in os.environ}
        remote_cmd = build_remote_command(host, argv, env, lane_count=lane_count)
        ssh_argv = build_ssh_argv(host, remote_cmd)
        log.info("dispatch role=%s → %s (%s) device=%s", role, host.name, host.host,
                 host.cuda_device)
        try:
            proc = (self._runner or run_ssh)(ssh_argv, self.timeout_seconds)
        except subprocess.TimeoutExpired as e:
            raise RemoteDispatchError(f"remote {role} on {host.name} timed out") from e
        if proc.returncode == 3:
            # Worker rc=3 = miner submission rejected (CorpusError): the last
            # stderr line is the one-line reason — no traceback to relay.
            reason = (proc.stderr or "").strip().splitlines()[-1:] or ["(no reason)"]
            raise RemoteDispatchError(
                f"remote {role} on {host.name}: miner submission rejected: {reason[0]}"
            )
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:]
            raise RemoteDispatchError(
                f"remote {role} on {host.name} failed (rc={proc.returncode}): {tail}"
            )
        entry = receipt_to_entry(parse_receipt(proc.stdout or ""))
        if entry.role != role:
            raise RemoteDispatchError(f"receipt role {entry.role!r} != dispatched {role!r}")
        return entry


def run_ssh(ssh_argv: list[str], timeout: int):
    """Run the ssh command, returning the CompletedProcess (text mode)."""
    return subprocess.run(ssh_argv, capture_output=True, text=True, timeout=timeout)
