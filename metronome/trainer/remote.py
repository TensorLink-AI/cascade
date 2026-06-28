"""Remote training dispatch — run king and challenger on separate GPU boxes.

The trainer can run a round's two (or more) trainings on separate rented GPU pods
(Lium, Targon, or any SSH-reachable host) **in parallel**, instead of
sequentially on one local device. Because the compute budget is a fixed
``train_tokens`` count (not wall-clock), splitting across devices keeps king and
challenger on *identical compute*; only byte-exact re-derivation relaxes to
tolerance — rented marketplace hardware varies (see ``chain.toml`` corpus_mode).

Design: the remote unit is a **round-worker**, not a remote ``BaseTrainer``. Each
pod pulls its generator from the Hippius registry by CID, builds the corpus in
its own sandbox, trains, uploads the checkpoint to the registry, and prints a
``TrainedEntry`` receipt (``metronome.trainer.worker``). The orchestrator (which
holds the wallet) collects the receipts and signs + publishes the manifest
locally — **the trainer hotkey never lands on a rented box**. A pod needs
metronome + torch + a GPU + registry/S3 access (seed its env once when you rent
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

log = logging.getLogger("metronome.trainer.remote")

# Marker the worker prints immediately before its JSON receipt, so the orchestrator
# can pick the receipt out of arbitrary stdout (banners, framework chatter).
RECEIPT_SENTINEL = "__METRONOME_RECEIPT__"


class RemoteDispatchError(RuntimeError):
    """An SSH dispatch or receipt parse failed."""


@dataclass(frozen=True)
class RemoteHost:
    """One SSH-reachable GPU pod that can run a training worker.

    ``forward_env`` names env vars the orchestrator copies from its own
    environment into the remote command (e.g. registry/S3 credentials) — use it
    only if you have not pre-seeded the pod's env; the bittensor wallet is never
    forwarded. ``cuda_device`` pins ``CUDA_VISIBLE_DEVICES`` on the pod.
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


def load_hosts(path: Path | str) -> list[RemoteHost]:
    """Load remote training hosts from a trainer-local TOML file.

    Schema (``[[host]]`` array of tables)::

        [[host]]
        name = "king-box"
        host = "1.2.3.4"
        port = 22
        user = "root"
        key_path = "~/.ssh/lium"
        remote_python = "/root/metronome/.venv/bin/python"
        workdir = "/root/metronome"
        cuda_device = "0"
        forward_env = ["HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY", "IPFS_NODE_URL"]
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
            )
        )
    return hosts


def worker_argv(
    host: RemoteHost,
    *,
    gen_cid: str,
    uid: int,
    hotkey: str,
    role: str,
    base_seed: int,
    block: int,
    trainer_spec: str,
) -> list[str]:
    """The ``metronome.trainer.worker`` argv to run on the pod (no env/cd)."""
    argv = [
        host.remote_python, "-m", "metronome.trainer.worker",
        "--gen-cid", gen_cid,
        "--uid", str(int(uid)),
        "--hotkey", hotkey,
        "--role", role,
        "--base-seed", str(int(base_seed)),
        "--block", str(int(block)),
        "--trainer", trainer_spec,
    ]
    if host.chain_toml:
        argv += ["--chain-toml", host.chain_toml]
    return argv


def build_remote_command(host: RemoteHost, argv: list[str], env: dict[str, str]) -> str:
    """The single shell string ssh runs on the pod: ``cd workdir && ENV… argv``.

    Everything is ``shlex.quote``d so credentials/paths with spaces or shell
    metacharacters can't break out.
    """
    prefix = ""
    full_env = dict(env)
    if host.cuda_device is not None:
        full_env["CUDA_VISIBLE_DEVICES"] = host.cuda_device
    if full_env:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(full_env.items())) + " "
    return f"cd {shlex.quote(host.workdir)} && {prefix}{shlex.join(argv)}"


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
            gen_cid=str(receipt["gen_cid"]),
            trained_pointer=str(receipt["trained_pointer"]),
            corpus_digest=str(receipt["corpus_digest"]),
            train_block=int(receipt["train_block"]),
            tar_digest=str(receipt.get("tar_digest", "")),
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
        gen_cid: str,
        uid: int,
        hotkey: str,
        role: str,
        base_seed: int,
        block: int,
    ) -> TrainedEntry:
        import os

        argv = worker_argv(
            host, gen_cid=gen_cid, uid=uid, hotkey=hotkey, role=role,
            base_seed=base_seed, block=block, trainer_spec=self.trainer_spec,
        )
        env = {k: os.environ[k] for k in host.forward_env if k in os.environ}
        remote_cmd = build_remote_command(host, argv, env)
        ssh_argv = build_ssh_argv(host, remote_cmd)
        log.info("dispatch role=%s → %s (%s) device=%s", role, host.name, host.host,
                 host.cuda_device)
        try:
            proc = (self._runner or _run_ssh)(ssh_argv, self.timeout_seconds)
        except subprocess.TimeoutExpired as e:
            raise RemoteDispatchError(f"remote {role} on {host.name} timed out") from e
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:]
            raise RemoteDispatchError(
                f"remote {role} on {host.name} failed (rc={proc.returncode}): {tail}"
            )
        entry = receipt_to_entry(parse_receipt(proc.stdout or ""))
        if entry.role != role:
            raise RemoteDispatchError(f"receipt role {entry.role!r} != dispatched {role!r}")
        return entry


def _run_ssh(ssh_argv: list[str], timeout: int):
    """Run the ssh command, returning the CompletedProcess (text mode)."""
    return subprocess.run(ssh_argv, capture_output=True, text=True, timeout=timeout)
