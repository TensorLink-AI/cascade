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

import contextlib
import json
import logging
import re
import secrets
import shlex
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..shared.manifest import TrainedEntry

log = logging.getLogger("cascade.trainer.remote")

# Marker the worker prints immediately before its JSON receipt, so the orchestrator
# can pick the receipt out of arbitrary stdout (banners, framework chatter).
RECEIPT_SENTINEL = "__CASCADE_RECEIPT__"


def error_tail(text: object, limit: int = 300) -> str:
    """The last ``limit`` characters of an error message WITHOUT cutting a line
    in half: whole trailing lines, plus the first line's context (``remote king
    on X failed (rc=1)``) when it fits. A raw ``str(e)[-300:]`` on a relayed
    traceback logged ``t cause of the following exception:`` — a sliced
    ``The above exception was the direct cause …`` line — exactly where the
    real cause was needed (2026-09-21 review)."""
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    lines = [ln.rstrip() for ln in s.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return s[-limit:]
    head = lines[0]
    if len(head) > limit // 2:
        head = head[: limit // 2 - 1] + "…"
    budget = limit - len(head) - 3           # " … " joiner
    tail: list[str] = []
    used = 0
    for ln in reversed(lines[1:]):
        if used + len(ln) + 1 > budget:
            break
        tail.append(ln)
        used += len(ln) + 1
    if not tail:
        last = lines[-1]
        tail = [last if len(last) <= budget else "…" + last[-(budget - 1):]]
    return head + " … " + "\n".join(reversed(tail))


class RemoteDispatchError(RuntimeError):
    """An SSH dispatch or receipt parse failed.

    ``returncode`` carries the ssh exit status when the failure was a non-zero
    remote/transport exit — 255 marks an SSH transport failure (host
    unreachable, auth refused, sshd not ready), which the heat fan-out counts to
    refuse caching a fleet-wide dead-pod wipeout as a completed heat. ``None``
    for parse/timeout/logic failures that carry no ssh status."""

    def __init__(self, *args, returncode: int | None = None) -> None:
        super().__init__(*args)
        self.returncode = returncode


HOST_STAGES = ("any", "heat", "final")


@dataclass(frozen=True)
class RemoteHost:
    """One SSH-reachable GPU pod that can run a training worker.

    ``forward_env`` names env vars the orchestrator copies from its own
    environment to the remote worker, piped over stdin per dispatch — never on
    the remote command line (e.g. registry/S3 credentials) — use it
    only if you have not pre-seeded the pod's env; the bittensor wallet is never
    forwarded. ``WANDB_API_KEY`` need not be listed here: when [wandb] is enabled
    the trainer auto-forwards it (RemoteDispatcher.extra_forward_env) so pod-side
    wandb logs, since the training — and its wandb run — happen on the pod.
    ``cuda_device`` pins ``CUDA_VISIBLE_DEVICES`` on the pod.

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
    # Fixed (name, value) pairs merged into the dispatch's stdin env AFTER the
    # forward_env copies — per-HOST values the orchestrator's own environment
    # cannot supply (e.g. a funded pod's staged-vault dir, which is a pod-local
    # path). Travels on stdin like every forwarded credential, never on the
    # remote command line.
    static_env: tuple[tuple[str, str], ...] = ()
    # An ISOLATED host receives nothing from the orchestrator's environment —
    # not its forward_env, not the dispatcher's global extras (WANDB_API_KEY)
    # — only its own static_env. Set on funded (payer-account) pods: the
    # payer has console access, so every forwarded value is theirs to read.
    isolated: bool = False
    # A PROFILE entry, never a lane: funded rentals mirror its key/python/
    # workdir/forward_env onto the pods they rent (``_funded_pod_profile``),
    # but nothing is ever dispatched to it. Loopback addresses count as
    # profile-only too (the provisioner's republish may drop this field).
    profile_only: bool = False
    # A pinned SSH host key line (``<keytype> <base64>``) for this host. When
    # set, every ssh to it runs StrictHostKeyChecking=yes against a
    # known_hosts file holding ONLY this key, so a container swapped under
    # the same address (a payer relaunching "their" pod) is refused at the
    # transport — a pod's host key is generated at its first boot.
    pinned_host_key: str = ""
    # GPU type of this lane (marketplace SKU name, e.g. "L40S"). Drives the
    # lane's own latest safe start via [round] funded_sku_wall_seconds — a
    # fast lane may still take a leg late in the epoch. "" = unknown (the
    # contract's max_train_seconds bounds it).
    sku: str = ""


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
        sku = "L40S"        # optional: the lane's GPU type (per-SKU latest safe start)
        isolated = true     # optional: the pod receives NOTHING from the orchestrator's
                            # environment (forward_env ignored); its legs train
                            # --local-only and the orchestrator harvests, verifies and
                            # uploads the checkpoint itself. Prefer it for every lane.
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
                static_env=tuple(sorted(
                    (str(k), str(v)) for k, v in dict(h.get("static_env", {})).items())),
                profile_only=bool(h.get("profile_only", False)),
                sku=str(h.get("sku", "") or "").strip(),
                isolated=bool(h.get("isolated", False)),
            )
        )
    return hosts


def is_profile_only(host: RemoteHost) -> bool:
    """True for a hosts.toml entry that exists only to be mirrored by funded
    rentals — never a dispatch target (explicit flag, or a loopback address)."""
    return bool(getattr(host, "profile_only", False)) or str(
        getattr(host, "host", "")) in ("127.0.0.1", "localhost", "::1")


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
    anneal: bool = False,
    local_only: bool = False,
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
    if anneal:
        # bench-anneal telemetry leg (DEC-CA-0030): pure cosine decay resume
        argv.append("--anneal")
    if local_only:
        # credential-free pod (DEC-CA-0036): no upload; the orchestrator
        # harvests the checkpoint and uploads it under its own identity
        argv.append("--local-only")
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
) -> tuple[str, str | None]:
    """The shell string ssh runs on the pod plus the stdin payload that carries
    the forwarded env: ``(command, stdin_env)``.

    ``env`` (the forwarded credentials) is deliberately NOT embedded in the
    command string: everything in the command lands in the remote shell's
    ``/proc/<pid>/cmdline``, which any process on the pod can read for the
    whole run — and heat pods execute miner-submitted generator code. Instead
    the assignments travel on stdin and the command sources them
    (``set -a && . /dev/stdin && set +a``) before exec'ing the worker; stdin
    is visible only to the receiving process. ``stdin_env`` is ``None`` (and
    the sourcing step absent) when there is nothing to forward. Values are
    ``shlex.quote``d in both channels so paths/credentials with spaces or
    shell metacharacters can't break out.

    ``CUDA_VISIBLE_DEVICES`` and the lane stamps stay inline — they are not
    secret, and on-cmdline visibility is what makes a pod's lane layout
    readable in ``ps`` when debugging.

    ``lane_count`` (the pod's lane fan-out, see :func:`pod_lane_count`) stamps
    ``CASCADE_LANE_INDEX``/``CASCADE_LANE_COUNT`` into the lane's env — the
    seam the sandbox's per-lane CPU fairness reads (``_lane_cpu_slice``). Both
    are OMITTED unless the pod runs >1 lane AND ``cuda_device`` is a single
    device ordinal, so local runs, single-lane pods, and non-lane masks keep
    today's behavior exactly.
    """
    prefix = _lane_prefix(host, lane_count)
    stdin_env = _stdin_env(env)
    source = "set -a && . /dev/stdin && set +a && " if stdin_env is not None else ""
    # The worker runs in its OWN process group and the wrapper kills whatever
    # that group left behind the moment the worker exits — a lingering child
    # (the generator sandbox) holding the session's stdout/stderr keeps sshd
    # waiting for EOF, so the orchestrator never sees the exit (2026-09-12: the
    # king's worker exited rc=3 at 11:08; the dispatch returned at 14:51).
    # `set -m` gives the backgrounded job its own pgid ($w); `wait` collects
    # the worker's rc; `kill -- -$w` reaps the group; the rc is preserved.
    guarded = _guarded_worker(prefix, argv)
    command = f"{PREEMPT_BENCHMARKS}cd {shlex.quote(host.workdir)} && {source}{guarded}"
    return command, stdin_env


def _lane_prefix(host: RemoteHost, lane_count: int | None) -> str:
    """The inline (non-secret) lane env: ``CUDA_VISIBLE_DEVICES`` plus the
    ``CASCADE_LANE_*`` stamps on multi-lane pods (see build_remote_command)."""
    lane_env: dict[str, str] = {}
    if host.cuda_device is not None:
        lane_env["CUDA_VISIBLE_DEVICES"] = host.cuda_device
        device = str(host.cuda_device).strip()
        if lane_count is not None and lane_count > 1 and device.isdigit():
            lane_env["CASCADE_LANE_INDEX"] = device
            lane_env["CASCADE_LANE_COUNT"] = str(int(lane_count))
    if not lane_env:
        return ""
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(lane_env.items())) + " "


def _guarded_worker(prefix: str, argv: list[str]) -> str:
    """The worker wrapped in its own reaped process group (see the comment in
    build_remote_command): ``bash -c 'set -m; <worker> & wait; kill group; exit rc'``."""
    worker = f"{prefix}{shlex.join(argv)}"
    return "bash -c " + shlex.quote(
        f"set -m; {worker} </dev/null & w=$!; wait $w; rc=$?; "
        f"kill -KILL -- -$w 2>/dev/null; exit $rc")


def _stdin_env(env: dict[str, str]) -> str | None:
    if not env:
        return None
    return "".join(f"{k}={shlex.quote(v)}\n" for k, v in sorted(env.items()))


# ── detached dispatch (2026-09-18) ───────────────────────────────────────────
#
# The attached form above ties a leg's FATE to one ssh session: the worker's
# stdout (receipt), stderr (reason) and exit code all ride the session, so a
# transport drop — provider edge blip, keepalive starvation under load — turns
# a healthy, still-training worker into ``rc=255`` on the orchestrator's side
# (2026-09-13 5Co2Te after 1h39m; 2026-09-17 5DoJQ after 1h12m; 2026-09-18
# 02:52 all seven Oslo lanes at once, 8 legs lost and re-queued). The pod-side
# benches died the same way (u201, three pods). Detached dispatch decouples
# them: the launch ssh starts the worker under ``setsid nohup`` in its own
# session writing stdout/stderr/pid/exit_code into a per-leg run dir on the
# pod and returns at once; the orchestrator then POLLS with short ssh calls —
# an unreachable pod is retried for ``reattach_grace_seconds`` before the leg
# is declared lost — and fetches the receipt when the exit code lands. The
# credentials still travel on stdin only: the launcher sources them exported,
# the detached session inherits them, nothing touches the pod's disk.

DETACHED_RUN_ROOT = "_train_work/_dispatch"
DETACHED_POLL_SECONDS = 30
DETACHED_REATTACH_GRACE_SECONDS = 900
DETACHED_STDOUT_TAIL_BYTES = 262144
DETACHED_STDERR_TAIL_BYTES = 20000
DETACHED_LAUNCH_TOKEN = "__CASCADE_DETACHED__"
DETACHED_STDERR_MARK = "__CASCADE_DETACHED_STDERR__"
_SLEEP = time.sleep  # test seam


def detached_run_dir(host: RemoteHost, tag: str) -> str:
    """A fresh per-leg run dir under the pod's workdir (absolute)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:80]
    return f"{host.workdir}/{DETACHED_RUN_ROOT}/{safe}-{secrets.token_hex(4)}"


def build_detached_command(host: RemoteHost, body: str, run_dir: str,
                           *, stdin_env_present: bool) -> str:
    """The launcher the ssh session runs for a detached leg: cd to the workdir,
    source the credentials from stdin (exported, so the detached session
    inherits them — never written to disk), create ``run_dir``, start ``body``
    under ``setsid nohup`` with stdout/stderr redirected into the run dir,
    record its pid, and return. The detached script appends the exit code to
    ``run_dir/exit_code`` when ``body`` ends; ``DETACHED_LAUNCH_TOKEN`` on the
    launcher's stdout proves the launch happened."""
    rd = shlex.quote(run_dir)
    script = f"echo $$ > {rd}/pid; {body}; rc=$?; echo $rc > {rd}/exit_code"
    inner = "bash -c " + shlex.quote(script)
    source = "set -a && . /dev/stdin && set +a && " if stdin_env_present else ""
    return (f"cd {shlex.quote(host.workdir)} && {source}mkdir -p {rd} && "
            f"(setsid nohup {inner} >{rd}/stdout 2>{rd}/stderr </dev/null &) && "
            f"echo {DETACHED_LAUNCH_TOKEN}")


def _detached_poll_command(host: RemoteHost, run_dir: str) -> str:
    rd = shlex.quote(run_dir)
    return (f"cd {shlex.quote(host.workdir)} && if [ -f {rd}/exit_code ]; then "
            f"echo EXIT:$(cat {rd}/exit_code); elif [ -f {rd}/pid ] && "
            f"kill -0 $(cat {rd}/pid) 2>/dev/null; then echo RUNNING; else echo GONE; fi")


def _detached_find_command(host: RemoteHost, tag: str) -> str:
    """List prior run dirs of ``tag`` on the pod that still matter: the worker
    is alive (pid answers ``kill -0``) or finished (``exit_code`` written)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:80]
    return (f"cd {shlex.quote(host.workdir)} && for d in {DETACHED_RUN_ROOT}/{safe}-*; do "
            f"[ -d \"$d\" ] || continue; if [ -f \"$d/exit_code\" ]; then echo ATTACH:$d; "
            f"elif [ -f \"$d/pid\" ] && kill -0 $(cat \"$d/pid\") 2>/dev/null; then "
            f"echo ATTACH:$d; fi; done; true")


def find_detached_run(host: RemoteHost, tag: str, *, runner=None) -> str | None:
    """The run dir of a prior detached launch of ``tag`` on ``host`` whose
    worker is still running or has finished, or ``None``.

    A trainer restart (or a retried leg on a pod that outlived the previous
    attempt) must ATTACH to that run — a second ``setsid nohup`` of the same
    leg on the same GPU doubles the wall and wastes the first run (2026-09-24:
    seven legs re-dispatched after a restart while their originals kept
    training). Unreachable pod ⇒ ``None`` (the caller launches and the poll's
    grace handles the transport). Absolute path under ``host.workdir``."""
    run = runner or run_ssh
    try:
        p = run(build_ssh_argv(host, _detached_find_command(host, tag)), 60, "")
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0:
        return None
    hits = [ln.strip()[len("ATTACH:"):] for ln in (p.stdout or "").splitlines()
            if ln.strip().startswith("ATTACH:")]
    if not hits:
        return None
    rel = hits[-1]
    return rel if rel.startswith("/") else f"{host.workdir}/{rel}"


def _detached_fetch_command(host: RemoteHost, run_dir: str) -> str:
    rd = shlex.quote(run_dir)
    return (f"cd {shlex.quote(host.workdir)} && tail -c {DETACHED_STDOUT_TAIL_BYTES} {rd}/stdout; "
            f"printf '\\n{DETACHED_STDERR_MARK}\\n'; tail -c {DETACHED_STDERR_TAIL_BYTES} {rd}/stderr")


def _detached_kill_command(host: RemoteHost, run_dir: str) -> str:
    rd = shlex.quote(run_dir)
    return (f"cd {shlex.quote(host.workdir)} && p=$(cat {rd}/pid 2>/dev/null); "
            f"[ -n \"$p\" ] && {{ kill -KILL -- -$p 2>/dev/null; kill -KILL $p 2>/dev/null; }}; true")


def run_detached(host: RemoteHost, body: str, stdin_env: str | None, run_dir: str, *,
                 timeout: int, poll_seconds: int = DETACHED_POLL_SECONDS,
                 grace_seconds: int = DETACHED_REATTACH_GRACE_SECONDS,
                 runner=None, describe: str = "remote leg",
                 preempt: str = PREEMPT_BENCHMARKS, launch: bool = True) -> subprocess.CompletedProcess:
    """Run ``body`` detached on ``host`` and return a CompletedProcess-shaped
    result (returncode, stdout, stderr) exactly like the attached ssh would —
    the caller's receipt/rc handling is unchanged.

    ``runner(argv, timeout, stdin_text)`` is the ssh call (``run_ssh`` in
    production, a scripted double in tests). Transport failures during the
    poll are tolerated for ``grace_seconds`` of consecutive unreachability;
    the leg is only declared lost (rc=255) past that. ``timeout`` is the
    overall wall for the leg; on expiry the detached process group is killed
    best-effort and the leg reported timed out."""
    run = runner or run_ssh
    if launch:
        launch_cmd = preempt + build_detached_command(
            host, body, run_dir, stdin_env_present=stdin_env is not None)
        try:
            launched = run(build_ssh_argv(host, launch_cmd), 180, stdin_env)
        except (subprocess.TimeoutExpired, OSError) as e:
            raise RemoteDispatchError(f"{describe}: detached launch failed: {e}",
                                      returncode=255) from e
        if launched.returncode != 0 or DETACHED_LAUNCH_TOKEN not in (launched.stdout or ""):
            raise RemoteDispatchError(
                f"{describe}: detached launch failed (rc={launched.returncode}): "
                f"{(launched.stderr or '')[-400:]}", returncode=launched.returncode or 255)
    else:
        # ``launch=False``: ATTACH to a run already on the pod (find_detached_run)
        # — no second worker, no credentials sent; poll and fetch as usual.
        log.info("%s: attaching to the run already on the pod (%s)", describe, run_dir)
    t0 = time.time()
    last_seen = t0
    misses = 0
    rc: int | None = None
    vanished = False
    poll_argv = build_ssh_argv(host, _detached_poll_command(host, run_dir))
    while True:
        _SLEEP(poll_seconds)
        now = time.time()
        if now - t0 > timeout:
            with contextlib.suppress(Exception):  # best-effort
                run(build_ssh_argv(host, _detached_kill_command(host, run_dir)), 60, "")
            raise RemoteDispatchError(f"{describe} timed out after {int(timeout)}s (detached)",
                                      returncode=None)
        try:
            p = run(poll_argv, 60, "")
            lines = [ln.strip() for ln in (p.stdout or "").splitlines() if ln.strip()]
            token = lines[-1] if (p.returncode == 0 and lines) else None
        except (subprocess.TimeoutExpired, OSError):
            token = None
        if token is None or not token.startswith(("EXIT:", "RUNNING", "GONE")):
            misses += 1
            gap = now - last_seen
            if gap > grace_seconds:
                raise RemoteDispatchError(
                    f"{describe}: pod unreachable for {int(gap)}s while the detached "
                    f"worker ran ({misses} failed polls) — leg lost", returncode=255)
            if misses in (1, 5) or misses % 20 == 0:
                log.warning("%s: poll unreachable (%d in a row, %.0fs since last contact; "
                            "grace %ds) — the detached worker keeps running",
                            describe, misses, gap, grace_seconds)
            continue
        if misses:
            log.info("%s: pod reachable again after %d failed poll(s)", describe, misses)
        misses = 0
        last_seen = now
        if token == "RUNNING":
            continue
        if token == "GONE":
            vanished = True
            rc = 1
            break
        try:
            rc = int(token.split(":", 1)[1])
        except ValueError:
            rc = 1
        break
    fetch_argv = build_ssh_argv(host, _detached_fetch_command(host, run_dir))
    out = None
    for _attempt in range(6):
        try:
            f = run(fetch_argv, 180, "")
            if f.returncode == 0:
                out = f.stdout or ""
                break
        except (subprocess.TimeoutExpired, OSError):
            pass
        _SLEEP(20)
    if out is None:
        raise RemoteDispatchError(
            f"{describe}: detached worker finished (rc={rc}) but its output could not "
            f"be fetched after 6 attempts", returncode=rc)
    stdout, _, stderr = out.partition(DETACHED_STDERR_MARK)
    if vanished:
        stderr = stderr.rstrip() + "\n[detached worker process vanished without an exit code]"
    return subprocess.CompletedProcess(args=fetch_argv, returncode=rc,
                                       stdout=stdout, stderr=stderr)


def pinned_known_hosts_file(host: RemoteHost) -> Path:
    """A known_hosts file holding exactly ``host``'s pinned key (idempotent).

    Keyed on the pin itself so concurrent legs never share or clobber a file.
    """
    import hashlib
    import tempfile

    line = f"[{host.host}]:{host.port} {host.pinned_host_key.strip()}\n"
    path = Path(tempfile.gettempdir()) / (
        "cascade-pin-" + hashlib.sha256(line.encode()).hexdigest()[:20])
    if not path.is_file() or path.read_text() != line:
        path.write_text(line)
    return path


def ssh_transport_options(host: RemoteHost) -> list[str]:
    """The option argv (everything between the command name and the
    ``user@host`` target) shared by every ssh/scp to ``host``: port,
    BatchMode, the host-key policy, identity, then ``host.ssh_options``.
    Uses ssh's ``-p``; scp callers rewrite it to ``-P``."""
    if host.pinned_host_key:
        # ssh honours the FIRST value given for an option, so the pin goes
        # first and nothing later (host.ssh_options included) can loosen it.
        argv = ["-p", str(host.port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes",
                "-o", f"UserKnownHostsFile={pinned_known_hosts_file(host)}"]
    else:
        argv = ["-p", str(host.port), "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new"]
    if host.key_path:
        argv += ["-i", str(Path(host.key_path).expanduser())]
    for opt in host.ssh_options:
        argv += ["-o", opt]
    return argv


_JOB_CONTROL_LINE = re.compile(r"^\[\d+\][+-]?\s+(Exit|Done|Killed|Terminated|Stopped|Hangup)\b")
_REJECTION_TAG = "miner submission rejected:"


def rejection_reason(stderr: str | None) -> str:
    """The worker's one-line rejection reason out of a dispatch's stderr.

    The remote command runs the worker as a background job under ``set -m``
    (so its process group can be killed as a unit), and bash then reports the
    job's exit on stderr — ``[1]+  Exit 3  CUDA_VISIBLE_DEVICES=0 …`` — as the
    LAST line. Taking "the last stderr line" (the pre-2026-09-14 rule) relayed
    that shell notice instead of the worker's reason, and downstream the
    ``generator_stalled`` marker went missing: a stall on a slow host was
    classed as the miner's own generator fault (their shot spent) instead of
    the unburned stall class (2026-09-14 09:38, uid 78). Prefer the worker's
    tagged line; otherwise the last line that is not a job-control notice.
    """
    lines = [ln.rstrip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in reversed(lines):
        if _REJECTION_TAG in ln:
            return ln.split(_REJECTION_TAG, 1)[1].strip() or ln
    for ln in reversed(lines):
        if not _JOB_CONTROL_LINE.match(ln):
            return ln
    return "(no reason)"


def build_ssh_argv(host: RemoteHost, remote_command: str) -> list[str]:
    """The local ``ssh`` argv that runs ``remote_command`` on ``host``."""
    return ["ssh", *ssh_transport_options(host), f"{host.user}@{host.host}", remote_command]


def probe_worker_runtime(host: RemoteHost, *, required_flags: tuple[str, ...] = ("--local-only",),
                         timeout: float = 90.0) -> str:
    """Functional attestation that ``host`` runs THIS release's worker.

    Lium's launch API cannot pull by digest — :func:`cascade.provision.core.
    lium_image_ref` degrades the pin to ``repo[:tag]`` — and the env-based
    digest gate is circular there (``CASCADE_TRAIN_IMAGE_DIGEST`` is injected
    from the REQUESTED ref into whatever container boots, so it attests the
    request, not the runtime). A host with the repo's ``latest``/tag cached
    from an older release therefore boots stale code and still looks pinned
    (observed live 2026-09-10: a funded pod booted a pre-harvest worker and
    the dispatch died ``unrecognized arguments: --local-only``). This probe
    asks the pod's worker itself: its ``--help`` must know every flag the
    dispatch relies on. Returns ``""`` when the runtime checks out, else a
    reason string (the caller classifies it as an infra fault — the payer
    did nothing wrong; retry on another executor)."""
    cmd = f"{host.remote_python} -m cascade.trainer.worker --help"
    try:
        proc = subprocess.run(build_ssh_argv(host, cmd), capture_output=True,
                              text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return f"worker runtime probe timed out after {timeout:.0f}s"
    if proc.returncode != 0:
        return f"worker runtime probe failed rc={proc.returncode}: {(proc.stderr or '')[-200:]}"
    out = (proc.stdout or "") + (proc.stderr or "")
    missing = [f for f in required_flags if f not in out]
    if missing:
        return (f"pod booted a STALE worker image (missing worker flags {missing}); "
                "provider served a cached tag instead of the pinned digest")
    return ""


@dataclass(frozen=True)
class PodHygiene:
    """What else is going on on a pod at harvest time."""

    foreign_procs: tuple[str, ...] = ()   # processes not ours whose command names the checkpoint dir
    sessions: int = 0                     # interactive ssh sessions besides the probe's own
    ide_server: bool = False              # a remote IDE server is installed on the pod
    error: str = ""                       # probe transport failure: the checks are inconclusive


# Command-line markers of the processes a leg legitimately runs on its pod.
_OWN_PROCESS_MARKERS = ("cascade.trainer.worker", "cascade.trainer.sandbox",
                        "cascade-benchmark", "cascade.benchmarks", "wandb")
_HYGIENE_SESSIONS_MARK = "__HYGIENE_SESSIONS__"
_HYGIENE_IDE_MARK = "__HYGIENE_IDE__"
_IDE_SERVER_DIRS = ("/root/.vscode-server", "/root/.vscode-remote", "/root/.cursor-server")


def _pod_relative(host: RemoteHost, path: str) -> str:
    """``path`` as the worker names it: relative to the pod workdir."""
    prefix = host.workdir.rstrip("/") + "/"
    return path[len(prefix):] if path.startswith(prefix) else path


def _hygiene_command(host: RemoteHost, checkpoint_dir: str) -> str:
    needle = shlex.quote(_pod_relative(host, checkpoint_dir))
    dirs = " ".join(shlex.quote(d) for d in _IDE_SERVER_DIRS)
    return (f"ps -eo pid,args 2>/dev/null | grep -F -- {needle}; "
            f"echo {_HYGIENE_SESSIONS_MARK}; "
            f"ps -eo args 2>/dev/null | grep -cE '^sshd: [A-Za-z0-9_.-]+@'; "
            f"echo {_HYGIENE_IDE_MARK}; "
            f"for d in {dirs}; do [ -d \"$d\" ] && echo \"$d\"; done; true")


def parse_hygiene_output(stdout: str) -> PodHygiene:
    """Classify the probe's stdout (see :func:`_hygiene_command`)."""
    procs: list[str] = []
    sessions = 0
    ide = False
    section = "procs"
    for raw in (stdout or "").splitlines():
        line = raw.strip()
        if line == _HYGIENE_SESSIONS_MARK:
            section = "sessions"
            continue
        if line == _HYGIENE_IDE_MARK:
            section = "ide"
            continue
        if not line:
            continue
        if section == "procs":
            if any(m in line for m in _OWN_PROCESS_MARKERS):
                continue
            if _HYGIENE_SESSIONS_MARK in line or "grep -F" in line:
                continue          # the probe's own shell and grep
            procs.append(line)
        elif section == "sessions":
            with contextlib.suppress(ValueError):
                sessions = max(0, int(line) - 1)    # minus the probe's own session
        elif section == "ide":
            ide = True
    return PodHygiene(foreign_procs=tuple(procs), sessions=sessions, ide_server=ide)


def probe_pod_hygiene(host: RemoteHost, checkpoint_dir: str, *, timeout: float = 60.0,
                      runner=None) -> PodHygiene:
    """Look at the pod before its checkpoint is harvested: which processes
    other than ours name the checkpoint dir, how many interactive ssh
    sessions are open besides this probe's, and whether a remote IDE server
    is installed. A transport failure is reported in ``error`` (nothing is
    inferred from silence)."""
    run = runner or run_ssh
    try:
        p = run(build_ssh_argv(host, _hygiene_command(host, checkpoint_dir)), int(timeout), "")
    except subprocess.TimeoutExpired:
        return PodHygiene(error=f"probe timed out after {timeout:.0f}s")
    except OSError as e:
        return PodHygiene(error=f"probe failed: {e}")
    if p.returncode != 0:
        return PodHygiene(error=f"probe rc={p.returncode}: {(p.stderr or '')[-200:]}")
    return parse_hygiene_output(p.stdout or "")


HOST_BENCH_PROBE_TIMEOUT_SECONDS = 120.0
_HOST_BENCH_PROBE_PY = (
    "import json; from cascade.trainer.host_probe import host_bench; "
    "print('HOSTBENCH ' + json.dumps(host_bench('cuda')))"
)


def probe_host_bench(host: RemoteHost, *,
                     timeout: float = HOST_BENCH_PROBE_TIMEOUT_SECONDS) -> tuple[float | None, str]:
    """Run the fixed calibration bench (:func:`cascade.trainer.host_probe.
    host_bench`, ~5 s) on ``host``'s lane and return ``(tokens_per_s, "")``,
    or ``(None, why)`` when it could not be measured.

    Same workload the worker times before every run (``host_bench_tokens_per_s``
    in the run telemetry), so a pre-dispatch number is directly comparable to
    the fleet's history. The pod's own pinned code runs it — nothing is shipped
    to the pod — and ``CUDA_VISIBLE_DEVICES`` follows the host's lane so a
    multi-lane pod benches the device the leg will get."""
    env = ""
    if host.cuda_device is not None and str(host.cuda_device).strip():
        env = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(host.cuda_device).strip())} "
    cmd = f"cd {shlex.quote(host.workdir)} && {env}{host.remote_python} -c {shlex.quote(_HOST_BENCH_PROBE_PY)}"
    try:
        proc = subprocess.run(build_ssh_argv(host, cmd), capture_output=True,
                              text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, f"host bench probe timed out after {timeout:.0f}s"
    except OSError as e:
        return None, f"host bench probe could not start: {e}"
    if proc.returncode != 0:
        return None, f"host bench probe failed rc={proc.returncode}: {(proc.stderr or '')[-200:]}"
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("HOSTBENCH "):
            try:
                facts = json.loads(line[len("HOSTBENCH "):])
                return float(facts["host_bench_tokens_per_s"]), ""
            except (ValueError, KeyError, TypeError) as e:
                return None, f"host bench probe returned malformed facts: {e}"
    return None, "host bench probe printed no result"


def build_scp_argv(host: RemoteHost, local_path: str, remote_path: str) -> list[str]:
    """The local ``scp`` argv copying ``local_path`` to ``host:remote_path``
    under exactly :func:`build_ssh_argv`'s transport policy (a pinned host
    key stays pinned for file copies too)."""
    opts = ["-P" if a == "-p" else a for a in ssh_transport_options(host)]
    return ["scp", *opts, local_path, f"{host.user}@{host.host}:{remote_path}"]


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


@dataclass(frozen=True)
class LocalTrainReceipt:
    """A ``--local-only`` worker's receipt: a trained checkpoint still ON the
    pod, awaiting orchestrator harvest + verify + upload (DEC-CA-0036
    credential-free pods). Deliberately NOT a :class:`TrainedEntry` — the
    entry class refuses anything but a real hub pointer, which is the guard
    that keeps an un-harvested checkpoint out of a manifest."""

    miner_hotkey: str
    miner_uid: int
    role: str
    gen_ref: str
    corpus_digest: str
    train_block: int
    checkpoint_dir: str            # pod-local path of the trained checkpoint
    gpu_name: str = ""
    size: str = ""
    # {tensor file: sha256} hashed in memory at save time (empty: not reported)
    tensor_digests: dict[str, str] = field(default_factory=dict, hash=False)


def receipt_to_local(receipt: dict) -> LocalTrainReceipt:
    """Validate a ``--local-only`` receipt dict into a :class:`LocalTrainReceipt`."""
    try:
        return LocalTrainReceipt(
            miner_hotkey=str(receipt["miner_hotkey"]),
            miner_uid=int(receipt["miner_uid"]),
            role=str(receipt["role"]),
            gen_ref=str(receipt["gen_ref"]),
            corpus_digest=str(receipt["corpus_digest"]),
            train_block=int(receipt["train_block"]),
            checkpoint_dir=str(receipt["local_checkpoint_dir"]),
            gpu_name=str(receipt.get("gpu_name", "")),
            size=str(receipt.get("size", "")),
            tensor_digests={str(k): str(v) for k, v in
                            dict(receipt.get("tensor_digests") or {}).items()},
        )
    except (KeyError, ValueError, TypeError) as e:
        raise RemoteDispatchError(f"receipt is not a valid local-train receipt: {e}") from e


def harvest_remote_dir(host: RemoteHost, remote_dir: str, dest: Path | str,
                       *, timeout: int = 600, runner=None) -> Path:
    """Copy ``host:remote_dir`` into local ``dest`` (tar over ssh, ``dest``
    wiped first), under exactly the dispatch transport policy — a pinned host
    key stays pinned for the harvest. The credential-free pods' return
    channel: the pod pushes nothing; the orchestrator pulls.

    The stream comes off a MINER-controlled box, so extraction runs under
    :mod:`tarfile`'s ``"data"`` filter (PEP 706): absolute paths, ``..``
    traversal, symlinks/hardlinks escaping ``dest``, and device nodes are all
    refused — a hostile archive fails the harvest instead of writing outside
    it (or smuggling ``/etc`` into the later upload via a symlink)."""
    import shutil
    import tarfile

    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    # The worker runs under ``cd host.workdir`` (see build_remote_command), so a
    # ``--local-only`` receipt's checkpoint dir is relative to THAT — but this is
    # a fresh ssh session whose cwd is the login home. Resolve the harvest in the
    # same workdir (a no-op for an absolute remote_dir); without it ``tar -C`` hit
    # ``$HOME/<relative>``, which does not exist → an empty archive (live
    # 2026-09-10, the first harvest-mode funded round).
    tar_cmd = f"cd {shlex.quote(host.workdir)} && tar -C {shlex.quote(remote_dir)} -cf - ."
    argv = build_ssh_argv(host, tar_cmd)
    if runner is not None:  # test seam: (argv, timeout) → CompletedProcess-like
        proc = runner(argv, timeout)
        if proc.returncode != 0:
            raise RemoteDispatchError(
                f"checkpoint harvest from {host.name}:{remote_dir} failed "
                f"(rc={proc.returncode})", returncode=proc.returncode)
        return dest
    ssh = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=ssh.stdout, mode="r|") as tf:
            tf.extractall(dest, filter="data")
    except tarfile.TarError as e:
        ssh.kill()
        ssh.wait()
        raise RemoteDispatchError(
            f"checkpoint harvest from {host.name}:{remote_dir}: refused or "
            f"malformed archive: {e}") from e
    except Exception:
        ssh.kill()
        ssh.wait()
        raise
    finally:
        ssh.stdout.close()
    ssh_rc = ssh.wait(timeout=60)
    if ssh_rc != 0:
        err = ssh.stderr.read()[-400:]
        ssh.stderr.close()
        raise RemoteDispatchError(
            f"checkpoint harvest from {host.name}:{remote_dir} failed "
            f"(ssh rc={ssh_rc}): {err!r}", returncode=ssh_rc)
    ssh.stderr.close()
    return dest


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
    # Env vars forwarded to EVERY pod on top of each host's own ``forward_env``,
    # when present in the orchestrator env. The seam for observability creds the
    # operator shouldn't have to list per-host: the trainer sets it to
    # ``("WANDB_API_KEY",)`` when [wandb] is enabled, so pod-side wandb (where the
    # training actually runs) gets the key and its per-step logs land — instead of
    # silently no-opping because the key never left the orchestrator.
    extra_forward_env: tuple[str, ...] = ()
    # ``(remote_name, local_env_name)`` pairs forwarded ONLY to isolated (payer)
    # hosts — credentials minted for exposure to the payer, e.g. a project-scoped
    # wandb key delivered as WANDB_API_KEY ([wandb] funded_key_env). Never the
    # operator's own values: an isolated host still gets none of forward_env /
    # extra_forward_env.
    isolated_forward_env: tuple[tuple[str, str], ...] = ()
    # Harvester for ISOLATED hosts: ``harvest(host, receipt, *, base_seed,
    # repo_suffix) -> TrainedEntry``. An isolated host carries no upload
    # credential, so its legs are forced ``--local-only`` and the checkpoint is
    # pulled, verified and uploaded from the orchestrator (the same path payer
    # pods use). A caller that passes ``local_checkpoint=True`` itself gets the
    # raw receipt back and harvests on its own terms; a caller that does not,
    # on an isolated host, needs this — unset it and the dispatch fails
    # closed rather than launching a worker that cannot deliver its result.
    harvest: object = None
    # Detached dispatch (see run_detached): the worker runs in its own session
    # on the pod and the orchestrator polls; an ssh drop no longer kills the
    # leg. Off by default here (the attached form is the test fixture shape);
    # the live trainer arms it from ``[round] detached_dispatch``.
    detached: bool = False
    poll_seconds: int = DETACHED_POLL_SECONDS
    reattach_grace_seconds: int = DETACHED_REATTACH_GRACE_SECONDS
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
        anneal: bool = False,
        local_checkpoint: bool = False,
    ) -> TrainedEntry | LocalTrainReceipt:
        import os

        auto_harvest = False
        if host.isolated and not local_checkpoint:
            if self.harvest is None:
                raise RemoteDispatchError(
                    f"remote {role} on {host.name}: isolated host (no credential is "
                    "forwarded) needs a harvesting dispatcher — refusing to launch a "
                    "worker that could not upload its checkpoint")
            local_checkpoint = True
            auto_harvest = True
        argv = worker_argv(
            host, gen_ref=gen_ref, uid=uid, hotkey=hotkey, role=role,
            base_seed=base_seed, block=block, trainer_spec=self.trainer_spec,
            arch_preset=arch_preset, train_hours=train_hours, repo_suffix=repo_suffix,
            warm_start_ref=warm_start_ref, anneal=anneal, local_only=local_checkpoint,
        )
        # Per-host forwards plus the trainer's global extras (e.g. WANDB_API_KEY).
        # dict.fromkeys de-dups while preserving order if a host lists one too.
        if host.isolated:
            env = {remote: os.environ[local]
                   for remote, local in self.isolated_forward_env if local in os.environ}
        else:
            names = dict.fromkeys((*host.forward_env, *self.extra_forward_env))
            env = {k: os.environ[k] for k in names if k in os.environ}
        # Host-pinned values win over forwarded copies: a funded pod's
        # CASCADE_VAULT_DIR must be the POD's staging path even when the
        # orchestrator exports its own store dir under the same name.
        env.update(dict(host.static_env))
        log.info("dispatch role=%s → %s (%s) device=%s%s", role, host.name, host.host,
                 host.cuda_device, " [detached]" if self.detached else "")
        prior = None
        if self.detached:
            body = _guarded_worker(_lane_prefix(host, lane_count), argv)
            tag = f"{role}-{hotkey[:12]}-{base_seed}"
            prior = find_detached_run(host, tag, runner=self._runner)
            if prior:
                log.warning("dispatch role=%s → %s: a run of this leg is already on the "
                            "pod (%s) — attaching, not relaunching", role, host.name, prior)
            run_dir = prior or detached_run_dir(host, tag)
            proc = run_detached(
                host, body, _stdin_env(env), run_dir,
                timeout=self.timeout_seconds, poll_seconds=self.poll_seconds,
                grace_seconds=self.reattach_grace_seconds, runner=self._runner,
                describe=f"remote {role} on {host.name}", launch=prior is None)
        else:
            remote_cmd, stdin_env = build_remote_command(host, argv, env, lane_count=lane_count)
            ssh_argv = build_ssh_argv(host, remote_cmd)
            try:
                proc = (self._runner or run_ssh)(ssh_argv, self.timeout_seconds, stdin_env)
            except subprocess.TimeoutExpired as e:
                raise RemoteDispatchError(f"remote {role} on {host.name} timed out") from e
        if proc.returncode == 3:
            # Worker rc=3 = miner submission rejected (CorpusError): the
            # worker's one-line reason — no traceback to relay.
            raise RemoteDispatchError(
                f"remote {role} on {host.name}: miner submission rejected: "
                f"{rejection_reason(proc.stderr)}",
                returncode=3,
            )
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:]
            raise RemoteDispatchError(
                f"remote {role} on {host.name} failed (rc={proc.returncode}): {tail}",
                returncode=proc.returncode,
            )
        receipt = parse_receipt(proc.stdout or "")
        if (local_checkpoint and self.detached and prior
                and "local_checkpoint_dir" not in receipt and "trained_pointer" in receipt):
            # We attached to a run some EARLIER dispatch launched — one that
            # ran without --local-only and uploaded its checkpoint itself (a
            # pod adopted across a code change, e.g. a credentialed king pod
            # rented before the isolated-king rollout). Its receipt is judged
            # by its shape, not by how THIS dispatcher would have launched it:
            # a complete pushed receipt is a finished leg, and rejecting it
            # would throw the trained checkpoint away and retrain from scratch.
            log.warning("remote %s on %s: attached run %s was launched without "
                        "--local-only and uploaded its own checkpoint — accepting "
                        "its pushed receipt (no harvest)", role, host.name, prior)
            entry = receipt_to_entry(receipt)
            if entry.role != role:
                raise RemoteDispatchError(f"receipt role {entry.role!r} != dispatched {role!r}")
            return entry
        entry = (receipt_to_local(receipt) if local_checkpoint
                 else receipt_to_entry(receipt))
        if entry.role != role:
            raise RemoteDispatchError(f"receipt role {entry.role!r} != dispatched {role!r}")
        if auto_harvest:
            return self.harvest(host, entry, base_seed=base_seed, repo_suffix=repo_suffix)
        return entry


def run_ssh(ssh_argv: list[str], timeout: int, stdin_text: str | None = None):
    """Run the ssh command, returning the CompletedProcess (text mode).

    ``stdin_text`` (the credential payload from :func:`build_remote_command`)
    is piped to the remote command's stdin. Always piped — even when empty —
    so ssh never inherits the orchestrator's own stdin."""
    return subprocess.run(ssh_argv, capture_output=True, text=True, timeout=timeout,
                          input=stdin_text if stdin_text is not None else "")


_HEAT_PROBE_TOKEN = "cascade-heat-probe-ok"


def probe_host(host: RemoteHost, *, timeout: int = 15) -> bool:
    """A cheap liveness probe before heat fan-out: an SSH echo over
    ``BatchMode`` + key auth.

    TCP reachability is NOT sufficient — a booting pod completes the TCP
    handshake while sshd (or the pod account) is not ready yet, and every
    dispatch there dies ``rc=255`` (2026-07-23: a dead pod burned all 48
    challengers in 11s, each a one-and-done ``rc=255``, and the 0/48 heat was
    cached as legitimate). Only a full SSH round-trip proves the host can run a
    worker. A host with no address to reach (``host`` unset — offline fakes)
    cannot be probed and is KEPT, never stranding a fleet on a local config
    slip; a real :class:`RemoteHost` always carries one. Returns True only when
    the echo round-trips."""
    if not getattr(host, "host", None):
        return True
    try:
        proc = run_ssh(build_ssh_argv(host, shlex.join(["echo", _HEAT_PROBE_TOKEN])), timeout)
    except Exception:  # noqa: BLE001 — any transport failure ⇒ treat as dead
        return False
    return (getattr(proc, "returncode", 1) == 0
            and _HEAT_PROBE_TOKEN in (getattr(proc, "stdout", "") or ""))
