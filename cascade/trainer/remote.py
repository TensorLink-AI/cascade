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
import os
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
        # one line: keep the HEAD (what failed) and the tail (why) — a bare
        # ``s[-limit:]`` logged ``alse, 'ended': True})`` for a lost leg
        # (2026-11-04)
        head_n = max(1, limit // 2 - 2)
        return s[:head_n] + " … " + s[-(limit - head_n - 3):]
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


def worker_image_refusal(returncode, stdout: str) -> bool:
    """DEC-CA-0067: the worker's own structured FINAL image refusal — exit 2
    AND :data:`cascade.trainer.contract.IMAGE_REFUSAL_SENTINEL` as the LAST
    non-empty stdout line. Never inferred from free text (miner output,
    stderr tails). Workers that predate the sentinel never match."""
    from .contract import IMAGE_REFUSAL_SENTINEL

    if returncode != 2:
        return False
    lines = [ln.strip() for ln in str(stdout or "").splitlines() if ln.strip()]
    return bool(lines) and lines[-1] == IMAGE_REFUSAL_SENTINEL


class HarvestStreamError(RuntimeError):
    """The checkpoint harvest stream broke (truncated / malformed archive)
    and did not recover within :data:`HARVEST_STREAM_ATTEMPTS` attempts.

    ``pod_reachable`` says whether the pod still answered ssh after the LAST
    break: True → the checkpoint is still on the pod, so the leg handler keeps
    the pod for a re-attach instead of tearing it down (``keep_pod``); False →
    the pod is gone and the loss is classified from the payer's evidence as
    before. Raised as a :class:`RemoteDispatchError` subclass (below)."""


class RemoteDispatchError(RuntimeError):
    """An SSH dispatch or receipt parse failed.

    ``returncode`` carries the ssh exit status when the failure was a non-zero
    remote/transport exit — 255 marks an SSH transport failure (host
    unreachable, auth refused, sshd not ready): a POD fault, which drains the
    pod's lanes rather than blaming the leg. ``None`` for parse/timeout/logic
    failures that carry no ssh status."""

    def __init__(self, *args, returncode: int | None = None,
                 final_exception: str = "") -> None:
        super().__init__(*args)
        self.returncode = returncode
        # set by the dispatcher from the worker's exit + stdout only
        # (:func:`worker_image_refusal`) — never from message text
        self.image_refusal = False
        # The worker's final traceback exception line, parsed from its stderr
        # by the dispatcher (final_exception_line) — the ONLY input the shard
        # fault classifier trusts, never the message text, which carries a
        # stderr tail anything on the pod could have printed.
        self.final_exception = final_exception


HOST_STAGES = ("any", "heat", "final", "init")


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

    ``stage`` tags which legs the host may serve: ``"final"`` and ``"any"``
    entries are operator lanes / the rented-pod profile; ``"heat"`` is still
    accepted in hosts files (the retired screen stage) and never dispatched to.
    ``"init"`` is an INIT-ONLY lane: an operator-rented host the trainer uses
    solely to build the seed epoch's random-init checkpoint (the DEC-CA-0063
    pointer) — never a leg target, never counted as capacity, so no miner code
    runs there (``cascade.trainer.runner_rolling._RollingMixin._rolling_init_lanes``).
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
    stage: str = "any"                      # "any" | "final" ("heat" accepted, unused)
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
    allocation_id: str = ""
    allocation_provider: str = ""
    allocation_incident: str = ""


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
        stage = "any"       # "final" | "any" (operator lane); "init" = init-build-only
                            # lane (never a leg); "heat" is accepted, unused
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
        fwd = tuple(str(x) for x in h.get("forward_env", ()))
        if fwd and not bool(h.get("isolated", False)):
            secrets = [n for n in fwd if _CREDENTIAL_ENV_RE.search(n)]
            if secrets:
                log.warning("host %s forwards credential(s) %s into the pod's worker "
                            "environment — a pod can read them; use isolated = true "
                            "(orchestrator harvest, PR #320) unless this is deliberate",
                            h.get("name", "?"), ", ".join(secrets))
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
    contract_block: int | None = None,
    domain: str | None = None,
    expert_selection: dict | None = None,
    router_slow: bool = False,
    corpus_dir: str | None = None,
    finetune_recipe: str | None = None,
    continuation_root: str | None = None,
) -> list[str]:
    """The ``cascade.trainer.worker`` argv to run on the pod (no env/cd).

    ``arch_preset`` pins which configured size the pod trains (the primary size
    or one of ``[[training.sizes]]``); omitted ⇒ the worker trains the primary
    size, preserving single-size behaviour. ``train_hours`` overrides the compute
    budget (a reduced-budget leg, e.g. bench-anneal); ``repo_suffix``
    disambiguates the checkpoint repo so parallel same-size runs (one per
    challenger uid) don't collide. ``warm_start_ref`` is the era's pinned
    Cascade init (a trained_pointer);
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
    if continuation_root:
        argv += ["--continuation-root", continuation_root]
    if arch_preset:
        argv += ["--arch-preset", arch_preset]
    if train_hours is not None:
        argv += ["--train-hours", repr(float(train_hours))]
    if repo_suffix:
        # `=` form: the suffix starts with '-' (e.g. -u3), which argparse
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
    if contract_block is not None:
        # DEC-CA-0047: the era's contract block — the worker resolves the
        # scheduled contract switch from it, never from its launch block
        argv += ["--contract-block", str(int(contract_block))]
    if router_slow:
        # DEC-CA-0050 slow router (emitted only when true: every other leg's
        # argv is byte-identical)
        argv.append("--router-slow")
    if corpus_dir:
        # DEC-CA-0053 shards leg: corpus.json + per-shard download links
        argv += ["--corpus-dir", corpus_dir]
    if role == "expert":
        # DEC-CA-0050 lane-B leg: the target domain + the expert rows it owns
        if not domain or expert_selection is None:
            raise ValueError("role 'expert' needs domain and expert_selection")
        import json as _json

        argv += ["--domain", domain, "--selection-json",
                 _json.dumps({str(k): sorted(v) for k, v in sorted(expert_selection.items())},
                             separators=(",", ":"))]
    if role == "finetune":
        # DEC-CA-0056 lane-C leg: the recipe (validated JSON; the worker
        # re-validates it) — the base king rides --warm-start-ref
        if not finetune_recipe or not warm_start_ref:
            raise ValueError("role 'finetune' needs finetune_recipe and warm_start_ref")
        argv += ["--recipe-json", finetune_recipe]
    if role == "mkt_finetune":
        # DEC-CA-0064 MarketCast leg: {"marketcast": recipe, "has_generator"};
        # the pinned Timestral base rides --warm-start-ref; always local-only
        if not finetune_recipe or not warm_start_ref or not local_only:
            raise ValueError("role 'mkt_finetune' needs finetune_recipe, warm_start_ref and "
                             "local_only")
        argv += ["--recipe-json", finetune_recipe]
    if host.chain_toml:
        argv += ["--chain-toml", host.chain_toml]
    return argv


def receipt_role(dispatched: str) -> str:
    """The role a worker's receipt carries for a dispatched role: lane-B
    expert and lane-C fine-tune legs upload as challengers (their own
    suffixed repo)."""
    return ("challenger" if dispatched in ("expert", "finetune", "mkt_finetune")
            else dispatched)


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
    *, lane_count: int | None = None, wrap=None,
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
    if wrap is not None:                 # e.g. the shard prefetch (with_prefetch)
        guarded = wrap(guarded)
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


# Detached dispatch keeps a worker alive across SSH transport failures.
# setsid/nohup writes stdout, stderr, PID and exit code into a per-leg run
# directory; short polls retry unreachable pods for reattach_grace_seconds.
# Credentials arrive on stdin and are inherited through the environment,
# never written to the pod's disk.

DETACHED_RUN_ROOT = "_train_work/_dispatch"
# Environment names that look like secrets — flagged when a non-isolated host
# forwards them (see load_hosts).
_CREDENTIAL_ENV_RE = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD)", re.IGNORECASE)

DETACHED_POLL_SECONDS = 30
DETACHED_REATTACH_GRACE_SECONDS = 900
DETACHED_STDOUT_TAIL_BYTES = 262144
# Large enough to reach past bash's job notice and the trailing log lines to
# the worker's real Traceback (2026-11-01: a 20 KB tail held only the notice).
DETACHED_STDERR_TAIL_BYTES = 131072
DETACHED_LAUNCH_TOKEN = "__CASCADE_DETACHED__"
DETACHED_STDERR_MARK = "__CASCADE_DETACHED_STDERR__"
_SLEEP = time.sleep  # test seam


# How often a past-grace unreachable pod's provider listing is re-checked.
ALIVE_CHECK_SECONDS = 300


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


# The worker's own progress record (see cascade.trainer.loop: a small JSON
# file rewritten on every logged step), named through this env var and read
# back by the detached poll — the orchestrator's only live view of a remote
# leg's throughput (the step log otherwise reaches S3 only at flush).
PROGRESS_FILE_ENV = "CASCADE_PROGRESS_FILE"
PROGRESS_FILE_NAME = "progress.json"
DETACHED_PROGRESS_MARK = "PROGRESS:"
_PROGRESS_MAX_BYTES = 2000


def detached_progress_path(run_dir: str) -> str:
    return f"{run_dir}/{PROGRESS_FILE_NAME}"


def _progress_prefix(run_dir: str) -> str:
    """The inline (non-secret) env naming the leg's progress file."""
    return f"{PROGRESS_FILE_ENV}={shlex.quote(detached_progress_path(run_dir))} "


# 2026-11-13: three legs died with the pod dark 6-15 min and back with its run
# dir GONE (empty log tails) at steps 14400-16450 — container recreated, or a
# leak of ours killing it? Every poll now reads the pod's resources (host
# MemAvailable, the container cgroup's memory use/limit, free disk in the
# workdir, kernel uptime, pid 1's start) on one line before the token, so the
# lead-up to a death is on record and a restart/reboot is PROVEN, not guessed.
DETACHED_RESOURCES_MARK = "RES:"
# fields, in order: mem_avail_kb cg_mem_bytes cg_max uptime_s pid1_start_ticks disk_free_kb
_RESOURCES_SHELL = (
    "printf '" + DETACHED_RESOURCES_MARK + "%s %s %s %s %s %s\\n' "
    "\"$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)\" "
    "\"$(cat /sys/fs/cgroup/memory.current 2>/dev/null"
    " || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)\" "
    "\"$(cat /sys/fs/cgroup/memory.max 2>/dev/null"
    " || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)\" "
    "\"$(cut -d' ' -f1 /proc/uptime 2>/dev/null)\" "
    "\"$(awk '{print $22}' /proc/1/stat 2>/dev/null)\" "
    "\"$(df -Pk . 2>/dev/null | awk 'NR==2{print $4}')\"; "
)
# Resource samples kept per leg (one per poll: 30 s × 120 = the last hour);
# one INFO line of them every RESOURCE_LOG_SECONDS while the worker runs.
RESOURCE_HISTORY = 120
RESOURCE_LOG_SECONDS = 1800


def parse_resources(lines) -> dict | None:
    """The resource record a poll printed (:data:`DETACHED_RESOURCES_MARK`),
    or ``None``. Missing / unreadable fields are ``None``; never raises."""
    names = ("mem_avail_kb", "cg_mem_bytes", "cg_max", "uptime_s", "pid1_start", "disk_free_kb")
    for ln in lines:
        if not ln.startswith(DETACHED_RESOURCES_MARK):
            continue
        vals = ln[len(DETACHED_RESOURCES_MARK):].split(" ")
        out: dict = {}
        for name, raw in zip(names, vals + [""] * len(names), strict=False):
            raw = raw.strip()
            if name == "cg_max" and raw == "max":
                out[name] = "max"
                continue
            try:
                out[name] = float(raw) if name == "uptime_s" else int(raw)
            except ValueError:
                out[name] = None
        return out
    return None


def resource_verdict(history: list[dict]) -> str:
    """What the resource samples PROVE about a death (``""`` = nothing):
    pid 1 started again (container recreated) or the kernel uptime went
    backwards (host rebooted) between the first and the last sample."""
    seen = [r for r in history if r]
    if len(seen) < 2:
        return ""
    first, last = seen[0], seen[-1]
    up0, up1 = first.get("uptime_s"), last.get("uptime_s")
    if up0 is not None and up1 is not None and up1 < up0:
        return (f"HOST REBOOTED (kernel uptime {up0:.0f}s → {up1:.0f}s)")
    p0, p1 = first.get("pid1_start"), last.get("pid1_start")
    if p0 is not None and p1 is not None and p1 != p0:
        return f"CONTAINER RESTARTED (pid 1 start {p0} → {p1} ticks after boot)"
    return ""


def _fmt_resources(r: dict | None) -> str:
    if not r:
        return "n/a"

    def gib(v, scale):
        return "?" if not isinstance(v, int) else f"{v * scale / 2**30:.1f}G"

    lim = r.get("cg_max")
    lim_s = "max" if lim == "max" else gib(lim, 1)
    up = r.get("uptime_s")
    return (f"mem_avail={gib(r.get('mem_avail_kb'), 1024)} cg_mem={gib(r.get('cg_mem_bytes'), 1)}"
            f"/{lim_s} disk_free={gib(r.get('disk_free_kb'), 1024)} "
            f"uptime={'?' if up is None else f'{up:.0f}s'} pid1_start={r.get('pid1_start')}")


def _detached_postmortem_command(host: RemoteHost, run_dir: str) -> str:
    """Read-only facts from a pod whose worker vanished, taken BEFORE teardown:
    resources, boot / pid-1 times, whether the run dir survived, the kernel's
    OOM lines when readable. Best-effort; every step tolerates failure."""
    rd = shlex.quote(run_dir)
    return (f"cd {shlex.quote(host.workdir)} 2>/dev/null; " + _RESOURCES_SHELL
            + "echo \"boot: $(uptime -s 2>/dev/null)  now: $(date -u +%FT%TZ)\"; "
            "echo \"pid1: $(ps -o lstart= -p 1 2>/dev/null) $(cat /proc/1/comm 2>/dev/null)\"; "
            f"if [ -d {rd} ]; then echo \"run dir: present\"; ls -la {rd} 2>/dev/null | head -20; "
            "else echo \"run dir: MISSING\"; fi; "
            f"ls -la --time-style=+%FT%T {shlex.quote(DETACHED_RUN_ROOT)} 2>/dev/null | head -10; "
            "(dmesg -T 2>/dev/null || journalctl -k --no-pager 2>/dev/null) "
            "| grep -iE 'oom|killed process|out of memory|nvrm|xid' | tail -20; "
            "free -m 2>/dev/null; df -h . /dev/shm 2>/dev/null; true")


def _detached_poll_command(host: RemoteHost, run_dir: str) -> str:
    rd = shlex.quote(run_dir)
    pf = shlex.quote(detached_progress_path(run_dir))
    # A RUNNING worker also prints its progress record (when it writes one)
    # on the line BEFORE the token — the token stays the last line. Every poll
    # leads with the pod's resource line (:data:`_RESOURCES_SHELL`).
    return (f"cd {shlex.quote(host.workdir)} && " + _RESOURCES_SHELL
            + f"if [ -f {rd}/exit_code ]; then "
            f"echo EXIT:$(cat {rd}/exit_code); elif [ -f {rd}/pid ] && "
            f"kill -0 $(cat {rd}/pid) 2>/dev/null; then "
            f"[ -f {pf} ] && printf '{DETACHED_PROGRESS_MARK}%s\\n' "
            f"\"$(head -c {_PROGRESS_MAX_BYTES} {pf} | tr -d '\\n')\"; "
            f"echo RUNNING; else echo GONE; fi")


def parse_progress(lines) -> dict | None:
    """The progress record a RUNNING poll printed, or ``None``."""
    for ln in lines:
        if ln.startswith(DETACHED_PROGRESS_MARK):
            try:
                rec = json.loads(ln[len(DETACHED_PROGRESS_MARK):])
            except ValueError:
                return None
            return rec if isinstance(rec, dict) else None
    return None


class LegAborted(RemoteDispatchError):
    """The orchestrator aborted a running detached leg on purpose (the
    progress watchdog): the worker was killed, nothing about the run is the
    miner's fault."""


def _detached_find_command(host: RemoteHost, tag: str) -> str:
    """List prior run dirs of ``tag`` on the pod that still matter: the worker
    is alive (pid answers ``kill -0``) or finished CLEANLY (``exit_code`` 0 —
    a result to harvest). A FAILED run (exit != 0) is never re-attached: that
    only re-reports the old failure, so every retry on the lane "fails" in
    seconds without relaunching (2026-11-01 Targon lanes, ~30 legs)."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:80]
    return (f"cd {shlex.quote(host.workdir)} && for d in {DETACHED_RUN_ROOT}/{safe}-*; do "
            f"[ -d \"$d\" ] || continue; if [ -f \"$d/exit_code\" ]; then "
            f"[ \"$(cat \"$d/exit_code\")\" = 0 ] && echo ATTACH:$d; "
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


_POD_PROBE_TOKEN = "__CASCADE_POD_PROBE__"
# exit code 0, whitespace-insensitive (the launcher writes "0\n")
_RC_ZERO = '[ "$(tr -d \' \\n\' < "$d/exit_code")" = 0 ]'


@dataclass(frozen=True)
class PodRunsProbe:
    """What the detached run dirs on a pod say about it (the sweep's guard).

    ``reachable`` False = the probe could not run (transport failure,
    host-key refusal, no answer): unknown, never "idle". ``live`` = run dirs
    whose worker pid answers ``kill -0``; ``finished_ok`` = ``(run dir, age
    seconds since exit)`` of runs that ended with exit code 0 — their
    checkpoint is on the pod until the orchestrator harvests it (the caller
    knows which it harvested; the pod is never written to)."""

    reachable: bool
    live: tuple[str, ...] = ()
    finished_ok: tuple[tuple[str, float], ...] = ()
    error: str = ""


def _pod_runs_probe_command(host: RemoteHost) -> str:
    return (
        "if cd " + shlex.quote(host.workdir) + " 2>/dev/null; then now=$(date +%s); "
        "for d in " + DETACHED_RUN_ROOT + "/*; do [ -d \"$d\" ] || continue; "
        "if [ -f \"$d/exit_code\" ]; then "
        "if " + _RC_ZERO + "; then "
        "echo FINISHED:$(( now - $(stat -c %Y \"$d/exit_code\" 2>/dev/null || echo $now) )):$d; "
        "fi; "
        "elif [ -f \"$d/pid\" ] && kill -0 $(cat \"$d/pid\") 2>/dev/null; then echo LIVE:$d; fi; "
        "done; fi; echo " + _POD_PROBE_TOKEN)


def probe_pod_runs(host: RemoteHost, *, runner=None, timeout: int = 60) -> PodRunsProbe:
    """Ask ``host`` which detached runs on it still matter. Read-only: nothing
    on the pod changes. The pod sweep calls this before it may terminate a
    pod — a live worker or an unharvested finished run keeps the pod."""
    run = runner or run_ssh
    try:
        p = run(build_ssh_argv(host, _pod_runs_probe_command(host)), timeout, "")
    except (subprocess.TimeoutExpired, OSError) as e:
        return PodRunsProbe(reachable=False, error=f"{type(e).__name__}: {e}")
    lines = [ln.strip() for ln in (p.stdout or "").splitlines()]
    if p.returncode != 0 or _POD_PROBE_TOKEN not in lines:
        return PodRunsProbe(reachable=False,
                            error=f"rc={p.returncode} {(p.stderr or '').strip()[-200:]}")
    live: list[str] = []
    finished: list[tuple[str, float]] = []
    for ln in lines:
        if ln.startswith("LIVE:"):
            live.append(ln[len("LIVE:"):])
        elif ln.startswith("FINISHED:"):
            age, _, d = ln[len("FINISHED:"):].partition(":")
            try:
                finished.append((d, max(0.0, float(age))))
            except ValueError:
                finished.append((d, 0.0))   # unknown age: treated as fresh (kept)
    return PodRunsProbe(reachable=True, live=tuple(live), finished_ok=tuple(finished))


def _detached_fetch_command(host: RemoteHost, run_dir: str) -> str:
    rd = shlex.quote(run_dir)
    pf = shlex.quote(detached_progress_path(run_dir))
    return (f"cd {shlex.quote(host.workdir)} && tail -c {DETACHED_STDOUT_TAIL_BYTES} {rd}/stdout; "
            f"printf '\\n{DETACHED_STDERR_MARK}\\n'; tail -c {DETACHED_STDERR_TAIL_BYTES} {rd}/stderr; "
            f"printf '\\n{DETACHED_FETCH_PROGRESS_MARK}\\n'; "
            f"head -c {_PROGRESS_MAX_BYTES} {pf} 2>/dev/null; true")


# ── failed-leg evidence (2026-11-01: 03:18 and 00:03 incidents) ─────────────
# A failed leg's logged reason was the stderr TAIL — only bash's job notice
# ("[1]+ Exit N  CASCADE_PROGRESS_FILE=... < /dev/null") — and the pod was
# removed seconds later, so the real cause was gone. The failure reason now
# carries the last Traceback and the last error-signal lines, and the fetched
# tails are saved on the orchestrator box before the exception reaches any
# teardown.

DETACHED_FETCH_PROGRESS_MARK = "__CASCADE_DETACHED_PROGRESS__"
FAILED_LEG_LOG_MAX_BYTES = 262144
FAILED_LEG_LOGS_KEEP = 200
_TRACEBACK_HEAD = "Traceback (most recent call last):"
_FAULT_RE = re.compile(r"NCCL|\bKilled\b|\bOOM\b|out of memory|OutOfMemoryError|"
                       r"CUDA error|Segmentation fault|core dumped|oom-kill", re.IGNORECASE)
_LEVEL_RE = re.compile(r"\b(ERROR|CRITICAL|FATAL)\b")


def _is_signal_line(line: str) -> bool:
    """An error-signal line: ERROR/CRITICAL/FATAL as a log level (upper case
    only — ``error_class=`` is not a signal), or an NCCL / kill / OOM / CUDA
    fault in any case."""
    return bool(_LEVEL_RE.search(line) or _FAULT_RE.search(line))


def split_job_notices(stderr: str) -> tuple[list[str], list[str]]:
    """``(lines, job statuses)``: stderr without bash's job-control notice
    lines, and each notice's status word(s) (``"Exit 1"``, ``"Killed"``)
    without the long command line it echoes."""
    lines, statuses = [], []
    for ln in (stderr or "").splitlines():
        m = _JOB_CONTROL_LINE.match(ln.strip())
        if m:
            rest = ln.strip()[m.start(1):].split()
            word = rest[0]
            if word == "Exit" and len(rest) > 1 and rest[1].isdigit():
                word = f"Exit {rest[1]}"
            statuses.append(word)
            continue
        lines.append(ln.rstrip())
    return lines, statuses


def last_traceback(lines: list[str], *, max_lines: int = 60) -> list[str]:
    """The last ``Traceback (most recent call last):`` block of ``lines``,
    through its exception line (the first unindented line after the frames);
    the last ``max_lines`` of it when longer."""
    idx = max((i for i, ln in enumerate(lines) if ln.lstrip().startswith(_TRACEBACK_HEAD)),
              default=-1)
    if idx < 0:
        return []
    block = [lines[idx]]
    for ln in lines[idx + 1:]:
        block.append(ln)
        if ln and not ln[:1].isspace():
            break
    return block[-max_lines:]


def final_exception_line(stderr: str) -> str:
    """The exception line that closes the worker's LAST real traceback in
    ``stderr`` — ``""`` when there is none, or when the block is cut off
    before its exception line (the tail ended inside the frames)."""
    lines, _ = split_job_notices(stderr or "")
    tb = last_traceback(lines)
    if len(tb) < 2:
        return ""
    last = tb[-1]
    if not last.strip() or last[:1].isspace() or last.lstrip().startswith(_TRACEBACK_HEAD):
        return ""
    return last.strip()


def summarize_failure(stderr: str, *, rc, progress: dict | None = None,
                      tail_chars: int = 1500) -> tuple[str, str]:
    """``(cause, detail)`` for a failed leg. ``cause`` is one line — exit
    code, job status, last progress step and the most specific reason found
    (the last traceback's exception line, else the last error-signal line,
    else the last log line); ``detail`` is the traceback block, the last
    error-signal lines and the cleaned tail (job notices stripped)."""
    lines, statuses = split_job_notices(stderr)
    tb = last_traceback(lines)
    tb_set = set(tb)
    signals = [ln for ln in lines if _is_signal_line(ln) and ln not in tb_set][-8:]
    nonblank = [ln for ln in lines if ln.strip()]
    if tb:
        reason = tb[-1].strip()
    elif signals:
        reason = signals[-1].strip()
    else:
        reason = nonblank[-1].strip() if nonblank else "(no stderr)"
    head = [f"exit_code={rc}"]
    if statuses:
        head.append(f"job={statuses[-1]}")
    if progress and progress.get("step") is not None:
        head.append(f"last_step={progress.get('step')}")
    cause = " ".join(head) + ": " + reason[:400]
    detail: list[str] = []
    if tb:
        detail += ["--- last traceback ---", *tb]
    if signals:
        detail += ["--- error signals ---", *signals]
    tail = "\n".join(nonblank)[-tail_chars:]
    if tail:
        detail += ["--- stderr tail ---", tail]
    return cause, "\n".join(detail)


def save_failed_leg_evidence(root, hotkey: str, *, header: dict, stdout: str, stderr: str,
                             clock=time.time) -> str | None:
    """Write the fetched stdout + stderr tails of a failed leg to
    ``root/<hotkey12>-<utc-ts>.log`` (at most ``FAILED_LEG_LOG_MAX_BYTES``;
    the newest ``FAILED_LEG_LOGS_KEEP`` files are kept). Runs BEFORE the
    failure propagates to any teardown. Best-effort: returns the path, or
    None when it could not be written."""
    if not root:
        return None
    try:
        d = Path(root)
        d.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(float(clock())))
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(hotkey or "unknown"))[:12]
        path = d / f"{safe}-{ts}.log"
        n = 1
        while path.exists():
            path = d / f"{safe}-{ts}-{n}.log"
            n += 1
        budget = FAILED_LEG_LOG_MAX_BYTES // 2
        out_b = (stdout or "").encode("utf-8", "replace")[-budget:]
        err_b = (stderr or "").encode("utf-8", "replace")[-budget:]
        text = (json.dumps(header, sort_keys=True, default=str) + "\n"
                + "=== stdout (tail) ===\n" + out_b.decode("utf-8", "replace")
                + "\n=== stderr (tail) ===\n" + err_b.decode("utf-8", "replace") + "\n")
        data = text.encode("utf-8", "replace")
        if len(data) > FAILED_LEG_LOG_MAX_BYTES:
            data = data[:1024] + data[-(FAILED_LEG_LOG_MAX_BYTES - 1024):]
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        logs = sorted((p for p in d.glob("*.log") if p != path),
                      key=lambda p: (p.stat().st_mtime, p.name))
        for old in logs[:max(0, len(logs) + 1 - FAILED_LEG_LOGS_KEEP)]:
            with contextlib.suppress(OSError):
                old.unlink()
        return str(path)
    except Exception as e:  # noqa: BLE001 — evidence never masks the failure itself
        log.warning("failed-leg evidence for %s not saved: %s", str(hotkey)[:12], e)
        return None


def _detached_kill_command(host: RemoteHost, run_dir: str) -> str:
    """Kill a detached leg: its whole SESSION (the launcher's ``setsid``
    makes the recorded pid the session id; the guarded worker runs under
    ``set -m`` in its own process GROUP inside it, so a group kill of the
    pid misses it), then the shard prefetch's own session (with_prefetch).
    Each pid is first checked against ``/proc/<pid>/cmdline`` — the leg's
    shell names its run dir, the prefetch its script — so a pid the kernel
    reused for an unrelated process is never signalled."""
    rd = shlex.quote(run_dir)
    return (f"cd {shlex.quote(host.workdir)} && p=$(cat {rd}/pid 2>/dev/null); "
            f"[ -n \"$p\" ] && tr '\\0' ' ' </proc/$p/cmdline 2>/dev/null "
            f"| grep -qF -- {rd} && "
            f"{{ pkill -KILL -s $p 2>/dev/null; kill -KILL -- -$p 2>/dev/null; "
            f"kill -KILL $p 2>/dev/null; }}; "
            f"q=$(cat {rd}/{PREFETCH_PID_NAME} 2>/dev/null); "
            f"[ -n \"$q\" ] && {_prefetch_alive('$q')} && "
            f"{{ pkill -KILL -s $q 2>/dev/null; kill -KILL -- -$q 2>/dev/null; "
            f"kill -KILL $q 2>/dev/null; }}; true")


def _prefetch_alive(pid_expr: str) -> str:
    """A shell test: ``pid_expr`` is a live process whose argv[1] IS a
    prefetch.sh path (``bash <corpus>/prefetch.sh <ppid>``). Matching argv[1]
    alone, not the whole command line: every prefetch-wrapped leg shell
    (``bash -c "... $pfd/prefetch.sh ..."``) also holds that text, and a stale
    pid the kernel reused for another lane's leg shell must never pass."""
    return (f"tr '\\0' '\\n' </proc/{pid_expr}/cmdline 2>/dev/null "
            f"| sed -n 2p | grep -q '/prefetch[.]sh$'")


def _quarantine_run_host(host, describe: str, why: str) -> None:
    """Quarantine the host a detached run was on (2026-11-05 5G6jNi6Y: a pod
    went dark 49 min mid-run, then its worker vanished at step 15,600 — and the
    host stayed rentable, because only OUR watchdog's hang abort quarantined).
    Best-effort; an ADOPTED pod skips the host check, so the run already on
    that host is still re-attached and harvested."""
    ip = str(getattr(host, "host", "") or "")
    try:
        from .host_preflight import quarantine_host_ip

        quarantine_host_ip(ip, f"{describe}: {why}")
        log.warning("%s: host %s quarantined — %s", describe, ip, why)
    except Exception as e:  # noqa: BLE001 — never fails the poll
        log.warning("%s: host quarantine of %s failed (ignored): %s", describe, ip, e)


def run_detached(host: RemoteHost, body: str, stdin_env: str | None, run_dir: str, *,
                 timeout: int, poll_seconds: int = DETACHED_POLL_SECONDS,
                 grace_seconds: int = DETACHED_REATTACH_GRACE_SECONDS,
                 runner=None, describe: str = "remote leg",
                 preempt: str = PREEMPT_BENCHMARKS, launch: bool = True,
                 on_running=None, cancel=None,
                 on_progress=None, alive=None,
                 alive_extend_seconds: int = 0) -> subprocess.CompletedProcess:
    """Run ``body`` detached on ``host`` and return a CompletedProcess-shaped
    result (returncode, stdout, stderr) exactly like the attached ssh would —
    the caller's receipt/rc handling is unchanged.

    ``runner(argv, timeout, stdin_text)`` is the ssh call (``run_ssh`` in
    production, a scripted double in tests). Transport failures during the
    poll are tolerated for ``grace_seconds`` of consecutive unreachability;
    the leg is only declared lost (rc=255) past that — unless ``alive()`` (the
    provider's own listing of the pod: True alive / False gone / None unknown)
    still says alive, which extends it by up to ``alive_extend_seconds``. ``timeout`` is the
    overall wall for the leg; on expiry the detached process group is killed
    best-effort and the leg reported timed out.

    ``on_progress(record | None)`` runs on every poll that finds the worker
    RUNNING, with the worker's progress record (:func:`parse_progress`); a
    non-empty string back is an abort: the process group is killed and
    :class:`LegAborted` raised with that reason."""
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
    alive_checked = 0.0
    alive_state: bool | None = None
    misses = 0
    rc: int | None = None
    vanished = False
    last_progress: dict | None = None    # the worker's last step record seen by a poll
    resources: list[dict] = []           # the pod's resource samples (parse_resources)
    res_logged = 0.0
    dark_quarantined = False
    poll_argv = build_ssh_argv(host, _detached_poll_command(host, run_dir))
    while True:
        if cancel is not None and cancel.is_set():
            with contextlib.suppress(Exception):
                run(build_ssh_argv(host, _detached_kill_command(host, run_dir)), 60, "")
            raise RemoteDispatchError(f"{describe}: redundant leg cancelled", returncode=254)
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
                if not dark_quarantined:
                    # Dark past the grace mid-run: no new leg may rent this
                    # host, whatever the provider says about THIS pod.
                    dark_quarantined = True
                    _quarantine_run_host(host, describe,
                                         f"ssh dark {int(gap)}s mid-run (past the "
                                         f"{grace_seconds}s grace)")
                # 2026-11-04 uid 78: a pod that stopped answering ssh ~3.5 h into
                # its run was declared lost after 900 s and torn down, unharvested
                # — the run was finished or nearly so. While the PROVIDER still
                # lists the pod alive, keep it and keep polling (an ssh/network
                # outage is not a dead worker), up to ``alive_extend_seconds``
                # more; only a provider "gone" (or the extension spent) ends it.
                verdict = None
                if alive is not None and gap <= grace_seconds + alive_extend_seconds:
                    if now - alive_checked >= ALIVE_CHECK_SECONDS:
                        alive_checked = now
                        try:
                            alive_state = alive()
                        except Exception as e:  # noqa: BLE001 — unsure ⇒ keep the pod
                            log.warning("%s: provider liveness check failed (%s) — keeping "
                                        "the pod", describe, e)
                            alive_state = True
                    verdict = alive_state
                if verdict:
                    if misses % 10 == 1:
                        log.warning("%s: ssh unreachable for %.0fs but the provider still "
                                    "lists the pod alive — keeping it and polling (up to "
                                    "%ds past the %ds grace)", describe, gap,
                                    alive_extend_seconds, grace_seconds)
                    continue
                raise RemoteDispatchError(
                    f"{describe}: pod unreachable for {int(gap)}s while the detached "
                    f"worker ran ({misses} failed polls"
                    + ("; the provider no longer lists it alive" if verdict is False else "")
                    + ") — leg lost", returncode=255)
            if misses in (1, 5) or misses % 20 == 0:
                log.warning("%s: poll unreachable (%d in a row, %.0fs since last contact; "
                            "grace %ds) — the detached worker keeps running",
                            describe, misses, gap, grace_seconds)
            continue
        res = parse_resources(lines)
        if res is not None:
            res["t"] = round(now, 1)
            resources.append(res)
            del resources[:-RESOURCE_HISTORY]
        if misses:
            log.info("%s: pod reachable again after %d failed poll(s) — %s%s", describe, misses,
                     _fmt_resources(res),
                     (f" — {resource_verdict(resources)}" if resource_verdict(resources) else ""))
        elif res is not None and now - res_logged >= RESOURCE_LOG_SECONDS:
            res_logged = now
            log.info("%s: pod resources (step %s): %s", describe,
                     (last_progress or {}).get("step"), _fmt_resources(res))
        misses = 0
        last_seen = now
        if token == "RUNNING":
            seen = parse_progress(lines)
            if seen is not None:
                last_progress = seen
            if on_running is not None:
                on_running()
            if on_progress is not None:
                why = on_progress(seen)
                if why:
                    with contextlib.suppress(Exception):  # best-effort
                        run(build_ssh_argv(host, _detached_kill_command(host, run_dir)), 60, "")
                    raise LegAborted(f"{describe}: {why}", returncode=None)
            continue
        if token == "GONE":
            vanished = True
            rc = 1
            step = (last_progress or {}).get("step")
            verdict = resource_verdict(resources)
            _quarantine_run_host(host, describe, "detached worker vanished mid-run without "
                                 "an exit code" + (f" (last step {step})" if step is not None
                                                   else "")
                                 + (f" — {verdict}" if verdict else ""))
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
        step = (last_progress or {}).get("step")
        at = f", last_step={step}" if step is not None else ""
        raise RemoteDispatchError(
            f"{describe}: detached worker finished (rc={rc}{at}) but its output could not "
            f"be fetched after 6 attempts", returncode=rc)
    stdout, _, stderr = out.partition(DETACHED_STDERR_MARK)
    stderr, _, fetched = stderr.partition(DETACHED_FETCH_PROGRESS_MARK)
    progress = _parse_fetched_progress(fetched) or last_progress
    evidence = ""
    if vanished:
        # read-only, best-effort, AFTER the log fetch (the pod is torn down
        # seconds later): what happened to the box itself
        postmortem = ""
        try:
            pm = run(build_ssh_argv(host, _detached_postmortem_command(host, run_dir)), 90, "")
            postmortem = (pm.stdout or "")[-8000:]
        except Exception as e:  # noqa: BLE001 — evidence only, never fail the leg on it
            postmortem = f"(postmortem failed: {e})"
        # The failure TEXT stays exactly as before: the fault classifier
        # (faults.classify_*) reads it to decide infra vs miner billing, and a
        # dmesg "Killed process" line must never move that decision. The
        # evidence rides separately (``proc.postmortem``) into the saved log.
        stderr = stderr.rstrip() + "\n[detached worker process vanished without an exit code]"
        verdict = resource_verdict(resources)
        evidence = ("=== verdict: " + (verdict or "not proven by the resource samples") + " ===\n"
                    "=== pod postmortem (after the log fetch, before teardown) ===\n"
                    + postmortem.rstrip()
                    + "\n=== pod resources, last " + str(len(resources)) + " poll(s) ===\n"
                    + "\n".join(f"{r.get('t')} {_fmt_resources(r)}" for r in resources))
    proc = subprocess.CompletedProcess(args=fetch_argv, returncode=rc,
                                       stdout=stdout, stderr=stderr)
    proc.progress = progress          # the worker's last step record (or None)
    proc.postmortem = evidence
    return proc


def _parse_fetched_progress(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        rec = json.loads(text.splitlines()[0])
    except ValueError:
        return None
    return rec if isinstance(rec, dict) else None


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


# Money safety II: before a watchdog abort on a MINER's pod is booked as our
# fault (an operator make-good, a quarantined host), the pod is asked whether
# something other than our worker holds its GPUs (a miner's own GPU hog).
_GPU_PROCS_TOKEN = "__CASCADE_GPU_PROCS__"
_OUR_GPU_PROC_MARKERS = ("cascade.trainer.worker", "cascade/trainer/worker", "cascade-train-worker",
                         "torchrun", "cascade.trainer", "host_probe", "nccl_preflight")


def _gpu_procs_command() -> str:
    return ("for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do "
            "printf 'GPU:%s:' \"$p\"; tr '\\0' ' ' < /proc/$p/cmdline 2>/dev/null; echo; done; "
            f"echo {_GPU_PROCS_TOKEN}")


def parse_foreign_gpu_procs(stdout: str) -> list[str]:
    """``GPU:<pid>:<cmdline>`` lines whose cmdline is READABLE and not one of
    ours. Inside a container nvidia-smi often reports host-namespace PIDs
    whose ``/proc/<pid>/cmdline`` is empty or missing: that is UNKNOWN, never
    foreign (review H1a — our own worker must never read as a miner's hog)."""
    out = []
    for ln in (stdout or "").splitlines():
        if not ln.startswith("GPU:"):
            continue
        _tag, _pid, cmd = (ln.split(":", 2) + ["", ""])[:3]
        if not cmd.strip():
            continue                        # unreadable: unknown, not foreign
        if not any(m in cmd for m in _OUR_GPU_PROC_MARKERS):
            out.append(ln[4:][:200])
    return out


def probe_foreign_gpu_procs(host: RemoteHost, *, timeout: float = 60.0,
                            runner=None) -> tuple[list[str] | None, str]:
    """``(foreign GPU processes, "")`` on ``host`` — compute apps whose
    cmdline is not our worker's — or ``(None, why)`` when the pod could not
    be asked (unknown is never read as "clean" nor as "foreign")."""
    run = runner or run_ssh
    try:
        p = run(build_ssh_argv(host, _gpu_procs_command()), int(timeout), "")
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, f"{type(e).__name__}: {e}"
    if p.returncode != 0 or _GPU_PROCS_TOKEN not in (p.stdout or ""):
        return None, f"rc={p.returncode} {(p.stderr or '').strip()[-160:]}"
    return parse_foreign_gpu_procs(p.stdout or ""), ""


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


# ── multi-GPU (NCCL) preflight ──────────────────────────────────────────────
# Hosts that cannot do multi-GPU passed every other check (2026-10-31/11-01):
# one hung before its first step on NCCL "Cuda failure 999", one hung on a
# plain 8-GPU all_reduce, and 8xH200 VMs trained 5x slower comm-bound (~155 W
# per GPU vs ~375 W). The preflight runs an N-rank all_reduce of a fixed
# buffer on the pod's OWN torch (its pinned venv) under a hard timeout. The
# script is self-contained and travels on stdin — nothing of this release has
# to be in the pod's image for it to run.
NCCL_PREFLIGHT_MARK = "NCCLCHECK "
NCCL_PREFLIGHT_BUFFER_MB = 256
NCCL_PREFLIGHT_ITERS = 20
_NCCL_PREFLIGHT_PY = """\
import datetime, json, os, time
import torch
import torch.distributed as dist
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=__INIT__))
world = dist.get_world_size()
probe = torch.ones(1024, device="cuda")
dist.all_reduce(probe)
torch.cuda.synchronize()
ok = bool(torch.all(probe == world).item())
buf = torch.zeros(__MB__ * 1024 * 1024 // 4, device="cuda")
for _ in range(3):
    dist.all_reduce(buf)
torch.cuda.synchronize()
dist.barrier()
t0 = time.perf_counter()
for _ in range(__ITERS__):
    dist.all_reduce(buf)
torch.cuda.synchronize()
dt = torch.tensor([(time.perf_counter() - t0) / __ITERS__], device="cuda",
                  dtype=torch.float64)
dist.all_reduce(dt, op=dist.ReduceOp.MAX)
if dist.get_rank() == 0:
    nbytes = buf.numel() * buf.element_size()
    sec = max(float(dt.item()), 1e-9)
    algbw = nbytes / sec / 1e9
    print("NCCLCHECK " + json.dumps({"world": world, "bytes": nbytes, "seconds": sec,
          "algbw_gbps": algbw, "busbw_gbps": algbw * 2 * (world - 1) / world,
          "ok": ok}), flush=True)
dist.destroy_process_group()
"""


@dataclass(frozen=True)
class NcclPreflight:
    """One multi-GPU preflight. ``ok`` = the all_reduce completed with the
    right sum; ``inconclusive`` = the probe could not reach the pod at all
    (ssh transport), which says nothing about its GPUs."""

    ok: bool
    world: int = 0
    busbw_gbps: float | None = None
    error: str = ""
    inconclusive: bool = False


def nccl_preflight_script(*, init_timeout: int = 45, buffer_mb: int = NCCL_PREFLIGHT_BUFFER_MB,
                          iters: int = NCCL_PREFLIGHT_ITERS) -> str:
    return (_NCCL_PREFLIGHT_PY.replace("__INIT__", str(int(init_timeout)))
            .replace("__MB__", str(int(buffer_mb))).replace("__ITERS__", str(int(iters))))


def nccl_preflight_command(host: RemoteHost, nprocs: int, timeout: float) -> str:
    """The pod-side shell: stdin → a temp script, ``torchrun --standalone``
    with ``nprocs`` ranks under ``timeout -k`` (a hung collective is KILLed,
    and so is any rank it orphaned), temp file removed, the rank-0 exit code
    returned (124/137 = the hard timeout fired)."""
    env = ""
    if host.cuda_device is not None and str(host.cuda_device).strip():
        env = f"CUDA_VISIBLE_DEVICES={shlex.quote(str(host.cuda_device).strip())} "
    t = max(1, int(timeout))
    return (f"cd {shlex.quote(host.workdir)} && f=$(mktemp /tmp/cascade-nccl-XXXXXX.py) && "
            f"cat > \"$f\" && {env}OMP_NUM_THREADS=1 timeout -k 10 {t} "
            f"{host.remote_python} -m torch.distributed.run --standalone --nnodes=1 "
            f"--nproc_per_node={int(nprocs)} \"$f\"; rc=$?; "
            f"pkill -KILL -f \"$f\" 2>/dev/null; rm -f \"$f\"; exit $rc")


def probe_nccl_allreduce(host: RemoteHost, *, nprocs: int, timeout: float = 90.0,
                         runner=None) -> NcclPreflight:
    """Run the NCCL all_reduce preflight on ``host`` with ``nprocs`` ranks.

    ``runner(argv, timeout, stdin_text)`` is the ssh call (``run_ssh``; a
    scripted double in tests). A hang (the pod-side ``timeout`` or the local
    one firing) and any non-zero exit (NCCL/CUDA error) are failures; only an
    ssh transport failure (rc 255 / the ssh binary unavailable) is
    inconclusive."""
    run = runner or run_ssh
    script = nccl_preflight_script(init_timeout=max(10, int(timeout) - 15))
    argv = build_ssh_argv(host, nccl_preflight_command(host, nprocs, timeout))
    try:
        proc = run(argv, int(timeout) + 60, script)
    except subprocess.TimeoutExpired:
        return NcclPreflight(ok=False, world=nprocs,
                             error=f"NCCL all_reduce hung: no answer within {timeout:.0f}s")
    except OSError as e:
        return NcclPreflight(ok=False, error=f"preflight could not start: {e}",
                             inconclusive=True)
    rc = proc.returncode
    tail = error_tail((proc.stderr or "") + (proc.stdout or ""), 240)
    if rc == 255:
        return NcclPreflight(ok=False, error=f"ssh rc=255: {tail}", inconclusive=True)
    if rc in (124, 137):
        return NcclPreflight(ok=False, world=nprocs,
                             error=f"NCCL all_reduce hung: {nprocs}-rank collective killed "
                                   f"by the {timeout:.0f}s hard timeout ({tail})")
    if rc != 0:
        return NcclPreflight(ok=False, world=nprocs,
                             error=f"NCCL all_reduce failed rc={rc}: {tail}")
    for line in reversed((proc.stdout or "").splitlines()):
        i = line.find(NCCL_PREFLIGHT_MARK)
        if i < 0:
            continue
        try:
            facts = json.loads(line[i + len(NCCL_PREFLIGHT_MARK):])
            world, bw = int(facts["world"]), float(facts["busbw_gbps"])
        except (ValueError, KeyError, TypeError) as e:
            return NcclPreflight(ok=False, world=nprocs,
                                 error=f"NCCL preflight returned malformed facts: {e}")
        if not facts.get("ok"):
            return NcclPreflight(ok=False, world=world, busbw_gbps=bw,
                                 error="NCCL all_reduce returned a wrong sum")
        if world != int(nprocs):
            return NcclPreflight(ok=False, world=world, busbw_gbps=bw,
                                 error=f"NCCL preflight ran {world} rank(s), expected {nprocs}")
        return NcclPreflight(ok=True, world=world, busbw_gbps=bw)
    return NcclPreflight(ok=False, world=nprocs, error="NCCL preflight printed no result")


# The public-benchmark sidecar needs NVIDIA driver >= 580 (crusoe 535 and
# Warsaw 570 hosts fail every bench in about a minute; 2026-10-04 the era-2559
# king pod, driver 570, burned three payer verifications to "dropped").
BENCH_MIN_DRIVER_MAJOR = 580


def probe_driver_major(host: RemoteHost, *, timeout: float = 45.0) -> tuple[int | None, str]:
    """``(major, "")`` for ``host``'s NVIDIA driver (e.g. 570 for 570.211.01),
    or ``(None, why)`` when it could not be read."""
    cmd = "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
    try:
        proc = subprocess.run(build_ssh_argv(host, cmd), capture_output=True,
                              text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, f"driver probe timed out after {timeout:.0f}s"
    except OSError as e:
        return None, f"driver probe could not start: {e}"
    if proc.returncode != 0:
        return None, f"driver probe failed rc={proc.returncode}: {(proc.stderr or '')[-160:]}"
    for line in (proc.stdout or "").splitlines():
        head = line.strip().split(".")[0]
        if head.isdigit():
            return int(head), ""
    return None, "driver probe printed no version"


def probe_gpu_count(host: RemoteHost, *, timeout: float = 45.0) -> tuple[int | None, str]:
    """``(n, "")`` — the GPUs ``host``'s driver lists (``nvidia-smi -L``) —
    or ``(None, why)`` when it could not be read."""
    try:
        proc = subprocess.run(build_ssh_argv(host, "nvidia-smi -L"), capture_output=True,
                              text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return None, f"gpu count probe timed out after {timeout:.0f}s"
    except OSError as e:
        return None, f"gpu count probe could not start: {e}"
    if proc.returncode != 0:
        return None, f"gpu count probe failed rc={proc.returncode}: {(proc.stderr or '')[-160:]}"
    n = sum(1 for line in (proc.stdout or "").splitlines() if line.startswith("GPU "))
    return (n, "") if n else (None, "gpu count probe listed no GPUs")


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
    init_pointer: str | None = None
    training_contract_digest: str = ""
    # {tensor file: sha256} hashed in memory at save time (empty: not reported)
    tensor_digests: dict[str, str] = field(default_factory=dict, hash=False)
    # DEC-CA-0064 MarketCast leg only: {base_ref, base_sha256,
    # checkpoint_sha256, recipe_digest, data_digest}
    marketcast: dict = field(default_factory=dict, hash=False)


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
            init_pointer=receipt.get("init_pointer"),
            training_contract_digest=str(receipt.get("training_contract_digest", "") or ""),
            tensor_digests={str(k): str(v) for k, v in
                            dict(receipt.get("tensor_digests") or {}).items()},
            marketcast={str(k): str(v) for k, v in
                        dict(receipt.get("marketcast") or {}).items()},
        )
    except (KeyError, ValueError, TypeError) as e:
        raise RemoteDispatchError(f"receipt is not a valid local-train receipt: {e}") from e


class _ProgressReader:
    """The harvest's ssh stdout as seen by :mod:`tarfile`, counting bytes and
    stamping the last moment any arrived (the stall watchdog's input)."""

    def __init__(self, raw, clock=time.monotonic) -> None:
        self._raw = raw
        self._clock = clock
        self.nbytes = 0
        self.last_progress = clock()

    def read(self, n: int = -1) -> bytes:
        data = self._raw.read(n)
        if data:
            self.nbytes += len(data)
            self.last_progress = self._clock()
        return data


# A checkpoint harvest's hard wall (the whole tar stream) and its no-progress
# bound: a stalled or trickling stream off the pod must fail the harvest (the
# caller's infra path), never hang the leg — for the era king, forever.
HARVEST_TIMEOUT_SECONDS = 3600
HARVEST_STALL_SECONDS = 300.0
# Past ``timeout`` a stream that is still MOVING at a usable average rate keeps
# going (2026-11-03: a 14.8 GB make-good checkpoint off a ~2.6 MB/s host was
# killed at 3600 s with 14.68 GB received and the pod torn down — the finished
# run lost). The stall guard above still kills a dead stream within 300 s; a
# trickle below HARVEST_MIN_RATE_BYTES_PER_S fails at ``timeout`` as before, and
# HARVEST_HARD_CAP_SECONDS bounds every harvest absolutely.
HARVEST_MIN_RATE_BYTES_PER_S = 1_000_000
HARVEST_HARD_CAP_SECONDS = 6 * 3600


# The rate judged past ``timeout`` is the RECENT one (money safety II, #418
# follow-up): an average since start let a fast-start-then-trickle stream
# live ~3.9 h on its early bytes.
HARVEST_RATE_WINDOW_SECONDS = 600.0
# A broken stream (early EOF / malformed archive, NOT a watchdog breach) is
# retried while the pod still answers: up to this many attempts in all, with
# a short backoff. 2026-11-04: one truncated stream failed a FINISHED leg
# outright and the pod was torn down with the only checkpoint copy on it.
HARVEST_STREAM_ATTEMPTS = 3
HARVEST_STREAM_RETRY_BACKOFF_SECONDS = 20.0
HARVEST_REACH_PROBE_SECONDS = 45


def harvest_total_breach(elapsed: float, nbytes: int, timeout: float, *,
                         recent_rate: float | None = None) -> str | None:
    """Why a still-flowing harvest stream must stop now, or ``None``: past
    ``timeout`` only when its rate is below the floor — ``recent_rate`` (the
    last :data:`HARVEST_RATE_WINDOW_SECONDS`) when given, else the average —
    and always past the hard cap."""
    if elapsed > HARVEST_HARD_CAP_SECONDS:
        return (f"no complete archive within the {HARVEST_HARD_CAP_SECONDS}s hard cap "
                f"({nbytes} bytes received)")
    rate = recent_rate if recent_rate is not None else nbytes / max(elapsed, 1e-9)
    what = (f"last {HARVEST_RATE_WINDOW_SECONDS:.0f}s" if recent_rate is not None
            else "average")
    if elapsed > timeout and rate < HARVEST_MIN_RATE_BYTES_PER_S:
        return (f"no complete archive within {timeout}s ({nbytes} bytes received, "
                f"{rate / 1e6:.2f} MB/s {what} < "
                f"{HARVEST_MIN_RATE_BYTES_PER_S / 1e6:.2f} MB/s floor)")
    return None


class _RateWindow:
    """``(t, nbytes)`` samples → the byte rate over the last ``window`` s
    (None until the window is covered)."""

    def __init__(self, window: float = HARVEST_RATE_WINDOW_SECONDS) -> None:
        import collections

        self.window = float(window)
        self.samples = collections.deque()

    def add(self, t: float, nbytes: int) -> float | None:
        self.samples.append((t, nbytes))
        while len(self.samples) > 1 and t - self.samples[1][0] >= self.window:
            self.samples.popleft()
        t0, n0 = self.samples[0]
        if t - t0 < self.window:
            return None
        return (nbytes - n0) / max(t - t0, 1e-9)


def harvest_remote_dir(host: RemoteHost, remote_dir: str, dest: Path | str,
                       *, timeout: int = HARVEST_TIMEOUT_SECONDS,
                       stall_timeout: float = HARVEST_STALL_SECONDS,
                       runner=None, max_bytes: int | None = None) -> Path:
    """Copy ``host:remote_dir`` into local ``dest`` (tar over ssh, ``dest``
    wiped first), under exactly the dispatch transport policy — a pinned host
    key stays pinned for the harvest. The credential-free pods' return
    channel: the pod pushes nothing; the orchestrator pulls.

    The stream comes off a MINER-controlled box, so extraction runs under
    :mod:`tarfile`'s ``"data"`` filter (PEP 706): absolute paths, ``..``
    traversal, symlinks/hardlinks escaping ``dest``, and device nodes are all
    refused — a hostile archive fails the harvest instead of writing outside
    it (or smuggling ``/etc`` into the later upload via a symlink).

    Bounded three ways: ``stall_timeout`` seconds without a single byte;
    ``timeout`` seconds for the whole stream unless it is still averaging at
    least :data:`HARVEST_MIN_RATE_BYTES_PER_S`; and
    :data:`HARVEST_HARD_CAP_SECONDS` absolutely. A watchdog thread kills
    ssh on either breach, and the harvest raises :class:`RemoteDispatchError`."""
    import shutil
    import threading

    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    if runner is not None:
        dest.mkdir(parents=True)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
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
    # Saved robustly (2026-11-02): extracted into a sibling ``.partial-*``
    # dir, fsynced, then renamed onto ``dest`` — a half-written checkpoint
    # never looks complete, and any failure (ENOSPC mid-write included)
    # removes the partial copy instead of leaving it to fill the disk.
    work = dest.with_name(f"{dest.name}.partial-{os.getpid()}-{threading.get_ident()}")
    for attempt in range(1, HARVEST_STREAM_ATTEMPTS + 1):
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        try:
            _harvest_stream_into(host, remote_dir, argv, work, timeout=timeout,
                                 stall_timeout=stall_timeout, max_bytes=max_bytes)
            _fsync_tree(work)
            os.replace(work, dest)
            _fsync_dir(dest.parent)
            return dest
        except RemoteDispatchError as e:
            shutil.rmtree(work, ignore_errors=True)
            if not _is_broken_stream(e):
                raise                   # a watchdog breach / refusal: not retried
            reachable = remote_dir_reachable(host, remote_dir)
            if not reachable or attempt >= HARVEST_STREAM_ATTEMPTS:
                raise _HarvestStreamFailure(
                    f"{e} [stream attempt {attempt}/{HARVEST_STREAM_ATTEMPTS}; pod "
                    f"{'still reachable — kept' if reachable else 'unreachable'}]",
                    pod_reachable=reachable) from e
            log.warning("harvest from %s:%s: broken stream on attempt %d/%d (%s); pod still "
                        "answers — retrying in %.0fs", host.name, remote_dir, attempt,
                        HARVEST_STREAM_ATTEMPTS, error_tail(e, 160),
                        HARVEST_STREAM_RETRY_BACKOFF_SECONDS)
            time.sleep(HARVEST_STREAM_RETRY_BACKOFF_SECONDS)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
    raise AssertionError("unreachable")  # pragma: no cover


_BROKEN_STREAM_MARKERS = ("unexpected end of data", "truncated", "empty file", "empty header")


def _is_broken_stream(e: RemoteDispatchError) -> bool:
    """A stream that ENDED early (retryable: the pod or the link dropped) vs a
    watchdog breach or a refused archive (``..`` members, symlink escapes, a
    continuation over its storage limit — a hostile or oversized archive is
    never re-pulled)."""
    text = str(e)
    if "timed out" in text or "malformed archive" not in text:
        return False
    return any(m in text for m in _BROKEN_STREAM_MARKERS)


class _HarvestStreamFailure(RemoteDispatchError, HarvestStreamError):
    def __init__(self, msg: str, *, pod_reachable: bool) -> None:
        RemoteDispatchError.__init__(self, msg)
        self.pod_reachable = bool(pod_reachable)
        # harvest_guard.keeps_pod(): a reachable pod still holds the checkpoint
        self.keep_pod = bool(pod_reachable)


def remote_dir_reachable(host: RemoteHost, remote_dir: str, *, runner=None,
                         timeout: int = HARVEST_REACH_PROBE_SECONDS) -> bool:
    """Does ``host`` still answer ssh AND hold ``remote_dir``? Read-only."""
    run = runner or run_ssh
    cmd = f"cd {shlex.quote(host.workdir)} && test -d {shlex.quote(remote_dir)} && echo HARVEST_DIR_OK"
    try:
        p = run(build_ssh_argv(host, cmd), timeout, "")
        return p.returncode == 0 and "HARVEST_DIR_OK" in (p.stdout or "")
    except Exception:  # noqa: BLE001 — any failure to ask = not proven reachable
        return False


def _fsync_dir(path: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _fsync_tree(root: Path) -> None:
    """fsync every regular file and directory under ``root`` (durable before
    the rename publishes it)."""
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            fd = os.open(p, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        _fsync_dir(Path(dirpath))


def _harvest_stream_into(host, remote_dir: str, argv: list[str], dest: Path, *,
                         timeout: int, stall_timeout: float, max_bytes: int | None) -> None:
    """The tar-over-ssh stream of :func:`harvest_remote_dir`, extracted into
    ``dest`` under the ``"data"`` filter and both deadlines."""
    import tarfile
    import threading

    ssh = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stream = _ProgressReader(ssh.stdout)
    started = time.monotonic()
    stop = threading.Event()
    breach: list[str] = []

    rate_window = _RateWindow()

    def _watchdog() -> None:
        poll = max(0.05, min(5.0, float(stall_timeout) / 4))
        while not stop.wait(poll):
            now = time.monotonic()
            recent = rate_window.add(now, stream.nbytes)
            why = harvest_total_breach(now - started, stream.nbytes, timeout,
                                       recent_rate=recent)
            if why:
                breach.append(why)
            elif now - stream.last_progress > stall_timeout:
                breach.append(f"stream stalled: no bytes for {stall_timeout:.0f}s "
                              f"({stream.nbytes} bytes received)")
            if breach:
                with contextlib.suppress(OSError):
                    ssh.kill()
                return

    dog = threading.Thread(target=_watchdog, name="harvest-watchdog", daemon=True)
    dog.start()

    def _timed_out() -> RemoteDispatchError | None:
        if not breach:
            return None
        return RemoteDispatchError(
            f"checkpoint harvest from {host.name}:{remote_dir} timed out: {breach[0]}")

    try:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            extracted_bytes = 0
            def checked_member(member, directory):
                nonlocal extracted_bytes
                checked = tarfile.data_filter(member, directory)
                if max_bytes is not None and checked is not None:
                    if not (checked.isdir() or checked.isfile()):
                        raise tarfile.FilterError('continuation archive has non-regular members')
                    extracted_bytes += checked.size
                    if extracted_bytes > max_bytes:
                        raise tarfile.FilterError('continuation archive exceeds its storage limit')
                return checked
            tf.extractall(dest, filter=checked_member)
    except tarfile.TarError as e:
        stop.set()
        rc = ssh.poll()                 # already exited (pod gone / tar died)?
        ssh.kill()
        ssh.wait()
        err_tail = ""
        with contextlib.suppress(Exception):
            err_tail = (ssh.stderr.read() or b"").decode("utf-8", "replace").strip()[-300:]
        evidence = f" [ssh rc={rc if rc is not None else 'running'}" + (
            f"; stderr: {err_tail}]" if err_tail else "]")
        raise (_timed_out() or RemoteDispatchError(
            f"checkpoint harvest from {host.name}:{remote_dir}: refused or "
            f"malformed archive: {e}{evidence}", returncode=rc)) from e
    except Exception as e:
        stop.set()
        ssh.kill()
        ssh.wait()
        late = _timed_out()
        if late is not None:
            raise late from e
        raise
    finally:
        stop.set()
        ssh.stdout.close()
    dog.join(timeout=10)
    late = _timed_out()
    if late is not None:
        # Killed at a member boundary: the truncated stream can parse as a
        # complete (short) archive — never accept it as the checkpoint.
        ssh.wait()
        ssh.stderr.close()
        raise late
    try:
        ssh_rc = ssh.wait(timeout=60)
    except subprocess.TimeoutExpired as e:
        ssh.kill()
        ssh.wait()
        ssh.stderr.close()
        raise RemoteDispatchError(
            f"checkpoint harvest from {host.name}:{remote_dir}: ssh did not exit "
            f"after the archive") from e
    if ssh_rc != 0:
        err = ssh.stderr.read()[-400:]
        ssh.stderr.close()
        raise RemoteDispatchError(
            f"checkpoint harvest from {host.name}:{remote_dir} failed "
            f"(ssh rc={ssh_rc}): {err!r}", returncode=ssh_rc)
    ssh.stderr.close()


# The pod-side hash of a whole checkpoint dir (~14.5 GB: weights,
# weights_stable, optimizer) — sha256 at disk speed, well inside this bound.
POD_MANIFEST_TIMEOUT_SECONDS = 900


def remote_file_manifest(host: RemoteHost, remote_dir: str, *,
                         timeout: int = POD_MANIFEST_TIMEOUT_SECONDS,
                         runner=None) -> dict[str, tuple[int, str]]:
    """``{relative path: (size, sha256)}`` of every regular file under
    ``host:remote_dir``, as the POD reports it — the reference a harvested
    copy is checked against before the pod is released (see
    :mod:`cascade.trainer.harvest_release`). Resolved in ``host.workdir``
    exactly like :func:`harvest_remote_dir`. Raises
    :class:`RemoteDispatchError` on any ssh failure or unparseable output
    (a filename the listing cannot represent included): the caller then
    simply keeps the pod."""
    rd = shlex.quote(remote_dir)
    cmd = (f"cd {shlex.quote(host.workdir)} && cd {rd} && "
           "find . -type f -printf 'S %s %p\\n' && "
           "find . -type f -exec sha256sum -- {} +")
    argv = build_ssh_argv(host, cmd)
    try:
        proc = (runner(argv, timeout) if runner is not None else
                subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout, check=False))
    except subprocess.TimeoutExpired as e:
        raise RemoteDispatchError(
            f"pod checkpoint manifest of {host.name}:{remote_dir} timed out "
            f"after {timeout}s") from e
    if proc.returncode != 0:
        raise RemoteDispatchError(
            f"pod checkpoint manifest of {host.name}:{remote_dir} failed "
            f"(rc={proc.returncode}): {(proc.stderr or '')[-200:]!r}",
            returncode=proc.returncode)
    sizes: dict[str, int] = {}
    hashes: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if not line:
            continue
        if line.startswith("S "):
            size, _, path = line[2:].partition(" ")
            if not size.isdigit() or not path.startswith("./"):
                raise RemoteDispatchError(f"pod checkpoint manifest: bad size line {line[:120]!r}")
            sizes[path[2:]] = int(size)
            continue
        digest, sep, path = line.partition("  ")
        if (not sep or len(digest) != 64 or not path.startswith("./")
                or any(c not in "0123456789abcdef" for c in digest)):
            raise RemoteDispatchError(f"pod checkpoint manifest: bad hash line {line[:120]!r}")
        hashes[path[2:]] = digest
    if set(sizes) != set(hashes):
        raise RemoteDispatchError("pod checkpoint manifest: size and hash listings disagree")
    return {p: (sizes[p], hashes[p]) for p in sizes}


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
            init_pointer=receipt.get("init_pointer"),
            training_contract_digest=str(receipt.get("training_contract_digest", "") or ""),
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
    cancel_event: object = field(default=None, repr=False)
    continuation_contract: object = field(default=None, repr=False)
    continuation_marketcast_protocol: str | None = None
    continuation_root: str | None = None
    continuation_interval: int = 300
    continuation_max_bytes: int = 17179869184
    # Detached legs only: ``on_progress(record | None) -> abort reason | None``
    # on every RUNNING poll (the funded leg's progress watchdog — see
    # cascade.trainer.cost_guards.ProgressWatchdog).
    on_progress: object = field(default=None, repr=False)
    # Detached legs only: ``pod_alive() -> True | False | None`` — the
    # provider's listing of the leg's pod. A pod past the reattach grace that
    # the provider still lists alive is kept for up to
    # ``unreachable_alive_extend_seconds`` more (2026-11-04: an ssh outage
    # tore down a finished, unharvested run).
    pod_alive: object = field(default=None, repr=False)
    unreachable_alive_extend_seconds: int = 0
    # Where a failed leg's fetched stdout/stderr tails are saved on the
    # orchestrator box (save_failed_leg_evidence); None = not saved.
    evidence_dir: str | None = None
    # DEC-CA-0067: ``host_hook(host, contract_block) -> host`` run before every
    # dispatch (cascade.trainer.leg_pins.dispatch_host_hook) — installed only
    # while an automatic image repin is configured; None = unchanged dispatch.
    host_hook: object = field(default=None, repr=False)
    # Shards legs: start the pod-side corpus prefetch beside the worker
    # (prefetch_files / with_prefetch; [round] funded_prefetch_shards). Off by
    # default here so test doubles keep their exact ssh call sequence; the
    # live trainer arms it from config.
    prefetch_shards: bool = False
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
        contract_block: int | None = None,
        domain: str | None = None,
        expert_selection: dict | None = None,
        router_slow: bool = False,
        corpus_files: dict[str, str] | None = None,
        finetune_recipe: str | None = None,
        cancel=None,
    ) -> TrainedEntry | LocalTrainReceipt:
        import os

        cancel = cancel if cancel is not None else self.cancel_event
        if cancel is not None and cancel.is_set():
            raise RemoteDispatchError("redundant king replica cancelled", returncode=254)
        if self.host_hook is not None:
            # the leg's era image: right worker code, right deployed config
            host = self.host_hook(host, contract_block)
        auto_harvest = False
        if host.isolated and not local_checkpoint:
            if self.harvest is None:
                raise RemoteDispatchError(
                    f"remote {role} on {host.name}: isolated host (no credential is "
                    "forwarded) needs a harvesting dispatcher — refusing to launch a "
                    "worker that could not upload its checkpoint")
            local_checkpoint = True
            auto_harvest = True
        mirror = None
        use_continuation = bool(self.continuation_root)
        if use_continuation:
            from .continuation import PROTOCOL
            if role == 'mkt_finetune' and self.continuation_marketcast_protocol is not None:
                use_continuation = self.continuation_marketcast_protocol == PROTOCOL
            elif self.continuation_contract is not None:
                effective = self.continuation_contract.at_block(contract_block if contract_block is not None else block)
                use_continuation = getattr(effective, 'continuation_protocol', '') == PROTOCOL
        if use_continuation:
            if not self.detached:
                raise RemoteDispatchError('continuation requires detached dispatch')
            from .continuation_transport import mirror_for
            mirror = mirror_for(self, host, {"gen_ref": gen_ref, "hotkey": hotkey,
                "role": role, "seed": base_seed, "arch": arch_preset,
                "init": warm_start_ref, "contract_block": contract_block, "experts": expert_selection,
                "domain": domain, "recipe": ({k: v for k, v in json.loads(finetune_recipe).items() if k != "train_time"}
                    if role == "mkt_finetune" and finetune_recipe else finetune_recipe), "suffix": repo_suffix,
                "train_hours": train_hours, "router_slow": router_slow, "anneal": anneal})
        corpus_dir = None
        prefetch = None
        if corpus_files:
            corpus_tag = f"{role}-{hotkey[:12]}-{int(base_seed)}"
            corpus_dir = push_corpus_files(host, corpus_files, corpus_tag,
                                           runner=self._runner)
            if self.prefetch_shards:
                prefetch = prefetch_files(corpus_files, corpus_dir)

        def _arm_prefetch(command: str, run_dir: str | None = None) -> str:
            # Push + start the corpus prefetch beside a FRESH launch only (a
            # reattach must not rewrite a script a live prefetch is running).
            # Any failure here launches the worker exactly as before.
            if not prefetch:
                return command
            try:
                push_corpus_files(host, prefetch, corpus_tag, runner=self._runner)
            except (RemoteDispatchError, subprocess.TimeoutExpired, OSError) as e:
                log.warning("dispatch role=%s → %s: shard prefetch not started (%s) — "
                            "the worker streams shards itself", role, host.name, e)
                return command
            log.info("dispatch role=%s → %s: shard prefetch armed (%d shards) in %s",
                     role, host.name, prefetch[PREFETCH_LIST_NAME].count("\n"), corpus_dir)
            return with_prefetch(command, corpus_dir, run_dir)
        argv = worker_argv(
            host, gen_ref=gen_ref, uid=uid, hotkey=hotkey, role=role,
            base_seed=base_seed, block=block, trainer_spec=self.trainer_spec,
            arch_preset=arch_preset, train_hours=train_hours, repo_suffix=repo_suffix,
            warm_start_ref=warm_start_ref, anneal=anneal, local_only=local_checkpoint,
            contract_block=contract_block, domain=domain, expert_selection=expert_selection,
            router_slow=router_slow, corpus_dir=corpus_dir,
            **({"finetune_recipe": finetune_recipe} if finetune_recipe else {}),
            **({"continuation_root": mirror.remote_root} if mirror else {}),
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
            tag = f"{role}-{hotkey[:12]}-{base_seed}"
            if mirror is not None:
                tag += '-' + mirror.root.name[:12]
            elif cancel is not None:
                import hashlib
                tag += '-' + hashlib.sha256(json.dumps(argv).encode()).hexdigest()[:12]
            prior = find_detached_run(host, tag, runner=self._runner)
            if prior:
                log.warning("dispatch role=%s → %s: a run of this leg is already on the "
                            "pod (%s) — attaching, not relaunching", role, host.name, prior)
            run_dir = prior or detached_run_dir(host, tag)
            body = _guarded_worker(_lane_prefix(host, lane_count) + _progress_prefix(run_dir),
                                   argv)
            if prior is None:
                body = _arm_prefetch(body, run_dir)
            if mirror is not None and prior is None:
                mirror.restore()
            try:
                proc = run_detached(
                    host, body, _stdin_env(env), run_dir,
                timeout=self.timeout_seconds, poll_seconds=self.poll_seconds,
                grace_seconds=self.reattach_grace_seconds, runner=self._runner,
                    describe=f"remote {role} on {host.name}", launch=prior is None,
                    **({"on_running": mirror.poll} if mirror else {}),
                    **({"cancel": cancel} if cancel is not None else {}),
                    **({"on_progress": self.on_progress} if self.on_progress else {}),
                    **({"alive": self.pod_alive,
                        "alive_extend_seconds": int(self.unreachable_alive_extend_seconds)}
                       if self.pod_alive is not None else {}))
            finally:
                if mirror is not None:
                    mirror.close()
        else:
            remote_cmd, stdin_env = build_remote_command(
                host, argv, env, lane_count=lane_count, wrap=_arm_prefetch)
            ssh_argv = build_ssh_argv(host, remote_cmd)
            try:
                proc = (self._runner or run_ssh)(ssh_argv, self.timeout_seconds, stdin_env)
            except subprocess.TimeoutExpired as e:
                raise RemoteDispatchError(f"remote {role} on {host.name} timed out") from e
        where = ""
        progress = getattr(proc, "progress", None)
        if proc.returncode != 0:
            # Save the evidence BEFORE the failure reaches any teardown: the
            # pod (and its run dir) is usually gone seconds later.
            saved = save_failed_leg_evidence(
                self.evidence_dir, hotkey,
                header={"role": role, "host": host.name, "address": f"{host.host}:{host.port}",
                        "base_seed": base_seed, "exit_code": proc.returncode,
                        "progress": progress},
                stdout=proc.stdout or "",
                stderr=(proc.stderr or "") + (("\n" + proc.postmortem)
                                              if getattr(proc, "postmortem", "") else ""))
            where = f" [evidence: {saved}]" if saved else ""
        if proc.returncode == 3:
            # Worker rc=3 = miner submission rejected (CorpusError): the
            # worker's one-line reason — no traceback to relay.
            raise RemoteDispatchError(
                f"remote {role} on {host.name}: miner submission rejected: "
                f"{rejection_reason(proc.stderr)}{where}",
                returncode=3,
            )
        if proc.returncode != 0:
            cause, detail = summarize_failure(proc.stderr or "", rc=proc.returncode,
                                              progress=progress)
            err = RemoteDispatchError(
                f"remote {role} on {host.name} failed (rc={proc.returncode}) {cause}{where}"
                + (f"\n{detail}" if detail else ""),
                final_exception=final_exception_line(proc.stderr or ""),
                returncode=proc.returncode,
            )
            err.image_refusal = worker_image_refusal(proc.returncode, proc.stdout or "")
            raise err
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
            if entry.role != receipt_role(role):
                raise RemoteDispatchError(f"receipt role {entry.role!r} != dispatched {role!r}")
            return entry
        entry = (receipt_to_local(receipt) if local_checkpoint
                 else receipt_to_entry(receipt))
        if entry.role != receipt_role(role):
            raise RemoteDispatchError(f"receipt role {entry.role!r} != dispatched {role!r}")
        if auto_harvest:
            return self.harvest(host, entry, base_seed=base_seed, repo_suffix=repo_suffix)
        return entry


def push_corpus_files(host: RemoteHost, files: dict[str, str], tag: str, *,
                      runner=None) -> str:
    """Write a shards leg's ``corpus.json`` + ``urls.json`` (DEC-CA-0053) into
    ``<workdir>/_corpus/<tag>`` on the pod and return that dir. The links go
    over stdin, never on a command line."""
    pod_dir = f"{host.workdir}/_corpus/{tag}"
    run = runner or run_ssh
    for name, text in files.items():
        if "/" in name or name.startswith("."):
            raise RemoteDispatchError(f"bad corpus file name {name!r}")
        cmd = (f"mkdir -p {shlex.quote(pod_dir)} && "
               f"cat > {shlex.quote(pod_dir + '/' + name)}")
        proc = run(build_ssh_argv(host, cmd), 120, text)
        if proc.returncode != 0:
            raise RemoteDispatchError(
                f"pushing {name} to {host.name} failed: {(proc.stderr or '')[-300:]}")
    return pod_dir


# ── shard prefetch (trainer-side; works with any worker image) ──────────────
#
# A shards leg's worker streams its corpus from pre-signed links while it
# trains; a reader on an older image has no retry, so one bad download kills
# the leg mid-run (2026-11-02 c4-00030: step 7650/23841 on an 8xH200). The
# dispatcher therefore pushes a small pod-side script next to corpus.json /
# urls.json and starts it in the BACKGROUND alongside the worker. It fetches
# every shard in the reader's own consumption order (shard k of every child
# before shard k+1), each with bounded retries + timeouts, checks the bytes
# against corpus.json's sha256, writes to a hidden temp name and only then
# renames it into place. Why that is safe against a reader that is already
# running (cascade.generation.shards._child_iter, unchanged since the rc15
# image's 5e3fade):
#
# * the reader only ever opens ``<dir>/<name>``; a rename(2) within one
#   directory is atomic, so it sees either no file or the complete, verified
#   one — never a partial write (temps are ``.<name>.prefetch-<pid>``, which
#   no reader looks at and which cannot collide with get_file's ``.part``);
# * a shard already on disk is read in place, NOT downloaded (``start(i)``
#   returns None when the file exists; the plain path when all exist) — and
#   the plain path reads with verify=False, which is why the prefetch must
#   verify BEFORE the rename;
# * same bytes ⇒ same series in the same order: the prefetch never changes
#   what trains, only where the reader gets the bytes from;
# * a rename over a file the reader holds open leaves its fd on the old inode
#   (identical bytes), and the reader's unlink-after-read of a remote shard
#   at worst makes a lagging prefetch re-land a file no one reads again.
#
# The script never fails the leg: it skips with a WARNING when the pod lacks
# the disk, gives up a shard after its retries (the worker then fetches that
# one itself, as before), always exits 0, and is killed when the worker
# exits (or stops itself when the leg's shell is gone). A prefetch problem is
# therefore at worst the status quo — never a charge to the miner.

PREFETCH_SCRIPT_NAME = "prefetch.sh"
PREFETCH_PID_NAME = "prefetch.pid"      # in the leg's detached run dir
PREFETCH_LIST_NAME = "prefetch.tsv"
PREFETCH_PARALLEL = 8
PREFETCH_ATTEMPTS = 5
# Free disk the prefetch requires beyond the corpus's stored bytes: +10% and
# a flat 20 GiB for checkpoints/logs the leg writes on the same volume.
PREFETCH_MARGIN_FRACTION = 0.10
PREFETCH_MARGIN_BYTES = 20 * 1024 ** 3
_PREFETCH_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_PREFETCH_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def prefetch_need_bytes(stored_bytes: int) -> int:
    """Free bytes the pod needs before a prefetch is started."""
    return int(stored_bytes * (1.0 + PREFETCH_MARGIN_FRACTION)) + PREFETCH_MARGIN_BYTES


def prefetch_list(index: dict, urls: dict[str, str]) -> str:
    """``name<TAB>sha256<TAB>expiry<TAB>url`` per shard, in the reader's
    consumption order (round-robin over children by shard position — the
    order iter_corpus reaches them), so the prefetch lands ahead of it.
    Raises ``ValueError`` on anything the shell side could misparse."""
    from ..generation.transfer import link_expiry_epoch

    children = sorted(index.get("children") or [], key=lambda c: int(c["index"]))
    depth = max((len(c.get("shards") or []) for c in children), default=0)
    rows = []
    for k in range(depth):
        for c in children:
            shards = c.get("shards") or []
            if k >= len(shards):
                continue
            sh = shards[k]
            name, sha = str(sh.get("name", "")), str(sh.get("sha256", ""))
            url = urls.get(name)
            if url is None:
                continue                    # a local shard: nothing to fetch
            if not _PREFETCH_NAME_RE.match(name) or not _PREFETCH_SHA_RE.match(sha):
                raise ValueError(f"unexpected shard entry {name[:40]!r}")
            if not url.startswith(("https://", "http://")) or any(
                    ch in url for ch in '"\\\t\r\n '):
                raise ValueError(f"unexpected download link for {name}")
            rows.append(f"{name}\t{sha}\t{link_expiry_epoch(url)}\t{url}\n")
    return "".join(rows)


def prefetch_script(corpus_dir: str, *, need_bytes: int, n_shards: int,
                    parallel: int = PREFETCH_PARALLEL,
                    attempts: int = PREFETCH_ATTEMPTS,
                    floor_bytes: int | None = None) -> str:
    """The pod-side bash prefetch (see the block comment above). Argument 1
    is the pid of the leg's shell; the script stops itself once that is gone.
    Links are fed to curl on stdin (``-K -``), never on a command line, and
    are never printed — log lines carry shard names, codes and counts only."""
    d = shlex.quote(corpus_dir)
    need_kb = (int(need_bytes) + 1023) // 1024
    floor_kb = (int(PREFETCH_MARGIN_BYTES if floor_bytes is None else floor_bytes) + 1023) // 1024
    return f"""#!/usr/bin/env bash
# cascade shard prefetch — generated by cascade.trainer.remote; do not edit.
set -u
dir={d}
list="$dir/{PREFETCH_LIST_NAME}"
owner="${{1:-}}"
P={int(parallel)}; A={int(attempts)}; need_kb={need_kb}; floor_kb={floor_kb}; total={int(n_shards)}
say() {{ echo "cascade-prefetch: $*" >&2; }}
alive() {{ [ -z "$owner" ] || kill -0 "$owner" 2>/dev/null; }}
free_kb() {{ df -Pk "$dir" 2>/dev/null | awk 'NR==2{{print $4}}'; }}
if ! command -v curl >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
  say "WARNING: curl/sha256sum missing; prefetch skipped, the worker streams shards itself"; exit 0
fi
avail_kb=$(free_kb)
case "$avail_kb" in ''|*[!0-9]*)
  say "WARNING: free disk unreadable for $dir; prefetch skipped, the worker streams shards itself"; exit 0;;
esac
if [ "$avail_kb" -lt "$need_kb" ]; then
  say "WARNING: free disk ${{avail_kb}}KiB < needed ${{need_kb}}KiB; prefetch skipped, the worker streams shards itself"; exit 0
fi
rm -f "$dir"/.*.prefetch-* "$dir/.prefetch-failed" "$dir/.prefetch-expired" "$dir/.prefetch-stop" 2>/dev/null
: > "$dir/.prefetch-failed"
t0=$(date +%s)
say "start: $total shards, $P parallel, ${{avail_kb}}KiB free"
fetch() {{
  local name=$1 sha=$2 exp=$3 url=$4 f="$dir/$1" a=1 delay=2 code rc tmp
  if [ -f "$f" ] && [ "$(sha256sum < "$f" | cut -c1-64)" = "$sha" ]; then return 0; fi
  tmp="$dir/.$name.prefetch-$BASHPID"
  while :; do
    [ -e "$dir/.prefetch-expired" ] || [ -e "$dir/.prefetch-stop" ] && return 1
    alive || return 1
    # Re-checked before every try: other lanes' legs on the same volume
    # (or the worker's own checkpoints) may have used the room since start.
    local fk; fk=$(free_kb)
    case "$fk" in ''|*[!0-9]*) fk=0;; esac
    if [ "$fk" -lt "$floor_kb" ]; then
      [ -e "$dir/.prefetch-stop" ] || say "WARNING: free disk ${{fk}}KiB under the ${{floor_kb}}KiB floor; prefetch stopped, the worker streams the rest itself"
      : > "$dir/.prefetch-stop"; return 1
    fi
    rm -f "$tmp"
    code=$(printf 'url = "%s"\\n' "$url" | curl -sS -K - -o "$tmp" -w '%{{http_code}}' \\
      --connect-timeout 30 --max-time 900 --speed-limit 102400 --speed-time 60 2>/dev/null)
    rc=$?
    if [ "$rc" -eq 0 ] && [ "$code" = 200 ] \\
       && [ "$(sha256sum < "$tmp" | cut -c1-64)" = "$sha" ]; then
      mv -f "$tmp" "$f" && return 0
    fi
    rm -f "$tmp"
    if [ "$code" = 401 ] || [ "$code" = 403 ]; then
      if [ "$exp" -gt 0 ] && [ "$(date +%s)" -ge "$exp" ]; then
        : > "$dir/.prefetch-expired"
        say "WARNING: $name: download link EXPIRED (HTTP $code); prefetch stopped — links need re-signing"
        echo "$name" >> "$dir/.prefetch-failed"; return 1
      fi
    fi
    if [ "$a" -ge "$A" ]; then
      say "WARNING: $name: gave up after $A tries (curl rc=$rc, HTTP ${{code:-none}}); the worker fetches it itself"
      echo "$name" >> "$dir/.prefetch-failed"; return 1
    fi
    say "$name: try $a/$A failed (curl rc=$rc, HTTP ${{code:-none}}); retrying"
    sleep $(( delay / 2 + RANDOM % (delay / 2 + 1) ))
    a=$((a + 1)); delay=$((delay * 2)); [ "$delay" -gt 60 ] && delay=60
  done
}}
running=0
while IFS=$'\\t' read -r name sha exp url; do
  [ -n "$name" ] || continue
  alive || {{ say "leg shell gone; prefetch stopped"; kill 0; exit 0; }}
  [ -e "$dir/.prefetch-expired" ] || [ -e "$dir/.prefetch-stop" ] && break
  fetch "$name" "$sha" "$exp" "$url" &
  running=$((running + 1))
  if [ "$running" -ge "$P" ]; then wait -n; running=$((running - 1)); fi
done < "$list"
wait
nf=$(wc -l < "$dir/.prefetch-failed" | tr -d ' ')
say "done: $total shards in $(( $(date +%s) - t0 ))s, $nf not prefetched"
exit 0
"""


def prefetch_files(corpus_files: dict[str, str], corpus_dir: str) -> dict[str, str] | None:
    """The prefetch script + its shard list for a remote corpus, or ``None``
    (with a WARNING) when there is nothing to prefetch or the index cannot
    size it. ``None`` means: launch as before, the worker streams."""
    from ..generation.shards import INDEX_NAME, URLS_NAME

    if URLS_NAME not in corpus_files or INDEX_NAME not in corpus_files:
        return None
    try:
        index = json.loads(corpus_files[INDEX_NAME])
        urls = json.loads(corpus_files[URLS_NAME])
        if not isinstance(urls, dict):
            raise ValueError("urls.json is not a mapping")
        listing = prefetch_list(index, urls)
        stored = int(index.get("stored_bytes") or 0)
    except (ValueError, TypeError, KeyError) as e:
        log.warning("shard prefetch skipped for %s: %s — the worker streams shards itself",
                    corpus_dir, e)
        return None
    n = listing.count("\n")
    if not n:
        return None
    if stored <= 0:
        log.warning("shard prefetch skipped for %s: corpus.json carries no stored_bytes "
                    "to size the disk check — the worker streams shards itself", corpus_dir)
        return None
    # The per-try floor also reserves room for every in-flight download (P
    # parallel, sized at twice the mean shard) so P concurrent tries that all
    # passed the check cannot overshoot it.
    floor = PREFETCH_MARGIN_BYTES + PREFETCH_PARALLEL * 2 * -(-stored // n)
    script = prefetch_script(corpus_dir, need_bytes=prefetch_need_bytes(stored), n_shards=n,
                             floor_bytes=floor)
    return {PREFETCH_LIST_NAME: listing, PREFETCH_SCRIPT_NAME: script}


def _ere_escape(text: str) -> str:
    """``text`` as a literal POSIX extended regex (for ``pgrep -f``)."""
    out = []
    for c in text:
        if c in ".[]()*+?{}|$":
            out.append(f"[{c}]")
        elif c in "^\\":
            out.append("\\" + c)
        else:
            out.append(c)
    return "".join(out)


def with_prefetch(command: str, corpus_dir: str, run_dir: str | None = None) -> str:
    """``command`` (the guarded worker) with the corpus prefetch started
    beside it in its OWN session (``setsid``: its own process group, so the
    worker's group reap never touches it and it is killed as a group the
    moment the worker exits). The worker's exit code is preserved; the
    prefetch's never matters.

    A prefetch of the same corpus left over from an earlier launch (a killed
    or timed-out leg) is killed first, by group. The pattern is a regex whose
    ``[.]`` cannot match this launching shell's own command line, and the
    script is started as ``"$pfd/prefetch.sh"`` so that command line never
    holds the expanded path either: the sweep never hits the leg itself.
    With ``run_dir`` the new prefetch's pid is recorded there for
    :func:`_detached_kill_command`."""
    script = f"{corpus_dir}/{PREFETCH_SCRIPT_NAME}"
    pattern = shlex.quote("bash " + _ere_escape(script))
    sweep = (f"for q in $(pgrep -f {pattern} 2>/dev/null); do "
             f"kill -KILL -- -$q 2>/dev/null; kill -KILL $q 2>/dev/null; done; ")
    record = (f"echo $pf > {shlex.quote(f'{run_dir}/{PREFETCH_PID_NAME}')}; "
              if run_dir else "")
    return (f"{{ pfd={shlex.quote(corpus_dir)}; {sweep}"
            f"setsid bash \"$pfd/{PREFETCH_SCRIPT_NAME}\" $$ </dev/null >/dev/null & "
            f"pf=$!; {record}"
            f"{command}; rc=$?; "
            f"{_prefetch_alive('$pf')} && {{ pkill -TERM -s $pf 2>/dev/null; "
            f"kill -TERM -- -$pf 2>/dev/null; kill -TERM $pf 2>/dev/null; }}; (exit $rc); }}")


def run_ssh(ssh_argv: list[str], timeout: int, stdin_text: str | None = None):
    """Run the ssh command, returning the CompletedProcess (text mode).

    ``stdin_text`` (the credential payload from :func:`build_remote_command`)
    is piped to the remote command's stdin. Always piped — even when empty —
    so ssh never inherits the orchestrator's own stdin."""
    return subprocess.run(ssh_argv, capture_output=True, text=True, timeout=timeout,
                          input=stdin_text if stdin_text is not None else "")
