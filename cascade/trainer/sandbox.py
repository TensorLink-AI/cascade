"""Network-isolated, rlimited subprocess sandbox for running a generator.

:func:`cascade.trainer.corpus.build_corpus` imports and executes
miner-controlled code. In production that must NOT run in the trainer's own
process: a hostile generator could read the trainer's secrets, reach the
network, fork-bomb, or exhaust memory. :func:`run_in_sandbox` runs the *same*
build in a fresh interpreter that is:

* **out-of-process** — a crash, hang, or memory blow-up can't take the trainer
  with it, and the child can't touch the parent's objects;
* **secret-free** — the child gets a minimal env allowlist, so HF/chain tokens
  in the trainer's environment are never visible to miner code;
* **rlimited** (POSIX) — address space (``max_memory_mb``), CPU seconds
  (``max_generate_seconds``), core dumps (off), and output file size are capped
  before ``exec``;
* **wall-clock bounded** — a hard ``communicate`` timeout backs up RLIMIT_CPU;
* **network-isolated** — wrapped in a network namespace via ``unshare`` when the
  host allows it (probed once, with graceful fallback), and Python-level
  networking is disabled in the child as defense-in-depth on top of the
  submit-time static guard.

Only validated ``float64`` arrays cross back, via a temp ``.npz`` loaded with
``allow_pickle=False`` — never a pickle of untrusted output. The parent
re-derives :func:`corpus_digest` from the returned arrays and rejects any
mismatch, so corruption or tampering in transit can't slip through.

The module doubles as the child entry point: ``python -m
cascade.trainer.sandbox <repo> <seed> <cfg_json> <out_dir>``.

Caveat: RLIMIT_AS caps *virtual* memory. numpy/scipy generators fit the default
4 GiB comfortably; a torch generator reserves far more address space than it
uses, so raise ``[generator] max_memory_mb`` for model generators.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import select
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..shared.config import GeneratorConfig
from ..shared.manifest import corpus_digest
from .corpus import CorpusError, CorpusResult, build_corpus, resolve_real_corpus

log = logging.getLogger("cascade.trainer.sandbox")

# Minimal env passed to the child — everything else (tokens, cloud creds) is
# stripped so untrusted generator code never sees the trainer's secrets.
_SAFE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")

# Extra env a GPU-resident (torch) generator needs to reach CUDA. Passed through
# only in the stream_gpu profile. None of these carry secrets; they select/locate
# the GPU and its libraries.
_GPU_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "CUDA_HOME", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH", "PYTORCH_CUDA_ALLOC_CONF",
)

# The env vars every mainstream BLAS/threadpool implementation honours. Set on
# the child only when a lane slice is active (see _lane_cpu_slice), so a
# generator's thread pool matches the cores it can actually run on.
_BLAS_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)

# Lane-geometry env stamped into each worker lane's environment at DISPATCH
# (see cascade.trainer.remote.build_remote_command). Only the orchestrator
# knows a pod's lane fan-out — the pod-side view (a masked
# CUDA_VISIBLE_DEVICES, torch seeing one device) can't distinguish a
# single-GPU pod from one lane of eight.
LANE_INDEX_ENV = "CASCADE_LANE_INDEX"
LANE_COUNT_ENV = "CASCADE_LANE_COUNT"

_NETNS_PROBE: bool | None = None
_MEMSCOPE_PROBE: bool | None = None


# ───────────────────────────────── parent ──────────────────────────────────


def _lane_cpu_slice() -> tuple[set[int], int] | None:
    """This worker lane's contiguous CPU-core slice, from dispatch-injected env.

    On a multi-GPU cluster pod the orchestrator runs one worker lane per GPU
    (``CUDA_VISIBLE_DEVICES=<lane>``) but the kernel gives every lane ALL
    cores, so one lane's multithreaded generator can starve its neighbours'
    training loops. The dispatcher — the only party that knows the pod's lane
    fan-out — stamps :data:`LANE_INDEX_ENV`/:data:`LANE_COUNT_ENV` into each
    lane's env (see ``remote.build_remote_command``); lane ``i`` of ``N`` then
    owns the contiguous cores ``[i*k, (i+1)*k)`` with ``k = max(1, cpu_count
    // N)``, so the split is deterministic and equal.

    The total-core figure is ``len(os.sched_getaffinity(0))``, not
    ``os.cpu_count()``: on docker-template pods (lium) ``cpu_count`` reports
    the HOST's cores even when the container is cgroup/cpuset-limited, while
    the affinity set reflects what this process may actually use — and the
    slice is carved out of the *allowed* core ids, so it stays inside any
    pre-existing container limit.

    Returns ``(cores, slice_size)``, or ``None`` when the env is absent or
    malformed (local runs, single-lane pods, tests) — callers then keep the
    uncapped legacy behavior. With more lanes than cores the slice wraps
    modulo the core count so every lane still gets at least one VALID core
    (an out-of-range set would make ``sched_setaffinity`` fail).
    """
    idx_s = os.environ.get(LANE_INDEX_ENV, "")
    cnt_s = os.environ.get(LANE_COUNT_ENV, "")
    if not (idx_s.isdigit() and cnt_s.isdigit()):
        return None
    idx, cnt = int(idx_s), int(cnt_s)
    if cnt <= 1 or idx >= cnt:  # single lane, or inconsistent geometry
        return None
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # non-Linux: no affinity API
        allowed = list(range(os.cpu_count() or 1))
    ncpu = len(allowed) or 1
    k = max(1, ncpu // cnt)
    start = (idx * k) % ncpu
    cores = set(allowed[start:start + k])
    return cores, len(cores)


def _memory_scope_available() -> bool:
    """Probe (once) whether systemd-run scopes can wrap a sandbox child.

    The GPU profile skips RLIMIT_AS (torch's virtual-address over-reserve
    would false-kill it), which leaves host RAM uncapped — a ballooning
    generator gets the TRAINER OOM-killed, not itself. A ``systemd-run
    --scope`` cgroup caps RESIDENT memory instead, so torch's VA reservation
    is harmless while a real balloon dies in its own scope. Inside
    docker-template pods there is no systemd, so the probe requires both the
    binary and a successful trial scope; on failure this degrades to current
    behavior with ONE warning naming the risk (the probe is cached, and only
    the GPU-profile wrap path consults it).
    """
    global _MEMSCOPE_PROBE
    if _MEMSCOPE_PROBE is None:
        try:
            _MEMSCOPE_PROBE = shutil.which("systemd-run") is not None and subprocess.run(
                ["systemd-run", "--scope", "--quiet", "true"],
                capture_output=True, timeout=10,
            ).returncode == 0
        except Exception:  # noqa: BLE001 - any failure means "no scopes here"
            _MEMSCOPE_PROBE = False
        if not _MEMSCOPE_PROBE:
            log.warning(
                "systemd-run scopes unavailable on this host: GPU-profile "
                "generator host RAM stays UNCAPPED (RLIMIT_AS is skipped for "
                "torch) — a ballooning generator can get the trainer "
                "OOM-killed; run inside a memory-limited container for a hard "
                "boundary"
            )
    return _MEMSCOPE_PROBE


def wrap_memory_scope(argv: list[str], max_mb: int, available: bool) -> list[str]:
    """Wrap ``argv`` in a resident-memory-capped systemd scope (pure).

    ``MemoryMax`` is 2× the profile's ``max_memory_mb``: the cgroup counts
    RESIDENT memory, unlike RLIMIT_AS's virtual-address accounting, so the
    headroom keeps an honest torch generator (whose RSS can legitimately run
    past the VA-calibrated knob) alive while still bounding a balloon.
    ``MemorySwapMax=0`` makes the cap real — without it the kernel swaps the
    balloon instead of killing it. ``--collect`` reaps the scope even when
    the child is killed. Composes OUTERMOST around the unshare netns wrapper;
    ``available=False`` (no systemd, see :func:`_memory_scope_available`)
    passes ``argv`` through unchanged.
    """
    if not available:
        return list(argv)
    return [
        "systemd-run", "--scope", "--quiet", "--collect",
        "-p", f"MemoryMax={int(max_mb) * 2}M", "-p", "MemorySwapMax=0",
        *argv,
    ]


def _nvidia_compute_pids() -> frozenset[int]:
    """PIDs currently holding a CUDA compute context, per nvidia-smi.

    Raises on ANY problem (missing binary, non-zero exit, timeout) — the
    caller (:func:`_reject_gpu_use`) treats that as "nothing to check", so
    CPU-only boxes skip silently.
    """
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi rc={r.returncode}")
    return frozenset(int(tok) for tok in r.stdout.split() if tok.strip().isdigit())


def _proc_child_pids(pid: int) -> set[int]:
    """Direct children of ``pid`` via /proc (Linux; empty when gone/elsewhere)."""
    kids: set[int] = set()
    with contextlib.suppress(OSError):
        for task in Path(f"/proc/{pid}/task").iterdir():
            with contextlib.suppress(OSError, ValueError):
                kids |= {int(t) for t in (task / "children").read_text().split()}
    return kids


def _reject_gpu_use(pid: int, query_fn=None) -> None:
    """Layer 2 of keeping CPU-mode generators off the GPU: detect and reject.

    Env blanking (layer 1, see :func:`_child_env`) only stops *accidental*
    CUDA — a generator that rewrites its own env still reaches the devices.
    A secretly-CUDA corpus in a byte-exact mode is nondeterministic, so the
    tier-1 audit re-derivation mismatches and reads as a FALSE FRAUD PROOF
    against the trainer; the child can also touch other lanes' GPUs. Raising
    :class:`CorpusError` here converts that audit corruption into the miner
    losing their entry — incentive-correct.

    Suspects are kept simple: the sandbox child pid plus its direct /proc
    children. Known sampling race, documented: the query runs when the stream
    closes / the batch child exits, so a cheat that finishes its GPU work
    early (or hides it behind a re-parented grandchild) can evade this check
    — container mode is the real boundary. ``query_fn`` is injected for
    tests; the default shells out to nvidia-smi, and a missing nvidia-smi or
    any query error skips silently (CPU-only boxes).
    """
    try:
        gpu_pids = frozenset(int(p) for p in (query_fn or _nvidia_compute_pids)())
    except Exception:  # noqa: BLE001 - unqueryable (no nvidia-smi): nothing to check
        return
    if not gpu_pids:
        return
    hits = sorted(({int(pid)} | _proc_child_pids(int(pid))) & gpu_pids)
    if hits:
        raise CorpusError(
            f"generator_used_gpu_in_cpu_mode: sandbox child pid(s) {hits} held a "
            "CUDA compute context in a CPU (byte-exact) corpus mode — the corpus "
            "digest is untrustworthy, so this entry is rejected"
        )


def _netns_available() -> bool:
    """Probe (once) whether an unprivileged network namespace can be created."""
    global _NETNS_PROBE
    if _NETNS_PROBE is None:
        try:
            r = subprocess.run(
                ["unshare", "--user", "--map-root-user", "--net", "true"],
                capture_output=True,
                timeout=5,
            )
            _NETNS_PROBE = r.returncode == 0
        except Exception:  # noqa: BLE001 - any failure means "no netns"
            _NETNS_PROBE = False
    return _NETNS_PROBE


def _apply_rlimits(
    max_memory_mb: int, max_cpu_seconds: int, max_fsize_bytes: int, *, set_as: bool = True
) -> None:
    """preexec_fn: cap the child's resources before it execs (best-effort).

    ``set_as=False`` skips the address-space cap: a CUDA/torch generator reserves
    far more *virtual* memory than it uses, so RLIMIT_AS would kill it spuriously.
    The CPU-seconds, core, and file-size caps still apply, and the GPU profile is
    documented as requiring a no-egress container for hard isolation.
    """
    import resource

    limits = [
        (resource.RLIMIT_CPU, (int(max_cpu_seconds), int(max_cpu_seconds) + 5)),
        (resource.RLIMIT_CORE, (0, 0)),
        (resource.RLIMIT_FSIZE, (int(max_fsize_bytes), int(max_fsize_bytes))),
    ]
    if set_as:
        mem = int(max_memory_mb) * 1024 * 1024
        limits.insert(0, (resource.RLIMIT_AS, (mem, mem)))
    for name, vals in limits:
        # not every limit is settable in every environment
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(name, vals)


def stream_cpu_rlimit(
    max_generate_seconds: int, max_wall_seconds: int | None, nproc: int
) -> int:
    """Cumulative ``RLIMIT_CPU`` for a *streaming* generator child.

    Batch mode reuses ``max_generate_seconds`` as the CPU cap and that is
    coherent there — the whole run is also wall-clock bounded by the same knob.
    A streaming child is different: it lives for the entire training run, so its
    legitimate cumulative CPU scales with the training wall budget × however
    many cores its BLAS decides to use (the child env carries no thread caps).
    Reusing the 600s stall window as the cap silently SIGXCPUs (rc=-24) any
    CPU-busy honest generator once finals run longer than a few minutes.

    ``max_wall_seconds`` is the caller's upper bound on how long the stream will
    be consumed (the contract's ``max_train_seconds``); the cap allows full
    ``nproc`` utilisation for that long, plus the stall window for sandbox boot
    and first-frame latency. Abuse is NOT this limit's job — the per-frame stall
    timeout (``_frame_iter``) kills a generator that stops emitting, and the
    parent terminates the child when training ends. This is only the backstop
    for a wedged parent, so generous is correct. ``None`` keeps the legacy cap
    (short-lived screens with no known wall bound).

    ``nproc`` is the child's real core ceiling: on a lane-pinned pod that is
    the lane's slice size (see :func:`_lane_cpu_slice`) — budgeting the whole
    box would let one lane's cap absorb every other lane's share.
    """
    if max_wall_seconds is None:
        return int(max_generate_seconds)
    return int(max_generate_seconds) + max(1, int(nproc)) * int(max_wall_seconds)


def _child_env(*, gpu: bool = False) -> dict[str, str]:
    """Minimal allowlisted env for the sandbox child, with lane thread caps.

    When a lane slice is active, BLAS/threadpool env caps the generator's
    threads at the slice size — a pool sized to the whole box would just
    contend inside the lane's affinity set. No slice ⇒ no caps (legacy
    behavior, local runs). NOTE: a threaded-BLAS corpus digest depends on the
    thread count, and the slice size varies with pod shape — a future change
    may pin this to a contract-configured constant for stream_cpu audit
    reproducibility.
    """
    keys = _SAFE_ENV_KEYS + (_GPU_ENV_KEYS if gpu else ())
    env = {k: os.environ[k] for k in keys if k in os.environ}
    lane = _lane_cpu_slice()
    if lane is not None:
        for key in _BLAS_ENV_KEYS:
            env[key] = str(lane[1])
    if not gpu:
        # Keep CPU-mode generators off the GPU, layer 1: BLANK, not stripped.
        # An ABSENT CUDA_VISIBLE_DEVICES means "all GPUs visible", and
        # pip-torch bundles its own CUDA runtime — so a dependency that
        # initializes CUDA in a byte-exact (stream_cpu / cache_reuse) child
        # would make the corpus nondeterministic and could touch OTHER lanes'
        # GPUs. The blank/void values stop that ACCIDENTAL CUDA; a malicious
        # generator can pop its own env — that is what layer 2
        # (_reject_gpu_use) and container mode are for.
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
    # Ensure the child can import cascade even without an editable install.
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    return env


def _preflight(repo: Path, cfg: GeneratorConfig, blocked: tuple[str, ...]) -> None:
    """Cheap checks on the repo *files* (no miner code runs) before spawning."""
    from ..interface.static_guard import scan_file
    from ..interface.validation import check_repo_layout, check_repo_size

    layout = check_repo_layout(repo)
    if not layout.ok:
        raise CorpusError(f"repo_layout: {layout.reason} {layout.details or ''}")
    size = check_repo_size(repo, cfg.max_repo_mb)
    if not size.ok:
        raise CorpusError(f"repo_too_large: {size.details}")
    guard = scan_file(repo / "generator.py", tuple(blocked))
    if not guard.ok:
        raise CorpusError(f"blocked_import: {guard.blocked_module} ({guard.reason})")


def _load_series(path: Path, n: int) -> list:
    """Load a materialised corpus; ``s<i>__<field>`` keys rebuild extended
    records (dict elements), plain ``s<i>`` keys stay bare arrays."""
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        out: list = []
        for i in range(n):
            extras = {
                k.split("__", 1)[1]: np.ascontiguousarray(z[k])
                for k in keys
                if k.startswith(f"s{i}__")
            }
            values = np.ascontiguousarray(z[f"s{i}"])
            out.append({"values": values, **extras} if extras else values)
        return out


def _assert_isolation(cfg: GeneratorConfig, *, allow_netns: bool) -> bool:
    """Whether to wrap the child in a netns — refusing or warning LOUDLY when
    the host can't provide one.

    ``allow_netns=False`` is an explicit caller opt-out (tests / trusted local
    smoke) and skips the policy. Otherwise, a host without unprivileged network
    namespaces leaves only the Python-level socket guard between miner code and
    the network: with ``[generator] sandbox_strict = true`` (production) that
    refuses to run; the default logs a loud warning instead of silently
    downgrading.
    """
    if not allow_netns:
        return False
    if _netns_available():
        return True
    msg = ("network namespaces unavailable on this host (no unprivileged userns): "
           "generator network isolation degrades to the Python-level socket guard only")
    if cfg.sandbox_strict:
        raise CorpusError(
            f"sandbox_isolation_unavailable: {msg}. [generator] sandbox_strict = true "
            "refuses to run in this state — use sandbox_mode = 'container' or a "
            "userns-enabled host."
        )
    log.warning("sandbox: %s — set [generator] sandbox_strict = true to refuse instead, "
                "or sandbox_mode = 'container' for kernel-enforced isolation", msg)
    return False


def run_in_sandbox(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
    *,
    blocked: tuple[str, ...] = (),
    allow_netns: bool = True,
    gpu_pid_query=None,
) -> CorpusResult:
    """Run :func:`build_corpus` in an isolated subprocess and return its result.

    ``blocked`` is the static-guard import blocklist (``[static_guard] blocked``),
    enforced before the generator is imported. ``allow_netns=False`` skips the
    network-namespace wrapper (used in tests; Python-level networking is still
    disabled in the child). Raises :class:`CorpusError` on any failure.

    Batch mode is always the byte-exact CPU profile, so after the child exits
    it is checked for GPU use (:func:`_reject_gpu_use`; ``gpu_pid_query`` is
    the injectable pid source for tests).

    ``[generator] sandbox_mode = "container"`` reroutes to the docker/podman
    sandbox (:mod:`cascade.trainer.sandbox_container`) with this rlimited child
    kept inside as defense in depth.
    """
    if cfg.sandbox_mode == "container":
        from .sandbox_container import run_in_container

        return run_in_container(repo_dir, generation_seed, cfg, blocked=blocked)
    repo = Path(repo_dir)
    _preflight(repo, cfg, tuple(blocked))
    # DEC-CA-0028: materialise the pinned shared real corpus HERE, parent-side
    # — the child is network-isolated and must find real_corpus_dir already
    # resolved in its cfg JSON. A no-op while real_corpus_ref is unarmed.
    cfg = resolve_real_corpus(cfg)
    use_netns = _assert_isolation(cfg, allow_netns=allow_netns)

    with tempfile.TemporaryDirectory(prefix="metro-sbx-") as td:
        out_dir = Path(td)
        argv = [
            sys.executable, "-m", "cascade.trainer.sandbox",
            str(repo), str(int(generation_seed)), json.dumps(asdict(cfg)), str(out_dir),
        ]
        if use_netns:
            argv = ["unshare", "--user", "--map-root-user", "--net", *argv]
            log.debug("sandbox: running generator inside a network namespace")

        max_fsize = int(cfg.max_total_points) * 8 * 2 + 64 * 1024 * 1024
        timeout = int(cfg.max_generate_seconds) + 30
        preexec = None
        if os.name == "posix":
            lane = _lane_cpu_slice()

            def preexec() -> None:  # runs post-fork, pre-exec in the child
                if lane is not None:
                    # Same degrade-silently posture as _apply_rlimits: affinity
                    # can fail in odd container setups, and fairness is
                    # best-effort — never the reason a generator fails.
                    with contextlib.suppress(OSError):
                        os.sched_setaffinity(0, lane[0])
                _apply_rlimits(cfg.max_memory_mb, cfg.max_generate_seconds, max_fsize)

        # Popen instead of subprocess.run: the GPU-use check below needs the
        # child's pid, which CompletedProcess does not carry.
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_child_env(), preexec_fn=preexec,
        )
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.communicate(timeout=5)
            raise CorpusError(f"generator_timeout: exceeded {timeout}s wall-clock") from e
        # Batch = byte-exact CPU profile: a child that touched the GPU means an
        # untrustworthy digest, regardless of what it wrote (see _reject_gpu_use
        # for the sampling race — post-exit, this catches lingering contexts).
        _reject_gpu_use(proc.pid, gpu_pid_query)

        meta_p = out_dir / "meta.json"
        if not meta_p.is_file():
            tail = (stderr or b"").decode("utf-8", "replace")[-2000:]
            raise CorpusError(f"sandbox_crashed (rc={proc.returncode}): {tail}")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if not meta.get("ok"):
            raise CorpusError(f"generator_output_rejected: {meta.get('error')}")

        series = _load_series(out_dir / "corpus.npz", int(meta["n_series"]))
        digest = corpus_digest(series)
        if digest != meta.get("digest"):
            raise CorpusError("sandbox_digest_mismatch: corpus altered in transit")
        total = int(sum(
            int((s["values"] if isinstance(s, dict) else s).size) for s in series
        ))
        return CorpusResult(series=series, digest=digest, n_series=len(series), total_points=total)


# ──────────────────────── streaming (stream_cpu mode) ───────────────────────


def _read_exact(rd: io.BufferedReader, n: int) -> bytes | None:
    """Read exactly ``n`` bytes; None on clean EOF, short bytes on mid-frame EOF."""
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = rd.read(n - got)
        if not chunk:
            return b"".join(chunks) if chunks else None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


# Record-frame marker (DEC-CA-0020): a plain frame's 8-byte BE length prefix
# can never be all-0xFF (that would be a ~2^64-byte frame), so this header
# unambiguously introduces an extended record — n_fields (8B BE), then per
# field in sorted order: name length (8B BE), name utf-8, and a normal
# length-prefixed .npy frame. Emitted ONLY for series that actually carry
# accepted extras; values-only series keep the byte-identical plain framing.
_RECORD_MARKER = b"\xff" * 8


def _read_frame_body(rd: io.BufferedReader) -> bytes | None:
    header = _read_exact(rd, 8)
    if not header or len(header) < 8:
        return None
    body = _read_exact(rd, int.from_bytes(header, "big"))
    if body is None or len(body) < int.from_bytes(header, "big"):
        return None
    return body


def _frame_iter(
    proc: subprocess.Popen, inactivity_timeout: int
) -> Iterator[np.ndarray | dict]:
    """Yield series from the child's stdout: plain length-prefixed .npy frames,
    or marker-introduced record frames (see :data:`_RECORD_MARKER`)."""
    rd = proc.stdout
    timeout = max(int(inactivity_timeout), 1)
    while True:
        ready, _, _ = select.select([rd], [], [], timeout)
        if not ready:
            raise CorpusError(f"generator_stalled: no series for {timeout}s")
        header = _read_exact(rd, 8)
        if not header or len(header) < 8:
            break  # clean EOF: child finished its prefix (or died — checked below)
        if header == _RECORD_MARKER:
            count_b = _read_exact(rd, 8)
            if not count_b or len(count_b) < 8:
                break
            rec: dict[str, np.ndarray] = {}
            broken = False
            for _ in range(int.from_bytes(count_b, "big")):
                name_len_b = _read_exact(rd, 8)
                if not name_len_b or len(name_len_b) < 8:
                    broken = True
                    break
                name_b = _read_exact(rd, int.from_bytes(name_len_b, "big"))
                body = None if name_b is None else _read_frame_body(rd)
                if name_b is None or body is None:
                    broken = True
                    break
                rec[name_b.decode("utf-8")] = np.ascontiguousarray(
                    np.lib.format.read_array(io.BytesIO(body), allow_pickle=False)
                )
            if broken:
                break  # child died mid-record
            yield rec
            continue
        body = _read_exact(rd, int.from_bytes(header, "big"))
        if body is None or len(body) < int.from_bytes(header, "big"):
            break  # child died mid-frame
        yield np.ascontiguousarray(np.lib.format.read_array(io.BytesIO(body), allow_pickle=False))
    rc = proc.poll()
    if rc not in (0, None):
        err = (proc.stderr.read() or b"").decode("utf-8", "replace")[-2000:]
        raise CorpusError(f"sandbox_stream_failed (rc={rc}): {err}")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=5)
    for stream in (proc.stdout, proc.stderr):
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()


@contextlib.contextmanager
def stream_series(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
    token_budget: int,
    *,
    blocked: tuple[str, ...] = (),
    allow_netns: bool = True,
    gpu: bool = False,
    max_wall_seconds: int | None = None,
    gpu_pid_query=None,
) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of fresh ``(C, L)`` series streamed from a sandboxed child.

    ``max_wall_seconds`` bounds how long the caller will consume the stream
    (pass the contract's ``max_train_seconds``); it scales the child's
    cumulative CPU rlimit — see :func:`stream_cpu_rlimit` for why the batch-mode
    cap must not be reused here.

    In the CPU (byte-exact) profile, a clean stream close checks the child for
    GPU use while its pid is still known and raises :class:`CorpusError` on a
    hit — see :func:`_reject_gpu_use` (``gpu_pid_query`` is the injectable pid
    source for tests).

    The child draws a prefix of ``generate`` long enough to cover ``token_budget``
    points — the generator is prefix-stable, so the consumed prefix is
    reproducible — and streams each validated series over a pipe; the caller stops
    once it has its budget. Same isolation as :func:`run_in_sandbox` (rlimits, env
    scrub, optional netns, Python-socket block, pre-flight). The child is always
    terminated on exit, including early stop.

    ``gpu=True`` is the ``stream_gpu`` profile for a CUDA/torch-resident
    generator: the address-space rlimit is dropped (torch over-reserves virtual
    memory) and the CUDA selection/library env is passed through, while the
    network namespace + socket block stay on. Run the trainer in a **no-egress
    container** for hard isolation here, since RLIMIT_AS no longer bounds the
    child and CUDA needs device access. Audit is tolerance/same-hardware, not
    byte-exact (see ``chain.toml [training] corpus_mode``).

    ``[generator] sandbox_mode = "container"`` reroutes to the docker/podman
    sandbox (:mod:`cascade.trainer.sandbox_container`), the rlimited child kept
    inside as defense in depth.
    """
    if cfg.sandbox_mode == "container":
        from .sandbox_container import stream_series_container

        with stream_series_container(
            repo_dir, generation_seed, cfg, token_budget, blocked=blocked, gpu=gpu,
            max_wall_seconds=max_wall_seconds,
        ) as frames:
            yield frames
        return
    repo = Path(repo_dir)
    _preflight(repo, cfg, tuple(blocked))
    # DEC-CA-0028: resolve the shared real corpus parent-side (the child
    # cannot fetch); no-op while unarmed.
    cfg = resolve_real_corpus(cfg)
    use_netns = _assert_isolation(cfg, allow_netns=allow_netns)
    n_upper = int(token_budget) // max(int(cfg.min_length), 1) + 2
    argv = [
        sys.executable, "-m", "cascade.trainer.sandbox", "--stream",
        str(repo), str(int(generation_seed)), json.dumps(asdict(cfg)), str(n_upper),
    ]
    if use_netns:
        argv = ["unshare", "--user", "--map-root-user", "--net", *argv]
    if gpu:
        # GPU profile: RLIMIT_AS is skipped below (torch VA over-reserve), so
        # a cgroup scope is the only resident-memory bound — systemd-run
        # OUTERMOST so the whole netns+python tree lands in one scope.
        argv = wrap_memory_scope(argv, cfg.max_memory_mb, _memory_scope_available())

    # nproc = the lane's slice size when pinned (that IS the child's real core
    # ceiling — a host-wide count would over-budget every lane's cumulative
    # CPU cap), the whole box otherwise.
    lane = _lane_cpu_slice()
    cpu_cap = stream_cpu_rlimit(
        cfg.max_generate_seconds, max_wall_seconds,
        lane[1] if lane is not None else (os.cpu_count() or 1),
    )
    preexec = None
    if os.name == "posix":
        def preexec() -> None:
            if lane is not None:
                # Degrade-silently like _apply_rlimits: affinity can fail in
                # odd container setups, and fairness must never fail a run.
                with contextlib.suppress(OSError):
                    os.sched_setaffinity(0, lane[0])
            _apply_rlimits(
                cfg.max_memory_mb, cpu_cap, 256 * 1024 * 1024, set_as=not gpu
            )

    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=_child_env(gpu=gpu), preexec_fn=preexec, bufsize=0,
    )
    try:
        # Per-frame stall window: [generator] stream_stall_seconds when set,
        # else max_generate_seconds (DEC-CA-0029 decoupled the two so raising
        # the batch drain budget cannot widen how long a stalled generator
        # holds a lane).
        yield _frame_iter(proc, cfg.effective_stall_seconds)
    except BaseException:
        _terminate(proc)
        raise
    else:
        # Clean close (including the normal early stop at the token budget):
        # in the byte-exact CPU profile, sample for GPU use BEFORE terminating
        # — the child is usually still alive here, so a live CUDA context is
        # attributable to its pid. On a hit the CorpusError propagates and
        # the entry is rejected.
        try:
            if not gpu:
                _reject_gpu_use(proc.pid, gpu_pid_query)
        finally:
            _terminate(proc)


# ───────────────────────────────── child ───────────────────────────────────


def _disable_network() -> None:
    """Defense-in-depth: make Python-level socket use raise inside the child."""
    # Pre-import ``socket`` plus the stdlib modules that subclass the real
    # ``socket`` class at import time (``ssl.SSLSocket(socket)``, and ``asyncio``
    # which pulls in ssl). A torch-based generator imports these lazily; if we
    # replace ``socket.socket`` with a plain function first, their
    # ``class X(socket)`` fails with a cryptic ``TypeError: function() argument
    # 'code' must be code, not str``. Importing the real base class now (all
    # before the reassignment below) lets those classes build, while new socket()
    # calls still raise. (Without this, NO torch/ssl-importing generator can run.)
    import asyncio  # noqa: F401
    import socket
    import ssl  # noqa: F401

    def _blocked(*_a: object, **_k: object) -> None:
        raise OSError("network access is disabled in the cascade generation sandbox")

    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(var, None)


def _maybe_self_rlimit(cfg: GeneratorConfig) -> None:
    """Container mode: the child applies its own POSIX rlimits.

    In subprocess mode the parent's ``preexec_fn`` sets them pre-exec; inside a
    container there is no cascade parent, so :mod:`.sandbox_container` sets
    ``CASCADE_SANDBOX_SELF_RLIMIT=1`` and the child self-limits here — the
    container's cgroup ceilings and this are independent layers (defense in
    depth). ``CASCADE_SANDBOX_GPU=1`` skips only the address-space cap (torch
    over-reserves virtual memory), matching the subprocess GPU profile.
    """
    if os.environ.get("CASCADE_SANDBOX_SELF_RLIMIT") != "1":
        return
    max_fsize = int(cfg.max_total_points) * 8 * 2 + 64 * 1024 * 1024
    # Streaming containers pass a scaled CPU cap (see stream_cpu_rlimit) since
    # there is no cascade parent to compute it in a preexec hook.
    cpu_cap = int(os.environ.get("CASCADE_SANDBOX_CPU_S") or cfg.max_generate_seconds)
    _apply_rlimits(
        cfg.max_memory_mb, cpu_cap, max_fsize,
        set_as=os.environ.get("CASCADE_SANDBOX_GPU") != "1",
    )


def _save_series(path: Path, series: list) -> None:
    """Materialise a corpus to .npz — extended-record extras ride as
    ``s<i>__<field>`` keys next to their ``s<i>`` values array."""
    payload: dict[str, np.ndarray] = {}
    for i, a in enumerate(series):
        if isinstance(a, dict):
            for name, arr in a.items():
                payload[f"s{i}" if name == "values" else f"s{i}__{name}"] = arr
        else:
            payload[f"s{i}"] = a
    np.savez(path, **payload)


def _write_frame(out: io.BufferedWriter, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, allow_pickle=False)
    data = buf.getvalue()
    out.write(len(data).to_bytes(8, "big"))
    out.write(data)


def _write_record_frame(out: io.BufferedWriter, rec: dict) -> None:
    """Stream one extended record (see :data:`_RECORD_MARKER`)."""
    out.write(_RECORD_MARKER)
    out.write(len(rec).to_bytes(8, "big"))
    for name in sorted(rec):
        nb = name.encode("utf-8")
        out.write(len(nb).to_bytes(8, "big"))
        out.write(nb)
        _write_frame(out, rec[name])


def _child_materialize(repo: str, seed: str, cfg_json: str, out_dir: str) -> int:
    out = Path(out_dir)
    try:
        cfg = GeneratorConfig(**json.loads(cfg_json))
        _maybe_self_rlimit(cfg)
        res = build_corpus(repo, int(seed), cfg)
        _save_series(out / "corpus.npz", res.series)
        meta = {
            "ok": True,
            "digest": res.digest,
            "n_series": res.n_series,
            "total_points": res.total_points,
        }
    except Exception as e:  # noqa: BLE001 - report any failure (incl. MemoryError) as meta
        meta = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    with contextlib.suppress(OSError):
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return 0 if meta.get("ok") else 1


def _child_stream(repo: str, seed: str, cfg_json: str, n_upper: str) -> int:
    from ..interface.generator import SeriesValidator
    from .corpus import _load_generator

    out = sys.stdout.buffer
    try:
        cfg = GeneratorConfig(**json.loads(cfg_json))
        _maybe_self_rlimit(cfg)
        from .channel_stats import corr_enforce_gate

        # One validation pipeline shared with the drain and the in-process
        # stream (SeriesValidator) — record fields, budgets, and gates cannot
        # drift between feed modes. Values-only records collapse to their
        # values array and cross as plain .npy frames — byte-identical to a
        # bare-array stream; series carrying an ACCEPTED extra (mask/roles,
        # armed via [training] accepted_fields) cross as marker-introduced
        # record frames (_write_record_frame).
        validator = SeriesValidator.from_config(
            cfg, extra_series_check=corr_enforce_gate(cfg)
        )
        # An armed cfg arrives with real_corpus_dir pre-resolved by the parent;
        # resolve here is verify-only (an empty dir would try to fetch and fail
        # loudly on the network block — a misplumbed parent, never a silent
        # no-corpus stream).
        cfg = resolve_real_corpus(cfg)
        gen = _load_generator(Path(repo), int(seed),
                              interface_version=cfg.interface_version,
                              real_corpus_dir=cfg.real_corpus_dir)
        for i, item in enumerate(gen.generate(int(n_upper))):
            element = validator.process(item, i)
            if isinstance(element, dict):
                _write_record_frame(out, element)
            else:
                _write_frame(out, element)
            out.flush()
    except BrokenPipeError:
        return 0  # parent stopped reading once it had its budget — a normal stop
    except Exception as e:  # noqa: BLE001
        with contextlib.suppress(Exception):
            sys.stderr.write(f"{type(e).__name__}: {e}\n")
            sys.stderr.flush()
        return 1
    return 0


def _child_main(argv: list[str]) -> int:
    _disable_network()
    if len(argv) > 1 and argv[1] == "--stream":
        return _child_stream(argv[2], argv[3], argv[4], argv[5])
    return _child_materialize(argv[1], argv[2], argv[3], argv[4])


if __name__ == "__main__":
    sys.exit(_child_main(sys.argv))
