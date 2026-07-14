"""Container-hardened generation sandbox (``[generator] sandbox_mode = "container"``).

Runs the exact same child entry point as the subprocess sandbox
(``python -m cascade.trainer.sandbox``) but inside a locked-down docker/podman
container, for hosts where ``unshare`` netns is unavailable (no unprivileged
user namespaces) or where kernel-level isolation is wanted regardless:

* ``--network=none``                      — no interfaces at all, kernel-enforced;
* ``--cap-drop=ALL --security-opt=no-new-privileges`` — no capabilities, no
  setuid escalation;
* ``--read-only`` rootfs + ``--tmpfs /tmp`` — the only writable paths are a
  size-capped tmpfs workdir and (materialise mode) the bind-mounted output dir;
* ``--memory/--memory-swap/--pids-limit/--cpus``  — resource ceilings;
* the generator repo and the cascade source are bind-mounted **read-only**.

Defense in depth is preserved: the child still applies the POSIX rlimits and
the Python-level socket block from :mod:`cascade.trainer.sandbox` *inside* the
container (``CASCADE_SANDBOX_SELF_RLIMIT=1``), so escaping the container's
limits still lands in the rlimited, socket-blocked interpreter.

The image (``[generator] sandbox_image``) must carry python3 + numpy (and
whatever the dependency allowlist admits — the worker image from
``deploy/Dockerfile`` works); ``[generator] sandbox_python`` names the
interpreter inside it. Pin the image by digest in production. All parent-side
validation is shared with the subprocess sandbox: pre-flight (layout, size,
static guard) runs before any container starts, and the returned corpus is
digest-verified by the parent.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..shared.config import GeneratorConfig
from ..shared.manifest import corpus_digest
from .corpus import CorpusError, CorpusResult

log = logging.getLogger("cascade.trainer.sandbox")

# CPU ceiling for a generation container. Generation is a single python process
# (numpy may thread); 4 cores is ample and keeps a fork-bombing generator from
# saturating the trainer box even before --pids-limit bites.
CONTAINER_CPUS = "4"
CONTAINER_PIDS_LIMIT = "256"
CONTAINER_TMPFS = "/tmp:rw,noexec,nosuid,size=256m"

# In-container mount points (fixed; the host paths vary per run).
_REPO_MNT = "/sandbox/repo"
_SRC_MNT = "/sandbox/src"
_OUT_MNT = "/sandbox/out"

_RUNTIME: str | None | bool = False  # False = unprobed


def container_runtime() -> str | None:
    """The available container runtime binary (docker, then podman), or None."""
    global _RUNTIME
    if _RUNTIME is False:
        _RUNTIME = next((rt for rt in ("docker", "podman") if shutil.which(rt)), None)
    return _RUNTIME


def _require_runtime(cfg: GeneratorConfig) -> str:
    rt = container_runtime()
    if rt is None:
        raise CorpusError(
            "sandbox_mode='container' but neither docker nor podman is on PATH"
        )
    if not cfg.sandbox_image:
        raise CorpusError(
            "sandbox_mode='container' needs [generator] sandbox_image "
            "(a digest-pinned image with python3 + numpy; the worker image works)"
        )
    return rt


def container_argv(
    cfg: GeneratorConfig,
    *,
    runtime: str,
    name: str,
    repo: Path,
    child_args: list[str],
    out_dir: Path | None = None,
    gpu: bool = False,
    cpu_seconds: int | None = None,
    lane_cores: tuple[int, ...] | None = None,
) -> list[str]:
    """The full ``docker run`` argv for one sandboxed generator run.

    Pure (no I/O), so the hardening flags are unit-testable. ``child_args`` is
    everything after ``-m cascade.trainer.sandbox``. ``cpu_seconds`` overrides
    the child's self-applied CPU rlimit (streaming runs scale it with the
    training wall budget — see ``sandbox.stream_cpu_rlimit``). ``lane_cores``
    is this worker lane's CPU slice (see ``sandbox._lane_cpu_slice``): lane
    fairness parity with the subprocess sandbox — the container is *placed*
    on the lane's cores (``--cpuset-cpus``) and its BLAS thread pool capped at
    the slice size, while ``--cpus`` stays the rate ceiling in both cases.
    """
    src_root = Path(__file__).resolve().parents[2]
    argv = [
        runtime, "run", "--rm", "--name", name,
        # Pin the entrypoint to the configured interpreter so an image's own
        # ENTRYPOINT can never intercept (or reinterpret) the sandbox command.
        "--entrypoint", cfg.sandbox_python,
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--tmpfs", CONTAINER_TMPFS,
        "--memory", f"{int(cfg.max_memory_mb)}m",
        "--memory-swap", f"{int(cfg.max_memory_mb)}m",   # = memory ⇒ no swap headroom
        "--pids-limit", CONTAINER_PIDS_LIMIT,
        "--cpus", CONTAINER_CPUS,
        "-v", f"{repo}:{_REPO_MNT}:ro",
        "-v", f"{src_root}:{_SRC_MNT}:ro",
        "-e", f"PYTHONPATH={_SRC_MNT}",
        "-e", "HOME=/tmp",
        "-e", "TMPDIR=/tmp",
        # Defense in depth: the child re-applies the POSIX rlimits + socket
        # block inside the container (see sandbox._maybe_self_rlimit).
        "-e", "CASCADE_SANDBOX_SELF_RLIMIT=1",
    ]
    if cpu_seconds is not None:
        argv += ["-e", f"CASCADE_SANDBOX_CPU_S={int(cpu_seconds)}"]
    if lane_cores:
        from .sandbox import _BLAS_ENV_KEYS

        # Comma-joined ids, not a range: a pre-existing container cpuset can
        # make the lane's allowed cores non-contiguous.
        argv += ["--cpuset-cpus", ",".join(str(c) for c in sorted(lane_cores))]
        for key in _BLAS_ENV_KEYS:
            argv += ["-e", f"{key}={len(lane_cores)}"]
    if out_dir is not None:
        argv += ["-v", f"{out_dir}:{_OUT_MNT}:rw"]
    if gpu:
        # stream_gpu profile: the device is exposed but the network still is
        # not; the child skips only the address-space rlimit (torch reserves
        # far more virtual memory than it uses).
        argv += ["--gpus", "all", "-e", "CASCADE_SANDBOX_GPU=1"]
    argv += [cfg.sandbox_image, "-m", "cascade.trainer.sandbox", *child_args]
    return argv


def _kill_container(runtime: str, name: str) -> None:
    with contextlib.suppress(Exception):
        subprocess.run([runtime, "kill", name], capture_output=True, timeout=30)


def run_in_container(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
    *,
    blocked: tuple[str, ...] = (),
) -> CorpusResult:
    """Container-mode :func:`cascade.trainer.sandbox.run_in_sandbox`.

    Same contract: pre-flight → run the child → load + digest-verify the
    returned arrays. Raises :class:`CorpusError` on any failure.
    """
    from .sandbox import _lane_cpu_slice, _load_series, _preflight

    repo = Path(repo_dir).resolve()
    _preflight(repo, cfg, tuple(blocked))
    runtime = _require_runtime(cfg)

    with tempfile.TemporaryDirectory(prefix="metro-csbx-") as td:
        out_dir = Path(td)
        out_dir.chmod(0o777)  # the image's user must be able to write the mount
        name = f"cascade-sbx-{os.urandom(6).hex()}"
        lane = _lane_cpu_slice()
        argv = container_argv(
            cfg, runtime=runtime, name=name, repo=repo, out_dir=out_dir,
            child_args=[_REPO_MNT, str(int(generation_seed)),
                        json.dumps(asdict(cfg)), _OUT_MNT],
            lane_cores=tuple(sorted(lane[0])) if lane is not None else None,
        )
        timeout = int(cfg.max_generate_seconds) + 120  # + container startup slack
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            _kill_container(runtime, name)
            raise CorpusError(f"generator_timeout: exceeded {timeout}s wall-clock "
                              "(container killed)") from e

        meta_p = out_dir / "meta.json"
        if not meta_p.is_file():
            tail = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
            raise CorpusError(f"container_sandbox_crashed (rc={proc.returncode}): {tail}")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if not meta.get("ok"):
            raise CorpusError(f"generator_output_rejected: {meta.get('error')}")

        series = _load_series(out_dir / "corpus.npz", int(meta["n_series"]))
        digest = corpus_digest(series)
        if digest != meta.get("digest"):
            raise CorpusError("sandbox_digest_mismatch: corpus altered in transit")
        total = int(sum(int(s.size) for s in series))
        return CorpusResult(series=series, digest=digest, n_series=len(series),
                            total_points=total)


@contextlib.contextmanager
def stream_series_container(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
    token_budget: int,
    *,
    blocked: tuple[str, ...] = (),
    gpu: bool = False,
    max_wall_seconds: int | None = None,
) -> Iterator[Iterator[np.ndarray]]:
    """Container-mode :func:`cascade.trainer.sandbox.stream_series`.

    The child streams length-prefixed ``.npy`` frames over the container's
    stdout; the caller stops at its budget and the container is always killed
    on exit.
    """
    from .sandbox import (
        _frame_iter,
        _lane_cpu_slice,
        _preflight,
        _terminate,
        stream_cpu_rlimit,
    )

    repo = Path(repo_dir).resolve()
    _preflight(repo, cfg, tuple(blocked))
    runtime = _require_runtime(cfg)

    n_upper = int(token_budget) // max(int(cfg.min_length), 1) + 2
    name = f"cascade-sbx-{os.urandom(6).hex()}"
    # Slice size when lane-pinned (the container is placed on the lane's cores
    # via --cpuset-cpus, so that IS its core ceiling), the whole box otherwise.
    lane = _lane_cpu_slice()
    cpu_cap = stream_cpu_rlimit(
        cfg.max_generate_seconds, max_wall_seconds,
        lane[1] if lane is not None else (os.cpu_count() or 1),
    )
    argv = container_argv(
        cfg, runtime=runtime, name=name, repo=repo, gpu=gpu, cpu_seconds=cpu_cap,
        child_args=["--stream", _REPO_MNT, str(int(generation_seed)),
                    json.dumps(asdict(cfg)), str(n_upper)],
        lane_cores=tuple(sorted(lane[0])) if lane is not None else None,
    )
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
    )
    try:
        yield _frame_iter(proc, cfg.max_generate_seconds)
    finally:
        _kill_container(runtime, name)
        _terminate(proc)
