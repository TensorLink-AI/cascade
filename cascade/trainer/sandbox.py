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
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..shared.config import GeneratorConfig
from ..shared.manifest import corpus_digest
from .corpus import CorpusError, CorpusResult, build_corpus

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

_NETNS_PROBE: bool | None = None


# ───────────────────────────────── parent ──────────────────────────────────


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


def _child_env(*, gpu: bool = False) -> dict[str, str]:
    keys = _SAFE_ENV_KEYS + (_GPU_ENV_KEYS if gpu else ())
    env = {k: os.environ[k] for k in keys if k in os.environ}
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


def _load_series(path: Path, n: int) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return [np.ascontiguousarray(z[f"s{i}"]) for i in range(n)]


def run_in_sandbox(
    repo_dir: Path | str,
    generation_seed: int,
    cfg: GeneratorConfig,
    *,
    blocked: tuple[str, ...] = (),
    allow_netns: bool = True,
) -> CorpusResult:
    """Run :func:`build_corpus` in an isolated subprocess and return its result.

    ``blocked`` is the static-guard import blocklist (``[static_guard] blocked``),
    enforced before the generator is imported. ``allow_netns=False`` skips the
    network-namespace wrapper (used in tests; Python-level networking is still
    disabled in the child). Raises :class:`CorpusError` on any failure.
    """
    repo = Path(repo_dir)
    _preflight(repo, cfg, tuple(blocked))

    with tempfile.TemporaryDirectory(prefix="metro-sbx-") as td:
        out_dir = Path(td)
        argv = [
            sys.executable, "-m", "cascade.trainer.sandbox",
            str(repo), str(int(generation_seed)), json.dumps(asdict(cfg)), str(out_dir),
        ]
        if allow_netns and _netns_available():
            argv = ["unshare", "--user", "--map-root-user", "--net", *argv]
            log.debug("sandbox: running generator inside a network namespace")

        max_fsize = int(cfg.max_total_points) * 8 * 2 + 64 * 1024 * 1024
        timeout = int(cfg.max_generate_seconds) + 30
        preexec = None
        if os.name == "posix":
            def preexec() -> None:  # runs post-fork, pre-exec in the child
                _apply_rlimits(cfg.max_memory_mb, cfg.max_generate_seconds, max_fsize)

        try:
            proc = subprocess.run(
                argv, capture_output=True, timeout=timeout,
                env=_child_env(), preexec_fn=preexec,
            )
        except subprocess.TimeoutExpired as e:
            raise CorpusError(f"generator_timeout: exceeded {timeout}s wall-clock") from e

        meta_p = out_dir / "meta.json"
        if not meta_p.is_file():
            tail = (proc.stderr or b"").decode("utf-8", "replace")[-2000:]
            raise CorpusError(f"sandbox_crashed (rc={proc.returncode}): {tail}")
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if not meta.get("ok"):
            raise CorpusError(f"generator_output_rejected: {meta.get('error')}")

        series = _load_series(out_dir / "corpus.npz", int(meta["n_series"]))
        digest = corpus_digest(series)
        if digest != meta.get("digest"):
            raise CorpusError("sandbox_digest_mismatch: corpus altered in transit")
        total = int(sum(int(s.size) for s in series))
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


def _frame_iter(proc: subprocess.Popen, inactivity_timeout: int) -> Iterator[np.ndarray]:
    """Yield arrays from the child's length-prefixed .npy frames on stdout."""
    rd = proc.stdout
    timeout = max(int(inactivity_timeout), 1)
    while True:
        ready, _, _ = select.select([rd], [], [], timeout)
        if not ready:
            raise CorpusError(f"generator_stalled: no series for {timeout}s")
        header = _read_exact(rd, 8)
        if not header or len(header) < 8:
            break  # clean EOF: child finished its prefix (or died — checked below)
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
) -> Iterator[Iterator[np.ndarray]]:
    """Yield an iterator of fresh ``(C, L)`` series streamed from a sandboxed child.

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
    """
    repo = Path(repo_dir)
    _preflight(repo, cfg, tuple(blocked))
    n_upper = int(token_budget) // max(int(cfg.min_length), 1) + 2
    argv = [
        sys.executable, "-m", "cascade.trainer.sandbox", "--stream",
        str(repo), str(int(generation_seed)), json.dumps(asdict(cfg)), str(n_upper),
    ]
    if allow_netns and _netns_available():
        argv = ["unshare", "--user", "--map-root-user", "--net", *argv]

    preexec = None
    if os.name == "posix":
        def preexec() -> None:
            _apply_rlimits(
                cfg.max_memory_mb, cfg.max_generate_seconds, 256 * 1024 * 1024, set_as=not gpu
            )

    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=_child_env(gpu=gpu), preexec_fn=preexec, bufsize=0,
    )
    try:
        yield _frame_iter(proc, cfg.max_generate_seconds)
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


def _save_series(path: Path, series: list[np.ndarray]) -> None:
    np.savez(path, **{f"s{i}": a for i, a in enumerate(series)})


def _write_frame(out: io.BufferedWriter, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.lib.format.write_array(buf, arr, allow_pickle=False)
    data = buf.getvalue()
    out.write(len(data).to_bytes(8, "big"))
    out.write(data)


def _child_materialize(repo: str, seed: str, cfg_json: str, out_dir: str) -> int:
    out = Path(out_dir)
    try:
        cfg = GeneratorConfig(**json.loads(cfg_json))
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
    from ..interface.generator import check_series
    from .corpus import _load_generator

    out = sys.stdout.buffer
    try:
        cfg = GeneratorConfig(**json.loads(cfg_json))
        gen = _load_generator(Path(repo), int(seed))
        for i, arr in enumerate(gen.generate(int(n_upper))):
            check_series(
                arr, min_length=cfg.min_length, max_length=cfg.max_length,
                max_channels=cfg.max_channels, index=i,
            )
            canon = np.ascontiguousarray(np.atleast_2d(np.asarray(arr, dtype=np.float64)))
            _write_frame(out, canon)
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
