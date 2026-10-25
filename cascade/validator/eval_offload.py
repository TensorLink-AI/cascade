"""GPU offload for all four public benchmarks and opt-in private inference.

Public sweeps return reports; only the designated main validator computes the
validator-sourced GIFT-Eval/BOOM/TIME/FEV rows, then other validators replay the
signed evidence. The legacy GIFT-only gate uses the same transport.

Private offload is separately authorized: selected histories and a hash-bound
checkpoint cross to an operator-owned host, samples return, and targets and
score computation stay local. A missing or invalid reply is infrastructure
failure. Neither path forwards wallets or provider keys.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shlex
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..eval.benchmarks import bench_uv_args, bench_workers_arg, gift_rows_from_report
from ..trainer.remote import (
    RemoteDispatchError,
    RemoteHost,
    build_ssh_argv,
    load_hosts,
    run_ssh,
)

log = logging.getLogger("cascade.validator.eval_offload")

# uv on the pod (runs the benchmarks/ sidecar's own locked env), matching
# cascade.trainer.bench_hook.BenchPlan.uv_bin.
DEFAULT_UV_BIN = "~/.local/bin/uv"


class PrivateEvalUnavailable(RuntimeError):
    """A trusted GPU transport failed; never a miner-caused rejection."""


def forecasts_via_host(host, checkpoint, windows, *, contract, num_samples,
                       workers=1, timeout_s=3600, runner=None):
    """Send only histories to an explicitly configured operator-owned host.

    Score returned samples locally against the original targets. A missing,
    corrupt, stale or partial reply holds the round; it never invents scores
    or silently switches a consensus comparison to another execution mode.
    """
    from ..eval.checkpoint_guard import CheckpointTampered, verify_checkpoint
    from .evaluator import EvaluatorError
    from .private_worker import CHECKPOINT_FILES, read_forecasts, write_request

    try:
        verify_checkpoint(checkpoint, contract)
    except CheckpointTampered as e:
        raise EvaluatorError(f"checkpoint_tampered: {e}") from e
    run = runner or run_ssh
    deadline = time.monotonic() + timeout_s
    work = f"_private_eval_offload/{uuid.uuid4().hex}"
    base = f"{host.workdir}/{work}"

    def step(argv):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PrivateEvalUnavailable("private evaluation deadline expired")
        proc = run(argv, remaining)
        if proc.returncode:
            raise PrivateEvalUnavailable("private evaluation transport/worker failed")
        return proc

    try:
        with tempfile.TemporaryDirectory(prefix="cascade-private-eval-") as tmp:
            local = Path(tmp)
            request = local / "request.npz"
            identity = write_request(request, checkpoint, windows,
                                     contract=contract, num_samples=num_samples)
            step(build_ssh_argv(host, f"umask 077 && mkdir -p {shlex.quote(base + '/checkpoint')}"))
            for name in CHECKPOINT_FILES:
                step(build_scp_argv(host, str(Path(checkpoint) / name), f"{base}/checkpoint/{name}"))
            step(build_scp_argv(host, str(request), f"{base}/request.npz"))
            argv = [host.remote_python, "-m", "cascade.validator.private_worker",
                    f"{work}/checkpoint", f"{work}/request.npz", f"{work}/samples.npz",
                    "--device", "cuda", "--workers", str(workers)]
            env = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            if host.cuda_device is not None:
                env += f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
            command = (f"cd {shlex.quote(host.workdir)} && " + env
                       + f"timeout --signal=TERM --kill-after=15s {max(1, int(deadline - time.monotonic()) - 15)}s "
                       + " ".join(shlex.quote(a) for a in argv))
            step(build_ssh_argv(host, command))
            size = step(build_ssh_argv(host, f"stat -c %s {shlex.quote(base + '/samples.npz')}"))
            budget = (sum(w.n_channels * num_samples * w.horizon * 8 for w in windows)
                      + (len(windows) + 1) * 8192)
            if not 0 < int(size.stdout.strip()) <= budget:
                raise PrivateEvalUnavailable("private evaluation response exceeds its shape budget")
            output = local / "samples.npz"
            download = build_scp_argv(host, "unused", "unused")
            download[-2:] = [f"{host.user}@{host.host}:{base}/samples.npz", str(output)]
            step(download)
            return read_forecasts(output, identity, windows, num_samples)
    except PrivateEvalUnavailable:
        raise
    except Exception as e:
        raise PrivateEvalUnavailable(f"private evaluation response unavailable: {type(e).__name__}") from e
    finally:
        try:
            cleanup = run(build_ssh_argv(host, f"rm -rf {shlex.quote(base)}"), 15)
            if cleanup.returncode:
                log.warning("private eval cleanup failed on %s; remove %s", host.name, base)
        except Exception:
            log.warning("private eval cleanup unavailable on %s; remove %s", host.name, base)


def make_eval_host_fn(path: Path | str, *, local_fallback: bool = True) -> Callable[[], RemoteHost | None]:
    """A resolver that re-reads the eval-hosts file AT EACH offloaded eval.

    The eval pod is elastic: the provisioner rents it when a round's manifest
    appears and publishes/clears ``path`` as the pod comes and goes, so a
    startup-time load would pin the validator to a box that may already be
    torn down — or miss one rented minutes later. Each call re-resolves:
    missing file, empty/cleared file, or no ``final``/``any``-stage entry ⇒
    ``None`` ⇒ that eval runs locally on the validator's own device. An
    unparseable file logs a WARNING and also falls back local — a bad hosts
    file must degrade an eval, never crash the validator. A static
    hand-maintained file that never changes behaves exactly like the old
    one-shot startup load. Appear/disappear/change transitions are logged
    once (INFO) per change, not per eval.
    """
    last_seen: str | None = None

    def resolve() -> RemoteHost | None:
        nonlocal last_seen
        host: RemoteHost | None = None
        p = Path(path)
        if p.is_file():
            try:
                cands = [h for h in load_hosts(p) if h.stage in ("any", "final")]
                host = cands[0] if cands else None
                if host is not None:
                    # Provisioned fleets fan out one entry per GPU. Public
                    # sweeps can use ALL visible cards of the selected VM,
                    # rather than silently pinning the first card only.
                    def endpoint(h):
                        return (h.host, h.port, h.user, h.key_path, h.workdir,
                                h.remote_python, h.ssh_options)

                    peers = [h for h in cands if endpoint(h) == endpoint(host)]
                    devices = list(dict.fromkeys(
                        d.strip() for h in peers for d in (h.cuda_device or "").split(",")
                        if d.strip()))
                    mask = (",".join(devices) if all(h.cuda_device is not None for h in peers)
                            else None)
                    host = replace(host, cuda_device=mask or None)
            except RemoteDispatchError as e:
                # Zero [[host]] entries is the provisioner's explicit
                # "no pod right now" clear — normal, not an error.
                log.debug("eval-hosts %s has no usable host (%s)", p, e)
            except Exception as e:  # noqa: BLE001 — a bad hosts file degrades, never crashes
                log.warning("eval-hosts %s unreadable (%s); %s", p, e,
                            "this eval runs locally" if local_fallback else "private evaluation waits")
        key = f"{host.name}@{host.host}:{host.port}/{host.cuda_device}" if host is not None else None
        if key != last_seen:
            if host is None:
                log.info("eval-offload host gone (%s); %s", p,
                         "heavy evals run locally" if local_fallback else "private evaluation waits")
            else:
                log.info("eval-offload host now %s (%s); heavy evals offload there "
                         "(wallet + consensus stay local)", host.name, host.host)
            last_seen = key
        return host

    return resolve


def build_scp_argv(host: RemoteHost, local_path: str, remote_path: str) -> list[str]:
    """The local ``scp -r`` argv that copies ``local_path`` → ``host:remote_path``,
    mirroring :func:`cascade.trainer.remote.build_ssh_argv`'s connection options.
    Note ``scp`` uses ``-P`` (capital) for the port, unlike ``ssh``'s ``-p``."""
    argv = ["scp", "-r", "-P", str(host.port), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new"]
    if host.key_path:
        argv += ["-i", str(Path(host.key_path).expanduser())]
    for opt in host.ssh_options:
        argv += ["-o", opt]
    argv += [local_path, f"{host.user}@{host.host}:{remote_path}"]
    return argv


def build_bench_remote_command(
    host: RemoteHost, remote_ckpt: str, remote_report: str, *,
    suites: str, num_samples: int = 100, batch_size: int = 512, max_series: int = 0,
    datasets: str = "", data_dir: str | None = None, device: str = "cuda",
    uv_bin: str = DEFAULT_UV_BIN, workers: str | int | None = None,
    extra: tuple[tuple[str, str], ...] = (),
    timeout_s: int | None = None,
) -> str:
    """The remote shell string that benchmarks ``remote_ckpt`` on the pod and
    writes ``remote_report``. Pure — safe to unit test. Mirrors the arg
    construction of :mod:`cascade.eval.benchmarks` so the report is identical to
    the local one. ``max_series`` (cascade bench) and ``datasets`` (gift gate's
    config subset) are emitted only when set; ``workers`` (the sidecar's
    ``--workers``) only when not the sequential 1; ``extra`` = further
    ``(remote_ckpt, remote_report)`` pairs swept in the same pass."""
    argv = [
        "cascade-benchmark", remote_ckpt, remote_report,
        "--suites", suites,
        "--num-samples", str(num_samples),
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    if max_series:
        argv += ["--max-series", str(max_series)]
    if datasets:
        argv += ["--gifteval-datasets", datasets]
    if data_dir:
        argv += ["--data-dir", data_dir]
    argv += bench_workers_arg(workers)
    for ck, rep in extra:
        argv += ["--extra-checkpoint", ck, rep]
    quoted = " ".join(shlex.quote(a) for a in argv)
    prefix = ""
    if host.cuda_device is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
    return (
        prefix
        + (f"timeout --signal=TERM --kill-after=15s {max(1, timeout_s)}s " if timeout_s else "")
        + f"{uv_bin} run --project {shlex.quote(f'{host.workdir}/benchmarks')} "
        + " ".join(bench_uv_args(suites)) + " "
        + quoted
    )


def run_bench_via_host(
    host: RemoteHost, ckpt_dir: str | Path, *, suites: str,
    num_samples: int = 100, batch_size: int = 512, max_series: int = 0,
    datasets: str = "", data_dir: str | None = None, device: str = "cuda",
    timeout_s: int = 3600, runner=None, workers: str | int | None = None,
) -> dict | None:
    """Benchmark one already-fetched ``ckpt_dir`` on ``host`` (GPU) and return the
    parsed ``cascade-benchmark`` report dict, or ``None`` on any failure.

    ``scp`` the checkpoint to the pod, run the sidecar, ``cat`` the report back.
    Never raises. ``runner`` is the subprocess bridge (defaults to
    :func:`cascade.trainer.remote.run_ssh`, which runs any argv — ssh or scp —
    under a timeout); injectable for tests.
    """
    return run_bench_via_host_many(
        host, [ckpt_dir], suites=suites, num_samples=num_samples, batch_size=batch_size,
        max_series=max_series, datasets=datasets, data_dir=data_dir, device=device,
        timeout_s=timeout_s, runner=runner, workers=workers)[0]


def run_bench_via_host_many(
    host: RemoteHost, ckpt_dirs: list[str | Path], *, suites: str,
    num_samples: int = 100, batch_size: int = 512, max_series: int = 0,
    datasets: str = "", data_dir: str | None = None, device: str = "cuda",
    timeout_s: int = 3600, runner=None, workers: str | int | None = None,
) -> list[dict | None]:
    """:func:`run_bench_via_host` for several checkpoints in ONE paired
    sidecar pass on the pod (``--extra-checkpoint``) — one report (or
    ``None``) per checkpoint, each identical to its solo sweep. Never raises."""
    run = runner or run_ssh
    deadline = time.monotonic() + timeout_s

    def step(argv, cap=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("public benchmark offload deadline expired")
        return run(argv, min(remaining, cap) if cap else remaining)
    n = len(ckpt_dirs)
    out: list[dict | None] = [None] * n
    if n == 0:
        return out
    name = Path(str(ckpt_dirs[0])).name or "ckpt"
    base = f"{host.workdir}/_eval_offload/{name}-{uuid.uuid4().hex}" + (f"-paired{n}" if n > 1 else "")
    if n == 1:
        ckpts, reports = [f"{base}/checkpoint"], [f"{base}/report.json"]
    else:
        ckpts = [f"{base}/checkpoint-{i}" for i in range(n)]
        reports = [f"{base}/report-{i}.json" for i in range(n)]
    try:
        prep = step(build_ssh_argv(
            host, f"rm -rf {shlex.quote(base)} && mkdir -p "
            + " ".join(shlex.quote(c) for c in ckpts)), 120)
        if prep.returncode != 0:
            log.warning("eval-offload prep failed on %s: %s", host.name, (prep.stderr or "")[-200:])
            return out
        for local, remote in zip(ckpt_dirs, ckpts, strict=True):
            # Copy the checkpoint's CONTENTS into the remote checkpoint dir.
            scp = step(build_scp_argv(host, f"{str(local).rstrip('/')}/.", remote))
            if scp.returncode != 0:
                log.warning("eval-offload scp to %s failed: %s", host.name,
                            (scp.stderr or "")[-300:])
                return out
        cmd = build_bench_remote_command(
            host, ckpts[0], reports[0], suites=suites,
            num_samples=num_samples, batch_size=batch_size, max_series=max_series,
            datasets=datasets, data_dir=data_dir, device=device, workers=workers,
            extra=tuple(zip(ckpts[1:], reports[1:], strict=True)),
            timeout_s=max(1, int(deadline - time.monotonic()) - 15),
        )
        proc = step(build_ssh_argv(host, cmd))
        if proc.returncode != 0:
            log.warning("eval-offload benchmark (%s) failed on %s (exit %s): %s",
                        suites, host.name, proc.returncode, (proc.stderr or "")[-400:])
            return out
        for i, rep in enumerate(reports):
            cat = step(build_ssh_argv(host, f"cat {shlex.quote(rep)}"), 120)
            if cat.returncode != 0:
                log.warning("eval-offload report missing on %s: %s", host.name,
                            (cat.stderr or "")[-200:])
                continue
            out[i] = json.loads(cat.stdout)
        return out
    except Exception as e:  # noqa: BLE001 — an eval helper must never raise into a round
        log.warning("eval-offload errored on %s: %s", host.name, e)
        return [None] * n
    finally:
        with contextlib.suppress(Exception):
            run(build_ssh_argv(host, f"rm -rf {shlex.quote(base)}"), 15)


def gift_rows_via_host(
    host: RemoteHost, ckpt_dir: str | Path, *,
    datasets: str = "", num_samples: int = 100, batch_size: int = 512,
    data_dir: str | None = None, device: str = "cuda", timeout_s: int = 3600,
    runner=None,
) -> dict | None:
    """GIFT-Eval gate rows for one checkpoint, run on ``host`` (GPU). Semantics
    match :func:`cascade.eval.benchmarks.run_gift_rows`: ``None`` ⇒ the gate is
    uncomputable; a ``status`` other than ``"ok"`` ⇒ gift-eval skipped/errored."""
    report = run_bench_via_host(
        host, ckpt_dir, suites="gift-eval", num_samples=num_samples,
        batch_size=batch_size, datasets=datasets, data_dir=data_dir,
        device=device, timeout_s=timeout_s, runner=runner,
    )
    return gift_rows_from_report(report)
