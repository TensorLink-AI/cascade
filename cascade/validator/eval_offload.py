"""Offload the validator's GIFT-Eval gate compute to a remote GPU pod.

The validator scores rounds on the (GPU-less) orchestrator. The private-pool
duel is CPU-tuned, but the public GIFT-Eval no-regression gate must run gift-eval
on BOTH the king and challenger checkpoints — a *paired* comparison the validator
must compute itself (it cannot take the trainer's word on a two-miner gate) — and
the full 97-config battery is too heavy for the CPU box inside a round's budget.

This dispatches one checkpoint's gift-eval run to a GPU pod, reusing the
``cascade-benchmark`` sidecar whose report already carries the per-config rows +
data revision the gate consumes (parsed by
:func:`cascade.eval.benchmarks.gift_rows_from_report`, identical to the local
path). The flow: the validator fetches the checkpoint locally (cheap download),
``scp``-s it to the pod, runs gift-eval on the GPU there, and pulls the report
back — every consensus decision (the paired bootstrap) stays on the orchestrator.

Wallet-safe: only a public checkpoint dir and the report cross to the pod; no
keys are forwarded. Best-effort and consensus-honest: any failure returns
``None`` so the caller treats the gate as *uncomputable* (mirrors
:func:`cascade.eval.benchmarks.run_gift_rows`) — it never raises into a round.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path

from ..eval.benchmarks import gift_rows_from_report
from ..trainer.remote import RemoteHost, build_ssh_argv, run_ssh

log = logging.getLogger("cascade.validator.eval_offload")

# uv on the pod (runs the benchmarks/ sidecar's own locked env), matching
# cascade.trainer.bench_hook.BenchPlan.uv_bin.
DEFAULT_UV_BIN = "~/.local/bin/uv"


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


def build_gift_remote_command(
    host: RemoteHost, remote_ckpt: str, remote_report: str, *,
    datasets: str = "", num_samples: int = 100, batch_size: int = 512,
    data_dir: str | None = None, device: str = "cuda", uv_bin: str = DEFAULT_UV_BIN,
) -> str:
    """The remote shell string that gift-eval-benchmarks ``remote_ckpt`` on the
    pod and writes ``remote_report``. Pure — safe to unit test. Mirrors the arg
    construction of :func:`cascade.eval.benchmarks.run_gift_rows` (single
    ``gift-eval`` suite) so the report is identical to the local one."""
    argv = [
        "cascade-benchmark", remote_ckpt, remote_report,
        "--suites", "gift-eval",
        "--num-samples", str(num_samples),
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    if datasets:
        argv += ["--gifteval-datasets", datasets]
    if data_dir:
        argv += ["--data-dir", data_dir]
    quoted = " ".join(shlex.quote(a) for a in argv)
    prefix = ""
    if host.cuda_device is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
    return (
        prefix
        + f"{uv_bin} run --project {shlex.quote(f'{host.workdir}/benchmarks')} "
        + quoted
    )


def gift_rows_via_host(
    host: RemoteHost, ckpt_dir: str | Path, *,
    datasets: str = "", num_samples: int = 100, batch_size: int = 512,
    data_dir: str | None = None, device: str = "cuda", timeout_s: int = 3600,
    runner=None,
) -> dict | None:
    """Run gift-eval for one already-fetched ``ckpt_dir`` on ``host`` (GPU) and
    return the gate rows ``{"status", "rows", "revision"}``.

    Semantics match :func:`cascade.eval.benchmarks.run_gift_rows`: ``None`` means
    the sidecar produced nothing (gate uncomputable); a returned ``status`` other
    than ``"ok"`` means gift-eval was skipped/errored. Never raises. ``runner`` is
    the subprocess bridge (defaults to :func:`cascade.trainer.remote.run_ssh`,
    which runs any argv — ssh or scp — under a timeout); it is injectable for
    tests.
    """
    run = runner or run_ssh
    base = f"{host.workdir}/_eval_offload/{Path(str(ckpt_dir)).name or 'ckpt'}"
    remote_ckpt = f"{base}/checkpoint"
    remote_report = f"{base}/gift_report.json"
    try:
        prep = run(build_ssh_argv(
            host, f"rm -rf {shlex.quote(base)} && mkdir -p {shlex.quote(remote_ckpt)}"), 120)
        if prep.returncode != 0:
            log.warning("eval-offload prep failed on %s: %s", host.name, (prep.stderr or "")[-200:])
            return None
        # Copy the checkpoint's CONTENTS into the remote checkpoint dir.
        scp = run(build_scp_argv(host, f"{str(ckpt_dir).rstrip('/')}/.", remote_ckpt), timeout_s)
        if scp.returncode != 0:
            log.warning("eval-offload scp to %s failed: %s", host.name, (scp.stderr or "")[-300:])
            return None
        cmd = build_gift_remote_command(
            host, remote_ckpt, remote_report,
            datasets=datasets, num_samples=num_samples, batch_size=batch_size,
            data_dir=data_dir, device=device,
        )
        proc = run(build_ssh_argv(host, cmd), timeout_s)
        if proc.returncode != 0:
            log.warning("eval-offload gift-eval failed on %s (exit %s): %s",
                        host.name, proc.returncode, (proc.stderr or "")[-400:])
            return None
        cat = run(build_ssh_argv(host, f"cat {shlex.quote(remote_report)}"), 120)
        if cat.returncode != 0:
            log.warning("eval-offload report missing on %s: %s", host.name, (cat.stderr or "")[-200:])
            return None
        rows = gift_rows_from_report(json.loads(cat.stdout))
        run(build_ssh_argv(host, f"rm -rf {shlex.quote(base)}"), 60)  # best-effort cleanup
        return rows
    except Exception as e:  # noqa: BLE001 — a consensus-gate helper must never raise into a round
        log.warning("eval-offload errored on %s (gate uncomputable): %s", host.name, e)
        return None
