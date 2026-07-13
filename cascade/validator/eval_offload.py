"""Offload the validator's GPU-heavy benchmark eval to a remote GPU pod.

The validator scores rounds on the (GPU-less) orchestrator. The private-pool
duel is CPU-tuned, but two benchmark paths are too heavy for the CPU box inside
a round's budget:

* the public **GIFT-Eval gate** — gift-eval on BOTH king and challenger (a
  *paired* compare the validator must compute itself; see ``_gift_rows``), and
* the **cascade bench** — GIFT-Eval + BOOM + TIME on the king checkpoint (the
  validator-side fallback in ``_bench_metrics_via_sidecar`` when a manifest
  carries no trainer-stamped ``bench_scores``).

Both funnel through one primitive here — :func:`run_bench_via_host` — which
``scp``-s an already-fetched checkpoint to a GPU pod, runs the ``cascade-benchmark``
sidecar there (the same one the trainer bench uses), and returns the parsed
report. :func:`gift_rows_via_host` and :func:`bench_scores_via_host` are the two
thin parsers on top, matching the local
:func:`cascade.eval.benchmarks.run_gift_rows` / ``run_benchmarks`` semantics.

Wallet-safe: only a public checkpoint dir and the report cross to the pod — no
keys are forwarded, and every consensus decision (the paired bootstrap) stays on
the orchestrator. Best-effort: any failure returns ``None`` so the caller treats
the result as unavailable/uncomputable; it never raises into a round.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path

from ..eval.benchmarks import extract_bench_scores, gift_rows_from_report
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


def build_bench_remote_command(
    host: RemoteHost, remote_ckpt: str, remote_report: str, *,
    suites: str, num_samples: int = 100, batch_size: int = 512, max_series: int = 0,
    datasets: str = "", data_dir: str | None = None, device: str = "cuda",
    uv_bin: str = DEFAULT_UV_BIN,
) -> str:
    """The remote shell string that benchmarks ``remote_ckpt`` on the pod and
    writes ``remote_report``. Pure — safe to unit test. Mirrors the arg
    construction of :mod:`cascade.eval.benchmarks` so the report is identical to
    the local one. ``max_series`` (cascade bench) and ``datasets`` (gift gate's
    config subset) are emitted only when set."""
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
    quoted = " ".join(shlex.quote(a) for a in argv)
    prefix = ""
    if host.cuda_device is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
    return (
        prefix
        + f"{uv_bin} run --project {shlex.quote(f'{host.workdir}/benchmarks')} "
        + quoted
    )


def run_bench_via_host(
    host: RemoteHost, ckpt_dir: str | Path, *, suites: str,
    num_samples: int = 100, batch_size: int = 512, max_series: int = 0,
    datasets: str = "", data_dir: str | None = None, device: str = "cuda",
    timeout_s: int = 3600, runner=None,
) -> dict | None:
    """Benchmark one already-fetched ``ckpt_dir`` on ``host`` (GPU) and return the
    parsed ``cascade-benchmark`` report dict, or ``None`` on any failure.

    ``scp`` the checkpoint to the pod, run the sidecar, ``cat`` the report back.
    Never raises. ``runner`` is the subprocess bridge (defaults to
    :func:`cascade.trainer.remote.run_ssh`, which runs any argv — ssh or scp —
    under a timeout); injectable for tests.
    """
    run = runner or run_ssh
    base = f"{host.workdir}/_eval_offload/{Path(str(ckpt_dir)).name or 'ckpt'}"
    remote_ckpt = f"{base}/checkpoint"
    remote_report = f"{base}/report.json"
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
        cmd = build_bench_remote_command(
            host, remote_ckpt, remote_report, suites=suites,
            num_samples=num_samples, batch_size=batch_size, max_series=max_series,
            datasets=datasets, data_dir=data_dir, device=device,
        )
        proc = run(build_ssh_argv(host, cmd), timeout_s)
        if proc.returncode != 0:
            log.warning("eval-offload benchmark (%s) failed on %s (exit %s): %s",
                        suites, host.name, proc.returncode, (proc.stderr or "")[-400:])
            return None
        cat = run(build_ssh_argv(host, f"cat {shlex.quote(remote_report)}"), 120)
        if cat.returncode != 0:
            log.warning("eval-offload report missing on %s: %s", host.name, (cat.stderr or "")[-200:])
            return None
        report = json.loads(cat.stdout)
        run(build_ssh_argv(host, f"rm -rf {shlex.quote(base)}"), 60)  # best-effort cleanup
        return report
    except Exception as e:  # noqa: BLE001 — an eval helper must never raise into a round
        log.warning("eval-offload errored on %s: %s", host.name, e)
        return None


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


def bench_scores_via_host(
    host: RemoteHost, ckpt_dir: str | Path, *,
    num_samples: int = 100, max_series: int = 0, batch_size: int = 512,
    data_dir: str | None = None, device: str = "cuda", timeout_s: int = 3600,
    runner=None,
) -> dict | None:
    """Cascade bench (GIFT-Eval + BOOM + TIME) for one king checkpoint on ``host``
    (GPU), returning the six numbers or ``None`` when any suite is missing/errored.
    Semantics match the local :func:`cascade.eval.benchmarks.run_benchmarks` +
    :func:`extract_bench_scores` path in ``_bench_metrics_via_sidecar``."""
    report = run_bench_via_host(
        host, ckpt_dir, suites="gift-eval,boom,time", num_samples=num_samples,
        max_series=max_series, batch_size=batch_size, data_dir=data_dir,
        device=device, timeout_s=timeout_s, runner=runner,
    )
    return extract_bench_scores(report)
