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
from dataclasses import asdict, replace
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


# Seconds the pre-upload compatibility probe may take.
WORKER_PROBE_SECONDS = 60
# Tail of a failed step's stderr kept in the validator's log.
STEP_STDERR_TAIL = 600
# What the host's code accepts, asked of the code itself so the probe also
# answers on a host that predates it. Run from ``host.workdir`` like the worker,
# so it imports the same package the worker would.
WORKER_PROBE = (
    "import dataclasses, json\n"
    "from cascade.shared.config.training import TrainingContractConfig as T\n"
    "from cascade.validator import private_worker as w\n"
    "print(json.dumps({'contract_fields': [f.name for f in dataclasses.fields(T)],\n"
    "                  'schemas': [getattr(w, 'SCHEMA', ''), getattr(w, 'SCHEMA_QUANTILES', '')]}))\n"
)


def check_private_worker(host, contract, *, quantile_levels=None, run=None,
                         timeout_s=WORKER_PROBE_SECONDS) -> None:
    """Refuse before any upload when ``host`` provably cannot serve this request.

    The worker rebuilds the contract with ``TrainingContractConfig(**fields)``,
    so a host on older code dies with a TypeError on every attempt, after the
    checkpoint has crossed the wire (2026-11-09: 8 hours of held rounds). The
    round holds either way; this names the cause once per attempt.

    Only a clear answer refuses. A probe that fails or returns nothing readable
    logs a warning and the eval proceeds exactly as it would have. A host that
    knows MORE fields than this validator only warns: the worker fills them
    from its defaults, as it always has.
    """
    run = run or run_ssh
    command = (f"cd {shlex.quote(host.workdir)} && "
               f"{shlex.quote(host.remote_python)} -c {shlex.quote(WORKER_PROBE)}")
    try:
        proc = run(build_ssh_argv(host, command), timeout_s)
        if proc.returncode:
            log.warning("private eval host %s: compatibility probe failed (exit %s): %s",
                        host.name, proc.returncode, _stderr_tail(proc))
            return
        info = json.loads(proc.stdout.strip().splitlines()[-1])
        remote_fields = {str(f) for f in info["contract_fields"]}
        remote_schemas = {str(s) for s in info["schemas"]}
    except Exception as e:  # noqa: BLE001 — the probe never decides a round on its own failure
        log.warning("private eval host %s: compatibility probe unreadable (%s: %s)",
                    host.name, type(e).__name__, e)
        return
    local_fields = set(asdict(contract))
    missing = sorted(local_fields - remote_fields)
    needs_quantiles = quantile_levels is not None
    from .private_worker import SCHEMA_QUANTILES

    if missing or (needs_quantiles and SCHEMA_QUANTILES not in remote_schemas):
        log.error("private eval host %s runs code older than this validator (contract fields "
                  "it lacks: %s; quantile requests supported: %s) — deploy this validator's "
                  "release to the host; the round holds until then",
                  host.name, ", ".join(missing) or "none", SCHEMA_QUANTILES in remote_schemas)
        raise PrivateEvalUnavailable("private evaluation worker is out of date")
    extra = sorted(remote_fields - local_fields)
    if extra:
        log.warning("private eval host %s runs code newer than this validator (extra contract "
                    "fields: %s)", host.name, ", ".join(extra))


def _stderr_tail(proc) -> str:
    return (getattr(proc, "stderr", "") or "")[-STEP_STDERR_TAIL:]


def forecasts_via_host(host, checkpoint, windows, *, contract, num_samples,
                       workers=1, timeout_s=3600, runner=None, quantile_levels=None):
    """Send only histories to an explicitly configured operator-owned host.

    Score returned samples locally against the original targets. A missing,
    corrupt, stale or partial reply holds the round; it never invents scores
    or silently switches a consensus comparison to another execution mode.
    ``quantile_levels`` set (DEC-CA-0067 ``quantile_head_after``) ⇒ the host is
    asked for each window's ``(C, H, num_q)`` quantiles at those levels
    instead of samples; a worker from before the knob refuses the request and
    the round holds — the validator never falls back to the sample path.
    """
    from ..eval.checkpoint_guard import CheckpointTampered, verify_checkpoint
    from .evaluator import EvaluatorError
    from .private_worker import CHECKPOINT_FILES, check_forecasts, write_request

    try:
        verify_checkpoint(checkpoint, contract)
    except CheckpointTampered as e:
        raise EvaluatorError(f"checkpoint_tampered: {e}") from e
    run = runner or run_ssh
    deadline = time.monotonic() + timeout_s
    work = f"_private_eval_offload/{uuid.uuid4().hex}"
    base = f"{host.workdir}/{work}"

    def step(argv, what):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PrivateEvalUnavailable("private evaluation deadline expired")
        proc = run(argv, remaining)
        if proc.returncode:
            log.error("private eval %s failed on %s (exit %s): %s",
                      what, host.name, proc.returncode, _stderr_tail(proc))
            raise PrivateEvalUnavailable("private evaluation transport/worker failed")
        return proc

    tmpdir = tempfile.TemporaryDirectory(prefix="cascade-private-eval-")
    try:
        check_private_worker(host, contract, quantile_levels=quantile_levels, run=run,
                             timeout_s=max(1, min(WORKER_PROBE_SECONDS,
                                                  deadline - time.monotonic())))
        local = Path(tmpdir.name)
        request = local / "request.npz"
        identity = write_request(request, checkpoint, windows, contract=contract,
                                 num_samples=num_samples, quantile_levels=quantile_levels)
        step(build_ssh_argv(host, f"umask 077 && mkdir -p {shlex.quote(base + '/checkpoint')}"),
             "workdir setup")
        for name in CHECKPOINT_FILES:
            step(build_scp_argv(host, str(Path(checkpoint) / name), f"{base}/checkpoint/{name}"),
                 f"upload of {name}")
        step(build_scp_argv(host, str(request), f"{base}/request.npz"), "upload of the request")
        argv = [host.remote_python, "-m", "cascade.validator.private_worker",
                f"{work}/checkpoint", f"{work}/request.npz", f"{work}/samples.npz",
                "--device", "cuda", "--workers", str(workers)]
        env = "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        if host.cuda_device is not None:
            env += f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
        command = (f"cd {shlex.quote(host.workdir)} && " + env
                   + f"timeout --signal=TERM --kill-after=15s {max(1, int(deadline - time.monotonic()) - 15)}s "
                   + " ".join(shlex.quote(a) for a in argv))
        step(build_ssh_argv(host, command), "worker")
        size = step(build_ssh_argv(host, f"stat -c %s {shlex.quote(base + '/samples.npz')}"),
                    "reply size check")
        depth = len(quantile_levels) if quantile_levels is not None else int(num_samples)
        budget = (sum(w.n_channels * depth * w.horizon * 8 for w in windows)
                  + (len(windows) + 1) * 8192)
        if not 0 < int(size.stdout.strip()) <= budget:
            raise PrivateEvalUnavailable("private evaluation response exceeds its shape budget")
        output = local / "samples.npz"
        download = build_scp_argv(host, "unused", "unused")
        download[-2:] = [f"{host.user}@{host.host}:{base}/samples.npz", str(output)]
        step(download, "download of the reply")
        # Validate EVERY window first (a corrupt/partial reply still holds
        # the round as infrastructure, never a miner fault), then stream
        # the samples one window at a time into the scorer: the full
        # duel's samples never sit in the validator's RAM at once.
        check_forecasts(output, identity, windows, num_samples, quantile_levels)
        streamed = StreamedForecasts(output, windows, num_samples, tmpdir,
                                     quantile_levels=quantile_levels)
        tmpdir = None              # owned by ``streamed`` now
        return streamed
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
        if tmpdir is not None:
            tmpdir.cleanup()


class StreamedForecasts:
    """The validated forecast samples of one private eval, yielded in window
    order straight from the downloaded NPZ (identical arrays and order to the
    old in-memory list). Owns the local temp dir; :meth:`close` (or exhausting
    the iterator) removes it. A read failure here is infrastructure
    (:class:`PrivateEvalUnavailable`) — the reply was already validated."""

    def __init__(self, path, windows, num_samples, tmpdir, quantile_levels=None):
        self.path, self.windows, self.num_samples = Path(path), list(windows), int(num_samples)
        self.quantile_levels = None if quantile_levels is None else tuple(quantile_levels)
        self._tmpdir = tmpdir
        self._gen = None

    def __len__(self) -> int:
        return len(self.windows)

    def __iter__(self):
        from .private_worker import iter_forecasts

        if self._tmpdir is None:
            raise PrivateEvalUnavailable("private evaluation samples already released")

        def gen():
            try:
                yield from iter_forecasts(self.path, self.windows, self.num_samples,
                                          self.quantile_levels)
            except Exception as e:  # noqa: BLE001 — validated already: never a miner fault
                raise PrivateEvalUnavailable(
                    f"private evaluation samples unreadable: {type(e).__name__}") from e
            finally:
                self.close()

        self._gen = gen()
        return self._gen

    def close(self) -> None:
        gen, self._gen = self._gen, None
        if gen is not None:
            with contextlib.suppress(Exception):
                gen.close()
        tmp, self._tmpdir = self._tmpdir, None
        if tmp is not None:
            with contextlib.suppress(Exception):
                tmp.cleanup()

    def __del__(self):
        self.close()


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


def build_scp_argv(host: RemoteHost, local_path: str | list[str],
                   remote_path: str) -> list[str]:
    """The local ``scp -r`` argv that copies ``local_path`` (one path, or a list
    of files into the ``remote_path`` directory) → ``host:remote_path``,
    mirroring :func:`cascade.trainer.remote.build_ssh_argv`'s connection options.
    Note ``scp`` uses ``-P`` (capital) for the port, unlike ``ssh``'s ``-p``."""
    argv = ["scp", "-r", "-P", str(host.port), "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new"]
    if host.key_path:
        argv += ["-i", str(Path(host.key_path).expanduser())]
    for opt in host.ssh_options:
        argv += ["-o", opt]
    argv += ([str(x) for x in local_path] if isinstance(local_path, list) else [local_path])
    argv += [f"{host.user}@{host.host}:{remote_path}"]
    return argv


# What the benchmark sidecar reads from a checkpoint (the wrapper loads
# config.json + weights.safetensors; model.py is its import). A trained
# checkpoint also carries optimizer.safetensors + weights_stable.safetensors
# (the warm-start lineage branch) — 2/3 of its bytes, never read by a bench,
# and with them a 2-checkpoint sweep spent ~3.5 h of its 4 h budget on scp
# (2026-11-03). Same set the private eval pod receives.
BENCH_UPLOAD_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")
# Upload budget per checkpoint, separate from the benchmark's own timeout.
BENCH_UPLOAD_SECONDS_PER_CKPT = 7200
# Content-addressed checkpoints kept on the pod between sweeps (a retried or
# re-held settlement re-uses them instead of uploading again).
BENCH_CACHE_KEEP = 6
_UPLOAD_MANIFEST = ".cascade_upload_manifest"


def bench_upload_files(ckpt_dir: str | Path) -> list[Path]:
    """The files of ``ckpt_dir`` a bench needs, in :data:`BENCH_UPLOAD_FILES`
    order (absent ones skipped)."""
    root = Path(str(ckpt_dir))
    return [root / f for f in BENCH_UPLOAD_FILES if (root / f).is_file()]


def bench_upload_manifest(files: list[Path]) -> str:
    """``name size`` per file — what a cached copy on the pod must match."""
    return "".join(f"{f.name} {f.stat().st_size}\n" for f in files)


def _cacheable(ckpt_dir: str | Path) -> bool:
    # only content-addressed fetch dirs (sha256-<digest>) are safe to key by name
    name = Path(str(ckpt_dir)).name
    return name.startswith("sha256-") and len(name) > len("sha256-") + 16


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
    upload_timeout_s: int | None = None,
) -> list[dict | None]:
    """:func:`run_bench_via_host` for several checkpoints in ONE paired
    sidecar pass on the pod (``--extra-checkpoint``) — one report (or
    ``None``) per checkpoint, each identical to its solo sweep. Never raises.

    Only :data:`BENCH_UPLOAD_FILES` are uploaded. Content-addressed
    checkpoints (``sha256-…`` dirs) go to a per-digest cache on the pod and
    are re-used when a previous upload of the same files completed, so a
    retried settlement does not upload again. The upload has its own budget
    (``upload_timeout_s``, default :data:`BENCH_UPLOAD_SECONDS_PER_CKPT` per
    checkpoint); ``timeout_s`` is the benchmark's, counted from when the
    uploads finish."""
    run = runner or run_ssh
    n = len(ckpt_dirs)
    out: list[dict | None] = [None] * n
    if n == 0:
        return out
    up_budget = (int(upload_timeout_s) if upload_timeout_s
                 else BENCH_UPLOAD_SECONDS_PER_CKPT * n)
    deadline = time.monotonic() + up_budget

    def step(argv, cap=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("public benchmark offload deadline expired")
        return run(argv, min(remaining, cap) if cap else remaining)
    name = Path(str(ckpt_dirs[0])).name or "ckpt"
    base = f"{host.workdir}/_eval_offload/{name}-{uuid.uuid4().hex}" + (f"-paired{n}" if n > 1 else "")
    cache = f"{host.workdir}/_eval_offload/cache"
    tag = uuid.uuid4().hex[:12]
    keys = [Path(str(d)).name if _cacheable(d) else "" for d in ckpt_dirs]
    if n == 1:
        scratch, reports = [f"{base}/checkpoint"], [f"{base}/report.json"]
    else:
        scratch = [f"{base}/checkpoint-{i}" for i in range(n)]
        reports = [f"{base}/report-{i}.json" for i in range(n)]
    ckpts = [f"{cache}/{k}" if k else s for k, s in zip(keys, scratch, strict=True)]
    try:
        # prep: the run dir, the cache, and a prune of old cache entries /
        # abandoned partial uploads (never the ones this sweep uses)
        keep = " ".join(f"-e {shlex.quote(k)}" for k in keys if k) or "-e ''"
        prune = (f"cd {shlex.quote(cache)} && rm -rf -- *.part-* && "
                 f"ls -1t | grep -vxF {keep} | tail -n +{BENCH_CACHE_KEEP} "
                 "| xargs -r rm -rf --")
        prep = step(build_ssh_argv(
            host, f"rm -rf {shlex.quote(base)} && mkdir -p {shlex.quote(base)} "
            + " ".join(shlex.quote(c) for k, c in zip(keys, ckpts, strict=True) if not k)
            + f" {shlex.quote(cache)} && ({prune} || true)"), 120)
        if prep.returncode != 0:
            log.warning("eval-offload prep failed on %s: %s", host.name, (prep.stderr or "")[-200:])
            return out
        for local, remote, key in zip(ckpt_dirs, ckpts, keys, strict=True):
            files = bench_upload_files(local)
            if not files:
                # not a checkpoint layout we know — copy the whole dir (legacy)
                scp = step(build_scp_argv(host, f"{str(local).rstrip('/')}/.", remote))
                if scp.returncode != 0:
                    log.warning("eval-offload scp to %s failed: %s", host.name,
                                (scp.stderr or "")[-300:])
                    return out
                continue
            manifest = bench_upload_manifest(files)
            if key:
                probe = step(build_ssh_argv(
                    host, f"head -c 65536 {shlex.quote(remote + '/' + _UPLOAD_MANIFEST)} "
                    f"2>/dev/null && touch {shlex.quote(remote)}"), 60)
                if probe.returncode == 0 and (probe.stdout or "") == manifest:
                    log.info("eval-offload: %s already on %s — upload skipped",
                             key[-24:], host.name)
                    continue
            dest = f"{remote}.part-{tag}" if key else remote
            if key:
                mk = step(build_ssh_argv(host, f"mkdir -p {shlex.quote(dest)}"), 60)
                if mk.returncode != 0:
                    log.warning("eval-offload prep failed on %s: %s", host.name,
                                (mk.stderr or "")[-200:])
                    return out
            t0 = time.monotonic()
            scp = step(build_scp_argv(host, [str(f) for f in files], dest + "/"))
            if scp.returncode != 0:
                log.warning("eval-offload scp to %s failed: %s", host.name,
                            (scp.stderr or "")[-300:])
                return out
            log.info("eval-offload: uploaded %d file(s), %.1f GB of %s to %s in %.0fs",
                     len(files), sum(f.stat().st_size for f in files) / 1e9,
                     Path(str(local)).name[-24:], host.name, time.monotonic() - t0)
            if key:
                # publish atomically: the manifest is written last, inside the
                # partial dir, then the dir replaces any stale cached copy
                fin = step(build_ssh_argv(
                    host, f"printf %s {shlex.quote(manifest)} > "
                    f"{shlex.quote(dest + '/' + _UPLOAD_MANIFEST)} && rm -rf {shlex.quote(remote)} "
                    f"&& mv {shlex.quote(dest)} {shlex.quote(remote)}"), 120)
                if fin.returncode != 0:
                    log.warning("eval-offload cache publish failed on %s: %s", host.name,
                                (fin.stderr or "")[-200:])
                    return out
        # the benchmark's own budget starts now, not at the first scp
        deadline = time.monotonic() + timeout_s
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
            deadline = max(deadline, time.monotonic() + 120)
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
            # the run dir (reports + uncached copies) and this sweep's partial
            # uploads; the cache itself stays for the next sweep
            run(build_ssh_argv(host, f"rm -rf {shlex.quote(base)} "
                               f"{shlex.quote(cache)}/*.part-{tag}"), 15)


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
