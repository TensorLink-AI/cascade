"""Host admission guards shared by every rent / join path (2026-10-31/11-01).

* :func:`sync_host_blocklist` / :func:`host_blocked_reason` — ``[round]
  funded_host_blocklist`` (permanent) merged with the expiring host
  quarantine file, for Lium rent selection and the operator-lane join gate;
* :func:`multi_gpu_preflight_failure` — the N-rank NCCL all_reduce preflight
  (:func:`cascade.trainer.remote.probe_nccl_allreduce`) a multi-GPU pod must
  pass before a leg is admitted to it, with the per-SKU bus-bandwidth floor;
* :func:`quarantine_host_ip` — the 24 h host quarantine a throughput abort, a
  first-step deadline or a failed preflight records.

Pure functions of the runner (``runner.cfg.round`` read with defaults), so the
TrainerRunner mixins and their unit fakes share them. Every verdict here is an
OPERATOR-side fault: nothing burns a miner's attempt.
"""

from __future__ import annotations

import logging

log = logging.getLogger("cascade.trainer")

NCCL_PREFLIGHT = "NCCL PREFLIGHT"
# A host that hung a collective, trained at a fraction of the fleet's rate or
# never reached its first step is a hardware/fabric fault: it stays out for a
# day (the stale-image class), not the 30-min transient first quarantine.
HOST_FAULT_QUARANTINE_SECONDS = 24 * 3600.0


def _round(runner):
    return getattr(getattr(runner, "cfg", None), "round", None)


def sync_host_blocklist(runner) -> tuple[str, ...]:
    """Point the process-wide provider blocklist at the live config's
    ``[round] funded_host_blocklist`` (replace, never merge: an IP removed
    from the knob is rentable again). Never raises."""
    try:
        from ..provision.core import set_host_blocklist

        return set_host_blocklist(getattr(_round(runner), "funded_host_blocklist", ()) or ())
    except Exception as e:  # noqa: BLE001 — a config glitch must never sink a rent
        log.warning("host blocklist sync failed (keeping the previous list): %s", e)
        return ()


def host_blocked_reason(runner, ip: str) -> str:
    """Why ``ip`` must not be used (blocklisted, or under quarantine), or ``""``."""
    ip = str(ip or "").strip()
    if not ip:
        return ""
    sync_host_blocklist(runner)
    try:
        from ..provision.core import quarantined_hosts

        why = quarantined_hosts().get(ip, "")
    except Exception as e:  # noqa: BLE001 — an unreadable file blocks nothing
        log.warning("host quarantine read failed (ignored): %s", e)
        return ""
    return f"host {ip} is blocked: {why}" if why else ""


def quarantine_host_ip(ip: str, reason: str, *,
                       seconds: float = HOST_FAULT_QUARANTINE_SECONDS) -> None:
    """Quarantine ``ip`` for ``seconds`` (best-effort; loopback skipped)."""
    ip = str(ip or "").strip()
    if not ip or ip.startswith("127.") or ip in ("localhost", "::1"):
        return
    try:
        from ..provision.core import record_host_quarantine

        record_host_quarantine(ip, reason, seconds=seconds, host_fault=True)
    except Exception as e:  # noqa: BLE001
        log.warning("host quarantine for %s failed (ignored): %s", ip, e)


def preflight_gpu_count(host, sku: str) -> int:
    """GPUs a leg on ``host`` trains on: its pinned ``CUDA_VISIBLE_DEVICES``
    list when set, else the layout of ``sku`` (``8xH200`` → 8)."""
    dev = getattr(host, "cuda_device", None)
    if dev is not None and str(dev).strip():
        return len([d for d in str(dev).split(",") if d.strip()])
    from ..provision.layout import layout_gpus

    try:
        return int(layout_gpus(sku or getattr(host, "sku", "") or ""))
    except Exception:  # noqa: BLE001 — an unparseable SKU is one GPU
        return 1


def multi_gpu_preflight_failure(runner, host, sku: str, label: str, *, probe=None) -> str:
    """Why ``host`` must NOT receive a leg on multi-GPU grounds, or ``""``.

    Single-GPU legs and ``funded_host_nccl_timeout_seconds = 0`` skip it. A
    hang, an NCCL/CUDA error or a wrong sum rejects the pod — always, floor
    or not; a measured bus bandwidth below ``funded_host_nccl_busbw_floor[sku]``
    rejects it too. A rejected host is quarantined for a day. A probe that
    could not even reach the pod (ssh transport) is logged and passes: the
    transport has its own guards, and this check must never sink a leg over
    an ssh hiccup."""
    rnd = _round(runner)
    timeout = float(getattr(rnd, "funded_host_nccl_timeout_seconds", 0) or 0)
    n = preflight_gpu_count(host, sku)
    if timeout <= 0 or n <= 1:
        return ""
    if probe is None:
        from .remote import probe_nccl_allreduce as probe
    res = probe(host, nprocs=n, timeout=timeout)
    ip = str(getattr(host, "host", "") or "")
    if res.inconclusive:
        log.warning("%s: %s inconclusive (%s) — %d-GPU fabric not verified on this pod",
                    label, NCCL_PREFLIGHT, res.error, n)
        return ""
    why = ""
    if not res.ok:
        why = f"{NCCL_PREFLIGHT} failed on {n} GPU(s): {res.error}"
    else:
        from ..shared.config import funded_host_bench_floor_for

        floor = funded_host_bench_floor_for(
            getattr(rnd, "funded_host_nccl_busbw_floor", ()) or (), sku)
        log.info("%s: %s %d-rank all_reduce bus bandwidth %.1f GB/s (floor %s, sku %s)",
                 label, NCCL_PREFLIGHT, res.world, res.busbw_gbps or 0.0,
                 f"{floor:g}" if floor > 0 else "off", sku)
        if floor > 0 and (res.busbw_gbps or 0.0) < floor:
            why = (f"{NCCL_PREFLIGHT}: comm-bound host — {res.world}-rank all_reduce bus "
                   f"bandwidth {res.busbw_gbps or 0.0:.1f} GB/s < floor {floor:g} for {sku}")
    if why:
        log.error("%s: %s — pod rejected, host %s quarantined", label, why, ip or "?")
        quarantine_host_ip(ip, f"{label}: {why}")
    return why
