"""Ephemeral GPU-pod provisioner for the cascade trainer's remote data plane.

The trainer (``cascade.trainer.remote``) SSHes into a *static* ``hosts.toml`` of
GPU pods; it does not provision, check availability, or fall back between
providers. This wrapper fills that gap **outside** the trainer: it rents N pods
of one GPU SKU on the first provider (in priority order) that has capacity,
waits for SSH, templates a ``hosts.toml`` matching
``scripts/remote_hosts.example.toml``, optionally kicks off
``cascade-trainer --remote-hosts hosts.toml``, and **always** tears the pods
back down — even on failure or Ctrl-C.

Design mirrors ``remote.py``: the pure parts (provider selection order,
``hosts.toml`` templating, provider-response parsing, and the launch/teardown
control flow) are unit-tested; the actual cloud calls live behind the
:class:`Provider` boundary (``lium`` CLI / Shadeform REST) and are the only
untested surface.

Contract with the pod image (``deploy/Dockerfile`` / ``entrypoint.sh``): the
image runs ``sshd`` and injects the orchestrator's public key from the
``SSH_PUBKEY`` container env at launch. It bakes **no** secrets and **no**
wallet. Hippius credentials never touch the pod's disk — they are listed under
``forward_env`` in ``hosts.toml`` so the orchestrator pipes them over each
dispatch's SSH stdin — never on the remote command line, which is world-readable
in ``/proc`` on pods running miner code (see ``remote.build_remote_command``).

Reproducibility: the worker image MUST be pinned by ``@sha256:`` digest, not a
tag — ``chain.toml [training] expected_gpu`` and byte-exact re-derivation depend
on every pod running the identical image on the identical SKU. A round is never
split across two SKUs or two providers.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

log = logging.getLogger("cascade.provision.core")

# ── defaults (all overridable on the CLI) ────────────────────────────────────

DEFAULT_SKU = "L40S"
DEFAULT_POD_COUNT = 2                       # king + challenger
# Priority order, first provider with capacity wins. Targon is intentionally
# absent: for L40S it only offers bigger cards. Register it in _PROVIDER_FACTORIES
# and add its name here to use it for other SKUs — the seam is provider-agnostic.
DEFAULT_PROVIDER_PRIORITY = ("lium", "shadeform")
# Pods carry NO credentials (2026-09-28): a miner is root on a payer pod, and
# any operator lane may lose its network namespace and run the generator as a
# plain child — either could read a forwarded key from the worker environment
# (the 2026-09-24 leak route). Isolated hosts are harvested by the orchestrator
# (PR #320) instead of uploading under their own identity. Listing a secret here
# again is a deliberate decision, never a convenience.
DEFAULT_FORWARD_ENV: tuple[str, ...] = ()
DEFAULT_REMOTE_PYTHON = "/root/cascade/.venv/bin/python"
DEFAULT_WORKDIR = "/root/cascade"
DEFAULT_SSH_OPTIONS = (
    "StrictHostKeyChecking=accept-new",
    "ServerAliveInterval=30",
    "ServerAliveCountMax=120",
)
DEFAULT_SSH_PORT = 22
DEFAULT_READY_TIMEOUT = 900.0               # provider "active" + sshd reachable
POD_POLL_INTERVAL = 10.0
# A pod that hasn't APPEARED in `lium ps` by now never will — the `lium up`
# failed (deploys are client-driven and show RUNNING within ~1 min live).
# Distinct from the ready timeout: appeared-but-booting keeps polling.
LIUM_APPEAR_TIMEOUT = 180.0


class ProvisionError(RuntimeError):
    """A provisioning step failed (capacity, launch, teardown, or config)."""


class UncertainCreateError(ProvisionError):
    """A create may have allocated; retain its intent regardless of error prose."""


class PriceCapUnavailable(ProvisionError):
    """This provider cannot honor a requested price ceiling before creating."""


# ── value types shared across providers ──────────────────────────────────────


@dataclass(frozen=True)
class LaunchSpec:
    """Everything a provider needs to launch a homogeneous batch of pods.

    All pods in one spec share ``sku`` and ``image`` — the ``expected_gpu`` pin
    forbids mixing SKUs, and the digest pin forbids mixing images.
    """

    sku: str
    count: int
    image: str                              # digest-pinned worker image (…@sha256:…)
    ssh_pubkey: str                         # injected into the pod as $SSH_PUBKEY
    ssh_port: int = DEFAULT_SSH_PORT
    name_prefix: str = "cascade-pod"
    # Pod shape: adapters must rent machines with EXACTLY this many GPUs of
    # ``sku`` — the fleet plan fans one hosts.toml lane out per GPU, so a
    # smaller machine strands lanes and a bigger one bills idle silicon. The
    # health gate re-asserts the shape on the booted pod.
    gpus_per_pod: int = 1
    # Marketplace machine ids a launch must NOT pick — the replacement path
    # lists offers deterministically, so without this a replacement re-rents
    # the exact lemon that just failed its boot gate. Adapters that cannot
    # map ids to offers may ignore it.
    exclude_ids: tuple[str, ...] = ()
    # Facilities (provider regions) a launch must NOT pick. An image_digest
    # gate failure is a host-image-cache property of the FACILITY, not the
    # machine (2026-08-30, r51: a kansascity replacement booted the same stale
    # v0.6.0 template its predecessor did), so the replacement path excludes
    # the whole region rather than one machine. Adapters without region
    # placement may ignore it.
    exclude_regions: tuple[str, ...] = ()
    # Open-market launch (owner 2026-09-20): when set, the adapter may place
    # the pod on ANY of these GPU types — the cheapest fitting executor by
    # per-leg cost wins (``LiumProvider.list_offers``) — and reports the type
    # it landed on through ``sku_of(pod_id)``. ``sku`` stays the nominal /
    # first choice for adapters without the seam.
    sku_choices: tuple[str, ...] = ()
    # Provider-side auto-terminate, seconds from create (0 = none): Lium
    # ``lium up --ttl``, Shadeform ``auto_delete.date_threshold``. The
    # backstop that keeps a pod from billing after the trainer that rented it
    # died; adapters without the feature ignore it.
    ttl_seconds: float = 0.0


def lium_ttl_arg(ttl_seconds: float) -> str:
    """``lium up --ttl`` value for ``ttl_seconds`` (whole minutes, rounded
    UP so the pod never dies before the wall it was sized for); "" = none."""
    if not ttl_seconds or ttl_seconds <= 0:
        return ""
    return f"{max(1, math.ceil(float(ttl_seconds) / 60.0))}m"


def shadeform_delete_at(ttl_seconds: float, *, now: float | None = None) -> str:
    """Shadeform ``auto_delete.date_threshold`` (RFC 3339, UTC) for a pod that
    must be gone ``ttl_seconds`` from now; "" = none."""
    if not ttl_seconds or ttl_seconds <= 0:
        return ""
    t = (time.time() if now is None else float(now)) + math.ceil(float(ttl_seconds))
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def executor_price_per_hour(executor: dict) -> float | None:
    """An executor's USD/h from ``lium ls`` JSON (``price_per_hour``, else
    ``price_per_gpu_hour``); None when the listing carries neither."""
    for key in ("price_per_hour", "price_per_gpu_hour"):
        v = executor.get(key)
        if v is None or isinstance(v, bool):
            continue
        try:
            import math

            price = float(v)
            if math.isfinite(price) and price > 0:
                return price
        except (TypeError, ValueError):
            continue
    return None


def image_digest_of(image: str) -> str:
    """The ``sha256:<hex>`` digest pinned in an image ref (empty when unpinned).

    Injected into pods as ``CASCADE_TRAIN_IMAGE_DIGEST`` at launch — the health
    gate REQUIRES it on image-boot pods and final workers refuse a mismatched
    runtime, so an image-boot pod without it can never pass health (found in a
    live image-boot dress rehearsal, 2026-07-15)."""
    _, sep, digest = image.partition("@")
    return digest if sep and digest.startswith("sha256:") else ""


def lium_image_ref(image: str) -> str:
    """The image ref to hand lium's launch API: any ``@sha256:`` pin stripped.

    Lium's API 400-rejects the canonical digest-pinned form
    ``repo[:tag]@sha256:<64hex>`` ("Digest must be sha256: followed by 64 hex
    characters" — their parser chokes on the full ref; observed live
    2026-08-28, killed every lium pod that round). So lium launches send the
    tag form (``repo:tag``) when the ref carries one, else the bare repo.

    This does NOT weaken the reproducibility pin: the launch env still carries
    the exact digest (``CASCADE_TRAIN_IMAGE_DIGEST``, derived from the
    ORIGINAL ref — see :meth:`LiumProvider.launch`), and the health gate
    byte-compares the booted pod's digest against the chain.toml pin before
    the pod can ever enter hosts.toml — a wrong-image pod is bounced there.
    """
    base, sep, digest = image.partition("@")
    if sep and digest.startswith("sha256:"):
        return base
    return image


LIUM_API_BASE = "https://lium.io/api"

# ── executor cooldown ────────────────────────────────────────────────────────
# Lium accepts a rent on an executor we tore a pod down on moments earlier and
# then deploys NOTHING (post-teardown cooldown): the readiness wait burns its
# full timeout and the pod is torn down as a lemon (2026-09-12: the king at
# 10:11 and two payer legs at 15:11/15:27, 15 min each, all on executors the
# previous attempt had just vacated). Every terminate records the executor;
# listings exclude it for LIUM_EXECUTOR_COOLDOWN_SECONDS. Persisted to a file
# so it survives trainer restarts and is shared by every provider instance
# (operator + per-payer) on this box.
LIUM_EXECUTOR_COOLDOWN_SECONDS = 1800.0
LIUM_COOLDOWN_FILE_ENV = "CASCADE_LIUM_COOLDOWN_FILE"


def lium_cooldown_path() -> Path:
    return Path(os.environ.get(
        LIUM_COOLDOWN_FILE_ENV, "~/.cache/cascade/lium_executor_cooldown.json")).expanduser()


def _load_cooldowns(path: Path) -> dict[str, float]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in dict(raw).items()}
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}


def record_executor_cooldown(executor_id: str, *, path: Path | None = None,
                             now: float | None = None) -> None:
    """Remember that ``executor_id`` just had a pod torn down (best-effort)."""
    if not executor_id:
        return
    path = path or lium_cooldown_path()
    now = time.time() if now is None else now
    try:
        d = _load_cooldowns(path)
        d = {k: v for k, v in d.items() if now - v < LIUM_EXECUTOR_COOLDOWN_SECONDS}
        d[str(executor_id)] = now
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        log.warning("lium: could not record executor cooldown for %s: %s", executor_id, e)


def cooling_executors(*, path: Path | None = None, now: float | None = None,
                      ttl: float = LIUM_EXECUTOR_COOLDOWN_SECONDS) -> set[str]:
    """Executor ids we vacated less than ``ttl`` seconds ago."""
    path = path or lium_cooldown_path()
    now = time.time() if now is None else now
    return {k for k, v in _load_cooldowns(path).items() if now - v < ttl}


# ── host quarantine ──────────────────────────────────────────────────────────
# One PHYSICAL host serves several executor ids, and a lemon host stays a lemon
# under every one of them (2026-09-13: 91.224.44.222 listed 37f19d50 / 8ae82436
# / bec5e572 / cafb2ca6 — it booted a stale cached worker under the pinned tag,
# burned three miners, then sat RUNNING with no ports for the full 900 s on
# every later rent). Cooling one executor id re-picks the SAME machine under its
# next id, so a lemon is remembered by host IP: every executor on that host is
# skipped for LIUM_HOST_QUARANTINE_SECONDS. Persisted like the cooldown file.
LIUM_HOST_QUARANTINE_SECONDS = 24 * 3600.0
LIUM_QUARANTINE_FILE_ENV = "CASCADE_LIUM_QUARANTINE_FILE"


def lium_quarantine_path() -> Path:
    return Path(os.environ.get(
        LIUM_QUARANTINE_FILE_ENV, "~/.cache/cascade/lium_host_quarantine.json")).expanduser()


def _load_quarantine(path: Path) -> dict[str, dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out = {}
        for k, v in dict(raw).items():
            if isinstance(v, dict) and "until" in v:
                out[str(k)] = {**v, "until": float(v["until"]), "reason": str(v.get("reason", ""))}
        return out
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return {}


def record_host_quarantine(host: str, reason: str, *,
                           seconds: float | None = None,
                           path: Path | None = None, now: float | None = None) -> None:
    """Exclude a faulty host; use a short first quarantine unless explicitly pinned."""
    from ..funding.faults import classify_rent_failure

    if classify_rent_failure(reason) in {"auth", "rate_limited", "no_capacity"}:
        log.warning("host quarantine skipped: account/API failure is not a hardware fault")
        return
    host = str(host or "").strip()
    if not host:
        return
    path = path or lium_quarantine_path()
    now = time.time() if now is None else now
    try:
        from ..shared.atomic import path_lock, write_text_atomic

        with path_lock(path):
            d = {k: v for k, v in _load_quarantine(path).items() if v["until"] > now}
            prev = d.get(host)
            count = int(prev.get("count", 1)) + 1 if prev else 1
            duration = seconds
            if duration is None:
                # One transient boot/transport failure gets thirty minutes;
                # repeated failures get three hours. A stale runtime/driver
                # needs intervention, so keep the existing 24-hour exclusion.
                stale = any(t in reason.lower() for t in (
                    "runtime rejected", "stale image", "driver too old", "wrong runtime"))
                duration = LIUM_HOST_QUARANTINE_SECONDS if stale else (10800 if count > 1 else 1800)
            until = max(now + duration, prev["until"] if prev else 0.0)
            d[host] = {"until": until, "reason": str(reason)[:300], "count": count}
            write_text_atomic(path, json.dumps(d, sort_keys=True))
        log.warning("lium: host %s quarantined for %.1f h — %s", host, (until - now) / 3600,
                    reason)
    except OSError as e:
        log.warning("lium: could not record host quarantine for %s: %s", host, e)


def quarantined_hosts(*, path: Path | None = None,
                      now: float | None = None) -> dict[str, str]:
    """``{host ip: reason}`` for every host still under quarantine."""
    path = path or lium_quarantine_path()
    now = time.time() if now is None else now
    return {k: v["reason"] for k, v in _load_quarantine(path).items() if v["until"] > now}
# Lium caps ``POST /templates`` at 20 requests per hour PER CLIENT IP, and
# ``lium up --image`` creates a throw-away template on EVERY call — so a round's
# worth of launches (king + funded legs + retries) silently exhausted the hour
# on 2026-09-12 (02:43–03:55: every launch "hung", no pod ever appeared, the
# CLI's "Rate limit exceeded" went to a log nobody reads). One PERSISTENT
# template per (image, pubkey, port, digest) is created once and rented by id.


def lium_template_payload(image: str, *, ssh_pubkey: str, ssh_port: int) -> dict:
    """The ``POST /templates`` body for the worker image — the same shape the
    CLI's ephemeral template uses (``lium/cli/up/actions.py``), minus
    ``one_time_template`` so it survives its pods and can be reused.

    ``image`` is the canonical digest-pinned ref; Lium's template takes the
    repo and tag separately (it rejects the ``repo@sha256:…`` form in
    ``docker_image``) but carries the digest in its own
    ``docker_image_digest`` field — that is what pins the pull. The digest
    ALSO rides the container env (``CASCADE_TRAIN_IMAGE_DIGEST``) for the
    health gate's byte-compare — the same posture as :func:`lium_image_ref`.

    2026-09-12: with ``docker_image_digest`` left empty, a host that already
    held an older image under the pinned TAG started that container, and the
    injected env made it pass the digest gate (91.224.44.222/.223 ran the
    Aug-27 worker-v0.7.0 code as "v0.8.0"; every leg there died at the 1800 s
    stall and the miners were burned). The digest field closes the pull side;
    ``[round] worker_code_fingerprint`` closes the verification side.
    """
    ref = lium_image_ref(image)
    repo, sep, tag = ref.rpartition(":")
    if not sep or "/" in tag:            # no tag (the colon belonged to a registry port)
        repo, tag = ref, "latest"
    env = {"SSH_PUBKEY": ssh_pubkey}
    digest = image_digest_of(image)
    if digest:
        env["CASCADE_TRAIN_IMAGE_DIGEST"] = digest
    ports = [22] + ([int(ssh_port)] if int(ssh_port) != 22 else [])
    payload = {
        "docker_image": repo,
        "docker_image_tag": tag,
        "docker_image_digest": digest or "",
        "internal_ports": ports,
        "startup_commands": "",
        "category": "UBUNTU",
        "container_start_immediately": True,
        "entrypoint": "",
        "environment": env,
        "is_private": True,
        "one_time_template": False,
        "is_temporary": False,
        "volumes": ["/workspace"],
    }
    payload["name"] = lium_template_name(payload)
    payload["description"] = payload["readme"] = f"cascade worker {repo}:{tag} (sshd on {ssh_port})"
    return payload


# The template fields that decide what a pod boots: the name is derived from
# them, but the payer owns the account and can pre-create a template under
# that name with other contents, so reuse compares every one of them.
_TEMPLATE_PINNED = ("docker_image", "docker_image_tag", "docker_image_digest",
                    "internal_ports", "environment", "entrypoint", "startup_commands")


def lium_template_name(payload: dict) -> str:
    """Deterministic, content-addressed template name: the same image, pubkey,
    port and digest always resolve to the same template on the account."""
    import hashlib
    import json

    keyed = {k: payload.get(k, "") for k in _TEMPLATE_PINNED}
    h = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"cascade-worker-{payload['docker_image_tag']}-{h}"


def _template_diff(t: dict, payload: dict) -> list[str]:
    """Pinned fields where template ``t`` differs from ``payload``; a field the
    API did not return counts as different (unverifiable ⇒ not ours)."""
    def norm(k, v):
        if k == "internal_ports":
            return sorted(int(x) for x in (v or []))
        if k == "environment":
            return {str(a): str(b) for a, b in (v or {}).items()}
        return "" if v is None else str(v)

    out = []
    for k in _TEMPLATE_PINNED:
        if k not in t:
            out.append(k)
            continue
        try:
            same = norm(k, t[k]) == norm(k, payload.get(k, ""))
        except (TypeError, ValueError, AttributeError):
            same = False
        if not same:
            out.append(k)
    return out


def lium_get_or_create_template(payload: dict, *, api_key: str,
                                base_url: str = LIUM_API_BASE, timeout: float = 30.0) -> str:
    """Resolve ``payload`` to a template id on the account — reuse one whose
    pinned contents match exactly, else create once. The account belongs to
    the payer, who can pre-create a template under our (derivable) name with
    their own startup commands, env or SSH key: a name match alone is never
    reused, and a squatted name makes us create under a nonce-suffixed name.
    A 429 on creation is surfaced as a :class:`ProvisionError` naming the
    retry-after, so the caller's infra path can requeue instead of burning a
    leg (and so the operator can SEE the limit)."""
    import secrets

    import requests

    if not api_key:
        raise ProvisionError("lium: no API key for the template lookup (LIUM_API_KEY unset)")
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    name = payload["name"]
    try:
        r = requests.get(f"{base_url}/templates", headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise ProvisionError(f"lium: template listing failed: {e}") from e
    taken = False
    if r.ok:
        body = r.json()
        items = body if isinstance(body, list) else (body.get("items") or [])
        for t in items:
            tname = str(t.get("name") or "")
            if not t.get("id") or not (tname == name or tname.startswith(name + "-")):
                continue
            taken = taken or tname == name
            if any(k not in t for k in _TEMPLATE_PINNED):
                t = _template_detail(t, base_url=base_url, headers=headers, timeout=timeout)
            diff = _template_diff(t, payload)
            if not diff:
                return str(t["id"])
            log.warning("lium: template %s (%s) is named like ours but differs in %s; "
                        "not reusing it", t.get("id"), tname, ",".join(diff))
    body = dict(payload)
    if taken:
        body["name"] = f"{name}-{secrets.token_hex(4)}"
    try:
        r = requests.post(f"{base_url}/templates", headers=headers, json=body,
                          timeout=timeout)
    except requests.RequestException as e:
        raise ProvisionError(f"lium: template creation failed: {e}") from e
    if r.status_code == 429:
        raise ProvisionError(
            "lium: template creation rate-limited (20 requests/hour per client IP); "
            f"retry after {r.headers.get('retry-after', '?')}s — {r.text[:160]}")
    if not r.ok:
        raise ProvisionError(f"lium: template creation failed (HTTP {r.status_code}): "
                             f"{r.text[:200]}")
    tid = str(r.json().get("id") or "")
    if not tid:
        raise ProvisionError(f"lium: template creation returned no id: {r.text[:200]}")
    log.info("lium: created persistent template %s as %s", tid, body["name"])
    return tid


def _template_detail(t: dict, *, base_url: str, headers: dict, timeout: float) -> dict:
    """``GET /templates/{id}`` for a listing row that omits pinned fields; the
    row itself on any failure (its missing fields then fail the compare)."""
    import requests

    try:
        r = requests.get(f"{base_url}/templates/{t['id']}", headers=headers, timeout=timeout)
    except requests.RequestException:
        return t
    if not r.ok:
        return t
    d = r.json()
    return {**t, **d} if isinstance(d, dict) else t


@dataclass(frozen=True)
class PodAddress:
    """Where the orchestrator SSHes to reach a launched pod."""

    ip: str
    ssh_port: int = DEFAULT_SSH_PORT


# ── provider protocol ────────────────────────────────────────────────────────


class Provider(Protocol):
    """A GPU marketplace adapter. The four verbs mirror the pod lifecycle.

    ``available`` is a non-destructive capacity probe used for priority
    selection (and for ``--dry-run``); ``launch``/``wait_ready``/``get_ip``/
    ``terminate`` drive one pod through its life. ``terminate`` MUST be
    idempotent — calling it on an already-gone pod is a no-op, never an error.
    """

    name: str

    def available(self, sku: str, count: int, *, gpus: int = 1) -> bool: ...
    def launch(self, spec: LaunchSpec) -> list[str]: ...        # → pod handles
    def wait_ready(self, pod_id: str, *, timeout: float) -> bool: ...
    def get_ip(self, pod_id: str) -> PodAddress | None: ...
    def terminate(self, pod_id: str) -> None: ...


# ── pure helpers (unit-tested) ───────────────────────────────────────────────


def validate_digest_pinned(image: str) -> None:
    """Reject a worker image that is not pinned by ``@sha256:`` digest.

    A moving tag would silently change the code/GPU a round trained on and break
    the ``expected_gpu`` reproducibility contract.
    """
    if "@sha256:" not in image:
        raise ProvisionError(
            f"worker image must be digest-pinned (…@sha256:<64hex>), got {image!r}. "
            "Pin by digest, not tag — expected_gpu re-derivation depends on it."
        )


def select_provider(
    providers: Sequence[Provider], sku: str, count: int
) -> Provider | None:
    """First provider (in the given order) with capacity for ``count`` × ``sku``.

    Returns ``None`` if none have capacity — the caller then exits non-zero
    rather than substituting a different SKU or splitting across providers.
    """
    for p in providers:
        try:
            ok = p.available(sku, count)
        except ProvisionError:
            raise
        except Exception as e:  # noqa: BLE001 — never let one adapter's fault mask the rest
            log.warning("provider %s availability probe failed: %s", p.name, e)
            continue
        log.info("provider %s: %s for %d×%s", p.name,
                 "AVAILABLE" if ok else "no capacity", count, sku)
        if ok:
            return p
    return None


def render_hosts_toml(
    addrs: Sequence[PodAddress],
    *,
    key_path: str,
    forward_env: Sequence[str],
    remote_python: str = DEFAULT_REMOTE_PYTHON,
    workdir: str = DEFAULT_WORKDIR,
    user: str = "root",
    chain_toml: str | None = None,
    ssh_options: Sequence[str] = DEFAULT_SSH_OPTIONS,
    name_prefix: str = "cascade-pod",
    provider: str = "",
    stage: str = "any",
    gpus_per_pod: int = 1,
    pod_names: Sequence[str] | None = None,
) -> str:
    """Render a trainer ``hosts.toml`` (schema: ``remote_hosts.example.toml``).

    One ``[[host]]`` per GPU: a single-GPU pod (``gpus_per_pod=1``, the default)
    is one entry named ``{prefix}-{i}`` with ``cuda_device = "0"``; a multi-GPU
    pod fans out into ``gpus_per_pod`` entries named ``{prefix}-{i}-g{g}`` with
    ``cuda_device`` ``"0"``…``"N-1"`` — same address, one training slot per GPU
    (``RemoteHost.cuda_device`` pins ``CUDA_VISIBLE_DEVICES`` per dispatch, so
    the trainer's round-robin lands one job per GPU). ``forward_env`` lists the
    credential env vars the orchestrator forwards inline per dispatch (they are
    NOT seeded onto the pod). The array literals are emitted as TOML/JSON-style
    ``["a", "b"]``.

    ``stage`` ("any" | "final"; "heat" is still accepted, unused) tags the
    lane class (see ``cascade.trainer.remote.RemoteHost``): the trainer's
    rolling fallback legs take ``final``/``any`` lanes. ``"any"`` is omitted (it
    is the schema default).

    ``pod_names`` (one per address) overrides the positional ``{prefix}-{i}``
    stem with a STABLE per-pod name: entries become ``{pod_names[i]}-g{g}``
    (always ``-g``-suffixed, even for one GPU). The trainer's final lane pool
    is keyed by name and only ever grows, so a lane that changes name across
    two publishes is a NEW lane to it — and the same GPU gets dispatched twice
    (2026-09-06 09:05: the second pod's publish renamed the first pod's lanes
    from the old round id to the new one and the king's GPU took a second
    leg). Callers that publish repeatedly must pass names derived from pod
    identity, never from the round or the position in the list.
    """
    if not addrs:
        raise ProvisionError("cannot render hosts.toml with zero pods")
    if pod_names is not None and len(pod_names) != len(addrs):
        raise ProvisionError(f"pod_names has {len(pod_names)} entries for {len(addrs)} pods")
    if gpus_per_pod < 1:
        raise ProvisionError(f"gpus_per_pod must be >= 1; got {gpus_per_pod}")

    def _arr(items: Sequence[str]) -> str:
        return json.dumps(list(items))

    lines = [
        "# Generated by cascade.provision — trainer-local remote GPU pods"
        + (f" ({provider})" if provider else "") + ".",
        "# Ephemeral: torn down by the provisioner. Keep OFF git and chain.toml.",
        "# Hippius creds are piped over stdin per dispatch (forward_env), never on the pod.",
    ]
    for i, addr in enumerate(addrs):
        for g in range(gpus_per_pod):
            if pod_names is not None:
                name = f"{pod_names[i]}-g{g}"
            else:
                name = f"{name_prefix}-{i}" if gpus_per_pod == 1 else f"{name_prefix}-{i}-g{g}"
            lines += [
                "",
                "[[host]]",
                f'name          = "{name}"',
                f'host          = "{addr.ip}"',
                f"port          = {addr.ssh_port}",
                f'user          = "{user}"',
                f'key_path      = "{key_path}"',
                f'remote_python = "{remote_python}"',
                f'workdir       = "{workdir}"',
                f'cuda_device   = "{g}"',
            ]
            if stage != "any":
                lines.append(f'stage         = "{stage}"')
            if chain_toml:
                lines.append(f'chain_toml    = "{chain_toml}"')
            lines += [
                f"forward_env   = {_arr(forward_env)}",
                f"ssh_options   = {_arr(ssh_options)}",
                # a lane that forwards nothing is harvested (PR #320); a lane that
                # forwards something is a deliberate exception and stays attached
                f"isolated      = {'true' if not forward_env else 'false'}",
            ]
    return "\n".join(lines) + "\n"


def parse_ssh_port(ssh_cmd: str, default: int = DEFAULT_SSH_PORT) -> int:
    """Pull the port out of a Lium ``ssh_cmd`` string (``ssh root@ip -p 22000``)."""
    m = re.search(r"-p\s+(\d+)", ssh_cmd or "")
    return int(m.group(1)) if m else default


def parse_ssh_host(ssh_cmd: str) -> str | None:
    """Pull the host out of a Lium ``ssh_cmd`` string (``…@1.2.3.4 …``)."""
    m = re.search(r"@([^\s:]+)", ssh_cmd or "")
    return m.group(1) if m else None


def parse_lium_executors(stdout: str) -> list[dict]:
    """Parse ``lium ls --format json`` → list of executor dicts (``[]`` if none).

    Empty list = no capacity for the filtered SKU (Lium exits 0 either way), the
    signal the wrapper uses to fall through to the next provider.
    """
    text = (stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Carry the CLI's actual words: on a bad key `lium` prints a plain-text
        # "Invalid API key" instead of JSON, and the funded path's fault
        # taxonomy (cascade.funding.faults) can only classify that as "auth"
        # if the text survives into the error (observed live 2026-09-02 —
        # the bare parse error classified a revoked key as "infra").
        raise ProvisionError(
            f"could not parse `lium ls` JSON ({e}); output was: "
            f"{text[:200]!r}") from e
    if not isinstance(data, list):
        raise ProvisionError(f"expected a JSON array from `lium ls`, got {type(data).__name__}")
    return data


def parse_lium_pods(stdout: str) -> list[dict]:
    """Parse ``lium ps --format json`` → list of pod dicts (``[]`` if none)."""
    text = (stdout or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Carry the CLI's actual words: on a bad key `lium` prints a plain-text
        # "Invalid API key" instead of JSON, and the funded path's fault
        # taxonomy (cascade.funding.faults) can only classify that as "auth"
        # if the text survives into the error (observed live 2026-09-02 —
        # the bare parse error classified a revoked key as "infra").
        raise ProvisionError(
            f"could not parse `lium ps` JSON ({e}); output was: "
            f"{text[:200]!r}") from e
    if not isinstance(data, list):
        raise ProvisionError(f"expected a JSON array from `lium ps`, got {type(data).__name__}")
    return data


def cpu_model_blocked(model: str, blocklist: tuple[str, ...] | list[str]) -> bool:
    """True when ``model`` contains any blocklist entry (case-insensitive)."""
    m = (model or "").lower()
    return any(b and b.lower() in m for b in blocklist)


def lium_pod_ready(pod: dict) -> bool:
    """A Lium pod is ready when it is RUNNING and exposes an SSH endpoint."""
    return str(pod.get("status", "")).upper() == "RUNNING" and bool(pod.get("ssh_cmd"))


def lium_pod_address(pod: dict, *, container_ssh_port: int = DEFAULT_SSH_PORT) -> PodAddress | None:
    """Extract the reachable ``(ip, port)`` from a Lium ``ps`` pod dict.

    Lium remaps the container's port 22 to a dynamic external port, so the port
    comes from ``ssh_cmd`` (authoritative) or the ``ports`` map, not a fixed 22.
    """
    ssh_cmd = str(pod.get("ssh_cmd", ""))
    ip = pod.get("ip") or parse_ssh_host(ssh_cmd)
    if not ip:
        return None
    port = parse_ssh_port(ssh_cmd, default=0)
    if not port:
        ports = pod.get("ports") or {}
        # ports maps internal→external (values may be int or str)
        ext = ports.get(str(container_ssh_port)) or ports.get(container_ssh_port)
        port = int(ext) if ext else DEFAULT_SSH_PORT
    return PodAddress(ip=str(ip), ssh_port=int(port))


def pick_shadeform_offer(types_json: dict, sku: str, *, gpus: int = 1,
                         exclude_regions: tuple[str, ...] = ()) -> dict | None:
    """Choose a ``(cloud, region, shade_instance_type)`` offer for ``sku``.

    Filters ``GET /instances/types`` results to the requested ``gpu_type`` AND
    the exact ``gpus`` pod shape (``configuration.num_gpus``) with an available
    region outside ``exclude_regions``, preferring the cheapest
    (``hourly_price``, in cents). Returns ``None`` if nothing is available —
    the fall-through signal.
    """
    offers: list[tuple[int, dict]] = []
    for t in types_json.get("instance_types", []):
        if str(t.get("configuration", {}).get("gpu_type", "")).upper() != sku.upper():
            continue
        if int(t.get("configuration", {}).get("num_gpus", 1) or 1) != int(gpus):
            continue
        region = next(
            (a.get("region") for a in t.get("availability", [])
             if a.get("available") and a.get("rental_type", "on_demand") == "on_demand"
             and a.get("region") not in exclude_regions),
            None,
        )
        if not region:
            continue
        offers.append((
            int(t.get("hourly_price", 1 << 30)),
            {
                "cloud": t.get("cloud"),
                "region": region,
                "shade_instance_type": t.get("shade_instance_type"),
            },
        ))
    if not offers:
        return None
    offers.sort(key=lambda x: x[0])
    return offers[0][1]


def pick_shadeform_cpu_offer(types_json: dict, min_vcpus: int,
                             max_price_usd_hr: float) -> list[dict]:
    """CPU-generation offers (DEC-CA-0053) from ``GET /instances/types``,
    cheapest first.

    The endpoint is documented as GPU instance types, but every type reports
    ``configuration.vcpus`` / ``memory_in_gb``, so a vCPU-rich cheap shape
    serves as a CPU VM. Keeps on-demand available entries with ``vcpus >=
    min_vcpus`` and ``hourly_price`` (CENTS) within ``max_price_usd_hr``. A
    GPU-less type (``num_gpus == 0``), when listed at all, sorts ahead of any
    GPU type at the same or higher price — we rent the cores, not the card.
    Each offer: ``{cloud, region, shade_instance_type, vcpus, memory_gb,
    price_usd_hr, num_gpus}``."""
    out: list[tuple[int, float, dict]] = []
    for t in types_json.get("instance_types", []):
        conf = t.get("configuration", {}) or {}
        vcpus = int(conf.get("vcpus", 0) or 0)
        price = int(t.get("hourly_price", 1 << 30)) / 100.0
        if vcpus < int(min_vcpus) or price > float(max_price_usd_hr):
            continue
        region = next(
            (a.get("region") for a in t.get("availability", [])
             if a.get("available") and a.get("rental_type", "on_demand") == "on_demand"),
            None,
        )
        if not region:
            continue
        gpus = int(conf.get("num_gpus", 0) or 0)
        out.append((0 if gpus == 0 else 1, price, {
            "cloud": t.get("cloud"),
            "region": region,
            "shade_instance_type": t.get("shade_instance_type"),
            "vcpus": vcpus,
            "memory_gb": float(conf.get("memory_in_gb", 0) or 0),
            "price_usd_hr": price,
            "num_gpus": gpus,
        }))
    out.sort(key=lambda x: (x[0], x[1]))
    return [o for _, _, o in out]


def shadeform_offer_price_usd_hr(types_json: dict, sku: str) -> float | None:
    """Cheapest available hourly price for ``sku`` in USD/hr (``None`` if none).

    The API reports ``hourly_price`` in CENTS; the provisioner's budget breaker
    (``policy.within_budget``) works in USD, so convert here — a silent
    cents-as-dollars mixup would either 100× the projection (refusing every
    round) or 1/100 it (defeating the breaker).
    """
    prices = [
        int(t.get("hourly_price", 1 << 30))
        for t in types_json.get("instance_types", [])
        if str(t.get("configuration", {}).get("gpu_type", "")).upper() == sku.upper()
        and any(a.get("available") for a in t.get("availability", []))
    ]
    return min(prices) / 100.0 if prices else None


def filter_tagged_names(pods: Sequence[dict], prefix: str, *, id_key: str = "name") -> list[str]:
    """The ``id_key`` of every pod whose ``name`` starts with ``prefix``.

    The orphan-reconcile primitive shared by both adapters: a provider listing
    is reduced to just the handles that carry OUR tag, so a shared marketplace
    account's unrelated pods are never candidates for termination.
    """
    out = []
    for p in pods:
        if str(p.get("name", "")).startswith(prefix):
            handle = p.get(id_key) or p.get("name")
            if handle:
                out.append(str(handle))
    return out


def shadeform_create_body(
    spec: LaunchSpec, offer: dict, *, name: str, ssh_key_id: str | None = None
) -> dict:
    """Build the ``POST /instances/create`` body for one pod of ``spec``.

    Two boot modes:

    * **docker** (default) — the pod runs ``spec.image`` (the digest-pinned
      worker image, whose entrypoint starts sshd and reads ``SSH_PUBKEY``).
    * **VM** (``ssh_key_id`` given) — no container: the bare VM boots with the
      account's registered SSH key and the provisioner's bootstrap_script
      provisions it over SSH (user ``shadeform``). This is the testnet path
      while no worker image is published — ``spec.image`` is ignored here.

    In neither mode do Hippius creds touch the pod.
    """
    body = {
        "cloud": offer["cloud"],
        "region": offer["region"],
        "shade_instance_type": offer["shade_instance_type"],
        "shade_cloud": True,
        "name": name,
    }
    if ssh_key_id:
        body["ssh_key_id"] = ssh_key_id
        return body
    body["launch_configuration"] = {
        "type": "docker",
        "docker_configuration": {
            "image": spec.image,
            "envs": [{"name": "SSH_PUBKEY", "value": spec.ssh_pubkey}]
                    + ([{"name": "CASCADE_TRAIN_IMAGE_DIGEST",
                         "value": image_digest_of(spec.image)}]
                       if image_digest_of(spec.image) else []),
            "port_mappings": [
                {"host_port": spec.ssh_port, "container_port": DEFAULT_SSH_PORT}
            ],
        },
    }
    return body


def shadeform_pod_address(info_json: dict, *, ssh_port: int = DEFAULT_SSH_PORT) -> PodAddress | None:
    """Extract ``(ip, port)`` from a Shadeform ``/instances/{id}/info`` response.

    Docker-mode: the container's sshd lives at the ``host_port`` of the
    port mapping echoed back in ``launch_configuration`` — NOT at the VM's
    port 22 (containers default to ``--network=host``, and the VM's own sshd
    owns 22; live 2026-07-15). Prefer the echoed mapping; fall back to the
    caller's ``ssh_port`` (VM-mode rentals carry no docker configuration).
    """
    ip = info_json.get("ip")
    if not ip:
        return None
    docker = (info_json.get("launch_configuration") or {}).get("docker_configuration") or {}
    for pm in docker.get("port_mappings") or []:
        if int(pm.get("container_port", 0) or 0) == DEFAULT_SSH_PORT:
            return PodAddress(ip=str(ip), ssh_port=int(pm["host_port"]))
    return PodAddress(ip=str(ip), ssh_port=ssh_port)


SHADEFORM_READY = "active"                  # the "running/live" status (no "running" state)
SHADEFORM_TERMINAL_BAD = {"error", "deleting", "deleted"}
# Docker-mode: the INSTANCE goes "active" while the container is still pulling
# ("downloading" → "running"). Probing at "active" reaches the VM's own sshd —
# which knows nothing of SSH_PUBKEY — and reads as a dead pod (2026-07-15:
# every image-boot rental failed its gate exactly this way).
SHADEFORM_CONTAINER_READY = "running"
SHADEFORM_CONTAINER_BAD = {"failed", "error", "stopped"}


# ── side-effecting helpers (adapter surface, not unit-tested) ─────────────────


def _run_cli(argv: list[str], timeout: float = 120.0,
             env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a CLI command and capture output (text mode).

    ``env`` (full environment, not a delta) overrides the child's environment —
    the per-payer rental path uses it to point the ``lium`` CLI at a miner's
    API key without ever mutating this process's environ.
    """
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)


def _default_lium_bin() -> str:
    """Resolve the ``lium`` binary: alongside the running Python (venv), then PATH.

    Running under ``.venv/bin/python`` does NOT put ``.venv/bin`` on ``$PATH``, so a
    bare ``lium`` lookup misses a venv-installed CLI. Prefer the sibling of the
    current interpreter.
    """
    sibling = Path(sys.executable).with_name("lium")
    if sibling.exists():
        return str(sibling)
    return shutil.which("lium") or "lium"


def _spawn_cli(argv: list[str], log_path: Path | None = None,
               env: dict[str, str] | None = None) -> subprocess.Popen:
    """Fire-and-forget a CLI command (e.g. ``lium up``, which attaches/streams).

    We don't wait on it — pod readiness is observed via ``lium ps`` polling, and
    killing the local client does not stop the remote pod. ``log_path`` captures
    the child's output: ``lium up`` DRIVES the deploy client-side, so its output
    is the only evidence when a pod silently never materializes (observed live:
    an executor in post-teardown cooldown accepts the up, deploys nothing, and
    a DEVNULL'd spawn leaves a 900s ghost hunt with zero diagnostics).
    """
    # Unbuffered child stdout: the CLI is Python and block-buffers when its
    # stdout is a file, so the up-log stayed EMPTY through every failure on
    # 2026-09-12 ("Rate limit exceeded" never reached disk).
    env = {**(env if env is not None else os.environ), "PYTHONUNBUFFERED": "1"}
    if log_path is None:
        return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                env=env)
    # Fresh log per launch: pod names are deterministic per (round, hotkey), so
    # an appended log showed a PREVIOUS rent's shell banner as this rent's
    # "output" in the not-ready error (2026-09-12 15:11).
    with open(log_path, "wb") as f:
        return subprocess.Popen(argv, stdout=f, stderr=f, env=env)


def wait_ssh_reachable(ip: str, port: int, *, timeout: float, interval: float = 5.0,
                       sleep: Callable[[float], None] = time.sleep,
                       now: Callable[[], float] = time.monotonic) -> bool:
    """Poll a TCP connect to ``ip:port`` until sshd answers or ``timeout`` elapses."""
    deadline = now() + timeout
    while now() < deadline:
        try:
            with socket.create_connection((ip, port), timeout=5.0):
                return True
        except OSError:
            sleep(interval)
    return False


# ── Lium adapter (CLI shell-out, mirrors remote.py shelling to ssh) ───────────


@dataclass
class LiumProvider:
    """Lium marketplace via its ``lium`` CLI (``--format json`` for machine output).

    Auth is the CLI's own (``LIUM_API_KEY`` / ``~/.lium/config.ini``) — unless
    ``api_key`` is set, in which case EVERY CLI call this instance makes runs
    with that key in its child environment instead (the miner-funded rental
    path, DEC-CA-0036: one provider instance per payer, mirroring PRISM's
    per-submission ``LiumClient``). The key never enters this process's own
    environ, never appears in argv (world-readable in ``/proc``), and the
    dataclass repr excludes it.

    Each pod is one ``lium up`` on a distinct executor; SSH is injected via the
    container's ``SSH_PUBKEY`` env (the CLI has no direct pubkey flag). Pods are
    addressed by the ``--name`` we assign (Lium's ``ps``/``rm`` accept name,
    huid, or id).
    """

    name: str = "lium"
    bin: str = field(default_factory=_default_lium_bin)
    api_key: str = field(default="", repr=False)
    poll_interval: float = POD_POLL_INTERVAL
    _run: Callable[[list[str]], subprocess.CompletedProcess] | None = field(default=None, repr=False)
    _spawn: Callable[[list[str]], object] | None = field(default=None, repr=False)
    # Template resolver seam: ``payload -> template id``. None ⇒ the account's
    # persistent template via lium_get_or_create_template (one per image).
    _template: Callable[[dict], str] | None = field(default=None, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    # pod name → executor id for pods this instance launched, so the loop can
    # exclude a failed pod's machine when renting its replacement.
    _executor_by_name: dict = field(default_factory=dict, repr=False)
    _executor_hosts_cache: tuple = field(default=(), repr=False)
    # CPU-model substrings (case-insensitive) whose executors are never listed
    # ([round] funded_cpu_blocklist); () = no filter.
    cpu_blocklist: tuple[str, ...] = ()
    _executor_cpus_cache: tuple = field(default=(), repr=False)
    # Price guards ([round] funded_max_price_per_hour / funded_max_leg_cost_usd,
    # USD; 0 = off) and the per-SKU measured leg wall behind the per-leg cap
    # (funded_sku_wall_seconds; default_wall_seconds = the contract's
    # max_train_seconds for SKUs without an entry). Listings are sorted by
    # per-leg cost, cheapest first, so ``launch`` picks the best-value machine.
    max_price_per_hour: float = 0.0
    max_leg_cost_usd: float = 0.0
    sku_wall_seconds: tuple[tuple[str, int], ...] = ()
    default_wall_seconds: float = 0.0
    # pod name → SKU it was launched on (sku_choices launches).
    _sku_by_name: dict = field(default_factory=dict, repr=False)
    # pod name → monotonic time its `lium up` was issued (``create_requested``
    # and the ghost settlement in ``terminate``).
    _issued_creates: dict = field(default_factory=dict, repr=False)

    def _subprocess_env(self) -> dict[str, str] | None:
        """Child env for CLI calls: the payer's key layered over ours, or None.

        ``None`` (no per-payer key) inherits the process environment untouched
        — the operator-account path, bit-identical to pre-DEC-CA-0036
        behaviour. With a key set we copy the environment and override
        ``LIUM_API_KEY`` so the CLI bills the payer; the copy also guards
        against the CLI falling back to ``~/.lium/config.ini`` *and* against
        us ever exporting the key to our own environ.
        """
        if not self.api_key:
            return None
        return {**os.environ, "LIUM_API_KEY": self.api_key}

    def _cli(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            runner = self._run or (
                lambda argv: _run_cli(argv, env=self._subprocess_env()))
            proc = runner([self.bin, *args])
        except FileNotFoundError as e:
            # A missing binary is a config error, not "no capacity" — surface it
            # loudly instead of silently falling through to another provider.
            raise ProvisionError(
                f"lium CLI not found at {self.bin!r}; install it (pip install lium.io)"
            ) from e
        if proc.returncode != 0:
            detail = (proc.stderr or "")[-500:]
            if self.api_key:
                # A CLI must never echo its key, but the one place it could
                # land is stderr — scrub before it can reach a log line.
                detail = detail.replace(self.api_key, "<redacted>")
            raise ProvisionError(
                f"`{self.bin} {' '.join(args)}` failed (rc={proc.returncode}): {detail}"
            )
        return proc

    def _scrub_key(self, e: ProvisionError) -> ProvisionError:
        """Re-raiseable copy of ``e`` with the payer key redacted.

        The JSON-parse errors quote raw CLI STDOUT (``output was: …``) —
        stderr is scrubbed in :meth:`_cli`, but a CLI that ever echoed the key
        on stdout would reach log lines through the parse path unscrubbed
        (review 2026-09-02). ``from None`` on the re-raise side: chaining the
        unscrubbed original would put the key one traceback away.
        """
        if self.api_key and self.api_key in str(e):
            return ProvisionError(str(e).replace(self.api_key, "<redacted>"))
        return e

    def _list_executors(self, sku: str, *, gpus: int = 1) -> list[dict]:
        """Marketplace executors of ``sku`` with EXACTLY ``gpus`` GPUs.

        Shape matters: the fleet plan fans one hosts.toml lane out per GPU, so
        a 1× machine rented against an 8-lane plan strands seven lanes (and the
        health gate then kills the pod anyway — filter here, before renting)."""
        from .layout import split_layout

        base, count = split_layout(sku)
        if count > 1:
            gpus = count            # a layout SKU ("8xH200") fixes its own shape
        try:
            execs = parse_lium_executors(
                self._cli(["ls", "--gpu", base, "--format", "json"]).stdout)
        except ProvisionError as e:
            raise self._scrub_key(e) from None
        shaped = [e for e in execs if int(e.get("gpu_count", 1) or 1) == int(gpus)]
        cooling = cooling_executors()
        if cooling:
            kept = [e for e in shaped if str(e.get("id")) not in cooling]
            if len(kept) != len(shaped):
                log.info("lium: %d %s executor(s) skipped — vacated < %.0f min ago "
                         "(post-teardown cooldown deploys nothing)",
                         len(shaped) - len(kept), sku, LIUM_EXECUTOR_COOLDOWN_SECONDS / 60)
            shaped = kept
        quarantined = quarantined_hosts()
        if quarantined and shaped:
            hosts = self._executor_hosts()
            kept = [e for e in shaped if hosts.get(str(e.get("id")), "") not in quarantined]
            if len(kept) != len(shaped):
                log.info("lium: %d %s executor(s) skipped — on a quarantined host (%s)",
                         len(shaped) - len(kept), sku, ", ".join(sorted(
                             {hosts[str(e["id"])] for e in shaped if e not in kept})))
            shaped = kept
        if self.cpu_blocklist and shaped:
            cpus = self._executor_cpus()
            kept = [e for e in shaped
                    if not cpu_model_blocked(cpus.get(str(e.get("id")), ""), self.cpu_blocklist)]
            if len(kept) != len(shaped):
                log.info("lium: %d %s executor(s) skipped — CPU model on the blocklist (%s)",
                         len(shaped) - len(kept), sku, ", ".join(sorted(
                             {cpus[str(e["id"])] for e in shaped if e not in kept})))
            shaped = kept
        return self._price_filter(sku, shaped)

    def wall_seconds_for(self, sku: str) -> float:
        """Measured full-budget leg wall for ``sku`` (``sku_wall_seconds``),
        else ``default_wall_seconds``; 0 when neither is known."""
        want = (sku or "").strip().lower()
        for name, secs in self.sku_wall_seconds:
            if name.lower() == want:
                return float(secs)
        return float(self.default_wall_seconds or 0.0)

    def leg_cost_usd(self, sku: str, executor: dict) -> float | None:
        """price_per_hour × the SKU's wall — what one leg would bill; None
        when the listing carries no price or no wall is known."""
        price = executor_price_per_hour(executor)
        wall = self.wall_seconds_for(sku)
        if price is None or wall <= 0:
            return None
        return price * wall / 3600.0

    def _price_filter(self, sku: str, execs: list[dict]) -> list[dict]:
        """Drop executors over the $/h ceiling or the per-leg cost cap (an
        UNPRICED listing is kept only when no cap is armed — a cap must never
        be dodged by a missing field), then order cheapest-per-leg first."""
        cap_h = float(self.max_price_per_hour or 0.0)
        cap_leg = float(self.max_leg_cost_usd or 0.0)
        if not cap_h and not cap_leg:
            return execs
        kept, dropped = [], []
        for e in execs:
            price = executor_price_per_hour(e)
            if price is None:
                dropped.append((e, "unpriced"))
                continue
            if cap_h and price > cap_h:
                dropped.append((e, f"${price:.2f}/h > ${cap_h:.2f}/h"))
                continue
            if cap_leg:
                cost = self.leg_cost_usd(sku, e)
                if cost is not None and cost > cap_leg:
                    dropped.append((e, f"${cost:.2f}/leg > ${cap_leg:.2f}/leg"))
                    continue
            kept.append(e)
        if dropped:
            log.info("lium: %d %s executor(s) skipped by the price caps (%s)",
                     len(dropped), sku, "; ".join(
                         f"{str(e.get('id'))[:8]} {why}" for e, why in dropped[:6]))
        kept.sort(key=lambda e: (self.leg_cost_usd(sku, e)
                                 if self.leg_cost_usd(sku, e) is not None
                                 else executor_price_per_hour(e) or 0.0,
                                 executor_price_per_hour(e) or 0.0))
        return kept

    def list_offers(self, skus: Sequence[str], *, gpus: int = 1,
                    exclude_ids: tuple[str, ...] = ()) -> list[tuple[str, dict]]:
        """``(sku, executor)`` pairs across ``skus`` (each listing filtered
        exactly like :meth:`_list_executors`), ordered cheapest-per-leg first
        (unpriced/unknown-wall offers last, in ``skus`` preference order).
        The open-market rent picks the first."""
        offers: list[tuple[int, float, str, dict]] = []
        for pref, sku in enumerate(skus):
            for e in self._list_executors(sku, gpus=gpus):
                if exclude_ids and str(e.get("id")) in exclude_ids:
                    continue
                cost = self.leg_cost_usd(sku, e)
                offers.append((0 if cost is not None else 1, cost if cost is not None
                               else float(pref), sku, e))
        offers.sort(key=lambda o: (o[0], o[1], skus.index(o[2])))
        return [(sku, e) for _, _, sku, e in offers]

    def sku_of(self, pod_id: str) -> str:
        """GPU type a pod launched through ``sku_choices`` landed on (this
        process's launches only); "" when unknown."""
        return self._sku_by_name.get(pod_id, "")

    def _list_pods(self) -> list[dict]:
        try:
            return parse_lium_pods(self._cli(["ps", "--format", "json"]).stdout)
        except ProvisionError as e:
            raise self._scrub_key(e) from None

    def _pod(self, pod_id: str) -> dict | None:
        # Names are owner-chosen and reusable: a rent retried under the same
        # name while the earlier pod still runs lists TWO pods by that name
        # (2026-09-24: seven such pairs after a trainer restart). The pod that
        # matters is the one answering — RUNNING with an SSH endpoint — so
        # identity, address and adoption all resolve to it, never to the
        # booting duplicate.
        hits = [p for p in self._list_pods()
                if pod_id in (p.get("name"), p.get("huid"), p.get("id"))]
        if not hits:
            return None
        self.__dict__.setdefault("_unresolved_creates", {})[pod_id] = False
        return next((p for p in hits if lium_pod_ready(p)), hits[0])

    def available(self, sku: str, count: int, *, gpus: int = 1,
                  exclude_ids: tuple[str, ...] = ()) -> bool:
        return self.capacity(sku, gpus=gpus, exclude_ids=exclude_ids) >= count

    def capacity(self, sku: str, *, gpus: int = 1,
                 exclude_ids: tuple[str, ...] = ()) -> int:
        """How many ``sku`` machines the marketplace offers RIGHT NOW, minus
        ``exclude_ids`` (executors this round already claimed — a probe that
        counts them says "1" while ``launch`` says "0 after exclusions", and
        the king rent loop spun on exactly that at ~1 attempt/1.5 s for 136
        attempts on 2026-09-13 02:26).

        A snapshot, not a reservation — other renters race it, so callers
        must treat the number as advisory and keep a requeue path for rents
        that lose the race (the funded taxonomy's ``no_capacity``)."""
        execs = self._list_executors(sku, gpus=gpus)
        if exclude_ids:
            execs = [e for e in execs if str(e.get("id")) not in exclude_ids]
        return len(execs)

    def launch(self, spec: LaunchSpec) -> list[str]:
        if spec.sku_choices:
            offers = self.list_offers(spec.sku_choices, gpus=spec.gpus_per_pod,
                                      exclude_ids=spec.exclude_ids)
            execs = [e for _, e in offers]
            sku_for = {str(e.get("id")): sku for sku, e in offers}
        else:
            execs = self._list_executors(spec.sku, gpus=spec.gpus_per_pod)
            if spec.exclude_ids:
                execs = [e for e in execs if str(e.get("id")) not in spec.exclude_ids]
            sku_for = {}
        if len(execs) < spec.count:
            want = "/".join(spec.sku_choices) if spec.sku_choices else spec.sku
            raise ProvisionError(
                f"lium: only {len(execs)} × {spec.gpus_per_pod}x{want} available"
                f"{' after exclusions' if spec.exclude_ids else ''}, need {spec.count}"
            )
        # Lium cannot parse a digest-pinned ref (400s on repo@sha256:… — see
        # lium_image_ref): degrade to the tag/repo form for the launch call
        # only. The env keeps the FULL pin and the health gate enforces it.
        image_ref = lium_image_ref(spec.image) if spec.image else ""
        if spec.image and image_ref != spec.image:
            log.warning(
                "lium: degrading image ref %r -> %r for launch (lium's API "
                "400-rejects digest-pinned refs); the digest pin is NOT "
                "weakened — CASCADE_TRAIN_IMAGE_DIGEST still carries %s and "
                "the health gate byte-compares it before the pod serves",
                spec.image, image_ref, image_digest_of(spec.image))
        # ONE persistent template per image for the whole batch (and every
        # later batch on this account): `lium up --image` would mint a
        # throw-away template per pod against Lium's 20/hour/IP cap.
        template_id = self._template_id_for(spec) if spec.image else ""
        names: list[str] = []
        for i, ex in enumerate(execs[: spec.count]):
            name = f"{spec.name_prefix}-{i}"
            argv = [self.bin, "up", str(ex["id"])]
            if spec.image:
                # docker-run style through the account's persistent template:
                # the worker image's entrypoint runs sshd and reads $SSH_PUBKEY;
                # the template carries that env + CASCADE_TRAIN_IMAGE_DIGEST
                # (see lium_template_payload).
                argv += ["--template_id", template_id]
            # empty image ⇒ lium's default SSH template (bootstrap mode): lium
            # injects the ACCOUNT's registered keys; a template NAME passed as
            # --image 400s ("image reference is not valid") — never do that.
            argv += ["--name", name, "--yes"]
            ttl = lium_ttl_arg(getattr(spec, "ttl_seconds", 0.0))
            if ttl:
                # Provider-side backstop: the pod terminates itself even if
                # nothing here is alive to `lium rm` it.
                argv += ["--ttl", ttl]
            self.__dict__.setdefault("_unresolved_creates", {})[name] = True
            # Issued is as certain as a fire-and-forget create gets: from here
            # the intent must outlive any failure (``create_requested``).
            self._issued_creates[name] = self._now()
            try:
                if self._spawn is not None:
                    self._spawn(argv)
                else:
                    _spawn_cli(argv, log_path=self._up_log_path(name),
                               env=self._subprocess_env())
            except Exception as exc:  # noqa: BLE001 — a spawn that may have run is uncertain
                raise UncertainCreateError(f"lium up {name}: {exc}") from None
            landed = sku_for.get(str(ex.get("id")), spec.sku)
            log.info("lium up → executor %s as %s (%s%s)", ex.get("id"), name, landed,
                     f", ${executor_price_per_hour(ex):.2f}/h"
                     if executor_price_per_hour(ex) is not None else "")
            self._executor_by_name[name] = str(ex.get("id"))
            self._sku_by_name[name] = landed
            names.append(name)
        return names

    def _template_id_for(self, spec: LaunchSpec) -> str:
        """The account's persistent template for ``spec``'s image (created once)."""
        payload = lium_template_payload(spec.image, ssh_pubkey=spec.ssh_pubkey,
                                        ssh_port=spec.ssh_port)
        if self._template is not None:
            return str(self._template(payload))
        return lium_get_or_create_template(
            payload, api_key=self.api_key or os.environ.get("LIUM_API_KEY", ""))

    def machine_of(self, pod_id: str) -> str | None:
        """Executor id a pod was launched on (this process's launches only)."""
        return self._executor_by_name.get(pod_id)

    def create_unresolved(self, pod_id: str) -> bool:
        return bool(self.__dict__.get("_unresolved_creates", {}).get(pod_id))

    def create_requested(self, pod_id: str) -> bool:
        """A ``lium up`` for ``pod_id`` was issued by this instance. False when
        ``launch`` failed before it (the ``lium ls`` listing, the template):
        nothing was created, so the write-ahead intent has nothing to guard."""
        return pod_id in self._issued_creates

    # `lium rm` prose for a target the account does not have. Anything else
    # (a 401 on a revoked key, a transport error) keeps the create unresolved.
    _RM_NOT_FOUND_MARKERS = ("not found", "no such", "does not exist", "doesn't exist", "no pod")

    def _settle_ghost(self, pod_id: str, rm_error: str) -> None:
        """Resolve a ``lium up`` that never listed as "deployed nothing".

        The create is fire-and-forget, so a pod that never appeared kept its
        leg name ``create_unresolved`` for the rest of the seed epoch (every
        later rent refused, the sweep could not clear an intent no listing
        can resolve). Evidence that nothing was allocated, ALL required: the
        appear window has passed since the ``up`` was issued, ``lium rm`` by
        name said the pod does not exist (not an auth or transport failure),
        and two listings ``poll_interval`` apart both lack it. Anything less
        keeps it unresolved — an uncertain create never authorizes a second
        rental.
        """
        from ..funding.faults import classify_rent_failure

        issued = self._issued_creates.get(pod_id)
        if issued is None or self._now() - issued < LIUM_APPEAR_TIMEOUT:
            return
        low = rm_error.lower()
        if (classify_rent_failure(low) in ("auth", "rate_limited")
                or not any(m in low for m in self._RM_NOT_FOUND_MARKERS)):
            return
        try:
            for attempt in range(2):
                if attempt:
                    self._sleep(self.poll_interval)
                if self._pod(pod_id) is not None:
                    return
        except ProvisionError:
            return                                  # cannot list: stays unresolved
        self.__dict__.setdefault("_unresolved_creates", {})[pod_id] = False
        log.warning("lium pod %s: `lium up` deployed nothing (never listed within %.0fs, "
                    "`lium rm` finds no such pod, two empty listings) — the create is "
                    "resolved as never made", pod_id, LIUM_APPEAR_TIMEOUT)

    def pod_identity(self, pod_id: str) -> dict | None:
        """The platform's identity of the pod currently answering to ``pod_id``.

        ``{"id", "huid", "status", "ip", "port"}`` from ``lium ps`` — the
        account owner picks a pod's NAME, so a name can be re-used by a
        replacement; the platform-assigned ``id``/``huid`` cannot. ``None``
        when nothing by that name is listed.
        """
        pod = self._pod(pod_id)
        if not pod:
            return None
        addr = lium_pod_address(pod)
        return {"id": str(pod.get("id") or ""), "huid": str(pod.get("huid") or ""),
                "status": str(pod.get("status") or ""),
                "ip": addr.ip if addr else "", "port": addr.ssh_port if addr else 0}

    @staticmethod
    def _up_log_path(name: str) -> Path:
        import tempfile
        return Path(tempfile.gettempdir()) / f"lium-up-{name}.log"

    def _up_log_tail(self, name: str, n: int = 400) -> str:
        try:
            return self._up_log_path(name).read_text(errors="ignore")[-n:].strip()
        except OSError:
            return ""

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        appear_by = self._now() + min(LIUM_APPEAR_TIMEOUT, timeout)
        seen = False
        while self._now() < deadline:
            pod = self._pod(pod_id)
            if pod:
                seen = True
                if lium_pod_ready(pod):
                    return True
            elif not seen and self._now() >= appear_by:
                # Never listed ⇒ the `lium up` failed; don't burn the full
                # ready timeout on a ghost. Surface the captured up output.
                tail = self._up_log_tail(pod_id, n=2000)
                log.warning("lium pod %s never appeared in `lium ps` within %.0fs — "
                            "`lium up` likely failed; its output: %s",
                            pod_id, LIUM_APPEAR_TIMEOUT, tail[-400:] or "(no up log captured)")
                from ..funding.faults import is_auth_or_permission, is_funding_failure

                if is_funding_failure(tail) or is_auth_or_permission(tail):
                    # The account refused the rent (2026-10-29: Lium 403 "renting
                    # needs a balance above 15 minutes of the node's hourly
                    # price…"). Not the host's fault: raise the account's own
                    # words so the caller classifies it as auth/funding and the
                    # payer is told to top up, instead of renting again as if
                    # the host were a lemon.
                    raise ProvisionError(f"lium up refused by the account: {tail[-600:]}")
                return False
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        pod = self._pod(pod_id)
        return lium_pod_address(pod) if pod else None

    def live_pod_address(self, pod_id: str) -> PodAddress | None:
        """``pod_id``'s SSH endpoint iff the pod EXISTS and is RUNNING with
        one (``lium_pod_ready``); ``None`` for an absent, booting, or
        port-less (cooldown) pod. Unlike :meth:`get_ip` — which answers for
        any listed pod — this is the adoption check: the trainer's king-rent
        retry uses it to reuse a pod that is still up from a prior attempt of
        the same round (2026-09-14: a finished 3 h king checkpoint sat on a
        live pod while the retry rented a fresh one and retrained)."""
        pod = self._pod(pod_id)
        if not pod or not lium_pod_ready(pod):
            return None
        return lium_pod_address(pod)

    def _executor_of_pod(self, pod_id: str) -> str:
        """The executor a live pod sits on: this instance's own launch record,
        else the API's pod record (the CLI's ``ps`` JSON carries no executor)."""
        known = self._executor_by_name.get(pod_id)
        if known:
            return known
        try:
            import requests

            key = self.api_key or os.environ.get("LIUM_API_KEY", "")
            if not key:
                return ""
            r = requests.get(f"{LIUM_API_BASE}/pods", headers={"X-API-KEY": key}, timeout=20)
            if not r.ok:
                return ""
            for p in r.json() or []:
                if pod_id in (p.get("pod_name"), p.get("name"), p.get("id"), p.get("huid")):
                    return str(p.get("executor_id") or "")
        except Exception:  # noqa: BLE001 — best-effort lookup
            return ""
        return ""

    _EXECUTOR_HOSTS_TTL = 120.0

    def _api_json(self, path: str):
        """``GET <LIUM_API_BASE>/<path>`` on this instance's key; ``None`` on any
        failure (best-effort lookups only — never a rent decision by itself)."""
        try:
            import requests

            key = self.api_key or os.environ.get("LIUM_API_KEY", "")
            if not key:
                return None
            r = requests.get(f"{LIUM_API_BASE}/{path.lstrip('/')}",
                             headers={"X-API-KEY": key}, timeout=20)
            if not r.ok:
                return None
            return r.json()
        except Exception:  # noqa: BLE001 — best-effort lookup
            return None

    def _executor_cpus(self) -> dict[str, str]:
        """executor id → CPU model (``GET /executors`` ``specs.cpu.model``; the
        CLI's ``ls`` JSON carries no CPU). Cached like the host map; empty
        when the API is unreachable (then the blocklist cannot steer — logged)."""
        cached = self._executor_cpus_cache
        if cached and self._now() - cached[0] < self._EXECUTOR_HOSTS_TTL:
            return cached[1]
        cpus: dict[str, str] = {}
        for e in self._api_json("executors") or []:
            if not (isinstance(e, dict) and e.get("id")):
                continue
            specs = e.get("specs") if isinstance(e.get("specs"), dict) else {}
            cpu = specs.get("cpu") if isinstance(specs.get("cpu"), dict) else {}
            model = cpu.get("model")
            if model:
                cpus[str(e["id"])] = str(model)
        if not cpus:
            log.warning("lium: executor CPU map unavailable — the CPU blocklist cannot "
                        "steer this listing")
        object.__setattr__(self, "_executor_cpus_cache", (self._now(), cpus))
        return cpus

    def _executor_hosts(self) -> dict[str, str]:
        """executor id → host IP for the whole marketplace (the CLI's ``ls`` JSON
        carries no address; ``GET /executors`` does). Cached briefly per
        instance; empty when the API is unreachable (the fingerprint check
        still guards every leg — this only steers listings)."""
        cached = self._executor_hosts_cache
        if cached and self._now() - cached[0] < self._EXECUTOR_HOSTS_TTL:
            return cached[1]
        hosts: dict[str, str] = {}
        for e in self._api_json("executors") or []:
            if isinstance(e, dict) and e.get("id") and e.get("executor_ip_address"):
                hosts[str(e["id"])] = str(e["executor_ip_address"])
        if not hosts:
            log.warning("lium: executor host map unavailable — host quarantine cannot "
                        "steer this listing")
        object.__setattr__(self, "_executor_hosts_cache", (self._now(), hosts))
        return hosts

    def host_of_pod(self, pod_id: str) -> str:
        """Host IP a live pod sits on (from the pod's API record, else this
        instance's launch record + the executor map); "" when unknown."""
        for p in self._api_json("pods") or []:
            if isinstance(p, dict) and pod_id in (p.get("pod_name"), p.get("name"),
                                                  p.get("id"), p.get("huid")):
                ex = p.get("executor") or {}
                ip = ex.get("executor_ip_address") if isinstance(ex, dict) else ""
                if ip:
                    return str(ip)
        executor = self._executor_by_name.get(pod_id, "")
        return self._executor_hosts().get(executor, "") if executor else ""

    def terminate(self, pod_id: str) -> None:
        # `lium rm <target>` takes a positional target and does NOT prompt — there
        # is no --yes flag (verified against the installed CLI). Passing one would
        # error and we'd mistake a live pod for a terminated one, i.e. leak it.
        executor = self._executor_of_pod(pod_id)      # before the pod vanishes
        rm_error = ""
        try:
            self._cli(["rm", pod_id])
            log.info("lium rm %s", pod_id)
        except ProvisionError as e:
            # Idempotent: an already-gone pod is success, not a leak.
            log.warning("lium rm %s: %s (treating as already terminated)", pod_id, e)
            rm_error = str(e)
        # The vacated executor deploys nothing for a while: keep every listing
        # (ours and each payer's) off it until the cooldown passes.
        record_executor_cooldown(executor)
        if rm_error and self.create_unresolved(pod_id):
            self._settle_ghost(pod_id, rm_error)

    def list_tagged(self, prefix: str) -> list[str]:
        """Live pod names starting with ``prefix`` (Lium addresses pods by name)."""
        return filter_tagged_names(self._list_pods(), prefix, id_key="name")


def scan_ssh_host_key(ip: str, port: int, *, timeout: float = 15.0,
                      runner: Callable[[list[str]], subprocess.CompletedProcess] | None = None) -> str:
    """``<keytype> <base64>`` of the sshd at ``ip:port`` (ed25519 preferred).

    Read once when a pod is ready and pinned for the leg
    (:attr:`RemoteHost.pinned_host_key`): a container generates its host
    key at first boot, so a pod replaced under the same address cannot
    present it. Raises ProvisionError when nothing scans.
    """
    argv = ["ssh-keyscan", "-T", str(int(timeout)), "-p", str(port), ip]
    proc = (runner or (lambda a: subprocess.run(a, capture_output=True, text=True,
                                                timeout=timeout + 10)))(argv)
    lines = [ln.split(None, 1) for ln in (proc.stdout or "").splitlines()
             if ln and not ln.startswith("#")]
    keys = {}
    for parts in lines:
        if len(parts) == 2:
            keytype_key = parts[1].split()
            if len(keytype_key) >= 2:
                keys[keytype_key[0]] = f"{keytype_key[0]} {keytype_key[1]}"
    for kt in ("ssh-ed25519", "ecdsa-sha2-nistp256", "ssh-rsa"):
        if kt in keys:
            return keys[kt]
    raise ProvisionError(f"no ssh host key scanned at {ip}:{port}")


# ── Shadeform adapter (REST) ─────────────────────────────────────────────────


def _shadeform_container_status(info: dict) -> str:
    launch = info.get("launch_configuration") or {}
    status = info.get("container_status") or launch.get("container_status")
    return str(status or ("pending" if launch.get("type") == "docker" else "")).lower()


@dataclass
class ShadeformProvider:
    """Shadeform marketplace via its REST API (``X-API-KEY``).

    API key from ``$SHADEFORM_API_KEY`` (read lazily so a lower-priority provider
    needs no key when a higher one wins). SSH is injected via the container's
    ``SSH_PUBKEY`` env; container port 22 is exposed via ``port_mappings``.
    """

    name: str = "shadeform"
    base_url: str = "https://api.shadeform.ai/v1"
    api_key_env: str = "SHADEFORM_API_KEY"
    # Registered account key id → VM-mode launches (bare VM + bootstrap_script,
    # user "shadeform"); empty → docker-mode (the worker-image contract).
    ssh_key_id: str = ""
    api_key: str = field(default="", repr=False)
    max_price_per_hour: float = 0.0
    max_leg_cost_usd: float = 0.0
    sku_wall_seconds: tuple[tuple[str, int], ...] = ()
    default_wall_seconds: float = 0.0
    poll_interval: float = POD_POLL_INTERVAL
    _session: object | None = field(default=None, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    # region each pod was placed in (this process's launches only) — feeds the
    # replacement path's facility exclusion after an image_digest gate failure.
    _region_by_pod: dict = field(default_factory=dict, repr=False)
    _sku_by_pod: dict = field(default_factory=dict, repr=False)
    # names whose create request went out and was not definitely refused.
    _issued_creates: set = field(default_factory=set, repr=False)

    def _http(self):
        if self._session is not None:
            return self._session
        try:
            import requests
        except ModuleNotFoundError as e:  # pragma: no cover - env guard
            raise ProvisionError(
                "the shadeform adapter needs `requests` (pip install -e '.[deploy]')"
            ) from e
        key = self.api_key or os.environ.get(self.api_key_env)
        if not key:
            raise ProvisionError(f"shadeform: set ${self.api_key_env}")
        s = requests.Session()
        s.headers.update({"X-API-KEY": key, "Content-Type": "application/json"})
        self._session = s
        return s

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self._http().get(f"{self.base_url}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict | None = None) -> dict:
        r = self._http().post(f"{self.base_url}{path}", json=body, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else {}

    def _offer(self, sku: str, *, gpus: int = 1,
               exclude_regions: tuple[str, ...] = ()) -> dict | None:
        types = self._get("/instances/types", {"gpu_type": sku, "available": "true"})
        limits = [self.max_price_per_hour] if self.max_price_per_hour > 0 else []
        if self.max_leg_cost_usd > 0:
            wall = dict(self.sku_wall_seconds).get(sku, self.default_wall_seconds)
            if wall <= 0:
                raise ProvisionError("shadeform: per-leg cap requires a positive training wall")
            limits.append(self.max_leg_cost_usd * 3600.0 / wall)
        cap = min(limits) if limits else math.inf
        filtered = []
        for row in types.get("instance_types", []):
            try:
                price = float(row["hourly_price"]) / 100.0
            except (ValueError, TypeError, KeyError):
                continue
            if math.isfinite(price) and 0 <= price <= cap:
                filtered.append(row)
        offer = pick_shadeform_offer({"instance_types": filtered}, sku, gpus=gpus,
                                     exclude_regions=exclude_regions)
        if offer is not None:
            quote = next(row for row in filtered
                         if row.get("cloud") == offer["cloud"]
                         and row.get("shade_instance_type") == offer["shade_instance_type"])
            offer = {**offer, "price_usd_hr": float(quote["hourly_price"]) / 100.0,
                     "sku": sku}
        return offer

    def available(self, sku: str, count: int, *, gpus: int = 1) -> bool:
        # Shadeform reports availability, not exact counts; an available offer
        # (in the requested pod shape) means we can create the batch of `count`.
        return self._offer(sku, gpus=gpus) is not None

    def launch(self, spec: LaunchSpec) -> list[str]:
        offers = [offer for sku in spec.sku_choices or (spec.sku,)
                  if (offer := self._offer(sku, gpus=spec.gpus_per_pod,
                                           exclude_regions=spec.exclude_regions)) is not None]
        offer = min(offers, key=lambda row: row["price_usd_hr"]) if offers else None
        if offer is None:
            raise ProvisionError("shadeform: no_capacity under the requested GPU/price limits")
        ids: list[str] = []
        for i in range(spec.count):
            name = f"{spec.name_prefix}-{i}"
            body = shadeform_create_body(spec, offer, name=name,
                                         ssh_key_id=self.ssh_key_id or None)
            if self.max_leg_cost_usd > 0:
                from decimal import ROUND_DOWN, Decimal

                threshold = Decimal(str(self.max_leg_cost_usd)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
                if threshold <= 0:
                    raise ProvisionError("shadeform: per-instance spend threshold must be at least $0.01")
                body["auto_delete"] = {"spend_threshold": str(threshold)}
            delete_at = shadeform_delete_at(getattr(spec, "ttl_seconds", 0.0))
            if delete_at:
                body.setdefault("auto_delete", {})["date_threshold"] = delete_at
            # From here the create may have reached the API (``create_requested``).
            # A 4xx answer is a definite refusal — nothing allocated; a 5xx, a
            # timeout or a 2xx without an id leaves the outcome uncertain.
            self._issued_creates.add(name)
            try:
                resp = self._post("/instances/create", body)
            except Exception as exc:  # noqa: BLE001 — sort refusal from uncertainty
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(status, int) and 400 <= status < 500 and status not in (408, 425):
                    self._issued_creates.discard(name)
                    raise ProvisionError(f"shadeform create refused ({status}): {exc}") from None
                raise UncertainCreateError(f"shadeform create {name}: {exc}") from None
            iid = resp.get("id")
            if not iid:
                raise UncertainCreateError(f"shadeform create returned no id: {resp}")
            log.info("shadeform create → %s (%s/%s)", iid, offer["cloud"], offer["region"])
            ids.append(str(iid))
            self._region_by_pod[str(iid)] = str(offer["region"] or "")
            self._sku_by_pod[str(iid)] = str(offer["sku"])
        return ids

    def region_of(self, pod_id: str) -> str | None:
        """Facility (region) a pod was placed in (this process's launches only)."""
        return self._region_by_pod.get(pod_id) or None

    def create_requested(self, pod_id: str) -> bool:
        """A create for ``pod_id`` (the requested NAME) went out and was not
        definitely refused; False when ``launch`` failed before it."""
        return pod_id in self._issued_creates

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        last_container = None
        while self._now() < deadline:
            info = self._get(f"/instances/{pod_id}/info")
            status = str(info.get("status", "")).lower()
            if status in SHADEFORM_TERMINAL_BAD:
                raise ProvisionError(f"shadeform instance {pod_id} entered {status!r}")
            if status == SHADEFORM_READY:
                container = _shadeform_container_status(info)
                if container in SHADEFORM_CONTAINER_BAD:
                    raise ProvisionError(
                        f"shadeform instance {pod_id} container entered {container!r}"
                    )
                # No container_status field ⇒ VM-mode rental: active is ready.
                if not container or container == SHADEFORM_CONTAINER_READY:
                    return True
                if container != last_container:
                    log.info("shadeform %s active, container %s — waiting for %s",
                             pod_id, container, SHADEFORM_CONTAINER_READY)
                    last_container = container
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        return shadeform_pod_address(self._get(f"/instances/{pod_id}/info"))

    def launched_image_digest(self, pod_id: str) -> str:
        """The image digest Shadeform's own record says this instance runs.

        Fallback attestation for the health gate: sshd-as-PID-1 worker images
        destroy their /proc/1/environ (setproctitle), so the launch-env digest
        can't be read off the pod itself (live 2026-07-15)."""
        try:
            info = self._get(f"/instances/{pod_id}/info")
        except Exception:  # noqa: BLE001 — attestation is best-effort, gate decides
            return ""
        docker = (info.get("launch_configuration") or {}).get("docker_configuration") or {}
        return image_digest_of(str(docker.get("image") or ""))

    def terminate(self, pod_id: str) -> None:
        try:
            self._post(f"/instances/{pod_id}/delete")
            log.info("shadeform delete %s", pod_id)
        except Exception as e:  # noqa: BLE001 — idempotent teardown, already-gone is fine
            log.warning("shadeform delete %s: %s (treating as already terminated)", pod_id, e)

    def pod_exists(self, pod_id: str) -> bool:
        # Shadeform IDs are UUIDs; list_tagged filters NAMES. Looking up a
        # UUID as a name prefix falsely confirms a failed deletion as gone.
        return any(str(row.get("id") or "") == pod_id and
                   str(row.get("status") or "").lower() not in {"deleted", "terminated"}
                   for row in self._get("/instances").get("instances", []))

    def list_tagged(self, prefix: str) -> list[str]:
        """Live instance IDs whose ``name`` starts with ``prefix`` (delete takes the id)."""
        return list(self.list_named(prefix))

    def machine_of(self, pod_id: str) -> str:
        # The marketplace allocates the instance atomically. Namespace its
        # claim so it cannot collide with a Lium executor ID.
        return f"shadeform:{pod_id}"

    def sku_of(self, pod_id: str) -> str:
        cached = self._sku_by_pod.get(pod_id)
        if cached:
            return cached
        info = self._get(f"/instances/{pod_id}/info")
        sku = str((info.get("configuration") or {}).get("gpu_type") or "")
        if not sku:
            raise ProvisionError("shadeform: cannot recover the allocated GPU type")
        return sku

    def list_named(self, prefix: str) -> dict[str, str]:
        return {str(row["id"]): str(row.get("name") or "")
                for row in self._get("/instances").get("instances", [])
                if row.get("id") and str(row.get("name") or "").startswith(prefix)
                and str(row.get("status") or "").lower() not in {"deleted", "terminated"}}

    def pod_identity(self, pod_id: str) -> dict | None:
        # A UUID identifies the allocation; the account's chosen name does not.
        info = self._get(f"/instances/{pod_id}/info")
        status = str(info.get("status") or "").lower()
        container = _shadeform_container_status(info)
        ready = status == SHADEFORM_READY and (not container or container == SHADEFORM_CONTAINER_READY)
        return {"id": str(info.get("id") or ""),
                "status": "RUNNING" if ready else status.upper()}

    def live_pod_address(self, pod_id: str) -> PodAddress | None:
        info = self._get(f"/instances/{pod_id}/info")
        if str(info.get("status") or "").lower() != SHADEFORM_READY:
            return None
        container = _shadeform_container_status(info)
        if container and container != SHADEFORM_CONTAINER_READY:
            return None
        return shadeform_pod_address(info)

    def offer_price(self, sku: str) -> float | None:
        """Cheapest available USD/hr for ``sku`` (the budget breaker's input)."""
        types = self._get("/instances/types", {"gpu_type": sku, "available": "true"})
        return shadeform_offer_price_usd_hr(types, sku)


_PROVIDER_FACTORIES: dict[str, Callable[[], Provider]] = {
    "lium": LiumProvider,
    "shadeform": ShadeformProvider,
    # "targon": TargonProvider,  # seam: add for SKUs where Targon has capacity.
}


def build_providers(
    priority: Sequence[str], options: dict[str, dict] | None = None
) -> list[Provider]:
    """Instantiate the requested providers in priority order.

    ``options`` maps provider name → constructor kwargs (e.g. shadeform's
    ``ssh_key_id`` for VM-mode launches) — unknown providers still raise."""
    out: list[Provider] = []
    for name in priority:
        factory = _PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise ProvisionError(
                f"unknown provider {name!r}; known: {', '.join(sorted(_PROVIDER_FACTORIES))}"
            )
        out.append(factory(**(options or {}).get(name, {})))
    return out


# ── orchestration (control flow is unit-tested with fake providers) ──────────


@dataclass
class RenderOpts:
    """Non-address inputs to :func:`render_hosts_toml`, carried through a run."""

    key_path: str
    forward_env: Sequence[str]
    remote_python: str = DEFAULT_REMOTE_PYTHON
    workdir: str = DEFAULT_WORKDIR
    chain_toml: str | None = None
    stage: str = "any"


def _sidecar_path(hosts_path: Path) -> Path:
    """Path of the pod-id record written beside ``hosts.toml`` for recovery."""
    return hosts_path.with_suffix(hosts_path.suffix + ".pods.json")


def provision_and_run(
    provider: Provider,
    spec: LaunchSpec,
    *,
    hosts_path: Path,
    render_opts: RenderOpts,
    ready_timeout: float = DEFAULT_READY_TIMEOUT,
    run_trainer: bool = False,
    trainer_argv_extra: Sequence[str] = (),
    # injectables (real implementations by default):
    ssh_probe: Callable[[str, int], bool] = lambda ip, port: wait_ssh_reachable(
        ip, port, timeout=DEFAULT_READY_TIMEOUT),
    trainer_runner: Callable[[Sequence[str]], int] | None = None,
    write_text: Callable[[Path, str], None] | None = None,
    remove_file: Callable[[Path], None] | None = None,
) -> Path:
    """Launch → wait → template → (optionally) train, with GUARANTEED teardown.

    On any error or Ctrl-C after launch, every launched pod is terminated
    (idempotently) in the ``finally`` block. Pods are left running only on a
    clean *hand-off* — a successful run without ``--run-trainer``, where the user
    drives the trainer themselves against the templated ``hosts.toml``.
    """
    _write = write_text or (lambda p, t: p.write_text(t, encoding="utf-8"))
    _remove = remove_file or (lambda p: p.exists() and p.unlink())
    _train = trainer_runner or _run_trainer_subprocess
    sidecar = _sidecar_path(hosts_path)

    launched: list[str] = []
    handoff = False
    try:
        launched = provider.launch(spec)
        # Record ids immediately so a hard kill still leaves a teardown trail.
        _write(sidecar, json.dumps({"provider": provider.name, "pod_ids": launched}, indent=2))
        log.info("launched %d pod(s) on %s: %s", len(launched), provider.name, ", ".join(launched))

        addrs: list[PodAddress] = []
        for pid in launched:
            if not provider.wait_ready(pid, timeout=ready_timeout):
                raise ProvisionError(f"{provider.name} pod {pid} not ready within {ready_timeout:.0f}s")
            addr = provider.get_ip(pid)
            if addr is None:
                raise ProvisionError(f"{provider.name} pod {pid} exposed no IP")
            if not ssh_probe(addr.ip, addr.ssh_port):
                raise ProvisionError(f"pod {pid} SSH {addr.ip}:{addr.ssh_port} unreachable")
            log.info("pod %s ready at %s:%d", pid, addr.ip, addr.ssh_port)
            addrs.append(addr)

        hosts_toml = render_hosts_toml(
            addrs,
            key_path=render_opts.key_path,
            forward_env=render_opts.forward_env,
            remote_python=render_opts.remote_python,
            workdir=render_opts.workdir,
            chain_toml=render_opts.chain_toml,
            name_prefix=spec.name_prefix,
            provider=provider.name,
            stage=render_opts.stage,
        )
        _write(hosts_path, hosts_toml)
        log.info("wrote %s (%d host(s))", hosts_path, len(addrs))

        if run_trainer:
            argv = ["cascade-trainer", "--remote-hosts", str(hosts_path), *trainer_argv_extra]
            log.info("running trainer: %s", " ".join(argv))
            rc = _train(argv)
            if rc != 0:
                raise ProvisionError(f"cascade-trainer exited {rc}")
        else:
            handoff = True
            log.info("pods left running for manual trainer use; tear down with: "
                     "python deploy/provision.py --teardown --hosts-out %s", hosts_path)
        return hosts_path
    finally:
        if launched and not handoff:
            teardown(provider, launched)
            _remove(sidecar)


def teardown(provider: Provider, pod_ids: Sequence[str]) -> None:
    """Idempotently terminate every pod — never leak a running pod."""
    for pid in pod_ids:
        try:
            provider.terminate(pid)
        except Exception as e:  # noqa: BLE001 — best-effort; keep tearing down the rest
            log.error("failed to terminate %s (may be leaked!): %s", pid, e)


def _run_trainer_subprocess(argv: Sequence[str]) -> int:
    """Run ``cascade-trainer`` inheriting stdio so training streams to the console."""
    return subprocess.run(list(argv)).returncode


# ── CLI ──────────────────────────────────────────────────────────────────────


def _read_pubkey(value: str) -> str:
    """Resolve the SSH public key: an inline ``ssh-…`` string or a path to a file."""
    if value.startswith(("ssh-", "ecdsa-", "sk-")):
        return value.strip()
    p = Path(value).expanduser()
    if not p.is_file():
        raise ProvisionError(f"ssh pubkey not found (and not an inline key): {value}")
    return p.read_text(encoding="utf-8").strip()


def _default_key_path(pubkey_arg: str, override: str | None) -> str:
    """Private-key path for hosts.toml: explicit override, else pubkey path minus .pub."""
    if override:
        return override
    if pubkey_arg.endswith(".pub"):
        return pubkey_arg[: -len(".pub")]
    return "~/.ssh/id_ed25519"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="provision.py",
        description="Provision ephemeral GPU pods across providers for a cascade training round.",
    )
    p.add_argument("--sku", default=DEFAULT_SKU, help=f"GPU SKU (default {DEFAULT_SKU}).")
    p.add_argument("-n", "--count", type=int, default=DEFAULT_POD_COUNT,
                   help=f"Number of pods, all the same SKU (default {DEFAULT_POD_COUNT}).")
    p.add_argument("--image", help="Digest-pinned worker image (…@sha256:<64hex>).")
    p.add_argument("--stage", default="any", choices=("any", "final"),
                   help="Lane class these pods serve (default any); rolling fallback legs take "
                        "final/any lanes. Finals must be a single SKU (validator gpu_name gate).")
    p.add_argument("--name-prefix", default="cascade-pod",
                   help="hosts.toml [[host]] name prefix (default cascade-pod); e.g. "
                        "'cascade-final' to tell lane classes apart.")
    p.add_argument("--ssh-pubkey",
                   help="Orchestrator SSH public key: inline 'ssh-…' or a path to a .pub file.")
    p.add_argument("--ssh-key-path", default=None,
                   help="Private key path written into hosts.toml (default: pubkey path minus .pub).")
    p.add_argument("--forward-env", action="append", default=None,
                   help="Hippius cred env var NAME to forward per dispatch (repeatable; "
                        f"default: {', '.join(DEFAULT_FORWARD_ENV)}).")
    p.add_argument("--providers", default=",".join(DEFAULT_PROVIDER_PRIORITY),
                   help="Comma-separated provider priority order (default: "
                        f"{','.join(DEFAULT_PROVIDER_PRIORITY)}).")
    p.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT,
                   help=f"Container SSH port to expose (default {DEFAULT_SSH_PORT}).")
    p.add_argument("--hosts-out", type=Path, default=Path("hosts.toml"),
                   help="Where to write the templated hosts.toml (default ./hosts.toml).")
    p.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    p.add_argument("--workdir", default=DEFAULT_WORKDIR)
    p.add_argument("--chain-toml", default=None, help="chain.toml path on the pod (if non-default).")
    p.add_argument("--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT,
                   help=f"Seconds to wait for each pod to be SSH-reachable (default {DEFAULT_READY_TIMEOUT:.0f}).")
    p.add_argument("--run-trainer", action="store_true",
                   help="Invoke `cascade-trainer --remote-hosts` after templating, then tear down.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (provider, pods, templated hosts.toml) without launching.")
    p.add_argument("--teardown", action="store_true",
                   help="Terminate the pods recorded next to --hosts-out and exit (idempotent).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("trainer_args", nargs="*",
                   help="Extra args passed through to cascade-trainer (after `--`).")
    return p


def _do_teardown(hosts_out: Path) -> int:
    """Read the sidecar pod record and terminate everything it lists."""
    sidecar = _sidecar_path(hosts_out)
    if not sidecar.is_file():
        log.error("no pod record at %s; nothing to tear down", sidecar)
        return 1
    rec = json.loads(sidecar.read_text(encoding="utf-8"))
    providers = {p.name: p for p in build_providers([rec["provider"]])}
    teardown(providers[rec["provider"]], rec.get("pod_ids", []))
    sidecar.unlink()
    log.info("torn down %d pod(s) from %s", len(rec.get("pod_ids", [])), sidecar)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.teardown:
        return _do_teardown(args.hosts_out)

    try:
        if not args.image:
            raise ProvisionError("--image is required (digest-pinned worker image)")
        if not args.ssh_pubkey:
            raise ProvisionError("--ssh-pubkey is required (orchestrator public key)")
        if args.count < 1:
            raise ProvisionError("--count must be >= 1")
        validate_digest_pinned(args.image)

        pubkey = _read_pubkey(args.ssh_pubkey)
        key_path = _default_key_path(args.ssh_pubkey, args.ssh_key_path)
        forward_env = tuple(args.forward_env) if args.forward_env else DEFAULT_FORWARD_ENV
        priority = [s.strip() for s in args.providers.split(",") if s.strip()]

        spec = LaunchSpec(
            sku=args.sku, count=args.count, image=args.image,
            ssh_pubkey=pubkey, ssh_port=args.ssh_port,
            name_prefix=args.name_prefix,
        )
        render_opts = RenderOpts(
            key_path=key_path, forward_env=forward_env,
            remote_python=args.remote_python, workdir=args.workdir,
            chain_toml=args.chain_toml, stage=args.stage,
        )

        providers = build_providers(priority)

        if args.dry_run:
            # A plan preview must not require live creds/CLI: probe if we can,
            # otherwise fall back to the top-priority provider and say so.
            chosen, confirmed = None, False
            try:
                chosen = select_provider(providers, args.sku, args.count)
                confirmed = chosen is not None
            except ProvisionError as e:
                log.warning("availability not probed (%s)", e)
            _print_dry_run(chosen or providers[0], spec, render_opts, args.hosts_out,
                           confirmed=confirmed)
            return 0

        chosen = select_provider(providers, args.sku, args.count)
        if chosen is None:
            log.error("no provider in [%s] has capacity for %d×%s — not substituting a different SKU",
                      ", ".join(priority), args.count, args.sku)
            return 3

        provision_and_run(
            chosen, spec,
            hosts_path=args.hosts_out,
            render_opts=render_opts,
            ready_timeout=args.ready_timeout,
            run_trainer=args.run_trainer,
            trainer_argv_extra=args.trainer_args,
            ssh_probe=lambda ip, port: wait_ssh_reachable(ip, port, timeout=args.ready_timeout),
        )
        return 0
    except KeyboardInterrupt:
        log.warning("interrupted — pods launched this run were torn down")
        return 130
    except ProvisionError as e:
        log.error("%s", e)
        return 2


def _print_dry_run(provider: Provider, spec: LaunchSpec, render_opts: RenderOpts,
                   hosts_out: Path, *, confirmed: bool) -> None:
    """Print the plan and the hosts.toml that WOULD be written (placeholder IPs)."""
    placeholders = [PodAddress(ip=f"<{spec.name_prefix}-{i}-ip>", ssh_port=spec.ssh_port)
                    for i in range(spec.count)]
    hosts_toml = render_hosts_toml(
        placeholders, key_path=render_opts.key_path, forward_env=render_opts.forward_env,
        remote_python=render_opts.remote_python, workdir=render_opts.workdir,
        chain_toml=render_opts.chain_toml, name_prefix=spec.name_prefix, provider=provider.name,
    )
    note = "first with capacity" if confirmed else "top priority — availability NOT probed"
    print("── provisioning plan (dry run) ──")
    print(f"provider : {provider.name} ({note})")
    print(f"sku      : {spec.sku}")
    print(f"pods     : {spec.count}")
    print(f"image    : {spec.image}")
    print(f"ssh port : {spec.ssh_port}")
    print(f"hosts.toml → {hosts_out}:\n")
    print(hosts_toml)


if __name__ == "__main__":
    sys.exit(main())
