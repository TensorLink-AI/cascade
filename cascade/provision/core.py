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
``forward_env`` in ``hosts.toml`` so the orchestrator passes them inline over SSH
per dispatch (see ``remote.build_remote_command``).

Reproducibility: the worker image MUST be pinned by ``@sha256:`` digest, not a
tag — ``chain.toml [training] expected_gpu`` and byte-exact re-derivation depend
on every pod running the identical image on the identical SKU. A round is never
split across two SKUs or two providers.
"""

from __future__ import annotations

import argparse
import json
import logging
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
DEFAULT_FORWARD_ENV = (
    "HIPPIUS_S3_ACCESS_KEY",
    "HIPPIUS_S3_SECRET_KEY",
    "HIPPIUS_HUB_USERNAME",
    "HIPPIUS_HUB_PASSWORD",
)
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


def image_digest_of(image: str) -> str:
    """The ``sha256:<hex>`` digest pinned in an image ref (empty when unpinned).

    Injected into pods as ``CASCADE_TRAIN_IMAGE_DIGEST`` at launch — the health
    gate REQUIRES it on image-boot pods and final workers refuse a mismatched
    runtime, so an image-boot pod without it can never pass health (found in a
    live image-boot dress rehearsal, 2026-07-15)."""
    _, sep, digest = image.partition("@")
    return digest if sep and digest.startswith("sha256:") else ""


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

    ``stage`` ("any" | "heat" | "final") tags which round stage these pods serve
    (see ``cascade.trainer.remote.RemoteHost``): a homogeneous batch is one stage,
    so provision a cheap heat fleet (``--stage heat``) and a single-SKU final pair
    (``--stage final``) as separate runs, then concatenate the ``[[host]]`` blocks.
    ``"any"`` is omitted (it is the schema default).
    """
    if not addrs:
        raise ProvisionError("cannot render hosts.toml with zero pods")
    if gpus_per_pod < 1:
        raise ProvisionError(f"gpus_per_pod must be >= 1; got {gpus_per_pod}")

    def _arr(items: Sequence[str]) -> str:
        return json.dumps(list(items))

    lines = [
        "# Generated by cascade.provision — trainer-local remote GPU pods"
        + (f" ({provider})" if provider else "") + ".",
        "# Ephemeral: torn down by the provisioner. Keep OFF git and chain.toml.",
        "# Hippius creds are forwarded inline per dispatch (forward_env), never on the pod.",
    ]
    for i, addr in enumerate(addrs):
        for g in range(gpus_per_pod):
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
        raise ProvisionError(f"could not parse `lium ls` JSON: {e}") from e
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
        raise ProvisionError(f"could not parse `lium ps` JSON: {e}") from e
    if not isinstance(data, list):
        raise ProvisionError(f"expected a JSON array from `lium ps`, got {type(data).__name__}")
    return data


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


def pick_shadeform_offer(types_json: dict, sku: str, *, gpus: int = 1) -> dict | None:
    """Choose a ``(cloud, region, shade_instance_type)`` offer for ``sku``.

    Filters ``GET /instances/types`` results to the requested ``gpu_type`` AND
    the exact ``gpus`` pod shape (``configuration.num_gpus``) with an available
    region, preferring the cheapest (``hourly_price``, in cents). Returns
    ``None`` if nothing is available — the fall-through signal.
    """
    offers: list[tuple[int, dict]] = []
    for t in types_json.get("instance_types", []):
        if str(t.get("configuration", {}).get("gpu_type", "")).upper() != sku.upper():
            continue
        if int(t.get("configuration", {}).get("num_gpus", 1) or 1) != int(gpus):
            continue
        region = next(
            (a.get("region") for a in t.get("availability", []) if a.get("available")),
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


def shadeform_offer_price_usd_hr(
    types_json: dict, sku: str, *, gpus: int | None = None
) -> float | None:
    """Cheapest available PER-POD hourly price for ``sku`` in USD/hr.

    The API reports ``hourly_price`` in CENTS; the provisioner's budget breaker
    (``policy.within_budget``) works in USD, so convert here — a silent
    cents-as-dollars mixup would either 100× the projection (refusing every
    round) or 1/100 it (defeating the breaker).

    ``gpus`` restricts the scan to the pod shape actually being rented, which
    is what :func:`pick_shadeform_offer` will launch. Leaving it ``None``
    scans every shape and therefore returns the 1× price for a 4× rung —
    a 4× UNDER-projection that hands the breaker a number it never spends
    against. The loop always passes the candidate's shape; ``None`` remains
    only for callers that genuinely want "cheapest at any shape".
    """
    prices = [
        int(t.get("hourly_price", 1 << 30))
        for t in types_json.get("instance_types", [])
        if str(t.get("configuration", {}).get("gpu_type", "")).upper() == sku.upper()
        and (gpus is None
             or int(t.get("configuration", {}).get("num_gpus", 1) or 1) == int(gpus))
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


# ── RunPod pure helpers ──────────────────────────────────────────────────────
#
# RunPod's GPU-type ids ARE the nvidia-smi device strings ("NVIDIA L40S",
# "NVIDIA GeForce RTX 4090"), so a stage needs no ``market_sku`` here — the
# health gate's exact device assertion and the marketplace name coincide.

RUNPOD_READY = "RUNNING"
RUNPOD_TERMINAL_BAD = {"FAILED", "TERMINATED", "EXITED", "DEAD"}
RUNPOD_SECURE = "SECURE"
RUNPOD_COMMUNITY = "COMMUNITY"
DEFAULT_RUNPOD_DISK_GB = 60


def _sku_key(name: str) -> str:
    """Whitespace-insensitive, case-insensitive key for a marketplace SKU name.

    The same silicon is spelled differently per marketplace — shadeform says
    ``RTX4090``, vast says ``RTX 4090``, runpod says ``NVIDIA GeForce RTX 4090``
    — but ``providers`` is a STAGE-level list, so one candidate's
    ``market_sku`` is offered to every adapter in the ladder. Folding spacing
    means a rung written for one marketplace still matches on another instead
    of silently probing as "no capacity". It never widens a match across
    genuinely different devices (``RTX 4090`` vs ``RTX 4090D``, ``L40`` vs
    ``L40S``) — those differ by more than spacing, which is the whole point of
    the health gate's exact device assertion downstream.
    """
    return "".join(name.split()).upper()


def _runpod_gpu_types(types_json: object) -> list[dict]:
    """Normalize ``GET /gputypes`` to a list across REST and GraphQL shapes."""
    if isinstance(types_json, list):
        return [t for t in types_json if isinstance(t, dict)]
    if isinstance(types_json, dict):
        for key in ("gpuTypes", "data"):
            inner = types_json.get(key)
            if isinstance(inner, dict):
                inner = inner.get("gpuTypes")
            if isinstance(inner, list):
                return [t for t in inner if isinstance(t, dict)]
    return []


def pick_runpod_gpu_type(
    types_json: object, sku: str, *, secure: bool = True
) -> dict | None:
    """The ``GET /gputypes`` entry for ``sku``, or ``None`` if it is not sold.

    Matches ``id`` (the nvidia-smi string) first, then ``displayName`` for the
    short aliases RunPod also answers to. ``secure`` requires the type to be
    offered on Secure Cloud — RunPod's own/partner datacenters — rather than
    Community Cloud's host marketplace; that tier choice is the whole reason
    to reach for this provider on the ``final`` stage, so a type that exists
    only on Community must NOT silently satisfy a secure request.

    RunPod publishes no per-shape stock count, so a returned entry means "this
    SKU is purchasable on this tier", not "N pods are free right now" — the
    same class of probe as Shadeform's, and the reason both are backed by the
    loop's dud counter and replacement path rather than trusted outright.
    """
    tier_flag = "secureCloud" if secure else "communityCloud"
    price_key = "securePrice" if secure else "communityPrice"
    want = _sku_key(sku)
    for t in _runpod_gpu_types(types_json):
        names = {_sku_key(str(t.get("id", ""))), _sku_key(str(t.get("displayName", "")))}
        if want not in names:
            continue
        # Absent flag ⇒ unknown tier support; only an explicit False excludes.
        if t.get(tier_flag) is False:
            continue
        price = t.get(price_key)
        if price is None or float(price) <= 0:
            continue
        return t
    return None


def runpod_gpu_price_usd_hr(
    types_json: object, sku: str, *, gpus: int = 1, secure: bool = True
) -> float | None:
    """PER-POD USD/hr for a ``gpus``-GPU pod of ``sku`` (``None`` if unsold).

    RunPod quotes **per GPU-hour**; ``policy.within_budget`` bills per POD-hour.
    Multiplying here is not cosmetic — a 4× pod priced per-GPU under-projects
    the round breaker 4×, which is exactly the failure mode the Shadeform
    cents-vs-dollars note guards against on the other adapter.
    """
    entry = pick_runpod_gpu_type(types_json, sku, secure=secure)
    if entry is None:
        return None
    price = entry.get("securePrice" if secure else "communityPrice")
    return float(price) * max(1, int(gpus))


def runpod_create_body(
    spec: LaunchSpec, gpu_type_id: str, *, name: str,
    cloud_type: str = RUNPOD_SECURE, disk_gb: int = DEFAULT_RUNPOD_DISK_GB,
) -> dict:
    """Build the ``POST /pods`` body for one pod of ``spec``.

    The pod boots the digest-pinned worker image, whose entrypoint starts sshd
    and reads ``SSH_PUBKEY``. ``PUBLIC_KEY`` carries the same key because
    RunPod's own base images seed ``authorized_keys`` from that name instead —
    setting both means a fallback image still admits the orchestrator. No
    credential is ever seeded; Hippius secrets ride the per-dispatch SSH env.

    ``interruptible`` is pinned False: a spot pod reclaimed mid-heat costs the
    round more than the discount saves, and mid-FINAL it costs the duel.
    """
    env = {"SSH_PUBKEY": spec.ssh_pubkey, "PUBLIC_KEY": spec.ssh_pubkey}
    digest = image_digest_of(spec.image)
    if digest:
        env["CASCADE_TRAIN_IMAGE_DIGEST"] = digest
    return {
        "name": name,
        "imageName": spec.image,
        "gpuTypeIds": [gpu_type_id],
        "gpuCount": int(spec.gpus_per_pod),
        "cloudType": cloud_type,
        "containerDiskInGb": int(disk_gb),
        "volumeInGb": 0,
        "ports": [f"{DEFAULT_SSH_PORT}/tcp"],
        "env": env,
        "supportPublicIp": True,
        "interruptible": False,
    }


def runpod_pod_address(
    info_json: dict, *, ssh_port: int = DEFAULT_SSH_PORT
) -> PodAddress | None:
    """Extract ``(ip, port)`` from a RunPod ``GET /pods/{id}`` response.

    RunPod NATs container 22 to a random public port, so the mapping — not
    port 22 — is where sshd answers. The mapping only materializes once the
    container is actually running, which makes "address resolvable" a
    stricter and more honest readiness signal than the pod's own status
    field (the Shadeform active-but-still-pulling lesson, 2026-07-15).

    Handles both the REST ``portMappings`` dict and the older
    ``runtime.ports`` list.
    """
    ip = str(info_json.get("publicIp") or "").strip()
    mappings = info_json.get("portMappings") or {}
    if isinstance(mappings, dict):
        mapped = mappings.get(str(DEFAULT_SSH_PORT)) or mappings.get(DEFAULT_SSH_PORT)
        if ip and mapped:
            return PodAddress(ip=ip, ssh_port=int(mapped))
    for p in ((info_json.get("runtime") or {}).get("ports") or []):
        if int(p.get("privatePort", 0) or 0) != DEFAULT_SSH_PORT:
            continue
        pub_ip = str(p.get("ip") or ip).strip()
        pub_port = p.get("publicPort")
        if pub_ip and pub_port:
            return PodAddress(ip=pub_ip, ssh_port=int(pub_port))
    return PodAddress(ip=ip, ssh_port=ssh_port) if ip else None


# ── Vast.ai pure helpers ─────────────────────────────────────────────────────
#
# Vast is a marketplace of INDIVIDUAL hosts, not a homogeneous fleet, so the
# offer filter carries quality floors the other adapters do not need. This is
# the DEC-CA-0010 constraint expressed as procurement: heat ranking compares
# runs across pods, so a fleet spanning wildly different CPU budgets or
# oversubscribed machines turns host variance into rank variance. Rent narrow.

VAST_READY = "running"
VAST_TERMINAL_BAD = {"exited", "error", "offline"}
DEFAULT_VAST_DISK_GB = 60
DEFAULT_VAST_MIN_RELIABILITY = 0.98
DEFAULT_VAST_MIN_CPU_CORES_PER_GPU = 4.0


def vast_offer_query(
    sku: str, *, gpus: int = 1, min_reliability: float = DEFAULT_VAST_MIN_RELIABILITY,
    verified_only: bool = True, limit: int = 64,
) -> dict:
    """The ``GET /bundles/`` search body for ``gpus``× ``sku`` offers.

    Server-side narrowing only — :func:`pick_vast_offer` re-applies every floor
    client-side, because the quality bar is a correctness property of the fleet
    and must not depend on the marketplace honouring a query parameter.
    """
    q: dict = {
        "gpu_name": {"eq": sku},
        "num_gpus": {"eq": int(gpus)},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "reliability2": {"gte": float(min_reliability)},
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "limit": int(limit),
    }
    if verified_only:
        q["verified"] = {"eq": True}
    return q


def filter_vast_offers(
    bundles_json: dict, sku: str, *, gpus: int = 1,
    exclude_machines: Sequence[str] = (),
    min_reliability: float = DEFAULT_VAST_MIN_RELIABILITY,
    min_cpu_cores_per_gpu: float = DEFAULT_VAST_MIN_CPU_CORES_PER_GPU,
    verified_only: bool = True,
) -> list[dict]:
    """Rentable offers matching the SKU, shape and quality floors, cheapest first.

    At most ONE offer per ``machine_id``: two lanes of the same physical box
    are co-tenants, and a "2-pod" fleet that lands twice on one machine is a
    1-pod fleet that bills double and shares a memory bus.

    ``min_cpu_cores_per_gpu`` is the floor that matters most for cascade: the
    corpus streams from a CPU generator (``[training] corpus_mode =
    stream_cpu``) and each lane's threads are capped to its slice of the box
    (``trainer.sandbox._lane_cpu_slice``), so a CPU-thin machine starves the
    run and reads as a slow generator in the heat.
    """
    excluded = {str(m) for m in exclude_machines}
    want = _sku_key(sku)
    out: list[tuple[float, dict]] = []
    for o in bundles_json.get("offers", []) or []:
        if _sku_key(str(o.get("gpu_name", ""))) != want:
            continue
        if int(o.get("num_gpus", 0) or 0) != int(gpus):
            continue
        if not o.get("rentable", False) or o.get("rented", False):
            continue
        if verified_only and not o.get("verified", False):
            continue
        if float(o.get("reliability2", 0.0) or 0.0) < float(min_reliability):
            continue
        cores = float(o.get("cpu_cores_effective", 0.0) or 0.0)
        if cores / max(1, int(gpus)) < float(min_cpu_cores_per_gpu):
            continue
        if str(o.get("machine_id", "")) in excluded:
            continue
        price = o.get("dph_total")
        if price is None or float(price) <= 0:
            continue
        out.append((float(price), o))
    out.sort(key=lambda x: (x[0], str(x[1].get("id", ""))))
    seen: set[str] = set()
    uniq: list[dict] = []
    for _, o in out:
        mid = str(o.get("machine_id", ""))
        if mid and mid in seen:
            continue
        if mid:
            seen.add(mid)
        uniq.append(o)
    return uniq


def pick_vast_offer(bundles_json: dict, sku: str, **kw) -> dict | None:
    """The cheapest qualifying offer, or ``None`` — the fall-through signal."""
    offers = filter_vast_offers(bundles_json, sku, **kw)
    return offers[0] if offers else None


def vast_offer_price_usd_hr(bundles_json: dict, sku: str, **kw) -> float | None:
    """Cheapest qualifying PER-POD USD/hr (``dph_total`` covers the whole box).

    Priced off the SAME filtered set the launch will choose from, so the
    breaker never approves a round against a cheap offer the quality floors
    would have rejected.
    """
    offer = pick_vast_offer(bundles_json, sku, **kw)
    return float(offer["dph_total"]) if offer else None


def vast_create_body(spec: LaunchSpec, *, label: str, disk_gb: int = DEFAULT_VAST_DISK_GB) -> dict:
    """Build the ``PUT /asks/{id}/`` body for one pod of ``spec``.

    ``runtype: "ssh"`` with ``direct`` asks Vast for a routable host:port into
    the container rather than its SSH proxy. The ``-p 22:22`` env key is Vast's
    docker-options channel (its CLI emits the same), and ``SSH_PUBKEY`` is the
    worker-image contract. NOTE the orchestrator's public key must ALSO be
    registered on the Vast account: Vast seeds its own authorized_keys on the
    ssh runtype, and an image whose entrypoint ignores ``SSH_PUBKEY`` would
    otherwise be unreachable.
    """
    env = {f"-p {DEFAULT_SSH_PORT}:{DEFAULT_SSH_PORT}": "1", "SSH_PUBKEY": spec.ssh_pubkey}
    digest = image_digest_of(spec.image)
    if digest:
        env["CASCADE_TRAIN_IMAGE_DIGEST"] = digest
    return {
        "client_id": "me",
        "image": spec.image,
        "disk": int(disk_gb),
        "runtype": "ssh",
        "direct": True,
        "label": label,
        "env": env,
    }


def vast_pod_address(inst: dict, *, ssh_port: int = DEFAULT_SSH_PORT) -> PodAddress | None:
    """Extract ``(ip, port)`` from a Vast instance record.

    Prefers the direct public mapping of container 22; falls back to Vast's
    ``ssh_host``/``ssh_port`` proxy, which resolves once the instance runs.
    """
    ports = inst.get("ports") or {}
    entry = ports.get(f"{DEFAULT_SSH_PORT}/tcp") or ports.get(str(DEFAULT_SSH_PORT))
    if isinstance(entry, list) and entry:
        entry = entry[0]
    ip = str(inst.get("public_ipaddr") or "").strip()
    if ip and isinstance(entry, dict) and entry.get("HostPort"):
        return PodAddress(ip=ip, ssh_port=int(entry["HostPort"]))
    if inst.get("ssh_host") and inst.get("ssh_port"):
        return PodAddress(ip=str(inst["ssh_host"]).strip(), ssh_port=int(inst["ssh_port"]))
    return PodAddress(ip=ip, ssh_port=ssh_port) if ip else None


# ── side-effecting helpers (adapter surface, not unit-tested) ─────────────────


def _run_cli(argv: list[str], timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run a CLI command and capture output (text mode)."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


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


def _spawn_cli(argv: list[str], log_path: Path | None = None) -> subprocess.Popen:
    """Fire-and-forget a CLI command (e.g. ``lium up``, which attaches/streams).

    We don't wait on it — pod readiness is observed via ``lium ps`` polling, and
    killing the local client does not stop the remote pod. ``log_path`` captures
    the child's output: ``lium up`` DRIVES the deploy client-side, so its output
    is the only evidence when a pod silently never materializes (observed live:
    an executor in post-teardown cooldown accepts the up, deploys nothing, and
    a DEVNULL'd spawn leaves a 900s ghost hunt with zero diagnostics).
    """
    if log_path is None:
        return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(log_path, "ab") as f:
        return subprocess.Popen(argv, stdout=f, stderr=f)


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

    Auth is the CLI's own (``LIUM_API_KEY`` / ``~/.lium/config.ini``). Each pod is
    one ``lium up`` on a distinct executor; SSH is injected via the container's
    ``SSH_PUBKEY`` env (the CLI has no direct pubkey flag). Pods are addressed by
    the ``--name`` we assign (Lium's ``ps``/``rm`` accept name, huid, or id).
    """

    name: str = "lium"
    bin: str = field(default_factory=_default_lium_bin)
    poll_interval: float = POD_POLL_INTERVAL
    _run: Callable[[list[str]], subprocess.CompletedProcess] | None = field(default=None, repr=False)
    _spawn: Callable[[list[str]], object] | None = field(default=None, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    # pod name → executor id for pods this instance launched, so the loop can
    # exclude a failed pod's machine when renting its replacement.
    _executor_by_name: dict = field(default_factory=dict, repr=False)

    def _cli(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            proc = (self._run or _run_cli)([self.bin, *args])
        except FileNotFoundError as e:
            # A missing binary is a config error, not "no capacity" — surface it
            # loudly instead of silently falling through to another provider.
            raise ProvisionError(
                f"lium CLI not found at {self.bin!r}; install it (pip install lium.io)"
            ) from e
        if proc.returncode != 0:
            raise ProvisionError(
                f"`{self.bin} {' '.join(args)}` failed (rc={proc.returncode}): "
                f"{(proc.stderr or '')[-500:]}"
            )
        return proc

    def _list_executors(self, sku: str, *, gpus: int = 1) -> list[dict]:
        """Marketplace executors of ``sku`` with EXACTLY ``gpus`` GPUs.

        Shape matters: the fleet plan fans one hosts.toml lane out per GPU, so
        a 1× machine rented against an 8-lane plan strands seven lanes (and the
        health gate then kills the pod anyway — filter here, before renting)."""
        execs = parse_lium_executors(self._cli(["ls", "--gpu", sku, "--format", "json"]).stdout)
        return [e for e in execs if int(e.get("gpu_count", 1) or 1) == int(gpus)]

    def _list_pods(self) -> list[dict]:
        return parse_lium_pods(self._cli(["ps", "--format", "json"]).stdout)

    def _pod(self, pod_id: str) -> dict | None:
        return next((p for p in self._list_pods()
                     if pod_id in (p.get("name"), p.get("huid"), p.get("id"))), None)

    def available(self, sku: str, count: int, *, gpus: int = 1) -> bool:
        return len(self._list_executors(sku, gpus=gpus)) >= count

    def launch(self, spec: LaunchSpec) -> list[str]:
        execs = self._list_executors(spec.sku, gpus=spec.gpus_per_pod)
        if spec.exclude_ids:
            execs = [e for e in execs if str(e.get("id")) not in spec.exclude_ids]
        if len(execs) < spec.count:
            raise ProvisionError(
                f"lium: only {len(execs)} × {spec.gpus_per_pod}x{spec.sku} available"
                f"{' after exclusions' if spec.exclude_ids else ''}, need {spec.count}"
            )
        names: list[str] = []
        for i, ex in enumerate(execs[: spec.count]):
            name = f"{spec.name_prefix}-{i}"
            argv = [self.bin, "up", str(ex["id"])]
            if spec.image:
                # docker-run style: the image must be a REAL docker ref whose
                # entrypoint runs sshd and reads $SSH_PUBKEY (the worker image).
                argv += ["--image", spec.image, "-e", f"SSH_PUBKEY={spec.ssh_pubkey}",
                         "--internal-ports", str(spec.ssh_port)]
                digest = image_digest_of(spec.image)
                if digest:
                    argv += ["-e", f"CASCADE_TRAIN_IMAGE_DIGEST={digest}"]
            # empty image ⇒ lium's default SSH template (bootstrap mode): lium
            # injects the ACCOUNT's registered keys; a template NAME passed as
            # --image 400s ("image reference is not valid") — never do that.
            argv += ["--name", name, "--yes"]
            if self._spawn is not None:
                self._spawn(argv)
            else:
                _spawn_cli(argv, log_path=self._up_log_path(name))
            log.info("lium up → executor %s as %s", ex.get("id"), name)
            self._executor_by_name[name] = str(ex.get("id"))
            names.append(name)
        return names

    def machine_of(self, pod_id: str) -> str | None:
        """Executor id a pod was launched on (this process's launches only)."""
        return self._executor_by_name.get(pod_id)

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
                tail = self._up_log_tail(pod_id)
                log.warning("lium pod %s never appeared in `lium ps` within %.0fs — "
                            "`lium up` likely failed; its output: %s",
                            pod_id, LIUM_APPEAR_TIMEOUT, tail or "(no up log captured)")
                return False
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        pod = self._pod(pod_id)
        return lium_pod_address(pod) if pod else None

    def terminate(self, pod_id: str) -> None:
        # `lium rm <target>` takes a positional target and does NOT prompt — there
        # is no --yes flag (verified against the installed CLI). Passing one would
        # error and we'd mistake a live pod for a terminated one, i.e. leak it.
        try:
            self._cli(["rm", pod_id])
            log.info("lium rm %s", pod_id)
        except ProvisionError as e:
            # Idempotent: an already-gone pod is success, not a leak.
            log.warning("lium rm %s: %s (treating as already terminated)", pod_id, e)

    def list_tagged(self, prefix: str) -> list[str]:
        """Live pod names starting with ``prefix`` (Lium addresses pods by name)."""
        return filter_tagged_names(self._list_pods(), prefix, id_key="name")


# ── Shadeform adapter (REST) ─────────────────────────────────────────────────


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
    poll_interval: float = POD_POLL_INTERVAL
    _session: object | None = field(default=None, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)

    def _http(self):
        if self._session is not None:
            return self._session
        try:
            import requests
        except ModuleNotFoundError as e:  # pragma: no cover - env guard
            raise ProvisionError(
                "the shadeform adapter needs `requests` (pip install -e '.[deploy]')"
            ) from e
        key = os.environ.get(self.api_key_env)
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

    def _offer(self, sku: str, *, gpus: int = 1) -> dict | None:
        types = self._get("/instances/types", {"gpu_type": sku, "available": "true"})
        return pick_shadeform_offer(types, sku, gpus=gpus)

    def available(self, sku: str, count: int, *, gpus: int = 1) -> bool:
        # Shadeform reports availability, not exact counts; an available offer
        # (in the requested pod shape) means we can create the batch of `count`.
        return self._offer(sku, gpus=gpus) is not None

    def launch(self, spec: LaunchSpec) -> list[str]:
        offer = self._offer(spec.sku, gpus=spec.gpus_per_pod)
        if offer is None:
            raise ProvisionError(f"shadeform: no available {spec.sku} offer")
        ids: list[str] = []
        for i in range(spec.count):
            body = shadeform_create_body(spec, offer, name=f"{spec.name_prefix}-{i}",
                                         ssh_key_id=self.ssh_key_id or None)
            resp = self._post("/instances/create", body)
            iid = resp.get("id")
            if not iid:
                raise ProvisionError(f"shadeform create returned no id: {resp}")
            log.info("shadeform create → %s (%s/%s)", iid, offer["cloud"], offer["region"])
            ids.append(str(iid))
        return ids

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        last_container = None
        while self._now() < deadline:
            info = self._get(f"/instances/{pod_id}/info")
            status = str(info.get("status", "")).lower()
            if status in SHADEFORM_TERMINAL_BAD:
                raise ProvisionError(f"shadeform instance {pod_id} entered {status!r}")
            if status == SHADEFORM_READY:
                container = str(info.get("container_status") or "").lower()
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

    def list_tagged(self, prefix: str) -> list[str]:
        """Live instance IDs whose ``name`` starts with ``prefix`` (delete takes the id)."""
        return filter_tagged_names(
            self._get("/instances").get("instances", []), prefix, id_key="id"
        )

    def offer_price(self, sku: str, *, gpus: int = 1) -> float | None:
        """Cheapest available USD/pod-hr for ``sku`` at the ``gpus`` pod shape."""
        types = self._get("/instances/types", {"gpu_type": sku, "available": "true"})
        return shadeform_offer_price_usd_hr(types, sku, gpus=gpus)


# ── RunPod + Vast adapters (REST, bearer auth) ───────────────────────────────


class _BearerRestMixin:
    """Lazy ``requests`` session with ``Authorization: Bearer`` from the env.

    Lazy on purpose, exactly as the Shadeform adapter is: a lower-priority
    provider in the ladder must not demand an API key when a higher one wins
    the round.
    """

    base_url: str
    api_key_env: str
    name: str

    def _http(self):
        if getattr(self, "_session", None) is not None:
            return self._session
        try:
            import requests
        except ModuleNotFoundError as e:  # pragma: no cover - env guard
            raise ProvisionError(
                f"the {self.name} adapter needs `requests` (pip install -e '.[deploy]')"
            ) from e
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProvisionError(f"{self.name}: set ${self.api_key_env}")
        s = requests.Session()
        s.headers.update({"Authorization": f"Bearer {key}",
                          "Content-Type": "application/json"})
        self._session = s
        return s

    def _request(self, method: str, path: str, *, params=None, body=None) -> dict:
        r = self._http().request(method, f"{self.base_url}{path}",
                                 params=params, json=body, timeout=30)
        r.raise_for_status()
        if not r.content:
            return {}
        out = r.json()
        return out if isinstance(out, (dict, list)) else {}

    def _get(self, path: str, params: dict | None = None):
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict | None = None):
        return self._request("POST", path, body=body)


@dataclass
class RunPodProvider(_BearerRestMixin):
    """RunPod via its REST v1 API (``$RUNPOD_API_KEY``).

    Defaults to **Secure Cloud** — RunPod's own and partner datacenters — which
    is the point of reaching for it: the ``final`` stage's L40S duel is the one
    leg where a zombie rental costs the round, and Secure Cloud is a real
    operator rather than a reseller pool. ``cloud_type = "COMMUNITY"`` opts into
    the cheaper host marketplace for heat-class work.

    NO ``machine_of``: RunPod schedules the machine itself and ``POST /pods``
    cannot target or avoid one, so the loop's lemon-exclusion path has nothing
    to exclude with (see ``LaunchSpec.exclude_ids`` — adapters that cannot map
    ids to offers may ignore it). A bad pod is handled by the replacement +
    dud-counter path instead.
    """

    name: str = "runpod"
    base_url: str = "https://rest.runpod.io/v1"
    api_key_env: str = "RUNPOD_API_KEY"
    cloud_type: str = RUNPOD_SECURE
    disk_gb: int = DEFAULT_RUNPOD_DISK_GB
    poll_interval: float = POD_POLL_INTERVAL
    _session: object | None = field(default=None, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)

    @property
    def _secure(self) -> bool:
        return str(self.cloud_type).upper() != RUNPOD_COMMUNITY

    def _gpu_types(self):
        return self._get("/gputypes")

    def available(self, sku: str, count: int, *, gpus: int = 1) -> bool:  # noqa: ARG002
        # RunPod publishes no per-shape stock, so neither `count` nor `gpus`
        # can narrow this: the probe answers "sold on this tier", and the
        # dud counter + replacement path absorb the rest (see the class doc).
        return pick_runpod_gpu_type(self._gpu_types(), sku, secure=self._secure) is not None

    def launch(self, spec: LaunchSpec) -> list[str]:
        entry = pick_runpod_gpu_type(self._gpu_types(), spec.sku, secure=self._secure)
        if entry is None:
            raise ProvisionError(
                f"runpod: {spec.sku} not offered on {self.cloud_type} cloud")
        gpu_type_id = str(entry.get("id") or spec.sku)
        ids: list[str] = []
        for i in range(spec.count):
            body = runpod_create_body(
                spec, gpu_type_id, name=f"{spec.name_prefix}-{i}",
                cloud_type=self.cloud_type, disk_gb=self.disk_gb)
            resp = self._post("/pods", body)
            pid = resp.get("id") if isinstance(resp, dict) else None
            if not pid:
                raise ProvisionError(f"runpod create returned no id: {resp}")
            log.info("runpod create → %s (%s × %s, %s)", pid, spec.gpus_per_pod,
                     gpu_type_id, self.cloud_type)
            ids.append(str(pid))
        return ids

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        """Ready = RUNNING **and** the container's port-22 mapping resolves.

        Status alone is not readiness: RunPod reports RUNNING while the image
        is still pulling, and probing then reaches nothing. The mapping only
        appears once the container runs, so it is the honest gate.
        """
        deadline = self._now() + timeout
        last = None
        while self._now() < deadline:
            info = self._get(f"/pods/{pod_id}")
            status = str(info.get("desiredStatus") or "").upper()
            if status in RUNPOD_TERMINAL_BAD:
                raise ProvisionError(f"runpod pod {pod_id} entered {status!r}")
            if status == RUNPOD_READY and runpod_pod_address(info) is not None:
                return True
            if status != last:
                log.info("runpod %s status %s — waiting for %s + port map",
                         pod_id, status or "?", RUNPOD_READY)
                last = status
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        return runpod_pod_address(self._get(f"/pods/{pod_id}"))

    def launched_image_digest(self, pod_id: str) -> str:
        """The digest RunPod's own record says this pod runs (best-effort)."""
        try:
            info = self._get(f"/pods/{pod_id}")
        except Exception:  # noqa: BLE001 — attestation is best-effort, gate decides
            return ""
        return image_digest_of(str(info.get("image") or info.get("imageName") or ""))

    def terminate(self, pod_id: str) -> None:
        try:
            self._request("DELETE", f"/pods/{pod_id}")
            log.info("runpod delete %s", pod_id)
        except Exception as e:  # noqa: BLE001 — idempotent teardown, already-gone is fine
            log.warning("runpod delete %s: %s (treating as already terminated)", pod_id, e)

    def list_tagged(self, prefix: str) -> list[str]:
        """Live pod IDs whose ``name`` starts with ``prefix`` (delete takes the id)."""
        pods = self._get("/pods")
        if isinstance(pods, dict):
            pods = pods.get("pods", [])
        return filter_tagged_names(pods or [], prefix, id_key="id")

    def offer_price(self, sku: str, *, gpus: int = 1) -> float | None:
        """PER-POD USD/hr — RunPod quotes per GPU, so this multiplies by shape."""
        return runpod_gpu_price_usd_hr(self._gpu_types(), sku, gpus=gpus,
                                       secure=self._secure)


@dataclass
class VastProvider(_BearerRestMixin):
    """Vast.ai marketplace via its REST API (``$VAST_API_KEY``).

    Vast is the price floor for consumer silicon and the most heterogeneous
    supply in the ladder, which is a genuine hazard for the HEAT: the screen
    ranks runs across pods, so machine-to-machine spread becomes rank spread
    (DEC-CA-0010). The quality floors below are therefore part of the
    adapter's contract, not tuning — ``verified_only`` keeps rentals in
    datacenter-verified hosts, ``min_reliability`` drops flaky machines, and
    ``min_cpu_cores_per_gpu`` guards the CPU generator that actually feeds
    training. Loosen them and the heat inherits the variance.

    Not recommended for ``final``: the duel wants one multi-GPU box from a
    single operator, which is what runpod/shadeform sell and Vast does not
    guarantee.
    """

    name: str = "vast"
    base_url: str = "https://console.vast.ai/api/v0"
    api_key_env: str = "VAST_API_KEY"
    disk_gb: int = DEFAULT_VAST_DISK_GB
    min_reliability: float = DEFAULT_VAST_MIN_RELIABILITY
    min_cpu_cores_per_gpu: float = DEFAULT_VAST_MIN_CPU_CORES_PER_GPU
    verified_only: bool = True
    poll_interval: float = POD_POLL_INTERVAL
    _session: object | None = field(default=None, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    # pod id → machine id, remembered at launch so lemon exclusion works even
    # after the instance is gone from /instances (the lookup it would need).
    _machines: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def _floors(self) -> dict:
        return {"min_reliability": self.min_reliability,
                "min_cpu_cores_per_gpu": self.min_cpu_cores_per_gpu,
                "verified_only": self.verified_only}

    def _bundles(self, sku: str, *, gpus: int) -> dict:
        q = vast_offer_query(sku, gpus=gpus, min_reliability=self.min_reliability,
                             verified_only=self.verified_only)
        out = self._get("/bundles/", {"q": json.dumps(q)})
        return out if isinstance(out, dict) else {"offers": []}

    def _qualifying(self, sku: str, *, gpus: int,
                    exclude: Sequence[str] = ()) -> list[dict]:
        return filter_vast_offers(self._bundles(sku, gpus=gpus), sku, gpus=gpus,
                                  exclude_machines=exclude, **self._floors)

    def available(self, sku: str, count: int, *, gpus: int = 1) -> bool:
        """Vast lists real per-machine offers, so this counts DISTINCT boxes."""
        return len(self._qualifying(sku, gpus=gpus)) >= count

    def launch(self, spec: LaunchSpec) -> list[str]:
        offers = self._qualifying(spec.sku, gpus=spec.gpus_per_pod,
                                  exclude=spec.exclude_ids)
        if len(offers) < spec.count:
            raise ProvisionError(
                f"vast: only {len(offers)} qualifying {spec.gpus_per_pod}×{spec.sku} "
                f"offer(s)"
                f"{' after exclusions' if spec.exclude_ids else ''}, need {spec.count}")
        ids: list[str] = []
        for i, offer in enumerate(offers[:spec.count]):
            body = vast_create_body(spec, label=f"{spec.name_prefix}-{i}",
                                    disk_gb=self.disk_gb)
            resp = self._request("PUT", f"/asks/{offer['id']}/", body=body)
            contract = resp.get("new_contract") if isinstance(resp, dict) else None
            if not resp.get("success", False) or not contract:
                raise ProvisionError(f"vast create on ask {offer['id']} failed: {resp}")
            log.info("vast rent → %s (machine %s, %s, $%.3f/hr, reliability %.3f)",
                     contract, offer.get("machine_id"), offer.get("geolocation"),
                     float(offer.get("dph_total", 0.0)),
                     float(offer.get("reliability2", 0.0)))
            ids.append(str(contract))
            self._machines[str(contract)] = str(offer.get("machine_id", ""))
        return ids

    def _instance(self, pod_id: str) -> dict:
        out = self._get(f"/instances/{pod_id}/")
        inst = out.get("instances") if isinstance(out, dict) else None
        if isinstance(inst, list):
            return inst[0] if inst else {}
        return inst if isinstance(inst, dict) else (out if isinstance(out, dict) else {})

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        last = None
        while self._now() < deadline:
            inst = self._instance(pod_id)
            status = str(inst.get("actual_status") or "").lower()
            if status in VAST_TERMINAL_BAD:
                raise ProvisionError(
                    f"vast instance {pod_id} entered {status!r}: "
                    f"{inst.get('status_msg') or 'no reason given'}")
            if status == VAST_READY and vast_pod_address(inst) is not None:
                return True
            if status != last:
                log.info("vast %s status %s — waiting for %s + address",
                         pod_id, status or "?", VAST_READY)
                last = status
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        return vast_pod_address(self._instance(pod_id))

    def machine_of(self, pod_id: str) -> str | None:
        """The physical machine behind a pod — the lemon-exclusion key.

        Vast DOES support this (unlike runpod): launches pick an explicit
        offer, so a machine that failed its boot gate can be excluded from the
        replacement's offer list instead of being re-rented deterministically.
        """
        cached = self._machines.get(str(pod_id))
        if cached:
            return cached
        mid = str(self._instance(pod_id).get("machine_id") or "")
        return mid or None

    def launched_image_digest(self, pod_id: str) -> str:
        try:
            inst = self._instance(pod_id)
        except Exception:  # noqa: BLE001 — attestation is best-effort, gate decides
            return ""
        return image_digest_of(str(inst.get("image") or inst.get("image_uuid") or ""))

    def terminate(self, pod_id: str) -> None:
        try:
            self._request("DELETE", f"/instances/{pod_id}/")
            log.info("vast destroy %s", pod_id)
        except Exception as e:  # noqa: BLE001 — idempotent teardown, already-gone is fine
            log.warning("vast destroy %s: %s (treating as already terminated)", pod_id, e)
        finally:
            self._machines.pop(str(pod_id), None)

    def list_tagged(self, prefix: str) -> list[str]:
        """Live instance IDs whose ``label`` starts with ``prefix``.

        Vast names the field ``label``; normalize to ``name`` so the shared
        reconcile primitive stays the one implementation.
        """
        out = self._get("/instances/")
        rows = out.get("instances", []) if isinstance(out, dict) else (out or [])
        return filter_tagged_names(
            [{"name": r.get("label"), "id": r.get("id")} for r in rows if isinstance(r, dict)],
            prefix, id_key="id")

    def offer_price(self, sku: str, *, gpus: int = 1) -> float | None:
        """Cheapest QUALIFYING per-pod USD/hr (``dph_total`` covers the box)."""
        return vast_offer_price_usd_hr(self._bundles(sku, gpus=gpus), sku, gpus=gpus,
                                       **self._floors)


_PROVIDER_FACTORIES: dict[str, Callable[[], Provider]] = {
    "lium": LiumProvider,
    "shadeform": ShadeformProvider,
    "runpod": RunPodProvider,
    "vast": VastProvider,
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
    p.add_argument("--stage", default="any", choices=("any", "heat", "final"),
                   help="Round stage these pods serve (default any). Heats can be a cheap "
                        "SKU; finals must be a single SKU (validator gpu_name gate). Provision "
                        "each stage as a separate run and concatenate the hosts.toml blocks.")
    p.add_argument("--name-prefix", default="cascade-pod",
                   help="hosts.toml [[host]] name prefix (default cascade-pod); e.g. "
                        "'cascade-heat' / 'cascade-final' to tell the fleets apart.")
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
