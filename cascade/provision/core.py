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


def shadeform_create_body(spec: LaunchSpec, offer: dict, *, name: str) -> dict:
    """Build the ``POST /instances/create`` body for one pod of ``spec``.

    Only ``SSH_PUBKEY`` is injected into the container env (the image contract);
    Hippius creds are never placed on the pod. Container port 22 is exposed so
    the orchestrator can SSH in.
    """
    return {
        "cloud": offer["cloud"],
        "region": offer["region"],
        "shade_instance_type": offer["shade_instance_type"],
        "shade_cloud": True,
        "name": name,
        "launch_configuration": {
            "type": "docker",
            "docker_configuration": {
                "image": spec.image,
                "envs": [{"name": "SSH_PUBKEY", "value": spec.ssh_pubkey}],
                "port_mappings": [
                    {"host_port": spec.ssh_port, "container_port": DEFAULT_SSH_PORT}
                ],
            },
        },
    }


def shadeform_pod_address(info_json: dict, *, ssh_port: int = DEFAULT_SSH_PORT) -> PodAddress | None:
    """Extract ``(ip, port)`` from a Shadeform ``/instances/{id}/info`` response.

    The container's sshd is reached at the mapped ``host_port`` (we expose 22),
    so we pair the reported ``ip`` with the port we mapped rather than the host
    box's own ``ssh_port``.
    """
    ip = info_json.get("ip")
    if not ip:
        return None
    return PodAddress(ip=str(ip), ssh_port=ssh_port)


SHADEFORM_READY = "active"                  # the "running/live" status (no "running" state)
SHADEFORM_TERMINAL_BAD = {"error", "deleting", "deleted"}


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


def _spawn_cli(argv: list[str]) -> subprocess.Popen:
    """Fire-and-forget a CLI command (e.g. ``lium up``, which attaches/streams).

    We don't wait on it — pod readiness is observed via ``lium ps`` polling, and
    killing the local client does not stop the remote pod.
    """
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


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
        if len(execs) < spec.count:
            raise ProvisionError(
                f"lium: only {len(execs)} × {spec.gpus_per_pod}x{spec.sku} available, "
                f"need {spec.count}"
            )
        spawn = self._spawn or _spawn_cli
        names: list[str] = []
        for i, ex in enumerate(execs[: spec.count]):
            name = f"{spec.name_prefix}-{i}"
            spawn([
                self.bin, "up", str(ex["id"]),
                "--image", spec.image,
                "-e", f"SSH_PUBKEY={spec.ssh_pubkey}",
                "--internal-ports", str(spec.ssh_port),
                "--name", name, "--yes",
            ])
            log.info("lium up → executor %s as %s", ex.get("id"), name)
            names.append(name)
        return names

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        while self._now() < deadline:
            pod = self._pod(pod_id)
            if pod and lium_pod_ready(pod):
                return True
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
            body = shadeform_create_body(spec, offer, name=f"{spec.name_prefix}-{i}")
            resp = self._post("/instances/create", body)
            iid = resp.get("id")
            if not iid:
                raise ProvisionError(f"shadeform create returned no id: {resp}")
            log.info("shadeform create → %s (%s/%s)", iid, offer["cloud"], offer["region"])
            ids.append(str(iid))
        return ids

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        while self._now() < deadline:
            info = self._get(f"/instances/{pod_id}/info")
            status = str(info.get("status", "")).lower()
            if status == SHADEFORM_READY:
                return True
            if status in SHADEFORM_TERMINAL_BAD:
                raise ProvisionError(f"shadeform instance {pod_id} entered {status!r}")
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        return shadeform_pod_address(self._get(f"/instances/{pod_id}/info"))

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

    def offer_price(self, sku: str) -> float | None:
        """Cheapest available USD/hr for ``sku`` (the budget breaker's input)."""
        types = self._get("/instances/types", {"gpu_type": sku, "available": "true"})
        return shadeform_offer_price_usd_hr(types, sku)


_PROVIDER_FACTORIES: dict[str, Callable[[], Provider]] = {
    "lium": LiumProvider,
    "shadeform": ShadeformProvider,
    # "targon": TargonProvider,  # seam: add for SKUs where Targon has capacity.
}


def build_providers(priority: Sequence[str]) -> list[Provider]:
    """Instantiate the requested providers in priority order."""
    out: list[Provider] = []
    for name in priority:
        factory = _PROVIDER_FACTORIES.get(name)
        if factory is None:
            raise ProvisionError(
                f"unknown provider {name!r}; known: {', '.join(sorted(_PROVIDER_FACTORIES))}"
            )
        out.append(factory())
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
