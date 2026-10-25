"""Targon CPU VMs as a generation-tier provider (DEC-CA-0053).

Targon's ``tha/v3`` REST API (docs.targon.com/api/workloads, /api/inventory,
/api/ssh-keys; read 2026-10-18):

* inventory — ``GET /tha/v3/inventory?type=vm``, no auth. VERIFIED live
  2026-10-18: CPU items are ``cpu-{small,medium,large,xlarge}-amd`` with
  ``gpu: false``, ``spec.vcpu`` in MILLICORES (32000 = 32 vCPU),
  ``spec.memory`` in MiB, ``cost_per_hour`` in USD and ``available`` units.
* VMs — workloads with ``type: "VM"`` under ``/tha/v3/orgs/{org}/workloads``:
  create (registers only) → ``POST …/{uid}/deploy`` (starts it) → poll
  ``GET …/{uid}/state`` (``status`` ``running``, ``public_ip``, ``ssh_port``)
  → ``DELETE …/{uid}`` (204). SSH user is always ``ubuntu``; the port is
  assigned, not 22. Auth ``Authorization: Bearer <token>``.
* SSH keys — ``POST /tha/v3/orgs/{org}/ssh-keys`` ``{name, ssh_key}`` → ``uid``,
  listed on the workload's ``ssh_keys``.

UNVERIFIED (documented, never exercised — no account on this box): every
org-scoped call above, the VM image names (``GET …/workloads/vm-images``),
and the exact state payload shape (read from the top level and from
``state``). Token from ``$TARGON_API_KEY``, org slug from ``$TARGON_ORG`` —
both read lazily, so listing offers needs neither.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .core import POD_POLL_INTERVAL, PodAddress, ProvisionError
from .cpu import CpuShape

log = logging.getLogger("cascade.provision.targon")

TARGON_API_BASE = "https://api.targon.com"
INVENTORY_PATH = "/tha/v3/inventory"
WORKLOADS_PATH = "/tha/v3/orgs/{org}/workloads"
SSH_KEYS_PATH = "/tha/v3/orgs/{org}/ssh-keys"

TARGON_READY = "running"
TARGON_TERMINAL_BAD = {"error", "deleted", "suspended", "stopped"}

# Fallback when an inventory item lacks a usable spec: the live CPU price list
# (owner-supplied and inventory-confirmed, 2026-10-18). (vCPU, GiB, USD/h).
STATIC_CPU_SHAPES: dict[str, tuple[int, float, float]] = {
    "cpu-small": (4, 52.0, 0.15),
    "cpu-medium": (8, 104.0, 0.30),
    "cpu-large": (16, 209.0, 0.60),
    "cpu-xlarge": (32, 419.0, 1.00),
}


def _static_shape(name: str) -> tuple[int, float, float] | None:
    for prefix, shape in STATIC_CPU_SHAPES.items():
        if name == prefix or name.startswith(prefix + "-"):
            return shape
    return None


def parse_cpu_inventory(items: list, min_vcpus: int, max_price: float) -> list[CpuShape]:
    """CPU shapes from ``GET /tha/v3/inventory`` with at least ``min_vcpus``
    vCPUs, at most ``max_price`` USD/h and at least one unit available,
    cheapest first. ``spec.vcpu`` is millicores; a missing spec falls back to
    :data:`STATIC_CPU_SHAPES` by resource name."""
    out: list[CpuShape] = []
    for it in items or []:
        if not isinstance(it, dict) or it.get("gpu") or it.get("type", "vm") != "vm":
            continue
        name = str(it.get("name") or "")
        if not name.startswith("cpu-") or int(it.get("available", 0) or 0) <= 0:
            continue
        spec = it.get("spec") if isinstance(it.get("spec"), dict) else {}
        milli = int(spec.get("vcpu") or spec.get("cpu_millicores") or 0)
        mem_mib = float(spec.get("memory") or spec.get("memory_mib") or 0)
        price = it.get("cost_per_hour")
        static = _static_shape(name)
        if (not milli or price is None) and static is None:
            continue
        vcpus = milli // 1000 if milli else static[0]           # type: ignore[index]
        memory_gb = mem_mib / 1024.0 if mem_mib else static[1]  # type: ignore[index]
        usd = float(price) if price is not None else static[2]  # type: ignore[index]
        if vcpus < int(min_vcpus) or usd > float(max_price):
            continue
        out.append(CpuShape(vcpus=vcpus, memory_gb=memory_gb, price_per_hour=usd,
                            provider="targon", type_id=name))
    out.sort(key=lambda s: (s.price_per_hour, s.vcpus))
    return out


def targon_vm_name(name: str) -> str:
    """A workload name Targon accepts: lowercase alphanumerics and hyphens,
    at most 32 characters."""
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (s[:32].rstrip("-")) or "cascade-gen"


def targon_vm_address(state: dict) -> PodAddress | None:
    """``(public_ip, ssh_port)`` from a ``GET …/state`` response — read at the
    top level, falling back to the nested ``state`` object."""
    inner = state.get("state") if isinstance(state.get("state"), dict) else {}
    ip = state.get("public_ip") or inner.get("public_ip")
    port = state.get("ssh_port") or inner.get("ssh_port")
    if not ip or not port:
        return None
    return PodAddress(ip=str(ip), ssh_port=int(port))


def _state_status(state: dict) -> str:
    inner = state.get("state") if isinstance(state.get("state"), dict) else {}
    return str(state.get("status") or inner.get("status") or "").lower()


@dataclass
class TargonCpuProvider:
    """Targon CPU VMs via the ``tha/v3`` REST API (see the module docstring
    for what is verified). ``vm_image`` is the VM image name to boot — the
    ``image`` launch argument overrides it."""

    name: str = "targon"
    ssh_user: str = "ubuntu"
    base_url: str = TARGON_API_BASE
    api_key_env: str = "TARGON_API_KEY"
    org_env: str = "TARGON_ORG"
    vm_image: str = ""
    poll_interval: float = POD_POLL_INTERVAL
    _session: object | None = field(default=None, repr=False)
    _public: object | None = field(default=None, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    _key_uids: dict = field(default_factory=dict, repr=False)

    def _requests(self):
        try:
            import requests
        except ModuleNotFoundError as e:  # pragma: no cover - env guard
            raise ProvisionError(
                "the targon adapter needs `requests` (pip install -e '.[deploy]')") from e
        return requests

    def _http(self):
        """The authenticated session (org-scoped calls)."""
        if self._session is not None:
            return self._session
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProvisionError(f"targon: set ${self.api_key_env}")
        s = self._requests().Session()
        s.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        self._session = s
        return s

    def _public_http(self):
        """An unauthenticated session — inventory needs no key."""
        if self._public is None:
            self._public = self._session or self._requests().Session()
        return self._public

    def _org(self) -> str:
        org = os.environ.get(self.org_env, "")
        if not org:
            raise ProvisionError(f"targon: set ${self.org_env} (the org slug)")
        return org

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _call(self, method: str, path: str, body: dict | None = None):
        r = getattr(self._http(), method)(self._url(path), json=body, timeout=30)
        r.raise_for_status()
        return r.json() if getattr(r, "content", b"") else {}

    def offers(self, min_vcpus: int, max_price: float) -> list[CpuShape]:
        r = self._public_http().get(self._url(INVENTORY_PATH), params={"type": "vm"},
                                    timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", []) if isinstance(data, dict) else data
        return parse_cpu_inventory(items, min_vcpus, max_price)

    def _ssh_key_uid(self, org: str, ssh_pubkey: str, name: str) -> str:
        uid = self._key_uids.get(ssh_pubkey)
        if uid:
            return uid
        resp = self._call("post", SSH_KEYS_PATH.format(org=org),
                          {"name": targon_vm_name(f"{name}-key"), "ssh_key": ssh_pubkey})
        uid = str(resp.get("uid") or "")
        if not uid:
            raise ProvisionError(f"targon ssh-key create returned no uid: {resp}")
        self._key_uids[ssh_pubkey] = uid
        return uid

    def launch(self, shape: CpuShape, *, name: str, ssh_pubkey: str, image: str = "") -> str:
        vm_image = image or self.vm_image
        if not vm_image:
            raise ProvisionError("targon: no VM image (set vm_image or pass image)")
        org = self._org()
        key_uid = self._ssh_key_uid(org, ssh_pubkey, name)
        base = WORKLOADS_PATH.format(org=org)
        resp = self._call("post", base, {
            "type": "VM", "name": targon_vm_name(name), "image": vm_image,
            "resource_name": shape.type_id, "ssh_keys": [key_uid],
        })
        uid = str(resp.get("uid") or "")
        if not uid:
            raise ProvisionError(f"targon workload create returned no uid: {resp}")
        try:
            self._call("post", f"{base}/{uid}/deploy")
        except Exception:
            self.terminate(uid)  # a registered-but-undeployed VM is never left behind
            raise
        log.info("targon create+deploy → %s (%s, %d vCPU, $%.2f/h)", uid, shape.type_id,
                 shape.vcpus, shape.price_per_hour)
        return uid

    def _state(self, vm_id: str) -> dict:
        return self._call("get", f"{WORKLOADS_PATH.format(org=self._org())}/{vm_id}/state")

    def find_by_name(self, name: str) -> list[str]:
        """Resolve a lost create response by exact VM name, across all pages.

        API contract: https://docs.targon.com/api/workloads#list-workloads
        An empty response does not prove that a timed-out create is absent.
        """
        base = WORKLOADS_PATH.format(org=self._org())
        cursor, seen, found = "", set(), set()
        for _ in range(100):
            params = {"type": "VM", "name": name, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            r = self._http().get(self._url(base), params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict) or not isinstance(data.get("items"), list):
                raise ProvisionError("targon: malformed workload listing")
            for item in data["items"]:
                if not isinstance(item, dict):
                    raise ProvisionError("targon: malformed workload item")
                if item.get("name") == name and item.get("type") == "VM":
                    uid = item.get("uid")
                    if not isinstance(uid, str) or not uid:
                        raise ProvisionError("targon: matching workload has no uid")
                    found.add(uid)
            cursor = data.get("next_cursor") or ""
            if not cursor:
                return sorted(found)
            if not isinstance(cursor, str) or cursor in seen:
                raise ProvisionError("targon: invalid/repeated workload cursor")
            seen.add(cursor)
        raise ProvisionError("targon: workload lookup exceeded 100 pages")

    def wait_ready(self, vm_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        while self._now() < deadline:
            state = self._state(vm_id)
            status = _state_status(state)
            if status in TARGON_TERMINAL_BAD:
                raise ProvisionError(f"targon VM {vm_id} entered {status!r}")
            if status == TARGON_READY and targon_vm_address(state) is not None:
                return True
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, vm_id: str) -> PodAddress | None:
        return targon_vm_address(self._state(vm_id))

    def terminate(self, vm_id: str, *, attempts: int = 3) -> None:
        """Delete the VM. 404/410 is "already gone"; any other failure is
        retried, then raised (the stage logs TERMINATE FAILED — billing runs)."""
        last: Exception | None = None
        for i in range(max(1, attempts)):
            try:
                self._call("delete", f"{WORKLOADS_PATH.format(org=self._org())}/{vm_id}")
                log.info("targon delete %s", vm_id)
                return
            except Exception as e:  # noqa: BLE001 — classified below
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (404, 410):
                    log.info("targon delete %s: already gone (HTTP %s)", vm_id, status)
                    return
                last = e
                log.warning("targon delete %s attempt %d failed: %s", vm_id, i + 1, e)
                if i + 1 < attempts:
                    self._sleep(self.poll_interval)
        raise ProvisionError(f"targon delete {vm_id} failed: {last}")
