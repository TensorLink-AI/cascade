"""RunPod adapter behind the :class:`~cascade.provision.core.Provider` seam.

Same contract as :class:`~cascade.provision.core.LiumProvider`: one instance per
payer key (DEC-CA-0036), the key travels only in the ``Authorization`` header,
pods are addressed by the name we assign, and the worker image's entrypoint
runs ``sshd`` with the orchestrator key from ``$SSH_PUBKEY``. Pod create/delete
needs a key with GraphQL/REST read-write; RunPod's "Restricted" keys can drop
every Serverless and AI-API scope (https://docs.runpod.io/get-started/api-keys).

REST surface (https://docs.runpod.io/api-reference): base ``rest.runpod.io/v1``,
``POST /pods`` create · ``GET /pods?name=`` list · ``DELETE /pods/{id}`` delete.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .core import (
    DEFAULT_SSH_PORT,
    POD_POLL_INTERVAL,
    LaunchSpec,
    PodAddress,
    PriceCapUnavailable,
    ProvisionError,
    UncertainCreateError,
    filter_tagged_names,
    image_digest_of,
    validate_digest_pinned,
)
from .vast import _requests_http

log = logging.getLogger("cascade.provision.runpod")

RUNPOD_API_BASE = "https://rest.runpod.io/v1"
RUNPOD_APPEAR_TIMEOUT = 300.0
DEFAULT_CONTAINER_DISK_GB = 200
# Marketplace SKU name → RunPod gpuTypeId.
RUNPOD_GPU_TYPES = {
    "H200": "NVIDIA H200",
    "H100": "NVIDIA H100 80GB HBM3",
    "A100": "NVIDIA A100-SXM4-80GB",
    "L40S": "NVIDIA L40S",
    "RTX 4090": "NVIDIA GeForce RTX 4090",
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "RTX3090": "NVIDIA GeForce RTX 3090",
    "A6000": "NVIDIA RTX A6000",
    "L40": "NVIDIA L40",
}


def runpod_gpu_type(sku: str) -> str:
    return RUNPOD_GPU_TYPES.get(sku, sku)


def runpod_pod_address(pod: dict, *, container_ssh_port: int = DEFAULT_SSH_PORT) -> PodAddress | None:
    ip = str(pod.get("publicIp") or "").strip()
    port = (pod.get("portMappings") or {}).get(str(container_ssh_port))
    if not ip or not port:
        return None
    try:
        return PodAddress(ip, int(port))
    except (TypeError, ValueError):
        return None


def runpod_pod_ready(pod: dict) -> bool:
    return str(pod.get("desiredStatus") or "") == "RUNNING" and runpod_pod_address(pod) is not None


@dataclass
class RunPodProvider:
    name: str = "runpod"
    api_key: str = field(default="", repr=False)
    base: str = RUNPOD_API_BASE
    container_disk_gb: int = DEFAULT_CONTAINER_DISK_GB
    cloud_type: str = "SECURE"
    poll_interval: float = POD_POLL_INTERVAL
    max_price_per_hour: float = 0.0
    max_leg_cost_usd: float = 0.0
    sku_wall_seconds: tuple[tuple[str, int], ...] = ()
    default_wall_seconds: float = 0.0
    _http: Callable = field(default=_requests_http, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    _sku_by_name: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProvisionError("runpod rental needs the payer's api key (empty)")
        if self.cloud_type not in ("SECURE", "COMMUNITY"):
            raise ProvisionError(f"unknown runpod cloud type {self.cloud_type!r}")

    def _scrub(self, text: str) -> str:
        return str(text).replace(self.api_key, "<redacted>") if self.api_key else str(text)

    def _call(self, method: str, path: str, body=None):
        try:
            status, data = self._http(method, f"{self.base}/{path.lstrip('/')}",
                                      headers={"Authorization": f"Bearer {self.api_key}",
                                               "Accept": "application/json"},
                                      json=body)
        except Exception as e:  # noqa: BLE001 — transport errors surface scrubbed
            error = UncertainCreateError if method == "POST" and path == "pods" else ProvisionError
            raise error(self._scrub(f"runpod {method} {path}: {e}")) from None
        if status >= 400:
            if method == "POST" and path == "pods" and (status >= 500 or status == 408):
                raise UncertainCreateError(self._scrub(f"runpod create unresolved ({status}): {data}"))
            raise ProvisionError(self._scrub(f"runpod {method} {path} failed ({status}): {data}"))
        return data

    def _check_price_caps(self):
        if any(not math.isfinite(v) or v < 0 for v in (self.max_price_per_hour, self.max_leg_cost_usd)):
            raise ProvisionError("runpod: invalid price ceilings")
        if self.max_price_per_hour > 0 or self.max_leg_cost_usd > 0:
            raise PriceCapUnavailable(
                "runpod: price_cap_unavailable: on-demand REST creates have no verified price-ceiling "
                "parameter; use another provider, or explicitly authorize an uncapped chosen GPU")

    def capacity(self, sku: str, *, gpus: int = 1, exclude_ids: tuple[str, ...] = ()) -> int:
        """RunPod exposes no reservable offer listing on the REST surface: a
        create is the capacity probe (it 4xx's "no instances available").
        Report one so callers attempt the create and classify its refusal."""
        self._check_price_caps()
        return 1

    def available(self, sku: str, count: int, *, gpus: int = 1,
                  exclude_ids: tuple[str, ...] = ()) -> bool:
        return self.capacity(sku, gpus=gpus, exclude_ids=exclude_ids) >= count

    def launch(self, spec: LaunchSpec) -> list[str]:
        self._check_price_caps()
        if spec.image:
            validate_digest_pinned(spec.image)
        env = {"SSH_PUBKEY": spec.ssh_pubkey}
        if spec.image:
            env["CASCADE_TRAIN_IMAGE_DIGEST"] = image_digest_of(spec.image)
        names = []
        for i in range(spec.count):
            name = f"{spec.name_prefix}-{i}"
            response = self._call("POST", "pods", {
                "name": name, "imageName": spec.image,
                "gpuTypeIds": [runpod_gpu_type(spec.sku)], "gpuCount": int(spec.gpus_per_pod),
                "cloudType": self.cloud_type, "interruptible": False,
                "containerDiskInGb": self.container_disk_gb,
                "ports": [f"{spec.ssh_port}/tcp"], "env": env})
            if not isinstance(response, dict) or not isinstance(response.get("id"), str) or not response["id"]:
                raise UncertainCreateError("runpod create returned no allocation identity; reconcile before retry")
            log.info("runpod create %s (%dx%s)", name, spec.gpus_per_pod, spec.sku)
            self._sku_by_name[name] = spec.sku
            names.append(name)
        return names

    def _pods(self, name: str | None = None) -> list[dict]:
        data = self._call("GET", "pods" + (f"?name={name}" if name else ""))
        if not isinstance(data, list) or any(
            not isinstance(p, dict) or not isinstance(p.get("id"), str) or not p["id"]
            or not isinstance(p.get("name"), str) or not isinstance(p.get("desiredStatus"), str)
            for p in data):
            raise ProvisionError("runpod: malformed pod listing; absence is not established")
        return data

    def _named(self, pod_id: str) -> list[dict]:
        return [p for p in self._pods() if pod_id in (p.get("name"), p.get("id"))
                and p.get("desiredStatus") != "TERMINATED"]

    def _pod(self, pod_id: str) -> dict | None:
        hits = self._named(pod_id)
        if not hits:
            return None
        return next((p for p in hits if runpod_pod_ready(p)), hits[0])

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        appear_by = self._now() + min(RUNPOD_APPEAR_TIMEOUT, timeout)
        seen = False
        while self._now() < deadline:
            pod = self._pod(pod_id)
            if pod:
                seen = True
                if runpod_pod_ready(pod):
                    return True
            elif not seen and self._now() >= appear_by:
                return False
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        pod = self._pod(pod_id)
        return runpod_pod_address(pod) if pod else None

    def live_pod_address(self, pod_id: str) -> PodAddress | None:
        pod = self._pod(pod_id)
        return runpod_pod_address(pod) if pod and runpod_pod_ready(pod) else None

    def pod_identity(self, pod_id: str) -> dict | None:
        pod = self._pod(pod_id)
        if not pod:
            return None
        addr = runpod_pod_address(pod)
        gpu = pod.get("gpu") or {}
        return {"id": str(pod.get("id") or ""), "huid": str((pod.get("machine") or {}).get("dataCenterId") or ""),
                "status": str(pod.get("desiredStatus") or ""),
                "gpus": int(gpu.get("count") or 0), "gpu_name": str(gpu.get("displayName") or ""),
                "ip": addr.ip if addr else "", "port": addr.ssh_port if addr else 0}

    def terminate(self, pod_id: str) -> None:
        for pod in self._named(pod_id):
            try:
                self._call("DELETE", f"pods/{pod['id']}")
                log.info("runpod delete %s (%s)", pod_id, pod["id"])
            except ProvisionError as e:
                log.warning("runpod delete %s: %s (treating as already terminated)", pod_id, e)

    def list_tagged(self, prefix: str) -> list[str]:
        live = [p for p in self._pods() if p.get("desiredStatus") != "TERMINATED"]
        return filter_tagged_names(live, prefix)

    def machine_of(self, pod_id: str) -> str | None:
        return None

    def sku_of(self, pod_id: str) -> str:
        cached = self._sku_by_name.get(pod_id)
        if cached:
            return cached
        pod = self._pod(pod_id) or {}
        gpu = pod.get("gpu") or {}
        names = {str(pod.get("gpuTypeId") or ""), str(gpu.get("id") or ""),
                 str(gpu.get("displayName") or "")}
        for sku, name in RUNPOD_GPU_TYPES.items():
            if name in names and " " not in sku:
                return sku
        raise ProvisionError("runpod: allocated GPU type is unavailable; cannot adopt unpinned hardware")
