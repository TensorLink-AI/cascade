"""Vast.ai marketplace adapter behind the :class:`~cascade.provision.core.Provider` seam.

Same contract as :class:`~cascade.provision.core.LiumProvider`: one instance per
payer key (DEC-CA-0036), the key travels only in the ``Authorization`` header
(never argv, never logged), pods are addressed by the label we assign, and the
worker image's entrypoint runs ``sshd`` with the orchestrator key from
``$SSH_PUBKEY``. A miner should fund with a scoped key holding only the
``instance_read``/``instance_write``/``misc`` categories
(https://docs.vast.ai/api-reference/permissions) — no billing or user scope.

REST surface (https://docs.vast.ai/api-reference; request shapes follow the
official ``vast-ai/vast-cli``): ``POST /api/v0/bundles/`` search offers ·
``PUT /api/v0/asks/{offer}/`` create · ``GET /api/v1/instances/`` list (paged;
the v0 list is deprecated) · ``DELETE /api/v0/instances/{id}/`` destroy.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import quote_plus

from ..funding.faults import classify_rent_failure
from .core import (
    DEFAULT_SSH_PORT,
    POD_POLL_INTERVAL,
    LaunchSpec,
    PodAddress,
    ProvisionError,
    UncertainCreateError,
    filter_tagged_names,
    image_digest_of,
    quarantined_hosts,
    validate_digest_pinned,
)

log = logging.getLogger("cascade.provision.vast")

VAST_API_ROOT = "https://console.vast.ai"
VAST_API_BASE = VAST_API_ROOT + "/api/v0"
VAST_APPEAR_TIMEOUT = 300.0
DEFAULT_DISK_GB = 200
VAST_PAGE_SIZE = 25
VAST_MAX_PAGES = 400
# Cascade GPU type -> Vast ``gpu_name`` values (spaces: the CLI turns RTX_4090
# into "RTX 4090" before it queries). A generic type matches its variants.
VAST_GPU_NAMES = {
    "RTX4090": ("RTX 4090",),
    "RTX3090": ("RTX 3090",),
    "RTX 4090": ("RTX 4090",),
    "A6000": ("RTX A6000",),
    "L40S": ("L40S",),
    "L40": ("L40",),
    "H100": ("H100 SXM", "H100 PCIE", "H100 NVL"),
    "H200": ("H200",),
    "A100": ("A100 SXM4", "A100 PCIE"),
}


def _requests_http(method: str, url: str, *, headers: dict, json=None, timeout: float = 30.0):
    import requests

    r = requests.request(method, url, headers=headers, json=json, timeout=timeout)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[-500:]}
    return r.status_code, body


def vast_gpu_names(sku: str) -> tuple[str, ...]:
    """Vast ``gpu_name`` values for a Cascade GPU type (``RTX4090`` -> ``RTX 4090``)."""
    return VAST_GPU_NAMES.get(sku, (sku.replace("_", " "),))


def vast_sku_of_gpu(gpu_name: str) -> str:
    """The Cascade GPU type a Vast ``gpu_name`` belongs to ("" when unknown)."""
    for sku, names in VAST_GPU_NAMES.items():
        if gpu_name in names and " " not in sku:
            return sku
    return ""


def vast_offer_price(offer: dict) -> float | None:
    """USD/h of an offer; None unless it is a finite number above zero (a
    zero, bool or non-finite ``dph_total`` is not a price — it would sort
    first and dodge every cap, same rule as Lium's ``executor_price_per_hour``)."""
    raw = offer.get("dph_total")
    if isinstance(raw, bool):
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 0 else None


def _excluded(offer: dict, exclude_ids) -> bool:
    machine = str(offer.get("machine_id"))
    return machine in exclude_ids or f"vast:{machine}" in exclude_ids


class VastCreateRefused(ProvisionError):
    """The API answered the create with a refusal: nothing was allocated."""


def vast_instance_ready(inst: dict) -> bool:
    return (str(inst.get("actual_status") or "") == "running"
            and vast_instance_address(inst) is not None)


def vast_instance_address(inst: dict, *, container_ssh_port: int = DEFAULT_SSH_PORT) -> PodAddress | None:
    ip = str(inst.get("public_ipaddr") or "").strip()
    mapped = (inst.get("ports") or {}).get(f"{container_ssh_port}/tcp") or []
    if not ip or not mapped:
        return None
    try:
        return PodAddress(ip, int(mapped[0]["HostPort"]))
    except (KeyError, TypeError, ValueError):
        return None


@dataclass
class VastProvider:
    name: str = "vast"
    api_key: str = field(default="", repr=False)
    base: str = VAST_API_BASE
    disk_gb: int = DEFAULT_DISK_GB
    poll_interval: float = POD_POLL_INTERVAL
    # Price guards (the [round] funded caps, applied by apply_price_caps): USD/h
    # and USD per leg over the GPU type's training wall. 0 = off.
    max_price_per_hour: float = 0.0
    max_leg_cost_usd: float = 0.0
    sku_wall_seconds: tuple[tuple[str, int], ...] = ()
    default_wall_seconds: float = 0.0
    _http: Callable = field(default=_requests_http, repr=False)
    _sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _now: Callable[[], float] = field(default=time.monotonic, repr=False)
    _machine_by_name: dict = field(default_factory=dict, repr=False)
    _sku_by_name: dict = field(default_factory=dict, repr=False)
    _contract_by_name: dict = field(default_factory=dict, repr=False)
    # Evidence this instance has gathered about the account, for
    # ``create_requested`` / ``create_unresolved`` (see those methods): labels
    # and ids ever seen in a listing, labels whose DELETE the API acknowledged,
    # and labels whose create request went out and was not definitely refused.
    _observed: set = field(default_factory=set, repr=False)
    _deleted: set = field(default_factory=set, repr=False)
    _issued_creates: set = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ProvisionError("vast rental needs the payer's api key (empty)")

    def _scrub(self, text: str) -> str:
        return str(text).replace(self.api_key, "<redacted>") if self.api_key else str(text)

    def _request(self, method: str, url: str, body=None):
        try:
            return self._http(method, url,
                              headers={"Authorization": f"Bearer {self.api_key}",
                                       "Accept": "application/json"},
                              json=body)
        except Exception as e:  # noqa: BLE001 — transport errors surface scrubbed
            raise ProvisionError(self._scrub(f"vast {method} {url}: {e}")) from None

    def _call(self, method: str, path: str, body=None):
        status, data = self._request(method, f"{self.base}/{path.lstrip('/')}", body)
        if status >= 400:
            raise ProvisionError(self._scrub(f"vast {method} {path} failed ({status}): {data}"))
        return data

    def _price_cap(self, sku: str) -> float:
        limits = [self.max_price_per_hour] if self.max_price_per_hour > 0 else []
        if self.max_leg_cost_usd > 0:
            wall = dict(self.sku_wall_seconds).get(sku, self.default_wall_seconds)
            if not wall or wall <= 0:
                raise ProvisionError("vast: per-leg cap requires a positive training wall")
            limits.append(self.max_leg_cost_usd * 3600.0 / wall)
        return min(limits) if limits else math.inf

    # ── offers ───────────────────────────────────────────────────────────────

    def _offers(self, sku: str, *, gpus: int = 1) -> list[dict]:
        names = vast_gpu_names(sku)
        cap = self._price_cap(sku)
        # ``direct_port_count``: the worker is reached over a direct port
        # mapping of the container's sshd; a host with none accepts the create,
        # reaches ``running`` and never maps 22/tcp — billed for the whole
        # ready timeout, then destroyed as a lemon. ``type`` is the CLI's
        # spelling ("on-demand").
        query = {"gpu_name": {"eq": names[0]} if len(names) == 1 else {"in": list(names)},
                 "num_gpus": {"eq": int(gpus)},
                 "rentable": {"eq": True}, "rented": {"eq": False},
                 "verified": {"eq": True}, "external": {"eq": False}, "type": "on-demand",
                 "direct_port_count": {"gte": 1},
                 "allocated_storage": self.disk_gb, "order": [["dph_total", "asc"]],
                 "limit": 64}
        if math.isfinite(cap):
            query["dph_total"] = {"lte": cap}
        offers = self._call("POST", "bundles/", query).get("offers") or []
        if isinstance(offers, dict):
            offers = [offers]
        # The server filter is not trusted alone: an unpriced or non-finite
        # offer never passes a cap (and never sorts first). A machine under
        # host quarantine (a lemon this deployment just tore down, keyed
        # ``vast:<machine_id>`` by ``host_of_pod``) is skipped whatever offer
        # id it is listed under now.
        quarantined = quarantined_hosts()
        priced = [(price, o) for o in offers
                  if isinstance(o, dict) and int(o.get("num_gpus") or 0) == int(gpus)
                  and (price := vast_offer_price(o)) is not None and price <= cap
                  and f"vast:{o.get('machine_id')}" not in quarantined]
        return [o for _, o in sorted(priced, key=lambda row: row[0])]

    def capacity(self, sku: str, *, gpus: int = 1, exclude_ids: tuple[str, ...] = ()) -> int:
        return len([o for o in self._offers(sku, gpus=gpus) if not _excluded(o, exclude_ids)])

    def available(self, sku: str, count: int, *, gpus: int = 1,
                  exclude_ids: tuple[str, ...] = ()) -> bool:
        return self.capacity(sku, gpus=gpus, exclude_ids=exclude_ids) >= count

    # ── lifecycle ────────────────────────────────────────────────────────────

    def launch(self, spec: LaunchSpec) -> list[str]:
        if spec.image:
            validate_digest_pinned(spec.image)
        offers = sorted(
            ((sku, o) for sku in (spec.sku_choices or (spec.sku,))
             for o in self._offers(sku, gpus=spec.gpus_per_pod)
             if not _excluded(o, spec.exclude_ids)),
            key=lambda row: vast_offer_price(row[1]))
        if len(offers) < spec.count:
            raise ProvisionError(f"vast: no_capacity: only {len(offers)} × "
                                 f"{spec.gpus_per_pod}x{spec.sku} available, need {spec.count}")
        env = {"SSH_PUBKEY": spec.ssh_pubkey, f"-p {spec.ssh_port}:{spec.ssh_port}": "1"}
        if spec.image:
            env["CASCADE_TRAIN_IMAGE_DIGEST"] = image_digest_of(spec.image)
        names = []
        for i, (sku, offer) in enumerate(offers[: spec.count]):
            name = f"{spec.name_prefix}-{i}"
            # runtype "args": the image's own entrypoint (sshd + $SSH_PUBKEY) runs,
            # exactly as on a Lium template; Vast injects no account keys.
            # From here the create may have reached the API: ``create_requested``
            # answers True until a definite refusal proves nothing was allocated.
            self._issued_creates.add(name)
            try:
                status, data = self._request("PUT", f"{self.base}/asks/{offer['id']}/", {
                    "client_id": "me", "image": spec.image, "label": name, "disk": self.disk_gb,
                    "runtype": "args", "env": env, "target_state": "running",
                    "cancel_unavail": True})
            except ProvisionError as exc:
                raise UncertainCreateError(str(exc)) from None
            if status in (401, 402, 403, 429):
                self._issued_creates.discard(name)
                raise ProvisionError(self._scrub(f"vast PUT asks/{offer['id']}/ failed ({status}): {data}"))
            contract = data.get("new_contract") if isinstance(data, dict) else None
            refused = (400 <= status < 500 and status not in (408, 425)) or (
                200 <= status < 300 and isinstance(data, dict) and data.get("success") is False)
            if refused and contract is None:
                self._issued_creates.discard(name)
                detail = self._scrub(f"offer {offer.get('id')} refused ({status}): {data}")
                # The body is classified BEFORE the capacity prefix: a refusal
                # for the PAYER's balance ("Insufficient credit") must reach the
                # taxonomy as auth — the key is released to the miner — not wait
                # forever as sold-out. Anything else a refusal says is the
                # offer going away under us (no_such_ask), i.e. capacity.
                if classify_rent_failure(detail) in ("auth", "rate_limited"):
                    raise VastCreateRefused(f"vast: {detail}")
                raise VastCreateRefused(f"vast: no_capacity: {detail}")
            if (not 200 <= status < 300 or not isinstance(data, dict)
                    or data.get("success") is not True
                    or isinstance(contract, bool) or not str(contract).isdigit() or int(contract) <= 0):
                raise UncertainCreateError(self._scrub(
                    f"vast create outcome unresolved ({status}): {data}"))
            log.info("vast create → offer %s as %s (%s, $%.2f/h)", offer.get("id"), name,
                     sku, vast_offer_price(offer))
            self._machine_by_name[name] = str(offer.get("machine_id") or "")
            self._sku_by_name[name] = sku
            if data.get("new_contract") is not None:
                self._contract_by_name[name] = str(data["new_contract"])
            names.append(name)
        return names

    def _instances(self) -> list[dict]:
        """Every instance on the account, through the paged v1 list."""
        rows: list[dict] = []
        seen_ids, seen_tokens = set(), set()
        expected_total = None
        params: dict = {"select_filters": {}, "order_by": [{"col": "id", "dir": "asc"}],
                        "limit": VAST_PAGE_SIZE}
        root = self.base.rsplit("/api/", 1)[0]
        for _ in range(VAST_MAX_PAGES):
            query = "&".join(f"{k}={quote_plus(v if isinstance(v, str) else json.dumps(v))}"
                             for k, v in params.items())
            status, data = self._request("GET", f"{root}/api/v1/instances/?{query}")
            if status >= 400:
                raise ProvisionError(self._scrub(f"vast GET instances failed ({status}): {data}"))
            if (status != 200 or not isinstance(data, dict) or data.get("success") is not True
                    or not isinstance(data.get("instances"), list)
                    or type(data.get("instances_found")) is not int
                    or type(data.get("total_instances")) is not int
                    or "next_token" not in data):
                raise ProvisionError("vast: malformed instance listing")
            page = data["instances"]
            total = data["total_instances"]
            if (data["instances_found"] != len(page) or total < 0
                    or expected_total is not None and total != expected_total):
                raise ProvisionError("vast: inconsistent instance listing counts")
            expected_total = total
            for row in page:
                if (not isinstance(row, dict) or isinstance(row.get("id"), bool)
                        or not str(row.get("id")).isdigit() or int(row["id"]) <= 0
                        or "label" not in row or row["label"] is not None
                        and not isinstance(row["label"], str)):
                    raise ProvisionError("vast: malformed instance listing row")
                identity = str(row["id"])
                if identity in seen_ids:
                    raise ProvisionError("vast: duplicate instance in paged listing")
                seen_ids.add(identity)
                self._observed.add(identity)
                if row["label"]:
                    self._observed.add(row["label"])
            rows.extend(page)
            if len(rows) > total:
                raise ProvisionError("vast: inconsistent instance listing counts")
            token = data["next_token"]
            if token is None:
                if len(rows) != total:
                    raise ProvisionError("vast: incomplete instance listing")
                return rows
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise ProvisionError("vast: invalid or repeated instance listing cursor")
            seen_tokens.add(token)
            params["after_token"] = token
        # A partial listing must never read as "the pod is gone".
        raise ProvisionError("vast: instance listing did not finish paging")

    def _named(self, pod_id: str) -> list[dict]:
        return [r for r in self._instances() if pod_id in (r.get("label"), str(r.get("id")))]

    def _pod(self, pod_id: str) -> dict | None:
        hits = self._named(pod_id)
        if not hits:
            return None
        return next((r for r in hits if vast_instance_ready(r)), hits[0])

    def wait_ready(self, pod_id: str, *, timeout: float) -> bool:
        deadline = self._now() + timeout
        appear_by = self._now() + min(VAST_APPEAR_TIMEOUT, timeout)
        seen = False
        while self._now() < deadline:
            pod = self._pod(pod_id)
            if pod:
                seen = True
                if vast_instance_ready(pod):
                    return True
            elif not seen and self._now() >= appear_by:
                return False
            self._sleep(self.poll_interval)
        return False

    def get_ip(self, pod_id: str) -> PodAddress | None:
        pod = self._pod(pod_id)
        return vast_instance_address(pod) if pod else None

    def live_pod_address(self, pod_id: str) -> PodAddress | None:
        pod = self._pod(pod_id)
        return vast_instance_address(pod) if pod and vast_instance_ready(pod) else None

    def pod_identity(self, pod_id: str) -> dict | None:
        pod = self._pod(pod_id)
        if not pod:
            return None
        addr = vast_instance_address(pod)
        return {"id": str(pod.get("id") or ""), "huid": str(pod.get("machine_id") or ""),
                "status": str(pod.get("actual_status") or ""),
                "gpus": int(pod.get("num_gpus") or 0), "gpu_name": str(pod.get("gpu_name") or ""),
                "ip": addr.ip if addr else "", "port": addr.ssh_port if addr else 0}

    def terminate(self, pod_id: str) -> None:
        # Idempotent: destroy every instance answering to the label (a retried
        # create may have produced a twin) AND the contract the create handed
        # back — the v1 listing lags a fresh create, and iterating the listing
        # alone sent zero DELETEs while the instance billed on. "Already gone"
        # (a 404 / "not found" answer) is success; a 200 whose body says
        # ``success: false`` is not an acknowledgement.
        targets = [str(inst["id"]) for inst in self._named(pod_id)]
        contract = self._contract_by_name.get(pod_id)
        if contract and contract not in targets:
            targets.append(contract)
        for iid in targets:
            try:
                status, data = self._request("DELETE", f"{self.base}/instances/{iid}/")
            except ProvisionError as e:
                log.warning("vast destroy %s (%s): %s (treating as already terminated)",
                            pod_id, iid, e)
                continue
            acked = (200 <= status < 300
                     and not (isinstance(data, dict) and data.get("success") is False)
                     ) or status == 404 or (
                400 <= status < 500 and "not found" in str(data).lower())
            if acked:
                self._deleted.add(pod_id)
                log.info("vast destroy %s (%s)", pod_id, iid)
            else:
                log.warning("vast destroy %s (%s) answered %s: %s (treating as already "
                            "terminated)", pod_id, iid, status, self._scrub(str(data))[-200:])

    def create_requested(self, pod_id: str) -> bool:
        """A create for ``pod_id`` reached (or may have reached) the API and was
        not definitely refused — the write-ahead intent must then outlive the
        failure (``rent_funded_pod``). False for a launch that failed before
        any create went out (offer search, digest check): nothing to recover."""
        return pod_id in self._issued_creates

    def create_unresolved(self, pod_id: str) -> bool:
        """Whether an empty listing could still hide ``pod_id``: nothing by
        that label or id was ever observed on this account AND no DELETE for
        it was acknowledged. ``terminate_verified`` then refuses to call the
        pod gone — the listing lags a fresh create, and "not listed" alone
        confirmed a live, billing instance with zero DELETE calls sent."""
        return pod_id not in self._observed and pod_id not in self._deleted

    def list_tagged(self, prefix: str) -> list[str]:
        return filter_tagged_names([{"name": r.get("label") or ""} for r in self._instances()],
                                   prefix)

    def machine_of(self, pod_id: str) -> str | None:
        # Namespaced like Shadeform's: a Vast machine id must never collide
        # with a Lium executor id in the shared claim set.
        machine = self._machine_by_name.get(pod_id)
        if not machine:
            pod = self._pod(pod_id)
            machine = str((pod or {}).get("machine_id") or "")
        return f"vast:{machine}" if machine else None

    def host_of_pod(self, pod_id: str) -> str:
        """The machine a pod sits on, as the host-quarantine ledger names it
        (``vast:<machine_id>``; "" when unknown): this instance's launch
        record, else the listing. ``quarantine_lemon_host`` records it and
        ``_offers`` skips it, so a lemon machine is not rented again under
        another offer id."""
        return self.machine_of(pod_id) or ""

    def sku_of(self, pod_id: str) -> str:
        cached = self._sku_by_name.get(pod_id)
        if cached:
            return cached
        pod = self._pod(pod_id)
        sku = vast_sku_of_gpu(str((pod or {}).get("gpu_name") or ""))
        if not sku:
            raise ProvisionError("vast: cannot recover the allocated GPU type")
        return sku
