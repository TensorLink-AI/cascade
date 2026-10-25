"""Miner-funded GPU gangs on the existing marketplace adapters (DEC-CA-0036).

One payer's job rents ONE pod with the whole gang (``world_size`` GPUs) on
the compute provider the miner chose — Lium, Vast.ai or RunPod — billed to the
key that miner deposited in the per-provider vault. The key is resolved per
call and never persisted here. Pods carry no credentials: the orchestrator
dispatches over SSH and harvests results (``cascade.direct.pods``).

Marketplaces have no client-token idempotency, so the pod NAME is the
reservation identity: it derives from the reservation key, a launch is
submitted at most once per key, and every later call looks the name up.
A launch that never appears is not refunded (it may still bill); the release
path terminates by name and confirms by re-listing (``terminate_verified``).
"""

from __future__ import annotations

import fcntl
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from ..funding.faults import classify_rent_failure, is_funding_failure
from ..provision.core import LaunchSpec, ProvisionError, scan_ssh_host_key, validate_digest_pinned
from ..provision.direct import (
    Allocation,
    AllocationPending,
    CapacityUnavailable,
    CredentialsRevoked,
    ProviderTransient,
    SpendingRefused,
)
from ..provision.funded import (
    COMPUTE_PROVIDERS,
    apply_price_caps,
    provider_for_key,
    terminate_verified,
)
from ..shared.direct_moe import digest

# A submitted launch that is still not listed after this long never started.
APPEAR_SECONDS = 600.0
_SLUG_RE = re.compile(r"[^0-9a-z]")
_SPEND_MARKERS = ("insufficient", "balance", "credit", "payment required", "402")


@dataclass(frozen=True)
class PodProfile:
    """A qualified gang shape on one marketplace: exact SKU, count and image."""

    sku: str
    world_size: int
    gpu_name: str
    memory_gib: int
    precision: str
    hourly_microusd: int
    max_seconds: int
    image: str
    qualification_digest: str

    def __post_init__(self):
        if (
            not all((self.sku, self.gpu_name, self.image))
            or min(self.world_size, self.memory_gib, self.hourly_microusd, self.max_seconds) <= 0
            or self.precision not in ("float32", "bfloat16")
            or len(self.qualification_digest) != 64
        ):
            raise ValueError("incomplete measured/qualified pod profile")
        validate_digest_pinned(self.image)

    @property
    def maximum_estimate(self):
        return math.ceil(self.hourly_microusd * self.max_seconds / 3600)


def direct_pod_prefix(key: str, payer: str, netuid: int) -> str:
    """``cascade-n<netuid>-direct<12hex>-<slug>``: deployment-scoped and
    payer-attributable, and deliberately NOT the rolling lane's
    ``cascade-n<netuid>-<digits>-funded-<slug>`` scheme, so the rolling
    reaper (``reconcile_funded``) never treats a direct gang as its orphan."""
    slug = _SLUG_RE.sub("", payer.lower())[:12]
    if not slug:
        raise ValueError("cannot derive a pod slug from the payer hotkey")
    return f"cascade-n{int(netuid)}-direct{digest({'direct-pod': key})[:12]}-{slug}"


def serialized_effect(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        # Serialize create/release across scheduler processes; a crash leaves the
        # recorded submission, which the next process reconciles by pod name.
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return method(self, *args, **kwargs)

    return guarded


class PayerPodProvider:
    """The direct controller's provider for one marketplace."""

    conservative_accounting = True

    def __init__(self, ledger, vault, compute, profiles, root, *, ssh_pubkey, netuid,
                 allow_paid=False, provider_factory=provider_for_key,
                 host_key_scanner=scan_ssh_host_key, appear_seconds=APPEAR_SECONDS,
                 clock=time.time):
        if compute not in COMPUTE_PROVIDERS:
            raise ValueError(f"unknown compute provider {compute!r}")
        if not ssh_pubkey:
            raise ValueError("orchestrator SSH public key required")
        self.ledger, self.vault, self.compute = ledger, vault, compute
        self.profiles, self.ssh_pubkey, self.netuid = profiles, ssh_pubkey, int(netuid)
        self.allow_paid, self.factory = allow_paid, provider_factory
        self.scan, self.appear_seconds, self.clock = host_key_scanner, appear_seconds, clock
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{compute}.sqlite"
        self.lock_path = Path(str(path) + ".lock")
        self.db = sqlite3.connect(str(path), isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, payer TEXT NOT NULL,
                 profile TEXT NOT NULL, prefix TEXT NOT NULL, deadline REAL NOT NULL,
                 cap INTEGER NOT NULL, released INTEGER NOT NULL DEFAULT 0,
                 pod_uid TEXT NOT NULL DEFAULT '', host_key TEXT NOT NULL DEFAULT '');
               CREATE TABLE IF NOT EXISTS submissions (key TEXT PRIMARY KEY, at REAL NOT NULL);
               CREATE TABLE IF NOT EXISTS rejected (key TEXT PRIMARY KEY);
               CREATE TABLE IF NOT EXISTS cancelled (key TEXT PRIMARY KEY);"""
        )

    # ── journal ──────────────────────────────────────────────────────────────

    def _request(self, key):
        row = self.db.execute(
            "SELECT payer,profile,prefix,deadline,cap,released,pod_uid,host_key FROM requests WHERE key=?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        names = ("payer", "profile", "prefix", "deadline", "cap", "released", "pod_uid", "host_key")
        req = dict(zip(names, row, strict=True))
        req["released"] = bool(req["released"])
        return req

    def _submitted_at(self, key):
        row = self.db.execute("SELECT at FROM submissions WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def _rejected(self, key):
        return self.db.execute("SELECT 1 FROM rejected WHERE key=?", (key,)).fetchone() is not None

    def _payer(self, key):
        matches = [j for j in self.ledger.state["jobs"].values()
                   if j["reservation"] and j["reservation"]["key"] == key]
        if len(matches) != 1:
            raise ValueError("reservation has no unique payer attribution")
        return matches[0]["terms"]["payer"]

    def _adapter(self, payer, profile=None):
        api_key = self.vault.get_shared(payer)
        from ..funding.providers import PREFIX, decode_credentials

        if api_key and api_key.startswith(PREFIX):
            api_key = next((c.api_key for c in decode_credentials(api_key)
                            if c.provider == self.compute), None)
        if not api_key:
            raise CredentialsRevoked(f"no {self.compute} key on file for the payer")
        adapter = self.factory(self.compute, api_key)
        if profile is not None:
            hourly = profile.hourly_microusd / 1e6
            apply_price_caps(adapter, max_price_per_hour=hourly)
            if hasattr(adapter, "max_price_per_hour"):
                adapter.max_price_per_hour = hourly
        return adapter

    # ── controller protocol ──────────────────────────────────────────────────

    def estimate(self, profile, world_size):
        p = self.profiles[profile]
        if p.world_size != world_size:
            raise ValueError("scientific world-size contract differs from qualified profile")
        return p.maximum_estimate

    def no_charge_evidence(self, key):
        if self._request(key) is None or self._submitted_at(key) is None:
            return "never_submitted"
        if self._rejected(key):
            return "provider_rejected"
        return None

    def lookup(self, key):
        req = self._request(key)
        if req is None or self._rejected(key):
            return None
        adapter = self._adapter(req["payer"])
        names = sorted(adapter.list_tagged(req["prefix"]))
        if not names:
            return None
        if len(names) > 1:
            raise CapacityUnavailable("reservation has duplicate pods; release and reconcile")
        if req["released"]:
            raise CapacityUnavailable("allocation no longer usable")
        name = names[0]
        identity = adapter.pod_identity(name) or {}
        uid = str(identity.get("id") or "")
        if not uid:
            raise AllocationPending("pod listed without a platform identity yet")
        if req["pod_uid"] and uid != req["pod_uid"]:
            # The payer controls the account: a different pod under our name is
            # theirs to answer for, never silently adopted.
            raise CredentialsRevoked("payer pod replaced under the reservation name")
        address = adapter.live_pod_address(name)
        if address is None:
            raise AllocationPending(f"{self.compute} pod is booting")
        p = self.profiles[req["profile"]]
        gpus = int(identity.get("gpus") or 0)
        if gpus and gpus != p.world_size:
            raise CapacityUnavailable("provider returned a different gang size")
        if not req["pod_uid"]:
            self.db.execute("UPDATE requests SET pod_uid=? WHERE key=?", (uid, key))
        if not req["host_key"]:
            # Pin the container's SSH host key on first sight; every dispatch
            # then runs StrictHostKeyChecking against exactly this key.
            try:
                host_key = self.scan(address.ip, address.ssh_port)
            except Exception as e:  # noqa: BLE001 — sshd may still be starting
                raise AllocationPending(f"pod ssh host key not readable yet: {e}") from None
            self.db.execute("UPDATE requests SET host_key=? WHERE key=?", (host_key, key))
        # VM state is not GPU readiness: the dispatcher attests every rank first.
        return Allocation(name, tuple(range(p.world_size)), req["deadline"], req["profile"], req["cap"])

    def address(self, key):
        """``(PodAddress, pinned host key)`` of the held pod for SSH dispatch."""
        req = self._request(key)
        if req is None or not req["host_key"]:
            raise AllocationPending("pod address not pinned yet")
        adapter = self._adapter(req["payer"])
        names = sorted(adapter.list_tagged(req["prefix"]))
        address = adapter.live_pod_address(names[0]) if len(names) == 1 else None
        if address is None:
            raise ProviderTransient("held pod is not reachable right now")
        return address, req["host_key"]

    @serialized_effect
    def reserve(self, key, *, profile, world_size, cap, now):
        if not self.allow_paid:
            raise PermissionError("paid pod provisioning requires explicit authorization")
        if self.db.execute("SELECT 1 FROM cancelled WHERE key=?", (key,)).fetchone():
            raise CapacityUnavailable("reservation was cancelled")
        p = self.profiles[profile]
        if world_size != p.world_size:
            raise ValueError("scientific world-size contract differs from qualified profile")
        if p.maximum_estimate > cap:
            raise SpendingRefused("admitted spending cap below bounded reservation estimate")
        payer = self._payer(key)
        prefix = direct_pod_prefix(key, payer, self.netuid)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            previous = self._request(key)
            if previous is None:
                self.db.execute(
                    "INSERT INTO requests(key,payer,profile,prefix,deadline,cap) VALUES (?,?,?,?,?,?)",
                    (key, payer, profile, prefix, now + p.max_seconds, p.maximum_estimate),
                )
            elif previous["profile"] != profile or previous["released"] or previous["payer"] != payer:
                raise ValueError("reservation key reused after release/profile/payer change")
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        if self._rejected(key):
            raise CapacityUnavailable("original allocation request was rejected")
        found = self.lookup(key)
        if found is not None:
            return found
        submitted = self._submitted_at(key)
        if submitted is not None:
            if now - submitted < self.appear_seconds:
                raise AllocationPending("launch submitted; pod not listed yet")
            # Never visible: not provably free, so the debit stays (no evidence).
            raise CapacityUnavailable("submitted launch never appeared")
        adapter = self._adapter(payer, p)
        try:
            if adapter.capacity(p.sku, gpus=p.world_size) < 1:
                self.db.execute("INSERT OR IGNORE INTO rejected VALUES (?)", (key,))
                raise CapacityUnavailable(f"no {p.world_size}x{p.sku} offer on {self.compute}")
        except ProvisionError as e:
            raise self._classified(key, e, created=False) from None
        self.db.execute("INSERT INTO submissions VALUES (?,?)", (key, now))
        try:
            adapter.launch(LaunchSpec(sku=p.sku, count=1, image=p.image, ssh_pubkey=self.ssh_pubkey,
                                      name_prefix=prefix, gpus_per_pod=p.world_size))
        except ProvisionError as e:
            raise self._classified(key, e, created=None) from None
        raise AllocationPending(f"{self.compute} launch accepted; awaiting pod")

    def _classified(self, key, error, *, created):
        """Map an adapter refusal onto the controller's attribution classes.

        A definitive refusal (auth, sold out, balance) means nothing was
        created and is recorded as rejected; anything else is ambiguous and
        keeps the submission so the next call reconciles by name.
        """
        from ..provision.core import UncertainCreateError

        text = str(error)
        if isinstance(error, UncertainCreateError):
            return ProviderTransient(text)
        cls = classify_rent_failure(text)
        definitive = cls in ("auth", "no_capacity") or any(m in text.lower() for m in _SPEND_MARKERS)
        if definitive or created is False:
            self.db.execute("INSERT OR IGNORE INTO rejected VALUES (?)", (key,))
        if cls == "auth" and not is_funding_failure(text):
            return CredentialsRevoked(text)
        if cls == "no_capacity":
            return CapacityUnavailable(text)
        if any(m in text.lower() for m in _SPEND_MARKERS):
            return SpendingRefused(text)
        return ProviderTransient(text)

    @serialized_effect
    def release(self, key):
        self.db.execute("INSERT OR IGNORE INTO cancelled VALUES (?)", (key,))
        req = self._request(key)
        if req is None:
            return
        self.db.execute("UPDATE requests SET released=1 WHERE key=?", (key,))
        if self._submitted_at(key) is None:
            return
        try:
            adapter = self._adapter(req["payer"])
        except CredentialsRevoked:
            # Teardown needs the payer's key (a pod on their account is invisible
            # to ours). Keep the reservation releasing and alert; the vault TTL
            # must outlive every legitimate pod lifetime.
            raise AllocationPending("payer key gone: cannot confirm pod teardown") from None
        names = adapter.list_tagged(req["prefix"])
        if not names:
            submitted = self._submitted_at(key)
            if self._rejected(key) or req["pod_uid"]:
                return
            if submitted is not None and self.clock() - submitted < self.appear_seconds:
                raise AllocationPending("unresolved launch; absence is not termination yet")
            return
        for name in names:
            if not terminate_verified(adapter, name):
                raise AllocationPending("termination requested; pod still listed")

    def reap_expired(self, now):
        keys = [r[0] for r in self.db.execute(
            "SELECT key FROM requests WHERE deadline<=? AND released=0", (now,))]
        waiting = []
        for key in keys:
            try:
                self.release(key)
            except AllocationPending:
                waiting.append(key)
        return waiting

    def close(self):
        self.db.close()
