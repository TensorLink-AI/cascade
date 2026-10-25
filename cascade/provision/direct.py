"""Direct-MoE provider boundary. Only the in-memory fault provider is qualified.

A real adapter must prove reservation lookup/idempotency, full gang readiness,
lease expiry and bounded billing before it may implement this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CapacityUnavailable(RuntimeError):
    pass


class CredentialsRevoked(RuntimeError):
    pass


class SpendingRefused(RuntimeError):
    pass


class AllocationPending(RuntimeError):
    """An existing reservation is booting/terminating; retain its original intent/key."""


class ProviderTransient(OSError):
    """A provider call failed ambiguously (5xx, timeout, rate limit). The
    scheduler retries the SAME reservation later; it never re-keys on this."""


@dataclass(frozen=True)
class Allocation:
    identity: str
    ranks: tuple[int, ...]
    lease_until: float
    profile: str
    max_charge: int


class Provider(Protocol):
    def lookup(self, key: str) -> Allocation | None: ...
    def reserve(self, key: str, *, profile: str, world_size: int, cap: int, now: float) -> Allocation: ...
    def release(self, key: str) -> None: ...


class FakeProvider:
    """Fault injection only. No network calls and no real charges."""
    def __init__(self, *, mode="ok", lease_seconds=60, price=1):
        self.mode, self.lease_seconds, self.price = mode, lease_seconds, price
        self.allocations = {}
        self.released = set()
        self.reservations = 0
        self.timeout_fired = False

    def lookup(self, key):
        return self.allocations.get(key)

    def reserve(self, key, *, profile, world_size, cap, now):
        if key in self.allocations:
            return self.allocations[key]
        if self.mode == "revoked":
            raise CredentialsRevoked("revoked")
        if self.mode == "scarcity":
            raise CapacityUnavailable("no qualified gang")
        if cap < self.price:
            raise SpendingRefused("spend cap")
        ranks = world_size - 1 if self.mode == "partial" else world_size
        a = Allocation(key, tuple(range(ranks)), now + self.lease_seconds, profile, self.price)
        self.allocations[key] = a
        self.reservations += 1
        if self.mode == "timeout_after_reserve" and not self.timeout_fired:
            self.timeout_fired = True
            raise TimeoutError("reservation result lost")
        return a

    def release(self, key):
        if self.mode == "release_timeout":
            raise TimeoutError("release uncertain")
        self.released.add(key)


class MemoryObjects:
    """Remote-object semantics for offline simulations; never production storage."""
    prefix = "memory://"

    def __init__(self):
        self.objects = {}
        self.outage = False

    def put(self, uri, body):
        if self.outage:
            raise OSError("object store unavailable")
        if uri in self.objects and self.objects[uri] != body:
            raise ValueError("immutable object conflict")
        self.objects[uri] = body

    def get(self, uri):
        if self.outage:
            raise OSError("object store unavailable")
        return self.objects[uri]
