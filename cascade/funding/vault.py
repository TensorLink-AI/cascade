"""Payer-key custody: memory-first, TTL-bounded, mode-0600 on disk, never logged.

Ports the semantics of PRISM's ``PayerKeyVault`` (``base`` repo,
``crates/prism-lium-payer``): a miner's Lium API key is held in master memory
for the life of its submission and *optionally* mirrored to an operator-local
directory so a control-plane restart can still stop the miner's pod (teardown
is the reason to persist, not convenience). Entries expire after a TTL sized
to outlive the longest possible train-plus-retry arc, so a key can never
linger indefinitely on disk.

What this deliberately is NOT: a database. Keys never enter any shared store
(the S3 buckets, the submissions db, the queue file — the queue references
hotkeys, never keys), never appear in logs or reprs, and never leave this
process except as the ``LIUM_API_KEY`` env of a rent subprocess
(``cascade.provision.core.LiumProvider``).

File format: one ``<hotkey>.json`` per entry under ``dir`` (mode 0600, dir
0700) holding ``{"api_key": …, "stored_at": <unix>}``. Plaintext-on-disk
matches the repo's existing posture for the operator's own key
(``.env.provisioner``); the protection is file mode + an operator-local path
that must stay off git and off any rsync'd pod tree.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULT_TTL_SECONDS", "PayerKeyVault"]

# 36h, the PRISM default: covers a 3h train leg plus every in-round requeue
# plus a full round of queue wait, with margin. An entry older than this that
# still matters is re-funded by the miner, not remembered by us.
DEFAULT_TTL_SECONDS = 36 * 3600.0

# Hotkeys are SS58 addresses (alphanumeric); anything else would let a
# hostile "hotkey" traverse out of the vault dir when used as a filename.
_SAFE_HOTKEY_RE = re.compile(r"^[0-9A-Za-z]{1,64}$")


def _safe_hotkey(hotkey: str) -> str:
    if not _SAFE_HOTKEY_RE.match(hotkey or ""):
        raise ValueError("hotkey is not a plain SS58 string")
    return hotkey


@dataclass
class PayerKeyVault:
    """``hotkey → Lium API key`` with TTL, in memory plus optional 0600 files."""

    dir: Path | None = None
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    clock: Callable[[], float] = time.time
    # hotkey → (api_key, stored_at). Not in repr — a dataclass repr that
    # printed this dict would put every key one log line away from leaking.
    _entries: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.dir is not None:
            self.dir = Path(self.dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.dir, 0o700)

    # ── writes ───────────────────────────────────────────────────────────────

    def insert(self, hotkey: str, api_key: str) -> None:
        """Store (or replace) ``hotkey``'s key with a fresh TTL."""
        hotkey = _safe_hotkey(hotkey)
        if not api_key:
            raise ValueError("refusing to store an empty api key")
        now = self.clock()
        self._entries[hotkey] = (api_key, now)
        if self.dir is not None:
            path = self.dir / f"{hotkey}.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"api_key": api_key, "stored_at": now}), encoding="utf-8"
            )
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)

    def refresh(self, hotkey: str) -> bool:
        """Re-stamp ``hotkey``'s TTL (measure-start / heartbeat re-seal).

        A key that survives to a training leg must outlive that leg however
        stale its intake was — PRISM re-seals on measure start and heartbeats
        for the same reason. False when the entry is gone/expired.
        """
        key = self.get(hotkey)
        if key is None:
            return False
        self.insert(hotkey, key)
        return True

    def remove(self, hotkey: str) -> None:
        """Forget ``hotkey``'s key (idempotent) — memory and disk."""
        hotkey = _safe_hotkey(hotkey)
        self._entries.pop(hotkey, None)
        if self.dir is not None:
            (self.dir / f"{hotkey}.json").unlink(missing_ok=True)

    # ── reads ────────────────────────────────────────────────────────────────

    def get(self, hotkey: str) -> str | None:
        """``hotkey``'s key, or ``None`` when absent or expired (expired ⇒ purged)."""
        hotkey = _safe_hotkey(hotkey)
        entry = self._entries.get(hotkey)
        if entry is None:
            return None
        api_key, stored_at = entry
        if self.clock() - stored_at > self.ttl_seconds:
            self.remove(hotkey)
            return None
        return api_key

    def has(self, hotkey: str) -> bool:
        return self.get(hotkey) is not None

    def hotkeys(self) -> list[str]:
        """Unexpired hotkeys, sorted (safe to log — carries no key material)."""
        return sorted(hk for hk in list(self._entries) if self.get(hk) is not None)

    # ── restart recovery ─────────────────────────────────────────────────────

    def hydrate(self) -> int:
        """Load unexpired entries from ``dir`` into memory; returns the count.

        Expired files are deleted rather than loaded — hydration is the one
        moment a restart naturally sweeps the directory. Unreadable files are
        skipped (never raised): a torn write must not block recovering the
        keys needed to tear down still-billing pods.
        """
        if self.dir is None:
            return 0
        loaded = 0
        now = self.clock()
        for path in sorted(self.dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                api_key = str(raw["api_key"])
                stored_at = float(raw["stored_at"])
            except (OSError, ValueError, KeyError):
                continue
            if not api_key or now - stored_at > self.ttl_seconds:
                path.unlink(missing_ok=True)
                continue
            self._entries[path.stem] = (api_key, stored_at)
            loaded += 1
        return loaded

    def purge_expired(self) -> int:
        """Drop every expired entry (memory + disk); returns how many."""
        expired = [
            hk for hk, (_, stored_at) in list(self._entries.items())
            if self.clock() - stored_at > self.ttl_seconds
        ]
        for hk in expired:
            self.remove(hk)
        return len(expired)
