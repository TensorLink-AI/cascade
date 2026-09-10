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
0700). With a seal key configured (``CASCADE_VAULT_KEY_FILE`` → 32 raw bytes
or 64 hex chars; the same file on the intake and the trainer) the entry is
**sealed at rest** — ``{"sealed": <b64 nonce||ciphertext||tag>, "stored_at":
<unix>, "v": 1}`` under AES-256-GCM with the hotkey as associated data, so a
copied vault directory (backup, disk image) is useless without the key file
(PRISM's ``prism-lium-payer/sealed`` shape). Without a seal key the legacy
plaintext ``{"api_key": …, "stored_at": …}`` form is written, protected by
file mode alone; legacy files still hydrate when a key is later configured
and are re-sealed on their next insert/refresh.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["DEFAULT_TTL_SECONDS", "PayerKeyVault", "VAULT_KEY_FILE_ENV", "load_seal_key"]

log = logging.getLogger("cascade.funding.vault")

VAULT_KEY_FILE_ENV = "CASCADE_VAULT_KEY_FILE"


def load_seal_key(path: str | os.PathLike | None = None) -> bytes | None:
    """The 32-byte vault seal key from ``path`` (or ``$CASCADE_VAULT_KEY_FILE``),
    accepting raw bytes or 64 hex characters; ``None`` when unconfigured.
    A configured-but-unreadable key fails loud: silently falling back to
    plaintext would be the worst outcome."""
    p = Path(path) if path else (Path(os.environ[VAULT_KEY_FILE_ENV])
                                 if os.environ.get(VAULT_KEY_FILE_ENV) else None)
    if p is None:
        return None
    raw = p.read_bytes()
    text = raw.strip()
    if len(text) == 64 and all(c in b"0123456789abcdefABCDEF" for c in text):
        return bytes.fromhex(text.decode())
    if len(raw) != 32:
        raise ValueError(f"vault seal key {p} must be 32 raw bytes or 64 hex chars "
                         f"(got {len(raw)} bytes)")
    return raw


def _seal(key: bytes, hotkey: str, api_key: str) -> str:
    from Crypto.Cipher import AES  # pycryptodome, via the chain extra

    nonce = secrets.token_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(hotkey.encode())
    ct, tag = cipher.encrypt_and_digest(api_key.encode())
    return base64.b64encode(nonce + ct + tag).decode()


def _unseal(key: bytes, hotkey: str, blob: str) -> str:
    from Crypto.Cipher import AES

    raw = base64.b64decode(blob)
    nonce, ct, tag = raw[:12], raw[12:-16], raw[-16:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    cipher.update(hotkey.encode())
    return cipher.decrypt_and_verify(ct, tag).decode()

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
    # AES-256-GCM key sealing entries at rest; None ⇒ resolve from
    # $CASCADE_VAULT_KEY_FILE at construction, and plaintext when unset.
    seal_key: bytes | None = field(default=None, repr=False)
    # hotkey → (api_key, stored_at). Not in repr — a dataclass repr that
    # printed this dict would put every key one log line away from leaking.
    _entries: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.seal_key is None:
            self.seal_key = load_seal_key()
        if self.seal_key is not None and len(self.seal_key) != 32:
            raise ValueError("vault seal key must be 32 bytes")
        if self.dir is not None:
            self.dir = Path(self.dir)
            self.dir.mkdir(parents=True, exist_ok=True)
            os.chmod(self.dir, 0o700)
            if self.seal_key is None:
                log.warning("payer vault %s is PLAINTEXT at rest (no %s) — set a "
                            "seal key on the intake and the trainer before mainnet",
                            self.dir, VAULT_KEY_FILE_ENV)

    @property
    def sealed(self) -> bool:
        return self.seal_key is not None

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
            # Create 0600 from the first byte — write_text + chmod leaves a
            # umask-default window with the key already on disk.
            if self.seal_key is not None:
                record = {"sealed": _seal(self.seal_key, hotkey, api_key),
                          "stored_at": now, "v": 1}
            else:
                record = {"api_key": api_key, "stored_at": now}
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(record))
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
                stored_at = float(raw["stored_at"])
                if "sealed" in raw:
                    if self.seal_key is None:
                        log.error("vault entry %s is sealed but no seal key is "
                                  "configured (%s) — skipped", path.name,
                                  VAULT_KEY_FILE_ENV)
                        continue
                    api_key = _unseal(self.seal_key, path.stem, str(raw["sealed"]))
                else:
                    api_key = str(raw["api_key"])   # legacy plaintext entry
            except (OSError, ValueError, KeyError):
                continue
            except Exception as e:  # noqa: BLE001 — wrong key / tampered blob
                log.error("vault entry %s could not be unsealed (%s) — skipped",
                          path.name, type(e).__name__)
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
