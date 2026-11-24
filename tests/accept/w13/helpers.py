"""Shared fixtures for the W13 acceptance tests (lane D king publication).

Everything here comes from docs/plans/LANE_D_PLAN.md: the W13 row (§3.0
lane D core table), §11.1 "Where checkpoints live", §1.2 "Storage" with the
rev6.4 finding (Hippius has no Object Lock: a king's immutability is the
content-addressed path + the re-hash on every read + the signed digest + the
R2/HF mirrors), and B13 (the Hub/HF takedown path).

Nothing from the PATHS modules is imported at module level or in a fixture:
``publish()`` imports ``cascade.lane_d.publish`` inside the calling test, so on
main every such test fails in its own body with ``ModuleNotFoundError`` for a
PATHS module (a right reason), never at collection.

No network: the mirrors are in-memory fakes (:class:`FakeMirror`), the vault
is the real ``FilesystemStorage`` adapter under ``tmp_path``, and the test
keys are derived from fixed ``//w13-*`` URIs (never a real key).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import socket
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE = "cascade.lane_d.publish"
WORKFLOW = REPO / ".github" / "workflows" / "laned-king-sync.yml"
LANE_A_WORKFLOW = REPO / ".github" / "workflows" / "king-sync.yml"
SYNC_SCRIPT = REPO / "scripts" / "laned_king_sync.py"
PUBLISH_SRC = REPO / "cascade" / "lane_d" / "publish.py"

MIRRORS = ("hub", "r2", "hf")
INDEX_PATH = "champions/lane_d/index.json"
INDEX_SCHEMA = "lane-d-index/1"
HF_REPO = "tensorlink-dev/cascade-lane-d-kings"
CARD_FILES = ("PROVENANCE.json", "MODEL_CARD.md")
KING_FIELDS = ("hotkey", "context", "digest", "crown_block", "lcb", "gate", "sentinel", "license")

# A published artifact (index, cards) must not claim a lock it does not have.
LOCK_CLAIM = re.compile(r"(?i)\b(object[ _-]?lock|locked|lock|retention|compliance|worm)\b")
# S3 Object Lock API surface: none of it may appear in the publication code.
LOCK_API = ("ObjectLockMode", "ObjectLockRetainUntilDate", "ObjectLockLegalHoldStatus",
            "ObjectLockEnabledForBucket", "put_object_retention", "put_object_legal_hold",
            "put_object_lock_configuration", "x-amz-object-lock", "BypassGovernanceRetention")

CHECKPOINT_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")


def publish():
    """The module under test (imported inside the test body)."""
    return importlib.import_module(MODULE)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any socket connect raises: publication is tested with fakes only.
    Test modules import this fixture so it applies to them (autouse)."""
    def refuse(*args, **kwargs):
        raise AssertionError("W13 tests must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeMirror:
    """An in-memory public mirror (Hippius Hub, R2 public bucket or the HF
    dataset), behind the five-method client protocol of README "Mirror
    clients". ``put`` takes exactly ``(key, data)``: there is no way to pass
    lock or retention options. Every call is logged."""

    def __init__(self, name: str):
        self.name = name
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []

    def put(self, key, data):
        assert isinstance(key, str) and key, key
        self.calls.append(("put", key))
        self.objects[key] = bytes(self.transform(key, bytes(data)))

    def transform(self, key: str, data: bytes) -> bytes:
        return data

    def get(self, key):
        self.calls.append(("get", key))
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    def list(self, prefix):
        self.calls.append(("list", prefix))
        return sorted(k for k in self.objects if k.startswith(prefix))

    def size(self, key):
        self.calls.append(("size", key))
        if key not in self.objects:
            raise KeyError(key)
        return len(self.objects[key])

    def delete(self, key):
        self.calls.append(("delete", key))
        self.objects.pop(key, None)

    # test-side helpers (not part of the protocol)
    def keys_with(self, digest: str) -> list[str]:
        return sorted(k for k in self.objects if digest in k.split("/"))

    def key_ending(self, digest: str, name: str) -> str:
        hits = [k for k in self.keys_with(digest) if k.rsplit("/", 1)[-1] == name]
        assert len(hits) == 1, (self.name, name, sorted(self.objects))
        return hits[0]


class CorruptingMirror(FakeMirror):
    """A mirror whose stored weights differ from what was sent (a broken or
    hostile upload path): the read-back re-hash must catch it."""

    def transform(self, key, data):
        if key.endswith("weights.safetensors") and data:
            return bytes([data[0] ^ 0xFF]) + data[1:]
        return data


def mirrors(**override) -> dict:
    out = {name: FakeMirror(name) for name in MIRRORS}
    out.update(override)
    return out


def write_checkpoint(directory: Path, seed: int = 0) -> Path:
    """A tiny lane D checkpoint dir (the W4 four-file allowlist); bytes are
    a pure function of ``seed``."""
    directory.mkdir(parents=True, exist_ok=True)
    blob = b"".join(hashlib.sha256(f"w13 weights {seed} {i}".encode()).digest() for i in range(64))
    contents = {
        "config.json": json.dumps({"arch": "cascade_pyramid", "seed": seed}, sort_keys=True).encode(),
        "forecast_wrapper.py": f"# w13 wrapper {seed}\n".encode(),
        "model.py": f"# w13 model {seed}\n".encode(),
        "weights.safetensors": blob,
    }
    for name, data in contents.items():
        (directory / name).write_bytes(data)
    return directory


def vault_with_king(tmp: Path, hotkey: str, seed: int = 0):
    """``(vault, digest, files)``: the real FilesystemStorage vault holding one
    entry uploaded by the real intake (``upload_checkpoint``)."""
    from cascade.lane_d.intake import upload_checkpoint
    from cascade.lane_d.storage import FilesystemStorage

    ckpt = write_checkpoint(tmp / f"ckpt-{seed}", seed)
    vault = FilesystemStorage(tmp / "vault")
    digest = upload_checkpoint(vault, hotkey, ckpt)
    files = {name: (ckpt / name).read_bytes() for name in CHECKPOINT_FILES}
    return vault, digest, files


def vault_key(hotkey: str, digest: str, name: str) -> str:
    return f"lane-d/{hotkey}/{digest}/{name}"


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w13-{name}")


def king_record(hotkey: str, digest: str, *, crown_block: int = 9_000_900,
                context: int = 8192, lcb: float = 0.0123) -> dict:
    """One index entry with every §11.1 field (gate and sentinel numbers,
    the sentinel suite named, the licence)."""
    return {
        "hotkey": hotkey,
        "context": context,
        "digest": digest,
        "crown_block": crown_block,
        "lcb": lcb,
        "gate": {"outcome": "pass", "ref": "king_d",
                 "suites": {"gift_eval": {"crps": 0.512, "mase": 0.861},
                            "boom": {"crps": 0.377, "mase": 0.902}}},
        "sentinel": {"suite": "time", "crps": 0.433, "mase": 0.915},
        "license": "MIT",
    }


def body(index: dict) -> bytes:
    """The index without its signature, canonical (the compared bytes)."""
    rest = {k: v for k, v in index.items() if k != "signature"}
    return json.dumps(rest, sort_keys=True, separators=(",", ":")).encode()


def tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` (relative posix path -> bytes)."""
    if not root.exists():
        return {}
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}
