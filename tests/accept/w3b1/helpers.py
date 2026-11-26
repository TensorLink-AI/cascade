"""Shared fixtures-as-functions for the W3b1 acceptance tests.

Nothing here imports ``cascade.validator.lane_d_gate`` (the module under
test): every test imports it in its body, so on main the failure is a
``ModuleNotFoundError`` for a PATHS module inside the test. The reference
sentinel draws below are test-side copies of hash-to-index mappings the gate
draw must be independent of; they are not the implementation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SHIPPED = ("chain.toml", "chain.testnet.toml")

ALL_SUITES = ("gift", "boom", "time", "fev")
N_HASHES = 6000
# A settlement boundary on both the mainnet (900) and testnet (150) grids.
BOUNDARY = 2698 * 3600 + 900


def synth_hash(tag: str | int) -> str:
    """A block-hash-shaped value (``0x`` + 64 lowercase hex), as
    ``ChainClient.block_hash`` returns it."""
    return "0x" + hashlib.sha256(f"cascade w3b1 boundary {tag}".encode()).hexdigest()


HASHES = tuple(synth_hash(i) for i in range(N_HASHES))


# ── reference draws the gate draw must be independent of ─────────────────────


def _hex(block_hash: str) -> str:
    return block_hash.lower().removeprefix("0x")


def ref_sentinel_ws1(block_hash: str, suites: Sequence[str]) -> str:
    """W-S1's sentinel mapping as proposed in PR #545
    (``cascade.shared.sentinel.sentinel_for_era``): tagged sha256 of the hex
    digits, big-endian, modulo the suite count."""
    d = hashlib.sha256(b"cascade|sentinel|v1|" + _hex(block_hash).encode("ascii")).digest()
    return suites[int.from_bytes(d, "big") % len(suites)]


def ref_sentinel_raw_int(block_hash: str, suites: Sequence[str]) -> str:
    """The untagged draw: the hash itself as an integer, modulo the count."""
    return suites[int(_hex(block_hash), 16) % len(suites)]


def ref_sentinel_plain_sha(block_hash: str, suites: Sequence[str]) -> str:
    """An untagged sha256 of the hex digits, modulo the count."""
    d = hashlib.sha256(_hex(block_hash).encode("ascii")).digest()
    return suites[int.from_bytes(d, "big") % len(suites)]


def ref_sentinel_round_seed(block_hash: str, suites: Sequence[str]) -> str:
    """The round base seed of the same hash (main's
    ``cascade.shared.chain.seed_from_block_hash``), modulo the count."""
    d = hashlib.blake2b(_hex(block_hash).encode(), digest_size=8).digest()
    return suites[int.from_bytes(d, "big") % len(suites)]


def reference_sentinels() -> dict[str, Callable[[str, Sequence[str]], str]]:
    refs: dict[str, Callable[[str, Sequence[str]], str]] = {
        "ws1_pr545": ref_sentinel_ws1,
        "raw_int": ref_sentinel_raw_int,
        "plain_sha256": ref_sentinel_plain_sha,
        "round_seed": ref_sentinel_round_seed,
    }
    # Once W-S1 lands, its real draw joins the references.
    if importlib.util.find_spec("cascade.shared.sentinel") is not None:
        import cascade.shared.sentinel as ws1

        if hasattr(ws1, "sentinel_for_era"):
            refs["ws1_module"] = lambda h, s: ws1.sentinel_for_era(h, list(s))
    return refs


# ── config ────────────────────────────────────────────────────────────────────


@cache
def _raw_text(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def shipped_raw(name: str) -> dict:
    return tomllib.loads(_raw_text(name))


def lane_d_cfg(name: str = "chain.toml", **lane_d: object):
    """The parsed ``[lane_d]`` of a shipped file with keys overwritten."""
    from cascade.shared.config import parse_chain_config

    raw = shipped_raw(name)
    raw["lane_d"] = {**dict(raw.get("lane_d") or {}), **lane_d}
    return parse_chain_config(raw).lane_d


# ── processes ─────────────────────────────────────────────────────────────────


DRAW_SCRIPT = (
    "import json, sys\n"
    "from cascade.validator.lane_d_gate import gate_suite_for\n"
    "hashes = json.loads(sys.argv[1]); suites = json.loads(sys.argv[2])\n"
    "weights = json.loads(sys.argv[3]); sentinel = json.loads(sys.argv[4])\n"
    "print(json.dumps([gate_suite_for(h, gate_suites=suites, gate_weights=weights,\n"
    "                                 sentinel=sentinel) for h in hashes]))\n"
)


def draw_in_process(hashes: Sequence[str], suites: Sequence[str], weights: dict | None,
                    sentinel: str | None, hash_seed: str) -> list[str]:
    """The draws computed by a separate interpreter (another validator) with
    its own ``PYTHONHASHSEED``."""
    env = {"PYTHONPATH": str(REPO), "PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"}
    proc = subprocess.run([sys.executable, "-P", "-c", DRAW_SCRIPT, json.dumps(list(hashes)),
                           json.dumps(list(suites)), json.dumps(weights),
                           json.dumps(sentinel)],
                          capture_output=True, text=True, timeout=120, check=False,
                          env=env, cwd=str(REPO))
    assert proc.returncode == 0, f"draw subprocess failed:\n{proc.stderr}"
    return json.loads(proc.stdout)
