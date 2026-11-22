"""W3b2 (RV5): the owner's read-only live verify of the pod gate-sweep evidence.

The pod run writes ``evidence/w3b2.json`` (schema ``laned_w3b2/2``): one paired
sweep of a candidate, KING_D and the house on one bench host, on the suite
W3b1 draws from ``draw.block_hash``. Offline, the evidence can only be checked
for internal consistency; this module re-checks the facts an offline writer
can't choose:

* ``draw.block`` is finalized and its hash on the chain is ``draw.block_hash``;
* ``gate_suite_for(<the chain's hash>)`` under the shipped ``chain.toml`` (with
  the evidence's sentinel) is the evidence's ``suite``;
* every ``refs.<role>`` resolves in the checkpoint registry to
  ``sha256:<rows.<role>.digest>``;
* ``refs.king_d`` / ``refs.house`` are the KING_D and house pointers in force
  at ``draw.block``.

Only then is a ``verified_live`` record (schema ``laned_w3b2_verify/1``)
returned, bound to the evidence body by ``body_sha256`` (W3a's canonical
form). Usage (owner, from a clean checkout of main; read-only chain access, no
wallet, no keys)::

    python -P -m cascade.laned.bench_evidence verify --evidence evidence/w3b2.json

Exit 0 after writing ``verified_live`` into the file; exit 2 on any refusal
(the code on stderr, the file byte-identical). ``not_pod``, ``wrong_network``
and ``malformed`` refusals come before any adapter is built or any network
use.

The CLI's adapters are read-only: the chain through ``ChainClient``'s
``finalized_block`` / ``block_hash`` (no wallet is ever opened, no extrinsic
is reachable from the facade), the registry through a manifest HEAD on the
Hippius Hub OCI registry (anonymous pull unless Hub credentials are in the
environment). The KING_D / house pointers at a historical block have no
published read-only source on main yet (lane D is unarmed, ``lane_d_state.json``
is validator-local and keeps no per-block history), so after the chain and
registry checks the CLI refuses with the typed code ``pointers_unavailable``.
Contract: ``tests/accept/w3b2/README.md`` "The live verify".
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

MODULE = "cascade.laned.bench_evidence"
VERIFY_SCHEMA = "laned_w3b2_verify/1"
NETWORK = "finney"
ROLES = ("candidate", "king_d", "house")
REFS = ("king_d", "house")
REPO = Path(__file__).resolve().parents[2]

HASH_RE = re.compile(r"0x[0-9a-f]{64}")
HEX64_RE = re.compile(r"[0-9a-f]{64}")
REF_RE = re.compile(r"([^\s@]+)@sha256:([0-9a-f]{64})")
MANIFEST_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")

POINTERS_UNAVAILABLE = "pointers_unavailable"


class VerifyRefused(Exception):
    """``verify_live`` refuses the evidence; ``code`` says why (README "The
    live verify": ``not_pod``, ``wrong_network``, ``malformed``,
    ``block_not_finalized``, ``block_hash_mismatch``, ``draw_mismatch``,
    ``pointer_mismatch``, ``ref_unpublished``; the CLI adds
    ``pointers_unavailable``)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


# ── the shipped config ───────────────────────────────────────────────────────


def _chain_raw() -> dict:
    return tomllib.loads((REPO / "chain.toml").read_text(encoding="utf-8"))


def shipped_netuid() -> int:
    """``[subnet] netuid`` of the shipped ``chain.toml``."""
    return int(_chain_raw()["subnet"]["netuid"])


def drawn_suite(block_hash: str, sentinel) -> str:
    """W3b1's gate suite for ``block_hash`` under the shipped ``chain.toml``."""
    from cascade.shared.config import parse_chain_config
    from cascade.validator.lane_d_gate import gate_suite_for

    cfg = parse_chain_config(_chain_raw()).lane_d
    return gate_suite_for(block_hash, gate_suites=cfg.gate_suites,
                          gate_weights=cfg.gate_weights, sentinel=sentinel)


# ── the record's bindings ────────────────────────────────────────────────────


def canonical_body(evidence: dict) -> bytes:
    """W3a's canonical form: the evidence without ``verified_live``, sorted
    keys, compact separators, ASCII, UTF-8."""
    body = {k: v for k, v in evidence.items() if k != "verified_live"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def body_sha256(evidence: dict) -> str:
    return hashlib.sha256(canonical_body(evidence)).hexdigest()


def tool_info() -> dict:
    """This checkout's commit and whether it is dirty (``git_dirty`` is True
    whenever git can't tell)."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=False)

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    sha = head.stdout.strip() if head.returncode == 0 else ""
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    return {"module": MODULE, "git_sha": sha, "git_dirty": dirty}


def _iso_now(now: datetime | None) -> str:
    at = datetime.now(UTC) if now is None else now
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return at.astimezone(UTC).replace(microsecond=0).isoformat()


# ── offline preconditions (no adapter use) ───────────────────────────────────


def preflight(evidence) -> dict:
    """The checks that need no chain, registry or pointer read, in the
    pinned order ``not_pod`` → ``wrong_network`` → ``malformed``. Returns the
    parsed facts the live checks use."""
    if not isinstance(evidence, dict) or evidence.get("source") != "pod":
        got = evidence.get("source") if isinstance(evidence, dict) else None
        raise VerifyRefused("not_pod", f"evidence source is {got!r}, not 'pod': only a pod "
                            "run's evidence can be verified")
    draw = evidence.get("draw")
    if not isinstance(draw, dict):
        raise VerifyRefused("malformed", "draw is not an object")
    netuid = shipped_netuid()
    if draw.get("network") != NETWORK:
        raise VerifyRefused("wrong_network", f"draw.network is {draw.get('network')!r}, not "
                            f"{NETWORK!r}: the gate draws from the mainnet boundary")
    dn = draw.get("netuid")
    if isinstance(dn, bool) or dn != netuid:
        raise VerifyRefused("wrong_network", f"draw.netuid is {dn!r}, not chain.toml's "
                            f"[subnet] netuid {netuid}")
    block = draw.get("block")
    if isinstance(block, bool) or not isinstance(block, int) or block <= 0:
        raise VerifyRefused("malformed", f"draw.block {block!r} is not a positive int")
    bh = draw.get("block_hash")
    if not isinstance(bh, str) or not HASH_RE.fullmatch(bh):
        raise VerifyRefused("malformed", "draw.block_hash is not 0x + 64 lowercase hex")
    suite, sentinel = evidence.get("suite"), evidence.get("sentinel")
    if not isinstance(suite, str) or not suite:
        raise VerifyRefused("malformed", "suite is not a non-empty str")
    if sentinel is not None and not isinstance(sentinel, str):
        raise VerifyRefused("malformed", "sentinel is neither None nor a str")
    refs, rows = evidence.get("refs"), evidence.get("rows")
    if not isinstance(refs, dict) or sorted(refs) != sorted(ROLES):
        raise VerifyRefused("malformed", f"refs must be exactly {list(ROLES)} "
                            "(<repo>@sha256:<64 hex>)")
    if not isinstance(rows, dict):
        raise VerifyRefused("malformed", "rows is not an object")
    digests = {}
    for role in ROLES:
        rec = rows.get(role)
        dg = rec.get("digest") if isinstance(rec, dict) else None
        if not isinstance(dg, str) or not HEX64_RE.fullmatch(dg):
            raise VerifyRefused("malformed", f"rows.{role}.digest is not 64 hex")
        m = REF_RE.fullmatch(refs[role]) if isinstance(refs[role], str) else None
        if m is None:
            raise VerifyRefused("malformed", f"refs.{role} {refs[role]!r} is not "
                                "<repo>@sha256:<64 hex>")
        if m.group(2) != dg:
            raise VerifyRefused("malformed", f"refs.{role} does not pin rows.{role}.digest")
        digests[role] = dg
    return {"netuid": netuid, "block": block, "block_hash": bh, "suite": suite,
            "sentinel": sentinel, "refs": dict(refs), "digests": digests}


# ── the live verify ──────────────────────────────────────────────────────────


def verify_live(evidence: dict, *, chain, registry, pointers, now: datetime | None = None
                ) -> dict:
    """Re-check ``evidence`` against the chain, the checkpoint registry and the
    KING_D / house pointers, through read-only adapters:

    * ``chain.finalized_block() -> int`` and ``chain.block_hash(block) -> "0x…"``;
    * ``registry.resolve(ref) -> "sha256:<hex>" | None``;
    * ``pointers.at_block(block) -> {"house": ref, "king_d": ref}``.

    Returns the ``verified_live`` record; never mutates ``evidence``. Raises
    :class:`VerifyRefused`."""
    f = preflight(evidence)
    block = f["block"]

    try:
        finalized = chain.finalized_block()
    except Exception as e:  # noqa: BLE001 — finality unreadable: fail closed
        raise VerifyRefused("block_not_finalized", f"cannot read the finalized block "
                            f"({type(e).__name__}: {e})") from None
    if isinstance(finalized, bool) or not isinstance(finalized, int):
        raise VerifyRefused("block_not_finalized", f"the finalized block {finalized!r} is "
                            "not an int")
    if block > finalized:
        raise VerifyRefused("block_not_finalized", f"draw.block {block} is above the "
                            f"finalized block {finalized}")

    try:
        chain_hash = chain.block_hash(block)
    except Exception as e:  # noqa: BLE001 — an unreadable block can't vouch for the draw
        raise VerifyRefused("block_hash_mismatch", f"cannot read the hash of block {block} "
                            f"({type(e).__name__}: {e})") from None
    if not isinstance(chain_hash, str) or chain_hash.lower() != f["block_hash"]:
        raise VerifyRefused("block_hash_mismatch", f"block {block} has hash {chain_hash!r} "
                            f"on the chain, not draw.block_hash {f['block_hash']!r}")

    try:
        want = drawn_suite(chain_hash.lower(), f["sentinel"])
    except (ValueError, TypeError) as e:
        raise VerifyRefused("draw_mismatch", f"gate_suite_for refused the chain's hash with "
                            f"sentinel {f['sentinel']!r}: {e}") from None
    if want != f["suite"]:
        raise VerifyRefused("draw_mismatch", f"the gate suite drawn from block {block}'s hash "
                            f"is {want!r}, not the evidence's suite {f['suite']!r}")

    resolved = {}
    for role in ROLES:
        ref, expect = f["refs"][role], "sha256:" + f["digests"][role]
        try:
            got = registry.resolve(ref)
        except Exception as e:  # noqa: BLE001 — an unreadable ref is not shown published
            raise VerifyRefused("ref_unpublished", f"refs.{role} {ref!r}: the registry read "
                                f"failed ({type(e).__name__}: {e})") from None
        if got != expect:
            raise VerifyRefused("ref_unpublished", f"refs.{role} {ref!r} resolves to {got!r} "
                                f"in the registry, not {expect!r}")
        resolved[role] = got

    try:
        at = pointers.at_block(block)
    except VerifyRefused:
        raise
    except Exception as e:  # noqa: BLE001 — no pointer read: the refs can't be confirmed
        raise VerifyRefused("pointer_mismatch", f"cannot read the KING_D / house pointers at "
                            f"block {block} ({type(e).__name__}: {e})") from None
    if not isinstance(at, dict):
        raise VerifyRefused("pointer_mismatch", f"the pointers at block {block} are "
                            f"{at!r}, not an object")
    for ref_role in REFS:
        if at.get(ref_role) != f["refs"][ref_role]:
            raise VerifyRefused("pointer_mismatch", f"the {ref_role} pointer at block {block} "
                                f"is {at.get(ref_role)!r}, not refs.{ref_role} "
                                f"{f['refs'][ref_role]!r}")

    return {"schema": VERIFY_SCHEMA, "verified": True, "body_sha256": body_sha256(evidence),
            "network": NETWORK, "netuid": f["netuid"], "block": block,
            "block_hash": f["block_hash"], "finalized_block": finalized,
            "suite": f["suite"], "sentinel": f["sentinel"], "refs": copy.deepcopy(f["refs"]),
            "pointers": {r: at[r] for r in REFS}, "resolved": resolved,
            "verified_at": _iso_now(now), "tool": tool_info()}


# ── the CLI's read-only adapters ─────────────────────────────────────────────


class ChainReads:
    """The two chain reads the verify uses, over a wallet-less
    :class:`cascade.shared.chain.ChainClient` (lazy connection; no write
    method is exposed)."""

    def __init__(self, netuid: int, network: str = NETWORK):
        from cascade.shared.chain import ChainClient

        self._client = ChainClient(netuid=netuid, network=network)

    def finalized_block(self) -> int:
        return self._client.finalized_block()

    def block_hash(self, block: int) -> str:
        return self._client.block_hash(block)


class HubRegistry:
    """``resolve(ref)``: HEAD the OCI manifest ``ref`` pins on the Hippius Hub
    registry (the read path ``hippius_hub.revision_exists`` uses) and return
    its ``Docker-Content-Digest``; ``None`` on a 404. Pull-scoped token,
    anonymous unless Hub credentials are in the environment."""

    def resolve(self, ref: str) -> str | None:
        from hippius_hub._oci import head_manifest
        from hippius_hub.auth import get_oci_bearer_token
        from hippius_hub.constants import resolve_registry

        from cascade.shared.hippius import _resolve_hub_token_for_pull

        m = REF_RE.fullmatch(ref or "")
        if m is None:
            raise ValueError(f"not a <repo>@sha256:<hex> ref: {ref!r}")
        repo, digest = m.group(1), "sha256:" + m.group(2)
        token = _resolve_hub_token_for_pull(f"Resolving {ref}")
        oci_token = get_oci_bearer_token(repo, token, push=False)
        resp = head_manifest(resolve_registry(None), repo, digest, oci_token)
        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise RuntimeError(f"registry answered HTTP {resp.status_code} for {ref}")
        got = resp.headers.get("Docker-Content-Digest")
        if not isinstance(got, str) or not MANIFEST_DIGEST_RE.fullmatch(got):
            raise RuntimeError(f"registry gave no sha256 Docker-Content-Digest for {ref}")
        return got


class NoPointerSource:
    """The KING_D / house pointers at a historical block. Main publishes no
    read-only per-block source for them: lane D is unarmed, KING_D lives only
    in the validator-local ``lane_d_state.json`` (no per-block history), and
    lane A's king checkpoint at a block has no signed, block-indexed reader.
    Reading it is a typed refusal, so the CLI never writes a record whose
    pointers weren't read."""

    def at_block(self, block: int) -> dict:
        raise VerifyRefused(POINTERS_UNAVAILABLE, f"no read-only source of the KING_D / house "
                            f"pointers in force at block {block} exists on main; the chain "
                            "block, the draw and the registry refs were checked, but the "
                            "record is not written until a published pointer source lands")


# ── CLI ──────────────────────────────────────────────────────────────────────


def _err(msg: str) -> None:
    print(f"bench_evidence verify: {msg}", file=sys.stderr)


def verify_main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog=f"python -P -m {MODULE} verify",
                                 description="re-check W3b2 pod evidence against the chain, "
                                             "the checkpoint registry and the pointers "
                                             "(read-only)")
    ap.add_argument("--evidence", required=True)
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return 2 if e.code else 0
    path = Path(args.evidence)
    try:
        raw = path.read_bytes()
    except OSError as e:
        _err(f"cannot read the evidence file {str(path)!r}: {type(e).__name__}")
        return 2
    try:
        evidence = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        _err(f"the evidence file {str(path)!r} is not JSON")
        return 2
    try:
        f = preflight(evidence)
        record = verify_live(evidence, chain=ChainReads(f["netuid"]), registry=HubRegistry(),
                             pointers=NoPointerSource())
    except VerifyRefused as e:
        _err(f"refused: {e}")
        return 2
    except Exception as e:  # noqa: BLE001 — any failure is a refusal; the file stays as is
        _err(f"refused: failed: {type(e).__name__}: {e}")
        return 2
    evidence["verified_live"] = record
    tmp = path.with_name(path.name + ".verify-tmp")
    tmp.write_text(json.dumps(evidence, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    print(json.dumps({"verified": True, "evidence": str(path),
                      "body_sha256": record["body_sha256"], "block": record["block"],
                      "suite": record["suite"], "verified_at": record["verified_at"]}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "verify":
        _err("usage: python -P -m cascade.laned.bench_evidence verify --evidence FILE")
        return 2
    return verify_main(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
