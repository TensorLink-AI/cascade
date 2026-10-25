"""DEC-CA-0055: the signed ``/v1/credentials`` key rotation, and the audit's
seed-epoch shape check."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.request
from dataclasses import replace
from pathlib import Path

from cascade.audit.checks import _epoch_start_for_seed
from cascade.funding.intake import FundingIntake, canonical_fund_message
from cascade.funding.queue import FundedQueue
from cascade.funding.vault import PayerKeyVault
from cascade.shared.config import load_chain_config
from cascade.shared.era import era_for_block, era_length_blocks, v2_first_era, v2_first_warm_era

ROOT = Path(__file__).resolve().parents[2]
HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "cascade-gen-abc123@sha256:" + "a" * 64


class _Clock:
    t = 1_000_000.0

    def __call__(self):
        return self.t


def _intake(tmp_path, *, resolve=lambda hk, ref: 500, registered=True, sign=False):
    clock = _Clock()
    signed: list[bytes] = []

    def verify(hk, msg, sig):
        signed.append(msg)
        return sig == "good"

    it = FundingIntake(FundedQueue(tmp_path / "q.json", clock=clock),
                       PayerKeyVault(dir=None, clock=clock), resolve_reveal=resolve,
                       require_signature=sign, verify=verify, clock=clock,
                       resolve_registered=lambda hk: registered)
    return it, signed


def _headers(key="sk-new", **extra):
    return {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": key, **extra}


def test_rotation_replaces_the_vaulted_key_and_queues_nothing(tmp_path):
    it, _ = _intake(tmp_path)
    it.vault.insert(HK, "sk-old")
    status, body = it.credentials(_headers())
    assert (status, body) == (200, {"status": "stored"})
    assert it.vault.get(HK) == "sk-new"
    assert it.queue.get(HK) is None                      # nothing queued
    # repeatable at any time (never the one lifetime submission)
    assert it.credentials(_headers("sk-3"))[0] == 200 and it.vault.get(HK) == "sk-3"


def test_rotation_gates(tmp_path):
    it, _ = _intake(tmp_path, resolve=lambda hk, ref: None)
    assert it.credentials(_headers())[1]["code"] == "not_revealed"
    it, _ = _intake(tmp_path, registered=False)
    assert it.credentials(_headers())[1]["code"] == "not_registered"
    it, _ = _intake(tmp_path)
    assert it.credentials({"X-Miner-Hotkey": HK, "X-Commit-Ref": REF})[1]["code"] == \
        "missing_lium_api_key"


def test_rotation_signature_binds_the_key_digest(tmp_path):
    it, signed = _intake(tmp_path, sign=True)
    ts = str(int(_Clock.t))
    status, body = it.credentials(_headers(**{"X-Timestamp": ts, "X-Signature": "good"}))
    assert status == 200
    digest = hashlib.sha256(b"sk-new").hexdigest()
    assert signed[-1] == canonical_fund_message("credentials", HK, REF, ts, digest)
    # a replay of the same timestamp is refused
    status, body = it.credentials(_headers(**{"X-Timestamp": ts, "X-Signature": "good"}))
    assert status == 409
    status, body = it.credentials(_headers(**{"X-Timestamp": str(int(_Clock.t) + 1),
                                              "X-Signature": "bad"}))
    assert body["code"] == "bad_signature"


def test_route_is_wired(tmp_path):
    it, _ = _intake(tmp_path)
    srv = it.make_server("127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/credentials", data=b"",
                                     headers=_headers(), method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200 and json.loads(resp.read())["status"] == "stored"
    finally:
        srv.shutdown()
        srv.server_close()
    assert it.vault.get(HK) == "sk-new"


def test_audit_accepts_only_legal_seed_epoch_starts():
    cfg = load_chain_config(ROOT / "chain.toml")
    era = era_length_blocks(cfg.round, 0)
    gate = 10_000 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate),
                  training=replace(cfg.training, v2_from_block=gate))
    first, warm = v2_first_era(cfg), v2_first_warm_era(cfg)
    e = era_for_block(cfg.round, (warm + 4) * era)
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(first - 1) * era)) == first
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(warm + 1) * era)) == warm + 2
    # a start inside the random-init window (not the first era) is no warm start
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(first + 3) * era)) is None
    # a later era, or a block off the era grid, never
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(warm + 5) * era)) is None
    assert _epoch_start_for_seed(cfg, replace(e, seed_block=(first - 1) * era + 1)) is None
    off = replace(cfg, v2=replace(cfg.v2, king_ladder=False))
    assert _epoch_start_for_seed(off, replace(e, seed_block=(first - 1) * era)) is None
