"""DEC-CA-0056 lane C — the pool CLI (lane A unchanged; lane C has no pool of
its own since MarketCast-only lane C, owner 2026-10-20) and the miner CLI
(`cascade lane-c`): local refusals and signed fund headers the intake
accepts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from cascade.funding.intake import FundingIntake
from cascade.funding.queue import FundedQueue, LaneCQueue, lane_c_queue_path, split_lane_c_ref
from cascade.funding.vault import PayerKeyVault
from cascade.shared.config.lane_c import LaneCConfig
from cascade.shared.hippius import ObjectNotFound, PoolSnapshotMeta

M, V = "mkt:returns", "mkt:vol"
SU = ("returns", "vol")
REF = "org/recipe@sha256:" + "a" * 64


class MemStore:
    def __init__(self):
        self.objs: dict[str, bytes] = {}

    def put_bytes(self, key, data, **kw):
        self.objs[key] = bytes(data)

    def put_text(self, key, text, **kw):
        self.objs[key] = text.encode()

    def get_bytes(self, key):
        if key not in self.objs:
            raise ObjectNotFound(key)
        return self.objs[key]

    def get_text(self, key):
        return self.get_bytes(key).decode()


@pytest.fixture
def lc_cfg(cfg):
    return replace(cfg, lane_c=LaneCConfig(targets=(M, V)))


def test_publish_lane_a_is_unchanged(lc_cfg, tmp_path, monkeypatch):
    from cascade.pool import cli
    from cascade.shared import hippius

    store = MemStore()
    monkeypatch.setattr(cli, "load_chain_config", lambda p=None: lc_cfg)
    monkeypatch.setattr(hippius, "pool_s3_store", lambda storage, **kw: store)
    assert cli.main(["publish", "--sources", "synthetic", "--effective-block", "1800",
                     "--out", str(tmp_path / "stage")]) == 0
    assert set(store.objs) == {"pool/snapshots/block-1800.tar", "pool/index.json"}
    prov = json.loads((tmp_path / "stage" / "provenance.json").read_text())
    assert "lane-c" not in json.dumps(prov)


def test_reveal_dir_and_prefix_resolution(lc_cfg):
    import argparse

    from cascade.pool.cli import pool_prefix
    from cascade.pool.reveal import reveal_dir_name

    meta = PoolSnapshotMeta(effective_block=7, key="k", sha256="s", size_bytes=1,
                            as_of="2026-10-19", n_series=1, context_length=1, horizon=1)
    assert reveal_dir_name(meta) == "snapshots/2026-10-19-block-7"
    assert reveal_dir_name(meta, "pool_mkt") == "pool_mkt/snapshots/2026-10-19-block-7"
    ns = argparse.Namespace
    assert pool_prefix(ns(lane="a", prefix=None), lc_cfg) == "pool"
    assert pool_prefix(ns(lane="mkt", prefix=None), lc_cfg) == "pool_mkt"
    assert pool_prefix(ns(lane="mkt", prefix="custom"), lc_cfg) == "custom"


# ── miner CLI ────────────────────────────────────────────────────────────────


def test_recipe_error_refuses_generic_and_inactive_targets(lc_cfg, tmp_path):
    from cascade.miner.lane_c import recipe_error

    d = tmp_path / "tree"
    d.mkdir()
    why = recipe_error(lc_cfg, d, "domain:energy")
    assert "forward-only" in why and "MarketCast (mkt:)" in why
    assert "not active" in recipe_error(lc_cfg, d, "mkt:returns")     # [marketcast] off
    assert "--target" in recipe_error(lc_cfg, d, None)


def test_cli_check_command_refuses_a_generic_target(lc_cfg, tmp_path, monkeypatch):
    from cascade.miner import cli
    from cascade.miner import lane_c as L
    from cascade.shared import config as shared_config

    monkeypatch.setattr(shared_config, "load_chain_config", lambda p=None: lc_cfg)
    d = tmp_path / "tree"
    d.mkdir()
    assert cli.main(["lane-c", "check", str(d), "--target", "domain:energy", "--json"]) == 1
    assert L.add_lane_c is not None


def test_fund_headers_are_accepted_by_the_intake(tmp_path):
    from cascade.miner.lane_c import build_lane_c_headers

    clock = lambda: 1_000_000.0  # noqa: E731
    lane_c = LaneCQueue(lane_c_queue_path(tmp_path / "q.json"), clock=clock)
    intake = FundingIntake(
        FundedQueue(tmp_path / "q.json", clock=clock), PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, r: None, require_signature=True,
        verify=lambda hk, msg, sig: bytes.fromhex(sig) == msg, clock=clock,
        lane_c_queue=lane_c, resolve_lane_c_reveal=lambda hk, t, r: 9,
        lane_c_targets=(M,), mkt_suites=SU)
    h = build_lane_c_headers("lane-c-fund", "HK", M, REF, "sk-1",
                             lambda m: m, now=clock, label="ft-1")
    status, body = intake.lane_c_fund(h)
    assert (status, body["status"]) == (202, "queued")
    assert split_lane_c_ref(lane_c.get("HK").ref) == (M, REF)
    assert lane_c.get("HK").label == "ft-1"
    # a header set signed for another target is refused
    h2 = build_lane_c_headers("lane-c-fund", "HK", M, REF, "sk-1",
                              lambda m: m, now=lambda: 1_000_001.0)
    h2["X-Lane-C-Target"] = V
    intake.lane_c_targets = (M, V)
    assert intake.lane_c_fund(h2)[1]["code"] == "bad_signature"
    # a generic target never reaches the queue
    h3 = build_lane_c_headers("lane-c-fund", "HK", "domain:energy", REF, "sk-1",
                              lambda m: m, now=lambda: 1_000_002.0)
    status, body = intake.lane_c_fund(h3)
    assert (status, body["code"]) == (400, "bad_target") and "forward-only" in body["message"]
