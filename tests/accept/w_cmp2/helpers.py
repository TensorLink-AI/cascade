"""Shared helpers for the W-CMP2 acceptance tests (DEC-CA-0075, plan rev6.25):
a NEW lane-C competition (an ``energy`` spec instance of the multi-instance
MarketCast runner) judged end to end, next to an untouched ``mkt``."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
KIND = "energy"
EB = 7200
BLOCK = 20 * EB
T0 = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp())
ANCHOR_T = T0 + 200_000
ASSETS = ("DE", "FR", "NL", "BE", "AT", "CH")
UIDS = {"alice": 7, "carol": 9}
REF = {hk: f"{hk}/recipe@sha256:" + hashlib.sha256(hk.encode()).hexdigest() for hk in UIDS}
SIGMA = 10.0
FACTORS = {"alice": 1.0, "carol": 3.0}

MC_RAW = {
    "from_block": EB, "suites": ["returns"],
    "frequencies": {"returns": ["1m", "5m"]}, "horizons": {"returns": ["1m", "5m"]},
    "eval_context_length": 64, "windows_per_cell": 30, "bootstrap_B": 1000,
    "min_windows": 100, "min_clusters": 5, "start_margin_seconds": 60,
    "verdict_wait_blocks": 2 * EB, "max_infra_deferrals": 2,
    "min_time_blocks": 4, "min_dollar_volume_24h": 0.0,
    "coverage_tolerance": 0.1, "coverage_tolerance_over": 0.15,
    "forward_horizons": ["1m", "5m"],
}


def register_kind():
    from cascade.shared.config.lane_c import register_spec_kind

    register_spec_kind(KIND)


def unregister_kind():
    """Tests only: forget the instance kind so other tests see the shipped
    registry (the kind registry is process-global)."""
    from cascade.shared.config import lane_c as L
    from cascade.trainer import lane_c_competitions as TC
    from cascade.validator import lane_c_competitions as VC

    L.SPEC_KINDS.discard(KIND)
    L.COMPETITION_KINDS.pop(KIND, None)
    VC._SPEC_INSTANCES.pop(KIND, None)
    TC._SPEC_TRAINERS.pop(KIND, None)


def mkt_mc():
    from cascade.shared.config.marketcast import parse_marketcast

    return parse_marketcast({**MC_RAW, "assets": {"crypto": ["BTC", "ETH", "SOL", "XRP",
                                                             "HYPE", "BNB"]}})


def instance_mc():
    from cascade.shared.config.marketcast import parse_marketcast

    return replace(parse_marketcast({**{k: v for k, v in MC_RAW.items() if k != "from_block"},
                                     "assets": {"power": list(ASSETS)},
                                     "pool_select": {"domain": "energy"},
                                     "pool_channels": {"close": "load"}}, open_classes=True),
                   from_block=EB)


def lane_c(with_instance: bool):
    from cascade.shared.config.lane_c import LaneCCompetition, parse_lane_c

    targets = ["mkt:returns"] + ([f"{KIND}:returns"] if with_instance else [])
    lc = replace(parse_lane_c({"targets": ["mkt:returns"], "bootstrap_B": 1000},
                              mkt_suites=("returns",)), from_block=EB, targets=tuple(targets))
    if not with_instance:
        return lc
    return replace(lc, competitions=(LaneCCompetition(KIND, from_block=EB),
                                     LaneCCompetition("mkt")),
                   instances=((KIND, instance_mc(), "{}"),))


def energy_pool(path: Path, minutes: int):
    """A generic standard pool from an 'energy builder': hourly-agnostic
    minute grid, channel ``load`` (read as the close), class / asset tags."""
    from cascade.shared.std_pool import write_std_pool

    rng = np.random.default_rng(11)
    series, arrays = {}, {}
    for a in ASSETS:
        r = rng.normal(0, SIGMA / 1e4, 12_000)[:minutes]
        c = 100.0 * np.exp(np.cumsum(r))
        sid = f"grid/{a}"
        ch = {}
        for name, arr in (("load", c), ("open", c), ("high", c), ("low", c),
                          ("volume", np.ones(minutes))):
            rel = f"grid/{a}/{name}.npy"
            arrays[rel] = arr
            ch[name] = rel
        series[sid] = {"grid": {"t0": T0, "step_seconds": 60, "steps": int(minutes)},
                       "channels": ch, "freq": "1m",
                       "tags": {"domain": "energy", "source": "entsoe", "class": "power",
                                "asset": a}}
    write_std_pool(path, pool_id="energy-demo", series=series, arrays=arrays,
                   clock={"anchor_block": BLOCK, "anchor_time": float(ANCHOR_T),
                          "block_seconds": 12.0})
    from cascade.pool.marketcast import load_snapshot

    return load_snapshot(path, verify_signature=False, select={"domain": "energy"},
                         channels={"close": "load"})


def fixture_cfg(tmp_path):
    """The unit suite's ``cfg`` fixture (tests/conftest.py), rebuilt so the
    test also runs with ``--noconftest``."""
    from cascade.shared.config import load_chain_config
    from tests.unit.predecision_main import predecision_path

    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    c = load_chain_config(predecision_path(tmp_path))
    raw = copy.deepcopy(c.raw)
    raw["v2"].update(launch_at_utc="", commit_open_at_utc="", commit_from_block=0,
                     image_decoupled_from_block=0)
    raw["netconfig"]["owner_hotkey"] = ""
    raw["scoring"].update(forfeit_from_block=0, margin_v2_from_block=0)
    return replace(c, training=replace(c.training, expected_gpu="", train_image_digest=""),
                   raw=raw, netconfig=replace(c.netconfig, owner_hotkey=""),
                   v2=replace(c.v2, launch_at_utc="", commit_open_at_utc="",
                              commit_from_block=0, prelaunch_burn_readiness="",
                              image_decoupled_from_block=0),
                   generator=replace(c.generator, corpus_target_points=0, min_channels=1,
                                     sandbox_strict=False),
                   round=replace(c.round, commit_floor_block=0, dedup_mode="off",
                                 epoch_blocks_prev=0, epoch_activation_block=0,
                                 funded_mode="off", funded_pods="off", funded_king_rent=False),
                   scoring=replace(c.scoring, win_margin_start_prev=0.0,
                                   margin_activation_block=0, forfeit_from_block=0,
                                   margin_v2_from_block=0),
                   manifest=replace(c.manifest, validator_hotkey=""))


def bhash(block):
    return "0x" + hashlib.sha256(f"block-{block}".encode()).hexdigest()


class Store:
    def __init__(self):
        self.objs: dict[str, object] = {}
        self.puts: list[str] = []

    def get_text(self, key):
        from cascade.shared.hippius import ObjectNotFound

        if key not in self.objs:
            raise ObjectNotFound(key)
        v = self.objs[key]
        return v if isinstance(v, str) else v.decode()

    def get_bytes(self, key):
        from cascade.shared.hippius import ObjectNotFound

        if key not in self.objs:
            raise ObjectNotFound(key)
        v = self.objs[key]
        return v if isinstance(v, bytes) else v.encode()

    def put_text(self, key, text, content_type="", acl=None):
        self.objs[key] = text
        self.puts.append(key)

    def put_bytes(self, key, data, content_type="", acl=None):
        self.objs[key] = bytes(data)
        self.puts.append(key)


def ptr(tag):
    from cascade.shared.manifest import format_trained_pointer

    return format_trained_pointer(f"cascade/ckpt-{tag}@sha256:"
                                  + hashlib.sha256(tag.encode()).hexdigest())


def tt(train_block):
    return float(ANCHOR_T + (int(train_block) - BLOCK) * 12)


def entry(hk, mc, target, *, train_block=BLOCK - 100):
    from cascade.shared.lane_c_manifest import LaneCEntry

    return LaneCEntry(miner_hotkey=hk, miner_uid=UIDS[hk], target=target,
                      recipe_ref=REF[hk], trained_pointer=ptr(hk + "-" + target.partition(":")[0]),
                      train_block=train_block, funded_by="miner",
                      marketcast={"base_ref": mc.base_ref, "base_sha256": mc.base_sha256,
                                  "checkpoint_sha256": hashlib.sha256(
                                      (hk + "-" + target.partition(":")[0]).encode()).hexdigest(),
                                  "train_time": tt(train_block),
                                  "embargo_scan": {"rule": "embargo-v1",
                                                   "result": "no_generator"}})


def history(entries):
    from cascade.shared.lane_c_manifest import format_finetune_commit

    return [SimpleNamespace(hotkey=e.miner_hotkey, commit_block=BLOCK - 500,
                            payload=format_finetune_commit(e.target, e.recipe_ref,
                                                           mkt_suites=("returns",)))
            for e in entries]
