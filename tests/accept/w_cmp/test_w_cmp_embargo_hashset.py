"""W-CMP (DEC-CA-0075) items 2 and 3: the embargo in force at a leg's
DISPATCH rides its signed record and the draw uses it (so ``embargo_days``
is overlay-settable within bounds); the lane B contamination hash set is
named by content hash + storage key, fetched and sha256-verified before use,
fail closed on a mismatch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from .helpers import ROOT, overlay, parse, shipped_raw

TT = 1_790_000_000.0
HASH = "0x" + "ab" * 32


def _entry(**rec):
    return SimpleNamespace(marketcast={"train_time": TT, **rec})


def test_w_cmp_draw_uses_the_recorded_dispatch_embargo():
    from cascade.validator.marketcast import entry_draw, entry_embargo_mc

    mc = parse(shipped_raw()).marketcast
    rec = {"embargo_days": int(mc.embargo_days), "embargo_eval_days": int(mc.embargo_eval_days)}
    want = entry_draw(_entry(**rec), mc, phase="judge", block=5, block_hash=HASH)
    # a later overlay raised the embargo: the recorded leg's draw is unchanged
    later = replace(mc, embargo_days=60, embargo_eval_days=50)
    assert entry_draw(_entry(**rec), later, phase="judge", block=5, block_hash=HASH) == want
    # a legacy entry (no record) is drawn on the config, as before
    assert entry_draw(_entry(), mc, phase="judge", block=5, block_hash=HASH) == want
    assert entry_draw(_entry(), later, phase="judge", block=5,
                      block_hash=HASH)["eval_days"] == 50
    assert entry_embargo_mc(_entry(embargo_days=True), mc) is mc     # malformed: ignored


def test_w_cmp_embargo_is_overlay_settable_within_bounds():
    import cascade.shared.competition_spec  # noqa: F401  (W-CMP)
    from cascade.shared.config import load_chain_config
    from cascade.shared.netconfig_bounds import value_refusals

    base = load_chain_config(ROOT / "chain.toml")
    out, why = overlay(base, {"marketcast": {"embargo_days": 35}})
    assert why is None, why
    assert out.marketcast.embargo_days == 35
    assert value_refusals({"marketcast": {"embargo_days": 1}})
    assert value_refusals({"marketcast": {"embargo_days": 500}})


def test_w_cmp_trainer_stamps_the_dispatch_embargo_into_the_signed_record():
    import cascade.shared.competition_spec  # noqa: F401  (W-CMP)

    src = (ROOT / "cascade/trainer/lane_c_driver.py").read_text()
    assert 'leg["embargo_days"] = int(mc_.embargo_days)' in src
    assert '"embargo_eval_days")' in src


def _artefact(tmp_path):
    """A tiny WB11 hash-set artefact (written by the WB11 writer) + its pin."""
    import numpy as np

    from cascade.eval import contamination as C

    suite = sorted(C.KNOWN_SUITES)[0]
    vals = [float(x) for x in np.random.default_rng(0).normal(size=256)]
    art = C.build_hashset({suite: {"version": "rev1", "series": {"item": vals}}}, version=3)
    p = tmp_path / "src.json"
    pin = C.write_artefact(art, p)
    raw = p.read_bytes()
    assert pin == "sha256:" + hashlib.sha256(raw).hexdigest()
    return raw, pin


def test_w_cmp_hashset_is_fetched_and_verified_by_content_hash(tmp_path):
    from cascade.interface.contamination_scan import HashsetMismatch, load_pinned_hashset

    raw, pin = _artefact(tmp_path)
    calls = []

    def fetch(key):
        calls.append(key)
        return raw

    hs = load_pinned_hashset(pin, tmp_path / "release", key="contamination/hs.json",
                             fetch=fetch, cache_dir=tmp_path / "cache")
    assert hs.pin == pin and calls == ["contamination/hs.json"]
    load_pinned_hashset(pin, tmp_path / "release", key="contamination/hs.json",
                        fetch=fetch, cache_dir=tmp_path / "cache")
    assert len(calls) == 1                               # cached at the pin
    with pytest.raises(HashsetMismatch, match="not the pinned"):
        load_pinned_hashset(pin, tmp_path / "release", key="k", fetch=lambda k: raw + b" ",
                            cache_dir=tmp_path / "cache2")
    assert not list((tmp_path / "cache2").glob("*.json"))
    with pytest.raises(ValueError, match="matches no artefact"):
        load_pinned_hashset(pin, tmp_path / "release")


def test_w_cmp_hashset_key_parses_and_overlay_accepts_the_pin():
    import copy

    import cascade.shared.competition_spec  # noqa: F401  (W-CMP)

    raw = shipped_raw()
    r = copy.deepcopy(raw)
    r["lane_b"]["contamination_hashset_key"] = "contamination/hashset-v3.json"
    assert parse(r).lane_b.contamination_hashset_key == "contamination/hashset-v3.json"
    for bad in ("../x", "/abs", 3):
        b = copy.deepcopy(raw)
        b["lane_b"]["contamination_hashset_key"] = bad
        with pytest.raises(ValueError, match="contamination_hashset_key"):
            parse(b)
    from cascade.shared import netconfig as nc

    nc.check_overrides({"lane_b": {"contamination_hashset": "sha256:" + "0" * 64,
                                   "contamination_hashset_key": "contamination/h.json"}})
    assert json.dumps(nc.overlay_policy())                # still JSON-serialisable
