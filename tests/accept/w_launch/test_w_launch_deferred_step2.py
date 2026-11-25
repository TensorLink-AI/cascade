"""W-LAUNCH lane D step 2 and its 2/3 upgrade gate (owner 2026-11-24, 22:30).

Lane D goes live AT LAUNCH (9586800) with lanes B and C. Step 1
(``launch-20261125.pre.*``) and then step 2 (``launch-20261125.*``, with
``[lane_d] from_block = 9586800``, D 50 % from lane A, ``public_weight = 0.0``)
are both published at least one era before the launch. Step 2 must clear the
gate: >= 2/3 of stake signals the release that carries the #913 satellite fix.

- ``deploy/overlays/lane-d-step2.{mainnet,testnet}.toml`` are FALLBACKS for a
  step 2 that misses the launch: the same body with
  ``[lane_d] from_block = "next-era"``;
- ``cascade-laned overlay publish --effective-block next-era --head-block H``
  resolves it to the first era start at least one era past H, and writes the
  resolved file that the printed validator command uses;
- the plan shows ``LANE D: OFF -> ON`` and ``LANE A SHARE: 0.5 -> 0.0``;
- arming lane D is refused below 2/3 of stake signalling the release
  (``cascade.laned.upgrade_tracker``) unless ``--force-upgrade-threshold``,
  and the share is printed.

Owner tooling and overlay files only: no validator behaviour changes."""

from __future__ import annotations

import json
import tomllib

import pytest

from .helpers import MAINNET_LAUNCH, PAIRS, ROOT, TESTNET_LAUNCH, overrides, shipped_cfg

STEP2 = {"mainnet": "deploy/overlays/lane-d-step2.mainnet.toml",
         "testnet": "deploy/overlays/lane-d-step2.testnet.toml"}
ERA = {"mainnet": 3600, "testnet": 600}
LAUNCH = {"mainnet": MAINNET_LAUNCH, "testnet": TESTNET_LAUNCH}
FEATURE = "release-913"


def _api():
    """The publish tool with the deferred-step-2 support; on a base without
    it this is an assertion failure, not a crash."""
    from cascade.laned import overlay as ov

    assert hasattr(ov, "NEXT_ERA") and hasattr(ov, "resolve_next_era"), (
        "cascade-laned overlay publish cannot resolve 'next-era'")
    assert hasattr(ov, "upgrade_share"), "no upgrade-tracker gate in the publish tool"
    return ov


def _files(tmp_path, net, signed: int, total: int = 100):
    """step-1 body file (seq 1, in force), tracker metagraph + notes files."""
    from cascade.shared import netconfig as nc

    base_name, pre, _, launch = PAIRS[net]
    base = shipped_cfg(base_name)
    b1 = nc.next_overlay(overrides(pre), launch, None, network=nc.network_identity(base))
    prev = tmp_path / "step1.json"
    prev.write_text(json.dumps(b1), encoding="utf-8")
    meta = {"hk_on": {"stake": signed, "validator_permit": True},
            "hk_off": {"stake": total - signed, "validator_permit": True}}
    notes = {"hk_on": f"cascade-ready:1:{FEATURE}:0:0", "hk_off": None}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "notes.json").write_text(json.dumps(notes), encoding="utf-8")
    return base_name, str(prev)


def _run(capsys, tmp_path, net, signed, *extra, tracker=True, launch_step2=False):
    from cascade.laned.cli import main

    base_name, prev = _files(tmp_path, net, signed)
    if launch_step2:
        # the launch step 2 as published before the launch (>= one era ahead)
        head = LAUNCH[net] - 2 * ERA[net]
        rel, eff = PAIRS[net][2], str(LAUNCH[net])
    else:
        head = LAUNCH[net] + 3 * ERA[net] + 7
        rel, eff = STEP2[net], "next-era"
    argv = ["overlay", "publish", "--config", str(ROOT / base_name),
            "--overrides", str(ROOT / rel), "--effective-block", eff,
            "--head-block", str(head), "--previous", prev,
            "--resolved-out", str(tmp_path / "resolved.toml")]
    if tracker:
        argv += ["--upgrade-metagraph", str(tmp_path / "meta.json"),
                 "--upgrade-notes", str(tmp_path / "notes.json"),
                 "--upgrade-feature", FEATURE]
    rc = main(argv + list(extra))
    out = capsys.readouterr().out
    return rc, out, head


def _vals(out: str, tag: str) -> list[str]:
    return [ln[len(tag) + 1:] for ln in out.splitlines() if ln.startswith(tag + " ")]


@pytest.mark.parametrize("net", sorted(STEP2))
def test_deferred_step2_file_is_the_launch_step2_with_a_next_era_start(net):
    _api()
    p = ROOT / STEP2[net]
    assert p.is_file(), f"missing {STEP2[net]}"
    step2 = tomllib.loads(p.read_text())
    _, pre, full, _ = PAIRS[net]
    want = overrides(full)
    assert step2["lane_d"].pop("from_block") == "next-era"
    want["lane_d"].pop("from_block")
    assert step2 == want                                 # same body otherwise
    # tests-correction (owner 2026-11-25, leak mitigation / cascade#935): step 2
    # also carries [lane_d] embargo_hours = 24 (from the shipped 48).
    assert step2["lane_d"] == {"share": 0.5, "takes_from": "lane_a", "public_weight": 0.0,
                               "embargo_hours": 24.0}
    # step 1's keys restated byte-identically (R3 `non-lane-d`)
    assert {k: v for k, v in step2.items() if k != "lane_d"} == overrides(pre)
    # the launch step 2 is live again (owner 2026-11-24 22:30): lane D at launch
    assert overrides(full)["lane_d"]["from_block"] == LAUNCH[net]
    assert "SUPERSEDED" not in (ROOT / full).read_text()


@pytest.mark.parametrize("net", sorted(STEP2))
def test_publish_resolves_next_era_and_shows_lane_d_on_lane_a_zero(net, tmp_path, capsys):
    _api()
    rc, out, head = _run(capsys, tmp_path, net, signed=70)
    assert rc == 0, out
    era = ERA[net]
    want = -(-(head + era) // era) * era                  # first era start >= head + era
    assert want - head >= era and want % era == 0
    resolved = tomllib.loads((tmp_path / "resolved.toml").read_text())
    assert resolved["lane_d"]["from_block"] == want
    assert _vals(out, "EFFECTIVE_BLOCK") == [str(want)]
    assert _vals(out, "CHECK") == ["VALID"], out
    lanes = _vals(out, "LANE")
    assert any(x.startswith("LANE D: OFF -> ON") for x in lanes), lanes
    assert any(x.startswith("LANE A SHARE: 0.5 -> 0.0") for x in lanes), lanes
    assert any(x.startswith("70.0% of eligible stake") for x in _vals(out, "UPGRADE")), out
    cmd = _vals(out, "COMMAND")[0]
    assert str(tmp_path / "resolved.toml") in cmd and f"--netconfig-effective-block {want}" in cmd
    # the resolved step 2 still passes the R3 lint after step 1
    from cascade.laned import overlay_lint as L
    from cascade.shared import netconfig as nc

    base = shipped_cfg(PAIRS[net][0])
    b1 = json.loads((tmp_path / "step1.json").read_text())
    b2 = nc.next_overlay(resolved, want, b1, network=nc.network_identity(base))
    assert L.lint(b2, b1, base, None) == []


def test_publish_refuses_lane_d_below_two_thirds_unless_forced(tmp_path, capsys):
    _api()
    rc, out, _ = _run(capsys, tmp_path, "mainnet", signed=66)
    assert rc == 1, out
    assert any(x.startswith("66.0% of eligible stake") for x in _vals(out, "UPGRADE")), out
    assert "--force-upgrade-threshold" in _vals(out, "CHECK")[0]
    assert "COMMAND" not in out
    rc, out, _ = _run(capsys, tmp_path, "mainnet", 66, "--force-upgrade-threshold")
    assert rc == 0, out
    assert any("--force-upgrade-threshold" in w for w in _vals(out, "WARNING")), out
    # no tracker inputs at all: refused, never assumed
    rc, out, _ = _run(capsys, tmp_path, "mainnet", signed=90, tracker=False)
    assert rc == 1 and _vals(out, "UPGRADE")[0].startswith("unknown"), out


@pytest.mark.parametrize("net", sorted(STEP2))
def test_launch_step2_is_gated_before_the_launch(net, tmp_path, capsys):
    """Lane D live at launch: step 2 (from_block = the launch block) after
    step 1, both an era ahead; the 2/3 gate applies to it."""
    _api()
    rc, out, _ = _run(capsys, tmp_path, net, 60, launch_step2=True)
    assert rc == 1 and "--force-upgrade-threshold" in _vals(out, "CHECK")[0], out
    assert any(x.startswith("60.0% of eligible stake") for x in _vals(out, "UPGRADE")), out
    rc, out, _ = _run(capsys, tmp_path, net, 67, launch_step2=True)
    assert rc == 0, out
    assert _vals(out, "SEQ") == ["2"] and _vals(out, "EFFECTIVE_BLOCK") == [str(LAUNCH[net])]
    assert _vals(out, "RESOLVED") == []                    # nothing to resolve
    lanes = _vals(out, "LANE")
    assert any(x.startswith("LANE D: OFF -> ON") for x in lanes), lanes
    assert any(x.startswith("LANE A SHARE: 0.5 -> 0.0") for x in lanes), lanes
    assert f"--netconfig-overlay {ROOT / PAIRS[net][2]}" in _vals(out, "COMMAND")[0]


def test_next_era_needs_a_head_and_only_fits_lane_d_from_block():
    ov = _api()
    base = shipped_cfg("chain.toml")
    with pytest.raises(ov.Invalid):
        ov.resolve_next_era(base, {"lane_d": {"from_block": "next-era"}}, None, 10)
    with pytest.raises(ov.Invalid):
        ov.resolve_next_era(base, {"lane_b": {"from_block": "next-era"}}, MAINNET_LAUNCH, 10)
    same = {"lane_d": {"from_block": MAINNET_LAUNCH}}
    assert ov.resolve_next_era(base, same, None, MAINNET_LAUNCH) == (same, MAINNET_LAUNCH, [])
