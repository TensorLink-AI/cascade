"""W-OV #911 (b274626d testnet rollout, containment drills, 2026-11-24): the
owner's publish tool after a REFUSED overlay, and an overlay that drops a
restatement.

1. Nodes refuse a bad seq N+1 and keep seq N in force. The next overlay
   takes ``seq`` / ``prev`` from the latest PUBLISHED body (N+1) but is
   judged (retroactivity, effective changes, lane guard) against the body in
   force (N): ``plan_overlay(..., in_force=)``, :func:`newest_valid`, the
   wrapper's ``--in-force`` and the validator's ``--netconfig-in-force-seq``.
2. The plan lists every key whose EFFECTIVE value changes vs the config in
   force (a dropped restatement marked ``DROPPED``), lane on/off moves
   (``LANE D: ON -> OFF``) and share moves, and refuses an overlay that turns
   an armed lane off without setting its switch keys unless
   ``allow_lane_off`` / ``--allow-lane-off``.

Owner-side only: nothing here changes what a validator accepts or derives
(``validate_overlay`` / ``build_config`` / the resolver are untouched)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .helpers import ROOT, shipped_cfg

LAUNCH = 9586800                     # mainnet launch era start (chain.toml)
ERA = 3600
EFF = LAUNCH + 2 * ERA               # the kill overlay's era start
# launch step 2 shape: lane D armed, A0 / B35 / C15 / D50
STEP2 = {"lane_d": {"from_block": LAUNCH, "share": 0.5, "takes_from": "lane_a"},
         "lane_b": {"share": 0.35}, "lane_c": {"share": 0.15, "takes_from": "lane_a"}}
BAD = {"lane_d": {"min_windows": 40}}                       # out of the envelope


def _copy(x):
    return json.loads(json.dumps(x))


def _kill_restate() -> dict:
    ov = _copy(STEP2)
    ov["lane_c"]["until_block"] = EFF                     # end lane C, keep lane D
    return ov


def _kill_omit() -> dict:
    ov = _kill_restate()
    del ov["lane_d"]                                      # forgot to restate lane D
    return ov


def _chain(base):
    """seq 1 = step 2 (valid, in force), seq 2 = refused (published)."""
    from cascade.shared import netconfig as nc

    s1 = nc.next_overlay(_copy(STEP2), LAUNCH, None, network=nc.network_identity(base))
    s2 = nc.next_overlay(_copy(BAD), LAUNCH + ERA, s1, network=nc.network_identity(base))
    return s1, s2


def test_refused_latest_is_skipped_and_the_successor_chains_from_it():
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    s1, s2 = _chain(base)
    with pytest.raises(nc.NetConfigError):
        nc.validate_overlay(base, s2)                     # every node refuses seq 2
    in_force, skipped = nc.newest_valid(base, [s1, s2])
    assert in_force == s1
    assert len(skipped) == 1 and "seq 2" in skipped[0]
    assert nc.newest_valid(base, [s2]) == (None, skipped)

    # the #911 bug: judged against the refused body, the restated lane D
    # start "lies before" the apply block
    old = nc.plan_overlay(base, _kill_restate(), EFF, s2)
    assert not old.ok and any("lies before" in e for e in old.errors), old.render()

    plan = nc.plan_overlay(base, _kill_restate(), EFF, s2, in_force=s1)
    assert plan.ok, plan.render()
    assert plan.body["seq"] == 3 and plan.body["prev"] == nc.body_digest(s2)
    assert "seq 1" in plan.in_force and "seq 1" in plan.render()
    assert plan.changes == [f"[lane_c] until_block: <release default> -> {EFF}"], plan.changes
    assert any(x.startswith("LANE C: ON -> OFF (set by [lane_c] until_block")
               for x in plan.lanes), plan.lanes
    assert not any(x.startswith("LANE D:") for x in plan.lanes), plan.lanes


def test_dropped_restatement_shows_and_an_implicit_lane_off_is_refused():
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    s1, s2 = _chain(base)
    plan = nc.plan_overlay(base, _kill_omit(), EFF, s2, in_force=s1)
    assert not plan.ok, plan.render()
    assert any("LANE D" in e and "--allow-lane-off" in e for e in plan.errors), plan.errors
    assert "LANE D: ON -> OFF (IMPLICIT" in "\n".join(plan.lanes), plan.lanes
    assert "LANE D SHARE: 0.5 -> 0.0" in plan.lanes, plan.lanes
    dropped = [c for c in plan.changes if c.startswith("[lane_d] from_block:")]
    assert dropped and "DROPPED" in dropped[0] and str(LAUNCH) in dropped[0], plan.changes
    # the chain.toml diff alone hides it (the original report)
    assert not any("[lane_d] from_block" in d for d in plan.diff)

    allowed = nc.plan_overlay(base, _kill_omit(), EFF, s2, in_force=s1, allow_lane_off=True)
    assert allowed.ok, allowed.render()
    assert any("LANE D" in w and "allowed" in w for w in allowed.warnings)


def test_an_explicit_lane_d_kill_is_allowed_and_reported():
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    s1, _ = _chain(base)
    ov = _copy(STEP2)
    ov["lane_d"]["until_block"] = EFF
    plan = nc.plan_overlay(base, ov, EFF, s1)             # in_force defaults to previous
    assert plan.ok, plan.render()
    assert "LANE D: ON -> OFF (set by [lane_d] until_block)" in plan.lanes, plan.lanes


def test_share_moves_and_unchanged_restatements():
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    s1, _ = _chain(base)
    same = nc.plan_overlay(base, _copy(STEP2), EFF, s1)
    assert same.ok and same.changes == [] and same.lanes == [], same.render()
    ov = _copy(STEP2)
    ov["lane_b"]["share"] = 0.3
    ov["lane_d"]["share"] = 0.45
    moved = nc.plan_overlay(base, ov, EFF, s1)
    assert moved.ok, moved.render()
    assert "LANE B SHARE: 0.35 -> 0.3" in moved.lanes
    assert "LANE D SHARE: 0.5 -> 0.45" in moved.lanes
    assert "[lane_d] share: 0.5 -> 0.45" in moved.changes


def test_validate_overlay_and_build_are_unchanged_by_the_guard():
    """The guard is the owner's: the omitting overlay still VALIDATES on a node
    (DEC-CA-0057 snapshots), so the tool is the only place it can be caught."""
    from cascade.shared import netconfig as nc

    base = shipped_cfg()
    s1, s2 = _chain(base)
    body = nc.next_overlay(_kill_omit(), EFF, s2, network=nc.network_identity(base))
    built = nc.validate_overlay(base, body)
    assert int(built.lane_d.from_block) == 0


# ── the wrapper (cascade-laned overlay publish) ──────────────────────────────


def _write(tmp_path, name, obj) -> str:
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def _toml(tmp_path, name, overrides) -> str:
    from cascade.laned.overlay import overrides_toml

    p = tmp_path / name
    p.write_text(overrides_toml(overrides), encoding="utf-8")
    return str(p)


def _run(capsys, *argv):
    from cascade.laned.cli import main

    rc = main(["overlay", "publish", "--config", str(ROOT / "chain.toml"), *argv])
    out = capsys.readouterr()
    return rc, out.out, out.err


def _vals(out: str, tag: str) -> list[str]:
    return [ln[len(tag) + 1:] for ln in out.splitlines() if ln.startswith(tag + " ")]


def test_wrapper_after_a_refused_overlay(tmp_path, capsys):
    base = shipped_cfg()
    s1, s2 = _chain(base)
    prev, held = _write(tmp_path, "s2.json", s2), _write(tmp_path, "s1.json", s1)
    ov = _toml(tmp_path, "kill.toml", _kill_restate())
    rc, out, _ = _run(capsys, "--overrides", ov, "--effective-block", str(EFF),
                      "--previous", prev)
    assert rc == 1, out
    assert "--in-force" in _vals(out, "CHECK")[0]
    rc, out, _ = _run(capsys, "--overrides", ov, "--effective-block", str(EFF),
                      "--previous", prev, "--in-force", held)
    assert rc == 0, out
    assert _vals(out, "SEQ") == ["3"] and _vals(out, "CHECK") == ["VALID"]
    assert _vals(out, "IN_FORCE")[0].startswith("seq 1")
    assert _vals(out, "CHANGE") == [f"[lane_c] until_block: <release default> -> {EFF}"]
    cmd = _vals(out, "COMMAND")[0]
    assert "--netconfig-in-force-seq 1" in cmd and "--netconfig-allow-lane-off" not in cmd
    # --in-force never names a body after --previous
    rc, _, err = _run(capsys, "--overrides", ov, "--effective-block", str(EFF),
                      "--previous", held, "--in-force", prev)
    assert rc == 2 and "--in-force" in err


def test_wrapper_refuses_an_implicit_lane_off_unless_allowed(tmp_path, capsys):
    base = shipped_cfg()
    s1, s2 = _chain(base)
    prev, held = _write(tmp_path, "s2.json", s2), _write(tmp_path, "s1.json", s1)
    ov = _toml(tmp_path, "omit.toml", _kill_omit())
    common = ("--overrides", ov, "--effective-block", str(EFF), "--previous", prev,
              "--in-force", held)
    rc, out, _ = _run(capsys, *common)
    assert rc == 1, out
    assert any(x.startswith("LANE D: ON -> OFF") for x in _vals(out, "LANE")), out
    assert any("DROPPED" in c for c in _vals(out, "CHANGE")), out
    assert "COMMAND" not in out
    rc, out, _ = _run(capsys, *common, "--allow-lane-off")
    assert rc == 0, out
    assert "--netconfig-allow-lane-off" in _vals(out, "COMMAND")[0]


# ── the validator's owner path (cascade-validator --netconfig-overlay) ───────


def test_validator_dry_run_judges_against_the_newest_valid_body(tmp_path, monkeypatch,
                                                                 capsys):
    from cascade.shared import hippius
    from cascade.shared import netconfig as nc
    from cascade.validator import main as vm

    base = shipped_cfg()
    s1, s2 = _chain(base)
    monkeypatch.setattr(hippius, "open_manifest_store", lambda storage: object())
    monkeypatch.setattr(nc, "fetch_overlays",
                        lambda store, owner: [{"body": s1}, {"body": s2}])

    def args(overrides, **kw):
        a = SimpleNamespace(chain_toml=ROOT / "chain.toml",
                            netconfig_overlay=tmp_path / "ov.toml",
                            netconfig_effective_block=EFF, netconfig_comment="",
                            netconfig_publish=False, netconfig_in_force_seq=None,
                            netconfig_allow_lane_off=False)
        (tmp_path / "ov.toml").write_text(_toml_text(overrides), encoding="utf-8")
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    assert vm._netconfig_overlay(args(_kill_restate())) == 0
    out = capsys.readouterr().out
    assert "overlay seq 3" in out and "seq 1" in out and "published seq 2" in out, out
    assert vm._netconfig_overlay(args(_kill_omit())) == 1
    assert "LANE D: ON -> OFF" in capsys.readouterr().out
    assert vm._netconfig_overlay(args(_kill_omit(), netconfig_allow_lane_off=True)) == 0
    capsys.readouterr()
    # an explicit seq that is not in the chain is a usage error
    assert vm._netconfig_overlay(args(_kill_restate(), netconfig_in_force_seq=9)) == 2


def _toml_text(overrides) -> str:
    from cascade.laned.overlay import overrides_toml

    return overrides_toml(overrides)
