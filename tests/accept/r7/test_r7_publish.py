"""R7: the runbook wrapper ``cascade-laned overlay publish`` (Approve-tier).

By default it only prints the exact ``cascade-validator --netconfig-overlay
<file> --netconfig-publish ...`` command and a summary for the owner; running
it needs ``--execute`` with the confirm token the dry run printed. These tests
never let it run: ``cascade-validator`` is a recording shim first on PATH and
every ``--execute`` here carries a missing or wrong token.

Plan: LANE_D_PLAN.md §3.0 row R7, §3 "Owner steps", §5 row 4, §5 "Kill switch
and latency" (T21 / T27: the wrapper's dry run for the kill switch).
Contract: tests/accept/r7/README.md.
"""

from __future__ import annotations

import json
import re

from .helpers import (
    ARM,
    ARM_EFFECTIVE,
    MAINNET,
    OPTIONAL_COMMAND_FLAGS,
    build_body,
    command_argv,
    copy_base,
    describe,
    lines,
    load,
    make_shims,
    nc,
    one,
    run_overlay,
    shim_calls,
    toml_text,
)

EFF = 9_802_800                     # an era start on the mainnet grid (2723 x 3600)
BENIGN = {"lane_d": {"from_block": 0, "share": 0.0}}
KILL = {"lane_d": {**ARM, "until_block": EFF}}
REFUSED = {"lane_d": {"share": 0.0, "remainder": "burn"}}
REQUIRED = ("--chain-toml", "--netconfig-overlay", "--netconfig-effective-block")


def _setup(tmp_path, overrides: dict, name: str = "overrides.toml"):
    base = copy_base(tmp_path, MAINNET)
    ov = tmp_path / name
    ov.write_text(toml_text(overrides), encoding="utf-8")
    return base, ov


def _publish(tmp_path, base, ov, eff: int, *extra: str):
    return run_overlay(tmp_path, "publish", "--config", str(base), "--overrides", str(ov),
                       "--effective-block", str(eff), *extra)


def _flag_values(argv: list[str]) -> dict[str, str]:
    """flag -> value for every ``--flag value`` pair; ``--netconfig-publish``
    is the one bare flag."""
    out: dict[str, str] = {}
    i = 1
    while i < len(argv):
        a = argv[i]
        assert a.startswith("--"), f"stray argument {a!r} in {argv}"
        if a == "--netconfig-publish":
            out[a] = ""
            i += 1
            continue
        assert i + 1 < len(argv), f"{a} has no value in {argv}"
        assert a not in out, f"{a} given twice in {argv}"
        out[a] = argv[i + 1]
        i += 2
    return out


def _no_key_talk(proc) -> None:
    text = (proc.stdout + proc.stderr).lower()
    for word in ("coldkey", "bt_wallet_password", ".bittensor/wallets"):
        assert word not in text, f"output mentions {word!r}\n{describe(proc)}"


def test_r7_publish_dry_run_prints_command_and_summary(tmp_path):
    base, ov = _setup(tmp_path, BENIGN)
    proc = _publish(tmp_path, base, ov, EFF)
    assert proc.returncode == 0, describe(proc)
    assert any(line.startswith("DRY RUN") for line in proc.stdout.splitlines()), (
        f"no DRY RUN line\n{describe(proc)}")

    cfg = load(base)
    body = build_body(cfg, BENIGN, EFF)
    assert one(proc, "SEQ") == "1"
    assert one(proc, "PREV") == "-"
    assert one(proc, "EFFECTIVE_BLOCK") == str(EFF)
    assert one(proc, "DIGEST") == nc().body_digest(body)
    assert one(proc, "NOTE") == nc().cfg_feature_name(body)
    assert lines(proc, "DIFF") == ["(none)"], f"benign overlay diff: {lines(proc, 'DIFF')}"
    assert one(proc, "CHECK") == "VALID"

    argv = command_argv(proc)
    assert argv[0] == "cascade-validator", argv
    flags = _flag_values(argv)
    assert flags.get("--chain-toml") == str(base), flags
    assert flags.get("--netconfig-overlay") == str(ov), flags
    assert flags.get("--netconfig-effective-block") == str(EFF), flags
    assert "--netconfig-publish" in flags, f"the printed command does not publish: {argv}"
    unknown = set(flags) - set(REQUIRED) - {"--netconfig-publish"} - set(OPTIONAL_COMMAND_FLAGS)
    assert not unknown, f"unexpected flags in the printed command: {sorted(unknown)}"
    assert not any(f.startswith("--wallet") for f in flags), (
        f"the wrapper invented wallet flags: {argv}")

    token = one(proc, "CONFIRM")
    assert re.fullmatch(r"\S+", token), f"confirm token {token!r}"
    assert "--execute" in proc.stdout, "the dry run does not tell the owner how to execute"
    assert shim_calls(tmp_path / "shim_calls.log") == [], "the dry run ran a shimmed program"
    _no_key_talk(proc)

    again = _publish(tmp_path, base, ov, EFF)
    assert again.stdout == proc.stdout, "two dry runs with the same inputs differ"


def test_r7_publish_dry_run_for_kill_switch(tmp_path):
    """T21 / T27 (CI half): the revert overlay is staged as seq N+1 and its
    dry run shows the exact diff the fleet would apply."""
    base, ov = _setup(tmp_path, KILL)
    cfg = load(base)
    previous = build_body(cfg, {"lane_d": dict(ARM)}, ARM_EFFECTIVE)
    prev_path = tmp_path / "previous.json"
    prev_path.write_text(json.dumps(previous), encoding="utf-8")
    proc = _publish(tmp_path, base, ov, EFF, "--previous", str(prev_path))
    assert proc.returncode == 0, describe(proc)

    m = nc()
    body = build_body(cfg, KILL, EFF, previous)
    assert one(proc, "SEQ") == "2"
    assert one(proc, "PREV") == m.body_digest(previous)
    assert one(proc, "DIGEST") == m.body_digest(body)
    want = m.diff_tables(cfg.raw or {}, m.merge_raw(cfg.raw or {}, KILL))
    assert want, "fixture: the kill overrides differ from the unarmed base"
    assert lines(proc, "DIFF") == want, f"DIFF lines != diff_tables\n{describe(proc)}"
    assert one(proc, "CHECK") == "VALID"
    flags = _flag_values(command_argv(proc))
    assert flags.get("--netconfig-overlay") == str(ov)
    assert "--netconfig-publish" in flags
    assert shim_calls(tmp_path / "shim_calls.log") == []


def test_r7_publish_passes_through_network_and_wallet_names(tmp_path):
    base, ov = _setup(tmp_path, BENIGN)
    proc = _publish(tmp_path, base, ov, EFF, "--network", "test", "--wallet-name", "ownerw",
                    "--wallet-hotkey", "mainhk", "--comment", "s0c seq-1 no-op")
    assert proc.returncode == 0, describe(proc)
    flags = _flag_values(command_argv(proc))
    assert flags.get("--network") == "test", flags
    assert flags.get("--wallet-name") == "ownerw", flags
    assert flags.get("--wallet-hotkey") == "mainhk", flags
    assert flags.get("--netconfig-comment") == "s0c seq-1 no-op", flags
    assert "--wallet-path" not in flags
    body = build_body(load(base), BENIGN, EFF)
    body["comment"] = "s0c seq-1 no-op"
    assert one(proc, "DIGEST") == nc().body_digest(body), "the comment is not in the digest"


def test_r7_publish_refused_overlay_offers_no_command(tmp_path):
    base, ov = _setup(tmp_path, REFUSED)
    proc = _publish(tmp_path, base, ov, EFF)
    assert proc.returncode == 1, describe(proc)
    check = one(proc, "CHECK")
    assert check.startswith("REFUSED") and "release-only" in check, check
    assert lines(proc, "COMMAND") == [], "a refused overlay was offered for publishing"
    assert lines(proc, "CONFIRM") == [], "a refused overlay got a confirm token"
    assert shim_calls(tmp_path / "shim_calls.log") == []


def test_r7_publish_refuses_non_era_start(tmp_path):
    base, ov = _setup(tmp_path, BENIGN)
    proc = _publish(tmp_path, base, ov, EFF + 900)
    assert proc.returncode == 1, describe(proc)
    assert one(proc, "CHECK").startswith("REFUSED")
    assert lines(proc, "COMMAND") == []
    assert shim_calls(tmp_path / "shim_calls.log") == []


def test_r7_confirm_token_binds_the_plan(tmp_path):
    base, ov = _setup(tmp_path, BENIGN)
    a = _publish(tmp_path, base, ov, EFF)
    b = _publish(tmp_path, base, ov, EFF + 3600)
    assert a.returncode == 0 and b.returncode == 0, describe(a) + describe(b)
    assert one(a, "CONFIRM") != one(b, "CONFIRM"), (
        "the confirm token does not change with the effective block")


def test_r7_execute_needs_confirm_token(tmp_path):
    base, ov = _setup(tmp_path, BENIGN)
    make_shims(tmp_path)
    proc = _publish(tmp_path, base, ov, EFF, "--execute")
    assert proc.returncode != 0, f"--execute without --confirm exited 0\n{describe(proc)}"
    assert shim_calls(tmp_path / "shim_calls.log") == [], (
        f"--execute without a token ran {shim_calls(tmp_path / 'shim_calls.log')}")


def test_r7_execute_refuses_wrong_confirm_token(tmp_path):
    base, ov = _setup(tmp_path, BENIGN)
    dry = _publish(tmp_path, base, ov, EFF)
    assert dry.returncode == 0, describe(dry)
    wrong = "x" + one(dry, "CONFIRM")
    proc = _publish(tmp_path, base, ov, EFF, "--execute", "--confirm", wrong)
    assert proc.returncode != 0, f"--execute with a wrong token exited 0\n{describe(proc)}"
    assert "confirm" in (proc.stdout + proc.stderr).lower(), describe(proc)
    assert shim_calls(tmp_path / "shim_calls.log") == []


def test_r7_execute_refuses_refused_overlay(tmp_path):
    base, ov = _setup(tmp_path, REFUSED)
    proc = _publish(tmp_path, base, ov, EFF, "--execute", "--confirm", "anything")
    assert proc.returncode != 0, describe(proc)
    assert shim_calls(tmp_path / "shim_calls.log") == []
