"""R3 acceptance: the overlay lint (plan §3.0 R3, §5 S6 + kill switch, §14.2,
§14.5; CI parts of T21 and T24). Contract: tests/accept/r3/README.md."""

from __future__ import annotations

import copy
import hashlib

import pytest

from .helpers import (
    ARM,
    BASE,
    EFF1,
    EFF2,
    EFF3,
    KILL_SHARE0,
    KILL_UNTIL,
    LANE_A,
    argv_for,
    assert_ok,
    assert_refused,
    body,
    findings,
    fleet,
    lane_d_over,
    lint,
    lint_module,
    raw_body,
    run_cli,
    run_main,
    seq1,
    seq2,
    with_rule,
)


def test_r3_module_has_main():
    mod = lint_module()
    assert callable(getattr(mod, "main", None)), "overlay_lint.main(argv) missing"


# ── T21: the kill switch passes ──────────────────────────────────────────────


@pytest.mark.parametrize("kill", [KILL_UNTIL, KILL_SHARE0], ids=["until_block", "share0"])
def test_r3_t21_kill_switch_passes(capsys, tmp_path, kill):
    prev = seq2()
    ov = body(lane_d_over(LANE_A, kill), EFF3, prev)
    assert_ok(*lint(capsys, tmp_path, ov, prev))


@pytest.mark.parametrize("kill", [KILL_UNTIL, KILL_SHARE0], ids=["until_block", "share0"])
def test_r3_t21_kill_switch_passes_on_upgraded_fleet(capsys, tmp_path, kill):
    """No release needed: every node already lists lane_d (post-S5)."""
    prev = seq2()
    ov = body(lane_d_over(LANE_A, kill), EFF3, prev)
    pol = fleet(validator_main=True, trainer_main=True, sat_a=True)
    assert_ok(*lint(capsys, tmp_path, ov, prev, pol))


def test_r3_activation_restating_lane_a_passes(capsys, tmp_path):
    prev = seq1()
    assert_ok(*lint(capsys, tmp_path, seq2(), prev))


def test_r3_seq1_lane_d_only_passes(capsys, tmp_path):
    ov = body({"lane_d": dict(KILL_UNTIL)}, EFF2, None)
    assert_ok(*lint(capsys, tmp_path, ov))


# ── non-lane-d: only byte-identical restatement of seq N-1 ───────────────────


def test_r3_seq1_lane_d_with_lane_a_key_refused(capsys, tmp_path):
    ov = body(lane_d_over(LANE_A, KILL_UNTIL), EFF2, None)
    code, lines = lint(capsys, tmp_path, ov)
    assert_refused(code, lines, "non-lane-d", "[scoring] win_margin_start")
    assert_refused(code, lines, "non-lane-d", "[scoring] min_windows")


def test_r3_changed_lane_a_value_refused(capsys, tmp_path):
    prev = seq2()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["min_windows"] = 301
    ov = body(lane_d_over(lane_a, KILL_UNTIL), EFF3, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    hits = assert_refused(code, lines, "non-lane-d", "[scoring] min_windows")
    assert not [ln for ln in with_rule(lines, "non-lane-d") if "win_margin_start" in ln], (
        "the unchanged restatement was refused too:\n" + "\n".join(lines))
    assert len(hits) == 1


def test_r3_int_vs_float_restatement_is_not_byte_identical(capsys, tmp_path):
    """300 == 300.0 in Python, but the canonical JSON differs."""
    prev = seq2()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["min_windows"] = 300.0
    ov = raw_body(lane_d_over(lane_a, KILL_UNTIL), EFF3, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "non-lane-d", "[scoring] min_windows")


def test_r3_dropped_lane_a_key_refused(capsys, tmp_path):
    """Overlays are snapshots: dropping a seq N-1 key reverts it to the base."""
    prev = seq2()
    ov = body(lane_d_over({"scoring": {"win_margin_start": 0.015}}, KILL_UNTIL), EFF3, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "non-lane-d", "[scoring] min_windows")


def test_r3_dropped_lane_a_table_refused(capsys, tmp_path):
    prev = seq2()
    ov = body({"lane_d": dict(KILL_UNTIL)}, EFF3, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "non-lane-d", "[scoring] win_margin_start")
    assert_refused(code, lines, "non-lane-d", "[scoring] min_windows")


def test_r3_new_lane_a_key_at_base_value_refused(capsys, tmp_path):
    """Not set by seq N-1: refused even when equal to chain.toml (base drift)."""
    prev = seq2()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["king_decay"] = 0.5
    ov = body(lane_d_over(lane_a, KILL_UNTIL), EFF3, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "non-lane-d", "[scoring] king_decay")


def test_r3_lane_a_only_overlay_not_subject_to_restatement(capsys, tmp_path):
    prev = seq1()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["min_windows"] = 320
    ov = body(lane_a, EFF2, prev)
    assert_ok(*lint(capsys, tmp_path, ov, prev))


# ── release-only burn destinations ───────────────────────────────────────────


@pytest.mark.parametrize("table,key,value", [
    ("lane_d", "remainder", "lane_a"),
    ("lane_c", "remainder", "lane_a"),
    ("lane_b", "remainder", "lane_a"),
    ("scoring", "burn_uid", 1),
])
def test_r3_release_only_refused_in_lane_d_overlay(capsys, tmp_path, table, key, value):
    prev = seq2()
    over = lane_d_over(LANE_A, KILL_UNTIL)
    over.setdefault(table, {})[key] = value
    ov = raw_body(over, EFF3, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "release-only", f"[{table}] {key}")


def test_r3_lane_c_remainder_restatement_refused(capsys, tmp_path):
    """RV1: a byte-identical [lane_c] remainder still splits a mixed fleet."""
    s1 = raw_body({**LANE_A, "lane_c": {"remainder": "burn"}}, EFF1, None)
    prev = raw_body({**LANE_A, "lane_c": {"remainder": "burn"}, "lane_d": dict(ARM)},
                    EFF2, s1)
    ov = raw_body({**LANE_A, "lane_c": {"remainder": "burn"}, "lane_d": dict(KILL_UNTIL)},
                  EFF3, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "release-only", "[lane_c] remainder")
    assert not [ln for ln in with_rule(lines, "non-lane-d") if "[lane_c] remainder" in ln], (
        "a byte-identical restatement was reported as non-lane-d:\n" + "\n".join(lines))



def test_r3_release_only_refused_in_lane_a_only_overlay(capsys, tmp_path):
    prev = seq1()
    ov = raw_body({**LANE_A, "lane_c": {"remainder": "lane_a"}}, EFF2, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "release-only", "[lane_c] remainder")


# ── T24: a [lane_d] overlay is unsafe while any node lacks lane_d ────────────


def test_r3_t24_lane_d_overlay_unsafe_on_mixed_fleet(capsys, tmp_path):
    prev = seq2()
    ov = body(lane_d_over(LANE_A, KILL_UNTIL), EFF3, prev)
    pol = fleet(validator_main=True, trainer_main=False, sat_old=False)
    code, lines = lint(capsys, tmp_path, ov, prev, pol)
    for node in ("trainer-main", "sat-old"):
        hits = assert_refused(code, lines, "unsafe-fleet", node)
        assert all("[lane_d]" in ln for ln in hits), "\n".join(hits)
    assert not [ln for ln in with_rule(lines, "unsafe-fleet") if "validator-main" in ln], (
        "an upgraded node was reported unsafe:\n" + "\n".join(lines))


def test_r3_t24_activation_unsafe_while_trainer_old(capsys, tmp_path):
    pol = fleet(validator_main=True, trainer_main=False)
    code, lines = lint(capsys, tmp_path, seq2(), seq1(), pol)
    assert_refused(code, lines, "unsafe-fleet", "trainer-main")


def test_r3_t24_lane_a_overlay_safe_on_mixed_fleet(capsys, tmp_path):
    """Lane A stays safe while the trainer waits for S5."""
    prev = seq1()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["min_windows"] = 320
    ov = body(lane_a, EFF2, prev)
    pol = fleet(validator_main=True, trainer_main=False)
    assert_ok(*lint(capsys, tmp_path, ov, prev, pol))


# ── chain and validity ───────────────────────────────────────────────────────


def test_r3_wrong_prev_digest_refused(capsys, tmp_path):
    prev = seq2()
    ov = body(lane_d_over(LANE_A, KILL_UNTIL), EFF3, prev)
    ov["prev"] = hashlib.sha256(b"r3-not-the-previous-body").hexdigest()
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "chain")


def test_r3_seq_gap_refused(capsys, tmp_path):
    prev = seq2()
    ov = body(lane_d_over(LANE_A, KILL_UNTIL), EFF3, prev)
    ov["seq"] = int(prev["seq"]) + 2
    code, lines = lint(capsys, tmp_path, ov, prev)
    assert_refused(code, lines, "chain")


def test_r3_seq_above_one_without_previous_refused(capsys, tmp_path):
    ov = body(lane_d_over(LANE_A, KILL_UNTIL), EFF3, seq2())
    code, lines = lint(capsys, tmp_path, ov)
    assert_refused(code, lines, "chain")


def test_r3_seq1_with_previous_refused(capsys, tmp_path):
    ov = body({"lane_d": dict(KILL_UNTIL)}, EFF2, None)
    code, lines = lint(capsys, tmp_path, ov, seq1())
    assert_refused(code, lines, "chain")


def test_r3_not_era_start_is_invalid(capsys, tmp_path):
    prev = seq2()
    ov = body(lane_d_over(LANE_A, KILL_UNTIL), EFF3 + 900, prev)
    code, lines = lint(capsys, tmp_path, ov, prev)
    hits = assert_refused(code, lines, "invalid")
    assert any("era start" in ln for ln in hits), "\n".join(hits)


# ── output, exit codes, determinism ──────────────────────────────────────────


def test_r3_findings_sorted_and_counted(capsys, tmp_path):
    prev = seq2()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["min_windows"] = 301
    lane_a["scoring"]["king_decay"] = 0.6
    over = lane_d_over(lane_a, KILL_UNTIL)
    over["lane_d"]["remainder"] = "lane_a"
    ov = raw_body(over, EFF3, prev)
    pol = fleet(validator_main=True, trainer_main=False)
    code, lines = lint(capsys, tmp_path, ov, prev, pol)
    for rule in ("non-lane-d", "release-only", "unsafe-fleet"):
        assert_refused(code, lines, rule)
    found = findings(lines)
    assert found == sorted(found), "findings not sorted:\n" + "\n".join(found)
    assert lines[-1] == f"RESULT REFUSED {len(found)}"


def test_r3_deterministic_output(capsys, tmp_path):
    prev = seq2()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["min_windows"] = 301
    ov = body(lane_d_over(lane_a, KILL_UNTIL), EFF3, prev)
    pol = fleet(sat_old=False, validator_main=True, trainer_main=False)
    a = run_main(capsys, argv_for(tmp_path / "a", ov, prev, pol))
    b = run_main(capsys, argv_for(tmp_path / "b", ov, prev, pol))
    c = run_main(capsys, argv_for(tmp_path / "c", ov, prev, pol, reverse=True))
    assert a[0] == 1, f"expected a refusal, got exit {a[0]}:\n" + "\n".join(a[1])
    assert a == b, "two runs differ"
    assert a == c, "JSON key order changed the output"


def test_r3_inputs_unchanged(capsys, tmp_path):
    prev = seq2()
    ov = body(lane_d_over(LANE_A, KILL_UNTIL), EFF3, prev)
    argv = argv_for(tmp_path, ov, prev, fleet(validator_main=True))
    files = [tmp_path / n for n in ("overlay.json", "previous.json", "policies.json")] + [BASE]
    before = [f.read_bytes() for f in files]
    code, lines = run_main(capsys, argv)
    assert code == 0, "\n".join(lines)
    assert [f.read_bytes() for f in files] == before, "an input file was modified"


def test_r3_missing_overlay_file_exit_2(capsys, tmp_path):
    code, _ = run_main(capsys, ["--overlay", str(tmp_path / "absent.json"), "--base", str(BASE)])
    assert code == 2, f"missing overlay file: exit {code}, expected 2"


def test_r3_non_json_overlay_exit_2(capsys, tmp_path):
    bad = tmp_path / "overlay.json"
    bad.write_text("[lane_d]\nshare = 0\n")
    code, _ = run_main(capsys, ["--overlay", str(bad), "--base", str(BASE)])
    assert code == 2, f"non-JSON overlay: exit {code}, expected 2"


def test_r3_missing_base_argument_exit_2(capsys, tmp_path):
    ov = body({"lane_d": dict(KILL_UNTIL)}, EFF2, None)
    argv = argv_for(tmp_path, ov)
    argv = argv[: argv.index("--base")]
    try:
        code, _ = run_main(capsys, argv)
    except SystemExit as e:          # argparse usage errors exit 2
        code = e.code
    assert code == 2, f"missing --base: exit {code}, expected 2"


def test_r3_cli_kill_switch_exit_0(tmp_path):
    prev = seq2()
    ov = body(lane_d_over(LANE_A, KILL_SHARE0), EFF3, prev)
    proc = run_cli(argv_for(tmp_path, ov, prev), tmp_path)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}"
    assert proc.stdout.splitlines()[-1] == "RESULT OK"


def test_r3_cli_refusal_exit_1(tmp_path):
    prev = seq2()
    lane_a = copy.deepcopy(LANE_A)
    lane_a["scoring"]["win_margin_start"] = 0.02
    ov = body(lane_d_over(lane_a, KILL_UNTIL), EFF3, prev)
    proc = run_cli(argv_for(tmp_path, ov, prev), tmp_path)
    assert proc.returncode == 1, f"exit {proc.returncode}\n{proc.stdout}\n{proc.stderr[-2000:]}"
    lines = proc.stdout.splitlines()
    assert with_rule(lines, "non-lane-d"), proc.stdout
    assert lines[-1] == f"RESULT REFUSED {len(findings(lines))}"
