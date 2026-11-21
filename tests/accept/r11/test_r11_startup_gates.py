"""R11: shipped-config startup-gate harness (plan §3.0 R11, §4 T26, §13, §14.1).

``python -P -m cascade.laned.startup_gates --config F --head-block N --role
validator|trainer`` runs every REAL validator / trainer startup gate in the
order the service runs them, with no network (the chain head is injected),
prints one ``<PASS|FAIL|WARN|SKIP> <gate>[: <reason>]`` line per gate and
exits non-zero on any FAIL. T26: the shipped ``chain.toml`` and
``chain.testnet.toml`` pass for both roles at a pinned live head (the #457 /
#460 crash-loop class: a config that loads but a startup gate refuses at the
live head). Contract: README.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from .helpers import (
    HEAD,
    HEAD_TS,
    HEAD_WORST,
    HEAD_WORST_TS,
    ROLES,
    SHIPPED,
    STARTUP_SCOPES,
    activation,
    crashloop_text,
    describe,
    harness,
    load_crash_fixture,
    parse_lines,
    plant_crashloop_state,
    predecision_text,
    read_source,
    run_gates,
    set_key,
    shipped_text,
    startup_calls,
    startup_gate_calls,
    write_config,
)

# Gate names the contract fixes (README "Gate list"); others are free.
REQUIRED = {
    "validator": ("code_path", "loader", "launch_ready", "activation_note",
                  "v2b_typed_notice", "calendar_launch"),
    "trainer": ("code_path", "loader", "launch_ready", "calendar_launch",
                "worker_release", "v2b_typed_notice"),
}
ADVISORY = {"validator": set(), "trainer": {"worker_release"}}

# 2026-11-13 timeline (OPSLOG): the external validator crash-loop on the note
# at ~00:13 UTC (head ~9493750, R2 fixture), the #460 calendar-trial
# regression reproduced at 9500400 (~13:10 UTC).
HEAD_460 = 9_500_400
HEAD_460_TS = int(datetime(2026, 11, 13, 13, 10, tzinfo=UTC).timestamp())


def _gates(role):
    return list(harness().GATES[role])


def _names(role):
    return [g.name for g in _gates(role)]


def _assert_lines_match_gates(proc, role):
    lines = parse_lines(proc)
    assert [g for _s, g, _r in lines] == _names(role), (
        "stdout must carry exactly one line per GATES[role] entry, in order\n" + describe(proc))
    return lines


def _status(lines, gate):
    found = [(s, r) for s, g, r in lines if g == gate]
    assert len(found) == 1, (gate, lines)
    return found[0]


def _assert_fails_at(proc, role, gate, *needles):
    """Exit 1; every gate before ``gate`` PASS (or WARN if advisory); ``gate``
    FAIL with each needle in its reason; every later gate SKIP."""
    assert proc.returncode == 1, describe(proc)
    lines = _assert_lines_match_gates(proc, role)
    names = [g for _s, g, _r in lines]
    i = names.index(gate)
    for status, g, _r in lines[:i]:
        allowed = {"PASS", "WARN"} if g in ADVISORY[role] else {"PASS"}
        assert status in allowed, (g, status, describe(proc))
    status, reason = lines[i][0], lines[i][2]
    assert status == "FAIL", describe(proc)
    for needle in needles:
        assert needle in reason, (f"FAIL {gate} reason lacks {needle!r}: {reason!r}\n"
                                  + describe(proc))
    assert all(s == "SKIP" for s, _g, _r in lines[i + 1:]), describe(proc)
    return reason


# ── the module and its gate list ─────────────────────────────────────────────


def test_startup_gates_module_exists():
    mod = harness()
    assert set(mod.GATES) == set(ROLES), sorted(mod.GATES)


@pytest.mark.parametrize("role", ROLES)
def test_gate_list_names_and_main_py_order(role):
    gates = _gates(role)
    names = [g.name for g in gates]
    assert len(names) == len(set(names)), names
    for name in REQUIRED[role]:
        assert name in names, (role, name, names)
    for g in gates:
        assert isinstance(g.calls, tuple) and g.calls, (g.name, g.calls)
        assert all(isinstance(c, str) and c for c in g.calls), (g.name, g.calls)
        assert isinstance(g.advisory, bool), (g.name, g.advisory)
    assert {g.name for g in gates if g.advisory} == ADVISORY[role]
    # main.py order: the code-path guard first; the config loads (under the
    # vote pin) before the launch check; validator: note, then calendar trial
    # (run_forever: _activation_startup, then check_calendar_launch); trainer:
    # calendar trial, worker release, then the typed-v2b notice.
    idx = names.index
    assert names[0] == "code_path", names
    assert idx("loader") < idx("launch_ready"), names
    if role == "validator":
        assert idx("launch_ready") < idx("activation_note") < idx("calendar_launch"), names
        assert idx("activation_note") < idx("v2b_typed_notice"), names
    else:
        assert (idx("launch_ready") < idx("calendar_launch") < idx("worker_release")
                < idx("v2b_typed_notice")), names


def test_gate_calls_name_the_real_gate_functions():
    by_name = {g.name: g.calls for role in ROLES for g in _gates(role)}
    assert "verify_release_code_path" in by_name["code_path"]
    assert "load_chain_config" in by_name["loader"]
    assert "assert_launch_ready" in by_name["launch_ready"]
    assert "v2b_note_problem" in by_name["activation_note"]
    assert "v2b_typed_notice_problem" in by_name["v2b_typed_notice"]
    assert "check_calendar_launch" in by_name["calendar_launch"]
    assert "check_worker_release" in by_name["worker_release"]
    for role in ROLES:
        covered = {c for g in _gates(role) for c in g.calls}
        assert "require_vote_pin" in covered, role


# ── kept in sync with main.py (AST) ──────────────────────────────────────────


@pytest.mark.parametrize("role", ROLES)
def test_every_listed_gate_is_called_on_the_startup_path(role):
    called = set(startup_calls(role))
    for g in _gates(role):
        missing = [c for c in g.calls if c not in called]
        assert not missing, (f"{role} gate {g.name!r} lists {missing}, which the startup "
                             f"path ({STARTUP_SCOPES[role]}) does not call")


@pytest.mark.parametrize("role", ROLES)
def test_every_startup_gate_call_is_in_the_harness(role):
    covered = {c for g in _gates(role) for c in g.calls}
    uncovered = sorted(startup_gate_calls(role) - covered)
    assert not uncovered, (f"{role} startup path calls gate(s) {uncovered} that "
                           "cascade.laned.startup_gates.GATES does not run")


def test_scanner_sees_a_new_gate_added_to_main_py():
    """Self-test of the sync check: a new gate-shaped call in a startup scope is
    seen (so the test above fails until the harness lists it); one inside the
    poll loop is not startup."""
    rel = "cascade/validator/main.py"
    src = read_source(rel)
    anchor = "    require_vote_pin()\n"
    assert src.count(anchor) == 1
    planted = src.replace(anchor, anchor + "    check_lane_d_startup(None)\n")
    assert "check_lane_d_startup" not in startup_gate_calls("validator")
    assert "check_lane_d_startup" in startup_gate_calls("validator", {rel: planted})

    rel = "cascade/trainer/loop.py"
    src = read_source(rel)
    anchor = "        self._image_decoupled_startup(client)\n        while True:\n"
    assert src.count(anchor) == 1
    before = src.replace(anchor, "        check_new_thing(client)\n" + anchor)
    inside = src.replace(anchor, anchor + "            check_poll_thing(client)\n")
    assert "check_new_thing" in startup_gate_calls("trainer", {rel: before})
    assert "check_poll_thing" not in startup_gate_calls("trainer", {rel: inside})


# ── T26: the shipped files start ─────────────────────────────────────────────


@pytest.mark.parametrize("head,ts", [(HEAD, HEAD_TS), (HEAD_WORST, HEAD_WORST_TS)],
                         ids=["head", "head_plus_horizon"])
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("config", SHIPPED)
def test_t26_shipped_config_passes_every_gate(tmp_path, config, role, head, ts):
    proc = run_gates(tmp_path, config, role, head, "--head-timestamp", str(ts))
    assert proc.returncode == 0, describe(proc)
    lines = _assert_lines_match_gates(proc, role)
    for status, gate, reason in lines:
        allowed = {"PASS", "WARN"} if gate in ADVISORY[role] else {"PASS"}
        assert status in allowed, (config, role, gate, status, reason)


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("config", SHIPPED)
def test_t26_shipped_config_passes_without_head_timestamp(tmp_path, config, role):
    """Both shipped files type [v2] from_block, so the calendar trial never
    needs chain time: an unreadable timestamp must not refuse them."""
    proc = run_gates(tmp_path, config, role, HEAD)
    assert proc.returncode == 0, describe(proc)
    _assert_lines_match_gates(proc, role)


# ── planted refusals: each fails at its gate with the real text ──────────────


def test_crashloop_20261113_fails_at_activation_note(tmp_path):
    """The 2026-11-13 external validator (R2 fixture): v2b vote open, rolling
    and pre-launch votes locked in its state. The real v2b_note_problem
    refusal, at the activation-note gate."""
    fx = load_crash_fixture()
    cfg = write_config(tmp_path, "chain.crashloop.toml", crashloop_text())
    state = tmp_path / "state"
    plant_crashloop_state(state, fx)
    proc = run_gates(tmp_path, cfg, "validator", fx["head_block"], "--state-dir", str(state))
    reason = _assert_fails_at(proc, "validator", "activation_note", fx["logged_refusal"])
    assert f"(note without it: '{fx['expected_note']}')" in reason


def test_crashloop_refusal_text_is_the_live_codes(tmp_path):
    """The needle above is what the real gate says on these inputs (so the
    harness cannot pass by echoing the fixture)."""
    from cascade.shared.config import load_chain_config
    from cascade.shared.config.loader import require_vote_pin

    act = activation()
    fx = load_crash_fixture()
    require_vote_pin()
    try:
        cfg = load_chain_config(write_config(tmp_path, "c.toml", crashloop_text()))
    finally:
        require_vote_pin(False)
    plant_crashloop_state(tmp_path / "s", fx)
    rec = act.record_for(cfg, act.ActivationStore(tmp_path / "s" / "activation_state.json").load())
    pspec = act.prelaunch_burn_feature(cfg)
    pre = act.record_for(cfg, act.ActivationStore(
        tmp_path / "s" / "activation_prelaunch_burn_state.json").load(), pspec)
    why = act.v2b_note_problem(cfg, rec, None, None, pre, head_block=fx["head_block"])
    assert why.startswith(fx["logged_refusal"]), why


def test_crashloop_config_without_state_still_reaches_the_note_gate(tmp_path):
    """Same config, fresh node (no state dir): the gates before the note pass,
    so the planted refusal above is the note gate's alone."""
    fx = load_crash_fixture()
    cfg = write_config(tmp_path, "chain.crashloop.toml", crashloop_text())
    proc = run_gates(tmp_path, cfg, "validator", fx["head_block"])
    lines = _assert_lines_match_gates(proc, "validator")
    names = [g for _s, g, _r in lines]
    for status, gate, _r in lines[:names.index("activation_note")]:
        assert status == "PASS", (gate, describe(proc))


def test_460_calendar_trial_fails_at_calendar_launch(tmp_path):
    """PR #457 typed v2b (note fixed) on a release whose dated launch was still
    pending: past the launch date the trial armed v2 at the era after the head
    and the typed v2b 9493200 "preceded" it (OPSLOG 2026-11-13 13:10, #460).
    With no chain time at the head the trial runs and refuses, for both roles."""
    cfg = write_config(tmp_path, "chain.predecision.toml", predecision_text())
    for role in ROLES:
        proc = run_gates(tmp_path, cfg, role, HEAD_460)
        _assert_fails_at(proc, role, "calendar_launch",
                         "image_decoupled_from_block=9493200 precedes [v2] from_block")


def test_460_fix_past_launch_date_passes_calendar(tmp_path):
    """#460: once chain time at the finalized head is past launch_at_utc the
    trial is skipped (``--head-timestamp`` serves that chain time)."""
    cfg = write_config(tmp_path, "chain.predecision.toml", predecision_text())
    for role in ROLES:
        proc = run_gates(tmp_path, cfg, role, HEAD_460, "--head-timestamp", str(HEAD_460_TS))
        lines = _assert_lines_match_gates(proc, role)
        assert _status(lines, "calendar_launch")[0] == "PASS", describe(proc)
        assert proc.returncode == 0, describe(proc)


@pytest.mark.parametrize("role", ROLES)
def test_invalid_lane_d_key_fails_at_loader(tmp_path, role):
    text = set_key(shipped_text(), "lane_d", "share", "1.5")
    cfg = write_config(tmp_path, "chain.bad_lane_d.toml", text)
    _assert_fails_at(run_gates(tmp_path, cfg, role, HEAD), role, "loader",
                     "[lane_d] share must be in [0, 1)")


@pytest.mark.parametrize("role", ROLES)
def test_loader_runs_under_the_vote_pin(tmp_path, role):
    """main.py calls require_vote_pin() before loading: a wrong pinned v2b
    feature only refuses in that mode, so the loader gate must refuse it."""
    text = set_key(shipped_text(), "v2", "image_decoupled_feature", '"v2b-00000001"')
    cfg = write_config(tmp_path, "chain.bad_pin.toml", text)
    _assert_fails_at(run_gates(tmp_path, cfg, role, HEAD), role, "loader",
                     "image_decoupled_feature = 'v2b-00000001'", "refusing to start")


@pytest.mark.parametrize("role", ROLES)
def test_placeholder_fails_at_launch_ready(tmp_path, role):
    text = set_key(shipped_text(), "manifest", "trainer_hotkey", '""')
    cfg = write_config(tmp_path, "chain.placeholder.toml", text)
    _assert_fails_at(run_gates(tmp_path, cfg, role, HEAD), role, "launch_ready",
                     "[manifest] trainer_hotkey is empty")


@pytest.mark.parametrize("role", ROLES)
def test_shadowed_release_fails_at_code_path(tmp_path, role):
    """The #431 guard: a declared release other than the imported tree refuses
    first, before any config is read."""
    other = tmp_path / "declared-release"
    other.mkdir()
    proc = run_gates(tmp_path, "chain.toml", role, HEAD,
                     env_extra={"CASCADE_RELEASE_PATH": str(other)})
    _assert_fails_at(proc, role, "code_path", "but the declared release is")


# ── usage ────────────────────────────────────────────────────────────────────


def test_head_block_is_required(tmp_path):
    proc = run_gates(tmp_path, "chain.toml", "validator", None)
    assert proc.returncode == 2, describe(proc)


def test_unknown_role_is_a_usage_error(tmp_path):
    proc = run_gates(tmp_path, "chain.toml", "satellite", HEAD)
    assert proc.returncode == 2, describe(proc)


def test_missing_config_is_a_usage_error(tmp_path):
    proc = run_gates(tmp_path, tmp_path / "absent.toml", "trainer", HEAD)
    assert proc.returncode == 2, describe(proc)
