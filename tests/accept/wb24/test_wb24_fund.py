"""WB24: ``cascade lane-b fund`` gains ``--declared-rows <file>`` (the WB14
header and its signed ``rows=`` tag, refused locally with the intake's own
``DECLARED_ROWS_CODES``), ``--target suite:<s>`` (WB18 suite targets) and
``--train-hours`` (WB23: refused locally until WB23 is built). Never prints
keys; with none of the new flags the request is today's, byte for byte.

Every request goes to a loopback fake: the recorder (``FakeIntake``) or the
real ``FundingIntake`` behind it (``intake.real``), so the CLI's headers are
checked by the intake's own verifier. No network, no wallet keyfile."""

from __future__ import annotations

import hashlib
import json
import time

import pytest

from . import helpers as h
from .helpers import cli_env, intake, net  # noqa: F401 — pytest fixtures

# sha256 of FakeIntake.Recorded.canonical() for today's requests (computed on
# main at 30a0495 with the fixed clock, hotkey, signer and key in helpers.py).
GOLDEN = {
    "fund": "6d364dcccd90427db4c3f5149710b1bda4fbce83627efa2c84831e61cfcecedb",
    "fund_gpu": "8f5ef51448defb7a1a94df5ba2c3f8998f42a7cc1f0516fa0d03d9c07fcbc8fa",
    "withdraw": "d4214192407390eb43df62f78ca1fd8bae7eb2a55079882e3c5c96155b8d605c",
}


def _no_key(run) -> None:
    assert h.FAKE_KEY not in run.text(), (
        f"`cascade {' '.join(run.argv)}` printed the provider key")


def _rows_file(tmp_path, raw, name="rows.json"):
    p = tmp_path / name
    p.write_text(raw if isinstance(raw, str) else json.dumps(raw, indent=2), encoding="utf-8")
    return p


def _valid_rows_raw(lane_b) -> str:
    """A valid declaration in a NON-canonical form: unsorted keys and rows,
    whitespace, on the first and last MoE ordinals."""
    epl = h.experts_per_layer()
    lo0, _ = h.clamp(epl[0], lane_b.min_experts_frac, lane_b.max_experts_frac)
    last = len(epl) - 1
    lo1, _ = h.clamp(epl[last], lane_b.min_experts_frac, lane_b.max_experts_frac)
    rows_last = list(range(epl[last] - 1, epl[last] - 1 - lo1, -1))
    rows_0 = list(range(lo0 + 2, 1, -1))
    return f'{{\n  "{last}": {json.dumps(rows_last)},\n  "0": {json.dumps(rows_0)}\n}}\n'


def _setup_real(tmp_path, srv, **toml_kw):
    toml = h.write_toml(tmp_path / "chain.toml", **toml_kw)
    cfg = h.load_cfg(toml)
    it, q = h.real_intake(tmp_path / "intake", cfg.lane_b)
    srv.real = it
    return toml, cfg, it, q


def _entry(q):
    e = q.get(h.HK)
    assert e is not None, "the real intake queued no lane B entry for the CLI's request"
    return e


def _accepted(run, srv) -> None:
    posts = srv.posts()
    assert run.rc == h.OK, (f"exit {run.rc}, expected 0; error {run.doc().get('error')}; "
                            f"requests {[(r.method, r.path) for r in srv.requests]}")
    assert len(posts) == 1 and posts[0].path == "/v1/lane-b/fund", (
        f"expected one POST /v1/lane-b/fund, got {[(r.method, r.path) for r in posts]}")


# ── --declared-rows: the signed WB14 header, round-tripped through the intake ──


def test_declared_rows_round_trip_through_intake(tmp_path, cli_env, intake, capsys):  # noqa: F811
    from cascade.funding.intake import canonical_fund_message, lane_b_signed_ref
    from cascade.trainer.declared_rows import canonical_declared_rows

    toml, cfg, _it, q = _setup_real(tmp_path, intake)
    raw = _valid_rows_raw(cfg.lane_b)
    parsed = h.parse_rows(raw, cfg.lane_b)
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--declared-rows",
                                        str(_rows_file(tmp_path, raw))))
    _accepted(run, intake)
    _no_key(run)
    rec = intake.posts()[0]
    assert rec.header(h.HEADER_ROWS) == canonical_declared_rows(parsed), (
        f"{h.HEADER_ROWS} must carry the canonical JSON of the parsed rows "
        f"(canonical_declared_rows), got {rec.header(h.HEADER_ROWS)!r}")
    e = _entry(q)
    assert getattr(e, "declared_rows", None) == parsed, (
        f"the intake stored declared rows {getattr(e, 'declared_rows', None)!r}, "
        f"the file declares {parsed!r}")
    ts = rec.header("X-Timestamp")
    want = canonical_fund_message(
        "lane-b-fund", h.HK, lane_b_signed_ref(("energy",), h.REF, parsed), ts,
        hashlib.sha256(h.FAKE_KEY.encode()).hexdigest())
    assert want in cli_env.signed, (
        "the hotkey must sign the canonical lane-b-fund message whose ref field is "
        "lane_b_signed_ref(targets, ref, rows) (the |rows=<sha256> tag); signed: "
        f"{cli_env.signed!r}")
    assert b"|rows=" in want


def test_declared_rows_signature_binds_the_rows(tmp_path, cli_env, intake, capsys):  # noqa: F811
    """The captured request, with the rows header swapped for another valid
    declaration, no longer verifies at a fresh intake: the rows are signed."""
    toml, cfg, _it, _q = _setup_real(tmp_path, intake)
    raw = _valid_rows_raw(cfg.lane_b)
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--declared-rows",
                                        str(_rows_file(tmp_path, raw))))
    _accepted(run, intake)
    rec = intake.posts()[0]
    assert rec.header(h.HEADER_ROWS) is not None, f"the request carries no {h.HEADER_ROWS}"
    epl = h.experts_per_layer()
    lo, _ = h.clamp(epl[0], cfg.lane_b.min_experts_frac, cfg.lane_b.max_experts_frac)
    other = json.dumps({"0": list(range(epl[0] - lo, epl[0]))})
    hdrs = {k: v for k, v in rec.headers if k.lower() != h.HEADER_ROWS.lower()}
    hdrs[h.HEADER_ROWS] = other
    fresh, _ = h.real_intake(tmp_path / "fresh", cfg.lane_b)
    status, body = fresh.lane_b_fund(hdrs)
    assert (status, body.get("code")) == (400, "bad_signature"), (
        f"a swapped declaration must fail the signature, got {status} {body}")
    status, body = h.real_intake(tmp_path / "fresh2", cfg.lane_b)[0].lane_b_fund(
        {k: v for k, v in rec.headers})
    assert status == 202, f"the unmodified request verifies at a fresh intake: {status} {body}"


def _case_out_of_range(epl, lo):
    return json.dumps({"0": [epl[0]] + list(range(max(lo - 1, 0)))})


def _case_bad_layer(epl, lo):
    return json.dumps({str(len(epl)): list(range(lo))})


def _case_count(epl, lo):
    return json.dumps({"0": list(range(max(lo - 1, 0)))})


def _case_malformed(epl, lo):
    return '{"0": [1, 2,'


CASES = {
    "declared_rows_out_of_range": _case_out_of_range,
    "declared_rows_bad_layer": _case_bad_layer,
    "declared_rows_count": _case_count,
    "declared_rows_malformed": _case_malformed,
}


@pytest.mark.parametrize("code", h.CODES)
def test_declared_rows_refused_locally_with_intake_codes(code, tmp_path, cli_env, intake,  # noqa: F811
                                                         capsys):
    toml, cfg, it, q = _setup_real(tmp_path, intake)
    epl = h.experts_per_layer()
    lo, _ = h.clamp(epl[0], cfg.lane_b.min_experts_frac, cfg.lane_b.max_experts_frac)
    raw = CASES[code](epl, lo)
    assert h.parse_code(raw, cfg.lane_b) == code, "fixture error: the parser's code differs"
    status, body = it.lane_b_fund({"X-Miner-Hotkey": h.HK, "X-Commit-Ref": h.REF,
                                   h.HEADER_TARGETS: "energy", h.HEADER_ROWS: raw})
    assert (status, body.get("code")) == (400, code), "fixture error: the intake's code differs"
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--declared-rows",
                                        str(_rows_file(tmp_path, raw))))
    assert run.rc == h.USAGE, f"exit {run.rc}, expected {h.USAGE} (refused locally)"
    assert run.code() == code, (
        f"the local refusal must carry the intake's code {code!r}, got {run.code()!r}")
    assert intake.requests == [], (
        f"nothing may be sent for a refused declaration, sent "
        f"{[(r.method, r.path) for r in intake.requests]}")
    assert q.get(h.HK) is None
    _no_key(run)


def test_declared_rows_clamp_read_from_chain_toml(tmp_path, cli_env, intake, capsys):  # noqa: F811
    """The local clamp is the ``[lane_b] min_experts_frac`` / ``max_experts_frac``
    of the chain.toml the command loads, not a built-in default."""
    shipped = h.load_cfg(h.CHAIN_TOML).lane_b
    epl = h.experts_per_layer()
    lo, _ = h.clamp(epl[0], shipped.min_experts_frac, shipped.max_experts_frac)
    raw = json.dumps({"0": list(range(lo))})
    h.parse_rows(raw, shipped)                      # valid under the shipped clamp
    tight = float(shipped.min_experts_frac) * 2
    toml = h.write_toml(tmp_path / "tight.toml", min_frac=tight)
    cfg = h.load_cfg(toml)
    assert h.parse_code(raw, cfg.lane_b) == "declared_rows_count"
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--declared-rows",
                                        str(_rows_file(tmp_path, raw))))
    assert run.rc == h.USAGE and run.code() == "declared_rows_count", (
        f"under min_experts_frac = {tight} the declaration is refused locally "
        f"(declared_rows_count); got exit {run.rc}, code {run.code()!r}")
    assert intake.requests == []


def test_declared_rows_unreadable_file_refused_locally(tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml = h.write_toml(tmp_path / "chain.toml")
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--declared-rows",
                                        str(tmp_path / "missing.json")))
    assert run.rc == h.USAGE, f"exit {run.rc}, expected {h.USAGE}"
    assert run.code() not in (None, "usage"), (
        f"--declared-rows must be a known flag with a typed refusal, got {run.code()!r}")
    assert intake.requests == []
    _no_key(run)


# ── --target suite:<s> (WB18 suite targets) ─────────────────────────────────

PIN = "sha256:" + "0" * 64


def test_target_suite_round_trip_through_intake(tmp_path, cli_env, intake, capsys):  # noqa: F811
    from cascade.funding.intake import canonical_fund_message, lane_b_signed_ref
    from cascade.funding.queue import split_lane_b_ref

    toml, cfg, _it, q = _setup_real(tmp_path, intake, pin=PIN, suite_targets=True)
    assert "suite:gift" in cfg.lane_b.targets, "fixture error: suite targets not armed"
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--target", "suite:gift",
                                        targets=None))
    _accepted(run, intake)
    _no_key(run)
    rec = intake.posts()[0]
    assert rec.header(h.HEADER_TARGETS) == "suite:gift"
    assert split_lane_b_ref(_entry(q).ref) == (("suite:gift",), h.REF)
    want = canonical_fund_message("lane-b-fund", h.HK, lane_b_signed_ref(("suite:gift",), h.REF),
                                  rec.header("X-Timestamp"),
                                  hashlib.sha256(h.FAKE_KEY.encode()).hexdigest())
    assert want in cli_env.signed


def test_target_suite_with_declared_rows(tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml, cfg, _it, q = _setup_real(tmp_path, intake, pin=PIN, suite_targets=True)
    raw = _valid_rows_raw(cfg.lane_b)
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--target", "suite:boom",
                                        "--declared-rows", str(_rows_file(tmp_path, raw)),
                                        targets=None))
    _accepted(run, intake)
    e = _entry(q)
    assert getattr(e, "declared_rows", None) == h.parse_rows(raw, cfg.lane_b)


def test_target_is_an_explicit_flag():
    """``--target`` is its own option of ``lane-b fund`` (main reaches
    ``--targets`` only through argparse's prefix abbreviation)."""
    import argparse

    from cascade.miner.cli import build_parser

    def sub(parser, name):
        for act in parser._actions:
            if isinstance(act, argparse._SubParsersAction) and name in act.choices:
                return act.choices[name]
        raise AssertionError(f"no {name!r} subcommand")

    fund = sub(sub(build_parser(), "lane-b"), "fund")
    assert "--target" in fund._option_string_actions, (
        "lane-b fund must define --target (suite:<s>) explicitly")
    assert "--targets" in fund._option_string_actions, "--targets must stay"


@pytest.mark.parametrize("armed,target", [(True, "suite:nope"), (False, "suite:gift")])
def test_target_suite_refused_locally(armed, target, tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml = h.write_toml(tmp_path / "chain.toml", pin=PIN if armed else "", suite_targets=armed)
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--target", target, targets=None))
    assert run.rc == h.USAGE, f"exit {run.rc}, expected {h.USAGE}"
    assert run.code() == "bad_targets", (
        f"--target {target} ({'armed' if armed else 'suite targets off'}) must be refused "
        f"locally as bad_targets, got {run.code()!r}")
    assert intake.requests == []
    _no_key(run)


# ── --train-hours (WB23 not built: refused locally) ─────────────────────────


def test_train_hours_refused_locally_until_wb23(tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml = h.write_toml(tmp_path / "chain.toml")
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--train-hours", "2"))
    assert run.rc == h.USAGE, f"exit {run.rc}, expected {h.USAGE}"
    assert run.code() not in (None, "usage"), (
        f"--train-hours must be a known flag refused with a typed code, got {run.code()!r}")
    assert "--train-hours" in run.err, "the refusal must name --train-hours"
    assert intake.requests == [], "nothing may be sent while WB23 is unbuilt"
    _no_key(run)


# ── intake replies: the WB13 contamination codes ────────────────────────────

RECEIPTS = {
    "benchmark_series_embedded": {"hashset_version": 1, "result": "embedded",
                                  "suite": "gift", "window_id": "gift/electricity/7@10"},
    "contamination_budget_exceeded": {"hashset_version": 1, "result": "budget_exceeded",
                                      "suite": None, "window_id": None},
}


@pytest.mark.parametrize("status,code", h.CONTAMINATION_REFUSALS)
def test_fund_surfaces_contamination_refusals(status, code, tmp_path, cli_env, intake,  # noqa: F811
                                              capsys):
    toml = h.write_toml(tmp_path / "chain.toml")
    body = {"code": code, "message": "refused by the intake contamination check"}
    if code in RECEIPTS:
        body["contamination"] = RECEIPTS[code]
    intake.routes[("POST", "/v1/lane-b/fund")] = (status, body)
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml))
    assert run.rc != h.OK and run.code() == code
    assert code in run.err
    _no_key(run)


@pytest.mark.parametrize("code", sorted(RECEIPTS))
def test_fund_shows_contamination_receipt(code, tmp_path, cli_env, intake, capsys):  # noqa: F811
    """A refusal carrying the WB13 receipt shows it: the JSON document carries
    the receipt, and the human output names the result and the window."""
    toml = h.write_toml(tmp_path / "chain.toml")
    receipt = RECEIPTS[code]
    status = dict((c, s) for s, c in h.CONTAMINATION_REFUSALS)[code]
    intake.routes[("POST", "/v1/lane-b/fund")] = (status, {
        "code": code, "message": "refused", "contamination": receipt})
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml))
    assert run.code() == code
    assert receipt in h.dicts_in(run.doc()), (
        f"the JSON document must carry the intake's contamination receipt {receipt}")
    human = h.run_cli(capsys, h.fund_argv(intake.base, toml, json_mode=False))
    assert receipt["result"] in human.text(), f"human output must name {receipt['result']!r}"
    if receipt["window_id"]:
        assert receipt["window_id"] in human.text(), "human output must name the window"


# ── never prints keys ───────────────────────────────────────────────────────


def test_fund_never_prints_key_even_if_intake_echoes_it(tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml = h.write_toml(tmp_path / "chain.toml")
    intake.routes[("POST", "/v1/lane-b/fund")] = (400, {
        "code": "bad_signature", "message": f"debug echo of X-Lium-Api-Key {h.FAKE_KEY}"})
    for json_mode in (True, False):
        run = h.run_cli(capsys, h.fund_argv(intake.base, toml, json_mode=json_mode))
        assert run.rc != h.OK
        _no_key(run)


# ── unarmed defaults: today's request bytes (golden from main) ──────────────


@pytest.fixture
def fixed_clock(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: h.FIXED_TIME)


GOLDEN_ARGS = {
    "fund": (),
    "fund_gpu": ("--gpu", "8xH100", "--max-price", "30"),
    "withdraw": ("--withdraw",),
}


def _golden_run(tmp_path, srv, capsys, name, *extra):
    toml = h.write_toml(tmp_path / "chain.toml")
    srv.routes[("POST", "/v1/lane-b/fund")] = (202, {"status": "queued", "lane": "b",
                                                    "targets": ["energy"]})
    srv.routes[("POST", "/v1/lane-b/withdraw")] = (200, {"status": "withdrawn"})
    before = len(srv.posts())
    run = h.run_cli(capsys, h.fund_argv(srv.base, toml, *GOLDEN_ARGS[name], *extra))
    posts = srv.posts()[before:]
    assert run.rc == h.OK, f"exit {run.rc}: {run.err[-1500:]}"
    assert len(posts) == 1, f"expected one POST, got {[(r.method, r.path) for r in posts]}"
    _no_key(run)
    return posts[0]


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_fund_request_bytes_unchanged_when_unarmed(name, tmp_path, cli_env, intake, capsys,  # noqa: F811
                                                   fixed_clock):
    rec = _golden_run(tmp_path, intake, capsys, name)
    got = hashlib.sha256(rec.canonical()).hexdigest()
    assert got == GOLDEN[name], (
        f"today's lane-b {name} request changed: sha256 {got} != golden {GOLDEN[name]}; "
        f"request {rec.canonical().decode()}")
    assert rec.header(h.HEADER_ROWS) is None


def test_fund_request_bytes_armed_nonempty_lane_d_effect(tmp_path, cli_env, intake, capsys,  # noqa: F811
                                                         fixed_clock):
    """With ``--declared-rows`` the request is today's plus exactly the rows
    header, with a signature over the rows-tagged message."""
    from cascade.trainer.declared_rows import canonical_declared_rows

    cfg = h.load_cfg(h.CHAIN_TOML)
    raw = _valid_rows_raw(cfg.lane_b)
    plain = _golden_run(tmp_path, intake, capsys, "fund")
    assert hashlib.sha256(plain.canonical()).hexdigest() == GOLDEN["fund"]
    rec = _golden_run(tmp_path, intake, capsys, "fund", "--declared-rows",
                      str(_rows_file(tmp_path, raw)))
    assert rec.header(h.HEADER_ROWS) == canonical_declared_rows(h.parse_rows(raw, cfg.lane_b))
    armed = dict(rec.identity_headers())
    today = dict(plain.identity_headers())
    assert set(armed) - set(today) == {h.HEADER_ROWS.lower()}, (
        f"--declared-rows must add exactly {h.HEADER_ROWS}; added "
        f"{sorted(set(armed) - set(today))}, removed {sorted(set(today) - set(armed))}")
    changed = {k for k in today if today[k] != armed.get(k)}
    assert changed == {"x-signature"}, f"headers other than X-Signature changed: {changed}"


# ── sanity (pass on main) ───────────────────────────────────────────────────


def test_codes_match_declared_rows_module():
    from cascade.trainer.declared_rows import DECLARED_ROWS_CODES

    assert tuple(DECLARED_ROWS_CODES) == h.CODES


def test_intake_contamination_codes_exist():
    consts = h.string_constants(h.REPO / "cascade" / "funding" / "intake.py")
    for _status, code in h.CONTAMINATION_REFUSALS:
        assert code in consts, f"{code} is not an intake reply code"


def test_plan_row_names_the_flags():
    plan = (h.REPO / "docs" / "plans" / "LANE_D_PLAN.md").read_text(encoding="utf-8")
    row = next(ln for ln in plan.splitlines() if ln.startswith("| WB24 |"))
    for token in ("--declared-rows", "--target suite:<s>", "--train-hours", "verify-repo",
                  "DECLARED_ROWS_CODES", "contamination_budget_exceeded", "rows_held",
                  "sentinel_target", "gate_closed_reason", "Never prints keys",
                  "unarmed defaults unchanged", "once built"):
        assert token in row, f"the WB24 row no longer says {token!r}"
