"""WB24 (plan rev6.22, owner 2026-11-24: miners "can target multiple
evals"): ``cascade lane-b fund --target T`` is repeatable, so one leg names
several targets one per flag — domains, horizons and suite targets mixed —
up to ``[lane_b] max_domains_per_leg``. The list is validated with the
``--expert-domain`` rule and sent in the ``[lane_b]`` slot order, exactly as
``--targets T1,T2`` sends it. Requests go to the loopback fake / real
intake (helpers.py); no network, no keyfile."""

from __future__ import annotations

import pytest

from . import helpers as h
from .helpers import cli_env, intake, net  # noqa: F401 — pytest fixtures


def _spec(fn):
    """A missing behaviour fails as an AssertionError (never an import,
    attribute, key or value error), so BASE fails for the right reason."""
    import functools

    @functools.wraps(fn)
    def run(*a, **kw):
        try:
            return fn(*a, **kw)
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001 - every other failure is the spec failing
            raise AssertionError(f"{type(e).__name__}: {e}") from e

    return run

PIN = "sha256:" + "0" * 64


def _setup_real(tmp_path, srv, **toml_kw):
    toml = h.write_toml(tmp_path / "chain.toml", **toml_kw)
    cfg = h.load_cfg(toml)
    it, q = h.real_intake(tmp_path / "intake", cfg.lane_b)
    srv.real = it
    return toml, cfg, it, q


@_spec
def test_repeated_target_round_trip_through_intake(tmp_path, cli_env, intake, capsys):  # noqa: F811
    from cascade.funding.queue import split_lane_b_ref

    toml, cfg, _it, q = _setup_real(tmp_path, intake, pin=PIN, suite_targets=True)
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--target", "suite:gift",
                                        "--target", "energy", targets=None))
    assert run.rc == h.OK, f"exit {run.rc}; error {run.doc().get('error')}"
    rec = intake.posts()[0]
    # slot order: domains first, then suite targets
    assert rec.header(h.HEADER_TARGETS) == "energy,suite:gift"
    e = q.get(h.HK)
    assert e is not None and split_lane_b_ref(e.ref) == (("energy", "suite:gift"), h.REF)


@_spec
def test_repeated_target_matches_targets_list(tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml = h.write_toml(tmp_path / "chain.toml")
    # the recorder only (no real intake behind it): compare what was SENT
    h.run_cli(capsys, h.fund_argv(intake.base, toml, "--target", "web_cloudops",
                                  "--target", "energy", targets=None))
    h.run_cli(capsys, h.fund_argv(intake.base, toml, targets="energy,web_cloudops"))
    posts = intake.posts()
    assert len(posts) == 2, [(r.method, r.path) for r in intake.requests]
    assert [p.header(h.HEADER_TARGETS) for p in posts] == ["energy,web_cloudops"] * 2


@pytest.mark.parametrize("flags", [
    ("--target", "energy", "--target", "web_cloudops", "--target", "sales",
     "--target", "nature", "--target", "transport"),  # 5 > max_domains_per_leg = 4
    ("--target", "energy", "--target", "energy"),   # repeated
    ("--target", "energy,web_cloudops"),            # one target per flag
    ("--target", "nope"),                           # not configured
])
@_spec
def test_repeated_target_refused_locally(flags, tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml = h.write_toml(tmp_path / "chain.toml")
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, *flags, targets=None))
    assert run.rc == h.USAGE and run.code() == "bad_targets", (flags, run.rc, run.code())
    assert intake.requests == []


@_spec
def test_target_and_targets_stay_exclusive(tmp_path, cli_env, intake, capsys):  # noqa: F811
    toml = h.write_toml(tmp_path / "chain.toml")
    run = h.run_cli(capsys, h.fund_argv(intake.base, toml, "--target", "web_cloudops"))
    assert run.rc != h.OK and intake.requests == []
