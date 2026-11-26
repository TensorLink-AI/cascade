"""WB24: ``cascade lane-b verify-repo <dir>`` runs the WB13 scan
(``scan_repo``) locally against the pinned hash set (``[lane_b]
contamination_hashset`` of the chain.toml, loaded with
``load_pinned_hashset``), so a miner can check a tree before funding. It
never uploads (no connection of any kind) and never executes the repo.

The hash set is a tiny seeded WB11 artefact written under ``tmp_path``;
``load_pinned_hashset`` is wrapped so the CLI's call (with the chain.toml
pin) loads it from there. ``scan_repo`` is wrapped to record its tree."""

from __future__ import annotations

import sys

import pytest

from . import helpers as h
from .helpers import cli_env, net  # noqa: F401 — pytest fixtures


@pytest.fixture
def pinned(tmp_path, cli_env, net):  # noqa: F811
    from cascade.interface import contamination_scan as cs

    real_load, real_scan = cs.load_pinned_hashset, cs.scan_repo
    release = tmp_path / "release"
    art, pin = h.write_hashset(release)
    loads, scans = [], []

    def load(pin, root=None):  # the real signature: load_pinned_hashset(pin, root)
        loads.append((pin, root))
        return real_load(pin, release)

    def scan(repo_dir, artefact, **kw):  # scan_repo(repo_dir, artefact, *, min_hits=...)
        scans.append(repo_dir)
        return real_scan(repo_dir, artefact, **kw)

    cli_env.monkeypatch.setattr(cs, "load_pinned_hashset", load)
    cli_env.monkeypatch.setattr(cs, "scan_repo", scan)

    def expected(repo, toml):
        cfg = h.load_cfg(toml)
        return dict(real_scan(repo, real_load(cfg.lane_b.contamination_hashset, release),
                              min_hits=cfg.lane_b.contamination_min_hits))

    return h.SimpleNamespace(pin=pin, art=art, loads=loads, scans=scans, expected=expected,
                             toml=h.write_toml(tmp_path / "chain.toml", pin=pin), net=net,
                             env=cli_env)


def _verify(capsys, repo, toml, *, json_mode=True):
    argv = ["lane-b", "verify-repo", str(repo), "--chain-toml", str(toml)]
    if json_mode:
        argv.append("--json")
    return h.run_cli(capsys, argv)


def _offline(p, run) -> None:
    assert p.net.attempts == [], f"verify-repo opened connections: {p.net.attempts}"
    assert p.env.opened == [], "verify-repo must not open a wallet"
    assert h.FAKE_KEY not in run.text(), "verify-repo printed the provider key"


def _receipt(run) -> dict:
    data = run.doc().get("data") or {}
    rec = data.get("contamination")
    assert isinstance(rec, dict), f"data.contamination must be the scan receipt; data: {data}"
    return rec


def test_verify_repo_clean(tmp_path, pinned, capsys):
    repo = h.clean_repo(tmp_path / "repo")
    run = _verify(capsys, repo, pinned.toml)
    assert run.rc == h.OK, f"exit {run.rc}: {run.doc().get('error')}"
    want = pinned.expected(repo, pinned.toml)
    assert want["result"] == "clean", "fixture error"
    assert _receipt(run) == want
    assert pinned.loads and {p for p, _ in pinned.loads} == {pinned.pin}, (
        f"load_pinned_hashset must be called with the chain.toml pin; calls {pinned.loads}")
    assert pinned.scans and all(
        h.Path(p).resolve() == repo.resolve() for p in pinned.scans), (
        f"scan_repo must scan the given tree; calls {pinned.scans}")
    human = _verify(capsys, repo, pinned.toml, json_mode=False)
    assert human.rc == h.OK and "clean" in human.out
    _offline(pinned, run)


def test_verify_repo_embedded_refused(tmp_path, pinned, capsys):
    repo = h.embedded_repo(tmp_path / "repo")
    want = pinned.expected(repo, pinned.toml)
    assert want["result"] == "embedded" and want["suite"] == "gift", "fixture error"
    run = _verify(capsys, repo, pinned.toml)
    assert run.rc == h.REJECTED, f"exit {run.rc}, expected {h.REJECTED}"
    assert run.code() == "benchmark_series_embedded", (
        f"the intake's code for an embedded window, got {run.code()!r}")
    assert _receipt(run) == want
    human = _verify(capsys, repo, pinned.toml, json_mode=False)
    assert want["window_id"] in human.text(), "human output must name the embedded window"
    _offline(pinned, run)


def test_verify_repo_budget_exceeded_refused(tmp_path, pinned, capsys):
    repo = h.budget_repo(tmp_path / "repo")
    want = pinned.expected(repo, pinned.toml)
    assert want["result"] == "budget_exceeded", "fixture error"
    run = _verify(capsys, repo, pinned.toml)
    assert run.rc == h.REJECTED, f"exit {run.rc}, expected {h.REJECTED}"
    assert run.code() == "contamination_budget_exceeded", (
        f"the intake's code for an over-budget scan, got {run.code()!r}")
    assert _receipt(run) == want
    _offline(pinned, run)


def test_verify_repo_uses_chain_toml_min_hits(tmp_path, pinned, capsys):
    repo = h.embedded_repo(tmp_path / "repo")
    toml = h.write_toml(tmp_path / "lenient.toml", pin=pinned.pin,
                        min_hits={"exact": 1000, "normalised": 1000})
    want = pinned.expected(repo, toml)
    assert want["result"] == "clean", "fixture error: min_hits did not change the result"
    run = _verify(capsys, repo, toml)
    assert run.rc == h.OK and _receipt(run) == want, (
        f"verify-repo must apply [lane_b] contamination_min_hits; got exit {run.rc}, "
        f"{(run.doc().get('data') or {}).get('contamination')}")


def test_verify_repo_never_executes_the_repo(tmp_path, pinned, capsys):
    marker = tmp_path / "EXECUTED"
    repo = h.exec_trap_repo(tmp_path / "repo", marker)
    path_before, cwd_before = list(sys.path), h.Path.cwd()
    want = pinned.expected(repo, pinned.toml)
    assert not marker.exists(), "fixture error: the reference scan executed the repo"
    run = _verify(capsys, repo, pinned.toml)
    assert not marker.exists(), f"verify-repo executed the repo: {marker.read_text()}"
    assert h.repo_modules_loaded(repo) == [], "verify-repo imported a module from the repo"
    assert sys.path == path_before and h.Path.cwd() == cwd_before
    assert "WB24: the submitted repo was executed" not in run.text()
    assert run.rc in (h.OK, h.REJECTED) and _receipt(run) == want
    _offline(pinned, run)


@pytest.mark.parametrize("pin", ["", "sha256:" + "f" * 64])
def test_verify_repo_fails_closed_without_the_pinned_hashset(pin, tmp_path, pinned, capsys):
    """Unpinned, or a pin no artefact matches: never reported clean."""
    repo = h.clean_repo(tmp_path / "repo")
    toml = h.write_toml(tmp_path / "other.toml", pin=pin)
    run = _verify(capsys, repo, toml)
    assert run.rc != h.OK, f"exit 0 without the pinned hash set ({pin or 'unpinned'})"
    assert run.code() not in (None, "usage"), (
        f"verify-repo must be a known subcommand with a typed refusal, got {run.code()!r}")
    rec = (run.doc().get("data") or {}).get("contamination")
    assert not (isinstance(rec, dict) and rec.get("result") == "clean")
    _offline(pinned, run)


def test_verify_repo_not_a_directory(tmp_path, pinned, capsys):
    run = _verify(capsys, tmp_path / "missing", pinned.toml)
    assert run.rc == h.USAGE, f"exit {run.rc}, expected {h.USAGE}"
    assert run.code() not in (None, "usage"), (
        f"verify-repo must be a known subcommand with a typed refusal, got {run.code()!r}")
    _offline(pinned, run)


# ── sanity (pass on main): the fixtures behave under the real scanner ───────


def test_fixtures_scan_as_intended(tmp_path, pinned):
    toml = pinned.toml
    assert pinned.expected(h.clean_repo(tmp_path / "c"), toml)["result"] == "clean"
    assert pinned.expected(h.embedded_repo(tmp_path / "e"), toml)["result"] == "embedded"
    assert pinned.expected(h.budget_repo(tmp_path / "b"), toml)["result"] == "budget_exceeded"
    marker = tmp_path / "EXECUTED"
    pinned.expected(h.exec_trap_repo(tmp_path / "t", marker), toml)
    assert not marker.exists()
