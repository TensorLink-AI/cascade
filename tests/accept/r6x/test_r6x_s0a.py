"""R6x acceptance: S0a execution (plan §3.0 R6x, §5 S0a).

Contract: tests/accept/r6x/README.md. The fixtures are R6's (public, neutral
values); nothing here reads the deployed host file unless LANED_S0A_DEPLOYED
names it.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from cascade.laned import dedrift as R6
from cascade.shared.config import parse_chain_config
from cascade.shared.era import next_era_start

from . import s0a_evidence as S

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
R6_FIX = REPO / "tests" / "accept" / "r6" / "fixtures"
DEPLOYED = R6_FIX / "deployed_chain.toml"
MAIN = R6_FIX / "main_chain.toml"
EXPECTED_MAIN = json.loads((HERE / "fixtures" / "expected_main_values.json").read_text())["values"]
EFF = next_era_start(parse_chain_config(tomllib.loads(MAIN.read_text())).round, 9_700_000)
# A minimal decided record: no votes, no pins (the fixture main is frozen at
# 714a823 and is not judged against today's record).
EMPTY_VOTES = "as_of_block = 0\n"
GIT = ("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false")


# ── the main PR: main types every main-class drift value ────────────────────


def test_main_types_the_s0a_main_class_values():
    raw = tomllib.loads((REPO / "chain.toml").read_text(encoding="utf-8"))
    wrong = {f"{t}.{k}": (R6.lookup(raw, t, k), v)
             for t, kv in EXPECTED_MAIN.items() for k, v in kv.items()
             if R6.lookup(raw, t, k) != v}
    assert not wrong, f"main does not carry the S0a main-class values (main, want): {wrong}"


def test_main_chain_toml_carries_no_live_override_text():
    assert R6.MARKER not in (REPO / "chain.toml").read_text(encoding="utf-8")


def test_expected_main_values_are_main_class():
    for table, kv in EXPECTED_MAIN.items():
        for key, value in kv.items():
            assert R6.classify(table, key, value)[0] == "main", f"{table}.{key}"


# ── helpers: a release repo built from R6's fixtures ────────────────────────


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *GIT, *args], check=True,
                          capture_output=True, text=True).stdout


def _release_repo(tmp_path: Path, main_text: str | None = None) -> Path:
    """A git repo whose HEAD chain.toml is R6's fixture main with R6's
    main.diff applied (the merged S0a main PR)."""
    repo = tmp_path / "release"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    if main_text is None:
        shutil.copy(MAIN, repo / "chain.toml")
        diff = R6.dedrift(DEPLOYED, MAIN, EFF)["main.diff"]
        (tmp_path / "main.diff").write_text(diff)
        _git(repo, "apply", str(tmp_path / "main.diff"))
    else:
        (repo / "chain.toml").write_text(main_text)
    (repo / "decided_votes.toml").write_text(EMPTY_VOTES)
    _git(repo, "add", "chain.toml", "decided_votes.toml")
    _git(repo, "commit", "-q", "-m", "release")
    return repo


def _collect(tmp_path: Path, deployed: Path = DEPLOYED, repo: Path | None = None) -> tuple[dict, Path]:
    repo = repo or _release_repo(tmp_path)
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return S.collect(deployed, repo, "HEAD", state, EFF), repo


@pytest.fixture
def good(tmp_path):
    ev, repo = _collect(tmp_path)
    return ev, repo


# ── collector + verifier ─────────────────────────────────────────────────────


def test_reconciled_fixture_passes(good):
    ev, repo = good
    assert ev["passed"] is True, (ev["consensus_diff"], ev["classification"], ev["audit_decided"])
    assert ev["consensus_diff"] == []
    assert ev["classification"]["main_drift"] == []
    assert ev["classification"]["counts"] == {"main": 13, "overlay": 1, "operator-only": 7}
    assert set(ev["overlay"]["overrides"]) == {"round"}
    assert set(ev["overlay"]["overrides"]["round"]) == {"blocked_hotkeys"}
    S.verify(copy.deepcopy(ev), repo, DEPLOYED)


def test_collect_is_deterministic(tmp_path):
    a, repo = _collect(tmp_path / "a")
    b = S.collect(DEPLOYED, repo, "HEAD", tmp_path / "a" / "state", EFF)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_unmerged_main_pr_is_main_drift(tmp_path):
    repo = _release_repo(tmp_path, MAIN.read_text())
    ev, _ = _collect(tmp_path, repo=repo)
    assert ev["passed"] is False
    # README fixture table: the four main-class markers that drift
    assert ev["classification"]["main_drift"] == [
        "round.funded_max_price_per_hour", "round.funded_operator_fallback",
        "round.funded_pod_skus", "storage.hf_backup_repo"]
    with pytest.raises(S.EvidenceError):
        S.verify(dict(ev, passed=True), repo)


def test_unmarked_consensus_drift_is_reported(tmp_path):
    text = DEPLOYED.read_text()
    raw = tomllib.loads(text)
    assert raw["scoring"]["burn_uid"] != 7
    edited = tmp_path / "deployed.toml"
    edited.write_text(re.sub(r"^burn_uid(\s*)=\s*\d+", r"burn_uid\1= 7", text, count=1, flags=re.M))
    assert tomllib.loads(edited.read_text())["scoring"]["burn_uid"] == 7
    ev, repo = _collect(tmp_path, deployed=edited)
    assert ev["passed"] is False
    assert [(d["table"], d["key"]) for d in ev["consensus_diff"]] == [("scoring", "burn_uid")]
    with pytest.raises(S.EvidenceError, match="consensus-key diff"):
        S.verify(dict(ev, passed=True), repo)


def test_operator_only_drift_is_not_consensus(good):
    ev, _ = good
    keys = {(t, k) for t, k, _fp in ev["deployed_fingerprints"]}
    for table, key in [("eval", "gift_gate_data_dir"), ("eval", "calib_horizons"),
                       ("generation", "providers"), ("generation", "vm_image"),
                       ("round", "ops_max_active_per_coldkey")]:
        assert (table, key) not in keys


def test_evidence_records_no_deployed_value(good):
    ev, _ = good
    text = json.dumps({k: v for k, v in ev.items() if k != "overlay"})
    raw = tomllib.loads(DEPLOYED.read_text())
    for table, key in [("eval", "gift_gate_data_dir"), ("generation", "vm_image")]:
        value = R6.lookup(raw, table, key)
        assert isinstance(value, str) and value and value not in text
    for hk in raw["round"]["blocked_hotkeys"]:
        assert hk not in text


def _tamper_fp(ev):
    ev["deployed_fingerprints"][0][2] = "0" * 64


def _drop_fp(ev):
    ev["deployed_fingerprints"].pop()


def _operator_key_in_overlay(ev):
    ev["overlay"]["overrides"]["eval"] = {"gift_gate_data_dir": "/x"}


def _nonconsensus_fp(ev):
    ev["deployed_fingerprints"].append(["generation", "zz", "0" * 64])


BREAKS = {
    "wrong_wp": lambda e: e.update(wp="R6"),
    "wrong_schema": lambda e: e.update(schema="laned_s0a_diff/0"),
    "not_passed": lambda e: e.update(passed=False),
    "extra_key": lambda e: e.update(extra=1),
    "short_deployed_sha": lambda e: e.update(deployed_sha256="ab" * 31),
    "short_commit": lambda e: e.update(main_commit=e["main_commit"][:7]),
    "wrong_main_sha": lambda e: e.update(main_chain_sha256="0" * 64),
    "wrong_votes_sha": lambda e: e.update(decided_votes_sha256="0" * 64),
    "wrong_rule": lambda e: e.update(consensus_rule="r6x-consensus/0"),
    "diff_not_empty": lambda e: e["consensus_diff"].append(
        {"table": "scoring", "key": "burn_uid", "deployed": "0" * 64, "main": "1" * 64}),
    "audit_rc": lambda e: e["audit_decided"].update(exit_code=1),
    "audit_fails": lambda e: e["audit_decided"].update(fails=["TYPE x v2.y"]),
    "main_drift": lambda e: e["classification"]["main_drift"].append("storage.hf_backup_repo"),
    "counts_off": lambda e: e["classification"]["counts"].update(main=99),
    "effective_block_mismatch": lambda e: e.update(effective_block=e["effective_block"] + 3600),
    "tampered_fingerprint": _tamper_fp,
    "dropped_fingerprint": _drop_fp,
    "operator_key_in_overlay": _operator_key_in_overlay,
    "nonconsensus_fingerprint": _nonconsensus_fp,
}


@pytest.mark.parametrize("name", sorted(BREAKS))
def test_broken_evidence_is_refused(good, name):
    ev, repo = good
    S.verify(copy.deepcopy(ev), repo)
    body = copy.deepcopy(ev)
    BREAKS[name](body)
    with pytest.raises(S.EvidenceError):
        S.verify(body, repo)


def test_other_deployed_file_is_refused(good, tmp_path):
    ev, repo = good
    other = tmp_path / "other.toml"
    other.write_text(DEPLOYED.read_text() + "\n# trailing\n")
    with pytest.raises(S.EvidenceError, match="deployed_sha256"):
        S.verify(copy.deepcopy(ev), repo, other)


def test_cli_round_trip(tmp_path):
    repo = _release_repo(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    out = tmp_path / "evidence" / "s0a_diff.json"
    env = dict(os.environ, PYTHONPATH=str(REPO))
    script = str(HERE / "s0a_evidence.py")
    py = sys.executable
    c = subprocess.run([py, "-P", script, "collect", "--deployed", str(DEPLOYED), "--repo", str(repo),
                        "--main-ref", "HEAD", "--state-dir", str(state), "--effective-block", str(EFF),
                        "--out", str(out)], capture_output=True, text=True, env=env, cwd=str(REPO))
    assert c.returncode == 0, c.stderr + c.stdout
    v = subprocess.run([py, "-P", script, "verify", "--evidence", str(out), "--repo", str(repo),
                        "--deployed", str(DEPLOYED)], capture_output=True, text=True, env=env,
                       cwd=str(REPO))
    assert v.returncode == 0, v.stderr + v.stdout


# ── L4: the owner's evidence (accept R6x --evidence evidence/s0a_diff.json) ─


def test_laned_evidence_s0a_diff():
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("no LANED_EVIDENCE: run `scripts/laned/accept R6x --evidence evidence/s0a_diff.json`")
    ev = json.loads(Path(path).read_text(encoding="utf-8"))
    deployed = os.environ.get("LANED_S0A_DEPLOYED")
    S.verify(ev, REPO, Path(deployed) if deployed else None)
