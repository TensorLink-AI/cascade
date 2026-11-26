"""R6 acceptance: the S0a de-drift classifier (plan §3.0 R6, §5 S0a, §14).

Contract: tests/accept/r6/README.md. The CLI runs as a subprocess from the
repo root; the rules are checked against the live netconfig/decided code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from cascade.shared import decided as D
from cascade.shared import netconfig as N
from cascade.shared.config import parse_chain_config
from cascade.shared.config.reliability import DEFAULTS as RELIABILITY_DEFAULTS
from cascade.shared.era import next_era_start

REPO = Path(__file__).resolve().parents[3]
FIX = Path(__file__).resolve().parent / "fixtures"
DEPLOYED = FIX / "deployed_chain.toml"
MAIN = FIX / "main_chain.toml"
EXPECTED = json.loads((FIX / "expected_classification.json").read_text())["markers"]
OUTPUTS = ("classification.json", "main.diff", "overlay.json")
TOP_KEYS = {"schema", "deployed_sha256", "main_sha256", "markers", "counts"}
MARKER_KEYS = {"line", "table", "key", "deployed", "main", "drift", "class", "reason"}
CLASSES = ("main", "overlay", "operator-only")

MAIN_RAW = tomllib.loads(MAIN.read_text())
DEPLOYED_RAW = tomllib.loads(DEPLOYED.read_text())
# an era start under the fixture's typed grid, from the real era code
EFF = next_era_start(parse_chain_config(copy.deepcopy(MAIN_RAW)).round, 9_700_000)


def run_dedrift(deployed: Path, main: Path, out: Path, eff: int = EFF,
                cwd: Path = REPO) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run(
        [sys.executable, "-m", "cascade.laned.dedrift", "--deployed", str(deployed),
         "--main", str(main), "--out", str(out), "--effective-block", str(eff)],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("r6out") / "out"
    proc = run_dedrift(DEPLOYED, MAIN, out)
    return proc, out


def ok(run):
    proc, out = run
    assert proc.returncode == 0, f"dedrift exited {proc.returncode}: {proc.stderr[-2000:]}"
    return out


def load_cls(out: Path) -> dict:
    return json.loads((out / "classification.json").read_text())


def apply_diff(main_text: str, diff: str, where: Path) -> str:
    where.mkdir(parents=True, exist_ok=True)
    (where / "chain.toml").write_text(main_text)
    if diff:
        (where / "main.diff").write_text(diff)
        chk = subprocess.run(["git", "apply", "--check", "main.diff"], cwd=where,
                             capture_output=True, text=True)
        assert chk.returncode == 0, f"main.diff does not apply: {chk.stderr}"
        app = subprocess.run(["git", "apply", "main.diff"], cwd=where,
                             capture_output=True, text=True)
        assert app.returncode == 0, app.stderr
    return (where / "chain.toml").read_text()


def expected_patched(main_raw: dict, markers: list[dict]) -> dict:
    raw = copy.deepcopy(main_raw)
    for m in markers:
        if m["class"] == "main" and m["drift"]:
            raw.setdefault(m["table"], {})[m["key"]] = m["deployed"]
    return raw


# ── CLI and schema ───────────────────────────────────────────────────────────


def test_cli_writes_the_three_outputs(run):
    out = ok(run)
    for name in OUTPUTS:
        assert (out / name).is_file(), f"missing {name}"


def test_fixture_has_exactly_21_markers():
    assert DEPLOYED.read_text().count("LIVE OVERRIDE") == 21
    assert "LIVE OVERRIDE" not in MAIN.read_text()
    assert len(EXPECTED) == 21


def test_classification_schema(run):
    out = ok(run)
    doc = load_cls(out)
    assert set(doc) == TOP_KEYS
    assert doc["schema"] == "cascade-dedrift/1"
    assert doc["deployed_sha256"] == hashlib.sha256(DEPLOYED.read_bytes()).hexdigest()
    assert doc["main_sha256"] == hashlib.sha256(MAIN.read_bytes()).hexdigest()
    for m in doc["markers"]:
        assert set(m) == MARKER_KEYS, m
        assert m["class"] in CLASSES
    lines = [m["line"] for m in doc["markers"]]
    assert lines == sorted(lines)


def test_classifies_the_21_markers_exactly(run):
    out = ok(run)
    got = load_cls(out)["markers"]
    assert [(m["line"], m["table"], m["key"]) for m in got] == \
        [(m["line"], m["table"], m["key"]) for m in EXPECTED]
    for g, e in zip(got, EXPECTED, strict=True):
        assert g == e, f"marker at line {e['line']} [{e['table']}] {e['key']}: {g} != {e}"


def test_counts(run):
    out = ok(run)
    doc = load_cls(out)
    assert doc["counts"] == {"main": 13, "overlay": 1, "operator-only": 7}
    for c in CLASSES:
        assert doc["counts"][c] == sum(1 for m in doc["markers"] if m["class"] == c)


def test_values_come_from_the_parsed_toml(run):
    out = ok(run)
    for m in load_cls(out)["markers"]:
        assert m["deployed"] == DEPLOYED_RAW[m["table"]][m["key"]]
        assert m["main"] == MAIN_RAW.get(m["table"], {}).get(m["key"])
        assert m["drift"] == (m["deployed"] != m["main"])


def test_deterministic_byte_identical(run, tmp_path):
    out = ok(run)
    elsewhere = tmp_path / "copy" / "deeper"
    elsewhere.mkdir(parents=True)
    dep2 = shutil.copy(DEPLOYED, elsewhere / "deployed.toml")
    main2 = shutil.copy(MAIN, elsewhere / "release.toml")
    proc = run_dedrift(Path(dep2), Path(main2), tmp_path / "out2", cwd=REPO)
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in OUTPUTS:
        assert (out / name).read_bytes() == (tmp_path / "out2" / name).read_bytes(), name
        text = (out / name).read_text()
        assert str(FIX) not in text and str(tmp_path) not in text, f"{name} carries a path"


def test_inputs_are_not_modified(run):
    ok(run)
    assert hashlib.sha256(DEPLOYED.read_bytes()).hexdigest() == \
        load_cls(run[1])["deployed_sha256"]
    assert tomllib.loads(MAIN.read_text()) == MAIN_RAW
    assert tomllib.loads(DEPLOYED.read_text()) == DEPLOYED_RAW


# ── main.diff ────────────────────────────────────────────────────────────────


def test_main_diff_headers(run):
    out = ok(run)
    diff = (out / "main.diff").read_text()
    assert "--- a/chain.toml" in diff and "+++ b/chain.toml" in diff


def test_main_diff_types_exactly_main_class_drift(run, tmp_path):
    out = ok(run)
    patched = apply_diff(MAIN.read_text(), (out / "main.diff").read_text(), tmp_path)
    assert tomllib.loads(patched) == expected_patched(MAIN_RAW, EXPECTED)


def test_patched_main_loads_and_carries_no_marker(run, tmp_path):
    out = ok(run)
    patched = apply_diff(MAIN.read_text(), (out / "main.diff").read_text(), tmp_path)
    assert "LIVE OVERRIDE" not in patched
    parse_chain_config(tomllib.loads(patched))


# ── overlay.json ─────────────────────────────────────────────────────────────


def test_overlay_overrides_exactly_overlay_class_drift(run):
    out = ok(run)
    body = json.loads((out / "overlay.json").read_text())
    want: dict = {}
    for m in EXPECTED:
        if m["class"] == "overlay" and m["drift"]:
            want.setdefault(m["table"], {})[m["key"]] = m["deployed"]
    assert body["overrides"] == want
    assert body["seq"] == 1 and body["prev"] is None
    assert body["effective_block"] == EFF
    assert body["protocol"] == N.PROTOCOL
    assert body["network"] == N.network_identity(parse_chain_config(copy.deepcopy(MAIN_RAW)))
    N.check_body(body)


def test_overlay_validates_on_the_patched_main_base(run, tmp_path):
    out = ok(run)
    patched = apply_diff(MAIN.read_text(), (out / "main.diff").read_text(), tmp_path)
    base = parse_chain_config(tomllib.loads(patched))
    body = json.loads((out / "overlay.json").read_text())
    N.validate_overlay(base, body)


def test_operator_only_and_release_only_never_in_overlay(run):
    out = ok(run)
    body = json.loads((out / "overlay.json").read_text())
    banned = {(m["table"], m["key"]) for m in EXPECTED if m["class"] != "overlay"}
    for table, kv in body["overrides"].items():
        assert table not in N.REFUSED_TABLES
        for key in kv:
            assert (table, key) not in banned
            assert (table, key) not in (("scoring", "burn_uid"), ("lane_b", "remainder"))


def test_operator_only_never_in_main_diff(run, tmp_path):
    out = ok(run)
    patched = tomllib.loads(apply_diff(MAIN.read_text(), (out / "main.diff").read_text(),
                                       tmp_path))
    for m in EXPECTED:
        if m["class"] == "operator-only":
            assert patched.get(m["table"], {}).get(m["key"]) == m["main"], m


# ── variants built in tmp_path ───────────────────────────────────────────────


BURN_LINE = "burn_uid             = 0         # weight sink (owner burn slot)\n"


def test_release_only_key_marker_goes_to_main_never_overlay(tmp_path):
    text = DEPLOYED.read_text()
    assert BURN_LINE in text
    variant = tmp_path / "deployed.toml"
    variant.write_text(text.replace(
        BURN_LINE, "burn_uid             = 7         # LIVE OVERRIDE 2026-11-01: neutral sink\n"))
    proc = run_dedrift(variant, MAIN, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = load_cls(tmp_path / "out")
    assert len(doc["markers"]) == 22
    burn = [m for m in doc["markers"] if (m["table"], m["key"]) == ("scoring", "burn_uid")]
    assert len(burn) == 1
    assert (burn[0]["class"], burn[0]["reason"], burn[0]["drift"]) == \
        ("main", "release_only_key", True)
    body = json.loads((tmp_path / "out" / "overlay.json").read_text())
    assert "burn_uid" not in body["overrides"].get("scoring", {})
    patched = apply_diff(MAIN.read_text(), (tmp_path / "out" / "main.diff").read_text(),
                         tmp_path / "apply")
    assert tomllib.loads(patched)["scoring"]["burn_uid"] == 7


def test_dangling_marker_is_refused(tmp_path):
    variant = tmp_path / "deployed.toml"
    variant.write_text(DEPLOYED.read_text() + "\n# LIVE OVERRIDE 2026-11-02: trailing note\n")
    proc = run_dedrift(variant, MAIN, tmp_path / "out")
    assert proc.returncode != 0
    assert "LIVE OVERRIDE" in (proc.stderr + proc.stdout)
    assert "Error while finding module" not in proc.stderr


def test_effective_block_not_an_era_start_is_refused(tmp_path):
    proc = run_dedrift(DEPLOYED, MAIN, tmp_path / "out", eff=EFF + 1)
    assert proc.returncode != 0
    assert "era start" in (proc.stderr + proc.stdout)


def test_no_markers_gives_empty_outputs(tmp_path):
    same = tmp_path / "deployed.toml"
    shutil.copy(MAIN, same)
    proc = run_dedrift(same, MAIN, tmp_path / "out")
    assert proc.returncode == 0, proc.stderr[-2000:]
    doc = load_cls(tmp_path / "out")
    assert doc["markers"] == []
    assert doc["counts"] == {"main": 0, "overlay": 0, "operator-only": 0}
    assert (tmp_path / "out" / "main.diff").read_bytes() == b""
    body = json.loads((tmp_path / "out" / "overlay.json").read_text())
    assert body["overrides"] == {}
    N.validate_overlay(parse_chain_config(copy.deepcopy(MAIN_RAW)), body)


# ── the classify() API against the live code ─────────────────────────────────


PINNED = [
    (("round", "blocked_hotkeys"), ("overlay", "overlay_key")),
    (("round", "blocked_from_block"), ("overlay", "overlay_key")),
    (("scoring", "burn_uid"), ("main", "release_only_key")),
    (("lane_b", "remainder"), ("main", "release_only_key")),
    (("marketcast", "from_block"), ("overlay", "overlay_key")),   # containment 2026-11-24
    (("round", "funded_pod_skus"), ("main", "release_only_key")),
    (("netconfig", "owner_hotkey"), ("main", "release_only_table")),
    (("storage", "hf_backup_repo"), ("main", "release_only_table")),
    (("manifest", "trainer_hotkey"), ("main", "release_only_table")),
    (("scoring", "forfeit_from_block"), ("main", "decided_block")),
    (("scoring", "margin_v2_from_block"), ("main", "decided_block")),
    (("scoring", "cohort_maxt_from_block"), ("main", "decided_block")),
    (("eval", "shared_ladder_from_block"), ("main", "decided_block")),
    (("v2", "prelaunch_burn_from_block"), ("main", "decided_block")),
    (("v2", "image_decoupled_from_block"), ("main", "decided_block")),
    (("v2", "from_block"), ("main", "decided_block")),
    (("round", "rolling_from_block"), ("main", "decided_block")),
    (("round", "funded_pod_image"), ("main", "decided_pin")),
    (("round", "worker_code_fingerprint"), ("main", "decided_pin")),
    (("round", "ops_gpu_cooldown_seconds"), ("operator-only", "trainer_only_policy")),
    (("generation", "enabled"), ("operator-only", "operator_only_table")),
    (("eval", "gift_gate_data_dir"), ("operator-only", "host_path")),
    (("eval", "calib_horizons"), ("operator-only", "log_only")),
]


@pytest.mark.parametrize("tk,want", PINNED, ids=[f"{t}.{k}" for (t, k), _ in PINNED])
def test_classify_pinned(tk, want):
    from cascade.laned.dedrift import classify

    assert tuple(classify(*tk)) == want


def test_classify_follows_the_live_allow_list():
    from cascade.laned.dedrift import classify

    for key in sorted(N.ALLOWED["marketcast"]):
        assert tuple(classify("marketcast", key)) == ("overlay", "overlay_key"), key
    for key in sorted(RELIABILITY_DEFAULTS):
        assert tuple(classify("round", key)) == ("operator-only", "trainer_only_policy"), key
    for table in sorted(N.REFUSED_TABLES):
        assert tuple(classify(table, "any_key"))[0] == "main", table
    assert tuple(classify("no_such_table", "x")) == ("main", "release_only_key")
    for key in D.PIN_KEYS:
        if key.count(".") == 1:
            assert tuple(classify(*key.split("."))) == ("main", "decided_pin"), key
    for key in sorted(set(N.ALLOWED["v2"]) - {"from_block", "prelaunch_burn_from_block"}):
        assert tuple(classify("v2", key)) == ("overlay", "overlay_key"), key
