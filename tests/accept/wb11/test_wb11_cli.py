"""WB11: the owner's CLI, ``python -P -m cascade.eval.contamination build|verify``.

Two builds from the same suite data, in different processes with different
``PYTHONHASHSEED``s and a different suite/item order, write the same bytes
and print the same sha256 (the Accept note: "hash set reproducible from the
suite identities"). ``verify`` re-derives the sha256 and, with ``--input``,
rebuilds and compares byte for byte. Contract: README.md."""

from __future__ import annotations

import json

from .helpers import (
    cli_json,
    expected_identities,
    fixture_suites,
    run_cli,
    sha256_pin,
    shuffled,
    suites_json,
)


def _build(tmp_path, name: str, suites: dict, *, version: int = 1, hashseed: str = "0"):
    inp = tmp_path / f"{name}-suites.json"
    inp.write_text(suites_json(suites), encoding="utf-8")
    out = tmp_path / f"{name}-hashset-v{version}.json"
    proc = run_cli("build", "--input", str(inp), "--version", str(version), "--out", str(out),
                   hashseed=hashseed)
    assert proc.returncode == 0, f"build rc={proc.returncode}\n{proc.stdout}\n{proc.stderr[-3000:]}"
    return cli_json(proc), out, inp


def test_cli_build_writes_the_artefact_and_reports_its_sha256(tmp_path):
    rep, out, _ = _build(tmp_path, "a", fixture_suites(), version=2)
    data = out.read_bytes()
    assert rep["sha256"] == sha256_pin(data), rep
    assert rep["version"] == 2
    assert rep["suites"] == expected_identities(fixture_suites())
    art = json.loads(data)
    assert rep["n_exact"] == len(art["exact"]) and rep["n_normalised"] == len(art["normalised"])
    assert rep["n_exact"] > 0 and rep["n_normalised"] > 0


def test_cli_build_is_reproducible_across_processes_hash_seeds_and_order(tmp_path):
    s = fixture_suites()
    r1, o1, _ = _build(tmp_path, "one", s, hashseed="1")
    r2, o2, _ = _build(tmp_path, "two", shuffled(s, "cli-2"), hashseed="987654")
    r3, o3, _ = _build(tmp_path, "three", shuffled(s, "cli-3"), hashseed="random")
    assert o1.read_bytes() == o2.read_bytes() == o3.read_bytes()
    assert r1["sha256"] == r2["sha256"] == r3["sha256"]


def test_cli_build_matches_the_in_process_builder(tmp_path):
    from .helpers import mod

    rep, out, _ = _build(tmp_path, "x", fixture_suites())
    m = mod()
    a = m.build_hashset(fixture_suites(), version=1)
    assert out.read_bytes() == m.artefact_bytes(a)
    assert rep["sha256"] == m.artefact_sha256(a)


def test_cli_verify_accepts_a_matching_artefact_and_rebuild(tmp_path):
    rep, out, inp = _build(tmp_path, "v", fixture_suites())
    proc = run_cli("verify", "--artefact", str(out), "--sha256", rep["sha256"])
    assert proc.returncode == 0, proc.stdout + proc.stderr[-3000:]
    assert cli_json(proc)["ok"] is True
    proc = run_cli("verify", "--artefact", str(out), "--sha256", rep["sha256"],
                   "--input", str(inp), hashseed="31337")
    assert proc.returncode == 0, proc.stdout + proc.stderr[-3000:]
    assert cli_json(proc)["ok"] is True


def test_cli_verify_refuses_a_wrong_pin_a_tampered_file_and_other_suite_data(tmp_path):
    rep, out, inp = _build(tmp_path, "t", fixture_suites())
    wrong = "sha256:" + "0123456789abcdef" * 4
    proc = run_cli("verify", "--artefact", str(out), "--sha256", wrong)
    assert proc.returncode == 1, proc.stdout + proc.stderr[-3000:]
    assert cli_json(proc)["ok"] is False

    art = json.loads(out.read_bytes())
    h = sorted(art["exact"])[0]
    art["exact"][("f" if h[0] != "f" else "e") + h[1:]] = art["exact"].pop(h)
    tampered = tmp_path / "tampered.json"
    tampered.write_bytes(json.dumps(art, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode("ascii"))
    proc = run_cli("verify", "--artefact", str(tampered), "--sha256", rep["sha256"])
    assert proc.returncode == 1, proc.stdout + proc.stderr[-3000:]
    # pinned to its own (tampered) digest, a rebuild from the suite data refuses it
    proc = run_cli("verify", "--artefact", str(tampered),
                   "--sha256", sha256_pin(tampered.read_bytes()), "--input", str(inp))
    assert proc.returncode == 1, proc.stdout + proc.stderr[-3000:]

    other = fixture_suites()
    other["time"]["version"] = "9a1b2c3d" + "4" * 32
    oinp = tmp_path / "other-suites.json"
    oinp.write_text(suites_json(other), encoding="utf-8")
    proc = run_cli("verify", "--artefact", str(out), "--sha256", rep["sha256"],
                   "--input", str(oinp))
    assert proc.returncode == 1, proc.stdout + proc.stderr[-3000:]
    assert cli_json(proc)["ok"] is False


def test_cli_build_refuses_bad_input(tmp_path):
    bad = fixture_suites()
    bad["m4"] = bad.pop("gift")
    inp = tmp_path / "bad.json"
    inp.write_text(suites_json(bad), encoding="utf-8")
    out = tmp_path / "never.json"
    proc = run_cli("build", "--input", str(inp), "--version", "1", "--out", str(out))
    assert proc.returncode == 2, proc.stdout + proc.stderr[-3000:]
    assert not out.exists(), "a refused build wrote an artefact"
