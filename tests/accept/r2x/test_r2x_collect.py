"""R2x: the owner's read-only S0b note collector, dry-run/fixture mode only.

``python -P -m cascade.laned.s0b_collect --network finney --netuid 91
--cfg <cfg|none> --out evidence/s0b_notes.json [--dry-run [--fixture F]]``
reads the head block, its hash, the metagraph stakes and every plain
commitment through ChainClient's read methods, runs the R2 decoder and
simulator on each permitted validator's note and writes the
``laned_s0b_notes/1`` evidence. These tests never reach a chain: every run is
``--dry-run`` with ``bittensor`` made unimportable. Contract: README.md.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from .helpers import (
    MODULE,
    PROBE_CFG,
    REPO,
    as_chain_evidence,
    dry_run,
    evidence_problems,
    load_fixture,
    require_collector,
    run_collect,
    write_json,
)

CRASH_NOTE = ("cascade-ready:1:rolling-era-king:9374400:9378000:prelaunch-95-r2:"
              "9486000:9486900:v2b-69630f92:0:0")
BIG_HK = "5E2LP6tqvRkWk3oUfZ8xQyAbcHt1rGmN4sJpLdE7wYvXcB9K"
SECOND_HK = "5GsQ8nVr1kTzPq3MwXyLb7cH2dJf6eRtUa9oNiBvCx4mSe1D"


def _fixture(tmp_path: Path, mutate=None) -> Path:
    fx = load_fixture()
    if mutate:
        mutate(fx)
    return write_json(tmp_path / "snapshot.json", fx)


# ── the module and its read-only surface ─────────────────────────────────────


def test_s0b_collect_module_exists():
    require_collector()


def test_collector_uses_only_chainclient_reads_and_no_wallet():
    require_collector()
    src_path = Path(importlib.util.find_spec(MODULE).origin)
    tree = ast.parse(src_path.read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    kwargs = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    imports = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imports |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            imports.add(n.module or "")
    assert not any(m == "bittensor" or m.startswith("bittensor.") for m in imports), imports
    for read in ("current_block", "block_hash", "validator_stakes", "read_plain_commitments"):
        assert read in attrs, f"the collector must read through ChainClient.{read}"
    writes = {"set_weights", "set_weight_vector", "set_equal_share_weights", "_set_weights",
              "set_plain_commitment", "commit_submission", "_submit_commitment",
              "set_reveal_commitment", "wallet", "subtensor", "substrate", "query",
              "query_map", "transfer", "add_stake", "unstake"}
    assert not (attrs & writes), f"chain writes / raw access in the collector: {attrs & writes}"
    assert not ({"wallet_name", "wallet_hotkey", "wallet_path"} & kwargs), kwargs
    assert "ChainClient" in names, "chain access must go through ChainClient"


# ── dry-run without a fixture: the fetch plan ────────────────────────────────


def test_dry_run_without_fixture_prints_the_fetch_plan_and_writes_nothing(tmp_path):
    out = tmp_path / "s0b_notes.json"
    printed, ev = dry_run(tmp_path, out=out, expect_rc=0)
    assert ev is None and not out.exists(), "a dry-run without --fixture writes no evidence"
    assert printed["dry_run"] is True
    assert (printed["network"], printed["netuid"]) == ("finney", 91)
    plan = "\n".join(printed["would_fetch"])
    for call in ("ChainClient.current_block()", "ChainClient.block_hash(",
                 "ChainClient.validator_stakes(", "ChainClient.read_plain_commitments("):
        assert call in plan, f"fetch plan lacks {call}: {plan}"
    assert "wallet" not in plan.replace("no wallet", ""), plan
    assert printed["fixture"] is None


def test_live_mode_requires_out(tmp_path):
    require_collector()
    proc = run_collect(tmp_path, "--network", "finney", "--netuid", "91", "--cfg", "none")
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)


def test_bad_cfg_is_a_usage_error(tmp_path):
    require_collector()
    for bad in ("cfg-XYZ", "cfg-1234567", "1234", ""):
        proc = run_collect(tmp_path, "--network", "finney", "--netuid", "91", "--cfg", bad,
                           "--dry-run")
        assert proc.returncode == 2, (bad, proc.returncode, proc.stdout, proc.stderr)
        assert proc.stderr.strip(), bad


def test_fixture_network_mismatch_is_refused(tmp_path):
    fx = _fixture(tmp_path, lambda d: d.update(netuid=259))
    require_collector()
    proc = run_collect(tmp_path, "--network", "finney", "--netuid", "91", "--cfg", "none",
                       "--dry-run", "--fixture", str(fx), "--out", str(tmp_path / "o.json"))
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)


# ── dry-run with a fixture: the evidence it would write ──────────────────────


def test_fixture_evidence_schema_and_rows(tmp_path):
    out = tmp_path / "s0b_notes.json"
    printed, ev = dry_run(tmp_path, fixture=_fixture(tmp_path), out=out, expect_rc=0)
    assert ev is not None, "a dry-run with --fixture writes the evidence it would write"
    assert ev["schema"] == "laned_s0b_notes/1" and ev["wp"] == "R2x"
    assert ev["source"] == "fixture" and ev["passed"] is False, (
        "fixture evidence must never pass: source is 'fixture', passed false")
    assert (ev["network"], ev["netuid"], ev["head_block"]) == ("finney", 91, 9600000)
    assert ev["target_cfg"] is None
    assert ev["simulate_params"] == {"cfg_feature": PROBE_CFG, "cfg_source": "probe",
                                     "cfg_lock": 0, "cfg_activation": 0, "worst_case": True}
    hks = [r["hotkey"] for r in ev["hotkeys"]]
    assert hks[:2] == [BIG_HK, SECOND_HK], "rows are by stake, largest first"
    assert len(hks) == 4, "only validator_permit and stake > 0 hotkeys are recorded"
    assert ev["counts"] == {"permitted": 4, "ok": 2, "missing": 1, "not_a_note": 1}
    assert ev["trimmed_pending"] == []
    row = ev["hotkeys"][0]
    assert bytes.fromhex(row["note_hex"]).decode() == load_fixture()["commitments"][BIG_HK]
    assert row["decoded"]["segments"][0]["role"] == "primary"
    assert row["headroom_bytes"] == 128 - row["simulate"]["bytes"]
    assert printed["dry_run"] is True and printed["passed"] is False
    assert printed["gate_clear"] is True, "exit 0 / gate_clear: no pending segment trimmed"


def test_fixture_evidence_relabelled_as_chain_passes_the_verifier(tmp_path):
    _, ev = dry_run(tmp_path, fixture=_fixture(tmp_path), out=tmp_path / "e.json", expect_rc=0)
    problems = evidence_problems(as_chain_evidence(ev), check_git=False)
    assert problems == [], problems


def test_crashloop_note_is_reported_as_trimmed_pending(tmp_path):
    def crash(d):
        d["commitments"][SECOND_HK] = CRASH_NOTE
        d["head_block"] = 9493750

    _, ev = dry_run(tmp_path, fixture=_fixture(tmp_path, crash), out=tmp_path / "e.json",
                    expect_rc=1)
    assert ev["trimmed_pending"] == [{"hotkey": SECOND_HK, "feature": "v2b-69630f92"}]
    chain = as_chain_evidence(ev)
    assert chain["passed"] is False
    assert evidence_problems(chain, check_git=False) == []


def test_cfg_none_gives_the_same_trim_as_any_target_name(tmp_path):
    fx = _fixture(tmp_path)
    _, none_ev = dry_run(tmp_path, fixture=fx, out=tmp_path / "a.json", expect_rc=0)
    _, tgt_ev = dry_run(tmp_path, fixture=fx, cfg="cfg-1a2b3c4d", out=tmp_path / "b.json",
                        expect_rc=0)
    assert tgt_ev["target_cfg"] == "cfg-1a2b3c4d"
    assert tgt_ev["simulate_params"]["cfg_source"] == "target"
    assert none_ev["headroom"]["min_bytes"] == tgt_ev["headroom"]["min_bytes"]
    assert none_ev["trimmed_pending"] == tgt_ev["trimmed_pending"]
    for a, b in zip(none_ev["hotkeys"], tgt_ev["hotkeys"], strict=True):
        if a["simulate"] is not None:
            assert a["simulate"]["bytes"] == b["simulate"]["bytes"]
            assert [t["feature"] for t in a["simulate"]["trimmed"]] == \
                   [t["feature"] for t in b["simulate"]["trimmed"]]


def test_cfg_accepts_8_hex_and_64_hex_digest_forms(tmp_path):
    fx = _fixture(tmp_path)
    _, ev8 = dry_run(tmp_path, fixture=fx, cfg="1a2b3c4d", out=tmp_path / "a.json", expect_rc=0)
    digest = "1a2b3c4d" + "0123456789abcdef" * 3 + "01234567"
    assert len(digest) == 64
    _, ev64 = dry_run(tmp_path, fixture=fx, cfg=digest, out=tmp_path / "b.json", expect_rc=0)
    from cascade.shared.netconfig import cfg_feature_name

    assert ev8["target_cfg"] == "cfg-1a2b3c4d"
    assert ev64["target_cfg"] == cfg_feature_name(digest)


def test_fixture_tool_sha_is_this_checkout(tmp_path):
    _, ev = dry_run(tmp_path, fixture=_fixture(tmp_path), out=tmp_path / "e.json", expect_rc=0)
    import subprocess

    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert ev["tool"]["module"] == MODULE
    assert ev["tool"]["git_sha"] == head
    assert isinstance(ev["tool"]["git_dirty"], bool)
