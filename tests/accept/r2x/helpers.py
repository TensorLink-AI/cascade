"""Shared helpers for the R2x acceptance tests (S0b live run, owner WP).

Two halves:

* the collector ``python -P -m cascade.laned.s0b_collect`` is driven as a
  subprocess in ``--dry-run`` mode ONLY, with ``bittensor`` made unimportable,
  so no test can reach a chain;
* :func:`evidence_problems` is the evidence verifier. It never trusts the
  collector: it re-runs the R2 decoder (``cascade.laned.notes``) on every
  recorded ``note_hex`` and requires the recorded decode, simulate result,
  headroom, ``trimmed_pending`` and verdict to match, so a hand-edited
  evidence file fails. Contract: README.md.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODULE = "cascade.laned.s0b_collect"
SCHEMA = "laned_s0b_notes/1"
WP = "R2x"
MAINNET = ("finney", 91)
# The probe ``cfg-`` name ``v2b_note_problem`` itself uses for the budget
# check; every cfg-<8 hex> name is 12 bytes, so the trim result is the same
# for any target name.
PROBE_CFG = "cfg-ffffffff"
NOTE_MAX_BYTES = 128

CFG_RE = re.compile(r"cfg-[0-9a-f]{8}")
HASH_RE = re.compile(r"0x[0-9a-f]{64}")
SHA_RE = re.compile(r"[0-9a-f]{40}")
NOTE_STATES = ("ok", "missing", "not_a_note")
TOP_KEYS = ("schema", "wp", "source", "network", "netuid", "head_block", "block_hash",
            "fetched_at", "tool", "target_cfg", "simulate_params", "hotkeys", "counts",
            "headroom", "trimmed_pending", "passed")
ROW_KEYS = ("hotkey", "stake", "validator_permit", "note_state", "note_hex", "decoded",
            "simulate", "headroom_bytes")

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def notes_module():
    import cascade.laned.notes as notes

    assert Path(notes.__file__).resolve().is_relative_to(REPO), notes.__file__
    return notes


def require_collector() -> None:
    try:
        spec = importlib.util.find_spec(MODULE)
    except ModuleNotFoundError as e:
        raise AssertionError(f"{MODULE} is missing (R2x PATHS): {e}") from None
    assert spec is not None, f"{MODULE} is missing (R2x PATHS)"


def load_fixture(name: str = "chain_snapshot.json") -> dict:
    return json.loads((FIXTURES / name).read_text())


def write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data, indent=2))
    return path


def _no_network_dir(tmp_path: Path) -> Path:
    """A ``bittensor`` package that refuses to import, first on the path: a
    dry-run that tried to open a chain connection fails loudly."""
    d = tmp_path / "no_network"
    (d / "bittensor").mkdir(parents=True, exist_ok=True)
    (d / "bittensor" / "__init__.py").write_text(
        "raise ImportError('R2x test: bittensor is blocked; --dry-run must not touch the chain')\n")
    return d


def run_collect(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_no_network_dir(tmp_path)), str(REPO)])
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    return subprocess.run([sys.executable, "-P", "-m", MODULE, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=120, check=False)


def dry_run(tmp_path: Path, *extra: str, fixture: Path | None = None, cfg: str = "none",
            out: Path | None = None, expect_rc: int) -> tuple[dict, dict | None]:
    """Run a dry-run; returns (stdout JSON, evidence JSON or None)."""
    require_collector()
    args = ["--network", "finney", "--netuid", "91", "--cfg", cfg, "--dry-run"]
    if fixture is not None:
        args += ["--fixture", str(fixture)]
    if out is not None:
        args += ["--out", str(out)]
    proc = run_collect(tmp_path, *args, *extra)
    assert proc.returncode == expect_rc, (
        f"`python -P -m {MODULE} {' '.join(args + list(extra))}` exited {proc.returncode}, "
        f"want {expect_rc}\nstdout: {proc.stdout[-3000:]}\nstderr: {proc.stderr[-3000:]}")
    try:
        printed = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout is not one JSON object ({e}): {proc.stdout[-2000:]!r}\n"
                             f"stderr: {proc.stderr[-2000:]}") from None
    assert isinstance(printed, dict), printed
    evidence = json.loads(out.read_text()) if out is not None and out.is_file() else None
    return printed, evidence


def as_chain_evidence(evidence: dict) -> dict:
    """A fixture-made evidence file relabelled as a chain read, to exercise the
    verifier offline. ``passed`` is restated from the recomputed rule, and
    ``git_dirty`` is cleared as a clean release checkout would record it (the
    self-tests must not depend on the dev checkout being clean)."""
    ev = copy.deepcopy(evidence)
    ev["source"] = "chain"
    ev["tool"]["git_dirty"] = False
    ev["passed"] = ev["trimmed_pending"] == [] and ev["counts"]["ok"] >= 1
    return ev


# ── the evidence verifier ────────────────────────────────────────────────────


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=False)


def _tool_problems(tool: object, check_git: bool) -> list[str]:
    if not isinstance(tool, dict):
        return ["tool: must be an object"]
    out = []
    if tool.get("module") != MODULE:
        out.append(f"tool.module: want {MODULE!r}, got {tool.get('module')!r}")
    sha = tool.get("git_sha")
    if not (isinstance(sha, str) and SHA_RE.fullmatch(sha)):
        out.append(f"tool.git_sha: want 40 lowercase hex, got {sha!r}")
    elif check_git:
        if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            out.append(f"tool.git_sha {sha} is not a commit in this repository")
        elif _git("cat-file", "-e", f"{sha}:cascade/laned/s0b_collect.py").returncode != 0:
            out.append(f"tool.git_sha {sha} has no cascade/laned/s0b_collect.py")
    if tool.get("git_dirty") is not False:
        out.append(f"tool.git_dirty: the run must be from a clean checkout, got {tool.get('git_dirty')!r}")
    return out


def _row_problems(i: int, row: object, head: int, cfg_feature: str) -> list[str]:
    notes = notes_module()
    where = f"hotkeys[{i}]"
    if not isinstance(row, dict):
        return [f"{where}: must be an object"]
    missing = [k for k in ROW_KEYS if k not in row]
    if missing:
        return [f"{where}: missing keys {missing}"]
    out = []
    if not isinstance(row["hotkey"], str) or not row["hotkey"]:
        out.append(f"{where}.hotkey: must be a non-empty string")
    stake = row["stake"]
    if isinstance(stake, bool) or not isinstance(stake, (int, float)) or not stake > 0:
        out.append(f"{where}.stake: must be a number > 0 (eligible validators only), got {stake!r}")
    if row["validator_permit"] is not True:
        out.append(f"{where}.validator_permit: only permitted validators are recorded")
    state = row["note_state"]
    if state not in NOTE_STATES:
        return out + [f"{where}.note_state: want one of {NOTE_STATES}, got {state!r}"]
    if state == "missing":
        for k in ("note_hex", "decoded", "simulate", "headroom_bytes"):
            if row[k] is not None:
                out.append(f"{where}.{k}: must be null for a missing note")
        return out
    try:
        note = notes.note_from_hex(row["note_hex"] or "")
    except (notes.NotesError, AttributeError) as e:
        return out + [f"{where}.note_hex: not decodable: {e}"]
    if not row["note_hex"]:
        return out + [f"{where}.note_hex: empty for a {state} note"]
    from cascade.shared.activation import parse_signals

    if state == "not_a_note":
        if parse_signals(note):
            out.append(f"{where}: note_state not_a_note, but note_hex decodes to a readiness note")
        for k in ("decoded", "simulate", "headroom_bytes"):
            if row[k] is not None:
                out.append(f"{where}.{k}: must be null for a non-readiness commitment")
        return out
    if not parse_signals(note):
        return out + [f"{where}: note_state ok, but note_hex is not a readiness note: {note!r}"]
    want_dec = notes.decode_note(note, head)
    if row["decoded"] != want_dec:
        out.append(f"{where}.decoded does not match the R2 decoder on note_hex: "
                   f"recorded {row['decoded']!r}, decoder {want_dec!r}")
    want_sim = notes.simulate_note(note, head, cfg_feature, 0, 0, worst_case=True)
    if row["simulate"] != want_sim:
        out.append(f"{where}.simulate does not match the R2 simulator on note_hex: "
                   f"recorded {row['simulate']!r}, simulator {want_sim!r}")
    if row["headroom_bytes"] != NOTE_MAX_BYTES - want_sim["bytes"]:
        out.append(f"{where}.headroom_bytes: want {NOTE_MAX_BYTES - want_sim['bytes']}, "
                   f"got {row['headroom_bytes']!r}")
    return out


def evidence_problems(data: object, *, check_git: bool = True) -> list[str]:
    """Every reason ``data`` is not acceptable S0b evidence (``[]`` = pass)."""
    if not isinstance(data, dict):
        return ["evidence: top level must be a JSON object"]
    missing = [k for k in TOP_KEYS if k not in data]
    if missing:
        return [f"evidence: missing keys {missing}"]
    out = []
    if data["schema"] != SCHEMA:
        out.append(f"schema: want {SCHEMA!r}, got {data['schema']!r}")
    if data["wp"] != WP:
        out.append(f"wp: want {WP!r}, got {data['wp']!r}")
    if data["source"] != "chain":
        out.append(f"source: want 'chain' (a live read), got {data['source']!r}")
    if (data["network"], data["netuid"]) != MAINNET:
        out.append(f"network/netuid: want {MAINNET}, got {(data['network'], data['netuid'])}")
    head = data["head_block"]
    if isinstance(head, bool) or not isinstance(head, int) or head <= 0:
        return out + [f"head_block: must be an int > 0, got {head!r}"]
    if not (isinstance(data["block_hash"], str) and HASH_RE.fullmatch(data["block_hash"])):
        out.append(f"block_hash: want 0x + 64 lowercase hex, got {data['block_hash']!r}")
    try:
        ts = datetime.fromisoformat(str(data["fetched_at"]))
        if ts.tzinfo is None:
            out.append("fetched_at: must carry a UTC offset")
    except ValueError:
        out.append(f"fetched_at: not ISO 8601: {data['fetched_at']!r}")
    out += _tool_problems(data["tool"], check_git)

    target = data["target_cfg"]
    sp = data["simulate_params"]
    if target is not None and not (isinstance(target, str) and CFG_RE.fullmatch(target)):
        out.append(f"target_cfg: want null or cfg-<8 lowercase hex>, got {target!r}")
    if not isinstance(sp, dict):
        return out + ["simulate_params: must be an object"]
    want_sp = {"cfg_feature": target if target is not None else PROBE_CFG,
               "cfg_source": "target" if target is not None else "probe",
               "cfg_lock": 0, "cfg_activation": 0, "worst_case": True}
    if sp != want_sp:
        out.append(f"simulate_params: want {want_sp}, got {sp}")
    cfg_feature = want_sp["cfg_feature"] if isinstance(want_sp["cfg_feature"], str) else PROBE_CFG

    rows = data["hotkeys"]
    if not isinstance(rows, list):
        return out + ["hotkeys: must be a list"]
    for i, row in enumerate(rows):
        out += _row_problems(i, row, head, cfg_feature)
    if out:
        return out
    hks = [r["hotkey"] for r in rows]
    if len(set(hks)) != len(hks):
        out.append("hotkeys: duplicate hotkey rows")
    counts = {"permitted": len(rows),
              **{s: sum(r["note_state"] == s for r in rows) for s in NOTE_STATES}}
    if data["counts"] != counts:
        out.append(f"counts: want {counts}, got {data['counts']}")
    ok_rows = [r for r in rows if r["note_state"] == "ok"]
    if not ok_rows:
        out.append("no permitted validator has a readiness note: nothing was checked")
    trimmed = [{"hotkey": r["hotkey"], "feature": f}
               for r in rows if r["note_state"] == "ok"
               for f in r["simulate"]["pending_trimmed"]]
    if data["trimmed_pending"] != trimmed:
        out.append(f"trimmed_pending: the decoder gives {trimmed}, recorded {data['trimmed_pending']}")
    if ok_rows:
        low = min(ok_rows, key=lambda r: (r["headroom_bytes"], r["hotkey"]))
        want_head = {"min_bytes": low["headroom_bytes"], "min_hotkey": low["hotkey"]}
    else:
        want_head = {"min_bytes": None, "min_hotkey": None}
    if data["headroom"] != want_head:
        out.append(f"headroom: want {want_head}, got {data['headroom']}")
    want_pass = trimmed == [] and bool(ok_rows)
    if data["passed"] is not want_pass:
        out.append(f"passed: the rule gives {want_pass}, recorded {data['passed']!r}")
    return out
