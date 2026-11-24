"""W18 recurring job: the ``laned_vault_ledger/1`` the sweeper reads, exported
from ``lane_d_state.json`` and its persisted reports
(``python -P -m cascade.lane_d.ops_jobs vault-ledger``; W17's
``build_vault_ledger``), then the sweeper's ``--dry-run`` on it. Idempotent
(an unchanged ledger is not rewritten), ``--dry-run`` writes nothing, a state
or closure it cannot prove fails closed (exit 1, nothing written). Every import
of the job is inside a test body (a base without it fails on a missing module
in W18's PATHS).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHAIN = REPO / "chain.testnet.toml"
HOUSE = {"digest": "e" * 64, "reveal_ts": 1_700_000_000}
LOSER, PENDING = "1" * 64, "2" * 64


def _run(argv: list[str]) -> int:
    import importlib

    try:
        return importlib.import_module("cascade.lane_d.ops_jobs").main(argv)
    except SystemExit as e:
        return int(e.code or 0)


def _state(tmp_path: Path, entries: dict, reports: list[dict]) -> Path:
    from cascade.validator.lane_d_state import genesis, report_path, save_state

    path = tmp_path / "lane_d_state.json"
    state = genesis(HOUSE)
    state["entries"] = entries
    save_state(path, state)
    for rep in reports:
        p = report_path(path, rep["boundary"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep))
    return path


def _argv(state: Path, out: Path, *extra: str) -> list[str]:
    return ["vault-ledger", "--chain-toml", str(CHAIN), "--state", str(state), "--out", str(out),
            *extra]


def _last(capsys) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_ledger_export_idempotent_and_dry_run(tmp_path, capsys):
    from cascade.shared.config import load_chain_config
    from cascade.shared.era import settlement_era

    cfg = load_chain_config(CHAIN)
    b = 9_000_000
    state = _state(tmp_path, {LOSER: {"hotkey": "5Loser", "status": "lost"},
                              PENDING: {"hotkey": "5Pend", "status": "pending"}},
                   [{"boundary": b, "entries": [{"digest": LOSER, "status": "lost"}]}])
    out = tmp_path / "ledger.json"
    assert _run(_argv(state, out, "--dry-run")) == 0
    assert _last(capsys)["action"] == "planned" and not out.exists()
    assert _run(_argv(state, out)) == 0
    assert _last(capsys)["action"] == "written"
    ledger = json.loads(out.read_text())
    assert ledger["schema"] == "laned_vault_ledger/1"
    rows = {r["digest"]: r for r in ledger["entries"]}
    assert rows[LOSER]["status"] == "lost"
    assert rows[LOSER]["closed_era"] == int(settlement_era(cfg.round, b).index)
    assert rows[PENDING]["status"] == "pending" and rows[PENDING]["closed_era"] is None
    before = out.stat().st_mtime_ns
    assert _run(_argv(state, out)) == 0
    assert _last(capsys)["action"] == "current" and out.stat().st_mtime_ns == before


def test_ledger_fails_closed(tmp_path, capsys):
    out = tmp_path / "ledger.json"
    assert _run(_argv(tmp_path / "absent.json", out)) == 1
    # a lost entry no report records: the closure era can't be proven
    state = _state(tmp_path, {LOSER: {"hotkey": "5Loser", "status": "lost"}},
                   [{"boundary": 9_000_000, "entries": []}])
    assert _run(_argv(state, out)) == 1
    assert _last(capsys)["action"] == "failed" and not out.exists()
    # no persisted report and no --block: nothing to take the era at
    (tmp_path / "s2").mkdir()
    bare = _state(tmp_path / "s2", {}, [])
    assert _run(_argv(bare, out)) == 1
    assert _run(_argv(bare, out, "--block", "9000000")) == 0
    assert _run(["vault-ledger"]) == 2


def test_sweeper_dry_run_reads_the_exported_ledger(tmp_path, capsys):
    import importlib

    state = _state(tmp_path, {PENDING: {"hotkey": "5Pend", "status": "pending"}},
                   [{"boundary": 9_000_000, "entries": []}])
    ledger = tmp_path / "ledger.json"
    assert _run(_argv(state, ledger)) == 0
    vault, archive, logs = tmp_path / "vault", tmp_path / "archive", tmp_path / "logs"
    vault.mkdir()
    archive.mkdir()
    sweeper = importlib.import_module("cascade.laned.vault_sweeper")
    rc = sweeper.main(["--vault-dir", str(vault), "--archive-dir", str(archive),
                       "--ledger", str(ledger), "--config", str(CHAIN), "--log-dir", str(logs),
                       "--dry-run"])
    assert rc == 0
    assert not list(archive.rglob("*")), "a dry-run sweep writes nothing to the archive"
