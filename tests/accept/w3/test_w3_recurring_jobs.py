"""W3 recurring jobs (lane D ops tooling): the published-weights registry
refresh as a scheduled job, and the ``cascade/lane_d/ops_jobs.py`` log tags.

* ``python -P -m cascade.lane_d.published_weights``: ``--dry-run`` reads and
  verifies every source and writes nothing; an unchanged registry is not
  rewritten (``current``), so an hourly timer is idempotent.
* every log line of the jobs module starts with a ``lane_d_ops_*`` tag (plan
  §3.1 build rule 5, ``scripts/laned/follower_patterns.py``), and the tags are
  exactly ``ops_jobs.LOG_TAGS``, the list the owner adds to cascade-ops.

Every import of a new name is inside a test body (a base without it fails on a
missing module / name in W3's PATHS, or an assertion).
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

from .helpers import checkpoint, weights_sha

REPO = Path(__file__).resolve().parents[3]
JOBS = REPO / "cascade" / "lane_d" / "ops_jobs.py"


def _main(argv: list[str]) -> int:
    try:
        return importlib.import_module("cascade.lane_d.published_weights").main(argv)
    except SystemExit as e:                   # an unknown flag on a base without it
        return int(e.code or 0)


def _ckdir(root: Path, files: dict[str, bytes]) -> Path:
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name).write_bytes(data)
    return root


def test_published_weights_dry_run_and_current(tmp_path, capsys):
    from cascade.lane_d.chain_reads import PublishedWeightsFile

    p = importlib.import_module("cascade.lane_d.published_weights")
    leg = checkpoint("leg")
    d = _ckdir(tmp_path / "leg", leg)
    out = tmp_path / "state" / "published_weights.json"
    assert _main(["--out", str(out), "--checkpoint-dir", str(d), "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
    assert not out.exists(), "a dry-run writes nothing"
    assert _main(["--out", str(out), "--checkpoint-dir", str(d)]) == 0
    assert PublishedWeightsFile(out).read() == {weights_sha(leg)}
    before = out.stat().st_mtime_ns
    assert _main(["--out", str(out), "--checkpoint-dir", str(d)]) == 0
    assert "current" in capsys.readouterr().out
    assert out.stat().st_mtime_ns == before, "an unchanged registry is not rewritten"
    # --replace may still overwrite an invalid file (unchanged behaviour)
    out.write_text('{"schema": "something-else/1"}')
    assert p.main(["--out", str(out), "--replace", "--checkpoint-dir", str(d)]) == 0
    assert PublishedWeightsFile(out).read() == {weights_sha(leg)}


def _log_messages(path: Path) -> list[tuple[int, str | None]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("info", "warning", "error", "critical", "exception")
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "log"):
            msg = node.args[0] if node.args else None
            out.append((node.lineno, msg.value if isinstance(msg, ast.Constant) else None))
    return out


def test_every_job_log_line_is_tagged_and_listed():
    assert JOBS.is_file(), "cascade/lane_d/ops_jobs.py is the recurring-jobs module"
    jobs = importlib.import_module("cascade.lane_d.ops_jobs")
    calls = _log_messages(JOBS)
    assert calls, "the jobs log their outcome"
    used = set()
    for line, msg in calls:
        assert msg is not None, f"ops_jobs.py:{line}: the log message must be a literal"
        m = re.match(r"lane_d_ops_[a-z0-9_]+", msg)
        assert m, f"ops_jobs.py:{line}: untagged log line {msg[:40]!r}"
        used.add(m.group(0))
    assert used == set(jobs.LOG_TAGS), (used ^ set(jobs.LOG_TAGS))
    assert set(jobs.LOG_TAGS.values()) <= {"info", "warning", "error", "critical"}


def test_follower_check_sees_the_jobs_module_tagged():
    """scripts/laned/follower_patterns.py over the jobs module with W3's tag
    list (fixtures/lane_d_intake_tags.py LANE_D_OPS_TAGS, which must equal
    ``ops_jobs.LOG_TAGS``): no ``untagged`` and no ``uncovered`` line."""
    from .fixtures.lane_d_intake_tags import LANE_D_OPS_TAGS
    from .helpers import TAGS_FILE

    proc = subprocess.run(
        [sys.executable, "-P", str(REPO / "scripts" / "laned" / "follower_patterns.py"),
         "--root", str(REPO), "--source", "cascade/lane_d/ops_jobs.py",
         "--patterns", str(TAGS_FILE)],
        capture_output=True, text=True, timeout=120, cwd=REPO)
    assert JOBS.is_file()
    assert proc.returncode == 0, proc.stdout + proc.stderr
    jobs = importlib.import_module("cascade.lane_d.ops_jobs")
    assert tuple(jobs.LOG_TAGS) == LANE_D_OPS_TAGS
