"""Shared spec and fixture builders for the W5h acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6.4):

- W5h row: "Queue selection: FIFO by T; `max_judged_per_coldkey_per_settlement`,
  the global cap, coldkey re-check at reveal (`coldkey_changed`),
  `coldkey_unresolved` bound, `max_reveal_delay_days` close; typed skip
  reasons in the report (A8, A9) (rev5) `deregistered` void (court slot
  burns, with W8) and `ready_expired` (`ready_ttl_blocks`) handling, both
  typed in state and report."
- §1.2 Enforcement: "It walks ready entries FIFO by T. An entry over
  `max_judged_per_coldkey_per_settlement` or the global cap is skipped with
  `coldkey_cap_settlement`, carried to the next settlement, and not charged."
- §1.2 D22 option b: "The validator re-checks it at reveal [...] A mismatch is
  a typed refusal, `coldkey_changed`." "An unresolvable coldkey is never a
  silent pending: [...] `coldkey_unresolved` in state and report, plus a
  cascade-ops alert (`lane_d_coldkey_unresolved`). After
  `max_coldkey_unresolved_settlements` (recommended 4) it closes unjudged."
- §1.5: "`[lane_d] max_reveal_delay_days` (recommended 14). An entry still
  pending at the limit closes unjudged (`closed_unjudged:
  reveal_delay_exceeded`), logged, and frees its slots."
- T6b: "Three entries from one coldkey ready in one settlement; one coldkey
  changed between ready and reveal; an unresolvable coldkey | 2 judged, 1
  skipped `coldkey_cap_settlement` and carried; `coldkey_changed` refusal;
  `coldkey_unresolved` alert, then closed after the bound".
- T18b: "An entry pending at `max_reveal_delay_days` | `closed_unjudged:
  reveal_delay_exceeded`, logged, slots freed".

README.md states the contract. The fixtures are W5g's (tests/accept/w5g):
real `settle_lane_d` calls over real `pool_d/1` snapshots, numpy only.
Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from tests.accept.w5g import helpers as g

REPO = Path(__file__).resolve().parents[3]
LANE_D_PY = REPO / "cascade" / "validator" / "lane_d.py"
QUEUE_PY = REPO / "cascade" / "validator" / "lane_d_queue.py"

DAY = g.DAY
HOUR = g.HOUR
T_X = g.T_X
ARM_BLOCK = g.ARM_BLOCK
SETTLEMENT = g.SETTLEMENT
# A9 "maximum reveal delay" is configured in days of 12 s blocks (the W17
# producer's `reveal_deadline_block = reveal_block + days * 7200`).
BLOCKS_PER_DAY = 7200
COLDKEY_TAG = "lane_d_coldkey_unresolved"

boundary = g.boundary
house = g.house
read_state = g.read_state
verdict_of = g.verdict_of
rows_of = g.rows_of
entry_of = g.entry_of
skips_of = g.skips_of
fixture_key = g.fixture_key


def lane_d_cfg(**over):
    return g.lane_d_cfg(**over)


def cold(name: str) -> str:
    """A fixture coldkey ss58 (a TEST key, never a real one)."""
    return fixture_key(f"w5h-cold-{name}").ss58_address


def entry(name: str, *, reveal_ts: int = T_X, reveal_block: int | None = None,
          ready_block: int | None = None, coldkey: str | None = None, **extra) -> dict:
    """One revealed entry. ``reveal_block`` defaults to just after arming;
    ``ready_block`` to 20 blocks before the reveal; ``coldkey`` is
    ``coldkey_at_ready`` (``cold(name)`` by default). ``extra`` carries e.g.
    the validator's re-check value ``coldkey_at_reveal``."""
    rb = ARM_BLOCK + 60 if reveal_block is None else int(reveal_block)
    return {"hotkey": fixture_key(f"w5h-{name}").ss58_address,
            "digest": g.digest(f"w5h {name}"), "context": 4096,
            "coldkey_at_ready": cold(name) if coldkey is None else coldkey,
            "ready_block": rb - 20 if ready_block is None else int(ready_block),
            "reveal_block": rb, "reveal_ts": int(reveal_ts), **extra}


def uids(*entries) -> dict:
    return {e["hotkey"]: 10 + i for i, e in enumerate(entries)}


def pool(tmp_path: Path, n: int, t: int = T_X + 600):
    """Snapshot ``n``: 24 windows over 4 sources, eligible for every T <= t
    (the W5g fixture)."""
    return g.snapshot(Path(tmp_path), n, t)


def early_pool(tmp_path: Path, n: int):
    """A snapshot whose windows are all too early for an entry revealed at
    ``T_X``: such an entry stays ``pending_floor`` (never frozen, never
    judged), whatever the boundary."""
    return g.snapshot(Path(tmp_path), n, T_X - 30 * DAY)


def models_for(entries, *, house_q: str = "bad", default: str = "sharp",
               quality: dict | None = None) -> dict:
    quality = quality or {}
    out = {house()["digest"]: (house_q, "house-0")}
    for e in entries:
        out[e["digest"]] = (quality.get(e["digest"], default), e["digest"][:8])
    return out


class Queue:
    """A sequence of settlements on one state file."""

    def __init__(self, tmp_path: Path, cfg=None):
        self.tmp = Path(tmp_path)
        self.cfg = cfg or lane_d_cfg()
        self.h0 = house()
        self.path = self.tmp / "state" / "lane_d_state.json"
        self.reports: dict[int, dict] = {}

    def settle(self, n: int, entries, pool_dir_recs, *, uid_of=None, house_q="bad",
               quality=None, cfg=None) -> dict:
        pool_dir, recs = pool_dir_recs
        fc = g.forecasts(recs, models_for(entries, house_q=house_q, quality=quality))
        out = g.settle(cfg or self.cfg, boundary(n), pool_dir, entries, self.h0, fc,
                       uids(*entries) if uid_of is None else uid_of, self.path)
        assert out.report is not None, "fixture: lane D is armed at every boundary here"
        self.reports[n] = out.report
        return out.report

    def state_entry(self, dg: str) -> dict:
        return read_state(self.path)["entries"][dg]


def status_reason(q: Queue, report: dict, dg: str) -> tuple[str, str]:
    """``(status, reason)`` of entry ``dg``, asserted identical in the state
    file and the signed report's ``entries``."""
    rec = q.state_entry(dg)
    e = entry_of(report, dg)
    assert (e["status"], e["reason"]) == (rec["status"], rec["reason"]), \
        f"the report's entry {(e['status'], e['reason'])} differs from the state's " \
        f"{(rec['status'], rec['reason'])}"
    return rec["status"], rec["reason"]


def assert_skip(q: Queue, report: dict, dg: str, reason: str, status: str) -> None:
    """``dg`` is typed ``(status, reason)`` in state and report, listed in
    ``skipped`` exactly once with that reason, never judged, has no rows, and
    the report validates against ``lane-d-report/1``."""
    from cascade.validator import lane_d_report

    assert status_reason(q, report, dg) == (status, reason), \
        f"{dg[:12]}: (status, reason) {status_reason(q, report, dg)} != {(status, reason)}"
    assert skips_of(report, dg) == [reason], f"{dg[:12]}: skipped {skips_of(report, dg)}"
    assert dg not in report["judged"], f"{dg[:12]} was judged"
    assert verdict_of(report, dg) is None and not rows_of(report, dg)
    assert reason in lane_d_report.SKIP_REASONS
    lane_d_report.validate_report(report)


def assert_judged(report: dict, dg: str) -> None:
    from cascade.validator import lane_d_report

    assert dg in report["judged"], \
        f"{dg[:12]} not judged (skipped {skips_of(report, dg)}, judged {report['judged']})"
    assert skips_of(report, dg) == [], f"{dg[:12]} judged and skipped {skips_of(report, dg)}"
    lane_d_report.validate_report(report)


def tagged(records, tag: str, dg: str, *, level: int = logging.WARNING) -> list[str]:
    """Messages at ``level`` or above that start with ``tag`` and name the
    entry (its first 12 hex)."""
    return [r.getMessage() for r in records
            if r.levelno >= level and r.getMessage().startswith(tag)
            and dg[:12] in r.getMessage()]


def _docstring_ids(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


def code_literals(path: Path) -> set[str]:
    """String constants of ``path`` that are not docstrings (the rule W5g's
    `_emitted_reasons` scan applies to `lane_d.py`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = _docstring_ids(tree)
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs}
