"""Shared spec and fixture builders for the W14 acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6), the W14 row:

  "(rev6) Unblocked: change 4 was accepted (owner 2026-11-21). Lane D-only
  fast kill: a `lane_d: {status: "absent"}` receipt state with a lane D-only
  burn. Today a missing lane D report makes satellites re-assert the old
  vector for one era (`stale_after_blocks = 3600`) and then burn the whole
  vector (`satellite.py:30-45`)"

Proposed design change 4: "A missing report is a check that can't run: the
satellite re-asserts the last verified vector for `stale_after_blocks`
(3600), then burns the whole vector. Proposed: a structured
`lane_d: {status: "absent"}` receipt state with a lane-D-only burn, reusing
MarketCast's `REPORT_ABSENT` / `INFRA_DEFERRED`."

The settlement chain is W7c's own 48-window `W7cChain`
(`tests/accept/w7c/helpers.py`): boundary(1) waits on the floors,
boundary(2) is X's first verdict, which crowns it (plan rev6.11 (b): no
confirmation duel), boundary(3) judges nothing. No PATHS module is imported at module level: every `cascade.*`
import is inside a function a test body calls (the W0c expect-fail
convention). README.md states the contract.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from tests.accept.w7c import helpers as w7c

REPO = Path(__file__).resolve().parents[3]

# ── the pinned contract (README "Contract") ─────────────────────────────────

ABSENT_KEYS = ("schema", "status", "boundary", "reason", "prior_state_sha256",
               "posterior_state_sha256")
PRESENT_KEYS = w7c.RECEIPT_KEYS
REQUIRED_REASONS = ("report_absent", "infra_deferred")
NEW_RECEIPT_CODES = ("absent_state_change", "absent_with_report")

boundary = w7c.boundary
lane_d_cfg = w7c.lane_d_cfg
unarmed_cfg = w7c.unarmed_cfg
canonical = w7c.canonical
sha = w7c.sha
state_sha = w7c.state_sha
read_bytes = w7c.read_bytes
# rev6.11 (b): a crown is decided at the first verdict, so the chain is W7c's own
# 48-window one (boundary 1 waits on the floors, boundary 2 is X's first,
# crowning, verdict, boundary 3 judges nothing).
Chain = w7c.W7cChain


def has_param(fn, name: str) -> bool:
    return name in inspect.signature(fn).parameters


def absent(chain: Chain, n: int, state_path: Path, *, reason: str = "report_absent",
           cfg=None, **extra):
    """`absent_settlement` at boundary(n) for the fixture chain."""
    from cascade.validator.lane_d import absent_settlement

    return absent_settlement(cfg=cfg if cfg is not None else lane_d_cfg(),
                             boundary=boundary(n), state_path=state_path,
                             house=dict(chain.house), reason=reason, **extra)


def settle_b4(chain: Chain, state_path: Path):
    """A fourth boundary on snapshot 3 (snapshots 1-2 retained): judges
    nothing once X is crowned, and pays the court the state holds."""
    from cascade.validator.lane_d import settle_lane_d
    from cascade.validator.lane_d_bench import SweepSession

    kw = {}
    if w7c.settle_takes_gate_inputs():
        kw.update(boundary_block_hash=w7c.block_hash("b4"), sentinel=None,
                  bench=SweepSession("w14-sess-b4", chain.passing_host()))
    return settle_lane_d(boundary_ts=2_000_000_000 + 12 * int(boundary(4)), cfg=lane_d_cfg(), boundary=boundary(4), pool_dir=chain.pools[2],
                         entries=[dict(chain.x)], house=dict(chain.house),
                         forecasts=chain.forecasts, uid_of=dict(chain.uid_of),
                         state_path=state_path, signer=chain.signer,
                         prior_pool_dirs=list(chain.pools[:2]), **kw)


def dir_snapshot(directory: Path) -> dict:
    """Every file under `directory` with its exact bytes."""
    d = Path(directory)
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def genesis_sha(chain: Chain) -> str:
    """The W7c genesis digest, re-derived through the W7c API on main."""
    from cascade.validator.lane_d_state import genesis, state_digest

    return state_digest(genesis(dict(chain.house)))


def receipt_error_code(excinfo) -> str:
    return getattr(excinfo.value, "code", None)


def json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True).encode()
