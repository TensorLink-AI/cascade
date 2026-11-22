"""WB17 testnet evidence (plan §3.0 WB17 row "ci + testnet", §4 TB17 / TB18
on netuid 259, §17.1 L4).

``scripts/laned/accept WB17 --evidence evidence/wb17_gate.json`` exports the
file as ``$LANED_EVIDENCE``. The ``test_evidence_*`` tests verify it: schema
``laned_wb17_gate/1``, a testnet run (netuid 259) under
``open_rule = "tokens_king_check"``, a driver state while the gate was closed
(TB18: typed ``gate_closed_reason``, empty record) and one after it opened
(TB17: ``gate_rule`` = the configured rule, ``gate_evidence`` showing the
tokens met and the passing receipt king check the gate read).

Without ``$LANED_EVIDENCE`` they SKIP with a clear reason, so the WP can reach
L3 before the testnet run; L4 needs a green ``accept WB17 --evidence ...`` run
on main. When it IS set, a missing file, invalid JSON or any failing check
fails hard. The ``test_verifier_*`` tests prove the verifier rejects
tampering, offline. Contract: README.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from . import helpers as h


def _evidence() -> dict:
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE is unset: no testnet evidence to verify yet. L4 needs "
                    "`scripts/laned/accept WB17 --evidence evidence/wb17_gate.json` green on "
                    "main (see tests/accept/wb17/README.md for the file)")
    p = Path(path)
    assert p.is_file(), f"LANED_EVIDENCE={path} does not exist"
    try:
        return h.strict_loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise AssertionError(f"LANED_EVIDENCE={path} is not strict JSON: {e}") from None


# ── the testnet evidence ─────────────────────────────────────────────────────


def test_evidence_tb17_tb18_gate_record():
    problems = h.evidence_problems(_evidence())
    assert problems == [], "WB17 testnet evidence rejected:\n  " + "\n  ".join(problems)


# ── the verifier itself (offline) ────────────────────────────────────────────


def test_verifier_accepts_good_evidence():
    assert h.evidence_problems(h.good_evidence()) == []


def _set(path: tuple, value):
    def mutate(ev: dict) -> None:
        node = ev
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return mutate


def _drop(path: tuple):
    def mutate(ev: dict) -> None:
        node = ev
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return mutate


TAMPER = {
    "fixture_source": _set(("source",), "fixture"),
    "mainnet_netuid": _set(("netuid",), 91),
    "wrong_schema": _set(("schema",), "laned_wb17_gate/0"),
    "rule_tokens": _set(("open_rule",), "tokens"),
    "receipt_check_failed": _set(("king_check", "result"), "fail"),
    "closed_reason_free_text": _set(("closed", "gate_closed_reason"), "gate closed (lineage)"),
    "closed_but_opened": _set(("closed", "gate_opened_block"), 5),
    "closed_with_record": _set(("closed", "gate_rule"), "tokens_king_check"),
    "opened_block_zero": _set(("opened", "gate_opened_block"), 0),
    "opened_rule_mismatch": _set(("opened", "gate_rule"), "cogate"),
    "opened_reason_left": _set(("opened", "gate_closed_reason"), "tokens_short"),
    "tokens_short": _set(("opened", "gate_evidence", "lineage_tokens"), h.OPEN_TOKENS - 1),
    "threshold_edited": _set(("opened", "gate_evidence", "open_tokens"), 1),
    "evidence_check_failed": _set(("opened", "gate_evidence", "king_check", "result"), "fail"),
    "evidence_other_round": _set(("opened", "gate_evidence", "king_check", "round_id"),
                                 "v2-r0041"),
    "evidence_other_king": _set(("opened", "gate_evidence", "king_check", "king_pointer"),
                                h.PREV_KING),
    "no_evidence": _set(("opened", "gate_evidence"), {}),
    "no_closed": _drop(("closed",)),
    "no_opened": _drop(("opened",)),
    "nan_tokens": _set(("opened", "gate_evidence", "lineage_tokens"), float("nan")),
}


@pytest.mark.parametrize("name", sorted(TAMPER))
def test_verifier_rejects_tampered_evidence(name):
    ev = json.loads(json.dumps(h.good_evidence()))
    TAMPER[name](ev)
    assert h.evidence_problems(ev) != [], f"verifier accepted tampered evidence ({name})"


def test_verifier_rejects_non_object():
    assert h.evidence_problems([h.good_evidence()]) != []
