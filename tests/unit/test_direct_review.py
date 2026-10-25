"""Regressions from the production-readiness review."""
from dataclasses import replace

import pytest

from cascade.eval.direct_guard import Case, judge_public
from cascade.funding.direct_ledger import replay_ledger
from cascade.shared.direct_moe import SUITES
from tests.unit.test_direct_moe import (  # noqa: F401  (pytest fixtures)
    admit,
    complete,
    environment,
    guard,
    rows,
)


def test_later_retry_preserves_original_event(environment, guard):  # noqa: F811
    ctl = environment[0]
    terms = admit(environment, guard)
    before = ctl.ledger.export()
    ctl.admit("A", terms, now=100)
    assert ctl.ledger.export() == before
    with pytest.raises(ValueError, match="idempotency"):
        ctl.admit("A", {**terms, "payer": "someone-else"}, now=101)


def test_completed_challenger_can_rescore_after_dethronement(environment, guard):  # noqa: F811
    ctl = environment[0]
    admit(environment, guard)
    a = complete(environment, guard)
    ctl.settle("A", now=3)
    admit(environment, guard, "B", role="challenger", reward=0, reference=a, now=4)
    admit(environment, guard, "C", role="challenger", reward=0, reference=a, now=4)
    b = complete(environment, guard, "B", now=5)
    complete(environment, guard, "C", now=5)
    terms = ctl.ledger.state["jobs"]["C"]["terms"]
    checkpoint = ctl.ledger.state["jobs"]["C"]["checkpoint"]
    ctl.settle("B", now=6)
    with pytest.raises(ValueError, match="current cached king"):
        ctl.settle("C", now=7)
    ctl.compare("C", guard, *rows(guard), reference_id=b, private_pass=True, now=8)
    ctl.settle("C", now=9)
    assert ctl.ledger.state["king"] == "C"
    assert ctl.ledger.state["jobs"]["C"]["terms"] == terms
    assert ctl.ledger.state["jobs"]["C"]["checkpoint"] == checkpoint
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_expired_slot_can_be_replaced_with_recorded_authority(environment, guard):  # noqa: F811
    ctl, _, provider, *_ = environment
    terms = admit(environment, guard)
    provider.mode = "revoked"
    ctl.reserve("A", now=2)
    ctl.apply("expired", "expire_funding", 23, job="A", evidence="revoked key uncured")
    ctl.apply("replace", "release_slot", 24, job="A", authorised_by="A", evidence="bootstrap payer replacement")
    ctl.admit("replacement", terms, now=25)
    assert ctl.ledger.state["slots"][terms["slot"]] == "replacement"
    assert ctl.ledger.state["jobs"]["A"]["state"] == "contract_failure"
    assert "A" not in ctl.ledger.state["rewards"]
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_disjoint_suites_do_not_randomly_lose_coverage(guard):  # noqa: F811
    p = replace(guard, cases=tuple(Case(s, str(i), "mase", f"{s}:{i}") for s in SUITES for i in range(5)),
                min_clusters=5, draws=10000)
    k, c = rows(p)
    v = judge_public(p, k, c, reference_id="k", candidate_id="c", private_pass=True)
    assert v["passed"], v["reason"]
    assert v["lcb"] == pytest.approx(.1)


def test_original_v1_journal_replays_unchanged():
    import json
    from pathlib import Path
    receipt = json.loads((Path(__file__).parents[1] / "fixtures/direct_ledger_v1.json").read_text())
    assert replay_ledger(receipt) == receipt["state"]
