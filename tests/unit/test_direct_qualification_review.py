"""Qualification review regressions; no paid resources."""


import pytest

from cascade.funding.direct_ledger import DirectLedger, replay_ledger
from cascade.provision.direct import FakeProvider
from tests.unit import test_direct_moe as helpers

admit = helpers.admit
environment = helpers.environment
guard = helpers.guard


def test_unusable_allocations_cannot_escape_cap(environment, guard):
    ctl = environment[0]

    class Billable(FakeProvider):
        conservative_accounting = True

        def estimate(self, profile, world_size):
            return 100

    provider = Billable(price=100, lease_seconds=-1)
    ctl.providers["fake"] = provider
    admit(environment, guard)
    ctl.reserve("A", now=2)
    ctl.reserve("A", now=20)
    assert provider.reservations == 1
    assert ctl.ledger.state["jobs"]["A"]["spent"] == 100
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_concurrent_intake_during_allocation(environment, guard):
    ctl = environment[0]
    admit(environment, guard)
    path = ctl.ledger.db.execute("PRAGMA database_list").fetchone()[2]
    other = DirectLedger(path, ctl.ledger.state["policy"])

    class Concurrent(FakeProvider):
        def reserve(self, key, **kwargs):
            other.apply(
                "concurrent-charge", "charge", now=10, job="A", charge_key="other", amount=0
            )
            return super().reserve(key, **kwargs)

    ctl.providers["fake"] = Concurrent()
    try:
        ctl.reserve("A", now=2)
        assert ctl.ledger.state["jobs"]["A"]["state"] == "reserved"
        assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state
    finally:
        other.close()


def test_spend_reservation_survives_lost_response(environment, guard):
    ctl = environment[0]

    class Billable(FakeProvider):
        conservative_accounting = True

        def estimate(self, profile, world_size):
            return 100

    provider = Billable(price=100, mode="timeout_after_reserve")
    ctl.providers["fake"] = provider
    admit(environment, guard)
    with pytest.raises(TimeoutError):
        ctl.reserve("A", now=2)
    assert ctl.ledger.state["jobs"]["A"]["spent"] == 100
    ctl.reserve("A", now=3)
    assert provider.reservations == 1
    assert ctl.ledger.state["jobs"]["A"]["spent"] == 100
    assert ctl.ledger.state["jobs"]["A"]["state"] == "reserved"


def test_definitive_no_charge_refund_and_retry(environment, guard):
    ctl = environment[0]

    class Rejected(FakeProvider):
        conservative_accounting = True

        def estimate(self, profile, world_size):
            return 100

        def no_charge_evidence(self, key):
            return "provider_rejected"

    provider = Rejected(price=100, mode="scarcity")
    ctl.providers["fake"] = provider
    admit(environment, guard)
    ctl.reserve("A", now=2)
    ctl.reserve("A", now=20)
    assert ctl.ledger.state["jobs"]["A"]["spent"] == 0
    assert ctl.ledger.state["jobs"]["A"]["state"] == "waiting_capacity"
    assert replay_ledger(ctl.ledger.export()) == ctl.ledger.state


def test_observed_lease_time_is_separate_from_commit_time(environment, guard):
    ctl = environment[0]
    admit(environment, guard)
    ctl.clock = lambda: 100
    ctl.reserve("A", now=2)
    assert ctl.ledger.state["jobs"]["A"]["state"] == "waiting_capacity"
    events = ctl.ledger.export()["events"]
    assert all(e["kind"] != "reserved" for e in events)


