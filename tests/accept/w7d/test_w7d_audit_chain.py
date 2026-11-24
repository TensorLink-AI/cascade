"""W7d (rev6.18 B2, owner 2026-11-23): "One lane D checker for outside
validators. The running satellite verifies through audit_receipt ->
check_lane_d. [...] delete the other". The satellite replay's rules now live
in the audit (`cascade.audit.lane_d_chain`) and reach outside validators
only through `check_lane_d` / `lane_d_findings`:

- per report, ported into `check_lane_d` itself (burn-class, rev6.18 B3):
  the paid court is KING_D then `prior_kings` with each slot's hotkey, uid
  and `burn_reason` (C5; `uid_of` pins each uid to the metagraph), and the
  whole `king_after` record (C2/C3 crown);
- chain mode (`chain=`): the chain rules (receipt links C1, `king_before`,
  `prior_kings`, holds, used windows, the house challenger, entry progress,
  state snapshots) as alert-class codes.

On main `check_lane_d` takes no `chain` / `uid_of` (TypeError) and checks
none of these.
"""

from __future__ import annotations

import copy

import pytest

from .helpers import build_all, digest, doctored


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-audit-chain"))


def _findings(pub, n, *, reports=None, receipts=None, uid_of=None, states=None):
    from cascade.audit.lane_d import lane_d_findings

    reports = reports if reports is not None else pub.reports()
    receipts = receipts if receipts is not None else pub.receipts
    b = pub.boundaries[n - 1]
    prior = [reports[x] for x in pub.boundaries[:n - 1]]
    chain = {"receipts": receipts[:n], "uid_of": dict(pub.uid_of)}
    if states is not None:
        chain["states"] = states
    return lane_d_findings(reports[b], pub.cfg, signer_hotkey=pub.signer_ss58,
                           prior_reports=prior, chain=chain,
                           uid_of=(uid_of if uid_of is not None else pub.uid_of.get(b)))


@pytest.mark.parametrize("name", ["gated", "held", "house_bar", "refresh"])
def test_an_honest_chain_has_no_finding_in_chain_mode(chains, name):
    pub = chains[name].pub
    for n in range(1, len(pub.boundaries) + 1):
        f = _findings(pub, n, states=dict(pub.states))
        assert not f.findings and not f.transient, (name, n, f.findings, f.transient)


def test_a_broken_receipt_link_is_an_alert_not_a_burn(chains):
    """C1 (replay-only before rev6.18): the receipt's prior is not the
    previous posterior. `ld_chain`, alert-class: the court stays paid."""
    pub = chains["gated"].pub
    receipts = copy.deepcopy(pub.receipts)
    receipts[2]["prior_state_sha256"] = digest("w7d a state nobody published")
    f = _findings(pub, 3, receipts=receipts)
    assert f.burn is None, f.findings
    assert any(c == "ld_chain" and "receipt.prior_state_sha256" in d for c, d in f.alerts), \
        f.findings


def test_a_court_uid_off_the_metagraph_burns(chains):
    """C5 (ported per report): `uid_of` pins the slot's uid (`ld_court`)."""
    pub = chains["gated"].pub
    n = next(i for i in range(1, len(pub.boundaries) + 1)
             if pub.report(i)["pay"]["court"][0]["hotkey"] is not None)
    b = pub.boundaries[n - 1]
    hk = pub.report(n)["pay"]["court"][0]["hotkey"]
    meta = dict(pub.uid_of[b])
    meta[hk] = int(meta[hk]) + 1
    f = _findings(pub, n, uid_of=meta)
    assert f.burn is not None and f.burn[0] == "ld_court", f.findings


def test_a_burn_reason_off_the_slot_burns(chains):
    """C5 (ported per report): a paid slot states a `burn_reason`."""
    pub = chains["gated"].pub
    n = next(i for i in range(1, len(pub.boundaries) + 1)
             if pub.report(i)["pay"]["court"][0]["hotkey"] is not None)

    def lie(rep):
        rep["pay"]["court"][0]["burn_reason"] = "deregistered"

    reports, receipts = doctored(pub, n, lie)
    f = _findings(pub, n, reports=reports, receipts=receipts)
    assert f.burn is not None and f.burn[0] == "ld_court", f.findings


def test_a_king_after_hotkey_lie_burns(chains):
    """The whole `king_after` record (ported per report): a crowned miner's
    `king_after.hotkey` must be its entry's (`ld_crown`)."""
    pub = chains["gated"].pub
    n = next(i for i in range(1, len(pub.boundaries) + 1) if pub.report(i).get("crowned"))

    def lie(rep):
        rep["king_after"]["reveal_ts"] = int(rep["king_after"]["reveal_ts"]) + 1

    reports, receipts = doctored(pub, n, lie)
    f = _findings(pub, n, reports=reports, receipts=receipts)
    assert f.burn is not None and f.burn[0] in ("ld_crown", "ld_court"), f.findings


def test_the_satellite_package_is_gone():
    """B2: no second checker module remains for outside validators."""
    import importlib.util

    assert importlib.util.find_spec("cascade.satellite") is None
