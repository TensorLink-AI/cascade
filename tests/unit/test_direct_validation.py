"""Direct-MoE credential intake and check-only validation (no evaluation)."""

import time

import pytest
from bittensor.sp_core import Keypair

from cascade.direct.journal import LATEST, fetch_journal, publish_journal
from cascade.direct.signing import sign
from cascade.funding.intake import FundingIntake, canonical_fund_message, compute_key_digest
from cascade.funding.queue import FundedQueue
from cascade.funding.vault import PayerKeyVault, compute_vaults
from cascade.validator.direct_moe import direct_weights
from tests.unit import test_direct_moe as helpers

environment = helpers.environment
guard = helpers.guard

HOTKEY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def intake(tmp_path, signed):
    vaults = compute_vaults(tmp_path / "vault")
    return FundingIntake(
        FundedQueue(tmp_path / "queue.json"), vaults["lium"],
        resolve_reveal=lambda hk, ref: 7 if ref == "vault/direct@sha256:" + "c" * 64 else None,
        require_signature=True, verify=lambda hk, msg, sig: msg in signed,
        compute_vaults=vaults), vaults


def headers(provider, key, *, sign_provider=None, ts=None):
    ts = ts or str(int(time.time()))
    ref = "vault/direct@sha256:" + "c" * 64
    message = canonical_fund_message("credentials", HOTKEY, ref, ts,
                                     compute_key_digest(sign_provider or provider, key))
    return {"X-Miner-Hotkey": HOTKEY, "X-Commit-Ref": ref, "X-Timestamp": ts,
            "X-Signature": "00", "X-Compute-Provider": provider, "X-Compute-Api-Key": key}, message


def test_credentials_land_only_in_the_chosen_marketplace_vault(tmp_path):
    h, message = headers("vast", "vast-scoped-key")
    service, vaults = intake(tmp_path, {message})
    assert service.direct_credentials(h) == (200, {"status": "stored", "provider": "vast"})
    assert vaults["vast"].get(HOTKEY) == "vast-scoped-key"
    assert vaults["lium"].get(HOTKEY) is None and vaults["runpod"].get(HOTKEY) is None
    # A different process (the direct scheduler) sees the deposit on disk.
    reader = PayerKeyVault(dir=tmp_path / "vault" / "vast")
    assert reader.get_shared(HOTKEY) == "vast-scoped-key"
    # The rolling lane's vault directory never holds the non-Lium key.
    assert not list((tmp_path / "vault").glob("*.json"))


def test_signature_binds_the_provider_as_well_as_the_key(tmp_path):
    h, message = headers("runpod", "k", sign_provider="vast")
    service, vaults = intake(tmp_path, {message})
    assert service.direct_credentials(h)[1]["code"] == "bad_signature"
    assert vaults["runpod"].get(HOTKEY) is None and vaults["vast"].get(HOTKEY) is None


@pytest.mark.parametrize("provider, key, code", [
    ("aws", "k", "unsupported_compute_provider"),
    ("vast", "", "missing_compute_api_key"),
])
def test_credentials_fail_closed(tmp_path, provider, key, code):
    h, message = headers(provider, key)
    service, _ = intake(tmp_path, {message})
    assert service.direct_credentials(h)[1]["code"] == code


class Store:
    def __init__(self):
        self.text = {}

    def put_text(self, key, text, **kwargs):
        self.text[key] = text

    def get_text(self, key):
        if key not in self.text:
            from cascade.shared.hippius import ObjectNotFound

            raise ObjectNotFound(key)
        return self.text[key]

    def put_bytes(self, key, body, **kwargs):
        self.text[key] = body

    get_bytes = get_text


def settled_journal(environment, guard, trainer):
    from dataclasses import asdict

    from cascade.shared.direct_rewards import RewardPolicy

    ctl = environment[0]
    ctl.apply("reward-policy", "reward_policy", 1, policy=asdict(RewardPolicy("1/2", "1/2", "1/4", 20, 5, "burn", "burn")))
    helpers.admit(environment, guard)
    helpers.complete(environment, guard)
    ctl.settle("A", now=3)
    ctl.apply("payout4", "payout_boundary", 4, boundary=4, emission=800, paused=False)
    ctl.apply("settle4", "settle_boundary", 4, boundary=4)
    return sign(ctl.ledger.export(), trainer, scope="journal")


def test_validator_checks_trainer_scores_and_derives_weights(environment, guard):
    trainer = Keypair.create_from_uri("//Alice")
    store = Store()
    key = publish_journal({}, settled_journal(environment, guard, trainer), store=store)
    assert LATEST in store.text and key in store.text
    signed = fetch_journal({}, store=store)
    weights = direct_weights(signed, expected_trainer=trainer.ss58_address,
                             uid_of={"A": 3}.get, n_uids=5, burn_uid=0)
    # King A: 150 throne + 10 continuation; the rest (unowed pool, lane B, dust) burns.
    assert weights == pytest.approx([640 / 800, 0, 0, 160 / 800, 0])
    with pytest.raises(ValueError, match="identity"):
        direct_weights(signed, expected_trainer=Keypair.create_from_uri("//Bob").ss58_address,
                       uid_of={"A": 3}.get, n_uids=5, burn_uid=0)


def test_validator_refuses_a_journal_whose_scores_do_not_replay(environment, guard):
    trainer = Keypair.create_from_uri("//Alice")
    signed = settled_journal(environment, guard, trainer)
    body = signed["body"]
    # The trainer claims a different settlement than its own events derive.
    body["state"]["payout_boundaries"]["4"]["throne_royalties"]["A"] += 1
    forged = sign(body, trainer, scope="journal")
    with pytest.raises(ValueError, match="replay mismatch"):
        direct_weights(forged, expected_trainer=trainer.ss58_address,
                       uid_of={"A": 3}.get, n_uids=5, burn_uid=0)


def test_journal_pointer_must_match_object():
    store = Store()
    trainer = Keypair.create_from_uri("//Alice")
    publish_journal({}, sign({"protocol": "direct-moe-ledger/1"}, trainer, scope="journal"), store=store)
    store.text[LATEST] = store.text[LATEST].replace('"sha256":"', '"sha256":"0')
    with pytest.raises(ValueError, match="pointer"):
        fetch_journal({}, store=store)


def test_retired_panel_reveal_round_trips_and_is_bound():
    import hashlib

    from cascade.direct.journal import fetch_reveal, reveal_panel
    from cascade.shared.direct_moe import digest

    body = b"npz bytes"
    panel = {"object": {"uri": "hippius://b/o/" + hashlib.sha256(body).hexdigest(),
                        "sha256": hashlib.sha256(body).hexdigest()}, "cases": []}
    store = Store()
    assert fetch_reveal({}, digest(panel), store=store) is None  # still private
    assert reveal_panel({}, panel, body, store=store) == digest(panel)
    assert fetch_reveal({}, digest(panel), store=store) == (panel, body)
    with pytest.raises(ValueError, match="descriptor"):
        reveal_panel({}, panel, b"other", store=store)
    store.text[f"direct/reveal/{digest(panel)}.npz"] = b"swapped"
    with pytest.raises(ValueError, match="binding"):
        fetch_reveal({}, digest(panel), store=store)


def test_audit_cli_replays_a_signed_journal(environment, guard, tmp_path, capsys):
    import json

    from cascade.audit.main import main as audit

    trainer = Keypair.create_from_uri("//Alice")
    signed = settled_journal(environment, guard, trainer)
    path = tmp_path / "journal.json"
    path.write_text(json.dumps(signed))
    assert audit(["direct-moe", str(path), "--trainer-hotkey", trainer.ss58_address,
                  "--reproduce", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    # The fixture journal's decisions carry no scoring binding: WARN, never PASS.
    assert report["ok"] and report["checks"][0]["status"] == "PASS"
    assert all(c["status"] == "WARN" for c in report["checks"][1:])
    bob = Keypair.create_from_uri("//Bob").ss58_address
    assert audit(["direct-moe", str(path), "--trainer-hotkey", bob]) == 1


def network_rules(environment, guard, trainer, signed, **overrides):
    from cascade.direct.network import journal_decisions
    from cascade.funding.direct_ledger import replay_ledger
    from cascade.shared.direct_moe import digest

    state = replay_ledger(signed["body"])
    return {"trainer": trainer, "journal": {"bucket": "journal"}, "burn_uid": 0,
            "burn_all": False, "ledger_policy": environment[4],
            "reward_policy": state["reward_policy"], "guards": [digest(guard.to_dict())],
            "private_policies": sorted({digest(d["private"]["policy"])
                                        for d in journal_decisions(state).values()}),
            "private_panels": [], "reveal_after": 3600, "poll_seconds": 60, **overrides}


def publish_rules(store, owner, rules):
    from cascade.direct.network import (
        latest_network_config,
        next_network_config,
        publish_network_config,
    )

    body = next_network_config(rules, latest_network_config({}, owner=owner.ss58_address, store=store))
    publish_network_config({}, sign(body, owner, scope="network-config"), store=store)
    return body


def test_owner_network_config_sets_every_validator_rule(environment, guard):
    from cascade.direct.network import fetch_network_config

    owner, trainer = Keypair.create_from_uri("//Charlie"), Keypair.create_from_uri("//Alice")
    signed = settled_journal(environment, guard, trainer)
    store = Store()
    rules = network_rules(environment, guard, trainer.ss58_address, signed)
    first = publish_rules(store, owner, rules)
    network = fetch_network_config({}, owner=owner.ss58_address, store=store)
    assert network == first and network["seq"] == 1 and network["prev"] is None

    def weights(net, **probes):
        probes = {"available": lambda uri: True, "revealed": lambda panel: True, **probes}
        return direct_weights(signed, uid_of={"A": 3}.get, n_uids=5, network=net, **probes)

    # The local trainer/burn arguments are ignored: the owner config decides.
    assert weights(network) == pytest.approx([640 / 800, 0, 0, 160 / 800, 0])
    assert direct_weights(signed, uid_of={"A": 3}.get, n_uids=5, network=network,
                          available=lambda uri: True, revealed=lambda panel: True,
                          expected_trainer="ignored", burn_uid=4) == weights(network)
    with pytest.raises(ValueError, match="probes required"):
        direct_weights(signed, uid_of={"A": 3}.get, n_uids=5, network=network)
    # Moving the burn uid is a republish, not a validator update.
    moved = publish_rules(store, owner, {**rules, "burn_uid": 2})
    network = fetch_network_config({}, owner=owner.ss58_address, last=network, store=store)
    assert network == moved and moved["prev"] is not None
    assert weights(network) == pytest.approx([0, 0, 640 / 800, 160 / 800, 0])
    assert direct_weights(None, uid_of={}.get, n_uids=5,
                          network={**network, "burn_all": True}) == [0, 0, 1.0, 0, 0]

    for change, error in (
        ({"guards": []}, "guard"),
        ({"ledger_policy": {**rules["ledger_policy"], "max_jobs": 99}}, "ledger policy"),
        ({"trainer": Keypair.create_from_uri("//Bob").ss58_address}, "identity"),
    ):
        with pytest.raises(ValueError, match=error):
            weights({**network, **change})


def test_network_config_refuses_forgery_and_rollback():
    import json

    from cascade.direct.network import LATEST, config_key, fetch_network_config
    from cascade.shared.direct_moe import digest

    owner = Keypair.create_from_uri("//Charlie")
    store = Store()
    rules = {"trainer": "t", "journal": {"bucket": "j"}, "burn_uid": 0, "burn_all": False,
             "ledger_policy": {}, "reward_policy": None, "guards": [], "private_policies": [],
             "private_panels": [], "reveal_after": 3600, "poll_seconds": 60}
    with pytest.raises(ValueError, match="no network config"):
        fetch_network_config({}, owner=owner.ss58_address, store=store)
    first = publish_rules(store, owner, rules)
    old_pointer = store.text[LATEST]
    second = publish_rules(store, owner, {**rules, "burn_all": True})
    seen = fetch_network_config({}, owner=owner.ss58_address, last=first, store=store)
    assert seen == second
    store.text[LATEST] = old_pointer
    with pytest.raises(ValueError, match="rolled back"):
        fetch_network_config({}, owner=owner.ss58_address, last=second, store=store)
    with pytest.raises(ValueError, match="identity"):
        fetch_network_config({}, owner=Keypair.create_from_uri("//Bob").ss58_address, store=store)
    # A same-key config signed by anyone but the owner is refused.
    mallory = Keypair.create_from_uri("//Mallory")
    forged = sign({**second, "burn_all": False}, mallory, scope="network-config")
    forged["signer"] = owner.ss58_address
    store.text[config_key(forged)] = json.dumps(forged)
    store.text[LATEST] = json.dumps({"key": config_key(forged), "sha256": digest(forged)})
    with pytest.raises(ValueError, match="signature"):
        fetch_network_config({}, owner=owner.ss58_address, store=store)


def test_validator_cli_takes_its_rules_from_the_owner_config(environment, guard, monkeypatch, capsys):
    import cascade.direct.journal as journal
    from cascade.validator.main import main as validator

    owner, trainer = Keypair.create_from_uri("//Charlie"), Keypair.create_from_uri("//Alice")
    signed = settled_journal(environment, guard, trainer)
    store = Store()
    monkeypatch.setattr(journal, "_store", lambda config: store)
    publish_journal({}, signed, store=store)
    publish_rules(store, owner, network_rules(environment, guard, trainer.ss58_address, signed))
    args = ["--offline", "--direct-owner-hotkey", owner.ss58_address, "--direct-config-bucket", "cfg"]
    assert validator(args) == 0
    assert "verified and replayed" in capsys.readouterr().out
    # The owner drops the journal's guard: every validator refuses it, no restart.
    publish_rules(store, owner, network_rules(environment, guard, trainer.ss58_address, signed, guards=[]))
    with pytest.raises(ValueError, match="guard"):
        validator(args)


def test_overdue_panel_reveal_burns_the_boundary(environment, guard, monkeypatch):
    import cascade.direct.network as network_module

    trainer = Keypair.create_from_uri("//Alice")
    signed = settled_journal(environment, guard, trainer)
    network = {**network_rules(environment, guard, trainer.ss58_address, signed), "burn_uid": 1}
    seen = {}

    def obligations(journal, state, config, *, available, revealed, boundary):
        seen["boundary"] = boundary
        return ["f" * 64]

    monkeypatch.setattr(network_module, "audit_obligations", obligations)
    weights = direct_weights(signed, uid_of={"A": 3}.get, n_uids=5, network=network,
                             available=lambda uri: True, revealed=lambda panel: False)
    assert weights == [0, 1.0, 0, 0, 0] and seen["boundary"] == 4


class HeadClient:
    def __init__(self, error=None):
        self.error = error

    def head_object(self, **kwargs):
        if self.error:
            raise self.error


def test_public_exists_reads_private_as_absent_and_faults_as_errors():
    from botocore.exceptions import ClientError

    from cascade.direct.storage import public_exists

    def error(code, status):
        return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "HeadObject")

    assert public_exists("b", "k", client=HeadClient())
    assert not public_exists("b", "k", client=HeadClient(error("404", 404)))
    assert not public_exists("b", "k", client=HeadClient(error("403", 403)))
    with pytest.raises(OSError):
        public_exists("b", "k", client=HeadClient(error("500", 500)))


def test_network_config_is_anchored_to_its_chain_commitment():
    from cascade.direct.network import commitment_payload, fetch_network_config

    owner = Keypair.create_from_uri("//Charlie")
    store = Store()
    rules = {"trainer": "t", "journal": {"bucket": "j"}, "burn_uid": 0, "burn_all": False,
             "ledger_policy": {}, "reward_policy": None, "guards": [], "private_policies": [],
             "private_panels": [{"panel": "a" * 64, "from_block": 100}],
             "reveal_after": 3600, "poll_seconds": 60}

    def fetch(last, committed):
        return fetch_network_config({}, owner=owner.ss58_address, last=last,
                                    committed=committed, store=store)

    first = publish_rules(store, owner, rules)
    # First sight trusts the schedule but only the committed config.
    assert fetch(None, (commitment_payload(first), 500)) == first
    with pytest.raises(ValueError, match="committed on chain"):
        fetch(None, ("cascade-direct:1:" + "0" * 64, 500))
    # A panel added later must start at or after its commitment block.
    late = publish_rules(store, owner, {**rules, "private_panels": rules["private_panels"] + [
        {"panel": "b" * 64, "from_block": 900}]})
    assert fetch(first, (commitment_payload(late), 900)) == late
    with pytest.raises(ValueError, match="before its config was committed"):
        fetch(first, (commitment_payload(late), 901))
    # Re-timing a known panel into the past is a new listing, and refused.
    moved = publish_rules(store, owner, {**rules, "private_panels": [
        {"panel": "a" * 64, "from_block": 50}, {"panel": "b" * 64, "from_block": 900}]})
    with pytest.raises(ValueError, match="before its config was committed"):
        fetch(late, (commitment_payload(moved), 950))
    with pytest.raises(ValueError, match="strictly increasing"):
        publish_rules(store, owner, {**rules, "private_panels": [
            {"panel": "a" * 64, "from_block": 5}, {"panel": "b" * 64, "from_block": 5}]})


def test_plain_commitment_returns_payload_and_chain_block():
    from cascade.shared.chain import ChainClient

    client = ChainClient(netuid=91, network="test")
    record = {"block": 4321, "info": {"fields": [{"Raw20": b"cascade-direct:1:abc"}]}}
    client.pending_commitment = lambda hotkey: record
    assert client.plain_commitment("owner") == ("cascade-direct:1:abc", 4321)
    client.pending_commitment = lambda hotkey: None
    assert client.plain_commitment("owner") is None
