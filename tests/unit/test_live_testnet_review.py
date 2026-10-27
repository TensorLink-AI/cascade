"""Regression evidence for the live-test harness; no services, keys or rentals."""
from __future__ import annotations

import json
from types import SimpleNamespace

from tests.e2e_testnet import run as E


def test_tracking_uses_current_submission_view(monkeypatch):
    observed = dict(status="done", phase="awaiting_settlement", operator_billed=False, attempts=1)
    doc = {"schema": "cascade-cli/1", "ok": True,
           "data": {"submissions": [dict(hotkey="miner", ref="ref", lane="a", observed=observed,
                                        status="done", freshness={"fresh": True})]}}
    monkeypatch.setattr(E, "cli", lambda *a, **kw: (0, doc, ""))
    ctx = SimpleNamespace(toml="test.toml", args=SimpleNamespace(intake="http://test"))
    entry = dict(scenario="s", hotkey="miner", ref="ref", phases=[])
    E._track(ctx, entry)
    assert entry["status"] == "done" and entry["operator_billed"] is False


def test_failed_preflight_cannot_proceed_to_paid_stage(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(E, "Ctx", lambda args, report: SimpleNamespace(args=args, r=report, save_state=lambda: None))
    monkeypatch.setattr(E, "stage_preflight", lambda ctx: ctx.r.add("preflight", "bad", "FAIL"))
    monkeypatch.setattr(E, "stage_legs", lambda ctx: called.append("paid"))
    rc = E.main(["--stages", "preflight,legs", "--spend", "--state", str(tmp_path / "s"),
                 "--report", str(tmp_path / "r"), "--workdir", str(tmp_path / "w")])
    assert rc == 1 and called == []


def test_cleanup_does_not_treat_failed_listing_as_an_empty_account(monkeypatch):
    monkeypatch.setenv("LIUM_API_KEY", "test-fixture-only")
    for key in ("SHADEFORM_API_KEY", "VAST_API_KEY", "RUNPOD_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(E, "run", lambda *a, **kw: (1, "", "list failed"))
    pods = E.testnet_pods()
    assert pods["lium"] == ["<list failed>"]


def test_rejected_round_cannot_be_used_as_a_successful_settlement(tmp_path, monkeypatch):
    rec = dict(status="rejected", round_id="r", reward_uids=[], weights=[], verdict={},
               participants=[dict(hotkey="miner", gen_ref="ref")])
    ctx = SimpleNamespace(r=E.Report(tmp_path / "report"), state={"entries": {
        "lium": dict(status="done", hotkey="miner", ref="ref")}, "rounds": {"lium": "r"}},
        args=SimpleNamespace(settle_timeout=0, poll=0, network="test"), toml="test.toml",
        chain=SimpleNamespace(weights_for_hotkey=lambda _: []), save_state=lambda: None)
    monkeypatch.setattr(E, "hotkey_of", lambda name: name)
    monkeypatch.setattr(E, "_receipt", lambda *a: rec)
    monkeypatch.setattr(E, "cli", lambda *a, **kw: (0, {"ok": True}, ""))
    E.stage_settlement(ctx)
    checks = json.loads((tmp_path / "report").read_text())["checks"]
    assert any(c["status"] == "FAIL" and c["name"].startswith("receipt_") for c in checks)


def test_stale_or_wrong_submission_is_unknown(monkeypatch):
    doc = {"ok": True, "data": {"submissions": [dict(hotkey="miner", ref="ref", lane="a",
        status="done", observed={"status": "done"}, freshness={"fresh": False})]}}
    monkeypatch.setattr(E, "cli", lambda *a, **kw: (3, doc, ""))
    ctx = SimpleNamespace(toml="test", args=SimpleNamespace(intake="http://test"))
    entry = dict(scenario="s", hotkey="miner", ref="ref", phases=[])
    E._track(ctx, entry)
    assert entry["status"] == "unknown" and entry["operator_billed"] is None


def test_matching_hotkey_with_another_ref_or_only_eligibility_is_not_judged():
    entry = dict(hotkey="miner", ref="new")
    rec = dict(status="scored", verdict={"dethroned": False}, participants=[dict(hotkey="miner", gen_ref="new")],
               manifest={"entries": [dict(role="challenger", miner_hotkey="miner", gen_ref="old")]})
    assert not E._judged_entry(rec, entry)
    rec["manifest"]["entries"][0]["gen_ref"] = "new"
    assert E._judged_entry(rec, entry)


def test_weights_compare_actual_values_not_only_reward_uids():
    rec = dict(weights=[0.95, 0.05, 0.0], reward_uids=[0, 1])
    assert E._weights_agree([0.950001, 0.049999, 0], rec)
    assert not E._weights_agree([0.5, 0.5, 0], rec)
    assert not E._weights_agree([float("nan"), 1, 0], rec)
    assert not E._weights_agree([], rec)


def test_receipt_signature_and_round_binding_are_required(monkeypatch):
    from cascade.shared import receipt as R

    raw = {"round_id": "r", "validator_hotkey": "validator"}
    signed = SimpleNamespace(round_id="r", validator_hotkey="validator")
    monkeypatch.setattr(E, "http_json", lambda *a: raw)
    monkeypatch.setattr(R, "load_receipt", lambda *a: signed)
    monkeypatch.setattr(R, "verify_receipt_signature", lambda *a: False)
    ctx = SimpleNamespace(bucket_url=lambda key: key)
    assert E._receipt(ctx, "validator", "r") is None
    monkeypatch.setattr(R, "verify_receipt_signature", lambda *a: True)
    assert E._receipt(ctx, "validator", "r") == raw
    assert E._receipt(ctx, "validator", "other") is None
    assert E._receipt(ctx, "another-validator", "r") is None


def test_existing_submissions_cannot_be_silently_discarded(tmp_path):
    import pytest

    path = tmp_path / "state"
    path.write_text(json.dumps({"entries": {"lium": {"ref": "ref"}}}))
    args = SimpleNamespace(chain_toml="test.toml", state=str(path), resume=False)
    with pytest.raises(ValueError, match="--resume"):
        E.Ctx(args, None)
    args.resume = True
    assert E.Ctx(args, None).state["entries"]["lium"]["ref"] == "ref"


def test_all_provider_inventories_use_the_configured_testnet_prefix(monkeypatch):
    from cascade.provision import funded

    for env in E.PROVIDER_ENV.values():
        monkeypatch.delenv(env, raising=False)
    for env in ("VAST_API_KEY", "RUNPOD_API_KEY"):
        monkeypatch.setenv(env, "fixture-only")
    prefixes = []
    monkeypatch.setattr(funded, "provider_for_key", lambda *a: SimpleNamespace(
        list_tagged=lambda prefix: prefixes.append(prefix) or ["pod1"]))
    assert E.testnet_pods(888) == {"vast": ["pod1"], "runpod": ["pod1"]}
    assert prefixes == ["cascade-n888-", "cascade-n888-"]


def test_unresolved_submit_intent_prevents_resubmission(tmp_path, monkeypatch):
    ctx = SimpleNamespace(r=E.Report(tmp_path / "report"), args=SimpleNamespace(scenarios=["lium"],
        leg_timeout=0), state={"entries": {"lium": {"submission_unresolved": True}}}, save_state=lambda: None)
    monkeypatch.setattr(E, "_submit", lambda *a: (_ for _ in ()).throw(AssertionError("duplicate submit")))
    E.stage_legs(ctx)
    assert ctx.r.failed() == 1


def test_mainnet_and_bad_scenario_are_refused_before_context(monkeypatch):
    import pytest

    monkeypatch.setattr(E, "Ctx", lambda *a: (_ for _ in ()).throw(AssertionError("must not open context")))
    for args in (["--network", "finney"], ["--scenarios", "typo"], ["--poll", "nan"]):
        with pytest.raises(SystemExit) as exc:
            E.main(args)
        assert exc.value.code == 2
