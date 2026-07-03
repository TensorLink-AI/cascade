"""cascade-audit: Tier-0 checks on the fixture receipt, one tamper test per
check, Tier-1 corpus re-derivation against the repo's example generator, and
the CLI surface (exit codes, JSON)."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import pytest

from cascade.audit import checks as C
from cascade.audit.main import fetch_receipt_text
from cascade.audit.main import main as audit_main
from cascade.audit.rederive import run_tier1
from cascade.shared.receipt import dump_receipt, sign_receipt

from .receipt_fixture import (
    EPOCH_START,
    make_rejected_receipt,
    make_scored_receipt,
)

bt = pytest.importorskip("bittensor")

TRAINER_KP = bt.Keypair.create_from_uri("//Trainer")
VALIDATOR_KP = bt.Keypair.create_from_uri("//Validator")


@pytest.fixture(scope="module")
def audit_cfg(cfg):
    """chain.toml with both trust anchors pinned to the fixture keypairs."""
    return replace(cfg, manifest=replace(
        cfg.manifest,
        trainer_hotkey=TRAINER_KP.ss58_address,
        validator_hotkey=VALIDATOR_KP.ss58_address,
    ))


@pytest.fixture(scope="module")
def signed_receipt(audit_cfg):
    receipt, _, _ = make_scored_receipt(
        audit_cfg, validator_hotkey=VALIDATOR_KP.ss58_address, trainer_wallet=TRAINER_KP
    )
    return sign_receipt(receipt, VALIDATOR_KP)


def _by_name(results):
    return {r.name: r for r in results}


# ── the acceptance shape: untampered receipt has zero FAILs ───────────────────


def test_tier0_untampered_receipt_never_fails(audit_cfg, signed_receipt):
    results = _by_name(C.run_tier0(signed_receipt, audit_cfg, client=None))
    fails = [r for r in results.values() if r.status == C.FAIL]
    assert fails == [], f"unexpected FAILs: {fails}"
    # signature/seed/digest/verdict/weights all PASS with no chain
    for name in ("receipt-signature", "manifest-signature", "base-seed", "round-seeds",
                 "epoch-alignment", "contract-digest", "base-arch-digest",
                 "koth-params", "verdict", "transition"):
        assert results[name].status == C.PASS, results[name]
    # chain-dependent halves WARN explicitly, never silently pass
    assert results["block-hash-onchain"].status == C.WARN
    assert results["weights"].status == C.WARN  # no chain to compare against


def test_tier0_rejected_receipt(audit_cfg):
    receipt = make_rejected_receipt(
        audit_cfg, reason="signature_invalid", validator_hotkey=VALIDATOR_KP.ss58_address
    )
    receipt = sign_receipt(receipt, VALIDATOR_KP)
    results = _by_name(C.run_tier0(receipt, audit_cfg, client=None))
    assert results["status"].status == C.PASS
    assert "signature_invalid" in results["status"].detail
    assert results["verdict"].status == C.SKIP
    assert results["weights"].status == C.SKIP
    # the fixture's placeholder manifest signature re-detects the recorded
    # rejection: the audit CONFIRMS the gate instead of failing the receipt
    assert results["manifest-signature"].status == C.PASS
    assert "gate was right" in results["manifest-signature"].detail
    assert not any(r.status == C.FAIL for r in results.values())


# ── one tamper per check ──────────────────────────────────────────────────────


def _tamper_manifest(receipt, **overrides):
    m = dict(receipt.manifest)
    m.update(overrides)
    return replace(receipt, manifest=m)


def test_tamper_receipt_signature(audit_cfg, signed_receipt):
    tampered = replace(signed_receipt, round_id="999")
    r = C.check_receipt_signature(tampered, audit_cfg)
    assert r.status == C.FAIL


def test_tamper_wrong_signer_rejected_when_pinned(audit_cfg):
    receipt, _, _ = make_scored_receipt(
        audit_cfg, validator_hotkey=TRAINER_KP.ss58_address, trainer_wallet=TRAINER_KP
    )
    signed = sign_receipt(receipt, TRAINER_KP)  # signed, but by the WRONG key
    r = C.check_receipt_signature(signed, audit_cfg)
    assert r.status == C.FAIL and "pinned" in r.detail


def test_unpinned_signer_warns(cfg, signed_receipt):
    # cfg without validator_hotkey: valid signature ⇒ WARN (self-declared signer)
    r = C.check_receipt_signature(signed_receipt, cfg)
    assert r.status == C.WARN and "self-declared" in r.detail


def test_tamper_manifest_signature(audit_cfg, signed_receipt):
    tampered = _tamper_manifest(signed_receipt, eval_dataset="benchmark-overfit-v1")
    r = C.check_manifest_signature(tampered, audit_cfg)
    assert r.status == C.FAIL


def test_tamper_block_hash_fails_base_seed(audit_cfg, signed_receipt):
    tampered = replace(signed_receipt, epoch_block_hash="0x" + "cd" * 32)
    assert C.check_base_seed(tampered).status == C.FAIL


def test_tamper_round_id_fails_base_seed(signed_receipt):
    assert C.check_base_seed(replace(signed_receipt, round_id="7")).status == C.FAIL


def test_tamper_generation_seed_fails_round_seeds(audit_cfg, signed_receipt):
    tampered = replace(signed_receipt, generation_seed=signed_receipt.generation_seed + 1)
    assert C.check_round_seeds(tampered, audit_cfg).status == C.FAIL


def test_tamper_training_seed_fails_round_seeds(audit_cfg, signed_receipt):
    tampered = replace(signed_receipt, training_seed=signed_receipt.training_seed ^ 1)
    assert C.check_round_seeds(tampered, audit_cfg).status == C.FAIL


def test_tamper_epoch_boundary_fails_alignment(audit_cfg, signed_receipt):
    tampered = replace(signed_receipt, epoch_start_block=EPOCH_START + 3)
    assert C.check_epoch_alignment(tampered, audit_cfg).status == C.FAIL


def test_tamper_contract_digest(audit_cfg, signed_receipt):
    tampered = _tamper_manifest(signed_receipt, contract_digest="0" * 64)
    assert C.check_contract_digest(tampered, audit_cfg).status == C.FAIL


def test_tamper_base_arch_digest(audit_cfg, signed_receipt):
    tampered = _tamper_manifest(signed_receipt, base_arch_digest="0" * 64)
    assert C.check_base_arch_digest(tampered, audit_cfg).status == C.FAIL


def test_tamper_late_commit_fails_cutoff(signed_receipt):
    late = replace(signed_receipt.participants[1], commit_block=EPOCH_START)
    tampered = replace(signed_receipt,
                       participants=(signed_receipt.participants[0], late))
    r = C.check_commit_cutoff(tampered)
    assert r.status == C.FAIL and "after the boundary" in r.detail


def test_tamper_entry_gen_ref_fails_cutoff(signed_receipt):
    m = dict(signed_receipt.manifest)
    m["entries"] = [dict(e) for e in m["entries"]]
    m["entries"][1]["gen_ref"] = "mallory/gen@sha256:" + "f" * 64
    tampered = replace(signed_receipt, manifest=m)
    r = C.check_commit_cutoff(tampered)
    assert r.status == C.FAIL and "committed" in r.detail


def test_missing_participants_warns_not_passes(signed_receipt):
    r = C.check_commit_cutoff(replace(signed_receipt, participants=()))
    assert r.status == C.WARN


def test_tamper_params_fails_koth_params(audit_cfg, signed_receipt):
    v = signed_receipt.verdict
    params = dict(v.params)
    params["bootstrap_B"] = params["bootstrap_B"] + 1
    tampered = replace(signed_receipt, verdict=replace(v, params=params))
    r = C.check_koth_params(tampered, audit_cfg)
    assert r.status == C.FAIL and "bootstrap_B" in r.detail


def test_tamper_score_fails_verdict(signed_receipt):
    es = signed_receipt.entry_scores
    chal = es[1]
    # dope one window WORSE for the challenger: the bootstrap's lower tail
    # (bags containing it) moves, so the recomputed LCB must diverge
    doped = replace(chal.scores[0], mase=chal.scores[0].mase * 100.0)
    tampered = replace(
        signed_receipt,
        entry_scores=(es[0], replace(chal, scores=(doped, *chal.scores[1:]))),
    )
    r = C.check_verdict(tampered)
    assert r.status == C.FAIL and "lcb" in r.detail


def test_tamper_lcb_fails_verdict(signed_receipt):
    v = signed_receipt.verdict
    tampered = replace(signed_receipt, verdict=replace(v, lcb=(v.lcb or 0.0) + 0.1))
    assert C.check_verdict(tampered).status == C.FAIL


def test_tamper_win_bit_fails_verdict(signed_receipt):
    v = signed_receipt.verdict
    tampered = replace(
        signed_receipt,
        verdict=replace(v, challenger_wins_round=not v.challenger_wins_round),
    )
    assert C.check_verdict(tampered).status == C.FAIL


def test_tamper_dethrone_fails_transition(signed_receipt):
    v = signed_receipt.verdict
    assert v.dethroned  # the fixture's challenger wins under dethrone_cp=1
    tampered = replace(signed_receipt, verdict=replace(v, dethroned=False))
    r = C.check_transition(tampered)
    assert r.status == C.FAIL


def test_tamper_resulting_king_fails_transition(signed_receipt):
    v = signed_receipt.verdict
    tampered = replace(signed_receipt, verdict=replace(v, king_hotkey="mallory_hk"))
    assert C.check_transition(tampered).status == C.FAIL


def test_tamper_weights_fails(audit_cfg, signed_receipt):
    w = list(signed_receipt.weights)
    w[0], w[1] = 0.9, 0.1
    tampered = replace(signed_receipt, weights=tuple(w))
    r = C.check_weights(tampered, audit_cfg)
    assert r.status == C.FAIL and "equal_share_vector" in r.detail


def test_tamper_reward_uids_fails_weights(audit_cfg, signed_receipt):
    tampered = replace(signed_receipt, reward_uids=(3,))
    assert C.check_weights(tampered, audit_cfg).status == C.FAIL


def test_missing_weights_warns(audit_cfg, signed_receipt):
    r = C.check_weights(replace(signed_receipt, weights=(), reward_uids=()), audit_cfg)
    assert r.status == C.WARN


# ── chain-backed halves ───────────────────────────────────────────────────────


class _FakeChain:
    """Fake client serving exactly the fixture's chain state."""

    def __init__(self, receipt):
        self._receipt = receipt

    def block_hash(self, block):
        assert block == self._receipt.epoch_start_block
        return self._receipt.epoch_block_hash

    def poll_commitments(self):
        from cascade.shared.chain import Commitment

        return [
            Commitment(p.uid, p.hotkey, None, f"metro-v1:gen:hippius:{p.gen_ref}",
                       p.commit_block)
            for p in self._receipt.participants
        ]

    def weights_for_hotkey(self, hotkey):
        return list(self._receipt.weights)


def test_chain_checks_pass_with_matching_chain(audit_cfg, signed_receipt):
    client = _FakeChain(signed_receipt)
    results = _by_name(C.run_tier0(signed_receipt, audit_cfg, client=client))
    assert results["block-hash-onchain"].status == C.PASS
    assert results["commit-cutoff"].status == C.PASS
    assert results["weights"].status == C.PASS
    assert not any(r.status in (C.FAIL, C.WARN) for r in results.values()), results


def test_chain_hash_mismatch_fails(audit_cfg, signed_receipt):
    client = _FakeChain(signed_receipt)
    client.block_hash = lambda block: "0x" + "ee" * 32
    assert C.check_block_hash_onchain(signed_receipt, client).status == C.FAIL


def test_chain_payload_contradiction_fails_cutoff(signed_receipt):
    client = _FakeChain(signed_receipt)
    orig = client.poll_commitments

    def lie():
        commits = orig()
        c = commits[0]
        from cascade.shared.chain import Commitment

        commits[0] = Commitment(c.uid, c.hotkey, None,
                                "metro-v1:gen:hippius:mallory/gen@sha256:" + "f" * 64,
                                c.commit_block)
        return commits

    client.poll_commitments = lie
    r = C.check_commit_cutoff(signed_receipt, client)
    assert r.status == C.FAIL and "chain shows" in r.detail


def test_onchain_weight_support_mismatch_fails(audit_cfg, signed_receipt):
    client = _FakeChain(signed_receipt)
    client.weights_for_hotkey = lambda hk: [1.0] + [0.0] * (len(signed_receipt.weights) - 1)
    r = C.check_weights(signed_receipt, audit_cfg, client)
    # fixture rewards uids (1, 0); a 1.0-on-uid-0 row is a different support
    assert r.status == C.FAIL


# ── Tier 1: corpus re-derivation against the example generator ───────────────


@pytest.fixture()
def tier1_setup(small_cfg, example_generator_dir, monkeypatch, tmp_path):
    """A cache_reuse config + a receipt whose corpus digests are the example
    generator's REAL digests, with the Hub fetch redirected to the local dir."""
    from cascade.trainer.corpus import build_round_corpus

    cfg = replace(small_cfg,
                  training=replace(small_cfg.training, corpus_mode="cache_reuse"),
                  manifest=replace(small_cfg.manifest,
                                   trainer_hotkey=TRAINER_KP.ss58_address,
                                   validator_hotkey=VALIDATOR_KP.ss58_address))
    receipt, _, _ = make_scored_receipt(
        cfg, validator_hotkey=VALIDATOR_KP.ss58_address, trainer_wallet=TRAINER_KP
    )
    real = build_round_corpus(
        example_generator_dir, receipt.generation_seed, cfg.generator, "cache_reuse",
        use_sandbox=True, blocked=cfg.static_guard.blocked,
    )
    m = dict(receipt.manifest)
    m["entries"] = [dict(e) for e in m["entries"]]
    for e in m["entries"]:
        e["corpus_digest"] = real.digest
    receipt = replace(receipt, manifest=m)

    import cascade.audit.rederive as rederive

    def fake_fetch(gen_ref, dest, cfg_):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copytree(example_generator_dir, dest)
        return dest

    monkeypatch.setattr(rederive, "_fetch_generator", fake_fetch)
    return cfg, receipt, tmp_path


def test_tier1_rederives_example_generator_corpus(tier1_setup):
    cfg, receipt, workdir = tier1_setup
    results = run_tier1(receipt, cfg, workdir=workdir)
    assert len(results) == 2  # king + challenger
    assert all(r.status == C.PASS for r in results), results


def test_tier1_tampered_corpus_digest_fails(tier1_setup):
    cfg, receipt, workdir = tier1_setup
    m = dict(receipt.manifest)
    m["entries"] = [dict(e) for e in m["entries"]]
    m["entries"][0]["corpus_digest"] = "9" * 64
    tampered = replace(receipt, manifest=m)
    results = run_tier1(tampered, cfg, workdir=workdir)
    assert results[0].status == C.FAIL and "re-derived corpus digest" in results[0].detail
    assert results[1].status == C.PASS  # only the tampered entry fails


def test_tier1_stream_mode_warns_without_full_stream(cfg, tmp_path):
    # shipped chain.toml is stream_cpu: Tier 1 must WARN, not silently pass
    receipt, _, _ = make_scored_receipt(cfg)
    results = run_tier1(receipt, cfg, workdir=tmp_path)
    assert all(r.status == C.WARN for r in results)
    assert "--full-stream" in results[0].detail


def test_tier1_rejected_round_skips(cfg, tmp_path):
    receipt = make_rejected_receipt(cfg)
    results = run_tier1(receipt, cfg, workdir=tmp_path)
    assert [r.status for r in results] == [C.SKIP]


# ── CLI surface ───────────────────────────────────────────────────────────────


def _write_receipt(tmp_path, receipt, name="r.json"):
    p = tmp_path / name
    p.write_text(dump_receipt(receipt), encoding="utf-8")
    return p


def test_cli_json_and_exit_zero_on_clean_receipt(audit_cfg, signed_receipt, tmp_path,
                                                 capsys, monkeypatch):
    import cascade.audit.main as audit_mod

    monkeypatch.setattr(audit_mod, "load_chain_config", lambda p: audit_cfg)
    path = _write_receipt(tmp_path, signed_receipt)
    rc = audit_main(["round", signed_receipt.round_id, "--receipt", str(path),
                     "--no-chain", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["ok"] is True and out["round_id"] == signed_receipt.round_id
    assert {c["status"] for c in out["checks"]} <= {"PASS", "WARN", "SKIP"}


def test_cli_exit_nonzero_on_tampered_receipt(audit_cfg, signed_receipt, tmp_path,
                                              capsys, monkeypatch):
    import cascade.audit.main as audit_mod

    monkeypatch.setattr(audit_mod, "load_chain_config", lambda p: audit_cfg)
    tampered = _tamper_manifest(signed_receipt, contract_digest="0" * 64)
    path = _write_receipt(tmp_path, tampered)
    rc = audit_main(["round", tampered.round_id, "--receipt", str(path), "--no-chain"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] contract-digest" in out


def test_cli_rejects_round_id_mismatch(audit_cfg, signed_receipt, tmp_path, monkeypatch):
    import cascade.audit.main as audit_mod

    monkeypatch.setattr(audit_mod, "load_chain_config", lambda p: audit_cfg)
    path = _write_receipt(tmp_path, signed_receipt)
    assert audit_main(["round", "12345", "--receipt", str(path), "--no-chain"]) == 2


def test_fetch_receipt_falls_back_to_credentialed(cfg, monkeypatch):
    import cascade.audit.main as audit_mod

    monkeypatch.setattr(audit_mod, "_unsigned_s3_text",
                        lambda cfg_, key: (_ for _ in ()).throw(RuntimeError("403")))

    class _Store:
        def __init__(self, s3cfg):
            pass

        def get_text(self, key):
            assert key == "receipts/round-9.json"
            return '{"ok": 1}'

    import cascade.shared.hippius as hippius

    monkeypatch.setattr(hippius, "S3Store", lambda s3cfg: _Store(s3cfg))
    assert fetch_receipt_text(cfg, "9") == '{"ok": 1}'
