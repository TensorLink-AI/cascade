"""DEC-CA-0043 trainer-side pieces outside the scheduler: the promotion
record's ``effective_era``, the persistent dedup registry, the per-leg
rent-wait deadline, and the rolling roster's audit claim."""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from cascade.audit import checks as C
from cascade.shared.chain import Commitment
from cascade.shared.promotion import (
    PromotionRecord,
    dump_promotion_record,
    load_promotion_record,
)
from cascade.trainer.dedup_registry import DedupRegistry
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.promotion import Candidate, TrainerPromotion

REF_A = "alfa/gen@sha256:" + "1" * 64
REF_B = "brav/gen@sha256:" + "2" * 64
REF_C = "char/gen@sha256:" + "3" * 64


# ── promotion record: effective_era ──────────────────────────────────────────

def test_promotion_record_effective_era_is_signed_drop_when_zero():
    rec = PromotionRecord(generation=3, king_hotkey="k", fired_round="r", fired_block=100)
    assert b"effective_era" not in rec.canonical_body()
    stamped = PromotionRecord(generation=3, king_hotkey="k", fired_round="r", fired_block=100,
                              effective_era=17)
    assert b'"effective_era":17' in stamped.canonical_body()
    assert load_promotion_record(dump_promotion_record(stamped)).effective_era == 17
    assert load_promotion_record(dump_promotion_record(rec)).effective_era == 0


def test_engine_stamps_and_persists_effective_era(tmp_path):
    eng = TrainerPromotion(reign_threshold=1.0, k_max=2, quality_epsilon=0.05,
                           state_path=tmp_path / "state.json",
                           pointer_path=tmp_path / "ptr.json")
    eng.king_hotkey = "king"
    eng.reign_start_block = 0
    eng.candidates = (Candidate(checkpoint_id="cascade/ckpt@sha256:" + "a" * 64,
                                size="toto2-4m", hotkey="hkA", role="king",
                                round_id="r1", epoch_index=1, score=0.9),)
    rec = eng.maybe_promote(epoch_block=7200 * 3, round_id="r3", effective_era=9)
    assert rec is not None and rec.effective_era == 9
    assert eng.pending_record.effective_era == 9
    again = TrainerPromotion.load(reign_threshold=1.0, k_max=2, quality_epsilon=0.05,
                                  state_path=tmp_path / "state.json",
                                  pointer_path=tmp_path / "ptr.json")
    assert again.pending_record is not None and again.pending_record.effective_era == 9
    # a pre-era fire keeps the record byte-identical to before the field
    eng2 = TrainerPromotion(reign_threshold=1.0, k_max=2, quality_epsilon=0.05)
    eng2.king_hotkey, eng2.reign_start_block = "king", 0
    eng2.candidates = eng.candidates or (Candidate(
        checkpoint_id="cascade/ckpt@sha256:" + "b" * 64, size="toto2-4m", hotkey="hkB",
        role="king", round_id="r1", epoch_index=1, score=0.9),)
    rec2 = eng2.maybe_promote(epoch_block=7200 * 3, round_id="r3")
    assert rec2 is not None and b"effective_era" not in rec2.canonical_body()


# ── the persistent dedup registry ────────────────────────────────────────────

def _commit(hk, ref, block):
    return Commitment(uid=1, hotkey=hk, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


def _registry(tmp_path, mode, prints):
    runner = SimpleNamespace(cfg=SimpleNamespace(round=SimpleNamespace(
        dedup_max_tokens=50_000, dedup_max_text_mb=4)), work_root=tmp_path)
    reg = DedupRegistry(runner, tmp_path / "dedup_registry.json", mode=mode)
    reg.fingerprint = lambda ref: prints.get(ref)
    return reg


def test_registry_drops_the_later_commit_of_an_identical_tree(tmp_path):
    same = {"tree_sha256": "T", "token_sha256": "K", "masked_sha256": "M"}
    other = {"tree_sha256": "t2", "token_sha256": "k2", "masked_sha256": "m2"}
    reg = _registry(tmp_path, "enforce", {REF_A: same, REF_B: dict(same), REF_C: other})
    history = [_commit("ALFA", REF_A, 100), _commit("BRAV", REF_B, 200), _commit("CHAR", REF_C, 50)]
    gen = lambda hk, ref, rb: SimpleNamespace(hotkey=hk, ref=ref, reveal_block=rb)  # noqa: E731
    assert reg.admit(gen("ALFA", REF_A, 100), None, history) is None
    # BRAV committed later with the same tree ⇒ dropped, ALFA named
    assert reg.admit(gen("BRAV", REF_B, 200), None, history) == ("ALFA", "tree_identical", True)
    # a different tree passes and is registered
    assert reg.admit(gen("CHAR", REF_C, 50), None, history) is None
    again = DedupRegistry(SimpleNamespace(cfg=None, work_root=tmp_path),
                          tmp_path / "dedup_registry.json", mode="enforce")
    assert set(again.entries) == {REF_A, REF_C}
    assert again.entries[REF_A]["commit_block"] == 100
    # shadow mode labels but never enforces; an unfingerprintable ref is admitted
    shadow = _registry(tmp_path, "shadow", {REF_B: dict(same)})
    assert shadow.admit(gen("BRAV", REF_B, 200), None, history) == ("ALFA", "tree_identical", False)
    assert shadow.admit(gen("DELT", "delt/gen@sha256:" + "4" * 64, 300), None, history) is None


def test_registry_earliest_commit_wins_even_when_the_copy_arrives_first(tmp_path):
    same = {"tree_sha256": "T", "token_sha256": "K", "masked_sha256": "M"}
    reg = _registry(tmp_path, "enforce", {REF_A: same, REF_B: dict(same)})
    history = [_commit("ALFA", REF_A, 100), _commit("BRAV", REF_B, 200)]
    gen = lambda hk, ref, rb: SimpleNamespace(hotkey=hk, ref=ref, reveal_block=rb)  # noqa: E731
    # the LATER commit is admitted first (it was funded first) …
    assert reg.admit(gen("BRAV", REF_B, 200), None, history) is None
    # … the earlier commit still keeps its entry when it arrives
    assert reg.admit(gen("ALFA", REF_A, 100), None, history) is None
    assert reg.entries[REF_A]["commit_block"] == 100


# ── per-leg rent-wait deadline ───────────────────────────────────────────────

def test_rent_wait_deadline_honours_the_legs_own_target(tmp_path):
    class _Fake:
        FUNDED_PUBLISH_MARGIN_SECONDS = 900.0
        _leg_local = TrainerRunner._leg_local
        _funded_rent_wait_deadline = TrainerRunner._funded_rent_wait_deadline

        def _leg_wall_seconds(self, sku):
            return 13500.0

    fake = _Fake()
    fake.cfg = SimpleNamespace(round=SimpleNamespace(epoch_blocks=900))
    fake._funded_epoch_end_wall = time.time() + 100.0
    fake._stage_ctx, fake._funded_gate_block = {}, None
    # no per-leg target: the round's epoch end
    assert fake._funded_rent_wait_deadline() == fake._funded_epoch_end_wall - 13500.0 - 900.0
    fake._leg_local.end_wall = fake._funded_epoch_end_wall + 50_000.0
    assert fake._funded_rent_wait_deadline() == fake._funded_epoch_end_wall + 50_000.0 - 14400.0
    fake._leg_local.end_wall = None
    assert fake._funded_rent_wait_deadline() == fake._funded_epoch_end_wall - 14400.0


# ── rolling roster in the audit ──────────────────────────────────────────────

def _receipt(challengers):
    from tests.unit.receipt_fixture import make_scored_receipt

    receipt, _, _ = make_scored_receipt()
    manifest = dict(receipt.manifest)
    king = [e for e in manifest["entries"] if e["role"] == "king"][0]
    manifest["entries"] = [king] + [{**king, "role": "challenger", "miner_hotkey": hk,
                                     "miner_uid": i + 1} for i, hk in enumerate(challengers)]
    from dataclasses import replace

    return replace(receipt, manifest=manifest)


def test_rolling_roster_claims_first_pick_of_fitting_executors():
    honest = {"mode": "rolling",
              "seated": [{"hotkey": "hkB", "reveal_block": 20}, {"hotkey": "hkA", "reveal_block": 10}],
              "rents": [{"hotkey": "hkB", "reveal_block": 20, "passed_over": []},
                        {"hotkey": "hkA", "reveal_block": 10, "passed_over": []}]}
    # settled order is NOT reveal order (legs land when their walls end) — fine
    assert C.check_funded_roster(_receipt(["hkB", "hkA"]), honest).status == C.PASS
    jumped = {"mode": "rolling", "seated": [{"hotkey": "hkB", "reveal_block": 20}],
              "rents": [{"hotkey": "hkB", "reveal_block": 20, "passed_over": ["hkA"]}]}
    r = C.check_funded_roster(_receipt(["hkB"]), jumped)
    assert r.status == C.WARN and "passed over" in r.detail
    stranger = {"mode": "rolling", "seated": [], "rents": []}
    assert C.check_funded_roster(_receipt(["hkZ"]), stranger).status == C.WARN


def test_era_rejections_confirm_the_gate_in_the_audit():
    assert any(reason == "era_" and check == "era" for reason, check in C._REJECTION_CHECK_FOR_REASON)
    assert ("manifest_chain_broken", "era") in C._REJECTION_CHECK_FOR_REASON


def test_state_and_registry_files_live_under_the_work_root(tmp_path):
    from cascade.trainer.rolling import STATE_FILE

    assert Path(STATE_FILE).name == "era_state.json"
