"""Validator loop robustness fixes: crash-safe state, follow-up resume,
read-failure holds, unsigned / backwards manifests, forfeiture ordering,
weight-push success, entry faults and same-round re-judgement."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from cascade.eval.scoring import WindowScore
from cascade.shared.chain import ChainError
from cascade.shared.hippius import MANIFEST_LATEST_KEY, StorageError
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    dump_manifest,
    format_trained_pointer,
)
from cascade.validator import state as state_mod
from cascade.validator.evaluator import EvaluatorError
from cascade.validator.loop import ValidatorRunner, ValidatorStateUnreadable, _load_state
from cascade.validator.state import ChampionState, genesis

CID = "alice/gen@sha256:" + "a" * 64
CID2 = "cascade/ckpt@sha256:" + "b" * 64


def _scores(scale, seed=0, n=300):
    rng = np.random.default_rng(seed)
    return [WindowScore(series_id=str(i), mase=float(rng.uniform(0.5, 1.5) * scale),
                        qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
                        abs_target=float(rng.uniform(5.0, 10.0)))
            for i in range(n)]


KING_SCORES = _scores(1.0)
WIN_SCORES = [WindowScore(s.series_id, s.mase * 0.6, s.qloss_per_q * 0.6, s.abs_target)
              for s in KING_SCORES]
LOSE_SCORES = [WindowScore(s.series_id, s.mase * 1.3, s.qloss_per_q * 1.3, s.abs_target)
               for s in KING_SCORES]


def _manifest(cfg, *, round_id="1", created_block=10, challengers=(("chal_hk", 1, 0),)):
    entries = [TrainedEntry("king_hk", 0, "king", CID, format_trained_pointer(CID2), "d", 10)]
    for hk, uid, rank in challengers:
        entries.append(TrainedEntry(hk, uid, "challenger", CID, format_trained_pointer(CID2),
                                    "d", 10, duel_rank=rank))
    return TrainingManifest(
        round_id=round_id, created_block=created_block,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset, entries=entries)


class _Store:
    def __init__(self, raw):
        self.raw = raw

    def get_text(self, key):
        assert key == MANIFEST_LATEST_KEY
        return self.raw


class _Client:
    def __init__(self, block=100, *, fail_uids=0, set_result=None):
        self.block = block
        self.fail_uids = fail_uids
        self.set_result = set_result
        self.set_calls = []

    def current_block(self):
        return self.block

    def n_uids(self):
        return 16

    def uid_for_hotkey(self, hk):
        if self.fail_uids:
            self.fail_uids -= 1
            raise ChainError("metagraph unavailable")
        return {"king_hk": 0, "chal_hk": 1}.get(hk)

    def set_equal_share_weights(self, uids, n, *, decay, burn_uid):
        self.set_calls.append(list(uids))
        return self.set_result


def _runner(cfg, tmp_path, *, scores=None, state=None, verify=False):
    cfg = replace(cfg, validator=replace(cfg.validator,
                                         state_db_path=str(tmp_path / "state.json")))
    calls = {"eval": 0, "receipts": [], "cascade": 0, "forfeit": 0}
    scores = scores or {"king": KING_SCORES, "challenger": WIN_SCORES}

    def fake_eval(entry, windows):
        calls["eval"] += 1
        s = scores[entry.role] if entry.miner_hotkey in ("king_hk", "chal_hk") else \
            scores[entry.miner_hotkey]
        if isinstance(s, Exception):
            raise s
        return s

    r = ValidatorRunner(cfg=cfg, state=state or genesis("king_hk", 0),
                        evaluate_fn=fake_eval, verify_signatures=verify)
    r._publish_chain_status = lambda *a, **k: None
    r._activation_tick = lambda *a, **k: None
    r._verdict_windows = lambda *a, **k: []
    r._lane_b_prepare = lambda m: None
    r._lane_b_commit = lambda rounds: None
    r._lane_b_receipt_kw = lambda rows: {}
    r._publish_round_receipt = lambda client, m, seed, **kw: calls["receipts"].append(
        (m.round_id, kw.get("reject_reason"), kw.get("outcome")))
    r._cascade_round = lambda m, o: calls.__setitem__("cascade", calls["cascade"] + 1)
    r._forfeit_tick = lambda c: calls.__setitem__("forfeit", calls["forfeit"] + 1)
    return r, calls


# ── V2: crash-safe champion state ────────────────────────────────────────────

def test_persist_is_atomic_and_load_falls_back_to_backup(cfg, tmp_path, caplog):
    r, _ = _runner(cfg, tmp_path, state=genesis("king_hk", 0))
    r._persist_state()
    r.state = replace(r.state, tenure_rounds=7)
    r._persist_state()
    path = tmp_path / "state.json"
    assert state_mod.loads((tmp_path / "state.json.bak").read_text()).tenure_rounds == 0
    path.write_text('{"king_hotkey": "king_h')          # torn main file
    st = _load_state(str(path))
    assert st.king_hotkey == "king_hk"                   # from .bak, not fresh
    assert "resumed from" in caplog.text


def test_load_refuses_unreadable_state(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("garbage")
    (tmp_path / "state.json.bak").write_text("{also garbage")
    with pytest.raises(ValidatorStateUnreadable):
        _load_state(str(path))
    assert _load_state(str(tmp_path / "absent.json")) == ChampionState()


def test_new_state_fields_roundtrip_and_drop_when_default():
    base = genesis("k", 1)
    assert set(json.loads(state_mod.dumps(base))) == set(json.loads(state_mod.dumps(
        state_mod.loads(state_mod.dumps(base)))))
    assert "pre_round" not in state_mod.dumps(base)
    st = replace(base, last_handled_block=5, reject_config_id="abc",
                 pre_round={"round_id": "1", "state": {}, "cascade": None},
                 followup_round_id="1")
    assert state_mod.loads(state_mod.dumps(st)) == st


# ── V3: follow-up resumes; V11: re-judgement replaces ───────────────────────

def test_followup_failure_is_finished_next_poll_without_rejudging(cfg, tmp_path):
    r, calls = _runner(cfg, tmp_path)
    store = _Store(dump_manifest(_manifest(cfg)))
    client = _Client(fail_uids=1)
    with pytest.raises(ChainError):
        r._poll_once(client, store, None)
    assert r.state.followup_round_id == "1" and calls["receipts"] == []
    assert r.state.king_hotkey == "chal_hk" and r.state.rounds_seen == 1
    evals = calls["eval"]
    r._poll_once(client, store, None)                    # finishes the follow-up
    assert calls["eval"] == evals                        # not re-judged
    assert [rid for rid, _, _ in calls["receipts"]] == ["1"] and calls["cascade"] == 1
    assert r.state.followup_round_id is None and r.state.rounds_seen == 1
    r._poll_once(client, store, None)                    # steady state: nothing new
    assert len(calls["receipts"]) == 1


def test_followup_after_restart_rejudges_from_snapshot(cfg, tmp_path):
    r, _ = _runner(cfg, tmp_path, scores={"king": KING_SCORES, "challenger": LOSE_SCORES})
    store = _Store(dump_manifest(_manifest(cfg)))
    with pytest.raises(ChainError):
        r._poll_once(_Client(fail_uids=1), store, None)
    assert r.state.tenure_rounds == 1
    # Restart: the closure is gone; the persisted state carries the marker.
    r2, calls2 = _runner(cfg, tmp_path, state=_load_state(r.cfg.validator.state_db_path),
                         scores={"king": KING_SCORES, "challenger": LOSE_SCORES})
    assert r2.state.followup_round_id == "1"
    r2._poll_once(_Client(), store, None)
    assert calls2["eval"] == 2 and len(calls2["receipts"]) == 1
    # Replaced, not stacked: one round of tenure, one round seen.
    assert r2.state.tenure_rounds == 1 and r2.state.rounds_seen == 1
    assert r2.state.followup_round_id is None


def test_same_round_republish_replaces_not_stacks(cfg, tmp_path):
    r, calls = _runner(cfg, tmp_path, scores={"king": KING_SCORES, "challenger": LOSE_SCORES})
    client = _Client()
    r._poll_once(client, _Store(dump_manifest(_manifest(cfg))), None)
    assert (r.state.tenure_rounds, r.state.rounds_seen) == (1, 1)
    # Same round id, different content (a later created_block).
    r._poll_once(client, _Store(dump_manifest(_manifest(cfg, created_block=11))), None)
    assert calls["eval"] == 4 and len(calls["receipts"]) == 2
    assert (r.state.tenure_rounds, r.state.rounds_seen) == (1, 1)
    assert r.state.last_handled_block == 11


# ── V4 / E1 / E2: read failures hold, never latch ───────────────────────────

def test_pool_pin_read_failure_holds_past_grace(cfg, tmp_path):
    import time

    r, calls = _runner(cfg, tmp_path)

    def unreadable(*a, **k):
        raise StorageError("s3_get_failed: pool/index.json: 403")

    r.check_pool_pin = unreadable
    r._read_holds["1|pool_pin"] = (time.monotonic() - 99999.0, 40)
    r._poll_once(_Client(), _Store(dump_manifest(_manifest(cfg))), None)
    assert r.state.last_handled_round_id is None and calls["receipts"] == []
    assert r._read_holds["1|pool_pin"][1] == 41
    assert calls["forfeit"] == 0                         # not caught up


def test_chain_error_in_gate_holds_and_gate_keys_are_separate(cfg, tmp_path):
    r, calls = _runner(cfg, tmp_path)
    r.check_manifest = lambda m: (_ for _ in ()).throw(ChainError("rpc down"))
    store = _Store(dump_manifest(_manifest(cfg)))
    r._poll_once(_Client(), store, None)
    assert r.state.last_handled_round_id is None and calls["receipts"] == []
    assert "1|era_refs" in r._read_holds
    r._read_holds["1|warm_start"] = (0.0, 1)
    r.check_manifest = lambda m: None
    r.check_pool_pin = lambda *a, **k: (_ for _ in ()).throw(StorageError("503"))
    r._poll_once(_Client(), store, None)
    # A warm-start/era success clears ITS keys; the pool pin holds its own.
    assert set(r._read_holds) == {"1|pool_pin"}


# ── V5: a reject is re-judged after a config change ─────────────────────────

def test_latched_reject_rejudged_after_config_change(cfg, tmp_path):
    r, calls = _runner(cfg, tmp_path)
    m = _manifest(cfg)
    store = _Store(dump_manifest(replace(m, eval_dataset="other/ds")))
    r._poll_once(_Client(), store, None)
    assert calls["receipts"][-1][1] == "eval_dataset_mismatch"
    assert r.state.reject_config_id == r._config_identity()
    r._poll_once(_Client(), store, None)                 # same config: stays latched
    assert len(calls["receipts"]) == 1
    r.cfg = replace(r.cfg, eval=replace(r.cfg.eval, eval_dataset="other/ds"),
                    raw={**r.cfg.raw, "x": 1})
    r._poll_once(_Client(), store, None)                 # re-judged, now scored
    assert len(calls["receipts"]) == 2 and calls["receipts"][-1][2] is not None
    assert r.state.reject_config_id == "" and r.state.king_hotkey == "chal_hk"


# ── V6: unsigned and backwards manifests are never recorded ─────────────────

def test_unsigned_manifest_is_skipped_not_recorded(cfg, tmp_path):
    r, calls = _runner(cfg, tmp_path, verify=True)
    r._poll_once(_Client(), _Store(dump_manifest(_manifest(cfg))), None)
    assert r.state.last_handled_round_id is None and calls["receipts"] == []
    assert len(r._skipped_digests) == 1


def test_older_manifest_never_moves_the_position_back(cfg, tmp_path):
    r, calls = _runner(cfg, tmp_path, scores={"king": KING_SCORES, "challenger": LOSE_SCORES})
    r._poll_once(_Client(), _Store(dump_manifest(_manifest(cfg, round_id="2",
                                                           created_block=500))), None)
    assert r.state.last_handled_round_id == "2"
    r._poll_once(_Client(), _Store(dump_manifest(_manifest(cfg, round_id="1",
                                                           created_block=10))), None)
    assert r.state.last_handled_round_id == "2" and len(calls["receipts"]) == 1


# ── V8: forfeiture only after the catch-up ──────────────────────────────────

def test_forfeit_tick_runs_only_when_caught_up(cfg, tmp_path):
    r, calls = _runner(cfg, tmp_path, scores={"king": KING_SCORES, "challenger": LOSE_SCORES})
    store = _Store(dump_manifest(_manifest(cfg)))
    r._poll_once(_Client(), store, None)
    assert calls["forfeit"] == 1                         # handled ⇒ caught up
    r._poll_once(_Client(), store, None)
    assert calls["forfeit"] == 2                         # unchanged ⇒ caught up


# ── V9: a failed weight push is not recorded ────────────────────────────────

def test_rejected_weight_push_not_stamped_and_retried(cfg, tmp_path):
    r, _ = _runner(cfg, tmp_path)
    r.cfg = replace(r.cfg, validator=replace(r.cfg.validator, weight_set_interval_blocks=180))
    client = _Client(block=1200, set_result=False)
    r._maybe_reassert_weights(client)
    assert r._last_weight_block is None and client.set_calls == [[0]]
    r._maybe_reassert_weights(client)                    # inside the retry back-off
    assert len(client.set_calls) == 1
    from cascade.validator.weights import WEIGHT_RETRY_BLOCKS
    client.block = 1200 + WEIGHT_RETRY_BLOCKS
    r._maybe_reassert_weights(client)                    # retried after the back-off
    assert len(client.set_calls) == 2
    client.block = 1200
    client.set_result = True
    assert r._apply_weights(client, "x", [0]) != ()
    assert r._last_weight_block == 1200


def test_set_weights_reports_rejection():
    bt = pytest.importorskip("bittensor")
    from types import SimpleNamespace

    from cascade.shared.chain import ChainClient

    class _Sub:
        def __init__(self, result=None, exc=None):
            self.result, self.exc = result, exc

        def execute(self, intent, wallet, **kw):
            if self.exc is not None:
                raise self.exc
            return self.result

    def client(sub):
        return ChainClient(netuid=91, wallet_name="w", wallet_hotkey="h",
                           _subtensor=sub, _wallet=object())

    assert client(_Sub(exc=bt.ChainError("rate limited"))).set_weight_vector([0.0, 1.0]) is False
    assert client(_Sub(SimpleNamespace(success=False, message="x"))).set_weight_vector(
        [0.0, 1.0]) is False
    assert client(_Sub(SimpleNamespace(success=True, message=""))).set_weight_vector(
        [0.0, 1.0]) is True
    assert client(_Sub(None)).set_equal_share_weights([1], 4) is True


# ── V10: entry faults ───────────────────────────────────────────────────────

def _fault_cfg(cfg, block):
    return replace(cfg, scoring=replace(cfg.scoring, entry_fault_from_block=block))


# Pre-era block whose epoch boundary is past a gate of 1 (block 10 floors to 0).
FAULT_BLOCK = 72000
NONFINITE = ValueError("forecaster produced non-finite samples on window 's' channel 0")


def test_entry_fault_off_still_raises(cfg, tmp_path):
    r, _ = _runner(cfg, tmp_path, scores={"king": KING_SCORES, "challenger": NONFINITE})
    with pytest.raises(ValueError):
        r.process_round(_manifest(cfg), [], 1)
    r.cfg = _fault_cfg(cfg, 11)                          # gate after this settlement
    with pytest.raises(ValueError):
        r.process_round(_manifest(cfg), [], 1)


def test_faulted_challenger_loses_other_is_judged(cfg, tmp_path):
    scores = {"king": KING_SCORES, "challenger": WIN_SCORES,
              "bad_hk": EvaluatorError("checkpoint_tampered: model.py differs")}
    r, _ = _runner(_fault_cfg(cfg, 1), tmp_path, scores=scores)
    r.state = replace(r.state, streaks={"bad_hk": 1})
    m = _manifest(cfg, created_block=FAULT_BLOCK,
                  challengers=(("bad_hk", 2, 0), ("chal_hk", 1, 1)))
    out = r.process_round(m, [], 1)
    assert out is not None and out.entry_faults.keys() == {"bad_hk"}
    assert "entry_fault bad_hk" in out.transition.note
    assert out.cohort_k == 2 and set(out.cohort_lcbs) == {"chal_hk"}
    assert all(s.hotkey != "bad_hk" for s in out.entry_scores)
    assert r.state.king_hotkey == "chal_hk" and "bad_hk" not in r.state.streaks


def test_all_challengers_faulted_king_holds(cfg, tmp_path):
    r, calls = _runner(_fault_cfg(cfg, 1), tmp_path,
                       scores={"king": KING_SCORES, "challenger": NONFINITE})
    before = r.state
    assert r.process_round(_manifest(cfg, created_block=FAULT_BLOCK), [], 1) is None
    assert r.state == before and r._round_none_reason.startswith("entry_fault")
    r._poll_once(_Client(), _Store(dump_manifest(_manifest(cfg, created_block=FAULT_BLOCK))),
                 None)
    assert calls["receipts"][-1][1].startswith("entry_fault")
    assert r.state.last_handled_round_id == "1"


@pytest.mark.parametrize("err", [StorageError("hub fetch 503"),
                                 EvaluatorError("wrapper_construct_failed: CUDA out of memory"),
                                 RuntimeError("CUDA error")])
def test_transient_challenger_failure_still_retries(cfg, tmp_path, err):
    r, _ = _runner(_fault_cfg(cfg, 1), tmp_path, scores={"king": KING_SCORES, "challenger": err})
    with pytest.raises(type(err)):
        r.process_round(_manifest(cfg, created_block=FAULT_BLOCK), [], 1)


def test_king_fault_is_an_infra_retry(cfg, tmp_path):
    r, _ = _runner(_fault_cfg(cfg, 1), tmp_path, scores={"king": NONFINITE,
                                                        "challenger": WIN_SCORES})
    with pytest.raises(ValueError):
        r.process_round(_manifest(cfg, created_block=FAULT_BLOCK), [], 1)
    assert r.state.king_hotkey == "king_hk"


def test_entry_fault_key_parsed(tmp_path):
    from cascade.shared.config import load_chain_config

    assert load_chain_config().scoring.entry_fault_from_block == 0
    from pathlib import Path

    assert load_chain_config(Path("chain.testnet.toml")).scoring.entry_fault_from_block == 0


def test_maxt_single_survivor_judged_at_cohort_bonferroni(cfg, tmp_path):
    from cascade.eval.koth import evaluate_round

    c = _fault_cfg(cfg, 1)
    c = replace(c, scoring=replace(c.scoring, cohort_maxt_from_block=1))
    scores = {"king": KING_SCORES, "challenger": WIN_SCORES, "bad_hk": NONFINITE}
    r, _ = _runner(c, tmp_path, scores=scores)
    m = _manifest(cfg, created_block=FAULT_BLOCK,
                  challengers=(("bad_hk", 2, 0), ("chal_hk", 1, 1)))
    out = r.process_round(m, [], 1)
    p = r.cfg.koth_params(block=FAULT_BLOCK)
    want = evaluate_round(KING_SCORES, WIN_SCORES,
                          replace(p, bootstrap_alpha=p.bootstrap_alpha / 2), seed=1,
                          king_tenure_rounds=0)
    assert out.result.lcb == pytest.approx(want.lcb)


def test_rejudge_restores_cascade_state(cfg, tmp_path):
    from cascade.validator.cascade import CascadeController, CascadeState

    r, _ = _runner(cfg, tmp_path)
    r.cascade = CascadeController(reign_days=7, state=CascadeState(),
                                  state_path=tmp_path / "cascade.json")

    def cascade_step(m, o):
        if o is not None and o.transition.dethroned:
            r.cascade.note_dethrone(o.transition.new_king_hotkey, block=0)

    r._cascade_round = cascade_step
    r._poll_once(_Client(), _Store(dump_manifest(_manifest(cfg))), None)
    assert r.cascade.state.king_hotkey == "chal_hk" and r.state.king_hotkey == "chal_hk"
    # Re-published round 1 where the challenger now LOSES: both states replaced.
    r.evaluate_fn = lambda e, w: KING_SCORES if e.role == "king" else LOSE_SCORES
    r._poll_once(_Client(), _Store(dump_manifest(_manifest(cfg, created_block=11))), None)
    assert r.state.king_hotkey == "king_hk" and r.state.rounds_seen == 1
    assert r.cascade.state.king_hotkey is None
