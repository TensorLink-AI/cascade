"""Audit finding 5 (DEC-CA-0050): validator lane-B state is a deterministic
FOLD over the chained settlements' bound expert manifests — a validator that
misses a settlement, restarts between persist steps, loses / tears its state
file, or joins late converges to the SAME ``LaneBState`` and weight vector as
one that saw everything. Judging is stubbed (deterministic per entry); what is
under test is the routing of settlements into the fold."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.shared.config import effective_epoch_blocks, parse_lane_b
from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest, expert_manifest_key
from cascade.shared.hippius import StorageError, manifest_round_key
from cascade.shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    dump_manifest,
    format_trained_pointer,
)
from cascade.validator import expert_lane as vel
from cascade.validator.loop import LaneBTransient, ValidatorRunner
from cascade.validator.state import genesis

GEN = "alice/metro-gen@sha256:" + "a" * 64
BAD_GEN = "mallory/metro-gen@sha256:" + "b" * 64
N = 6


def _ptr(tag: str) -> str:
    return format_trained_pointer(
        f"cascade/ckpt-{tag}@sha256:" + hashlib.sha256(tag.encode()).hexdigest())


class _Store:
    def __init__(self) -> None:
        self.objs: dict[str, str] = {}
        self.fail: set[str] = set()

    def get_text(self, key: str) -> str:
        if key in self.fail:
            raise StorageError(f"503 on {key}")
        return self.objs[key]


# per settlement i: (hotkey, uid, domain, gen) legs; mallory's never pass
LEGS = {
    1: [("alice", 7, "energy", GEN)],
    2: [("bob", 8, "web", GEN), ("mallory", 9, "energy", BAD_GEN)],
    3: [],
    4: [("carol", 10, "energy", GEN)],
    5: [("dave", 11, "web", GEN)],
}


def _chain(cfg):
    """N chained settlements (round ids 1000..), every one past from_block;
    settlement 3 binds NO expert manifest."""
    eb = effective_epoch_blocks(cfg.round, 10**6)
    store = _Store()
    ms, prev = [], ""
    for i in range(N):
        rid, block = str(1000 + i), (i + 10) * eb
        legs = LEGS.get(i, [])
        sha = ""
        if legs:
            em = ExpertManifest(
                round_id=rid, created_block=block, king_pointer=_ptr(f"king{i}"),
                lineage_tokens=1, entries=tuple(ExpertEntry(
                    miner_hotkey=hk, miner_uid=uid, gen_ref=g, domain=d,
                    selection={0: [uid]}, exclude={}, base_king_pointer=_ptr(f"king{i}"),
                    rebased_pointer=_ptr(f"x{i}-{hk}"), corpus_digest="c", train_block=block - 1,
                    commit_block=block - 5) for hk, uid, d, g in legs))
            store.objs[expert_manifest_key(rid)] = em.to_json()
            sha = em.digest()
        m = TrainingManifest(
            round_id=rid, created_block=block, contract_digest="1" * 64,
            base_arch_digest="2" * 64, eval_dataset="d",
            entries=[TrainedEntry("K", 0, "king", GEN, _ptr(f"king{i}"), "3" * 64, block - 1)],
            prev_round_id=prev, expert_manifest_sha256=sha)
        store.objs[manifest_round_key(rid)] = dump_manifest(m)
        ms.append(m)
        prev = rid
    return ms, store


@pytest.fixture(autouse=True)
def _stub_judge(monkeypatch):
    def judge_manifest(em, **kw):
        return [vel.ExpertJudgement(
            entry=e, passed=e.gen_ref == GEN, stage="verdict", reason="stub",
            verdict=SimpleNamespace(target_lcb=0.01 * e.miner_uid, guard_lcbs={},
                                    public_lcbs={}),
            king_pointer=kw["king_pointer"]) for e in em.entries
            if e.domain in kw["cfg"].domains]

    monkeypatch.setattr(vel, "judge_manifest", judge_manifest)


def _runner(cfg, path, store, replay=64):
    lb = parse_lane_b({"from_block": 1, "domains": ["energy", "web"],
                       "replay_settlements": replay})
    v = replace(cfg.validator, state_db_path=str(path / "state.json"))
    r = ValidatorRunner(cfg=replace(cfg, lane_b=lb, validator=v), state=genesis("K", 0),
                        verify_signatures=False)
    r._lane_b_store = store
    r._lane_b_window_source = object()
    r._verdict_windows = lambda *a, **k: []
    r._evaluate = lambda *a, **k: []
    return r


def _handle(r, m):
    """The settlement path of the live loop: prepare (reads + judging) before
    lane A, fold before the handled marker."""
    r._lane_b_block = r._epoch_start_block(m)
    return r._lane_b_commit(r._lane_b_prepare(m))


class _Client:
    def __init__(self, block):
        self.block = block

    def current_block(self):
        return self.block

    def n_uids(self):
        return 16

    def uid_for_hotkey(self, hk):
        return {"K": 0, "alice": 7, "bob": 8, "mallory": 9, "carol": 10, "dave": 11}.get(hk)

    def set_weight_vector(self, vec):
        pass

    def set_equal_share_weights(self, *a, **k):
        pass


def _fingerprint(r, block):
    st = r._lane_b()
    r._lane_b_block = block
    vec = r._apply_weights(_Client(block), "x", [0])
    return st.wins, st.history, set(st.applied), st.settled_block, vec


@pytest.fixture
def world(cfg, tmp_path):
    ms, store = _chain(cfg)
    ref = _runner(cfg, tmp_path / "ref", store)
    (tmp_path / "ref").mkdir()
    for m in ms:
        _handle(ref, m)
    last = ref._epoch_start_block(ms[-1])
    return ms, store, _fingerprint(ref, last), last


def test_reference_fold(world):
    ms, _, (wins, history, applied, settled, vec), last = world
    assert {d: w.hotkey for d, w in wins.items()} == {"energy": "carol", "web": "dave"}
    assert {w.hotkey for w in history} == {"alice", "bob", "carol", "dave"}
    assert applied == {m.round_id for m in ms}           # every settlement folded
    assert settled == last and vec[10] > 0 and vec[11] > 0


@pytest.mark.parametrize("recovery", ["catch_up", "on_demand"])
def test_historical_round_uses_its_overlay_and_pay_tiers(cfg, tmp_path, monkeypatch, recovery):
    ms, store = _chain(cfg)
    r = _runner(cfg, tmp_path, store)
    base = r.cfg
    cutoff = r._epoch_start_block(ms[3])
    newer = replace(base, lane_b=replace(base.lane_b, gain_margin=0.2,
                                         full_pay_tiers=((0.01, 1),)))
    r.netconfig = SimpleNamespace(enabled=True,
        config_at=lambda cfg, block: newer if block >= cutoff else base)
    seen = []
    judge = vel.judge_manifest

    def recording_judge(em, **kw):
        seen.append((em.round_id, kw["cfg"].gain_margin))
        return judge(em, **kw)

    monkeypatch.setattr(vel, "judge_manifest", recording_judge)
    with r._settlement_cfg(r._epoch_start_block(ms[-1])):
        if recovery == "catch_up":
            _handle(r, ms[-1])
        else:
            r._lane_b_fold_on_demand(ms[1].round_id)
        assert r.cfg is newer
    assert r.cfg is base
    alice = next(w for w in r._lane_b().history if w.hotkey == "alice")
    assert alice.full_eras == 10 and alice.tier == 0.05
    assert (ms[1].round_id, base.lane_b.gain_margin) in seen
    if recovery == "catch_up":
        assert (ms[4].round_id, 0.2) in seen
        assert r._lane_b().wins["energy"].full_eras == 1


def test_missed_settlement_converges(cfg, tmp_path, world):
    ms, store, want, last = world
    (tmp_path / "b").mkdir()
    r = _runner(cfg, tmp_path / "b", store)
    for i, m in enumerate(ms):
        if i == 2:
            continue                              # downtime / gate-rejected round
        _handle(r, m)
    assert _fingerprint(r, last) == want


def test_restart_between_persist_steps_is_idempotent(cfg, tmp_path, world):
    ms, store, want, last = world
    (tmp_path / "c").mkdir()
    r = _runner(cfg, tmp_path / "c", store)
    for m in ms[:5]:
        _handle(r, m)
    # crash AFTER the lane-B fold persisted, BEFORE the lane-A marker: the
    # restarted validator re-handles settlement 4 — no double record
    r2 = _runner(cfg, tmp_path / "c", store)
    _handle(r2, ms[4])
    _handle(r2, ms[5])
    assert _fingerprint(r2, last) == want


def test_torn_state_file_rebuilds_by_catch_up(cfg, tmp_path, world):
    ms, store, want, last = world
    (tmp_path / "d").mkdir()
    r = _runner(cfg, tmp_path / "d", store)
    for m in ms[:5]:
        _handle(r, m)
    path = r._lane_b_state_path()
    path.write_text(path.read_text()[:37])        # torn mid-write
    r2 = _runner(cfg, tmp_path / "d", store)
    assert r2._lane_b().last_round is None        # discarded, not silently trusted
    _handle(r2, ms[5])                            # the walk replays the chain
    assert _fingerprint(r2, last) == want


def test_new_validator_catches_up_from_the_latest_settlement(cfg, tmp_path, world):
    ms, store, want, last = world
    (tmp_path / "e").mkdir()
    r = _runner(cfg, tmp_path / "e", store)
    _handle(r, ms[-1])
    assert _fingerprint(r, last) == want


def test_replay_window_bounds_the_walk(cfg, tmp_path, world):
    ms, store, _, _ = world
    (tmp_path / "f").mkdir()
    r = _runner(cfg, tmp_path / "f", store, replay=1)
    _handle(r, ms[-1])
    assert set(r._lane_b().applied) == {ms[-1].round_id, ms[-2].round_id}


def test_acceptance_recovery_ignores_payout_ttl_and_folds_oldest_first(cfg, tmp_path, world):
    import json

    from cascade.shared.lane_b import merge_identity

    ms, store, _, _ = world
    (tmp_path / "acceptance-replay").mkdir()
    runner = _runner(cfg, tmp_path / "acceptance-replay", store, replay=1)
    runner.cfg = replace(runner.cfg, lane_b=replace(runner.cfg.lane_b, acceptance_from_block=1))
    original = runner._lane_b_judge_round
    seen = []

    def judge(manifest):
        seen.append(manifest.round_id)
        if int(manifest.round_id) > 1002:
            assert runner._lane_b().accepted  # earlier acceptance already folded
        result = original(manifest)
        if manifest.round_id == "1002":
            win = next(w for w in runner._lane_b().qualifications if w.hotkey == "alice")
            merge = {"rebased_pointer": win.rebased_pointer, "king_pointer": win.king_pointer,
                     "selection": win.selection, "round_id": win.round_id}
            result.acceptances = [(merge, [win], result.block)]
        return result

    runner._lane_b_judge_round = judge
    _handle(runner, ms[-1])
    assert seen == [m.round_id for m in ms]
    assert set(runner._lane_b().applied) == {m.round_id for m in ms}
    win = runner._lane_b().wins["energy"]
    assert win.hotkey == "alice"
    assert runner._lane_b().accepted[json.dumps(merge_identity(win), separators=(",", ":"))]


def test_expert_manifest_read_failure_is_transient_and_folds_nothing(cfg, tmp_path, world):
    ms, store, want, last = world
    (tmp_path / "g").mkdir()
    r = _runner(cfg, tmp_path / "g", store)
    for m in ms[:4]:
        _handle(r, m)
    before = r._lane_b().to_json()
    store.fail.add(expert_manifest_key(ms[4].round_id))
    with pytest.raises(LaneBTransient):
        r._lane_b_prepare(ms[4])
    assert r._lane_b().to_json() == before         # nothing folded: retried next poll
    store.fail.clear()
    _handle(r, ms[4])
    _handle(r, ms[5])
    assert _fingerprint(r, last) == want


def test_sha_mismatch_rejects_lane_b_part_only(cfg, tmp_path, world):
    ms, store, _, _ = world
    (tmp_path / "h").mkdir()
    r = _runner(cfg, tmp_path / "h", store)
    for m in ms[:4]:
        _handle(r, m)
    key = expert_manifest_key(ms[4].round_id)
    store.objs[key] = store.objs[key].replace('"lineage_tokens": 1', '"lineage_tokens": 2')
    rows = _handle(r, ms[4])
    assert rows == [{"error": "expert_manifest_sha_mismatch"}]
    assert r._lane_b().wins["energy"].hotkey == "alice"   # carol's leg not recorded
    assert ms[4].round_id in r._lane_b().applied          # deterministic verdict, folded


def test_no_bound_expert_manifest_means_no_legs(cfg, tmp_path, world):
    ms, store, _, _ = world
    (tmp_path / "i").mkdir()
    r = _runner(cfg, tmp_path / "i", store)
    for m in ms[:3]:
        _handle(r, m)
    # an expert manifest exists in the bucket but settlement 3 names none
    store.objs[expert_manifest_key(ms[3].round_id)] = store.objs[
        expert_manifest_key(ms[4].round_id)]
    assert _handle(r, ms[3]) == []
    assert "carol" not in {w.hotkey for w in r._lane_b().history}


def test_merged_init_unrecorded_merge_is_judged_on_demand(cfg, tmp_path, world):
    """A merged era init naming a merge this validator never folded: the
    merge's settlement is judged on demand (converge, not reject); a READ
    failure while verifying the init raises StorageError (the gate's retry
    path), never a latched era_init_unverifiable rejection."""
    ms, store, _, _ = world
    (tmp_path / "j").mkdir()
    r = _runner(cfg, tmp_path / "j", store)
    merge = {"rebased_pointer": _ptr("x4-carol"), "king_pointer": _ptr("king4"),
             "selection": {"0": [10]}, "round_id": ms[4].round_id}
    era_m = replace(ms[5], warm_start_ckpt=_ptr("merged"), warm_start_base=_ptr("member"),
                    warm_start_merges=(merge,))

    def _unreachable(entry):
        raise OSError("registry 503")

    r._fetch_checkpoint_dir = _unreachable
    with pytest.raises(StorageError, match="era_init_unverifiable"):
        r._verify_merged_init(era_m)
    st = r._lane_b()
    assert st.has_merge(merge["rebased_pointer"], merge["king_pointer"], merge["selection"])
    assert ms[4].round_id in st.applied and st.last_round is None   # out of chain order
    # an unrecorded merge that names no settlement stays a mismatch
    orphan = dict(merge, rebased_pointer=_ptr("nowhere"), round_id="")
    assert "era_init_mismatch" in r._verify_merged_init(
        replace(era_m, warm_start_merges=(orphan,)))
    # the on-demand fold does not break the later in-order fold
    for m in ms:
        _handle(r, m)
    ref = _runner(cfg, tmp_path / "j2", store)
    (tmp_path / "j2").mkdir()
    for m in ms:
        _handle(ref, m)
    last = ref._epoch_start_block(ms[-1])
    assert _fingerprint(r, last) == _fingerprint(ref, last)
