"""W-E2E lane B: one full cycle on fake infrastructure, and the week's lane B
failures as regression fixtures (retrospective 2026-11-28, P1a).

The cycle (tests/e2e_lanes/lane_b.py): submit → intake (HTTP) → funded queue
→ dispatch (gate, probe on the eval pod, selection) → generation (Shadeform
stand-in) → train (Lium stand-in) → harvest + Hub upload → settlement
(bench, prepare-ahead, rebase / merge check / judge on the eval pod, publish)
→ validator verification (its own fetch + scoring, byte-identical rows, the
LaneBState fold) → ``cascade lane-b status``.

Each regression test runs the cycle with the incident's condition twice: as
main ships it (the cycle completes) and under a revert patch of that fix
(the cycle fails by assertion, at the stage the incident failed). Two
bugs were still on main when this harness first ran (fixed in this PR):

* the trainer box imported torch at every lane B dispatch (ESFT selection,
  ``expert_lane.select_routed_experts``);
* a settled lane B entry never stored its verdict (``_settle_funding`` looked
  it up by the REBASED pointer); only the driver-state fallback showed it.

Torch-free (the box guard refuses torch on every box stage); no pod, chain,
key or network. The torch equivalence of the selection runs in CI only."""

from __future__ import annotations

import json
import os

import pytest

from tests.e2e_lanes import harness as H
from tests.e2e_lanes import lane_b as LB


@pytest.fixture
def world(cfg, tmp_path, monkeypatch):
    w = LB.LaneBWorld(cfg, tmp_path, monkeypatch)
    yield w
    w.close()


def _world(cfg, tmp_path, monkeypatch, name):
    d = tmp_path / name
    d.mkdir()
    return LB.LaneBWorld(cfg, d, monkeypatch)


def _cycle(cfg, tmp_path, monkeypatch, name, *, setup=None, **kw):
    """One cycle in its own world; returns it (closed)."""
    w = _world(cfg, tmp_path, monkeypatch, name)
    try:
        if setup is not None:
            setup(w)
        LB.full_cycle(w, **kw)
    finally:
        w.close()
    return w


def _verdict_published(w) -> bool:
    return "published_verdict" in w.cycle.names()


# ── 1. the happy cycle ───────────────────────────────────────────────────────


def test_lane_b_full_cycle_publishes_a_verdict_the_validator_reproduces(world):
    w = world
    LB.full_cycle(w)
    w.cycle.assert_full(LB.STAGES)
    # the verdict: the leg qualified on its declared target
    (res,) = w.drv.st.submission_results
    assert res["hotkey"] == LB.HK and res["outcome"] == "qualified", res
    assert {m["target"]: m["passed"] for m in res["margins"]}[LB.TARGET] is True
    # published (public-read, W-EVALPUB) and bound by the settlement digest
    from cascade.shared.expert_manifest import expert_manifest_key, load_expert_manifest

    key = expert_manifest_key(w.round_id)
    em = load_expert_manifest(w.store.objs[key])
    assert w.store.acl[key] == "public-read"
    assert em.digest() == w.drv.expert_manifest_digest(w.round_id)
    # the validator, from the stored bytes and its own fetches, judges the
    # SAME rows byte for byte and folds the win
    assert LB.validator_rows(w) == LB.trainer_rows(w)
    assert w.validator._lane_b().wins[LB.TARGET].hotkey == LB.HK
    # the box never fetched a checkpoint for model work: the trainer's eval
    # pod and the validator's eval host did (pod ops ran there)
    assert {"probe", "eval"} <= set(w.eval_pod.ops), w.eval_pod.ops
    assert "merge_check" in w.val_pod.ops, w.val_pod.ops
    # one generation, one rent, one harvest for one funded entry
    assert (w.generations, len(w.lium.rents)) == (1, 1)


def test_lane_b_settled_entry_stores_its_verdict_and_the_cli_prints_it(world):
    w = world
    LB.full_cycle(w)
    w.cycle.assert_full(LB.STAGES)
    stored = w.stored_entry()
    # #996's primary path: mark_done(verdict=) on the entry itself (the
    # driver-state fallback is for entries settled before the field existed)
    assert (stored.get("verdict") or {}).get("outcome") == "qualified", stored
    text = w.miner_status()
    assert f"Verdict: qualified in round {w.round_id}" in text, text
    assert f"{LB.TARGET} (declared): LCB" in text and "PASS" in text, text


# ── 2. regressions: each fails the cycle under its revert patch ──────────────


def test_regression_pod_reply_with_download_chatter_983(cfg, tmp_path, monkeypatch, caplog):
    """#983 (2026-11-27 19:27): hippius_hub printed 'Downloading …' to stdout
    ahead of the pod's one-line result.json; the reply was 'not JSON'."""
    from cascade.trainer import lane_b_offload as off

    def chatter(w):
        w.pod_chatter = True

    ok = _cycle(cfg, tmp_path, monkeypatch, "fixed", setup=chatter)
    ok.cycle.assert_full(LB.STAGES)
    with monkeypatch.context() as m:
        m.setattr(off, "pod_reply_json", json.loads)          # the pre-#983 parse
        bad = _cycle(cfg, tmp_path, monkeypatch, "reverted", setup=chatter)
    # the first pod reply of the cycle (the dispatch's routing probe) already
    # fails: the leg never dispatches, the entry waits, no verdict
    assert not _verdict_published(bad), bad.cycle.names()
    assert "dispatch" not in bad.cycle.names() and bad.queue_row()["status"] == "queued"
    assert "pod reply is not JSON" in caplog.text


def test_regression_chunked_v2_checkpoint_on_the_validator_998(cfg, tmp_path, monkeypatch):
    """#998 (2026-11-28 04:27): the rebased candidate was pushed from an eval
    pod whose hippius_hub (off the lock) wrote chunked-v2; the pinned 0.5.1
    saved the pointer AS weights.safetensors, every validator failed
    checkpoint_tampered and re-judged it each settlement."""
    import cascade.shared.hippius as hip
    from cascade.validator.lane_b import LaneBTransient

    w = _world(cfg, tmp_path, monkeypatch, "chunked")
    try:
        w.hub.chunked_repos = ("-expert-",)                # the drifted pod's pushes
        w.submit()
        w.start_intake()
        w.fund()
        w.start_trainer()
        w.dispatch()
        assert w.settle() is not None
        assert any(p["chunked"] for p in w.hub.pushes), w.hub.pushes
        with monkeypatch.context() as m:                    # the pre-#998 fetch
            m.setattr(hip, "reassemble_chunked_v2", lambda ref, snap: [])
            with pytest.raises(LaneBTransient, match="checkpoint_tampered"):
                w.validate()
        assert "validator_verify" not in w.cycle.names()
        # as main ships it: the cached pointer is healed (refetched,
        # reassembled, verified) and the verdict reproduces byte for byte
        w.validate()
        w.miner_status()
    finally:
        w.close()
    w.cycle.assert_full(LB.STAGES)
    assert LB.validator_rows(w) == LB.trainer_rows(w)


def test_regression_box_side_model_work_952(cfg, tmp_path, monkeypatch):
    """#952-#955 / #965 (2026-11-26): lane B probe, rebase, judge and merge
    ran on the trainer box (checkpoint download + torch). Reverted: the box
    guard refuses the first torch import, in the dispatch's routing probe."""
    from cascade.trainer import lane_b_offload as off

    _cycle(cfg, tmp_path, monkeypatch, "fixed").cycle.assert_full(LB.STAGES)
    with monkeypatch.context() as m:
        m.setattr(off, "local_eval_allowed", lambda: True)   # model work on the box
        with pytest.raises(H.BoxImportRefused, match="torch"):
            _cycle(cfg, tmp_path, monkeypatch, "reverted")


def test_regression_box_side_torch_in_the_dispatch_selection(cfg, tmp_path, monkeypatch):
    """Found by this harness (still on main before this PR): every lane B
    dispatch tick imported torch + moe_model on the trainer box for the ESFT
    selection. The numpy port is exact; the pre-fix call is refused."""
    from cascade.trainer import expert_lane as tel

    _cycle(cfg, tmp_path, monkeypatch, "fixed").cycle.assert_full(LB.STAGES)

    def pre_fix(affinity, *, p, min_n, max_n, exclude=None):
        import torch  # noqa: F401 — what the pre-fix body did first

        raise AssertionError("unreachable under the box guard")

    with monkeypatch.context() as m:
        m.setattr(tel, "select_routed_experts", pre_fix)
        with pytest.raises(H.BoxImportRefused, match="torch"):
            _cycle(cfg, tmp_path, monkeypatch, "reverted")


def test_regression_reattach_with_no_pod_995(cfg, tmp_path, monkeypatch):
    """#995 (2026-11-28, 5H63pHtyHgkf): the trainer restarted between the
    dispatch and the rent; the re-attach ran attach-only, generated the whole
    CPU corpus (~1 h Shadeform) and only then the rent refused it."""
    import cascade.trainer.runner_funded_legs as rfl
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    ok = _cycle(cfg, tmp_path, monkeypatch, "fixed", dispatch_hold=True)
    ok.cycle.assert_full(LB.STAGES)
    assert (ok.generations, len(ok.lium.rents)) == (1, 1)
    with monkeypatch.context() as m:                         # neither #995 half
        m.setattr(RunnerLaneBOps, "leg_pod_ledgered", lambda self, leg: True)
        m.setattr(rfl, "_leg_allocated", lambda runner, rid, hk: True)
        bad = _cycle(cfg, tmp_path, monkeypatch, "reverted", dispatch_hold=True)
    assert bad.generations == 1 and bad.lium.rents == [], (bad.generations, bad.lium.rents)
    assert not _verdict_published(bad)
    e = bad.queue_row()
    assert e["status"] == "queued" and e["last_error_class"] == "no_capacity", e


def _failed_leg(cfg, tmp_path, monkeypatch, name, fail):
    w = _world(cfg, tmp_path, monkeypatch, name)
    try:
        w.submit()
        w.start_intake()
        w.fund()
        w.start_trainer()
        w.lium.fail = fail
        w.dispatch()
        row = w.queue_row()
    finally:
        w.close()
    return w, row


def test_regression_recipe_gpu_oom_ends_the_entry_989(cfg, tmp_path, monkeypatch):
    """#989 (2026-11-27): a leg's own CUDA OOM was classed infra and re-rented
    on the miner's key (same GPU, same OOM). Now: the miner's sizing, the
    entry ends with a miner-visible reason, no re-rent."""
    import cascade.trainer.faults as faults

    w, row = _failed_leg(cfg, tmp_path, monkeypatch, "fixed", "oom")
    assert (row["status"], row["last_error_class"]) == ("failed", "gpu_oom"), row
    assert row["reason"].startswith("out of GPU memory") and "not re-rented" in row["reason"]
    assert not w.drv.st.finished and len(w.lium.rents) == 1
    with monkeypatch.context() as m:
        m.setattr(faults, "own_run_cuda_oom", lambda line: False)   # the pre-#989 class
        _, bad = _failed_leg(cfg, tmp_path, monkeypatch, "reverted", "oom")
    assert (bad["status"], bad["last_error_class"]) != ("failed", "gpu_oom"), bad
    assert bad["status"] == "queued" and bad["last_error_class"] == "infra", bad


def test_lane_b_nonfinite_training_is_never_the_miners_fault(cfg, tmp_path, monkeypatch):
    """A lane B worker's non-finite gradient (our fixed architecture) never
    ends the miner's entry as their fault; it is retried."""
    w, row = _failed_leg(cfg, tmp_path, monkeypatch, "nan", "nan")
    assert row["status"] == "queued", row
    assert row["last_error_class"] not in ("generator", "gpu_oom", "recipe"), row
    assert not [o for o in w.drv.st.outcomes if o.get("outcome") == "miner_fault"]


# ── 3. the selection's torch equivalence (CI only: torch never runs here) ────


@pytest.mark.skipif(os.environ.get("CI") != "true", reason="torch runs in CI only")
def test_numpy_selection_equals_the_digest_pinned_rule():
    torch = pytest.importorskip("torch")
    import numpy as np

    from cascade.trainer.expert_lane import select_routed_experts
    from cascade.trainer.moe_model import select_domain_experts

    rng = np.random.default_rng(5)
    for trial in range(200):
        aff = rng.integers(0, 50, size=(3, 32)).astype(np.int64)
        aff[rng.random(aff.shape) < 0.3] = 0
        excl = {0: {int(x) for x in rng.integers(0, 32, size=3)}} if trial % 2 else None
        banned = {int(k): set(v) for k, v in (excl or {}).items()}
        for li in range(aff.shape[0]):
            banned.setdefault(li, set()).update(int(e) for e in np.nonzero(aff[li] <= 0)[0])
        want = select_domain_experts(torch.as_tensor(aff), p=0.2, min_n=4, max_n=12,
                                     exclude=banned)
        assert select_routed_experts(aff, p=0.2, min_n=4, max_n=12, exclude=excl) == want
