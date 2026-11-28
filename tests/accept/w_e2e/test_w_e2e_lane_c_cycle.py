"""W-E2E lane C (MarketCast): one full cycle on fake infrastructure, through
the real intake, trainer, validator and miner CLI code (tests/e2e_lanes/lane_c.py).

submit → intake (HTTP; GPU fit + sample geometry) → queue → dispatch → train
(the real ``train_mkt_leg`` + fault classifier) → harvest (Hub) → settle_c
(signed lane C manifest bound into lane A) → judge (the main validator, the
judge on a forked MarketCast pod, the pinned base resolved on the box) →
report → record (the trainer's next manifest) → verify (a peer validator
reaches the same verdict from the recorded bytes; an independent validator
recomputes the report byte for byte) → miner_status (``cascade lane-c
status`` over the intake, and the forward-judging progress).

The regression fixtures each replay one 2026-11-25..28 bug. Main has every
fix, so each pair is: the cycle with today's code (the fixed behaviour), and
the same cycle under a REVERT PATCH (the pre-fix behaviour) which the harness
must catch by assertion.
"""

from __future__ import annotations

import pytest

from tests.e2e_lanes import lane_c as C
from tests.e2e_lanes.harness import BoxBudgetExceeded


@pytest.fixture
def make_world(tmp_path, monkeypatch):
    worlds = []

    def make(**kw):
        w = C.LaneCWorld(tmp_path / f"w{len(worlds)}", monkeypatch, **kw)
        worlds.append(w)
        return w

    yield make
    for w in worlds:
        w.close()


def _assert_full_and_reproducible(w, out):
    from cascade.shared.marketcast_report import load_mkt_report, verify_mkt_report

    w.cycle.assert_full(C.LANE_C_STAGES)
    assert out["refused"] == [], "a GPU / model library loaded in a box stage"
    rep = load_mkt_report(out["report_text"])
    assert verify_mkt_report(rep, C.MAIN_VALIDATOR), "report not signed by the main validator"
    e = out["report_entry"]
    assert e.windows and not e.fault and e.status != "infra_deferred", e
    main_rows, peer_rows = out["judgements"]
    (row,) = [r for r in main_rows if r["hotkey"] == C.HK]
    assert (row["stage"], row["passed"], row["outcome"]) == ("verdict", True, "pending"), row
    assert peer_rows == main_rows, "a peer validator judged the recorded report differently"
    assert out["recomputed"] == out["report_text"], "the report is not byte-reproducible"
    # every artefact on the Hub is single-blob (the pinned hippius_hub reads it)
    assert w.hub.pushes and not any(chunked for _r, _d, chunked in w.hub.pushes)
    assert "Verdict: published in round 9100" in out["miner"]
    assert "Forward evaluation passed initially" in out["evaluation"]


def test_lane_c_full_cycle_publishes_a_reproducible_verdict_the_miner_sees(make_world, capsys):
    w = make_world()
    out = C.run_cycle(w, capsys)
    _assert_full_and_reproducible(w, out)
    assert w.pod.rents == [C.GPU] and w.pod.prep == 1
    assert w.base_resolves >= 1                  # the pinned base resolved on the box


# ── #1003: box-side timestral import resolving the pinned base ───────────────


def _pre_1003_resolve_pinned_base(base_ref, base_sha256, *, cache_dir=None):
    """resolve_pinned_base before #1003: through timestral.loading (torch)."""
    from pathlib import Path

    from cascade.eval import timestral_adapter as TA
    from cascade.shared.config.marketcast import split_base_ref

    repo, rev = split_base_ref(base_ref)
    TA.check_base_sha256(base_sha256, base_sha256)
    TA.load_timestral()
    from timestral.loading import resolve_checkpoint  # type: ignore[import-not-found]

    d = Path(resolve_checkpoint(repo, revision=rev, cache_dir=cache_dir))
    TA.verify_checkpoint(d / TA.WEIGHTS_FILE, base_sha256)
    return d


def test_regression_1003_box_side_timestral_import_is_caught(make_world, capsys, monkeypatch):
    from cascade.eval import timestral_adapter as TA

    w = make_world()
    monkeypatch.setattr(TA, "resolve_pinned_base", _pre_1003_resolve_pinned_base)
    out = C.run_cycle(w, capsys)
    # the box guard refused the call-time import ...
    assert any(h.startswith("timestral") for h in out["refused"]), out["refused"]
    # ... and the cycle stops: main still SWALLOWS it into an infra deferral
    # (retro C4) — the report is published, every entry infra_deferred
    e = out["report_entry"]
    assert e is not None and e.status == "infra_deferred" and not e.windows
    assert "pinned base failed to load: TimestralUnavailable" in e.infra_reason, e.infra_reason
    with pytest.raises(AssertionError, match="report"):
        _assert_full_and_reproducible(w, out)


# ── #1005: the MarketCast judge's windows / covariate rows built in-process ──


def test_pod_judge_keeps_a_wide_round_inside_the_box_budget(make_world, capsys):
    w = make_world(context=C.WIDE_CONTEXT, minutes=C.WIDE_MINUTES)
    out = C.run_cycle(w, capsys)
    _assert_full_and_reproducible(w, out)
    assert max(p for _n, p in w.peaks) < C.BOX_JUDGE_BUDGET, w.peaks


def test_regression_1005_in_process_judge_over_the_box_budget_is_caught(make_world, capsys):
    w = make_world(context=C.WIDE_CONTEXT, minutes=C.WIDE_MINUTES)
    v = w.validator("main", main=True, on_pod=False)     # pre-#1005: no pod judge ...
    v.mkt_box_judge_cap_bytes = 1 << 40                  # ... and no box cap
    with pytest.raises(BoxBudgetExceeded, match="validator main lane C"):
        C.run_cycle(w, capsys)
    assert "report" not in w.cycle.names()
    assert max(p for _n, p in w.peaks) > C.BOX_JUDGE_BUDGET, w.peaks


# ── #991: the flat-context NaN (our adapter) blamed on the miner ─────────────


def _assert_nan_is_ours(w, out):
    outcomes = [o["detail"]["class"] for o in w.driver.st.outcomes
                if o["outcome"] in ("requeued", "miner_fault")]
    assert outcomes == ["operator"], f"the NaN was not classed operator: {outcomes}"
    assert w.pod.rents == [C.GPU, C.GPU]                 # re-run uncharged, then judged
    _assert_full_and_reproducible(w, out)


def test_nan_loss_is_an_operator_fault_and_the_leg_reruns(make_world, capsys):
    w = make_world(pod_script=("nan", "ok"))
    out = C.run_cycle(w, capsys)
    _assert_nan_is_ours(w, out)
    assert w.queue.get(C.HK).attempts == 0               # the miner was not charged a try


def test_regression_991_nan_blamed_on_the_miner_is_caught(make_world, capsys, monkeypatch):
    from cascade.trainer import faults

    w = make_world(pod_script=("nan", "ok"))
    monkeypatch.setattr(faults, "nonfinite_training", lambda text: False)   # pre-#991
    out = C.run_cycle(w, capsys)
    assert w.queue.get(C.HK).status == "failed"          # the entry ended on the miner
    with pytest.raises(AssertionError, match="operator"):
        _assert_nan_is_ours(w, out)


# ── #989: the recipe's own CUDA OOM retried as infra (re-rented) ─────────────


def _assert_oom_ends_without_rerent(w, out):
    e = w.queue.get(C.HK)
    assert w.pod.rents == [C.GPU], f"the OOM was re-rented: {w.pod.rents}"
    assert e.status == "failed" and e.last_error_class == "gpu_oom", e
    assert "out of GPU memory on 1xA100" in out["miner"], out["miner"]
    assert "train" not in w.cycle.names()


def test_recipe_oom_ends_the_entry_with_a_reason_and_no_rerent(make_world, capsys):
    w = make_world(pod_script=("oom", "ok"))
    out = C.run_cycle(w, capsys)
    _assert_oom_ends_without_rerent(w, out)


def test_regression_989_oom_classified_as_infra_is_caught(make_world, capsys, monkeypatch):
    from cascade.trainer import faults

    w = make_world(pod_script=("oom", "ok"))
    monkeypatch.setattr(faults, "own_run_cuda_oom", lambda line: False)     # pre-#989
    out = C.run_cycle(w, capsys)
    with pytest.raises(AssertionError, match="re-rented"):
        _assert_oom_ends_without_rerent(w, out)


# ── #995: a re-attach with no ledgered pod: generate, then refuse the rent ───


def _assert_reattach_runs_once(w, out):
    refused = [o for o in w.driver.st.outcomes if "re-attach refused" in str(o.get("detail"))]
    assert w.pod.prep == 1 and not refused, \
        f"pre-rent work done {w.pod.prep}x; refused re-attaches: {refused}"
    _assert_full_and_reproducible(w, out)


def test_restart_mid_leg_with_no_pod_runs_the_leg_normally(make_world, capsys):
    w = make_world()
    out = C.run_cycle(w, capsys, restart_mid_leg=True)
    _assert_reattach_runs_once(w, out)


def test_regression_995_reattach_without_a_pod_is_caught(make_world, capsys, monkeypatch):
    from cascade.trainer.lane_c_driver import LaneCDriver

    w = make_world()
    monkeypatch.setattr(LaneCDriver, "_leg_pod_ledgered", lambda self, leg: True)  # pre-#995
    out = C.run_cycle(w, capsys, restart_mid_leg=True)
    with pytest.raises(AssertionError, match="pre-rent work done 2x"):
        _assert_reattach_runs_once(w, out)


# ── #998: a chunked-v2 checkpoint the pinned hippius_hub cannot read ─────────


def test_chunked_v2_checkpoint_on_the_hub_is_reassembled_and_judged(make_world, capsys,
                                                                    monkeypatch):
    from cascade.shared import hippius

    w = make_world()
    monkeypatch.setattr(hippius, "force_single_blob_layout", lambda: None)  # a 0.7.2 pusher
    out = C.run_cycle(w, capsys)
    assert any(chunked for _r, _d, chunked in w.hub.pushes)
    w.cycle.assert_full(C.LANE_C_STAGES)
    assert out["recomputed"] == out["report_text"]


def test_regression_998_chunked_v2_unreadable_is_caught(make_world, capsys, monkeypatch):
    from cascade.shared import hippius

    w = make_world()
    monkeypatch.setattr(hippius, "force_single_blob_layout", lambda: None)  # pre-#998 push
    monkeypatch.setattr(hippius, "chunked_index_files", lambda d: [])      # pre-#998 fetch
    out = C.run_cycle(w, capsys)
    e = out["report_entry"]
    # the fetched "weights" are the chunk index: the MINER's fine-tune is faulted
    assert e is not None and e.fault.startswith("fine-tune unloadable"), e
    with pytest.raises(AssertionError):
        _assert_full_and_reproducible(w, out)
