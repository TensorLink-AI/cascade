"""Submission transparency: real file-backed queues, CLI and mocked network boundaries."""

import json
import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from cascade.funding.queue import FundedQueue, LaneBQueue, LaneCQueue
from cascade.miner import cli
from cascade.miner.live_status import status_snapshot
from cascade.miner.submission_status import read_queue, submission_view
from cascade.trainer.progress import progress_scope, report_progress

HK = "miner-public-hotkey"
REF = "miner/gen@sha256:" + "a" * 64
OTHER = "miner/gen@sha256:" + "b" * 64


def flight(tmp_path, lane="a", *, age=0):
    clock = [time.time() - age]
    cls = {"a": FundedQueue, "b": LaneBQueue, "c": LaneCQueue}[lane]
    q = cls(tmp_path / f"{lane}.json", clock=lambda: clock[0])
    if lane == "b":
        q.add_lane_b(HK, ("energy",), REF, 10)
    elif lane == "c":
        q.add_lane_c(HK, "mkt:returns", REF, 10)
    else:
        q.add(HK, REF, 10)
    e = q.get(HK)
    q.mark_in_flight(HK, e.ref, target_boundary=900, era_index=5, started_block=501)
    return q, clock


def phase(q, name, reason=""):
    e = q.get(HK)
    return q.set_phase(HK, name, reason=reason, expect_ref=e.ref,
                       expect_started_block=e.started_block, expect_funded_at=e.funded_at,
                       expect_attempt_id=e.attempt_id)


def snapshot(queue=None):
    stamp = datetime.now(UTC).isoformat()
    return status_snapshot({"chain": {"as_of": stamp, "current_block": 500, "era_blocks": 100},
        "competition": {"as_of": stamp, "warm_start": {
            "mode": "random_init", "pointer": "", "era_start_block": 500},
            **({"lane_a_queue": queue} if queue is not None else {})}},
        now=datetime.now(UTC), max_age=180)


@pytest.mark.parametrize("lane", ["a", "b", "c"])
def test_operator_loss_limit_reports_payer_pause_without_claiming_recovery_capacity(tmp_path, lane):
    from cascade.funding.queue import OPERATOR_FAULT_RETRIES

    q, _ = flight(tmp_path, lane)
    for _ in range(OPERATOR_FAULT_RETRIES):
        q.set_operator_billed(HK, False)
        q.set_operator_billed(HK, True)
    view = submission_view(snapshot(), hotkey=HK, lane=lane, ref=REF, queue=q.public_view())
    assert view["observed"]["operator_faults"] == OPERATOR_FAULT_RETRIES
    assert view["payer_retry_allowed"] is False
    assert "Payer-funded retries are paused after 3" in view["next_action"]
    assert "configured operator capacity" in view["next_action"]
    assert "do not resubmit" in view["next_action"]


@pytest.mark.parametrize("lane", ["a", "b", "c"])
def test_three_hour_gpu_wait_is_visible_and_never_a_training_claim(tmp_path, lane):
    q, clock = flight(tmp_path, lane, age=10800)
    phase(q, "waiting_gpu", "market_capacity")
    clock[0] += 10800
    view = submission_view(snapshot(), hotkey=HK, lane=lane, ref=REF, queue=q.public_view())
    assert view["status"] == "waiting_gpu" and view["observed"]["phase_reason"] == "market_capacity"
    assert 10795 <= view["phase_age_seconds"] <= 10805
    assert view["auto_retry"] is True and view["action_required"] is False
    assert view["eta_seconds"] is None and view["worker_liveness"] == "not_measured"
    assert "particular round is not guaranteed" in view["participation"]


def test_requeue_across_eras_keeps_identity_and_exposes_history_and_backoff(tmp_path):
    q, clock = flight(tmp_path)
    phase(q, "waiting_gpu", "market_capacity")
    clock[0] += 10
    q.requeue(HK, error="private provider diagnostic", error_class="operator",
              burn_attempt=False, operator_billed=True)
    view = submission_view(snapshot(), hotkey=HK, lane="a", ref=REF, queue=q.public_view())
    assert view["auto_retry"] and view["retry_in_seconds"] > 0
    assert view["observed"]["operator_billed"] and view["observed"]["requeues"] == 1
    assert view["observed"]["attempts"] == 0
    assert "private provider diagnostic" not in json.dumps(q.public_view())
    assert view["observed"]["phase_history"][-1]["error_class"] == "operator"
    clock[0] += 121
    assert q.mark_in_flight(HK, REF, target_boundary=1800, era_index=6, started_block=1001)
    phase(q, "training")
    history = q.public_view()["entries"][0]["phase_history"]
    assert {e["target_boundary"] for e in history} >= {900, 1800}
    assert q.get(HK).ref == REF


def test_old_attempt_cannot_relabel_next_flight_or_parked_commitment(tmp_path):
    q, clock = flight(tmp_path)
    old = q.get(HK)
    q.add(HK, OTHER, 20)
    view = submission_view(snapshot(), hotkey=HK, lane="a", ref=OTHER, queue=q.public_view())
    assert view["status"] == "queued_behind" and view["observed"]["phase_history"] == []
    assert view["observed"]["started_block"] is None
    q.mark_done(HK, expect_ref=REF, expect_started_block=501)
    clock[0] += 1
    q.mark_in_flight(HK, OTHER, target_boundary=1800, era_index=6, started_block=1001)
    assert not q.set_phase(HK, "training", expect_ref=old.ref,
                          expect_started_block=old.started_block, expect_funded_at=old.funded_at,
                          expect_attempt_id=old.attempt_id)
    assert q.get(HK).phase == "admitted"
    mismatch = submission_view(snapshot(), hotkey=HK, lane="a", ref=REF, queue=q.public_view())
    assert mismatch["status"] == "not_found" and mismatch["observed"] is None


def test_same_block_retry_still_has_a_distinct_attempt_identity(tmp_path):
    q, _ = flight(tmp_path)
    old = q.get(HK)
    q.requeue(HK, error="capacity", error_class="no_capacity", burn_attempt=False)
    q.mark_in_flight(HK, old.ref, target_boundary=900, era_index=5, started_block=501)
    new = q.get(HK)
    assert new.attempt_id != old.attempt_id
    assert not q.set_phase(HK, "training", expect_ref=old.ref,
                          expect_started_block=old.started_block, expect_funded_at=old.funded_at,
                          expect_attempt_id=old.attempt_id)
    assert q.get(HK).phase == "admitted"


def test_target_identity_does_not_confuse_same_generator_in_two_b_jobs(tmp_path):
    q, _ = flight(tmp_path, "b")
    q.add_lane_b(HK, ("horizon:64",), REF, 20)
    old = submission_view(snapshot(), hotkey=HK, lane="b", ref=REF, targets="energy", queue=q.public_view())
    new = submission_view(snapshot(), hotkey=HK, lane="b", ref=REF, targets="h64", queue=q.public_view())
    assert old["status"] == "admitted"
    assert new["status"] == "queued_behind"
    assert new["observed"]["targets"] == ["horizon:64"]


def test_missing_stale_empty_and_pending_reveal_are_different(tmp_path):
    q = FundedQueue(tmp_path / "q.json")
    q.add_pending(HK, REF)
    doc = q.public_view()
    assert doc["pending_reveal_count"] == 1 and doc["entries"] == []
    view = submission_view(snapshot(), hotkey=HK, lane="a", queue=doc)
    assert view["status"] == "not_found" and "Pending reveals are hidden" in view["message"]
    assert view["action_required"] is None and REF not in json.dumps(doc) and HK not in json.dumps(doc)
    assert submission_view({}, hotkey=HK, lane="a")["status"] == "unknown"
    doc["as_of"] = "2000-01-01T00:00:00+00:00"
    assert submission_view({}, hotkey=HK, lane="a", queue=doc)["status"] == "stale"


def test_legacy_in_flight_does_not_imply_training_and_done_does_not_imply_win(tmp_path):
    q, _ = flight(tmp_path)
    doc = q.public_view()
    doc["entries"][0].pop("phase")
    view = submission_view({}, hotkey=HK, lane="a", queue=doc)
    assert view["status"] == "in_flight" and "does not report" in view["message"]
    q.mark_done(HK)
    view = submission_view({}, hotkey=HK, lane="b", queue=q.public_view())
    assert view["status"] == "done" and view["verdict"] == "not_checked"
    assert "acceptance/rewards are separate" in view["message"]


def test_stale_observation_never_promises_automatic_retry(tmp_path):
    q, _ = flight(tmp_path)
    phase(q, "training")
    doc = q.public_view()
    doc["as_of"] = "2000-01-01T00:00:00+00:00"
    view = submission_view({}, hotkey=HK, lane="a", queue=doc)
    assert view["status"] == "stale" and view["auto_retry"] is None
    assert view["observed"]["phase"] == "training"


def test_progress_is_attempt_scoped_persisted_bounded_and_never_exposes_errors(tmp_path):
    q, _ = flight(tmp_path)
    runner = SimpleNamespace(_funded_queue=lambda: q)
    gen = SimpleNamespace(hotkey=HK, ref=REF)
    with progress_scope(runner, gen):
        for _ in range(20):
            report_progress(runner, "waiting_gpu", "market_capacity")
            report_progress(runner, "training")
    row = FundedQueue(q.path).public_view()["entries"][0]
    assert row["phase"] == "training" and len(row["phase_history"]) == 16
    assert runner._leg_local_obj.funded_progress is None
    with pytest.raises(ValueError):
        phase(q, "training", "arbitrary private provider text")
    with pytest.raises(ValueError):
        phase(q, "done")  # telemetry must not fabricate a terminal lifecycle decision


def test_funded_scope_restores_outer_identity_for_a_operator_handoff(tmp_path):
    from cascade.trainer.faults import _FundedOperatorFallback
    from cascade.trainer.runner_funded_legs import funded_leg_scope

    q, _ = flight(tmp_path)
    runner = SimpleNamespace(_funded_queue=lambda: q)
    gen = SimpleNamespace(hotkey=HK, ref=REF)
    with progress_scope(runner, gen):
        outer = runner._leg_local_obj.funded_progress
        try:
            with funded_leg_scope(runner, q, object(), gen=gen):
                report_progress(runner, "waiting_gpu")
                raise _FundedOperatorFallback(HK)
        except _FundedOperatorFallback:
            assert runner._leg_local_obj.funded_progress == outer
            report_progress(runner, "waiting_operator", "operator_recovery")
            assert q.get(HK).phase == "waiting_operator"


def test_progress_write_failure_does_not_fail_the_job_or_publish_private_text(tmp_path, monkeypatch, caplog):
    q, _ = flight(tmp_path)
    runner = SimpleNamespace(_funded_queue=lambda: q)
    gen = SimpleNamespace(hotkey=HK, ref=REF)
    monkeypatch.setattr(q, "set_phase", lambda *a, **kw: (_ for _ in ()).throw(OSError("private diagnostic")))
    with progress_scope(runner, gen):
        report_progress(runner, "training")
    assert "private diagnostic" not in caplog.text
    assert q.get(HK).status == "in_flight"


def test_status_json_is_one_document_no_wallet_and_ref_scoped(tmp_path, monkeypatch, capsys):
    q, _ = flight(tmp_path)
    phase(q, "waiting_gpu", "market_capacity")
    monkeypatch.setattr("cascade.miner.live_status.fetch_snapshot", lambda *a, **kw: snapshot(q.public_view()))
    monkeypatch.setattr(cli, "load_signing_wallet", lambda *a: pytest.fail("wallet touched"))
    rc = cli.main(["status", "--hotkey", HK, "--ref", REF, "--json"])
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert rc == 0 and doc["data"]["submissions"][0]["status"] == "waiting_gpu"
    assert "No fitting capacity" not in captured.out  # no last failure fabricated before a requeue
    assert "Status: Waiting for GPU" in captured.err


def test_intake_failure_does_not_fall_back_to_healthy_public_queue(tmp_path, monkeypatch, capsys):
    q, _ = flight(tmp_path)
    phase(q, "training")
    monkeypatch.setattr("cascade.miner.live_status.fetch_snapshot", lambda *a, **kw: snapshot(q.public_view()))
    monkeypatch.setattr("cascade.miner.submission_status.read_queue",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("down")))
    assert cli.main(["status", "--hotkey", HK, "--intake", "https://intake.example", "--json"]) == 3
    doc = json.loads(capsys.readouterr().out)
    view = doc["data"]["submissions"][0]
    assert view["status"] == "unknown" and view["auto_retry"] is None
    assert view["observed"] is None and not doc["data"]["requested_submission_fresh"]


def test_watch_refreshes_without_writes_and_json_watch_refuses_before_network(tmp_path, monkeypatch, capsys):
    q, _ = flight(tmp_path)
    calls = []

    def fetch(*a, **kw):
        calls.append(1)
        phase(q, "waiting_gpu" if len(calls) == 1 else "training")
        return snapshot(q.public_view())

    def pause(_):
        if len(calls) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("cascade.miner.live_status.fetch_snapshot", fetch)
    monkeypatch.setattr("time.sleep", pause)
    assert cli.main(["status", "--hotkey", HK, "--watch"]) == 0
    captured = capsys.readouterr().out
    assert len(calls) == 2 and "Waiting for GPU" in captured and "Training worker dispatched" in captured
    assert cli.main(["status", "--hotkey", HK, "--watch", "--json"]) == 2
    assert len(calls) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "bad_watch_options"


@pytest.mark.parametrize("doc", [[], {}, {"entries": {}}, {"entries": ["bad"]}])
def test_malformed_intake_is_not_an_empty_queue(monkeypatch, doc):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, _):
            return json.dumps(doc).encode()

    monkeypatch.setattr("cascade.miner.submission_status.urlopen", lambda *a, **k: Response())
    with pytest.raises(ValueError):
        read_queue("https://intake.example", "a", timeout=1)


def test_real_http_intake_updates_stage_and_disables_caching(tmp_path):
    from urllib.request import urlopen

    from cascade.funding.intake import FundingIntake

    q, _ = flight(tmp_path)
    intake = FundingIntake(q, SimpleNamespace(purge_expired=lambda: None),
                           resolve_reveal=lambda hk, ref: 10, require_signature=False)
    server = intake.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        phase(q, "waiting_gpu", "market_capacity")
        doc = read_queue(base, "a", timeout=2)
        assert doc["entries"][0]["phase"] == "waiting_gpu"
        phase(q, "training")
        with urlopen(base + "/v1/queue", timeout=2) as response:
            assert response.headers["Cache-Control"] == "no-store"
            assert json.loads(response.read())["entries"][0]["phase"] == "training"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def scientific_snapshot(q, lane, stage):
    snap = snapshot()
    e = q.get(HK)
    targets = ["energy"] if lane == "b" else ["mkt:returns"]
    record = {"hotkey": HK, "ref": REF, "targets": targets,
              "attempt_id": e.attempt_id, "trained_pointer": "immutable-trained-artifact",
              "train_block": e.started_block,
              "stage": stage, "round_id": "origin-round"}
    snap["queues"][lane] = {"as_of": datetime.now(UTC).isoformat(),
                             "submission_progress": [record]}
    return snap, record


@pytest.mark.parametrize("stage", ["awaiting_qualification", "qualified_waiting_acceptance",
    "qualification_rejected", "acceptance_rejected", "accepted"])
def test_b_scientific_stage_is_separate_from_training_and_never_proves_payment(tmp_path, stage):
    q, _ = flight(tmp_path, "b")
    q.mark_done(HK, expect_ref=q.get(HK).ref, expect_started_block=501)
    snap, _ = scientific_snapshot(q, "b", stage)
    view = submission_view(snap, hotkey=HK, lane="b", ref=REF, queue=q.public_view())
    assert view["status"] == "done"
    assert view["evaluation"]["stage"] == stage
    assert view["evaluation"]["reward_status"] == "not_verified"
    assert view["auto_retry"] is False and view["action_required"] is False


@pytest.mark.parametrize("outcome", ["pending", "deferred", "confirmed", "below holder LCB", "dropped", ""])
def test_c_joins_forward_receipt_only_to_the_exact_trained_artifact(tmp_path, outcome):
    q, _ = flight(tmp_path, "c")
    snap, record = scientific_snapshot(q, "c", "awaiting_forward_judgement")
    snap["submission_receipt"] = {"status": "scored", "round_id": "confirm-round", "boundary": 900,
        "lane_c": [{"hotkey": HK, "target": "mkt:returns", "passed": False,
                    "train_block": record["train_block"],
                    "trained_pointer": record["trained_pointer"], "outcome": outcome}]}
    view = submission_view(snap, hotkey=HK, lane="c", ref=REF, queue=q.public_view())
    assert view["evaluation"]["stage"] == (outcome or "not_passed")
    assert view["evaluation"]["reported"]["receipt_boundary"] == 900
    assert view["evaluation"]["reward_status"] == "not_verified"
    snap["submission_receipt"]["lane_c"][0]["trained_pointer"] = "different-artifact"
    view = submission_view(snap, hotkey=HK, lane="c", ref=REF, queue=q.public_view())
    assert view["evaluation"]["stage"] == "awaiting_forward_judgement"


@pytest.mark.parametrize("lane", ["b", "c"])
def test_fresh_intake_does_not_make_old_scientific_data_current_or_match_a_retry(tmp_path, lane):
    q, _ = flight(tmp_path, lane)
    snap, record = scientific_snapshot(q, lane, "accepted" if lane == "b" else "pending")
    record["attempt_id"] = "earlier-attempt-of-same-hotkey-ref-targets"
    assert submission_view(snap, hotkey=HK, lane=lane, queue=q.public_view())["evaluation"]["stage"] == "unknown"
    record["attempt_id"] = q.get(HK).attempt_id
    snap["queues"][lane]["as_of"] = "2000-01-01T00:00:00+00:00"
    view = submission_view(snap, hotkey=HK, lane=lane, queue=q.public_view())
    assert view["freshness"]["fresh"] and view["evaluation"]["stage"] == "unknown"
    assert not view["evaluation"]["freshness"]["fresh"]


def test_b_acceptance_of_old_round_cannot_bind_a_retry_with_identical_weights():
    from cascade.shared.submission_progress import lane_b_progress

    common = {"hotkey": HK, "gen_ref": REF, "domain": "energy", "targets": ["energy"],
              "commit_block": 5, "leg_pointer": "same-weights", "outcome": "qualified"}
    state = SimpleNamespace(submission_results=[
        {**common, "round_id": "old-round", "attempt_id": "old"},
        {**common, "round_id": "new-round", "attempt_id": "retry"}],
        outcomes=[{**common, "round_id": "old-round", "outcome": "accepted"}],
        pending_merges=[], finished=[])
    rows = lane_b_progress(state)
    assert rows[-1]["stage"] == "accepted" and rows[-1]["attempt_id"] == "old"
    assert any(r["stage"] == "qualified" and r["attempt_id"] == "retry" for r in rows)


def test_optional_attempt_identity_read_failure_cannot_fail_training(caplog):
    from cascade.trainer.progress import funded_attempt_id

    def broken_read(_):
        raise OSError("private credential diagnostic")
    assert funded_attempt_id(SimpleNamespace(get=broken_read), HK, REF, 10) is None
    assert "private credential diagnostic" not in caplog.text


def test_c_identical_weights_do_not_mix_two_miners_submission_identities():
    from cascade.shared.submission_progress import lane_c_progress

    entries = [{"miner_hotkey": hk, "recipe_ref": ref, "target": "mkt:returns",
                "trained_pointer": "identical-weights", "train_block": 500}
               for hk, ref in ((HK, REF), ("another-miner", OTHER))]
    identities = [{"hotkey": e["miner_hotkey"], "ref": e["recipe_ref"],
                   "targets": ["mkt:returns"], "attempt_id": e["miner_hotkey"],
                   "trained_pointer": e["trained_pointer"]} for e in entries]
    state = SimpleNamespace(settled={"round": {"text": json.dumps({"entries": entries}),
                                              "submission_identities": identities}},
                            finished=[], submission_verdicts=[])
    rows = lane_c_progress(state)
    assert [(r["hotkey"], r["ref"], r["attempt_id"]) for r in rows] == [
        (HK, REF, HK), ("another-miner", OTHER, "another-miner")]
