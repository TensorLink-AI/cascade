"""Trainer benches BOTH duel checkpoints AFTER the manifest publishes.

When [scoring] cascade_enabled, the manifest goes out untouched (validators
start scoring immediately) and the trainer then scores the king's AND the
challenger's checkpoints on GIFT-Eval / BOOM / TIME, publishing the numbers as
a separate signed bench report (cascade.shared.bench_report). The provisioner's
teardown hold is driven by the bench_pending / bench_complete markers written
around that sweep.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

import cascade.trainer.loop as loop_mod
from cascade.shared.bench_report import load_bench_report
from cascade.shared.chain import Commitment
from cascade.shared.manifest import BenchScores
from cascade.trainer.contract import TrainResult
from cascade.trainer.loop import TrainerRunner

REF_A = "alice/gen-a@sha256:" + "a" * 64
REF_B = "bob/gen-b@sha256:" + "b" * 64


class _FakeStream:
    n_series, total_points = 3, 192

    def __init__(self, digest="d"):
        # per-generator digest (from the fetched gen dir): a constant would make
        # the king and challenger byte-identical and (rightly) trip the trainer's
        # content-clone drop, which is not what these tests probe
        self.digest = digest

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def series(self):
        for _ in range(3):
            yield np.ones((1, 64))


class _FakeBaseTrainer:
    def train(self, stream, contract, *, training_seed, token_budget, out_dir, logger=None):
        for _ in stream:
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "weights.safetensors").write_bytes(b"x")
        return TrainResult(local_dir=out_dir, param_count=4_000_000, train_seconds=1.0, metrics={})


def _fake_upload(local_dir, repo, hub=None, **kw):
    # Distinct pointer per checkpoint repo (as in production — the repo name
    # carries round/role/size): the bench report joins on trained_pointer, so
    # a constant ref would collapse the king's and challenger's rows.
    import hashlib

    from cascade.shared.hippius import HubRef, HubUpload

    dg = hashlib.sha256(str(repo).encode("utf-8")).hexdigest()
    return HubUpload(ref=HubRef.parse(f"cascade/ckpt-{dg[:12]}@sha256:{dg}"), size_bytes=1)


def _commit(uid, hotkey, ref, block):
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


_BENCH = BenchScores(
    gifteval_crps=0.42, gifteval_mase=0.81, boom_crps=0.55,
    boom_mase=0.90, time_crps=0.38, time_mase=0.77,
)


class _FakeStore:
    def __init__(self):
        self.texts: dict[str, str] = {}
        self.reads: list[str] = []

    def put_text(self, key, text, *, content_type="text/plain", acl=None):
        self.texts[key] = text

    def get_text(self, key):
        self.reads.append(key)
        if key not in self.texts:
            raise KeyError(key)
        return self.texts[key]


class _FakeWallet:
    class _HK:
        def sign(self, body: bytes) -> bytes:
            return b"SIG:" + body[:8]

    hotkey = _HK()


@pytest.fixture()
def cascade_cfg(cfg):
    return replace(cfg, scoring=replace(cfg.scoring, cascade_enabled=True))


def _patch(monkeypatch):
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda ref, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream",
                        lambda mode, gen_dir, *a, **k: _FakeStream(digest=f"d-{gen_dir}"))
    monkeypatch.setattr(loop_mod, "upload_dir_to_hub_or_hf", _fake_upload)


def _runner(cfg, tmp_path, monkeypatch, *, bench_eval_fn, wallet=None):
    _patch(monkeypatch)
    runner = TrainerRunner(cfg=cfg, base_trainer=_FakeBaseTrainer(), work_root=tmp_path,
                           use_sandbox=False, bench_eval_fn=bench_eval_fn, wallet=wallet)
    runner._manifest_store = _FakeStore()
    return runner


def test_run_round_never_stamps_and_never_benches(cascade_cfg, tmp_path, monkeypatch):
    # The manifest must publish untouched and immediately: even with
    # cascade_enabled, run_round neither calls the bench nor stamps
    # bench_scores — the eval is strictly post-publish now.
    called: list = []
    runner = _runner(cascade_cfg, tmp_path, monkeypatch,
                     bench_eval_fn=lambda d: called.append(d) or _BENCH)
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    assert called == []
    assert manifest.entry_for_role("king").bench_scores is None
    assert manifest.entry_for_role("challenger").bench_scores is None
    assert b"bench_scores" not in manifest.canonical_body()


def test_post_publish_bench_reports_both_roles_signed(cascade_cfg, tmp_path, monkeypatch):
    seen: list = []

    def _bench_eval(ckpt_dir):
        seen.append(ckpt_dir)
        return _BENCH

    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=_bench_eval,
                     wallet=_FakeWallet())
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)

    report = runner.run_post_publish_bench(manifest)
    assert report is not None
    assert len(seen) == 2                                    # king AND challenger benched
    assert {e.role for e in report.entries} == {"king", "challenger"}
    assert report.entry_for("king").scores == _BENCH
    assert report.signature is not None                      # trainer-hotkey signed
    # Uploaded to the manifest-bucket store under the round key, parseable.
    text = runner._manifest_store.texts["benchmarks/round-1.json"]
    assert load_bench_report(text) == report


def test_markers_bracket_the_bench(cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a",
                                base_seed=1, block=10)
    # The live loop writes the pending marker BEFORE publish (the provisioner's
    # teardown-hold arm)…
    assert runner.will_run_post_publish_bench()
    runner._mark_bench_pending(manifest.round_id)
    pending = tmp_path / "1" / "bench_pending.json"
    assert pending.is_file()
    # …and the bench releases it on completion, recording the outcome.
    runner.run_post_publish_bench(manifest)
    assert not pending.exists()
    done = json.loads((tmp_path / "1" / "bench_complete.json").read_text())
    assert done == {"round_id": "1", "uploaded": True}


def test_bench_failure_never_raises_and_releases_hold(cascade_cfg, tmp_path, monkeypatch):
    def _boom(_ckpt):
        raise RuntimeError("sidecar down")

    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=_boom)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a",
                                base_seed=1, block=10)
    runner._mark_bench_pending(manifest.round_id)
    assert runner.run_post_publish_bench(manifest) is None   # swallowed, never raises
    assert runner._manifest_store.texts == {}                # nothing published
    # The hold is still released — a failed bench must free the pod now, not
    # ride out the provisioner's hold cap.
    assert not (tmp_path / "1" / "bench_pending.json").exists()
    done = json.loads((tmp_path / "1" / "bench_complete.json").read_text())
    assert done["uploaded"] is False


def test_one_role_miss_publishes_partial_report(cascade_cfg, tmp_path, monkeypatch):
    # extract-side miss for one checkpoint: the report ships what it has.
    results = iter([_BENCH, None])
    runner = _runner(cascade_cfg, tmp_path, monkeypatch,
                     bench_eval_fn=lambda d: next(results))
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    report = runner.run_post_publish_bench(manifest)
    assert [e.role for e in report.entries] == ["king"]


def test_disabled_or_unwired_is_a_no_op(cfg, cascade_cfg, tmp_path, monkeypatch):
    # cascade_enabled off (explicit — chain.toml ships armed since 2026-08-05)
    # ⇒ no bench, no report, no marker.
    off_cfg = replace(cfg, scoring=replace(cfg.scoring, cascade_enabled=False))
    called: list = []
    runner = _runner(off_cfg, tmp_path, monkeypatch,
                     bench_eval_fn=lambda d: called.append(d) or _BENCH)
    manifest = runner.run_round([_commit(0, "a", REF_A, 1)], king_hotkey="a",
                                base_seed=1, block=10)
    assert runner.will_run_post_publish_bench() is False
    assert runner.run_post_publish_bench(manifest) is None
    assert called == [] and runner._manifest_store.texts == {}
    # And no marker lifecycle either — the hold markers exist only where the
    # bench (and therefore the provisioner hold) is armed.
    assert not (tmp_path / "1" / "bench_pending.json").exists()
    assert not (tmp_path / "1" / "bench_complete.json").exists()
    # cascade_enabled on but no eval path wired ⇒ equally a no-op predicate.
    unwired = _runner(cascade_cfg, tmp_path / "b", monkeypatch, bench_eval_fn=None)
    assert unwired.will_run_post_publish_bench() is False


def test_restart_releases_a_stale_bench_hold(cascade_cfg, tmp_path, monkeypatch):
    # A trainer restart between publish and bench completion kills the bench
    # thread but leaves bench_pending armed; the restart re-entry skip path
    # must release the hold instead of billing the pod to the cap.
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    runner._mark_bench_pending("42")
    runner._release_stale_bench_hold("42")
    assert not (tmp_path / "42" / "bench_pending.json").exists()
    done = json.loads((tmp_path / "42" / "bench_complete.json").read_text())
    assert done == {"round_id": "42", "uploaded": False}
    # Idempotent-safe: an already-completed round is left alone…
    before = (tmp_path / "42" / "bench_complete.json").read_text()
    runner._release_stale_bench_hold("42")
    assert (tmp_path / "42" / "bench_complete.json").read_text() == before
    # …and a round with no markers at all stays marker-free.
    runner._release_stale_bench_hold("43")
    assert not (tmp_path / "43").exists()


def test_wandb_pair_logged_per_round(cascade_cfg, tmp_path, monkeypatch):
    # open_wandb_run also serves the per-run training mirrors; the bench pair
    # opens its own run (role "bench-<size>") — assert on that sink alone.
    sinks: dict[str, list] = {}

    class _Sink:
        def __init__(self, records):
            self.records = records

        def emit(self, record):
            self.records.append(record)

        def finish(self):
            self.records.append("finished")

    def _open(wcfg, **kw):
        return _Sink(sinks.setdefault(kw["role"], []))

    monkeypatch.setattr(loop_mod, "open_wandb_run", _open)
    wandb_cfg = replace(cascade_cfg, wandb=replace(cascade_cfg.wandb, enabled=True))
    runner = _runner(wandb_cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    commits = [_commit(0, "a", REF_A, 1), _commit(1, "b", REF_B, 1)]
    manifest = runner.run_round(commits, king_hotkey="a", base_seed=1, block=10)
    runner.run_post_publish_bench(manifest)
    size = cascade_cfg.throne_contracts()[0].arch_preset
    emitted = sinks[f"bench-{size}"]
    # One record per role, then closed.
    roles = [r["role"] for r in emitted if isinstance(r, dict)]
    assert sorted(roles) == ["challenger", "king"]
    assert emitted[-1] == "finished"
    assert all(r["gifteval_crps"] == _BENCH.gifteval_crps
               for r in emitted if isinstance(r, dict))


# ── cohort finals bench at uid-suffixed work dirs (r51 regression) ───────────
# A DEC-CA-0012 cohort trains challengers under challenger-u<uid> dirs
# (_final_repo_suffix), but the remote bench sweep looked at the bare
# "challenger" path — on the first 2-challenger mainnet final (r51) every
# challenger leg 404'd and the signed report shipped king-only.


def _duel_entry(role, uid, hotkey):
    from types import SimpleNamespace

    return SimpleNamespace(role=role, size="toto2-4m", miner_uid=uid,
                           miner_hotkey=hotkey, trained_pointer=f"ptr-{hotkey}")


def test_bench_role_dir_suffixes_only_cohort_challengers():
    king = _duel_entry("king", 7, "k")
    solo = [king, _duel_entry("challenger", 1, "a")]
    # pre-cohort shapes keep their exact dirs (inert-default bit-identity)
    assert loop_mod._bench_role_dir(solo, solo[0]) == "king"
    assert loop_mod._bench_role_dir(solo, solo[1]) == "challenger"
    cohort = [king, _duel_entry("challenger", 1, "a"), _duel_entry("challenger", 2, "b")]
    assert loop_mod._bench_role_dir(cohort, cohort[0]) == "king"
    assert loop_mod._bench_role_dir(cohort, cohort[1]) == "challenger-u1"
    assert loop_mod._bench_role_dir(cohort, cohort[2]) == "challenger-u2"


def test_cohort_remote_sweep_benches_every_suffixed_challenger(
        cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    runner.cascade_bench_plan = object()      # remote plan wired
    runner.remote_hosts = {"final": ["pod"]}  # truthiness gate only
    duel = [_duel_entry("king", 7, "k"),
            _duel_entry("challenger", 1, "a"), _duel_entry("challenger", 2, "b")]
    from types import SimpleNamespace

    host = SimpleNamespace(host="1.2.3.4", name="pod-0")
    for e in duel:
        runner._final_role_hosts[(e.role, e.size, e.miner_hotkey)] = host

    benched_dirs: list[str] = []

    def _fake_bench(host, rid, size, plan, *, work_root=None, role="king", **kw):
        benched_dirs.append(role)
        return {"report": role}

    monkeypatch.setattr("cascade.trainer.bench_hook.run_post_round_benchmark",
                        _fake_bench)
    monkeypatch.setattr(
        "cascade.eval.benchmarks.extract_bench_scores",
        lambda report: dict(gifteval_crps=0.4, gifteval_mase=0.8, boom_crps=0.5,
                            boom_mase=0.9, time_crps=0.3, time_mase=0.7))

    out = runner._bench_duel_checkpoints(duel, "9988", "toto2-4m")

    # every leg benched at its actual work dir — and every entry scored
    assert sorted(benched_dirs) == ["challenger-u1", "challenger-u2", "king"]
    assert set(out) == {e.trained_pointer for e in duel}


# ── boundary replay of published bench reports ───────────────────────────────
# The candidate pool otherwise only sees reports the in-process bench thread
# publishes: an out-of-band report (a mop-up after a failed leg, a publish
# during trainer downtime) never yielded candidates, and the promotion fired
# off an incomplete reign (r26's better leg missed the gen-3 member set).


class _FakePromotion:
    king_hotkey = "5King"

    def __init__(self):
        self.calls: list = []

    def record_bench(self, manifest, report):
        self.calls.append((manifest.round_id, tuple(e.role for e in report.entries)))
        return len(report.entries)


def _index_row(rid, esb, king):
    return {"round_id": rid, "status": "scored", "post_round_king_hotkey": king,
            "epoch_start_block": esb, "validator_hotkey": "v"}


def _stub_manifest_loader(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "cascade.shared.manifest.load_manifest",
        lambda text: SimpleNamespace(round_id=json.loads(text)["round_id"],
                                     warm_start_ckpt=""))


def _seed_replay_store(store, *, signed_ok=True):
    from cascade.shared.bench_report import (
        BenchEntry,
        BenchReport,
        bench_report_key,
        dump_bench_report,
    )
    from cascade.shared.hippius import RECEIPT_INDEX_KEY, manifest_round_key

    store.texts[RECEIPT_INDEX_KEY] = json.dumps({"rounds": [
        _index_row("700", 100, "OldKing"),   # prior reign: outside the tail
        _index_row("776", 200, "5King"),     # reign round, report never published
        _index_row("777", 300, "5King"),     # reign round with an OOB report
    ]})
    store.texts[manifest_round_key("777")] = json.dumps({"round_id": "777"})
    report = BenchReport(round_id="777", created_block=5, entries=(
        BenchEntry(role="king", size="toto2-4m", miner_hotkey="a", miner_uid=1,
                   trained_pointer="ptr-k", scores=_BENCH),
        BenchEntry(role="challenger", size="toto2-4m", miner_hotkey="b",
                   miner_uid=2, trained_pointer="ptr-c", scores=_BENCH),
    ))
    store.texts[bench_report_key("777")] = dump_bench_report(report)


def test_boundary_replay_ingests_out_of_band_bench_reports(cascade_cfg, tmp_path,
                                                           monkeypatch):
    # trainer_hotkey blanked: this test probes the WIRING (tail → fetch →
    # record_bench); the signature discipline gets its own test below.
    cfg = replace(cascade_cfg, manifest=replace(cascade_cfg.manifest,
                                                trainer_hotkey=""))
    runner = _runner(cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    _stub_manifest_loader(monkeypatch)
    store = runner.manifest_store()
    _seed_replay_store(store)
    runner.promotion = _FakePromotion()

    runner._replay_reign_bench_reports()

    # 777 replayed; 776 (no report) skipped silently; 700 (prior reign) never
    # part of the tail. record_bench owns dedup/generation gating downstream.
    assert runner.promotion.calls == [("777", ("king", "challenger"))]
    from cascade.shared.hippius import manifest_round_key

    assert manifest_round_key("700") not in store.reads

    # idempotent to re-run at the next boundary (record_bench dedupes; the
    # replay itself must not error or grow state)
    runner._replay_reign_bench_reports()
    assert len(runner.promotion.calls) == 2


def test_boundary_replay_rejects_unsigned_reports_under_a_pinned_hotkey(
        cascade_cfg, tmp_path, monkeypatch):
    # With a pinned trainer hotkey, an unsigned/forged report replays NOTHING —
    # same discipline as the deploy-time backfill.
    cfg = replace(cascade_cfg, manifest=replace(cascade_cfg.manifest,
                                                trainer_hotkey="5Pinned"))
    runner = _runner(cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    _stub_manifest_loader(monkeypatch)
    store = runner.manifest_store()
    _seed_replay_store(store)
    runner.promotion = _FakePromotion()

    runner._replay_reign_bench_reports()
    assert runner.promotion.calls == []


def test_boundary_replay_is_a_no_op_without_an_anchored_engine(
        cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch,
                     bench_eval_fn=lambda d: _BENCH)
    runner.promotion = None
    runner._replay_reign_bench_reports()    # must not raise, must not read
    assert runner.manifest_store().reads == []


# ── DEC-CA-0036: funded challengers bench on their payer's pod, verified on the king's ──

def _scores(x):
    return dict(gifteval_crps=x, gifteval_mase=x, boom_crps=x,
                boom_mase=x, time_crps=x, time_mase=x)


def _payer_sweep(runner, monkeypatch, *, payer_reported, operator_actual,
                 verify_top=1, king_host=True):
    """Wire a king (operator pod) + two funded challengers (payer pods, kept
    for the bench). ``payer_reported`` maps hotkey → the number the payer pod
    reports; ``operator_actual`` maps hotkey → what the king pod re-bench
    finds. Returns (duel, benched-on log, torn-down hotkeys, pushed dirs)."""
    from types import SimpleNamespace

    runner.cascade_bench_plan = object()
    runner.remote_hosts = {}                       # no standing fleet (JIT king)
    runner.cfg = replace(runner.cfg, telemetry=replace(
        runner.cfg.telemetry, funded_bench_verify_top=verify_top))
    duel = [_duel_entry("king", 7, "k"),
            _duel_entry("challenger", 1, "a"), _duel_entry("challenger", 2, "b")]
    king = SimpleNamespace(host="1.1.1.1", name="king-pod", isolated=False)
    if king_host:
        runner._final_role_hosts[("king", "toto2-4m", "k")] = king
    for hk in ("a", "b"):
        runner._final_role_hosts[("challenger", "toto2-4m", hk)] = SimpleNamespace(
            host=f"9.9.9.{ord(hk)}", name=f"funded-{hk}", isolated=True)
        runner._funded_bench_pods[hk] = SimpleNamespace(instance_id=f"pod-{hk}",
                                                        payer_hotkey=hk)
    runner._funded_field = {"a": "ref-a", "b": "ref-b"}
    torn: list[str] = []
    runner._teardown_funded_pod = lambda pod: torn.append(pod.payer_hotkey)
    benched: list[tuple[str, str]] = []
    pushed: list[str] = []

    def _fake_bench(host, rid, size, plan, *, work_root=None, role="king", **kw):
        benched.append((host.name, role))
        if host.isolated:
            hk = host.name.split("-")[1]
            return {"scores": payer_reported[hk]}
        if role.endswith("-verify"):
            hk = role.split("-u")[1].split("-")[0]
            hk = {"1": "a", "2": "b"}[hk]
            return {"scores": operator_actual[hk]}
        return {"scores": 0.5}

    monkeypatch.setattr("cascade.trainer.bench_hook.run_post_round_benchmark",
                        _fake_bench)
    monkeypatch.setattr("cascade.trainer.bench_hook.push_checkpoint",
                        lambda host, local, rid, size, role, **kw: pushed.append(role))
    monkeypatch.setattr("cascade.eval.benchmarks.extract_bench_scores",
                        lambda report: _scores(report["scores"]))
    runner._fetch_checkpoint_dir = lambda ptr: runner.work_root / ptr
    return duel, benched, torn, pushed


def test_payer_pod_sweep_tears_each_pod_down_when_its_bench_ends(
        cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    duel, benched, torn, _ = _payer_sweep(
        runner, monkeypatch, payer_reported={"a": 0.4, "b": 0.45},
        operator_actual={"a": 0.4, "b": 0.45})
    out = runner._bench_duel_checkpoints(duel, "9988", "toto2-4m")
    # every checkpoint benched where it was trained (three pods, in parallel)
    assert sorted(benched) == [("funded-a", "challenger-u1"),
                               ("funded-b", "challenger-u2"), ("king-pod", "king")]
    assert set(out) == {"ptr-k", "ptr-a", "ptr-b"}
    assert runner._payer_benched == {"ptr-a", "ptr-b"}
    # each payer pod went down as its own sweep ended; nothing left kept
    assert sorted(torn) == ["a", "b"] and runner._funded_bench_pods == {}


def test_payer_numbers_never_publish_unverified(cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    duel, benched, _, pushed = _payer_sweep(
        runner, monkeypatch, payer_reported={"a": 0.40, "b": 0.45},
        operator_actual={"a": 0.405, "b": 0.30}, verify_top=1)
    scored = runner._bench_duel_checkpoints(duel, "9988", "toto2-4m")
    out = runner._verify_payer_benches(duel, scored, "9988", "toto2-4m")
    # only the best payer-reported challenger (a) was verified, on the KING pod
    assert pushed == ["challenger-u1-verify"]
    assert ("king-pod", "challenger-u1-verify") in benched
    assert ("king-pod", "challenger-u2-verify") not in benched
    # a's published numbers are the OPERATOR's, not the payer's
    assert out["ptr-a"].gifteval_crps == 0.405
    # b (unverified) is absent from what gets signed; the king is untouched
    assert "ptr-b" not in out and out["ptr-k"] == scored["ptr-k"]


def test_forged_payer_bench_is_dropped(cascade_cfg, tmp_path, monkeypatch, caplog):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    duel, _, _, _ = _payer_sweep(
        runner, monkeypatch, payer_reported={"a": 0.20, "b": 0.45},
        operator_actual={"a": 0.40, "b": 0.45})
    scored = runner._bench_duel_checkpoints(duel, "9988", "toto2-4m")
    with caplog.at_level("ERROR", logger="cascade.trainer"):
        out = runner._verify_payer_benches(duel, scored, "9988", "toto2-4m")
    # neither the payer's number nor the operator's stands in for a forger
    assert "ptr-a" not in out and "ptr-k" in out
    assert any("forged sweep" in r.message for r in caplog.records)


def test_payer_bench_within_tolerance_publishes_the_operators_numbers(
        cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    # payer reports 1% better than the operator finds: sweep noise, not forgery
    duel, _, _, _ = _payer_sweep(
        runner, monkeypatch, payer_reported={"a": 0.396, "b": 0.45},
        operator_actual={"a": 0.40, "b": 0.45})
    scored = runner._bench_duel_checkpoints(duel, "9988", "toto2-4m")
    out = runner._verify_payer_benches(duel, scored, "9988", "toto2-4m")
    assert out["ptr-a"].boom_mase == 0.40


def test_verify_top_two_re_benches_both(cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    duel, _, _, pushed = _payer_sweep(
        runner, monkeypatch, payer_reported={"a": 0.40, "b": 0.45},
        operator_actual={"a": 0.40, "b": 0.45}, verify_top=2)
    scored = runner._bench_duel_checkpoints(duel, "9988", "toto2-4m")
    out = runner._verify_payer_benches(duel, scored, "9988", "toto2-4m")
    assert pushed == ["challenger-u1-verify", "challenger-u2-verify"]
    assert {"ptr-a", "ptr-b"} <= set(out)


def test_no_king_pod_means_no_payer_number_publishes(cascade_cfg, tmp_path, monkeypatch):
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    duel, _, _, pushed = _payer_sweep(
        runner, monkeypatch, payer_reported={"a": 0.40, "b": 0.45},
        operator_actual={"a": 0.40, "b": 0.45}, king_host=False)
    scored = runner._bench_duel_checkpoints(duel, "9988", "toto2-4m")
    out = runner._verify_payer_benches(duel, scored, "9988", "toto2-4m")
    assert pushed == [] and not ({"ptr-a", "ptr-b"} & set(out))


def test_funded_challenger_without_a_tracked_host_is_never_guessed(
        cascade_cfg, tmp_path, monkeypatch):
    """A funded challenger's checkpoint only ever lived on its payer's pod;
    losing the tracking dict must not bench a stranger's path on the king."""
    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=None)
    runner._funded_field = {"a": "ref-a"}
    runner._hosts_for = lambda stage: ["king-pod"]
    assert runner._bench_host_for(_duel_entry("challenger", 1, "a"), "toto2-4m") is None
    # an operator-lane challenger keeps the legacy heuristic
    assert runner._bench_host_for(_duel_entry("challenger", 2, "z"), "toto2-4m") == "king-pod"


def test_bench_thread_exit_sweeps_kept_payer_pods(cascade_cfg, tmp_path, monkeypatch):
    """The bench thread's finally is the last line of defence for a kept
    payer pod: a sweep that never reached _bench_pod_group still ends its bill."""
    from types import SimpleNamespace

    runner = _runner(cascade_cfg, tmp_path, monkeypatch, bench_eval_fn=lambda d: _BENCH)
    torn: list[str] = []
    runner._teardown_funded_pod = lambda pod: torn.append(pod.payer_hotkey)
    runner._funded_bench_pods["a"] = SimpleNamespace(instance_id="p", payer_hotkey="a")
    monkeypatch.setattr(runner, "_post_publish_bench",
                        lambda manifest: (_ for _ in ()).throw(RuntimeError("boom")))
    manifest = SimpleNamespace(round_id="1", created_block=1, entries=[],
                               warm_start_ckpt="")
    runner.run_post_publish_bench(manifest)
    assert torn == ["a"] and runner._funded_bench_pods == {}
