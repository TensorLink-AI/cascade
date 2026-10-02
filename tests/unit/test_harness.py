"""The mining gauntlet (cascade.miner.harness): config, round window, spend
ledger + Lium executor, submit guardrails, worker queue, and an end-to-end
gauntlet run on fake compute (no GPU, no network, no LLM)."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.scoring import WindowScore
from cascade.miner.harness import config as hconfig
from cascade.miner.harness import rounds as hrounds
from cascade.miner.harness.executor import LiumExecutor, LocalExecutor, SpendLedger
from cascade.miner.harness.jobs import scores_from_json, scores_to_json
from cascade.miner.harness.selftest import (
    ScriptedWorkers,
    build_selftest,
    make_generator,
    make_round,
    run_selftest,
    selftest_chain,
)
from cascade.miner.harness.submit import Submitter
from cascade.miner.harness.workers import (
    Outcome,
    Proposal,
    QueueWorkers,
    run_agent,
    serve_queue,
)
from cascade.miner.replay import SHA_MARKER
from cascade.shared.receipt import (
    dump_receipt,
)

REPO = Path(__file__).resolve().parents[2]

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def rcfg(cfg):
    return selftest_chain(cfg)


_gen = make_generator
_make_round = make_round


def _gauntlet(rcfg, tmp_path, *, script, submit_mode="approval", n_rounds=5, **sections):
    calls: list = []
    g, compute, sub = build_selftest(tmp_path, rcfg, script=script, submit_mode=submit_mode,
                                     n_rounds=n_rounds,
                                     submit_runner=lambda argv: calls.append(argv) or 0,
                                     **sections)
    return g, compute, sub, calls


def _hcfg(tmp_path: Path, **sections) -> hconfig.HarnessConfig:
    king = _gen(tmp_path / "king")
    h = hconfig.HarnessConfig(workdir=tmp_path / "gw", start_dir=king, king_dir=king,
                              rounds=hconfig.RoundsConfig(snapshot_root=tmp_path / "pool",
                                                          sync_reveals=False, n_a=3, n_b=2))
    return hconfig.with_overrides(h, **sections) if sections else h


# ── config ────────────────────────────────────────────────────────────────────


def test_example_config_loads_and_validates(tmp_path):
    p = tmp_path / "harness.toml"
    p.write_text((REPO / "deploy/harness/harness.example.toml").read_text())
    h = hconfig.load_harness_config(p)
    assert h.compute.executor == "lium" and h.compute.daily_usd_cap == 40.0
    assert h.workdir == tmp_path / "gauntlet"
    assert h.rounds.snapshot_root == tmp_path / "eval-pool"
    assert h.submit.mode == "approval"


@pytest.mark.parametrize("patch, match", [
    ("daily_usd_cap      = 40.0", "daily_usd_cap"),
    ("max_price_per_hour = 1.10", "max_price_per_hour"),
])
def test_lium_needs_spend_bounds(tmp_path, patch, match):
    p = tmp_path / "harness.toml"
    src = (REPO / "deploy/harness/harness.example.toml").read_text()
    p.write_text(src.replace(patch, patch.split("=")[0] + "= 0"))
    with pytest.raises(ValueError, match=match):
        hconfig.load_harness_config(p)


def test_autonomous_needs_hotkeys_and_a_real_margin():
    h = hconfig.HarnessConfig(submit=hconfig.SubmitConfig(mode="autonomous"))
    with pytest.raises(ValueError, match="hotkeys"):
        h.validate()
    h = hconfig.HarnessConfig(submit=hconfig.SubmitConfig(
        mode="autonomous", intake="https://x", wallet_name="w", hotkeys=("hk",), margin=0.001))
    with pytest.raises(ValueError, match="margin"):
        h.validate()


def test_unknown_keys_are_refused(tmp_path):
    p = tmp_path / "h.toml"
    p.write_text("[compute]\nexecuter = 'lium'\n")
    with pytest.raises(ValueError, match="unknown"):
        hconfig.load_harness_config(p)


# ── rounds ────────────────────────────────────────────────────────────────────


def test_window_slides_as_rounds_become_replayable(rcfg, tmp_path):
    root, rdir = tmp_path / "pool", tmp_path / "receipts"
    receipts = [_make_round(rcfg, root, i)[0] for i in range(4)]
    rows = [{"round_id": r.round_id, "status": "scored", "epoch_start_block":
             r.epoch_start_block, "receipt_key": f"k/{r.round_id}"} for r in receipts]
    texts = {f"k/{r.round_id}": dump_receipt(r) for r in receipts}
    n = hrounds.refresh_receipts(rcfg, rdir, index_fetch=lambda: {"rounds": rows[:3]},
                                 text_fetch=texts.__getitem__)
    assert n == 3
    refs = hrounds.replayable_rounds(rdir, root)
    w1 = hrounds.build_window(refs, n_a=2, n_b=1)
    assert [r.round_id for r in w1.b] == [receipts[2].round_id]
    assert [r.round_id for r in w1.a] == [receipts[0].round_id, receipts[1].round_id]
    hrounds.refresh_receipts(rcfg, rdir, index_fetch=lambda: {"rounds": rows},
                             text_fetch=texts.__getitem__)
    w2 = hrounds.build_window(hrounds.replayable_rounds(rdir, root), n_a=2, n_b=1)
    assert [r.round_id for r in w2.b] == [receipts[3].round_id]        # slid by one
    assert hrounds.fingerprint(w1, king="a") != hrounds.fingerprint(w2, king="a")
    assert hrounds.fingerprint(w2, king="a") != hrounds.fingerprint(w2, king="b")
    # an unreachable index never raises: the window just does not move
    assert hrounds.refresh_receipts(rcfg, rdir, index_fetch=lambda: 1 / 0) == 0


def test_rounds_without_a_local_snapshot_are_not_replayable(rcfg, tmp_path):
    root, rdir = tmp_path / "pool", tmp_path / "receipts"
    r, _ = _make_round(rcfg, root, 0)
    rdir.mkdir()
    (rdir / f"{r.round_id}.json").write_text(dump_receipt(r))
    assert len(hrounds.replayable_rounds(rdir, root)) == 1
    assert hrounds.replayable_rounds(rdir, tmp_path / "empty") == []
    assert hrounds.needed_snapshot_blocks(rdir) == {5000}


def test_sync_snapshots_downloads_only_revealed_missing_blocks(tmp_path):
    root = tmp_path / "pool"
    (root / "snapshots" / "2026-09-01-block-100").mkdir(parents=True)
    (root / "snapshots" / "2026-09-01-block-100" / SHA_MARKER).write_text("aa")
    got = []
    res = hrounds.sync_snapshots(
        {100, 200, 300}, root, repo="x",
        list_folders=lambda: ["snapshots/2026-09-01-block-100", "snapshots/2026-09-02-block-200"],
        download=lambda folder, r: got.append(folder))
    assert got == ["snapshots/2026-09-02-block-200"]
    assert res == {"fetched": ["snapshots/2026-09-02-block-200"], "unrevealed": [300]}


# ── spend ledger + executors ─────────────────────────────────────────────────


def test_ledger_accrues_per_utc_day(tmp_path):
    t = {"now": 1_790_000_000.0 - 1_790_000_000.0 % 86400 + 86400 - 1800}   # 23:30 UTC
    led = SpendLedger(tmp_path / "s.json", now=lambda: t["now"])
    led.open("p1", 2.0)
    t["now"] += 3600                                                      # 00:30 next day
    assert led.spent_today() == pytest.approx(1.0)                        # 30 min after midnight
    assert led.spent_total() == pytest.approx(2.0)
    led.close("p1")
    t["now"] += 3600
    assert led.spent_today() == pytest.approx(1.0) and led.live() == []


class _FakeProvider:
    def __init__(self, tagged=()):
        self.launched, self.terminated, self.tagged = [], [], list(tagged)

    def list_tagged(self, prefix):
        return [t for t in self.tagged if t.startswith(prefix)]

    def launch(self, spec):
        self.launched.append(spec)
        return [f"{spec.name_prefix}-0"]

    def wait_ready(self, name, timeout):
        return True

    def get_ip(self, name):
        return SimpleNamespace(ip="10.0.0.1", ssh_port=2222)

    def terminate(self, name):
        self.terminated.append(name)


def _lium(tmp_path, provider, clock, **kw):
    key = tmp_path / "key"
    key.write_text("k")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA test")
    c = hconfig.ComputeConfig(executor="lium", daily_usd_cap=kw.pop("cap", 10.0),
                              max_price_per_hour=2.0, image="img@sha256:" + "a" * 64,
                              ssh_key=key, max_parallel=kw.pop("par", 1), idle_minutes=10)
    ok = SimpleNamespace(returncode=0, stdout="", stderr="")

    class FakeTransfer:
        pushed: list = []

        def push(self, pod, src, dst):
            self.pushed.append(dst)
            return True

        def pull_file(self, pod, src, dst):
            if src.endswith("result.json"):
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps({"kind": "replay", "geomean": 1.0}))
            return True

        def pull_dir(self, pod, src, dst):
            return True

    return LiumExecutor(tmp_path / "wd", c, provider=provider, transfer=FakeTransfer(),
                        ssh=lambda ip, port, cmd, t: ok, now=lambda: clock["t"])


def test_tar_ssh_transfer_round_trips_files_and_dirs(tmp_path):
    from cascade.miner.harness.executor import TarSsh

    # a local stand-in for ssh: drop the "root@host" argument, run the command
    shim = tmp_path / "fake-ssh"
    shim.write_text('#!/bin/sh\nshift\nexec sh -c "$1"\n')
    shim.chmod(0o755)
    t = TarSsh(lambda port: [str(shim)])
    pod = SimpleNamespace(ip="h", port=22)
    src = _gen(tmp_path / "src")
    (src / "sub").mkdir()
    (src / "sub" / "x.bin").write_bytes(b"\x00\x01")
    remote = tmp_path / "remote"
    assert t.push(pod, src, str(remote / "gen"))
    assert (remote / "gen" / "sub" / "x.bin").read_bytes() == b"\x00\x01"
    assert t.push(pod, src / "config.json", str(remote / "deep" / "c.json"))
    assert t.pull_file(pod, str(remote / "deep" / "c.json"), tmp_path / "back" / "c.json")
    assert (tmp_path / "back" / "c.json").read_text() == (src / "config.json").read_text()
    assert t.pull_dir(pod, str(remote / "gen"), tmp_path / "back" / "gen")
    assert (tmp_path / "back" / "gen" / "sub" / "x.bin").is_file()
    assert not t.pull_file(pod, str(remote / "missing"), tmp_path / "back" / "m")
    assert not (tmp_path / "back" / "m").exists()


def test_lium_reaps_orphans_runs_jobs_and_respects_the_cap(tmp_path):
    clock = {"t": 1_790_000_000.0 - 1_790_000_000.0 % 86400 + 3600}
    prov = _FakeProvider(tagged=["cascade-gauntlet-old-0", "someone-else"])
    ex = _lium(tmp_path, prov, clock, cap=10.0)
    assert prov.terminated == ["cascade-gauntlet-old-0"]        # a crashed run's leftover
    gen = _gen(tmp_path / "g")
    res = ex.run({"kind": "replay", "inputs": {"gen": str(gen)}, "params": {}}, est_hours=1.0)
    assert res == {"kind": "replay", "geomean": 1.0}
    assert len(prov.launched) == 1 and ex.ledger.live()
    # 4.5 h later the pod has accrued $9 at the $2 cap: a 1 h job ($2) would pass $10
    clock["t"] += 4.5 * 3600
    res = ex.run({"kind": "replay", "inputs": {}, "params": {}}, est_hours=1.0)
    assert res.get("budget_exhausted") and len(prov.launched) == 1
    ex.reap()                                                    # idle 4 h > 10 min
    assert ex.ledger.live() == [] and len(prov.terminated) == 2


def test_lium_refuses_to_launch_past_the_cap(tmp_path):
    clock = {"t": 1_790_000_000.0}
    prov = _FakeProvider()
    ex = _lium(tmp_path, prov, clock, cap=1.0)                   # $1 < one 1 h job at $2/h
    res = ex.run({"kind": "replay", "inputs": {}, "params": {}}, est_hours=1.0)
    assert res["budget_exhausted"] and prov.launched == []


def test_local_executor_runs_jobs_in_slots(tmp_path):
    seen = []

    def runner(argv, env, timeout):
        seen.append(env.get("CUDA_VISIBLE_DEVICES"))
        Path(argv[-1]).write_text(json.dumps({"ok": True}))
        return 0

    ex = LocalExecutor(tmp_path, max_parallel=2, runner=runner)
    assert ex.run({"kind": "throughput", "inputs": {}}) == {"ok": True}
    assert seen == ["0"]
    ex2 = LocalExecutor(tmp_path, runner=lambda a, e, t: 1)
    assert ex2.run({"kind": "throughput", "inputs": {}})["candidate_fault"] is False


def test_scores_round_trip_through_json():
    s = [WindowScore("w1", 0.9, np.array([0.1, 0.2]), 5.0, source="x")]
    back = scores_from_json(json.loads(json.dumps(scores_to_json(s))))
    assert back[0].series_id == "w1" and back[0].source == "x"
    assert list(back[0].qloss_per_q) == [0.1, 0.2]


# ── submit (G5) ───────────────────────────────────────────────────────────────


def _sub(tmp_path, **kw):
    calls = []
    cfg = hconfig.SubmitConfig(**{"mode": "autonomous", "intake": "https://x",
                                  "wallet_name": "w", "hotkeys": ("hk1", "hk2"),
                                  "margin": 0.01, **kw})
    return Submitter(cfg, tmp_path / "wd", runner=lambda argv: calls.append(argv) or 0), calls


GOOD = {"g4": {"pass": True, "rel": 0.02}, "g45": {"pass": True, "rel": 0.015},
        "g45_required": True}


def test_autonomous_submits_when_every_guardrail_holds(tmp_path, monkeypatch):
    monkeypatch.setenv("LIUM_API_KEY", "x")
    s, calls = _sub(tmp_path)
    out = s.offer("c00001", _gen(tmp_path / "t1", quality=2), GOOD)
    assert out["action"] == "submitted" and out["hotkey"] == "hk1"
    assert calls[0][:3] == ["cascade", "submit", str(tmp_path / "wd/submit/frozen/c00001")]
    # max_per_day = 1: the next finalist waits for a person instead
    out = s.offer("c00002", _gen(tmp_path / "t2", quality=3), GOOD)
    assert out["action"] == "pending" and any("24h" in r for r in out["refused"])
    assert len(calls) == 1


@pytest.mark.parametrize("evidence, env, why", [
    ({**GOOD, "g45": {"pass": True, "rel": 0.004}}, True, "margin"),
    ({**GOOD, "g45": {"pass": False}}, True, "did not pass"),
    (GOOD, False, "LIUM_API_KEY"),
])
def test_autonomous_guardrails_fall_back_to_approval(tmp_path, monkeypatch, evidence, env, why):
    if env:
        monkeypatch.setenv("LIUM_API_KEY", "x")
    else:
        monkeypatch.delenv("LIUM_API_KEY", raising=False)
    s, calls = _sub(tmp_path)
    out = s.offer("c00001", _gen(tmp_path / "t"), evidence)
    assert out["action"] == "pending" and any(why in r for r in out["refused"])
    assert calls == [] and [p["id"] for p in s.pending()] == ["c00001"]


def test_kill_switch_and_hotkey_pool(tmp_path, monkeypatch):
    monkeypatch.setenv("LIUM_API_KEY", "x")
    s, calls = _sub(tmp_path, max_per_day=5, hotkeys=("hk1",))
    (s.dir / "HOLD").touch()
    assert s.offer("c1", _gen(tmp_path / "a", quality=2), GOOD)["action"] == "pending"
    (s.dir / "HOLD").unlink()
    assert s.offer("c2", _gen(tmp_path / "b", quality=3), GOOD)["action"] == "submitted"
    out = s.offer("c3", _gen(tmp_path / "c", quality=4), GOOD)
    assert any("no unused hotkey" in r for r in out["refused"])


def test_approval_mode_waits_for_a_person(tmp_path, monkeypatch):
    s, calls = _sub(tmp_path, mode="approval")
    assert s.offer("c1", _gen(tmp_path / "a"), GOOD)["action"] == "pending"
    assert calls == []
    out = s.approve("c1", "hk9")
    assert out["action"] == "submitted" and calls[0][-3] == "hk9"
    with pytest.raises(ValueError):
        s.approve("c1", "hk9")                                    # no longer pending


# ── workers ───────────────────────────────────────────────────────────────────


def _wcfg():
    return hconfig.WorkersConfig(llm_provider="anthropic", agent_timeout=30)


def _editing_runner(p: Proposal, env):
    cfg = json.loads((p.tree / "config.json").read_text())
    cfg["quality"] = cfg["quality"] + 0.5
    (p.tree / "config.json").write_text(json.dumps(cfg))
    (p.tree / ".mine-note.md").write_text("raise quality\n")
    (p.tree / ".lesson.md").write_text("quality matters")
    return 0


def test_run_agent_collects_note_and_lesson_and_leaves_tree_clean(tmp_path):
    tree = _gen(tmp_path / "c" / "tree")
    o = run_agent(Proposal("c1", "king", tree, "prompt"), _wcfg(), home=tmp_path,
                  runner=_editing_runner)
    assert o.ok and o.note == "raise quality" and o.lesson == "quality matters"
    assert not (tree / ".mine-note.md").exists() and not (tree / ".lesson.md").exists()


def test_queue_round_trip_isolates_the_worker(tmp_path):
    q = tmp_path / "queue"
    judge = QueueWorkers(q, timeout=30, poll=0.05)
    tree = _gen(tmp_path / "cands" / "c1" / "tree")
    t = threading.Thread(target=lambda: judge_out.extend(
        judge.run([Proposal("c1", "king", tree, "p")])))
    judge_out: list[Outcome] = []
    t.start()
    while not any((q / "pending").iterdir()):
        pass
    assert serve_queue(q, _wcfg(), home=tmp_path / "wh", once=True,
                       runner=_editing_runner) == 1
    t.join(10)
    assert judge_out and judge_out[0].ok and judge_out[0].note == "raise quality"
    assert json.loads((tree / "config.json").read_text())["quality"] == 1.5


# ── the gauntlet, end to end on fake compute ─────────────────────────────────


def test_gauntlet_end_to_end_funnel(rcfg, tmp_path):
    # four proposals: a clear winner, a slow winner, a no-op loser, a mild loser
    g, compute, sub, _ = _gauntlet(rcfg, tmp_path, script=[
        (1.6, 1.0),      # big improvement → member → G4 → G4.5 → pending approval
        (1.6, 0.5),      # same quality but 50 % slower → dies at G1
        (0.8, 1.0),      # worse → dies at G2
        (1.0005, 1.0)])  # inside the noise margin → dies at G2
    assert g.cycle() == "ran"
    st = g.state
    assert st["epoch"] == 1 and st["king_ready"]
    w = g.window
    assert len(w.a) == 3 and len(w.b) == 2
    metas = {m["id"]: m for m in g.all_metas()}
    assert set(metas) == {"c00001", "c00002", "c00003", "c00004"}
    assert metas["c00002"]["reason"].startswith("G1")
    assert metas["c00003"]["reason"].startswith("G2")
    assert metas["c00004"]["reason"].startswith("G2")
    win = metas["c00001"]
    assert win["stages"]["G3"]["pass"] and win["stages"]["G4"]["pass"]
    assert win["stages"]["G4"]["wins"] == 3
    assert win["stages"]["G4.5"]["pass"] and win["status"] == "finalist"
    assert [p["id"] for p in sub.pending()] == ["c00001"]
    # every G2 screen of the cycle used ONE pool-A round, paired with the king there
    g2_rounds = {metas[c]["stages"]["G2"]["round"] for c in ("c00001", "c00003", "c00004")}
    assert g2_rounds <= {r.round_id for r in w.a}
    # the operator view carries no G3+ numbers
    op = json.loads((g.wd / "operator" / "status.json").read_text())
    assert op["finalists"][0]["id"] == "c00001"
    assert "lcb" not in json.dumps(op) and "G3" not in json.dumps(op["population"])
    # G4 kept the checkpoint of the newest round for G4.5
    newest = w.newest(1)[0]
    assert (g.wd / "candidates" / "c00001" / "ckpt" / newest.round_id).is_dir()


def test_gauntlet_autonomous_mode_submits_the_finalist(rcfg, tmp_path, monkeypatch):
    monkeypatch.setenv("LIUM_API_KEY", "x")
    g, _, sub, calls = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)],
                                 submit_mode="autonomous")
    assert g.cycle() == "ran"
    assert g.meta("c00001")["status"] == "submitted"
    assert calls and calls[0][1] == "submit" and "hk1" in calls[0]


def test_gauntlet_epochs_and_population_survive_a_window_slide(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (0.8, 1.0)])
    assert g.cycle() == "ran"
    assert g.meta("c00001")["status"] == "finalist"
    # the same window again: no new epoch, the king baseline is reused
    n_jobs = len(compute.jobs)
    assert g.refresh_window(force=True) is False and g.epoch == 1
    # a changed king is a new epoch
    (Path(g.h.king_dir) / "generator.py").write_text("# new king\n")
    assert g.refresh_window(force=True) is True and g.epoch == 2
    assert not g.state["king_ready"] and len(compute.jobs) == n_jobs


def test_gauntlet_requeues_infra_faults_and_kills_candidate_faults(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (1.7, 1.0)])
    real = compute.run
    flaky = {"n": 0}

    def run(spec, **kw):
        gen = spec["inputs"].get("gen", "")
        if spec["kind"] == "replay" and "c00001" in gen and spec["params"]["train_hours"] == 0.25:
            flaky["n"] += 1
            if flaky["n"] == 1:
                return {"error": "ssh reset", "candidate_fault": False}     # retried
        if spec["kind"] == "replay" and "c00002" in gen:
            return {"error": "CorpusError: generator crashed", "candidate_fault": True}
        return real(spec, **kw)

    compute.run = run
    assert g.cycle() == "ran"
    assert flaky["n"] == 2 and g.meta("c00001")["status"] in ("member", "finalist")
    assert g.meta("c00002")["status"] == "dead"
    assert g.meta("c00002")["reason"].startswith("G2: CorpusError")


def test_gauntlet_waits_until_enough_rounds_are_replayable(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)], n_rounds=2)
    g.h = hconfig.with_overrides(g.h, rounds={"n_b": 2})
    assert g.cycle() == "wait" and "waiting" in g.state["status"]
    assert compute.jobs == []


def test_gauntlet_stop_file_and_operator_parent_pin(rcfg, tmp_path):
    g, _, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (0.9, 1.0), (0.9, 1.0)])
    assert g.cycle() == "ran"
    (g.wd / "operator" / "DIRECTIVES.md").write_text("focus on energy\nparent: c00001\n")
    assert g.parents(3) == ["c00001"] * 3
    (g.wd / "operator" / "STOP").touch()
    assert g.stopped()
    g.run(max_cycles=5)                                  # returns at once
    assert g.state["phase"] == "stopped"


def test_new_epoch_reconfirms_the_population(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (1.3, 1.0)])
    assert g.cycle() == "ran"
    assert g.meta("c00001")["status"] == "finalist"
    assert g.meta("c00002")["status"] == "member"
    (Path(g.h.king_dir) / "generator.py").write_text("# new king\n")   # → epoch 2
    g.workers = ScriptedWorkers([(0.9, 1.0)])
    g.h = hconfig.with_overrides(g.h, search={"proposals_per_cycle": 1})
    g.state["last_refresh"] = 0
    assert g.cycle() == "ran" and g.epoch == 2
    # both re-ran G3 on the epoch-2 window and kept their places; the finalist
    # is NOT re-run through G4, so this epoch's G4 slot goes to c00002
    for cid in ("c00001", "c00002"):
        m = g.meta(cid)
        assert m["status"] in ("member", "finalist") and m["member_epoch"] == 2
        assert m["stages"]["G3"]["epoch"] == 2
    assert g.meta("c00001")["stages"]["G4"]["epoch"] == 1      # not repeated
    assert g.meta("c00002")["stages"]["G4"]["epoch"] == 2


def test_throughput_job_runs_the_real_stream(cfg):
    from cascade.miner.harness.jobs import run_job

    gen = str(REPO / "scripts" / "example_generator")
    res = run_job({"kind": "throughput", "inputs": {"gen": gen, "king": gen},
                   "params": {"seconds": 1.0}}, cfg=cfg)
    assert res["verify_ok"], res
    assert res["gen"]["points_per_sec"] > 0 and res["king"]["series"] > 0


def test_selftest_command_passes(capsys):
    assert run_selftest() == 0
    assert "gauntlet selftest OK" in capsys.readouterr().out


# ── self-update handshake + live king ────────────────────────────────────────


def test_restart_handshake_parks_between_cycles(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    (g.wd / "RESTART").touch()
    g.run(max_cycles=3)                         # never starts a cycle
    assert (g.wd / "RESTART.ack").is_file() and compute.jobs == []
    assert g.state["phase"] == "restarting for an image update"
    # restarted by its policy before the updater recreated it: parked, no work
    slept = []

    def sleep(s):
        slept.append(s)
        (g.wd / "RESTART").unlink()             # the updater clears the handshake
        (g.wd / "RESTART.ack").unlink()

    g._sleep = sleep
    g.workers = ScriptedWorkers([(1.6, 1.0)])
    g.run(max_cycles=1)
    assert slept == [10.0] and len(compute.jobs) > 0   # un-parked, then a cycle ran


def _signed_receipt_doc(king_hk="kingB", ref="v/direct@sha256:" + "b" * 64, dethroned=True):
    return {"status": "scored", "round_id": "77",
            "verdict": {"king_hotkey": king_hk, "dethroned": dethroned},
            "manifest": {"entries": [
                {"miner_hotkey": "kingA", "role": "king", "gen_ref": "v/x@sha256:" + "a" * 64},
                {"miner_hotkey": king_hk, "role": "challenger", "gen_ref": ref}]}}


def test_live_king_follows_the_anchor_receipt_and_is_sticky(cfg, tmp_path):
    from cascade.miner.harness.king import refresh_live_king

    fetched = []

    def fetch(ref, out):
        fetched.append(ref)
        return _gen(out, quality=2.0)

    text = json.dumps(_signed_receipt_doc())
    k = refresh_live_king(cfg, tmp_path, None, receipt_text=lambda: text,
                          verify=lambda t: True, fetch=fetch)
    assert k["hotkey"] == "kingB" and k["digest"] == "b" * 64
    assert (Path(k["dir"]) / "generator.py").is_file() and len(fetched) == 1
    # same king again: no refetch
    assert refresh_live_king(cfg, tmp_path, k, receipt_text=lambda: text,
                             verify=lambda t: True, fetch=fetch) == k and len(fetched) == 1
    # an unverifiable receipt, a dead endpoint or a forfeited king: keep the current one
    assert refresh_live_king(cfg, tmp_path, k, receipt_text=lambda: text,
                             verify=lambda t: False, fetch=fetch) == k
    assert refresh_live_king(cfg, tmp_path, k, receipt_text=lambda: 1 / 0,
                             verify=lambda t: True, fetch=fetch) == k
    forfeit = replace(cfg, scoring=replace(cfg.scoring, forfeit_hotkeys=("kingB",)))
    assert refresh_live_king(forfeit, tmp_path, None, receipt_text=lambda: text,
                             verify=lambda t: True, fetch=fetch) is None


def test_live_king_change_starts_a_new_epoch(rcfg, tmp_path):
    g, _, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    docs = {"doc": _signed_receipt_doc()}
    g.h = replace(g.h, king_source="live")
    g._king_hooks = {"receipt_text": lambda: json.dumps(docs["doc"]),
                     "verify": lambda t: True,
                     "fetch": lambda ref, out: _gen(out, quality=1.0 + ord(ref[-1]) / 1000)}
    assert g.refresh_window(force=True) and g.epoch == 1
    assert g.tree("king") == g.wd / "king" / ("b" * 64)
    assert g.refresh_window(force=True) is False                       # same king
    docs["doc"] = _signed_receipt_doc(king_hk="kingC", ref="v/direct@sha256:" + "c" * 64)
    assert g.refresh_window(force=True) is True and g.epoch == 2       # dethrone → epoch
    assert g.tree("king") == g.wd / "king" / ("c" * 64)


def test_park_on_stop_idles_until_stop_is_deleted(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    (g.wd / "STOP").touch()
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        assert g.state["phase"] == "stopped: delete STOP to resume"
        (g.wd / "STOP").unlink()                 # a person resumes it

    g._sleep = sleep
    g.run(max_cycles=1, park_on_stop=True)
    assert sleeps == [30.0] and compute.jobs     # parked once, then ran a cycle


def test_total_cap_bounds_spend_across_days(tmp_path):
    clock = {"t": 1_790_000_000.0 - 1_790_000_000.0 % 86400 + 3600}
    prov = _FakeProvider()
    ex = _lium(tmp_path, prov, clock, cap=100.0)
    ex.cfg = replace(ex.cfg, total_usd_cap=5.0)
    assert "error" not in ex.run({"kind": "replay", "inputs": {}, "params": {}}, est_hours=1.0)
    clock["t"] += 86400                     # next UTC day: the daily cap resets…
    res = ex.run({"kind": "replay", "inputs": {}, "params": {}}, est_hours=1.0)
    assert res["budget_exhausted"] and "total cap" in res["error"]   # …the total does not


def test_a_machine_that_fails_to_boot_is_never_rented_again(tmp_path):
    clock = {"t": 1_790_000_000.0}

    class Flaky(_FakeProvider):
        def wait_ready(self, name, timeout):
            return len(self.launched) > 1           # first machine never boots

        def machine_of(self, name):
            return f"m{len(self.launched)}"

    prov = Flaky()
    ex = _lium(tmp_path, prov, clock, cap=10.0)
    assert "no pod" in ex.run({"kind": "replay", "inputs": {}, "params": {}})["error"]
    ex.run({"kind": "replay", "inputs": {}, "params": {}})
    assert prov.launched[1].exclude_ids == ("m1",)


def test_receipt_reference_trains_no_king(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (0.8, 1.0)],
                                 stages={"reference": "receipt"})
    assert g.cycle() == "ran"
    king_tree = str(g.tree("king"))
    assert not any(j["inputs"].get("gen") == king_tree for j in compute.jobs
                   if j["kind"] == "replay")                    # zero king training
    assert g.meta("c00001")["status"] == "finalist" and g.meta("c00002")["status"] == "dead"


def test_cached_reference_trains_each_round_king_leg_once(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (1.7, 1.0), (0.8, 1.0)])
    assert g.cycle() == "ran"
    king = str(g.tree("king"))
    legs = [(j["inputs"]["receipt"], j["params"]["train_hours"]) for j in compute.jobs
            if j["kind"] == "replay" and j["inputs"]["gen"] == king]
    assert len(legs) == len(set(legs))                 # never the same (round, budget) twice
    assert g.meta("c00002")["status"] == "finalist"    # the best (1.7) is chosen
    n_king = len(legs)
    g.workers = ScriptedWorkers([(1.8, 1.0), (1.9, 1.0)])
    g.h = hconfig.with_overrides(g.h, search={"proposals_per_cycle": 2})
    assert g.cycle() == "ran"
    legs2 = [j for j in compute.jobs if j["kind"] == "replay" and j["inputs"]["gen"] == king]
    used = {g.meta(c)["stages"]["G2"]["round"] for c in ("c00004", "c00005")}
    first = {g.meta(c)["stages"]["G2"]["round"] for c in ("c00001", "c00002", "c00003")}
    assert len(legs2) - n_king == len(used - first)    # only rounds never screened before


def test_gpu_shortage_waits_instead_of_burning_retries(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    real, calls, sleeps = compute.run, {"n": 0}, []

    def run(spec, **kw):
        calls["n"] += 1
        if calls["n"] <= 4:
            return {"error": "no pod: lium: only 0 × 1xRTX6000 available, need 1",
                    "candidate_fault": False}
        return real(spec, **kw)

    compute.run, g._interruptible_sleep = run, lambda s: sleeps.append(s)
    assert g.cycle() == "ran"
    assert sleeps and g.meta("c00001")["status"] in ("member", "finalist")


def test_worker_image_follows_the_round_contract(tmp_path):
    from cascade.miner.harness.images import resolve_worker_image

    tags = {"worker-v0.9.0": "sha256:" + "4" * 64, "worker-v0.13.0": "sha256:" + "2" * 64}
    cache = tmp_path / "c.json"
    ref = resolve_worker_image("sha256:" + "2" * 64, cache=cache, lister=lambda r: tags)
    assert ref == "ghcr.io/tensorlink-ai/cascade-worker:worker-v0.13.0@sha256:" + "2" * 64
    # cached: no registry call the second time
    assert resolve_worker_image("2" * 64, cache=cache, lister=lambda r: 1 / 0) == ref
    assert resolve_worker_image("sha256:" + "9" * 64, cache=cache, lister=lambda r: tags) is None


def test_set_image_retires_pods_on_the_old_image(tmp_path):
    clock = {"t": 1_790_000_000.0}
    prov = _FakeProvider()
    ex = _lium(tmp_path, prov, clock, cap=10.0)
    ex.run({"kind": "replay", "inputs": {}, "params": {}})
    assert len(ex._pods) == 1
    ex.set_image("ghcr.io/x/w:new@sha256:" + "b" * 64)
    assert ex._pods == {} and prov.terminated                   # idle old-image pod retired
    ex.run({"kind": "replay", "inputs": {}, "params": {}})
    assert prov.launched[-1].image.endswith("b" * 64)


def test_live_contract_overrides_local_billing(cfg):
    from cascade.miner.live_contract import with_live_contract
    from cascade.shared.manifest import contract_digest, contract_payload, dump_manifest

    from .receipt_fixture import make_manifest

    body = {**contract_payload(cfg.training), "budget_denomination": "points+mv20"}
    m = replace(make_manifest(cfg, base_seed=1), contract_body=body,
                contract_digest=contract_digest(body))
    doc = json.loads(dump_manifest(m))
    live, info = with_live_contract(cfg, fetch=lambda: doc)
    assert info["source"] == "live" and live.training.budget_denomination == "points+mv20"
    assert live.screen_contract().budget_denomination == "points+mv20"
    bad = {**doc, "contract_digest": "0" * 64}
    same, info = with_live_contract(cfg, fetch=lambda: bad)
    assert info["source"] == "local" and same is cfg
    offline, info = with_live_contract(cfg, fetch=lambda: 1 / 0)
    assert info["source"] == "local" and offline is cfg


def test_ledger_bills_the_real_price(tmp_path):
    clock = {"t": 1_790_000_000.0 - 1_790_000_000.0 % 86400 + 3600}

    class Priced(_FakeProvider):
        def price_of(self, name):
            return 0.70

    ex = _lium(tmp_path, Priced(), clock, cap=10.0)     # cap price 2.0/h in _lium
    ex.run({"kind": "replay", "inputs": {}, "params": {}})
    clock["t"] += 3600
    assert ex.ledger.spent_today() == pytest.approx(0.70, rel=1e-3)


def test_orphaned_proposals_are_retired(rcfg, tmp_path):
    g, _, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    g.refresh_window(force=True)
    g.propose(1)                                        # judge "dies" before the workers report
    assert g.meta("c00001")["status"] == "proposed"
    g.workers = ScriptedWorkers([(0.8, 1.0)])
    g.cycle()
    assert g.meta("c00001")["status"] == "dead"
    assert "orphaned" in g.meta("c00001")["reason"]


def test_dethrone_progress_is_capped_by_the_deepest_stage(rcfg, tmp_path):
    g, _, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (0.8, 1.0)])
    cp = g.candidate_progress
    assert cp({"stages": {"G2": {"pass": True, "rel": 0.05}}})["score"] == 40.0   # cheap: capped
    assert cp({"stages": {"G2": {"pass": True, "rel": 0.05},
                          "G3": {"pass": True, "rel": 0.005}}})["score"] == 35.0
    assert cp({"stages": {"G4.5": {"pass": True, "rel": 0.02}}})["score"] == 100.0
    assert cp({"stages": {"G2": {"pass": False, "rel": 0.05}}})["score"] == 0.0
    assert g.cycle() == "ran"
    pr = g.state["progress"]
    assert pr["id"] == "c00001" and pr["stage"] == "G4.5" and pr["score"] == 100.0
    assert (g.wd / "progress.jsonl").read_text().count("\n") == 3     # after G2, G3, cycle
    from cascade.miner.harness.gauntlet import progress_bar
    bar = progress_bar(pr)
    assert bar.startswith("dethrone [" + "█" * 30 + "] 100/100") and "c00001" in bar
    assert progress_bar({"score": 23.4, "id": None}).count("█") == 7


def test_progress_ignores_candidates_rejected_later(rcfg, tmp_path):
    g, _, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    g.refresh_window(force=True)
    g.put({"id": "c00090", "epoch": g.epoch, "status": "dead", "reason": "G3: did not confirm",
           "stages": {"G2": {"pass": True, "rel": 0.009}, "G3": {"pass": False, "rel": -0.001}}})
    assert g.progress()["score"] == 0.0                 # passed G2, failed G3: counts nothing
    g.put({"id": "c00091", "epoch": g.epoch, "status": "in_gauntlet",
           "stages": {"G2": {"pass": True, "rel": 0.005}}})
    assert g.progress() == {**g.progress(), "id": "c00091", "score": 20.0, "stage": "G2"}


def test_knowledge_folder_holds_briefs_and_attempts_not_the_prompt(rcfg, tmp_path):
    g, _, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0), (0.8, 1.0)])
    op = g.wd / "operator"
    (op / "LINEAGE.md").write_text("lineage: try X\n")
    (op / "DETHRONES.md").write_text("dethrones: Y won on energy\n")
    (op / "RESEARCH.md").write_text("research: " + "z" * 20000)
    assert g.cycle() == "ran"
    g.put({"id": "c00099", "epoch": g.epoch, "status": "dead", "note": "GBFS fork",
           "reason": "G1: runtime verify: NameError: name 'rng2' is not defined",
           "stages": {"G0": {"pass": True}}})
    p = g.propose(1)[0]
    k = p.knowledge
    assert (k / "LINEAGE.md").read_text() == "lineage: try X\n" and (k / "RESEARCH.md").is_file()
    rows = [json.loads(x) for x in (k / "attempts.jsonl").read_text().splitlines()]
    dead = next(r for r in rows if r["id"] == "c00099")
    assert "NameError: name 'rng2'" in dead["reason"] and dead["furthest_stage"] == "G0"
    win = next(r for r in rows if r["id"] == "c00001")
    assert "G4.5" in win["passed"] and win["screen_rel"] is not None
    assert "lcb" not in json.dumps(rows) and "rels" not in json.dumps(rows)  # G3+ stays pass/fail
    assert "z" * 100 not in p.prompt and len(p.prompt) < 8000          # briefs are not inlined
    assert "{KNOWLEDGE_DIR}/attempts.jsonl" in p.prompt and "NameError: name 'rng2'" in p.prompt


def test_queue_ships_the_knowledge_next_to_the_tree(tmp_path):
    q = tmp_path / "queue"
    k = tmp_path / "know"
    k.mkdir()
    (k / "attempts.jsonl").write_text('{"id": "c1"}\n')
    tree = _gen(tmp_path / "cands" / "c2" / "tree")
    seen = {}

    def runner(p, env):
        seen["k"] = (p.knowledge / "attempts.jsonl").read_text()
        seen["inside"] = (p.tree / "attempts.jsonl").exists()
        return _editing_runner(p, env)

    judge = QueueWorkers(q, timeout=30, poll=0.05)
    out: list = []
    t = threading.Thread(target=lambda: out.extend(
        judge.run([Proposal("c2", "king", tree, "p", knowledge=k)])))
    t.start()
    while not any((q / "pending").iterdir()):
        pass
    serve_queue(q, _wcfg(), home=tmp_path / "wh", once=True, runner=runner)
    t.join(10)
    assert out[0].ok and seen == {"k": '{"id": "c1"}\n', "inside": False}
    assert not (tree / "knowledge").exists()                           # never shipped


def test_running_jobs_reserve_their_cost_so_parallel_jobs_cannot_overshoot(tmp_path):
    clock = {"t": 1_790_000_000.0 - 1_790_000_000.0 % 86400 + 3600}
    ex = _lium(tmp_path, _FakeProvider(), clock, cap=10.0, par=4)   # $2/h cap price
    # a 2 h job reserves min(timeout, 2*1.25+0.5)=3 h x $2 = $6; a second can't fit
    pod = ex._acquire(2.0)
    assert ex._committed == pytest.approx(6.0)
    with pytest.raises(Exception, match="reserved"):
        ex._acquire(2.0)
    ex._release(pod)
    assert ex._committed == 0.0
    assert ex._acquire(2.0)                       # fits again once the first is done


def test_pods_get_the_harness_chain_toml(tmp_path):
    clock = {"t": 1_790_000_000.0}
    ex = _lium(tmp_path, _FakeProvider(), clock, cap=10.0)
    toml = tmp_path / "my-chain.toml"
    toml.write_text("x")
    ex.chain_toml = toml
    ex.run({"kind": "replay", "inputs": {}, "params": {}})
    assert ex.transfer.pushed[-2].endswith("harness-chain.toml") or any(
        d.endswith("harness-chain.toml") for d in ex.transfer.pushed)
    spec = json.loads(next((tmp_path / "wd" / "jobs").glob("*/spec.remote.json")).read_text())
    assert spec["params"]["chain_toml"].endswith("harness-chain.toml")


def test_a_capped_batch_keeps_its_paid_results_and_g4_legs(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    real = compute.run
    state = {"cap": True}

    def run(spec, **kw):
        if (spec["kind"] == "replay" and spec["params"]["train_hours"] is None
                and state["cap"] and spec["inputs"]["receipt"].endswith(
                    g.window.newest(1)[0].receipt_path.name)):
            return {"error": "daily cap", "candidate_fault": False, "budget_exhausted": True}
        return real(spec, **kw)

    compute.run = run
    assert g.cycle() == "budget"                         # the newest G4 leg hit the cap
    legs = g.meta("c00001")["g4_legs"]
    assert len(legs) == 2                                # the two paid legs were kept
    assert g.meta("c00001")["status"] == "member"
    n = len(compute.jobs)
    state["cap"] = False
    g.workers = ScriptedWorkers([])
    g.h = hconfig.with_overrides(g.h, search={"proposals_per_cycle": 0})
    assert g.cycle() == "ran"
    g4_jobs = [j for j in compute.jobs[n:] if j["kind"] == "replay"
               and j["params"]["train_hours"] is None]
    assert len(g4_jobs) == 1                             # only the missing leg re-ran
    assert g.meta("c00001")["status"] == "finalist"


def test_a_finalist_is_never_offered_twice(tmp_path, monkeypatch):
    s, calls = _sub(tmp_path, mode="approval")
    assert s.offer("c1", _gen(tmp_path / "a"), GOOD)["action"] == "pending"
    assert s.offer("c1", _gen(tmp_path / "a"), GOOD) == {"action": "pending", "already": True}
    s.reject("c1")
    assert s.offer("c1", _gen(tmp_path / "a"), GOOD)["action"] == "already"


def test_string_for_a_list_setting_is_refused(tmp_path):
    p = tmp_path / "h.toml"
    p.write_text('[submit]\nhotkeys = "5Fabc"\n')
    with pytest.raises(ValueError, match="must be a list"):
        hconfig.load_harness_config(p)


def test_unreplayable_rounds_never_enter_the_window(rcfg, tmp_path):
    g, compute, _, _ = _gauntlet(rcfg, tmp_path, script=[(1.6, 1.0)])
    g.refresh_window(force=True)
    first = g.window.all[0].round_id
    g2, compute2, _, _ = _gauntlet(rcfg, tmp_path / "b", script=[(1.6, 1.0)])
    # tamper one cached receipt's window ids: its draw no longer reproduces
    g2.refresh_window(force=True)
    target = next(p for p in (g2.wd / "receipts").glob("*.json") if p.stem == first)
    doc = json.loads(target.read_text())
    doc["eval_context"]["window_ids"] = list(reversed(doc["eval_context"]["window_ids"]))
    target.write_text(json.dumps(doc))
    g2.state["round_ok"] = {}
    g2.refresh_window(force=True)
    assert first not in [r.round_id for r in g2.window.all]
    assert g2.state["round_ok"][first] is False


def test_a_refused_rental_fails_fast(tmp_path):
    clock = {"t": 1_790_000_000.0}

    class Refused(_FakeProvider):
        calls = 0

        def wait_ready(self, name, timeout):
            Refused.calls += 1
            clock["t"] += timeout
            return False

        def _up_log_tail(self, name):
            return '{"status_code":400,"message":"Another rental is already in progress"}'

    ex = _lium(tmp_path, Refused(), clock, cap=10.0)
    res = ex.run({"kind": "replay", "inputs": {}, "params": {}})
    assert "refused the rental" in res["error"] and Refused.calls == 1   # one poll, not 30 min
