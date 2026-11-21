"""The deploy gate (2026-11-16 review of deploys that cascaded to validators and
miners): blast-radius classification and the timing / in-flight guards."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from scripts import deploy_gate as g

NOW = datetime(2026, 11, 16, 3, 0, tzinfo=UTC)          # a Monday


def _vline(ts: str, text: str) -> str:
    return f"2026-11-16 {ts},000 INFO cascade.validator: {text}"


def _gate(tmp_path, *, service="trainer", paths=(), diff="", vlines=(), now=NOW, acks=(),
          on_main=True):
    return g.gate(service=service, areas=g.classify(list(paths)), diff_text=diff, now=now,
                  trainer_lines=[], validator_lines=list(vlines), work=tmp_path,
                  on_main=on_main, acks=set(acks))


# 03:00 with block 9516100 sampled at 02:59 → next boundary 9516600 ≈ 1 h 40 min away
CLOCK = [_vline("02:59:00", "re-asserting weights (last set 3 blocks ago, block 9516100)")]


def test_classification():
    a = g.classify(["cascade/eval/koth.py", "cascade/shared/era.py", "cascade/trainer/x.py",
                    "deploy/Dockerfile", "cascade/website/index.html", "chain.toml",
                    "cascade/validator/eval_offload.py"])
    assert set(a["consensus"]) == {"cascade/eval/koth.py", "cascade/shared/era.py"}
    assert a["shared"] == ["cascade/shared/era.py"] and "worker_image" in a
    assert "website" in a and "chain_toml" in a
    assert a["validator"] == ["cascade/validator/eval_offload.py"]
    assert "consensus" not in g.classify(["cascade/validator/eval_offload.py"])


def test_ungated_consensus_change_is_refused_and_cannot_be_acked(tmp_path):
    v = _gate(tmp_path, paths=["cascade/eval/koth.py"], diff="+    x = 1\n", vlines=CLOCK,
              acks={"consensus"})
    assert [r for r, _ in v.refuse] == ["consensus"]
    gated = _gate(tmp_path, paths=["cascade/eval/koth.py"],
                  diff="+    margin_v3_from_block: int = 0\n", vlines=CLOCK)
    assert not gated.refuse and any("activation key" in n for n in gated.notes)


def test_off_main_and_freeze_window(tmp_path):
    assert ("provenance" in [r for r, _ in _gate(tmp_path, vlines=CLOCK, on_main=False).refuse])
    wed = datetime(2026, 11, 18, 23, 0, tzinfo=UTC)
    thu = datetime(2026, 11, 19, 7, 59, tzinfo=UTC)
    assert g.freeze_window(wed) and g.freeze_window(thu)
    assert not g.freeze_window(datetime(2026, 11, 19, 8, 0, tzinfo=UTC))


def test_settlement_window(tmp_path):
    assert not _gate(tmp_path, vlines=CLOCK).refuse
    near = [_vline("02:59:00", "x block 9516510")]          # ~18 min before 9516600
    assert [r for r, _ in _gate(tmp_path, vlines=near).refuse] == ["settlement"]
    after = [_vline("02:59:00", "x block 9516640")]          # ~8 min after 9516600
    assert [r for r, _ in _gate(tmp_path, vlines=after).refuse] == ["settlement"]
    assert [r for r, _ in _gate(tmp_path, vlines=near, acks={"settlement"}).refuse] == []
    assert [r for r, _ in _gate(tmp_path).refuse] == ["clock"]


def test_validator_scoring_window(tmp_path):
    scoring = CLOCK + [_vline("02:59:30", "new manifest round=42 entries=2 (king:uid1); …")]
    assert [r for r, _ in _gate(tmp_path, service="validator", vlines=scoring).refuse] == [
        "scoring"]
    done = scoring + [_vline("02:59:50", "round=42 weights set under the X policy {}: {}")]
    assert not _gate(tmp_path, service="validator", vlines=done).refuse
    # the trainer is not held by the validator's scoring
    assert not _gate(tmp_path, service="trainer", vlines=scoring).refuse


def test_trainer_refused_while_work_holds_pods(tmp_path):
    q = {"entries": {"a": {"status": "in_flight", "phase": "waiting_gpu", "hotkey": "5Aaaa"},
                     "b": {"status": "done", "phase": "training", "hotkey": "5Bbbb"}}}
    (tmp_path / "funded_queue.json").write_text(json.dumps(q))
    assert not _gate(tmp_path, vlines=CLOCK).refuse            # waiting for a GPU: no pod
    q["entries"]["a"]["phase"] = "training"
    (tmp_path / "funded_queue.json").write_text(json.dumps(q))
    v = _gate(tmp_path, vlines=CLOCK)
    assert [r for r, _ in v.refuse] == ["in_flight"] and "5Aaaa" in v.refuse[0][1]
    (tmp_path / "funded_queue.json").write_text(json.dumps({"entries": {}}))
    (tmp_path / "era_state.json").write_text(json.dumps({"rents": [{"pod": "k"}],
                                                         "pending_publish": None}))
    assert [r for r, _ in _gate(tmp_path, vlines=CLOCK).refuse] == ["in_flight"]
    # the validator is not held by trainer legs
    assert not _gate(tmp_path, service="validator", vlines=CLOCK).refuse


def test_new_services_scoring_and_notes(tmp_path):
    scoring = CLOCK + [_vline("02:59:30", "new manifest round=42 entries=2 (king:uid1)")]
    for svc in ("intake-laned", "eval-host"):
        assert [r for r, _ in _gate(tmp_path, service=svc, vlines=scoring).refuse] == ["scoring"]
        assert not _gate(tmp_path, service=svc, vlines=CLOCK).refuse
    lane_d = _gate(tmp_path, service="trainer", paths=["cascade/lane_d/intake.py"], vlines=CLOCK)
    assert any("intake-laned" in n for n in lane_d.notes)
    own = _gate(tmp_path, service="intake-laned", paths=["cascade/lane_d/intake.py"], vlines=CLOCK)
    assert not any("deploy the intake-laned" in n for n in own.notes)
    ev = _gate(tmp_path, service="validator", paths=["cascade/validator/private_worker.py"],
               vlines=CLOCK)
    assert any("eval-host" in n for n in ev.notes)
    # the eval-host sync is a validator-deploy step, not a trainer one
    tr = _gate(tmp_path, service="trainer", paths=["cascade/validator/private_worker.py"],
               vlines=CLOCK)
    assert not any("eval-host" in n for n in tr.notes)


def test_every_area_and_new_trees_classify():
    a = g.classify(["cascade/lane_d/x.py", "cascade/laned/dedrift.py", "scripts/sync_king.py",
                    "deploy/pod_bootstrap.sh", "benchmarks/Dockerfile", "base_generator/generator.py",
                    "cascade/miner/cli.py", "cascade/pool/builder.py", "champions/king/x.json",
                    ".github/workflows/ci.yml", "decided_votes.toml", ".python-version",
                    ".dockerignore"])
    assert a["lane_d_intake"] == ["cascade/lane_d/x.py"]
    assert a["ops_tooling"] == ["cascade/laned/dedrift.py", "scripts/sync_king.py"]
    assert a["deploy"] == ["deploy/pod_bootstrap.sh"] and a["benchmarks"] == ["benchmarks/Dockerfile"]
    assert a["miner_kit"] == ["base_generator/generator.py", "cascade/miner/cli.py"]
    assert a["pool"] == ["cascade/pool/builder.py"] and a["champions"] == ["champions/king/x.json"]
    assert a["ci"] == [".github/workflows/ci.yml"] and a["chain_toml"] == ["decided_votes.toml"]
    assert a["eval_host"] == ["cascade/miner/cli.py", "cascade/pool/builder.py", ".python-version"]
    assert ".dockerignore" in a["worker_image"] and "cascade/lane_d/x.py" not in a["worker_image"]
    # non-Python files under on-pod packages are not image code unless listed
    assert "worker_image" not in g.classify(["cascade/website/index.html"])


def _log(tmp_path, *lines):
    p = tmp_path / "validator.log"
    p.write_text("".join(f"{ln}\n" for ln in ("head",) + lines))
    return p


def test_verify_eval_host_from_validator_log(tmp_path):
    since = datetime(2026, 11, 16, 2, 0, tzinfo=UTC)
    up = _vline("02:30:00", "private eval: uploaded checkpoint abcd (0.1 GB) to valpriv in 9s")
    hit = _vline("02:30:00", "private eval: checkpoint abcd already on valpriv (sha256 verified)")
    old = _vline("01:30:00", "private eval: uploaded checkpoint abcd (0.1 GB) to valpriv in 9s")
    bad = _vline("02:40:00", "private eval host valpriv runs code older than this validator")
    assert g.verify("eval-host", tmp_path, _log(tmp_path, up), since=since) == []
    assert g.verify("eval-host", tmp_path, _log(tmp_path, hit), since=since) == []
    assert "not verified yet" in g.verify("eval-host", tmp_path, _log(tmp_path, old),
                                          since=since)[0]
    assert "failure" in g.verify("eval-host", tmp_path, _log(tmp_path, up, bad), since=since)[0]
    assert g.verify("eval-host", tmp_path, tmp_path / "missing.log", since=since)


def test_intake_process_match():
    assert g._runs_module("intake-laned", [b"python", b"-P", b"-m", b"cascade.lane_d.intake"])
    assert g._runs_module("intake-laned", [b"python", b"-m", b"cascade.lane_d"])
    assert not g._runs_module("intake-laned", [b"python", b"-P", b"-m", b"cascade.lane_dx"])
    assert not g._runs_module("intake-laned", [b"python", b"-P", b"-m"])
    assert not g._runs_module("validator", [b"python", b"-P", b"-m", b"cascade.validator.main"])
