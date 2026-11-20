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
