"""WB27 follow-up (owner sweep 2026-11-27): the VALIDATOR never loads lane B
checkpoint weights in its own process (a 7 GB CPU box; a model load
OOM-killed every service on 2026-10-31). Its weight-level lane B checks — the
judge's stage ``merge`` (``merge_check_dirs``), the merged-init rebuild
(``_verify_merged_init``) and each acceptance step's rebuild
(``verify_acceptance``) — run on its ``--eval-hosts`` eval host through the
trainer's ``LaneBPodRunner`` (ops ``merge_check`` / ``merge_verify``), and
fail closed as an INFRA transient (never a verdict, nothing burnt) when no
host is configured.

Every test runs a real ``ValidatorRunner`` with ``torch`` and ``safetensors``
POISONED (``sys.modules[...] = None``: any import raises) and a fake remote;
it asserts no local fetch or load happened and that the remote got the right
arguments. Torch-free: safe on the 7 GB box."""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from types import SimpleNamespace

import pytest

GEN = "alice/metro-gen@sha256:" + "a" * 64
POISONED = ("torch", "safetensors", "safetensors.torch", "cascade.trainer.moe_model")


def _ptr(tag: str) -> str:
    from cascade.shared.manifest import format_trained_pointer

    return format_trained_pointer(
        f"cascade/ckpt-{tag}@sha256:" + hashlib.sha256(tag.encode()).hexdigest())


KING, CAND, BASE, MERGED = _ptr("king"), _ptr("cand"), _ptr("member"), _ptr("merged")
SEL = {0: [3, 5]}


@pytest.fixture(autouse=True)
def poisoned(monkeypatch):
    """No torch / safetensors in this process, and no local opt-in."""
    for name in POISONED:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delenv("CASCADE_LANE_B_LOCAL_EVAL", raising=False)


class FakeRemote:
    """The eval host: records every call, answers what the test scripted."""

    def __init__(self, *, merge_check=None, merge_verify=None):
        self.calls: list[tuple] = []
        self._mc, self._mv = merge_check, merge_verify or {"equal": True}

    def merge_check(self, king_pointer, cand_pointer, selection, strict):
        self.calls.append(("merge_check", king_pointer, cand_pointer, selection, strict))
        return self._mc

    def merge_verify(self, base_pointer, merges, target_pointer, *, strict):
        self.calls.append(("merge_verify", base_pointer, list(merges), target_pointer, strict))
        return self._mv


def _runner(cfg, tmp_path, *, remote=None, lane_b=None):
    from cascade.shared.config import parse_lane_b
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    lb = parse_lane_b(lane_b or {"from_block": 1, "domains": ["energy", "web"]})
    v = dataclasses.replace(cfg.validator, state_db_path=str(tmp_path / "state.json"))
    r = ValidatorRunner(cfg=dataclasses.replace(cfg, lane_b=lb, validator=v),
                        state=genesis("K", 0), verify_signatures=False)
    r.lane_b_merge_ops = remote          # the injected eval host (None: the real wiring)
    fetched: list[str] = []

    def _no_local_fetch(entry):
        fetched.append(entry.trained_pointer)
        return tmp_path

    r._fetch_checkpoint_dir = _no_local_fetch
    r.fetched = fetched
    return r


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001 — the test asserts on the outcome
        return ("raised", type(e).__name__, str(e))


def _no_local_load(r):
    assert r.fetched == [], f"the validator fetched checkpoints locally: {r.fetched}"
    for name in POISONED:
        assert sys.modules.get(name) is None, f"{name} was imported into the validator"


# ── the judge (stage ``merge``) ──────────────────────────────────────────────


def _settlement(r):
    from cascade.shared.expert_manifest import ExpertEntry, ExpertManifest, expert_manifest_key
    from cascade.shared.manifest import TrainedEntry, TrainingManifest

    em = ExpertManifest(
        round_id="1001", created_block=9000, king_pointer=KING, lineage_tokens=1,
        entries=(ExpertEntry(miner_hotkey="alice", miner_uid=7, gen_ref=GEN, domain="energy",
                             selection=SEL, exclude={}, base_king_pointer=KING,
                             rebased_pointer=CAND, corpus_digest="c", train_block=8999,
                             commit_block=8990),))
    r._lane_b_store = SimpleNamespace(get_text=lambda key: {
        expert_manifest_key("1001"): em.to_json()}[key])
    r._lane_b_window_source = object()
    r._verdict_windows = lambda *a, **k: []
    r._evaluate = lambda *a, **k: []
    r._lane_b_leg_base_contract = lambda block: None
    r._lane_b_sentinel_kw = lambda manifest: {}
    return TrainingManifest(
        round_id="1001", created_block=9000, contract_digest="1" * 64,
        base_arch_digest="2" * 64, eval_dataset="d",
        entries=[TrainedEntry("K", 0, "king", GEN, KING, "3" * 64, 8999)],
        expert_manifest_sha256=em.digest())


@pytest.mark.parametrize("guard", [False, True])
def test_judge_merge_stage_runs_on_the_eval_host(cfg, tmp_path, guard):
    """The pod's refusal is the verdict, unchanged; the remote got the king,
    the candidate, the entry's rows and the settlement's merge guard."""
    lane_b = {"from_block": 1, "domains": ["energy", "web"],
              **({"merge_guard_from_block": 1} if guard else {})}
    remote = FakeRemote(merge_check="MergeError: blocks.0.ffn.w_in differs off its rows")
    r = _runner(cfg, tmp_path, remote=remote, lane_b=lane_b)
    m = _settlement(r)
    out = _outcome(lambda: r._lane_b_judge_experts(m))
    assert remote.calls == [("merge_check", KING, CAND, SEL, guard)], out
    assert out[0] == "ok", out
    (j,) = out[1].judgements
    assert (j.passed, j.stage, j.reason) == (
        False, "merge", "MergeError: blocks.0.ffn.w_in differs off its rows")
    _no_local_load(r)


def test_judge_without_an_eval_host_fails_closed_as_infra(cfg, tmp_path):
    """No ``--eval-hosts``: a LaneBTransient naming the missing host — the
    settlement retries; no judgement exists, so nothing is refused or burnt."""
    from cascade.validator.lane_b import LaneBTransient

    r = _runner(cfg, tmp_path)
    assert r.eval_host_fn is None
    m = _settlement(r)
    out = _outcome(lambda: r._lane_b_judge_experts(m))
    assert out[0] == "raised" and out[1] == LaneBTransient.__name__, (
        f"lane B judged without an eval host (a local load or a miner refusal): {out}")
    assert "no eval pod configured" in out[2] and "--eval-hosts" in out[2], out
    assert "not a miner fault" in out[2], out
    _no_local_load(r)


def test_judge_routes_to_the_configured_eval_host(cfg, tmp_path):
    """Production wiring: the runner's ``eval_host_fn`` (``--eval-hosts``) is
    the host the pod runner dials, with the ``merge_check`` op and args."""
    from cascade.trainer import lane_b_offload as off

    host = SimpleNamespace(name="valeval-blackwell4")
    seen: list = []

    def fake_call(self, op, pointer, request, contexts, payload=None, timeout_s=None,
                  host=None):
        seen.append((self.host().name, op, pointer, request, payload))
        return {"merge_error": "MergeError: refused on the pod"}

    r = _runner(cfg, tmp_path)
    r.eval_host_fn = lambda: host
    m = _settlement(r)
    import unittest.mock as mock

    with mock.patch.object(off.LaneBPodRunner, "_call", fake_call):
        out = _outcome(lambda: r._lane_b_judge_experts(m))
    assert seen == [("valeval-blackwell4", "merge_check", KING,
                     {"pointers": [KING, CAND]}, {"selection": SEL, "strict": False})], out
    assert out[0] == "ok" and out[1].judgements[0].reason == "MergeError: refused on the pod"
    _no_local_load(r)


# ── the merged era init (``_verify_merged_init``) ────────────────────────────

MERGE = {"rebased_pointer": CAND, "king_pointer": KING, "selection": {"0": [3, 5]},
         "round_id": "1001"}


def _era_manifest():
    from cascade.shared.manifest import TrainedEntry, TrainingManifest

    return TrainingManifest(
        round_id="2002", created_block=20000, contract_digest="1" * 64,
        base_arch_digest="2" * 64, eval_dataset="d",
        entries=[TrainedEntry("K", 0, "king", GEN, KING, "3" * 64, 19999)],
        warm_start_ckpt=MERGED, warm_start_base=BASE, warm_start_merges=(MERGE,))


def _recorded(r):
    r._lane_b = lambda: SimpleNamespace(has_merge=lambda *a: True, qualified_for=lambda m: [])
    return r


@pytest.mark.parametrize("reply,want", [
    ({"equal": True}, ("ok", None)),
    ({"equal": False}, ("ok", "era_init_mismatch: merged init bytes differ from member + "
                              "recorded merges")),
    ({"merge_error": "lane-B merges overlap on MoE layer 0 expert row 3"},
     ("ok", "era_init_mismatch: lane-B merges overlap on MoE layer 0 expert row 3")),
])
@pytest.mark.parametrize("strict", [False, True])
def test_merged_init_rebuild_runs_on_the_eval_host(cfg, tmp_path, reply, want, strict):
    remote = FakeRemote(merge_verify=reply)
    r = _recorded(_runner(cfg, tmp_path, remote=remote))
    out = _outcome(lambda: r._verify_merged_init(_era_manifest(), strict=strict))
    assert remote.calls == [("merge_verify", BASE, [MERGE], MERGED, strict)], out
    assert out == want
    _no_local_load(r)


def test_merged_init_value_error_stays_transient(cfg, tmp_path):
    """A non-MergeError ValueError was a transient locally; it still is."""
    remote = FakeRemote(merge_verify={"value_error": "bad header"})
    r = _recorded(_runner(cfg, tmp_path, remote=remote))
    out = _outcome(lambda: r._verify_merged_init(_era_manifest()))
    assert out[:2] == ("raised", "StorageError") and "era_init_unverifiable" in out[2], out
    assert len(remote.calls) == 1
    _no_local_load(r)


def test_merged_init_without_an_eval_host_is_unverifiable_not_rejected(cfg, tmp_path):
    r = _recorded(_runner(cfg, tmp_path))
    out = _outcome(lambda: r._verify_merged_init(_era_manifest()))
    assert out[:2] == ("raised", "StorageError"), (
        f"the merged init was judged without an eval host: {out}")
    assert "era_init_unverifiable" in out[2] and "--eval-hosts" in out[2], out
    _no_local_load(r)


# ── merge acceptance (``verify_acceptance``) ─────────────────────────────────


def _acceptance(cfg, tmp_path, monkeypatch, remote):
    from cascade.shared.lane_b import DomainWin
    from cascade.validator import expert_lane as vel
    from cascade.validator import merge_acceptance as ma

    block = 20000
    monkeypatch.setattr(ma, "acceptance_block", lambda cfg, manifest, runner=None: block)
    monkeypatch.setattr(vel, "verdict_windows", lambda *a, **k: [])
    win = DomainWin("energy", "alice", 7, "1001", 9000, GEN, 0.1, CAND, KING,
                    {"0": [3, 5]}, commit_block=8990, targets=("energy",),
                    qualified_block=9000)
    merge = dict(MERGE, qualified_wins=[dataclasses.asdict(win)],
                 acceptance={"block": block, "before": BASE, "candidate": MERGED})
    r = _runner(cfg, tmp_path, remote=remote,
                lane_b={"from_block": 1, "domains": ["energy", "web"],
                        "acceptance_from_block": 1})
    r._lane_b = lambda: SimpleNamespace(accepted={}, qualified_for=lambda m: [win])
    r._lane_b_window_source = object()
    r._evaluate = lambda *a, **k: []
    manifest = dataclasses.replace(_era_manifest(), warm_start_merges=(merge,))
    return r, manifest, merge, ma


def test_acceptance_rebuild_runs_on_the_eval_host(cfg, tmp_path, monkeypatch):
    remote = FakeRemote(merge_verify={"equal": False})
    r, manifest, merge, ma = _acceptance(cfg, tmp_path, monkeypatch, remote)
    out = _outcome(lambda: ma.verify_acceptance(r, manifest))
    assert remote.calls == [("merge_verify", BASE, [merge], MERGED, True)], out
    assert out == ("raised", "ValueError",
                   "acceptance intermediate checkpoint differs from its deltas")
    _no_local_load(r)


def test_acceptance_merge_refusal_keeps_its_message(cfg, tmp_path, monkeypatch):
    remote = FakeRemote(merge_verify={"merge_error": "x: non-finite values on the merge's "
                                                     "selected expert rows"})
    r, manifest, _, ma = _acceptance(cfg, tmp_path, monkeypatch, remote)
    out = _outcome(lambda: ma.verify_acceptance(r, manifest))
    assert out == ("raised", "ValueError",
                   "x: non-finite values on the merge's selected expert rows"), out
    _no_local_load(r)


def test_acceptance_without_an_eval_host_is_transient(cfg, tmp_path, monkeypatch):
    r, manifest, _, ma = _acceptance(cfg, tmp_path, monkeypatch, None)
    out = _outcome(lambda: ma.verify_acceptance(r, manifest))
    assert out[:2] == ("raised", "StorageError") and "--eval-hosts" in out[2], (
        f"acceptance was judged without an eval host: {out}")
    _no_local_load(r)


# ── the pod op itself (box half; the pod half is CI-only torch) ──────────────


def test_pod_runner_merge_verify_request_and_reply_shape():
    from cascade.trainer import lane_b_offload as off

    seen = []

    def fake_call(op, pointer, request, contexts, payload=None, timeout_s=None, host=None):
        seen.append((op, pointer, request, contexts, payload, timeout_s))
        return replies.pop(0)

    pod = off.LaneBPodRunner(lambda: SimpleNamespace(name="h"))
    pod._call = fake_call
    assert callable(getattr(pod, "merge_verify", None)), "LaneBPodRunner has no merge_verify op"
    replies = [{"equal": True}, {"merge_error": "m"}, {"value_error": "v"},
               {"equal": 1}, {"equal": True, "merge_error": "m"}, ["x"]]
    assert pod.merge_verify(BASE, [MERGE], MERGED, strict=True) == {"equal": True}
    assert seen == [("merge_verify", BASE, {"base": BASE, "merges": [MERGE],
                                            "target": MERGED, "strict": True},
                     None, None, off.MODEL_TIMEOUT_S)]
    assert pod.merge_verify(BASE, [], MERGED, strict=False) == {"merge_error": "m"}
    assert pod.merge_verify(BASE, [], MERGED, strict=False) == {"value_error": "v"}
    for _ in range(3):
        with pytest.raises(off.LaneBOffloadUnavailable, match="malformed"):
            pod.merge_verify(BASE, [], MERGED, strict=False)
