"""One GPU type per manifest: the validators' ``_check_gpu`` rejects a
manifest whose entries sit on different GPUs, so the trainer keeps the
king's type and requeues the rest unburned (2026-09-20: king + 5 on RTX
4090, 8 on L40S — a manifest nobody would have scored)."""

from __future__ import annotations

from cascade.shared.manifest import TrainedEntry, format_trained_pointer
from cascade.trainer.loop import TrainerRunner, _split_mixed_gpu_entries
from tests.unit.test_funded_pod_wiring import REF, _runner

TP = format_trained_pointer(REF)


def _e(hotkey, gpu, role="challenger", size="s1", uid=1):
    return TrainedEntry(miner_hotkey=hotkey, miner_uid=uid, role=role, gen_ref=REF,
                        trained_pointer=TP, corpus_digest="d", train_block=1, size=size,
                        gpu_name=gpu)


def test_keeps_the_kings_type_and_drops_the_rest():
    entries = [_e("king", "NVIDIA GeForce RTX 4090", role="king"),
               _e("a", "NVIDIA GeForce RTX 4090"), _e("b", "NVIDIA L40S"),
               _e("c", ""), _e("d", "NVIDIA L40S")]
    kept, dropped = _split_mixed_gpu_entries(entries)
    assert [e.miner_hotkey for e in kept] == ["king", "a", "c"]      # no gpu_name passes
    assert [e.miner_hotkey for e in dropped] == ["b", "d"]


def test_type_is_judged_per_size_like_the_validator():
    entries = [_e("king", "NVIDIA L40S", role="king", size="s1"),
               _e("king", "NVIDIA H100", role="king", size="s2"),
               _e("a", "NVIDIA L40S", size="s1"), _e("a", "NVIDIA L40S", size="s2"),
               _e("b", "NVIDIA H100", size="s2"), _e("z", "NVIDIA H100", size="s9")]
    kept, dropped = _split_mixed_gpu_entries(entries)
    assert [(e.miner_hotkey, e.size) for e in dropped] == [("a", "s2")]
    # A size with no king entry is not duelled at all: nothing to enforce.
    assert ("z", "s9") in [(e.miner_hotkey, e.size) for e in kept]


def test_allow_mixed_and_no_king_keep_everything():
    entries = [_e("a", "NVIDIA L40S"), _e("b", "NVIDIA H100")]
    assert _split_mixed_gpu_entries(entries) == (entries, [])
    with_king = [_e("king", "NVIDIA L40S", role="king")] + entries
    assert _split_mixed_gpu_entries(with_king, allow_mixed=True) == (with_king, [])


def test_runner_records_dropped_funded_entries_as_sold_out(tmp_path):
    # locked-type mode: the open market (funded_sku_per_leg, on by default
    # since #294) allows mixed types because its validators lift the gate
    r = _runner(tmp_path, funded_sku_per_leg=False)
    r._enforce_single_gpu_manifest = TrainerRunner._enforce_single_gpu_manifest.__get__(r)
    entries = [_e("king", "NVIDIA GeForce RTX 4090", role="king"),
               _e("hkA", "NVIDIA GeForce RTX 4090"), _e("hkB", "NVIDIA L40S", uid=7)]
    kept = r._enforce_single_gpu_manifest(entries)
    assert [e.miner_hotkey for e in kept] == ["king", "hkA"]
    msg, miner_fault, cls, burn = r._funded_leg_failures["hkB"]
    assert (miner_fault, cls, burn) == (False, "no_capacity", False)
    assert "gpu_mismatch" in msg and "L40S" in msg
    assert "hkA" not in r._funded_leg_failures


def test_runner_allows_mixed_types_in_per_leg_mode(tmp_path):
    r = _runner(tmp_path)
    r.cfg.round = type("R", (), {"funded_sku_per_leg": True})()   # open-market mode
    r._enforce_single_gpu_manifest = TrainerRunner._enforce_single_gpu_manifest.__get__(r)
    entries = [_e("king", "NVIDIA GeForce RTX 4090", role="king"), _e("hkB", "NVIDIA L40S")]
    assert r._enforce_single_gpu_manifest(entries) == entries
    assert r._funded_leg_failures == {}
