"""In-memory checkpoint digests (cascade.trainer.ckpt_digest) end to end:
the saver hashes what it writes, the worker reports the digests, the
orchestrator refuses a harvested checkpoint whose tensor files differ, and
the pod hygiene probe names processes of anyone else on the checkpoint dir."""
from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from cascade.trainer import ckpt_digest as cd
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import (
    PodHygiene,
    RemoteDispatchError,
    RemoteHost,
    parse_hygiene_output,
    probe_pod_hygiene,
    receipt_to_local,
)


def _tensors(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {"w": torch.randn(3, 4, generator=g), "b": torch.zeros(4)}


# ── the saver ────────────────────────────────────────────────────────────────


def test_hashed_save_writes_the_same_bytes_as_save_file_and_records_the_digest(tmp_path):
    from safetensors.torch import save_file

    ref = tmp_path / "ref.safetensors"
    save_file(_tensors(), str(ref))
    out = tmp_path / "ckpt" / "weights.safetensors"
    digest = cd.save_tensors_hashed(_tensors(), out)
    assert out.read_bytes() == ref.read_bytes()                  # format unchanged
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    assert cd.digests_for(out.parent) == {"weights.safetensors": digest}
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".")]   # no tmp left


def test_digests_are_per_directory(tmp_path):
    a = cd.save_tensors_hashed(_tensors(1), tmp_path / "a" / "weights.safetensors")
    b = cd.save_tensors_hashed(_tensors(2), tmp_path / "b" / "weights.safetensors")
    assert cd.digests_for(tmp_path / "a") == {"weights.safetensors": a}
    assert cd.digests_for(tmp_path / "b") == {"weights.safetensors": b}
    assert cd.digests_for(tmp_path) == {}


def test_mismatches_name_replaced_missing_and_planted_files(tmp_path):
    d = tmp_path / "ckpt"
    w = cd.save_tensors_hashed(_tensors(1), d / "weights.safetensors")
    o = cd.save_tensors_hashed(_tensors(2), d / "optimizer.safetensors")
    expected = {"weights.safetensors": w, "optimizer.safetensors": o}
    assert cd.tensor_mismatches(d, expected) == []
    # replaced after the save
    cd.save_tensors_hashed(_tensors(3), tmp_path / "other.safetensors")
    (d / "weights.safetensors").write_bytes((tmp_path / "other.safetensors").read_bytes())
    bad = cd.tensor_mismatches(d, expected)
    assert len(bad) == 1 and bad[0].startswith("weights.safetensors: sha256")
    # planted beside the run's files
    (d / "weights_stable.safetensors").write_bytes(b"x")
    assert any(r.startswith("weights_stable.safetensors: tensor file the worker never wrote")
               for r in cd.tensor_mismatches(d, expected))
    # gone
    (d / "optimizer.safetensors").unlink()
    assert any(r.startswith("optimizer.safetensors: missing") for r in cd.tensor_mismatches(d, expected))


# ── the receipt ──────────────────────────────────────────────────────────────


def _base_receipt() -> dict:
    return {"miner_hotkey": "hk", "miner_uid": 5, "role": "challenger", "gen_ref": "r@sha256:" + "a" * 64,
            "corpus_digest": "cd", "train_block": 10, "local_checkpoint_dir": "_train_work/1/x/checkpoint"}


def test_local_receipt_carries_tensor_digests_and_defaults_empty():
    r = receipt_to_local({**_base_receipt(), "tensor_digests": {"weights.safetensors": "ab" * 32}})
    assert r.tensor_digests == {"weights.safetensors": "ab" * 32}
    assert receipt_to_local(_base_receipt()).tensor_digests == {}
    with pytest.raises(RemoteDispatchError):
        receipt_to_local({**_base_receipt(), "tensor_digests": 7})


def test_worker_receipt_reports_the_digests_of_the_files_it_wrote(tmp_path, cfg):
    contract = cfg.training.primary_size
    out = tmp_path / "7" / contract.arch_preset / "challenger" / "checkpoint"

    def _train(gen, role, seeds, *, contract, token_budget, repo_suffix, heat, warm_start_ref):
        cd.save_tensors_hashed(_tensors(), out / "weights.safetensors")
        return out, "cd", "gpu"

    fake = SimpleNamespace(cfg=cfg, TRAIN_COMPLETE_MARKER=TrainerRunner.TRAIN_COMPLETE_MARKER,
                           _train_for_entry=_train)
    fake._tensor_digests_of = lambda d: TrainerRunner._tensor_digests_of(fake, d)
    gen = SimpleNamespace(hotkey="hk", uid=5, ref="r@sha256:" + "a" * 64)
    receipt = TrainerRunner.train_one_local(
        fake, gen, "challenger", SimpleNamespace(base_seed=7), 10, contract=contract, token_budget=1)
    assert receipt["tensor_digests"] == cd.digests_for(out)
    assert receipt["tensor_digests"]["weights.safetensors"] == cd.file_sha256(out / "weights.safetensors")


def test_marker_carries_the_digests_and_a_reused_checkpoint_reports_them(tmp_path, cfg):
    contract = cfg.training.primary_size
    out = tmp_path / "ckpt"
    w = cd.save_tensors_hashed(_tensors(), out / "weights.safetensors")
    gen = SimpleNamespace(hotkey="hk", uid=5, ref="r@sha256:" + "a" * 64)
    fake = SimpleNamespace(TRAIN_COMPLETE_MARKER=TrainerRunner.TRAIN_COMPLETE_MARKER)
    TrainerRunner._write_train_complete_marker(fake, out, contract, gen, corpus_digest="cd", gpu_name="g")
    payload = json.loads((out / TrainerRunner.TRAIN_COMPLETE_MARKER).read_text())
    assert payload["tensor_digests"] == {"weights.safetensors": w}
    # another process (no in-memory record for this dir) reads them off the marker
    cd._DIGESTS.clear()
    assert TrainerRunner._tensor_digests_of(fake, out) == {"weights.safetensors": w}
    assert TrainerRunner._tensor_digests_of(fake, tmp_path / "nowhere") == {}


# ── the pod hygiene probe ────────────────────────────────────────────────────

_CKPT = "/root/cascade/_train_work/1/toto2-4m/challenger/checkpoint"
_OUT = f"""\
   123 /root/cascade/.venv/bin/python -m cascade.trainer.worker --work-root ./_train_work
   130 /root/cascade/.venv/bin/python -m cascade.trainer.sandbox --stream _train_work/1/toto2-4m/challenger/x
   555 python3 watch.py --source ./w.safetensors --target-dir {_CKPT} --name weights.safetensors
   556 bash -c ps -eo pid,args 2>/dev/null | grep -F -- _train_work/1/toto2-4m/challenger/checkpoint; echo __HYGIENE_SESSIONS__
__HYGIENE_SESSIONS__
3
__HYGIENE_IDE__
/root/.vscode-server
"""


def test_hygiene_parse_names_foreign_processes_extra_sessions_and_an_ide():
    h = parse_hygiene_output(_OUT)
    assert len(h.foreign_procs) == 1 and "watch.py" in h.foreign_procs[0]
    assert h.sessions == 2                       # three minus the probe's own
    assert h.ide_server is True and h.error == ""


def test_hygiene_parse_of_a_clean_pod_is_empty():
    h = parse_hygiene_output("__HYGIENE_SESSIONS__\n1\n__HYGIENE_IDE__\n")
    assert h == PodHygiene()


def _host(tmp_path) -> RemoteHost:
    return RemoteHost(name="p", host="10.0.0.1", port=22, user="root", key_path=str(tmp_path / "k"),
                      remote_python="/root/cascade/.venv/bin/python", workdir="/root/cascade")


def test_hygiene_probe_asks_for_the_workdir_relative_dir_and_reports_transport_failures(tmp_path):
    seen = []

    def _run(argv, timeout, stdin):
        seen.append(argv[-1])
        return SimpleNamespace(returncode=0, stdout=_OUT, stderr="")

    h = probe_pod_hygiene(_host(tmp_path), _CKPT, runner=_run)
    assert "_train_work/1/toto2-4m/challenger/checkpoint" in seen[0]
    assert "/root/cascade/_train_work" not in seen[0].split("grep -F --")[1].split(";")[0]
    assert h.foreign_procs and h.ide_server

    def _dead(argv, timeout, stdin):
        raise subprocess.TimeoutExpired(argv, timeout)

    h = probe_pod_hygiene(_host(tmp_path), _CKPT, runner=_dead)
    assert h.error and h == PodHygiene(error=h.error)

    h = probe_pod_hygiene(_host(tmp_path), _CKPT,
                          runner=lambda a, t, s: SimpleNamespace(returncode=255, stdout="", stderr="no route"))
    assert "rc=255" in h.error


# ── the knob ─────────────────────────────────────────────────────────────────


def test_require_digests_knob_defaults_off_and_is_loader_parsed(cfg):
    import inspect

    from cascade.shared import config as cfg_mod
    from cascade.shared.config import RoundConfig

    assert RoundConfig().funded_require_tensor_digests is False
    assert cfg.round.funded_require_tensor_digests is False
    assert ('funded_require_tensor_digests=bool(r.get("funded_require_tensor_digests", False))'
            in inspect.getsource(cfg_mod.load_chain_config))
