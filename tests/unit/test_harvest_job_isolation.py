"""Parallel king/replica harvests keep independent files and pending uploads."""
from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cascade.trainer import runner_common as C
from cascade.trainer.loop import TrainerRunner


def test_pending_jobs_are_independent_and_cleanup_removes_only_the_uploaded_job(tmp_path):
    for job, suffix in [("one", ""), ("two", "-backup")]:
        path = C.harvest_dir(tmp_path, 123, "king", job_id=job)
        path.mkdir(parents=True)
        (path / "weights").write_text(job)
        C.write_pending_upload(tmp_path, 123, "king", dict(dest=str(path), harvest_job_id=job,
                                                        suffix=suffix, gen_ref="ref"))
    assert C.read_pending_upload(tmp_path, 123, "king") is None  # Ambiguous without job identity.
    one = C.read_pending_upload(tmp_path, 123, "king", suffix="", gen_ref="ref")
    two = C.read_pending_upload(tmp_path, 123, "king", suffix="-backup", gen_ref="ref")
    assert one["harvest_job_id"] == "one"
    assert (Path(one["dest"]) / "weights").read_text() == "one"
    assert two["harvest_job_id"] == "two"
    C.clear_harvest(tmp_path, 123, "king", job_id="one")
    assert not Path(one["dest"]).exists()
    assert (Path(two["dest"]) / "weights").read_text() == "two"
    assert C.read_pending_upload(tmp_path, 123, "king", suffix="-backup") == two


def test_two_actual_harvest_paths_do_not_overwrite_each_other(cfg, tmp_path, monkeypatch):
    from cascade.eval import checkpoint_guard as G
    from cascade.trainer import loop as L
    from cascade.trainer import remote as R

    r = TrainerRunner(cfg=replace(cfg, round=replace(cfg.round, funded_require_tensor_digests=False)),
                      base_trainer=None, work_root=tmp_path)
    barrier = threading.Barrier(2)
    results, errors = {}, []
    def harvest(host, remote, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "bytes").write_text(host)
    def upload(runner, record, *, base_seed):
        barrier.wait(timeout=5)
        results[record["suffix"]] = (Path(record["dest"]) / "bytes").read_text()
        C.clear_harvest(tmp_path, base_seed, "king", job_id=record["harvest_job_id"])
        return "entry"
    monkeypatch.setattr(R, "harvest_remote_dir", harvest)
    monkeypatch.setattr(G, "verify_checkpoint", lambda *a: None)
    monkeypatch.setattr(G, "verify_weights_finite", lambda *a: None)
    monkeypatch.setattr(L, "upload_harvested_checkpoint", upload)
    receipt = SimpleNamespace(miner_hotkey="king", miner_uid=1, role="king", size="size", gen_ref="ref",
        checkpoint_dir="remote", tensor_digests={}, corpus_digest="digest", train_block=1, gpu_name="fake",
        init_pointer="init", training_contract_digest="contract")
    def run(host, suffix):
        try:
            r._harvest_funded_checkpoint(host, receipt, cfg.training.primary_size,
                                         SimpleNamespace(base_seed=123), suffix)
        except Exception as exc:
            errors.append(exc)
    threads = [threading.Thread(target=run, args=("primary", "")),
               threading.Thread(target=run, args=("backup", "-backup"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert not errors and results == {"": "primary", "-backup": "backup"}


def test_job_binding_changes_with_init_contract_source_role_and_suffix():
    rec = SimpleNamespace(miner_hotkey="king", role="king", size="m", gen_ref="ref",
                          init_pointer="init", training_contract_digest="contract")
    original = C.harvest_job_id(rec, "")
    assert C.harvest_job_id(rec, "-backup") != original
    for key in ("role", "size", "gen_ref", "init_pointer", "training_contract_digest"):
        other = SimpleNamespace(**{**vars(rec), key: "other"})
        assert C.harvest_job_id(other, "") != original
