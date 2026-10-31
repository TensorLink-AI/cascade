"""Two launch defects of the first v2 era (OPSLOG 2026-10-31):

1. The payer-funded king leg dispatches as ``role=challenger``, so the pod's
   corpus stream applied the release channel floor instead of the genesis
   king's ``king_min_channels`` — the king's corpus was refused on the pod.
   ``for_king`` is ref-gated, so the stream now applies it by ref.
2. A stale ``_funded_wait_abort`` (set when a king rent gave up, to release
   challengers) cancelled the payer-funded KING leg's own capacity wait.
"""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from .test_funded_rent_wait import _arm_wait, _runner


def _stream_floor(cfg, tmp_path, monkeypatch, *, ref, role):
    from cascade.shared.hippius import HubRef, HubUpload
    from cascade.trainer import loop as loop_mod
    from cascade.trainer.contract import RoundSeeds, TrainResult
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

    seen = []

    class _FakeStream:
        digest, n_series, total_points = "corpusdigest", 1, 64

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def series(self):
            yield np.ones((1, 64))

    def _open(contract_mode, gen_dir, seed, generator, **kw):
        seen.append(generator)
        return _FakeStream()

    class _Trainer:
        def train(self, stream, contract, *, training_seed, token_budget, out_dir, logger=None):
            for _ in stream:
                pass
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "weights.safetensors").write_bytes(b"x")
            return TrainResult(local_dir=out_dir, param_count=4, train_seconds=1.0, metrics={})

    # Contract shaping (accepted fields, v2's own floor) is not under test: the
    # stream sees the [generator] floor as is, so 32 vs the king's 1 is visible.
    monkeypatch.setattr(type(cfg), "generator_for_contract", lambda self, contract: self.generator)
    monkeypatch.setattr(loop_mod, "fetch_from_hub", lambda r, dest, hub=None: dest)
    monkeypatch.setattr(loop_mod, "open_round_stream", _open)
    monkeypatch.setattr(
        loop_mod, "upload_dir_to_hub_or_hf",
        lambda local_dir, repo, hub=None, *, hf_repo=None: HubUpload(
            ref=HubRef.parse("cascade/ckpt-out@sha256:" + "e" * 64), size_bytes=1))
    runner = TrainerRunner(cfg=cfg, base_trainer=_Trainer(), work_root=tmp_path,
                           use_sandbox=False)
    runner.train_one(ResolvedGenerator("k", 1, ref), role, RoundSeeds.derive(1, cfg.training), 10)
    return seen[0].min_channels


def _king_cfg(cfg, genesis):
    gen = replace(cfg.generator, min_channels=32, max_channels=max(32, cfg.generator.max_channels))
    return replace(cfg, generator=replace(gen, king_min_channels=1,
                                          king_min_channels_refs=(genesis,)))


@pytest.mark.parametrize("role", ["king", "challenger"])
def test_genesis_king_ref_gets_its_floor_whatever_the_dispatch_role(cfg, tmp_path, monkeypatch, role):
    genesis = "vault/direct@sha256:" + "a" * 64
    assert _stream_floor(_king_cfg(cfg, genesis), tmp_path, monkeypatch,
                         ref=genesis, role=role) == 1


def test_any_other_ref_keeps_the_release_floor(cfg, tmp_path, monkeypatch):
    genesis = "vault/direct@sha256:" + "a" * 64
    got = _stream_floor(_king_cfg(cfg, genesis), tmp_path, monkeypatch,
                        ref="bob/gen@sha256:" + "b" * 64, role="king")
    assert got == 32        # a re-committing king / any challenger: no exemption


# ── the challenger release signal never cancels a king leg ────────────────────


def test_stale_abort_still_releases_a_challenger(tmp_path):
    r = _runner(tmp_path)
    r._funded_wait_abort = threading.Event()
    r._funded_wait_abort.set()
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 0, 2])
    assert r._wait_for_funded_capacity("RTX4090", describe="funded leg x") is False


def test_stale_abort_does_not_cancel_the_funded_king_leg(tmp_path):
    r = _runner(tmp_path)
    r._funded_wait_abort = threading.Event()
    r._funded_wait_abort.set()
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 0, 2])
    r._leg_local_obj = SimpleNamespace(funded_is_king=True)   # this thread's leg is the king
    assert r._wait_for_funded_capacity("H200", describe="funded leg king") is True


# ── the 30-min pod sweep never takes the era king's own-key pod ───────────────


def _king_sched(cfg, tmp_path):
    from .test_rolling import REF, Clock, FakeClient, _armed, _commit, _join, _sched

    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    b0 = armed.round.rolling_from_block + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100)]
    ops.queue().add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    sched.state.current.king_hotkey = "KINGHK"
    return sched, ops, client, clock, b0


def test_live_king_leg_keeps_its_own_key_pod_through_the_sweep(cfg, tmp_path):
    sched, ops, client, clock, b0 = _king_sched(cfg, tmp_path)
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, daemon=True)
    t.start()
    sched._king_threads[sched.state.current.index] = t
    try:
        sched._reconciled_at = -1e12
        sched._maybe_reconcile(client, b0 + 1, clock())
        assert "KINGHK" in ops.sweeps[-1][1]
    finally:
        stop.set()
        t.join()
    # leg over: the king's leftover payer pod is swept like any other
    sched._reconciled_at = -1e12
    sched._maybe_reconcile(client, b0 + 2, clock())
    assert "KINGHK" not in ops.sweeps[-1][1]


def test_startup_sweep_keeps_the_era_king_payer(cfg, tmp_path):
    sched, ops, client, clock, b0 = _king_sched(cfg, tmp_path)
    sched._startup(client, b0 + 1)
    assert "KINGHK" in ops.sweeps[-1][1]
