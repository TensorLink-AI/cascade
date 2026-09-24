"""Legacy round entry trains the RECEIPT king, not the lagging incentive king.

Failure class (2026-09-23 22:04): the validator crowned a new king at 21:13;
on-chain incentive still named the deposed king at the 22:02 boundary; the
round trained the old king and was judged ``king_resyncing`` — every
challenger's paid leg wasted. The receipt trail is the prompt signal.
"""
from __future__ import annotations

from cascade.shared.chain import Commitment
from cascade.trainer import loop as loop_mod
from cascade.trainer.loop import TrainerRunner

OLD = "5Eo6DSBhKyhigwzknosNrVoGAdhkTjsY75C3jpXJMqxHyGBn"
NEW = "5GuWuTLLKunUWNrT7MzjUQT6Bi5GFnBHyN3YZ8wSGUCAc9bt"


class _Client:
    def __init__(self, incentive):
        self._incentive = incentive

    def highest_incentive_hotkey(self):
        return self._incentive


def _commit(uid, hotkey):
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload="metro-v1:gen:hippius:r/g@sha256:" + "0" * 64,
                      commit_block=1)


def _runner(cfg, tmp_path, receipt_king):
    runner = TrainerRunner(cfg=cfg, base_trainer=object(), work_root=tmp_path,
                           use_sandbox=False)
    runner._receipt_king = lambda: receipt_king  # type: ignore[method-assign]
    return runner


def _warnings(monkeypatch):
    # Capture the module logger directly: other suites toggle propagation on
    # the cascade loggers, which makes caplog order-dependent.
    seen: list[str] = []
    monkeypatch.setattr(loop_mod.log, "warning",
                        lambda msg, *a, **k: seen.append(msg % a if a else msg))
    return seen


def test_receipt_king_wins_over_lagging_incentive(cfg, tmp_path, monkeypatch):
    runner = _runner(cfg, tmp_path, receipt_king=NEW)
    seen = _warnings(monkeypatch)
    king = runner._round_entry_king(_Client(OLD), [_commit(87, NEW), _commit(165, OLD)])
    assert king == NEW
    assert any("the receipt decides" in m for m in seen), seen


def test_incentive_king_when_receipts_agree_or_are_absent(cfg, tmp_path):
    agree = _runner(cfg, tmp_path, receipt_king=OLD)
    assert agree._round_entry_king(_Client(OLD), [_commit(165, OLD)]) == OLD
    absent = _runner(cfg, tmp_path, receipt_king=None)
    assert absent._round_entry_king(_Client(OLD), [_commit(165, OLD)]) == OLD


def test_receipt_king_without_a_commitment_falls_back_loudly(cfg, tmp_path, monkeypatch):
    # A champion the trainer cannot resolve must not be silently swapped in:
    # plan_round would warn and the validator would resync anyway — so keep
    # training the incentive king and say why.
    runner = _runner(cfg, tmp_path, receipt_king=NEW)
    seen = _warnings(monkeypatch)
    king = runner._round_entry_king(_Client(OLD), [_commit(165, OLD)])
    assert king == OLD
    assert any("no commitment on file" in m for m in seen), seen
