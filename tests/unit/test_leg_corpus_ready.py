"""A leg whose corpus is already stored needs no CPU generation lead
(2026-10-30: six ready challengers were refused admission because the
deadline still reserved the 3.5 h generation lead)."""

from __future__ import annotations

from types import SimpleNamespace

from cascade.shared.config import load_chain_config
from cascade.trainer.runner_generation import _GenerationMixin


class _R(_GenerationMixin):
    def __init__(self, stage):
        self.cfg = load_chain_config()
        self._stage = stage

    def _generation_stage(self):
        if isinstance(self._stage, Exception):
            raise self._stage
        return self._stage


GEN = SimpleNamespace(hotkey="5Fminer", ref="ns/gen@sha256:" + "a" * 64, uid=3)
SEEDS = SimpleNamespace(base_seed=14349492220266598742, generation_seed=99)


def test_round_id_is_the_seed_epoch_and_ref():
    rid = _GenerationMixin._leg_round_id(GEN, SEEDS)
    assert rid.startswith("14349492220266598742-") and len(rid.split("-")[1]) == 12
    assert rid == _GenerationMixin._leg_round_id(GEN, SimpleNamespace(base_seed=SEEDS.base_seed,
                                                                       generation_seed=1))


def test_ready_only_when_the_stage_finds_the_corpus():
    seen = {}

    def existing(**kw):
        seen.update(kw)
        return object()

    assert _R(SimpleNamespace(existing=existing))._leg_corpus_ready(GEN, SEEDS)
    assert seen["seed"] == 99 and seen["gen_ref"] == GEN.ref and seen["min_channels"] is None
    assert not _R(SimpleNamespace(existing=lambda **kw: None))._leg_corpus_ready(GEN, SEEDS)
    assert not _R(RuntimeError("s3 down"))._leg_corpus_ready(GEN, SEEDS)
