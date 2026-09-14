"""The worker aborts a diverged run on the spot (toto2_trainer.check_loss_finite)."""

from __future__ import annotations

import math

import pytest

from cascade.trainer.corpus import DIVERGED_MARKER, CorpusError
from cascade.trainer.toto2_trainer import check_loss_finite


def test_finite_loss_is_silent():
    check_loss_finite(0.118, step=150, tokens=39321600)
    check_loss_finite(0.0, step=1, tokens=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_loss_is_a_corpus_error_the_worker_exits_3_on(bad):
    # CorpusError ⇒ worker rc=3 ⇒ classify_funded_worker_failure → "generator":
    # the contract's loop is identical for every leg, the corpus is not.
    with pytest.raises(CorpusError) as ei:
        check_loss_finite(bad, step=150, tokens=39321600)
    msg = str(ei.value)
    assert msg.startswith(DIVERGED_MARKER)
    assert "step 150" in msg and "39321600" in msg
    assert not math.isfinite(bad)


def test_marker_is_not_the_stall_marker():
    # The rc=3 classifier keys the STALL exemption on its own marker; a
    # divergence must never read as a stall (infra-side, unburned).
    from cascade.trainer.loop import _STALL_MARKER, classify_funded_worker_failure
    assert _STALL_MARKER not in DIVERGED_MARKER
    assert classify_funded_worker_failure(
        3, f"miner submission rejected: {DIVERGED_MARKER}: non-finite loss",
        stalled_before=False) == (True, "generator", False)
