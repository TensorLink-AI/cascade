"""A `lium up` the platform REFUSES outright ("Node with ID … not found":
the executor vanished between listing and rent) allocated nothing: it must
resolve its create intent, never read as a LEAKED pod, and book $0
(2026-11-04: 5GLSfDut's leg was blocked as "prior rental is not resolved"
after such a refusal, with a phantom $40/h spend record left open). A clean
`lium rm` (exit 0) of a never-listed create is settled on listing evidence."""
from __future__ import annotations

import subprocess

from cascade.funding.providers import create_intents
from cascade.provision.core import LiumProvider, lium_up_rejected
from cascade.provision.funded import LEMON_CLASS

from .test_fix_create_intents_and_losses import (
    HK,  # noqa: F401
    NAME,
    GhostLium,
    _lium_market,
)


def test_up_rejection_markers():
    assert lium_up_rejected("Error: Node with ID '28fc8858-41d3' not found")
    assert lium_up_rejected("executor abc not found")
    assert not lium_up_rejected("connection reset by peer")
    assert not lium_up_rejected("")


class RefusedLium(GhostLium):
    """`lium up` prints the platform's refusal; `lium ps` never lists the pod."""

    def __init__(self, **kw):
        super().__init__(**kw)


def _refused(monkeypatch, tmp_path, ghost):
    vault, rent = _lium_market(monkeypatch, tmp_path, ghost)
    monkeypatch.setattr(LiumProvider, "_up_log_tail",
                        lambda self, name, n=400: "Node with ID 'exec-1' not found")
    return vault, rent


def test_a_refused_create_is_resolved_not_leaked_and_flagged(monkeypatch, tmp_path):
    ghost = RefusedLium(rm_stderr="401 unauthorized")   # even an rm that cannot answer
    vault, rent = _refused(monkeypatch, tmp_path, ghost)
    first = rent()
    assert not first.ok and first.error_class == LEMON_CLASS
    assert first.create_refused and not first.leaked_pod
    assert not create_intents(vault, NAME)
    second = rent()                                     # the leg name is free again
    assert "create_unresolved" not in second.error and len(ghost.ups) == 2


class CleanRmGhost(GhostLium):
    def run(self, argv):
        if argv[1] == "rm":
            self.rms.append(argv[2])
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return super().run(argv)


def test_a_clean_rm_of_a_never_listed_create_settles_on_listings(monkeypatch, tmp_path):
    ghost = CleanRmGhost()
    vault, rent = _lium_market(monkeypatch, tmp_path, ghost)
    first = rent()
    assert not first.ok and not first.leaked_pod and not first.create_refused
    assert not create_intents(vault, NAME)
