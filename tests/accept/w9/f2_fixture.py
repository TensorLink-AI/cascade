"""The F2 multichannel chain fixture, `tests/accept/w7a/mv.py`, reached
lazily. A combined-mode BASE run overlays only this WP's test dir, so on a
main without F2 the fixture is absent: the first use inside a test body
fails on an `AssertionError` naming it, never a collection error.
"""

from __future__ import annotations


class _LazyMV:
    def __getattr__(self, name: str):
        try:
            from tests.accept.w7a import mv as module
        except ImportError:
            raise AssertionError(
                "F2 (rev6.18 A4): tests/accept/w7a/mv.py, the multichannel chain fixture of "
                "the multichannel lane D scoring, is missing") from None
        return getattr(module, name)


mv = _LazyMV()
