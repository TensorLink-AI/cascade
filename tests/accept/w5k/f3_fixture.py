"""The F3 public-source mirror fixtures, `tests/accept/w6p/fixtures.py`,
reached lazily. A combined-mode BASE run overlays only this WP's test dir,
so on a main without F3 the fixture is absent: the first use inside a test
body fails on an `AssertionError` naming it, never a collection error.
"""

from __future__ import annotations


class _LazyFX:
    def __getattr__(self, name: str):
        try:
            from tests.accept.w6p import fixtures as module
        except ImportError:
            raise AssertionError(
                "F3 (rev6.18 A1/A6): tests/accept/w6p/fixtures.py, the public-source mirror "
                "fixtures, is missing") from None
        return getattr(module, name)


fx = _LazyFX()
