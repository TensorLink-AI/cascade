"""Lane D rollout tooling (``cascade-laned``).

Operator-side helpers for the lane D rollout. Each tool is its own module
(``blocks`` is the block-arithmetic tool); :mod:`cascade.laned.cli` dispatches
the ``cascade-laned <command>`` console entry point to them. Nothing here is
consensus code: the tools only read :mod:`cascade.shared`.
"""
