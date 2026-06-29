"""Miner CLI: build a generator, verify it, and commit its pointer on-chain."""

from __future__ import annotations

from .verify import VerifyReport, verify_repo

__all__ = ["VerifyReport", "verify_repo"]
