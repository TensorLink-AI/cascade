"""Miner-facing contract: the DataGenerator ABC, output checks, the on-chain
commit format, and the static-import guard."""

from __future__ import annotations

from .generator import DataGenerator, check_series, drain_generator
from .static_guard import GuardResult, scan_file, scan_source
from .validation import (
    ParsedCommit,
    ValidationResult,
    check_config,
    check_repo_layout,
    check_repo_size,
    check_requirements_hash_locked,
    format_commit,
    parse_commit,
)

__all__ = [
    "DataGenerator",
    "check_series",
    "drain_generator",
    "GuardResult",
    "scan_file",
    "scan_source",
    "ParsedCommit",
    "ValidationResult",
    "check_config",
    "check_repo_layout",
    "check_repo_size",
    "check_requirements_hash_locked",
    "format_commit",
    "parse_commit",
]
