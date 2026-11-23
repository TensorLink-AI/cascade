"""The intake's chain and registry reads on the live subnet (W3).

:class:`ChainLaneDReads` is the production
:class:`~cascade.lane_d.service.LaneDReads`:

* ``current_block`` / ``coldkey_of`` / ``reveal_block`` go through the one
  :class:`~cascade.shared.chain.ChainClient` (read-only: the intake signs and
  commits nothing). ``reveal_block`` reads the hotkey's
  ``Commitments::RevealedCommitments`` and returns the first block that
  revealed the entry's lane D payload
  (``metro-v1:weights:c<ctx>:hippius:<vault_prefix>/<hotkey>/<digest>``).
* ``published_weights`` reads the published-weights registry: one JSON file
  ``{"schema": "lane-d-published-weights/1", "weights_sha256": [...]}`` that
  lists the weights sha256 of every published checkpoint (lane A legs and
  kings, lane D kings) and the current house. A missing or malformed file
  raises, so dedup fails closed (``503 dedup_unavailable``); it is re-read
  whenever its mtime or size changes.

Every failure raises; nothing is ever read as "absent".
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from .service import LaneDReads, entry_prefix, parse_commit_payload

PUBLISHED_WEIGHTS_SCHEMA = "lane-d-published-weights/1"
PUBLISHED_WEIGHTS_FILENAME = "published_weights.json"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class PublishedWeightsError(RuntimeError):
    """The published-weights registry cannot be read."""


class PublishedWeightsFile:
    """The published-weights registry file, cached by ``(mtime_ns, size)``."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._stamp: tuple[int, int] | None = None
        self._cached: frozenset[str] = frozenset()
        self._lock = threading.Lock()

    def read(self) -> frozenset[str]:
        with self._lock:
            try:
                st = self.path.stat()
            except OSError as e:
                raise PublishedWeightsError(
                    f"published-weights registry {self.path} is unavailable "
                    f"({type(e).__name__})") from None
            stamp = (int(st.st_mtime_ns), int(st.st_size))
            if stamp == self._stamp:
                return self._cached
            try:
                doc = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, ValueError) as e:
                raise PublishedWeightsError(
                    f"published-weights registry {self.path} is unreadable "
                    f"({type(e).__name__})") from None
            shas = doc.get("weights_sha256") if isinstance(doc, dict) else None
            if (not isinstance(doc, dict) or doc.get("schema") != PUBLISHED_WEIGHTS_SCHEMA
                    or not isinstance(shas, list)
                    or not all(isinstance(s, str) and _HEX64.fullmatch(s) for s in shas)):
                raise PublishedWeightsError(
                    f"published-weights registry {self.path} is not a "
                    f"{PUBLISHED_WEIGHTS_SCHEMA} file of 64-hex sha256 values")
            self._cached, self._stamp = frozenset(shas), stamp
            return self._cached


class ChainLaneDReads(LaneDReads):
    """``LaneDReads`` over a :class:`~cascade.shared.chain.ChainClient` and the
    published-weights registry file."""

    def __init__(self, client, *, vault_prefix: str, published_weights: PublishedWeightsFile):
        self.client = client
        self.vault_prefix = str(vault_prefix)
        self.published = published_weights

    def current_block(self) -> int:
        return int(self.client.current_block())

    def coldkey_of(self, hotkey: str) -> str | None:
        for neuron in self.client._metagraph().neurons:
            if str(neuron.hotkey) == hotkey:
                return str(neuron.coldkey)
        return None

    def reveal_block(self, hotkey: str, digest: str) -> int | None:
        ref = entry_prefix(self.vault_prefix, hotkey, digest).rstrip("/")
        entries = self.client._raw_revealed_entries(self.client.subtensor(), hotkey)
        blocks = []
        for block, payload in entries:
            parsed = parse_commit_payload(payload)
            if parsed is not None and parsed[1] == ref:
                blocks.append(int(block))
        return min(blocks) if blocks else None

    def published_weights(self) -> frozenset[str]:
        return self.published.read()


__all__ = ["ChainLaneDReads", "PUBLISHED_WEIGHTS_FILENAME", "PUBLISHED_WEIGHTS_SCHEMA",
           "PublishedWeightsError", "PublishedWeightsFile"]
