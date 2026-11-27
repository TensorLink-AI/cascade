"""The lane D score cache (LANE_D_PLAN rev6.18 B6; §1.5 "State").

Forecasts the eval host computed for ``(digest, context, horizon, window_id)``
are kept so a later settlement does not forecast the same model on the same
window twice. The key also carries the **host identity** (B6:
``sha256(env ‖ GPU model ‖ driver ‖ torch build ‖ fingerprint)``,
:func:`cascade.shared.config.lane_d.host_identity`) and the code identity
(the W4 package hash): a cached forecast is only ever returned to a run on
the same host running the same code. A host change therefore misses for
EVERY model and the whole cohort is re-scored on the new host, so every
model in a cohort is always scored on one host (B6: "same machine, same
run").

Layout: ``<root>/<namespace>/<digest>-<context>-<horizon>.npz`` holding
``window_ids`` and ``quantiles`` (float64, exactly as the eval host
returned them), ``namespace = sha256(host identity ‖ package hash)``. A
read requires every requested window to be present (else a miss: the model
is re-forecast on all of them, never a mixed answer). Writes are atomic
(temp file + rename). :meth:`ScoreCache.retain` drops every other
namespace and every key the current run did not use, so disk stays bounded
by one settlement's cohort. Malformed files are misses and are deleted.

**Public half** (rev6.18 F4): the eval host's per-window public scores for
``(digest, context, horizon)`` on one settlement's public draws are kept as
``<root>/<namespace>/public-<digest>-<context>-<horizon>-<draw key>.json``
(``{scores, dropped}`` exactly as the host replied),
``draw key = sha256(canonical draws ‖ block hash ‖ max_channels)``
(:func:`public_key`), so a retry of the same settlement never re-reads and
re-scores 10,000 public windows per digest. :meth:`ScoreCache.retain` leaves
them alone; :meth:`ScoreCache.retain_public` keeps only the current run's.

Not consensus: satellites re-derive verdicts from the signed rows, never
from this cache; it only saves eval host GPU time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger("cascade.validator.lane_d_score_cache")

N_LEVELS = 9
_HEX64 = re.compile(r"[0-9a-f]{64}")
PUBLIC_PREFIX = "public-"


def public_key(draws, *, block_hash, max_channels: int) -> str:
    """The public draw identity a cached public reply is keyed by."""
    blob = json.dumps({"draws": draws, "block_hash": block_hash,
                       "max_channels": int(max_channels)},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(f"lane-d-public-score-cache/1:{blob}".encode()).hexdigest()


def namespace(host_identity: str, package_hash: str) -> str:
    """The cache directory for one (host identity, code identity) pair."""
    for name, v in (("host_identity", host_identity), ("package_hash", package_hash)):
        if not isinstance(v, str) or not _HEX64.fullmatch(v):
            raise ValueError(f"score cache: {name} must be 64 lower-case hex; got {v!r}")
    return hashlib.sha256(f"lane-d-score-cache/1:{host_identity}:{package_hash}"
                          .encode("ascii")).hexdigest()


class ScoreCache:
    """See the module docstring. ``root`` is created on first write."""

    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def _name(digest: str, context: int, horizon: int) -> str:
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise ValueError(f"score cache: bad digest {digest!r}")
        return f"{digest}-{int(context)}-{int(horizon)}.npz"

    def _path(self, ns: str, digest: str, context: int, horizon: int) -> Path:
        return self.root / ns / self._name(digest, context, horizon)

    def get(self, ns: str, digest: str, *, context: int, horizon: int,
            window_ids: Sequence[str]) -> np.ndarray | None:
        """The cached ``(len(window_ids), horizon, 9)`` forecasts in
        ``window_ids`` order, or ``None`` (absent, another host, a window
        missing, or a malformed file)."""
        p = self._path(ns, digest, context, horizon)
        if not p.is_file():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                ids = [str(x) for x in z["window_ids"]]
                q = np.asarray(z["quantiles"], dtype=np.float64)
        except (OSError, ValueError, KeyError) as e:
            log.warning("lane D score cache: %s unreadable (%s); dropped", p.name, e)
            p.unlink(missing_ok=True)
            return None
        if q.shape != (len(ids), int(horizon), N_LEVELS) or len(set(ids)) != len(ids):
            log.warning("lane D score cache: %s malformed; dropped", p.name)
            p.unlink(missing_ok=True)
            return None
        at = {wid: i for i, wid in enumerate(ids)}
        if any(w not in at for w in window_ids):
            return None
        return q[[at[w] for w in window_ids]]

    def put(self, ns: str, digest: str, *, context: int, horizon: int,
            window_ids: Sequence[str], quantiles) -> None:
        """Store finite ``(len(window_ids), horizon, 9)`` forecasts (a
        malformed or non-finite array is never cached)."""
        q = np.asarray(quantiles, dtype=np.float64)
        ids = [str(w) for w in window_ids]
        if q.shape != (len(ids), int(horizon), N_LEVELS) or not np.isfinite(q).all() \
                or len(set(ids)) != len(ids):
            return
        p = self._path(ns, digest, context, horizon)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, window_ids=np.asarray(ids, dtype=str), quantiles=q)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # ── the public half ─────────────────────────────────────────────────────
    def _public_name(self, digest: str, context: int, horizon: int, key: str) -> str:
        if not isinstance(key, str) or not _HEX64.fullmatch(key):
            raise ValueError(f"score cache: bad public key {key!r}")
        return f"{PUBLIC_PREFIX}{self._name(digest, context, horizon)[:-4]}-{key}.json"

    def get_public(self, ns: str, digest: str, *, context: int, horizon: int,
                   key: str) -> dict | None:
        """The cached public reply ``{scores, dropped}``, or ``None``."""
        p = self.root / ns / self._public_name(digest, context, horizon, key)
        if not p.is_file():
            return None
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("lane D score cache: %s unreadable (%s); dropped", p.name, e)
            p.unlink(missing_ok=True)
            return None
        if not isinstance(doc, dict) or set(doc) != {"scores", "dropped"} \
                or not isinstance(doc["scores"], dict) or not isinstance(doc["dropped"], dict):
            log.warning("lane D score cache: %s malformed; dropped", p.name)
            p.unlink(missing_ok=True)
            return None
        return doc

    def put_public(self, ns: str, digest: str, *, context: int, horizon: int, key: str,
                   reply) -> None:
        """Store a well-formed public reply (``{scores, dropped}`` dicts)."""
        if not isinstance(reply, dict) or not isinstance(reply.get("scores"), dict) \
                or not isinstance(reply.get("dropped"), dict):
            return
        p = self.root / ns / self._public_name(digest, context, horizon, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"scores": reply["scores"], "dropped": reply["dropped"]}, f,
                          sort_keys=True)
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def retain_public(self, ns: str, keys: Iterable[tuple[str, int, int, str]]) -> None:
        """Keep only namespace ``ns``'s public replies for ``(digest, context,
        horizon, key)`` ``keys`` (the current settlement's)."""
        d = self.root / ns
        if not d.is_dir():
            return
        keep = {self._public_name(dg, c, h, k) for dg, c, h, k in keys}
        for f in d.iterdir():
            if f.name.startswith(PUBLIC_PREFIX) and f.name not in keep:
                f.unlink(missing_ok=True)

    def retain(self, ns: str, keys: Iterable[tuple[str, int, int]]) -> None:
        """Keep only namespace ``ns`` and its ``(digest, context, horizon)``
        ``keys``; everything else is deleted (a host change drops the old
        host's forecasts). Public replies are :meth:`retain_public`'s."""
        if not self.root.is_dir():
            return
        keep = {self._name(d, c, h) for d, c, h in keys}
        for child in self.root.iterdir():
            if child.name != ns:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                continue
            for f in child.iterdir():
                if f.name not in keep and not f.name.startswith(PUBLIC_PREFIX):
                    f.unlink(missing_ok=True)


__all__ = ["PUBLIC_PREFIX", "ScoreCache", "namespace", "public_key"]
