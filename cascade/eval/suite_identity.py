"""Identity of a swept benchmark suite (LANE_D_PLAN §1.3b "Suite identity is
verified per drawn suite", §13 "Suite identity / partial").

A swept suite is identified by three things the repo already carries for it:
its guard key (a :data:`cascade.eval.benchmarks.ITEM_SUITE_KEY` value), its
pinned data revision (``data_revision``, which
:func:`cascade.eval.benchmarks.item_rows_from_report` copies from the report's
``data_revisions``) and the sha256 of its canonical item-id list. The
identity is a pure function of the item set and the version, so every
validator computing it from the same suite gets the same digest whatever the
row order or the process hash seed. Lane A / lane B can re-use it (WB11's
hash set, LB1's "refreshed only when a suite identity changes").
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

from cascade.eval.benchmarks import ITEM_SUITE_KEY

KNOWN_SUITES = tuple(ITEM_SUITE_KEY.values())
IDENTITY_FIELDS = ("suite", "version", "items_sha256")


def _sha256_ascii(obj: object, *, sort_keys: bool = False) -> str:
    canon = json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def _canonical_ids(item_ids: Iterable[object]) -> list[str]:
    if isinstance(item_ids, (str, bytes)):
        raise TypeError(f"item_ids must be a sequence of ids; got {item_ids!r}")
    ids: set[str] = set()
    for i in item_ids:
        if not isinstance(i, str):
            raise TypeError(f"item id must be a str; got {i!r}")
        if not i:
            raise ValueError("item id must be non-empty")
        ids.add(i)
    return sorted(ids)


def item_set_digest(item_ids: Iterable[object]) -> str:
    """sha256 (lower-case hex) of the sorted, de-duplicated item-id list as
    compact ASCII JSON. Order and repeats don't matter; a non-str id raises
    ``TypeError`` and an empty id ``ValueError``."""
    return _sha256_ascii(_canonical_ids(item_ids))


def _identity_digest(suite: str, version: str | None, items_sha256: str) -> str:
    return _sha256_ascii({"items_sha256": items_sha256, "suite": suite, "version": version},
                         sort_keys=True)


def suite_identity(suite: str, entry: Mapping) -> dict:
    """``{"suite", "version", "items_sha256", "n_items", "digest"}`` of one
    sweep-record suite entry (``{"status": "ok", "rows": […],
    "data_revision"?: str}``). Every row's ``item`` counts, whatever its
    metric values. Raises ``ValueError`` for an unknown suite key or a
    non-``"ok"`` entry, ``TypeError`` / ``ValueError`` for malformed rows."""
    if not isinstance(suite, str):
        raise TypeError(f"suite must be a str; got {suite!r}")
    if suite not in KNOWN_SUITES:
        raise ValueError(f"suite {suite!r} is not one of {list(KNOWN_SUITES)}")
    if not isinstance(entry, Mapping):
        raise TypeError(f"suite entry must be a mapping; got {type(entry).__name__}")
    if entry.get("status") != "ok":
        raise ValueError(f"suite {suite!r} status {entry.get('status')!r} is not 'ok': "
                         "an unswept suite has no identity")
    rows = entry.get("rows")
    if not isinstance(rows, list):
        raise TypeError(f"suite {suite!r} rows must be a list; got {type(rows).__name__}")
    ids = []
    for i, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"suite {suite!r} row {i} is not an object")
        ids.append(row.get("item"))
    canon = _canonical_ids(ids)
    rev = entry.get("data_revision")
    version = None if rev is None else str(rev)
    items_sha256 = _sha256_ascii(canon)
    return {"suite": suite, "version": version, "items_sha256": items_sha256,
            "n_items": len(canon), "digest": _identity_digest(suite, version, items_sha256)}


def identity_mismatch(observed: Mapping, pinned: Mapping) -> list[str]:
    """The fields among ``("suite", "version", "items_sha256")``, in that
    order, whose values differ; ``[]`` means the same identity."""
    return [f for f in IDENTITY_FIELDS if observed.get(f) != pinned.get(f)]


def identities_from_rows(item_rows: Mapping | None) -> dict:
    """``{suite: suite_identity(suite, entry)}`` for every ``"ok"`` suite of an
    :func:`cascade.eval.benchmarks.item_rows_from_report` result; non-ok
    suites are left out."""
    if not item_rows:
        return {}
    return {suite: suite_identity(suite, entry) for suite, entry in sorted(item_rows.items())
            if isinstance(entry, Mapping) and entry.get("status") == "ok"}


__all__ = ["IDENTITY_FIELDS", "KNOWN_SUITES", "identities_from_rows", "identity_mismatch",
           "item_set_digest", "suite_identity"]
