"""``[lane_d_ops]`` — lane D operator settings (LANE_D_PLAN §14.2, W2b).

Non-consensus: where the lane D vault and ``pool_d`` live, which host runs the
lane D forecasts, and the intake / dispatch queue policy. Only the main
validator and the lane D intake read it, straight from the loaded config's
``raw`` table (the ``[satellite]`` pattern), so the chain loader, every digest,
every receipt and every overlay are untouched by it.

Satellites never read it: this module is not re-exported from
``cascade.shared.config`` and the loader does not import it. A module on the
main validator's load path (``cascade.validator`` loads its loop at import)
reads the table only through a function-local import of this module.

The table carries names and locations only. Credentials are env-only, so a
credential-shaped key is refused as such, and its value is never echoed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TABLE = "lane_d_ops"

DEFAULT_VAULT_BUCKET = "cascade-lane-d-vault"   # §1.2: dedicated private bucket
DEFAULT_VAULT_PREFIX = "lane-d"                 # §11.1: lane-d/<hotkey>/<digest>/
DEFAULT_POOL_D_BUCKET = "cascade-lane-d-pool"   # §1.7: pool_d lives in a private bucket
DEFAULT_POOL_D_PREFIX = "pool_d"
DEFAULT_EVAL_HOST = ""                          # unset until the S4 bring-up (D11)
DEFAULT_QUEUE_MAX_INFLIGHT = 1                  # one launch eval host (24 GB class)
DEFAULT_QUEUE_MAX_PENDING = 64                  # advisory; consensus caps stay in [lane_d]
DEFAULT_QUEUE_POLL_SECONDS = 60

KEYS = ("vault_bucket", "vault_prefix", "pool_d_bucket", "pool_d_prefix", "eval_host",
        "queue_max_inflight", "queue_max_pending", "queue_poll_seconds")

# [storage] buckets other services share or publish from; the private lane D
# buckets are never one of them (§1.2 / §13: private code once leaked through
# a shared-bucket archive).
SHARED_STORAGE_BUCKETS = ("manifest_bucket", "logs_bucket", "pool_bucket", "backup_bucket",
                          "pool_backup_bucket", "king_archive_bucket")

_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_HOST_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]{0,252}$")
_SECRET_NAMES = frozenset({"secret", "token", "password", "passwd", "key", "credential",
                           "credentials"})
_SECRET_SUFFIXES = ("_secret", "_token", "_key", "_password", "_passwd", "_credential",
                    "_credentials")


@dataclass(frozen=True)
class LaneDOpsConfig:
    """``vault_bucket`` / ``vault_prefix``: the private bucket and object prefix
    holding lane D entries. ``pool_d_bucket`` / ``pool_d_prefix``: where the
    private ``pool_d`` snapshots live. ``eval_host``: the host or inventory name
    that runs lane D forecasts (``""`` = none configured). ``queue_max_inflight``:
    forecast jobs in flight on the eval host. ``queue_max_pending``: the intake's
    backpressure cap on pending entries (advisory). ``queue_poll_seconds``: the
    ready-queue poll interval."""

    vault_bucket: str = DEFAULT_VAULT_BUCKET
    vault_prefix: str = DEFAULT_VAULT_PREFIX
    pool_d_bucket: str = DEFAULT_POOL_D_BUCKET
    pool_d_prefix: str = DEFAULT_POOL_D_PREFIX
    eval_host: str = DEFAULT_EVAL_HOST
    queue_max_inflight: int = DEFAULT_QUEUE_MAX_INFLIGHT
    queue_max_pending: int = DEFAULT_QUEUE_MAX_PENDING
    queue_poll_seconds: int = DEFAULT_QUEUE_POLL_SECONDS


def _is_secret_name(key: object) -> bool:
    k = str(key).strip().lower()
    return k in _SECRET_NAMES or k.endswith(_SECRET_SUFFIXES)


def _bucket(key: str, v: object) -> str:
    if not isinstance(v, str) or not _BUCKET_RE.match(v):
        raise ValueError(f"[{TABLE}] {key} must be an S3 bucket name "
                         f"(lowercase a-z 0-9 . -, 3..63 chars); got {v!r}")
    return v


def _prefix(key: str, v: object) -> str:
    if (not isinstance(v, str) or not v or v.startswith("/") or ".." in v.split("/")
            or any(c.isspace() for c in v)):
        raise ValueError(f"[{TABLE}] {key} must be a non-empty object prefix with no "
                         f"leading '/', no '..' segment and no whitespace; got {v!r}")
    return v


def _host(key: str, v: object) -> str:
    if not isinstance(v, str) or (v and not _HOST_RE.match(v)):
        raise ValueError(f"[{TABLE}] {key} must be '' or a bare host / inventory name "
                         f"(no user@, scheme, port, path, whitespace or leading '-'); "
                         f"got {v!r}")
    return v


def _positive_int(key: str, v: object) -> int:
    if type(v) is not int or v < 1:
        raise ValueError(f"[{TABLE}] {key} must be an integer >= 1; got {v!r}")
    return v


def parse_lane_d_ops(table: dict | None, storage: dict | None = None) -> LaneDOpsConfig:
    """Parse a ``[lane_d_ops]`` table (None/absent = the defaults), checking its
    buckets against the ``[storage]`` table's shared buckets."""
    if table is None:
        table = {}
    if not isinstance(table, dict):
        raise ValueError(f"[{TABLE}] must be a table")
    secret = sorted(str(k) for k in table if _is_secret_name(k))
    if secret:
        raise ValueError(f"[{TABLE}] secret-shaped key(s) {secret} refused: the table carries "
                         f"names and locations only; credentials are env-only "
                         f"(values not shown)")
    unknown = sorted(str(k) for k in set(table) - set(KEYS))
    if unknown:
        raise ValueError(f"[{TABLE}] unknown key(s): {unknown}")
    out = LaneDOpsConfig(
        vault_bucket=_bucket("vault_bucket", table.get("vault_bucket", DEFAULT_VAULT_BUCKET)),
        vault_prefix=_prefix("vault_prefix", table.get("vault_prefix", DEFAULT_VAULT_PREFIX)),
        pool_d_bucket=_bucket("pool_d_bucket",
                              table.get("pool_d_bucket", DEFAULT_POOL_D_BUCKET)),
        pool_d_prefix=_prefix("pool_d_prefix",
                              table.get("pool_d_prefix", DEFAULT_POOL_D_PREFIX)),
        eval_host=_host("eval_host", table.get("eval_host", DEFAULT_EVAL_HOST)),
        queue_max_inflight=_positive_int(
            "queue_max_inflight", table.get("queue_max_inflight", DEFAULT_QUEUE_MAX_INFLIGHT)),
        queue_max_pending=_positive_int(
            "queue_max_pending", table.get("queue_max_pending", DEFAULT_QUEUE_MAX_PENDING)),
        queue_poll_seconds=_positive_int(
            "queue_poll_seconds", table.get("queue_poll_seconds", DEFAULT_QUEUE_POLL_SECONDS)),
    )
    if out.pool_d_bucket == out.vault_bucket:
        raise ValueError(f"[{TABLE}] pool_d_bucket must differ from vault_bucket "
                         f"(the vault is a dedicated bucket); both are {out.vault_bucket!r}")
    st = storage if isinstance(storage, dict) else {}
    for skey in SHARED_STORAGE_BUCKETS:
        shared = st.get(skey)
        if not shared:
            continue
        for key in ("vault_bucket", "pool_d_bucket"):
            if getattr(out, key) == shared:
                raise ValueError(f"[{TABLE}] {key} {shared!r} is the shared [storage] {skey}; "
                                 f"lane D's private buckets are never a shared bucket")
    return out


def lane_d_ops_config(cfg: Any) -> LaneDOpsConfig:
    """The ``[lane_d_ops]`` table of a loaded :class:`ChainConfig`."""
    raw = getattr(cfg, "raw", None) or {}
    return parse_lane_d_ops(raw.get(TABLE), raw.get("storage"))


__all__ = ["LaneDOpsConfig", "lane_d_ops_config", "parse_lane_d_ops"]
