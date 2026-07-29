#!/usr/bin/env python3
"""Verify a cascade VALIDATOR's .env / credentials are correctly configured.

Run from your cascade checkout, with the same env your validator runs under:

    set -a && . ./.env && set +a
    .venv/bin/python scripts/check_validator_env.py [--config chain.toml]

It never prints a credential value — only variable NAMES, lengths, and
PASS/WARN/FAIL per check. Exit code 0 means your validator's storage access
is correctly configured; 1 means at least one required check failed.

What it checks, in the order the validator needs them at runtime:

1. Env presence + shape (set, non-empty, no stray quotes/whitespace).
2. LIVE manifest read  — ``open_manifest_store``: fetch ``manifests/latest.json``
   exactly as the poll loop does.
3. LIVE pool read      — ``pool_s3_store``: fetch ``pool/index.json`` and count
   snapshots. This is the read behind the pool-pin gate; a credential pair
   without POOL-bucket authorization fails HERE and produces
   ``pool_pin_unverifiable: … resolved no snapshot for the round`` rejects.
4. Hub auth resolvable — a Hub bearer token can be resolved for checkpoint
   pulls (env token, cached token, or username/password login).

The functional reads go through cascade's own store builders, so endpoint /
bucket / credential-selection logic is byte-for-byte what your validator does
— including the POOL_S3_* → HIPPIUS_S3_* fallback and any [storage]
pool_s3_endpoint override in your chain.toml.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"
_fails = 0
_warns = 0


def _p(tag: str, colour: str, msg: str) -> None:
    use_colour = sys.stdout.isatty()
    print(f"{colour if use_colour else ''}{tag:5s}{RESET if use_colour else ''} {msg}")


def ok(msg: str) -> None:
    _p("PASS", GREEN, msg)


def warn(msg: str) -> None:
    global _warns
    _warns += 1
    _p("WARN", YELLOW, msg)


def fail(msg: str) -> None:
    global _fails
    _fails += 1
    _p("FAIL", RED, msg)


def env_shape(name: str) -> str:
    """'unset' | 'empty' | 'ok:<len>' | 'malformed:<why>' — never the value."""
    v = os.environ.get(name)
    if v is None:
        return "unset"
    if not v:
        return "empty"
    problems = []
    if v[0] in "\"'" or v[-1] in "\"'":
        problems.append("quoted")
    if v != v.strip():
        problems.append("surrounding whitespace")
    if problems:
        return "malformed:" + "+".join(problems)
    return f"ok:{len(v)}"


def check_env(name: str, *, required: bool, why: str) -> bool:
    shape = env_shape(name)
    if shape.startswith("ok:"):
        ok(f"{name} set (len={shape[3:]}) — {why}")
        return True
    if shape.startswith("malformed:"):
        fail(f"{name} set but {shape[10:]} — fix the .env line ({why})")
        return False
    if required:
        fail(f"{name} {shape} — {why}")
    else:
        warn(f"{name} {shape} — optional: {why}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="chain.toml", help="path to your chain.toml")
    args = ap.parse_args()

    try:
        from cascade.shared.config import load_chain_config
        from cascade.shared.hippius import (
            HUB_PASSWORD_ENV_NAMES,
            HUB_TOKEN_ENV_NAMES,
            HUB_USERNAME_ENV_NAMES,
            MANIFEST_LATEST_KEY,
            StorageError,
            _resolve_hub_token,
            open_manifest_store,
            pool_s3_store,
        )
    except ImportError as e:
        fail(f"cannot import cascade — run via your checkout's venv python: {e}")
        return 1

    print(f"== cascade validator env check (config: {args.config}) ==")
    try:
        cfg = load_chain_config(args.config)
    except Exception as e:  # noqa: BLE001
        fail(f"could not load {args.config}: {e}")
        return 1

    # ---- 1. env presence ------------------------------------------------
    print("-- required credentials --")
    check_env(
        "HIPPIUS_S3_ACCESS_KEY",
        required=True,
        why="manifest reads + receipt writes (and pool fallback)",
    )
    check_env("HIPPIUS_S3_SECRET_KEY", required=True, why="pairs with HIPPIUS_S3_ACCESS_KEY")

    print("-- pool credentials (the pool_pin_unverifiable fix) --")
    pool_a, pool_s = env_shape("POOL_S3_ACCESS_KEY"), env_shape("POOL_S3_SECRET_KEY")
    if pool_a.startswith("ok:") and pool_s.startswith("ok:"):
        ok("POOL_S3_ACCESS_KEY + POOL_S3_SECRET_KEY set — pool store will use this pair")
    elif pool_a == "unset" and pool_s == "unset":
        warn(
            "POOL_S3_* unset — pool store falls back to your HIPPIUS_S3_* pair; "
            "fine ONLY if that pair is authorized for the pool bucket "
            "(the live read below is the real test)"
        )
    else:
        fail(
            "POOL_S3_ACCESS_KEY / POOL_S3_SECRET_KEY must be set as a PAIR "
            f"(access={pool_a}, secret={pool_s}); a half-set pair still selects "
            "POOL_* env and the store builds with a missing key"
        )

    print("-- hub auth (checkpoint pulls) --")
    have_token = any(env_shape(n).startswith("ok:") for n in HUB_TOKEN_ENV_NAMES)
    have_login = any(env_shape(n).startswith("ok:") for n in HUB_USERNAME_ENV_NAMES) and any(
        env_shape(n).startswith("ok:") for n in HUB_PASSWORD_ENV_NAMES
    )
    if have_token:
        ok(f"hub token env set (one of {', '.join(HUB_TOKEN_ENV_NAMES)})")
    elif have_login:
        ok("hub username+password set (token will be minted on demand)")
    else:
        fail(
            "no hub auth: set HIPPIUS_HUB_USERNAME + HIPPIUS_HUB_PASSWORD "
            f"(or a token via one of {', '.join(HUB_TOKEN_ENV_NAMES)}) — "
            "without it checkpoint pulls fail"
        )

    print("-- optional --")
    check_env("HF_TOKEN", required=False, why="HF manifest fallback during Hippius outages")
    if getattr(cfg.storage, "backup_s3_endpoint", ""):
        check_env(
            "BACKUP_S3_ACCESS_KEY",
            required=True,
            why="[storage] backup_s3_endpoint is set in your config",
        )
        check_env("BACKUP_S3_SECRET_KEY", required=True, why="pairs with BACKUP_S3_ACCESS_KEY")

    # ---- config sanity ---------------------------------------------------
    pool_ep = getattr(cfg.storage, "pool_s3_endpoint", "") or ""
    if pool_ep:
        warn(
            f"[storage] pool_s3_endpoint override is set ({pool_ep}) — the "
            "standard setup leaves it empty (default Hippius endpoint); make "
            "sure that is deliberate"
        )
    else:
        ok("[storage] pool_s3_endpoint empty — pool reads use the default Hippius endpoint")

    # ---- 2-3. LIVE reads through the validator's own store builders ------
    print("-- live reads (the real test — same code paths as the validator) --")
    try:
        store = open_manifest_store(cfg.storage)
        latest = json.loads(store.get_text(MANIFEST_LATEST_KEY))
        ok(f"manifest read OK — latest.json points at round {latest.get('round_id', '?')}")
    except (StorageError, Exception) as e:  # noqa: BLE001
        fail(f"manifest read FAILED ({type(e).__name__}: {e}) — validator cannot poll rounds")

    pool_env = "POOL_S3_*" if os.environ.get("POOL_S3_ACCESS_KEY") else "HIPPIUS_S3_* (fallback)"
    try:
        pstore = pool_s3_store(cfg.storage)
        index = json.loads(pstore.get_text("pool/index.json"))
        snaps = index.get("snapshots", index if isinstance(index, list) else [])
        n = len(snaps)
        if n > 0:
            ok(
                f"pool read OK using {pool_env} — pool/index.json lists {n} snapshot(s); "
                "the pool-pin gate can resolve snapshots"
            )
        else:
            fail(
                f"pool/index.json readable via {pool_env} but lists 0 snapshots — "
                "pool-pin gate would reject every round"
            )
    except (StorageError, Exception) as e:  # noqa: BLE001
        fail(
            f"pool read FAILED using {pool_env} ({type(e).__name__}: {e}) — this is "
            "exactly the 'pool_pin_unverifiable: resolved no snapshot' condition; "
            "your credential pair is not authorized for the pool bucket"
        )

    # ---- 4. hub token actually resolvable ---------------------------------
    try:
        _resolve_hub_token("pull")
        ok("hub bearer token resolved — checkpoint pulls will authenticate")
    except Exception as e:  # noqa: BLE001
        fail(
            f"hub token resolution FAILED ({type(e).__name__}: {e}) — "
            "checkpoint pulls will fail at scoring time"
        )

    print()
    if _fails == 0:
        print(f"RESULT: PASS — validator .env is correctly configured ({_warns} warning(s))")
        return 0
    print(
        f"RESULT: FAIL — {_fails} failure(s), {_warns} warning(s); "
        "fix the FAIL lines, then restart your validator"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
