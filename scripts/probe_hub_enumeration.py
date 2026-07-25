#!/usr/bin/env python
"""Probe whether the Hippius Hub OCI registry lets THIRD PARTIES enumerate repos.

Why this matters (docs/MINER.md §5a): `cascade deploy` hides the on-chain
pointer with a timed timelock reveal, but the generator CONTENT is uploaded to
the miner's Hub repo at deploy time — before the reveal. The timed reveal only
protects a submission end-to-end if that content is undiscoverable without the
(still-hidden) ref. `--hub-namespace` gives each submission a non-guessable
repo name, which is sufficient **iff** the registry does not let strangers list
a namespace's repos. This script settles that empirically:

* ``GET /v2/_catalog``      — the registry-wide repo listing (should be denied)
* ``GET /v2/<repo>/tags/list`` — tag listing on someone else's known repo
  (a digest-pinned push may expose tags; listing them leaks fresh digests)

Each is tried anonymously and (if ``HIPPIUS_HUB_TOKEN`` is set) authenticated
as a *different* user than the repo owner — the attacker model is "any Hub
account watching your namespace", not "no account at all".

Usage::

    python scripts/probe_hub_enumeration.py                       # anonymous only
    python scripts/probe_hub_enumeration.py --repo other-acct/gen-abc123
    HIPPIUS_HUB_TOKEN=… python scripts/probe_hub_enumeration.py   # + authenticated

Exit codes: 0 = not enumerable (random repo names suffice), 1 = ENUMERABLE
(escalate to sealed submissions — encrypt the artifact, key rides in the
timelocked payload), 2 = inconclusive (endpoints unreachable / all denied
ambiguously).
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

DEFAULT_REGISTRY = "https://registry.hippius.com"


def _get(url: str, token: str | None) -> tuple[int, str]:
    """GET ``url`` → (status, body[:2000]); network errors → (0, reason)."""
    req = urllib.request.Request(url, headers={"User-Agent": "cascade-hub-probe"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 — https probe
            return int(resp.status), resp.read(2000).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return int(e.code), e.read(2000).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 — DNS/timeout/TLS all mean "unreachable"
        return 0, f"{type(e).__name__}: {e}"


def _judge_catalog(status: int, body: str) -> str | None:
    """Return a leak description if the catalog response exposes repos."""
    if status != 200:
        return None
    try:
        repos = json.loads(body).get("repositories", [])
    except ValueError:
        return None
    return f"_catalog returned {len(repos)} repo name(s), e.g. {repos[:3]}" if repos else None


def probe(registry: str, repo: str | None, token: str | None) -> int:
    leaks: list[str] = []
    unreachable = 0
    modes = [("anonymous", None)] + ([("authenticated (other account)", token)] if token else [])

    for label, tok in modes:
        status, body = _get(f"{registry}/v2/_catalog?n=100", tok)
        print(f"[{label}] GET /v2/_catalog → {status if status else body}")
        if status == 0:
            unreachable += 1
        leak = _judge_catalog(status, body)
        if leak:
            leaks.append(f"{label}: {leak}")

        if repo:
            status, body = _get(f"{registry}/v2/{repo}/tags/list", tok)
            print(f"[{label}] GET /v2/{repo}/tags/list → {status if status else body}")
            if status == 200 and '"tags"' in body:
                leaks.append(f"{label}: tags/list on a third-party repo returned {body[:120]!r}")

    print()
    if leaks:
        print("ENUMERABLE — the timed reveal is guarding an empty vault:")
        for line in leaks:
            print(f"  - {line}")
        print("→ escalate to sealed submissions (upload ciphertext; decryption key +\n"
              "  plaintext digest ride in the timelocked payload).")
        return 1
    if unreachable == len(modes):
        print("INCONCLUSIVE — registry unreachable from this network; rerun where "
              f"{registry} is routable.")
        return 2
    print("NOT ENUMERABLE via catalog/tag listing — fresh non-guessable repo names "
          "(`cascade deploy --hub-namespace`) keep pre-reveal content undiscoverable.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--repo", default=None,
                    help="A known repo you do NOT own (namespace/name), to test tag listing.")
    args = ap.parse_args(argv)
    return probe(args.registry.rstrip("/"), args.repo, os.environ.get("HIPPIUS_HUB_TOKEN"))


if __name__ == "__main__":
    raise SystemExit(main())
