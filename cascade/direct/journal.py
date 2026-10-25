"""Signed direct journal transport over Hippius S3 (trainer → validators).

The trainer publishes its sr25519-signed journal; validators fetch it, verify
the trainer signature and replay every event and decision. They never run an
evaluation: the journal carries the trainer's per-case scores and validators
re-derive each verdict, cohort result and payout split from them.

Whether those scores are TRUE is settled by reproduction, not by validators:
every decision binds the checkpoints, panels and scorer settings that produced
it, checkpoints are public-read, and a private panel is revealed here once the
operator has rotated it out. ``cascade-audit direct-moe --reproduce`` re-scores
and fails any decision whose recorded scores or verdict do not reproduce.
"""

from __future__ import annotations

import json

from ..shared.direct_moe import canonical, digest

LATEST = "direct/journal/latest.json"


def _store(config):
    from ..shared.hippius import S3Config, S3Store

    return S3Store(S3Config(endpoint=config.get("endpoint", "https://s3.hippius.com"),
                            region=config.get("region", "decentralized"),
                            bucket=config["bucket"]))


def journal_key(signed):
    return f"direct/journal/{digest(signed)}.json"


def publish_journal(config, signed, *, store=None):
    """Write the immutable journal object, then advance the ``latest`` pointer."""
    store = store or _store(config)
    key = journal_key(signed)
    store.put_text(key, canonical(signed), content_type="application/json", acl="public-read")
    store.put_text(LATEST, canonical({"key": key, "sha256": digest(signed)}),
                   content_type="application/json", acl="public-read")
    return key


def fetch_journal(config, *, store=None):
    store = store or _store(config)
    pointer = json.loads(store.get_text(LATEST))
    signed = json.loads(store.get_text(pointer["key"]))
    if digest(signed) != pointer["sha256"]:
        raise ValueError("journal object does not match its pointer")
    return signed


def reveal_key(panel_digest, suffix):
    return f"direct/reveal/{panel_digest}.{suffix}"


def reveal_panel(config, panel, body, *, store=None):
    """Publish a RETIRED private panel (descriptor + arrays) for reproduction."""
    import hashlib

    if hashlib.sha256(body).hexdigest() != panel["object"]["sha256"]:
        raise ValueError("panel body does not match its descriptor")
    store = store or _store(config)
    store.put_bytes(reveal_key(digest(panel), "npz"), body, acl="public-read")
    store.put_text(reveal_key(digest(panel), "json"), canonical(panel),
                   content_type="application/json", acl="public-read")
    return digest(panel)


def fetch_reveal(config, panel_digest, *, store=None):
    """The revealed panel bound by a decision, or None while it is still private."""
    import hashlib

    from ..shared.hippius import ObjectNotFound

    store = store or _store(config)
    try:
        panel = json.loads(store.get_text(reveal_key(panel_digest, "json")))
        body = store.get_bytes(reveal_key(panel_digest, "npz"))
    except (ObjectNotFound, KeyError):
        return None
    if digest(panel) != panel_digest or hashlib.sha256(body).hexdigest() != panel["object"]["sha256"]:
        raise ValueError("revealed panel does not match the decision's binding")
    return panel, body
