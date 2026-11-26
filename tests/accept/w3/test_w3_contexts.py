"""W3 (plan rev6.6, owner 2026-11-23; §1.1 context bullet): which contexts
the intake accepts. Contract: README.md "Contexts (rev6.6)".

``[lane_d] contexts`` is the set a miner may DECLARE. Since rev6.6 it lists
every variant from launch (4k..128k). Arming is computed per settlement by
the main validator (W5d/W7a) and is not the intake's business: an entry that
declares an unarmed variant is accepted, and is scored at the highest armed
variant below its declared context until its own arms (none below it: not
scored that settlement, ``context_below_armed``, plan rev6.15). A context outside ``contexts`` is still refused
(``400 context_not_armed``, the code W3 has always used for it).
"""

from __future__ import annotations

from .helpers import (
    Env,
    Miner,
    checkpoint,
    digest_of,
    lane_d_cfg,
    manifest,
)

ALL_CONTEXTS = (4096, 8192, 16384, 32768, 65536, 131072)


def _enter(env: Env, name: str, context: int):
    m = Miner(env, name, coldkey=f"ck-{name}")
    files = checkpoint(name)
    r = m.submit(files, context=context)
    assert r.status == 201, (context, r)
    assert r.json["context"] == context
    m.upload(files)
    rdy = m.ready(files, context=context)
    return m, files, rdy


def test_every_rev66_context_is_declarable_with_the_default_config(tmp_path):
    """The default ``LaneDConfig`` (rev6.6: all six variants) accepts every
    variant at submit and at ready, and the ready record keeps the declared
    context (the scorer, not the intake, picks the context it runs at)."""
    env = Env(tmp_path)
    assert tuple(env.cfg.contexts) == ALL_CONTEXTS, (
        f"rev6.6: [lane_d] contexts defaults to every variant {ALL_CONTEXTS}; "
        f"got {tuple(env.cfg.contexts)}")
    for c in ALL_CONTEXTS:
        m, files, r = _enter(env, f"ctx{c}", c)
        assert r.status == 200, (c, r)
        assert r.json["context"] == c
        assert r.json["commit_payload"].startswith(f"metro-v1:weights:c{c}:"), r.json
        row = env.entry(m.hotkey, digest_of(manifest(files)))
        assert row["context"] == c and row["state"] == "ready"


def test_an_unarmed_variant_in_contexts_is_accepted(tmp_path):
    """Intake checks only membership in ``contexts``; a long variant that no
    settlement has armed yet is a valid declaration."""
    env = Env(tmp_path, cfg=lane_d_cfg(contexts=(4096, 131072)))
    m, files, r = _enter(env, "unarmed", 131072)
    assert r.status == 200, r
    assert r.json["context"] == 131072


def test_a_context_not_in_contexts_is_refused(tmp_path):
    env = Env(tmp_path, cfg=lane_d_cfg(contexts=(4096, 8192)))
    m = Miner(env, "outside")
    for c in (16384, 131072, 262144, 5000):
        r = m.submit(checkpoint(f"out{c}"), context=c)
        assert r.status == 400 and r.code == "context_not_armed", (c, r)
    assert env.minter.minted == []
