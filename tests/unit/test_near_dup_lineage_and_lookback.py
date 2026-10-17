"""2026-10-17 live miss: the near-dup guard (enforce 0.985) let a sprayed
coldkey through because its variants are ~all PUBLISHED-champion lineage
(18,701 shingles, 75 private after subtracting the 11 champions — under
``near_dup_min_private_shingles``, so the private-only comparison failed
open), and the lookback counted SETTLEMENTS (900 blocks) not eras (3600).
Fixes: ``near_dup_same_coldkey`` compares one coldkey's entries on their full
trees; the lookback is in eras; ``near_dup_lookback_eras = 0`` = unlimited;
a coldkey-cap deferral records its reason on the queue entry."""

from __future__ import annotations

import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.funding.queue import FundedQueue
from cascade.shared.config import load_chain_config
from cascade.trainer.dedup_registry import NEAR_DUP_TIER
from tests.unit.test_admission_near_dup_guard import _admit, _body, _reg, _tree
from tests.unit.test_rolling import REF, Clock, _armed, _ccommit, _join, _sched

CHAMP = "vault/direct@sha256:" + "f" * 64


@pytest.fixture
def trees(monkeypatch):
    """ref -> tree dir; fetch_from_hub resolves refs to these (text only)."""
    src: dict[str, Path] = {}
    import cascade.shared.hippius as H

    monkeypatch.setattr(H, "fetch_from_hub", lambda ref, root, hub=None: src[ref])
    return src


def _champion(reg, trees, tmp_path, src):
    trees[CHAMP] = _tree(tmp_path, "champ", src)
    reg.register(CHAMP, hotkey="KING", commit_block=0, kind="champion",
                 digests={"tree_sha256": "t", "token_sha256": "k", "masked_sha256": "m"})
    reg.compute_nd_sig(CHAMP)


def _variant(trees, tmp_path, name, src, tweak):
    """The champion plus a TINY own layer (far under min_private shingles)."""
    ref = "vault/direct@sha256:" + name * 64
    shapes = {"l2infra": lambda i: f"round(c_{i}) + held(c_{i - 1})",
              "l2cau": lambda i: f"relu(p_{i} * w) if p_{i} else sigmoid(p_{i - 1})",
              "mine": lambda i: f"clip(m_{i}, lo, hi) - drift[m_{i}]",
              "yours": lambda i: f"[s for s in season(y_{i})][::2]"}
    own = "\n".join(f"{tweak}_{i} = " + shapes[tweak](i) for i in range(1, 6 + len(tweak)))
    trees[ref] = _tree(tmp_path, name, src + own + "\n")
    return ref


def test_same_coldkey_spray_on_champion_lineage_is_refused(tmp_path, trees):
    src = _body(2000)
    reg = _reg(tmp_path, near_dup_threshold=0.985, near_dup_same_coldkey=True)
    _champion(reg, trees, tmp_path, src)
    a = _variant(trees, tmp_path, "a", src, "l2infra")
    b = _variant(trees, tmp_path, "b", src, "l2cau")
    assert _admit(reg, "ALFA", a, 1000, ck="CK-SPRAY") is None
    assert _admit(reg, "BRAV", b, 1010, ck="CK-SPRAY") == ("ALFA", NEAR_DUP_TIER, True)


def test_cross_coldkey_champion_tweakers_are_never_flagged(tmp_path, trees):
    src = _body(2000)
    reg = _reg(tmp_path, near_dup_threshold=0.985, near_dup_same_coldkey=True)
    _champion(reg, trees, tmp_path, src)
    a = _variant(trees, tmp_path, "a", src, "mine")
    b = _variant(trees, tmp_path, "b", src, "yours")
    assert _admit(reg, "ALFA", a, 1000, ck="CK-ONE") is None
    assert _admit(reg, "BRAV", b, 1010, ck="CK-TWO") is None      # building on the king


def test_same_coldkey_rule_is_inert_by_default(tmp_path, trees):
    src = _body(2000)
    reg = _reg(tmp_path, near_dup_threshold=0.985)
    _champion(reg, trees, tmp_path, src)
    a = _variant(trees, tmp_path, "a", src, "l2infra")
    b = _variant(trees, tmp_path, "b", src, "l2cau")
    assert _admit(reg, "ALFA", a, 1000, ck="CK-SPRAY") is None
    assert _admit(reg, "BRAV", b, 1010, ck="CK-SPRAY") is None    # private part too small


def test_lookback_counts_eras_not_settlements(tmp_path, trees):
    a, b = "vault/direct@sha256:" + "a" * 64, "vault/direct@sha256:" + "b" * 64
    trees[a] = _tree(tmp_path, "a", _body(2000))
    trees[b] = _tree(tmp_path, "b", _body(2000, "b", changed=(5,)))
    # the post-switch grid: 900-block settlements, 4 per era ⇒ an era is 3600
    reg = _reg(tmp_path, near_dup_lookback_eras=1, epoch_blocks=900, era_settlements=4)
    _admit(reg, "ALFA", a, 10_000)
    assert reg._nd_lookback_blocks(12_000) == 3600
    assert _admit(reg, "BRAV", b, 10_000 + 2000) == ("ALFA", NEAR_DUP_TIER, True)


def test_lookback_zero_means_unlimited(tmp_path, trees):
    a, b = "vault/direct@sha256:" + "a" * 64, "vault/direct@sha256:" + "b" * 64
    trees[a] = _tree(tmp_path, "a", _body(2000))
    trees[b] = _tree(tmp_path, "b", _body(2000, "b", changed=(5,)))
    reg = _reg(tmp_path, near_dup_lookback_eras=0)
    _admit(reg, "ALFA", a, 1000)
    assert reg._nd_lookback_blocks(500_000) is None
    assert _admit(reg, "BRAV", b, 500_000) == ("ALFA", NEAR_DUP_TIER, True)


def test_loader_keeps_lookback_zero_and_parses_same_coldkey(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "chain.toml").read_text(encoding="utf-8")
    block = textwrap.dedent("""\
        near_dup_lookback_eras = 0
        near_dup_same_coldkey = true
        """)
    p = tmp_path / "chain.toml"
    p.write_text(text.replace("[round]\n", "[round]\n" + block, 1), encoding="utf-8")
    r = load_chain_config(p).round
    assert (r.near_dup_lookback_eras, r.near_dup_same_coldkey) == (0, True)
    d = load_chain_config(repo / "chain.toml").round
    assert (d.near_dup_lookback_eras, d.near_dup_same_coldkey) == (4, False)


def test_queue_note_records_a_reason_without_changing_state(tmp_path):
    q = FundedQueue(tmp_path / "fq.json", entry_ttl_seconds=3600.0)
    q.add("hkA", REF["ALFA"], reveal_block=10)
    assert q.note("hkA", error="deferred: cap", error_class="deferred")
    e = q.get("hkA")
    assert (e.status, e.last_error_class, e.last_error) == ("queued", "deferred", "deferred: cap")
    assert not q.note("hkA", error="deferred: cap", error_class="deferred")   # idempotent


def test_coldkey_cap_deferral_is_visible_on_the_entry(cfg, tmp_path):
    armed = _armed(cfg)
    b0 = armed.round.rolling_from_block + 5
    armed = replace(armed, round=replace(armed.round, max_inflight_per_coldkey=1))
    sched, ops = _sched(armed, tmp_path, Clock())
    q = ops.queue()
    ops.commits += [_ccommit("ALFA", REF["ALFA"], b0 - 100, "COLD-A"),
                    _ccommit("BRAV", REF["BRAV"], b0 - 90, "COLD-A")]
    for hk, rb in (("ALFA", b0 - 100), ("BRAV", b0 - 90)):
        q.add(hk, REF[hk], reveal_block=rb)
    ops.hold = {"ALFA": __import__("threading").Event()}
    from tests.unit.test_rolling import FakeClient

    sched.tick(FakeClient(), b0)
    e = q.get("BRAV")
    assert e.status == "queued" and e.last_error_class == "deferred"
    assert "max_inflight_per_coldkey=1" in e.last_error
    for ev in ops.hold.values():
        ev.set()
    _join(sched)
