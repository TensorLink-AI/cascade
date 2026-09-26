"""Embedded tiers on the rolling registry (DEC-CA-0008, amended 2026-09-26):
a packed source equal to another entry's whole code — or its packed source —
is an exact duplicate in either direction; the registry holds the era king and
every published champion so newcomers are judged against them too."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cascade.shared.chain import Commitment
from cascade.trainer.dedup_registry import DedupRegistry

VAULT_A = "vault/direct@sha256:" + "a" * 64
VAULT_B = "vault/direct@sha256:" + "b" * 64
VAULT_K = "vault/direct@sha256:" + "e" * 64      # the king's ref
CHAMP_D = "c" * 64                               # a published champion's digest
VAULT_C = "vault/direct@sha256:" + CHAMP_D


def _commit(hk, ref, block):
    return Commitment(uid=1, hotkey=hk, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


def _gen(hk, ref, rb):
    return SimpleNamespace(hotkey=hk, ref=ref, reveal_block=rb)


def _plain(tag: str) -> dict:
    """Digests of a one-file repo whose .py stream is <tag>."""
    return {"tree_sha256": f"T{tag}", "token_sha256": f"K{tag}", "masked_sha256": f"M{tag}",
            "py_sha256": f"PY{tag}", "py_masked_sha256": f"PM{tag}", "components": []}


def _wrapper(tag: str, *inner: str, renamed: bool = False) -> dict:
    """Digests of a wrapper whose packed sources are the <inner> plain repos."""
    comps = [{"name": f"generator.py#packed{i}", "n_tokens": 5000,
              "token_sha256": f"PY{t}" if not renamed else f"PYx{t}",
              "masked_sha256": f"PM{t}"} for i, t in enumerate(inner, 1)]
    return {"tree_sha256": f"T{tag}", "token_sha256": f"K{tag}", "masked_sha256": f"M{tag}",
            "py_sha256": f"PY{tag}", "py_masked_sha256": f"PM{tag}", "components": comps}


def _registry(tmp_path, prints, *, mode="enforce", emb_mode="enforce", published=(),
              store_fails=False):
    def get_text(key):
        if store_fails:
            raise RuntimeError("store down")
        return json.dumps({"champions": [{"digest": d, "hotkey": hk} for d, hk in published]})
    runner = SimpleNamespace(
        cfg=SimpleNamespace(round=SimpleNamespace(
            dedup_max_tokens=50_000, dedup_max_text_mb=4, private_copy_mode="off",
            private_copy_min_containment=0.8, private_copy_min_tokens=2000,
            dedup_embedded_mode=emb_mode)),
        work_root=tmp_path,
        manifest_store=lambda: SimpleNamespace(get_text=get_text),
    )
    reg = DedupRegistry(runner, tmp_path / "dedup_registry.json", mode=mode)
    reg.fingerprint = lambda ref: prints.get(ref)
    return reg


def test_wrapper_of_earlier_plain_entry_is_dropped(tmp_path):
    reg = _registry(tmp_path, {VAULT_A: _plain("a"), VAULT_B: _wrapper("b", "a")})
    hist = [_commit("A", VAULT_A, 100), _commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("A", VAULT_A, 100), None, hist) is None
    assert reg.admit(_gen("B", VAULT_B, 200), None, hist) == ("A", "embedded_token_identical", True)
    assert VAULT_B not in reg.entries          # enforce: not registered


def test_plain_resubmission_of_someones_blob_is_dropped(tmp_path):
    reg = _registry(tmp_path, {VAULT_A: _wrapper("a", "z"), VAULT_B: _plain("z")})
    hist = [_commit("A", VAULT_A, 100), _commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("A", VAULT_A, 100), None, hist) is None
    assert reg.admit(_gen("B", VAULT_B, 200), None, hist) == ("A", "embedded_token_identical", True)


def test_renamed_blob_is_embedded_rename_and_shadow_admits(tmp_path):
    reg = _registry(tmp_path, {VAULT_A: _plain("a"), VAULT_B: _wrapper("b", "a", renamed=True)},
                    emb_mode="shadow")
    hist = [_commit("A", VAULT_A, 100), _commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("A", VAULT_A, 100), None, hist) is None
    assert reg.admit(_gen("B", VAULT_B, 200), None, hist) == ("A", "embedded_rename_identical", False)
    assert VAULT_B in reg.entries              # shadow: on record for the next comer


def test_earlier_commit_keeps_the_entry(tmp_path):
    # B committed BEFORE A: B is not a copy of A even though A is registered first
    reg = _registry(tmp_path, {VAULT_A: _plain("a"), VAULT_B: _wrapper("b", "a")})
    hist = [_commit("A", VAULT_A, 300), _commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("A", VAULT_A, 300), None, hist) is None
    assert reg.admit(_gen("B", VAULT_B, 200), None, hist) is None
    assert VAULT_B in reg.entries


def test_off_mode_ignores_components(tmp_path):
    reg = _registry(tmp_path, {VAULT_A: _plain("a"), VAULT_B: _wrapper("b", "a")}, emb_mode="off")
    hist = [_commit("A", VAULT_A, 100), _commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("A", VAULT_A, 100), None, hist) is None
    assert reg.admit(_gen("B", VAULT_B, 200), None, hist) is None


def test_king_is_registered_and_wrappers_of_it_are_caught(tmp_path):
    reg = _registry(tmp_path, {VAULT_K: _plain("k"), VAULT_B: _wrapper("b", "k")})
    era = SimpleNamespace(king_ref=VAULT_K, king_hotkey="KING")
    hist = [_commit("KING", VAULT_K, 50), _commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("B", VAULT_B, 200), era, hist) == ("KING", "embedded_token_identical", True)
    rec = reg.entries[VAULT_K]
    assert rec["kind"] == "king" and rec["hotkey"] == "KING" and rec["commit_block"] == 50
    # the king re-submitting its own code is not a copy of itself
    assert reg.admit(_gen("KING", VAULT_K, 50), era, hist) is None


def test_published_champions_are_registered(tmp_path):
    reg = _registry(tmp_path, {VAULT_C: _plain("c"), VAULT_B: _wrapper("b", "c")},
                    published=[(CHAMP_D, "CHAMP"), ("not-a-digest", "X")])
    hist = [_commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("B", VAULT_B, 200), None, hist) == ("CHAMP", "embedded_token_identical", True)
    rec = reg.entries[VAULT_C]
    assert rec["kind"] == "champion" and rec["commit_block"] == 0
    assert len([r for r in reg.entries.values() if r["kind"] == "champion"]) == 1


def test_champion_index_failure_is_fail_open(tmp_path):
    reg = _registry(tmp_path, {VAULT_B: _plain("b")}, store_fails=True)
    assert reg.admit(_gen("B", VAULT_B, 200), None, [_commit("B", VAULT_B, 200)]) is None
    assert VAULT_B in reg.entries


def test_pre_tier_entries_are_backfilled(tmp_path):
    reg = _registry(tmp_path, {VAULT_A: _plain("a"), VAULT_B: _wrapper("b", "a")})
    # an entry written before the embedded tiers existed: no components/py digests
    reg.entries[VAULT_A] = {"tree_sha256": "Ta", "token_sha256": "Ka", "masked_sha256": "Ma",
                            "hotkey": "A", "commit_block": 100, "kind": "queue", "private": True}
    reg._save()
    hist = [_commit("A", VAULT_A, 100), _commit("B", VAULT_B, 200)]
    assert reg.admit(_gen("B", VAULT_B, 200), None, hist) == ("A", "embedded_token_identical", True)
    assert reg.entries[VAULT_A]["py_sha256"] == "PYa" and reg.entries[VAULT_A]["components"] == []


def test_config_round_trip(tmp_path):
    import re

    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    src = DEFAULT_CHAIN_TOML.read_text()
    p = tmp_path / "chain.toml"
    p.write_text(src)
    assert load_chain_config(p).round.dedup_embedded_mode == "shadow"

    def armed(extra: str) -> str:
        out = re.sub(r"(?m)^(dedup_mode\s*=.*)$", lambda m: m.group(1) + "\n" + extra, src, count=1)
        assert out != src
        return out

    p.write_text(armed("dedup_embedded_mode = 'enforce'"))
    assert load_chain_config(p).round.dedup_embedded_mode == "enforce"
    p.write_text(armed("dedup_embedded_mode = 'loud'"))
    with pytest.raises(ValueError, match="dedup_embedded_mode"):
        load_chain_config(p)
