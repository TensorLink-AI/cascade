"""private_copy tier: a module lifted from an earlier, never-published vault
submission (renamed, repacked, wrapped in another tree) is flagged; public
material never is."""

from __future__ import annotations

import base64
import json
import zlib
from types import SimpleNamespace

import pytest

from cascade.interface.dedup import module_sketches, sketch_containment, sketch_source
from cascade.shared.chain import Commitment
from cascade.trainer.dedup_registry import DedupRegistry

# ── a "module" big enough to own: ~2.6k masked tokens of distinctive code ─────


def _module(prefix: str, n: int = 90) -> str:
    lines = ["import numpy as np", ""]
    for i in range(n):
        lines += [
            f"def {prefix}_step_{i}({prefix}_x, {prefix}_y, {prefix}_z):",
            f"    {prefix}_a = np.sin({prefix}_x * {i + 1}) + np.cos({prefix}_y / {i + 2})",
            f"    {prefix}_b = np.where({prefix}_a > {i % 7}, {prefix}_z ** 2, -{prefix}_z)",
            f"    return np.cumsum({prefix}_b)[{i % 5}:] * {i % 3 + 1}",
            "",
        ]
    return "\n".join(lines)


def _rename(src: str, old: str, new: str) -> str:
    return src.replace(old, new)


def _lineage(prefix: str = "pub") -> str:
    # a second, unrelated module — the "public lineage" everyone builds on
    lines = ["import numpy as np", ""]
    for i in range(90):
        lines += [
            f"class {prefix}Block{i}:",
            f"    def run(self, {prefix}_v, {prefix}_w):",
            f"        {prefix}_t = np.tanh({prefix}_v - {i}) * np.exp(-{prefix}_w / {i + 3})",
            f"        return np.clip({prefix}_t, -{i % 4 + 1}, {i % 6 + 1}).mean()",
            "",
        ]
    return "\n".join(lines)


def _repo(tmp_path, name, files: dict[str, str]):
    d = tmp_path / name
    d.mkdir(parents=True)
    for rel, body in files.items():
        f = d / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    return d


def _packed(src: str, how: str = "b64z") -> str:
    b = src.encode()
    blob = {"plain": repr(src),
            "b64": repr(base64.b64encode(b).decode()),
            "b64z": repr(base64.b64encode(zlib.compress(b)).decode())}[how]
    return f"import base64, zlib\n_P = {blob}\nexec(compile(_P, '<p>', 'exec'))\n"


# ── sketches ─────────────────────────────────────────────────────────────────


def test_rename_only_copy_is_fully_contained():
    a = _module("alpha")
    n, sa = sketch_source(a)
    assert n >= 2000 and len(sa) > 50
    _, sb = sketch_source(_rename(a, "alpha", "zulu"))
    assert sketch_containment(sb, sa) == 1.0 and sketch_containment(sa, sb) == 1.0


def test_unrelated_modules_do_not_overlap():
    _, sa = sketch_source(_module("alpha"))
    _, sb = sketch_source(_lineage())
    assert sketch_containment(sa, sb) < 0.05 and sketch_containment(sb, sa) < 0.05


def test_small_helpers_are_not_modules():
    n, sk = sketch_source("import numpy as np\n\ndef f(x):\n    return np.sin(x)\n")
    assert n < 2000 and sk == []


def test_sketch_is_stable_across_processes():
    # BLAKE2b, not hash(): the sketch persists in the registry across restarts.
    import subprocess
    import sys
    from pathlib import Path

    src = _module("alpha")
    _, here = sketch_source(src)
    root = str(Path(__file__).resolve().parents[2])
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]); "
            "from cascade.interface.dedup import sketch_source; "
            "print(json.dumps(sketch_source(sys.stdin.read())[1]))")
    out = subprocess.run([sys.executable, "-c", code, root], input=src,
                         capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == here


@pytest.mark.parametrize("how", ["plain", "b64", "b64z"])
def test_packed_modules_are_sketched(tmp_path, how):
    inner = _module("inner")
    d = _repo(tmp_path, f"packed-{how}", {"generator.py": _packed(inner, how), "config.json": "{}"})
    mods = module_sketches(d)
    names = [m["name"] for m in mods]
    assert "generator.py#packed1" in names
    _, direct = sketch_source(inner)
    packed = next(m for m in mods if m["name"] == "generator.py#packed1")
    assert sketch_containment(packed["sketch"], direct) == 1.0
    # the outer wrapper is a handful of tokens (the packed string is ONE token)
    assert "generator.py" not in names


def test_nested_packing_and_sibling_files(tmp_path):
    inner = _module("inner")
    middle = _packed(inner, "b64") + "\n" + _lineage("mid")
    d = _repo(tmp_path, "nested", {"generator.py": _packed(middle, "b64z"),
                                   "lib/helper.py": _module("help", 60), "config.json": "{}"})
    names = {m["name"] for m in module_sketches(d)}
    assert {"generator.py#packed1", "generator.py#packed2", "lib/helper.py"} <= names


def test_junk_dirs_are_skipped(tmp_path):
    d = _repo(tmp_path, "junk", {"generator.py": _module("g"), "__pycache__/x.py": _module("x")})
    assert [m["name"] for m in module_sketches(d)] == ["generator.py"]


# ── registry: the private_copy tier ──────────────────────────────────────────

VAULT_A = "vault/direct@sha256:" + "a" * 64      # A's private submission
VAULT_B = "vault/direct@sha256:" + "b" * 64      # B's later submission
VAULT_C = "vault/direct@sha256:" + "c" * 64
PUBLIC_P = "pub/gen@sha256:" + "d" * 64          # a public Hub generator


def _commit(hk, ref, block):
    return Commitment(uid=1, hotkey=hk, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


def _gen(hk, ref, rb):
    return SimpleNamespace(hotkey=hk, ref=ref, reveal_block=rb)


def _digests(tag: str, *sources: str) -> dict:
    mods = []
    for i, src in enumerate(sources):
        n, sk = sketch_source(src)
        mods.append({"name": f"m{i}.py", "tokens": n, "sketch": sk})
    return {"tree_sha256": f"T{tag}", "token_sha256": f"K{tag}", "masked_sha256": f"M{tag}",
            "modules": mods}


def _registry(tmp_path, prints, *, mode="enforce", pc_mode="enforce", thr=0.8,
              published=()):
    runner = SimpleNamespace(
        cfg=SimpleNamespace(round=SimpleNamespace(
            dedup_max_tokens=50_000, dedup_max_text_mb=4, private_copy_mode=pc_mode,
            private_copy_min_containment=thr, private_copy_min_tokens=2000)),
        work_root=tmp_path,
        manifest_store=lambda: SimpleNamespace(get_text=lambda key: json.dumps(
            {"champions": [{"digest": d} for d in published]})),
    )
    reg = DedupRegistry(runner, tmp_path / "dedup_registry.json", mode=mode)
    reg.fingerprint = lambda ref: prints.get(ref)
    return reg


def test_wrapped_renamed_private_module_is_a_private_copy(tmp_path):
    a_mod, lineage = _module("alpha"), _lineage()
    prints = {VAULT_A: _digests("a", a_mod),
              VAULT_B: _digests("b", lineage, _rename(a_mod, "alpha", "zulu"))}
    reg = _registry(tmp_path, prints)
    history = [_commit("ALFA", VAULT_A, 100), _commit("BRAV", VAULT_B, 200)]
    assert reg.admit(_gen("ALFA", VAULT_A, 100), None, history) is None
    assert reg.admit(_gen("BRAV", VAULT_B, 200), None, history) == ("ALFA", "private_copy", True)
    assert VAULT_B not in reg.entries          # enforce: the copy is not registered


def test_shadow_flags_admits_and_registers(tmp_path):
    a_mod = _module("alpha")
    prints = {VAULT_A: _digests("a", a_mod), VAULT_B: _digests("b", _lineage(), a_mod)}
    reg = _registry(tmp_path, prints, pc_mode="shadow")
    history = [_commit("ALFA", VAULT_A, 100), _commit("BRAV", VAULT_B, 200)]
    assert reg.admit(_gen("ALFA", VAULT_A, 100), None, history) is None
    assert reg.admit(_gen("BRAV", VAULT_B, 200), None, history) == ("ALFA", "private_copy", False)
    assert VAULT_B in reg.entries


def test_off_never_consults_the_tier(tmp_path):
    a_mod = _module("alpha")
    prints = {VAULT_A: _digests("a", a_mod), VAULT_B: _digests("b", a_mod)}
    reg = _registry(tmp_path, prints, pc_mode="off")
    history = [_commit("ALFA", VAULT_A, 100), _commit("BRAV", VAULT_B, 200)]
    assert reg.admit(_gen("ALFA", VAULT_A, 100), None, history) is None
    # different tree/token/masked digests ⇒ the exact tiers pass; tier off ⇒ admitted
    assert reg.admit(_gen("BRAV", VAULT_B, 200), None, history) is None


def test_copying_public_material_is_never_flagged(tmp_path):
    lineage = _lineage()
    # P: a public Hub generator carrying the lineage. A: a vault entry that
    # wraps the SAME lineage (public material inside a private submission).
    # B: another vault entry built on the lineage. A does not own it.
    prints = {PUBLIC_P: _digests("p", lineage), VAULT_A: _digests("a", lineage, _module("own")),
              VAULT_B: _digests("b", _rename(lineage, "pub", "mine"))}
    reg = _registry(tmp_path, prints)
    history = [_commit("PUBL", PUBLIC_P, 10), _commit("ALFA", VAULT_A, 100),
               _commit("BRAV", VAULT_B, 200)]
    assert reg.admit(_gen("PUBL", PUBLIC_P, 10), None, history) is None
    assert reg.admit(_gen("ALFA", VAULT_A, 100), None, history) is None
    assert reg.admit(_gen("BRAV", VAULT_B, 200), None, history) is None


def test_published_champion_code_is_public(tmp_path):
    a_mod = _module("alpha")
    prints = {VAULT_A: _digests("a", a_mod), VAULT_B: _digests("b", a_mod)}
    # A's vault digest was published by the champion policy ⇒ public ⇒ copyable
    reg = _registry(tmp_path, prints, published=("a" * 64,))
    history = [_commit("ALFA", VAULT_A, 100), _commit("BRAV", VAULT_B, 200)]
    assert reg.admit(_gen("ALFA", VAULT_A, 100), None, history) is None
    assert reg.admit(_gen("BRAV", VAULT_B, 200), None, history) is None


def test_same_hotkey_and_earlier_commit_are_not_copies(tmp_path):
    a_mod = _module("alpha")
    prints = {VAULT_A: _digests("a", a_mod), VAULT_B: _digests("b", a_mod, _lineage()),
              VAULT_C: _digests("c", a_mod)}
    reg = _registry(tmp_path, prints)
    history = [_commit("ALFA", VAULT_A, 100), _commit("ALFA", VAULT_B, 200),
               _commit("CHAR", VAULT_C, 50)]
    assert reg.admit(_gen("ALFA", VAULT_A, 100), None, history) is None
    # the same hotkey iterating on its own private code
    assert reg.admit(_gen("ALFA", VAULT_B, 200), None, history) is None
    # committed EARLIER than A: not a copy of A (A would have been, had it come later)
    assert reg.admit(_gen("CHAR", VAULT_C, 50), None, history) is None


def test_partial_reuse_below_threshold_passes(tmp_path):
    a_mod = _module("alpha")
    half = "\n".join(a_mod.splitlines()[: len(a_mod.splitlines()) // 2]) + "\n" + _module("beta", 45)
    prints = {VAULT_A: _digests("a", a_mod), VAULT_B: _digests("b", half)}
    reg = _registry(tmp_path, prints)
    history = [_commit("ALFA", VAULT_A, 100), _commit("BRAV", VAULT_B, 200)]
    assert reg.admit(_gen("ALFA", VAULT_A, 100), None, history) is None
    assert reg.admit(_gen("BRAV", VAULT_B, 200), None, history) is None


def test_backfill_sketches_pre_tier_private_entries(tmp_path):
    a_mod = _module("alpha")
    # a registry written before the tier existed: no "modules", no "private"
    (tmp_path / "dedup_registry.json").write_text(json.dumps({"refs": {
        VAULT_A: {"tree_sha256": "Ta", "token_sha256": "Ka", "masked_sha256": "Ma",
                  "hotkey": "ALFA", "commit_block": 100, "kind": "queue"}}}))
    prints = {VAULT_A: _digests("a", a_mod), VAULT_B: _digests("b", a_mod, _lineage())}
    reg = _registry(tmp_path, prints)
    history = [_commit("ALFA", VAULT_A, 100), _commit("BRAV", VAULT_B, 200)]
    assert reg.admit(_gen("BRAV", VAULT_B, 200), None, history) == ("ALFA", "private_copy", True)
    assert reg.entries[VAULT_A]["modules"] and reg.entries[VAULT_A]["private"] is True


def test_config_round_trip(tmp_path):
    import re

    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    src = DEFAULT_CHAIN_TOML.read_text()
    p = tmp_path / "chain.toml"
    p.write_text(src)
    rnd = load_chain_config(p).round
    assert (rnd.private_copy_mode, rnd.private_copy_min_containment,
            rnd.private_copy_min_tokens) == ("shadow", 0.8, 2000)

    def armed(extra: str) -> str:
        out = re.sub(r"(?m)^(dedup_mode\s*=.*)$", lambda m: m.group(1) + "\n" + extra, src, count=1)
        assert out != src
        return out

    p.write_text(armed("private_copy_mode = 'enforce'\nprivate_copy_min_containment = 0.9\n"
                       "private_copy_min_tokens = 3000"))
    rnd = load_chain_config(p).round
    assert (rnd.private_copy_mode, rnd.private_copy_min_containment,
            rnd.private_copy_min_tokens) == ("enforce", 0.9, 3000)
    p.write_text(armed("private_copy_mode = 'loud'"))
    with pytest.raises(ValueError, match="private_copy_mode"):
        load_chain_config(p)
