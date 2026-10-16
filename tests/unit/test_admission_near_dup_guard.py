"""Admission guards against one operator spraying near-identical variants
(DEC-CA-0048 amendment 2026-10-16): the per-coldkey seat cap (defer) and the
near-duplicate tier (shadow / enforce, no burn), at intake AND at rent time."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.interface.near_dup import (
    normalized_lines,
    private_jaccard,
    shingle_signature,
    signature_of_dir,
)
from cascade.provision.core import LaunchSpec, PodAddress
from cascade.provision.funded import rent_funded_pod
from cascade.shared.config import load_chain_config
from cascade.trainer import dedup_registry as DR
from cascade.trainer.dedup_registry import NEAR_DUP_TIER, DedupRegistry
from tests.unit.test_rolling import (
    REF,
    Clock,
    FakeClient,
    _armed,
    _ccommit,
    _join,
    _sched,
)

# ── signatures (text only) ───────────────────────────────────────────────────


def _body(n: int, tag: str = "", changed: tuple[int, ...] = ()) -> str:
    """A synthetic n-line generator: distinct identifiers per line (numbers
    inside identifiers are not literals), numeric literals that a constant
    sweep may change, optional changed lines."""
    lines = ['"""A generator docstring that must not count."""', "import numpy as np"]
    for i in range(n):
        if i in changed:
            lines.append(f"other_{tag}_{i} = [v_{i}, (v_{i},)] if v_{i} else {{}}  # tweak")
        else:
            lines.append(f"v_{i + 1} = np.sin(v_{i}) * {0.5 + i % 7}  # comment {i}")
    return "\n".join(lines) + "\n"


def _tree(tmp: Path, name: str, src: str) -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "generator.py").write_text(src, encoding="utf-8")
    return d


def test_normalisation_masks_numbers_comments_and_docstrings():
    a = normalized_lines('"""doc"""\nx = 1.5  # c\n\n  y=f(x,  2e-3)\n')
    b = normalized_lines("x = 9.25\ny=f(x, 7)\n")
    assert a == b == ["x = 0", "y=f(x, 0)"]
    # identifiers that contain digits are not literals
    assert normalized_lines("v_12 = 3\n") == ["v_12 = 0"]


def test_constant_sweep_is_identical_and_small_edit_is_near(tmp_path):
    base = signature_of_dir(_tree(tmp_path, "a", _body(2000)))
    sweep = _body(2000).replace("* 0.5", "* 0.75")
    assert signature_of_dir(_tree(tmp_path, "b", sweep)) == base
    near = signature_of_dir(_tree(tmp_path, "c", _body(2000, "c", changed=(10, 900, 1500))))
    j = private_jaccard(base, near, frozenset(), min_private=200)
    assert 0.97 <= j < 1.0


def test_private_part_excludes_public_lineage():
    king = shingle_signature([f"k{i}" for i in range(1000)])
    a = king | shingle_signature([f"a{i}" for i in range(300)])
    b = king | shingle_signature([f"b{i}" for i in range(300)])
    assert private_jaccard(a, b, frozenset(), min_private=200) > 0.5      # raw: lineage
    assert private_jaccard(a, b, king, min_private=200) == 0.0            # private: unrelated
    # a tree that is nearly all public has nothing of its own to compare
    tiny = king | shingle_signature(["x1", "x2", "x3"])
    assert private_jaccard(tiny, a, king, min_private=200) is None


# ── registry tier ────────────────────────────────────────────────────────────


def _rcfg(**kw):
    base = dict(dedup_max_tokens=50_000, dedup_max_text_mb=4, private_copy_mode="off",
                dedup_embedded_mode="off", near_dup_mode="enforce", near_dup_threshold=0.97,
                near_dup_lookback_eras=4, near_dup_min_private_shingles=200,
                epoch_blocks=3600, private_copy_min_containment=0.8,
                private_copy_min_tokens=2000)
    base.update(kw)
    return SimpleNamespace(round=SimpleNamespace(**base))


class _Runner:
    def __init__(self, tmp, cfg):
        self.work_root = tmp / "work"
        self.cfg = cfg

    def hub(self):
        return None

    def _archive_generator_tree(self, ref, d):
        pass

    def manifest_store(self):
        raise RuntimeError("no store in tests")


@pytest.fixture
def trees(tmp_path, monkeypatch):
    """ref -> tree dir; fetch_from_hub resolves refs to these (text only)."""
    src: dict[str, Path] = {}
    import cascade.shared.hippius as H

    monkeypatch.setattr(H, "fetch_from_hub", lambda ref, root, hub=None: src[ref])
    return src


def _reg(tmp_path, **kw):
    return DedupRegistry(_Runner(tmp_path, _rcfg(**kw)), tmp_path / "work" / "reg.json",
                         mode="enforce")


def _gen(hk, ref, block):
    return SimpleNamespace(hotkey=hk, ref=ref, reveal_block=block)


def _hist(*rows):
    return [_ccommit(hk, ref, b, ck) for hk, ref, b, ck in rows]


def _admit(reg, hk, ref, block, ck="CK-" + "x"):
    return reg.admit(_gen(hk, ref, block), SimpleNamespace(king_ref="", king_hotkey=""),
                     _hist((hk, ref, block, ck)))


def test_near_duplicate_enforce_refuses_later_commit_and_earliest_keeps(tmp_path, trees):
    trees["vault/direct@sha256:" + "a" * 64] = _tree(tmp_path, "a", _body(2000))
    trees["vault/direct@sha256:" + "b" * 64] = _tree(tmp_path, "b", _body(2000, "b", changed=(5, 700)))
    reg = _reg(tmp_path)
    assert _admit(reg, "ALFA", "vault/direct@sha256:" + "a" * 64, 1000) is None
    hit = _admit(reg, "BRAV", "vault/direct@sha256:" + "b" * 64, 1010)
    assert hit == ("ALFA", NEAR_DUP_TIER, True)
    # refused entries are not registered (the next variant still meets ALFA)
    assert "vault/direct@sha256:" + "b" * 64 not in reg.entries


def test_near_duplicate_shadow_logs_and_admits(tmp_path, trees, monkeypatch):
    trees["vault/direct@sha256:" + "a" * 64] = _tree(tmp_path, "a", _body(2000))
    trees["vault/direct@sha256:" + "b" * 64] = _tree(tmp_path, "b", _body(2000, "b", changed=(5,)))
    reg = _reg(tmp_path, near_dup_mode="shadow")
    _admit(reg, "ALFA", "vault/direct@sha256:" + "a" * 64, 1000)
    lines: list[str] = []
    monkeypatch.setattr(DR.log, "warning", lambda msg, *a: lines.append(msg % a))
    hit = _admit(reg, "BRAV", "vault/direct@sha256:" + "b" * 64, 1010)
    assert hit == ("ALFA", NEAR_DUP_TIER, False)
    assert "vault/direct@sha256:" + "b" * 64 in reg.entries                  # admitted + on record
    assert any("near-duplicate shadow" in x and "J=" in x for x in lines)


def test_threshold_lookback_and_off_mode(tmp_path, trees):
    a, b = "vault/direct@sha256:" + "a" * 64, "vault/direct@sha256:" + "b" * 64
    trees[a] = _tree(tmp_path, "a", _body(2000))
    trees[b] = _tree(tmp_path, "b", _body(2000, "b", changed=tuple(range(0, 2000, 40))))
    # ~50 changed lines in 2000: below a 0.97 bar
    reg = _reg(tmp_path)
    _admit(reg, "ALFA", a, 1000)
    assert _admit(reg, "BRAV", b, 1010) is None
    # outside the lookback window: never compared
    trees["vault/direct@sha256:" + "c" * 64] = _tree(tmp_path, "c", _body(2000, "c", changed=(3,)))
    reg2 = _reg(tmp_path / "2", near_dup_lookback_eras=1)
    _admit(reg2, "ALFA", a, 1000)
    assert _admit(reg2, "CHAR", "vault/direct@sha256:" + "c" * 64, 1000 + 3600 + 1) is None
    # off: the tier never runs
    reg3 = _reg(tmp_path / "3", near_dup_mode="off")
    _admit(reg3, "ALFA", a, 1000)
    assert _admit(reg3, "CHAR", "vault/direct@sha256:" + "c" * 64, 1010) is None


def test_kings_and_champions_are_public_lineage_never_rivals(tmp_path, trees):
    king_src = _body(2000)
    k = "vault/direct@sha256:" + "f" * 64
    trees[k] = _tree(tmp_path, "k", king_src)
    reg = _reg(tmp_path)
    reg.register(k, hotkey="KING", commit_block=0, kind="champion",
                 digests={"tree_sha256": "t", "token_sha256": "k", "masked_sha256": "m"})
    reg.compute_nd_sig(k)
    # a copy of the champion with numbers changed: exact tiers are not tripped
    # (different digests) and the near-dup tier never flags building on it
    c = "vault/direct@sha256:" + "c" * 64
    trees[c] = _tree(tmp_path, "c", king_src.replace("* 0.5", "* 0.6") + "\n".join(
        f"own_{i} = g(own_{i - 1})" for i in range(1, 400)))
    assert _admit(reg, "CHAR", c, 1000) is None
    # two miners extending the champion with DIFFERENT own code: not near-dups
    d = "vault/direct@sha256:" + "d" * 64
    trees[d] = _tree(tmp_path, "d", king_src + "\n".join(
        f"mine_{i} = h(mine_{i - 1})" for i in range(1, 400)))
    assert _admit(reg, "BRAV", d, 1010) is None


def test_same_block_burst_keeps_exactly_one(tmp_path, trees):
    refs = {hk: "vault/direct@sha256:" + c * 64 for hk, c in (("ALFA", "a"), ("BRAV", "b"),
                                                                  ("CHAR", "c"))}
    for i, (hk, ref) in enumerate(refs.items()):
        trees[ref] = _tree(tmp_path, hk, _body(2000, hk, changed=(100 + i,)))
    reg = _reg(tmp_path)
    for hk, ref in refs.items():          # all committed in ONE block
        reg.register(ref, hotkey=hk, commit_block=5000, kind="queue",
                     digests={"tree_sha256": hk, "token_sha256": hk, "masked_sha256": hk})
    verdicts = {hk: reg.near_dup_verdict(ref, hotkey=hk, commit_block=5000)
                for hk, ref in refs.items()}
    assert verdicts["ALFA"] is None                       # (5000, "ALFA") is first
    assert verdicts["BRAV"][0] == "ALFA" and verdicts["BRAV"][2] == "enforce"
    assert verdicts["CHAR"][0] == "ALFA"


# ── rolling: coldkey cap (defer) + near-dup refusal path (no burn) ──────────


def test_coldkey_cap_defers_extra_entries_unburned(cfg, tmp_path):
    armed = _armed(cfg)
    b0 = armed.round.rolling_from_block + 5
    armed = replace(armed, round=replace(armed.round, max_inflight_per_coldkey=1))
    sched, ops = _sched(armed, tmp_path, Clock())
    q = ops.queue()
    ops.commits += [_ccommit("ALFA", REF["ALFA"], b0 - 100, "COLD-A"),
                    _ccommit("BRAV", REF["BRAV"], b0 - 90, "COLD-A"),
                    _ccommit("CHAR", REF["CHAR"], b0 - 80, "COLD-B")]
    for hk, rb in (("ALFA", b0 - 100), ("BRAV", b0 - 90), ("CHAR", b0 - 80)):
        q.add(hk, REF[hk], reveal_block=rb)
    ops.hold = {hk: __import__("threading").Event() for hk in ("ALFA", "CHAR")}
    sched.tick(FakeClient(), b0)
    assert q.get("ALFA").status == "in_flight"
    assert q.get("BRAV").status == "queued"              # deferred: same coldkey, cap 1
    assert q.get("CHAR").status == "in_flight"           # another coldkey seats
    assert "BRAV" not in ops.burnt
    for ev in ops.hold.values():
        ev.set()
    _join(sched)


def test_coldkey_cap_exempts_the_king_and_unknown_coldkeys(cfg, tmp_path):
    armed = _armed(cfg)
    b0 = armed.round.rolling_from_block + 5
    armed = replace(armed, round=replace(armed.round, max_inflight_per_coldkey=1))
    sched, ops = _sched(armed, tmp_path, Clock())
    sched.tick(FakeClient(), b0)
    _join(sched)
    king = sched.state.current.king_hotkey
    era = sched.state.current.index
    hist = [_ccommit(king, "x", b0 - 1, "COLD-A")]
    assert sched._coldkey_cap_check(SimpleNamespace(hotkey=king, ref="x"), era, hist) is False
    assert sched._coldkey_cap_check(SimpleNamespace(hotkey="ALFA", ref="x"), era, []) is False


class _NDReg:
    """dedup_registry stub: one near-duplicate verdict at intake/rent."""

    def __init__(self, matched="ALFA", mode="enforce"):
        self.matched, self.mode = matched, mode

    def admit(self, gen, era, history):
        if gen.hotkey == "BRAV":
            return self.matched, NEAR_DUP_TIER, self.mode == "enforce"
        return None

    def near_dup_verdict(self, ref, *, hotkey, commit_block):
        return (self.matched, 0.991, self.mode) if hotkey == "BRAV" else None


def test_near_duplicate_refusal_at_intake_never_burns(cfg, tmp_path):
    armed = _armed(cfg)
    b0 = armed.round.rolling_from_block + 5
    sched, ops = _sched(armed, tmp_path, Clock())
    ops.dedup_registry = lambda: _NDReg()
    q = ops.queue()
    ops.commits += [_ccommit("ALFA", REF["ALFA"], b0 - 100, "C1"),
                    _ccommit("BRAV", REF["BRAV"], b0 - 90, "C2")]
    for hk, rb in (("ALFA", b0 - 100), ("BRAV", b0 - 90)):
        q.add(hk, REF[hk], reveal_block=rb)
    sched.tick(FakeClient(), b0)
    _join(sched)
    e = q.get("BRAV")
    assert e.status == "failed" and e.last_error_class == "near_duplicate"
    assert "near-duplicate" in e.last_error and "BRAV" not in ops.burnt
    assert q.get("ALFA").status in ("in_flight", "done")


def test_rent_veto_refuses_near_dup_and_defers_over_cap(cfg, tmp_path):
    armed = _armed(cfg)
    b0 = armed.round.rolling_from_block + 5
    armed = replace(armed, round=replace(armed.round, max_inflight_per_coldkey=1))
    sched, ops = _sched(armed, tmp_path, Clock())
    q = ops.queue()
    ops.commits += [_ccommit("ALFA", REF["ALFA"], b0 - 100, "COLD-A"),
                    _ccommit("CHAR", REF["CHAR"], b0 - 80, "COLD-A"),
                    _ccommit("BRAV", REF["BRAV"], b0 - 90, "COLD-B")]
    sched.tick(FakeClient(), b0)
    _join(sched)
    era = sched.state.current.index
    # two legs of COLD-A already in flight (admitted before the cap existed)
    for hk in ("ALFA", "CHAR"):
        q.add(hk, REF[hk], reveal_block=b0 - 100)
        q.mark_in_flight(hk, REF[hk], target_boundary=b0 + 900, era_index=era,
                         started_block=b0)
    sched.__dict__["_last_history"] = list(ops.commits)
    # no sibling has a live pod yet: the first to rent proceeds
    assert sched.rent_veto("ALFA", REF["ALFA"], started=lambda hk: False) is None
    # once ALFA holds a pod, CHAR (same coldkey) defers — requeued, not failed
    v = sched.rent_veto("CHAR", REF["CHAR"], started=lambda hk: hk == "ALFA")
    assert v is not None and v[1] == "deferred" and v[2] is False
    # a near-duplicate is refused (miner's entry, never burned) at rent time
    ops.dedup_registry = lambda: _NDReg()
    q.add("BRAV", REF["BRAV"], reveal_block=b0 - 90)
    q.mark_in_flight("BRAV", REF["BRAV"], target_boundary=b0 + 900, era_index=era,
                     started_block=b0)
    v = sched.rent_veto("BRAV", REF["BRAV"], started=lambda hk: False)
    assert v is not None and v[1] == "near_duplicate" and v[2] is True
    # shadow mode never stops a rent
    ops.dedup_registry = lambda: _NDReg(mode="shadow")
    assert sched.rent_veto("BRAV", REF["BRAV"], started=lambda hk: False) is None
    # the reigning king is never vetoed
    assert sched.rent_veto(sched.state.current.king_hotkey, "x", started=lambda hk: True) is None


# ── rent path: the veto only runs when NO live pod exists ────────────────────


@dataclass
class _Prov:
    name: str = "lium"
    live: bool = False
    launched: list = field(default_factory=list)
    tagged: list = field(default_factory=list)

    def launch(self, spec: LaunchSpec):
        self.launched.append(spec.name_prefix)
        return [f"{spec.name_prefix}-0"]

    def wait_ready(self, pod_id, *, timeout):
        return True

    def get_ip(self, pod_id):
        return PodAddress(ip="10.0.0.9", ssh_port=2222)

    def terminate(self, pod_id):
        pass

    def list_tagged(self, prefix):
        return [f"{prefix}-0"] if self.live else []

    def live_pod_address(self, pod_id):
        return PodAddress(ip="10.0.0.9", ssh_port=2222) if self.live else None

    def pod_identity(self, pod_id):
        return {"id": f"uid-{pod_id}", "status": "RUNNING", "ip": "10.0.0.9", "port": 2222}


def _rent(prov, veto):
    return rent_funded_pod(round_id="7", hotkey="5HK", api_key="sk", sku="L40S",
                           image="img@sha256:" + "0" * 64, ssh_pubkey="ssh-ed25519 A",
                           netuid=91, provider_factory=lambda k: prov,
                           before_launch=veto)


def test_before_launch_vetoes_a_fresh_rent_without_launching():
    calls = []
    prov = _Prov()
    res = _rent(prov, lambda: calls.append(1) or ("near-dup of X", "near_duplicate"))
    assert not res.ok and res.error_class == "near_duplicate" and res.burn_attempt is False
    assert prov.launched == [] and calls == [1]


def test_before_launch_never_touches_an_adopted_live_pod():
    calls = []
    prov = _Prov(live=True)
    res = _rent(prov, lambda: calls.append(1) or ("x", "near_duplicate"))
    assert res.ok and res.adopted and calls == []        # training leg left alone


def test_before_launch_fault_fails_open():
    def boom():
        raise RuntimeError("guard down")

    prov = _Prov()
    assert _rent(prov, boom).ok and prov.launched


# ── loader round-trip ────────────────────────────────────────────────────────


def test_loader_parses_the_guard_knobs(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    text = (repo / "chain.toml").read_text(encoding="utf-8")
    block = textwrap.dedent("""\
        max_inflight_per_coldkey = 3
        near_dup_mode = "enforce"
        near_dup_threshold = 0.985
        near_dup_lookback_eras = 2
        near_dup_min_private_shingles = 150
        """)
    p = tmp_path / "chain.toml"
    p.write_text(text.replace("[round]\n", "[round]\n" + block, 1), encoding="utf-8")
    r = load_chain_config(p).round
    assert (r.max_inflight_per_coldkey, r.near_dup_mode, r.near_dup_threshold,
            r.near_dup_lookback_eras, r.near_dup_min_private_shingles) == (3, "enforce", 0.985,
                                                                           2, 150)
    d = load_chain_config(repo / "chain.toml").round
    assert (d.max_inflight_per_coldkey, d.near_dup_mode) == (0, "off")    # inert defaults
    for bad in ('near_dup_mode = "enforced"', "near_dup_threshold = 0", "near_dup_threshold = 1.5"):
        p.write_text(text.replace("[round]\n", "[round]\n" + bad + "\n", 1), encoding="utf-8")
        with pytest.raises(ValueError):
            load_chain_config(p)


def test_registry_module_exports_the_tier_name():
    assert DR.NEAR_DUP_TIER == "near_duplicate"
