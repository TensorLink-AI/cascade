"""DEC-CA-0050 go-live on the validator + audit: the era contract gate, the
v2 lineage reset in the era envelope, the receipt `v2_block` stamp, and the
`v2` audit check — each settlement judged under its own block's rules, and
everything before the v2 block byte-identical."""
from __future__ import annotations

import hashlib
from dataclasses import replace

from cascade.audit import checks as C
from cascade.shared.config import V2Config, load_chain_config
from cascade.shared.era import settlement_era, v2_first_era
from cascade.shared.hippius import manifest_round_key
from cascade.shared.manifest import contract_digest, dump_manifest
from tests.unit.test_era_validator import (
    M1A,
    M1B,
    _armed,
    _cascade,
    _manifest,
    _noisy,
    _rollover,
    _runner,
    _scores,
    _Store,
)


def _v2(cfg, *, eras_after_rollover=2):
    """``_armed`` plus a v2 block ``eras_after_rollover`` eras past the rollover."""
    armed = _armed(cfg)
    era_len = armed.round.epoch_blocks * armed.round.era_settlements
    n = _rollover(armed) + eras_after_rollover * era_len
    v2c = load_chain_config().training.v2_contract
    lb = replace(armed.lane_b, from_block=n + 1, domains=("energy", "web"))
    return replace(armed, v2=V2Config(from_block=n), lane_b=lb,
                   training=replace(armed.training, v2_from_block=n, v2_contract=v2c)), n


def _moe_manifest(cfg, block, **kw):
    m = _manifest(cfg, block, **kw)
    v2c = cfg.training.v2_contract
    entries = tuple(replace(e, size=v2c.arch_preset, init_pointer=m.warm_start_ckpt,
                            training_contract_digest=contract_digest(v2c)) for e in m.entries)
    return replace(m, contract_digest=contract_digest(v2c),
                   base_arch_digest=v2c.base_arch_digest, entries=entries,
                   warm_start_size=v2c.arch_preset if m.warm_start_ckpt else "")


def test_contract_gate_follows_the_settlement_era(cfg):
    v2cfg, n = _v2(cfg)
    eb = v2cfg.round.epoch_blocks
    r = _runner(v2cfg)
    # the settlement AT the v2 block belongs to the last toto2 era
    assert r.check_manifest(_manifest(v2cfg, n)) is None
    assert r.check_manifest(_moe_manifest(v2cfg, n)).startswith("contract_digest_mismatch")
    # the first v2 settlement: the MoE contract, never toto2
    assert r.check_manifest(_moe_manifest(v2cfg, n + eb)) is None
    assert r.check_manifest(_manifest(v2cfg, n + eb)).startswith("contract_digest_mismatch")


def test_v2_lineage_starts_from_random_init(cfg, tmp_path):
    v2cfg, n = _v2(cfg)
    eb = v2cfg.round.epoch_blocks
    era_len = eb * v2cfg.round.era_settlements
    # a live toto2 generation (installed before the upgrade: effective era unknown)
    cas = _cascade(v2cfg, tmp_path, generation=1, members=(M1A, M1B))
    store = _Store()
    v = _runner(v2cfg, cascade=cas, store=store)
    pre = n - era_len + eb                     # a pre-v2 era keeps the toto2 init
    era_pre = settlement_era(v2cfg.round, pre)
    predecessor = _manifest(v2cfg, pre, warm_start=(M1A, M1B)[era_pre.index % 2],
                             generation=1, member_index=era_pre.index % 2)
    assert v.check_manifest(predecessor) is None
    store.texts[manifest_round_key(predecessor.round_id)] = dump_manifest(predecessor)
    # the first v2 eras: random init, generation 0 — a toto2 member is refused
    for block in (n + eb, n + era_len + eb, n + 2 * era_len + eb):
        assert v.check_manifest(_moe_manifest(v2cfg, block)) is None, block
        era = settlement_era(v2cfg.round, block)
        bad = _moe_manifest(v2cfg, block, warm_start=(M1A, M1B)[era.index % 2],
                            generation=1, member_index=era.index % 2)
        assert v.check_manifest(bad).startswith("era_init_mismatch"), block
    # a toto2 generation that took effect inside v2 (fired before it) is masked too
    cas.state = replace(cas.state, generation=2, members=(M1A,),
                        generation_effective_era=v2_first_era(v2cfg) + 1)
    assert v.check_manifest(_moe_manifest(v2cfg, n + 3 * era_len + eb)) is None
    # a v2-lineage generation (effective >= first + 2) is the MoE init
    cas.state = replace(cas.state, generation=3, members=(M1B,),
                        generation_effective_era=v2_first_era(v2cfg) + 2)
    block = n + 3 * era_len + eb
    promoted = _moe_manifest(v2cfg, block, warm_start=M1B, generation=3, member_index=0,
                             prev_round_id=predecessor.round_id)
    assert v.check_manifest(promoted).startswith("manifest_chain_broken")
    promoted = replace(promoted, prev_manifest_sha256=hashlib.sha256(
        dump_manifest(predecessor).encode()).hexdigest())
    assert v.check_manifest(promoted) is None
    assert v.check_manifest(_moe_manifest(v2cfg, block)).startswith("era_init_mismatch")


def test_receipt_stamps_v2_block_and_audit_replays_it(cfg):
    v2cfg, n = _v2(cfg)
    eb = v2cfg.round.epoch_blocks
    strong = {"chal_hk": _noisy(_scores(1.0, 0), 0.6, 22)}

    def _receipt(c, m, block):
        r = _runner(c, chal=strong)
        out = r.process_settlement(m, windows=[], base_seed=block)
        return r.build_round_receipt(m, base_seed=block, epoch_start_block=block,
                                     epoch_block_hash="0x" + "ab" * 32,
                                     outcome=out, windows=[], era_base_seed=7)

    # pre-v2 (incl. the settlement AT the block): byte-identical to a v2-off validator
    plain, _ = _v2(cfg)
    plain = replace(plain, v2=V2Config(), lane_b=replace(plain.lane_b, from_block=0),
                    training=replace(plain.training, v2_from_block=0, v2_contract=None))
    for block in (n - eb, n):
        a = _receipt(plain, _manifest(plain, block), block)
        b = _receipt(v2cfg, _manifest(v2cfg, block), block)
        assert a.canonical_body() == b.canonical_body()
        assert b"v2_block" not in b.canonical_body() and b.v2_block == 0
        assert C.check_v2(b, v2cfg).status == C.PASS
        assert C.check_contract_digest(b, v2cfg).status == C.PASS
    # the first v2 settlement: stamped, judged under the MoE contract
    block = n + eb
    rc = _receipt(v2cfg, _moe_manifest(v2cfg, block), block)
    assert rc.status == "scored" and rc.v2_block == n
    assert C.check_v2(rc, v2cfg).status == C.PASS, C.check_v2(rc, v2cfg).detail
    assert C.check_contract_digest(rc, v2cfg).status == C.PASS
    assert C.check_base_arch_digest(rc, v2cfg).status == C.PASS
    from cascade.shared.receipt import dump_receipt, load_receipt

    assert load_receipt(dump_receipt(rc)).v2_block == n
    # a validator that dropped (or invented) the stamp is caught
    assert C.check_v2(replace(rc, v2_block=0), v2cfg).status == C.FAIL
    early = _receipt(v2cfg, _manifest(v2cfg, n), n)
    assert C.check_v2(replace(early, v2_block=n), v2cfg).status == C.FAIL
    # a v2 era warm-started before any v2-lineage generation could exist
    ws = replace(rc, manifest={**rc.manifest, "warm_start_ckpt": M1A,
                               "warm_start_size": v2cfg.training.v2_contract.arch_preset})
    assert C.check_v2(ws, v2cfg).status == C.FAIL
    toto = replace(rc, manifest={**rc.manifest, "warm_start_ckpt": M1A,
                                 "warm_start_size": "toto2-4m"})
    assert C.check_v2(toto, v2cfg).status == C.FAIL
