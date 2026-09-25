"""Stake-weighted activation of the DEC-CA-0043 rollover (DEC-CA-0045).

Validators post a readiness note on chain; every node tallies eligible stake
at each boundary; the first boundary at/over the threshold locks in and the
NEXT boundary is the rollover; the typed-in config always wins.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.config import load_chain_config
from cascade.shared.era import era_king_active, rolling_active, tenure_blocks_active

REPO = Path(__file__).resolve().parents[2]
FEATURE = "rolling-era-king"
GRID = 3600
# A mainnet boundary past cohort_maxt_from_block (9046800): the era king must
# not precede the corrections it stacks on.
B0 = 9046800 + GRID


@pytest.fixture
def cfg():
    return load_chain_config(REPO / "chain.toml")


def _v(hk, stake, *, permit=True, last_update=0):
    return A.ValidatorStake(hotkey=hk, stake=stake, permit=permit, last_update=last_update)


class FakeChain:
    """Enough of ChainClient for the resolver: live + as-of reads, a note
    writer, a block counter."""

    def __init__(self, validators, signals, *, block, hotkey="v1"):
        self.validators = list(validators)
        self.signals = dict(signals)
        self.block = block
        self.hotkey = hotkey
        self.history: dict[int, tuple[list, dict]] = {}
        self.fail_as_of = False
        self.fail_all = False
        self.written: list[str] = []
        self.as_of_reads: list[int] = []

    def current_block(self):
        return self.block

    def _view(self, block):
        if self.fail_all:
            raise RuntimeError("chain down")
        if block is not None:
            if self.fail_as_of:
                raise RuntimeError("state pruned")
            if block in self.history:
                return self.history[block]
        return self.validators, self.signals

    def validator_stakes(self, block=None):
        if block is not None:
            self.as_of_reads.append(int(block))
        return list(self._view(block)[0])

    def read_plain_commitments(self, block=None):
        return dict(self._view(block)[1])

    def set_plain_commitment(self, payload):
        self.written.append(payload)
        self.signals[self.hotkey] = payload

    def hotkey_ss58(self):
        return self.hotkey


# ── the note ─────────────────────────────────────────────────────────────────


def test_signal_round_trips_and_rejects_malformed():
    s = A.format_signal(FEATURE)
    assert s == "cascade-ready:1:rolling-era-king:0:0"
    assert A.parse_signal(s) == A.ReadySignal(FEATURE, 0, 0)
    locked = A.format_signal(FEATURE, lock_block=100, activation_block=200)
    assert A.parse_signal(locked) == A.ReadySignal(FEATURE, 100, 200)
    assert A.is_signal_payload(locked) and not A.is_signal_payload("metro-v1:gen:x")
    for bad in ("metro-v1:gen:hippius:a/b@sha256:0", "cascade-ready:2:f:0:0",
                "cascade-ready:1:f:0", "cascade-ready:1::0:0", "cascade-ready:1:f:0:5",
                "cascade-ready:1:f:9:5", "cascade-ready:1:f:x:0", None, 3):
        assert A.parse_signal(bad) is None
    with pytest.raises(ValueError):
        A.format_signal("has:colon")


# ── the tally ────────────────────────────────────────────────────────────────


def test_tally_counts_eligible_stake_behind_the_feature():
    vals = [_v("a", 40), _v("b", 35), _v("c", 25), _v("m", 500, permit=False),
            _v("z", 10, permit=True)]
    vals[-1] = _v("z", 0)                       # permit but no stake: ignored
    sig = {"a": A.format_signal(FEATURE), "b": A.format_signal("other"),
           "m": A.format_signal(FEATURE), "c": "metro-v1:gen:hippius:a/b@sha256:0"}
    t = A.tally(FEATURE, vals, sig, threshold=0.51, block=B0)
    assert t.eligible == ("a", "b", "c") and t.signed == ("a",)
    assert t.total_stake == 100 and t.signed_stake == 40 and not t.locked
    sig["c"] = A.format_signal(FEATURE)
    t = A.tally(FEATURE, vals, sig, threshold=0.51, block=B0)
    assert t.signed == ("a", "c") and t.ratio == pytest.approx(0.65) and t.locked
    js = t.to_json()
    assert js["locked"] and js["n_signed"] == 2 and js["n_eligible"] == 3


def test_tally_threshold_is_inclusive_and_dormant_stake_can_be_excluded():
    vals = [_v("a", 51, last_update=B0 - 10), _v("b", 49, last_update=B0 - 50_000)]
    sig = {"a": A.format_signal(FEATURE)}
    assert A.tally(FEATURE, vals, sig, threshold=0.51, block=B0).locked
    assert not A.tally(FEATURE, vals, sig, threshold=0.52, block=B0).locked
    # b has not set weights in 50k blocks: with the dormant cut it drops out
    # of the denominator and a alone is 100%.
    t = A.tally(FEATURE, vals, sig, threshold=0.9, block=B0, dormant_after_blocks=10_000)
    assert t.eligible == ("a",) and t.locked
    assert not A.tally(FEATURE, [], sig, threshold=0.51, block=B0).locked


def test_agreed_activation_needs_threshold_stake_on_one_block():
    vals = [_v("a", 40), _v("b", 30), _v("c", 30)]
    note = lambda act: A.format_signal(FEATURE, lock_block=act - GRID, activation_block=act)  # noqa: E731
    assert A.agreed_activation(FEATURE, vals, {"a": note(B0)}, threshold=0.51, block=B0) is None
    assert A.agreed_activation(FEATURE, vals, {"a": note(B0), "b": note(B0)},
                               threshold=0.51, block=B0) == (B0 - GRID, B0)
    # split notes: the biggest group is under the line
    assert A.agreed_activation(FEATURE, vals, {"a": note(B0), "b": note(B0 + GRID),
                                               "c": note(B0 + GRID)},
                               threshold=0.75, block=B0) is None
    # plain readiness notes carry no block and never count as agreement
    assert A.agreed_activation(FEATURE, vals, {h: A.format_signal(FEATURE) for h in "abc"},
                               threshold=0.51, block=B0) is None


# ── block arithmetic + applying the block ────────────────────────────────────


def test_next_boundary_is_on_the_grid_in_force(cfg):
    assert A.latest_boundary(cfg.round, B0 + 5) == B0
    assert A.activation_block_for(cfg.round, B0) == B0 + GRID
    armed = A.apply_activation(cfg, B0 + GRID)
    # after the switch the grid is 900: the boundary after the rollover is 900 on
    assert A.activation_block_for(armed.round, B0 + GRID) == B0 + GRID + 900
    assert A.latest_boundary(armed.round, B0 + GRID + 950) == B0 + GRID + 900


def test_apply_activation_writes_every_rollover_key_and_the_grid_switch(cfg):
    assert cfg.round.rolling_from_block == 0 and cfg.activation.epoch_blocks_after == 900
    armed = A.apply_activation(cfg, B0)
    assert armed.round.rolling_from_block == B0
    assert armed.round.epoch_activation_block == B0
    assert armed.round.epoch_blocks_prev == 3600 and armed.round.epoch_blocks == 900
    assert armed.scoring.era_king_from_block == B0
    assert armed.scoring.tenure_blocks_from_block == B0
    assert armed.scoring.cohort_maxt_increment_from_block == B0
    assert armed.scoring.cohort_maxt_from_block == 9046800       # already set: kept
    assert armed.activation.resolved_block == B0
    assert A.resolved_rollover(armed) == B0 and A.configured_rollover(armed) == 0
    assert rolling_active(armed.round, B0) and era_king_active(armed.scoring, B0)
    assert tenure_blocks_active(armed.scoring, B0)
    assert not rolling_active(armed.round, B0 - 1)
    # idempotent on the same block; a lock-in is one-way
    assert A.apply_activation(armed, B0) is armed
    with pytest.raises(ValueError, match="one-way"):
        A.apply_activation(armed, B0 + GRID)
    # untouched elsewhere
    assert armed.training == cfg.training and armed.storage == cfg.storage


def test_apply_activation_refuses_a_block_the_loader_would_refuse(cfg):
    with pytest.raises(ValueError, match="boundary"):
        A.apply_activation(cfg, B0 + 900)          # not on the 3600 grid before it
    with pytest.raises(ValueError, match="cohort_maxt"):
        A.apply_activation(cfg, 9046800 - GRID)     # precedes the max-T correction
    assert A.apply_activation(cfg, 0) is cfg


def test_apply_activation_keeps_the_grid_when_no_after_grid_is_set(cfg):
    keep = replace(cfg, activation=replace(cfg.activation, epoch_blocks_after=0))
    # On the unchanged 3600 grid an era is 4 × 3600 = 14400 blocks, and the
    # rollover must start an era — the loader's rule, applied here too.
    with pytest.raises(ValueError, match="start an era"):
        A.apply_activation(keep, B0)
    era_start = ((9046800 // 14400) + 1) * 14400
    armed = A.apply_activation(keep, era_start)
    assert armed.round.epoch_blocks == armed.round.epoch_blocks_prev == 3600
    assert armed.round.epoch_activation_block == era_start


def test_typed_in_rollover_is_the_owner_override(cfg):
    typed = replace(cfg, round=replace(cfg.round, rolling_from_block=B0, epoch_blocks=900,
                                       epoch_blocks_prev=3600, epoch_activation_block=B0),
                    scoring=replace(cfg.scoring, era_king_from_block=B0,
                                    tenure_blocks_from_block=B0,
                                    cohort_maxt_increment_from_block=B0))
    assert A.configured_rollover(typed) == B0
    assert A.apply_activation(typed, B0 + GRID) is typed


# ── the resolver ─────────────────────────────────────────────────────────────


def _fleet(*stakes):
    return [_v(f"v{i}", s) for i, s in enumerate(stakes, 1)]


def test_resolver_tallies_each_boundary_once_and_locks_in_on_the_first_majority(cfg):
    chain = FakeChain(_fleet(30, 30, 40), {"v1": A.format_signal(FEATURE)}, block=B0 + 5)
    rec = A.ActivationRecord()
    res = A.resolve_activation(cfg, chain, now_block=B0 + 5, record=rec)
    assert res.changed and res.tally is not None and not res.tally.locked
    assert res.record.last_checked_boundary == B0 and not res.record.locked
    assert chain.as_of_reads == [B0]                       # read AS OF the boundary
    # same boundary again: nothing re-read, nothing changes
    again = A.resolve_activation(cfg, chain, now_block=B0 + 100, record=res.record)
    assert not again.changed and chain.as_of_reads == [B0]
    # v3 upgrades before the next boundary: 70% signed at B0 + 3600
    chain.signals["v3"] = A.format_signal(FEATURE)
    chain.block = B0 + GRID + 1
    locked = A.resolve_activation(cfg, chain, now_block=chain.block, record=again.record)
    assert locked.changed and locked.record.locked
    assert locked.record.lock_block == B0 + GRID
    assert locked.record.activation_block == B0 + 2 * GRID       # the NEXT boundary
    assert locked.record.source == "tally"
    # one-way: the stake drops back under the line, nothing moves
    chain.signals.clear()
    chain.block = B0 + 5 * GRID
    later = A.resolve_activation(cfg, chain, now_block=chain.block, record=locked.record)
    assert not later.changed and later.record == locked.record


def test_resolver_adopts_the_block_validators_agree_on(cfg):
    note = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    chain = FakeChain(_fleet(30, 30, 40), {"v1": note, "v3": note}, block=B0 + 7 * GRID)
    res = A.resolve_activation(cfg, chain, now_block=chain.block, record=A.ActivationRecord())
    assert res.changed and res.record.activation_block == B0 + GRID
    assert res.record.lock_block == B0 and res.record.source == "signals"
    assert chain.as_of_reads == []                          # no historical read needed


def test_resolver_never_counts_from_a_later_view_when_the_boundary_is_pruned(cfg):
    """Two nodes counting different chain states is the fork this exists to
    prevent: a failed as-of read is a retry, never a live-view count."""
    chain = FakeChain(_fleet(60, 40), {"v1": A.format_signal(FEATURE)}, block=B0 + 3)
    chain.fail_as_of = True
    res = A.resolve_activation(cfg, chain, now_block=B0 + 3, record=A.ActivationRecord())
    assert not res.changed and not res.record.locked
    assert res.record.last_checked_boundary == 0            # the boundary is retried
    # ... and the fleet's decision reaches it through the notes instead
    chain.signals["v1"] = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    res = A.resolve_activation(cfg, chain, now_block=B0 + 30, record=res.record)
    assert res.record.activation_block == B0 + GRID and res.record.source == "signals"


def test_resolver_reads_the_chain_only_at_a_new_boundary(cfg):
    chain = FakeChain(_fleet(30, 70), {"v1": A.format_signal(FEATURE)}, block=B0 + 1)
    calls = {"n": 0}
    orig = chain.read_plain_commitments

    def counting(block=None):
        calls["n"] += 1
        return orig(block)

    chain.read_plain_commitments = counting
    rec = A.resolve_activation(cfg, chain, now_block=B0 + 1, record=A.ActivationRecord()).record
    assert calls["n"] == 2                                   # notes (live) + tally (as-of)
    for b in range(B0 + 2, B0 + 200, 7):
        A.resolve_activation(cfg, chain, now_block=b, record=rec)
    assert calls["n"] == 2                                   # nothing until the next boundary


def test_record_for_blanks_a_renamed_features_record(cfg):
    stale = A.ActivationRecord(feature="old-feature", lock_block=B0, activation_block=B0 + GRID,
                               source="tally")
    assert A.record_for(cfg, stale) == A.ActivationRecord()
    mine = replace(stale, feature=FEATURE)
    assert A.record_for(cfg, mine) is mine
    # the resolver applies the same rule, so a stale record never arms anything
    chain = FakeChain(_fleet(30, 70), {}, block=B0 + 1)
    res = A.resolve_activation(cfg, chain, now_block=B0 + 1, record=stale)
    assert not res.record.locked


def test_apply_activation_refuses_zero_era_settlements(cfg):
    zero = replace(cfg, round=replace(cfg.round, era_settlements=0))
    with pytest.raises(ValueError, match="era_settlements"):
        A.apply_activation(zero, B0)


def test_watcher_ticks_once_per_boundary_and_hands_back_the_armed_grid(cfg):
    note = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 5)
    w = A.ActivationWatcher(cfg, store_path=None)
    armed = w.tick(chain, B0 + 5)
    assert armed is not None and armed.round.epoch_activation_block == B0 + GRID
    assert (armed.round.epoch_blocks, armed.round.epoch_blocks_prev) == (900, 3600)
    assert w.tick(chain, B0 + 6) is None                     # already armed
    assert A.startup_activation(cfg, chain, store_path=None).round.rolling_from_block == B0 + GRID


def test_resolver_is_a_no_op_on_a_dead_chain_or_a_typed_in_rollover(cfg):
    chain = FakeChain(_fleet(60, 40), {"v1": A.format_signal(FEATURE)}, block=B0 + 3)
    chain.fail_all = True
    res = A.resolve_activation(cfg, chain, now_block=B0 + 3, record=A.ActivationRecord())
    assert not res.changed and not res.record.locked
    off = replace(cfg, activation=replace(cfg.activation, feature=""))
    chain.fail_all = False
    assert not A.resolve_activation(off, chain, now_block=B0 + 3, record=A.ActivationRecord()).changed
    typed = replace(cfg, round=replace(cfg.round, rolling_from_block=B0, epoch_blocks=900,
                                       epoch_blocks_prev=3600, epoch_activation_block=B0),
                    scoring=replace(cfg.scoring, era_king_from_block=B0,
                                    tenure_blocks_from_block=B0,
                                    cohort_maxt_increment_from_block=B0))
    res = A.resolve_activation(typed, chain, now_block=B0 + 3, record=A.ActivationRecord())
    assert res.record.source == "config" and res.record.activation_block == B0
    assert chain.as_of_reads == []


def test_store_round_trips_and_tolerates_garbage(tmp_path):
    store = A.ActivationStore(tmp_path / "activation_state.json")
    assert store.load() == A.ActivationRecord()
    rec = A.ActivationRecord(feature=FEATURE, lock_block=B0, activation_block=B0 + GRID,
                             last_checked_boundary=B0, source="tally")
    store.save(rec)
    assert store.load() == rec
    assert json.loads((tmp_path / "activation_state.json").read_text())["activation_block"] == B0 + GRID
    (tmp_path / "activation_state.json").write_text("{not json")
    assert store.load() == A.ActivationRecord()
    assert A.ActivationStore(None).load() == A.ActivationRecord()


def test_ensure_signal_writes_once_then_rewrites_with_the_agreed_block(cfg):
    chain = FakeChain(_fleet(60, 40), {}, block=B0, hotkey="v1")
    rec = A.ActivationRecord()
    assert A.ensure_signal(chain, cfg, rec, hotkey="v1")
    assert chain.written == [A.format_signal(FEATURE)]
    assert not A.ensure_signal(chain, cfg, rec, hotkey="v1")        # already on chain
    locked = replace(rec, lock_block=B0, activation_block=B0 + GRID, source="tally")
    assert A.ensure_signal(chain, cfg, locked, hotkey="v1")
    assert chain.signals["v1"] == A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    off = replace(cfg, activation=replace(cfg.activation, feature=""))
    assert not A.ensure_signal(chain, off, rec, hotkey="v1")


def test_startup_activation_arms_from_the_notes_and_persists(cfg, tmp_path):
    note = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 2 * GRID)
    armed = A.startup_activation(cfg, chain, store_path=tmp_path / "a.json")
    assert rolling_active(armed.round, B0 + GRID)
    assert A.ActivationStore(tmp_path / "a.json").load().activation_block == B0 + GRID
    # a dead chain leaves the typed-in config untouched
    chain.fail_all = True
    assert A.startup_activation(cfg, chain, store_path=tmp_path / "b.json") is cfg


# ── config loading ───────────────────────────────────────────────────────────


def test_shipped_configs_carry_the_activation_section():
    main = load_chain_config(REPO / "chain.toml")
    assert main.activation.enabled and main.activation.feature == FEATURE
    assert main.activation.threshold == 0.51 and main.activation.epoch_blocks_after == 900
    assert main.round.rolling_from_block == 0            # the fleet decides the block
    test = load_chain_config(REPO / "chain.testnet.toml")
    assert test.activation.enabled and A.configured_rollover(test) == 600  # typed in wins


def test_loader_rejects_a_bad_threshold_or_grid(tmp_path):
    src = (REPO / "chain.toml").read_text(encoding="utf-8")
    p = tmp_path / "chain.toml"
    p.write_text(src.replace("threshold = 0.51", "threshold = 1.5"), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_chain_config(p)
    p.write_text(src.replace("epoch_blocks_after = 900", "epoch_blocks_after = 1000"),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="epoch_blocks_after"):
        load_chain_config(p)
    # keeping the 3600 grid with 4-settlement eras: 3 of 4 boundaries could
    # never start an era, so a lock-in there would resolve an unusable block
    p.write_text(src.replace("epoch_blocks_after = 900", "epoch_blocks_after = 0"),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="cannot start an era"):
        load_chain_config(p)
    p.write_text(src.replace('feature = "rolling-era-king"', 'feature = "rolling era:king"'),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="feature"):
        load_chain_config(p)


def test_miner_cli_times_on_the_grid_the_validators_decided(cfg):
    """The miner's chain.toml keeps the rollover keys at 0; the CLI reads the
    validators' notes so countdowns and reveal timing follow the switch."""
    from cascade.miner.cli import _live_cfg

    note = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 2 * GRID)
    live = _live_cfg(cfg, chain)
    assert live.round.epoch_activation_block == B0 + GRID
    assert live.round.epoch_blocks == 900 and live.round.epoch_blocks_prev == 3600
    dead = FakeChain([], {}, block=B0)
    dead.fail_all = True
    assert _live_cfg(cfg, dead) is cfg                       # never fails a command


def test_provisioner_loop_switches_its_grid_from_the_activation_hook():
    from cascade.provision.loop import ProvisionerLoop

    loop = ProvisionerLoop.__new__(ProvisionerLoop)
    loop.epoch_blocks, loop.epoch_blocks_prev, loop.epoch_activation_block = 3600, 0, 0
    loop.chain_client = object()
    seen = []
    loop.activation_fn = lambda client, block: seen.append(block) or (900, 3600, B0 + GRID)
    loop._maybe_apply_activation(B0 + 5)
    assert seen == [B0 + 5]
    assert (loop.epoch_blocks, loop.epoch_blocks_prev, loop.epoch_activation_block) == (
        900, 3600, B0 + GRID)
    assert loop.epoch_at(B0 + GRID - 1) == 3600 and loop.epoch_at(B0 + GRID) == 900
    loop.activation_fn = lambda client, block: (_ for _ in ()).throw(RuntimeError("down"))
    loop._maybe_apply_activation(B0 + 6)                     # swallowed, grid kept
    assert loop.epoch_blocks == 900


# ── chain client (fake subtensor) ────────────────────────────────────────────


class _Meta:
    def __init__(self):
        self.n = 3
        self.hotkeys = ["h0", "h1", "h2"]
        self.S = [10.0, 0.0, 5.5]
        self.validator_permit = [True, False, True]
        self.last_update = [100, 0, 90]
        self.coldkeys = ["c0", "c1", "c2"]


class _Sub:
    def __init__(self):
        self.meta_calls = []
        self.commit_calls = []
        self.map_calls = []
        self.commit_fail = False
        self.plain = {"h0": "cascade-ready:1:rolling-era-king:0:0",
                      "h2": "metro-v1:gen:hippius:a/b@sha256:" + "0" * 64}

    def metagraph(self, **kw):
        self.meta_calls.append(kw)
        return _Meta()

    def get_all_commitments(self, netuid, block=None):  # pragma: no cover — not used
        raise AssertionError("the client must read CommitmentOf itself (no SDK decoder spam)")

    def query_map(self, module, name, params, block=None):
        assert (module, name, params) == ("Commitments", "CommitmentOf", [91])
        self.map_calls.append(block)
        rows = [(hk, {"block": 5, "info": {"fields": [{"Raw64": "0x" + v.encode().hex()}]}})
                for hk, v in self.plain.items()]
        # a sealed miner commit: no Raw field — skipped, never an error
        rows.append(("h1", {"block": 7, "info": {"fields": [
            {"TimelockEncrypted": {"encrypted": "0xdead", "reveal_round": 9}}]}}))
        return rows

    def set_commitment(self, wallet, netuid, data, **kw):
        self.commit_calls.append((netuid, data, kw))
        return SimpleNamespace(success=not self.commit_fail,
                               message="rate limited" if self.commit_fail else "ok")

    # No bulk/per-uid REVEALED store on this build: poll_commitments falls
    # through to the plain store — the only path where a note could surface.

    def get_commitment(self, netuid, uid, block=None):
        return self.plain.get(f"h{uid}")


def _client(sub):
    from cascade.shared.chain import ChainClient
    c = ChainClient.__new__(ChainClient)
    c.netuid = 91
    c._subtensor = sub
    c._wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="h0"))
    return c


def test_chain_client_reads_stake_permits_and_notes_as_of_a_block():
    sub = _Sub()
    c = _client(sub)
    rows = c.validator_stakes(block=123)
    assert sub.meta_calls[-1] == {"netuid": 91, "lite": True, "block": 123}
    assert rows == [A.ValidatorStake("h0", 10.0, True, 100), A.ValidatorStake("h1", 0.0, False, 0),
                    A.ValidatorStake("h2", 5.5, True, 90)]
    assert c.read_plain_commitments() == sub.plain             # Raw only; sealed skipped
    assert c.read_plain_commitments(block=123) == sub.plain
    assert sub.map_calls == [None, 123]                         # as-of read forwarded
    c.set_plain_commitment("cascade-ready:1:rolling-era-king:0:0")
    assert sub.commit_calls == [(91, "cascade-ready:1:rolling-era-king:0:0",
                                 {"wait_for_inclusion": True, "wait_for_finalization": False})]
    assert c.hotkey_ss58() == "h0"
    # a rejected extrinsic is a failure, never "signalled"
    from cascade.shared.chain import ChainError
    sub.commit_fail = True
    with pytest.raises(ChainError, match="rate limited"):
        c.set_plain_commitment("cascade-ready:1:rolling-era-king:0:0")


def test_poll_commitments_never_surfaces_an_activation_note():
    """The plain-store fallback is where a validator's note lives; a field
    builder must never see it as a submission."""
    sub = _Sub()
    c = _client(sub)
    got = c.poll_commitments()
    assert [x.hotkey for x in got] == ["h2"]
    assert all(not A.is_signal_payload(x.payload) for x in got)


# ── receipts ─────────────────────────────────────────────────────────────────


def _receipt(**kw):
    from cascade.shared.receipt import RoundReceipt
    base = dict(round_id="1", status="rejected", epoch_start_block=B0, epoch_block_hash="0x",
                base_seed=1, generation_seed=2, training_seed=3, manifest={},
                reject_reason="x")
    base.update(kw)
    return RoundReceipt(**base)


def test_receipt_activation_block_is_drop_when_default():
    from cascade.shared.receipt import dump_receipt, load_receipt, summarize_receipt

    plain = _receipt()
    assert "activation_block" not in json.loads(plain.canonical_body())
    assert load_receipt(dump_receipt(plain)).activation_block == 0
    stamped = _receipt(activation_block=B0)
    assert json.loads(stamped.canonical_body())["activation_block"] == B0
    assert load_receipt(dump_receipt(stamped)).activation_block == B0
    assert summarize_receipt(stamped)["activation_block"] == B0
    assert summarize_receipt(plain)["activation_block"] == 0


# ── the validator runner ─────────────────────────────────────────────────────


def test_validator_runner_locks_in_applies_and_stamps(cfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    cascade = SimpleNamespace(round_cfg=cfg.round, threshold_fn=None)
    runner = ValidatorRunner(cfg=cfg, cascade=cascade,
                             activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 1, hotkey="v1")
    runner._activation_startup(chain)
    # signalled BEFORE the tally, so its own 60% is over the line at boundary
    # B0; the note is then rewritten with the agreed block in the same step
    assert chain.written[0] == A.format_signal(FEATURE) and len(chain.written) == 2
    assert runner._activation.locked and runner._activation.activation_block == B0 + GRID
    assert runner.activation_block == B0 + GRID
    assert rolling_active(runner.cfg.round, B0 + GRID) and era_king_active(runner.cfg.scoring, B0 + GRID)
    assert cascade.round_cfg is runner.cfg.round and callable(cascade.threshold_fn)
    # the note now carries the block for late joiners
    assert chain.signals["v1"] == A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    assert A.ActivationStore(tmp_path / "act.json").load().activation_block == B0 + GRID
    # a restart restores it without a chain read
    fresh = ValidatorRunner(cfg=cfg, activation_store=A.ActivationStore(tmp_path / "act.json"))
    dead = FakeChain([], {}, block=B0 + 2 * GRID)
    dead.fail_all = True
    fresh._activation_startup(dead)
    assert fresh.activation_block == B0 + GRID and rolling_active(fresh.cfg.round, B0 + GRID)


def test_validator_runner_records_nothing_for_a_typed_in_rollover(cfg):
    from cascade.validator.loop import ValidatorRunner

    typed = replace(cfg, round=replace(cfg.round, rolling_from_block=B0, epoch_blocks=900,
                                       epoch_blocks_prev=3600, epoch_activation_block=B0),
                    scoring=replace(cfg.scoring, era_king_from_block=B0,
                                    tenure_blocks_from_block=B0,
                                    cohort_maxt_increment_from_block=B0))
    runner = ValidatorRunner(cfg=typed)
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 1, hotkey="v1")
    runner._activation_startup(chain)
    assert runner.activation_block == 0 and runner._activation.source == "config"
    assert chain.written == []          # a typed-in rollover makes the note inert: no extrinsic


def test_validator_runner_off_when_the_feature_is_blank(cfg):
    from cascade.validator.loop import ValidatorRunner

    off = replace(cfg, activation=replace(cfg.activation, feature=""))
    runner = ValidatorRunner(cfg=off)
    chain = FakeChain(_fleet(60, 40), {"v2": A.format_signal(FEATURE)}, block=B0 + 1)
    runner._activation_startup(chain)
    runner._activation_tick(chain)
    assert chain.written == [] and runner.activation_block == 0
    assert not rolling_active(runner.cfg.round, B0 + GRID)


def test_chain_status_carries_the_activation_view_and_the_grid_in_force(cfg):
    from cascade.shared.chain_status import build_chain_status

    armed = A.apply_activation(cfg, B0 + GRID)
    rec = A.ActivationRecord(feature=FEATURE, lock_block=B0, activation_block=B0 + GRID,
                             source="tally")
    t = A.tally(FEATURE, _fleet(60, 40), {"v1": A.format_signal(FEATURE)}, threshold=0.51, block=B0)
    doc = build_chain_status(armed, current_block=B0 + 10, commitments=[],
                             activation=A.summary(armed, rec, t))
    assert doc["epoch_blocks"] == 3600                     # before the switch
    assert doc["activation"]["activation_block"] == B0 + GRID
    assert doc["activation"]["tally"]["ratio"] == pytest.approx(0.6)
    after = build_chain_status(armed, current_block=B0 + GRID + 10, commitments=[])
    assert after["epoch_blocks"] == 900 and "activation" not in after


# ── the trainer runner ───────────────────────────────────────────────────────


def test_trainer_tick_arms_rolling_from_the_fleet_decision(cfg, tmp_path):
    from cascade.trainer.loop import TrainerRunner

    runner = TrainerRunner(cfg=cfg, base_trainer=None, work_root=tmp_path,
                           activation_store=A.ActivationStore(tmp_path / "act.json"))
    runner.promotion = SimpleNamespace(round_cfg=cfg.round, scoring_cfg=cfg.scoring)
    note = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 100)
    runner._activation_tick(chain, B0 + 100)
    assert rolling_active(runner.cfg.round, B0 + GRID)
    assert not rolling_active(runner.cfg.round, B0 + GRID - 1)
    assert runner.promotion.round_cfg is runner.cfg.round
    # The tenure gate rides the same armed config: the engine's ripeness
    # threshold rescales with the grid only if it sees the armed scoring.
    assert runner.promotion.scoring_cfg is runner.cfg.scoring
    assert runner.promotion.scoring_cfg.tenure_blocks_from_block == B0 + GRID
    assert A.ActivationStore(tmp_path / "act.json").load().activation_block == B0 + GRID


# ── the audit ────────────────────────────────────────────────────────────────


def test_audit_replays_under_the_recorded_block_and_checks_agreement(cfg):
    from cascade.audit.checks import check_activation

    r0 = _receipt()
    assert A.apply_receipt_activation(cfg, r0) is cfg
    assert check_activation(r0, cfg).status == "PASS"
    r1 = _receipt(activation_block=B0 + GRID)
    replay = A.apply_receipt_activation(cfg, r1)
    assert era_king_active(replay.scoring, B0 + GRID)
    assert check_activation(r1, replay).status == "WARN"          # no chain
    note = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    agree = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 3 * GRID)
    assert check_activation(r1, replay, agree).status == "PASS"
    other = FakeChain(_fleet(60, 40), {"v1": A.format_signal(
        FEATURE, lock_block=B0 + GRID, activation_block=B0 + 2 * GRID)}, block=B0 + 3 * GRID)
    assert check_activation(r1, replay, other).status == "FAIL"
    quiet = FakeChain(_fleet(60, 40), {}, block=B0 + 3 * GRID)
    assert check_activation(r1, replay, quiet).status == "WARN"
    assert check_activation(_receipt(activation_block=B0 + 900), cfg).status == "FAIL"
    off = replace(cfg, activation=replace(cfg.activation, feature=""))
    assert A.apply_receipt_activation(off, r1) is off        # never applied under "off"
    assert check_activation(r1, off).status == "FAIL"
    # a block the audit config cannot apply is a FAIL, not a traceback
    bad = _receipt(activation_block=9046800 - GRID)          # precedes the max-T correction
    assert A.apply_receipt_activation(cfg, bad) is cfg
    res = check_activation(bad, cfg)
    assert res.status == "FAIL" and "cannot be applied" in res.detail
    # the owner later pins the decided block: the receipt must match it
    typed = replace(cfg, round=replace(cfg.round, rolling_from_block=B0 + GRID, epoch_blocks=900,
                                       epoch_blocks_prev=3600,
                                       epoch_activation_block=B0 + GRID),
                    scoring=replace(cfg.scoring, era_king_from_block=B0 + GRID,
                                    tenure_blocks_from_block=B0 + GRID,
                                    cohort_maxt_increment_from_block=B0 + GRID))
    assert A.apply_receipt_activation(typed, r1) is typed
    assert check_activation(r1, typed).status == "PASS"
    assert check_activation(_receipt(activation_block=B0), typed).status == "FAIL"


# ── audit fixes (2026-09-22): in-order tallying, admissible notes, records ────


class _PerBlockChain(FakeChain):
    """FakeChain whose as-of reads fail for specific blocks only."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fail_blocks: set[int] = set()

    def _view(self, block):
        if block is not None and int(block) in self.fail_blocks:
            raise RuntimeError("state pruned")
        return super()._view(block)


def test_agreed_activation_ignores_inadmissible_pairs(cfg):
    vals = [_v("a", 60), _v("b", 40)]
    good = A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + GRID)
    for bad in (A.format_signal(FEATURE, lock_block=5, activation_block=7),
                A.format_signal(FEATURE, lock_block=B0, activation_block=B0 + 2 * GRID),
                A.format_signal(FEATURE, lock_block=B0 + 1, activation_block=B0 + GRID)):
        sig = A.parse_signal(bad)
        assert not A.valid_pair(cfg.round, sig.lock_block, sig.activation_block)
        # without the grid the pair is taken at face value; with it, never
        assert A.agreed_activation(FEATURE, vals, {"a": bad}, threshold=0.51, block=B0) is not None
        assert A.agreed_activation(FEATURE, vals, {"a": bad}, threshold=0.51, block=B0,
                                   round_cfg=cfg.round) is None
        chain = FakeChain(vals, {"a": bad}, block=B0 + 5 * GRID)
        res = A.resolve_activation(cfg, chain, now_block=chain.block, record=A.ActivationRecord())
        assert res.record.source == "tally"             # counted as ready; never adopted
        assert res.record.lock_block == B0 + 5 * GRID
    assert A.agreed_activation(FEATURE, vals, {"a": good}, threshold=0.51, block=B0,
                               round_cfg=cfg.round) == (B0, B0 + GRID)


def test_resolver_retries_a_missed_boundary_in_order_and_never_counts_a_later_one(cfg):
    """The fork this closes: a node down across the lock-in boundary B must
    not count the next boundary B2 and name B2+grid while its peers named
    B+grid. It retries B; the notes may rescue it meanwhile."""
    B, B2 = B0 + GRID, B0 + 2 * GRID
    ready = A.format_signal(FEATURE)
    at_b = ([_v("a", 49.1), _v("o", 24.5), _v("c", 11.6), _v("d", 9.5), _v("e", 5.3)],
            {"a": ready, "o": ready, "c": ready})
    # healthy peer: locks at B, rollover B2
    peer = _PerBlockChain(*at_b, block=B + 5)
    peer.history[B] = at_b
    p = A.resolve_activation(cfg, peer, now_block=B + 5,
                             record=A.ActivationRecord(feature=FEATURE, last_checked_boundary=B0))
    assert p.record.lock_block == B and p.record.activation_block == B2
    # this node: tallied B0, then down across B; back at B2+5 with B pruned.
    # By now stake drifted so the two rewritten notes are under 51% of total.
    locked = A.format_signal(FEATURE, lock_block=B, activation_block=B2)
    at_b2 = ([_v("a", 46.0), _v("o", 4.0), _v("c", 30.0), _v("d", 15.0), _v("e", 5.0)],
             {"a": locked, "o": locked, "c": ready})
    node = _PerBlockChain(*at_b2, block=B2 + 5)
    node.history[B] = at_b
    node.history[B2] = at_b2
    node.fail_blocks = {B}
    rec = A.ActivationRecord(feature=FEATURE, last_checked_boundary=B0)
    res = A.resolve_activation(cfg, node, now_block=B2 + 5, record=rec)
    assert not res.changed and not res.record.locked           # waits on B
    assert node.as_of_reads == [B] and res.record.last_checked_boundary == B0
    # B becomes readable (or the node moved to an archive endpoint): same
    # state, same block, same decision as the peer
    node.fail_blocks.clear()
    res = A.resolve_activation(cfg, node, now_block=B2 + 40, record=res.record)
    assert res.record.lock_block == B and res.record.activation_block == B2
    assert res.record.source == "tally" and node.as_of_reads == [B, B]


def test_resolver_late_joiner_adopts_the_signers_earlier_lock_over_its_own_boundary(cfg):
    """A fresh record's first tally lands on a later boundary; the signers
    it counts already name an earlier lock-in with most of the SIGNED stake
    (under 51% of the total after drift) — theirs is the fleet's block."""
    B, B2 = B0 + GRID, B0 + 2 * GRID
    locked = A.format_signal(FEATURE, lock_block=B, activation_block=B2)
    ready = A.format_signal(FEATURE)
    vals = [_v("a", 46.0), _v("o", 4.0), _v("c", 30.0), _v("d", 15.0), _v("e", 5.0)]
    sig = {"a": locked, "o": locked, "c": ready}
    chain = FakeChain(vals, sig, block=B2 + 5)
    res = A.resolve_activation(cfg, chain, now_block=B2 + 5, record=A.ActivationRecord())
    assert res.tally is not None and res.tally.locked          # 80% signed at B2
    assert (res.record.lock_block, res.record.activation_block) == (B, B2)
    assert res.record.source == "signals" and res.record.last_checked_boundary == B2
    # signers naming an INADMISSIBLE earlier pair are ignored: own lock stands
    chain.signals["a"] = chain.signals["o"] = A.format_signal(
        FEATURE, lock_block=B, activation_block=B2 + GRID)
    res = A.resolve_activation(cfg, chain, now_block=B2 + 5, record=A.ActivationRecord())
    assert (res.record.lock_block, res.record.activation_block) == (B2, B2 + GRID)
    assert res.record.source == "tally"


def test_resolver_blanks_a_typed_in_record_once_the_keys_are_cleared(cfg):
    typed = replace(cfg, round=replace(cfg.round, rolling_from_block=B0 + GRID,
                                       epoch_blocks_prev=GRID, epoch_activation_block=B0 + GRID,
                                       epoch_blocks=900),
                    scoring=replace(cfg.scoring, era_king_from_block=B0 + GRID,
                                    tenure_blocks_from_block=B0 + GRID,
                                    cohort_maxt_increment_from_block=B0 + GRID))
    chain = FakeChain(_fleet(30, 70), {"v1": A.format_signal(FEATURE)}, block=B0 + 3)
    rec = A.resolve_activation(typed, chain, now_block=B0 + 3, record=A.ActivationRecord()).record
    assert rec.locked and rec.source == "config"
    # the owner removes the typed-in block: the chain decides again — the
    # record is blanked, and the next pass tallies as if fresh
    res = A.resolve_activation(cfg, chain, now_block=B0 + 3, record=rec)
    assert res.changed and res.record == A.ActivationRecord()
    res = A.resolve_activation(cfg, chain, now_block=B0 + 3, record=res.record)
    assert res.tally is not None and res.record.last_checked_boundary == B0
    assert not res.record.locked


def test_resolver_does_not_consume_a_boundary_on_an_empty_metagraph(cfg):
    chain = FakeChain([], {}, block=B0 + 3)
    res = A.resolve_activation(cfg, chain, now_block=B0 + 3, record=A.ActivationRecord())
    assert not res.changed and res.record.last_checked_boundary == 0
    # the read comes back real next poll: the SAME boundary is counted
    chain.validators = _fleet(60, 40)
    chain.signals = {"v1": A.format_signal(FEATURE)}
    res = A.resolve_activation(cfg, chain, now_block=B0 + 30, record=res.record)
    assert res.record.lock_block == B0 and res.record.activation_block == B0 + GRID


def test_own_signal_is_silent_under_a_typed_in_rollover(cfg):
    typed = replace(cfg, round=replace(cfg.round, rolling_from_block=B0 + GRID,
                                       epoch_blocks_prev=GRID, epoch_activation_block=B0 + GRID,
                                       epoch_blocks=900),
                    scoring=replace(cfg.scoring, era_king_from_block=B0 + GRID,
                                    tenure_blocks_from_block=B0 + GRID,
                                    cohort_maxt_increment_from_block=B0 + GRID))
    rec = A.ActivationRecord(feature=FEATURE, activation_block=B0 + GRID, source="config")
    assert A.own_signal_payload(typed, rec) is None
    chain = FakeChain(_fleet(30, 70), {}, block=B0 + 3)
    assert not A.ensure_signal(chain, typed, rec, hotkey="v1") and chain.written == []
    assert A.own_signal_payload(cfg, A.ActivationRecord()) == A.format_signal(FEATURE)


def test_resolver_skips_a_pruned_boundary_nobody_locked_in_at(cfg):
    """A public endpoint discards state after ~256 blocks. A boundary this
    node cannot read any more, that no validator's note marks as a lock-in,
    is skipped — otherwise a node restarted across it (or a fresh record
    whose latest boundary is already old) would never count again."""
    B, B2 = B0 + GRID, B0 + 2 * GRID
    ready = A.format_signal(FEATURE)
    chain = _PerBlockChain(_fleet(60, 40), {"v1": ready}, block=B + A.PRUNED_AFTER_BLOCKS + 5)
    chain.fail_blocks = {B}
    # transient: younger than the prune window ⇒ retried, not skipped
    res = A.resolve_activation(cfg, chain, now_block=B + 50, record=A.ActivationRecord())
    assert not res.changed and res.record.last_checked_boundary == 0
    # old enough and nobody names a lock ⇒ skipped, no lock-in
    res = A.resolve_activation(cfg, chain, now_block=chain.block, record=A.ActivationRecord())
    assert res.changed and not res.record.locked and res.record.last_checked_boundary == B
    # the next boundary is counted as of its own block
    chain.block = B2 + 5
    res = A.resolve_activation(cfg, chain, now_block=chain.block, record=res.record)
    assert res.record.lock_block == B2 and res.record.activation_block == B2 + GRID
    assert chain.as_of_reads[-1] == B2
    # ... but NOT skipped when a validator's note names a lock-in there:
    # the decision exists and must be adopted, never tallied past
    locked = A.format_signal(FEATURE, lock_block=B, activation_block=B2)
    chain2 = _PerBlockChain(_fleet(60, 40), {"v2": locked}, block=B + A.PRUNED_AFTER_BLOCKS + 5)
    chain2.fail_blocks = {B}
    res = A.resolve_activation(cfg, chain2, now_block=chain2.block, record=A.ActivationRecord())
    assert not res.changed and res.record.last_checked_boundary == 0
