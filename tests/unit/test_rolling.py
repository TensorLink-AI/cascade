"""DEC-CA-0043 on the trainer: rolling intake, era king, settlements.

The scheduler's policy runs against a fake :class:`LegOps` (no pods, no
buckets) and a real :class:`FundedQueue` on disk, so every scenario the DEC
requires is exercised end to end: cross-era admission and within-era slip,
bench at completion with the payer pod torn down inside the bench window,
an era without a king leg publishing nothing, the stale-metagraph king, a
dethrone mid-era adopting the winner's checkpoint, restart re-entry, ref
rebinding, and the generation switch landing on its announced era. Every
settlement manifest is then fed to an armed validator to prove the two
sides agree on the envelope."""
from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from cascade.funding.queue import FundedQueue
from cascade.shared.chain import Commitment
from cascade.shared.era import era_for_block
from cascade.shared.manifest import TrainedEntry, format_trained_pointer
from cascade.trainer import rolling as R
from cascade.trainer.rolling import LegOps, RollingScheduler, admit, target_boundary
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis

REF = {hk: f"{hk.lower()}/gen@sha256:" + (hk[-1] * 64) for hk in ("KING", "ALFA", "BRAV", "CHAR")}
REF["KING"] = "king/gen@sha256:" + "0" * 64
REF["ALFA"] = "alfa/gen@sha256:" + "1" * 64
REF["BRAV"] = "brav/gen@sha256:" + "2" * 64
REF["CHAR"] = "char/gen@sha256:" + "3" * 64
REF_ALFA2 = "alfa/gen@sha256:" + "9" * 64
UID = {"KING": 0, "ALFA": 1, "BRAV": 2, "CHAR": 3}
WALL = 3.75 * 3600
MARGIN = 900.0
EB = 900                      # the post-rollover grid


def _ptr(tag: str) -> str:
    h = (tag.encode().hex() * 64)[:64]
    return format_trained_pointer(f"cascade/ckpt@sha256:{h}")


def _armed(cfg, rollover_eras: int = 10):
    rollover = EB * 4 * rollover_eras
    round_cfg = replace(cfg.round, epoch_blocks=EB, rolling_from_block=rollover,
                        era_settlements=4, funded_field_cap=3, dedup_mode="off",
                        one_submission_per_hotkey=False)
    scoring = replace(cfg.scoring, era_king_from_block=rollover,
                      tenure_blocks_from_block=rollover)
    return replace(cfg, round=round_cfg, scoring=scoring)


def _commit(hk: str, ref: str, block: int) -> Commitment:
    return Commitment(uid=UID[hk], hotkey=hk, coldkey=None,
                      payload=f"metro-v1:gen:hippius:{ref}", commit_block=block)


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeOps(LegOps):
    """Every pod/bucket touch recorded; legs return instantly."""

    def __init__(self, tmp_path: Path, clock: Clock) -> None:
        self.tmp = tmp_path
        self.clock = clock
        self.king_hk = "KING"
        self.chain_king = "KING"
        self.commits: list[Commitment] = [_commit("KING", REF["KING"], 1)]
        self.failures: dict[str, tuple] = {}
        self.fail_king = False
        self.king_exc: Exception | None = None   # raised by train_king when set
        self.rotated: list[tuple[int, str]] = []
        self.king_calls: list[tuple] = []
        self.leg_calls: list[tuple] = []
        self.torn_down: list[tuple[str, float]] = []
        self.retired: list[int] = []
        self.manifests: list = []
        self.benches: list[tuple] = []
        self.rosters: list = []
        self.burnt: list = []
        self.boundaries: list[tuple] = []
        self.gens: tuple = (0, [], 0)
        self.sweeps: list = []
        self.cached: dict[tuple, TrainedEntry] = {}
        self.latest_round = "LEGACY"          # latest.json's round_id (the chain root)
        self.hold: dict[str, threading.Event] = {}   # legs that block until released
        self.records: dict[int, int] = {}     # published promotions/gen-<n>.json effective eras
        self.bench_scores = {"gifteval_crps": 0.5, "gifteval_mase": 0.5, "boom_crps": 0.5,
                             "boom_mase": 0.5, "time_crps": 0.5, "time_mase": 0.5}

    # chain / storage
    def block_seed(self, client, block):
        return int(block) * 7 + 1

    def commitments(self, client):
        return list(self.commits)

    def receipt_king(self):
        return self.king_hk

    def latest_round_id(self):
        return self.latest_round

    def promotion_effective_era(self, generation):
        return self.records.get(int(generation))

    def metagraph_king(self, client):
        return self.chain_king

    # legs
    def train_challenger(self, gen, era, block, *, end_wall):
        self.leg_calls.append((gen.hotkey, era.index, block, end_wall))
        gate = self.hold.get(gen.hotkey)
        if gate is not None:
            gate.wait(10)
        if gen.hotkey in self.failures:
            raise RuntimeError("leg failed")
        return TrainedEntry(gen.hotkey, gen.uid, "challenger", gen.ref,
                            _ptr(f"{gen.hotkey}-{era.index}"), "d", block, gpu_name="RTX 4090")

    def train_king(self, gen, era, block):
        self.king_calls.append((gen.hotkey, era.index, block))
        if self.fail_king:
            raise self.king_exc or RuntimeError("king rent failed")
        return TrainedEntry(gen.hotkey, gen.uid, "king", gen.ref,
                            _ptr(f"K{gen.hotkey}-{era.index}"), "d", block, gpu_name="RTX 4090")

    def cached_leg(self, era, role, gen):
        return self.cached.get((era.index, role, gen.hotkey))

    def bench_challenger(self, entry, king, era):
        self.benches.append(("challenger", entry.miner_hotkey, era.index, self.clock()))
        return dict(self.bench_scores)

    def bench_king(self, entry, era):
        self.benches.append(("king", entry.miner_hotkey, era.index, self.clock()))
        return dict(self.bench_scores)

    def leg_failure(self, hotkey):
        return self.failures.get(hotkey, ("boom", False, "infra", True))

    def teardown_kept_pod(self, hotkey):
        self.torn_down.append((hotkey, self.clock()))

    def retire_king_pod(self, era):
        self.retired.append(era.index)

    def rotate_king_pod(self, era, reason):
        self.rotated.append((era.index, reason))

    # publication
    def publish_manifest(self, manifest):
        self.manifests.append(manifest)

    def publish_bench(self, round_id, created_block, entries):
        report = SimpleNamespace(round_id=round_id, created_block=created_block,
                                 entries=[(e.role, e.miner_hotkey) for e, _ in entries])
        self.benches.append(("report", round_id, report.entries))
        return report

    def publish_roster(self, round_id, roster):
        self.rosters.append((round_id, roster))

    def promotion_boundary(self, king_hotkey, epoch_start, round_id, effective_era):
        self.boundaries.append((king_hotkey, epoch_start, round_id, effective_era))

    def promotion_generations(self):
        return self.gens

    def record_bench_candidates(self, manifest, report):
        pass

    def publish_champion(self, gen, round_id):
        pass

    # queue / burn / dedup
    def queue(self):
        return FundedQueue(self.tmp / "funded_queue.json", clock=self.clock)

    def burned(self):
        return set()

    def burn(self, gens):
        self.burnt.extend(g.hotkey for g in gens)

    def vault_owned(self, gens):
        return list(gens)

    def dedup_registry(self):
        return None

    def sweep_pods(self, *, keep_round_ids, keep_payers):
        self.sweeps.append((keep_round_ids, set(keep_payers)))

    def wall_seconds(self):
        return WALL

    def margin_seconds(self):
        return MARGIN

    def fitting_skus(self):
        return ("RTX4090",)


class FakeClient:
    def uid_for_hotkey(self, hk):
        return UID.get(hk, -1)


def _sched(cfg, tmp_path, clock, ops=None, provenance=None):
    runner = SimpleNamespace(cfg=cfg, work_root=tmp_path, promotion=None,
                             pool_provenance_fn=provenance,
                             _rolling_note_king_host=lambda era: None)
    ops = ops or FakeOps(tmp_path, clock)
    return RollingScheduler(runner, ops, clock=clock), ops


def _join(sched, timeout=10.0):
    end = time.time() + timeout
    while (sched._threads or sched._king_threads) and time.time() < end:
        time.sleep(0.01)
    assert not sched._threads and not sched._king_threads, "legs still running"


def _advance(clock, ops, sched, client, *, from_block, to_block):
    """Move the chain forward: the clock follows the block grid."""
    clock.t += (to_block - from_block) * R.BLOCK_SECONDS
    sched.tick(client, to_block)
    _join(sched)


def _validator(cfg, ops=None, *, last_handled="LEGACY"):
    """A validator that handled the last LEGACY round (its persisted position)
    and binds refs against the fake chain's reveal history."""
    def fake_eval(entry, windows):
        return []

    state = replace(genesis("KING", 0), last_handled_round_id=last_handled)
    history = (lambda: list(ops.commits)) if ops is not None else (lambda: [])
    return ValidatorRunner(cfg=cfg, state=state, evaluate_fn=fake_eval,
                           verify_signatures=False, commitment_history_fn=history)


# ── pure policy ──────────────────────────────────────────────────────────────

def test_target_boundary_is_the_first_the_wall_clears(cfg):
    armed = _armed(cfg)
    now = 1_000_000.0
    b0 = armed.round.rolling_from_block + 10
    # a 3.75h wall + 15min margin at 12s/block = 1350 blocks ⇒ skips two 900-block boundaries
    b = target_boundary(armed.round, block_now=b0, now=now, wall_seconds=WALL,
                        margin_seconds=MARGIN)
    assert b % EB == 0 and b > b0
    assert R.wall_of_block(b, now=now, block_now=b0) > now + WALL + MARGIN
    assert R.wall_of_block(b - EB, now=now, block_now=b0) <= now + WALL + MARGIN


def test_cross_era_admission_waits_for_the_pretrain_window(cfg):
    armed = _armed(cfg)
    now = 1_000_000.0
    era_start = armed.round.rolling_from_block
    era_len = EB * 4
    cur = era_for_block(armed.round, era_start).index
    # early in the era: lands in this era, starts now
    a = admit(armed.round, block_now=era_start + 10, now=now, wall_seconds=WALL,
              margin_seconds=MARGIN, current_era=cur)
    assert a.era_index == cur and a.start_after == now and not a.cross_era
    # a leg that can still clear the era's LAST settlement (its end block)
    # stays in this era, however late
    late = era_start + era_len - 1400          # 16800 s before the era rolls (> wall + margin)
    a = admit(armed.round, block_now=late, now=now, wall_seconds=WALL,
              margin_seconds=MARGIN, current_era=cur)
    assert not a.cross_era and a.era_index == cur and a.target_boundary == era_start + era_len
    # inside the last wall + margin it cannot: it starts NOW under the next
    # era (the pre-train window is open by construction — no dead zone)
    later = era_start + era_len - 1100
    a = admit(armed.round, block_now=later, now=now, wall_seconds=WALL,
              margin_seconds=MARGIN, current_era=cur)
    nxt = era_start + era_len
    assert a.cross_era and a.era_index == cur + 1 and a.start_after == now
    assert a.target_boundary >= nxt + EB
    assert R.king_pretrain_open(armed.round, block_now=later, now=now, wall_seconds=WALL,
                                margin_seconds=MARGIN)
    assert not R.king_pretrain_open(armed.round, block_now=late, now=now, wall_seconds=WALL,
                                    margin_seconds=MARGIN)


def test_generation_ledger_resolves_the_init_per_era():
    gens = {"1": {"members": [["m1a", "s"], ["m1b", "s"]], "effective_era": 0},
            "2": {"members": [["m2a", "s"]], "effective_era": 12}}
    assert R.era_init(gens, 10) == (1, 0, "m1a", "s")
    assert R.era_init(gens, 11) == (1, 1, "m1b", "s")
    assert R.era_init(gens, 12) == (2, 0, "m2a", "s")
    assert R.era_init({}, 5) == (0, 0, "", "")


# ── the scheduler ────────────────────────────────────────────────────────────

def test_settlement_manifest_chains_and_the_validator_accepts_it(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits += [_commit("ALFA", REF["ALFA"], b0 - 100), _commit("BRAV", REF["BRAV"], b0 - 50)]
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    q.add("BRAV", REF["BRAV"], reveal_block=b0 - 50)
    sched.tick(client, b0)
    _join(sched)
    # king leg trained under the era, both challengers in flight → finished
    assert [c[0] for c in ops.king_calls] == ["KING"]
    assert {c[0] for c in ops.leg_calls} == {"ALFA", "BRAV"}
    st = sched.state
    assert st.current.index == era_for_block(armed.round, b0).index
    assert st.current.king_entry is not None
    assert len(st.finished) == 2
    # the payer pods were torn down at completion, not at a boundary
    assert {hk for hk, _ in ops.torn_down} == {"ALFA", "BRAV"}
    # nothing settles before a boundary passes
    assert ops.manifests == []
    # first boundary after the legs landed: ONE manifest, king + both
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 3)
    assert len(ops.manifests) == 1
    m = ops.manifests[0]
    assert m.round_id == str(b1 * 7 + 1)
    assert m.era["index"] == st.current.index and m.era["start_block"] == era_start
    # the first settlement chains to latest.json (the last legacy round)
    assert m.prev_round_id == "LEGACY"
    assert [e.role for e in m.entries] == ["king", "challenger", "challenger"]
    assert [e.miner_hotkey for e in m.entries][1:] == ["ALFA", "BRAV"]
    assert all(e.train_block == b0 for e in m.entries)
    # the bench report carries the king ONCE (era's first report) + both challengers
    reports = [b for b in ops.benches if b[0] == "report"]
    assert reports[0][2] == [("king", "KING"), ("challenger", "ALFA"), ("challenger", "BRAV")]
    # queue: settled entries are done and burned; the roster is published
    assert {q.get(hk).status for hk in ("ALFA", "BRAV")} == {"done"}
    assert sorted(ops.burnt) == ["ALFA", "BRAV"]
    assert ops.rosters[-1][1]["mode"] == "rolling" and len(ops.rosters[-1][1]["seated"]) == 2
    # a validator that handled the legacy round accepts exactly this manifest;
    # one positioned elsewhere walks the chain instead of judging it
    v = _validator(armed, ops)
    assert v.check_manifest(m) is None
    assert _validator(armed, ops, last_handled="OTHER").check_manifest(m).startswith(
        "manifest_chain_broken")
    # second boundary, nothing new: no manifest; a third leg later chains to the first
    b2 = b1 + EB
    _advance(clock, ops, sched, client, from_block=b1 + 3, to_block=b2 + 3)
    assert len(ops.manifests) == 1
    ops.commits.append(_commit("CHAR", REF["CHAR"], b2))
    q.add("CHAR", REF["CHAR"], reveal_block=b2)
    sched.tick(client, b2 + 4)
    _join(sched)
    b3 = b2 + EB
    _advance(clock, ops, sched, client, from_block=b2 + 4, to_block=b3 + 1)
    assert len(ops.manifests) == 2
    m2 = ops.manifests[1]
    assert m2.prev_round_id == m.round_id
    assert [e.miner_hotkey for e in m2.entries] == ["KING", "CHAR"]
    # king's bench is NOT repeated in the second report
    reports = [b for b in ops.benches if b[0] == "report"]
    assert reports[1][2] == [("challenger", "CHAR")]
    v.state = replace(v.state, last_handled_round_id=m.round_id)
    assert v.check_manifest(m2) is None


def test_era_without_a_king_leg_publishes_nothing_and_holds_challengers(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    ops.fail_king = True
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits.append(_commit("ALFA", REF["ALFA"], b0 - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    assert sched.state.current.king_entry is None
    assert sched.state.current.king_leg_failed
    assert len(sched.state.finished) == 1
    _advance(clock, ops, sched, client, from_block=b0, to_block=era_start + EB + 2)
    assert ops.manifests == []                       # never judged against another era's king
    assert len(sched.state.finished) == 1
    assert q.get("ALFA").status == "in_flight"
    # the king leg retries each tick; once it lands the next boundary settles
    ops.fail_king = False
    _advance(clock, ops, sched, client, from_block=era_start + EB + 2, to_block=era_start + EB + 10)
    assert sched.state.current.king_entry is not None
    _advance(clock, ops, sched, client, from_block=era_start + EB + 10, to_block=era_start + 2 * EB + 1)
    assert len(ops.manifests) == 1
    assert [e.miner_hotkey for e in ops.manifests[0].entries] == ["KING", "ALFA"]


def test_stale_metagraph_king_never_reaches_a_manifest(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    ops.king_hk, ops.chain_king = "BRAV", "KING"           # receipts moved, incentive lags
    ops.commits.append(_commit("BRAV", REF["BRAV"], 5))
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    ops.commits.append(_commit("ALFA", REF["ALFA"], era_start - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=era_start - 100)
    sched.tick(client, era_start + 5)
    _join(sched)
    assert sched.state.current.king_hotkey == "BRAV"
    assert ops.king_calls[0][0] == "BRAV"
    _advance(clock, ops, sched, client, from_block=era_start + 5, to_block=era_start + EB + 1)
    assert ops.manifests[0].entry_for_role("king").miner_hotkey == "BRAV"


def test_dethrone_mid_era_adopts_the_winner_and_keeps_the_era(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits.append(_commit("ALFA", REF["ALFA"], b0 - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 1)
    winner_ptr = ops.manifests[0].entry_for_role("challenger").trained_pointer
    era_idx = sched.state.current.index
    # validators crowned ALFA at that settlement
    ops.king_hk = "ALFA"
    ops.commits.append(_commit("BRAV", REF["BRAV"], b1))
    q.add("BRAV", REF["BRAV"], reveal_block=b1)
    sched.tick(client, b1 + 2)
    _join(sched)
    cur = sched.state.current
    assert cur.index == era_idx                       # the era did not end
    assert cur.king_hotkey == "ALFA" and cur.king().trained_pointer == winner_ptr
    assert cur.king_bench_published                   # its numbers already landed
    assert len(ops.king_calls) == 1                   # no operator retrain
    b2 = b1 + EB
    _advance(clock, ops, sched, client, from_block=b1 + 2, to_block=b2 + 1)
    m2 = ops.manifests[1]
    king = m2.entry_for_role("king")
    assert king.miner_hotkey == "ALFA" and king.trained_pointer == winner_ptr
    assert m2.era["index"] == era_idx
    # the validator that crowned ALFA judges the next settlement at that pointer
    v = _validator(armed, ops)
    v.state = replace(v.state, king_hotkey="ALFA", king_uid=1, king_pointer=winner_ptr,
                      era_index=era_idx, last_handled_round_id=ops.manifests[0].round_id)
    assert v.check_manifest(m2) is None


def test_restart_mid_era_restores_state_and_reattaches_in_flight_legs(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits.append(_commit("ALFA", REF["ALFA"], b0 - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    king_ptr = sched.state.current.king().trained_pointer
    # simulate a crash: the finished leg is persisted; put a second leg in flight
    ops.commits.append(_commit("BRAV", REF["BRAV"], b0))
    q.add("BRAV", REF["BRAV"], reveal_block=b0)
    q.mark_in_flight("BRAV", REF["BRAV"], target_boundary=era_start + 2 * EB,
                     era_index=sched.state.current.index, started_block=b0 + 1)
    ops2 = FakeOps(tmp_path, clock)
    ops2.commits = list(ops.commits)
    sched2, _ = _sched(armed, tmp_path, clock, ops=ops2)
    assert sched2.state.current.index == sched.state.current.index
    assert sched2.state.current.king().trained_pointer == king_ptr
    assert len(sched2.state.finished) == 1
    sched2.tick(client, b0 + 3)
    _join(sched2)
    # the in-flight leg was re-attached (not re-rented from scratch by the
    # intake), its pod kept through the startup sweep, the king leg reused
    assert ops2.king_calls == []
    assert [c[0] for c in ops2.leg_calls] == ["BRAV"]
    assert ops2.sweeps[0][1] == {"ALFA", "BRAV"}       # every in-flight payer pod kept
    assert str(sched2.state.current.base_seed) in ops2.sweeps[0][0]
    assert len(sched2.state.finished) == 2
    assert q.get("BRAV").status == "in_flight"


def test_recommit_in_flight_queues_behind_and_never_rebinds(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits.append(_commit("ALFA", REF["ALFA"], b0 - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    # hold the leg inside the intake by making the leg thread slow
    orig = ops.train_challenger
    gate = {"go": False}

    def slow(gen, era, block, *, end_wall):
        while not gate["go"]:
            time.sleep(0.005)
        return orig(gen, era, block, end_wall=end_wall)

    ops.train_challenger = slow
    sched.tick(client, b0)
    assert q.get("ALFA").status == "in_flight"
    assert q.add("ALFA", REF_ALFA2, reveal_block=b0 + 1) == "queued-behind"
    assert q.get("ALFA").ref == REF["ALFA"] and q.get("ALFA").queued_ref == REF_ALFA2
    gate["go"] = True
    _join(sched)
    ops.commits.append(_commit("ALFA", REF_ALFA2, b0 + 1))    # the re-commit reveals
    ops.train_challenger = orig
    _advance(clock, ops, sched, client, from_block=b0, to_block=era_start + EB + 1)
    judged = ops.manifests[0].entry_for_role("challenger")
    assert judged.gen_ref == REF["ALFA"]              # the train_block ref, not the re-commit
    # the parked ref became ALFA's next entry and was admitted in its own right
    nxt = q.get("ALFA")
    assert nxt.ref == REF_ALFA2 and nxt.status == "in_flight" and nxt.attempts == 0
    assert ops.leg_calls[-1][0] == "ALFA" and len(ops.leg_calls) == 2


def test_failed_leg_settles_from_the_fault_taxonomy(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    for hk in ("ALFA", "BRAV"):
        ops.commits.append(_commit(hk, REF[hk], b0 - 100))
        q.add(hk, REF[hk], reveal_block=b0 - 100)
    ops.failures["ALFA"] = ("rc=3 generator crashed", True, "generator", False)
    ops.failures["BRAV"] = ("sold out", False, "no_capacity", False)
    sched.tick(client, b0)
    _join(sched)
    assert q.get("ALFA").status == "failed" and "ALFA" in ops.burnt
    assert q.get("BRAV").status == "queued" and q.get("BRAV").attempts == 0
    assert sched.state.finished == []


def test_late_funding_starts_under_the_next_era_and_the_king_pretrains(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    era_len = EB * 4
    sched.tick(client, era_start + 5)
    _join(sched)
    era0 = sched.state.current.index
    # still able to clear the era's LAST settlement (its end block): this era
    early = era_start + era_len - 1400          # 16800 s before the era rolls (> wall + margin)
    ops.commits.append(_commit("ALFA", REF["ALFA"], early))
    q.add("ALFA", REF["ALFA"], reveal_block=early)
    _advance(clock, ops, sched, client, from_block=era_start + 5, to_block=early)
    assert q.get("ALFA").era_index == era0 and q.get("ALFA").target_boundary == era_start + era_len
    assert sched.state.next is None
    # inside the last wall + margin: the next era's king pre-trains and a leg
    # funded now starts at once under the NEXT era's seeds (no dead zone)
    window = era_start + era_len - 1100
    ops.commits.append(_commit("BRAV", REF["BRAV"], window))
    q.add("BRAV", REF["BRAV"], reveal_block=window)
    _advance(clock, ops, sched, client, from_block=early, to_block=window)
    assert sched.state.next is not None and sched.state.next.index == era0 + 1
    assert [c[1] for c in ops.king_calls] == [era0, era0 + 1]
    assert q.get("BRAV").status == "in_flight" and q.get("BRAV").era_index == era0 + 1
    assert q.get("BRAV").target_boundary >= era_start + era_len + EB
    assert {f.hotkey: f.era_index for f in sched.state.finished} == {"ALFA": era0, "BRAV": era0 + 1}
    # the era's last settlement (its end block) judges ALFA against era0's king,
    # then the pre-trained king is adopted and the old pod retired
    end = era_start + era_len
    _advance(clock, ops, sched, client, from_block=window, to_block=end + 1)
    assert len(ops.manifests) == 1
    m = ops.manifests[0]
    assert m.era["index"] == era0 and [e.miner_hotkey for e in m.entries] == ["KING", "ALFA"]
    assert sched.state.current.index == era0 + 1 and sched.state.next is None
    assert ops.retired == [era0]
    assert len(ops.king_calls) == 2
    v = _validator(armed, ops)
    assert v.check_manifest(m) is None
    # the new era's first settlement carries BRAV under the new seeds
    _advance(clock, ops, sched, client, from_block=end + 1, to_block=end + EB + 1)
    m2 = ops.manifests[1]
    assert m2.era["index"] == era0 + 1 and m2.era["seed_block"] == era_start
    assert [e.miner_hotkey for e in m2.entries] == ["KING", "BRAV"]
    assert m2.prev_round_id == m.round_id
    v.state = replace(v.state, last_handled_round_id=m.round_id)
    assert v.check_manifest(m2) is None


def test_generation_switch_lands_on_its_announced_era(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    m1 = format_trained_pointer("cascade/ckpt@sha256:" + "a" * 64)
    m2 = format_trained_pointer("cascade/ckpt@sha256:" + "b" * 64)
    ops.gens = (1, [(m1, "toto2-4m")], 0)
    sched.tick(client, era_start + 5)
    _join(sched)
    era0 = sched.state.current.index
    assert sched.state.current.warm_start_ckpt == m1
    # the engine fires generation 2 effective two eras out
    ops.gens = (2, [(m2, "toto2-4m")], era0 + 2)
    sched.tick(client, era_start + 6)
    assert sched.state.generations["2"]["effective_era"] == era0 + 2
    assert R.era_init(sched.state.generations, era0 + 1)[2] == m1
    assert R.era_init(sched.state.generations, era0 + 2)[2] == m2
    # the pre-train window of era0+1 resolves the OLD init; era0+2's the new
    era_len = EB * 4
    _advance(clock, ops, sched, client, from_block=era_start + 6,
             to_block=era_start + era_len - 1100)
    assert sched.state.next.index == era0 + 1 and sched.state.next.warm_start_ckpt == m1
    _advance(clock, ops, sched, client, from_block=era_start + era_len - 1100,
             to_block=era_start + 2 * era_len - 1100)
    assert sched.state.current.index == era0 + 1
    assert sched.state.next.index == era0 + 2 and sched.state.next.warm_start_ckpt == m2
    assert sched.state.next.generation == 2


def test_promotion_boundary_step_runs_once_per_settlement_with_the_notice_era(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    sched.tick(client, era_start + 5)
    _join(sched)
    sched.tick(client, era_start + 6)
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=era_start + 6, to_block=b1 + 1)
    sched.tick(client, b1 + 2)
    assert [b[1] for b in ops.boundaries] == [era_start, b1]
    from cascade.shared.era import min_effective_era
    assert ops.boundaries[-1][3] == min_effective_era(armed.round, b1)


# ── the funded queue's in-flight lifecycle ───────────────────────────────────

def test_queue_in_flight_lifecycle(tmp_path):
    clock = Clock()
    q = FundedQueue(tmp_path / "q.json", clock=clock)
    q.add("ALFA", REF["ALFA"], reveal_block=10)
    assert not q.mark_in_flight("ALFA", "other", target_boundary=900, era_index=1, started_block=5)
    assert q.mark_in_flight("ALFA", REF["ALFA"], target_boundary=900, era_index=1, started_block=5)
    e = q.get("ALFA")
    assert (e.status, e.target_boundary, e.era_index, e.started_block) == ("in_flight", 900, 1, 5)
    assert [x.hotkey for x in q.in_flight()] == ["ALFA"]
    # a round-entry recovery never touches a flight; a withdraw is refused
    assert q.recover_in_round() == 0 and q.get("ALFA").status == "in_flight"
    assert not q.withdraw("ALFA")
    assert q.retarget_flight("ALFA", target_boundary=1800)
    assert q.get("ALFA").target_boundary == 1800
    # same-ref re-fund is proof of life; new ref queues behind
    assert q.add("ALFA", REF["ALFA"], reveal_block=10) == "already-queued"
    assert q.add("ALFA", REF_ALFA2, reveal_block=11) == "queued-behind"
    # requeue clears the flight fields; the parked ref supersedes it
    assert q.requeue("ALFA", error="sold out", error_class="no_capacity", burn_attempt=False)
    e = q.get("ALFA")
    assert (e.status, e.ref, e.target_boundary, e.queued_ref) == ("queued", REF_ALFA2, 0, "")
    # done on a flight without a parked ref stays done
    assert q.mark_in_flight("ALFA", REF_ALFA2, target_boundary=2700, era_index=1, started_block=6)
    q.mark_done("ALFA")
    assert q.get("ALFA").status == "done"
    # public view shows the target for flights
    q.add("BRAV", REF["BRAV"], reveal_block=12)
    q.mark_in_flight("BRAV", REF["BRAV"], target_boundary=2700, era_index=1, started_block=7)
    row = next(r for r in q.public_view()["entries"] if r["hotkey"] == "BRAV")
    assert row["target_boundary"] == 2700 and row["era_index"] == 1


def test_queue_file_round_trips_flight_fields(tmp_path):
    q = FundedQueue(tmp_path / "q.json")
    q.add("ALFA", REF["ALFA"], reveal_block=10)
    q.mark_in_flight("ALFA", REF["ALFA"], target_boundary=900, era_index=3, started_block=5)
    q.add("ALFA", REF_ALFA2, reveal_block=11)
    again = FundedQueue(tmp_path / "q.json").get("ALFA")
    assert (again.status, again.target_boundary, again.era_index, again.started_block,
            again.queued_ref, again.queued_reveal_block) == ("in_flight", 900, 3, 5, REF_ALFA2, 11)


# ── bit-identity before ROLLOVER ─────────────────────────────────────────────

def test_rolling_is_inert_before_the_gate(cfg):
    from cascade.shared.era import rolling_active

    armed = _armed(cfg)
    assert not rolling_active(cfg.round, 10**9)
    assert not rolling_active(armed.round, armed.round.rolling_from_block - 1)
    assert rolling_active(armed.round, armed.round.rolling_from_block)


def test_state_file_round_trip(tmp_path):
    st = R.RollingState(
        current=R.EraState(index=3, start_block=10800, seed_block=7200, base_seed=99,
                           king_hotkey="K", king_entry=None),
        generations={"1": {"members": [["m", "s"]], "effective_era": 0}},
        finished=[R.FinishedLeg(hotkey="A", uid=1, ref="r", era_index=3, started_block=10801,
                                reveal_block=5, entry={"miner_hotkey": "A", "miner_uid": 1,
                                                       "role": "challenger", "gen_ref": "r",
                                                       "trained_pointer": _ptr("x"),
                                                       "corpus_digest": "d", "train_block": 1})],
        last_published_round_id="7", last_settled_boundary=10800)
    p = tmp_path / "era_state.json"
    R.save_state(p, st)
    again = R.load_state(p)
    assert again == st
    assert R.load_state(tmp_path / "missing.json") == R.RollingState()


# ── review fixes (PR #296) ───────────────────────────────────────────────────

def test_pool_pin_is_the_settlement_boundary_not_the_era_start(cfg, tmp_path):
    """Validators verify the pin at the settlement's epoch boundary; an era
    that spans a daily snapshot's effective block would fail every settlement
    after the crossing if the trainer pinned at the era start."""
    armed = _armed(cfg)
    clock = Clock()
    pins = []

    def provenance(seed, block):
        pins.append((seed, block))
        return f"pool/{block}.tar", "s" * 64

    sched, ops = _sched(armed, tmp_path, clock, provenance=provenance)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits.append(_commit("ALFA", REF["ALFA"], b0 - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 3)
    assert len(ops.manifests) == 1
    assert pins == [(b1 * 7 + 1, b1)]
    assert ops.manifests[0].eval_pool_key == f"pool/{b1}.tar"


def test_a_boundary_missed_during_an_outage_is_settled_late_not_thrown_away(cfg, tmp_path):
    """The trainer is away across the era's LAST boundary: on return the
    settlement is published stamped with that boundary (validators floor
    created_block, so it resolves to the era the legs trained under) and the
    payers are never re-billed for checkpoints that already exist."""
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    era_end = era_start + EB * 4
    b3 = era_start + EB * 3
    b0 = era_start + 5
    sched.tick(client, b0)                      # king leg only
    _join(sched)
    era_idx = sched.state.current.index
    # two legs admitted with the era's LAST boundary as their target (a
    # 1200-block wall + margin from here clears era_end, nothing earlier)
    bs = era_end - 1300
    _advance(clock, ops, sched, client, from_block=b0, to_block=bs)
    ops.commits += [_commit("ALFA", REF["ALFA"], bs - 10), _commit("BRAV", REF["BRAV"], bs - 9)]
    q.add("ALFA", REF["ALFA"], reveal_block=bs - 10)
    q.add("BRAV", REF["BRAV"], reveal_block=bs - 9)
    ops.hold = {"ALFA": threading.Event(), "BRAV": threading.Event()}
    sched.tick(client, bs)
    assert {e.target_boundary for e in q.in_flight()} == {era_end}
    # the third settlement passes with the legs still running: nothing published
    clock.t += (b3 + 3 - bs) * R.BLOCK_SECONDS
    sched.tick(client, b3 + 3)
    assert ops.manifests == []
    # the legs land before era_end …
    for gate in ops.hold.values():
        gate.set()
    _join(sched)
    assert len(sched.state.finished) == 2
    # … but the trainer is away across era_end and returns one grid step later
    late = era_end + EB + 3
    _advance(clock, ops, sched, client, from_block=b3 + 3, to_block=late)
    assert len(ops.manifests) == 1
    m = ops.manifests[0]
    assert m.created_block == era_end and m.round_id == str(era_end * 7 + 1)
    assert m.era["index"] == era_idx
    assert [e.miner_hotkey for e in m.entries][1:] == ["ALFA", "BRAV"]
    assert {q.get(hk).status for hk in ("ALFA", "BRAV")} == {"done"}
    assert sorted(ops.burnt) == ["ALFA", "BRAV"]
    # the era rolled afterwards; nothing was requeued
    assert sched.state.current.index == era_idx + 1
    assert sched.state.finished == []
    # a validator resolves the late manifest to the era it belongs to
    v = _validator(armed, ops)
    assert v.check_manifest(m) is None


def test_no_settlement_is_published_against_an_unknown_chain_root(cfg, tmp_path):
    """latest.json unreadable at the first settlement: an unchained manifest
    would be rejected by every validator AFTER the legs were marked done and
    burned — so nothing is published and the legs wait."""
    armed = _armed(cfg)
    clock = Clock()
    ops = FakeOps(tmp_path, clock)
    ops.latest_round = ""
    sched, ops = _sched(armed, tmp_path, clock, ops=ops)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    ops.commits.append(_commit("ALFA", REF["ALFA"], b0 - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=b0, to_block=b1 + 3)
    assert ops.manifests == [] and ops.burnt == []
    assert q.get("ALFA").status == "in_flight" and len(sched.state.finished) == 1
    # the root becomes readable: the next boundary settles, chained to it
    ops.latest_round = "LEGACY"
    b2 = b1 + EB
    _advance(clock, ops, sched, client, from_block=b1 + 3, to_block=b2 + 3)
    assert len(ops.manifests) == 1 and ops.manifests[0].prev_round_id == "LEGACY"
    assert q.get("ALFA").status == "done"


def test_generation_ledger_rebuilds_from_the_published_record(cfg, tmp_path):
    """The engine clears its pending record at publish; a lost era_state.json
    must learn the generation's effective era from promotions/gen-<n>.json
    (what validators install from), never treat first sight as 'live now'."""
    armed = _armed(cfg)
    clock = Clock()
    ops = FakeOps(tmp_path, clock)
    ops.gens = (2, [("cascade/ckpt@sha256:" + "a" * 64, "toto2-4m")], 0)
    ops.records = {2: 12}
    sched, ops = _sched(armed, tmp_path, clock, ops=ops)
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    sched.tick(client, era_start + 5)
    _join(sched)
    assert sched.state.generations["2"]["effective_era"] == 12
    era_idx = sched.state.current.index
    assert era_idx < 12 and sched.state.current.generation == 0      # random init until era 12
    # a pre-era record (effective_era 0) is live at once
    ops2 = FakeOps(tmp_path / "b", clock)
    ops2.gens, ops2.records = ops.gens, {2: 0}
    sched2, _ = _sched(armed, tmp_path / "b", clock, ops=ops2)
    sched2.tick(client, era_start + 5)
    _join(sched2)
    assert sched2.state.generations["2"]["effective_era"] == 0
    assert sched2.state.current.generation == 2
    # a NEW generation whose record cannot be read (store outage) is never
    # guessed: the ledger waits, and picks up the published era on a later tick
    ops.gens = (3, [("cascade/ckpt@sha256:" + "b" * 64, "toto2-4m")], 0)
    sched.tick(client, era_start + 6)
    _join(sched)
    assert "3" not in sched.state.generations
    ops.records[3] = 14
    sched.tick(client, era_start + 7)
    _join(sched)
    assert sched.state.generations["3"]["effective_era"] == 14


def test_seniors_that_could_not_start_are_not_passed_over(cfg, tmp_path):
    """The roster's seniority evidence lists only startable seniors that were
    jumped: an unrevealed or cap-held senior is not a jump (the audit would
    otherwise WARN on nearly every settlement with a queue)."""
    armed = _armed(cfg)                       # funded_field_cap = 3
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    client = FakeClient()
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    # ALFA is the most senior but unrevealed on chain; BRAV and CHAR revealed
    q.add("ALFA", REF["ALFA"], reveal_block=b0 - 300)
    ops.commits += [_commit("BRAV", REF["BRAV"], b0 - 200), _commit("CHAR", REF["CHAR"], b0 - 100)]
    q.add("BRAV", REF["BRAV"], reveal_block=b0 - 200)
    q.add("CHAR", REF["CHAR"], reveal_block=b0 - 100)
    sched.tick(client, b0)
    _join(sched)
    rents = {r["hotkey"]: r for r in sched.state.rents}
    assert set(rents) == {"BRAV", "CHAR"}
    assert rents["BRAV"]["passed_over"] == [] and rents["CHAR"]["passed_over"] == []
    assert q.get("ALFA").status == "queued"


# ── king pod rotation: a booted pod that keeps failing is a lemon host ────────


def test_king_pod_rotates_after_the_same_pod_retry_is_spent(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    ops.fail_king = True
    ops.king_exc = RuntimeError("king training failed on remote: hippius read stalled")
    client = FakeClient()
    era_start = armed.round.rolling_from_block
    b0 = era_start + 5
    sched.tick(client, b0)
    _join(sched)
    cur = sched.state.current
    # first failure: the same pod gets ONE retry — nothing rotated
    assert cur.king_leg_failures == 1 and ops.rotated == []
    assert "hippius read stalled" in cur.king_leg_failed
    _advance(clock, ops, sched, client, from_block=b0, to_block=b0 + 1)
    # second consecutive failure on that pod: rotated (host quarantined, pod
    # torn down by the runner), counter reset so the NEXT pod gets its own retry
    assert [i for i, _ in ops.rotated] == [cur.index]
    assert "failed 2x on this pod" in ops.rotated[0][1]
    assert cur.king_leg_failures == 0
    _advance(clock, ops, sched, client, from_block=b0 + 1, to_block=b0 + 2)
    assert len(ops.rotated) == 1 and cur.king_leg_failures == 1
    # the leg lands on the fresh pod: counters clear
    ops.fail_king = False
    _advance(clock, ops, sched, client, from_block=b0 + 2, to_block=b0 + 3)
    assert cur.king_entry is not None
    assert cur.king_leg_failures == 0 and cur.king_leg_failed == ""


def test_king_pod_rotates_at_once_on_a_transport_failure(cfg, tmp_path):
    from cascade.trainer.remote import RemoteDispatchError

    armed = _armed(cfg)
    clock = Clock()

    sched, ops = _sched(armed, tmp_path, clock)
    ops.fail_king = True
    ops.king_exc = RemoteDispatchError("remote king on funded-king: pod unreachable for 901s",
                                       returncode=255)
    client = FakeClient()
    b0 = armed.round.rolling_from_block + 5
    sched.tick(client, b0)
    _join(sched)
    cur = sched.state.current
    assert [i for i, _ in ops.rotated] == [cur.index]
    assert cur.king_leg_failures == 0            # fresh pod next tick
    assert cur.king_entry is None


def test_king_pod_rotation_state_round_trips(cfg, tmp_path):
    armed = _armed(cfg)
    clock = Clock()
    sched, ops = _sched(armed, tmp_path, clock)
    ops.fail_king = True
    client = FakeClient()
    b0 = armed.round.rolling_from_block + 5
    sched.tick(client, b0)
    _join(sched)
    assert sched.state.current.king_leg_failures == 1
    sched2, _ = _sched(armed, tmp_path, clock, ops=ops)
    assert sched2.state.current.king_leg_failures == 1
