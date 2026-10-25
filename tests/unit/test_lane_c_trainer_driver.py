"""DEC-CA-0056 lane C — the trainer's LaneCDriver: dispatch per target from the
lane-C queue, recipe rejection = miner fault, infra requeue, orphans at
restart, and the settlement (signed manifest, exact-text digest, write-ahead
republish of the SAME bytes after a crash). MarketCast-only lane C (owner
2026-10-20): the targets are MarketCast ``mkt:`` suites."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.funding.queue import LaneCQueue, lane_c_entry_ref
from cascade.shared.chain import Commitment
from cascade.shared.config.lane_c import LaneCConfig
from cascade.shared.config.marketcast import parse_marketcast
from cascade.shared.lane_c_manifest import (
    format_finetune_commit,
    lane_c_manifest_key,
    lane_c_text_digest,
    load_lane_c_manifest,
)
from cascade.shared.manifest import format_trained_pointer
from cascade.trainer.finetune import LegRecipe, RecipeError
from cascade.trainer.lane_c_driver import LaneCDriver, LegFailure

E, S = "mkt:returns", "mkt:vol"
SU = ("returns", "vol")
FROM = 9000
MC = replace(parse_marketcast({}), from_block=FROM)       # armed: dispatches from FROM
REF = {hk: f"{hk.lower()}/recipe@sha256:" + (str(i) * 64) for i, hk in
       enumerate(("ALFA", "BRAV", "CHAR", "DELT"), start=1)}
UID = {"ALFA": 1, "BRAV": 2, "CHAR": 3, "DELT": 4}


def _ptr(tag: str) -> str:
    return format_trained_pointer(f"cascade/ckpt@sha256:{(tag.encode().hex() * 64)[:64]}")


KING = _ptr("king1")


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeOps:
    def __init__(self, tmp_path, clock):
        self.q = LaneCQueue(tmp_path / "lane_c_queue.json", clock=clock)
        self.commits: list = []
        self.busy: set[str] = set()
        self.recipes: dict = {}
        self.trained: list = []
        self.published: list = []
        self.fail_publish = False
        self.train = lambda leg, recipe: {"trained_pointer": _ptr("ft-" + leg["hotkey"]),
                                          "corpus_digest": "c" * 64,
                                          "train_block": leg["started_block"],
                                          "gpu_name": "H100"}
        self.defer = False
        self.deferred: list = []

    def commitments(self):
        return self.commits

    def queue(self):
        return self.q

    def lane_a_busy(self, hk):
        return hk in self.busy

    def now(self):
        return 1_000_000.0

    def fetch_recipe(self, ref, target, **kw):
        r = self.recipes.get(ref, LegRecipe(target=target, lr_scale=0.5, steps_fraction=1.0,
                                            warmup_fraction=0.05, schedule="cosine"))
        if isinstance(r, Exception):
            raise r
        return r

    def train_leg(self, leg, recipe):
        self.trained.append((leg["hotkey"], recipe))
        return self.train(leg, recipe)

    def sign(self, m):
        return replace(m, signature="ab" * 32)

    def publish(self, key, text):
        if self.fail_publish:
            raise OSError("bucket 503")
        self.published.append((key, text))

    def spawn(self, fn, name):
        if self.defer:
            self.deferred.append(fn)
        else:
            fn()


def _cfg(**kw):
    lc = LaneCConfig(from_block=FROM, targets=(E, S), legs_per_target=1, **kw)
    return SimpleNamespace(lane_c=lc, marketcast=MC)


def _fund(ops, hk, target, block=100):
    ops.q.add_lane_c(hk, target, REF[hk], block)
    ops.commits.append(Commitment(uid=UID[hk], hotkey=hk, coldkey=None,
                                  payload=format_finetune_commit(target, REF[hk],
                                                                 mkt_suites=SU),
                                  commit_block=block))


@pytest.fixture
def env(tmp_path):
    clock = Clock()
    ops = FakeOps(tmp_path, clock)
    drv = LaneCDriver(_cfg(), ops, tmp_path / "lane_c_driver.json")
    return drv, ops, tmp_path


def test_inert_before_from_block(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    assert drv.tick(FROM - 5, 9, KING, FROM - 3600) == []
    assert ops.q.get("ALFA").status == "queued"


def test_dispatch_respects_legs_per_target_and_settles(env):
    drv, ops, _ = env
    for hk, t in (("ALFA", E), ("BRAV", E), ("CHAR", S)):
        _fund(ops, hk, t)
    ops.defer = True
    started = drv.tick(FROM + 10, 10, KING, FROM)
    # one slot per target: ALFA (returns, senior) and CHAR (vol); BRAV waits
    assert sorted(leg["hotkey"] for leg in started) == ["ALFA", "CHAR"]
    assert ops.q.get("BRAV").status == "queued"
    assert ops.q.get("ALFA").status == "in_flight"
    assert {leg["base_pointer"] for leg in started} == {KING}
    for fn in ops.deferred:
        fn()
    assert len(drv.st.finished) == 2 and drv.st.open == []
    progress = drv.status()["submission_progress"]
    assert {r["stage"] for r in progress} == {"awaiting_settlement"}
    assert all(r["attempt_id"] == ops.q.get(r["hotkey"]).attempt_id for r in progress)
    # the finished leg keeps its queue entry in flight until it settles
    assert ops.q.get("ALFA").status == "in_flight"
    digest = drv.settle(round_id="777", block=FROM + 900, era_index=10, base_pointer=KING)
    (key, text), = ops.published
    assert key == lane_c_manifest_key("777")
    assert digest == lane_c_text_digest(text)
    m = load_lane_c_manifest(text)
    assert m.base_pointer == KING and m.round_id == "777"
    assert [(e.miner_hotkey, e.target, e.recipe_ref) for e in m.entries] == [
        ("ALFA", E, REF["ALFA"]), ("CHAR", S, REF["CHAR"])]
    assert all(e.funded_by == "miner" for e in m.entries)
    assert ops.q.get("ALFA").status == "done" and drv.st.finished == []
    progress = drv.status()["submission_progress"]
    assert {r["stage"] for r in progress} == {"awaiting_forward_judgement"}
    assert {r["round_id"] for r in progress} == {"777"}
    restored = LaneCDriver(drv.cfg, ops, drv.path)
    assert restored.status()["submission_progress"] == progress
    # idempotent per round: the same digest, nothing republished
    assert drv.settle(round_id="777", block=FROM + 900, era_index=10, base_pointer=KING) == digest
    assert len(ops.published) == 1
    assert drv.published("777") == {"key": key, "text": text}
    # the next tick starts the waiting returns entry
    ops.defer = False
    assert [leg["hotkey"] for leg in drv.tick(FROM + 920, 10, KING, FROM)] == ["BRAV"]


def test_invalid_recipe_is_a_miner_fault_that_ends_the_entry(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    ops.recipes[REF["ALFA"]] = RecipeError("invalid recipe.json: lr_scale 9 must be in (0, 1]")
    drv.tick(FROM + 10, 10, KING, FROM)
    e = ops.q.get("ALFA")
    assert (e.status, e.last_error_class) == ("failed", "recipe")
    assert ops.trained == [] and drv.st.finished == [] and drv.st.open == []
    assert drv.st.outcomes[-1]["outcome"] == "miner_fault"


def test_infra_failure_requeues_and_operator_fault_backs_off(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)

    def boom(leg, recipe):
        raise LegFailure("pod died", miner_fault=False, error_class="infra", burn=True)

    ops.train = boom
    drv.tick(FROM + 10, 10, KING, FROM)
    e = ops.q.get("ALFA")
    assert (e.status, e.attempts) == ("queued", 1)
    ops.recipes[REF["ALFA"]] = OSError("hub 503")      # fetching the ref is our side
    drv.tick(FROM + 20, 10, KING, FROM)
    e = ops.q.get("ALFA")
    assert (e.status, e.attempts, e.last_error_class) == ("queued", 1, "operator")
    assert e.not_before > 0


def test_superseded_commitment_and_lane_a_busy(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    _fund(ops, "BRAV", S)
    ops.busy.add("BRAV")                                # a lane-A leg of BRAV runs
    ops.commits.append(Commitment(uid=1, hotkey="ALFA", coldkey=None,
                                  payload=f"metro-v1:gen:hippius:{REF['ALFA']}",
                                  commit_block=200))
    assert drv.tick(FROM + 10, 10, KING, FROM) == []
    assert ops.q.get("ALFA").status == "failed"
    assert ops.q.get("ALFA").last_error_class == "superseded"
    assert ops.q.get("BRAV").status == "queued"        # waits, never fails


def test_restart_requeues_open_legs_uncharged(env):
    drv, ops, tmp = env
    _fund(ops, "ALFA", E)
    ops.defer = True
    drv.tick(FROM + 10, 10, KING, FROM)
    assert ops.q.get("ALFA").status == "in_flight"
    # the process dies with the leg's thread; a new driver loads the state
    drv2 = LaneCDriver(_cfg(), ops, tmp / "lane_c_driver.json")
    assert len(drv2.st.open) == 1
    ops.defer = False
    ops.commits = []                                    # nothing re-dispatches this tick
    drv2.tick(FROM + 20, 10, KING, FROM)
    e = ops.q.get("ALFA")
    assert (e.status, e.attempts) == ("queued", 0) and drv2.st.open == []
    assert e.operator_billed
    assert drv2.st.outcomes[-1]["outcome"] == "orphaned"
    # an orphan's payer pod is not protected from the lane-A sweep
    assert drv2.live_hotkeys() == set()


def test_in_flight_entry_without_a_leg_is_requeued(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    assert ops.q.mark_in_flight("ALFA", lane_c_entry_ref(E, REF["ALFA"]), target_boundary=0,
                                era_index=10, started_block=5)
    ops.commits = []
    drv.tick(FROM + 10, 10, KING, FROM)
    assert ops.q.get("ALFA").status == "queued"


def test_crash_after_write_ahead_republishes_the_same_bytes(env):
    drv, ops, tmp = env
    _fund(ops, "ALFA", E)
    drv.tick(FROM + 10, 10, KING, FROM)
    ops.fail_publish = True
    with pytest.raises(OSError):
        drv.settle(round_id="42", block=FROM + 900, era_index=10, base_pointer=KING)
    written = drv.st.pending["text"]
    # restart: more legs finish in between — the round's bytes must not change
    drv2 = LaneCDriver(_cfg(), ops, tmp / "lane_c_driver.json")
    drv2.st.finished.append(json.loads(json.dumps(drv2.st.finished[0])) | {
        "trained_pointer": _ptr("ft-other")})
    ops.fail_publish = False
    digest = drv2.settle(round_id="42", block=FROM + 905, era_index=10, base_pointer=KING)
    (key, text), = ops.published
    assert text == written and digest == lane_c_text_digest(written)
    assert len(drv2.st.finished) == 1                   # the newcomer waits for the next round


def test_unbound_write_ahead_of_another_round_is_discarded(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    drv.tick(FROM + 10, 10, KING, FROM)
    ops.fail_publish = True
    with pytest.raises(OSError):
        drv.settle(round_id="42", block=FROM + 900, era_index=10, base_pointer=KING)
    ops.fail_publish = False
    # the lane-A manifest of 42 went out without it; the leg settles with 43
    digest = drv.settle(round_id="43", block=FROM + 1800, era_index=10, base_pointer=KING)
    (key, text), = ops.published
    assert key == lane_c_manifest_key("43") and digest == lane_c_text_digest(text)
    assert drv.lane_c_digest("42") == ""


def test_status_reports_slots_queue_and_legs(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    _fund(ops, "BRAV", E)
    ops.defer = True
    drv.tick(FROM + 10, 10, KING, FROM)
    st = drv.status()
    assert st["targets"][E] == {"queued": 1, "open": 1, "finished": 0, "slots": 1}
    assert st["queue"] == {"in_flight": 1, "queued": 1}
    assert drv.live_hotkeys() == {"ALFA"}


def test_submission_confirmation_survives_omission_restart_and_old_receipt(env):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    drv.tick(FROM + 10, 10, KING, FROM)
    drv.settle(round_id="first", block=FROM + 900, era_index=10, base_pointer=KING)
    item = drv.status()["submission_progress"][-1]
    receipt = {"round_id": "confirm", "epoch_start_block": FROM + 1800, "status": "scored",
               "lane_c": [{"hotkey": "ALFA", "target": E, "outcome": "confirmed",
                           "trained_pointer": item["trained_pointer"],
                           "train_block": item["train_block"]}]}
    drv.observe_receipt(receipt)
    assert drv.status()["submission_progress"][-1]["stage"] == "confirmed"
    drv.observe_receipt({"status": "scored", "epoch_start_block": FROM + 2700, "lane_c": []})
    restored = LaneCDriver(drv.cfg, ops, drv.path)
    assert restored.status()["submission_progress"][-1]["stage"] == "confirmed"
    older = {**receipt, "epoch_start_block": FROM + 900,
             "lane_c": [{**receipt["lane_c"][0], "outcome": "pending"}]}
    restored.observe_receipt(older)
    assert restored.status()["submission_progress"][-1]["stage"] == "confirmed"
    other = {**receipt, "epoch_start_block": FROM + 3600,
             "lane_c": [{**receipt["lane_c"][0], "train_block": FROM - 100,
                         "outcome": "dropped"}]}
    restored.observe_receipt(other)
    assert restored.status()["submission_progress"][-1]["stage"] == "confirmed"


def test_submission_receipt_write_failure_can_retry_without_changing_funding(env, monkeypatch):
    drv, ops, _ = env
    _fund(ops, "ALFA", E)
    drv.tick(FROM + 10, 10, KING, FROM)
    drv.settle(round_id="first", block=FROM + 900, era_index=10, base_pointer=KING)
    item = drv.status()["submission_progress"][-1]
    receipt = {"round_id": "confirm", "epoch_start_block": FROM + 1800, "status": "scored",
               "lane_c": [{"hotkey": "ALFA", "target": E, "outcome": "confirmed",
                           "trained_pointer": item["trained_pointer"],
                           "train_block": item["train_block"]}]}
    save = drv.save
    def unavailable():
        raise OSError("fixture disk unavailable")
    monkeypatch.setattr(drv, "save", unavailable)
    with pytest.raises(OSError, match="fixture disk unavailable"):
        drv.observe_receipt(receipt)
    assert drv.st.submission_verdicts == []
    assert ops.q.get("ALFA").status == "done"
    monkeypatch.setattr(drv, "save", save)
    drv.observe_receipt(receipt)
    assert drv.status()["submission_progress"][-1]["stage"] == "confirmed"
