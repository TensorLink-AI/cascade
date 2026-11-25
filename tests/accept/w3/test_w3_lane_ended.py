"""W3 (#931): the lane D intake honours the ``[lane_d] until_block`` kill switch.

Containment drill 2026-11-25: after a lane-wide kill (an owner-signed
overlay setting ``until_block``) the funding intake kept admitting. The lane
D intake had the same gap: it read ``[lane_d]`` once from ``chain.toml`` and
never checked ``until_block``. Contract (service docstring "Kill switch"):

* at or after ``until_block`` in the config IN FORCE at the head (the static
  config, or ``cfg_at(head)`` — the adopted overlay) a new submit, and a
  ready that would open an entry, answer ``409 lane_ended`` with the end
  block; nothing is stored, no grant is minted;
* before it (or with no end) nothing changes;
* entries already in the queue are not rewritten (the validator reads the
  queue file);
* ``cascade.lane_d.main.LaneDView`` resolves ``[lane_d]`` through the
  read-only netconfig resolver at the head.
"""

from __future__ import annotations

from types import SimpleNamespace

from .helpers import BLOCK, Env, Miner, checkpoint, digest_of, lane_d_cfg, manifest


def _require():
    """Fail on main at the missing contract (an assertion, not a crash)."""
    from cascade.lane_d import service

    assert "lane_ended" in service.REFUSAL_CODES, "lane D has no lane_ended refusal"


class ViewEnv(Env):
    """An intake whose ``[lane_d]`` in force comes from ``cfg_at(head)``."""

    def __init__(self, tmp, *, cfg_at, **kw):
        self.cfg_at = cfg_at
        super().__init__(tmp, **kw)

    def build(self):
        from cascade.lane_d.service import LaneDIntake

        return LaneDIntake(storage=self.store, reads=self.reads, minter=self.minter,
                           cfg=self.cfg, ops=self.ops, state_dir=self.state_dir,
                           now=self.clock, per_ip_rpm=self.per_ip_rpm,
                           per_hotkey_rpm=self.per_hotkey_rpm, cfg_at=self.cfg_at)


def test_w3_submit_after_until_block_is_lane_ended(tmp_path):
    _require()
    env = Env(tmp_path, cfg=lane_d_cfg(until_block=BLOCK))
    m = Miner(env, "alice")
    r = m.submit(checkpoint("alice"))
    assert (r.status, r.code) == (409, "lane_ended"), r
    assert r.json["until_block"] == BLOCK and r.json["lane"] == "d"
    assert env.minter.minted == [] and env.queue() == []


def test_w3_before_until_block_and_no_end_are_unchanged(tmp_path):
    _require()
    for i, cfg in enumerate((lane_d_cfg(until_block=BLOCK + 1), lane_d_cfg())):
        env = Env(tmp_path / str(i), cfg=cfg)
        m = Miner(env, "alice")
        r = m.enter(checkpoint("alice"))
        assert r.status == 200 and r.json["status"] == "ready", r


def test_w3_overlay_kill_closes_ready_and_keeps_existing_rows(tmp_path):
    _require()
    view = {"until": 0}

    def cfg_at(block):
        return lane_d_cfg(until_block=view["until"])

    env = ViewEnv(tmp_path, cfg_at=cfg_at)
    early = Miner(env, "alice")
    assert early.enter(checkpoint("alice")).status == 200
    late = Miner(env, "bob")
    files = checkpoint("bob")
    assert late.submit(files).status == 201
    late.upload(files)
    # the overlay in force now ends lane D at the head
    view["until"] = BLOCK
    r = late.ready(files)
    assert (r.status, r.code) == (409, "lane_ended"), r
    assert r.json["until_block"] == BLOCK
    r = Miner(env, "carol").submit(checkpoint("carol"))
    assert (r.status, r.code) == (409, "lane_ended"), r
    # rows already in the queue are untouched (the validator reads the file)
    states = {e["hotkey"]: e["state"] for e in env.queue()}
    assert states == {early.hotkey: "ready", late.hotkey: "submitted"}
    assert env.entry(late.hotkey, digest_of(manifest(files)))["state"] == "submitted"


def test_w3_unreadable_view_keeps_the_static_config(tmp_path):
    _require()

    def broken(block):
        raise RuntimeError("persistence pending")

    env = ViewEnv(tmp_path, cfg_at=broken)
    assert Miner(env, "alice").submit(checkpoint("alice")).status == 201
    env2 = ViewEnv(tmp_path / "b", cfg_at=broken, cfg=lane_d_cfg(until_block=BLOCK))
    assert Miner(env2, "alice").submit(checkpoint("alice")).code == "lane_ended"


def test_w3_lane_d_view_follows_the_resolver(tmp_path):
    from cascade.lane_d.main import LaneDView

    class Resolver:
        def __init__(self):
            self.ticks = []

        def tick(self, cfg, client, *, now_block):
            self.ticks.append(now_block)
            return False

        def config_at(self, cfg, block):
            until = 500 if block >= 400 else 0
            return SimpleNamespace(lane_d=lane_d_cfg(until_block=until))

    t = {"now": 0.0}
    res = Resolver()
    view = LaneDView(SimpleNamespace(), res, object(), poll_seconds=12.0,
                     clock=lambda: t["now"])
    assert view(100).until_block == 0
    assert view(450).until_block == 500        # inside the poll window: no tick
    assert res.ticks == [100]
    t["now"] = 13.0
    assert view(460).until_block == 500 and res.ticks == [100, 460]
