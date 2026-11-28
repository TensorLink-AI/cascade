"""W-EVALPUB: publish each settlement's judged eval set once it can never be
judged again; signed lane B / lane C manifests and lane D reports public-read.

Torch-free; the S3 side is an in-memory fake (no network) and the eval host
runs in-process. Every test reaches the new code through ``helpers.mod``,
which asserts it exists: on main every test fails by assertion.
"""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.validator.lane_d_forward import eligible_windows

from .helpers import (
    DAY,
    HORIZON,
    HOUR,
    LAG,
    T0,
    FakeS3,
    LocalHost,
    builder,
    evals,
    lane_d_cfg,
    load,
    report_for,
    write_pool,
)

END = T0 + 200 * HOUR                     # the newest data point of the fixture snapshot
STEP = 300                                # 5-minute points: a 64-point window spans 5.25 h
BASE = "https://s3.example/cascade-manifests"


def _band_world(tmp_path, *, step=STEP, freq="5min"):
    root = tmp_path / "pool_d"
    d = write_pool(root / "pool_d-1", end=END, band=True, step=step, freq=freq)
    return root, d, load(d, band=True)


def _cfg():
    return lane_d_cfg(horizon_bands=("short",), max_horizon=2048, band_slots=2)


def _retire_now(windows) -> int:
    """The earliest ``now`` at which the windows are retired (no open entry,
    state written at ``now``): max first-seen + 6 h slack + 48 h embargo."""
    return max(int(max(w["first_seen_ts"])) for w in windows) + 54 * HOUR


# ── lane D: the proof ───────────────────────────────────────────────────────


def test_laned_judged_window_is_redrawable_so_immediate_publication_would_leak(tmp_path):
    """The current draw RE-USES windows: a window judged at settlement N (for
    an entry revealed before N) is still eligible for an entry revealed 30 h
    AFTER N (as an embargo window), so it is not retired at N, nor 30 h on."""
    m = evals()
    _, _, pool = _band_world(tmp_path)
    cfg = _cfg()
    t_n = END + LAG + 600                                 # the settlement (snapshot as_of)
    judged = [w for w, v in zip(pool["windows"], eligible_windows(
        pool, t_entry=END - 10 * HOUR, t_king=0, cfg=cfg), strict=True) if v.eligible]
    assert judged, "the fixture must judge something at N"
    later = {v.window_id for v in eligible_windows(pool, t_entry=t_n + 30 * HOUR, t_king=0,
                                                   cfg=cfg) if v.eligible}
    assert {w["window_id"] for w in judged} & later, "a judged window is re-drawable after N"
    for now in (t_n, t_n + 30 * HOUR):
        _, cut = m.laned_cut({"entries": {}}, now=now, state_mtime=now, embargo_hours=48.0)
        assert any(m.window_retired(w["target_ts"], w["first_seen_ts"], cut=cut, now=now)
                   for w in judged)


def test_laned_retired_values_never_enter_an_eligible_window(tmp_path):
    """At the first instant the rule retires the judged windows, every
    window (any band, either snapshot) holding a published value is
    ineligible for every entry the state may still judge or that may still
    be revealed (any reveal >= protect_ts), at every feed embargo <= 48 h."""
    m = evals()
    root = tmp_path / "pool_d"
    pools = [load(write_pool(root / f"pool_d-{i}", end=END + i * 3 * HOUR, band=True,
                             step=STEP, freq="5min"), band=True) for i in range(2)]
    cfg = _cfg()
    judged = [w for w, v in zip(pools[0]["windows"], eligible_windows(
        pools[0], t_entry=END - 2 * HOUR, t_king=0, cfg=cfg), strict=True) if v.eligible]
    assert judged
    now = _retire_now(judged)
    protect, cut = m.laned_cut({"entries": {}}, now=now, state_mtime=now, embargo_hours=48.0)
    assert all(m.window_retired(w["target_ts"], w["first_seen_ts"], cut=cut, now=now) is None
               for w in judged)
    # one second earlier: not retired
    _, cut1 = m.laned_cut({"entries": {}}, now=now - 1, state_mtime=now - 1, embargo_hours=48.0)
    assert any(m.window_retired(w["target_ts"], w["first_seen_ts"], cut=cut1, now=now - 1)
               for w in judged)
    published_max = max(int(max(w["target_ts"])) for w in judged)   # history is older
    for e in (0.0, 12.0, 48.0):
        c = lane_d_cfg(horizon_bands=("short",), band_slots=2, embargo_hours=e)
        for t in range(protect, protect + 400 * HOUR, HOUR):
            for p in pools:
                for w, v in zip(p["windows"], eligible_windows(p, t_entry=t, t_king=0, cfg=c), strict=True):
                    if int(min(w["target_ts"])) <= published_max:
                        assert not v.eligible, (w["window_id"], t, e)


def test_laned_open_older_entry_delays_and_malformed_reveal_publishes_nothing():
    m = evals()
    now = END + 100 * HOUR
    st = {"entries": {"old": {"status": "pending", "reveal_ts": END - 70 * HOUR},
                      "done": {"status": "lost", "reveal_ts": 1}}}
    protect, cut = m.laned_cut(st, now=now, state_mtime=now, embargo_hours=48.0)
    assert protect == END - 76 * HOUR and cut == END - 124 * HOUR
    assert m.laned_cut({"entries": {"x": {"status": "pending"}}}, now=now, state_mtime=now,
                       embargo_hours=48.0) == (None, None)
    ts = [END - STEP * i for i in range(HORIZON)][::-1]
    assert m.window_retired(ts, [t + LAG for t in ts], cut=cut, now=now) == "target_after_cut"
    assert m.window_retired(ts, [t + LAG for t in ts], cut=END, now=now) == "first_seen_after_cut"
    assert m.window_retired(ts, [t - 1 for t in ts], cut=END, now=now) is None
    assert m.window_retired(ts, ts, cut=END + 1, now=END - 1) == "future_target"
    assert m.window_retired([], [], cut=END, now=now) == "missing_clock"


# ── lane D: the host build ──────────────────────────────────────────────────


@pytest.mark.parametrize("band", [False, True])
def test_host_build_is_the_settlement_cut(tmp_path, band):
    """windows.npz holds each judged window's history (last history_cap
    points) and target exactly as cascade.validator.lane_d.load_pool cut it."""
    b = builder()
    root = tmp_path / "pool_d"
    d = write_pool(root / "pool_d-1", end=END, band=band, step=HOUR, freq="H")
    pool = load(d, band=band)
    assert b.snapshot_sha256(d) == pool["sha256"]
    rep = report_for(9000, pool, pool["windows"], boundary_ts=END)
    req = evals().build_request(rep, cut=END + LAG, now=END + LAG, history_cap=100)
    reply = b.build(req, pool_root=root, out_dir=tmp_path / "out")
    assert reply["ok"] and reply["windows"] == len(pool["windows"])
    z = np.load(reply["npz"])
    by = {w["window_id"]: w for w in pool["windows"]}
    for i, wid in enumerate(z["window_id"].tolist()):
        w = by[wid]
        h0, h1 = z["history_offsets"][i], z["history_offsets"][i + 1]
        t0, t1 = z["target_offsets"][i], z["target_offsets"][i + 1]
        k0, k1 = z["time_offsets"][i], z["time_offsets"][i + 1]
        assert np.array_equal(z["history"][h0:h1], np.asarray(w["history"][-100:], np.float32))
        assert np.array_equal(z["target"][t0:t1], np.asarray(w["target"], np.float32))
        assert np.array_equal(z["target_ts"][k0:k1], np.asarray(w["target_ts"]))
        assert np.array_equal(z["first_seen_ts"][k0:k1], np.asarray(w["first_seen_ts"]))
        assert int(z["origin_ts"][i]) == int(w["target_ts"][0])
        assert int(z["h"][i]) == len(w["target_ts"]) and int(z["n_channels"][i]) == 1


def test_host_refuses_unretired_and_wrong_snapshot(tmp_path):
    b = builder()
    root = tmp_path / "pool_d"
    d = write_pool(root / "pool_d-1", end=END, band=True)
    pool = load(d, band=True)
    rep = report_for(9000, pool, pool["windows"], boundary_ts=END)
    req = evals().build_request(rep, cut=END + LAG - 1, now=END + DAY, history_cap=50)
    reply = b.build(req, pool_root=root, out_dir=tmp_path / "out")
    assert reply["ok"] is False and reply["code"] == "not_retired"
    assert not (tmp_path / "out" / "windows.npz").exists()
    bad = json.loads(json.dumps(req))
    bad["windows"][0]["snapshot_sha256"] = "0" * 64
    for w in bad["windows"]:
        w["snapshot_sha256"] = "0" * 64
    bad["cut_ts"] = END + DAY
    with pytest.raises(b.BuildError, match="snapshot_digest_mismatch"):
        b.build(bad, pool_root=root, out_dir=tmp_path / "out")


# ── lane D: the box pass ────────────────────────────────────────────────────


def _laned_setup(tmp_path):
    root = tmp_path / "pool_d"
    d = write_pool(root / "pool_d-1", end=END, band=True, step=STEP, freq="5min")
    pool = load(d, band=True)
    reports = tmp_path / "lane_d_state_reports"
    reports.mkdir()
    rep = report_for(9603900, pool, pool["windows"], boundary_ts=END + HOUR)
    (reports / "lane_d_report_9603900.json").write_text(json.dumps(rep))
    (reports / "lane_d_report_9603000.json").write_text(json.dumps(
        report_for(9603000, pool, [], boundary_ts=END)))
    return root, reports, pool


def _run_laned(m, s3, reports, host, tmp_path, now, dry_run=False):
    return m.publish_laned(client=s3, bucket="b", base_url=BASE, reports_dir=reports,
                           state={"entries": {}}, state_mtime=now, embargo_hours=48.0,
                           host=host, stage=tmp_path / "stage", now=now, history_cap=64,
                           dry_run=dry_run, log=lambda s: None)


def test_publish_laned_waits_then_publishes_public_read_manifest_last(tmp_path):
    m = evals()
    root, reports, pool = _laned_setup(tmp_path)
    s3, host = FakeS3(), LocalHost(root, tmp_path / "host")
    now = _retire_now(pool["windows"])
    far = _run_laned(m, s3, reports, host, tmp_path, now - 6 * HOUR)
    assert far["built"] == [] and not host.requests            # the report check waits
    assert [w["boundary"] for w in far["plan"]["waiting"]] == [9603900]
    assert far["plan"]["empty"] == [9603000]
    early = _run_laned(m, s3, reports, host, tmp_path, now - 1)  # the host's full check waits
    assert early["built"] == [] and len(host.requests) == 1
    idx = json.loads(s3.objects[f"{m.LANED_PREFIX}/index.json"]["body"])
    assert idx["items"] == [] and idx["waiting"] == [{"boundary": 9603900,
                                                      "reason": "not_retired"}]
    assert [k for k in s3.objects if not k.endswith("index.json")] == []
    dry = _run_laned(m, s3, reports, host, tmp_path, now, dry_run=True)
    assert dry["built"] == [9603900] and set(s3.objects) <= {f"{m.LANED_PREFIX}/index.json"}
    live = _run_laned(m, s3, reports, host, tmp_path, now)
    assert live["built"] == [9603900]
    pre = f"{m.LANED_PREFIX}/9603900/"
    keys = [k for k in s3.puts if k.startswith(pre)]
    assert keys == [pre + "windows.npz", pre + "README.md", pre + "manifest.json"]
    assert all(s3.objects[k]["ACL"] == "public-read" for k in keys)
    man = json.loads(s3.objects[pre + "manifest.json"]["body"])
    assert man["windows"] == len(pool["windows"])
    assert man["max_first_seen_ts"] <= man["retired_rule"]["cut_ts"]
    idx = json.loads(s3.objects[f"{m.LANED_PREFIX}/index.json"]["body"])
    assert s3.objects[f"{m.LANED_PREFIX}/index.json"]["ACL"] == "public-read"
    assert [i["boundary"] for i in idx["items"]] == [9603900]
    z = np.load(io.BytesIO(s3.objects[pre + "windows.npz"]["body"]))
    assert int(z["first_seen_ts"].max()) <= man["retired_rule"]["cut_ts"]
    n = len(s3.puts)
    again = _run_laned(m, s3, reports, host, tmp_path, now + HOUR)
    assert again["built"] == [] and len(s3.puts) == n            # idempotent
    assert not (tmp_path / "host" / "9603900").exists()          # host stage cleaned


def test_box_reverifies_the_fetched_file(tmp_path):
    """A host file reaching past the cut is refused on the box."""
    m = evals()
    p = tmp_path / "w.npz"
    with zipfile.ZipFile(p, "w") as zf:
        for name, arr in (("target_ts", np.array([5, 9], np.int64)),
                          ("first_seen_ts", np.array([6, 11], np.int64)),
                          ("origin_ts", np.array([5], np.int64))):
            with zf.open(name + ".npy", "w") as f:
                np.lib.format.write_array(f, arr)
    assert m.verify_npz(p, cut=11, now=20)["max_first_seen_ts"] == 11
    with pytest.raises(m.EmbargoViolation):
        m.verify_npz(p, cut=10, now=20)


def test_eval_host_runs_niced_bounded_and_streams_to_disk(tmp_path):
    m = evals()
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\nname = "laned-eval"\nhost = "1.2.3.4"\nport = 22\nuser = "root"\n'
                     'remote_python = "/r/.venv/bin/python"\nworkdir = "/r"\n')
    inputs = tmp_path / "in.toml"
    inputs.write_text(f'eval_hosts = "{hosts}"\neval_host_name = "laned-eval"\n'
                      'eval_pool_root = "/root/lane_d_pool_d/pool_d"\n')
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        if "stdout" in kw:
            kw["stdout"].write(b"NPZ")
            return SimpleNamespace(returncode=0, stdout=None, stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b'{"ok": true}\n', stderr=b"")

    h = m.EvalHost(inputs, run=run)
    assert h.build({"boundary": 7}) == {"ok": True}
    cmd = calls[0][0][-1]
    for part in ("timeout", "nice -n 19", "ionice -c 3", "cascade.eval.settlement_eval_build",
                 "--pool-root /root/lane_d_pool_d/pool_d", f"{m.HOST_STAGE}/7"):
        assert part in cmd
    h.fetch(7, tmp_path / "got.npz")
    assert (tmp_path / "got.npz").read_bytes() == b"NPZ"


# ── lane C / MarketCast ─────────────────────────────────────────────────────


def _mc():
    from cascade.shared.config.marketcast import MarketCastConfig

    return MarketCastConfig()


def _entry(train_time, **mc):
    return {"entry": {"marketcast": {"train_time": train_time, **mc}}}


def test_mkt_draw_reuses_bars_for_the_embargo_and_the_cut_retires_them():
    """A bar judged for a leg dispatched at d is inside the backtest span of a
    leg dispatched a day later (re-use: immediate publication would leak);
    no bar at or before cut_c is inside ANY later draw's spans."""
    from cascade.pool.marketcast import embargo_draw

    m = evals()
    mc = _mc()
    bh = "0x" + "11" * 32
    d = float(T0 + 40 * DAY)
    first = embargo_draw(mc, train_time=d, phase="judge", block=1, block_hash=bh,
                         after=d + 3600)
    nxt = embargo_draw(mc, train_time=d + DAY, phase="judge", block=2, block_hash=bh,
                       after=d + DAY + 3600)
    bar = first["hi"] - 3 * 3600                            # a backtest bar of the first leg
    assert nxt["lo"] < bar <= nxt["hi"]
    fwd_bar = first["forward_after"] + 600                  # a forward bar of the first leg
    assert nxt["lo"] < fwd_bar <= nxt["hi"]
    now = int(d + 2 * DAY)
    st = {"mkt_backlog": [{"candidates": [_entry(d + DAY, embargo_days=30,
                                                  embargo_eval_days=28)]}], "pending": []}
    t_c, cut, e_c = m.mkt_cut(st, now=now, embargo_days=[30, 28])
    assert t_c == int(d + DAY) - 72 * HOUR and e_c == 30 and cut == t_c - 30 * DAY - 1
    for t in range(t_c, t_c + 60 * DAY, 3 * HOUR):
        dr = embargo_draw(mc, train_time=float(t), phase="judge", block=3, block_hash=bh,
                          after=float(t) + 3600)
        assert dr["lo"] > cut and dr["forward_after"] > cut
    # a confirm draw (same dispatch, other half) is covered too
    assert embargo_draw(mc, train_time=float(t_c), phase="confirm", block=4, block_hash=bh,
                        after=float(t_c) + 3600)["lo"] > cut


def test_mkt_cut_fails_closed_and_follows_recorded_embargo():
    m = evals()
    now = T0 + 90 * DAY
    assert m.mkt_cut({"mkt_backlog": [{"candidates": [{"entry": {}}]}]}, now=now,
                     embargo_days=[30])[:2] == (None, None)
    assert m.mkt_cut({"pending": [{"target": "mkt:returns", "entry": {"marketcast": {}}}]},
                     now=now, embargo_days=[30])[:2] == (None, None)
    # a non-MarketCast pending entry does not protect; a recorded longer embargo does
    t_c, cut, e_c = m.mkt_cut({"pending": [{"target": "lane:x", "entry": {}}],
                               "comp": {"x": {"open": [{"marketcast": {
                                   "train_time": now - DAY, "embargo_days": 45}}]}}},
                              now=now, embargo_days=[30, 28], slack_hours=0)
    assert t_c == now - DAY and e_c == 45 and cut == now - DAY - 45 * DAY - 1


def _mkt_report(end):
    return {"round_id": "123", "created_block": 1,
            "snapshot": {"key": "pool_mkt/snapshots/block-1.tar", "sha256": "aa" * 32,
                         "digest": "bb" * 32, "t0": end - 30 * DAY, "end": end},
            "entries": [{"hotkey": "5H", "target": "mkt:market", "phase": "judge",
                         "trained_pointer": "p", "after": end - 5 * HOUR,
                         "draw": {"hi": end - 6 * HOUR, "lo": end - 34 * DAY},
                         "windows": [{"wid": f"crypto:BTC|1mx60|1mx60|{end - 7 * HOUR}",
                                      "base": 1.0, "ft": 0.9},
                                     {"wid": f"crypto:BTC|1mx60|1mx60|{end - 2 * HOUR}",
                                      "base": 1.1, "ft": 1.0}]},
                        {"hotkey": "5G", "target": "mkt:market", "phase": "judge",
                         "status": "infra_deferred", "windows": []}]}


def test_publish_mkt_waits_for_the_embargo_then_publishes(tmp_path):
    m = evals()
    s3 = FakeS3()
    end = T0 + 10 * DAY
    s3.objects["marketcast/round-123.json"] = {
        "body": json.dumps(_mkt_report(end)).encode(),
        "LastModified": datetime.fromtimestamp(end + 600, UTC)}
    s3.objects["marketcast/public-bars/index.json"] = {"body": b"{}"}
    kw = dict(client=s3, bucket="b", base_url=BASE, lane_c_state={}, embargo_days=[30, 28],
              stage=tmp_path / "stage", log=lambda s: None)
    retire = end + 30 * DAY + 1 + 72 * HOUR
    assert m.publish_mkt(now=retire - 1, **kw)["built"] == []
    assert m.publish_mkt(now=retire, dry_run=True, **kw)["built"] == ["123"]
    assert not any(k.startswith("marketcast/evals/123/") for k in s3.objects)
    assert m.publish_mkt(now=retire, **kw)["built"] == ["123"]
    pre = "marketcast/evals/123/"
    assert [k for k in s3.puts if k.startswith(pre)] == [
        pre + "windows.json", pre + "README.md", pre + "manifest.json"]
    assert all(s3.objects[pre + n]["ACL"] == "public-read"
               for n in ("windows.json", "README.md", "manifest.json"))
    rows = json.loads(s3.objects[pre + "windows.json"]["body"])["windows"]
    assert [r["kind"] for r in rows] == ["backtest", "forward"]
    idx = json.loads(s3.objects["marketcast/evals/index.json"]["body"])
    assert [i["round_id"] for i in idx["items"]] == ["123"]
    assert m.publish_mkt(now=retire + HOUR, **kw)["built"] == []      # idempotent


# ── public-read ACLs ────────────────────────────────────────────────────────


def test_acl_backfill_public_reads_the_signed_objects_once(tmp_path):
    m = evals()
    s3 = FakeS3()
    for k in ("lane_b/round-1.json", "lane_c/round-2.json", "lane-d/reports/lane_d_report_9.json",
              "marketcast/round-3.json", "lane_b/other.json", "receipts/x.json"):
        s3.objects[k] = {"body": b"{}"}
    done = tmp_path / "acl_done.txt"
    dry = m.acl_backfill(s3, "b", done_file=done, dry_run=True, log=lambda s: None)
    assert dry["todo"] == ["lane-d/reports/lane_d_report_9.json", "lane_b/round-1.json",
                           "lane_c/round-2.json"] and s3.acls == []
    m.acl_backfill(s3, "b", done_file=done, log=lambda s: None)
    assert sorted(s3.acls) == sorted((k, "public-read") for k in dry["todo"])
    s3.objects["lane_c/round-4.json"] = {"body": b"{}"}
    again = m.acl_backfill(s3, "b", done_file=done, log=lambda s: None)
    assert again["applied"] == ["lane_c/round-4.json"]


class _Store:
    def __init__(self):
        self.calls = []

    def put_text(self, key, text, **kw):
        self.calls.append((key, kw))


def test_trainer_publishes_lane_b_and_lane_c_manifests_public_read():
    evals()                                       # the WP's module (assertion on main)
    from cascade.trainer.lane_b_driver import RunnerLaneBOps
    from cascade.trainer.lane_c_driver import RunnerLaneCOps

    store = _Store()
    runner = SimpleNamespace(wallet=None, manifest_store=lambda: store)
    b = object.__new__(RunnerLaneBOps)
    b.r = runner
    b.publish_expert_manifest(SimpleNamespace(round_id="7", to_json=lambda: "{}"))
    c = object.__new__(RunnerLaneCOps)
    c.r = runner
    c.publish("lane_c/round-8.json", "{}")
    assert [k for k, _ in store.calls] == ["lane_b/round-7.json", "lane_c/round-8.json"]
    assert all(kw.get("acl") == "public-read" for _, kw in store.calls)


def test_templates_ship_dry_run_and_docs_link_the_indexes():
    m = evals()
    repo = Path(__file__).resolve().parents[3]
    svc = (repo / "deploy/laned/systemd/cascade-settlement-evals.service").read_text()
    assert "cascade.pool.settlement_evals run" in svc and "--dry-run" in svc
    assert "MemoryMax=1G" in svc and "Nice=19" in svc
    assert (repo / "deploy/laned/systemd/cascade-settlement-evals.timer").is_file()
    assert f"{m.LANED_PREFIX}/index.json" in (repo / "docs/MINER_LANE_D.md").read_text()
    assert f"{m.MKT_PREFIX}/index.json" in (repo / "docs/MINER_LANE_C_MARKETCAST.md").read_text()
    assert f"{m.LANED_PREFIX}/index.json" in (repo / "cascade/website/index.html").read_text()
