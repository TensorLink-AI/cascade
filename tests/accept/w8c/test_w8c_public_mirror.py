"""W8c (owner 2026-11-24, issue #894): the eval host's public half reads the
Hippius mirror. Without ``eval_public_root`` the box's ``public`` request
names no local mirror and the host reads ONLY the drawn series through
:class:`cascade.eval.lane_d_public_mirror.PublicMirror` (volume cache, then
Hippius, then R2); with it, the local mirror path is unchanged. Contract:
README.md "Public half (rev6.18 F4)" / "Hippius mirror (#894)".
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from . import helpers as h


def test_public_scores_entry_reads_the_drawn_series_through_the_mirror(tmp_path, monkeypatch):
    fc = h.forecasts_mod()
    from cascade.validator import lane_d_public

    rng = np.random.default_rng(3)
    wins = [{"window_id": f"gift_eval_pretrain:s{i}:200:h8", "series_id": f"s{i}",
             "source": "gift_eval_pretrain", "family": "fam", "freq": "H",
             "seasonal_period": 24, "history": rng.random(300) + 1.0,
             "target": rng.random(8) + 1.0, "channels": None} for i in range(2)]
    seen = {}

    class Mirror:
        def reader(self, draw):
            seen["draw"] = draw
            return tmp_path / "ix", (lambda row: np.ones((1, 10)))

    def fake_read(draw, *, index_dir, block_hash, max_channels, data_root=None, reader=None):
        seen.update(index_dir=index_dir, data_root=data_root, reader=reader)
        return list(wins), []

    monkeypatch.setattr(lane_d_public, "read_public_windows", fake_read)
    q = np.sort(rng.random((2, 8, 9)) + 1.0, axis=-1)

    class Cache:
        def get(self, digest):
            return tmp_path

    got = fc.public_scores_entry(
        Cache(), "d" * 64, [{"source": "gift_eval_pretrain", "windows": []}], public_root=None,
        context=128, horizon=8, contexts=(128,), block_hash=None, max_channels=32,
        forecast=lambda *a, **k: q, mirror=Mirror())
    assert seen["draw"]["source"] == "gift_eval_pretrain"
    assert seen["index_dir"] == tmp_path / "ix" and seen["data_root"] is None
    assert callable(seen["reader"]), "the mirror's reader feeds read_public_windows"
    assert [r["window_id"] for r in got["scores"]["gift_eval_pretrain"]] == [
        w["window_id"] for w in wins]


def test_public_scores_entry_refuses_without_any_mirror(tmp_path):
    fc = h.forecasts_mod()
    import pytest

    with pytest.raises(fc.LaneDForecastError, match="no public mirror"):
        fc.public_scores_entry(object(), "d" * 64, [{"source": "rmisc", "windows": []}],
                               public_root=None, context=8, horizon=8, contexts=(8,),
                               block_hash=None, max_channels=1, mirror=None)


def _ssh_host(tmp_path, **kw):
    m = h.inputs_mod()
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\nname = "laned-eval"\nhost = "10.0.0.9"\nstage = "final"\n',
                     encoding="utf-8")
    from cascade.eval.lane_d_forecasts import PUBLIC_REPLY_SCHEMA

    argvs = []

    def run(argv, *, input, capture_output, timeout, check):
        argvs.append(argv[-1])
        if "hello" in argv[-1].split():
            out = json.dumps(h.hello())
        else:
            out = json.dumps({"schema": PUBLIC_REPLY_SCHEMA, "ok": True, "scores": {},
                              "dropped": {}})
        return SimpleNamespace(returncode=0, stdout=out.encode(), stderr=b"")

    host = m.SSHEvalHost(hosts, cache_dir="/c", budget_bytes=10, contexts=h.CONTEXTS, run=run,
                         **kw)
    host.hello()
    host.public_scores("d" * 64, context=128, horizon=8, draws=[], block_hash="0x00",
                       max_channels=4)
    return argvs[-1]


def test_ssh_eval_host_uses_the_hippius_mirror_without_a_public_root(tmp_path):
    cmd = _ssh_host(tmp_path)
    assert " public " in f" {cmd} " and "--public-root" not in cmd
    cmd = _ssh_host(tmp_path, public_cache_dir="/workspace/lane_d_public_cache",
                    public_cache_budget_bytes=123)
    assert "--public-cache-dir /workspace/lane_d_public_cache" in cmd
    assert "--public-cache-budget-bytes 123" in cmd
    cmd = _ssh_host(tmp_path, public_root="/root/lane_d_public")
    assert "--public-root /root/lane_d_public" in cmd, "a local mirror still works"


def test_inputs_file_accepts_the_public_cache_keys():
    m = h.inputs_mod()
    assert {"eval_public_cache_dir", "eval_public_cache_budget_bytes"} <= set(m.INPUTS_FILE_KEYS)
    assert "eval_public_root" not in m._REQUIRED_KEYS


def test_the_host_cli_defaults_to_the_release_mirror_config():
    fc = h.forecasts_mod()
    args = fc._build_parser().parse_args(["public", "--cache-dir", "/c", "--budget-bytes", "1"])
    assert args.public_root is None and args.mirror_config is None
    from cascade.eval.lane_d_public_mirror import MIRROR_CONFIG_PATH, load_mirror_config

    cfg = load_mirror_config()
    assert MIRROR_CONFIG_PATH.is_file()
    assert [s.name for s in cfg["stores"]] == ["hippius", "r2"]
