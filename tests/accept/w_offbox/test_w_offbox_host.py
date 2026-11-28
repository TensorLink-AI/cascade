"""W-OFFBOX: the box's SSH settle request (bounded on the host, serialised,
the grants added), the production inputs file, the eval host's own checks
(each pin re-hashed against the host's copy) and the box's binding of the
reply to its pins. Contract: README.md "The eval host" and "The box"."""

from __future__ import annotations

import json
import shlex
from types import SimpleNamespace

import pytest

from . import helpers as h


def _hosts(tmp_path):
    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\nname = "laned-eval"\nhost = "10.0.0.9"\nstage = "final"\n'
                     'remote_python = "python"\nworkdir = "."\n', encoding="utf-8")
    return hosts


def test_ssh_settle_is_bounded_serialised_and_carries_the_grants(tmp_path):
    h.require_offbox()
    from cascade.validator.lane_d_inputs import SETTLE_REPLY_SCHEMA

    m = h.inputs_mod()
    seen = {}

    def run(argv, *, input, capture_output, timeout, check):
        if "hello" in argv[-1].split():
            out = json.dumps(h.w8c.hello())
        else:
            seen.update(cmd=argv[-1], payload=json.loads(input), timeout=timeout)
            out = "Downloading model (chatter)\n" + json.dumps(
                {"schema": SETTLE_REPLY_SCHEMA, "ok": False, "code": "pool_unavailable"})
        return SimpleNamespace(returncode=0, stdout=out.encode(), stderr=b"")

    grants = {"a" * 64: {"digest": "a" * 64, "files": {}}}
    host = m.SSHEvalHost(_hosts(tmp_path), cache_dir="/c", budget_bytes=10,
                         contexts=h.w8c.CONTEXTS, run=run, timeout_s=600.0,
                         grants=grants.get, pool_root="/root/lane_d_pool_d",
                         score_cache_dir="/root/lane_d_score_cache")
    host.hello()
    reply = host.settle({"need": {"a" * 64: 4096, "b" * 64: 8192}, "x": 1})
    assert reply["code"] == "pool_unavailable", "the reply is the last stdout line"
    words = shlex.split(seen["cmd"].split("&&")[-1])
    assert words[:4] == ["timeout", "-k", "30", str(2 * 600 - 30)], \
        "bounded ON the host at eval_timeout_seconds per needed digest"
    assert words[4] == "flock", "serialised with forecast / public on the cache lock"
    i = words.index("-m")
    assert words[i + 1:i + 3] == ["cascade.eval.lane_d_forecasts", "settle"]
    assert words[words.index("--pool-root") + 1] == "/root/lane_d_pool_d"
    assert words[words.index("--score-cache-dir") + 1] == "/root/lane_d_score_cache"
    assert seen["timeout"] == 1200.0
    assert seen["payload"]["grants"] == {"a" * 64: grants["a" * 64], "b" * 64: None}
    assert seen["payload"]["x"] == 1


def test_eval_host_cli_has_a_settle_command():
    h.require_offbox()
    from cascade.eval import lane_d_forecasts as f

    assert f.HOST_MODULES.get("settle") == f.HOST_MODULES["forecast"]
    args = f._build_parser().parse_args(["settle", "--cache-dir", "/c", "--budget-bytes", "1",
                                         "--pool-root", "/p"])
    assert args.cmd == "settle" and str(args.pool_root) == "/p"


def _inputs_toml(tmp_path, **extra):
    keys = {"intake_state_dir": "/i", "pool_pins": "/p.json", "house_pointer": "/h.json",
            "house_signer_ss58": "5x", "eval_hosts": "/e.toml", "eval_cache_dir": "/c",
            "readiness": "/r.json", "t7": "/t.json", **extra}
    lines = [f'{k} = "{v}"' for k, v in keys.items()] + ["eval_cache_budget_bytes = 10"]
    p = tmp_path / "inputs.toml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_inputs_file_eval_pool_root_settles_on_the_host(tmp_path):
    h.require_offbox()
    m = h.inputs_mod()
    for k in ("eval_pool_root", "eval_score_cache_dir", "box_pool_budget_bytes"):
        assert k in m.INPUTS_FILE_KEYS and k not in m._REQUIRED_KEYS, k
    cfg = h.w8b.chain_cfg(tmp_path / "v")
    on = m.load_inputs_services(cfg, _inputs_toml(tmp_path, eval_pool_root="/root/pd"))
    assert on["settle_on_host"] is True and on["eval_host"].pool_root == "/root/pd"
    assert on["box_pool_budget_bytes"] == m.BOX_POOL_BUDGET_BYTES
    off = m.load_inputs_services(cfg, _inputs_toml(tmp_path))
    assert off["settle_on_host"] is False, "no eval_pool_root: the budget-guarded path"


def _settle_reply(tmp_path, mutate_root=None, **over):
    """One offloaded B1 settle request answered by the host fixture."""
    fx = h.w8b.fixture(tmp_path / "fx")
    rig = h.w8c.Rig(tmp_path / "rig", fx=fx)
    root = h.copy_pins(fx, tmp_path / "host_pool")
    if mutate_root is not None:
        mutate_root(root)
    host = h.offloaded(rig, root)
    inp = rig.source(h.B1)
    assert isinstance(inp.forecasts, h.inputs_mod().RemoteForecasts)
    assert all(v is None for v in inp.forecasts.values()), "no forecast reaches the box"
    assert host.hellos == 1 and not host.requests, "nothing is judged before the settlement"
    return rig, inp, host


def test_host_refuses_a_pin_its_copy_does_not_hash_to(tmp_path):
    def corrupt(root):
        meta = root / "pool_d_1" / "metadata.json"
        meta.write_text(meta.read_text() + " ", encoding="utf-8")

    rig, _, _ = _settle_reply(tmp_path, corrupt)
    with pytest.raises(Exception) as ei:
        h.step(rig)(h.B1)
    h.w8c.assert_inputs_error(ei.value.__cause__, "pool_digest_mismatch")
    assert not rig.state_path.exists(), "nothing is written on a refusal"


def test_host_refuses_a_pin_it_does_not_hold(tmp_path):
    import shutil

    rig, _, _ = _settle_reply(tmp_path, lambda root: shutil.rmtree(root / "pool_d_1"))
    with pytest.raises(Exception) as ei:
        h.step(rig)(h.B1)
    h.w8c.assert_inputs_error(ei.value.__cause__, "pool_unavailable")


@pytest.mark.parametrize("lie", ["snapshots", "seed", "pool_rec"])
def test_box_refuses_a_reply_not_bound_to_its_pins(tmp_path, lie):
    rig, _, host = _settle_reply(tmp_path)
    real = host.settle

    def lying(request):
        reply = real(request)
        if lie == "snapshots":
            reply["snapshots"] = ["0" * 64]
        elif lie == "seed":
            reply["judged"]["seed"] = "lane-d/1:0:" + "0" * 64
        else:
            reply["judged"]["pool_rec"]["snapshots"][0]["sha256"] = "0" * 64
        return reply

    host.settle = lying
    with pytest.raises(Exception) as ei:
        h.step(rig)(h.B1)
    h.w8c.assert_inputs_error(ei.value.__cause__, "pool_digest_mismatch")
    assert not rig.state_path.exists()
