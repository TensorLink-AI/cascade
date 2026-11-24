"""R13 (owner 2026-11-24, issue #894): the lane D eval host's public-mirror
warm cache is a Lium persistent volume. ``[eval_host] public_cache_volumes``
names the volume HUIDs; host i of a cohort is rented with ``lium up
--volume id:<HUID i>`` and the volume is mounted at ``public_cache_mount``
(where ``deploy/laned/public_mirror.toml``'s ``cache_dir`` lives). A host
past the list is alerted and fetches from Hippius. Contract: README.md.
"""

from __future__ import annotations

import json
import types

import pytest

from .helpers import FakeClock, FakeLium, config, mod

H200 = [("H200", {"id": f"ex-{i}", "price_per_hour": 3.0 + i}) for i in range(5)]


def _has_volumes(m):
    assert "public_cache_volumes" in m.EvalConfig.__dataclass_fields__, (
        "#894: [eval_host] public_cache_volumes (the Lium volume per eval host) is missing")


def test_each_host_is_rented_with_its_volume(tmp_path):
    m = mod()
    _has_volumes(m)
    cfg = config(tmp_path, public_cache_volumes=("brave-fox-3a", "calm-owl-7b"))
    lium = FakeLium(executors=H200)
    out = m.eval_up(cfg, lium, entries=5, cohort="b200", dry_run=False, clock=FakeClock())
    assert out["hosts"] == 3
    assert [s.volumes for _n, _ex, s in lium.launched] == [("brave-fox-3a",), ("calm-owl-7b",),
                                                            ()]
    assert [i["volume"] for i in out["instances"]] == ["brave-fox-3a", "calm-owl-7b", None]
    alerts = (tmp_path / "state").rglob("alerts*")
    text = "".join(p.read_text() for p in alerts if p.is_file())
    assert "lane_d_eval_no_volume" in text, "a host without a volume is alerted"


def test_dry_run_plans_the_volumes(tmp_path):
    m = mod()
    _has_volumes(m)
    cfg = config(tmp_path, public_cache_volumes=("brave-fox-3a",))
    out = m.eval_up(cfg, FakeLium(executors=H200), entries=1, cohort="b1", dry_run=True,
                    clock=FakeClock())
    assert out["volumes"] == ["brave-fox-3a"] and out["public_cache_mount"] == "/workspace"


@pytest.mark.parametrize("bad", [["a b"], ["x", "x"], "brave-fox", [""]])
def test_bad_volume_lists_are_refused(tmp_path, bad):
    m = mod()
    _has_volumes(m)
    with pytest.raises(m.ElasticError):
        m._eval_from({"public_cache_volumes": bad})


def test_the_shipped_config_names_the_volume_mount_of_the_mirror_cache():
    import tomllib

    m = mod()
    _has_volumes(m)
    from .helpers import CONFIG, REPO

    ev = m.load_config(CONFIG).eval_host
    assert ev.public_cache_mount == "/workspace"
    mirror = REPO / "deploy" / "laned" / "public_mirror.toml"
    assert mirror.is_file(), "#894: deploy/laned/public_mirror.toml is missing"
    cache_dir = tomllib.loads(mirror.read_text())["cache_dir"]
    assert cache_dir.startswith(ev.public_cache_mount + "/")


def test_lium_up_carries_the_volume(monkeypatch, tmp_path):
    from cascade.provision import core

    assert "volumes" in core.LaunchSpec.__dataclass_fields__, (
        "#894: LaunchSpec.volumes (lium up --volume) is missing")
    monkeypatch.setenv(core.LIUM_COOLDOWN_FILE_ENV, str(tmp_path / "cooldown.json"))
    monkeypatch.setenv(core.LIUM_QUARANTINE_FILE_ENV, str(tmp_path / "quarantine.json"))
    spawned = []

    def _run(argv):
        out = json.dumps([{"id": "exec-1", "gpu_count": 1}]) if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = core.LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: spawned.append(argv),
                             _template=lambda p: "tmpl")
    prov.launch(core.LaunchSpec(sku="H200", count=1, image="img@sha256:" + "0" * 64,
                                ssh_pubkey="k", name_prefix="laned-eval-x",
                                volumes=("brave-fox-3a",)))
    argv = spawned[-1]
    assert argv[argv.index("--volume") + 1] == "id:brave-fox-3a"
    prov.launch(core.LaunchSpec(sku="H200", count=1, image="img@sha256:" + "0" * 64,
                                ssh_pubkey="k", name_prefix="laned-eval-y"))
    assert "--volume" not in spawned[-1], "no volume configured: none attached"
