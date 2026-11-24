"""Shared fakes for the R13 tests (README.md has the contract).

No network, no torch, nothing rented: a fake Shadeform HTTP surface (the
``_get`` / ``_post`` / ``wait_ready`` / ``get_ip`` / ``terminate`` /
``pod_exists`` subset of ``cascade.provision.core.ShadeformProvider``) and a
fake Lium adapter (the ``LiumProvider`` subset the controller uses). Nothing
here imports ``cascade.laned.elastic``: each test imports it first, so on a
base without it every test fails at that import.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "deploy" / "laned" / "elastic.toml"
T0 = 1_795_132_800 + 5 * 60          # 2026-11-20 00:05 UTC, just after a 3-h boundary


def mod():
    return importlib.import_module("cascade.laned.elastic")


def shade_type(cloud, type_id, *, vcpus, mem, disk, cents, gpus=0, available=True):
    return {"cloud": cloud, "shade_instance_type": type_id, "hourly_price": cents,
            "configuration": {"vcpus": vcpus, "memory_in_gb": mem, "storage_in_gb": disk,
                              "num_gpus": gpus, "gpu_type": "H100" if gpus else ""},
            "availability": [{"region": "us-east", "available": available,
                              "rental_type": "on_demand"}]}


# The 2026-11-24 00:4x UTC snapshot from the owner brief, plus a CHEAPER GPU
# type that must never be offered (CPU only) and an unavailable type.
MARKET = {"instance_types": [
    shade_type("massedcompute", "cpu_medium", vcpus=16, mem=80, disk=1600, cents=44),
    shade_type("verda", "cpu_16", vcpus=16, mem=64, disk=1000, cents=47),
    shade_type("massedcompute", "cpu_small", vcpus=8, mem=40, disk=800, cents=22),
    shade_type("massedcompute", "gpu_cheap", vcpus=32, mem=128, disk=2000, cents=10, gpus=1),
    shade_type("verda", "cpu_gone", vcpus=32, mem=128, disk=2000, cents=5, available=False),
]}


@dataclass
class Addr:
    ip: str
    ssh_port: int


@dataclass
class FakeShadeform:
    market: dict = field(default_factory=lambda: MARKET)
    ssh_key_id: str = "key-123"
    ready: bool = True
    delete_works: bool = True
    posts: list = field(default_factory=list)
    live: set = field(default_factory=set)
    deleted: list = field(default_factory=list)

    def _get(self, path, params=None):
        assert path == "/instances/types", path
        return self.market

    def _post(self, path, body=None):
        self.posts.append((path, body))
        if path == "/instances/create":
            iid = f"sf-{len(self.posts)}"
            self.live.add(iid)
            return {"id": iid}
        raise AssertionError(path)

    def wait_ready(self, iid, *, timeout):
        return self.ready

    def get_ip(self, iid):
        return Addr("203.0.113.7", 22)

    def terminate(self, iid):
        self.deleted.append(iid)
        if self.delete_works:
            self.live.discard(iid)

    def pod_exists(self, iid):
        return iid in self.live


@dataclass
class FakeLium:
    """``list_offers`` → ``[(sku, executor)]``; ``launch`` returns the pod name."""

    executors: list = field(default_factory=list)   # [(sku, {"id", "price_per_hour"})]
    ready: bool = True
    launched: list = field(default_factory=list)
    live: set = field(default_factory=set)
    faults: list = field(default_factory=list)
    calls: int = 0

    def list_offers(self, skus, *, gpus=1, exclude_ids=()):
        self.calls += 1
        return [(s, e) for s, e in self.executors
                if s in skus and str(e["id"]) not in exclude_ids]

    def launch(self, spec):
        offers = self.list_offers(spec.sku_choices or (spec.sku,), exclude_ids=spec.exclude_ids)
        assert offers, "launch with no offer"
        assert "@sha256:" in spec.image
        name = f"{spec.name_prefix}-0"
        self.launched.append((name, offers[0][1]["id"], spec))
        self.live.add(name)
        return [name]

    def price_of(self, pod):
        for name, ex_id, _spec in self.launched:
            if name == pod:
                return next(e["price_per_hour"] for _s, e in self.executors if e["id"] == ex_id)
        return None

    def wait_ready(self, pod, *, timeout):
        return self.ready

    def get_ip(self, pod):
        return Addr(f"198.51.100.{len(self.launched)}", 40000 + len(self.launched))

    def record_executor_fault(self, pod, reason):
        self.faults.append((pod, reason))
        return reason

    def terminate(self, pod):
        self.live.discard(pod)

    def pod_exists(self, pod):
        return pod in self.live


class FakeClock:
    def __init__(self, t=T0):
        self.t = float(t)
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def config(tmp_path, **eval_over):
    """The shipped config with state under tmp_path (and eval overrides)."""
    from dataclasses import replace

    m = mod()
    cfg = m.load_config(CONFIG)
    ev = replace(cfg.eval_host, image="ghcr.io/x/worker@sha256:" + "a" * 64,
                 ssh_pubkey_path=str(tmp_path / "id.pub"),
                 scoring_envs=str(tmp_path / "no_scoring_envs.toml"), **eval_over)
    (tmp_path / "id.pub").write_text("ssh-ed25519 AAAA test\n")
    return replace(cfg, eval_host=ev, state_dir=str(tmp_path / "state"))
