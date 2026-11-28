"""Shared fixtures for the W-OFFBOX acceptance tests (README.md has the
contract).

The fixture is W8c's (W7a's synthetic chain through W8b's step, numpy only).
The offloaded rig is the same validator with ``settle_on_host=True``: its eval
host is :class:`OffloadHost`, which answers ``hello`` like W8c's fake and
``settle`` by running the eval host's own ``serve_settle`` over a COPY of the
pinned snapshots (the host's ``pool_root``) with W8c's table forecaster,
through a JSON round trip both ways (the SSH transport). ``in_child`` runs it
in a forked child instead, so a memory measurement in this process sees the
box alone. Reports are signed by :class:`DetSigner` (the fixture key's ss58,
a deterministic 64-byte signature) so whole report bytes compare.

No chain, storage, pod, torch or real key. Nothing here imports a module in
PATHS at import time.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from pathlib import Path

from tests.accept.w8c import helpers as w8c

w8b = w8c.w8b
B1, B2, B3 = w8c.B1, w8c.B2, w8c.B3

INPUTS = "cascade.validator.lane_d_inputs"
SETTLE = "cascade.eval.lane_d_settle"
LANE_D = "cascade.validator.lane_d"
WORKER = "cascade.validator.offthread_worker"


def inputs_mod():
    return importlib.import_module(INPUTS)


def require_offbox() -> None:
    """The W-OFFBOX surface. On a build without it every test fails here, on
    an assertion (never an ImportError or a TypeError)."""
    import inspect

    m = inputs_mod()
    for name in ("HostJudge", "RemoteForecasts", "pinned_windows", "forecast_windows",
                 "BOX_POOL_BUDGET_BYTES"):
        assert hasattr(m, name), f"W-OFFBOX: lane_d_inputs.{name} is missing"
    p = inspect.signature(m.build_lane_d_inputs).parameters
    assert "settle_on_host" in p, "W-OFFBOX: build_lane_d_inputs(settle_on_host=)"
    assert "box_pool_budget_bytes" in p, "W-OFFBOX: build_lane_d_inputs(box_pool_budget_bytes=)"
    lane_d = importlib.import_module(LANE_D)
    assert "pool_phase" in inspect.signature(lane_d.settle_lane_d).parameters, \
        "W-OFFBOX: settle_lane_d(pool_phase=)"
    assert importlib.util.find_spec(SETTLE) is not None, \
        "W-OFFBOX: the eval host's cascade.eval.lane_d_settle is missing"


class DetSigner:
    """The fixture report signer's ss58 with a deterministic signature (sr25519
    signatures are randomised; this makes whole report bytes comparable)."""

    def __init__(self):
        self.ss58_address = w8b.signer().ss58_address

    def sign(self, body: bytes) -> bytes:
        return hashlib.sha512(b"w-offbox|" + bytes(body)).digest()


def copy_pins(fx: dict, root: Path) -> Path:
    """The eval host's copy of the fixture's snapshots: ``<root>/<name>/``."""
    root.mkdir(parents=True, exist_ok=True)
    for d in fx["chain"]["pools"].values():
        shutil.copytree(d, root / Path(d).name)
    return root


def copy_pin_as(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def in_child(fn, *args) -> bytes:
    """``fn(*args)`` (returning bytes) in a forked child; its result bytes."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                              # the child: never returns
        code = 0
        try:
            os.close(r)
            data = fn(*args)
            with os.fdopen(w, "wb") as f:
                f.write(data)
        except BaseException:                 # noqa: BLE001 — reported by the exit code
            code = 1
        os._exit(code)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        data = f.read()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0, "the eval host child failed"
    return data


class OffloadHost:
    """The lane D eval host as an offloaded box sees it: ``hello()`` (W8c's
    fake) and ``settle(request)`` (the host's ``serve_settle`` over
    ``pool_root`` with W8c's table forecaster, JSON both ways). ``forecast``
    is never called on the box."""

    def __init__(self, fake: w8c.EvalHost, pool_root: Path, *, fork: bool = False):
        self.fake = fake
        self.pool_root = Path(pool_root)
        self.fork = fork
        self.requests: list[dict] = []
        self.hellos = 0

    def hello(self):
        self.hellos += 1
        return self.fake.hello()

    def forecast(self, *a, **k):
        raise AssertionError("an offloaded box asked the eval host for raw forecasts")

    def _serve(self, raw: bytes) -> bytes:
        settle = importlib.import_module(SETTLE)
        reply = settle.serve_settle(raw, pool_root=self.pool_root, forecaster=self.fake)
        return json.dumps(reply, sort_keys=True).encode("utf-8")

    def settle(self, request: dict) -> dict:
        req = {**request, "grants": {dg: None for dg in request["need"]}}
        raw = json.dumps(req, sort_keys=True).encode("utf-8")
        self.requests.append(json.loads(raw))
        out = in_child(self._serve, raw) if self.fork else self._serve(raw)
        return json.loads(out.decode("utf-8"))


def offloaded(rig: w8c.Rig, pool_root: Path, *, fork: bool = False) -> OffloadHost:
    """Turn ``rig``'s source into the offloaded one (same fixture, same fakes)."""
    require_offbox()
    host = OffloadHost(rig.eval_host, pool_root, fork=fork)
    rig.reader.block_hash = lambda b: w8b.block_hash(int(b))
    rig.source = rig.build(eval_host=host, settle_on_host=True)
    return host


def step(rig: w8c.Rig, store=None):
    """W8b's production step over ``rig``'s source, signed by :class:`DetSigner`."""
    from cascade.validator.lane_d_wiring import build_lane_d_step

    return build_lane_d_step(rig.cfg, state_dir=rig.state_dir, signer=DetSigner(),
                             chain_factory=w8b.ReaderFactory(),
                             store=store if store is not None else w8b.Store(),
                             inputs=rig.source, bench=rig.source.bench)


def report_bytes(rig: w8c.Rig, boundary: int) -> bytes:
    from cascade.validator.lane_d_state import report_path

    return report_path(rig.state_path, boundary).read_bytes()
