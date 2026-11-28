"""Shared fixtures for the W-MKTOFF acceptance tests (README.md has the
contract).

A tiny MarketCast round (3 crypto assets, 1-minute bars, 64-step contexts)
judged by the validator's own ``_mkt_publish_round`` against a real
:class:`~cascade.validator.mkt_worker.Worker` serving the framed protocol in
a FORKED CHILD (:class:`ForkProc`: the "pod" is another process, so what the
box allocates or calls is the box alone). The ssh / scp of the session run
locally (:func:`local_run`). Fine-tunes are deterministic numpy forecasters
(:class:`Fc`) that read their covariates, so the covariate rows matter.
Reports are signed by :class:`DetSigner` so whole report bytes compare.

No chain, storage, pod, torch or real key. Nothing here imports a module in
PATHS at import time.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EB = 7200
BLOCK = 20 * EB
T0 = int(datetime(2026, 9, 1, tzinfo=UTC).timestamp())
ANCHOR_T = T0 + 200_000
ASSETS = ("BTC", "ETH", "SOL")
TARGET = "mkt:returns"


def mc():
    from cascade.shared.config.marketcast import parse_marketcast

    return parse_marketcast({
        "from_block": EB, "suites": ["returns"], "assets": {"crypto": list(ASSETS)},
        "frequencies": {"returns": ["1m", "5m"]}, "horizons": {"returns": ["1m", "5m"]},
        "eval_context_length": 64, "windows_per_cell": 12, "backtest_windows_per_cell": 12,
        "min_windows": 10, "min_clusters": 2, "start_margin_seconds": 60,
        "verdict_wait_blocks": 2 * EB, "max_infra_deferrals": 2, "min_time_blocks": 2,
        "min_dollar_volume_24h": 0.0, "forward_horizons": ["1m", "5m"]})


def snapshot(root: Path, minutes: int = 5000):
    from cascade.pool import marketcast as P

    rng = np.random.default_rng(7)
    bars = {}
    for a in ASSETS:
        c = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, minutes)))
        bars[("crypto", a)] = {"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c,
                               "volume": rng.uniform(1.0, 3.0, minutes)}
    P.write_snapshot(root, t0=T0, minutes=minutes, bars=bars, sources={"crypto": "hyperliquid"},
                     anchor_block=BLOCK, anchor_time=float(ANCHOR_T))
    return P.load_snapshot(root, verify_signature=False)


def bhash(block: int) -> str:
    return "0x" + hashlib.sha256(f"block-{block}".encode()).hexdigest()


def ptr(tag: str) -> str:
    from cascade.shared.manifest import format_trained_pointer

    return format_trained_pointer(f"cascade/ckpt-{tag}@sha256:"
                                  + hashlib.sha256(tag.encode()).hexdigest())


def entry(hk: str, uid: int, *, train_block: int = BLOCK - 100, target: str = TARGET):
    from cascade.shared.lane_c_manifest import LaneCEntry

    m = mc()
    return LaneCEntry(miner_hotkey=hk, miner_uid=uid, target=target,
                      recipe_ref=f"{hk}/recipe@sha256:" + "a" * 64, trained_pointer=ptr(hk),
                      train_block=train_block, funded_by="miner",
                      marketcast={"base_ref": m.base_ref, "base_sha256": m.base_sha256,
                                  "checkpoint_sha256": hashlib.sha256(hk.encode()).hexdigest(),
                                  "train_time": float(ANCHOR_T + (train_block - BLOCK) * 12),
                                  "embargo_scan": {"rule": "embargo-v1",
                                                   "result": "no_generator"}})


class Fc:
    """Deterministic quantiles from the context's level / scale, nudged by the
    covariate rows (so a wrong covariate row changes the bytes)."""

    def __init__(self, factor: float, fail: Exception | None = None):
        from cascade.shared.config.marketcast import TIMESTRAL_KNOTS

        self.quantile_levels = tuple(TIMESTRAL_KNOTS)
        self.factor, self.fail = float(factor), fail

    def forecast_batch(self, ctx, horizon, *, past_covariates=None, future_covariates=None):
        if self.fail is not None:
            raise self.fail
        from scipy.stats import norm

        c = np.asarray(ctx, dtype=np.float64)
        fin = np.isfinite(c)
        z = np.where(fin, c, 0.0)
        n = np.maximum(fin.sum(axis=1), 1)
        mu = z.sum(axis=1) / n
        sd = np.sqrt(np.where(fin, (c - mu[:, None]) ** 2, 0.0).sum(axis=1) / n) + 1e-6
        bump = np.zeros(c.shape[0])
        for cov in (past_covariates, future_covariates):
            if cov is not None:
                v = np.asarray(cov, dtype=np.float64)
                ok = np.isfinite(v)
                bump += np.where(ok, v, 0.0).sum(axis=(1, 2)) / np.maximum(ok.sum(axis=(1, 2)), 1)
        zk = norm.ppf(np.asarray(self.quantile_levels))
        q = mu[:, None] + 1e-3 * bump[:, None] + self.factor * sd[:, None] * zk[None, :]
        return np.repeat(q[:, None, :], int(horizon), axis=1)


class DetSigner:
    """A deterministic stand-in wallet (the report body's sha512)."""

    def sign(self, body: bytes) -> bytes:
        return hashlib.sha512(bytes(body)).digest()


class Store:
    def __init__(self):
        self.objs: dict[str, object] = {}
        self.puts: list[str] = []

    def get_text(self, key):
        from cascade.shared.hippius import ObjectNotFound

        if key not in self.objs:
            raise ObjectNotFound(key)
        v = self.objs[key]
        return v if isinstance(v, str) else v.decode()

    def put_text(self, key, text, content_type="", acl=None):
        self.objs[key] = text
        self.puts.append(key)

    def put_bytes(self, key, data, content_type="", acl=None):
        self.objs[key] = bytes(data)
        self.puts.append(key)


class ForkProc:
    """A Popen stand-in: the worker serves the frames in a forked child."""

    def __init__(self, worker):
        a_r, a_w = os.pipe()          # session -> worker
        b_r, b_w = os.pipe()          # worker -> session
        pid = os.fork()
        if pid == 0:                  # the "pod"
            os.close(a_w)
            os.close(b_r)
            code = 0
            try:
                worker.serve(os.fdopen(a_r, "rb"), os.fdopen(b_w, "wb"))
            except BaseException:  # noqa: BLE001
                code = 1
            os._exit(code)
        os.close(a_r)
        os.close(b_w)
        self.pid = pid
        self.stdin, self.stdout = os.fdopen(a_w, "wb"), os.fdopen(b_r, "rb")

    def wait(self, timeout=None):
        os.waitpid(self.pid, 0)
        return 0

    def kill(self):
        with contextlib.suppress(ProcessLookupError):
            os.kill(self.pid, signal.SIGKILL)


def local_run(argv, timeout):
    """ssh / scp of the session, run on this machine."""
    if argv[0] == "scp":
        src, dest = argv[-2], argv[-1].split(":", 1)[1]
        shutil.copytree(src, dest)
        return subprocess.CompletedProcess(argv, 0, "", "")
    return subprocess.run(["bash", "-c", argv[-1]], capture_output=True, text=True,
                          timeout=timeout)


# the round: alice (four covariates) and carol (none) on the target, erin the
# slot holder (near-duplicate reference + holder pairing), dave a load fault,
# frank a forward that raises (F7), gina a failed load (infra deferral)
FACTORS = {"base": 3.0, "alice": 1.0, "carol": 1.0001, "erin": 1.5, "frank": 1.0}
COVS = {"alice": ("calendar", "session", "cross_asset", "volume")}
UIDS = {"alice": 7, "carol": 9, "dave": 10, "erin": 11, "frank": 12, "gina": 13}


def models():
    out = {"base": Fc(FACTORS["base"])}
    for hk in ("alice", "carol", "erin"):
        out[ptr(hk)] = Fc(FACTORS[hk])
    out[ptr("frank")] = Fc(1.0, fail=KeyError("boom"))
    return out


def item():
    from cascade.validator.marketcast import _candidate

    cands = []
    for i, hk in enumerate(("alice", "carol", "dave", "frank", "gina")):
        cands.append(_candidate(entry(hk, UIDS[hk]), "judge", None,
                                commit_block=BLOCK - 500 + 10 * i))
    return {"round_id": "7001", "block": BLOCK, "created_block": BLOCK + 10,
            "candidates": cands}


def holders():
    return {TARGET: ("erin", ptr("erin"), BLOCK - 900)}


def rig(tmp_path: Path, snap, *, on_pod: bool = True, worker_models=None):
    """``(validator, store, session)``: a main validator mid publish run, its
    live session on a forked worker."""
    from cascade.trainer.remote import RemoteHost
    from cascade.validator.marketcast import MarketCastMixin, MktFault
    from cascade.validator.mkt_offload import MktSession, RemoteForecaster, cache_root_of
    from cascade.validator.mkt_worker import Worker

    host = RemoteHost(name="pod", host="localhost", workdir=str(tmp_path / "pod"))
    root = Path(cache_root_of(host))
    root.mkdir(parents=True, exist_ok=True)
    w = Worker("cpu", root)
    w.models.update(worker_models if worker_models is not None else models())
    sess = MktSession(host, run=local_run, popen=lambda *a, **k: ForkProc(w))
    sess.start()
    levels = tuple(float(x) for x in Fc(1.0).quantile_levels)

    class V(MarketCastMixin):
        pass

    v = V()
    v.cfg = SimpleNamespace(marketcast=mc())
    store = Store()
    v._mkt_store_obj = store
    v.mkt_eval_hosts = tmp_path / "mkt_hosts.toml"
    v.__dict__["_mkt_sess"] = sess
    v.mkt_snapshot_fn = lambda block: (snap, f"pool_mkt/snapshots/block-{block}.tar", "e" * 64)
    v.mkt_block_hash_fn = bhash
    v.bench_signer = DetSigner()
    v.mkt_judge_on_pod = on_pod

    def fc_fn(key):
        if key == "base":
            return RemoteForecaster(sess, "base", levels), ()
        if key == ptr("dave"):
            raise MktFault("fine-tune unloadable: not the base architecture")
        if key == ptr("gina"):
            raise RuntimeError("registry 503 fetching the checkpoint")
        hk = next(h for h in UIDS if ptr(h) == key)
        return RemoteForecaster(sess, key, levels), COVS.get(hk, ())

    v.mkt_forecaster_fn = fc_fn
    return v, store, sess
