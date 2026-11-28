"""MarketCast forecast worker (DEC-CA-0064): the GPU half of the MAIN
validator's MarketCast rows, on an operator-rented pod
(:mod:`cascade.validator.mkt_offload`).

One long-lived process per validator publish run, spoken to over the ssh
session's stdin/stdout: length-prefixed NPZ frames (numeric and string arrays
only, never pickle). It holds Timestral checkpoints the validator uploaded
(sha256-verified here before every load) and answers ``forecast_batch`` calls
— contexts and covariates in, quantiles out.

The judge (W-MKTOFF, 2026-11-28): the box once cut every window, built
every covariate row and scored every window in the validator process — for a
``mkt:market`` entry with ten covariates that is ~30k windows x 19 float64
rows x 4096 steps (~16 GB), and the 7 GB box OOM-killed the validator at
10:54. The ``judge_*`` ops run that phase HERE, on the pod, with the very
functions the in-process path calls (:func:`~cascade.validator.marketcast.
compute_entry_rows`, :func:`~cascade.validator.marketcast.near_dup_record`,
:func:`~cascade.validator.marketcast.holder_record`) on a snapshot the
validator uploaded (sha256 + digest + view verified here and on the box) and
the validator's own ``[marketcast]`` table (bound by sha256). Each reply is
the entry's rows (a few hundred bytes a window) and its quantile sidecar,
bound by sha256; the verdict, the report, its signature and the wallet stay
on the validator.

    python -m cascade.validator.mkt_worker --serve --device cuda
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np

SCHEMA = "cascade-mkt-forecast/1"
FRAME_MAX = 1 << 31                     # a frame larger than this is a broken peer
_LEN = struct.Struct(">Q")
# the ops this worker serves beyond ``load`` / ``forecast`` (the box refuses
# to judge in-process at scale when a pod's worker lacks them)
OPS = ("judge",)
# an error message crosses whole for the judge ops: the rows record it and
# red-team F7 compares the attempts' messages
JUDGE_MSG_MAX = 1 << 20
SNAPSHOTS_KEEP = 2


def encode_frame(header: dict, arrays: dict | None = None) -> bytes:
    """``header`` (JSON) + named numeric arrays as one length-prefixed NPZ."""
    buf = io.BytesIO()
    np.savez(buf, h=np.array(json.dumps(header, sort_keys=True)),
             **{f"a_{k}": np.asarray(v) for k, v in (arrays or {}).items() if v is not None})
    data = buf.getvalue()
    return _LEN.pack(len(data)) + data


def decode_frame(data: bytes) -> tuple[dict, dict]:
    # incident-ok: IR010 runs on the MarketCast GPU pod and decodes its own request
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        header = json.loads(str(z["h"][()]))
        arrays = {k[2:]: z[k] for k in z.files if k.startswith("a_")}
    if not isinstance(header, dict):
        raise ValueError("frame header is not an object")
    return header, arrays


def read_frame(stream) -> tuple[dict, dict] | None:
    """One frame from a blocking binary stream (None = clean EOF)."""
    head = stream.read(_LEN.size)
    if not head:
        return None
    if len(head) != _LEN.size:
        raise EOFError("truncated frame length")
    (n,) = _LEN.unpack(head)
    if not 0 < n <= FRAME_MAX:
        raise ValueError(f"frame length {n} out of range")
    data = stream.read(n)
    if len(data) != n:
        raise EOFError("truncated frame")
    return decode_frame(data)


def _load_timestral(d: Path, device: str):
    from ..eval.timestral_adapter import TimestralQuantileForecaster

    return TimestralQuantileForecaster.from_dir(d, device=device)


# model family -> loader(dir, device) returning a forecaster with
# ``quantile_levels`` and ``forecast_batch(ctx, horizon, past_covariates,
# future_covariates)``. Another lane-C competition with a different model
# registers its loader here and sends ``"family"`` with its ``load``.
LOADERS: dict = {"timestral": _load_timestral}


def _infra_like(e: BaseException) -> bool:
    from .marketcast import infra_like

    return infra_like(e)


# ── the judge's wire forms (box and pod) ─────────────────────────────────────


def _tuples(v):
    """JSON arrays back to the tuples a frozen config carries."""
    if isinstance(v, list):
        return tuple(_tuples(x) for x in v)
    if isinstance(v, dict):
        return {k: _tuples(x) for k, x in v.items()}
    return v


def mc_to_wire(mc) -> str:
    """A ``[marketcast]``-shaped frozen dataclass as JSON (class + fields)."""
    import dataclasses

    t = type(mc)
    return json.dumps({"cls": f"{t.__module__}:{t.__qualname__}",
                       "fields": {f.name: getattr(mc, f.name) for f in dataclasses.fields(mc)}},
                      sort_keys=True)


def mc_from_wire(text: str):
    import importlib

    o = json.loads(text)
    mod, _, name = str(o["cls"]).partition(":")
    if not mod.startswith("cascade.") or not name.isidentifier():
        raise ValueError(f"config class {o['cls']!r} is not a cascade config")
    cls = getattr(importlib.import_module(mod), name)
    return cls(**{k: _tuples(v) for k, v in dict(o["fields"]).items()})


def wire_sha256(text: str) -> str:
    import hashlib

    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def rows_to_wire(rows) -> str:
    """:class:`~cascade.shared.marketcast_report.MktEntryRows` as JSON,
    lossless (NaN / inf kept: the report's own canonical form drops them
    later, exactly as on the in-process path)."""
    from dataclasses import asdict

    return json.dumps(asdict(rows), sort_keys=True)


def rows_from_wire(text: str):
    from ..shared.marketcast_report import MktEntryRows, MktWindowRow

    d = json.loads(text)
    wins = tuple(MktWindowRow(**{k: (tuple(v) if isinstance(v, list) else v)
                                 for k, v in w.items()}) for w in d.pop("windows", ()))
    return MktEntryRows(**d, windows=wins)


def snapshot_meta_sha256(meta: dict) -> str:
    return wire_sha256(json.dumps(meta, sort_keys=True, default=str))


class MktRemoteError(RuntimeError):
    """An exception the judge raised on the other side, by its type name:
    ``text`` is what the in-process path records (``"<Type>: <message>"``)."""

    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = str(kind)
        self.text = f"{self.kind}: {msg}"


class PodForecaster:
    """A pod-local model seen exactly as the box saw it through
    :class:`~cascade.validator.mkt_offload.RemoteForecaster` (the judge runs
    the same rows code, so every error must surface as it did over the
    wire): float64 inputs, a ``ValueError`` kept, an infra-shaped failure an
    offload error, anything else a ``RuntimeError("<Type>: <msg>")``; the
    reply shape checked."""

    def __init__(self, fc, name: str = "pod"):
        self.fc, self.name = fc, str(name)
        self.quantile_levels = tuple(float(x) for x in fc.quantile_levels)

    def forecast_batch(self, contexts, horizon: int, *, past_covariates=None,
                       future_covariates=None) -> np.ndarray:
        from .mkt_offload import MktOffloadError

        c = np.asarray(contexts, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError(f"contexts must be (B, T); got {c.shape}")
        pc = None if past_covariates is None else np.asarray(past_covariates, np.float64)
        kc = None if future_covariates is None else np.asarray(future_covariates, np.float64)
        try:
            q = np.asarray(self.fc.forecast_batch(c, int(horizon), past_covariates=pc,
                                                  future_covariates=kc), dtype=np.float64)
        except Exception as e:  # noqa: BLE001 — classified as the wire did
            err, msg = type(e).__name__, str(e)[:2000]
            if _infra_like(e):
                raise MktOffloadError(f"{self.name}: {err}: {msg}") from e
            if err == "ValueError":
                raise ValueError(msg) from e
            raise RuntimeError(f"{err}: {msg}") from e
        if q.ndim != 3 or q.shape[:2] != (c.shape[0], int(horizon)) \
                or q.shape[2] != len(self.quantile_levels):
            raise MktOffloadError(f"forecast reply shape {q.shape} != ({c.shape[0]}, "
                                  f"{int(horizon)}, {len(self.quantile_levels)})")
        return q

    def forecast_quantiles(self, context, horizon: int, *, past_covariates=None,
                           future_covariates=None) -> np.ndarray:
        pc = None if past_covariates is None else np.asarray(past_covariates)[None]
        kc = None if future_covariates is None else np.asarray(future_covariates)[None]
        return self.forecast_batch(np.asarray(context, dtype=np.float64)[None], horizon,
                                   past_covariates=pc, future_covariates=kc)[0]


def _peak_rss_mb() -> int:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    except Exception:  # noqa: BLE001 — informational
        return -1


class Worker:
    def __init__(self, device: str, root: Path):
        self.device, self.root = str(device), Path(root).resolve()
        self.models: dict[str, object] = {}
        # the judge (one round at a time, as on the box)
        self.snaps: dict[str, object] = {}
        self.mc = None
        self.mc_sha = ""
        self.base_cache: dict = {}
        self.ft_cache: dict = {}
        self.datas: dict = {}

    def _dir(self, rel: str) -> Path:
        d = (self.root / str(rel)).resolve()
        if self.root not in d.parents:
            raise ValueError("directory escapes the worker root")
        return d

    def _pod_fc(self, model: str) -> PodForecaster:
        return PodForecaster(self.models[str(model)], "pod")

    def _load_ft(self, specs: dict):
        """The judge's ``load_ft``: the box loaded (fetched, checked,
        uploaded) every pointer it may need and says what came of it."""
        from .marketcast import MktFault

        def load_ft(pointer: str):
            sp = specs.get(str(pointer))
            if sp is None:
                raise MktRemoteError("KeyError", repr(str(pointer)))
            if "fault" in sp:
                raise MktFault(str(sp["fault"]))
            if "error" in sp:
                raise MktRemoteError(str(sp["error"]), str(sp.get("msg", "")))
            return self._pod_fc(sp["model"]), tuple(sp.get("covariates") or ())

        return load_ft

    def _judge(self, op: str, header: dict, arrays: dict) -> tuple[dict, dict]:
        from types import SimpleNamespace

        from . import marketcast as M

        if op == "judge_snapshot":
            from ..pool.marketcast import load_snapshot

            digest = str(header["digest"])
            snap = self.snaps.get(digest)
            if snap is None:
                # incident-ok: IR010 runs on the MarketCast pod, not the box
                snap = load_snapshot(self._dir(header["dir"]), verify_signature=False,
                                     schema=str(header.get("schema", "")),
                                     select=dict(_tuples(header.get("select") or {})),
                                     channels=dict(_tuples(header.get("channels") or {})))
                if snap.digest != digest:
                    raise ValueError(f"snapshot digest {snap.digest[:16]} != {digest[:16]}")
                while len(self.snaps) >= SNAPSHOTS_KEEP:
                    self.snaps.pop(next(iter(self.snaps)))
                self.snaps[digest] = snap
            return {"digest": snap.digest, "meta_sha256": snapshot_meta_sha256(snap.meta)}, {}
        if op == "judge_round":
            text = str(header["mc"])
            self.mc, self.mc_sha = mc_from_wire(text), wire_sha256(text)
            self.ft_cache, self.datas = {}, {}
            if len(self.base_cache) > M.MKT_BASE_CACHE_MAX:
                self.base_cache = {}
            return {"mc_sha256": wire_sha256(mc_to_wire(self.mc))}, {}
        if self.mc is None:
            raise ValueError("judge op before judge_round")
        snap = self.snaps[str(header["snapshot"])]
        rid = str(header["round_id"])
        specs = dict(header.get("specs") or {})
        if op == "judge_rows":
            e = header["entry"]
            entry = SimpleNamespace(miner_hotkey=str(e["miner_hotkey"]),
                                    trained_pointer=str(e["trained_pointer"]),
                                    target=str(e["target"]))
            rows, data = M.compute_entry_rows(
                entry, phase=str(header["phase"]), after=float(header["after"]), snap=snap,
                mc=self.mc, base_fc=self._pod_fc(header["base"]), load_ft=self._load_ft(specs),
                base_cache=self.base_cache, round_id=rid, ft_cache=self.ft_cache,
                draw=header.get("draw") or None)
            if data is not None:
                self.datas[(rows.hotkey, rows.trained_pointer, rows.phase)] = data
            return self._reply(rows, data, n=len(rows.windows))
        rows = rows_from_wire(str(header["rows"]))
        data = self.datas[(rows.hotkey, rows.trained_pointer, rows.phase)]
        if op == "judge_near_dup":
            refs = [tuple(r) for r in header.get("refs") or ()]
            nd, blob = M.near_dup_record(rows, data, refs, snap=snap, mc=self.mc,
                                         load_ft=self._load_ft(specs), ft_cache=self.ft_cache,
                                         round_id=rid)
            out, arr = self._reply(None, blob, n=len(rows.windows))
            return {**out, "near_dup": nd}, arr
        if op == "judge_holder":
            h = header.get("holder")
            got, blob = M.holder_record(rows, data, None if h is None else tuple(h), snap=snap,
                                        mc=self.mc, load_ft=self._load_ft(specs),
                                        ft_cache=self.ft_cache, round_id=rid)
            return self._reply(got, blob, n=len(rows.windows))
        raise ValueError(f"unknown op {op!r}")

    @staticmethod
    def _reply(rows, blob, *, n: int) -> tuple[dict, dict]:
        import hashlib

        head = {"windows": int(n), "pod_peak_rss_mb": _peak_rss_mb()}
        if rows is not None:
            head["rows"] = rows_to_wire(rows)
        arr = {}
        if blob is not None:
            head["blob_sha256"] = hashlib.sha256(blob).hexdigest()
            arr["blob"] = np.frombuffer(bytes(blob), dtype=np.uint8)
        return head, arr

    def handle(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        op = header.get("op")
        if op == "hello":
            gpu = ""
            try:
                import torch

                if self.device.startswith("cuda") and torch.cuda.is_available():
                    gpu = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001 — informational
                pass
            return {"schema": SCHEMA, "device": self.device, "gpu": gpu, "ops": list(OPS)}, {}
        if op == "load":
            from ..eval.timestral_adapter import WEIGHTS_FILE, sha256_file

            family = str(header.get("family", "timestral"))
            if family not in LOADERS:
                raise ValueError(f"unknown model family {family!r}")

            d = (self.root / str(header["dir"])).resolve()
            if self.root not in d.parents:
                raise ValueError("checkpoint dir escapes the worker root")
            got = sha256_file(d / WEIGHTS_FILE)
            if got != str(header["sha256"]):
                raise ValueError(f"checkpoint sha256 {got[:16]} != requested "
                                 f"{str(header['sha256'])[:16]}")
            fc = LOADERS[family](d, self.device)
            self.models[str(header["model"])] = fc
            return {"levels": [float(x) for x in fc.quantile_levels]}, {}
        if op == "forecast":
            fc = self.models[str(header["model"])]
            q = fc.forecast_batch(arrays["ctx"], int(header["horizon"]),
                                  past_covariates=arrays.get("pc"),
                                  future_covariates=arrays.get("kc"))
            return {}, {"q": np.asarray(q, dtype=np.float64)}
        if isinstance(op, str) and op.startswith("judge_"):
            return self._judge(op, header, arrays)
        raise ValueError(f"unknown op {op!r}")

    def serve(self, inp, out) -> int:
        while True:
            got = read_frame(inp)
            if got is None:
                return 0
            header, arrays = got
            if header.get("op") == "bye":
                out.write(encode_frame({"ok": True}))
                out.flush()
                return 0
            try:
                reply, rarr = self.handle(header, arrays)
                reply = {"ok": True, **reply}
            except Exception as e:  # noqa: BLE001 — every error goes back, classified
                judge = str(header.get("op", "")).startswith("judge_")
                kind = e.kind if isinstance(e, MktRemoteError) else type(e).__name__
                # a judge's replayed box error / the fine-tune's own forward
                # error are what the in-process path raised: never the pod's
                infra = _infra_like(e) and not (judge and (
                    isinstance(e, MktRemoteError) or kind == "MktForwardError"))
                reply, rarr = {"ok": False, "error": kind,
                               "msg": str(e)[:JUDGE_MSG_MAX if judge else 2000],
                               "infra": bool(infra)}, {}
            out.write(encode_frame(reply, rarr))
            out.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cascade.validator.mkt_worker", description=__doc__)
    ap.add_argument("--serve", action="store_true", help="answer frames on stdin/stdout")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--root", default=".", help="checkpoint dirs are resolved under this")
    args = ap.parse_args(argv)
    if not args.serve:
        print(json.dumps({"schema": SCHEMA}))
        return 0
    # the protocol owns stdout: anything a library prints goes to stderr
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    return Worker(args.device, Path(args.root)).serve(sys.stdin.buffer, out)


if __name__ == "__main__":
    sys.exit(main())
