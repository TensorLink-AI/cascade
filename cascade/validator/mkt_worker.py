"""MarketCast forecast worker (DEC-CA-0064): the GPU half of the MAIN
validator's MarketCast rows, on an operator-rented pod
(:mod:`cascade.validator.mkt_offload`).

One long-lived process per validator publish run, spoken to over the ssh
session's stdin/stdout: length-prefixed NPZ frames (numeric and string arrays
only, never pickle). It holds Timestral checkpoints the validator uploaded
(sha256-verified here before every load) and answers ``forecast_batch`` calls
— contexts and covariates in, quantiles out. Windows' targets, losses, the
verdict, the report and the wallet never leave the validator.

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


def encode_frame(header: dict, arrays: dict | None = None) -> bytes:
    """``header`` (JSON) + named numeric arrays as one length-prefixed NPZ."""
    buf = io.BytesIO()
    np.savez(buf, h=np.array(json.dumps(header, sort_keys=True)),
             **{f"a_{k}": np.asarray(v) for k, v in (arrays or {}).items() if v is not None})
    data = buf.getvalue()
    return _LEN.pack(len(data)) + data


def decode_frame(data: bytes) -> tuple[dict, dict]:
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


class Worker:
    def __init__(self, device: str, root: Path):
        self.device, self.root = str(device), Path(root).resolve()
        self.models: dict[str, object] = {}

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
            return {"schema": SCHEMA, "device": self.device, "gpu": gpu}, {}
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
                reply, rarr = {"ok": False, "error": type(e).__name__, "msg": str(e)[:2000],
                               "infra": bool(_infra_like(e))}, {}
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
