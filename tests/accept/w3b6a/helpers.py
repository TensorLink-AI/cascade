"""Shared fixtures for the W3b6a acceptance tests (README.md has the contract).

From docs/plans/LANE_D_PLAN.md, rev6.13 "W3b6 spec" and the W3b6a row: the
bench host's end of the lane D sweep (W15 cache + W4 ingest per digest,
materialised in lane A's sidecar layout at the effective context, one paired
sidecar pass, ``item_rows_from_report`` rows), the box's ``SSHBenchHost``
(the ``SweepSession`` sweep over SSH, grants only) and the optional
``bench_hosts`` key of the ``--lane-d-inputs`` TOML.

Every external service is a fake: the sidecar (:class:`Sidecar`, returning
recorded-shape sidecar reports), the SSH runner (:class:`Run`), W4's
``ingest_presigned``. Cache entries carry the release's real pinned
``cascade_pyramid`` code bytes so the materialised dir passes
``verify_checkpoint_code``. No chain, storage, pod, torch or GPU. Nothing here
imports a module in PATHS at import time.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

SWEEP = "cascade.eval.lane_d_bench_sweep"
HOST = "cascade.validator.lane_d_bench_host"
REQUEST_SCHEMA = "lane-d-bench-request/1"
REPLY_SCHEMA = "lane-d-bench-reply/1"
SUITES = ("gift", "boom")
FINGERPRINT = {"gpu": "w3b6a test GPU", "driver": "999.1", "torch": "2.4.1+cu124"}
REPO = Path(__file__).resolve().parents[3]


def sweep_mod():
    return importlib.import_module(SWEEP)


def host_mod():
    return importlib.import_module(HOST)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(files: dict[str, bytes]) -> str:
    """The W1.5/W4 entry digest: sha256 of the canonical file manifest."""
    manifest = {"files": {n: {"sha256": sha256(d), "size": len(d)}
                          for n, d in sorted(files.items())}}
    return sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())


class Entry:
    """A W4-shaped lane D checkpoint: the release's own pinned pyramid code,
    a config.json and opaque weight bytes (the cache does not parse them)."""

    def __init__(self, seed: int, context: int = 4096):
        from cascade.eval.checkpoint_guard import expected_checkpoint_code

        code = expected_checkpoint_code("cascade_pyramid")
        cfg = {"context_length": context, "patch_size": 32, "seed": seed}
        self.files = {"config.json": json.dumps(cfg, sort_keys=True).encode(),
                      "forecast_wrapper.py": code["forecast_wrapper.py"],
                      "model.py": code["model.py"],
                      "weights.safetensors": hashlib.sha256(b"w3b6a-%d" % seed).digest() * 8}
        self.digest = digest_of(self.files)

    def fill(self, directory) -> None:
        for name, data in self.files.items():
            (Path(directory) / name).write_bytes(data)


def cache(tmp: Path, *entries):
    from cascade.eval.lane_d_cache import PrefetchCache

    c = PrefetchCache(Path(tmp) / "cache", budget_bytes=1 << 24)
    for e in entries:
        c.put(e.digest, e.fill)
    return c


SIDECAR_NAME = {"gift": "gift-eval", "boom": "boom", "time": "time", "fev": "fev"}


def report_for(seed: int, suites=SUITES) -> dict:
    """A raw sidecar report in the shape ``item_rows_from_report`` parses
    (sidecar suite names, ``full`` / ``CRPS`` / ``MASE`` rows)."""
    return {"suites": [{"suite": SIDECAR_NAME[s], "status": "ok",
                        "rows": [{"full": f"{s}/item{i}", "CRPS": 0.1 * (i + 1) + 0.01 * seed,
                                  "MASE": 1.0 + 0.1 * i + 0.01 * seed} for i in range(4)]}
                       for s in suites]}


class Sidecar:
    """The paired sidecar pass: ``sidecar(dirs, suites) -> [report | None]``.
    ``reports`` maps a digest (taken from the dir's ``.digest`` marker or its
    name) to the report to return; a missing one returns ``None``."""

    def __init__(self, reports: dict[str, dict] | None = None, fail: bool = False):
        self.reports = dict(reports or {})
        self.calls: list[dict] = []
        self.fail = fail

    def __call__(self, dirs, suites):
        dirs = [Path(d) for d in dirs]
        self.calls.append({"dirs": dirs, "suites": tuple(suites)})
        if self.fail:
            return [None] * len(dirs)
        return [self.reports.get(self.digest_of_dir(d)) for d in dirs]

    @staticmethod
    def digest_of_dir(d: Path) -> str:
        return d.name.split("-c")[0]


def request(digests, *, suites=SUITES, contexts=None, grants=None) -> dict:
    ds = list(digests)
    return {"schema": REQUEST_SCHEMA, "digests": ds, "suites": list(suites),
            "contexts": dict(contexts) if contexts is not None else {d: 4096 for d in ds},
            "grants": dict(grants) if grants is not None else {d: None for d in ds}}


def grant(digest: str) -> dict:
    return {"hotkey": "w3b6aMinerHotkey", "digest": digest,
            "manifest_url": "https://vault.invalid/manifest.json", "files": {}}


def fake_ingest(monkeypatch, entry: Entry | None, exc: BaseException | None = None):
    """W4's ``ingest_presigned`` stubbed: writes ``entry``'s files, or raises."""
    import cascade.eval.lane_d_ingest as ingest

    calls = []

    def ingest_presigned(g, dest_dir, *, contract=None, expect=None):
        calls.append(dict(g))
        if exc is not None:
            raise exc
        out = Path(dest_dir) / g["digest"]
        out.mkdir(parents=True, exist_ok=True)
        entry.fill(out)
        return out

    monkeypatch.setattr(ingest, "ingest_presigned", ingest_presigned)
    return calls


# ── the box side ────────────────────────────────────────────────────────────


def hosts_file(tmp: Path, *, stage: str = "final") -> Path:
    p = Path(tmp) / "bench_hosts.toml"
    p.write_text(f'[[host]]\nname = "laned-bench"\nhost = "10.0.0.7"\nstage = "{stage}"\n'
                 'workdir = "/root/cascade"\nremote_python = "/root/cascade/.venv/bin/python"\n',
                 encoding="utf-8")
    return p


class Run:
    """The SSH runner (``subprocess.run`` shape): records argv and stdin;
    answers with ``reply`` (a dict, printed as one JSON line) or ``raw``."""

    def __init__(self, reply=None, *, raw: bytes | None = None, returncode: int = 0):
        self.reply, self.raw, self.returncode = reply, raw, returncode
        self.calls: list[dict] = []

    def __call__(self, argv, *, input=None, capture_output=True, timeout=None, check=False):
        self.calls.append({"argv": list(argv), "input": input, "timeout": timeout})
        out = self.raw if self.raw is not None else (json.dumps(self.reply) + "\n").encode()
        return SimpleNamespace(returncode=self.returncode, stdout=b"log line\n" + out,
                               stderr=b"")

    def request(self, i: int = -1) -> dict:
        return json.loads(self.calls[i]["input"].decode("utf-8"))


def reply(rows: dict) -> dict:
    return {"schema": REPLY_SCHEMA, "rows": rows, "fingerprint": dict(FINGERPRINT)}


def rows_entry(seed: int, suites=SUITES) -> dict:
    return {s: {"status": "ok", "rows": [{"item": f"{s}/item{i}", "crps": 0.1 + 0.01 * seed,
                                          "mase": 1.1 + 0.01 * i}
                                         for i in range(3)]} for s in suites}
