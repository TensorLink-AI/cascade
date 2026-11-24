"""Shared spec and fixtures for the W8c acceptance tests (README.md has the
contract).

Copied from docs/plans/LANE_D_PLAN.md, W8c row (rev6.7, owner 2026-11-23):
"W8c builds a `LaneDInputs` per armed boundary from signed or pinned sources
only: (1) the queued entries from the W3 intake state and vault
(`cascade/lane_d/`), revealed by the boundary; (2) each entry's, KING_D's and
the house's forecasts on the boundary's frozen windows, computed on the lane D
eval host by the W5d scorer (`cascade/eval/lane_d_scorer.py`) over
checkpoints fetched through the W15 prefetch cache (`cascade/eval/
lane_d_cache.py`) and W4 ingest, behind the W4 handshake; (3) the pinned
`pool_d` snapshot(s) (W6d), each re-hashed against its pin, and their windows
(W5a/W5b); (4) the house record from the W5f signed pointer, and the bench
`SweepSession` (W3b*) for W8b's `bench`. **Deterministic** [...]
**Fail-closed:** each missing or unverifiable input is its own typed code
[...] **Off the poll thread** [...] **No new chain reads beyond
`ChainClient`.** **Unarmed:** no source is built and nothing is read
(invariant 1); nothing reachable from `cascade.audit` imports it (invariant
12). Wired into `cascade/validator/main.py` through `build_lane_d_step`'s
`inputs` and `bench` parameters."

The fixture is W8b's (W7a's synthetic chain, numpy only): challenger X
first-clears at B1, is confirmed and crowned (uid 6) at B2. Every external
service is a fake: the intake queue file (the real ``lane-d-intake-queue/1``
format, written here), the W6d pin list (:class:`Pool`), the eval host
(:class:`EvalHost`: the W4 hello plus W7a's synthetic forecasts), the
source's chain reader (:class:`Reader`: block timestamps and the registered
hotkeys, ``ChainClient`` method names) and the bench pod (W8b's
``BenchHost``). rev6.6: the readiness and T7 documents (:class:`Doc`) arm
4096 and 8192 by default (16384 lacks long series). The house pointer is a
real W5f signed pointer under a test key. No chain, storage, pod, torch or real key. Nothing here imports a module
in PATHS at import time.
"""

from __future__ import annotations

import importlib
import json
import threading
from pathlib import Path

import numpy as np

from tests.accept.w7a import helpers as w7a
from tests.accept.w8b import helpers as w8b

INPUTS = "cascade.validator.lane_d_inputs"
FORECASTS = "cascade.eval.lane_d_forecasts"

B1, B2, B3 = w8b.B1, w8b.B2, w8b.B3
ARM_BLOCK = w8b.ARM_BLOCK
CHALLENGER_UID = w8b.CHALLENGER_UID
N_UIDS = 16
HORIZON = w7a.HORIZON

# The codes README "Fail closed" pins, one per input.
INPUT_CODES = ("not_active", "chain_unavailable", "intake_unreadable", "reveal_unreadable",
               "metagraph_unreadable", "state_unreadable", "house_unavailable",
               "pool_unavailable", "pool_digest_mismatch", "handshake_mismatch",
               "forecasts_unavailable", "forecast_malformed")
FORECAST_CODES = ("checkpoint_unavailable", "checkpoint_refused", "forecast_failed",
                  "forecast_malformed")
MINER_FAULT_CODES = ("checkpoint_refused", "forecast_malformed")

GIT_HEAD = "a" * 40
PACKAGE_HASH = "b" * 64

# rev6.6 armed contexts (README "Armed contexts"): the codes a refused
# readiness / T7 document or an empty armed set gives, and the fixture's
# default armed set (W7a's contexts are 4096 / 8192 / 16384).
ARMED_CODES = ("readiness_unavailable", "t7_unavailable", "armed_invalid", "no_armed_context")
CONTEXTS = (4096, 8192, 16384)
DEFAULT_ARMED = (4096, 8192)


def assert_takes_armed_docs() -> None:
    """rev6.6: ``build_lane_d_inputs`` takes the readiness and T7 document
    sources as required keywords. On a build without them every test that
    builds a source fails here, on an assertion, never a ``TypeError``."""
    import inspect

    p = inspect.signature(inputs_mod().build_lane_d_inputs).parameters
    for name in ("readiness", "t7"):
        assert name in p, f"rev6.6: build_lane_d_inputs(..., {name}=) takes the {name} document"
        assert p[name].kind is inspect.Parameter.KEYWORD_ONLY, f"{name} is keyword-only"
        assert p[name].default is inspect.Parameter.empty, \
            f"{name} is required: a missing document fails closed, it never defaults"


def readiness_doc(armed=DEFAULT_ARMED, contexts=CONTEXTS) -> bytes:
    """A ``laned_readiness/1`` document: a context in ``armed`` has enough
    long series from enough sources (4633 from 42), the others don't (12
    from 3)."""
    rungs = [{"context": c, "series": 4633 if c in armed else 12,
              "sources": 42 if c in armed else 3} for c in contexts]
    return json.dumps({"schema": "laned_readiness/1", "rungs": rungs},
                      sort_keys=True).encode("utf-8")


def t7_doc(fits=CONTEXTS, contexts=CONTEXTS, *, lie: int | None = None) -> bytes:
    """A ``laned_t7/1`` pod document: a variant in ``fits`` is inside its
    budget, the others are over it. ``lie`` states ``fits`` wrongly for that
    variant (a hand edit the scorer refuses)."""
    rows = []
    for v in contexts:
        ok = v in fits
        rows.append({"variant": v, "wall_s": 600.0 if ok else 99999.0,
                     "peak_mem_gb": 8.0, "budget": {"wall_s": 5400.0, "mem_gb": 24.0},
                     "fits": (not ok) if v == lie else ok, "input_len": v, "history_min": v,
                     "peak_mem_source": "cuda_max_allocated"})
    return json.dumps({"schema": "laned_t7/1", "source": "pod", "rows": rows},
                      sort_keys=True).encode("utf-8")


def inputs_mod():
    """The W8c box-side module (in PATHS): imported inside test bodies only."""
    return importlib.import_module(INPUTS)


def forecasts_mod():
    """The W8c eval-host module (in PATHS): imported inside test bodies only."""
    return importlib.import_module(FORECASTS)


def house_key():
    """The W5f house pointer signer (a TEST key)."""
    return w7a.fixture_key("w8c-house-signer")


def identity():
    from cascade.eval.lane_d_handshake import CodeIdentity

    return CodeIdentity(git_head=GIT_HEAD, package_hash=PACKAGE_HASH)


def hello(**over) -> dict:
    from cascade.eval.lane_d_handshake import HANDSHAKE_SCHEMA

    return {"schema": HANDSHAKE_SCHEMA, "git_head": GIT_HEAD, "package_hash": PACKAGE_HASH,
            **over}


# ── the fakes ───────────────────────────────────────────────────────────────


class Reader:
    """The source's own read-only chain reader (``ChainClient`` method names:
    ``block_timestamp``, ``registered_hotkeys``). Records its calls and
    their threads."""

    def __init__(self, hotkeys: list[str], timestamps: dict[int, int]):
        self.hotkeys = list(hotkeys)
        self.timestamps = dict(timestamps)
        self.calls: list[tuple[str, int]] = []
        self.fail_timestamp = False
        self.fail_hotkeys = False

    def block_timestamp(self, block=None):
        self.calls.append(("block_timestamp", threading.get_ident()))
        if self.fail_timestamp:
            raise ConnectionError(f"w8c: block {block} timestamp unreadable")
        return float(self.timestamps[int(block)])

    def registered_hotkeys(self):
        self.calls.append(("registered_hotkeys", threading.get_ident()))
        if self.fail_hotkeys:
            raise ConnectionError("w8c: metagraph unreadable")
        return list(self.hotkeys)


class ReaderFactory:
    """``chain_factory`` for the source: records the threads it was called on."""

    def __init__(self, reader: Reader):
        self.reader = reader
        self.threads: list[int] = []
        self.fail = False

    def __call__(self):
        self.threads.append(threading.get_ident())
        if self.fail:
            raise ConnectionError("w8c: no chain endpoint")
        return self.reader


class Doc:
    """A pinned document source (README "Armed contexts"): ``doc()`` → the
    document's bytes, re-read at every call. Records its calls' threads;
    ``fail`` makes it raise (absent or unreadable)."""

    def __init__(self, data: bytes, name: str):
        self.data = data
        self.name = name
        self.calls: list[int] = []
        self.fail = False

    def __call__(self):
        self.calls.append(threading.get_ident())
        if self.fail:
            raise FileNotFoundError(f"w8c: {self.name} document absent")
        return self.data


class Pool:
    """The W6d pin list: ``pool(boundary)`` → ``[{"dir", "sha256"}]``, oldest
    first. B1 settles on snapshot 1, later boundaries on snapshot 2 (W8b)."""

    def __init__(self, fx: dict):
        self.fx = fx
        self.calls: list[int] = []
        self.fail = False
        self.override = None          # a callable boundary -> list, when set

    def pins(self, snap: int) -> dict:
        from cascade.validator.lane_d import load_pool

        d = self.fx["chain"]["pools"][snap]
        return {"dir": d, "sha256": load_pool(d, HORIZON)["sha256"]}

    def __call__(self, boundary):
        self.calls.append(int(boundary))
        if self.fail:
            raise OSError("w8c: pool_d pin list unreadable")
        if self.override is not None:
            return self.override(int(boundary))
        return [self.pins(1 if int(boundary) <= B1 else 2)]


def _hkey(history) -> bytes:
    return np.asarray(history, dtype=np.float64).tobytes()


class EvalHost:
    """The lane D eval host as the box sees it: ``hello()`` (W4) and
    ``forecast(digest, *, context, horizon, windows)`` → ``(n, horizon, 9)``,
    W7a's synthetic forecasts for that digest on those windows. ``faults``
    maps a digest to an exception to raise for it; ``shapes`` to an array to
    return instead."""

    def __init__(self, fx: dict, *, hello_doc: dict | None = None):
        self.fx = fx
        self.hello_doc = hello() if hello_doc is None else hello_doc
        self.calls: list[dict] = []
        self.hellos = 0
        # rev6.6: (armed contexts, forecasts made before) per set_armed call.
        self.armed_calls: list[tuple[list[int], int]] = []
        self.faults: dict[str, BaseException] = {}
        self.shapes: dict[str, np.ndarray] = {}
        # A window id can recur across snapshots with a different history (the
        # W7a trap series), so a forecast is looked up by (window_id, history).
        from cascade.validator.lane_d import load_pool

        fc = fx["chain"]["forecasts"]
        self.table: dict[str, dict] = {}
        for snap in sorted(fc):
            hist = {w["window_id"]: _hkey(w["history"])
                    for w in load_pool(fx["chain"]["pools"][snap], HORIZON)["windows"]}
            for dg, per in fc[snap].items():
                self.table.setdefault(dg, {}).update(
                    {(wid, hist[wid]): q for wid, q in per.items()})

    def hello(self):
        self.hellos += 1
        return dict(self.hello_doc)

    def set_armed(self, contexts):
        """rev6.6: the settlement's armed set (the contexts a forecast may run at)."""
        self.armed_calls.append(([int(c) for c in contexts], len(self.calls)))

    def forecast(self, digest, *, context, horizon, windows):
        windows = list(windows)
        self.calls.append({"digest": digest, "context": int(context), "horizon": int(horizon),
                           "window_ids": [w["window_id"] for w in windows],
                           "keys": sorted({k for w in windows for k in w}),
                           "histories": [np.asarray(w["history"]) for w in windows],
                           "thread": threading.get_ident()})
        if digest in self.faults:
            raise self.faults[digest]
        if digest in self.shapes:
            return self.shapes[digest]
        return np.stack([self.table[digest][(w["window_id"], _hkey(w["history"]))]
                         for w in windows])

    def digests(self) -> list[str]:
        return [c["digest"] for c in self.calls]


class Untouchable:
    """Any attribute use or call raises: proves a collaborator is never used."""

    def __init__(self, name: str):
        object.__setattr__(self, "_name", name)

    def __getattr__(self, attr):
        raise AssertionError(f"{self._name}.{attr} was used")

    def __call__(self, *a, **k):
        raise AssertionError(f"{self._name} was called")


# ── fixture files ───────────────────────────────────────────────────────────


def queue_row(entry: dict, *, state: str = "revealed", reveal_block=None) -> dict:
    """An intake queue row (``cascade/lane_d/service.py`` record shape)."""
    from cascade.lane_d.service import commit_payload, entry_prefix

    rb = entry["reveal_block"] if reveal_block is None else reveal_block
    prefix = entry_prefix("lane-d", entry["hotkey"], entry["digest"])
    return {"hotkey": entry["hotkey"], "digest": entry["digest"],
            "context": int(entry["context"]), "state": state,
            "files": {"config.json": 100, "model.safetensors": 1000},
            "weights": "c" * 64,
            "reveal_block": None if state in ("submitted", "ready") else rb,
            "closed_reason": "lost" if state == "closed" else None,
            "coldkey_at_ready": entry["coldkey_at_ready"],
            "ready_block": int(entry["ready_block"]),
            "ready_expires_block": int(entry["ready_block"]) + 7200,
            "weights_sha256": "c" * 64, "licence": "MIT",
            "licence_attestation_sha256": "d" * 64,
            "commit_payload": commit_payload(int(entry["context"]), prefix)}


def write_queue(intake_dir: Path, rows: list[dict]) -> Path:
    from cascade.lane_d.service import QUEUE_FILENAME, QUEUE_SCHEMA

    intake_dir.mkdir(parents=True, exist_ok=True)
    p = intake_dir / QUEUE_FILENAME
    p.write_text(json.dumps({"schema": QUEUE_SCHEMA,
                             "entries": sorted(rows, key=lambda r: r["hotkey"])},
                            sort_keys=True, indent=1), encoding="utf-8")
    return p


def write_house_pointer(path: Path, house: dict, key=None) -> Path:
    from cascade.validator.lane_d_house import sign_house_pointer

    pointer = sign_house_pointer(digest=house["digest"], reveal_ts=int(house["reveal_ts"]),
                                 signer=key or house_key())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pointer, sort_keys=True), encoding="utf-8")
    return path


def hotkeys(fx: dict) -> list[str]:
    """The metagraph: X at uid 6, fixture filler elsewhere."""
    out = [f"w8cFillerHotkey{u:03d}" for u in range(N_UIDS)]
    out[CHALLENGER_UID] = fx["entry"]["hotkey"]
    return out


def uid_of(fx: dict) -> dict:
    return {hk: u for u, hk in enumerate(hotkeys(fx))}


def window_ids(snap_dir) -> list[str]:
    from cascade.validator.lane_d import load_pool

    return sorted({w["window_id"] for w in load_pool(snap_dir, HORIZON)["windows"]
                   if w["window_id"]})


class Rig:
    """One validator's W8c source over the fakes: the lane D state dir (the
    W8b step's), the intake state dir with X revealed, the pinned pool, the
    signed house pointer and the eval host."""

    def __init__(self, tmp: Path, *, armed: bool = True, bench: bool = True,
                 entries: list[dict] | None = None, fx: dict | None = None):
        tmp = Path(tmp)
        self.tmp = tmp
        self.cfg = w8b.chain_cfg(tmp / "v", armed=armed)
        self.fx = fx if fx is not None else w8b.fixture(tmp)
        self.state_dir = tmp / "lane_d"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.intake_dir = tmp / "intake"
        rows = [queue_row(e) for e in ([self.fx["entry"]] if entries is None else entries)]
        write_queue(self.intake_dir, rows)
        self.house_path = write_house_pointer(tmp / "house" / "pointer.json", self.fx["house"])
        self.reader = Reader(hotkeys(self.fx), {w7a.REVEAL_BLOCK: w7a.T0})
        self.factory = ReaderFactory(self.reader)
        self.pool = Pool(self.fx)
        self.eval_host = EvalHost(self.fx)
        self.bench_host = w8b.BenchHost(self.fx["house"]["digest"]) if bench else None
        self.readiness = Doc(readiness_doc(), "readiness")
        self.t7 = Doc(t7_doc(), "t7")
        self.source = self.build()

    def build(self, **over):
        assert_takes_armed_docs()
        kw = dict(state_dir=self.state_dir, intake_state_dir=self.intake_dir,
                  chain_factory=self.factory, pool=self.pool, eval_host=self.eval_host,
                  house_pointer=self.house_path,
                  house_signer_ss58=house_key().ss58_address,
                  bench_host=self.bench_host, expected_identity=identity(),
                  readiness=self.readiness, t7=self.t7)
        kw.update(over)
        return inputs_mod().build_lane_d_inputs(self.cfg, **kw)

    def expected_armed(self) -> dict:
        """The W5d armed record of the documents the source reads now."""
        from cascade.eval.lane_d_scorer import armed_record

        return armed_record(self.cfg.lane_d.contexts, readiness=self.readiness.data,
                            t7=self.t7.data, min_long_series=self.cfg.lane_d.min_long_series)

    def expected(self, boundary: int) -> dict:
        """``expected_inputs`` plus the armed record (``LaneDInputs`` kwargs)."""
        return dict(expected_inputs(self.fx, boundary), armed=self.expected_armed())

    @property
    def state_path(self) -> Path:
        return self.state_dir / "lane_d_state.json"

    def files(self) -> dict:
        return {"state": w8b.dir_bytes(self.state_dir), "intake": w8b.dir_bytes(self.intake_dir),
                "house": w8b.dir_bytes(self.house_path.parent)}

    def step(self, *, store=None, factory=None):
        """W8b's production step with this source as ``inputs`` and its bench
        factory as ``bench``."""
        from cascade.validator.lane_d_wiring import build_lane_d_step

        return build_lane_d_step(self.cfg, state_dir=self.state_dir, signer=w8b.signer(),
                                 chain_factory=factory or w8b.ReaderFactory(),
                                 store=store if store is not None else w8b.Store(),
                                 inputs=self.source, bench=self.source.bench)


def expected_inputs(fx: dict, boundary: int) -> dict:
    """What the source must return at ``boundary`` (README "The inputs"):
    W8b's hand-built inputs with the metagraph's full ``{hotkey: uid}``."""
    snap = 1 if int(boundary) <= B1 else 2
    chain = fx["chain"]
    return {"pool_dir": Path(chain["pools"][snap]), "prior_pool_dirs": (),
            "entries": [dict(fx["entry"])],
            # Only the needed digests (X and the house): W7a's chain also
            # carries forecasts for its `loser` digest (main @ 2cd79be, #758),
            # which no W8c settlement reads.
            "forecasts": {dg: f for dg, f in chain["forecasts"][snap].items()
                          if dg in (fx["x"], fx["house"]["digest"])},
            "uid_of": uid_of(fx), "house": dict(chain["house"])}


def assert_inputs_error(exc: BaseException, code: str) -> None:
    m = inputs_mod()
    assert isinstance(exc, m.LaneDInputsError), f"expected LaneDInputsError, got {exc!r}"
    assert exc.code in m.INPUTS_ERROR_CODES, exc.code
    assert exc.code == code, f"code {exc.code!r} != {code!r} ({exc})"


def forecasts_equal(got: dict, want: dict) -> bool:
    if set(got) != set(want):
        return False
    for dg in want:
        if set(got[dg]) != set(want[dg]):
            return False
        for wid in want[dg]:
            if not np.array_equal(np.asarray(got[dg][wid], dtype=np.float64),
                                  np.asarray(want[dg][wid], dtype=np.float64)):
                return False
    return True
