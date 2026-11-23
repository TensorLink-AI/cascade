"""Shared spec and fixtures for the W8b acceptance tests (README.md has the
contract).

Copied from docs/plans/LANE_D_PLAN.md, W8b row (rev6.2; owner accepted
2026-11-23): "The validator loop calls it [`settle_lane_d`] at each lane D
settlement boundary with: the boundary block hash (read-only chain read); the
era's sentinel (W-S1 draw from the era-start hash); a bench `SweepSession` on
the configured bench host (the W3b2/W5e gate inputs; an armed crown without
them is a `bench_hold`); `expect_prior_sha256` from the last published receipt
(W7c); the low-frequency check (W3b5); and the house/king state. After
settlement: W3b3 `sweep_and_publish` for a new king; W-S3
`build_sentinel_board` with the era's kings list (from W7c state / W17);
publish the signed report + receipt; lane D pay reaches weights (W8 emission)
**only via the published report**. **Fail-closed:** any unreadable input ⇒ no
settlement this tick (retry next tick), never a guessed input.
**Restart-safe:** a re-call after restart is idempotent (W7c). **Weight
loop:** lane D's bench work runs off the weight loop (W19) [...] **Unarmed:**
byte-identical (invariant 1)."

Every settlement is a real `settle_lane_d` run on W7a's synthetic chain
(`tests/accept/w7a/helpers.py`: numpy only, no torch): challenger X
is listed from B2, where it clears against the house at its first verdict
and is crowned (uid 6) through the gate (plan rev6.11 (b): no confirmation
duel); B1 lists no entry and B3 judges nothing. The external services are fakes: the step's chain
reader (:class:`Reader`, block hashes only), the poll thread's chain client
(:class:`PollChain`, the weight loop's ``ChainClient`` duck type), the bench
pod (:class:`BenchHost`) and the storage bucket (:class:`Store`). No real
chain, storage, pod or signing key. Nothing here imports a module in PATHS at
import time.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import struct
import threading
import tomllib
from dataclasses import fields, replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace

from tests.accept.w7a import helpers as w7a

REPO = Path(__file__).resolve().parents[3]
WIRING = "cascade.validator.lane_d_wiring"
W19_WORKER = "cascade.validator.offthread_worker"   # W19 (#691: not lane_d_*)

ARM_BLOCK = w7a.ARM_BLOCK                 # 9,720,000: an era boundary (3600 grid)
SETTLEMENT = w7a.SETTLEMENT               # 900
B1 = w7a.BOUNDARY_1                       # no entry listed; the house holds KING_D
B2 = w7a.BOUNDARY_2                       # X's first verdict: cleared, gated, crowned
B3 = ARM_BLOCK + 3 * SETTLEMENT           # judges nothing; X keeps KING_D
ERA_START = ARM_BLOCK                     # the era B1..B3 belong to
CHALLENGER_UID = w7a.CHALLENGER_UID       # 6: KING_D's uid once crowned
OTHER_UID = 5                             # a second court uid for decay checks

KING = "w8bKingHotkey"                    # lane A's king (genesis state)
KING_UID = 7
N_UIDS = 256
INTERVAL = 180                            # [validator] weight_set_interval_blocks
POLL_BLOCKS = 10                          # poll_seconds = 120 at 12 s blocks
TICK_BUDGET_S = 1.0
HANG_S = 60.0

# Unarmed goldens recorded on main @ 54e7f23 (README "Unarmed goldens"): the
# shipped chain.toml (real loader parse), genesis(KING, 7), 256 uids, the
# head re-assert at the block. The vector is hashed as little-endian float64.
UNARMED_VECTOR_SHA256 = "951fc9ae3c3ed1021ed21ae196ba771a07a3c85aa1131d75380529c5eeeea35c"
UNARMED_RECEIPT_SHA256 = {
    ARM_BLOCK + 905: "c1b7b52481fb7a90ff568a1fba3e8b929b13f56b82cb4d6ca0f7f66f50a47f22",
    ARM_BLOCK + 1805: "e34e5474e89b6a788b378a9860085a48f1a38682e3bda5548856283d729e6bf9",
}
RECEIPT_BASE_SEED = 1234567
RECEIPT_VALIDATOR = "w8bValidatorHotkey"


def wiring():
    """The W8b module (in PATHS): imported inside test bodies only."""
    return importlib.import_module(WIRING)


def signer():
    return w7a.fixture_key("signer")


def block_hash(block: int) -> str:
    """The fixture chain's block hash at ``block`` (``0x`` + 64 hex)."""
    return "0x" + hashlib.sha256(f"w8b block {int(block)}".encode()).hexdigest()


# ── config ──────────────────────────────────────────────────────────────────


@cache
def _shipped_raw() -> str:
    return (REPO / "chain.toml").read_text(encoding="utf-8")


def shipped_cfg(state_dir: Path):
    """The shipped chain.toml (real loader parse), validator state in
    ``state_dir``; the config the unarmed goldens were recorded under."""
    from cascade.shared.config import parse_chain_config

    cfg = parse_chain_config(tomllib.loads(_shipped_raw()))
    v = replace(cfg.validator, weight_set_interval_blocks=INTERVAL, force_burn=False,
                state_db_path=str(Path(state_dir) / "validator_state.db"))
    return replace(cfg, validator=v)


def chain_cfg(state_dir: Path, *, armed: bool = True, sentinel: bool = False,
              lanes_bc: bool = True, force_burn: bool = False):
    """The shipped config with ``[lane_d]`` = W7a's armed table (from_block
    ARM_BLOCK, share 0.25 from lane A) when ``armed``; ``sentinel`` arms W-S1
    (``[v2] sentinel_from_block = ARM_BLOCK``); ``lanes_bc=False`` switches
    lanes B and C off; the lane D report signer (`[manifest]
    validator_hotkey`, W7b `pinned_report_signer`) is the fixture key."""
    cfg = shipped_cfg(state_dir)
    if armed:
        cfg = replace(cfg, lane_d=w7a.lane_d_cfg())
    if sentinel:
        cfg = replace(cfg, v2=replace(cfg.v2, sentinel_from_block=ARM_BLOCK))
    if not lanes_bc:
        cfg = replace(cfg, lane_b=replace(cfg.lane_b, from_block=0),
                      lane_c=replace(cfg.lane_c, from_block=0))
    cfg = replace(cfg, manifest=replace(cfg.manifest, validator_hotkey=signer().ss58_address))
    if force_burn:
        cfg = replace(cfg, validator=replace(cfg.validator, force_burn=True))
    return cfg


# ── the fake services ───────────────────────────────────────────────────────


class Reader:
    """The lane D step's OWN read-only chain reader (block hashes only)."""

    def __init__(self):
        self.calls: list[tuple[int, int]] = []     # (block, thread)
        self.fail: set[int] = set()                 # blocks whose read raises
        self.malformed: set[int] = set()            # blocks read as a bad hash

    def block_hash(self, block):
        self.calls.append((int(block), threading.get_ident()))
        if int(block) in self.fail:
            raise ConnectionError(f"w8b: block {block} unreadable")
        if int(block) in self.malformed:
            return "0xnot-a-block-hash"
        return block_hash(block)

    def blocks(self) -> list[int]:
        return [b for b, _ in self.calls]


class ReaderFactory:
    """``chain_factory``: builds the step's own reader; records its thread."""

    def __init__(self, reader: Reader | None = None):
        self.reader = reader or Reader()
        self.threads: list[int] = []
        self.fail = False

    def __call__(self):
        self.threads.append(threading.get_ident())
        if self.fail:
            raise ConnectionError("w8b: no chain endpoint")
        return self.reader


class BenchHost:
    """The bench pod's sweep (W7a's healthy rows: every candidate passes the
    paired gate). ``hang`` blocks every call until released; ``fail_suites``
    makes a sweep of exactly those suites raise (a pod error); a suite in
    ``error_suites`` comes back as an errored suite record for every model."""

    def __init__(self, house_digest: str, *, hang: bool = False):
        self.healthy = w7a.HealthyBench(house_digest)
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        self.threads: list[int] = []
        self.hang = hang
        self.release = threading.Event()
        self.started = threading.Event()
        self.fail_suites: set[tuple[str, ...]] = set()
        self.error_suites: set[str] = set()

    def __call__(self, digests, suites):
        self.calls.append((tuple(digests), tuple(suites)))
        self.threads.append(threading.get_ident())
        self.started.set()
        if self.hang:
            self.release.wait(HANG_S)
        if tuple(suites) in self.fail_suites:
            raise ConnectionError(f"w8b: bench pod lost sweeping {suites}")
        out = self.healthy(digests, suites)
        for dg in out:
            for s in self.error_suites & set(out[dg]):
                out[dg][s] = {"status": "error", "rows": []}
        return out


class BenchFactory:
    """``bench``: one :class:`SweepSession` per boundary on ``host``."""

    def __init__(self, host: BenchHost):
        self.host = host
        self.boundaries: list[int] = []
        self.fail = False

    def __call__(self, boundary):
        from cascade.validator.lane_d_bench import SweepSession

        self.boundaries.append(int(boundary))
        if self.fail:
            raise ConnectionError("w8b: bench host unreachable")
        return SweepSession(session_id(boundary), self.host)


def session_id(boundary: int) -> str:
    return f"w8b-bench/{int(boundary)}"


class Store:
    """The bucket (``get_text`` / ``put_text``, the Hippius store duck type):
    a missing key is ``ObjectNotFound``; ``fail_get`` / ``fail_put`` hold
    key predicates whose call raises ``StorageError``."""

    def __init__(self):
        self.objects: dict[str, str] = {}
        self.puts: list[str] = []
        self.gets: list[str] = []
        self.fail_get = None
        self.fail_put = None

    def get_text(self, key):
        from cascade.shared.hippius import ObjectNotFound, StorageError

        self.gets.append(key)
        if self.fail_get is not None and self.fail_get(key):
            raise StorageError(f"w8b: GET {key} failed")
        if key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key]

    def put_text(self, key, text, content_type=None, acl=None):
        from cascade.shared.hippius import StorageError

        if self.fail_put is not None and self.fail_put(key):
            raise StorageError(f"w8b: PUT {key} failed")
        self.objects[key] = str(text)
        self.puts.append(key)


class Inputs:
    """``inputs(boundary)``: the settlement's pool snapshot, revealed entries,
    eval-host forecasts, metagraph and house (sourced outside W8b; README).
    B1 settles on snapshot 1, later boundaries on snapshot 2. The entries are
    listed from B2 (rev6.11 (b): a challenger listed at B1 would be crowned
    at its first verdict there)."""

    def __init__(self, fx: dict, *, entries: list[dict] | None = None, raw: bool = False,
                 listed_from: int | None = None):
        self.fx = fx
        self.listed_from = B2 if listed_from is None else int(listed_from)
        self.entries = [fx["entry"]] if entries is None else list(entries)
        self.calls: list[int] = []
        self.fail = False
        self.raw = raw            # the oracle's plain record (no module under test)

    def __call__(self, boundary):
        self.calls.append(int(boundary))
        if self.fail:
            raise OSError("w8b: pool_d snapshot unreadable")
        snap = 1 if int(boundary) <= B1 else 2
        chain = self.fx["chain"]
        kw = {"pool_dir": chain["pools"][snap], "prior_pool_dirs": (),
              "entries": ([dict(e) for e in self.entries]
                          if int(boundary) >= self.listed_from else []),
              "forecasts": chain["forecasts"][snap], "uid_of": dict(self.fx["uid_of"]),
              "house": dict(chain["house"])}
        if self.raw:
            return SimpleNamespace(**kw)
        return wiring().LaneDInputs(**kw)


# ── fixtures ────────────────────────────────────────────────────────────────


def fixture(tmp: Path) -> dict:
    """W7a's synthetic chain: two pool_d snapshots, challenger X, the house."""
    chain = w7a.build_chain(tmp / "chain")
    e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
    return {"chain": chain, "entry": e, "uid_of": {e["hotkey"]: CHALLENGER_UID},
            "x": e["digest"], "house": chain["house"]}


class Rig:
    """One validator's lane D wiring: the step built by
    ``build_lane_d_step`` over the fakes, its state dir and its bucket."""

    def __init__(self, tmp: Path, cfg, fx: dict, *, store: Store | None = None,
                 host: BenchHost | None = None, inputs: Inputs | None = None,
                 bench: bool = True, factory: ReaderFactory | None = None,
                 state_dir: Path | None = None, with_inputs: bool = True):
        self.cfg, self.fx = cfg, fx
        self.state_dir = Path(state_dir) if state_dir is not None else Path(tmp) / "lane_d"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = store if store is not None else Store()
        self.host = host if host is not None else BenchHost(fx["house"]["digest"])
        self.bench = BenchFactory(self.host) if bench else None
        self.inputs = (inputs if inputs is not None else Inputs(fx)) if with_inputs else None
        self.factory = factory if factory is not None else ReaderFactory()
        self.step = self.build()

    def build(self):
        return wiring().build_lane_d_step(
            self.cfg, state_dir=self.state_dir, signer=signer(),
            chain_factory=self.factory, store=self.store, inputs=self.inputs,
            bench=self.bench)

    def restart(self):
        """A fresh process: a new step over the same state dir and bucket."""
        self.step = self.build()
        return self.step

    @property
    def state_path(self) -> Path:
        return self.state_dir / "lane_d_state.json"

    def report_file(self, boundary: int) -> Path:
        from cascade.validator.lane_d_state import report_path

        return report_path(self.state_path, boundary)

    def published_report(self, boundary: int) -> dict:
        return json.loads(self.store.objects[wiring().report_key(boundary)])

    def published_receipt(self, boundary: int) -> dict:
        return json.loads(self.store.objects[wiring().receipt_key(boundary)])


def era_sentinel(cfg, boundary: int) -> str | None:
    """The §16 sentinel suite of the era ``boundary`` settles in (None while
    W-S1 is unarmed there), drawn from the era-start block hash."""
    from cascade.shared.sentinel import sentinel_armed, sentinel_for_era, settlement_era_start

    era = settlement_era_start(cfg, int(boundary))
    if not sentinel_armed(cfg, era):
        return None
    return sentinel_for_era(block_hash(era), cfg.v2.sentinel_suites)


class Oracle:
    """The same settlements made by hand with the W8b row's inputs: a direct
    ``settle_lane_d`` call (boundary hash, era sentinel, the boundary's bench
    session; the prior is checked here against the last receipt), then W3b3
    ``sweep_and_publish`` for a new king and the W17 status document (which
    embeds the W-S3 board) into its own bucket."""

    def __init__(self, tmp: Path, cfg, fx: dict):
        self.cfg, self.fx = cfg, fx
        self.state_path = Path(tmp) / "oracle" / "lane_d_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.store = Store()
        self.host = BenchHost(fx["house"]["digest"])
        self.inputs = Inputs(fx, raw=True)
        self.settlements: dict[int, object] = {}
        self.receipts: list[dict] = []

    def settle(self, boundary: int):
        from cascade.shared.sentinel import settlement_era_start
        from cascade.validator.lane_d import settle_lane_d
        from cascade.validator.lane_d_bench import SweepSession
        from cascade.validator.lane_d_sentinel import sweep_and_publish
        from cascade.validator.lane_d_state import load_state
        from cascade.validator.lane_d_status import build_lane_d_status, publish_lane_d_status

        inp = self.inputs(boundary)
        session = SweepSession(session_id(boundary), self.host)
        prior = self.receipts[-1]["posterior_state_sha256"] if self.receipts else None
        s = settle_lane_d(cfg=self.cfg.lane_d, boundary=int(boundary), pool_dir=inp.pool_dir,
                          entries=inp.entries, house=inp.house, forecasts=inp.forecasts,
                          uid_of=inp.uid_of, state_path=self.state_path, signer=signer(),
                          prior_pool_dirs=inp.prior_pool_dirs, expect_prior_sha256=prior,
                          boundary_block_hash=block_hash(boundary),
                          sentinel=era_sentinel(self.cfg, boundary), bench=session)
        era = settlement_era_start(self.cfg, int(boundary))
        if s.report.get("crowned") is not None and era_sentinel(self.cfg, boundary):
            sweep_and_publish(s.report, era_block=era, era_block_hash=block_hash(era),
                              sentinel_suites=self.cfg.v2.sentinel_suites,
                              gate_suite=s.report["gate"]["suite"], bench=session,
                              store=self.store)
        state = load_state(self.state_path, inp.house)
        doc = build_lane_d_status(self.cfg, state=state, report=s.report, block=int(boundary),
                                  era_block=era, era_block_hash=block_hash(era),
                                  store=self.store)
        publish_lane_d_status(self.store, doc)
        self.settlements[int(boundary)] = s
        self.receipts.append(s.receipt)
        return s

    def status_text(self) -> str:
        from cascade.validator.lane_d_status import LANE_D_STATUS_KEY

        return self.store.objects[LANE_D_STATUS_KEY]


# ── generic helpers ─────────────────────────────────────────────────────────


def dir_bytes(d: Path) -> dict[str, bytes]:
    """Every file under ``d`` (relative path → bytes)."""
    d = Path(d)
    if not d.exists():
        return {}
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def body_sha256(report: dict) -> str:
    from cascade.validator.lane_d_state import report_digest

    return report_digest(report)


def vector_sha256(vec) -> str:
    vals = [float(x) for x in vec]
    return hashlib.sha256(struct.pack(f"<{len(vals)}d", *vals)).hexdigest()


def assert_wiring_error(exc: BaseException, code: str) -> None:
    w = wiring()
    assert isinstance(exc, w.LaneDWiringError), f"expected LaneDWiringError, got {exc!r}"
    assert exc.code in w.WIRING_ERROR_CODES, exc.code
    assert exc.code == code, f"code {exc.code!r} != {code!r} ({exc})"


# ── the weight loop ─────────────────────────────────────────────────────────


class PollChain:
    """The poll thread's chain client as the weight loop calls it. Every push
    is logged with its block, vector and thread; ``block_hash`` calls are
    logged too (the lane D step must never use this client)."""

    def __init__(self, block: int, n: int = N_UIDS):
        self.block = int(block)
        self.hotkeys = [f"w8bhk{i}" for i in range(n)]
        self.hotkeys[KING_UID] = KING
        self.pushes: list[tuple[int, tuple[float, ...], int]] = []
        self.hash_threads: list[int] = []
        self._lock = threading.Lock()

    def current_block(self) -> int:
        return self.block

    def n_uids(self) -> int:
        return len(self.hotkeys)

    def registered_hotkeys(self) -> list[str]:
        return list(self.hotkeys)

    def uid_for_hotkey(self, hotkey):
        return self.hotkeys.index(hotkey) if hotkey in self.hotkeys else None

    def block_hash(self, block):
        self.hash_threads.append(threading.get_ident())
        return block_hash(block)

    def _log(self, vec) -> None:
        with self._lock:
            self.pushes.append((self.block, tuple(float(w) for w in vec),
                                threading.get_ident()))

    def set_equal_share_weights(self, reward_uids, n_uids, *, decay, burn_uid):
        from cascade.shared.chain import decayed_share_vector

        self._log(decayed_share_vector(list(reward_uids), n_uids, decay=decay,
                                       burn_uid=burn_uid))

    def set_weight_vector(self, weights):
        self._log(weights)
        return True

    def push_blocks(self) -> list[int]:
        return [b for b, _, _ in self.pushes]

    def last_vector(self) -> tuple[float, ...]:
        assert self.pushes, "no weight vector was pushed"
        return self.pushes[-1][1]


class CourtStep:
    """A stand-in lane D step that only answers ``court_for`` (the weight
    fold's one read of lane D, README "Weights")."""

    def __init__(self, court=None, *, raises: BaseException | None = None):
        self.court = court
        self.raises = raises
        self.court_calls: list[tuple[int, int]] = []
        self.step_calls: list[int] = []

    def __call__(self, boundary):
        self.step_calls.append(int(boundary))

    def court_for(self, boundary):
        self.court_calls.append((int(boundary), threading.get_ident()))
        if self.raises is not None:
            raise self.raises
        return None if self.court is None else list(self.court)


def runner(cfg, *, step=None):
    """A real ValidatorRunner (genesis champion state, lane A's king at uid
    7) with ``lane_d_step`` = ``step``."""
    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.state import genesis

    r = ValidatorRunner(cfg=cfg, state=genesis(KING, KING_UID))
    r.lane_d_step = step
    return r


def vector_at(cfg, block: int, *, step=None):
    """The vector a fresh runner pushes at its first head re-assert at
    ``block`` (and the runner)."""
    r = runner(cfg, step=step)
    chain = PollChain(block)
    r._reassert_weights_at_head(chain)
    return chain.last_vector(), r


def want_vector(r, cfg, boundary: int, court) -> list[float]:
    """W8's emission for a vector at ``boundary`` with lanes B and C off:
    lane A's court ``[KING_UID]`` in lane A's pool under the runner's
    DEC-CA-0055 policy, lane D's court ``court`` (king first; ``None`` and an
    empty court burn) in lane D's share, lane D's part always in
    ``emission_vector``'s second argument (RV3 note 10)."""
    from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts

    pol = r._emission_policy(int(boundary))
    if pol.inert:
        pol = EmissionPolicy()
    a, b = lane_parts([KING_UID], N_UIDS, king_decay=cfg.scoring.king_decay,
                      burn_uid=cfg.scoring.burn_uid, lane_d_cfg=cfg.lane_d,
                      d_court_uids=list(court))
    return emission_vector(a, b, pol, burn_uid=cfg.scoring.burn_uid, lane_a_uids=[KING_UID])


def close(got, want, rel: float = 1e-12) -> bool:
    import math

    return len(got) == len(want) and all(
        math.isclose(float(g), float(w), rel_tol=rel, abs_tol=1e-15)
        for g, w in zip(got, want, strict=True))


def w19_surface() -> None:
    """W19's seam (tests/accept/w19/README.md) must be on the runner."""
    from cascade.validator.loop import ValidatorRunner

    assert importlib.util.find_spec(W19_WORKER) is not None, \
        f"W19's {W19_WORKER} is not on this tree (W8b builds on W19)"
    names = {f.name for f in fields(ValidatorRunner)}
    assert "lane_d_step" in names, "ValidatorRunner needs W19's `lane_d_step` field"
    assert callable(getattr(ValidatorRunner, "_lane_d_tick", None)), \
        "ValidatorRunner needs W19's `_lane_d_tick(client)`"


def drive(r, chain: PollChain, blocks) -> list[float]:
    """run_forever's per-poll lane D tick then head re-assert, one poll per
    block; returns each poll's wall-clock seconds."""
    import time

    spent = []
    for b in blocks:
        chain.block = int(b)
        t0 = time.monotonic()
        r._lane_d_tick(chain)
        r._reassert_weights_at_head(chain)
        spent.append(time.monotonic() - t0)
    return spent


def polls(start: int, stop: int) -> list[int]:
    return list(range(int(start), int(stop), POLL_BLOCKS))


def module_source(name: str) -> str:
    mod = importlib.import_module(name)
    return Path(mod.__file__).read_text(encoding="utf-8")


def unarmed_receipt_sha256(r, vec, block: int) -> str:
    """sha256 of the round receipt the runner assembles for the settlement
    at ``block``'s boundary (a no-contest round carrying ``vec``), as recorded
    for :data:`UNARMED_RECEIPT_SHA256`."""
    from cascade.shared.receipt import dump_receipt
    from tests.unit.receipt_fixture import make_manifest

    m = make_manifest(None, base_seed=RECEIPT_BASE_SEED)
    rec = r.build_round_receipt(m, base_seed=RECEIPT_BASE_SEED,
                                epoch_start_block=(int(block) // SETTLEMENT) * SETTLEMENT,
                                epoch_block_hash="0x" + "ab" * 32,
                                reject_reason="no_king_challenger_pair",
                                reward_uids=(KING_UID,), weights=tuple(vec),
                                validator_hotkey=RECEIPT_VALIDATOR,
                                emission=getattr(r, "_last_emission", None))
    return hashlib.sha256(dump_receipt(rec).encode("utf-8")).hexdigest()
