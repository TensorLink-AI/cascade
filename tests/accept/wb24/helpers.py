"""Shared spec and fixtures for the WB24 acceptance tests (lane B miner CLI).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB24 row (§3.0
lane B track, rev6.3) and the rows it builds on (WB13 contamination scan,
WB14 declared rows, WB15 / WB17 / WB18 status reasons, WB23 train hours).
The contract is in README.md.

Nothing from a PATHS module is imported at module level, so on main every
failure is an assertion inside a test body.

No network: every test runs under ``network_guard`` (a socket-level guard
that refuses any connection except to the loopback fake servers the test
started). The fake intake is either a recorder (``FakeIntake``: records
each request exactly as received, answers canned JSON) or the real
``FundingIntake`` served on loopback (``real_intake``), so the CLI's
request is checked by the intake's own verifier. The public status
document is served by the same recorder at
``<s3_endpoint>/<manifest_bucket>/status/lane_b.json`` of a fixture
chain.toml (``[storage]`` points at the recorder).

Miner repos are untrusted. The fixture repos are plain files in
``tmp_path``; the tests never import or run them. ``exec_trap_repo``
writes modules whose top level leaves a marker file and raises, so a
``verify-repo`` that imports or runs the tree is caught.

Keys: the wallet is a stand-in (``FakeWallet``, deterministic signer); no
keyfile is read. The provider key is the fake ``FAKE_KEY``; tests scan the
CLI's stdout and stderr for it.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import math
import random
import re
import socket
import sys
import threading
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
CHAIN_TOML = REPO / "chain.toml"
PYRAMID = REPO / "cascade" / "trainer" / "pyramid_model.py"
EXPERT_LANE = REPO / "cascade" / "trainer" / "expert_lane.py"
ROW_CLAIMS = REPO / "cascade" / "trainer" / "row_claims.py"
DRIVER = REPO / "cascade" / "trainer" / "lane_b_driver.py"
LANE_B_CLI = REPO / "cascade" / "miner" / "lane_b.py"

HEADER_ROWS = "X-Lane-B-Declared-Rows"
HEADER_TARGETS = "X-Lane-B-Targets"
STATUS_KEY = "status/lane_b.json"
BUCKET = "wb24-manifests"

# WB14 DECLARED_ROWS_CODES (cascade/trainer/declared_rows.py)
CODES = ("declared_rows_out_of_range", "declared_rows_bad_layer",
         "declared_rows_count", "declared_rows_malformed")
# WB15 / WB18 deferral reasons (cascade/trainer/row_claims.py)
ROWS_HELD = "rows_held"
SENTINEL_TARGET = "sentinel_target"
# WB17 gate_closed_reason values (cascade/trainer/expert_lane.py GATE_CLOSED_REASONS)
GATE_REASONS = ("tokens_short", "king_check_not_passed", "king_check_unavailable")
# WB13 intake contamination refusals (cascade/funding/intake.py)
CONTAMINATION_REFUSALS = (
    (400, "benchmark_series_embedded"),
    (413, "contamination_budget_exceeded"),
    (413, "repo_too_large"),
    (503, "contamination_hashset_unavailable"),
    (503, "repo_unavailable"),
    (503, "contamination_scan_failed"),
)

# exit codes (cascade/miner/output.py; part of the CLI contract)
OK, REJECTED, USAGE, NETWORK = 0, 1, 2, 3

HK = "5FakeHotkeyWB24AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "vault/direct@sha256:" + "d" * 64
KEY_ENV = "WB24_LIUM_KEY"
FAKE_KEY = "sk-wb24-FAKE-provider-key-0d1e2f3a4b5c6d7e"
FIXED_TIME = 1_763_000_000.0
LABEL = "wb24-miner"


# ── chain.toml fixtures (the shipped file, edited in [lane_b] / [storage]) ──


def _section_span(text: str, name: str) -> tuple[int, int]:
    m = re.search(rf"^\[{re.escape(name)}\][^\n]*\n", text, re.M)
    assert m, f"chain.toml has no [{name}] table"
    nxt = re.search(r"^\[", text[m.end():], re.M)
    return m.end(), (m.end() + nxt.start()) if nxt else len(text)


def _set_key(text: str, section: str, key: str, value: str) -> str:
    lo, hi = _section_span(text, section)
    body = text[lo:hi]
    pat = re.compile(rf"^{re.escape(key)}\s*=.*$", re.M)
    if pat.search(body):
        body = pat.sub(lambda _m: f"{key} = {value}", body, count=1)
    else:
        body = f"{key} = {value}\n" + body
    return text[:lo] + body + text[hi:]


def write_toml(path: Path, *, pin: str = "", suite_targets: bool = False,
               min_frac: float | None = None, max_frac: float | None = None,
               min_hits: dict | None = None, s3_endpoint: str | None = None) -> Path:
    """The shipped chain.toml with ``[lane_b]`` / ``[storage]`` edits."""
    text = CHAIN_TOML.read_text(encoding="utf-8")
    if pin:
        text = _set_key(text, "lane_b", "contamination_hashset", json.dumps(pin))
    if suite_targets:
        text = _set_key(text, "lane_b", "suite_targets", "true")
    if min_frac is not None:
        text = _set_key(text, "lane_b", "min_experts_frac", repr(float(min_frac)))
    if max_frac is not None:
        text = _set_key(text, "lane_b", "max_experts_frac", repr(float(max_frac)))
    if min_hits is not None:
        inline = ", ".join(f"{k} = {int(v)}" for k, v in min_hits.items())
        text = _set_key(text, "lane_b", "contamination_min_hits", "{ " + inline + " }")
    if s3_endpoint is not None:
        text = _set_key(text, "storage", "s3_endpoint", json.dumps(s3_endpoint))
        text = _set_key(text, "storage", "manifest_bucket", json.dumps(BUCKET))
    path.write_text(text, encoding="utf-8")
    return path


def load_cfg(path: Path):
    from cascade.shared.config import load_chain_config

    return load_chain_config(path)


# ── model geometry (pyramid_model.py read by AST; it imports torch) ─────────


def experts_per_layer() -> tuple[int, ...]:
    tree = ast.parse(PYRAMID.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CascadeMoEConfig":
            for st in node.body:
                if (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name)
                        and st.target.id == "experts_per_layer" and st.value is not None):
                    return tuple(int(e) for e in ast.literal_eval(st.value))
    raise AssertionError("CascadeMoEConfig.experts_per_layer not found in pyramid_model.py")


def clamp(n_experts: int, min_frac: float, max_frac: float) -> tuple[int, int]:
    return math.ceil(min_frac * n_experts), math.floor(max_frac * n_experts)


def parse_rows(raw, lane_b) -> dict:
    """The intake's own parse (WB14 ``parse_declared_rows``) under ``lane_b``."""
    from cascade.trainer.declared_rows import model_experts_per_layer, parse_declared_rows

    return parse_declared_rows(raw, experts_per_layer=model_experts_per_layer(),
                               min_frac=float(lane_b.min_experts_frac),
                               max_frac=float(lane_b.max_experts_frac))


def parse_code(raw, lane_b) -> str:
    """The intake parser's refusal code for ``raw`` (AssertionError if accepted)."""
    from cascade.trainer.declared_rows import DeclaredRowsError

    try:
        got = parse_rows(raw, lane_b)
    except DeclaredRowsError as e:
        return e.code
    raise AssertionError(f"fixture error: {raw!r} was accepted as {got!r}")


def string_constants(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str)}


# ── wallet stand-in and signer ──────────────────────────────────────────────


def sign(msg: bytes) -> bytes:
    """Deterministic stand-in for the hotkey's sr25519 signer."""
    return hashlib.sha256(HK.encode() + msg).digest()


def verify(hotkey: str, msg: bytes, sig_hex: str) -> bool:
    return sig_hex == hashlib.sha256(hotkey.encode() + msg).hexdigest()


class FakeWallet:
    def __init__(self, signed: list) -> None:
        def _sign(msg: bytes) -> bytes:
            signed.append(bytes(msg))
            return sign(msg)

        self.hotkey = SimpleNamespace(ss58_address=HK, sign=_sign)


# ── network guard ───────────────────────────────────────────────────────────


class NetworkGuard:
    """Refuses every socket connection except to the allowed loopback ports;
    records each attempt."""

    def __init__(self) -> None:
        self.allowed: set[int] = set()
        self.attempts: list = []
        self.refused: list = []

    def allow(self, port: int) -> None:
        self.allowed.add(int(port))


@contextmanager
def guarded_sockets(guard: NetworkGuard):
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def check(addr):
        guard.attempts.append(addr)
        host, port = (addr[0], addr[1]) if isinstance(addr, tuple) and len(addr) >= 2 else (addr, None)
        if host in ("127.0.0.1", "::1", "localhost") and port in guard.allowed:
            return True
        guard.refused.append(addr)
        return False

    def connect(self, addr):
        if not check(addr):
            raise OSError(f"WB24 network guard: connection to {addr!r} refused (no network)")
        return real_connect(self, addr)

    def connect_ex(self, addr):
        if not check(addr):
            raise OSError(f"WB24 network guard: connection to {addr!r} refused (no network)")
        return real_connect_ex(self, addr)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    try:
        yield guard
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex


# ── the recorder intake + public bucket ─────────────────────────────────────

_TRANSPORT = {"host", "user-agent", "accept-encoding", "connection", "content-length"}


class Recorded(SimpleNamespace):
    """One request as the server received it."""

    def header(self, name: str):
        for k, v in self.headers:
            if k.lower() == name.lower():
                return v
        return None

    def identity_headers(self) -> list[tuple[str, str]]:
        """Every header but the transport ones, sorted by lower-cased name."""
        return sorted(((k.lower(), v) for k, v in self.headers
                       if k.lower() not in _TRANSPORT))

    def canonical(self) -> bytes:
        """The request bytes the golden digests: method, path, body, and the
        identity headers (transport headers excluded)."""
        doc = {"method": self.method, "path": self.path,
               "headers": self.identity_headers(), "body": self.body.hex()}
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class FakeIntake:
    """A loopback HTTP server: records every request; answers ``routes``
    (``(method, path) -> (status, json body)``). The public bucket path
    ``/<BUCKET>/status/lane_b.json`` is one of the routes when a status
    document is set. Anything else is 404."""

    def __init__(self) -> None:
        self.requests: list[Recorded] = []
        self.routes: dict[tuple[str, str], tuple[int, dict]] = {}
        self.real = None          # optional real FundingIntake for POST /v1/lane-b/*
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                return

            def _handle(self, method: str) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                rec = Recorded(method=method, path=self.path,
                               headers=list(self.headers.items()), body=body,
                               message=self.headers)
                owner.requests.append(rec)
                status, doc = owner._answer(method, self.path, self.headers)
                data = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                self._handle("GET")

            def do_POST(self):  # noqa: N802
                self._handle("POST")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()

    def _answer(self, method, path, headers):
        if self.real is not None and method == "POST" and path == "/v1/lane-b/fund":
            return self.real.lane_b_fund(headers)
        if self.real is not None and method == "POST" and path == "/v1/lane-b/withdraw":
            return self.real.lane_b_withdraw(headers)
        if self.real is not None and method == "GET" and path == "/v1/lane-b/queue":
            return self.real.lane_b_queue_view()
        return self.routes.get((method, path), (404, {"code": "not_found", "message": path}))

    def set_status_doc(self, doc: dict) -> None:
        self.routes[("GET", f"/{BUCKET}/{STATUS_KEY}")] = (200, doc)

    def set_queue(self, entries: list, as_of: str | None = None) -> None:
        self.routes[("GET", "/v1/lane-b/queue")] = (200, {
            "entries": entries, "queued_depth": sum(1 for e in entries
                                                    if e.get("status") == "queued"),
            "as_of": as_of or now_iso()})

    def posts(self) -> list[Recorded]:
        return [r for r in self.requests if r.method == "POST"]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def real_intake(tmp_path: Path, lane_b, *, clock=None):
    """A real ``FundingIntake`` with a real lane B queue on disk, signature
    required. Only the reveal oracle and the signature verifier are
    stand-ins. ``lane_b`` is the loaded ``[lane_b]`` config whose slots,
    per-leg cap and row clamps it applies."""
    import time

    from cascade.funding.intake import FundingIntake
    from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
    from cascade.funding.vault import PayerKeyVault

    clk = clock or time.time
    lane_a = FundedQueue(tmp_path / "queue.json", clock=clk)
    lane_b_q = LaneBQueue(lane_b_queue_path(tmp_path / "queue.json"), clock=clk)
    it = FundingIntake(
        lane_a, PayerKeyVault(dir=None, clock=clk),
        resolve_reveal=lambda hk, ref: None, require_signature=True, verify=verify,
        clock=clk, lane_b_queue=lane_b_q, resolve_lane_b_reveal=lambda hk, t, r: 700,
        lane_b_targets=tuple(lane_b.targets),
        lane_b_max_targets=int(lane_b.max_domains_per_leg), lane_b_cfg=lane_b)
    return it, lane_b_q


# ── CLI runner ──────────────────────────────────────────────────────────────


class Run(SimpleNamespace):
    def doc(self) -> dict:
        lines = [ln for ln in self.out.splitlines() if ln.strip()]
        assert lines, f"--json printed nothing on stdout (rc {self.rc}); stderr: {self.err[-2000:]}"
        try:
            doc = json.loads(lines[-1])
        except ValueError as e:
            raise AssertionError(f"--json stdout is not one JSON document: {e}: "
                                 f"{self.out[-2000:]}") from None
        assert isinstance(doc, dict), f"--json document is {type(doc).__name__}"
        return doc

    def code(self) -> str | None:
        err = self.doc().get("error")
        return err.get("code") if isinstance(err, dict) else None

    def text(self) -> str:
        return self.out + "\n" + self.err


def run_cli(capsys, argv: list[str]) -> Run:
    """``cascade <argv>`` in process. Every exception becomes an assertion
    failure (a missing flag or subcommand is a SystemExit from argparse)."""
    from cascade.miner.cli import main

    capsys.readouterr()
    try:
        rc = main(list(argv))
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (0 if e.code is None else USAGE)
    except Exception as e:  # noqa: BLE001 — surfaced as an assertion
        cap = capsys.readouterr()
        raise AssertionError(f"`cascade {' '.join(argv)}` raised {type(e).__name__}: {e}\n"
                             f"stdout: {cap.out[-2000:]}\nstderr: {cap.err[-2000:]}") from None
    cap = capsys.readouterr()
    return Run(rc=rc, out=cap.out, err=cap.err, argv=list(argv))


def fund_argv(base: str, toml: Path, *extra: str, targets: str | None = "energy",
              json_mode: bool = True) -> list[str]:
    argv = ["lane-b", "fund", base]
    if targets is not None:
        argv += ["--targets", targets]
    argv += ["--ref", REF, "--yes", "--lium-key-env", KEY_ENV, "--label", LABEL,
             "--chain-toml", str(toml), "--wallet-name", "wb24w", "--wallet-hotkey", "wb24h",
             *extra]
    if json_mode:
        argv.append("--json")
    return argv


def walk(obj):
    """Every dict, list and scalar inside ``obj`` (depth first)."""
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list | tuple):
        for v in obj:
            yield from walk(v)


def dicts_in(obj) -> list[dict]:
    return [o for o in walk(obj) if isinstance(o, dict)]


# ── WB11 / WB13 contamination fixtures (tiny, seeded, in memory) ────────────


def _walk_series(rng: random.Random, n: int, level: float, scale: float) -> list[float]:
    out, x = [], level
    for _ in range(n):
        x += rng.gauss(0.0, scale)
        out.append(round(x + rng.gauss(0.0, scale * 0.5), 6))
    return out


GIFT_ITEM = "gift/electricity/7"


def fixture_suites() -> dict:
    rng = random.Random("cascade wb24 fixture suites")
    return {
        "gift": {"version": "a" * 40, "series": {GIFT_ITEM: _walk_series(rng, 72, 5000.0, 40.0)}},
        "boom": {"version": "b" * 40, "series": {"boom/svc/cpu": _walk_series(rng, 48, 0.4, 0.02)}},
        "time": {"version": "c" * 40, "series": {"time/traffic/3": _walk_series(rng, 56, 300.0, 9.0)}},
        "fev": {"version": "e" * 40, "series": {"fev/solar/2": _walk_series(rng, 44, 7.0, 0.3)}},
    }


def gift_window() -> list[float]:
    s = fixture_suites()["gift"]["series"][GIFT_ITEM]
    return list(s[10:58])


def write_hashset(root: Path, version: int = 1) -> tuple[dict, str]:
    """Build the WB11 artefact from the fixture suites and write it as
    ``<root>/benchmarks/contamination/hashset-v<version>.json``; return
    ``(artefact, pin)``."""
    from cascade.eval import contamination as wb11

    art = wb11.build_hashset(fixture_suites(), version=version)
    pin = wb11.write_artefact(art, root / "benchmarks" / "contamination"
                              / f"hashset-v{version}.json")
    return art, pin


CLEAN_GENERATOR = '''\
"""A clean code-only generator."""
import math
import random


def generate(seed, n_series):
    rng = random.Random(seed)
    return [[math.sin(t / 7.0) + rng.gauss(0.0, 1.0) for t in range(64)]
            for _ in range(n_series)]
'''


def write_repo(root: Path, files: dict[str, str | bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data.encode() if isinstance(data, str) else data)
    return root


def clean_repo(root: Path) -> Path:
    return write_repo(root, {"generator.py": CLEAN_GENERATOR,
                             "config.json": json.dumps({"name": "walks", "noise": 1.0}),
                             "README.md": "A clean generator.\n"})


def embedded_repo(root: Path) -> Path:
    window = gift_window()
    body = "TABLE = [\n" + "".join(f"    {v!r},\n" for v in window) + "]\n"
    return write_repo(root, {"generator.py": CLEAN_GENERATOR + "\n" + body,
                             "README.md": "Embeds a benchmark window.\n"})


def budget_repo(root: Path) -> Path:
    """A tree whose scan stops on a memory cap: an archive with more members
    than ``MAX_ARCHIVE_MEMBERS`` (1024) -> ``budget_exceeded``."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for i in range(1100):
            z.writestr(f"m{i:04d}.txt", b"")
    return write_repo(root, {"generator.py": CLEAN_GENERATOR, "data.npz": buf.getvalue()})


EXEC_MODULES = ("generator.py", "__init__.py", "setup.py", "conftest.py",
                "sitecustomize.py", "usercustomize.py", "__main__.py")


def exec_trap_repo(root: Path, marker: Path) -> Path:
    """Every module's top level writes ``marker`` and raises: importing or
    running any file of the tree is detectable."""
    trap = ("import pathlib\n"
            f"pathlib.Path({str(marker)!r}).write_text('executed ' + __name__)\n"
            "raise RuntimeError('WB24: the submitted repo was executed')\n")
    files: dict[str, str | bytes] = {name: trap + CLEAN_GENERATOR for name in EXEC_MODULES}
    files["pkg/__init__.py"] = trap
    files["pkg/data.pkl"] = b"\x80\x04cbuiltins\nexec\n."
    files["config.json"] = json.dumps({"name": "trap"})
    return write_repo(root, files)


class _TorchBlocker:
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(
                f"WB24: '{name}' must not be imported (the miner CLI is torch-free)", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


# ── pytest fixtures (imported by name into each test module; --noconftest) ──

import pytest  # noqa: E402


@pytest.fixture
def net():
    """No network: only loopback ports a test allows."""
    guard = NetworkGuard()
    with guarded_sockets(guard), torch_blocked():
        yield guard


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """The miner's environment: the fake provider key in ``KEY_ENV``, no
    ``.env`` loading, a stand-in wallet (records what it signs), the default
    chain.toml redirected to a fixture copy (``tmp_path/default-chain.toml``)."""
    import cascade.shared.chain as chain_mod
    import cascade.shared.config.loader as loader

    monkeypatch.setenv("CASCADE_NO_DOTENV", "1")
    monkeypatch.delenv("CASCADE_JSON", raising=False)
    monkeypatch.setenv(KEY_ENV, FAKE_KEY)
    signed: list = []
    opened: list = []

    def open_wallet(name, hotkey, path=None):
        opened.append((name, hotkey, path))
        return FakeWallet(signed)

    monkeypatch.setattr(chain_mod, "open_wallet", open_wallet)
    default = write_toml(tmp_path / "default-chain.toml")
    monkeypatch.setattr(loader, "DEFAULT_CHAIN_TOML", default)
    return SimpleNamespace(signed=signed, opened=opened, default_toml=default,
                           monkeypatch=monkeypatch)


@pytest.fixture
def intake(net):
    srv = FakeIntake()
    net.allow(srv.port)
    try:
        yield srv
    finally:
        srv.close()


def repo_modules_loaded(root: Path) -> list[str]:
    """Modules in ``sys.modules`` whose file lies inside ``root``."""
    out = []
    r = str(root.resolve())
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if f and str(Path(f).resolve()).startswith(r):
            out.append(name)
    return out
