"""Shared spec and fakes for the W10 acceptance tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6.4):

- W10 row: "Miner CLI `cascade lane-d export | verify | submit | ready |
  status`, **W3a-chosen upload path only**. (rev5, D43) The commit reveals
  immediately (`blocks_until_reveal = min_blocks`), never at boundary −
  margin; a test asserts the lane D commit path never uses the lane A
  boundary timing".
- §1.2 Flow: "1. POST /v1/lane-d/submit [...] Returns upload grants for
  `lane-d/<hotkey>/<digest>/`. 2. The miner's CLI does a direct multipart
  upload. Nothing transits the box. 3. POST /v1/lane-d/ready, signed [...]
  returns `commit_payload`. 4. The miner commits through a timelocked reveal
  [...] (D43) The lane D CLI sets `blocks_until_reveal = min_blocks`".
- §1.2: "Credentials and URLs are never logged, printed or put on a command
  line".

README.md states the contract. The only fakes are external services:

- the chain: ``ChainClient.from_config`` returns :class:`FakeChain` (block
  clock, the wallet's hotkey, ``commit_submission``);
- the intake and the vault: :class:`FakeIntake`, a real HTTP server on
  127.0.0.1 that speaks W3's wire contract (tests/accept/w3/README.md) and
  accepts the grant PUTs;
- the network: every non-loopback connect is refused and recorded.

Keys are deterministic TEST keys (``//w10-...``), never real ones. Nothing
here imports a module in PATHS at import time.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[3]
DECISION = REPO / "decisions" / "STORAGE-LANE-D.md"
CHAIN_TOML = REPO / "chain.toml"
LANE_D_PY = REPO / "cascade" / "miner" / "lane_d.py"

# W3's path ids (tests/accept/w3/README.md "Upload path"); the decision
# record's table must list exactly these (test_w10_decision.py).
PATH_IDS = ("presigned_multipart", "single_presigned_put", "upload_relay")
ALLOWED_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")
SUBCOMMANDS = ("export", "verify", "submit", "ready", "status")
LICENCES = ("Apache-2.0", "OpenMDW-1.1")
CONTEXT = 4096
SUBMIT = "/v1/lane-d/submit"
READY = "/v1/lane-d/ready"
QUEUE = "/v1/lane-d/queue"
FRESHNESS_SECONDS = 300
# Grant secrets the fake mints; none may ever reach stdout, stderr or a log.
GRANT_SECRET = "w10GrantSigR4Tq8Lm"
RELAY_TICKET = "w10RelayTktX7pN2Vb"
PART_SIZE = 1024


# ── the decision record ─────────────────────────────────────────────────────


def decision_path_ids() -> tuple[str, ...]:
    """The path ids of the record's ``| Path id |`` table, in order."""
    ids, in_table = [], False
    for line in DECISION.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.strip().startswith("|") and cells and cells[0] == "Path id":
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                break
            m = re.fullmatch(r"`([a-z_]+)`", cells[0])
            if m:
                ids.append(m.group(1))
    return tuple(ids)


def chosen_upload_path() -> str | None:
    """The one path id on the record's ``Chosen path:`` line, else None
    (unfilled, ambiguous or missing record). Same rule as W3's helper."""
    if not DECISION.is_file():
        return None
    lines = [ln for ln in DECISION.read_text(encoding="utf-8").splitlines()
             if ln.strip().startswith("Chosen path:")]
    if len(lines) != 1:
        return None
    value = lines[0].split(":", 1)[1].strip().strip("`*").strip()
    return value if value in decision_path_ids() else None


def require_chosen_path() -> str:
    path = chosen_upload_path()
    assert path is not None, (
        "decisions/STORAGE-LANE-D.md has no filled 'Chosen path:' (the W3a owner run): "
        "W10 builds only the chosen path, so it is blocked until the record names one")
    return path


# ── keys ────────────────────────────────────────────────────────────────────


def keypair_class():
    try:
        from bittensor.sp_core import Keypair
    except ImportError:
        from bittensor import Keypair
    return Keypair


def fixture_key(name: str):
    """A deterministic TEST keypair (never a real key)."""
    return keypair_class().create_from_uri(f"//w10-{name}")


def key_secrets(key) -> list[str]:
    """Strings that would reveal ``key``'s private half if printed."""
    out = []
    pk = getattr(key, "private_key", None)
    if pk:
        out.append(bytes(pk).hex())
    seed = getattr(key, "seed_hex", None)
    if seed:
        out.append(str(seed).removeprefix("0x"))
    mn = getattr(key, "mnemonic", None)
    if mn:
        out.append(str(mn))
    return [s for s in out if s]


# ── checkpoint fixtures (inert bytes, no torch) ─────────────────────────────


def safetensors_bytes(tensors: dict[str, tuple[str, tuple[int, ...]]], fill: int = 7) -> bytes:
    """A safetensors file written by hand: ``{name: (dtype, shape)}``."""
    width = {"BF16": 2, "F16": 2, "F32": 4}
    header, data, off = {}, b"", 0
    for name in sorted(tensors):
        dt, shape = tensors[name]
        n = width[dt]
        for s in shape:
            n *= s
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        data += bytes([(fill + i) % 251 for i in range(n)])
        off += n
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + data


def read_header(path: Path) -> dict:
    blob = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", blob[:8])
    return json.loads(blob[8:8 + n])


TENSORS = {"head.weight": ("BF16", (8, 16)), "embed.weight": ("BF16", (4, 16))}


def repo_code() -> dict[str, bytes]:
    """The repo's pyramid code bytes a lane D checkpoint must carry (§1.1)."""
    trainer = REPO / "cascade" / "trainer"
    return {"model.py": (trainer / "pyramid_model.py").read_bytes(),
            "forecast_wrapper.py": (trainer / "pyramid_wrapper.py").read_bytes()}


def write_checkpoint(directory: Path, *, tag: str = "x", extra: dict | None = None,
                     drop: tuple[str, ...] = ()) -> Path:
    """A lane D entry directory: the four allowed files (repo code bytes, a
    small config, a BF16 weights file), plus ``extra`` {name: bytes}."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    files = {**repo_code(),
             "config.json": json.dumps({"tag": tag, "moe": {"d_model": 16}}).encode(),
             "weights.safetensors": safetensors_bytes(TENSORS, fill=len(tag))}
    for name, blob in {**files, **(extra or {})}.items():
        if name in drop:
            continue
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
    return d


def manifest_of(directory: Path) -> dict:
    files = {}
    for p in sorted(Path(directory).rglob("*")):
        if p.is_file():
            files[p.relative_to(directory).as_posix()] = {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size": p.stat().st_size}
    return {"files": files}


def canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_of(directory: Path) -> str:
    return hashlib.sha256(canonical(manifest_of(directory))).hexdigest()


# ── the W3 wire contract (signing) ──────────────────────────────────────────


def canonical_message(action: str, hotkey: str, digest: str, context, timestamp,
                      body: bytes) -> bytes:
    """W3 README "Ownership": the signed bytes."""
    return (f"cascade-lane-d-{action}:v1:{hotkey}:{digest}:{context}:{timestamp}:"
            f"{hashlib.sha256(body).hexdigest()}").encode()


def verify_signature(hotkey: str, message: bytes, signature_hex: str) -> bool:
    from cascade.shared.sr25519 import verify_hotkey

    return verify_hotkey(hotkey, message, signature_hex)


# ── the chain (an external service) ─────────────────────────────────────────


class FakeWallet:
    """The ChainClient wallet: its hotkey signs; the coldkey is never touched."""

    def __init__(self, chain: FakeChain):
        self._chain = chain

    @property
    def hotkey(self):
        return self._chain.key

    @property
    def coldkey(self):
        self._chain.coldkey_touched = True
        raise RuntimeError("w10 fake: the lane D CLI must never touch the coldkey")

    @property
    def coldkeypub(self):
        self._chain.coldkey_touched = True
        raise RuntimeError("w10 fake: the lane D CLI must never touch the coldkey")


class FakeChain:
    """What the lane D CLI may use of ``ChainClient``: ``from_config`` (patched
    to return this), ``current_block``, ``wallet().hotkey``, ``hotkey_ss58``
    and ``commit_submission(payload, blocks_until_reveal)``."""

    def __init__(self, key, *, block: int = 1_000_000, fail_commit: bool = False):
        self.key = key
        self.block = int(block)
        self.fail_commit = fail_commit
        self.commits: list[tuple[str, int]] = []
        self.from_config_calls: list[dict] = []
        self.coldkey_touched = False

    def bind(self, cfg, kwargs):
        self.from_config_calls.append(dict(kwargs))
        return self

    def current_block(self) -> int:
        return self.block

    def wallet(self):
        return FakeWallet(self)

    def hotkey_ss58(self) -> str:
        return self.key.ss58_address

    def commit_submission(self, payload: str, blocks_until_reveal: int = 1) -> None:
        from cascade.shared.chain import ChainError

        if self.fail_commit:
            raise ChainError("set_reveal_commitment_rejected: w10 fake chain refused")
        self.commits.append((str(payload), int(blocks_until_reveal)))


def install_chain(monkeypatch, chain: FakeChain) -> FakeChain:
    from cascade.shared.chain import ChainClient

    monkeypatch.setattr(ChainClient, "from_config",
                        classmethod(lambda cls, cfg, **kw: chain.bind(cfg, kw)))
    return chain


# ── the network ─────────────────────────────────────────────────────────────


_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def block_external_network(monkeypatch) -> list:
    """Refuse (and record) every non-loopback connect in this process."""
    attempts: list = []
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            attempts.append(address)
            raise OSError(f"w10 test: connect to {host!r} is blocked")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    return attempts


# ── the intake + vault (an external service) ────────────────────────────────


class FakeIntake:
    """W3's `/v1/lane-d/*` wire contract plus the vault's grant endpoints.

    ``upload_path`` is the path id the submit answers with (the chosen one,
    unless a test makes the intake misbehave). Refusals are injected with
    ``refuse[route] = (status, code)``; ``grants_edit`` rewrites the grants of
    a submit answer; ``fail_put`` makes every grant PUT answer 500."""

    def __init__(self, upload_path: str, *, coldkey: str, ready_block: int = 5000,
                 ttl: int = 7200):
        self.upload_path = upload_path
        self.coldkey = coldkey
        self.ready_block = ready_block
        self.ttl = ttl
        self.refuse: dict[str, tuple[int, str]] = {}
        self.grants_edit = None
        self.payload_edit = None
        self.fail_put = False
        self.requests: list[dict] = []
        self.puts: dict[tuple[str, int], bytes] = {}
        self.entries: dict[str, dict] = {}
        self.signatures: list[str] = []
        self.queue_rows: list[dict] | None = None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    # the grants for each path id: a list of byte-range PUTs per file
    def grants_for(self, manifest: dict) -> dict:
        files = {}
        for name, meta in sorted(manifest["files"].items()):
            size = int(meta["size"])
            if self.upload_path == "presigned_multipart":
                reqs = [{"method": "PUT",
                         "url": f"{self.base}/vault/{name}/{i // PART_SIZE + 1}"
                                f"?uploadId=w10&X-Amz-Signature={GRANT_SECRET}",
                         "offset": i, "length": min(PART_SIZE, size - i), "headers": {}}
                        for i in range(0, size, PART_SIZE)]
            elif self.upload_path == "single_presigned_put":
                reqs = [{"method": "PUT",
                         "url": f"{self.base}/vault/{name}/1?X-Amz-Signature={GRANT_SECRET}",
                         "offset": 0, "length": size, "headers": {}}]
            else:
                reqs = [{"method": "PUT", "url": f"{self.base}/relay/{name}",
                         "offset": 0, "length": size,
                         "headers": {"X-Relay-Ticket": RELAY_TICKET}}]
            files[name] = {"requests": reqs}
        return {"files": files}

    def assembled(self, name: str) -> bytes:
        parts = sorted((off, b) for (n, off), b in self.puts.items() if n == name)
        return b"".join(b for _, b in parts)

    def routes(self, path: str) -> list[dict]:
        return [r for r in self.requests if r["path"] == path]

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep the server quiet
                return

            def _send(self, status: int, doc: dict):
                raw = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _body(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n > 0 else b""

            def _signed(self, action: str, body: bytes):
                h = self.headers
                hk, dg = h.get("X-Miner-Hotkey", ""), h.get("X-Content-Digest", "")
                ctx, ts, sig = h.get("X-Context-Length", ""), h.get("X-Timestamp", ""), \
                    h.get("X-Signature", "")
                fake.signatures.append(sig)
                try:
                    fresh = abs(int(ts) - time.time()) <= FRESHNESS_SECONDS
                except ValueError:
                    fresh = False
                if not fresh:
                    return None, (400, "stale_timestamp")
                msg = canonical_message(action, hk, dg, ctx, ts, body)
                if not verify_signature(hk, msg, sig):
                    return None, (400, "bad_signature")
                return (hk, dg, ctx), None

            def do_GET(self):
                path = urlsplit(self.path).path
                fake.requests.append({"method": "GET", "path": path, "headers": dict(self.headers)})
                if path == QUEUE:
                    if QUEUE in fake.refuse:
                        st, code = fake.refuse[QUEUE]
                        return self._send(st, {"code": code, "message": code})
                    rows = fake.queue_rows if fake.queue_rows is not None else \
                        list(fake.entries.values())
                    return self._send(200, {"entries": rows})
                return self._send(404, {"code": "not_found", "message": path})

            def do_PUT(self):
                parts = urlsplit(self.path)
                body = self._body()
                fake.requests.append({"method": "PUT", "path": parts.path,
                                      "headers": dict(self.headers), "query": parts.query,
                                      "size": len(body)})
                if fake.fail_put:
                    return self._send(500, {"code": "vault_error", "message": "w10 fake"})
                seg = parts.path.strip("/").split("/")
                if seg[0] == "relay" and self.headers.get("X-Relay-Ticket") != RELAY_TICKET:
                    return self._send(403, {"code": "bad_ticket", "message": "no ticket"})
                if seg[0] not in ("vault", "relay") or len(seg) < 2:
                    return self._send(404, {"code": "not_found", "message": parts.path})
                name = seg[1]
                offset = (int(seg[2]) - 1) * PART_SIZE if seg[0] == "vault" and len(seg) > 2 \
                    and fake.upload_path == "presigned_multipart" else 0
                fake.puts[(name, offset)] = body
                self.send_response(200)
                self.send_header("ETag", '"' + hashlib.md5(body).hexdigest() + '"')
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self):
                path = urlsplit(self.path).path
                body = self._body()
                fake.requests.append({"method": "POST", "path": path,
                                      "headers": dict(self.headers), "body": body})
                if path not in (SUBMIT, READY):
                    return self._send(404, {"code": "not_found", "message": path})
                action = "submit" if path == SUBMIT else "ready"
                ident, why = self._signed(action, body)
                if why:
                    return self._send(why[0], {"code": why[1], "message": why[1]})
                if path in fake.refuse:
                    st, code = fake.refuse[path]
                    return self._send(st, {"code": code, "message": f"w10 fake {code}",
                                           "advisory": code == "coldkey_cap_open"})
                hk, dg, ctx = ident
                return self._submit(hk, dg, ctx, body) if action == "submit" else \
                    self._ready(hk, dg, ctx, body)

            def _submit(self, hk, dg, ctx, body):
                try:
                    manifest = json.loads(body)
                except ValueError:
                    return self._send(400, {"code": "bad_manifest", "message": "json"})
                if hashlib.sha256(body).hexdigest() != dg:
                    return self._send(400, {"code": "digest_mismatch", "message": "digest"})
                if set(manifest.get("files", {})) != set(ALLOWED_FILES):
                    return self._send(400, {"code": "file_not_allowed", "message": "files"})
                grants = fake.grants_for(manifest)
                if fake.grants_edit is not None:
                    grants = fake.grants_edit(grants)
                fake.entries[dg] = {"hotkey": hk, "digest": dg, "state": "submitted",
                                    "context": int(ctx), "manifest": manifest,
                                    "coldkey_at_ready": None, "ready_block": None,
                                    "ready_expires_block": None,
                                    "weights_sha256": None, "licence": None,
                                    "licence_attestation_sha256": None}
                return self._send(201, {"status": "granted", "upload_path": fake.upload_path,
                                        "hotkey": hk, "digest": dg, "context": int(ctx),
                                        "prefix": f"lane-d/{hk}/{dg}/", "grants": grants})

            def _ready(self, hk, dg, ctx, body):
                e = fake.entries.get(dg)
                if e is None:
                    return self._send(404, {"code": "unknown_entry", "message": dg})
                try:
                    doc = json.loads(body)
                except ValueError:
                    return self._send(400, {"code": "bad_body", "message": "json"})
                if doc.get("licence") not in LICENCES:
                    return self._send(400, {"code": "licence_not_open", "message": "licence"})
                for name, meta in e["manifest"]["files"].items():
                    blob = fake.assembled(name)
                    if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
                        return self._send(409, {"code": "missing_object", "message": name})
                payload = f"metro-v1:weights:c{ctx}:hippius:lane-d/{hk}/{dg}"
                if fake.payload_edit is not None:
                    payload = fake.payload_edit(payload)
                att = str(doc.get("attestation", ""))
                e.update(state="ready", coldkey_at_ready=fake.coldkey,
                         ready_block=fake.ready_block,
                         ready_expires_block=fake.ready_block + fake.ttl,
                         weights_sha256=e["manifest"]["files"]["weights.safetensors"]["sha256"],
                         licence=doc["licence"],
                         licence_attestation_sha256=hashlib.sha256(att.encode()).hexdigest())
                row = {k: v for k, v in e.items() if k != "manifest"}
                return self._send(200, {"status": "ready", **row, "commit_payload": payload})

        return Handler


# ── running the CLI in-process ──────────────────────────────────────────────


class Result:
    def __init__(self, rc: int, out: str, err: str, logs: str):
        self.rc = rc
        self.out = out
        self.err = err
        self.logs = logs
        lines = [ln for ln in out.strip().splitlines() if ln.strip()]
        self.doc = json.loads(lines[-1]) if lines else None

    @property
    def data(self) -> dict:
        return (self.doc or {}).get("data") or {}

    @property
    def code(self):
        return ((self.doc or {}).get("error") or {}).get("code")

    def text(self) -> str:
        return "\n".join((self.out, self.err, self.logs))

    def __repr__(self):
        return f"Result(rc={self.rc}, code={self.code!r}, err={self.err[-1500:]!r})"


def cli_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CASCADE_NO_DOTENV", "1")
    monkeypatch.delenv("CASCADE_JSON", raising=False)
    monkeypatch.chdir(tmp_path)


def run_cli(capsys, caplog, *argv: str) -> Result:
    """``cascade <argv> --json`` in-process. The W10 module is imported first:
    on main that is the right-reason failure (``cascade.miner.lane_d``)."""
    import logging

    import cascade.miner.lane_d  # noqa: F401  (PATHS)

    from cascade.miner.cli import main

    caplog.set_level(logging.DEBUG)
    capsys.readouterr()
    rc = main([*argv, "--json"])
    out, err = capsys.readouterr()
    logs = "\n".join(r.getMessage() for r in caplog.records)
    return Result(int(rc), out, err, logs)


def wallet_args(name: str = "w10") -> list[str]:
    return ["--wallet-name", name, "--wallet-hotkey", "lane-d", "--chain-toml", str(CHAIN_TOML)]


def assert_no_secrets(res: Result, *, key=None, extra: tuple[str, ...] = ()) -> None:
    text = res.text()
    for s in (GRANT_SECRET, RELAY_TICKET, "X-Amz-Signature", *extra):
        assert s not in text, f"a grant secret ({s[:8]}...) reached stdout/stderr/logs"
    if key is not None:
        for s in key_secrets(key):
            assert s not in text, "the hotkey's private material was printed"


@contextlib.contextmanager
def intake(upload_path: str, *, coldkey: str, **kw):
    with FakeIntake(upload_path, coldkey=coldkey, **kw) as f:
        yield f
