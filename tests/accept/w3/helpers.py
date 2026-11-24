"""Shared fixtures for the W3 acceptance tests (contract: README.md).

Nothing here imports a PATHS module at module level: every ``cascade.lane_d``
import is inside a function a test calls, so on main the failures are a
missing ``cascade.lane_d.service`` / ``cascade.lane_d.main`` (both in PATHS) or
an ``AssertionError``. The only fakes are the external services: the chain
reads (:class:`LaneDReads`), the Hippius grant minter (:class:`GrantMinter`)
and a failing storage wrapper. The vault is the real fs adapter.
Checkpoint "files" are inert bytes; nothing uploaded is ever executed.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DECISION = REPO / "decisions" / "STORAGE-LANE-D.md"
UNIT = REPO / "deploy" / "cascade-laned-intake.service"
CADDY = REPO / "deploy" / "caddy" / "lane-d-intake.caddy"
TAGS_FILE = Path(__file__).resolve().parent / "fixtures" / "lane_d_intake_tags.py"

PATH_IDS = ("presigned_multipart", "single_presigned_put", "upload_relay")
ALLOWED_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")
NOW = 1_800_000_000.0
BLOCK = 1000
TTL = 100
CONTEXT = 4096
# A grant secret the fake minter embeds; it must never reach a log, a state
# file or the queue view.
GRANT_SECRET = "w3GrantSigQ9ZkV7"

SUBMIT = "/v1/lane-d/submit"
READY = "/v1/lane-d/ready"
QUEUE = "/v1/lane-d/queue"


# ── the decision record ─────────────────────────────────────────────────────


def chosen_upload_path() -> str | None:
    """The one path id on the record's ``Chosen path:`` line, else None
    (unfilled, ambiguous or missing record)."""
    if not DECISION.is_file():
        return None
    lines = [ln for ln in DECISION.read_text(encoding="utf-8").splitlines()
             if ln.strip().startswith("Chosen path:")]
    if len(lines) != 1:
        return None
    value = lines[0].split(":", 1)[1].strip().strip("`*").strip()
    return value if value in PATH_IDS else None


# ── keys and signing (deterministic TEST keys, never real ones) ─────────────


def keypair_class():
    from bittensor.sp_core import Keypair

    return Keypair


_KEYS: dict[str, object] = {}


def key(name: str):
    if name not in _KEYS:
        _KEYS[name] = keypair_class().create_from_uri(f"//w3-{name}")
    return _KEYS[name]


def ss58(name: str) -> str:
    return key(name).ss58_address


def message(action: str, hotkey: str, digest: str, context: int, timestamp: str,
            body: bytes) -> bytes:
    """The pinned canonical request (README "Ownership"), built here
    independently of the producer."""
    return (f"cascade-lane-d-{action}:v1:{hotkey}:{digest}:{context}:{timestamp}:"
            f"{hashlib.sha256(body).hexdigest()}").encode()


# ── checkpoints ──────────────────────────────────────────────────────────────


def checkpoint(tag: str, *, weights: bytes | None = None, extra: dict | None = None,
               drop: tuple[str, ...] = ()) -> dict[str, bytes]:
    """Four inert files; ``weights`` fixes the weights bytes (dedup cases).
    ``config.json`` is an allowed lane D config (the whole file is pinned at
    ready: cascade.lane_d.entry_config) whose ``arch`` carries the tag."""
    files = {
        "config.json": json.dumps(entry_config(arch=f"tag-{tag}"), sort_keys=True).encode(),
        "forecast_wrapper.py": f"# wrapper {tag}\n".encode(),
        "model.py": f"# model {tag}\n".encode(),
        "weights.safetensors": weights if weights is not None else f"W3W{tag}".encode() * 8,
    }
    for name in drop:
        files.pop(name)
    files.update(extra or {})
    return files


PYRAMID_MODEL = REPO / "cascade" / "trainer" / "pyramid_model.py"
PYRAMID_TRAINER = REPO / "cascade" / "trainer" / "pyramid_trainer.py"


def pyramid_config_defaults() -> dict:
    """``CascadeMoEConfig().to_dict()`` read from the source by AST (no torch):
    every field's default, tuples as lists, the two module constants inlined."""
    tree = ast.parse(PYRAMID_MODEL.read_text())
    consts = {"NUM_QUANTILES": 21, "COPULA_RANK": 1}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CascadeMoEConfig":
            out = {}
            for s in node.body:
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name):
                    v = s.value
                    val = consts[v.id] if isinstance(v, ast.Name) else ast.literal_eval(v)
                    out[s.target.id] = list(val) if isinstance(val, tuple) else val
            return out
    raise AssertionError("no CascadeMoEConfig in pyramid_model.py")


def entry_config(**top) -> dict:
    """A lane D ``config.json`` as pyramid_trainer.save_checkpoint writes it."""
    doc = {"moe": pyramid_config_defaults(),
           "quantile_levels": [0.01, *(i / 20 for i in range(1, 20)), 0.99],
           "base_arch": "cascade_pyramid", "arch": "cascade-moe-17m",
           "input_transform": "none"}
    doc.update(top)
    return doc


def manifest(files: dict[str, bytes]) -> dict:
    return {"files": {n: {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b)}
                      for n, b in files.items()}}


def canonical(man: dict) -> bytes:
    return json.dumps(man, sort_keys=True, separators=(",", ":")).encode()


def digest_of(man: dict) -> str:
    return hashlib.sha256(canonical(man)).hexdigest()


def weights_sha(files: dict[str, bytes]) -> str:
    return hashlib.sha256(files["weights.safetensors"]).hexdigest()


# ── configs ──────────────────────────────────────────────────────────────────


def lane_d_cfg(**over):
    from cascade.shared.config.lane_d import LaneDConfig

    base = LaneDConfig()
    return replace(base, **{"from_block": 1, "ready_ttl_blocks": TTL,
                            "max_open_per_coldkey": 2, **over})


def ops_cfg(**over):
    from cascade.shared.config.lane_d_ops import LaneDOpsConfig

    return replace(LaneDOpsConfig(), **over)


# ── external-service fakes (built on the producer's ABCs) ───────────────────


class Clock:
    def __init__(self, t: float = NOW):
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


def make_reads(**kw):
    """A :class:`cascade.lane_d.service.LaneDReads` over in-memory chain state."""
    from cascade.lane_d.service import LaneDReads

    class FakeReads(LaneDReads):
        def __init__(self):
            self.block = kw.get("block", BLOCK)
            self.coldkeys: dict[str, str] = dict(kw.get("coldkeys", {}))
            self.reveals: dict[tuple[str, str], int] = {}
            self.published: set[str] = set(kw.get("published", ()))
            self.fail: set[str] = set()
            self.calls: list[str] = []

        def _maybe_fail(self, name):
            self.calls.append(name)
            if name in self.fail:
                raise ConnectionError(f"w3 fake chain: {name} unreachable")

        def current_block(self) -> int:
            self._maybe_fail("current_block")
            return self.block

        def coldkey_of(self, hotkey: str):
            self._maybe_fail("coldkey_of")
            return self.coldkeys.get(hotkey)

        def reveal_block(self, hotkey: str, digest: str):
            self._maybe_fail("reveal_block")
            return self.reveals.get((hotkey, digest))

        def published_weights(self):
            self._maybe_fail("published_weights")
            return frozenset(self.published)

    return FakeReads()


def make_minter(upload_path: str | None = None):
    """A :class:`cascade.lane_d.service.GrantMinter` standing in for Hippius."""
    from cascade.lane_d import service

    path = upload_path or service.UPLOAD_PATH

    class FakeMinter(service.GrantMinter):
        def __init__(self):
            self.minted: list[tuple[str, dict]] = []
            self.finalized: list[tuple[str, dict]] = []
            self.fail = False

        def mint(self, prefix, files):
            if self.fail:
                raise ConnectionError("w3 fake minter: Hippius unreachable")
            self.minted.append((prefix, dict(files)))
            return {"files": {n: {"url": f"https://vault.w3.invalid/{prefix}{n}"
                                         f"?X-Amz-Signature={GRANT_SECRET}"}
                              for n in sorted(files)}}

        def finalize(self, prefix, files):
            if self.fail:
                raise ConnectionError("w3 fake minter: Hippius unreachable")
            self.finalized.append((prefix, dict(files)))

    FakeMinter.upload_path = path
    return FakeMinter()


def failing_storage(inner, ops: tuple[str, ...] = ("put", "get", "list", "size")):
    from cascade.lane_d.storage import LaneDStorage

    class Failing(LaneDStorage):
        def __init__(self):
            self.failing = False

        def _check(self, op):
            if self.failing and op in ops:
                raise OSError(f"w3 fake storage: {op} unreachable")

        def put(self, key, data):
            self._check("put")
            return inner.put(key, data)

        def get(self, key):
            self._check("get")
            return inner.get(key)

        def list(self, prefix):
            self._check("list")
            return inner.list(prefix)

        def size(self, key):
            self._check("size")
            return inner.size(key)

    return Failing()


# ── the WSGI driver ──────────────────────────────────────────────────────────


class Response:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = status
        self.headers = headers
        self.raw = body
        try:
            self.json = json.loads(body.decode("utf-8")) if body else None
        except ValueError:
            self.json = None

    @property
    def code(self):
        return (self.json or {}).get("code")

    def __repr__(self):
        return f"<{self.status} {self.raw[:300]!r}>"


class NoRead(io.RawIOBase):
    """A request stream that fails the test if the app reads it."""

    def readable(self):
        return True

    def read(self, n=-1):
        raise AssertionError("the app read a body it should have refused unread")

    def readline(self, n=-1):
        raise AssertionError("the app read a body it should have refused unread")


def call(app, method: str, path: str, *, headers: dict | None = None, body: bytes = b"",
         ip: str = "203.0.113.7", remote: str | None = None, stream=None,
         content_length: int | None = None) -> Response:
    """Drive ``app`` as a WSGI callable; ``remote`` is REMOTE_ADDR (default:
    127.0.0.1 with ``ip`` as the proxy's X-Forwarded-For, as behind Caddy)."""
    from wsgiref.util import setup_testing_defaults

    env: dict = {}
    setup_testing_defaults(env)
    env["REQUEST_METHOD"] = method
    env["PATH_INFO"] = path
    env["QUERY_STRING"] = ""
    env["wsgi.input"] = stream if stream is not None else io.BytesIO(body)
    env["CONTENT_LENGTH"] = str(len(body) if content_length is None else content_length)
    if remote is None:
        env["REMOTE_ADDR"] = "127.0.0.1"
        env["HTTP_X_FORWARDED_FOR"] = ip
    else:
        env["REMOTE_ADDR"] = remote
    for k, v in (headers or {}).items():
        env["HTTP_" + k.upper().replace("-", "_")] = v
    got: dict = {}

    def start_response(status, hdrs, exc_info=None):
        got["status"] = int(status.split()[0])
        got["headers"] = {k.lower(): v for k, v in hdrs}
        return lambda _data: None

    chunks = app(env, start_response)
    try:
        data = b"".join(chunks)
    finally:
        close = getattr(chunks, "close", None)
        if close:
            close()
    resp = Response(got["status"], got["headers"], data)
    if resp.status >= 400:
        from cascade.lane_d import service

        assert resp.code in service.REFUSAL_CODES, (
            f"refusal code {resp.code!r} is not in service.REFUSAL_CODES: {resp!r}")
    return resp


class Miner:
    """One hotkey's client: signs each request with a fresh, strictly
    increasing timestamp (per action), as the W10 CLI will."""

    def __init__(self, env: Env, name: str, coldkey: str | None = "ck-a", ip: str | None = None):
        self.env = env
        self.name = name
        self.hotkey = ss58(name)
        self.ip = ip or f"198.51.100.{(sum(map(ord, name)) % 200) + 1}"
        self._ts = int(env.clock.t) - 250
        if coldkey is not None:
            env.reads.coldkeys[self.hotkey] = ss58(coldkey)

    def next_ts(self) -> str:
        self._ts += 1
        return str(self._ts)

    def headers(self, action: str, digest: str, body: bytes, *, context: int = CONTEXT,
                ts: str | None = None, signer: str | None = None, hotkey: str | None = None,
                extra: dict | None = None) -> dict:
        ts = self.next_ts() if ts is None else ts
        hk = hotkey or self.hotkey
        sig = key(signer or self.name).sign(message(action, hk, digest, context, ts, body)).hex()
        h = {"X-Miner-Hotkey": hk, "X-Content-Digest": digest,
             "X-Context-Length": str(context), "X-Timestamp": ts, "X-Signature": sig}
        h.update(extra or {})
        return h

    def submit(self, files: dict[str, bytes], *, context: int = CONTEXT, **kw) -> Response:
        man = manifest(files)
        body = canonical(man)
        dig = digest_of(man)
        return call(self.env.app, "POST", SUBMIT, ip=self.ip,
                    headers=self.headers("submit", dig, body, context=context, **kw), body=body)

    def upload(self, files: dict[str, bytes], prefix: str | None = None) -> None:
        """The miner's direct upload (simulated straight into the fs vault)."""
        dig = digest_of(manifest(files))
        prefix = prefix or f"lane-d/{self.hotkey}/{dig}/"
        for n, b in files.items():
            self.env.store.put(prefix + n, b)

    def ready(self, files: dict[str, bytes], *, licence: str = "Apache-2.0",
              attestation: str | None = None, context: int = CONTEXT,
              body: bytes | None = None, **kw) -> Response:
        from cascade.lane_d import service

        dig = digest_of(manifest(files))
        if body is None:
            body = json.dumps({"licence": licence,
                               "attestation": (service.LICENCE_ATTESTATION
                                               if attestation is None else attestation)}).encode()
        return call(self.env.app, "POST", READY, ip=self.ip,
                    headers=self.headers("ready", dig, body, context=context, **kw), body=body)

    def enter(self, files: dict[str, bytes], **kw) -> Response:
        """submit → upload → ready; asserts the first two succeed."""
        r = self.submit(files)
        assert r.status == 201, r
        self.upload(files)
        return self.ready(files, **kw)


class Env:
    """One intake over a tmp fs vault, fake chain reads and a fake minter."""

    def __init__(self, tmp: Path, *, cfg=None, ops=None, reads=None, minter=None,
                 storage=None, per_ip_rpm: int = 10_000, per_hotkey_rpm: int = 10_000,
                 clock: Clock | None = None):
        from cascade.lane_d.storage import FilesystemStorage

        self.tmp = Path(tmp)
        self.state_dir = self.tmp / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.store = storage or FilesystemStorage(self.tmp / "vault")
        self.reads = reads or make_reads()
        self.minter = minter or make_minter()
        self.cfg = cfg or lane_d_cfg()
        self.ops = ops or ops_cfg()
        self.clock = clock or Clock()
        self.per_ip_rpm = per_ip_rpm
        self.per_hotkey_rpm = per_hotkey_rpm
        self.app = self.build()

    def build(self):
        from cascade.lane_d.service import LaneDIntake

        return LaneDIntake(storage=self.store, reads=self.reads, minter=self.minter,
                           cfg=self.cfg, ops=self.ops, state_dir=self.state_dir,
                           now=self.clock, per_ip_rpm=self.per_ip_rpm,
                           per_hotkey_rpm=self.per_hotkey_rpm)

    def restart(self):
        self.app = self.build()
        return self.app

    def queue(self) -> list[dict]:
        r = call(self.app, "GET", QUEUE, ip="192.0.2.250")
        assert r.status == 200, r
        assert isinstance(r.json, dict) and isinstance(r.json.get("entries"), list), r
        return r.json["entries"]

    def entry(self, hotkey: str, digest: str) -> dict | None:
        rows = [e for e in self.queue() if e.get("hotkey") == hotkey and e.get("digest") == digest]
        assert len(rows) <= 1, rows
        return rows[0] if rows else None

    def files_text(self) -> str:
        """Every file the intake wrote under its state dir, as text."""
        out = []
        for p in sorted(self.state_dir.rglob("*")):
            if p.is_file():
                out.append(p.read_bytes().decode("utf-8", "replace"))
        return "\n".join(out)


# ── the testnet evidence checker (independent of the producer) ──────────────

EVIDENCE_SCHEMA = "laned_w3_testnet/1"
# (case id, expected refusal code or "ok") the testnet run must record.
EVIDENCE_CASES = (
    ("T2", "duplicate_weights"),          # exact copy of the house refused at dedup
    ("T6_second_entry", "hotkey_entry_open"),
    ("T6_coldkey_cap", "coldkey_cap_open"),
    ("T16_front_run", "duplicate_weights"),
    ("T18d", "ready_expired"),
    ("T7c", "ok"),                        # 2.5 GB direct upload, ready 200
)
SECRET_SHAPES = re.compile(r"X-Amz-Signature|X-Amz-Credential|AWS4-HMAC|Authorization:|"
                           r"HIPPIUS_S3_SECRET|secret_key", re.IGNORECASE)
T7C_MIN_WEIGHTS_BYTES = 2_000_000_000
T7C_MAX_BOX_DELTA_MB = 64


def evidence_problems(ev: object, raw: bytes, chosen: str | None) -> list[str]:
    probs: list[str] = []
    if not isinstance(ev, dict):
        return ["evidence is not a JSON object"]
    if ev.get("schema") != EVIDENCE_SCHEMA:
        probs.append(f"schema must be {EVIDENCE_SCHEMA!r}")
    if ev.get("wp") != "W3":
        probs.append("wp must be 'W3'")
    if ev.get("source") != "testnet":
        probs.append("source must be 'testnet' (fixture evidence is never accepted)")
    if ev.get("network") != "test" or ev.get("netuid") != 259:
        probs.append("network must be 'test' and netuid 259 (plan §4)")
    if chosen is None:
        probs.append("decisions/STORAGE-LANE-D.md names no chosen upload path")
    elif ev.get("upload_path") != chosen:
        probs.append(f"upload_path {ev.get('upload_path')!r} is not the chosen {chosen!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(ev.get("intake_commit", ""))):
        probs.append("intake_commit must be the 40-hex commit the intake ran")
    cases = ev.get("cases")
    if not isinstance(cases, list):
        probs.append("cases must be a list")
        cases = []
    by_id = {}
    for c in cases:
        if isinstance(c, dict) and isinstance(c.get("id"), str):
            by_id.setdefault(c["id"], []).append(c)
    for cid, want in EVIDENCE_CASES:
        rows = by_id.get(cid, [])
        if len(rows) != 1:
            probs.append(f"case {cid}: expected exactly one row, found {len(rows)}")
            continue
        row = rows[0]
        if row.get("expected") != want:
            probs.append(f"case {cid}: expected must be {want!r}")
        if row.get("observed") != want:
            probs.append(f"case {cid}: observed {row.get('observed')!r} != {want!r}")
        status = row.get("http_status")
        ok_status = (status == 200) if want == "ok" else (isinstance(status, int)
                                                          and 400 <= status < 500)
        if not ok_status:
            probs.append(f"case {cid}: http_status {status!r} does not match {want!r}")
    t7c = ev.get("t7c")
    if not isinstance(t7c, dict):
        probs.append("t7c must be an object")
    else:
        wb = t7c.get("weights_bytes")
        if not isinstance(wb, int) or wb < T7C_MIN_WEIGHTS_BYTES:
            probs.append(f"t7c.weights_bytes must be >= {T7C_MIN_WEIGHTS_BYTES}")
        for k in ("box_rss_delta_mb", "box_disk_delta_mb"):
            v = t7c.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v > T7C_MAX_BOX_DELTA_MB:
                probs.append(f"t7c.{k} must be a number <= {T7C_MAX_BOX_DELTA_MB}")
    if SECRET_SHAPES.search(raw.decode("utf-8", "replace")):
        probs.append("evidence contains a credential- or signature-shaped string")
    if ev.get("passed") is not True:
        probs.append("passed must be true")
    elif probs:
        probs.append("passed is true but the checks above fail")
    return probs
