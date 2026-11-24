"""``cascade lane-d`` — miner commands for lane D, open weights (DEC-CA-0062,
DEC-CA-0070; docs/plans/LANE_D_PLAN.md §1.1, §1.2 "Flow", D43, B13).

A lane D entry is a cascade_pyramid checkpoint directory holding exactly the
four allowed files: ``config.json``, ``weights.safetensors`` (BF16 only,
D14) and the repo's own ``model.py`` / ``forecast_wrapper.py`` bytes.

* ``export <src> --out <dir>`` — write an entry directory from a trained
  checkpoint: the four files only, the repo's code bytes, the weights copied
  byte-identical. Prints the entry digest.
* ``verify <dir>`` — run lane D's ingest guards (W4) locally. The file
  allowlist runs first and needs no torch; nothing in the entry is ever
  imported or executed.
* ``submit <intake> <dir> --context C`` — sign the entry's canonical manifest
  with your hotkey (``POST /v1/lane-d/submit``), then upload every file
  straight to the vault through the grants the intake returns, on the one
  upload path ``decisions/STORAGE-LANE-D.md`` chose. Nothing transits the
  operator's box; no grant is ever printed or logged. Commits nothing.
* ``ready <intake> --digest D --context C --licence L --attest`` — attest the
  open licence (B13, ``POST /v1/lane-d/ready``), then commit the returned
  ``commit_payload`` on chain with an immediate reveal (D43:
  ``blocks_until_reveal`` is the one-block floor, whatever the block).
* ``status <intake> [--hotkey H]`` — the public lane D queue
  (``GET /v1/lane-d/queue``); no wallet.
* ``test <ckpt> --king K --context C --pool-dir P | --series S`` — score a
  local checkpoint offline with the validators' own lane D code (draw,
  ``score_window``, ``judge_cohort``, the crown bar); directional, submits
  nothing (``cascade.miner.lane_d_preview``, loaded at call time).

No flag takes a credential or a URL to a grant (§1.2). The hotkey signs; the
coldkey is never opened. The lane D intake and ingest helpers are loaded at
call time (``importlib``): this module sits in the static import closure of
``cascade.miner.cli``, which must reach no lane D producer module (W9
invariant 12).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import struct
import time
from pathlib import Path
from urllib.parse import urlsplit

from .output import NETWORK, OK, REGISTRY, REJECTED, USAGE, reporter

# D43: a lane D commitment reveals at once — the chain's one-block floor
# (lane A's "reveal now"), never timed to a settlement boundary.
REVEAL_NOW_BLOCKS = 1

SUBMIT_ROUTE = "/v1/lane-d/submit"
READY_ROUTE = "/v1/lane-d/ready"
QUEUE_ROUTE = "/v1/lane-d/queue"
COMMIT_PREFIX = "metro-v1:weights:"
WEIGHTS_FILE = "weights.safetensors"
CONFIG_FILE = "config.json"
# The repo files a lane D entry's code files must equal, byte for byte (§1.1).
CODE_SOURCES = {"model.py": "pyramid_model.py", "forecast_wrapper.py": "pyramid_wrapper.py"}
LANE_D_ARCH = "cascade_pyramid"
WEIGHTS_DTYPE = "BF16"
MAX_HEADER_BYTES = 4 * 1024 * 1024
INTAKE_TIMEOUT_S = 60.0
PUT_TIMEOUT_S = 300.0
_READ_CHUNK = 1 << 20
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
_COMMIT_RE = re.compile(r"metro-v1:weights:c(?P<ctx>[0-9]{1,7}):hippius:(?P<ref>[^\s:]+)")

# Read from the W3 intake service at first use (one source of truth: the
# intake's constants are pinned to decisions/STORAGE-LANE-D.md and B13).
_SERVICE_NAMES = ("UPLOAD_PATH", "UPLOAD_PATH_IDS", "LICENCES", "LICENCE_ATTESTATION")


def _service():
    return importlib.import_module("cascade.lane_d.service")


def _intake_helpers():
    return importlib.import_module("cascade.lane_d.intake")


def _ingest():
    return importlib.import_module("cascade.eval.lane_d_ingest")


def __getattr__(name: str):
    if name in _SERVICE_NAMES:
        return getattr(_service(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class LaneDFailure(Exception):
    """A typed refusal: ``code`` (stable, snake_case), a human message and
    the CLI exit code."""

    def __init__(self, code: str, message: str, exit_code: int = REJECTED, **fields):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.fields = fields


# ── the command group ────────────────────────────────────────────────────────


def _chain_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    p.add_argument("--wallet-hotkey", required=True,
                   help="Bittensor wallet hotkey name (the hotkey signs; the coldkey is "
                        "never opened).")
    p.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")


LANE_D_EPILOG = """\
examples (one dedicated hotkey; BF16 weights; MIT is the only licence):
  cascade lane-d test ./trained-ckpt --king ./king --context 16384 --pool-dir ./pool_d --device cuda
  cascade lane-d export ./trained-ckpt --out ./entry
  cascade lane-d verify ./entry
  cascade lane-d submit https://submissions.cascadesub.net ./entry --context 16384 --wallet-name W --wallet-hotkey H
  cascade lane-d ready https://submissions.cascadesub.net --digest D --context 16384 --licence MIT --attest --wallet-name W --wallet-hotkey H
  cascade lane-d status https://submissions.cascadesub.net --hotkey SS58
rules: `cascade rules d` prints the live [lane_d] values (contexts, bar, caps);
guide: docs/MINER_LANE_D.md (competition brief), docs/LANE_D.md (mechanics)."""


def add_lane_d(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("lane-d", help="Lane D (open weights): test offline, export, verify, "
                                      "submit, ready and track a checkpoint entry.",
                       epilog=LANE_D_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    lsub = p.add_subparsers(dest="lane_d_cmd", required=True)

    e = lsub.add_parser("export", help="Write a lane D entry directory from a trained "
                                       "cascade_pyramid checkpoint.")
    e.add_argument("src", type=Path, help="The trained checkpoint directory (config.json + "
                                          "weights.safetensors).")
    e.add_argument("--out", type=Path, required=True,
                   help="The entry directory to write (must be absent or empty).")
    e.set_defaults(func=cmd_export, command="lane-d export")

    v = lsub.add_parser("verify", help="Run lane D's ingest guards on a local entry directory "
                                       "(nothing in it is executed).")
    v.add_argument("entry_dir", type=Path, help="The entry directory.")
    v.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    v.set_defaults(func=cmd_verify, command="lane-d verify")

    s = lsub.add_parser("submit", help="Sign the entry's manifest and upload it straight to "
                                       "the vault (commits nothing).")
    s.add_argument("intake_url", help="Operator lane D intake base URL (https://…).")
    s.add_argument("entry_dir", type=Path, help="The entry directory (from `lane-d export`).")
    s.add_argument("--context", type=int, required=True,
                   help="The context length the entry is judged at (one of chain.toml "
                        "[lane_d] contexts, see `cascade rules d`; scored at the highest "
                        "armed context at or below it, never above).")
    _chain_args(s)
    s.set_defaults(func=cmd_submit, command="lane-d submit")

    r = lsub.add_parser("ready", help="Attest the open licence, then commit the entry on "
                                      "chain (immediate reveal).")
    r.add_argument("intake_url", help="Operator lane D intake base URL (https://…).")
    r.add_argument("--digest", required=True, help="The entry digest (from `lane-d submit`).")
    r.add_argument("--context", type=int, required=True, help="The submitted context length.")
    r.add_argument("--licence", required=True,
                   help="The open licence the checkpoint is published under: MIT, the "
                        "only accepted licence (weights and training data; no "
                        "restricted-licence teacher outputs).")
    r.add_argument("--attest", action="store_true",
                   help="Confirm the licence attestation statement (shown without this "
                        "flag; nothing is sent until you confirm).")
    _chain_args(r)
    r.set_defaults(func=cmd_ready, command="lane-d ready")

    q = lsub.add_parser("status", help="The public lane D queue (optionally one hotkey).")
    q.add_argument("intake_url", help="Operator lane D intake base URL.")
    q.add_argument("--hotkey", default=None, help="Show only this hotkey's entries.")
    q.set_defaults(func=cmd_status, command="lane-d status")

    # `lane-d test`: the offline preview, loaded at call time (W9 invariant 12:
    # it imports the validators' lane D scoring code).
    importlib.import_module("cascade.miner.lane_d_preview").add_lane_d_test(lsub)


def _run(args: argparse.Namespace, body) -> int:
    out = reporter(args)
    try:
        return body(args, out)
    except LaneDFailure as f:
        return out.fail(f.code, f.message, f.exit_code, **f.fields)


# ── local checks (pure; no network, no torch) ────────────────────────────────


def repo_code() -> dict[str, bytes]:
    """The repo's pyramid code bytes every lane D entry carries (§1.1)."""
    trainer = Path(__file__).resolve().parents[1] / "trainer"
    return {name: (trainer / src).read_bytes() for name, src in CODE_SOURCES.items()}


def entry_files(directory: Path) -> list[str]:
    """Every regular file below ``directory`` as a ``/``-relative name."""
    root = Path(directory)
    if not root.is_dir():
        raise LaneDFailure("missing_checkpoint", f"{root} is not a directory", USAGE)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def check_allowlist(directory: Path) -> list[str]:
    """W4's allowlist (``file_not_allowed``) over a local directory, by
    name only: nothing is opened, imported or executed."""
    names = entry_files(directory)
    ingest = _ingest()
    try:
        ingest._allowlist({n: {} for n in names})
    except ingest.IngestError as e:
        raise LaneDFailure(e.code.value, f"entry refused: {e}") from None
    return names


def weights_dtypes(path: Path) -> set[str]:
    """The tensor dtypes in a safetensors header, parsed by hand (no torch)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise LaneDFailure("bad_header", f"{path.name}: truncated safetensors length")
        (n,) = struct.unpack("<Q", raw)
        if n == 0 or n > MAX_HEADER_BYTES:
            raise LaneDFailure("bad_header", f"{path.name}: header length {n} out of bounds")
        blob = f.read(n)
    try:
        header = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise LaneDFailure("bad_header", f"{path.name}: unparseable safetensors header") from None
    if not isinstance(header, dict):
        raise LaneDFailure("bad_header", f"{path.name}: header is not an object")
    return {str(v.get("dtype")) if isinstance(v, dict) else "" for k, v in header.items()
            if k != "__metadata__"}


def check_commit_payload(payload, *, context: int, hotkey: str, digest: str) -> str:
    """The intake's ``commit_payload``, refused unless it names exactly this
    entry: ``metro-v1:weights:c<context>:hippius:<vault>/<hotkey>/<digest>``."""
    m = _COMMIT_RE.fullmatch(payload) if isinstance(payload, str) else None
    if (m is None or int(m.group("ctx")) != int(context)
            or not m.group("ref").endswith(f"/{hotkey}/{digest}")):
        raise LaneDFailure("bad_commit_payload",
                           "the intake's commit payload does not name this entry "
                           f"({COMMIT_PREFIX}c{context}:hippius:…/{hotkey}/{digest}); "
                           "nothing committed")
    return payload


def _loopback(url: str) -> bool:
    return (urlsplit(url).hostname or "") in _LOOPBACK_HOSTS


def transport_ok(url: str) -> bool:
    """https, or plain http to this machine only."""
    scheme = urlsplit(url).scheme
    return scheme == "https" or (scheme == "http" and _loopback(url))


def check_grants(grants, manifest_files: dict) -> dict[str, list[dict]]:
    """The submit answer's grants, checked before any byte is sent: exactly
    the manifest's files; each file's PUTs cover ``[0, size)`` contiguously
    in order; every URL https (or http to loopback). Returns
    ``{name: [request, ...]}``."""
    files = grants.get("files") if isinstance(grants, dict) else None
    if not isinstance(files, dict) or set(files) != set(manifest_files):
        raise LaneDFailure("bad_grants", "the upload grants do not name exactly the entry's "
                                         "files; nothing uploaded")
    plan: dict[str, list[dict]] = {}
    for name in sorted(manifest_files):
        size = int(manifest_files[name]["size"])
        reqs = files[name].get("requests") if isinstance(files[name], dict) else None
        if not isinstance(reqs, list):
            raise LaneDFailure("bad_grants", f"{name}: the grant has no request list; "
                                             "nothing uploaded")
        pos = 0
        for r in reqs:
            ok = (isinstance(r, dict) and r.get("method") == "PUT"
                  and isinstance(r.get("url"), str)
                  and all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool)
                          for k in ("offset", "length"))
                  and isinstance(r.get("headers", {}), dict)
                  and all(isinstance(k, str) and isinstance(v, str)
                          for k, v in r.get("headers", {}).items()))
            if not ok or r["offset"] != pos or r["length"] <= 0:
                raise LaneDFailure("bad_grants", f"{name}: the grant's byte ranges are not "
                                                 "contiguous PUTs; nothing uploaded")
            pos += r["length"]
        if pos != size:
            raise LaneDFailure("bad_grants", f"{name}: the grant covers {pos} of {size} bytes; "
                                             "nothing uploaded")
        plan[name] = reqs
    for name, reqs in plan.items():
        if not all(transport_ok(r["url"]) for r in reqs):
            raise LaneDFailure("insecure_grant", f"{name}: an upload grant is not https; "
                                                 "nothing uploaded")
    return plan


# ── HTTP (stdlib only) ───────────────────────────────────────────────────────


def _opener(url: str):
    import urllib.request

    if _loopback(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _decode(raw: bytes) -> dict:
    try:
        doc = json.loads(raw or b"{}")
    except ValueError:
        return {"code": "non_json_response",
                "message": raw.decode("utf-8", errors="replace").strip()[:200]}
    return doc if isinstance(doc, dict) else {"message": str(doc)[:200]}


def intake_call(url: str, *, method: str, headers: dict | None = None,
                body: bytes | None = None) -> dict:
    """One intake request; the 2xx JSON answer. A 4xx is the intake's own
    refusal (exit 1, its ``code``); a 5xx or no answer is exit 3."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
    try:
        with _opener(url).open(req, timeout=INTAKE_TIMEOUT_S) as resp:
            return _decode(resp.read())
    except urllib.error.HTTPError as e:
        doc = _decode(e.read())
        code = str(doc.get("code") or f"http_{e.code}")
        msg = f"intake refused ({e.code}): {code} — {str(doc.get('message', ''))[:300]}"
        raise LaneDFailure(code, msg, NETWORK if e.code >= 500 else REJECTED,
                           http_status=e.code) from None
    except (urllib.error.URLError, OSError, ValueError) as e:
        reason = getattr(e, "reason", None) or type(e).__name__
        raise LaneDFailure("intake_unreachable", f"intake unreachable: {reason}",
                           NETWORK) from None


class _FileSlice:
    """``file[offset:offset+length]`` as a readable body, streamed in chunks
    (a 2.5 GB weights file is never held in RAM)."""

    def __init__(self, path: Path, offset: int, length: int):
        self._f = open(path, "rb")  # noqa: SIM115 — closed by close()
        self._f.seek(offset)
        self._left = length

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        want = self._left if n is None or n < 0 else min(n, self._left)
        chunk = self._f.read(min(want, _READ_CHUNK))
        self._left -= len(chunk)
        return chunk

    def close(self) -> None:
        self._f.close()


def upload(directory: Path, plan: dict[str, list[dict]]) -> int:
    """PUT every granted byte range; returns the bytes sent. A failure names
    the file and part only (a grant URL carries a signature)."""
    import http.client
    import urllib.error
    import urllib.request

    sent = 0
    for name, reqs in plan.items():
        for i, r in enumerate(reqs, start=1):
            headers = {"Content-Type": "application/octet-stream", **r.get("headers", {}),
                       "Content-Length": str(r["length"])}
            body = None
            try:
                body = _FileSlice(directory / name, r["offset"], r["length"])
                req = urllib.request.Request(r["url"], data=body, method="PUT", headers=headers)
                with _opener(r["url"]).open(req, timeout=PUT_TIMEOUT_S) as resp:
                    resp.read()
            except urllib.error.HTTPError as e:
                raise LaneDFailure("upload_failed", f"upload of {name} part {i}/{len(reqs)} "
                                                    f"answered HTTP {e.code}", REGISTRY) from None
            except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
                raise LaneDFailure("upload_failed", f"upload of {name} part {i}/{len(reqs)} "
                                                    f"failed: {type(e).__name__}",
                                   REGISTRY) from None
            finally:
                if body is not None:
                    body.close()
            sent += r["length"]
    return sent


# ── signing ──────────────────────────────────────────────────────────────────


def signed_headers(action: str, hotkey, digest: str, context: int, body: bytes) -> dict:
    """W3's ownership headers: ``hotkey`` (a keypair) signs the intake's
    canonical message over the action, identity, timestamp and body hash."""
    ts = str(int(time.time()))
    message = _service().canonical_message(action, hotkey.ss58_address, digest, str(context),
                                           ts, body)
    sig = hotkey.sign(message)
    sig_hex = sig.hex() if isinstance(sig, bytes | bytearray) else str(sig).removeprefix("0x")
    return {"Content-Type": "application/json", "X-Miner-Hotkey": hotkey.ss58_address,
            "X-Content-Digest": digest, "X-Context-Length": str(context), "X-Timestamp": ts,
            "X-Signature": sig_hex}


def _chain_client(cfg, args):
    from ..shared.chain import ChainClient, ChainError

    try:
        return ChainClient.from_config(cfg, network=args.network, wallet_name=args.wallet_name,
                                       wallet_hotkey=args.wallet_hotkey,
                                       wallet_path=args.wallet_path)
    except ChainError as e:
        raise LaneDFailure("chain_error", f"chain error: {e}", NETWORK) from None


def _hotkey(client):
    try:
        return client.wallet().hotkey
    except Exception as e:  # noqa: BLE001 — wallet errors are usage errors here
        raise LaneDFailure("wallet_unavailable", f"could not load the hotkey for signing: "
                                                 f"{type(e).__name__}", USAGE) from None


def _armed_context(cfg, context: int) -> int:
    contexts = tuple(int(c) for c in cfg.lane_d.contexts)
    if int(context) not in contexts:
        raise LaneDFailure("context_not_armed", f"--context {context} is not one of chain.toml "
                                                f"[lane_d] contexts {list(contexts)}", USAGE)
    return int(context)


def _base(url: str) -> str:
    base = url.rstrip("/")
    if not transport_ok(base):
        raise LaneDFailure("insecure_intake", "refusing plain http to a non-local intake; "
                                              "use https", USAGE)
    return base


# ── commands ─────────────────────────────────────────────────────────────────


def cmd_export(args: argparse.Namespace) -> int:
    return _run(args, _export)


def _export(args, out) -> int:
    src, dest = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise LaneDFailure("missing_checkpoint", f"{src} is not a directory", USAGE)
    for name, code in (("weights.safetensors", "missing_weights"),
                       ("config.json", "missing_config")):
        if not (src / name).is_file():
            raise LaneDFailure(code, f"{src} has no {name}", USAGE)
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise LaneDFailure("out_not_empty", f"{dest} exists and is not an empty directory; "
                                            "nothing written", USAGE)
    code = repo_code()
    for name, want in code.items():
        p = src / name
        if p.exists() and p.read_bytes() != want:
            raise LaneDFailure("code_mismatch", f"{name} in {src} is not this release's "
                                                f"cascade/trainer/{CODE_SOURCES[name]}; lane D "
                                                "entries carry the repo's code unchanged")
    try:
        json.loads((src / CONFIG_FILE).read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise LaneDFailure("config_mismatch", f"{src / CONFIG_FILE} is not valid JSON") from None
    dtypes = weights_dtypes(src / WEIGHTS_FILE)
    if dtypes != {WEIGHTS_DTYPE}:
        raise LaneDFailure("dtype_mismatch", f"{WEIGHTS_FILE} holds {sorted(dtypes)}; lane D "
                                             f"entries are {WEIGHTS_DTYPE} only (D14): cast "
                                             "the weights to bfloat16 and save again")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{dest.name}.export-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        for name, blob in code.items():
            (staging / name).write_bytes(blob)
        shutil.copyfile(src / CONFIG_FILE, staging / CONFIG_FILE)
        shutil.copyfile(src / WEIGHTS_FILE, staging / WEIGHTS_FILE)
        if dest.exists():
            dest.rmdir()
        os.replace(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    helpers = _intake_helpers()
    manifest = helpers.checkpoint_manifest(dest)
    digest = helpers.manifest_digest(manifest)
    out.result(digest=digest, out=str(dest), files=manifest["files"])
    out.info(f"exported {dest} (digest {digest}); next: cascade lane-d verify {dest}")
    return OK


def cmd_verify(args: argparse.Namespace) -> int:
    return _run(args, _verify)


def _lane_d_contract(cfg):
    for c in (getattr(cfg.training, "v2_contract", None), cfg.training.primary_size):
        if c is not None and getattr(c, "base_arch", "") == LANE_D_ARCH:
            return c
    raise LaneDFailure("no_lane_d_contract", f"this chain.toml pins no {LANE_D_ARCH} "
                                             "contract to check the entry against", USAGE)


def _verify(args, out) -> int:
    d = Path(args.entry_dir)
    names = check_allowlist(d)          # by name only: before torch, before any read
    ingest = _ingest()
    for name, want in repo_code().items():      # W4 step 5, before anything needs torch
        if (d / name).read_bytes() != want:
            raise LaneDFailure(ingest.IngestCode.CODE_MISMATCH.value,
                               f"entry refused: {name} differs from this release's copy")
    from ..shared.config import load_chain_config

    cfg = load_chain_config(args.chain_toml)
    contract = _lane_d_contract(cfg)
    try:
        expect = ingest.expectation_for(contract)
    except ImportError as e:
        raise LaneDFailure("torch_unavailable", f"the allowlist and code checks passed; the "
                                                f"config and weights checks need torch "
                                                f"({e.name or e}): install the [train] extra",
                           USAGE) from None
    manifest = _intake_helpers().checkpoint_manifest(d)
    try:
        ingest._budget(manifest["files"], expect)
        ingest._check_checkpoint(d, expect)
    except ingest.IngestError as e:
        raise LaneDFailure(e.code.value, f"entry refused: {e}") from None
    digest = _intake_helpers().manifest_digest(manifest)
    out.result(ok=True, digest=digest, files=names)
    out.info(f"entry OK — the validators' ingest guards accept it (digest {digest})")
    return OK


def cmd_submit(args: argparse.Namespace) -> int:
    return _run(args, _submit)


def _submit(args, out) -> int:
    from ..shared.config import load_chain_config

    d = Path(args.entry_dir)
    check_allowlist(d)
    cfg = load_chain_config(args.chain_toml)
    context = _armed_context(cfg, args.context)
    base = _base(args.intake_url)
    helpers = _intake_helpers()
    manifest = helpers.checkpoint_manifest(d)
    body = helpers.canonical_manifest(manifest)
    digest = helpers.manifest_digest(manifest)
    chosen = _service().UPLOAD_PATH
    hotkey = _hotkey(_chain_client(cfg, args))
    headers = {**signed_headers("submit", hotkey, digest, context, body),
               "X-Upload-Path": chosen}
    answer = intake_call(base + SUBMIT_ROUTE, method="POST", headers=headers, body=body)
    if answer.get("upload_path") != chosen:
        raise LaneDFailure("upload_path_not_offered",
                           f"the intake answered for upload path "
                           f"{str(answer.get('upload_path'))[:40]!r}; this CLI uploads only "
                           f"{chosen!r} (decisions/STORAGE-LANE-D.md); nothing uploaded")
    plan = check_grants(answer.get("grants"), manifest["files"])
    sent = upload(d, plan)
    out.result(digest=digest, hotkey=hotkey.ss58_address, context=context,
               upload_path=chosen, uploaded_bytes=sent)
    out.info(f"uploaded {sent} bytes for entry {digest}; next, attest the licence and commit:\n"
             f"  cascade lane-d ready {base} --digest {digest} --context {context} "
             "--licence MIT --attest ...")
    return OK


def cmd_ready(args: argparse.Namespace) -> int:
    return _run(args, _ready)


def _ready(args, out) -> int:
    from ..shared.chain import ChainError
    from ..shared.config import load_chain_config

    service = _service()
    if args.licence not in service.LICENCES:
        raise LaneDFailure("licence_not_open", f"--licence must be one of "
                                               f"{list(service.LICENCES)}; nothing sent", USAGE)
    if not args.attest:
        out.info("The statement you attest at ready:\n\n  " + service.LICENCE_ATTESTATION
                 + "\n\nRe-run with --attest to confirm it.")
        raise LaneDFailure("licence_attestation_missing", "ready needs --attest (the licence "
                                                          "attestation); nothing sent", USAGE)
    digest = str(args.digest)
    if not _HEX64.fullmatch(digest):
        raise LaneDFailure("bad_digest", "--digest must be 64 lowercase hex", USAGE)
    cfg = load_chain_config(args.chain_toml)
    context = _armed_context(cfg, args.context)
    base = _base(args.intake_url)
    client = _chain_client(cfg, args)
    hotkey = _hotkey(client)
    body = json.dumps({"licence": args.licence, "attestation": service.LICENCE_ATTESTATION},
                      separators=(",", ":")).encode("utf-8")
    headers = signed_headers("ready", hotkey, digest, context, body)
    answer = intake_call(base + READY_ROUTE, method="POST", headers=headers, body=body)
    payload = check_commit_payload(answer.get("commit_payload"), context=context,
                                   hotkey=hotkey.ss58_address, digest=digest)
    try:
        commit_block = int(client.current_block())
        client.commit_submission(payload, blocks_until_reveal=REVEAL_NOW_BLOCKS)
    except ChainError as e:
        raise LaneDFailure("chain_error", f"chain error: {e}; the entry is ready at the "
                                          "intake but not committed — re-run ready before "
                                          "it expires", NETWORK) from None
    out.result(digest=digest, hotkey=hotkey.ss58_address, context=context,
               licence=args.licence, coldkey_at_ready=answer.get("coldkey_at_ready"),
               ready_block=answer.get("ready_block"),
               ready_expires_block=answer.get("ready_expires_block"),
               commit_payload=payload, blocks_until_reveal=REVEAL_NOW_BLOCKS,
               commit_block=commit_block)
    out.info(f"committed: {payload} (reveals at once; ready until block "
             f"{answer.get('ready_expires_block')})")
    return OK


def cmd_status(args: argparse.Namespace) -> int:
    return _run(args, _status)


def _status(args, out) -> int:
    answer = intake_call(args.intake_url.rstrip("/") + QUEUE_ROUTE, method="GET")
    entries = answer.get("entries")
    if not isinstance(entries, list):
        raise LaneDFailure("bad_queue", "the intake's queue answer has no entry list", NETWORK)
    if args.hotkey:
        entries = [e for e in entries if isinstance(e, dict) and e.get("hotkey") == args.hotkey]
    out.result(entries=entries)
    if not entries:
        out.info("no lane D entries" + (f" for {args.hotkey}" if args.hotkey else ""))
    for e in entries:
        if isinstance(e, dict):
            out.info(f"{e.get('state', '?'):<14} c{e.get('context', '?')}  "
                     f"{e.get('hotkey', '?')}  {e.get('digest', '?')}")
    return OK


__all__ = [
    "LaneDFailure", "REVEAL_NOW_BLOCKS", "add_lane_d", "check_allowlist", "check_commit_payload",
    "check_grants", "repo_code", "transport_ok", "weights_dtypes",
]
