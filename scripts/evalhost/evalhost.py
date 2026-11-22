#!/usr/bin/env python3
"""Eval-host provisioning kit (LANE_D_PLAN R12; §10.2, §10.4, §10.5 S4).

    python -P scripts/evalhost/evalhost.py <subcommand> ...

Subcommands (contract: tests/accept/r12/README.md):

* ``on-rent-check``   CAPTURED nvidia-smi / df output vs a host profile.
* ``pin-host-key``    pin a CAPTURED ssh-keyscan key; refuse a changed one.
* ``resolve-endpoint`` find the pod in CAPTURED ``lium ps`` JSON (Lium's NAT
  moves RUNNING pods; resolve before ever calling one dead).
* ``sync``            copy the release (cascade/, benchmarks/, the lock) over a
  pinned-key rsync, then ``uv sync --frozen`` on the host.
* ``identities``      fold the pod's bench_identity + file-count captures into
  the identities JSON ``handshake`` reads.
* ``handshake``       the four suite identities + the W4 code handshake.
* ``evidence``        re-run every check over a captures dir; write
  ``laned_r12_pod/1`` evidence.
* ``profile-info``    one profile value, for the owner-paste scripts.

Every check reads captures only; the only subcommand that reaches a host is
``sync`` without ``--dry-run``. Nothing here reads or prints a secret.

Exit codes: 0 ok; 1 evidence did not pass; 2 usage / unparseable input;
3 a sync transfer or remote step failed; 10-13 on-rent; 20-21 host key;
30-34 endpoint; 40-41 handshake.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = KIT_DIR.parents[1]
PROFILE_DIR = KIT_DIR / "profiles"

PROFILE_SCHEMA = "laned_evalhost_profile/1"
EVIDENCE_SCHEMA = "laned_r12_pod/1"
WP = "R12"
DEFAULT_BENCH_DIR = "/data/bench"

EXIT_OK, EXIT_NOT_PASSED, EXIT_USAGE, EXIT_REMOTE = 0, 1, 2, 3
ON_RENT_CODES = {"wrong_gpu": 10, "vram_low": 11, "driver_low": 12, "disk_low": 13}
EXIT_HOST_KEY_CHANGED, EXIT_FINGERPRINT_MISMATCH = 20, 21
RESOLVE_CODES = {"same_endpoint": 0, "same_pod_repin": 30, "endpoint_key_changed": 31,
                 "not_found": 32, "not_running": 33, "endpoint_changed_unverified": 34}
EXIT_SUITE_MISMATCH, EXIT_CODE_MISMATCH = 40, 41

SMI_LIST_CMD = "nvidia-smi -L"
SMI_QUERY_CMD = ("nvidia-smi --query-gpu=name,memory.total,driver_version "
                 "--format=csv,noheader,nounits")
LIUM_PS_CMD = "lium ps --format json"
# What the pod runs for the handshake captures (inside the synced release).
# bench_identity.py is stdlib-only, run as a file so no package __init__ loads;
# the hello runs from the release dir so `cascade` resolves to that checkout.
BENCH_IDENTITY_CMD = ".venv/bin/python -P cascade/validator/bench_identity.py --data-dir {bench_dir}"
BENCH_FILES_CMD = "find {bench_dir} -type f | wc -l"
HELLO_CMD = (".venv/bin/python -c 'import json; from cascade.eval.lane_d_handshake import "
             "worker_hello; print(json.dumps(worker_hello()))'")
# bench_identity's suite names -> the profile's [suite_identities] keys.
SUITE_ALIASES = {"gift-eval": "gift"}

COPY_TREES = ("cascade", "benchmarks")
ROOT_REQUIRED = ("pyproject.toml", "uv.lock")
ROOT_OPTIONAL = (".python-version",)
SKIP_DIRS = frozenset({"__pycache__", ".venv", ".git"})
SECRET_NAME_RES = tuple(re.compile(p) for p in (
    r"^\.env$", r"^\.env\..*$", r"^.*\.env$", r"^.*\.pem$", r"^.*\.key$", r"^id_rsa.*$",
    r"^id_ecdsa.*$", r"^id_ed25519.*$", r"^known_hosts.*$", r"^\.netrc$", r"secret",
    r"credential",
))
SAFE_PATH_RE = re.compile(r"[A-Za-z0-9_./@:+,=-]+")
USER_HOST_RE = re.compile(r"[A-Za-z0-9._-]+@[A-Za-z0-9.:_-]+")
HOST_RE = re.compile(r"[A-Za-z0-9.:_-]+")
HEX_RE = re.compile(r"[0-9a-f]+")
DRIVER_RE = re.compile(r"\d+(?:\.\d+)*")
SMI_LIST_RE = re.compile(r"GPU (\d+): (.+?) \(UUID: [^)]*\)")


class UsageError(Exception):
    """Bad arguments or an unparseable capture: exit 2, never a pass."""


# ── small helpers ────────────────────────────────────────────────────────────


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_text(path: str, what: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise UsageError(f"cannot read {what} {path}: {e.__class__.__name__}") from None


def read_json(path: str, what: str):
    try:
        return json.loads(read_text(path, what))
    except json.JSONDecodeError as e:
        raise UsageError(f"{what} {path} is not JSON: {e}") from None


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split("."))


def parse_port(value) -> int:
    try:
        port = int(str(value))
    except ValueError:
        raise UsageError(f"port {value!r} is not an integer") from None
    if not 0 < port < 65536:
        raise UsageError(f"port {port} is out of range")
    return port


def parse_endpoint(value: str) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    host = host.strip("[]")
    if not sep or not host or not HOST_RE.fullmatch(host):
        raise UsageError(f"want HOST:PORT, got {value!r}")
    return host, parse_port(port)


def safe_path(value: str, what: str) -> str:
    if not SAFE_PATH_RE.fullmatch(value):
        raise UsageError(f"{what} {value!r} must be a plain path (no spaces or quotes)")
    return value


# ── profiles ─────────────────────────────────────────────────────────────────

PROFILE_KEYS = {"name": str, "gpu_model": str, "min_vram_mib": int, "min_driver": str,
                "disk_path": str, "min_disk_free_bytes": int}


def load_profile(path: str) -> dict:
    try:
        data = tomllib.loads(read_text(path, "profile"))
    except tomllib.TOMLDecodeError as e:
        raise UsageError(f"profile {path} does not parse: {e}") from None
    if data.get("schema") != PROFILE_SCHEMA:
        raise UsageError(f"profile {path}: schema must be {PROFILE_SCHEMA!r}, "
                         f"got {data.get('schema')!r}")
    for key, typ in PROFILE_KEYS.items():
        value = data.get(key)
        if not isinstance(value, typ) or isinstance(value, bool) or value == "":
            raise UsageError(f"profile {path}: missing or invalid {key!r}")
    if not DRIVER_RE.fullmatch(data["min_driver"]):
        raise UsageError(f"profile {path}: min_driver {data['min_driver']!r} is not dotted ints")
    if not data["disk_path"].startswith("/") or not SAFE_PATH_RE.fullmatch(data["disk_path"]):
        raise UsageError(f"profile {path}: disk_path must be a plain absolute path")
    bench_files = data.get("bench_files")
    if bench_files is not None and (not isinstance(bench_files, int)
                                    or isinstance(bench_files, bool) or bench_files < 0):
        raise UsageError(f"profile {path}: bench_files must be a non-negative integer")
    pins = data.get("suite_identities") or {}
    if not isinstance(pins, dict) or not all(
            isinstance(v, str) and len(v) == 8 and HEX_RE.fullmatch(v) for v in pins.values()):
        raise UsageError(f"profile {path}: suite_identities must map suite -> 8 lowercase hex")
    bench_dir = data.get("bench_dir", DEFAULT_BENCH_DIR)
    if not isinstance(bench_dir, str) or not bench_dir.startswith("/") \
            or not SAFE_PATH_RE.fullmatch(bench_dir):
        raise UsageError(f"profile {path}: bench_dir must be a plain absolute path")
    data["suite_identities"] = dict(pins)
    data["bench_dir"] = bench_dir
    return data


# ── on-rent check ────────────────────────────────────────────────────────────


def parse_smi_list(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = SMI_LIST_RE.fullmatch(line.strip())
        if not m:
            raise UsageError(f"nvidia-smi -L capture line is not `GPU <i>: <name> (UUID: …)`: "
                             f"{line.strip()[:120]!r}")
        names.append(m.group(2).strip())
    return names


def parse_smi_query(text: str) -> list[tuple[str, int, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not parts[0] or not parts[1].isdigit() \
                or not DRIVER_RE.fullmatch(parts[2]):
            raise UsageError(f"nvidia-smi query capture line is not `<name>, <MiB>, <driver>`: "
                             f"{line.strip()[:120]!r}")
        rows.append((parts[0], int(parts[1]), parts[2]))
    return rows


def parse_df(text: str) -> int:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].split()[0] == "Avail":
        lines = lines[1:]
    if len(lines) != 1:
        raise UsageError("df capture must be one header line then one `<avail bytes> <mount>` line")
    m = re.fullmatch(r"\s*(\d+)\s+(\S.*)", lines[0])
    if not m:
        raise UsageError(f"df capture line is not `<avail bytes> <mount>`: {lines[0][:120]!r}")
    return int(m.group(1))


def on_rent(profile: dict, list_text: str, query_text: str, df_text: str) -> tuple[dict, int]:
    names = parse_smi_list(list_text)
    rows = parse_smi_query(query_text)
    free = parse_df(df_text)
    qnames = [r[0] for r in rows]
    vram = [r[1] for r in rows]
    drivers = sorted({r[2] for r in rows}, key=version_tuple)
    driver = drivers[0] if drivers else ""
    want = profile["gpu_model"]
    reasons = []
    if not names or not rows:
        reasons.append(("wrong_gpu", "no GPU listed by nvidia-smi"))
    elif names != qnames:
        reasons.append(("wrong_gpu", f"nvidia-smi -L lists {names} but the query reports "
                                     f"{qnames}"))
    else:
        bad = sorted({n for n in names + qnames if n != want})
        if bad:
            reasons.append(("wrong_gpu", f"GPU model {bad} is not the pinned {want!r}"))
    low = [v for v in vram if v < profile["min_vram_mib"]]
    if low:
        reasons.append(("vram_low", f"GPU memory {vram} MiB, need >= {profile['min_vram_mib']} "
                                    "MiB on every GPU"))
    if not driver:
        reasons.append(("driver_low", "no driver version reported"))
    elif version_tuple(driver) < version_tuple(profile["min_driver"]):
        reasons.append(("driver_low", f"driver {driver} < {profile['min_driver']}"))
    if len(drivers) > 1:
        reasons.append(("driver_low", f"GPUs report different drivers {drivers}"))
    if free < profile["min_disk_free_bytes"]:
        reasons.append(("disk_low", f"{free} bytes free on {profile['disk_path']}, need >= "
                                    f"{profile['min_disk_free_bytes']}"))
    ordered = []
    for code in ON_RENT_CODES:
        details = [d for c, d in reasons if c == code]
        if details:
            ordered.append({"code": code, "detail": "; ".join(details)})
    out = {"ok": not ordered, "reasons": ordered,
           "observed": {"gpu_models": names, "vram_mib": vram, "driver": driver,
                        "disk_free_bytes": free},
           "required": {"gpu_model": want, "min_vram_mib": profile["min_vram_mib"],
                        "min_driver": profile["min_driver"],
                        "min_disk_free_bytes": profile["min_disk_free_bytes"]}}
    return out, (ON_RENT_CODES[ordered[0]["code"]] if ordered else EXIT_OK)


def df_cmd(disk_path: str) -> str:
    return f"df -B1 --output=avail,target {disk_path}"


def cmd_on_rent(args) -> int:
    profile = load_profile(args.profile)
    if args.dry_run:
        emit({"dry_run": True,
              "would_run": [SMI_LIST_CMD, SMI_QUERY_CMD, df_cmd(profile["disk_path"])]})
        return EXIT_OK
    missing = [f for f, v in (("--nvidia-smi-L", args.nvidia_smi_L),
                              ("--nvidia-smi-query", args.nvidia_smi_query),
                              ("--df", args.df)) if not v]
    if missing:
        raise UsageError(f"on-rent-check needs {missing} (or --dry-run)")
    out, rc = on_rent(profile, read_text(args.nvidia_smi_L, "capture"),
                      read_text(args.nvidia_smi_query, "capture"), read_text(args.df, "capture"))
    emit(out)
    return rc


# ── host keys ────────────────────────────────────────────────────────────────


def host_pattern(host: str, port: int) -> str:
    return host if port == 22 else f"[{host}]:{port}"


def key_fingerprint(blob: str) -> str | None:
    try:
        raw = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 8:
        return None
    return "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")


def _pattern_matches(field: str, pattern: str) -> bool:
    if field.startswith("|1|"):
        parts = field.split("|")
        if len(parts) != 4:
            return False
        try:
            salt = base64.b64decode(parts[2], validate=True)
            want = base64.b64decode(parts[3], validate=True)
        except (binascii.Error, ValueError):
            return False
        got = hmac.new(salt, pattern.encode(), hashlib.sha1).digest()
        return hmac.compare_digest(got, want)
    return pattern in field.split(",")


def parse_key_lines(text: str) -> list[dict]:
    """known_hosts-format lines: ``[@marker] patterns keytype blob [comment]``."""
    entries = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        marker = ""
        if parts[0].startswith("@"):
            marker, parts = parts[0], parts[1:]
        if len(parts) < 3:
            continue
        fp = key_fingerprint(parts[2])
        if fp is None:
            continue
        entries.append({"marker": marker, "patterns": parts[0], "type": parts[1],
                        "blob": parts[2], "fingerprint": fp})
    return entries


def keys_for(entries: list[dict], host: str, port: int) -> dict[tuple[str, str], str]:
    """``{(keytype, blob): fingerprint}`` of plain entries matching host:port."""
    pat = host_pattern(host, port)
    return {(e["type"], e["blob"]): e["fingerprint"] for e in entries
            if not e["marker"] and _pattern_matches(e["patterns"], pat)}


def revoked_blobs(entries: list[dict]) -> set[str]:
    return {e["blob"] for e in entries if e["marker"] == "@revoked"}


def key_conflict(scanned: dict, pinned: dict) -> bool:
    """A scanned key differs from what is pinned: same key type with another
    blob, or pins exist and no scanned key is among them."""
    if not pinned:
        return False
    pinned_types = {t for t, _ in pinned}
    for kt, blob in scanned:
        if kt in pinned_types and (kt, blob) not in pinned:
            return True
    return not any(k in pinned for k in scanned)


def scanned_keys(path: str, host: str, port: int) -> dict:
    keys = keys_for(parse_key_lines(read_text(path, "ssh-keyscan capture")), host, port)
    if not keys:
        raise UsageError(f"the ssh-keyscan capture {path} holds no key for "
                         f"{host_pattern(host, port)}")
    return keys


def read_known_hosts(path: str) -> bytes:
    p = Path(path)
    if not p.exists():
        return b""
    try:
        return p.read_bytes()
    except OSError as e:
        raise UsageError(f"cannot read known_hosts {path}: {e.__class__.__name__}") from None


def append_known_hosts(path: str, existing: bytes, lines: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prefix = existing if not existing or existing.endswith(b"\n") else existing + b"\n"
    data = prefix + "".join(ln + "\n" for ln in lines).encode()
    mode = stat.S_IMODE(p.stat().st_mode) if p.exists() else 0o600
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def cmd_pin_host_key(args) -> int:
    port = parse_port(args.port)
    host = args.host
    if not HOST_RE.fullmatch(host):
        raise UsageError(f"--host {host!r} is not a host name or address")
    scanned = scanned_keys(args.scanned, host, port)
    raw = read_known_hosts(args.known_hosts)
    entries = parse_key_lines(raw.decode("utf-8", errors="replace"))
    pinned = keys_for(entries, host, port)
    out = {"host": host, "port": port, "fingerprints": sorted(set(scanned.values())),
           "pinned_fingerprints": sorted(set(pinned.values())), "dry_run": bool(args.dry_run)}

    def refuse(reason: str, rc: int) -> int:
        out.update(action="refused", reason=reason)
        emit(out)
        return rc

    if args.fingerprint and args.fingerprint not in out["fingerprints"]:
        return refuse("fingerprint_mismatch", EXIT_FINGERPRINT_MISMATCH)
    if any(blob in revoked_blobs(entries) for _, blob in scanned):
        return refuse("host_key_revoked", EXIT_HOST_KEY_CHANGED)
    if key_conflict(scanned, pinned):
        return refuse("host_key_changed", EXIT_HOST_KEY_CHANGED)
    if args.previous:
        phost, pport = parse_endpoint(args.previous)
        prev = keys_for(entries, phost, pport)
        out["previous"] = f"{phost}:{pport}"
        out["previous_fingerprints"] = sorted(set(prev.values()))
        if not prev:
            return refuse("previous_not_pinned", EXIT_HOST_KEY_CHANGED)
        if key_conflict(scanned, prev):
            return refuse("host_key_changed", EXIT_HOST_KEY_CHANGED)
        to_add = [k for k in sorted(scanned) if k in prev and k not in pinned]
        action = "repinned" if to_add else "unchanged"
    else:
        to_add = [k for k in sorted(scanned) if k not in pinned]
        action = "added" if to_add else "unchanged"
    if to_add and not args.dry_run:
        pat = host_pattern(host, port)
        append_known_hosts(args.known_hosts, raw, [f"{pat} {kt} {blob}" for kt, blob in to_add])
    out["action"] = action
    emit(out)
    return EXIT_OK


# ── endpoint resolution ──────────────────────────────────────────────────────

SSH_OPTS_WITH_ARG = frozenset(("-B", "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L",
                               "-l", "-m", "-O", "-o", "-Q", "-R", "-S", "-W", "-w"))


def endpoint_from_ssh_cmd(cmd: str) -> tuple[str, int] | None:
    try:
        toks = shlex.split(cmd)
    except ValueError:
        return None
    if not toks or os.path.basename(toks[0]) != "ssh":
        return None
    host, port, i = None, 22, 1
    while i < len(toks):
        tok = toks[i]
        if tok == "-p" and i + 1 < len(toks):
            port, i = toks[i + 1], i + 2
            continue
        if tok.startswith("-p") and len(tok) > 2:
            port, i = tok[2:], i + 1
            continue
        if tok in SSH_OPTS_WITH_ARG:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if host is None:
            host = tok.rpartition("@")[2]
        i += 1
    if not host or not HOST_RE.fullmatch(host):
        return None
    try:
        return host, parse_port(port)
    except UsageError:
        return None


def pod_endpoint(pod: dict) -> tuple[str, int] | None:
    if isinstance(pod.get("ssh_cmd"), str) and pod["ssh_cmd"].strip():
        return endpoint_from_ssh_cmd(pod["ssh_cmd"])
    host, port = pod.get("host"), pod.get("ssh_port")
    if isinstance(host, str) and HOST_RE.fullmatch(host) and port is not None:
        try:
            return host, parse_port(port)
        except UsageError:
            return None
    return None


def cmd_resolve(args) -> int:
    chost, cport = parse_endpoint(args.configured)
    if args.dry_run:
        emit({"dry_run": True, "would_run": [LIUM_PS_CMD]})
        return EXIT_OK
    if not args.lium_ps:
        raise UsageError("resolve-endpoint needs --lium-ps (or --dry-run)")
    if bool(args.scanned) != bool(args.known_hosts):
        raise UsageError("--scanned and --known-hosts go together")
    pods = read_json(args.lium_ps, "lium ps capture")
    if not isinstance(pods, list) or not all(isinstance(p, dict) for p in pods):
        raise UsageError("the lium ps capture must be a JSON array of pod objects")
    matches = [p for p in pods if args.pod in (p.get("name"), p.get("huid"))]
    out = {"verdict": None, "host": None, "port": None, "endpoint_changed": None,
           "previous": f"{chost}:{cport}", "fingerprints": [], "pod": None}
    if len(matches) > 1:
        raise UsageError(f"{len(matches)} pods answer to {args.pod!r}: name it by huid")
    if not matches:
        out["verdict"] = "not_found"
        emit(out)
        return RESOLVE_CODES["not_found"]
    pod = matches[0]
    out["pod"] = {"name": pod.get("name"), "huid": pod.get("huid"), "status": pod.get("status")}
    ep = pod_endpoint(pod)
    if ep:
        out["host"], out["port"] = ep
        out["endpoint_changed"] = ep != (chost, cport)
    if str(pod.get("status") or "").upper() != "RUNNING":
        out["verdict"] = "not_running"
        emit(out)
        return RESOLVE_CODES["not_running"]
    if ep is None:
        raise UsageError(f"pod {args.pod!r} is RUNNING but its lium ps record has no "
                         "parseable ssh endpoint")
    host, port = ep
    moved = out["endpoint_changed"]
    if args.scanned:
        scanned = scanned_keys(args.scanned, host, port)
        out["fingerprints"] = sorted(set(scanned.values()))
        entries = parse_key_lines(read_known_hosts(args.known_hosts).decode("utf-8", "replace"))
        pinned = keys_for(entries, chost, cport)
        if pinned and key_conflict(scanned, pinned):
            verdict = "endpoint_key_changed"
        elif not pinned and moved:
            verdict = "endpoint_changed_unverified"
        else:
            verdict = "same_pod_repin" if moved else "same_endpoint"
    else:
        verdict = "endpoint_changed_unverified" if moved else "same_endpoint"
    out["verdict"] = verdict
    if args.endpoint_out and verdict in ("same_endpoint", "same_pod_repin",
                                         "endpoint_changed_unverified"):
        Path(args.endpoint_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.endpoint_out).write_text(f"{host} {port}\n", encoding="utf-8")
    emit(out)
    return RESOLVE_CODES[verdict]


# ── sync ─────────────────────────────────────────────────────────────────────


def is_secret_name(name: str) -> bool:
    low = name.lower()
    return any(rx.search(low) for rx in SECRET_NAME_RES)


def release_files(release: Path) -> list[str]:
    for name in ROOT_REQUIRED:
        p = release / name
        if p.is_symlink() or not p.is_file():
            raise UsageError(f"the release {release} has no regular {name}: refusing to sync "
                             "an unpinned environment")
    files = list(ROOT_REQUIRED)
    for name in ROOT_OPTIONAL:
        p = release / name
        if p.is_file() and not p.is_symlink():
            files.append(name)
    for tree in COPY_TREES:
        top = release / tree
        if top.is_symlink() or not top.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS
                                 and not is_secret_name(d)
                                 and not (Path(dirpath) / d).is_symlink())
            for name in filenames:
                p = Path(dirpath) / name
                if name.endswith((".pyc", ".pyo")) or is_secret_name(name):
                    continue
                try:
                    st = p.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                files.append(p.relative_to(release).as_posix())
    return sorted(set(files))


def ssh_base(port: int, known_hosts: str) -> list[str]:
    return ["ssh", "-p", str(port), "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes"]


def sync_commands(release: Path, user_host: str, port: int, known_hosts: str, dest: str,
                  bench_venv: bool) -> list[list[str]]:
    ssh = ssh_base(port, known_hosts)
    cmds = [["rsync", "-a", "--from0", "--files-from=-", "-e", " ".join(ssh),
             f"{release}/", f"{user_host}:{dest}/"],
            [*ssh, user_host, f"cd {shlex.quote(dest)} && uv sync --frozen"]]
    if bench_venv:
        cmds.append([*ssh, user_host,
                     f"cd {shlex.quote(dest + '/benchmarks')} && uv sync --frozen"])
    return cmds


def cmd_sync(args) -> int:
    port = parse_port(args.port)
    if not USER_HOST_RE.fullmatch(args.host):
        raise UsageError(f"--host must be USER@HOST, got {args.host!r}")
    known_hosts = safe_path(args.known_hosts, "--known-hosts")
    dest = safe_path(args.dest, "--dest")
    if not dest.startswith("/"):
        raise UsageError("--dest must be an absolute path on the host")
    release = Path(args.release_dir).resolve()
    if not SAFE_PATH_RE.fullmatch(str(release)):
        raise UsageError("--release-dir must be a plain path (no spaces or quotes)")
    files = release_files(release)
    if args.bench_venv and "benchmarks/uv.lock" not in files:
        raise UsageError("--bench-venv needs benchmarks/uv.lock in the release")
    cmds = sync_commands(release, args.host, port, known_hosts, dest, args.bench_venv)
    for rel in files:
        print(f"COPY {rel}")
    for cmd in cmds:
        print(f"RUN {shlex.join(cmd)}")
    sys.stdout.flush()
    if args.dry_run:
        return EXIT_OK
    if not Path(known_hosts).is_file():
        raise UsageError(f"--known-hosts {known_hosts} does not exist: pin the host key first")
    listing = b"".join(rel.encode() + b"\0" for rel in files)
    for i, cmd in enumerate(cmds):
        if i == 0:
            proc = subprocess.run(cmd, input=listing, check=False)
        else:
            proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, check=False)
        if proc.returncode != 0:
            print(f"FAILED exit {proc.returncode}: {cmd[0]}", file=sys.stderr)
            return EXIT_REMOTE
    print("DONE")
    return EXIT_OK


# ── handshake ────────────────────────────────────────────────────────────────


def _load_w4():
    """W4's ``cascade/eval/lane_d_handshake.py`` from the kit's own checkout,
    loaded by path: it is stdlib-only, while the ``cascade.eval`` package
    import would pull numpy into the owner box's plain python3."""
    path = REPO_ROOT / "cascade" / "eval" / "lane_d_handshake.py"
    name = "cascade.eval.lane_d_handshake"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise UsageError(f"W4 handshake module missing at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_suites(profile: dict, identities) -> tuple[dict, list[str], dict | None]:
    if not isinstance(identities, dict) or not isinstance(identities.get("suites"), dict):
        raise UsageError('identities must be {"suites": {...}, "bench_files": N}')
    observed = identities["suites"]
    rows, bad = {}, []
    for suite, pin in sorted(profile["suite_identities"].items()):
        got = observed.get(suite)
        ok = (isinstance(got, str) and len(got) >= 8 and HEX_RE.fullmatch(got) is not None
              and got.startswith(pin))
        rows[suite] = {"pinned": pin, "observed": got, "ok": ok}
        if not ok:
            bad.append(suite)
    bench = None
    if profile.get("bench_files") is not None:
        got = identities.get("bench_files")
        ok = isinstance(got, int) and not isinstance(got, bool) and got == profile["bench_files"]
        bench = {"pinned": profile["bench_files"], "observed": got, "ok": ok}
        if not ok:
            bad.append("bench_files")
    return rows, bad, bench


def check_code(hello, release_dir: str) -> dict:
    w4 = _load_w4()
    try:
        expected = w4.local_identity(release_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        raise UsageError(f"--release-dir {release_dir}: {e}") from None
    try:
        w4.check_handshake(hello, expected=expected)
    except w4.HandshakeMismatch as e:
        return {"ok": False, "fields": list(e.fields)}
    return {"ok": True, "fields": []}


def handshake(profile: dict, identities, hello, release_dir: str | None,
              *, have_identities: bool, have_hello: bool) -> tuple[dict, int]:
    pins = profile["suite_identities"]
    if pins and not have_identities:
        raise UsageError("this profile pins suite identities: --identities is required")
    if not pins and not have_hello:
        raise UsageError("this profile pins no suite identities: the code handshake "
                         "(--hello with --release-dir) is required")
    if have_hello and not release_dir:
        raise UsageError("--hello needs --release-dir (the release it must match)")
    suites, bad, bench = ({}, [], None)
    if pins or (have_identities and profile.get("bench_files") is not None):
        suites, bad, bench = check_suites(profile, identities)
    code = check_code(hello, release_dir) if have_hello else None
    ok = not bad and (code is None or code["ok"])
    out = {"ok": ok, "mismatched": sorted(bad), "suites": suites, "bench_files": bench,
           "code": code}
    rc = EXIT_SUITE_MISMATCH if bad else EXIT_CODE_MISMATCH if code and not code["ok"] else 0
    return out, rc


def handshake_pod_commands(profile: dict) -> list[str]:
    cmds = []
    if profile["suite_identities"]:
        cmds += [BENCH_IDENTITY_CMD.format(bench_dir=profile["bench_dir"]),
                 BENCH_FILES_CMD.format(bench_dir=profile["bench_dir"])]
    return cmds + [HELLO_CMD]


def cmd_handshake(args) -> int:
    profile = load_profile(args.profile)
    if args.dry_run:
        emit({"dry_run": True, "would_run": handshake_pod_commands(profile)})
        return EXIT_OK
    identities = read_json(args.identities, "identities") if args.identities else None
    hello = read_json(args.hello, "worker hello") if args.hello else None
    out, rc = handshake(profile, identities, hello, args.release_dir,
                        have_identities=bool(args.identities), have_hello=bool(args.hello))
    emit(out)
    return rc


def cmd_identities(args) -> int:
    if args.dry_run:
        emit({"dry_run": True, "would_write": args.out})
        return EXIT_OK
    revisions = read_json(args.bench_identity, "bench_identity capture")
    if revisions is None:
        revisions = {}
    if not isinstance(revisions, dict):
        raise UsageError("the bench_identity capture must be a JSON object or null")
    count = read_text(args.bench_files, "bench file count").strip()
    if not count.isdigit():
        raise UsageError("the bench file count capture must be one integer")
    suites = {SUITE_ALIASES.get(k, k): v for k, v in sorted(revisions.items())
              if isinstance(v, str)}
    out = {"suites": suites, "bench_files": int(count)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, sort_keys=True) + "\n", encoding="utf-8")
    emit(out)
    return EXIT_OK


# ── evidence ─────────────────────────────────────────────────────────────────

CAPTURE_FILES = {"nvidia_smi_L": "nvidia_smi_L.txt", "nvidia_smi_query": "nvidia_smi_query.txt",
                 "df": "df.txt", "ssh_keyscan": "ssh_keyscan.txt"}
OPTIONAL_CAPTURES = {"identities": "identities.json", "hello": "hello.json"}


def _git(*args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                          check=False, env=env, timeout=60)
    if proc.returncode != 0:
        raise UsageError(f"git {args[0]} failed in {REPO_ROOT}: the kit must run from a checkout")
    return proc.stdout


def repo_relative_profile(path: str) -> str:
    p = Path(path).resolve()
    try:
        rel = p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        raise UsageError("evidence needs a committed profile under scripts/evalhost/profiles/ "
                         "of the kit's checkout") from None
    if not rel.startswith("scripts/evalhost/profiles/"):
        raise UsageError(f"evidence profile {rel} is not under scripts/evalhost/profiles/")
    return rel


def cmd_evidence(args) -> int:
    profile = load_profile(args.profile)
    rel_profile = repo_relative_profile(args.profile)
    port = parse_port(args.port)
    if not HOST_RE.fullmatch(args.host):
        raise UsageError(f"--host {args.host!r} is not a host name or address")
    caps_dir = Path(args.captures)
    caps = {k: read_text(str(caps_dir / f), "capture") for k, f in CAPTURE_FILES.items()}
    for k, f in OPTIONAL_CAPTURES.items():
        p = caps_dir / f
        caps[k] = read_json(str(p), k) if p.is_file() else None
    if any("PRIVATE KEY" in v for v in caps.values() if isinstance(v, str)):
        raise UsageError("a capture holds private key material: refusing to record it")
    on_rent_out, _ = on_rent(profile, caps["nvidia_smi_L"], caps["nvidia_smi_query"], caps["df"])
    fps = sorted(set(keys_for(parse_key_lines(caps["ssh_keyscan"]), args.host, port).values()))
    hs_out, hs_rc = handshake(profile, caps["identities"], caps["hello"], args.release_dir,
                              have_identities=(caps_dir / "identities.json").is_file(),
                              have_hello=(caps_dir / "hello.json").is_file())
    passed = on_rent_out["ok"] and bool(fps) and hs_out["ok"] and hs_rc == 0
    sha = _git("rev-parse", "HEAD").strip()
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no").strip())
    evidence = {
        "schema": EVIDENCE_SCHEMA, "wp": WP, "source": "pod",
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool": {"git_sha": sha, "git_dirty": dirty},
        "profile": rel_profile,
        "profile_sha256": hashlib.sha256(Path(args.profile).read_bytes()).hexdigest(),
        "pod": {"name": args.pod, "host": args.host, "port": port},
        "captures": caps,
        "on_rent": on_rent_out,
        "host_key": {"fingerprints": fps},
        "handshake": hs_out,
        "passed": passed,
    }
    if not args.dry_run:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit(evidence)
    return EXIT_OK if passed else EXIT_NOT_PASSED


# ── profile-info ─────────────────────────────────────────────────────────────

PROFILE_INFO_KEYS = ("name", "gpu_model", "disk_path", "bench_dir", "suite_pins",
                     "smi_list_cmd", "smi_query_cmd", "df_cmd", "bench_identity_cmd",
                     "bench_files_cmd", "hello_cmd")


def cmd_profile_info(args) -> int:
    profile = load_profile(args.profile)
    info = {"name": profile["name"], "gpu_model": profile["gpu_model"],
            "disk_path": profile["disk_path"], "bench_dir": profile["bench_dir"],
            "suite_pins": "1" if profile["suite_identities"] else "0",
            "smi_list_cmd": SMI_LIST_CMD, "smi_query_cmd": SMI_QUERY_CMD,
            "df_cmd": df_cmd(profile["disk_path"]),
            "bench_identity_cmd": BENCH_IDENTITY_CMD.format(bench_dir=profile["bench_dir"]),
            "bench_files_cmd": BENCH_FILES_CMD.format(bench_dir=profile["bench_dir"]),
            "hello_cmd": HELLO_CMD}
    if args.get:
        print(info[args.get])
    else:
        emit(info)
    return EXIT_OK


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalhost.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("on-rent-check", help="GPU / VRAM / driver / disk vs the profile")
    p.add_argument("--profile", required=True)
    p.add_argument("--nvidia-smi-L", dest="nvidia_smi_L")
    p.add_argument("--nvidia-smi-query", dest="nvidia_smi_query")
    p.add_argument("--df")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_on_rent)

    p = sub.add_parser("pin-host-key", help="pin a scanned host key; refuse a changed one")
    p.add_argument("--known-hosts", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--scanned", required=True)
    p.add_argument("--fingerprint")
    p.add_argument("--previous", help="HOST:PORT the same pod was pinned under (Lium NAT move)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_pin_host_key)

    p = sub.add_parser("resolve-endpoint", help="find the pod in captured lium ps JSON")
    p.add_argument("--lium-ps")
    p.add_argument("--pod", required=True)
    p.add_argument("--configured", required=True)
    p.add_argument("--known-hosts")
    p.add_argument("--scanned")
    p.add_argument("--endpoint-out", help="write `HOST PORT` here for a RUNNING pod")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_resolve)

    p = sub.add_parser("sync", help="copy the release and uv sync --frozen on the host")
    p.add_argument("--release-dir", required=True)
    p.add_argument("--host", required=True, help="USER@HOST")
    p.add_argument("--port", required=True)
    p.add_argument("--known-hosts", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--bench-venv", action="store_true",
                   help="also uv sync --frozen the benchmarks/ sidecar project")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("identities", help="fold bench_identity + file count captures")
    p.add_argument("--bench-identity", required=True)
    p.add_argument("--bench-files", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_identities)

    p = sub.add_parser("handshake", help="suite identities + the W4 code handshake")
    p.add_argument("--profile", required=True)
    p.add_argument("--identities")
    p.add_argument("--hello")
    p.add_argument("--release-dir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_handshake)

    p = sub.add_parser("evidence", help="re-run every check; write laned_r12_pod/1 evidence")
    p.add_argument("--profile", required=True)
    p.add_argument("--pod", required=True)
    p.add_argument("--host", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--captures", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--release-dir")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_evidence)

    p = sub.add_parser("profile-info", help="print profile values for the owner scripts")
    p.add_argument("--profile", required=True)
    p.add_argument("--get", choices=PROFILE_INFO_KEYS)
    p.set_defaults(fn=cmd_profile_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except UsageError as e:
        print(f"evalhost: {e}", file=sys.stderr)
        if args.cmd != "sync":
            emit({"ok": False, "error": str(e)})
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
