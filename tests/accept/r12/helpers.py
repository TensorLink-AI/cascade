"""Shared helpers for the R12 acceptance tests (eval-host provisioning kit).

Everything here is offline. The kit is driven as subprocesses only:

* ``python -P scripts/evalhost/evalhost.py <subcommand> ...`` over CAPTURED
  pod output (``nvidia-smi``, ``df``, ``ssh-keyscan``, ``lium ps``, the
  suite identities and the worker hello) written into ``tmp_path``;
* ``bash scripts/evalhost/{setup,recover}_eval_host.sh`` in ``--dry-run`` /
  ``--print`` mode only.

Every command that could reach a pod, a chain or a cloud API is put first on
``PATH`` as a shim that records its argv and exits 97, and ``torch`` and
``bittensor`` are made unimportable, so a kit step that tried to touch the
outside world (or load torch on the 7 GB box) fails loudly. Secret env vars
are only ever set to sentinel strings, which must never appear in output.

:func:`evidence_problems` is the ``evidence/r12_pod.json`` verifier: it
re-derives every recorded verdict from the recorded captures with the kit
itself and the profile as committed at ``tool.git_sha``. Contract: README.md.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import io
import json
import os
import re
import struct
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
KIT = REPO / "scripts" / "evalhost"
TOOL = KIT / "evalhost.py"
SETUP = KIT / "setup_eval_host.sh"
RECOVER = KIT / "recover_eval_host.sh"
PROFILES = KIT / "profiles"
GATE_PROFILE = PROFILES / "bench_gate.toml"
LAUNCH_PROFILE = PROFILES / "launch_eval.toml"

PROFILE_SCHEMA = "laned_evalhost_profile/1"
EVIDENCE_SCHEMA = "laned_r12_pod/1"
WP = "R12"

# Real history (targon-eval.toml header, OPSLOG 2026-11-04 / 11-12 / 11-22).
PINNED_IDENTITIES = {"gift": "30841734", "boom": "69325b54", "time": "83e3d0b3",
                     "fev": "6de245af"}
BENCH_FILES = 17747
BLACKWELL = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
BLACKWELL_MIB = 97887
ADA = "NVIDIA RTX 6000 Ada Generation"
ADA_MIB = 49140
RTX4090 = "NVIDIA GeForce RTX 4090"
RTX4090_MIB = 24564
TWO_CHECKPOINTS_BYTES = 9_200_000_000

# Typed exit codes (README "Exit codes").
EXIT = {
    "usage": 2,
    "wrong_gpu": 10, "vram_low": 11, "driver_low": 12, "disk_low": 13,
    "host_key_changed": 20, "fingerprint_mismatch": 21,
    "same_pod_repin": 30, "endpoint_key_changed": 31, "not_found": 32,
    "not_running": 33, "endpoint_changed_unverified": 34,
    "suite_identity_mismatch": 40, "code_mismatch": 41,
}
ON_RENT_ORDER = ("wrong_gpu", "vram_low", "driver_low", "disk_low")

SMI_QUERY = "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits"

SHIMMED = ("ssh", "scp", "rsync", "sftp", "ssh-keyscan", "lium", "nvidia-smi", "uv",
           "docker", "curl", "wget", "btcli", "sudo", "systemctl", "nc")
SECRET_ENV = {
    "HIPPIUS_S3_ACCESS_KEY": "r12-sentinel-hippius-access-7c41",
    "HIPPIUS_S3_SECRET_KEY": "r12-sentinel-hippius-secret-9a03",
    "HF_TOKEN": "r12-sentinel-hf-token-55e2",
    "WANDB_API_KEY": "r12-sentinel-wandb-31bd",
    "LIUM_API_KEY": "r12-sentinel-lium-key-0f6a",
    "BT_WALLET_PASSWORD": "r12-sentinel-wallet-pw-e81c",
    "CASCADE_OPS_WEBHOOK_URL": "https://r12-sentinel.invalid/hook-4d2f",
}
FORBIDDEN_TEXT = (
    re.compile(r"StrictHostKeyChecking\s*=\s*(no|accept-new|off)\b", re.I),
    re.compile(r"UserKnownHostsFile\s*=\s*/dev/null"),
    re.compile(r"\bset\s+-[a-wyz]*x"),
    re.compile(r"\bset\s+-o\s+xtrace\b"),
    re.compile(r"\bprintenv\b"),
    re.compile(r"\bcat\s+\S*\.env\b"),
    re.compile(r"BT_WALLET_PASSWORD"),
    re.compile(r"coldkey", re.I),
    re.compile(r"\.bittensor/wallets"),
    re.compile(r"PRIVATE KEY"),
)
REMOTE_RE = re.compile(r"(^|[\s'\"(=])(ssh|scp|rsync)(\s|$)")
SHA_RE = re.compile(r"[0-9a-f]{40}")
FP_RE = re.compile(r"SHA256:[A-Za-z0-9+/]{43}")

OLD_HOST, NEW_HOST, PORT = "198.51.100.7", "203.0.113.9", 41100


# ── generic subprocess plumbing ──────────────────────────────────────────────


def require(path: Path) -> None:
    rel = path.relative_to(REPO)
    assert path.is_file(), f"R12 kit file {rel} is missing (R12 PATHS, not built yet)"


def output(proc: subprocess.CompletedProcess) -> str:
    return f"exit {proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"


def as_json(proc: subprocess.CompletedProcess) -> dict:
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise AssertionError(f"stdout is not one JSON object ({e})\n{output(proc)}") from None
    assert isinstance(data, dict), output(proc)
    return data


def tree_state(root: Path) -> dict:
    """Every path under ``root`` with its bytes (files) or link target."""
    state = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if rel.startswith("shims") or rel.startswith("no_import"):
            continue
        if p.is_symlink():
            state[rel] = ("link", os.readlink(p))
        elif p.is_file():
            state[rel] = ("file", p.read_bytes())
        else:
            state[rel] = ("dir", None)
    return state


class Kit:
    """One sandbox per test: shims, poisoned imports, sentinel secrets."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.shims = tmp_path / "shims"
        self.calls = tmp_path / "shims" / "calls.log"
        self.shims.mkdir(parents=True)
        for name in SHIMMED:
            p = self.shims / name
            p.write_text("#!/bin/sh\n"
                         f"printf '%s\\n' \"{name} $*\" >> '{self.calls}'\n"
                         "exit 97\n")
            p.chmod(0o755)
        poison = tmp_path / "no_import"
        for mod in ("torch", "bittensor"):
            (poison / mod).mkdir(parents=True)
            (poison / mod / "__init__.py").write_text(
                f"raise ImportError('R12 test: {mod} is blocked; the kit must not import it')\n")
        self.poison = poison
        self.home = tmp_path / "home"
        self.home.mkdir()

    def env(self) -> dict:
        env = os.environ.copy()
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "SSH_AUTH_SOCK"):
            env.pop(k, None)
        env.update(SECRET_ENV)
        env["PATH"] = str(self.shims) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = os.pathsep.join([str(self.poison), str(REPO)])
        env["HOME"] = str(self.home)
        return env

    def shim_calls(self) -> list[str]:
        return self.calls.read_text().splitlines() if self.calls.is_file() else []

    def tool(self, *args: str) -> subprocess.CompletedProcess:
        require(TOOL)
        return subprocess.run([sys.executable, "-P", str(TOOL), *map(str, args)],
                              cwd=self.tmp, env=self.env(), capture_output=True, text=True,
                              timeout=120, check=False)

    def script(self, path: Path, *args: str) -> subprocess.CompletedProcess:
        require(path)
        return subprocess.run(["bash", str(path), *map(str, args)], cwd=self.tmp,
                              env=self.env(), capture_output=True, text=True, timeout=120,
                              check=False)

    def write(self, name: str, text: str) -> Path:
        p = self.tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    # ── profiles and captures ──

    def profile(self, name: str = "profile.toml", **over) -> Path:
        return self.write(name, profile_text(**over))

    def captures(self, *, model=BLACKWELL, vram=BLACKWELL_MIB, driver="595.99.02",
                 avail=14_800_000_000, n=1, models=None, query_models=None,
                 tag: str = "cap") -> dict:
        names = list(models) if models is not None else [model] * n
        qnames = list(query_models) if query_models is not None else names
        return {
            "L": self.write(f"{tag}/nvidia_smi_L.txt", smi_list(names)),
            "query": self.write(f"{tag}/nvidia_smi_query.txt", smi_query(qnames, vram, driver)),
            "df": self.write(f"{tag}/df.txt", df_text(avail)),
        }

    def on_rent(self, profile: Path, caps: dict) -> subprocess.CompletedProcess:
        return self.tool("on-rent-check", "--profile", profile, "--nvidia-smi-L", caps["L"],
                         "--nvidia-smi-query", caps["query"], "--df", caps["df"])


def profile_text(*, schema=PROFILE_SCHEMA, name="test_gate", gpu_model=BLACKWELL,
                 min_vram_mib=97000, min_driver="580", disk_path="/data",
                 min_disk_free_bytes=TWO_CHECKPOINTS_BYTES, bench_files=BENCH_FILES,
                 suite_identities=PINNED_IDENTITIES) -> str:
    lines = [f'schema = "{schema}"', f'name = "{name}"', f'gpu_model = "{gpu_model}"',
             f"min_vram_mib = {min_vram_mib}", f'min_driver = "{min_driver}"',
             f'disk_path = "{disk_path}"', f"min_disk_free_bytes = {min_disk_free_bytes}"]
    if bench_files is not None:
        lines.append(f"bench_files = {bench_files}")
    if suite_identities:
        lines.append("")
        lines.append("[suite_identities]")
        lines += [f'{k} = "{v}"' for k, v in suite_identities.items()]
    return "\n".join(lines) + "\n"


def smi_list(names: list[str]) -> str:
    return "".join(f"GPU {i}: {n} (UUID: GPU-5f0c1a2b-7d3e-4c81-9a6f-0e2d4b8c{i:04d})\n"
                   for i, n in enumerate(names))


def smi_query(names: list[str], vram, driver: str) -> str:
    vrams = vram if isinstance(vram, list) else [vram] * len(names)
    return "".join(f"{n}, {v}, {driver}\n" for n, v in zip(names, vrams, strict=True))


def df_text(avail, target: str = "/data") -> str:
    return f"          Avail Mounted on\n{avail} {target}\n"


def reason_codes(out: dict) -> list[str]:
    reasons = out.get("reasons")
    assert isinstance(reasons, list), out
    for r in reasons:
        assert isinstance(r, dict) and isinstance(r.get("code"), str), out
        assert isinstance(r.get("detail"), str) and r["detail"].strip(), (
            f"every refusal carries a non-empty detail: {r}")
    return [r["code"] for r in reasons]


def assert_no_secrets(text: str) -> None:
    for k, v in SECRET_ENV.items():
        assert v not in text, f"the value of ${k} leaked into the output"


def assert_safe_text(text: str, *, what: str) -> None:
    assert_no_secrets(text)
    for rx in FORBIDDEN_TEXT:
        m = rx.search(text)
        assert m is None, f"{what}: forbidden text {m.group(0)!r}"


def assert_remote_pinned(text: str, known_hosts: Path, *, what: str) -> int:
    """Every ssh/scp/rsync command line pins the host key; returns the count."""
    n = 0
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or not REMOTE_RE.search(s):
            continue
        n += 1
        assert f"UserKnownHostsFile={known_hosts}" in s, (
            f"{what}: remote command does not pin the known_hosts file: {s}")
        assert "StrictHostKeyChecking=yes" in s, (
            f"{what}: remote command without StrictHostKeyChecking=yes: {s}")
    return n


# ── host keys ────────────────────────────────────────────────────────────────


def key_blob(seed: str) -> str:
    """A well-formed ssh-ed25519 public key blob (base64), deterministic."""
    raw = hashlib.sha256(seed.encode()).digest()
    kt = b"ssh-ed25519"
    blob = struct.pack(">I", len(kt)) + kt + struct.pack(">I", len(raw)) + raw
    return base64.b64encode(blob).decode()


def fingerprint(blob_b64: str) -> str:
    digest = hashlib.sha256(base64.b64decode(blob_b64)).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def host_pattern(host: str, port: int) -> str:
    return host if int(port) == 22 else f"[{host}]:{port}"


def known_line(host: str, port: int, blob: str) -> str:
    return f"{host_pattern(host, port)} ssh-ed25519 {blob}"


def hashed_line(host: str, port: int, blob: str, salt: bytes = b"r12-salt-0123456789") -> str:
    salt = hashlib.sha1(salt).digest()
    mac = hmac.new(salt, host_pattern(host, port).encode(), hashlib.sha1).digest()
    return (f"|1|{base64.b64encode(salt).decode()}|{base64.b64encode(mac).decode()} "
            f"ssh-ed25519 {blob}")


def keyscan_text(host: str, port: int, blob: str) -> str:
    return f"# {host}:{port} SSH-2.0-OpenSSH_9.6p1 Ubuntu-3ubuntu13\n{known_line(host, port, blob)}\n"


def pod_entry(*, huid="lunar-wolf-6a", name="valeval-blackwell4", status="RUNNING",
              host=NEW_HOST, port=PORT, form="ssh_cmd") -> dict:
    pod = {"huid": huid, "name": name, "status": status, "gpu": "1x RTX PRO 6000 Blackwell",
           "price_per_hour": 1.48}
    if form == "ssh_cmd":
        pod["ssh_cmd"] = f"ssh root@{host} -p {port}"
    else:
        pod["host"] = host
        pod["ssh_port"] = port
    return pod


# ── release dirs ─────────────────────────────────────────────────────────────


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=r12", "-c",
                           "user.email=r12@test.invalid", "-c", "commit.gpgsign=false", *args],
                          capture_output=True, text=True, check=True, env=env)


def make_git_release(root: Path) -> Path:
    """A tiny release checkout: a git repo with a cascade/ package."""
    (root / "cascade" / "eval").mkdir(parents=True)
    (root / "cascade" / "__init__.py").write_text('__version__ = "r12"\n')
    (root / "cascade" / "eval" / "__init__.py").write_text("")
    (root / "cascade" / "eval" / "worker.py").write_text("def forecast():\n    return 1\n")
    (root / "pyproject.toml").write_text('[project]\nname = "cascade"\nversion = "0"\n')
    (root / "uv.lock").write_text("version = 1\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "release")
    return root


# ── the evidence verifier ────────────────────────────────────────────────────

TOP_KEYS = ("schema", "wp", "source", "checked_at", "tool", "profile", "profile_sha256",
            "pod", "captures", "on_rent", "host_key", "handshake", "passed")
CAPTURE_KEYS = ("nvidia_smi_L", "nvidia_smi_query", "df", "ssh_keyscan", "identities", "hello")


def _repo_git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, check=False)


def _scanned_fingerprints(text: str, host: str, port: int) -> list[str]:
    pat = host_pattern(host, port)
    fps = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and not line.lstrip().startswith("#") and pat in parts[0].split(","):
            try:
                fps.append(fingerprint(parts[2]))
            except (ValueError, base64.binascii.Error):
                continue
    return sorted(set(fps))


def _package_hash_at(sha: str) -> str | None:
    """W4's package hash of the ``cascade/`` tree committed at ``sha``."""
    from cascade.eval.lane_d_handshake import package_hash

    proc = _repo_git("archive", "--format=tar", sha, "cascade")
    if proc.returncode != 0:
        return None
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
            tf.extractall(d, filter="data")
        return package_hash(d)


def _iso_with_offset(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return datetime.fromisoformat(value).utcoffset() is not None
    except ValueError:
        return False


def _rerun_on_rent(kit: Kit, profile: Path, caps: dict) -> dict:
    files = {"L": kit.write("verify/L.txt", caps["nvidia_smi_L"]),
             "query": kit.write("verify/query.txt", caps["nvidia_smi_query"]),
             "df": kit.write("verify/df.txt", caps["df"])}
    proc = kit.on_rent(profile, files)
    out = as_json(proc)
    out["_rc"] = proc.returncode
    return out


def _rerun_identities(kit: Kit, profile: Path, identities: dict) -> dict:
    ids = kit.write("verify/identities.json", json.dumps(identities))
    proc = kit.tool("handshake", "--profile", profile, "--identities", ids)
    out = as_json(proc)
    out["_rc"] = proc.returncode
    return out


def evidence_problems(data: object, *, check_git: bool = True,
                      work: Path | None = None) -> list[str]:
    """Every reason ``data`` is not acceptable R12 live-run evidence."""
    if not isinstance(data, dict):
        return ["evidence must be a JSON object"]
    probs = [f"missing key {k!r}" for k in TOP_KEYS if k not in data]
    if probs:
        return probs
    text = json.dumps(data)
    for k, v in SECRET_ENV.items():
        if v in text or (os.environ.get(k) and os.environ[k] in text):
            probs.append(f"the value of ${k} appears in the evidence")
    if "PRIVATE KEY" in text:
        probs.append("a private key appears in the evidence")
    if data["schema"] != EVIDENCE_SCHEMA:
        probs.append(f"schema: want {EVIDENCE_SCHEMA!r}, got {data['schema']!r}")
    if data["wp"] != WP:
        probs.append(f"wp: want {WP!r}, got {data['wp']!r}")
    if data["source"] != "pod":
        probs.append(f"source: want 'pod', got {data['source']!r}")
    if not _iso_with_offset(data["checked_at"]):
        probs.append(f"checked_at: want ISO 8601 with an offset, got {data['checked_at']!r}")

    tool = data["tool"]
    sha = tool.get("git_sha") if isinstance(tool, dict) else None
    if not (isinstance(sha, str) and SHA_RE.fullmatch(sha)):
        return probs + [f"tool.git_sha: want 40 lowercase hex, got {sha!r}"]
    if tool.get("git_dirty") is not False:
        probs.append(f"tool.git_dirty: the run must be from a clean checkout, got {tool.get('git_dirty')!r}")
    if check_git:
        if _repo_git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            return probs + [f"tool.git_sha {sha} is not a commit in this repository"]
        if _repo_git("cat-file", "-e", f"{sha}:scripts/evalhost/evalhost.py").returncode != 0:
            return probs + [f"tool.git_sha {sha} has no scripts/evalhost/evalhost.py"]

    prof = data["profile"]
    if not (isinstance(prof, str) and prof.startswith("scripts/evalhost/profiles/")
            and prof.endswith(".toml") and ".." not in prof):
        return probs + [f"profile: want a repo path under scripts/evalhost/profiles/, got {prof!r}"]
    shown = _repo_git("show", f"{sha}:{prof}")
    if shown.returncode != 0:
        return probs + [f"profile {prof} does not exist at {sha}"]
    if data["profile_sha256"] != hashlib.sha256(shown.stdout).hexdigest():
        probs.append("profile_sha256 does not match the profile committed at tool.git_sha")
    try:
        prof_data = tomllib.loads(shown.stdout.decode())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return probs + [f"profile {prof} at {sha} does not parse: {e}"]

    pod = data["pod"]
    if not (isinstance(pod, dict) and isinstance(pod.get("name"), str) and pod["name"]
            and isinstance(pod.get("host"), str) and pod["host"]
            and isinstance(pod.get("port"), int) and 0 < pod["port"] < 65536):
        return probs + [f"pod: want {{name, host, port}}, got {pod!r}"]
    caps = data["captures"]
    if not isinstance(caps, dict) or any(k not in caps for k in CAPTURE_KEYS):
        return probs + [f"captures: want keys {CAPTURE_KEYS}"]
    for k in ("nvidia_smi_L", "nvidia_smi_query", "df", "ssh_keyscan"):
        if not isinstance(caps[k], str) or not caps[k].strip():
            probs.append(f"captures.{k}: want the captured text")
    if probs:
        return probs

    tmp_ctx = tempfile.TemporaryDirectory() if work is None else None
    base = Path(tmp_ctx.name) if tmp_ctx else work
    try:
        kit = Kit(base / "verifier")
    except FileExistsError:
        kit = Kit(Path(tempfile.mkdtemp(dir=base)))
    try:
        profile_file = kit.write("verify/profile.toml", shown.stdout.decode())
        # on-rent: re-derived from the captures
        rerun = _rerun_on_rent(kit, profile_file, caps)
        rec = data["on_rent"]
        if not isinstance(rec, dict):
            probs.append("on_rent: want the on-rent-check JSON")
        else:
            if rec.get("ok") is not rerun.get("ok"):
                probs.append(f"on_rent.ok {rec.get('ok')!r} but the captures give {rerun.get('ok')!r}")
            want = [r.get("code") for r in rerun.get("reasons", [])]
            got = [r.get("code") for r in rec.get("reasons", []) if isinstance(r, dict)]
            if got != want:
                probs.append(f"on_rent.reasons {got} but the captures give {want}")
            if rec.get("observed") != rerun.get("observed"):
                probs.append("on_rent.observed does not match the captures")
        on_rent_ok = rerun.get("ok") is True and rerun["_rc"] == 0

        # host key: the scanned fingerprints for the pod's endpoint
        fps = _scanned_fingerprints(caps["ssh_keyscan"], pod["host"], pod["port"])
        hk = data["host_key"]
        if not isinstance(hk, dict) or sorted(hk.get("fingerprints") or []) != fps:
            probs.append(f"host_key.fingerprints do not match the scanned keys {fps}")
        key_ok = bool(fps)
        if not fps:
            probs.append("ssh_keyscan holds no key for the pod's host:port")

        # handshake
        hs = data["handshake"]
        if not isinstance(hs, dict):
            return probs + ["handshake: want the handshake JSON"]
        pins = prof_data.get("suite_identities") or {}
        ids_ok = True
        if pins:
            if not isinstance(caps["identities"], dict):
                probs.append("captures.identities: the profile pins suite identities")
                ids_ok = False
            else:
                rh = _rerun_identities(kit, profile_file, caps["identities"])
                ids_ok = rh.get("ok") is True and rh["_rc"] == 0
                if sorted(hs.get("mismatched") or []) != sorted(rh.get("mismatched") or []):
                    probs.append(f"handshake.mismatched {hs.get('mismatched')} but the captures "
                                 f"give {rh.get('mismatched')}")
        code_ok = True
        hello = caps["hello"]
        if not pins and not isinstance(hello, dict):
            probs.append("captures.hello: a profile without suite identities needs the worker hello")
            code_ok = False
        if isinstance(hello, dict):
            head = hello.get("git_head")
            if not (isinstance(head, str) and SHA_RE.fullmatch(head)) or (
                    check_git and _repo_git("cat-file", "-e", f"{head}^{{commit}}").returncode != 0):
                probs.append(f"captures.hello.git_head {head!r} is not a commit in this repository")
                code_ok = False
            else:
                want_hash = _package_hash_at(head) if check_git else hello.get("package_hash")
                if hello.get("package_hash") != want_hash:
                    probs.append("captures.hello.package_hash is not the cascade/ package hash "
                                 f"of {head}")
                    code_ok = False
        if hs.get("ok") is not (ids_ok and code_ok):
            probs.append(f"handshake.ok {hs.get('ok')!r} but the captures give {ids_ok and code_ok!r}")

        passed = on_rent_ok and key_ok and ids_ok and code_ok
        if data["passed"] is not passed:
            probs.append(f"passed {data['passed']!r} but re-derived {passed!r}")
        if not passed:
            probs.append("the pod did not pass the on-rent check, host-key capture and handshake")
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()
    return probs


def gate_captures_dir(kit: Kit, *, model=None, vram=BLACKWELL_MIB, driver="595.99.02",
                      avail=14_800_000_000, identities=None, hello=None,
                      host=NEW_HOST, port=PORT, blob=None, tag="pod") -> Path:
    """A captures dir in the shape `evalhost.py evidence --captures` reads."""
    if model is None:
        require(GATE_PROFILE)
        model = tomllib.loads(GATE_PROFILE.read_text())["gpu_model"]
    d = kit.tmp / tag
    kit.write(f"{tag}/nvidia_smi_L.txt", smi_list([model]))
    kit.write(f"{tag}/nvidia_smi_query.txt", smi_query([model], vram, driver))
    kit.write(f"{tag}/df.txt", df_text(avail))
    kit.write(f"{tag}/ssh_keyscan.txt", keyscan_text(host, port, blob or key_blob("pod")))
    if identities is not None:
        kit.write(f"{tag}/identities.json", json.dumps(identities))
    if hello is not None:
        kit.write(f"{tag}/hello.json", json.dumps(hello))
    return d


def as_clean_run(evidence: dict) -> dict:
    """Self-tests run from a dev checkout that may be dirty; a clean release
    checkout records ``git_dirty: false``."""
    ev = copy.deepcopy(evidence)
    ev["tool"]["git_dirty"] = False
    return ev
