"""Live end-to-end suite for the v2 Lane A competition on testnet.

Drives the real mechanism (CLI → intake → chain commitment + timelock reveal →
trainer admission + pod rental → training → settlement manifest → both
validators' receipts + weights → audit) and asserts each step. Stages:

  preflight   free   services, chain, contract, image pin, fingerprint, wallets, keys
  admission   free   local verify: one passing fixture, four rejection classes
  legs        PAID   one funded submission per provider scenario, tracked to terminal
  settlement  free   receipts from every validator agree; weights match; duel/round views
  validator   free   v2 validator outputs: receipt index, chain status, bench_v, ladder
  audit       free   cascade-audit tier 0 on every settled round
  cleanup     free   no testnet pods left beyond the start-of-run baseline

Paid stages run only with --spend. State is kept in --state so a rerun with
--resume keeps tracking the submissions instead of submitting again.

  .venv/bin/python -m tests.e2e_testnet.run --stages preflight,admission
  .venv/bin/python -m tests.e2e_testnet.run --spend            # everything
  .venv/bin/python -m tests.e2e_testnet.run --spend --resume   # continue a run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERATORS = HERE / "generators"
VENV_BIN = ROOT / ".venv" / "bin"
STAGES = ("preflight", "admission", "legs", "settlement", "validator", "audit", "cleanup")
PAID = {"legs"}
TERMINAL_STATUSES = {"done", "failed", "withdrawn", "rejected", "spent", "cancelled"}
WALLETS_ROOT = Path.home() / ".bittensor" / "wallets"

# One funded entry per scenario; each hotkey holds one chain commitment, so each
# scenario owns a miner wallet. "fund" scenarios submit unfunded, then exercise
# `cascade fund` with a GPU choice.
SCENARIOS = (
    {"name": "lium", "wallet": "miner0", "providers": ["lium"], "flow": "submit"},
    {"name": "shadeform", "wallet": "miner1", "providers": ["shadeform"], "flow": "fund",
     "gpu": "RTX4090", "allow_fallback": True},
    {"name": "fallback", "wallet": "miner2", "providers": ["lium", "shadeform"], "flow": "fund",
     "gpu": "L40S", "allow_fallback": False},
    {"name": "vast", "wallet": "miner3", "providers": ["vast"], "flow": "submit"},
    {"name": "runpod", "wallet": "miner4", "providers": ["runpod"], "flow": "fund",
     "gpu": "RTX4090", "max_price": 0, "allow_fallback": False},
)
PROVIDER_ENV = {"lium": "LIUM_API_KEY", "shadeform": "SHADEFORM_API_KEY",
                "vast": "VAST_API_KEY", "runpod": "RUNPOD_API_KEY"}


# ── results ────────────────────────────────────────────────────────────────────

@dataclass
class Check:
    stage: str
    name: str
    status: str  # PASS | FAIL | WARN | SKIP
    detail: str = ""
    data: dict = field(default_factory=dict)
    seconds: float = 0.0


class Report:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.checks: list[Check] = []
        self.started = time.time()

    def add(self, stage: str, name: str, status: str, detail: str = "", t0: float | None = None,
            **data) -> Check:
        c = Check(stage, name, status, detail, data, round(time.time() - t0, 1) if t0 else 0.0)
        self.checks.append(c)
        print(f"[{status:4}] {stage}/{name}: {detail}", flush=True)
        self.save()
        return c

    def check(self, stage: str, name: str, ok: bool, detail: str = "", *, warn: bool = False,
              t0: float | None = None, **data) -> bool:
        self.add(stage, name, "PASS" if ok else ("WARN" if warn else "FAIL"), detail, t0, **data)
        return ok

    def save(self) -> None:
        counts = {s: sum(c.status == s for c in self.checks) for s in ("PASS", "FAIL", "WARN", "SKIP")}
        doc = {"started_at": self.started, "updated_at": time.time(), "counts": counts,
               "checks": [asdict(c) for c in self.checks]}
        self.path.write_text(json.dumps(doc, indent=1, default=str))

    def failed(self) -> int:
        return sum(c.status == "FAIL" for c in self.checks)


# ── helpers ────────────────────────────────────────────────────────────────────

class Ctx:
    def __init__(self, args: argparse.Namespace, report: Report) -> None:
        self.args = args
        self.r = report
        self.toml = str(Path(args.chain_toml).resolve())
        self.state_path = Path(args.state)
        self.state = json.loads(self.state_path.read_text()) if (args.resume and self.state_path.is_file()) else {}
        self._cfg = None
        self._chain = None

    def save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1))

    @property
    def cfg(self):
        if self._cfg is None:
            from cascade.shared.config import load_chain_config
            self._cfg = load_chain_config(self.toml)
        return self._cfg

    @property
    def chain(self):
        if self._chain is None:
            from cascade.shared.chain import ChainClient
            self._chain = ChainClient.from_config(self.cfg, network=self.args.network)
        return self._chain

    def bucket_url(self, key: str) -> str:
        return f"{self.args.s3_base.rstrip('/')}/{self.cfg.storage.manifest_bucket}/{key}"


def run(cmd: list[str], *, timeout: float = 900, env: dict | None = None, cwd: Path = ROOT) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", f"timeout after {timeout}s"


def cli(ctx: Ctx, *argv: str, timeout: float = 900) -> tuple[int, dict | None, str]:
    """Run `cascade <argv> --json`; return (rc, envelope, stderr)."""
    rc, out, err = run([str(VENV_BIN / "cascade"), *argv, "--json"], timeout=timeout)
    doc = None
    for line in reversed(out.strip().splitlines()):
        try:
            doc = json.loads(line)
            break
        except ValueError:
            continue
    if doc is None:
        try:
            doc = json.loads(out)
        except ValueError:
            doc = None
    return rc, doc, err[-2000:]


def http_json(url: str, timeout: float = 30) -> dict | list | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, ValueError, TimeoutError):
        return None


def find_value(obj, pattern: re.Pattern) -> str | None:
    """First string anywhere in obj matching pattern (tolerant of CLI shape drift)."""
    if isinstance(obj, str):
        m = pattern.search(obj)
        return m.group(0) if m else None
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        for v in obj:
            hit = find_value(v, pattern)
            if hit:
                return hit
    return None


def hotkey_of(wallet: str) -> str | None:
    p = WALLETS_ROOT / wallet / "hotkeys" / "default"
    try:
        return json.loads(p.read_text())["ss58Address"]
    except (OSError, ValueError, KeyError):
        return None


def proc_running(pattern: str) -> list[str]:
    rc, out, _ = run(["pgrep", "-af", pattern], timeout=10)
    return [ln for ln in out.splitlines() if "pgrep" not in ln] if rc == 0 else []


def testnet_pods() -> dict[str, list[str]]:
    """Live pods whose names carry the testnet netuid prefix, per provider we can list."""
    netuid_tag = "cascade-n259-"
    pods: dict[str, list[str]] = {}
    if os.environ.get("LIUM_API_KEY"):
        rc, out, _ = run(["lium", "ps", "--format", "json"], timeout=90)
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', out) if rc == 0 else []
        pods["lium"] = sorted(n for n in names if n.startswith(netuid_tag))
    key = os.environ.get("SHADEFORM_API_KEY")
    if key:
        req = urllib.request.Request("https://api.shadeform.ai/v1/instances", headers={"X-API-KEY": key})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                doc = json.loads(r.read().decode())
            pods["shadeform"] = sorted(i.get("name", "") for i in doc.get("instances", [])
                                       if str(i.get("name", "")).startswith(netuid_tag)
                                       and i.get("status") not in ("deleted", "deleting"))
        except (urllib.error.URLError, ValueError, TimeoutError):
            pods["shadeform"] = ["<list failed>"]
    return pods


def ghcr_digest(ref: str) -> str | None:
    m = re.match(r"ghcr\.io/([^:@]+)(?::([^@]+))?(?:@(sha256:[0-9a-f]{64}))?$", ref)
    if not m:
        return None
    repo, tag, digest = m.groups()
    tok = http_json(f"https://ghcr.io/token?scope=repository:{repo}:pull") or {}
    req = urllib.request.Request(
        f"https://ghcr.io/v2/{repo}/manifests/{digest or tag}", method="HEAD",
        headers={"Authorization": f"Bearer {tok.get('token', '')}",
                 "Accept": ", ".join(("application/vnd.oci.image.manifest.v1+json",
                                      "application/vnd.oci.image.index.v1+json",
                                      "application/vnd.docker.distribution.manifest.v2+json"))})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.headers.get("docker-content-digest")
    except urllib.error.URLError:
        return None


def local_worker_fingerprint() -> str:
    import hashlib

    from cascade.provision.codeprint import canonical_fingerprint, parse_sha256sum
    lines = []
    for p in sorted((ROOT / "cascade").rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  /root/cascade/{p.relative_to(ROOT)}")
    return canonical_fingerprint(parse_sha256sum("\n".join(lines), workdir="/root/cascade"))


def offline_digest(ctx: Ctx) -> str | None:
    rc, out, _ = run([str(VENV_BIN / "cascade-trainer"), "--offline", "--chain-toml", ctx.toml,
                      "--network", ctx.args.network], timeout=300)
    m = re.search(r"contract_digest:\s+([0-9a-f]{64})", out)
    return m.group(1) if rc == 0 and m else None


def scenario_generator(ctx: Ctx, sc: dict) -> Path:
    """A per-run copy of the mv32 fixture with a unique salt (unique corpus digest)."""
    run_id = ctx.state.setdefault("run_id", str(int(time.time())))
    d = Path(ctx.args.workdir) / f"gen-{sc['name']}-{run_id}"
    if not d.exists():
        shutil.copytree(GENERATORS / "mv32", d)
        salt = int(run_id) * 16 + SCENARIOS.index(sc)
        (d / "config.json").write_text(json.dumps({"salt": salt}))
    return d


# ── stages ─────────────────────────────────────────────────────────────────────

def stage_preflight(ctx: Ctx) -> None:
    S = "preflight"
    r, cfg = ctx.r, ctx.cfg

    t0 = time.time()
    digest = offline_digest(ctx)
    r.check(S, "config_loads", digest is not None, f"contract_digest={digest}", t0=t0)
    ctx.state["contract_digest"] = digest

    for role, pat in (("trainer", r"cascade\.trainer\.main"), ("validator", r"cascade\.validator\.main"),
                      ("intake", r"cascade\.funding\.main")):
        procs = proc_running(pat)
        need = 2 if role == "validator" else 1
        r.check(S, f"service_{role}", len(procs) >= need, f"{len(procs)} running (need {need})")

    q = http_json(f"{ctx.args.intake.rstrip('/')}/v1/queue", timeout=15)
    r.check(S, "intake_http", q is not None, f"GET {ctx.args.intake}/v1/queue")

    head = ctx.chain.current_block()
    v2_from = int(cfg.v2.from_block) if getattr(cfg, "v2", None) else 0
    ctx.state["head_at_start"] = head
    r.check(S, "v2_active", v2_from and head >= v2_from, f"head={head} [v2] from_block={v2_from}",
            warn=head < v2_from)

    rc, st, _ = cli(ctx, "status", "--chain-toml", ctx.toml, timeout=120)
    data = (st or {}).get("data") or {}
    fresh = bool(data.get("fresh"))
    r.check(S, "status_feed_fresh", fresh, json.dumps(data.get("freshness", {}))[:300])
    pre = data.get("launch_preflight") or {}
    r.check(S, "launch_preflight", pre.get("state") not in ("blocked", None),
            f"{pre.get('state')}: {pre.get('reason', '')}"[:300])
    published = (data.get("training_contract") or {}).get("digest")
    # Once v2 is active the feed publishes the v2 contract, not the base one
    # that ``cascade-trainer --offline`` prints.
    if v2_from and head >= v2_from and cfg.training.v2_contract is not None:
        from cascade.shared.manifest import contract_digest
        digest = contract_digest(cfg.training.v2_contract)
    r.check(S, "contract_digest_matches_feed", published == digest,
            f"feed={published} expected={digest}", warn=not fresh)
    providers = (data.get("compute") or {}).get("rolling_providers") or []
    r.check(S, "rolling_providers_published", bool(providers), f"{providers}")

    img = cfg.training.train_image_digest if hasattr(cfg.training, "train_image_digest") else ""
    funded = getattr(cfg.round, "funded_pod_image", "")
    # The contract pins a bare digest; the pod image ref carries it after "@".
    r.check(S, "image_pinned", bool(re.fullmatch(r"sha256:[0-9a-f]{64}", img or "")),
            f"train_image_digest={img}")
    same = bool(img) and img.split("@")[-1] == funded.split("@")[-1]
    r.check(S, "funded_image_matches_train_image", same, f"funded_pod_image={funded}")
    resolved = ghcr_digest(funded) if funded else None
    r.check(S, "image_resolvable", resolved == funded.split("@")[-1], f"GHCR → {resolved}")

    pinned = getattr(cfg.round, "worker_code_fingerprint", "")
    local = local_worker_fingerprint()
    r.check(S, "worker_code_fingerprint", pinned == local, f"pinned={pinned[:16]}… local={local[:16]}…")

    for w in ["owner", "validator0", "validator1"] + sorted({s["wallet"] for s in SCENARIOS}):
        hk = hotkey_of(w)
        uid = ctx.chain.uid_for_hotkey(hk) if hk else None
        scenarios = [s for s in SCENARIOS if s["wallet"] == w]
        runnable = [s for s in scenarios if all(os.environ.get(PROVIDER_ENV[p]) for p in s["providers"])]
        if uid is None and scenarios and not runnable:
            r.add(S, f"wallet_{w}", "SKIP", f"not registered; only needed by {[s['name'] for s in scenarios]} "
                  "whose provider keys are not set")
            continue
        r.check(S, f"wallet_{w}", uid is not None, f"hotkey={hk and hk[:10]} uid={uid}")

    for p, env in PROVIDER_ENV.items():
        present = bool(os.environ.get(env))
        r.add(S, f"provider_key_{p}", "PASS" if present else "SKIP", f"{env} {'set' if present else 'not set'}")

    baseline = testnet_pods()
    ctx.state.setdefault("pod_baseline", baseline)
    r.add(S, "pod_baseline", "PASS", json.dumps(baseline))
    ctx.save_state()


def stage_admission(ctx: Ctx) -> None:
    S = "admission"
    cases = (
        ("accepts_mv32", "mv32", None),
        ("rejects_univariate", "univariate", r"channel"),
        ("rejects_packed_source", "packed_source", r"pack|string|exec|static"),
        ("rejects_nondeterministic", "nondeterministic", r"not deterministic"),
        ("rejects_network_access", "net_access", r"blocked_import|socket"),
        # Known gap probe: a sub-MIN_PACKED_CHARS exec'd module with a blocked import.
        ("gap_short_packed_import", "short_packed", r"blocked_import|packed_source|socket"),
    )
    known_gaps = {"gap_short_packed_import"}

    def one(case):
        name, fixture, expect = case
        t0 = time.time()
        rc, out, err = run([str(VENV_BIN / "cascade"), "verify", str(GENERATORS / fixture),
                            "--chain-toml", ctx.toml], timeout=1800)
        text = (out + err)[-1500:]
        return name, expect, rc, text, t0

    with ThreadPoolExecutor(max_workers=ctx.args.verify_workers) as pool:
        for name, expect, rc, text, t0 in pool.map(one, cases):
            if expect is None:
                ctx.r.check(S, name, rc == 0 and "OK" in text, text.strip().splitlines()[-1:] and
                            text.strip().splitlines()[-1], t0=t0)
            else:
                rejected = rc != 0 or "FAIL" in text
                reason_ok = bool(re.search(expect, text, re.I))
                crashed = "Traceback (most recent call last)" in text
                if crashed:
                    ctx.r.add(S, name, "FAIL", f"verify crashed instead of reporting a rejection: "
                              f"{text.strip().splitlines()[-1][:300]}", t0=t0)
                    continue
                if name in known_gaps:
                    ctx.r.add(S, name, "PASS" if rejected else "WARN",
                              "static guard rejected it" if rejected else
                              "KNOWN GAP: verified OK; a short exec'd string hides a blocked import "
                              "from the static scan (runtime sandbox is the only backstop)", t0=t0)
                    continue
                ctx.r.check(S, name, rejected and reason_ok,
                            f"rejected={rejected} reason_matches={reason_ok}: {text.strip()[-300:]}",
                            warn=rejected and not reason_ok, t0=t0)

    # Upload path without touching chain or funding: --no-commit prints the payload.
    sc = SCENARIOS[0]
    t0 = time.time()
    rc, doc, err = cli(ctx, "submit", str(scenario_generator(ctx, sc)), ctx.args.intake,
                       "--chain-toml", ctx.toml, "--network", ctx.args.network,
                       "--wallet-name", sc["wallet"], "--wallet-hotkey", "default",
                       "--no-commit", "--no-fund", "--skip-runtime", timeout=1800)
    ref = find_value(doc, re.compile(r"vault/direct@sha256:[0-9a-f]{64}")) if doc else None
    ctx.r.check(S, "submit_dry_run", rc == 0 and bool(ref), f"rc={rc} ref={ref} {err[-200:]}", t0=t0)


def _submit(ctx: Ctx, sc: dict) -> dict:
    gen = scenario_generator(ctx, sc)
    label = f"e2e-{sc['name']}-{ctx.state['run_id']}"[:32]
    keys = [a for p in sc["providers"] for a in ("--provider-key-env", f"{p}={PROVIDER_ENV[p]}")]
    common = ["--chain-toml", ctx.toml, "--network", ctx.args.network,
              "--wallet-name", sc["wallet"], "--wallet-hotkey", "default"]
    if sc["flow"] == "submit":
        rc, doc, err = cli(ctx, "submit", str(gen), ctx.args.intake, *common, *keys,
                           "--label", label, "--yes", timeout=2400)
    else:
        rc, doc, err = cli(ctx, "submit", str(gen), ctx.args.intake, *common, "--label", label,
                           "--no-fund", timeout=2400)
    ref = find_value(doc, re.compile(r"vault/direct@sha256:[0-9a-f]{64}")) if doc else None
    entry = {"scenario": sc["name"], "wallet": sc["wallet"], "hotkey": hotkey_of(sc["wallet"]),
             "providers": sc["providers"], "ref": ref, "submit_rc": rc, "submit_err": err[-400:],
             "submitted_at": time.time(), "phases": [], "status": None}
    if rc == 0 and ref and sc["flow"] == "fund":
        # fund talks only to the intake: no --chain-toml / --network.
        fund = ["fund", ctx.args.intake, "--ref", ref, "--wallet-name", sc["wallet"],
                "--wallet-hotkey", "default", *keys, "--label", label, "--yes"]
        if sc.get("gpu"):
            fund += ["--gpu", sc["gpu"]]
        if "max_price" in sc:
            fund += ["--max-price", str(sc["max_price"])]
        if sc.get("allow_fallback"):
            fund += ["--allow-fallback"]
        frc, fdoc, ferr = cli(ctx, *fund, timeout=600)
        entry.update(fund_rc=frc, fund_err=ferr[-400:],
                     fund_warning=find_value(fdoc, re.compile(r".*warning.*", re.I)) if fdoc else None)
    return entry


def _track(ctx: Ctx, entry: dict) -> None:
    rc, doc, _ = cli(ctx, "status", "--chain-toml", ctx.toml, "--hotkey", entry["hotkey"], "--lane", "a",
                     "--ref", entry["ref"], "--intake", ctx.args.intake, timeout=120)
    data = (doc or {}).get("data") or {}
    obs = data.get("observed") or {}
    status, phase = obs.get("status") or data.get("status"), obs.get("phase")
    if not entry["phases"] or entry["phases"][-1][1:] != [status, phase]:
        entry["phases"].append([time.time(), status, phase])
        print(f"    {entry['scenario']}: status={status} phase={phase} "
              f"err={obs.get('last_error_class') or ''}", flush=True)
    entry.update(status=status, phase=phase, gpu=obs.get("gpu"), attempts=obs.get("attempts"),
                 requeues=obs.get("requeues"), last_error_class=obs.get("last_error_class"),
                 operator_billed=obs.get("operator_billed"), verdict=data.get("verdict"),
                 started_block=obs.get("started_block"), era_index=obs.get("era_index"))


def _provider_used(entry: dict) -> str | None:
    """Which provider rented this entry's pod, from the trainer log."""
    log = ROOT / "trainer.log"
    if not log.is_file() or not entry.get("hotkey"):
        return None
    tag = entry["hotkey"][:8]
    hits = [ln for ln in log.read_text(errors="replace").splitlines()[-200000:]
            if tag in ln and re.search(r"\b(lium|shadeform|vast|runpod)\b.*\b(up|create|rent)", ln, re.I)]
    m = re.search(r"\b(lium|shadeform|vast|runpod)\b", hits[-1], re.I) if hits else None
    return m.group(1).lower() if m else None


def stage_legs(ctx: Ctx) -> None:
    S = "legs"
    entries = ctx.state.setdefault("entries", {})
    selected = [s for s in SCENARIOS if not ctx.args.scenarios or s["name"] in ctx.args.scenarios]
    for sc in selected:
        if sc["name"] in entries:
            continue
        missing = [p for p in sc["providers"] if not os.environ.get(PROVIDER_ENV[p])]
        if missing or not hotkey_of(sc["wallet"]):
            ctx.r.add(S, f"submit_{sc['name']}", "SKIP",
                      f"missing keys {missing}" if missing else f"no wallet {sc['wallet']}")
            continue
        t0 = time.time()
        e = _submit(ctx, sc)
        entries[sc["name"]] = e
        ctx.save_state()
        ok = e["submit_rc"] == 0 and bool(e["ref"]) and e.get("fund_rc", 0) == 0
        ctx.r.check(S, f"submit_{sc['name']}", ok,
                    f"ref={e['ref']} rc={e['submit_rc']}/{e.get('fund_rc', '-')} "
                    f"{(e.get('submit_err') or '') + (e.get('fund_err') or '')}"[:400], t0=t0)

    live = {n: e for n, e in entries.items() if e.get("ref")}
    for name, e in live.items():
        if not e.get("revealed"):
            t0 = time.time()
            rc, doc, _ = cli(ctx, "reveal-status", e["hotkey"], "--chain-toml", ctx.toml,
                             "--network", ctx.args.network, "--watch", "--timeout-s",
                             str(int(ctx.args.reveal_timeout)), timeout=ctx.args.reveal_timeout + 120)
            state = ((doc or {}).get("data") or {}).get("state")
            e["revealed"] = state not in (None, "none", "pending")
            ctx.r.check(S, f"reveal_{name}", e["revealed"], f"state={state}", t0=t0)
            ctx.save_state()

    deadline = time.time() + ctx.args.leg_timeout
    while time.time() < deadline:
        pending = [e for e in live.values() if e.get("status") not in TERMINAL_STATUSES]
        if not pending:
            break
        for e in pending:
            _track(ctx, e)
        ctx.save_state()
        time.sleep(ctx.args.poll)

    for name, e in live.items():
        seen = [p[2] for p in e["phases"] if p[2]]
        trained = any(p in ("training", "in_flight", "uploading", "harvest", "settling") for p in seen) \
            or e.get("status") == "done"
        ctx.r.check(S, f"leg_{name}_terminal", e.get("status") in TERMINAL_STATUSES,
                    f"status={e.get('status')} phase={e.get('phase')}")
        ctx.r.check(S, f"leg_{name}_done", e.get("status") == "done",
                    f"err={e.get('last_error_class')} attempts={e.get('attempts')} requeues={e.get('requeues')}")
        ctx.r.check(S, f"leg_{name}_trained", trained, f"phases={seen}")
        ctx.r.check(S, f"leg_{name}_payer_billed", e.get("operator_billed") is False,
                    f"operator_billed={e.get('operator_billed')}")
        used = _provider_used(e)
        e["provider_used"] = used
        sc = next(s for s in SCENARIOS if s["name"] == name)
        if sc["name"] == "fallback":
            ctx.r.add(S, f"leg_{name}_provider", "PASS" if used == "shadeform" else "WARN",
                      f"used={used}; fallback is exercised only when {sc['providers'][0]} has no "
                      f"{sc.get('gpu')} capacity")
        else:
            ctx.r.check(S, f"leg_{name}_provider", used in sc["providers"], f"used={used}",
                        warn=used is None)
    ctx.save_state()


def _receipt(ctx: Ctx, validator_hk: str, round_id: str | None = None) -> dict | None:
    key = f"receipts/{validator_hk}/{'round-' + round_id if round_id else 'latest'}.json"
    doc = http_json(ctx.bucket_url(key))
    if isinstance(doc, dict) and "body" in doc and isinstance(doc["body"], dict):
        return {**doc["body"], "_envelope": True}
    return doc if isinstance(doc, dict) else None


def _participant_hotkeys(receipt: dict) -> set[str]:
    out = set()
    for p in receipt.get("participants") or []:
        if isinstance(p, dict) and p.get("hotkey"):
            out.add(p["hotkey"])
    for k in ("chal_hotkey", "king_hotkey"):
        if (receipt.get("verdict") or {}).get(k):
            out.add(receipt["verdict"][k])
    return out


def stage_settlement(ctx: Ctx) -> None:
    S = "settlement"
    vals = [hotkey_of(w) for w in ("validator0", "validator1")]
    entries = {n: e for n, e in (ctx.state.get("entries") or {}).items() if e.get("status") == "done"}
    if not entries:
        ctx.r.add(S, "settled_entries", "SKIP", "no completed legs to settle")
    rounds = ctx.state.setdefault("rounds", {})
    deadline = time.time() + ctx.args.settle_timeout
    while entries and time.time() < deadline and any(n not in rounds for n in entries):
        latest = _receipt(ctx, vals[0])
        if latest:
            hks = _participant_hotkeys(latest)
            for n, e in entries.items():
                if n not in rounds and e["hotkey"] in hks:
                    rounds[n] = str(latest.get("round_id"))
                    print(f"    {n}: judged in round {rounds[n]}", flush=True)
            ctx.save_state()
        if any(n not in rounds for n in entries):
            time.sleep(ctx.args.poll)

    for n, e in entries.items():
        rid = rounds.get(n)
        ctx.r.check(S, f"judged_{n}", rid is not None, f"round={rid}")
        if not rid:
            continue
        recs = {v: _receipt(ctx, v, rid) for v in vals}
        for v, rec in recs.items():
            ctx.r.check(S, f"receipt_{n}_{v[:8]}", rec is not None and rec.get("status") in ("scored", "rejected"),
                        f"status={rec and rec.get('status')} reject={rec and rec.get('reject_reason')}")
        a, b = recs.values()
        if a and b:
            fields = ("status", "reward_uids", "weights")
            same = {f: a.get(f) == b.get(f) for f in fields}
            va, vb = a.get("verdict") or {}, b.get("verdict") or {}
            for f in ("king_hotkey", "chal_hotkey", "challenger_wins_round", "dethroned"):
                same[f"verdict.{f}"] = va.get(f) == vb.get(f)
            ctx.r.check(S, f"validators_agree_{n}", all(same.values()),
                        json.dumps({k: v for k, v in same.items() if not v}) or "all fields equal")
        for v in vals:
            on_chain = ctx.chain.weights_for_hotkey(v) or []
            uids = sorted(i for i, x in enumerate(on_chain) if x > 0)
            rec = recs.get(v) or {}
            ctx.r.check(S, f"weights_match_{n}_{v[:8]}", uids == sorted(rec.get("reward_uids") or []),
                        f"chain={uids} receipt={sorted(rec.get('reward_uids') or [])}",
                        warn=bool(uids and rec))
        rc, doc, _ = cli(ctx, "duel", "--chain-toml", ctx.toml, "--round", rid, "--hotkey", e["hotkey"])
        ctx.r.check(S, f"duel_view_{n}", rc == 0 and (doc or {}).get("ok"), f"rc={rc}")
    rc, doc, _ = cli(ctx, "round", "--chain-toml", ctx.toml, "--network", ctx.args.network, "--once")
    ctx.r.check(S, "round_view", rc == 0 and (doc or {}).get("ok"), f"rc={rc}")


def stage_validator(ctx: Ctx) -> None:
    S = "validator"
    cfg = ctx.cfg
    idx = http_json(ctx.bucket_url("receipts/index.json"))
    rounds = (idx or {}).get("rounds") or []
    ctx.r.check(S, "receipt_index", bool(rounds), f"{len(rounds)} rounds, updated_at={(idx or {}).get('updated_at')}")
    for w in ("validator0", "validator1"):
        hk = hotkey_of(w)
        latest = _receipt(ctx, hk)
        ctx.r.check(S, f"latest_receipt_{w}", latest is not None,
                    f"round={latest and latest.get('round_id')} validator_hotkey="
                    f"{latest and str(latest.get('validator_hotkey'))[:10]}")
        if latest:
            ctx.r.check(S, f"receipt_signer_{w}", latest.get("validator_hotkey") in ("", hk), "signer is this validator")
    chain_status = http_json(ctx.bucket_url("status/chain.json"))
    ctx.r.check(S, "chain_status_feed", chain_status is not None, f"as_of={(chain_status or {}).get('as_of')}")

    v2 = getattr(cfg, "v2", None)
    src = getattr(v2, "bench_source", "") if v2 else ""
    if src == "validator":
        rids = sorted(set((ctx.state.get("rounds") or {}).values()))
        for rid in rids:
            rep = http_json(ctx.bucket_url(f"bench_v/round-{rid}.json"))
            ctx.r.check(S, f"bench_v_{rid}", rep is not None, "main validator bench report published")
    else:
        ctx.r.add(S, "bench_v", "SKIP", f"[v2] bench_source={src!r}: validator-sourced Lane A benches are off")
    if v2 and getattr(v2, "king_ladder", False):
        ctx.r.add(S, "king_ladder", "PASS", "king ladder armed; era-gate fields checked via receipts")
    else:
        ctx.r.add(S, "king_ladder", "SKIP", "[v2] king_ladder=false")


def stage_audit(ctx: Ctx) -> None:
    S = "audit"
    rids = sorted(set((ctx.state.get("rounds") or {}).values()))
    if not rids:
        ctx.r.add(S, "audit_rounds", "SKIP", "no settled e2e rounds")
        return
    for rid in rids:
        for w in ("validator0", "validator1"):
            t0 = time.time()
            rc, out, err = run([str(VENV_BIN / "cascade-audit"), "round", rid, "--tier", "0", "--json",
                                "--config", ctx.toml, "--network", ctx.args.network,
                                "--validator", hotkey_of(w) or ""], timeout=1800)
            fails = len(re.findall(r'"(status|result)"\s*:\s*"FAIL"', out))
            ctx.r.check(S, f"audit_{rid}_{w}", rc == 0 and fails == 0,
                        f"rc={rc} fails={fails} {err[-200:]}", t0=t0)


def stage_cleanup(ctx: Ctx) -> None:
    S = "cleanup"
    baseline = ctx.state.get("pod_baseline") or {}
    deadline = time.time() + ctx.args.cleanup_timeout
    while True:
        now = testnet_pods()
        extra = {p: sorted(set(v) - set(baseline.get(p, []))) for p, v in now.items()}
        extra = {p: v for p, v in extra.items() if v}
        if not extra or time.time() > deadline:
            break
        time.sleep(ctx.args.poll)
    ctx.r.check(S, "no_leaked_pods", not extra, json.dumps(extra) if extra else "only baseline pods remain")


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain-toml", default=str(ROOT / "chain.testnet.toml"))
    ap.add_argument("--network", default="test")
    ap.add_argument("--intake", default="http://127.0.0.1:8790")
    ap.add_argument("--s3-base", default="https://s3.hippius.com")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--scenarios", default="", help="comma list of scenario names (default: all)")
    ap.add_argument("--spend", action="store_true", help="allow paid stages (submissions rent GPUs)")
    ap.add_argument("--resume", action="store_true", help="continue the run recorded in --state")
    ap.add_argument("--state", default=str(ROOT / "_e2e" / "state.json"))
    ap.add_argument("--report", default=str(ROOT / "_e2e" / "report.json"))
    ap.add_argument("--workdir", default=str(ROOT / "_e2e" / "work"))
    ap.add_argument("--poll", type=float, default=60)
    ap.add_argument("--verify-workers", type=int, default=3)
    ap.add_argument("--reveal-timeout", type=float, default=3 * 3600)
    ap.add_argument("--leg-timeout", type=float, default=6 * 3600)
    ap.add_argument("--settle-timeout", type=float, default=4 * 3600)
    ap.add_argument("--cleanup-timeout", type=float, default=1800)
    args = ap.parse_args(argv)
    args.scenarios = [s for s in args.scenarios.split(",") if s]
    for p in (args.state, args.report):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    report = Report(Path(args.report))
    ctx = Ctx(args, report)
    for stage in [s for s in args.stages.split(",") if s]:
        if stage not in STAGES:
            ap.error(f"unknown stage {stage}")
        if stage in PAID and not args.spend:
            report.add(stage, "stage", "SKIP", "paid stage: rerun with --spend")
            continue
        print(f"== {stage}", flush=True)
        try:
            globals()[f"stage_{stage}"](ctx)
        except Exception as exc:  # noqa: BLE001 — one broken stage must not hide the others
            report.add(stage, "stage_crashed", "FAIL", f"{type(exc).__name__}: {exc}")
        ctx.save_state()

    report.save()
    counts = {s: sum(c.status == s for c in report.checks) for s in ("PASS", "FAIL", "WARN", "SKIP")}
    print(f"\n{counts}  report: {args.report}")
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
