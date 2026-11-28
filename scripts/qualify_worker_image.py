#!/usr/bin/env python3
"""The worker image QUALIFICATION JOB driver (W-IMG retro item 8, 2026-11-28).

A pin may point at an image only after the pipeline BUILT it (publish-worker
manifest), QUALIFIED it bit for bit against the previous qualified image,
SMOKE-TESTED every role for 30 steps and MEASURED its walls. The job is
defined in ``deploy/qualify/worker_image_job.toml``; the runbook (in order)
is docs/OPERATOR.md "Releasing a worker image". Nothing here rents, pushes,
pins or signs anything: the operator rents the pod (Approve-tier spend), this
script runs on it, and the result comes back as a PR.

``plan``  (ops box, torch-free) what this release needs: the FROM/TO images,
          the world sizes, the roles, the SKUs whose walls to measure, and the
          exact commands.
``pod``   (ON the pod, inside the image under test) checks the pod really
          runs the manifest's image (worker code fingerprint, uv.lock, the
          lock-drift marker), then runs ``qualify_pyramid_bitwise.py --solo``
          at every world size, ``image_smoke.py`` per role, and writes
          ``smoke.json`` + ``walls-<sku>.json``. ``--walls-only --sku S``
          on a pod of each rentable SKU measures that SKU's wall.
``patch`` (ops box, torch-free) the ONE config patch a qualified record
          implies (every pin of the release + the measured walls), printed
          for the owner's Approve — never written to a chain.toml here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOB_SCHEMA = "cascade-image-qualify-job/1"
DEFAULT_JOB = REPO / "deploy" / "qualify" / "worker_image_job.toml"


def _load(name: str, rel: str):
    """A sibling module by path (nothing heavy imported on the box)."""
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def qmod():
    return _load("_cascade_image_qualification", "cascade/shared/image_qualification.py")


def load_job(path: Path) -> dict:
    job = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if job.get("schema") != JOB_SCHEMA:
        raise ValueError(f"{path}: schema {job.get('schema')!r} != {JOB_SCHEMA!r}")
    for section in ("pod", "bitwise", "smoke", "walls"):
        if not isinstance(job.get(section), dict):
            raise ValueError(f"{path}: [{section}] missing")
    return job


def _round_terms(chain_toml: Path) -> dict:
    """The terms the trainer qualifies against, read from a chain.toml with
    tomllib (the same fields as cascade.shared.image_qualification)."""
    raw = tomllib.loads(Path(chain_toml).read_text(encoding="utf-8"))
    rnd, v2 = raw.get("round", {}), raw.get("v2", {})
    skus = [str(s) for s in (rnd.get("funded_pod_skus", []) or [])]
    if rnd.get("funded_pod_sku"):
        skus.append(str(rnd["funded_pod_sku"]))
    v2t = v2.get("training", {}) if isinstance(v2.get("training"), dict) else {}
    return {"skus": list(dict.fromkeys(skus)),
            "unmeasured_cap": int(rnd.get("unmeasured_image_wall_cap_seconds", 18000) or 0),
            "from_image": str(v2t.get("train_image_digest", "")
                              or raw.get("training", {}).get("train_image_digest", "")),
            "from_fingerprint": str(rnd.get("worker_code_fingerprint", "")),
            "mkt_skus": [str(s) for s in (raw.get("marketcast", {}).get("funded_pod_skus", [])
                                          or [])],
            "mkt_image": str(raw.get("marketcast", {}).get("pod_image", "")),
            "walls": dict(rnd.get("funded_sku_wall_seconds", {}) or {})}


def plan(job: dict, chain_toml: Path, manifest: dict, *, mkt: bool = False,
         from_image: str = "") -> dict:
    """What qualifying ``manifest``'s image needs (torch-free; tested)."""
    q = qmod()
    t = _round_terms(chain_toml)
    worlds = sorted(set(q.required_worlds_for_skus(t["skus"]))
                    | {int(w) for w in job["bitwise"].get("extra_worlds", [])})
    qual_gpus = q.sku_gpus(job["pod"]["qual_sku"])
    problems = []
    if max(worlds) > qual_gpus:
        problems.append(f"the qualification pod {job['pod']['qual_sku']} has {qual_gpus} GPUs; "
                        f"world {max(worlds)} is required")
    bad = q.validate_build(manifest, image=str(manifest.get("image", "")))
    problems += [f"manifest: {b}" for b in bad]
    if not mkt and not job["smoke"].get("lane_b", {}).get("king_ref"):
        problems.append("[smoke.lane_b] king_ref is empty: the lane B smoke would fail")
    if mkt and not job["smoke"].get("mkt", {}).get("recipe"):
        problems.append("[smoke.mkt] recipe is empty: the MarketCast smoke would fail")
    frm = from_image or (t["mkt_image"] if mkt else t["from_image"])
    wall_skus = (t["mkt_skus"] or t["skus"]) if mkt else t["skus"]
    return {"to_image": manifest.get("image", ""),
            "to_fingerprint": manifest.get("worker_code_fingerprint", ""),
            "from_image": frm, "kind": "marketcast" if mkt else "fleet",
            "worlds": worlds, "qual_sku": job["pod"]["qual_sku"],
            "roles": list(job["smoke"]["mkt_roles" if mkt else "roles"]),
            "wall_skus": wall_skus,
            "walls_required": "every rentable SKU" if (not mkt and t["unmeasured_cap"] <= 0)
            else "at least one rentable SKU (the rest rent under the unmeasured-image cap)",
            "golden_from": (f"qualifications/golden/{q.golden_filename(frm)}"
                            if q.image_hex(frm) and not mkt else ""),
            "problems": problems}


def pod_identity_problems(manifest: dict, *, root: Path) -> list[str]:
    """Why this pod is NOT running the manifest's image (empty = it is)."""
    wim = _load("_worker_image_manifest", "scripts/worker_image_manifest.py")
    fp, _ = wim.code_fingerprint(root)
    bad = []
    if fp != manifest.get("worker_code_fingerprint"):
        bad.append(f"pod code fingerprint {fp[:12]}… != the manifest's "
                   f"{str(manifest.get('worker_code_fingerprint'))[:12]}… (wrong image booted)")
    if wim.lock_sha256(root) != manifest.get("lock_sha256"):
        bad.append("pod uv.lock differs from the manifest's")
    if not (root / wim.LOCK_MARKER).is_file():
        bad.append("no lock-drift marker in the pod venv")
    return bad


def walls_entry(*, sku: str, lane_a: dict, training_steps: int, gpu_name: str) -> dict:
    """One measured wall from a lane A smoke on a pod of ``sku`` (tested)."""
    q = qmod()
    sps, startup = lane_a.get("steps_per_s"), lane_a.get("startup_s") or 0.0
    if not lane_a.get("passed") or not sps:
        raise ValueError(f"the lane A smoke on {sku} did not pass with a measured rate")
    if int(lane_a.get("gpus") or 0) != q.sku_gpus(sku):
        raise ValueError(f"the lane A smoke ran on {lane_a.get('gpus')} GPUs, {sku} has "
                         f"{q.sku_gpus(sku)}")
    return {"gpus": q.sku_gpus(sku), "gpu_name": gpu_name, "steps_per_s": float(sps),
            "startup_s": float(startup),
            "measured_steps": max(0, int(lane_a.get("steps") or 0) - 2),
            "training_steps": int(training_steps),
            "wall_seconds": q.wall_seconds_from_rate(int(training_steps), float(sps),
                                                     float(startup)),
            "run_id": str(lane_a.get("run_id", ""))}


def gather_smoke(out: Path, roles: list[str], image: str) -> dict:
    roles_doc = {}
    for role in roles:
        p = Path(out) / f"{role}.json"
        roles_doc[role] = (json.loads(p.read_text()) if p.is_file()
                           else {"role": role, "passed": False, "detail": "not run"})
    q = qmod()
    return {"image": "sha256:" + q.image_hex(image), "roles": roles_doc}


def _toml_str(v: str) -> str:
    return json.dumps(str(v))


def config_patch(record: dict, manifest: dict, chain_toml: Path) -> str:
    """The ONE config patch a qualified record implies (tested): every pin of
    the release (the "5 pins + wall gate" set, through the DEC-CA-0067
    ``*_after`` path) and the merged measured-wall table. Printed for the
    owner; activation stays Approve-tier and the seed-epoch rule."""
    q = qmod()
    t = _round_terms(chain_toml)
    image = str(manifest["image"])
    to = "sha256:" + q.image_hex(image)
    walls = dict(t["walls"])
    for sku, w in sorted(((record.get("measured_walls") or {}).get("skus") or {}).items()):
        walls[sku] = int(w["wall_seconds"])
    table = ", ".join(f"{_toml_str(k)} = {v}" for k, v in sorted(walls.items()))
    if record.get("schema") == q.MKT_SCHEMA:
        return "\n".join([
            "# [marketcast] — the qualified MarketCast image "
            f"(record qualifications/{q.mkt_record_filename(to)})",
            f"pod_image = {_toml_str(image)}",
            f"worker_code_fingerprint = {_toml_str(manifest['worker_code_fingerprint'])}",
            "# delete pod_image_unqualified_override_until", ""])
    frm = str(record["from_image"])
    return "\n".join([
        f"# record qualifications/{q.record_filename(frm, to)}; activation: at an observed",
        "# seed-epoch start (or via [v2] image_epoch_from_block, release-then-activate)",
        "[v2]",
        f"train_image_digest_after = {_toml_str(to)}",
        f"funded_pod_image_after = {_toml_str(image)}",
        f"worker_code_fingerprint_after = {_toml_str(manifest['worker_code_fingerprint'])}",
        f"funded_sku_wall_measured_image_after = {_toml_str(to)}",
        f"# prior_train_image_digests: append {_toml_str(frm)}",
        "image_after_unqualified_override = false",
        "# delete image_after_unqualified_override_until",
        "[round]",
        f"ops_continuation_image_digest = {_toml_str(to)}   # else every continuation leg dies",
        f"funded_sku_wall_seconds = {{ {table} }}",
        "# then: restart the trainer AND the funding intake (one change, no leg in flight)", ""])


# ── the pod run ──────────────────────────────────────────────────────────────


def _gpus() -> tuple[int, str]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    names = [n.strip() for n in out.splitlines() if n.strip()]
    if len(set(names)) > 1:
        raise SystemExit(f"mixed GPU models on one pod: {sorted(set(names))}")
    return len(names), (names[0] if names else "")


def _run(cmd: list[str], log: Path, timeout: float) -> int:
    print("+ " + shlex.join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO),
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout:.0f}s\n")
            return 124


def cmd_pod(a) -> int:
    job = load_job(a.job)
    manifest = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = pod_identity_problems(manifest, root=REPO)
    if bad:
        for b in bad:
            print(f"REFUSED: {b}")
        return 2
    n, gpu = _gpus()
    q = qmod()
    run_id = a.run_id or f"qual-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    py = sys.executable
    smoke_dir = out / "smoke"
    common = ["--chain-toml", str(a.chain_toml), "--out", str(smoke_dir), "--run-id", run_id,
              "--contract-block", str(a.contract_block)] + (["--arch", a.arch] if a.arch else [])
    timeout = float(job["smoke"].get("timeout_seconds", 1800))
    lb, mk = job["smoke"].get("lane_b", {}), job["smoke"].get("mkt", {})
    if a.walls_only:
        if q.sku_gpus(a.sku) != n:
            print(f"REFUSED: --sku {a.sku} needs {q.sku_gpus(a.sku)} GPUs, the pod has {n}")
            return 2
        roles, steps = ["lane_a_leg"], int(job["walls"].get("steps", 60))
    else:
        roles, steps = list(job["smoke"]["mkt_roles" if a.mkt else "roles"]), int(
            job["smoke"].get("steps", 30))
        if not a.mkt and not a.skip_bitwise:
            p = plan(job, a.chain_toml, manifest)
            worlds = [w for w in p["worlds"] if w <= n]
            b = job["bitwise"]
            rc = _run([py, "-P", "scripts/qualify_pyramid_bitwise.py", "--solo",
                       "--cand", str(REPO), "--worlds", ",".join(map(str, worlds)),
                       "--steps", str(b["steps"]), "--resume-at", str(b["resume_at"]),
                       "--seed", str(b["seed"]), "--channels", str(b["channels"]),
                       "--timing-steps", "0", "--chain-toml", str(a.chain_toml),
                       "--python", py, "--out", str(out / "qual-out")],
                      out / "bitwise.log", timeout * 4)
            print(f"bitwise --solo rc={rc} (compared against the golden file on the box)")
    for role in roles:
        cmd = [py, "-P", "scripts/image_smoke.py", role, *common, "--steps", str(steps),
               "--king-ref", str(lb.get("king_ref", "")),
               "--selection-json", str(lb.get("selection_json", "")),
               "--domain", str(lb.get("domain", "energy")),
               "--mkt-recipe", str((Path(a.job).parent / mk["recipe"]) if mk.get("recipe") else "")]
        rc = _run(cmd, out / f"smoke-{role}.log", timeout)
        print(f"smoke {role} rc={rc}")
    smoke = gather_smoke(smoke_dir, roles, manifest["image"])
    sku = a.sku or job["pod"]["qual_sku"]
    lane_a = smoke["roles"].get("lane_a_leg") or {}
    if lane_a.get("passed"):
        sys.path.insert(0, str(REPO))
        from cascade.shared.config import load_chain_config  # torch-free

        full = load_chain_config(Path(a.chain_toml)).training.at_block(int(a.contract_block))
        entry = walls_entry(sku=sku, lane_a=lane_a, training_steps=int(
            full.primary_size.training_steps), gpu_name=gpu)
        walls = {"image": smoke["image"], "skus": {sku: entry}}
        (out / f"walls-{sku}.json").write_text(json.dumps(walls, sort_keys=True, indent=2) + "\n")
        print(f"wall {sku}: {entry['wall_seconds']} s ({entry['steps_per_s']:.3f} steps/s)")
    if not a.walls_only:
        (out / "smoke.json").write_text(json.dumps(smoke, sort_keys=True, indent=2) + "\n")
    failed = [r for r, v in smoke["roles"].items() if not v.get("passed")]
    print(f"{'FAIL' if failed else 'PASS'}: {len(roles) - len(failed)}/{len(roles)} roles "
          f"({gpu} x{n}); artifacts {out}")
    return 1 if failed else 0


def cmd_plan(a) -> int:
    job = load_job(a.job)
    manifest = json.loads(Path(a.manifest).read_text())
    p = plan(job, a.chain_toml, manifest, mkt=a.mkt, from_image=a.from_image)
    print(json.dumps(p, indent=2))
    return 1 if p["problems"] else 0


def cmd_patch(a) -> int:
    record = json.loads(Path(a.record).read_text())
    manifest = json.loads(Path(a.manifest).read_text())
    sys.stdout.write(config_patch(record, manifest, a.chain_toml))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "pod", "patch"):
        s = sub.add_parser(name)
        s.add_argument("--chain-toml", type=Path, required=True,
                       help="the DEPLOYED chain config (the trainer's effective SKUs)")
        s.add_argument("--manifest", type=Path, required=True,
                       help="the image's publish-worker image-manifest.json")
        if name != "patch":
            s.add_argument("--job", type=Path, default=DEFAULT_JOB)
            s.add_argument("--mkt", action="store_true", help="a [marketcast] pod_image")
        if name == "plan":
            s.add_argument("--from-image", default="")
        if name == "pod":
            s.add_argument("--out", type=Path, default=Path("/root/qualify-out"))
            s.add_argument("--run-id", default="")
            s.add_argument("--contract-block", type=int, required=True,
                           help="the era start the smoke legs' contract is effective at")
            s.add_argument("--arch", default="")
            s.add_argument("--skip-bitwise", action="store_true")
            s.add_argument("--walls-only", action="store_true")
            s.add_argument("--sku", default="", help="this pod's SKU (walls), e.g. 8xH200")
        if name == "patch":
            s.add_argument("--record", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "pod" and a.walls_only and not a.sku:
        ap.error("--walls-only needs --sku")
    return {"plan": cmd_plan, "pod": cmd_pod, "patch": cmd_patch}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
