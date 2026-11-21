"""Upgrade-from-mainnet-state harness (lane D plan §3.0 R10, §4 T25, §10.8).

``python -P -m cascade.laned.restore <command> ...``

* ``snapshot``  copies a node's state files (validator state, Cascade state,
  warm-start init, netconfig state, ``activation_*`` lock-ins and their
  ``.bak`` siblings) into a timestamped bundle with a sha256 manifest. It
  reads the source and never writes to it; it bundles nothing else (no keys,
  env files, configs, logs, trainer db or settlement journal), and refuses
  while a settlement is in flight (§10.8: never restore across a settlement).
* ``verify``    checks a bundle against its manifest.
* ``restore``   places a verified bundle into a testnet node dir at the file
  names the target config gives, byte for byte, and writes
  ``laned_restore.json``. It refuses a target whose netuid or network_id is
  the source's. ``--start-cmd`` only prints the release start command.
* ``check``     loads the restored files through the real loaders and prints
  ``CLEAN START`` only when every one loads from its main file.
* ``verify-evidence`` checks the T25 testnet evidence (``laned_t25/1``).

Only the node identity changes on restore (file placement and the
netuid / network_id recorded in ``laned_restore.json``); no byte of any
state file is rewritten. Exit codes: 0 pass, 1 refused, 2 usage error.
Contract: ``tests/accept/r10/README.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cascade.shared.atomic import read_text_with_backup
from cascade.shared.config.loader import load_chain_config

BUNDLE_SCHEMA = "laned_state_bundle/1"
RESTORE_SCHEMA = "laned_state_restore/1"
EVIDENCE_SCHEMA = "laned_t25/1"
MANIFEST = "manifest.json"
RESTORE_RECORD = "laned_restore.json"
BUNDLE_FILES_DIR = "files"
NETCONFIG_NAME = "netconfig_state.json"
ACTIVATION_MAIN = "activation_state.json"
ACTIVATION_GLOB = "activation_*_state.json"
NETWORK_PREFIX = "bittensor:"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CONFIG = REPO_ROOT / "chain.toml"
DEFAULT_TARGET_CONFIG = REPO_ROOT / "chain.testnet.toml"

ROLES = ("validator_state", "cascade_state", "warm_start_init", "netconfig_state",
         "activation")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_HEX40 = re.compile(r"[0-9a-f]{40}")
_VERDICT_KEYS = ("round_id", "outcome", "king_hotkey")


class Refused(RuntimeError):
    """A safety or consistency refusal: exit 1."""


class UsageError(ValueError):
    """Unusable input: exit 2."""


# ── node layout (from the config) ────────────────────────────────────────────


@dataclass(frozen=True)
class NodeConfig:
    path: Path
    netuid: int
    network_id: str
    state_db_path: str
    cascade_state_db_path: str
    warm_start_init_path: str


def load_node_config(path: Path) -> NodeConfig:
    if not path.is_file():
        raise UsageError(f"config {path} does not exist")
    cfg = load_chain_config(path)
    return NodeConfig(path=path.resolve(), netuid=int(cfg.subnet.netuid),
                      network_id=str(cfg.netconfig.network_id),
                      state_db_path=str(cfg.validator.state_db_path),
                      cascade_state_db_path=str(cfg.validator.cascade_state_db_path),
                      warm_start_init_path=str(cfg.validator.warm_start_init_path))


def _resolve(node_dir: Path, configured: str) -> Path:
    p = Path(configured)
    return p if p.is_absolute() else node_dir / p


def role_paths(node_dir: Path, cfg: NodeConfig) -> dict[str, Path]:
    """The main path of each single-file role in ``node_dir``."""
    state = _resolve(node_dir, cfg.state_db_path)
    return {"validator_state": state,
            "cascade_state": _resolve(node_dir, cfg.cascade_state_db_path),
            "warm_start_init": _resolve(node_dir, cfg.warm_start_init_path),
            "netconfig_state": state.with_name(NETCONFIG_NAME)}


def activation_paths(state_dir: Path) -> list[Path]:
    found = sorted(state_dir.glob(ACTIVATION_GLOB))
    main = state_dir / ACTIVATION_MAIN
    if main.is_file():
        found.insert(0, main)
    return [p for p in found if p.is_file()]


def _bak(path: Path) -> Path:
    return path.with_name(path.name + ".bak")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ── snapshot ─────────────────────────────────────────────────────────────────


def settlement_problem(state_path: Path) -> str | None:
    """Refusal reason when ``<state_db_path>.settlement.json`` says a
    settlement is in flight (or can't be read, which can't be told apart)."""
    journal = Path(str(state_path) + ".settlement.json")
    if not journal.is_file():
        return None
    try:
        phase = json.loads(journal.read_text(encoding="utf-8")).get("phase")
    except (ValueError, AttributeError) as e:
        return f"settlement journal {journal} is unreadable ({e}); a settlement may be in flight"
    if phase == "committed":
        return None
    return (f"settlement in flight: {journal} has phase {phase!r}; §10.8 forbids "
            "restoring state across a settlement, snapshot after it commits")


def collect_source_files(src: Path, cfg: NodeConfig) -> list[tuple[str, Path]]:
    """(role, path) of every state file to bundle, main before ``.bak``."""
    paths = role_paths(src, cfg)
    if not paths["validator_state"].is_file():
        raise Refused(f"validator state {paths['validator_state']} is missing; "
                      "it is required")
    out: list[tuple[str, Path]] = []
    for role in ("validator_state", "cascade_state", "warm_start_init", "netconfig_state"):
        for p in (paths[role], _bak(paths[role])):
            if p.is_file():
                out.append((role, p))
    for p in activation_paths(paths["validator_state"].parent):
        out.append(("activation", p))
        if _bak(p).is_file():
            out.append(("activation", _bak(p)))
    names = [p.name for _, p in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise Refused(f"two state files share a name ({', '.join(dup)}); "
                      "the bundle can't tell them apart")
    return out


def snapshot(src: Path, cfg_path: Path, out: Path, block: int | None) -> tuple[Path, str]:
    src = src.resolve()
    if not src.is_dir():
        raise UsageError(f"--source {src} is not a directory")
    if block is not None and block <= 0:
        raise UsageError("--block must be > 0")
    cfg = load_node_config(cfg_path)
    problem = settlement_problem(role_paths(src, cfg)["validator_state"])
    if problem:
        raise Refused(problem)
    files = collect_source_files(src, cfg)
    out = out.resolve()
    if out == src or src in out.parents:
        raise Refused(f"--out {out} is inside --source {src}; the source is never written")
    now = datetime.now(UTC)
    bundle = out / now.strftime("state-%Y%m%dT%H%M%SZ")
    if bundle.exists():
        raise Refused(f"bundle {bundle} already exists; retry in a second")
    staging = out / f".{bundle.name}.partial-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        entries = []
        for role, path in files:
            data = path.read_bytes()
            rel = f"{BUNDLE_FILES_DIR}/{path.name}"
            _write_atomic(staging / rel, data)
            entries.append({"role": role, "source_name": path.name, "path": rel,
                            "sha256": _sha256(data), "bytes": len(data)})
        manifest = {"schema": BUNDLE_SCHEMA,
                    "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "source": {"dir": str(src), "netuid": cfg.netuid,
                               "network_id": cfg.network_id, "block": block},
                    "files": entries}
        raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_atomic(staging / MANIFEST, raw)
        os.replace(staging, bundle)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return bundle, _sha256(raw)


# ── verify ───────────────────────────────────────────────────────────────────


def read_manifest(bundle: Path) -> dict:
    path = bundle / MANIFEST
    if not path.is_file():
        raise Refused(f"{path} is missing")
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise Refused(f"{path} does not parse: {e}") from None
    if not isinstance(m, dict) or m.get("schema") != BUNDLE_SCHEMA:
        raise Refused(f"{path} is not a {BUNDLE_SCHEMA} manifest")
    src = m.get("source")
    if not isinstance(src, dict) or not isinstance(src.get("netuid"), int) \
            or not isinstance(src.get("network_id"), str):
        raise Refused(f"{path}: source.netuid / source.network_id missing")
    if not isinstance(m.get("files"), list):
        raise Refused(f"{path}: files is not a list")
    return m


def _entry_problem(bundle: Path, e: object) -> tuple[str | None, Path | None]:
    if not isinstance(e, dict):
        return f"manifest entry {e!r} is not an object", None
    rel = e.get("path")
    name = e.get("source_name")
    if not isinstance(rel, str) or not isinstance(name, str) or e.get("role") not in ROLES:
        return f"manifest entry {e!r} lacks role / source_name / path", None
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or Path(name).name != name:
        return f"manifest entry {rel} escapes the bundle", None
    path = bundle / rel_path
    if not path.is_file():
        return f"{rel}: listed in the manifest but missing", path
    if _sha256_file(path) != e.get("sha256"):
        return f"{rel}: sha256 differs from the manifest", path
    return None, path


def verify_bundle(bundle: Path) -> tuple[dict, list[str]]:
    """(manifest, problems). Problems name the offending file."""
    bundle = bundle.resolve()
    if not bundle.is_dir():
        raise Refused(f"bundle {bundle} is not a directory")
    m = read_manifest(bundle)
    problems: list[str] = []
    listed: set[Path] = set()
    for e in m["files"]:
        problem, path = _entry_problem(bundle, e)
        if path is not None:
            listed.add(path)
        if problem:
            problems.append(problem)
    for p in sorted(bundle.rglob("*")):
        if p.is_file() and p != bundle / MANIFEST and p not in listed:
            problems.append(f"{p.relative_to(bundle)}: not listed in the manifest")
    return m, problems


# ── restore ──────────────────────────────────────────────────────────────────


def target_placement(entry: dict, tgt: Path, cfg: NodeConfig, src_main: dict[str, str]) -> Path:
    """Where the target config puts one bundled file."""
    role, name = entry["role"], entry["source_name"]
    paths = role_paths(tgt, cfg)
    if role == "activation":
        return paths["validator_state"].parent / name
    base = src_main.get(role)
    if name == base:
        return paths[role]
    if base is not None and name == base + ".bak":
        return _bak(paths[role])
    raise Refused(f"bundle file {name} ({role}) is neither the role's main file nor its .bak")


def _source_main_names(entries: list[dict]) -> dict[str, str]:
    """Each single-file role's main source name (the entry that isn't ``.bak``
    of another entry of the same role)."""
    out: dict[str, str] = {}
    for role in ("validator_state", "cascade_state", "warm_start_init", "netconfig_state"):
        names = {e["source_name"] for e in entries if e["role"] == role}
        mains = [n for n in names if not (n.endswith(".bak") and n[:-4] in names)]
        if len(mains) > 1:
            raise Refused(f"bundle lists more than one {role} file: {sorted(mains)}")
        if mains:
            out[role] = mains[0]
    return out


def _stale_target_files(tgt: Path, cfg: NodeConfig, placing: set[Path]) -> list[Path]:
    """State files in the target that the restore would not overwrite but a
    loader would still read (a stale ``.bak`` or lock-in from an earlier run)."""
    paths = role_paths(tgt, cfg)
    candidates = [q for p in paths.values() for q in (p, _bak(p))]
    for p in activation_paths(paths["validator_state"].parent):
        candidates += [p, _bak(p)]
    return sorted({p for p in candidates if p.is_file() and p.resolve() not in placing})


def _relative_to(path: Path, root: Path, what: str) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        raise Refused(f"{what} {path} lies outside the target dir {root}") from None


def start_command(tgt: Path, cfg: NodeConfig) -> tuple[str, str]:
    if not cfg.network_id.startswith(NETWORK_PREFIX):
        raise Refused(f"target network_id {cfg.network_id!r} has no {NETWORK_PREFIX} prefix")
    net = cfg.network_id[len(NETWORK_PREFIX):]
    argv = [sys.executable, "-P", "-m", "cascade.validator.main",
            "--chain-toml", str(cfg.path), "--network", net]
    return str(tgt), shlex.join(argv)


def restore(bundle: Path, tgt: Path, cfg_path: Path, *, force: bool) -> dict:
    bundle = bundle.resolve()
    m, problems = verify_bundle(bundle)
    if problems:
        raise Refused("bundle does not verify; nothing written:\n  " + "\n  ".join(problems))
    cfg = load_node_config(cfg_path)
    src = m["source"]
    if cfg.netuid == src["netuid"]:
        raise Refused(f"target netuid {cfg.netuid} equals the source netuid {src['netuid']}; "
                      "a mainnet copy is never restored onto mainnet")
    if cfg.network_id == src["network_id"]:
        raise Refused(f"target network_id {cfg.network_id} equals the source's "
                      f"(netuid {src['netuid']}); a mainnet copy is never restored onto mainnet")
    tgt = tgt.resolve()
    src_dir = Path(str(src.get("dir") or "")).resolve() if src.get("dir") else None
    if tgt == bundle or bundle in tgt.parents or tgt in bundle.parents:
        raise Refused(f"target {tgt} overlaps the bundle {bundle}")
    if src_dir is not None and (tgt == src_dir or src_dir in tgt.parents
                                or tgt in src_dir.parents):
        raise Refused(f"target {tgt} overlaps the snapshot source {src_dir}")
    if tgt.exists() and not tgt.is_dir():
        raise Refused(f"target {tgt} is not a directory")
    if tgt.is_dir() and any(tgt.iterdir()) and not force:
        raise Refused(f"target {tgt} is not empty; pass --force-target to restore over it")
    entries = m["files"]
    if not any(e["role"] == "validator_state" for e in entries):
        raise Refused("bundle holds no validator_state; it is required")
    src_main = _source_main_names(entries)
    plan = []
    for e in entries:
        dest = target_placement(e, tgt, cfg, src_main)
        plan.append((e, dest, _relative_to(dest, tgt, f"target path for {e['source_name']}")))
    dests = [rel for _, _, rel in plan]
    if len(set(dests)) != len(dests) or RESTORE_RECORD in dests:
        raise Refused(f"target config maps two bundle files onto one path: {sorted(dests)}")
    stale = _stale_target_files(tgt, cfg, {(tgt / rel).resolve() for rel in dests})
    if stale:
        raise Refused("target holds state files this bundle would not overwrite "
                      f"(a loader would read them): {', '.join(map(str, stale))}; "
                      "remove them deliberately or restore into an empty dir")
    tgt.mkdir(parents=True, exist_ok=True)
    record_files = []
    for e, dest, rel in plan:
        data = (bundle / e["path"]).read_bytes()
        if _sha256(data) != e["sha256"]:
            raise Refused(f"{e['path']} changed during the restore")
        _write_atomic(dest, data)
        if _sha256_file(dest) != e["sha256"]:
            raise Refused(f"{dest} does not hold the bundled bytes after writing")
        record_files.append({"role": e["role"], "source_name": e["source_name"],
                             "target_path": rel, "sha256": e["sha256"]})
    record = {"schema": RESTORE_SCHEMA,
              "bundle_sha256": _sha256_file(bundle / MANIFEST),
              "source": {"netuid": src["netuid"], "network_id": src["network_id"],
                         "block": src.get("block")},
              "target": {"dir": str(tgt), "netuid": cfg.netuid, "network_id": cfg.network_id},
              "files": record_files}
    _write_atomic(tgt / RESTORE_RECORD,
                  (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return record


# ── check (real loaders) ─────────────────────────────────────────────────────


def _loaders() -> dict[str, tuple[Callable[[str], object], Callable[[Path], object]]]:
    """role -> (the parser the loader applies to file text, the real loader)."""
    from cascade.shared import activation as act
    from cascade.shared import netconfig as nc
    from cascade.validator import cascade as casc
    from cascade.validator import loop
    from cascade.validator import state as champion

    return {"validator_state": (champion.loads, lambda p: loop._load_state(str(p))),
            "cascade_state": (casc.loads, casc.load_state),
            "netconfig_state": (nc.NetConfigState.from_json, nc.load_state),
            "activation": (act._parse_record, lambda p: act.ActivationStore(p).load())}


def _load_problem(role: str, path: Path, parse, load) -> str | None:
    _, source = read_text_with_backup(path, parse)
    if source == "backup":
        return (f"{role} {path.name}: the main file does not parse and the loader "
                f"resumed from {path.name}.bak; a torn main file isn't a clean copy")
    try:
        load(path)
    except Exception as e:  # noqa: BLE001 — any loader failure is a refusal
        return f"{role} {path.name}: the loader raised {type(e).__name__}: {e}"
    return None


def check(tgt: Path, cfg_path: Path) -> list[str]:
    tgt = tgt.resolve()
    cfg = load_node_config(cfg_path)
    rec_path = tgt / RESTORE_RECORD
    if not rec_path.is_file():
        return [f"{RESTORE_RECORD} is missing in {tgt}; not a restored node"]
    try:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
    except ValueError as e:
        return [f"{RESTORE_RECORD} does not parse: {e}"]
    if not isinstance(rec, dict) or rec.get("schema") != RESTORE_SCHEMA \
            or not isinstance(rec.get("files"), list):
        return [f"{RESTORE_RECORD} is not a {RESTORE_SCHEMA} record"]
    problems: list[str] = []
    listed_activation: list[Path] = []
    for f in rec["files"]:
        rel = f.get("target_path") if isinstance(f, dict) else None
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            problems.append(f"{RESTORE_RECORD}: bad entry {f!r}")
            continue
        path = tgt / rel
        if not path.is_file():
            problems.append(f"{rel}: restored but now missing")
        elif _sha256_file(path) != f.get("sha256"):
            problems.append(f"{rel}: sha256 differs from the restore record (drift after restore)")
        if f.get("role") == "activation" and not rel.endswith(".bak"):
            listed_activation.append(path)
    paths = role_paths(tgt, cfg)
    if not paths["validator_state"].is_file():
        problems.append(f"validator_state {paths['validator_state'].name}: missing; "
                        "the validator state is required")
    loaders = _loaders()
    to_load = [(role, paths[role]) for role in ("validator_state", "cascade_state",
                                                "netconfig_state")]
    activation = {p.resolve(): p for p in listed_activation}
    for p in activation_paths(paths["validator_state"].parent):
        activation.setdefault(p.resolve(), p)
    to_load += [("activation", p) for p in activation.values()]
    for role, path in to_load:
        if not path.is_file() and not _bak(path).is_file():
            continue
        parse, load = loaders[role]
        problem = _load_problem(role, path, parse, load)
        if problem:
            problems.append(problem)
    warm = paths["warm_start_init"]
    if warm.is_file():
        try:
            obj = json.loads(warm.read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"warm_start_init {warm.name}: does not parse: {e}")
        else:
            if not isinstance(obj, dict):
                problems.append(f"warm_start_init {warm.name}: is not a JSON object")
    return problems


# ── T25 evidence ─────────────────────────────────────────────────────────────


EVIDENCE_KEYS = ("schema", "wp", "row", "passed", "source_network_id", "source_netuid",
                 "source_block", "bundle_sha256", "target_node", "target_netuid",
                 "release_head", "clean_start", "first_verdict", "mainnet_verdict",
                 "verdict_identical")


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _nonempty_str(v: object) -> bool:
    return isinstance(v, str) and bool(v.strip())


def evidence_problems(ev: object, source: NodeConfig, target: NodeConfig,
                      bundle: Path | None) -> list[str]:
    if not isinstance(ev, dict):
        return ["evidence is not a JSON object"]
    missing = [k for k in EVIDENCE_KEYS if k not in ev]
    if missing:
        return [f"missing key(s): {', '.join(missing)}"]
    p: list[str] = []
    if ev["schema"] != EVIDENCE_SCHEMA:
        p.append(f"schema is {ev['schema']!r}, expected {EVIDENCE_SCHEMA}")
    if ev["wp"] != "R10":
        p.append(f"wp is {ev['wp']!r}, expected R10")
    if ev["row"] != "T25":
        p.append(f"row is {ev['row']!r}, expected T25")
    if ev["source_network_id"] != source.network_id:
        p.append(f"source_network_id {ev['source_network_id']!r} is not the mainnet "
                 f"{source.network_id!r}")
    if not _is_int(ev["source_netuid"]) or ev["source_netuid"] != source.netuid:
        p.append(f"source_netuid {ev['source_netuid']!r} is not the mainnet {source.netuid}")
    if not _is_int(ev["target_netuid"]) or ev["target_netuid"] != target.netuid:
        p.append(f"target_netuid {ev['target_netuid']!r} is not the testnet {target.netuid}")
    if not _is_int(ev["source_block"]) or ev["source_block"] <= 0:
        p.append(f"source_block {ev['source_block']!r} is not an int > 0")
    if not isinstance(ev["bundle_sha256"], str) or not _HEX64.fullmatch(ev["bundle_sha256"]):
        p.append("bundle_sha256 is not 64 lowercase hex")
    if not isinstance(ev["release_head"], str) or not _HEX40.fullmatch(ev["release_head"]):
        p.append("release_head is not a 40-hex lowercase git sha")
    if not _nonempty_str(ev["target_node"]):
        p.append("target_node is empty")
    for k in ("passed", "clean_start", "verdict_identical"):
        if not isinstance(ev[k], bool):
            p.append(f"{k} is not a boolean")
    verdicts_ok = True
    for k in ("first_verdict", "mainnet_verdict"):
        v = ev[k]
        if not isinstance(v, dict):
            p.append(f"{k} is not an object")
            verdicts_ok = False
            continue
        for f in _VERDICT_KEYS:
            if not _nonempty_str(v.get(f)):
                p.append(f"{k}.{f} is missing or empty")
                verdicts_ok = False
    if verdicts_ok:
        identical = ev["first_verdict"] == ev["mainnet_verdict"]
        if ev["verdict_identical"] is not identical:
            p.append(f"verdict_identical is {ev['verdict_identical']!r} but the verdicts "
                     f"{'match' if identical else 'differ'}")
    if ev["passed"] is not True:
        p.append("passed is not true")
    elif ev["clean_start"] is not True or ev["verdict_identical"] is not True:
        p.append("passed is true without clean_start and verdict_identical")
    if bundle is not None:
        p += _bundle_cross_check(ev, bundle)
    return p


def _bundle_cross_check(ev: dict, bundle: Path) -> list[str]:
    path = bundle.resolve() / MANIFEST
    if not path.is_file():
        return [f"--bundle: {path} is missing"]
    p = []
    real = _sha256_file(path)
    if ev["bundle_sha256"] != real:
        p.append(f"bundle_sha256 {ev['bundle_sha256']} is not the sha256 of {path} ({real})")
    try:
        src = read_manifest(bundle.resolve())["source"]
    except Refused as e:
        return p + [str(e)]
    if src["netuid"] != ev["source_netuid"] or src["network_id"] != ev["source_network_id"]:
        p.append("source netuid / network_id differ from the bundle manifest")
    if src.get("block") is not None and src["block"] != ev["source_block"]:
        p.append(f"source_block {ev['source_block']!r} differs from the bundle's "
                 f"{src['block']}")
    return p


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -P -m cascade.laned.restore",
                                 description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="bundle a node's state files (read-only on the source)")
    s.add_argument("--source", type=Path, required=True, help="source node dir")
    s.add_argument("--config", type=Path, required=True, help="the source node's chain.toml")
    s.add_argument("--out", type=Path, required=True, help="dir to create the bundle under")
    s.add_argument("--block", type=int, default=None, help="chain block of the copy")
    v = sub.add_parser("verify", help="check a bundle against its manifest")
    v.add_argument("--bundle", type=Path, required=True)
    r = sub.add_parser("restore", help="place a bundle into a testnet node dir")
    r.add_argument("--bundle", type=Path, required=True)
    r.add_argument("--target", type=Path, required=True, help="target node dir")
    r.add_argument("--config", type=Path, required=True, help="the target node's chain.toml")
    r.add_argument("--force-target", action="store_true",
                   help="restore into a non-empty target dir")
    r.add_argument("--start-cmd", action="store_true",
                   help="print the release start command (never runs it)")
    c = sub.add_parser("check", help="load the restored state through the real loaders")
    c.add_argument("--target", type=Path, required=True)
    c.add_argument("--config", type=Path, required=True)
    e = sub.add_parser("verify-evidence", help="check T25 testnet evidence")
    e.add_argument("--evidence", type=Path, required=True)
    e.add_argument("--bundle", type=Path, default=None)
    e.add_argument("--source-config", type=Path, default=DEFAULT_SOURCE_CONFIG,
                   help="mainnet config naming the source netuid / network_id")
    e.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG,
                   help="testnet config naming the target netuid")
    return ap


def _report(problems: list[str], ok_line: str) -> int:
    if problems:
        for line in problems:
            print(f"FAIL {line}", file=sys.stderr)
        return 1
    print(ok_line)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "snapshot":
            bundle, sha = snapshot(args.source, args.config, args.out, args.block)
            print(f"BUNDLE {bundle} sha256={sha}")
            return 0
        if args.cmd == "verify":
            _, problems = verify_bundle(args.bundle)
            return _report(problems, f"OK {args.bundle.resolve()}")
        if args.cmd == "restore":
            rec = restore(args.bundle, args.target, args.config, force=args.force_target)
            print(f"RESTORED {rec['target']['dir']} files={len(rec['files'])} "
                  f"bundle_sha256={rec['bundle_sha256']}")
            if args.start_cmd:
                cwd, cmd = start_command(Path(rec["target"]["dir"]),
                                         load_node_config(args.config))
                print(f"START_CWD {cwd}")
                print(f"START {cmd}")
            return 0
        if args.cmd == "check":
            return _report(check(args.target, args.config), "CLEAN START")
        ev_path = args.evidence
        if not ev_path.is_file():
            raise UsageError(f"--evidence {ev_path} does not exist")
        try:
            ev = json.loads(ev_path.read_text(encoding="utf-8"))
        except ValueError as e:
            return _report([f"{ev_path} does not parse: {e}"], "")
        problems = evidence_problems(ev, load_node_config(args.source_config),
                                     load_node_config(args.target_config), args.bundle)
        return _report(problems, f"OK T25 evidence {ev_path}")
    except Refused as e:
        print(f"REFUSED {e}", file=sys.stderr)
        return 1
    except UsageError as e:
        print(f"usage error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
