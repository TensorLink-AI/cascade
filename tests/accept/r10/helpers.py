"""Shared helpers for the R10 acceptance tests.

The tool is driven as a subprocess (``python -P -m cascade.laned.restore``)
so the tests pin the CLI contract. Fixture node state is written in
``tmp_path`` by the REAL state writers, so it has the real on-disk formats;
real node state is never read.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cascade.shared import netconfig as nc  # noqa: E402
from cascade.shared.activation import ActivationRecord, ActivationStore  # noqa: E402
from cascade.validator import cascade as casc  # noqa: E402
from cascade.validator import state as champion  # noqa: E402

MAINNET_TOML = REPO / "chain.toml"
TESTNET_TOML = REPO / "chain.testnet.toml"

SOURCE_BLOCK = 9_876_543

# Source (mainnet chain.toml) file names and their testnet placements.
SOURCE_NAMES = {
    "validator_state": "validator_state.db",
    "cascade_state": "cascade_state.json",
    "warm_start_init": "warm_start_init.json",
    "netconfig_state": "netconfig_state.json",
}
TARGET_NAMES = {
    "validator_state": "validator_state.testnet.db",
    "cascade_state": "cascade_state.testnet.json",
    "warm_start_init": "warm_start_init.testnet.json",
    "netconfig_state": "netconfig_state.json",
}
ACTIVATION_NAMES = (
    "activation_state.json",
    "activation_forfeit_state.json",
    "activation_margin_v2_state.json",
    "activation_shared_ladder_state.json",
)

KING = "5HKingHotkeyR10fixtureXXXXXXXXXXXXXXXXXXXXXXXXXX"
FORMER = "5FFormerKingR10fixtureYYYYYYYYYYYYYYYYYYYYYYYYYY"
CHALLENGER = "5CChallengerR10fixtureZZZZZZZZZZZZZZZZZZZZZZZZZZ"

_CREDENTIAL_ENV = (
    "HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY", "HIPPIUS_HUB_TOKEN", "HIPPIUS_TOKEN",
    "HIPPIUS_HUB_USERNAME", "HIPPIUS_REGISTRY_USERNAME", "HIPPIUS_HUB_PASSWORD",
    "HIPPIUS_REGISTRY_PASSWORD", "BACKUP_S3_ACCESS_KEY", "BACKUP_S3_SECRET_KEY",
    "HF_TOKEN", "WANDB_API_KEY",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cli(*args: object, timeout: int = 120) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _CREDENTIAL_ENV}
    env["PYTHONPATH"] = str(REPO)
    return subprocess.run(
        [sys.executable, "-P", "-m", "cascade.laned.restore", *[str(a) for a in args]],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout,
    )


def output(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout + proc.stderr


def ok(proc: subprocess.CompletedProcess) -> subprocess.CompletedProcess:
    assert proc.returncode == 0, (
        f"exit {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def refused(proc: subprocess.CompletedProcess, *needles: str) -> str:
    text = output(proc)
    assert proc.returncode != 0, f"expected a refusal, got exit 0:\n{text}"
    assert "No module named" not in text, f"the tool is missing:\n{text}"
    for n in needles:
        assert n in text, f"refusal does not mention {n!r}:\n{text}"
    return text


# ── fixture node state (real writers) ─────────────────────────────────────────


def champion_state(rounds_seen: int = 41) -> champion.ChampionState:
    return champion.ChampionState(
        king_hotkey=KING, king_uid=17, tenure_rounds=6,
        streaks={CHALLENGER: 1}, rounds_seen=rounds_seen, former_kings=(FORMER,),
        last_handled_round_id="r-0041", last_handled_manifest_sha="ab" * 32,
        king_since_block=9_870_300, king_pointer="hippius/cascade/king@sha256:" + "1f" * 32,
        era_index=12, last_handled_block=9_876_000, king_share_bp=6000,
        court_hotkey=FORMER, court_until_block=9_880_000,
        king_check_pass={"round_id": "r-0040", "block": 9_875_100,
                         "king_pointer": "hippius/cascade/king@sha256:" + "1f" * 32,
                         "king_hotkey": KING},
        epoch_init={"epoch": 3, "init": "hippius/cascade/init@sha256:" + "2e" * 32},
    )


def cascade_state() -> casc.CascadeState:
    rec = casc.CheckpointRecord(
        checkpoint_id="hippius/cascade/ckpt@sha256:" + "3d" * 32,
        gifteval_crps=0.512, gifteval_mase=0.871, boom_crps=0.433, boom_mase=1.02,
        time_crps=0.601, time_mase=0.954, score=0.7012, timestamp=1_763_000_000.0,
        size="toto2-4m", role="king")
    return casc.CascadeState(
        king_hotkey=KING, reign_start_block=9_870_300, checkpoints=(rec,), generation=2,
        members=("hippius/cascade/ckpt@sha256:" + "4c" * 32,), clock_observed=True)


def netconfig_state() -> nc.NetConfigState:
    body = {"protocol": nc.PROTOCOL, "seq": 1, "prev": "", "effective_block": 9_860_000,
            "overrides": {"scoring": {"king_decay": 0.5}},
            "network": {"chain": "bittensor:finney", "netuid": 91}}
    envelope = {"body": body, "signer": KING, "signature": "0x" + "5b" * 64}
    adopted = nc.Adopted(apply_block=9_860_000, seq=1, digest=nc.body_digest(body),
                         envelope=envelope, lock_block=9_858_000)
    return nc.NetConfigState(
        last=body, candidate=envelope, refused=[], history=[adopted],
        record=ActivationRecord(feature=nc.cfg_feature_name(body), lock_block=9_858_000,
                                activation_block=9_860_000,
                                last_checked_boundary=9_876_000, source="tally"))


def activation_record(name: str) -> ActivationRecord:
    feature = {"activation_state.json": "rolling-era-king",
               "activation_forfeit_state.json": "forfeit-v1",
               "activation_margin_v2_state.json": "margin-v2-a",
               "activation_shared_ladder_state.json": "lad-1"}[name]
    return ActivationRecord(feature=feature, lock_block=9_500_000, activation_block=9_501_900,
                            last_checked_boundary=9_876_000, source="tally")


def build_source(root: Path) -> Path:
    """A mainnet-shaped node dir: every state role, a ``.bak`` of the champion
    state, and the things that must never be bundled."""
    src = root / "mainnet-node"
    src.mkdir(parents=True)
    (src / SOURCE_NAMES["validator_state"]).write_text(champion.dumps(champion_state()))
    (src / (SOURCE_NAMES["validator_state"] + ".bak")).write_text(
        champion.dumps(champion_state(rounds_seen=40)))
    (src / SOURCE_NAMES["cascade_state"]).write_text(casc.dumps(cascade_state()))
    (src / SOURCE_NAMES["warm_start_init"]).write_text(json.dumps(
        {"generation": 2, "members": [{"checkpoint_id": cascade_state().members[0]}]}))
    (src / SOURCE_NAMES["netconfig_state"]).write_text(netconfig_state().to_json())
    for name in ACTIVATION_NAMES:
        ActivationStore(src / name).save(activation_record(name))
    # never bundled: keys, env, configs, logs, trainer db, a committed journal
    (src / "wallets" / "owner" / "hotkeys").mkdir(parents=True)
    (src / "wallets" / "owner" / "coldkey").write_text('{"fixture": "never copied"}')
    (src / "wallets" / "owner" / "hotkeys" / "default").write_text('{"fixture": "never"}')
    (src / ".env").write_text("HIPPIUS_S3_ACCESS_KEY=fixture-never-copied\n")
    (src / "chain.toml").write_text(MAINNET_TOML.read_text())
    (src / "validator.log").write_text("fixture log line\n")
    (src / "submissions.db").write_bytes(b"fixture trainer db")
    (src / (SOURCE_NAMES["validator_state"] + ".settlement.json")).write_text(
        '{"schema":1,"phase":"committed"}')
    return src


def expected_source_names() -> set[str]:
    names = set(SOURCE_NAMES.values()) | set(ACTIVATION_NAMES)
    names.add(SOURCE_NAMES["validator_state"] + ".bak")
    return names


def tree_digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def snapshot(src: Path, out: Path, *extra: object) -> tuple[Path, str]:
    """Run ``snapshot`` and return (bundle dir, printed sha256)."""
    proc = ok(run_cli("snapshot", "--source", src, "--config", MAINNET_TOML, "--out", out,
                      "--block", SOURCE_BLOCK, *extra))
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("BUNDLE ")]
    assert len(lines) == 1, f"expected one BUNDLE line:\n{proc.stdout}"
    parts = lines[0].split()
    assert len(parts) == 3 and parts[2].startswith("sha256="), lines[0]
    return Path(parts[1]), parts[2][len("sha256="):]


def manifest(bundle: Path) -> dict:
    return json.loads((bundle / "manifest.json").read_text())


def restore(bundle: Path, target: Path, *extra: object) -> subprocess.CompletedProcess:
    return run_cli("restore", "--bundle", bundle, "--target", target,
                   "--config", TESTNET_TOML, *extra)


def restored_source(tmp_path: Path) -> tuple[Path, Path, Path]:
    """(source dir, bundle dir, target dir) after a successful restore."""
    src = build_source(tmp_path)
    bundle, _ = snapshot(src, tmp_path / "bundles")
    target = tmp_path / "testnet-node"
    ok(restore(bundle, target))
    return src, bundle, target
