"""Shared spec and fixture builders for the W1.5 walking-skeleton tests.

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W1.5 row, §1.1-§1.5,
§3.2 invariants 1, 2, 9 and T13, §14.3, D1/D21/D48. README.md states the
module contract. Nothing here imports a W1.5 module at import time, so on
main every test fails at its own first import of a PATHS module (a
``ModuleNotFoundError`` / ``ImportError`` for a path in PATHS) or at an
``assert``, never at collection.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from statistics import NormalDist

import numpy as np

REPO = Path(__file__).resolve().parents[3]

# Era length 3600 blocks (900 x 4, mainnet grid): ARM_BLOCK is an era boundary.
ERA = 3600
SETTLEMENT = 900
ARM_BLOCK = 2700 * ERA                 # lane D from_block when armed
BOUNDARY_1 = ARM_BLOCK + SETTLEMENT    # first settlement that judges lane D
BOUNDARY_2 = ARM_BLOCK + 2 * SETTLEMENT
READY_BLOCK = ARM_BLOCK + 40
REVEAL_BLOCK = ARM_BLOCK + 60          # T_E (immediate reveal, D43)

HORIZON = 64                           # §1.1: lane D scores h = 64 only
CONTEXT = 4096                         # the declared context of the entry
HISTORY = 512                          # 16 patches: a patch multiple, no truncation
N_SOURCES = 6                          # clusters (sources)
PER_SOURCE = 8
HOUR = 3600
T0 = 1_790_000_000                     # wall time of the challenger's reveal (unix s)
HOUSE_T = T0 - 2 * 86_400              # the house's T (lane A king published earlier)
# Every target point (data time and first-seen) lies well after
# max(T_E, T_K) + forward_margin (6 h): every window is forward-eligible.
SNAPSHOT_START = {1: T0 + 12 * HOUR, 2: T0 + 12 * HOUR + 4 * 86_400}
FIRST_SEEN_LAG = 1800                  # first-seen = data time + 30 min

# §1.5 / lane A under quantile_head_after: the 9-level scoring grid.
SCORING_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Metagraph of the fixture: lane A's court is [king 3, prior king 4]; the lane D
# challenger is uid 6; uid 0 burns.
N_UIDS = 8
BURN_UID = 0
LANE_A_COURT = (3, 4)
LANE_A_KING_DECAY = 0.5
CHALLENGER_UID = 6

# The armed [lane_d] table (D1/D48: 0.25 of the total, taken from lane A).
# Floors and B are lowered so a 48-window fixture meets them (min_windows,
# min_clusters and bootstrap_B are W2 keys; their parse lower bounds allow it).
ARMED = {
    "from_block": ARM_BLOCK,
    "share": 0.25,
    "takes_from": "lane_a",
    "remainder": "burn",
    "contexts": [4096, 8192, 16384],
    "horizon": HORIZON,
    "forward_margin_hours": 6.0,
    "min_windows": 24,
    "min_clusters": 4,
    "bootstrap_B": 200,
}


def lane_d_cfg(**over):
    """The real W2 parse of ``ARMED`` with ``over`` applied."""
    from cascade.shared.config.lane_d import parse_lane_d

    return parse_lane_d({**ARMED, **over}, era_blocks_at=lambda _block: ERA)


def window_id(series_id: str, target_ts: list[int]) -> str:
    """§1.3 window identity: ``<series_id>:<target_start_ts>:h64``."""
    return f"{series_id}:{int(target_ts[0])}:h{HORIZON}"


def make_series(snapshot: int, seed: int) -> list[dict]:
    """The fixture series of one ``pool_d`` snapshot (no targets yet)."""
    rng = np.random.default_rng(seed)
    out = []
    t = np.arange(HISTORY, dtype=np.float64)
    for j in range(N_SOURCES):
        for k in range(PER_SOURCE):
            sid = f"src{j}-s{k}"
            level = 10.0 + 5.0 * j + k
            hist = level + 2.0 * np.sin(2 * np.pi * t / 24 + k) + rng.normal(0, 0.5, HISTORY)
            start = SNAPSHOT_START[snapshot] + (j * PER_SOURCE + k) * HOUR
            target_ts = [int(start + i * HOUR) for i in range(HORIZON)]
            out.append({
                "series_id": sid, "source": f"src{j}", "history": hist,
                "target_ts": target_ts,
                "first_seen_ts": [ts + FIRST_SEEN_LAG for ts in target_ts],
            })
    return out


def write_pool_d(directory: Path, snapshot: int, series: list[dict],
                 targets: dict[str, np.ndarray]) -> Path:
    """Write one ``pool_d/1`` snapshot (README "pool_d fixture format")."""
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "pool_d/1",
        "snapshot": f"w1_5-{snapshot}",
        "horizon": HORIZON,
        "series": [{
            "series_id": s["series_id"], "source": s["source"], "freq": "H",
            "seasonal_period": 24,
            "first_target_ts": s["target_ts"][0], "end_ts": s["target_ts"][-1],
            "target_ts": s["target_ts"], "first_seen_ts": s["first_seen_ts"],
        } for s in series],
    }
    (directory / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    np.savez(directory / "hist.npz", **{s["series_id"]: s["history"] for s in series})
    np.savez(directory / "tgt.npz",
             **{s["series_id"]: np.asarray(targets[s["series_id"]], dtype=np.float64)
                for s in series})
    return directory


_Z = tuple(NormalDist().inv_cdf(level) for level in SCORING_LEVELS)


def quantiles_around(median: np.ndarray, spread: float) -> np.ndarray:
    """A ``(horizon, 9)`` quantile forecast: ``median`` plus a normal spread."""
    return np.stack([median + spread * z for z in _Z], axis=-1)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture_hotkey(name: str) -> str:
    """A deterministic TEST ss58 address (never a real key)."""
    import bittensor as bt

    return bt.Keypair.create_from_uri(f"//w1_5-{name}").ss58_address


def make_signer(directory: Path):
    """A fresh sr25519 test keypair whose secret lives only in ``directory``."""
    import secrets

    import bittensor as bt

    directory.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(16)
    path = directory / "signer.uri"
    path.write_text(secret)
    path.chmod(0o600)
    return bt.Keypair.create_from_uri("//w1_5-signer-" + path.read_text())


def pyramid_contract():
    """The smallest pyramid size the code allows (``cascade-moe-v2-2m``), via
    the repo's own ``scripts/pyramid_config.py`` (loaded by path)."""
    from cascade.shared.config import parse_chain_config

    spec = importlib.util.spec_from_file_location(
        "w1_5_pyramid_config", REPO / "scripts" / "pyramid_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return parse_chain_config(mod.local_config("2m")).training


def tiny_checkpoint(directory: Path, contract, *, seed: int, median_shift: float,
                    bf16: bool) -> Path:
    """A random-init 2m pyramid checkpoint written by the real trainer code.

    ``median_shift`` adds to the head's median channel bias (standardized
    units), so the model forecasts ``loc + shift * scale``: the house gets 3.0,
    a clearly worse forecaster. ``bf16`` casts every floating tensor to BF16
    (D14: a lane D entry is BF16 only); the house stays F32 like a lane A king.
    """
    import torch

    from cascade.trainer.pyramid_model import HEAD_CHANNELS, CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint

    config = CascadeMoEConfig.from_contract(contract)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = CascadeMoEModel(config)
    if median_shift:
        with torch.no_grad():
            model.head.bias.view(config.patch_size, HEAD_CHANNELS)[:, 0] += median_shift
    state = {}
    for name, tensor in model.state_dict().items():
        t = tensor.detach()
        if bf16 and t.is_floating_point():
            t = t.to(torch.bfloat16)
        state[name] = t.contiguous().clone()
    save_checkpoint(directory, state, config, contract)
    return directory


# ── weight vectors (cascade.shared.emission) ─────────────────────────────────


def today_parts() -> tuple[list[float], list[float]]:
    """Today's lane parts for the fixture court (no lane B, C or D)."""
    from cascade.shared.emission import lane_parts

    return lane_parts(list(LANE_A_COURT), N_UIDS, king_decay=LANE_A_KING_DECAY,
                      burn_uid=BURN_UID)


def today_vector() -> list[float]:
    from cascade.shared.emission import EmissionPolicy, emission_vector

    a, _ = today_parts()
    return emission_vector(a, None, EmissionPolicy(), burn_uid=BURN_UID,
                           lane_a_uids=list(LANE_A_COURT))


def parts_with_lane_d(pay: dict | None, cfg) -> tuple[list[float], list[float]]:
    """The lane parts the validator builds from a settlement's ``pay`` (None =
    lane D did not pay at this settlement), as it does for lane C."""
    from cascade.shared.emission import lane_parts

    return lane_parts(list(LANE_A_COURT), N_UIDS, king_decay=LANE_A_KING_DECAY,
                      burn_uid=BURN_UID,
                      lane_d_cfg=(cfg if pay is not None else None),
                      d_court_uids=(tuple(pay["court_uids"]) if pay is not None else ()))


def vector_with_lane_d(pay: dict | None, cfg) -> list[float]:
    from cascade.shared.emission import EmissionPolicy, emission_vector

    a, b = parts_with_lane_d(pay, cfg)
    return emission_vector(a, b if pay is not None else None, EmissionPolicy(),
                           burn_uid=BURN_UID, lane_a_uids=list(LANE_A_COURT))


def as_bytes(vec) -> bytes:
    """Byte-identity of a weight vector: its float64 bytes."""
    values = [float(x) for x in vec]
    return struct.pack(f"<{len(values)}d", *values)
