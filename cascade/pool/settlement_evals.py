"""W-EVALPUB: publish each settlement's judged eval set once it is RETIRED,
and make the signed lane B / lane C manifests and lane D reports public-read
(owner 2026-11-28: "publish the eval dataset for lane D after it occurs, and
lane C: not 30 days"; ``tests/accept/w_evalpub/README.md``).

The box runs ``run`` hourly after the settlements (systemd timer
``cascade-settlement-evals``). Three independent steps:

* **lane D** (``lane-d/evals/<boundary>/{windows.npz, README.md,
  manifest.json}`` + ``lane-d/evals/index.json``): the windows a settlement
  judged (its signed report's ``windows``), each with its history and target
  as the settlement cut them, built on the lane D eval host
  (:mod:`cascade.eval.settlement_eval_build`; the box never loads a pool);
* **MarketCast / lane C** (``marketcast/evals/<round>/{windows.json,
  README.md, manifest.json}`` + ``marketcast/evals/index.json``): the windows
  the MarketCast judge drew for each entry of the round (wid, kind, span, the
  per-window losses, the draw record), the bars being the public post-embargo
  bars (``marketcast/public-bars/``);
* **ACL**: ``public-read`` on every ``lane_b/round-*``, ``lane_c/round-*`` and
  ``lane-d/reports/*`` object (per-object ACLs; idempotent; a local done-file
  keeps each run to the new keys).

**The safety rule: publish only what can never be judged again.**

*Lane D.* A window ``w`` judged at settlement ``N`` CAN be judged again: the
draw re-runs every settlement over the pinned snapshots, and ``w`` stays
eligible (``lane_d_forward`` rule 4) for any entry whose bound ``b =
max(T_E, T_K) + forward_margin`` has ``b - e < min(data_ts, first_seen)`` of
its target: every still-open entry, and every entry revealed up to ``e``
(``embargo_hours``, 48 h) after the target was first seen (as an embargo
window). So it is published only when retired: with ``T = min(now,
state_mtime, reveal_ts of every open entry) - slack`` (the
:func:`cascade.pool.pool_d_spent.protect_ts` of #997, slack 6 h) and ``cut =
T - 3600 E`` (``E`` = max(deployed, template) ``embargo_hours``), every
target point's data time and first-seen time ``<= cut``. Then every published
value has data time ``<= cut`` (the history precedes the target), and any
window holding one fails ``data_ok`` for every bound ``b >= T`` (``b - e >=
cut``), whatever the snapshot or band. Delay: the last judged target point's
first-seen + 54 h, i.e. at most 54 h after the settlement (the judged targets
were all seen by it), longer only while an older entry is still open.

*Lane C.* A judged window CAN be judged again for ~30 days: every entry's
backtests are drawn from the ``embargo_eval_days`` (28) before ITS dispatch,
so a bar judged for a leg dispatched at ``d`` (backtest or forward) is in the
backtest span of every leg dispatched before the bar's time + 28 d. A round
is retired when ``snapshot.end <= cut_c = T_c - 86400 E_c - 1`` with ``T_c =
min(now, train_time of every MarketCast entry still in the validator's lane C
state) - slack_c`` (72 h: legs dispatched and not yet registered) and ``E_c``
the largest ``embargo_days`` / ``embargo_eval_days`` (deployed, template,
recorded on any open entry): a future backtest starts after ``t - 86400
eval_days >= T_c - 86400 E_c > cut_c`` and a forward one after its dispatch,
so no bar at or before ``cut_c`` can be a future target. Delay: ~33 days.
Immediate publication needs a draw change (see the README).

Credentials by env NAME only (``HIPPIUS_S3_ACCESS_KEY`` /
``HIPPIUS_S3_SECRET_KEY``); values are never printed. Memory-light: reports
and states are JSON, the lane D file is streamed from the eval host to the
stage dir and uploaded from disk. No torch, no chain, no keys.

CLI::

    python -P -m cascade.pool.settlement_evals run --chain-toml C \\
        --stage DIR [--laned-state S --laned-inputs I] \\
        [--lane-c-state L] [--acl] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

HOUR = 3600
DAY = 86400
LANED_PREFIX = "lane-d/evals"
MKT_PREFIX = "marketcast/evals"
INDEX_NAME = "index.json"
LANED_SCHEMA = "lane-d-settlement-evals/1"
MKT_SCHEMA = "marketcast-round-evals/1"
LANED_INDEX_SCHEMA = "lane-d-evals-index/1"
MKT_INDEX_SCHEMA = "marketcast-evals-index/1"
LANED_FILES = ("windows.npz", "README.md", "manifest.json")      # manifest last
MKT_FILES = ("windows.json", "README.md", "manifest.json")
DEFAULT_LANED_SLACK_HOURS = 6.0
DEFAULT_MKT_SLACK_HOURS = 72.0
PUBLIC_ACL = "public-read"
# The signed objects every validator reads; ACLs are per object on Hippius.
PUBLIC_PREFIXES = ("lane_b/round-", "lane_c/round-", "lane-d/reports/")
REPORT_RE = re.compile(r"^lane_d_report_(\d+)\.json$")
MKT_REPORT_RE = re.compile(r"^marketcast/(?:([a-z0-9][a-z0-9_-]*)/)?round-([0-9A-Za-z_-]+)\.json$")
HOST_STAGE = "/root/lane_d_evals_stage"
MKT_PUBLIC_BARS = "marketcast/public-bars/index.json"
MKT_PUBLIC_BARS_HF = "https://huggingface.co/datasets/tensorlink-dev/cascade-marketcast-bars"


# ── lane D: the rule ────────────────────────────────────────────────────────


def laned_cut(state: Mapping, *, now: int, state_mtime: int | None, embargo_hours: float,
              slack_hours: float = DEFAULT_LANED_SLACK_HOURS) -> tuple[int | None, int | None]:
    """``(protect_ts, cut_ts)`` (#997's rule); ``(None, None)`` = publish
    nothing (an open entry's ``reveal_ts`` is malformed)."""
    from .pool_d_spent import leak_cut, protect_ts

    p = protect_ts(state, now=now, state_mtime=state_mtime, slack_hours=slack_hours)
    return (None, None) if p is None else (p, leak_cut(p, embargo_hours))


def window_retired(target_ts, first_seen_ts, *, cut: int, now: int) -> str | None:
    """See :func:`cascade.eval.settlement_eval_build.window_retired`."""
    from ..eval.settlement_eval_build import window_retired as rule

    return rule(target_ts, first_seen_ts, cut=cut, now=now)


def report_precheck(report: Mapping, *, cut: int) -> str | None:
    """The cheap, report-only part of the rule (``None`` = ask the host): a
    window whose origin (its first target point's data time) or min
    first-seen time is after ``cut`` is not retired."""
    for w in report.get("windows") or []:
        if int(w["origin"]) > cut or int(w["min_target_ts"]) > cut:
            return "target_after_cut"
        if int(w["min_first_seen_ts"]) > cut:
            return "first_seen_after_cut"
    return None


def build_request(report: Mapping, *, cut: int, now: int, history_cap: int) -> dict:
    """The eval host's build request of one report's judged windows."""
    from ..eval.settlement_eval_build import HISTORY_CAP, REQUEST_SCHEMA, parse_window_id

    snaps = {str(s["sha256"]): str(s["snapshot"])
             for s in (report.get("pool") or {}).get("snapshots") or []}
    ws = []
    for w in report.get("windows") or []:
        sid, origin, h = parse_window_id(w["window_id"])
        sha = str(w["snapshot_sha256"])
        if sha not in snaps:
            raise ValueError(f"{w['window_id']}: snapshot {sha} not in pool.snapshots")
        ws.append({"window_id": str(w["window_id"]), "series_id": sid, "origin": origin,
                   "h": h, "snapshot": snaps[sha], "snapshot_sha256": sha,
                   "band": str(w.get("band") or ""),
                   "embargo_window": bool(w.get("embargo_window", False))})
    return {"schema": REQUEST_SCHEMA, "boundary": int(report["boundary"]), "cut_ts": int(cut),
            "now": int(now), "history_cap": int(history_cap or HISTORY_CAP), "windows": ws}


def laned_reports(reports_dir: Path | str) -> list[tuple[int, Path]]:
    """``(boundary, path)`` of every persisted signed report, oldest first."""
    d = Path(reports_dir)
    out = []
    for p in d.glob("lane_d_report_*.json") if d.is_dir() else ():
        m = REPORT_RE.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def laned_plan(reports_dir, state: Mapping, *, now: int, state_mtime: int | None,
               embargo_hours: float, slack_hours: float = DEFAULT_LANED_SLACK_HOURS,
               published: Iterable[int] = ()) -> dict:
    """``{"protect_ts", "cut_ts", "candidates": [{boundary, path, windows}],
    "published": [b], "waiting": [{boundary, reason}], "empty": [b]}``.
    Candidates pass the report-only check; the host makes the final call."""
    protect, cut = laned_cut(state, now=now, state_mtime=state_mtime,
                             embargo_hours=embargo_hours, slack_hours=slack_hours)
    done = {int(b) for b in published}
    out = {"protect_ts": protect, "cut_ts": cut, "embargo_hours": float(embargo_hours),
           "slack_hours": float(slack_hours), "candidates": [], "published": [],
           "waiting": [], "empty": []}
    for b, p in laned_reports(reports_dir):
        if b in done:
            out["published"].append(b)
            continue
        if cut is None:
            out["waiting"].append({"boundary": b, "reason": "malformed_reveal_ts"})
            continue
        rep = json.loads(p.read_text(encoding="utf-8"))
        n = len(rep.get("windows") or [])
        if n == 0:
            out["empty"].append(b)
            continue
        why = report_precheck(rep, cut=cut)
        if why is not None:
            out["waiting"].append({"boundary": b, "reason": why})
        else:
            out["candidates"].append({"boundary": b, "path": str(p), "windows": n})
    return out


# ── lane D: the eval host ───────────────────────────────────────────────────


class EvalHost:
    """The lane D eval host of the validator's ``--lane-d-inputs`` file
    (``eval_hosts``, ``eval_host_name``, ``eval_pool_root``), over SSH."""

    def __init__(self, inputs_path, *, timeout_s: float = 3600.0, run: Callable | None = None):
        import tomllib

        raw = tomllib.loads(Path(inputs_path).read_text(encoding="utf-8"))
        self.hosts_path = Path(raw["eval_hosts"])
        self.host_name = str(raw.get("eval_host_name") or "")
        self.pool_root = str(raw.get("eval_pool_root") or "")
        if not self.pool_root:
            raise ValueError(f"{inputs_path}: eval_pool_root unset (W-OFFBOX host copy)")
        self.timeout_s = float(timeout_s)
        self._run = run or subprocess.run

    def _host(self):
        from ..trainer.remote import load_hosts

        hosts = load_hosts(self.hosts_path)
        if self.host_name:
            hosts = [h for h in hosts if h.name == self.host_name]
        if not hosts:
            raise RuntimeError(f"{self.hosts_path}: no lane D eval host {self.host_name!r}")
        return hosts[0]

    def _argv(self, host, cmd: str) -> list[str]:
        from ..trainer.remote import build_ssh_argv

        return build_ssh_argv(host, f"cd {shlex.quote(host.workdir)} && {cmd}")

    @staticmethod
    def out_dir(boundary: int) -> str:
        return f"{HOST_STAGE}/{int(boundary)}"

    def build(self, request: Mapping) -> dict:
        host = self._host()
        bound = max(1, int(self.timeout_s) - 30)
        cmd = " ".join(shlex.quote(a) for a in [
            "timeout", "-k", "30", str(bound), "nice", "-n", "19", "ionice", "-c", "3",
            host.remote_python, "-m", "cascade.eval.settlement_eval_build", "build",
            "--pool-root", self.pool_root, "--out-dir", self.out_dir(request["boundary"])])
        proc = self._run(self._argv(host, cmd), input=json.dumps(request).encode(),
                         capture_output=True, timeout=self.timeout_s, check=False)
        lines = [ln for ln in (proc.stdout or b"").decode("utf-8", "replace").splitlines()
                 if ln.strip()]
        if proc.returncode != 0 or not lines:
            err = (proc.stderr or b"").decode("utf-8", "replace")[-400:]
            raise RuntimeError(f"{host.name}: eval build exited {proc.returncode}: {err}")
        return json.loads(lines[-1])

    def fetch(self, boundary: int, dest: Path) -> None:
        """Stream the host's ``windows.npz`` to ``dest`` (disk, never memory)."""
        host = self._host()
        src = f"{self.out_dir(boundary)}/windows.npz"
        with open(dest, "wb") as f:
            proc = self._run(self._argv(host, f"cat {shlex.quote(src)}"), stdout=f,
                             stderr=subprocess.PIPE, timeout=self.timeout_s, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"{host.name}: fetch {src} exited {proc.returncode}")

    def cleanup(self, boundary: int) -> None:
        host = self._host()
        self._run(self._argv(host, f"rm -rf -- {shlex.quote(self.out_dir(boundary))}"),
                  capture_output=True, timeout=600, check=False)


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_npz(npz: Path, *, cut: int, now: int) -> dict:
    """Re-check the fetched file on the box, streaming (belt and braces):
    every target clock ``<= cut`` (and ``<= now``); returns the maxima."""
    from ..eval.settlement_eval_build import npy_max

    with zipfile.ZipFile(npz) as z:
        tts, fs, org = npy_max(z, "target_ts"), npy_max(z, "first_seen_ts"), npy_max(z, "origin_ts")
    for name, v, lim in (("target_ts", tts, min(cut, now)), ("first_seen_ts", fs, cut),
                         ("origin_ts", org, cut)):
        if v is not None and v > lim:
            raise EmbargoViolation(f"{npz}: {name} max {v} after the cut {lim}")
    return {"max_target_ts": tts, "max_first_seen_ts": fs}


class EmbargoViolation(RuntimeError):
    """A window that may still be judged reached a file to publish."""


def laned_manifest(report: Mapping, report_path: Path, reply: Mapping, *, plan: Mapping,
                   now: int, base_url: str, npz: Path) -> dict:
    from ..validator.lane_d_wiring import report_key

    b = int(report["boundary"])
    rk = report_key(b)
    return {
        "schema": LANED_SCHEMA, "boundary": b, "boundary_ts": report.get("boundary_ts"),
        "kind": "the windows this settlement judged (its signed report's windows), "
                "each as the settlement cut it",
        "report_key": rk, "report_url": f"{base_url}/{rk}",
        "report_sha256": _sha_file(report_path), "rows_sha256": report.get("rows_sha256"),
        "seed": (report.get("draw") or {}).get("seed"),
        "snapshots": reply.get("snapshots"), "history_cap": reply.get("history_cap"),
        "windows": reply.get("windows"), "built_ts": int(now),
        "retired_rule": {"protect_ts": plan["protect_ts"], "cut_ts": plan["cut_ts"],
                         "embargo_hours": plan["embargo_hours"],
                         "slack_hours": plan["slack_hours"],
                         "statement": "every target point's data time and first-seen time "
                                      "<= cut_ts (history precedes the target): no published "
                                      "value can enter a window eligible for any entry "
                                      "revealed at or after protect_ts"},
        "max_target_ts": reply.get("max_target_ts"),
        "max_first_seen_ts": reply.get("max_first_seen_ts"),
        "files": {"windows.npz": {"sha256": _sha_file(npz), "bytes": npz.stat().st_size}},
    }


def laned_readme(man: Mapping) -> str:
    r, f = man["retired_rule"], man["files"]["windows.npz"]
    return f"""# Lane D settlement `{man['boundary']}`: the judged windows

The {man['windows']} windows the lane D settlement at boundary `{man['boundary']}` judged,
each with its history (up to {man['history_cap']} points) and target exactly as the
settlement cut them. Per-window scores per model are the `rows` of the signed report
[`{man['report_key']}`]({man['report_url']}) (sha256 `{man['report_sha256']}`).

- **Why it is public now:** every target point was first seen, and lies, at or before
  `cut_ts = {r['cut_ts']}` (= `protect_ts` {r['protect_ts']} − {r['embargo_hours']} h embargo;
  `protect_ts` = the earliest reveal of any entry still to be judged − {r['slack_hours']} h).
  No value here can be in a window still eligible for any entry, so none of these windows
  can be judged again.
- **Integrity:** `windows.npz` sha256 `{f['sha256']}` ({f['bytes']} bytes).

## Load

```python
import numpy as np
z = np.load("windows.npz")
i = 0
c = int(z["n_channels"][i])
h0, h1 = z["history_offsets"][i], z["history_offsets"][i + 1]
t0, t1 = z["target_offsets"][i], z["target_offsets"][i + 1]
history = z["history"][h0:h1].reshape(c, -1)          # float32, channel-major
target = z["target"][t0:t1].reshape(c, -1)            # (c, h)
k0, k1 = z["time_offsets"][i], z["time_offsets"][i + 1]
target_ts, first_seen = z["target_ts"][k0:k1], z["first_seen_ts"][k0:k1]
print(z["window_id"][i], z["band"][i], z["h"][i])
```

Columns: `history`, `history_offsets`, `target`, `target_offsets`, `target_ts`,
`first_seen_ts`, `time_offsets`, `window_id`, `series_id`, `snapshot`, `source`,
`feed`, `freq`, `band`, `h`, `origin_ts`, `n_channels`, `history_len`,
`seasonal_period`, `embargo_window`. Every published settlement is listed in
`../index.json`.
"""


# ── the store side (shared) ─────────────────────────────────────────────────


def _list_keys(client, bucket: str, prefix: str) -> list[dict]:
    out, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = client.list_objects_v2(**kw)
        out += list(page.get("Contents") or [])
        if not page.get("IsTruncated"):
            return out
        token = page.get("NextContinuationToken")


def _get_json(client, bucket: str, key: str) -> dict | None:
    try:
        return json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception as e:                                       # noqa: BLE001
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def _ctype(name: str) -> str:
    return {"json": "application/json", "md": "text/markdown",
            "npz": "application/octet-stream"}[name.rsplit(".", 1)[-1]]


def published_ids(client, bucket: str, prefix: str) -> list[str]:
    """Ids ``<prefix>/<id>/manifest.json`` exists for."""
    out = set()
    for o in _list_keys(client, bucket, prefix + "/"):
        rest = o["Key"][len(prefix) + 1:]
        head, _, tail = rest.partition("/")
        if head and tail == "manifest.json":
            out.add(head)
    return sorted(out)


def upload_dir(client, bucket: str, prefix: str, stage: Path, names) -> list[str]:
    """``public-read`` uploads from disk, ``manifest.json`` last."""
    keys = []
    for n in names:
        key = f"{prefix}/{n}"
        client.upload_file(str(Path(stage) / n), bucket, key,
                           ExtraArgs={"ACL": PUBLIC_ACL, "ContentType": _ctype(n)})
        keys.append(key)
    return keys


def refresh_index(client, bucket: str, prefix: str, *, schema: str, base_url: str, now: int,
                  entry: Callable[[str, dict], dict], waiting: list, dry_run: bool) -> tuple:
    """Rebuild ``<prefix>/index.json`` from the published manifests; PUT it
    only when its list (or the waiting list) changed."""
    items = [entry(i, _get_json(client, bucket, f"{prefix}/{i}/manifest.json") or {})
             for i in published_ids(client, bucket, prefix)]
    doc = {"schema": schema, "generated_ts": int(now), "count": len(items), "items": items,
           "waiting": waiting}
    old = _get_json(client, bucket, f"{prefix}/{INDEX_NAME}") or {}
    if old.get("items") == doc["items"] and old.get("waiting") == doc["waiting"]:
        return old, False
    if not dry_run:
        client.put_object(Bucket=bucket, Key=f"{prefix}/{INDEX_NAME}",
                          Body=(json.dumps(doc, indent=1, sort_keys=True) + "\n").encode(),
                          ContentType="application/json", ACL=PUBLIC_ACL)
    return doc, True


def _laned_entry(base_url: str):
    def entry(i: str, man: dict) -> dict:
        f = (man.get("files") or {}).get("windows.npz") or {}
        p = f"{LANED_PREFIX}/{i}"
        return {"boundary": int(i), "windows": man.get("windows"),
                "windows_sha256": f.get("sha256"), "windows_bytes": f.get("bytes"),
                "cut_ts": (man.get("retired_rule") or {}).get("cut_ts"),
                "built_ts": man.get("built_ts"), "report_url": man.get("report_url"),
                "manifest_url": f"{base_url}/{p}/manifest.json",
                "windows_url": f"{base_url}/{p}/windows.npz",
                "readme_url": f"{base_url}/{p}/README.md"}
    return entry


def publish_laned(*, client, bucket: str, base_url: str, reports_dir: Path, state: Mapping,
                  state_mtime: int, embargo_hours: float, host, stage: Path, now: int,
                  slack_hours: float = DEFAULT_LANED_SLACK_HOURS, history_cap: int = 16384,
                  max_per_run: int = 2, dry_run: bool = False,
                  log: Callable[[str], None] = print) -> dict:
    """One lane D pass: plan, build each candidate on the host, verify,
    upload (unless ``dry_run``), refresh the index."""
    published = [int(x) for x in published_ids(client, bucket, LANED_PREFIX) if x.isdigit()]
    plan = laned_plan(reports_dir, state, now=now, state_mtime=state_mtime,
                      embargo_hours=embargo_hours, slack_hours=slack_hours, published=published)
    log("evals laned: " + json.dumps({k: plan[k] for k in (
        "protect_ts", "cut_ts", "embargo_hours", "slack_hours", "published", "waiting",
        "empty")} | {"candidates": [c["boundary"] for c in plan["candidates"]]},
        sort_keys=True))
    waiting = list(plan["waiting"])
    done = []
    for c in plan["candidates"][:max(0, int(max_per_run))]:
        b, rp = int(c["boundary"]), Path(c["path"])
        rep = json.loads(rp.read_text(encoding="utf-8"))
        req = build_request(rep, cut=int(plan["cut_ts"]), now=now, history_cap=history_cap)
        reply = host.build(req)
        if not reply.get("ok"):
            waiting.append({"boundary": b, "reason": str(reply.get("code"))})
            log(f"evals laned: {b} waits ({reply.get('code')}: "
                f"{reply.get('detail') or reply.get('n_waiting')})")
            continue
        d = Path(stage) / f"laned-{b}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        try:
            npz = d / "windows.npz"
            host.fetch(b, npz)
            if _sha_file(npz) != reply.get("sha256"):
                raise RuntimeError(f"{b}: fetched windows.npz does not hash to the host's reply")
            verify_npz(npz, cut=int(plan["cut_ts"]), now=now)
            man = laned_manifest(rep, rp, reply, plan=plan, now=now, base_url=base_url, npz=npz)
            (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n",
                                             encoding="utf-8")
            (d / "README.md").write_text(laned_readme(man), encoding="utf-8")
            sizes = ", ".join(f"{n} ({(d / n).stat().st_size} B)" for n in LANED_FILES)
            log(f"evals laned: {'would upload' if dry_run else 'upload'} "
                f"{LANED_PREFIX}/{b}/ windows={man['windows']}: {sizes}")
            if not dry_run:
                upload_dir(client, bucket, f"{LANED_PREFIX}/{b}", d, LANED_FILES)
            done.append(b)
        finally:
            shutil.rmtree(d, ignore_errors=True)
            host.cleanup(b)
    doc, wrote = refresh_index(client, bucket, LANED_PREFIX, schema=LANED_INDEX_SCHEMA,
                               base_url=base_url, now=now, entry=_laned_entry(base_url),
                               waiting=sorted(waiting, key=lambda x: x["boundary"]),
                               dry_run=dry_run)
    log(f"evals laned: index {LANED_PREFIX}/{INDEX_NAME} items={doc.get('count')} "
        + ("unchanged" if not wrote else ("would write" if dry_run else "written")))
    return {"plan": plan, "built": done}


# ── MarketCast / lane C ─────────────────────────────────────────────────────


def _template_mkt():
    from ..shared.config.marketcast import MarketCastConfig

    return MarketCastConfig()


def _walk(obj, key: str):
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _open_mkt_records(lane_c_state: Mapping) -> list:
    """The entry records of every MarketCast entry the state may still judge:
    every ``mkt_backlog`` candidate, every ``mkt:`` pending entry, and every
    record under another competition's state (``comp``)."""
    recs = []
    for rnd in lane_c_state.get("mkt_backlog") or []:
        for c in (rnd or {}).get("candidates") or []:
            recs.append((c or {}).get("entry"))
    for p in lane_c_state.get("pending") or []:
        if str((p or {}).get("target", "")).startswith("mkt:"):
            recs.append((p or {}).get("entry"))
    for comp in (lane_c_state.get("comp") or {}).values():
        recs += [{"marketcast": m} for m in _walk(comp, "marketcast")]
    return recs


def mkt_cut(lane_c_state: Mapping, *, now: int, embargo_days: Iterable[int],
            slack_hours: float = DEFAULT_MKT_SLACK_HOURS) -> tuple[int | None, int | None, int]:
    """``(T_c, cut_c, E_c)``. ``T_c`` = min(now, every ``train_time`` of a
    MarketCast entry the state may still judge, :func:`_open_mkt_records`) -
    slack; ``E_c`` = the largest embargo day count (the given ones and every
    open entry's recorded ``embargo_days`` / ``embargo_eval_days``);
    ``cut_c = T_c - 86400 E_c - 1``. ``(None, None, E_c)`` (publish nothing)
    when an open entry has no ``marketcast`` record or a malformed
    ``train_time``."""
    ts = [float(now)]
    days = [int(d) for d in embargo_days]
    for e in _open_mkt_records(lane_c_state):
        mc = (e or {}).get("marketcast") if isinstance(e, Mapping) else None
        t = mc.get("train_time") if isinstance(mc, Mapping) else None
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            return None, None, max(days)
        ts.append(float(t))
        for k in ("embargo_days", "embargo_eval_days"):
            v = mc.get(k)
            if isinstance(v, int) and not isinstance(v, bool) and v > 0:
                days.append(v)
    e_c = max(days)
    t_c = int(math.floor(min(ts))) - round(float(slack_hours) * HOUR)
    return t_c, t_c - e_c * DAY - 1, e_c


def mkt_round_retired(report: Mapping, *, cut: int) -> str | None:
    """``None`` when every bar any window of the round used (all lie in its
    snapshot, at or before ``snapshot.end``) is at or before ``cut``."""
    snap = report.get("snapshot") or {}
    end = snap.get("end")
    if isinstance(end, bool) or not isinstance(end, (int, float)):
        return "missing_snapshot_end"
    return None if float(end) <= float(cut) else "snapshot_after_cut"


def parse_mkt_wid(wid: str) -> dict:
    """``<class>:<asset>|<suite>|<horizon>|<start>``."""
    ca, suite, horizon, start = str(wid).split("|")
    cls, _, asset = ca.partition(":")
    return {"class": cls, "asset": asset, "suite": suite, "horizon": horizon,
            "start": float(start)}


def mkt_windows(report: Mapping) -> list[dict]:
    """One row per (entry, window): the entry's identity, phase and draw
    kind, the window's span and its signed per-window losses."""
    out = []
    for e in report.get("entries") or []:
        draw = e.get("draw") or {}
        for w in e.get("windows") or []:
            row = {"hotkey": e.get("hotkey"), "target": e.get("target"), "phase": e.get("phase"),
                   "trained_pointer": e.get("trained_pointer"), "wid": w["wid"],
                   **parse_mkt_wid(w["wid"]),
                   # cascade.pool.marketcast.window_kind (not imported: heavy)
                   "kind": "forward" if not draw or parse_mkt_wid(w["wid"])["start"]
                   > float(draw["hi"]) else "backtest",
                   "base": w.get("base"), "ft": w.get("ft")}
            out.append(row)
    return sorted(out, key=lambda r: (r["wid"], str(r["hotkey"]), str(r["phase"])))


def mkt_round_files(report: Mapping, report_bytes: bytes, key: str, *, plan: Mapping,
                    now: int, base_url: str, stage: Path) -> dict:
    rid = str(report["round_id"])
    rows = mkt_windows(report)
    (stage / "windows.json").write_text(json.dumps(
        {"round_id": rid, "windows": rows}, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    snap = report.get("snapshot") or {}
    man = {
        "schema": MKT_SCHEMA, "round_id": rid, "report_key": key,
        "report_url": f"{base_url}/{key}",
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "snapshot": {k: snap.get(k) for k in ("key", "sha256", "digest", "t0", "end")},
        "entries": [{"hotkey": e.get("hotkey"), "target": e.get("target"),
                     "phase": e.get("phase"), "status": e.get("status") or "computed",
                     "windows": len(e.get("windows") or []), "draw": e.get("draw") or {}}
                    for e in report.get("entries") or []],
        "windows": len(rows), "distinct_windows": len({r["wid"] for r in rows}),
        "bars": {"hippius_index": f"{base_url}/{MKT_PUBLIC_BARS}", "hf_dataset": MKT_PUBLIC_BARS_HF,
                 "note": "every bar of these windows is at or before cut_ts, older than the "
                         "public-bars embargo: cut them from the public post-embargo bars"},
        "retired_rule": {k: plan[k] for k in ("protect_ts", "cut_ts", "embargo_days",
                                              "slack_hours")}
        | {"statement": "snapshot.end <= cut_ts = protect_ts - 86400 * embargo_days - 1: "
                        "every later draw's backtests start after its dispatch minus "
                        "embargo_eval_days and its forwards after its dispatch, so no bar of "
                        "this round can be a target again"},
        "built_ts": int(now),
        "files": {"windows.json": {"sha256": _sha_file(stage / "windows.json"),
                                   "bytes": (stage / "windows.json").stat().st_size}},
    }
    (stage / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n",
                                         encoding="utf-8")
    (stage / "README.md").write_text(f"""# MarketCast round `{rid}`: the judged windows

Every window the MarketCast judge scored in round `{rid}` ({man['windows']} rows,
{man['distinct_windows']} distinct windows): per entry and phase, the window id
`<class>:<asset>|<suite>|<horizon>|<start>`, its kind (`backtest` = drawn from the
embargo before the leg's dispatch, `forward` = after it), its start, and the signed
per-window losses (`base` = the pinned base, `ft` = the fine-tune). The draw records
(block hash, spans) are in `manifest.json`; the full signed report is
[`{key}`]({man['report_url']}).

**Bars.** Every bar of these windows is older than the public-bars embargo, so they are
in the public post-embargo bars: {man['bars']['hf_dataset']} and
`{MKT_PUBLIC_BARS}`. The judged snapshot was `{snap.get('key')}` (sha256
`{snap.get('sha256')}`, bars `{snap.get('t0')}`..`{snap.get('end')}`).

**Why it is public now:** the snapshot ends at or before `cut_ts =
{plan['cut_ts']}` (the earliest dispatch of any entry still to be judged, minus
{plan['slack_hours']} h, minus {plan['embargo_days']} days): no later draw can use these bars
as a target.
""", encoding="utf-8")
    return man


def _mkt_entry(base_url: str):
    def entry(i: str, man: dict) -> dict:
        p = f"{MKT_PREFIX}/{i}"
        return {"round_id": man.get("round_id") or i, "windows": man.get("windows"),
                "snapshot_end": (man.get("snapshot") or {}).get("end"),
                "cut_ts": (man.get("retired_rule") or {}).get("cut_ts"),
                "built_ts": man.get("built_ts"), "report_url": man.get("report_url"),
                "manifest_url": f"{base_url}/{p}/manifest.json",
                "windows_url": f"{base_url}/{p}/windows.json",
                "readme_url": f"{base_url}/{p}/README.md"}
    return entry


def _mkt_id(key: str) -> str | None:
    m = MKT_REPORT_RE.match(key)
    if not m or (m.group(1) or "") in ("evals", "public-bars"):
        return None
    return f"{m.group(1)}-{m.group(2)}" if m.group(1) else m.group(2)


def publish_mkt(*, client, bucket: str, base_url: str, lane_c_state: Mapping,
                embargo_days: Iterable[int], stage: Path, now: int,
                slack_hours: float = DEFAULT_MKT_SLACK_HOURS, dry_run: bool = False,
                log: Callable[[str], None] = print) -> dict:
    t_c, cut, e_c = mkt_cut(lane_c_state, now=now, embargo_days=embargo_days,
                            slack_hours=slack_hours)
    plan = {"protect_ts": t_c, "cut_ts": cut, "embargo_days": e_c,
            "slack_hours": float(slack_hours)}
    done_ids = set(published_ids(client, bucket, MKT_PREFIX))
    built, waiting = [], []
    for o in sorted(_list_keys(client, bucket, "marketcast/"), key=lambda o: o["Key"]):
        rid = _mkt_id(o["Key"])
        if rid is None or rid in done_ids:
            continue
        if cut is None:
            waiting.append({"round_id": rid, "reason": "malformed_train_time"})
            continue
        lm = o.get("LastModified")
        lm_ts = lm.timestamp() if hasattr(lm, "timestamp") else None
        if lm_ts is not None and lm_ts > cut + 2 * DAY:
            continue                         # written well after the cut: not read yet
                                             # (its snapshot ends near its write; a skip
                                             # only delays, the rule below decides)
        body = client.get_object(Bucket=bucket, Key=o["Key"])["Body"].read()
        rep = json.loads(body)
        why = mkt_round_retired(rep, cut=cut)
        if why is not None:
            continue                         # waits silently (~33 days by design)
        if not rep.get("entries") or not any(e.get("windows") for e in rep["entries"]):
            continue                         # nothing judged (all deferred)
        d = Path(stage) / f"mkt-{rid}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        try:
            man = mkt_round_files(rep, body, o["Key"], plan=plan, now=now, base_url=base_url,
                                  stage=d)
            log(f"evals mkt: {'would upload' if dry_run else 'upload'} {MKT_PREFIX}/{rid}/ "
                f"windows={man['windows']}")
            if not dry_run:
                upload_dir(client, bucket, f"{MKT_PREFIX}/{rid}", d, MKT_FILES)
            built.append(rid)
        finally:
            shutil.rmtree(d, ignore_errors=True)
    log("evals mkt: " + json.dumps(plan | {"built": built, "waiting": waiting}, sort_keys=True))
    doc, wrote = refresh_index(client, bucket, MKT_PREFIX, schema=MKT_INDEX_SCHEMA,
                               base_url=base_url, now=now, entry=_mkt_entry(base_url),
                               waiting=waiting, dry_run=dry_run)
    log(f"evals mkt: index {MKT_PREFIX}/{INDEX_NAME} items={doc.get('count')} "
        + ("unchanged" if not wrote else ("would write" if dry_run else "written")))
    return {"plan": plan, "built": built}


# ── ACL backfill ────────────────────────────────────────────────────────────


def acl_backfill(client, bucket: str, *, done_file: Path, prefixes=PUBLIC_PREFIXES,
                 dry_run: bool = False, log: Callable[[str], None] = print) -> dict:
    """``public-read`` on every object under ``prefixes`` not yet in
    ``done_file`` (one key per line). Idempotent."""
    done_file = Path(done_file)
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.is_file() else set()
    todo = []
    for p in prefixes:
        todo += [o["Key"] for o in _list_keys(client, bucket, p) if o["Key"] not in done]
    todo = sorted(set(todo))
    applied = []
    for k in todo:
        if not dry_run:
            client.put_object_acl(Bucket=bucket, Key=k, ACL=PUBLIC_ACL)
            applied.append(k)
    if applied:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        with open(done_file, "a", encoding="utf-8") as f:
            f.write("".join(k + "\n" for k in applied))
    log(f"evals acl: {len(todo)} object(s) {'would get' if dry_run else 'set'} {PUBLIC_ACL} "
        f"under {list(prefixes)}" + (f", e.g. {todo[:3]}" if todo else ""))
    return {"todo": todo, "applied": applied}


# ── CLI ─────────────────────────────────────────────────────────────────────


def _client(chain_toml: Path):
    from ..shared.config import load_chain_config
    from ..shared.hippius import S3Config, S3Store

    st = load_chain_config(chain_toml).storage
    bucket = str(getattr(st, "manifest_bucket", "") or "")
    if not bucket:
        raise SystemExit("[storage] manifest_bucket unset")
    endpoint = str(getattr(st, "s3_endpoint", "") or "https://s3.hippius.com").rstrip("/")
    return S3Store(S3Config.from_storage(st, bucket=bucket)).client(), bucket, \
        f"{endpoint}/{bucket}"


def cmd_run(a) -> int:
    from ..shared.config import load_chain_config
    from .pool_d_spent import effective_embargo_hours

    now = int(a.now) if a.now is not None else int(time.time())
    cfg = load_chain_config(a.chain_toml)
    client, bucket, base_url = _client(a.chain_toml)
    stage = Path(a.stage)
    stage.mkdir(parents=True, exist_ok=True)
    rc = 0
    if a.laned_state:
        try:
            sp = Path(a.laned_state)
            state = json.loads(sp.read_text(encoding="utf-8"))
            reports = sp.parent / f"{sp.stem}_reports"
            host = EvalHost(a.laned_inputs)
            publish_laned(client=client, bucket=bucket, base_url=base_url, reports_dir=reports,
                          state=state, state_mtime=int(os.stat(sp).st_mtime),
                          embargo_hours=effective_embargo_hours(cfg.lane_d.embargo_hours),
                          host=host, stage=stage, now=now, slack_hours=a.laned_slack_hours,
                          history_cap=a.history_cap, dry_run=a.dry_run)
        except Exception as e:                                   # noqa: BLE001
            print(f"evals laned: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            rc = 2
    if a.lane_c_state:
        try:
            st = json.loads(Path(a.lane_c_state).read_text(encoding="utf-8"))
            tpl, dep = _template_mkt(), cfg.marketcast
            days = [int(x) for x in (tpl.embargo_days, tpl.embargo_eval_days,
                                     getattr(dep, "embargo_days", 0) or 0,
                                     getattr(dep, "embargo_eval_days", 0) or 0)]
            publish_mkt(client=client, bucket=bucket, base_url=base_url, lane_c_state=st,
                        embargo_days=days, stage=stage, now=now,
                        slack_hours=a.mkt_slack_hours, dry_run=a.dry_run)
        except Exception as e:                                   # noqa: BLE001
            print(f"evals mkt: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            rc = 2
    if a.acl:
        try:
            acl_backfill(client, bucket, done_file=stage / "acl_done.txt", dry_run=a.dry_run)
        except Exception as e:                                   # noqa: BLE001
            print(f"evals acl: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            rc = 2
    return rc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m cascade.pool.settlement_evals",
                                 description="publish retired settlement eval sets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--chain-toml", type=Path, required=True)
    p.add_argument("--stage", type=Path, required=True, help="scratch dir (one set at a time)")
    p.add_argument("--laned-state", type=Path, default=None, help="lane_d_state.json")
    p.add_argument("--laned-inputs", type=Path, default=None, help="the --lane-d-inputs file")
    p.add_argument("--laned-slack-hours", type=float, default=DEFAULT_LANED_SLACK_HOURS)
    p.add_argument("--history-cap", type=int, default=16384)
    p.add_argument("--lane-c-state", type=Path, default=None,
                   help="the validator's lane C state (validator_state.db.lane_c.json)")
    p.add_argument("--mkt-slack-hours", type=float, default=DEFAULT_MKT_SLACK_HOURS)
    p.add_argument("--acl", action="store_true", help="public-read on the signed manifests")
    p.add_argument("--now", type=int, default=None, help="unix seconds (tests)")
    p.add_argument("--dry-run", action="store_true", help="build + verify; upload nothing")
    a = ap.parse_args(argv)
    if a.laned_state and not a.laned_inputs:
        ap.error("--laned-state needs --laned-inputs")
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EmbargoViolation", "EvalHost", "LANED_PREFIX", "MKT_PREFIX", "PUBLIC_PREFIXES",
    "acl_backfill", "build_request", "laned_cut", "laned_plan", "mkt_cut", "mkt_round_retired",
    "mkt_windows", "publish_laned", "publish_mkt", "report_precheck", "verify_npz",
    "window_retired",
]
