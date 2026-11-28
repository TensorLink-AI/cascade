# W-RELFP: one release fingerprint, synced on every run (retro P4)

Remote hosts used to get their code once, at bootstrap. After that, someone reset them by hand on every release:
- The pool builder ran old code: #976, and 11-28 02:12.
- The lane D eval host was reset by hand four times on 11-28.
- #994 fixed the builder alone.

## Contract

1. **The fingerprint.** `cascade.shared.release_fp.Fingerprint` has three parts:
   - git `HEAD`;
   - `+dirty.<sha16 of git diff HEAD>` when tracked files have changed;
   - `package_hash`, which is byte-identical to the lane D handshake's.

   Its token is `<head12>[+dirty.x]/<pkg16>`.
2. **The probe.** The host probe (`probe_command`) uses only the standard library. A host with old code, or no code at all, still answers with its package hash, its git head and dirtiness, and the `.release_fp` written by the last sync.
3. **Check and sync.** `ensure_release(host, role=…)` runs at the start of every run.
   - **In sync:** the run proceeds.
   - **Skewed:**
     - In `release_sync = "sync"` mode a git checkout is moved to the release commit, and a hand-edited checkout is refused, never discarded. Any other workdir gets the release's `cascade/` as a tar stream, swapped in atomically, with the old tree kept as `cascade.prev` and `.release_fp` recorded. The host is then probed again. If it is still skewed, the run fails with `ReleaseSyncFailed` (`release_sync_failed`) and never runs on old code.
     - In `check` mode it only logs `release_skew` at ERROR and never blocks.
     - In `off` mode it does nothing.
   - **Default mode.** The mode comes from the host's TOML entry. If the entry has none, `$CASCADE_RELEASE_SYNC_DEFAULT` sets it, and otherwise it is `off`. External validators run this same code, so they are untouched unless they opt in.
4. **Where it is wired.**
   - Lane D eval host: `SSHEvalHost.hello`, before the W4 handshake. The handshake still fails closed.
   - MarketCast pod: `MktSession.start`, before the worker starts.
   - Private eval host: `forecasts_via_host`, before `check_private_worker`.
   - Pool builder: #994's `.release_sha` is now computed by `builder_token`. The format is unchanged, so warm builders do not re-sync.
   - Lane B pods: already get this release's package per content hash on every call.
5. **Box services.** Each box service logs `release fingerprint <token>` at start, from `verify_release_code_path`. With `CASCADE_RELEASE_STATUS_DIR` set, it also writes `<service>.json` for cascade-ops.
6. **Cleanup report.** `scripts/release_sprawl.py` lists release dirs and restart scripts as IN USE/LIVE/REFERENCED/STALE. It is report-only, and it reads nothing from a process environment except `PYTHONPATH`.

Out of scope: venv and dependency drift. The image and `uv sync --frozen` path (#999) owns that.
