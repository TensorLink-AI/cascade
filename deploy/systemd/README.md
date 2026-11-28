# Box services as systemd units: migration runbook (retro P2 ii)

**Status: the units are files only. No tool installs them.** The owner runs every
step below by hand, one service at a time. Each step is Approve-tier because it
restarts a live service. Before a step for the trainer or validator, run
`cascade-ops restart-check <svc>` and go ahead only on GO.

## Why

Today the validator, trainer, intake and MarketCast provisioner run from 63
hand-edited `restart_*.sh` nohup scripts. Those scripts have no memory limit. A
heavy in-process step can OOM-kill whatever the kernel picks: three validator
deaths on 11-28, and on 10-31 every service died. The units here:

| unit | MemoryHigh / MemoryMax | OOMScoreAdjust | Restart | box guard |
|---|---|---|---|---|
| `cascade-validator` | 2G / 3G | -900 (protected) | always (replaces the watchdog) | enforce |
| `cascade-trainer` | 1500M / 2G | 0 | **no** (restarts have side effects) | warn, then enforce |
| `cascade-intake` | 384M / 512M | 200 | on-failure | enforce |
| `cascade-mkt-provisioner` | 512M / 1G | 200 | on-failure | enforce |

- `MemoryMax` kills the offending unit only, never a neighbour.
- `MemoryHigh` throttles and reclaims memory first.
- `MemorySwapMax` keeps a runaway unit from filling the 6 G swap.
- The lane D intake and the timers already have units with `MemoryMax`. They are
  unchanged.

## What changes in how a service is started

- **Code.** Code comes from `PYTHONPATH=/root/cascade-releases/<svc>.current`. That is a
  symlink to a release checkout. A release switch becomes
  `ln -sfn /root/cascade-val-<sha> /root/cascade-releases/<svc>.current`
  followed by `systemctl restart`. You no longer write a new restart script.
- **Flags.** Flags live in `/etc/cascade/<svc>.args` (mode 600; examples are in `args/`).
  They are one variable that systemd splits on whitespace.
- **Box guard.** `CASCADE_BOX_GUARD` sets the runtime box guard
  (`cascade/shared/box_guard.py`). The validator entry point defaults it to OFF,
  because external validators run the same entry point on GPU hosts. On our box,
  the unit sets `enforce`. The trainer starts in `warn`: today it has torch mapped
  from a path nobody has named yet. Read `trainer.log` for `box_guard_refused`
  across one era, fix or allow each path, then set `enforce`.
- **Stopping.** `KillMode=process` on the validator and trainer. Only the main pid gets TERM, as
  with the scripts, so ssh sessions to pods and harvests are not cut.

## Pre-flight (once)

1. `/root/cascade/.env` has 2 `export NAME=…` lines. systemd's `EnvironmentFile=`
   does not take the `export` prefix. Make a systemd copy without it:
   `sed 's/^export //' /root/cascade/.env > /etc/cascade/cascade.env && chmod 600 /etc/cascade/cascade.env`.
   The units read that copy. Never print the file.
2. `mkdir -p /root/cascade-releases /etc/cascade && chmod 700 /etc/cascade`.
3. Run `systemd-analyze verify deploy/systemd/*.service`. Expect no errors other than the
   missing args files.

## Per service (repeat for intake → mkt-provisioner → validator → trainer)

1. Read the live command: `ps -o args= -p <pid>`. Write `/etc/cascade/<svc>.args` from
   it. Confirm it matches the newest `restart_<svc>_*.sh` (the PYTHONPATH and every flag).
2. `ln -sfn <the PYTHONPATH from step 1> /root/cascade-releases/<svc>.current`.
3. `cp deploy/systemd/cascade-<svc>.service /etc/systemd/system/ && systemctl daemon-reload`.
   For intake: the box has a stale, inactive `cascade-intake.service` (PYTHONPATH
   `/root/cascade-release`). Back it up first (`.bak-pre-p2`).
4. Validator and trainer only: run `cascade-ops restart-check <svc>` and require GO.
   - Do not do this inside a settlement or scoring window.
   - Do not do this with legs, generation VMs or harvests in flight.
5. Stop the nohup process with the old script's TERM path (`kill -TERM <pid>`; wait at
   most 30 s). Then `systemctl enable --now cascade-<svc>`.
   - Validator: stop `cascade-validator-watchdog.service` FIRST. Otherwise the
     watchdog starts a second validator. The unit's `Restart=always` replaces it.
6. Verify:
   - `systemctl status cascade-<svc>` shows active.
   - `systemctl show -p MemoryMax,MemoryHigh cascade-<svc>` shows the limits.
   - The log has `box guard <mode> for <svc>`.
   - Only one process is running: `pgrep -af '[c]ascade-<svc>'`.
   - Validator: a weight set lands within 180 blocks.
   - Trainer: legs re-attach.
7. OPSLOG entry. Rename the old restart scripts `*.retired`; do not delete them.
   `scripts/release_sprawl.py` (P4) lists them.

## Rollback

Run `systemctl disable --now cascade-<svc>`, then the last `restart_<svc>_*.sh`. The
args file and symlink stay in place and do nothing.

## When a limit trips

`journalctl -u cascade-<svc>` shows `oom-kill` / `memory.high`. The service log shows a
`box_budget_exceeded:` ERROR (`cascade/shared/rss_budget.py`) for the step that
grew. That step's work belongs on a pod. Do not raise `MemoryMax` to make it fit.
