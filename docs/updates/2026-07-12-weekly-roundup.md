# Cascade Weekly Roundup — July 12–19, 2026

The short version: **this was the week we committed to mainnet.** On Monday
we decided Cascade's permanent home on Bittensor mainnet (subnet 91), wrote
down the launch checklist, and then spent the rest of the week making sure
every piece of the system is solid enough to run there for real. 119 commits
and about 37 pull requests landed between Monday and Friday.

## Mainnet is locked in

Monday was decision day. The subnet number (91) is now baked into the code so
operators can't misconfigure it, the launch checklist lives in
`docs/MAINNET_LAUNCH.md`, and the key launch settings are pinned down: which
training image is allowed to run, which GPU it must run on, and a "go-live
block" — a point on the blockchain before which nothing counts, so everyone
starts on equal footing.

We also set up an independent, tamper-proof archive of audit records in a
storage account that the main system can't touch. If anyone ever questions a
result, there's a copy nobody could have edited.

## Making the infrastructure boring (in a good way)

The bulk of the week went into hardening the provisioner — the part of the
system that rents GPU machines, sets them up, and replaces them when they
fail. Running against real cloud providers surfaced a lot of rough edges,
and we fixed them one by one:

- Machines that take a long time to accept logins no longer get given up on
  too early — but machines that are clearly dead get abandoned fast.
- When a rented machine fails, its replacement is now guaranteed to be a
  *different* machine, not the same broken one again.
- Our logs kept going silent because a library we depend on quietly disables
  them. Logging now heals itself, so we never fly blind.
- Several hangs and stalls (waiting on the blockchain, shutting down
  connections) now have hard time limits instead of freezing forever.

## Fairness and correctness fixes

A handful of subtle bugs that could have affected scoring were found and
fixed: duplicate submissions are now detected by their actual content rather
than their label, a finished score can never be accidentally overwritten by
a later rejected one, and a stale instruction can no longer tear down
machines that are mid-evaluation. Miners whose submissions can't be
downloaded are now correctly marked as their fault, not ours.

We also introduced the "genesis baseline king": until a miner genuinely
beats the built-in baseline model, rewards are burned rather than paid out.
Nobody earns for showing up — only for winning.

## Raising the bar for miners

The training budget miners compete under went up 20x. This is deliberate:
the target is calibrated to a well-built training pipeline, and pipelines
that can't keep up will miss the deadline. That pressure is the point — it
rewards miners who do the engineering work.

## Transparency and tooling

- Evaluation results are now published daily to a public dataset (with a
  24-hour delay), so anyone can verify what happened.
- The repo got its first automated test gate: every change now has to pass
  the test suite and a code linter before it can merge.
- Training runs now report to a public experiment-tracking dashboard
  (Weights & Biases).
- Our standing design decisions moved into a structured decision log
  (`decisions/`) that connects to the company-wide strategy graph.

## What's next

Working through the remaining items on the mainnet launch checklist:
finalizing the worker image pin, the container sandbox, and moving the gift
gate from shadow mode to enforcement.
