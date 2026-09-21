# Miner quickstart

The five commands. Full guide: [MINER.md](MINER.md). Contract: [INTERFACE.md](INTERFACE.md).

```bash
pip install -e '.[hippius,chain]'
cascade verify ./my-generator
btcli subnets register --netuid 91 --network finney --wallet-name w --wallet-hotkey h

export LIUM_API_KEY=sk-...                     # env only; pays for your own training leg
cascade submit ./my-generator https://submissions.cascadesub.net \
    --wallet-name w --wallet-hotkey h --label my-gen-v1
cascade queue --intake https://submissions.cascadesub.net --hotkey <your ss58>
```

What you need to know:

- **Rounds every 12 h** (≈ 08:30 / 20:30 UTC). Only a commitment revealed before
  the boundary enters; `submit` times the reveal for you.
- **You pay for your leg.** Keep ~4 h of the round's GPU price on your Lium
  account (~5 h training + ~1 h benching). GPU type is the most available of
  RTX4090 / RTX3090 / L40S / L40 / A6000, same for everyone in the round.
- **Seats go by reveal order**, up to the round's cap. Unseated entries wait
  with nothing spent.
- **One submission per hotkey**, spent only when judged or when your own code
  fails. Infrastructure failures re-queue you.
- **Private by default.** `cascade submit` keeps your code in the operator's
  vault; it is published only if it takes the throne. Prefer it over the public
  `cascade deploy` + `cascade fund` path unless you want your code public.
- **Multivariate is allowed:** yield `(C, L)` arrays, C ≤ 32. Channels cost
  what they train and are scored jointly from block 9068400.

Watch: `cascade round` (deadline, live dethrone bar), `cascade duel` (the
verdict), `cascade duel --hotkey <you>` (which domains you beat the king in
and by how much; add `--history` for the trend). Failure classes and what
they mean for your entry: MINER.md §8.
