# Champions: published generator code

Code submitted to cascade stays private until it wins. Winners are published
here by the scheduled `king-sync` workflow (`.github/workflows/king-sync.yml`),
which reads what the trainer's champion step (`cascade.funding.champion`,
`[round] champion_publish` policy) put in the public manifest bucket. Every copy
passes the static scan in `scripts/scan_generator.py` first and carries
`SCAN.json` and `PROVENANCE.json`. **These are static copies of untrusted
competitor code. Never run them outside a sandbox, whatever the scan says.**

| Path | What it holds |
|---|---|
| `king/` | the sitting lane A king (`scripts/sync_king.py`); `git log champions/king` is the reign timeline |
| `archive/` | every lane A reign, oldest first (`archive/README.md`) |
| `lane_b/<domain>/<slot>/` | the current holder of each lane B slot: the generator whose experts were merged into the king. `<domain>` is the target (`energy`, `horizon-720`, `suite-gift`), `<slot>` the expert rows it trained (`rows-<12 hex>`) |
| `lane_b/archive/<domain>/<slot>/<round>/` | earlier holders of a re-won slot (and holders published only when they were displaced) |
| `lane_b/index.json` | the published lane B index, verbatim (`scripts/sync_lane_b_champions.py`) |
| `lane_d/` | lane D king cards; the weights live on the Hub, R2 and HF (`scripts/laned_king_sync.py`) |

## Lane B index

`lane_b/index.json` ties the lane B generators to the checkpoints they changed,
which live in other repositories (Hippius Hub, the miner's own Hub repo):

- `slots`: per target, the holder's expert rows per layer (`rows`, `slot`),
  `hotkey`, `uid`, `gen_ref` and `generator_digest`, `zip_key` (vault code),
  `merged_into` (the king checkpoint the experts were merged into),
  `lane_a_king_pointer` (the lane A king checkpoint at the time),
  `rebased_pointer` / `rebased_onto` (the judged leg checkpoint and the king it
  was rebased onto), `round_id`, `merge_block`, `era`, and `published`;
- `composed`: every lane B generator in the lineage of `king_checkpoint`;
- `archive`: displaced holders whose code is public;
- `sha256`: sha256 of the canonical JSON body without this field.

To verify one generator: check `sha256` (`cascade.funding.champion.verify_lane_b_index`),
re-hash the code against `generator_digest` (a vault ZIP is content-addressed),
then match `merged_into` and the leg's rows against the era manifest's
`warm_start_merges` and the round receipts with `cascade-audit`.
