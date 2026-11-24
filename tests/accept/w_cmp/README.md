# W-CMP acceptance tests: declarative lane C competitions, standard pool, rule menu

W-CMP (plan rev6.25, DEC-CA-0075) depends on W-OV. Owner 2026-11-24: "new
test or formula and embargo days etc. should be up to our choice, and we
should bring new lane C competitions without validators updating."
Validators never run owner code: the owner picks among registry blocks by
signed overlay.

| File | Pins |
|---|---|
| `test_w_cmp_std_pool.py` | the standard pool writes / reads / refuses tampered, unlisted, unsigned and wrong-schema pools; the MarketCast snapshot re-emitted as a standard pool gives the judge byte-identical bars, meta and windows; the spec's schema and series `select` reach the reader |
| `test_w_cmp_spec.py` | the registry (runners, metrics, scores, guards, schemas, menu) is published by `overlay_policy()`; `marketcast_spec` restates both shipped configs byte-identically (MarketCast config, competitions, targets, split); spec values reach `[marketcast]`; unknown blocks / keys / runners and out-of-bounds values are typed refusals; a refused MarketCast spec is inert and burns; a spec for a new id burns its own share only; an overlay sets a spec, refuses out-of-bounds values and adopts unknown names as a refusal |
| `test_w_cmp_menu.py` | lane defaults are today's rules (lane B: rev6.22 max-T families, paid until replaced); menu choices parse, unsupported ones are refused per lane; family math; pay curves; a lane C win records a non-default curve, pays on it and keeps legacy state bytes |
| `test_w_cmp_embargo_hashset.py` | the draw uses the embargo recorded at dispatch (a later config change does not move it; legacy entries use the config); `embargo_days` is overlay-settable within bounds; the trainer stamps it; the hash set is fetched by key, verified against the pin, cached, refused on a mismatch; the storage key parses and the pin is overlay-settable |

Every test imports a W-CMP module (`std_pool`, `competition_spec`,
`rules_menu`) or a W-CMP field in its body, so on a base without the WP it
fails inside `PATHS`. Local run: `-m 'not torch'`, `--noconftest`.
