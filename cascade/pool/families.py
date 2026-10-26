"""Producer-side bootstrap families, recorded in immutable pool metadata.

Unknown dependence is handled conservatively by origin, not by counting every
feed or sensor as independent. Catalog authors may supply a curated family.
Historical pools keep their recorded source keys; validators need no inference.
"""
from __future__ import annotations

from urllib.parse import urlsplit


def catalog_families(catalog: list[dict]) -> dict[str, str]:
    parents: dict[str, str] = {}
    by_id: dict[str, str] = {}

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(left, right):
        a, b = sorted((root(left), root(right)))
        parents[b] = a

    for entry in catalog:
        sid = str(entry.get('id') or '')
        if not sid:
            continue
        if sid in by_id:
            raise ValueError(f'duplicate source id in family catalog: {sid}')
        explicit = entry.get('source_family')
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit.strip() or len(explicit) > 200:
                raise ValueError(f'{sid}: source_family must be a nonempty string <= 200 characters')
            family = 'curated:' + explicit.strip()
        else:
            dgp = str(entry.get('dgp_class') or '').lower()
            endpoint = entry.get('endpoint') or {}
            url = endpoint.get('url', '') if isinstance(endpoint, dict) else ''
            origin = urlsplit(str(url)).hostname
            # Same underlying market across venues/cadences is one family.
            crypto = ('crypto' in dgp or sid.startswith(('coinbase_', 'binance_', 'kraken_',
                      'bitfinex_', 'bitstamp_', 'bitvavo_', 'gemini_', 'okx_', 'bitso_',
                      'bybit_', 'kucoin_', 'gateio_', 'bitget_', 'poloniex_', 'deribit_')) or
                      any(token in sid.lower() for token in ('btc', 'bitcoin', 'ethusd', 'eth_usd')))
            if dgp == 'fx_rate':
                family = 'market:fx'
            elif crypto and dgp in ('market_price', 'market_orderbook', 'market_microstructure',
                                    'crypto_spot_price', 'crypto_price', 'market_volume'):
                family = 'market:crypto'
            elif sid.startswith('osem_'):
                family = 'network:opensensemap'
            elif sid.startswith('sensor_community_'):
                family = 'network:sensor-community'
            elif origin:
                family = 'origin:' + origin.lower()
            else:
                # No inferred independence: unresolved entries are marked and
                # rejected by the primary family's required-label build rule.
                family = 'unresolved:' + sid
        by_id[sid] = family
        root(family)
    for entry in catalog:
        sid = str(entry.get('id') or '')
        paired = entry.get('paired_with') or ()
        if isinstance(paired, str):
            paired = [paired]
        for other in paired:
            if sid in by_id and other in by_id:
                join(by_id[sid], by_id[other])
    return {sid: root(family) for sid, family in by_id.items()}
