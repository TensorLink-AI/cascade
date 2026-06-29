"""Wikimedia pageviews source — keyless, free, daily web-traffic breadth.

Uses the Wikimedia REST pageviews API
(https://wikimedia.org/api/rest_v1/#/Pageviews%20data), no key required. Each
article becomes one daily series with weekly seasonality — a distinct domain
(human attention / web traffic) from the physical weather backbone, which is
what makes the pool broad enough that a generator can't win by matching one
distribution.

Caveat by design: per-article pageviews are *daily*, so these series are far
shorter than ``context_length = 4096`` and contribute shorter-context windows
(the builder's ``min_context`` floor admits them; the hourly sources fill the
full context). Future pageviews are unknowable, so freshness still applies.
"""

from __future__ import annotations

import datetime as dt
import urllib.parse
from collections.abc import Iterable

import numpy as np

from ..source import FetchJson, HarvestContext, HarvestedSeries

# Per-article daily endpoint. Daily data needs years of history to be useful, so
# this source pulls a fixed multi-year window regardless of ctx.span_days.
REST_BASE = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
PROJECT = "en.wikipedia"
HISTORY_DAYS = 1500  # ~4 years of daily points

# Stable, high-traffic articles across topics (countries, science, companies,
# culture) so the set is reliably populated and topically diverse.
ARTICLES: tuple[str, ...] = (
    "United_States", "India", "China", "Germany", "Japan", "Brazil", "Russia",
    "France", "United_Kingdom", "Canada", "Australia", "Italy", "Spain", "Mexico",
    "Indonesia", "Nigeria", "Egypt", "South_Africa", "Turkey", "Argentina",
    "Climate_change", "Artificial_intelligence", "World_War_II", "World_War_I",
    "Mathematics", "Physics", "Chemistry", "Biology", "Economics", "Philosophy",
    "Photosynthesis", "DNA", "Black_hole", "Evolution", "Quantum_mechanics",
    "Solar_System", "Periodic_table", "Human_brain", "Vaccine", "Electricity",
    "Apple_Inc.", "Google", "Microsoft", "Amazon_(company)", "Tesla,_Inc.",
    "Meta_Platforms", "Netflix", "Nvidia", "Samsung", "Toyota",
    "Bitcoin", "Ethereum", "Stock_market", "Inflation", "Cryptocurrency",
    "Association_football", "Olympic_Games", "Basketball", "Chess", "Cricket",
    "Tennis", "Formula_One", "FIFA_World_Cup", "National_Basketball_Association",
    "The_Beatles", "Taylor_Swift", "Michael_Jackson", "Beyoncé", "Elvis_Presley",
    "William_Shakespeare", "Albert_Einstein", "Isaac_Newton", "Leonardo_da_Vinci",
    "Charles_Darwin", "Nikola_Tesla", "Marie_Curie", "Stephen_Hawking",
    "Pyramids_of_Giza", "Mount_Everest", "Pacific_Ocean", "Amazon_rainforest",
    "Sahara", "Great_Barrier_Reef", "Niagara_Falls", "Grand_Canyon",
)


class WikimediaSource:
    name = "wikimedia"

    def __init__(self, articles: tuple[str, ...] = ARTICLES, project: str = PROJECT) -> None:
        self.articles = articles
        self.project = project

    def harvest(self, fetch: FetchJson, ctx: HarvestContext) -> Iterable[HarvestedSeries]:
        end = ctx.as_of
        start = end - dt.timedelta(days=HISTORY_DAYS)
        start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
        emitted = 0
        for article in self.articles:
            if emitted >= ctx.max_series:
                return
            art = urllib.parse.quote(article, safe="")
            url = (
                f"{REST_BASE}/{self.project}/all-access/all-agents/"
                f"{art}/daily/{start_s}/{end_s}"
            )
            data = fetch(url, None)
            items = (data or {}).get("items") or []
            if len(items) < 2:
                continue
            values = np.array([it.get("views", np.nan) for it in items], dtype=np.float64)
            yield HarvestedSeries(
                series_id=f"wikimedia__{article}",
                values=values,
                freq="D",
                domain="web_traffic",
                seasonal_period=7,
                attrs={"article": article, "project": self.project},
            )
            emitted += 1
