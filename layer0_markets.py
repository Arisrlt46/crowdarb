"""
CrowdArb Layer 0 — General-purpose Polymarket market discovery.

API findings (from empirical investigation)
--------------------------------------------
- /events/keyset  : cursor-based, walks ALL events oldest-first (ID ~2890 = Dec 2021).
                    Useless for finding current markets without traversing tens of
                    thousands of resolved events.
- /markets        : offset-based, honours active=true&closed=false, caps at ~2900.
                    Returns only currently open markets. This is the right endpoint.
- category field  : present on old (2021-2022) events; null on most current markets.
- tags field      : every event has only {"slug": "all"}. Not useful for filtering.

Strategy
--------
- Page /markets?active=true&closed=false with offset pagination (the same approach
  that successfully finds Fed contracts in layer0_data.py).
- Filter client-side on question text for keywords (OR logic).
- Optionally narrow by market.category for the minority that have one set.
- Rank results by volume descending.
"""

from __future__ import annotations

import json
import time
from typing import Iterator

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
DEFAULT_TIMEOUT = 15    # seconds per request
DEFAULT_MAX_PAGES = 30  # 30 × 100 = 3 000 markets; covers all active open contracts


# ── Internal helpers ──────────────────────────────────────────────────────────

def _iter_active_markets(max_pages: int = DEFAULT_MAX_PAGES) -> Iterator[dict]:
    """
    Yield every currently open market from /markets?active=true&closed=false.
    Stops at the API's offset cap (~2900) or when a page returns empty.
    """
    for page in range(max_pages):
        params = {
            "limit": 100,
            "offset": page * 100,
            "active": "true",
            "closed": "false",
        }
        resp = requests.get(f"{GAMMA_BASE}/markets", params=params, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 422:
            break   # API offset cap reached (~2100 with active+closed filters)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            break
        yield from data


def _extract_yes_prob(market: dict) -> float | None:
    """Parse the YES implied probability from outcomePrices (index 0 = YES)."""
    raw = market.get("outcomePrices") or "[]"
    try:
        prices = json.loads(raw)
        return float(prices[0])
    except (IndexError, ValueError, TypeError):
        return None


def _match_keyword(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


# ── Public API ────────────────────────────────────────────────────────────────

def find_markets(
    keyword: str | list[str],
    *,
    category: str | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> list[dict]:
    """
    Return all currently open Polymarket markets whose question contains `keyword`.

    Parameters
    ----------
    keyword   : str or list[str] — any match triggers inclusion (OR logic).
    category  : optional client-side filter on market.category (case-insensitive).
                Useful for old markets that have a category set; pass None to skip.
    max_pages : stop after this many pages of 100 markets each (default 30 = 3 000).

    Each returned dict is the raw market object augmented with:
      _yes_prob : float | None   — parsed YES probability
    Sorted by volume descending.
    """
    keywords = [keyword.lower()] if isinstance(keyword, str) else [k.lower() for k in keyword]
    cat_filter = category.lower() if category else None

    results: list[dict] = []

    for market in _iter_active_markets(max_pages=max_pages):
        if cat_filter:
            mcat = (market.get("category") or "").lower()
            if cat_filter not in mcat:
                continue

        question = market.get("question") or ""
        if not _match_keyword(question, keywords):
            continue

        enriched = dict(market)
        enriched["_yes_prob"] = _extract_yes_prob(market)
        results.append(enriched)

    results.sort(key=lambda m: float(m.get("volume") or 0), reverse=True)
    return results


def print_markets(markets: list[dict], label: str = "Markets found") -> None:
    """Pretty-print a list of markets returned by find_markets."""
    print(f"\n{'=' * 70}")
    print(f"  {label}  ({len(markets)} result{'s' if len(markets) != 1 else ''})")
    print(f"{'=' * 70}")
    if not markets:
        print("  (none)")
        return

    for m in markets:
        q    = m.get("question", "N/A")
        p    = m["_yes_prob"]
        vol  = float(m.get("volume") or 0)
        end  = (m.get("endDate") or "")[:10]
        cat  = m.get("category") or "(no category)"
        slug = m.get("slug") or ""

        print(f"\n  Q      : {q}")
        print(f"  slug   : {slug}")
        print(f"  cat    : {cat}")
        print(f"  P(YES) : {f'{p:.1%}' if p is not None else '—':<8}  "
              f"Vol: ${vol:>12,.0f}  End: {end}")
    print(f"\n{'=' * 70}")


# ── Main: demonstration search ────────────────────────────────────────────────

def main() -> None:
    searches = [
        # (label, keyword, category)
        ("S&P 500 / SPX level contracts",     ["s&p", "spx", "s&p 500", "sp500"],  None),
        ("Equity index (broader)",            ["index", "indices", "nasdaq", "dow"], None),
        ("Fed / FOMC rate contracts",         ["fed", "fomc", "rate cut", "federal reserve"], None),
        ("Bitcoin price level contracts",     ["bitcoin", "btc"],                   None),
    ]

    for label, kw, cat in searches:
        print(f"\nSearching: {label!r}  keywords={kw}  category={cat!r}")
        markets = find_markets(kw, category=cat, max_pages=30)
        print_markets(markets, label=label)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
