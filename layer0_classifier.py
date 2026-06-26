"""
CrowdArb Phase G1 — contract classifier for real-time market auto-discovery.

classify_market() is a pure function (no network I/O); it inspects a raw Gamma
API market dict and returns a string tag indicating what kind of market it is
and whether a professional-market counterpart can be computed.

Tag priority (first match wins):
  crypto_level   — crypto asset price-level contract (BTC/ETH/…, dollar strike, reach verb)
  rate_decision  — central-bank rate decision contract
  equity_level   — equity index price-level contract (reach verb, numeric level)
  unsupported    — everything else

Design notes:
  - Short keywords (btc, eth, dow, fed, …) use \\b word boundaries to avoid
    substring collisions (e.g. "downgrade" contains "dow", "alfred" contains "fed").
  - Long/distinctive phrases (fomc, federal reserve, bitcoin) use substring match.
  - equity_level requires a reach verb to exclude "s&p 500 company buys bitcoin"
    and "s&p rating downgrade" false positives.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date

# ── Keyword patterns ──────────────────────────────────────────────────────────

# All crypto tickers (broad; used for informational scans)
_CRYPTO_RE = re.compile(
    r"\b(btc|eth|sol|xrp|doge|avax|bnb|ada|dot|link|ltc|atom|matic)\b"
    r"|bitcoin|ethereum|solana|ripple|dogecoin|avalanche|polygon|cardano"
    r"|polkadot|chainlink|litecoin|cosmos",
    re.IGNORECASE,
)

# Priceable crypto: BTC and ETH only (have Deribit DVOL implied vol — Path A)
_PRICEABLE_CRYPTO_RE = re.compile(
    r"\b(btc|eth)\b|bitcoin|ethereum",
    re.IGNORECASE,
)

# Reach verbs: indicate a price-level (directional) contract
_REACH_RE = re.compile(
    r"\b(hit|reach|above|exceed|surpass|break|cross)\b",
    re.IGNORECASE,
)

# Dollar strike: "$150k", "$150,000", "$1m", "$1.5b" — quick boolean check only
_DOLLAR_RE = re.compile(r"\$[\d,]+(?:\.\d+)?\s*[kmb]?", re.IGNORECASE)

# Strike parser: requires multiplier suffix not to continue into a word (fixes the
# "$100,000 by December" bug where "b" from "by" would be consumed as a billion)
_STRIKE_RE = re.compile(r"\$([\d,]+(?:\.\d+)?)\s*([kmb](?!\w))?", re.IGNORECASE)

_MULTIPLIERS: dict[str, float] = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

# Rate-decision keywords; "fed/ecb/boe/boj" get \b to avoid "alfred", "downgrade", etc.
_RATE_RE = re.compile(
    r"\b(fed|ecb|boe|boj)\b"
    r"|fomc|federal reserve|rate cut|rate hike|rate increase|rate decrease"
    r"|basis points|\bbps\b|interest rate",
    re.IGNORECASE,
)

# Equity index keywords; "dow/dax/russell" get \b to avoid "downgrade", "F1 driver Russell", etc.
_EQUITY_RE = re.compile(
    r"\b(dow|dax|russell)\b"
    r"|s&p|spx|\bsp500\b|nasdaq|\bndx\b|nikkei|ftse",
    re.IGNORECASE,
)

# Numeric price level (3–6 digits); used as fallback when no "$" prefix is present
_LEVEL_RE = re.compile(r"\b\d{3,6}\b")

# ── Public interface ──────────────────────────────────────────────────────────

SUPPORTED_TAGS: frozenset[str] = frozenset({"crypto_level", "rate_decision", "equity_level"})


def classify_market(market: dict) -> str:
    """
    Classify a raw Gamma API market dict.  Pure function — no network I/O.

    Parameters
    ----------
    market : raw dict from /markets endpoint

    Returns
    -------
    One of: "crypto_level", "rate_decision", "equity_level", "unsupported"
    """
    q = (market.get("question") or "").lower()

    has_crypto  = bool(_PRICEABLE_CRYPTO_RE.search(q))  # BTC/ETH only (Path A)
    has_reach   = bool(_REACH_RE.search(q))
    has_dollar  = bool(_DOLLAR_RE.search(q))
    has_rate    = bool(_RATE_RE.search(q))
    has_equity  = bool(_EQUITY_RE.search(q))
    has_level   = has_dollar or bool(_LEVEL_RE.search(q))

    if has_crypto and has_reach and has_dollar:
        return "crypto_level"

    if has_rate:
        return "rate_decision"

    # equity_level requires a reach verb to exclude "s&p company buys X" type questions
    if has_equity and has_reach and has_level:
        return "equity_level"

    return "unsupported"


def parse_strike(question: str) -> float | None:
    """Extract dollar strike from a contract question. Returns None if not found.

    Handles $150k, $150K, $150,000, $1m, $1M, $3,500, $1.5m.
    Safe against '$100,000 by December' (does not eat the 'b' from 'by').
    """
    m = _STRIKE_RE.search(question)
    if not m:
        return None
    val = float(m.group(1).replace(",", ""))
    mult = _MULTIPLIERS.get((m.group(2) or "").lower(), 1)
    return val * mult


def parse_end_date(market: dict) -> date | None:
    """Parse resolution date from a Gamma API market dict. Returns None on failure."""
    raw = (market.get("endDate") or market.get("endDateIso") or "")[:10]
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


# ── Scanner / main ────────────────────────────────────────────────────────────

def scan_catalogue(max_pages: int = 30) -> dict[str, list[dict]]:
    """
    Paginate the live Polymarket catalogue, classify every market, and return
    a dict mapping each tag to the list of matching market dicts.
    """
    from layer0_markets import _iter_active_markets

    by_tag: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for market in _iter_active_markets(max_pages=max_pages):
        tag = classify_market(market)
        by_tag[tag].append(market)
        total += 1
        if total % 500 == 0:
            print(f"  … {total} markets classified")

    return dict(by_tag), total


def _top_by_volume(markets: list[dict], n: int = 5) -> list[dict]:
    return sorted(markets, key=lambda m: float(m.get("volume") or 0), reverse=True)[:n]


def main() -> None:
    print("CrowdArb Phase G1 — scanning live Polymarket catalogue")
    print("─" * 58)
    print()

    by_tag, total = scan_catalogue(max_pages=30)

    # ── Tag distribution ──────────────────────────────────────────────────────
    counts = Counter({tag: len(markets) for tag, markets in by_tag.items()})
    W = 58

    print()
    print("=" * W)
    print("  TAG DISTRIBUTION")
    print("=" * W)
    print(f"  {'Tag':<22}  {'Count':>6}  {'Share':>7}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*7}")
    for tag, count in counts.most_common():
        flag = " ✓" if tag in SUPPORTED_TAGS else ""
        print(f"  {tag:<22}  {count:>6}  {count/total*100:>6.1f}%{flag}")
    print(f"  {'─'*22}  {'─'*6}")
    print(f"  {'TOTAL':<22}  {total:>6}")
    print("=" * W)

    # ── Top contracts per supported tag ──────────────────────────────────────
    print()
    for tag in ("crypto_level", "rate_decision", "equity_level"):
        markets = by_tag.get(tag, [])
        if not markets:
            print(f"  {tag.upper()}: (none found)")
            continue

        print(f"  TOP {tag.upper()} (by volume):")
        col_q = 52
        print(f"    {'Question':<{col_q}}  {'P(YES)':>6}  {'Volume':>12}")
        print("    " + "─" * (col_q + 22))
        for m in _top_by_volume(markets, n=8):
            q    = m.get("question", "")
            disp = q if len(q) <= col_q else q[:col_q - 3] + "..."
            try:
                import json
                prices = json.loads(m.get("outcomePrices") or "[]")
                p = f"{float(prices[0]):.1%}"
            except Exception:
                p = "—"
            vol = float(m.get("volume") or 0)
            print(f"    {disp:<{col_q}}  {p:>6}  ${vol:>11,.0f}")
        print()

    # ── False-positive check ──────────────────────────────────────────────────
    print("  FALSE-POSITIVE AUDIT")
    print(f"  {'─'*55}")
    for tag in SUPPORTED_TAGS:
        markets = by_tag.get(tag, [])
        suspicious = [
            m for m in markets
            if not any(v in m.get("question","").lower()
                       for v in ["hit", "reach", "above"])
               and tag in ("crypto_level", "equity_level")
        ]
        if suspicious:
            print(f"  {tag}: {len(suspicious)} potentially misclassified:")
            for m in suspicious[:3]:
                print(f"    → {m.get('question','')[:80]}")
        else:
            print(f"  {tag}: no obvious false positives")
    print()


if __name__ == "__main__":
    main()
