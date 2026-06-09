"""Layer 0 — fetch Fed rate-decision contract probabilities from Polymarket Gamma API."""

import json

import requests
import pandas as pd
from datetime import datetime

GAMMA_API = "https://gamma-api.polymarket.com"
KEYWORDS = ["fed", "fomc", "federal reserve", "rate cut"]


def fetch_fed_markets():
    """Paginate the Gamma /markets endpoint and return all Fed-related markets."""
    url = f"{GAMMA_API}/markets"
    found = []

    for page in range(4):
        offset = page * 100
        params = {"limit": 100, "offset": offset, "active": "true", "closed": "false"}

        print(f"  Fetching page {page + 1} (offset {offset})…")
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        markets = response.json()
        if not markets:
            break

        for market in markets:
            question = market.get("question", "").lower()
            slug = market.get("slug", "").lower()
            if any(k in question or k in slug for k in KEYWORDS):
                found.append(market)

    return found


def extract_probability(market):
    """Return the YES implied probability from outcomePrices, or None on parse error."""
    raw = market.get("outcomePrices", "[]")
    try:
        prices = json.loads(raw)
        return float(prices[0])  # index 0 is always YES on binary Polymarket contracts
    except (IndexError, ValueError, TypeError):
        return None


def main():
    print("Contacting Polymarket Gamma API…")
    fed_markets = fetch_fed_markets()
    print(f"Fed-related markets found: {len(fed_markets)}\n")

    if not fed_markets:
        print("No Fed rate markets found.")
        return

    # Prefer "rate cut" contracts; fall back to all Fed matches.
    rate_cut_markets = [m for m in fed_markets if "rate cut" in m.get("question", "").lower()]
    display_markets = rate_cut_markets or fed_markets

    print("=" * 60)
    print("FED RATE DECISION CONTRACTS ON POLYMARKET")
    print("=" * 60)

    rows = []
    for market in display_markets:
        question = market.get("question", "N/A")
        prob = extract_probability(market)
        volume = float(market.get("volume", 0))

        if prob is None:
            continue

        print(f"\nContract : {question}")
        print(f"  YES probability : {prob:.1%}")
        print(f"  Total volume    : ${volume:,.0f}")

        rows.append({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "yes_probability": prob,
            "volume_usd": volume,
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("layer0_polymarket.csv", index=False)
        print(f"\nSaved {len(rows)} rows to layer0_polymarket.csv")
    else:
        print("No data to save.")


if __name__ == "__main__":
    main()
