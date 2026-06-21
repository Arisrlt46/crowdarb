"""CrowdArb Layer 0 — S&P 500 Polymarket crowd vs Black-Scholes implied probability."""

import os
import re
from datetime import date, datetime, timezone

import pandas as pd
import requests

from layer0_bs import implied_probability_above, years_to
from layer0_data import GAMMA_API, extract_probability
from layer0_sp500 import (
    DIVIDEND_YIELD,
    STRIKE,
    TARGET_DATE,
    fetch_sp500,
    fetch_tnx,
    fetch_vix,
    historical_vol_spy,
)

SP500_KEYWORDS = ["s&p 500", "s&p", "spx", "sp500", "6500", "6,500"]
HISTORY_FILE = "layer0_sp500_history.csv"
W = 66


def fetch_sp500_markets() -> list[dict]:
    """Paginate the Gamma /markets endpoint and return all S&P 500 level contracts."""
    url = f"{GAMMA_API}/markets"
    found = []
    for page in range(4):
        offset = page * 100
        params = {"limit": 100, "offset": offset, "active": "true", "closed": "false"}
        print(f"  Fetching page {page + 1} (offset {offset})...")
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        markets = resp.json()
        if not markets:
            break
        for m in markets:
            question = m.get("question", "").lower()
            slug = m.get("slug", "").lower()
            if any(k in question or k in slug for k in SP500_KEYWORDS):
                found.append(m)
    return found


def _parse_strike(question: str) -> float | None:
    """Extract the first plausible S&P 500 level (3000–9000) from a contract question."""
    for token in re.findall(r"[\d,]+", question):
        val = float(token.replace(",", ""))
        if 3000.0 <= val <= 9000.0:
            return val
    return None


def pick_best_match(markets: list[dict], target_strike: float) -> dict | None:
    """
    Return the contract whose parsed strike is closest to target_strike.
    Ties broken by descending volume. Falls back to highest-volume contract
    when no strike can be parsed from any question text.
    """
    if not markets:
        return None
    scored = []
    for m in markets:
        k = _parse_strike(m.get("question", ""))
        distance = abs(k - target_strike) if k is not None else float("inf")
        scored.append((distance, -float(m.get("volume", 0)), m))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]


def log_history(
    ts: datetime,
    poly_prob: float | None,
    bs_prob: float,
    poly_question: str,
    S: float,
    r: float,
    sigma: float,
) -> None:
    """Append one row to layer0_sp500_history.csv, creating headers on first write."""
    row = {
        "timestamp": ts.isoformat(),
        "polymarket_probability": poly_prob,
        "bs_probability": bs_prob,
        "polymarket_question": poly_question,
        "spot": S,
        "strike": STRIKE,
        "r": r,
        "sigma": sigma,
    }
    write_header = not os.path.exists(HISTORY_FILE)
    pd.DataFrame([row]).to_csv(HISTORY_FILE, mode="a", index=False, header=write_header)


def main() -> None:
    today = date.today()

    # ── Search Polymarket for S&P 500 contracts ───────────────────────────────
    print(f"\nSearching Polymarket Gamma API for S&P 500 contracts...")
    sp_markets = fetch_sp500_markets()

    print(f"\n  {len(sp_markets)} contract(s) found matching S&P / SPX / 6500\n")

    if sp_markets:
        col_q, col_p, col_v = 44, 8, 10
        print("  " + "─" * (col_q + col_p + col_v + 4))
        print(f"  {'Question':<{col_q}}  {'P(YES)':>{col_p}}  {'Volume':>{col_v}}")
        print("  " + "─" * (col_q + col_p + col_v + 4))
        for m in sorted(sp_markets, key=lambda x: -float(x.get("volume", 0))):
            q = m.get("question", "N/A")
            q_display = q if len(q) <= col_q else q[: col_q - 3] + "..."
            prob = extract_probability(m)
            vol = float(m.get("volume", 0))
            prob_str = f"{prob:.1%}" if prob is not None else "  —"
            print(f"  {q_display:<{col_q}}  {prob_str:>{col_p}}  ${vol:>{col_v - 1},.0f}")
        print("  " + "─" * (col_q + col_p + col_v + 4))

    # ── Select the closest match to K=6500 ────────────────────────────────────
    best = pick_best_match(sp_markets, STRIKE)
    poly_prob = extract_probability(best) if best else None
    poly_question = best.get("question", "none") if best else "none"
    parsed_k = _parse_strike(poly_question) if best else None

    if best:
        match_note = (
            f"strike parsed as {parsed_k:,.0f}" if parsed_k is not None else "no strike parsed — using highest-volume"
        )
        print(f"\n  Best match : {poly_question}")
        print(f"  ({match_note})")

    # ── Black-Scholes implied probability ─────────────────────────────────────
    print(f"\nFetching market data for Black-Scholes...")
    S = fetch_sp500()
    r = fetch_tnx()

    sigma_source = "^VIX"
    try:
        sigma = fetch_vix()
    except Exception as exc:
        print(f"  VIX unavailable ({exc}) — falling back to SPY 252-day historical vol")
        sigma = historical_vol_spy()
        sigma_source = "SPY historical (252d)"

    T = years_to(TARGET_DATE)
    bs_prob = implied_probability_above(S, STRIKE, T, r, sigma, q=DIVIDEND_YIELD)

    # ── Side-by-side table ────────────────────────────────────────────────────
    gap = (poly_prob - bs_prob) if poly_prob is not None else None
    lean = ""
    if gap is not None:
        lean = "Polymarket more bullish" if gap > 0 else "Black-Scholes more bullish"

    print()
    print("=" * W)
    print("  LAYER 0 — S&P 500 CROSS-MARKET PROBABILITY COMPARISON")
    print(f"  {today}  |  S&P 500 > {STRIKE:,.0f} by {TARGET_DATE}")
    print("=" * W)
    print(f"  {'Source':<20}  {'Detail':<28}  {'P(above)':>8}")
    print("-" * W)

    if poly_prob is not None:
        q_short = poly_question[:28] if len(poly_question) <= 28 else poly_question[:25] + "..."
        print(f"  {'Polymarket':<20}  {q_short:<28}  {poly_prob:.1%}")
    else:
        print(f"  {'Polymarket':<20}  {'no matching contract':<28}  {'—':>8}")

    bs_detail = f"S={S:,.0f}  σ={sigma:.3f}  r={r:.3f}"
    print(f"  {'Black-Scholes':<20}  {bs_detail:<28}  {bs_prob:.1%}")
    print("=" * W)
    print()
    print(f"  Inputs  :  S={S:,.2f}  K={STRIKE:,.0f}  T={T:.3f} yr  r={r:.4f}  σ={sigma:.4f}  q={DIVIDEND_YIELD:.4f}")
    print(f"  Vol     :  {sigma_source}")
    if gap is not None:
        print(f"  Gap (Poly − BS) :  {gap:+.1%}  ({lean})")
    print()
    print("  Note: Polymarket = crowd odds on a binary outcome contract.")
    print("        Black-Scholes = risk-neutral probability under log-normal price dynamics.")
    print("        A positive gap means the crowd is more bullish than options pricing implies.")
    print("=" * W)

    # ── Append to history ─────────────────────────────────────────────────────
    log_history(
        ts=datetime.now(timezone.utc),
        poly_prob=poly_prob,
        bs_prob=bs_prob,
        poly_question=poly_question,
        S=S,
        r=r,
        sigma=sigma,
    )
    print(f"\n  Row appended to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
