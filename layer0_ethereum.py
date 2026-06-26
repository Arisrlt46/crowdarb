"""CrowdArb Layer 0 — Ethereum Polymarket vs Black-Scholes implied probability."""

from __future__ import annotations

import math
import os
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from layer0_bs import implied_probability_above, years_to
from layer0_classifier import parse_end_date, parse_strike
from layer0_markets import find_markets

HISTORY_FILE = "layer0_eth_history.csv"
W = 68
MIN_DAYS = 90

_UPSIDE   = ["hit", "reach", "above", "exceed", "surpass"]
_DOWNSIDE = ["dip", "below", "drop", "fall", "crash"]


def fetch_eth_price() -> float:
    """Current Ethereum price (USD) from yfinance."""
    hist = yf.Ticker("ETH-USD").history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch ETH-USD from yfinance.")
    return float(hist["Close"].iloc[-1])


def fetch_tnx() -> float:
    """10-year Treasury yield as a decimal risk-free rate proxy."""
    hist = yf.Ticker("^TNX").history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch ^TNX from yfinance.")
    return float(hist["Close"].iloc[-1]) / 100.0


def fetch_deribit_eth_dvol() -> float:
    """ETH 30-day implied vol from Deribit's DVOL index (returned as decimal)."""
    now_ms = int(time.time() * 1000)
    resp = requests.get(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        params={
            "currency": "ETH",
            "start_timestamp": now_ms - 3_600_000,
            "end_timestamp": now_ms,
            "resolution": "3600",
        },
        timeout=10,
    )
    resp.raise_for_status()
    candles = resp.json()["result"]["data"]
    if not candles:
        raise RuntimeError("Deribit ETH DVOL returned no candles.")
    return float(candles[-1][4]) / 100.0   # close column; DVOL is in %, convert to decimal


def historical_vol_eth(lookback_days: int = 90) -> float:
    """Annualised ETH vol from daily log-returns (365-day basis; crypto trades 24/7)."""
    closes = yf.Ticker("ETH-USD").history(period="1y")["Close"].dropna()
    log_rets = np.log(closes / closes.shift(1)).dropna()
    return float(log_rets.tail(lookback_days).std()) * math.sqrt(365)


def _is_reach_contract(market: dict) -> bool:
    """True for upside contracts ('hit/reach'), False for downside ('dip/drop')."""
    q = market.get("question", "").lower()
    return any(w in q for w in _UPSIDE) and not any(w in q for w in _DOWNSIDE)



def log_history(
    ts: datetime,
    poly_prob: float,
    bs_prob: float,
    question: str,
    S: float,
    K: float,
    r: float,
    sigma: float,
) -> None:
    """Append one row to layer0_eth_history.csv, creating headers on first write."""
    row = {
        "timestamp": ts.isoformat(),
        "polymarket_probability": poly_prob,
        "bs_probability": bs_prob,
        "polymarket_question": question,
        "spot": S,
        "strike": K,
        "r": r,
        "sigma": sigma,
    }
    write_header = not os.path.exists(HISTORY_FILE)
    pd.DataFrame([row]).to_csv(HISTORY_FILE, mode="a", index=False, header=write_header)


def main() -> None:
    today = date.today()

    print("Searching Polymarket for Ethereum price contracts...")
    all_eth = find_markets(["ethereum", "eth"])
    reach   = [m for m in all_eth if _is_reach_contract(m)]

    if not reach:
        print("No upside Ethereum contracts found.")
        return

    print(f"\n  {len(reach)} 'reach/hit' contract(s) found (sorted by volume):\n")
    col_q = 54
    print(f"  {'Question':<{col_q}}  {'P(YES)':>7}  {'Volume':>12}")
    print("  " + "─" * (col_q + 23))
    for m in reach:
        q_text = m.get("question", "")
        q_disp = q_text if len(q_text) <= col_q else q_text[:col_q - 3] + "..."
        p      = m["_yes_prob"]
        vol    = float(m.get("volume") or 0)
        print(f"  {q_disp:<{col_q}}  {f'{p:.1%}' if p is not None else '—':>7}  ${vol:>11,.0f}")
    print()

    # ── Select highest-volume contract with ≥90 days remaining ───────────────
    eligible = [
        m for m in reach
        if (d := parse_end_date(m)) is not None and years_to(d) * 365 >= MIN_DAYS
    ]
    if not eligible:
        print(f"  No contracts with ≥{MIN_DAYS} days remaining. Exiting.")
        return
    best      = eligible[0]
    poly_prob = best["_yes_prob"] or 0.0
    question  = best.get("question", "")

    K = parse_strike(question)
    if K is None:
        print(f"  Cannot parse strike from: {question!r}. Exiting.")
        return
    T_date = parse_end_date(best)
    if T_date is None:
        print(f"  Cannot parse end date for selected contract. Exiting.")
        return
    T      = years_to(T_date)

    print(f"  Selected : {question}")
    print(f"  Strike K : ${K:,.0f}")
    print(f"  Expiry   : {T_date}  (T = {T:.3f} yr)\n")

    # ── Fetch market data ─────────────────────────────────────────────────────
    print("Fetching market data...")
    S = fetch_eth_price()
    print(f"  S  (ETH-USD) : ${S:>12,.2f}")

    r = fetch_tnx()
    print(f"  r  (^TNX)    : {r:.4f}  ({r * 100:.2f}%)")

    sigma_source = "Deribit ETH DVOL"
    try:
        sigma = fetch_deribit_eth_dvol()
        print(f"  σ  (Deribit ETH DVOL) : {sigma:.4f}  ({sigma * 100:.1f}%)")
    except Exception as exc:
        print(f"  Deribit ETH DVOL unavailable ({exc})")
        print("  Falling back to ETH 90-day historical vol...")
        sigma = historical_vol_eth(lookback_days=90)
        sigma_source = "ETH historical (90d)"
        print(f"  σ  (ETH hist 90d) : {sigma:.4f}  ({sigma * 100:.1f}%)")

    bs_prob = implied_probability_above(S, K, T, r, sigma, q=0.0)

    gap  = poly_prob - bs_prob
    lean = "Polymarket more bullish" if gap > 0 else "Black-Scholes more bullish"

    print()
    print("=" * W)
    print("  LAYER 0 — ETHEREUM CROSS-MARKET PROBABILITY COMPARISON")
    print(f"  {today}  |  ETH > ${K:,.0f} by {T_date}")
    print("=" * W)
    print(f"  {'Source':<20}  {'Detail':<30}  {'P(above)':>8}")
    print("-" * W)

    q_short  = question[:30] if len(question) <= 30 else question[:27] + "..."
    bs_detail = f"S=${S:,.0f}  σ={sigma:.3f}  r={r:.3f}"
    print(f"  {'Polymarket':<20}  {q_short:<30}  {poly_prob:.1%}")
    print(f"  {'Black-Scholes':<20}  {bs_detail:<30}  {bs_prob:.1%}")
    print("=" * W)
    print()
    print(f"  Inputs  :  S=${S:,.2f}  K=${K:,.0f}  T={T:.3f} yr  r={r:.4f}  σ={sigma:.4f}  q=0")
    print(f"  Vol     :  {sigma_source}")
    print(f"  Gap (Poly − BS) :  {gap:+.1%}  ({lean})")
    print("=" * W)

    log_history(
        ts=datetime.now(timezone.utc),
        poly_prob=poly_prob, bs_prob=bs_prob,
        question=question, S=S, K=K, r=r, sigma=sigma,
    )
    print(f"\n  Row appended to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
