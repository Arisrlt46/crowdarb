"""CrowdArb Layer 0 — Bitcoin Polymarket vs Black-Scholes implied probability."""

from __future__ import annotations

import math
import os
import re
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from layer0_bs import implied_probability_above, years_to
from layer0_markets import find_markets

HISTORY_FILE = "layer0_btc_history.csv"
W = 68

# "reach/hit" implies BTC goes above K; exclude "dip/drop/fall" which resolve below K
_UPSIDE   = ["hit", "reach", "above", "exceed", "surpass"]
_DOWNSIDE = ["dip", "below", "drop", "fall", "crash"]


def fetch_btc_price() -> float:
    """Current Bitcoin price (USD) from yfinance."""
    hist = yf.Ticker("BTC-USD").history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch BTC-USD from yfinance.")
    return float(hist["Close"].iloc[-1])


def fetch_tnx() -> float:
    """10-year Treasury yield as a decimal risk-free rate proxy.

    Note: stablecoin lending rates (e.g. USDC ~4-5%) would be more precise
    for crypto, but ^TNX is used here for consistency with the S&P side.
    """
    hist = yf.Ticker("^TNX").history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch ^TNX from yfinance.")
    return float(hist["Close"].iloc[-1]) / 100.0


def fetch_deribit_dvol() -> float:
    """BTC 30-day implied vol from Deribit's DVOL index (returned as decimal)."""
    now_ms = int(time.time() * 1000)
    resp = requests.get(
        "https://www.deribit.com/api/v2/public/get_volatility_index_data",
        params={
            "currency": "BTC",
            "start_timestamp": now_ms - 3_600_000,
            "end_timestamp": now_ms,
            "resolution": "3600",
        },
        timeout=10,
    )
    resp.raise_for_status()
    candles = resp.json()["result"]["data"]
    if not candles:
        raise RuntimeError("Deribit DVOL returned no candles.")
    return float(candles[-1][4]) / 100.0   # close column; DVOL is in %, convert to decimal


def historical_vol_btc(lookback_days: int = 90) -> float:
    """Annualised BTC vol from daily log-returns (365-day basis; crypto trades 24/7)."""
    closes = yf.Ticker("BTC-USD").history(period="1y")["Close"].dropna()
    log_rets = np.log(closes / closes.shift(1)).dropna()
    return float(log_rets.tail(lookback_days).std()) * math.sqrt(365)


def _is_reach_contract(market: dict) -> bool:
    """True for upside contracts ('hit/reach'), False for downside ('dip/drop')."""
    q = market.get("question", "").lower()
    return any(w in q for w in _UPSIDE) and not any(w in q for w in _DOWNSIDE)


def _parse_strike(question: str) -> float:
    """Extract the dollar strike from a Polymarket question (handles $150k, $150,000, $1m)."""
    m = re.search(r"\$([\d,]+(?:\.\d+)?)\s*([kmb]?)", question.lower())
    if not m:
        raise ValueError(f"Cannot parse strike from: {question!r}")
    val = float(m.group(1).replace(",", ""))
    return val * {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "": 1}[m.group(2)]


def _parse_end_date(market: dict) -> date:
    """Parse the contract resolution date from the endDate field."""
    raw = (market.get("endDate") or market.get("endDateIso") or "")[:10]
    return date.fromisoformat(raw)


def log_history(
    ts: datetime,
    poly_prob: float | None,
    bs_prob: float,
    question: str,
    S: float,
    K: float,
    r: float,
    sigma: float,
) -> None:
    """Append one row to layer0_btc_history.csv, creating headers on first write."""
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

    # ── Find BTC upside contracts on Polymarket ───────────────────────────────
    print("Searching Polymarket for Bitcoin price contracts...")
    all_btc  = find_markets(["bitcoin", "btc"])
    reach    = [m for m in all_btc if _is_reach_contract(m)]

    if not reach:
        print("No upside Bitcoin contracts found.")
        return

    print(f"\n  {len(reach)} 'reach/hit' contract(s) found (sorted by volume):\n")
    col_q = 52
    print(f"  {'Question':<{col_q}}  {'P(YES)':>7}  {'Volume':>12}")
    print("  " + "─" * (col_q + 23))
    for m in reach:
        q_text = m.get("question", "")
        q_disp = q_text if len(q_text) <= col_q else q_text[: col_q - 3] + "..."
        p      = m["_yes_prob"]
        vol    = float(m.get("volume") or 0)
        print(f"  {q_disp:<{col_q}}  {f'{p:.1%}' if p is not None else '—':>7}  ${vol:>11,.0f}")
    print()

    # ── Select highest-volume contract with ≥90 days remaining ───────────────
    MIN_DAYS = 90
    eligible = [m for m in reach if (years_to(_parse_end_date(m)) * 365) >= MIN_DAYS]
    if not eligible:
        print(f"  No contracts with ≥{MIN_DAYS} days remaining. Exiting.")
        return
    best      = eligible[0]   # already sorted by volume descending
    poly_prob = best["_yes_prob"]
    question  = best.get("question", "")

    K      = _parse_strike(question)
    T_date = _parse_end_date(best)
    T      = years_to(T_date)

    if T <= 0:
        print(f"  Selected contract expired ({T_date}). Exiting.")
        return

    print(f"  Selected : {question}")
    print(f"  Strike K : ${K:,.0f}")
    print(f"  Expiry   : {T_date}  (T = {T:.3f} yr)\n")

    # ── Fetch market data ─────────────────────────────────────────────────────
    print("Fetching market data...")
    S = fetch_btc_price()
    print(f"  S  (BTC-USD) : ${S:>12,.2f}")

    r = fetch_tnx()
    print(f"  r  (^TNX)    : {r:.4f}  ({r * 100:.2f}%)")

    sigma_source = "Deribit DVOL"
    try:
        sigma = fetch_deribit_dvol()
        print(f"  σ  (Deribit DVOL)      : {sigma:.4f}  ({sigma * 100:.1f}%)")
    except Exception as exc:
        print(f"  Deribit DVOL unavailable ({exc})")
        print("  Falling back to BTC 90-day historical vol...")
        sigma = historical_vol_btc(lookback_days=90)
        sigma_source = "BTC historical (90d)"
        print(f"  σ  (BTC hist 90d) : {sigma:.4f}  ({sigma * 100:.1f}%)")

    bs_prob = implied_probability_above(S, K, T, r, sigma, q=0.0)

    # ── Side-by-side comparison table ─────────────────────────────────────────
    gap  = (poly_prob - bs_prob) if poly_prob is not None else None
    lean = ("Polymarket more bullish" if gap > 0 else "Black-Scholes more bullish") if gap is not None else ""

    print()
    print("=" * W)
    print("  LAYER 0 — BITCOIN CROSS-MARKET PROBABILITY COMPARISON")
    print(f"  {today}  |  BTC > ${K:,.0f} by {T_date}")
    print("=" * W)
    print(f"  {'Source':<20}  {'Detail':<30}  {'P(above)':>8}")
    print("-" * W)

    q_short   = question[:30] if len(question) <= 30 else question[:27] + "..."
    poly_str  = f"{poly_prob:.1%}" if poly_prob is not None else "—"
    bs_detail = f"S=${S:,.0f}  σ={sigma:.3f}  r={r:.3f}"

    print(f"  {'Polymarket':<20}  {q_short:<30}  {poly_str:>8}")
    print(f"  {'Black-Scholes':<20}  {bs_detail:<30}  {bs_prob:.1%}")
    print("=" * W)
    print()
    print(f"  Inputs  :  S=${S:,.2f}  K=${K:,.0f}  T={T:.3f} yr  r={r:.4f}  σ={sigma:.4f}  q=0")
    print(f"  Vol     :  {sigma_source}")
    if gap is not None:
        print(f"  Gap (Poly − BS) :  {gap:+.1%}  ({lean})")
    print()
    print("  Note: Polymarket = crowd odds on a specific binary contract.")
    print("        Black-Scholes = risk-neutral probability under log-normal BTC dynamics.")
    print("        A positive gap means the crowd is more bullish than options pricing implies.")
    print("=" * W)

    # ── Append to history ─────────────────────────────────────────────────────
    log_history(
        ts=datetime.now(timezone.utc),
        poly_prob=poly_prob,
        bs_prob=bs_prob,
        question=question,
        S=S,
        K=K,
        r=r,
        sigma=sigma,
    )
    print(f"\n  Row appended to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
