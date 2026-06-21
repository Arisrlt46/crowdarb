"""CrowdArb Layer 0 — S&P 500 market data and Black-Scholes implied probability."""

import math
from datetime import date

import numpy as np
import yfinance as yf

from layer0_bs import implied_probability_above, years_to

TARGET_DATE = date(2026, 12, 31)
STRIKE = 6500.0
DIVIDEND_YIELD = 0.014   # S&P 500 continuous dividend yield, approximated


def fetch_sp500() -> float:
    """Latest S&P 500 index level from ^GSPC."""
    hist = yf.Ticker("^GSPC").history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch ^GSPC from yfinance.")
    return float(hist["Close"].iloc[-1])


def fetch_vix() -> float:
    """Latest VIX close as a decimal implied vol (e.g. 18.5 → 0.185)."""
    hist = yf.Ticker("^VIX").history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch ^VIX from yfinance.")
    return float(hist["Close"].iloc[-1]) / 100.0


def fetch_tnx() -> float:
    """10-year Treasury yield as a decimal (e.g. 4.52 → 0.0452)."""
    hist = yf.Ticker("^TNX").history(period="1d")
    if hist.empty:
        raise RuntimeError("Could not fetch ^TNX from yfinance.")
    return float(hist["Close"].iloc[-1]) / 100.0


def historical_vol_spy(lookback_days: int = 252) -> float:
    """Annualised vol from SPY daily log-returns over the trailing year."""
    closes = yf.Ticker("SPY").history(period="1y")["Close"].dropna()
    log_rets = np.log(closes / closes.shift(1)).dropna()
    return float(log_rets.tail(lookback_days).std()) * math.sqrt(252)



def main() -> float:
    today = date.today()
    T = years_to(TARGET_DATE)

    print("CrowdArb Layer 0 — S&P 500 implied probability")
    print("─" * 52)
    print(f"  Target   : S&P 500 > {STRIKE:,.0f} by {TARGET_DATE}")
    print(f"  T        : {(TARGET_DATE - today).days} calendar days  ({T:.4f} yr)")
    print()

    print("Fetching market data …")
    S = fetch_sp500()
    print(f"  S  (^GSPC) : {S:,.2f}")

    r = fetch_tnx()
    print(f"  r  (^TNX)  : {r:.4f}  ({r * 100:.2f}%)")

    sigma_source = "^VIX"
    try:
        sigma = fetch_vix()
        print(f"  σ  (^VIX)  : {sigma:.4f}  ({sigma * 100:.1f}%)")
    except Exception as exc:
        print(f"  VIX unavailable ({exc}) — falling back to SPY 252-day historical vol")
        sigma = historical_vol_spy()
        sigma_source = "SPY historical (252d)"
        print(f"  σ  (SPY hist) : {sigma:.4f}  ({sigma * 100:.1f}%)")

    q = DIVIDEND_YIELD
    print(f"  q  (div yield) : {q:.4f}  ({q * 100:.2f}%)")
    print(f"  K              : {STRIKE:,.0f}")
    print()

    # d2 components, printed individually so the arithmetic is auditable
    log_moneyness  = math.log(S / STRIKE)
    carry_term     = (r - q - 0.5 * sigma ** 2) * T
    vol_sqrt_T     = sigma * math.sqrt(T)
    d2             = (log_moneyness + carry_term) / vol_sqrt_T
    prob           = implied_probability_above(S, STRIKE, T, r, sigma, q=q)

    print("Black-Scholes d2 decomposition")
    print(f"  ln(S/K)          : {log_moneyness:+.4f}")
    print(f"  (r-q-σ²/2)·T     : {carry_term:+.4f}")
    print(f"  σ·√T             : {vol_sqrt_T:.4f}")
    print(f"  d2               : {d2:+.4f}")
    print()
    print(
        f"  P(S&P > {STRIKE:,.0f} by {TARGET_DATE})"
        f" = N({d2:.4f}) = {prob:.4f}  ({prob:.1%})"
    )
    print(f"  Volatility source : {sigma_source}")

    return prob


if __name__ == "__main__":
    main()
