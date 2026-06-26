"""
Layer 0b — implied Fed rate-cut probability derived from CME 30-day Fed Funds Futures.

Method: replicates the CME FedWatch probability formula using publicly available data.
  - Rate proxy  : ^IRX (13-week T-bill yield) via yfinance — trades within ~5 bps of EFFR
  - Futures data: ZQ contracts (30-day Fed Funds Futures) via yfinance
  - Formula     : P(cut) = (r_current - r_implied) / (0.25 * days_after / days_in_month)

No API key required. yfinance must be installed: pip install yfinance
"""

import calendar
from datetime import date, datetime, timezone

import pandas as pd
import yfinance as yf

# 2026 FOMC meeting dates (second day = decision day). Source: federalreserve.gov
FOMC_2026 = [
    date(2026, 1, 29),
    date(2026, 3, 19),
    date(2026, 5, 7),
    date(2026, 6, 18),
    date(2026, 7, 30),
    date(2026, 9, 17),
    date(2026, 10, 29),
    date(2026, 12, 10),
]

# CME month codes for ZQ futures tickers (A=Jan offset to standard futures convention)
_MONTH_CODES = "FGHJKMNQUVXZ"


def zq_ticker(year: int, month: int) -> str:
    """Return the Yahoo Finance ticker for the ZQ contract for a given year/month."""
    return f"ZQ{_MONTH_CODES[month - 1]}{str(year)[-2:]}.CBT"


def get_current_rate() -> float:
    """Return the 13-week T-bill yield as a proxy for the current effective Fed funds rate."""
    hist = yf.Ticker("^IRX").history(period="5d")
    if hist.empty:
        raise RuntimeError("Could not fetch ^IRX from yfinance")
    return float(hist["Close"].iloc[-1])


def get_futures_implied_rate(year: int, month: int) -> float:
    """Return the annualised rate implied by the ZQ futures contract for a given month."""
    ticker = zq_ticker(year, month)
    hist = yf.Ticker(ticker).history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No data for {ticker}")
    price = float(hist["Close"].iloc[-1])
    return 100.0 - price  # futures price = 100 - expected monthly avg rate


def next_fomc_after(today: date) -> date:
    """Return the first FOMC decision date strictly after today."""
    future = [d for d in FOMC_2026 if d > today]
    if not future:
        raise RuntimeError("No remaining 2026 FOMC dates. Update FOMC_2026.")
    return future[0]


def cut_probability(meeting: date, r_current: float) -> float:
    """
    Compute P(25 bp rate cut) at a given FOMC meeting using the CME FedWatch formula.

    The 30-day Fed Funds Futures contract represents the average overnight rate
    for the entire calendar month. The meeting splits the month into two regimes:
      - days 1 … D-1   : rate = r_current   (pre-decision)
      - days D … N      : rate = r_after     (post-decision)

    Implied monthly average = (days_before * r_current + days_after * r_after) / N
    Solving for r_after and comparing to r_current - 0.25 gives P(cut).

    Near-month-end fallback: when days_after < 5, back-solving amplifies noise by
    N/days_after (up to 15× for a day-30 meeting), producing probabilities > 1.
    CME's own methodology handles this by reading the next month's ZQ contract
    directly as r_after — its full 31-day window is entirely post-decision.
    """
    N = calendar.monthrange(meeting.year, meeting.month)[1]  # days in meeting month
    D = meeting.day
    days_before = D - 1
    days_after  = N - D + 1

    r_implied = get_futures_implied_rate(meeting.year, meeting.month)

    if days_after < 5:
        # Switch to next-month ZQ: all its days are post-decision, so its implied
        # rate is a direct, stable read on r_after without division-by-small-number.
        nm = meeting.month % 12 + 1
        ny = meeting.year + (1 if meeting.month == 12 else 0)
        r_after = get_futures_implied_rate(ny, nm)
    else:
        # Standard back-solve: r_implied = (days_before*r_current + days_after*r_after) / N
        r_after = (r_implied * N - days_before * r_current) / days_after

    # r_after = r_current - P(cut) * 0.25  =>  P(cut) = (r_current - r_after) / 0.25
    prob = (r_current - r_after) / 0.25
    return max(0.0, min(1.0, prob))  # clamp residual futures noise


def main():
    today = date.today()
    meeting = next_fomc_after(today)

    print(f"Fetching rate data for FOMC meeting on {meeting}...")

    r_current = get_current_rate()
    r_implied = get_futures_implied_rate(meeting.year, meeting.month)
    prob = cut_probability(meeting, r_current)

    print()
    print("=" * 50)
    print("CME FEDWATCH (REPLICATED)")
    print("=" * 50)
    print(f"  Next FOMC meeting    : {meeting}")
    print(f"  Current rate proxy   : {r_current:.3f}%  (^IRX, 13-week T-bill)")
    print(f"  Futures implied avg  : {r_implied:.3f}%  ({zq_ticker(meeting.year, meeting.month)})")
    print(f"  P(25 bp rate cut)    : {prob:.1%}")

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "CME ZQ futures via yfinance",
        "meeting_date": meeting.isoformat(),
        "rate_proxy_pct": r_current,
        "futures_implied_rate_pct": r_implied,
        "cut_probability": prob,
    }
    pd.DataFrame([row]).to_csv("layer0_fedwatch.csv", index=False)
    print(f"\nSaved to layer0_fedwatch.csv")


if __name__ == "__main__":
    main()
