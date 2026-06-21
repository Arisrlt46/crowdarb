"""CrowdArb Layer 0 — shared Black-Scholes implied-probability utilities.

Market-agnostic: works for equities, crypto, or any log-normally modelled asset.
Callers supply q=0 for assets with no carry (Bitcoin) or q=dividend_yield for indices.
"""

import math
from datetime import date

from scipy.stats import norm


def years_to(target: date) -> float:
    """Calendar days from today to target in years (Act/365)."""
    return (target - date.today()).days / 365.0


def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> float:
    return (math.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def implied_probability_above(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Risk-neutral P(S_T > K) = N(d2) under Black-Scholes."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    return float(norm.cdf(_d2(S, K, T, r, sigma, q)))


def implied_probability_below(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Risk-neutral P(S_T < K) = N(-d2) under Black-Scholes."""
    if T <= 0:
        return 1.0 if S < K else 0.0
    return float(norm.cdf(-_d2(S, K, T, r, sigma, q)))
