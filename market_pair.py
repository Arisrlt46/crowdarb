"""
CrowdArb Phase C — MarketPair interface and unified pipeline runner.

Abstract base class defines the three-method contract every market must fulfil.
Concrete implementations wrap the existing Layer 0 data fetchers.
run_crowdarb() runs the full Layer 1-3 pipeline on any MarketPair.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np

# Layer 0 — Fed rate sources
from layer0_compare import fetch_cme, fetch_polymarket

# Layer 0 — Bitcoin sources
from layer0_bitcoin import (
    _is_reach_contract,
    _parse_end_date,
    _parse_strike,
    fetch_btc_price,
    fetch_deribit_dvol,
    fetch_tnx,
    historical_vol_btc,
)
from layer0_bs import implied_probability_above, years_to
from layer0_markets import find_markets

# Layers 1-3
from layer1_backtest import brier_score, generate_price_series, log_loss, run_backtest
from layer1_belief import BetaBelief, NaiveMarketMaker
from layer2_avellaneda import ASParams, AvellanedaStoikovMM, estimate_sigma2, run_as_backtest
from layer3_calibration import BayesianBlender


# ── Abstract interface ────────────────────────────────────────────────────────

class MarketPair(ABC):
    """
    Contract for any binary-event market with two probability sources.

    Implementors provide crowd (Polymarket) and professional (model/futures)
    probability estimates, plus metadata needed by the pipeline and UI.
    """

    @abstractmethod
    def get_polymarket_probability(self) -> float:
        """Crowd-sourced probability from Polymarket (YES price on binary contract)."""

    @abstractmethod
    def get_professional_probability(self) -> float:
        """Model- or futures-derived probability from a professional source."""

    @abstractmethod
    def metadata(self) -> dict:
        """
        Return a dict with at minimum:
          name             : str   — short display name
          description      : str   — what event this contract resolves on
          resolution_date  : date  — when the contract settles
        """


# ── Concrete implementation: Fed rate cuts ────────────────────────────────────

class FedRateMarket(MarketPair):
    """
    Polymarket 'no cuts in 2026' contract (inverted) vs CME FedWatch implied probability.
    Results are fetched once and cached on first access.
    """

    def __init__(self) -> None:
        self._loaded      = False
        self._p_poly      = 0.0
        self._p_cme       = 0.0
        self._question    = ""
        self._meeting     : date = date.today()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._question, self._p_poly = fetch_polymarket()
        meeting, _, _, _, p_cme = fetch_cme(date.today())
        self._p_cme    = p_cme
        self._meeting  = meeting
        self._loaded   = True

    def get_polymarket_probability(self) -> float:
        self._ensure_loaded()
        return self._p_poly

    def get_professional_probability(self) -> float:
        self._ensure_loaded()
        return self._p_cme

    def metadata(self) -> dict:
        self._ensure_loaded()
        return {
            "name":            "Fed Rate Cut 2026",
            "description":     "P(any 25 bp rate cut in 2026)",
            "resolution_date": date(2026, 12, 31),
            "poly_source":     self._question,
            "prof_source":     f"CME ZQ futures — meeting {self._meeting}",
        }


# ── Concrete implementation: Bitcoin price level ──────────────────────────────

class BitcoinMarket(MarketPair):
    """
    Highest-volume Polymarket BTC 'reach $X' contract (≥90 days) vs Black-Scholes.
    Contract selection and market data are fetched once and cached on first access.
    """

    MIN_DAYS = 90

    def __init__(self) -> None:
        self._loaded          = False
        self._p_poly          = 0.0
        self._p_bs            = 0.0
        self._question        = ""
        self._resolution_date : date = date.today()
        self._sigma_source    = ""

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        # Contract discovery
        all_btc  = find_markets(["bitcoin", "btc"])
        reach    = [m for m in all_btc if _is_reach_contract(m)]
        eligible = [m for m in reach
                    if (years_to(_parse_end_date(m)) * 365) >= self.MIN_DAYS]
        if not eligible:
            raise RuntimeError(f"No BTC upside contracts with ≥{self.MIN_DAYS} days remaining.")

        best              = eligible[0]   # sorted by volume descending
        self._p_poly      = best["_yes_prob"] or 0.0
        self._question    = best.get("question", "")
        self._resolution_date = _parse_end_date(best)

        K = _parse_strike(self._question)
        T = years_to(self._resolution_date)
        S = fetch_btc_price()
        r = fetch_tnx()

        try:
            sigma             = fetch_deribit_dvol()
            self._sigma_source = "Deribit DVOL"
        except Exception:
            sigma             = historical_vol_btc(lookback_days=90)
            self._sigma_source = "BTC historical (90d)"

        self._p_bs  = implied_probability_above(S, K, T, r, sigma, q=0.0)
        self._loaded = True

    def get_polymarket_probability(self) -> float:
        self._ensure_loaded()
        return self._p_poly

    def get_professional_probability(self) -> float:
        self._ensure_loaded()
        return self._p_bs

    def metadata(self) -> dict:
        self._ensure_loaded()
        return {
            "name":            "Bitcoin Price Level",
            "description":     self._question,
            "resolution_date": self._resolution_date,
            "poly_source":     "Polymarket Gamma API",
            "prof_source":     f"Black-Scholes ({self._sigma_source})",
        }


# ── Unified pipeline runner ───────────────────────────────────────────────────

def run_crowdarb(
    market: MarketPair,
    seed: int = 42,
    n_steps: int = 200,
    eta: float = 0.1,
) -> None:
    """
    Run the full CrowdArb Layer 1-3 pipeline on any MarketPair.

    Layer 1: Beta-Bernoulli belief + naive market-maker backtest.
    Layer 2: Avellaneda-Stoikov inventory-aware market-maker.
    Layer 3: Hedge algorithm blending polymarket vs professional source.
    """
    meta   = market.metadata()
    p_poly = market.get_polymarket_probability()
    p_prof = market.get_professional_probability()
    p_blend = 0.5 * p_poly + 0.5 * p_prof

    W = 66
    print()
    print("=" * W)
    print(f"  CrowdArb — {meta['name']}")
    print(f"  {meta['description']}")
    print(f"  Resolution: {meta['resolution_date']}   seed={seed}")
    print("=" * W)
    print(f"  Polymarket      : {p_poly:.4f}  ({p_poly:.1%})")
    print(f"  Professional    : {p_prof:.4f}  ({p_prof:.1%})  [{meta['prof_source']}]")
    print(f"  Blended prior   : {p_blend:.4f}  ({p_blend:.1%})  [50/50 initial]")
    print()

    # ── Layer 1 + 2: synthetic price series ──────────────────────────────────
    prices = generate_price_series(p0=p_blend, n=n_steps, vol=0.02, seed=seed)
    sigma2 = estimate_sigma2(prices, warmup=20)

    naive_belief = BetaBelief.from_price(p_blend, strength=10.0)
    naive_mm     = NaiveMarketMaker(belief=naive_belief, half_spread=0.02)
    naive_result = run_backtest(naive_mm, prices)

    params    = ASParams(gamma=10.0, k=45.0, sigma2=sigma2, T=1.0)
    as_belief = BetaBelief.from_price(p_blend, strength=10.0)
    as_mm     = AvellanedaStoikovMM(belief=as_belief, params=params)
    as_result = run_as_backtest(as_mm, prices)

    # ── Layer 3: Hedge calibration ────────────────────────────────────────────
    # Synthetic series centred on each source's live snapshot.
    # Poly: proportional variance (30% relative); Prof: tighter (10% relative).
    rng      = np.random.default_rng(seed)
    sig_poly = max(p_poly * 0.30, 0.005)
    sig_prof = max(p_prof * 0.10, 0.001)
    p_poly_sim = np.clip(rng.normal(p_poly, sig_poly, n_steps), 0.001, 0.999)
    p_prof_sim = np.clip(rng.normal(p_prof, sig_prof, n_steps), 0.001, 0.999)
    outcomes   = rng.binomial(1, p_blend, n_steps).astype(int)

    blender    = BayesianBlender(source_names=["polymarket", "professional"], eta=eta)
    bs_blend   = bs_poly = bs_prof = 0.0
    for pp, pr, y in zip(p_poly_sim, p_prof_sim, outcomes):
        pb       = blender.blend([pp, pr])
        bs_blend += (pb - y) ** 2
        bs_poly  += (pp - y) ** 2
        bs_prof  += (pr - y) ** 2
        blender.update([pp, pr], y)

    bs_blend /= n_steps
    bs_poly  /= n_steps
    bs_prof  /= n_steps

    ws               = blender.weight_dict()
    improvement      = (bs_poly - bs_blend) / bs_poly * 100
    pnl_var_naive    = naive_result.history["total_pnl"].var()
    pnl_var_as       = as_result.history["total_pnl"].var()
    var_reduction    = (pnl_var_naive - pnl_var_as) / pnl_var_naive * 100
    p_blend_learned  = ws["polymarket"] * p_poly + ws["professional"] * p_prof

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"  {'LAYER 1 — Naive market-maker'}")
    print(f"    Fills          : {len(naive_result.fills)}")
    print(f"    Final P&L      : {naive_result.final_pnl:+.4f}")
    print(f"    Final inventory: {naive_result.final_inventory:+d}")
    print(f"    Brier score    : {brier_score(naive_result.fills):.4f}")
    print()
    print(f"  {'LAYER 2 — Avellaneda-Stoikov'}")
    print(f"    Fills          : {len(as_result.fills)}")
    print(f"    Final P&L      : {as_result.final_pnl:+.4f}")
    print(f"    Final inventory: {as_result.final_inventory:+d}")
    print(f"    P&L var reduction vs naive: {var_reduction:.1f}%")
    print()
    print(f"  {'LAYER 3 — Bayesian Hedge blender'}")
    print(f"    Polymarket Brier    : {bs_poly:.4f}  (weight → {ws['polymarket']:.3f})")
    print(f"    Professional Brier  : {bs_prof:.4f}  (weight → {ws['professional']:.3f})")
    print(f"    Blended Brier       : {bs_blend:.4f}  ({improvement:+.1f}% vs Polymarket alone)")
    print(f"    Learned-weight blend: {p_blend_learned:.4f}  ({p_blend_learned:.1%})")
    print("=" * W)


# ── Demo: run both markets ────────────────────────────────────────────────────

def main() -> None:
    print("CrowdArb — market-agnostic pipeline demo")
    print("Fetching live data for both markets...\n")

    for market in [FedRateMarket(), BitcoinMarket()]:
        run_crowdarb(market, seed=42, n_steps=200)
        print()


if __name__ == "__main__":
    main()
