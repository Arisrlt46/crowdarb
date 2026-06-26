"""
CrowdArb Phase C/D — MarketPair interface and unified pipeline runner.

Abstract base class defines the three-method contract every market must fulfil.
Concrete implementations wrap the existing Layer 0 data fetchers.
run_crowdarb() runs the full Layer 1-3 pipeline on any MarketPair.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import date

import numpy as np

# Layer 0 — Fed rate sources
from layer0_compare import fetch_cme, fetch_polymarket

# Layer 0 — Bitcoin sources
from layer0_bitcoin import (
    _is_reach_contract,
    fetch_btc_price,
    fetch_deribit_dvol,
    fetch_tnx,
    historical_vol_btc,
)

# Layer 0 — Ethereum sources
from layer0_ethereum import (
    fetch_deribit_eth_dvol,
    fetch_eth_price,
    historical_vol_eth,
    _is_reach_contract as _eth_is_reach_contract,
)

# Shared classifier + parsers (G1/G2)
from layer0_classifier import classify_market, parse_end_date, parse_strike

from layer0_bs import implied_probability_above, years_to
from layer0_markets import _iter_active_markets, find_markets

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
        eligible = [
            m for m in reach
            if (d := parse_end_date(m)) is not None and years_to(d) * 365 >= self.MIN_DAYS
        ]
        if not eligible:
            raise RuntimeError(f"No BTC upside contracts with ≥{self.MIN_DAYS} days remaining.")

        best              = eligible[0]   # sorted by volume descending
        self._p_poly      = best["_yes_prob"] or 0.0
        self._question    = best.get("question", "")
        self._resolution_date = parse_end_date(best)

        K = parse_strike(self._question)
        if K is None:
            raise RuntimeError(f"Cannot parse strike from: {self._question!r}")
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



# ── Concrete implementation: Ethereum price level ─────────────────────────────

class EthereumMarket(MarketPair):
    """
    Highest-volume Polymarket ETH 'reach $X' contract (≥90 days) vs Black-Scholes.
    Uses Deribit ETH DVOL for implied vol; falls back to 90-day historical vol.
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

        all_eth  = find_markets(["ethereum", "eth"])
        reach    = [m for m in all_eth if _eth_is_reach_contract(m)]
        eligible = [
            m for m in reach
            if (d := parse_end_date(m)) is not None and years_to(d) * 365 >= self.MIN_DAYS
        ]
        if not eligible:
            raise RuntimeError(f"No ETH upside contracts with ≥{self.MIN_DAYS} days remaining.")

        best              = eligible[0]
        self._p_poly      = best["_yes_prob"] or 0.0
        self._question    = best.get("question", "")
        self._resolution_date = parse_end_date(best)

        K = parse_strike(self._question)
        if K is None:
            raise RuntimeError(f"Cannot parse strike from: {self._question!r}")
        T = years_to(self._resolution_date)
        S = fetch_eth_price()
        r = fetch_tnx()   # reuse BTC module's ^TNX fetcher (identical call)

        try:
            sigma              = fetch_deribit_eth_dvol()
            self._sigma_source = "Deribit ETH DVOL"
        except Exception:
            sigma              = historical_vol_eth(lookback_days=90)
            self._sigma_source = "ETH historical (90d)"

        self._p_bs   = implied_probability_above(S, K, T, r, sigma, q=0.0)
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
            "name":            "Ethereum Price Level",
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


# ── Market auto-discovery (G3) ────────────────────────────────────────────────

# Ticker patterns used by _btc_or_eth(); word boundaries prevent "beth" / "btcoin" matches
_BTC_RE = re.compile(r"\bbtc\b|bitcoin", re.IGNORECASE)
_ETH_RE = re.compile(r"\beth\b|ethereum", re.IGNORECASE)

# BS uncertainty bounds: skip contracts where the professional probability is near-certain
# in either direction — no meaningful crowd-vs-professional comparison is possible there.
_BS_P_MIN = 0.02
_BS_P_MAX = 0.98


def _btc_or_eth(question: str) -> str | None:
    """Return 'BTC', 'ETH', or None — identifies the underlying in a crypto_level contract."""
    if _BTC_RE.search(question):
        return "BTC"
    if _ETH_RE.search(question):
        return "ETH"
    return None


def discover_markets(
    min_volume_usd: float = 50_000,
    min_days: int = 30,
    max_pages: int = 30,
) -> list[MarketPair]:
    """
    Scan the live Polymarket catalogue and return a MarketPair for every supported
    contract that passes volume, expiry, and BS uncertainty filters.

    Strategy:
      1. Pre-fetch spot price, implied vol, and risk-free rate for BTC and ETH once.
      2. Paginate all active markets; classify each contract.
      3. For crypto_level: apply volume, expiry, and BS sanity filter —
         skip any contract where P(BS) < _BS_P_MIN or > _BS_P_MAX (no meaningful
         crowd-vs-professional comparison when the outcome is near-certain).
         For rate_decision: volume filter only (FedRateMarket owns date logic).
      4. De-duplicate: one winner per (tag, ticker) group — highest volume wins.
      5. Instantiate the appropriate MarketPair subclass for each winner.

    Note: BitcoinMarket / EthereumMarket re-discover the contract internally on first
    access. G4 will replace them with a CryptoLevelMarket that accepts pre-fetched data.
    """
    today = date.today()

    # ── Pre-fetch market data for the BS filter (once per asset) ──────────────
    print("  Pre-fetching market data for BS uncertainty filter...")
    r = 0.0
    try:
        r = fetch_tnx()
    except Exception:
        pass

    spot:  dict[str, float | None] = {"BTC": None, "ETH": None}
    sigma: dict[str, float | None] = {"BTC": None, "ETH": None}

    try:
        spot["BTC"] = fetch_btc_price()
        try:
            sigma["BTC"] = fetch_deribit_dvol()
        except Exception:
            sigma["BTC"] = historical_vol_btc(90)
    except Exception as exc:
        print(f"  Warning: BTC data unavailable ({exc}); BTC contracts skipped.")

    try:
        spot["ETH"] = fetch_eth_price()
        try:
            sigma["ETH"] = fetch_deribit_eth_dvol()
        except Exception:
            sigma["ETH"] = historical_vol_eth(90)
    except Exception as exc:
        print(f"  Warning: ETH data unavailable ({exc}); ETH contracts skipped.")

    for ticker in ("BTC", "ETH"):
        S, sig = spot[ticker], sigma[ticker]
        if S is not None and sig is not None:
            print(f"    {ticker}: S=${S:,.0f}  σ={sig:.1%}")
        else:
            print(f"    {ticker}: unavailable")
    print(f"    r={r:.4f}  ({r*100:.2f}%)\n")

    # ── Scan, classify, filter ─────────────────────────────────────────────────
    # Each record: {"volume": float, "question": str, "p_bs": float|None, "market": dict}
    best: dict[tuple[str, str], dict] = {}
    n_bs_filtered = 0

    for market in _iter_active_markets(max_pages=max_pages):
        tag = classify_market(market)
        if tag in ("unsupported", "equity_level"):
            continue

        vol = float(market.get("volume") or 0)
        if vol < min_volume_usd:
            continue

        question = market.get("question") or ""
        p_bs: float | None = None

        if tag == "crypto_level":
            ticker = _btc_or_eth(question)
            if ticker is None:
                continue

            K = parse_strike(question)
            if K is None:
                continue

            end_date = parse_end_date(market)
            if end_date is None or (end_date - today).days < min_days:
                continue

            S   = spot.get(ticker)
            sig = sigma.get(ticker)
            if S is None or sig is None:
                continue

            p_bs = implied_probability_above(S, K, years_to(end_date), r, sig, q=0.0)
            if not (_BS_P_MIN <= p_bs <= _BS_P_MAX):
                n_bs_filtered += 1
                continue

            key: tuple[str, str] = (tag, ticker)

        elif tag == "rate_decision":
            key = (tag, "FED")

        else:
            continue

        record = {"volume": vol, "question": question, "p_bs": p_bs, "market": market}
        if key not in best or vol > best[key]["volume"]:
            best[key] = record

    # ── Discovery summary ─────────────────────────────────────────────────────
    W = 66
    bs_range = f"BS∈[{_BS_P_MIN:.0%},{_BS_P_MAX:.0%}]"
    print(f"\n{'─' * W}")
    print(f"  discover_markets()  "
          f"[vol≥${min_volume_usd/1_000:.0f}k  days≥{min_days}  {bs_range}  pages={max_pages}]")
    print(f"{'─' * W}")

    sorted_groups = sorted(best.items(), key=lambda kv: kv[1]["volume"], reverse=True)

    if not sorted_groups:
        print(f"  No supported markets found.  ({n_bs_filtered} dropped by BS filter)")
        print(f"{'─' * W}\n")
        return []

    print(f"  {'Type':<18}  {'Asset':<5}  {'P(BS)':>6}  {'Volume':>12}  Question (truncated)")
    print(f"  {'─'*18}  {'─'*5}  {'─'*6}  {'─'*12}  {'─'*36}")
    for (tag, ticker), rec in sorted_groups:
        p_str = f"{rec['p_bs']:.1%}" if rec["p_bs"] is not None else "    —"
        q     = rec["question"][:48]
        print(f"  {tag:<18}  {ticker:<5}  {p_str:>6}  ${rec['volume']:>11,.0f}  {q}")
    print(f"{'─' * W}")
    print(f"  {len(sorted_groups)} selected  |  {n_bs_filtered} dropped by BS filter  "
          f"(P(BS) < {_BS_P_MIN:.0%} or > {_BS_P_MAX:.0%})")
    print(f"{'─' * W}\n")

    # ── Instantiate ───────────────────────────────────────────────────────────
    pairs: list[MarketPair] = []
    for (tag, ticker), _ in sorted_groups:
        if tag == "crypto_level" and ticker == "BTC":
            pairs.append(BitcoinMarket())
        elif tag == "crypto_level" and ticker == "ETH":
            pairs.append(EthereumMarket())
        elif tag == "rate_decision":
            pairs.append(FedRateMarket())

    return pairs


# ── Demo entrypoints ──────────────────────────────────────────────────────────

def main() -> None:
    """Phase D demo: run the full pipeline on the three hardcoded markets."""
    print("CrowdArb — market-agnostic pipeline demo (Phase D)")
    print("Fetching live data for all three markets...\n")

    for market in [FedRateMarket(), BitcoinMarket(), EthereumMarket()]:
        run_crowdarb(market, seed=42, n_steps=200)
        print()


def discover_main() -> None:
    """G3 demo: auto-discover markets from the live catalogue, then run the pipeline."""
    print("CrowdArb G3 — market auto-discovery")
    print("=" * 66)

    pairs = discover_markets(min_volume_usd=50_000, min_days=30)

    if not pairs:
        print("No supported markets discovered.")
        return

    print(f"Running pipeline on {len(pairs)} discovered market(s)...\n")
    for market in pairs:
        run_crowdarb(market, seed=42, n_steps=200)
        print()


if __name__ == "__main__":
    discover_main()
