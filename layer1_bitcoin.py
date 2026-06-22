"""CrowdArb Layer 1 — Bitcoin: Bayesian prior from Layer 0 + naive market-maker backtest."""

from __future__ import annotations

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
from layer1_backtest import brier_score, generate_price_series, log_loss, print_summary, run_backtest
from layer1_belief import BetaBelief, NaiveMarketMaker

BACKTEST_SEED  = 42
BACKTEST_N     = 200
BACKTEST_VOL   = 0.02
MIN_DAYS       = 90
POLY_WEIGHT    = 0.5   # equal blend; Layer 3 learns optimal weights via Hedge algorithm
BS_WEIGHT      = 0.5


def fetch_bitcoin_probs() -> tuple[float, float, float, dict]:
    """
    Locate the best eligible BTC contract, fetch market data, return
    (poly_prob, bs_prob, blended_prob, meta).
    """
    all_btc  = find_markets(["bitcoin", "btc"])
    reach    = [m for m in all_btc if _is_reach_contract(m)]
    eligible = [m for m in reach if (years_to(_parse_end_date(m)) * 365) >= MIN_DAYS]
    if not eligible:
        raise RuntimeError(f"No BTC upside contracts with ≥{MIN_DAYS} days remaining.")

    best      = eligible[0]   # sorted by volume descending
    poly_prob = best["_yes_prob"] or 0.0
    question  = best.get("question", "")
    K         = _parse_strike(question)
    T         = years_to(_parse_end_date(best))

    S = fetch_btc_price()
    r = fetch_tnx()

    try:
        sigma        = fetch_deribit_dvol()
        sigma_source = "Deribit DVOL"
    except Exception:
        sigma        = historical_vol_btc(lookback_days=90)
        sigma_source = "BTC historical (90d)"

    bs_prob = implied_probability_above(S, K, T, r, sigma, q=0.0)
    blended = POLY_WEIGHT * poly_prob + BS_WEIGHT * bs_prob

    meta = {
        "question": question,
        "S": S, "K": K, "T": T, "r": r,
        "sigma": sigma, "sigma_source": sigma_source,
    }
    return poly_prob, bs_prob, blended, meta


def main() -> None:
    print("CrowdArb Layer 1 — Bitcoin market-maker backtest")
    print("─" * 54)
    print()

    # ── Step 1: probabilities from Layer 0 ───────────────────────────────────
    print("Step 1: Fetching probabilities from Layer 0...")
    poly_prob, bs_prob, blended, meta = fetch_bitcoin_probs()

    print(f"  Contract      : {meta['question']}")
    print(f"  Polymarket    : {poly_prob:.4f}  ({poly_prob:.1%})")
    print(f"  Black-Scholes : {bs_prob:.4f}  ({bs_prob:.1%})")
    print(f"  Blended prior : {blended:.4f}  ({blended:.1%})"
          f"  [Poly {POLY_WEIGHT:.0%} / BS {BS_WEIGHT:.0%}]")
    print(f"  σ source      : {meta['sigma_source']}")
    print()

    # ── Step 2: Beta-Bernoulli prior ──────────────────────────────────────────
    print("Step 2: Initialising Beta-Bernoulli belief from blended prior...")
    belief = BetaBelief.from_price(blended, strength=10.0)
    mm     = NaiveMarketMaker(belief=belief, half_spread=0.02)
    print(f"  {belief}")
    print()

    # ── Step 3: backtest on synthetic price series ────────────────────────────
    print(f"Step 3: Running backtest  "
          f"(n={BACKTEST_N}, p0={blended:.4f}, vol={BACKTEST_VOL}, seed={BACKTEST_SEED})...")
    prices = generate_price_series(p0=blended, n=BACKTEST_N, vol=BACKTEST_VOL, seed=BACKTEST_SEED)
    result = run_backtest(mm, prices)

    print_summary(result, blended)

    result.history.to_csv("layer1_bitcoin_backtest.csv", index=False)
    print(f"  History saved to layer1_bitcoin_backtest.csv")

    # ── Market-agnosticism confirmation ───────────────────────────────────────
    print()
    print("─" * 54)
    print("  Machinery confirmed market-agnostic:")
    print(f"    Prior p0    : {blended:.4f}")
    print(f"    Fills       : {len(result.fills)}")
    print(f"    Final P&L   : {result.final_pnl:+.4f}")
    print(f"    Brier score : {brier_score(result.fills):.4f}")
    print("─" * 54)


if __name__ == "__main__":
    main()
