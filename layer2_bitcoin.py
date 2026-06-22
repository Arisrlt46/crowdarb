"""
CrowdArb Layer 2 — Bitcoin Avellaneda-Stoikov market-maker backtest.

Same A-S model as layer2_avellaneda.py; prior pinned to the live blended
estimate (Poly 50% + BS 50%) from layer0_bitcoin / layer1_bitcoin.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from layer1_backtest import brier_score, generate_price_series, log_loss, run_backtest
from layer1_belief import BetaBelief, NaiveMarketMaker
from layer2_avellaneda import (
    ASParams,
    AvellanedaStoikovMM,
    estimate_sigma2,
    print_comparison,
    run_as_backtest,
)

# Pinned from live blended prior (Poly 4.7% × 0.5 + BS 0.2% × 0.5 on 2026-06-22)
BACKTEST_P0   = 0.0243
BACKTEST_SEED = 42

CHART_FILE = "layer2_bitcoin_comparison.png"
NAIVE_CSV  = "layer1_bitcoin_backtest.csv"
AS_CSV     = "layer2_bitcoin_backtest.csv"


def plot_comparison_btc(naive, asmm) -> None:
    """Four-panel comparison chart; saves to CHART_FILE."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("CrowdArb Layer 2 — Bitcoin: Naive vs Avellaneda-Stoikov",
                 fontsize=13, fontweight="bold")

    t = naive.history["t"]

    ax = axes[0]
    ax.plot(t, naive.history["market_price"], color="black", lw=1, label="Market price")
    ax.plot(t, naive.history["fair_value"], color="grey", lw=1, ls="--", label="Fair value")
    ax.fill_between(t, naive.history["bid"], naive.history["ask"],
                    alpha=0.15, color="steelblue", label="Naive bid-ask")
    ax.fill_between(t, asmm.history["bid"], asmm.history["ask"],
                    alpha=0.15, color="darkorange", label="A-S bid-ask")
    ax.plot(t, asmm.history["reservation_price"], color="darkorange", lw=1, ls=":",
            label="A-S reservation price")
    ax.set_ylabel("Price")
    ax.set_ylim(0, max(naive.history["market_price"].max() * 1.5, 0.15))
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(t, naive.history["total_pnl"], color="steelblue", lw=1.2, label="Naive")
    ax.plot(t, asmm.history["total_pnl"], color="darkorange", lw=1.2, label="A-S")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Total P&L")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.step(t, naive.history["inventory"], color="steelblue", lw=1.2, where="post", label="Naive")
    ax.step(t, asmm.history["inventory"], color="darkorange", lw=1.2, where="post", label="A-S")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Inventory")
    ax.legend(fontsize=8)

    ax = axes[3]
    naive_spread = naive.history["ask"] - naive.history["bid"]
    as_spread    = asmm.history["ask"]  - asmm.history["bid"]
    ax.plot(t, naive_spread, color="steelblue", lw=1, label="Naive spread")
    ax.plot(t, as_spread,    color="darkorange", lw=1, label="A-S spread")
    ax.set_ylabel("Spread width")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    print(f"  Chart saved to {CHART_FILE}")


def main() -> None:
    print("CrowdArb Layer 2 — Bitcoin Avellaneda-Stoikov backtest")
    print("─" * 56)
    print(f"  p0 = {BACKTEST_P0:.4f}  seed = {BACKTEST_SEED}")
    print()

    prices = generate_price_series(p0=BACKTEST_P0, n=200, vol=0.02, seed=BACKTEST_SEED)
    sigma2 = estimate_sigma2(prices, warmup=20)
    print(f"  Estimated σ² from warmup window: {sigma2:.6f}")
    print()

    # ── Naive baseline (Layer 1) ──────────────────────────────────────────────
    naive_belief = BetaBelief.from_price(BACKTEST_P0, strength=10.0)
    naive_mm     = NaiveMarketMaker(belief=naive_belief, half_spread=0.02)
    naive_result = run_backtest(naive_mm, prices)

    # ── Avellaneda-Stoikov ────────────────────────────────────────────────────
    # Same γ=10, k=45 as the Fed-rate run; adverse-selection half-spread ≈ 0.02
    # on a [0,1] binary market regardless of the prior level.
    params    = ASParams(gamma=10.0, k=45.0, sigma2=sigma2, T=1.0)
    as_belief = BetaBelief.from_price(BACKTEST_P0, strength=10.0)
    as_mm     = AvellanedaStoikovMM(belief=as_belief, params=params)
    as_result = run_as_backtest(as_mm, prices)

    print_comparison(naive_result, as_result, BACKTEST_P0)

    # ── Variance reduction summary ────────────────────────────────────────────
    pnl_var_naive = naive_result.history["total_pnl"].var()
    pnl_var_as    = as_result.history["total_pnl"].var()
    var_reduction = (pnl_var_naive - pnl_var_as) / pnl_var_naive

    naive_range = (naive_result.history["total_pnl"].max()
                   - naive_result.history["total_pnl"].min())
    as_range    = (as_result.history["total_pnl"].max()
                   - as_result.history["total_pnl"].min())

    print()
    print(f"  P&L variance  — Naive: {pnl_var_naive:.4f}  A-S: {pnl_var_as:.4f}"
          f"  reduction: {var_reduction:.1%}")
    print(f"  P&L range     — Naive: {naive_range:.4f}   A-S: {as_range:.4f}")
    print(f"  Final inv     — Naive: {naive_result.final_inventory:+d}"
          f"  A-S: {as_result.final_inventory:+d}")
    print()

    naive_result.history.to_csv(NAIVE_CSV, index=False)
    as_result.history.to_csv(AS_CSV, index=False)
    print(f"  CSVs saved: {NAIVE_CSV}, {AS_CSV}")

    plot_comparison_btc(naive_result, as_result)


if __name__ == "__main__":
    main()
