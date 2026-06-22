"""
CrowdArb Layer 2 — Avellaneda-Stoikov inventory-aware market-maker.

Reservation price  : r = p̂ - q·γ·σ²·(T-t)
Optimal spread     : δ = γ·σ²·(T-t) + (2/γ)·ln(1 + γ/k)
Quotes             : bid = r - δ/2,  ask = r + δ/2

p̂  = fair value from Beta-Bernoulli belief
q   = signed YES inventory (positive = long)
γ   = risk-aversion parameter
σ²  = estimated price variance per step
T-t = normalised time remaining in [0, 1]
k   = order-arrival intensity (controls adverse-selection term)
"""

from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer1_belief import BetaBelief
from layer1_backtest import BacktestResult, Fill, brier_score, generate_price_series, log_loss

BACKTEST_P0 = 0.72   # fixed prior; keeps backtest results reproducible independent of live feed
BACKTEST_SEED = 42


@dataclass
class ASParams:
    gamma: float   # risk-aversion coefficient (γ)
    k: float       # order-arrival intensity
    sigma2: float  # price variance per step (estimated from data)
    T: float = 1.0 # total normalised time horizon


@dataclass
class AvellanedaStoikovMM:
    """Inventory-aware market-maker using the Avellaneda-Stoikov optimal quoting policy."""

    belief: BetaBelief
    params: ASParams
    inventory: int = 0
    t: float = 0.0  # normalised elapsed time

    @property
    def fair_value(self) -> float:
        return self.belief.mean

    @property
    def time_remaining(self) -> float:
        # Clamp to a small positive value so spread stays finite at expiry
        return max(self.params.T - self.t, 1e-4)

    @property
    def reservation_price(self) -> float:
        """Inventory-adjusted mid: shifts away from fair value to reduce position risk."""
        p = self.params
        return self.fair_value - self.inventory * p.gamma * p.sigma2 * self.time_remaining

    @property
    def half_spread(self) -> float:
        """Half the optimal spread; independent of inventory, grows with risk and time."""
        p = self.params
        tau = self.time_remaining
        inventory_risk = p.gamma * p.sigma2 * tau
        adverse_selection = (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.k)
        return 0.5 * (inventory_risk + adverse_selection)

    @property
    def bid(self) -> float:
        return max(0.001, self.reservation_price - self.half_spread)

    @property
    def ask(self) -> float:
        return min(0.999, self.reservation_price + self.half_spread)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def observe(self, outcome: int) -> "AvellanedaStoikovMM":
        """Return updated market-maker after one Bernoulli observation."""
        return AvellanedaStoikovMM(
            belief=self.belief.update(outcome),
            params=self.params,
            inventory=self.inventory,
            t=self.t,
        )

    def step(self, dt: float) -> "AvellanedaStoikovMM":
        """Advance the clock by dt (normalised time units)."""
        return AvellanedaStoikovMM(
            belief=self.belief,
            params=self.params,
            inventory=self.inventory,
            t=min(self.t + dt, self.params.T),
        )

    def with_inventory(self, inventory: int) -> "AvellanedaStoikovMM":
        return AvellanedaStoikovMM(
            belief=self.belief, params=self.params, inventory=inventory, t=self.t
        )

    def __repr__(self) -> str:
        return (
            f"ASMM  fair={self.fair_value:.4f}  r={self.reservation_price:.4f}  "
            f"bid={self.bid:.4f}  ask={self.ask:.4f}  "
            f"spread={self.spread:.4f}  inv={self.inventory:+d}  t={self.t:.3f}"
        )


def estimate_sigma2(prices: np.ndarray, warmup: int = 20) -> float:
    """Estimate price variance per step from the first `warmup` steps of the series."""
    diffs = np.diff(prices[:warmup])
    return float(np.var(diffs))


def run_as_backtest(mm: AvellanedaStoikovMM, prices: np.ndarray) -> BacktestResult:
    """Replay a price series through the A-S market-maker."""
    cash = 0.0
    inventory = 0
    rows = []
    fills = []
    n = len(prices)
    dt = 1.0 / n  # each step consumes this fraction of the time horizon

    for t, market_price in enumerate(prices):
        bid, ask, r = mm.bid, mm.ask, mm.reservation_price
        fair = mm.fair_value
        fill_side = None
        outcome = None

        if market_price > ask:
            cash += ask
            inventory -= 1
            fill_side = "sell"
            outcome = 1
            mm = mm.observe(1)
        elif market_price < bid:
            cash -= bid
            inventory += 1
            fill_side = "buy"
            outcome = 0
            mm = mm.observe(0)

        if fill_side is not None:
            fill_price = ask if fill_side == "sell" else bid
            fills.append(Fill(t=t, side=fill_side, price=fill_price, pred=fair, outcome=outcome))

        mm = mm.with_inventory(inventory).step(dt)

        unrealized = inventory * market_price
        rows.append({
            "t": t,
            "market_price": market_price,
            "fair_value": fair,
            "reservation_price": r,
            "bid": bid,
            "ask": ask,
            "fill": fill_side,
            "cash": cash,
            "inventory": inventory,
            "unrealized_pnl": unrealized,
            "total_pnl": cash + unrealized,
        })

    return BacktestResult(history=pd.DataFrame(rows), fills=fills)


def print_comparison(naive: BacktestResult, asmm: BacktestResult, p0: float) -> None:
    W = 62
    print()
    print("=" * W)
    print("  LAYER 2 — NAIVE vs AVELLANEDA-STOIKOV COMPARISON")
    print("=" * W)
    print(f"  {'Metric':<28}  {'Naive':>10}  {'A-S':>10}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*10}")

    def row(label, nv, av, fmt=".4f"):
        nv_s = format(nv, fmt)
        av_s = format(av, fmt)
        print(f"  {label:<28}  {nv_s:>10}  {av_s:>10}")

    row("Final P&L", naive.final_pnl, asmm.final_pnl, "+.4f")
    row("P&L std dev", naive.pnl_std, asmm.pnl_std)
    row("Final inventory", naive.final_inventory, asmm.final_inventory, "+d")
    row("Total fills", len(naive.fills), len(asmm.fills), "d")
    row("Brier score", brier_score(naive.fills), brier_score(asmm.fills))
    row("Log-loss", log_loss(naive.fills), log_loss(asmm.fills))

    naive_range = naive.history["total_pnl"].max() - naive.history["total_pnl"].min()
    as_range = asmm.history["total_pnl"].max() - asmm.history["total_pnl"].min()
    row("P&L range (max-min)", naive_range, as_range)

    print("=" * W)


def plot_comparison(naive: BacktestResult, asmm: BacktestResult) -> None:
    """Four-panel comparison: price/quotes, P&L curves, inventory, spread width."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    fig.suptitle("CrowdArb Layer 2 — Naive vs Avellaneda-Stoikov", fontsize=13, fontweight="bold")

    t = naive.history["t"]

    # Panel 1: market price and fair value (shared)
    ax = axes[0]
    ax.plot(t, naive.history["market_price"], color="black", lw=1, label="Market price")
    ax.plot(t, naive.history["fair_value"], color="grey", lw=1, ls="--", label="Fair value")
    ax.fill_between(t, naive.history["bid"], naive.history["ask"], alpha=0.15, color="steelblue", label="Naive bid-ask")
    ax.fill_between(t, asmm.history["bid"], asmm.history["ask"], alpha=0.15, color="darkorange", label="A-S bid-ask")
    ax.plot(t, asmm.history["reservation_price"], color="darkorange", lw=1, ls=":", label="A-S reservation price")
    ax.set_ylabel("Price")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: cumulative P&L
    ax = axes[1]
    ax.plot(t, naive.history["total_pnl"], color="steelblue", lw=1.2, label="Naive")
    ax.plot(t, asmm.history["total_pnl"], color="darkorange", lw=1.2, label="A-S")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Total P&L")
    ax.legend(fontsize=8)

    # Panel 3: inventory
    ax = axes[2]
    ax.step(t, naive.history["inventory"], color="steelblue", lw=1.2, where="post", label="Naive")
    ax.step(t, asmm.history["inventory"], color="darkorange", lw=1.2, where="post", label="A-S")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Inventory")
    ax.legend(fontsize=8)

    # Panel 4: quoted spread width
    ax = axes[3]
    naive_spread = naive.history["ask"] - naive.history["bid"]
    as_spread = asmm.history["ask"] - asmm.history["bid"]
    ax.plot(t, naive_spread, color="steelblue", lw=1, label="Naive spread")
    ax.plot(t, as_spread, color="darkorange", lw=1, label="A-S spread")
    ax.set_ylabel("Spread width")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("layer2_comparison.png", dpi=150)
    print("  Chart saved to layer2_comparison.png")
    plt.show()


def main():
    from layer1_backtest import run_backtest, NaiveMarketMaker

    prices = generate_price_series(p0=BACKTEST_P0, n=200, vol=0.02, seed=BACKTEST_SEED)
    sigma2 = estimate_sigma2(prices, warmup=20)
    print(f"Estimated σ² from warmup window: {sigma2:.6f}")

    # Naive market-maker (Layer 1 baseline)
    naive_belief = BetaBelief.from_price(BACKTEST_P0, strength=10.0)
    naive_mm = NaiveMarketMaker(belief=naive_belief, half_spread=0.02)
    naive_result = run_backtest(naive_mm, prices)

    # Avellaneda-Stoikov market-maker
    # γ=10, k=45 calibrated so adverse-selection half-spread ≈ 0.02 on a [0,1] market.
    # With equity-scale γ=0.1, the 2/γ term dominates and quotes collapse to the clamps.
    params = ASParams(gamma=10.0, k=45.0, sigma2=sigma2, T=1.0)
    as_belief = BetaBelief.from_price(BACKTEST_P0, strength=10.0)
    as_mm = AvellanedaStoikovMM(belief=as_belief, params=params)
    as_result = run_as_backtest(as_mm, prices)

    print_comparison(naive_result, as_result, BACKTEST_P0)

    naive_result.history.to_csv("layer1_backtest.csv", index=False)
    as_result.history.to_csv("layer2_backtest.csv", index=False)
    print("  CSVs saved.\n")

    plot_comparison(naive_result, as_result)


if __name__ == "__main__":
    main()
