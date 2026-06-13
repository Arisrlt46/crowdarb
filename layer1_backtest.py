"""CrowdArb Layer 1 — backtest harness for the naive symmetric market-maker."""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer1_belief import BetaBelief, NaiveMarketMaker


def generate_price_series(p0: float, n: int = 200, vol: float = 0.02, seed: int = 42) -> np.ndarray:
    """Generate a mean-reverting bounded random walk starting at p0."""
    rng = np.random.default_rng(seed)
    prices = [p0]
    for _ in range(n - 1):
        # Weak mean reversion toward p0 keeps series from drifting to 0 or 1
        drift = 0.05 * (p0 - prices[-1])
        shock = rng.normal(drift, vol)
        prices.append(float(np.clip(prices[-1] + shock, 0.01, 0.99)))
    return np.array(prices)


@dataclass
class Fill:
    t: int
    side: str        # "buy" (we buy) or "sell" (we sell)
    price: float
    pred: float      # belief.mean immediately before this fill
    outcome: int     # 1 if YES signal (sell fill), 0 if NO signal (buy fill)


@dataclass
class BacktestResult:
    history: pd.DataFrame
    fills: list[Fill]

    @property
    def final_pnl(self) -> float:
        return float(self.history["total_pnl"].iloc[-1])

    @property
    def pnl_std(self) -> float:
        return float(self.history["total_pnl"].std())

    @property
    def final_inventory(self) -> int:
        return int(self.history["inventory"].iloc[-1])


def run_backtest(mm: NaiveMarketMaker, prices: np.ndarray) -> BacktestResult:
    """Replay a price series through the market-maker, recording fills and P&L each step."""
    cash = 0.0
    inventory = 0
    rows = []
    fills = []

    for t, market_price in enumerate(prices):
        bid, ask, fair = mm.bid, mm.ask, mm.fair_value
        fill_side = None
        outcome = None

        if market_price > ask:
            # Buyer lifts our ask — we sell YES: inventory down, cash up
            cash += ask
            inventory -= 1
            fill_side = "sell"
            outcome = 1  # buyer signals YES
            mm = mm.observe(1)
        elif market_price < bid:
            # Seller hits our bid — we buy YES: inventory up, cash down
            cash -= bid
            inventory += 1
            fill_side = "buy"
            outcome = 0  # seller signals NO
            mm = mm.observe(0)

        if fill_side is not None:
            fill_price = ask if fill_side == "sell" else bid
            fills.append(Fill(t=t, side=fill_side, price=fill_price, pred=fair, outcome=outcome))

        unrealized = inventory * market_price
        rows.append({
            "t": t,
            "market_price": market_price,
            "fair_value": fair,
            "bid": bid,
            "ask": ask,
            "fill": fill_side,
            "cash": cash,
            "inventory": inventory,
            "unrealized_pnl": unrealized,
            "total_pnl": cash + unrealized,
        })

    return BacktestResult(history=pd.DataFrame(rows), fills=fills)


def brier_score(fills: list[Fill]) -> float:
    """
    Brier score: mean squared error between predicted probability and binary outcome.

    BS = (1/N) * Σ (p_i - y_i)²

    Range [0, 1]; 0 is perfect. A naive predictor always guessing 0.5 scores 0.25.
    Each fill contributes one (prediction, outcome) pair: p_i is belief.mean before
    the fill, y_i is 1 for a YES signal (buyer lifts ask) and 0 for a NO signal.
    """
    if not fills:
        return float("nan")
    errors = [(f.pred - f.outcome) ** 2 for f in fills]
    return float(np.mean(errors))


def log_loss(fills: list[Fill], eps: float = 1e-7) -> float:
    """
    Binary log-loss (cross-entropy): penalises confident wrong predictions logarithmically.

    LL = -(1/N) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]

    Range [0, ∞); 0 is perfect. A coin-flip predictor always guessing 0.5 scores log(2) ≈ 0.693.
    Log-loss is much harsher than Brier score for confident errors: predicting p=0.99 on a
    NO outcome incurs -log(0.01) ≈ 4.6, versus a Brier penalty of only 0.98².
    eps clips predictions away from 0 and 1 to avoid log(0).
    """
    if not fills:
        return float("nan")
    terms = []
    for f in fills:
        p = float(np.clip(f.pred, eps, 1.0 - eps))
        terms.append(f.outcome * np.log(p) + (1 - f.outcome) * np.log(1 - p))
    return float(-np.mean(terms))


_SPARK = " ▁▂▃▄▅▆▇█"


def _sparkline(series: pd.Series, width: int = 40) -> str:
    """Render a numeric series as a fixed-width unicode sparkline."""
    lo, hi = series.min(), series.max()
    span = hi - lo or 1e-9
    buckets = len(_SPARK) - 1
    chars = [_SPARK[int((v - lo) / span * buckets)] for v in series.iloc[:: max(1, len(series) // width)]]
    return "".join(chars)


def print_summary(result: BacktestResult, p0: float) -> None:
    df = result.history
    n_buys = sum(1 for f in result.fills if f.side == "buy")
    n_sells = sum(1 for f in result.fills if f.side == "sell")
    bs = brier_score(result.fills)
    ll = log_loss(result.fills)
    W = 54

    print()
    print("=" * W)
    print("  BACKTEST RESULTS — NAIVE SYMMETRIC MARKET-MAKER")
    print("=" * W)
    print(f"  Starting fair value  : {p0:.4f}")
    print(f"  Half-spread          : 0.0200  (4-cent total spread)")
    print(f"  Timesteps            : {len(df)}")
    print(f"  Total fills          : {len(result.fills)}  (buys: {n_buys}  sells: {n_sells})")
    print(f"  Final inventory      : {result.final_inventory:+d}")
    print(f"  Final P&L            : {result.final_pnl:+.4f}")
    print(f"  P&L std dev          : {result.pnl_std:.4f}")
    print(f"  P&L range            : {df['total_pnl'].min():+.4f} … {df['total_pnl'].max():+.4f}")
    print()
    print(f"  Brier score          : {bs:.4f}  (0 = perfect, 0.25 = coin flip)")
    print(f"  Log-loss             : {ll:.4f}  (0 = perfect, 0.693 = coin flip)")
    print()
    print(f"  P&L curve  {_sparkline(df['total_pnl'])}")
    print(f"  Inventory  {_sparkline(df['inventory'].astype(float))}")
    print()

    if result.fills:
        print(f"  {'t':>4}  {'side':<5}  {'price':>6}  {'pred':>6}  {'y':>2}  {'inv':>5}  {'pnl':>8}")
        print(f"  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*2}  {'-'*5}  {'-'*8}")
        for f in result.fills:
            row = df.iloc[f.t]
            print(f"  {f.t:>4}  {f.side:<5}  {f.price:>6.4f}  {f.pred:>6.4f}  {f.outcome:>2}  "
                  f"{int(row['inventory']):>5}  {row['total_pnl']:>+8.4f}")

    print("=" * W)


def plot_results(result: BacktestResult, p0: float) -> None:
    """Three-panel chart: price + quotes, cumulative P&L, inventory."""
    df = result.history
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("CrowdArb Layer 1 — Naive Symmetric Market-Maker", fontsize=13, fontweight="bold")

    # Panel 1: market price, fair value, bid-ask band, fill markers
    ax = axes[0]
    ax.plot(df["t"], df["market_price"], color="black", lw=1, label="Market price")
    ax.plot(df["t"], df["fair_value"], color="steelblue", lw=1.2, label="Fair value")
    ax.fill_between(df["t"], df["bid"], df["ask"], alpha=0.2, color="steelblue", label="Bid-ask spread")
    buys = df[df["fill"] == "buy"]
    sells = df[df["fill"] == "sell"]
    ax.scatter(buys["t"], buys["bid"], marker="^", color="green", s=35, zorder=5, label="Buy fill")
    ax.scatter(sells["t"], sells["ask"], marker="v", color="red", s=35, zorder=5, label="Sell fill")
    ax.set_ylabel("Price")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: cumulative total P&L (cash + mark-to-market inventory)
    ax = axes[1]
    ax.plot(df["t"], df["total_pnl"], color="darkorange", lw=1.2)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Total P&L")

    # Panel 3: YES inventory held over time
    ax = axes[2]
    ax.step(df["t"], df["inventory"], color="purple", lw=1.2, where="post")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("Inventory")
    ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig("layer1_backtest.png", dpi=150)
    print("  Chart saved to layer1_backtest.png")
    plt.show()


def main():
    from layer0_compare import fetch_polymarket

    print("Fetching Polymarket prior...")
    _, p_poly = fetch_polymarket()

    belief = BetaBelief.from_price(p_poly, strength=10.0)
    mm = NaiveMarketMaker(belief=belief, half_spread=0.02)

    prices = generate_price_series(p0=p_poly, n=200, vol=0.02, seed=42)

    print(f"Running backtest: 200 steps, p0={p_poly:.3f}, half_spread=0.02, vol=0.02")
    result = run_backtest(mm, prices)

    print_summary(result, p_poly)
    result.history.to_csv("layer1_backtest.csv", index=False)
    print(f"  History saved to layer1_backtest.csv\n")

    plot_results(result, p_poly)


if __name__ == "__main__":
    main()
