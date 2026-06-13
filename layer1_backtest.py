"""CrowdArb Layer 1 — backtest harness for the naive symmetric market-maker."""

from dataclasses import dataclass, field

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
    side: str   # "buy" (we buy) or "sell" (we sell)
    price: float


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

        if market_price > ask:
            # Buyer lifts our ask — we sell YES: inventory down, cash up
            cash += ask
            inventory -= 1
            fill_side = "sell"
            mm = mm.observe(1)  # buyer signals YES (bullish)
        elif market_price < bid:
            # Seller hits our bid — we buy YES: inventory up, cash down
            cash -= bid
            inventory += 1
            fill_side = "buy"
            mm = mm.observe(0)  # seller signals NO (bearish)

        if fill_side:
            fills.append(Fill(t=t, side=fill_side, price=ask if fill_side == "sell" else bid))

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
    print(f"  P&L curve  {_sparkline(df['total_pnl'])}")
    print(f"  Inventory  {_sparkline(df['inventory'].astype(float))}")
    print()

    if result.fills:
        print(f"  {'t':>4}  {'side':<5}  {'price':>6}  {'inventory':>9}  {'total_pnl':>9}")
        print(f"  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*9}")
        for f in result.fills:
            row = df.iloc[f.t]
            print(f"  {f.t:>4}  {f.side:<5}  {f.price:>6.4f}  {int(row['inventory']):>9}  {row['total_pnl']:>+9.4f}")

    print("=" * W)


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
    print(f"  History saved to layer1_backtest.csv")


if __name__ == "__main__":
    main()
