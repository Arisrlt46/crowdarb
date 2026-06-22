"""
CrowdArb Layer 3 — Bitcoin: Bayesian model averaging over Polymarket + Black-Scholes.

Same Hedge algorithm as layer3_calibration.py; sources adapted for the BTC tail regime.
Polymarket overprices tail events (crowd bullishness); Black-Scholes underprices them
(log-normal dynamics miss fat tails). The blender learns which to trust from outcomes.

Weight update rule:  log wᵢ ← log wᵢ + η·[y·log(pᵢ) + (1−y)·log(1−pᵢ)]
Blend:               p̂ = Σ wᵢ·pᵢ
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from layer3_calibration import BayesianBlender

# Live snapshots (2026-06-22); anchor the synthetic simulation to observed regime
P_POLY_LIVE = 0.045   # Polymarket: 4.5%  — BTC > $150k by Dec 31 2026
P_BS_LIVE   = 0.002   # Black-Scholes: 0.2% — σ=41.3%, T=0.53yr, S=$63k, K=$150k

BACKTEST_SEED = 7
N_STEPS       = 200
P_TRUE        = 0.03   # user-specified calibration target; realistic for a 22× OTM BTC contract
ETA           = 0.1

CHART_FILE = "layer3_bitcoin_calibration.png"
CSV_FILE   = "layer3_bitcoin_calibration.csv"


def generate_btc_pairs(
    n: int = N_STEPS,
    p_true: float = P_TRUE,
    seed: int = BACKTEST_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate (p_poly, p_bs) price paths and binary outcomes around p_true.

    Polymarket: upward-biased (+1.5%), moderate variance — retail crowd overprices tail events.
    Black-Scholes: downward-biased (−2.5%), low variance — log-normal model misses fat tails.
    Both biases calibrated to match the observed live snapshot (4.5% vs 0.2% at p_true≈3%).
    """
    rng    = np.random.default_rng(seed)
    p_poly = np.clip(rng.normal(loc=p_true + 0.015, scale=0.012, size=n), 0.001, 0.50)
    p_bs   = np.clip(rng.normal(loc=p_true - 0.025, scale=0.003,  size=n), 0.001, 0.50)
    outcomes = rng.binomial(1, p_true, size=n).astype(int)
    return p_poly, p_bs, outcomes


def run_btc_calibration(
    p_poly: np.ndarray,
    p_bs: np.ndarray,
    outcomes: np.ndarray,
    eta: float = ETA,
) -> pd.DataFrame:
    """Replay (p_poly, p_bs, outcome) triples through the Hedge blender."""
    blender    = BayesianBlender(source_names=["polymarket", "bs"], eta=eta)
    rows       = []
    brier_sums = {"blend": 0.0, "poly": 0.0, "bs": 0.0}

    for i, (pp, pb, y) in enumerate(zip(p_poly, p_bs, outcomes)):
        p_blend = blender.blend([pp, pb])
        ws      = blender.weights

        brier_sums["blend"] += (p_blend - y) ** 2
        brier_sums["poly"]  += (pp - y) ** 2
        brier_sums["bs"]    += (pb - y) ** 2

        rows.append({
            "step":        i,
            "p_poly":      pp,
            "p_bs":        pb,
            "p_blend":     p_blend,
            "outcome":     y,
            "w_poly":      ws[0],
            "w_bs":        ws[1],
            "brier_blend": brier_sums["blend"] / (i + 1),
            "brier_poly":  brier_sums["poly"]  / (i + 1),
            "brier_bs":    brier_sums["bs"]    / (i + 1),
        })

        blender.update([pp, pb], y)   # update after recording prediction (no look-ahead)

    return pd.DataFrame(rows)


def print_btc_summary(df: pd.DataFrame, blender: BayesianBlender) -> None:
    W     = 64
    final = df.iloc[-1]
    ws    = blender.weight_dict()

    print()
    print("=" * W)
    print("  LAYER 3 — BITCOIN CROSS-MARKET CALIBRATION RESULTS")
    print("=" * W)
    print(f"  Steps     : {len(df)}   p_true = {P_TRUE:.3f}   seed = {BACKTEST_SEED}")
    print()
    print(f"  {'Source':<24}  {'Brier score':>12}  {'Final weight':>12}")
    print(f"  {'-'*24}  {'-'*12}  {'-'*12}")
    print(f"  {'Polymarket':<24}  {final['brier_poly']:>12.4f}  {ws['polymarket']:>12.4f}")
    print(f"  {'Black-Scholes':<24}  {final['brier_bs']:>12.4f}  {ws['bs']:>12.4f}")
    print(f"  {'Blended (Layer 3)':<24}  {final['brier_blend']:>12.4f}  {'—':>12}")
    print()

    improvement = (final["brier_poly"] - final["brier_blend"]) / final["brier_poly"] * 100
    print(f"  Brier improvement vs Polymarket alone : {improvement:+.1f}%")
    print(f"  Final trust weights : Poly {ws['polymarket']:.3f}   BS {ws['bs']:.3f}")
    print()

    # Apply learned weights to the live snapshot
    p_blend_live = ws["polymarket"] * P_POLY_LIVE + ws["bs"] * P_BS_LIVE
    print(f"  Live values  : Poly={P_POLY_LIVE:.1%}  BS={P_BS_LIVE:.1%}")
    print(f"  Learned-weight blend of live values : {p_blend_live:.4f}  ({p_blend_live:.1%})")
    print("=" * W)


def plot_btc_calibration(df: pd.DataFrame) -> None:
    """Three-panel chart: probability paths, cumulative Brier scores, weight evolution."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "CrowdArb Layer 3 — Bitcoin: Polymarket vs Black-Scholes Calibration",
        fontsize=13, fontweight="bold",
    )

    t = df["step"]

    ax = axes[0]
    ax.plot(t, df["p_poly"],  color="steelblue",  lw=1,   label="Polymarket")
    ax.plot(t, df["p_bs"],    color="darkorange",  lw=1,   label="Black-Scholes")
    ax.plot(t, df["p_blend"], color="green",        lw=1.4, label="Blended (Layer 3)")
    ax.axhline(P_TRUE, color="black", lw=0.8, ls="--", label=f"p_true = {P_TRUE:.2f}")
    yes_steps = t[df["outcome"] == 1]
    ax.scatter(yes_steps, [P_TRUE * 0.4] * len(yes_steps),
               marker="|", color="black", s=25, alpha=0.7, label="YES outcome")
    ax.set_ylabel("P(BTC > $150k)")
    ax.set_ylim(0, 0.12)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(t, df["brier_poly"],  color="steelblue",  lw=1.2, label="Polymarket")
    ax.plot(t, df["brier_bs"],    color="darkorange",  lw=1.2, label="Black-Scholes")
    ax.plot(t, df["brier_blend"], color="green",        lw=1.4, label="Blended")
    ax.set_ylabel("Cumulative Brier score")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(t, df["w_poly"], color="steelblue",  lw=1.2, label="w(Polymarket)")
    ax.plot(t, df["w_bs"],   color="darkorange",  lw=1.2, label="w(Black-Scholes)")
    ax.axhline(0.5, color="black", lw=0.5, ls="--", label="Equal weights")
    ax.set_ylabel("Trust weight")
    ax.set_xlabel("Observation")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    print(f"  Chart saved to {CHART_FILE}")


def main() -> None:
    print("CrowdArb Layer 3 — Bitcoin cross-market calibration")
    print("─" * 54)
    print(f"  Live snapshot : Poly={P_POLY_LIVE:.1%}   BS={P_BS_LIVE:.1%}")
    print(f"  p_true={P_TRUE:.3f}   n={N_STEPS}   eta={ETA}   seed={BACKTEST_SEED}")
    print()

    print("Running 200-step calibration backtest (synthetic)...")
    p_poly_sim, p_bs_sim, outcomes_sim = generate_btc_pairs(
        n=N_STEPS, p_true=P_TRUE, seed=BACKTEST_SEED,
    )
    df = run_btc_calibration(p_poly_sim, p_bs_sim, outcomes_sim, eta=ETA)

    # Reconstruct terminal blender state to read final weights
    final_blender = BayesianBlender(source_names=["polymarket", "bs"], eta=ETA)
    for pp, pb, y in zip(p_poly_sim, p_bs_sim, outcomes_sim):
        final_blender.update([pp, pb], int(y))

    print_btc_summary(df, final_blender)
    df.to_csv(CSV_FILE, index=False)
    print(f"  Data saved to {CSV_FILE}")

    plot_btc_calibration(df)


if __name__ == "__main__":
    main()
