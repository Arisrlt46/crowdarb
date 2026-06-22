"""
CrowdArb Layer 3 — cross-market calibration layer.

Blends K probability sources (Polymarket, CME, optionally LLM) into a single
fair-value estimate using Bayesian model averaging with online weight learning.

Weight update (Hedge algorithm) after observing outcome y ∈ {0,1}:
  log wᵢ ← log wᵢ + η × [y·log(pᵢ) + (1−y)·log(1−pᵢ)]
Normalise: wᵢ = exp(log wᵢ) / Σ exp(log wⱼ)   [log-sum-exp for stability]
Blend:     p̂  = Σ wᵢ × pᵢ

Result 2: blended Brier score < raw Polymarket Brier score.
"""

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class SourceRecord:
    name: str
    log_weight: float = 0.0  # cumulative log-trust; positive = performed well historically


class BayesianBlender:
    """
    Online Bayesian blender using the Hedge (exponential weights) algorithm.

    Sources are identified by name; probabilities are passed in at blend/update time
    so the blender holds only trust state, not live prices.
    """

    def __init__(self, source_names: list[str], eta: float = 0.1):
        self.sources = [SourceRecord(name=n) for n in source_names]
        self.eta = eta  # learning rate: how fast weights respond to new outcomes

    @property
    def weights(self) -> np.ndarray:
        log_ws = np.array([s.log_weight for s in self.sources])
        log_ws -= log_ws.max()  # shift before exp for numerical stability
        ws = np.exp(log_ws)
        return ws / ws.sum()

    def blend(self, probabilities: list[float]) -> float:
        """Return the trust-weighted average of the per-source probabilities."""
        return float(np.dot(self.weights, probabilities))

    def update(self, probabilities: list[float], outcome: int) -> None:
        """Update source log-weights in-place after observing one binary outcome."""
        eps = 1e-7
        for s, p in zip(self.sources, probabilities):
            p_c = float(np.clip(p, eps, 1.0 - eps))
            log_likelihood = outcome * np.log(p_c) + (1 - outcome) * np.log(1.0 - p_c)
            s.log_weight += self.eta * log_likelihood

    def weight_dict(self) -> dict[str, float]:
        return {s.name: float(w) for s, w in zip(self.sources, self.weights)}


def generate_synthetic_pairs(
    n: int = 200,
    p_true: float = 0.30,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate (p_poly, p_cme) price paths and binary outcomes around a true probability.

    Polymarket: retail-biased upward, higher variance — models tendency to overprice tail events.
    CME:        slight downward bias, lower variance — models professional hedger conservatism.
    """
    rng = np.random.default_rng(seed)
    p_poly = np.clip(rng.normal(loc=p_true + 0.06, scale=0.08, size=n), 0.05, 0.95)
    p_cme  = np.clip(rng.normal(loc=p_true - 0.03, scale=0.04, size=n), 0.05, 0.95)
    outcomes = rng.binomial(1, p_true, size=n).astype(int)
    return p_poly, p_cme, outcomes


def run_calibration_backtest(
    p_poly: np.ndarray,
    p_cme: np.ndarray,
    outcomes: np.ndarray,
    eta: float = 0.1,
) -> pd.DataFrame:
    """
    Replay (p_poly, p_cme, outcome) triples through the blender.
    Records per-step blended estimate, trust weights, and cumulative Brier scores.
    """
    blender = BayesianBlender(source_names=["polymarket", "cme"], eta=eta)
    rows = []
    brier_sums = {"blend": 0.0, "poly": 0.0, "cme": 0.0}

    for i, (pp, pc, y) in enumerate(zip(p_poly, p_cme, outcomes)):
        p_blend = blender.blend([pp, pc])
        ws = blender.weights

        b_blend = (p_blend - y) ** 2
        b_poly  = (pp - y) ** 2
        b_cme   = (pc - y) ** 2
        brier_sums["blend"] += b_blend
        brier_sums["poly"]  += b_poly
        brier_sums["cme"]   += b_cme

        rows.append({
            "step":               i,
            "p_poly":             pp,
            "p_cme":              pc,
            "p_blend":            p_blend,
            "outcome":            y,
            "w_poly":             ws[0],
            "w_cme":              ws[1],
            "brier_blend":        brier_sums["blend"] / (i + 1),
            "brier_poly":         brier_sums["poly"]  / (i + 1),
            "brier_cme":          brier_sums["cme"]   / (i + 1),
        })

        blender.update([pp, pc], y)  # update after recording prediction

    return pd.DataFrame(rows)


def print_calibration_summary(df: pd.DataFrame, blender: BayesianBlender) -> None:
    W = 58
    final = df.iloc[-1]
    print()
    print("=" * W)
    print("  LAYER 3 — CROSS-MARKET CALIBRATION RESULTS")
    print("=" * W)
    print(f"  Steps simulated       : {len(df)}")
    print()
    print(f"  {'Source':<18}  {'Final Brier':>12}  {'Final weight':>12}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*12}")
    ws = blender.weight_dict()
    print(f"  {'Polymarket':<18}  {final['brier_poly']:>12.4f}  {ws['polymarket']:>12.4f}")
    print(f"  {'CME futures':<18}  {final['brier_cme']:>12.4f}  {ws['cme']:>12.4f}")
    print(f"  {'Blended (Layer 3)':<18}  {final['brier_blend']:>12.4f}  {'—':>12}")
    print()

    improvement = (final["brier_poly"] - final["brier_blend"]) / final["brier_poly"] * 100
    print(f"  Brier improvement vs Polymarket alone: {improvement:+.1f}%")
    print(f"  Final trust weights  : Polymarket {ws['polymarket']:.3f}  CME {ws['cme']:.3f}")
    print("=" * W)


def plot_calibration(df: pd.DataFrame, live_blend: float = None) -> None:
    """Three-panel chart: price paths, cumulative Brier scores, weight evolution."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("CrowdArb Layer 3 — Cross-Market Calibration", fontsize=13, fontweight="bold")

    t = df["step"]

    ax = axes[0]
    ax.plot(t, df["p_poly"],  color="steelblue",  lw=1,   label="Polymarket")
    ax.plot(t, df["p_cme"],   color="darkorange",  lw=1,   label="CME futures")
    ax.plot(t, df["p_blend"], color="green",        lw=1.4, label="Blended (Layer 3)")
    ax.scatter(t[df["outcome"] == 1], [0.03] * df["outcome"].sum(),
               marker="|", color="black", s=20, alpha=0.4, label="YES outcome")
    ax.set_ylabel("P(cut)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(t, df["brier_poly"],  color="steelblue",  lw=1.2, label="Polymarket")
    ax.plot(t, df["brier_cme"],   color="darkorange",  lw=1.2, label="CME futures")
    ax.plot(t, df["brier_blend"], color="green",        lw=1.4, label="Blended")
    ax.axhline(0.25, color="black", lw=0.5, ls="--", label="Coin-flip baseline (0.25)")
    ax.set_ylabel("Cumulative Brier score")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    ax.plot(t, df["w_poly"], color="steelblue",  lw=1.2, label="w(Polymarket)")
    ax.plot(t, df["w_cme"],  color="darkorange",  lw=1.2, label="w(CME)")
    ax.axhline(0.5, color="black", lw=0.5, ls="--", label="Equal weights")
    ax.set_ylabel("Trust weight")
    ax.set_xlabel("Observation")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("layer3_calibration.png", dpi=150)
    print("  Chart saved to layer3_calibration.png")


BACKTEST_SEED = 7   # fixed seed for generate_synthetic_pairs; keeps Brier results reproducible


def main():
    from layer0_compare import fetch_cme, fetch_polymarket
    from datetime import date

    # Live snapshot (display only — does not affect backtest metrics)
    p_blend_live = None
    try:
        print("Fetching live data...")
        _, p_poly_live = fetch_polymarket()
        today = date.today()
        _, _, _, _, p_cme_live = fetch_cme(today)
        blender_live = BayesianBlender(source_names=["polymarket", "cme"], eta=0.1)
        p_blend_live = blender_live.blend([p_poly_live, p_cme_live])
        print()
        print(f"  Polymarket P(any cut in 2026) : {p_poly_live:.4f}")
        print(f"  CME P(cut at next FOMC)       : {p_cme_live:.4f}")
        print(f"  Blended (equal weights)        : {p_blend_live:.4f}")
        print()
    except Exception as e:
        print(f"  Live fetch skipped: {e}\n")

    # Synthetic calibration backtest to demonstrate weight learning and Brier improvement
    print("Running 200-step calibration backtest (synthetic, p_true=0.30, seed=7)...")
    p_poly_sim, p_cme_sim, outcomes_sim = generate_synthetic_pairs(n=200, p_true=0.30, seed=BACKTEST_SEED)
    df = run_calibration_backtest(p_poly_sim, p_cme_sim, outcomes_sim, eta=0.1)

    # Reconstruct final blender state to read terminal weights
    final_blender = BayesianBlender(source_names=["polymarket", "cme"], eta=0.1)
    for pp, pc, y in zip(p_poly_sim, p_cme_sim, outcomes_sim):
        final_blender.update([pp, pc], int(y))

    print_calibration_summary(df, final_blender)
    df.to_csv("layer3_calibration.csv", index=False)
    print("  Data saved to layer3_calibration.csv")

    plot_calibration(df, live_blend=p_blend_live)


if __name__ == "__main__":
    main()
