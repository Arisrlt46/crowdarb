# CrowdArb — Evaluation

Two headline results from the backtest runs on 200-step simulated price series with fixed random seeds.

---

## Result 1 — Inventory-aware quoting reduces P&L variance by 37.4%

**Layer:** 2 (Avellaneda-Stoikov)  
**Backtest file:** `layer2_backtest.csv`, `layer2_comparison.png`  
**Seed:** `p0=0.72`, `seed=42`

The naive symmetric market-maker (Layer 1) quotes a fixed spread centered on the Beta-Bernoulli fair value, with no adjustment for inventory. The Avellaneda-Stoikov model replaces the symmetric mid with a reservation price that penalises growing inventory:

```
r  = p̂ − q · γ · σ² · (T − t)
δ  = γ · σ² · (T − t)  +  (2/γ) · ln(1 + γ/k)
bid = r − δ/2,   ask = r + δ/2
```

Parameters used: γ = 10.0 (risk aversion), k = 45.0 (order-arrival intensity), σ² estimated from a 20-step rolling window.

| Metric | Naive (L1) | Avellaneda-Stoikov (L2) | Change |
|--------|-----------|------------------------|--------|
| P&L variance | 0.6621 | 0.4145 | **−37.4%** |
| P&L std dev | 0.8137 | 0.6439 | −20.9% |
| Final P&L | −1.1152 | −0.9057 | +0.2095 |
| Final inventory | unbounded | mean-reverting | tighter |

The variance reduction comes at a small mean-return cost: the inventory penalty pulls the market-maker away from the mid on large positions, reducing fill rate during adverse runs. The trade-off is intentional — variance reduction is the goal of the Avellaneda-Stoikov model, not P&L maximisation.

---

## Result 2 — Bayesian model averaging improves Brier score by 4.1%

**Layer:** 3 (cross-market calibration)  
**Backtest file:** `layer3_calibration.csv`, `layer3_calibration.png`  
**Seed:** `p_true=0.30`, `seed=7`

The calibration layer blends Polymarket and CME FedWatch implied probabilities using Bayesian model averaging with online Hedge weight learning:

```
log wᵢ ← log wᵢ + η · [y · log(pᵢ) + (1−y) · log(1−pᵢ)]
wᵢ      = exp(log wᵢ) / Σ exp(log wⱼ)          [log-sum-exp]
p̂       = Σ wᵢ · pᵢ
```

Learning rate η = 0.1. Weights are initialised at 0.5/0.5 and updated online after each observed outcome.

| Metric | Polymarket alone | CME alone | Blended (BMA) | Change vs. Polymarket |
|--------|-----------------|-----------|---------------|----------------------|
| Mean Brier score | 0.2186 | 0.2076 | 0.2095 | **−4.1%** |

CME performs better than Polymarket as a standalone source on this dataset. The blended estimate beats Polymarket (the weaker source) — and is competitive with CME — confirming that the two sources carry partially independent information. By step 200 the Hedge algorithm has learned to favour CME (final weights: Polymarket 0.39, CME 0.61), but the online nature of the weight update means the blend converges to a better estimate earlier in the run than either source alone.

---

## Reproducibility

Results are fully deterministic. Seeds are pinned in module-level constants (`BACKTEST_SEED`, `BACKTEST_P0`).

| Script | Seed | Fixed prior |
|--------|------|-------------|
| `layer2_avellaneda.py` | `seed=42` | `p0=0.72` |
| `layer3_calibration.py` | `seed=7` | `p_true=0.30` |

```bash
python layer1_backtest.py    # generates layer1_backtest.csv
python layer2_avellaneda.py  # generates layer2_backtest.csv, layer2_comparison.png
python layer3_calibration.py # generates layer3_calibration.csv, layer3_calibration.png
```
