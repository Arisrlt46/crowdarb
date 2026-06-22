# CrowdArb

CrowdArb is a cross-market probability calibration engine for binary prediction-market contracts. It reconciles prices from Polymarket, CME FedWatch implied probabilities, and live news signals into a single blended fair value, then makes markets using an Avellaneda-Stoikov inventory-control model. The core thesis is that prediction-market crowds and professional futures markets frequently disagree on the same binary event — and that disagreement is exploitable. Targeting Fed rate-decision contracts, where the Polymarket/CME spread is cleanest, the system delivered a **37.4% reduction in P&L variance** by switching from a naive symmetric spread to inventory-aware quoting, and a **4.1% Brier score improvement** by blending Polymarket and CME probabilities through Bayesian model averaging rather than using either source alone.

## Results

| Result | Metric | Naive baseline | CrowdArb | Improvement |
|--------|--------|---------------|----------|-------------|
| Inventory-aware quoting (Layer 2) | P&L variance | 0.6621 | 0.4145 | −37.4% |
| Cross-market calibration (Layer 3) | Brier score | 0.2186 (Polymarket alone) | 0.2095 (blended) | −4.1% |

## Layers

| Layer | File | What it does |
|-------|------|--------------|
| 0 | `layer0_*.py` | Data ingestion — fetches live Polymarket and CME FedWatch probabilities |
| 1 | `layer1_belief.py`, `layer1_backtest.py` | Beta-Bernoulli belief updater; naive symmetric market-maker; backtest harness |
| 2 | `layer2_avellaneda.py` | Avellaneda-Stoikov reservation price and optimal spread; inventory-aware quotes |
| 3 | `layer3_calibration.py` | Bayesian model averaging over Polymarket and CME; online Hedge weight learning |
| 4 | `layer4_llm_signal.py` | LLM signal layer — headline in, likelihood ratio out, belief updated |

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

**Run the Layer 1 backtest (naive market-maker):**
```bash
python layer1_backtest.py
```

**Run the Layer 2 backtest (Avellaneda-Stoikov vs. naive comparison):**
```bash
python layer2_avellaneda.py
```

**Run the Layer 3 calibration backtest (blended vs. Polymarket Brier score):**
```bash
python layer3_calibration.py
```

**Score a news headline (Layer 4 LLM signal):**
```bash
python layer4_llm_signal.py "Fed signals openness to rate cut amid cooling inflation"
```

Both backtests use fixed random seeds (Layer 2: `seed=42`, `p0=0.72`; Layer 3: `seed=7`) for reproducibility. Output charts are saved as PNG files alongside the CSVs.

## Requirements

Python 3.11+. See `requirements.txt`. An `ANTHROPIC_API_KEY` is required only for Layer 4.

## License

MIT
