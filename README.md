# CrowdArb

**A cross-market probability calibration engine for binary prediction markets.**

CrowdArb pairs each Polymarket contract with a professional-market counterpart — CME Fed Funds Futures for rate decisions, Black-Scholes on Deribit implied volatility for crypto price levels — and blends the two probability estimates into a single fair value. That fair value feeds an Avellaneda-Stoikov inventory-aware market-maker and a Bayesian model-averaging calibrator that learns online which source to trust. The thesis: prediction-market crowds and professional markets frequently disagree on the same binary event, and that disagreement is measurable and exploitable. A live Streamlit scanner auto-discovers priceable contracts and ranks them by the size of the crowd-vs-professional gap.

- **Live app:** https://crowdarb.streamlit.app
- **Repository:** https://github.com/Arisrlt46/crowdarb

## Key documents

Full write-ups live in [`deliverables/`](deliverables/). Each is provided as a formatted PDF and a Markdown source.

| Document | PDF | Markdown |
|---|---|---|
| Overview — thesis, results, run instructions | [PDF](deliverables/CrowdArb_Overview.pdf) | [MD](deliverables/CrowdArb_Overview.md) |
| Technical Architecture — the six layers and the MarketPair interface | [PDF](deliverables/CrowdArb_Technical_Architecture.pdf) | [MD](deliverables/CrowdArb_Technical_Architecture.md) |
| Problems Encountered & Solutions — bugs, root causes, design decisions | [PDF](deliverables/CrowdArb_Problems_and_Solutions.pdf) | [MD](deliverables/CrowdArb_Problems_and_Solutions.md) |
| Demo video (≈2 min) | [MP4](deliverables/CrowdArb_Demo.mp4) | — |

**Presentation materials:**

| Material | File |
|---|---|
| Slide deck | [CrowdArb_Deck.pptx](CrowdArb_Deck.pptx) |
| Explainer video script (scene-by-scene, ~2 min) | [CrowdArb_Explainer_Script.md](CrowdArb_Explainer_Script.md) |
| Live-dashboard demo script (click-by-click) | [CrowdArb_Demo_Script.md](CrowdArb_Demo_Script.md) |

## Results

Tested on 200-step synthetic price series with fixed random seeds.

| Layer | Metric | Baseline | CrowdArb | Improvement |
|---|---|---|---|---|
| 2 — Avellaneda-Stoikov (Fed) | P&L variance | 0.6621 | 0.4145 | −37.4% |
| 2 — Avellaneda-Stoikov (Bitcoin) | P&L variance | 4.4339 | 1.7658 | −60.2% |
| 3 — Bayesian blending (Fed) | Brier score | 0.2186 | 0.2095 | −4.1% |
| 3 — Bayesian blending (Bitcoin) | Brier score | 0.0112 | 0.0101 | −9.6% |

Inventory-aware quoting reduces P&L variance by 37%–60% across markets. Bayesian model averaging improves calibration by up to 9.6% relative to using Polymarket alone.

## How it works

The engine is organised as six layers, each with a single responsibility, wired together through a common `MarketPair` interface. Adding a new market is one class of roughly 20 lines; the whole pipeline then works on it automatically.

| Layer | Role |
|---|---|
| 0 — Data | Extracts the implied probability of the same event from two independent sources (Polymarket + a professional market) |
| 1 — Belief | Beta-Bernoulli conjugate updating over simulated order flow |
| 2 — Market-making | Avellaneda-Stoikov inventory-aware reservation price and optimal spread |
| 3 — Calibration | Bayesian model averaging with online Hedge weight learning |
| 4 — News signal | Turns a news headline into a likelihood ratio via the Anthropic API and applies a Bayesian update |
| 5 — Interpretation | Converts the numerical output into plain-language readings and one-line scanner labels |

See [`deliverables/CrowdArb_Technical_Architecture.pdf`](deliverables/CrowdArb_Technical_Architecture.pdf) for the full breakdown.

## Quickstart

```
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY (required for Layer 4 only)

streamlit run dashboard_scanner.py
```

The dashboard auto-discovers markets on first load (≈15 s cold start), caches results for 5 minutes, and shows a ranked scanner table sorted by the crowd-vs-professional gap, with live A-S quotes and a headline scorer in the detail panel.

For per-market backtests and the full pipeline, see the run instructions in the [Overview](deliverables/CrowdArb_Overview.pdf).

## Requirements

Python 3.9+. Dependencies in `requirements.txt`. An `ANTHROPIC_API_KEY` is required only for Layer 4 (the LLM headline scorer).

## License

MIT
