# CrowdArb — Technical Architecture

**Live system:** https://crowdarb.streamlit.app
**Repository:** https://github.com/Arisrlt46/crowdarb

A layered cross-market calibration engine. Each layer has a single responsibility and
feeds the next. Adding a new market is a small, isolated change thanks to the MarketPair
interface.

---

## System overview

```
Polymarket (crowd)  ─┐
                     ├─► Layer 0: extract probabilities ─► Layer 1: Bayesian belief
Professional market ─┘                                              │
(futures / options)                                                 ▼
                                            Layer 3: blend (Hedge weights) ◄─ both sources
                                                                    │
                                                                    ▼
                                            Layer 2: Avellaneda-Stoikov quotes
                                                                    │
                                            Layer 4: LLM news signal ─┘
                                                                    │
                                                                    ▼
                                            Layer 5: plain-language interpretation
                                                                    │
                                                                    ▼
                                            Scanner: auto-discover + rank by gap
```

---

## Layer 0 — Probability extraction

Pulls the implied probability of the same event from two independent sources.

- **Fed rates:** algebra on CME Fed funds (ZQ) futures. Recovers the post-meeting rate
  from the month-average rate; converts to P(cut). Uses the next-month contract near
  month-end for numerical stability.
- **Crypto:** Black-Scholes N(d₂) on the asset's price level, using Deribit implied
  volatility (90-day realized fallback).
- Shared `layer0_bs.py` holds the Black-Scholes functions so crypto markets reuse one
  tested implementation.

## Layer 1 — Bayesian belief updating

A Beta-Bernoulli conjugate model maintains a running probability as simulated order flow
arrives. YES fills increment α, NO fills increment β; the posterior mean α/(α+β) is the
current belief, and α+β is its confidence.

## Layer 2 — Avellaneda-Stoikov market-making

Computes an inventory-adjusted reservation price and an optimal two-component spread,
then quotes bid/ask. Shifts quotes against inventory to control one-sided risk.
Headline result: 37–63% P&L variance reduction vs naive quoting.

## Layer 3 — Cross-market calibration

Bayesian model averaging with online Hedge weight learning blends the two probability
sources, automatically up-weighting whichever has been better calibrated. Headline
result: up to 9.6% Brier improvement vs the best single source.

## Layer 4 — LLM news signal

Converts a news headline into a likelihood ratio via the Anthropic API, then applies a
Bayesian odds update. Uses a Fed-specific prompt for rate markets and a crypto-specific
prompt for digital assets, selected by market type.

## Layer 5 — Interpretation engine

Reads the output of Layers 0–4 and produces plain-language explanations of every figure.

Three entry points:

- **`interpret(sig) → Interpretation`** — pure, deterministic, no API calls.
  Classifies the crowd-vs-professional gap against three thresholds (3 / 8 / 15 pp →
  aligned / modest / notable / large), sets direction, and builds one sentence per fact
  present in the signals bundle: source probabilities, weighted blend, reservation-price
  tilt vs mid, spread, and any LLM likelihood ratio.
- **`compact_read(sig) → str`** — one-line scanner label, e.g. `"Crowd 21 pts high"`.
- **`narrate(sig, interp, client) → str`** — optional LLM paragraph (2–3 sentences,
  plain English, no jargon) for explicit user requests. One API call per click; cached
  in session state.

`GLOSSARY` exports 10 jargon-free definitions wired into `st.metric(help=...)` tooltips
and a dashboard expander.

## Scanner — real-time auto-discovery

`discover_markets()` scans ~2,100 live Polymarket markets, classifies each
(crypto level / rate decision / equity level / unsupported), parses strike and date,
filters by volume, expiry, and a Black-Scholes sanity band, builds a MarketPair for each
survivor, and ranks them by crowd-vs-professional gap. The dashboard renders the ranked
table (with a `"Read"` column from Layer 5), metric tooltips, and a per-market detail
panel with a full Layer 5 interpretation.

---

## The MarketPair interface

The architectural core. Every market implements three methods:

- `get_polymarket_probability() -> float`
- `get_professional_probability() -> float`
- `metadata() -> dict` (name, description, resolution date, type)

`run_crowdarb(market)` runs Layers 1–5 on any MarketPair with no market-specific logic
in the pipeline body. Adding a market is one class of roughly 20 lines; the entire engine
then works on it automatically.

---

## Key files

| File | Role |
|---|---|
| `layer0_data.py`, `layer0_fedwatch.py` | Fed: Polymarket + CME extraction |
| `layer0_bs.py` | Shared Black-Scholes functions |
| `layer0_bitcoin.py`, `layer0_ethereum.py` | Crypto extraction |
| `layer0_classifier.py` | Market classification + strike/date parsing |
| `layer1_belief.py`, `layer1_backtest.py` | Bayesian belief + backtest |
| `layer2_avellaneda.py` | Market-making |
| `layer3_calibration.py` | Hedge blending |
| `layer4_llm_signal.py` | LLM news signal (per-market prompts) |
| `layer5_interpret.py` | Interpretation engine + GLOSSARY |
| `market_pair.py` | MarketPair interface + `discover_markets()` |
| `dashboard_scanner.py` | Streamlit scanner (deployed app) |

---

## Reproducibility and engineering notes

- Backtests use fixed seeds and a fixed initial price.
- `PROBLEMS_AND_SOLUTIONS.md` logs six bugs and one design decision with root-cause
  analysis.
- `CHANGELOG.md` documents the Layer 5 addition.
- Deployed on Streamlit Community Cloud; the Anthropic API key is stored in Streamlit
  secrets, never committed.
- A monthly API spend cap and disabled auto-reload bound the cost of the public LLM
  feature. The Layer 5 narrative call is opt-in and cached per market per session.
