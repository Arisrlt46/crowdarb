# CrowdArb — Roadmap

**Status: COMPLETE** — all layers shipped. Live at https://crowdarb.streamlit.app

## Overview

**PolyQuant** is a cross-market calibration engine for binary prediction-market contracts. It reconciles prices from three independent sources — a prediction market, the real financial instrument referencing the same event, and an LLM news signal — to produce a blended fair-value estimate, then makes markets using an Avellaneda-Stoikov inventory-control model.

**Core thesis:** Prediction-market crowds and professional financial markets frequently disagree on the same probability (e.g. a Fed rate-cut contract on Polymarket vs. CME FedWatch implied probabilities). The cross-market calibration layer exploits this gap.

**Target event:** Fed rate decisions — cleanest overlap between Polymarket contracts and CME Fed funds futures.

**Scope constraints:**
- Binary contracts only (prices in [0, 1])
- Paper trading only — no real funds
- One event type end-to-end before expanding

**Tech stack:** Python (numpy, pandas, scipy, matplotlib), Anthropic SDK, Streamlit, Polymarket Gamma/CLOB APIs, CME FedWatch data, GitHub.

**Two headline results:**
1. Inventory-aware Avellaneda-Stoikov strategy achieves lower P&L variance and tighter final inventory than a naive symmetric strategy, at a small mean-return cost.
2. The cross-market blended fair value has lower Brier score / log-loss than the raw Polymarket price alone.

---

## Phase 0 — Setup

- [x] Python 3.11+, Git, VS Code installed and verified
- [x] GitHub account and `polyquant` repo created (public, MIT license, Python .gitignore)
- [x] Repo cloned locally; first commit + push completed
- [x] Virtual environment created; `requirements.txt` in place
- [x] API keys (Polymarket, Anthropic) stored in `.env`; `.env` confirmed in `.gitignore`
- [x] This roadmap committed to the repo

---

## Layer 0 — Data Ingestion

Prove all data sources return usable numbers before writing any model code.

- [x] Fetch live Fed rate-decision contract probability from Polymarket Gamma API
- [x] Fetch matching implied probability from CME FedWatch (or Fed funds futures)
- [x] Print both side-by-side; quantify the spread
- [x] Save a short historical series of both to CSV

---

## Layer 1 — Bayesian Belief Model + Naive Market-Maker

Minimal end-to-end pipeline: belief update → quotes → backtest. Demoable on its own.

- [x] Beta-Bernoulli belief updater: prior → evidence update → posterior mean
- [x] Naive symmetric market-maker: fixed spread centered on fair value
- [x] Backtest harness: replay historical prices, simulate fills, track P&L and inventory
- [x] P&L curve and inventory chart
- [x] Calibration metrics: Brier score and log-loss on the belief estimates

---

## Layer 2 — Avellaneda-Stoikov Pricing Core

Replace the naive spread with the inventory-aware optimal spread.

- [x] Reservation price: `r = p̂ - q·γ·σ²·(T - t)`
- [x] Optimal spread: inventory-risk component + trade-frequency component
- [x] Adapt model for bounded [0, 1] prices via logit transform
- [x] Run inventory-aware strategy through the backtest harness
- [x] **Result 1:** P&L variance and final inventory — inventory-aware vs. naive
- [x] Derivation write-up in `docs/avellaneda_stoikov.md`

---

## Layer 3 — Cross-Market Calibration Layer

The original contribution. Blend multiple probability sources into a single calibrated estimate.

- [x] Bayesian model averaging over prediction-market price, real-instrument implied probability, and (optionally) LLM signal
- [x] Trust weights per source; optional online weight learning from historical calibration track record
- [x] **Result 2:** Brier score — blended fair value vs. raw Polymarket price
- [x] Derivation write-up in `docs/calibration_layer.md`

---

## Layer 4 — LLM Signal (optional)

Adds the third probability source and live news reactivity. Build only after Layers 1–3 are solid.

- [x] Anthropic API call: headline in → likelihood-ratio update + written justification out
- [x] Structured output via tool use (not free text)
- [x] Likelihood-ratio fed into the calibration layer as the third source
- [x] Output clamping and decision logging

---

## Layer 5 — Streamlit Dashboard (optional)

Live paper-trading demo. Build after the math layers are complete.

- [x] Display: three source probabilities, blended fair value, current bid/ask, inventory, P&L
- [x] Real-time gap chart: prediction market vs. real instrument
- [x] Headline input box → fair value and quote reaction
- [x] Read-only live data feeds
- [x] Calibration plot

---

## Layer 6 — Polish

- [x] README: pitch, screenshot, one-command reproduction
- [x] `docs/EVALUATION.md`: both headline results with metrics
- [x] Reproducible backtest (fixed random seed, single entry point)
- [x] 60-second demo video or GIF
- [x] 3-minute pitch script
