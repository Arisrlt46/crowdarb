# CrowdArb — Research Findings Report

**Author:** Aris Rault
**Live system:** https://crowdarb.streamlit.app
**Repository:** https://github.com/Arisrlt46/crowdarb

---

## Abstract

CrowdArb reconciles prediction-market prices (Polymarket) with professional-market implied probabilities (CME Fed funds futures; Bitcoin and Ethereum options via Black-Scholes) across three live markets. This report documents two headline quantitative results and two structural findings, all reproducible from the repository with fixed random seeds.

---

## 1. Methodology

### 1.1 Probability extraction

Two extraction methods are used depending on instrument type.

**Linear instruments (Fed funds futures).** A Fed funds futures contract prices the average overnight rate across a calendar month. With a known pre-meeting rate, the post-meeting rate is recovered algebraically, and the implied probability of a 25 bp cut is the rate change divided by 0.25. Near month-end meetings, the next-month contract is used to avoid a day-count instability (see Finding 2).

**Non-linear instruments (crypto options).** The risk-neutral probability that an asset finishes above a strike is N(d2), where d2 = [ln(S/K) + (r − q − σ²/2)T] / (σ√T). Volatility is taken from Deribit's implied-volatility index where available, with a 90-day realized-volatility fallback annualized with √365 to reflect 24/7 crypto trading.

### 1.2 Calibration via Bayesian model averaging

The two probability sources are blended with weights learned online by the Hedge algorithm. After each resolved observation, each source's log-weight is updated by its log-likelihood, then renormalized via the log-sum-exp transform. Sources that predict well gain weight; poor predictors lose it.

### 1.3 Market-making via Avellaneda-Stoikov

Quotes are centered on an inventory-adjusted reservation price r = p − q·γ·σ²·(T−t) with a two-component optimal spread. Backtests use fixed seeds and a fixed initial price to ensure reproducibility.

---

## 2. Headline Result 1 — Inventory-aware quoting reduces P&L variance

Avellaneda-Stoikov quoting was compared against a naive fixed-spread market-maker across all three markets, measuring the variance of profit-and-loss over a seeded backtest.

| Market | P&L variance reduction vs naive |
|---|---|
| Fed rate cut | 37% |
| Bitcoin price level | 60% |
| Ethereum price level | 63% |

**Interpretation.** The benefit grows as the contract probability approaches the tails. At extreme probabilities, order flow is nearly one-sided (almost all fills are buys), so inventory accumulates monotonically; the inventory-control term in the reservation price is precisely what arrests that accumulation. The model earns its keep most where naive quoting fails worst.

---

## 3. Headline Result 2 — Blending improves calibration

Calibration was measured by Brier score (mean squared error of probability forecasts; lower is better). The Hedge-blended estimate was compared against the best single source.

| Market | Brier improvement (blended vs best single source) |
|---|---|
| Fed rate cut | ~4% |
| Bitcoin price level | up to 9.6% |

**Interpretation.** The weighting direction is data-driven, not hardcoded. In near-50% markets (Fed) the blend stayed balanced; in tail markets (crypto) it shifted majority weight to the better-calibrated professional source. The same algorithm adapts its trust to the market regime automatically.

---

## 4. Finding 1 — Bitcoin's crowd prices fat-tail risk the model ignores

On "will Bitcoin reach $150k," Black-Scholes implied roughly 0.2% while Polymarket priced roughly 4.5% — a ~22x divergence.

Black-Scholes assumes log-normal returns, under which a 2.4x move over the horizon is many standard deviations away and therefore near-impossible. Bitcoin's empirical return distribution has fat tails: extreme moves occur far more frequently than log-normal predicts. The crowd, conditioned on Bitcoin's history of violent moves, prices in that tail; the model structurally cannot. The divergence is therefore not noise but a measurable signature of a model assumption failing on this asset class.

**Caveat for presentation.** The exact magnitude depends on the volatility input and strike chosen; the qualitative finding (crowd > model on upside crypto tails) is robust, the precise multiple is not.

---

## 5. Finding 2 — A day-count instability in the naive FedWatch formula

When an FOMC meeting falls near month-end, the current-month futures contract contains very few post-meeting days. Recovering the post-meeting rate then divides by that small day count, amplifying input noise by a factor of N/days_after (≈15.5x for a day-30 meeting in a 31-day month). This produced an implied 1.74 rate cuts — clamped to a misleading 100%.

The fix follows CME's published methodology: when fewer than five post-meeting days remain, read the implied rate from the next-month contract, which provides a full post-meeting window and a stable estimate. Post-fix, the Fed professional probability reads a sensible value and reveals the project's clearest crowd-vs-professional gap (~20 points).

---

## 6. Reproducibility

All results are reproducible from the repository. Backtests use fixed seeds (e.g. seed = 42) and a fixed initial price to prevent live data leaking into controlled comparisons. A separate log (PROBLEMS_AND_SOLUTIONS.md) documents the bugs found and fixed during development, including the reproducibility fix itself.

---

## 7. Limitations and honest scope

- The professional probability is only as good as its volatility input; crypto results inherit Deribit/realized-vol assumptions.
- The scanner surfaces only markets with a computable professional counterpart — currently Fed rates and BTC/ETH price levels. Roughly 95% of Polymarket markets (politics, sports, entertainment) have no financial-market analogue and are correctly excluded.
- The set of markets shown depends on a tunable Black-Scholes sanity filter (currently a 0.5%–98% band). This is a design parameter, not a property of the data, and is documented as such.
- Backtests are on synthetic price paths around the blended prior; they demonstrate the machinery's behavior, not live trading P&L.
