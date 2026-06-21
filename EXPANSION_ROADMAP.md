# CrowdArb — Expansion Roadmap: Multi-Market Generalization

## Where things stand

The core framework (Layers 1-4) is already market-agnostic — the Bayesian belief
updater, Avellaneda-Stoikov pricing, cross-market calibration, and LLM signal
layer all operate on a generic probability in [0,1]. They do not need to change.

What's market-specific is Layer 0: fetching a Polymarket contract and its
"professional market" counterpart, and computing an implied probability from
that counterpart. Generalizing means repeating Layer 0's work for each new
market type, then plugging the result into the same Layers 1-4.

Proven so far: Fed rate decisions (Polymarket vs CME Fed funds futures).

## Phase A — Fix market discovery

Polymarket's Gamma API search parameter does not reliably filter results.
Before adding new market types, fix how markets are found at all.

- [ ] Confirm the Gamma API's `tag_slug` / `category` fields and how to filter
      by them reliably (e.g. `tag_slug=finance`, `tag_slug=stocks`)
- [ ] Build a general-purpose `find_markets(category, keyword)` function that
      paginates through a category and filters client-side on question text
- [ ] Replace the ad-hoc keyword search in layer0_data.py with this function
- [ ] Verify it correctly finds the known S&P 500 (SPX) Indices contracts

## Phase B — Add a second market type: S&P 500 / equity indices

This reuses the Black-Scholes work already built in layer0_sp500.py.

- [ ] Use the Phase A discovery function to pull live SPX/S&P 500 Polymarket
      contracts (strike, expiry, current price)
- [ ] Match each Polymarket contract's strike/date to a Black-Scholes implied
      probability call (already implemented)
- [ ] Build layer0_sp500_compare.py: prints Polymarket vs Black-Scholes side
      by side, same pattern as the Fed comparison
- [ ] Log historical snapshots the same way as layer0_history.csv
- [ ] Feed both probabilities into the existing Layer 1-3 machinery and
      confirm the belief updater, calibration, and backtest all work
      unmodified on this new market

## Phase C — Generalize the market interface

Once two market types work, formalize the pattern so adding a third is fast.

- [ ] Define a common `MarketPair` interface: each market type provides
      (a) a Polymarket-side probability fetcher, (b) a "real instrument"
      probability fetcher, (c) metadata (event name, resolution date)
- [ ] Refactor layer0_data.py (Fed) and layer0_sp500.py (equities) to both
      implement this interface
- [ ] Build a registry/config (e.g. a Python dict or small JSON file) listing
      every supported market pair, so adding one is a config entry, not new
      plumbing
- [ ] Layer 3's calibration logic should accept any MarketPair, not just Fed

## Phase D — Add a third market type (proves the interface generalizes)

Pick one, in order of ease:

- [ ] Individual stock price/earnings contracts (Black-Scholes already
      reusable, just swap the underlying ticker)
- [ ] Bitcoin price levels (Polymarket has these; needs CME BTC futures or
      Deribit options for the "real market" side)
- [ ] Inflation / macro contracts (would need TIPS breakevens — new formula
      work, save for later)

## Phase E — Multi-market dashboard

Extend Layer 5 (Streamlit) to support market selection rather than being
hardcoded to Fed rates.

- [ ] Dashboard has a dropdown/selector for which MarketPair to display
- [ ] Each market shows the same panel layout: live probabilities, blended
      estimate, A-S quotes, calibration history
- [ ] A summary table across all tracked markets, ranked by gap size between
      Polymarket and the real-instrument probability — this is the actual
      "arbitrage scanner" view

## Phase F — Polish for the arbitrage-site pitch

- [ ] README update: reframe the pitch from "a Fed rate tool" to "a
      cross-market calibration framework, proven on Fed rates and equities"
- [ ] Document the MarketPair interface so it reads as a platform, not a
      one-off script
- [ ] Update resume description once Phase B is verified and reproducible