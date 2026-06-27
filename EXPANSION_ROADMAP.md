# CrowdArb — Expansion Roadmap: Multi-Market Generalization

**Status: COMPLETE** — all phases A–G shipped. Live at https://crowdarb.streamlit.app

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

- [x] Confirm the Gamma API's `tag_slug` / `category` fields and how to filter
      by them reliably (e.g. `tag_slug=finance`, `tag_slug=stocks`)
- [x] Build a general-purpose `find_markets(category, keyword)` function that
      paginates through a category and filters client-side on question text
- [x] Replace the ad-hoc keyword search in layer0_data.py with this function
- [x] Verify it correctly finds the known S&P 500 (SPX) Indices contracts

## Phase B — Add a second market type: S&P 500 / equity indices

This reuses the Black-Scholes work already built in layer0_sp500.py.

- [x] Use the Phase A discovery function to pull live SPX/S&P 500 Polymarket
      contracts (strike, expiry, current price)
- [x] Match each Polymarket contract's strike/date to a Black-Scholes implied
      probability call (already implemented)
- [x] Build layer0_sp500_compare.py: prints Polymarket vs Black-Scholes side
      by side, same pattern as the Fed comparison
- [x] Log historical snapshots the same way as layer0_history.csv
- [x] Feed both probabilities into the existing Layer 1-3 machinery and
      confirm the belief updater, calibration, and backtest all work
      unmodified on this new market

## Phase C — Generalize the market interface

Once two market types work, formalize the pattern so adding a third is fast.

- [x] Define a common `MarketPair` interface: each market type provides
      (a) a Polymarket-side probability fetcher, (b) a "real instrument"
      probability fetcher, (c) metadata (event name, resolution date)
- [x] Refactor layer0_data.py (Fed) and layer0_sp500.py (equities) to both
      implement this interface
- [x] Build a registry/config (e.g. a Python dict or small JSON file) listing
      every supported market pair, so adding one is a config entry, not new
      plumbing
- [x] Layer 3's calibration logic should accept any MarketPair, not just Fed

## Phase D — Add a third market type (proves the interface generalizes)

Pick one, in order of ease:

- [x] Individual stock price/earnings contracts (Black-Scholes already
      reusable, just swap the underlying ticker)
- [x] Bitcoin price levels (Polymarket has these; needs CME BTC futures or
      Deribit options for the "real market" side)
- [x] Inflation / macro contracts (would need TIPS breakevens — new formula
      work, save for later)

## Phase E — Multi-market dashboard

Extend Layer 5 (Streamlit) to support market selection rather than being
hardcoded to Fed rates.

- [x] Dashboard has a dropdown/selector for which MarketPair to display
- [x] Each market shows the same panel layout: live probabilities, blended
      estimate, A-S quotes, calibration history
- [x] A summary table across all tracked markets, ranked by gap size between
      Polymarket and the real-instrument probability — this is the actual
      "arbitrage scanner" view

## Phase F — Polish for the arbitrage-site pitch

- [x] README update: reframe the pitch from "a Fed rate tool" to "a
      cross-market calibration framework, proven on Fed rates and equities"
- [x] Document the MarketPair interface so it reads as a platform, not a
      one-off script
- [x] Update resume description once Phase B is verified and reproducible

## Phase G — Real-time auto-discovery

**Goal:** replace the three hardcoded `MarketPair` classes in the scanner with a
`discover_markets()` call that paginates Polymarket live, classifies every active
contract, and constructs a `MarketPair` for each one that has a computable
professional-market counterpart. The scanner table then populates dynamically —
no code change needed to add a new crypto strike or rate meeting.

### G1 — Contract classifier

Build `classify_market(market: dict) -> str` in a new `layer0_classifier.py`.

Input: a raw Gamma API market dict.  
Output: one of `"crypto_level"`, `"rate_decision"`, `"equity_level"`, `"unsupported"`.

Rules (applied in order; first match wins):

| Tag | Condition |
|---|---|
| `crypto_level` | Question contains a crypto ticker (BTC, ETH, SOL, …) **and** a parseable dollar strike **and** a reach/hit/above verb |
| `rate_decision` | Question contains a rate-decision keyword (fed, fomc, ecb, boe, rate cut, basis points) |
| `equity_level` | Question contains an equity index keyword (s&p, spx, nasdaq, dow, nikkei) **and** a parseable dollar strike |
| `unsupported` | Anything that does not match the above |

- Export a `SUPPORTED_TAGS = {"crypto_level", "rate_decision", "equity_level"}` set
  so callers can filter without re-implementing the logic.
- Keep the classifier pure (no network I/O) so it can be unit-tested in isolation.
- Log the tag distribution across one full page scan as a sanity check.

### G2 — Unified strike/date parser

Extract `parse_strike(question: str) -> float | None` and
`parse_end_date(market: dict) -> date | None` into `layer0_classifier.py`
(or a small `layer0_parse.py` if the file grows large).

- Consolidate the three copies currently living in `layer0_bitcoin.py`,
  `layer0_ethereum.py`, and `layer0_sp500_compare.py` into one shared
  implementation.
- `parse_strike` must handle: `$150k`, `$150,000`, `$1m`, `6500`, `6,500`
  (no dollar sign for equity index levels).
- `parse_end_date` reads `endDate` then `endDateIso`; returns `None` if neither
  parses cleanly.
- Refactor existing callers to import from the new shared location.

### G3 — `discover_markets()` function

Build `discover_markets(min_volume_usd=50_000, min_days=90, max_pages=30)`
in `layer0_classifier.py`.

Steps:
1. Paginate `/markets?active=true&closed=false` via `_iter_active_markets()`
   (already in `layer0_markets.py` — import it, don't duplicate).
2. For each market dict, call `classify_market()`.
3. Skip `"unsupported"` immediately.
4. Apply minimum filters:
   - `float(market.get("volume") or 0) >= min_volume_usd`
   - `parse_end_date(market)` is not None and days remaining ≥ `min_days`
   - `parse_strike(market["question"])` is not None (for level contracts)
5. Instantiate the right `MarketPair` subclass:
   - `crypto_level` → `BitcoinMarket` or `EthereumMarket` based on ticker keyword,
     or a new generic `CryptoLevelMarket(ticker, question, market_dict)` that
     accepts the contract and underlying ticker as constructor args (avoids
     re-discovery inside `_ensure_loaded`).
   - `rate_decision` → `FedRateMarket` (or a future `ECBMarket` once that's built).
   - `equity_level` → `SP500Market` (Phase D equivalent for equities).
6. Return `list[MarketPair]` sorted by volume descending.
7. De-duplicate: if the same underlying event appears in multiple contracts
   (e.g. BTC > $100k and BTC > $150k both pass), keep only the highest-volume
   one per (classifier_tag, underlying_ticker) pair to avoid redundant rows in
   the scanner.

### G4 — Generic `CryptoLevelMarket`

The current `BitcoinMarket` and `EthereumMarket` classes are structurally
identical except for the ticker and Deribit currency string. Merge them into
`CryptoLevelMarket(MarketPair)` in `market_pair.py`:

```python
class CryptoLevelMarket(MarketPair):
    def __init__(self, yf_ticker: str, deribit_currency: str,
                 question: str, market_dict: dict) -> None: ...
```

- `yf_ticker`: Yahoo Finance ticker for spot price, e.g. `"BTC-USD"`, `"ETH-USD"`.
- `deribit_currency`: Deribit DVOL currency string, e.g. `"BTC"`, `"ETH"`.
- `question` and `market_dict`: pre-fetched from `discover_markets()`, so
  `_ensure_loaded` does not need to call `find_markets()` again.
- Historical vol fallback uses `yf_ticker` directly (no asset-specific function
  needed — all crypto trades 24/7, so annualisation is always ×√365).
- Keep `BitcoinMarket` and `EthereumMarket` as thin subclasses or aliases
  for backwards compatibility with the existing tests and CLI scripts.

### G5 — Wire into the scanner dashboard

In `dashboard_scanner.py`, replace the three hardcoded loader calls with:

```python
@st.cache_data(ttl=300)   # 5-min TTL; discovery is slow (~10 s full scan)
def discover_all():
    return discover_markets(min_volume_usd=50_000, min_days=90)
```

- Render the scanner table from the returned list, same column schema as today.
- Show a "Discovered N markets" caption below the table.
- Keep a "Manual override" expander that lets the user type a Polymarket
  contract slug and force-add it to the table (useful for testing new types).
- The Refresh button should call `discover_all.clear()` in addition to the
  individual market cache clears.

### G6 — Caching strategy

| Layer | Cache mechanism | TTL | Rationale |
|---|---|---|---|
| `discover_markets()` | `@st.cache_data` | 300 s | Full scan is ~10 s; 5-min staleness acceptable |
| Per-market spot/vol | `@st.cache_data` on each `_ensure_loaded` equivalent | 60 s | Prices move; quotes should be near-live |
| `classify_market()` | No cache needed | — | Pure function; microseconds per call |
| `parse_strike/date` | No cache needed | — | Pure function; microseconds per call |
| Hedge trust weights | `@st.cache_data` (keyed on `(p_poly, p_prof)`) | No TTL | Deterministic given inputs; cache forever |

### G7 — Acceptance criteria

- [x] `classify_market()` correctly tags at least 20 manually verified contracts
      from a live scan (document the test cases in a docstring or small test file).
- [x] `discover_markets()` returns ≥ 3 supported contracts on a live run without
      any hardcoded market names in the scanner.
- [x] The scanner table populates end-to-end from `discover_markets()` with no
      manual intervention.
- [x] Adding a new crypto asset (e.g. SOL price-level contracts) requires only
      adding `"sol"` to the keyword list in the classifier — no new class,
      no new dashboard loader function.
- [x] The existing `BitcoinMarket` / `EthereumMarket` / `FedRateMarket` classes
      still work as before (backwards compatibility).