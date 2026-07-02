# CrowdArb — Changelog

## Layer 5 — Interpretation Engine

**Date:** July 2026

The pipeline previously produced correct numbers with no explanation of what they meant.
Layer 5 closes that gap: it reads the output of Layers 0–4 and generates a plain-language
reading of every figure, so a non-expert looking at the dashboard immediately understands
what the gap means, which source the model trusts and why, and what the market-making
model is doing.

---

### New file: `layer5_interpret.py`

**`MarketSignals`** — dataclass holding everything known about one market after Layers 0–4
have run: both source probabilities, the blended value, learned Hedge weights, A-S quotes,
and any LLM signal from the headline scorer.

**`Interpretation`** — dataclass returned by the engine: a one-line `verdict`, a
`direction` (`crowd_high` / `crowd_low` / `aligned`), a `magnitude`
(`aligned` / `modest` / `notable` / `large`), a list of per-fact plain-language `lines`,
and the full `glossary`.

**`GLOSSARY`** — 10 jargon-free definitions: Polymarket probability, Professional
probability, Gap, Blended, Trusted (Hedge), Bid, Ask, Reservation price, Spread,
Likelihood ratio.

**`interpret(sig) → Interpretation`** — pure, deterministic engine. Classifies the
absolute gap against three thresholds (3 / 8 / 15 percentage points → aligned / modest /
notable / large), sets direction from the sign, and builds one plain sentence per fact
that is actually present in the signals bundle: source probabilities, weighted blend,
reservation-price tilt, spread, and latest LLM likelihood ratio. No API calls. Never
raises on valid inputs.

**`compact_read(sig) → str`** — single-line label for the scanner table, e.g.
`"Crowd 21 pts high"` / `"Aligned"`.

**`narrate(sig, interp, client) → str`** — optional LLM paragraph via
`client.messages.parse(..., output_format=NarrativeOut)`. Reuses `layer4_llm_signal`'s
model and client pattern. System prompt instructs plain language, no jargon, no invented
facts. Only called on explicit user action; cached per market in session state.

**`format_interpretation(interp) → str`** — CLI formatter with `─` borders, used by
`run_crowdarb()` in `market_pair.py`.

---

### Changes: `dashboard_scanner.py`

- **`_hedge_weights(p_poly, p_prof) → (w_poly, w_prof)`** — new cached helper that runs
  the same 200-step Hedge simulation as before but returns both weights rather than just
  the winning name. `_hedge_trusted` is now derived from it, so no duplicate simulation.

- **Scanner table** — new `"Read"` column with a `compact_read()` label per row
  (`"Crowd 21 pts high"`, `"Aligned"`, etc.).

- **`st.metric` tooltips** — all eight metric calls (Polymarket, Professional, Gap,
  Blended, Bid, Ask, Spread, Reservation price, Likelihood ratio) now carry a `help=`
  argument populated from `GLOSSARY`.

- **"What these numbers mean" section** — appears after the A-S quotes block for the
  selected market. Builds a `MarketSignals` from values already computed on the page,
  calls `interpret()`, and renders:
  - Verdict in `st.success` (aligned) or `st.info` (diverged).
  - Bulleted plain-language lines, one per fact.
  - "Explain in plain English" button that calls `narrate()` once per market and caches
    the result in session state.

- **Glossary expander** — appears both at the top of the page (below the scanner table)
  and inline in the Market Detail panel.

- **Page header** — now shows the live app URL and GitHub link as a caption.

---

### Changes: `market_pair.py`

`run_crowdarb()` now calls `format_interpretation(interpret(sig))` at the end of each
pipeline run. The `MarketSignals` bundle is assembled from the values already computed in
the function body (p\_poly, p\_prof, learned Hedge weights, A-S reservation price and
spread).

---

### Design decisions

- The deterministic engine is the default and has no dependencies beyond the existing
  project stack.  It works with no API key.
- The LLM narrative is opt-in, one call per explicit click, and respects the existing
  $2/month API spend posture.
- `layer5_interpret.py` carries a self-test (`python layer5_interpret.py`) that exercises
  three synthetic cases (aligned; crowd-high-large; notable with LR) and requires no API
  key.
