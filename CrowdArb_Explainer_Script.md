# CrowdArb — 2-Minute Explainer Video Script

**Format:** screen capture + voiceover, dark quant-terminal aesthetic (dark background, monospace accents, subtle grid). Add an ambient music bed in iMovie under the narration (keep it low, ~ −18 dB).
**Total runtime:** ~120 s. Word counts are tuned for a calm ~150 wpm delivery; trim if you speak faster.

> Note: I can't generate the actual video or audio — this is a shot-by-shot script for you to record yourself (screen capture + your voice, or TTS) and score in iMovie.

---

### Scene 1 — Hook (0:00–0:12)

**On screen:** Black title card. `CrowdArb` types out in monospace, cursor blinking. Underneath, two numbers fade in side by side: `Polymarket: 20%` and `Professional: 0%`.

**Narration:**
"Two markets, looking at the exact same event, can disagree completely. Here, a prediction-market crowd says twenty percent — the professional futures market says zero. CrowdArb is built to find, measure, and act on that gap."

---

### Scene 2 — The idea (0:12–0:28)

**On screen:** Split panel. Left: Polymarket logo / a crowd contract. Right: CME futures + a Deribit options chain. An arrow from each meets in the middle at a single value labelled `fair value`.

**Narration:**
"For every binary contract, we pair the crowd's price with a professional counterpart — Fed Funds futures for rate decisions, Black-Scholes on crypto options for price levels — and blend them into one calibrated fair value."

---

### Scene 3 — The pipeline (0:28–0:42)

**On screen:** The six-layer stack draws in top to bottom, each layer highlighting as it's named: Data → Belief → Market-making → Calibration → News signal → Interpretation.

**Narration:**
"That fair value flows through a six-layer engine: it ingests live data, maintains a Bayesian belief, and quotes a market."

---

### Scene 4 — Layer 2, market-making (0:42–0:56)

**On screen:** A bid/ask ladder. Inventory counter ticks up; the quotes visibly shift down against the position. Caption: `Avellaneda-Stoikov · inventory-aware`.

**Narration:**
"Layer two is an Avellaneda-Stoikov market-maker. It shifts its quotes against inventory to control one-sided risk — cutting profit-and-loss variance by thirty-seven to sixty percent versus naive quoting."

---

### Scene 5 — Layer 3, calibration (0:56–1:10)

**On screen:** Two source lines (crowd, professional) with a weight bar between them re-balancing over time. Caption: `Bayesian model averaging · online Hedge weights`.

**Narration:**
"Layer three blends the two sources with online Bayesian model averaging, automatically trusting whichever has been better calibrated — improving the Brier score by up to nine-point-six percent over any single source."

---

### Scene 6 — Results (1:10–1:24)

**On screen:** Clean results table animating in: the four rows (P&L variance and Brier, Fed and Bitcoin) with the improvement column in green.

**Narration:**
"These aren't hand-waves. Every result is from reproducible backtests with fixed seeds — the numbers are on screen and in the repo."

---

### Scene 7 — The live scanner (1:24–1:44)

**On screen:** Screen recording of the live Streamlit dashboard. The ranked scanner table loads; the row with the largest gap is at the top. Cursor hovers a `Read` cell to show the plain-language label.

**Narration:**
"And it runs live. A scanner sweeps roughly two thousand Polymarket markets, keeps the ones that are actually priceable, and ranks them by the size of the crowd-versus-professional gap — each one explained in plain English."

---

### Scene 8 — Close (1:44–2:00)

**On screen:** Back to the title card. URLs type out: `crowdarb.streamlit.app` and `github.com/Arisrlt46/crowdarb`. Cursor blinks. Music resolves.

**Narration:**
"CrowdArb — where the crowd and the professionals disagree, and why it matters. Live link and full source in the description."

---

## Recording checklist
- Record screen segments (Scenes 7, and any dashboard b-roll) at 1920×1080, 30 fps.
- Keep the terminal / dashboard in dark mode for a consistent palette.
- Record narration in one pass per scene; leave ~0.5 s of silence at each scene boundary for clean cuts.
- In iMovie: lay narration first, then drop the music bed underneath and duck it to roughly −18 dB so speech stays clear.
