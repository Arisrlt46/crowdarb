# CrowdArb — Demo Walkthrough Script

**Format:** screen recording of the live dashboard at **https://crowdarb.streamlit.app** + voiceover. Click-by-click. Dark mode for a consistent palette.
**Target runtime:** ~2:00–2:30. Timings are guides — the cold start can vary, so record the load separately and trim.

> Note: I can't generate the video itself. This is the click-by-click narration for you to record.

---

### 0:00–0:12 — Open the app

**Action:** Type `crowdarb.streamlit.app` into the address bar and hit enter. The app begins its cold start (spinner, ≈15 s).

**Narration:**
"This is CrowdArb, live. On first load it wakes up and auto-discovers markets — it's scanning roughly two thousand live Polymarket contracts right now, which takes about fifteen seconds."

*(Tip: pause the recording during the load and cut back in when the table appears, so the demo stays tight.)*

---

### 0:12–0:35 — The scanner table

**Action:** The ranked scanner table appears. Slowly move the cursor down the rows. Point at the header column showing the gap.

**Narration:**
"Every row is a live market where we can price both sides — the crowd's Polymarket probability and an independent professional probability. The table is sorted by the gap between them, so the biggest disagreements sit right at the top."

---

### 0:35–0:52 — The "Read" column

**Action:** Hover over a `Read` cell on the top row so the plain-language label is visible (e.g. "Crowd 21 pts high").

**Narration:**
"You don't need to read the raw numbers. The 'Read' column is a one-line, plain-English summary from the interpretation layer — here it's telling us the crowd is twenty-one points higher than the professional market on this contract."

---

### 0:52–1:15 — Metric tooltips / glossary

**Action:** Hover over the little `?` help icon on one of the metrics (e.g. reservation price or spread) to show the tooltip. Then expand the glossary section.

**Narration:**
"Every metric has a tooltip in plain language, and there's a full glossary built in — so 'reservation price' or 'Brier score' aren't just jargon on the screen. This is meant to be readable by someone who doesn't trade for a living."

---

### 1:15–1:40 — Detail panel and live quotes

**Action:** Click the top-ranked market to open its detail panel. Point at the live Avellaneda-Stoikov bid/ask quotes and the fair value.

**Narration:**
"Clicking a market opens its detail panel. Here's the blended fair value, and the live Avellaneda-Stoikov quotes — a bid and an ask shifted around that fair value, adjusted for inventory. This is the market-maker layer running on live data."

---

### 1:40–2:05 — Headline scorer (Layer 4)

**Action:** Scroll to the headline scorer. Type a sample headline (e.g. "Fed signals it may hold rates steady through year-end") and submit. The likelihood ratio and the updated belief appear.

**Narration:**
"I can also feed in a news headline. The model returns a likelihood ratio — how much more likely that headline is if the event happens versus if it doesn't — and applies it as a Bayesian update, nudging the fair value and the quotes. The result is scoped to this market, so it sticks as I switch around."

---

### 2:05–2:25 — Refresh and close

**Action:** Click the **Refresh** button to clear the discovery cache and reload. As it re-scans, cut to a clean shot of the top of the table.

**Narration:**
"Refresh clears the cache and re-scans everything from scratch, so the board is always current. That's CrowdArb — a live, explainable read on where the crowd and the professionals disagree. Source is on GitHub."

---

## Recording checklist
- 1920×1080, 30 fps, browser in dark mode, bookmarks bar hidden.
- Pre-warm the app once before recording so the *real* take isn't a full 15 s cold start (or record the load and trim it).
- Have the sample headline copied to your clipboard so you can paste instead of typing on camera.
- Keep the cursor movements slow and deliberate — they read as confident on video.
