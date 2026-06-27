"""CrowdArb Phase G5 — multi-market scanner dashboard with auto-discovery."""

import os

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from market_pair import discover_markets
from layer1_backtest import generate_price_series
from layer1_belief import BetaBelief
from layer2_avellaneda import ASParams, AvellanedaStoikovMM, estimate_sigma2
from layer3_calibration import BayesianBlender
from layer4_llm_signal import lr_update, score_headline

st.set_page_config(page_title="CrowdArb Scanner", layout="wide")
st.title("CrowdArb — Multi-Market Scanner")

# ── Discovery + data loading ──────────────────────────────────────────────────
# One cached call covers the full Gamma scan AND per-market live data fetch.
# 300s TTL: a cold scan + data fetch takes ~15 s; 5-min staleness is acceptable.
# Returns list[tuple]: (name, p_poly, p_prof, meta, error_str | None)

@st.cache_data(ttl=300)
def discover_and_load() -> list[tuple]:
    """Discover markets from live catalogue; return (name, p_poly, p_prof, meta, err)."""
    try:
        records = discover_markets(min_volume_usd=50_000, min_days=30)
    except Exception as exc:
        return [("Discovery", None, None,
                 {"name": "Discovery failed", "description": str(exc),
                  "resolution_date": "—", "prof_source": "—"}, str(exc))]

    # discover_markets() already sorted by gap descending; preserve that order.
    results: list[tuple] = []
    for rec in records:
        meta = {
            "name":            rec["name"],
            "description":     rec["question"],
            "resolution_date": rec["resolution_date"],
            "prof_source":     rec["prof_source"],
        }
        results.append((rec["name"], rec["p_poly"], rec["p_prof"], meta, None))

    return results


# ── Deterministic helpers (cached indefinitely — inputs fully determine outputs) ─

@st.cache_data
def _hedge_trusted(p_poly: float, p_prof: float) -> str:
    """Run 200-step Hedge simulation; return name of the higher-weight source."""
    rng        = np.random.default_rng(42)
    p_blend    = 0.5 * p_poly + 0.5 * p_prof
    p_poly_sim = np.clip(rng.normal(p_poly, max(p_poly * 0.30, 0.005), 200), 0.001, 0.999)
    p_prof_sim = np.clip(rng.normal(p_prof, max(p_prof * 0.10, 0.001), 200), 0.001, 0.999)
    outcomes   = rng.binomial(1, p_blend, 200).astype(int)
    blender    = BayesianBlender(["polymarket", "professional"], eta=0.1)
    for pp, pr, y in zip(p_poly_sim, p_prof_sim, outcomes):
        blender.update([pp, pr], y)
    ws = blender.weight_dict()
    return "Polymarket" if ws["polymarket"] >= ws["professional"] else "Professional"


@st.cache_data
def _as_quotes(p_blend: float):
    """Compute A-S bid/ask/spread/reservation price for a given blended prior."""
    prices = generate_price_series(p0=p_blend, n=200, vol=0.02, seed=42)
    sigma2 = estimate_sigma2(prices, warmup=20)
    belief = BetaBelief.from_price(p_blend, strength=10.0)
    params = ASParams(gamma=10.0, k=45.0, sigma2=sigma2, T=1.0)
    mm     = AvellanedaStoikovMM(belief=belief, params=params)
    return mm.bid, mm.ask, mm.spread, mm.reservation_price


# ── Load markets ──────────────────────────────────────────────────────────────

hdr_col, btn_col = st.columns([6, 1])
hdr_col.subheader("Market Scanner")
with btn_col:
    st.write("")
    if st.button("↺ Refresh"):
        discover_and_load.clear()
        st.rerun()

market_results = discover_and_load()   # list of (name, p_poly, p_prof, meta, err|None)

# ── Scanner table ─────────────────────────────────────────────────────────────

def _pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.1%}"

rows       = []
load_errors = []

for name, p_poly, p_prof, meta, err in market_results:
    if err:
        load_errors.append((name, err))
        rows.append({"Market": name, "Polymarket": None, "Professional": None,
                     "|Gap|": None, "Trusted (Hedge)": "—", "_sort": -1.0})
    else:
        gap = abs(p_poly - p_prof)
        rows.append({"Market": name, "Polymarket": p_poly, "Professional": p_prof,
                     "|Gap|": gap, "Trusted (Hedge)": _hedge_trusted(p_poly, p_prof),
                     "_sort": gap})

df_raw = (
    pd.DataFrame(rows)
    .sort_values("_sort", ascending=False)
    .drop(columns=["_sort"])
    .reset_index(drop=True)
)

display_df = df_raw.copy()
for col in ["Polymarket", "Professional", "|Gap|"]:
    display_df[col] = display_df[col].apply(_pct)

st.dataframe(display_df, use_container_width=True, hide_index=True)

n_ok = sum(1 for *_, err in market_results if err is None)
st.caption(f"Discovered {n_ok} market(s) from live Polymarket scan · cache TTL 300 s")

for mkt_name, err_msg in load_errors:
    st.caption(f"⚠ {mkt_name}: {err_msg}")

# ── Market detail panel ───────────────────────────────────────────────────────

st.divider()
st.subheader("Market Detail")

ok_markets = {
    name: (p_poly, p_prof, meta)
    for name, p_poly, p_prof, meta, err in market_results
    if err is None
}

if not ok_markets:
    st.warning("No markets loaded successfully. Check API connectivity and try refreshing.")
    st.stop()

selected = st.selectbox("Select market", list(ok_markets))
p_poly, p_prof, meta = ok_markets[selected]
p_blend    = 0.5 * p_poly + 0.5 * p_prof
gap_signed = p_poly - p_prof

st.caption(
    f"**{meta['description']}** · "
    f"Resolves {meta['resolution_date']} · "
    f"Prof. source: {meta.get('prof_source', '—')}"
)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Polymarket",        f"{p_poly:.1%}")
c2.metric("Professional",      f"{p_prof:.1%}")
c3.metric("Gap (Poly − Prof)", f"{gap_signed:+.1%}")
c4.metric("Blended (50/50)",   f"{p_blend:.1%}")

st.divider()

st.subheader("Avellaneda-Stoikov Quotes")
bid, ask, spread, r_price = _as_quotes(p_blend)
q1, q2, q3, q4 = st.columns(4)
q1.metric("Bid",               f"{bid:.4f}")
q2.metric("Ask",               f"{ask:.4f}")
q3.metric("Spread",            f"{spread:.4f}")
q4.metric("Reservation price", f"{r_price:.4f}")
st.caption(f"From blended prior p₀ = {p_blend:.4f}, zero inventory, γ=10, k=45")

st.divider()

# LLM scorer — session state scoped per market so results persist across market switches
st.subheader("LLM Headline Scorer")

scorer_key   = f"llm_{selected}"
headline_key = f"headline_{selected}"
if scorer_key not in st.session_state:
    st.session_state[scorer_key] = None

headline = st.text_input(
    "News headline",
    placeholder="e.g. Fed signals rate cut amid cooling inflation data",
    key=headline_key,
)

if st.button("Score", key=f"score_{selected}", disabled=not bool(headline)):
    with st.spinner("Scoring…"):
        try:
            client = anthropic.Anthropic()
            prior  = BetaBelief.from_price(p_blend, strength=10.0)
            signal = score_headline(headline, client)
            st.session_state[scorer_key] = {
                "headline":  headline,
                "prior":     prior,
                "signal":    signal,
                "posterior": lr_update(prior, signal.likelihood_ratio),
            }
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")

result = st.session_state[scorer_key]
if result is not None:
    prior     = result["prior"]
    posterior = result["posterior"]
    signal    = result["signal"]
    shift     = posterior.mean - prior.mean

    r1, r2, r3 = st.columns(3)
    r1.metric("Prior P",           f"{prior.mean:.4f}")
    r2.metric("Posterior P",       f"{posterior.mean:.4f}", delta=f"{shift:+.4f}")
    r3.metric("Likelihood ratio",  f"{signal.likelihood_ratio:.3f}")
    st.caption(f"**Justification:** {signal.justification}")

    if result["headline"] != headline:
        st.caption(f'_Showing result for: "{result["headline"]}"_')

st.caption(
    "Scorer prompt is calibrated for rate-cut signals. "
    "For crypto markets, interpret LR > 1 as bullish and LR < 1 as bearish."
)
