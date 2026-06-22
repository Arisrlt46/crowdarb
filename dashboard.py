"""CrowdArb Layer 5 — Streamlit live dashboard."""

import os
from datetime import date

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from layer0_compare import fetch_cme, fetch_polymarket
from layer1_belief import BetaBelief
from layer1_backtest import generate_price_series
from layer2_avellaneda import (
    BACKTEST_P0,
    BACKTEST_SEED,
    ASParams,
    AvellanedaStoikovMM,
    estimate_sigma2,
)
from layer3_calibration import BayesianBlender
from layer4_llm_signal import lr_update, score_headline

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="CrowdArb", layout="wide")
st.title("CrowdArb")

# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_live_data():
    """Fetch Polymarket and CME probabilities; return (p_poly, p_cme, p_blend, errors)."""
    errors = []
    p_poly, p_cme = None, None

    try:
        _, p_poly = fetch_polymarket()
    except Exception as exc:
        errors.append(f"Polymarket: {exc}")

    try:
        _, _, _, _, p_cme = fetch_cme(date.today())
    except Exception as exc:
        errors.append(f"CME FedWatch: {exc}")

    p_blend = None
    if p_poly is not None and p_cme is not None:
        blender = BayesianBlender(["polymarket", "cme"], eta=0.1)
        p_blend = blender.blend([p_poly, p_cme])

    return p_poly, p_cme, p_blend, errors


@st.cache_data
def load_backtest_frames():
    """Load pre-generated Layer 1 and Layer 2 backtest CSVs."""
    try:
        return pd.read_csv("layer1_backtest.csv"), pd.read_csv("layer2_backtest.csv")
    except FileNotFoundError:
        return None, None


@st.cache_data
def get_sigma2() -> float:
    prices = generate_price_series(p0=BACKTEST_P0, n=200, vol=0.02, seed=BACKTEST_SEED)
    return estimate_sigma2(prices, warmup=20)


# ── Section 1: Live probabilities ─────────────────────────────────────────────

st.subheader("Live Probabilities")

p_poly, p_cme, p_blend, fetch_errors = load_live_data()

col_poly, col_cme, col_blend, col_btn = st.columns([1, 1, 1, 0.35])

col_poly.metric(
    "Polymarket",
    f"{p_poly:.1%}" if p_poly is not None else "—",
    help="P(any Fed rate cut in 2026)",
)
col_cme.metric(
    "CME FedWatch",
    f"{p_cme:.1%}" if p_cme is not None else "—",
    help="P(cut at next FOMC meeting)",
)
col_blend.metric(
    "Blended (BMA)",
    f"{p_blend:.1%}" if p_blend is not None else "—",
    help="Bayesian model average of Polymarket and CME",
)
with col_btn:
    st.write("")  # push button down to align with metrics
    if st.button("↺ Refresh"):
        load_live_data.clear()
        st.rerun()

for err in fetch_errors:
    st.caption(f"⚠ {err}")

# ── Section 2: Avellaneda-Stoikov quotes ──────────────────────────────────────

st.divider()
st.subheader("Avellaneda-Stoikov Quotes")

p_for_quotes = p_blend if p_blend is not None else BACKTEST_P0
sigma2 = get_sigma2()

belief = BetaBelief.from_price(p_for_quotes, strength=10.0)
params = ASParams(gamma=10.0, k=45.0, sigma2=sigma2, T=1.0)
mm = AvellanedaStoikovMM(belief=belief, params=params, inventory=0, t=0.0)

q1, q2, q3, q4 = st.columns(4)
q1.metric("Bid", f"{mm.bid:.4f}")
q2.metric("Ask", f"{mm.ask:.4f}")
q3.metric("Spread", f"{mm.spread:.4f}")
q4.metric("Reservation price", f"{mm.reservation_price:.4f}")

source_label = "blended live estimate" if p_blend is not None else f"fixed prior p₀={BACKTEST_P0}"
st.caption(f"Quoted from {source_label}, zero inventory, γ=10, k=45")

# ── Section 3: Layer 2 backtest chart ─────────────────────────────────────────

st.divider()
st.subheader("Layer 2 Backtest — P&L and Inventory")

df_naive, df_as = load_backtest_frames()

if df_naive is not None and df_as is not None:
    fig, (ax_pnl, ax_inv) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    t = df_as["t"]

    ax_pnl.plot(df_naive["t"], df_naive["total_pnl"], color="#5b9cf6", lw=1.3, label="Naive")
    ax_pnl.plot(t, df_as["total_pnl"], color="#f4845f", lw=1.3, label="Avellaneda-Stoikov")
    ax_pnl.axhline(0, color="#888888", lw=0.6, ls="--")
    ax_pnl.set_ylabel("Total P&L")
    ax_pnl.legend(fontsize=8, framealpha=0.5)

    ax_inv.step(df_naive["t"], df_naive["inventory"], color="#5b9cf6", lw=1, where="post")
    ax_inv.step(t, df_as["inventory"], color="#f4845f", lw=1, where="post")
    ax_inv.axhline(0, color="#888888", lw=0.6, ls="--")
    ax_inv.set_ylabel("Inventory")
    ax_inv.set_xlabel("Timestep")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    l1_var = df_naive["total_pnl"].var()
    l2_var = df_as["total_pnl"].var()
    reduction = (l1_var - l2_var) / l1_var * 100
    st.caption(
        f"P&L variance: Naive {l1_var:.4f} → A-S {l2_var:.4f} ({reduction:.1f}% reduction)  "
        f"· seed=42, p₀={BACKTEST_P0}"
    )
else:
    st.info("Run `python layer2_avellaneda.py` to generate backtest data first.")

# ── Section 4: LLM headline scorer ───────────────────────────────────────────

st.divider()
st.subheader("LLM Headline Scorer")

if "llm_result" not in st.session_state:
    st.session_state.llm_result = None

prior_p = p_blend if p_blend is not None else BACKTEST_P0

headline = st.text_input(
    "News headline",
    placeholder="Fed officials signal openness to rate cut amid cooling inflation data",
)

if st.button("Score", disabled=not bool(headline)):
    with st.spinner("Scoring…"):
        try:
            client = anthropic.Anthropic()
            prior = BetaBelief.from_price(prior_p, strength=10.0)
            signal = score_headline(headline, client)
            posterior = lr_update(prior, signal.likelihood_ratio)
            st.session_state.llm_result = {
                "headline": headline,
                "prior": prior,
                "signal": signal,
                "posterior": posterior,
            }
        except Exception as exc:
            st.error(f"Scoring failed: {exc}")

result = st.session_state.llm_result
if result is not None:
    prior = result["prior"]
    posterior = result["posterior"]
    signal = result["signal"]
    shift = posterior.mean - prior.mean

    r1, r2, r3 = st.columns(3)
    r1.metric("Prior P(cut)", f"{prior.mean:.4f}")
    r2.metric("Posterior P(cut)", f"{posterior.mean:.4f}", delta=f"{shift:+.4f}")
    r3.metric("Likelihood ratio", f"{signal.likelihood_ratio:.3f}")

    st.caption(f"**Justification:** {signal.justification}")
    if result["headline"] != headline:
        st.caption(f'_Showing result for: "{result["headline"]}"_')
