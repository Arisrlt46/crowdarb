"""CrowdArb Layer 5 — interpretation engine: numbers → plain-language readings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class MarketSignals:
    """All known facts about one market, gathered from Layers 0–4."""

    name:            str
    description:     str
    resolution_date: str
    prof_source:     str
    tag:             str      # "rate_decision" | "crypto_level"
    p_poly:          float
    p_prof:          float
    p_blend:         float    # trust-weighted blend (50/50 if weights unknown)
    trusted_source:  str      # "Polymarket" | "Professional"

    # Optional — populated by the full pipeline
    poly_weight:          float | None = None
    prof_weight:          float | None = None
    reservation_price:    float | None = None
    spread:               float | None = None
    mid:                  float | None = None   # 50/50 blend used as A-S input
    llm_lr:               float | None = None
    llm_justification:    str   | None = None


Direction = Literal["crowd_high", "crowd_low", "aligned"]
Magnitude = Literal["aligned", "modest", "notable", "large"]


@dataclass
class Interpretation:
    """Output of interpret(): structured plain-language reading of one market."""

    verdict:   str
    direction: Direction
    magnitude: Magnitude
    lines:     list[str]
    glossary:  dict[str, str]


# ── Glossary ──────────────────────────────────────────────────────────────────

GLOSSARY: dict[str, str] = {
    "Polymarket probability": (
        "The crowd's forecast — the current market price on Polymarket, "
        "where traders buy YES or NO shares and pool their views into a single number."
    ),
    "Professional probability": (
        "A probability extracted from a regulated financial instrument "
        "(futures price or options model), representing professional-market pricing of the same event."
    ),
    "Gap": (
        "The percentage-point difference between the crowd and professional probabilities — "
        "a large gap signals meaningful disagreement between the two sources."
    ),
    "Blended": (
        "A trust-weighted average of the crowd and professional probabilities, "
        "where weights are learned by the Hedge algorithm from each source's historical accuracy."
    ),
    "Trusted (Hedge)": (
        "The source — Polymarket crowd or professional model — that has accumulated "
        "more trust in a simulated historical calibration run."
    ),
    "Bid": (
        "The highest price the market-maker will pay to buy YES — "
        "set below the reservation price by half the spread."
    ),
    "Ask": (
        "The lowest price the market-maker will accept to sell YES — "
        "set above the reservation price by half the spread."
    ),
    "Reservation price": (
        "The inventory-adjusted fair value the market-maker quotes around — "
        "shifted away from the blended probability to reduce one-sided inventory risk."
    ),
    "Spread": (
        "The gap between the buy (ask) and sell (bid) price — "
        "the market-maker's compensation for providing liquidity and bearing inventory risk."
    ),
    "Likelihood ratio": (
        "A headline-scoring number: above 1 is bullish (raises the probability), "
        "below 1 is bearish (lowers it), and exactly 1 means the headline carries no signal."
    ),
}


# ── Deterministic interpretation engine ──────────────────────────────────────

def interpret(sig: MarketSignals) -> Interpretation:
    """
    Pure deterministic reading of MarketSignals.
    Never raises on valid inputs. No API calls.
    """
    gap     = sig.p_poly - sig.p_prof
    abs_gap = abs(gap)
    pts     = round(abs_gap * 100)

    # Magnitude
    if abs_gap < 0.03:
        magnitude: Magnitude = "aligned"
    elif abs_gap < 0.08:
        magnitude = "modest"
    elif abs_gap < 0.15:
        magnitude = "notable"
    else:
        magnitude = "large"

    # Direction
    if magnitude == "aligned":
        direction: Direction = "aligned"
    elif gap > 0:
        direction = "crowd_high"
    else:
        direction = "crowd_low"

    # Verdict
    if direction == "aligned":
        verdict = "Crowd and professional pricing agree — no meaningful disagreement."
    else:
        side = "higher" if direction == "crowd_high" else "lower"
        desc = {
            "modest":  "a modest divergence",
            "notable": "a notable divergence",
            "large":   "a large divergence",
        }[magnitude]
        verdict = (
            f"The crowd is {pts} point{'s' if pts != 1 else ''} {side} "
            f"than professional pricing — {desc}."
        )

    # Per-fact lines
    lines: list[str] = []

    lines.append(f"Polymarket traders price this event at {sig.p_poly:.1%}.")

    source_label = f" ({sig.prof_source})" if sig.prof_source else ""
    lines.append(f"The professional model puts it at {sig.p_prof:.1%}{source_label}.")

    if sig.poly_weight is not None and sig.prof_weight is not None:
        heavier = "Polymarket" if sig.poly_weight >= sig.prof_weight else "the professional model"
        lines.append(
            f"The blended estimate is {sig.p_blend:.1%}, leaning toward {heavier} "
            f"(weights: crowd {sig.poly_weight:.0%} / professional {sig.prof_weight:.0%})."
        )
    else:
        lines.append(
            f"The blended estimate is {sig.p_blend:.1%}; the Hedge algorithm currently "
            f"trusts the {sig.trusted_source.lower()} source more."
        )

    if sig.reservation_price is not None and sig.mid is not None:
        tilt = sig.reservation_price - sig.mid
        if abs(tilt) < 0.001:
            lines.append(
                "The market-maker's reservation price equals the blended mid — "
                "inventory is neutral."
            )
        else:
            direction_word = "above" if tilt > 0 else "below"
            lines.append(
                f"The reservation price ({sig.reservation_price:.4f}) sits "
                f"{abs(tilt):.4f} {direction_word} the blended mid, "
                "reflecting current inventory tilt."
            )

    if sig.llm_lr is not None:
        if abs(sig.llm_lr - 1.0) < 0.05:
            lr_read = "neutral — no meaningful update to the probability"
        elif sig.llm_lr > 1.0:
            lr_read = f"bullish (LR = {sig.llm_lr:.2f}) — raised the blended probability"
        else:
            lr_read = f"bearish (LR = {sig.llm_lr:.2f}) — lowered the blended probability"
        suffix = f": \"{sig.llm_justification}\"" if sig.llm_justification else ""
        lines.append(f"The latest headline scored as {lr_read}{suffix}.")

    return Interpretation(
        verdict=verdict,
        direction=direction,
        magnitude=magnitude,
        lines=lines,
        glossary=GLOSSARY,
    )


# ── Optional LLM narrative ────────────────────────────────────────────────────

class NarrativeOut(BaseModel):
    summary: str


def narrate(sig: MarketSignals, interp: Interpretation, client) -> str:
    """
    Produce a 2-3 sentence plain-English paragraph via the LLM.
    Called only when the user explicitly clicks the button; never by interpret().
    """
    from layer4_llm_signal import MODEL

    facts   = "\n".join(f"- {line}" for line in interp.lines)
    content = f"Market: {sig.name}\nVerdict: {interp.verdict}\nFacts:\n{facts}"
    system  = (
        "You are explaining financial probability data to a curious non-expert. "
        "Write exactly 2-3 sentences in plain English. "
        "Use only the facts provided — invent nothing, add no caveats or hedging, use no jargon. "
        "Start directly with what the numbers mean."
    )
    response = client.messages.parse(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_format=NarrativeOut,
    )
    return response.parsed_output.summary


# ── Compact one-line read for scanner table ───────────────────────────────────

def compact_read(sig: MarketSignals) -> str:
    """Short label for the scanner table, e.g. 'Crowd 21 pts high' / 'Aligned'."""
    interp = interpret(sig)
    if interp.direction == "aligned":
        return "Aligned"
    pts  = round(abs(sig.p_poly - sig.p_prof) * 100)
    side = "high" if interp.direction == "crowd_high" else "low"
    return f"Crowd {pts} pts {side}"


# ── CLI formatter ─────────────────────────────────────────────────────────────

def format_interpretation(interp: Interpretation) -> str:
    """Formatted multi-line string for terminal output."""
    W   = 66
    sep = "─" * W
    out = [sep, "  What this means", sep, f"  {interp.verdict}", ""]
    for line in interp.lines:
        out.append(f"  · {line}")
    out.append(sep)
    return "\n".join(out)


# ── Self-test (no API key required) ──────────────────────────────────────────

def _self_test() -> None:
    cases = [
        # 1. Aligned — crowd and professional agree
        MarketSignals(
            name="Synthetic Aligned Market",
            description="A hypothetical market where both sources agree",
            resolution_date="2026-12-31",
            prof_source="Hypothetical futures",
            tag="rate_decision",
            p_poly=0.48,
            p_prof=0.50,
            p_blend=0.49,
            trusted_source="Professional",
            poly_weight=0.48,
            prof_weight=0.52,
            reservation_price=0.49,
            spread=0.040,
            mid=0.49,
        ),
        # 2. Crowd high, large — Fed-like gap with weights
        MarketSignals(
            name="Fed Rate Cut 2026",
            description="P(any 25 bp rate cut in 2026)",
            resolution_date="2026-12-31",
            prof_source="CME ZQ futures",
            tag="rate_decision",
            p_poly=0.228,
            p_prof=0.000,
            p_blend=0.114,
            trusted_source="Professional",
            poly_weight=0.39,
            prof_weight=0.61,
            reservation_price=0.114,
            spread=0.044,
            mid=0.114,
        ),
        # 3. Crowd high, notable — crypto market with LLM signal
        MarketSignals(
            name="ETH $4k by Jan 01, 2027",
            description="Will Ethereum reach $4,000 by January 1, 2027?",
            resolution_date="2027-01-01",
            prof_source="Black-Scholes (Deribit ETH DVOL)",
            tag="crypto_level",
            p_poly=0.105,
            p_prof=0.021,
            p_blend=0.063,
            trusted_source="Professional",
            llm_lr=1.8,
            llm_justification="Ethereum ETF approval signals major institutional demand.",
        ),
    ]

    for sig in cases:
        interp = interpret(sig)
        print(format_interpretation(interp))
        print(f"  compact: {compact_read(sig)}")
        print()


if __name__ == "__main__":
    _self_test()
