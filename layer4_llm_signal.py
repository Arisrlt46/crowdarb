"""CrowdArb Layer 4 — LLM signal layer: headline → likelihood ratio → belief update."""

import argparse

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from layer1_belief import BetaBelief

MODEL = "claude-opus-4-8"
DEFAULT_PRIOR_P = 0.72  # fallback prior if no market feed is available

# Fed / rates: headline scored against a 25 bp rate-cut event
_SYSTEM_FED = (
    "You are a central-bank analyst scoring Fed news headlines for a rate-cut prediction market. "
    "Given a headline, return LR = P(headline | rate cut) / P(headline | no rate cut). "
    "LR > 1 means the headline is more consistent with a cut; LR < 1 means the opposite. "
    "Clamp LR to [0.1, 10.0]. Provide exactly one sentence of justification."
)

# Crypto: headline scored against an asset reaching a specific price target
_SYSTEM_CRYPTO = (
    "You are a crypto-markets analyst scoring news headlines for a binary price-level prediction market. "
    "Given a headline, return LR = P(headline | asset reaches its price target) / P(headline | asset does not). "
    "LR > 1 means the headline is bullish and makes the target more likely; LR < 1 means bearish. "
    "Clamp LR to [0.1, 10.0]. Provide exactly one sentence of justification."
)

_SYSTEMS: dict[str, str] = {
    "rate_decision": _SYSTEM_FED,
    "crypto_level":  _SYSTEM_CRYPTO,
}


class LLMSignal(BaseModel):
    likelihood_ratio: float = Field(
        ge=0.1, le=10.0,
        description="P(headline|cut) / P(headline|no_cut), clamped to [0.1, 10.0]",
    )
    justification: str = Field(
        description="One sentence explaining the assigned likelihood ratio",
    )


def score_headline(
    headline: str,
    client: anthropic.Anthropic,
    market_type: str = "rate_decision",
) -> LLMSignal:
    """Return a structured LLM signal for a news headline using the market-appropriate prompt."""
    system = _SYSTEMS.get(market_type, _SYSTEM_FED)
    response = client.messages.parse(
        model=MODEL,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": f'Headline: "{headline}"'}],
        output_format=LLMSignal,
    )
    return response.parsed_output


def lr_update(belief: BetaBelief, lr: float) -> BetaBelief:
    """Bayesian LR update: p_new = lr*p / (lr*p + 1 - p), preserving prior strength."""
    p = belief.mean
    p_new = (lr * p) / (lr * p + (1.0 - p))
    return BetaBelief.from_price(p_new, strength=belief.strength)


def run(headline: str, prior_p: float = DEFAULT_PRIOR_P) -> None:
    load_dotenv()
    client = anthropic.Anthropic()

    prior = BetaBelief.from_price(prior_p, strength=10.0)

    print(f"Headline  : {headline!r}")
    print(f"Prior     : {prior}")

    signal = score_headline(headline, client)

    print(f"\nLLM signal")
    print(f"  LR            : {signal.likelihood_ratio:.3f}")
    print(f"  Justification : {signal.justification}")

    posterior = lr_update(prior, signal.likelihood_ratio)
    shift = posterior.mean - prior.mean

    print(f"\nPosterior : {posterior}")
    print(f"Shift     : {shift:+.4f}  ({prior.mean:.4f} → {posterior.mean:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrowdArb Layer 4 — LLM signal layer")
    parser.add_argument(
        "headline",
        nargs="?",
        default="Fed officials signal openness to rate cut amid cooling inflation data",
        help="News headline to score",
    )
    parser.add_argument(
        "--prior", type=float, default=DEFAULT_PRIOR_P,
        help=f"Prior probability of a rate cut (default: {DEFAULT_PRIOR_P})",
    )
    args = parser.parse_args()
    run(args.headline, args.prior)
