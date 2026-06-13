"""CrowdArb Layer 1 — Beta-Bernoulli Bayesian belief updater."""

from dataclasses import dataclass

from scipy import stats


@dataclass
class BetaBelief:
    """Bayesian belief over a binary-event probability, represented as a Beta distribution."""

    alpha: float  # pseudo-count of YES observations
    beta: float   # pseudo-count of NO observations

    @classmethod
    def from_price(cls, p: float, strength: float = 10.0) -> "BetaBelief":
        """Initialise from a market price p with a given total pseudo-count (confidence)."""
        # Moment-match: mean = alpha / (alpha + beta) = p, total = strength
        return cls(alpha=p * strength, beta=(1.0 - p) * strength)

    def update(self, outcome: int) -> "BetaBelief":
        """Return the posterior after observing one Bernoulli trial (1 = YES, 0 = NO)."""
        return BetaBelief(alpha=self.alpha + outcome, beta=self.beta + (1 - outcome))

    def update_batch(self, yes: int, no: int) -> "BetaBelief":
        """Return the posterior after observing a batch of yes successes and no failures."""
        return BetaBelief(alpha=self.alpha + yes, beta=self.beta + no)

    @property
    def mean(self) -> float:
        """Posterior mean — point estimate of the event probability."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Posterior variance — uncertainty around the mean."""
        n = self.alpha + self.beta
        return (self.alpha * self.beta) / (n**2 * (n + 1))

    @property
    def std(self) -> float:
        return self.variance**0.5

    @property
    def strength(self) -> float:
        """Total pseudo-count — equivalent number of observations backing this belief."""
        return self.alpha + self.beta

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the equal-tailed credible interval at the given probability level."""
        dist = stats.beta(self.alpha, self.beta)
        tail = (1.0 - level) / 2.0
        return dist.ppf(tail), dist.ppf(1.0 - tail)

    def __repr__(self) -> str:
        lo, hi = self.credible_interval()
        return (
            f"BetaBelief(α={self.alpha:.2f}, β={self.beta:.2f})  "
            f"mean={self.mean:.4f}  std={self.std:.4f}  "
            f"95% CI=[{lo:.4f}, {hi:.4f}]  strength={self.strength:.1f}"
        )


@dataclass
class NaiveMarketMaker:
    """
    Symmetric market-maker that quotes a fixed spread around a Bayesian fair value.

    At each step:
      fair_value = belief.mean
      bid        = fair_value - half_spread
      ask        = fair_value + half_spread

    Quotes are clamped to (0, 1) so they remain valid prediction-market prices.
    No inventory adjustment — that arrives in Layer 2 (Avellaneda-Stoikov).
    """

    belief: BetaBelief
    half_spread: float = 0.02  # 2 cents each side = 4-cent total spread

    @property
    def fair_value(self) -> float:
        return self.belief.mean

    @property
    def bid(self) -> float:
        return max(0.001, self.fair_value - self.half_spread)

    @property
    def ask(self) -> float:
        return min(0.999, self.fair_value + self.half_spread)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    def observe(self, outcome: int) -> "NaiveMarketMaker":
        """Return an updated market-maker after observing one Bernoulli outcome."""
        return NaiveMarketMaker(
            belief=self.belief.update(outcome),
            half_spread=self.half_spread,
        )

    def __repr__(self) -> str:
        return (
            f"NaiveMarketMaker  fair={self.fair_value:.4f}  "
            f"bid={self.bid:.4f}  ask={self.ask:.4f}  "
            f"spread={self.spread:.4f}  ({self.belief})"
        )


def demo():
    """Show belief updates and quote evolution using live Polymarket data as the prior."""
    from layer0_compare import fetch_polymarket

    print("Fetching Polymarket prior...")
    _, p_poly = fetch_polymarket()

    belief = BetaBelief.from_price(p_poly, strength=10.0)
    mm = NaiveMarketMaker(belief=belief, half_spread=0.02)

    print(f"\nPrior (Polymarket p={p_poly:.3f}, strength=10)")
    print(f"  belief : {belief}")
    print(f"  quotes : {mm}")

    # Simulate a sequence of cut signals (YES) followed by no-cut signals (NO)
    observations = [(1, "YES — cut signal"), (1, "YES — cut signal"), (1, "YES — cut signal"),
                    (0, "NO  — no-cut signal"), (0, "NO  — no-cut signal")]

    print()
    for outcome, label in observations:
        mm = mm.observe(outcome)
        print(f"  [{label}]")
        print(f"    {mm}")


if __name__ == "__main__":
    demo()
