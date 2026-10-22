"""Uncertainty and baselines for the scorecard's summaries. Pure: no I/O.

A hit rate or an average return on its own says little. These helpers put an
interval around each one and compare the calls with two no-skill baselines:
always long the same names on the same dates (market drift), and the same
calls with their directions shuffled (does the direction matter at all).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import combinations

from scipy.stats import norm, t

CONFIDENCE_LEVEL = 0.95
"""Fixed rather than tunable so every scorecard run reads the same way: the
intervals are a reading aid, not a decision threshold."""

SHUFFLE_SAMPLES = 10_000
"""Random relabelings for the shuffle test when there are too many to list.
With 10,000 the p-value's own Monte Carlo error is about 0.005 at p = 0.05,
well below anything worth reading into."""

SHUFFLE_SEED = 20261022
"""Fixed so the same data always prints the same p-value."""

_TIE_TOLERANCE = 1e-12  # float noise between equal means must count as a tie


def wilson_interval(hits: int, n: int, level: float = CONFIDENCE_LEVEL) -> tuple[float, float] | None:
    """The Wilson score interval for ``hits`` out of ``n``; None when n is 0.

    Unlike the textbook p +/- z*SE it stays inside [0, 1] and behaves at
    small n and at 0% or 100%, which is where the scorecard lives for now.
    """
    if n <= 0:
        return None
    z = float(norm.ppf(0.5 + level / 2))
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_interval(values: Sequence[float], level: float = CONFIDENCE_LEVEL) -> tuple[float, float] | None:
    """A Student-t interval for the mean; None with fewer than two values."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half = float(t.ppf(0.5 + level / 2, n - 1)) * math.sqrt(variance / n)
    return mean - half, mean + half


def shuffle_p_value(
    signs: Sequence[int],
    moves: Sequence[float],
    *,
    samples: int = SHUFFLE_SAMPLES,
    seed: int = SHUFFLE_SEED,
) -> float | None:
    """One-sided p-value of the calls' mean signed return against shuffled labels.

    ``signs`` are the calls (+1 bullish, -1 bearish) and ``moves`` what the
    underlying did. The null keeps the moves and the count of bullish calls
    and reassigns which call went with which move. Every relabeling is listed
    when there are at most ``samples`` of them (an exact p-value); otherwise
    ``samples`` random ones are drawn with a fixed seed. None with no calls.
    """
    n = len(signs)
    if n == 0 or n != len(moves):
        return None
    observed = sum(sign * move for sign, move in zip(signs, moves))
    bullish = sum(1 for sign in signs if sign > 0)
    total = sum(moves)
    if math.comb(n, bullish) <= samples:
        as_high = count = 0
        for chosen in combinations(range(n), bullish):
            up = sum(moves[i] for i in chosen)
            count += 1
            if up - (total - up) >= observed - _TIE_TOLERANCE:
                as_high += 1
        return as_high / count
    rng = random.Random(seed)
    shuffled = list(signs)
    as_high = 0
    for _ in range(samples):
        rng.shuffle(shuffled)
        if sum(sign * move for sign, move in zip(shuffled, moves)) >= observed - _TIE_TOLERANCE:
            as_high += 1
    # the observed labeling is one of the possible ones, so count it once
    return (as_high + 1) / (samples + 1)
