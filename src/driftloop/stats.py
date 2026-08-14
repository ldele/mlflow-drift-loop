"""How much to believe the numbers this project reports.

Every headline in the README is a statistic computed on a few dozen weekly
windows. Published as bare point estimates, ``+43.7%`` and ``+0.2%`` sit in one
table in one typeface with nothing to separate a result from noise. This module
attaches a range to each.

## Why the ordinary method is wrong here

The usual way to put a range on a statistic is to draw the observations again,
at random and with replacement, a few thousand times, and see how far the answer
moves. That works when each observation is independent of the others. These are
not, for two reasons that compound.

The windows overlap. A monitor window covers 14 days and the replay advances 7,
so every window shares half its hours with the one before it. Two neighbouring
errors are computed partly on the same data.

The weather persists. Pollution episodes run for days or weeks, so a bad week is
followed by a bad week far more often than chance would give.

Redrawing single weeks destroys both structures. The resampled series then looks
like it carries more independent information than it really does, and the range
comes out too narrow, which is the direction that turns noise into a finding.

The repair is to redraw runs of consecutive weeks instead of individual ones, so
the dependence inside a run survives. Its name is the moving-block bootstrap
(Künsch, 1989). The only judgement it needs is how long the runs should be.

## Choosing the block length

Default ``L = max(2, round(n ** (1/3)))``, the usual rate (Künsch 1989; Politis
& Romano 1994). A 48-week replay gets 4; a 19-week one gets 3.

Two caveats.

The rate is asymptotic and n here runs from 19 to 48, so these intervals are
approximations rather than exact.

Block length is the one free parameter, which makes it the one place a
convenient answer could be chosen. ``sensitivity_to_block_length`` reruns the
interval across a range of L so the spread is visible. Where a conclusion holds
at one block length and not another, the conclusion is the block length.

## What this module does not do

No p-values. The question is never whether an effect differs from zero in
isolation, but how large it is and whether it could plausibly be nothing. A
range answers that directly. Where a reader wants the null-hypothesis version,
an interval clear of zero carries it.

## References

- Künsch, H. R. (1989). *The Jackknife and the Bootstrap for General Stationary
  Observations.* Annals of Statistics, 17(3), 1217-1241.
- Politis, D. N., & Romano, J. P. (1994). *The Stationary Bootstrap.* JASA,
  89(428), 1303-1313.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.*
  Chapman & Hall. The percentile interval used here is §13.3.
- Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and
  Statistical Inference.* JASA, 22(158), 209-212.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

# Fixed so every reported interval is reproducible; an interval that moves
# between runs cannot be checked against the published table.
DEFAULT_SEED = 20260811
DEFAULT_RESAMPLES = 10_000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """An estimate and the range the resampling could not rule out.

    ``n`` counts the observations. ``n_effective`` counts how many independent
    observations they are worth once the correlation between neighbours is taken
    out. Both are published: ``n=47`` beside ``n_eff=5`` explains the width of
    the interval on its own.
    """

    point: float
    lo: float
    hi: float
    n: int
    n_effective: float
    block_length: int
    resamples: int

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely on one side of zero.

        Not called "significant" on purpose: it is one interval at one alpha,
        and six are read off a single table, which is a multiple-comparisons
        problem this project does not correct for. A reading aid.
        """
        return (self.lo > 0 and self.hi > 0) or (self.lo < 0 and self.hi < 0)

    def format(self, unit: str = "%", places: int = 1) -> str:
        return (
            f"{self.point:+.{places}f}{unit} "
            f"[{self.lo:+.{places}f}, {self.hi:+.{places}f}]"
        )

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "point": self.point,
            "lo": self.lo,
            "hi": self.hi,
            "n": self.n,
            # Two decimals, because consumers round again for display: at one
            # decimal an effective n of 5.45 is stored as 5.5 and displays as 6,
            # while the same figure from the raw value displays as 5.
            "n_effective": round(self.n_effective, 2),
            "block_length": self.block_length,
            "resamples": self.resamples,
            "excludes_zero": self.excludes_zero,
        }


def lag1_autocorrelation(values: np.ndarray) -> float:
    """Lag-1 autocorrelation, NaN-dropped, clipped to [-0.99, 0.99].

    Clipped because it feeds a variance-inflation formula that divides by
    ``1 - rho``: an estimate of 1.0 on a short noisy series would report
    an effective sample size of zero, which is an artefact of the estimator
    rather than a fact about the data.
    """
    clean = values[~np.isnan(values)]
    if clean.size < 3:
        return 0.0
    centred = clean - clean.mean()
    denominator = float(np.sum(centred**2))
    if denominator == 0:
        return 0.0
    rho = float(np.sum(centred[:-1] * centred[1:]) / denominator)
    return float(np.clip(rho, -0.99, 0.99))


def effective_sample_size(values: np.ndarray) -> float:
    """How many independent observations a correlated series is worth.

    ``n_eff = n * (1 - rho) / (1 + rho)``, the standard first-order
    approximation. The dependence here is not precisely first-order, so the
    figure is indicative rather than sharp. Its job is to make the cost of
    correlation visible. At rho = 0.5, 48 weeks are worth 16.
    """
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return 0.0
    rho = lag1_autocorrelation(clean)
    return float(max(1.0, clean.size * (1 - rho) / (1 + rho)))


def default_block_length(n: int) -> int:
    """``n ** (1/3)``, floored at 2 and capped so blocks cannot exceed n // 2.

    The cap matters on short replays: a block as long as the series redraws that
    same series every time and reports an interval of zero width.
    """
    if n < 4:
        return max(1, n)
    return int(max(2, min(round(n ** (1 / 3)), n // 2)))


def _moving_blocks(n: int, block: int, rng: np.random.Generator, resamples: int) -> np.ndarray:
    """Index matrix of shape (resamples, n) built from contiguous blocks."""
    n_starts = n - block + 1
    per_draw = int(np.ceil(n / block))
    starts = rng.integers(0, n_starts, size=(resamples, per_draw))
    offsets = np.arange(block)
    # (resamples, per_draw, block) -> flatten the last two axes, then truncate
    return (starts[:, :, None] + offsets[None, None, :]).reshape(resamples, -1)[:, :n]


def block_bootstrap(
    columns: Sequence[np.ndarray],
    statistic: Callable[..., float],
    block: int | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Percentile interval for a statistic of one or more aligned series.

    ``columns`` are redrawn together using the same block indices, so paired
    observations stay paired. Any statistic comparing two models on one window
    needs this; broken pairing compares one model's week 4 against another's
    week 31 and reports the difference as uncertainty.

    Rows where any column is NaN are dropped jointly before resampling, so the
    columns stay aligned.
    """
    arrays = [np.asarray(c, dtype=float) for c in columns]
    if not arrays or arrays[0].size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, 0.0, 0, 0)

    keep = ~np.any(np.isnan(np.vstack(arrays)), axis=0)
    arrays = [a[keep] for a in arrays]
    n = arrays[0].size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, 0.0, 0, 0)

    point = float(statistic(*arrays))
    if n < 4:
        # Too short to resample meaningfully. Report the point estimate with an
        # explicitly infinite interval rather than a narrow fabricated one.
        return Interval(point, float("-inf"), float("inf"), n, float(n), 0, 0)

    block = block or default_block_length(n)
    rng = np.random.default_rng(seed)
    index = _moving_blocks(n, block, rng, resamples)
    draws = np.array([statistic(*[a[row] for a in arrays]) for row in index], dtype=float)
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return Interval(point, float("nan"), float("nan"), n, float(n), block, 0)

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(
        point=point,
        lo=float(lo),
        hi=float(hi),
        n=n,
        n_effective=effective_sample_size(arrays[0]),
        block_length=block,
        resamples=int(draws.size),
    )


def sensitivity_to_block_length(
    columns: Sequence[np.ndarray],
    statistic: Callable[..., float],
    blocks: Sequence[int] = (1, 2, 3, 4, 6, 8),
    **kwargs,
) -> dict[int, Interval]:
    """The same interval at several block lengths.

    ``blocks=1`` redraws single weeks, which shows how much narrower ignoring
    the correlation would have made the interval.
    """
    n = int(np.asarray(columns[0]).size)
    return {
        b: block_bootstrap(columns, statistic, block=b, **kwargs)
        for b in blocks
        if b <= max(1, n // 2)
    }


def wilson_interval(successes: int, trials: int, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Range for a proportion, as a percentage. The method is Wilson's.

    Used for win rates, where redrawing blocks breaks down: a city that won
    every window it acted on wins every window in every redraw, so the interval
    collapses to [100, 100], which is the method failing at a boundary rather
    than certainty.

    Wilson handles 0% and 100% properly where the normal approximation does not.
    It assumes independent trials, which weekly wins are not, so it stays a
    little narrow. Read a 100% win rate as "at least this good".
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return (float(max(0.0, centre - half) * 100), float(min(1.0, centre + half) * 100))


# --------------------------------------------------------------------------- #
# The statistics this project reports, as functions the resampler can be given. #
# They live here rather than inline at each call site so the redrawn statistic  #
# and the reported estimate are the same code: two implementations drift apart, #
# and the interval stops bracketing the number printed beside it.               #
# --------------------------------------------------------------------------- #


def pct_improvement_paired(served: np.ndarray, frozen: np.ndarray) -> float:
    """The week-by-week premium: ``(1 - median(served / frozen)) * 100``."""
    ratio = served / frozen
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return float("nan")
    return float((1 - np.median(ratio)) * 100)


def pct_improvement_unpaired(served: np.ndarray, frozen: np.ndarray) -> float:
    """The across-replay figure: ``(1 - median(served) / median(frozen)) * 100``."""
    denominator = np.median(frozen)
    if denominator == 0:
        return float("nan")
    return float((1 - np.median(served) / denominator) * 100)


def win_rate(served: np.ndarray, frozen: np.ndarray) -> float:
    """Percentage of windows where the served model beat the frozen one."""
    if served.size == 0:
        return float("nan")
    return float(np.mean(served < frozen) * 100)


def mean_stat(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def differing(a: np.ndarray, b: np.ndarray, rtol: float = 1e-9) -> np.ndarray:
    """Mask of the positions where two paired series are not the same number.

    The median of paired ratios is the right statistic when the pairs are
    different measurements and the wrong one once most pairs are identical.
    Comparing two replay arms is the second case: a changed trigger leaves the
    serving model untouched in most weeks, so most ratios are 1.0, the median
    lands on 1.0, and the comparison reports no difference however badly the arm
    behaves in the weeks it does act.

    Kraków is the example. Against the trigger left alone, the floor arm's
    median ratio is +0.00% while its median error over the replay is 7% worse.
    Reporting only the first hides the harm; reporting only the second
    attributes a whole-replay gap to a handful of weeks. Both are published, and
    this supplies the mask for the second.

    Same conditioning ``retraining_value`` applies when it scores only the
    windows a retrained model served, and the same caveat: the subset is chosen
    by what the arm did, so it answers "when it acted, did it help" rather than
    "should it be switched on".
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.abs(a - b) > rtol * np.maximum(1.0, np.abs(b))
