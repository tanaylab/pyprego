"""Genomic integration layer (optional pymisha dependency).

Mirrors the misha.R module from R prego. Provides functions that operate
on genomic intervals and tracks, extracting sequences and computing PWM
scores over genomic regions.

All functions in this module require the ``pymisha`` package to be installed.
Import will succeed regardless, but functions will raise ``ImportError``
with a clear message at call time if pymisha is not available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def _require_pymisha():
    """Raise ImportError if pymisha is not available."""
    try:
        import pymisha  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The genomic module requires pymisha. Install it with: pip install pymisha"
        ) from exc


def _normalize_intervals(intervals: pd.DataFrame, size: int) -> pd.DataFrame:
    """Normalize intervals to a given size by centering.

    Resizes each interval to *size* bp around its center. This mirrors
    ``pymisha.gintervals_normalize`` but works without a live misha database
    (no chromosome-boundary clamping).

    Parameters
    ----------
    intervals : pd.DataFrame
        Intervals with ``chrom``, ``start``, ``end`` columns.
    size : int
        Target interval size in base pairs.

    Returns
    -------
    pd.DataFrame
        Intervals resized to *size* bp around their centers.
    """
    ivs = intervals.copy()
    centers = (ivs["start"] + ivs["end"]) // 2
    half = size // 2
    ivs["start"] = centers - half
    ivs["end"] = ivs["start"] + size
    return ivs


def intervals_to_seq(
    intervals: pd.DataFrame,
    size: int | None = None,
) -> list[str]:
    """Extract DNA sequences for genomic intervals.

    Mirrors the R ``intervals_to_seq()`` function. Uses
    ``pymisha.gseq_extract`` to retrieve sequences and optionally normalizes
    interval sizes around their centers first.

    Parameters
    ----------
    intervals : pd.DataFrame
        Genomic intervals with columns ``chrom``, ``start``, ``end``.
    size : int | None
        If provided, normalize intervals to this size (bp) around their
        center before extraction. ``None`` keeps original intervals.

    Returns
    -------
    list[str]
        List of uppercase DNA sequences, one per interval.
    """
    _require_pymisha()
    import pymisha as pm

    ivs = intervals.copy()

    if size is not None:
        ivs = pm.gintervals_normalize(ivs, size)

    sequences = pm.gseq_extract(ivs)

    # Ensure uppercase strings
    return [s.upper() for s in sequences]


def gextract_pwm(
    intervals: pd.DataFrame,
    pssm: pd.DataFrame,
    *,
    spat: pd.DataFrame | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
    func: str = "logSumExp",
    size: int | None = None,
) -> np.ndarray:
    """Extract PWM scores for genomic intervals.

    Extracts sequences from the genome for the given intervals, then scores
    each with ``compute_pwm``. Mirrors the R ``gextract_pwm()`` function.

    Parameters
    ----------
    intervals : pd.DataFrame
        Genomic intervals with ``chrom``, ``start``, ``end`` columns.
    pssm : pd.DataFrame
        PSSM DataFrame with columns A, C, G, T.
    spat : pd.DataFrame | None
        Spatial model DataFrame (``bin``, ``spat_factor`` columns).
    bidirect : bool
        Score both orientations.
    prior : float
        Uniform prior added to PSSM probabilities.
    func : str
        Aggregation function: ``"logSumExp"`` or ``"max"``.
    size : int | None
        Normalize intervals to this size before extraction.

    Returns
    -------
    np.ndarray
        1-D array of PWM scores, one per interval.
    """
    from .compute import compute_pwm

    sequences = intervals_to_seq(intervals, size=size)
    return compute_pwm(
        sequences, pssm, spat=spat, bidirect=bidirect, prior=prior, func=func
    )


def gextract_pwm_quantile(
    intervals: pd.DataFrame,
    pssm: pd.DataFrame,
    quantiles: np.ndarray | list[float],
    *,
    spat: pd.DataFrame | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
    func: str = "logSumExp",
    size: int | None = None,
    bg_intervals: pd.DataFrame | None = None,
    n_sequences: int = 10_000,
) -> np.ndarray:
    """Extract quantiles of PWM scores for genomic intervals.

    Computes PWM scores for the input intervals and maps them to quantiles
    estimated from background intervals. Mirrors the R
    ``gextract_pwm.quantile()`` function.

    Parameters
    ----------
    intervals : pd.DataFrame
        Genomic intervals.
    pssm : pd.DataFrame
        PSSM DataFrame (A, C, G, T columns).
    quantiles : array-like
        Quantile breakpoints for the background CDF (e.g. ``np.arange(0, 1.01, 0.01)``).
    spat : pd.DataFrame | None
        Spatial model.
    bidirect : bool
        Score both orientations.
    prior : float
        Uniform prior.
    func : str
        Aggregation function.
    size : int | None
        Normalize intervals to this size.
    bg_intervals : pd.DataFrame | None
        Background intervals for quantile estimation. If ``None``,
        ``pymisha.gintervals_random`` is used to sample *n_sequences*
        random intervals of the same size as the input intervals.
    n_sequences : int
        Number of background sequences to sample when *bg_intervals* is
        ``None``.

    Returns
    -------
    np.ndarray
        1-D array of quantile values (0--1) per interval.
    """
    _require_pymisha()
    import pymisha as pm

    # Determine interval size
    if size is not None:
        interval_size = size
    else:
        interval_size = int((intervals["end"] - intervals["start"]).iloc[0])

    # Score the input intervals
    scores = gextract_pwm(
        intervals, pssm, spat=spat, bidirect=bidirect, prior=prior,
        func=func, size=size,
    )

    # Build background distribution
    if bg_intervals is None:
        bg_intervals = pm.gintervals_random(interval_size, n_sequences)

    bg_scores = gextract_pwm(
        bg_intervals, pssm, spat=spat, bidirect=bidirect, prior=prior, func=func,
    )

    # Compute quantile breakpoints from background
    quantiles_arr = np.asarray(quantiles, dtype=np.float64)
    bg_quantile_values = np.nanquantile(bg_scores, quantiles_arr)

    # Map each input score to its quantile via interpolation
    result = np.interp(scores, bg_quantile_values, quantiles_arr)
    return result


def gextract_local_pwm(
    intervals: pd.DataFrame,
    pssm: pd.DataFrame,
    *,
    spat: pd.DataFrame | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
    size: int | None = None,
) -> np.ndarray:
    """Extract per-position PWM scores for genomic intervals.

    Extracts sequences from the genome and computes per-position PWM scores
    using ``compute_local_pwm``. Mirrors the R ``gextract.local_pwm()``
    function.

    Parameters
    ----------
    intervals : pd.DataFrame
        Genomic intervals.
    pssm : pd.DataFrame
        PSSM DataFrame (A, C, G, T columns).
    spat : pd.DataFrame | None
        Spatial model.
    bidirect : bool
        Score both orientations.
    prior : float
        Uniform prior.
    size : int | None
        Normalize intervals to this size.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_intervals, seq_length)`` with per-position
        scores. Positions where the PSSM does not fit contain NaN.
    """
    from .compute import compute_local_pwm

    sequences = intervals_to_seq(intervals, size=size)
    return compute_local_pwm(
        sequences, pssm, spat=spat, bidirect=bidirect, prior=prior,
    )


def gintervals_center_by_pssm(
    intervals: pd.DataFrame,
    pssm: pd.DataFrame,
    size: int,
    *,
    spat: pd.DataFrame | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
) -> pd.DataFrame:
    """Center intervals by the position with maximum PSSM score.

    For each interval, computes per-position PWM scores and finds the
    position with the highest score. The interval is then re-centered on
    that position and normalized to *size* bp. Mirrors the R
    ``gintervals.center_by_pssm()`` function.

    Parameters
    ----------
    intervals : pd.DataFrame
        Genomic intervals.
    pssm : pd.DataFrame
        PSSM DataFrame (A, C, G, T columns).
    size : int
        Target interval size after re-centering.
    spat : pd.DataFrame | None
        Spatial model.
    bidirect : bool
        Score both orientations.
    prior : float
        Uniform prior.

    Returns
    -------
    pd.DataFrame
        Re-centered intervals with ``chrom``, ``start``, ``end`` (and any
        extra columns from the input).
    """
    _require_pymisha()
    import pymisha as pm

    # Compute per-position scores
    local_pwm = gextract_local_pwm(
        intervals, pssm, spat=spat, bidirect=bidirect, prior=prior,
    )

    # Find the position of maximum score per interval
    max_positions = np.nanargmax(local_pwm, axis=1)

    # Shift intervals so the max-score position becomes the center
    result = intervals.copy()
    result["start"] = result["start"] + max_positions
    result["end"] = result["start"] + 1

    # Normalize to target size
    result = pm.gintervals_normalize(result, size)

    # Ensure standard column order
    cols = ["chrom", "start", "end"] + [
        c for c in result.columns if c not in ("chrom", "start", "end")
    ]
    return result[cols]
