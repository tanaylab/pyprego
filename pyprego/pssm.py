"""PSSM (Position-Specific Scoring Matrix) operations.

Functions for creating, manipulating, comparing, and analysing PSSMs,
mirroring the pssm-utils.R and pssm-cor.R modules from the R prego package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .types import NUCLEOTIDES, pssm_dataframe, pssm_to_array

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_with_prior(mat: np.ndarray, prior: float = 0.01) -> np.ndarray:
    """Add *prior* to each element, then row-normalise to probabilities."""
    m = mat.copy()
    if prior > 0:
        m = m + prior
    row_sums = m.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return m / row_sums


# ---------------------------------------------------------------------------
# Information content
# ---------------------------------------------------------------------------


def bits_per_pos(pssm: pd.DataFrame, prior: float = 0.01) -> np.ndarray:
    """Compute information content (bits) at each position.

    Matches the R ``bits_per_pos`` exactly:
    ``bits = log2(4) + sum(p * log2(p))`` per position, floored at 0.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame with columns ``A``, ``C``, ``G``, ``T``.
    prior : float
        Prior probability added to each value before normalisation.

    Returns
    -------
    np.ndarray
        1-D array of bits per position (max 2 for DNA).
    """
    mat = pssm_to_array(pssm)
    prob = _normalize_with_prior(mat, prior)
    # Avoid log(0) -- after adding prior, values should be > 0
    prob = np.clip(prob, 1e-300, None)
    bits = np.log2(4) + np.sum(prob * np.log2(prob), axis=1)
    return np.maximum(bits, 0.0)


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


def consensus_from_pssm(
    pssm: pd.DataFrame,
    single_thresh: float = 0.5,
    double_thresh: float = 0.75,
) -> str:
    """Derive a consensus sequence from a PSSM.

    At each position the dominant nucleotide is used if its probability exceeds
    *single_thresh*.  If two nucleotides together exceed *double_thresh* (but
    neither alone exceeds *single_thresh*), an IUPAC ambiguity code is used.
    Otherwise ``N`` is emitted.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame with columns ``A``, ``C``, ``G``, ``T``.
    single_thresh : float
        Threshold for a single-nucleotide call.
    double_thresh : float
        Threshold for a two-nucleotide ambiguity call.

    Returns
    -------
    str
        Consensus string.
    """
    mat = pssm_to_array(pssm)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prob = mat / row_sums

    iupac_pair = {
        frozenset("AC"): "M",
        frozenset("AG"): "R",
        frozenset("AT"): "W",
        frozenset("CG"): "S",
        frozenset("CT"): "Y",
        frozenset("GT"): "K",
    }

    consensus: list[str] = []
    for row in prob:
        order = np.argsort(row)[::-1]
        if row[order[0]] >= single_thresh:
            consensus.append(NUCLEOTIDES[order[0]])
        elif row[order[0]] + row[order[1]] >= double_thresh:
            pair = frozenset((NUCLEOTIDES[order[0]], NUCLEOTIDES[order[1]]))
            consensus.append(iupac_pair.get(pair, "N"))
        else:
            consensus.append("N")
    return "".join(consensus)


# ---------------------------------------------------------------------------
# Reverse complement
# ---------------------------------------------------------------------------


def pssm_rc(pssm: pd.DataFrame) -> pd.DataFrame:
    """Return the reverse-complement of a PSSM.

    Rows are reversed and columns swapped (A<->T, C<->G).

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame.

    Returns
    -------
    pd.DataFrame
        Reverse-complemented PSSM.
    """
    mat = pssm_to_array(pssm)
    # Reverse rows and swap: A(0)<->T(3), C(1)<->G(2)
    rc_mat = mat[::-1, [3, 2, 1, 0]].copy()
    return pssm_dataframe(rc_mat)


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


def pssm_trim(pssm: pd.DataFrame, bits_thresh: float = 0.1) -> pd.DataFrame:
    """Trim low-information positions from the edges of a PSSM.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame.
    bits_thresh : float
        Minimum bits per position to keep at the edges.

    Returns
    -------
    pd.DataFrame
        Trimmed PSSM with ``pos`` column reset.
    """
    bits = bits_per_pos(pssm)
    above = np.where(bits > bits_thresh)[0]
    if len(above) == 0:
        return pssm.iloc[0:0].copy()
    start, end = above[0], above[-1] + 1
    trimmed = pssm.iloc[start:end].copy()
    trimmed = trimmed.reset_index(drop=True)
    trimmed["pos"] = np.arange(len(trimmed))
    return trimmed


# Alias to match R name
trim_pssm = pssm_trim


# ---------------------------------------------------------------------------
# Add prior
# ---------------------------------------------------------------------------


def pssm_add_prior(pssm: pd.DataFrame, prior: float) -> pd.DataFrame:
    """Add a uniform prior to a PSSM and re-normalise.

    ``new = (pssm + prior) / rowSums(pssm + prior)``

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame.
    prior : float
        Prior value added to each cell.

    Returns
    -------
    pd.DataFrame
        New PSSM DataFrame with the prior applied.
    """
    mat = pssm_to_array(pssm)
    mat = _normalize_with_prior(mat, prior)
    return pssm_dataframe(mat)


# ---------------------------------------------------------------------------
# Theoretical scores
# ---------------------------------------------------------------------------


def pssm_theoretical_max(
    pssm: pd.DataFrame,
    prior: float = 0.01,
    regularization: float = 0.01,
) -> float:
    """Theoretical maximum log-likelihood score for a PSSM.

    Matches the R implementation: ``sum(log(regularization + rowMax(normalized_pssm)))``.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame.
    prior : float
        Prior added before normalisation.
    regularization : float
        Value added inside the log to prevent ``-Inf``.

    Returns
    -------
    float
        Maximum possible score.
    """
    prob = _normalize_with_prior(pssm_to_array(pssm), prior)
    return float(np.sum(np.log(regularization + np.max(prob, axis=1))))


def pssm_theoretical_min(
    pssm: pd.DataFrame,
    prior: float = 0.01,
    regularization: float = 0.01,
) -> float:
    """Theoretical minimum log-likelihood score for a PSSM.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame.
    prior : float
        Prior added before normalisation.
    regularization : float
        Value added inside the log to prevent ``-Inf``.

    Returns
    -------
    float
        Minimum possible score.
    """
    prob = _normalize_with_prior(pssm_to_array(pssm), prior)
    return float(np.sum(np.log(regularization + np.min(prob, axis=1))))


def pssm_quantile(
    pssm: pd.DataFrame,
    q: float,
    prior: float = 0.01,
    regularization: float = 0.01,
) -> float:
    """Quantile of the theoretical score distribution.

    Linearly interpolates between ``pssm_theoretical_min`` and
    ``pssm_theoretical_max``.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame.
    q : float
        Quantile (0 to 1).
    prior, regularization : float
        Passed through to the min/max functions.

    Returns
    -------
    float
        Score at the requested quantile.
    """
    theo_max = pssm_theoretical_max(pssm, prior, regularization)
    theo_min = pssm_theoretical_min(pssm, prior, regularization)
    return float(theo_min + q * (theo_max - theo_min))


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------


def pssm_concat(*pssms: pd.DataFrame, gap: int = 0) -> pd.DataFrame:
    """Concatenate multiple PSSMs vertically, optionally with a gap.

    Parameters
    ----------
    *pssms : pd.DataFrame
        PSSM DataFrames to concatenate.
    gap : int
        Number of uniform (0.25 each) positions to insert between
        successive PSSMs.

    Returns
    -------
    pd.DataFrame
        Combined PSSM with ``pos`` column renumbered from 0.
    """
    if not pssms:
        raise ValueError("At least one PSSM is required")
    gap_row = np.full((gap, 4), 0.25) if gap > 0 else None
    parts: list[np.ndarray] = []
    for i, p in enumerate(pssms):
        if i > 0 and gap_row is not None:
            parts.append(gap_row)
        parts.append(pssm_to_array(p))
    combined = np.vstack(parts)
    return pssm_dataframe(combined)


# Alias to match R name
concat_pssm = pssm_concat


# ---------------------------------------------------------------------------
# Correlation / KL divergence helpers (sliding window, matching R C++ code)
# ---------------------------------------------------------------------------


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two 1-D arrays."""
    n = len(x)
    if n == 0:
        return 0.0
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()
    sum_y2 = (y * y).sum()
    num = n * sum_xy - sum_x * sum_y
    den = np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return float(num / den) if den > 0 else 0.0


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation (Pearson on ranks, with average tie-breaking)."""
    rx = rankdata(x, method="average")
    ry = rankdata(y, method="average")
    return _pearson_correlation(rx, ry)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D(p || q)."""
    mask = (p > 0) & (q > 0)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def _max_pssm_score(
    pssm1: np.ndarray,
    pssm2: np.ndarray,
    method: str = "spearman",
    prior: float = 0.01,
) -> float:
    """Compute best similarity score between two raw PSSM matrices.

    Slides the shorter PSSM along the longer one and returns the best
    score (max for correlations, min for KL divergence). This exactly
    mirrors the C++ ``compute_max_pssm_correlation`` in PSSMCorelation.cpp.
    """
    n1, n2 = pssm1.shape[0], pssm2.shape[0]
    if n1 <= n2:
        pssm_s, pssm_l = pssm1, pssm2
    else:
        pssm_s, pssm_l = pssm2, pssm1

    window_size = pssm_s.shape[0]
    max_pos = pssm_l.shape[0]

    # Normalize shorter PSSM with prior
    vec_s = _normalize_with_prior(pssm_s, prior).ravel()

    if method == "kl":
        best_score = np.inf
    else:
        best_score = -1.0

    for start in range(max_pos - window_size + 1):
        window = pssm_l[start : start + window_size]
        vec_l = _normalize_with_prior(window, prior).ravel()

        if method == "kl":
            kl_sl = _kl_divergence(vec_s, vec_l)
            kl_ls = _kl_divergence(vec_l, vec_s)
            score = (kl_sl + kl_ls) / 2.0
            best_score = min(best_score, score)
        elif method == "pearson":
            score = _pearson_correlation(vec_s, vec_l)
            best_score = max(best_score, score)
        else:  # spearman
            score = _spearman_correlation(vec_s, vec_l)
            best_score = max(best_score, score)

    return float(best_score)


# ---------------------------------------------------------------------------
# Public correlation / divergence API
# ---------------------------------------------------------------------------


def pssm_cor(
    pssm1: pd.DataFrame,
    pssm2: pd.DataFrame,
    method: str = "spearman",
    prior: float = 0.01,
) -> float:
    """Compute correlation between two PSSMs at the best alignment.

    The shorter PSSM is slid along the longer one, and the maximum
    correlation (Spearman or Pearson) across all alignments is returned.

    Parameters
    ----------
    pssm1, pssm2 : pd.DataFrame
        PSSM DataFrames with columns ``A``, ``C``, ``G``, ``T``.
    method : str
        ``"spearman"`` (default) or ``"pearson"``.
    prior : float
        Prior added before normalisation.

    Returns
    -------
    float
        Maximum correlation across all alignments.

    Raises
    ------
    ValueError
        If either PSSM is empty.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    mat1 = pssm_to_array(pssm1)
    mat2 = pssm_to_array(pssm2)
    if mat1.shape[0] == 0 or mat2.shape[0] == 0:
        raise ValueError("PSSM matrices cannot be empty")

    return _max_pssm_score(mat1, mat2, method=method, prior=prior)


def pssm_diff(
    pssm1: pd.DataFrame,
    pssm2: pd.DataFrame,
    prior: float = 0.01,
) -> float:
    """Compute symmetric KL divergence between two PSSMs at the best alignment.

    The shorter PSSM is slid along the longer one, and the minimum
    (symmetric) KL divergence across all alignments is returned.

    Parameters
    ----------
    pssm1, pssm2 : pd.DataFrame
        PSSM DataFrames.
    prior : float
        Prior added before normalisation.

    Returns
    -------
    float
        Minimum symmetric KL divergence (lower = more similar).
    """
    mat1 = pssm_to_array(pssm1)
    mat2 = pssm_to_array(pssm2)
    if mat1.shape[0] == 0 or mat2.shape[0] == 0:
        raise ValueError("PSSM matrices cannot be empty")

    return _max_pssm_score(mat1, mat2, method="kl", prior=prior)


# ---------------------------------------------------------------------------
# Dataset-level correlation / divergence
# ---------------------------------------------------------------------------


def _pssm_dataset_score(
    pssm_dict1: dict[str, np.ndarray],
    pssm_dict2: dict[str, np.ndarray] | None = None,
    method: str = "spearman",
    prior: float = 0.01,
) -> np.ndarray:
    """Compute pairwise score matrix between two collections of raw PSSM arrays."""
    names1 = list(pssm_dict1.keys())
    mats1 = [pssm_dict1[n] for n in names1]

    if pssm_dict2 is None:
        # Within-set comparison
        n = len(mats1)
        if method == "kl":
            result = np.zeros((n, n), dtype=np.float64)
        else:
            result = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            if method == "kl":
                result[i, i] = 0.0
            else:
                result[i, i] = 1.0
            for j in range(i + 1, n):
                s = _max_pssm_score(mats1[i], mats1[j], method=method, prior=prior)
                result[i, j] = s
                result[j, i] = s
        return result
    else:
        names2 = list(pssm_dict2.keys())
        mats2 = [pssm_dict2[n] for n in names2]
        n1 = len(mats1)
        n2 = len(mats2)
        result = np.zeros((n1, n2), dtype=np.float64)
        for i in range(n1):
            for j in range(n2):
                result[i, j] = _max_pssm_score(mats1[i], mats2[j], method=method, prior=prior)
        return result


def _dataset_df_to_dict(dataset: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert a dataset DataFrame (with 'motif' column) to a dict of raw arrays."""
    if "motif" not in dataset.columns:
        raise ValueError("Dataset must have a 'motif' column")
    result: dict[str, np.ndarray] = {}
    for motif_name, group in dataset.groupby("motif", sort=False):
        result[str(motif_name)] = group[list(NUCLEOTIDES)].to_numpy(dtype=np.float64)
    return result


def pssm_dataset_cor(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame | None = None,
    method: str = "spearman",
    prior: float = 0.01,
) -> pd.DataFrame:
    """Compute pairwise correlation matrix for PSSM datasets.

    Parameters
    ----------
    dataset1 : pd.DataFrame
        DataFrame with columns ``motif``, ``A``, ``C``, ``G``, ``T``.
    dataset2 : pd.DataFrame | None
        Optional second dataset. If ``None``, computes within-set
        correlations.
    method : str
        ``"spearman"`` or ``"pearson"``.
    prior : float
        Prior for normalisation.

    Returns
    -------
    pd.DataFrame
        Correlation matrix as a DataFrame with motif-name row and column
        labels.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    dict1 = _dataset_df_to_dict(dataset1)
    names1 = list(dict1.keys())

    if dataset2 is None:
        score_mat = _pssm_dataset_score(dict1, method=method, prior=prior)
        return pd.DataFrame(score_mat, index=names1, columns=names1)
    else:
        dict2 = _dataset_df_to_dict(dataset2)
        names2 = list(dict2.keys())
        score_mat = _pssm_dataset_score(dict1, dict2, method=method, prior=prior)
        return pd.DataFrame(score_mat, index=names1, columns=names2)


def pssm_dataset_diff(
    dataset1: pd.DataFrame,
    dataset2: pd.DataFrame | None = None,
    prior: float = 0.01,
) -> pd.DataFrame:
    """Compute pairwise KL divergence matrix for PSSM datasets.

    Parameters
    ----------
    dataset1 : pd.DataFrame
        DataFrame with columns ``motif``, ``A``, ``C``, ``G``, ``T``.
    dataset2 : pd.DataFrame | None
        Optional second dataset.
    prior : float
        Prior for normalisation.

    Returns
    -------
    pd.DataFrame
        KL divergence matrix as a DataFrame.
    """
    dict1 = _dataset_df_to_dict(dataset1)
    names1 = list(dict1.keys())

    if dataset2 is None:
        score_mat = _pssm_dataset_score(dict1, method="kl", prior=prior)
        return pd.DataFrame(score_mat, index=names1, columns=names1)
    else:
        dict2 = _dataset_df_to_dict(dataset2)
        names2 = list(dict2.keys())
        score_mat = _pssm_dataset_score(dict1, dict2, method="kl", prior=prior)
        return pd.DataFrame(score_mat, index=names1, columns=names2)


# ---------------------------------------------------------------------------
# Match against a motif database
# ---------------------------------------------------------------------------


def pssm_match(
    pssm: pd.DataFrame,
    motifs: dict[str, pd.DataFrame] | pd.DataFrame,
    best: bool = False,
    method: str = "spearman",
    prior: float = 0.01,
) -> pd.DataFrame | str:
    """Match a PSSM against a motif database.

    Parameters
    ----------
    pssm : pd.DataFrame
        Query PSSM DataFrame.
    motifs : dict[str, pd.DataFrame] | pd.DataFrame
        Either a dict mapping motif names to PSSM DataFrames, or a
        DataFrame with a ``motif`` column (long format).
    best : bool
        If ``True``, return only the best-matching motif name.
    method : str
        ``"spearman"``, ``"pearson"``, or ``"kl"``.
    prior : float
        Prior for normalisation.

    Returns
    -------
    pd.DataFrame | str
        If *best* is ``True``, a string with the best matching motif
        name.  Otherwise a DataFrame with columns ``motif`` and either
        ``cor`` or ``kl``, sorted by decreasing similarity (correlations)
        or increasing divergence (KL).
    """
    if method not in ("spearman", "pearson", "kl"):
        raise ValueError(f"method must be 'spearman', 'pearson', or 'kl', got {method!r}")

    query_mat = pssm_to_array(pssm)

    # Convert dict to individual arrays
    if isinstance(motifs, pd.DataFrame):
        motif_dict = _dataset_df_to_dict(motifs)
    else:
        motif_dict = {name: pssm_to_array(p) for name, p in motifs.items()}

    results: list[tuple[str, float]] = []
    for name, db_mat in motif_dict.items():
        score = _max_pssm_score(query_mat, db_mat, method=method, prior=prior)
        results.append((name, score))

    col_name = "kl" if method == "kl" else "cor"
    df = pd.DataFrame(results, columns=["motif", col_name])

    if method == "kl":
        df = df.sort_values(col_name, ascending=True).reset_index(drop=True)
    else:
        df = df.sort_values(col_name, ascending=False).reset_index(drop=True)

    if best:
        return str(df["motif"].iloc[0])

    return df
