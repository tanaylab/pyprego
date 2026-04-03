"""K-mer generation, scoring, and PSSM conversion.

Mirrors kmers.R and kmer-regression.R from the R prego package.
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .types import NUCLEOTIDES, pssm_dataframe, pssm_to_array

if TYPE_CHECKING:
    pass


def generate_kmers(
    k: int,
    alphabet: str = "ACGT",
    max_gap: int = 0,
    min_gap: int = 0,
) -> list[str]:
    """Generate all possible DNA k-mers of length *k*, optionally with gaps.

    Gaps are represented by ``'N'`` at certain positions in the k-mer. When
    *max_gap* > 0, the function generates k-mers where 1..max_gap contiguous
    positions are replaced with ``'N'``, at every possible offset within the
    k-mer.

    This mirrors the R ``generate_kmers()`` function in ``kmers.R``.

    Parameters
    ----------
    k : int
        K-mer length (number of positions, including gap positions). Must be >= 1.
    alphabet : str
        Nucleotide alphabet (default ``"ACGT"``).
    max_gap : int
        Maximum number of contiguous gap (wildcard ``N``) positions. Default 0
        means no gaps.
    min_gap : int
        Minimum gap length. Default 0.

    Returns
    -------
    list[str]
        List of k-mers. With ``max_gap=0`` this is all ``4^k`` standard k-mers.
        With gaps, also includes gapped variants.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if min_gap < 0:
        raise ValueError(f"min_gap must be >= 0, got {min_gap}")
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")
    if max_gap > k:
        raise ValueError(f"max_gap ({max_gap}) must be <= k ({k})")
    if max_gap < min_gap:
        raise ValueError(f"max_gap ({max_gap}) must be >= min_gap ({min_gap})")

    letters = list(alphabet)
    # Generate all base k-mers (no gaps)
    base_kmers = ["".join(p) for p in itertools.product(letters, repeat=k)]

    gap_kmers: list[str] = []
    for g in range(min_gap, max_gap + 1):
        if g == 0:
            continue
        gap_str = "N" * g
        for pos in range(k - g + 1):
            for km in base_kmers:
                gapped = km[:pos] + gap_str + km[pos + g :]
                gap_kmers.append(gapped)

    if min_gap > 0:
        # Only return gapped k-mers (matching R behavior)
        return gap_kmers

    return list(dict.fromkeys(base_kmers + gap_kmers))


def kmer_matrix(
    sequences: list[str] | np.ndarray,
    kmers: list[str] | int,
    max_gap: int = 0,
) -> pd.DataFrame:
    """Count k-mer occurrences in each sequence.

    If *kmers* is an integer, it is treated as the k-mer length and all
    standard k-mers (plus gapped variants if *max_gap* > 0) are generated.
    If *kmers* is a list of strings, those exact k-mers are counted.

    For k-mers containing ``'N'`` (wildcard), any nucleotide at that position
    is considered a match.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences.
    kmers : list[str] | int
        Either a k-mer length (int) or an explicit list of k-mer strings.
    max_gap : int
        Maximum gap length for auto-generated k-mers (only used when *kmers*
        is an int). Default 0.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape ``(n_sequences, n_kmers)`` with occurrence counts.
        Columns are the k-mer strings.
    """
    if isinstance(kmers, (int, np.integer)):
        kmer_list = generate_kmers(int(kmers), max_gap=max_gap)
    else:
        kmer_list = list(kmers)

    if len(kmer_list) == 0:
        return pd.DataFrame(index=range(len(sequences)))

    k = len(kmer_list[0])

    # Separate pure k-mers (no N) from gapped k-mers
    pure_kmers = []
    gapped_kmers = []
    for km in kmer_list:
        if "N" in km:
            gapped_kmers.append(km)
        else:
            pure_kmers.append(km)

    kmer_to_idx = {km: i for i, km in enumerate(kmer_list)}
    n_kmers = len(kmer_list)
    n_seqs = len(sequences)

    counts = np.zeros((n_seqs, n_kmers), dtype=np.int32)

    # Build regex patterns for gapped k-mers (N matches any nucleotide)
    gapped_patterns: list[tuple[re.Pattern[str], int]] | None = None
    if gapped_kmers:
        gapped_patterns = []
        for km in gapped_kmers:
            # Build regex: replace N with [ACGT], other chars literal
            pat = "".join("[ACGT]" if c == "N" else c for c in km)
            gapped_patterns.append((re.compile(pat), kmer_to_idx[km]))

    pure_kmer_set = {km: kmer_to_idx[km] for km in pure_kmers}

    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for j in range(len(seq) - k + 1):
            sub = seq[j : j + k]

            # Check pure k-mers (direct hash lookup)
            idx = pure_kmer_set.get(sub)
            if idx is not None:
                counts[i, idx] += 1

            # Check gapped k-mers
            if gapped_patterns:
                for pat, kidx in gapped_patterns:
                    if pat.fullmatch(sub):
                        counts[i, kidx] += 1

    return pd.DataFrame(counts, columns=kmer_list)


def screen_kmers(
    sequences: list[str] | np.ndarray,
    response: np.ndarray,
    kmer_len: int | None = None,
    kmers: list[str] | None = None,
    max_gap: int = 0,
    min_gap: int = 0,
    seed: int | None = None,
    min_cor: float = 0.0,
) -> pd.DataFrame:
    """Screen k-mers for correlation with response variable(s).

    For each k-mer, compute its frequency across sequences and correlate with
    *response*. This mirrors the R ``screen_kmers()`` function.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences.
    response : np.ndarray
        Response variable(s). Shape ``(n_sequences,)`` for a single response,
        or ``(n_sequences, n_responses)`` for multiple response columns.
    kmer_len : int | None
        K-mer length. Either this or *kmers* must be provided.
    kmers : list[str] | None
        Explicit list of k-mers to screen. If given, overrides *kmer_len*.
    max_gap : int
        Maximum gap length. Default 0.
    min_gap : int
        Minimum gap length. Default 0.
    seed : int | None
        Random seed (for reproducibility; currently only sets numpy seed).
    min_cor : float
        Minimum absolute correlation to include in the result. Default 0.0
        (include all).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:

        - ``kmer``: the k-mer string
        - ``max_r2``: maximum R^2 across response columns
        - ``avg_n``: average count of the k-mer per sequence
        - ``avg_var``: variance of the count across sequences
        - One column per response variable with the Pearson correlation

        Sorted by ``max_r2`` descending.

    Raises
    ------
    ValueError
        If neither *kmer_len* nor *kmers* is provided, or if dimensions
        mismatch.
    """
    if seed is not None:
        np.random.seed(seed)

    if kmers is None and kmer_len is None:
        raise ValueError("Either kmer_len or kmers must be provided")

    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 1:
        response = response.reshape(-1, 1)
        resp_names = ["r0"]
    else:
        resp_names = [f"r{i}" for i in range(response.shape[1])]

    n_seqs = len(sequences)
    if response.shape[0] != n_seqs:
        raise ValueError(
            f"Number of sequences ({n_seqs}) != number of response rows ({response.shape[0]})"
        )

    n_resp = response.shape[1]

    # Build k-mer count matrix
    if kmers is not None:
        km_df = kmer_matrix(sequences, kmers)
        kmer_list = list(km_df.columns)
    else:
        assert kmer_len is not None
        kmer_list_gen = generate_kmers(kmer_len, max_gap=max_gap, min_gap=min_gap)
        km_df = kmer_matrix(sequences, kmer_list_gen)
        kmer_list = list(km_df.columns)

    counts = km_df.to_numpy(dtype=np.float64)

    # Compute statistics for each k-mer
    # Mean and variance of counts
    avg_n = counts.mean(axis=0)
    avg_var = counts.var(axis=0)

    # Response means and variances
    resp_mean = response.mean(axis=0)
    resp_var = response.var(axis=0)

    # Correlations: for each k-mer and each response column, compute Pearson r
    # Pearson r = cov(x, y) / (std(x) * std(y))
    results = []
    for ki in range(len(kmer_list)):
        x = counts[:, ki]
        x_mean = avg_n[ki]
        x_var = avg_var[ki]

        if x_var < 1e-15:
            # No variance in k-mer counts -> skip or set correlation to 0
            continue

        cors = np.zeros(n_resp)
        max_r2 = 0.0
        for ri in range(n_resp):
            if resp_var[ri] < 1e-15:
                cors[ri] = 0.0
                continue
            cov_xy = np.mean(x * response[:, ri]) - x_mean * resp_mean[ri]
            r = cov_xy / np.sqrt(x_var * resp_var[ri])
            cors[ri] = r
            r2 = r * r
            if r2 > max_r2:
                max_r2 = r2

        if min_cor > 0 and max_r2 < min_cor * min_cor:
            continue

        row = {
            "kmer": kmer_list[ki],
            "max_r2": max_r2,
            "avg_n": x_mean,
            "avg_var": x_var,
        }
        for ri in range(n_resp):
            row[resp_names[ri]] = cors[ri]
        results.append(row)

    if not results:
        cols = ["kmer", "max_r2", "avg_n", "avg_var"] + resp_names
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(results)
    df = df.sort_values("max_r2", ascending=False).reset_index(drop=True)
    return df


def kmers_to_pssm(
    kmer: str | list[str],
    prior: float = 0.01,
) -> pd.DataFrame:
    """Convert k-mer string(s) to PSSM DataFrame(s).

    Each position gets high probability for the matching nucleotide and low
    (``prior``) for others. For gap positions (``N``), use uniform 0.25.

    This mirrors the R ``kmers_to_pssm()`` function, which accepts a vector of
    k-mers and returns a combined DataFrame with a ``kmer`` column.

    Parameters
    ----------
    kmer : str | list[str]
        K-mer string or list of k-mers. May contain ``N`` for wildcard positions.
    prior : float
        Prior probability for non-matching nucleotides. Default 0.01.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``kmer``, ``pos``, ``A``, ``C``, ``G``, ``T``.
        Each row sums to 1 across the four nucleotides.

    Raises
    ------
    ValueError
        If k-mer contains invalid characters.
    """
    if isinstance(kmer, str):
        kmer_list = [kmer]
    else:
        kmer_list = list(kmer)

    # Validate
    for km in kmer_list:
        if re.search(r"[^ACGTN]", km.upper()):
            raise ValueError(f"kmers must have only valid nucleotides, got '{km}'")

    nuc_idx = {n: i for i, n in enumerate(NUCLEOTIDES)}  # A=0, C=1, G=2, T=3

    all_rows = []
    for km in kmer_list:
        km = km.upper()
        L = len(km)
        mat = np.full((L, 4), prior, dtype=np.float64)

        for pos, ch in enumerate(km):
            if ch == "N":
                mat[pos, :] = 0.25
            else:
                mat[pos, nuc_idx[ch]] = 1.0

        # Renormalize each row to sum to 1
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = mat / row_sums

        for pos in range(L):
            all_rows.append(
                {
                    "kmer": km,
                    "pos": pos + 1,  # 1-based like R
                    "A": mat[pos, 0],
                    "C": mat[pos, 1],
                    "G": mat[pos, 2],
                    "T": mat[pos, 3],
                }
            )

    return pd.DataFrame(all_rows, columns=["kmer", "pos", "A", "C", "G", "T"])


def pssm_to_kmer(
    pssm: pd.DataFrame,
    kmer_length: int | None = None,
    pos_bits_thresh: float | None = 0.5,
    prior: float = 0.01,
) -> str:
    """Convert PSSM back to a k-mer string.

    Finds the window of *kmer_length* positions with the highest total
    information content, then at each position uses the dominant nucleotide.
    If *pos_bits_thresh* is set, positions below the threshold are replaced
    with ``N`` (wildcard).

    This mirrors the R ``pssm_to_kmer()`` function.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame with columns ``A``, ``C``, ``G``, ``T``.
    kmer_length : int | None
        Length of the returned k-mer. If ``None``, uses the full PSSM length.
    pos_bits_thresh : float | None
        Minimum information content (bits) per position. Positions below
        this threshold are set to ``N``. If ``None``, all positions use
        the dominant nucleotide.
    prior : float
        Prior added before computing bits. Default 0.01.

    Returns
    -------
    str
        K-mer string, possibly containing ``N`` for low-information positions.

    Raises
    ------
    ValueError
        If PSSM has fewer rows than *kmer_length*.
    """
    mat = pssm_to_array(pssm)  # (L, 4)
    L = mat.shape[0]

    if kmer_length is None:
        kmer_length = L

    if L < kmer_length:
        raise ValueError(f"PSSM has {L} rows but kmer_length is {kmer_length}")

    # Compute bits per position (with prior)
    mat_with_prior = mat + prior
    mat_norm = mat_with_prior / mat_with_prior.sum(axis=1, keepdims=True)
    mat_norm = np.clip(mat_norm, 1e-10, None)
    entropy = -np.sum(mat_norm * np.log2(mat_norm), axis=1)
    bits = 2.0 - entropy
    bits = np.nan_to_num(bits, nan=0.0)

    # Rolling sum to find best window
    if L == kmer_length:
        best_pos = 0
    else:
        rollsum = np.convolve(bits, np.ones(kmer_length), mode="valid")
        best_pos = int(np.argmax(rollsum))

    # Extract the window
    window_mat = mat[best_pos : best_pos + kmer_length, :]

    # Dominant nucleotide at each position
    nucs = list(NUCLEOTIDES)
    kmer_chars = [nucs[j] for j in np.argmax(window_mat, axis=1)]

    # Apply bits threshold
    if pos_bits_thresh is not None:
        window_with_prior = window_mat + prior
        window_norm = window_with_prior / window_with_prior.sum(axis=1, keepdims=True)
        window_norm = np.clip(window_norm, 1e-10, None)
        window_entropy = -np.sum(window_norm * np.log2(window_norm), axis=1)
        window_bits = 2.0 - window_entropy
        window_bits = np.nan_to_num(window_bits, nan=0.0)

        kmer_chars = [
            c if window_bits[i] > pos_bits_thresh else "N"
            for i, c in enumerate(kmer_chars)
        ]

    return "".join(kmer_chars)
