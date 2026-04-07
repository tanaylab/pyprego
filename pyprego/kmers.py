"""K-mer generation, scoring, and PSSM conversion.

Mirrors kmers.R and kmer-regression.R from the R prego package.
"""

from __future__ import annotations

import itertools
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._fast_encode import encode_sequences_fast
from .types import NUCLEOTIDES, pssm_to_array

if TYPE_CHECKING:
    pass

# Try importing C extension for fast k-mer counting
try:
    from pyprego._pyprego import kmer_matrix as _kmer_matrix_c
except (ImportError, AttributeError):
    _kmer_matrix_c = None

# Try importing C extension for fast k-mer screening
try:
    from pyprego._pyprego import screen_kmers as _screen_kmers_c
except (ImportError, AttributeError):
    _screen_kmers_c = None

# Powers of 4 for base-4 integer hashing of k-mers
_BASE4_CHARS = {"A": 0, "C": 1, "G": 2, "T": 3}


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
    # Generate all base k-mers (no gaps) - use NumPy for speed when k is large
    if k <= 10:
        base_kmers = ["".join(p) for p in itertools.product(letters, repeat=k)]
    else:
        # For large k, itertools.product is still fine
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


def _kmer_to_int(kmer: str) -> int:
    """Convert a pure (no-N) k-mer string to a base-4 integer."""
    val = 0
    for ch in kmer:
        val = val * 4 + _BASE4_CHARS[ch]
    return val


def _windows_to_ints(encoded: np.ndarray, k: int) -> np.ndarray:
    """Convert encoded sliding windows to base-4 integers.

    Parameters
    ----------
    encoded : np.ndarray
        Integer-encoded sequences, shape (N, L), values 0-3, -1 for N.

    Returns
    -------
    np.ndarray
        Shape (N, num_windows). Values are base-4 ints for valid windows,
        -1 for windows containing any N-base.
    """
    N, L = encoded.shape
    num_wins = L - k + 1
    if num_wins <= 0:
        return np.empty((N, 0), dtype=np.int64)

    # Sliding window indices: (num_wins, k)
    win_idx = np.arange(k)[None, :] + np.arange(num_wins)[:, None]
    windows = encoded[:, win_idx]  # (N, num_wins, k)

    # Mark windows with any invalid base
    has_invalid = np.any(windows < 0, axis=2)  # (N, num_wins)

    # Compute base-4 integer for each window
    powers = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)  # [4^(k-1), ..., 4^0]
    win_ints = (windows.astype(np.int64) * powers[None, None, :]).sum(axis=2)  # (N, num_wins)
    win_ints[has_invalid] = -1

    return win_ints


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
    # ── Try fast C extension path ──
    # When kmers is an integer (generate all k-mers of that length),
    # the C extension can handle the complete pipeline.
    if _kmer_matrix_c is not None and isinstance(kmers, (int, np.integer)):
        k_int = int(kmers)
        seq_list = [
            s.upper() if isinstance(s, str) else str(s)
            for s in (sequences.tolist() if isinstance(sequences, np.ndarray) else sequences)
        ]
        try:
            arr, names = _kmer_matrix_c(seq_list, k_int, max_gap)
            return pd.DataFrame(arr, columns=names)
        except Exception:
            pass  # fall through to Python implementation

    kmer_list = generate_kmers(int(kmers), max_gap=max_gap) if isinstance(kmers, (int, np.integer)) else list(kmers)

    if len(kmer_list) == 0:
        return pd.DataFrame(index=range(len(sequences)))

    k = len(kmer_list[0])
    n_seqs = len(sequences)
    n_kmers = len(kmer_list)

    # Separate pure k-mers (no N) from gapped k-mers
    pure_kmers = [km for km in kmer_list if "N" not in km]
    gapped_kmers = [km for km in kmer_list if "N" in km]
    kmer_to_idx = {km: i for i, km in enumerate(kmer_list)}

    # Encode sequences once
    encoded = encode_sequences_fast([s.upper() if isinstance(s, str) else s for s in sequences])

    counts = np.zeros((n_seqs, n_kmers), dtype=np.int32)

    # ── Pure k-mers: vectorized base-4 hashing ──
    if pure_kmers:
        max_int = 4**k
        int_to_col = np.full(max_int, -1, dtype=np.int32)
        for km in pure_kmers:
            int_to_col[_kmer_to_int(km)] = kmer_to_idx[km]

        # Precompute which base-4 ints map to which output columns
        active_ints = np.where(int_to_col >= 0)[0]
        active_cols = int_to_col[active_ints]

        # Convert all sliding windows to base-4 ints: (N, num_wins)
        win_ints = _windows_to_ints(encoded, k)

        # Per-sequence bincount — fast and memory-efficient
        for i in range(n_seqs):
            row = win_ints[i]
            valid = row[row >= 0]
            if len(valid) == 0:
                continue
            bc = np.bincount(valid, minlength=max_int)
            counts[i, active_cols] = bc[active_ints]

    # ── Gapped k-mers: vectorized mask-based matching ──
    if gapped_kmers:
        N_seq, L = encoded.shape
        num_wins = L - k + 1
        if num_wins > 0:
            # Sliding window indices
            win_idx = np.arange(k)[None, :] + np.arange(num_wins)[:, None]
            windows = encoded[:, win_idx]  # (N, num_wins, k)

            for km in gapped_kmers:
                col_idx = kmer_to_idx[km]
                # Build mask: which positions in the k-mer are fixed (not N)
                fixed_positions = []
                fixed_values = []
                for pos_i, ch in enumerate(km):
                    if ch != "N":
                        fixed_positions.append(pos_i)
                        fixed_values.append(_BASE4_CHARS[ch])

                fixed_pos = np.array(fixed_positions, dtype=int)
                fixed_val = np.array(fixed_values, dtype=np.int8)

                # Extract just the fixed positions from all windows
                fixed_windows = windows[:, :, fixed_pos]  # (N, num_wins, n_fixed)

                # A window matches if all fixed positions match their expected values
                # AND no fixed position contains an N-base (-1)
                matches = np.all(fixed_windows == fixed_val[None, None, :], axis=2)
                counts[:, col_idx] = matches.sum(axis=1)

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
        raise ValueError(f"Number of sequences ({n_seqs}) != number of response rows ({response.shape[0]})")

    n_resp = response.shape[1]

    # ── Fast C extension path (pure and gapped k-mers) ──
    if (
        _screen_kmers_c is not None
        and kmer_len is not None
        and kmers is None
        and seed is None
    ):
        try:
            encoded = encode_sequences_fast(
                [s.upper() if isinstance(s, str) else str(s) for s in (sequences.tolist() if isinstance(sequences, np.ndarray) else sequences)]
            )
            encoded = np.ascontiguousarray(encoded)
            response_c = np.ascontiguousarray(response)
            names, max_r2_arr, corr_arr, avg_n_arr, avg_var_arr = _screen_kmers_c(
                encoded, response_c, kmer_len, min_cor, min_gap, max_gap
            )
            if len(names) == 0:
                cols = ["kmer", "max_r2", "avg_n", "avg_var"] + resp_names
                return pd.DataFrame(columns=cols)
            result_data = {
                "kmer": names,
                "max_r2": max_r2_arr,
                "avg_n": avg_n_arr,
                "avg_var": avg_var_arr,
            }
            for ri in range(n_resp):
                result_data[resp_names[ri]] = corr_arr[:, ri]
            df = pd.DataFrame(result_data)
            return df.sort_values("max_r2", ascending=False).reset_index(drop=True)
        except Exception:
            pass  # fall through to Python implementation

    # Build k-mer count matrix
    if kmers is not None:
        km_df = kmer_matrix(sequences, kmers)
        kmer_list = list(km_df.columns)
    elif min_gap > 0:
        assert kmer_len is not None
        kmer_list_gen = generate_kmers(kmer_len, max_gap=max_gap, min_gap=min_gap)
        km_df = kmer_matrix(sequences, kmer_list_gen)
        kmer_list = list(km_df.columns)
    else:
        assert kmer_len is not None
        km_df = kmer_matrix(sequences, kmer_len, max_gap=max_gap)
        kmer_list = list(km_df.columns)

    counts = km_df.to_numpy(dtype=np.float64)
    counts.shape[1]

    # ── Vectorized statistics and correlations ──
    avg_n = counts.mean(axis=0)  # (n_km,)
    avg_var = counts.var(axis=0)  # (n_km,)

    resp_mean = response.mean(axis=0)  # (n_resp,)
    resp_var = response.var(axis=0)  # (n_resp,)

    # Center counts and response
    counts_centered = counts - avg_n[None, :]  # (N, n_km)
    resp_centered = response - resp_mean[None, :]  # (N, n_resp)

    # Covariance matrix: (n_km, n_resp) via matrix multiply
    cov_matrix = (counts_centered.T @ resp_centered) / n_seqs  # (n_km, n_resp)

    # Standard deviations
    counts_std = np.sqrt(avg_var)  # (n_km,)
    resp_std = np.sqrt(resp_var)  # (n_resp,)

    # Pearson r = cov / (std_x * std_y), handle zero-variance
    denom = np.outer(counts_std, resp_std)  # (n_km, n_resp)
    safe_denom = np.where(denom > 1e-15, denom, 1.0)
    r_matrix = cov_matrix / safe_denom  # (n_km, n_resp)
    r_matrix[denom < 1e-15] = 0.0

    # R-squared and max across response columns
    r2_matrix = r_matrix**2  # (n_km, n_resp)
    max_r2 = r2_matrix.max(axis=1)  # (n_km,)

    # Filter by min_cor and zero variance
    valid = avg_var >= 1e-15
    if min_cor > 0:
        valid &= max_r2 >= min_cor * min_cor

    valid_idx = np.where(valid)[0]

    if len(valid_idx) == 0:
        cols = ["kmer", "max_r2", "avg_n", "avg_var"] + resp_names
        return pd.DataFrame(columns=cols)

    # Build result DataFrame
    result_data = {
        "kmer": [kmer_list[i] for i in valid_idx],
        "max_r2": max_r2[valid_idx],
        "avg_n": avg_n[valid_idx],
        "avg_var": avg_var[valid_idx],
    }
    for ri in range(n_resp):
        result_data[resp_names[ri]] = r_matrix[valid_idx, ri]

    df = pd.DataFrame(result_data)
    return df.sort_values("max_r2", ascending=False).reset_index(drop=True)


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
    kmer_list = [kmer] if isinstance(kmer, str) else list(kmer)

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

    if kmer_length > L:
        raise ValueError(f"PSSM has {L} rows but kmer_length is {kmer_length}")

    # Compute bits per position (with prior)
    mat_with_prior = mat + prior
    mat_norm = mat_with_prior / mat_with_prior.sum(axis=1, keepdims=True)
    mat_norm = np.clip(mat_norm, 1e-10, None)
    entropy = -np.sum(mat_norm * np.log2(mat_norm), axis=1)
    bits = 2.0 - entropy
    bits = np.nan_to_num(bits, nan=0.0)

    # Rolling sum to find best window
    if kmer_length == L:
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

        kmer_chars = [c if window_bits[i] > pos_bits_thresh else "N" for i, c in enumerate(kmer_chars)]

    return "".join(kmer_chars)
