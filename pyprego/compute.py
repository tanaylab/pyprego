"""PWM scoring / energy computation.

Mirrors the core ``compute_pwm`` / ``compute_local_pwm`` functions from the
R prego package. Given a PSSM and optional spatial model, compute the predicted
PWM energy for each input sequence.

All computation uses NumPy arrays; the interfaces accept and return arrays
so that a torch backend could be swapped in later with minimal changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .types import NUCLEOTIDES, pssm_to_array

if TYPE_CHECKING:
    pass

# Mapping nucleotide -> column index in the (L,4) PSSM array
_NUC_IDX = {n: i for i, n in enumerate(NUCLEOTIDES)}
_NUC_IDX.update({"a": 0, "c": 1, "g": 2, "t": 3})

# Complement mapping: A<->T (0<->3), C<->G (1<->2)
_COMPLEMENT = np.array([3, 2, 1, 0], dtype=np.intp)


def _encode_sequences(sequences: list[str] | np.ndarray) -> np.ndarray:
    """Encode DNA sequences as an integer matrix.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences (equal length).

    Returns
    -------
    np.ndarray
        Integer matrix of shape ``(n_sequences, seq_length)`` where values
        are 0=A, 1=C, 2=G, 3=T, -1=N/unknown.
    """
    n = len(sequences)
    L = len(sequences[0])
    encoded = np.full((n, L), -1, dtype=np.int8)
    for i, seq in enumerate(sequences):
        for j, ch in enumerate(seq):
            idx = _NUC_IDX.get(ch, -1)
            encoded[i, j] = idx
    return encoded


def _prepare_pssm(pssm: pd.DataFrame, prior: float) -> np.ndarray:
    """Normalise PSSM with prior, returning (K, 4) probability array.

    Matches the R preprocessing:
    - ``pssm_mat / rowSums(pssm_mat)`` (initial normalise)
    - ``(pssm_mat + prior) / rowSums(pssm_mat + prior)`` (add prior + re-normalise)

    Then the C++ ``DnaProbVec::normalize()`` is applied internally.
    In ``compute_pwm`` R wrapper, the sequence is:
    ``pssm_mat <- pssm_mat / rowSums(pssm_mat)``
    then ``pssm_mat <- pssm_mat + prior; pssm_mat <- pssm_mat / rowSums(pssm_mat + prior)``

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM with columns A, C, G, T.
    prior : float
        Prior probability added uniformly.

    Returns
    -------
    np.ndarray
        Normalised (K, 4) array.
    """
    mat = pssm_to_array(pssm).copy()  # (K, 4)
    # First normalize
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums
    # Then add prior and re-normalize
    if prior > 0:
        mat = mat + prior
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = mat / row_sums
    return mat


def _prepare_pssm_local(pssm: pd.DataFrame, prior: float) -> np.ndarray:
    """Normalise PSSM for compute_local_pwm (R adds prior but does NOT
    pre-normalize before, then C++ normalizes).

    In ``compute_local_pwm`` R wrapper:
    ``pssm_mat <- pssm_mat + prior`` (no division by rowSums first)
    Then C++ ``pssm.normalize()`` divides by row sums.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM with columns A, C, G, T.
    prior : float
        Prior probability added uniformly.

    Returns
    -------
    np.ndarray
        Normalised (K, 4) array.
    """
    mat = pssm_to_array(pssm).copy()  # (K, 4)
    # Add prior
    if prior > 0:
        mat = mat + prior
    # Normalize (as C++ does)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat = mat / row_sums
    return mat


def _compute_log_pssm(prob: np.ndarray) -> np.ndarray:
    """Convert probability array to log-probability, handling zeros like C++.

    In C++, zero probabilities get ``-REAL_MAX / 100`` (a very large negative number).
    We use ``-1e30`` as a practical equivalent.

    Parameters
    ----------
    prob : np.ndarray
        (K, 4) probability array.

    Returns
    -------
    np.ndarray
        (K, 4) log-probability array.
    """
    log_pssm = np.full_like(prob, -1e30)
    mask = prob > 0
    log_pssm[mask] = np.log(prob[mask])
    return log_pssm


def _score_windows(
    encoded: np.ndarray,
    log_pssm: np.ndarray,
    avg_log_prob: np.ndarray,
    *,
    reverse_complement: bool = False,
) -> np.ndarray:
    """Score all sliding windows of PSSM length across encoded sequences.

    Parameters
    ----------
    encoded : np.ndarray
        (N, L) integer-encoded sequences (0-3 valid, -1 for N/*).
    log_pssm : np.ndarray
        (K, 4) log-probabilities.
    avg_log_prob : np.ndarray
        (K,) average log-probability per position (for N-handling).
    reverse_complement : bool
        If True, score the reverse complement orientation.

    Returns
    -------
    np.ndarray
        (N, n_windows) array of log-scores where n_windows = L - K + 1.
        Windows containing N bases get the average log-prob at those positions
        (matching C++ behavior).
    """
    N, L = encoded.shape
    K = log_pssm.shape[0]
    n_windows = L - K + 1

    if n_windows <= 0:
        return np.full((N, 0), -np.inf)

    if reverse_complement:
        # RC: reverse PSSM positions, complement the sequence nucleotides
        # In C++: iterate PSSM in reverse, swap A<->T C<->G on seq char
        # Equivalent: use reversed log_pssm with complemented column indices
        rc_log_pssm = log_pssm[::-1, _COMPLEMENT]  # (K, 4) - reversed positions, swapped columns
        rc_avg = avg_log_prob[::-1]
        return _score_windows(encoded, rc_log_pssm, rc_avg, reverse_complement=False)

    # Build score array using vectorised window extraction
    # For each window start position w, gather (N, K) encoded values
    scores = np.zeros((N, n_windows), dtype=np.float64)

    for d in range(K):
        # encoded[:, w:w+K] for position d within the PSSM
        enc_slice = encoded[:, d : d + n_windows]  # (N, n_windows)

        # Valid bases: use log_pssm[d, base]
        valid_mask = enc_slice >= 0  # (N, n_windows)

        # For valid bases, look up log_pssm[d, base]
        # Clamp to 0 for indexing, then mask
        safe_enc = np.where(valid_mask, enc_slice, 0)
        pos_scores = log_pssm[d, safe_enc]  # (N, n_windows)

        # For invalid bases (N/*), use avg_log_prob[d]
        pos_scores = np.where(valid_mask, pos_scores, avg_log_prob[d])

        scores += pos_scores

    return scores


def _log_sum_exp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-sum-exp along an axis.

    Matches the C++ ``log_sum_exp`` and iterative ``log_sum_log``.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int
        Axis to reduce.

    Returns
    -------
    np.ndarray
        Reduced array.
    """
    max_val = np.max(x, axis=axis, keepdims=True)
    # Handle all-inf case
    all_inf = np.all(np.isneginf(x), axis=axis, keepdims=True)
    max_val_safe = np.where(all_inf, 0.0, max_val)
    result = max_val_safe + np.log(np.sum(np.exp(x - max_val_safe), axis=axis, keepdims=True))
    result = np.where(all_inf, -np.inf, result)
    return result.squeeze(axis=axis)


def compute_pwm(
    sequences: list[str] | np.ndarray,
    pssm: pd.DataFrame,
    spat: pd.DataFrame | None = None,
    *,
    spat_min: int = 1,
    spat_max: int | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
    func: str = "logSumExp",
) -> np.ndarray:
    """Compute PWM energy scores for sequences given a PSSM and spatial model.

    Mirrors the R ``compute_pwm()`` function. For each sequence, slides the
    PSSM across all valid positions, computes the log-likelihood at each
    window, applies spatial weighting, and aggregates via logSumExp or max.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences.
    pssm : pd.DataFrame
        PSSM DataFrame (pos, A, C, G, T).
    spat : pd.DataFrame | None
        Spatial model DataFrame (bin, spat_factor). If ``None``, uniform
        spatial weighting is used.
    spat_min : int
        Minimum position in the sequence to consider (1-based, as in R).
    spat_max : int | None
        Maximum position. ``None`` means use full sequence length.
    bidirect : bool
        Score both orientations and combine.
    prior : float
        Uniform prior added to PSSM probabilities.
    func : str
        Combination function: ``"logSumExp"`` or ``"max"``.

    Returns
    -------
    np.ndarray
        1-D array of scores, one per sequence.
    """
    if func not in ("logSumExp", "max"):
        raise ValueError(f"func must be 'logSumExp' or 'max', got {func!r}")

    if not all(c in pssm.columns for c in ("A", "C", "G", "T")):
        raise ValueError("PSSM must have columns A, C, G, T")

    sequences = [s.upper() for s in sequences]
    seq_len = len(sequences[0])

    # Handle spat_min/spat_max by trimming sequences (matching R behavior)
    if spat_max is None:
        spat_max = seq_len
    if spat_min is None:
        spat_min = 1

    if not (spat_min == 1 and spat_max == seq_len):
        # Python 0-based slicing: R's str_sub(s, start, end) is 1-based inclusive
        sequences = [s[spat_min - 1 : spat_max] for s in sequences]
        seq_len = len(sequences[0])

    # Parse spatial model
    if spat is None:
        spat_factors = np.array([1.0])
        bin_size = seq_len
    else:
        spat_factors = spat["spat_factor"].to_numpy(dtype=np.float64)
        bins = spat["bin"].to_numpy()
        bin_diffs = np.diff(bins)
        if len(bin_diffs) == 0:
            bin_size = seq_len
        else:
            bin_size = int(bin_diffs[0])

    # Prepare PSSM
    prob = _prepare_pssm(pssm, prior)
    K = prob.shape[0]
    log_pssm = _compute_log_pssm(prob)

    # Average log-prob per position (for N-handling): mean of all 4 log probs
    avg_log_prob = log_pssm.mean(axis=1)  # (K,)

    # Encode sequences
    encoded = _encode_sequences(sequences)  # (N, L)
    N, L = encoded.shape

    if L < K:
        return np.full(N, -np.inf)

    n_windows = L - K + 1

    # Score forward windows
    fwd_scores = _score_windows(encoded, log_pssm, avg_log_prob, reverse_complement=False)

    # Apply spatial weighting: each window position w maps to bin w // bin_size
    spat_log = np.log(spat_factors)
    window_bins = np.arange(n_windows) // bin_size
    # Clamp to valid range
    window_bins = np.clip(window_bins, 0, len(spat_factors) - 1)
    spat_weights = spat_log[window_bins]  # (n_windows,)

    fwd_scores = fwd_scores + spat_weights[np.newaxis, :]  # (N, n_windows)

    if bidirect:
        rc_scores = _score_windows(encoded, log_pssm, avg_log_prob, reverse_complement=True)
        rc_scores = rc_scores + spat_weights[np.newaxis, :]

    # Aggregate
    if func == "logSumExp":
        if bidirect:
            # logSumExp of all forward and RC scores together
            all_scores = np.concatenate([fwd_scores, rc_scores], axis=1)  # (N, 2*n_windows)
            result = _log_sum_exp(all_scores, axis=1)
        else:
            result = _log_sum_exp(fwd_scores, axis=1)
    else:  # max
        if bidirect:
            # For max: at each position, logSumExp(fwd, rc), then take max across positions
            # Matching C++ integrate_energy_max: at each pos, logSumLog(fwd, rc), then max over pos
            combined = np.logaddexp(fwd_scores, rc_scores)  # (N, n_windows)
            result = np.max(combined, axis=1)
        else:
            result = np.max(fwd_scores, axis=1)

    return result


def compute_local_pwm(
    sequences: list[str] | np.ndarray,
    pssm: pd.DataFrame,
    *,
    spat: pd.DataFrame | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
) -> np.ndarray:
    """Compute per-position PWM scores across each sequence.

    Mirrors the R ``compute_local_pwm()`` function. At each valid position,
    computes the log-likelihood of the PSSM alignment. Positions where the
    PSSM does not fit are set to NaN.

    In the R implementation, ``compute_local_pwm_cpp`` extracts a substring of
    motif_len at each position and calls ``integrate_energy`` on it. With a
    single-bin uniform spatial factor, this is equivalent to computing
    logSumExp(forward_score, rc_score) at each position when bidirect=True,
    or just the forward score when bidirect=False.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences.
    pssm : pd.DataFrame
        PSSM DataFrame.
    spat : pd.DataFrame | None
        Spatial model DataFrame. If provided, spatial weighting is applied.
        If ``None``, uniform weighting (factor=1) is used.
    bidirect : bool
        Score both orientations.
    prior : float
        Uniform prior.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(n_sequences, seq_length)`` with per-position
        scores. Positions where the PSSM window does not fit contain NaN.
    """
    if not all(c in pssm.columns for c in ("A", "C", "G", "T")):
        raise ValueError("PSSM must have columns A, C, G, T")

    sequences = [s.upper() for s in sequences]

    # For compute_local_pwm, use the "local" normalization (add prior, then normalize)
    prob = _prepare_pssm_local(pssm, prior)
    K = prob.shape[0]
    log_pssm = _compute_log_pssm(prob)
    avg_log_prob = log_pssm.mean(axis=1)

    # Parse spatial model
    if spat is None:
        spat_factors = np.array([1.0])
        bin_size = len(sequences[0])  # single bin covering entire sequence
    else:
        spat_factors = spat["spat_factor"].to_numpy(dtype=np.float64)
        bins = spat["bin"].to_numpy()
        bin_diffs = np.diff(bins)
        if len(bin_diffs) == 0:
            bin_size = len(sequences[0])
        else:
            bin_size = int(bin_diffs[0])

    encoded = _encode_sequences(sequences)
    N, L = encoded.shape

    # Initialize output with NaN
    result = np.full((N, L), np.nan, dtype=np.float64)

    if L < K:
        return result

    n_windows = L - K + 1

    # Score all windows
    fwd_scores = _score_windows(encoded, log_pssm, avg_log_prob, reverse_complement=False)

    # Apply spatial weighting
    # In compute_local_pwm_cpp, each position j extracts substr(j, motif_len)
    # and calls integrate_energy. For that substring, the spatial bin is
    # position 0 within the substring / bin_size = 0, so spat_factors[0].
    # But this is ONLY because the substring is exactly motif-length.
    #
    # Actually, looking more carefully at the C++ code: compute_local_pwm_cpp
    # creates DnaPWML with the full spat_fac and bin_size, then calls
    # integrate_energy on a substring. The substring starts at position 0,
    # and the loop iterates with pos starting at m_min_range=0 (since spat_min=0).
    # Since the substring is motif_len long, only position 0 is valid.
    # spat_bin = 0 / bin_size = 0, so it uses spat_factors[0].
    # This means spatial weighting in compute_local_pwm always uses spat_factors[0].
    spat_log_0 = np.log(spat_factors[0])

    fwd_scores = fwd_scores + spat_log_0

    if bidirect:
        rc_scores = _score_windows(encoded, log_pssm, avg_log_prob, reverse_complement=True)
        rc_scores = rc_scores + spat_log_0
        # logSumExp of forward and RC at each position
        # C++ integrate_energy uses log_sum_log to combine fwd and rc
        combined = np.logaddexp(fwd_scores, rc_scores)
        result[:, :n_windows] = combined
    else:
        result[:, :n_windows] = fwd_scores

    return result
