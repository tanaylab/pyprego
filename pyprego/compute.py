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

from ._fast_encode import encode_sequences_fast
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

    Uses fast vectorized byte lookup. See :func:`_fast_encode.encode_sequences_fast`.
    """
    return encode_sequences_fast(sequences)


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
    return mat / row_sums


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


def batch_extract_energies(
    encoded: np.ndarray,
    log_pssm_list: list[np.ndarray],
    spat_factors_list: list[np.ndarray],
    spat_bin_sizes: list[int] | np.ndarray,
    bidirect: bool = True,
) -> np.ndarray:
    """Batch compute PWM energies for multiple motifs via C++.

    Parameters
    ----------
    encoded : np.ndarray
        (N, L) int8 encoded sequences.
    log_pssm_list : list[np.ndarray]
        List of M arrays, each (K_m, 4) float64 log-probability PSSMs.
    spat_factors_list : list[np.ndarray]
        List of M arrays, each (B_m,) float64 raw spatial factors.
    spat_bin_sizes : list[int] | np.ndarray
        Bin size per motif.
    bidirect : bool
        Score both orientations.

    Returns
    -------
    np.ndarray
        (N, M) float64 energy scores.
    """
    N = encoded.shape[0]
    M = len(log_pssm_list)
    output = np.empty((N, M), dtype=np.float64)

    # Ensure arrays are contiguous and correct dtype
    c_log_pssm_list = [np.ascontiguousarray(p, dtype=np.float64) for p in log_pssm_list]
    c_spat_list = [np.ascontiguousarray(s, dtype=np.float64) for s in spat_factors_list]
    c_bin_sizes = np.asarray(spat_bin_sizes, dtype=np.int32)

    try:
        from pyprego._pyprego import batch_extract_energies as _batch_c

        _batch_c(
            encoded,
            c_log_pssm_list,
            c_spat_list,
            c_bin_sizes,
            int(bidirect),
            output,
        )
    except (ImportError, AttributeError):
        # Fallback: pure-Python loop using existing scoring functions
        for m_idx in range(M):
            log_pssm = c_log_pssm_list[m_idx]
            spat_facs = c_spat_list[m_idx]
            bin_size = int(c_bin_sizes[m_idx])
            K = log_pssm.shape[0]
            L = encoded.shape[1]
            n_windows = L - K + 1

            if n_windows <= 0:
                output[:, m_idx] = -np.inf
                continue

            avg_log_prob = log_pssm.mean(axis=1)

            fwd_scores = _score_windows(encoded, log_pssm, avg_log_prob, reverse_complement=False)
            spat_log = np.log(spat_facs)
            window_bins = np.arange(n_windows) // bin_size
            window_bins = np.clip(window_bins, 0, len(spat_facs) - 1)
            spat_weights = spat_log[window_bins]
            fwd_scores = fwd_scores + spat_weights[np.newaxis, :]

            if bidirect:
                rc_scores = _score_windows(encoded, log_pssm, avg_log_prob, reverse_complement=True)
                rc_scores = rc_scores + spat_weights[np.newaxis, :]
                all_scores = np.concatenate([fwd_scores, rc_scores], axis=1)
                output[:, m_idx] = _log_sum_exp(all_scores, axis=1)
            else:
                output[:, m_idx] = _log_sum_exp(fwd_scores, axis=1)

    return output


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
        bin_size = seq_len if len(bin_diffs) == 0 else int(bin_diffs[0])

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
        len(sequences[0])  # single bin covering entire sequence
    else:
        spat_factors = spat["spat_factor"].to_numpy(dtype=np.float64)
        bins = spat["bin"].to_numpy()
        bin_diffs = np.diff(bins)
        len(sequences[0]) if len(bin_diffs) == 0 else int(bin_diffs[0])

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


# ---------------------------------------------------------------------------
# Expected local PWM scoring over base frequency matrices
# ---------------------------------------------------------------------------

# Widest motif block. Wider blocks amortise the BLAS call and keep the fold's
# innermost axis long enough to vectorise; past ~64 the intermediate stops
# fitting in cache and nothing more is gained.
_FREQ_BLOCK_MAX = 64

# Narrowest motif block. Below this the (m, 4) x (4, D*B) product becomes too
# skinny for BLAS and the fold's innermost axis too short to vectorise; a
# 61-motif database measured 1.4x slower at 8 than at 16, in both the
# single-matrix and the batched regime.
_FREQ_BLOCK_MIN = 16


class _serial_blas:
    """Clamp nested BLAS thread pools to one thread.

    The parallelism layer here is a thread pool over (matrix, motif block)
    tasks; letting each per-block ``matmul`` spin up its own BLAS pool on top
    of that oversubscribes the machine and costs roughly 3x. Mirrors R prego's
    ``local_serial_blas()``. Uses threadpoolctl when importable and falls back
    to the environment variable, matching the pattern in ``regression.py``.
    """

    def __init__(self) -> None:
        self._ctx = None
        self._prev_omp: str | None = None

    def __enter__(self) -> _serial_blas:
        import os

        self._prev_omp = os.environ.get("OMP_NUM_THREADS")
        os.environ["OMP_NUM_THREADS"] = "1"
        try:
            from threadpoolctl import threadpool_limits  # type: ignore

            self._ctx = threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            self._ctx = None
        return self

    def __exit__(self, *exc: object) -> None:
        import os

        if self._ctx is not None:
            self._ctx.unregister()
            self._ctx = None
        if self._prev_omp is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = self._prev_omp


def freq_local_pwm_block_size(n_motifs: int, n_threads: int) -> int:
    """Number of motifs per block, given the database size and the thread count.

    Deliberately independent of how many frequency matrices are scored in one
    call: the block width is the inner dimension of the per-block ``matmul``,
    so letting the batch size feed into it would make the same matrix score
    one or two ULP differently depending on what was passed alongside it.

    The block is narrow enough that a *single* frequency matrix already splits
    into at least ``n_threads`` tasks, so a one-matrix call is not serial.

    Parameters
    ----------
    n_motifs : int
        Number of motifs in the database.
    n_threads : int
        Number of worker threads the tasks will be spread over.

    Returns
    -------
    int
        Motifs per block.
    """
    per_thread = n_motifs // max(1, n_threads)
    return int(np.clip(per_thread, _FREQ_BLOCK_MIN, _FREQ_BLOCK_MAX))


def freq_local_pwm_plan(
    mat: np.ndarray,
    rc_mat: np.ndarray,
    motif_lengths: np.ndarray,
    *,
    multiply: bool = True,
    bidirect: bool = True,
    n_threads: int = 1,
) -> list[dict]:
    """Precompute the per-block operands for :func:`batch_freq_local_pwm`.

    Motifs are sorted by length before blocking, so each block only pays for
    its own longest motif rather than for the longest motif in the database.
    On the 3867-motif database (D=35, mean length 12.7) this cuts about a
    fifth off the run time, and it is numerically free -- positions past a
    motif's length contribute exactly zero either way.

    The result depends only on the database, the two mode flags and
    *n_threads*, so it can be cached and reused across calls.

    Parameters
    ----------
    mat : np.ndarray
        Stacked log-scale PWM matrix, shape ``(D*4, n_motifs)``.
    rc_mat : np.ndarray
        Reverse-complement log-scale PWM matrix, same shape as *mat*.
    motif_lengths : np.ndarray
        Length of each motif, shape ``(n_motifs,)``.
    multiply : bool
        ``True`` for ``combine="multiply"``, ``False`` for ``combine="sum"``.
    bidirect : bool
        Whether the reverse-complement operand is needed.
    n_threads : int
        Thread count the tasks will be spread over.

    Returns
    -------
    list[dict]
        One entry per motif block.
    """
    n_motifs = mat.shape[1]
    block_size = freq_local_pwm_block_size(n_motifs, n_threads)
    order = np.argsort(motif_lengths, kind="stable")

    blocks: list[dict] = []
    for start in range(0, n_motifs, block_size):
        cols = order[start : start + block_size]
        n_block = len(cols)
        lens = motif_lengths[cols]
        width = int(lens.max())

        operands = []
        for src in (mat, rc_mat) if bidirect else (mat,):
            # (4*width, n_block) -> (4, width*n_block), so that the product
            # reshapes to (positions, width, n_block) with the block index
            # innermost and contiguous.
            sub = src[: 4 * width, cols]
            op = np.ascontiguousarray(sub.reshape(width, 4, n_block).transpose(1, 0, 2).reshape(4, width * n_block))
            operands.append(np.exp(op) if multiply else op)

        # Offsets past each motif's own length, as one contiguous run per
        # motif position (the block is length-sorted, so the motifs needing
        # padding at offset o are exactly the first searchsorted(lens, o) ones).
        pad = [(o, int(np.searchsorted(lens, o, side="right"))) for o in range(int(lens.min()), width)]
        blocks.append(
            {
                "cols": cols,
                "n_block": n_block,
                "width": width,
                "lens": lens,
                "operands": operands,
                "pad": [(o, k) for o, k in pad if k > 0],
            }
        )
    return blocks


def _freq_fold(
    freqs_padded: np.ndarray,
    block: dict,
    n_pos: int,
    multiply: bool,
    operand: np.ndarray,
    buf: np.ndarray,
) -> np.ndarray:
    """Score one strand of one motif block against one frequency matrix."""
    width, n_block = block["width"], block["n_block"]
    n_rows = n_pos + width - 1

    prod = np.matmul(
        freqs_padded[:n_rows],
        operand,
        out=buf[: n_rows * width * n_block].reshape(n_rows, width * n_block),
    ).reshape(n_rows, width, n_block)

    if multiply:
        np.log(prod, out=prod)
    # Offsets past a motif's own length must contribute *exactly* zero, so
    # that a motif's score never depends on how the block happened to be
    # formed. Under "sum" the padded log-probabilities are 0 and the product
    # already is; under "multiply" they are exp(0) = 1, whose product is the
    # row sum of the frequencies -- 1 only to within rounding. Zeroing both
    # keeps the guarantee from resting on how BLAS accumulates.
    for offset, k in block["pad"]:
        prod[:, offset, :k] = 0.0

    # out[j, b] = sum_l prod[j + l, l, b]: the diagonal band, as a stride trick
    # rather than a copy. The last element lands exactly on the last element of
    # `prod`, so the view never reads past the buffer.
    item = prod.itemsize
    band = np.lib.stride_tricks.as_strided(
        prod,
        shape=(n_pos, width, n_block),
        strides=(width * n_block * item, (width * n_block + n_block) * item, item),
    )
    return band.sum(axis=1)


def batch_freq_local_pwm(
    freq_list: list[np.ndarray],
    mat: np.ndarray,
    rc_mat: np.ndarray,
    motif_lengths: np.ndarray,
    *,
    multiply: bool = True,
    bidirect: bool = True,
    n_threads: int = 1,
    plan: list[dict] | None = None,
) -> list[np.ndarray]:
    """Expected local PWM scores for a list of base frequency matrices.

    The array-level kernel behind :func:`~pyprego.motif_db.calc_freq_local_pwm`;
    see that function for the definition of the two combine modes.

    Parameters
    ----------
    freq_list : list[np.ndarray]
        Base frequency matrices, each ``(m, 4)`` in A, C, G, T order with rows
        summing to 1. Lengths may differ.
    mat : np.ndarray
        Stacked log-scale PWM matrix, shape ``(D*4, n_motifs)``.
    rc_mat : np.ndarray
        Reverse-complement log-scale PWM matrix, same shape as *mat*.
    motif_lengths : np.ndarray
        Length of each motif, shape ``(n_motifs,)``.
    multiply : bool
        ``True`` for ``combine="multiply"``, ``False`` for ``combine="sum"``.
    bidirect : bool
        Score both strands, combining them per position as
        ``log(exp(fwd) + exp(rev))``.
    n_threads : int
        Worker threads to spread the (matrix, motif block) tasks over.
    plan : list[dict] | None
        Precomputed operands from :func:`freq_local_pwm_plan`. Built on the fly
        when ``None``.

    Returns
    -------
    list[np.ndarray]
        One ``(n_motifs, m)`` float64 array per input matrix, row *i* holding
        the scores of motif *i* and column *j* the score of a motif *starting*
        at *j*. The last ``motif_lengths[i] - 1`` entries of row *i* are NaN.
    """
    n_motifs = mat.shape[1]
    max_pos = mat.shape[0] // 4
    motif_lengths = np.asarray(motif_lengths, dtype=np.int64)

    if plan is None:
        plan = freq_local_pwm_plan(
            mat, rc_mat, motif_lengths, multiply=multiply, bidirect=bidirect, n_threads=n_threads
        )

    # Pad each matrix so that the band never runs off the end. Padded rows are
    # reached only at offsets past a motif's length (whose contribution is
    # forced to zero) or at start positions where the motif does not fit
    # (which are masked to NaN), so the value only has to stay finite -- a
    # uniform distribution keeps the log in "multiply" mode well defined.
    padded = [
        np.ascontiguousarray(np.vstack([q, np.full((max_pos - 1, 4), 0.25, dtype=np.float64)])) for q in freq_list
    ]
    outputs = [np.empty((n_motifs, q.shape[0]), dtype=np.float64) for q in freq_list]

    block_max = max(b["width"] * b["n_block"] for b in plan)
    buf_size = (max(q.shape[0] for q in freq_list) + max_pos - 1) * block_max

    def run(task: tuple[int, int], buf: np.ndarray) -> None:
        k, bi = task
        block = plan[bi]
        n_pos = freq_list[k].shape[0]
        acc = _freq_fold(padded[k], block, n_pos, multiply, block["operands"][0], buf)
        if bidirect:
            acc = np.logaddexp(acc, _freq_fold(padded[k], block, n_pos, multiply, block["operands"][1], buf))
        acc = acc.T.copy()
        # Start positions where a motif does not fit are NaN, per motif length.
        # Vectorised rather than a loop over the block: these tasks are short
        # enough that per-task Python time shows up in the thread scaling.
        acc[np.arange(n_pos) > (n_pos - block["lens"])[:, None]] = np.nan
        outputs[k][block["cols"], :] = acc

    tasks = [(k, bi) for k in range(len(freq_list)) for bi in range(len(plan))]
    # More workers than tasks only adds pool and threadpoolctl overhead. The
    # block layout was already fixed from the *requested* thread count, so
    # capping here cannot change the numbers.
    n_threads = min(n_threads, len(tasks))

    if n_threads <= 1:
        buf = np.empty(buf_size, dtype=np.float64)
        for task in tasks:
            run(task, buf)
    else:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        local = threading.local()

        def run_threaded(task: tuple[int, int]) -> None:
            buf = getattr(local, "buf", None)
            if buf is None:
                buf = local.buf = np.empty(buf_size, dtype=np.float64)
            run(task, buf)

        with _serial_blas(), ThreadPoolExecutor(max_workers=n_threads) as pool:
            list(pool.map(run_threaded, tasks))

    return outputs
