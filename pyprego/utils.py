"""General-purpose utility functions for pyprego."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

# Complement mapping for DNA bases (handles upper and lower case)
_COMPLEMENT: dict[str, str] = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "a": "t",
    "t": "a",
    "c": "g",
    "g": "c",
    "N": "N",
    "n": "n",
}

_COMP_TABLE = str.maketrans(_COMPLEMENT)


def rc(dna: str) -> str:
    """Return the reverse complement of a DNA sequence.

    Parameters
    ----------
    dna : str
        A DNA string consisting of characters in ``{A, C, G, T, N}``
        (case-insensitive; case is preserved in the output).

    Returns
    -------
    str
        Reverse complement of *dna*.

    Raises
    ------
    TypeError
        If *dna* is not a string.
    ValueError
        If *dna* contains characters outside the recognised alphabet.

    Examples
    --------
    >>> rc("ACGT")
    'ACGT'
    >>> rc("AAACCC")
    'GGGTTT'
    >>> rc("acgt")
    'acgt'
    """
    if not isinstance(dna, str):
        raise TypeError(f"Expected str, got {type(dna).__name__}")
    result = dna.translate(_COMP_TABLE)
    # Check for untranslated characters (would remain unchanged but are invalid)
    bad = set(result) - set(_COMPLEMENT.values())
    if bad:
        raise ValueError(f"Invalid characters in DNA sequence: {bad!r}")
    return result[::-1]


def rc_array(sequences: list[str] | np.ndarray) -> list[str]:
    """Reverse complement every sequence in an array.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        Collection of DNA strings.

    Returns
    -------
    list[str]
        Reverse complements in the same order.
    """
    return [rc(s) for s in sequences]


def validate_sequences(sequences: list[str] | np.ndarray) -> np.ndarray:
    """Validate and convert sequences to a NumPy object array.

    Checks that all sequences are non-empty strings of equal length
    containing only valid DNA characters.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        Input DNA sequences.

    Returns
    -------
    np.ndarray
        1-D object array of uppercase DNA strings.

    Raises
    ------
    ValueError
        If sequences are empty, have different lengths, or contain
        invalid characters.
    """
    seqs = np.asarray(sequences, dtype=object)
    if seqs.ndim != 1 or len(seqs) == 0:
        raise ValueError("sequences must be a non-empty 1-D collection of strings")

    # Upper-case
    seqs = np.array([s.upper() for s in seqs], dtype=object)

    lengths = np.array([len(s) for s in seqs])
    if not np.all(lengths == lengths[0]):
        raise ValueError("All sequences must have the same length")

    valid_chars = set("ACGTN")
    for i, s in enumerate(seqs):
        if set(s) - valid_chars:
            bad = set(s) - valid_chars
            raise ValueError(f"Sequence {i} contains invalid characters: {bad!r}")

    return seqs


# ---------------------------------------------------------------------------
# Nucleotide helpers
# ---------------------------------------------------------------------------

_BASES = ("A", "C", "G", "T")
_NUC_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def _generate_n_mers(n: int) -> list[str]:
    """Generate all possible n-mers of length *n* in lexicographic order (A < C < G < T)."""
    return ["".join(combo) for combo in product(_BASES, repeat=n)]


def calc_sequences_dinucs(sequences: list[str] | np.ndarray) -> np.ndarray:
    """Calculate dinucleotide counts for each sequence.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        Collection of DNA sequences (case-insensitive).

    Returns
    -------
    np.ndarray
        Integer matrix of shape ``(len(sequences), 16)`` where each column
        corresponds to a dinucleotide in the order
        ``AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT``.
        The returned array has a ``dtype`` of ``int64`` and an attribute
        ``columns`` holding the dinucleotide names (for convenience; use
        :func:`dinuc_names` to obtain the list).
    """
    dinucs = _generate_n_mers(2)
    dinuc_to_idx = {d: i for i, d in enumerate(dinucs)}

    seqs = [s.upper() for s in sequences]
    n_seqs = len(seqs)
    result = np.zeros((n_seqs, 16), dtype=np.int64)

    for i, seq in enumerate(seqs):
        if len(seq) < 2:
            raise ValueError(f"Sequence {i} is too short for dinucleotide counting (length {len(seq)})")
        for j in range(len(seq) - 1):
            dn = seq[j : j + 2]
            idx = dinuc_to_idx.get(dn)
            if idx is not None:
                result[i, idx] += 1

    return result


def dinuc_names() -> list[str]:
    """Return the list of 16 dinucleotide names in canonical order."""
    return _generate_n_mers(2)


def _calc_sequences_n_nuc_dist(
    sequences: list[str] | np.ndarray,
    n: int,
    size: int | None = None,
) -> pd.DataFrame:
    """Calculate the positional n-nucleotide frequency distribution.

    Internal implementation shared by :func:`calc_sequences_dinuc_dist` and
    :func:`calc_sequences_trinuc_dist`.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        Collection of DNA sequences.
    n : int
        Length of the n-mer (2 for dinucleotides, 3 for trinucleotides).
    size : int | None
        Number of positions to consider. If ``None``, uses the maximum
        sequence length.

    Returns
    -------
    pd.DataFrame
        DataFrame with column ``pos`` (1-indexed) and one column per
        n-nucleotide. The last ``n - 1`` positions are ``NaN`` because a
        complete n-mer cannot start there.
    """
    if not isinstance(sequences, (list, np.ndarray)):
        raise TypeError("sequences must be a list or numpy array of strings")
    if not all(isinstance(s, str) for s in sequences):
        raise TypeError("All elements of sequences must be strings")

    seqs = [s.upper() for s in sequences]

    if size is None:
        size = max(len(s) for s in seqs)

    # Validate that all sequences are at least 'size' long
    for s in seqs:
        if len(s) < size:
            raise ValueError("Some sequences are shorter than the size")

    n_mers = _generate_n_mers(n)
    n_mer_to_idx = {nm: i for i, nm in enumerate(n_mers)}
    num_nmers = len(n_mers)
    n_seqs = len(seqs)

    # Count n-mer occurrences at each position
    counts = np.zeros((size, num_nmers), dtype=np.float64)
    for seq in seqs:
        for pos in range(size - n + 1):
            nmer = seq[pos : pos + n]
            idx = n_mer_to_idx.get(nmer)
            if idx is not None:
                counts[pos, idx] += 1

    # Convert to fractions
    freq = counts / n_seqs

    # Build DataFrame
    df = pd.DataFrame(freq, columns=n_mers)
    df.insert(0, "pos", np.arange(1, size + 1))

    # Set last n-1 positions to NaN
    if n > 1:
        df.iloc[(size - n + 1) : size, 1:] = np.nan

    return df


def calc_sequences_dinuc_dist(
    sequences: list[str] | np.ndarray,
    size: int | None = None,
) -> pd.DataFrame:
    """Calculate dinucleotide frequency distribution at each position.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        Collection of DNA sequences.
    size : int | None
        Number of positions to analyse. Defaults to the maximum sequence
        length.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``pos`` and 16 dinucleotide frequency columns.
        The last position is ``NaN`` (no dinucleotide can start there).
    """
    return _calc_sequences_n_nuc_dist(sequences, n=2, size=size)


def calc_sequences_trinuc_dist(
    sequences: list[str] | np.ndarray,
    size: int | None = None,
) -> pd.DataFrame:
    """Calculate trinucleotide frequency distribution at each position.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        Collection of DNA sequences.
    size : int | None
        Number of positions to analyse. Defaults to the maximum sequence
        length.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``pos`` and 64 trinucleotide frequency columns.
        The last two positions are ``NaN``.
    """
    return _calc_sequences_n_nuc_dist(sequences, n=3, size=size)


def sample_quantile_matched_rows(
    data_frame: pd.DataFrame,
    reference: np.ndarray | pd.Series,
    sample_fraction: float,
    num_quantiles: int = 10,
    seed: int | None = 60427,
) -> pd.DataFrame:
    """Stratified sampling that preserves the quantile distribution of a reference variable.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Data frame from which to sample rows.
    reference : array-like
        Numeric vector of the same length as the number of rows in
        *data_frame*.
    sample_fraction : float
        Fraction of rows to sample (0 < sample_fraction <= 1).
    num_quantiles : int
        Number of quantile bins to use for stratification (default 10).
    seed : int | None
        Random seed for reproducibility. ``None`` for no seeding.

    Returns
    -------
    pd.DataFrame
        Sampled subset of *data_frame* with approximately
        ``sample_fraction * len(data_frame)`` rows, preserving the
        quantile structure of *reference*.
    """
    rng = np.random.default_rng(seed)
    reference = np.asarray(reference, dtype=np.float64)
    n = len(data_frame)
    sample_size = int(np.floor(n * sample_fraction))

    if sample_size < 1:
        raise ValueError(f"Sample fraction {sample_fraction} is too small for the given data frame size {n}.")

    # Adjust num_quantiles if it's larger than sample_size
    num_quantiles = min(sample_size, num_quantiles)

    # Calculate quantile breaks
    probs = np.linspace(0, 1, num_quantiles + 1)
    breaks = np.quantile(reference, probs)
    breaks = np.unique(breaks)

    # Assign each row to a quantile bin
    bin_labels = np.digitize(reference, breaks[1:-1])  # 0-indexed bins

    # Group indices by bin
    unique_bins = np.unique(bin_labels)
    groups: dict[int, np.ndarray] = {}
    for b in unique_bins:
        groups[b] = np.where(bin_labels == b)[0]

    # Find the minimum group size and target per-bin sample count
    min_count = min(len(idx) for idx in groups.values())
    n_per_bin = min(min_count, int(np.floor(sample_size / len(unique_bins))))

    if n_per_bin < 1:
        n_per_bin = 1

    # Sample from each bin
    sampled_indices = []
    for b in unique_bins:
        idx = groups[b]
        chosen = rng.choice(idx, size=min(n_per_bin, len(idx)), replace=False)
        sampled_indices.append(chosen)

    sampled_indices = np.concatenate(sampled_indices)
    return data_frame.iloc[sampled_indices].reset_index(drop=True)
