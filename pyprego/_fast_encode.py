"""Fast sequence encoding using NumPy vectorized byte operations.

Shared by compute.py, regression.py, and kmers.py.
"""

from __future__ import annotations

import numpy as np

# Lookup table: ASCII byte value -> nucleotide index (0=A, 1=C, 2=G, 3=T, -1=other)
_BYTE_TO_IDX = np.full(256, -1, dtype=np.int8)
_BYTE_TO_IDX[ord("A")] = 0
_BYTE_TO_IDX[ord("C")] = 1
_BYTE_TO_IDX[ord("G")] = 2
_BYTE_TO_IDX[ord("T")] = 3
_BYTE_TO_IDX[ord("a")] = 0
_BYTE_TO_IDX[ord("c")] = 1
_BYTE_TO_IDX[ord("g")] = 2
_BYTE_TO_IDX[ord("t")] = 3


def encode_sequences_fast(sequences: list[str] | np.ndarray) -> np.ndarray:
    """Encode DNA sequences as int8 array using vectorized byte lookup.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences of equal length.

    Returns
    -------
    np.ndarray
        Shape ``(n_sequences, seq_length)``, dtype int8.
        Values: 0=A, 1=C, 2=G, 3=T, -1=N/unknown.
    """
    if isinstance(sequences, np.ndarray):
        sequences = sequences.tolist()
    n = len(sequences)
    if n == 0:
        return np.empty((0, 0), dtype=np.int8)
    L = len(sequences[0])
    # Join all sequences into one big string, convert to bytes, then reshape
    big_str = "".join(sequences)
    byte_arr = np.frombuffer(big_str.encode("ascii"), dtype=np.uint8)
    return _BYTE_TO_IDX[byte_arr].reshape(n, L)
