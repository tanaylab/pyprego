"""Core type definitions for pyprego.

This module defines the data structures used throughout the package.
PSSM matrices and spatial models are represented as pandas DataFrames
to keep things simple, inspectable, and consistent with the R prego package.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PSSM Matrix
# ---------------------------------------------------------------------------
# A PSSM (Position-Specific Scoring Matrix) is stored as a pandas DataFrame
# with columns: pos (int), A (float), C (float), G (float), T (float).
# Each row corresponds to one position in the motif.
#
# Helper constructors are provided below; users may also build DataFrames
# directly.

NUCLEOTIDES = ("A", "C", "G", "T")


def pssm_dataframe(matrix: np.ndarray) -> pd.DataFrame:
    """Create a PSSM DataFrame from a (L, 4) NumPy array.

    Parameters
    ----------
    matrix : np.ndarray
        Array of shape (L, 4) with columns ordered A, C, G, T.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``pos``, ``A``, ``C``, ``G``, ``T``.

    Raises
    ------
    ValueError
        If *matrix* does not have exactly 4 columns.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 4:
        raise ValueError(f"Expected (L, 4) array, got shape {matrix.shape}")
    df = pd.DataFrame(matrix, columns=list(NUCLEOTIDES))
    df.insert(0, "pos", np.arange(len(df)))
    return df


def pssm_to_array(pssm: pd.DataFrame) -> np.ndarray:
    """Extract the (L, 4) NumPy array from a PSSM DataFrame.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame with at least columns ``A``, ``C``, ``G``, ``T``.

    Returns
    -------
    np.ndarray
        Array of shape (L, 4).
    """
    return pssm[list(NUCLEOTIDES)].to_numpy(dtype=np.float64)


# ---------------------------------------------------------------------------
# Spatial Model
# ---------------------------------------------------------------------------
# A spatial model is a pandas DataFrame with columns: bin (int), spat_factor (float).
# Each row gives the multiplicative spatial weight for one positional bin.


def spatial_dataframe(bins: np.ndarray, factors: np.ndarray) -> pd.DataFrame:
    """Create a spatial model DataFrame.

    Parameters
    ----------
    bins : np.ndarray
        1-D array of bin start positions.
    factors : np.ndarray
        1-D array of spatial factors (same length as *bins*).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``bin`` and ``spat_factor``.
    """
    bins = np.asarray(bins)
    factors = np.asarray(factors, dtype=np.float64)
    if bins.shape != factors.shape or bins.ndim != 1:
        raise ValueError("bins and factors must be 1-D arrays of the same length")
    return pd.DataFrame({"bin": bins, "spat_factor": factors})


# ---------------------------------------------------------------------------
# Regression Result
# ---------------------------------------------------------------------------


@dataclass
class RegressionResult:
    """Container for the output of :func:`pyprego.regression.regress_pwm`.

    Attributes
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame (pos, A, C, G, T) for the inferred motif.
    spat : pd.DataFrame
        Spatial model DataFrame (bin, spat_factor).
    pred : np.ndarray
        Predicted PWM score for each input sequence.
    consensus : str
        Consensus sequence derived from the PSSM.
    r2 : float | None
        R-squared of prediction vs response (continuous response).
    ks : float | None
        KS statistic (binary response).
    seed_motif : str | None
        The seed motif / kmer that initialised the regression.
    bidirect : bool
        Whether the model is bidirectional (uses reverse complement).
    spat_min : int
        Minimum spatial position used.
    spat_max : int | None
        Maximum spatial position used.
    seq_length : int | None
        Length of input sequences.
    _predict_fn : Callable | None
        Internal prediction function (set after fitting).
    """

    pssm: pd.DataFrame
    spat: pd.DataFrame
    pred: np.ndarray
    consensus: str
    r2: float | None = None
    ks: float | None = None
    seed_motif: str | None = None
    bidirect: bool = True
    spat_min: int = 1
    spat_max: int | None = None
    seq_length: int | None = None
    _predict_fn: Callable[..., np.ndarray] | None = field(default=None, repr=False)

    # -- optional fields populated when match_with_db=True --
    db_match_motif: str | None = None
    db_match_cor: float | None = None
    db_match_pssm: pd.DataFrame | None = None

    def predict(self, sequences: list[str] | np.ndarray) -> np.ndarray:
        """Predict PWM scores for new sequences.

        Parameters
        ----------
        sequences : list[str] | np.ndarray
            DNA sequences to score.

        Returns
        -------
        np.ndarray
            Predicted scores, one per sequence.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._predict_fn is None:
            raise RuntimeError("Model has not been fitted; no predict function available.")
        return self._predict_fn(sequences)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a plain dictionary (for YAML/JSON export)."""
        return {
            "pssm": self.pssm.to_dict(orient="list"),
            "spat": self.spat.to_dict(orient="list"),
            "consensus": self.consensus,
            "r2": self.r2,
            "ks": self.ks,
            "seed_motif": self.seed_motif,
            "bidirect": self.bidirect,
            "spat_min": self.spat_min,
            "spat_max": self.spat_max,
            "seq_length": self.seq_length,
        }
