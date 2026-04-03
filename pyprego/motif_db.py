"""Motif database management.

Mirrors MotifDB.R and motif-dbs.R from the R prego package. Provides
functionality for loading, storing, and querying collections of motif PSSMs
(e.g. JASPAR, HOMER databases).

The :class:`MotifDB` class stores stacked log-scale PWM matrices for
efficient batch scoring across many motifs.
"""

from __future__ import annotations

import importlib.resources
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .compute import compute_pwm
from .types import NUCLEOTIDES, pssm_dataframe, pssm_to_array

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# MotifDB class
# ---------------------------------------------------------------------------


class MotifDB:
    """A collection of motif PSSMs stored as stacked log-scale matrices.

    This is the Python port of the R S4 ``MotifDB`` class.  All PWM data
    is stored in two dense matrices (forward and reverse-complement), each
    of shape ``(max_len * 4, n_motifs)`` where positions beyond a motif's
    actual length are zeroed out.

    Parameters
    ----------
    mat : np.ndarray
        Stacked log-scale PWM matrix, shape ``(D*4, n_motifs)``.
    rc_mat : np.ndarray
        Reverse-complement log-scale PWM matrix, same shape as *mat*.
    motif_lengths : dict[str, int]
        Mapping from motif name to its length in positions.
    prior : float
        The PSSM prior probability used when creating the matrices.
    spat_factors : np.ndarray
        Spatial factor matrix, shape ``(n_motifs, n_bins)``.
    spat_bin_size : float
        Size of spatial bins.
    spat_min : float | None
        Starting position of the sequence, or ``None``.
    spat_max : float | None
        Ending position of the sequence, or ``None``.
    """

    def __init__(
        self,
        mat: np.ndarray,
        rc_mat: np.ndarray,
        motif_lengths: dict[str, int],
        prior: float,
        spat_factors: np.ndarray,
        spat_bin_size: float = 1.0,
        spat_min: float | None = None,
        spat_max: float | None = None,
    ) -> None:
        self.mat = mat
        self.rc_mat = rc_mat
        self.motif_lengths = motif_lengths
        self.prior = prior
        self.spat_factors = spat_factors
        self.spat_bin_size = spat_bin_size
        self.spat_min = spat_min
        self.spat_max = spat_max
        self._validate()

    def _validate(self) -> None:
        """Validate internal consistency (mirrors R ``validObject``)."""
        errors: list[str] = []

        if self.mat.shape[0] % 4 != 0:
            errors.append("Matrix rows must be a multiple of 4 (A, C, G, T)")

        if self.mat.shape != self.rc_mat.shape:
            errors.append(
                "Reverse complement matrix must have the same dimensions as the main matrix"
            )

        n_motifs = self.mat.shape[1]

        if len(self.motif_lengths) != n_motifs:
            errors.append(
                f"Length of motif_lengths ({len(self.motif_lengths)}) must match "
                f"number of matrix columns ({n_motifs})"
            )

        if any(v <= 0 for v in self.motif_lengths.values()):
            errors.append("All motif lengths must be positive")

        if not (0 < self.prior < 1):
            errors.append("Prior must be between 0 and 1 (exclusive)")

        # Spatial factors
        if self.spat_factors.shape[0] > 0:
            if self.spat_factors.shape[0] != n_motifs:
                errors.append(
                    "Number of rows in spatial factors must match number of motifs"
                )
            if np.any(self.spat_factors < 0):
                errors.append("Spatial factors must be non-negative")

        if self.spat_bin_size <= 0:
            errors.append("Spatial bin size must be positive")

        if self.spat_min is not None and self.spat_max is not None:
            if self.spat_min > self.spat_max:
                errors.append("Spatial min must be less than or equal to spatial max")
            if self.spat_min < 0:
                errors.append("Spatial min must be non-negative")

        if errors:
            raise ValueError("MotifDB validation failed:\n  " + "\n  ".join(errors))

    # -- Container protocol --------------------------------------------------

    def __len__(self) -> int:
        return self.mat.shape[1]

    def names(self) -> list[str]:
        """Return the names of all motifs in the database."""
        return list(self.motif_lengths.keys())

    def __contains__(self, item: str) -> bool:
        return item in self.motif_lengths

    def __iter__(self) -> Iterator[str]:
        return iter(self.motif_lengths)

    def __repr__(self) -> str:
        n = len(self)
        parts = [f"MotifDB with {n} motif{'s' if n != 1 else ''}, prior={self.prior}"]
        if self.spat_factors.shape[1] > 1 or self.spat_bin_size > 1:
            parts.append(
                f"  Spatial: bin_size={self.spat_bin_size}, "
                f"bins_per_motif={self.spat_factors.shape[1]}"
            )
        if self.spat_min is not None and self.spat_max is not None:
            parts.append(f"  Spatial range: {self.spat_min} to {self.spat_max}")
        return "\n".join(parts)

    # -- Subscript -----------------------------------------------------------

    def __getitem__(self, key: str | list[str] | int | list[int]) -> MotifDB:
        """Subset the MotifDB by motif name(s) or integer index/indices.

        Supports exact matching for strings and integer indexing.

        Parameters
        ----------
        key : str | list[str] | int | list[int]
            Motif name(s) or integer index/indices.

        Returns
        -------
        MotifDB
            A new MotifDB containing only the selected motifs.
        """
        all_names = self.names()

        if isinstance(key, (int, np.integer)):
            key = [int(key)]
        elif isinstance(key, str):
            key = [key]

        if isinstance(key, (list, tuple, np.ndarray)):
            # Determine if integer or string keys
            if len(key) == 0:
                raise IndexError("Empty selection")

            if isinstance(key[0], (int, np.integer)):
                indices = [int(k) for k in key]
                for idx in indices:
                    if idx < 0 or idx >= len(all_names):
                        raise IndexError(
                            f"Index {idx} out of bounds for MotifDB with {len(all_names)} motifs"
                        )
            else:
                # String keys -- exact match
                missing = [k for k in key if k not in self.motif_lengths]
                if missing:
                    raise KeyError(f"Motifs not found in MotifDB: {missing}")
                indices = [all_names.index(k) for k in key]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        new_names = [all_names[i] for i in indices]
        new_motif_lengths = {n: self.motif_lengths[n] for n in new_names}

        return MotifDB(
            mat=self.mat[:, indices],
            rc_mat=self.rc_mat[:, indices],
            motif_lengths=new_motif_lengths,
            prior=self.prior,
            spat_factors=self.spat_factors[indices, :],
            spat_bin_size=self.spat_bin_size,
            spat_min=self.spat_min,
            spat_max=self.spat_max,
        )

    def grep(self, pattern: str | list[str]) -> MotifDB:
        """Subset the MotifDB by regex pattern matching on motif names.

        Parameters
        ----------
        pattern : str | list[str]
            Regex pattern(s) to match against motif names (case-insensitive).

        Returns
        -------
        MotifDB
            A new MotifDB containing matched motifs.
        """
        if isinstance(pattern, str):
            pattern = [pattern]

        all_names = self.names()
        matched: list[str] = []
        for pat in pattern:
            pat_matches = [n for n in all_names if re.search(pat, n, re.IGNORECASE)]
            if not pat_matches:
                warnings.warn(f"Pattern {pat!r} matched no motifs", stacklevel=2)
            matched.extend(pat_matches)

        # Remove duplicates while preserving order
        seen: set[str] = set()
        unique_matched: list[str] = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                unique_matched.append(m)

        if not unique_matched:
            raise KeyError("No motifs matched any of the provided patterns")

        return self[unique_matched]


# ---------------------------------------------------------------------------
# Internal helpers for matrix construction
# ---------------------------------------------------------------------------


def _motif_db_to_mat(
    motif_df: pd.DataFrame, prior: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a tidy motif DataFrame to stacked forward and RC matrices.

    Mirrors the R ``motif_db_to_mat`` function.

    Parameters
    ----------
    motif_df : pd.DataFrame
        DataFrame with columns ``motif``, ``A``, ``C``, ``G``, ``T``.
    prior : float
        Prior probability added before normalisation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(forward_mat, rc_mat)`` each of shape ``(D*4, n_motifs)`` where
        D is the maximum motif length.
    """
    nuc_cols = list(NUCLEOTIDES)  # A, C, G, T

    # Add position within each motif
    motif_df = motif_df.copy()
    motif_df["_pos"] = motif_df.groupby("motif", sort=False).cumcount() + 1

    D = int(motif_df["_pos"].max())
    motif_names = list(dict.fromkeys(motif_df["motif"]))  # preserve order
    n_motifs = len(motif_names)
    motif_idx = {name: i for i, name in enumerate(motif_names)}

    # Initialise matrices with zeros
    forward_mat = np.zeros((D * 4, n_motifs), dtype=np.float64)
    rc_mat = np.zeros((D * 4, n_motifs), dtype=np.float64)

    for motif_name, group in motif_df.groupby("motif", sort=False):
        col = motif_idx[motif_name]
        positions = group["_pos"].values
        values = group[nuc_cols].values  # (L, 4)

        # Normalise: first by row sum, then add prior, then renormalise
        row_sums = values.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        values = values / row_sums
        values = values + prior
        row_sums2 = values.sum(axis=1, keepdims=True)
        values = values / row_sums2

        motif_len = len(positions)

        # Fill forward matrix: rows are interleaved (A_1, C_1, G_1, T_1, A_2, ...)
        for i, pos in enumerate(positions):
            base_row = (int(pos) - 1) * 4
            for nuc_j in range(4):
                forward_mat[base_row + nuc_j, col] = values[i, nuc_j]

        # Fill reverse complement matrix
        # RC: reverse positions, complement nucleotides (A<->T, C<->G)
        complement_map = [3, 2, 1, 0]  # A->T, C->G, G->C, T->A
        for i, pos in enumerate(positions):
            rc_pos = motif_len - int(pos) + 1
            base_row_rc = (rc_pos - 1) * 4
            for nuc_j in range(4):
                rc_nuc = complement_map[nuc_j]
                rc_mat[base_row_rc + rc_nuc, col] = values[i, nuc_j]

    return forward_mat, rc_mat


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_motif_db(
    pssm_df: pd.DataFrame,
    prior: float = 0.01,
    spat_factors: np.ndarray | None = None,
    spat_bin_size: float = 1.0,
    spat_min: float | None = None,
    spat_max: float | None = None,
) -> MotifDB:
    """Create a :class:`MotifDB` from a tidy DataFrame of PSSMs.

    Mirrors the R ``create_motif_db()`` function.

    Parameters
    ----------
    pssm_df : pd.DataFrame
        DataFrame with columns ``motif``, ``A``, ``C``, ``G``, ``T``.
        Each group of rows with the same ``motif`` value defines one PSSM.
    prior : float
        Pseudocount prior to add to probabilities (must be in (0, 1)).
    spat_factors : np.ndarray | None
        Spatial factor matrix of shape ``(n_motifs, n_bins)``, with rows
        ordered the same as the unique motifs in *pssm_df*. If ``None``,
        a default all-ones vector is used.
    spat_bin_size : float
        Size of spatial bins.
    spat_min : float | None
        Starting position of the sequence, or ``None``.
    spat_max : float | None
        Ending position of the sequence, or ``None``.

    Returns
    -------
    MotifDB
        A validated MotifDB object.
    """
    required_cols = {"motif", "A", "C", "G", "T"}
    if not required_cols.issubset(pssm_df.columns):
        missing = required_cols - set(pssm_df.columns)
        raise ValueError(f"pssm_df is missing required columns: {missing}")

    # Calculate matrices
    forward_mat, rc_mat = _motif_db_to_mat(pssm_df, prior)

    # Calculate motif lengths
    motif_lengths_series = pssm_df.groupby("motif", sort=False).size()
    motif_names = list(dict.fromkeys(pssm_df["motif"]))  # preserve order
    motif_lengths = {name: int(motif_lengths_series[name]) for name in motif_names}

    # Convert to log scale
    with np.errstate(divide="ignore"):
        forward_mat = np.log(forward_mat)
    with np.errstate(divide="ignore"):
        rc_mat = np.log(rc_mat)

    # Replace -inf (from log(0)) with 0 -- matching R behaviour
    # where positions beyond motif length are zeroed out
    D = forward_mat.shape[0] // 4
    for i, name in enumerate(motif_names):
        ml = motif_lengths[name]
        if ml * 4 < forward_mat.shape[0]:
            forward_mat[ml * 4 :, i] = 0.0
            rc_mat[ml * 4 :, i] = 0.0

    # Handle -inf within valid positions (shouldn't happen with prior > 0,
    # but be safe)
    forward_mat = np.where(np.isneginf(forward_mat), 0.0, forward_mat)
    rc_mat = np.where(np.isneginf(rc_mat), 0.0, rc_mat)

    # Create default spatial factors if none provided
    n_motifs = len(motif_names)
    if spat_factors is None:
        spat_factors_arr = np.ones((n_motifs, 1), dtype=np.float64)
    else:
        spat_factors_arr = np.asarray(spat_factors, dtype=np.float64)
        if spat_factors_arr.ndim == 1:
            spat_factors_arr = spat_factors_arr.reshape(-1, 1)

    return MotifDB(
        mat=forward_mat,
        rc_mat=rc_mat,
        motif_lengths=motif_lengths,
        prior=prior,
        spat_factors=spat_factors_arr,
        spat_bin_size=spat_bin_size,
        spat_min=spat_min,
        spat_max=spat_max,
    )


# ---------------------------------------------------------------------------
# Conversion back to DataFrame
# ---------------------------------------------------------------------------


def motif_db_to_dataframe(db: MotifDB) -> pd.DataFrame:
    """Convert a :class:`MotifDB` back to a tidy DataFrame.

    Mirrors the R ``motif_db_to_dataframe()`` function.

    Parameters
    ----------
    db : MotifDB
        A MotifDB object.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``motif``, ``pos``, ``A``, ``C``, ``G``, ``T``.
    """
    # Convert from log to probability
    prob_mat = np.exp(db.mat)

    motif_names = db.names()
    D = db.mat.shape[0] // 4

    rows: list[dict] = []
    for col_idx, name in enumerate(motif_names):
        ml = db.motif_lengths[name]
        for pos in range(1, ml + 1):
            base_row = (pos - 1) * 4
            probs = prob_mat[base_row : base_row + 4, col_idx]
            # Reverse the prior: prob_orig = (sum(prob + prior) * prob - prior)
            total = np.sum(probs + db.prior)
            original = total * probs - db.prior
            rows.append(
                {
                    "motif": name,
                    "pos": pos,
                    "A": original[0],
                    "C": original[1],
                    "G": original[2],
                    "T": original[3],
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prior modification
# ---------------------------------------------------------------------------


def set_prior(db: MotifDB, new_prior: float) -> MotifDB:
    """Create a new MotifDB with a different prior.

    Mirrors the R ``prior<-`` replacement method.

    Parameters
    ----------
    db : MotifDB
        Original MotifDB.
    new_prior : float
        New prior value (must be in (0, 1)).

    Returns
    -------
    MotifDB
        New MotifDB with the updated prior.
    """
    df = motif_db_to_dataframe(db)
    return create_motif_db(
        df,
        prior=new_prior,
        spat_factors=db.spat_factors,
        spat_bin_size=db.spat_bin_size,
        spat_min=db.spat_min,
        spat_max=db.spat_max,
    )


# ---------------------------------------------------------------------------
# extract_pwm
# ---------------------------------------------------------------------------


def extract_pwm(
    sequences: list[str],
    motif_db: MotifDB | pd.DataFrame,
    *,
    motifs: list[str] | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
    func: str = "logSumExp",
    spat_min: int | None = None,
    spat_max: int | None = None,
) -> pd.DataFrame:
    """Compute PWM scores for all motifs in a database.

    For each motif in *motif_db*, extracts its PSSM and calls
    :func:`~pyprego.compute.compute_pwm`.

    Parameters
    ----------
    sequences : list[str]
        DNA sequences (should all be the same length).
    motif_db : MotifDB | pd.DataFrame
        A MotifDB object or a tidy DataFrame with ``motif`` column.
    motifs : list[str] | None
        Subset of motif names to extract. If ``None``, all motifs are used.
    bidirect : bool
        Score both orientations.
    prior : float
        PSSM prior.
    func : str
        Aggregation function (``"logSumExp"`` or ``"max"``).
    spat_min : int | None
        Minimum spatial position (1-based).
    spat_max : int | None
        Maximum spatial position.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per motif and one row per sequence.
    """
    if isinstance(motif_db, pd.DataFrame):
        motif_db = create_motif_db(motif_db, prior=prior)

    if motifs is not None:
        motif_db = motif_db[motifs]

    if motif_db.prior != prior:
        motif_db = set_prior(motif_db, prior)

    # Convert back to individual PSSMs and call compute_pwm for each
    df = motif_db_to_dataframe(motif_db)
    motif_names = motif_db.names()

    results: dict[str, np.ndarray] = {}
    for name in motif_names:
        motif_pssm = df[df["motif"] == name][["pos", "A", "C", "G", "T"]].copy()

        # Get spatial model for this motif
        idx = motif_names.index(name)
        spat_facs = motif_db.spat_factors[idx, :]
        if len(spat_facs) > 1 or motif_db.spat_bin_size > 1:
            bins = np.arange(len(spat_facs)) * motif_db.spat_bin_size
            spat = pd.DataFrame({"bin": bins, "spat_factor": spat_facs})
        else:
            spat = None

        s_min = spat_min
        s_max = spat_max
        if s_min is None and motif_db.spat_min is not None:
            s_min = int(motif_db.spat_min)
        if s_max is None and motif_db.spat_max is not None:
            s_max = int(motif_db.spat_max)

        scores = compute_pwm(
            sequences,
            motif_pssm,
            spat=spat,
            spat_min=s_min if s_min is not None else 1,
            spat_max=s_max,
            bidirect=bidirect,
            prior=prior,
            func=func,
        )
        results[name] = scores

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# screen_pwm
# ---------------------------------------------------------------------------


def _is_binary(response: np.ndarray) -> bool:
    """Check if a response vector is binary (only 0 and 1 values)."""
    unique_vals = np.unique(response[~np.isnan(response)])
    return len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0})


def screen_pwm(
    sequences: list[str],
    response: np.ndarray,
    motif_db: MotifDB | pd.DataFrame,
    *,
    motifs: list[str] | None = None,
    metric: str | None = None,
    bidirect: bool = True,
    prior: float = 0.01,
    only_best: bool = False,
) -> pd.DataFrame:
    """Screen all motifs in a database against a response variable.

    For each motif, computes the PWM score for all sequences and then
    correlates (or KS-tests) with the response.

    Parameters
    ----------
    sequences : list[str]
        DNA sequences.
    response : np.ndarray
        Response variable (1-D array, same length as *sequences*).
        For binary responses the KS metric is used by default; for
        continuous responses R-squared is used.
    motif_db : MotifDB | pd.DataFrame
        Motif database.
    motifs : list[str] | None
        Subset of motifs to screen.
    metric : str | None
        ``"r2"`` or ``"ks"``. If ``None``, auto-detected from *response*.
    bidirect : bool
        Score both orientations.
    prior : float
        PSSM prior.
    only_best : bool
        If ``True``, return only the top-scoring motif.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``motif`` and ``score``, sorted descending.
    """
    response = np.asarray(response, dtype=np.float64)

    if response.ndim == 2:
        response = np.nanmean(response, axis=1)

    if len(sequences) != len(response):
        raise ValueError(
            f"Number of sequences ({len(sequences)}) and response length "
            f"({len(response)}) do not match"
        )

    if np.any(pd.isna(sequences)):
        raise ValueError("There are missing values in the sequences")

    if metric is None:
        metric = "ks" if _is_binary(response) else "r2"

    if metric == "ks" and not _is_binary(response):
        raise ValueError("The metric cannot be 'ks' for a continuous response")

    if metric not in ("r2", "ks"):
        raise ValueError(f"metric must be 'r2' or 'ks', got {metric!r}")

    if isinstance(motif_db, pd.DataFrame):
        motif_db = create_motif_db(motif_db, prior=prior)

    if motifs is not None:
        motif_db = motif_db[motifs]

    df = motif_db_to_dataframe(motif_db)
    motif_names = motif_db.names()

    results: list[dict[str, object]] = []
    for name in motif_names:
        motif_pssm = df[df["motif"] == name][["pos", "A", "C", "G", "T"]].copy()
        pwm_scores = compute_pwm(
            sequences, motif_pssm, bidirect=bidirect, prior=prior
        )

        if metric == "ks":
            mask_pos = response.astype(bool)
            mask_neg = ~mask_pos
            if mask_pos.sum() > 0 and mask_neg.sum() > 0:
                stat, _ = scipy_stats.ks_2samp(
                    pwm_scores[mask_pos], pwm_scores[mask_neg]
                )
                score = float(stat)
            else:
                score = 0.0
        else:  # r2
            corr = np.corrcoef(pwm_scores, response)[0, 1]
            score = float(corr**2) if not np.isnan(corr) else 0.0

        results.append({"motif": name, "score": score})

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)

    if only_best:
        result_df = result_df.head(1)

    return result_df


# ---------------------------------------------------------------------------
# Loading motif datasets
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent / "data"


def _load_csv_dataset(path: str | Path) -> pd.DataFrame:
    """Load a motif dataset from a CSV file.

    Parameters
    ----------
    path : str | Path
        Path to CSV file with columns ``motif``, ``pos``, ``A``, ``C``,
        ``G``, ``T``.

    Returns
    -------
    pd.DataFrame
        Motif dataset.
    """
    df = pd.read_csv(path)
    required = {"motif", "pos", "A", "C", "G", "T"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV file must have columns {required}, got {set(df.columns)}"
        )
    return df


def all_motif_datasets(
    datasets: list[str] | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Load built-in motif datasets.

    Looks for CSV files in the bundled ``data/`` directory (or the R
    package's exported CSVs).

    Parameters
    ----------
    datasets : list[str] | None
        Which datasets to load (e.g. ``["HOMER", "JASPAR"]``).
        If ``None``, loads all available datasets.
    data_dir : str | Path | None
        Directory containing ``<NAME>_motifs.csv`` files.
        If ``None``, uses the bundled data directory and falls back to
        ``/tmp`` if files were exported from R.

    Returns
    -------
    pd.DataFrame
        Combined motif DataFrame with columns ``motif``, ``pos``, ``A``,
        ``C``, ``G``, ``T``, ``dataset``, ``motif_orig``.
    """
    if data_dir is None:
        if _DATA_DIR.exists():
            data_dir = _DATA_DIR
        else:
            data_dir = Path("/tmp")

    data_dir = Path(data_dir)

    available = ["HOMER", "JASPAR", "JOLMA", "HOCOMOCO"]
    if datasets is not None:
        for ds in datasets:
            if ds not in available:
                raise ValueError(
                    f"Unknown dataset {ds!r}, available: {available}"
                )
        to_load = datasets
    else:
        to_load = available

    frames: list[pd.DataFrame] = []
    for ds_name in to_load:
        path = data_dir / f"{ds_name}_motifs.csv"
        if not path.exists():
            warnings.warn(
                f"Dataset file not found: {path}. Skipping {ds_name}.",
                stacklevel=2,
            )
            continue
        df = _load_csv_dataset(path)
        df["dataset"] = ds_name
        df["motif_orig"] = df["motif"]
        df["motif"] = ds_name + "." + df["motif"].astype(str)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No motif dataset files found in {data_dir}. "
            f"Expected files like HOMER_motifs.csv, JASPAR_motifs.csv, etc."
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# get_motif_pssm
# ---------------------------------------------------------------------------


def get_motif_pssm(
    motif_name: str,
    dataset: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Get the PSSM for a specific motif by name.

    Parameters
    ----------
    motif_name : str
        Name of the motif (e.g. ``"JASPAR.HNF1A"``).
    dataset : pd.DataFrame | None
        Motif dataset DataFrame. If ``None``, loads via
        :func:`all_motif_datasets`.
    data_dir : str | Path | None
        Passed to :func:`all_motif_datasets` if *dataset* is ``None``.

    Returns
    -------
    pd.DataFrame
        PSSM DataFrame with columns ``pos``, ``A``, ``C``, ``G``, ``T``.

    Raises
    ------
    KeyError
        If *motif_name* is not found in the dataset.
    """
    if dataset is None:
        dataset = all_motif_datasets(data_dir=data_dir)

    mask = dataset["motif"] == motif_name
    if not mask.any():
        raise KeyError(f"Motif {motif_name!r} not found in dataset")

    return dataset.loc[mask, ["pos", "A", "C", "G", "T"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# motif_enrichment
# ---------------------------------------------------------------------------


def motif_enrichment(
    pwm_q: np.ndarray,
    groups: np.ndarray | list[str],
    threshold: float = 0.99,
    type: str = "relative",
) -> pd.DataFrame:
    """Calculate motif enrichment for groups of loci.

    Mirrors the R ``motif_enrichment()`` function. Given a matrix of PWM
    quantile values and group assignments, computes enrichment of motifs
    in each group.

    Parameters
    ----------
    pwm_q : np.ndarray
        Matrix of shape ``(n_loci, n_motifs)`` with quantile values.
    groups : np.ndarray | list[str]
        Group labels (length ``n_loci``).
    threshold : float
        Quantile threshold for considering a motif as present (default 0.99).
    type : str
        ``"relative"`` (enrichment vs other groups) or ``"absolute"``
        (enrichment vs random).

    Returns
    -------
    pd.DataFrame
        Enrichment matrix with groups as rows and motifs as columns.
    """
    pwm_q = np.asarray(pwm_q, dtype=np.float64)
    if pwm_q.ndim != 2:
        raise ValueError("pwm_q must be a 2-D matrix")

    groups = np.asarray(groups)
    if len(groups) != pwm_q.shape[0]:
        raise ValueError(
            "Number of rows in pwm_q must match length of groups"
        )

    if type not in ("relative", "absolute"):
        raise ValueError(f"type must be 'relative' or 'absolute', got {type!r}")

    # Binary matrix: motif is "present" if quantile >= threshold
    pwm_mf = (pwm_q >= threshold).astype(np.float64)

    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    n_motifs = pwm_q.shape[1]

    # Count motif occurrences per group
    group_motifs = np.zeros((n_groups, n_motifs), dtype=np.float64)
    group_n = np.zeros(n_groups, dtype=np.float64)

    for gi, g in enumerate(unique_groups):
        mask = groups == g
        group_n[gi] = mask.sum()
        group_motifs[gi, :] = pwm_mf[mask, :].sum(axis=0)

    n_fg_ok = group_motifs
    n_fg = group_n[:, np.newaxis] * np.ones((1, n_motifs))

    total_motif_occurrences = group_motifs.sum(axis=0)
    n_bg_ok = total_motif_occurrences[np.newaxis, :] - n_fg_ok

    total_loci = group_n.sum()
    n_bg = total_loci - n_fg

    if type == "relative":
        with np.errstate(divide="ignore", invalid="ignore"):
            enrichment = (n_fg_ok / n_fg) / (n_bg_ok / n_bg)
    else:  # absolute
        with np.errstate(divide="ignore", invalid="ignore"):
            enrichment = (n_fg_ok / n_fg) / (1 - threshold)

    return pd.DataFrame(
        enrichment,
        index=unique_groups,
        columns=[f"motif_{i}" for i in range(n_motifs)]
        if pwm_q.shape[1] > 0
        else [],
    )
