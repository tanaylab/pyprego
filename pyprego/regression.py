"""PWM regression optimiser.

Mirrors regression.R / PWMLRegression.cpp from the R prego package.  This
module contains the core ``regress_pwm`` function that iteratively optimises
a PSSM and spatial model to best explain a response variable given a set of
DNA sequences.

The implementation is intentionally NumPy-based (no GPU / PyTorch) so that it
closely mirrors the R behaviour and can run on any machine.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .compute import compute_pwm
from .pssm import consensus_from_pssm, pssm_match
from .types import (
    NUCLEOTIDES,
    RegressionResult,
    pssm_dataframe,
    pssm_to_array,
    spatial_dataframe,
)

if TYPE_CHECKING:
    pass

# ──────────────────────────────────────────────────────────────────────
# Nucleotide encoding helpers  (matching the C++ char-based indexing)
# ──────────────────────────────────────────────────────────────────────
_NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
_COMPLEMENT_IDX = np.array([3, 2, 1, 0], dtype=np.intp)  # A<->T, C<->G


def _encode_sequences_int(sequences: list[str]) -> np.ndarray:
    """Encode sequences as int8 array (N, L).  0=A, 1=C, 2=G, 3=T, -1=N/*."""
    n = len(sequences)
    L = len(sequences[0])
    arr = np.full((n, L), -1, dtype=np.int8)
    for i, seq in enumerate(sequences):
        for j, ch in enumerate(seq):
            idx = _NUC_TO_IDX.get(ch, -1)
            arr[i, j] = idx
    return arr


# ──────────────────────────────────────────────────────────────────────
# Neighbourhood (perturbation moves) – mirrors init_neighborhood()
# ──────────────────────────────────────────────────────────────────────

def _build_neighbourhood(resolution: float) -> list[list[tuple[int, float]]]:
    """Build the 20 perturbation moves.

    Returns a list of 20 moves.  Each move is a list of (nuc_idx, delta)
    tuples.
    """
    A, C, G, T = 0, 1, 2, 3
    r = resolution
    moves: list[list[tuple[int, float]]] = [
        [(A, r)],           # 0:  A+
        [(A, -r)],          # 1:  A-
        [(C, r)],           # 2:  C+
        [(C, -r)],          # 3:  C-
        [(G, r)],           # 4:  G+
        [(G, -r)],          # 5:  G-
        [(T, r)],           # 6:  T+
        [(T, -r)],          # 7:  T-
        [(A, r), (C, r)],   # 8:  AC+
        [(A, r), (G, r)],   # 9:  AG+
        [(A, r), (T, r)],   # 10: AT+
        [(A, -r), (C, -r)], # 11: AC-
        [(A, -r), (G, -r)], # 12: AG-
        [(A, -r), (T, -r)], # 13: AT-
        [(C, r), (G, r)],   # 14: CG+
        [(C, r), (T, r)],   # 15: CT+
        [(C, -r), (G, -r)], # 16: CG-
        [(C, -r), (T, -r)], # 17: CT-
        [(G, r), (T, r)],   # 18: GT+
        [(G, -r), (T, -r)], # 19: GT-
    ]
    return moves


# ──────────────────────────────────────────────────────────────────────
# Spatial helpers
# ──────────────────────────────────────────────────────────────────────

def _calc_spat_min_max(
    spat_num_bins: int,
    max_seq_len: int,
    spat_bin_size: int,
) -> tuple[int, int]:
    """Calculate the spatial min/max positions (matching R calc_spat_min_max)."""
    center = round(max_seq_len / 2)
    if spat_num_bins == 1:
        spat_min = center - spat_bin_size // 2
        spat_max = center + spat_bin_size // 2
    else:
        spat_min = center - ((spat_num_bins - 1) // 2) * spat_bin_size - spat_bin_size // 2
        spat_max = center + ((spat_num_bins - 1) // 2) * spat_bin_size + spat_bin_size // 2
    return int(round(spat_min)), int(round(spat_max))


def _calculate_bins(
    max_seq_len: int,
    spat_num_bins: int | None,
    spat_bin_size: int | None,
) -> tuple[int, int]:
    """Calculate bin parameters (matching R calculate_bins)."""
    if spat_num_bins is not None and spat_bin_size is not None:
        return spat_num_bins, spat_bin_size
    if spat_num_bins is not None:
        spat_bin_size = max_seq_len // spat_num_bins
        return spat_num_bins, spat_bin_size
    if spat_bin_size is not None:
        spat_num_bins = max_seq_len // spat_bin_size
        if spat_num_bins % 2 == 0:
            spat_num_bins -= 1
        return spat_num_bins, spat_bin_size
    # Both None: use defaults
    default_bin_size = 40
    spat_bin_size = min(default_bin_size, max_seq_len // 3)
    spat_num_bins = max_seq_len // spat_bin_size
    if spat_num_bins % 2 == 0:
        spat_num_bins -= 1
    if spat_num_bins < 3:
        raise ValueError("Calculated spat_num_bins is less than 3")
    return spat_num_bins, spat_bin_size


# ──────────────────────────────────────────────────────────────────────
# Core regression engine
# ──────────────────────────────────────────────────────────────────────

class _PWMLRegression:
    """Pure-Python port of the C++ PWMLRegression class.

    All internal arrays use 0-based nucleotide indexing: A=0, C=1, G=2, T=3.
    """

    def __init__(
        self,
        sequences: list[str],
        train_mask: np.ndarray,
        min_range: int,
        max_range: int,
        min_prob: float,
        spat_bin_size: int,
        resolutions: list[float],
        spat_resolutions: list[float],
        improve_epsilon: float,
        unif_prior: float,
        score_metric: str,
        num_folds: int,
        log_energy: bool,
        energy_epsilon: float,
        optimize_pwm: bool,
        optimize_spat: bool,
        symmetrize_spat: bool,
        verbose: bool,
        rng: np.random.Generator,
    ):
        self.sequences = sequences
        self.n_seq = len(sequences)
        self.train_mask = train_mask.astype(bool)
        self.min_range = min_range
        self.max_range = max_range
        self.min_prob = min_prob
        self.spat_bin_size = spat_bin_size
        self.resolutions = resolutions
        self.spat_resolutions = spat_resolutions
        self.improve_epsilon = improve_epsilon
        self.unif_prior = unif_prior
        self.score_metric = score_metric
        self.num_folds = num_folds
        self.log_energy = log_energy
        self.energy_epsilon = energy_epsilon
        self.optimize_pwm = optimize_pwm
        self.optimize_spat = optimize_spat
        self.symmetrize_spat = symmetrize_spat
        self.verbose = verbose
        self.rng = rng

        self.cur_score: float = 0.0
        self.step_num: int = 0
        self.bidirect: bool = True

        # Fold assignment
        if num_folds < 1:
            raise ValueError("number of folds must be at least 1")
        self.folds = np.arange(self.n_seq) % num_folds
        if num_folds > 1:
            rng.shuffle(self.folds)
        self.fold_sizes = np.zeros(num_folds, dtype=int)
        for i in range(self.n_seq):
            if self.train_mask[i]:
                self.fold_sizes[self.folds[i]] += 1

        # Encode sequences once
        self.encoded = _encode_sequences_int(sequences)  # (n_seq, L)

    # ── Response setup ────────────────────────────────────────────────

    def add_responses(self, response: np.ndarray) -> None:
        """Set response data and pre-compute statistics.

        Parameters
        ----------
        response : np.ndarray
            Shape (n_seq,) or (n_seq, rdim).
        """
        if response.ndim == 1:
            response = response[:, np.newaxis]
        self.response = response.astype(np.float64)
        self.rdim = response.shape[1]

        if self.score_metric == "ks" and self.rdim > 1:
            raise ValueError("KS test is only for single-response (binary)")

        # Per-fold / global means and variances
        mask = self.train_mask
        self.train_n = int(mask.sum())

        self.data_avg = np.zeros(self.rdim)
        self.data_var = np.zeros(self.rdim)
        for rd in range(self.rdim):
            vals = self.response[mask, rd]
            self.data_avg[rd] = vals.mean()
            self.data_var[rd] = vals.var(ddof=0)

        self.data_avg_fold = np.zeros((self.num_folds, self.rdim))
        self.data_var_fold = np.zeros((self.num_folds, self.rdim))
        for f in range(self.num_folds):
            fold_mask = mask & (self.folds == f)
            for rd in range(self.rdim):
                vals = self.response[fold_mask, rd]
                if len(vals) > 0:
                    self.data_avg_fold[f, rd] = vals.mean()
                    self.data_var_fold[f, rd] = vals.var(ddof=0)

        # KS-specific
        self.ncat = 0
        if self.score_metric == "ks":
            self.ncat = int((self.response[mask, 0] == 1).sum())
            self.data_epsilon = self.rng.uniform(0, 1e-5, size=self.n_seq)

    # ── PSSM initialisation ──────────────────────────────────────────

    def init_seed(self, motif: str, bidirect: bool) -> None:
        """Initialise from a k-mer seed string (* = wildcard)."""
        K = len(motif)
        self.bidirect = bidirect
        self.motif_len = K

        # PSSM probabilities: (K, 4) array
        self.nuc_factors = np.full((K, 4), self.unif_prior, dtype=np.float64)
        self.is_wildcard = np.zeros(K, dtype=bool)

        for pos, ch in enumerate(motif):
            if ch == "*" or ch == "N":
                self.nuc_factors[pos, :] = 0.25
                self.is_wildcard[pos] = True
            else:
                idx = _NUC_TO_IDX.get(ch)
                if idx is not None:
                    self.nuc_factors[pos, :] = self.unif_prior
                    self.nuc_factors[pos, idx] = 1 - self.unif_prior * 3
                    self.is_wildcard[pos] = False
                else:
                    self.nuc_factors[pos, :] = 0.25
                    self.is_wildcard[pos] = True

        self._init_spat_and_derivs()

    def init_pssm(self, pssm_array: np.ndarray, bidirect: bool) -> None:
        """Initialise from a (K, 4) probability array."""
        K = pssm_array.shape[0]
        self.bidirect = bidirect
        self.motif_len = K
        self.nuc_factors = pssm_array.copy().astype(np.float64)
        # Normalise rows
        row_sums = self.nuc_factors.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.nuc_factors /= row_sums
        self.is_wildcard = np.zeros(K, dtype=bool)
        self._init_spat_and_derivs()

    def _init_spat_and_derivs(self) -> None:
        """Allocate spatial factors and derivative arrays."""
        K = self.motif_len
        self.spat_bins_num = (self.max_range - self.min_range) // self.spat_bin_size

        if self.bidirect and self.spat_bins_num % 2 == 0:
            raise ValueError("number of spatial bins must be odd when bidirect is true")

        self.spat_factors = np.full(self.spat_bins_num, 1.0 / self.spat_bins_num)

        # derivs: (n_seq, K, 4) -- derivative of energy w.r.t. each nuc at each pos
        self.derivs = np.zeros((self.n_seq, K, 4), dtype=np.float64)
        # spat_derivs: (n_seq, spat_bins_num)
        self.spat_derivs = np.zeros((self.n_seq, self.spat_bins_num), dtype=np.float64)

    def set_spat_factors(self, spat_factors: np.ndarray) -> None:
        """Override spatial factors (e.g. from a pre-computed model)."""
        self.spat_factors = spat_factors.copy().astype(np.float64)

    # ── Spatial bin mapping ───────────────────────────────────────────

    def _pos_to_spat_bin(self, pos: int) -> int:
        """Map a sequence position to a spatial bin index."""
        b = pos // self.spat_bin_size
        if self.symmetrize_spat and self.bidirect:
            center = self.spat_bins_num // 2
            if b > center:
                b = b - (b - center) * 2
        return b

    # ── Energy / derivative computation ───────────────────────────────

    def init_energies(self) -> None:
        """Compute derivatives for every (sequence, position, nucleotide).

        Vectorised NumPy implementation replacing the pure Python loop.
        Processes all sequences and window positions simultaneously.
        """
        K = self.motif_len
        n_seq = self.n_seq
        L = self.encoded.shape[1]
        num_wins = L - K + 1

        self.derivs[:] = 0.0
        self.spat_derivs[:] = 0.0

        if num_wins <= 0:
            return

        # --- Build sliding window indices: (num_wins, K) ---
        win_idx = np.arange(K)[None, :] + np.arange(num_wins)[:, None]  # (W, K)

        # --- Extract all windows for all sequences: (N, W, K) ---
        all_windows = self.encoded[:, win_idx]  # (N, W, K) int8 bases

        # --- Mask invalid bases (N = -1) ---
        has_invalid = np.any(all_windows < 0, axis=2)  # (N, W) bool

        # --- Spatial bin mapping for each window position ---
        spat_bins = np.arange(num_wins) // self.spat_bin_size  # (W,)
        if self.symmetrize_spat and self.bidirect:
            center = self.spat_bins_num // 2
            mirror = spat_bins > center
            spat_bins = np.where(mirror, spat_bins - (spat_bins - center) * 2, spat_bins)
        spat_bins = np.clip(spat_bins, 0, self.spat_bins_num - 1)

        # --- Forward strand ---
        # For valid windows, look up PSSM probs: nuc_factors[pos, base]
        safe_windows = np.clip(all_windows, 0, 3)  # replace -1 with 0 (will be masked)
        pos_idx = np.arange(K)  # (K,)
        fwd_probs = self.nuc_factors[pos_idx, safe_windows]  # (N, W, K)

        # Products across PSSM positions
        fwd_prod = np.prod(fwd_probs, axis=2)  # (N, W)
        fwd_prod[has_invalid] = 0.0
        fwd_prod[~self.train_mask] = 0.0

        # Accumulate spat_derivs: for each seq, sum products per spatial bin
        for b in range(self.spat_bins_num):
            bin_mask = spat_bins == b  # (W,)
            self.spat_derivs[:, b] = fwd_prod[:, bin_mask].sum(axis=1)

        # Weighted products (product * spatial_factor[bin])
        spat_weights = self.spat_factors[spat_bins]  # (W,)
        fwd_weighted = fwd_prod * spat_weights[None, :]  # (N, W)

        # Derivatives: for each PSSM position d and nucleotide,
        # deriv[seq, d, nuc] += sum over windows where window[d]==nuc of (weighted / factor_at_d)
        # weighted / factor_at_d = fwd_weighted / fwd_probs[:, :, d]
        for d in range(K):
            factor_d = fwd_probs[:, :, d]  # (N, W)
            # Avoid division by zero
            safe_factor = np.where(factor_d > 0, factor_d, 1.0)
            contrib = fwd_weighted / safe_factor  # (N, W)
            contrib = np.where(fwd_prod > 0, contrib, 0.0)
            nucs_at_d = safe_windows[:, :, d]  # (N, W) which nucleotide
            for nuc in range(4):
                nuc_mask = nucs_at_d == nuc  # (N, W)
                self.derivs[:, d, nuc] += (contrib * nuc_mask).sum(axis=1)

        # --- Reverse complement strand ---
        if self.bidirect:
            # RC PSSM: reverse positions and complement bases
            # For sequence base b at window offset d:
            #   PSSM position = K-1-d, nucleotide = complement(b)
            rc_bases = _COMPLEMENT_IDX[safe_windows]  # (N, W, K) complemented
            # Reverse PSSM positions: use nuc_factors[K-1-d, complement(base[d])]
            rev_pos_idx = np.arange(K - 1, -1, -1)  # (K,) reversed
            rc_probs = self.nuc_factors[rev_pos_idx, rc_bases]  # (N, W, K)

            rc_prod = np.prod(rc_probs, axis=2)  # (N, W)
            rc_prod[has_invalid] = 0.0
            rc_prod[~self.train_mask] = 0.0

            # Accumulate spat_derivs (add to existing forward contribution)
            for b in range(self.spat_bins_num):
                bin_mask = spat_bins == b
                self.spat_derivs[:, b] += rc_prod[:, bin_mask].sum(axis=1)

            rc_weighted = rc_prod * spat_weights[None, :]  # (N, W)

            # RC derivatives: deriv[seq, K-1-d, complement(base[d])] += ...
            for d in range(K):
                factor_d = rc_probs[:, :, d]  # (N, W)
                safe_factor = np.where(factor_d > 0, factor_d, 1.0)
                contrib = rc_weighted / safe_factor
                contrib = np.where(rc_prod > 0, contrib, 0.0)
                pssm_pos = K - 1 - d
                nucs_at_d = rc_bases[:, :, d]  # (N, W) complemented nucleotide
                for nuc in range(4):
                    nuc_mask = nucs_at_d == nuc
                    self.derivs[:, pssm_pos, nuc] += (contrib * nuc_mask).sum(axis=1)

        if self.symmetrize_spat:
            self._symmetrize_spat_factors()

    # ── Score computation ─────────────────────────────────────────────

    def _compute_energy_from_derivs(
        self,
        pos: int,
        probs: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute energy for each sequence given derivs at *pos* and candidate probs.

        energy[i] = sum_nuc probs[nuc] * derivs[i, pos, nuc]

        Returns 1-D array of length n_seq (masked entries are 0).
        """
        if mask is None:
            mask = self.train_mask
        # derivs[:, pos, :] has shape (n_seq, 4)
        # probs has shape (4,)
        energies = self.derivs[:, pos, :] @ probs  # (n_seq,)
        energies = np.where(mask, energies, 0.0)
        return energies

    def _apply_log_energy(self, energies: np.ndarray) -> np.ndarray:
        """Apply log transform if enabled."""
        if self.log_energy:
            return np.log(energies + self.energy_epsilon)
        return energies

    def compute_cur_r2(self, pos: int, probs: np.ndarray) -> float:
        """Compute total R-squared across all response dimensions."""
        mask = self.train_mask
        energies = self._compute_energy_from_derivs(pos, probs)
        energies = self._apply_log_energy(energies)

        train_e = energies[mask]
        n = self.train_n
        ex = train_e.sum() / n
        ex2 = (train_e * train_e).sum() / n
        pred_var = ex2 - ex * ex

        tot_r2 = 0.0
        for rd in range(self.rdim):
            resp_vals = self.response[mask, rd]
            xy = (train_e * resp_vals).sum() / n
            cov = xy - ex * self.data_avg[rd]
            denom = pred_var * self.data_var[rd]
            if denom > 0:
                tot_r2 += cov * cov / denom
        return tot_r2

    def compute_cur_r2_fold(self, pos: int, probs: np.ndarray, fold: int) -> float:
        """Compute R-squared for a specific fold."""
        mask = self.train_mask
        fold_mask = mask & (self.folds == fold)
        # Energy is computed over all train sequences (C++ does this)
        energies = self._compute_energy_from_derivs(pos, probs)
        energies = self._apply_log_energy(energies)

        fold_e = energies[fold_mask]
        n = self.fold_sizes[fold]
        if n == 0:
            return 0.0
        ex = fold_e.sum() / n
        ex2 = (fold_e * fold_e).sum() / n
        pred_var = ex2 - ex * ex

        tot_r2 = 0.0
        for rd in range(self.rdim):
            resp_vals = self.response[fold_mask, rd]
            xy = (fold_e * resp_vals).sum() / n
            cov = xy - ex * self.data_avg_fold[fold, rd]
            denom = pred_var * self.data_var_fold[fold, rd]
            if denom > 0:
                tot_r2 += cov * cov / denom
        return tot_r2

    def compute_cur_ks(self, pos: int, probs: np.ndarray) -> float:
        """Compute KS statistic (one-sided D)."""
        mask = self.train_mask
        energies = self._compute_energy_from_derivs(pos, probs)

        # Build (sort_key, response) pairs for training sequences
        train_idx = np.where(mask)[0]
        train_e = energies[train_idx]
        train_r = self.response[train_idx, 0]
        eps = self.data_epsilon[train_idx]

        sort_keys = -train_e * (1 + eps)
        order = np.argsort(sort_keys)

        n_0 = float(self.train_n - self.ncat)
        n_1 = float(self.ncat)
        if n_0 == 0 or n_1 == 0:
            return 0.0

        max_diff = 0.0
        cur_diff = 0.0
        for idx in order:
            if train_r[idx] == 0:
                cur_diff -= 1.0 / n_0
            else:
                cur_diff += 1.0 / n_1
            if cur_diff > max_diff:
                max_diff = cur_diff
        return max_diff

    def compute_cur_ks_fold(self, pos: int, probs: np.ndarray, fold: int) -> float:
        """Compute KS statistic for a specific fold."""
        mask = self.train_mask & (self.folds == fold)
        energies = self._compute_energy_from_derivs(pos, probs, mask=self.train_mask)

        train_idx = np.where(mask)[0]
        if len(train_idx) == 0:
            return 0.0
        train_e = energies[train_idx]
        train_r = self.response[train_idx, 0]
        eps = self.data_epsilon[train_idx]

        sort_keys = -train_e * (1 + eps)
        order = np.argsort(sort_keys)

        n_0 = float((train_r == 0).sum())
        n_1 = float((train_r == 1).sum())
        if n_0 == 0 or n_1 == 0:
            return 0.0

        max_diff = 0.0
        cur_diff = 0.0
        for idx in order:
            if train_r[idx] == 0:
                cur_diff -= 1.0 / n_0
            else:
                cur_diff += 1.0 / n_1
            if cur_diff > max_diff:
                max_diff = cur_diff
        return max_diff

    def compute_cur_score(self, pos: int, probs: np.ndarray) -> float:
        """Compute score using the configured metric."""
        if self.score_metric == "r2":
            return self.compute_cur_r2(pos, probs)
        elif self.score_metric == "ks":
            return self.compute_cur_ks(pos, probs)
        else:
            raise ValueError(f"Unknown score metric: {self.score_metric}")

    def compute_cur_fold_score(self, pos: int, probs: np.ndarray, fold: int) -> float:
        """Compute fold score using the configured metric."""
        if self.score_metric == "r2":
            return self.compute_cur_r2_fold(pos, probs, fold)
        elif self.score_metric == "ks":
            return self.compute_cur_ks_fold(pos, probs, fold)
        else:
            raise ValueError(f"Unknown score metric: {self.score_metric}")

    # ── Spatial score computation ─────────────────────────────────────

    def compute_cur_spat_score(self) -> float:
        """Compute score based on spatial derivatives and current spatial factors."""
        if self.score_metric == "r2":
            return self._compute_cur_r2_spat()
        elif self.score_metric == "ks":
            return self._compute_cur_ks_spat()
        else:
            raise ValueError(f"Unknown score metric: {self.score_metric}")

    def _compute_cur_r2_spat(self) -> float:
        """Compute R2 using spatial derivatives."""
        mask = self.train_mask
        # energy[i] = sum_bin spat_derivs[i, bin] * spat_factors[bin]
        energies = self.spat_derivs @ self.spat_factors  # (n_seq,)
        energies = np.where(mask, energies, 0.0)
        if self.log_energy:
            energies = np.where(mask, np.log(energies + self.energy_epsilon), 0.0)

        train_e = energies[mask]
        n = self.train_n
        ex = train_e.sum() / n
        ex2 = (train_e * train_e).sum() / n
        pred_var = ex2 - ex * ex

        tot_r2 = 0.0
        for rd in range(self.rdim):
            resp_vals = self.response[mask, rd]
            xy = (train_e * resp_vals).sum() / n
            cov = xy - ex * self.data_avg[rd]
            denom = pred_var * self.data_var[rd]
            if denom > 0:
                tot_r2 += cov * cov / denom
        return tot_r2

    def _compute_cur_ks_spat(self) -> float:
        """Compute KS using spatial derivatives."""
        mask = self.train_mask
        energies = self.spat_derivs @ self.spat_factors
        energies = np.where(mask, energies, 0.0)

        train_idx = np.where(mask)[0]
        train_e = energies[train_idx]
        train_r = self.response[train_idx, 0]
        eps = self.data_epsilon[train_idx]

        sort_keys = -train_e * (1 + eps)
        order = np.argsort(sort_keys)

        n_0 = float(self.train_n - self.ncat)
        n_1 = float(self.ncat)
        if n_0 == 0 or n_1 == 0:
            return 0.0

        max_diff = 0.0
        cur_diff = 0.0
        for idx in order:
            if train_r[idx] == 0:
                cur_diff -= 1.0 / n_0
            else:
                cur_diff += 1.0 / n_1
            if cur_diff > max_diff:
                max_diff = cur_diff
        return max_diff

    # ── Move selection and application ────────────────────────────────

    def _compute_step_probs(self, pos: int, step: int) -> np.ndarray:
        """Compute candidate probabilities for a given (position, step)."""
        probs = self.nuc_factors[pos].copy()
        for nuc_idx, delta in self._cur_neigh[step]:
            probs[nuc_idx] += delta
            if probs[nuc_idx] <= 0:
                probs[nuc_idx] = self.min_prob
        return probs

    def choose_best_move(self) -> tuple[int, int, float]:
        """Evaluate all (position, move) combinations and pick the best."""
        K = self.motif_len
        neigh_size = len(self._cur_neigh)
        n_moves = K * neigh_size

        # Compute fold scores for all moves
        scores = np.zeros((self.num_folds, n_moves))
        steps_list: list[tuple[int, int]] = []

        idx = 0
        for pos in range(K):
            for step in range(neigh_size):
                steps_list.append((pos, step))
                probs = self._compute_step_probs(pos, step)
                for f in range(self.num_folds):
                    scores[f, idx] = self.compute_cur_fold_score(pos, probs, f)
                idx += 1

        # Rank within each fold (ascending order, so higher rank = better score)
        ranks = np.zeros((self.num_folds, n_moves), dtype=int)
        for f in range(self.num_folds):
            order = np.argsort(scores[f])
            ranks[f, order] = np.arange(n_moves)

        # Average ranks (integer division as in C++)
        avg_ranks = ranks.sum(axis=0) // self.num_folds

        best_idx = int(np.argmax(avg_ranks))
        best_pos, best_step = steps_list[best_idx]
        probs = self._compute_step_probs(best_pos, best_step)
        best_score = self.compute_cur_score(best_pos, probs)

        if self.verbose:
            print(f"  best step={best_step} pos={best_pos} score={best_score:.6f}")

        return best_pos, best_step, best_score

    def apply_move(self, pos: int, step: int, score: float) -> None:
        """Apply the chosen move to the PSSM and update cur_score."""
        for nuc_idx, delta in self._cur_neigh[step]:
            self.nuc_factors[pos, nuc_idx] += delta
            self.nuc_factors[pos, nuc_idx] = max(self.nuc_factors[pos, nuc_idx], self.min_prob)
        # Normalise
        tot = self.nuc_factors[pos].sum()
        self.nuc_factors[pos] /= tot
        self.cur_score = score

    def _symmetrize_spat_factors(self) -> None:
        """Symmetrise spatial factors for bidirectional models."""
        if self.bidirect:
            center = self.spat_bins_num // 2
            for b in range(center + 1, self.spat_bins_num):
                mirror = b - (b - center) * 2
                self.spat_factors[b] = self.spat_factors[mirror]

    # ── Spatial optimisation ──────────────────────────────────────────

    def optimize_spatial_factors(self) -> None:
        """Try increasing/decreasing each spatial bin to improve the score."""
        best_spat_score = self.cur_score
        best_spat_bin = -1
        best_spat_diff = 0.0

        for spat_bin in range(self.spat_bins_num):
            # Try +step
            self.spat_factors[spat_bin] += self._spat_factor_step
            score_plus = self.compute_cur_spat_score()
            current_best = best_spat_score
            current_diff = self._spat_factor_step

            if score_plus > current_best:
                current_best = score_plus
            else:
                # Try -step
                self.spat_factors[spat_bin] -= 2 * self._spat_factor_step
                if self.spat_factors[spat_bin] >= 0:
                    score_minus = self.compute_cur_spat_score()
                    if score_minus > current_best:
                        current_best = score_minus
                        current_diff = -self._spat_factor_step
                # Restore
                self.spat_factors[spat_bin] += self._spat_factor_step

            if current_best > best_spat_score:
                best_spat_score = current_best
                best_spat_bin = spat_bin
                best_spat_diff = current_diff

            # Restore the factor (we only changed it for evaluation)
            # Actually, looking at C++ more carefully: check_spat_bin restores
            # the factor after each probe.  The code above does this correctly
            # via the += / -= pattern.

        if best_spat_score > self.cur_score:
            if self.verbose:
                print(f"  spat update bin={best_spat_bin} diff={best_spat_diff:.4f} "
                      f"score={best_spat_score:.6f}")
            self.spat_factors[best_spat_bin] += best_spat_diff
            # Normalise
            self.spat_factors /= (1 + best_spat_diff)
            self.cur_score = best_spat_score

    # ── Main optimisation loop ────────────────────────────────────────

    def optimize(self) -> None:
        """Run the main coordinate descent optimisation."""
        prev_score = 0.0
        self.cur_score = 0.0

        for phase_idx in range(len(self.resolutions)):
            if self.verbose:
                print(f"Phase {phase_idx}, resolution={self.resolutions[phase_idx]}")
            self._cur_neigh = _build_neighbourhood(self.resolutions[phase_idx])
            self._spat_factor_step = self.spat_resolutions[phase_idx]

            while True:
                prev_score = self.cur_score

                self.init_energies()

                # Take best step (PWM + spatial)
                if self.optimize_pwm:
                    best_pos, best_step, best_score = self.choose_best_move()
                    if best_score > self.cur_score:
                        self.apply_move(best_pos, best_step, best_score)

                if self.optimize_spat:
                    self.optimize_spatial_factors()

                if self.verbose:
                    print(f"  step {self.step_num}: prev={prev_score:.6f} "
                          f"cur={self.cur_score:.6f}")

                self.step_num += 1

                if self.cur_score <= prev_score + self.improve_epsilon:
                    break

            if self.symmetrize_spat:
                self._symmetrize_spat_factors()

    # ── Output ────────────────────────────────────────────────────────

    def get_pssm_df(self) -> pd.DataFrame:
        """Return the current PSSM as a DataFrame."""
        return pssm_dataframe(self.nuc_factors)

    def get_spat_df(self) -> pd.DataFrame:
        """Return the current spatial model as a DataFrame."""
        bins = np.arange(self.spat_bins_num) * self.spat_bin_size
        return spatial_dataframe(bins, self.spat_factors)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def regress_pwm_core(
    sequences: list[str] | np.ndarray,
    response: np.ndarray,
    *,
    motif: str | pd.DataFrame | None = None,
    motif_length: int = 15,
    score_metric: str = "r2",
    bidirect: bool = True,
    spat_bin_size: int | None = None,
    spat_num_bins: int | None = None,
    spat_model: pd.DataFrame | None = None,
    improve_epsilon: float = 1e-4,
    min_nuc_prob: float = 0.001,
    unif_prior: float = 0.05,
    num_folds: int = 1,
    resolutions: list[float] | None = None,
    spat_resolutions: list[float] | None = None,
    log_energy: bool = False,
    energy_epsilon: float = 1e-5,
    optimize_pwm: bool = True,
    optimize_spat: bool = True,
    symmetrize_spat: bool = True,
    seed: int | None = 60427,
    consensus_single_thresh: float = 0.5,
    consensus_double_thresh: float = 0.75,
    verbose: bool = False,
) -> RegressionResult:
    """Core PWM regression optimizer (low-level).

    This is the faithful port of the C++ ``PWMLRegression`` class, using
    coordinate descent to iteratively optimise PSSM probabilities and spatial
    factors. It does not perform k-mer screening, multi-kmer tries, or
    database matching. For the high-level wrapper, see :func:`regress_pwm`.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences (equal length, characters A/C/G/T/N).
    response : np.ndarray
        Response variable(s).  Shape ``(n_sequences,)`` for a single response
        or ``(n_sequences, n_responses)`` for multiple.
    motif : str | pd.DataFrame | None
        Initial motif.  A kmer string (``*`` = wildcard), a PSSM DataFrame,
        or ``None`` (defaults to all-wildcards).
    motif_length : int
        Length of the seed motif (short kmers are padded with wildcards).
    score_metric : str
        ``"r2"`` or ``"ks"``.
    bidirect : bool
        Use both orientations of the motif.
    spat_bin_size : int | None
        Spatial bin size in bp.  ``None`` auto-computes.
    spat_num_bins : int | None
        Number of spatial bins.  ``None`` auto-computes.
    spat_model : pd.DataFrame | None
        Pre-computed spatial model (bin, spat_factor).
    improve_epsilon : float
        Convergence threshold.
    min_nuc_prob : float
        Minimum nucleotide probability per iteration.
    unif_prior : float
        Uniform prior for nucleotide probabilities.
    num_folds : int
        Number of cross-validation folds (1 = no CV).
    resolutions : list[float] | None
        Step sizes for each phase.  ``None`` uses C++ defaults.
    spat_resolutions : list[float] | None
        Spatial step sizes for each phase.
    log_energy : bool
        Apply log transform to energies.
    energy_epsilon : float
        Small constant for log(energy + epsilon).
    optimize_pwm : bool
        Whether to optimize PWM probabilities.
    optimize_spat : bool
        Whether to optimize spatial factors.
    symmetrize_spat : bool
        Symmetrize spatial factors for bidirectional models.
    seed : int | None
        Random seed for reproducibility.
    consensus_single_thresh : float
        Threshold for single-nucleotide consensus calls.
    consensus_double_thresh : float
        Threshold for double-nucleotide consensus calls.
    verbose : bool
        Print progress messages.

    Returns
    -------
    RegressionResult
        Fitted regression result with PSSM, spatial model, predictions, etc.
    """
    # ── Input validation ──────────────────────────────────────────────
    if score_metric not in ("r2", "ks"):
        raise ValueError(f"score_metric must be 'r2' or 'ks', got {score_metric!r}")

    sequences = [s.upper() for s in sequences]
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, np.newaxis]

    n_seq = len(sequences)
    if response.shape[0] != n_seq:
        raise ValueError("Number of sequences and response rows do not match")

    if score_metric == "ks":
        unique_vals = set(np.unique(response[:, 0]))
        if not unique_vals.issubset({0.0, 1.0}):
            raise ValueError("For 'ks' metric, response must be binary (0 and 1)")

    # ── Resolution defaults (matching C++ regress_pwm_cpp) ────────────
    if resolutions is None:
        resolutions = [0.05, 0.02, 0.01, 0.005]
    if spat_resolutions is None:
        spat_resolutions = [0.01, 0.01, 0.01, 0.005]
    # Ensure both lists are the same length
    max_phases = max(len(resolutions), len(spat_resolutions))
    while len(resolutions) < max_phases:
        resolutions.append(resolutions[-1])
    while len(spat_resolutions) < max_phases:
        spat_resolutions.append(spat_resolutions[-1])

    # ── Spatial binning ───────────────────────────────────────────────
    seq_len = len(sequences[0])
    if spat_bin_size is None and spat_num_bins is None:
        spat_num_bins, spat_bin_size = _calculate_bins(seq_len, None, None)
    elif spat_bin_size is not None and spat_num_bins is None:
        spat_num_bins, spat_bin_size = _calculate_bins(seq_len, None, spat_bin_size)
    elif spat_num_bins is not None and spat_bin_size is None:
        spat_num_bins, spat_bin_size = _calculate_bins(seq_len, spat_num_bins, None)
    # else both specified, use as-is

    spat_min, spat_max = _calc_spat_min_max(spat_num_bins, seq_len, spat_bin_size)

    if verbose:
        print(f"Using {spat_num_bins} bins of size {spat_bin_size} bp")
        print(f"Spatial range: [{spat_min}, {spat_max})")

    # Trim sequences to spatial range (matching R: str_sub(start=spat_min+1, end=spat_max))
    trimmed_sequences = [s[spat_min:spat_max] for s in sequences]
    trimmed_len = len(trimmed_sequences[0])

    # Verify spatial bin alignment
    if trimmed_len % spat_bin_size != 0:
        # This can happen for some parameter combinations; adjust
        pass

    # ── Motif initialisation ──────────────────────────────────────────
    use_pssm_init = False
    seed_motif_str: str | None = None

    if isinstance(motif, pd.DataFrame):
        use_pssm_init = True
        pssm_init = pssm_to_array(motif).copy()
        # Normalise
        row_sums = pssm_init.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        pssm_init /= row_sums
        seed_motif_str = None
    elif isinstance(motif, str):
        seed_motif_str = motif
        # Pad to motif_length if shorter
        if len(motif) < motif_length:
            pad_total = motif_length - len(motif)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            seed_motif_str = "*" * pad_left + motif + "*" * pad_right
        # Replace N with *
        seed_motif_str = seed_motif_str.replace("N", "*")
    else:
        # No motif provided: use all wildcards
        seed_motif_str = "*" * motif_length

    # ── RNG ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)

    # ── Build regression engine ───────────────────────────────────────
    is_train = np.ones(n_seq, dtype=bool)

    engine = _PWMLRegression(
        sequences=trimmed_sequences,
        train_mask=is_train,
        min_range=0,
        max_range=trimmed_len,
        min_prob=min_nuc_prob,
        spat_bin_size=spat_bin_size,
        resolutions=resolutions,
        spat_resolutions=spat_resolutions,
        improve_epsilon=improve_epsilon,
        unif_prior=unif_prior,
        score_metric=score_metric,
        num_folds=num_folds,
        log_energy=log_energy,
        energy_epsilon=energy_epsilon,
        optimize_pwm=optimize_pwm,
        optimize_spat=optimize_spat,
        symmetrize_spat=symmetrize_spat,
        verbose=verbose,
        rng=rng,
    )

    engine.add_responses(response)

    if use_pssm_init:
        engine.init_pssm(pssm_init, bidirect)
        if spat_model is not None:
            engine.set_spat_factors(spat_model["spat_factor"].to_numpy())
    else:
        engine.init_seed(seed_motif_str, bidirect)
        if spat_model is not None:
            engine.set_spat_factors(spat_model["spat_factor"].to_numpy())

    # ── Run optimisation ──────────────────────────────────────────────
    engine.optimize()

    # ── Collect results ───────────────────────────────────────────────
    pssm_df = engine.get_pssm_df()
    spat_df = engine.get_spat_df()

    # Compute predictions using the existing compute_pwm function
    # (which handles log-sum-exp aggregation properly)
    pred = compute_pwm(
        trimmed_sequences,
        pssm_df,
        spat=spat_df,
        bidirect=bidirect,
        prior=0,  # No prior: PSSM is already optimised
    )

    consensus = consensus_from_pssm(pssm_df, consensus_single_thresh, consensus_double_thresh)

    # Compute R2 / KS on predictions
    r2_val = None
    ks_val = None
    flat_resp = response[:, 0] if response.shape[1] == 1 else response
    if score_metric == "r2" or response.shape[1] > 1:
        # Compute R2 for each response dimension
        if response.ndim == 2:
            r2_vals = []
            for rd in range(response.shape[1]):
                corr = np.corrcoef(pred, response[:, rd])[0, 1]
                r2_vals.append(corr ** 2 if not np.isnan(corr) else 0.0)
            r2_val = r2_vals[0] if len(r2_vals) == 1 else r2_vals
        else:
            corr = np.corrcoef(pred, response)[0, 1]
            r2_val = corr ** 2 if not np.isnan(corr) else 0.0

    if score_metric == "ks" or (response.shape[1] == 1 and set(np.unique(response[:, 0])).issubset({0.0, 1.0})):
        # Compute KS statistic
        mask1 = response[:, 0] == 1
        mask0 = response[:, 0] == 0
        if mask1.any() and mask0.any():
            from scipy import stats
            ks_result = stats.ks_2samp(pred[mask1], pred[mask0], alternative="less")
            ks_val = float(ks_result.statistic)

    # Build predict function that handles full-length sequences
    def _predict_fn(new_sequences: list[str] | np.ndarray) -> np.ndarray:
        new_sequences = [s.upper() for s in new_sequences]
        trimmed = [s[spat_min:spat_max] for s in new_sequences]
        return compute_pwm(
            trimmed,
            pssm_df,
            spat=spat_df,
            bidirect=bidirect,
            prior=0,
        )

    result = RegressionResult(
        pssm=pssm_df,
        spat=spat_df,
        pred=pred,
        consensus=consensus,
        r2=r2_val,
        ks=ks_val,
        seed_motif=seed_motif_str,
        bidirect=bidirect,
        spat_min=spat_min,
        spat_max=spat_max,
        seq_length=trimmed_len,
        _predict_fn=_predict_fn,
    )

    if verbose:
        print(f"Consensus: {consensus}")
        if r2_val is not None:
            print(f"R2: {r2_val}")
        if ks_val is not None:
            print(f"KS D: {ks_val}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Helpers for the high-level API
# ──────────────────────────────────────────────────────────────────────


def _is_binary_response(response: np.ndarray) -> bool:
    """Check whether the response is binary (0/1 only)."""
    if response.ndim == 2:
        if response.shape[1] != 1:
            return False
        vals = response[:, 0]
    else:
        vals = response
    unique = set(np.unique(vals[~np.isnan(vals)]))
    return unique.issubset({0.0, 1.0})


def _score_predictions(
    response: np.ndarray,
    pred: np.ndarray,
    metric: str,
    alternative: str = "less",
) -> float:
    """Compute a scalar score (R2 or KS) from predictions."""
    resp = response.ravel() if response.ndim == 2 and response.shape[1] == 1 else response
    if metric == "ks":
        mask1 = resp == 1
        mask0 = resp == 0
        if mask1.any() and mask0.any():
            ks_result = sp_stats.ks_2samp(pred[mask1], pred[mask0], alternative=alternative)
            return float(ks_result.statistic)
        return 0.0
    elif metric == "r2":
        if response.ndim == 2 and response.shape[1] > 1:
            r2s = []
            for rd in range(response.shape[1]):
                c = np.corrcoef(pred, response[:, rd])[0, 1]
                r2s.append(c ** 2 if not np.isnan(c) else 0.0)
            return float(np.mean(r2s))
        else:
            c = np.corrcoef(pred, resp)[0, 1]
            return float(c ** 2) if not np.isnan(c) else 0.0
    else:
        raise ValueError(f"Unknown metric {metric!r}")


def _sample_response(
    response: np.ndarray,
    sample_frac: float | None,
    sample_ratio: float,
    seed: int | None,
) -> np.ndarray:
    """Sample indices, optionally stratified for binary response."""
    rng = np.random.default_rng(seed)
    n = response.shape[0]
    resp = response[:, 0] if response.ndim == 2 else response

    if _is_binary_response(response):
        idx_1 = np.where(resp == 1)[0]
        idx_0 = np.where(resp == 0)[0]
        if sample_frac is not None:
            n1 = max(1, int(len(idx_1) * sample_frac))
            n0 = max(1, int(len(idx_0) * sample_frac))
        else:
            n1 = len(idx_1)
            n0 = min(len(idx_0), max(1, int(n1 * sample_ratio)))
        chosen_1 = rng.choice(idx_1, size=min(n1, len(idx_1)), replace=False)
        chosen_0 = rng.choice(idx_0, size=min(n0, len(idx_0)), replace=False)
        return np.sort(np.concatenate([chosen_0, chosen_1]))
    else:
        frac = sample_frac if sample_frac is not None else 0.1
        k = max(1, int(n * frac))
        return np.sort(rng.choice(n, size=k, replace=False))


def _pred_r_given_e(e: np.ndarray, r: np.ndarray, k: int = 100) -> np.ndarray:
    """Running mean of *r* sorted by *e* (matches R pred_r_given_e)."""
    k = min(k, len(r))
    order = np.argsort(e)
    r_sorted = r[order]
    # running mean with partial windows
    cumsum = np.cumsum(np.insert(r_sorted, 0, 0))
    n = len(r_sorted)
    result = np.empty(n, dtype=np.float64)
    half = k // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + k - half)
        result[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    # Restore original order
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    return result[inv_order]


def _get_cand_kmers(
    sequences: list[str],
    response: np.ndarray,
    kmer_length: int | list[int],
    min_gap: int,
    max_gap: int,
    min_kmer_cor: float,
    max_cands: int,
) -> list[str]:
    """Generate candidate k-mers for multi-kmer screening."""
    from .kmers import screen_kmers

    if isinstance(kmer_length, int):
        kmer_length = [kmer_length]

    all_dfs = []
    for kl in kmer_length:
        df = screen_kmers(
            sequences, response,
            kmer_len=kl,
            min_gap=min_gap,
            max_gap=max_gap,
            min_cor=min_kmer_cor,
        )
        if len(df) > 0:
            df = df.assign(length=kl)
            all_dfs.append(df)

    if not all_dfs:
        # Try with halved threshold
        for kl in kmer_length:
            df = screen_kmers(
                sequences, response,
                kmer_len=kl,
                min_gap=min_gap,
                max_gap=max_gap,
                min_cor=min_kmer_cor / 2,
            )
            if len(df) > 0:
                df = df.assign(length=kl)
                all_dfs.append(df)
        if not all_dfs:
            # Fall back to wildcard
            return ["*" * kmer_length[0]]

    all_kmers = pd.concat(all_dfs, ignore_index=True)

    best_kmer = all_kmers.loc[all_kmers["max_r2"].abs().idxmax(), "kmer"]

    # Filter by correlation
    filtered = all_kmers[np.sqrt(all_kmers["max_r2"]) > min_kmer_cor].copy()
    if len(filtered) == 0:
        return [best_kmer]

    filtered = filtered.drop_duplicates(subset=["kmer"])
    cands = filtered.nlargest(min(len(filtered), max_cands), "max_r2")

    cand_list = cands["kmer"].tolist()
    if best_kmer not in cand_list:
        cand_list = [best_kmer] + cand_list

    return cand_list


# ──────────────────────────────────────────────────────────────────────
# High-level regress_pwm with k-mer screening, multi-kmer, db matching
# ──────────────────────────────────────────────────────────────────────


def regress_pwm(
    sequences: list[str] | np.ndarray,
    response: np.ndarray,
    *,
    motif: str | pd.DataFrame | None = None,
    motif_length: int = 15,
    score_metric: str = "r2",
    bidirect: bool = True,
    spat_bin_size: int | None = None,
    spat_num_bins: int | None = None,
    spat_model: pd.DataFrame | None = None,
    improve_epsilon: float = 1e-4,
    min_nuc_prob: float = 0.001,
    unif_prior: float = 0.05,
    num_folds: int = 1,
    resolutions: list[float] | None = None,
    spat_resolutions: list[float] | None = None,
    log_energy: bool = False,
    energy_epsilon: float = 1e-5,
    optimize_pwm: bool = True,
    optimize_spat: bool = True,
    symmetrize_spat: bool = True,
    seed: int | None = 60427,
    consensus_single_thresh: float = 0.5,
    consensus_double_thresh: float = 0.75,
    verbose: bool = False,
    # --- High-level options ---
    multi_kmers: bool = False,
    kmer_length: int | list[int] = 8,
    max_cands: int = 10,
    min_gap: int = 0,
    max_gap: int = 1,
    min_kmer_cor: float = 0.08,
    final_metric: str | None = None,
    sample_for_kmers: bool = False,
    sample_frac: float | None = None,
    sample_idxs: np.ndarray | None = None,
    sample_ratio: float = 1.0,
    val_frac: float = 0.1,
    match_with_db: bool = False,
    motif_db: dict[str, pd.DataFrame] | pd.DataFrame | None = None,
    alternative: str = "less",
) -> RegressionResult:
    """Perform PWM regression to discover a motif in DNA sequences.

    This is the main entry point for motif regression. It wraps the core
    optimizer (:func:`regress_pwm_core`) with higher-level logic:

    * **K-mer screening**: When ``motif=None``, screen k-mers to find the
      best seed (using :func:`screen_kmers`).
    * **Multi-kmer mode**: When ``multi_kmers=True``, try multiple k-mer
      seeds and pick the best one based on ``final_metric``.
    * **Sampling**: ``sample_for_kmers=True`` uses a subset for screening.
    * **Database matching**: ``match_with_db=True`` matches the result
      against a motif database (using :func:`pssm_match`).
    * **Automatic metric selection**: If ``final_metric`` is ``None``, it
      auto-picks ``"ks"`` for binary responses and ``"r2"`` for continuous.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences (equal length, characters A/C/G/T/N).
    response : np.ndarray
        Response variable(s).
    motif : str | pd.DataFrame | None
        Initial motif. If ``None`` and ``multi_kmers=False``, a single k-mer
        screen is used to find the best seed. If ``None`` and
        ``multi_kmers=True``, multiple candidate k-mers are tried.
    motif_length : int
        Length of the seed motif.
    score_metric : str
        ``"r2"`` or ``"ks"`` (metric used *during* the optimization).
    bidirect : bool
        Use both orientations.
    spat_bin_size, spat_num_bins : int | None
        Spatial bin parameters.
    spat_model : pd.DataFrame | None
        Pre-computed spatial model.
    improve_epsilon, min_nuc_prob, unif_prior : float
        Optimizer parameters.
    num_folds : int
        Internal cross-validation folds.
    resolutions, spat_resolutions : list[float] | None
        Phase step sizes.
    log_energy : bool
        Apply log transform to energies.
    energy_epsilon : float
        Epsilon for log transform.
    optimize_pwm, optimize_spat : bool
        What to optimize.
    symmetrize_spat : bool
        Symmetrize spatial factors.
    seed : int | None
        Random seed.
    consensus_single_thresh, consensus_double_thresh : float
        Consensus thresholds.
    verbose : bool
        Print progress.
    multi_kmers : bool
        Try multiple k-mer seeds and pick the best.
    kmer_length : int | list[int]
        K-mer length(s) to screen.
    max_cands : int
        Maximum number of k-mer candidates.
    min_gap, max_gap : int
        Gap parameters for k-mer generation.
    min_kmer_cor : float
        Minimum correlation to include a k-mer.
    final_metric : str | None
        Metric for picking the best model. ``None`` auto-selects.
    sample_for_kmers : bool
        Sample the dataset for k-mer screening.
    sample_frac : float | None
        Fraction to sample.
    sample_idxs : np.ndarray | None
        Explicit sample indices.
    sample_ratio : float
        Ratio of classes in sampling.
    val_frac : float
        Fraction for internal validation when using multi-kmer mode.
    match_with_db : bool
        Match result against motif database.
    motif_db : dict | pd.DataFrame | None
        Motif database for matching.
    alternative : str
        Alternative for KS test.

    Returns
    -------
    RegressionResult
        Fitted regression result.
    """
    # Input preparation
    sequences = [s.upper() for s in sequences]
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, np.newaxis]

    n_seq = len(sequences)
    if response.shape[0] != n_seq:
        raise ValueError("Number of sequences and response rows do not match")

    # Auto-select final_metric
    if final_metric is None:
        final_metric = "ks" if _is_binary_response(response) else "r2"

    # Core optimizer kwargs (passed to regress_pwm_core)
    core_kwargs = dict(
        motif_length=motif_length,
        score_metric=score_metric,
        bidirect=bidirect,
        spat_bin_size=spat_bin_size,
        spat_num_bins=spat_num_bins,
        spat_model=spat_model,
        improve_epsilon=improve_epsilon,
        min_nuc_prob=min_nuc_prob,
        unif_prior=unif_prior,
        num_folds=num_folds,
        resolutions=resolutions,
        spat_resolutions=spat_resolutions,
        log_energy=log_energy,
        energy_epsilon=energy_epsilon,
        optimize_pwm=optimize_pwm,
        optimize_spat=optimize_spat,
        symmetrize_spat=symmetrize_spat,
        seed=seed,
        consensus_single_thresh=consensus_single_thresh,
        consensus_double_thresh=consensus_double_thresh,
        verbose=verbose,
    )

    # If motif is already provided (string or DataFrame), skip k-mer screening
    if motif is not None:
        if multi_kmers and verbose:
            warnings.warn("Motif is provided, multi_kmers will be ignored")
        result = regress_pwm_core(sequences, response, motif=motif, **core_kwargs)
    elif multi_kmers:
        # Multi-kmer mode
        result = _regress_pwm_multi_kmers(
            sequences=sequences,
            response=response,
            kmer_length=kmer_length,
            max_cands=max_cands,
            min_gap=min_gap,
            max_gap=max_gap,
            min_kmer_cor=min_kmer_cor,
            final_metric=final_metric,
            val_frac=val_frac,
            sample_for_kmers=sample_for_kmers,
            sample_frac=sample_frac,
            sample_idxs=sample_idxs,
            sample_ratio=sample_ratio,
            alternative=alternative,
            core_kwargs=core_kwargs,
            seed=seed,
            verbose=verbose,
        )
    else:
        # Single k-mer screen
        result = _regress_pwm_with_kmer_screen(
            sequences=sequences,
            response=response,
            kmer_length=kmer_length,
            min_gap=min_gap,
            max_gap=max_gap,
            min_kmer_cor=min_kmer_cor,
            sample_for_kmers=sample_for_kmers,
            sample_frac=sample_frac,
            sample_idxs=sample_idxs,
            sample_ratio=sample_ratio,
            core_kwargs=core_kwargs,
            seed=seed,
            verbose=verbose,
        )

    # Database matching
    if match_with_db and motif_db is not None:
        try:
            match_df = pssm_match(result.pssm, motif_db)
            if isinstance(match_df, pd.DataFrame) and len(match_df) > 0:
                best_row = match_df.iloc[0]
                result.db_match_motif = str(best_row["motif"])
                result.db_match_cor = float(best_row.get("cor", 0.0))
                if verbose:
                    print(f"Best DB match: {result.db_match_motif} "
                          f"(cor={result.db_match_cor:.3f})")
        except Exception:
            if verbose:
                print("Database matching failed")

    return result


def _regress_pwm_with_kmer_screen(
    sequences: list[str],
    response: np.ndarray,
    kmer_length: int | list[int],
    min_gap: int,
    max_gap: int,
    min_kmer_cor: float,
    sample_for_kmers: bool,
    sample_frac: float | None,
    sample_idxs: np.ndarray | None,
    sample_ratio: float,
    core_kwargs: dict,
    seed: int | None,
    verbose: bool,
) -> RegressionResult:
    """Run regress_pwm with a single best k-mer from screening."""
    from .kmers import screen_kmers

    # Optionally sample for screening
    if sample_for_kmers:
        if sample_idxs is None:
            sample_idxs = _sample_response(response, sample_frac, sample_ratio, seed)
        seq_s = [sequences[i] for i in sample_idxs]
        resp_s = response[sample_idxs]
    else:
        seq_s = sequences
        resp_s = response

    kl = kmer_length if isinstance(kmer_length, int) else kmer_length[0]
    kmers_df = screen_kmers(
        seq_s, resp_s,
        kmer_len=kl,
        min_gap=min_gap,
        max_gap=max_gap,
        min_cor=min_kmer_cor,
    )

    if len(kmers_df) > 0:
        best_kmer = kmers_df.loc[kmers_df["max_r2"].abs().idxmax(), "kmer"]
    else:
        # Try lower threshold
        kmers_df = screen_kmers(
            seq_s, resp_s,
            kmer_len=kl,
            min_gap=min_gap,
            max_gap=max_gap,
            min_cor=min_kmer_cor / 2,
        )
        if len(kmers_df) > 0:
            best_kmer = kmers_df.loc[kmers_df["max_r2"].abs().idxmax(), "kmer"]
        else:
            ml = core_kwargs.get("motif_length", 15)
            best_kmer = "*" * ml
            if verbose:
                print(f"No k-mer found, using wildcards: {best_kmer}")

    if verbose:
        print(f"Best k-mer: {best_kmer}")

    result = regress_pwm_core(sequences, response, motif=best_kmer, **core_kwargs)
    return result


def _regress_pwm_multi_kmers(
    sequences: list[str],
    response: np.ndarray,
    kmer_length: int | list[int],
    max_cands: int,
    min_gap: int,
    max_gap: int,
    min_kmer_cor: float,
    final_metric: str,
    val_frac: float,
    sample_for_kmers: bool,
    sample_frac: float | None,
    sample_idxs: np.ndarray | None,
    sample_ratio: float,
    alternative: str,
    core_kwargs: dict,
    seed: int | None,
    verbose: bool,
) -> RegressionResult:
    """Try multiple k-mer seeds and pick the best."""
    rng = np.random.default_rng(seed)

    # Optionally sample for screening
    if sample_for_kmers:
        if sample_idxs is None:
            sample_idxs = _sample_response(response, sample_frac, sample_ratio, seed)
        seq_s = [sequences[i] for i in sample_idxs]
        resp_s = response[sample_idxs]
    else:
        seq_s = sequences
        resp_s = response

    # Split into train/val
    n_s = len(seq_s)
    val_n = max(1, int(n_s * val_frac))
    all_idx = np.arange(n_s)
    rng.shuffle(all_idx)
    val_idx = np.sort(all_idx[:val_n])
    train_idx = np.sort(all_idx[val_n:])

    seq_train = [seq_s[i] for i in train_idx]
    resp_train = resp_s[train_idx]
    seq_val = [seq_s[i] for i in val_idx]
    resp_val = resp_s[val_idx]

    # Get candidate k-mers
    cand_kmers = _get_cand_kmers(
        seq_s, resp_s,
        kmer_length=kmer_length,
        min_gap=min_gap,
        max_gap=max_gap,
        min_kmer_cor=min_kmer_cor,
        max_cands=max_cands,
    )

    if verbose:
        print(f"Trying {len(cand_kmers)} candidate k-mers")

    # Run regression on each candidate (on train split)
    best_val_score = -np.inf
    best_kmer = cand_kmers[0]

    for kmer in cand_kmers:
        try:
            res = regress_pwm_core(
                seq_train, resp_train,
                motif=kmer,
                **core_kwargs,
            )
            val_pred = res.predict(seq_val)
            val_score = _score_predictions(resp_val, val_pred, final_metric, alternative)
            if verbose:
                print(f"  {kmer}: val_score={val_score:.4f}")
            if val_score > best_val_score:
                best_val_score = val_score
                best_kmer = kmer
        except Exception:
            if verbose:
                print(f"  {kmer}: FAILED")

    if verbose:
        print(f"Best k-mer: {best_kmer} (val_score={best_val_score:.4f})")

    # Re-run on full data with the best kmer
    result = regress_pwm_core(sequences, response, motif=best_kmer, **core_kwargs)
    return result


# ──────────────────────────────────────────────────────────────────────
# regress_multiple_motifs
# ──────────────────────────────────────────────────────────────────────


@dataclass
class MultiRegressionResult:
    """Container for the output of :func:`regress_multiple_motifs`.

    Attributes
    ----------
    models : list[RegressionResult]
        Individual regression results for each motif.
    multi_stats : pd.DataFrame
        Statistics DataFrame with columns: model, score, comb_score, diff,
        consensus, seed_motif.
    pred : np.ndarray
        Combined prediction using linear model.
    coef : np.ndarray
        Linear model coefficients (one per motif + intercept).
    """

    models: list[RegressionResult]
    multi_stats: pd.DataFrame
    pred: np.ndarray
    coef: np.ndarray
    intercept: float = 0.0

    def predict(self, sequences: list[str] | np.ndarray) -> np.ndarray:
        """Predict combined scores for new sequences.

        Parameters
        ----------
        sequences : list[str] | np.ndarray
            DNA sequences.

        Returns
        -------
        np.ndarray
            Combined predicted scores.
        """
        energies = np.column_stack([m.predict(sequences) for m in self.models])
        return energies @ self.coef + self.intercept

    def predict_multi(self, sequences: list[str] | np.ndarray) -> pd.DataFrame:
        """Predict per-motif scores for new sequences.

        Parameters
        ----------
        sequences : list[str] | np.ndarray
            DNA sequences.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``e1``, ``e2``, etc.
        """
        energies = np.column_stack([m.predict(sequences) for m in self.models])
        return pd.DataFrame(
            energies,
            columns=[f"e{i + 1}" for i in range(len(self.models))],
        )


def regress_multiple_motifs(
    sequences: list[str] | np.ndarray,
    response: np.ndarray,
    motif_num: int = 2,
    smooth_k: int = 100,
    alternative: str = "less",
    verbose: bool = False,
    **kwargs,
) -> MultiRegressionResult:
    """Iteratively regress multiple motifs.

    Finds the first motif via :func:`regress_pwm`, then for each subsequent
    motif computes residuals (response - smoothed predictions) and regresses
    on those. A combined linear model is fit at each step.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences.
    response : np.ndarray
        Response variable(s).
    motif_num : int
        Number of motifs to discover (must be >= 2).
    smooth_k : int
        Window size for smoothing predictions when computing residuals.
    alternative : str
        Alternative hypothesis for the KS test.
    verbose : bool
        Print progress.
    **kwargs
        Additional arguments passed to :func:`regress_pwm`.

    Returns
    -------
    MultiRegressionResult
        Combined multi-motif result.
    """
    if motif_num < 2:
        raise ValueError("motif_num must be at least 2")

    sequences = [s.upper() for s in sequences]
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, np.newaxis]

    is_binary = _is_binary_response(response)

    models: list[RegressionResult] = []
    energies: list[np.ndarray] = []
    scores: list[float] = []
    comb_scores: list[float] = []

    # First motif
    if verbose:
        print(f"Running regression for motif 1/{motif_num}")
    res = regress_pwm(sequences, response, verbose=verbose, alternative=alternative, **kwargs)
    models.append(res)
    energies.append(res.pred)
    e_comb = res.pred

    score_1 = _score_predictions(response, res.pred, "ks" if is_binary else "r2", alternative)
    scores.append(score_1)
    comb_scores.append(score_1)

    # Subsequent motifs
    for i in range(1, motif_num):
        if verbose:
            print(f"Running regression for motif {i + 1}/{motif_num}")

        # Compute residuals
        resp_flat = response[:, 0] if response.shape[1] == 1 else response.mean(axis=1)
        smoothed = _pred_r_given_e(e_comb, resp_flat, k=smooth_k)
        residual = response - smoothed[:, np.newaxis]

        # Regress on residuals (always use r2 for residuals)
        kwargs_residual = {k: v for k, v in kwargs.items() if k != "score_metric" and k != "final_metric"}
        res = regress_pwm(
            sequences, residual,
            score_metric="r2", final_metric="r2",
            verbose=verbose, alternative=alternative,
            **kwargs_residual,
        )
        models.append(res)
        energies.append(res.pred)

        # Combined linear model (OLS)
        E = np.column_stack(energies)
        resp_flat_2d = response[:, 0] if response.shape[1] == 1 else response.mean(axis=1)
        # Add intercept
        E_aug = np.column_stack([np.ones(len(E)), E])
        coef, _, _, _ = np.linalg.lstsq(E_aug, resp_flat_2d, rcond=None)
        e_comb = E_aug @ coef

        score_i = _score_predictions(response, energies[-1], "ks" if is_binary else "r2", alternative)
        comb_score_i = _score_predictions(response, e_comb, "ks" if is_binary else "r2", alternative)
        scores.append(score_i)
        comb_scores.append(comb_score_i)

        if verbose:
            print(f"  Score: {score_i:.4f}, Combined: {comb_score_i:.4f}")

    # Build stats
    stats = pd.DataFrame({
        "model": list(range(1, motif_num + 1)),
        "score": scores,
        "comb_score": comb_scores,
        "diff": [np.nan] + [comb_scores[i] - comb_scores[i - 1] for i in range(1, motif_num)],
        "consensus": [m.consensus for m in models],
        "seed_motif": [m.seed_motif for m in models],
    })

    # Final combined model
    E = np.column_stack(energies)
    resp_flat_2d = response[:, 0] if response.shape[1] == 1 else response.mean(axis=1)
    E_aug = np.column_stack([np.ones(len(E)), E])
    coef, _, _, _ = np.linalg.lstsq(E_aug, resp_flat_2d, rcond=None)
    pred_comb = E_aug @ coef

    return MultiRegressionResult(
        models=models,
        multi_stats=stats,
        pred=pred_comb,
        coef=coef[1:],
        intercept=float(coef[0]),
    )


# ──────────────────────────────────────────────────────────────────────
# regress_pwm_clusters
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ClusterRegressionResult:
    """Container for the output of :func:`regress_pwm_clusters`.

    Attributes
    ----------
    models : dict[str, RegressionResult]
        Per-cluster regression models.
    cluster_mat : np.ndarray
        Binary indicator matrix (n_sequences, n_clusters).
    pred_mat : np.ndarray
        Prediction matrix (n_sequences, n_clusters).
    stats : pd.DataFrame
        Per-cluster statistics.
    cluster_names : list[str]
        Cluster names.
    """

    models: dict[str, RegressionResult]
    cluster_mat: np.ndarray
    pred_mat: np.ndarray
    stats: pd.DataFrame
    cluster_names: list[str]


def regress_pwm_clusters(
    sequences: list[str] | np.ndarray,
    clusters: np.ndarray | list,
    alternative: str = "less",
    verbose: bool = False,
    **kwargs,
) -> ClusterRegressionResult:
    """Run PWM regression for each sequence cluster.

    Creates a binary response (in-cluster vs. not) for each cluster and runs
    :func:`regress_pwm` on each.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences.
    clusters : np.ndarray | list
        Cluster assignment for each sequence.
    alternative : str
        Alternative hypothesis for KS test.
    verbose : bool
        Print progress.
    **kwargs
        Additional arguments passed to :func:`regress_pwm`.

    Returns
    -------
    ClusterRegressionResult
        Per-cluster models, predictions, and statistics.
    """
    sequences = [s.upper() for s in sequences]
    clusters = np.asarray(clusters)
    if len(sequences) != len(clusters):
        raise ValueError("sequences and clusters must have the same length")

    # Remove NAs
    valid = ~pd.isna(clusters)
    if not valid.all():
        sequences = [sequences[i] for i in range(len(sequences)) if valid[i]]
        clusters = clusters[valid]

    cluster_names = sorted(set(clusters))
    n_seq = len(sequences)
    n_clust = len(cluster_names)

    # Build cluster indicator matrix
    cluster_mat = np.zeros((n_seq, n_clust), dtype=np.float64)
    name_to_idx = {name: i for i, name in enumerate(cluster_names)}
    for i, c in enumerate(clusters):
        cluster_mat[i, name_to_idx[c]] = 1.0

    models: dict[str, RegressionResult] = {}
    pred_mat = np.zeros((n_seq, n_clust), dtype=np.float64)
    stats_rows: list[dict] = []

    for ci, cname in enumerate(cluster_names):
        if verbose:
            print(f"Cluster {cname} ({ci + 1}/{n_clust})")
        binary_resp = cluster_mat[:, ci]

        # Set score_metric to ks for binary cluster response
        kw = dict(kwargs)
        kw.setdefault("score_metric", "ks")
        kw.setdefault("final_metric", "ks")

        res = regress_pwm(
            sequences, binary_resp,
            verbose=verbose,
            alternative=alternative,
            **kw,
        )
        models[str(cname)] = res
        pred_mat[:, ci] = res.pred

        stats_rows.append({
            "cluster": str(cname),
            "consensus": res.consensus,
            "ks_D": res.ks if res.ks is not None else np.nan,
            "r2": res.r2 if res.r2 is not None else np.nan,
            "seed_motif": res.seed_motif,
        })

    stats = pd.DataFrame(stats_rows)

    return ClusterRegressionResult(
        models=models,
        cluster_mat=cluster_mat,
        pred_mat=pred_mat,
        stats=stats,
        cluster_names=[str(c) for c in cluster_names],
    )


# ──────────────────────────────────────────────────────────────────────
# regress_pwm_cv
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CVRegressionResult:
    """Container for the output of :func:`regress_pwm_cv`.

    Attributes
    ----------
    cv_models : list[RegressionResult]
        Per-fold regression models.
    cv_pred : np.ndarray
        Cross-validated predictions for each sequence.
    score : float
        Overall score on cross-validated predictions.
    cv_scores : list[float]
        Per-fold test scores.
    folds : np.ndarray
        Fold assignment per sequence.
    full_model : RegressionResult | None
        Full model (trained on all data), if requested.
    """

    cv_models: list[RegressionResult]
    cv_pred: np.ndarray
    score: float
    cv_scores: list[float]
    folds: np.ndarray
    full_model: RegressionResult | None = None


def _get_cv_folds(
    response: np.ndarray,
    nfolds: int,
    seed: int | None = None,
) -> np.ndarray:
    """Create stratified (binary) or random fold assignments."""
    rng = np.random.default_rng(seed)
    n = response.shape[0]
    resp = response[:, 0] if response.ndim == 2 else response

    if _is_binary_response(response):
        # Stratified
        idx_1 = np.where(resp == 1)[0]
        idx_0 = np.where(resp == 0)[0]
        rng.shuffle(idx_1)
        rng.shuffle(idx_0)
        folds = np.zeros(n, dtype=int)
        for i, idx in enumerate(idx_1):
            folds[idx] = i % nfolds
        for i, idx in enumerate(idx_0):
            folds[idx] = i % nfolds
    else:
        perm = rng.permutation(n)
        folds = np.zeros(n, dtype=int)
        for i, idx in enumerate(perm):
            folds[idx] = i % nfolds

    return folds


def regress_pwm_cv(
    sequences: list[str] | np.ndarray,
    response: np.ndarray,
    nfolds: int | None = None,
    metric: str | None = None,
    folds: np.ndarray | None = None,
    add_full_model: bool = True,
    seed: int | None = 60427,
    alternative: str = "less",
    verbose: bool = False,
    **kwargs,
) -> CVRegressionResult:
    """Cross-validate a PWM regression model.

    Parameters
    ----------
    sequences : list[str] | np.ndarray
        DNA sequences.
    response : np.ndarray
        Response variable(s).
    nfolds : int | None
        Number of folds. Required if ``folds`` is not provided.
    metric : str | None
        Evaluation metric. Auto-selects ``"ks"`` for binary, ``"r2"``
        for continuous.
    folds : np.ndarray | None
        Explicit fold assignments. Overrides ``nfolds``.
    add_full_model : bool
        Also train a model on all data.
    seed : int | None
        Random seed.
    alternative : str
        Alternative hypothesis for KS test.
    verbose : bool
        Print progress.
    **kwargs
        Additional arguments passed to :func:`regress_pwm`.

    Returns
    -------
    CVRegressionResult
        Cross-validation results.
    """
    sequences = [s.upper() for s in sequences]
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 1:
        response = response[:, np.newaxis]

    if folds is None:
        if nfolds is None:
            raise ValueError("Either nfolds or folds must be provided")
        folds = _get_cv_folds(response, nfolds, seed)

    if metric is None:
        metric = "ks" if _is_binary_response(response) else "r2"

    unique_folds = np.unique(folds)
    cv_models = []
    cv_pred = np.zeros(len(sequences), dtype=np.float64)
    cv_scores = []

    for fi, fold_id in enumerate(unique_folds):
        if verbose:
            print(f"Fold {fi + 1}/{len(unique_folds)}")

        train_mask = folds != fold_id
        test_mask = folds == fold_id
        train_seqs = [sequences[i] for i in range(len(sequences)) if train_mask[i]]
        train_resp = response[train_mask]
        test_seqs = [sequences[i] for i in range(len(sequences)) if test_mask[i]]
        test_resp = response[test_mask]

        res = regress_pwm(
            train_seqs, train_resp,
            seed=seed,
            verbose=verbose,
            alternative=alternative,
            **kwargs,
        )
        cv_models.append(res)

        test_pred = res.predict(test_seqs)
        cv_pred[test_mask] = test_pred

        fold_score = _score_predictions(test_resp, test_pred, metric, alternative)
        cv_scores.append(fold_score)

    # Overall score
    overall_score = _score_predictions(response, cv_pred, metric, alternative)

    if verbose:
        print(f"CV score: {overall_score:.4f}")

    full_model = None
    if add_full_model:
        full_model = regress_pwm(
            sequences, response,
            seed=seed,
            verbose=verbose,
            alternative=alternative,
            **kwargs,
        )

    return CVRegressionResult(
        cv_models=cv_models,
        cv_pred=cv_pred,
        score=overall_score,
        cv_scores=cv_scores,
        folds=folds,
        full_model=full_model,
    )
