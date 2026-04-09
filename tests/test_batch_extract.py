"""Tests for batch_extract_energies: C++ batch path vs per-motif Python compute_pwm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyprego.compute import (
    _compute_log_pssm,
    _encode_sequences,
    _prepare_pssm,
    batch_extract_energies,
    compute_pwm,
)
from pyprego.motif_db import MotifDB, create_motif_db, extract_pwm, motif_db_to_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.RandomState(42)
_NUCS = list("ACGT")


def _random_sequences(n: int, length: int, seed: int = 42) -> list[str]:
    rng = np.random.RandomState(seed)
    return ["".join(rng.choice(_NUCS, size=length)) for _ in range(n)]


def _random_sequences_with_N(n: int, length: int, seed: int = 42, n_frac: float = 0.05) -> list[str]:
    """Random sequences with some N bases."""
    rng = np.random.RandomState(seed)
    seqs = []
    for _ in range(n):
        s = list(rng.choice(_NUCS, size=length))
        n_pos = rng.choice(length, size=int(length * n_frac), replace=False)
        for p in n_pos:
            s[p] = "N"
        seqs.append("".join(s))
    return seqs


def _random_pssm_df(motif_name: str, length: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    vals = rng.dirichlet([1, 1, 1, 1], size=length)
    return pd.DataFrame(
        {
            "motif": motif_name,
            "pos": np.arange(1, length + 1),
            "A": vals[:, 0],
            "C": vals[:, 1],
            "G": vals[:, 2],
            "T": vals[:, 3],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBatchExtractEnergies:
    """Compare batch C++ extraction with per-motif Python compute_pwm."""

    def test_single_motif_uniform_spat(self):
        """Single motif, uniform spatial, bidirectional."""
        seqs = _random_sequences(50, 100, seed=1)
        pssm_df = _random_pssm_df("M1", 8, seed=10)
        prior = 0.01

        # Per-motif reference
        ref = compute_pwm(seqs, pssm_df, bidirect=True, prior=prior, func="logSumExp")

        # Batch path
        encoded = _encode_sequences([s.upper() for s in seqs])
        prob = _prepare_pssm(pssm_df, prior)
        log_pssm = _compute_log_pssm(prob)

        result = batch_extract_energies(
            encoded,
            [log_pssm],
            [np.array([1.0])],
            [100],  # bin_size = seq_len => single bin
            bidirect=True,
        )

        np.testing.assert_allclose(result[:, 0], ref, rtol=1e-10)

    def test_multiple_motifs_uniform_spat(self):
        """Multiple motifs with different lengths, uniform spatial."""
        seqs = _random_sequences(30, 80, seed=2)
        prior = 0.01

        pssm_dfs = [
            _random_pssm_df("M1", 6, seed=100),
            _random_pssm_df("M2", 10, seed=200),
            _random_pssm_df("M3", 4, seed=300),
        ]

        encoded = _encode_sequences([s.upper() for s in seqs])
        log_pssm_list = []
        for pdf in pssm_dfs:
            prob = _prepare_pssm(pdf, prior)
            log_pssm_list.append(_compute_log_pssm(prob))

        spat_list = [np.array([1.0])] * 3
        bin_sizes = [80, 80, 80]

        result = batch_extract_energies(
            encoded, log_pssm_list, spat_list, bin_sizes, bidirect=True
        )

        for i, pdf in enumerate(pssm_dfs):
            ref = compute_pwm(seqs, pdf, bidirect=True, prior=prior, func="logSumExp")
            np.testing.assert_allclose(result[:, i], ref, rtol=1e-10, err_msg=f"Motif {i}")

    def test_non_uniform_spatial(self):
        """Motif with non-trivial spatial model."""
        seqs = _random_sequences(20, 120, seed=3)
        prior = 0.01
        pssm_df = _random_pssm_df("M1", 7, seed=50)

        # Spatial model: 4 bins of size 30
        spat_factors = np.array([0.5, 1.0, 1.5, 2.0])
        bin_size = 30

        # Reference
        bins = np.arange(4) * bin_size
        spat = pd.DataFrame({"bin": bins, "spat_factor": spat_factors})
        ref = compute_pwm(seqs, pssm_df, spat=spat, bidirect=True, prior=prior)

        # Batch
        encoded = _encode_sequences([s.upper() for s in seqs])
        prob = _prepare_pssm(pssm_df, prior)
        log_pssm = _compute_log_pssm(prob)

        result = batch_extract_energies(
            encoded, [log_pssm], [spat_factors], [bin_size], bidirect=True
        )

        np.testing.assert_allclose(result[:, 0], ref, rtol=1e-10)

    def test_no_bidirect(self):
        """Forward-only scoring."""
        seqs = _random_sequences(25, 60, seed=4)
        prior = 0.01
        pssm_df = _random_pssm_df("M1", 5, seed=60)

        ref = compute_pwm(seqs, pssm_df, bidirect=False, prior=prior, func="logSumExp")

        encoded = _encode_sequences([s.upper() for s in seqs])
        prob = _prepare_pssm(pssm_df, prior)
        log_pssm = _compute_log_pssm(prob)

        result = batch_extract_energies(
            encoded, [log_pssm], [np.array([1.0])], [60], bidirect=False
        )

        np.testing.assert_allclose(result[:, 0], ref, rtol=1e-10)

    def test_sequences_with_N_bases(self):
        """Sequences containing N bases should use avg_log_prob."""
        seqs = _random_sequences_with_N(20, 80, seed=5, n_frac=0.1)
        prior = 0.01
        pssm_df = _random_pssm_df("M1", 6, seed=70)

        ref = compute_pwm(seqs, pssm_df, bidirect=True, prior=prior, func="logSumExp")

        encoded = _encode_sequences([s.upper() for s in seqs])
        prob = _prepare_pssm(pssm_df, prior)
        log_pssm = _compute_log_pssm(prob)

        result = batch_extract_energies(
            encoded, [log_pssm], [np.array([1.0])], [80], bidirect=True
        )

        np.testing.assert_allclose(result[:, 0], ref, rtol=1e-10)


class TestExtractPwmBatch:
    """Test that extract_pwm() uses the batch path and produces correct results."""

    def test_extract_pwm_matches_per_motif(self):
        """extract_pwm batch output matches per-motif compute_pwm."""
        seqs = _random_sequences(40, 100, seed=10)
        prior = 0.01

        # Build a MotifDB with 3 motifs
        dfs = [
            _random_pssm_df("M1", 8, seed=110),
            _random_pssm_df("M2", 6, seed=120),
            _random_pssm_df("M3", 10, seed=130),
        ]
        combined = pd.concat(dfs, ignore_index=True)
        mdb = create_motif_db(combined, prior=prior)

        # extract_pwm (uses batch)
        batch_result = extract_pwm(seqs, mdb, bidirect=True, prior=prior)

        # Per-motif reference
        df = motif_db_to_dataframe(mdb)
        for name in mdb.names():
            motif_pssm = df[df["motif"] == name][["pos", "A", "C", "G", "T"]].copy()
            ref = compute_pwm(seqs, motif_pssm, bidirect=True, prior=prior, func="logSumExp")
            np.testing.assert_allclose(
                batch_result[name].values, ref, rtol=1e-10, err_msg=f"Motif {name}"
            )

    def test_extract_pwm_with_spatial(self):
        """extract_pwm with non-trivial spatial model."""
        seqs = _random_sequences(20, 120, seed=11)
        prior = 0.01

        dfs = [
            _random_pssm_df("M1", 7, seed=210),
            _random_pssm_df("M2", 5, seed=220),
        ]
        combined = pd.concat(dfs, ignore_index=True)

        spat_facs = np.array([[0.8, 1.2, 0.9, 1.1], [1.0, 0.5, 1.5, 2.0]])
        mdb = create_motif_db(combined, prior=prior, spat_factors=spat_facs, spat_bin_size=30)

        batch_result = extract_pwm(seqs, mdb, bidirect=True, prior=prior)

        # Per-motif reference
        df = motif_db_to_dataframe(mdb)
        for idx, name in enumerate(mdb.names()):
            motif_pssm = df[df["motif"] == name][["pos", "A", "C", "G", "T"]].copy()
            sf = mdb.spat_factors[idx, :]
            bins = np.arange(len(sf)) * mdb.spat_bin_size
            spat = pd.DataFrame({"bin": bins, "spat_factor": sf})
            ref = compute_pwm(
                seqs, motif_pssm, spat=spat, bidirect=True, prior=prior, func="logSumExp"
            )
            np.testing.assert_allclose(
                batch_result[name].values, ref, rtol=1e-10, err_msg=f"Motif {name}"
            )

    def test_extract_pwm_with_spat_min_max(self):
        """extract_pwm with spat_min/spat_max trimming."""
        seqs = _random_sequences(15, 150, seed=12)
        prior = 0.01

        dfs = [_random_pssm_df("M1", 6, seed=310)]
        combined = pd.concat(dfs, ignore_index=True)
        mdb = create_motif_db(combined, prior=prior)

        batch_result = extract_pwm(seqs, mdb, bidirect=True, prior=prior, spat_min=20, spat_max=130)

        # Per-motif reference
        df = motif_db_to_dataframe(mdb)
        motif_pssm = df[df["motif"] == "M1"][["pos", "A", "C", "G", "T"]].copy()
        ref = compute_pwm(
            seqs, motif_pssm, bidirect=True, prior=prior, func="logSumExp",
            spat_min=20, spat_max=130,
        )
        np.testing.assert_allclose(
            batch_result["M1"].values, ref, rtol=1e-10
        )

    def test_extract_pwm_motif_subset(self):
        """extract_pwm with a subset of motifs."""
        seqs = _random_sequences(20, 80, seed=13)
        prior = 0.01

        dfs = [
            _random_pssm_df("M1", 6, seed=410),
            _random_pssm_df("M2", 8, seed=420),
            _random_pssm_df("M3", 5, seed=430),
        ]
        combined = pd.concat(dfs, ignore_index=True)
        mdb = create_motif_db(combined, prior=prior)

        batch_result = extract_pwm(seqs, mdb, motifs=["M1", "M3"], bidirect=True, prior=prior)

        assert list(batch_result.columns) == ["M1", "M3"]
        df = motif_db_to_dataframe(mdb)
        for name in ["M1", "M3"]:
            motif_pssm = df[df["motif"] == name][["pos", "A", "C", "G", "T"]].copy()
            ref = compute_pwm(seqs, motif_pssm, bidirect=True, prior=prior, func="logSumExp")
            np.testing.assert_allclose(
                batch_result[name].values, ref, rtol=1e-10, err_msg=f"Motif {name}"
            )
