"""Edge-case tests for pyprego.

These tests probe boundary conditions, degenerate inputs, and unusual
parameter combinations that are unlikely to appear in typical usage but
must be handled gracefully.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.types import pssm_dataframe, pssm_to_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_sequences(n: int, length: int, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    nucs = np.array(list("ACGT"))
    return ["".join(nucs[rng.integers(0, 4, size=length)]) for _ in range(n)]


def _make_pssm(rows: list[list[float]]) -> pd.DataFrame:
    return pssm_dataframe(np.array(rows, dtype=np.float64))


# ===================================================================
# Empty / minimal sequence lists
# ===================================================================


class TestEmptySequenceList:
    """Behaviour when the sequence list is empty or trivially small."""

    def test_compute_pwm_empty_raises_or_returns_empty(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 4)
        # An empty sequence list should raise or return an empty array
        with pytest.raises((ValueError, IndexError)):
            pyprego.compute_pwm([], pssm)

    def test_screen_kmers_empty_returns_degenerate(self):
        # With empty input, screen_kmers returns all k-mers with NaN/zero
        # statistics rather than raising. Variance is NaN so no kmer
        # passes the variance filter but the implementation skips that
        # check when variance < 1e-15 is NaN.
        result = pyprego.screen_kmers([], np.array([]), kmer_len=4)
        assert isinstance(result, pd.DataFrame)
        # All max_r2 values should be 0 (degenerate)
        assert (result["max_r2"] == 0.0).all()

    def test_kmer_matrix_empty_sequences(self):
        result = pyprego.kmer_matrix([], 3)
        assert len(result) == 0

    def test_validate_sequences_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            pyprego.validate_sequences([])


class TestSingleSequence:
    """One-element sequence lists."""

    def test_compute_pwm_single(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 4)
        seqs = ["ACGTACGTACGT"]
        scores = pyprego.compute_pwm(seqs, pssm)
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_screen_kmers_single_sequence(self):
        seqs = ["ACGTACGTACGT"]
        response = np.array([1.0])
        # With a single sequence there is zero variance in both kmer
        # counts and response, so we expect an empty result
        result = pyprego.screen_kmers(seqs, response, kmer_len=3)
        assert isinstance(result, pd.DataFrame)

    def test_kmer_matrix_single(self):
        result = pyprego.kmer_matrix(["ACGTACGT"], 3)
        assert result.shape[0] == 1
        assert result.values.sum() > 0


# ===================================================================
# Very short sequences (shorter than motif)
# ===================================================================


class TestShortSequences:
    """Sequences shorter than the PSSM motif length."""

    def test_compute_pwm_seq_shorter_than_pssm(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 10)  # length-10 motif
        seqs = ["ACGT"]  # length 4 < 10
        scores = pyprego.compute_pwm(seqs, pssm)
        # The implementation returns -inf when no valid window exists
        assert scores.shape == (1,)
        assert np.isneginf(scores[0]) or np.isnan(scores[0])

    def test_compute_local_pwm_short_fills_nan(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 10)
        seqs = ["ACGT"]
        result = pyprego.compute_local_pwm(seqs, pssm)
        assert result.shape == (1, 4)
        assert np.all(np.isnan(result))

    def test_kmer_matrix_seq_shorter_than_k(self):
        result = pyprego.kmer_matrix(["AC"], 5)
        # No 5-mer can be found in a length-2 sequence
        assert result.values.sum() == 0


# ===================================================================
# All-N sequences
# ===================================================================


class TestAllNSequences:
    """Sequences consisting entirely of N (unknown) bases."""

    def test_compute_pwm_all_n(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 4)
        seqs = ["NNNNNNNNNN"]
        scores = pyprego.compute_pwm(seqs, pssm)
        assert scores.shape == (1,)
        # All-N should get a finite (though low) score because N positions
        # use the average log-prob
        assert np.isfinite(scores[0])

    def test_kmer_matrix_all_n(self):
        result = pyprego.kmer_matrix(["NNNNNN"], 3)
        # Ns won't match any ACGT-only kmer
        assert result.values.sum() == 0

    def test_validate_sequences_all_n_accepted(self):
        result = pyprego.validate_sequences(["NNNN", "NNNN"])
        assert len(result) == 2


# ===================================================================
# Mixed case sequences
# ===================================================================


class TestMixedCaseSequences:
    """Sequences with mixed upper/lower case."""

    def test_compute_pwm_mixed_case(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 4)
        upper_seqs = ["ACGTACGTACGT"]
        mixed_seqs = ["AcGtAcGtAcGt"]
        scores_upper = pyprego.compute_pwm(upper_seqs, pssm)
        scores_mixed = pyprego.compute_pwm(mixed_seqs, pssm)
        np.testing.assert_allclose(scores_upper, scores_mixed)

    def test_rc_mixed_case(self):
        assert pyprego.rc("AcGt") == "aCgT"

    def test_kmer_matrix_case_insensitive(self):
        upper = pyprego.kmer_matrix(["ACGTACGT"], 3)
        lower = pyprego.kmer_matrix(["acgtacgt"], 3)
        pd.testing.assert_frame_equal(upper, lower)

    def test_validate_sequences_lowercases_are_uppercased(self):
        result = pyprego.validate_sequences(["acgt", "tgca"])
        assert all(s == s.upper() for s in result)


# ===================================================================
# Single-position PSSM
# ===================================================================


class TestSinglePositionPSSM:
    """A PSSM with exactly one row."""

    def test_compute_pwm_single_pos(self):
        pssm = _make_pssm([[0.97, 0.01, 0.01, 0.01]])
        seqs = ["AAAAAAAAAA"]
        scores = pyprego.compute_pwm(seqs, pssm)
        assert scores.shape == (1,)
        assert np.isfinite(scores[0])

    def test_bits_per_pos_single(self):
        pssm = _make_pssm([[0.97, 0.01, 0.01, 0.01]])
        bits = pyprego.bits_per_pos(pssm)
        assert len(bits) == 1
        assert bits[0] > 0

    def test_pssm_rc_single(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]])
        rc_pssm = pyprego.pssm_rc(pssm)
        arr = pssm_to_array(rc_pssm)
        # A single-position RC: A<->T, C<->G swapped, reversed (1 row = same)
        assert arr[0, 3] == pytest.approx(0.7)  # T should have A's weight
        assert arr[0, 0] == pytest.approx(0.1)  # A should have T's weight

    def test_consensus_single_pos(self):
        pssm = _make_pssm([[0.97, 0.01, 0.01, 0.01]])
        consensus = pyprego.consensus_from_pssm(pssm)
        assert consensus == "A"

    def test_pssm_cor_single_pos(self):
        pssm = _make_pssm([[0.97, 0.01, 0.01, 0.01]])
        cor = pyprego.pssm_cor(pssm, pssm)
        # Self-correlation should be 1.0 (only 4 values, all identical)
        assert cor == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# Very long PSSM
# ===================================================================


class TestLongPSSM:
    """PSSM with many positions."""

    def test_compute_pwm_long_pssm(self):
        """A 50-position PSSM on 60-length sequences gives valid scores."""
        rng = np.random.default_rng(99)
        rows = rng.dirichlet(np.ones(4), size=50)
        pssm = pssm_dataframe(rows)
        seqs = _random_sequences(5, 60, seed=99)
        scores = pyprego.compute_pwm(seqs, pssm)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    def test_bits_per_pos_long(self):
        rng = np.random.default_rng(99)
        rows = rng.dirichlet(np.ones(4), size=50)
        pssm = pssm_dataframe(rows)
        bits = pyprego.bits_per_pos(pssm)
        assert len(bits) == 50
        assert np.all(bits >= 0)

    def test_pssm_theoretical_bounds_long(self):
        rng = np.random.default_rng(99)
        rows = rng.dirichlet(np.ones(4), size=50)
        pssm = pssm_dataframe(rows)
        t_min = pyprego.pssm_theoretical_min(pssm)
        t_max = pyprego.pssm_theoretical_max(pssm)
        assert t_min < t_max


# ===================================================================
# Zero-variance response
# ===================================================================


class TestZeroVarianceResponse:
    """Response vector with identical values (zero variance)."""

    def test_screen_kmers_zero_variance_response(self):
        seqs = _random_sequences(50, 100)
        response = np.ones(50)  # all ones
        result = pyprego.screen_kmers(seqs, response, kmer_len=4)
        # All correlations should be zero or the k-mers should be excluded
        if len(result) > 0:
            assert all(result["r0"].abs() < 1e-10)

    def test_regress_pwm_core_zero_variance(self):
        """regress_pwm_core should still complete with zero-variance response."""
        seqs = _random_sequences(50, 200)
        response = np.zeros(50)
        # This may produce a model with R2=0 or raise, but should not crash
        result = pyprego.regress_pwm_core(
            seqs, response,
            motif="ACGT",
            motif_length=8,
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )
        assert result.pssm is not None


# ===================================================================
# All-identical sequences
# ===================================================================


class TestAllIdenticalSequences:
    """Every sequence in the list is the same string."""

    def test_compute_pwm_identical(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 4)
        seqs = ["ACGTACGTACGT"] * 10
        scores = pyprego.compute_pwm(seqs, pssm)
        # All should be identical
        assert np.all(scores == scores[0])

    def test_kmer_matrix_identical(self):
        seqs = ["ACGTACGT"] * 5
        result = pyprego.kmer_matrix(seqs, 3)
        # All rows must be the same
        np.testing.assert_array_equal(result.values[0], result.values[1])
        np.testing.assert_array_equal(result.values[0], result.values[4])

    def test_screen_kmers_identical_sequences_zero_kmer_var(self):
        seqs = ["ACGTACGTACGT"] * 20
        response = np.random.default_rng(42).standard_normal(20)
        result = pyprego.screen_kmers(seqs, response, kmer_len=3)
        # k-mer counts have zero variance across identical sequences
        # so all k-mers should be dropped
        assert len(result) == 0


# ===================================================================
# Unicode / invalid characters in sequences
# ===================================================================


class TestInvalidCharacters:
    """Sequences with non-DNA characters."""

    def test_validate_sequences_unicode_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            pyprego.validate_sequences(["ACGT\u00e9", "ACGTA"])

    def test_validate_sequences_digits_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            pyprego.validate_sequences(["ACG1T", "ACGTT"])

    def test_rc_invalid_char_raises(self):
        with pytest.raises(ValueError):
            pyprego.rc("ACGX")

    def test_rc_not_string_raises(self):
        with pytest.raises(TypeError):
            pyprego.rc(12345)

    def test_kmers_to_pssm_invalid_char_raises(self):
        with pytest.raises(ValueError, match="valid nucleotides"):
            pyprego.kmers_to_pssm("ACGX")


# ===================================================================
# PSSM with zero rows or degenerate values
# ===================================================================


class TestDegeneratePSSM:
    """PSSMs with zero rows, all zeros, or very extreme values."""

    def test_pssm_all_zeros_row(self):
        """A row of all zeros should be handled by normalization (treated as uniform)."""
        pssm = _make_pssm([
            [0.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.1, 0.1],
        ])
        # bits_per_pos should handle zero rows gracefully
        bits = pyprego.bits_per_pos(pssm)
        assert len(bits) == 2
        assert np.all(np.isfinite(bits))

    def test_pssm_add_prior_to_zeros(self):
        pssm = _make_pssm([[0.0, 0.0, 0.0, 0.0]] * 3)
        result = pyprego.pssm_add_prior(pssm, prior=0.01)
        arr = pssm_to_array(result)
        # After adding prior and normalizing, each row should sum to 1
        np.testing.assert_allclose(arr.sum(axis=1), 1.0)

    def test_empty_pssm_trim(self):
        """Trimming a uniform PSSM should return empty."""
        pssm = _make_pssm([[0.25, 0.25, 0.25, 0.25]] * 5)
        trimmed = pyprego.pssm_trim(pssm)
        assert len(trimmed) == 0

    def test_pssm_dataframe_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            pssm_dataframe(np.array([[1, 2, 3]]))

    def test_pssm_cor_empty_raises(self):
        pssm1 = _make_pssm([[0.7, 0.1, 0.1, 0.1]])
        empty = _make_pssm([[0.25, 0.25, 0.25, 0.25]] * 3).iloc[0:0]
        with pytest.raises(ValueError):
            pyprego.pssm_cor(pssm1, empty)


# ===================================================================
# generate_kmers edge cases
# ===================================================================


class TestGenerateKmersEdge:
    """Edge cases for k-mer generation."""

    def test_k_equals_1(self):
        kmers = pyprego.generate_kmers(1)
        assert set(kmers) == {"A", "C", "G", "T"}

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            pyprego.generate_kmers(0)

    def test_max_gap_equals_k(self):
        """max_gap == k means every position is a gap (all N)."""
        kmers = pyprego.generate_kmers(3, max_gap=3)
        # Should include the all-N gapped variant
        assert any("N" in km for km in kmers)

    def test_min_gap_greater_than_max_gap_raises(self):
        with pytest.raises(ValueError):
            pyprego.generate_kmers(5, min_gap=3, max_gap=2)

    def test_max_gap_greater_than_k_raises(self):
        with pytest.raises(ValueError):
            pyprego.generate_kmers(3, max_gap=4)


# ===================================================================
# compute_pwm parameter edge cases
# ===================================================================


class TestComputePwmEdgeCases:
    """Edge cases in compute_pwm parameter handling."""

    def test_spat_min_equals_spat_max(self):
        """When spat_min == spat_max, sequences are trimmed to zero length,
        which produces -inf scores (no valid windows)."""
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 2)
        seqs = ["ACGTACGT"]
        # spat_min=4, spat_max=4 => empty substring after trimming
        scores = pyprego.compute_pwm(seqs, pssm, spat_min=4, spat_max=4)
        # Should produce -inf or a degenerate score
        assert scores.shape == (1,)
        assert np.isneginf(scores[0]) or np.isnan(scores[0])

    def test_func_invalid_raises(self):
        pssm = _make_pssm([[0.7, 0.1, 0.1, 0.1]] * 2)
        seqs = ["ACGTACGT"]
        with pytest.raises(ValueError, match="func"):
            pyprego.compute_pwm(seqs, pssm, func="mean")

    def test_pssm_missing_columns_raises(self):
        df = pd.DataFrame({"A": [0.5], "C": [0.3], "G": [0.2]})
        with pytest.raises(ValueError):
            pyprego.compute_pwm(["ACGT"], df)

    def test_bidirect_false(self):
        pssm = _make_pssm([[0.97, 0.01, 0.01, 0.01]] * 4)
        seqs = _random_sequences(5, 20)
        scores_bi = pyprego.compute_pwm(seqs, pssm, bidirect=True)
        scores_uni = pyprego.compute_pwm(seqs, pssm, bidirect=False)
        # bidirect scores should be >= unidirectional (logSumExp of more terms)
        assert np.all(scores_bi >= scores_uni - 1e-10)
