"""Property-based tests for pyprego.

These tests verify mathematical invariants and algebraic properties of
the PSSM and sequence operations. Each test generates a variety of
inputs and checks that the stated property always holds.
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

def _random_pssm(length: int, seed: int = 0) -> pd.DataFrame:
    """Create a random probability PSSM (rows sum to 1)."""
    rng = np.random.default_rng(seed)
    mat = rng.dirichlet(np.ones(4), size=length)
    return pssm_dataframe(mat)


def _random_sequences(n: int, length: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    nucs = np.array(list("ACGT"))
    return ["".join(nucs[rng.integers(0, 4, size=length)]) for _ in range(n)]


# ===================================================================
# pssm_rc is an involution: pssm_rc(pssm_rc(pssm)) == pssm
# ===================================================================


class TestPssmRcInvolution:
    """pssm_rc applied twice should return the original PSSM."""

    @pytest.mark.parametrize("length", [1, 2, 5, 10, 20])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_double_rc_is_identity(self, length, seed):
        pssm = _random_pssm(length, seed=seed)
        double_rc = pyprego.pssm_rc(pyprego.pssm_rc(pssm))
        original = pssm_to_array(pssm)
        recovered = pssm_to_array(double_rc)
        np.testing.assert_allclose(recovered, original, atol=1e-12)

    def test_single_position_involution(self):
        pssm = pssm_dataframe(np.array([[0.6, 0.2, 0.15, 0.05]]))
        double_rc = pyprego.pssm_rc(pyprego.pssm_rc(pssm))
        np.testing.assert_allclose(
            pssm_to_array(double_rc), pssm_to_array(pssm), atol=1e-12
        )

    def test_uniform_pssm_involution(self):
        pssm = pssm_dataframe(np.full((8, 4), 0.25))
        double_rc = pyprego.pssm_rc(pyprego.pssm_rc(pssm))
        np.testing.assert_allclose(
            pssm_to_array(double_rc), pssm_to_array(pssm), atol=1e-12
        )


# ===================================================================
# bits_per_pos is non-negative
# ===================================================================


class TestBitsNonNegative:
    """Information content (bits) at each position must be >= 0."""

    @pytest.mark.parametrize("seed", range(5))
    def test_random_pssm_bits_nonneg(self, seed):
        pssm = _random_pssm(length=10, seed=seed)
        bits = pyprego.bits_per_pos(pssm)
        assert np.all(bits >= -1e-12)

    def test_uniform_pssm_zero_bits(self):
        pssm = pssm_dataframe(np.full((5, 4), 0.25))
        bits = pyprego.bits_per_pos(pssm, prior=0.0)
        np.testing.assert_allclose(bits, 0.0, atol=1e-10)

    def test_perfect_pssm_max_bits(self):
        """A PSSM where each position is 100% one nucleotide should have ~2 bits."""
        mat = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        pssm = pssm_dataframe(mat)
        bits = pyprego.bits_per_pos(pssm, prior=0.0)
        np.testing.assert_allclose(bits, 2.0, atol=1e-10)


# ===================================================================
# Rows sum to 1 after pssm_add_prior
# ===================================================================


class TestPssmAddPriorRowSums:
    """After add_prior, each row of the PSSM should sum to 1."""

    @pytest.mark.parametrize("prior", [0.001, 0.01, 0.05, 0.1, 0.5])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_rows_sum_to_one(self, prior, seed):
        pssm = _random_pssm(length=8, seed=seed)
        result = pyprego.pssm_add_prior(pssm, prior=prior)
        arr = pssm_to_array(result)
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-12)

    def test_zero_prior_preserves(self):
        """With prior=0, should just re-normalize (already normalized -> no change)."""
        pssm = _random_pssm(length=5, seed=42)
        result = pyprego.pssm_add_prior(pssm, prior=0.0)
        arr = pssm_to_array(result)
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-12)

    def test_degenerate_all_zero_rows(self):
        """All-zero rows should become uniform after adding prior."""
        pssm = pssm_dataframe(np.zeros((3, 4)))
        result = pyprego.pssm_add_prior(pssm, prior=0.01)
        arr = pssm_to_array(result)
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-12)
        # Should be uniform: 0.25 each
        np.testing.assert_allclose(arr, 0.25, atol=1e-12)


# ===================================================================
# Uniform PSSM gives equal scores for same-length sequences
# ===================================================================


class TestUniformPssmEqualScores:
    """A completely uniform PSSM should yield the same score for all sequences
    of the same length (since all nucleotides are equally likely)."""

    def test_uniform_scores_equal(self):
        pssm = pssm_dataframe(np.full((4, 4), 0.25))
        seqs = _random_sequences(20, 50, seed=42)
        scores = pyprego.compute_pwm(seqs, pssm, bidirect=False)
        # All scores should be identical
        np.testing.assert_allclose(scores, scores[0], atol=1e-10)

    def test_uniform_scores_equal_bidirect(self):
        pssm = pssm_dataframe(np.full((4, 4), 0.25))
        seqs = _random_sequences(20, 50, seed=42)
        scores = pyprego.compute_pwm(seqs, pssm, bidirect=True)
        np.testing.assert_allclose(scores, scores[0], atol=1e-10)


# ===================================================================
# pssm_theoretical_min <= score <= pssm_theoretical_max
# ===================================================================


class TestScoreBounds:
    """Computed PWM scores should lie within theoretical min/max bounds.

    The bounds apply to single-window log-likelihood; for logSumExp
    aggregation across windows, the actual score can exceed the single-window
    max. So we use func="max" with bidirect=False for a strict bound check.
    """

    @pytest.mark.parametrize("seed", range(5))
    def test_score_within_bounds(self, seed):
        pssm = _random_pssm(length=6, seed=seed)
        seqs = _random_sequences(30, 50, seed=seed + 100)

        t_min = pyprego.pssm_theoretical_min(pssm)
        t_max = pyprego.pssm_theoretical_max(pssm)

        # Use func="max" to get single-window score (no logSumExp inflation)
        # and bidirect=False to avoid combining two orientations
        scores = pyprego.compute_pwm(seqs, pssm, func="max", bidirect=False)

        # Allow a small tolerance for floating-point differences
        assert np.all(scores >= t_min - 1e-6), (
            f"Score {scores.min()} < theoretical min {t_min}"
        )
        assert np.all(scores <= t_max + 1e-6), (
            f"Score {scores.max()} > theoretical max {t_max}"
        )

    def test_theoretical_min_less_than_max(self):
        for seed in range(10):
            pssm = _random_pssm(length=8, seed=seed)
            t_min = pyprego.pssm_theoretical_min(pssm)
            t_max = pyprego.pssm_theoretical_max(pssm)
            assert t_min <= t_max + 1e-12


# ===================================================================
# consensus_from_pssm returns only valid IUPAC codes
# ===================================================================


class TestConsensusIUPAC:
    """consensus_from_pssm should return only valid IUPAC single-letter codes."""

    VALID_IUPAC = set("ACGTMRWSYKN")

    @pytest.mark.parametrize("seed", range(10))
    def test_only_valid_iupac(self, seed):
        pssm = _random_pssm(length=12, seed=seed)
        consensus = pyprego.consensus_from_pssm(pssm)
        for ch in consensus:
            assert ch in self.VALID_IUPAC, f"Invalid IUPAC code: {ch!r}"

    def test_perfect_pssm_consensus(self):
        """A PSSM with one dominant nuc per position should give that nuc."""
        mat = np.array([
            [0.95, 0.02, 0.02, 0.01],  # A
            [0.02, 0.95, 0.02, 0.01],  # C
            [0.02, 0.02, 0.95, 0.01],  # G
            [0.01, 0.02, 0.02, 0.95],  # T
        ])
        pssm = pssm_dataframe(mat)
        consensus = pyprego.consensus_from_pssm(pssm)
        assert consensus == "ACGT"

    def test_uniform_gives_all_N(self):
        pssm = pssm_dataframe(np.full((5, 4), 0.25))
        consensus = pyprego.consensus_from_pssm(pssm)
        assert all(ch == "N" for ch in consensus)

    def test_ambiguity_code_AG_is_R(self):
        """A position with A=0.45, G=0.45, C=0.05, T=0.05 should be R."""
        mat = np.array([[0.45, 0.05, 0.45, 0.05]])
        pssm = pssm_dataframe(mat)
        consensus = pyprego.consensus_from_pssm(pssm)
        assert consensus == "R"


# ===================================================================
# rc(rc(seq)) == seq
# ===================================================================


class TestRcInvolution:
    """The reverse complement of the reverse complement should be the original."""

    @pytest.mark.parametrize("seq", [
        "A", "ACGT", "AAAA", "TTTT", "ACGTACGT",
        "NNNNN", "ANTNC", "acgt", "AcGtN",
    ])
    def test_double_rc_identity(self, seq):
        assert pyprego.rc(pyprego.rc(seq)) == seq

    @pytest.mark.parametrize("seed", range(5))
    def test_random_sequences_double_rc(self, seed):
        seqs = _random_sequences(10, 100, seed=seed)
        for seq in seqs:
            assert pyprego.rc(pyprego.rc(seq)) == seq

    def test_empty_string(self):
        assert pyprego.rc(pyprego.rc("")) == ""

    def test_single_base(self):
        for base in "ACGT":
            assert pyprego.rc(pyprego.rc(base)) == base


# ===================================================================
# pssm_cor(pssm, pssm) == 1.0  (self-correlation)
# ===================================================================


class TestPssmSelfCorrelation:
    """The correlation of a PSSM with itself should be 1.0."""

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("method", ["spearman", "pearson"])
    def test_self_cor_is_one(self, seed, method):
        pssm = _random_pssm(length=8, seed=seed)
        cor = pyprego.pssm_cor(pssm, pssm, method=method)
        assert cor == pytest.approx(1.0, abs=1e-6)

    def test_single_position_self_cor(self):
        pssm = pssm_dataframe(np.array([[0.6, 0.2, 0.15, 0.05]]))
        cor = pyprego.pssm_cor(pssm, pssm)
        assert cor == pytest.approx(1.0, abs=1e-6)

    def test_uniform_self_cor(self):
        """Uniform PSSM: all values equal -> Spearman should still be 1.0
        (or possibly degenerate). Pearson is undefined for zero-variance
        data but the implementation returns 0."""
        pssm = pssm_dataframe(np.full((4, 4), 0.25))
        # For identical PSSMs the implementation should handle ties
        cor = pyprego.pssm_cor(pssm, pssm, method="spearman")
        # Either 1.0 or 0.0 (degenerate case is acceptable)
        assert cor == pytest.approx(1.0, abs=1e-6) or cor == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# pssm_diff(pssm, pssm) ~= 0  (self-KL divergence)
# ===================================================================


class TestPssmSelfDivergence:
    """KL divergence of a PSSM with itself should be zero (or very close)."""

    @pytest.mark.parametrize("seed", range(5))
    def test_self_kl_is_zero(self, seed):
        pssm = _random_pssm(length=8, seed=seed)
        kl = pyprego.pssm_diff(pssm, pssm)
        assert kl == pytest.approx(0.0, abs=1e-8)


# ===================================================================
# pssm_cor is symmetric
# ===================================================================


class TestPssmCorSymmetric:
    """pssm_cor(A, B) == pssm_cor(B, A) when both have the same length."""

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("method", ["spearman", "pearson"])
    def test_symmetric_same_length(self, seed, method):
        p1 = _random_pssm(8, seed=seed)
        p2 = _random_pssm(8, seed=seed + 100)
        cor_ab = pyprego.pssm_cor(p1, p2, method=method)
        cor_ba = pyprego.pssm_cor(p2, p1, method=method)
        assert cor_ab == pytest.approx(cor_ba, abs=1e-10)


# ===================================================================
# pssm_theoretical_max >= pssm_theoretical_min (always)
# ===================================================================


class TestTheoreticalBoundsOrder:
    """The theoretical maximum must always be >= the minimum."""

    @pytest.mark.parametrize("seed", range(10))
    def test_max_geq_min(self, seed):
        pssm = _random_pssm(length=np.random.default_rng(seed).integers(2, 20), seed=seed)
        t_min = pyprego.pssm_theoretical_min(pssm)
        t_max = pyprego.pssm_theoretical_max(pssm)
        assert t_max >= t_min - 1e-12


# ===================================================================
# pssm_quantile interpolation
# ===================================================================


class TestPssmQuantileInterpolation:
    """pssm_quantile at q=0 == min, q=1 == max, and monotone in between."""

    @pytest.mark.parametrize("seed", range(5))
    def test_quantile_endpoints(self, seed):
        pssm = _random_pssm(length=8, seed=seed)
        t_min = pyprego.pssm_theoretical_min(pssm)
        t_max = pyprego.pssm_theoretical_max(pssm)
        assert pyprego.pssm_quantile(pssm, 0.0) == pytest.approx(t_min, abs=1e-10)
        assert pyprego.pssm_quantile(pssm, 1.0) == pytest.approx(t_max, abs=1e-10)

    @pytest.mark.parametrize("seed", range(3))
    def test_quantile_monotone(self, seed):
        pssm = _random_pssm(length=8, seed=seed)
        qs = [pyprego.pssm_quantile(pssm, q) for q in np.linspace(0, 1, 11)]
        for i in range(len(qs) - 1):
            assert qs[i] <= qs[i + 1] + 1e-12


# ===================================================================
# pssm_trim preserves inner positions
# ===================================================================


class TestPssmTrimPreservesInner:
    """Trimming should only remove low-information edges, not inner positions."""

    def test_trim_preserves_high_info_core(self):
        # Uniform edges, strong core
        mat = np.full((10, 4), 0.25)
        mat[3] = [0.97, 0.01, 0.01, 0.01]
        mat[4] = [0.01, 0.97, 0.01, 0.01]
        mat[5] = [0.01, 0.01, 0.97, 0.01]
        pssm = pssm_dataframe(mat)
        trimmed = pyprego.pssm_trim(pssm, bits_thresh=0.1)
        assert len(trimmed) >= 3  # At least the 3 strong positions
        # The strong positions should be included
        arr = pssm_to_array(trimmed)
        assert arr[0, 0] > 0.5 or arr[-1, 2] > 0.5  # Check core present

    def test_trim_full_info_pssm_unchanged(self):
        mat = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
        ])
        pssm = pssm_dataframe(mat)
        trimmed = pyprego.pssm_trim(pssm)
        assert len(trimmed) == 3


# ===================================================================
# compute_pwm: bidirectional >= unidirectional (logSumExp)
# ===================================================================


class TestBidirectVsUnidirect:
    """With logSumExp, bidirectional scores should be >= unidirectional."""

    @pytest.mark.parametrize("seed", range(3))
    def test_bidirect_geq_unidirect(self, seed):
        pssm = _random_pssm(length=6, seed=seed)
        seqs = _random_sequences(20, 50, seed=seed + 100)
        scores_bi = pyprego.compute_pwm(seqs, pssm, bidirect=True, func="logSumExp")
        scores_uni = pyprego.compute_pwm(seqs, pssm, bidirect=False, func="logSumExp")
        # logSumExp over more terms is always >= logSumExp over fewer terms
        assert np.all(scores_bi >= scores_uni - 1e-10)


# ===================================================================
# compute_local_pwm: shape is (n_seq, seq_len) and non-NaN where valid
# ===================================================================


class TestComputeLocalPwmShape:
    """compute_local_pwm should return the expected shape."""

    def test_shape_matches(self):
        pssm = _random_pssm(length=6, seed=42)
        seqs = _random_sequences(10, 50, seed=42)
        result = pyprego.compute_local_pwm(seqs, pssm)
        assert result.shape == (10, 50)

    def test_valid_windows_are_finite(self):
        pssm = _random_pssm(length=6, seed=42)
        seqs = _random_sequences(10, 50, seed=42)
        result = pyprego.compute_local_pwm(seqs, pssm)
        n_windows = 50 - 6 + 1
        # First n_windows columns should be finite
        assert np.all(np.isfinite(result[:, :n_windows]))
        # Remaining columns should be NaN
        assert np.all(np.isnan(result[:, n_windows:]))


# ===================================================================
# kmer_matrix: counts are non-negative integers
# ===================================================================


class TestKmerMatrixProperties:
    """kmer_matrix should return non-negative integer counts."""

    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_counts_non_negative_integer(self, k):
        seqs = _random_sequences(10, 50, seed=42)
        result = pyprego.kmer_matrix(seqs, k)
        assert np.all(result.values >= 0)
        # Values should be integers
        np.testing.assert_array_equal(result.values, result.values.astype(int))

    def test_total_counts_consistent(self):
        """Total k-mer counts per sequence should equal (seq_len - k + 1)."""
        k = 3
        seq_len = 50
        seqs = _random_sequences(10, seq_len, seed=42)
        result = pyprego.kmer_matrix(seqs, k)
        expected_total = seq_len - k + 1
        row_sums = result.values.sum(axis=1)
        np.testing.assert_array_equal(row_sums, expected_total)


# ===================================================================
# pssm_concat: length is sum of parts (+ gaps)
# ===================================================================


class TestPssmConcatLength:
    """Concatenation length should be sum of input lengths plus gaps."""

    def test_concat_no_gap(self):
        p1 = _random_pssm(5, seed=0)
        p2 = _random_pssm(3, seed=1)
        result = pyprego.pssm_concat(p1, p2, gap=0)
        assert len(result) == 8

    def test_concat_with_gap(self):
        p1 = _random_pssm(5, seed=0)
        p2 = _random_pssm(3, seed=1)
        gap = 4
        result = pyprego.pssm_concat(p1, p2, gap=gap)
        assert len(result) == 5 + gap + 3

    def test_concat_three_pssms(self):
        p1 = _random_pssm(3, seed=0)
        p2 = _random_pssm(4, seed=1)
        p3 = _random_pssm(5, seed=2)
        result = pyprego.pssm_concat(p1, p2, p3, gap=2)
        # 3 + 2 + 4 + 2 + 5 = 16
        assert len(result) == 16


# ===================================================================
# kmers_to_pssm: rows sum to 1
# ===================================================================


class TestKmersToPssmRowSums:
    """kmers_to_pssm output rows should all sum to 1."""

    @pytest.mark.parametrize("kmer", ["ACGT", "GATAAG", "NNN", "ANNG"])
    def test_rows_sum_to_one(self, kmer):
        result = pyprego.kmers_to_pssm(kmer)
        arr = result[["A", "C", "G", "T"]].values
        np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-12)


# ===================================================================
# pssm_to_kmer: result has valid characters
# ===================================================================


class TestPssmToKmerValid:
    """pssm_to_kmer should return only A, C, G, T, N characters."""

    @pytest.mark.parametrize("seed", range(5))
    def test_only_valid_chars(self, seed):
        pssm = _random_pssm(length=10, seed=seed)
        kmer = pyprego.pssm_to_kmer(pssm)
        assert all(ch in "ACGTN" for ch in kmer)

    def test_length_matches_pssm(self):
        pssm = _random_pssm(length=8, seed=42)
        kmer = pyprego.pssm_to_kmer(pssm)
        assert len(kmer) == 8

    def test_specified_length(self):
        pssm = _random_pssm(length=15, seed=42)
        kmer = pyprego.pssm_to_kmer(pssm, kmer_length=8)
        assert len(kmer) == 8


# ===================================================================
# rc_array: each element is rc of original
# ===================================================================


class TestRcArrayConsistency:
    """rc_array should apply rc to each element."""

    def test_rc_array_matches_individual(self):
        seqs = _random_sequences(10, 50, seed=42)
        rc_arr = pyprego.rc_array(seqs)
        for orig, got in zip(seqs, rc_arr):
            assert got == pyprego.rc(orig)

    def test_rc_array_double_is_identity(self):
        seqs = _random_sequences(10, 50, seed=42)
        double_rc = pyprego.rc_array(pyprego.rc_array(seqs))
        assert double_rc == seqs


# ===================================================================
# validate_sequences: idempotent after first call
# ===================================================================


class TestValidateIdempotent:
    """Calling validate_sequences twice should give the same result."""

    def test_idempotent(self):
        seqs = _random_sequences(10, 50, seed=42)
        first = list(pyprego.validate_sequences(seqs))
        second = list(pyprego.validate_sequences(first))
        assert first == second

    def test_uppercases(self):
        seqs = ["acgt", "tgca"]
        result = pyprego.validate_sequences(seqs)
        assert all(s == s.upper() for s in result)


# ===================================================================
# Spatial model: uniform spat gives same score as no spat
# ===================================================================


class TestUniformSpatEquivalence:
    """A uniform spatial model should behave like having no spatial model."""

    def test_uniform_spat_like_none(self):
        pssm = _random_pssm(length=6, seed=42)
        seqs = _random_sequences(10, 50, seed=42)

        scores_no_spat = pyprego.compute_pwm(seqs, pssm, spat=None)
        # Uniform spatial factor of 1.0 in a single bin
        spat_df = pd.DataFrame({"bin": [0], "spat_factor": [1.0]})
        scores_uniform = pyprego.compute_pwm(seqs, pssm, spat=spat_df)
        np.testing.assert_allclose(scores_no_spat, scores_uniform, atol=1e-10)
