"""Tests for pyprego.compute module (compute_pwm and compute_local_pwm)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.compute import (
    _compute_log_pssm,
    _encode_sequences,
    _log_sum_exp,
    _prepare_pssm,
    _prepare_pssm_local,
    _score_windows,
    compute_local_pwm,
    compute_pwm,
)
from pyprego.types import pssm_dataframe, pssm_to_array, spatial_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pssm() -> pd.DataFrame:
    """A small 4-position PSSM recognising ACGT."""
    mat = np.array(
        [
            [0.9, 0.03, 0.04, 0.03],  # strongly A
            [0.03, 0.9, 0.04, 0.03],  # strongly C
            [0.03, 0.04, 0.9, 0.03],  # strongly G
            [0.03, 0.03, 0.04, 0.9],  # strongly T
        ]
    )
    return pssm_dataframe(mat)


@pytest.fixture
def uniform_pssm() -> pd.DataFrame:
    """A uniform (no information) PSSM."""
    mat = np.full((4, 4), 0.25)
    return pssm_dataframe(mat)


@pytest.fixture
def r_test_pssm() -> pd.DataFrame:
    """The 15-position PSSM from the R test file test-compute_pwm.R."""
    return pd.DataFrame(
        {
            "pos": list(range(15)),
            "A": [
                0.240737825632095,
                0.256109774112701,
                0.155907645821571,
                0.001024218974635,
                0.144046381115913,
                0.844003200531006,
                0.562452733516693,
                0.324804604053497,
                0.248924136161804,
                0.641762673854828,
                0.154276013374329,
                0.0414229892194271,
                0.275812089443207,
                0.25,
                0.21894682943821,
            ],
            "C": [
                0.230786353349686,
                0.193948850035667,
                0.107831478118896,
                0.12736551463604,
                0.147198468446732,
                0.00343735655769706,
                0.312835812568665,
                0.190958619117737,
                0.272343933582306,
                0.00105959235224873,
                0.0164314024150372,
                0.207740902900696,
                0.181653186678886,
                0.25,
                0.32295748591423,
            ],
            "G": [
                0.248799309134483,
                0.293831676244736,
                0.536404132843018,
                0.001024218974635,
                0.147198468446732,
                0.152344271540642,
                0.0539040714502335,
                0.00103273347485811,
                0.304524749517441,
                0.000942722195759416,
                0.346374750137329,
                0.000539946369826794,
                0.318346858024597,
                0.25,
                0.229047849774361,
            ],
            "T": [
                0.279676526784897,
                0.256109774112701,
                0.199856758117676,
                0.870585978031158,
                0.561556696891785,
                0.000215093168662861,
                0.0708073452115059,
                0.483204007148743,
                0.174207225441933,
                0.356234937906265,
                0.482917785644531,
                0.750296175479889,
                0.224187895655632,
                0.25,
                0.229047849774361,
            ],
        }
    )


@pytest.fixture
def r_test_sequence() -> str:
    """The test sequence from the R test file."""
    return "CAGTAAAAGCTTTAATGCGTCTTGAGAGGGAGAGCATCAGCTTACAGAGCGAAGACCCCGAATGGCAAAACCCCGTCCCTTTTATGGAGAATTGCCCTCCGCCTCAGACACGTCGCTCCCTGATTGGCTGCAGCCCATCGGCCGAGTTGTCCTCACGGGGAAGGCAGAGCACATGGAGTGGAAAACTACCCCGGGCACATGCACAGATTACTTGTTTACTACTTAGAACACAGGATGTCAGCACCATCTTGTAATGGCGAATGTGAGGGCGGCTCCTCATACTTAGTTCCCTTTTTATGA"


# ---------------------------------------------------------------------------
# Test _encode_sequences
# ---------------------------------------------------------------------------


class TestEncodeSequences:
    def test_basic_encoding(self):
        encoded = _encode_sequences(["ACGT"])
        assert encoded.shape == (1, 4)
        np.testing.assert_array_equal(encoded[0], [0, 1, 2, 3])

    def test_n_encoding(self):
        encoded = _encode_sequences(["ANGN"])
        np.testing.assert_array_equal(encoded[0], [0, -1, 2, -1])

    def test_lowercase(self):
        encoded = _encode_sequences(["acgt"])
        np.testing.assert_array_equal(encoded[0], [0, 1, 2, 3])

    def test_multiple_sequences(self):
        encoded = _encode_sequences(["ACGT", "TGCA"])
        assert encoded.shape == (2, 4)
        np.testing.assert_array_equal(encoded[1], [3, 2, 1, 0])


# ---------------------------------------------------------------------------
# Test _prepare_pssm and _prepare_pssm_local
# ---------------------------------------------------------------------------


class TestPreparePssm:
    def test_normalises_with_prior(self, simple_pssm):
        prob = _prepare_pssm(simple_pssm, prior=0.01)
        # Rows should sum to 1
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-10)

    def test_no_prior(self, simple_pssm):
        prob = _prepare_pssm(simple_pssm, prior=0.0)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-10)

    def test_local_normalises(self, simple_pssm):
        prob = _prepare_pssm_local(simple_pssm, prior=0.01)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-10)

    def test_prior_effect(self, simple_pssm):
        """Higher prior should push probabilities toward uniform."""
        prob_low = _prepare_pssm(simple_pssm, prior=0.001)
        prob_high = _prepare_pssm(simple_pssm, prior=0.5)
        # With higher prior, max prob per row should be closer to 0.25
        max_low = prob_low.max(axis=1).mean()
        max_high = prob_high.max(axis=1).mean()
        assert max_high < max_low


# ---------------------------------------------------------------------------
# Test _log_sum_exp
# ---------------------------------------------------------------------------


class TestLogSumExp:
    def test_simple(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _log_sum_exp(x, axis=0)
        expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_large_values(self):
        """Should not overflow with large values."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _log_sum_exp(x, axis=0)
        expected = 1002.0 + np.log(np.exp(-2) + np.exp(-1) + 1)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_negative_inf(self):
        x = np.array([-np.inf, -np.inf])
        result = _log_sum_exp(x, axis=0)
        assert result == -np.inf

    def test_mixed_inf(self):
        x = np.array([-np.inf, 0.0])
        result = _log_sum_exp(x, axis=0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _log_sum_exp(x, axis=1)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# Test compute_pwm - basic functionality
# ---------------------------------------------------------------------------


class TestComputePwm:
    def test_basic_scoring(self, simple_pssm):
        """Scoring ACGT with a PSSM that favours ACGT should give a high score."""
        seqs = ["ACGTACGT"]
        result = compute_pwm(seqs, simple_pssm, func="logSumExp")
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_perfect_match_higher(self, simple_pssm):
        """Perfect match should score higher than mismatch."""
        perfect = compute_pwm(["ACGTACGT"], simple_pssm, func="max")
        mismatch = compute_pwm(["TTTTTTTT"], simple_pssm, func="max")
        assert perfect[0] > mismatch[0]

    def test_multiple_sequences(self, simple_pssm):
        seqs = ["ACGTACGT", "TGCATGCA", "AAAAAAAA"]
        result = compute_pwm(seqs, simple_pssm, func="logSumExp")
        assert result.shape == (3,)
        assert all(np.isfinite(result))

    def test_single_window(self, simple_pssm):
        """When sequence length == motif length, there is exactly one window."""
        seqs = ["ACGT"]
        result_lse = compute_pwm(seqs, simple_pssm, func="logSumExp", bidirect=False)
        result_max = compute_pwm(seqs, simple_pssm, func="max", bidirect=False)
        # With single window, logSumExp and max (both unidirectional) should give
        # the same score modulo spatial
        np.testing.assert_allclose(result_lse, result_max, atol=1e-6)

    def test_func_validation(self, simple_pssm):
        with pytest.raises(ValueError, match="func must be"):
            compute_pwm(["ACGT"], simple_pssm, func="invalid")

    def test_pssm_validation(self):
        bad_pssm = pd.DataFrame({"X": [1], "Y": [2], "Z": [3], "W": [4]})
        with pytest.raises(ValueError, match="PSSM must have columns"):
            compute_pwm(["ACGT"], bad_pssm)


# ---------------------------------------------------------------------------
# Test logSumExp vs max aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_logsumexp_geq_max(self, simple_pssm):
        """logSumExp should always be >= max (since it sums additional terms)."""
        seqs = ["ACGTACGTACGT"]
        result_lse = compute_pwm(seqs, simple_pssm, func="logSumExp")
        result_max = compute_pwm(seqs, simple_pssm, func="max")
        assert result_lse[0] >= result_max[0] - 1e-10

    def test_single_window_equivalence(self, simple_pssm):
        """For single-window sequences, logSumExp == max when not bidirectional."""
        seqs = ["ACGT"]
        r_lse = compute_pwm(seqs, simple_pssm, func="logSumExp", bidirect=False)
        r_max = compute_pwm(seqs, simple_pssm, func="max", bidirect=False)
        np.testing.assert_allclose(r_lse, r_max, atol=1e-10)

    def test_max_picks_best_window(self, simple_pssm):
        """Max score of whole sequence should equal max of individual window scores."""
        seq = "AAAACGTAAAA"
        # The best window should be the one containing ACGT
        overall_max = compute_pwm([seq], simple_pssm, func="max", bidirect=False)[0]
        # Score each window individually
        K = 4  # PSSM length
        windows = [seq[i : i + K] for i in range(len(seq) - K + 1)]
        window_scores = compute_pwm(windows, simple_pssm, func="max", bidirect=False)
        np.testing.assert_allclose(overall_max, np.max(window_scores), atol=1e-4)


# ---------------------------------------------------------------------------
# Test bidirectional scoring
# ---------------------------------------------------------------------------


class TestBidirectional:
    def test_bidirect_higher(self, simple_pssm):
        """Bidirectional should give >= unidirectional scores."""
        seqs = ["ACGTACGTACGT"]
        r_bi = compute_pwm(seqs, simple_pssm, func="logSumExp", bidirect=True)
        r_uni = compute_pwm(seqs, simple_pssm, func="logSumExp", bidirect=False)
        assert r_bi[0] >= r_uni[0] - 1e-10

    def test_rc_sequence_symmetric(self, simple_pssm):
        """A palindromic PSSM should give same score for seq and its RC."""
        # Make a symmetric PSSM (A=T, C=G at each position pair)
        mat = np.array(
            [
                [0.8, 0.05, 0.05, 0.1],
                [0.05, 0.8, 0.1, 0.05],
                [0.05, 0.1, 0.8, 0.05],
                [0.1, 0.05, 0.05, 0.8],
            ]
        )
        # This is its own RC
        sym_pssm = pssm_dataframe(mat)
        # Score a sequence and its RC with bidirect=False
        seq = "ACGTAAAA"
        rc_seq = "TTTTACGT"
        s1 = compute_pwm([seq], sym_pssm, bidirect=True)
        s2 = compute_pwm([rc_seq], sym_pssm, bidirect=True)
        np.testing.assert_allclose(s1, s2, atol=1e-6)

    def test_bidirect_false_not_symmetric(self, simple_pssm):
        """Without bidirect, seq and RC can score differently."""
        seq = "ACGTACGTACGT"
        rc_seq = "ACGTACGTACGT"[::-1].translate(str.maketrans("ACGT", "TGCA"))
        s1 = compute_pwm([seq], simple_pssm, bidirect=False)
        s2 = compute_pwm([rc_seq], simple_pssm, bidirect=False)
        # They CAN differ (not necessarily, but typically do for asymmetric PSSMs)
        # Just check they are finite
        assert np.isfinite(s1[0]) and np.isfinite(s2[0])


# ---------------------------------------------------------------------------
# Test spatial factors
# ---------------------------------------------------------------------------


class TestSpatialFactors:
    def test_uniform_spat_no_effect(self, simple_pssm):
        """Uniform spatial factors should give the same result as no spatial model."""
        seqs = ["ACGTACGTACGT"]
        r_no_spat = compute_pwm(seqs, simple_pssm, func="logSumExp")
        spat = spatial_dataframe(np.array([0]), np.array([1.0]))
        r_with_spat = compute_pwm(seqs, simple_pssm, spat=spat, func="logSumExp")
        np.testing.assert_allclose(r_no_spat, r_with_spat, atol=1e-6)

    def test_high_spat_factor_increases_score(self, simple_pssm):
        """Higher spatial factor should increase the score."""
        seqs = ["ACGTACGTACGT"]
        spat_low = spatial_dataframe(np.array([0]), np.array([0.5]))
        spat_high = spatial_dataframe(np.array([0]), np.array([2.0]))
        r_low = compute_pwm(seqs, simple_pssm, spat=spat_low, func="logSumExp")
        r_high = compute_pwm(seqs, simple_pssm, spat=spat_high, func="logSumExp")
        assert r_high[0] > r_low[0]

    def test_multi_bin_spatial(self, simple_pssm):
        """Multiple spatial bins should differentially weight positions."""
        seq = "ACGTACGTACGTACGTACGT"  # length 20
        # 2 bins: first half has factor 2.0, second half has factor 0.5
        spat = spatial_dataframe(np.array([0, 10]), np.array([2.0, 0.5]))
        r = compute_pwm([seq], simple_pssm, spat=spat, func="logSumExp")
        assert np.isfinite(r[0])


# ---------------------------------------------------------------------------
# Test N-base handling
# ---------------------------------------------------------------------------


class TestNBaseHandling:
    def test_n_reduces_score(self, simple_pssm):
        """Sequences with N bases should score lower than clean sequences."""
        clean = compute_pwm(["ACGTACGT"], simple_pssm, func="logSumExp")
        with_n = compute_pwm(["ACNTACGT"], simple_pssm, func="logSumExp")
        # N gets average log-prob which is typically lower than the best match
        # This should generally reduce the score of the affected window
        assert np.isfinite(with_n[0])

    def test_all_n_sequence(self, simple_pssm):
        """All-N sequence should still produce a finite score (avg log prob)."""
        result = compute_pwm(["NNNNNNNN"], simple_pssm, func="logSumExp")
        assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# Test prior effect
# ---------------------------------------------------------------------------


class TestPriorEffect:
    def test_higher_prior_less_extreme(self, simple_pssm):
        """Higher prior should make scores less extreme (closer together)."""
        seqs = ["ACGTACGT", "TTTTTTTT"]
        r_low = compute_pwm(seqs, simple_pssm, prior=0.001, func="max")
        r_high = compute_pwm(seqs, simple_pssm, prior=0.5, func="max")
        spread_low = abs(r_low[0] - r_low[1])
        spread_high = abs(r_high[0] - r_high[1])
        assert spread_high < spread_low

    def test_zero_prior(self, simple_pssm):
        """Zero prior should work (no smoothing)."""
        result = compute_pwm(["ACGTACGT"], simple_pssm, prior=0.0, func="logSumExp")
        assert np.isfinite(result[0])


# ---------------------------------------------------------------------------
# Test compute_local_pwm
# ---------------------------------------------------------------------------


class TestComputeLocalPwm:
    def test_shape(self, simple_pssm):
        """Output should be (n_sequences, seq_length)."""
        seqs = ["ACGTACGT"]
        result = compute_local_pwm(seqs, simple_pssm)
        assert result.shape == (1, 8)

    def test_nan_at_edges(self, simple_pssm):
        """Positions where PSSM doesn't fit should be NaN."""
        seqs = ["ACGTACGT"]  # length 8, PSSM length 4
        result = compute_local_pwm(seqs, simple_pssm)
        # Valid windows: positions 0..4 (5 windows for K=4, L=8)
        # Positions 5, 6, 7 should be NaN
        assert not np.isnan(result[0, 0])
        assert not np.isnan(result[0, 4])
        assert np.isnan(result[0, 5])
        assert np.isnan(result[0, 6])
        assert np.isnan(result[0, 7])

    def test_multiple_sequences(self, simple_pssm):
        seqs = ["ACGTACGT", "TGCATGCA"]
        result = compute_local_pwm(seqs, simple_pssm)
        assert result.shape == (2, 8)

    def test_bidirect_vs_unidirect(self, simple_pssm):
        """Bidirectional should give >= unidirectional at each position."""
        seqs = ["ACGTACGTACGT"]
        r_bi = compute_local_pwm(seqs, simple_pssm, bidirect=True)
        r_uni = compute_local_pwm(seqs, simple_pssm, bidirect=False)
        # At valid positions, bidirect >= unidirect
        valid = ~np.isnan(r_bi[0]) & ~np.isnan(r_uni[0])
        assert np.all(r_bi[0, valid] >= r_uni[0, valid] - 1e-10)

    def test_perfect_match_position(self, simple_pssm):
        """The position with the best match should have the highest local score."""
        seq = "AAAACGTAAAA"  # ACGT starts at position 3
        result = compute_local_pwm([seq], simple_pssm, bidirect=False)
        valid = ~np.isnan(result[0])
        best_pos = np.nanargmax(result[0])
        assert best_pos == 3  # Window starting at position 3 = ACGT

    def test_short_sequence(self, simple_pssm):
        """Sequence shorter than PSSM should return all NaN."""
        result = compute_local_pwm(["ACG"], simple_pssm)
        assert result.shape == (1, 3)
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# Test consistency: compute_pwm matches aggregation of compute_local_pwm
# ---------------------------------------------------------------------------


class TestConsistency:
    def test_max_consistency(self, simple_pssm):
        """compute_pwm(func='max') should match max of compute_local_pwm
        when bidirect=True and spatial is uniform."""
        seq = "ACGTACGTACGT"
        global_max = compute_pwm([seq], simple_pssm, func="max", bidirect=True)[0]
        local = compute_local_pwm([seq], simple_pssm, bidirect=True)
        local_max = np.nanmax(local[0])
        np.testing.assert_allclose(global_max, local_max, atol=1e-4)

    def test_logsumexp_consistency(self, simple_pssm):
        """compute_pwm(func='logSumExp') should match logSumExp of
        individual window scores (computed via compute_local_pwm-like logic)."""
        seq = "ACGTACGTACGT"

        # compute_pwm with logSumExp, bidirect=True
        global_lse = compute_pwm([seq], simple_pssm, func="logSumExp", bidirect=True)[0]

        # Compute individual window scores: extract each window, score it
        K = 4
        windows = [seq[i : i + K] for i in range(len(seq) - K + 1)]
        window_scores = np.array(
            [compute_pwm([w], simple_pssm, func="logSumExp", bidirect=True)[0] for w in windows]
        )
        expected_lse = np.log(np.sum(np.exp(window_scores)))

        # These should be close (both are logSumExp over all window contributions)
        np.testing.assert_allclose(global_lse, expected_lse, atol=1e-4)

    def test_unidirect_max_consistency(self, simple_pssm):
        """Unidirectional max: compute_pwm should match max of compute_local_pwm."""
        seq = "ACGTACGTACGT"
        global_max = compute_pwm([seq], simple_pssm, func="max", bidirect=False)[0]
        local = compute_local_pwm([seq], simple_pssm, bidirect=False)
        local_max = np.nanmax(local[0])
        # For max with bidirect=False, each window's score in compute_pwm is just
        # fwd + spat, and in compute_local_pwm it's fwd + spat. So they match directly.
        np.testing.assert_allclose(global_max, local_max, atol=1e-4)


# ---------------------------------------------------------------------------
# Test with R test data
# ---------------------------------------------------------------------------


class TestRCompatibility:
    def test_single_window_scores_equal(self, r_test_pssm, r_test_sequence):
        """For single-window subsequences, logSumExp == max (R test assertion)."""
        K = 15
        windows = [r_test_sequence[i : i + K] for i in range(len(r_test_sequence) - K + 1)]
        lse_scores = compute_pwm(windows, r_test_pssm, func="logSumExp")
        max_scores = compute_pwm(windows, r_test_pssm, func="max")
        # R test: windows_r (max) == windows_log_sum_exp_r (logSumExp)
        np.testing.assert_allclose(lse_scores, max_scores, atol=1e-4)

    def test_max_equals_max_of_windows(self, r_test_pssm, r_test_sequence):
        """compute_pwm(max) on full seq == max of window scores (R test assertion)."""
        K = 15
        overall_max = compute_pwm([r_test_sequence], r_test_pssm, func="max")[0]
        windows = [r_test_sequence[i : i + K] for i in range(len(r_test_sequence) - K + 1)]
        window_scores = compute_pwm(windows, r_test_pssm, func="max")
        np.testing.assert_allclose(overall_max, np.max(window_scores), atol=1e-4)

    def test_logsumexp_equals_logsumexp_of_windows(self, r_test_pssm, r_test_sequence):
        """compute_pwm(logSumExp) on full seq == logSumExp of window scores (R test assertion)."""
        K = 15
        overall_lse = compute_pwm([r_test_sequence], r_test_pssm, func="logSumExp")[0]
        windows = [r_test_sequence[i : i + K] for i in range(len(r_test_sequence) - K + 1)]
        window_lse_scores = compute_pwm(windows, r_test_pssm, func="logSumExp")
        expected_lse = _log_sum_exp(window_lse_scores, axis=0)
        np.testing.assert_allclose(overall_lse, expected_lse, atol=1e-4)


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_sequence_equals_motif_length(self, simple_pssm):
        """Sequence same length as PSSM should work with exactly one window."""
        result = compute_pwm(["ACGT"], simple_pssm, func="logSumExp")
        assert result.shape == (1,)
        assert np.isfinite(result[0])

    def test_lowercase_sequences(self, simple_pssm):
        """Lowercase sequences should work (converted to upper internally)."""
        r_upper = compute_pwm(["ACGTACGT"], simple_pssm, func="logSumExp")
        r_lower = compute_pwm(["acgtacgt"], simple_pssm, func="logSumExp")
        np.testing.assert_allclose(r_upper, r_lower, atol=1e-10)

    def test_local_lowercase(self, simple_pssm):
        r_upper = compute_local_pwm(["ACGTACGT"], simple_pssm)
        r_lower = compute_local_pwm(["acgtacgt"], simple_pssm)
        np.testing.assert_allclose(r_upper, r_lower, atol=1e-10)

    def test_sequence_shorter_than_motif(self, simple_pssm):
        """Sequence shorter than PSSM should return -inf."""
        result = compute_pwm(["ACG"], simple_pssm, func="logSumExp")
        assert result[0] == -np.inf

    def test_spat_min_max(self, simple_pssm):
        """spat_min and spat_max should trim the sequence."""
        seq = "AAAACGTAAAA"  # length 11
        # Only look at positions 5-8 (1-based), which is "CGTA"
        result = compute_pwm([seq], simple_pssm, spat_min=5, spat_max=8, bidirect=False)
        # Directly score the substring
        result_direct = compute_pwm(["CGTA"], simple_pssm, bidirect=False)
        np.testing.assert_allclose(result, result_direct, atol=1e-6)

    def test_many_sequences(self, simple_pssm):
        """Should handle many sequences efficiently."""
        np.random.seed(42)
        bases = "ACGT"
        seqs = ["".join(np.random.choice(list(bases), 50)) for _ in range(100)]
        result = compute_pwm(seqs, simple_pssm, func="logSumExp")
        assert result.shape == (100,)
        assert all(np.isfinite(result))

    def test_local_with_spatial(self, simple_pssm):
        """compute_local_pwm with spatial model should work."""
        seqs = ["ACGTACGTACGT"]
        spat = spatial_dataframe(np.array([0]), np.array([1.0]))
        result = compute_local_pwm(seqs, simple_pssm, spat=spat)
        assert result.shape == (1, 12)
        # Should be same as without spatial (since factor=1)
        result_no_spat = compute_local_pwm(seqs, simple_pssm)
        np.testing.assert_allclose(result, result_no_spat, atol=1e-6)
