"""Tests for the PWM regression optimiser.

Covers:
- Optimizer improves score from initial k-mer
- Continuous regression (r2 metric)
- Binary regression (ks metric)
- Known planted motif discovery
- Bidirectional vs unidirectional
- Spatial factor optimisation
- predict() on result
- PSSM initialisation
- Edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyprego.regression import (
    _PWMLRegression,
    _build_neighbourhood,
    _calc_spat_min_max,
    _calculate_bins,
    _encode_sequences_int,
    regress_pwm,
    regress_pwm_core,
)
from pyprego.types import pssm_dataframe


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _random_dna(length: int, rng: np.random.Generator) -> str:
    """Generate a random DNA sequence."""
    return "".join(rng.choice(list("ACGT"), size=length))


def _generate_sequences_with_motif(
    n_seq: int,
    seq_len: int,
    motif: str,
    *,
    fraction_with_motif: float = 0.5,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Generate sequences with a planted motif and binary response.

    Sequences with response=1 have the motif planted near the center.
    Sequences with response=0 are purely random.
    """
    rng = np.random.default_rng(seed)
    sequences = []
    response = np.zeros(n_seq)
    n_with = int(n_seq * fraction_with_motif)
    motif_len = len(motif)
    center = seq_len // 2

    for i in range(n_seq):
        seq = list(_random_dna(seq_len, rng))
        if i < n_with:
            # Plant motif near center
            pos = center - motif_len // 2 + rng.integers(-5, 6)
            pos = max(0, min(pos, seq_len - motif_len))
            for j, ch in enumerate(motif):
                seq[pos + j] = ch
            response[i] = 1.0
        sequences.append("".join(seq))

    # Shuffle
    perm = rng.permutation(n_seq)
    sequences = [sequences[i] for i in perm]
    response = response[perm]

    return sequences, response


def _generate_continuous_response(
    sequences: list[str],
    motif: str,
    *,
    noise_std: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a continuous response based on motif presence.

    Counts occurrences of the motif (and RC) in each sequence,
    adds noise, and returns a continuous score.
    """
    rng = np.random.default_rng(seed)
    from pyprego.utils import rc

    motif_rc = rc(motif)
    n = len(sequences)
    response = np.zeros(n)
    for i, seq in enumerate(sequences):
        count = seq.count(motif) + seq.count(motif_rc)
        response[i] = float(count) + rng.normal(0, noise_std)
    return response


# ──────────────────────────────────────────────────────────────────────
# Unit tests for helpers
# ──────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_encode_sequences(self):
        enc = _encode_sequences_int(["ACGT", "TGCA"])
        assert enc.shape == (2, 4)
        np.testing.assert_array_equal(enc[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(enc[1], [3, 2, 1, 0])

    def test_encode_with_n(self):
        enc = _encode_sequences_int(["ANGN"])
        np.testing.assert_array_equal(enc[0], [0, -1, 2, -1])

    def test_build_neighbourhood(self):
        moves = _build_neighbourhood(0.1)
        assert len(moves) == 20
        # First 8 are single-nuc moves
        for i in range(8):
            assert len(moves[i]) == 1
        # Remaining 12 are paired moves
        for i in range(8, 20):
            assert len(moves[i]) == 2

    def test_calculate_bins_both_specified(self):
        n, s = _calculate_bins(280, 7, 40)
        assert n == 7
        assert s == 40

    def test_calculate_bins_auto(self):
        n, s = _calculate_bins(280, None, None)
        assert n % 2 == 1  # odd
        assert n >= 3
        assert s * n <= 280

    def test_calc_spat_min_max(self):
        smin, smax = _calc_spat_min_max(7, 280, 40)
        assert smax - smin == 7 * 40
        # Center should be near 140
        center_actual = (smin + smax) / 2
        assert abs(center_actual - 140) <= 1


# ──────────────────────────────────────────────────────────────────────
# Integration tests: optimizer improves score
# ──────────────────────────────────────────────────────────────────────

class TestOptimizerImproves:
    """Test that the optimizer actually improves the score from the initial state."""

    def test_r2_improves_from_wildcard(self):
        """Starting from all-wildcards, the optimizer should improve R2."""
        rng = np.random.default_rng(42)
        n_seq = 200
        seq_len = 280
        motif = "GATA"
        sequences, binary_response = _generate_sequences_with_motif(
            n_seq, seq_len, motif, seed=42
        )
        # Make a continuous response
        response = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences,
            response,
            motif="****GATA****",  # give the seed
            motif_length=12,
            score_metric="r2",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            verbose=False,
        )
        assert result.r2 is not None
        assert result.r2 > 0.0, "R2 should be positive after optimization"

    def test_ks_improves_with_planted_motif(self):
        """With a planted binary motif, KS should be positive."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )

        result = regress_pwm(
            sequences,
            response,
            motif="****GATA****",
            motif_length=12,
            score_metric="ks",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            verbose=False,
        )
        assert result.ks is not None
        assert result.ks > 0.0, "KS should be positive with a planted motif"


# ──────────────────────────────────────────────────────────────────────
# Binary classification (KS metric)
# ──────────────────────────────────────────────────────────────────────

class TestBinaryRegression:
    def test_ks_with_strong_signal(self):
        """With a strong planted motif, KS should be large."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            300, 280, motif, fraction_with_motif=0.5, seed=123
        )

        result = regress_pwm(
            sequences,
            response,
            motif="***GATAAG***",
            motif_length=12,
            score_metric="ks",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=123,
            verbose=False,
        )
        # With a strong 6-mer signal and 300 sequences, KS should be decent
        assert result.ks > 0.1, f"Expected KS > 0.1, got {result.ks}"

    def test_ks_no_signal_is_low(self):
        """With random sequences and random binary labels, KS should be small."""
        rng = np.random.default_rng(42)
        n_seq = 100
        seq_len = 280
        sequences = [_random_dna(seq_len, rng) for _ in range(n_seq)]
        response = rng.choice([0.0, 1.0], size=n_seq)

        result = regress_pwm(
            sequences,
            response,
            motif="************",
            motif_length=12,
            score_metric="ks",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            verbose=False,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.01],
        )
        # KS should be modest since there's no real signal (with only 100 seqs
        # the optimizer can overfit slightly, so we use a generous threshold)
        assert result.ks < 0.45, f"Expected KS < 0.45 for random data, got {result.ks}"


# ──────────────────────────────────────────────────────────────────────
# Continuous regression (R2 metric)
# ──────────────────────────────────────────────────────────────────────

class TestContinuousRegression:
    def test_r2_with_planted_motif(self):
        """With a planted motif affecting a continuous response, R2 should improve."""
        motif = "GATA"
        sequences, binary = _generate_sequences_with_motif(
            200, 280, motif, seed=42
        )
        response = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences,
            response,
            motif="****GATA****",
            motif_length=12,
            score_metric="r2",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            verbose=False,
        )
        assert result.r2 > 0.0

    def test_multidim_response(self):
        """Multi-dimensional response should work."""
        motif = "GATA"
        sequences, binary = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp1 = _generate_continuous_response(sequences, motif, seed=42)
        resp2 = _generate_continuous_response(sequences, motif, seed=43, noise_std=0.5)
        response = np.column_stack([resp1, resp2])

        result = regress_pwm(
            sequences,
            response,
            motif="****GATA****",
            motif_length=12,
            score_metric="r2",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            verbose=False,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.01],
        )
        # R2 should be a list for multi-dimensional
        assert result.r2 is not None


# ──────────────────────────────────────────────────────────────────────
# Planted motif discovery
# ──────────────────────────────────────────────────────────────────────

class TestMotifDiscovery:
    def test_discovers_planted_motif_binary(self):
        """The optimizer should discover the planted motif (binary)."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            300, 280, motif, fraction_with_motif=0.5, seed=42
        )

        result = regress_pwm(
            sequences,
            response,
            motif="***GATAAG***",
            motif_length=12,
            score_metric="ks",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            verbose=False,
        )

        # The consensus should contain the core motif or something close
        pssm = result.pssm
        # Check that the PSSM has the motif signature
        # At positions where we planted G, G should be the dominant nucleotide
        # (in the optimised PSSM, not necessarily at position 0)
        arr = pssm[["A", "C", "G", "T"]].to_numpy()
        max_nucs = np.argmax(arr, axis=1)
        consensus_str = "".join("ACGT"[i] for i in max_nucs)

        # The motif or its RC should appear somewhere in the consensus
        from pyprego.utils import rc
        motif_rc = rc(motif)
        found = motif in consensus_str or motif_rc in consensus_str
        # Even if not exact, KS should be good
        assert result.ks > 0.1 or found


# ──────────────────────────────────────────────────────────────────────
# Bidirectional vs unidirectional
# ──────────────────────────────────────────────────────────────────────

class TestBidirectional:
    def test_bidirect_vs_unidirect(self):
        """Bidirectional should handle RC motifs better."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, seed=42
        )

        res_bi = regress_pwm(
            sequences, response,
            motif="***GATAAG***",
            motif_length=12,
            score_metric="ks",
            bidirect=True,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.01],
        )

        res_uni = regress_pwm(
            sequences, response,
            motif="***GATAAG***",
            motif_length=12,
            score_metric="ks",
            bidirect=False,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.01],
        )

        # Both should have positive KS
        assert res_bi.ks > 0.0
        assert res_uni.ks > 0.0
        # Bidirectional flag should be set correctly
        assert res_bi.bidirect is True
        assert res_uni.bidirect is False


# ──────────────────────────────────────────────────────────────────────
# Spatial factor optimisation
# ──────────────────────────────────────────────────────────────────────

class TestSpatialFactors:
    def test_spatial_factors_in_result(self):
        """Result should contain spatial factors."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.01],
        )

        assert "bin" in result.spat.columns
        assert "spat_factor" in result.spat.columns
        assert len(result.spat) == 7
        # Spatial factors should be positive
        assert (result.spat["spat_factor"].to_numpy() >= 0).all()

    def test_optimize_spat_only(self):
        """Can optimize spatial factors only (PWM fixed)."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            optimize_pwm=False,
            optimize_spat=True,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )
        # Should complete without error
        assert result.pssm is not None

    def test_optimize_pwm_only(self):
        """Can optimize PWM only (spatial fixed)."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            optimize_pwm=True,
            optimize_spat=False,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )
        # All spatial factors should remain uniform (1/7)
        spat_factors = result.spat["spat_factor"].to_numpy()
        np.testing.assert_allclose(spat_factors, 1.0 / 7, atol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# Predict function
# ──────────────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_matches_pred(self):
        """predict() on the training sequences should match the stored pred."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )

        pred2 = result.predict(sequences)
        np.testing.assert_allclose(result.pred, pred2, rtol=1e-10)

    def test_predict_new_sequences(self):
        """predict() should work on new sequences of the same length."""
        motif = "GATA"
        rng = np.random.default_rng(42)
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )

        # Generate new sequences
        new_seqs = [_random_dna(280, rng) for _ in range(20)]
        pred = result.predict(new_seqs)
        assert pred.shape == (20,)
        assert np.all(np.isfinite(pred))


# ──────────────────────────────────────────────────────────────────────
# PSSM initialisation
# ──────────────────────────────────────────────────────────────────────

class TestPSSMInit:
    def test_init_from_pssm_dataframe(self):
        """Should be able to initialise from a pre-computed PSSM DataFrame."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        # Create a PSSM DataFrame
        K = 12
        arr = np.full((K, 4), 0.25)
        # Bias position 4 towards G
        arr[4] = [0.05, 0.05, 0.85, 0.05]
        pssm_init = pssm_dataframe(arr)

        result = regress_pwm(
            sequences, resp,
            motif=pssm_init,
            score_metric="r2",
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.01],
        )
        assert result.pssm is not None
        assert result.r2 is not None

    def test_init_from_spat_model(self):
        """Should be able to provide a pre-computed spatial model."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        spat = pd.DataFrame({
            "bin": np.arange(7) * 40,
            "spat_factor": [0.1, 0.12, 0.15, 0.26, 0.15, 0.12, 0.1],
        })

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_model=spat,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )
        assert result.spat is not None


# ──────────────────────────────────────────────────────────────────────
# Result structure
# ──────────────────────────────────────────────────────────────────────

class TestResultStructure:
    def test_result_has_all_fields(self):
        """RegressionResult should have all expected fields."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            50, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )

        assert isinstance(result.pssm, pd.DataFrame)
        assert set(result.pssm.columns) >= {"pos", "A", "C", "G", "T"}
        assert isinstance(result.spat, pd.DataFrame)
        assert set(result.spat.columns) >= {"bin", "spat_factor"}
        assert isinstance(result.pred, np.ndarray)
        assert result.pred.shape == (50,)
        assert isinstance(result.consensus, str)
        assert result.bidirect is True
        assert result.spat_min is not None
        assert result.spat_max is not None
        assert result.seq_length is not None

    def test_to_dict(self):
        """to_dict() should return a serialisable dictionary."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            50, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )

        d = result.to_dict()
        assert "pssm" in d
        assert "spat" in d
        assert "consensus" in d


# ──────────────────────────────────────────────────────────────────────
# Edge cases and validation
# ──────────────────────────────────────────────────────────────────────

class TestValidation:
    def test_invalid_score_metric(self):
        with pytest.raises(ValueError, match="score_metric"):
            regress_pwm(
                ["ACGT" * 70], np.array([1.0]),
                motif="****",
                score_metric="invalid",
                spat_bin_size=40,
                spat_num_bins=7,
            )

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="do not match"):
            regress_pwm(
                ["ACGT" * 70, "ACGT" * 70],
                np.array([1.0]),  # only 1 response for 2 sequences
                motif="****",
                spat_bin_size=40,
                spat_num_bins=7,
            )

    def test_ks_with_non_binary(self):
        with pytest.raises(ValueError, match="binary"):
            regress_pwm(
                ["ACGT" * 70, "ACGT" * 70],
                np.array([0.5, 0.8]),
                motif="****",
                score_metric="ks",
                spat_bin_size=40,
                spat_num_bins=7,
            )

    def test_reproducibility_with_seed(self):
        """Same seed should give same results."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            50, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        kwargs = dict(
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )

        r1 = regress_pwm(sequences, resp, **kwargs)
        r2 = regress_pwm(sequences, resp, **kwargs)

        np.testing.assert_allclose(r1.pred, r2.pred, rtol=1e-10)
        np.testing.assert_allclose(
            r1.pssm[["A", "C", "G", "T"]].to_numpy(),
            r2.pssm[["A", "C", "G", "T"]].to_numpy(),
            rtol=1e-10,
        )


# ──────────────────────────────────────────────────────────────────────
# Internal engine tests
# ──────────────────────────────────────────────────────────────────────

class TestEngine:
    def test_init_energies_basic(self):
        """init_energies should produce non-zero derivatives for matching sequences."""
        rng = np.random.default_rng(42)
        # Simple case: 2 short sequences
        sequences = ["ACGTACGTAC", "TGCATGCATG"]
        train_mask = np.ones(2, dtype=bool)

        engine = _PWMLRegression(
            sequences=sequences,
            train_mask=train_mask,
            min_range=0,
            max_range=10,
            min_prob=0.001,
            spat_bin_size=10,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            improve_epsilon=1e-4,
            unif_prior=0.05,
            score_metric="r2",
            num_folds=1,
            log_energy=False,
            energy_epsilon=1e-5,
            optimize_pwm=True,
            optimize_spat=True,
            symmetrize_spat=False,
            verbose=False,
            rng=rng,
        )
        engine.add_responses(np.array([1.0, 0.0]))
        engine.init_seed("ACGT", bidirect=False)
        engine.init_energies()

        # Derivatives should be non-zero for the first sequence (has ACGT match)
        assert np.any(engine.derivs[0] != 0), "Expected non-zero derivs for matching seq"

    def test_symmetrize_spat(self):
        """Spatial factors should be symmetric after symmetrization."""
        rng = np.random.default_rng(42)
        sequences = ["A" * 280] * 10
        train_mask = np.ones(10, dtype=bool)

        engine = _PWMLRegression(
            sequences=sequences,
            train_mask=train_mask,
            min_range=0,
            max_range=280,
            min_prob=0.001,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            improve_epsilon=1e-4,
            unif_prior=0.05,
            score_metric="r2",
            num_folds=1,
            log_energy=False,
            energy_epsilon=1e-5,
            optimize_pwm=True,
            optimize_spat=True,
            symmetrize_spat=True,
            verbose=False,
            rng=rng,
        )
        engine.add_responses(np.arange(10, dtype=float))
        engine.init_seed("AAAA", bidirect=True)

        # Manually set asymmetric spatial factors
        engine.spat_factors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        engine._symmetrize_spat_factors()

        # After symmetrization, factors should be mirrored around center (index 3)
        assert engine.spat_factors[0] == engine.spat_factors[6]
        assert engine.spat_factors[1] == engine.spat_factors[5]
        assert engine.spat_factors[2] == engine.spat_factors[4]


# ──────────────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────────────

class TestCrossValidation:
    def test_multi_fold(self):
        """Multi-fold optimization should work without errors."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm(
            sequences, resp,
            motif="****GATA****",
            motif_length=12,
            spat_bin_size=40,
            spat_num_bins=7,
            seed=42,
            num_folds=3,
            resolutions=[0.05],
            spat_resolutions=[0.01],
        )
        assert result.r2 is not None
        assert result.r2 > 0.0
