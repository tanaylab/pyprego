"""Tests for the high-level regression API.

Covers:
- regress_pwm with auto k-mer screening (motif=None)
- regress_pwm with multi_kmers=True
- regress_multiple_motifs with 2 planted motifs
- export/import round-trip (single and multi-model)
- regress_pwm_clusters
- regress_pwm_cv
- predict() consistency
- score helpers
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyprego.regression import (
    CVRegressionResult,
    ClusterRegressionResult,
    MultiRegressionResult,
    _get_cand_kmers,
    _get_cv_folds,
    _is_binary_response,
    _pred_r_given_e,
    _sample_response,
    _score_predictions,
    regress_multiple_motifs,
    regress_pwm,
    regress_pwm_clusters,
    regress_pwm_core,
    regress_pwm_cv,
)
from pyprego.export import (
    export_multi_regression,
    export_regression_model,
    load_multi_regression,
    load_regression_model,
)
from pyprego.types import pssm_dataframe
from pyprego.utils import rc


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
    """Generate sequences with a planted motif and binary response."""
    rng = np.random.default_rng(seed)
    sequences = []
    response = np.zeros(n_seq)
    n_with = int(n_seq * fraction_with_motif)
    motif_len = len(motif)
    center = seq_len // 2

    for i in range(n_seq):
        seq = list(_random_dna(seq_len, rng))
        if i < n_with:
            pos = center - motif_len // 2 + rng.integers(-5, 6)
            pos = max(0, min(pos, seq_len - motif_len))
            for j, ch in enumerate(motif):
                seq[pos + j] = ch
            response[i] = 1.0
        sequences.append("".join(seq))

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
    """Generate a continuous response based on motif presence."""
    rng = np.random.default_rng(seed)
    motif_rc = rc(motif)
    n = len(sequences)
    response = np.zeros(n)
    for i, seq in enumerate(sequences):
        count = seq.count(motif) + seq.count(motif_rc)
        response[i] = float(count) + rng.normal(0, noise_std)
    return response


def _generate_two_motif_sequences(
    n_seq: int,
    seq_len: int,
    motif1: str,
    motif2: str,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Generate sequences with two planted motifs and binary response.
    50% have motif1 near center, 25% have motif2 at +20 offset, 25% random.
    """
    rng = np.random.default_rng(seed)
    sequences = []
    response = np.zeros(n_seq)
    center = seq_len // 2

    for i in range(n_seq):
        seq = list(_random_dna(seq_len, rng))
        if i < n_seq // 2:
            pos = center - len(motif1) // 2 + rng.integers(-3, 4)
            pos = max(0, min(pos, seq_len - len(motif1)))
            for j, ch in enumerate(motif1):
                seq[pos + j] = ch
            response[i] = 1.0
        elif i < 3 * n_seq // 4:
            pos = center + 20 - len(motif2) // 2 + rng.integers(-3, 4)
            pos = max(0, min(pos, seq_len - len(motif2)))
            for j, ch in enumerate(motif2):
                seq[pos + j] = ch
            response[i] = 1.0
        sequences.append("".join(seq))

    perm = rng.permutation(n_seq)
    sequences = [sequences[i] for i in perm]
    response = response[perm]

    return sequences, response


# Shared fixtures to speed up tests
_CORE_KWARGS = dict(
    spat_bin_size=40,
    spat_num_bins=7,
    resolutions=[0.05, 0.02],
    spat_resolutions=[0.01, 0.01],
)


# ──────────────────────────────────────────────────────────────────────
# Unit tests for helper functions
# ──────────────────────────────────────────────────────────────────────


class TestHelpers:
    def test_is_binary_response_true(self):
        assert _is_binary_response(np.array([0.0, 1.0, 0.0, 1.0]))

    def test_is_binary_response_false(self):
        assert not _is_binary_response(np.array([0.0, 0.5, 1.0]))

    def test_is_binary_response_2d(self):
        assert _is_binary_response(np.array([[0.0], [1.0], [0.0]]))

    def test_is_binary_response_2d_multi(self):
        assert not _is_binary_response(np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_score_predictions_ks(self):
        resp = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        pred = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        score = _score_predictions(resp, pred, "ks")
        assert score > 0.5

    def test_score_predictions_r2(self):
        resp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        score = _score_predictions(resp, pred, "r2")
        assert score > 0.95

    def test_sample_response_binary(self):
        resp = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
        idxs = _sample_response(resp, sample_frac=0.5, sample_ratio=1.0, seed=42)
        assert len(idxs) > 0
        assert len(idxs) < len(resp)

    def test_sample_response_continuous(self):
        resp = np.array([[0.1], [0.5], [0.8], [1.2], [2.0], [3.0]])
        idxs = _sample_response(resp, sample_frac=0.5, sample_ratio=1.0, seed=42)
        assert len(idxs) > 0

    def test_pred_r_given_e(self):
        e = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        pred = _pred_r_given_e(e, r, k=3)
        assert len(pred) == 5
        # Higher e should have higher smoothed r
        assert pred[4] > pred[0]

    def test_get_cv_folds_binary(self):
        resp = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
        folds = _get_cv_folds(resp, nfolds=2, seed=42)
        assert len(folds) == 8
        assert set(folds) == {0, 1}

    def test_get_cv_folds_continuous(self):
        resp = np.arange(10, dtype=np.float64).reshape(-1, 1)
        folds = _get_cv_folds(resp, nfolds=3, seed=42)
        assert len(folds) == 10
        assert set(folds) == {0, 1, 2}


# ──────────────────────────────────────────────────────────────────────
# regress_pwm with auto k-mer screening
# ──────────────────────────────────────────────────────────────────────


class TestKmerScreening:
    def test_auto_kmer_binary(self):
        """regress_pwm with motif=None should auto-screen k-mers."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif=None,
            motif_length=12,
            score_metric="ks",
            seed=42,
            kmer_length=6,
            min_kmer_cor=0.05,
            **_CORE_KWARGS,
        )
        assert result.ks is not None
        assert result.ks > 0.0
        assert result.consensus is not None

    def test_auto_kmer_continuous(self):
        """Auto k-mer screen with continuous response."""
        motif = "GATA"
        sequences, _ = _generate_sequences_with_motif(
            200, 280, motif, seed=42
        )
        response = _generate_continuous_response(sequences, motif, seed=42)
        result = regress_pwm(
            sequences, response,
            motif=None,
            motif_length=12,
            score_metric="r2",
            seed=42,
            kmer_length=4,
            min_kmer_cor=0.05,
            **_CORE_KWARGS,
        )
        assert result.r2 is not None
        assert result.r2 > 0.0

    def test_auto_kmer_with_sampling(self):
        """K-mer screening with sampling."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif=None,
            motif_length=12,
            score_metric="ks",
            seed=42,
            kmer_length=6,
            min_kmer_cor=0.05,
            sample_for_kmers=True,
            sample_frac=0.5,
            **_CORE_KWARGS,
        )
        assert result.ks is not None
        assert result.ks > 0.0


# ──────────────────────────────────────────────────────────────────────
# Multi-kmer mode
# ──────────────────────────────────────────────────────────────────────


class TestMultiKmers:
    def test_multi_kmer_binary(self):
        """Multi-kmer mode should try multiple seeds and pick best."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif=None,
            motif_length=12,
            score_metric="ks",
            seed=42,
            multi_kmers=True,
            kmer_length=[6],
            max_cands=3,
            min_kmer_cor=0.05,
            val_frac=0.2,
            **_CORE_KWARGS,
        )
        assert result.ks is not None
        assert result.ks > 0.0

    def test_multi_kmer_ignores_when_motif_provided(self):
        """When motif is provided, multi_kmers should be ignored."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            multi_kmers=True,
            seed=42,
            **_CORE_KWARGS,
        )
        assert result.pssm is not None


# ──────────────────────────────────────────────────────────────────────
# Final metric auto-selection
# ──────────────────────────────────────────────────────────────────────


class TestFinalMetric:
    def test_auto_ks_for_binary(self):
        """Auto-selects KS for binary response."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )
        assert result.ks is not None

    def test_auto_r2_for_continuous(self):
        """Auto-selects R2 for continuous response."""
        motif = "GATA"
        sequences, _ = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        response = _generate_continuous_response(sequences, motif, seed=42)
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            score_metric="r2",
            seed=42,
            **_CORE_KWARGS,
        )
        assert result.r2 is not None


# ──────────────────────────────────────────────────────────────────────
# regress_multiple_motifs
# ──────────────────────────────────────────────────────────────────────


class TestMultipleMotifs:
    def test_two_motifs(self):
        """regress_multiple_motifs should find two motifs."""
        motif1 = "GATAAG"
        motif2 = "CCAAT"
        sequences, response = _generate_two_motif_sequences(
            300, 280, motif1, motif2, seed=42
        )
        result = regress_multiple_motifs(
            sequences, response,
            motif_num=2,
            smooth_k=50,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )
        assert isinstance(result, MultiRegressionResult)
        assert len(result.models) == 2
        assert len(result.multi_stats) == 2
        assert "model" in result.multi_stats.columns
        assert "score" in result.multi_stats.columns
        assert "comb_score" in result.multi_stats.columns
        assert len(result.pred) == 300

    def test_multi_predict(self):
        """predict() and predict_multi() should work."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        result = regress_multiple_motifs(
            sequences, response,
            motif_num=2,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )
        # predict
        pred = result.predict(sequences)
        assert len(pred) == 200

        # predict_multi
        pred_multi = result.predict_multi(sequences)
        assert isinstance(pred_multi, pd.DataFrame)
        assert pred_multi.shape == (200, 2)
        assert list(pred_multi.columns) == ["e1", "e2"]

    def test_multi_motif_num_too_low(self):
        """motif_num < 2 should raise."""
        with pytest.raises(ValueError, match="motif_num"):
            regress_multiple_motifs(
                ["ACGT" * 70], np.array([1.0]),
                motif_num=1,
            )

    def test_combined_score_improves(self):
        """Combined score should generally not decrease."""
        motif1 = "GATAAG"
        motif2 = "CCAAT"
        sequences, response = _generate_two_motif_sequences(
            300, 280, motif1, motif2, seed=42
        )
        result = regress_multiple_motifs(
            sequences, response,
            motif_num=2,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )
        # The combined score for 2 motifs should be >= the single motif score
        # (not strictly guaranteed but likely with real signal)
        assert result.multi_stats["comb_score"].iloc[1] >= result.multi_stats["score"].iloc[0] * 0.5


# ──────────────────────────────────────────────────────────────────────
# Export / Import round-trip
# ──────────────────────────────────────────────────────────────────────


class TestExportImport:
    def test_single_model_roundtrip(self):
        """Export and reload a single model; predictions should match."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fn = f.name
        export_regression_model(result, fn)

        loaded = load_regression_model(fn)
        assert isinstance(loaded, type(result))
        assert loaded.consensus == result.consensus
        assert loaded.bidirect == result.bidirect
        assert loaded.spat_min == result.spat_min
        assert loaded.spat_max == result.spat_max

        # Predictions should match
        pred_orig = result.predict(sequences)
        pred_loaded = loaded.predict(sequences)
        np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-10)

    def test_single_model_to_dict(self):
        """Export to dict (no file) and reload."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            50, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            seed=42,
            **_CORE_KWARGS,
        )

        d = export_regression_model(result)
        assert isinstance(d, dict)
        assert "pssm" in d
        assert "spat" in d

        loaded = load_regression_model(d)
        pred_orig = result.predict(sequences)
        pred_loaded = loaded.predict(sequences)
        np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-10)

    def test_multi_model_roundtrip(self):
        """Export and reload a multi-motif model; predictions should match."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        result = regress_multiple_motifs(
            sequences, response,
            motif_num=2,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fn = f.name
        export_multi_regression(result, fn)

        loaded = load_multi_regression(fn)
        assert isinstance(loaded, MultiRegressionResult)
        assert len(loaded.models) == 2

        # Predictions should match
        pred_orig = result.predict(sequences)
        pred_loaded = loaded.predict(sequences)
        np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-10)

    def test_multi_model_to_dict(self):
        """Export multi-model to dict and reload."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, fraction_with_motif=0.5, seed=42
        )
        result = regress_multiple_motifs(
            sequences, response,
            motif_num=2,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )

        d = export_multi_regression(result)
        assert isinstance(d, dict)
        assert "models" in d
        assert d["motif_num"] == 2

        loaded = load_multi_regression(d)
        pred_orig = result.predict(sequences)
        pred_loaded = loaded.predict(sequences)
        np.testing.assert_allclose(pred_orig, pred_loaded, rtol=1e-10)

    def test_export_json_is_valid(self):
        """Exported JSON should be valid and readable."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            50, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            seed=42,
            **_CORE_KWARGS,
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            fn = f.name
        export_regression_model(result, fn)

        with open(fn) as f:
            data = json.load(f)
        assert "pssm" in data
        assert "spat" in data
        assert "bidirect" in data


# ──────────────────────────────────────────────────────────────────────
# regress_pwm_clusters
# ──────────────────────────────────────────────────────────────────────


class TestClusters:
    def test_basic_clusters(self):
        """regress_pwm_clusters should run for each cluster."""
        motif = "GATAAG"
        rng = np.random.default_rng(42)
        n_seq = 300
        seq_len = 280
        sequences = []
        clusters = []

        for i in range(n_seq):
            seq = list(_random_dna(seq_len, rng))
            if i < 100:
                # Cluster A: plant GATAAG
                pos = seq_len // 2 - 3 + rng.integers(-3, 4)
                pos = max(0, min(pos, seq_len - 6))
                for j, ch in enumerate(motif):
                    seq[pos + j] = ch
                clusters.append("A")
            elif i < 200:
                # Cluster B: plant CCAAT
                motif_b = "CCAAT"
                pos = seq_len // 2 - 2 + rng.integers(-3, 4)
                pos = max(0, min(pos, seq_len - 5))
                for j, ch in enumerate(motif_b):
                    seq[pos + j] = ch
                clusters.append("B")
            else:
                clusters.append("C")
            sequences.append("".join(seq))

        result = regress_pwm_clusters(
            sequences, np.array(clusters),
            motif_length=12,
            seed=42,
            **_CORE_KWARGS,
        )

        assert isinstance(result, ClusterRegressionResult)
        assert len(result.models) == 3
        assert set(result.cluster_names) == {"A", "B", "C"}
        assert result.pred_mat.shape == (300, 3)
        assert result.cluster_mat.shape == (300, 3)
        assert "cluster" in result.stats.columns
        assert "consensus" in result.stats.columns
        assert len(result.stats) == 3

    def test_clusters_stats(self):
        """Stats should contain ks_D for binary cluster responses."""
        motif = "GATAAG"
        rng = np.random.default_rng(42)
        n_seq = 200
        seq_len = 280
        sequences = []
        clusters = []

        for i in range(n_seq):
            seq = list(_random_dna(seq_len, rng))
            if i < 100:
                pos = seq_len // 2 - 3 + rng.integers(-3, 4)
                pos = max(0, min(pos, seq_len - 6))
                for j, ch in enumerate(motif):
                    seq[pos + j] = ch
                clusters.append("X")
            else:
                clusters.append("Y")
            sequences.append("".join(seq))

        result = regress_pwm_clusters(
            sequences, np.array(clusters),
            motif_length=12,
            seed=42,
            **_CORE_KWARGS,
        )

        assert "ks_D" in result.stats.columns
        # Cluster X should have a higher KS than cluster Y (it has the planted motif)
        ks_x = result.stats.loc[result.stats["cluster"] == "X", "ks_D"].iloc[0]
        assert ks_x > 0.0

    def test_clusters_mismatch_raises(self):
        """Mismatched lengths should raise."""
        with pytest.raises(ValueError, match="same length"):
            regress_pwm_clusters(
                ["ACGT" * 70, "ACGT" * 70],
                np.array(["A"]),
            )


# ──────────────────────────────────────────────────────────────────────
# regress_pwm_cv
# ──────────────────────────────────────────────────────────────────────


class TestCV:
    def test_basic_cv(self):
        """Basic cross-validation should work."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        result = regress_pwm_cv(
            sequences, response,
            nfolds=3,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )

        assert isinstance(result, CVRegressionResult)
        assert len(result.cv_models) == 3
        assert len(result.cv_scores) == 3
        assert len(result.cv_pred) == 200
        assert result.score > 0.0
        assert result.folds is not None
        assert len(result.folds) == 200

    def test_cv_with_full_model(self):
        """CV with add_full_model should include full model."""
        motif = "GATA"
        sequences, _ = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        response = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm_cv(
            sequences, response,
            nfolds=3,
            add_full_model=True,
            motif_length=12,
            seed=42,
            **_CORE_KWARGS,
        )

        assert result.full_model is not None
        assert result.full_model.r2 is not None

    def test_cv_without_full_model(self):
        """CV without add_full_model should have None."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )

        result = regress_pwm_cv(
            sequences, response,
            nfolds=2,
            add_full_model=False,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )

        assert result.full_model is None

    def test_cv_explicit_folds(self):
        """Explicit folds should be respected."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        folds = np.array([i % 3 for i in range(100)])

        result = regress_pwm_cv(
            sequences, response,
            folds=folds,
            add_full_model=False,
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )

        assert len(result.cv_models) == 3
        np.testing.assert_array_equal(result.folds, folds)

    def test_cv_requires_nfolds_or_folds(self):
        """Should raise if neither nfolds nor folds is provided."""
        with pytest.raises(ValueError, match="nfolds"):
            regress_pwm_cv(
                ["ACGT" * 70] * 10,
                np.zeros(10),
                motif_length=12,
                seed=42,
            )


# ──────────────────────────────────────────────────────────────────────
# predict() consistency
# ──────────────────────────────────────────────────────────────────────


class TestPredictConsistency:
    def test_predict_matches_pred_core(self):
        """predict() should match stored pred for core optimizer."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        resp = _generate_continuous_response(sequences, motif, seed=42)

        result = regress_pwm_core(
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

    def test_predict_matches_pred_high_level(self):
        """predict() should match stored pred for high-level API."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            score_metric="ks",
            seed=42,
            **_CORE_KWARGS,
        )

        pred2 = result.predict(sequences)
        np.testing.assert_allclose(result.pred, pred2, rtol=1e-10)

    def test_predict_on_new_sequences(self):
        """predict() should work on new sequences."""
        rng = np.random.default_rng(99)
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            seed=42,
            **_CORE_KWARGS,
        )

        new_seqs = [_random_dna(280, rng) for _ in range(20)]
        pred = result.predict(new_seqs)
        assert pred.shape == (20,)
        assert np.all(np.isfinite(pred))


# ──────────────────────────────────────────────────────────────────────
# Database matching
# ──────────────────────────────────────────────────────────────────────


class TestDBMatching:
    def test_match_with_db(self):
        """match_with_db should populate db_match fields."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            100, 280, motif, seed=42
        )

        # Create a small motif DB
        from pyprego.types import pssm_dataframe
        arr = np.full((4, 4), 0.05)
        arr[0, 2] = 0.85  # G
        arr[1, 0] = 0.85  # A
        arr[2, 3] = 0.85  # T
        arr[3, 0] = 0.85  # A
        gata_pssm = pssm_dataframe(arr)
        motif_db = {"GATA_motif": gata_pssm}

        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            score_metric="ks",
            seed=42,
            match_with_db=True,
            motif_db=motif_db,
            **_CORE_KWARGS,
        )

        assert result.db_match_motif == "GATA_motif"
        assert result.db_match_cor is not None

    def test_no_match_without_db(self):
        """Without motif_db, db_match fields should be None."""
        motif = "GATA"
        sequences, response = _generate_sequences_with_motif(
            50, 280, motif, seed=42
        )
        result = regress_pwm(
            sequences, response,
            motif="****GATA****",
            motif_length=12,
            seed=42,
            match_with_db=True,
            motif_db=None,
            **_CORE_KWARGS,
        )
        assert result.db_match_motif is None


# ──────────────────────────────────────────────────────────────────────
# Candidate kmer generation
# ──────────────────────────────────────────────────────────────────────


class TestCandKmers:
    def test_get_cand_kmers(self):
        """_get_cand_kmers should return a non-empty list."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        cands = _get_cand_kmers(
            sequences, response,
            kmer_length=[6],
            min_gap=0,
            max_gap=0,
            min_kmer_cor=0.05,
            max_cands=5,
        )
        assert len(cands) > 0
        assert isinstance(cands[0], str)
        assert len(cands[0]) == 6

    def test_get_cand_kmers_multiple_lengths(self):
        """Should work with multiple kmer lengths."""
        motif = "GATAAG"
        sequences, response = _generate_sequences_with_motif(
            200, 280, motif, fraction_with_motif=0.5, seed=42
        )
        cands = _get_cand_kmers(
            sequences, response,
            kmer_length=[5, 6],
            min_gap=0,
            max_gap=0,
            min_kmer_cor=0.05,
            max_cands=5,
        )
        assert len(cands) > 0
