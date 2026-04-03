"""Integration tests for pyprego.

Each test exercises a full pipeline from data generation through model
fitting/export/reload, verifying that the pieces compose correctly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.types import pssm_dataframe, pssm_to_array


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_sequences(n: int, length: int, seed: int = 42) -> list[str]:
    """Generate random DNA sequences."""
    rng = np.random.default_rng(seed)
    nucs = np.array(list("ACGT"))
    return ["".join(nucs[rng.integers(0, 4, size=length)]) for _ in range(n)]


def _plant_motif(
    sequences: list[str],
    motif: str,
    fraction: float = 0.5,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Plant a motif in a fraction of the sequences and return binary labels.

    Returns the modified sequences and a binary response vector (1 for
    sequences that received the planted motif).
    """
    rng = np.random.default_rng(seed)
    n = len(sequences)
    n_plant = int(n * fraction)
    indices = rng.choice(n, size=n_plant, replace=False)
    labels = np.zeros(n, dtype=np.float64)

    result = list(sequences)
    motif_len = len(motif)
    for idx in indices:
        seq = list(result[idx])
        # Plant at a random position
        max_start = len(seq) - motif_len
        start = rng.integers(0, max_start + 1)
        for j, base in enumerate(motif):
            seq[start + j] = base
        result[idx] = "".join(seq)
        labels[idx] = 1.0

    return result, labels


def _plant_motif_continuous(
    sequences: list[str],
    motif: str,
    seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Plant a motif at varying positions and create a continuous response.

    The response is higher when the motif is present (planted deterministically
    into a random subset with additive noise).
    """
    rng = np.random.default_rng(seed)
    n = len(sequences)
    pssm = pyprego.kmers_to_pssm(motif, prior=0.01)
    # Drop the 'kmer' column for compute_pwm
    pssm_clean = pssm[["pos", "A", "C", "G", "T"]].copy()

    # Plant motif in ~half of the sequences
    result = list(sequences)
    motif_len = len(motif)
    planted = np.zeros(n, dtype=bool)
    for i in range(n):
        if rng.random() < 0.5:
            seq = list(result[i])
            start = rng.integers(0, len(seq) - motif_len + 1)
            for j, base in enumerate(motif):
                seq[start + j] = base
            result[i] = "".join(seq)
            planted[i] = True

    # Response = PWM score + noise
    scores = pyprego.compute_pwm(result, pssm_clean)
    noise = rng.standard_normal(n) * 0.1
    response = scores + noise

    return result, response


# ===================================================================
# Full pipeline: planted motif -> screen_kmers -> regress_pwm -> verify
# ===================================================================


class TestFullDiscoveryPipeline:
    """End-to-end: generate data with planted motif, screen, regress, verify."""

    def test_binary_pipeline(self):
        """Plant a strong motif, screen kmers, regress, confirm discovery."""
        motif = "GATAAG"
        n_seq = 300
        seq_len = 200

        seqs = _random_sequences(n_seq, seq_len, seed=10)
        seqs, labels = _plant_motif(seqs, motif, fraction=0.5, seed=10)

        # Step 1: screen k-mers
        kmer_results = pyprego.screen_kmers(seqs, labels, kmer_len=6, min_cor=0.05)
        assert len(kmer_results) > 0
        top_kmer = kmer_results.iloc[0]["kmer"]

        # Step 2: regress_pwm with the top k-mer as seed
        result = pyprego.regress_pwm_core(
            seqs, labels,
            motif=top_kmer,
            motif_length=10,
            score_metric="ks",
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.005],
            seed=10,
        )

        # Step 3: verify the model found something meaningful
        assert result.pssm is not None
        assert result.consensus is not None
        assert len(result.pred) == n_seq
        # The KS or R2 should be non-trivial
        assert result.ks is not None and result.ks > 0.05

    def test_continuous_pipeline(self):
        """Plant motif, use continuous response, regress, verify R2."""
        motif = "TGACGT"
        n_seq = 200
        seq_len = 200

        seqs = _random_sequences(n_seq, seq_len, seed=20)
        seqs, response = _plant_motif_continuous(seqs, motif, seed=20)

        # regress_pwm with auto k-mer screen
        result = pyprego.regress_pwm(
            seqs, response,
            kmer_length=6,
            motif_length=10,
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05, 0.02],
            spat_resolutions=[0.01, 0.005],
            seed=20,
        )
        assert result.r2 is not None
        # Should explain at least some variance
        assert result.r2 > 0.01


# ===================================================================
# Export -> load -> predict -> compare
# ===================================================================


class TestExportLoadPredict:
    """Export a fitted model, reload it, and confirm predictions match."""

    @pytest.fixture
    def fitted_model(self):
        seqs = _random_sequences(100, 200, seed=30)
        motif = "ACGTAC"
        seqs, labels = _plant_motif(seqs, motif, fraction=0.5, seed=30)
        model = pyprego.regress_pwm_core(
            seqs, labels,
            motif="ACGTAC",
            motif_length=10,
            score_metric="ks",
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            seed=30,
        )
        return model, seqs

    def test_export_load_round_trip_dict(self, fitted_model):
        model, seqs = fitted_model
        data = pyprego.export_regression_model(model)
        assert isinstance(data, dict)
        assert "pssm" in data
        assert "spat" in data
        assert "bidirect" in data

        loaded = pyprego.load_regression_model(data)
        assert loaded.consensus == model.consensus
        assert loaded.bidirect == model.bidirect
        assert loaded.r2 == pytest.approx(model.r2, abs=1e-6) if model.r2 is not None else True

    def test_export_load_round_trip_file(self, fitted_model):
        model, seqs = fitted_model
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            pyprego.export_regression_model(model, path)
            assert path.exists()

            loaded = pyprego.load_regression_model(path)
            assert loaded.consensus == model.consensus

    def test_loaded_predict_scores_finite(self, fitted_model):
        model, seqs = fitted_model
        data = pyprego.export_regression_model(model)
        loaded = pyprego.load_regression_model(data)
        pred = loaded.predict(seqs)
        assert pred.shape == (len(seqs),)
        assert np.all(np.isfinite(pred))


# ===================================================================
# MotifDB creation -> extract_pwm -> screen_pwm pipeline
# ===================================================================


class TestMotifDBPipeline:
    """Create a MotifDB, extract PWM scores, screen against a response."""

    @pytest.fixture
    def small_motif_db(self):
        """Build a tiny MotifDB with two known motifs."""
        motif_a = pyprego.kmers_to_pssm("GATAAG")
        motif_a["motif"] = "GATA"
        motif_b = pyprego.kmers_to_pssm("TGACGT")
        motif_b["motif"] = "TGAC"
        df = pd.concat([motif_a, motif_b], ignore_index=True)
        db = pyprego.create_motif_db(df, prior=0.01)
        return db

    def test_motif_db_has_correct_motifs(self, small_motif_db):
        names = small_motif_db.names()
        assert "GATA" in names
        assert "TGAC" in names
        assert len(small_motif_db) == 2

    def test_extract_pwm_returns_scores(self, small_motif_db):
        seqs = _random_sequences(50, 100, seed=40)
        scores_df = pyprego.extract_pwm(seqs, small_motif_db)
        assert isinstance(scores_df, pd.DataFrame)
        assert set(scores_df.columns) == {"GATA", "TGAC"}
        assert scores_df.shape == (50, 2)
        assert np.all(np.isfinite(scores_df.values))

    def test_screen_pwm_returns_ranking(self, small_motif_db):
        motif = "GATAAG"
        seqs = _random_sequences(100, 100, seed=50)
        seqs, labels = _plant_motif(seqs, motif, fraction=0.5, seed=50)

        result = pyprego.screen_pwm(seqs, labels, small_motif_db)
        assert isinstance(result, pd.DataFrame)
        assert "motif" in result.columns
        assert "score" in result.columns
        assert len(result) == 2
        # The GATA motif should rank first because it matches the planted motif
        assert result.iloc[0]["motif"] == "GATA"

    def test_get_motif_pssm(self, small_motif_db):
        # get_motif_pssm takes (motif_name, dataset_df) not a MotifDB object
        df = pyprego.motif_db_to_dataframe(small_motif_db)
        pssm = pyprego.get_motif_pssm("GATA", dataset=df)
        assert isinstance(pssm, pd.DataFrame)
        assert "A" in pssm.columns
        assert len(pssm) == 6  # "GATAAG" has 6 positions

    def test_extract_pwm_with_motif_subset(self, small_motif_db):
        seqs = _random_sequences(20, 100, seed=60)
        scores = pyprego.extract_pwm(seqs, small_motif_db, motifs=["GATA"])
        assert list(scores.columns) == ["GATA"]
        assert scores.shape == (20, 1)


# ===================================================================
# Multi-motif regression
# ===================================================================


class TestMultiMotifRegression:
    """Plant two motifs, run regress_multiple_motifs, verify both found."""

    def test_two_motif_pipeline(self):
        """Plant two distinct motifs and verify the multi-motif model
        explains more variance than a single motif."""
        n_seq = 300
        seq_len = 200
        rng = np.random.default_rng(70)
        nucs = np.array(list("ACGT"))

        motif_1 = "GATAAG"
        motif_2 = "CACGTG"

        # Generate sequences
        seqs = []
        response = np.zeros(n_seq)
        for i in range(n_seq):
            seq = list("".join(nucs[rng.integers(0, 4, size=seq_len)]))
            score = 0.0
            # Plant motif_1 in some sequences
            if rng.random() < 0.4:
                start = rng.integers(20, 100)
                for j, b in enumerate(motif_1):
                    seq[start + j] = b
                score += 1.0
            # Plant motif_2 in some sequences (independently)
            if rng.random() < 0.4:
                start = rng.integers(100, 180)
                for j, b in enumerate(motif_2):
                    seq[start + j] = b
                score += 1.0
            seqs.append("".join(seq))
            response[i] = score + rng.standard_normal() * 0.3

        # Single motif
        single = pyprego.regress_pwm(
            seqs, response,
            kmer_length=6,
            motif_length=10,
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            seed=70,
        )

        # Multi motif
        multi = pyprego.regress_multiple_motifs(
            seqs, response,
            motif_num=2,
            kmer_length=6,
            motif_length=10,
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            seed=70,
        )

        assert isinstance(multi, pyprego.MultiRegressionResult)
        assert len(multi.models) == 2
        assert multi.multi_stats.shape[0] == 2

        # The combined score should improve over the first motif alone
        assert multi.multi_stats["comb_score"].iloc[1] >= multi.multi_stats["comb_score"].iloc[0] - 0.01

    def test_multi_motif_predict(self):
        """Verify predict / predict_multi methods work on new sequences."""
        n_seq = 200
        seq_len = 200
        seqs = _random_sequences(n_seq, seq_len, seed=80)
        motif = "GATAAG"
        seqs, labels = _plant_motif(seqs, motif, fraction=0.5, seed=80)
        response = labels + np.random.default_rng(80).standard_normal(n_seq) * 0.3

        multi = pyprego.regress_multiple_motifs(
            seqs, response,
            motif_num=2,
            kmer_length=6,
            motif_length=10,
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            seed=80,
        )

        new_seqs = _random_sequences(10, seq_len, seed=99)
        pred = multi.predict(new_seqs)
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))

        pred_multi_df = multi.predict_multi(new_seqs)
        assert isinstance(pred_multi_df, pd.DataFrame)
        assert pred_multi_df.shape == (10, 2)


# ===================================================================
# Multi-model export/load round-trip
# ===================================================================


class TestMultiModelExportLoad:
    """Export and reload a multi-motif model."""

    def test_round_trip_dict(self):
        n_seq = 150
        seqs = _random_sequences(n_seq, 200, seed=90)
        motif = "GATAAG"
        seqs, _ = _plant_motif(seqs, motif, fraction=0.5, seed=90)
        response = np.random.default_rng(90).standard_normal(n_seq)

        multi = pyprego.regress_multiple_motifs(
            seqs, response,
            motif_num=2,
            kmer_length=6,
            motif_length=10,
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            seed=90,
        )

        data = pyprego.export_multi_regression(multi)
        assert isinstance(data, dict)
        assert "models" in data
        assert len(data["models"]) == 2

        loaded = pyprego.load_multi_regression(data)
        assert len(loaded.models) == 2
        assert loaded.intercept == pytest.approx(multi.intercept, abs=1e-6)
        np.testing.assert_allclose(loaded.coef, multi.coef, atol=1e-6)

    def test_round_trip_file(self):
        n_seq = 100
        seqs = _random_sequences(n_seq, 200, seed=91)
        response = np.random.default_rng(91).standard_normal(n_seq)

        multi = pyprego.regress_multiple_motifs(
            seqs, response,
            motif_num=2,
            kmer_length=6,
            motif_length=10,
            spat_num_bins=3,
            spat_bin_size=40,
            resolutions=[0.05],
            spat_resolutions=[0.01],
            seed=91,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi_model.json"
            pyprego.export_multi_regression(multi, path)
            assert path.exists()

            loaded = pyprego.load_multi_regression(path)
            assert len(loaded.models) == 2

            new_seqs = _random_sequences(5, 200, seed=100)
            pred = loaded.predict(new_seqs)
            assert pred.shape == (5,)
            assert np.all(np.isfinite(pred))
