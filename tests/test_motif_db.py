"""Tests for the MotifDB class and motif database functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.motif_db import (
    MotifDB,
    _is_binary,
    _motif_db_to_mat,
    create_motif_db,
    extract_pwm,
    motif_db_to_dataframe,
    motif_enrichment,
    screen_pwm,
    set_prior,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_motif_df(n_motifs: int = 2) -> pd.DataFrame:
    """Create a simple test motif DataFrame (matches R test helper)."""
    rows = []
    if n_motifs >= 1:
        # motif1: 4 positions
        rows.extend(
            [
                {"motif": "motif1", "A": 0.7, "C": 0.1, "G": 0.1, "T": 0.1},
                {"motif": "motif1", "A": 0.1, "C": 0.7, "G": 0.1, "T": 0.1},
                {"motif": "motif1", "A": 0.2, "C": 0.3, "G": 0.3, "T": 0.2},
                {"motif": "motif1", "A": 0.3, "C": 0.2, "G": 0.3, "T": 0.2},
            ]
        )
    if n_motifs >= 2:
        # motif2: 4 positions
        rows.extend(
            [
                {"motif": "motif2", "A": 0.1, "C": 0.2, "G": 0.6, "T": 0.1},
                {"motif": "motif2", "A": 0.6, "C": 0.1, "G": 0.2, "T": 0.1},
                {"motif": "motif2", "A": 0.2, "C": 0.6, "G": 0.1, "T": 0.1},
                {"motif": "motif2", "A": 0.1, "C": 0.2, "G": 0.1, "T": 0.6},
            ]
        )
    return pd.DataFrame(rows)


def _create_test_db(
    prior: float = 0.01,
    spat_min: float | None = None,
    spat_max: float | None = None,
) -> MotifDB:
    """Create a simple test MotifDB."""
    return create_motif_db(
        _make_test_motif_df(),
        prior=prior,
        spat_min=spat_min,
        spat_max=spat_max,
    )


# ---------------------------------------------------------------------------
# MotifDB creation and validation
# ---------------------------------------------------------------------------


class TestMotifDBCreation:
    """Tests for MotifDB object creation."""

    def test_create_basic(self) -> None:
        db = _create_test_db()
        assert isinstance(db, MotifDB)
        assert len(db) == 2
        assert db.names() == ["motif1", "motif2"]

    def test_create_with_spatial_bounds(self) -> None:
        db = _create_test_db(spat_min=0, spat_max=100)
        assert db.spat_min == 0
        assert db.spat_max == 100
        assert len(db) == 2

    def test_mat_dimensions(self) -> None:
        db = _create_test_db()
        # 4 positions * 4 nucleotides = 16 rows, 2 motifs
        assert db.mat.shape == (16, 2)
        assert db.rc_mat.shape == (16, 2)

    def test_mat_log_scale(self) -> None:
        db = _create_test_db()
        # All values in the matrix should be <= 0 (log of probability)
        # or 0 (zero-padded positions)
        assert np.all(db.mat <= 0)
        assert np.all(db.rc_mat <= 0)

    def test_motif_lengths(self) -> None:
        db = _create_test_db()
        assert db.motif_lengths == {"motif1": 4, "motif2": 4}

    def test_single_motif(self) -> None:
        df = _make_test_motif_df(n_motifs=1)
        db = create_motif_db(df)
        assert len(db) == 1
        assert db.names() == ["motif1"]
        assert db.mat.shape == (16, 1)

    def test_default_spat_factors(self) -> None:
        db = _create_test_db()
        assert db.spat_factors.shape == (2, 1)
        np.testing.assert_array_equal(db.spat_factors, np.ones((2, 1)))


class TestMotifDBValidation:
    """Tests for MotifDB validation."""

    def test_prior_too_low(self) -> None:
        with pytest.raises(ValueError, match="Prior must be between"):
            _create_test_db(prior=0)

    def test_prior_too_high(self) -> None:
        with pytest.raises(ValueError, match="Prior must be between"):
            _create_test_db(prior=1)

    def test_prior_negative(self) -> None:
        with pytest.raises(ValueError, match="Prior must be between"):
            _create_test_db(prior=-0.1)

    def test_valid_spatial_bounds(self) -> None:
        # Should not raise
        _create_test_db(spat_min=0, spat_max=100)
        _create_test_db(spat_min=0, spat_max=0)

    def test_invalid_spatial_bounds_reversed(self) -> None:
        with pytest.raises(ValueError, match="less than or equal"):
            _create_test_db(spat_min=100, spat_max=0)

    def test_invalid_spatial_bounds_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _create_test_db(spat_min=-1, spat_max=100)

    def test_none_spatial_bounds(self) -> None:
        # None bounds are valid
        db = _create_test_db(spat_min=None, spat_max=None)
        assert db.spat_min is None
        assert db.spat_max is None

    def test_missing_columns(self) -> None:
        df = pd.DataFrame({"motif": ["m1"], "A": [0.25], "C": [0.25]})
        with pytest.raises(ValueError, match="missing required columns"):
            create_motif_db(df)

    def test_spatial_factors_wrong_size(self) -> None:
        df = _make_test_motif_df()
        # 3 rows but only 2 motifs
        bad_spat = np.ones((3, 1))
        with pytest.raises(ValueError, match="spatial factors"):
            create_motif_db(df, spat_factors=bad_spat)

    def test_spatial_factors_negative(self) -> None:
        df = _make_test_motif_df()
        bad_spat = np.array([[-1.0], [1.0]])
        with pytest.raises(ValueError, match="non-negative"):
            create_motif_db(df, spat_factors=bad_spat)


# ---------------------------------------------------------------------------
# Subscript operator
# ---------------------------------------------------------------------------


class TestMotifDBSubscript:
    """Tests for MotifDB __getitem__."""

    def test_getitem_by_name(self) -> None:
        db = _create_test_db()
        sub = db["motif1"]
        assert len(sub) == 1
        assert sub.names() == ["motif1"]
        assert sub.mat.shape[1] == 1

    def test_getitem_by_name_list(self) -> None:
        db = _create_test_db()
        sub = db[["motif1", "motif2"]]
        assert len(sub) == 2

    def test_getitem_by_index(self) -> None:
        db = _create_test_db()
        sub = db[0]
        assert len(sub) == 1
        assert sub.names() == ["motif1"]

    def test_getitem_by_index_list(self) -> None:
        db = _create_test_db()
        sub = db[[0, 1]]
        assert len(sub) == 2

    def test_getitem_preserves_spatial(self) -> None:
        db = _create_test_db(spat_min=0, spat_max=100)
        sub = db["motif1"]
        assert sub.spat_min == 0
        assert sub.spat_max == 100

    def test_getitem_preserves_prior(self) -> None:
        db = _create_test_db(prior=0.05)
        sub = db["motif1"]
        assert sub.prior == 0.05

    def test_getitem_missing_name(self) -> None:
        db = _create_test_db()
        with pytest.raises(KeyError, match="not found"):
            db["nonexistent"]

    def test_getitem_index_out_of_bounds(self) -> None:
        db = _create_test_db()
        with pytest.raises(IndexError, match="out of bounds"):
            db[5]

    def test_getitem_empty_list(self) -> None:
        db = _create_test_db()
        with pytest.raises(IndexError, match="Empty"):
            db[[]]


class TestMotifDBGrep:
    """Tests for MotifDB grep (pattern matching)."""

    def test_grep_basic(self) -> None:
        db = _create_test_db()
        sub = db.grep("motif1")
        assert len(sub) == 1
        assert sub.names() == ["motif1"]

    def test_grep_matches_both(self) -> None:
        db = _create_test_db()
        sub = db.grep("motif")
        assert len(sub) == 2

    def test_grep_case_insensitive(self) -> None:
        db = _create_test_db()
        sub = db.grep("MOTIF")
        assert len(sub) == 2

    def test_grep_no_match(self) -> None:
        db = _create_test_db()
        with pytest.raises(KeyError, match="No motifs matched"):
            db.grep("xyz123")


# ---------------------------------------------------------------------------
# Container protocol
# ---------------------------------------------------------------------------


class TestMotifDBContainer:
    """Tests for len, names, contains, iter, repr."""

    def test_len(self) -> None:
        db = _create_test_db()
        assert len(db) == 2

    def test_contains(self) -> None:
        db = _create_test_db()
        assert "motif1" in db
        assert "nonexistent" not in db

    def test_iter(self) -> None:
        db = _create_test_db()
        assert list(db) == ["motif1", "motif2"]

    def test_repr(self) -> None:
        db = _create_test_db()
        r = repr(db)
        assert "2 motifs" in r
        assert "prior=0.01" in r


# ---------------------------------------------------------------------------
# motif_db_to_dataframe round-trip
# ---------------------------------------------------------------------------


class TestMotifDBToDataframe:
    """Tests for motif_db_to_dataframe and round-trip consistency."""

    def test_basic_conversion(self) -> None:
        db = _create_test_db()
        df = motif_db_to_dataframe(db)
        assert set(df.columns) == {"motif", "pos", "A", "C", "G", "T"}
        assert len(df) == 8  # 2 motifs * 4 positions
        assert set(df["motif"].unique()) == {"motif1", "motif2"}

    def test_round_trip_values(self) -> None:
        """Verify that create -> to_dataframe recovers original values."""
        original_df = _make_test_motif_df()
        db = create_motif_db(original_df, prior=0.01)
        recovered = motif_db_to_dataframe(db)

        # Merge and compare
        original_sorted = original_df.copy()
        original_sorted["pos"] = original_sorted.groupby("motif", sort=False).cumcount() + 1
        original_sorted = original_sorted.sort_values(["motif", "pos"]).reset_index(
            drop=True
        )
        recovered_sorted = recovered.sort_values(["motif", "pos"]).reset_index(
            drop=True
        )

        for nuc in ["A", "C", "G", "T"]:
            np.testing.assert_allclose(
                recovered_sorted[nuc].values,
                original_sorted[nuc].values,
                atol=1e-6,
                err_msg=f"Round-trip mismatch for {nuc}",
            )

    def test_spatial_bounds_dont_affect_conversion(self) -> None:
        db1 = _create_test_db(spat_min=0, spat_max=100)
        db2 = _create_test_db()
        df1 = motif_db_to_dataframe(db1)
        df2 = motif_db_to_dataframe(db2)
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# set_prior
# ---------------------------------------------------------------------------


class TestSetPrior:
    """Tests for set_prior."""

    def test_set_prior_changes_prior(self) -> None:
        db = _create_test_db(prior=0.01)
        db2 = set_prior(db, 0.05)
        assert db2.prior == 0.05
        # Original unchanged
        assert db.prior == 0.01

    def test_set_prior_preserves_spatial(self) -> None:
        db = _create_test_db(prior=0.01, spat_min=0, spat_max=100)
        db2 = set_prior(db, 0.02)
        assert db2.spat_min == 0
        assert db2.spat_max == 100

    def test_set_prior_preserves_motif_count(self) -> None:
        db = _create_test_db()
        db2 = set_prior(db, 0.05)
        assert len(db2) == len(db)
        assert db2.names() == db.names()


# ---------------------------------------------------------------------------
# Reverse complement matrix
# ---------------------------------------------------------------------------


class TestReverseComplement:
    """Tests for reverse complement matrix correctness."""

    def test_rc_matrix_consistency(self) -> None:
        """Forward and RC matrices should be consistent (complement + reverse)."""
        db = _create_test_db()
        # For motif1, position 1 nucleotide A should match
        # RC position 4 nucleotide T (complement)
        motif_idx = 0
        motif_len = db.motif_lengths["motif1"]

        for pos in range(1, motif_len + 1):
            for nuc_idx in range(4):
                fwd_val = db.mat[(pos - 1) * 4 + nuc_idx, motif_idx]
                rc_pos = motif_len - pos + 1
                # Complement: A(0)->T(3), C(1)->G(2), G(2)->C(1), T(3)->A(0)
                rc_nuc = [3, 2, 1, 0][nuc_idx]
                rc_val = db.rc_mat[(rc_pos - 1) * 4 + rc_nuc, motif_idx]
                np.testing.assert_allclose(
                    fwd_val,
                    rc_val,
                    atol=1e-10,
                    err_msg=f"RC mismatch at pos={pos}, nuc={nuc_idx}",
                )


# ---------------------------------------------------------------------------
# extract_pwm
# ---------------------------------------------------------------------------


class TestExtractPWM:
    """Tests for extract_pwm."""

    def test_basic_extraction(self) -> None:
        db = _create_test_db()
        seqs = ["ACGTACGT", "TGCATGCA", "AAAACCCC"]
        result = extract_pwm(seqs, db)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["motif1", "motif2"]
        assert len(result) == 3

    def test_result_is_numeric(self) -> None:
        db = _create_test_db()
        seqs = ["ACGTACGT", "TGCATGCA"]
        result = extract_pwm(seqs, db)
        assert result.dtypes.apply(lambda x: np.issubdtype(x, np.floating)).all()

    def test_single_motif(self) -> None:
        db = _create_test_db()
        seqs = ["ACGTACGT"]
        result = extract_pwm(seqs, db, motifs=["motif1"])
        assert list(result.columns) == ["motif1"]

    def test_from_dataframe(self) -> None:
        df = _make_test_motif_df()
        seqs = ["ACGTACGT", "TGCATGCA"]
        result = extract_pwm(seqs, df)
        assert list(result.columns) == ["motif1", "motif2"]
        assert len(result) == 2

    def test_r_compatibility(self) -> None:
        """Values should approximately match R extract_pwm output."""
        db = _create_test_db()
        seqs = ["ACGTACGT", "TGCATGCA", "AAAACCCC"]
        result = extract_pwm(seqs, db)

        # R output:
        # motif1: -2.150250, -3.916885, -3.056117
        # motif2: -4.107756, -3.766947, -3.334878
        np.testing.assert_allclose(
            result["motif1"].values,
            [-2.150250, -3.916885, -3.056117],
            atol=0.01,
        )
        np.testing.assert_allclose(
            result["motif2"].values,
            [-4.107756, -3.766947, -3.334878],
            atol=0.01,
        )


# ---------------------------------------------------------------------------
# screen_pwm
# ---------------------------------------------------------------------------


class TestScreenPWM:
    """Tests for screen_pwm."""

    def test_basic_screen(self) -> None:
        db = _create_test_db()
        seqs = ["ACGTACGT", "TGCATGCA", "AAAACCCC", "GCGCGCGC", "ATATATAT"]
        response = np.array([1.0, 0.5, 0.8, 0.2, 0.6])
        result = screen_pwm(seqs, response, db, metric="r2")

        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"motif", "score"}
        assert len(result) == 2
        # Should be sorted descending by score
        assert result["score"].iloc[0] >= result["score"].iloc[1]

    def test_only_best(self) -> None:
        db = _create_test_db()
        seqs = ["ACGTACGT", "TGCATGCA", "AAAACCCC", "GCGCGCGC", "ATATATAT"]
        response = np.array([1.0, 0.5, 0.8, 0.2, 0.6])
        result = screen_pwm(seqs, response, db, metric="r2", only_best=True)
        assert len(result) == 1

    def test_binary_response_auto_metric(self) -> None:
        db = _create_test_db()
        seqs = ["ACGTACGT", "TGCATGCA", "AAAACCCC", "GCGCGCGC", "ATATATAT"]
        response = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        # Should auto-detect KS metric
        result = screen_pwm(seqs, response, db)
        assert len(result) == 2
        assert all(result["score"] >= 0)

    def test_mismatched_lengths(self) -> None:
        db = _create_test_db()
        seqs = ["ACGT"]
        response = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="do not match"):
            screen_pwm(seqs, response, db)

    def test_ks_on_continuous_raises(self) -> None:
        db = _create_test_db()
        seqs = ["ACGTACGT", "TGCATGCA", "AAAACCCC"]
        response = np.array([0.1, 0.5, 0.9])
        with pytest.raises(ValueError, match="cannot be 'ks'"):
            screen_pwm(seqs, response, db, metric="ks")

    def test_from_dataframe(self) -> None:
        df = _make_test_motif_df()
        seqs = ["ACGTACGT", "TGCATGCA", "AAAACCCC", "GCGCGCGC", "ATATATAT"]
        response = np.array([1.0, 0.5, 0.8, 0.2, 0.6])
        result = screen_pwm(seqs, response, df, metric="r2")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# motif_enrichment
# ---------------------------------------------------------------------------


class TestMotifEnrichment:
    """Tests for motif_enrichment."""

    def test_relative_enrichment(self) -> None:
        rng = np.random.default_rng(42)
        pwm_q = rng.random((20, 3))
        groups = np.array(["A"] * 10 + ["B"] * 10)
        result = motif_enrichment(pwm_q, groups, threshold=0.5, type="relative")
        assert result.shape == (2, 3)
        assert list(result.index) == ["A", "B"]

    def test_absolute_enrichment(self) -> None:
        rng = np.random.default_rng(42)
        pwm_q = rng.random((20, 3))
        groups = np.array(["A"] * 10 + ["B"] * 10)
        result = motif_enrichment(pwm_q, groups, threshold=0.5, type="absolute")
        assert result.shape == (2, 3)

    def test_mismatched_sizes(self) -> None:
        pwm_q = np.random.random((10, 3))
        groups = np.array(["A"] * 5)
        with pytest.raises(ValueError, match="must match"):
            motif_enrichment(pwm_q, groups)

    def test_invalid_type(self) -> None:
        pwm_q = np.random.random((10, 3))
        groups = np.array(["A"] * 10)
        with pytest.raises(ValueError, match="must be"):
            motif_enrichment(pwm_q, groups, type="invalid")


# ---------------------------------------------------------------------------
# _is_binary helper
# ---------------------------------------------------------------------------


class TestIsBinary:
    """Tests for _is_binary."""

    def test_binary(self) -> None:
        assert _is_binary(np.array([0.0, 1.0, 0.0, 1.0]))

    def test_continuous(self) -> None:
        assert not _is_binary(np.array([0.0, 0.5, 1.0]))

    def test_all_zeros(self) -> None:
        assert _is_binary(np.array([0.0, 0.0, 0.0]))

    def test_with_nan(self) -> None:
        assert _is_binary(np.array([0.0, 1.0, np.nan]))


# ---------------------------------------------------------------------------
# R-compatible matrix values
# ---------------------------------------------------------------------------


class TestRCompatibility:
    """Tests verifying exact values match the R implementation."""

    def test_forward_matrix_values(self) -> None:
        """Spot-check forward matrix values against R output."""
        db = _create_test_db()
        # R: mat[1, "motif1"] (A_1 for motif1) = -0.381711
        # In Python, row 0 (A at pos 1), col 0 (motif1)
        np.testing.assert_allclose(db.mat[0, 0], -0.381711, atol=1e-4)
        # R: mat[1, "motif2"] (A_1 for motif2) = -2.246496
        np.testing.assert_allclose(db.mat[0, 1], -2.246496, atol=1e-4)

    def test_rc_matrix_values(self) -> None:
        """Spot-check RC matrix values against R output."""
        db = _create_test_db()
        # R: rc_mat[1, "motif1"] (A_1 for motif1 RC) = -1.599868
        np.testing.assert_allclose(db.rc_mat[0, 0], -1.599868, atol=1e-4)
