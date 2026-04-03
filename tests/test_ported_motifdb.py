"""Ported from R prego tests/testthat/test-MotifDB.R

Tests for MotifDB class creation, validation, subsetting, and conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.motif_db import create_motif_db, motif_db_to_dataframe, set_prior


# ---------------------------------------------------------------------------
# Helper: create_test_motif_db (mirrors the R helper)
# ---------------------------------------------------------------------------


def _create_test_motif_db(
    prior: float = 0.01,
    spat_min: float | None = None,
    spat_max: float | None = None,
) -> pyprego.MotifDB:
    """Create a simple motif database with two motifs, each of length 4."""
    motif_data = pd.DataFrame(
        {
            "motif": ["motif1"] * 4 + ["motif2"] * 4,
            "pos": [1] * 8,
            "A": [0.7, 0.1, 0.2, 0.3, 0.1, 0.6, 0.2, 0.1],
            "C": [0.1, 0.7, 0.3, 0.2, 0.2, 0.1, 0.6, 0.2],
            "G": [0.1, 0.1, 0.3, 0.3, 0.6, 0.2, 0.1, 0.1],
            "T": [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.6],
        }
    )
    return create_motif_db(
        motif_data,
        prior=prior,
        spat_min=spat_min,
        spat_max=spat_max,
    )


# ---------------------------------------------------------------------------
# MotifDB creation and validation
# ---------------------------------------------------------------------------


class TestMotifDBCreation:
    """MotifDB object creation works with valid input."""

    def test_basic_creation(self):
        motif_db = _create_test_motif_db()
        assert isinstance(motif_db, pyprego.MotifDB)
        assert len(motif_db) == 2

    def test_creation_with_spatial_boundaries(self):
        motif_db = _create_test_motif_db(spat_min=0, spat_max=100)
        assert isinstance(motif_db, pyprego.MotifDB)
        assert motif_db.spat_min == 0
        assert motif_db.spat_max == 100


class TestMotifDBSpatialValidation:
    """MotifDB validates spatial boundaries correctly."""

    def test_valid_spatial_boundaries(self):
        # Should not raise
        _create_test_motif_db(spat_min=0, spat_max=100)

    def test_valid_equal_boundaries(self):
        _create_test_motif_db(spat_min=0, spat_max=0)

    def test_invalid_min_greater_than_max(self):
        with pytest.raises(ValueError):
            _create_test_motif_db(spat_min=100, spat_max=0)

    def test_invalid_negative_min(self):
        with pytest.raises(ValueError):
            _create_test_motif_db(spat_min=-1, spat_max=100)

    def test_none_spatial_boundaries(self):
        # None values should be valid
        _create_test_motif_db(spat_min=None, spat_max=None)

    def test_partial_none_spatial_boundaries(self):
        # One None, one set -- should be valid (no min > max check)
        _create_test_motif_db(spat_min=0, spat_max=None)
        _create_test_motif_db(spat_min=None, spat_max=100)


class TestMotifDBMatrixDimensions:
    """MotifDB validates matrix dimensions."""

    def test_single_position_motif(self):
        motif_data = pd.DataFrame(
            {
                "motif": ["motif1"],
                "pos": [1],
                "A": [0.7],
                "C": [0.1],
                "G": [0.1],
                "T": [0.1],
            }
        )
        # Should not raise
        db = create_motif_db(motif_data)
        assert len(db) == 1


class TestMotifDBPriorValidation:
    """MotifDB validates prior constraints."""

    def test_prior_zero_raises(self):
        with pytest.raises(ValueError):
            _create_test_motif_db(prior=0)

    def test_prior_one_raises(self):
        with pytest.raises(ValueError):
            _create_test_motif_db(prior=1)

    def test_prior_negative_raises(self):
        with pytest.raises(ValueError):
            _create_test_motif_db(prior=-0.1)


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------


class TestMotifDBSubsetting:
    """MotifDB subsetting preserves spatial boundaries."""

    def test_character_subsetting(self):
        motif_db = _create_test_motif_db(spat_min=0, spat_max=100)
        subset_db = motif_db["motif1"]
        assert subset_db.spat_min == 0
        assert subset_db.spat_max == 100
        assert len(subset_db) == 1

    def test_numeric_subsetting(self):
        motif_db = _create_test_motif_db(spat_min=0, spat_max=100)
        subset_db = motif_db[0]
        assert subset_db.spat_min == 0
        assert subset_db.spat_max == 100
        assert len(subset_db) == 1

    def test_multiple_motif_selection(self):
        motif_db = _create_test_motif_db(spat_min=0, spat_max=100)
        multi_subset = motif_db[["motif1", "motif2"]]
        assert multi_subset.spat_min == 0
        assert multi_subset.spat_max == 100
        assert len(multi_subset) == 2


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


class TestMotifDBToDataframe:
    """motif_db_to_dataframe conversion works with spatial boundaries."""

    def test_conversion_same_regardless_of_bounds(self):
        motif_db = _create_test_motif_db(spat_min=0, spat_max=100)
        df_with_bounds = motif_db_to_dataframe(motif_db)

        motif_db_no_bounds = _create_test_motif_db()
        df_no_bounds = motif_db_to_dataframe(motif_db_no_bounds)

        pd.testing.assert_frame_equal(
            df_with_bounds,
            df_no_bounds,
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Spatial factors validation
# ---------------------------------------------------------------------------


class TestSpatialFactors:
    """spatial factors validation works."""

    def test_valid_spatial_factors(self):
        motif_db = _create_test_motif_db()
        df = motif_db_to_dataframe(motif_db)

        spat_factors = np.ones((2, 3), dtype=np.float64)
        # Should not raise
        create_motif_db(
            df,
            spat_factors=spat_factors,
            spat_min=0,
            spat_max=100,
        )

    def test_invalid_dimensions_raises(self):
        motif_db = _create_test_motif_db()
        df = motif_db_to_dataframe(motif_db)

        invalid_spat_factors = np.ones((3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            create_motif_db(df, spat_factors=invalid_spat_factors)

    def test_negative_values_raises(self):
        motif_db = _create_test_motif_db()
        df = motif_db_to_dataframe(motif_db)

        spat_factors = np.ones((2, 3), dtype=np.float64)
        spat_factors[0, 0] = -1
        with pytest.raises(ValueError):
            create_motif_db(df, spat_factors=spat_factors)


# ---------------------------------------------------------------------------
# Reverse complement matrix
# ---------------------------------------------------------------------------


class TestReverseComplementMatrix:
    """reverse complement matrix is correctly computed with spatial boundaries."""

    def test_rc_matrix_consistency(self):
        motif_db = _create_test_motif_db(spat_min=0, spat_max=100)

        # For the first motif, check that forward and RC matrices are
        # consistent with the complement mapping.
        # In R:
        #   forward[pos * 4 - (4 - nuc_idx), col] == rc[rc_pos * 4 - (4 - rc_nuc_idx), col]
        # where rc_pos = motif_len - pos + 1 and rc_nuc_idx = complement of nuc_idx
        #
        # In Python, the mat is stored as (D*4, n_motifs) in log-scale.
        # The RC should be: at position rc_pos for complement nucleotide,
        # the log-probability should match the forward at position pos for original nuc.

        complement_map = {0: 3, 1: 2, 2: 1, 3: 0}  # A->T, C->G, G->C, T->A
        motif_len = motif_db.motif_lengths["motif1"]

        for pos in range(1, motif_len + 1):
            for nuc_idx in range(4):
                fwd_row = (pos - 1) * 4 + nuc_idx
                rc_pos = motif_len - pos + 1
                rc_nuc_idx = complement_map[nuc_idx]
                rc_row = (rc_pos - 1) * 4 + rc_nuc_idx

                fwd_val = motif_db.mat[fwd_row, 0]
                rc_val = motif_db.rc_mat[rc_row, 0]
                assert fwd_val == pytest.approx(rc_val, abs=1e-10), (
                    f"RC mismatch at pos={pos}, nuc={nuc_idx}: "
                    f"fwd={fwd_val}, rc={rc_val}"
                )


# ---------------------------------------------------------------------------
# Prior modification
# ---------------------------------------------------------------------------


class TestPriorModification:
    """prior modification preserves spatial boundaries."""

    def test_set_prior_preserves_spatial(self):
        motif_db = _create_test_motif_db(prior=0.01, spat_min=0, spat_max=100)
        new_db = set_prior(motif_db, 0.02)

        assert new_db.spat_min == 0
        assert new_db.spat_max == 100
        assert new_db.prior == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Motif lengths
# ---------------------------------------------------------------------------


class TestMotifLengths:
    """motif_lengths validation works."""

    def test_lengths_match_matrix(self):
        motif_db = _create_test_motif_db()
        # Number of motif_lengths entries should match number of matrix columns
        assert len(motif_db.motif_lengths) == motif_db.mat.shape[1]

    def test_names_match(self):
        motif_db = _create_test_motif_db()
        assert list(motif_db.motif_lengths.keys()) == motif_db.names()

    def test_lengths_are_positive(self):
        motif_db = _create_test_motif_db()
        assert all(v > 0 for v in motif_db.motif_lengths.values())


# ---------------------------------------------------------------------------
# as.data.frame equivalent
# ---------------------------------------------------------------------------


class TestAsDataFrame:
    """as.data.frame works correctly with spatial boundaries."""

    def test_conversion_consistent(self):
        motif_db = _create_test_motif_db(spat_min=0, spat_max=100)
        df1 = motif_db_to_dataframe(motif_db)

        # Compare with non-bounded version
        motif_db_no_bounds = _create_test_motif_db()
        df_no_bounds = motif_db_to_dataframe(motif_db_no_bounds)

        pd.testing.assert_frame_equal(df1, df_no_bounds, atol=1e-10)
