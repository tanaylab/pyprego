"""Ported from R prego tests/testthat/test-dinucs.R

Tests for dinucleotide/trinucleotide counting and distribution functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.utils import dinuc_names


# ---------------------------------------------------------------------------
# calc_sequences_dinucs
# ---------------------------------------------------------------------------


class TestCalcSequencesDinucs:
    """calc_sequences_dinucs works correctly (from R test_that block)."""

    def test_basic_output(self):
        sequences = ["ATCG", "GCTA", "AATT"]
        result = pyprego.calc_sequences_dinucs(sequences)

        # Should be a matrix (2D array)
        assert result.ndim == 2
        assert result.shape == (3, 16)

    def test_column_names(self):
        expected_colnames = [
            "AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT",
            "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT",
        ]
        assert dinuc_names() == expected_colnames

    def test_specific_counts(self):
        sequences = ["ATCG", "GCTA", "AATT"]
        result = pyprego.calc_sequences_dinucs(sequences)

        # ATCG: dinucs are AT, TC, CG => AT=1, TC=1, CG=1
        # Row 0: expect [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        expected_row0 = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        np.testing.assert_array_equal(result[0], expected_row0)

        # GCTA: dinucs are GC, CT, TA
        expected_row1 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
        np.testing.assert_array_equal(result[1], expected_row1)

        # AATT: dinucs are AA, AT, TT
        expected_row2 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        np.testing.assert_array_equal(result[2], expected_row2)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            pyprego.calc_sequences_dinucs([""])

    def test_long_sequence(self):
        long_seq = "ATCG" * 1000
        result = pyprego.calc_sequences_dinucs([long_seq])
        assert result.shape == (1, 16)
        # Total dinucleotides in sequence of length 4000 = 3999
        assert result.sum() == 3999

    def test_identical_sequences(self):
        identical_seqs = ["ATCG"] * 5
        result = pyprego.calc_sequences_dinucs(identical_seqs)
        assert result.shape == (5, 16)
        # All rows should be identical
        for i in range(1, 5):
            np.testing.assert_array_equal(result[i], result[0])

    def test_case_insensitive(self):
        lower = pyprego.calc_sequences_dinucs(["atcg"])
        upper = pyprego.calc_sequences_dinucs(["ATCG"])
        np.testing.assert_array_equal(lower, upper)

    def test_many_sequences(self):
        rng = np.random.default_rng(42)
        nucs = list("ACGT")
        many_seqs = [
            "".join(rng.choice(nucs, size=10)) for _ in range(10000)
        ]
        result = pyprego.calc_sequences_dinucs(many_seqs)
        assert result.shape == (10000, 16)


# ---------------------------------------------------------------------------
# calc_sequences_dinuc_dist
# ---------------------------------------------------------------------------


class TestCalcSequencesDinucDist:
    """calc_sequences_dinuc_dist tests (from R test_that blocks)."""

    def test_valid_input_no_error(self):
        sequences = ["AACGT", "CGTAA", "GGCCA"]
        result = pyprego.calc_sequences_dinuc_dist(sequences, size=5)
        assert isinstance(result, pd.DataFrame)

    def test_non_character_input_raises(self):
        # R: expect_error(calc_sequences_dinuc_dist(list(...), size=5))
        # In Python, passing a list of non-strings should raise
        with pytest.raises(TypeError):
            pyprego.calc_sequences_dinuc_dist([1, 2, 3], size=5)  # type: ignore[list-item]

    def test_null_size_uses_max(self):
        sequences = ["AACGT", "CGTAA", "GGCCA"]
        result = pyprego.calc_sequences_dinuc_dist(sequences, size=None)
        assert len(result) == 5  # nrow(res) == 5

    def test_sequences_shorter_than_size_raises(self):
        sequences = ["AACGT", "CGTAA", "GG"]
        with pytest.raises(ValueError):
            pyprego.calc_sequences_dinuc_dist(sequences, size=5)

    def test_correct_distribution(self):
        sequences = ["AA", "CC", "GG", "TT", "AC"]
        result = pyprego.calc_sequences_dinuc_dist(sequences, size=2)

        # At position 1 (row 0), each dinucleotide appears once out of 5 = 0.2
        assert result["AA"].iloc[0] == pytest.approx(0.2)
        assert result["CC"].iloc[0] == pytest.approx(0.2)
        assert result["GG"].iloc[0] == pytest.approx(0.2)
        assert result["TT"].iloc[0] == pytest.approx(0.2)
        assert result["AC"].iloc[0] == pytest.approx(0.2)

        # At position 2 (row 1), last position should be NaN
        assert np.isnan(result["AA"].iloc[1])
        assert np.isnan(result["CC"].iloc[1])
        assert np.isnan(result["GG"].iloc[1])
        assert np.isnan(result["TT"].iloc[1])
        assert np.isnan(result["AC"].iloc[1])


# ---------------------------------------------------------------------------
# calc_sequences_trinuc_dist
# ---------------------------------------------------------------------------


class TestCalcSequencesTrinucDist:
    """calc_sequences_trinuc_dist tests (from R test_that blocks)."""

    def test_valid_input_no_error(self):
        sequences = ["AACGTT", "CGTAAG", "GGCCAA"]
        result = pyprego.calc_sequences_trinuc_dist(sequences, size=6)
        assert isinstance(result, pd.DataFrame)

    def test_non_character_input_raises(self):
        with pytest.raises(TypeError):
            pyprego.calc_sequences_trinuc_dist([1, 2, 3], size=6)  # type: ignore[list-item]

    def test_null_size_uses_max(self):
        sequences = ["AACGTT", "CGTAAG", "GGCCAA"]
        result = pyprego.calc_sequences_trinuc_dist(sequences, size=None)
        assert len(result) == 6

    def test_sequences_shorter_than_size_raises(self):
        sequences = ["AACGTT", "CGTAAG", "GG"]
        with pytest.raises(ValueError):
            pyprego.calc_sequences_trinuc_dist(sequences, size=6)

    def test_correct_distribution(self):
        sequences = ["AAACCC", "GGGAAA", "TTTGGG", "CCCAAA", "AAAGGG"]
        result = pyprego.calc_sequences_trinuc_dist(sequences, size=6)

        # Position 1 (row 0): AAA=2/5=0.4, GGG=1/5=0.2, TTT=1/5=0.2, CCC=1/5=0.2
        assert result["AAA"].iloc[0] == pytest.approx(0.4)
        assert result["GGG"].iloc[0] == pytest.approx(0.2)
        assert result["TTT"].iloc[0] == pytest.approx(0.2)
        assert result["CCC"].iloc[0] == pytest.approx(0.2)

        # Position 2 (row 1): AAC=1/5=0.2, GGA=1/5=0.2, TTG=1/5=0.2, CCA=1/5=0.2, AAG=1/5=0.2
        assert result["AAC"].iloc[1] == pytest.approx(0.2)
        assert result["GGA"].iloc[1] == pytest.approx(0.2)
        assert result["TTG"].iloc[1] == pytest.approx(0.2)
        assert result["CCA"].iloc[1] == pytest.approx(0.2)
        assert result["AAG"].iloc[1] == pytest.approx(0.2)

        # Last two rows (pos 5 and 6, i.e. indices 4 and 5) should be NaN
        # (excluding the 'pos' column)
        trinuc_cols = [c for c in result.columns if c != "pos"]
        for col in trinuc_cols:
            assert np.isnan(result[col].iloc[4])
            assert np.isnan(result[col].iloc[5])

    def test_edge_case_short_sequences(self):
        short_sequences = ["AAA", "CCC", "GGG", "TTT"]
        result = pyprego.calc_sequences_trinuc_dist(short_sequences, size=3)
        assert len(result) == 3
        # Last row (pos 3, index 2) should be NaN
        trinuc_cols = [c for c in result.columns if c != "pos"]
        for col in trinuc_cols:
            assert np.isnan(result[col].iloc[2])

    def test_edge_case_single_sequence(self):
        result = pyprego.calc_sequences_trinuc_dist(["AAAAAA"], size=6)
        # AAA should be 1.0 at positions 1 through 4 (indices 0-3)
        assert result["AAA"].iloc[0] == pytest.approx(1.0)
        assert result["AAA"].iloc[1] == pytest.approx(1.0)
        assert result["AAA"].iloc[2] == pytest.approx(1.0)
        assert result["AAA"].iloc[3] == pytest.approx(1.0)
        # Last two positions should be NaN
        assert np.isnan(result["AAA"].iloc[4])
        assert np.isnan(result["AAA"].iloc[5])
