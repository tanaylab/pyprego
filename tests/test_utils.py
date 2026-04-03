"""Tests for pyprego utility functions (utils.py).

Covers: rc, calc_sequences_dinucs, calc_sequences_dinuc_dist,
calc_sequences_trinuc_dist, sample_quantile_matched_rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego


# ---------------------------------------------------------------------------
# rc
# ---------------------------------------------------------------------------


class TestRc:
    def test_simple(self):
        assert pyprego.rc("ATCG") == "CGAT"

    def test_palindrome(self):
        assert pyprego.rc("ACGT") == "ACGT"

    def test_all_same(self):
        assert pyprego.rc("AAACCC") == "GGGTTT"

    def test_lowercase_preserved(self):
        assert pyprego.rc("acgt") == "acgt"

    def test_empty_string(self):
        assert pyprego.rc("") == ""

    def test_single_base(self):
        assert pyprego.rc("A") == "T"
        assert pyprego.rc("C") == "G"

    def test_N_preserved(self):
        assert pyprego.rc("ANG") == "CNT"

    def test_not_string_raises(self):
        with pytest.raises(TypeError):
            pyprego.rc(123)  # type: ignore[arg-type]

    def test_invalid_char_raises(self):
        with pytest.raises(ValueError):
            pyprego.rc("ATXG")

    def test_long_sequence(self):
        """Matches R test: rc of 'ATCG' repeated 1000 times."""
        long_seq = "ATCG" * 1000
        expected = "CGAT" * 1000
        assert pyprego.rc(long_seq) == expected


# ---------------------------------------------------------------------------
# calc_sequences_dinucs
# ---------------------------------------------------------------------------


class TestCalcSequencesDinucs:
    def test_basic_counts(self):
        sequences = ["ATCG", "GCTA", "AATT"]
        result = pyprego.calc_sequences_dinucs(sequences)

        assert result.shape == (3, 16)

        # ATCG -> AT, TC, CG
        expected_0 = [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        np.testing.assert_array_equal(result[0], expected_0)

        # GCTA -> GC, CT, TA
        expected_1 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
        np.testing.assert_array_equal(result[1], expected_1)

        # AATT -> AA, AT, TT
        expected_2 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        np.testing.assert_array_equal(result[2], expected_2)

    def test_long_sequence(self):
        long_seq = "ATCG" * 1000
        result = pyprego.calc_sequences_dinucs([long_seq])
        assert result.shape == (1, 16)
        # Total dinucleotides = len(seq) - 1 = 3999
        assert result.sum() == 3999

    def test_identical_sequences(self):
        seqs = ["ATCG"] * 5
        result = pyprego.calc_sequences_dinucs(seqs)
        assert result.shape == (5, 16)
        # All rows should be identical
        for i in range(1, 5):
            np.testing.assert_array_equal(result[i], result[0])

    def test_case_insensitive(self):
        r1 = pyprego.calc_sequences_dinucs(["atcg"])
        r2 = pyprego.calc_sequences_dinucs(["ATCG"])
        np.testing.assert_array_equal(r1, r2)

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError):
            pyprego.calc_sequences_dinucs(["A"])


# ---------------------------------------------------------------------------
# calc_sequences_dinuc_dist
# ---------------------------------------------------------------------------


class TestCalcSequencesDinucDist:
    def test_output_is_dataframe(self):
        sequences = ["AACGT", "CGTAA", "GGCCA"]
        result = pyprego.calc_sequences_dinuc_dist(sequences, size=5)
        assert isinstance(result, pd.DataFrame)

    def test_correct_num_rows_auto_size(self):
        sequences = ["AACGT", "CGTAA", "GGCCA"]
        result = pyprego.calc_sequences_dinuc_dist(sequences)
        assert len(result) == 5

    def test_last_position_is_nan(self):
        """Last position can't form a dinucleotide; should be NaN."""
        sequences = ["AA", "CC", "GG", "TT", "AC"]
        result = pyprego.calc_sequences_dinuc_dist(sequences, size=2)
        assert result.iloc[1, 1:].isna().all()

    def test_correct_frequencies(self):
        sequences = ["AA", "CC", "GG", "TT", "AC"]
        result = pyprego.calc_sequences_dinuc_dist(sequences, size=2)
        # At position 1: 1 each of AA, CC, GG, TT, AC => 0.2 each
        assert abs(result["AA"].iloc[0] - 0.2) < 1e-10
        assert abs(result["CC"].iloc[0] - 0.2) < 1e-10
        assert abs(result["GG"].iloc[0] - 0.2) < 1e-10
        assert abs(result["TT"].iloc[0] - 0.2) < 1e-10
        assert abs(result["AC"].iloc[0] - 0.2) < 1e-10

    def test_short_sequence_raises(self):
        with pytest.raises(ValueError):
            pyprego.calc_sequences_dinuc_dist(["AACGT", "CGTAA", "GG"], size=5)

    def test_non_character_raises(self):
        with pytest.raises(TypeError):
            pyprego.calc_sequences_dinuc_dist([1, 2, 3], size=5)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# calc_sequences_trinuc_dist
# ---------------------------------------------------------------------------


class TestCalcSequencesTrinucDist:
    def test_output_is_dataframe(self):
        sequences = ["AACGTT", "CGTAAG", "GGCCAA"]
        result = pyprego.calc_sequences_trinuc_dist(sequences, size=6)
        assert isinstance(result, pd.DataFrame)

    def test_correct_num_rows_auto_size(self):
        sequences = ["AACGTT", "CGTAAG", "GGCCAA"]
        result = pyprego.calc_sequences_trinuc_dist(sequences)
        assert len(result) == 6

    def test_last_two_positions_nan(self):
        sequences = ["AACGTT", "CGTAAG", "GGCCAA"]
        result = pyprego.calc_sequences_trinuc_dist(sequences, size=6)
        # Last 2 rows (index 4,5) should be NaN (except pos column)
        assert result.iloc[4, 1:].isna().all()
        assert result.iloc[5, 1:].isna().all()

    def test_correct_frequencies(self):
        """Matches R test with known sequences."""
        sequences = ["AAACCC", "GGGAAA", "TTTGGG", "CCCAAA", "AAAGGG"]
        result = pyprego.calc_sequences_trinuc_dist(sequences, size=6)

        # Position 1: AAA=2/5=0.4, GGG=1/5=0.2, TTT=1/5=0.2, CCC=1/5=0.2
        assert abs(result["AAA"].iloc[0] - 0.4) < 1e-10
        assert abs(result["GGG"].iloc[0] - 0.2) < 1e-10
        assert abs(result["TTT"].iloc[0] - 0.2) < 1e-10
        assert abs(result["CCC"].iloc[0] - 0.2) < 1e-10

        # Position 2: AAC=0.2, GGA=0.2, TTG=0.2, CCA=0.2, AAG=0.2
        assert abs(result["AAC"].iloc[1] - 0.2) < 1e-10
        assert abs(result["GGA"].iloc[1] - 0.2) < 1e-10
        assert abs(result["TTG"].iloc[1] - 0.2) < 1e-10
        assert abs(result["CCA"].iloc[1] - 0.2) < 1e-10
        assert abs(result["AAG"].iloc[1] - 0.2) < 1e-10

    def test_short_sequences_edge(self):
        """Sequences exactly 3 nucleotides long."""
        short = ["AAA", "CCC", "GGG", "TTT"]
        result = pyprego.calc_sequences_trinuc_dist(short, size=3)
        assert len(result) == 3
        assert result.iloc[2, 1:].isna().all()

    def test_single_sequence(self):
        result = pyprego.calc_sequences_trinuc_dist(["AAAAAA"], size=6)
        assert result["AAA"].iloc[0] == 1.0
        assert all(result["AAA"].iloc[1:4] == 1.0)
        assert result.iloc[4, 1:].isna().all()
        assert result.iloc[5, 1:].isna().all()


# ---------------------------------------------------------------------------
# sample_quantile_matched_rows
# ---------------------------------------------------------------------------


class TestSampleQuantileMatchedRows:
    def test_basic_sampling(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 1, 1000), "y": rng.normal(0, 1, 1000)})
        result = pyprego.sample_quantile_matched_rows(df, df["x"], sample_fraction=0.1)
        assert len(result) > 0
        assert len(result) <= 100  # 10% of 1000

    def test_preserves_columns(self):
        df = pd.DataFrame({"a": range(100), "b": range(100, 200)})
        ref = np.arange(100, dtype=float)
        result = pyprego.sample_quantile_matched_rows(df, ref, sample_fraction=0.5)
        assert list(result.columns) == ["a", "b"]

    def test_fraction_too_small_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            pyprego.sample_quantile_matched_rows(df, np.array([1, 2, 3]), sample_fraction=0.001)

    def test_quantile_distribution_preserved(self):
        """Sampled quantiles should roughly match original quantiles."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 10000)
        df = pd.DataFrame({"val": values})
        result = pyprego.sample_quantile_matched_rows(df, values, sample_fraction=0.1)

        # Compare quartiles
        orig_q = np.quantile(values, [0.25, 0.5, 0.75])
        samp_q = np.quantile(result["val"].values, [0.25, 0.5, 0.75])
        np.testing.assert_allclose(orig_q, samp_q, atol=0.3)

    def test_reproducible_with_seed(self):
        df = pd.DataFrame({"x": range(1000)})
        ref = np.arange(1000, dtype=float)
        r1 = pyprego.sample_quantile_matched_rows(df, ref, sample_fraction=0.1, seed=42)
        r2 = pyprego.sample_quantile_matched_rows(df, ref, sample_fraction=0.1, seed=42)
        pd.testing.assert_frame_equal(r1, r2)
