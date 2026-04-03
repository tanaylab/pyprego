"""Ported from R prego tests/testthat/test-kmer-regression.R

Tests for k-mer generation, k-mer matrix, and kmers_to_pssm functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.kmers import generate_kmers, kmer_matrix, kmers_to_pssm


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _substr_count(string: str, substr: str) -> int:
    """Count non-overlapping occurrences of substr in string."""
    count = 0
    start = 0
    while True:
        pos = string.find(substr, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


# ---------------------------------------------------------------------------
# generate_kmers
# ---------------------------------------------------------------------------


class TestGenerateKmers:
    """generate_kmers function works correctly (from R test_that block)."""

    def test_length2_no_gaps(self):
        """Test 1: kmers of length 2 without gaps."""
        result = generate_kmers(2)
        assert len(result) == 4**2
        assert all(len(km) == 2 for km in result)
        assert not any("N" in km for km in result)

    def test_length3_single_gap(self):
        """Test 2: kmers of length 3 with a single gap."""
        result = generate_kmers(3, min_gap=1, max_gap=1)
        assert all(len(km) == 3 for km in result)
        assert all(km.count("N") == 1 for km in result)

    def test_length3_gap_1_to_2(self):
        """Test 3: kmers of length 3 with a gap of 1 to 2 Ns."""
        result = generate_kmers(3, min_gap=1, max_gap=2)
        assert all(len(km) == 3 for km in result)
        assert all(1 <= km.count("N") <= 2 for km in result)

    def test_length3_gap_2_only(self):
        """Test 4: kmers of length 3 with exactly 2 Ns."""
        result = generate_kmers(3, min_gap=2, max_gap=2)
        assert all(len(km) == 3 for km in result)
        assert all(km.count("N") == 2 for km in result)


# ---------------------------------------------------------------------------
# kmer_matrix
# ---------------------------------------------------------------------------


class TestKmerMatrix:
    """kmer_matrix function works correctly (from R test_that block)."""

    def test_dimensions(self):
        """Test 1: Check dimensions of the output matrix."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 2)
        assert res.shape[0] == 2
        # R returns only non-zero columns (3), but Python returns all 16
        # The R test checks dim == c(2, 3), which counts only non-zero k-mers
        # We check that exactly 3 columns are non-zero
        assert (res.sum(axis=0) > 0).sum() == 3

    def test_frequency_no_gaps(self):
        """Test 2: Check correct frequency calculation without gaps."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 2)
        assert res.loc[0, "AT"] == 1
        assert res.loc[1, "CG"] == 1

    def test_from_range_to_range(self):
        """Test 4: from_range and to_range.

        R: kmer_matrix(sequences, kmer_length, from_range=1, to_range=3)
        Slices sequences to positions 1-3 (R 1-based), i.e. 'ATC' (first 3 chars).
        In 'ATC', the 2-mers are AT, TC.

        Python equivalent: slice sequences before passing to kmer_matrix.
        """
        sequences = ["ATCG", "ATCG"]
        # R from_range=1, to_range=3 means substring(seq, 1, 3) = "ATC" (1-indexed, inclusive)
        # Python: seq[0:3] = "ATC"
        sliced = [s[0:3] for s in sequences]
        res = kmer_matrix(sliced, 2)
        assert res.loc[0, "AT"] == 1
        assert res.loc[1, "TC"] == 1

    def test_with_max_gap_1(self):
        """Test 6: frequency calculation with max_gap=1."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 4, max_gap=1)
        assert res.loc[0, "ATCG"] == 1
        assert res.loc[1, "NTCG"] == 1
        assert res.loc[1, "ANCG"] == 1
        assert res.loc[1, "ATNG"] == 1
        assert res.loc[1, "ATCN"] == 1

    def test_with_max_gap_2(self):
        """Test 7: frequency calculation with max_gap=2."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 4, max_gap=2)
        assert res.loc[0, "ATCG"] == 1
        assert res.loc[1, "NNCG"] == 1
        assert res.loc[1, "ATNN"] == 1
        assert res.loc[1, "ANNG"] == 1

    def test_with_max_gap_0(self):
        """Test 8: max_gap=0 is equivalent to no gaps."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 2, max_gap=0)
        assert res.loc[0, "AT"] == 1
        assert res.loc[1, "CG"] == 1


# ---------------------------------------------------------------------------
# kmers_to_pssm
# ---------------------------------------------------------------------------


class TestKmersToPssm:
    """kmers_to_pssm tests (from R test_that blocks)."""

    def test_single_kmer(self):
        """kmers_to_pssm handles single kmer correctly."""
        result = kmers_to_pssm("ACGT", prior=0.01)
        assert result.shape[0] == 4
        assert result.shape[1] == 6  # kmer, pos, A, C, G, T

        # Each position should sum to 1 across A, C, G, T
        for pos_val in [1, 2, 3, 4]:
            row_sum = result[result["pos"] == pos_val][["A", "C", "G", "T"]].sum(axis=1).iloc[0]
            assert row_sum == pytest.approx(1.0)

    def test_multiple_kmers(self):
        """kmers_to_pssm handles multiple kmers correctly."""
        result = kmers_to_pssm(["ACGT", "TGCA"], prior=0.01)
        assert result.shape[0] == 8
        assert result.shape[1] == 6

        # For each position, sum across both kmers should be 2
        for pos_val in [1, 2, 3, 4]:
            pos_rows = result[result["pos"] == pos_val][["A", "C", "G", "T"]]
            total = pos_rows.sum().sum()
            assert total == pytest.approx(2.0)

    def test_n_wildcard(self):
        """kmers_to_pssm handles 'N' correctly."""
        result = kmers_to_pssm("ACGN", prior=0.01)
        assert result.shape[0] == 4
        assert result.shape[1] == 6

        # Each position sums to 1
        for pos_val in [1, 2, 3, 4]:
            row_sum = result[result["pos"] == pos_val][["A", "C", "G", "T"]].sum(axis=1).iloc[0]
            assert row_sum == pytest.approx(1.0)

        # N position (pos=4) should have uniform 0.25
        n_row = result[result["pos"] == 4].iloc[0]
        assert n_row["A"] == pytest.approx(0.25)
        assert n_row["C"] == pytest.approx(0.25)
        assert n_row["G"] == pytest.approx(0.25)
        assert n_row["T"] == pytest.approx(0.25)
