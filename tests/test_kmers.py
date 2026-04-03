"""Tests for pyprego k-mer functions.

Mirrors tests/testthat/test-kmer-regression.R and test-screen_kmers.R
from the R prego package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.kmers import (
    generate_kmers,
    kmer_matrix,
    kmers_to_pssm,
    pssm_to_kmer,
    screen_kmers,
)


# ---------------------------------------------------------------------------
# generate_kmers
# ---------------------------------------------------------------------------


class TestGenerateKmers:
    def test_length2_no_gaps(self):
        """4^2 = 16 k-mers, all of length 2, no Ns."""
        result = generate_kmers(2)
        assert len(result) == 4**2
        assert all(len(km) == 2 for km in result)
        assert not any("N" in km for km in result)

    def test_length3_single_gap(self):
        """min_gap=1, max_gap=1: all k-mers have exactly 1 N."""
        result = generate_kmers(3, min_gap=1, max_gap=1)
        assert all(len(km) == 3 for km in result)
        assert all(km.count("N") == 1 for km in result)

    def test_length3_gap_1_to_2(self):
        """min_gap=1, max_gap=2: every k-mer has 1 or 2 Ns."""
        result = generate_kmers(3, min_gap=1, max_gap=2)
        assert all(len(km) == 3 for km in result)
        assert all(1 <= km.count("N") <= 2 for km in result)

    def test_length3_gap_2_only(self):
        """min_gap=2, max_gap=2: all k-mers have exactly 2 Ns."""
        result = generate_kmers(3, min_gap=2, max_gap=2)
        assert all(len(km) == 3 for km in result)
        assert all(km.count("N") == 2 for km in result)

    def test_no_gaps_includes_all(self):
        """max_gap=0 returns all 4^k k-mers without duplicates."""
        result = generate_kmers(3)
        assert len(result) == 4**3
        assert len(set(result)) == 4**3

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            generate_kmers(0)

    def test_max_gap_exceeds_k(self):
        with pytest.raises(ValueError):
            generate_kmers(2, max_gap=3)

    def test_max_gap_with_no_gaps_includes_base(self):
        """max_gap=1 with min_gap=0 should include base k-mers plus gapped ones."""
        result = generate_kmers(2, max_gap=1)
        # Should contain all pure 2-mers
        pure = generate_kmers(2)
        for km in pure:
            assert km in result
        # Should also contain gapped 2-mers like "NA", "AN", etc.
        assert any("N" in km for km in result)


# ---------------------------------------------------------------------------
# kmer_matrix
# ---------------------------------------------------------------------------


class TestKmerMatrix:
    def test_dimensions(self):
        """ATCG with k=2 has 3 distinct 2-mers: AT, TC, CG."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 2)
        assert res.shape[0] == 2
        # Columns include all 4^2=16 k-mers, but only 3 have non-zero counts
        assert res.shape[1] == 16  # all possible 2-mers
        assert (res.sum(axis=0) > 0).sum() == 3  # only 3 are nonzero

    def test_counts_no_gaps(self):
        """Verify specific count values."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 2)
        assert res.loc[0, "AT"] == 1
        assert res.loc[1, "CG"] == 1
        assert res.loc[0, "TC"] == 1

    def test_repeated_kmer_count(self):
        """AAAA should have 3 occurrences of AA."""
        sequences = ["AAAA"]
        res = kmer_matrix(sequences, 2)
        assert res.loc[0, "AA"] == 3

    def test_with_max_gap_1(self):
        """Gapped k-mers with max_gap=1."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 4, max_gap=1)
        # Pure k-mer
        assert res.loc[0, "ATCG"] == 1
        # Gapped k-mers: NTCG, ANCG, ATNG, ATCN
        assert res.loc[1, "NTCG"] == 1
        assert res.loc[1, "ANCG"] == 1
        assert res.loc[1, "ATNG"] == 1
        assert res.loc[1, "ATCN"] == 1

    def test_with_max_gap_2(self):
        """Gapped k-mers with max_gap=2."""
        sequences = ["ATCG", "ATCG"]
        res = kmer_matrix(sequences, 4, max_gap=2)
        assert res.loc[0, "ATCG"] == 1
        # 2-gap k-mers
        assert res.loc[1, "NNCG"] == 1
        assert res.loc[1, "ATNN"] == 1
        assert res.loc[1, "ANNG"] == 1

    def test_explicit_kmer_list(self):
        """Pass explicit k-mer list instead of integer."""
        sequences = ["ATCG", "GGGG"]
        res = kmer_matrix(sequences, ["AT", "GG"])
        assert res.loc[0, "AT"] == 1
        assert res.loc[0, "GG"] == 0
        assert res.loc[1, "AT"] == 0
        assert res.loc[1, "GG"] == 3

    def test_case_insensitive(self):
        """Sequences should be uppercased internally."""
        sequences = ["atcg"]
        res = kmer_matrix(sequences, 2)
        assert res.loc[0, "AT"] == 1


# ---------------------------------------------------------------------------
# screen_kmers
# ---------------------------------------------------------------------------


def _random_nuc(rng, n, length):
    """Generate n random nucleotide sequences of given length."""
    nucs = np.array(list("ACGT"))
    return ["".join(nucs[rng.integers(0, 4, size=length)]) for _ in range(n)]


def _get_pat_seq(rng, n, pat, flank_len=50):
    """Generate n sequences with a pattern embedded in the middle."""
    left = _random_nuc(rng, n, flank_len)
    right = _random_nuc(rng, n, flank_len)
    return [l + pat + r for l, r in zip(left, right)]


class TestScreenKmers:
    def test_1d_detects_pattern(self):
        """screen_kmers should detect a planted 7-mer pattern."""
        rng = np.random.default_rng(60427)
        n = 1000
        pat_seqs = _get_pat_seq(rng, n // 2, "GATAAGA", 50)
        no_pat_seqs = _random_nuc(rng, n // 2, len(pat_seqs[0]))
        seqs = pat_seqs + no_pat_seqs
        resp = np.array([1.0] * (n // 2) + [0.0] * (n // 2))

        res = screen_kmers(seqs, resp, kmer_len=7, seed=60427)
        # Find the row for GATAAGA
        gataaga = res[res["kmer"] == "GATAAGA"]
        assert len(gataaga) == 1
        assert gataaga.iloc[0]["max_r2"] > 0.25  # r^2 > 0.25 => r > 0.5

    def test_output_columns(self):
        """Check that output has correct column structure."""
        rng = np.random.default_rng(42)
        seqs = _random_nuc(rng, 50, 20)
        resp = rng.random(50)

        res = screen_kmers(seqs, resp, kmer_len=3, seed=42)
        assert "kmer" in res.columns
        assert "max_r2" in res.columns
        assert "avg_n" in res.columns
        assert "avg_var" in res.columns

    def test_2d_response(self):
        """screen_kmers with a matrix response (2 columns)."""
        rng = np.random.default_rng(60427)
        n = 400
        pat_seqs = _get_pat_seq(rng, n // 2, "GATAAGA", 50)
        no_pat_seqs = _random_nuc(rng, n // 2, len(pat_seqs[0]))
        seqs = pat_seqs + no_pat_seqs
        resp1 = np.array([1.0] * (n // 2) + [0.0] * (n // 2))
        resp2 = np.array([0.0] * (n // 2) + [1.0] * (n // 2))
        resp = np.column_stack([resp1, resp2])

        res = screen_kmers(seqs, resp, kmer_len=7, seed=60427)
        # Should have 2 response correlation columns (r0, r1)
        assert "r0" in res.columns
        assert "r1" in res.columns

    def test_sorted_by_max_r2(self):
        """Results should be sorted by max_r2 descending."""
        rng = np.random.default_rng(42)
        seqs = _random_nuc(rng, 100, 30)
        resp = rng.random(100)
        res = screen_kmers(seqs, resp, kmer_len=3, seed=42)
        if len(res) > 1:
            assert all(
                res.iloc[i]["max_r2"] >= res.iloc[i + 1]["max_r2"]
                for i in range(len(res) - 1)
            )

    def test_min_cor_filter(self):
        """min_cor should filter out low-correlation k-mers."""
        rng = np.random.default_rng(42)
        seqs = _random_nuc(rng, 100, 30)
        resp = rng.random(100)

        res_all = screen_kmers(seqs, resp, kmer_len=3, seed=42, min_cor=0.0)
        res_filt = screen_kmers(seqs, resp, kmer_len=3, seed=42, min_cor=0.3)
        assert len(res_filt) <= len(res_all)
        if len(res_filt) > 0:
            assert all(res_filt["max_r2"] >= 0.3**2)


# ---------------------------------------------------------------------------
# kmers_to_pssm
# ---------------------------------------------------------------------------


class TestKmersToPssm:
    def test_single_kmer(self):
        """Single k-mer ACGT should produce 4 rows, each summing to 1."""
        result = kmers_to_pssm("ACGT", prior=0.01)
        assert result.shape[0] == 4
        assert set(result.columns) == {"kmer", "pos", "A", "C", "G", "T"}
        # Each row sums to ~1
        for _, row in result.iterrows():
            s = row["A"] + row["C"] + row["G"] + row["T"]
            assert abs(s - 1.0) < 1e-10

    def test_multiple_kmers(self):
        """Two k-mers should produce 8 rows total."""
        result = kmers_to_pssm(["ACGT", "TGCA"], prior=0.01)
        assert result.shape[0] == 8
        # Each position sums to 1 per k-mer
        for _, row in result.iterrows():
            s = row["A"] + row["C"] + row["G"] + row["T"]
            assert abs(s - 1.0) < 1e-10

    def test_n_wildcard(self):
        """N positions should have uniform 0.25 distribution."""
        result = kmers_to_pssm("ACGN", prior=0.01)
        n_row = result[result["pos"] == 4].iloc[0]
        assert abs(n_row["A"] - 0.25) < 1e-10
        assert abs(n_row["C"] - 0.25) < 1e-10
        assert abs(n_row["G"] - 0.25) < 1e-10
        assert abs(n_row["T"] - 0.25) < 1e-10

    def test_dominant_nucleotide(self):
        """At a non-N position, the matching nucleotide should dominate."""
        result = kmers_to_pssm("A", prior=0.01)
        row = result.iloc[0]
        # A should be ~1/(1+3*0.01) ~ 0.9709
        assert row["A"] > 0.95
        assert row["C"] < 0.05

    def test_invalid_character(self):
        with pytest.raises(ValueError, match="valid nucleotides"):
            kmers_to_pssm("AXG")


# ---------------------------------------------------------------------------
# pssm_to_kmer
# ---------------------------------------------------------------------------


class TestPssmToKmer:
    def test_roundtrip_simple(self):
        """A strong PSSM for ACGT should convert back to ACGT (without threshold)."""
        mat = np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.97, 0.01, 0.01],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
        pssm = pyprego.pssm_dataframe(mat)
        result = pssm_to_kmer(pssm, pos_bits_thresh=None)
        assert result == "ACGT"

    def test_with_threshold_all_pass(self):
        """Strong PSSM should still yield ACGT with threshold."""
        mat = np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.97, 0.01, 0.01],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
        pssm = pyprego.pssm_dataframe(mat)
        result = pssm_to_kmer(pssm, pos_bits_thresh=0.5)
        assert result == "ACGT"

    def test_uniform_becomes_n(self):
        """A uniform position should become N when threshold is set."""
        mat = np.array(
            [
                [0.97, 0.01, 0.01, 0.01],
                [0.25, 0.25, 0.25, 0.25],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
        pssm = pyprego.pssm_dataframe(mat)
        result = pssm_to_kmer(pssm, pos_bits_thresh=0.5)
        assert result[0] == "A"
        assert result[1] == "N"
        assert result[2] == "G"
        assert result[3] == "T"

    def test_kmer_length_selects_best_window(self):
        """When kmer_length < PSSM length, the most informative window is selected."""
        # 6-position PSSM: first 2 uniform, last 4 strong
        mat = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.97, 0.01, 0.01, 0.01],
                [0.01, 0.97, 0.01, 0.01],
                [0.01, 0.01, 0.97, 0.01],
                [0.01, 0.01, 0.01, 0.97],
            ]
        )
        pssm = pyprego.pssm_dataframe(mat)
        result = pssm_to_kmer(pssm, kmer_length=4, pos_bits_thresh=None)
        assert result == "ACGT"

    def test_too_short_pssm(self):
        mat = np.array([[0.97, 0.01, 0.01, 0.01]])
        pssm = pyprego.pssm_dataframe(mat)
        with pytest.raises(ValueError, match="kmer_length"):
            pssm_to_kmer(pssm, kmer_length=3)
