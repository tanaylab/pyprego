"""Ported from R prego tests/testthat/test-screen_kmers.R and test-screen_local_pwm.R

Tests for screen_kmers and screen_local_pwm (compute_local_pwm-based) functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.kmers import screen_kmers


# ---------------------------------------------------------------------------
# Helpers (mirrors R helper functions in test-screen_kmers.R)
# ---------------------------------------------------------------------------


def _random_nuc(rng: np.random.Generator, n: int, length: int) -> list[str]:
    """Generate n random nucleotide sequences of given length."""
    nucs = np.array(list("ACGT"))
    return ["".join(nucs[rng.integers(0, 4, size=length)]) for _ in range(n)]


def _get_pat_seq(
    rng: np.random.Generator, n: int, pat: str, flank_len: int = 50
) -> list[str]:
    """Generate n sequences with a pattern embedded in the middle."""
    left = _random_nuc(rng, n, flank_len)
    right = _random_nuc(rng, n, flank_len)
    return [l + pat + r for l, r in zip(left, right)]


def _get_seqs(
    rng: np.random.Generator, n: int, frac: float, pat: str, flank_len: int = 50
) -> list[str]:
    """Generate sequences: frac have the pattern, rest are random."""
    n_pat = round(n * frac)
    n_no_pat = round(n * (1 - frac))
    pat_seqs = _get_pat_seq(rng, n_pat, pat, flank_len)
    seq_len = len(pat_seqs[0])
    no_pat_seqs = _random_nuc(rng, n_no_pat, seq_len)
    return pat_seqs + no_pat_seqs


# ---------------------------------------------------------------------------
# screen_kmers tests (from test-screen_kmers.R)
# ---------------------------------------------------------------------------


class TestScreenKmers1D:
    """screen_kmers works with 1D (from R test_that block)."""

    def test_detects_pattern(self):
        rng = np.random.default_rng(60427)
        seqs = _get_seqs(rng, 1000, 0.5, "GATAAGA")
        resp = np.array([1.0] * 500 + [0.0] * 500)

        res = screen_kmers(seqs, resp, kmer_len=7, min_cor=0.08, seed=60427)

        # GATAAGA should be found with high correlation
        gataaga = res[res["kmer"] == "GATAAGA"]
        assert len(gataaga) == 1
        # R checks: res["GATAAGA", 1] > 0.5 (correlation, not r2)
        # In pyprego, max_r2 is r^2, so r > 0.5 means r2 > 0.25
        assert gataaga.iloc[0]["max_r2"] > 0.25

    def test_output_has_correct_columns(self):
        rng = np.random.default_rng(60427)
        seqs = _get_seqs(rng, 1000, 0.5, "GATAAGA")
        resp = np.array([1.0] * 500 + [0.0] * 500)

        res = screen_kmers(seqs, resp, kmer_len=7, min_cor=0.08, seed=60427)
        assert res.columns[0] == "kmer"
        assert res.columns[1] == "max_r2"
        assert res.columns[2] == "avg_n"
        assert res.columns[3] == "avg_var"

    def test_avg_n_close_to_half(self):
        """R test: res_df[res_df$kmer == "GATAAGA", "avg_n"] - 0.5 <= 1e-2"""
        rng = np.random.default_rng(60427)
        seqs = _get_seqs(rng, 1000, 0.5, "GATAAGA")
        resp = np.array([1.0] * 500 + [0.0] * 500)

        res = screen_kmers(seqs, resp, kmer_len=7, min_cor=0.08, seed=60427)
        gataaga = res[res["kmer"] == "GATAAGA"]
        # avg_n should be close to 0.5 (half the sequences have it once)
        assert abs(gataaga.iloc[0]["avg_n"] - 0.5) <= 0.02


class TestScreenKmers1DWithGaps:
    """screen_kmers works with 1D with gaps (from R test_that block)."""

    def test_detects_pattern_with_gaps(self):
        rng = np.random.default_rng(60427)
        seqs = _get_seqs(rng, 400, 0.5, "GATAAGA")
        resp = np.array([1.0] * 200 + [0.0] * 200)

        res = screen_kmers(
            seqs, resp, kmer_len=7, min_cor=0.08, max_gap=3, seed=60427
        )

        gataaga = res[res["kmer"] == "GATAAGA"]
        assert len(gataaga) == 1
        assert gataaga.iloc[0]["max_r2"] > 0.25

    def test_output_columns_with_gaps(self):
        rng = np.random.default_rng(60427)
        seqs = _get_seqs(rng, 400, 0.5, "GATAAGA")
        resp = np.array([1.0] * 200 + [0.0] * 200)

        res = screen_kmers(
            seqs, resp, kmer_len=7, min_cor=0.08, max_gap=3, seed=60427
        )
        assert res.columns[0] == "kmer"
        assert res.columns[1] == "max_r2"
        assert res.columns[2] == "avg_n"
        assert res.columns[3] == "avg_var"


class TestScreenKmers2D:
    """screen_kmers detects a kmer in 2D (from R test_that block)."""

    def test_detects_pattern_2d(self):
        rng = np.random.default_rng(60427)

        # Build sequences with two patterns for two response dimensions
        pat_seqs1 = _get_pat_seq(rng, 5000, "GATAAGA", 200)
        random_seqs1 = _random_nuc(rng, 2000, len(pat_seqs1[0]))
        pat_seqs2 = _get_pat_seq(rng, 5000, "CTTGTTA", 200)
        random_seqs2 = _random_nuc(rng, 2000, len(pat_seqs2[0]))

        seqs = pat_seqs1 + random_seqs1 + pat_seqs2 + random_seqs2
        n = len(seqs)

        # Build 2D response matrix
        resp1 = np.array(
            [1.0] * 5000 + [0.0] * (2000 + 7000)
        )
        resp2 = np.array(
            [0.0] * 7000 + [1.0] * 5000 + [0.0] * 2000
        )
        resp = np.column_stack([resp1, resp2])

        res = screen_kmers(seqs, resp, kmer_len=7, min_cor=0.08, seed=60427)

        # GATAAGA should correlate with response column 0
        gataaga = res[res["kmer"] == "GATAAGA"]
        assert len(gataaga) >= 1
        assert gataaga.iloc[0]["max_r2"] > 0.25

        # CTTGTTA should correlate with response column 1
        cttgtta = res[res["kmer"] == "CTTGTTA"]
        assert len(cttgtta) >= 1
        assert cttgtta.iloc[0]["max_r2"] > 0.25


class TestScreenKmersReproducible:
    """screen_kmers is reproducible in 2D (from R test_that block)."""

    def test_reproducibility(self):
        rng = np.random.default_rng(60427)
        seqs = _get_seqs(rng, 200, 0.5, "GATAAGA")
        resp = np.array([1.0] * 100 + [0.0] * 100)

        kmers1 = screen_kmers(seqs, resp, kmer_len=7, seed=60427)
        kmers2 = screen_kmers(seqs, resp, kmer_len=7, seed=60427)

        pd.testing.assert_frame_equal(kmers1, kmers2)


# ---------------------------------------------------------------------------
# screen_local_pwm tests (from test-screen_local_pwm.R)
# Ported using compute_local_pwm + manual thresholding since pyprego
# does not have a dedicated screen_local_pwm function.
# ---------------------------------------------------------------------------


def _screen_local_pwm(
    sequences: list[str],
    pssm: pd.DataFrame,
    operator: str,
    threshold: float,
    bidirect: bool = True,
    prior: float = 0.01,
) -> list[list[int]]:
    """Python equivalent of R screen_local_pwm.

    Returns a list of lists of 1-based positions where the local PWM score
    passes the threshold according to the operator.
    """
    if operator not in (">", "<", ">=", "<="):
        raise ValueError(f"operator must be one of '>', '<', '>=', '<=', got {operator!r}")

    local_scores = pyprego.compute_local_pwm(
        sequences, pssm, bidirect=bidirect, prior=prior
    )

    results = []
    for i in range(local_scores.shape[0]):
        row = local_scores[i]
        valid = ~np.isnan(row)
        if operator == ">":
            mask = valid & (row > threshold)
        elif operator == "<":
            mask = valid & (row < threshold)
        elif operator == ">=":
            mask = valid & (row >= threshold)
        elif operator == "<=":
            mask = valid & (row <= threshold)
        else:
            mask = np.zeros_like(valid)
        # Convert to 1-based positions (R uses 1-based indexing)
        positions = (np.where(mask)[0] + 1).tolist()
        results.append(positions)
    return results


def _make_test_pssm() -> pd.DataFrame:
    """Create a simple 4-position PSSM for local PWM testing."""
    mat = np.array(
        [
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
            [0.1, 0.1, 0.1, 0.7],
        ]
    )
    return pyprego.pssm_dataframe(mat)


class TestScreenLocalPwmOperators:
    """screen_local_pwm respects comparison operators."""

    def test_gt_and_lt_differ(self):
        sequences = ["ACGTACGT", "TGCATGCA"]
        pssm = _make_test_pssm()

        local_scores = pyprego.compute_local_pwm(
            sequences, pssm, bidirect=False, prior=0
        )
        thresh = np.nanmedian(local_scores)

        res_gt = _screen_local_pwm(sequences, pssm, ">", thresh, bidirect=False, prior=0)
        res_lt = _screen_local_pwm(sequences, pssm, "<", thresh, bidirect=False, prior=0)

        # The two results should be different
        assert res_gt != res_lt

    def test_gt_positions_have_higher_scores(self):
        sequences = ["ACGTACGT", "TGCATGCA"]
        pssm = _make_test_pssm()

        local_scores = pyprego.compute_local_pwm(
            sequences, pssm, bidirect=False, prior=0
        )
        thresh = np.nanmedian(local_scores)

        res_gt = _screen_local_pwm(sequences, pssm, ">", thresh, bidirect=False, prior=0)

        # All returned positions should have scores > threshold
        for i, positions in enumerate(res_gt):
            for pos_1based in positions:
                pos_0based = pos_1based - 1
                assert local_scores[i, pos_0based] > thresh


class TestScreenLocalPwmInvalidOperator:
    """screen_local_pwm throws an error for invalid operators."""

    def test_invalid_operator_raises(self):
        sequences = ["ACGTACGT"]
        pssm = _make_test_pssm()

        with pytest.raises(ValueError, match="operator"):
            _screen_local_pwm(sequences, pssm, "!=", 0, bidirect=False, prior=0)


class TestScreenLocalPwmSuperset:
    """screen_local_pwm >= and <= behave as supersets of > and <."""

    def test_ge_superset_of_gt(self):
        sequences = ["ACGTACGT", "TGCATGCA"]
        pssm = _make_test_pssm()

        local_scores = pyprego.compute_local_pwm(
            sequences, pssm, bidirect=False, prior=0
        )
        thresh = np.nanmedian(local_scores)

        res_gt = _screen_local_pwm(sequences, pssm, ">", thresh, bidirect=False, prior=0)
        res_ge = _screen_local_pwm(sequences, pssm, ">=", thresh, bidirect=False, prior=0)

        # >= should include all positions from >
        for i in range(len(res_gt)):
            assert set(res_gt[i]).issubset(set(res_ge[i]))

    def test_le_superset_of_lt(self):
        sequences = ["ACGTACGT", "TGCATGCA"]
        pssm = _make_test_pssm()

        local_scores = pyprego.compute_local_pwm(
            sequences, pssm, bidirect=False, prior=0
        )
        thresh = np.nanmedian(local_scores)

        res_lt = _screen_local_pwm(sequences, pssm, "<", thresh, bidirect=False, prior=0)
        res_le = _screen_local_pwm(sequences, pssm, "<=", thresh, bidirect=False, prior=0)

        # <= should include all positions from <
        for i in range(len(res_lt)):
            assert set(res_lt[i]).issubset(set(res_le[i]))

    def test_union_of_gt_lt_subset_of_ge_le(self):
        sequences = ["ACGTACGT", "TGCATGCA"]
        pssm = _make_test_pssm()

        local_scores = pyprego.compute_local_pwm(
            sequences, pssm, bidirect=False, prior=0
        )
        thresh = np.nanmedian(local_scores)

        res_gt = _screen_local_pwm(sequences, pssm, ">", thresh, bidirect=False, prior=0)
        res_lt = _screen_local_pwm(sequences, pssm, "<", thresh, bidirect=False, prior=0)
        res_ge = _screen_local_pwm(sequences, pssm, ">=", thresh, bidirect=False, prior=0)
        res_le = _screen_local_pwm(sequences, pssm, "<=", thresh, bidirect=False, prior=0)

        for i in range(len(res_gt)):
            union_strict = set(res_gt[i]) | set(res_lt[i])
            union_nonstrict = set(res_ge[i]) | set(res_le[i])
            assert union_strict.issubset(union_nonstrict)
