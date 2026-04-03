"""Tests for pyprego PSSM operations (pssm.py).

Covers: pssm_cor, pssm_diff, pssm_match, pssm_concat / concat_pssm,
pssm_trim / trim_pssm, pssm_rc, pssm_add_prior, pssm_theoretical_max,
pssm_theoretical_min, pssm_quantile, consensus_from_pssm, bits_per_pos,
pssm_to_kmer, pssm_dataset_cor, pssm_dataset_diff.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego
from pyprego.types import pssm_dataframe, pssm_to_array


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gata_pssm() -> pd.DataFrame:
    """A GATA-like motif PSSM (8 positions)."""
    mat = np.array([
        [0.05, 0.05, 0.85, 0.05],  # G
        [0.85, 0.05, 0.05, 0.05],  # A
        [0.05, 0.05, 0.05, 0.85],  # T
        [0.85, 0.05, 0.05, 0.05],  # A
        [0.05, 0.05, 0.85, 0.05],  # G
        [0.85, 0.05, 0.05, 0.05],  # A
        [0.05, 0.05, 0.05, 0.85],  # T
        [0.85, 0.05, 0.05, 0.05],  # A
    ])
    return pssm_dataframe(mat)


@pytest.fixture
def short_pssm() -> pd.DataFrame:
    """A short 4-position PSSM."""
    mat = np.array([
        [0.9, 0.03, 0.04, 0.03],
        [0.03, 0.9, 0.04, 0.03],
        [0.03, 0.04, 0.9, 0.03],
        [0.03, 0.03, 0.04, 0.9],
    ])
    return pssm_dataframe(mat)


@pytest.fixture
def uniform_pssm() -> pd.DataFrame:
    """A uniform (no information) PSSM."""
    mat = np.full((6, 4), 0.25)
    return pssm_dataframe(mat)


# ---------------------------------------------------------------------------
# bits_per_pos
# ---------------------------------------------------------------------------


class TestBitsPerPos:
    def test_uniform_has_zero_bits(self, uniform_pssm: pd.DataFrame):
        bits = pyprego.bits_per_pos(uniform_pssm)
        np.testing.assert_allclose(bits, 0.0, atol=1e-6)

    def test_perfect_pssm_has_high_bits(self):
        # Nearly perfect information (one nucleotide dominant)
        mat = np.array([[0.97, 0.01, 0.01, 0.01]] * 4)
        pssm = pssm_dataframe(mat)
        bits = pyprego.bits_per_pos(pssm)
        # With prior of 0.01, it won't be exactly 2 but should be close to 2
        assert np.all(bits > 1.5)

    def test_matches_r_formula(self):
        """bits = log2(4) + sum(p * log2(p)), floored at 0."""
        mat = np.array([
            [0.2, 0.1, 0.4, 0.3],
            [0.1, 0.2, 0.3, 0.4],
        ])
        pssm = pssm_dataframe(mat)
        bits = pyprego.bits_per_pos(pssm, prior=0.01)

        # Manually compute expected
        m = mat + 0.01
        m = m / m.sum(axis=1, keepdims=True)
        expected = np.log2(4) + np.sum(m * np.log2(m), axis=1)
        expected = np.maximum(expected, 0.0)
        np.testing.assert_allclose(bits, expected, atol=1e-10)

    def test_prior_zero(self):
        mat = np.array([[0.25, 0.25, 0.25, 0.25]])
        pssm = pssm_dataframe(mat)
        bits = pyprego.bits_per_pos(pssm, prior=0.0)
        np.testing.assert_allclose(bits, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# consensus_from_pssm
# ---------------------------------------------------------------------------


class TestConsensusFromPssm:
    def test_clear_motif(self, gata_pssm: pd.DataFrame):
        consensus = pyprego.consensus_from_pssm(gata_pssm)
        assert consensus == "GATAGATA"

    def test_uniform_gives_N(self, uniform_pssm: pd.DataFrame):
        consensus = pyprego.consensus_from_pssm(uniform_pssm)
        assert all(c == "N" for c in consensus)

    def test_ambiguity_codes(self):
        # Two nucleotides each at 0.4
        mat = np.array([
            [0.4, 0.0, 0.4, 0.2],  # A+G = 0.8 > 0.75 => R
        ])
        pssm = pssm_dataframe(mat)
        consensus = pyprego.consensus_from_pssm(pssm, single_thresh=0.5, double_thresh=0.75)
        assert consensus == "R"


# ---------------------------------------------------------------------------
# pssm_rc
# ---------------------------------------------------------------------------


class TestPssmRc:
    def test_double_rc_is_identity(self, gata_pssm: pd.DataFrame):
        rc_pssm = pyprego.pssm_rc(gata_pssm)
        rc_rc_pssm = pyprego.pssm_rc(rc_pssm)
        np.testing.assert_allclose(
            pssm_to_array(gata_pssm),
            pssm_to_array(rc_rc_pssm),
            atol=1e-10,
        )

    def test_rc_swaps_AT_CG(self):
        # Simple 2-pos PSSM: [A=1,C=0,G=0,T=0], [A=0,C=1,G=0,T=0]
        mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        pssm = pssm_dataframe(mat)
        rc = pyprego.pssm_rc(pssm)
        # Reversed and complemented:
        # pos0 was C -> becomes G at last pos; pos1 was A -> becomes T at first pos
        expected = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        np.testing.assert_allclose(pssm_to_array(rc), expected)

    def test_rc_has_correct_shape(self, short_pssm: pd.DataFrame):
        rc = pyprego.pssm_rc(short_pssm)
        assert rc.shape == short_pssm.shape
        assert list(rc.columns) == list(short_pssm.columns)


# ---------------------------------------------------------------------------
# pssm_trim / trim_pssm
# ---------------------------------------------------------------------------


class TestPssmTrim:
    def test_trims_uniform_edges(self):
        # Build PSSM with uniform flanking positions and informative center
        uniform = np.full((3, 4), 0.25)
        informative = np.array([[0.9, 0.03, 0.04, 0.03]] * 4)
        mat = np.vstack([uniform, informative, uniform])
        pssm = pssm_dataframe(mat)
        trimmed = pyprego.pssm_trim(pssm, bits_thresh=0.1)
        # Should keep only the 4 informative positions
        assert len(trimmed) == 4

    def test_all_uniform_gives_empty(self, uniform_pssm: pd.DataFrame):
        trimmed = pyprego.pssm_trim(uniform_pssm, bits_thresh=0.1)
        assert len(trimmed) == 0

    def test_alias(self):
        assert pyprego.trim_pssm is pyprego.pssm_trim


# ---------------------------------------------------------------------------
# pssm_add_prior
# ---------------------------------------------------------------------------


class TestPssmAddPrior:
    def test_rows_sum_to_one(self, short_pssm: pd.DataFrame):
        result = pyprego.pssm_add_prior(short_pssm, prior=0.05)
        row_sums = pssm_to_array(result).sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_prior_smooths_distribution(self):
        # Nearly pure A at every position
        mat = np.array([[0.97, 0.01, 0.01, 0.01]] * 3)
        pssm = pssm_dataframe(mat)
        smoothed = pyprego.pssm_add_prior(pssm, prior=0.1)
        arr = pssm_to_array(smoothed)
        # After adding 0.1 to each: [1.07, 0.11, 0.11, 0.11] / 1.4
        # A should still be dominant but less extreme
        assert np.all(arr[:, 0] < 0.97)
        assert np.all(arr[:, 0] > 0.5)


# ---------------------------------------------------------------------------
# pssm_theoretical_max / min / quantile
# ---------------------------------------------------------------------------


class TestTheoreticalScores:
    def test_max_greater_than_min(self, short_pssm: pd.DataFrame):
        mx = pyprego.pssm_theoretical_max(short_pssm)
        mn = pyprego.pssm_theoretical_min(short_pssm)
        assert mx > mn

    def test_quantile_interpolation(self, short_pssm: pd.DataFrame):
        mx = pyprego.pssm_theoretical_max(short_pssm)
        mn = pyprego.pssm_theoretical_min(short_pssm)
        q50 = pyprego.pssm_quantile(short_pssm, 0.5)
        assert abs(q50 - (mn + 0.5 * (mx - mn))) < 1e-10

    def test_quantile_0_is_min(self, short_pssm: pd.DataFrame):
        mn = pyprego.pssm_theoretical_min(short_pssm)
        q0 = pyprego.pssm_quantile(short_pssm, 0.0)
        assert abs(q0 - mn) < 1e-10

    def test_quantile_1_is_max(self, short_pssm: pd.DataFrame):
        mx = pyprego.pssm_theoretical_max(short_pssm)
        q1 = pyprego.pssm_quantile(short_pssm, 1.0)
        assert abs(q1 - mx) < 1e-10

    def test_matches_r_formula(self):
        """sum(log(regularization + rowMax/Min(pssm)))."""
        mat = np.array([
            [0.7, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.7, 0.1],
        ])
        pssm = pssm_dataframe(mat)
        prior = 0.01
        reg = 0.01

        # Normalize
        m = mat + prior
        m = m / m.sum(axis=1, keepdims=True)
        expected_max = np.sum(np.log(reg + np.max(m, axis=1)))
        expected_min = np.sum(np.log(reg + np.min(m, axis=1)))

        assert abs(pyprego.pssm_theoretical_max(pssm, prior, reg) - expected_max) < 1e-10
        assert abs(pyprego.pssm_theoretical_min(pssm, prior, reg) - expected_min) < 1e-10


# ---------------------------------------------------------------------------
# pssm_concat / concat_pssm
# ---------------------------------------------------------------------------


class TestPssmConcat:
    def test_basic_concat(self, short_pssm: pd.DataFrame):
        result = pyprego.pssm_concat(short_pssm, short_pssm)
        assert len(result) == 2 * len(short_pssm)

    def test_concat_with_gap(self, short_pssm: pd.DataFrame):
        result = pyprego.pssm_concat(short_pssm, short_pssm, gap=3)
        assert len(result) == 2 * len(short_pssm) + 3
        # Gap positions should be uniform
        gap_start = len(short_pssm)
        gap_arr = pssm_to_array(result)[gap_start : gap_start + 3]
        np.testing.assert_allclose(gap_arr, 0.25)

    def test_pos_is_reset(self, short_pssm: pd.DataFrame):
        result = pyprego.pssm_concat(short_pssm, short_pssm)
        np.testing.assert_array_equal(result["pos"].values, np.arange(len(result)))

    def test_alias(self):
        assert pyprego.concat_pssm is pyprego.pssm_concat

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            pyprego.pssm_concat()


# ---------------------------------------------------------------------------
# pssm_cor
# ---------------------------------------------------------------------------


class TestPssmCor:
    def test_identical_pssms(self, gata_pssm: pd.DataFrame):
        c = pyprego.pssm_cor(gata_pssm, gata_pssm, method="pearson")
        assert abs(c - 1.0) < 1e-6

    def test_identical_spearman(self, gata_pssm: pd.DataFrame):
        c = pyprego.pssm_cor(gata_pssm, gata_pssm, method="spearman")
        assert abs(c - 1.0) < 1e-6

    def test_different_lengths_still_works(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        """Shorter PSSM is slid along longer one."""
        c = pyprego.pssm_cor(gata_pssm, short_pssm, method="spearman")
        assert isinstance(c, float)
        assert not np.isnan(c)

    def test_empty_raises(self, gata_pssm: pd.DataFrame):
        empty = pssm_dataframe(np.empty((0, 4)))
        with pytest.raises(ValueError):
            pyprego.pssm_cor(gata_pssm, empty)

    def test_symmetric(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        c1 = pyprego.pssm_cor(gata_pssm, short_pssm, method="pearson")
        c2 = pyprego.pssm_cor(short_pssm, gata_pssm, method="pearson")
        assert abs(c1 - c2) < 1e-10

    def test_rc_pssm_correlation(self, gata_pssm: pd.DataFrame):
        """GATA and its RC have a defined correlation value."""
        rc = pyprego.pssm_rc(gata_pssm)
        c = pyprego.pssm_cor(gata_pssm, rc, method="spearman")
        # GATA -> TATCTATC under RC; not palindromic, so moderate correlation is expected
        assert isinstance(c, float)
        assert not np.isnan(c)

    def test_invalid_method(self, gata_pssm: pd.DataFrame):
        with pytest.raises(ValueError):
            pyprego.pssm_cor(gata_pssm, gata_pssm, method="kendall")


# ---------------------------------------------------------------------------
# pssm_diff
# ---------------------------------------------------------------------------


class TestPssmDiff:
    def test_identical_pssms_zero_divergence(self, gata_pssm: pd.DataFrame):
        d = pyprego.pssm_diff(gata_pssm, gata_pssm)
        assert d < 1e-6

    def test_different_pssms_positive_divergence(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        d = pyprego.pssm_diff(gata_pssm, short_pssm)
        assert d > 0

    def test_empty_raises(self, gata_pssm: pd.DataFrame):
        empty = pssm_dataframe(np.empty((0, 4)))
        with pytest.raises(ValueError):
            pyprego.pssm_diff(gata_pssm, empty)


# ---------------------------------------------------------------------------
# pssm_match
# ---------------------------------------------------------------------------


class TestPssmMatch:
    def test_find_self_in_database(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        db = {"gata": gata_pssm, "short": short_pssm}
        result = pyprego.pssm_match(gata_pssm, db, best=True, method="spearman")
        assert result == "gata"

    def test_returns_dataframe(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        db = {"gata": gata_pssm, "short": short_pssm}
        result = pyprego.pssm_match(gata_pssm, db, method="spearman")
        assert isinstance(result, pd.DataFrame)
        assert "motif" in result.columns
        assert "cor" in result.columns
        assert len(result) == 2

    def test_kl_method(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        db = {"gata": gata_pssm, "short": short_pssm}
        result = pyprego.pssm_match(gata_pssm, db, method="kl")
        assert "kl" in result.columns
        # Self should be best (lowest KL)
        assert result.iloc[0]["motif"] == "gata"

    def test_accepts_dataframe_motifs(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        """Test with a long-format DataFrame (with 'motif' column)."""
        rows = []
        for name, p in [("gata", gata_pssm), ("short", short_pssm)]:
            arr = pssm_to_array(p)
            for i in range(arr.shape[0]):
                rows.append({"motif": name, "A": arr[i, 0], "C": arr[i, 1], "G": arr[i, 2], "T": arr[i, 3]})
        db_df = pd.DataFrame(rows)
        result = pyprego.pssm_match(gata_pssm, db_df, best=True, method="spearman")
        assert result == "gata"


# ---------------------------------------------------------------------------
# pssm_dataset_cor / pssm_dataset_diff
# ---------------------------------------------------------------------------


def _make_dataset(*named_pssms: tuple[str, pd.DataFrame]) -> pd.DataFrame:
    """Helper to build a dataset DataFrame from named PSSMs."""
    rows = []
    for name, pssm in named_pssms:
        arr = pssm_to_array(pssm)
        for i in range(arr.shape[0]):
            rows.append({
                "motif": name,
                "A": arr[i, 0],
                "C": arr[i, 1],
                "G": arr[i, 2],
                "T": arr[i, 3],
            })
    return pd.DataFrame(rows)


class TestPssmDatasetCor:
    def test_self_correlation_is_one(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        ds = _make_dataset(("gata", gata_pssm), ("short", short_pssm))
        cm = pyprego.pssm_dataset_cor(ds, method="spearman")
        assert isinstance(cm, pd.DataFrame)
        assert cm.shape == (2, 2)
        np.testing.assert_allclose(np.diag(cm.values), 1.0, atol=1e-6)

    def test_cross_dataset(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        ds1 = _make_dataset(("gata", gata_pssm))
        ds2 = _make_dataset(("short", short_pssm))
        cm = pyprego.pssm_dataset_cor(ds1, ds2, method="pearson")
        assert cm.shape == (1, 1)

    def test_symmetric(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        ds = _make_dataset(("gata", gata_pssm), ("short", short_pssm))
        cm = pyprego.pssm_dataset_cor(ds, method="spearman")
        np.testing.assert_allclose(cm.values, cm.values.T, atol=1e-10)


class TestPssmDatasetDiff:
    def test_self_divergence_is_zero(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        ds = _make_dataset(("gata", gata_pssm), ("short", short_pssm))
        dm = pyprego.pssm_dataset_diff(ds)
        np.testing.assert_allclose(np.diag(dm.values), 0.0, atol=1e-6)

    def test_cross_dataset(self, gata_pssm: pd.DataFrame, short_pssm: pd.DataFrame):
        ds1 = _make_dataset(("gata", gata_pssm))
        ds2 = _make_dataset(("short", short_pssm))
        dm = pyprego.pssm_dataset_diff(ds1, ds2)
        assert dm.shape == (1, 1)
        assert dm.values[0, 0] > 0


# ---------------------------------------------------------------------------
# pssm_to_kmer
# ---------------------------------------------------------------------------


class TestPssmToKmer:
    def test_basic(self, gata_pssm: pd.DataFrame):
        kmer = pyprego.pssm_to_kmer(gata_pssm, pos_bits_thresh=None)
        assert kmer == "GATAGATA"

    def test_short_kmer(self, gata_pssm: pd.DataFrame):
        kmer = pyprego.pssm_to_kmer(gata_pssm, kmer_length=4, pos_bits_thresh=None)
        assert len(kmer) == 4
        # Should be a substring of GATAGATA
        assert kmer in "GATAGATA"

    def test_too_long_raises(self, short_pssm: pd.DataFrame):
        with pytest.raises(ValueError):
            pyprego.pssm_to_kmer(short_pssm, kmer_length=100)

    def test_with_bits_threshold(self):
        """Uniform positions should become N when threshold is set."""
        informative = np.array([[0.9, 0.03, 0.04, 0.03]] * 2)
        uniform = np.full((2, 4), 0.25)
        mat = np.vstack([informative, uniform])
        pssm = pssm_dataframe(mat)
        kmer = pyprego.pssm_to_kmer(pssm, pos_bits_thresh=0.5)
        # The first 2 positions should be 'A', last 2 should be 'N'
        assert kmer[:2] == "AA"
        assert kmer[2:] == "NN"
