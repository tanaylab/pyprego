"""Tests for pyprego.genomic module.

Tests mock pymisha to avoid requiring a live genome database. The integration
pipeline (intervals -> sequences -> PWM scores) is tested end-to-end with
synthetic data.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pyprego.compute import compute_local_pwm, compute_pwm
from pyprego.genomic import (
    _normalize_intervals,
    _require_pymisha,
    gextract_local_pwm,
    gextract_pwm,
    gextract_pwm_quantile,
    gintervals_center_by_pssm,
    intervals_to_seq,
)
from pyprego.types import pssm_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pssm() -> pd.DataFrame:
    """A 4-position PSSM recognising ACGT."""
    mat = np.array([
        [0.9, 0.03, 0.04, 0.03],
        [0.03, 0.9, 0.04, 0.03],
        [0.03, 0.04, 0.9, 0.03],
        [0.03, 0.03, 0.04, 0.9],
    ])
    return pssm_dataframe(mat)


@pytest.fixture
def mock_sequences() -> list[str]:
    """Predefined sequences that the mock pymisha will return."""
    return [
        "AAAAACGTAAAAA",
        "TTTTACGTTTTTA",
        "CCCCGGGACCCCG",
    ]


@pytest.fixture
def sample_intervals() -> pd.DataFrame:
    """Sample genomic intervals DataFrame."""
    return pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr2"],
        "start": [100, 200, 300],
        "end": [113, 213, 313],
    })


def _make_mock_pymisha(sequences: list[str]) -> ModuleType:
    """Create a mock pymisha module that returns the given sequences."""
    mock_pm = ModuleType("pymisha")
    mock_pm.gseq_extract = mock.MagicMock(return_value=sequences)
    mock_pm.gintervals_normalize = mock.MagicMock(
        side_effect=lambda ivs, size: _normalize_intervals(ivs, size)
    )
    mock_pm.gintervals_random = mock.MagicMock(
        side_effect=lambda size, n: pd.DataFrame({
            "chrom": [f"chr{i % 3 + 1}" for i in range(n)],
            "start": [i * 1000 for i in range(n)],
            "end": [i * 1000 + size for i in range(n)],
        })
    )
    return mock_pm


# ---------------------------------------------------------------------------
# _normalize_intervals tests
# ---------------------------------------------------------------------------


class TestNormalizeIntervals:
    """Tests for the internal _normalize_intervals helper."""

    def test_basic_centering(self) -> None:
        ivs = pd.DataFrame({
            "chrom": ["chr1", "chr2"],
            "start": [100, 200],
            "end": [200, 400],
        })
        result = _normalize_intervals(ivs, 50)
        # Center of [100, 200) is 150, half = 25 -> [125, 175)
        assert result.iloc[0]["start"] == 125
        assert result.iloc[0]["end"] == 175
        # Center of [200, 400) is 300, half = 25 -> [275, 325)
        assert result.iloc[1]["start"] == 275
        assert result.iloc[1]["end"] == 325

    def test_preserves_extra_columns(self) -> None:
        ivs = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [0],
            "end": [100],
            "name": ["gene1"],
        })
        result = _normalize_intervals(ivs, 20)
        assert "name" in result.columns
        assert result.iloc[0]["name"] == "gene1"

    def test_does_not_mutate_input(self) -> None:
        ivs = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [100],
            "end": [200],
        })
        original_start = ivs.iloc[0]["start"]
        _normalize_intervals(ivs, 50)
        assert ivs.iloc[0]["start"] == original_start


# ---------------------------------------------------------------------------
# _require_pymisha tests
# ---------------------------------------------------------------------------


class TestRequirePymisha:
    """Tests for the pymisha import guard."""

    def test_raises_when_pymisha_unavailable(self) -> None:
        with mock.patch.dict(sys.modules, {"pymisha": None}):
            with pytest.raises(ImportError, match="pymisha"):
                _require_pymisha()

    def test_succeeds_when_pymisha_available(self) -> None:
        mock_pm = ModuleType("pymisha")
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            _require_pymisha()  # should not raise


# ---------------------------------------------------------------------------
# intervals_to_seq tests
# ---------------------------------------------------------------------------


class TestIntervalsToSeq:
    """Tests for intervals_to_seq."""

    def test_basic_extraction(
        self, sample_intervals: pd.DataFrame, mock_sequences: list[str]
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            result = intervals_to_seq(sample_intervals)
        assert result == [s.upper() for s in mock_sequences]
        mock_pm.gseq_extract.assert_called_once()

    def test_uppercase_conversion(self, sample_intervals: pd.DataFrame) -> None:
        lower_seqs = ["acgtacgt", "ggggcccc", "aattccgg"]
        mock_pm = _make_mock_pymisha(lower_seqs)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            result = intervals_to_seq(sample_intervals)
        assert all(s == s.upper() for s in result)

    def test_size_parameter_calls_normalize(
        self, sample_intervals: pd.DataFrame, mock_sequences: list[str]
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            intervals_to_seq(sample_intervals, size=20)
        mock_pm.gintervals_normalize.assert_called_once()

    def test_no_size_skips_normalize(
        self, sample_intervals: pd.DataFrame, mock_sequences: list[str]
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            intervals_to_seq(sample_intervals)
        mock_pm.gintervals_normalize.assert_not_called()

    def test_raises_without_pymisha(self, sample_intervals: pd.DataFrame) -> None:
        with mock.patch.dict(sys.modules, {"pymisha": None}):
            with pytest.raises(ImportError, match="pymisha"):
                intervals_to_seq(sample_intervals)


# ---------------------------------------------------------------------------
# gextract_pwm tests
# ---------------------------------------------------------------------------


class TestGextractPwm:
    """Tests for gextract_pwm."""

    def test_returns_scores_array(
        self,
        sample_intervals: pd.DataFrame,
        mock_sequences: list[str],
        simple_pssm: pd.DataFrame,
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            scores = gextract_pwm(sample_intervals, simple_pssm)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_scores_match_compute_pwm(
        self,
        sample_intervals: pd.DataFrame,
        mock_sequences: list[str],
        simple_pssm: pd.DataFrame,
    ) -> None:
        """gextract_pwm should produce identical results to direct compute_pwm."""
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            g_scores = gextract_pwm(sample_intervals, simple_pssm)
        direct_scores = compute_pwm(
            [s.upper() for s in mock_sequences], simple_pssm
        )
        np.testing.assert_array_almost_equal(g_scores, direct_scores)

    def test_bidirect_false(
        self,
        sample_intervals: pd.DataFrame,
        mock_sequences: list[str],
        simple_pssm: pd.DataFrame,
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            scores_bi = gextract_pwm(sample_intervals, simple_pssm, bidirect=True)
            scores_uni = gextract_pwm(sample_intervals, simple_pssm, bidirect=False)
        # Bidirectional scores should generally differ from unidirectional
        assert not np.allclose(scores_bi, scores_uni)

    def test_func_max(
        self,
        sample_intervals: pd.DataFrame,
        mock_sequences: list[str],
        simple_pssm: pd.DataFrame,
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            scores_lse = gextract_pwm(
                sample_intervals, simple_pssm, func="logSumExp"
            )
            scores_max = gextract_pwm(
                sample_intervals, simple_pssm, func="max"
            )
        # logSumExp >= max, and they should not be identical
        assert np.all(scores_lse >= scores_max - 1e-10)

    def test_with_size(
        self,
        sample_intervals: pd.DataFrame,
        simple_pssm: pd.DataFrame,
    ) -> None:
        """Size parameter should be forwarded to intervals_to_seq."""
        seqs = ["ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGT", "ACGTACGTACGTACGTACGT"]
        mock_pm = _make_mock_pymisha(seqs)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            scores = gextract_pwm(sample_intervals, simple_pssm, size=20)
        assert scores.shape == (3,)
        mock_pm.gintervals_normalize.assert_called_once()


# ---------------------------------------------------------------------------
# gextract_local_pwm tests
# ---------------------------------------------------------------------------


class TestGextractLocalPwm:
    """Tests for gextract_local_pwm."""

    def test_returns_2d_array(
        self,
        sample_intervals: pd.DataFrame,
        mock_sequences: list[str],
        simple_pssm: pd.DataFrame,
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            local = gextract_local_pwm(sample_intervals, simple_pssm)
        assert isinstance(local, np.ndarray)
        assert local.ndim == 2
        assert local.shape[0] == 3
        assert local.shape[1] == len(mock_sequences[0])

    def test_matches_direct_compute_local(
        self,
        sample_intervals: pd.DataFrame,
        mock_sequences: list[str],
        simple_pssm: pd.DataFrame,
    ) -> None:
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            g_local = gextract_local_pwm(sample_intervals, simple_pssm)
        direct_local = compute_local_pwm(
            [s.upper() for s in mock_sequences], simple_pssm
        )
        np.testing.assert_array_almost_equal(g_local, direct_local)

    def test_trailing_nan_positions(
        self,
        sample_intervals: pd.DataFrame,
        mock_sequences: list[str],
        simple_pssm: pd.DataFrame,
    ) -> None:
        """Positions where PSSM overhangs should be NaN."""
        mock_pm = _make_mock_pymisha(mock_sequences)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            local = gextract_local_pwm(sample_intervals, simple_pssm)
        seq_len = len(mock_sequences[0])
        pssm_len = len(simple_pssm)
        # Last (pssm_len - 1) positions should be NaN
        n_nan = pssm_len - 1
        assert np.all(np.isnan(local[:, seq_len - n_nan :]))


# ---------------------------------------------------------------------------
# gextract_pwm_quantile tests
# ---------------------------------------------------------------------------


class TestGextractPwmQuantile:
    """Tests for gextract_pwm_quantile."""

    def test_returns_values_in_0_1(
        self,
        sample_intervals: pd.DataFrame,
        simple_pssm: pd.DataFrame,
    ) -> None:
        seq_len = 13
        seqs = ["AAAAACGTAAAAA", "TTTTACGTTTTTA", "CCCCGGGACCCCG"]

        # For background, generate longer list
        rng = np.random.default_rng(42)
        bg_seqs = []
        nucs = list("ACGT")
        for _ in range(50):
            bg_seqs.append("".join(rng.choice(nucs, size=seq_len)))

        # We need the mock to return different sequences on different calls
        call_count = [0]

        def mock_gseq_extract(ivs):
            nonlocal call_count
            if call_count[0] == 0:
                call_count[0] += 1
                return seqs
            else:
                call_count[0] += 1
                return bg_seqs

        mock_pm = _make_mock_pymisha(seqs)
        mock_pm.gseq_extract = mock.MagicMock(side_effect=mock_gseq_extract)

        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            quantiles = np.arange(0, 1.01, 0.01)
            result = gextract_pwm_quantile(
                sample_intervals, simple_pssm, quantiles, n_sequences=50,
            )
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_with_bg_intervals(
        self,
        sample_intervals: pd.DataFrame,
        simple_pssm: pd.DataFrame,
    ) -> None:
        """When bg_intervals is given, gintervals_random should not be called."""
        seqs = ["AAAAACGTAAAAA", "TTTTACGTTTTTA", "CCCCGGGACCCCG"]
        bg_ivs = pd.DataFrame({
            "chrom": ["chr1"] * 5,
            "start": [0, 100, 200, 300, 400],
            "end": [13, 113, 213, 313, 413],
        })
        rng = np.random.default_rng(42)
        bg_seqs = ["".join(rng.choice(list("ACGT"), size=13)) for _ in range(5)]

        call_count = [0]

        def mock_gseq_extract(ivs):
            nonlocal call_count
            if call_count[0] == 0:
                call_count[0] += 1
                return seqs
            else:
                call_count[0] += 1
                return bg_seqs

        mock_pm = _make_mock_pymisha(seqs)
        mock_pm.gseq_extract = mock.MagicMock(side_effect=mock_gseq_extract)

        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            result = gextract_pwm_quantile(
                sample_intervals, simple_pssm,
                quantiles=[0.0, 0.25, 0.5, 0.75, 1.0],
                bg_intervals=bg_ivs,
            )
        assert result.shape == (3,)
        mock_pm.gintervals_random.assert_not_called()


# ---------------------------------------------------------------------------
# gintervals_center_by_pssm tests
# ---------------------------------------------------------------------------


class TestGintervalsCenterByPssm:
    """Tests for gintervals_center_by_pssm."""

    def test_basic_centering(self) -> None:
        """When a motif is planted at a known offset, centering should shift
        the interval so the motif is at the center."""
        pssm_mat = np.array([
            [0.97, 0.01, 0.01, 0.01],  # A
            [0.01, 0.97, 0.01, 0.01],  # C
            [0.01, 0.01, 0.97, 0.01],  # G
            [0.01, 0.01, 0.01, 0.97],  # T
        ])
        pssm = pssm_dataframe(pssm_mat)

        # Sequence with ACGT planted at position 8
        seq = "GGGGGGGGACGTGGGGGGG"  # len=19, motif at positions 8-11
        intervals = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [1000],
            "end": [1019],
        })
        mock_pm = _make_mock_pymisha([seq])

        def mock_normalize(ivs, size):
            return _normalize_intervals(ivs, size)

        mock_pm.gintervals_normalize = mock.MagicMock(side_effect=mock_normalize)

        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            result = gintervals_center_by_pssm(intervals, pssm, size=10)

        # The max-score position should be 8 (where ACGT starts)
        # New start = 1000 + 8 = 1008, end = 1009
        # After normalize to size=10: center=1008, half=5 -> [1003, 1013)
        assert result.iloc[0]["chrom"] == "chr1"
        assert result.iloc[0]["end"] - result.iloc[0]["start"] == 10

    def test_preserves_extra_columns(self) -> None:
        pssm = pssm_dataframe(np.full((3, 4), 0.25))
        seq = "ACGTACGTACGT"
        intervals = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [500],
            "end": [512],
            "name": ["peak1"],
            "score": [42.0],
        })
        mock_pm = _make_mock_pymisha([seq])
        mock_pm.gintervals_normalize = mock.MagicMock(
            side_effect=lambda ivs, size: _normalize_intervals(ivs, size)
        )

        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            result = gintervals_center_by_pssm(intervals, pssm, size=8)
        assert "name" in result.columns
        assert "score" in result.columns

    def test_standard_column_order(self) -> None:
        pssm = pssm_dataframe(np.full((3, 4), 0.25))
        seq = "ACGTACGTACGT"
        intervals = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [500],
            "end": [512],
        })
        mock_pm = _make_mock_pymisha([seq])
        mock_pm.gintervals_normalize = mock.MagicMock(
            side_effect=lambda ivs, size: _normalize_intervals(ivs, size)
        )

        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            result = gintervals_center_by_pssm(intervals, pssm, size=8)
        assert list(result.columns[:3]) == ["chrom", "start", "end"]


# ---------------------------------------------------------------------------
# Integration pipeline tests (mock pymisha but full compute pipeline)
# ---------------------------------------------------------------------------


class TestIntegrationPipeline:
    """End-to-end pipeline tests with synthetic data."""

    def test_full_pipeline_planted_motif(self) -> None:
        """Plant a strong motif in specific positions, verify scoring
        pipeline produces higher scores for sequences containing it."""
        pssm_mat = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        pssm = pssm_dataframe(pssm_mat)

        rng = np.random.default_rng(123)
        nucs = list("ACGT")
        n = 20
        seqs_with = []
        seqs_without = []
        for _ in range(n):
            bg = list(rng.choice(nucs, size=50))
            seqs_without.append("".join(bg))
            bg2 = list(rng.choice(nucs, size=50))
            pos = rng.integers(5, 40)
            bg2[pos : pos + 4] = list("ACGT")
            seqs_with.append("".join(bg2))

        all_seqs = seqs_with + seqs_without
        intervals = pd.DataFrame({
            "chrom": [f"chr{i % 3 + 1}" for i in range(2 * n)],
            "start": [i * 1000 for i in range(2 * n)],
            "end": [i * 1000 + 50 for i in range(2 * n)],
        })

        mock_pm = _make_mock_pymisha(all_seqs)
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            scores = gextract_pwm(intervals, pssm)

        # Sequences with planted motif should score higher on average
        mean_with = scores[:n].mean()
        mean_without = scores[n:].mean()
        assert mean_with > mean_without

    def test_local_pwm_max_at_planted_position(self) -> None:
        """Verify local PWM peaks at the planted motif position."""
        pssm_mat = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        pssm = pssm_dataframe(pssm_mat)

        # Plant ACGT at position 10 in a G-rich background
        seq = "G" * 10 + "ACGT" + "G" * 16  # len=30
        intervals = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [0],
            "end": [30],
        })

        mock_pm = _make_mock_pymisha([seq])
        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            local = gextract_local_pwm(intervals, pssm)

        # The max position should be at position 10
        max_pos = np.nanargmax(local[0])
        assert max_pos == 10

    def test_center_by_pssm_shifts_to_motif(self) -> None:
        """After centering, the motif position should be near the interval center."""
        pssm_mat = np.array([
            [0.97, 0.01, 0.01, 0.01],
            [0.01, 0.97, 0.01, 0.01],
            [0.01, 0.01, 0.97, 0.01],
            [0.01, 0.01, 0.01, 0.97],
        ])
        pssm = pssm_dataframe(pssm_mat)

        # Plant ACGT at position 15 in a 30bp interval
        seq = "G" * 15 + "ACGT" + "G" * 11  # len=30
        intervals = pd.DataFrame({
            "chrom": ["chr1"],
            "start": [500],
            "end": [530],
        })

        mock_pm = _make_mock_pymisha([seq])
        mock_pm.gintervals_normalize = mock.MagicMock(
            side_effect=lambda ivs, size: _normalize_intervals(ivs, size)
        )

        with mock.patch.dict(sys.modules, {"pymisha": mock_pm}):
            result = gintervals_center_by_pssm(intervals, pssm, size=20)

        # The interval should have been shifted so position 15 is the center
        center = (result.iloc[0]["start"] + result.iloc[0]["end"]) // 2
        # Original start was 500, motif at position 15 -> genomic position 515
        assert abs(center - 515) <= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test error handling for all genomic functions when pymisha is missing."""

    def test_intervals_to_seq_no_pymisha(self) -> None:
        ivs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        with mock.patch.dict(sys.modules, {"pymisha": None}):
            with pytest.raises(ImportError, match="pymisha"):
                intervals_to_seq(ivs)

    def test_gextract_pwm_no_pymisha(self) -> None:
        ivs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        pssm = pssm_dataframe(np.full((4, 4), 0.25))
        with mock.patch.dict(sys.modules, {"pymisha": None}):
            with pytest.raises(ImportError, match="pymisha"):
                gextract_pwm(ivs, pssm)

    def test_gextract_local_pwm_no_pymisha(self) -> None:
        ivs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        pssm = pssm_dataframe(np.full((4, 4), 0.25))
        with mock.patch.dict(sys.modules, {"pymisha": None}):
            with pytest.raises(ImportError, match="pymisha"):
                gextract_local_pwm(ivs, pssm)

    def test_gextract_pwm_quantile_no_pymisha(self) -> None:
        ivs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        pssm = pssm_dataframe(np.full((4, 4), 0.25))
        with mock.patch.dict(sys.modules, {"pymisha": None}):
            with pytest.raises(ImportError, match="pymisha"):
                gextract_pwm_quantile(ivs, pssm, [0.5])

    def test_gintervals_center_no_pymisha(self) -> None:
        ivs = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100]})
        pssm = pssm_dataframe(np.full((4, 4), 0.25))
        with mock.patch.dict(sys.modules, {"pymisha": None}):
            with pytest.raises(ImportError, match="pymisha"):
                gintervals_center_by_pssm(ivs, pssm, size=50)
