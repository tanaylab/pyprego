"""Smoke tests for pyprego.visualization.

These tests verify that the plotting functions run without errors and return
the expected matplotlib types.  They do NOT verify visual output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import matplotlib.figure

import pyprego
from pyprego.visualization import (
    _is_binary_response,
    _plot_logo_bars,
    plot_pssm_logo,
    plot_regression_prediction,
    plot_regression_prediction_binary,
    plot_regression_qc,
    plot_regression_qc_multi,
    plot_spat_model,
)
from pyprego.types import RegressionResult, pssm_dataframe, spatial_dataframe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pssm() -> pd.DataFrame:
    """PSSM with clear A-C-G-T pattern."""
    mat = np.array([
        [0.9, 0.03, 0.04, 0.03],
        [0.03, 0.9, 0.04, 0.03],
        [0.03, 0.04, 0.9, 0.03],
        [0.03, 0.03, 0.04, 0.9],
        [0.5, 0.2, 0.2, 0.1],
        [0.1, 0.1, 0.3, 0.5],
    ])
    return pssm_dataframe(mat)


@pytest.fixture
def sample_spat() -> pd.DataFrame:
    """Simple spatial model with 5 bins."""
    return spatial_dataframe(
        bins=np.array([0, 50, 100, 150, 200]),
        factors=np.array([0.8, 1.0, 1.2, 1.0, 0.8]),
    )


@pytest.fixture
def sample_regression_result(sample_pssm, sample_spat) -> RegressionResult:
    """Minimal RegressionResult for testing."""
    n = 100
    rng = np.random.default_rng(42)
    return RegressionResult(
        pssm=sample_pssm,
        spat=sample_spat,
        pred=rng.standard_normal(n),
        consensus="ACGTAW",
        r2=0.5,
        ks=None,
        seed_motif="ACGT",
        bidirect=True,
        spat_min=1,
        spat_max=200,
        seq_length=250,
    )


@pytest.fixture
def continuous_response() -> np.ndarray:
    """Continuous response vector."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(100)


@pytest.fixture
def binary_response_vec() -> np.ndarray:
    """Binary response vector."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=100).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests: plot_pssm_logo
# ---------------------------------------------------------------------------

class TestPlotPssmLogo:
    def test_returns_axes(self, sample_pssm):
        ax = plot_pssm_logo(sample_pssm)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_title(self, sample_pssm):
        ax = plot_pssm_logo(sample_pssm, title="My motif")
        assert ax.get_title() == "My motif"
        plt.close("all")

    def test_with_existing_axes(self, sample_pssm):
        fig, ax = plt.subplots()
        returned_ax = plot_pssm_logo(sample_pssm, ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_probability_method(self, sample_pssm):
        ax = plot_pssm_logo(sample_pssm, method="probability")
        assert "probability" in ax.get_ylabel().lower()
        plt.close("all")

    def test_bits_method(self, sample_pssm):
        ax = plot_pssm_logo(sample_pssm, method="bits")
        assert "bits" in ax.get_ylabel().lower()
        plt.close("all")

    def test_single_position_pssm(self):
        mat = np.array([[1.0, 0.0, 0.0, 0.0]])
        pssm = pssm_dataframe(mat)
        ax = plot_pssm_logo(pssm)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _plot_logo_bars fallback
# ---------------------------------------------------------------------------

class TestPlotLogoBars:
    def test_runs_without_error(self, sample_pssm):
        from pyprego.pssm import bits_per_pos as bpp
        from pyprego.types import pssm_to_array, NUCLEOTIDES
        mat = pssm_to_array(sample_pssm)
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        prob = mat / row_sums
        bits = bpp(sample_pssm)
        height_mat = prob * bits[:, np.newaxis]
        height_df = pd.DataFrame(height_mat, columns=list(NUCLEOTIDES))

        fig, ax = plt.subplots()
        _plot_logo_bars(height_df, ax)
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: plot_spat_model
# ---------------------------------------------------------------------------

class TestPlotSpatModel:
    def test_returns_axes(self, sample_spat):
        ax = plot_spat_model(sample_spat)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_title(self, sample_spat):
        ax = plot_spat_model(sample_spat, title="Custom spatial")
        assert ax.get_title() == "Custom spatial"
        plt.close("all")

    def test_default_title(self, sample_spat):
        ax = plot_spat_model(sample_spat)
        assert ax.get_title() == "Spatial model"
        plt.close("all")

    def test_with_existing_axes(self, sample_spat):
        fig, ax = plt.subplots()
        returned_ax = plot_spat_model(sample_spat, ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_no_title(self, sample_spat):
        ax = plot_spat_model(sample_spat, title=None)
        assert ax.get_title() == ""
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: plot_regression_prediction
# ---------------------------------------------------------------------------

class TestPlotRegressionPrediction:
    def test_returns_axes(self, continuous_response):
        pred = continuous_response + np.random.default_rng(99).standard_normal(100) * 0.5
        ax = plot_regression_prediction(pred, continuous_response)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_title(self, continuous_response):
        pred = continuous_response * 0.9
        ax = plot_regression_prediction(pred, continuous_response, title="My scatter")
        assert "My scatter" in ax.get_title()
        plt.close("all")

    def test_2d_response_averaged(self):
        rng = np.random.default_rng(42)
        response_2d = rng.standard_normal((50, 3))
        pred = rng.standard_normal(50)
        ax = plot_regression_prediction(pred, response_2d)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_custom_point_size_and_alpha(self, continuous_response):
        pred = continuous_response * 0.5
        ax = plot_regression_prediction(pred, continuous_response, point_size=2.0, alpha=0.3)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: plot_regression_prediction_binary
# ---------------------------------------------------------------------------

class TestPlotRegressionPredictionBinary:
    def test_returns_axes(self, binary_response_vec):
        rng = np.random.default_rng(42)
        pred = rng.standard_normal(100) + binary_response_vec * 0.5
        ax = plot_regression_prediction_binary(pred, binary_response_vec)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_with_title(self, binary_response_vec):
        rng = np.random.default_rng(42)
        pred = rng.standard_normal(100)
        ax = plot_regression_prediction_binary(pred, binary_response_vec, title="Binary test")
        assert "Binary test" in ax.get_title()
        plt.close("all")

    def test_shows_ks_statistic(self, binary_response_vec):
        rng = np.random.default_rng(42)
        pred = rng.standard_normal(100) + binary_response_vec * 2.0
        ax = plot_regression_prediction_binary(pred, binary_response_vec)
        title_text = ax.get_title()
        assert "Kolmogorov-Smirnov" in title_text
        plt.close("all")

    def test_legend_present(self, binary_response_vec):
        rng = np.random.default_rng(42)
        pred = rng.standard_normal(100)
        ax = plot_regression_prediction_binary(pred, binary_response_vec)
        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: _is_binary_response
# ---------------------------------------------------------------------------

class TestIsBinaryResponse:
    def test_binary_true(self):
        assert _is_binary_response(np.array([0, 1, 0, 1, 1]))

    def test_continuous_false(self):
        assert not _is_binary_response(np.array([0.5, 1.2, -0.3]))

    def test_all_zeros(self):
        assert _is_binary_response(np.array([0, 0, 0]))

    def test_all_ones(self):
        assert _is_binary_response(np.array([1, 1, 1]))

    def test_2d_binary(self):
        assert _is_binary_response(np.array([[0, 1], [1, 0]]))


# ---------------------------------------------------------------------------
# Tests: plot_regression_qc
# ---------------------------------------------------------------------------

class TestPlotRegressionQc:
    def test_returns_figure_continuous(self, sample_regression_result, continuous_response):
        fig = plot_regression_qc(sample_regression_result, continuous_response)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) >= 3  # at least 3 panels
        plt.close("all")

    def test_returns_figure_binary(self, sample_regression_result, binary_response_vec):
        fig = plot_regression_qc(sample_regression_result, binary_response_vec)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_custom_title(self, sample_regression_result, continuous_response):
        fig = plot_regression_qc(sample_regression_result, continuous_response, title="Custom")
        assert fig._suptitle.get_text() == "Custom"
        plt.close("all")

    def test_default_title_contains_consensus(self, sample_regression_result, continuous_response):
        fig = plot_regression_qc(sample_regression_result, continuous_response)
        assert "ACGTAW" in fig._suptitle.get_text()
        plt.close("all")

    def test_raises_without_response(self, sample_regression_result):
        with pytest.raises(ValueError, match="response is required"):
            plot_regression_qc(sample_regression_result)

    def test_2d_response(self, sample_regression_result):
        rng = np.random.default_rng(42)
        response_2d = rng.standard_normal((100, 3))
        fig = plot_regression_qc(sample_regression_result, response_2d)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")


# ---------------------------------------------------------------------------
# Tests: plot_regression_qc_multi
# ---------------------------------------------------------------------------

class TestPlotRegressionQcMulti:
    @pytest.fixture
    def mock_multi_result(self, sample_pssm, sample_spat):
        """Fake MultiRegressionResult-like object."""
        rng = np.random.default_rng(42)
        n = 100

        class FakeMulti:
            pass

        model1 = RegressionResult(
            pssm=sample_pssm,
            spat=sample_spat,
            pred=rng.standard_normal(n),
            consensus="ACGT",
            r2=0.4,
        )
        model2 = RegressionResult(
            pssm=sample_pssm,
            spat=sample_spat,
            pred=rng.standard_normal(n),
            consensus="TGCA",
            r2=0.3,
        )

        result = FakeMulti()
        result.models = [model1, model2]
        result.multi_stats = pd.DataFrame({
            "model": [1, 2],
            "score": [0.4, 0.3],
            "comb_score": [0.4, 0.55],
        })
        result.pred = rng.standard_normal(n)
        result.coef = np.array([0.6, 0.4])
        return result

    def test_returns_figure(self, mock_multi_result, continuous_response):
        fig = plot_regression_qc_multi(mock_multi_result, continuous_response)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_binary_response(self, mock_multi_result, binary_response_vec):
        fig = plot_regression_qc_multi(mock_multi_result, binary_response_vec)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_raises_without_response(self, mock_multi_result):
        with pytest.raises(ValueError, match="response is required"):
            plot_regression_qc_multi(mock_multi_result)

    def test_raises_with_invalid_result(self, continuous_response):
        with pytest.raises(ValueError, match="MultiRegressionResult"):
            plot_regression_qc_multi("not a result", continuous_response)

    def test_custom_title(self, mock_multi_result, continuous_response):
        fig = plot_regression_qc_multi(mock_multi_result, continuous_response, title="Multi QC")
        assert "Multi QC" in fig._suptitle.get_text()
        plt.close("all")
