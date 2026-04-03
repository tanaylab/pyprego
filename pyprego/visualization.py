"""Visualization functions for motif logos and regression diagnostics.

Mirrors plot-logo.R and plot-regression.R from the R prego package.
Uses matplotlib as the base; logomaker is optional for sequence logos.
All matplotlib imports are lazy so pyprego works without matplotlib installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .pssm import bits_per_pos
from .types import NUCLEOTIDES, pssm_to_array

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from .types import RegressionResult


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _require_matplotlib():
    """Raise a clear error if matplotlib is not installed."""
    try:
        import matplotlib.pyplot  # noqa: F401
    except ImportError as exc:
        raise ImportError("Plotting requires matplotlib. Install it with: pip install matplotlib") from exc


def _try_import_logomaker():
    """Return the logomaker module if available, else None."""
    try:
        import logomaker

        return logomaker
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Nucleotide colour map (matches ggseqlogo "classic" scheme)
# ---------------------------------------------------------------------------

_NUC_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}


# ---------------------------------------------------------------------------
# Sequence logo
# ---------------------------------------------------------------------------


def plot_pssm_logo(
    pssm: pd.DataFrame,
    *,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = None,
    method: str = "bits",
) -> matplotlib.axes.Axes:
    """Plot a sequence logo for a PSSM.

    Tries to use ``logomaker`` for high-quality logos.  If logomaker is not
    installed, falls back to a stacked bar chart rendered with plain
    matplotlib.

    Parameters
    ----------
    pssm : pd.DataFrame
        PSSM DataFrame with columns ``pos``, ``A``, ``C``, ``G``, ``T``.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.  If ``None``, a new figure is created.
    title : str | None
        Optional title for the plot.
    method : str
        ``"bits"`` (default) scales letters by information content;
        ``"probability"`` scales by raw probability.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the logo drawn.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    mat = pssm_to_array(pssm)
    # Normalise to probabilities
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    prob = mat / row_sums

    if method == "bits":
        bits = bits_per_pos(pssm)
        height_mat = prob * bits[:, np.newaxis]
    else:
        height_mat = prob

    height_df = pd.DataFrame(height_mat, columns=list(NUCLEOTIDES))

    if ax is None:
        _, ax = plt.subplots(figsize=(max(3, len(pssm) * 0.4), 2.5))

    logomaker = _try_import_logomaker()
    if logomaker is not None:
        logomaker.Logo(height_df, ax=ax, color_scheme="classic")
    else:
        # Fallback: stacked bar chart
        _plot_logo_bars(height_df, ax)

    ax.set_ylabel("bits" if method == "bits" else "probability")
    ax.set_xlabel("position")
    if title:
        ax.set_title(title)
    return ax


def _plot_logo_bars(
    height_df: pd.DataFrame,
    ax,
) -> None:
    """Fallback stacked-bar logo when logomaker is not available."""
    positions = np.arange(len(height_df))
    for i, pos in enumerate(positions):
        row = height_df.iloc[i]
        # Sort nucleotides by height so largest is on top
        order = row.sort_values().index.tolist()
        bottom = 0.0
        for nuc in order:
            h = row[nuc]
            if h > 0:
                ax.bar(pos, h, bottom=bottom, color=_NUC_COLORS[nuc], width=0.8, edgecolor="none")
                if h > 0.15:
                    ax.text(
                        pos, bottom + h / 2, nuc, ha="center", va="center", fontsize=8, fontweight="bold", color="white"
                    )
                bottom += h
    ax.set_xlim(-0.5, len(height_df) - 0.5)


# ---------------------------------------------------------------------------
# Spatial model
# ---------------------------------------------------------------------------


def plot_spat_model(
    spat: pd.DataFrame,
    *,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = "Spatial model",
) -> matplotlib.axes.Axes:
    """Plot a spatial model as a line-and-point chart.

    Mirrors ``plot_spat_model()`` from R which uses ``geom_line + geom_point``.

    Parameters
    ----------
    spat : pd.DataFrame
        Spatial model DataFrame with columns ``bin`` and ``spat_factor``.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.  If ``None``, a new figure is created.
    title : str | None
        Optional title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))

    bins = spat["bin"].to_numpy()
    factors = spat["spat_factor"].to_numpy()

    ax.plot(bins, factors, marker="o", markersize=4, linewidth=1.2, color="#333333")
    ax.set_xlabel("Position")
    ax.set_ylabel("Spatial factor")
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Regression prediction: continuous
# ---------------------------------------------------------------------------


def plot_regression_prediction(
    pred: np.ndarray,
    response: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    point_size: float = 0.5,
    alpha: float = 1.0,
    title: str | None = "Regression prediction",
) -> matplotlib.axes.Axes:
    """Scatter plot of predicted vs observed response.

    Mirrors the R ``plot_regression_prediction`` which shows response on
    x-axis and prediction on y-axis, annotated with R-squared and r.

    Parameters
    ----------
    pred : np.ndarray
        Predicted scores.
    response : np.ndarray
        Observed response values.
    ax : matplotlib.axes.Axes | None
        Axes to plot on.
    point_size : float
        Marker size (matplotlib ``s`` parameter).
    alpha : float
        Marker transparency.
    title : str | None
        Optional title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    pred = np.asarray(pred, dtype=np.float64).ravel()
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 2:
        response = response.mean(axis=1)
    response = response.ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(response, pred, s=point_size, alpha=alpha, color="#333333", edgecolors="none")
    ax.set_xlabel("Response")
    ax.set_ylabel("Prediction")

    # Compute correlation and R-squared
    mask = np.isfinite(pred) & np.isfinite(response)
    if mask.sum() > 2:
        correlation = float(np.corrcoef(pred[mask], response[mask])[0, 1])
        r2 = correlation**2
        subtitle = f"$r^2$ = {r2:.3f}, $r$ = {correlation:.3f}"
        ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=10)
    elif title:
        ax.set_title(title)

    ax.set_aspect("equal", adjustable="datalim")
    return ax


# ---------------------------------------------------------------------------
# Regression prediction: binary (1 - ECDF plot, matching R KS approach)
# ---------------------------------------------------------------------------


def plot_regression_prediction_binary(
    pred: np.ndarray,
    response: np.ndarray,
    *,
    ax: matplotlib.axes.Axes | None = None,
    title: str | None = "Regression prediction",
) -> matplotlib.axes.Axes:
    """Plot 1-ECDF of predictions stratified by binary response class.

    Mirrors the R ``plot_regression_prediction_binary`` which shows the
    inverted empirical CDF (1 - ECDF) for class 0 (blue) and class 1 (red),
    plus a KS D statistic annotation.

    Parameters
    ----------
    pred : np.ndarray
        Predicted scores.
    response : np.ndarray
        Binary response (0/1).
    ax : matplotlib.axes.Axes | None
        Axes to plot on.
    title : str | None
        Optional title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    pred = np.asarray(pred, dtype=np.float64).ravel()
    response = np.asarray(response, dtype=np.float64).ravel()

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.5))

    mask_1 = response == 1
    mask_0 = response == 0
    pred_1 = pred[mask_1]
    pred_0 = pred[mask_0]

    # KS test (alternative='less' because we invert the CDF)
    ks_result = ks_2samp(pred_1, pred_0, alternative="less")

    # Plot 1 - ECDF for each class
    for pred_class, label, color in [(pred_0, "0", "blue"), (pred_1, "1", "red")]:
        sorted_vals = np.sort(pred_class)
        # 1 - ECDF: y goes from 1 down to 0
        ecdf_y = 1.0 - np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        # Prepend the starting point (min value, y=1)
        x_plot = np.concatenate([[sorted_vals[0]], sorted_vals])
        y_plot = np.concatenate([[1.0], ecdf_y])
        ax.step(x_plot, y_plot, where="post", label=label, color=color, linewidth=1.2)

    # KS D segment: find x where the gap is largest
    all_vals = np.sort(np.concatenate([pred_0, pred_1]))
    ecdf_0 = np.searchsorted(np.sort(pred_0), all_vals, side="right") / len(pred_0)
    ecdf_1 = np.searchsorted(np.sort(pred_1), all_vals, side="right") / len(pred_1)
    inv_ecdf_0 = 1.0 - ecdf_0
    inv_ecdf_1 = 1.0 - ecdf_1
    gap = np.abs(inv_ecdf_0 - inv_ecdf_1)
    best_idx = np.argmax(gap)
    x0 = all_vals[best_idx]
    y0 = inv_ecdf_0[best_idx]
    y1 = inv_ecdf_1[best_idx]
    ax.plot([x0, x0], [y0, y1], linestyle="--", color="gray", linewidth=1)

    ax.set_xlabel("")
    ax.set_ylabel("1 - ECDF")
    ax.legend(title="Response")

    subtitle = f"Kolmogorov-Smirnov D = {ks_result.statistic:.3f}, p-value = {ks_result.pvalue:.3f}"
    ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=10)

    return ax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_binary_response(response: np.ndarray) -> bool:
    """Check if a response vector is binary (contains only 0 and 1)."""
    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 2:
        response = response.ravel()
    unique = np.unique(response[np.isfinite(response)])
    return set(unique).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Regression QC (multi-panel)
# ---------------------------------------------------------------------------


def plot_regression_qc(
    result: RegressionResult,
    response: np.ndarray | None = None,
    *,
    title: str | None = None,
    point_size: float = 0.5,
    alpha: float = 0.5,
) -> matplotlib.figure.Figure:
    """Multi-panel QC figure for a single-motif regression result.

    Creates a figure with three panels:

    1. PSSM sequence logo
    2. Spatial model
    3. Prediction vs response scatter (continuous) or 1-ECDF (binary)

    Mirrors ``plot_regression_qc`` from R.

    Parameters
    ----------
    result : RegressionResult
        Output of :func:`pyprego.regress_pwm`.
    response : np.ndarray | None
        Response variable.  Required -- used for the prediction panel.
    title : str | None
        Overall figure title.  Defaults to showing the consensus.
    point_size : float
        Point size for scatter plot.
    alpha : float
        Transparency for scatter plot.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all panels.

    Raises
    ------
    ValueError
        If *response* is ``None``.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if response is None:
        raise ValueError("response is required for plot_regression_qc. Pass the response array used during regression.")

    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 2:
        response = response.mean(axis=1)

    if title is None:
        title = f"Motif regression results (consensus: {result.consensus})"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Panel 1: PSSM logo
    plot_pssm_logo(result.pssm, ax=axes[0], title="Sequence model")

    # Panel 2: Spatial model
    plot_spat_model(result.spat, ax=axes[1], title="Spatial model")

    # Panel 3: Prediction plot
    if _is_binary_response(response):
        plot_regression_prediction_binary(result.pred, response, ax=axes[2])
    else:
        plot_regression_prediction(
            result.pred,
            response,
            ax=axes[2],
            point_size=point_size,
            alpha=alpha,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Standalone prediction scatter (convenience wrapper)
# ---------------------------------------------------------------------------
# plot_regression_prediction and plot_regression_prediction_binary
# are already defined above as standalone functions.


# ---------------------------------------------------------------------------
# Multi-motif QC
# ---------------------------------------------------------------------------


def plot_regression_qc_multi(
    result,
    response: np.ndarray | None = None,
    *,
    title: str | None = None,
    point_size: float = 0.01,
    alpha: float = 0.5,
) -> matplotlib.figure.Figure:
    """Multi-panel QC figure for a multi-motif regression result.

    For each motif, shows: PSSM logo, spatial model, and prediction panel.
    Also shows a score summary panel at the bottom.

    Mirrors ``plot_regression_qc_multi`` from R.

    Parameters
    ----------
    result : MultiRegressionResult
        Output of :func:`pyprego.regress_multiple_motifs`.
    response : np.ndarray | None
        Response variable.  Required -- used for prediction panels.
    title : str | None
        Overall figure title.
    point_size : float
        Point size for scatter plots.
    alpha : float
        Transparency for scatter plots.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing all panels.

    Raises
    ------
    ValueError
        If *response* is ``None`` or *result* has no ``models`` attribute.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if not hasattr(result, "models") or not hasattr(result, "multi_stats"):
        raise ValueError("result must be a MultiRegressionResult with 'models' and 'multi_stats' attributes.")

    if response is None:
        raise ValueError(
            "response is required for plot_regression_qc_multi. Pass the response array used during regression."
        )

    response = np.asarray(response, dtype=np.float64)
    if response.ndim == 2:
        response = response.mean(axis=1)

    models = result.models
    n_models = len(models)
    is_binary = _is_binary_response(response)

    if title is None:
        consensuses = [m.consensus for m in models]
        title = f"Multi-motif regression ({n_models} motifs: {', '.join(consensuses)})"

    # Layout: n_models rows x 3 columns (logo, spatial, prediction)
    # Plus 1 extra row for the score summary
    n_rows = n_models + 1
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 3.5 * n_rows),
        gridspec_kw={"height_ratios": [1.0] * n_models + [0.6]},
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # axes shape is (n_rows, n_cols) -- ensure 2D
    if n_models == 1 and axes.ndim == 1:
        axes = axes.reshape(n_rows, n_cols)

    stats = result.multi_stats

    for i, model in enumerate(models):
        # Motif score annotation
        if "score" in stats.columns and i < len(stats):
            score_val = stats["score"].iloc[i]
            comb_val = stats["comb_score"].iloc[i] if "comb_score" in stats.columns else None
            subtitle = f"score = {score_val:.3f}"
            if comb_val is not None:
                subtitle += f", combined = {comb_val:.3f}"
        else:
            subtitle = None

        motif_title = f"Motif #{i + 1}"
        if subtitle:
            motif_title += f"\n{subtitle}"

        plot_pssm_logo(model.pssm, ax=axes[i, 0], title=motif_title)
        plot_spat_model(model.spat, ax=axes[i, 1])

        if is_binary:
            plot_regression_prediction_binary(model.pred, response, ax=axes[i, 2])
        else:
            plot_regression_prediction(
                model.pred,
                response,
                ax=axes[i, 2],
                point_size=point_size,
                alpha=alpha,
            )

    # Bottom row: score summary bar chart
    # Merge into one axes spanning all 3 columns
    for j in range(n_cols):
        axes[n_models, j].remove()
    gs = axes[0, 0].get_gridspec()
    ax_scores = fig.add_subplot(gs[n_models, :])

    if "score" in stats.columns:
        x = np.arange(len(stats))
        width = 0.35
        score_vals = stats["score"].to_numpy()
        bars1 = ax_scores.bar(x - width / 2, score_vals, width, label="Score", color="#4472C4")
        for bar, val in zip(bars1, score_vals, strict=False):
            ax_scores.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=8
            )

        if "comb_score" in stats.columns:
            comb_vals = stats["comb_score"].to_numpy()
            bars2 = ax_scores.bar(x + width / 2, comb_vals, width, label="Combined score", color="#ED7D31")
            for bar, val in zip(bars2, comb_vals, strict=False):
                ax_scores.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax_scores.set_xticks(x)
        ax_scores.set_xticklabels([f"Model {i + 1}" for i in range(len(stats))])
        ax_scores.set_ylabel("Score")
        ax_scores.legend()

    fig.tight_layout()
    return fig
