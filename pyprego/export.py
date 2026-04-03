"""Export and load regression models.

Mirrors export.R from the R prego package. Provides JSON-based
serialisation for single and multi-motif regression models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .compute import compute_pwm
from .pssm import consensus_from_pssm
from .types import RegressionResult, pssm_dataframe, spatial_dataframe


# ──────────────────────────────────────────────────────────────────────
# Single model export/load
# ──────────────────────────────────────────────────────────────────────


def export_regression_model(
    model: RegressionResult,
    fn: str | Path | None = None,
) -> dict[str, Any] | None:
    """Export a single-motif regression model.

    Parameters
    ----------
    model : RegressionResult
        Fitted regression model.
    fn : str | Path | None
        File path to save JSON. If ``None``, returns the dict instead.

    Returns
    -------
    dict | None
        If *fn* is ``None``, returns the serialisable dict.
    """
    data = {
        "pssm": model.pssm.to_dict(orient="list"),
        "spat": model.spat.to_dict(orient="list"),
        "spat_min": int(model.spat_min),
        "spat_max": int(model.spat_max) if model.spat_max is not None else None,
        "bidirect": bool(model.bidirect),
        "seq_length": int(model.seq_length) if model.seq_length is not None else None,
        "seed_motif": model.seed_motif,
        "consensus": model.consensus,
        "r2": float(model.r2) if model.r2 is not None else None,
        "ks": float(model.ks) if model.ks is not None else None,
    }

    if fn is not None:
        fn = Path(fn)
        fn.parent.mkdir(parents=True, exist_ok=True)
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)
        return None

    return data


def load_regression_model(fn: str | Path | dict) -> RegressionResult:
    """Load a single-motif regression model from file or dict.

    Parameters
    ----------
    fn : str | Path | dict
        JSON file path or a dict (as returned by :func:`export_regression_model`).

    Returns
    -------
    RegressionResult
        Loaded model with a functioning ``predict()`` method.
    """
    if isinstance(fn, dict):
        data = fn
    else:
        with open(fn) as f:
            data = json.load(f)

    pssm_df = pd.DataFrame(data["pssm"])
    spat_df = pd.DataFrame(data["spat"])
    spat_min = data["spat_min"]
    spat_max = data["spat_max"]
    bidirect = data["bidirect"]

    def _predict_fn(sequences: list[str] | np.ndarray) -> np.ndarray:
        sequences = [s.upper() for s in sequences]
        trimmed = [s[spat_min:spat_max] for s in sequences]
        return compute_pwm(trimmed, pssm_df, spat=spat_df, bidirect=bidirect, prior=0)

    return RegressionResult(
        pssm=pssm_df,
        spat=spat_df,
        pred=np.array([]),  # No stored predictions for loaded models
        consensus=data.get("consensus", ""),
        r2=data.get("r2"),
        ks=data.get("ks"),
        seed_motif=data.get("seed_motif"),
        bidirect=bidirect,
        spat_min=spat_min,
        spat_max=spat_max,
        seq_length=data.get("seq_length"),
        _predict_fn=_predict_fn,
    )


# ──────────────────────────────────────────────────────────────────────
# Multi-model export/load
# ──────────────────────────────────────────────────────────────────────


def export_multi_regression(
    reg,  # MultiRegressionResult
    fn: str | Path | None = None,
) -> dict[str, Any] | None:
    """Export a multi-motif regression model.

    Parameters
    ----------
    reg : MultiRegressionResult
        Multi-motif regression result.
    fn : str | Path | None
        File path to save JSON. If ``None``, returns the dict.

    Returns
    -------
    dict | None
        If *fn* is ``None``, returns the serialisable dict.
    """
    models_data = []
    for m in reg.models:
        models_data.append({
            "pssm": m.pssm.to_dict(orient="list"),
            "spat": m.spat.to_dict(orient="list"),
            "spat_min": int(m.spat_min),
            "spat_max": int(m.spat_max) if m.spat_max is not None else None,
            "bidirect": bool(m.bidirect),
            "seq_length": int(m.seq_length) if m.seq_length is not None else None,
            "consensus": m.consensus,
            "seed_motif": m.seed_motif,
        })

    data = {
        "models": models_data,
        "motif_num": len(reg.models),
        "coef": reg.coef.tolist() if isinstance(reg.coef, np.ndarray) else list(reg.coef),
        "intercept": float(reg.intercept),
        "multi_stats": reg.multi_stats.to_dict(orient="list"),
    }

    if fn is not None:
        fn = Path(fn)
        fn.parent.mkdir(parents=True, exist_ok=True)
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)
        return None

    return data


def load_multi_regression(fn: str | Path | dict):
    """Load a multi-motif regression model.

    Parameters
    ----------
    fn : str | Path | dict
        JSON file path or dict.

    Returns
    -------
    MultiRegressionResult
        Loaded multi-motif model with ``predict()`` and ``predict_multi()``.
    """
    from .regression import MultiRegressionResult

    if isinstance(fn, dict):
        data = fn
    else:
        with open(fn) as f:
            data = json.load(f)

    models = []
    for md in data["models"]:
        pssm_df = pd.DataFrame(md["pssm"])
        spat_df = pd.DataFrame(md["spat"])
        spat_min = md["spat_min"]
        spat_max = md["spat_max"]
        bidirect = md["bidirect"]

        def _make_predict(pssm, spat, smin, smax, bi):
            def _predict_fn(sequences):
                sequences = [s.upper() for s in sequences]
                trimmed = [s[smin:smax] for s in sequences]
                return compute_pwm(trimmed, pssm, spat=spat, bidirect=bi, prior=0)
            return _predict_fn

        result = RegressionResult(
            pssm=pssm_df,
            spat=spat_df,
            pred=np.array([]),
            consensus=md.get("consensus", ""),
            seed_motif=md.get("seed_motif"),
            bidirect=bidirect,
            spat_min=spat_min,
            spat_max=spat_max,
            seq_length=md.get("seq_length"),
            _predict_fn=_make_predict(pssm_df, spat_df, spat_min, spat_max, bidirect),
        )
        models.append(result)

    coef = np.array(data["coef"], dtype=np.float64)
    intercept = float(data["intercept"])
    multi_stats = pd.DataFrame(data["multi_stats"])

    return MultiRegressionResult(
        models=models,
        multi_stats=multi_stats,
        pred=np.array([]),
        coef=coef,
        intercept=intercept,
    )
