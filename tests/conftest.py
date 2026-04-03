"""Shared test fixtures for pyprego."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyprego


@pytest.fixture
def simple_pssm() -> pd.DataFrame:
    """A small 4-position PSSM for testing."""
    mat = np.array([
        [0.9, 0.03, 0.04, 0.03],  # A
        [0.03, 0.9, 0.04, 0.03],  # C
        [0.03, 0.04, 0.9, 0.03],  # G
        [0.03, 0.03, 0.04, 0.9],  # T
    ])
    return pyprego.pssm_dataframe(mat)


@pytest.fixture
def uniform_pssm() -> pd.DataFrame:
    """A uniform (no information) PSSM."""
    mat = np.full((6, 4), 0.25)
    return pyprego.pssm_dataframe(mat)


@pytest.fixture
def random_sequences() -> list[str]:
    """100 random DNA sequences of length 200."""
    rng = np.random.default_rng(42)
    nucs = np.array(list("ACGT"))
    seqs = []
    for _ in range(100):
        idx = rng.integers(0, 4, size=200)
        seqs.append("".join(nucs[idx]))
    return seqs


@pytest.fixture
def binary_response() -> np.ndarray:
    """Binary response vector of length 100."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=100).astype(np.float64)
