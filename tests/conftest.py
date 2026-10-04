"""
Shared pytest fixtures for ChatAutoML test suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_classification_df() -> pd.DataFrame:
    """Small binary classification dataset for AutoML pipeline tests."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "age": rng.integers(18, 80, size=n),
        "income": rng.normal(45000, 15000, size=n),
        "score": rng.uniform(0, 1, size=n),
        "category": rng.choice(["A", "B", "C"], size=n),
        "target": rng.integers(0, 2, size=n),
    })


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Small regression dataset."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 3))
    y = 2.5 * X[:, 0] - 1.3 * X[:, 1] + rng.normal(0, 0.5, n)
    df = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3"])
    df["target"] = y
    return df
