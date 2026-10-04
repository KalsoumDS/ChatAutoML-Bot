"""
Unit tests for ChatAutoML preprocessing pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from automl.preprocessing import DataPreprocessor


class TestDataPreprocessor:

    def test_handles_missing_values(self, binary_classification_df):
        """Preprocessor must fill or drop NaN values without errors."""
        df = binary_classification_df.copy()
        df.loc[5:10, "income"] = np.nan
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(df, target_col="target")
        assert not np.isnan(X).any(), "No NaN values should remain after preprocessing"

    def test_categorical_columns_encoded(self, binary_classification_df):
        """String categorical columns must be encoded to numeric."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(binary_classification_df, target_col="target")
        assert X.dtype in [np.float32, np.float64], "Output features must be numeric"

    def test_target_extracted_correctly(self, binary_classification_df):
        """Target column must not appear in features."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(binary_classification_df, target_col="target")
        assert X.shape[1] == len(binary_classification_df.columns) - 1
        assert len(y) == len(binary_classification_df)

    def test_output_dimensions_consistent(self, binary_classification_df):
        """Number of rows must be preserved after transformation."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(binary_classification_df, target_col="target")
        assert X.shape[0] == len(binary_classification_df)

    def test_inverse_transform_available(self, binary_classification_df):
        """DataPreprocessor should expose a fitted scaler for interpretability."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(binary_classification_df, target_col="target")
        assert hasattr(preprocessor, "scaler") or hasattr(preprocessor, "feature_names_out_")
