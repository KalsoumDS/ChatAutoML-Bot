"""
Unit tests for ChatAutoML model zoo and evaluation.
"""
from __future__ import annotations

import numpy as np
import pytest
from automl.model_zoo import get_models
from automl.evaluation import evaluate_model
from sklearn.datasets import make_classification


class TestModelZoo:

    def test_get_models_returns_dict(self):
        """get_models() must return a non-empty dictionary of estimators."""
        models = get_models(task="classification")
        assert isinstance(models, dict)
        assert len(models) >= 3, "At least 3 models expected in the zoo"

    def test_all_models_have_fit_predict(self):
        """Every model in the zoo must implement fit() and predict()."""
        models = get_models(task="classification")
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} must have a fit() method"
            assert hasattr(model, "predict"), f"{name} must have a predict() method"

    def test_models_train_on_small_dataset(self):
        """All models must train without error on a 100-sample dataset."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        models = get_models(task="classification")
        for name, model in models.items():
            try:
                model.fit(X, y)
                preds = model.predict(X)
                assert len(preds) == len(y), f"{name}: prediction length mismatch"
            except Exception as e:
                pytest.fail(f"{name} failed to train/predict: {e}")


class TestEvaluateModel:

    def test_evaluate_returns_dict_with_metrics(self):
        """evaluate_model must return a dict containing 'accuracy' or 'f1'."""
        from sklearn.ensemble import RandomForestClassifier
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = evaluate_model(model, X, y, task="classification")
        assert isinstance(result, dict)
        assert any(k in result for k in ["accuracy", "f1", "f1_score"])

    def test_accuracy_in_valid_range(self):
        """Model accuracy must be in [0, 1]."""
        from sklearn.ensemble import RandomForestClassifier
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        result = evaluate_model(model, X, y, task="classification")
        accuracy = result.get("accuracy", result.get("f1", 0))
        assert 0.0 <= accuracy <= 1.0
