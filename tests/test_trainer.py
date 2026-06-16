import logging

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from src.models.trainer import SklearnTrainer, compute_metrics


def test_compute_metrics_all():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.3, 0.4, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "roc_auc" in metrics

    assert metrics["accuracy"] == pytest.approx(0.8333, abs=1e-3)


def test_compute_metrics_custom_and_invalid(caplog):
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    with caplog.at_level(logging.WARNING):
        metrics = compute_metrics(y_true, y_pred, metric_names=["accuracy", "non_existent"])

    assert "accuracy" in metrics
    assert "non_existent" not in metrics
    assert any("non_existent" in record.message for record in caplog.records)


def test_sklearn_trainer():
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 3], [6, 7]])
    y_test = np.array([0, 1])

    trainer = SklearnTrainer(metric_names=["accuracy"])
    model = DummyClassifier(strategy="most_frequent")

    trained_model, metrics = trainer.fit_evaluate(model, X_train, y_train, X_test, y_test)
    assert trained_model is model
    assert "accuracy" in metrics
    # Only accuracy should be computed
    assert list(metrics.keys()) == ["accuracy"]
