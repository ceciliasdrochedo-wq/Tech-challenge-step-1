from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Cada métrica recebe (y_true, y_pred, y_prob) e retorna float | None.
METRICS_REGISTRY: dict[str, Callable] = {
    "accuracy": lambda yt, yp, _prob: accuracy_score(yt, yp),
    "f1_macro": lambda yt, yp, _prob: f1_score(yt, yp, average="macro"),
    "precision": lambda yt, yp, _prob: precision_score(yt, yp, zero_division=0),
    "recall": lambda yt, yp, _prob: recall_score(yt, yp, zero_division=0),
    "roc_auc": lambda yt, _yp, prob: roc_auc_score(yt, prob) if prob is not None else None,
}

DEFAULT_METRICS = list(METRICS_REGISTRY.keys())


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    metric_names: list | None = None,
) -> dict[str, float]:
    names = metric_names or DEFAULT_METRICS
    results: dict[str, float] = {}
    for name in names:
        if name not in METRICS_REGISTRY:
            logger.warning("Metric '%s' not found in METRICS_REGISTRY — skipping", name)
            continue
        value = METRICS_REGISTRY[name](y_true, y_pred, y_prob)
        if value is not None:
            results[name] = round(float(value), 4)
    return results


class SklearnTrainer:
    """Treina e avalia qualquer classificador compatível com sklearn."""

    def __init__(self, metric_names: list | None = None) -> None:
        self.metric_names = metric_names or DEFAULT_METRICS

    def fit_evaluate(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple:
        """Retorna (modelo_treinado, dicionário_de_métricas)."""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_prob: np.ndarray | None = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob, self.metric_names)
        logger.info("[%s] %s", type(model).__name__, metrics)
        return model, metrics
