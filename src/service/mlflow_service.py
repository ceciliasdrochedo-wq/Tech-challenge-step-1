import logging
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn

logger = logging.getLogger(__name__)


class MLflowService:
    """Standardized MLflow logging for sklearn and PyTorch models.

    Decouples all tracking calls from training code so the same experiment
    structure is used regardless of which model or dataset is being logged.
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlruns.db",
        experiment_name: str = "default",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow → uri=%s | experiment=%s", tracking_uri, experiment_name)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_sklearn_run(
        self,
        run_name: str,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        dataset_info: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            self._log_common(metrics, params, dataset_info, tags)
            mlflow.sklearn.log_model(model, artifact_path="model")
            run_id = run.info.run_id
        logger.info("Logged sklearn run '%s' (run_id=%s)", run_name, run_id)
        return run_id

    def log_pytorch_run(
        self,
        run_name: str,
        model: Any,
        metrics: dict[str, float],
        params: dict[str, Any] | None = None,
        train_losses: list[float] | None = None,
        dataset_info: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            self._log_common(metrics, params, dataset_info, tags)
            if train_losses:
                for step, loss in enumerate(train_losses, start=1):
                    mlflow.log_metric("train_loss", loss, step=step)
            mlflow.pytorch.log_model(model, artifact_path="model")
            run_id = run.info.run_id
        logger.info("Logged PyTorch run '%s' (run_id=%s)", run_name, run_id)
        return run_id

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_common(
        self,
        metrics: dict[str, float],
        params: dict[str, Any] | None,
        dataset_info: dict[str, str] | None,
        tags: dict[str, str] | None,
    ) -> None:
        if params:
            mlflow.log_params(params)
        if dataset_info:
            mlflow.log_params({f"dataset_{k}": v for k, v in dataset_info.items()})
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(scalar_metrics)
        if tags:
            mlflow.set_tags(tags)
