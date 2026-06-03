from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.pytorch
import mlflow.sklearn
from mlflow.tracking import MlflowClient

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
        register: bool = False,
        model_description: str | None = None,
        version_tags: dict[str, str] | None = None,
        version_description: str | None = None,
        version_alias: str | None = None,
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            self._log_common(metrics, params, dataset_info, tags)
            mlflow.sklearn.log_model(
                model,
                artifact_path=run_name,
                registered_model_name=run_name if register else None,
            )
            run_id = run.info.run_id
        if register:
            self._configure_registered_version(
                run_name, run_id,
                model_description, version_tags, version_description, version_alias,
            )
        logger.info("Logged sklearn run '%s' (run_id=%s, registered=%s)", run_name, run_id, register)
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
        register: bool = False,
        model_description: str | None = None,
        version_tags: dict[str, str] | None = None,
        version_description: str | None = None,
        version_alias: str | None = None,
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            self._log_common(metrics, params, dataset_info, tags)
            if train_losses:
                for step, loss in enumerate(train_losses, start=1):
                    mlflow.log_metric("train_loss", loss, step=step)
            mlflow.pytorch.log_model(
                model,
                artifact_path=run_name,
                registered_model_name=run_name if register else None,
            )
            run_id = run.info.run_id
        if register:
            self._configure_registered_version(
                run_name, run_id,
                model_description, version_tags, version_description, version_alias,
            )
        logger.info("Logged PyTorch run '%s' (run_id=%s, registered=%s)", run_name, run_id, register)
        return run_id

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _configure_registered_version(
        self,
        model_name: str,
        run_id: str,
        model_description: str | None,
        version_tags: dict[str, str] | None,
        version_description: str | None,
        version_alias: str | None,
    ) -> None:
        client = MlflowClient()

        versions = client.search_model_versions(f"run_id='{run_id}' and name='{model_name}'")
        if not versions:
            logger.warning("No registered version found for run_id=%s model=%s", run_id, model_name)
            return
        version = versions[0].version

        if model_description:
            client.update_registered_model(model_name, description=model_description)

        if version_description:
            client.update_model_version(model_name, version, description=version_description)

        if version_tags:
            for key, value in version_tags.items():
                client.set_model_version_tag(model_name, version, key, str(value))

        if version_alias:
            client.set_registered_model_alias(model_name, version_alias, version)

        logger.info(
            "Configured version %s of '%s' — alias=%s tags=%s",
            version, model_name, version_alias, version_tags,
        )

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
