from __future__ import annotations

import logging

from src.config import DataConfig, MLflowConfig, MLPConfig
from src.data.loader import ChurnDataLoader
from src.models.mlp_trainer import PyTorchMLPTrainer
from src.models.registry import MODEL_REGISTRY, PYTORCH_REGISTRY
from src.models.trainer import SklearnTrainer
from src.service.mlflow_service import MLflowService

logger = logging.getLogger(__name__)


class ChurnPipeline:
    """Orchestrates the full ML pipeline: load → preprocess → train → log.

    Usage:
        pipeline = ChurnPipeline()
        results = pipeline.run()

    Or override any config:
        pipeline = ChurnPipeline(
            mlflow_config=MLflowConfig(experiment_name="my-experiment"),
            models_to_run=["logistic_regression", "random_forest"],
        )
    """

    def __init__(
        self,
        data_config: DataConfig | None = None,
        mlp_config: MLPConfig | None = None,
        mlflow_config: MLflowConfig | None = None,
        models_to_run: list | None = None,
        run_pytorch_mlp: bool = True,
    ) -> None:
        self.data_cfg = data_config or DataConfig()
        self.mlp_cfg = mlp_config or MLPConfig()
        self.mlflow_cfg = mlflow_config or MLflowConfig()
        self.models_to_run = models_to_run or list(MODEL_REGISTRY.keys())
        self.run_pytorch_mlp = run_pytorch_mlp

        self.mlflow = MLflowService(
            tracking_uri=self.mlflow_cfg.tracking_uri,
            experiment_name=self.mlflow_cfg.experiment_name,
        )
        self.sklearn_trainer = SklearnTrainer()
        self.pytorch_trainer = PyTorchMLPTrainer(
            hidden_sizes=self.mlp_cfg.hidden_sizes,
            dropout_rates=self.mlp_cfg.dropout_rates,
            epochs=self.mlp_cfg.epochs,
            batch_size=self.mlp_cfg.batch_size,
            learning_rate=self.mlp_cfg.learning_rate,
            weight_decay=self.mlp_cfg.weight_decay,
            early_stopping_patience=self.mlp_cfg.early_stopping_patience,
            random_state=self.mlp_cfg.random_state,
        )

    def run(self) -> dict:
        """Run the full pipeline and return a dict of {model_name: metrics}."""
        loader = ChurnDataLoader(
            path=self.data_cfg.path,
            target=self.data_cfg.target,
            test_size=self.data_cfg.test_size,
            random_state=self.data_cfg.random_state,
        )
        df = loader.load()
        X_train, X_test, y_train, y_test, _preprocessor = loader.get_splits(df)
        dataset_info = loader.dataset_info()

        results: dict[str, dict] = {}

        # ── Sklearn models ──────────────────────────────────────────────
        for name in self.models_to_run:
            if name not in MODEL_REGISTRY:
                logger.warning("'%s' not found in MODEL_REGISTRY — skipping", name)
                continue

            entry = MODEL_REGISTRY[name]
            model, metrics = self.sklearn_trainer.fit_evaluate(
                entry["model"], X_train, y_train, X_test, y_test
            )
            mlflow_meta = entry.get("mlflow", {})
            self.mlflow.log_sklearn_run(
                run_name=name,
                model=model,
                metrics=metrics,
                params=entry["params"],
                dataset_info=dataset_info,
                tags={"stage": "etapa2", "model_family": "sklearn"},
                register=bool(mlflow_meta),
                **mlflow_meta,
            )
            results[name] = metrics

        # ── PyTorch MLP ─────────────────────────────────────────────────
        if self.run_pytorch_mlp:
            def _epoch_cb(epoch: int, train_loss: float, val_loss: float) -> None:
                if epoch % 10 == 0:
                    logger.info(
                        "Epoch %3d | train_loss=%.4f | val_loss=%.4f",
                        epoch, train_loss, val_loss,
                    )

            model_pt, metrics_pt, train_losses = self.pytorch_trainer.fit_evaluate(
                X_train, y_train, X_test, y_test, epoch_callback=_epoch_cb
            )
            pytorch_meta = PYTORCH_REGISTRY.get("MLP_PyTorch", {}).get("mlflow", {})
            self.mlflow.log_pytorch_run(
                run_name="MLP_PyTorch",
                model=model_pt,
                metrics=metrics_pt,
                params=self.pytorch_trainer.params,
                train_losses=train_losses,
                dataset_info=dataset_info,
                tags={"stage": "etapa2", "model_family": "pytorch"},
                register=bool(pytorch_meta),
                **pytorch_meta,
            )
            results["mlp_pytorch"] = metrics_pt

        logger.info("Pipeline complete — %d models logged", len(results))
        return results
