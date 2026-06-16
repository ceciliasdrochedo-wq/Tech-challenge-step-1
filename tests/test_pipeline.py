from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.config import DataConfig, MLflowConfig, MLPConfig
from src.pipeline import ChurnPipeline


@patch("src.pipeline.MLflowService")
def test_pipeline_init(mock_mlflow_service):
    data_cfg = DataConfig()
    mlp_cfg = MLPConfig()
    mlflow_cfg = MLflowConfig(experiment_name="custom-exp")

    pipeline = ChurnPipeline(
        data_config=data_cfg,
        mlp_config=mlp_cfg,
        mlflow_config=mlflow_cfg,
        models_to_run=["dummy"],
        run_pytorch_mlp=False,
    )

    assert pipeline.data_cfg == data_cfg
    assert pipeline.mlp_cfg == mlp_cfg
    assert pipeline.mlflow_cfg == mlflow_cfg
    assert pipeline.models_to_run == ["dummy"]
    assert not pipeline.run_pytorch_mlp
    mock_mlflow_service.assert_called_once_with(
        tracking_uri=mlflow_cfg.tracking_uri, experiment_name="custom-exp"
    )


@patch("src.pipeline.ChurnDataLoader")
@patch("src.pipeline.MLflowService")
@patch("src.pipeline.SklearnTrainer")
@patch("src.pipeline.PyTorchMLPTrainer")
def test_pipeline_run(
    mock_pytorch_trainer_class,
    mock_sklearn_trainer_class,
    mock_mlflow_service_class,
    mock_data_loader_class,
):
    mock_loader = MagicMock()
    mock_data_loader_class.return_value = mock_loader
    mock_loader.load.return_value = pd.DataFrame({"dummy": [1, 2]})
    mock_loader.get_splits.return_value = (
        np.array([[1]]),
        np.array([[1]]),
        np.array([0]),
        np.array([0]),
        MagicMock(),
    )
    mock_loader.dataset_info.return_value = {"name": "dummy.csv"}

    mock_sklearn_trainer = MagicMock()
    mock_sklearn_trainer_class.return_value = mock_sklearn_trainer
    mock_sklearn_trainer.fit_evaluate.return_value = (MagicMock(), {"accuracy": 0.9})

    mock_pytorch_trainer = MagicMock()
    mock_pytorch_trainer_class.return_value = mock_pytorch_trainer
    mock_pytorch_trainer.fit_evaluate.return_value = (MagicMock(), {"accuracy": 0.8}, [0.1])
    mock_pytorch_trainer.params = {"lr": 0.001}

    pipeline = ChurnPipeline(models_to_run=["dummy"], run_pytorch_mlp=True)

    results = pipeline.run()

    mock_loader.load.assert_called_once()
    mock_loader.get_splits.assert_called_once()
    mock_sklearn_trainer.fit_evaluate.assert_called_once()
    mock_pytorch_trainer.fit_evaluate.assert_called_once()

    pipeline.mlflow.log_sklearn_run.assert_called_once()
    pipeline.mlflow.log_pytorch_run.assert_called_once()

    # Verify results output
    assert "dummy" in results
    assert "mlp_pytorch" in results
    assert results["dummy"]["accuracy"] == 0.9
    assert results["mlp_pytorch"]["accuracy"] == 0.8
