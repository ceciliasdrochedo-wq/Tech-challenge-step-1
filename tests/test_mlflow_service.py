from unittest.mock import MagicMock, patch

from src.service.mlflow_service import MLflowService


@patch("src.service.mlflow_service.mlflow")
def test_mlflow_service_init(mock_mlflow):
    service = MLflowService(tracking_uri="mock_uri", experiment_name="mock_exp")
    mock_mlflow.set_tracking_uri.assert_called_once_with("mock_uri")
    mock_mlflow.set_experiment.assert_called_once_with("mock_exp")
    assert service.tracking_uri == "mock_uri"
    assert service.experiment_name == "mock_exp"


@patch("src.service.mlflow_service.mlflow")
def test_log_sklearn_run(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id_123"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    service = MLflowService(tracking_uri="mock_uri", experiment_name="mock_exp")

    mock_model = MagicMock()
    metrics = {"accuracy": 0.95}
    params = {"C": 1.0}
    dataset_info = {"name": "churn.csv", "sha256": "abc"}
    tags = {"env": "test"}

    run_id = service.log_sklearn_run(
        run_name="my_sklearn_model",
        model=mock_model,
        metrics=metrics,
        params=params,
        dataset_info=dataset_info,
        tags=tags,
        register=False,
    )

    assert run_id == "test_run_id_123"
    mock_mlflow.log_params.assert_any_call(params)
    mock_mlflow.log_params.assert_any_call({"dataset_name": "churn.csv", "dataset_sha256": "abc"})
    mock_mlflow.log_metrics.assert_called_once_with({"accuracy": 0.95})
    mock_mlflow.set_tags.assert_called_once_with(tags)
    mock_mlflow.sklearn.log_model.assert_called_once_with(
        mock_model, artifact_path="my_sklearn_model", registered_model_name=None
    )


@patch("src.service.mlflow_service.mlflow")
def test_log_pytorch_run(mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id_456"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    service = MLflowService(tracking_uri="mock_uri", experiment_name="mock_exp")

    mock_model = MagicMock()
    metrics = {"loss": 0.1}
    params = {"lr": 0.001}
    train_losses = [0.5, 0.3, 0.1]

    run_id = service.log_pytorch_run(
        run_name="my_pytorch_model",
        model=mock_model,
        metrics=metrics,
        params=params,
        train_losses=train_losses,
        register=False,
    )

    assert run_id == "test_run_id_456"
    mock_mlflow.log_params.assert_called_once_with(params)
    mock_mlflow.log_metrics.assert_called_once_with({"loss": 0.1})
    mock_mlflow.pytorch.log_model.assert_called_once_with(
        mock_model, artifact_path="my_pytorch_model", registered_model_name=None
    )
    assert mock_mlflow.log_metric.call_count == 3
    mock_mlflow.log_metric.assert_any_call("train_loss", 0.5, step=1)
    mock_mlflow.log_metric.assert_any_call("train_loss", 0.3, step=2)
    mock_mlflow.log_metric.assert_any_call("train_loss", 0.1, step=3)


@patch("src.service.mlflow_service.MlflowClient")
@patch("src.service.mlflow_service.mlflow")
def test_configure_registered_version(mock_mlflow, mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_version = MagicMock()
    mock_version.version = "2"
    mock_client.search_model_versions.return_value = [mock_version]

    service = MLflowService(tracking_uri="mock_uri", experiment_name="mock_exp")
    service._configure_registered_version(
        model_name="my_model",
        run_id="run_123",
        model_description="Model Description",
        version_tags={"stage": "production"},
        version_description="Version Description",
        version_alias="challenger",
    )

    mock_client.update_registered_model.assert_called_once_with(
        "my_model", description="Model Description"
    )
    mock_client.update_model_version.assert_called_once_with(
        "my_model", "2", description="Version Description"
    )
    mock_client.set_model_version_tag.assert_called_once_with(
        "my_model", "2", "stage", "production"
    )
    mock_client.set_registered_model_alias.assert_called_once_with("my_model", "challenger", "2")
