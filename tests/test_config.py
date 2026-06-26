import logging
from pathlib import Path

from src.config import DataConfig, MLflowConfig, MLPConfig, setup_logging


def test_data_config_defaults():
    config = DataConfig()
    assert config.target == "Churn"
    assert config.test_size == 0.2
    assert config.random_state == 42
    assert isinstance(config.path, Path)
    assert config.path.name == "churn.csv"


def test_mlp_config_defaults():
    config = MLPConfig()
    assert config.hidden_sizes == [128, 64, 32]
    assert config.dropout_rates == [0.3, 0.2, 0.0]
    assert config.batch_size == 256
    assert config.epochs == 100
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.early_stopping_patience == 10
    assert config.random_state == 42


def test_mlflow_config_defaults():
    config = MLflowConfig()
    assert config.tracking_uri == "sqlite:///mlruns.db"
    assert config.experiment_name == "telco-churn-etapa2"


def test_setup_logging():
    setup_logging(level="DEBUG")
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG
    logger.setLevel(logging.INFO)
