import logging
import logging.config
from dataclasses import dataclass, field
from pathlib import Path

RANDOM_STATE = 42

_PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DataConfig:
    path: Path = field(default_factory=lambda: _PROJECT_ROOT / "src" / "data" / "churn.csv")
    target: str = "Churn"
    test_size: float = 0.2
    random_state: int = RANDOM_STATE


@dataclass
class MLPConfig:
    hidden_sizes: list = field(default_factory=lambda: [128, 64, 32])
    dropout_rates: list = field(default_factory=lambda: [0.3, 0.2, 0.0])
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    random_state: int = RANDOM_STATE


@dataclass
class MLflowConfig:
    # Relative URI — works because notebooks do os.chdir(ROOT) before any MLflow call.
    # Avoids Windows backslash issues with absolute paths in sqlite:/// URIs.
    tracking_uri: str = "sqlite:///mlruns.db"
    experiment_name: str = "telco-churn-etapa2"


def setup_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "structured",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": level, "handlers": ["console"]},
        }
    )
