import numpy as np
import pandas as pd
import pytest

from src.data.loader import ChurnDataLoader


@pytest.fixture
def sample_csv_path(tmp_path):
    df = pd.DataFrame(
        {
            "customerID": ["1", "2", "3", "4", "5"],
            "gender": ["Male", "Female", "Male", "Female", "Male"],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 89.10],
            "TotalCharges": ["29.85", "56.95", " ", "169.2", "445.5"],
            "Churn": ["No", "No", "Yes", "No", "Yes"],
        }
    )
    csv_file = tmp_path / "test_churn.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


def test_loader_init(sample_csv_path):
    loader = ChurnDataLoader(path=sample_csv_path, target="Churn", test_size=0.2, random_state=42)
    assert loader.path == sample_csv_path
    assert loader.target == "Churn"
    assert loader.test_size == 0.2
    assert loader.random_state == 42


def test_loader_load_and_clean(sample_csv_path):
    loader = ChurnDataLoader(path=sample_csv_path)
    df = loader.load()

    assert "customerID" not in df.columns

    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])

    assert len(df) == 4

    assert set(df["Churn"]).issubset({0, 1})


def test_loader_get_splits(sample_csv_path):
    df_large = pd.DataFrame(
        {
            "customerID": [str(i) for i in range(10)],
            "gender": ["Male", "Female"] * 5,
            "MonthlyCharges": [10.0 * i for i in range(10)],
            "TotalCharges": [10.0 * i for i in range(10)],
            "Churn": ["No", "Yes"] * 5,
        }
    )
    csv_file = sample_csv_path.parent / "large_churn.csv"
    df_large.to_csv(csv_file, index=False)

    loader = ChurnDataLoader(path=csv_file, test_size=0.2, random_state=42)
    df = loader.load()
    X_train, X_test, y_train, y_test, preprocessor = loader.get_splits(df)

    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32


def test_loader_dataset_info(sample_csv_path):
    loader = ChurnDataLoader(path=sample_csv_path)
    info = loader.dataset_info()
    assert info["name"] == "test_churn.csv"
    assert "sha256" in info
    assert "size_bytes" in info
    assert info["path"] == str(sample_csv_path)
