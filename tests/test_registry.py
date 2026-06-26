from src.models.registry import MODEL_REGISTRY, PYTORCH_REGISTRY


def test_model_registry_keys():
    expected_keys = {
        "dummy",
        "logistic_regression",
        "ridge",
        "decision_tree",
        "random_forest",
        "mlp_sklearn",
    }
    assert expected_keys.issubset(MODEL_REGISTRY.keys())


def test_model_registry_entries():
    for _name, entry in MODEL_REGISTRY.items():
        assert "model" in entry
        assert "params" in entry
        estimator = entry["model"]
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "predict")


def test_pytorch_registry():
    assert "MLP_PyTorch" in PYTORCH_REGISTRY
    assert "mlflow" in PYTORCH_REGISTRY["MLP_PyTorch"]
