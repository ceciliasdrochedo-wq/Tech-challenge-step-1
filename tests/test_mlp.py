import pytest
import torch

from src.models.mlp import ChurnMLP


def test_mlp_initialization_defaults():
    model = ChurnMLP(input_dim=10)
    assert model.net is not None
    last_layer = list(model.net.children())[-1]
    assert isinstance(last_layer, torch.nn.Linear)
    assert last_layer.out_features == 1


def test_mlp_initialization_custom():
    model = ChurnMLP(input_dim=10, hidden_sizes=[64, 32], dropout_rates=[0.1, 0.0])
    assert model.net is not None
    linear_layers = [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]
    assert len(linear_layers) == 3
    assert linear_layers[0].out_features == 64
    assert linear_layers[1].out_features == 32
    assert linear_layers[2].out_features == 1


def test_mlp_initialization_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="hidden_sizes e dropout_rates devem ter o mesmo tamanho"):
        ChurnMLP(input_dim=10, hidden_sizes=[64, 32], dropout_rates=[0.1])


def test_mlp_forward_pass():
    batch_size = 4
    input_dim = 15
    model = ChurnMLP(input_dim=input_dim, hidden_sizes=[32], dropout_rates=[0.2])
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    assert output.shape == (batch_size,)
