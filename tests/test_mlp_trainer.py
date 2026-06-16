import numpy as np

from src.models.mlp import ChurnMLP
from src.models.mlp_trainer import EarlyStopping, PyTorchMLPTrainer


def test_early_stopping_behavior():
    stopper = EarlyStopping(patience=3, min_delta=0.01)
    assert not stopper.should_stop

    assert not stopper.step(0.5)
    assert not stopper.step(0.4)
    assert not stopper.step(0.399)
    assert not stopper.step(0.41)
    assert stopper.step(0.42)
    assert stopper.should_stop


def test_mlp_trainer_init_and_params():
    trainer = PyTorchMLPTrainer(
        hidden_sizes=[32, 16], dropout_rates=[0.1, 0.0], epochs=5, batch_size=16, learning_rate=0.01
    )
    assert trainer.epochs == 5
    assert trainer.batch_size == 16

    params = trainer.params
    assert params["architecture"] == "32-16"
    assert params["lr"] == 0.01
    assert params["max_epochs"] == 5


def test_mlp_trainer_fit_evaluate():
    np.random.seed(42)
    X_train = np.random.randn(100, 10).astype(np.float32)
    y_train = np.random.randint(0, 2, size=100).astype(np.float32)
    X_test = np.random.randn(20, 10).astype(np.float32)
    y_test = np.random.randint(0, 2, size=20).astype(np.float32)

    trainer = PyTorchMLPTrainer(
        hidden_sizes=[16, 8],
        dropout_rates=[0.0, 0.0],
        epochs=3,
        batch_size=32,
        early_stopping_patience=2,
        random_state=42,
    )

    epochs_called = []

    def epoch_callback(epoch, train_loss, val_loss):
        epochs_called.append(epoch)

    model, metrics, train_losses = trainer.fit_evaluate(
        X_train, y_train, X_test, y_test, epoch_callback=epoch_callback
    )

    assert isinstance(model, ChurnMLP)
    assert len(epochs_called) > 0
    assert len(train_losses) == len(epochs_called)
    assert "accuracy" in metrics
    assert "stopped_epoch" in metrics
    assert "f1_macro" in metrics
