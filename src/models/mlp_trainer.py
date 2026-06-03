from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import ChurnMLP
from src.models.trainer import DEFAULT_METRICS, compute_metrics

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Para o treino quando a loss de validação para de melhorar."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0

    @property
    def should_stop(self) -> bool:
        return self._counter >= self.patience

    def step(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self.should_stop


class PyTorchMLPTrainer:
    """Treina o ChurnMLP com early stopping e callbacks por época."""

    def __init__(
        self,
        hidden_sizes: list | None = None,
        dropout_rates: list | None = None,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        random_state: int = 42,
        metric_names: list | None = None,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [128, 64, 32]
        self.dropout_rates = dropout_rates or [0.3, 0.2, 0.0]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.metric_names = metric_names or DEFAULT_METRICS

        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("PyTorchMLPTrainer — dispositivo: %s", self.device)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epoch_callback=None,
    ) -> tuple:
        """Treina, aplica early stopping e avalia.

        Retorna (modelo, dicionário_de_métricas, losses_de_treino).
        epoch_callback(epoch, train_loss, val_loss) é chamado a cada época.
        """
        # Separa 10% do treino para validação do early stopping
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=self.random_state,
            stratify=y_train,
        )

        model = ChurnMLP(X_fit.shape[1], self.hidden_sizes, self.dropout_rates).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        stopper = EarlyStopping(patience=self.early_stopping_patience)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_fit, dtype=torch.float32),
                torch.tensor(y_fit, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        X_te_t = torch.tensor(X_test, dtype=torch.float32)

        train_losses: list[float] = []
        stopped_epoch = self.epochs

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            if epoch_callback:
                epoch_callback(epoch, avg_train_loss, val_loss)

            if stopper.step(val_loss):
                stopped_epoch = epoch
                logger.info("Early stopping na época %d/%d", epoch, self.epochs)
                break

        logger.info("Treino concluído — épocas rodadas: %d/%d", stopped_epoch, self.epochs)

        model.eval()
        with torch.no_grad():
            logits = model(X_te_t.to(self.device)).cpu()
            y_prob = torch.sigmoid(logits).numpy()
            y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob, self.metric_names)
        metrics["stopped_epoch"] = stopped_epoch
        logger.info("[MLP_PyTorch] %s", metrics)
        return model, metrics, train_losses

    @property
    def params(self) -> dict:
        return {
            "architecture": "-".join(str(h) for h in self.hidden_sizes),
            "optimizer": "Adam",
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }
