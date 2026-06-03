from __future__ import annotations

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """MLP configurável para classificação binária de churn.

    Arquitetura por camada oculta:
      Linear → (BatchNorm → ReLU → Dropout)  quando dropout > 0
      Linear → ReLU                           quando dropout == 0

    A saída é um único logit (usar BCEWithLogitsLoss no treino,
    sigmoid na inferência).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list | None = None,
        dropout_rates: list | None = None,
    ) -> None:
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64, 32]
        dropout_rates = dropout_rates or [0.3, 0.2, 0.0]

        if len(hidden_sizes) != len(dropout_rates):
            raise ValueError("hidden_sizes e dropout_rates devem ter o mesmo tamanho")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim, drop in zip(hidden_sizes, dropout_rates, strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            if drop > 0:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if drop > 0:
                layers.append(nn.Dropout(drop))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
