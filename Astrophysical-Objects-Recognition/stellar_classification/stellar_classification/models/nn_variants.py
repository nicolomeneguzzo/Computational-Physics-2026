"""Additional neural network architectures for stellar classification."""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Medium Complexity Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class MediumNN(nn.Module):
    """
    Medium-depth fully connected neural network.

    Architecture:
        input → 128 → 64 → output
    """

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Linear(input_size, 128),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(128, 64),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(64, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Complex Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class ComplexNN(nn.Module):
    """
    Deeper fully connected neural network.

    Architecture:
        input → 512 → 256 → 128 → 64 → output
    """

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.0):
        super().__init__()

        layers = [
            nn.Linear(input_size, 512),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(512, 256),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(256, 128),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers += [
            nn.Linear(128, 64),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(64, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)