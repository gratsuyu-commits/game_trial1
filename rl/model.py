import torch
import torch.nn as nn


class DQN(nn.Module):
    """FC 128 → 128 → 64 → output  (Double DQN shared backbone)."""

    def __init__(self, input_dim: int = 24, output_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
