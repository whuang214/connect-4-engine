"""
Value Network for Connect 4 RL Agent.

Input:  (batch, 3, 6, 7) float tensor
    Channel 0: current player's pieces
    Channel 1: opponent's pieces
    Channel 2: all 1s if current player is Player 1, all 0s if Player 2

Output: (batch, 1) float tensor in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueNetwork(nn.Module):
    def __init__(self, num_filters=128, num_res_blocks=4):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)
        return self.value_head(x)


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ValueNetworkSmall(nn.Module):
    """Smaller network for ablation comparison."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.value_head(self.features(x))


def encode_board(game) -> np.ndarray:
    """
    Encode board from the CURRENT player's perspective.
    Channel 0 = my pieces, Channel 1 = opponent pieces, Channel 2 = turn indicator.
    """
    board = np.array(game.board, dtype=np.float32)
    current = game.current_player
    opponent = 2 if current == 1 else 1

    my_pieces = (board == current).astype(np.float32)
    opp_pieces = (board == opponent).astype(np.float32)
    turn_plane = np.full((6, 7), float(current == 1), dtype=np.float32)

    return np.stack([my_pieces, opp_pieces, turn_plane], axis=0)