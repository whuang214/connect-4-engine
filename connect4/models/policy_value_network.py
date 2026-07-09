from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return F.relu(out, inplace=True)


class PolicyValueNet(nn.Module):
    """
    Residual CNN with two heads:
    - policy logits over 7 columns
    - value in [-1, 1]
    """

    def __init__(self, channels: int = 128, num_blocks: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * 6 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32 * 6 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class PolicyValueNetSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 7),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        return self.policy_head(x), self.value_head(x)


def encode_board(game) -> np.ndarray:
    """
    Encode from the current player's perspective.

    Channels:
    0 = current player's pieces
    1 = opponent pieces
    2 = side-to-move plane (1 if player 1 to move else 0)
    3 = normalized height map repeated by row
    """
    board = np.array(game.board, dtype=np.int8)
    current = game.current_player
    opponent = 2 if current == 1 else 1

    my_pieces = (board == current).astype(np.float32)
    opp_pieces = (board == opponent).astype(np.float32)
    turn_plane = np.full((6, 7), float(current == 1), dtype=np.float32)

    heights = np.zeros((7,), dtype=np.float32)
    for c in range(7):
        filled = np.count_nonzero(board[:, c] != 0)
        heights[c] = filled / 6.0
    height_plane = np.tile(heights, (6, 1)).astype(np.float32)

    return np.stack([my_pieces, opp_pieces, turn_plane, height_plane], axis=0)


def mirror_encoded_state(state: np.ndarray) -> np.ndarray:
    return state[:, :, ::-1].copy()


def mirror_action(action: int) -> int:
    return 6 - action