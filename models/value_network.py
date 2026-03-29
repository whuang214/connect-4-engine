"""
Value Network for Connect 4 RL Agent.

Architecture: CNN with 3-channel input (current player pieces, opponent pieces, turn indicator)
Output: single scalar in [-1, 1] estimating position value for the current player.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ValueNetwork(nn.Module):
    """
    A convolutional neural network that estimates the value of a Connect 4
    board position from the current player's perspective.

    Input:  (batch, 3, 6, 7) float tensor
        Channel 0: current player's pieces (1 where piece exists, 0 otherwise)
        Channel 1: opponent's pieces
        Channel 2: all 1s if current player is Player 1, all 0s if Player 2
                   (provides context about who is moving)

    Output: (batch, 1) float tensor in [-1, 1]
        +1 = winning position, -1 = losing position, 0 = even
    """

    def __init__(self, num_filters=128, num_res_blocks=4):
        super().__init__()

        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks

        # Initial convolution
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual blocks for deeper feature extraction
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x):
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)
        return self.value_head(x)


class ResidualBlock(nn.Module):
    """Standard residual block with two conv layers and a skip connection."""

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
        out = out + residual
        return F.relu(out)


def encode_board(game) -> np.ndarray:
    """
    Convert a Connect4 game state into a 3-channel numpy array
    from the perspective of the CURRENT player.

    This is critical: the network always sees itself as 'channel 0'
    and the opponent as 'channel 1', regardless of which player number
    it actually is. This means we only need ONE network for both sides.

    Args:
        game: Connect4 game instance

    Returns:
        np.ndarray of shape (3, 6, 7) with float32 values
    """
    board = np.array(game.board, dtype=np.float32)
    current = game.current_player
    opponent = 2 if current == 1 else 1

    # Channel 0: current player's pieces
    my_pieces = (board == current).astype(np.float32)

    # Channel 1: opponent's pieces
    opp_pieces = (board == opponent).astype(np.float32)

    # Channel 2: turn indicator (all 1s if Player 1, all 0s if Player 2)
    # This gives the network awareness of first-move advantage
    turn_plane = np.full((6, 7), float(current == 1), dtype=np.float32)

    return np.stack([my_pieces, opp_pieces, turn_plane], axis=0)


def encode_board_batch(games) -> torch.Tensor:
    """Encode a list of game states into a batched tensor."""
    boards = [encode_board(g) for g in games]
    return torch.tensor(np.array(boards), dtype=torch.float32)


# ----- Small network variant for ablation studies -----

class ValueNetworkSmall(nn.Module):
    """
    Smaller network for ablation comparison.
    Uses fewer filters and no residual connections.
    """

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
        x = self.features(x)
        return self.value_head(x)