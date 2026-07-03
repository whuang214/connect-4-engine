"""Shared fixtures and helpers for the connect4 test suite.

No test here needs pygame or a trained checkpoint: RL agents are built from a
randomly initialised PolicyValueNetSmall on CPU.
"""

from __future__ import annotations

import pytest

from connect4.engine import Connect4

# A scripted 42-move game that fills the board with no winner.
# Verified against the engine: every move is legal and the game ends in a draw.
DRAW_SEQUENCE = [
    4, 5, 1, 4, 3, 3, 5, 2, 0, 6, 4, 0, 0, 5, 6, 3, 5, 6, 6, 5, 5,
    0, 4, 3, 2, 1, 6, 2, 6, 0, 1, 0, 2, 2, 4, 2, 1, 4, 3, 1, 3, 1,
]


def play_moves(game: Connect4, cols) -> None:
    """Apply a sequence of column moves to a game."""
    for col in cols:
        game.make_move(col)


@pytest.fixture
def game() -> Connect4:
    return Connect4()


@pytest.fixture
def draw_sequence() -> list[int]:
    return list(DRAW_SEQUENCE)
