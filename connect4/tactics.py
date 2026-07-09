"""Shared tactical helpers: immediate win/block detection and move ordering.

These were previously duplicated across the MCTS, RL, and rule-based agents.
"""

from __future__ import annotations

CENTER_ORDER: tuple[int, ...] = (3, 2, 4, 1, 5, 0, 6)
"""Columns ordered by distance from the center — strongest-first scan order."""


def ordered_legal_moves(game) -> list[int]:
    """The game's legal moves, reordered center-first."""
    legal = game.get_legal_moves()
    return [col for col in CENTER_ORDER if col in legal]


def find_immediate_win(game, player) -> int | None:
    """Return a column that gives ``player`` an immediate win, or None.

    Never corrupts ``game``: when it is already ``player``'s turn the probe
    uses make_move/undo_move on the live object (fast path — no copying);
    otherwise it simulates on a clone with ``current_player`` overridden.
    """
    legal_moves = game.get_legal_moves()

    for move in CENTER_ORDER:
        if move not in legal_moves:
            continue

        if game.current_player == player:
            game.make_move(move)
            is_win = game.winner == player
            game.undo_move()
        else:
            tmp = game.clone()
            tmp.current_player = player
            tmp.make_move(move)
            is_win = tmp.winner == player

        if is_win:
            return move

    return None
