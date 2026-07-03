"""Tests for the shared tactical helpers in connect4.tactics."""

from __future__ import annotations

from connect4.engine import Connect4
from connect4.tactics import CENTER_ORDER, find_immediate_win, ordered_legal_moves
from conftest import play_moves


def test_finds_win_when_it_is_the_players_turn(game):
    # P1 has (5,0), (5,1), (5,2) and it is P1's turn.
    play_moves(game, [0, 0, 1, 1, 2, 2])
    assert game.current_player == Connect4.PLAYER1
    assert find_immediate_win(game, Connect4.PLAYER1) == 3


def test_finds_win_when_it_is_not_the_players_turn(game):
    # P2 has (5,3), (5,4), (5,5) but it is P1's turn: the probe must use the
    # clone path with current_player overridden. Both 2 and 6 win; the
    # center-first scan order reaches 2 first.
    play_moves(game, [0, 3, 0, 4, 0, 5])
    assert game.current_player == Connect4.PLAYER1
    assert find_immediate_win(game, Connect4.PLAYER2) == 2


def test_returns_none_when_no_win_exists(game):
    play_moves(game, [3, 3])
    assert find_immediate_win(game, Connect4.PLAYER1) is None
    assert find_immediate_win(game, Connect4.PLAYER2) is None


def test_probe_does_not_mutate_the_game(game):
    play_moves(game, [0, 0, 1, 1, 2, 2])
    state_before = game.get_state()
    history_len_before = len(game.move_history)

    find_immediate_win(game, Connect4.PLAYER1)  # live make_move/undo path
    find_immediate_win(game, Connect4.PLAYER2)  # clone path

    assert game.get_state() == state_before
    assert len(game.move_history) == history_len_before


def test_ordered_legal_moves_is_center_first(game):
    assert ordered_legal_moves(game) == list(CENTER_ORDER)
    # Fill column 3 completely: it must disappear from the ordering.
    play_moves(game, [3] * 6)
    assert ordered_legal_moves(game) == [2, 4, 1, 5, 0, 6]
