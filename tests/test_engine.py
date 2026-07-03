"""Tests for the core Connect4 engine: win detection, legality, undo, clone."""

from __future__ import annotations

import random

import pytest

from connect4.engine import Connect4
from conftest import play_moves


class TestWinDetection:
    def test_horizontal_win(self, game):
        # P1 builds cols 0-3 on the bottom row.
        play_moves(game, [0, 0, 1, 1, 2, 2, 3])
        assert game.done
        assert game.winner == Connect4.PLAYER1

    def test_vertical_win_at_left_edge(self, game):
        # P1 stacks column 0 four times.
        play_moves(game, [0, 1, 0, 1, 0, 1, 0])
        assert game.done
        assert game.winner == Connect4.PLAYER1

    def test_diagonal_up_right_win(self, game):
        # P1 pieces at (5,0), (4,1), (3,2), (2,3).
        play_moves(game, [0, 1, 1, 2, 2, 3, 2, 3, 3, 5, 3])
        assert game.done
        assert game.winner == Connect4.PLAYER1

    def test_diagonal_down_right_win(self, game):
        # Mirror image: P1 pieces at (5,6), (4,5), (3,4), (2,3).
        play_moves(game, [6, 5, 5, 4, 4, 3, 4, 3, 3, 1, 3])
        assert game.done
        assert game.winner == Connect4.PLAYER1

    def test_horizontal_win_ending_at_right_edge(self, game):
        # P1 builds cols 3-6 on the bottom row.
        play_moves(game, [3, 0, 4, 0, 5, 1, 6])
        assert game.done
        assert game.winner == Connect4.PLAYER1

    def test_vertical_win_reaching_top_row(self, game):
        # P2 fills rows 5-4 of column 0, then P1 stacks rows 3-0.
        play_moves(game, [1, 0, 1, 0, 0, 2, 0, 2, 0, 3, 0])
        assert game.done
        assert game.winner == Connect4.PLAYER1
        assert game.last_move == (0, 0)  # winning piece in the top row

    def test_three_in_a_row_is_not_a_win(self, game):
        play_moves(game, [0, 0, 1, 1, 2])
        assert not game.done
        assert game.winner is None


class TestIllegalMoves:
    def test_full_column_raises(self, game):
        play_moves(game, [0] * 6)  # alternating pieces, no win
        assert not game.done
        with pytest.raises(ValueError):
            game.make_move(0)

    @pytest.mark.parametrize("col", [-1, 7, 100])
    def test_out_of_range_raises(self, game, col):
        with pytest.raises(ValueError):
            game.make_move(col)

    def test_move_after_game_over_raises(self, game):
        play_moves(game, [0, 0, 1, 1, 2, 2, 3])  # P1 wins
        assert game.done
        with pytest.raises(ValueError):
            game.make_move(5)


class TestDraw:
    def test_scripted_42_move_draw(self, game, draw_sequence):
        assert len(draw_sequence) == 42
        for col in draw_sequence[:-1]:
            result = game.make_move(col)
            assert not result.done
        result = game.make_move(draw_sequence[-1])
        assert result.done
        assert result.draw
        assert result.winner is None
        assert game.done
        assert game.winner is None
        assert game.get_legal_moves() == []


class TestUndo:
    def test_undo_round_trip_restores_every_snapshot(self, game):
        rng = random.Random(1234)
        snapshots = []
        for _ in range(15):
            snapshots.append(game.get_state())
            game.make_move(rng.choice(game.get_legal_moves()))
        for _ in range(15):
            game.undo_move()
            assert game.get_state() == snapshots.pop()
        assert game.move_history == []

    def test_undo_winning_move_restores_live_game(self, game):
        play_moves(game, [0, 0, 1, 1, 2, 2])
        before = game.get_state()
        game.make_move(3)  # P1 wins
        assert game.done and game.winner == Connect4.PLAYER1
        game.undo_move()
        assert game.get_state() == before
        assert not game.done
        assert game.winner is None
        assert game.current_player == Connect4.PLAYER1
        # The game is playable again.
        assert game.get_legal_moves() == list(range(7))

    def test_undo_with_no_history_raises(self, game):
        with pytest.raises(ValueError):
            game.undo_move()


class TestClone:
    def test_mutating_clone_leaves_original_untouched(self, game):
        play_moves(game, [3, 3])
        original_state = game.get_state()
        clone = game.clone()
        clone.make_move(0)
        clone.make_move(4)
        assert game.get_state() == original_state

    def test_mutating_original_leaves_clone_untouched(self, game):
        play_moves(game, [3, 3])
        clone = game.clone()
        clone_state = clone.get_state()
        game.make_move(0)
        game.make_move(4)
        assert clone.get_state() == clone_state
