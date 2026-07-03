"""Tests for the head-to-head evaluation runner using tiny scripted agents."""

from __future__ import annotations

import pytest

from connect4.agents.base import BaseAgent
from connect4.engine import Connect4
from connect4.evaluation.evaluate import evaluate_agents, play_one_game


class FirstLegalAgent(BaseAgent):
    """Always plays the lowest-numbered legal column. Fully deterministic:
    whoever moves first wins by completing row 5 across columns 0-3."""

    def choose_action(self, game) -> int:
        return game.get_legal_moves()[0]


class IllegalMoveAgent(BaseAgent):
    def choose_action(self, game) -> int:
        return 99


def _run(num_games: int):
    return evaluate_agents(
        game_class=Connect4,
        agent1=FirstLegalAgent(name="Alpha"),
        agent2=FirstLegalAgent(name="Beta"),
        num_games=num_games,
        print_each_game=False,
    )


def test_alternates_first_player_across_games():
    summary = _run(4)
    assert [g.player1_agent for g in summary.game_results] == [
        "Alpha", "Beta", "Alpha", "Beta",
    ]
    assert [g.player2_agent for g in summary.game_results] == [
        "Beta", "Alpha", "Beta", "Alpha",
    ]


def test_win_draw_counts_add_up():
    summary = _run(6)
    assert summary.total_games == 6
    assert len(summary.game_results) == 6
    assert summary.agent1_wins + summary.agent2_wins + summary.draws == 6
    # With this deterministic script, player 1 always wins.
    assert summary.draws == 0
    assert summary.agent1_wins == 3
    assert summary.agent2_wins == 3


def test_p1_p2_split_sums_to_totals():
    summary = _run(5)
    assert summary.agent1_as_p1_wins + summary.agent1_as_p2_wins == summary.agent1_wins
    assert summary.agent2_as_p1_wins + summary.agent2_as_p2_wins == summary.agent2_wins
    # Odd game count: agent1 starts 3 games, agent2 starts 2.
    assert summary.agent1_as_p1_wins == 3
    assert summary.agent2_as_p1_wins == 2
    assert summary.agent1_as_p2_wins == 0
    assert summary.agent2_as_p2_wins == 0


def test_move_totals_are_consistent():
    summary = _run(4)
    total_moves = sum(g.moves_played for g in summary.game_results)
    assert summary.total_moves_agent1 + summary.total_moves_agent2 == total_moves
    for g in summary.game_results:
        assert g.move_count_p1 + g.move_count_p2 == g.moves_played


def test_illegal_move_raises_value_error():
    with pytest.raises(ValueError, match="Illegal move"):
        play_one_game(
            game_class=Connect4,
            player1_agent=IllegalMoveAgent(name="Cheater"),
            player2_agent=FirstLegalAgent(name="Honest"),
        )
