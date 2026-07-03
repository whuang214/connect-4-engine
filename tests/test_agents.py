"""Behavioural tests shared across all agent implementations.

The RL agent runs a randomly initialised PolicyValueNetSmall on CPU — no
trained checkpoint is required (its win/block behaviour comes from the
tactical override, not the network).
"""

from __future__ import annotations

import pytest
import torch

from connect4.agents.base import BaseAgent
from connect4.agents.mcts import MCTSAgent
from connect4.agents.minimax import MinimaxAgent
from connect4.agents.random import RandomAgent
from connect4.agents.rl_policy import RLPolicyAgent
from connect4.agents.rule_based import RuleBasedAgent
from connect4.engine import Connect4
from connect4.models.policy_value_network import PolicyValueNetSmall
from conftest import DRAW_SEQUENCE, play_moves


def _make_rl_agent():
    torch.manual_seed(0)
    return RLPolicyAgent(
        name="rl-small-random",
        model=PolicyValueNetSmall(),
        device=torch.device("cpu"),
    )


AGENT_FACTORIES = {
    "random": RandomAgent,
    "rule": RuleBasedAgent,
    "minimax": lambda: MinimaxAgent(depth=2),
    "mcts": lambda: MCTSAgent(iterations=50),
    "rl": _make_rl_agent,
}

# RandomAgent has no tactical awareness, so exclude it from win/block tests.
TACTICAL_FACTORIES = {k: v for k, v in AGENT_FACTORIES.items() if k != "random"}


@pytest.fixture(params=sorted(AGENT_FACTORIES), name="any_agent")
def _any_agent(request):
    return AGENT_FACTORIES[request.param]()


@pytest.fixture(params=sorted(TACTICAL_FACTORIES), name="tactical_agent")
def _tactical_agent(request):
    return TACTICAL_FACTORIES[request.param]()


def test_returns_legal_move_on_fresh_board(any_agent, game):
    move = any_agent.choose_action(game)
    assert move in game.get_legal_moves()


def test_returns_the_only_legal_move(any_agent, game):
    play_moves(game, DRAW_SEQUENCE[:41])  # one empty cell left
    legal = game.get_legal_moves()
    assert len(legal) == 1
    assert any_agent.choose_action(game) == legal[0]


def test_picks_immediate_win(tactical_agent, game):
    # P1 to move with (5,0), (5,1), (5,2): the only winning column is 3.
    play_moves(game, [0, 0, 1, 1, 2, 2])
    assert game.current_player == Connect4.PLAYER1
    assert tactical_agent.choose_action(game) == 3


def test_picks_immediate_block(tactical_agent, game):
    # P2 has stacked three in column 3; P1 must block there.
    play_moves(game, [0, 3, 1, 3, 0, 3])
    assert game.current_player == Connect4.PLAYER1
    assert tactical_agent.choose_action(game) == 3


def test_stats_contract(any_agent, game):
    any_agent.reset_stats()
    any_agent.choose_action(game)
    stats = any_agent.get_stats()
    assert isinstance(stats, dict)


def test_minimax_is_a_base_agent():
    assert isinstance(MinimaxAgent(depth=2), BaseAgent)


def test_minimax_name_default_and_override():
    assert MinimaxAgent(depth=3).name == "minimax-3"
    assert MinimaxAgent(depth=3, name="Custom").name == "Custom"
