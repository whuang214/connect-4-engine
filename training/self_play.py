"""
Self-Play Training with Replay Buffer and Monte Carlo Returns.

Key differences from the old approach:
1. Monte Carlo returns instead of TD bootstrapping — targets are ACTUAL game
   outcomes, not noisy network estimates. Much more stable.
2. Replay buffer — stores (state, target) pairs from many games and samples
   random batches, breaking correlations between consecutive states.
3. Mixed opponents — trains against Random, RuleBased, and past versions of
   itself, preventing the self-play collapse problem.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from engine import Connect4
from models.value_network import encode_board


class ReplayBuffer:
    """
    Fixed-size buffer that stores (encoded_state, target_value) pairs.
    When full, oldest entries are dropped.
    Random sampling breaks temporal correlations between training examples.
    """

    def __init__(self, capacity=200_000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state: np.ndarray, target: float):
        self.buffer.append((state, target))

    def add_batch(self, states: list, targets: list):
        for s, t in zip(states, targets):
            self.buffer.append((s, t))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([b[0] for b in batch])
        targets = np.array([b[1] for b in batch], dtype=np.float32)
        return states, targets

    def __len__(self):
        return len(self.buffer)


def play_game_collect_data(agent_model, opponent, device, epsilon=0.1,
                           agent_is_p1=True):
    """
    Play one full game between the RL agent and an opponent.
    Collect all board states and compute Monte Carlo targets from the
    actual game outcome.

    The agent uses epsilon-greedy with agent_model.
    The opponent uses its own choose_action method.

    Returns:
        states: list of encoded board states (from current player's perspective)
        targets: list of float targets (+1 win, -1 loss, 0 draw)
        winner: game winner (1, 2, or None)
        num_moves: total moves in game
    """
    game = Connect4()
    positions = []  # List of (encoded_state, player_at_that_state)

    while not game.is_terminal():
        current_player = game.current_player
        is_agent_turn = (current_player == 1) == agent_is_p1

        # Record the board state from current player's perspective
        encoded = encode_board(game)
        positions.append((encoded, current_player))

        # Choose move
        if is_agent_turn:
            move = _agent_choose(agent_model, game, epsilon, device)
        else:
            move = opponent.choose_action(game)

        game.make_move(move)

    # Game is over — compute targets using actual outcome
    winner = game.winner
    states = []
    targets = []

    for encoded, player in positions:
        if winner is None:
            target = 0.0  # Draw
        elif winner == player:
            target = 1.0  # This player won
        else:
            target = -1.0  # This player lost

        states.append(encoded)
        targets.append(target)

    return states, targets, winner, len(positions)


def play_selfplay_collect_data(agent_model, device, epsilon=0.1):
    """
    Self-play: same model plays both sides with epsilon-greedy.
    Collects positions from BOTH players' perspectives.
    """
    game = Connect4()
    positions = []

    while not game.is_terminal():
        current_player = game.current_player
        encoded = encode_board(game)
        positions.append((encoded, current_player))

        move = _agent_choose(agent_model, game, epsilon, device)
        game.make_move(move)

    winner = game.winner
    states = []
    targets = []

    for encoded, player in positions:
        if winner is None:
            target = 0.0
        elif winner == player:
            target = 1.0
        else:
            target = -1.0

        states.append(encoded)
        targets.append(target)

    return states, targets, winner, len(positions)


def _agent_choose(model, game, epsilon, device):
    """Epsilon-greedy move selection using the value network."""
    legal_moves = game.get_legal_moves()

    if random.random() < epsilon:
        return random.choice(legal_moves)

    model.eval()
    best_value = float("-inf")
    best_move = legal_moves[0]

    with torch.no_grad():
        for move in legal_moves:
            temp_game = game.clone()
            result = temp_game.make_move(move)

            if result.winner == game.current_player:
                return move  # Instant win — always take it

            if result.draw:
                value = 0.0
            elif result.done:
                value = -1.0
            else:
                board_tensor = torch.tensor(
                    encode_board(temp_game), dtype=torch.float32
                ).unsqueeze(0).to(device)
                opp_value = model(board_tensor).item()
                value = -opp_value

            if value > best_value:
                best_value = value
                best_move = move

    return best_move


def train_on_batch(model, optimizer, replay_buffer, device, batch_size=512):
    """
    Sample a random batch from the replay buffer and do one gradient step.
    Returns the loss value.
    """
    if len(replay_buffer) < batch_size:
        return 0.0

    model.train()

    states, targets = replay_buffer.sample(batch_size)

    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)

    predictions = model(states_tensor)
    loss = F.mse_loss(predictions, targets_tensor)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()