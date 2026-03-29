"""
Self-Play Training Loop for Connect 4 RL Agent.

Uses Temporal Difference (TD) learning with self-play.
Both players share the same value network. After each move,
we compute a TD target and accumulate gradients.

TD update rule:
    - If game ongoing:  target = -V(s_{t+1})   (negated because opponent's perspective)
    - If game ended:    target = actual reward   (+1 win, -1 loss, 0 draw)
    - Loss = MSE(V(s_t), target)
"""

import random
import numpy as np
import torch
import torch.nn.functional as F

from engine import Connect4
from models.value_network import encode_board


def play_training_episode(model, optimizer, device, epsilon=0.1, gamma=1.0):
    """
    Play one full game of self-play and update the model after each move
    using TD(0) learning.

    Both sides use the same model with epsilon-greedy exploration.

    Args:
        model: the ValueNetwork
        optimizer: torch optimizer
        device: torch device
        epsilon: exploration rate
        gamma: discount factor (1.0 is standard for finite games)

    Returns:
        dict with episode statistics:
            - winner: 1, 2, or None (draw)
            - num_moves: total moves in the game
            - total_loss: sum of TD losses over the episode
    """
    game = Connect4()
    model.train()

    states = []     # Encoded board states collected during the game
    total_loss = 0.0
    num_moves = 0

    while not game.is_terminal():
        # Encode current state BEFORE the move
        state_before = encode_board(game)
        current_player = game.current_player

        # Choose action: epsilon-greedy
        legal_moves = game.get_legal_moves()

        if random.random() < epsilon:
            action = random.choice(legal_moves)
        else:
            action = _greedy_action(model, game, legal_moves, device)

        # Evaluate current state
        state_tensor = torch.tensor(
            state_before, dtype=torch.float32
        ).unsqueeze(0).to(device)
        value_pred = model(state_tensor)  # V(s_t)

        # Make the move
        result = game.make_move(action)
        num_moves += 1

        # Compute TD target
        if result.done:
            # Game is over - use actual reward
            if result.winner == current_player:
                target = 1.0
            elif result.winner is not None:
                target = -1.0
            else:
                target = 0.0  # Draw
            target_tensor = torch.tensor(
                [[target]], dtype=torch.float32
            ).to(device)
        else:
            # Game ongoing - bootstrap from next state
            # The next state is from the OPPONENT's perspective (it's their turn now)
            # V_opponent(s') estimates how good s' is for the opponent
            # Our value of s' is -V_opponent(s')
            with torch.no_grad():
                next_state = encode_board(game)
                next_tensor = torch.tensor(
                    next_state, dtype=torch.float32
                ).unsqueeze(0).to(device)
                next_value = model(next_tensor)
                target_tensor = -gamma * next_value  # Negate for opponent

        # TD update
        loss = F.mse_loss(value_pred, target_tensor)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return {
        "winner": game.winner,
        "num_moves": num_moves,
        "total_loss": total_loss,
        "avg_loss": total_loss / max(num_moves, 1),
    }


def _greedy_action(model, game, legal_moves, device):
    """Pick the legal move with the highest estimated value."""
    model.eval()
    best_value = float("-inf")
    best_move = legal_moves[0]

    with torch.no_grad():
        for move in legal_moves:
            temp_game = game.clone()
            result = temp_game.make_move(move)

            if result.winner == game.current_player:
                model.train()
                return move  # Instant win

            if result.draw:
                value = 0.0
            elif result.done:
                value = -1.0  # Opponent won somehow (shouldn't happen)
            else:
                board_tensor = torch.tensor(
                    encode_board(temp_game), dtype=torch.float32
                ).unsqueeze(0).to(device)
                opp_value = model(board_tensor).item()
                value = -opp_value

            if value > best_value:
                best_value = value
                best_move = move

    model.train()
    return best_move


def play_training_episode_batched(model, optimizer, device, epsilon=0.1, gamma=1.0):
    """
    Alternative: collect all transitions in an episode, then do a single
    batched update at the end. Can be more stable for some configurations.

    Returns same stats dict as play_training_episode.
    """
    game = Connect4()
    model.eval()

    transitions = []  # List of (state_encoded, target) tuples

    while not game.is_terminal():
        state_before = encode_board(game)
        current_player = game.current_player

        legal_moves = game.get_legal_moves()

        if random.random() < epsilon:
            action = random.choice(legal_moves)
        else:
            action = _greedy_action(model, game, legal_moves, device)

        result = game.make_move(action)

        # We'll fill in the target after the game ends for terminal states,
        # and use bootstrap for intermediate states
        transitions.append({
            "state": state_before,
            "player": current_player,
            "result": result,
        })

    # Now compute targets
    states_list = []
    targets_list = []

    for i, t in enumerate(transitions):
        states_list.append(t["state"])

        if t["result"].done:
            # Terminal state - use actual outcome
            if t["result"].winner == t["player"]:
                targets_list.append(1.0)
            elif t["result"].winner is not None:
                targets_list.append(-1.0)
            else:
                targets_list.append(0.0)
        else:
            # Non-terminal - bootstrap from next state
            # The next state in the game is from the opponent's perspective
            next_state = transitions[i + 1]["state"] if i + 1 < len(transitions) else None
            if next_state is not None:
                with torch.no_grad():
                    next_tensor = torch.tensor(
                        next_state, dtype=torch.float32
                    ).unsqueeze(0).to(device)
                    model.eval()
                    next_value = model(next_tensor).item()
                targets_list.append(-gamma * next_value)
            else:
                targets_list.append(0.0)

    # Batched update
    if states_list:
        model.train()
        states_tensor = torch.tensor(
            np.array(states_list), dtype=torch.float32
        ).to(device)
        targets_tensor = torch.tensor(
            targets_list, dtype=torch.float32
        ).unsqueeze(1).to(device)

        predictions = model(states_tensor)
        loss = F.mse_loss(predictions, targets_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss = loss.item()
    else:
        total_loss = 0.0

    return {
        "winner": game.winner,
        "num_moves": len(transitions),
        "total_loss": total_loss,
        "avg_loss": total_loss,
    }