"""
Reinforcement Learning Agent for Connect 4.

Uses a trained value network to evaluate board positions.
During training: epsilon-greedy exploration.
During evaluation: purely greedy (pick the highest-value move).
"""

import os
import random
import numpy as np
import torch

from agents.base_agent import BaseAgent
from models.value_network import ValueNetwork, ValueNetworkSmall, encode_board


class RLAgent(BaseAgent):
    """
    An agent that uses a neural network value function to play Connect 4.

    At decision time, it simulates each legal move, encodes the resulting
    board state, evaluates it with the value network, and picks the move
    leading to the best position (from the opponent's perspective, negated).
    """

    def __init__(
        self,
        name="RLAgent",
        model=None,
        model_path=None,
        epsilon=0.0,
        device=None,
        small_network=False,
    ):
        super().__init__(name)

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.epsilon = epsilon  # Exploration rate (0.0 = greedy)

        # Load or create model
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path, small_network)
        else:
            if small_network:
                self.model = ValueNetworkSmall()
            else:
                self.model = ValueNetwork()

        self.model = self.model.to(self.device)

    def _load_model(self, path, small_network=False):
        """Load a trained model from a checkpoint file."""
        if small_network:
            model = ValueNetworkSmall()
        else:
            model = ValueNetwork()

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Support both raw state_dict and full checkpoint dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def choose_action(self, game) -> int:
        """
        Choose a column to play.

        With probability epsilon, pick a random legal move (exploration).
        Otherwise, evaluate all legal moves and pick the best one (exploitation).
        """
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        return self._greedy_action(game, legal_moves)

    def _greedy_action(self, game, legal_moves) -> int:
        """
        Evaluate each legal move by simulating it and running the resulting
        board through the value network. Pick the move with the highest value.
        """
        self.model.eval()
        best_value = float("-inf")
        best_move = legal_moves[0]

        with torch.no_grad():
            for move in legal_moves:
                # Simulate the move
                temp_game = game.clone()
                result = temp_game.make_move(move)

                if result.winner is not None:
                    # We just won — this is the best possible move
                    return move

                if result.draw:
                    # Draw is neutral
                    value = 0.0
                else:
                    # Evaluate from the NEXT player's perspective, then negate.
                    # After our move, it's the opponent's turn.
                    # The network evaluates from current player's POV.
                    # So the opponent's value is the negative of ours.
                    board_tensor = torch.tensor(
                        encode_board(temp_game),
                        dtype=torch.float32,
                    ).unsqueeze(0).to(self.device)

                    opp_value = self.model(board_tensor).item()
                    value = -opp_value  # Negate: opponent's gain is our loss

                if value > best_value:
                    best_value = value
                    best_move = move

        return best_move

    def evaluate_position(self, game) -> float:
        """
        Get the raw value estimate for the current board position.
        Useful for debugging and analysis.
        """
        self.model.eval()
        with torch.no_grad():
            board_tensor = torch.tensor(
                encode_board(game),
                dtype=torch.float32,
            ).unsqueeze(0).to(self.device)
            return self.model(board_tensor).item()

    def set_epsilon(self, epsilon):
        """Update exploration rate (used during training)."""
        self.epsilon = epsilon

    def save_model(self, path, optimizer=None, episode=None, metadata=None):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if episode is not None:
            checkpoint["episode"] = episode
        if metadata is not None:
            checkpoint["metadata"] = metadata
        torch.save(checkpoint, path)