"""
Reinforcement Learning Agent for Connect 4.
Uses a trained value network to evaluate positions.
"""

import os
import random
import numpy as np
import torch

from agents.base_agent import BaseAgent
from models.value_network import ValueNetwork, ValueNetworkSmall, encode_board


class RLAgent(BaseAgent):
    def __init__(self, name="RLAgent", model=None, model_path=None,
                 epsilon=0.0, device=None, small_network=False):
        super().__init__(name)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.epsilon = epsilon

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(model_path, small_network)
        else:
            self.model = ValueNetworkSmall() if small_network else ValueNetwork()

        self.model = self.model.to(self.device)

    def _load_model(self, path, small_network=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        # Determine architecture from saved config, then key names, then caller hint
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            cfg = checkpoint["config"]
            if cfg.get("small_network"):
                model = ValueNetworkSmall()
            else:
                model = ValueNetwork(cfg.get("num_filters", 128), cfg.get("num_res_blocks", 4))
        elif any(k.startswith("features.") for k in state_dict):
            model = ValueNetworkSmall()
        else:
            model = ValueNetworkSmall() if small_network else ValueNetwork()

        model.load_state_dict(state_dict)
        model.eval()
        return model

    def choose_action(self, game) -> int:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        return self._greedy_action(game, legal_moves)

    def _greedy_action(self, game, legal_moves) -> int:
        """Evaluate each legal move and pick the best one."""
        self.model.eval()
        best_value = float("-inf")
        best_move = legal_moves[0]

        with torch.no_grad():
            for move in legal_moves:
                temp_game = game.clone()
                result = temp_game.make_move(move)

                if result.winner is not None and result.winner == game.current_player:
                    return move  # Instant win

                if result.draw:
                    value = 0.0
                elif result.done:
                    value = -1.0
                else:
                    board_tensor = torch.tensor(
                        encode_board(temp_game), dtype=torch.float32
                    ).unsqueeze(0).to(self.device)
                    opp_value = self.model(board_tensor).item()
                    value = -opp_value

                if value > best_value:
                    best_value = value
                    best_move = move

        return best_move

    def evaluate_position(self, game) -> float:
        self.model.eval()
        with torch.no_grad():
            board_tensor = torch.tensor(
                encode_board(game), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            return self.model(board_tensor).item()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def save_model(self, path, optimizer=None, episode=None, metadata=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {"model_state_dict": self.model.state_dict()}
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if episode is not None:
            checkpoint["episode"] = episode
        if metadata is not None:
            checkpoint["metadata"] = metadata
        torch.save(checkpoint, path)