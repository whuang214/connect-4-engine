from __future__ import annotations

import os
import random
from typing import Optional

import torch

from agents.base_agent import BaseAgent
from models.policy_value_network import (
    PolicyValueNet,
    PolicyValueNetSmall,
    encode_board,
)


class RLPolicyAgent(BaseAgent):
    def __init__(
        self,
        name: str = "RLPolicyAgent",
        model=None,
        model_path: Optional[str] = None,
        epsilon: float = 0.0,
        temperature: float = 0.0,
        device=None,
        small_network: bool = False,
    ) -> None:
        super().__init__(name)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.epsilon = epsilon
        self.temperature = temperature

        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = self._load_model(path=model_path, small_network=small_network)
        else:
            self.model = PolicyValueNetSmall() if small_network else PolicyValueNet()

        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_model(self, path: str, small_network: bool = False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )

        model = None

        # Prefer saved config if present
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            cfg = checkpoint["config"]
            if cfg.get("small_network", False):
                model = PolicyValueNetSmall()
            else:
                # FIXED: use correct arg names matching PolicyValueNet.__init__
                # (channels + num_blocks, not num_filters + num_res_blocks)
                try:
                    model = PolicyValueNet(
                        channels=cfg.get("channels", 128),
                        num_blocks=cfg.get("num_blocks", 6),
                        dropout=cfg.get("dropout", 0.1),
                    )
                except TypeError:
                    model = PolicyValueNet()

        # Fallback if no config
        if model is None:
            if any(k.startswith("features.") for k in state_dict):
                model = PolicyValueNetSmall()
            else:
                model = PolicyValueNetSmall() if small_network else PolicyValueNet()

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def choose_action(self, game) -> int:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Tactical override: immediate win > immediate block > network move
        # Matches the training-time override so the agent plays consistently
        # with the tactical assumptions baked into its training data.
        win_move = self._find_immediate_win(game, game.current_player)
        if win_move is not None:
            return win_move

        opponent = 2 if game.current_player == 1 else 1
        block_move = self._find_immediate_win(game, opponent)
        if block_move is not None:
            return block_move

        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        state = torch.tensor(
            encode_board(game), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(state)
            logits = logits.squeeze(0)

        mask = torch.full_like(logits, -1e9)
        mask[legal_moves] = 0.0
        logits = logits + mask

        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=0)
            return int(torch.multinomial(probs, num_samples=1).item())

        return int(torch.argmax(logits).item())

    @staticmethod
    def _find_immediate_win(game, player) -> int | None:
        """Return a column that gives `player` an immediate win, or None."""
        center_order = [3, 2, 4, 1, 5, 0, 6]
        legal_moves = game.get_legal_moves()

        for col in center_order:
            if col not in legal_moves:
                continue
            if game.current_player == player:
                game.make_move(col)
                won = game.winner == player
                game.undo_move()
            else:
                tmp = game.clone()
                tmp.current_player = player
                tmp.make_move(col)
                won = tmp.winner == player
            if won:
                return col
        return None

    def evaluate_position(self, game) -> float:
        state = torch.tensor(
            encode_board(game), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            _, value = self.model(state)

        return float(value.item())

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def save_model(self, path: str, optimizer=None, episode=None, metadata=None) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

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