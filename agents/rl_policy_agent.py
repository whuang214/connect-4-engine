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
                # If your PolicyValueNet supports custom args, use them.
                # Otherwise just use PolicyValueNet()
                try:
                    model = PolicyValueNet(
                        num_filters=cfg.get("num_filters", 128),
                        num_res_blocks=cfg.get("num_res_blocks", 4),
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