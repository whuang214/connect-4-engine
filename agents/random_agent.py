import random
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, name="RandomAgent"):
        super().__init__(name)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.moves_chosen = 0

    def choose_action(self, game) -> int:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        self.moves_chosen += 1
        return random.choice(legal_moves)

    def get_stats(self) -> dict:
        return {
            "Decisions": {
                "Moves chosen": self.moves_chosen,
            }
        }