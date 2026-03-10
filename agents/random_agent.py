import random
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):

    def __init__(self, name="RandomAgent"):
        super().__init__(name)

    def choose_action(self, game) -> int:
        legal_moves = game.get_legal_moves()
        return random.choice(legal_moves)