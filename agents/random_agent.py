import random
from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):

    def __init__(self, name="RandomAgent"):
        super().__init__(name)

    def choose_action(self, game):
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return None

        return random.choice(legal_moves)