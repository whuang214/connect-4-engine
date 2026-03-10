from agents.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    def __init__(self, name="Human"):
        super().__init__(name)

    def choose_action(self, game):
        while True:
            legal_moves = game.get_legal_moves()
            print(f"Legal moves: {legal_moves}")

            user_input = input(f"{self.name}, choose a column: ")

            try:
                move = int(user_input)
            except ValueError:
                print("Please enter a valid integer column.")
                continue

            if move not in legal_moves:
                print("That move is not legal. Try again.")
                continue

            return move