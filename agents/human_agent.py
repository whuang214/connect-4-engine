from agents.base_agent import BaseAgent


class HumanAgent(BaseAgent):
    def __init__(self, name="Human"):
        super().__init__(name)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.total_prompts = 0
        self.invalid_text_inputs = 0
        self.illegal_move_attempts = 0
        self.moves_chosen = 0

    def choose_action(self, game) -> int:
        while True:
            legal_moves = game.get_legal_moves()
            print(f"Legal moves: {legal_moves}")

            self.total_prompts += 1
            user_input = input(f"{self.name}, choose a column: ")

            try:
                move = int(user_input)
            except ValueError:
                self.invalid_text_inputs += 1
                print("Please enter a valid integer column.")
                continue

            if move not in legal_moves:
                self.illegal_move_attempts += 1
                print("That move is not legal. Try again.")
                continue

            self.moves_chosen += 1
            return move

    def get_stats(self) -> dict:
        return {
            "Interaction": {
                "Total prompts": self.total_prompts,
                "Invalid text inputs": self.invalid_text_inputs,
                "Illegal move attempts": self.illegal_move_attempts,
                "Moves chosen": self.moves_chosen,
            }
        }