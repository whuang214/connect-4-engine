from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    def __init__(self, name="RuleBasedAgent"):
        super().__init__(name)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.moves_chosen = 0
        self.rule1_immediate_win = 0
        self.rule2_immediate_block = 0
        self.rule3_center = 0
        self.rule4_center_preference = 0
        self.fallback_count = 0

    def choose_action(self, game) -> int:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        my_player = game.current_player
        opponent = 1 if my_player == 2 else 2

        # Rule 1: If we can win immediately, do it.
        for move in legal_moves:
            temp_game = game.clone()
            result = temp_game.make_move(move)
            if result.winner == my_player:
                self.moves_chosen += 1
                self.rule1_immediate_win += 1
                return move

        # Rule 2: If opponent can win next move, block it.
        for move in legal_moves:
            temp_game = game.clone()
            temp_game.current_player = opponent
            result = temp_game.make_move(move)
            if result.winner == opponent:
                self.moves_chosen += 1
                self.rule2_immediate_block += 1
                return move

        # Rule 3: Prefer center column if available.
        center_col = game.COLS // 2
        if center_col in legal_moves:
            self.moves_chosen += 1
            self.rule3_center += 1
            return center_col

        # Rule 4: Prefer columns closer to center.
        preferred_order = self.get_center_preferred_order(game.COLS)

        for move in preferred_order:
            if move in legal_moves:
                self.moves_chosen += 1
                self.rule4_center_preference += 1
                return move

        # Fallback
        self.moves_chosen += 1
        self.fallback_count += 1
        return legal_moves[0]

    def get_center_preferred_order(self, cols: int) -> list[int]:
        """
        Returns columns ordered by closeness to center.
        For 7 columns, this gives: [3, 2, 4, 1, 5, 0, 6]
        """
        center = cols // 2
        order = [center]

        for offset in range(1, cols):
            left = center - offset
            right = center + offset

            if left >= 0:
                order.append(left)
            if right < cols:
                order.append(right)

        return order

    def get_stats(self) -> dict:
        return {
            "Decisions": {
                "Moves chosen": self.moves_chosen,
            },
            "Rule Usage": {
                "Rule 1 immediate wins": self.rule1_immediate_win,
                "Rule 2 immediate blocks": self.rule2_immediate_block,
                "Rule 3 center plays": self.rule3_center,
                "Rule 4 center-preference plays": self.rule4_center_preference,
                "Fallback plays": self.fallback_count,
            }
        }