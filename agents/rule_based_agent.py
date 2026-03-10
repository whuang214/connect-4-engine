from agents.base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    def __init__(self, name="RuleBasedAgent"):
        super().__init__(name)

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
                return move

        # Rule 2: If opponent can win next move, block it.
        for move in legal_moves:
            temp_game = game.clone()
            temp_game.current_player = opponent
            result = temp_game.make_move(move)
            if result.winner == opponent:
                return move

        # Rule 3: Prefer center column if available.
        center_col = game.COLS // 2
        if center_col in legal_moves:
            return center_col

        # Rule 4: Prefer columns closer to center.
        preferred_order = self.get_center_preferred_order(game.COLS)

        for move in preferred_order:
            if move in legal_moves:
                return move

        # Fallback (should almost never matter)
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