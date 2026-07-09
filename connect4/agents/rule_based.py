from connect4.agents.base import BaseAgent
from connect4.tactics import find_immediate_win, ordered_legal_moves


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
        move = find_immediate_win(game, my_player)
        if move is not None:
            self.moves_chosen += 1
            self.rule1_immediate_win += 1
            return move

        # Rule 2: If opponent can win next move, block it.
        move = find_immediate_win(game, opponent)
        if move is not None:
            self.moves_chosen += 1
            self.rule2_immediate_block += 1
            return move

        # Rules 3 + 4: prefer the center column, then columns closest to it.
        ordered = ordered_legal_moves(game)
        if ordered:
            move = ordered[0]
            self.moves_chosen += 1
            if move == game.COLS // 2:
                self.rule3_center += 1
            else:
                self.rule4_center_preference += 1
            return move

        # Fallback (unreachable while legal_moves is non-empty)
        self.moves_chosen += 1
        self.fallback_count += 1
        return legal_moves[0]

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