"""
minimax_agent.py
"""


"""
Thought process:
- Build general structure
- Find best heuristic function, etc.
- Add improvements/refinements based on evaluation

"""

from __future__ import annotations

from typing import Optional
from engine import Connect4


class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning for Connect-4.

    Parameters
    ----------
    player : int
        The player this agent controls (Connect4.PLAYER1 or PLAYER2).
    depth : int
        Maximum search depth. Higher = stronger but slower.
        Recommended: 5–6 for a strong agent without extra optimizations.
    """

    # Scores for terminal states — must dominate all heuristic scores
    WIN_SCORE  =  1_000_000
    LOSS_SCORE = -1_000_000

    def __init__(self, player: int, depth: int = 5) -> None:
        self.player = player
        self.opponent = Connect4.PLAYER2 if player == Connect4.PLAYER1 else Connect4.PLAYER1
        self.depth = depth
        self.name = f"minimax-{depth}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def choose_action(self, game: Connect4) -> int:
        """
        Return the column index of the best move for this agent.
        Call this on the agent's turn with the live game object.
        """
        best_col = -1
        best_score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for col in self._ordered_moves(game):
            game.make_move(col)
            score = self._minimax(game, self.depth - 1, alpha, beta, is_maximizing=False)
            game.undo_move()

            if score > best_score:
                best_score = score
                best_col   = col

            alpha = max(alpha, best_score)

        return best_col

    # ------------------------------------------------------------------
    # Minimax with alpha-beta pruning
    # ------------------------------------------------------------------

    def _minimax(
        self,
        game: Connect4,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
    ) -> float:
        # --- Terminal / depth-limit check ---
        if game.is_terminal():
            if game.winner == self.player:
                return self.WIN_SCORE + depth   # prefer faster wins
            if game.winner == self.opponent:
                return self.LOSS_SCORE - depth  # prefer slower losses
            return 0                            # draw

        if depth == 0:
            return self._evaluate(game.board)

        moves = self._ordered_moves(game)

        if is_maximizing:
            value = float("-inf")
            for col in moves:
                game.make_move(col)
                value = max(value, self._minimax(game, depth - 1, alpha, beta, False))
                game.undo_move()
                alpha = max(alpha, value)
                if value >= beta:
                    break   # beta cut-off
            return value
        else:
            value = float("inf")
            for col in moves:
                game.make_move(col)
                value = min(value, self._minimax(game, depth - 1, alpha, beta, True))
                game.undo_move()
                beta = min(beta, value)
                if value <= alpha:
                    break   # alpha cut-off
            return value

    # ------------------------------------------------------------------
    # Move ordering — center columns first for better pruning
    # ------------------------------------------------------------------

    def _ordered_moves(self, game: Connect4) -> list[int]:
        """
        Sort legal moves by proximity to the center column.
        Exploring stronger moves first maximises alpha-beta cut-offs.
        """
        center = Connect4.COLS // 2
        return sorted(game.get_legal_moves(), key=lambda c: abs(c - center))

    # ------------------------------------------------------------------
    # Heuristic board evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, board: list[list[int]]) -> int:
        """
        Score the board from the agent's perspective.
        Scans every window of 4 cells across all directions.
        """
        score = 0

        # Center column bonus — center pieces are more flexible
        center_col = Connect4.COLS // 2
        center_cells = [board[r][center_col] for r in range(Connect4.ROWS)]
        score += center_cells.count(self.player) * 3

        # Horizontal windows
        for r in range(Connect4.ROWS):
            for c in range(Connect4.COLS - 3):
                window = [board[r][c + i] for i in range(4)]
                score  += self._score_window(window)

        # Vertical windows
        for c in range(Connect4.COLS):
            for r in range(Connect4.ROWS - 3):
                window = [board[r + i][c] for i in range(4)]
                score  += self._score_window(window)

        # Diagonal (down-right)
        for r in range(Connect4.ROWS - 3):
            for c in range(Connect4.COLS - 3):
                window = [board[r + i][c + i] for i in range(4)]
                score  += self._score_window(window)

        # Diagonal (down-left)
        for r in range(Connect4.ROWS - 3):
            for c in range(3, Connect4.COLS):
                window = [board[r + i][c - i] for i in range(4)]
                score  += self._score_window(window)

        return score

    def _score_window(self, window: list[int]) -> int:
        """
        Score a single window of 4 cells.

        Positive scores favour the agent; negative scores favour the opponent.
        A window is only scored if it doesn't contain both players' pieces
        (i.e. it is still "live" / winnable).
        """
        agent_count    = window.count(self.player)
        opp_count      = window.count(self.opponent)
        empty_count    = window.count(Connect4.EMPTY)

        # Mixed window — neither side can win through it
        if agent_count > 0 and opp_count > 0:
            return 0

        if agent_count == 4:
            return 100          # shouldn't be reached; handled as terminal
        if agent_count == 3 and empty_count == 1:
            return 5            # one away from winning
        if agent_count == 2 and empty_count == 2:
            return 2            # two in a row with space

        if opp_count == 3 and empty_count == 1:
            return -4           # block opponent's winning threat
        if opp_count == 2 and empty_count == 2:
            return -1

        return 0


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    game  = Connect4()
    agent = MinimaxAgent(player=Connect4.PLAYER2, depth=5)

    game.render()

    while not game.is_terminal():
        if game.current_player == Connect4.PLAYER1:
            # Human move
            col = int(input("Your move (0-6): "))
        else:
            # Agent move
            col = agent.choose_action(game)
            print(f"Agent plays column {col}")

        try:
            game.make_move(col)
        except ValueError as e:
            print(e)
            continue

        game.render()
