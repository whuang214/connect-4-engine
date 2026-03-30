from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class MoveResult:
    row: int
    col: int
    player: int
    winner: Optional[int]
    done: bool
    draw: bool


@dataclass
class MoveHistory:
    row: int
    col: int
    player: int
    previous_winner: Optional[int]
    previous_done: bool
    previous_last_move: Optional[Tuple[int, int]]


class Connect4:
    ROWS = 6
    COLS = 7

    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset the game to the starting position."""
        self.board: List[List[int]] = [
            [self.EMPTY for _ in range(self.COLS)] for _ in range(self.ROWS)
        ]
        self.current_player: int = self.PLAYER1
        self.winner: Optional[int] = None
        self.done: bool = False
        self.last_move: Optional[Tuple[int, int]] = None
        self.move_history: List[MoveHistory] = []

    def clone(self) -> "Connect4":
        """Return a deep copy of the game state."""
        new_game = Connect4()
        new_game.board = deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.winner = self.winner
        new_game.done = self.done
        new_game.last_move = self.last_move
        new_game.move_history = deepcopy(self.move_history)
        return new_game

    def get_state(self) -> dict:
        """Return a snapshot of the current game state."""
        return {
            "board": deepcopy(self.board),
            "current_player": self.current_player,
            "winner": self.winner,
            "done": self.done,
            "last_move": self.last_move,
        }

    def get_legal_moves(self) -> List[int]:
        """Return a list of columns where a piece can still be dropped."""
        if self.done:
            return []
        return [col for col in range(self.COLS) if self.board[0][col] == self.EMPTY]

    def is_legal_move(self, col: int) -> bool:
        """Check whether a move is legal."""
        return (
            0 <= col < self.COLS
            and not self.done
            and self.board[0][col] == self.EMPTY
        )

    def make_move(self, col: int) -> MoveResult:
        """
        Drop a piece into the given column for the current player.

        Returns a MoveResult describing the outcome of the move.
        Raises ValueError if the move is illegal.
        """
        if not self.is_legal_move(col):
            raise ValueError(f"Illegal move: column {col}")

        row = self._get_drop_row(col)
        played_by = self.current_player

        # Save state so we can undo later
        self.move_history.append(
            MoveHistory(
                row=row,
                col=col,
                player=played_by,
                previous_winner=self.winner,
                previous_done=self.done,
                previous_last_move=self.last_move,
            )
        )

        self.board[row][col] = played_by
        self.last_move = (row, col)

        if self.check_winner(row, col):
            self.winner = played_by
            self.done = True
            return MoveResult(
                row=row,
                col=col,
                player=played_by,
                winner=self.winner,
                done=True,
                draw=False,
            )

        if self.is_draw():
            self.done = True
            return MoveResult(
                row=row,
                col=col,
                player=played_by,
                winner=None,
                done=True,
                draw=True,
            )

        self.current_player = (
            self.PLAYER2 if self.current_player == self.PLAYER1 else self.PLAYER1
        )

        return MoveResult(
            row=row,
            col=col,
            player=played_by,
            winner=None,
            done=False,
            draw=False,
        )

    def undo_move(self) -> None:
        """
        Undo the most recent move.

        Raises ValueError if there is no move to undo.
        """
        if not self.move_history:
            raise ValueError("No moves to undo.")

        last = self.move_history.pop()

        # Remove the piece from the board
        self.board[last.row][last.col] = self.EMPTY

        # Restore game state
        self.current_player = last.player
        self.winner = last.previous_winner
        self.done = last.previous_done
        self.last_move = last.previous_last_move

    def _get_drop_row(self, col: int) -> int:
        """Return the row where the piece will land in the given column."""
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == self.EMPTY:
                return row
        raise ValueError(f"Column {col} is full")

    def is_draw(self) -> bool:
        """Return True if the board is full and there is no winner."""
        return self.winner is None and all(
            self.board[0][col] != self.EMPTY for col in range(self.COLS)
        )

    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        return self.done

    def check_winner(self, row: int, col: int) -> bool:
        """
        Check whether the piece placed at (row, col) created 4 in a row.
        """
        player = self.board[row][col]
        if player == self.EMPTY:
            return False

        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right / up-left
            (1, -1),  # diagonal down-left / up-right
        ]

        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)
            if count >= 4:
                return True

        return False

    def _count_direction(
        self, row: int, col: int, dr: int, dc: int, player: int
    ) -> int:
        """Count consecutive pieces for player in one direction."""
        count = 0
        r, c = row + dr, col + dc

        while 0 <= r < self.ROWS and 0 <= c < self.COLS:
            if self.board[r][c] != player:
                break
            count += 1
            r += dr
            c += dc

        return count

    def get_reward(self, player: int) -> int:
        """
        Reward helper for RL-style usage.
        +1 if player won
        -1 if player lost
         0 otherwise / draw / ongoing
        """
        if self.winner is None:
            return 0
        return 1 if self.winner == player else -1

    def render(self) -> None:
        """Print the board in a readable format."""
        symbols = {
            self.EMPTY: ".",
            self.PLAYER1: "X",
            self.PLAYER2: "O",
        }

        print()
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print("0 1 2 3 4 5 6")
        print(f"Current player: {self.current_player}")
        if self.winner is not None:
            print(f"Winner: Player {self.winner}")
        elif self.done:
            print("Game ended in a draw.")
        print()

    def __str__(self) -> str:
        symbols = {
            self.EMPTY: ".",
            self.PLAYER1: "X",
            self.PLAYER2: "O",
        }
        rows = [" ".join(symbols[cell] for cell in row) for row in self.board]
        return "\n".join(rows)


if __name__ == "__main__":
    game = Connect4()

    moves = [3, 3, 2, 2, 1, 1, 0]

    for move in moves:
        result = game.make_move(move)
        game.render()
        print(result)
        if result.done:
            break

    print("Undoing last move...")
    game.undo_move()
    game.render()