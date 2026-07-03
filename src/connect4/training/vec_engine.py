"""
VecConnect4 — vectorized batch Connect 4 engine for parallel RL training.

Sits in training/vec_connect4.py alongside train_policy_rl.py.
Does NOT replace engine.py — MCTS/Minimax/eval still use Connect4.

Board layout:
    boards[n, r, c]  —  0=empty, 1=player1, 2=player2
    Row 0 = TOP of board (matches Connect4.board convention).
    _heights[n, c]   —  pieces already in column c for env n.
    Drop row for column c = (ROWS - 1) - _heights[n, c].
"""

from __future__ import annotations
import numpy as np


class VecConnect4:
    ROWS = 6
    COLS = 7

    def __init__(self, n_envs: int) -> None:
        self.n = n_envs
        self._init_arrays()

    # ------------------------------------------------------------------
    # Internal initialisation
    # ------------------------------------------------------------------

    def _init_arrays(self) -> None:
        self.boards         = np.zeros((self.n, self.ROWS, self.COLS), dtype=np.int8)
        self._heights       = np.zeros((self.n, self.COLS),            dtype=np.int8)
        self.current_player = np.ones (self.n,                         dtype=np.int8)
        self.winner         = np.zeros(self.n,                         dtype=np.int8)  # 0 = no winner
        self.done           = np.zeros(self.n,                         dtype=bool)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Reset all envs, or only those where mask is True."""
        if mask is None:
            self._init_arrays()
            return
        idx = np.where(mask)[0]
        self.boards[idx]         = 0
        self._heights[idx]       = 0
        self.current_player[idx] = 1
        self.winner[idx]         = 0
        self.done[idx]           = False

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> None:
        """
        Apply actions[i] for every env that is not done.
        actions : int64 (n,) — column indices 0..6.
        Done envs and full-column moves are silently skipped.
        """
        active = np.where(~self.done)[0]
        if len(active) == 0:
            return

        cols    = actions[active].astype(np.int8)
        players = self.current_player[active]

        heights = self._heights[active, cols]
        rows    = (self.ROWS - 1) - heights

        # guard: skip full columns
        legal  = heights < self.ROWS
        active = active[legal]
        cols   = cols[legal]
        players= players[legal]
        rows   = rows[legal]

        if len(active) == 0:
            return

        # place pieces
        self.boards[active, rows, cols]  = players
        self._heights[active, cols]     += 1

        # win detection
        won         = _check_wins_batch(self.boards, active, rows, cols, players)
        won_envs    = active[won]
        self.winner[won_envs] = players[won]
        self.done[won_envs]   = True

        # draw detection (board full, no winner)
        not_won = active[~won]
        if len(not_won):
            draws = (self._heights[not_won] >= self.ROWS).all(axis=1)
            self.done[not_won[draws]] = True

        # flip player for envs still in play
        still_going = active[~self.done[active]]
        self.current_player[still_going] = 3 - self.current_player[still_going]

    # ------------------------------------------------------------------
    # Queries — no agent_player argument needed
    # These are used by play_selfplay_vectorized which treats both
    # sides symmetrically (both are the network).
    # ------------------------------------------------------------------

    def active_mask(self) -> np.ndarray:
        """Bool (n,): envs still running."""
        return ~self.done

    def get_legal_moves_batch(self) -> np.ndarray:
        """Bool (n, 7): legal columns per env. Done envs have all-False rows."""
        legal            = self._heights < self.ROWS        # (n, 7)
        legal[self.done] = False
        return legal

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Encode board states from each env's CURRENT player's perspective.

        mask  : bool (n,) — encode only these envs. None = encode all.
        Returns float32 (M, 4, 6, 7).

        Channels (matches encode_board in policy_value_network.py):
            0 — current player's pieces
            1 — opponent's pieces
            2 — side-to-move plane (1.0 if player 1 to move, else 0.0)
            3 — normalised height map tiled over rows
        """
        idx    = np.where(mask)[0] if mask is not None else np.arange(self.n)
        M      = len(idx)
        boards = self.boards[idx]                                     # (M,6,7)
        cp     = self.current_player[idx]                             # (M,)
        opp    = (3 - cp).astype(np.int8)                            # (M,)

        my_pieces  = (boards == cp [:, None, None]).astype(np.float32)
        opp_pieces = (boards == opp[:, None, None]).astype(np.float32)

        turn_plane = np.broadcast_to(
            (cp == 1).astype(np.float32)[:, None, None],
            (M, self.ROWS, self.COLS),
        ).copy()

        h_norm  = self._heights[idx].astype(np.float32) / self.ROWS   # (M,7)
        h_plane = np.broadcast_to(
            h_norm[:, None, :], (M, self.ROWS, self.COLS),
        ).copy()

        return np.stack([my_pieces, opp_pieces, turn_plane, h_plane], axis=1)

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def get_rewards(self, agent_player: np.ndarray) -> np.ndarray:
        """
        Float32 (n,): +1 win | -1 loss | 0 draw/ongoing.
        agent_player : int8 (n,) — player index each env's agent controlled.
        """
        r = np.zeros(self.n, dtype=np.float32)
        r[self.winner == agent_player]                            =  1.0
        r[(self.winner != 0) & (self.winner != agent_player)]    = -1.0
        return r


# ---------------------------------------------------------------------------
# Win detection — module-level so MCTS vec agent can import _check_win_single
# ---------------------------------------------------------------------------

def _check_wins_batch(
    boards:  np.ndarray,   # (n, 6, 7) full board array
    env_idx: np.ndarray,   # (M,) which envs to check
    rows:    np.ndarray,   # (M,) row of last placed piece
    cols:    np.ndarray,   # (M,) col of last placed piece
    players: np.ndarray,   # (M,) player who just moved
) -> np.ndarray:
    """Bool (M,): True where that move created 4-in-a-row."""
    won = np.zeros(len(env_idx), dtype=bool)
    for m in range(len(env_idx)):
        won[m] = _check_win_single(
            boards[env_idx[m]], int(rows[m]), int(cols[m]), int(players[m])
        )
    return won


def _check_win_single(board: np.ndarray, row: int, col: int, player: int) -> bool:
    """4-in-a-row check for a single board after placing at (row, col)."""
    ROWS, COLS = 6, 7
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        for sign in (1, -1):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < ROWS and 0 <= c < COLS and board[r, c] == player:
                count += 1
                r += sign * dr
                c += sign * dc
        if count >= 4:
            return True
    return False