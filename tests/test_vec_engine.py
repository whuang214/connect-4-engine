"""Cross-validation of VecConnect4 against the scalar Connect4 engine."""

from __future__ import annotations

import random

import numpy as np

from connect4.engine import Connect4
from connect4.models.policy_value_network import encode_board
from connect4.training.vec_engine import VecConnect4


def _assert_envs_match(vec: VecConnect4, games: list[Connect4]) -> None:
    for i, game in enumerate(games):
        np.testing.assert_array_equal(
            vec.boards[i], np.array(game.board, dtype=np.int8),
            err_msg=f"board mismatch in env {i}",
        )
        assert vec.current_player[i] == game.current_player, f"env {i}"
        assert vec.done[i] == game.done, f"env {i}"
        scalar_winner = 0 if game.winner is None else game.winner
        assert vec.winner[i] == scalar_winner, f"env {i}"


def _play_mirrored(n_envs: int, seed: int, max_steps: int = 42):
    """Step a VecConnect4 and n_envs scalar games with identical random moves."""
    rng = random.Random(seed)
    vec = VecConnect4(n_envs)
    games = [Connect4() for _ in range(n_envs)]

    for _ in range(max_steps):
        if vec.done.all():
            break
        actions = np.zeros(n_envs, dtype=np.int64)
        for i, game in enumerate(games):
            if not game.done:
                actions[i] = rng.choice(game.get_legal_moves())
        vec.step(actions)
        for i, game in enumerate(games):
            if not game.done:
                game.make_move(int(actions[i]))
        _assert_envs_match(vec, games)

    return vec, games


def test_random_playout_matches_scalar_engine_in_64_envs():
    vec, games = _play_mirrored(n_envs=64, seed=42)
    assert vec.done.all()
    assert all(g.done for g in games)


def test_encode_matches_encode_board_on_random_midgame_positions():
    # 50 parallel games advanced 12 random plies -> 50 distinct midgame
    # positions, each encoded by both implementations.
    vec, games = _play_mirrored(n_envs=50, seed=7, max_steps=12)

    encoded = vec.encode()
    assert encoded.shape == (50, 4, 6, 7)
    assert encoded.dtype == np.float32
    for i, game in enumerate(games):
        np.testing.assert_array_equal(encoded[i], encode_board(game))

    # The masked variant must agree with the full encoding.
    mask = np.zeros(50, dtype=bool)
    mask[::3] = True
    np.testing.assert_array_equal(vec.encode(mask=mask), encoded[mask])


def test_step_skips_done_envs_and_full_columns():
    vec = VecConnect4(2)
    # Env 0: player 1 wins vertically in column 0. Env 1: fill column 6.
    for env0_col, env1_col in zip([0, 1, 0, 1, 0, 1, 0], [6] * 7):
        vec.step(np.array([env0_col, env1_col], dtype=np.int64))

    assert vec.done[0]
    assert vec.winner[0] == 1
    # Env 1: column 6 was full on the 7th step -> that move was ignored.
    assert not vec.done[1]
    assert vec.boards[1].sum() > 0
    assert np.count_nonzero(vec.boards[1]) == 6
    assert vec.current_player[1] == 1  # skipped move must not flip the player

    # Stepping a done env changes nothing.
    board0_before = vec.boards[0].copy()
    vec.step(np.array([3, 0], dtype=np.int64))
    np.testing.assert_array_equal(vec.boards[0], board0_before)
    assert np.count_nonzero(vec.boards[1]) == 7  # env 1 still advanced


def test_get_legal_moves_batch():
    vec = VecConnect4(2)
    for env0_col, env1_col in zip([0, 1, 0, 1, 0, 1, 0], [6] * 7):
        vec.step(np.array([env0_col, env1_col], dtype=np.int64))

    legal = vec.get_legal_moves_batch()
    assert legal.shape == (2, 7)
    assert not legal[0].any()  # done env: all False
    assert not legal[1, 6]     # full column: False
    assert legal[1, :6].all()  # everything else open


def test_partial_reset_clears_only_masked_envs():
    vec = VecConnect4(3)
    vec.step(np.array([3, 4, 5], dtype=np.int64))
    vec.step(np.array([3, 4, 5], dtype=np.int64))

    board1_before = vec.boards[1].copy()
    vec.reset(mask=np.array([True, False, True]))

    assert vec.boards[0].sum() == 0
    assert vec.current_player[0] == 1
    assert not vec.done[0]
    np.testing.assert_array_equal(vec.boards[1], board1_before)
    assert vec.current_player[1] == 1  # two moves played -> back to player 1
    assert vec.boards[2].sum() == 0
