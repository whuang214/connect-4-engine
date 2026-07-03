"""Tests for the policy-value networks, board encoding, and replay buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from connect4.engine import Connect4
from connect4.models.policy_value_network import (
    PolicyValueNet,
    PolicyValueNetSmall,
    encode_board,
    mirror_action,
    mirror_encoded_state,
)
from connect4.training.trainer import ReplayBuffer


@pytest.mark.parametrize("net_class", [PolicyValueNet, PolicyValueNetSmall])
def test_forward_shapes_and_value_range(net_class):
    torch.manual_seed(0)
    net = net_class()
    net.eval()
    x = torch.randn(5, 4, 6, 7)
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (5, 7)
    assert value.shape == (5, 1)
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_encode_board_fresh_game():
    game = Connect4()
    state = encode_board(game)
    assert state.shape == (4, 6, 7)
    assert state.dtype == np.float32
    np.testing.assert_array_equal(state[0], np.zeros((6, 7)))  # my pieces
    np.testing.assert_array_equal(state[1], np.zeros((6, 7)))  # opp pieces
    np.testing.assert_array_equal(state[2], np.ones((6, 7)))   # P1 to move
    np.testing.assert_array_equal(state[3], np.zeros((6, 7)))  # empty heights


def test_encode_board_perspective_flips_after_a_move():
    game = Connect4()
    game.make_move(3)
    state = encode_board(game)  # now from P2's perspective
    assert state[0].sum() == 0.0            # P2 has no pieces
    assert state[1][5][3] == 1.0            # P1's piece is the opponent's
    assert state[1].sum() == 1.0
    np.testing.assert_array_equal(state[2], np.zeros((6, 7)))  # P2 to move
    assert state[3][0][3] == pytest.approx(1.0 / 6.0)
    assert state[3][:, 3].sum() == pytest.approx(1.0)


def test_mirror_encoded_state_is_an_involution_and_reverses_columns():
    game = Connect4()
    for col in [0, 2, 3, 3, 6]:
        game.make_move(col)
    state = encode_board(game)
    mirrored = mirror_encoded_state(state)
    np.testing.assert_array_equal(mirrored, state[:, :, ::-1])
    np.testing.assert_array_equal(mirror_encoded_state(mirrored), state)

    # A mirrored position encodes identically to playing the mirrored moves.
    mirrored_game = Connect4()
    for col in [6, 4, 3, 3, 0]:
        mirrored_game.make_move(col)
    np.testing.assert_array_equal(mirrored, encode_board(mirrored_game))


def test_mirror_action_mapping_and_involution():
    assert mirror_action(0) == 6
    assert mirror_action(6) == 0
    assert mirror_action(3) == 3
    for a in range(7):
        assert mirror_action(mirror_action(a)) == a


def test_replay_buffer_wraparound():
    buf = ReplayBuffer(capacity=10)
    states = np.arange(7 * 4 * 6 * 7, dtype=np.float32).reshape(7, 4, 6, 7)
    actions = np.arange(7, dtype=np.int64)
    returns = np.linspace(-1.0, 1.0, 7).astype(np.float32)

    buf.add_batch(states, actions, returns, augment_mirror=False)
    assert len(buf) == 7
    assert buf._write_ptr == 7

    buf.add_batch(states, actions, returns, augment_mirror=False)
    assert len(buf) == 10  # size caps at capacity
    assert buf._write_ptr == 4  # (7 + 7) % 10

    # Slots 7-9 hold the first 3 items of the second batch; slots 0-3 the rest.
    np.testing.assert_array_equal(buf._actions[7:10], actions[:3])
    np.testing.assert_array_equal(buf._actions[0:4], actions[3:7])
    np.testing.assert_array_equal(buf._states[9], states[2])
    np.testing.assert_array_equal(buf._states[3], states[6])


def test_replay_buffer_mirror_augmentation():
    buf = ReplayBuffer(capacity=100)
    states = np.random.default_rng(0).random((4, 4, 6, 7)).astype(np.float32)
    actions = np.array([0, 2, 3, 6], dtype=np.int64)
    returns = np.array([1.0, -1.0, 0.0, 1.0], dtype=np.float32)

    buf.add_batch(states, actions, returns, augment_mirror=True)
    assert len(buf) == 8  # doubled

    np.testing.assert_array_equal(buf._actions[:4], actions)
    np.testing.assert_array_equal(buf._actions[4:8], np.array([6, 4, 3, 0]))
    np.testing.assert_array_equal(buf._returns[4:8], returns)
    for i in range(4):
        np.testing.assert_array_equal(buf._states[4 + i], states[i][:, :, ::-1])


def test_replay_buffer_sample_shapes():
    buf = ReplayBuffer(capacity=32)
    states = np.zeros((8, 4, 6, 7), dtype=np.float32)
    actions = np.arange(8, dtype=np.int64) % 7
    returns = np.zeros(8, dtype=np.float32)
    buf.add_batch(states, actions, returns, augment_mirror=False)

    s, a, r = buf.sample(batch_size=16, device=torch.device("cpu"))
    assert s.shape == (16, 4, 6, 7)
    assert a.shape == (16,)
    assert r.shape == (16, 1)
