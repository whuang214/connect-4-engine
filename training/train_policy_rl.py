"""
Train a strong policy-value RL agent for Connect 4.

Modes: selfplay | vs_mcts | vs_minimax | mixed_teachers

Run from project root:
    python -m training.train_policy_rl --mode selfplay --episodes 500000 --run-name rl_pure_selfplay --n-envs 512 --updates-per-batch 128
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine import Connect4
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.rl_policy_agent import RLPolicyAgent
from models.policy_value_network import (
    PolicyValueNet,
    PolicyValueNetSmall,
    mirror_action,
    mirror_encoded_state,
)
from training.vec_engine import VecConnect4

try:
    from agents.mcts_agent import MCTSAgent
except Exception:
    MCTSAgent = None

try:
    from agents.minimax_agent import MinimaxAgent
except Exception:
    MinimaxAgent = None


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.states:  Deque[np.ndarray] = deque(maxlen=capacity)
        self.actions: Deque[int]        = deque(maxlen=capacity)
        self.returns: Deque[float]      = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state: np.ndarray, action: int, ret: float) -> None:
        self.states.append(state.astype(np.float32, copy=False))
        self.actions.append(int(action))
        self.returns.append(float(ret))

    def add_batch(
        self,
        states:  np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        augment_mirror: bool = True,
    ) -> None:
        for i in range(len(actions)):
            self.add(states[i], int(actions[i]), float(returns[i]))
            if augment_mirror:
                self.add(
                    mirror_encoded_state(states[i]),
                    mirror_action(int(actions[i])),
                    float(returns[i]),
                )

    def sample(self, batch_size: int, device: torch.device):
        idx     = np.random.randint(0, len(self.states), size=batch_size)
        states  = torch.tensor(
            np.stack([self.states[i] for i in idx]), dtype=torch.float32, device=device
        )
        actions = torch.tensor(
            [self.actions[i] for i in idx], dtype=torch.long, device=device
        )
        returns = torch.tensor(
            [self.returns[i] for i in idx], dtype=torch.float32, device=device
        ).unsqueeze(1)
        return states, actions, returns


# ---------------------------------------------------------------------------
# Frozen checkpoint opponent
# FIX: store state_dict on CPU, move to device only at inference time.
# This avoids accumulating N full model copies in GPU VRAM.
# ---------------------------------------------------------------------------

class FrozenPolicyAgent:
    def __init__(self, model: nn.Module, device: torch.device, name: str = "Frozen") -> None:
        self.name   = name
        self.device = device
        # Store weights on CPU — zero persistent VRAM cost
        self.state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self._model: nn.Module | None = None  # lazily loaded at inference time

    def _get_model(self, reference_model: nn.Module) -> nn.Module:
        """Load weights into a temporary model on device for inference."""
        if self._model is None:
            self._model = copy.deepcopy(reference_model)
            self._model.load_state_dict({k: v.to(self.device) for k, v in self.state_dict.items()})
            self._model.eval()
        return self._model

    def choose_action(self, game: Connect4) -> int:
        from models.policy_value_network import encode_board, PolicyValueNet
        legal = game.get_legal_moves()
        if len(legal) == 1:
            return legal[0]

        # Lazily build model if not yet loaded
        if self._model is None:
            # Infer architecture from state_dict keys
            if any(k.startswith("features.") for k in self.state_dict):
                m = PolicyValueNetSmall()
            else:
                m = PolicyValueNet()
            m.load_state_dict({k: v.to(self.device) for k, v in self.state_dict.items()})
            m.eval()
            self._model = m.to(self.device)

        state = torch.tensor(
            encode_board(game), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._model(state)
            logits = logits.squeeze(0)
        mask = torch.full_like(logits, -1e9)
        mask[legal] = 0.0
        return int(torch.argmax(logits + mask).item())

    def unload(self) -> None:
        """Release GPU memory when this frozen agent is evicted from the pool."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Selfplay runner — FULLY BATCHED
# ---------------------------------------------------------------------------

def play_selfplay_vectorized(
    model:          nn.Module,
    device:         torch.device,
    n_envs:         int,
    epsilon:        float,
    temperature:    float,
    augment_mirror: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    vec = VecConnect4(n_envs=n_envs)
    vec.reset()

    ep_states:  List[List[List[np.ndarray]]] = [[[], []] for _ in range(n_envs)]
    ep_actions: List[List[List[int]]]        = [[[], []] for _ in range(n_envs)]

    while not vec.done.all():
        active_mask = vec.active_mask()
        active_idx  = np.where(active_mask)[0]
        if len(active_idx) == 0:
            break

        states_np = vec.encode(active_mask)
        states_t  = torch.tensor(states_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            logits_t, _ = model(states_t)

        legal_np = vec.get_legal_moves_batch()[active_mask]
        logits_t[~torch.tensor(legal_np, device=device)] = -1e9

        if epsilon > 0.0:
            rand_mask = torch.rand(len(active_idx), device=device) < epsilon
        else:
            rand_mask = torch.zeros(len(active_idx), dtype=torch.bool, device=device)

        if temperature > 0:
            probs   = torch.softmax(logits_t / temperature, dim=1)
            actions = torch.multinomial(probs, 1).squeeze(1)
        else:
            actions = torch.argmax(logits_t, dim=1)

        if rand_mask.any():
            for local_i in torch.where(rand_mask)[0].cpu().numpy():
                legal_cols = np.where(legal_np[local_i])[0]
                actions[local_i] = int(np.random.choice(legal_cols))

        actions_np = actions.cpu().numpy()

        cp = vec.current_player[active_idx]
        for local_i, env_i in enumerate(active_idx):
            player_idx = int(cp[local_i]) - 1
            ep_states [env_i][player_idx].append(states_np[local_i])
            ep_actions[env_i][player_idx].append(int(actions_np[local_i]))

        full_actions              = np.zeros(n_envs, dtype=np.int64)
        full_actions[active_mask] = actions_np
        vec.step(full_actions)

    all_states:  List[np.ndarray] = []
    all_actions: List[int]        = []
    all_returns: List[float]      = []

    for env_i in range(n_envs):
        w = int(vec.winner[env_i])
        for player_idx in range(2):
            player  = player_idx + 1
            outcome = 1.0 if w == player else (-1.0 if w != 0 else 0.0)
            for s, a in zip(ep_states[env_i][player_idx], ep_actions[env_i][player_idx]):
                all_states.append(s)
                all_actions.append(a)
                all_returns.append(outcome)
                if augment_mirror:
                    all_states.append(mirror_encoded_state(s))
                    all_actions.append(mirror_action(a))
                    all_returns.append(outcome)

    return (
        np.stack(all_states,  axis=0).astype(np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_returns, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# vs-opponent runner
# ---------------------------------------------------------------------------

def play_vs_opponent_vectorized(
    model:          nn.Module,
    device:         torch.device,
    opponent:       Any,
    n_envs:         int,
    epsilon:        float,
    temperature:    float,
    augment_mirror: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    vec       = VecConnect4(n_envs=n_envs)
    vec.reset()
    py_games  = [Connect4() for _ in range(n_envs)]

    agent_player = np.where(
        np.arange(n_envs) % 2 == 0,
        np.int8(1), np.int8(2)
    )

    ep_states:  List[List[np.ndarray]] = [[] for _ in range(n_envs)]
    ep_actions: List[List[int]]        = [[] for _ in range(n_envs)]

    while not vec.done.all():

        agent_mask  = (~vec.done) & (vec.current_player == agent_player)
        agent_idx   = np.where(agent_mask)[0]

        if len(agent_idx) > 0:
            states_np = vec.encode(agent_mask)
            states_t  = torch.tensor(states_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits_t, _ = model(states_t)

            legal_np = vec.get_legal_moves_batch()[agent_mask]
            logits_t[~torch.tensor(legal_np, device=device)] = -1e9

            if epsilon > 0.0:
                rand_mask = torch.rand(len(agent_idx), device=device) < epsilon
            else:
                rand_mask = torch.zeros(len(agent_idx), dtype=torch.bool, device=device)

            if temperature > 0:
                probs   = torch.softmax(logits_t / temperature, dim=1)
                actions = torch.multinomial(probs, 1).squeeze(1)
            else:
                actions = torch.argmax(logits_t, dim=1)

            if rand_mask.any():
                for local_i in torch.where(rand_mask)[0].cpu().numpy():
                    legal_cols = np.where(legal_np[local_i])[0]
                    actions[local_i] = int(np.random.choice(legal_cols))

            actions_np = actions.cpu().numpy()

            for local_i, env_i in enumerate(agent_idx):
                ep_states [env_i].append(states_np[local_i])
                ep_actions[env_i].append(int(actions_np[local_i]))

            full_actions             = np.zeros(n_envs, dtype=np.int64)
            full_actions[agent_mask] = actions_np
            vec.step(full_actions)

            for local_i, env_i in enumerate(agent_idx):
                if not py_games[env_i].done:
                    py_games[env_i].make_move(int(actions_np[local_i]))

        opp_mask = (~vec.done) & (vec.current_player != agent_player)
        opp_idx  = np.where(opp_mask)[0]

        if len(opp_idx) > 0:
            opp_actions = np.zeros(n_envs, dtype=np.int64)
            for env_i in opp_idx:
                if py_games[env_i].done:
                    continue
                action = opponent.choose_action(py_games[env_i])
                opp_actions[env_i] = action
                py_games[env_i].make_move(action)
            vec.step(opp_actions)

    rewards = vec.get_rewards(agent_player)

    all_states:  List[np.ndarray] = []
    all_actions: List[int]        = []
    all_returns: List[float]      = []

    for env_i in range(n_envs):
        r = float(rewards[env_i])
        for s, a in zip(ep_states[env_i], ep_actions[env_i]):
            all_states.append(s)
            all_actions.append(a)
            all_returns.append(r)
            if augment_mirror:
                all_states.append(mirror_encoded_state(s))
                all_actions.append(mirror_action(a))
                all_returns.append(r)

    return (
        np.stack(all_states,  axis=0).astype(np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_returns, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

class MinimaxFactory:
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def create(self):
        if MinimaxAgent is None:
            raise RuntimeError("MinimaxAgent import failed.")
        try:
            return MinimaxAgent(name=f"Minimax-{self.depth}", depth=self.depth)
        except TypeError:
            return MinimaxAgent(depth=self.depth)


class MCTSFactory:
    def __init__(self, iterations: int) -> None:
        self.iterations = iterations

    def create(self):
        if MCTSAgent is None:
            raise RuntimeError("MCTSAgent import failed.")
        try:
            return MCTSAgent(name=f"MCTS-{self.iterations}", iterations=self.iterations)
        except TypeError:
            return MCTSAgent(iterations=self.iterations)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    wins:     int
    losses:   int
    draws:    int
    win_rate: float


def evaluate_against(agent_under_test, opponent, num_games: int) -> EvalSummary:
    wins = losses = draws = 0
    for g in range(num_games):
        game       = Connect4()
        test_is_p1 = (g % 2 == 0)
        while not game.done:
            is_test = (game.current_player == 1) == test_is_p1
            action  = (agent_under_test.choose_action(game)
                       if is_test else opponent.choose_action(game))
            game.make_move(action)
        tp = 1 if test_is_p1 else 2
        if   game.winner == tp:   wins   += 1
        elif game.winner is None: draws  += 1
        else:                     losses += 1
    return EvalSummary(
        wins=wins, losses=losses, draws=draws,
        win_rate=wins / max(num_games, 1),
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        self.run_dir  = os.path.join(args.output_dir, args.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.model = (
            PolicyValueNetSmall() if args.small_network else
            PolicyValueNet(channels=args.channels, num_blocks=args.num_blocks, dropout=args.dropout)
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(args.episodes, 1), eta_min=args.min_lr
        )
        self.buffer           = ReplayBuffer(args.buffer_size)
        self.checkpoint_pool: List[FrozenPolicyAgent] = []
        self.best_metric      = -float("inf")
        self.start_episode    = 0

        if args.resume:
            self._load_checkpoint(args.resume)

        self.random_agent    = RandomAgent("Random")
        self.rule_agent      = RuleBasedAgent("RuleBased")
        self.minimax_factory = MinimaxFactory(args.minimax_depth)
        self.mcts_factory    = MCTSFactory(args.mcts_iterations)
        self._train_mcts     = self.mcts_factory.create()    if MCTSAgent    is not None else None
        self._train_minimax  = self.minimax_factory.create() if MinimaxAgent is not None else None

        self.log: dict = {k: [] for k in [
            "episodes", "avg_loss", "policy_loss", "value_loss", "entropy",
            "epsilon", "temperature", "buffer_size", "lr",
            "eval_vs_random", "eval_vs_rule", "eval_vs_mcts", "eval_vs_minimax",
        ]}
        self._save_config()

    def _save_config(self) -> None:
        cfg = vars(self.args).copy()
        cfg["device"] = str(self.device)
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "episode"     in ckpt: self.start_episode = int(ckpt["episode"])
        if "best_metric" in ckpt: self.best_metric   = float(ckpt["best_metric"])

    def current_epsilon(self, ep_idx: int) -> float:
        if ep_idx >= self.args.epsilon_decay_episodes:
            return self.args.epsilon_end
        r = ep_idx / max(self.args.epsilon_decay_episodes, 1)
        return self.args.epsilon_start + r * (self.args.epsilon_end - self.args.epsilon_start)

    def current_temperature(self, ep_idx: int) -> float:
        if ep_idx >= self.args.temperature_decay_episodes:
            return self.args.temperature_end
        r = ep_idx / max(self.args.temperature_decay_episodes, 1)
        return self.args.temperature_start + r * (self.args.temperature_end - self.args.temperature_start)

    def _pick_opponent(self, mode: str):
        """Returns an opponent agent or None (None = pure network selfplay path)."""
        if mode == "selfplay":
            if self.checkpoint_pool and random.random() < 0.45:
                return random.choice(self.checkpoint_pool[-self.args.max_checkpoint_pool:])
            return None

        r = random.random()
        if mode == "vs_mcts":
            if r < 0.70: return self._train_mcts
            if r < 0.80: return self.rule_agent
            if r < 0.85: return self.random_agent
            if self.checkpoint_pool and r < 0.90: return self.checkpoint_pool[-1]
            return None
        if mode == "vs_minimax":
            if r < 0.70: return self._train_minimax
            if r < 0.80: return self.rule_agent
            if r < 0.85: return self.random_agent
            if self.checkpoint_pool and r < 0.90: return self.checkpoint_pool[-1]
            return None
        if mode == "mixed_teachers":
            if r < 0.35: return self._train_mcts
            if r < 0.70: return self._train_minimax
            if r < 0.80: return self.rule_agent
            if r < 0.85: return self.random_agent
            if self.checkpoint_pool and r < 0.90: return self.checkpoint_pool[-1]
            return None
        raise ValueError(f"Unknown mode: {mode}")

    def train_batch(self) -> tuple[float, float, float]:
        states, actions, returns = self.buffer.sample(self.args.batch_size, self.device)
        self.model.train()
        logits, values = self.model(states)
        policy_loss = F.cross_entropy(logits, actions)
        value_loss  = F.mse_loss(values, returns)
        probs       = torch.softmax(logits, dim=1)
        entropy     = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        loss = (
            self.args.policy_weight * policy_loss
            + self.args.value_weight * value_loss
            - self.args.entropy_weight * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        return float(policy_loss), float(value_loss), float(entropy)

    def maybe_snapshot(self, ep: int) -> None:
        if ep % self.args.snapshot_interval != 0:
            return
        # FIX: FrozenPolicyAgent now stores weights on CPU, no VRAM cost
        self.checkpoint_pool.append(
            FrozenPolicyAgent(self.model, self.device, f"Frozen-{ep}")
        )
        if len(self.checkpoint_pool) > self.args.max_checkpoint_pool:
            # FIX: explicitly unload evicted agent's GPU model before dropping
            evicted = self.checkpoint_pool.pop(0)
            evicted.unload()

    def save_checkpoint(self, path: str, ep: int) -> None:
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode":              ep,
            "best_metric":          self.best_metric,
            "config":               vars(self.args),
        }, path)

    def run(self) -> None:
        print(f"PyTorch {torch.__version__} | {self.device}")
        print(f"Params:  {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"n_envs:  {self.args.n_envs} | mode: {self.args.mode}")

        recent_pl:  List[float] = []
        recent_vl:  List[float] = []
        recent_ent: List[float] = []

        train_start   = time.time()
        episodes_done = self.start_episode
        total_target  = self.start_episode + self.args.episodes
        n_envs        = self.args.n_envs

        pbar = tqdm(total=self.args.episodes, desc="Training", ncols=120)

        while episodes_done < total_target:
            ep_idx   = episodes_done - self.start_episode
            epsilon  = self.current_epsilon(ep_idx)
            temp     = self.current_temperature(ep_idx)
            opponent = self._pick_opponent(self.args.mode)

            if opponent is None:
                states, actions, returns = play_selfplay_vectorized(
                    self.model, self.device, n_envs, epsilon, temp,
                    augment_mirror=False,
                )
            else:
                states, actions, returns = play_vs_opponent_vectorized(
                    self.model, self.device, opponent, n_envs, epsilon, temp,
                    augment_mirror=False,
                )

            self.buffer.add_batch(states, actions, returns,
                                  augment_mirror=self.args.augment_mirror)
            episodes_done += n_envs
            pbar.update(n_envs)

            if len(self.buffer) >= self.args.batch_size:
                n_updates = (
                    self.args.updates_per_batch if self.args.updates_per_batch > 0
                    else self.args.updates_per_episode * n_envs
                )
                for _ in range(n_updates):
                    pl, vl, ent = self.train_batch()
                    recent_pl.append(pl)
                    recent_vl.append(vl)
                    recent_ent.append(ent)
                self.scheduler.step()

            # FIX: snapshot uses a single pass over the range, not per-ep loop
            # Only fire once per outer loop step — check the batch boundary
            if episodes_done % self.args.snapshot_interval < n_envs:
                self.maybe_snapshot(episodes_done)

            avg_pl  = float(np.mean(recent_pl[-200:]))  if recent_pl  else 0.0
            avg_vl  = float(np.mean(recent_vl[-200:]))  if recent_vl  else 0.0
            avg_ent = float(np.mean(recent_ent[-200:])) if recent_ent else 0.0

            pbar.set_postfix(
                eps=f"{epsilon:.2f}", temp=f"{temp:.2f}",
                pl=f"{avg_pl:.3f}",  vl=f"{avg_vl:.3f}",
                buf=f"{len(self.buffer)//1000}k",
            )

            ep_idx_now = episodes_done - self.start_episode

            if ep_idx_now % self.args.log_interval < n_envs:
                speed = ep_idx_now / max(time.time() - train_start, 1e-6)
                tqdm.write(
                    f"Ep {episodes_done:>7d} | eps={epsilon:.3f} | temp={temp:.3f} | "
                    f"policy={avg_pl:.4f} | value={avg_vl:.4f} | entropy={avg_ent:.4f} | "
                    f"buffer={len(self.buffer):,} | {speed:.1f} ep/s"
                )

            if ep_idx_now % self.args.eval_interval < n_envs:
                # FIX: build eval agent on CPU to avoid allocating another GPU model copy
                eval_model = copy.deepcopy(self.model).cpu()
                eval_agent = RLPolicyAgent(
                    name="RL-Eval",
                    model=eval_model,
                    epsilon=0.0, temperature=0.0,
                    device=torch.device("cpu"),
                    small_network=self.args.small_network,
                )
                res_rand = evaluate_against(eval_agent, self.random_agent, self.args.eval_games)
                res_rule = evaluate_against(eval_agent, self.rule_agent,   self.args.eval_games)
                eval_mcts = eval_minimax = None

                if MCTSAgent is not None:
                    try:
                        eval_mcts = evaluate_against(
                            eval_agent, self.mcts_factory.create(), self.args.eval_games_small
                        )
                    except Exception as e:
                        tqdm.write(f"MCTS eval skipped: {e}")

                if MinimaxAgent is not None:
                    try:
                        eval_minimax = evaluate_against(
                            eval_agent, self.minimax_factory.create(), self.args.eval_games_small
                        )
                    except Exception as e:
                        tqdm.write(f"Minimax eval skipped: {e}")

                # Explicitly free the eval model copy
                del eval_model, eval_agent
                torch.cuda.empty_cache()

                tqdm.write(
                    f"Eval @ {episodes_done}: "
                    f"Random={res_rand.win_rate:.1%}  Rule={res_rule.win_rate:.1%}"
                    + (f"  MCTS={eval_mcts.win_rate:.1%}"       if eval_mcts    else "")
                    + (f"  Minimax={eval_minimax.win_rate:.1%}" if eval_minimax else "")
                )

                score = res_rule.win_rate + 0.5 * res_rand.win_rate
                if eval_mcts    is not None: score += 0.75 * eval_mcts.win_rate
                if eval_minimax is not None: score += 0.75 * eval_minimax.win_rate

                if score > self.best_metric:
                    self.best_metric = score
                    best_path = os.path.join(self.run_dir, "best_model.pt")
                    self.save_checkpoint(best_path, episodes_done)
                    tqdm.write(f"New best -> {best_path}")

                self.log["episodes"].append(episodes_done)
                self.log["avg_loss"].append(
                    float(self.args.policy_weight * avg_pl + self.args.value_weight * avg_vl)
                )
                for k, v in [
                    ("policy_loss",    avg_pl),
                    ("value_loss",     avg_vl),
                    ("entropy",        avg_ent),
                    ("epsilon",        epsilon),
                    ("temperature",    temp),
                    ("buffer_size",    len(self.buffer)),
                    ("lr",             float(self.optimizer.param_groups[0]["lr"])),
                    ("eval_vs_random", float(res_rand.win_rate)),
                    ("eval_vs_rule",   float(res_rule.win_rate)),
                    ("eval_vs_mcts",    None if eval_mcts    is None else float(eval_mcts.win_rate)),
                    ("eval_vs_minimax", None if eval_minimax is None else float(eval_minimax.win_rate)),
                ]:
                    self.log[k].append(v)

                with open(os.path.join(self.run_dir, "training_log.json"), "w") as f:
                    json.dump(self.log, f, indent=2)

            if ep_idx_now % self.args.save_interval < n_envs:
                ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_ep{episodes_done}.pt")
                self.save_checkpoint(ckpt_path, episodes_done)
                tqdm.write(f"Checkpoint -> {ckpt_path}")

        pbar.close()
        final_path = os.path.join(self.run_dir, "final_model.pt")
        self.save_checkpoint(final_path, episodes_done)
        with open(os.path.join(self.run_dir, "training_log.json"), "w") as f:
            json.dump(self.log, f, indent=2)
        elapsed = time.time() - train_start
        print(f"Done -> {final_path}")
        print(f"Total: {elapsed/60:.1f} min | "
              f"{(episodes_done - self.start_episode) / max(elapsed, 1e-6):.1f} ep/s")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       type=str, default="selfplay",
                   choices=["selfplay", "vs_mcts", "vs_minimax", "mixed_teachers"])
    p.add_argument("--episodes",   type=int, default=200000)
    p.add_argument("--run-name",   type=str, default="rl_policy")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--n-envs",     type=int, default=512,
                   help="Parallel games per step. Use 512-1024 for selfplay, "
                        "64-128 for vs_mcts/vs_minimax (opponent is the bottleneck).")

    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--min-lr",       type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip",    type=float, default=1.0)

    p.add_argument("--batch-size",          type=int,   default=512)
    p.add_argument("--buffer-size",         type=int,   default=500000)
    p.add_argument("--updates-per-episode", type=int,   default=2)
    p.add_argument("--updates-per-batch",   type=int,   default=0,
                   help="If >0, do exactly this many gradient steps per outer loop.")

    p.add_argument("--channels",      type=int,   default=128)
    p.add_argument("--num-blocks",    type=int,   default=6)
    p.add_argument("--dropout",       type=float, default=0.10)
    p.add_argument("--small-network", action="store_true")

    p.add_argument("--epsilon-start",              type=float, default=0.25)
    p.add_argument("--epsilon-end",                type=float, default=0.02)
    p.add_argument("--epsilon-decay-episodes",     type=int,   default=25000)
    p.add_argument("--temperature-start",          type=float, default=1.25)
    p.add_argument("--temperature-end",            type=float, default=0.15)
    p.add_argument("--temperature-decay-episodes", type=int,   default=40000)

    p.add_argument("--policy-weight",  type=float, default=1.0)
    p.add_argument("--value-weight",   type=float, default=1.0)
    p.add_argument("--entropy-weight", type=float, default=0.01)
    p.add_argument("--augment-mirror", action="store_true", default=True)

    p.add_argument("--snapshot-interval",    type=int, default=10000)
    p.add_argument("--max-checkpoint-pool",  type=int, default=8)
    p.add_argument("--eval-interval",        type=int, default=25000)
    p.add_argument("--eval-games",           type=int, default=100)
    p.add_argument("--eval-games-small",     type=int, default=20)
    p.add_argument("--save-interval",        type=int, default=25000)
    p.add_argument("--log-interval",         type=int, default=2048)
    p.add_argument("--mcts-iterations",      type=int, default=200)
    p.add_argument("--minimax-depth",        type=int, default=5)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Trainer(args).run()


if __name__ == "__main__":
    main()
