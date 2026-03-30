"""
Train a strong policy-value RL agent for Connect 4.

Supports three main modes:
- selfplay        : pure self-play RL
- vs_mcts         : RL trained mainly against MCTS
- vs_minimax      : RL trained mainly against Minimax

Also supports optional opponent mixtures so you can create hybrids if you want.

Examples:
    python -m training.train_policy_rl --mode selfplay --episodes 60000 --run-name rl_selfplay
    python -m training.train_policy_rl --mode vs_mcts --episodes 60000 --run-name rl_vs_mcts --mcts-iterations 800
    python -m training.train_policy_rl --mode vs_minimax --episodes 60000 --run-name rl_vs_minimax --minimax-depth 5
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, Optional

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
    encode_board,
    mirror_action,
    mirror_encoded_state,
)

try:
    from agents.mcts_agent import MCTSAgent
except Exception:
    MCTSAgent = None

try:
    from agents.minimax_agent import MinimaxAgent
except Exception:
    MinimaxAgent = None


@dataclass
class StepRecord:
    state: np.ndarray
    action: int
    player: int
    legal_moves: list[int]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.states: Deque[np.ndarray] = deque(maxlen=capacity)
        self.actions: Deque[int] = deque(maxlen=capacity)
        self.returns: Deque[float] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state: np.ndarray, action: int, ret: float) -> None:
        self.states.append(state.astype(np.float32, copy=False))
        self.actions.append(int(action))
        self.returns.append(float(ret))

    def add_episode(
        self,
        states: list[np.ndarray],
        actions: list[int],
        returns: list[float],
        augment_mirror: bool = True,
    ) -> None:
        for s, a, r in zip(states, actions, returns):
            self.add(s, a, r)
            if augment_mirror:
                self.add(mirror_encoded_state(s), mirror_action(a), r)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, len(self.states), size=batch_size)
        states = torch.tensor(np.stack([self.states[i] for i in idx]), dtype=torch.float32, device=device)
        actions = torch.tensor([self.actions[i] for i in idx], dtype=torch.long, device=device)
        returns = torch.tensor([self.returns[i] for i in idx], dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, returns


class NetworkPolicy:
    def __init__(self, model: nn.Module, device: torch.device, temperature: float, epsilon: float = 0.0) -> None:
        self.model = model
        self.device = device
        self.temperature = temperature
        self.epsilon = epsilon

    def choose_action(self, game: Connect4) -> int:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves.")
        if len(legal_moves) == 1:
            return legal_moves[0]
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            return random.choice(legal_moves)

        state = torch.tensor(encode_board(game), dtype=torch.float32, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(state)
            logits = logits.squeeze(0)

        mask = torch.full_like(logits, -1e9)
        mask[legal_moves] = 0.0
        logits = logits + mask

        if self.temperature > 0:
            probs = torch.softmax(logits / self.temperature, dim=0)
            return int(torch.multinomial(probs, 1).item())
        return int(torch.argmax(logits).item())


class OpponentPool:
    def __init__(self) -> None:
        self.entries: list[tuple[Any, float]] = []

    def add(self, agent: Any, weight: float) -> None:
        if weight > 0:
            self.entries.append((agent, weight))

    def sample(self):
        if not self.entries:
            return None
        agents, weights = zip(*self.entries)
        return random.choices(agents, weights=weights, k=1)[0]


class FrozenPolicyAgent:
    def __init__(self, model: nn.Module, device: torch.device, name: str = "FrozenSelf") -> None:
        self.name = name
        self.device = device
        self.model = copy.deepcopy(model).to(device)
        self.model.eval()

    def choose_action(self, game: Connect4) -> int:
        legal = game.get_legal_moves()
        if len(legal) == 1:
            return legal[0]
        state = torch.tensor(encode_board(game), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(state)
            logits = logits.squeeze(0)
        mask = torch.full_like(logits, -1e9)
        mask[legal] = 0.0
        logits = logits + mask
        return int(torch.argmax(logits).item())


class PolicyOnlyAgent:
    def __init__(self, policy: NetworkPolicy, name: str = "TrainPolicy") -> None:
        self.policy = policy
        self.name = name

    def choose_action(self, game: Connect4) -> int:
        return self.policy.choose_action(game)


class DummySelfWrapper:
    def __init__(self, chooser) -> None:
        self._chooser = chooser
        self.name = "SelfPlay"

    def choose_action(self, game: Connect4) -> int:
        return self._chooser(game)


class MinimaxFactory:
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def create(self):
        if MinimaxAgent is None:
            raise RuntimeError("MinimaxAgent import failed. Update the import path in training/train_policy_rl.py.")
        try:
            return MinimaxAgent(name=f"Minimax-{self.depth}", depth=self.depth)
        except TypeError:
            return MinimaxAgent(depth=self.depth)


class MCTSFactory:
    def __init__(self, iterations: int) -> None:
        self.iterations = iterations

    def create(self):
        if MCTSAgent is None:
            raise RuntimeError("MCTSAgent import failed. Update the import path in training/train_policy_rl.py.")
        try:
            return MCTSAgent(name=f"MCTS-{self.iterations}", iterations=self.iterations)
        except TypeError:
            return MCTSAgent(iterations=self.iterations)


@dataclass
class EpisodeSummary:
    winner: Optional[int]
    moves: int
    agent_player: Optional[int]


@dataclass
class EvalSummary:
    wins: int
    losses: int
    draws: int
    win_rate: float


class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = os.path.join(args.output_dir, args.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if args.small_network:
            self.model = PolicyValueNetSmall().to(self.device)
        else:
            self.model = PolicyValueNet(
                channels=args.channels,
                num_blocks=args.num_blocks,
                dropout=args.dropout,
            ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(args.episodes, 1),
            eta_min=args.min_lr,
        )
        self.buffer = ReplayBuffer(args.buffer_size)
        self.checkpoint_pool: list[FrozenPolicyAgent] = []
        self.best_metric = -float("inf")
        self.start_episode = 0

        if args.resume:
            self._load_checkpoint(args.resume)

        self.random_agent = RandomAgent("Random")
        self.rule_agent = RuleBasedAgent("RuleBased")
        self.minimax_factory = MinimaxFactory(args.minimax_depth)
        self.mcts_factory = MCTSFactory(args.mcts_iterations)

        self.log = {
            "episodes": [],
            "avg_loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "epsilon": [],
            "temperature": [],
            "buffer_size": [],
            "lr": [],
            "eval_vs_random": [],
            "eval_vs_rule": [],
            "eval_vs_mcts": [],
            "eval_vs_minimax": [],
        }

        self._save_config()

    def _save_config(self) -> None:
        config = vars(self.args).copy()
        config["device"] = str(self.device)
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "episode" in checkpoint:
            self.start_episode = int(checkpoint["episode"])
        if "best_metric" in checkpoint:
            self.best_metric = float(checkpoint["best_metric"])

    def current_epsilon(self, episode_idx: int) -> float:
        if episode_idx >= self.args.epsilon_decay_episodes:
            return self.args.epsilon_end
        ratio = episode_idx / max(self.args.epsilon_decay_episodes, 1)
        return self.args.epsilon_start + ratio * (self.args.epsilon_end - self.args.epsilon_start)

    def current_temperature(self, episode_idx: int) -> float:
        if episode_idx >= self.args.temperature_decay_episodes:
            return self.args.temperature_end
        ratio = episode_idx / max(self.args.temperature_decay_episodes, 1)
        return self.args.temperature_start + ratio * (self.args.temperature_end - self.args.temperature_start)

    def build_opponent_pool(self) -> OpponentPool:
        pool = OpponentPool()
        mode = self.args.mode

        if mode == "selfplay":
            pool.add("self_current", 0.55)
            if self.checkpoint_pool:
                for cp in self.checkpoint_pool[-self.args.max_checkpoint_pool:]:
                    pool.add(cp, 0.45 / min(len(self.checkpoint_pool), self.args.max_checkpoint_pool))
        elif mode == "vs_mcts":
            pool.add(self.mcts_factory.create(), 0.70)
            pool.add(self.rule_agent, 0.10)
            pool.add(self.random_agent, 0.05)
            pool.add("self_current", 0.10)
            if self.checkpoint_pool:
                pool.add(self.checkpoint_pool[-1], 0.05)
        elif mode == "vs_minimax":
            pool.add(self.minimax_factory.create(), 0.70)
            pool.add(self.rule_agent, 0.10)
            pool.add(self.random_agent, 0.05)
            pool.add("self_current", 0.10)
            if self.checkpoint_pool:
                pool.add(self.checkpoint_pool[-1], 0.05)
        elif mode == "mixed_teachers":
            pool.add(self.mcts_factory.create(), 0.35)
            pool.add(self.minimax_factory.create(), 0.35)
            pool.add(self.rule_agent, 0.10)
            pool.add(self.random_agent, 0.05)
            pool.add("self_current", 0.10)
            if self.checkpoint_pool:
                pool.add(self.checkpoint_pool[-1], 0.05)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return pool

    def pick_opponent(self, pool: OpponentPool, train_policy: NetworkPolicy):
        selected = pool.sample()
        if selected == "self_current":
            return DummySelfWrapper(train_policy.choose_action)
        return selected

    def play_episode(
        self,
        train_policy: NetworkPolicy,
        opponent: Any,
        agent_is_player1: bool,
    ) -> tuple[list[np.ndarray], list[int], list[float], EpisodeSummary]:
        game = Connect4()
        agent_player = 1 if agent_is_player1 else 2
        steps: list[StepRecord] = []

        while not game.done:
            to_move = game.current_player
            if to_move == agent_player:
                state = encode_board(game)
                legal = game.get_legal_moves()
                action = train_policy.choose_action(game)
                steps.append(StepRecord(state=state, action=action, player=to_move, legal_moves=legal))
            else:
                action = opponent.choose_action(game)

            game.make_move(action)

        outcome = game.get_reward(agent_player)
        returns = [outcome for _ in range(len(steps))]
        states = [s.state for s in steps]
        actions = [s.action for s in steps]
        summary = EpisodeSummary(winner=game.winner, moves=len(game.move_history), agent_player=agent_player)
        return states, actions, returns, summary

    def train_batch(self) -> tuple[float, float, float]:
        states, actions, returns = self.buffer.sample(self.args.batch_size, self.device)
        self.model.train()
        logits, values = self.model(states)

        policy_loss = F.cross_entropy(logits, actions)
        value_loss = F.mse_loss(values, returns)
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()

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
        return float(policy_loss.item()), float(value_loss.item()), float(entropy.item())

    def evaluate_against(self, agent_under_test: RLPolicyAgent, opponent: Any, num_games: int) -> EvalSummary:
        wins = 0
        losses = 0
        draws = 0
        for g in range(num_games):
            game = Connect4()
            test_is_p1 = (g % 2 == 0)
            while not game.done:
                if (game.current_player == 1 and test_is_p1) or (game.current_player == 2 and not test_is_p1):
                    action = agent_under_test.choose_action(game)
                else:
                    action = opponent.choose_action(game)
                game.make_move(action)

            test_player = 1 if test_is_p1 else 2
            if game.winner == test_player:
                wins += 1
            elif game.winner is None:
                draws += 1
            else:
                losses += 1
        return EvalSummary(wins=wins, losses=losses, draws=draws, win_rate=wins / max(num_games, 1))

    def maybe_snapshot(self, episode_number: int) -> None:
        if episode_number % self.args.snapshot_interval != 0:
            return
        self.checkpoint_pool.append(FrozenPolicyAgent(self.model, self.device, name=f"Frozen-{episode_number}"))
        if len(self.checkpoint_pool) > self.args.max_checkpoint_pool:
            self.checkpoint_pool = self.checkpoint_pool[-self.args.max_checkpoint_pool :]

    def save_checkpoint(self, path: str, episode_number: int) -> None:
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "episode": episode_number,
                "best_metric": self.best_metric,
                "config": vars(self.args),
            },
            path,
        )

    def run(self) -> None:
        print(f"Using device: {self.device}")
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable params: {param_count:,}")

        recent_policy_losses: list[float] = []
        recent_value_losses: list[float] = []
        recent_entropies: list[float] = []
        train_start = time.time()

        pbar = tqdm(range(self.start_episode, self.start_episode + self.args.episodes), desc="Training", ncols=120)
        for episode in pbar:
            ep_idx = episode - self.start_episode
            epsilon = self.current_epsilon(ep_idx)
            temperature = self.current_temperature(ep_idx)
            train_policy = NetworkPolicy(self.model, self.device, temperature=temperature, epsilon=epsilon)
            pool = self.build_opponent_pool()
            opponent = self.pick_opponent(pool, train_policy)
            agent_is_p1 = (episode % 2 == 0)

            states, actions, returns, _summary = self.play_episode(
                train_policy=train_policy,
                opponent=opponent,
                agent_is_player1=agent_is_p1,
            )
            self.buffer.add_episode(states, actions, returns, augment_mirror=self.args.augment_mirror)

            if len(self.buffer) >= self.args.batch_size:
                for _ in range(self.args.updates_per_episode):
                    p_loss, v_loss, ent = self.train_batch()
                    recent_policy_losses.append(p_loss)
                    recent_value_losses.append(v_loss)
                    recent_entropies.append(ent)

            self.scheduler.step()
            self.maybe_snapshot(episode + 1)

            avg_pl = np.mean(recent_policy_losses[-200:]) if recent_policy_losses else 0.0
            avg_vl = np.mean(recent_value_losses[-200:]) if recent_value_losses else 0.0
            avg_ent = np.mean(recent_entropies[-200:]) if recent_entropies else 0.0
            pbar.set_postfix(
                eps=f"{epsilon:.2f}",
                temp=f"{temperature:.2f}",
                pl=f"{avg_pl:.3f}",
                vl=f"{avg_vl:.3f}",
                buf=f"{len(self.buffer)//1000}k",
            )

            if (ep_idx + 1) % self.args.log_interval == 0:
                elapsed = time.time() - train_start
                speed = (ep_idx + 1) / max(elapsed, 1e-6)
                tqdm.write(
                    f"Ep {episode + 1:>7d} | eps={epsilon:.3f} | temp={temperature:.3f} | "
                    f"policy={avg_pl:.4f} | value={avg_vl:.4f} | entropy={avg_ent:.4f} | "
                    f"buffer={len(self.buffer):,} | {speed:.1f} ep/s"
                )

            if (ep_idx + 1) % self.args.eval_interval == 0:
                eval_agent = RLPolicyAgent(
                    name="RL-Eval",
                    model=copy.deepcopy(self.model).to(self.device),
                    epsilon=0.0,
                    temperature=0.0,
                    device=self.device,
                    small_network=self.args.small_network,
                )
                res_rand = self.evaluate_against(eval_agent, self.random_agent, self.args.eval_games)
                res_rule = self.evaluate_against(eval_agent, self.rule_agent, self.args.eval_games)

                eval_mcts = None
                if MCTSAgent is not None:
                    try:
                        eval_mcts = self.evaluate_against(eval_agent, self.mcts_factory.create(), self.args.eval_games_small)
                    except Exception as exc:
                        tqdm.write(f"Skipping MCTS eval: {exc}")

                eval_minimax = None
                if MinimaxAgent is not None:
                    try:
                        eval_minimax = self.evaluate_against(eval_agent, self.minimax_factory.create(), self.args.eval_games_small)
                    except Exception as exc:
                        tqdm.write(f"Skipping Minimax eval: {exc}")

                tqdm.write(
                    f"Eval @ {episode + 1}: "
                    f"Random WR={res_rand.win_rate:.1%} | Rule WR={res_rule.win_rate:.1%}"
                    + (f" | MCTS WR={eval_mcts.win_rate:.1%}" if eval_mcts else "")
                    + (f" | Minimax WR={eval_minimax.win_rate:.1%}" if eval_minimax else "")
                )

                self.log["episodes"].append(episode + 1)
                self.log["avg_loss"].append(float(self.args.policy_weight * avg_pl + self.args.value_weight * avg_vl))
                self.log["policy_loss"].append(float(avg_pl))
                self.log["value_loss"].append(float(avg_vl))
                self.log["entropy"].append(float(avg_ent))
                self.log["epsilon"].append(float(epsilon))
                self.log["temperature"].append(float(temperature))
                self.log["buffer_size"].append(int(len(self.buffer)))
                self.log["lr"].append(float(self.optimizer.param_groups[0]["lr"]))
                self.log["eval_vs_random"].append(float(res_rand.win_rate))
                self.log["eval_vs_rule"].append(float(res_rule.win_rate))
                self.log["eval_vs_mcts"].append(None if eval_mcts is None else float(eval_mcts.win_rate))
                self.log["eval_vs_minimax"].append(None if eval_minimax is None else float(eval_minimax.win_rate))

                with open(os.path.join(self.run_dir, "training_log.json"), "w", encoding="utf-8") as f:
                    json.dump(self.log, f, indent=2)

                score = res_rule.win_rate + 0.5 * res_rand.win_rate
                if eval_mcts is not None:
                    score += 0.75 * eval_mcts.win_rate
                if eval_minimax is not None:
                    score += 0.75 * eval_minimax.win_rate

                if score > self.best_metric:
                    self.best_metric = score
                    best_path = os.path.join(self.run_dir, "best_model.pt")
                    self.save_checkpoint(best_path, episode + 1)
                    tqdm.write(f"Saved new best model -> {best_path}")

            if (ep_idx + 1) % self.args.save_interval == 0:
                ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_ep{episode + 1}.pt")
                self.save_checkpoint(ckpt_path, episode + 1)
                tqdm.write(f"Checkpoint saved -> {ckpt_path}")

        final_path = os.path.join(self.run_dir, "final_model.pt")
        self.save_checkpoint(final_path, self.start_episode + self.args.episodes)
        with open(os.path.join(self.run_dir, "training_log.json"), "w", encoding="utf-8") as f:
            json.dump(self.log, f, indent=2)
        total_time = time.time() - train_start
        print(f"Final model saved -> {final_path}")
        print(f"Training finished in {total_time / 60:.1f} minutes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy-value RL agent for Connect 4.")

    parser.add_argument("--mode", type=str, default="selfplay", choices=["selfplay", "vs_mcts", "vs_minimax", "mixed_teachers"])
    parser.add_argument("--episodes", type=int, default=60000)
    parser.add_argument("--run-name", type=str, default="rl_policy")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=250000)
    parser.add_argument("--updates-per-episode", type=int, default=2)

    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--small-network", action="store_true")

    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=25000)
    parser.add_argument("--temperature-start", type=float, default=1.25)
    parser.add_argument("--temperature-end", type=float, default=0.15)
    parser.add_argument("--temperature-decay-episodes", type=int, default=40000)

    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--augment-mirror", action="store_true")

    parser.add_argument("--snapshot-interval", type=int, default=2500)
    parser.add_argument("--max-checkpoint-pool", type=int, default=8)

    parser.add_argument("--eval-interval", type=int, default=2500)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--eval-games-small", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=500)

    parser.add_argument("--mcts-iterations", type=int, default=800)
    parser.add_argument("--minimax-depth", type=int, default=5)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()