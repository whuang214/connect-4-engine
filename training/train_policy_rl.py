"""
Train a strong policy-value RL agent for Connect 4 via selfplay.

Run from project root:
    python -m training.train_policy_rl --episodes 1000000 --run-name rl_pure_selfplay_v3 --n-envs 512 --updates-per-batch 128
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, List

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


# ---------------------------------------------------------------------------
# Replay buffer — circular numpy arrays for O(1) random access
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular buffer backed by pre-allocated numpy arrays for O(1) random access."""

    def __init__(self, capacity: int, state_shape: tuple = (4, 6, 7)) -> None:
        self.capacity    = capacity
        self._states     = np.zeros((capacity, *state_shape), dtype=np.float32)
        self._actions    = np.zeros(capacity, dtype=np.int64)
        self._returns    = np.zeros(capacity, dtype=np.float32)
        self._write_ptr  = 0
        self._size       = 0

    def __len__(self) -> int:
        return self._size

    def add(self, state: np.ndarray, action: int, ret: float) -> None:
        i = self._write_ptr
        self._states[i]  = state
        self._actions[i] = action
        self._returns[i] = ret
        self._write_ptr  = (i + 1) % self.capacity
        self._size       = min(self._size + 1, self.capacity)

    def add_batch(
        self,
        states:  np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        augment_mirror: bool = True,
    ) -> None:
        if augment_mirror:
            mir_states  = np.stack([mirror_encoded_state(s) for s in states])
            mir_actions = np.array([mirror_action(int(a)) for a in actions], dtype=np.int64)
            states  = np.concatenate([states,  mir_states],  axis=0)
            actions = np.concatenate([actions, mir_actions], axis=0)
            returns = np.concatenate([returns, returns],     axis=0)

        n = len(actions)
        start = self._write_ptr
        if start + n <= self.capacity:
            self._states [start:start + n] = states
            self._actions[start:start + n] = actions
            self._returns[start:start + n] = returns
        else:
            first = self.capacity - start
            self._states [start:]  = states[:first]
            self._actions[start:]  = actions[:first]
            self._returns[start:]  = returns[:first]
            rest = n - first
            self._states [:rest]  = states[first:]
            self._actions[:rest]  = actions[first:]
            self._returns[:rest]  = returns[first:]
        self._write_ptr = (start + n) % self.capacity
        self._size      = min(self._size + n, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx     = np.random.randint(0, self._size, size=batch_size)
        states  = torch.as_tensor(self._states[idx],  dtype=torch.float32, device=device)
        actions = torch.as_tensor(self._actions[idx], dtype=torch.long,    device=device)
        returns = torch.as_tensor(self._returns[idx], dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, returns


# ---------------------------------------------------------------------------
# Frozen checkpoint opponent for selfplay diversity
# ---------------------------------------------------------------------------

class FrozenPolicyAgent:
    def __init__(self, model: nn.Module, device: torch.device, name: str = "Frozen") -> None:
        self.name   = name
        self.device = device
        self.state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self._model: nn.Module | None = None

    def choose_action(self, game: Connect4) -> int:
        from models.policy_value_network import encode_board
        legal = game.get_legal_moves()
        if len(legal) == 1:
            return legal[0]

        if self._model is None:
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
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tactical helper — immediate win/block detection on numpy boards
# ---------------------------------------------------------------------------

def _find_tactical_move(
    boards:  np.ndarray,   # (M, 6, 7) int8
    heights: np.ndarray,   # (M, 7)    int8
    players: np.ndarray,   # (M,)      int8
) -> np.ndarray:
    from training.vec_engine import _check_win_single
    M            = len(players)
    result       = np.full(M, -1, dtype=np.int64)
    center_order = [3, 2, 4, 1, 5, 0, 6]

    for i in range(M):
        board  = boards[i]
        player = int(players[i])
        h      = heights[i]
        for col in center_order:
            if h[col] >= 6:
                continue
            row = 5 - int(h[col])
            board[row, col] = player
            win = _check_win_single(board, row, col, player)
            board[row, col] = 0
            if win:
                result[i] = col
                break

    return result


# ---------------------------------------------------------------------------
# Selfplay runner — FULLY BATCHED
# Tactical override: win > block > network/epsilon move.
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

        # Tactical override: immediate win > immediate block > network move
        cur_players    = vec.current_player[active_idx]
        opp_players    = (3 - cur_players).astype(np.int8)
        boards_active  = vec.boards[active_idx].copy()
        heights_active = vec._heights[active_idx]

        win_moves   = _find_tactical_move(boards_active.copy(), heights_active, cur_players)
        block_moves = _find_tactical_move(boards_active.copy(), heights_active, opp_players)

        for local_i in range(len(active_idx)):
            if win_moves[local_i] >= 0:
                actions_np[local_i] = int(win_moves[local_i])
            elif block_moves[local_i] >= 0:
                actions_np[local_i] = int(block_moves[local_i])

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
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalSummary:
    wins:     int
    losses:   int
    draws:    int
    win_rate: float


def evaluate_against(
    agent_under_test,
    opponent,
    num_games: int,
    debug: bool = False,
    label: str = "",
) -> EvalSummary:
    wins = losses = draws = 0
    for g in range(num_games):
        game       = Connect4()
        test_is_p1 = (g % 2 == 0)
        max_moves  = Connect4.ROWS * Connect4.COLS
        move_count = 0

        if debug:
            tqdm.write(f"  [DEBUG] {label} game {g+1}/{num_games} starting")

        while not game.done and move_count < max_moves:
            move_count += 1
            is_test = (game.current_player == 1) == test_is_p1

            if debug:
                agent_name = agent_under_test.name if is_test else opponent.name
                t0 = time.perf_counter()

            action = (agent_under_test.choose_action(game.clone())
                      if is_test else opponent.choose_action(game.clone()))

            if debug:
                elapsed = time.perf_counter() - t0 # type: ignore
                tqdm.write(
                    f"    [DEBUG] {label} g{g+1} move {move_count:>2} "
                    f"{agent_name} -> col {action} ({elapsed:.3f}s)" # type: ignore
                )

            game.make_move(action)

        if debug:
            tqdm.write(
                f"  [DEBUG] {label} game {g+1} done in {move_count} moves | "
                f"winner={game.winner}"
            )

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
        n_outer_loops = max(args.episodes // args.n_envs, 1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_outer_loops, eta_min=args.min_lr
        )
        self.buffer           = ReplayBuffer(args.buffer_size)
        self.checkpoint_pool: List[FrozenPolicyAgent] = []
        self.best_metric      = -float("inf")
        self.start_episode    = 0

        if args.resume:
            self._load_checkpoint(args.resume)

        self.random_agent = RandomAgent("Random")
        self.rule_agent   = RuleBasedAgent("RuleBased")

        self.log: dict = {k: [] for k in [
            "episodes", "avg_loss", "policy_loss", "value_loss", "entropy",
            "epsilon", "temperature", "buffer_size", "lr",
            "eval_vs_random", "eval_vs_rule", "eval_vs_mcts",
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
        # Skip scheduler restore — T_max was corrected; resume via last_epoch in run()
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

    def _pick_opponent(self):
        """Selfplay: occasionally play against a frozen past checkpoint."""
        if self.checkpoint_pool and random.random() < 0.45:
            return random.choice(self.checkpoint_pool[-self.args.max_checkpoint_pool:])
        return None

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
        self.checkpoint_pool.append(
            FrozenPolicyAgent(self.model, self.device, f"Frozen-{ep}")
        )
        if len(self.checkpoint_pool) > self.args.max_checkpoint_pool:
            evicted = self.checkpoint_pool.pop(0)
            evicted.unload()

    def save_checkpoint(self, path: str, ep: int) -> None:
        torch.save({
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "episode":              ep,
            "best_metric":          self.best_metric,
            "config":               vars(self.args),
        }, path)

    def run(self) -> None:
        print(f"PyTorch {torch.__version__} | {self.device}")
        print(f"Params:  {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"n_envs:  {self.args.n_envs} | mode: selfplay")

        recent_pl:  List[float] = []
        recent_vl:  List[float] = []
        recent_ent: List[float] = []

        train_start   = time.time()
        episodes_done = self.start_episode
        total_target  = self.args.episodes
        n_envs        = self.args.n_envs

        # Advance scheduler to match resumed position
        if self.start_episode > 0:
            self.scheduler.last_epoch = self.start_episode // n_envs

        pbar = tqdm(total=total_target, desc="Training", ncols=120, initial=self.start_episode)

        while episodes_done < total_target:
            epsilon = self.current_epsilon(episodes_done)
            temp    = self.current_temperature(episodes_done)

            states, actions, returns = play_selfplay_vectorized(
                self.model, self.device, n_envs, epsilon, temp,
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
                eval_model = copy.deepcopy(self.model).cpu()
                eval_agent = RLPolicyAgent(
                    name="RL-Eval",
                    model=eval_model,
                    epsilon=0.0, temperature=0.0,
                    device=torch.device("cpu"),
                    small_network=self.args.small_network,
                )
                dbg = getattr(self.args, 'eval_debug', False)
                res_rand = evaluate_against(eval_agent, self.random_agent, self.args.eval_games, debug=dbg, label="Random")
                res_rule = evaluate_against(eval_agent, self.rule_agent,   self.args.eval_games, debug=dbg, label="Rule")
                eval_mcts = None

                if MCTSAgent is not None:
                    try:
                        mcts_opp = MCTSAgent(name="MCTS-Eval", iterations=self.args.mcts_eval_iterations)
                        eval_mcts = evaluate_against(
                            eval_agent, mcts_opp, self.args.eval_games_small,
                            debug=dbg, label="MCTS"
                        )
                    except Exception as e:
                        tqdm.write(f"MCTS eval skipped: {e}")

                del eval_model, eval_agent
                torch.cuda.empty_cache()

                tqdm.write(
                    f"Eval @ {episodes_done}: "
                    f"Random={res_rand.win_rate:.1%}  Rule={res_rule.win_rate:.1%}"
                    + (f"  MCTS={eval_mcts.win_rate:.1%}" if eval_mcts else "")
                )

                score = res_rule.win_rate + 0.5 * res_rand.win_rate
                if eval_mcts is not None:
                    score += 0.75 * eval_mcts.win_rate

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
                    ("eval_vs_mcts",   None if eval_mcts is None else float(eval_mcts.win_rate)),
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
    p.add_argument("--episodes",   type=int, default=200000)
    p.add_argument("--run-name",   type=str, default="rl_policy")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--resume",     type=str, default=None)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--n-envs",     type=int, default=512)

    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--min-lr",       type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip",    type=float, default=1.0)

    p.add_argument("--batch-size",          type=int, default=1024)
    p.add_argument("--buffer-size",         type=int, default=1000000)
    p.add_argument("--updates-per-episode", type=int, default=2)
    p.add_argument("--updates-per-batch",   type=int, default=256,
                   help="If >0, do exactly this many gradient steps per outer loop.")

    p.add_argument("--channels",      type=int,   default=128)
    p.add_argument("--num-blocks",    type=int,   default=6)
    p.add_argument("--dropout",       type=float, default=0.10)
    p.add_argument("--small-network", action="store_true")

    p.add_argument("--epsilon-start",              type=float, default=0.3)
    p.add_argument("--epsilon-end",                type=float, default=0.05)
    p.add_argument("--epsilon-decay-episodes",     type=int,   default=600000)
    p.add_argument("--temperature-start",          type=float, default=2.0)
    p.add_argument("--temperature-end",            type=float, default=0.3)
    p.add_argument("--temperature-decay-episodes", type=int,   default=800000)

    p.add_argument("--policy-weight",  type=float, default=1.0)
    p.add_argument("--value-weight",   type=float, default=1.0)
    p.add_argument("--entropy-weight", type=float, default=0.05)
    p.add_argument("--augment-mirror", action="store_true", default=True)

    p.add_argument("--snapshot-interval",    type=int, default=10000)
    p.add_argument("--max-checkpoint-pool",  type=int, default=8)
    p.add_argument("--eval-interval",        type=int, default=25000)
    p.add_argument("--eval-games",           type=int, default=100)
    p.add_argument("--eval-games-small",     type=int, default=20)
    p.add_argument("--save-interval",        type=int, default=25000)
    p.add_argument("--log-interval",         type=int, default=2048)
    p.add_argument("--mcts-eval-iterations", type=int, default=200,
                   help="MCTS iterations for evaluation opponent only.")

    p.add_argument("--eval-debug", action="store_true", default=False,
                   help="Print per-move debug output during eval games")

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