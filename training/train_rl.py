"""
Main Training Script for Connect 4 RL Agent (v2).

Key improvements over v1:
- Monte Carlo returns: targets are actual game outcomes, not bootstrapped estimates
- Replay buffer: random sampling breaks correlations, stabilizes learning
- Mixed opponents: trains against Random, RuleBased, and self-play
- Alternates playing as P1/P2 to learn both sides

Usage:
    python training/train_rl.py --episodes 100000 --run-name main_run
    python training/train_rl.py --episodes 100000 --small-network --run-name small_net
"""

import argparse
import json
import os
import sys
import time
import random

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.value_network import ValueNetwork, ValueNetworkSmall
from agents.rl_agent import RLAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from training.self_play import (
    ReplayBuffer,
    play_game_collect_data,
    play_selfplay_collect_data,
    train_on_batch,
)
from training.evaluate import evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train Connect 4 RL Agent")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--updates-per-game", type=int, default=2,
                        help="Gradient steps per game played")

    # Exploration
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=30000)

    # Opponent mix (fractions, must sum to 1.0)
    parser.add_argument("--frac-random", type=float, default=0.3)
    parser.add_argument("--frac-rulebased", type=float, default=0.4)
    parser.add_argument("--frac-selfplay", type=float, default=0.3)

    # Network
    parser.add_argument("--small-network", action="store_true")
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=4)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=10000)

    # Output
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def get_epsilon(episode, start, end, decay_episodes):
    if episode >= decay_episodes:
        return end
    return start + (end - start) * (episode / decay_episodes)


def choose_opponent(frac_random, frac_rulebased, random_opp, rulebased_opp):
    """
    Randomly select an opponent type.
    Returns (opponent, name) or (None, "selfplay") for self-play.
    """
    r = random.random()
    if r < frac_random:
        return random_opp, "random"
    elif r < frac_random + frac_rulebased:
        return rulebased_opp, "rulebased"
    else:
        return None, "selfplay"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup directories
    run_dir = os.path.join(args.output_dir, args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["device"] = str(device)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create model
    if args.small_network:
        model = ValueNetworkSmall()
        print("Using SMALL network (ablation)")
    else:
        model = ValueNetwork(args.num_filters, args.num_res_blocks)
        print(f"Using standard network ({args.num_filters} filters, {args.num_res_blocks} res blocks)")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.episodes, eta_min=1e-5
    )

    start_episode = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "episode" in checkpoint:
            start_episode = checkpoint["episode"]

    # Opponents
    random_opp = RandomAgent("Random")
    rulebased_opp = RuleBasedAgent("RuleBased")

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=args.buffer_size)

    # Training log
    training_log = {
        "episodes": [], "avg_loss": [], "epsilon": [],
        "win_rate_vs_random": [], "win_rate_vs_rulebased": [],
        "avg_decision_time_ms": [], "buffer_size": [],
        "lr": [],
    }

    # Rolling stats
    recent_losses = []
    opp_counts = {"random": 0, "rulebased": 0, "selfplay": 0}
    opp_wins = {"random": 0, "rulebased": 0, "selfplay": 0}

    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Opponent mix: {args.frac_random:.0%} Random, "
          f"{args.frac_rulebased:.0%} RuleBased, {args.frac_selfplay:.0%} Self-play")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} over "
          f"{args.epsilon_decay_episodes} episodes")
    print(f"LR: {args.lr} (cosine decay to 1e-5), Batch size: {args.batch_size}")
    print(f"Replay buffer capacity: {args.buffer_size:,}")
    print("=" * 70)

    train_start = time.time()

    pbar = tqdm(range(start_episode, start_episode + args.episodes),
                desc="Training", unit="ep", ncols=110)

    for episode in pbar:
        ep_idx = episode - start_episode
        epsilon = get_epsilon(ep_idx, args.epsilon_start, args.epsilon_end,
                              args.epsilon_decay_episodes)

        # --- Play one game and collect data ---
        opponent, opp_type = choose_opponent(
            args.frac_random, args.frac_rulebased, random_opp, rulebased_opp
        )
        opp_counts[opp_type] += 1

        # Alternate which side the agent plays
        agent_is_p1 = (episode % 2 == 0)

        if opp_type == "selfplay":
            states, targets, winner, n_moves = play_selfplay_collect_data(
                model, device, epsilon=epsilon
            )
        else:
            states, targets, winner, n_moves = play_game_collect_data(
                model, opponent, device, epsilon=epsilon,
                agent_is_p1=agent_is_p1
            )

        # Track wins (for agent, not opponent)
        if opp_type != "selfplay":
            agent_player = 1 if agent_is_p1 else 2
            if winner == agent_player:
                opp_wins[opp_type] += 1

        # Add to replay buffer
        replay_buffer.add_batch(states, targets)

        # --- Train on replay buffer ---
        if len(replay_buffer) >= args.batch_size:
            for _ in range(args.updates_per_game):
                loss = train_on_batch(model, optimizer, replay_buffer, device,
                                      args.batch_size)
                recent_losses.append(loss)

        scheduler.step()

        # Keep rolling window
        if len(recent_losses) > 2000:
            recent_losses = recent_losses[-1000:]

        # Update progress bar
        if recent_losses:
            avg_loss = np.mean(recent_losses[-200:])
            wr_rb = opp_wins["rulebased"] / max(opp_counts["rulebased"], 1)
            pbar.set_postfix({
                "ε": f"{epsilon:.2f}",
                "loss": f"{avg_loss:.4f}",
                "buf": f"{len(replay_buffer)//1000}k",
                "rb_wr": f"{wr_rb:.0%}",
            })

        # --- Periodic logging ---
        if (ep_idx + 1) % 500 == 0:
            avg_loss = np.mean(recent_losses[-500:]) if recent_losses else 0
            elapsed = time.time() - train_start
            eps_per_sec = (ep_idx + 1) / elapsed

            wr_rand = opp_wins["random"] / max(opp_counts["random"], 1)
            wr_rb = opp_wins["rulebased"] / max(opp_counts["rulebased"], 1)

            tqdm.write(
                f"Ep {episode + 1:>7d} | ε={epsilon:.3f} | "
                f"loss={avg_loss:.4f} | buf={len(replay_buffer):,} | "
                f"train_wr: rand={wr_rand:.0%} rb={wr_rb:.0%} | "
                f"{eps_per_sec:.1f} ep/s"
            )

            # Reset opponent tracking every 500 eps for rolling stats
            opp_counts = {"random": 0, "rulebased": 0, "selfplay": 0}
            opp_wins = {"random": 0, "rulebased": 0, "selfplay": 0}

        # --- Periodic evaluation ---
        if (ep_idx + 1) % args.eval_interval == 0:
            pbar.clear()
            tqdm.write(f"\n--- Evaluation at episode {episode + 1} ---")

            eval_agent = RLAgent(name="RL_eval", model=model, epsilon=0.0,
                                 device=device)

            res_rand = evaluate_agent(eval_agent, random_opp,
                                      num_games=args.eval_games,
                                      alternate_start=True)
            tqdm.write(
                f"  vs Random:    WR={res_rand['win_rate']:.1%}  "
                f"(W={res_rand['wins']} L={res_rand['losses']} D={res_rand['draws']})  "
                f"[P1:{res_rand['p1_wins']} P2:{res_rand['p2_wins']}]  "
                f"time={res_rand['avg_decision_time']*1000:.1f}ms"
            )

            res_rb = evaluate_agent(eval_agent, rulebased_opp,
                                    num_games=args.eval_games,
                                    alternate_start=True)
            tqdm.write(
                f"  vs RuleBased: WR={res_rb['win_rate']:.1%}  "
                f"(W={res_rb['wins']} L={res_rb['losses']} D={res_rb['draws']})  "
                f"[P1:{res_rb['p1_wins']} P2:{res_rb['p2_wins']}]  "
                f"time={res_rb['avg_decision_time']*1000:.1f}ms"
            )

            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = np.mean(recent_losses[-500:]) if recent_losses else 0

            training_log["episodes"].append(episode + 1)
            training_log["avg_loss"].append(avg_loss)
            training_log["epsilon"].append(epsilon)
            training_log["win_rate_vs_random"].append(res_rand["win_rate"])
            training_log["win_rate_vs_rulebased"].append(res_rb["win_rate"])
            training_log["avg_decision_time_ms"].append(
                res_rand["avg_decision_time"] * 1000)
            training_log["buffer_size"].append(len(replay_buffer))
            training_log["lr"].append(current_lr)

            with open(os.path.join(run_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)

            tqdm.write("---\n")

        # --- Save checkpoint ---
        if (ep_idx + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir,
                                     f"checkpoint_ep{episode + 1}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": episode + 1,
                "epsilon": epsilon,
                "config": config,
            }, ckpt_path)
            tqdm.write(f"  Checkpoint saved: {ckpt_path}")

    pbar.close()

    # Save final model
    final_path = os.path.join(run_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": start_episode + args.episodes,
        "config": config,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    with open(os.path.join(run_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - train_start
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Average speed: {args.episodes / total_time:.1f} episodes/sec")


if __name__ == "__main__":
    main()