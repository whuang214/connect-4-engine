"""
Main Training Script for Connect 4 RL Agent.

Usage:
    python training/train_rl.py                          # Default settings
    python training/train_rl.py --episodes 200000        # More training
    python training/train_rl.py --small-network          # Ablation with smaller net
    python training/train_rl.py --lr 0.0005              # Different learning rate

All checkpoints, logs, and plots are saved under the specified run directory.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Add project root to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.value_network import ValueNetwork, ValueNetworkSmall
from agents.rl_agent import RLAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from training.self_play import play_training_episode
from training.evaluate import evaluate_agent


def parse_args():
    parser = argparse.ArgumentParser(description="Train Connect 4 RL Agent")

    # Training parameters
    parser.add_argument("--episodes", type=int, default=100000,
                        help="Total training episodes (default: 100000)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for regularization (default: 1e-5)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount factor (default: 1.0)")

    # Exploration schedule
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Starting exploration rate (default: 1.0)")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Final exploration rate (default: 0.05)")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=80000,
                        help="Episodes over which to decay epsilon (default: 80000)")

    # Network
    parser.add_argument("--small-network", action="store_true",
                        help="Use smaller network (for ablation study)")
    parser.add_argument("--num-filters", type=int, default=128,
                        help="Number of CNN filters (default: 128)")
    parser.add_argument("--num-res-blocks", type=int, default=4,
                        help="Number of residual blocks (default: 4)")

    # Evaluation and checkpointing
    parser.add_argument("--eval-interval", type=int, default=2000,
                        help="Evaluate every N episodes (default: 2000)")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Games per evaluation (default: 100)")
    parser.add_argument("--save-interval", type=int, default=10000,
                        help="Save checkpoint every N episodes (default: 10000)")

    # Output
    parser.add_argument("--run-name", type=str, default="default",
                        help="Name for this training run (default: 'default')")
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Base output directory (default: 'runs')")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


def get_epsilon(episode, start, end, decay_episodes):
    """Linear epsilon decay schedule."""
    if episode >= decay_episodes:
        return end
    return start + (end - start) * (episode / decay_episodes)


def main():
    args = parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    run_dir = os.path.join(args.output_dir, args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    config = vars(args)
    config["device"] = str(device)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {run_dir}/config.json")

    # Create model
    if args.small_network:
        model = ValueNetworkSmall()
        print("Using SMALL network (ablation mode)")
    else:
        model = ValueNetwork(
            num_filters=args.num_filters,
            num_res_blocks=args.num_res_blocks,
        )
        print(f"Using standard network ({args.num_filters} filters, {args.num_res_blocks} res blocks)")

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_episode = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "episode" in checkpoint:
            start_episode = checkpoint["episode"]
            print(f"Resuming from episode {start_episode}")

    # Baseline opponents for evaluation
    random_opponent = RandomAgent("Random")
    rulebased_opponent = RuleBasedAgent("RuleBased")

    # Training metrics log
    training_log = {
        "episodes": [],
        "avg_loss": [],
        "epsilon": [],
        "win_rate_vs_random": [],
        "win_rate_vs_rulebased": [],
        "p1_wr_random": [],
        "p1_wr_rulebased": [],
        "avg_decision_time_ms": [],
    }

    # Rolling stats
    recent_losses = []
    recent_outcomes = []  # 1=p1 win, 2=p2 win, 0=draw

    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Epsilon: {args.epsilon_start} -> {args.epsilon_end} over {args.epsilon_decay_episodes} episodes")
    print(f"LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print("=" * 70)

    train_start_time = time.time()

    # Main training loop with progress bar
    from tqdm import tqdm
    pbar = tqdm(
        range(start_episode, start_episode + args.episodes),
        desc="Training",
        unit="ep",
        ncols=100,
    )

    for episode in pbar:
        # Compute current epsilon
        epsilon = get_epsilon(
            episode - start_episode,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_episodes,
        )

        # Play one self-play episode
        stats = play_training_episode(model, optimizer, device, epsilon=epsilon)

        recent_losses.append(stats["avg_loss"])
        recent_outcomes.append(stats["winner"] if stats["winner"] else 0)

        # Keep rolling window of 1000
        if len(recent_losses) > 1000:
            recent_losses.pop(0)
            recent_outcomes.pop(0)

        # Update progress bar with rolling stats
        if len(recent_losses) > 0:
            avg_loss = np.mean(recent_losses[-100:])  # last 100 for snappy display
            wr_p1 = sum(1 for o in recent_outcomes[-100:] if o == 1)
            n_recent = min(len(recent_outcomes), 100)
            pbar.set_postfix({
                "ε": f"{epsilon:.3f}",
                "loss": f"{avg_loss:.4f}",
                "P1%": f"{wr_p1/max(n_recent,1):.0%}",
            })

        if (episode + 1) % 500 == 0:
            avg_loss = np.mean(recent_losses)
            p1_wins = sum(1 for o in recent_outcomes if o == 1)
            p2_wins = sum(1 for o in recent_outcomes if o == 2)
            d = sum(1 for o in recent_outcomes if o == 0)
            n = len(recent_outcomes)
            elapsed = time.time() - train_start_time
            eps_per_sec = (episode + 1 - start_episode) / elapsed

            tqdm.write(
                f"Ep {episode + 1:>7d} | "
                f"ε={epsilon:.3f} | "
                f"loss={avg_loss:.4f} | "
                f"self-play P1/P2/D={p1_wins}/{p2_wins}/{d} (last {n}) | "
                f"{eps_per_sec:.1f} ep/s"
            )

        # Periodic evaluation
        if (episode + 1) % args.eval_interval == 0:
            pbar.clear()
            tqdm.write(f"\n--- Evaluation at episode {episode + 1} ---")

            # Create a greedy evaluation agent from the current model
            eval_agent = RLAgent(
                name="RL_eval",
                model=model,
                epsilon=0.0,
                device=device,
            )

            # Evaluate vs Random
            res_rand = evaluate_agent(
                eval_agent, random_opponent,
                num_games=args.eval_games, alternate_start=True,
            )
            tqdm.write(f"  vs Random:    WR={res_rand['win_rate']:.1%}  "
                       f"(W={res_rand['wins']} L={res_rand['losses']} D={res_rand['draws']})  "
                       f"time={res_rand['avg_decision_time']*1000:.1f}ms")

            # Evaluate vs RuleBased
            res_rb = evaluate_agent(
                eval_agent, rulebased_opponent,
                num_games=args.eval_games, alternate_start=True,
            )
            tqdm.write(f"  vs RuleBased: WR={res_rb['win_rate']:.1%}  "
                       f"(W={res_rb['wins']} L={res_rb['losses']} D={res_rb['draws']})  "
                       f"time={res_rb['avg_decision_time']*1000:.1f}ms")

            # Log
            training_log["episodes"].append(episode + 1)
            training_log["avg_loss"].append(np.mean(recent_losses))
            training_log["epsilon"].append(epsilon)
            training_log["win_rate_vs_random"].append(res_rand["win_rate"])
            training_log["win_rate_vs_rulebased"].append(res_rb["win_rate"])
            training_log["p1_wr_random"].append(
                res_rand["p1_wins"] / max(args.eval_games // 2, 1)
            )
            training_log["p1_wr_rulebased"].append(
                res_rb["p1_wins"] / max(args.eval_games // 2, 1)
            )
            training_log["avg_decision_time_ms"].append(
                res_rand["avg_decision_time"] * 1000
            )

            # Save log after each evaluation
            with open(os.path.join(run_dir, "training_log.json"), "w") as f:
                json.dump(training_log, f, indent=2)

            tqdm.write("---\n")

        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"checkpoint_ep{episode + 1}.pt"
            )
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "episode": episode + 1,
                "epsilon": epsilon,
                "config": config,
            }
            torch.save(save_dict, ckpt_path)
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

    # Save final training log
    with open(os.path.join(run_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    total_time = time.time() - train_start_time
    print(f"\nTraining complete in {total_time / 60:.1f} minutes")
    print(f"Average speed: {args.episodes / total_time:.1f} episodes/sec")


if __name__ == "__main__":
    main()