"""
Plotting utilities for generating paper-quality figures.

Usage:
    python training/plot_results.py --run-dir runs/default
    python training/plot_results.py --run-dir runs/default --compare runs/small_net
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
})


def load_log(run_dir):
    """Load training log from a run directory."""
    log_path = os.path.join(run_dir, "training_log.json")
    with open(log_path, "r") as f:
        return json.load(f)


def plot_training_curves(log, run_dir, title_prefix=""):
    """Generate the main training curve plot (win rates over episodes)."""
    episodes = log["episodes"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Win rate vs Random
    ax = axes[0, 0]
    ax.plot(episodes, log["win_rate_vs_random"], "b-", linewidth=1.5, label="Win Rate")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"{title_prefix}Win Rate vs Random Agent")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Win rate vs RuleBased
    ax = axes[0, 1]
    ax.plot(episodes, log["win_rate_vs_rulebased"], "r-", linewidth=1.5, label="Win Rate")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% baseline")
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"{title_prefix}Win Rate vs Rule-Based Agent")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Training loss
    ax = axes[1, 0]
    ax.plot(episodes, log["avg_loss"], "g-", linewidth=1.5)
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Average TD Loss")
    ax.set_title(f"{title_prefix}Training Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 4. Epsilon schedule
    ax = axes[1, 1]
    ax.plot(episodes, log["epsilon"], "m-", linewidth=1.5)
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Epsilon")
    ax.set_title(f"{title_prefix}Exploration Rate Schedule")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(run_dir, "training_curves.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_decision_time(log, run_dir):
    """Plot average decision time over training."""
    if "avg_decision_time_ms" not in log:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(log["episodes"], log["avg_decision_time_ms"], "b-", linewidth=1.5)
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Avg Decision Time (ms)")
    ax.set_title("RL Agent Decision Time Over Training")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(run_dir, "decision_time.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comparison(run_dirs, labels, output_path="comparison.png"):
    """
    Compare multiple training runs on the same plot.
    Useful for ablation studies (e.g., small vs large network, different LRs).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for run_dir, label in zip(run_dirs, labels):
        log = load_log(run_dir)

        axes[0].plot(log["episodes"], log["win_rate_vs_random"],
                     linewidth=1.5, label=label)
        axes[1].plot(log["episodes"], log["win_rate_vs_rulebased"],
                     linewidth=1.5, label=label)

    axes[0].set_xlabel("Training Episodes")
    axes[0].set_ylabel("Win Rate")
    axes[0].set_title("Win Rate vs Random Agent")
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Training Episodes")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_title("Win Rate vs Rule-Based Agent")
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_tournament_results(results_dict, output_path="tournament.png"):
    """
    Bar chart of final tournament results.
    results_dict: {agent_name: {opponent_name: win_rate, ...}, ...}
    """
    agents = list(results_dict.keys())
    opponents = list(results_dict[agents[0]].keys())

    x = np.arange(len(opponents))
    width = 0.8 / len(agents)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, agent in enumerate(agents):
        win_rates = [results_dict[agent][opp] for opp in opponents]
        bars = ax.bar(x + i * width, win_rates, width, label=agent)

    ax.set_xlabel("Opponent")
    ax.set_ylabel("Win Rate")
    ax.set_title("Tournament Results")
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels(opponents)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--compare", type=str, nargs="*", default=[],
                        help="Additional run dirs to compare against")
    parser.add_argument("--labels", type=str, nargs="*", default=[],
                        help="Labels for comparison runs")
    args = parser.parse_args()

    log = load_log(args.run_dir)
    plot_training_curves(log, args.run_dir)
    plot_decision_time(log, args.run_dir)

    if args.compare:
        all_dirs = [args.run_dir] + args.compare
        if args.labels:
            all_labels = args.labels
        else:
            all_labels = [os.path.basename(d) for d in all_dirs]
        output = os.path.join(args.run_dir, "comparison.png")
        plot_comparison(all_dirs, all_labels, output)


if __name__ == "__main__":
    main()