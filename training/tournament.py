"""
Tournament Runner.

Runs round-robin matches between all agents and produces a results table.
This generates the final comparison data for the paper.

Usage:
    python training/tournament.py --model-path runs/default/final_model.pt
    python training/tournament.py --model-path runs/default/final_model.pt --games 200
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.rl_agent import RLAgent
from training.evaluate import evaluate_agent


def main():
    parser = argparse.ArgumentParser(description="Run tournament between agents")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained RL model checkpoint")
    parser.add_argument("--small-network", action="store_true",
                        help="Use small network architecture")
    parser.add_argument("--games", type=int, default=200,
                        help="Games per matchup (default: 200)")
    parser.add_argument("--output", type=str, default="tournament_results.json",
                        help="Output file for results")
    args = parser.parse_args()

    # Create agents
    rl_agent = RLAgent(
        name="RL Agent",
        model_path=args.model_path,
        epsilon=0.0,
        small_network=args.small_network,
    )
    random_agent = RandomAgent("Random")
    rulebased_agent = RuleBasedAgent("RuleBased")

    agents = {
        "RL Agent": rl_agent,
        "Random": random_agent,
        "RuleBased": rulebased_agent,
    }

    # If teammates have implemented their agents, add them here:
    # from agents.minimax_agent import MinimaxAgent
    # from agents.mcts_agent import MCTSAgent
    # agents["Minimax(d=6)"] = MinimaxAgent("Minimax", depth=6)
    # agents["MCTS(1000)"] = MCTSAgent("MCTS", simulations=1000)

    agent_names = list(agents.keys())
    results = {}

    print(f"Tournament: {args.games} games per matchup")
    print(f"Agents: {', '.join(agent_names)}")
    print("=" * 70)

    for i, name_a in enumerate(agent_names):
        results[name_a] = {}
        for j, name_b in enumerate(agent_names):
            if i == j:
                results[name_a][name_b] = {"win_rate": None, "note": "self"}
                continue

            print(f"\n{name_a} vs {name_b}:")
            res = evaluate_agent(
                agents[name_a],
                agents[name_b],
                num_games=args.games,
                alternate_start=True,
            )

            results[name_a][name_b] = {
                "win_rate": res["win_rate"],
                "loss_rate": res["loss_rate"],
                "draw_rate": res["draw_rate"],
                "wins": res["wins"],
                "losses": res["losses"],
                "draws": res["draws"],
                "avg_moves": res["avg_moves"],
                "avg_decision_time_ms": res["avg_decision_time"] * 1000,
                "p1_wins": res["p1_wins"],
                "p2_wins": res["p2_wins"],
            }

            print(f"  W={res['wins']} L={res['losses']} D={res['draws']}  "
                  f"WR={res['win_rate']:.1%}  "
                  f"time={res['avg_decision_time']*1000:.1f}ms/move")

    # Print summary table
    print("\n" + "=" * 70)
    print("TOURNAMENT RESULTS (Win Rate)")
    print("=" * 70)

    # Header
    header = f"{'':>15}"
    for name in agent_names:
        header += f" {name:>12}"
    print(header)

    for name_a in agent_names:
        row = f"{name_a:>15}"
        for name_b in agent_names:
            if name_a == name_b:
                row += f" {'---':>12}"
            else:
                wr = results[name_a][name_b]["win_rate"]
                row += f" {wr:>11.1%}"
        print(row)

    # Print decision time comparison
    print(f"\n{'':>15} {'Avg Time (ms)':>15}")
    for name in agent_names:
        times = []
        for opp_name in agent_names:
            if name != opp_name and "avg_decision_time_ms" in results[name][opp_name]:
                times.append(results[name][opp_name]["avg_decision_time_ms"])
        if times:
            avg_t = sum(times) / len(times)
            print(f"{name:>15} {avg_t:>14.2f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()