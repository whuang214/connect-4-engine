"""``experiment`` subcommand — focused MCTS-vs-minimax scaling comparisons.

Four independent parts that can run in separate terminals simultaneously:

    connect4 experiment --part 1    # MCTS-700 vs Minimax depths (~3h)
    connect4 experiment --part 2    # MCTS scaling vs Minimax-7 (~4h)
    connect4 experiment --part 3    # MCTS-700 vs Minimax-7 extended, 50 games (~5h)
    connect4 experiment --part 4    # MCTS-200 vs Minimax depths (~2h)
    connect4 experiment --part all  # everything sequentially
"""

from __future__ import annotations

import argparse
import os
from typing import List


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--part", type=str, required=True,
                        choices=["1", "2", "3", "4", "all"],
                        help="Which part to run (1-4 or all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test with 6 games per matchup")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save results (default: results/)")
    parser.set_defaults(func=run_experiment)


def _make_minimax(depth):
    def factory():
        from connect4.agents.minimax import MinimaxAgent
        return MinimaxAgent(depth=depth)
    return factory


def _make_mcts(iters):
    def factory():
        from connect4.agents.mcts import MCTSAgent
        return MCTSAgent(iterations=iters)
    return factory


def build_part1(quick: bool = False) -> List:
    """MCTS-700 vs Minimax at depth 3, 5, 7, 9 — 30 games each."""
    from connect4.evaluation.tournament import Matchup

    g = 6 if quick else 30
    return [
        Matchup(
            number=i + 1, category="MCTS-700 vs Minimax Depths",
            agent1_factory=_make_mcts(700),
            agent2_factory=_make_minimax(depth),
            num_games=g,
            description=f"MCTS-700 vs Minimax-{depth}",
        )
        for i, depth in enumerate([3, 5, 7, 9])
    ]


def build_part2(quick: bool = False) -> List:
    """MCTS at 200, 500, 700, 1000, 1500, 2000 vs Minimax-7 — 30 games each."""
    from connect4.evaluation.tournament import Matchup

    g = 6 if quick else 30
    return [
        Matchup(
            number=i + 1, category="MCTS Scaling vs Minimax-7",
            agent1_factory=_make_mcts(iters),
            agent2_factory=_make_minimax(7),
            num_games=g,
            description=f"MCTS-{iters} vs Minimax-7",
        )
        for i, iters in enumerate([200, 500, 700, 1000, 1500, 2000])
    ]


def build_part3(quick: bool = False) -> List:
    """The main event: MCTS-700 vs Minimax-7, 50 games for tight confidence."""
    from connect4.evaluation.tournament import Matchup

    g = 10 if quick else 50
    return [Matchup(
        number=1, category="Core Head-to-Head",
        agent1_factory=_make_mcts(700),
        agent2_factory=_make_minimax(7),
        num_games=g,
        description="MCTS-700 vs Minimax-7 (extended)",
    )]


def build_part4(quick: bool = False) -> List:
    """MCTS-200 vs Minimax at depth 3, 5, 7, 9 — 30 games each."""
    from connect4.evaluation.tournament import Matchup

    g = 6 if quick else 30
    return [
        Matchup(
            number=i + 1, category="MCTS-200 vs Minimax Depths",
            agent1_factory=_make_mcts(200),
            agent2_factory=_make_minimax(depth),
            num_games=g,
            description=f"MCTS-200 vs Minimax-{depth}",
        )
        for i, depth in enumerate([3, 5, 7, 9])
    ]


PARTS = {
    "1": ("mcts700_vs_depths", build_part1),
    "2": ("mcts_scaling_vs_mm7", build_part2),
    "3": ("mcts700_vs_mm7_extended", build_part3),
    "4": ("mcts200_vs_depths", build_part4),
}


def run_experiment(args: argparse.Namespace) -> None:
    from connect4.evaluation.tournament import run_matchups

    parts_to_run = ["1", "2", "3", "4"] if args.part == "all" else [args.part]

    for part_id in parts_to_run:
        name, builder = PARTS[part_id]
        matchups = builder(quick=args.quick)
        output_path = os.path.join(
            args.output_dir, f"mcts_vs_minimax_part{part_id}_{name}.json"
        )

        total_games = sum(m.num_games for m in matchups)
        print(f"\n{'=' * 70}")
        print(f"PART {part_id}: {name}")
        print(f"  {len(matchups)} matchups, {total_games} games")
        print(f"{'=' * 70}\n")

        run_matchups(matchups, output_path)
