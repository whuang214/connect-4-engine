"""
run_mcts_vs_minimax.py — MCTS vs Minimax Deep Comparison

Focused tournament comparing MCTS and Minimax at multiple
configurations. Split into 4 independent parts that can run
in separate terminals simultaneously.

Usage:
    # Run all 4 parts in separate terminals:
    python run_mcts_vs_minimax.py --part 1    # MCTS-700 vs Minimax depths (~3h)
    python run_mcts_vs_minimax.py --part 2    # MCTS scaling vs Minimax-7 (~4h)
    python run_mcts_vs_minimax.py --part 3    # MCTS-700 vs Minimax-7 main (50 games, ~5h)
    python run_mcts_vs_minimax.py --part 4    # MCTS-200 vs Minimax depths (~2h)

    # Or run everything sequentially:
    python run_mcts_vs_minimax.py --part all

    # Quick smoke test:
    python run_mcts_vs_minimax.py --part 1 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from engine import Connect4
from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from evaluation.evaluate import evaluate_agents, print_evaluation_summary


@dataclass
class Matchup:
    number: int
    category: str
    agent1_factory: Any
    agent2_factory: Any
    num_games: int
    description: str


def make_minimax(depth):
    def factory():
        return MinimaxAgent(depth=depth)
    return factory

def make_mcts(iters):
    def factory():
        return MCTSAgent(iterations=iters)
    return factory


def build_part1(quick=False):
    """MCTS-700 vs Minimax at depth 3, 5, 7, 9 — 30 games each"""
    g = 6 if quick else 30
    matchups = []
    n = 0
    for depth in [3, 5, 7, 9]:
        n += 1
        matchups.append(Matchup(
            number=n, category="MCTS-700 vs Minimax Depths",
            agent1_factory=make_mcts(700),
            agent2_factory=make_minimax(depth),
            num_games=g,
            description=f"MCTS-700 vs Minimax-{depth}",
        ))
    return matchups


def build_part2(quick=False):
    """MCTS at 200, 500, 700, 1000, 1500, 2000 vs Minimax-7 — 30 games each"""
    g = 6 if quick else 30
    matchups = []
    n = 0
    for iters in [200, 500, 700, 1000, 1500, 2000]:
        n += 1
        matchups.append(Matchup(
            number=n, category="MCTS Scaling vs Minimax-7",
            agent1_factory=make_mcts(iters),
            agent2_factory=make_minimax(7),
            num_games=g,
            description=f"MCTS-{iters} vs Minimax-7",
        ))
    return matchups


def build_part3(quick=False):
    """The main event: MCTS-700 vs Minimax-7, 50 games for tight confidence"""
    g = 10 if quick else 50
    return [Matchup(
        number=1, category="Core Head-to-Head",
        agent1_factory=make_mcts(700),
        agent2_factory=make_minimax(7),
        num_games=g,
        description="MCTS-700 vs Minimax-7 (extended)",
    )]


def build_part4(quick=False):
    """MCTS-200 vs Minimax at depth 3, 5, 7, 9 — 30 games each"""
    g = 6 if quick else 30
    matchups = []
    n = 0
    for depth in [3, 5, 7, 9]:
        n += 1
        matchups.append(Matchup(
            number=n, category="MCTS-200 vs Minimax Depths",
            agent1_factory=make_mcts(200),
            agent2_factory=make_minimax(depth),
            num_games=g,
            description=f"MCTS-200 vs Minimax-{depth}",
        ))
    return matchups


PARTS = {
    '1': ('mcts700_vs_depths', build_part1),
    '2': ('mcts_scaling_vs_mm7', build_part2),
    '3': ('mcts700_vs_mm7_extended', build_part3),
    '4': ('mcts200_vs_depths', build_part4),
}


def run_matchups(matchups: List[Matchup], output_path: str) -> Dict[str, Any]:
    all_results: List[Dict[str, Any]] = []
    start = time.perf_counter()

    total_matchups = len(matchups)
    total_games = sum(m.num_games for m in matchups)
    games_done = 0

    print("=" * 70)
    print(f"  Matchups:    {total_matchups}")
    print(f"  Total games: {total_games}")
    print(f"  Output:      {output_path}")
    print("=" * 70)
    print()

    for matchup in matchups:
        print("-" * 70)
        print(f"Matchup #{matchup.number}/{total_matchups} [{matchup.category}]")
        print(f"  {matchup.description}")
        print(f"  Games: {matchup.num_games}")
        print("-" * 70)

        agent1 = matchup.agent1_factory()
        agent2 = matchup.agent2_factory()
        m_start = time.perf_counter()

        summary = evaluate_agents(
            game_class=Connect4,
            agent1=agent1,
            agent2=agent2,
            num_games=matchup.num_games,
            render=False,
            print_each_game=True,
            print_moves=False,
        )

        m_dur = time.perf_counter() - m_start
        games_done += matchup.num_games
        print_evaluation_summary(summary)

        result = {
            "matchup_number": matchup.number,
            "category": matchup.category,
            "description": matchup.description,
            "num_games": matchup.num_games,
            "duration_seconds": round(m_dur, 2),
            "agent1_name": summary.agent1_name,
            "agent2_name": summary.agent2_name,
            "agent1_wins": summary.agent1_wins,
            "agent2_wins": summary.agent2_wins,
            "draws": summary.draws,
            "agent1_win_rate": round(summary.agent1_wins / matchup.num_games, 4),
            "agent2_win_rate": round(summary.agent2_wins / matchup.num_games, 4),
            "draw_rate": round(summary.draws / matchup.num_games, 4),
            "agent1_as_p1_wins": summary.agent1_as_p1_wins,
            "agent1_as_p2_wins": summary.agent1_as_p2_wins,
            "agent2_as_p1_wins": summary.agent2_as_p1_wins,
            "agent2_as_p2_wins": summary.agent2_as_p2_wins,
            "avg_game_length": round(summary.avg_game_length, 2),
            "avg_move_time_agent1": round(summary.avg_move_time_agent1, 6),
            "avg_move_time_agent2": round(summary.avg_move_time_agent2, 6),
        }
        all_results.append(result)

        # Save after every matchup
        _save(all_results, output_path)

        elapsed = time.perf_counter() - start
        remaining = total_games - games_done
        eta = (elapsed / games_done * remaining) if games_done else 0

        print(f"\n  Matchup #{matchup.number} done in {m_dur:.1f}s")
        print(f"  Progress: {games_done}/{total_games} ({games_done/total_games*100:.0f}%)")
        print(f"  Elapsed: {elapsed/3600:.1f}h | ETA: ~{eta/3600:.1f}h")
        print()

    total_dur = time.perf_counter() - start
    tournament_data = {
        "tournament_date": datetime.now().isoformat(),
        "total_matchups": total_matchups,
        "total_games": total_games,
        "total_duration_seconds": round(total_dur, 2),
        "total_duration_hours": round(total_dur / 3600, 2),
        "results": all_results,
    }
    _save(all_results, output_path, tournament_data)

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  {'#':>2}  {'Agent 1':<16} vs {'Agent 2':<16} {'Games':>5} "
          f"{'A1 Win':>7} {'A2 Win':>7} {'Draw':>6} {'Time':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"  {r['matchup_number']:>2}  {r['agent1_name']:<16} vs {r['agent2_name']:<16} "
              f"{r['num_games']:>5} {r['agent1_win_rate']:>6.1%} "
              f"{r['agent2_win_rate']:>6.1%} {r['draw_rate']:>5.1%} "
              f"{r['duration_seconds']:>7.1f}s")
    print(f"\nTotal: {total_dur:.0f}s ({total_dur/3600:.1f}h)")
    print(f"Saved to: {output_path}")

    return tournament_data


def _save(results, path, tournament_data=None):
    data = tournament_data or {"results": results}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MCTS vs Minimax focused tournament")
    parser.add_argument('--part', type=str, required=True,
                        choices=['1', '2', '3', '4', 'all'],
                        help='Which part to run (1-4 or all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick smoke test with 6 games per matchup')
    args = parser.parse_args()

    if args.part == 'all':
        parts_to_run = ['1', '2', '3', '4']
    else:
        parts_to_run = [args.part]

    for part_id in parts_to_run:
        name, builder = PARTS[part_id]
        matchups = builder(quick=args.quick)
        output_path = os.path.join('results', f'mcts_vs_minimax_part{part_id}_{name}.json')

        total_games = sum(m.num_games for m in matchups)
        print(f"\n{'='*70}")
        print(f"PART {part_id}: {name}")
        print(f"  {len(matchups)} matchups, {total_games} games")
        print(f"{'='*70}\n")

        run_matchups(matchups, output_path)


if __name__ == "__main__":
    main()