"""Shared tournament runner.

Plays a list of :class:`Matchup` definitions through the head-to-head
evaluator, persists results as JSON after every matchup (so an interrupted
run never loses data), and prints progress/summary tables. Used by both the
``connect4 tournament`` and ``connect4 experiment`` subcommands.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from connect4.engine import Connect4
from connect4.evaluation.evaluate import (
    EvaluationSummary,
    evaluate_agents,
    print_evaluation_summary,
)


@dataclass
class Matchup:
    number: int
    category: str
    agent1_factory: Callable[[], Any]
    agent2_factory: Callable[[], Any]
    num_games: int
    description: str


def summarize_matchup(
    matchup: Matchup,
    summary: EvaluationSummary,
    duration_seconds: float,
) -> Dict[str, Any]:
    """Flatten one matchup's evaluation into a JSON-friendly result row."""
    return {
        "matchup_number": matchup.number,
        "category": matchup.category,
        "description": matchup.description,
        "num_games": matchup.num_games,
        "duration_seconds": round(duration_seconds, 2),
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


def save_results(
    results: List[Dict[str, Any]],
    output_path: str,
    tournament_data: Optional[Dict[str, Any]] = None,
) -> None:
    data = tournament_data or {"results": results}
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def run_matchups(matchups: List[Matchup], output_path: str) -> Dict[str, Any]:
    """Play every matchup, saving cumulative results to output_path as we go."""
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
        matchup_start = time.perf_counter()

        summary = evaluate_agents(
            game_class=Connect4,
            agent1=agent1,
            agent2=agent2,
            num_games=matchup.num_games,
            render=False,
            print_each_game=True,
            print_moves=False,
        )

        matchup_duration = time.perf_counter() - matchup_start
        games_done += matchup.num_games
        print_evaluation_summary(summary)

        all_results.append(summarize_matchup(matchup, summary, matchup_duration))

        # Save after every matchup — never lose data
        save_results(all_results, output_path)

        elapsed = time.perf_counter() - start
        remaining = total_games - games_done
        eta = (elapsed / games_done * remaining) if games_done else 0

        print(f"\n  Matchup #{matchup.number} done in {matchup_duration:.1f}s")
        print(f"  Progress: {games_done}/{total_games} games "
              f"({games_done / total_games * 100:.0f}%)")
        print(f"  Elapsed: {elapsed / 3600:.1f}h | ETA: ~{eta / 3600:.1f}h remaining")
        print()

    total_duration = time.perf_counter() - start
    tournament_data = {
        "tournament_date": datetime.now().isoformat(),
        "total_matchups": total_matchups,
        "total_games": total_games,
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_hours": round(total_duration / 3600, 2),
        "results": all_results,
    }
    save_results(all_results, output_path, tournament_data)
    return tournament_data


def print_tournament_summary(tournament_data: Dict[str, Any]) -> None:
    results = tournament_data["results"]
    duration = tournament_data["total_duration_seconds"]

    print("\n" + "=" * 94)
    print("TOURNAMENT SUMMARY")
    print("=" * 94)

    current_category = None

    print(f"  {'#':>2}  {'Agent 1':<20} {'':>3} {'Agent 2':<20} "
          f"{'Games':>5} {'A1 Win':>7} {'A2 Win':>7} {'Draw':>6} {'Time':>8}")
    print("-" * 94)

    for r in results:
        if r["category"] != current_category:
            current_category = r["category"]
            print(f"\n  {current_category.upper()}")
            print(f"  {'-' * 90}")

        print(
            f"  {r['matchup_number']:>2}  "
            f"{r['agent1_name']:<20} vs {r['agent2_name']:<20} "
            f"{r['num_games']:>5} "
            f"{r['agent1_win_rate']:>6.1%} {r['agent2_win_rate']:>6.1%} "
            f"{r['draw_rate']:>5.1%} "
            f"{r['duration_seconds']:>7.1f}s"
        )

    print("\n" + "=" * 94)
    print(f"Total time: {duration:.0f}s ({duration / 60:.0f} min, {duration / 3600:.1f}h)")
    print("=" * 94)

    # P1/P2 split and timing detail for the headline matchups, when present
    core = [r for r in results if r["category"] == "Core"]
    if core:
        print("\nFIRST-PLAYER ADVANTAGE (Core matchups):")
        print(f"  {'Matchup':<45} {'A1 as P1':>9} {'A1 as P2':>9} "
              f"{'A2 as P1':>9} {'A2 as P2':>9}")
        print("  " + "-" * 85)
        for r in core:
            label = f"{r['agent1_name']} vs {r['agent2_name']}"
            print(
                f"  {label:<45} "
                f"{r['agent1_as_p1_wins']:>9} {r['agent1_as_p2_wins']:>9} "
                f"{r['agent2_as_p1_wins']:>9} {r['agent2_as_p2_wins']:>9}"
            )

        print("\nAVERAGE MOVE TIMES:")
        for r in core:
            print(f"  {r['description']}:")
            print(f"    {r['agent1_name']}: {r['avg_move_time_agent1']:.4f}s/move")
            print(f"    {r['agent2_name']}: {r['avg_move_time_agent2']:.4f}s/move")
