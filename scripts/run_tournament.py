"""
run_tournament.py — Connect 4 Phase 1 Tournament Runner (Overnight Edition)

Designed to complete in ~8-10 hours overnight while still producing
statistically meaningful results for the paper.

Strategy: MCTS games take ~3-5 min each, so those get fewer games.
          Non-MCTS games are instant, so those get more games.

Game counts:
  - Non-MCTS matchups: 100 games (fast — seconds total)
  - MCTS-700 matchups: 20 games (~60-80 min each matchup)
  - MCTS-200 matchups: 20 games (~20-30 min each matchup)
  - MCTS-1000 matchups: 14 games (~60-90 min each matchup)
  - MCTS-2000 matchups: 10 games (~80-100 min each matchup)

Total: ~860 games, estimated ~8-10 hours

Results save after every matchup so you lose nothing if interrupted.

Usage:
    python run_tournament.py                # full overnight run
    python run_tournament.py --quick        # smoke test (~15 min)
    python run_tournament.py --skip-slow    # skip MCTS-2000 + depth 9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from engine import Connect4
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts_agent import MCTSAgent
from agents.minimax_agent import MinimaxAgent
from agents.rl_policy_agent import RLPolicyAgent
from evaluation.evaluate import evaluate_agents, print_evaluation_summary


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RL_RUN_DIR = os.path.join("runs", "rl_pure_selfplay_v3")
RL_BEST_MODEL = os.path.join(RL_RUN_DIR, "best_model.pt")
RL_CHECKPOINTS = {
    "100k": os.path.join(RL_RUN_DIR, "checkpoints", "checkpoint_ep100352.pt"),
    "200k": os.path.join(RL_RUN_DIR, "checkpoints", "checkpoint_ep200192.pt"),
    "300k": os.path.join(RL_RUN_DIR, "checkpoints", "checkpoint_ep300032.pt"),
    "400k": os.path.join(RL_RUN_DIR, "checkpoints", "checkpoint_ep400384.pt"),
    "500k": os.path.join(RL_RUN_DIR, "checkpoints", "checkpoint_ep500224.pt"),
}


# ---------------------------------------------------------------------------
# Matchup definition
# ---------------------------------------------------------------------------

@dataclass
class Matchup:
    number: int
    category: str
    agent1_factory: Any
    agent2_factory: Any
    num_games: int
    description: str


def build_matchups(args) -> List[Matchup]:
    """
    Build matchups with game counts tuned for overnight runtime.

    MCTS games are expensive (~3-5 min each at 700 iter), so they get
    fewer games. Non-MCTS games are instant, so they get 100 games
    for tight confidence intervals.
    """

    matchups: List[Matchup] = []
    n = 0

    # Game counts by speed tier
    fast = args.fast_games       # 100 — RL vs minimax, baselines without MCTS
    mcts_700 = args.mcts_games   # 20  — any matchup with MCTS-700
    mcts_200 = args.mcts_games   # 20  — MCTS-200 scaling
    mcts_1000 = max(args.mcts_games - 6, 14)  # 14
    mcts_2000 = max(args.mcts_games - 10, 10) # 10

    # --- Agent factories ---
    # temperature=0.3 so RL samples from its policy (not argmax every time)
    # This prevents identical repeated games while still playing near-best moves

    def make_rl_best():
        return RLPolicyAgent(name="RL-best", model_path=RL_BEST_MODEL, temperature=0.3)

    def make_rl_checkpoint(label, path):
        def factory():
            return RLPolicyAgent(name=f"RL-{label}", model_path=path, temperature=0.3)
        return factory

    def make_minimax(depth):
        def factory():
            return MinimaxAgent(depth=depth)
        return factory

    def make_mcts(iters):
        def factory():
            return MCTSAgent(iterations=iters)
        return factory

    def make_random():
        return RandomAgent()

    def make_rule_based():
        return RuleBasedAgent()

    # ===================================================================
    # 1. CORE MATCHUPS
    #    RL vs Minimax is fast -> 100 games
    #    Anything with MCTS -> 20 games
    # ===================================================================

    n += 1
    matchups.append(Matchup(
        number=n, category="Core",
        agent1_factory=make_rl_best,
        agent2_factory=make_minimax(7),
        num_games=fast,
        description="RL (best) vs Minimax (depth 7)",
    ))

    n += 1
    matchups.append(Matchup(
        number=n, category="Core",
        agent1_factory=make_rl_best,
        agent2_factory=make_mcts(700),
        num_games=mcts_700,
        description="RL (best) vs MCTS (700 iter)",
    ))

    n += 1
    matchups.append(Matchup(
        number=n, category="Core",
        agent1_factory=make_mcts(700),
        agent2_factory=make_minimax(7),
        num_games=mcts_700,
        description="MCTS (700 iter) vs Minimax (depth 7)",
    ))

    # ===================================================================
    # 2. BASELINES — 100 games except MCTS ones get 20
    # ===================================================================

    n += 1
    matchups.append(Matchup(
        number=n, category="Baseline",
        agent1_factory=make_rl_best,
        agent2_factory=make_random,
        num_games=fast,
        description="RL (best) vs Random",
    ))

    n += 1
    matchups.append(Matchup(
        number=n, category="Baseline",
        agent1_factory=make_rl_best,
        agent2_factory=make_rule_based,
        num_games=fast,
        description="RL (best) vs Rule-Based",
    ))

    n += 1
    matchups.append(Matchup(
        number=n, category="Baseline",
        agent1_factory=make_mcts(700),
        agent2_factory=make_random,
        num_games=mcts_700,
        description="MCTS (700 iter) vs Random",
    ))

    n += 1
    matchups.append(Matchup(
        number=n, category="Baseline",
        agent1_factory=make_mcts(700),
        agent2_factory=make_rule_based,
        num_games=mcts_700,
        description="MCTS (700 iter) vs Rule-Based",
    ))

    n += 1
    matchups.append(Matchup(
        number=n, category="Baseline",
        agent1_factory=make_minimax(7),
        agent2_factory=make_random,
        num_games=fast,
        description="Minimax (depth 7) vs Random",
    ))

    n += 1
    matchups.append(Matchup(
        number=n, category="Baseline",
        agent1_factory=make_minimax(7),
        agent2_factory=make_rule_based,
        num_games=fast,
        description="Minimax (depth 7) vs Rule-Based",
    ))

    # ===================================================================
    # 3. RL LEARNING CURVE — all fast (RL vs Minimax, no MCTS)
    # ===================================================================

    for label, path in sorted(RL_CHECKPOINTS.items(), key=lambda x: x[0]):
        if not os.path.exists(path):
            print(f"  [WARN] Checkpoint not found, skipping: {path}")
            continue
        n += 1
        matchups.append(Matchup(
            number=n, category="RL Learning Curve",
            agent1_factory=make_rl_checkpoint(label, path),
            agent2_factory=make_minimax(7),
            num_games=fast,
            description=f"RL @ {label} vs Minimax (depth 7)",
        ))

    # ===================================================================
    # 4. MINIMAX DEPTH SCALING
    #    RL vs minimax variants = fast -> 100 games
    #    MCTS vs minimax variants = slow -> 20 games
    # ===================================================================

    for depth in [3, 5, 9]:
        if depth == 9 and args.skip_slow:
            print(f"  [SKIP] Minimax depth 9 matchups (--skip-slow)")
            continue

        n += 1
        matchups.append(Matchup(
            number=n, category="Minimax Depth Scaling",
            agent1_factory=make_rl_best,
            agent2_factory=make_minimax(depth),
            num_games=fast,
            description=f"RL (best) vs Minimax (depth {depth})",
        ))

        n += 1
        matchups.append(Matchup(
            number=n, category="Minimax Depth Scaling",
            agent1_factory=make_mcts(700),
            agent2_factory=make_minimax(depth),
            num_games=mcts_700,
            description=f"MCTS (700 iter) vs Minimax (depth {depth})",
        ))

    # ===================================================================
    # 5. MCTS ITERATION SCALING
    #    Each tier gets games proportional to speed
    # ===================================================================

    for iters, games in [(200, mcts_200), (1000, mcts_1000), (2000, mcts_2000)]:
        if iters == 2000 and args.skip_slow:
            print(f"  [SKIP] MCTS 2000 matchups (--skip-slow)")
            continue

        n += 1
        matchups.append(Matchup(
            number=n, category="MCTS Iteration Scaling",
            agent1_factory=make_mcts(iters),
            agent2_factory=make_minimax(7),
            num_games=games,
            description=f"MCTS ({iters} iter) vs Minimax (depth 7)",
        ))

        n += 1
        matchups.append(Matchup(
            number=n, category="MCTS Iteration Scaling",
            agent1_factory=make_mcts(iters),
            agent2_factory=make_rl_best,
            num_games=games,
            description=f"MCTS ({iters} iter) vs RL (best)",
        ))

    return matchups


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------

def run_tournament(matchups: List[Matchup], output_dir: str) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    tournament_start = time.perf_counter()

    total_matchups = len(matchups)
    total_games = sum(m.num_games for m in matchups)

    print("=" * 70)
    print("CONNECT 4 — PHASE 1 TOURNAMENT")
    print("=" * 70)
    print(f"  Matchups:    {total_matchups}")
    print(f"  Total games: {total_games}")
    print(f"  Output dir:  {output_dir}")
    print("=" * 70)
    print()

    games_completed = 0

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
        games_completed += matchup.num_games

        print_evaluation_summary(summary)

        result = {
            "matchup_number": matchup.number,
            "category": matchup.category,
            "description": matchup.description,
            "num_games": matchup.num_games,
            "duration_seconds": round(matchup_duration, 2),
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

        # Save after every matchup — never lose data
        _save_results(all_results, output_dir)

        elapsed = time.perf_counter() - tournament_start
        avg_time_per_game = elapsed / games_completed if games_completed else 0
        remaining_games = total_games - games_completed
        eta_seconds = avg_time_per_game * remaining_games

        print(f"\n  Matchup #{matchup.number} done in {matchup_duration:.1f}s")
        print(f"  Progress: {games_completed}/{total_games} games "
              f"({games_completed / total_games * 100:.0f}%)")
        print(f"  Elapsed: {elapsed / 3600:.1f}h | "
              f"ETA: ~{eta_seconds / 3600:.1f}h remaining")
        print()

    tournament_duration = time.perf_counter() - tournament_start

    tournament_data = {
        "tournament_date": datetime.now().isoformat(),
        "total_matchups": total_matchups,
        "total_games": total_games,
        "total_duration_seconds": round(tournament_duration, 2),
        "total_duration_hours": round(tournament_duration / 3600, 2),
        "results": all_results,
    }

    _save_results(all_results, output_dir, tournament_data)
    return tournament_data


def _save_results(
    results: List[Dict],
    output_dir: str,
    tournament_data: Optional[Dict] = None,
) -> None:
    data = tournament_data or {"results": results}
    path = os.path.join(output_dir, "tournament_results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

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

    # P1/P2 split
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

    # Timing
    print("\nAVERAGE MOVE TIMES:")
    for r in core:
        print(f"  {r['description']}:")
        print(f"    {r['agent1_name']}: {r['avg_move_time_agent1']:.4f}s/move")
        print(f"    {r['agent2_name']}: {r['avg_move_time_agent2']:.4f}s/move")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Connect 4 Phase 1 Tournament Runner"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--fast-games", type=int, default=100,
        help="Games for fast matchups: RL vs Minimax, baselines w/o MCTS (default: 100)",
    )
    parser.add_argument(
        "--mcts-games", type=int, default=20,
        help="Games for MCTS-700 matchups (default: 20)",
    )
    parser.add_argument(
        "--skip-slow", action="store_true",
        help="Skip MCTS-2000 and Minimax depth 9 matchups",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: 10 fast / 6 MCTS games per matchup",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.fast_games = 10
        args.mcts_games = 6
        print("[QUICK MODE] 10 fast / 6 MCTS games per matchup\n")

    if not os.path.exists(RL_BEST_MODEL):
        print(f"ERROR: RL best model not found at {RL_BEST_MODEL}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)

    matchups = build_matchups(args)

    total_games = sum(m.num_games for m in matchups)
    print(f"\nScheduled {len(matchups)} matchups ({total_games} total games)\n")

    tournament_data = run_tournament(matchups, args.output_dir)
    print_tournament_summary(tournament_data)

    out_path = os.path.join(args.output_dir, "tournament_results.json")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()