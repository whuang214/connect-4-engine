"""``tournament`` subcommand — the full round-robin used for the project report.

Game counts are tuned so a full run finishes overnight (~8-10 hours):
MCTS games take minutes each and get fewer games; non-MCTS games are
near-instant and get 100 games for tight confidence intervals.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--rl-run-dir", type=str, default=os.path.join("runs", "rl_pure_selfplay_v3"),
        help="RL run directory containing best_model.pt (default: runs/rl_pure_selfplay_v3)",
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
    parser.set_defaults(func=run_tournament)


def build_matchups(args: argparse.Namespace) -> List:
    from connect4.agents.factory import create_agent
    from connect4.agents.mcts import MCTSAgent
    from connect4.agents.minimax import MinimaxAgent
    from connect4.agents.random import RandomAgent
    from connect4.agents.rule_based import RuleBasedAgent
    from connect4.evaluation.tournament import Matchup

    rl_best_model = os.path.join(args.rl_run_dir, "best_model.pt")
    rl_checkpoints = {
        label: os.path.join(args.rl_run_dir, "checkpoints", f"checkpoint_ep{ep}.pt")
        for label, ep in [
            ("100k", 100352), ("200k", 200192), ("300k", 300032),
            ("400k", 400384), ("500k", 500224),
        ]
    }

    matchups: List[Matchup] = []
    n = 0

    # Game counts by speed tier
    fast = args.fast_games       # 100 — RL vs minimax, baselines without MCTS
    mcts_700 = args.mcts_games   # 20  — any matchup with MCTS-700
    mcts_200 = args.mcts_games   # 20  — MCTS-200 scaling
    mcts_1000 = max(args.mcts_games - 6, 14)   # 14
    mcts_2000 = max(args.mcts_games - 10, 10)  # 10

    # --- Agent factories ---
    # temperature=0.3 so RL samples from its policy (not argmax every time).
    # This prevents identical repeated games while still playing near-best moves.

    def make_rl_best():
        from connect4.agents.rl_policy import RLPolicyAgent
        return RLPolicyAgent(name="RL-best", model_path=rl_best_model, temperature=0.3)

    def make_rl_checkpoint(label, path):
        def factory():
            from connect4.agents.rl_policy import RLPolicyAgent
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
    # 3. RL LEARNING CURVE — skipped automatically when the intermediate
    #    checkpoints are not present (only best_model.pt ships in-repo).
    # ===================================================================

    for label, path in sorted(rl_checkpoints.items(), key=lambda x: x[0]):
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


def run_tournament(args: argparse.Namespace) -> None:
    from connect4.evaluation.tournament import print_tournament_summary, run_matchups

    if args.quick:
        args.fast_games = 10
        args.mcts_games = 6
        print("[QUICK MODE] 10 fast / 6 MCTS games per matchup\n")

    rl_best_model = os.path.join(args.rl_run_dir, "best_model.pt")
    if not os.path.exists(rl_best_model):
        print(f"ERROR: RL best model not found at {rl_best_model}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)

    matchups = build_matchups(args)

    total_games = sum(m.num_games for m in matchups)
    print(f"\nScheduled {len(matchups)} matchups ({total_games} total games)\n")

    output_path = os.path.join(args.output_dir, "tournament_results.json")
    tournament_data = run_matchups(matchups, output_path)
    print_tournament_summary(tournament_data)

    print(f"\nResults saved to: {output_path}")
