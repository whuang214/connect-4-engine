"""
Evaluation utilities for the RL Agent.

Runs the trained agent against baseline opponents and collects
win rate, draw rate, and average decision time metrics.
"""

import time
import numpy as np
from engine import Connect4


def evaluate_agent(agent, opponent, num_games=100, alternate_start=True, verbose=False):
    """
    Evaluate an agent against an opponent over multiple games.

    Args:
        agent: the agent to evaluate
        opponent: the opponent agent
        num_games: number of games to play
        alternate_start: if True, agent plays as Player 1 for half
                         the games and Player 2 for the other half
        verbose: print per-game results

    Returns:
        dict with evaluation metrics:
            - win_rate: fraction of games won by agent
            - loss_rate: fraction of games lost
            - draw_rate: fraction of draws
            - wins, losses, draws: raw counts
            - avg_moves: average game length
            - avg_decision_time: average time per agent decision (seconds)
            - p1_wins, p2_wins: wins when playing as each side
    """
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    decision_times = []
    p1_wins = 0
    p2_wins = 0

    for game_idx in range(num_games):
        # Alternate who goes first
        if alternate_start:
            agent_is_p1 = (game_idx % 2 == 0)
        else:
            agent_is_p1 = True

        if agent_is_p1:
            p1, p2 = agent, opponent
        else:
            p1, p2 = opponent, agent

        game = Connect4()
        move_count = 0

        while not game.is_terminal():
            if game.current_player == 1:
                current_agent = p1
            else:
                current_agent = p2

            # Time the agent's decisions (not the opponent's)
            is_our_agent = (current_agent is agent)

            if is_our_agent:
                start_time = time.perf_counter()

            move = current_agent.choose_action(game)

            if is_our_agent:
                elapsed = time.perf_counter() - start_time
                decision_times.append(elapsed)

            game.make_move(move)
            move_count += 1

        total_moves += move_count

        # Determine outcome from agent's perspective
        if game.winner is None:
            draws += 1
            outcome = "draw"
        elif (game.winner == 1 and agent_is_p1) or (game.winner == 2 and not agent_is_p1):
            wins += 1
            outcome = "win"
            if agent_is_p1:
                p1_wins += 1
            else:
                p2_wins += 1
        else:
            losses += 1
            outcome = "loss"

        if verbose:
            side = "P1" if agent_is_p1 else "P2"
            print(f"  Game {game_idx + 1}: {agent.name} ({side}) -> {outcome} in {move_count} moves")

    total = num_games
    avg_time = np.mean(decision_times) if decision_times else 0.0

    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "avg_moves": total_moves / total,
        "avg_decision_time": avg_time,
        "median_decision_time": np.median(decision_times) if decision_times else 0.0,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
    }


def run_evaluation_suite(agent, opponents_dict, num_games=100, verbose=False):
    """
    Run evaluation against multiple opponents.

    Args:
        agent: the agent to evaluate
        opponents_dict: dict of {name: opponent_agent}
        num_games: games per opponent
        verbose: print details

    Returns:
        dict of {opponent_name: results_dict}
    """
    results = {}
    for opp_name, opponent in opponents_dict.items():
        if verbose:
            print(f"\n{agent.name} vs {opp_name} ({num_games} games):")
        result = evaluate_agent(agent, opponent, num_games=num_games, verbose=verbose)
        results[opp_name] = result

        print(f"  vs {opp_name}: "
              f"W={result['wins']} L={result['losses']} D={result['draws']} "
              f"(WR={result['win_rate']:.1%}) "
              f"avg_time={result['avg_decision_time']*1000:.1f}ms")

    return results