from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional


@dataclass
class GameStats:
    game_number: int
    player1_agent: str
    player2_agent: str
    starting_player: int
    winner: Optional[int]
    winner_agent: str
    is_draw: bool
    moves_played: int
    duration_seconds: float

    move_count_p1: int
    move_count_p2: int

    avg_move_time_p1: float
    avg_move_time_p2: float
    total_move_time_p1: float
    total_move_time_p2: float


@dataclass
class EvaluationSummary:
    total_games: int
    agent1_name: str
    agent2_name: str

    agent1_wins: int = 0
    agent2_wins: int = 0
    draws: int = 0

    agent1_as_p1_wins: int = 0
    agent1_as_p2_wins: int = 0
    agent2_as_p1_wins: int = 0
    agent2_as_p2_wins: int = 0

    avg_game_length: float = 0.0
    min_game_length: int = 0
    max_game_length: int = 0

    avg_move_time_agent1: float = 0.0
    avg_move_time_agent2: float = 0.0
    total_move_time_agent1: float = 0.0
    total_move_time_agent2: float = 0.0

    total_moves_agent1: int = 0
    total_moves_agent2: int = 0

    avg_game_duration: float = 0.0

    agent1_internal_stats: Dict[str, Any] = field(default_factory=dict)
    agent2_internal_stats: Dict[str, Any] = field(default_factory=dict)

    game_results: List[GameStats] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_games": self.total_games,
            "agent1_name": self.agent1_name,
            "agent2_name": self.agent2_name,
            "agent1_wins": self.agent1_wins,
            "agent2_wins": self.agent2_wins,
            "draws": self.draws,
            "agent1_win_rate": self.agent1_wins / self.total_games if self.total_games else 0.0,
            "agent2_win_rate": self.agent2_wins / self.total_games if self.total_games else 0.0,
            "draw_rate": self.draws / self.total_games if self.total_games else 0.0,
            "agent1_as_p1_wins": self.agent1_as_p1_wins,
            "agent1_as_p2_wins": self.agent1_as_p2_wins,
            "agent2_as_p1_wins": self.agent2_as_p1_wins,
            "agent2_as_p2_wins": self.agent2_as_p2_wins,
            "avg_game_length": self.avg_game_length,
            "min_game_length": self.min_game_length,
            "max_game_length": self.max_game_length,
            "avg_move_time_agent1": self.avg_move_time_agent1,
            "avg_move_time_agent2": self.avg_move_time_agent2,
            "total_move_time_agent1": self.total_move_time_agent1,
            "total_move_time_agent2": self.total_move_time_agent2,
            "total_moves_agent1": self.total_moves_agent1,
            "total_moves_agent2": self.total_moves_agent2,
            "avg_game_duration": self.avg_game_duration,
            "agent1_internal_stats": self.agent1_internal_stats,
            "agent2_internal_stats": self.agent2_internal_stats,
            "game_results": self.game_results,
        }


def _safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def _reset_agent_stats_if_supported(agent: Any) -> None:
    if hasattr(agent, "reset_stats") and callable(agent.reset_stats):
        agent.reset_stats()


def _get_agent_stats_if_supported(agent: Any) -> Dict[str, Any]:
    if hasattr(agent, "get_stats") and callable(agent.get_stats):
        stats = agent.get_stats()
        if isinstance(stats, dict):
            return stats
    return {}


def play_one_game(
    game_class,
    player1_agent,
    player2_agent,
    game_number: int = 1,
    render: bool = False,
    print_moves: bool = False,
) -> GameStats:
    game = game_class()

    agents_by_player = {
        1: player1_agent,
        2: player2_agent,
    }

    move_times = {
        1: [],
        2: [],
    }

    moves_played = 0
    game_start_time = time.perf_counter()

    while not game.is_terminal():
        current_player = game.current_player
        current_agent = agents_by_player[current_player]

        state_for_agent = game.clone()

        start = time.perf_counter()
        move = current_agent.choose_action(state_for_agent)
        end = time.perf_counter()

        if not game.is_legal_move(move):
            raise ValueError(
                f"Illegal move {move} chosen by {current_agent.name}. "
                f"Legal moves: {game.get_legal_moves()}"
            )

        move_times[current_player].append(end - start)
        result = game.make_move(move)
        moves_played += 1

        if print_moves:
            print(
                f"Game {game_number} | "
                f"Player {current_player} ({current_agent.name}) -> column {move}"
            )

        if render:
            game.render()

        if result.done:
            break

    game_end_time = time.perf_counter()
    duration_seconds = game_end_time - game_start_time

    if game.winner is None:
        winner_agent = "Draw"
        is_draw = True
    elif game.winner == 1:
        winner_agent = player1_agent.name
        is_draw = False
    elif game.winner == 2:
        winner_agent = player2_agent.name
        is_draw = False
    else:
        raise ValueError(f"Unexpected winner value: {game.winner}")

    move_count_p1 = len(move_times[1])
    move_count_p2 = len(move_times[2])

    total_move_time_p1 = sum(move_times[1])
    total_move_time_p2 = sum(move_times[2])

    return GameStats(
        game_number=game_number,
        player1_agent=player1_agent.name,
        player2_agent=player2_agent.name,
        starting_player=1,
        winner=game.winner,
        winner_agent=winner_agent,
        is_draw=is_draw,
        moves_played=moves_played,
        duration_seconds=duration_seconds,
        move_count_p1=move_count_p1,
        move_count_p2=move_count_p2,
        avg_move_time_p1=(total_move_time_p1 / move_count_p1) if move_count_p1 else 0.0,
        avg_move_time_p2=(total_move_time_p2 / move_count_p2) if move_count_p2 else 0.0,
        total_move_time_p1=total_move_time_p1,
        total_move_time_p2=total_move_time_p2,
    )


def evaluate_agents(
    game_class,
    agent1,
    agent2,
    num_games: int = 100,
    render: bool = False,
    print_each_game: bool = True,
    print_moves: bool = False,
) -> EvaluationSummary:
    """
    Runs multiple games between two agents.
    Alternates which agent is Player 1 to reduce first-player bias.
    """

    # Reset ONCE for the whole evaluation
    _reset_agent_stats_if_supported(agent1)
    _reset_agent_stats_if_supported(agent2)

    summary = EvaluationSummary(
        total_games=num_games,
        agent1_name=agent1.name,
        agent2_name=agent2.name,
    )

    all_game_lengths: List[int] = []
    all_game_durations: List[float] = []

    for game_number in range(1, num_games + 1):
        if game_number % 2 == 1:
            p1_agent = agent1
            p2_agent = agent2
            agent1_is_player = 1
        else:
            p1_agent = agent2
            p2_agent = agent1
            agent1_is_player = 2

        game_stats = play_one_game(
            game_class=game_class,
            player1_agent=p1_agent,
            player2_agent=p2_agent,
            game_number=game_number,
            render=render,
            print_moves=print_moves,
        )

        summary.game_results.append(game_stats)
        all_game_lengths.append(game_stats.moves_played)
        all_game_durations.append(game_stats.duration_seconds)

        if agent1_is_player == 1:
            summary.total_move_time_agent1 += game_stats.total_move_time_p1
            summary.total_move_time_agent2 += game_stats.total_move_time_p2
            summary.total_moves_agent1 += game_stats.move_count_p1
            summary.total_moves_agent2 += game_stats.move_count_p2
        else:
            summary.total_move_time_agent1 += game_stats.total_move_time_p2
            summary.total_move_time_agent2 += game_stats.total_move_time_p1
            summary.total_moves_agent1 += game_stats.move_count_p2
            summary.total_moves_agent2 += game_stats.move_count_p1

        if game_stats.is_draw:
            summary.draws += 1
        else:
            if agent1_is_player == 1:
                if game_stats.winner == 1:
                    summary.agent1_wins += 1
                    summary.agent1_as_p1_wins += 1
                elif game_stats.winner == 2:
                    summary.agent2_wins += 1
                    summary.agent2_as_p2_wins += 1
            else:
                if game_stats.winner == 1:
                    summary.agent2_wins += 1
                    summary.agent2_as_p1_wins += 1
                elif game_stats.winner == 2:
                    summary.agent1_wins += 1
                    summary.agent1_as_p2_wins += 1

        if print_each_game:
            starter_name = p1_agent.name
            print(
                f"Game {game_number}/{num_games} | "
                f"P1: {p1_agent.name} vs P2: {p2_agent.name} | "
                f"Starter: {starter_name} | "
                f"Winner: {game_stats.winner_agent} | "
                f"Moves: {game_stats.moves_played} | "
                f"Duration: {game_stats.duration_seconds:.4f}s"
            )

    summary.avg_game_length = _safe_mean(all_game_lengths)
    summary.min_game_length = min(all_game_lengths) if all_game_lengths else 0
    summary.max_game_length = max(all_game_lengths) if all_game_lengths else 0
    summary.avg_game_duration = _safe_mean(all_game_durations)

    summary.avg_move_time_agent1 = (
        summary.total_move_time_agent1 / summary.total_moves_agent1
        if summary.total_moves_agent1 else 0.0
    )
    summary.avg_move_time_agent2 = (
        summary.total_move_time_agent2 / summary.total_moves_agent2
        if summary.total_moves_agent2 else 0.0
    )

    summary.agent1_internal_stats = _get_agent_stats_if_supported(agent1)
    summary.agent2_internal_stats = _get_agent_stats_if_supported(agent2)

    return summary


def print_nested_stats(title: str, stats: Dict[str, Any]) -> None:
    if not stats:
        return

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for section, values in stats.items():
        print(f"\n{section}:")
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {values}")


def print_evaluation_summary(summary: EvaluationSummary) -> None:
    total_games = summary.total_games
    agent1_win_rate = summary.agent1_wins / total_games if total_games else 0.0
    agent2_win_rate = summary.agent2_wins / total_games if total_games else 0.0
    draw_rate = summary.draws / total_games if total_games else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Agents: {summary.agent1_name} vs {summary.agent2_name}")
    print(f"Total games: {summary.total_games}")

    print("\nResults:")
    print(f"  {summary.agent1_name} wins: {summary.agent1_wins}")
    print(f"  {summary.agent2_name} wins: {summary.agent2_wins}")
    print(f"  Draws: {summary.draws}")

    print("\nWin rates:")
    print(f"  {summary.agent1_name}: {agent1_win_rate:.2%}")
    print(f"  {summary.agent2_name}: {agent2_win_rate:.2%}")
    print(f"  Draw rate: {draw_rate:.2%}")

    print("\nFirst-player split:")
    print(f"  {summary.agent1_name} as P1 wins: {summary.agent1_as_p1_wins}")
    print(f"  {summary.agent1_name} as P2 wins: {summary.agent1_as_p2_wins}")
    print(f"  {summary.agent2_name} as P1 wins: {summary.agent2_as_p1_wins}")
    print(f"  {summary.agent2_name} as P2 wins: {summary.agent2_as_p2_wins}")

    print("\nGame length:")
    print(f"  Average moves/game: {summary.avg_game_length:.2f}")
    print(f"  Shortest game: {summary.min_game_length}")
    print(f"  Longest game: {summary.max_game_length}")

    print("\nTiming:")
    print(f"  Average game duration: {summary.avg_game_duration:.4f}s")
    print(f"  {summary.agent1_name} total moves: {summary.total_moves_agent1}")
    print(f"  {summary.agent1_name} avg move time: {summary.avg_move_time_agent1:.6f}s")
    print(f"  {summary.agent1_name} total move time: {summary.total_move_time_agent1:.4f}s")
    print("")
    print(f"  {summary.agent2_name} total moves: {summary.total_moves_agent2}")
    print(f"  {summary.agent2_name} avg move time: {summary.avg_move_time_agent2:.6f}s")
    print(f"  {summary.agent2_name} total move time: {summary.total_move_time_agent2:.4f}s")

    print_nested_stats(f"{summary.agent1_name} INTERNAL STATS", summary.agent1_internal_stats)
    print_nested_stats(f"{summary.agent2_name} INTERNAL STATS", summary.agent2_internal_stats)