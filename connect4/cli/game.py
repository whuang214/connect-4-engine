"""``play``, ``ui``, and ``eval`` subcommands."""

from __future__ import annotations

import argparse
import sys

from connect4.agents.factory import DEFAULT_RL_CHECKPOINT
from connect4.engine import Connect4


def _add_agent_args(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--agent1", type=str, default="human")
    subparser.add_argument("--agent2", type=str, default="mcts")
    subparser.add_argument("--name1",  type=str, default=None)
    subparser.add_argument("--name2",  type=str, default=None)
    subparser.add_argument("--iterations1", type=int, default=None,
                           help="strength for a bare 'mcts'/'minimax' spec "
                                "(iterations / depth); suffixed specs like "
                                "mcts-700 carry their own")
    subparser.add_argument("--iterations2", type=int, default=None,
                           help="strength for a bare 'mcts'/'minimax' spec "
                                "(iterations / depth); suffixed specs like "
                                "mcts-700 carry their own")
    subparser.add_argument("--model1", type=str, default=None,
                           help="run folder name inside runs/ for an RL agent")
    subparser.add_argument("--model2", type=str, default=None,
                           help="run folder name inside runs/ for an RL agent")
    subparser.add_argument(
        "--checkpoint1", type=str, default=DEFAULT_RL_CHECKPOINT,
        choices=["best", "final"],
    )
    subparser.add_argument(
        "--checkpoint2", type=str, default=DEFAULT_RL_CHECKPOINT,
        choices=["best", "final"],
    )
    subparser.add_argument("--model-path1", type=str, default=None,
                           help="explicit path to an RL .pt checkpoint")
    subparser.add_argument("--model-path2", type=str, default=None,
                           help="explicit path to an RL .pt checkpoint")


def configure_play_parser(parser: argparse.ArgumentParser) -> None:
    _add_agent_args(parser)
    parser.add_argument("--no-render", action="store_true")
    parser.set_defaults(func=run_play)


def configure_ui_parser(parser: argparse.ArgumentParser) -> None:
    _add_agent_args(parser)
    parser.set_defaults(func=run_ui)


def configure_eval_parser(parser: argparse.ArgumentParser) -> None:
    _add_agent_args(parser)
    parser.add_argument("--games",              type=int,  default=10)
    parser.add_argument("--render",             action="store_true")
    parser.add_argument("--no-print-each-game", action="store_true",
                        help="suppress per-game result lines (printed by default)")
    parser.add_argument("--print-moves",        action="store_true")
    parser.set_defaults(func=run_eval)


def print_run_header(mode: str, agent1, agent2, extra: str | None = None) -> None:
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Player 1: {agent1.name}")
    print(f"Player 2: {agent2.name}")
    if extra:
        print(extra)
    print("=" * 60)


def play_game(agent1, agent2, render: bool = True):
    game = Connect4()

    while not game.is_terminal():
        if render:
            game.render()

        current_agent = agent1 if game.current_player == 1 else agent2
        move = current_agent.choose_action(game.clone())

        print(f"{current_agent.name} chooses column {move}")
        game.make_move(move)

    if render:
        game.render()

    if game.winner is not None:
        print(f"Player {game.winner} wins!")
    else:
        print("Game ended in a draw.")

    return game.winner


def _create_agents(args: argparse.Namespace):
    from connect4.agents.factory import create_agent

    agent1 = create_agent(
        args.agent1, args.name1, args.iterations1,
        args.model1, args.checkpoint1, args.model_path1,
    )
    agent2 = create_agent(
        args.agent2, args.name2, args.iterations2,
        args.model2, args.checkpoint2, args.model_path2,
    )
    return agent1, agent2


def run_play(args: argparse.Namespace) -> None:
    agent1, agent2 = _create_agents(args)

    print_run_header(
        mode="CLI Play", agent1=agent1, agent2=agent2,
        extra=f"Render: {not args.no_render}",
    )

    play_game(agent1, agent2, render=not args.no_render)


def run_ui(args: argparse.Namespace) -> None:
    try:
        from connect4.ui.game_ui import GameUI
    except ImportError:
        print(
            "The graphical UI requires pygame, which is not installed.\n"
            "Install it with:  pip install pygame   (pygame-ce on Windows ARM64)"
        )
        sys.exit(1)

    from connect4.agents.factory import create_agent, parse_agent_config

    parsed_agent1_type, _ = parse_agent_config(args.agent1, args.iterations1)
    parsed_agent2_type, _ = parse_agent_config(args.agent2, args.iterations2)

    agent1 = None if parsed_agent1_type == "human" else create_agent(
        args.agent1, args.name1, args.iterations1,
        args.model1, args.checkpoint1, args.model_path1,
    )
    agent2 = None if parsed_agent2_type == "human" else create_agent(
        args.agent2, args.name2, args.iterations2,
        args.model2, args.checkpoint2, args.model_path2,
    )

    p1_name = "Human" if agent1 is None else agent1.name
    p2_name = "Human" if agent2 is None else agent2.name

    print("=" * 60)
    print("Mode: UI")
    print(f"Player 1: {p1_name}")
    print(f"Player 2: {p2_name}")
    print("=" * 60)

    GameUI(player1_agent=agent1, player2_agent=agent2).run()


def run_eval(args: argparse.Namespace) -> None:
    from connect4.evaluation.evaluate import evaluate_agents, print_evaluation_summary

    agent1, agent2 = _create_agents(args)

    print_run_header(
        mode="Evaluation", agent1=agent1, agent2=agent2,
        extra=(
            f"Games: {args.games} | Render: {args.render} | "
            f"PrintEachGame: {not args.no_print_each_game}"
        ),
    )

    summary = evaluate_agents(
        game_class=Connect4,
        agent1=agent1,
        agent2=agent2,
        num_games=args.games,
        render=args.render,
        print_each_game=not args.no_print_each_game,
        print_moves=args.print_moves,
    )

    print_evaluation_summary(summary)
