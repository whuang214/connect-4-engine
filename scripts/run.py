import argparse
import os

from engine import Connect4
from evaluation.evaluate import evaluate_agents, print_evaluation_summary

from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts_agent import MCTSAgent
from agents.rl_agent import RLAgent


DEFAULT_RL_MODEL_PATH = "runs/run1/final_model.pt"


def parse_agent_config(agent_type: str, iterations: int) -> tuple[str, int]:
    agent_type = agent_type.lower().strip()

    if agent_type.startswith("mcts-"):
        _, value = agent_type.split("-", 1)
        if not value.isdigit():
            raise ValueError(
                f"Invalid MCTS agent format: '{agent_type}'. "
                f"Expected format like 'mcts-5000'."
            )
        return "mcts", int(value)

    return agent_type, iterations


def resolve_rl_model_path(model_path: str | None) -> str:
    """
    Return the RL model path to use.
    If no path is provided, fall back to the default final model path.
    """
    path = model_path or DEFAULT_RL_MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL model file not found: '{path}'. "
            f"Pass --model1/--model2 explicitly or place the model at "
            f"'{DEFAULT_RL_MODEL_PATH}'."
        )

    return path


def create_agent(
    agent_type: str,
    name: str | None = None,
    iterations: int = 1000,
    model_path: str | None = None,
):
    agent_type, iterations = parse_agent_config(agent_type, iterations)

    if agent_type == "human":
        return HumanAgent(name=name) if name else HumanAgent()

    if agent_type == "random":
        return RandomAgent(name=name) if name else RandomAgent()

    if agent_type == "rule":
        return RuleBasedAgent(name=name) if name else RuleBasedAgent()

    if agent_type == "mcts":
        return (
            MCTSAgent(name=name, iterations=iterations)
            if name
            else MCTSAgent(iterations=iterations)
        )

    if agent_type == "rl":
        resolved_model_path = resolve_rl_model_path(model_path)
        return (
            RLAgent(name=name, model_path=resolved_model_path)
            if name
            else RLAgent(model_path=resolved_model_path)
        )

    raise ValueError(
        f"Unknown agent type: '{agent_type}'. "
        f"Use: human, random, rule, mcts, mcts-<iterations>, rl"
    )


def print_run_header(mode: str, agent1, agent2, extra: str | None = None):
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
        move = current_agent.choose_action(game)

        print(f"{current_agent.name} chooses column {move}")
        game.make_move(move)

    if render:
        game.render()

    if game.winner is not None:
        print(f"Player {game.winner} wins!")
    else:
        print("Game ended in a draw.")

    return game.winner


def run_play_mode(args):
    agent1 = create_agent(args.agent1, args.name1, args.iterations1, args.model1)
    agent2 = create_agent(args.agent2, args.name2, args.iterations2, args.model2)

    print_run_header(
        mode="CLI Play",
        agent1=agent1,
        agent2=agent2,
        extra=f"Render: {not args.no_render}",
    )

    play_game(agent1, agent2, render=not args.no_render)


def run_ui_mode(args):
    from ui.game_ui import GameUI  # lazy import for pygame

    parsed_agent1_type, _ = parse_agent_config(args.agent1, args.iterations1)
    parsed_agent2_type, _ = parse_agent_config(args.agent2, args.iterations2)

    agent1 = None if parsed_agent1_type == "human" else create_agent(
        args.agent1, args.name1, args.iterations1, args.model1
    )
    agent2 = None if parsed_agent2_type == "human" else create_agent(
        args.agent2, args.name2, args.iterations2, args.model2
    )

    p1_name = "Human" if agent1 is None else agent1.name
    p2_name = "Human" if agent2 is None else agent2.name

    print("=" * 60)
    print("Mode: UI")
    print(f"Player 1: {p1_name}")
    print(f"Player 2: {p2_name}")
    print("=" * 60)

    GameUI(player1_agent=agent1, player2_agent=agent2).run()


def run_eval_mode(args):
    agent1 = create_agent(args.agent1, args.name1, args.iterations1, args.model1)
    agent2 = create_agent(args.agent2, args.name2, args.iterations2, args.model2)

    print_run_header(
        mode="Evaluation",
        agent1=agent1,
        agent2=agent2,
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

    if hasattr(agent1, "print_stats"):
        print(f"\n{agent1.name} stats:")
        agent1.print_stats()

    if hasattr(agent2, "print_stats"):
        print(f"\n{agent2.name} stats:")
        agent2.print_stats()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Connect 4 runner: CLI, UI, or evaluation."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    def add_agent_args(subparser):
        subparser.add_argument("--agent1", type=str, default="human")
        subparser.add_argument("--agent2", type=str, default="mcts")

        subparser.add_argument("--name1", type=str, default=None)
        subparser.add_argument("--name2", type=str, default=None)

        subparser.add_argument("--iterations1", type=int, default=1000)
        subparser.add_argument("--iterations2", type=int, default=1000)

        # If agent is rl and no model path is passed, it will use:
        # runs/run1/final_model.pt
        subparser.add_argument("--model1", type=str, default=None)
        subparser.add_argument("--model2", type=str, default=None)

    # CLI
    play_parser = subparsers.add_parser("play")
    add_agent_args(play_parser)
    play_parser.add_argument("--no-render", action="store_true")

    # UI
    ui_parser = subparsers.add_parser("ui")
    add_agent_args(ui_parser)

    # EVAL
    eval_parser = subparsers.add_parser("eval")
    add_agent_args(eval_parser)
    eval_parser.add_argument("--games", type=int, default=10)
    eval_parser.add_argument("--render", action="store_true")
    eval_parser.add_argument("--no-print-each-game", action="store_true")
    eval_parser.add_argument("--print-moves", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "play":
        run_play_mode(args)
    elif args.mode == "ui":
        run_ui_mode(args)
    elif args.mode == "eval":
        run_eval_mode(args)
    else:
        parser.error("Unknown mode.")


if __name__ == "__main__":
    main()