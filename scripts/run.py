import argparse

from engine import Connect4
from evaluation.evaluate import evaluate_agents, print_evaluation_summary

from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts_agent import MCTSAgent

from ui.game_ui import GameUI


def create_agent(agent_type: str, name: str | None = None,
                 iterations: int = 1000,
                 model_path: str | None = None):
    agent_type = agent_type.lower()
    if agent_type == "human":
        return HumanAgent(name=name) if name else HumanAgent()

    if agent_type == "random":
        return RandomAgent(name=name) if name else RandomAgent()

    if agent_type == "rule":
        return RuleBasedAgent(name=name) if name else RuleBasedAgent()

    if agent_type == "mcts":
        return MCTSAgent(name=name, iterations=iterations) if name else MCTSAgent(iterations=iterations)
    
    if agent_type == "rl":
        pass  # Placeholder for future RL agent implementation

    raise ValueError(f"Unknown agent type: {agent_type}")


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
    agent1 = create_agent(
        agent_type=args.agent1,
        name=args.name1,
        iterations=args.iterations1,
        model_path=args.model1,
    )
    agent2 = create_agent(
        agent_type=args.agent2,
        name=args.name2,
        iterations=args.iterations2,
        model_path=args.model2,
    )

    play_game(agent1, agent2, render=not args.no_render)


def run_ui_mode(args):
    agent1 = None if args.agent1 == "human" else create_agent(
        agent_type=args.agent1,
        name=args.name1,
        iterations=args.iterations1,
        model_path=args.model1,
    )

    agent2 = None if args.agent2 == "human" else create_agent(
        agent_type=args.agent2,
        name=args.name2,
        iterations=args.iterations2,
        model_path=args.model2,
    )

    GameUI(player1_agent=agent1, player2_agent=agent2).run()


def run_eval_mode(args):
    agent1 = create_agent(
        agent_type=args.agent1,
        name=args.name1,
        iterations=args.iterations1,
        model_path=args.model1,
    )
    agent2 = create_agent(
        agent_type=args.agent2,
        name=args.name2,
        iterations=args.iterations2,
        model_path=args.model2,
    )

    summary = evaluate_agents(
        game_class=Connect4,
        agent1=agent1,
        agent2=agent2,
        num_games=args.games,
        render=args.render,
        print_each_game=args.print_each_game,
        print_moves=args.print_moves,
    )

    print_evaluation_summary(summary)

    if hasattr(agent1, "print_stats"):
        print()
        print(f"{agent1.name} stats:")
        agent1.print_stats()

    if hasattr(agent2, "print_stats"):
        print()
        print(f"{agent2.name} stats:")
        agent2.print_stats()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Connect 4 runner: play games, launch UI, or evaluate agents."
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared agent args helper
    def add_agent_args(subparser):
        subparser.add_argument("--agent1", type=str, default="human",
                               choices=["human", "random", "rule", "mcts", "rl"])
        subparser.add_argument("--agent2", type=str, default="mcts",
                               choices=["human", "random", "rule", "mcts", "rl"])

        subparser.add_argument("--name1", type=str, default=None)
        subparser.add_argument("--name2", type=str, default=None)

        subparser.add_argument("--iterations1", type=int, default=1000,
                               help="Used if agent1 is MCTS")
        subparser.add_argument("--iterations2", type=int, default=1000,
                               help="Used if agent2 is MCTS")

        subparser.add_argument("--model1", type=str, default=None,
                               help="Optional model path for RL agent1")
        subparser.add_argument("--model2", type=str, default=None,
                               help="Optional model path for RL agent2")

    # play mode
    play_parser = subparsers.add_parser("play", help="Play a terminal/CLI game")
    add_agent_args(play_parser)
    play_parser.add_argument("--no-render", action="store_true",
                             help="Disable board rendering in terminal")

    # ui mode
    ui_parser = subparsers.add_parser("ui", help="Launch graphical UI")
    add_agent_args(ui_parser)

    # eval mode
    eval_parser = subparsers.add_parser("eval", help="Run evaluation matches")
    add_agent_args(eval_parser)
    eval_parser.add_argument("--games", type=int, default=10)
    eval_parser.add_argument("--render", action="store_true")
    eval_parser.add_argument("--print-each-game", action="store_true")
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