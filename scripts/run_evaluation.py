from engine import Connect4
from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.minimax_agent import MinimaxAgent
from evaluation.evaluate import evaluate_agents, print_evaluation_summary


def main():
    agent1 = MCTSAgent(name="MCTS-200", iterations=200)
    agent2 = MCTSAgent(name="MCTS-500", iterations=500)

    summary = evaluate_agents(
        game_class=Connect4,
        agent1=agent1,
        agent2=agent2,
        num_games=10,
        render=False,
        print_each_game=True,
        print_moves=False,
    )

    print_evaluation_summary(summary)

    agent1.print_stats()
    agent2.print_stats()


if __name__ == "__main__":
    main()
