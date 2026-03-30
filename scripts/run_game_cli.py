from engine import Connect4
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts_agent import MCTSAgent
from agents.rl_agent import RLAgent

def play_game(agent1, agent2, render=True):
    game = Connect4()

    while not game.is_terminal():
        if render:
            game.render()

        if game.current_player == 1:
            current_agent = agent1
        else:
            current_agent = agent2

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


if __name__ == "__main__":
    mcts_agent = MCTSAgent("MCTSAI", iterations=10000)
    mcts_agent2 = MCTSAgent("MCTSAI2", iterations=10000)
    # human_agent = HumanAgent("Human")

    play_game(mcts_agent, mcts_agent2)