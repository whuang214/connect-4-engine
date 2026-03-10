from engine import Connect4
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent



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
    agent1 = HumanAgent("HumanAgent")
    agent2 = RandomAgent("RandomAgent")

    play_game(agent1, agent2)