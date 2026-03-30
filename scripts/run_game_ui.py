from ui.game_ui import GameUI
from agents.random_agent import RandomAgent
from agents.human_agent import HumanAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.mcts_agent import MCTSAgent

if __name__ == "__main__":
    # Human vs Random AI
    GameUI(player1_agent=None, player2_agent=MCTSAgent("MCTSAI_50000", iterations=50000)).run()