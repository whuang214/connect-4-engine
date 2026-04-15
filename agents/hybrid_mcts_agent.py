"""
Hybrid MCTS + Neural Network Agent for Connect 4.

Instead of random rollouts (slow, noisy), this agent evaluates leaf nodes
with the trained RL value network (fast, informed). This combines the
strategic depth of MCTS tree search with the learned position evaluation
of the RL agent.

Key differences from pure MCTS:
- No rollouts: leaf nodes are evaluated by the neural network in ~1ms
- Each simulation is much faster: no need to play to terminal state
- Evaluations are more accurate: network learned from 100k+ games
- Fewer simulations needed for strong play

This is essentially a simplified AlphaZero approach (Phase 2 prototype).
"""

import math
import random
import torch
import numpy as np

from agents.base_agent import BaseAgent
from models.value_network import ValueNetwork, ValueNetworkSmall, encode_board


class HybridNode:
    """Node in the MCTS search tree."""

    __slots__ = [
        "parent", "move", "player_just_moved",
        "children", "untried_moves",
        "visits", "value_sum",
    ]

    def __init__(self, parent=None, move=None, player_just_moved=None):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.children = []
        self.untried_moves = None
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def uct_score(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value_sum / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_weight: float):
        return max(self.children, key=lambda c: c.uct_score(exploration_weight))


class HybridMCTSAgent(BaseAgent):
    """
    Hybrid MCTS agent that uses a trained neural network for leaf evaluation
    instead of random rollouts (Phase 2 prototype).

    Parameters
    ----------
    name : str
        Display name for the agent.
    model_path : str
        Path to the trained value network checkpoint.
    iterations : int
        Number of MCTS simulations per move decision.
    exploration_weight : float
        UCT exploration constant (default sqrt(2)).
    small_network : bool
        Whether to use the small network architecture.
    device : torch.device
        Device to run the network on.
    """

    def __init__(
        self,
        name=None,
        model_path=None,
        model=None,
        iterations=800,
        exploration_weight=1.414,
        small_network=True,
        device=None,
    ):
        name = name or f"HybridMCTS-{iterations}"
        super().__init__(name)

        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.center_order = [3, 2, 4, 1, 5, 0, 6]

        # Load the trained value network
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path, small_network)
        else:
            raise ValueError("Must provide either model or model_path")

        self.model.eval()

        # Stats tracking
        self.reset_stats()

    def _load_model(self, path, small_network):
        model = ValueNetworkSmall() if small_network else ValueNetwork()
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        return model

    def reset_stats(self):
        self.total_simulations = 0
        self.total_nn_evals = 0
        self.moves_chosen = 0
        self.immediate_win_hits = 0
        self.immediate_block_hits = 0

    def choose_action(self, game) -> int:
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves available.")

        if len(legal_moves) == 1:
            self.moves_chosen += 1
            return legal_moves[0]

        root_player = game.current_player
        opponent = 2 if root_player == 1 else 1

        # Tactical overrides: instant win / block
        win_move = self._find_immediate_win(game, root_player)
        if win_move is not None:
            self.moves_chosen += 1
            self.immediate_win_hits += 1
            return win_move

        block_move = self._find_immediate_win(game, opponent)
        if block_move is not None:
            self.moves_chosen += 1
            self.immediate_block_hits += 1
            return block_move

        # Run MCTS with neural network evaluation
        root = HybridNode(parent=None, move=None, player_just_moved=opponent)
        sims = 0

        for _ in range(self.iterations):
            node = root
            applied_moves = []

            # --- Selection ---
            while not game.is_terminal():
                if node.untried_moves is None:
                    node.untried_moves = self._get_ordered_moves(game)

                if not node.is_fully_expanded():
                    break
                if not node.children:
                    break

                node = node.best_child(self.exploration_weight)
                game.make_move(node.move)
                applied_moves.append(node.move)

            # --- Expansion ---
            if not game.is_terminal():
                if node.untried_moves is None:
                    node.untried_moves = self._get_ordered_moves(game)

                if node.untried_moves:
                    move = node.untried_moves.pop(0)
                    player_making_move = game.current_player

                    game.make_move(move)
                    applied_moves.append(move)

                    child = HybridNode(
                        parent=node,
                        move=move,
                        player_just_moved=player_making_move,
                    )
                    node.children.append(child)
                    node = child

            # --- Evaluation (neural network instead of rollout) ---
            if game.is_terminal():
                # Terminal state: use exact outcome
                if game.winner is None:
                    value_for_current = 0.0
                elif game.winner == game.current_player:
                    value_for_current = 1.0
                else:
                    value_for_current = -1.0
                # But since game is terminal, current_player is whoever
                # would move next. The last person who moved is node.player_just_moved
                if game.winner is None:
                    nn_value = 0.0
                elif game.winner == root_player:
                    nn_value = 1.0
                else:
                    nn_value = -1.0
            else:
                # Non-terminal: evaluate with neural network
                nn_value = self._evaluate_position(game, root_player)
                self.total_nn_evals += 1

            sims += 1

            # --- Backpropagation ---
            # Propagate value from root_player's perspective
            while node is not None:
                node.visits += 1
                if node.player_just_moved == root_player:
                    node.value_sum += nn_value
                else:
                    node.value_sum += (1.0 - nn_value)  # opponent's value is inverted
                node = node.parent

            # Undo all moves to restore game state
            for _ in applied_moves:
                game.undo_move()

        self.total_simulations += sims
        self.moves_chosen += 1

        # Select best move by visit count (most robust)
        if not root.children:
            return legal_moves[0]

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _evaluate_position(self, game, root_player) -> float:
        """
        Evaluate the position using the neural network.
        Returns value from root_player's perspective in [0, 1].
        """
        with torch.no_grad():
            encoded = encode_board(game)
            tensor = torch.tensor(
                encoded, dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            raw_value = self.model(tensor).item()  # in [-1, 1] from current player's POV

        # raw_value is from current_player's perspective
        # Convert to root_player's perspective in [0, 1]
        if game.current_player == root_player:
            return (raw_value + 1.0) / 2.0  # map [-1,1] to [0,1]
        else:
            return (-raw_value + 1.0) / 2.0  # opponent's value, inverted

    def _find_immediate_win(self, game, player) -> int | None:
        """Return a column that gives player an immediate win, or None."""
        legal_moves = game.get_legal_moves()
        for col in self.center_order:
            if col not in legal_moves:
                continue
            if game.current_player == player:
                game.make_move(col)
                won = game.winner == player
                game.undo_move()
            else:
                tmp = game.clone()
                tmp.current_player = player
                tmp.make_move(col)
                won = tmp.winner == player
            if won:
                return col
        return None

    def _get_ordered_moves(self, game) -> list:
        """Return legal moves ordered by center preference."""
        legal = game.get_legal_moves()
        return [m for m in self.center_order if m in legal]

    def get_stats(self) -> dict:
        return {
            "Decisions": {
                "Moves chosen": self.moves_chosen,
                "Immediate wins": self.immediate_win_hits,
                "Immediate blocks": self.immediate_block_hits,
            },
            "Search": {
                "Total simulations": self.total_simulations,
                "Total NN evaluations": self.total_nn_evals,
                "Avg sims per move": (
                    self.total_simulations / max(self.moves_chosen, 1)
                ),
            },
        }
