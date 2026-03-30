import math
import random
from agents.base_agent import BaseAgent


class MCTSNode:
    def __init__(
        self,
        parent=None,
        move=None,
        player_just_moved=None,
        state_key=None,
    ):
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.state_key = state_key

        self.children = []
        self.untried_moves = None

        self.visits = 0
        self.wins = 0.0
        # wins are from the perspective of player_just_moved:
        #   win  = 1.0
        #   draw = 0.5
        #   loss = 0.0

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def uct_score(self, exploration_weight: float) -> float:
        if self.visits == 0:
            return float("inf")

        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_weight: float):
        return max(self.children, key=lambda child: child.uct_score(exploration_weight))

    def update(self, terminal_winner):
        """
        Update this node from the perspective of player_just_moved.
        """
        self.visits += 1

        if terminal_winner is None:
            self.wins += 0.5
        elif terminal_winner == self.player_just_moved:
            self.wins += 1.0
        else:
            self.wins += 0.0


class MCTSAgent(BaseAgent):
    def __init__(self, name=None, iterations=2000, exploration_weight=1.414):
        name = name or f"MCTS-{iterations}" # name is MCTS-{iterations} by default, but can be overridden by args
        super().__init__(name)

        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.center_order = [3, 2, 4, 1, 5, 0, 6]

        # persistent root for tree reuse
        self.root = None

        self.reset_stats()

    def reset_stats(self) -> None:
        self.total_search_calls = 0
        self.total_simulations = 0
        self.total_rollout_moves = 0
        self.moves_chosen = 0

        self.immediate_win_hits = 0
        self.immediate_block_hits = 0

        self.simulations_per_move = []
        self.rollout_lengths = []
        self.root_children_counts = []
        self.chosen_move_visits = []
        self.chosen_move_win_rates = []

        # reuse stats
        self.tree_reuse_hits = 0
        self.tree_rebuilds = 0

        # clear saved tree when stats reset
        self.root = None

    def choose_action(self, game) -> int:
        legal_moves = game.get_legal_moves()

        if not legal_moves:
            raise ValueError("No legal moves available.")

        self.total_search_calls += 1
        root_player = game.current_player
        opponent = self.get_opponent(root_player)

        if len(legal_moves) == 1:
            self.moves_chosen += 1
            self.simulations_per_move.append(0)
            self.root_children_counts.append(1)
            self.chosen_move_visits.append(0)
            self.chosen_move_win_rates.append(0.0)

            # tree becomes unreliable if we skip search entirely
            self.root = None
            return legal_moves[0]

        # 1. Immediate win
        winning_move = self.find_immediate_win(game, root_player)
        if winning_move is not None:
            self.moves_chosen += 1
            self.immediate_win_hits += 1
            self.simulations_per_move.append(0)
            self.root_children_counts.append(0)
            self.chosen_move_visits.append(0)
            self.chosen_move_win_rates.append(1.0)

            # safest simple behavior: clear tree on tactical shortcut
            self.root = None
            return winning_move

        # 2. Immediate block
        block_move = self.find_immediate_win(game, opponent)
        if block_move is not None:
            self.moves_chosen += 1
            self.immediate_block_hits += 1
            self.simulations_per_move.append(0)
            self.root_children_counts.append(0)
            self.chosen_move_visits.append(0)
            self.chosen_move_win_rates.append(0.0)

            # safest simple behavior: clear tree on tactical shortcut
            self.root = None
            return block_move

        # reuse or rebuild root
        root = self.sync_root_to_game(game)

        sims_this_move = 0

        for _ in range(self.iterations):
            sims_this_move += 1
            node = root
            path = [root]
            applied_moves = []

            # -----------------
            # 1. Selection
            # -----------------
            while not game.is_terminal():
                self.initialize_untried_moves(node, game)

                if not node.is_fully_expanded():
                    break

                if not node.children:
                    break

                node = node.best_child(self.exploration_weight)
                game.make_move(node.move)
                applied_moves.append(node.move)
                path.append(node)

            # -----------------
            # 2. Expansion
            # -----------------
            if not game.is_terminal():
                self.initialize_untried_moves(node, game)

                if node.untried_moves:
                    move = node.untried_moves.pop(0)
                    player_making_move = game.current_player

                    game.make_move(move)
                    applied_moves.append(move)

                    child = MCTSNode(
                        parent=node,
                        move=move,
                        player_just_moved=player_making_move,
                        state_key=self.get_state_key(game),
                    )
                    node.children.append(child)
                    node = child
                    path.append(node)

            # -----------------
            # 3. Rollout
            # -----------------
            terminal_winner = self.rollout(game)

            # -----------------
            # 4. Backpropagation
            # -----------------
            for visited_node in path:
                visited_node.update(terminal_winner)

            # Undo selection/expansion path
            for _ in range(len(applied_moves)):
                game.undo_move()

        self.moves_chosen += 1
        self.total_simulations += sims_this_move
        self.simulations_per_move.append(sims_this_move)
        self.root_children_counts.append(len(root.children))

        if not root.children:
            fallback_moves = self.get_ordered_moves(game)
            chosen_move = fallback_moves[0]
            self.chosen_move_visits.append(0)
            self.chosen_move_win_rates.append(0.0)

            self.root = None
            return chosen_move

        # Final move choice:
        # choose child with best empirical value for ROOT player
        best_child = max(
            root.children,
            key=lambda child: self.child_value_for_root(child, root_player),
        )

        self.chosen_move_visits.append(best_child.visits)
        self.chosen_move_win_rates.append(
            self.child_value_for_root(best_child, root_player)
        )

        # advance tree root to chosen move for reuse next turn
        best_child.parent = None
        self.root = best_child

        return best_child.move

    def rollout(self, game):
        """
        Safer rollout:
        1. play immediate win if available
        2. block opponent immediate win if needed
        3. prefer safe moves that do not allow opponent immediate win
        4. among safe moves, prefer center-biased order
        """
        rollout_move_count = 0

        while not game.is_terminal():
            current_player = game.current_player
            move = self.choose_rollout_move(game, current_player)
            game.make_move(move)
            rollout_move_count += 1

        terminal_winner = game.winner

        for _ in range(rollout_move_count):
            game.undo_move()

        self.total_rollout_moves += rollout_move_count
        self.rollout_lengths.append(rollout_move_count)

        return terminal_winner

    def choose_rollout_move(self, game, current_player) -> int:
        opponent = self.get_opponent(current_player)
        legal_moves = game.get_legal_moves()

        # 1. Win now
        winning_move = self.find_immediate_win(game, current_player)
        if winning_move is not None:
            return winning_move

        # 2. Block opponent win now
        block_move = self.find_immediate_win(game, opponent)
        if block_move is not None:
            return block_move

        # 3. Safe moves: after we play, opponent should not have immediate win
        safe_moves = []

        for move in self.center_order:
            if move not in legal_moves:
                continue

            game.make_move(move)
            opp_winning_reply = self.find_immediate_win(game, opponent)
            game.undo_move()

            if opp_winning_reply is None:
                safe_moves.append(move)

        if safe_moves:
            # small randomness to keep rollouts varied
            top_k = safe_moves[:3] if len(safe_moves) >= 3 else safe_moves
            return random.choice(top_k)

        # 4. If all moves are dangerous, still prefer center ordering
        ordered_legal = [m for m in self.center_order if m in legal_moves]
        return random.choice(ordered_legal)

    def child_value_for_root(self, child: MCTSNode, root_player: int) -> float:
        """
        Convert child node stats into value from root player's perspective.
        Since child.wins is from child.player_just_moved perspective:
        - if child.player_just_moved == root_player, use wins/visits
        - otherwise use 1 - wins/visits
        Draws remain symmetric around 0.5
        """
        if child.visits == 0:
            return 0.0

        raw = child.wins / child.visits

        if child.player_just_moved == root_player:
            return raw
        return 1.0 - raw

    def initialize_untried_moves(self, node: MCTSNode, game) -> None:
        if node.untried_moves is None:
            node.untried_moves = self.get_ordered_moves(game)

    def get_ordered_moves(self, game) -> list[int]:
        """
        Tactical move ordering:
        1. immediate win
        2. immediate block
        3. safe center-biased moves
        4. remaining moves in center-biased order
        """
        current_player = game.current_player
        opponent = self.get_opponent(current_player)
        legal_moves = game.get_legal_moves()

        winning_moves = []
        blocking_moves = []
        safe_moves = []
        remaining_moves = []

        opp_immediate_win = self.find_immediate_win(game, opponent)

        for move in self.center_order:
            if move not in legal_moves:
                continue

            # immediate win for current player?
            game.make_move(move)
            current_wins = game.winner == current_player
            game.undo_move()

            if current_wins:
                winning_moves.append(move)
                continue

            if opp_immediate_win is not None and move == opp_immediate_win:
                blocking_moves.append(move)
                continue

            game.make_move(move)
            opp_wins_after = self.find_immediate_win(game, opponent)
            game.undo_move()

            if opp_wins_after is None:
                safe_moves.append(move)
            else:
                remaining_moves.append(move)

        seen = set()
        ordered = []

        for group in (winning_moves, blocking_moves, safe_moves, remaining_moves):
            for move in group:
                if move not in seen:
                    ordered.append(move)
                    seen.add(move)

        return ordered

    def find_immediate_win(self, game, player):
        legal_moves = game.get_legal_moves()
        original_player = game.current_player

        for move in self.center_order:
            if move not in legal_moves:
                continue

            game.current_player = player
            game.make_move(move)

            is_win = game.winner == player

            game.undo_move()
            game.current_player = original_player

            if is_win:
                return move

        return None

    def get_state_key(self, game):
        """
        Hashable representation of the current game state.
        Includes current_player because same board with different side to move
        is a different search state.
        """
        return (
            tuple(tuple(row) for row in game.board),
            game.current_player,
        )

    def sync_root_to_game(self, game):
        """
        Try to reuse the existing tree.
        Cases:
        1. No saved tree -> build fresh root
        2. Saved root exactly matches current state -> reuse it
        3. One of saved root's children matches current state -> promote child
        4. Otherwise -> rebuild fresh root
        """
        current_key = self.get_state_key(game)
        opponent = self.get_opponent(game.current_player)

        if self.root is None:
            self.tree_rebuilds += 1
            self.root = MCTSNode(
                parent=None,
                move=None,
                player_just_moved=opponent,
                state_key=current_key,
            )
            return self.root

        if self.root.state_key == current_key:
            self.tree_reuse_hits += 1
            return self.root

        for child in self.root.children:
            if child.state_key == current_key:
                child.parent = None
                self.root = child
                self.tree_reuse_hits += 1
                return self.root

        self.tree_rebuilds += 1
        self.root = MCTSNode(
            parent=None,
            move=None,
            player_just_moved=opponent,
            state_key=current_key,
        )
        return self.root

    def get_opponent(self, player):
        return 2 if player == 1 else 1

    def get_stats(self) -> dict:
        return {
            "Decisions": {
                "Moves chosen": self.moves_chosen,
                "Total search calls": self.total_search_calls,
            },
            "Search": {
                "Total simulations": self.total_simulations,
                "Avg simulations per move": (
                    sum(self.simulations_per_move) / len(self.simulations_per_move)
                    if self.simulations_per_move else 0.0
                ),
                "Avg root children": (
                    sum(self.root_children_counts) / len(self.root_children_counts)
                    if self.root_children_counts else 0.0
                ),
            },
            "Rollouts": {
                "Total rollout moves": self.total_rollout_moves,
                "Avg rollout length": (
                    sum(self.rollout_lengths) / len(self.rollout_lengths)
                    if self.rollout_lengths else 0.0
                ),
                "Max rollout length": (
                    max(self.rollout_lengths) if self.rollout_lengths else 0
                ),
            },
            "Tactics": {
                "Immediate win hits": self.immediate_win_hits,
                "Immediate block hits": self.immediate_block_hits,
            },
            "Chosen Move Quality": {
                "Avg chosen move visits": (
                    sum(self.chosen_move_visits) / len(self.chosen_move_visits)
                    if self.chosen_move_visits else 0.0
                ),
                "Avg chosen move win rate": (
                    sum(self.chosen_move_win_rates) / len(self.chosen_move_win_rates)
                    if self.chosen_move_win_rates else 0.0
                ),
            },
            "Tree Reuse": {
                "Reuse hits": self.tree_reuse_hits,
                "Tree rebuilds": self.tree_rebuilds,
            },
        }