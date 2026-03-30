import math
import random
from agents.base_agent import BaseAgent


class MCTSNode:
	def __init__(self, parent=None, move=None):
		self.parent = parent
		self.move = move  # move that led to this node

		self.children = []
		self.untried_moves = None  # filled in when node is first visited

		self.visits = 0
		self.value = 0.0  # accumulated value from root player's perspective

	def is_fully_expanded(self):
		return self.untried_moves is not None and len(self.untried_moves) == 0

	def best_child(self, exploration_weight=1.414):
		"""
		Select child using UCB1:
		exploitation + exploration
		"""
		best_score = float("-inf")
		best_node = None

		log_parent_visits = math.log(self.visits)

		for child in self.children:
			if child.visits == 0:
				score = float("inf")
			else:
				exploitation = child.value / child.visits
				exploration = exploration_weight * math.sqrt(
					log_parent_visits / child.visits
				)
				score = exploitation + exploration

			if score > best_score:
				best_score = score
				best_node = child

		return best_node

	def backpropagate(self, result):
		"""
		result should be from the root player's perspective:
		win = 1.0, draw = 0.5, loss = 0.0
		"""
		node = self
		while node is not None:
			node.visits += 1
			node.value += result
			node = node.parent


class MCTSAgent(BaseAgent):
	def __init__(self, name="MCTSAgent", iterations=1000, exploration_weight=1.414):
		super().__init__(name)
		self.iterations = iterations
		self.exploration_weight = exploration_weight
		self.center_order = [3, 2, 4, 1, 5, 0, 6]

	def choose_action(self, game) -> int:
		legal_moves = game.get_legal_moves()

		if not legal_moves:
			raise ValueError("No legal moves available.")

		# If only one move exists, just play it
		if len(legal_moves) == 1:
			return legal_moves[0]

		current_player = game.current_player
		opponent = self.get_opponent(current_player)

		# 1. Immediate winning move
		winning_move = self.find_immediate_win(game, current_player)
		if winning_move is not None:
			return winning_move

		# 2. Immediate block if opponent can win next turn
		block_move = self.find_immediate_win(game, opponent)
		if block_move is not None:
			return block_move

		root_player = game.current_player
		root = MCTSNode()

		for _ in range(self.iterations):
			node = root
			path_moves = []

			# 1. Selection
			while not game.is_terminal():
				self.initialize_untried_moves(node, game)

				if not node.is_fully_expanded():
					break

				if not node.children:
					break

				node = node.best_child(self.exploration_weight)
				game.make_move(node.move)
				path_moves.append(node.move)

			# 2. Expansion
			if not game.is_terminal():
				self.initialize_untried_moves(node, game)

				if node.untried_moves:
					move = node.untried_moves.pop(0)  # center-prioritized
					game.make_move(move)
					path_moves.append(move)

					child_node = MCTSNode(parent=node, move=move)
					node.children.append(child_node)
					node = child_node

			# 3. Simulation / Rollout
			result = self.rollout(game, root_player)

			# 4. Backpropagation
			node.backpropagate(result)

			# Undo selection + expansion moves
			for _ in range(len(path_moves)):
				game.undo_move()

		# Choose the child with the most visits
		best_child = max(root.children, key=lambda child: child.visits)
		return best_child.move

	def rollout(self, game, root_player) -> float:
		"""
		Play semi-smart moves until terminal state.
		Return result from the root player's perspective:
			win  -> 1.0
			draw -> 0.5
			loss -> 0.0

		IMPORTANT:
		This rollout mutates the real game object temporarily,
		then undoes all rollout moves before returning.
		"""
		rollout_move_count = 0

		while not game.is_terminal():
			current_player = game.current_player
			opponent = self.get_opponent(current_player)
			legal_moves = game.get_legal_moves()

			# 1. If current player can win now, do it
			move = self.find_immediate_win(game, current_player)
			if move is not None:
				game.make_move(move)
				rollout_move_count += 1
				continue

			# 2. If opponent can win next turn, block it
			move = self.find_immediate_win(game, opponent)
			if move is not None and move in legal_moves:
				game.make_move(move)
				rollout_move_count += 1
				continue

			# 3. Prefer center columns if available
			center_moves = [m for m in self.center_order if m in legal_moves]
			move = random.choice(center_moves[:3]) if len(center_moves) >= 3 else random.choice(center_moves)

			game.make_move(move)
			rollout_move_count += 1

		# Read terminal result before undoing
		if game.winner is None:
			result = 0.5
		elif game.winner == root_player:
			result = 1.0
		else:
			result = 0.0

		# Undo rollout moves
		for _ in range(rollout_move_count):
			game.undo_move()

		return result

	def find_immediate_win(self, game, player):
		"""
		Return a move that lets 'player' win immediately, if one exists.
		Otherwise return None.

		This now uses make_move + undo_move instead of cloning.
		"""
		legal_moves = game.get_legal_moves()
		original_player = game.current_player

		for move in self.center_order:
			if move not in legal_moves:
				continue

			# Temporarily force the player whose win we want to test
			game.current_player = player
			game.make_move(move)

			is_win = game.winner == player

			game.undo_move()
			game.current_player = original_player

			if is_win:
				return move

		return None

	def initialize_untried_moves(self, node, game):
		"""
		Lazily initialize untried moves for a node based on the CURRENT
		game state reached during tree traversal.
		"""
		if node.untried_moves is None:
			legal_moves = game.get_legal_moves()
			node.untried_moves = [m for m in self.center_order if m in legal_moves]

	def get_opponent(self, player):
		return 2 if player == 1 else 1