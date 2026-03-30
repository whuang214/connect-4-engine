import math
import random
from agents.base_agent import BaseAgent


class MCTSNode:
	def __init__(self, game, parent=None, move=None):
		self.game = game
		self.parent = parent
		self.move = move  # move that led to this node

		self.children = []
		self.untried_moves = game.get_legal_moves()

		self.visits = 0
		self.value = 0.0  # accumulated value from root player's perspective

	def is_fully_expanded(self):
		return len(self.untried_moves) == 0

	def best_child(self, exploration_weight=1.414):
		"""
		Select child using UCB1:
		exploitation + exploration
		"""
		best_score = float("-inf")
		best_node = None

		for child in self.children:
			if child.visits == 0:
				score = float("inf")
			else:
				exploitation = child.value / child.visits
				exploration = exploration_weight * math.sqrt(
					math.log(self.visits) / child.visits
				)
				score = exploitation + exploration

			if score > best_score:
				best_score = score
				best_node = child

		return best_node

	def expand(self):
		"""
		Take one untried move, create a new child node from it.
		"""
		move = random.choice(self.untried_moves)
		self.untried_moves.remove(move)

		next_game = self.game.clone()
		next_game.make_move(move)

		child_node = MCTSNode(game=next_game, parent=self, move=move)
		self.children.append(child_node)

		return child_node

	def backpropagate(self, result):
		"""
		result should be from the root player's perspective:
		win = 1.0, draw = 0.5, loss = 0.0
		"""
		self.visits += 1
		self.value += result

		if self.parent is not None:
			self.parent.backpropagate(result)


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
		root = MCTSNode(game=game.clone())

		for _ in range(self.iterations):
			node = root

			# 1. Selection
			while not node.game.is_terminal() and node.is_fully_expanded():
				node = node.best_child(self.exploration_weight)

			# 2. Expansion
			if not node.game.is_terminal() and not node.is_fully_expanded():
				node = node.expand()

			# 3. Simulation / Rollout
			result = self.rollout(node.game.clone(), root_player)

			# 4. Backpropagation
			node.backpropagate(result)

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
		"""
		while not game.is_terminal():
			current_player = game.current_player
			opponent = self.get_opponent(current_player)
			legal_moves = game.get_legal_moves()

			# 1. If current player can win now, do it
			move = self.find_immediate_win(game, current_player)
			if move is not None:
				game.make_move(move)
				continue

			# 2. If opponent can win next turn, block it
			move = self.find_immediate_win(game, opponent)
			if move is not None and move in legal_moves:
				game.make_move(move)
				continue

			# 3. Prefer center columns if available
			center_moves = [m for m in self.center_order if m in legal_moves]
			move = random.choice(center_moves[:3]) if len(center_moves) >= 3 else random.choice(center_moves)

			game.make_move(move)

		if game.winner is None:
			return 0.5
		elif game.winner == root_player:
			return 1.0
		else:
			return 0.0

	def find_immediate_win(self, game, player):
		"""
		Return a move that lets 'player' win immediately, if one exists.
		Otherwise return None.
		"""
		for move in self.center_order:
			if move not in game.get_legal_moves():
				continue

			test_game = game.clone()

			# Make sure the right player is making the move in the clone
			test_game.current_player = player
			test_game.make_move(move)

			if test_game.winner == player:
				return move

		return None

	def get_opponent(self, player):
		return 2 if player == 1 else 1