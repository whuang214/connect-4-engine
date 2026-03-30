from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pygame

from engine import Connect4


@dataclass
class MoveInfo:
	move: Optional[int]
	time_ms: float
	extra: str = ""


class GameUI:
	# Window + layout
	CELL_SIZE = 100
	BOARD_PADDING = 16
	TOP_BAR = 110
	SIDE_PANEL = 330
	FOOTER = 70
	FPS = 60

	# Colors
	BG = (18, 22, 30)
	SURFACE = (28, 33, 44)
	SURFACE_2 = (36, 42, 56)
	BOARD_BLUE = (47, 99, 200)
	BOARD_BLUE_DARK = (34, 76, 160)
	HOLE = (14, 18, 25)

	RED = (230, 82, 82)
	RED_DARK = (176, 48, 48)

	YELLOW = (246, 213, 73)
	YELLOW_DARK = (198, 165, 35)

	WHITE = (245, 247, 250)
	GRAY = (180, 187, 198)
	DIM = (125, 135, 150)
	GREEN = (76, 201, 122)
	CYAN = (90, 184, 255)
	SHADOW = (0, 0, 0)

	MODE_LABELS = {
		"hvh": "Human vs Human",
		"hva": "Human vs AI",
		"ava": "AI vs AI",
	}

	def __init__(self, player1_agent=None, player2_agent=None):
		pygame.init()
		pygame.font.init()

		self.rows = Connect4.ROWS
		self.cols = Connect4.COLS

		self.board_width = self.cols * self.CELL_SIZE
		self.board_height = self.rows * self.CELL_SIZE

		self.width = self.board_width + self.SIDE_PANEL + self.BOARD_PADDING * 2
		self.height = (
			self.TOP_BAR + self.board_height + self.FOOTER + self.BOARD_PADDING * 2
		)

		self.screen = pygame.display.set_mode((self.width, self.height))
		pygame.display.set_caption("Connect 4 Visualizer")

		self.clock = pygame.time.Clock()

		self.title_font = pygame.font.SysFont("arial", 34, bold=True)
		self.heading_font = pygame.font.SysFont("arial", 24, bold=True)
		self.text_font = pygame.font.SysFont("arial", 20)
		self.small_font = pygame.font.SysFont("arial", 17)
		self.big_status_font = pygame.font.SysFont("arial", 26, bold=True)

		self.game = Connect4()
		self.hover_col = 0
		self.drop_animation: Optional[Dict[str, Any]] = None

		self.agents = {
			Connect4.PLAYER1: player1_agent,
			Connect4.PLAYER2: player2_agent,
		}

		self.last_info: Dict[int, Optional[MoveInfo]] = {
			Connect4.PLAYER1: None,
			Connect4.PLAYER2: None,
		}

		self.winning_cells: list[tuple[int, int]] = []
		self.ai_think_delay_ms = 250
		self.last_ai_time = 0

		self.ai_thinking = False
		self.thinking_start_time = 0

	# -------------------------
	# Helpers
	# -------------------------
	def _player_name(self, player: int) -> str:
		agent = self.agents[player]
		if agent is None:
			return "Human"
		return getattr(agent, "name", agent.__class__.__name__)

	def _mode_label(self) -> str:
		p1_human = self.agents[Connect4.PLAYER1] is None
		p2_human = self.agents[Connect4.PLAYER2] is None
		if p1_human and p2_human:
			return self.MODE_LABELS["hvh"]
		if p1_human != p2_human:
			return self.MODE_LABELS["hva"]
		return self.MODE_LABELS["ava"]

	def _color(self, player: int):
		return self.RED if player == Connect4.PLAYER1 else self.YELLOW

	def _dark_color(self, player: int):
		return self.RED_DARK if player == Connect4.PLAYER1 else self.YELLOW_DARK

	def _board_left(self) -> int:
		return self.BOARD_PADDING

	def _board_top(self) -> int:
		return self.TOP_BAR + self.BOARD_PADDING

	def _board_x(self, col: int) -> int:
		return self._board_left() + col * self.CELL_SIZE + self.CELL_SIZE // 2

	def _board_y(self, row: int) -> int:
		return self._board_top() + row * self.CELL_SIZE + self.CELL_SIZE // 2

	def _board_rect(self) -> pygame.Rect:
		return pygame.Rect(
			self._board_left(),
			self._board_top(),
			self.board_width,
			self.board_height,
		)

	def _side_panel_rect(self) -> pygame.Rect:
		return pygame.Rect(
			self._board_left() + self.board_width + self.BOARD_PADDING,
			self.BOARD_PADDING,
			self.SIDE_PANEL,
			self.height - self.BOARD_PADDING * 2,
		)

	def _extract_move(self, result):
		if isinstance(result, int):
			return result
		if isinstance(result, tuple):
			return int(result[0])
		if hasattr(result, "move"):
			return int(result.move)
		raise ValueError("Invalid agent output. Return int, tuple, or object with .move")

	def _find_drop_row(self, col: int) -> int:
		for row in range(self.rows - 1, -1, -1):
			if self.game.board[row][col] == Connect4.EMPTY:
				return row
		raise ValueError(f"Column {col} is full")

	def _check_win_line(self, row: int, col: int) -> list[tuple[int, int]]:
		player = self.game.board[row][col]
		if player == Connect4.EMPTY:
			return []

		directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

		for dr, dc in directions:
			cells = [(row, col)]

			r, c = row + dr, col + dc
			while 0 <= r < self.rows and 0 <= c < self.cols and self.game.board[r][c] == player:
				cells.append((r, c))
				r += dr
				c += dc

			r, c = row - dr, col - dc
			while 0 <= r < self.rows and 0 <= c < self.cols and self.game.board[r][c] == player:
				cells.insert(0, (r, c))
				r -= dr
				c -= dc

			if len(cells) >= 4:
				return cells

		return []

	def _animate(self, col: int, row: int, player: int):
		self.drop_animation = {
			"col": col,
			"row": row,
			"y": self._board_top() - self.CELL_SIZE // 2,
			"target": self._board_y(row),
			"player": player,
			"velocity": 0.0,
			"landed": False,
			"land_time": 0,
		}

	def reset_game(self):
		self.game.reset()
		self.drop_animation = None
		self.last_info = {
			Connect4.PLAYER1: None,
			Connect4.PLAYER2: None,
		}
		self.winning_cells = []
		self.ai_thinking = False
		self.thinking_start_time = 0

	# -------------------------
	# Logic
	# -------------------------
	def _update_animation(self):
		if not self.drop_animation:
			return

		anim = self.drop_animation

		# falling phase
		if not anim["landed"]:
			anim["velocity"] += 2.4
			anim["y"] += anim["velocity"]

			if anim["y"] >= anim["target"]:
				anim["y"] = anim["target"]
				anim["landed"] = True
				anim["land_time"] = pygame.time.get_ticks()
			return

		# hold the piece in place for a short moment after landing
		if pygame.time.get_ticks() - anim["land_time"] < 70:
			return

		col = anim["col"]
		row = anim["row"]
		self.drop_animation = None

		result = self.game.make_move(col)

		if result.winner is not None:
			self.winning_cells = self._check_win_line(row, col)
		elif result.draw:
			self.winning_cells = []

	def _handle_human(self, col: int):
		if self.game.done or self.drop_animation:
			return
		if self.agents[self.game.current_player] is not None:
			return
		if not self.game.is_legal_move(col):
			return

		row = self._find_drop_row(col)
		player = self.game.current_player
		self.last_info[player] = MoveInfo(move=col, time_ms=0.0, extra="Human move")
		self._animate(col, row, player)

	def _handle_ai(self):
		if self.game.done or self.drop_animation:
			self.ai_thinking = False
			return

		agent = self.agents[self.game.current_player]
		if agent is None:
			self.ai_thinking = False
			return

		now = pygame.time.get_ticks()

		if not self.ai_thinking:
			self.ai_thinking = True
			self.thinking_start_time = now
			return

		if now - self.thinking_start_time < self.ai_think_delay_ms:
			return

		start = time.perf_counter()
		result = agent.choose_action(self.game.clone())
		move = self._extract_move(result)
		end = time.perf_counter()

		if not self.game.is_legal_move(move):
			self.ai_thinking = False
			raise ValueError(f"{self._player_name(self.game.current_player)} returned illegal move {move}")

		player = self.game.current_player
		row = self._find_drop_row(move)
		self.last_info[player] = MoveInfo(
			move=move,
			time_ms=(end - start) * 1000.0,
			extra=self._player_name(player),
		)

		self._animate(move, row, player)
		self.last_ai_time = now
		self.ai_thinking = False

	def _undo_action(self):
		if self.drop_animation or not self.game.move_history:
			return

		p1_human = self.agents[Connect4.PLAYER1] is None
		p2_human = self.agents[Connect4.PLAYER2] is None
		human_vs_ai = p1_human != p2_human

		if human_vs_ai:
			self.game.undo_move()
			if self.game.move_history:
				self.game.undo_move()
		else:
			self.game.undo_move()

		self.winning_cells = []
		self.drop_animation = None
		self.last_info = {
			Connect4.PLAYER1: None,
			Connect4.PLAYER2: None,
		}
		self.ai_thinking = False
		self.thinking_start_time = 0

	# -------------------------
	# Drawing
	# -------------------------
	def _draw_text(self, text, font, color, pos):
		surf = font.render(text, True, color)
		self.screen.blit(surf, pos)

	def _draw_shadowed_circle(self, color, center, radius):
		shadow_pos = (center[0] + 3, center[1] + 5)
		pygame.draw.circle(self.screen, (0, 0, 0, 80), shadow_pos, radius)
		pygame.draw.circle(self.screen, color, center, radius)
		pygame.draw.circle(self.screen, (255, 255, 255), center, radius, width=2)

	def _draw_top_bar(self):
		rect = pygame.Rect(
			self.BOARD_PADDING,
			self.BOARD_PADDING,
			self.board_width,
			self.TOP_BAR - 10,
		)
		pygame.draw.rect(self.screen, self.SURFACE, rect, border_radius=20)

		self._draw_text("Connect 4", self.title_font, self.WHITE, (rect.x + 18, rect.y + 12))
		self._draw_text(self._mode_label(), self.text_font, self.GRAY, (rect.x + 20, rect.y + 54))

		if self.game.winner is not None:
			status = f"Player {self.game.winner} wins!"
			status_color = self._color(self.game.winner)
		elif self.game.done:
			status = "Draw game"
			status_color = self.WHITE
		else:
			status = f"Player {self.game.current_player}'s turn"
			status_color = self._color(self.game.current_player)

		status_surf = self.big_status_font.render(status, True, status_color)
		status_x = rect.right - status_surf.get_width() - 20
		self.screen.blit(status_surf, (status_x, rect.y + 25))

	def _draw_hover_preview(self):
		if self.game.done or self.drop_animation:
			return
		if self.agents[self.game.current_player] is not None:
			return
		if not self.game.is_legal_move(self.hover_col):
			return

		center = (self._board_x(self.hover_col), self._board_top() - 28)
		preview = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
		pygame.draw.circle(
			preview,
			(*self._color(self.game.current_player), 120),
			(self.CELL_SIZE // 2, self.CELL_SIZE // 2),
			self.CELL_SIZE // 2 - 14,
		)
		self.screen.blit(preview, (center[0] - self.CELL_SIZE // 2, center[1] - self.CELL_SIZE // 2))

	def _draw_board(self):
		board_rect = self._board_rect()

		shadow = board_rect.move(6, 8)
		pygame.draw.rect(self.screen, (10, 12, 18), shadow, border_radius=28)

		pygame.draw.rect(self.screen, self.BOARD_BLUE_DARK, board_rect, border_radius=28)
		inner = board_rect.inflate(-6, -6)
		pygame.draw.rect(self.screen, self.BOARD_BLUE, inner, border_radius=24)

		for r in range(self.rows):
			for c in range(self.cols):
				center = (self._board_x(c), self._board_y(r))
				pygame.draw.circle(self.screen, self.HOLE, center, self.CELL_SIZE // 2 - 12)

				piece = self.game.board[r][c]
				if piece != Connect4.EMPTY:
					self._draw_shadowed_circle(self._color(piece), center, self.CELL_SIZE // 2 - 16)

		for r, c in self.winning_cells:
			pygame.draw.circle(
				self.screen,
				self.CYAN,
				(self._board_x(c), self._board_y(r)),
				self.CELL_SIZE // 2 - 8,
				width=5,
			)

		if self.game.last_move is not None:
			row, col = self.game.last_move
			pygame.draw.circle(
				self.screen,
				self.WHITE,
				(self._board_x(col), self._board_y(row)),
				self.CELL_SIZE // 2 - 6,
				width=3,
			)

		if self.drop_animation:
			center = (self._board_x(self.drop_animation["col"]), int(self.drop_animation["y"]))
			self._draw_shadowed_circle(
				self._color(self.drop_animation["player"]),
				center,
				self.CELL_SIZE // 2 - 16,
			)

	def _draw_footer(self):
		rect = pygame.Rect(
			self.BOARD_PADDING,
			self._board_top() + self.board_height + 10,
			self.board_width,
			self.FOOTER - 10,
		)
		pygame.draw.rect(self.screen, self.SURFACE, rect, border_radius=18)

		tips = "Click a column to play • R = restart • U = undo • ESC = quit"
		self._draw_text(tips, self.small_font, self.GRAY, (rect.x + 18, rect.y + 16))

	def _draw_side_panel(self):
		rect = self._side_panel_rect()
		pygame.draw.rect(self.screen, self.SURFACE, rect, border_radius=22)

		x = rect.x + 18
		y = rect.y + 18

		self._draw_text("Match Info", self.heading_font, self.WHITE, (x, y))
		y += 40

		self._draw_text("Player 1", self.text_font, self.RED, (x, y))
		y += 24
		self._draw_text(self._player_name(Connect4.PLAYER1), self.small_font, self.GRAY, (x, y))
		y += 38

		self._draw_text("Player 2", self.text_font, self.YELLOW, (x, y))
		y += 24
		self._draw_text(self._player_name(Connect4.PLAYER2), self.small_font, self.GRAY, (x, y))
		y += 46

		self._draw_text("Current State", self.heading_font, self.WHITE, (x, y))
		y += 38

		legal = self.game.get_legal_moves()
		state_lines = [
			f"Current player: {self.game.current_player}" if not self.game.done else "Game finished",
			f"Legal moves: {legal if legal else 'none'}",
			f"Moves played: {len(self.game.move_history)}",
		]
		for line in state_lines:
			self._draw_text(line, self.small_font, self.GRAY, (x, y))
			y += 24

		y += 16
		self._draw_text("Last Move", self.heading_font, self.WHITE, (x, y))
		y += 38

		for player in [Connect4.PLAYER1, Connect4.PLAYER2]:
			info = self.last_info[player]
			self._draw_text(f"Player {player}", self.text_font, self._color(player), (x, y))
			y += 24

			if info is None:
				self._draw_text("No move yet", self.small_font, self.DIM, (x, y))
				y += 34
				continue

			lines = [
				f"Column: {info.move}",
				f"Time: {info.time_ms:.1f} ms",
			]
			if info.extra:
				lines.append(info.extra)

			for line in lines:
				self._draw_text(line, self.small_font, self.GRAY, (x, y))
				y += 22
			y += 12

		y += 8
		self._draw_text("Controls", self.heading_font, self.WHITE, (x, y))
		y += 36
		controls = [
			"Mouse click  → play column",
			"R            → reset board",
			"U            → undo last move",
			"ESC          → quit",
		]
		for line in controls:
			self._draw_text(line, self.small_font, self.GRAY, (x, y))
			y += 22

	def _draw_thinking_indicator(self):
		if not self.ai_thinking or self.game.done:
			return

		elapsed = pygame.time.get_ticks() // 400
		dots = "." * (elapsed % 4)

		text = f"{self._player_name(self.game.current_player)} thinking{dots}"

		width = 250
		height = 44
		x = self._board_left() + self.board_width - width - 20
		y = self.BOARD_PADDING + 18

		rect = pygame.Rect(x, y, width, height)
		pygame.draw.rect(self.screen, self.SURFACE_2, rect, border_radius=12)
		pygame.draw.rect(self.screen, self.CYAN, rect, width=2, border_radius=12)

		label = self.small_font.render(text, True, self.WHITE)
		self.screen.blit(
			label,
			(x + (width - label.get_width()) // 2, y + (height - label.get_height()) // 2),
		)

	def draw(self):
		self.screen.fill(self.BG)
		self._draw_top_bar()
		self._draw_hover_preview()
		self._draw_board()
		self._draw_side_panel()
		self._draw_footer()
		self._draw_thinking_indicator()
		pygame.display.flip()

	# -------------------------
	# Main loop
	# -------------------------
	def run(self):
		running = True

		while running:
			self.clock.tick(self.FPS)

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					running = False

				elif event.type == pygame.MOUSEMOTION:
					mx, my = pygame.mouse.get_pos()
					board_rect = self._board_rect()
					if board_rect.left <= mx <= board_rect.right:
						raw_col = (mx - board_rect.left) // self.CELL_SIZE
						self.hover_col = max(0, min(self.cols - 1, raw_col))

				elif event.type == pygame.MOUSEBUTTONDOWN:
					mx, my = pygame.mouse.get_pos()
					if self._board_rect().collidepoint(mx, my):
						raw_col = (mx - self._board_rect().left) // self.CELL_SIZE
						self._handle_human(raw_col)

				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_ESCAPE:
						running = False
					elif event.key == pygame.K_r:
						self.reset_game()
					elif event.key == pygame.K_u:
						self._undo_action()

			self._update_animation()
			self._handle_ai()
			self.draw()

		pygame.quit()
		sys.exit()