# Minimax Agent

## Algorithm

Depth-limited minimax with alpha-beta pruning: exhaustively search every move
sequence `depth` plies deep, assume both sides play best-for-themselves, and
score the horizon positions with a hand-written positional heuristic.
Deterministic, complete within its horizon — it never misses a forced win or
loss that lies within `depth` moves.

## How it works

### Search

`choose_action` records which side it is playing (`game.current_player`),
then runs a standard alpha-beta root loop: for each ordered legal column,
`make_move` → recurse with `_minimax(depth-1, alpha, beta, minimizing)` →
`undo_move`, keeping the best-scoring column. The recursion alternates
maximizing (agent) and minimizing (opponent) levels with beta/alpha cutoffs.
All of this mutates the single game object in place via the engine's undo
stack — no cloning per node.

### Terminal scores are depth-adjusted

Terminal positions dominate the heuristic and encode urgency:

- Win for the agent: `WIN_SCORE + depth` = `1_000_000 + remaining_depth` —
  a win found earlier (more depth remaining) scores higher, so the agent
  **prefers faster wins**.
- Loss: `LOSS_SCORE - depth` = `-1_000_000 - remaining_depth` — **prefers
  slower losses**, maximizing the opponent's chance to slip.
- Draw: `0`.

### Move ordering

`_ordered_moves` sorts legal columns by distance from the center
(`abs(c - 3)`), yielding 3, 2, 4, 1, 5, 0, 6 — the same order as
`tactics.CENTER_ORDER`, derived independently. Center moves are usually
strongest in Connect 4, so exploring them first tightens alpha/beta early and
maximizes cutoffs.

### Heuristic evaluation (`_evaluate`)

At depth 0 the board is scored by scanning **every window of 4 cells** —
horizontal (24), vertical (21), and both diagonals (12 + 12) = 69 windows —
plus a center-column bonus. Exact score table from `_score_window`:

| Window contents | Score |
|---|---|
| 4 agent pieces | +100 (normally unreachable — terminal check fires first) |
| 3 agent + 1 empty | **+5** |
| 2 agent + 2 empty | **+2** |
| 3 opponent + 1 empty | **−4** |
| 2 opponent + 2 empty | **−1** |
| Mixed (both players present — dead window) | 0 |

plus **+3 per agent piece in the center column** (column 3). The asymmetry
(−4 for an opponent three vs +5 for its own) makes it value its own attack
slightly above defense at equal material; mixed windows score zero because
neither side can ever complete them.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `depth` | 5 | Plies of lookahead; cost grows roughly geometrically |
| `name` | `minimax-<depth>` | Auto-generated when omitted |

CLI spec: `minimax` (depth 5) or `minimax-<depth>`, e.g.
`python -m connect4 eval --agent1 minimax-7 --agent2 mcts-700 --games 20`.

## Tournament performance

Combined from [`results/tournament_results.json`](../../results/tournament_results.json)
and [`results/mcts_vs_minimax_part1_mcts700_vs_depths.json`](../../results/mcts_vs_minimax_part1_mcts700_vs_depths.json):

| Matchup | Games | Minimax wins | Win rate |
|---|---|---|---|
| Minimax-7 vs Random | 100 | 100 | 100% |
| Minimax-7 vs Rule-based | 100 | 100 | 100% |
| Minimax (any depth 3–9) vs RL-best | 100 each | 100 each | 100% |
| Minimax-3 vs MCTS-700 | 50 | 8 | 16% |
| Minimax-5 vs MCTS-700 | 50 | 11 | 22% |
| Minimax-7 vs MCTS-700 | 130 | 35 | 27% (17 draws) |
| Minimax-9 vs MCTS-700 | 50 | 19 | 38% |

Depth buys real strength against MCTS-700 (16% → 38% from depth 3 to 9) but
at steep cost — measured average decision time:

| Depth | Avg time/move |
|---|---|
| 3 | ~0.004 s |
| 5 | ~0.03–0.05 s |
| 7 | ~0.2–0.5 s |
| 9 | ~2.6–3.9 s |

Depth 7 is the sweet spot used throughout the report: crushing against
non-search agents, competitive with MCTS-700 at ~20× less time per move.
Against the *cheaper* MCTS-200, minimax already edges ahead from depth 5 up
([part 4 results](../../results/mcts_vs_minimax_part4_mcts200_vs_depths.json),
30 games per depth: minimax wins 15-13, 15-13, and 17-11 at depths 5/7/9,
and loses 13-16 at depth 3).

## Implementation notes

Source: [`agents/minimax.py`](../../connect4/agents/minimax.py).

- **In-place search.** The evaluator hands agents a `game.clone()`, and
  minimax searches directly on that clone with `make_move`/`undo_move` — the
  engine's `MoveHistory` stack is what makes deep recursion allocation-free.
  See [the two-engine design](../04-architecture.md#the-two-engine-design).
- **No explicit win/block pre-check.** Unlike rule-based/MCTS/RL, it does not
  call `tactics.find_immediate_win` — any depth ≥ 2 search proves immediate
  wins and forced blocks on its own, and the depth-adjusted terminal scores
  make it take the fastest one.
- The heuristic re-scans all 69 windows at every leaf; combined with pure-
  Python board access this is the depth-9 cost driver. Incremental evaluation
  or bitboards are the obvious optimizations, deliberately out of scope.
- `_evaluate` scores from the perspective fixed in `choose_action`
  (`self.player`), so the same function serves both maximizing and minimizing
  levels.
