# MCTS Agent

## Algorithm

Monte Carlo Tree Search with UCT selection — the tournament winner. Instead
of exhaustive lookahead, it grows an asymmetric game tree by running a fixed
number of simulations: repeatedly walk the tree by upper-confidence bounds,
expand one new node, estimate the resulting position by playing a fast
randomized game to the end, and propagate the outcome back up. This
implementation hardens the textbook algorithm with tactical
overrides, threat-aware move ordering and rollouts, and cross-move tree reuse.

## How it works

Each `choose_action` call:

### 0. Pre-search tactical overrides

Before any search (shared [`tactics.py`](../../connect4/tactics.py)):

- Only one legal move → play it.
- `find_immediate_win(game, me)` → play the winning column immediately.
- `find_immediate_win(game, opponent)` → block it.

Override hits are counted (`immediate_win_hits` / `immediate_block_hits`) and
discard the reuse tree, since no search ran.

### 1. Selection — UCT

From the root, while a node is fully expanded, descend to the child
maximizing

```
UCT(child) = wins/visits + c · sqrt( ln(parent.visits) / visits )
```

with exploration weight `c = 1.414 (≈ √2)`; unvisited children score `∞`.
Each descent step applies the child's move to the **live game object** with
`make_move`.

### 2. Expansion — ordered untried moves

A node's untried moves are initialized once by `get_ordered_moves`, which
buckets legal columns as **win > block > safe > rest** (center-first within
each bucket): columns that win immediately, then the column blocking the
opponent's immediate win, then "safe" columns whose reply position gives the
opponent no immediate win, then the remainder. Expansion pops the first
untried move, applies it, and adds the child (tagged with
`player_just_moved` and a state key).

### 3. Rollout — threat-aware, not uniform random

From the expanded position, play to a terminal state using
`choose_rollout_move` at every step:

1. Take an immediate win if one exists.
2. Block the opponent's immediate win.
3. Otherwise compute the *safe* moves — those that do **not** hand the
   opponent an immediate winning reply — and pick uniformly among the top 3
   in center order.
4. If no move is safe, pick randomly among center-ordered legal moves.

This removes most of the noise that makes uniform-random rollouts
misevaluate tactical positions. All rollout moves are made on the live game
and undone afterward.

### 4. Backpropagation

Every node on the selection/expansion path updates from the perspective of
its own `player_just_moved`: win = 1.0, **draw = 0.5**, loss = 0.0, so a
node's `wins/visits` is directly "how good was moving here for the player who
moved here". After backprop the applied moves are undone, restoring the root
position for the next simulation.

### 5. Move choice and tree reuse

After `iterations` simulations the agent picks the root child with the best
**win rate from the root player's perspective** (`child_value_for_root`
flips `1 - wins/visits` for children where the opponent just moved). The
chosen child is detached and kept as `self.root`.

On the next turn, `sync_root_to_game` re-anchors by state key
(`(board_tuple, current_player)`): if the stored root or one of its children
matches the current position (i.e. the opponent played a move the tree
already explored), that subtree — with all its accumulated visit statistics —
is reused; otherwise the tree is rebuilt from scratch. Reuse hits and
rebuilds are reported in `get_stats()`.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `iterations` | 2000 (constructor) / **500 via the CLI factory** | Simulations per decision |
| `exploration_weight` | 1.414 (≈ √2) | UCT exploration constant |
| `name` | `MCTS-<iterations>` | Auto-generated when omitted |

CLI spec: `mcts` (500 iterations) or `mcts-<iterations>`, e.g.
`python -m connect4 eval --agent1 mcts-700 --agent2 minimax-7 --games 20`.

## Tournament performance

Headline: **MCTS-700 beat minimax-7 in 60% of 130 games** (78 W / 35 L /
17 D), combining the tournament core matchup with experiment parts 1–3, and
won from both seats. Full data:
[`tournament_results.json`](../../results/tournament_results.json) and
[`mcts_vs_minimax_part1..4`](../../results).

Versus minimax depth (MCTS-700, combined samples):

| Minimax depth | Games | MCTS wins | Win rate |
|---|---|---|---|
| 3 | 50 | 42 | 84% |
| 5 | 50 | 35 | 70% |
| 7 | 130 | 78 | 60% |
| 9 | 50 | 28 | 56% |

Iteration scaling versus minimax-7 (combined samples):

| Iterations | Games | MCTS wins | Win rate | Avg time/move |
|---|---|---|---|---|
| 200 | 80 | 31 | 39% | ~2.0–2.7 s |
| 500 | 30 | 20 | 67% | ~4.4 s |
| 700 | 130 | 78 | 60% | ~5.6–9.2 s |
| 1000 | 44 | 25 | 57% | ~8.3–10.1 s |
| 1500 | 30 | 16 | 53% | ~12.4 s |
| 2000 | 10 | 8 | 80% | ~15.7–22.5 s |

The jump from 200 → 500 iterations is decisive; beyond that, returns flatten
into sample noise at these game counts (the 2000-iteration figure is only 10
games). Against everything else, MCTS-700 was near-perfect: 20/20 vs Random,
19/20 vs Rule-based, 20/20 vs the RL agent — and MCTS won all 44 games against
RL at 200/1000/2000 iterations too.

The trade-off is cost: at ~6–9 s per move, MCTS-700 spends roughly 20× longer
per decision than minimax-7 for its 60/40 edge.

## Implementation notes

Source: [`agents/mcts.py`](../../connect4/agents/mcts.py).

- **In-place search via the undo stack.** All selection, expansion, and
  rollout moves are played on the game object the evaluator passes in (a
  `clone()` of the real game) and unwound with `undo_move` — thousands of
  make/undo pairs per decision, zero per-node board copies. This is the
  workload the OO engine is designed for
  ([architecture](../04-architecture.md#the-two-engine-design)).
- Nodes store `untried_moves = None` until first visited, so ordering work
  (`get_ordered_moves` runs several win-probes per column) is only paid for
  nodes the search actually reaches.
- Per-move stat histories (`simulations_per_move`, `rollout_lengths`, ...)
  are bounded `deque`s (`maxlen=50_000`) so long tournament runs don't grow
  memory without limit.
- The state key used for tree reuse hashes the full board plus side to move —
  exact-match only; transpositions are not merged.
- If the root somehow has no children after search (all simulations hit
  terminal states before expansion), the agent falls back to the first
  ordered move.
