# Rule-Based Agent

## Algorithm

A fixed four-rule priority list — the minimum viable Connect 4 player:
take an immediate win, block an immediate loss, otherwise play as close to
the center as possible. No lookahead beyond one ply.

## How it works

`choose_action` walks the rules in order and returns at the first hit:

1. **Immediate win** — `find_immediate_win(game, me)` from
   [`tactics.py`](../../connect4/tactics.py): probe each legal column
   (center-first) with make/undo and play any that wins on the spot.
2. **Immediate block** — `find_immediate_win(game, opponent)`: if the
   opponent has a winning column, occupy it.
3. **Center** — otherwise take the first entry of `ordered_legal_moves(game)`,
   which scans `CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)`. Column 3 while it is
   open (rule 3), else the nearest-to-center legal column (rule 4).

A per-rule counter (`rule1_immediate_win`, `rule2_immediate_block`, ...) is
tracked and reported through `get_stats()`, so evaluation output shows how
often each rule fired. The rule-3/4 split is bookkeeping only — both come
from the same center-first scan.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `name` | `"RuleBasedAgent"` | Display name only |

CLI spec: `rule`.

## Tournament performance

From [`results/tournament_results.json`](../../results/tournament_results.json):

| Opponent | Games | Rule-based wins | Draws | Win rate |
|---|---|---|---|---|
| RL-best | 100 | 59 | 4 | 59% |
| MCTS-700 | 20 | 1 | 0 | 5% |
| Minimax-7 | 100 | 0 | 0 | 0% |

Its headline role in the results: it **beats the trained RL agent** (59-37-4
over 100 games) despite being ~50 lines of if-statements, which anchors the
RL failure analysis in [rl-policy.md](rl-policy.md).

## Implementation notes

Source: [`agents/rule_based.py`](../../connect4/agents/rule_based.py).

- Fully deterministic given a position: same board, same move. In evaluation
  it only produces varied games because opponents vary (the head-to-head
  runner alternates who plays first).
- The center preference means its non-tactical play is a fixed column-filling
  pattern — strong against Random, trivially exploitable by anything that
  searches. Both minimax and MCTS punish it with near-perfect scores.
- The final fallback branch (`legal_moves[0]`) is unreachable while legal
  moves exist; it is kept as a guard and counted separately in stats.
- All tactical logic lives in shared [`tactics.py`](../../connect4/tactics.py)
  — this agent contains no game-mechanics code of its own.
