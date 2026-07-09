# Random Agent

## Algorithm

Uniform random play: pick any legal column with equal probability. It exists
purely as the floor baseline — any agent that cannot dominate it has learned
nothing.

## How it works

`choose_action` asks the engine for `get_legal_moves()` (the non-full
columns) and returns `random.choice(...)`. It raises `ValueError` if called
with no legal moves. The only stat tracked is `moves_chosen`.

It is the one agent with **no tactical layer**: it will happily ignore its own
winning move and leave an opponent's open three unblocked. That property is
useful — beating Random measures whether an agent has any signal at all,
uncontaminated by shared win/block logic.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `name` | `"RandomAgent"` | Display name only |

CLI spec: `random` (e.g. `python -m connect4 eval --agent1 random --agent2 rl`).

## Tournament performance

From [`results/tournament_results.json`](../../results/tournament_results.json):

| Opponent | Games | Random wins | Draws |
|---|---|---|---|
| RL-best | 100 | 7 | 1 |
| MCTS-700 | 20 | 0 | 0 |
| Minimax-7 | 100 | 0 | 0 |

The 7 wins against the RL agent are themselves a data point: a policy that
drops 7% of games to noise has real tactical blind spots (see
[rl-policy.md](rl-policy.md)).

## Implementation notes

Source: [`agents/random.py`](../../connect4/agents/random.py) (~25 lines).

Fastest agent in the suite (a few microseconds per move — one legality scan
plus one RNG call), which makes it the standard smoke-test opponent for
new agents and CLI wiring.
