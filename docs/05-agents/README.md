# Agents

Six agents implement the [`BaseAgent`](../../connect4/agents/base.py)
contract (`choose_action(game) -> int` plus optional stats). Strength and
latency figures below are measured from the tournament and scaling
experiments in [`results/`](../../results) (per-move latency is the average
over full games; search agents think longer in the midgame).

| Agent | Spec | Algorithm | Typical strength (measured) | Per-move latency | Source |
|---|---|---|---|---|---|
| Human* | `human` | Keyboard input with validation | — | — | [`human.py`](../../connect4/agents/human.py) |
| [Random](random.md) | `random` | Uniform random legal move | Floor baseline: 0/100 vs minimax-7, 0/20 vs MCTS-700, 7/100 vs RL | < 0.1 ms | [`random.py`](../../connect4/agents/random.py) |
| [Rule-based](rule-based.md) | `rule` | Win → block → center | Beats RL-best 59/100; 1/20 vs MCTS-700; 0/100 vs minimax-7 | ~1 ms | [`rule_based.py`](../../connect4/agents/rule_based.py) |
| [Minimax](minimax.md) | `minimax-<depth>` | Alpha-beta + windowed heuristic | Depth 7: 100% vs random/rule/RL; 27% vs MCTS-700 (130 games) | ~0.2–0.5 s @ depth 7 (0.004 s @ 3, ~3 s @ 9) | [`minimax.py`](../../connect4/agents/minimax.py) |
| [MCTS](mcts.md) | `mcts-<iterations>` | UCT + tactical overrides + threat-aware rollouts | Tournament winner: 60% vs minimax-7 over 130 games; ≥ 95% vs everything else | ~6–9 s @ 700 iters (~2 s @ 200, ~16–22 s @ 2000) | [`mcts.py`](../../connect4/agents/mcts.py) |
| [RL policy](rl-policy.md) | `rl` | Self-play policy-value ResNet | 92% vs random, 37% vs rule, **0% vs every search agent** | ~4 ms | [`rl_policy.py`](../../connect4/agents/rl_policy.py) |

\* The human agent has no separate doc page — it is a thin input loop that
re-prompts on non-integer or illegal input and tracks interaction stats.

## The shared tactical layer

[`tactics.py`](../../connect4/tactics.py) centralizes the two things
every competent Connect 4 player does before anything else:

- `find_immediate_win(game, player)` — scan columns center-first and return
  one that wins on the spot for `player` (probing with make/undo on the live
  game, so no copying on the fast path).
- `CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)` / `ordered_legal_moves(game)` —
  the strongest-first column ordering.

The rule-based, MCTS, and RL agents all call `find_immediate_win` twice
(own win first, then opponent block) **before** their main policy, so none of
them ever misses a one-move win or hands over an open four. MCTS additionally
uses it inside every rollout step and for expansion ordering, and the trainer
applies the same win > block override when generating self-play data. Minimax
is the exception by design: it does not need the pre-check, because a depth ≥ 2
alpha-beta search proves immediate wins and forced blocks itself.

## Per-agent docs

- [random.md](random.md)
- [rule-based.md](rule-based.md)
- [minimax.md](minimax.md)
- [mcts.md](mcts.md)
- [rl-policy.md](rl-policy.md)
