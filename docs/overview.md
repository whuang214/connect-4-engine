# Project Overview

## What this is

A head-to-head comparison of three game-AI paradigms on Connect 4, each
implemented from scratch:

1. **Classical search** — depth-limited minimax with alpha-beta pruning and a
   hand-written positional heuristic.
2. **Stochastic search** — Monte Carlo Tree Search (UCT) with tactical
   overrides and threat-aware rollouts.
3. **Deep reinforcement learning** — a policy-value residual CNN trained by
   pure self-play on a custom vectorized engine (512 parallel games).

All three (plus random and rule-based baselines) share one
[game engine](../src/connect4/engine.py) and one
[agent interface](../src/connect4/agents/base.py), and were compared in a
round-robin tournament of 26 matchups / 1,528 games (~8.2 hours) plus ~440
additional games of MCTS-vs-minimax scaling experiments — raw JSON in
[results/](../results/).

## Why Connect 4

- **Solved game** — Connect 4 is a first-player win with perfect play, so
  there is a known ceiling and no ambiguity about what "strong" means.
- **Low branching factor** — at most 7 legal moves per turn keeps tree search
  tractable at meaningful depths without move-generation tricks.
- **Tactically dense** — immediate wins, forced blocks, and double threats
  appear constantly, which cleanly exposes the difference between agents that
  can calculate and agents that can only pattern-match. This property ended up
  central to the RL failure analysis.

## The three agents

**Minimax** ([agents/minimax.py](../src/connect4/agents/minimax.py)) —
depth-limited alpha-beta search. Leaf positions are scored by scanning every
4-cell window in all directions (open threes and twos, mixed windows are
dead) plus a center-column bonus. Moves are explored center-first to maximize
pruning. Depth 7 beats the random and rule-based baselines 100-0 and moves in
~0.4 s; each depth step costs roughly 9x more time.

**MCTS** ([agents/mcts.py](../src/connect4/agents/mcts.py)) — UCT tree search
with three practical hardenings: tactical overrides (immediate wins/blocks are
played without searching), threat-aware rollouts (rollout moves win, block,
then avoid handing the opponent a win, instead of playing uniformly at
random), and tree reuse between moves (the chosen child becomes the next
root). It searches by mutating the live game object with `make_move`/`undo_move`
rather than copying boards.

**RL policy** ([agents/rl_policy.py](../src/connect4/agents/rl_policy.py)) —
a ~2.3M-parameter residual CNN
([models/policy_value_network.py](../src/connect4/models/policy_value_network.py))
with a 7-way policy head and a tanh value head, trained for 500k pure
self-play episodes on an RTX 2080 Ti using the vectorized batch engine
([training/vec_engine.py](../src/connect4/training/vec_engine.py)). At play
time it masks illegal columns and applies the same win/block tactical override
used during training. It moves in ~0.004 s.

## Headline findings

The hierarchy was **MCTS > Minimax > RL**, with an unambiguous negative
result for pure self-play RL:

| Finding | Numbers |
|---|---|
| MCTS-700 beats Minimax-7 | **60% of 130 games** (78W-35L-17D), winning from both seats |
| MCTS-700 vs Minimax depth 3 / 5 / 7 / 9 | 84% / 70% / 60% / 56% MCTS wins — depth 9 costs ~3.9 s/move and still loses the series |
| MCTS iterations vs Minimax-7 | 36% @ 200 iters → 67% @ 500 → ~60% @ 700–1500 → 80% @ 2000 (10 games) |
| Minimax-7 vs Random / Rule-Based | 100% / 100% |
| RL vs Random | 92% (100 games) |
| RL vs Rule-Based | 37% (100 games) |
| RL vs every search agent | **0%** — at every checkpoint from 100k to 500k episodes (flat learning curve) |
| Decision time | RL 0.004 s · Minimax-7 0.44 s · MCTS-700 ~6 s per move |

The RL result is the most instructive: behavior cloning from self-play
without search-improved targets plateaued hard — the network learned openings
and general shape but never learned to calculate forced sequences. The full
analysis is in [results.md](results.md); planned improvements are laid out
in [future-work.md](future-work.md).

## Tech stack

- **Python 3.10+**, src-layout package (`pip install -e .`), `connect4` CLI
- **PyTorch** — network and training loop (CUDA if available)
- **NumPy** — the vectorized `(n, 6, 7)` int8 batch engine and the circular
  replay buffer
- **pygame** (optional `[ui]` extra) — graphical board; `pygame-ce` works as
  a drop-in where pygame lacks wheels
- **pytest** (optional `[dev]` extra) — unit tests for engine, agents,
  training, and CLI

## Course context

Final project for **CS5100 — Foundations of Artificial Intelligence**
(Northeastern University), by Will Huang, Joseph Winterlich, and Soham
Santra. MIT licensed.

## Read next

- Full result tables and analysis: [results.md](results.md)
- The original report: [report.pdf](report.pdf)
- How to run everything yourself: [commands.md](commands.md)
