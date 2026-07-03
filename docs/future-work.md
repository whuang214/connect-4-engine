# Future Work

## Hybrid MCTS + Neural Network Agent (AlphaZero-lite)

A prototype `HybridMCTSAgent` was built during the project (originally
`agents/hybrid_mcts_agent.py`, removed from the codebase — see
[why it was removed](#why-it-was-removed) below; the full source remains in git
history prior to its removal). It is the natural next step suggested by the
tournament results: combine MCTS's search strength with the RL network's fast
learned evaluation.

### Design

The agent replaced pure MCTS's random rollouts with **neural-network leaf
evaluation**:

- **Selection/expansion** — identical to the pure MCTS agent: UCT
  (`exploration_weight = √2`), center-ordered untried moves, tactical
  immediate-win/immediate-block overrides before search.
- **Evaluation** — instead of playing a rollout to a terminal state, the leaf
  position is encoded with the shared 4-channel `encode_board()` encoder and
  scored by the trained policy-value network's **value head** (one forward pass,
  ~1 ms) mapped from `[-1, 1]` to a `[0, 1]` win probability from the root
  player's perspective.
- **Backpropagation** — each node on the path accumulates the value from its
  own perspective (`v` for nodes where the root player just moved, `1 − v`
  otherwise).
- **Move selection** — most-visited root child (visit count is more robust than
  mean value at low simulation counts).

The intended payoff: each "simulation" costs one network call instead of a
20–30-move rollout, so an equal time budget buys far more tree growth — and the
evaluations are learned rather than noisy random playouts.

### Why it was removed

The prototype was never entered in the tournament, and a code review found two
correctness bugs that meant it had likely never been run end-to-end:

1. **Tuple-unpack crash** — leaf evaluation called
   `self.model(tensor).item()`, but both `PolicyValueNet` and
   `PolicyValueNetSmall` return a `(policy_logits, value)` **tuple**, so the
   first non-terminal leaf evaluation raises
   `AttributeError: 'tuple' object has no attribute 'item'`.
   The fix is `_, value = self.model(tensor)`.
2. **Terminal-value scale mismatch** — terminal leaves were scored on a
   `[-1, 1]` scale (win = 1.0, draw = 0.0, loss = −1.0) while non-terminal
   leaves and the backpropagation logic use a `[0, 1]` win-probability scale
   (the opponent inversion is `1.0 − v`). A terminal **loss** therefore
   back-propagated as `1.0 − (−1.0) = 2.0` for opponent nodes, corrupting the
   UCT averages exactly at the tactically decisive parts of the tree.
   Terminal outcomes must map to the same `[0, 1]` scale: win = 1.0,
   draw = 0.5, loss = 0.0.
   (The method also contained a dead `value_for_current` computation that was
   overwritten immediately — a leftover from an earlier perspective scheme.)

Rather than ship an unvalidated agent, the prototype was removed and its design
recorded here.

### Roadmap for a proper revival

In order of impact:

1. **Fix the two bugs above** and validate with unit tests (terminal-leaf
   backup values, tuple unpacking) before any strength testing.
2. **PUCT selection with policy priors** — the network already has a policy
   head that the prototype ignored. Replace vanilla UCT with AlphaZero's PUCT:
   `Q(s,a) + c · P(s,a) · √N(s) / (1 + N(s,a))`, using the masked softmax of the
   policy logits as `P`. This focuses search on moves the network already
   believes in.
3. **Close the training loop (true AlphaZero)** — train the network on MCTS
   **visit-count distributions** instead of the self-play behavior-cloning
   targets used in this project. The tournament analysis (see
   [results.md](results.md)) showed pure self-play without search plateaus hard;
   search-improved targets are the standard fix.
4. **Batched leaf evaluation** — collect several leaves per batch and evaluate
   them in one forward pass (virtual loss on pending nodes) to amortize GPU
   latency.
5. **Re-run the tournament** — hybrid-800 vs `mcts-700` and `minimax-7`, same
   protocol as [results.md](results.md) (alternating first player, ≥50 games).
