# Future Work

Three planned directions, in priority order.

## 1. Maximize the minimax agent

The MCTS agent got the full enhancement treatment (tactical overrides,
threat-aware rollouts, tree reuse); minimax should get the same. In rough
order of impact:

- **Bitboards** — represent each side's pieces as one 64-bit integer and
  detect four-in-a-row with a handful of shift-AND operations. This is the
  single biggest speedup available (typically 1–2 orders of magnitude in node
  throughput over the current list-of-lists board), which converts directly
  into deeper search at the same time budget.
- **Iterative deepening with a time budget** — search depth 1, 2, 3…
  until a per-move clock expires instead of a fixed depth. This makes
  strength specs time-based (`minimax-500ms`) and makes compute-fair
  comparisons against MCTS trivial.
- **Transposition table (Zobrist hashing)** — Connect 4 positions transpose
  heavily (different move orders reach the same board). Caching evaluated
  positions prunes repeated work and pairs naturally with iterative
  deepening (previous-iteration results seed move ordering).
- **Stronger move ordering** — principal variation from the previous
  deepening pass first, then killer moves and a history heuristic. Alpha-beta
  cutoff quality is the whole game; center-first ordering is a good start but
  leaves a lot on the table.
- **Evaluation upgrades** — odd/even threat parity (Connect 4 theory: the
  first player wants threats on odd rows, the second on even rows) and
  differentiating open vs. blocked three-in-a-rows.
- **Opening book** — Connect 4 is solved (first player wins by opening in the
  center); even a shallow book saves the deepest searches where they matter
  least.

Then re-run the scaling experiments (`connect4 experiment`) at **equal
wall-clock per move** rather than fixed depth. The current results show the
gap closing to 56–44 at depth 9 as compute approaches parity — a tuned
minimax could plausibly flip the headline MCTS result.

## 2. Train the RL agent against stronger opponents

The v3 agent trained purely against itself, and the results analysis
([results.md](results.md)) attributes its failure to exactly that: self-play
never generated the precise tactical sequences that search agents produce.
Planned changes:

- **Wire the frozen-checkpoint opponent pool into the vectorized self-play
  loop.** The trainer already maintains a pool of past-self snapshots; the
  batched loop needs opponent-aware stepping (a subset of environments where
  one seat is played by a frozen policy) so games stop being 100%
  live-network mirror matches.
- **Curriculum against the classical agents** — rule-based first, then
  shallow minimax (depths 1–3). This directly injects the forced sequences
  and fork threats the self-play distribution lacks. Deeper minimax is too
  slow to sit inside a 512-env loop, but shallow depths are batchable.
- **Outcome-weighted policy targets** — the current policy loss imitates
  whatever move was played regardless of whether the game was won; only the
  value head sees the outcome. Advantage-weighted cross-entropy (or plain
  REINFORCE weighting) makes the policy head learn from results rather than
  imitation.
- **Keep the evaluation gauntlet fixed** (Random / Rule / MCTS-200) so any
  new run is directly comparable to the v3 learning curve.

## 3. Hybrid MCTS + neural network agent

Combine the two strongest ideas in the project: MCTS's search with the
network's learned evaluation — a simplified AlphaZero.

- **Network leaf evaluation instead of rollouts** — expand the tree exactly
  as the pure MCTS agent does (UCT selection, tactical win/block overrides,
  center-ordered expansion), but score leaves with the policy-value network's
  value head (~1 ms per forward pass) rather than playing a full random
  rollout. Each simulation becomes much cheaper, so an equal time budget
  buys a much larger tree. One design constraint to get right: terminal
  outcomes and network values must be backed up on a single consistent
  scale (win = 1, draw = 0.5, loss = 0 from each node's own perspective).
- **PUCT selection with policy priors** — the network's policy head provides
  move priors, so selection can use AlphaZero's PUCT rule
  (`Q + c · P · √N / (1 + n)`) instead of vanilla UCT, focusing search on
  moves the network already believes in.
- **Close the training loop** — train the network on the hybrid's MCTS
  visit-count distributions instead of raw self-play moves (true AlphaZero
  training). This is also the strongest version of item 2's
  "search-improved targets".
- **Batched leaf evaluation** — buffer several pending leaves and evaluate
  them in one forward pass (with virtual loss) to amortize inference latency.
- **Enter it in the tournament** — same protocol as [results.md](results.md)
  (alternating seats, ≥50 games) against `mcts-700` and `minimax-7`.
