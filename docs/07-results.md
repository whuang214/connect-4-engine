# Empirical Results

Every number below comes from the raw JSON in [`results/`](../results/),
produced by the shared tournament harness
([`evaluation/tournament.py`](../connect4/evaluation/tournament.py) over
[`evaluation/evaluate.py`](../connect4/evaluation/evaluate.py)). The
training-side story behind the RL numbers is in [06-training.md](06-training.md);
the proposed fix is in [08-future-work.md](08-future-work.md).

## Experimental setup

- **Protocol** — head-to-head matchups; the first player alternates every game
  (odd games: agent 1 starts), and wins are recorded per seat so first-player
  advantage is visible. Per-move wall-clock time is recorded for every agent.
  Results are persisted to JSON after every matchup, so interrupted runs lose
  nothing.
- **Game counts by cost tier** (from
  [`cli/tournament.py`](../connect4/cli/tournament.py)) — fast matchups
  (RL vs minimax, non-MCTS baselines) get 100 games; matchups involving
  MCTS-700 or MCTS-200 get 20; MCTS-1000 gets 14; MCTS-2000 gets 10. The
  follow-up MCTS-vs-minimax scaling sweeps use 30 games per matchup, and 50
  for the extended head-to-head.
- **Agents** — [minimax](../connect4/agents/minimax.py) (alpha-beta,
  fixed depth), [MCTS](../connect4/agents/mcts.py) (UCT, tactical
  overrides, safe rollouts, tree reuse),
  [RL](../connect4/agents/rl_policy.py) (policy-value ResNet loaded from
  `runs/rl_pure_selfplay_v3/best_model.pt`, played at temperature 0.3 so
  repeated games differ; it applies the same one-ply win/block override as its
  training loop), plus [random](../connect4/agents/random.py) and
  [rule-based](../connect4/agents/rule_based.py) (win → block → center)
  baselines.
- **Scale** — five recorded runs, 1,968 games total:

| Run | File | Matchups | Games | Duration |
|---|---|---|---|---|
| Main tournament | [`tournament_results.json`](../results/tournament_results.json) | 26 | 1,528 | 8.24 h |
| Part 1: MCTS-700 vs minimax depths | [`mcts_vs_minimax_part1_*.json`](../results/mcts_vs_minimax_part1_mcts700_vs_depths.json) | 4 | 120 | 3.80 h |
| Part 2: MCTS iteration scaling vs Minimax-7 | [`mcts_vs_minimax_part2_*.json`](../results/mcts_vs_minimax_part2_mcts_scaling_vs_mm7.json) | 5 | 150 | ~5.23 h |
| Part 3: MCTS-700 vs Minimax-7 extended | [`mcts_vs_minimax_part3_*.json`](../results/mcts_vs_minimax_part3_mcts700_vs_mm7_extended.json) | 1 | 50 | 1.63 h |
| Part 4: MCTS-200 vs minimax depths | [`mcts_vs_minimax_part4_*.json`](../results/mcts_vs_minimax_part4_mcts200_vs_depths.json) | 4 | 120 | 1.49 h |
| **Total** | | **40** | **1,968** | **~20.4 h** |

(Part 2's file holds five matchups, 200–1500 iterations, and no summary
header; its duration is the sum of the recorded matchup durations. The
MCTS-2000 data point comes from the main tournament.)

## Headline: MCTS > Minimax > RL

The final hierarchy across all runs is **MCTS-700 > Minimax-7 > Rule-Based >
RL > Random**. The flagship matchup, **MCTS-700 vs Minimax-7, was sampled four
independent times for 130 total games: MCTS won 78 (60.0%), lost 35 (26.9%),
drew 17 (13.1%).**

| Sample | Games | MCTS W–L–D | MCTS win rate |
|---|---|---|---|
| Tournament, core matchup | 20 | 14–5–1 | 70% |
| Part 3 extended head-to-head | 50 | 30–14–6 | 60% |
| Part 1, depth-7 row | 30 | 17–6–7 | 57% |
| Part 2, 700-iteration row | 30 | 17–10–3 | 57% |
| **Combined** | **130** | **78–35–17** | **60.0%** |

The result is not a first-player artifact — MCTS-700 won from both seats. In
the 50-game part 3 sample it won 14 of 25 games as Player 1 and 16 of 25 as
Player 2 (Minimax-7 won 4 as P1 and 10 as P2; 6 draws). Across all four
samples, 42 of MCTS's 78 wins came from the second seat.

## MCTS vs minimax depth

MCTS-700 against increasing minimax depth (part 1, 30 games each; tournament,
20 games each):

| Minimax depth | Part 1 (MCTS W–L–D) | Tournament (MCTS W–L–D) | Combined MCTS win rate |
|---|---|---|---|
| 3 | 25–5–0 (83%) | 17–3–0 (85%) | **84%** (42/50) |
| 5 | 22–6–2 (73%) | 13–5–2 (65%) | **70%** (35/50) |
| 7 | 17–6–7 (57%) | 14–5–1 (70%) | **60%** (78/130, all four samples) |
| 9 | 13–15–2 (43%) | 15–4–1 (75%) | **56%** (28/50) |

The same sweep with a weaker MCTS-200 (part 4, 30 games each) flips the
outcome — 200 iterations is not enough to out-search depth-5+ minimax:

| Minimax depth | MCTS-200 W–L–D | MCTS-200 win rate |
|---|---|---|
| 3 | 16–13–1 | 53% |
| 5 | 13–15–2 | 43% |
| 7 | 13–15–2 | 43% |
| 9 | 11–17–2 | 37% |

## MCTS iteration scaling vs Minimax-7

Part 2 (30 games per row) plus the tournament's iteration-scaling rows:

| MCTS iterations | Part 2 (W–L–D) | Tournament (W–L–D) | Combined MCTS win rate |
|---|---|---|---|
| 200 | 8–21–1 (27%) | 10–8–2 (50%, 20 g) | **36%** (18/50) |
| 500 | 20–7–3 (67%) | — | **67%** (20/30) |
| 700 | 17–10–3 (57%) | 14–5–1 (70%, 20 g) | **60%** (78/130, all samples) |
| 1000 | 17–8–5 (57%) | 8–4–2 (57%, 14 g) | **57%** (25/44) |
| 1500 | 16–10–4 (53%) | — | **53%** (16/30) |
| 2000 | — | 8–1–1 (80%, 10 g) | **80%** (8/10) |

The crossover sits between 200 and 500 iterations; from 500 to 1500 the win
rate plateaus around 53–67%, and the 80% at 2000 rests on only 10 games.
MCTS-200 vs Minimax-7 is the noisiest cell: pooling its three independent
samples (parts 2 and 4 plus the tournament) gives 31–44–5 over 80 games (39%).

## Baselines

| Agent | vs Random | vs Rule-Based |
|---|---|---|
| Minimax-7 | 100% (100 g) | 100% (100 g) |
| MCTS-700 | 100% (20 g) | 95% (20 g, 19–1–0) |
| RL (best) | 92% (100 g, 92–7–1) | 37% (100 g, 37–59–4) |

## The RL agent: a clean negative result

The self-play RL agent beat Random 92% of the time but lost to everything
else — including the trivial rule-based agent (37%) and **every search
configuration at every checkpoint**. Across 464 games against search agents
(RL-best vs Minimax-3/5/7/9 at 100 games each, plus MCTS-200/700/1000/2000),
it scored **zero wins and zero draws**.

Learning curve — intermediate checkpoints vs Minimax-7, 100 games each:

| Checkpoint (episode) | 100k (100,352) | 200k (200,192) | 300k (300,032) | 400k (400,384) | 500k (500,224) | best (275,456) |
|---|---|---|---|---|---|---|
| Win rate vs Minimax-7 | 0% | 0% | 0% | 0% | 0% | 0% |

The curve is perfectly flat: 400,000 additional episodes bought nothing
against search. **The 37% against rule-based is the tell.** Both agents share
an identical hard-coded one-ply win/block override, so that matchup isolates
the quality of the network's *non-forced* moves — and against an opponent
whose entire remaining policy is "play the most central column," the network's
positional choices lost 59% of games. The network learned enough board sense
to punish random play, but it routinely builds (and walks into) positions
where the opponent creates a double threat, which no one-ply override can
stop. Search agents manufacture such threats deliberately, hence 0%. The
training-side diagnosis — a behavior-cloning policy target (outcomes only
supervise the unused value head) plus a training distribution that never
contains strong adversarial positions — is laid out in
[06-training.md](06-training.md#honest-outcome-what-the-log-shows), and the
AlphaZero-style remedy in [08-future-work.md](08-future-work.md).

## Decision time and compute fairness

Average seconds per move, aggregated across the matchup JSONs (per-move cost
varies with opponent and game length, hence the ranges):

| Agent | Avg s/move (range across matchups) |
|---|---|
| Random | < 0.0001 |
| Rule-Based | 0.001 |
| RL (best) | 0.003–0.004 |
| Minimax-3 | 0.003–0.004 |
| Minimax-5 | 0.031–0.051 |
| Minimax-7 | 0.20–0.46 |
| Minimax-9 | 2.6–3.9 |
| MCTS-200 | 1.9–2.7 |
| MCTS-500 | 4.4 |
| MCTS-700 | 5.6–9.2 |
| MCTS-1000 | 8.3–10.1 |
| MCTS-1500 | 12.4 |
| MCTS-2000 | 15.7–22.5 |

**Caveat: the headline matchup is not compute-fair.** In their head-to-head
games MCTS-700 spent 5.6–6.3 s/move against Minimax-7's 0.20–0.28 s/move —
roughly 13–28× the per-move compute depending on which matchups you measure
(the report cites 13×, using Minimax-7's 0.44 s/move from its faster games).
The closest thing to compute parity in the data is Minimax-9 (2.6–3.9 s/move)
against MCTS-700 (~5.8 s/move): over the combined 50-game depth-9 sample the
score narrows to 28–19–3 (56% vs 38%), and in the equal-condition part 1
sweep alone Minimax-9 actually led 15–13. MCTS converts compute into the
strongest play here, but minimax extracts far more strength per second; under
a strict per-move time budget the ranking could flip.

## Regenerating the results

Run from the repo root (the model loads from `runs/`, output goes to
`results/`, and the JSON is rewritten after every matchup):

```bash
python -m connect4 tournament            # 26 matchups, 1,528 games (recorded: 8.24 h; --quick ~15 min)
```

The four `mcts_vs_minimax_part1..4` files were produced by scaling sweeps
whose runner has since been folded into the tournament harness; the recorded
durations stand as historical facts (part 1: 3.8 h for 120 games; part 2:
~5.2 h for 150; part 3: 1.6 h for 50; part 4: 1.5 h for 120). Today the
systematic round-robin is `python -m connect4 tournament`, and any individual
matchup can be reproduced with `eval` (which prints a summary but does not
write JSON):

```bash
python -m connect4 eval --agent1 mcts-700 --agent2 minimax-7 --games 50   # part 3's extended head-to-head
```

`python -m connect4 tournament` accepts `--quick` for a smoke-test pass with
reduced game counts and `--skip-slow` to drop the MCTS-2000 and Minimax-9
tiers. Note one divergence between the shipped JSON and a fresh run: the
tournament's five RL learning-curve matchups are skipped automatically unless
the intermediate checkpoints exist locally (only `best_model.pt` ships in the
repo). Raw output: [`results/tournament_results.json`](../results/tournament_results.json)
and [`results/mcts_vs_minimax_part1..4_*.json`](../results/).
