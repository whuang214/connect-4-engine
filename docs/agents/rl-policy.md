# RL Policy Agent

## Algorithm

A deep policy-value network trained by pure self-play (no search at training
or play time), wrapped with the shared tactical override. At play time it is
a single forward pass: encode the board, mask illegal columns, and take the
highest-probability move. It is the project's honest negative result — fast
(~4 ms/move) and clearly better than random, but it lost **every game**
against every search agent. The analysis below explains why.

## How it works

### Network

[`PolicyValueNet`](../../src/connect4/models/policy_value_network.py) —
a residual CNN with ~2.3M parameters:

| Stage | Structure |
|---|---|
| Stem | 3×3 conv, 4 → 128 channels + BatchNorm + ReLU |
| Trunk | 6 residual blocks (two 3×3 convs + BN each, identity skip) |
| Policy head | 1×1 conv → 32 ch → FC 1344 → 128 → **7 logits** (one per column) |
| Value head | 1×1 conv → 32 ch → FC 1344 → 256 → 1 → **tanh, value in [-1, 1]** |

(A smaller `PolicyValueNetSmall` plain-CNN variant exists for quick runs;
the shipped model is the full network.)

### Input encoding

`encode_board(game)` produces a `(4, 6, 7)` tensor **from the current
player's perspective**: channel 0 = my pieces, 1 = opponent pieces,
2 = side-to-move plane (all-ones iff player 1 moves), 3 = normalized
per-column heights tiled over rows. The identical encoding is produced by the
vectorized training engine (`VecConnect4.encode`), so training and play see
bit-identical inputs.

### Action selection

`choose_action`, in order:

1. Single legal move → play it.
2. **Tactical override** — `find_immediate_win` for self (play the win), then
   for the opponent (block). This mirrors the win > block override applied
   during training-time data generation, so play stays consistent with the
   tactical assumptions baked into the training data.
3. With probability `epsilon`, a random legal move (0 at eval time).
4. Otherwise one network forward pass. Illegal columns are masked by adding
   −1e9 to their logits. If `temperature > 0`, sample from
   `softmax(logits / temperature)`; if `temperature == 0`, take the argmax.

The value head is exposed via `evaluate_position(game) -> float` but is
**not consulted** during move selection — a fact central to the failure
analysis below.

## Parameters

| Constructor arg | Default | Notes |
|---|---|---|
| `model` / `model_path` | fresh network | Pass a loaded module or a `.pt` checkpoint path |
| `epsilon` | 0.0 | Probability of a uniform-random legal move |
| `temperature` | 0.0 | 0 = argmax; > 0 = softmax sampling (tournament used 0.3 to vary games) |
| `device` | CUDA if available | |
| `small_network` | False | Use `PolicyValueNetSmall` when no config is in the checkpoint |

CLI spec: `rl`, which resolves to
`runs/rl_pure_selfplay_v3/best_model.pt` by default. Overrides:
`--model1/--model2 <run-folder>` (a folder inside `runs/`),
`--checkpoint1/--checkpoint2 {best,final}`, or an explicit
`--model-path1/--model-path2 <file.pt>`. Example:

```bash
connect4 eval --agent1 rl --model1 rl_pure_selfplay_v3 --checkpoint1 best --agent2 rule --games 50
```

## Tournament performance

From [`results/tournament_results.json`](../../results/tournament_results.json)
(RL-best played with `temperature=0.3`):

| Opponent | Games | RL record (W-L-D) | Win rate |
|---|---|---|---|
| Random | 100 | 92-7-1 | **92%** |
| Rule-based | 100 | 37-59-4 | **37%** |
| Minimax-3 / 5 / 7 / 9 | 100 each | 0-100-0 each | **0%** |
| MCTS-200 / 700 / 1000 / 2000 | 20 / 20 / 14 / 10 | 0 wins, 0 draws | **0%** |

The learning curve is flat where it matters: checkpoints at 100k, 200k, 300k,
400k, and 500k episodes **each went 0-100 against minimax-7**. The
training-time eval log
([`runs/rl_pure_selfplay_v3/training_log.json`](../../runs/rl_pure_selfplay_v3/training_log.json))
tells the same story — win rate vs a 200-iteration MCTS opponent never
exceeded 10% across the entire run, while vs Random it reached ~100%.

### Why it failed against search

The failure is structural, not a tuning miss — it follows from the training
setup in [`training/trainer.py`](../../src/connect4/training/trainer.py):

1. **The policy imitates played moves without outcome weighting.** The policy
   loss is plain `cross_entropy(logits, actions)` over whatever moves the
   (epsilon/temperature-noised) policy actually played — a move from a lost
   game is reinforced exactly as strongly as one from a won game. Game
   outcomes (±1 returns) reach only the **value head** via MSE, and the value
   head is never used at play time. The net effect is behavior cloning of a
   noisy version of itself: the policy converges toward its own average, and
   the plateau at ~500k episodes is the expected result.
2. **Pure self-play distribution mismatch.** Every training position comes
   from the network playing itself (the trainer maintains a frozen-checkpoint
   opponent pool, but the vectorized self-play loop never draws from it, so
   both seats are always the live network). Minimax and MCTS immediately
   steer games into precise forced lines — multi-move threats, double
   attacks — that the self-play distribution essentially never produced, so
   the policy has no learned response to them.
3. **One forward pass sees only one ply.** The win > block override (applied
   both in training data and at play time) keeps it from losing to
   one-move tactics — which is enough to beat Random 92% — but two-ply
   threats such as a fork are invisible to a single policy evaluation. The
   rule-based agent, which shares the same override plus a center bias, beats
   it 59-37: the network's learned "positional" play is worth less than a
   fixed center heuristic against exact tactics.

The established fix is to put search back in the loop — train against
stronger opponents and on search-improved targets. The planned approaches are
laid out in [future-work.md](../future-work.md).

## Implementation notes

Source: [`agents/rl_policy.py`](../../src/connect4/agents/rl_policy.py).

- **Lazy torch.** The factory imports `RLPolicyAgent` only inside the `rl`
  branch and the `agents` package re-exports lazily (PEP 562), so every other
  agent — and `connect4 --help` — never pays the torch import.
- **Checkpoint loading** (`_load_model`) prefers the `config` dict saved in
  the checkpoint to reconstruct the architecture (`channels`, `num_blocks`,
  `dropout`); with no config it sniffs state-dict keys (a `features.` prefix
  means the small network). Checkpoints store model + optimizer + episode +
  metadata (`save_model`).
- The shipped weights are `runs/rl_pure_selfplay_v3/best_model.pt` — "best"
  as gated by the trainer's periodic eval score against Random/Rule/MCTS, the
  only `.pt` committed to the repo.
- Training context (512-env vectorized self-play at 93–165 episodes/s on an
  RTX 2080 Ti, mirror augmentation, circular replay buffer, the v1→v3
  fixes): see [training/trainer.py](../../src/connect4/training/trainer.py)
  and the run config in
  [`runs/rl_pure_selfplay_v3/config.json`](../../runs/rl_pure_selfplay_v3/config.json).
