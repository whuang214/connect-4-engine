# RL Training Pipeline

How the self-play RL agent (`rl` in the CLI) was trained, what every knob does,
how to reproduce or resume a run, and — importantly — what went wrong across
three training runs and what the final run actually achieved.

Everything here is grounded in [`trainer.py`](../connect4/training/trainer.py),
[`runs/rl_pure_selfplay_v3/config.json`](../runs/rl_pure_selfplay_v3/config.json),
[`runs/rl_pure_selfplay_v3/training_log.json`](../runs/rl_pure_selfplay_v3/training_log.json),
and the [project report](report.pdf).

## What gets trained

A 2,306,839-parameter policy-value ResNet
([`PolicyValueNet`](../connect4/models/policy_value_network.py): 128-channel
stem, 6 residual blocks, 7-logit policy head, tanh value head). Boards are
encoded from the current player's perspective as 4 channels (my pieces,
opponent pieces, side-to-move, normalized column heights) by `encode_board`,
which is bit-identical between the sequential engine and the vectorized
training engine (pinned by tests).

At play time, [`RLPolicyAgent`](../connect4/agents/rl_policy.py) masks
illegal columns, applies the same win > block tactical override used during
training, and picks argmax (or samples at a configurable temperature). Only the
policy head is used to select moves; the value head is trained but unused at
inference.

## Pipeline walkthrough

One outer loop of `Trainer.run()` in
[`trainer.py`](../connect4/training/trainer.py):

1. **Vectorized self-play** — `play_selfplay_vectorized` plays 512 games at
   once on [`VecConnect4`](../connect4/training/vec_engine.py), a batched
   `(n, 6, 7)` NumPy engine with O(1) height-based drops. All 512 positions are
   encoded and pushed through the network in a single forward pass. Moves are
   sampled from the temperature-scaled masked softmax; an epsilon fraction of
   environments play a uniform random legal move instead.
2. **Tactical override** — before actions are committed, `_find_tactical_move`
   scans each board: an immediate winning column overrides everything, else an
   immediate block of the opponent's win, else the network/epsilon move stands.
   The *overridden* action is what gets recorded, so the training data contains
   no one-ply blunders.
3. **Trajectories and returns** — states and actions are collected per
   environment *and per player*. When a game ends, every move by the winner is
   labeled `+1`, every move by the loser `-1`, draws `0`. There is no reward
   shaping and no discounting — pure terminal returns.
4. **Mirror augmentation** — Connect 4 is left-right symmetric, so each
   (state, action, return) is duplicated with `mirror_encoded_state` /
   `mirror_action`, doubling data for free. (The trainer defers this to
   `ReplayBuffer.add_batch` rather than doing it inside the self-play loop.)
5. **Replay buffer** — a 1,000,000-transition `ReplayBuffer` backed by
   pre-allocated NumPy arrays with a circular write pointer. `add_batch` writes
   whole batches with a vectorized wraparound split; `sample` is O(1) uniform
   indexing. (This class exists because of a v2 bug — see the debugging journey
   below.)
6. **Updates** — after each outer loop, 128 gradient steps (`updates_per_batch`)
   on batches of 1,024:
   `loss = 1.0 · CE(policy logits, played action) + 1.0 · MSE(value, return) − 0.05 · entropy`,
   optimized with AdamW (lr 3e-4, weight decay 1e-4) and gradient clipping at
   norm 1.0. A `CosineAnnealingLR` schedule steps once per outer loop over
   `episodes // n_envs` total steps, decaying to 1e-5. Note the policy target:
   cross-entropy toward the action *actually played*, regardless of outcome.
   The ±1 return supervises only the value head. This matters for the results —
   see [Honest outcome](#honest-outcome-what-the-log-shows).
7. **Frozen-checkpoint opponent pool** — `maybe_snapshot` freezes a CPU copy of
   the network every 10k episodes into a pool capped at 8 (FIFO eviction), and
   `_pick_opponent` implements a 45% chance of selecting a past self from that
   pool. **However, verified against the code: the fully-batched self-play loop
   never calls `_pick_opponent`** — every v3 training game was
   current-network-vs-current-network, and opponent diversity came only from
   the epsilon/temperature noise. The pool is maintained but unused; wiring it
   into the vectorized loop is an open improvement.
8. **Periodic eval and best-model gate** — every ~25k episodes (the check
   fires on the first 512-episode step past each 25k boundary) a greedy copy
   of the network (ε = 0, τ = 0, CPU) plays 100 games vs Random, 100 vs Rule-Based,
   and 20 vs MCTS-200, alternating seats. The gate score is
   `rule_wr + 0.5 · random_wr + 0.75 · mcts_wr`; a new high saves
   `best_model.pt`. The shipped `best_model.pt` is from **episode 275,456**
   (score 1.50: rule 100%, random 100%, MCTS 0%). Full checkpoints
   (model + optimizer + scheduler + episode + config) save every 25k episodes
   to `runs/<run-name>/checkpoints/`, and `training_log.json` is rewritten at
   every eval.

## Hyperparameters (v3)

Values from [`runs/rl_pure_selfplay_v3/config.json`](../runs/rl_pure_selfplay_v3/config.json);
flags from [`cli/train.py`](../connect4/cli/train.py) (`python -m connect4 train --help`).

| Config key | v3 value | CLI flag | Notes |
|---|---|---|---|
| `episodes` | 1,000,000 | `--episodes` | Run stopped at ~525k (see below) |
| `n_envs` | 512 | `--n-envs` | Parallel self-play games per outer loop |
| `lr` / `min_lr` | 3e-4 / 1e-5 | `--lr` / `--min-lr` | Cosine annealing between the two |
| `weight_decay` | 1e-4 | `--weight-decay` | AdamW |
| `grad_clip` | 1.0 | `--grad-clip` | Global norm |
| `batch_size` | 1,024 | `--batch-size` | |
| `buffer_size` | 1,000,000 | `--buffer-size` | Circular NumPy replay buffer |
| `updates_per_batch` | 128 | `--updates-per-batch` | Gradient steps per outer loop (CLI default is 256) |
| `channels` / `num_blocks` | 128 / 6 | `--channels` / `--num-blocks` | ~2.3M params |
| `dropout` | 0.1 | `--dropout` | |
| `epsilon_start → end` | 0.3 → 0.05 | `--epsilon-start` / `--epsilon-end` | Linear over 600k episodes (`--epsilon-decay-episodes`) |
| `temperature_start → end` | 2.0 → 0.3 | `--temperature-start` / `--temperature-end` | Linear over 800k episodes (`--temperature-decay-episodes`) |
| `policy` / `value` / `entropy` weight | 1.0 / 1.0 / 0.05 | `--policy-weight` etc. | Loss mix |
| `augment_mirror` | true | `--augment-mirror` | Left-right mirroring in the buffer |
| `snapshot_interval` / `max_checkpoint_pool` | 10,000 / 8 | `--snapshot-interval` / `--max-checkpoint-pool` | Pool maintained but not consulted (see step 7) |
| `eval_interval` / `eval_games` / `eval_games_small` | 25,000 / 100 / 20 | `--eval-interval` etc. | 20 games apply to the MCTS opponent |
| MCTS eval opponent | 200 iterations | `--mcts-eval-iterations` | |
| `save_interval` / `log_interval` | 25,000 / 2,048 | `--save-interval` / `--log-interval` | |
| `seed` | 42 | `--seed` | Python/NumPy/torch/CUDA |

## How to train

From the repo root (paths resolve relative to the CWD):

```bash
pip install -r requirements.txt
python -m connect4 train --episodes 1000000 --run-name rl_pure_selfplay_v3 \
    --n-envs 512 --updates-per-batch 128
```

Output lands in `runs/<run-name>/`: `config.json`, `training_log.json`,
`best_model.pt`, `final_model.pt` (on completion), and
`checkpoints/checkpoint_ep<N>.pt` every 25k episodes.

**Resume** by pointing `--resume` at a checkpoint, keeping the same run name
and env count:

```bash
python -m connect4 train --episodes 1000000 --run-name rl_pure_selfplay_v3 \
    --n-envs 512 --updates-per-batch 128 \
    --resume runs/rl_pure_selfplay_v3/checkpoints/checkpoint_ep500224.pt
```

Resume restores the model and optimizer, then **deliberately skips the saved
scheduler state**: because the cosine `T_max` was corrected between runs
(see below), `run()` instead re-derives `scheduler.last_epoch =
start_episode // n_envs` so the LR lands exactly where the corrected schedule
says it should be. Epsilon and temperature are pure functions of the absolute
episode count, so they resume correctly for free. Note that `config.json` in
the run directory is overwritten with the current arguments on every launch.

Hardware for the shipped run: an NVIDIA RTX 2080 Ti with CUDA 11.8, sustaining
93–165 episodes/s (report, §2 and §4.4).

## The debugging journey: v1 → v3

The shipped `rl_pure_selfplay_v3` is the third attempt. The first two failed
for instructive reasons (report, §2.3 and §4.4).

**v1 — wrong opponent.** The first run trained the network against the
rule-based and random agents rather than against itself. It was abandoned not
for a technical reason but a methodological one: the project required each
paradigm (minimax, MCTS, RL) to be built independently with no shared
components or information exchange, and training against another agent
violates that. v1's data was discarded.

**v2 — pure self-play, two silent bugs.** The rewrite trained by self-play but
crawled at roughly 2 episodes/s and never improved. Two root causes:

1. *O(n) replay buffer.* The buffer was a `collections.deque`, which has O(n)
   random access — so uniform sampling got slower as the buffer filled, and
   throughput degraded linearly with buffer size. The fix is the current
   `ReplayBuffer`: pre-allocated NumPy arrays, a circular write pointer, a
   vectorized `add_batch` that splits writes across the wraparound boundary,
   and O(1) fancy-indexed sampling. Per the report this was "the single most
   important performance fix in the RL training pipeline"; together with the
   512-env vectorized engine it took throughput from ~2 to 93–165 episodes/s —
   roughly two orders of magnitude.
2. *Misconfigured cosine schedule.* `CosineAnnealingLR`'s `T_max` was set
   relative to the total episode count rather than the number of times the
   scheduler actually steps (once per 512-episode outer loop), so the LR
   sat essentially flat at its initial value for the whole run. The fix in
   `Trainer.__init__` is `T_max = episodes // n_envs`, with the resume path
   re-deriving `last_epoch` instead of restoring stale scheduler state (see
   the comments in `_load_checkpoint` and `run()`).

The v2 model collapsed to near-zero policy entropy and never beat a non-random
opponent.

**v3 — both fixes plus the tactical override.** v3 fixed both bugs, added the
win > block override to the data-generation loop, and trained from scratch.
It was configured for 1M episodes but stopped a little past 500k (last logged
eval at episode 525,312) after the evaluation curves had been flat for
hundreds of thousands of episodes. Its `best_model.pt` (episode 275,456) is
the only shipped weights file and the model used in every tournament result.

## Honest outcome: what the log shows

[`training_log.json`](../runs/rl_pure_selfplay_v3/training_log.json) has 21
eval points from episode 25,088 to 525,312:

- **`eval_vs_random`**: 0.95–1.00 throughout. Learned almost immediately,
  stayed there.
- **`eval_vs_rule`**: oscillates between exactly 0.0, 0.5, and 1.0 with no
  trend. The quantization is itself diagnostic: the eval agent is greedy
  (ε = 0, τ = 0) and the rule-based agent is deterministic, so the "100 games"
  collapse to 2 distinct games (one per seat) replayed 50 times each — the
  metric can only take three values, and it flips between checkpoints as small
  weight changes reroute those two games.
- **`eval_vs_mcts`** (20 games vs MCTS-200): 0.00–0.10, mostly 0.00, no trend.
- Training losses are equally flat: policy loss ~1.54–1.57, value loss
  0.54–0.65, entropy ~1.55 (near-uniform over 7 moves is ln 7 ≈ 1.95) from
  50k episodes to the end, while the LR followed the corrected cosine from
  3e-4 down to 1.4e-4.

In short: the network learned strong-vs-random positional play early and then
plateaued permanently. Two compounding explanations. First, the **policy
target is behavior cloning**: cross-entropy toward the played action with no
outcome weighting means losing moves are imitated exactly as hard as winning
ones — the ±1 terminal returns only ever supervise the value head, and the
value head is never consulted when choosing a move. The policy can therefore
converge to "what I tend to do" rather than "what wins." Second,
**distribution mismatch**: every training position comes from the network's
own (noised, tactically overridden) self-play, so positions created by strong
adversarial search — the double threats minimax and MCTS manufacture — are
essentially absent from the buffer, and the network has no gradient signal
about them. The tournament consequences (92% vs Random, 37% vs Rule-Based,
0% vs every search agent at every checkpoint) are analyzed in
[07-results.md](07-results.md); the standard fix — search-improved targets in an
AlphaZero-style loop — is sketched in [08-future-work.md](08-future-work.md).
