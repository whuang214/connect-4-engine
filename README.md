# Connect 4 AI Engine

Compare different AI agents:  
**Minimax • MCTS • RL • Rule-Based • Random • Human**

Run everything from one entrypoint:

```bash
python -m scripts.run <mode> [options]
```

### Run Options

- `--agent1`, `--agent2` → agent type (see "Agents" below)
- `--name1`, `--name2` → custom display names
- `--model1`, `--model2` → path to RL checkpoint (for `rl` agent)

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+.

## Modes (examples)

### 🖥 UI (Graphical Game) — Recommended to Visualize

```bash
python -m scripts.run ui --agent1 mcts-500 --agent2 minimax-5
python -m scripts.run ui --agent1 human --agent2 minimax-7
python -m scripts.run ui --agent1 rule --agent2 rl --model2 runs/rl_pure_selfplay_v3/best_model.pt
```

### 🎮 CLI (Terminal Game)

```bash
python -m scripts.run play --agent1 human --agent2 minimax-5
python -m scripts.run play --agent1 mcts-500 --agent2 minimax-5
```

Specific Options:
- `--no-render` → disable board display (for faster self-play)

### 📊 EVAL (Benchmark)

Plays 10 games by default, but can specify more with `--games`.

```bash
python -m scripts.run eval --agent1 mcts-700 --agent2 minimax-7 --games 20
```

Specific Options:
- `--games` → number of games to play
- `--print-each-game` → print result of each game (instead of just final stats)

## Agents

- `human` → user input
- `random` → random legal move
- `rule` → win/block/center heuristic
- `mcts` → Monte Carlo Tree Search (default: 500 iterations)
- `minimax` → Minimax with alpha-beta pruning (default: depth 5)
- `rl` → trained reinforcement learning model

### Agent naming shortcuts

```
mcts-200     → MCTS with 200 iterations
mcts-700     → MCTS with 700 iterations
mcts-2000    → MCTS with 2000 iterations
minimax-3    → Minimax at depth 3
minimax-7    → Minimax at depth 7
minimax-9    → Minimax at depth 9
```

## Training the RL Agent

```bash
python -m training.train_policy_rl --episodes 500000 --run-name my_run --n-envs 512 --updates-per-batch 128
```

Checkpoints save to `runs/<run-name>/checkpoints/`. Use `--resume` to continue from a checkpoint.

## Running the Tournament

```bash
# Full tournament (~8 hours)
python scripts/run_tournament.py

# Quick smoke test (~15 min)
python scripts/run_tournament.py --quick

# MCTS vs Minimax deep comparison (run parts in parallel)
python scripts/run_mcts_vs_minimax.py --part 1
python scripts/run_mcts_vs_minimax.py --part 2
python scripts/run_mcts_vs_minimax.py --part 3
python scripts/run_mcts_vs_minimax.py --part 4
```

Results save to `results/` as JSON.

## Project Structure

```
connect-4-engine/
├── engine.py                    # Game logic
├── agents/                      # All agent implementations
├── models/                      # Neural network architecture
├── training/                    # RL training pipeline + vectorized engine
├── evaluation/                  # Head-to-head game runner
├── scripts/                     # Entrypoints (play, eval, tournament)
├── ui/                          # Pygame GUI
├── results/                     # Tournament output (JSON)
└── runs/                        # RL checkpoints and training logs
```