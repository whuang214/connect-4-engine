# Connect 4 AI Engine

Compare different AI agents:
MCTS • RL • Rule-Based • Random • Human

Run everything from one entrypoint:

```bash
python -m scripts.run <mode> [options]
```

## Modes

### 🎮 CLI (Terminal Game)

Play in terminal.

```bash
python -m scripts.run play --agent1 human --agent2 mcts --iterations2 2000
```

### 🖥 UI (Graphical Game)

Launch GUI.

```bash
python -m scripts.run ui --agent1 human --agent2 mcts --iterations2 2000
```

### 📊 EVAL (Benchmark)

Run multiple games + stats.

```bash
python -m scripts.run eval --agent1 mcts --agent2 rule --iterations1 1000 --games 50
```

## Agents

- `human` → user input
- `random` → random moves
- `rule` → simple heuristic baseline
- `mcts` → Monte Carlo Tree Search
- `rl` → trained reinforcement learning model

## Agent Options

### General

- `--agent1`, `--agent2` → human | random | rule | mcts | rl
- `--name1`, `--name2` → custom names

### MCTS

- `--iterations1`, `--iterations2` → number of simulations  
  (higher = stronger, slower)

### RL

- `--model1`, `--model2` → path to trained model

### CLI / Eval Extras

- `--no-render` → disable terminal board
- `--games` → number of eval games
- `--render` → show board during eval
- `--print-each-game`
- `--print-moves`

## Notes

- Use `eval` mode for benchmarking agents
