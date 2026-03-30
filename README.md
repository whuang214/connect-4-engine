# Connect 4 AI Engine

Compare different AI agents:  
**MCTS • RL • Rule-Based • Random • Human**

Run everything from one entrypoint:

```bash
python -m scripts.run <mode> [options]
```

## Modes

### 🎮 CLI (Terminal Game)

Play in terminal.

```bash
python -m scripts.run play --agent1 human --agent2 mcts-2000
```

### 🖥 UI (Graphical Game)

Launch GUI.

```bash
python -m scripts.run ui --agent1 human --agent2 mcts-2000
```

### 📊 EVAL (Benchmark)

Run multiple games + stats.

```bash
python -m scripts.run eval --agent1 mcts-1000 --agent2 rule --games 50
```

## Agents

- `human` → user input  
- `random` → random moves  
- `rule` → simple heuristic baseline  
- `mcts` → Monte Carlo Tree Search  
- `rl` → trained reinforcement learning model  

## Agent Naming (MCTS Shortcut)

You can specify MCTS iterations directly in the agent name:

```
mcts-200     → MCTS with 200 iterations
mcts-1000    → MCTS with 1000 iterations
mcts-5000    → MCTS with 5000 iterations
```

This replaces the need for `--iterations1` and `--iterations2`.

## Agent Options

### General

- `--agent1`, `--agent2` → human | random | rule | mcts | rl | mcts-<n>  
- `--name1`, `--name2` → custom names  

### MCTS (Optional)

- `--iterations1`, `--iterations2` → number of simulations  
  *(only needed if not using `mcts-<n>` format)*  


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
- `mcts-<n>` is the recommended way to configure MCTS strength quickly  
