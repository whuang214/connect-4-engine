# Documentation

Documentation for the Connect 4 AI engine — three game-AI paradigms (minimax,
MCTS, self-play deep RL) implemented from scratch and compared head-to-head.
For install and quickstart, see the [project README](../README.md).

## Start here

| You want to... | Read |
|---|---|
| Play against the AI right now | [commands.md](commands.md) |
| Understand the design | [architecture.md](architecture.md) |
| See the numbers | [results.md](results.md) |
| Navigate the code | [file-structure.md](file-structure.md) |

## 30-second tour

| Doc | One line |
|---|---|
| [overview.md](overview.md) | What the project is, why Connect 4, the three agents, headline findings, tech stack |
| [architecture.md](architecture.md) | Design rationale: the OO engine vs. the vectorized engine, agent contract, lazy imports |
| [file-structure.md](file-structure.md) | Annotated repo map: every module, its key symbols, and where to look for a given change |
| [commands.md](commands.md) | Complete CLI reference: every subcommand, every flag, agent-spec grammar, common workflows |
| [results.md](results.md) | Full tournament tables, MCTS/minimax scaling curves, decision-time analysis, RL failure analysis |
| [training.md](training.md) | Self-play RL pipeline: vectorized engine, hyperparameters, and the v1→v3 debugging journey |
| [agents/](agents/README.md) | Per-agent algorithm write-ups: minimax, MCTS, RL policy, and the baselines |
| [future-work.md](future-work.md) | Roadmap: minimax upgrades, RL training against stronger opponents, and beyond |
| [report.pdf](report.pdf) | The original CS5100 final report this project was submitted as |
