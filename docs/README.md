# Documentation

Documentation for the Connect 4 AI engine — three game-AI paradigms (minimax,
MCTS, self-play deep RL) implemented from scratch and compared head-to-head.
For install and quickstart, see the [project README](../README.md).

## Start here

| You want to... | Read |
|---|---|
| Play against the AI right now | [02-commands.md](02-commands.md) |
| Understand the design | [04-architecture.md](04-architecture.md) |
| See the numbers | [07-results.md](07-results.md) |
| Navigate the code | [03-file-structure.md](03-file-structure.md) |

## Suggested reading order

Setup first, then the code, then the science:

| Doc | One line |
|---|---|
| [01-overview.md](01-overview.md) | What the project is, why Connect 4, the three agents, headline findings, tech stack |
| [02-commands.md](02-commands.md) | Install steps + complete CLI reference: every subcommand, every flag, agent-spec grammar, common workflows |
| [03-file-structure.md](03-file-structure.md) | Annotated repo map: every module, its key symbols, and where to look for a given change |
| [04-architecture.md](04-architecture.md) | Design rationale: the OO engine vs. the vectorized engine, agent contract, lazy imports |
| [05-agents/](05-agents/README.md) | Per-agent algorithm write-ups: minimax, MCTS, RL policy, and the baselines |
| [06-training.md](06-training.md) | Self-play RL pipeline: vectorized engine, hyperparameters, and the v1→v3 debugging journey |
| [07-results.md](07-results.md) | Full tournament tables, MCTS/minimax scaling curves, decision-time analysis, RL failure analysis |
| [08-future-work.md](08-future-work.md) | Roadmap: minimax upgrades, RL training against stronger opponents, and beyond |
| [report.pdf](report.pdf) | The original CS5100 final report this project was submitted as |
