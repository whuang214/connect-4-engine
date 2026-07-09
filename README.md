# Connect 4 Arena

[![CI](https://github.com/whuang214/connect4-arena/actions/workflows/ci.yml/badge.svg)](https://github.com/whuang214/connect4-arena/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Can a self-play RL agent beat classical search?**

To find out, this project builds three very different Connect 4 AIs from
scratch — a **minimax** agent that calculates ahead, a **Monte Carlo Tree
Search** agent that simulates thousands of games per move, and a **deep
reinforcement-learning** agent that taught itself by playing 500,000 games
against itself — then throws them into a ~2,000-game tournament to see who
comes out on top.

**Spoiler:** search won. MCTS took the crown (beating depth-7 minimax in 60%
of 130 games), minimax swept the baselines, and the RL agent — despite
learning solid positional play — never beat a search agent once. Why that
happened turned out to be the most interesting part: the full story is in
[docs/07-results.md](docs/07-results.md).

## Get started

```bash
git clone https://github.com/whuang214/connect4-arena
cd connect4-arena
pip install -r requirements.txt
```

That's it — no install step for the project itself. Everything runs through
one command from the repo root:

```bash
python -m connect4 play --agent1 human --agent2 minimax-5     # play in the terminal
python -m connect4 eval --agent1 mcts-700 --agent2 minimax-7 --games 20
python -m connect4 eval --agent1 rule --agent2 rl --games 50  # the trained RL model ships in-repo
```

Want the graphical board? `pip install pygame` (or `pygame-ce` on
Windows ARM64), then:

```bash
python -m connect4 ui --agent1 human --agent2 minimax-7
```

Requires Python 3.10+. CUDA users: install torch from
[pytorch.org](https://pytorch.org/get-started/locally/) first.

## Commands

| Command | What it does |
|---|---|
| `python -m connect4 play` | Terminal game between any two agents (or you) |
| `python -m connect4 ui` | Pygame board with move timings, undo, and hover preview |
| `python -m connect4 eval` | Head-to-head benchmark with per-move timing |
| `python -m connect4 train` | Self-play RL training (512 parallel envs, resumable) |
| `python -m connect4 tournament` | The full round-robin from the report (~8h; `--quick` ≈ 15 min) |

Full flag reference and workflows: [docs/02-commands.md](docs/02-commands.md).

## The agents

| Spec | Agent | In one line |
|---|---|---|
| `human` | You | Keyboard / mouse input |
| `random` | Random | Uniform random legal move (baseline) |
| `rule` | Rule-based | Win if you can, block if you must, else play center (baseline) |
| `minimax`, `minimax-<depth>` | Minimax | Alpha-beta search with a positional evaluation (default depth 5) |
| `mcts`, `mcts-<iterations>` | MCTS | UCT search + tactical overrides + smart rollouts — the tournament winner |
| `rl` | Deep RL | 2.3M-parameter policy-value ResNet trained purely by self-play |

Per-agent write-ups: [docs/05-agents/](docs/05-agents/README.md).

## Results at a glance

| Matchup | Result |
|---|---|
| MCTS-700 vs Minimax-7 | **60% MCTS** over 130 games |
| MCTS-700 vs Minimax-3/5/9 | 84% / 70% / 56% MCTS |
| Minimax-7 vs Random & Rule | 100% |
| RL vs Random | 92% |
| RL vs Rule-Based | 37% |
| RL vs any search agent | 0% — flat from 100k to 500k training episodes |

Full tables and analysis: [docs/07-results.md](docs/07-results.md) ·
original report: [docs/report.pdf](docs/report.pdf).

## Training your own RL agent

```bash
python -m connect4 train --episodes 1000000 --run-name my_run --n-envs 512 --updates-per-batch 128
```

Checkpoints land in `runs/<run-name>/checkpoints/`; resume with `--resume`.
Pipeline details, hyperparameters, and the debugging war stories:
[docs/06-training.md](docs/06-training.md).

## Project structure

```
connect4/
├── engine.py            # game engine — undo stack enables in-place tree search
├── tactics.py           # shared win/block detection + center ordering
├── agents/              # human, random, rule-based, minimax, MCTS, RL + factory
├── models/              # policy-value residual CNN + board encoding
├── training/            # vectorized 512-env self-play engine + trainer
├── evaluation/          # head-to-head runner + tournament harness
├── ui/                  # pygame board
└── cli/                 # the `python -m connect4` subcommands
tests/                   # 90 unit tests (run: pytest)
docs/                    # numbered docs — start at docs/README.md
runs/                    # shipped tournament model + training logs
results/                 # tournament output (JSON)
```

Navigation map: [docs/03-file-structure.md](docs/03-file-structure.md) ·
design rationale (incl. why there are two game engines):
[docs/04-architecture.md](docs/04-architecture.md).

## Credits

CS5100 (Foundations of Artificial Intelligence) final project — Will Huang,
Joseph Winterlich, Soham Santra. MIT licensed.
