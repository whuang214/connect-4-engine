# File Structure

Navigation map for the repository. Paths are relative to the repo root; all
CLI commands assume the repo root as working directory (model loading and
result output are CWD-relative).

## Annotated tree

```
connect4-arena/
├── requirements.txt            # torch, numpy, tqdm, pytest (pygame is a
│                               #   separate manual install for the UI)
├── conftest.py                 # puts the repo root on sys.path so plain
│                               #   `pytest` works without installing anything
├── README.md                   # quickstart + results at a glance
├── connect4/                   # the package itself, importable from the repo root
│   ├── engine.py               # Connect4: the reference OO game engine (undo stack,
│   │                           #   incremental win detection anchored at the last move)
│   ├── tactics.py              # shared tactical helpers: CENTER_ORDER,
│   │                           #   ordered_legal_moves, find_immediate_win
│   ├── agents/
│   │   ├── base.py             # BaseAgent ABC: choose_action + stats contract
│   │   ├── human.py            # terminal input agent
│   │   ├── random.py           # uniform random baseline
│   │   ├── rule_based.py       # win > block > center heuristic baseline
│   │   ├── minimax.py          # alpha-beta minimax + windowed positional heuristic
│   │   ├── mcts.py             # UCT search + tactical overrides + safe rollouts
│   │   │                       #   + tree reuse between moves
│   │   ├── rl_policy.py        # masked-logit policy agent (torch), tactical override
│   │   ├── factory.py          # parse_agent_config / create_agent — CLI spec strings
│   │   └── __init__.py         # lazy re-exports (PEP 562): torch loads only for RL
│   ├── models/
│   │   └── policy_value_network.py  # PolicyValueNet (residual CNN, ~2.3M params),
│   │                                #   PolicyValueNetSmall, encode_board, mirror_*
│   ├── training/
│   │   ├── vec_engine.py       # VecConnect4: batched (n,6,7) int8 NumPy engine
│   │   └── trainer.py          # self-play Trainer, ReplayBuffer, FrozenPolicyAgent,
│   │                           #   play_selfplay_vectorized, evaluate_against
│   ├── evaluation/
│   │   ├── evaluate.py         # head-to-head runner: evaluate_agents, per-move timing
│   │   └── tournament.py       # Matchup runner: run_matchups, JSON persistence, ETA
│   ├── ui/
│   │   └── game_ui.py          # pygame board (optional pygame install)
│   ├── cli/
│   │   ├── main.py             # argparse dispatcher (stdlib-only import time)
│   │   ├── game.py             # play / ui / eval subcommands + shared agent flags
│   │   ├── train.py            # train subcommand (full hyperparameter surface)
│   │   └── tournament.py       # tournament subcommand (the report's round-robin)
│   ├── __init__.py
│   └── __main__.py             # entry point for `python -m connect4`
├── tests/                      # unit tests for engine/agents/training/CLI
├── docs/                       # this documentation + report.pdf
├── runs/                       # training runs; rl_pure_selfplay_v3/ ships the
│                               #   tournament model (config.json, training_log.json,
│                               #   best_model.pt, checkpoints/)
└── results/                    # tournament output JSON + the four
                                #   mcts_vs_minimax part files from earlier
                                #   scaling sweeps
```

## Module reference

### Core

| File | Key symbols | What it does | Open it to... |
|---|---|---|---|
| [engine.py](../connect4/engine.py) | `Connect4`, `MoveResult`, `MoveHistory` | 6x7 game state; `make_move`/`undo_move` (full undo stack), `check_winner(row, col)` scans only from the last move, `get_legal_moves`, `clone`, `render`, `SYMBOLS` | change rules, win detection, rendering, or the state API agents consume |
| [tactics.py](../connect4/tactics.py) | `CENTER_ORDER`, `ordered_legal_moves`, `find_immediate_win` | Immediate win/block probing (undo-based fast path, clone fallback) and center-first move ordering, shared by rule-based, MCTS, and RL agents | change tactical detection used by three agents at once |

### Agents

| File | Key symbols | What it does | Open it to... |
|---|---|---|---|
| [agents/base.py](../connect4/agents/base.py) | `BaseAgent` | ABC: abstract `choose_action(game) -> int`; optional `reset_stats`/`get_stats`/`print_stats` contract used by the evaluator | see the contract a new agent must satisfy |
| [agents/human.py](../connect4/agents/human.py) | `HumanAgent` | Terminal column prompt with legality re-asking | change terminal input handling |
| [agents/random.py](../connect4/agents/random.py) | `RandomAgent` | Uniform random legal move | — (baseline) |
| [agents/rule_based.py](../connect4/agents/rule_based.py) | `RuleBasedAgent` | Rules: win, block, center, center-preference; per-rule counters | change the heuristic baseline |
| [agents/minimax.py](../connect4/agents/minimax.py) | `MinimaxAgent` | Alpha-beta minimax on the live game via make/undo; `_evaluate` scans all 4-cell windows, `_score_window` scores live windows, center-first `_ordered_moves` | tune the heuristic or search |
| [agents/mcts.py](../connect4/agents/mcts.py) | `MCTSAgent`, `MCTSNode` | UCT (`uct_score`, exploration 1.414); tactical win/block override before search; `choose_rollout_move` (win > block > safe > center); `get_ordered_moves` orders expansions tactically; `sync_root_to_game` reuses the subtree between moves; rich `get_stats` | change search, rollout policy, or tree reuse |
| [agents/rl_policy.py](../connect4/agents/rl_policy.py) | `RLPolicyAgent` | Loads a checkpoint (arch inferred from its saved config), tactical win/block override, illegal-move masking, temperature sampling or argmax; `evaluate_position` exposes the value head | change inference-time behavior or checkpoint loading |
| [agents/factory.py](../connect4/agents/factory.py) | `parse_agent_config`, `create_agent`, `resolve_rl_model_path`, `DEFAULT_*` | Turns spec strings (`mcts-700`, `minimax-7`, `rl`, ...) into agent instances; resolves RL checkpoints under `runs/` (default `rl_pure_selfplay_v3` + `best`) | register a new agent spec or change defaults |
| [agents/\_\_init\_\_.py](../connect4/agents/__init__.py) | `__getattr__` | PEP 562 lazy re-exports so importing torch-free agents never imports torch | add a new agent to the package namespace |

### Model and training

| File | Key symbols | What it does | Open it to... |
|---|---|---|---|
| [models/policy_value_network.py](../connect4/models/policy_value_network.py) | `PolicyValueNet`, `PolicyValueNetSmall`, `encode_board`, `mirror_encoded_state`, `mirror_action` | Residual CNN (128-ch stem, 6 blocks; policy head 7 logits, value head tanh); 4-channel current-player-perspective encoding (my pieces / opp pieces / side-to-move / normalized heights); mirror augmentation helpers | change architecture or board encoding |
| [training/vec_engine.py](../connect4/training/vec_engine.py) | `VecConnect4`, `_check_wins_batch`, `_check_win_single` | Batched `(n, 6, 7)` int8 NumPy engine: heights-based O(1) drop rows, masked `step`/`reset`, `encode` produces batches identical to `encode_board` | change the training-time engine (must stay consistent with `engine.py`) |
| [training/trainer.py](../connect4/training/trainer.py) | `Trainer`, `ReplayBuffer`, `FrozenPolicyAgent`, `play_selfplay_vectorized`, `evaluate_against`, `set_seed` | Self-play loop over 512 parallel envs with tactical override (win > block > net); circular NumPy replay buffer (O(1) sampling); mirror augmentation; frozen-checkpoint opponent pool; epsilon/temperature annealing; AdamW + cosine LR (`T_max = episodes // n_envs`, resume re-derives `last_epoch`); periodic eval vs Random/Rule/MCTS drives `best_model.pt` | change the training pipeline, loss, schedules, or checkpointing |

### Evaluation, UI, CLI

| File | Key symbols | What it does | Open it to... |
|---|---|---|---|
| [evaluation/evaluate.py](../connect4/evaluation/evaluate.py) | `evaluate_agents`, `play_one_game`, `EvaluationSummary`, `GameStats`, `print_evaluation_summary` | Plays N games alternating first player, records per-move timing and per-game stats, aggregates internal agent stats | change how head-to-head runs are scored/reported |
| [evaluation/tournament.py](../connect4/evaluation/tournament.py) | `Matchup`, `run_matchups`, `summarize_matchup`, `print_tournament_summary` | Runs a list of matchups, persists JSON after every matchup (interruption-safe), prints ETA and summary tables; used by the `tournament` subcommand | change result persistence or the summary format |
| [ui/game_ui.py](../connect4/ui/game_ui.py) | `GameUI`, `MoveInfo` | pygame board: drop animation, hover preview, undo, per-move timing side panel; `None` agent slots mean human mouse input | change the graphical UI |
| [cli/main.py](../connect4/cli/main.py) | `build_parser`, `main` | Argparse dispatcher; imports only stdlib at module level so `--help` is instant | add a subcommand |
| [cli/game.py](../connect4/cli/game.py) | `configure_*_parser`, `run_play`/`run_ui`/`run_eval`, `_add_agent_args` | `play`/`ui`/`eval` flags and handlers; `_add_agent_args` defines the shared `--agent1/2 --name1/2 --model1/2 --checkpoint1/2 --model-path1/2` surface | change game-facing flags |
| [cli/train.py](../connect4/cli/train.py) | `configure_parser`, `run_train` | The trainer's full hyperparameter surface as grouped argparse flags | see every training flag and default |
| [cli/tournament.py](../connect4/cli/tournament.py) | `build_matchups`, `run_tournament` | Defines the report's 26-matchup round-robin (game counts tiered by agent speed); RL learning-curve rows auto-skip missing checkpoints | change the tournament lineup |

### Data directories

| Path | Contents |
|---|---|
| [runs/rl_pure_selfplay_v3/](../runs/rl_pure_selfplay_v3/) | The shipped model: `config.json` (exact training hyperparameters), `training_log.json` (loss/eval curves), `best_model.pt` (loaded by the `rl` agent spec), `checkpoints/` |
| [results/](../results/) | `tournament_results.json` (26 matchups / 1,528 games) and `mcts_vs_minimax_part1..4*.json` (data from earlier MCTS-vs-minimax scaling sweeps) |
| `tests/` | Unit tests for engine/agents/training/CLI; run with `pytest` from the repo root |

## Where to look when...

| Task | Files |
|---|---|
| Change win detection | [engine.py](../connect4/engine.py) `check_winner`/`_count_direction` **and** [vec_engine.py](../connect4/training/vec_engine.py) `_check_win_single`/`_check_wins_batch` — two implementations that must agree; then the engine tests in `tests/` |
| Add a new agent | Implement `choose_action` per [agents/base.py](../connect4/agents/base.py); register the spec string in [factory.py](../connect4/agents/factory.py) (`parse_agent_config` + `create_agent`); add the lazy export in [agents/\_\_init\_\_.py](../connect4/agents/__init__.py); agent flags already flow through [cli/game.py](../connect4/cli/game.py) `_add_agent_args` |
| Change training | [training/trainer.py](../connect4/training/trainer.py) for behavior, [cli/train.py](../connect4/cli/train.py) for the flag surface and defaults |
| Change board encoding | [policy_value_network.py](../connect4/models/policy_value_network.py) `encode_board` **and** [vec_engine.py](../connect4/training/vec_engine.py) `encode` — training encodes via the vectorized path, inference via `encode_board`; they must stay bit-identical (pinned by tests) |
| Change the network architecture | [policy_value_network.py](../connect4/models/policy_value_network.py); note [rl_policy.py](../connect4/agents/rl_policy.py) `_load_model` reconstructs the architecture from the `config` dict saved inside checkpoints (`channels`/`num_blocks`/`dropout`) |
| Change tactical logic (win/block/center) | [tactics.py](../connect4/tactics.py) (shared by rule-based/MCTS/RL at play time) and [trainer.py](../connect4/training/trainer.py) `_find_tactical_move` (the vectorized training-time equivalent) |
| Tune MCTS strength | [agents/mcts.py](../connect4/agents/mcts.py) — `iterations`/`exploration_weight` ctor args, `choose_rollout_move`, `sync_root_to_game` |
| Tune minimax strength | [agents/minimax.py](../connect4/agents/minimax.py) — `depth` ctor arg, `_evaluate`/`_score_window` weights |
| Change which RL model loads by default | [factory.py](../connect4/agents/factory.py) `DEFAULT_RL_MODEL`/`DEFAULT_RL_CHECKPOINT` (`rl_pure_selfplay_v3` + `best`) |
| Change the tournament lineup | [cli/tournament.py](../connect4/cli/tournament.py) `build_matchups` |
| Change how eval results are computed/printed | [evaluation/evaluate.py](../connect4/evaluation/evaluate.py); JSON row format in [evaluation/tournament.py](../connect4/evaluation/tournament.py) `summarize_matchup` |
| Add or change a CLI flag | [cli/main.py](../connect4/cli/main.py) wires subparsers; each subcommand's flags live in its own `cli/*.py` `configure_*` function |
| Change the pygame UI | [ui/game_ui.py](../connect4/ui/game_ui.py) |

## Conventions worth knowing

- **Import discipline** — `cli/main.py` and every `configure_*` function import
  only stdlib; torch/pygame/agents are imported inside `run_*` handlers, and
  `agents/__init__.py` re-exports lazily. Keep it that way or `python -m connect4 --help`
  gets slow.
- **Two engines, one truth** — `engine.py` is the reference implementation
  (search agents mutate it via make/undo); `vec_engine.py` exists only to step
  hundreds of self-play games per call. Any rule change must land in both.
- **CWD-relative paths** — `runs/` and `results/` resolve against the current
  directory; run all commands from the repo root.
- **Agents never mutate the real game in play** — the game loop passes
  `game.clone()` to `choose_action` ([cli/game.py](../connect4/cli/game.py)),
  so search agents are free to make/undo on the object they receive.
