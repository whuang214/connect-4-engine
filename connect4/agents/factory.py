"""Build agents from CLI-style specifiers such as ``mcts-700`` or ``minimax-7``."""

from __future__ import annotations

import os

from connect4.agents.human import HumanAgent
from connect4.agents.mcts import MCTSAgent
from connect4.agents.minimax import MinimaxAgent
from connect4.agents.random import RandomAgent
from connect4.agents.rule_based import RuleBasedAgent

DEFAULT_RL_MODEL = "rl_pure_selfplay_v3"
DEFAULT_RL_CHECKPOINT = "best"
DEFAULT_MCTS_ITERATIONS = 500
DEFAULT_MINIMAX_DEPTH = 5


def parse_agent_config(agent_type: str, iterations: int | None = None) -> tuple[str, int]:
    """Resolve an agent spec to (type, strength).

    Suffixed specs (``mcts-700``, ``minimax-7``) carry their own strength;
    bare ``mcts``/``minimax`` use ``iterations`` when given, else their
    per-type default (500 iterations / depth 5).
    """
    agent_type = agent_type.lower().strip()

    if agent_type.startswith("mcts-"):
        _, value = agent_type.split("-", 1)
        if not value.isdigit():
            raise ValueError(
                f"Invalid MCTS agent format: '{agent_type}'. "
                f"Expected format like 'mcts-500'."
            )
        return "mcts", int(value)

    if agent_type.startswith("minimax-"):
        _, value = agent_type.split("-", 1)
        if not value.isdigit():
            raise ValueError(
                f"Invalid minimax agent format: '{agent_type}'. "
                f"Expected format like 'minimax-5'."
            )
        return "minimax", int(value)

    # Bare agent names: explicit --iterations wins, else the per-type default
    if agent_type == "mcts":
        return "mcts", iterations if iterations is not None else DEFAULT_MCTS_ITERATIONS
    if agent_type == "minimax":
        return "minimax", iterations if iterations is not None else DEFAULT_MINIMAX_DEPTH

    return agent_type, iterations if iterations is not None else DEFAULT_MCTS_ITERATIONS


def resolve_rl_model_path(
    model_name: str | None = None,
    checkpoint: str = DEFAULT_RL_CHECKPOINT,
    model_path: str | None = None,
) -> str:
    if model_path is not None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RL model file not found: '{model_path}'."
            )
        return model_path

    checkpoint = checkpoint.lower().strip()
    if checkpoint not in {"best", "final"}:
        raise ValueError(
            f"Invalid checkpoint '{checkpoint}'. Use 'best' or 'final'."
        )

    model_name = model_name or DEFAULT_RL_MODEL
    resolved_path = os.path.join("runs", model_name, f"{checkpoint}_model.pt")

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"RL model file not found: '{resolved_path}'. "
            f"Use --model1/--model2 with a valid run folder name inside 'runs/', "
            f"or pass --model-path1/--model-path2 with a full .pt path."
        )

    return resolved_path


def create_agent(
    agent_type: str,
    name: str | None = None,
    iterations: int | None = None,
    model_name: str | None = None,
    checkpoint: str = DEFAULT_RL_CHECKPOINT,
    model_path: str | None = None,
):
    agent_type, iterations = parse_agent_config(agent_type, iterations)

    if agent_type == "human":
        return HumanAgent(name=name) if name else HumanAgent()

    if agent_type == "random":
        return RandomAgent(name=name) if name else RandomAgent()

    if agent_type == "rule":
        return RuleBasedAgent(name=name) if name else RuleBasedAgent()

    if agent_type == "mcts":
        return (
            MCTSAgent(name=name, iterations=iterations)
            if name
            else MCTSAgent(iterations=iterations)
        )

    if agent_type == "minimax":
        return MinimaxAgent(depth=iterations, name=name)

    if agent_type == "rl":
        # Imported here so torch-free agents never pay the torch import.
        from connect4.agents.rl_policy import RLPolicyAgent

        resolved_model_path = resolve_rl_model_path(
            model_name=model_name,
            checkpoint=checkpoint,
            model_path=model_path,
        )

        run_folder = os.path.basename(os.path.dirname(resolved_model_path))
        run_folder_clean = run_folder.removeprefix("rl_")
        default_name = f"RL-{run_folder_clean}-{checkpoint}"

        return RLPolicyAgent(
            name=name or default_name,
            model_path=resolved_model_path,
        )

    raise ValueError(
        f"Unknown agent type: '{agent_type}'. "
        f"Use: human, random, rule, mcts, mcts-<iterations>, "
        f"minimax, minimax-<depth>, rl"
    )
