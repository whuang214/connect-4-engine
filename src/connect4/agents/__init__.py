"""Agent implementations.

Re-exports are lazy (PEP 562) so that importing this package — or any
torch-free agent like MCTS or minimax — never pays the torch import that
``RLPolicyAgent`` requires.
"""

from importlib import import_module

_EXPORTS = {
    "BaseAgent": "connect4.agents.base",
    "HumanAgent": "connect4.agents.human",
    "MCTSAgent": "connect4.agents.mcts",
    "MinimaxAgent": "connect4.agents.minimax",
    "RandomAgent": "connect4.agents.random",
    "RLPolicyAgent": "connect4.agents.rl_policy",
    "RuleBasedAgent": "connect4.agents.rule_based",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
