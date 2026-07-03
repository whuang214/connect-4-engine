"""Tests for agent-spec parsing, the agent factory, and the CLI parser."""

from __future__ import annotations

import pytest

from connect4.agents.factory import (
    create_agent,
    parse_agent_config,
    resolve_rl_model_path,
)
from connect4.cli.main import build_parser


class TestParseAgentConfig:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("mcts-700", ("mcts", 700)),
            ("minimax-7", ("minimax", 7)),
            (" MCTS-700 ", ("mcts", 700)),  # normalized: lowercased + stripped
        ],
    )
    def test_suffixed_specs_carry_their_own_strength(self, spec, expected):
        assert parse_agent_config(spec, iterations=999) == expected

    def test_bare_names_use_per_type_defaults(self):
        assert parse_agent_config("mcts") == ("mcts", 500)
        assert parse_agent_config("minimax") == ("minimax", 5)

    def test_bare_names_honor_explicit_iterations(self):
        assert parse_agent_config("mcts", iterations=800) == ("mcts", 800)
        assert parse_agent_config("minimax", iterations=7) == ("minimax", 7)

    @pytest.mark.parametrize("spec", ["mcts-abc", "minimax-", "mcts-", "minimax-x7"])
    def test_invalid_specs_raise(self, spec):
        with pytest.raises(ValueError):
            parse_agent_config(spec, iterations=999)

    def test_unknown_type_passes_through_with_iterations(self):
        assert parse_agent_config("human", iterations=123) == ("human", 123)


class TestCreateAgent:
    def test_unknown_agent_type_raises(self):
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_agent("alphago")

    def test_error_message_lists_only_supported_agents(self):
        with pytest.raises(ValueError) as excinfo:
            create_agent("nonsense")
        # Regression: removed agent types must not resurface in the help text.
        assert "hybrid" not in str(excinfo.value).lower()

    def test_creates_parametrized_agents_from_specs(self):
        assert create_agent("minimax-7").depth == 7
        assert create_agent("mcts-700").iterations == 700
        assert create_agent("random").name == "RandomAgent"


class TestResolveRlModelPath:
    def test_bad_checkpoint_name_raises_value_error(self):
        with pytest.raises(ValueError, match="checkpoint"):
            resolve_rl_model_path(checkpoint="latest")

    def test_missing_run_folder_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            resolve_rl_model_path(model_name="no_such_run_anywhere")

    def test_missing_explicit_model_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_rl_model_path(model_path=str(tmp_path / "missing.pt"))

    def test_existing_explicit_model_path_returned_as_is(self, tmp_path):
        model_file = tmp_path / "model.pt"
        model_file.write_bytes(b"weights")
        assert resolve_rl_model_path(model_path=str(model_file)) == str(model_file)


class TestBuildParser:
    def test_play_subcommand(self):
        args = build_parser().parse_args(
            ["play", "--agent1", "random", "--agent2", "rule", "--no-render"]
        )
        assert args.mode == "play"
        assert args.agent1 == "random"
        assert args.no_render is True

    def test_ui_subcommand(self):
        args = build_parser().parse_args(["ui", "--agent2", "minimax-7"])
        assert args.mode == "ui"
        assert args.agent1 == "human"
        assert args.agent2 == "minimax-7"

    def test_eval_subcommand_and_defaults(self):
        args = build_parser().parse_args(
            ["eval", "--agent1", "mcts-700", "--agent2", "minimax-7",
             "--games", "5", "--no-print-each-game", "--print-moves"]
        )
        assert args.mode == "eval"
        assert args.games == 5
        assert args.no_print_each_game is True
        assert args.print_moves is True
        # Regression: RL checkpoint selection must default to 'best'.
        assert args.checkpoint1 == "best"
        assert args.checkpoint2 == "best"

    def test_train_subcommand(self):
        args = build_parser().parse_args(
            ["train", "--episodes", "1000", "--run-name", "smoke", "--n-envs", "8"]
        )
        assert args.mode == "train"
        assert args.episodes == 1000
        assert args.run_name == "smoke"
        assert args.n_envs == 8

    def test_tournament_subcommand(self):
        args = build_parser().parse_args(["tournament", "--quick", "--skip-slow"])
        assert args.mode == "tournament"
        assert args.quick is True
        assert args.skip_slow is True

    def test_experiment_subcommand(self):
        args = build_parser().parse_args(["experiment", "--part", "3"])
        assert args.mode == "experiment"
        assert args.part == "3"

    def test_every_subcommand_sets_a_run_handler(self):
        parser = build_parser()
        samples = [
            ["play"], ["ui"], ["eval"],
            ["train"], ["tournament"], ["experiment", "--part", "1"],
        ]
        for argv in samples:
            args = parser.parse_args(argv)
            assert callable(args.func), argv
