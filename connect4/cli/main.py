"""``python -m connect4`` command-line entry point.

This module (and every ``configure_*`` function it calls) imports only the
standard library, so ``python -m connect4 --help`` stays instant. Heavy imports
(torch, pygame, the agents) happen inside each subcommand's ``run`` handler.
"""

from __future__ import annotations

import argparse

from connect4.cli import game, tournament, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="connect4",
        description="Connect 4 AI engine: play, evaluate, train, and run tournaments.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    game.configure_play_parser(
        sub.add_parser("play", help="play a game in the terminal"))
    game.configure_ui_parser(
        sub.add_parser("ui", help='play a game in the graphical UI (needs pygame)'))
    game.configure_eval_parser(
        sub.add_parser("eval", help="benchmark two agents head-to-head"))
    train.configure_parser(
        sub.add_parser("train", help="train the RL agent via self-play"))
    tournament.configure_parser(
        sub.add_parser("tournament", help="run the full round-robin tournament"))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
