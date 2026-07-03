"""``train`` subcommand — the self-play trainer's full hyperparameter surface."""

from __future__ import annotations

import argparse


def configure_parser(parser: argparse.ArgumentParser) -> None:
    run = parser.add_argument_group("run")
    run.add_argument("--episodes",   type=int, default=200000)
    run.add_argument("--run-name",   type=str, default="rl_policy")
    run.add_argument("--output-dir", type=str, default="runs")
    run.add_argument("--resume",     type=str, default=None)
    run.add_argument("--seed",       type=int, default=42)
    run.add_argument("--n-envs",     type=int, default=512)

    opt = parser.add_argument_group("optimization")
    opt.add_argument("--lr",           type=float, default=3e-4)
    opt.add_argument("--min-lr",       type=float, default=1e-5)
    opt.add_argument("--weight-decay", type=float, default=1e-4)
    opt.add_argument("--grad-clip",    type=float, default=1.0)
    opt.add_argument("--batch-size",          type=int, default=1024)
    opt.add_argument("--buffer-size",         type=int, default=1000000)
    opt.add_argument("--updates-per-episode", type=int, default=2)
    opt.add_argument("--updates-per-batch",   type=int, default=256,
                     help="If >0, do exactly this many gradient steps per outer loop.")

    net = parser.add_argument_group("network")
    net.add_argument("--channels",      type=int,   default=128)
    net.add_argument("--num-blocks",    type=int,   default=6)
    net.add_argument("--dropout",       type=float, default=0.10)
    net.add_argument("--small-network", action="store_true")

    explore = parser.add_argument_group("exploration schedule")
    explore.add_argument("--epsilon-start",              type=float, default=0.3)
    explore.add_argument("--epsilon-end",                type=float, default=0.05)
    explore.add_argument("--epsilon-decay-episodes",     type=int,   default=600000)
    explore.add_argument("--temperature-start",          type=float, default=2.0)
    explore.add_argument("--temperature-end",            type=float, default=0.3)
    explore.add_argument("--temperature-decay-episodes", type=int,   default=800000)

    loss = parser.add_argument_group("loss weights")
    loss.add_argument("--policy-weight",  type=float, default=1.0)
    loss.add_argument("--value-weight",   type=float, default=1.0)
    loss.add_argument("--entropy-weight", type=float, default=0.05)
    loss.add_argument("--augment-mirror", action="store_true", default=True)

    ckpt = parser.add_argument_group("checkpointing & eval")
    ckpt.add_argument("--snapshot-interval",    type=int, default=10000)
    ckpt.add_argument("--max-checkpoint-pool",  type=int, default=8)
    ckpt.add_argument("--eval-interval",        type=int, default=25000)
    ckpt.add_argument("--eval-games",           type=int, default=100)
    ckpt.add_argument("--eval-games-small",     type=int, default=20)
    ckpt.add_argument("--save-interval",        type=int, default=25000)
    ckpt.add_argument("--log-interval",         type=int, default=2048)
    ckpt.add_argument("--mcts-eval-iterations", type=int, default=200,
                      help="MCTS iterations for evaluation opponent only.")
    ckpt.add_argument("--eval-debug", action="store_true", default=False,
                      help="Print per-move debug output during eval games")

    parser.set_defaults(func=run_train)


def run_train(args: argparse.Namespace) -> None:
    # Imported here so `connect4 --help` never pays the torch import.
    from connect4.training.trainer import Trainer, set_seed

    set_seed(args.seed)
    Trainer(args).run()
