#!/usr/bin/env python3
"""
Train RL Policy with LSTM for Dropout Handling
===============================================

Same as train_dropout_rl_policy.py but uses LSTM layers to enable:
- Temporal memory across dropout periods
- Dead reckoning by remembering velocity trends
- Smoother state transitions during GPS re-acquisition

LSTM lets the policy "remember" the trajectory it was on and maintain
that intent during blind periods, like a pilot flying through clouds.

Usage:
    python examples/train_dropout_rl_lstm.py \\
        --steps 100000 \\
        --lstm-layers 2 \\
        --lstm-hidden-size 256
"""

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
    )
except ImportError:
    print("Error: stable-baselines3 not installed.")
    sys.exit(1)

from AI_UAV_Tests.rl_dropout_policy import DroneDropoutRLEnv


def parse_args():
    p = argparse.ArgumentParser(
        description="Train RL policy with LSTM for dropout handling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps (default: 100000)",
    )
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=2048,
        help="Steps per epoch (default: 2048)",
    )
    p.add_argument(
        "--lstm-layers",
        type=int,
        default=2,
        help="Number of LSTM layers (default: 2)",
    )
    p.add_argument(
        "--lstm-hidden-size",
        type=int,
        default=256,
        help="LSTM hidden state size (default: 256)",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="models/dropout_rl_lstm/",
        help="Directory to save model checkpoints",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate every N steps (default: 10000)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  RL Policy Training with LSTM: GPS Dropout Handling")
    print(f"{'=' * 70}")
    print(f"  Training steps      : {args.steps:,}")
    print(f"  LSTM layers         : {args.lstm_layers}")
    print(f"  LSTM hidden size    : {args.lstm_hidden_size}")
    print(f"  Learning rate       : {args.learning_rate:.2e}")
    print(f"  Save directory      : {save_dir}")
    print(f"\n  Key Capability: LSTM enables dead reckoning during GPS dropout")
    print(f"  Memory horizon     : {args.steps_per_epoch * 0.002:.1f}s (can vary)")
    print(f"{'=' * 70}\n")

    # Environment
    env = DroneDropoutRLEnv(
        render_mode=None,
        dropout_randomize=True,
    )

    eval_env = DroneDropoutRLEnv(
        render_mode=None,
        dropout_randomize=False,
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=save_dir,
        name_prefix="rl_dropout_lstm",
        save_replay_buffer=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=3,
        deterministic=False,  # LSTM needs stochastic evaluation
        render=False,
    )

    # PPO with LSTM policy
    print("[1/3]  Creating PPO with LSTM policy...\n")

    policy_kwargs = {
        "net_arch": [256, 256],  # Shared network before LSTM
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.lstm_layers,
        "shared_lstm": False,  # Separate LSTM for policy and value
    }

    model = PPO(
        "MlpLstmPolicy",  # LSTM policy instead of feedforward MLP
        env,
        learning_rate=args.learning_rate,
        n_steps=args.steps_per_epoch,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
    )

    print(f"\n[2/3]  Training for {args.steps:,} steps...\n")
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="dropout_rl_lstm_training",
    )

    # Save
    print(f"\n[3/3]  Saving final model...\n")
    final_path = save_dir / "rl_dropout_lstm_final"
    model.save(final_path)
    print(f"  → {final_path}.zip")

    env.close()
    eval_env.close()

    print(f"\n{'=' * 70}")
    print(f"  Training Complete!")
    print(f"  Best model   : {save_dir / 'best_model.zip'}")
    print(f"  Final model  : {final_path}.zip")
    print(f"  TensorBoard  : tensorboard --logdir {log_dir}")
    print(f"{'=' * 70}\n")
    print(f"  Next: Evaluate with evaluate_dropout_rl_policy.py")
    print(f"  $ python examples/evaluate_dropout_rl_policy.py \\")
    print(f"      --model {final_path}.zip \\")
    print(f"      --n-trials 10\n")


if __name__ == "__main__":
    main()
