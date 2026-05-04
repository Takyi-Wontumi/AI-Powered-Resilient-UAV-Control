#!/usr/bin/env python3
"""
Train RL Policy for Dropout Handling
=====================================

Trains a PPO policy to handle GPS dropout by observing:
- EKF state estimates + covariance (uncertainty)
- Dropout status flag

Domain randomization of dropout timing/duration forces robustness.

Usage:
    python examples/train_dropout_rl_policy.py [--steps 100000] [--save-dir ./models/]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        StopTrainingOnNoModelImprovement,
    )
    from stable_baselines3.common.env_util import make_vec_env
except ImportError:
    print("Error: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

from AI_UAV_Tests.rl_dropout_policy import DroneDropoutRLEnv


def parse_args():
    p = argparse.ArgumentParser(
        description="Train RL policy for dropout handling.",
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
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1, more = faster but more memory)",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="models/dropout_rl/",
        help="Directory to save model checkpoints",
    )
    p.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate policy every N steps (default: 10000)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    p.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during evaluation",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  RL Policy Training: GPS Dropout Handling")
    print(f"{'=' * 60}")
    print(f"  Training steps      : {args.steps:,}")
    print(f"  Steps per epoch     : {args.steps_per_epoch:,}")
    print(f"  Parallel envs       : {args.n_envs}")
    print(f"  Learning rate       : {args.learning_rate:.2e}")
    print(f"  Save directory      : {save_dir}")
    print(f"{'=' * 60}\n")

    # Create environment
    def make_env():
        def _init():
            return DroneDropoutRLEnv(
                render_mode=None if args.no_render else "human",
                dropout_randomize=True,
            )
        return _init

    if args.n_envs == 1:
        env = DroneDropoutRLEnv(render_mode=None, dropout_randomize=True)
    else:
        # Vectorized environments for parallel training
        env = make_vec_env(
            make_env(),
            n_envs=args.n_envs,
            seed=0,
        )

    # Callbacks (simplified to avoid nesting issues)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=save_dir,
        name_prefix="rl_dropout_policy",
        save_replay_buffer=False,
    )

    # Learning rate schedule: only decay for long training runs
    def linear_lr_schedule(initial_value):
        def func(progress_fraction):
            return initial_value * (1 - progress_fraction)
        return func

    # For short runs (< 100k steps), use fixed LR to prevent decay killing learning
    # For long runs (>= 100k steps), use schedule to prevent overfitting
    if args.steps > 100000:
        lr = linear_lr_schedule(args.learning_rate)
        lr_mode = "scheduled (decays over training)"
    else:
        lr = args.learning_rate
        lr_mode = "fixed"

    # Create PPO policy
    print("[1/3]  Creating PPO policy…\n")
    print(f"  Learning rate mode: {lr_mode}\n")
    policy_kwargs = {
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    }

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=args.steps_per_epoch,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        max_grad_norm=0.1,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
    )

    # Train
    print(f"\n[2/3]  Training for {args.steps:,} steps…\n")
    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_callback,
        tb_log_name="dropout_rl_training",
    )

    # Save final model
    print(f"\n[3/3]  Saving final model…\n")
    final_model_path = save_dir / "rl_dropout_policy_final"
    model.save(final_model_path)
    print(f"  → {final_model_path}.zip")

    env.close()

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best model saved to: {save_dir / 'best_model.zip'}")
    print(f"  Final model saved to: {final_model_path}.zip")
    print(f"  TensorBoard logs: {log_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
