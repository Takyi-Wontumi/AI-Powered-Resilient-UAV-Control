#!/usr/bin/env python3
"""Train a residual PPO policy for dropout-aware EKF tracking."""

import argparse
import os
import sys
from pathlib import Path

import torch as th

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        StopTrainingOnNoModelImprovement,
    )
    from stable_baselines3.common.utils import get_schedule_fn
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )
except ImportError:
    print("Error: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)

from AI_UAV_Tests.rl.rl_dropout_policy import DroneDropoutRLEnv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train residual PPO for dropout-aware trajectory tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Total training steps (default: 100000)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=2048,
        help="Rollout horizon per PPO update (default: 2048)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments (default: 1)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/dropout_rl/",
        help="Directory for checkpoints, best model, and normalization stats",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["hover", "circle", "full"],
        default="full",
        help="Mission preset to train on (default: full)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate every N environment steps (default: 10000)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Base learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--init-model",
        type=str,
        default=None,
        help="Optional PPO checkpoint to warm-start from for curriculum training",
    )
    parser.add_argument(
        "--reset-timesteps",
        action="store_true",
        help="Reset timestep counter when continuing from --init-model instead of appending",
    )
    parser.add_argument(
        "--decay-lr",
        action="store_true",
        help="Linearly decay learning rate over training instead of keeping it fixed",
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=0.25,
        help="Maximum residual mixing gain (default: 0.25)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.08,
        help="PPO KL target before early stop inside an update (default: 0.08)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.002,
        help="Entropy regularization coefficient (default: 0.002)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during evaluation rollouts",
    )
    return parser.parse_args()


def linear_lr_schedule(initial_value: float):
    def schedule(progress_remaining: float) -> float:
        return float(initial_value) * float(progress_remaining)

    return schedule


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Residual PPO Training: EKF + PID + RL")
    print(f"{'=' * 60}")
    print(f"  Training steps      : {args.steps:,}")
    print(f"  Steps per epoch     : {args.steps_per_epoch:,}")
    print(f"  Parallel envs       : {args.n_envs}")
    print(f"  Task                : {args.task}")
    print(f"  Learning rate       : {args.learning_rate:.2e}")
    print(f"  Init model          : {args.init_model or 'none'}")
    print(f"  Save directory      : {save_dir}")
    print(f"{'=' * 60}\n")

    def make_train_env():
        return DroneDropoutRLEnv(
            task=args.task,
            render_mode=None,
            dropout_randomize=True,
            residual_alpha=args.residual_alpha,
            track_trajectory_during_dropout=True,
        )

    def make_eval_env():
        return DroneDropoutRLEnv(
            task=args.task,
            render_mode=None if args.no_render else "human",
            dropout_randomize=False,
            residual_alpha=args.residual_alpha,
            track_trajectory_during_dropout=True,
        )

    vec_env_cls = DummyVecEnv if args.n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        make_train_env,
        n_envs=args.n_envs,
        seed=0,
        vec_env_cls=vec_env_cls,
    )
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )

    eval_env = make_vec_env(
        make_eval_env,
        n_envs=1,
        seed=123,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = VecNormalize(
        eval_env,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.995,
    )
    eval_env.obs_rms = env.obs_rms

    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=str(save_dir),
        name_prefix="rl_dropout_policy",
        save_replay_buffer=False,
    )
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=8,
        min_evals=5,
        verbose=1,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
    )

    lr = linear_lr_schedule(args.learning_rate) if args.decay_lr else args.learning_rate
    lr_mode = "scheduled" if args.decay_lr else "fixed"

    print("[1/3] Creating PPO policy\n")
    print(f"  Learning rate mode: {lr_mode}\n")

    policy_kwargs = {
        "activation_fn": th.nn.Tanh,
        "net_arch": dict(pi=[256, 256], vf=[256, 256]),
        "ortho_init": False,
    }

    try:
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        alpha = args.residual_alpha
        print(
            f"  Env obs shape: {obs_shape}, action shape: {act_shape}, residual alpha: {alpha}"
        )
    except Exception:
        pass

    if args.init_model:
        print(f"  Loading warm-start checkpoint: {args.init_model}")
        model = PPO.load(args.init_model, env=env, device="auto")
        if model.n_steps != args.steps_per_epoch:
            print(
                "  [Warm-start] Keeping loaded rollout horizon "
                f"n_steps={model.n_steps} (requested {args.steps_per_epoch})."
            )
        model.learning_rate = lr
        model.lr_schedule = get_schedule_fn(lr)
        model.batch_size = 512
        model.n_epochs = 5
        model.gamma = 0.995
        model.gae_lambda = 0.98
        model.clip_range = get_schedule_fn(0.15)
        model.ent_coef = args.ent_coef
        model.vf_coef = 0.5
        model.max_grad_norm = 0.5
        model.target_kl = args.target_kl
        model.tensorboard_log = str(log_dir)
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=lr,
            n_steps=args.steps_per_epoch,
            batch_size=512,
            n_epochs=5,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.15,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=args.target_kl,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
        )

    print(f"\n[2/3] Training for {args.steps:,} steps\n")
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name="dropout_rl_training",
        reset_num_timesteps=args.reset_timesteps or not bool(args.init_model),
    )

    print("\n[3/3] Saving final model\n")
    final_model_path = save_dir / "rl_dropout_policy_final"
    model.save(str(final_model_path))
    env.save(str(save_dir / "vecnormalize.pkl"))
    print(f"  -> {final_model_path}.zip")

    env.close()
    eval_env.close()

    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Best model saved to: {save_dir / 'best_model.zip'}")
    print(f"  Final model saved to: {final_model_path}.zip")
    print(f"  VecNormalize stats: {save_dir / 'vecnormalize.pkl'}")
    print(f"  TensorBoard logs: {log_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
