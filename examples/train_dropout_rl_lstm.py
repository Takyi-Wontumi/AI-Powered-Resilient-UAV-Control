#!/usr/bin/env python3
"""Train a recurrent residual PPO policy for dropout-aware EKF tracking."""

import argparse
import os
import sys
from pathlib import Path

import torch as th
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecNormalize,
    )
except ImportError:
    print("Error: sb3-contrib not installed.")
    print("Install with: pip install sb3-contrib stable-baselines3")
    sys.exit(1)

from AI_UAV_Tests.rl_dropout_policy import DroneDropoutRLEnv


class RecurrentEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        best_model_save_path: str,
        max_no_improvement_evals: int = 8,
        min_evals: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.best_model_save_path = Path(best_model_save_path)
        self.best_model_save_path.mkdir(parents=True, exist_ok=True)
        self.max_no_improvement_evals = int(max_no_improvement_evals)
        self.min_evals = int(min_evals)
        self.best_mean_reward = -float("inf")
        self.no_improvement_evals = 0
        self.eval_count = 0

    def _evaluate_policy(self) -> tuple[float, float]:
        episode_rewards = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            lstm_state = None
            episode_start = np.ones((self.eval_env.num_envs,), dtype=bool)
            done = np.array([False])
            ep_reward = 0.0

            while not bool(done[0]):
                action, lstm_state = self.model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
                obs, reward, done, _info = self.eval_env.step(action)
                ep_reward += float(reward[0])
                episode_start = done

            episode_rewards.append(ep_reward)

        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        return mean_reward, std_reward

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        self.eval_count += 1
        mean_reward, std_reward = self._evaluate_policy()
        if self.verbose:
            print(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )

        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/std_reward", std_reward)
        self.logger.record("eval/num_timesteps", self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.no_improvement_evals = 0
            self.model.save(str(self.best_model_save_path / "best_model"))
            if self.verbose:
                print("New best mean reward!")
        else:
            self.no_improvement_evals += 1

        if (
            self.eval_count >= self.min_evals
            and self.no_improvement_evals > self.max_no_improvement_evals
        ):
            if self.verbose:
                print(
                    "Stopping training because there was no new best model "
                    f"in the last {self.no_improvement_evals} evaluations"
                )
            return False

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train recurrent residual PPO for dropout-aware trajectory tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--steps-per-epoch", type=int, default=1024)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="models/dropout_rl_lstm/")
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--residual-alpha", type=float, default=0.25)
    parser.add_argument("--target-kl", type=float, default=0.06)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 72}")
    print("  Recurrent Residual PPO Training: EKF + PID + LSTM PPO")
    print(f"{'=' * 72}")
    print(f"  Training steps      : {args.steps:,}")
    print(f"  Steps per epoch     : {args.steps_per_epoch:,}")
    print(f"  Parallel envs       : {args.n_envs}")
    print(f"  LSTM hidden size    : {args.lstm_hidden_size}")
    print(f"  LSTM layers         : {args.lstm_layers}")
    print(f"  Learning rate       : {args.learning_rate:.2e}")
    print(f"  Save directory      : {save_dir}")
    print(f"{'=' * 72}\n")

    def make_train_env():
        return DroneDropoutRLEnv(
            render_mode=None,
            dropout_randomize=True,
            residual_alpha=args.residual_alpha,
            track_trajectory_during_dropout=True,
        )

    def make_eval_env():
        return DroneDropoutRLEnv(
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
        name_prefix="rl_dropout_lstm",
        save_replay_buffer=False,
    )
    eval_callback = RecurrentEvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=3,
        max_no_improvement_evals=8,
        min_evals=5,
        verbose=1,
    )

    print("[1/3] Creating recurrent PPO policy\n")
    policy_kwargs = {
        "activation_fn": th.nn.Tanh,
        "net_arch": dict(pi=[256], vf=[256]),
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.lstm_layers,
        "shared_lstm": False,
        "enable_critic_lstm": True,
        "ortho_init": False,
    }

    print(
        "  Env obs shape:"
        f" {env.observation_space.shape}, action shape: {env.action_space.shape},"
        f" residual alpha: {args.residual_alpha}"
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.steps_per_epoch,
        batch_size=256,
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
        tb_log_name="dropout_rl_lstm_training",
    )

    print("\n[3/3] Saving final model\n")
    final_path = save_dir / "rl_dropout_lstm_final"
    model.save(str(final_path))
    env.save(str(save_dir / "vecnormalize.pkl"))
    print(f"  -> {final_path}.zip")

    env.close()
    eval_env.close()

    print(f"\n{'=' * 72}")
    print("  Training complete!")
    print(f"  Best model saved to: {save_dir / 'best_model.zip'}")
    print(f"  Final model saved to: {final_path}.zip")
    print(f"  VecNormalize stats: {save_dir / 'vecnormalize.pkl'}")
    print(f"  TensorBoard logs: {log_dir}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
