#!/usr/bin/env python3
"""
PPO training -- residual correction policy on the square-trajectory env.
========================================================================
Trains an MLP policy that emits per-axis confidence multipliers
[alpha_x, alpha_y] in [0, 2] over the frozen LSTM drift prediction.

Goal: drive the mean dropout XY position error down toward ~5 cm.

State estimator: EKF + frozen LSTM (loaded by the env).
Action: per-axis alpha in [0, 2].
Observation: 42-dim vector built inside UAVSquarePrecompEnv._get_obs().

Outputs:
  experiments/rl_precomp_policy_square.zip
  experiments/best_model_square.zip            (best by eval mean reward)
  experiments/rl_logs_square/                  (Tensorboard)
  experiments/rl_checkpoints_square/           (periodic snapshots)
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pathlib import Path
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from uav_rl_env_square import UAVSquarePrecompEnv, TARGET_ERR_M


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on the square-trajectory env.")
    p.add_argument("--total-timesteps", type=int, default=1_500_000,
                   help="Total environment steps. 1.5M is a good first pass; bump to 3M+ for full convergence.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel envs. Each spins up its own PyBullet -- 1-2 is usually all a CPU can handle.")
    p.add_argument("--checkpoint-freq", type=int, default=50_000,
                   help="Steps between checkpoint snapshots.")
    p.add_argument("--eval-freq", type=int, default=25_000,
                   help="Steps between eval passes (also when dropout-error metric is reported).")
    p.add_argument("--n-eval-episodes", type=int, default=4,
                   help="Episodes per eval pass.")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005,
                   help="Small entropy bonus keeps exploration alive while alpha shrinks toward optimum.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume-from", type=str, default=None,
                   help="Path to a .zip to warm-start from (e.g., a previous policy).")
    return p.parse_args()


# ---- Dropout-error tracker ---------------------------------------------------
class DropoutErrorCallback(BaseCallback):
    """
    Walks the eval env once every `eval_freq` steps, plays a deterministic
    rollout, and logs the mean XY position error during dropout.
    Lets you watch progress toward the 5 cm target without staring at MSE.
    """
    def __init__(self, eval_env, eval_freq: int, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = max(int(eval_freq), 1)

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or self.num_timesteps % self.eval_freq != 0:
            return True
        try:
            errs = []
            for _ in range(2):
                obs = self.eval_env.reset()
                done = np.array([False])
                while not done[0]:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, info = self.eval_env.step(action)
                    inf = info[0] if isinstance(info, (list, tuple)) else info
                    if not bool(inf.get('gps_ok', True)) and inf.get('ekf_error') is not None:
                        errs.append(float(inf['ekf_error']))
            if errs:
                arr = np.asarray(errs, dtype=float)
                mean_cm  = float(arr.mean()) * 100.0
                p95_cm   = float(np.percentile(arr, 95)) * 100.0
                hit_rate = float((arr < TARGET_ERR_M).mean())
                self.logger.record('dropout/mean_err_cm', mean_cm)
                self.logger.record('dropout/p95_err_cm',  p95_cm)
                self.logger.record('dropout/under_5cm',   hit_rate)
                if self.verbose:
                    print(f"  [dropout-eval @ {self.num_timesteps:,}] "
                          f"mean={mean_cm:.2f} cm  p95={p95_cm:.2f} cm  "
                          f"<5cm fraction={hit_rate*100:.1f}%")
        except Exception as e:
            if self.verbose:
                print(f"  [dropout-eval] skipped: {e}")
        return True


def main():
    args = parse_args()

    SAVE_DIR         = Path('experiments')
    LOG_DIR          = Path('experiments/rl_logs_square')
    POLICY_SAVE_PATH = SAVE_DIR / 'rl_precomp_policy_square'
    CHECKPOINT_DIR   = SAVE_DIR / 'rl_checkpoints_square'

    SAVE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 76)
    print("PPO TRAINING -- square trajectory + frozen LSTM + EKF")
    print("=" * 76)
    print(f"  Total timesteps  : {args.total_timesteps:,}")
    print(f"  Parallel envs    : {args.n_envs}")
    print(f"  Eval every       : {args.eval_freq:,} steps  ({args.n_eval_episodes} ep)")
    print(f"  Checkpoint every : {args.checkpoint_freq:,} steps")
    print(f"  Best model       : {SAVE_DIR / 'best_model_square.zip'}")
    print(f"  Final policy     : {POLICY_SAVE_PATH}.zip")
    print(f"  Tensorboard      : tensorboard --logdir {LOG_DIR}")
    print(f"  Goal             : mean dropout XY error -> {TARGET_ERR_M*100:.0f} cm\n")

    train_env = DummyVecEnv([
        (lambda: UAVSquarePrecompEnv(render_mode=None, lstm_device='cpu'))
        for _ in range(args.n_envs)
    ])
    eval_env = DummyVecEnv([
        lambda: UAVSquarePrecompEnv(render_mode=None, lstm_device='cpu')
    ])

    if args.resume_from is not None and Path(args.resume_from).exists():
        print(f"  Resuming from: {args.resume_from}\n")
        model = PPO.load(
            args.resume_from,
            env             = train_env,
            tensorboard_log = str(LOG_DIR),
            device          = 'cpu',
        )
    else:
        model = PPO(
            policy          = 'MlpPolicy',
            env             = train_env,
            n_steps         = args.n_steps,
            batch_size      = args.batch_size,
            n_epochs        = args.n_epochs,
            gamma           = args.gamma,
            gae_lambda      = args.gae_lambda,
            clip_range      = args.clip_range,
            ent_coef        = args.ent_coef,
            learning_rate   = args.learning_rate,
            seed            = args.seed,
            verbose         = 1,
            device          = 'cpu',
            tensorboard_log = str(LOG_DIR),
            policy_kwargs   = dict(net_arch=[128, 128]),
        )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Policy: MLP [128, 128]  ({n_params:,} parameters)")
    print(f"Observation: {model.observation_space.shape[0]}-dim")
    print(f"Action:      {model.action_space.shape[0]}-dim ([alpha_x, alpha_y] in [0.0, 2.0])\n")

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(args.checkpoint_freq // args.n_envs, 1),
        save_path   = str(CHECKPOINT_DIR),
        name_prefix = 'rl_precomp_square',
        verbose     = 1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(SAVE_DIR),
        log_path             = str(LOG_DIR),
        eval_freq            = max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes      = args.n_eval_episodes,
        deterministic        = True,
        verbose              = 1,
    )
    dropout_cb = DropoutErrorCallback(
        eval_env  = eval_env,
        eval_freq = max(args.eval_freq // args.n_envs, 1),
        verbose   = 1,
    )

    print("Starting training -- press Ctrl+C any time to save and exit.\n")
    try:
        model.learn(
            total_timesteps = args.total_timesteps,
            callback        = [checkpoint_cb, eval_cb, dropout_cb],
        )
        print("\n[OK] Training complete.")
    except KeyboardInterrupt:
        print("\n[!] Training interrupted -- saving current model.")

    # EvalCallback writes best model to SAVE_DIR/best_model.zip; rename to disambiguate
    default_best = SAVE_DIR / 'best_model.zip'
    target_best  = SAVE_DIR / 'best_model_square.zip'
    if default_best.exists():
        try:
            if target_best.exists():
                target_best.unlink()
            default_best.rename(target_best)
            print(f"[OK] Best eval model -> {target_best}")
        except Exception as e:
            print(f"[!] Could not rename best_model.zip: {e}")

    model.save(str(POLICY_SAVE_PATH))
    print(f"[OK] Final policy   -> {POLICY_SAVE_PATH}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
