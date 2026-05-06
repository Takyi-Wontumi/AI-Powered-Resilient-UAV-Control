#!/usr/bin/env python3
"""Evaluate residual PPO against the zero-residual EKF + PID baseline."""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("Error: stable-baselines3 not installed.")
    sys.exit(1)

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from AI_UAV_Tests.rl_dropout_policy import DroneDropoutRLEnv


@dataclass
class TrialMetrics:
    trial_id: int
    policy_name: str
    rmse_during_dropout: float = 0.0
    max_error_during_dropout: float = 0.0
    avg_dropout_error: float = 0.0
    crashed: bool = False
    success: bool = False
    log: list = field(default_factory=list)


def make_env(
    render_mode: str | None,
    dropout_randomize: bool = True,
    residual_alpha: float = 0.25,
    task: str = "full",
) -> DroneDropoutRLEnv:
    return DroneDropoutRLEnv(
        task=task,
        render_mode=render_mode,
        dropout_randomize=dropout_randomize,
        residual_alpha=residual_alpha,
        track_trajectory_during_dropout=True,
    )


def evaluate_baseline(env: DroneDropoutRLEnv, n_episodes: int = 5) -> list:
    results = []

    for trial_id in range(n_episodes):
        print(f"  Baseline trial {trial_id + 1}/{n_episodes}... ", end="", flush=True)
        state, _ = env.reset()
        metrics = TrialMetrics(trial_id=trial_id, policy_name="Baseline")

        while True:
            action = np.zeros(4, dtype=np.float32)
            state, reward, done, truncated, info = env.step(action)

            dropout_active = bool(info["dropout_active"])
            pos_error = float(info["pos_error"])
            if dropout_active and pos_error > 0.0:
                metrics.max_error_during_dropout = max(
                    metrics.max_error_during_dropout, pos_error
                )

            metrics.log.append(
                {
                    "dropout_active": dropout_active,
                    "pos_error": pos_error,
                    "reward": float(reward),
                }
            )

            if done or truncated:
                break

        dropout_errors = [
            item["pos_error"] for item in metrics.log if item["dropout_active"]
        ]
        if dropout_errors:
            metrics.rmse_during_dropout = float(
                np.sqrt(np.mean(np.square(dropout_errors)))
            )
            metrics.avg_dropout_error = float(np.mean(dropout_errors))
        metrics.crashed = bool(info.get("crashed", False))
        metrics.success = bool(info.get("success", False))
        status = "CRASHED" if metrics.crashed else "OK"
        print(f"RMSE={metrics.rmse_during_dropout:.3f}m ({status})")
        results.append(metrics)

    return results


def evaluate_rl(
    render_mode: str | None,
    model_path: str,
    n_episodes: int = 5,
    residual_alpha: float = 0.25,
    task: str = "full",
) -> list:
    print(f"\n  Loading model from {model_path}...")
    model = None
    is_recurrent = False
    load_errors = []

    if RecurrentPPO is not None:
        try:
            model = RecurrentPPO.load(model_path)
            is_recurrent = True
        except Exception as exc:
            load_errors.append(f"RecurrentPPO: {exc}")

    if model is None:
        try:
            model = PPO.load(model_path)
        except Exception as exc:
            load_errors.append(f"PPO: {exc}")
            raise RuntimeError(
                "Failed to load model as RecurrentPPO or PPO:\n" + "\n".join(load_errors)
            ) from exc

    vecnormalize_path = Path(model_path).with_name("vecnormalize.pkl")

    results = []
    for trial_id in range(n_episodes):
        print(f"  RL trial {trial_id + 1}/{n_episodes}... ", end="", flush=True)
        raw_env = make_env(
            render_mode,
            dropout_randomize=True,
            residual_alpha=residual_alpha,
            task=task,
        )
        vec_env = DummyVecEnv([lambda env=raw_env: env])
        if vecnormalize_path.exists():
            vec_env = VecNormalize.load(str(vecnormalize_path), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        state = vec_env.reset()
        metrics = TrialMetrics(trial_id=trial_id, policy_name="RL")
        lstm_state = None
        episode_start = np.ones((vec_env.num_envs,), dtype=bool)

        while True:
            if is_recurrent:
                action, lstm_state = model.predict(
                    state,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = model.predict(state, deterministic=True)
            state, reward, done, info = vec_env.step(action)

            trial_info = info[0]
            dropout_active = bool(trial_info["dropout_active"])
            pos_error = float(trial_info["pos_error"])
            if dropout_active and pos_error > 0.0:
                metrics.max_error_during_dropout = max(
                    metrics.max_error_during_dropout, pos_error
                )

            metrics.log.append(
                {
                    "dropout_active": dropout_active,
                    "pos_error": pos_error,
                    "reward": float(reward[0]),
                }
            )

            if bool(done[0]):
                break
            episode_start = done

        dropout_errors = [
            item["pos_error"] for item in metrics.log if item["dropout_active"]
        ]
        if dropout_errors:
            metrics.rmse_during_dropout = float(
                np.sqrt(np.mean(np.square(dropout_errors)))
            )
            metrics.avg_dropout_error = float(np.mean(dropout_errors))
        metrics.crashed = bool(trial_info.get("crashed", False))
        metrics.success = bool(trial_info.get("success", False))
        status = "CRASHED" if metrics.crashed else "OK"
        print(f"RMSE={metrics.rmse_during_dropout:.3f}m ({status})")
        vec_env.close()
        results.append(metrics)

    return results


def plot_comparison(baseline_results: list, rl_results: list, save_dir: str = "results/"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    baseline_rmse = np.array([r.rmse_during_dropout for r in baseline_results])
    rl_rmse = np.array([r.rmse_during_dropout for r in rl_results])
    baseline_crash_rate = np.mean([r.crashed for r in baseline_results])
    rl_crash_rate = np.mean([r.crashed for r in rl_results])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.boxplot(
        [baseline_rmse, rl_rmse],
        labels=["Baseline\n(zero residual)", "Residual PPO"],
        patch_artist=True,
    )
    ax.set_ylabel("RMSE during dropout [m]")
    ax.set_title("Position Error During Dropout")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.bar(
        [0, 1],
        [baseline_crash_rate * 100.0, rl_crash_rate * 100.0],
        color=["green" if baseline_crash_rate == 0 else "red", "green" if rl_crash_rate == 0 else "red"],
        alpha=0.7,
        width=0.5,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Residual PPO"])
    ax.set_ylabel("Crash Rate [%]")
    ax.set_title("Safety")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    bl_log = baseline_results[0].log
    rl_log = rl_results[0].log
    t_bl = np.arange(len(bl_log)) * 0.002
    t_rl = np.arange(len(rl_log)) * 0.002
    ax.plot(t_bl, [item["pos_error"] * 100.0 for item in bl_log], label="Baseline", alpha=0.7)
    ax.plot(t_rl, [item["pos_error"] * 100.0 for item in rl_log], label="Residual PPO", alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [cm]")
    ax.set_title("Sample Trial #1")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = Path(save_dir) / "dropout_rl_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  -> {fig_path}")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate residual PPO policy.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a trained PPO model zip file",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of evaluation trials (default: 10)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/",
        help="Directory for plots",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation rollouts",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["both", "baseline", "rl"],
        default="both",
        help="Which policy to run during evaluation (default: both)",
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        default=0.25,
        help="Residual mixing gain used by the trained environment (default: 0.25)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["hover", "circle", "full"],
        default="full",
        help="Mission preset to evaluate (default: full)",
    )
    return parser.parse_args()


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, 0.0
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, half_width


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  Residual PPO Evaluation")
    print(f"{'=' * 60}")
    print(f"  Model path          : {args.model}")
    print(f"  Evaluation trials   : {args.n_trials}")
    print(f"  Task                : {args.task}")
    print(f"  Mode                : {args.mode}")
    print(f"{'=' * 60}\n")

    baseline_results = []
    rl_results = []

    if args.mode in ("both", "baseline"):
        baseline_env = make_env(
            "human" if args.render else None,
            dropout_randomize=True,
            residual_alpha=args.residual_alpha,
            task=args.task,
        )
        print("[1/2] Evaluating baseline (EKF + PID, zero residual)\n")
        baseline_results = evaluate_baseline(baseline_env, n_episodes=args.n_trials)
        baseline_env.close()

    if args.mode in ("both", "rl"):
        if args.mode == "both":
            print("\n[2/2] Evaluating residual PPO\n")
        else:
            print("[1/1] Evaluating residual PPO\n")
        rl_results = evaluate_rl(
            "human" if args.render else None,
            args.model,
            n_episodes=args.n_trials,
            residual_alpha=args.residual_alpha,
            task=args.task,
        )

    if args.mode == "baseline":
        baseline_rmse = np.array([r.rmse_during_dropout for r in baseline_results])
        baseline_crash = np.mean([r.crashed for r in baseline_results])
        baseline_success = np.mean([r.success for r in baseline_results])
        baseline_rmse_mean, baseline_rmse_ci = mean_ci95(baseline_rmse)
        print(f"\n{'=' * 60}")
        print("  BASELINE RESULTS")
        print(f"{'=' * 60}\n")
        print("Position RMSE during dropout:")
        print(f"  Baseline : {baseline_rmse_mean:.4f} ± {baseline_rmse_ci:.4f} m (95% CI)\n")
        print("Crash rate:")
        print(f"  Baseline : {baseline_crash * 100:.1f} %\n")
        print("Success rate (<20 cm mean dropout error, no crash):")
        print(f"  Baseline : {baseline_success * 100:.1f} %\n")
        print("Evaluation complete!")
        return

    if args.mode == "rl":
        rl_rmse = np.array([r.rmse_during_dropout for r in rl_results])
        rl_crash = np.mean([r.crashed for r in rl_results])
        rl_success = np.mean([r.success for r in rl_results])
        rl_rmse_mean, rl_rmse_ci = mean_ci95(rl_rmse)
        print(f"\n{'=' * 60}")
        print("  RL RESULTS")
        print(f"{'=' * 60}\n")
        print("Position RMSE during dropout:")
        print(f"  RL       : {rl_rmse_mean:.4f} ± {rl_rmse_ci:.4f} m (95% CI)\n")
        print("Crash rate:")
        print(f"  RL       : {rl_crash * 100:.1f} %\n")
        print("Success rate (<20 cm mean dropout error, no crash):")
        print(f"  RL       : {rl_success * 100:.1f} %\n")
        print("Evaluation complete!")
        return

    baseline_rmse = np.array([r.rmse_during_dropout for r in baseline_results])
    rl_rmse = np.array([r.rmse_during_dropout for r in rl_results])
    baseline_crash = np.mean([r.crashed for r in baseline_results])
    rl_crash = np.mean([r.crashed for r in rl_results])
    baseline_success = np.mean([r.success for r in baseline_results])
    rl_success = np.mean([r.success for r in rl_results])
    baseline_rmse_mean, baseline_rmse_ci = mean_ci95(baseline_rmse)
    rl_rmse_mean, rl_rmse_ci = mean_ci95(rl_rmse)

    print(f"\n{'=' * 60}")
    print("  EVALUATION RESULTS")
    print(f"{'=' * 60}\n")
    print("Position RMSE during dropout:")
    print(f"  Baseline : {baseline_rmse_mean:.4f} ± {baseline_rmse_ci:.4f} m (95% CI)")
    print(f"  RL       : {rl_rmse_mean:.4f} ± {rl_rmse_ci:.4f} m (95% CI)")
    improvement = (1.0 - rl_rmse_mean / (baseline_rmse_mean + 1e-9)) * 100.0
    print(f"  RL improvement: {improvement:+.1f} %\n")
    print("Crash rate:")
    print(f"  Baseline : {baseline_crash * 100:.1f} %")
    print(f"  RL       : {rl_crash * 100:.1f} %\n")
    print("Success rate (<20 cm mean dropout error, no crash):")
    print(f"  Baseline : {baseline_success * 100:.1f} %")
    print(f"  RL       : {rl_success * 100:.1f} %\n")

    print("Generating comparison plots...")
    plot_comparison(baseline_results, rl_results, args.save_dir)
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
