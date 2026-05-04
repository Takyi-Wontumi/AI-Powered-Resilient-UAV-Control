#!/usr/bin/env python3
"""
Evaluate RL Policy vs Baseline
===============================

Compares the trained RL policy against the baseline PID+frozen-position controller.

Metrics:
- Position RMSE during dropout
- Recovery time after dropout
- Covariance bounds coverage
- NEES consistency

Usage:
    python examples/evaluate_dropout_rl_policy.py \\
        --model models/dropout_rl/best_model.zip \\
        --n-trials 10 \\
        --render
"""

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

try:
    from stable_baselines3 import PPO
except ImportError:
    print("Error: stable-baselines3 not installed.")
    sys.exit(1)

from AI_UAV_Tests.rl_dropout_policy import DroneDropoutRLEnv


@dataclass
class TrialMetrics:
    """Metrics from a single trial."""
    trial_id: int
    policy_name: str  # "RL" or "Baseline"
    rmse_during_dropout: float = 0.0
    rmse_after_dropout: float = 0.0
    max_error_during_dropout: float = 0.0
    recovery_time: float = 0.0
    crashed: bool = False
    mean_nees: float = 0.0
    mean_nis: float = 0.0
    log: list = field(default_factory=list)


def evaluate_baseline(env: DroneDropoutRLEnv, n_episodes: int = 5) -> list:
    """Evaluate baseline (PID with frozen-position control during dropout)."""
    results = []

    for trial_id in range(n_episodes):
        print(f"  Baseline trial {trial_id + 1}/{n_episodes}… ", end="", flush=True)

        state, _ = env.reset()
        metrics = TrialMetrics(trial_id=trial_id, policy_name="Baseline")

        dropout_active_prev = False
        recovery_time = 0.0

        while True:
            # Baseline: no RL action (use zero action, only PID operates)
            action = np.zeros(4, dtype=np.float32)
            state, reward, done, truncated, info = env.step(action)

            dropout_active = info["dropout_active"]
            pos_error = info["pos_error"]

            if dropout_active:
                if pos_error > 0:
                    metrics.rmse_during_dropout += pos_error ** 2
                    metrics.max_error_during_dropout = max(metrics.max_error_during_dropout, pos_error)
            else:
                if dropout_active_prev:
                    recovery_time += 1

            dropout_active_prev = dropout_active
            metrics.log.append({
                "dropout_active": dropout_active,
                "pos_error": pos_error,
            })

            if done or truncated:
                break

        metrics.crashed = info.get("crashed", False)
        if len(metrics.log) > 0:
            dropout_errors = [l["pos_error"] for l in metrics.log if l["dropout_active"] and l["pos_error"] > 0]
            if dropout_errors:
                metrics.rmse_during_dropout = np.sqrt(np.mean(np.array(dropout_errors) ** 2))

        status = "CRASHED" if metrics.crashed else "OK"
        print(f"RMSE={metrics.rmse_during_dropout:.3f}m ({status})")
        results.append(metrics)

    return results


def evaluate_rl(
    env: DroneDropoutRLEnv,
    model_path: str,
    n_episodes: int = 5,
) -> list:
    """Evaluate RL policy."""
    print(f"\n  Loading model from {model_path}…")
    model = PPO.load(model_path)

    results = []

    for trial_id in range(n_episodes):
        print(f"  RL trial {trial_id + 1}/{n_episodes}… ", end="", flush=True)

        state, _ = env.reset()
        metrics = TrialMetrics(trial_id=trial_id, policy_name="RL")

        dropout_active_prev = False

        while True:
            # RL action
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)

            dropout_active = info["dropout_active"]
            pos_error = info["pos_error"]

            if dropout_active:
                if pos_error > 0:
                    metrics.rmse_during_dropout += pos_error ** 2
                    metrics.max_error_during_dropout = max(metrics.max_error_during_dropout, pos_error)

            metrics.log.append({
                "dropout_active": dropout_active,
                "pos_error": pos_error,
                "reward": reward,
            })

            if done or truncated:
                break

        metrics.crashed = info.get("crashed", False)
        if len(metrics.log) > 0:
            dropout_errors = [l["pos_error"] for l in metrics.log if l["dropout_active"] and l["pos_error"] > 0]
            if dropout_errors:
                metrics.rmse_during_dropout = np.sqrt(np.mean(np.array(dropout_errors) ** 2))

        status = "CRASHED" if metrics.crashed else "OK"
        print(f"RMSE={metrics.rmse_during_dropout:.3f}m ({status})")
        results.append(metrics)

    return results


def plot_comparison(baseline_results: list, rl_results: list, save_dir: str = "results/"):
    """Plot comparison between baseline and RL."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    baseline_rmse = np.array([r.rmse_during_dropout for r in baseline_results])
    rl_rmse = np.array([r.rmse_during_dropout for r in rl_results])

    baseline_crash_rate = np.mean([r.crashed for r in baseline_results])
    rl_crash_rate = np.mean([r.crashed for r in rl_results])

    # Summary figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # RMSE comparison
    ax = axes[0]
    ax.boxplot(
        [baseline_rmse, rl_rmse],
        labels=["Baseline\n(PID+frozen)", "RL Policy"],
        patch_artist=True,
    )
    ax.set_ylabel("RMSE during dropout [m]")
    ax.set_title("Position Error During Dropout")
    ax.grid(alpha=0.3)

    # Crash rate
    ax = axes[1]
    x = [0, 1]
    y = [baseline_crash_rate, rl_crash_rate]
    colors = ["red" if y_i > 0 else "green" for y_i in y]
    ax.bar(x, [c * 100 for c in y], color=colors, alpha=0.7, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "RL"])
    ax.set_ylabel("Crash Rate [%]")
    ax.set_title("Safety: Crash Rate")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")

    # Sample trajectories (first trial)
    ax = axes[2]
    bl_log = baseline_results[0].log
    rl_log = rl_results[0].log
    t_bl = np.arange(len(bl_log)) * 0.002
    t_rl = np.arange(len(rl_log)) * 0.002
    ax.plot(t_bl, [l["pos_error"] * 100 for l in bl_log], label="Baseline", alpha=0.7)
    ax.plot(t_rl, [l["pos_error"] * 100 for l in rl_log], label="RL", alpha=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Error [cm]")
    ax.set_title("Sample Trial #1: Error Trajectory")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = Path(save_dir) / "dropout_rl_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n  → {fig_path}")

    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RL policy for dropout handling.")
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained RL model (e.g., models/dropout_rl/best_model.zip)",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of evaluation trials (default: 10)",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="results/",
        help="Directory to save comparison plots",
    )
    p.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  RL Policy Evaluation: Dropout Handling")
    print(f"{'=' * 60}")
    print(f"  Model path          : {args.model}")
    print(f"  Evaluation trials   : {args.n_trials}")
    print(f"{'=' * 60}\n")

    # Evaluation environment (randomized for testing robustness)
    env = DroneDropoutRLEnv(
        render_mode="human" if args.render else None,
        dropout_randomize=True,
    )

    print("[1/2]  Evaluating baseline (PID + frozen position)…\n")
    baseline_results = evaluate_baseline(env, n_episodes=args.n_trials)

    print("\n[2/2]  Evaluating RL policy…\n")
    rl_results = evaluate_rl(env, args.model, n_episodes=args.n_trials)

    # Summary statistics
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'=' * 60}\n")

    baseline_rmse = np.array([r.rmse_during_dropout for r in baseline_results])
    rl_rmse = np.array([r.rmse_during_dropout for r in rl_results])

    print(f"Position RMSE during dropout:")
    print(f"  Baseline : {baseline_rmse.mean():.4f} ± {baseline_rmse.std():.4f} m")
    print(f"  RL       : {rl_rmse.mean():.4f} ± {rl_rmse.std():.4f} m")
    improvement = (1.0 - rl_rmse.mean() / (baseline_rmse.mean() + 1e-9)) * 100
    print(f"  RL improvement: {improvement:+.1f} %\n")

    baseline_crash = np.mean([r.crashed for r in baseline_results])
    rl_crash = np.mean([r.crashed for r in rl_results])
    print(f"Crash rate:")
    print(f"  Baseline : {baseline_crash * 100:.1f} %")
    print(f"  RL       : {rl_crash * 100:.1f} %\n")

    print(f"{'=' * 60}\n")

    # Plot comparison
    print("Generating comparison plots…")
    plot_comparison(baseline_results, rl_results, args.save_dir)

    env.close()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
