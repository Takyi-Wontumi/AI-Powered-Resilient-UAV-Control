#!/usr/bin/env python3
"""Collect LSTM drift-training data using the 12-state QuadcopterEKF."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import PhoenixEKFAdapter
from AI_UAV_Tests.ekf_lstm_precomp import (
    DEFAULT_DATA_DIR,
    SHADOW_RESET_INTERVAL,
    STANDARD_TRAJECTORIES,
    STANDARD_TRAJECTORY_WEIGHTS,
    build_randomized_standard_mission,
    build_standard_mission,
    build_lstm_feature,
    run_reference_velocity_dropout_step,
    thrust_to_action,
)


TRAJECTORIES = STANDARD_TRAJECTORIES
TRAJECTORY_WEIGHTS = STANDARD_TRAJECTORY_WEIGHTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect EKF drift data for QuadcopterEKF LSTM pre-compensation."
    )
    parser.add_argument("--missions", type=int, default=40)
    parser.add_argument("--steps-per-mission", type=int, default=3500)
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--min-steps", type=int, default=350)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trajectory-filter",
        type=str,
        default="all",
        choices=["all", *TRAJECTORIES],
        help="Collect only one trajectory type instead of the mixed set.",
    )
    parser.add_argument(
        "--deterministic-missions",
        action="store_true",
        help="Disable parameter randomization and use the fixed standard missions.",
    )
    return parser.parse_args()


def make_env() -> DroneMissionEnv:
    env = DroneMissionEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode=None,
    )
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP,
    )
    return env


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 76}")
    print("Collect QuadcopterEKF Drift Training Data")
    print(f"{'=' * 76}")
    print(f"Missions            : {args.missions}")
    print(f"Steps per mission   : {args.steps_per_mission}")
    print(f"Min retained steps  : {args.min_steps}")
    print(f"Seed                : {args.seed}")
    print(f"Randomized missions : {not args.deterministic_missions}")
    print(f"Save dir            : {save_dir}")
    print(f"{'=' * 76}\n")

    env = make_env()
    obs, _info = env.reset()
    print(f"TIME_STEP={env.TIME_STEP:.4f}s  obs_shape={obs.shape}\n")

    active_trajectories = TRAJECTORIES if args.trajectory_filter == "all" else (args.trajectory_filter,)
    active_weights = (
        TRAJECTORY_WEIGHTS
        if args.trajectory_filter == "all"
        else (1.0,)
    )
    trajectory_counts = {name: 0 for name in active_trajectories}
    skipped = 0
    missions: list[dict] = []

    print(f"{'#':>4}  {'traj':>12}  {'steps':>6}  {'drift_max':>10}  {'cumulative':>12}")
    print("-" * 54)

    for mission_idx in range(args.missions):
        trajectory_name = str(rng.choice(active_trajectories, p=active_weights))
        trajectory_counts[trajectory_name] += 1
        if args.deterministic_missions:
            mission = build_standard_mission(trajectory_name)
        else:
            mission = build_randomized_standard_mission(trajectory_name, rng=rng)
        mission_steps = min(
            int(args.steps_per_mission),
            int(np.ceil(mission.total_time / env.TIME_STEP)) + 1,
        )

        try:
            env.reset()
            env.drone.control = AttitudeRate(
                bc=env.bc,
                drone=env.drone,
                time_step=env.TIME_STEP,
            )
        except Exception as exc:
            print(f"[!] env.reset() failed on mission {mission_idx}: {exc}")
            skipped += 1
            continue

        quad = QuadcopterPID(dt=env.TIME_STEP)
        quad.reset()

        shadow = PhoenixEKFAdapter(
            dt=env.TIME_STEP,
            use_attitude_measurements=True,
            use_velocity_measurements=True,
        )
        shadow.decouple_after_s = None
        shadow.reset(
            position=env.drone.xyz,
            velocity=env.drone.xyz_dot,
            attitude=env.drone.rpy,
            rates=env.drone.rpy_dot,
        )

        feature_list = []
        true_list = []
        dr_list = []
        t_norm_list = []
        done = False

        for step in range(mission_steps):
            t = float(step * env.TIME_STEP)
            true_pos = np.asarray(env.drone.xyz, dtype=float)
            vel_true = np.asarray(env.drone.xyz_dot, dtype=float)
            ang_true = np.asarray(env.drone.rpy, dtype=float)
            rate_true = np.asarray(env.drone.rpy_dot, dtype=float)
            ref, vel_ref = mission(t)
            z_ref = float(ref[2])
            phase_name = mission.phase_name_at(t)

            quad.inject_external_state(true_pos, vel_true, ang_true, rate_true)
            ctrl = quad.step(ref, vel_ref, z_ref=z_ref)

            action = np.zeros(4, dtype=np.float32)
            action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
            action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

            _obs, _reward, done, truncated, _info = env.step(action)

            if step > 0 and step % SHADOW_RESET_INTERVAL == 0:
                shadow.reset(
                    position=env.drone.xyz,
                    velocity=env.drone.xyz_dot,
                    attitude=env.drone.rpy,
                    rates=env.drone.rpy_dot,
                )
                shadow.decouple_after_s = None

            estimate = run_reference_velocity_dropout_step(
                shadow,
                motor_forces=np.asarray(env.drone.y, dtype=float),
                position=env.drone.xyz,
                velocity=env.drone.xyz_dot,
                attitude=env.drone.rpy,
                rates=env.drone.rpy_dot,
                velocity_ref=vel_ref,
                position_ref=ref,
                dt=env.TIME_STEP,
            )
            steps_since_reset = step % SHADOW_RESET_INTERVAL
            t_norm = steps_since_reset / float(SHADOW_RESET_INTERVAL)

            if phase_name in {"takeoff", "landing"}:
                if done or truncated:
                    break
                continue

            feature_list.append(
                build_lstm_feature(
                    np.asarray(estimate["v"], dtype=float),
                    np.asarray(estimate["measurement"]["rates"], dtype=float),
                    t_norm,
                )
            )
            true_list.append(np.asarray(env.drone.xyz, dtype=np.float32))
            dr_list.append(np.asarray(estimate["x"], dtype=np.float32))
            t_norm_list.append(np.float32(t_norm))

            if done or truncated:
                break

        n_steps = len(feature_list)
        if n_steps < int(args.min_steps):
            print(f"{mission_idx + 1:4d}  {trajectory_name:>12}  {n_steps:6d}  {'skip':>10}  {'-':>12}")
            skipped += 1
            continue

        true_arr = np.asarray(true_list, dtype=np.float32)
        dr_arr = np.asarray(dr_list, dtype=np.float32)
        drift = dr_arr - true_arr
        max_drift = float(np.max(np.linalg.norm(drift, axis=1)))

        missions.append(
            {
                "features": np.asarray(feature_list, dtype=np.float32),
                "true_pos": true_arr,
                "dr_pos": dr_arr,
                "t_norm_meas": np.asarray(t_norm_list, dtype=np.float32),
                "mission_type": trajectory_name,
                "duration_steps": n_steps,
                "source": "quadcopter_ekf_shadow",
                "collector_version": "runtime_consistent_v2",
                "collector_seed": int(args.seed),
                "randomized_mission": bool(not args.deterministic_missions),
                "max_drift_m": max_drift,
            }
        )

        cumulative = sum(item["duration_steps"] for item in missions)
        print(f"{mission_idx + 1:4d}  {trajectory_name:>12}  {n_steps:6d}  {max_drift:9.3f}m  {cumulative:12,}")

    env.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = save_dir / f"quadcopter_ekf_drift_{len(missions)}missions_{timestamp}.pkl"
    with out_path.open("wb") as f:
        pickle.dump(missions, f)

    total_steps = sum(item["duration_steps"] for item in missions)
    print(f"\n{'=' * 76}")
    print("Collection Complete")
    print(f"{'=' * 76}")
    print(f"Missions collected   : {len(missions)}  (skipped: {skipped})")
    print(f"Total steps          : {total_steps:,}")
    print(f"Trajectory mix       : {trajectory_counts}")
    print(f"Saved to             : {out_path}")


if __name__ == "__main__":
    main()
