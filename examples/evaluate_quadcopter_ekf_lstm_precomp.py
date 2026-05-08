#!/usr/bin/env python3
"""A/B evaluation for QuadcopterEKF baseline vs LSTM reference pre-compensation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import PhoenixEKFAdapter
from AI_UAV_Tests.ekf_lstm_precomp import (
    BUFFER_LEN,
    DEFAULT_MODEL_PATH,
    TAKEOFF_Z,
    NUM_HORIZONS,
    STANDARD_TRAJECTORIES,
    QuadcopterPositionDriftLSTM,
    build_phase_dropout_windows,
    build_standard_mission,
    build_lstm_feature,
    compute_lstm_state_correction,
    run_reference_velocity_dropout_step,
    thrust_to_action,
)


@dataclass
class TrialMetrics:
    policy_name: str
    trial_id: int
    rmse_dropout: float
    mean_dropout_error: float
    max_dropout_error: float
    within_5cm_rate: float
    overall_mean_error: float
    mean_shift: float
    dropout_windows: list[tuple[float, float]]
    time_s: np.ndarray
    err_m: np.ndarray
    shift_m: np.ndarray
    dropout_mask: np.ndarray
    phase_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate QuadcopterEKF baseline vs LSTM pre-compensation on identical dropout windows."
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="circle",
        choices=list(STANDARD_TRAJECTORIES),
    )
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--dropout-duration", type=float, default=2.0)
    parser.add_argument("--dropout-margin", type=float, default=0.75)
    parser.add_argument("--save-dir", type=str, default="results/lstm_precomp_eval")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--playback-slowdown", type=float, default=1.0)
    parser.add_argument("--target-error-cm", type=float, default=5.0)
    parser.add_argument("--state-gain-xy", type=float, default=0.75)
    parser.add_argument("--state-gain-z", type=float, default=0.40)
    parser.add_argument("--state-max-xy", type=float, default=0.35)
    parser.add_argument("--state-max-z", type=float, default=0.10)
    parser.add_argument("--state-lead-time", type=float, default=0.5)
    parser.add_argument("--state-warmup", type=float, default=0.75)
    parser.add_argument("--use-position-prior", action="store_true")
    return parser.parse_args()


def draw_line(bc, start: np.ndarray, end: np.ndarray, color: tuple[float, float, float], width: float = 2.0) -> None:
    bc.addUserDebugLine(
        lineFromXYZ=np.asarray(start, dtype=float).tolist(),
        lineToXYZ=np.asarray(end, dtype=float).tolist(),
        lineColorRGB=list(color),
        lineWidth=width,
    )


def make_env(render: bool) -> DroneMissionEnv:
    env = DroneMissionEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode="human" if render else None,
    )
    env.drone.control = AttitudeRate(
        bc=env.bc,
        drone=env.drone,
        time_step=env.TIME_STEP,
    )
    return env


def load_model(model_path: Path, device: str) -> QuadcopterPositionDriftLSTM:
    model = QuadcopterPositionDriftLSTM(horizon_steps=NUM_HORIZONS, hidden_size=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def run_trial(
    *,
    trajectory: str,
    trial_id: int,
    use_lstm: bool,
    model: QuadcopterPositionDriftLSTM | None,
    render: bool,
    playback_slowdown: float,
    dropout_duration: float,
    dropout_margin: float,
    target_error_m: float,
    state_gain_xy: float,
    state_gain_z: float,
    state_max_xy: float,
    state_max_z: float,
    state_lead_time: float,
    state_warmup: float,
    use_position_prior: bool,
) -> TrialMetrics:
    np.random.seed(int(trial_id))
    torch.manual_seed(int(trial_id))

    env = make_env(render)
    env.reset()

    mission = build_standard_mission(trajectory)
    include_hover = trajectory == "hover"
    dropout_windows = build_phase_dropout_windows(
        mission,
        duration_s=dropout_duration,
        margin_s=dropout_margin,
        max_windows=2,
        include_hover=include_hover,
    )
    if not dropout_windows:
        raise RuntimeError(
            f"No valid dropout windows found for trajectory '{trajectory}'. "
            "Use a moving trajectory or reduce the dropout duration/margin."
        )

    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    ekf = PhoenixEKFAdapter(
        dt=env.TIME_STEP,
        use_attitude_measurements=True,
        use_velocity_measurements=True,
    )
    ekf.decouple_after_s = None
    ekf.reset(
        position=env.drone.xyz,
        velocity=env.drone.xyz_dot,
        attitude=env.drone.rpy,
        rates=env.drone.rpy_dot,
    )
    estimate = ekf.ekf.as_dict()

    imu_buffer = np.zeros((BUFFER_LEN, 7), dtype=np.float32)
    buffer_idx = 0
    last_lstm_preds = np.zeros((NUM_HORIZONS, 3), dtype=np.float32)
    steps_since_dropout = 0
    prev_gps_ok = True

    total_time = float(mission.total_time)
    steps = int(np.ceil(total_time / env.TIME_STEP)) + 1

    log_t: list[float] = []
    log_err: list[float] = []
    log_shift: list[float] = []
    log_dropout: list[bool] = []
    phase_names: list[str] = []
    prev_draw_pos = np.asarray(env.drone.xyz, dtype=float).copy()

    try:
        for k in range(steps):
            if render:
                time.sleep(env.TIME_STEP * max(playback_slowdown, 0.0))

            t = float(k * env.TIME_STEP)
            gps_ok = not any(start <= t < end for start, end in dropout_windows)

            true_pos = np.asarray(env.drone.xyz, dtype=float)
            vel_true = np.asarray(env.drone.xyz_dot, dtype=float)
            ang_true = np.asarray(env.drone.rpy, dtype=float)
            rate_true = np.asarray(env.drone.rpy_dot, dtype=float)
            phase_name = mission.phase_name_at(t)
            nominal_ref, nominal_vel = mission(t)

            ref_target = nominal_ref.copy()
            ref_vel = nominal_vel.copy()
            moving_phase = phase_name in {"circle", "square", "helix", "sine", "figure_eight"}
            if t < 3.0:
                ref_target = np.array([0.0, 0.0, TAKEOFF_Z], dtype=float)
                ref_vel = np.zeros(3, dtype=float)
            z_ref = float(ref_target[2])

            if gps_ok:
                pos_ctrl = true_pos.copy()
                vel_ctrl = vel_true.copy()
            else:
                pos_ctrl = np.array([estimate["x"][0], estimate["x"][1], true_pos[2]], dtype=float)
                vel_ctrl = vel_true.copy()

            quad.inject_external_state(pos_ctrl, vel_ctrl, ang_true, rate_true)
            ctrl = quad.step(ref_target, ref_vel, z_ref=z_ref)

            action = np.zeros(4, dtype=np.float32)
            action[0] = thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
            action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)

            _obs, _reward, done, truncated, _info = env.step(action)

            if gps_ok:
                estimate = ekf.step(
                    motor_omega=np.asarray(ctrl["omega_cmd"], dtype=float),
                    position=env.drone.xyz,
                    velocity=env.drone.xyz_dot,
                    attitude=env.drone.rpy,
                    rates=env.drone.rpy_dot,
                    dropout_active=False,
                    dt=env.TIME_STEP,
                )
            else:
                estimate = run_reference_velocity_dropout_step(
                    ekf,
                    motor_forces=np.asarray(env.drone.y, dtype=float),
                    position=env.drone.xyz,
                    velocity=env.drone.xyz_dot,
                    attitude=env.drone.rpy,
                    rates=env.drone.rpy_dot,
                    velocity_ref=nominal_vel,
                    position_ref=nominal_ref if use_position_prior else None,
                    dt=env.TIME_STEP,
                )

            if prev_gps_ok and not gps_ok:
                steps_since_dropout = 0
            elif not prev_gps_ok and gps_ok:
                steps_since_dropout = 0
            elif not gps_ok:
                steps_since_dropout += 1
            t_norm = min(steps_since_dropout / 2500.0, 1.0)

            feature = build_lstm_feature(
                np.asarray(estimate["v"], dtype=float),
                np.asarray(estimate["measurement"]["rates"], dtype=float),
                t_norm,
            )
            imu_buffer[buffer_idx % BUFFER_LEN] = feature
            buffer_idx += 1

            if use_lstm and model is not None and buffer_idx >= BUFFER_LEN and not gps_ok:
                ordered = np.roll(imu_buffer, -(buffer_idx % BUFFER_LEN), axis=0)
                tensor = torch.from_numpy(ordered.astype(np.float32)).unsqueeze(0)
                tensor = tensor.to(next(model.parameters()).device)
                with torch.no_grad():
                    preds = model(tensor)
                last_lstm_preds = preds[0].cpu().numpy()
            elif gps_ok:
                last_lstm_preds[:] = 0.0

            prev_gps_ok = gps_ok

            metric_correction = np.zeros(3, dtype=float)
            if use_lstm and not gps_ok and buffer_idx >= BUFFER_LEN and moving_phase:
                metric_correction = compute_lstm_state_correction(
                    lstm_preds=last_lstm_preds,
                    dt=env.TIME_STEP,
                    ref_vel=nominal_vel,
                    dropout_time_s=float(ekf.dropout_time),
                    gain_xy=state_gain_xy,
                    gain_z=state_gain_z,
                    max_xy=state_max_xy,
                    max_z=state_max_z,
                    lead_time_s=state_lead_time,
                    warmup_s=state_warmup,
                )
            corrected_estimate_metric = np.asarray(estimate["x"], dtype=float).copy() - metric_correction

            ekf_error = float(np.linalg.norm(corrected_estimate_metric - np.asarray(env.drone.xyz, dtype=float)))
            shift_mag = float(np.linalg.norm(metric_correction))

            if render and k > 0:
                draw_line(
                    env.bc,
                    prev_draw_pos,
                    np.asarray(env.drone.xyz, dtype=float),
                    (0.95, 0.15, 0.15) if not gps_ok else (0.10, 0.35, 0.95),
                )
            prev_draw_pos = np.asarray(env.drone.xyz, dtype=float).copy()

            log_t.append(t)
            log_err.append(ekf_error)
            log_shift.append(shift_mag)
            log_dropout.append(not gps_ok)
            phase_names.append(phase_name)

            if done or truncated or t >= total_time:
                break
    finally:
        env.close()

    t_arr = np.asarray(log_t, dtype=float)
    err_arr = np.asarray(log_err, dtype=float)
    shift_arr = np.asarray(log_shift, dtype=float)
    dropout_mask = np.asarray(log_dropout, dtype=bool)

    dropout_errors = err_arr[dropout_mask]
    rmse_dropout = float(np.sqrt(np.mean(np.square(dropout_errors)))) if dropout_errors.size else 0.0
    mean_dropout_error = float(np.mean(dropout_errors)) if dropout_errors.size else 0.0
    max_dropout_error = float(np.max(dropout_errors)) if dropout_errors.size else 0.0
    within_5cm_rate = float(np.mean(dropout_errors <= target_error_m)) if dropout_errors.size else 1.0

    return TrialMetrics(
        policy_name="LSTM" if use_lstm else "Baseline",
        trial_id=trial_id,
        rmse_dropout=rmse_dropout,
        mean_dropout_error=mean_dropout_error,
        max_dropout_error=max_dropout_error,
        within_5cm_rate=within_5cm_rate,
        overall_mean_error=float(np.mean(err_arr)) if err_arr.size else 0.0,
        mean_shift=float(np.mean(shift_arr[dropout_mask])) if dropout_errors.size else 0.0,
        dropout_windows=dropout_windows,
        time_s=t_arr,
        err_m=err_arr,
        shift_m=shift_arr,
        dropout_mask=dropout_mask,
        phase_names=phase_names,
    )


def summarize(label: str, results: list[TrialMetrics], target_error_m: float) -> dict:
    rmse = np.array([item.rmse_dropout for item in results], dtype=float)
    mean_err = np.array([item.mean_dropout_error for item in results], dtype=float)
    max_err = np.array([item.max_dropout_error for item in results], dtype=float)
    within = np.array([item.within_5cm_rate for item in results], dtype=float)
    shifts = np.array([item.mean_shift for item in results], dtype=float)
    pass_rate = np.mean((rmse <= target_error_m) & (max_err <= target_error_m)) if rmse.size else 0.0
    summary = {
        "label": label,
        "rmse_mean": float(np.mean(rmse)) if rmse.size else 0.0,
        "rmse_std": float(np.std(rmse)) if rmse.size else 0.0,
        "mean_error": float(np.mean(mean_err)) if mean_err.size else 0.0,
        "max_error": float(np.mean(max_err)) if max_err.size else 0.0,
        "within_target_rate": float(np.mean(within)) if within.size else 0.0,
        "pass_rate": float(pass_rate),
        "mean_shift": float(np.mean(shifts)) if shifts.size else 0.0,
    }
    print(f"{label}:")
    print(f"  mean dropout RMSE     : {summary['rmse_mean']:.3f} m  (std {summary['rmse_std']:.3f})")
    print(f"  mean dropout error    : {summary['mean_error']:.3f} m")
    print(f"  mean max dropout err  : {summary['max_error']:.3f} m")
    print(f"  within +/-{target_error_m * 100:.0f} cm rate : {100.0 * summary['within_target_rate']:.1f} %")
    print(f"  pass rate             : {100.0 * summary['pass_rate']:.1f} %")
    print(f"  mean correction shift : {summary['mean_shift']:.3f} m")
    return summary


def save_plot(
    *,
    baseline: TrialMetrics,
    lstm: TrialMetrics,
    save_dir: Path,
    target_error_m: float,
    trajectory: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(baseline.time_s, baseline.err_m, label="baseline error", linewidth=1.5)
    axs[0].plot(lstm.time_s, lstm.err_m, label="lstm error", linewidth=1.5)
    axs[0].axhline(target_error_m, color="tab:green", linestyle="--", linewidth=1.0, label="5 cm target")
    for start, end in baseline.dropout_windows:
        axs[0].axvspan(start, end, color="0.9", alpha=0.5, zorder=0)
    axs[0].set_ylabel("Position Error [m]")
    axs[0].set_title(f"Dropout Error A/B: {trajectory}")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].plot(baseline.time_s, baseline.shift_m, label="baseline shift", linewidth=1.2)
    axs[1].plot(lstm.time_s, lstm.shift_m, label="lstm shift", linewidth=1.2)
    for start, end in baseline.dropout_windows:
        axs[1].axvspan(start, end, color="0.9", alpha=0.5, zorder=0)
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Reference Shift [m]")
    axs[1].set_title("Applied Correction Magnitude")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    fig.tight_layout()
    out_path = save_dir / f"ab_eval_{trajectory}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main() -> None:
    args = parse_args()
    target_error_m = float(args.target_error_cm) / 100.0
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM model not found at {model_path}. "
            "Run train_quadcopter_ekf_drift_lstm.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)

    print(f"\n{'=' * 76}")
    print("QuadcopterEKF LSTM A/B Evaluation")
    print(f"{'=' * 76}")
    print(f"Trajectory          : {args.trajectory}")
    print(f"Trials              : {args.n_trials}")
    print(f"Dropout duration    : {args.dropout_duration:.1f} s")
    print(f"Dropout margin      : {args.dropout_margin:.1f} s")
    print(f"Target envelope     : +/-{args.target_error_cm:.1f} cm")
    print(
        "State correction    : "
        f"gain_xy={args.state_gain_xy:.2f}, gain_z={args.state_gain_z:.2f}, "
        f"max_xy={args.state_max_xy:.2f}, max_z={args.state_max_z:.2f}, "
        f"lead={args.state_lead_time:.2f}s, warmup={args.state_warmup:.2f}s"
    )
    print(f"Position prior      : {'ON' if args.use_position_prior else 'OFF'}")
    print(f"Model               : {model_path}")
    print(f"{'=' * 76}\n")

    baseline_results: list[TrialMetrics] = []
    lstm_results: list[TrialMetrics] = []

    for trial_id in range(int(args.n_trials)):
        baseline = run_trial(
            trajectory=args.trajectory,
            trial_id=trial_id,
            use_lstm=False,
            model=None,
            render=False,
            playback_slowdown=args.playback_slowdown,
            dropout_duration=args.dropout_duration,
            dropout_margin=args.dropout_margin,
            target_error_m=target_error_m,
            state_gain_xy=args.state_gain_xy,
            state_gain_z=args.state_gain_z,
            state_max_xy=args.state_max_xy,
            state_max_z=args.state_max_z,
            state_lead_time=args.state_lead_time,
            state_warmup=args.state_warmup,
            use_position_prior=args.use_position_prior,
        )
        lstm = run_trial(
            trajectory=args.trajectory,
            trial_id=trial_id,
            use_lstm=True,
            model=model,
            render=args.render and trial_id == 0,
            playback_slowdown=args.playback_slowdown,
            dropout_duration=args.dropout_duration,
            dropout_margin=args.dropout_margin,
            target_error_m=target_error_m,
            state_gain_xy=args.state_gain_xy,
            state_gain_z=args.state_gain_z,
            state_max_xy=args.state_max_xy,
            state_max_z=args.state_max_z,
            state_lead_time=args.state_lead_time,
            state_warmup=args.state_warmup,
            use_position_prior=args.use_position_prior,
        )
        baseline_results.append(baseline)
        lstm_results.append(lstm)
        print(
            f"trial {trial_id + 1:02d}: "
            f"baseline RMSE={baseline.rmse_dropout:.3f} m, "
            f"lstm RMSE={lstm.rmse_dropout:.3f} m, "
            f"baseline max={baseline.max_dropout_error:.3f} m, "
            f"lstm max={lstm.max_dropout_error:.3f} m"
        )

    print()
    baseline_summary = summarize("Baseline", baseline_results, target_error_m)
    lstm_summary = summarize("LSTM", lstm_results, target_error_m)
    rmse_improvement = (
        100.0 * (baseline_summary["rmse_mean"] - lstm_summary["rmse_mean"])
        / max(1.0e-9, baseline_summary["rmse_mean"])
    )
    within_improvement = 100.0 * (
        lstm_summary["within_target_rate"] - baseline_summary["within_target_rate"]
    )
    print("\nComparison:")
    print(f"  RMSE improvement       : {rmse_improvement:+.1f} %")
    print(f"  within-5cm improvement : {within_improvement:+.1f} percentage points")

    save_plot(
        baseline=baseline_results[0],
        lstm=lstm_results[0],
        save_dir=Path(args.save_dir),
        target_error_m=target_error_m,
        trajectory=args.trajectory,
    )


if __name__ == "__main__":
    main()
