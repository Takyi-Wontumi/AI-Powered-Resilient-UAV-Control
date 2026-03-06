"""Real-world Crazyflie execution for QuadcopterPID.

This script connects to a Crazyflie, runs QuadcopterPID with live state, and writes
dated CSV telemetry to Drone_Logs:
- odometry (state estimator)
- IMU (best-effort discovery across common CF variable names)
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(THIS_FILE), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.trajectories_library import Trajectories as path

try:
    from AI_UAV_Tests.trajectories_library import MissionPlannerTrajectory
except Exception:
    MissionPlannerTrajectory = None


URI = "radio://0/80/2M/E7E7E7E7E7"
DT_CTRL = 0.02
KILL_Z = 1.8
KILL_XY = 2.0
HOVER_THRUST = 40000
MIN_THRUST = 10001
MAX_THRUST = 60000
LOG_ROOT = os.path.join(REPO_ROOT, "Drone_Logs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QuadcopterPID on Crazyflie and log flight data."
    )
    parser.add_argument("--uri", default=URI)
    parser.add_argument("--dt", type=float, default=DT_CTRL)
    parser.add_argument(
        "--trajectory",
        choices=["hover", "point", "square", "circle", "sine", "helix"],
        default="circle",
    )
    parser.add_argument(
        "--mission",
        default=None,
        help="Mission Planner file path (.mission JSON or .waypoints). Overrides --trajectory.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional max flight time in seconds.",
    )
    parser.add_argument(
        "--log-dir",
        default=LOG_ROOT,
        help="Base log directory. A dated subfolder is created automatically.",
    )
    parser.add_argument("--kill-z", type=float, default=KILL_Z)
    parser.add_argument("--kill-xy", type=float, default=KILL_XY)
    parser.add_argument("--hover-thrust", type=float, default=HOVER_THRUST)
    return parser.parse_args()


def u1_newtons_to_thrust_int(U1_N: float, m: float, g: float, hover_thrust: float) -> int:
    scale = float(U1_N / (m * g))
    thrust = int(np.clip(hover_thrust * scale, MIN_THRUST, MAX_THRUST))
    return thrust


def make_reference(
    traj_name: str, mission_file: Optional[str]
) -> Tuple[Callable[[float], Tuple[np.ndarray, np.ndarray]], str]:
    if mission_file:
        if MissionPlannerTrajectory is None:
            raise RuntimeError(
                "MissionPlannerTrajectory requires pyproj. Install with: pip install pyproj"
            )
        mission_path = os.path.abspath(mission_file)
        mission = MissionPlannerTrajectory(mission_path)

        def mission_ref(t: float):
            return mission(t)

        return mission_ref, f"mission:{os.path.basename(mission_path)}"

    if traj_name == "hover":
        return lambda t: path.hover_traj(t, pos=(0.0, 0.0, 1.0)), "hover"
    if traj_name == "point":
        return lambda t: path.point_traj((0.0, 0.0, 1.0)), "point(0,0,1)"
    if traj_name == "square":
        return path.square_traj, "square"
    if traj_name == "circle":
        return path.circle_traj, "circle"
    if traj_name == "sine":
        return path.sine_traj, "sine"
    return path.helix_traj, "helix"


def make_log_files(base_dir: str):
    started = datetime.now()
    date_dir = Path(base_dir) / started.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    stamp = started.strftime("%Y%m%d_%H%M%S")
    csv_path = date_dir / f"crazyflie_pid_{stamp}.csv"
    meta_path = date_dir / f"crazyflie_pid_{stamp}_meta.json"
    return started, csv_path, meta_path


def start_csv_writer(path: Path):
    fields = [
        "timestamp_s",
        "flight_t_s",
        "dt_s",
        # Odometry (estimator output)
        "odom_x",
        "odom_y",
        "odom_z",
        "odom_vx",
        "odom_vy",
        "odom_vz",
        "odom_roll_rad",
        "odom_pitch_rad",
        "odom_yaw_rad",
        "odom_rate_x_rad_s",
        "odom_rate_y_rad_s",
        "odom_rate_z_rad_s",
        # IMU (best-effort discovery in firmware)
        "imu_accel_x",
        "imu_accel_y",
        "imu_accel_z",
        "imu_gyro_x",
        "imu_gyro_y",
        "imu_gyro_z",
        "imu_temp_c",
        # Controller target/reference
        "ref_x",
        "ref_y",
        "ref_z",
        "ref_vx",
        "ref_vy",
        "ref_vz",
        # Actuation
        "thrust_cmd_n",
        "thrust_cf",
        "roll_rate_dps",
        "pitch_rate_dps",
        "yaw_rate_dps",
        "rates_des_roll_rad_s",
        "rates_des_pitch_rad_s",
        "rates_des_yaw_rad_s",
        "safety_ok",
    ]
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


def write_meta(meta_path: Path, start_time: datetime, args, reference_name: str, csv_path: Path, imu_vars: dict):
    payload = {
        "start_time": start_time.isoformat(),
        "trajectory": reference_name,
        "uri": args.uri,
        "dt": args.dt,
        "hover_thrust": args.hover_thrust,
        "kill_z": args.kill_z,
        "kill_xy": args.kill_xy,
        "duration_s": args.duration,
        "log_csv": str(csv_path),
        "imu_variables": imu_vars,
        "argv": vars(args),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_log_row(
    writer,
    elapsed: float,
    dt: float,
    odom: dict,
    imu: dict,
    pos_ref: np.ndarray,
    vel_ref: np.ndarray,
    thrust_n: float,
    thrust_cf: int,
    rates_des: np.ndarray,
    safety_ok: bool,
):
    writer.writerow(
        {
            "timestamp_s": datetime.now().timestamp(),
            "flight_t_s": round(elapsed, 6),
            "dt_s": round(dt, 6),
            "odom_x": odom.get("x"),
            "odom_y": odom.get("y"),
            "odom_z": odom.get("z"),
            "odom_vx": odom.get("vx"),
            "odom_vy": odom.get("vy"),
            "odom_vz": odom.get("vz"),
            "odom_roll_rad": odom.get("roll"),
            "odom_pitch_rad": odom.get("pitch"),
            "odom_yaw_rad": odom.get("yaw"),
            "odom_rate_x_rad_s": odom.get("rate_x"),
            "odom_rate_y_rad_s": odom.get("rate_y"),
            "odom_rate_z_rad_s": odom.get("rate_z"),
            "imu_accel_x": imu.get("accel_x"),
            "imu_accel_y": imu.get("accel_y"),
            "imu_accel_z": imu.get("accel_z"),
            "imu_gyro_x": imu.get("gyro_x"),
            "imu_gyro_y": imu.get("gyro_y"),
            "imu_gyro_z": imu.get("gyro_z"),
            "imu_temp_c": imu.get("temp_c"),
            "ref_x": pos_ref[0],
            "ref_y": pos_ref[1],
            "ref_z": pos_ref[2],
            "ref_vx": vel_ref[0],
            "ref_vy": vel_ref[1],
            "ref_vz": vel_ref[2],
            "thrust_cmd_n": thrust_n,
            "thrust_cf": thrust_cf,
            "roll_rate_dps": float(np.rad2deg(rates_des[0])),
            "pitch_rate_dps": float(np.rad2deg(rates_des[1])),
            "yaw_rate_dps": float(np.rad2deg(rates_des[2])),
            "rates_des_roll_rad_s": float(rates_des[0]),
            "rates_des_pitch_rad_s": float(rates_des[1]),
            "rates_des_yaw_rad_s": float(rates_des[2]),
            "safety_ok": int(bool(safety_ok)),
        }
    )


def set_flightmode_rate(cf: Crazyflie):
    cf.param.set_value("flightmode.stabModeRoll", "0")
    cf.param.set_value("flightmode.stabModePitch", "0")
    cf.param.set_value("flightmode.stabModeYaw", "0")
    cf.param.set_value("flightmode.althold", "0")
    cf.param.set_value("flightmode.poshold", "0")


def _add_log_var(cfg: LogConfig, var: str, dtype: str) -> bool:
    try:
        cfg.add_variable(var, dtype)
        return True
    except Exception:
        return False


def _pick_first_available_var(cfg: LogConfig, candidates: list[str], dtype: str) -> Optional[str]:
    for var in candidates:
        if _add_log_var(cfg, var, dtype):
            return var
    return None


def make_logconfs(period_ms: int):
    lg_state = LogConfig(name="state", period_in_ms=period_ms)
    lg_state.add_variable("stateEstimate.x", "float")
    lg_state.add_variable("stateEstimate.y", "float")
    lg_state.add_variable("stateEstimate.z", "float")
    lg_state.add_variable("stateEstimate.vx", "float")
    lg_state.add_variable("stateEstimate.vy", "float")
    lg_state.add_variable("stateEstimate.vz", "float")

    lg_att = LogConfig(name="att", period_in_ms=period_ms)
    lg_att.add_variable("stateEstimate.roll", "float")
    lg_att.add_variable("stateEstimate.pitch", "float")
    lg_att.add_variable("stateEstimate.yaw", "float")
    lg_att.add_variable("stateEstimateZ.rateRoll", "int16")
    lg_att.add_variable("stateEstimateZ.ratePitch", "int16")
    lg_att.add_variable("stateEstimateZ.rateYaw", "int16")

    lg_imu = LogConfig(name="imu", period_in_ms=period_ms)
    imu_map = {
        "accel_x": _pick_first_available_var(lg_imu, ["imu.acc.x", "acc.x"], "float"),
        "accel_y": _pick_first_available_var(lg_imu, ["imu.acc.y", "acc.y"], "float"),
        "accel_z": _pick_first_available_var(lg_imu, ["imu.acc.z", "acc.z"], "float"),
        "gyro_x": _pick_first_available_var(lg_imu, ["imu.gyro.x", "gyro.x"], "float"),
        "gyro_y": _pick_first_available_var(lg_imu, ["imu.gyro.y", "gyro.y"], "float"),
        "gyro_z": _pick_first_available_var(lg_imu, ["imu.gyro.z", "gyro.z"], "float"),
        "temp_c": _pick_first_available_var(lg_imu, ["imu.temp", "temp"], "float"),
    }
    # drop missing ones
    imu_map = {k: v for k, v in imu_map.items() if v is not None}

    if not imu_map:
        lg_imu = None

    return lg_state, lg_att, lg_imu, imu_map


def ramp_thrust(
    cf: Crazyflie,
    min_thrust: float,
    hover_thrust: float,
    dt: float,
    t_s: float = 1.0,
):
    steps = max(1, int(t_s / dt))
    for i in range(steps):
        u = (i + 1) / steps
        thrust = int(min_thrust + u * (hover_thrust - min_thrust))
        cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
        time.sleep(dt)


def land_and_stop(
    cf: Crazyflie,
    min_thrust: float,
    hover_thrust: float,
    dt: float,
    t_s: float = 1.0,
):
    steps = max(1, int(t_s / dt))
    for i in range(steps):
        u = 1.0 - (i + 1) / steps
        thrust = int(min_thrust + u * (hover_thrust - min_thrust))
        cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
        time.sleep(dt)
    cf.commander.send_stop_setpoint()


def main():
    cflib.crtp.init_drivers(enable_debug_driver=False)
    args = parse_args()

    ref_fn, ref_name = make_reference(args.trajectory, args.mission)
    log_started, csv_path, meta_path = make_log_files(args.log_dir)
    period_ms = int(max(1, round(args.dt * 1000)))
    csv_file, csv_writer = start_csv_writer(csv_path)
    quad = QuadcopterPID(dt=args.dt)
    lg_state, lg_att, lg_imu, imu_map = make_logconfs(period_ms)

    write_meta(meta_path, log_started, args, ref_name, csv_path, imu_map)
    print(f"[INFO] Log file: {csv_path}")
    print(f"[INFO] Meta file: {meta_path}")
    if imu_map:
        print(f"[INFO] IMU vars logged: {imu_map}")

    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        set_flightmode_rate(cf)

        cf.log.add_config(lg_state)
        cf.log.add_config(lg_att)
        if lg_imu is not None:
            cf.log.add_config(lg_imu)

        lg_state.start()
        lg_att.start()
        if lg_imu is not None:
            lg_imu.start()

        quad.reset()
        ramp_thrust(
            cf,
            MIN_THRUST,
            args.hover_thrust,
            dt=args.dt,
            t_s=1.2,
        )

        latest = {}
        last_send = 0.0
        last_log_t = None

        try:
            loggers = [lg_state, lg_att] if lg_imu is None else [lg_state, lg_att, lg_imu]

            with SyncLogger(scf, loggers) as logger:
                t0 = time.time()
                safety_ok = True
                t_end = None if args.duration is None else (t0 + args.duration)

                for _, data, _ in logger:
                    latest.update(data)
                    now = time.time()
                    if now - last_send < args.dt:
                        continue
                    last_send = now

                    if t_end is not None and now >= t_end:
                        print("[INFO] Duration finished. Landing.")
                        break

                    odom = {
                        "x": float(latest.get("stateEstimate.x", 0.0)),
                        "y": float(latest.get("stateEstimate.y", 0.0)),
                        "z": float(latest.get("stateEstimate.z", 0.0)),
                        "vx": float(latest.get("stateEstimate.vx", 0.0)),
                        "vy": float(latest.get("stateEstimate.vy", 0.0)),
                        "vz": float(latest.get("stateEstimate.vz", 0.0)),
                        "roll": np.deg2rad(float(latest.get("stateEstimate.roll", 0.0))),
                        "pitch": -np.deg2rad(float(latest.get("stateEstimate.pitch", 0.0))),
                        "yaw": np.deg2rad(float(latest.get("stateEstimate.yaw", 0.0))),
                        "rate_x": 1e-3 * float(latest.get("stateEstimateZ.rateRoll", 0.0)),
                        "rate_y": 1e-3 * float(latest.get("stateEstimateZ.ratePitch", 0.0)),
                        "rate_z": 1e-3 * float(latest.get("stateEstimateZ.rateYaw", 0.0)),
                    }

                    imu = {
                        "accel_x": float(latest.get(imu_map.get("accel_x"), np.nan))
                        if "accel_x" in imu_map
                        else None,
                        "accel_y": float(latest.get(imu_map.get("accel_y"), np.nan))
                        if "accel_y" in imu_map
                        else None,
                        "accel_z": float(latest.get(imu_map.get("accel_z"), np.nan))
                        if "accel_z" in imu_map
                        else None,
                        "gyro_x": float(latest.get(imu_map.get("gyro_x"), np.nan))
                        if "gyro_x" in imu_map
                        else None,
                        "gyro_y": float(latest.get(imu_map.get("gyro_y"), np.nan))
                        if "gyro_y" in imu_map
                        else None,
                        "gyro_z": float(latest.get(imu_map.get("gyro_z"), np.nan))
                        if "gyro_z" in imu_map
                        else None,
                        "temp_c": float(latest.get(imu_map.get("temp_c"), np.nan))
                        if "temp_c" in imu_map
                        else None,
                    }

                    x = np.array([odom["x"], odom["y"], odom["z"]], dtype=float)
                    v = np.array([odom["vx"], odom["vy"], odom["vz"]], dtype=float)
                    ang = np.array([odom["roll"], odom["pitch"], odom["yaw"]], dtype=float)
                    rate = np.array([odom["rate_x"], odom["rate_y"], odom["rate_z"]], dtype=float)

                    if x[2] > args.kill_z or np.linalg.norm(x[:2]) > args.kill_xy or x[2] < 0.05:
                        print("[SAFETY] Kill condition triggered. Landing.")
                        safety_ok = False
                        break

                    elapsed = now - t0
                    pos_ref, vel_ref = ref_fn(elapsed)

                    quad.inject_external_state(x, v, ang, rate)
                    ctrl = quad.step(pos_ref, vel_ref, z_ref=float(pos_ref[2]))
                    rates_des = np.clip(
                        ctrl["rates_des"],
                        -np.deg2rad(200),
                        np.deg2rad(200),
                    )
                    thrust_n = float(ctrl["thrust_cmd"])
                    thrust_cf = u1_newtons_to_thrust_int(
                        thrust_n, m=quad.m, g=quad.g, hover_thrust=args.hover_thrust
                    )

                    cf.commander.send_setpoint(
                        float(np.rad2deg(rates_des[0])),
                        float(np.rad2deg(rates_des[1])),
                        float(np.rad2deg(rates_des[2])),
                        thrust_cf,
                    )

                    dt = 0.0 if last_log_t is None else max(0.0, now - last_log_t)
                    last_log_t = now
                    write_log_row(
                        csv_writer,
                        elapsed,
                        dt,
                        odom,
                        imu,
                        pos_ref,
                        vel_ref,
                        thrust_n,
                        thrust_cf,
                        rates_des,
                        safety_ok,
                    )
                    csv_file.flush()

        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt. Landing.")
        except Exception:
            print("\n[ERROR] Run failed. Landing and saving logs.")
            raise
        finally:
            land_and_stop(
                cf,
                MIN_THRUST,
                args.hover_thrust,
                dt=args.dt,
                t_s=1.0,
            )
            lg_state.stop()
            lg_att.stop()
            if lg_imu is not None:
                lg_imu.stop()
            csv_file.close()


if __name__ == "__main__":
    main()
