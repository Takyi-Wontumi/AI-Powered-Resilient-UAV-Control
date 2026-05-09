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

from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.core.trajectories_library import FlightMission, Trajectories as path

try:
    from AI_UAV_Tests.core.trajectories_library import MissionPlannerTrajectory
except Exception:
    MissionPlannerTrajectory = None


URI = "radio://0/80/2M/E7E7E7E7E7"
DT_CTRL = 0.02
KILL_Z = 1.8
KILL_XY = 2.0
HOVER_THRUST = 40000
MIN_THRUST = 10001
MAX_THRUST = 60000
STARTUP_TIMEOUT_S = 3.0
LOG_ROOT = os.path.join(REPO_ROOT, "Drone_Logs")
CONTROL_LOG_CANDIDATES = {
    "x": (("stateEstimate.x", "identity"),),
    "y": (("stateEstimate.y", "identity"),),
    "z": (("stateEstimate.z", "identity"),),
    "vx": (("stateEstimate.vx", "identity"),),
    "vy": (("stateEstimate.vy", "identity"),),
    "vz": (("stateEstimate.vz", "identity"),),
    "roll": (
        ("stateEstimate.roll", "deg_to_rad"),
        ("stabilizer.roll", "deg_to_rad"),
    ),
    "pitch": (
        ("stateEstimate.pitch", "neg_deg_to_rad"),
        ("stabilizer.pitch", "neg_deg_to_rad"),
    ),
    "yaw": (
        ("stateEstimate.yaw", "deg_to_rad"),
        ("stabilizer.yaw", "deg_to_rad"),
    ),
    "rate_x": (
        ("stateEstimateZ.rateRoll", "mrad_to_rad"),
        ("gyro.x", "deg_to_rad"),
        ("imu.gyro.x", "deg_to_rad"),
    ),
    "rate_y": (
        ("stateEstimateZ.ratePitch", "mrad_to_rad"),
        ("gyro.y", "deg_to_rad"),
        ("imu.gyro.y", "deg_to_rad"),
    ),
    "rate_z": (
        ("stateEstimateZ.rateYaw", "mrad_to_rad"),
        ("gyro.z", "deg_to_rad"),
        ("imu.gyro.z", "deg_to_rad"),
    ),
}
IMU_LOG_CANDIDATES = {
    "accel_x": (("imu.acc.x", "identity"), ("acc.x", "identity")),
    "accel_y": (("imu.acc.y", "identity"), ("acc.y", "identity")),
    "accel_z": (("imu.acc.z", "identity"), ("acc.z", "identity")),
    "gyro_x": (("imu.gyro.x", "identity"), ("gyro.x", "identity")),
    "gyro_y": (("imu.gyro.y", "identity"), ("gyro.y", "identity")),
    "gyro_z": (("imu.gyro.z", "identity"), ("gyro.z", "identity")),
    "temp_c": (("imu.temp", "identity"), ("temp", "identity")),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QuadcopterPID on Crazyflie and log flight data."
    )
    parser.add_argument("--uri", default=URI)
    parser.add_argument("--dt", type=float, default=DT_CTRL)
    parser.add_argument(
        "--trajectory",
        choices=["hover", "point", "square", "circle", "sine", "helix", "flight_mission"],
        default="flight_mission",
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
    parser.add_argument(
        "--mission-z",
        type=float,
        default=0.6,
        help="Target altitude [m] for the built-in FlightMission.",
    )
    parser.add_argument(
        "--ground-z",
        type=float,
        default=0.05,
        help="Ground/touchdown altitude [m] used by the built-in FlightMission.",
    )
    parser.add_argument(
        "--takeoff-duration",
        type=float,
        default=3.0,
        help="Takeoff phase duration [s] for the built-in FlightMission.",
    )
    parser.add_argument(
        "--hover-duration",
        type=float,
        default=5.0,
        help="Hover phase duration [s] for the built-in FlightMission.",
    )
    parser.add_argument(
        "--landing-duration",
        type=float,
        default=3.0,
        help="Landing phase duration [s] for the built-in FlightMission.",
    )
    parser.add_argument(
        "--startup-hover-z",
        type=float,
        default=0.18,
        help="Low hover altitude [m] reached before the main mission takeoff.",
    )
    parser.add_argument(
        "--startup-hover-duration",
        type=float,
        default=1.0,
        help="Time [s] to hold the low hover before climbing to mission-z.",
    )
    parser.add_argument(
        "--startup-takeoff-duration",
        type=float,
        default=1.5,
        help="Time [s] to climb from ground to startup-hover-z.",
    )
    parser.add_argument(
        "--estimator-stable-time",
        type=float,
        default=0.5,
        help="Require a full estimator packet for this long before starting control.",
    )
    parser.add_argument(
        "--startup-ramp-scale",
        type=float,
        default=0.65,
        help="Fraction of hover thrust used for the initial open-loop spin-up ramp.",
    )
    parser.add_argument(
        "--startup-ramp-duration",
        type=float,
        default=0.6,
        help="Duration [s] of the gentle initial spin-up ramp.",
    )
    args = parser.parse_args()
    if args.dt <= 0.0:
        parser.error("--dt must be > 0")
    if args.kill_z <= 0.0:
        parser.error("--kill-z must be > 0")
    if args.kill_xy <= 0.0:
        parser.error("--kill-xy must be > 0")
    if args.hover_thrust <= 0.0:
        parser.error("--hover-thrust must be > 0")
    if args.mission_z <= 0.0:
        parser.error("--mission-z must be > 0")
    if args.ground_z < 0.0:
        parser.error("--ground-z must be >= 0")
    if args.takeoff_duration <= 0.0 or args.hover_duration <= 0.0 or args.landing_duration <= 0.0:
        parser.error("--takeoff-duration, --hover-duration, and --landing-duration must be > 0")
    if args.startup_hover_z < 0.0:
        parser.error("--startup-hover-z must be >= 0")
    if args.startup_hover_duration < 0.0:
        parser.error("--startup-hover-duration must be >= 0")
    if args.startup_takeoff_duration <= 0.0:
        parser.error("--startup-takeoff-duration must be > 0")
    if args.estimator_stable_time < 0.0:
        parser.error("--estimator-stable-time must be >= 0")
    if not (0.0 < args.startup_ramp_scale <= 1.0):
        parser.error("--startup-ramp-scale must be in (0, 1]")
    if args.startup_ramp_duration <= 0.0:
        parser.error("--startup-ramp-duration must be > 0")
    return args


def u1_newtons_to_thrust_int(U1_N: float, m: float, g: float, hover_thrust: float) -> int:
    scale = float(U1_N / (m * g))
    thrust = int(np.clip(hover_thrust * scale, MIN_THRUST, MAX_THRUST))
    return thrust


def build_takeoff_hover_land_mission(args) -> FlightMission:
    mission = FlightMission(default_z=args.mission_z, ground_z=args.ground_z)
    startup_hover_z = min(float(args.startup_hover_z), float(args.mission_z))
    if startup_hover_z > float(args.ground_z) + 1e-3:
        mission.add_takeoff(duration=args.startup_takeoff_duration, target_z=startup_hover_z)
        if args.startup_hover_duration > 0.0:
            mission.add_hover(duration=args.startup_hover_duration, z=startup_hover_z)
    mission.add_takeoff(duration=args.takeoff_duration, target_z=args.mission_z)
    mission.add_hover(duration=args.hover_duration, z=args.mission_z)
    mission.add_landing(duration=args.landing_duration, ground_z=args.ground_z)
    return mission


def make_reference(
    args,
) -> Tuple[Callable[[float], Tuple[np.ndarray, np.ndarray]], str, Optional[float], Optional[dict]]:
    traj_name = args.trajectory
    mission_file = args.mission
    if mission_file:
        if MissionPlannerTrajectory is None:
            raise RuntimeError(
                "MissionPlannerTrajectory requires pyproj. Install with: pip install pyproj"
            )
        mission_path = os.path.abspath(mission_file)
        mission = MissionPlannerTrajectory(mission_path)

        def mission_ref(t: float):
            return mission(t)

        return (
            mission_ref,
            f"mission:{os.path.basename(mission_path)}",
            float(mission.summary()["base_mission_end_time"]),
            mission.summary(),
        )

    if traj_name == "flight_mission":
        mission = build_takeoff_hover_land_mission(args)

        def mission_ref(t: float):
            return mission(t)

        return mission_ref, "flight_mission:takeoff-hover-land", float(mission.total_time), mission.summary()

    if traj_name == "hover":
        return lambda t: path.hover_traj(t, pos=(0.0, 0.0, 1.0)), "hover", None, None
    if traj_name == "point":
        return lambda t: path.point_traj((0.0, 0.0, 1.0)), "point(0,0,1)", None, None
    if traj_name == "square":
        return path.square_traj, "square", None, None
    if traj_name == "circle":
        return path.circle_traj, "circle", None, None
    if traj_name == "sine":
        return path.sine_traj, "sine", None, None
    return path.helix_traj, "helix", None, None


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
        "motor_map_force_1_n",
        "motor_map_force_2_n",
        "motor_map_force_3_n",
        "motor_map_force_4_n",
        "motor_map_omega_1_rad_s",
        "motor_map_omega_2_rad_s",
        "motor_map_omega_3_rad_s",
        "motor_map_omega_4_rad_s",
        "safety_ok",
    ]
    f = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    return f, writer


def write_meta(
    meta_path: Path,
    start_time: datetime,
    args,
    reference_name: str,
    reference_summary: Optional[dict],
    csv_path: Path,
    control_vars: dict,
    imu_vars: dict,
    motor_map_info: dict,
):
    payload = {
        "start_time": start_time.isoformat(),
        "trajectory": reference_name,
        "uri": args.uri,
        "dt": args.dt,
        "hover_thrust": args.hover_thrust,
        "kill_z": args.kill_z,
        "kill_xy": args.kill_xy,
        "duration_s": args.duration,
        "reference_summary": reference_summary,
        "log_csv": str(csv_path),
        "control_variables": control_vars,
        "imu_variables": imu_vars,
        "motor_map": motor_map_info,
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
    mapped_motor_outputs: dict,
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
            "motor_map_force_1_n": float(mapped_motor_outputs["motor_forces"][0]),
            "motor_map_force_2_n": float(mapped_motor_outputs["motor_forces"][1]),
            "motor_map_force_3_n": float(mapped_motor_outputs["motor_forces"][2]),
            "motor_map_force_4_n": float(mapped_motor_outputs["motor_forces"][3]),
            "motor_map_omega_1_rad_s": float(mapped_motor_outputs["omega_cmd"][0]),
            "motor_map_omega_2_rad_s": float(mapped_motor_outputs["omega_cmd"][1]),
            "motor_map_omega_3_rad_s": float(mapped_motor_outputs["omega_cmd"][2]),
            "motor_map_omega_4_rad_s": float(mapped_motor_outputs["omega_cmd"][3]),
            "safety_ok": int(bool(safety_ok)),
        }
    )


def set_flightmode_rate(cf: Crazyflie):
    cf.param.set_value("flightmode.stabModeRoll", "0")
    cf.param.set_value("flightmode.stabModePitch", "0")
    cf.param.set_value("flightmode.stabModeYaw", "0")
    cf.param.set_value("flightmode.althold", "0")
    cf.param.set_value("flightmode.poshold", "0")


def _toc_element(cf: Crazyflie, var: str):
    return cf.log.toc.get_element_by_complete_name(var)


def _add_log_var_if_available(
    cf: Crazyflie,
    cfg: LogConfig,
    var: str,
    dtype: Optional[str] = None,
) -> bool:
    element = _toc_element(cf, var)
    if element is None:
        return False

    cfg.add_variable(var, dtype or element.ctype)
    return True


def _pick_first_available_var(
    cf: Crazyflie,
    cfg: LogConfig,
    candidates: list[str],
    dtype: Optional[str] = None,
) -> Optional[str]:
    for var in candidates:
        if _add_log_var_if_available(cf, cfg, var, dtype):
            return var
    return None


def _convert_logged_value(raw_value, mode: str) -> float:
    value = float(raw_value)
    if mode == "identity":
        return value
    if mode == "deg_to_rad":
        return float(np.deg2rad(value))
    if mode == "neg_deg_to_rad":
        return float(-np.deg2rad(value))
    if mode == "mrad_to_rad":
        return 1.0e-3 * value
    raise ValueError(f"Unknown conversion mode: {mode}")


def _resolve_log_field(
    cf: Crazyflie,
    cfg: LogConfig,
    label: str,
    candidates,
    *,
    required: bool,
    excluded_vars: Optional[set[str]] = None,
):
    excluded_vars = set() if excluded_vars is None else set(excluded_vars)
    attempted = []
    for var_name, converter in candidates:
        attempted.append(var_name)
        if var_name in excluded_vars:
            continue
        if _add_log_var_if_available(cf, cfg, var_name):
            return {
                "var": var_name,
                "converter": converter,
            }
    if required:
        raise RuntimeError(
            f"Missing required log field '{label}'. Tried: {', '.join(attempted)}"
        )
    return None


def _extract_logged_fields(
    latest: dict,
    field_map: dict,
    *,
    strict: bool = True,
    missing_value=None,
) -> dict:
    extracted = {}
    for label, spec in field_map.items():
        if spec["var"] not in latest:
            if strict:
                raise KeyError(spec["var"])
            extracted[label] = missing_value
            continue
        extracted[label] = _convert_logged_value(latest[spec["var"]], spec["converter"])
    return extracted


def make_logconfs(cf: Crazyflie, period_ms: int):
    lg_state = LogConfig(name="state", period_in_ms=period_ms)
    lg_att = LogConfig(name="att", period_in_ms=period_ms)
    control_map = {}
    state_labels = {"x", "y", "z", "vx", "vy", "vz"}
    for label, candidates in CONTROL_LOG_CANDIDATES.items():
        target_cfg = lg_state if label in state_labels else lg_att
        control_map[label] = _resolve_log_field(
            cf,
            target_cfg,
            label,
            candidates,
            required=True,
        )

    used_vars = {spec["var"] for spec in control_map.values()}
    lg_imu = LogConfig(name="imu", period_in_ms=period_ms)
    imu_map = {}
    for label, candidates in IMU_LOG_CANDIDATES.items():
        resolved = _resolve_log_field(
            cf,
            lg_imu,
            label,
            candidates,
            required=False,
            excluded_vars=used_vars,
        )
        if resolved is not None:
            imu_map[label] = resolved

    if not imu_map:
        lg_imu = None

    log_configs = [lg_state, lg_att]
    if lg_imu is not None:
        log_configs.append(lg_imu)

    return log_configs, control_map, imu_map


def ramp_thrust(
    cf: Crazyflie,
    min_thrust: float,
    hover_thrust: float,
    dt: float,
    t_s: float = 1.0,
):
    sleep_dt = max(dt, 1e-3)
    steps = max(1, int(t_s / sleep_dt))
    for i in range(steps):
        u = (i + 1) / steps
        thrust = int(min_thrust + u * (hover_thrust - min_thrust))
        cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
        time.sleep(sleep_dt)


def land_and_stop(
    cf: Crazyflie,
    min_thrust: float,
    hover_thrust: float,
    dt: float,
    t_s: float = 1.0,
):
    sleep_dt = max(dt, 1e-3)
    steps = max(1, int(t_s / sleep_dt))
    for i in range(steps):
        u = 1.0 - (i + 1) / steps
        thrust = int(min_thrust + u * (hover_thrust - min_thrust))
        cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
        time.sleep(sleep_dt)
    cf.commander.send_stop_setpoint()


def extract_native_motor_outputs(ctrl: dict, controller: QuadcopterPID) -> dict:
    """Use the controller's native motor outputs directly."""
    if "omega_cmd" not in ctrl:
        raise KeyError("control dict must include 'omega_cmd'")
    omega_cmd = np.asarray(ctrl["omega_cmd"], dtype=float).copy()
    if omega_cmd.shape != (4,):
        raise ValueError("control dict 'omega_cmd' must have shape (4,)")
    if "motor_forces" in ctrl:
        motor_forces = np.asarray(ctrl["motor_forces"], dtype=float).copy()
        if motor_forces.shape != (4,):
            raise ValueError("control dict 'motor_forces' must have shape (4,)")
    else:
        motor_forces = float(controller.b) * np.square(omega_cmd)
    return {
        "channel_names": ("motor_1", "motor_2", "motor_3", "motor_4"),
        "omega_cmd": omega_cmd,
        "motor_forces": motor_forces,
        "thrust_cmd": float(ctrl["thrust_cmd"]),
    }


def main():
    cflib.crtp.init_drivers(enable_debug_driver=False)
    args = parse_args()

    ref_fn, ref_name, ref_total_time, ref_summary = make_reference(args)
    log_started, csv_path, meta_path = make_log_files(args.log_dir)
    period_ms = int(max(1, round(args.dt * 1000)))
    csv_file, csv_writer = start_csv_writer(csv_path)
    quad = QuadcopterPID(dt=args.dt)
    motor_map_info = {
        "enabled": False,
        "mode": "controller_native_outputs",
        "channel_names": ["motor_1", "motor_2", "motor_3", "motor_4"],
    }
    safety_ok = True
    airborne = False
    liftoff_z = max(float(args.ground_z) + 0.10, 0.12)

    print(f"[INFO] Log file: {csv_path}")
    print(f"[INFO] Meta file: {meta_path}")
    print(f"[INFO] Motor outputs: {motor_map_info['channel_names']}")

    try:
        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            cf = scf.cf
            try:
                cf.platform.send_arming_request(True)
                time.sleep(1.0)
            except Exception:
                pass
            set_flightmode_rate(cf)
            loggers, control_map, imu_map = make_logconfs(cf, period_ms)
            control_var_names = {label: spec["var"] for label, spec in control_map.items()}
            imu_var_names = {label: spec["var"] for label, spec in imu_map.items()}
            write_meta(
                meta_path,
                log_started,
                args,
                ref_name,
                ref_summary,
                csv_path,
                control_var_names,
                imu_var_names,
                motor_map_info,
            )
            print(f"[INFO] Control vars logged: {control_var_names}")
            if imu_var_names:
                print(f"[INFO] IMU vars logged: {imu_var_names}")
            if ref_summary is not None:
                print(f"[INFO] Reference summary: {ref_summary}")

            quad.reset()

            latest = {}
            last_send = 0.0
            last_log_t = None
            missing_report_t = 0.0
            startup_deadline = time.monotonic() + STARTUP_TIMEOUT_S
            estimator_ready_since = None
            mission_t0 = None
            startup_ramp_target = int(
                np.clip(args.hover_thrust * args.startup_ramp_scale, MIN_THRUST, MAX_THRUST)
            )

            try:
                with SyncLogger(scf, loggers) as logger:
                    max_run_time = args.duration
                    if max_run_time is None and ref_total_time is not None:
                        max_run_time = ref_total_time

                    for _, data, _ in logger:
                        latest.update(data)
                        now = time.monotonic()
                        ctrl_vars = {spec["var"] for spec in control_map.values()}
                        missing = [var for var in ctrl_vars if var not in latest]
                        if missing:
                            estimator_ready_since = None
                            if now > startup_deadline:
                                raise RuntimeError(
                                    "Missing required estimator fields after startup: "
                                    + ", ".join(sorted(missing))
                                )
                            if now - missing_report_t >= 1.0:
                                print(
                                    "[INFO] Waiting for estimator fields: "
                                    + ", ".join(sorted(missing))
                                )
                                missing_report_t = now
                            cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
                            continue

                        if mission_t0 is None:
                            if estimator_ready_since is None:
                                estimator_ready_since = now
                            stable_for = now - estimator_ready_since
                            if stable_for < args.estimator_stable_time:
                                if now - missing_report_t >= 1.0:
                                    print(
                                        f"[INFO] Waiting for stable estimator packet: "
                                        f"{stable_for:.2f}/{args.estimator_stable_time:.2f}s"
                                    )
                                    missing_report_t = now
                                cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
                                continue

                            print(
                                f"[INFO] Estimator stable for {stable_for:.2f}s. "
                                f"Starting gentle spin-up to {startup_ramp_target}."
                            )
                            ramp_thrust(
                                cf,
                                MIN_THRUST,
                                startup_ramp_target,
                                dt=args.dt,
                                t_s=args.startup_ramp_duration,
                            )
                            mission_t0 = time.monotonic()
                            last_send = mission_t0
                            last_log_t = mission_t0
                            continue

                        if now - last_send < args.dt:
                            continue
                        last_send = now

                        odom = _extract_logged_fields(latest, control_map, strict=True)
                        imu = (
                            _extract_logged_fields(
                                latest,
                                imu_map,
                                strict=False,
                                missing_value=None,
                            )
                            if imu_map
                            else {}
                        )
                        imu.setdefault("accel_x", None)
                        imu.setdefault("accel_y", None)
                        imu.setdefault("accel_z", None)
                        imu.setdefault("gyro_x", None)
                        imu.setdefault("gyro_y", None)
                        imu.setdefault("gyro_z", None)
                        imu.setdefault("temp_c", None)

                        x = np.array([odom["x"], odom["y"], odom["z"]], dtype=float)
                        v = np.array([odom["vx"], odom["vy"], odom["vz"]], dtype=float)
                        ang = np.array([odom["roll"], odom["pitch"], odom["yaw"]], dtype=float)
                        rate = np.array([odom["rate_x"], odom["rate_y"], odom["rate_z"]], dtype=float)
                        elapsed = now - mission_t0
                        pos_ref, vel_ref = ref_fn(elapsed)

                        if not airborne and x[2] >= liftoff_z:
                            airborne = True

                        touchdown_expected = bool(
                            ref_total_time is not None
                            and pos_ref[2] <= (args.ground_z + 0.08)
                        )

                        ground_impact = bool(
                            airborne
                            and x[2] < max(float(args.ground_z), 0.05)
                            and not touchdown_expected
                        )

                        if (
                            x[2] > args.kill_z
                            or np.linalg.norm(x[:2]) > args.kill_xy
                            or ground_impact
                        ):
                            print("[SAFETY] Kill condition triggered. Landing.")
                            safety_ok = False
                            break

                        if max_run_time is not None and elapsed >= max_run_time:
                            print("[INFO] Reference finished. Landing.")
                            break

                        quad.inject_external_state(x, v, ang, rate)
                        ctrl = quad.step(pos_ref, vel_ref, z_ref=float(pos_ref[2]))
                        mapped_motor_outputs = extract_native_motor_outputs(ctrl, quad)
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
                            mapped_motor_outputs,
                            safety_ok,
                        )
                        csv_file.flush()

            except KeyboardInterrupt:
                print("\n[INFO] Keyboard interrupt. Landing.")
            except Exception:
                print("\n[ERROR] Run failed. Landing and saving logs.")
                raise
            finally:
                try:
                    land_and_stop(
                        cf,
                        MIN_THRUST,
                        args.hover_thrust,
                        dt=args.dt,
                        t_s=1.0,
                    )
                except Exception as exc:
                    print(f"[WARN] Failed to send landing stop sequence: {exc}")
    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
