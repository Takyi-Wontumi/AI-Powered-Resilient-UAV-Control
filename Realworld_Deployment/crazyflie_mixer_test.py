"""
Bench or tethered Crazyflie mixer test driven by QuadcopterPID.

This script uses the repo's QuadcopterPID to generate the rate/thrust commands
sent to the Crazyflie. Each test stage injects a synthetic state error into the
controller, runs one controller step, prints the predicted per-motor forces from
the controller mixer, and then sends the resulting command for a short duration.

This script supports a separate motor-mapping layer on top of `QuadcopterPID`.
The controller still computes body-rate and thrust commands, but users can pick
how those commands are allocated to four motor channels for diagnostics:
- `internal_plus` matches the controller's native `w1/w2/w3/w4` abstraction
- `crazyflie_x` matches the Crazyflie corner numbering you provided
- `--motor-map-spec` lets users define their own 4-motor layout

Use this before the full closed-loop flight test. Run with props removed or with
the vehicle firmly restrained.
"""

import argparse
import os
import sys
import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(THIS_FILE), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.motor_mapping import make_motor_map_layer


URI = "radio://0/80/2M/E7E7E7E7E7"
MIN_THRUST = 10001
MAX_THRUST = 60000
DEFAULT_HOVER_THRUST = 13000
DEFAULT_CMD_DT = 0.05

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a QuadcopterPID-based Crazyflie mixer preflight test."
    )
    parser.add_argument("--uri", default=URI, help="Crazyflie radio URI.")
    parser.add_argument(
        "--motor-map-preset",
        choices=["internal_plus", "crazyflie_x"],
        default="crazyflie_x",
        help="Named motor-allocation layer used for diagnostic per-motor outputs.",
    )
    parser.add_argument(
        "--motor-map-spec",
        default=None,
        help=(
            "Optional custom motor map overriding the preset. "
            "Formats: 'M1=FR,M2=RR,M3=RL,M4=FL' or "
            "'name:x_right:y_front:yaw_sign,...' for 4 channels."
        ),
    )
    parser.add_argument(
        "--hover-thrust",
        type=float,
        default=DEFAULT_HOVER_THRUST,
        help="Bench hover-thrust mapping used for N->Crazyflie thrust conversion.",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=1.0,
        help="Synthetic hover altitude used inside the PID state/reference.",
    )
    parser.add_argument(
        "--attitude-error-deg",
        type=float,
        default=5.0,
        help="Injected roll/pitch attitude error magnitude in degrees.",
    )
    parser.add_argument(
        "--yaw-error-deg",
        type=float,
        default=10.0,
        help="Injected yaw attitude error magnitude in degrees.",
    )
    parser.add_argument(
        "--max-rate-dps",
        type=float,
        default=80.0,
        help="Safety clamp applied to the controller's desired body rates.",
    )
    parser.add_argument(
        "--spinup-s",
        type=float,
        default=1.0,
        help="Ramp duration from zero to the PID hover-equivalent thrust.",
    )
    parser.add_argument(
        "--pulse-s",
        type=float,
        default=1.0,
        help="Duration of each PID-generated mixer test stage.",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=0.8,
        help="Neutral hover hold time between non-hover stages.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_CMD_DT,
        help="Command resend interval in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the PID-generated stage commands without connecting.",
    )
    args = parser.parse_args()

    if args.hover_thrust < MIN_THRUST or args.hover_thrust > MAX_THRUST:
        parser.error(f"--hover-thrust must be in [{MIN_THRUST}, {MAX_THRUST}]")
    if args.target_z <= 0.0:
        parser.error("--target-z must be > 0")
    if args.attitude_error_deg <= 0.0:
        parser.error("--attitude-error-deg must be > 0")
    if args.yaw_error_deg <= 0.0:
        parser.error("--yaw-error-deg must be > 0")
    if args.max_rate_dps <= 0.0:
        parser.error("--max-rate-dps must be > 0")
    if args.spinup_s <= 0.0:
        parser.error("--spinup-s must be > 0")
    if args.pulse_s <= 0.0:
        parser.error("--pulse-s must be > 0")
    if args.settle_s < 0.0:
        parser.error("--settle-s must be >= 0")
    if args.dt <= 0.0:
        parser.error("--dt must be > 0")

    return args


def set_flightmode_rate(cf):
    cf.param.set_value("flightmode.stabModeRoll", "0")
    cf.param.set_value("flightmode.stabModePitch", "0")
    cf.param.set_value("flightmode.stabModeYaw", "0")
    cf.param.set_value("flightmode.althold", "0")
    cf.param.set_value("flightmode.poshold", "0")


def u1_newtons_to_thrust_int(u1_n, mass, gravity, hover_thrust):
    scale = float(u1_n / (mass * gravity))
    return int(np.clip(hover_thrust * scale, MIN_THRUST, MAX_THRUST))


def send_for(cf, roll, pitch, yawrate, thrust, duration_s, dt):
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
        time.sleep(dt)


def ramp_thrust(cf, thrust_target, duration_s, dt):
    steps = max(1, int(duration_s / dt))
    for i in range(steps):
        u = float(i + 1) / float(steps)
        thrust = int(round(u * thrust_target))
        cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
        time.sleep(dt)


def stop_and_disarm(cf, dt):
    for _ in range(20):
        cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
        time.sleep(dt)
    cf.commander.send_stop_setpoint()


def build_stages(args):
    hover_z = float(args.target_z)
    att_err = np.deg2rad(float(args.attitude_error_deg))
    yaw_err = np.deg2rad(float(args.yaw_error_deg))

    base_pos = np.array([0.0, 0.0, hover_z], dtype=float)
    base_vel = np.zeros(3, dtype=float)
    base_rate = np.zeros(3, dtype=float)

    return [
        {
            "name": "hover_trim",
            "duration_s": max(args.settle_s, 0.5),
            "x": base_pos.copy(),
            "v": base_vel.copy(),
            "ang": np.zeros(3, dtype=float),
            "rate": base_rate.copy(),
            "pos_ref": base_pos.copy(),
            "vel_ref": base_vel.copy(),
        },
        {
            "name": "roll_pos",
            "duration_s": args.pulse_s,
            "x": base_pos.copy(),
            "v": base_vel.copy(),
            "ang": np.array([-att_err, 0.0, 0.0], dtype=float),
            "rate": base_rate.copy(),
            "pos_ref": base_pos.copy(),
            "vel_ref": base_vel.copy(),
        },
        {
            "name": "roll_neg",
            "duration_s": args.pulse_s,
            "x": base_pos.copy(),
            "v": base_vel.copy(),
            "ang": np.array([att_err, 0.0, 0.0], dtype=float),
            "rate": base_rate.copy(),
            "pos_ref": base_pos.copy(),
            "vel_ref": base_vel.copy(),
        },
        {
            "name": "pitch_pos",
            "duration_s": args.pulse_s,
            "x": base_pos.copy(),
            "v": base_vel.copy(),
            "ang": np.array([0.0, -att_err, 0.0], dtype=float),
            "rate": base_rate.copy(),
            "pos_ref": base_pos.copy(),
            "vel_ref": base_vel.copy(),
        },
        {
            "name": "pitch_neg",
            "duration_s": args.pulse_s,
            "x": base_pos.copy(),
            "v": base_vel.copy(),
            "ang": np.array([0.0, att_err, 0.0], dtype=float),
            "rate": base_rate.copy(),
            "pos_ref": base_pos.copy(),
            "vel_ref": base_vel.copy(),
        },
        {
            "name": "yaw_pos",
            "duration_s": args.pulse_s,
            "x": base_pos.copy(),
            "v": base_vel.copy(),
            "ang": np.array([0.0, 0.0, -yaw_err], dtype=float),
            "rate": base_rate.copy(),
            "pos_ref": base_pos.copy(),
            "vel_ref": base_vel.copy(),
        },
        {
            "name": "yaw_neg",
            "duration_s": args.pulse_s,
            "x": base_pos.copy(),
            "v": base_vel.copy(),
            "ang": np.array([0.0, 0.0, yaw_err], dtype=float),
            "rate": base_rate.copy(),
            "pos_ref": base_pos.copy(),
            "vel_ref": base_vel.copy(),
        },
    ]


def evaluate_stage(args, stage, motor_map_layer):
    quad = QuadcopterPID(dt=args.dt)
    quad.reset()
    quad.inject_external_state(stage["x"], stage["v"], stage["ang"], stage["rate"])
    ctrl = quad.step(stage["pos_ref"], stage["vel_ref"], z_ref=float(stage["pos_ref"][2]))
    mapped = motor_map_layer.map_control(ctrl)

    rates_des = np.asarray(ctrl["rates_des"], dtype=float)
    max_rate_rad = np.deg2rad(args.max_rate_dps)
    rates_des = np.clip(rates_des, -max_rate_rad, max_rate_rad)

    thrust_n = float(ctrl["thrust_cmd"])
    thrust_cf = u1_newtons_to_thrust_int(
        thrust_n,
        mass=quad.m,
        gravity=quad.g,
        hover_thrust=args.hover_thrust,
    )

    return {
        "name": stage["name"],
        "duration_s": float(stage["duration_s"]),
        "state_ang_deg": np.rad2deg(stage["ang"]),
        "state_rate_dps": np.rad2deg(stage["rate"]),
        "rates_des_rad_s": rates_des,
        "rates_des_dps": np.rad2deg(rates_des),
        "thrust_cmd_n": thrust_n,
        "thrust_cf": thrust_cf,
        "tau_cmd": np.asarray(ctrl["tau_cmd"], dtype=float),
        "motor_forces_n": np.asarray(mapped["motor_forces"], dtype=float),
        "omega_cmd": np.asarray(mapped["omega_cmd"], dtype=float),
        "channel_names": tuple(mapped["channel_names"]),
    }


def format_force_delta(force_delta, channel_names):
    parts = []
    for label, delta in zip(channel_names, force_delta):
        parts.append(f"{label}={delta:+.5f}N")
    return ", ".join(parts)


def print_stage(stage_cmd, hover_forces):
    print(
        f"{stage_cmd['name']:10s}  "
        f"thrust_cf={stage_cmd['thrust_cf']:5d}  "
        f"thrust_n={stage_cmd['thrust_cmd_n']:.5f}  "
        f"rates_dps={np.array2string(stage_cmd['rates_des_dps'], precision=2, suppress_small=True)}"
    )
    print(
        "  injected ang deg: "
        f"{np.array2string(stage_cmd['state_ang_deg'], precision=2, suppress_small=True)}"
    )
    print(
        "  motor forces N:   "
        f"{np.array2string(stage_cmd['motor_forces_n'], precision=5, suppress_small=True)}"
    )
    print(
        "  tau cmd Nm:       "
        f"{np.array2string(stage_cmd['tau_cmd'], precision=6, suppress_small=True)}"
    )
    print(
        "  delta vs hover:   "
        + format_force_delta(
            stage_cmd["motor_forces_n"] - hover_forces,
            stage_cmd["channel_names"],
        )
    )


def main():
    args = parse_args()
    stages = build_stages(args)
    quad = QuadcopterPID(dt=args.dt)
    motor_map_layer = make_motor_map_layer(
        quad,
        preset=args.motor_map_preset,
        spec=args.motor_map_spec,
    )
    stage_cmds = [evaluate_stage(args, stage, motor_map_layer) for stage in stages]
    hover_stage = stage_cmds[0]
    hover_forces = hover_stage["motor_forces_n"]

    print("Crazyflie mixer preflight from QuadcopterPID")
    print("Safety: remove props or hard-tether the vehicle before running.")
    print("Motor mapping layer: " + ", ".join(hover_stage["channel_names"]))
    print(f"URI: {args.uri}")
    print(f"PID hover-thrust mapping: {args.hover_thrust:.0f}")
    print(
        "Injected state errors: "
        f"roll/pitch={args.attitude_error_deg:.2f} deg, "
        f"yaw={args.yaw_error_deg:.2f} deg"
    )
    if args.motor_map_spec:
        print(f"Custom motor map spec: {args.motor_map_spec}")
    else:
        print(f"Motor map preset: {args.motor_map_preset}")
    print("Controller-generated stages:")
    for stage_cmd in stage_cmds:
        print_stage(stage_cmd, hover_forces)

    if args.dry_run:
        return

    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf
        cf.platform.send_arming_request(True)
        time.sleep(1.0)
        set_flightmode_rate(cf)

        print("Unlocking commander and ramping to PID hover-equivalent thrust...")
        send_for(cf, 0.0, 0.0, 0.0, 0, 0.25, args.dt)
        ramp_thrust(cf, hover_stage["thrust_cf"], args.spinup_s, args.dt)

        try:
            for idx, stage_cmd in enumerate(stage_cmds, start=1):
                print(
                    f"[{idx}/{len(stage_cmds)}] {stage_cmd['name']} "
                    f"rates_dps={np.array2string(stage_cmd['rates_des_dps'], precision=2, suppress_small=True)} "
                    f"thrust_cf={stage_cmd['thrust_cf']}"
                )
                send_for(
                    cf,
                    float(stage_cmd["rates_des_dps"][0]),
                    float(stage_cmd["rates_des_dps"][1]),
                    float(stage_cmd["rates_des_dps"][2]),
                    int(stage_cmd["thrust_cf"]),
                    float(stage_cmd["duration_s"]),
                    args.dt,
                )
                if idx not in (1, len(stage_cmds)) and args.settle_s > 0.0:
                    print(f"Settling at hover_trim for {args.settle_s:.2f}s")
                    send_for(
                        cf,
                        float(hover_stage["rates_des_dps"][0]),
                        float(hover_stage["rates_des_dps"][1]),
                        float(hover_stage["rates_des_dps"][2]),
                        int(hover_stage["thrust_cf"]),
                        args.settle_s,
                        args.dt,
                    )
        finally:
            print("Stopping motors...")
            stop_and_disarm(cf, args.dt)


if __name__ == "__main__":
    main()
