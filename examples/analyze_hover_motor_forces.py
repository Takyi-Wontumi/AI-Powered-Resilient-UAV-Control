"""
Log the force produced by each motor while the standalone PID quadcopter
climbs to and hovers near 1 meter.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from AI_UAV_Tests.quadcopter_env import QuadcopterPID


def build_parser():
    parser = argparse.ArgumentParser(
        description="Analyze per-motor force required to hover at 1 meter."
    )
    parser.add_argument("--target-z", type=float, default=1.0, help="Target hover altitude [m].")
    parser.add_argument("--sim-time", type=float, default=8.0, help="Simulation duration [s].")
    parser.add_argument("--dt", type=float, default=0.002, help="Controller timestep [s].")
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print one terminal row every N control steps.",
    )
    parser.add_argument(
        "--steady-window",
        type=float,
        default=1.0,
        help="Continuous hover time window used for the final force average [s].",
    )
    parser.add_argument(
        "--hover-z-tol",
        type=float,
        default=0.05,
        help="Altitude tolerance for steady hover detection [m].",
    )
    parser.add_argument(
        "--hover-vz-tol",
        type=float,
        default=0.08,
        help="Vertical-speed tolerance for steady hover detection [m/s].",
    )
    parser.add_argument(
        "--hover-tilt-tol-deg",
        type=float,
        default=5.0,
        help="Roll/pitch tolerance for steady hover detection [deg].",
    )
    return parser


def validate_args(parser, args):
    if args.target_z <= 0.0:
        parser.error("--target-z must be > 0.")
    if args.sim_time <= 0.0:
        parser.error("--sim-time must be > 0.")
    if args.dt <= 0.0:
        parser.error("--dt must be > 0.")
    if args.log_every <= 0:
        parser.error("--log-every must be > 0.")
    if args.steady_window <= 0.0:
        parser.error("--steady-window must be > 0.")
    if args.hover_z_tol <= 0.0 or args.hover_vz_tol <= 0.0:
        parser.error("--hover-z-tol and --hover-vz-tol must be > 0.")
    if args.hover_tilt_tol_deg <= 0.0:
        parser.error("--hover-tilt-tol-deg must be > 0.")


def plot_xyz_history(t_log, pos_log, pos_ref):
    t_arr = np.asarray(t_log, dtype=float)
    pos_arr = np.vstack(pos_log)
    ref_arr = np.tile(np.asarray(pos_ref, dtype=float), (len(t_arr), 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["X [m]", "Y [m]", "Z [m]"]

    for idx, axis in enumerate(axes):
        axis.plot(t_arr, pos_arr[:, idx], label="actual")
        axis.plot(t_arr, ref_arr[:, idx], "--", label="reference")
        axis.set_ylabel(labels[idx])
        axis.grid(True)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Hover Position Response")
    plt.tight_layout()
    plt.show()


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    quad = QuadcopterPID(dt=args.dt)
    quad.reset()

    pos_ref = np.array([0.0, 0.0, args.target_z], dtype=float)
    vel_ref = np.zeros(3, dtype=float)

    steps = int(args.sim_time / quad.dt)
    steady_steps_needed = max(1, int(args.steady_window / quad.dt))
    hover_tilt_tol = np.deg2rad(args.hover_tilt_tol_deg)

    stable_force_window = []
    stable_time = 0.0
    last_ctrl = None
    t_log = []
    pos_log = []

    header = (
        f"{'step':>6}  {'t(s)':>7}  {'z(m)':>8}  {'vz(m/s)':>9}  {'total(N)':>10}  "
        f"{'m1(N)':>8}  {'m2(N)':>8}  {'m3(N)':>8}  {'m4(N)':>8}"
    )
    print("Hover force analysis started.")
    print(f"Target hover altitude: {args.target_z:.3f} m")
    print(header)
    print("-" * len(header))

    for step in range(steps):
        t = step * quad.dt
        ctrl = quad.step(pos_ref, vel_ref, z_ref=args.target_z)
        last_ctrl = ctrl
        t_log.append(t)
        pos_log.append(np.asarray(ctrl["x"], dtype=float).copy())

        motor_forces = np.asarray(ctrl["motor_forces"], dtype=float)
        total_force = float(np.sum(motor_forces))
        z = float(ctrl["x"][2])
        vz = float(ctrl["v"][2])
        tilt = float(np.linalg.norm(ctrl["ang"][:2]))
        z_err = args.target_z - z

        in_stable_hover = (
            abs(z_err) <= args.hover_z_tol
            and abs(vz) <= args.hover_vz_tol
            and tilt <= hover_tilt_tol
        )

        if in_stable_hover:
            stable_time += quad.dt
            stable_force_window.append(motor_forces.copy())
            if len(stable_force_window) > steady_steps_needed:
                stable_force_window.pop(0)
        else:
            stable_time = 0.0
            stable_force_window.clear()

        if step % args.log_every == 0 or step == steps - 1:
            print(
                f"{step:6d}  {t:7.3f}  {z:8.3f}  {vz:9.3f}  {total_force:10.5f}  "
                f"{motor_forces[0]:8.5f}  {motor_forces[1]:8.5f}  "
                f"{motor_forces[2]:8.5f}  {motor_forces[3]:8.5f}"
            )

    print("-" * len(header))
    theoretical_hover_force = quad.m * quad.g / 4.0
    print(f"Theoretical symmetric hover force per motor: {theoretical_hover_force:.5f} N")

    if stable_time >= args.steady_window and stable_force_window:
        avg_motor_forces = np.mean(np.vstack(stable_force_window), axis=0)
        avg_total_force = float(np.sum(avg_motor_forces))
        print(
            "Estimated steady hover motor forces over the last "
            f"{args.steady_window:.2f} s:"
        )
        print(
            "  "
            f"m1={avg_motor_forces[0]:.5f} N, "
            f"m2={avg_motor_forces[1]:.5f} N, "
            f"m3={avg_motor_forces[2]:.5f} N, "
            f"m4={avg_motor_forces[3]:.5f} N, "
            f"total={avg_total_force:.5f} N"
        )
    elif last_ctrl is not None:
        motor_forces = np.asarray(last_ctrl["motor_forces"], dtype=float)
        print(
            "Steady hover window was not reached within the simulation time. "
            "Last motor-force sample:"
        )
        print(
            "  "
            f"m1={motor_forces[0]:.5f} N, "
            f"m2={motor_forces[1]:.5f} N, "
            f"m3={motor_forces[2]:.5f} N, "
            f"m4={motor_forces[3]:.5f} N, "
            f"total={np.sum(motor_forces):.5f} N"
        )

    if t_log:
        plot_xyz_history(t_log, pos_log, pos_ref)


if __name__ == "__main__":
    main()
