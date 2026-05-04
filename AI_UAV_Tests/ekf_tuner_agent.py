#!/usr/bin/env python3
"""
EKF Tuner Agent  (dual-knob: sigma_a for NEES, r_scale for NIS)
=================================================================
Targets (statistically consistent EKF):
    Mean NEES  <= 12.0           chi-squared expected value for 12-state filter
    NIS/DOF    in (0.95, 1.0]    innovation-consistent (S matches actual spread)

Two-knob Newton loop (corrected knob assignment)
-------------------------------------------------
  sigma_a  scales Q (process noise) -> controls NEES via P_ss.
    Large Q -> large P_ss -> small NEES.
    Newton: sigma_a *= sqrt(NEES / nees_target)
    NEES too high -> sigma_a increases (Q grows, P_ss grows, NEES falls).

  r_scale  scales R (measurement noise) -> controls NIS via S = P + R.
    Large R -> large S -> small NIS.  R must match actual sensor noise.
    Newton: r_scale *= NIS / nis_mid
    NIS too low (S too big vs actual innov) -> r_scale decreases.

The _R_BASE_DIAG values are set to match the physical sensor noise (SensorNoise
in phoenix_drone_simulation), so r_scale=1 is the physically correct baseline.
The loop corrects any mismatch between R_BASE and true noise.

Usage:
    python AI_UAV_Tests/ekf_tuner_agent.py
    python AI_UAV_Tests/ekf_tuner_agent.py --n-trials 8 --max-iter 20
    python AI_UAV_Tests/ekf_tuner_agent.py --sigma-start 10 --r-scale-start 1.0
"""

import argparse
import math
import os
import re
import sys
import time

import numpy as np
import scipy.optimize as opt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.followpath_dropout_mission import (
    DroneFollowPathDropoutMissionEnv,
)
from phoenix_drone_simulation.envs.sensors import SensorNoise
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import (
    QuadcopterEKF,
    STATE_DIM,
    _Q_EMPIRICAL_DIAG,    # MATLAB-derived Q diagonal from real Crazyflie + Lighthouse
    _R_EMPIRICAL_DIAG,    # MATLAB-derived R diagonal from same logs
    _Q_INFLATION,         # 1.2x safety factor
    _COV_FLOOR,           # 1e-6 epsilon regularization
)
from AI_UAV_Tests.trajectories_library import FlightMission

# ── targets ────────────────────────────────────────────────────────────────────
# Narrow acceptance bands centered on the theoretical chi-squared expectations.
NEES_LOW    = 11.97   # PASS requires NEES in [NEES_LOW, NEES_HIGH]
NEES_HIGH   = 12.05
NEES_MID    = 0.5 * (NEES_LOW + NEES_HIGH)
NIS_LOW     =  0.975  # PASS requires NIS/DOF in [NIS_LOW, NIS_HIGH]
NIS_HIGH    =  1.05
NIS_MID     =  math.sqrt(NIS_LOW * NIS_HIGH)   # geometric mean ≈ 1.012

MAX_ITER    = 30
N_EVAL      =  5      # MC trials per iteration
TOL_NEES    =  0.04   # convergence tolerance on NEES
DT_LOG      = 0.01    # dt at which the empirical Q was identified

_H = np.eye(STATE_DIM, dtype=float)


def _build_Q(q_scale, dt=0.002):
    """Anchor on the empirical (MATLAB-derived) Q diagonal.

    Q = q_base * q_scale * (dt / DT_LOG) * inflation + epsilon

    q_scale = 1.0  ->  exactly the real-Crazyflie noise level (recommended).
    The optimizer is bounded so it cannot collapse Q to zero (which would
    make P tiny and explode NEES).
    """
    return np.diag(
        _Q_EMPIRICAL_DIAG * q_scale * (dt / DT_LOG) * _Q_INFLATION
        + _COV_FLOOR
    )


def _build_R(r_scale):
    """Anchor on the empirical (Lighthouse-derived) R diagonal.

    R = r_base * r_scale + epsilon
    r_scale = 1.0  ->  exactly the real Lighthouse noise level.
    """
    return np.diag(_R_EMPIRICAL_DIAG * r_scale + _COV_FLOOR)


# ── env factory ────────────────────────────────────────────────────────────────

def _make_env():
    mission = FlightMission(default_z=1.0, ground_z=0.0)
    mission.add_takeoff(duration=3.0, target_z=1.0)
    mission.add_hover(duration=2.0, z=1.0)
    env = DroneFollowPathDropoutMissionEnv(
        trajectory_fn=mission,
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode=None,
        observation_noise=1.0,
    )
    env.drone.control = AttitudeRate(
        bc=env.bc, drone=env.drone, time_step=env.TIME_STEP
    )
    return env, mission


def _thrust_to_action(U1, mass, g=9.81):
    return float(np.clip((U1 / (mass * g) - 0.9) / 0.4, -1.0, 1.0))


# ── single trial ──────────────────────────────────────────────────────────────

def run_trial(sigma_a, r_scale, seed=0):
    """One no-dropout episode. Returns (mean_nees, mean_nis_dof)."""
    np.random.seed(seed)
    env, mission = _make_env()
    env.reset()
    dt = env.TIME_STEP
    Q  = _build_Q(sigma_a, dt=dt)
    R  = _build_R(r_scale)

    ekf   = QuadcopterEKF(dt=dt, dt_log=dt)
    ekf.Q = Q
    # Section 4 — initialize from first measurement to minimize innovation spike.
    z0 = np.concatenate([
        env.drone.xyz, env.drone.xyz_dot,
        env.drone.rpy, env.drone.rpy_dot,
    ]).astype(float)
    ekf.init_from_measurement(z0)

    quad      = QuadcopterPID(dt=dt)
    quad.reset()
    noise_gen = SensorNoise()

    nees_buf, nis_buf = [], []

    for _ in range(int(mission.total_time / dt)):
        pos_ref, vel_ref = env.current_reference()
        pos_ref = np.asarray(pos_ref, dtype=float)
        vel_ref = np.asarray(vel_ref, dtype=float)

        # Use GROUND TRUTH for control to isolate EKF accuracy from
        # closed-loop instability — we want to measure the filter's
        # tracking performance, not the controller's reaction to a bad EKF.
        quad.inject_external_state(
            env.drone.xyz, env.drone.xyz_dot,
            env.drone.rpy, env.drone.rpy_dot,
        )
        ctrl = quad.step(pos_ref, vel_ref, z_ref=float(pos_ref[2]))

        action = np.zeros(4, dtype=np.float32)
        action[0]   = _thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)
        env.step(action)

        n_pos, n_vel, n_att, n_rate, _ = noise_gen.add_noise(
            env.drone.xyz, env.drone.xyz_dot,
            env.drone.rpy, env.drone.rpy_dot,
            np.zeros(3, dtype=float), dt,
        )
        z = np.concatenate([n_pos, n_vel, n_att, n_rate])

        # Task 1 — feed the EKF the *actual* post-motor-lag state from PyBullet,
        # converted to the EKF's quadratic-thrust unit (u = x_pybullet^2).
        omega_actual = np.asarray(env.drone.x, dtype=float) ** 2
        ekf.predict(omega=omega_actual, dt=dt)

        # NIS from pre-update innovation
        innov = z - _H @ ekf.x
        S = _H @ ekf.P @ _H.T + R
        try:
            nis = float(innov @ np.linalg.solve(S, innov)) / STATE_DIM
            if math.isfinite(nis) and nis < 1e6:
                nis_buf.append(nis)
        except np.linalg.LinAlgError:
            pass

        ekf.update(z, _H, R)

        # NEES post-update vs ground truth
        x_true = np.concatenate([
            env.drone.xyz, env.drone.xyz_dot,
            env.drone.rpy, env.drone.rpy_dot,
        ])
        e = ekf.x - x_true
        try:
            nees = float(e @ np.linalg.solve(ekf.P + 1e-12*np.eye(STATE_DIM), e))
            if math.isfinite(nees) and 0 < nees < 1e6:
                nees_buf.append(nees)
        except np.linalg.LinAlgError:
            pass

    env.close()
    return (
        float(np.mean(nees_buf)) if nees_buf else float("nan"),
        float(np.mean(nis_buf))  if nis_buf  else float("nan"),
    )


def evaluate(sigma_a, r_scale, n_trials, seed_offset=0):
    nees_vals, nis_vals = [], []
    for i in range(n_trials):
        n, s = run_trial(sigma_a, r_scale, seed=seed_offset + i)
        nees_vals.append(n)
        nis_vals.append(s)
        print(f"      trial {i+1}/{n_trials}   NEES={n:8.2f}   NIS/DOF={s:.4f}")
    return float(np.nanmean(nees_vals)), float(np.nanmean(nis_vals))


# ── patch quadcopter_ekf.py ────────────────────────────────────────────────────

def patch_ekf_defaults(q_scale, r_scale, dt=0.002):
    """Write the tuned scales in symbolic (anchored) form so the MATLAB
    baseline is preserved and the multiplier is human-readable."""
    q_line = (f"        self.Q = np.diag(_Q_EMPIRICAL_DIAG * "
              f"{q_scale:.4f} * _Q_INFLATION + _COV_FLOOR)")
    r_line = (f"        self.R_default = np.diag(_R_EMPIRICAL_DIAG * "
              f"{r_scale:.4f} + _COV_FLOOR)")

    ekf_path = os.path.join(ROOT_DIR, "AI_UAV_Tests", "quadcopter_ekf.py")
    with open(ekf_path, "r", encoding="utf-8") as fh:
        src = fh.read()

    patched, n_q = re.subn(
        r"[ \t]*self\.Q\s*=\s*np\.diag\([^\n]+\)",
        q_line, src, count=1,
    )
    patched, n_r = re.subn(
        r"[ \t]*self\.R_default\s*=\s*np\.diag\([^\n]+\)",
        r_line, patched, count=1,
    )

    if n_q == 0 and n_r == 0:
        print("  [patch] WARNING: no matching lines found; file not modified.")
        return

    with open(ekf_path, "w", encoding="utf-8") as fh:
        fh.write(patched)

    print(f"  [patch] quadcopter_ekf.py updated  (q_scale={q_scale:.4f}  r_scale={r_scale:.4f})")
    print(f"          {q_line.strip()}")
    print(f"          {r_line.strip()}")
    if n_q:
        print(f"          self.Q        = np.diag([{q_str}])")
    if n_r:
        print(f"          self.R_default= np.diag([{r_str}])")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nees-low",      type=float, default=NEES_LOW,   metavar="N",
                        help=f"Lower edge of NEES PASS band (default {NEES_LOW})")
    parser.add_argument("--nees-high",     type=float, default=NEES_HIGH,  metavar="N",
                        help=f"Upper edge of NEES PASS band (default {NEES_HIGH})")
    parser.add_argument("--nis-low",       type=float, default=NIS_LOW,    metavar="N",
                        help=f"Lower edge of NIS/DOF PASS band (default {NIS_LOW})")
    parser.add_argument("--nis-high",      type=float, default=NIS_HIGH,   metavar="N",
                        help=f"Upper edge of NIS/DOF PASS band (default {NIS_HIGH})")
    parser.add_argument("--n-trials",      type=int,   default=N_EVAL,      metavar="N")
    parser.add_argument("--max-iter",      type=int,   default=MAX_ITER,    metavar="N",
                        help="Max objective evaluations for Nelder-Mead")
    parser.add_argument("--sigma-start",   type=float, default=1.0,         metavar="S",
                        help="Initial q_scale — multiplier on empirical Q (default 1.0)")
    parser.add_argument("--r-scale-start", type=float, default=1.0,         metavar="S",
                        help="Initial r_scale — multiplier on empirical R (default 1.0)")
    parser.add_argument("--sigma-min",     type=float, default=0.5,         metavar="S",
                        help="Lower bound on q_scale (default 0.5 — no less than half empirical)")
    parser.add_argument("--sigma-max",     type=float, default=5.0,         metavar="S",
                        help="Upper bound on q_scale (default 5.0 — no more than 5x empirical)")
    parser.add_argument("--r-scale-min",   type=float, default=0.5,         metavar="S",
                        help="Lower bound on r_scale (default 0.5)")
    parser.add_argument("--r-scale-max",   type=float, default=5.0,         metavar="S",
                        help="Upper bound on r_scale (default 5.0)")
    parser.add_argument("--beta-nis",      type=float, default=50.0,        metavar="B",
                        help="NIS weight in cost: cost = NEES_excess^2 + beta*(log10 NIS)^2")
    parser.add_argument("--patch",         action="store_true",
                        help="Write the tuned q_scale/r_scale into "
                             "quadcopter_ekf.py (default: keep MATLAB baseline locked)")
    parser.add_argument("--no-patch",      action="store_true",
                        help="Deprecated; patching is off by default now")
    args = parser.parse_args()

    nees_lo     = args.nees_low
    nees_hi     = args.nees_high
    nees_mid    = 0.5 * (nees_lo + nees_hi)
    nis_lo      = args.nis_low
    nis_hi      = args.nis_high
    nis_mid     = math.sqrt(nis_lo * nis_hi)
    sigma_min   = args.sigma_min
    sigma_max   = args.sigma_max
    rs_min      = args.r_scale_min
    rs_max      = args.r_scale_max
    beta_nis    = args.beta_nis

    # Bounds in log10-space — anchored on empirical baseline (q_scale=r_scale=1).
    log_lo = (math.log10(sigma_min), math.log10(rs_min))
    log_hi = (math.log10(sigma_max), math.log10(rs_max))

    # Clip starting point to bounds.
    sigma0 = float(np.clip(args.sigma_start,  sigma_min, sigma_max))
    rs0    = float(np.clip(args.r_scale_start, rs_min, rs_max))
    log_x0 = np.array([math.log10(sigma0), math.log10(rs0)])

    W = 84
    print(f"\n{'=' * W}")
    print(f"  EKF Tuner Agent  (Nelder-Mead in log10-space, anchored on empirical Q/R)")
    print(f"  PASS bands:  NEES in [{nees_lo:.2f}, {nees_hi:.2f}]   "
          f"NIS/DOF in [{nis_lo:.3f}, {nis_hi:.3f}]")
    print(f"  Cost:     dist(NEES, band)^2 + {beta_nis:.0f} * dist(log10 NIS, log10 band)^2")
    print(f"  Bounds:   q_scale in [{sigma_min:.2f}, {sigma_max:.2f}]   "
          f"r_scale in [{rs_min:.2f}, {rs_max:.2f}]   "
          f"(1.0 = MATLAB baseline)")
    print(f"  Start:    q_scale={sigma0:.4f}   r_scale={rs0:.4f}")
    print(f"  Trials/eval = {args.n_trials}   max_evals = {args.max_iter}")
    print(f"{'=' * W}")
    print(f"\n  {'Eval':>4}  {'q_scale':>10}  {'r_scale':>9}  "
          f"{'NEES':>9}  {'NIS/DOF':>9}  {'Cost':>9}  Status")
    print(f"  {'-'*4}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*22}")

    history = []  # list of (sigma_a, r_scale, nees, nis, cost, passed)

    def _band_dist(value, lo, hi):
        """Signed distance from a closed band: 0 inside, |excursion| outside."""
        if value < lo:
            return lo - value
        if value > hi:
            return value - hi
        return 0.0

    def cost(log_params):
        # Hard-clip into bounds.
        log_sa = float(np.clip(log_params[0], log_lo[0], log_hi[0]))
        log_rs = float(np.clip(log_params[1], log_lo[1], log_hi[1]))
        sa     = 10.0 ** log_sa
        rs     = 10.0 ** log_rs

        seed_offset = len(history) * args.n_trials
        t0 = time.time()
        mean_nees, mean_nis = evaluate(sa, rs, args.n_trials, seed_offset)
        elapsed = time.time() - t0

        # NEES penalty: distance from [nees_lo, nees_hi] band, normalized by mid.
        if not math.isfinite(mean_nees):
            nees_pen = 100.0
        else:
            nees_pen = _band_dist(mean_nees, nees_lo, nees_hi) / nees_mid

        # NIS penalty: log-symmetric, distance from [log nis_lo, log nis_hi] band.
        if not math.isfinite(mean_nis) or mean_nis <= 0:
            nis_pen = 6.0
        else:
            nis_pen = _band_dist(math.log10(mean_nis),
                                 math.log10(nis_lo),
                                 math.log10(nis_hi))

        c = nees_pen ** 2 + beta_nis * nis_pen ** 2

        nees_ok = math.isfinite(mean_nees) and nees_lo <= mean_nees <= nees_hi
        nis_ok  = math.isfinite(mean_nis)  and nis_lo  <= mean_nis  <= nis_hi
        passed  = nees_ok and nis_ok

        history.append((sa, rs, mean_nees, mean_nis, c, passed))

        nees_str = ("ok" if nees_ok else
                    (f"NEES {mean_nees:.2f}<{nees_lo:.2f}" if mean_nees < nees_lo
                     else f"NEES {mean_nees:.2f}>{nees_hi:.2f}"))
        nis_str  = ("ok" if nis_ok else
                    (f"NIS {mean_nis:.4f}<{nis_lo:.3f}" if mean_nis < nis_lo
                     else f"NIS {mean_nis:.4f}>{nis_hi:.3f}"))
        status   = "PASS" if passed else f"{nees_str}  {nis_str}"

        print(f"  {len(history):>4}  {sa:>10.4f}  {rs:>9.4f}  "
              f"{mean_nees:>9.2f}  {mean_nis:>9.4f}  {c:>9.4f}  "
              f"{status}  ({elapsed:.0f}s)")
        return c

    # Run Nelder-Mead in log-space. Use a wide explicit initial simplex
    # (~half a decade per vertex) so it actually explores the [0.01, 5/10]
    # box instead of collapsing at the start.  Cost is O(1e5) and noisy at
    # the O(1e3) level, so fatol must be loose.
    init_simplex = np.array([
        log_x0,
        np.clip(log_x0 + np.array([+0.6, +0.0]), log_lo, log_hi),
        np.clip(log_x0 + np.array([+0.0, +0.6]), log_lo, log_hi),
    ])
    bounds_log = list(zip(log_lo, log_hi))
    nm_options = {
        "maxfev":          args.max_iter,
        "xatol":           0.1,        # log-space: factor of ~1.26
        "fatol":           500.0,      # ~0.25% of typical cost magnitude
        "adaptive":        True,
        "initial_simplex": init_simplex,
    }
    try:
        result = opt.minimize(
            cost, log_x0, method="Nelder-Mead",
            bounds=bounds_log, options=nm_options,
        )
    except TypeError:
        result = opt.minimize(
            cost, log_x0, method="Nelder-Mead", options=nm_options,
        )

    # Pick best entry from history (lowest cost).
    best_idx = int(np.argmin([h[4] for h in history]))
    best_sigma_a, best_r_scale, best_nees, best_nis, best_cost, _ = history[best_idx]
    print(f"\n  Optimizer finished: {result.message}")
    print(f"  Best cost = {best_cost:.4f} at evaluation {best_idx + 1}.")

    # ── final report ──────────────────────────────────────────────────────────
    nees_pass = math.isfinite(best_nees) and nees_lo <= best_nees <= nees_hi
    nis_pass  = math.isfinite(best_nis)  and nis_lo  <= best_nis  <= nis_hi

    Q_fin = _build_Q(best_sigma_a)
    R_fin = _build_R(best_r_scale)
    q_str = ", ".join(f"{v:.4e}" for v in np.diag(Q_fin))
    r_str = ", ".join(f"{v:.4e}" for v in np.diag(R_fin))

    print(f"\n{'=' * W}")
    print(f"  TUNING RESULT")
    print(f"  {'-' * (W-4)}")
    print(f"  q_scale      = {best_sigma_a:.6f}")
    print(f"  r_scale      = {best_r_scale:.6f}")
    print(f"  Q diag       = [{q_str}]")
    print(f"  R diag       = [{r_str}]")
    print(f"  {'-' * (W-4)}")
    print(f"  Mean NEES    = {best_nees:.2f}   "
          f"{'PASS' if nees_pass else 'FAIL'}  "
          f"(band [{nees_lo:.2f}, {nees_hi:.2f}])")
    print(f"  Mean NIS/DOF = {best_nis:.4f}   "
          f"{'PASS' if nis_pass else 'FAIL'}  "
          f"(band [{nis_lo:.3f}, {nis_hi:.3f}])")
    print(f"  Overall      = {'PASS' if (nees_pass and nis_pass) else 'FAIL'}")
    print(f"{'=' * W}")

    if args.patch and not args.no_patch:
        print()
        patch_ekf_defaults(best_sigma_a, best_r_scale)
        print()
        print("  Confirm with:")
        print("    python examples/ekf_validation.py "
              "--n-trials 5 --no-dropout")
    else:
        print("\n  MATLAB baseline kept locked (q_scale=1.0, r_scale=1.0).")
        print(f"  Tuned candidate (not written): "
              f"q_scale={best_sigma_a:.4f}  r_scale={best_r_scale:.4f}")
        print(f"  To overwrite the file, re-run with --patch.")
    print()


if __name__ == "__main__":
    main()
