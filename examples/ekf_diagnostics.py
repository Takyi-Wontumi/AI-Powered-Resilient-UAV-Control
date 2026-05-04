"""
EKF Diagnostics — Physics-Based Q Derivation and Covariance Health Checks
===========================================================================
5-step diagnostic workflow (no full Monte Carlo needed until step 5):

  Step 1  Print physics-based Q values vs. old hand-tuned defaults
  Step 2  Covariance growth plot — open-loop predict-only (no simulation)
  Step 3  Sanity check: does 3σ bound grow at a reasonable rate?
  Step 4  Innovation (NIS) consistency from a single simulation run
  Step 5  Q-scale mini-sweep — coarse search, then run Monte Carlo on winner

Usage:
  python examples/ekf_diagnostics.py              # all steps
  python examples/ekf_diagnostics.py --step 2     # covariance growth only
  python examples/ekf_diagnostics.py --step 4     # NIS single-run only
  python examples/ekf_diagnostics.py --step 5     # Q-scale sweep only
  python examples/ekf_diagnostics.py --save-dir results/
"""

import argparse
import os
import sys
from pathlib import Path

# Force UTF-8 on Windows terminals (cp1252 can't encode Greek/box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from phoenix_drone_simulation.envs.control import AttitudeRate
from phoenix_drone_simulation.envs.followpath_dropout_mission import (
    DroneFollowPathDropoutMissionEnv,
)
from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.quadcopter_ekf import QuadcopterEKF, PhoenixEKFAdapter, STATE_DIM
from AI_UAV_Tests.trajectories_library import FlightMission

FIG_DPI   = 150
C_PHYS    = "#1565C0"   # blue  — physics-based Q
C_OLD     = "#C62828"   # red   — old hand-tuned Q
C_GOOD    = "#2E7D32"   # green — in-bounds
C_WARN    = "#F57F17"   # amber — out-of-bounds


# -- helpers --------------------------------------------------------------------

def _save(fig, save_dir: str | None, name: str):
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / name
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  → {out}")


def _build_mission() -> FlightMission:
    m = FlightMission(default_z=1.0, ground_z=0.0)
    m.add_takeoff(duration=3.0, target_z=1.0)
    m.add_circle(duration=12.0, radius=0.5, period=12.0, z=1.0, center_xy=(0.0, 0.0))
    m.add_hover(duration=2.0, z=1.0)
    m.add_landing(duration=6.0, ground_z=0.055)
    return m


def _make_env():
    mission = _build_mission()
    env = DroneFollowPathDropoutMissionEnv(
        trajectory_fn=mission,
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode=None,
        observation_noise=1.0,
    )
    env.drone.control = AttitudeRate(bc=env.bc, drone=env.drone, time_step=env.TIME_STEP)
    return env, mission


def _thrust_to_action(U1: float, mass: float, g: float = 9.81) -> float:
    return float(np.clip((U1 / (mass * g) - 0.9) / 0.4, -1.0, 1.0))


def _physics_q_diag(dt: float, sigma_a: float = 3.0, sigma_alpha: float = 30.0) -> list:
    q_pos  = 0.25 * dt**4 * sigma_a**2
    q_vel  = dt**2        * sigma_a**2
    q_att  = 0.25 * dt**4 * sigma_alpha**2
    q_rate = dt**2        * sigma_alpha**2
    return [q_pos]*3 + [q_vel]*3 + [q_att]*3 + [q_rate]*3


_OLD_Q_DIAG = [
    5e-6, 5e-6, 1e-4,
    5e-4, 5e-4, 5e-3,
    5e-6, 5e-6, 5e-6,
    5e-4, 5e-4, 5e-4,
]

STATE_LABELS = ["x", "y", "z", "vx", "vy", "vz", "φ", "θ", "ψ", "p", "q", "r"]
GROUPS = {
    "Position [m²]":   slice(0, 3),
    "Velocity [m²/s²]": slice(3, 6),
    "Attitude [rad²]": slice(6, 9),
    "Rates [rad²/s²]": slice(9, 12),
}


# ==============================================================================
# STEP 1 — Physics-based Q values
# ==============================================================================

def step1_print_q(dt: float = 0.002, save_dir: str | None = None):
    """Print physics-based Q vs old hand-tuned Q and produce a bar chart."""
    phys = _physics_q_diag(dt)
    old  = _OLD_Q_DIAG

    print("\n-- Step 1: Physics-Based Q vs Old Hand-Tuned Q --")
    print(f"  dt={dt}s   σ_a=3.0 m/s²   σ_α=30 rad/s²")
    print(f"  {'State':<8}  {'Physics':>14}  {'Old (tuned)':>14}  {'Ratio':>8}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*8}")
    for i, lbl in enumerate(STATE_LABELS):
        ratio = phys[i] / old[i] if old[i] != 0 else float("inf")
        flag  = "  ← SAME" if 0.5 < ratio < 2.0 else ""
        print(f"  {lbl:<8}  {phys[i]:14.3e}  {old[i]:14.3e}  {ratio:8.2f}x{flag}")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, (grp, sl) in zip(axes, GROUPS.items()):
        lbls = STATE_LABELS[sl]
        pv   = np.array(phys[sl])
        ov   = np.array(old[sl])
        x    = np.arange(len(lbls))
        ax.bar(x - 0.2, pv, 0.38, label="Physics-based", color=C_PHYS, alpha=0.85)
        ax.bar(x + 0.2, ov, 0.38, label="Old (tuned)",   color=C_OLD,  alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(lbls)
        ax.set_yscale("log")
        ax.set_title(grp, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        if sl.start == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Step 1 — Q diagonal: physics-derived vs hand-tuned\n"
                 "Physics model: CWNA with σ_a=3.0 m/s², σ_α=30 rad/s²",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "diag1_q_comparison.png")
    return fig


# ==============================================================================
# STEP 2 — Covariance growth (open-loop, no simulation)
# ==============================================================================

def step2_covariance_growth(
    dt: float = 0.002,
    T: float = 5.0,
    save_dir: str | None = None,
):
    """
    Propagate P forward using a simple kinematic F (no measurements).
    Shows how fast uncertainty grows under both Q options.
    Target: smooth growth, NOT flat, NOT exploding.
    """
    print("\n-- Step 2: Open-Loop Covariance Growth --")

    N = int(T / dt)
    t = np.arange(N) * dt

    F = np.eye(STATE_DIM)
    F[0, 3] = F[1, 4] = F[2, 5] = dt   # pos ← vel
    F[6, 9] = F[7, 10] = F[8, 11] = dt  # att ← rate

    def _propagate(Q_diag, P0_diag):
        P   = np.diag(P0_diag)
        Q   = np.diag(Q_diag)
        log = []
        for _ in range(N):
            P = F @ P @ F.T + Q
            log.append(3.0 * np.sqrt(np.maximum(np.diag(P), 0.0)))
        return np.array(log)    # (N, 12)

    P0 = [0.05]*3 + [0.1]*3 + [0.05]*3 + [0.1]*3
    phys_q = _physics_q_diag(dt)
    phys_s = _propagate(phys_q, P0)
    old_s  = _propagate(_OLD_Q_DIAG, P0)

    print(f"  After T={T}s  |  Physics-Q 3σ_pos={phys_s[-1, 0]:.4f} m  "
          f"|  Old-Q 3σ_pos={old_s[-1, 0]:.4f} m")
    print(f"  After T={T}s  |  Physics-Q 3σ_vel={phys_s[-1, 3]:.4f} m/s  "
          f"|  Old-Q 3σ_vel={old_s[-1, 3]:.4f} m/s")

    groups = [
        ("Position 3σ [m]",     slice(0, 3), ["x", "y", "z"]),
        ("Velocity 3σ [m/s]",   slice(3, 6), ["vx", "vy", "vz"]),
        ("Attitude 3σ [rad]",   slice(6, 9), ["φ", "θ", "ψ"]),
        ("Body rates 3σ [r/s]", slice(9, 12), ["p", "q", "r"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    styles = ["-", "--", ":"]

    for ax, (title, sl, lbls) in zip(axes, groups):
        for j, (lbl, sty) in enumerate(zip(lbls, styles)):
            ax.plot(t, phys_s[:, sl.start + j], color=C_PHYS, ls=sty,
                    lw=1.4, label=f"Physics {lbl}")
            ax.plot(t, old_s[:, sl.start + j],  color=C_OLD,  ls=sty,
                    lw=1.4, label=f"Old {lbl}")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("3σ bound")
        ax.grid(alpha=0.3)
        if sl.start == 0:
            ax.legend(fontsize=7, ncol=2)

    for ax in axes[2:]:
        ax.set_xlabel("Time [s]")

    fig.suptitle("Step 2 — Open-Loop Covariance Growth (no GPS updates)\n"
                 "Good: smooth growth.  Bad: flat (Q too small) or instant explosion (Q too big).",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "diag2_covariance_growth.png")
    return fig


# ==============================================================================
# STEP 3 & 4 — Single-run NIS + NEES consistency check
# ==============================================================================

def _run_single_trial(q_diag=None, seed: int = 0, label: str = "EKF"):
    """Run one episode, return time/NIS/NEES/pos_err logs."""
    np.random.seed(seed)
    env, mission = _make_env()
    env.reset()

    inner_ekf = QuadcopterEKF(dt=env.TIME_STEP,
                               process_noise_diag=q_diag)  # None → physics default
    adapter = PhoenixEKFAdapter(dt=env.TIME_STEP, ekf=inner_ekf)
    adapter.reset(env.drone.xyz, env.drone.xyz_dot, env.drone.rpy, env.drone.rpy_dot)
    ekf: QuadcopterEKF = adapter.ekf

    quad = QuadcopterPID(dt=env.TIME_STEP)
    quad.reset()

    steps = int(mission.total_time / env.TIME_STEP)
    last_valid_ref = np.array([0.0, 0.0, 1.0], dtype=float)

    t_log, nis_log, nees_log, err_log, sigma_log = [], [], [], [], []

    for _ in range(steps):
        t = float(env.mission_time)
        pos_ref, vel_ref = env.current_reference()
        pos_ref = np.asarray(pos_ref, dtype=float)
        vel_ref = np.asarray(vel_ref, dtype=float)
        last_valid_ref = pos_ref.copy()

        est = ekf.as_dict()
        quad.inject_external_state(est["x"], est["v"], est["ang"], est["rate"])
        ctrl = quad.step(pos_ref, vel_ref, z_ref=float(pos_ref[2]))

        action = np.zeros(4, dtype=np.float32)
        action[0]   = _thrust_to_action(ctrl["thrust_cmd"], quad.m, quad.g)
        action[1:4] = np.clip(ctrl["rates_des"] / (np.pi / 3.0), -1.0, 1.0)
        env.step(action)

        # Predict
        ekf.predict(omega=ctrl["omega_cmd"], dt=env.TIME_STEP)

        # Update + NIS
        z, H, R = ekf.build_measurement(
            position=env.drone.xyz,
            velocity=env.drone.xyz_dot,
            attitude=env.drone.rpy,
            rates=env.drone.rpy_dot,
        )
        innov = ekf.innovation(z, H)
        S     = H @ ekf.P @ H.T + R
        nis   = float(innov @ np.linalg.solve(S, innov)) / z.size
        ekf.update(z, H, R)

        # NEES
        x_true = np.concatenate([
            env.drone.xyz, env.drone.xyz_dot,
            env.drone.rpy, env.drone.rpy_dot,
        ])
        e    = ekf.x - x_true
        nees = float(e @ np.linalg.solve(ekf.P + 1e-12 * np.eye(STATE_DIM), e))

        t_log.append(t)
        nis_log.append(nis)
        nees_log.append(nees)
        err_log.append(np.abs(ekf.position - env.drone.xyz).copy())
        sigma_log.append(3.0 * np.sqrt(np.maximum(np.diag(ekf.P)[:3], 0.0)))

    env.close()
    return {
        "label":  label,
        "t":      np.array(t_log),
        "nis":    np.array(nis_log),
        "nees":   np.array(nees_log),
        "err":    np.array(err_log),
        "sigma":  np.array(sigma_log),
    }


def step3_sanity_check(dt: float = 0.002, save_dir: str | None = None):
    """Step 3: verify 3σ bounds cover errors in a single run."""
    print("\n-- Step 3: Single-Run Sanity Check (physics Q) --")
    run = _run_single_trial(q_diag=None, seed=0, label="Physics Q")

    t     = run["t"]
    err   = run["err"]
    sigma = run["sigma"]
    lbls  = ["X", "Y", "Z"]

    covered = [(np.abs(err[:, i]) < sigma[:, i]).mean() * 100 for i in range(3)]
    print(f"  3σ coverage: x={covered[0]:.1f}%  y={covered[1]:.1f}%  z={covered[2]:.1f}%  "
          f"(target ≈ 99.7 %)")

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for i, ax in enumerate(axs):
        ax.fill_between(t, 0, sigma[:, i] * 100, color=C_PHYS, alpha=0.25,
                        label=f"3σ EKF bound")
        ax.plot(t, err[:, i] * 100, color=C_OLD, lw=1.4, label=f"|error|")
        ax.set_ylabel(f"|error {lbls[i]}| [cm]")
        ax.set_title(f"{lbls[i]}-axis — 3σ coverage: {covered[i]:.1f}%", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("Time [s]")

    fig.suptitle("Step 3 — Single-Run Sanity Check: Does 3σ Bound Cover Errors?\n"
                 "Good: error inside shaded band (≥95% of steps).  Bad: error escapes.",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "diag3_sanity_check.png")
    return fig


def step4_nis_check(dt: float = 0.002, save_dir: str | None = None):
    """Step 4: NIS time-series — should hover near 1.0 (normalised by DOF)."""
    print("\n-- Step 4: Innovation Consistency (NIS/DOF) — Single Run --")

    phys_run = _run_single_trial(q_diag=None, seed=0, label="Physics Q")
    old_run  = _run_single_trial(q_diag=_OLD_Q_DIAG, seed=0, label="Old Q")

    for run in (phys_run, old_run):
        mn_nis  = run["nis"].mean()
        mn_nees = run["nees"].mean()
        nis_ok  = "GOOD" if 0.5 < mn_nis < 2.0 else "OUT OF RANGE"
        nees_ok = "GOOD" if 8.1 < mn_nees < 16.7 else "OUT OF RANGE"
        print(f"  [{run['label']:12s}]  NIS/DOF={mn_nis:7.3f} ({nis_ok:14s})"
              f"  NEES={mn_nees:8.2f} ({nees_ok})")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for run, color in ((phys_run, C_PHYS), (old_run, C_OLD)):
        ax1.plot(run["t"], run["nis"], color=color, lw=0.8, alpha=0.7,
                 label=run["label"])
        # 200-step rolling mean
        kernel = np.ones(200) / 200
        smoothed = np.convolve(run["nis"], kernel, mode="same")
        ax1.plot(run["t"], smoothed, color=color, lw=2.0)

    ax1.axhline(1.0, color="black", ls="--", lw=1.5, label="Expected = 1.0")
    ax1.axhline(2.0, color=C_WARN,  ls=":",  lw=1.2, label="Danger threshold = 2.0")
    ax1.set_ylabel("NIS / DOF")
    ax1.set_title("NIS/DOF — should stay near 1.0 (thick = 200-step rolling mean)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, min(ax1.get_ylim()[1], 10.0))

    for run, color in ((phys_run, C_PHYS), (old_run, C_OLD)):
        ax2.plot(run["t"], run["nees"], color=color, lw=0.8, alpha=0.5,
                 label=run["label"])

    ax2.axhline(STATE_DIM, color="black", ls="--", lw=1.5,
                label=f"Expected NEES = {STATE_DIM}")
    ax2.set_ylabel("NEES")
    ax2.set_xlabel("Time [s]")
    ax2.set_title("NEES — should stay near 12 (1 DOF per state)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, min(ax2.get_ylim()[1], 500.0))

    fig.suptitle("Step 4 — Innovation (NIS) & Estimation Error (NEES) — Single Run\n"
                 "NIS/DOF ≈ 1 and NEES ≈ 12 → filter is consistent",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "diag4_nis_nees.png")
    return fig


# ==============================================================================
# STEP 5 — Q-scale mini-sweep
# ==============================================================================

def step5_q_sweep(
    scales: list | None = None,
    seed: int = 0,
    save_dir: str | None = None,
):
    """
    Scale the physics-based Q diagonal by each factor, run one trajectory,
    report RMSE / mean-NIS / 3σ coverage.  Pick the best region, then run
    full Monte Carlo on that scale.
    """
    if scales is None:
        scales = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

    print("\n-- Step 5: Q-Scale Mini-Sweep --")
    print(f"  Scales tested: {scales}")
    print(f"  {'Scale':>8}  {'RMSE [cm]':>10}  {'NIS/DOF':>9}  {'NEES':>8}  "
          f"{'3σ x':>7}  {'3σ y':>7}  {'3σ z':>7}  Notes")
    print(f"  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*25}")

    phys_base = np.array(_physics_q_diag(dt=0.002))
    results   = []

    for scale in scales:
        q_scaled = list(phys_base * scale)
        run = _run_single_trial(q_diag=q_scaled, seed=seed, label=f"x{scale}")

        rmse = float(np.sqrt(np.mean(run["err"]**2)) * 100)
        nis  = float(run["nis"].mean())
        nees = float(run["nees"].mean())
        cov3 = [(np.abs(run["err"][:, i]) < run["sigma"][:, i]).mean() * 100 for i in range(3)]

        nis_ok  = "NIS:ok" if 0.5 < nis  < 2.0  else ("NIS:lo" if nis < 0.5 else "NIS:hi")
        nees_ok = "NEES:ok" if 8.1 < nees < 16.7 else ("NEES:lo" if nees < 8.1 else "NEES:hi")
        cov_ok  = "cov:ok" if all(c > 95.0 for c in cov3) else "cov:lo"
        note = f"{nis_ok} {nees_ok} {cov_ok}"

        print(f"  {scale:>8.1f}  {rmse:>10.3f}  {nis:>9.3f}  {nees:>8.2f}  "
              f"{cov3[0]:>6.1f}%  {cov3[1]:>6.1f}%  {cov3[2]:>6.1f}%  {note}")
        results.append((scale, rmse, nis, nees, cov3))

    # -- plot ------------------------------------------------------------------
    sc   = [r[0] for r in results]
    rm   = [r[1] for r in results]
    ns   = [r[2] for r in results]
    ne   = [r[3] for r in results]
    covs = [r[4] for r in results]

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax1, ax2, ax_nees, ax3 = axes

    ax1.semilogx(sc, rm, "o-", color=C_PHYS, lw=2)
    ax1.set_ylabel("Position RMSE [cm]")
    ax1.set_title("RMSE vs Q scale factor")
    ax1.grid(alpha=0.3)

    ax2.semilogx(sc, ns, "o-", color=C_PHYS, lw=2)
    ax2.axhline(1.0, color="black", ls="--", lw=1.2, label="NIS/DOF = 1.0 (ideal)")
    ax2.axhspan(0.5, 2.0, color=C_GOOD, alpha=0.12, label="Acceptable [0.5, 2.0]")
    ax2.set_ylabel("Mean NIS / DOF")
    ax2.set_title("NIS consistency vs Q scale factor")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    ax_nees.semilogx(sc, ne, "o-", color=C_OLD, lw=2)
    ax_nees.axhline(STATE_DIM, color="black", ls="--", lw=1.2, label=f"Expected = {STATE_DIM}")
    ax_nees.axhspan(8.1, 16.7, color=C_GOOD, alpha=0.12, label="95% CI [8.1, 16.7]")
    ax_nees.set_ylabel("Mean NEES")
    ax_nees.set_title("NEES consistency vs Q scale factor")
    ax_nees.legend(fontsize=9)
    ax_nees.grid(alpha=0.3)

    for i, (c, lbl) in enumerate(zip(zip(*covs), ["x", "y", "z"])):
        ax3.semilogx(sc, list(c), "o-", lw=1.6, label=f"3σ {lbl}")
    ax3.axhline(99.7, color="black", ls="--", lw=1.2, label="Ideal 99.7 %")
    ax3.axhline(95.0, color=C_WARN,  ls=":",  lw=1.2, label="Min acceptable 95 %")
    ax3.set_ylabel("3σ Coverage [%]")
    ax3.set_xlabel("Q scale factor")
    ax3.set_title("3σ covariance coverage vs Q scale factor")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    fig.suptitle("Step 5 — Q-Scale Mini-Sweep (single run per scale)\n"
                 "Find the sweet spot: low RMSE + NIS≈1.0 + 3σ coverage ≥ 95 %",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, save_dir, "diag5_q_sweep.png")
    return fig, results


# ==============================================================================
# entry point
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step",     type=int, default=0,
                   help="Run only this step (1–5). Default: run all.")
    p.add_argument("--save-dir", type=str, default=None, metavar="DIR",
                   help="Directory to save PNG figures.")
    p.add_argument("--show",     action="store_true",
                   help="Open interactive matplotlib window.")
    p.add_argument("--scales",   type=float, nargs="+", default=None,
                   help="Scale factors for step 5 (default: 0.1 0.5 1 5 10 50 100).")
    return p.parse_args()


def main():
    args = parse_args()

    if args.show:
        matplotlib.use("TkAgg")
        import importlib, matplotlib.pyplot as _p; importlib.reload(_p)

    save_dir = args.save_dir
    only     = args.step

    print(f"\n{'='*56}")
    print(f"  EKF Diagnostics — Physics-Based Q Workflow")
    print(f"{'='*56}")

    dt = 0.002

    if only in (0, 1):
        fig1 = step1_print_q(dt=dt, save_dir=save_dir)

    if only in (0, 2):
        fig2 = step2_covariance_growth(dt=dt, save_dir=save_dir)

    if only in (0, 3):
        print("\n  [Running simulation for step 3 — single trial ...]")
        fig3 = step3_sanity_check(dt=dt, save_dir=save_dir)

    if only in (0, 4):
        print("\n  [Running 2 simulation trials for step 4 ...]")
        fig4 = step4_nis_check(dt=dt, save_dir=save_dir)

    if only in (0, 5):
        print(f"\n  [Running {len(args.scales or [0.1,0.5,1,5,10,50,100])} "
              f"simulation trials for step 5 ...]")
        fig5, results = step5_q_sweep(scales=args.scales, save_dir=save_dir)

    print(f"\n{'='*56}")
    print("  Done. Check figures for Q selection guidance.")
    print(f"{'='*56}\n")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
