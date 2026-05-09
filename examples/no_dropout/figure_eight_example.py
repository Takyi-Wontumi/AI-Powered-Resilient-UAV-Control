#!/usr/bin/env python3
"""
Example: Figure-Eight Trajectory

Demonstrates how to use the new figure_eight_traj() and add_figure_eight() methods
from the Trajectories library.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from AI_UAV_Tests.core.trajectories_library import Trajectories, FlightMission


def example_1_raw_trajectory():
    """Example 1: Use the raw figure_eight_traj() static method."""
    print("\n=== Example 1: Raw Figure-Eight Trajectory ===")

    scale = 1.0      # Size of the figure-eight [m]
    z = 2.0          # Altitude [m]
    period = 10.0    # Time to complete one figure-eight [s]

    # Sample the trajectory at 10 time points
    t_samples = np.linspace(0, period, 10)

    for t in t_samples:
        pos, vel = Trajectories.figure_eight_traj(t, scale=scale, z=z, period=period)
        speed = np.linalg.norm(vel)
        print(f"  t={t:5.2f}s | pos=({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:5.2f}) m | speed={speed:6.3f} m/s")


def example_2_mission_with_figure_eight():
    """Example 2: Create a flight mission with figure-eight phase."""
    print("\n=== Example 2: Flight Mission with Figure-Eight ===")

    # Create a mission
    mission = FlightMission(default_z=2.0, ground_z=0.0, auto_takeoff=True, auto_takeoff_duration=3.0)

    # Add phases
    mission.add_takeoff(duration=3.0, target_z=2.0)
    mission.add_figure_eight(
        duration=20.0,
        scale=1.5,
        period=10.0,
        z=2.0,
        center_xy=(0.0, 0.0)
    )
    mission.add_hover(duration=3.0, z=2.0)
    mission.add_landing(duration=3.0, ground_z=0.0)

    # Print mission summary
    summary = mission.summary()
    print(f"  Mission total time: {summary['total_time']:.1f} s")
    print(f"  Number of phases: {summary['num_phases']}")
    print(f"  Phases:")
    for phase in summary['phases']:
        print(f"    - {phase['name']:15s} [{phase['t_start']:6.2f}s, {phase['t_end']:6.2f}s] ({phase['duration']:5.1f}s)")

    # Sample the mission
    print(f"\n  Sample trajectory points:")
    for t in np.linspace(0, summary['total_time'], 15):
        pos, vel = mission(t)
        speed = np.linalg.norm(vel)
        phase_name = mission.phase_name_at(t)
        print(f"    t={t:5.2f}s ({phase_name:12s}) | pos=({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:5.2f}) m | speed={speed:6.3f} m/s")


def example_3_parametric_exploration():
    """Example 3: Explore different figure-eight parameters."""
    print("\n=== Example 3: Parameter Exploration ===")

    z = 1.5
    t_mid = 2.5  # Sample at mid-period

    print(f"  (sampling at t={t_mid}s, z={z}m)")
    print(f"  {'scale':>8} {'period':>8} {'x':>10} {'y':>10} {'speed':>10}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

    for scale in [0.5, 1.0, 1.5, 2.0]:
        for period in [8.0, 10.0, 12.0]:
            pos, vel = Trajectories.figure_eight_traj(t_mid, scale=scale, z=z, period=period)
            speed = np.linalg.norm(vel)
            print(f"  {scale:8.1f} {period:8.1f} {pos[0]:10.3f} {pos[1]:10.3f} {speed:10.3f}")


if __name__ == "__main__":
    print("Figure-Eight Trajectory Examples")
    print("=" * 60)

    example_1_raw_trajectory()
    example_2_mission_with_figure_eight()
    example_3_parametric_exploration()

    print("\n" + "=" * 60)
    print("To visualize a mission with preflight_check():")
    print("  mission = FlightMission(...)")
    print("  mission.add_figure_eight(...)")
    print("  mission.preflight_check()  # Shows interactive visualization")
    print("\nDone!")
