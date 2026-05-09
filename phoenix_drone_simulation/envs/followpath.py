"""
Author: Lawrence Wontumi (2025)
"""
import logging
import numpy as np
from phoenix_drone_simulation.envs.base import DroneBaseEnv
from AI_UAV_Tests.core.trajectories_library import Trajectories as path
import matplotlib.pyplot as plt


class DroneFollowPathEnv(DroneBaseEnv):
    """This is meant to train the drone to follow arbitrary trajectories
    (square, helix, circle, waypoints)."""

    def __init__(self, trajectory_fn=None, control_mode='PWM', log_errors=True, **kwargs):
      self.trajectory_fn = trajectory_fn
      self.done_dist_threshold = 0.3
      self.penalty_action = 1e-4
      self.penalty_velocity = 1e-4
      self.penalty_spin = 1e-4
      self.penalty_terminal = 100
      self.ARP = 1e-3

      # Logging
      self.log_errors = log_errors
      self.error_log = []

      # REQUIRED base parameters
      init_xyz = np.array([0.0, 0.0, 1.0])
      init_rpy = np.array([0.0, 0.0, 0.0])
      init_xyz_dot = np.zeros(3)
      init_rpy_dot = np.zeros(3)

      super().__init__(
         physics='PyBulletPhysics',
         control_mode=control_mode,
         drone_model='cf21x_bullet',
         observation_frequency=100,
         sim_freq=200,

         init_xyz=init_xyz,
         init_rpy=init_rpy,
         init_xyz_dot=init_xyz_dot,
         init_rpy_dot=init_rpy_dot,

         **kwargs
      )


    # ===============================================================
    # OBSERVATIONS
    # ===============================================================
    def compute_observation(self):
        """Return concatenated state + error vector."""

        t = self.iteration / self.SIM_FREQ
        pos_ref, vel_ref = self.trajectory_fn(t)
        self.target_pos = pos_ref

        pos_error = pos_ref - self.drone.xyz
        vel_error = vel_ref - self.drone.xyz_dot

        # log error
        if self.log_errors:
            self.error_log.append([t, *pos_error, np.linalg.norm(pos_error)])

        obs = np.concatenate([
            self.drone.xyz,
            self.drone.quaternion,
            self.drone.xyz_dot,
            self.drone.rpy_dot,
            pos_error
        ]).astype(np.float32)

        return obs

    # ===============================================================
    # REWARD
    # ===============================================================
    def compute_reward(self, action):
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)

        penalties = (
            self.penalty_action * np.linalg.norm(action)
            + self.penalty_velocity * np.linalg.norm(self.drone.xyz_dot)
            + self.penalty_spin * np.linalg.norm(self.drone.rpy_dot)
        )

        reward = -dist - penalties

        if self.compute_done():
            reward -= self.penalty_terminal

        return reward

    # ===============================================================
    # TERMINATION
    # ===============================================================
    def compute_done(self):
        return np.linalg.norm(self.drone.xyz - self.target_pos) > self.done_dist_threshold

    # ===============================================================
    # STATE ACCESS FOR CONTROLLER
    # ===============================================================
    def get_state(self):
        return {
            "pos": self.drone.xyz,
            "vel": self.drone.xyz_dot,
            "rpy": self.drone.rpy,
            "rates": self.drone.rpy_dot
        }

    # ===============================================================
    # REQUIRED ABSTRACT METHODS (now implemented)
    # ===============================================================

    def _setup_task_specifics(self):
        """No specialized task setup needed."""
        pass

    def compute_info(self, action=None):
        """Return optional info dict."""
        return {}

    def compute_potential(self):
        """Potential shaping (distance to target)."""
        return -np.linalg.norm(self.drone.xyz - self.target_pos)

    def get_reference_trajectory(self):
        """Return trajectory function used."""
        return self.trajectory_fn

    # ===============================================================
    # RESET BEHAVIOR
    # ===============================================================
    def task_specific_reset(self):
        """Reset drone on the active trajectory start state."""
        pos0, _ = self.trajectory_fn(0.0)
        pos0 = np.asarray(pos0, dtype=np.float32)
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos0,
            ornObj=self.init_quaternion
        )
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=np.zeros(3),
            angularVelocity=np.zeros(3),
        )


# ===============================================================
# ERROR PLOTTING
# ===============================================================
def plot_error(self):
    """Plot tracking error vs time."""
    arr = np.array(self.error_log)
    if arr.size == 0:
        print("No error data recorded.")
        return None

    t, ex, ey, ez, err_norm = arr.T

    plt.figure(figsize=(8, 5))
    plt.plot(t, ex, label='X error', color='tab:red')
    plt.plot(t, ey, label='Y error', color='tab:green')
    plt.plot(t, ez, label='Z error', color='tab:blue', linestyle='--', linewidth=1.5)
    plt.plot(t, err_norm, label='‖Error‖', color='k', linewidth=2)

    plt.title("Trajectory Tracking Error vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Error [m]")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    return arr
