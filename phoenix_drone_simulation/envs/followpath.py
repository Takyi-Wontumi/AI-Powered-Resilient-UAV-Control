"""
Author: Lawrence Wontumi (2025)
"""
import logging
import numpy as np
from phoenix_drone_simulation.envs.base import DroneBaseEnv
from AI_UAV_Tests.trajectories_library import Trajectories as path
import matplotlib.pyplot as plt

class DroneFollowPathEnv(DroneBaseEnv):
   """This is meant to train the drone to follow an arbitrary trajectory ( square, helx, waypoints)"""

   #setting up logger
   logging.config.fileConfig('temp.conf')
   # create logger
   logger = logging.getLogger('simpleExample')

   def __init__(self, trajectory_fn=None, control_mode='PWM', log_errors=True,  **kwargs):
      self.trajectory_fn =  trajectory_fn
      self.done_dist_threshold = 0.3
      self.penalty_action = 1e-4
      self.penalty_velocity = 1e-4
      self.penalty_spin = 1e-4
      self.penalty_terminal = 100
      self.ARP = 1e-3

      #new functionality added, logging errors
      self.log_errors = log_errors
      self.error_log = []     #storing as (time, ex, ey, ez, norm)

      #initializing the base environment
      super().__init__(physics='PyBulletPhysics', control_mode=control_mode, drone_model='cf21x_bullet', observation_frequency=100, sim_freq=200, **kwargs)

   def compute_observation(self):
      """"""
      
      # the observation needs 
      t =  self.iteration / self.SIM_FREQ
      pos_ref, vel_ref = self.trajectory_fn(t)
      self.target_pos = pos_ref
      error_to_ref = pos_ref - self.drone.xyz

      # Log tracking error
      if self.log_errors:
         self.error_log.append([t, *error_to_ref, np.linalg.norm(error_to_ref)])

      obs = np.contatenate([self.drone.xyz, self.drone.quaternion, self.drone.xyz_dot, self.drone.rpy_dot, error_to_ref])

      return obs
   
   # Reward section
   """
      self.penalty_action x ||action|| --> punishes the magnitude of the control signal (action)
      self.penalty_velocity x ||self.drone.xyz_dot|| --> penalizes the magnitude of the drone's linear velocity . This is often used to encourage the drone to reach a target and then hover or to conserve kinetic energy
      self.penalty_spin x ||self.drone.rpy_dot|| --> discourages rapid rotations or unnecessary spinning
      """

   def compute_reward(self, action):
      dist = np.linalg.norm(self.drone.xyz -  self.target_pos)    #consider plotting to see the normalized plot
      penalties = self.penalty_action * np.linalg.norm(action) + self.penalty_velocity * np.linalg.norm(self.drone.xyz_dot) + self.penalty_spin * np.linalg.norm(self.drone.rpy_dot)

      reward = -dist - penalties
      if self.compute_done():
         reward -= self.penalty_terminal      
      return reward
   
   def compute_done(self):
      return np.linalg.norm(self.drone.xyz - self.target_pos) > self.done_dist_threshold
   
   #other functionality

   def task_specific_reset(self):
      self.bc.resetBasePositionAndOrientation(self.drone.body_unique_id, posObj=np.array([0, 0, 1]), ornObj=self.init_quaternion)

   # -------------------------------
   

def plot_error(self):
    """Plot tracking error over time instead of saving to CSV."""
    arr = np.array(self.error_log)
    if arr.size == 0:
        print("No error data recorded.")
        return None

    # Extract logged data
    t, ex, ey, ez, err_norm = arr.T

    # Plot setup
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

