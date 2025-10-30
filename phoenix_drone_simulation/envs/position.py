import numpy as np
from phoenix_drone_simulation.envs.control import Attitude

class PositionController:
   """This is for Controlling the Position"""

   def __init__(self, drone, bc, time_step):
      self.drone = drone
      self.bc = bc
      self.time_step = time_step
      self.attitude = Attitude(drone=drone, bc=bc, time_step=time_step)    #the attitude handles the motor controls

      # PID gains for positions(x, y, z)  
      # by Lawrence; these values need to be tuned
      self.kp = np.array([1.0, 1.0, 2.0])
      self.ki = np.array([0.0, 0.0, 0.1])
      self.kd = np.array([0.4, 0.4, 0.6])

      self.integral = np.zeros(3)
      self.prev_error = np.zeros(3)
      self.gravity = 9.81
   
   def reset(self):
        self.integral[:] = 0
        self.prev_error[:] = 0
        self.attitude.reset()

   def act(self, desired_position):
      """Computes thrust and attitude from desired 3D position."""
      # Get current position and velocity
      pos = np.array(self.drone.position())
      vel = np.array(self.drone.linear_velocity())

      # --- Compute errors ---
      e_p = desired_position - pos
      e_v = -vel
      self.integral += e_p * self.time_step

      # --- PID control for position ---
      acc_cmd = (self.kp * e_p) + (self.ki * self.integral) + (self.kd * e_v)  #computes the acceleration command

      # Add gravity compensation
      acc_cmd[2] += self.gravity

      # --- Convert to desired attitude ---
      roll_des  = np.clip(acc_cmd[1] / self.gravity, -0.4, 0.4)
      pitch_des = np.clip(-acc_cmd[0] / self.gravity, -0.4, 0.4)
      thrust = np.clip(acc_cmd[2], 0, 20)

      # --- Send to inner controller ---
      action = np.array([thrust, roll_des, pitch_des, 0.0])
      pwm = self.attitude.act(action)

      return pwm