# -*- coding: utf-8 -*-
#
#  ...........       ____  _ __
#  |  ,-^-,  |      / __ )(_) /_______________ _____  ___
#  | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  | / ,..´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#     +.......   /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  GNU general public license v3.0
#
#  Crazyflie hover + return-to-origin controller
#
#  Author: ChatGPT (Bitcraze-style port)
#

"""
file: return_to_home.py

A Bitcraze-style controller class that generates velocity commands
to make the Crazyflie hover at a fixed target point or return to
the origin from any position.

This is intentionally written in the same programming style and
structure as wall_following.py.

Author: ChatGPT (2025)
"""

import math
from enum import Enum


class ReturnToHome():
    """
    Hover + Return-to-Origin state machine controller.

    Produces body-frame velocity commands:
       vx (m/s), vy (m/s), yaw_rate (rad/s)

    The altitude is handled separately by the velocity PID controller
    (pid_velocity_fixed_height_controller), exactly like in the
    Bitcraze Webots demo.
    """

    class State(Enum):
        INIT = 1
        GO_TO_TARGET = 2
        HOVER = 3

    def __init__(self,
                 target_x=0.0,
                 target_y=0.0,
                 target_z=1.0,
                 max_speed=0.5,
                 max_yaw_rate=1.0,
                 position_tolerance=0.05,
                 heading_tolerance=0.15,
                 wait_for_sensor_seconds=1.0,
                 init_state=State.INIT):

        # --- desired hover/return coordinates ---
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z

        # --- control parameters ---
        self.max_speed = max_speed
        self.max_yaw_rate = max_yaw_rate
        self.position_tolerance = position_tolerance
        self.heading_tolerance = heading_tolerance
        self.wait_for_sensor_seconds = wait_for_sensor_seconds

        # --- internal state ---
        self.state = init_state
        self.first_run = True
        self.state_start_time = 0.0
        self.time_now = 0.0

    # Utility services
    def wrap_to_pi(self, x):
        if x > math.pi:
            return x - 2*math.pi
        elif x < -math.pi:
            return x + 2*math.pi
        else:
            return x

    def value_is_close(self, a, b, eps):
        return (a > b-eps) and (a < b+eps)

    # ================================
    #  Command helpers
    # ================================

    def command_hover(self):
        """Hold position: zero XY velocity, zero yaw rate."""
        return 0.0, 0.0, 0.0

    def command_goal_seek(self, dx, dy, desired_heading, current_heading):
        """
        Move toward the target (in world frame)
        but translate to body-frame velocities.

        Inputs:
          dx = x_error
          dy = y_error
          desired_heading = angle towards target
          current_heading = IMU yaw
        """

        # yaw controller
        heading_err = self.wrap_to_pi(desired_heading - current_heading)
        yaw_rate_cmd = max(-self.max_yaw_rate,
                           min(self.max_yaw_rate, 1.5 * heading_err))

        # Forward speed toward goal (when facing target)
        distance_error = math.sqrt(dx*dx + dy*dy)
        forward_speed = min(self.max_speed, distance_error * 1.0)

        # Convert world-frame forward speed to body-frame (X forward)
        # If yaw error is large, we reduce forward speed to prevent side-slip.
        if abs(heading_err) > 0.8:
            forward_speed *= 0.3

        # Body-frame velocities:
        vx_body = forward_speed
        vy_body = 0.0

        return vx_body, vy_body, yaw_rate_cmd

    # ================================
    #  Main state machine
    # ================================
    def update(self, x, y, z, yaw, time_outer_loop):
        """
        Update the state machine and compute new velocity commands.

        Returns:
            vx (m/s), vy (m/s), yaw_rate (rad/s), state
        """

        self.time_now = time_outer_loop

        # --- First run delay (sensor warmup) ---
        if self.first_run:
            if self.time_now < self.wait_for_sensor_seconds:
                return self.command_hover() + (self.state,)
            self.first_run = False
            self.state = self.State.GO_TO_TARGET
            self.state_start_time = self.time_now

        # --- Position errors ---
        dx = self.target_x - x
        dy = self.target_y - y
        dz = self.target_z - z  # handled upstream by height controller

        distance_xy = math.sqrt(dx*dx + dy*dy)
        desired_heading = math.atan2(dy, dx)

        # ============================
        #   STATE LOGIC
        # ============================
        if self.state == self.State.INIT:
            # Just hover until sensors ready
            vx, vy, yaw_rate = self.command_hover()

        elif self.state == self.State.GO_TO_TARGET:
            # Check if close enough
            if distance_xy < self.position_tolerance:
                self.state = self.State.HOVER
                self.state_start_time = self.time_now
                vx, vy, yaw_rate = self.command_hover()
            else:
                vx, vy, yaw_rate = self.command_goal_seek(
                    dx, dy, desired_heading, yaw)

        elif self.state == self.State.HOVER:
            vx, vy, yaw_rate = self.command_hover()

        else:
            # Safety fallback
            vx, vy, yaw_rate = self.command_hover()

        return vx, vy, yaw_rate, self.state
