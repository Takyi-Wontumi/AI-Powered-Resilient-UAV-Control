import os
import sys

# Get absolute path to project root
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

# Add to sys.path if missing
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

print("Added to PYTHONPATH:", ROOT_DIR)

# =========================================================
# Multi-Trajectory Follower for Webots Crazyflie
# Author: Lawrence Wontumi (2025)
# =========================================================

import math
import numpy as np
from controller import Robot
from AI_UAV_Tests.trajectory_manager import TrajectoryManager
from pid_controller import pid_velocity_fixed_height_controller
from AI_UAV_Tests.test_class import TestClass

YAW_GAIN = 1.2

# =========================================================
# Init Webots Robot
# =========================================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
m1 = robot.getDevice("m1_motor"); m1.setPosition(float('inf'))
m2 = robot.getDevice("m2_motor"); m2.setPosition(float('inf'))
m3 = robot.getDevice("m3_motor"); m3.setPosition(float('inf'))
m4 = robot.getDevice("m4_motor"); m4.setPosition(float('inf'))

# Sensors
imu = robot.getDevice("inertial_unit"); imu.enable(timestep)
gps = robot.getDevice("gps"); gps.enable(timestep)
gyro = robot.getDevice("gyro"); gyro.enable(timestep)

# PID controller instance (Crazyflie-style firmware controller)
PID = pid_velocity_fixed_height_controller()

# === Trajectory Manager ===
traj = TrajectoryManager(default="circle")   # start with circle

# Initial state tracking
past_x = gps.getValues()[0]
past_y = gps.getValues()[1]
past_time = robot.getTime()

print("\n=== Multi-Trajectory Follower Started ===\n")


# =========================================================
# MAIN LOOP
# =========================================================
while robot.step(timestep) != -1:

    msg = TestClass("This is working in Webot")
    msg.sendMsg()

    # ----- Time update ----- #
    now = robot.getTime()
    dt = now - past_time
    past_time = now

    # ----- Switch Trajectories (Example) ----- #
    # You can change these or trigger via keyboard
    if now > 10 and traj.active != "square":
        traj.switch("square", now)
    if now > 20 and traj.active != "helix":
        traj.switch("helix", now)

    # ----- Sensor readings ----- #
    roll, pitch, yaw = imu.getRollPitchYaw()
    yaw_rate = gyro.getValues()[2]

    x, y, z = gps.getValues()

    vx_global = (x - past_x) / dt
    vy_global = (y - past_y) / dt
    past_x, past_y = x, y

    cy, sy = math.cos(yaw), math.sin(yaw)

    # Convert to body frame velocities
    vx_body = vx_global * cy + vy_global * sy
    vy_body = -vx_global * sy + vy_global * cy

    # ============================================================
    #                    Trajectory Reference
    # ============================================================
    pos_ref, vel_ref = traj.get_reference(now)

    target_x, target_y, target_z = pos_ref
    desired_vx_global, desired_vy_global, desired_vz_global = vel_ref

    # Convert desired velocity to body frame
    desired_vx = desired_vx_global * cy + desired_vy_global * sy
    desired_vy = -desired_vx_global * sy + desired_vy_global * cy

    # ============================================================
    #          Yaw heading = direction of global velocity
    # ============================================================
    desired_heading = math.atan2(desired_vy_global, desired_vx_global)
    heading_error = (desired_heading - yaw + math.pi) % (2 * math.pi) - math.pi
    desired_yaw_rate = YAW_GAIN * heading_error

    # ============================================================
    #               Compute PID motor commands
    # ============================================================
    motor_powers = PID.pid(
        dt,
        desired_vx,
        desired_vy,
        desired_yaw_rate,
        target_z,
        roll, pitch, yaw_rate,
        z,
        vx_body, vy_body
    )

    # ============================================================
    #               Send motor commands to Webots motors
    # ============================================================
    m1.setVelocity(-motor_powers[0])
    m2.setVelocity( motor_powers[1])
    m3.setVelocity(-motor_powers[2])
    m4.setVelocity( motor_powers[3])

# import os
# import sys

# # Get absolute path to project root
# ROOT_DIR = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../../../")
# )

# # Add to sys.path if missing
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

# print("Added to PYTHONPATH:", ROOT_DIR)

# # =========================================================
# # Circular Trajectory Follower for Webots Crazyflie
# # Author: Lawrence Wontumi (2025)
# # =========================================================

# import math
# import numpy as np
# from controller import Robot
# from trajectory_follower import TrajectoryManager
# from pid_controller import pid_velocity_fixed_height_controller
# from AI_UAV_Tests.test_class import TestClass

# # =========================================================
# # Circular Trajectory Parameters
# # =========================================================
# RADIUS = 1.0             # meters
# ANGULAR_SPEED = 0.25     # rad/s  (~25 seconds per circle)
# CENTER_X = 0.0
# CENTER_Y = 0.0

# TARGET_ALTITUDE = 2.0
# YAW_GAIN = 1.2
# VEL_GAIN = 0.8

# # =========================================================
# # Init Webots Robot
# # =========================================================
# robot = Robot()
# timestep = int(robot.getBasicTimeStep())

# # Motors
# m1 = robot.getDevice("m1_motor"); m1.setPosition(float('inf'))
# m2 = robot.getDevice("m2_motor"); m2.setPosition(float('inf'))
# m3 = robot.getDevice("m3_motor"); m3.setPosition(float('inf'))
# m4 = robot.getDevice("m4_motor"); m4.setPosition(float('inf'))

# # Sensors
# imu = robot.getDevice("inertial_unit"); imu.enable(timestep)
# gps = robot.getDevice("gps"); gps.enable(timestep)
# gyro = robot.getDevice("gyro"); gyro.enable(timestep)

# # PID controller instance (your firmware-based controller)
# PID = pid_velocity_fixed_height_controller()

# # Initial state tracking
# past_x = gps.getValues()[0]
# past_y = gps.getValues()[1]
# past_time = robot.getTime()

# print("\n=== Circular Trajectory Follower Started ===\n")


# # =========================================================
# # MAIN LOOP
# # =========================================================
# while robot.step(timestep) != -1:

#     msg = TestClass("This is working in Webot")
#     msg.sendMsg()

#     # ----- Time update ----- #
#     now = robot.getTime()
#     dt = now - past_time
#     past_time = now

#     # ----- Sensor readings ----- #
#     roll, pitch, yaw = imu.getRollPitchYaw()
#     yaw_rate = gyro.getValues()[2]

#     x, y, z = gps.getValues()

#     vx_global = (x - past_x) / dt
#     vy_global = (y - past_y) / dt
#     past_x, past_y = x, y

#     cy, sy = math.cos(yaw), math.sin(yaw)

#     # Convert to body frame
#     vx_body = vx_global * cy + vy_global * sy
#     vy_body = -vx_global * sy + vy_global * cy


#     # ============================================================
#     #                Circular Trajectory Reference
#     # ============================================================
#     theta = ANGULAR_SPEED * now

#     # Position reference
#     target_x = CENTER_X + RADIUS * math.cos(theta)
#     target_y = CENTER_Y + RADIUS * math.sin(theta)

#     # Velocity reference (global)
#     desired_vx_global = -RADIUS * ANGULAR_SPEED * math.sin(theta)
#     desired_vy_global =  RADIUS * ANGULAR_SPEED * math.cos(theta)

#     # Convert to body frame
#     desired_vx = desired_vx_global * cy + desired_vy_global * sy
#     desired_vy = -desired_vx_global * sy + desired_vy_global * cy


#     # ============================================================
#     #               Yaw → tangent direction of circle
#     # ============================================================
#     desired_heading = math.atan2(desired_vy_global, desired_vx_global)

#     heading_error = (desired_heading - yaw + math.pi) % (2*math.pi) - math.pi
#     desired_yaw_rate = YAW_GAIN * heading_error


#     # ============================================================
#     #               Compute PID motor commands
#     # ============================================================
#     motor_powers = PID.pid(
#         dt,
#         desired_vx,
#         desired_vy,
#         desired_yaw_rate,
#         TARGET_ALTITUDE,
#         roll, pitch, yaw_rate,
#         z,
#         vx_body, vy_body
#     )


#     # ============================================================
#     #               Send motor commands to Webots motors
#     # ============================================================
#     m1.setVelocity(-motor_powers[0])
#     m2.setVelocity( motor_powers[1])
#     m3.setVelocity(-motor_powers[2])
#     m4.setVelocity( motor_powers[3])
