# -*- coding: utf-8 -*-

from controller import Robot
import math
import numpy as np

# Import your existing PID controller
from pid_controller import pid_velocity_fixed_height_controller


# ================================================================
# USER PARAMETERS
# ================================================================
TAKEOFF_ALT = 1.0
SIDE = 1.0              # 1 meter square side
WAYPOINT_TOL = 0.08     # waypoint switch tolerance
Kp_xy = 1.2             # position → velocity gain
YAW_GAIN = 2.0          # rate to align yaw
TAKEOFF_Z_ERR = 0.05    # altitude tolerance for switching to trajectory


# ================================================================
# SQUARE TRAJECTORY WAYPOINTS (1 m², starting at origin)
# ================================================================
SQUARE_POINTS = [
    (0.0, 0.0),
    (SIDE, 0.0),
    (SIDE, SIDE),
    (0.0, SIDE)
]


# ================================================================
# INITIALIZE WEBOTS
# ================================================================
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

# PID Controller
PID = pid_velocity_fixed_height_controller()


# ================================================================
# STATE VARIABLES
# ================================================================
state = "TAKEOFF"
current_wp_index = 0

past_time = robot.getTime()
past_x = gps.getValues()[0]
past_y = gps.getValues()[1]

print("\n=== Crazyflie Square Trajectory Controller ===")
print("State: TAKEOFF\n")


# ================================================================
# MAIN LOOP
# ================================================================
while robot.step(timestep) != -1:

    # Time
    now = robot.getTime()
    dt = now - past_time
    past_time = now

    # Sensor Readings
    roll, pitch, yaw = imu.getRollPitchYaw()
    yaw_rate = gyro.getValues()[2]

    x = gps.getValues()[0]
    y = gps.getValues()[1]
    z = gps.getValues()[2]

    # Compute global velocity
    vx_global = (x - past_x) / dt
    vy_global = (y - past_y) / dt
    past_x, past_y = x, y

    # Convert to body frame
    cy, sy = math.cos(yaw), math.sin(yaw)
    vx_body = vx_global * cy + vy_global * sy
    vy_body = -vx_global * sy + vy_global * cy


    # ============================================================
    # TAKEOFF LOGIC
    # ============================================================
    if state == "TAKEOFF":

        desired_vx = 0.0
        desired_vy = 0.0
        desired_yaw_rate = 0.0
        desired_altitude = TAKEOFF_ALT

        # Switch to square trajectory
        if abs(z - TAKEOFF_ALT) < TAKEOFF_Z_ERR:
            print("=== Takeoff Complete → Starting Square Trajectory ===")
            state = "SQUARE"
            continue

        # PID for takeoff
        motor_powers = PID.pid(
            dt, desired_vx, desired_vy,
            desired_yaw_rate, desired_altitude,
            roll, pitch, yaw_rate,
            z, vx_body, vy_body
        )

        m1.setVelocity(-motor_powers[0])
        m2.setVelocity(motor_powers[1])
        m3.setVelocity(-motor_powers[2])
        m4.setVelocity(motor_powers[3])

        continue


    # ============================================================
    # SQUARE TRAJECTORY TRACKING
    # ============================================================
    if state == "SQUARE":

        target_x, target_y = SQUARE_POINTS[current_wp_index]

        # Position errors in global frame
        ex = target_x - x
        ey = target_y - y
        dist = math.sqrt(ex*ex + ey*ey)

        # Waypoint reached
        if dist < WAYPOINT_TOL:
            current_wp_index = (current_wp_index + 1) % len(SQUARE_POINTS)
            print(f"Reached waypoint → Now going to {current_wp_index}")
            continue

        # Convert position → desired global velocity
        desired_vx_global = Kp_xy * ex
        desired_vy_global = Kp_xy * ey

        # Convert to body-frame desired velocity
        desired_vx = desired_vx_global * cy + desired_vy_global * sy
        desired_vy = -desired_vx_global * sy + desired_vy_global * cy

        # Align yaw with direction of motion
        desired_heading = math.atan2(ey, ex)
        heading_error = (desired_heading - yaw + math.pi) % (2*math.pi) - math.pi
        desired_yaw_rate = YAW_GAIN * heading_error

        # Keep constant altitude during square flight
        desired_altitude = TAKEOFF_ALT


    # ============================================================
    # PID CONTROL + MOTOR OUTPUT
    # ============================================================
    motor_powers = PID.pid(
        dt,
        desired_vx,
        desired_vy,
        desired_yaw_rate,
        desired_altitude,
        roll, pitch, yaw_rate,
        z,
        vx_body, vy_body
    )

    m1.setVelocity(-motor_powers[0])
    m2.setVelocity(motor_powers[1])
    m3.setVelocity(-motor_powers[2])
    m4.setVelocity(motor_powers[3])
