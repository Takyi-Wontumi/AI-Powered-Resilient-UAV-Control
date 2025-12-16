"""
Offboard Crazyflie rate+thrust control using your QuadcopterPID.

- Logs: stateEstimate.{x,y,z,vx,vy,vz,roll,pitch,yaw} and stateEstimateZ.rate{Roll,Pitch,Yaw}
- Sends: roll_rate, pitch_rate, yaw_rate (deg/s) + thrust (uint16-ish int) via commander.send_setpoint()
- Forces rate mode via flightmode.stabMode{Roll,Pitch,Yaw} = 0  (rate) :contentReference[oaicite:2]{index=2}

REQUIRES:
pip install cflib

SAFETY:
- Test with props OFF first (or tethered) and verify signs.
"""

import os
import sys

THIS_FILE = os.path.abspath(__file__)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(THIS_FILE), ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


import time
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger


from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.trajectories_library import Trajectories as path

# -------------------------
# User config
# -------------------------
URI = "radio://0/80/2M/E7E7E7E7E7"   # change to your Crazyflie URI
DT_CTRL = 0.02                      # 50 Hz control loop (don’t spam the radio)

KILL_Z = 1.8                        # meters (pick sane indoor value)
KILL_XY = 2.0                       # meters radius

# Thrust mapping calibration (YOU MUST TUNE THIS)
# Typical hover thrust setpoint is often ~38000–42000, but varies.
HOVER_THRUST = 40000
MIN_THRUST = 10001
MAX_THRUST = 60000

def u1_newtons_to_thrust_int(U1_N: float, m: float, g: float = 9.81) -> int:
    """
    Convert your U1 in Newtons to Crazyflie thrust int by scaling around hover.
    This is a crude mapping; you should re-identify HOVER_THRUST for your exact drone.
    """
    scale = float(U1_N / (m * g))
    thrust = int(np.clip(HOVER_THRUST * scale, MIN_THRUST, MAX_THRUST))
    return thrust

def set_flightmode_rate(cf: Crazyflie):
    # 0 = rate, 1 = angle :contentReference[oaicite:3]{index=3}
    cf.param.set_value("flightmode.stabModeRoll", "0")
    cf.param.set_value("flightmode.stabModePitch", "0")
    cf.param.set_value("flightmode.stabModeYaw", "0")

    # Make sure we are NOT in altitude hold / position hold (those remap inputs)
    cf.param.set_value("flightmode.althold", "0")
    cf.param.set_value("flightmode.poshold", "0")

def make_logconfs():
    """
    You can’t cram everything into one log block due to packet limits,
    so we split into two. :contentReference[oaicite:4]{index=4}
    """
    lg_state = LogConfig(name="state", period_in_ms=int(DT_CTRL * 1000))
    lg_state.add_variable("stateEstimate.x", "float")
    lg_state.add_variable("stateEstimate.y", "float")
    lg_state.add_variable("stateEstimate.z", "float")
    lg_state.add_variable("stateEstimate.vx", "float")
    lg_state.add_variable("stateEstimate.vy", "float")
    lg_state.add_variable("stateEstimate.vz", "float")

    lg_att = LogConfig(name="att", period_in_ms=int(DT_CTRL * 1000))
    lg_att.add_variable("stateEstimate.roll", "float")   # deg :contentReference[oaicite:5]{index=5}
    lg_att.add_variable("stateEstimate.pitch", "float")  # deg (inverted!) :contentReference[oaicite:6]{index=6}
    lg_att.add_variable("stateEstimate.yaw", "float")    # deg
    lg_att.add_variable("stateEstimateZ.rateRoll", "int16")   # mrad/s :contentReference[oaicite:7]{index=7}
    lg_att.add_variable("stateEstimateZ.ratePitch", "int16")  # mrad/s :contentReference[oaicite:8]{index=8}
    lg_att.add_variable("stateEstimateZ.rateYaw", "int16")    # mrad/s :contentReference[oaicite:9]{index=9}

    return lg_state, lg_att

def ramp_thrust(cf: Crazyflie, t_s: float = 1.0):
    # gentle spin-up to hover to avoid instant flip
    steps = max(1, int(t_s / DT_CTRL))
    for i in range(steps):
        u = (i + 1) / steps
        thrust = int(MIN_THRUST + u * (HOVER_THRUST - MIN_THRUST))
        cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)  # roll, pitch, yawrate, thrust :contentReference[oaicite:10]{index=10}
        time.sleep(DT_CTRL)

def land_and_stop(cf: Crazyflie, t_s: float = 1.0):
    steps = max(1, int(t_s / DT_CTRL))
    for i in range(steps):
        u = 1.0 - (i + 1) / steps
        thrust = int(MIN_THRUST + u * (HOVER_THRUST - MIN_THRUST))
        cf.commander.send_setpoint(0.0, 0.0, 0.0, thrust)
        time.sleep(DT_CTRL)
    cf.commander.send_stop_setpoint()

def main():
    cflib.crtp.init_drivers(enable_debug_driver=False)

    # ---- instantiate your controller ----
    quad = QuadcopterPID(dt=DT_CTRL)   # your class
    # Choose trajectory:
    # traj_fn = path.square_traj
    traj_fn = None  # replace with your trajectory fn(t)->(pos_ref, vel_ref)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache="./cache")) as scf:
        cf = scf.cf

        # Commander / setpoint framework is the right place for offboard setpoints :contentReference[oaicite:11]{index=11}
        set_flightmode_rate(cf)

        lg_state, lg_att = make_logconfs()
        cf.log.add_config(lg_state)
        cf.log.add_config(lg_att)
        lg_state.start()
        lg_att.start()

        # Prime controller integrators
        quad.reset()

        # Takeoff ramp
        ramp_thrust(cf, t_s=1.2)

        # We’ll pull latest values from both log streams
        latest = {}

        def update_latest(data):
            latest.update(data)

        # SyncLogger can take multiple log configs in one go :contentReference[oaicite:12]{index=12}
        with SyncLogger(scf, [lg_state, lg_att]) as logger:
            t0 = time.time()
            last_send = time.time()

            try:
                for log_entry in logger:
                    _, data, _ = log_entry
                    update_latest(data)

                    now = time.time()
                    if now - last_send < DT_CTRL:
                        continue
                    last_send = now

                    # ---- read state ----
                    x = np.array([
                        latest.get("stateEstimate.x", 0.0),
                        latest.get("stateEstimate.y", 0.0),
                        latest.get("stateEstimate.z", 0.0),
                    ], dtype=float)

                    v = np.array([
                        latest.get("stateEstimate.vx", 0.0),
                        latest.get("stateEstimate.vy", 0.0),
                        latest.get("stateEstimate.vz", 0.0),
                    ], dtype=float)

                    roll_deg  = float(latest.get("stateEstimate.roll", 0.0))
                    pitch_deg = float(latest.get("stateEstimate.pitch", 0.0))
                    yaw_deg   = float(latest.get("stateEstimate.yaw", 0.0))

                    # IMPORTANT: pitch is inverted in this log var :contentReference[oaicite:13]{index=13}
                    ang = np.array([
                        np.deg2rad(roll_deg),
                        -np.deg2rad(pitch_deg),
                        np.deg2rad(yaw_deg),
                    ], dtype=float)

                    # rates in mrad/s -> rad/s :contentReference[oaicite:14]{index=14}
                    rate = np.array([
                        1e-3 * float(latest.get("stateEstimateZ.rateRoll", 0.0)),
                        1e-3 * float(latest.get("stateEstimateZ.ratePitch", 0.0)),
                        1e-3 * float(latest.get("stateEstimateZ.rateYaw", 0.0)),
                    ], dtype=float)

                    # ---- hard safety ----
                    if x[2] > KILL_Z or np.linalg.norm(x[:2]) > KILL_XY or x[2] < 0.05:
                        print("[SAFETY] Kill condition triggered. Landing.")
                        break

                    # ---- reference ----
                    t = now - t0
                    if traj_fn is None:
                        # hover fallback at 1.0m
                        pos_ref = np.array([0.0, 0.0, 1.0])
                        vel_ref = np.zeros(3)
                    else:
                        pos_ref, vel_ref = traj_fn(t)

                    # ---- run your controller (feed state exactly like Phoenix injection) ----
                    quad.inject_external_state(x, v, ang, rate)
                    out = quad.step(pos_ref, vel_ref, z_ref=float(pos_ref[2]))

                    rates_des = out["rates_des"]      # rad/s (body rates)
                    U1 = float(out["thrust_cmd"])     # Newtons

                    # Clamp rates harder than your sim: don’t be stupid with 500 deg/s indoors
                    rates_des = np.clip(rates_des, np.deg2rad(-200), np.deg2rad(200))

                    roll_rate_dps  = float(np.rad2deg(rates_des[0]))
                    pitch_rate_dps = float(np.rad2deg(rates_des[1]))
                    yaw_rate_dps   = float(np.rad2deg(rates_des[2]))

                    thrust_int = u1_newtons_to_thrust_int(U1, m=quad.m, g=quad.g)

                    # Send setpoint: roll, pitch, yawrate, thrust.
                    # With stabModeRoll/Pitch/Yaw = 0 these are interpreted as rates. :contentReference[oaicite:15]{index=15}
                    cf.commander.send_setpoint(
                        roll_rate_dps,
                        pitch_rate_dps,
                        yaw_rate_dps,
                        thrust_int
                    )

            except KeyboardInterrupt:
                print("\n[CTRL-C] Landing.")
            finally:
                land_and_stop(cf, t_s=1.0)
                lg_state.stop()
                lg_att.stop()

if __name__ == "__main__":
    main()
