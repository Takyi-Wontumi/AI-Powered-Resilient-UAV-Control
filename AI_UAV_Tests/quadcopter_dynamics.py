"""
Full PID-Based Crazyflie-Style Quadcopter Simulator
Author: Lawrence Wontumi (2025) + ChatGPT ruthless cleanup

Architecture:
    Trajectory → PID Position → PID Attitude → PID Rate → Mixer → Motors → Dynamics
With:
    - Full PID for altitude (z)
    - PD for xy (good enough)
    - PID angle control
    - PID rate control
    - Anti-windup everywhere
    - Hard safety limits
"""

import numpy as np
import matplotlib.pyplot as plt

KILL_HEIGHT = 100

# =========================================================
# 1. Full CF-Style Quadcopter With PID Control
# =========================================================
class QuadcopterPID:

    def __init__(self, dt=0.002, g=9.81):

        # Physical constants
        self.dt = dt
        self.m = 0.028
        self.l = 0.046
        self.b = 1.4e-6
        self.d = 1.1e-7
        self.g = g
        self.Ix, self.Iy, self.Iz = 16.6e-6, 16.6e-6, 29.3e-6
        self.J = np.diag([self.Ix, self.Iy, self.Iz])

        # Motor mixer matrix
        self.M = np.array([
            [ self.b,            self.b,            self.b,            self.b ],
            [ 0,                -self.l*self.b,     0,                 self.l*self.b ],
            [-self.l*self.b,     0,                 self.l*self.b,      0 ],
            [ self.d,           -self.d,            self.d,            -self.d ]
        ])
        self.M_inv = np.linalg.inv(self.M)

        # =========================
        # Position PID (XY)
        # =========================
        self.Kp_xy = np.array([.8, .8])
        self.Kd_xy = np.array([0.6, 0.6])
        # XY uses PD only (integral in xy is dumb unless you like oscillations)

        # =========================
        # Altitude PID (full PID)
        # =========================
        self.Kp_z = 4.0
        self.Ki_z = 1.5
        self.Kd_z = 2.0
        self.z_int = 0.0
        self.z_int_limit = 1.0

        # =========================
        # Attitude PID
        # =========================
        self.Kp_att = np.array([8.0, 8.0, 4.0])
        self.Ki_att = np.array([1.0, 1.0, 0.5])
        self.att_int = np.zeros(3)
        self.att_int_limit = np.array([0.3, 0.3, 0.3])

        # =========================
        # Rate PID
        # =========================
        self.Kp_rate = np.array([2.5e-4, 2.5e-4, 1.8e-4])
        self.Ki_rate = np.array([1.5e-5, 1.5e-5, 1.5e-5])
        self.Kd_rate = np.array([1.0e-5, 1.0e-5, 1.0e-5])
        self.rate_int = np.zeros(3)
        self.rate_int_limit = np.array([5e-4, 5e-4, 5e-4])

        # Limits
        self.max_angle = np.deg2rad(20)
        self.max_rate  = np.deg2rad(500)
        self.max_omega = 2500
        self.tau_max   = 3e-4

        # State
        self.reset()

    # ------------------------
    # Rotation utilities
    # ------------------------
    @staticmethod
    def Rx(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])

    @staticmethod
    def Ry(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])

    @staticmethod
    def Rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    def R(self):
        phi, th, psi = self.ang
        return self.Rz(psi) @ self.Ry(th) @ self.Rx(phi)

    # ------------------------
    # Reset state
    # ------------------------
    def reset(self):
        self.x    = np.zeros(3)
        self.v    = np.zeros(3)
        self.ang  = np.zeros(3)
        self.rate = np.zeros(3)
        self.z_int = 0.0
        self.att_int[:] = 0
        self.rate_int[:] = 0

    # =========================================================
    #  PID LAYER 1: Position PID
    # =========================================================
    def position_pid(self, pos_ref, vel_ref, z_ref=1.0):

        # XY PD
        exy  = pos_ref[:2] - self.x[:2]
        evxy = vel_ref[:2] - self.v[:2]
        a_xy = self.Kp_xy*exy + self.Kd_xy*evxy
        a_xy = np.clip(a_xy, -3.0, 3.0)

        # Z PID
        ez  = z_ref - self.x[2]
        evz = -self.v[2]

        # Anti-windup for altitude
        self.z_int += ez * self.dt
        self.z_int = np.clip(self.z_int, -self.z_int_limit, self.z_int_limit)

        a_z = (self.Kp_z * ez +
               self.Kd_z * evz +
               self.Ki_z * self.z_int)

        a_z = np.clip(a_z, -4.0, 4.0)

        # Convert to thrust
        U1 = self.m * (self.g + a_z)
        U1 = np.clip(U1, 0.5*self.m*self.g, 1.3*self.m*self.g)

        # Desired angles from XY acceleration
        psi = self.ang[2]
        ax, ay = a_xy

        phi_des   = (ax*np.sin(psi) - ay*np.cos(psi)) / self.g
        theta_des = (ax*np.cos(psi) + ay*np.sin(psi)) / self.g

        phi_des   = np.clip(phi_des,  -self.max_angle, self.max_angle)
        theta_des = np.clip(theta_des,-self.max_angle, self.max_angle)

        psi_des = 0.0

        return U1, phi_des, theta_des, psi_des

    # =========================================================
    #  PID LAYER 2: Attitude PID
    # =========================================================
    def attitude_pid(self, phi_des, theta_des, psi_des):

        phi, th, psi = self.ang

        e_att = np.array([
            phi_des - phi,
            theta_des - th,
            psi_des - psi
        ])

        self.att_int += e_att * self.dt
        self.att_int = np.clip(self.att_int, -self.att_int_limit, self.att_int_limit)

        rates_des = (
            self.Kp_att * e_att +
            self.Ki_att * self.att_int
        )

        rates_des = np.clip(rates_des, -self.max_rate, self.max_rate)
        return rates_des

    # =========================================================
    #  PID LAYER 3: Rate PID
    # =========================================================
    def rate_pid(self, rates_des):

        er = rates_des - self.rate

        self.rate_int += er * self.dt
        self.rate_int = np.clip(self.rate_int, -self.rate_int_limit, self.rate_int_limit)

        tau = (
            self.Kp_rate * er +
            self.Ki_rate * self.rate_int -
            self.Kd_rate * self.rate
        )

        tau = np.clip(tau, -self.tau_max, self.tau_max)
        return tau

    # =========================================================
    #  Mixer
    # =========================================================
    def mixer(self, U1, tau):

        tau_phi, tau_theta, tau_psi = tau

        u = np.array([U1, tau_phi, tau_theta, tau_psi])
        thrusts = self.M_inv @ u

        thrusts = np.clip(thrusts, 0, np.inf)

        omega_sq = thrusts / self.b
        omega_sq = np.clip(omega_sq, 0, self.max_omega**2)
        return np.sqrt(omega_sq)

    # =========================================================
    #  Dynamics
    # =========================================================
    def f(self, state, omega):
        x,y,z, vx,vy,vz, phi,th,psi, p,q,r = state

        R = self.R()
        thrust = self.b * np.sum(omega**2)
        acc = (thrust * R @ np.array([0,0,1]))/self.m - np.array([0,0,self.g])

        b,d,l = self.b, self.d, self.l
        w1,w2,w3,w4 = omega

        tau_phi   = l*b*(w4**2 - w2**2)
        tau_theta = l*b*(w3**2 - w1**2)
        tau_psi   = d*(w1**2 - w2**2 + w3**2 - w4**2)

        p_dot = ((self.Iy - self.Iz)/self.Ix)*q*r + tau_phi/self.Ix
        q_dot = ((self.Iz - self.Ix)/self.Iy)*p*r + tau_theta/self.Iy
        r_dot = ((self.Ix - self.Iy)/self.Iz)*p*q + tau_psi/self.Iz

        cth = max(np.cos(th), 1e-3)
        sth = np.sin(th)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        T = np.array([
            [1, sphi*sth/cth, cphi*sth/cth],
            [0, cphi, -sphi],
            [0, sphi/cth, cphi/cth]
        ])
        ang_dot = T @ np.array([p,q,r])

        return np.array([
            vx,vy,vz,
            acc[0], acc[1], acc[2],
            ang_dot[0], ang_dot[1], ang_dot[2],
            p_dot, q_dot, r_dot
        ])

    # =========================================================
    #  One full PID step
    # =========================================================
    def step(self, pos_ref, vel_ref, z_ref=1.0):

        # Layer 1
        U1, phi_des, theta_des, psi_des = self.position_pid(pos_ref, vel_ref, z_ref)

        # Layer 2
        rates_des = self.attitude_pid(phi_des, theta_des, psi_des)

        # Layer 3
        tau = self.rate_pid(rates_des)

        # Mix
        omega = self.mixer(U1, tau)

        # Dynamics integration RK4
        state = np.hstack([self.x, self.v, self.ang, self.rate])
        f = lambda s: self.f(s, omega)
        dt = self.dt

        k1 = f(state)
        k2 = f(state + 0.5*dt*k1)
        k3 = f(state + 0.5*dt*k2)
        k4 = f(state + dt*k3)

        state = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

        self.x    = state[0:3]
        self.v    = state[3:6]
        self.ang  = state[6:9]
        self.rate = state[9:12]

        # Safety kill: altitude runaway
        if self.x[2] > KILL_HEIGHT:
            print("[SAFETY] Altitude > 3m, cutting motors.")
            self.reset()

        return {
            "ex": pos_ref[0] - self.x[0],
            "ey": pos_ref[1] - self.x[1],
            "ez": z_ref - self.x[2]
        }


# =========================================================
# 2. SIMPLE TRAJECTORY FOR TESTING
# =========================================================
def hover_traj(t):
    return np.array([0.0,0.0,1.0]), np.array([0,0,0])


# =========================================================
# 3. DEMO RUN
# =========================================================
if __name__ == "__main__":
    quad = QuadcopterPID(dt=0.002)

    T = 10
    steps = int(T/quad.dt)
    z_log = []

    for k in range(steps):
        pos_ref, vel_ref = hover_traj(k*quad.dt)
        diag = quad.step(pos_ref, vel_ref, z_ref=1.0)
        z_log.append(quad.x[2])

    plt.plot(z_log)
    plt.title("Altitude Stabilization (PID)")
    plt.xlabel("Step")
    plt.ylabel("z [m]")
    plt.grid(True)
    plt.show()
