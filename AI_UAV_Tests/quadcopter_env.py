import numpy as np

KILL_HEIGHT = 5

# =========================================================
#  Full PID-Based Crazyflie-Style Quadcopter Simulator
#  + External-State Injection for Phoenix
# =========================================================
class QuadcopterPID:

    def __init__(self, dt=0.002, g=9.81):

        # -----------------------------
        # Physical constants
        # -----------------------------
        self.dt = dt
        self.m = 0.028
        self.l = 0.046
        self.b = 1.4e-6
        self.d = 1.1e-7
        self.g = g
        self.Ix, self.Iy, self.Iz = 16.6e-6, 16.6e-6, 29.3e-6
        self.J = np.diag([self.Ix, self.Iy, self.Iz])

        # -----------------------------
        # Motor mixer matrix
        # -----------------------------
        self.M = np.array([
            [ self.b,            self.b,            self.b,            self.b ],
            [ 0,                -self.l*self.b,     0,                 self.l*self.b ],
            [-self.l*self.b,     0,                 self.l*self.b,      0 ],
            [ self.d,           -self.d,            self.d,            -self.d ]
        ])
        self.M_inv = np.linalg.inv(self.M)

        # -----------------------------
        # XY Position PD
        # -----------------------------
        self.Kp_xy = np.array([8, 8])
        self.Kd_xy = np.array([15, 15])

        # -----------------------------
        # Altitude PID
        # -----------------------------
        self.Kp_z = 16.2
        self.Ki_z = 7.0175
        self.Kd_z = 9.3494
        self.z_int = 0.0
        self.z_int_limit = 1.0

        # -----------------------------
        # Attitude PID
        # -----------------------------
        self.Kp_att = np.array([8.0, 8.0, 4.0])
        self.Ki_att = np.array([1.0, 1.0, 0.5])
        self.att_int = np.zeros(3)
        self.att_int_limit = np.array([0.3, 0.3, 0.3])

        # -----------------------------
        # Rate PID
        # -----------------------------
        self.Kp_rate = np.array([2.5e-4, 2.5e-4, 1.8e-4])
        self.Ki_rate = np.array([1.5e-5, 1.5e-5, 1.5e-5])
        self.Kd_rate = np.array([1.0e-5, 1.0e-5, 1.0e-5])
        self.rate_int = np.zeros(3)
        self.rate_int_limit = np.array([5e-4, 5e-4, 5e-4])

        # -----------------------------
        # Limits
        # -----------------------------
        self.max_angle = np.deg2rad(20)
        self.max_rate = np.deg2rad(500)
        self.max_omega = 2500
        self.tau_max = 3e-4

        # -----------------------------
        # NEW: Support external state injection for Phoenix
        # -----------------------------
        self.use_external_state = False

        self.reset()

    # =========================================================
    #   External state injection (Phoenix → controller)
    # =========================================================
    def inject_external_state(self, x, v, ang, rate):
        """
        Phoenix calls this every step.

        Overwrites internal state so the controller uses
        real PyBullet dynamics instead of internal RK4.
        """
        self.x = np.array(x, dtype=float)
        self.v = np.array(v, dtype=float)
        self.ang = np.array(ang, dtype=float)
        self.rate = np.array(rate, dtype=float)

        # Disable internal dynamics
        self.use_external_state = True

    # =========================================================
    # Reset state
    # =========================================================
    def reset(self):
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.ang = np.zeros(3)
        self.rate = np.zeros(3)

        self.z_int = 0.0
        self.att_int[:] = 0
        self.rate_int[:] = 0

        self.use_external_state = False

    # =========================================================
    # Rotation utilities
    # =========================================================
    @staticmethod
    def Rx(phi):  c, s = np.cos(phi), np.sin(phi); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    @staticmethod
    def Ry(th):   c, s = np.cos(th),  np.sin(th);  return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    @staticmethod
    def Rz(psi):  c, s = np.cos(psi), np.sin(psi); return np.array([[c,-s,0],[s,c,0],[0,0,1]])

    def R(self):
        phi, th, psi = self.ang
        return self.Rz(psi) @ self.Ry(th) @ self.Rx(phi)

    # =========================================================
    #  PID LAYER 1: Position → Desired Angles
    # =========================================================
    def position_pid(self, pos_ref, vel_ref, z_ref=1.0):

        # XY PD
        exy = pos_ref[:2] - self.x[:2]
        evxy = vel_ref[:2] - self.v[:2]
        a_xy = self.Kp_xy * exy + self.Kd_xy * evxy
        a_xy = np.clip(a_xy, -3, 3)

        # Z PID
        ez = z_ref - self.x[2]
        evz = -self.v[2]

        self.z_int += ez * self.dt
        self.z_int = np.clip(self.z_int, -self.z_int_limit, self.z_int_limit)

        a_z = (
            self.Kp_z * ez +
            self.Kd_z * evz +
            self.Ki_z * self.z_int
        )
        a_z = np.clip(a_z, -4, 4)

        # Convert acceleration → thrust
        U1 = self.m * (self.g + a_z)
        U1 = np.clip(U1, 0.5*self.m*self.g, 1.3*self.m*self.g)

        # Map XY acceleration → desired roll & pitch
        psi = self.ang[2]
        ax, ay = a_xy

        phi_des =  (ax*np.sin(psi) - ay*np.cos(psi)) / self.g
        theta_des = (ax*np.cos(psi) + ay*np.sin(psi)) / self.g

        phi_des   = np.clip(phi_des,   -self.max_angle, self.max_angle)
        theta_des = np.clip(theta_des, -self.max_angle, self.max_angle)

        return U1, phi_des, theta_des, 0.0

    # =========================================================
    #  PID LAYER 2: Attitude → Desired Rate
    # =========================================================
    def attitude_pid(self, phi_des, th_des, psi_des):

        phi, th, psi = self.ang
        e = np.array([phi_des - phi, th_des - th, psi_des - psi])

        self.att_int += e * self.dt
        self.att_int = np.clip(self.att_int, -self.att_int_limit, self.att_int_limit)

        rates_des = self.Kp_att * e + self.Ki_att * self.att_int
        return np.clip(rates_des, -self.max_rate, self.max_rate)

    # =========================================================
    #  PID LAYER 3: Rate PID → Torques
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
        return np.clip(tau, -self.tau_max, self.tau_max)

    # =========================================================
    # Mixer
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
    # Dynamics (for standalone sim)
    # =========================================================
    def f(self, state, omega):
        x,y,z, vx,vy,vz, phi,th,psi, p,q,r = state

        # Linear acceleration
        R = self.R()
        thrust = self.b * np.sum(omega**2)
        acc = (thrust * (R @ np.array([0,0,1]))) / self.m - np.array([0,0,self.g])

        # Torques
        b,d,l = self.b, self.d, self.l
        w1,w2,w3,w4 = omega
        tau_phi   = l*b*(w4**2 - w2**2)
        tau_theta = l*b*(w3**2 - w1**2)
        tau_psi   = d*(w1**2 - w2**2 + w3**2 - w4**2)

        # Angular acceleration
        p_dot = ((self.Iy - self.Iz)/self.Ix)*q*r + tau_phi/self.Ix
        q_dot = ((self.Iz - self.Ix)/self.Iy)*p*r + tau_theta/self.Iy
        r_dot = ((self.Ix - self.Iy)/self.Iz)*p*q + tau_psi/self.Iz

        # Euler rates
        cth = max(np.cos(th), 1e-3)
        sth = np.sin(th)
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        T = np.array([
            [1, sphi*sth/cth, cphi*sth/cth],
            [0, cphi,        -sphi       ],
            [0, sphi/cth,     cphi/cth   ]
        ])
        ang_dot = T @ np.array([p,q,r])

        return np.array([
            vx,vy,vz,
            acc[0],acc[1],acc[2],
            ang_dot[0],ang_dot[1],ang_dot[2],
            p_dot,q_dot,r_dot
        ])

    # =========================================================
    #  Unified Step: Works for BOTH Phoenix and RK4 simulator
    # =========================================================
    def step(self, pos_ref, vel_ref, z_ref=1.0):

        # Position PID
        U1, phi_des, th_des, psi_des = self.position_pid(pos_ref, vel_ref, z_ref)

        # Attitude PID → desired rates
        rates_des = self.attitude_pid(phi_des, th_des, psi_des)

        # Rate PID → torques
        tau = self.rate_pid(rates_des)

        # Mixer → motor speeds
        omega = self.mixer(U1, tau)

        # =====================================================
        #  CASE 1: Phoenix is controlling physics
        # =====================================================
        if self.use_external_state:
            return {
                "rates_des": rates_des,
                "thrust_cmd": U1,
                "omega_cmd": omega,
                "x": self.x,
                "v": self.v,
                "ang": self.ang,
                "rate": self.rate
            }

        # =====================================================
        #  CASE 2: Standalone RK4 simulation (your dashboard)
        # =====================================================
        state = np.hstack([self.x, self.v, self.ang, self.rate])
        f = lambda s: self.f(s, omega)
        dt = self.dt

        k1 = f(state)
        k2 = f(state + 0.5*dt*k1)
        k3 = f(state + 0.5*dt*k2)
        k4 = f(state + dt*k3)

        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        # unpack
        self.x    = state[:3]
        self.v    = state[3:6]
        self.ang  = state[6:9]
        self.rate = state[9:12]

        # safety
        if self.x[2] > KILL_HEIGHT:
            print("[SAFETY] Altitude runaway detected.")
            self.reset()

        return {
            "rates_des": rates_des,
            "thrust_cmd": U1,
            "omega_cmd": omega,
            "x": self.x,
            "v": self.v,
            "ang": self.ang,
            "rate": self.rate
        }
