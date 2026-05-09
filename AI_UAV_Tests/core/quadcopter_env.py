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
        # Position / attitude / rate control profiles
        # -----------------------------
        self.control_profiles = {
            "nominal": {
                "Kp_xy": np.array([5.0, 5.0]),
                "Kd_xy": np.array([3.0, 3.0]),
                "Kp_z": 16.2,
                "Ki_z": 7.0175,
                "Kd_z": 9.3494,
                "Kp_att": np.array([8.0, 8.0, 4.0]),
                "Ki_att": np.array([3.0, 3.0, 0.5]),
                "Kp_rate": np.array([2.5e-4, 2.5e-4, 1.8e-4]),
                "Ki_rate": np.array([1.5e-5, 1.5e-5, 1.5e-5]),
                "Kd_rate": np.array([1.0e-5, 1.0e-5, 1.0e-5]),
                "max_angle": np.deg2rad(20.0),
                "max_rate": np.deg2rad(500.0),
                "tau_max": 3.0e-4,
                "max_xy_acc": 3.0,
                "max_z_acc": 4.0,
                "thrust_min_scale": 0.5,
                "thrust_max_scale": 1.3,
            },
            "dropout": {
                "Kp_xy": np.array([1.2, 1.2]),
                "Kd_xy": np.array([1.8, 1.8]),
                "Kp_z": 8.0,
                "Ki_z": 1.0,
                "Kd_z": 5.0,
                "Kp_att": np.array([5.5, 5.5, 3.0]),
                "Ki_att": np.array([0.8, 0.8, 0.2]),
                "Kp_rate": np.array([2.0e-4, 2.0e-4, 1.4e-4]),
                "Ki_rate": np.array([0.5e-5, 0.5e-5, 0.5e-5]),
                "Kd_rate": np.array([0.9e-5, 0.9e-5, 0.9e-5]),
                "max_angle": np.deg2rad(12.0),
                "max_rate": np.deg2rad(330.0),
                "tau_max": 2.2e-4,
                "max_xy_acc": 1.5,
                "max_z_acc": 2.5,
                "thrust_min_scale": 0.65,
                "thrust_max_scale": 1.18,
            },
        }

        nominal = self.control_profiles["nominal"]
        self.Kp_xy = nominal["Kp_xy"].copy()
        self.Kd_xy = nominal["Kd_xy"].copy()
        self.Kp_z = float(nominal["Kp_z"])
        self.Ki_z = float(nominal["Ki_z"])
        self.Kd_z = float(nominal["Kd_z"])
        self.z_int = 0.0
        self.z_int_limit = 1.0

        # -----------------------------
        # Attitude PID
        # -----------------------------
        self.Kp_att = nominal["Kp_att"].copy()
        self.Ki_att = nominal["Ki_att"].copy()
        self.att_int = np.zeros(3)
        self.att_int_limit = np.array([0.3, 0.3, 0.3])

        # -----------------------------
        # Rate PID
        # -----------------------------
        self.Kp_rate = nominal["Kp_rate"].copy()
        self.Ki_rate = nominal["Ki_rate"].copy()
        self.Kd_rate = nominal["Kd_rate"].copy()
        self.rate_int = np.zeros(3)
        self.rate_int_limit = np.array([5e-4, 5e-4, 5e-4])

        # -----------------------------
        # Limits
        # -----------------------------
        self.max_angle = float(nominal["max_angle"])
        self.max_rate = float(nominal["max_rate"])
        self.max_omega = 2500
        self.tau_max = float(nominal["tau_max"])

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

    def _control_profile(self, control_profile="nominal"):
        profile_name = str(control_profile).strip().lower()
        if profile_name not in self.control_profiles:
            raise ValueError(
                f"Unknown control profile '{control_profile}'. "
                f"Expected one of: {sorted(self.control_profiles)}"
            )
        return profile_name, self.control_profiles[profile_name]

    # =========================================================
    #  PID LAYER 1: Position → Desired Angles
    # =========================================================
    def position_pid(
        self,
        pos_ref,
        vel_ref,
        z_ref=1.0,
        freeze_z_integrator=False,
        control_profile="nominal",
    ):
        _, profile = self._control_profile(control_profile)

        # XY PD
        exy = pos_ref[:2] - self.x[:2]
        evxy = vel_ref[:2] - self.v[:2]
        a_xy = profile["Kp_xy"] * exy + profile["Kd_xy"] * evxy
        a_xy = np.clip(a_xy, -float(profile["max_xy_acc"]), float(profile["max_xy_acc"]))

        # Z PID
        ez = z_ref - self.x[2]
        evz = -self.v[2]

        if not freeze_z_integrator:
            self.z_int += ez * self.dt
            self.z_int = np.clip(self.z_int, -self.z_int_limit, self.z_int_limit)

        a_z = (
            float(profile["Kp_z"]) * ez +
            float(profile["Kd_z"]) * evz +
            float(profile["Ki_z"]) * self.z_int
        )
        a_z = np.clip(a_z, -float(profile["max_z_acc"]), float(profile["max_z_acc"]))

        # Convert acceleration → thrust
        U1 = self.m * (self.g + a_z)
        U1 = np.clip(
            U1,
            float(profile["thrust_min_scale"]) * self.m * self.g,
            float(profile["thrust_max_scale"]) * self.m * self.g,
        )

        # Map XY acceleration → desired roll & pitch
        psi = self.ang[2]
        ax, ay = a_xy

        phi_des =  (ax*np.sin(psi) - ay*np.cos(psi)) / self.g
        theta_des = (ax*np.cos(psi) + ay*np.sin(psi)) / self.g

        max_angle = float(profile["max_angle"])
        phi_des = np.clip(phi_des, -max_angle, max_angle)
        theta_des = np.clip(theta_des, -max_angle, max_angle)

        return U1, phi_des, theta_des, 0.0

    # =========================================================
    #  PID LAYER 2: Attitude → Desired Rate
    # =========================================================
    def attitude_pid(self, phi_des, th_des, psi_des, control_profile="nominal"):
        _, profile = self._control_profile(control_profile)

        phi, th, psi = self.ang
        e = np.array([phi_des - phi, th_des - th, psi_des - psi])

        self.att_int += e * self.dt
        self.att_int = np.clip(self.att_int, -self.att_int_limit, self.att_int_limit)

        rates_des = profile["Kp_att"] * e + profile["Ki_att"] * self.att_int
        max_rate = float(profile["max_rate"])
        return np.clip(rates_des, -max_rate, max_rate)

    # =========================================================
    #  PID LAYER 3: Rate PID → Torques
    # =========================================================
    def rate_pid(self, rates_des, control_profile="nominal"):
        _, profile = self._control_profile(control_profile)
        er = rates_des - self.rate

        self.rate_int += er * self.dt
        self.rate_int = np.clip(self.rate_int, -self.rate_int_limit, self.rate_int_limit)

        tau = (
            profile["Kp_rate"] * er +
            profile["Ki_rate"] * self.rate_int -
            profile["Kd_rate"] * self.rate
        )
        tau_max = float(profile["tau_max"])
        return np.clip(tau, -tau_max, tau_max)

    # =========================================================
    # Mixer
    # =========================================================
    def mixer(self, U1, tau):
        tau_phi, tau_theta, tau_psi = tau
        u = np.array([U1, tau_phi, tau_theta, tau_psi])

        omega_sq = self.M_inv @ u
        omega_sq = np.clip(omega_sq, 0, np.inf)
        omega_sq = np.clip(omega_sq, 0, self.max_omega**2)
        return np.sqrt(omega_sq)

    def motor_forces(self, omega):
        return self.b * np.asarray(omega, dtype=float) ** 2

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
    def step(
        self,
        pos_ref,
        vel_ref,
        z_ref=1.0,
        freeze_z_integrator=False,
        control_profile="nominal",
    ):
        profile_name, profile = self._control_profile(control_profile)

        # Position PID
        U1, phi_des, th_des, psi_des = self.position_pid(
            pos_ref,
            vel_ref,
            z_ref,
            freeze_z_integrator=freeze_z_integrator,
            control_profile=profile_name,
        )

        # Attitude PID → desired rates
        rates_des = self.attitude_pid(
            phi_des,
            th_des,
            psi_des,
            control_profile=profile_name,
        )

        # Rate PID → torques
        tau = self.rate_pid(rates_des, control_profile=profile_name)

        # Mixer → motor speeds
        omega = self.mixer(U1, tau)
        motor_forces = self.motor_forces(omega)

        # =====================================================
        #  CASE 1: Phoenix is controlling physics
        # =====================================================
        if self.use_external_state:
            return {
                "rates_des": rates_des,
                "thrust_cmd": U1,
                "tau_cmd": tau,
                "omega_cmd": omega,
                "motor_forces": motor_forces,
                "control_profile": profile_name,
                "tau_limit": float(profile["tau_max"]),
                "max_rate": float(profile["max_rate"]),
                "rate_p_gain": np.asarray(profile["Kp_rate"], dtype=float).copy(),
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
            "tau_cmd": tau,
            "omega_cmd": omega,
            "motor_forces": motor_forces,
            "control_profile": profile_name,
            "tau_limit": float(profile["tau_max"]),
            "max_rate": float(profile["max_rate"]),
            "rate_p_gain": np.asarray(profile["Kp_rate"], dtype=float).copy(),
            "x": self.x,
            "v": self.v,
            "ang": self.ang,
            "rate": self.rate
        }

# =========================================================
# State Buffer 
# =========================================================
class StateBuffer:
    def __init__(self):
        self.x = None
        self.v = None
        self.last_t = None

    def update(self, x, v, t):
        self.x = x.copy()
        self.v = v.copy()
        self.last_t = t

    def predict(self, t):
        if self.x is None:
            raise RuntimeError("StateBuffer used before initialization")
        dt = t - self.last_t
        x_pred = self.x + self.v * dt
        v_pred = self.v
        return x_pred, v_pred
