import numpy as np
import matplotlib.pyplot as plt

# Import Crazyflie inner-loop controllers
from phoenix_drone_simulation.envs.control import Attitude, AttitudeRate, rpy_control_factors_to_PWM

# =====================================================================
#   QUADCOPTER PLANT — CRAZYFLIE-STYLE CASCADED CONTROL SIMULATION
# =====================================================================

class QuadcopterPlant:
    """
    12-state quadcopter with cascaded control identical to Crazyflie firmware:
        Position PD  → Attitude PID → Rate PID → Mixer → Motors → Dynamics
    """

    def __init__(self, dt=0.002, g=9.81):
        self.dt = dt
        self.g = g

        # =============================
        # Crazyflie Physical Parameters
        # =============================
        self.m = 0.028      # kg
        self.l = 0.046      # arm length [m]
        self.Ix = 16.6e-6
        self.Iy = 16.6e-6
        self.Iz = 29.3e-6
        self.b = 1.4e-6     # thrust coeff
        self.d = 1.1e-7     # torque coeff

        # =============================
        # Outer-loop Position + Altitude gains (SAFE)
        # =============================
        self.Kp_z = 0.8
        self.Kd_z = 0.3

        self.Kp_xy = np.array([0.8, 0.8])
        self.Kd_xy = np.array([0.6, 0.6])

        # =============================
        # Thrust model (SAFE LINEAR APPROX)
        # =============================
        # 60000 PWM ≈ 0.04 N per motor → ~0.16 N total → enough for hover
        self.k_thrust = 7e-7      # N per PWM unit
        self.PWM_HOVER = 45000.0

        self.U1_min = 0.3 * self.m * self.g
        self.U1_max = 1.7 * self.m * self.g

        # =============================
        # State
        # =============================
        self.reset()

        # =============================
        # Crazyflie Inner-loop Controllers
        # =============================
        self.attitude_ctrl = Attitude(
            drone=self, bc=None, time_step=self.dt
        )
        self.rate_ctrl = self.attitude_ctrl.attitude_rate_controller

        # =============================
        # Fix 1: Clamp derivative inside AttitudeRate
        # =============================
        self._patch_attitude_rate()

    # -------------------------------------------------------
    #  Patch derivative in AttitudeRate to avoid explosion
    # -------------------------------------------------------
    def _patch_attitude_rate(self):
        original = self.rate_ctrl.compute_output

        def safe_compute(self_rate, rpy_dot_target):
            dt = self_rate.time_step

            # error in degrees (Crazyflie firmware convention)
            error = self_rate.rad_to_degree(
                rpy_dot_target - self_rate.drone.rpy_dot
            )

            # clamp derivative
            derivative = (error - self_rate.last_error) / dt
            derivative = np.clip(derivative, -500, 500)

            self_rate.last_error = error
            self_rate.integral += error * dt
            self_rate.integral = np.clip(
                self_rate.integral,
                -self_rate.rpy_rate_integral_limits,
                self_rate.rpy_rate_integral_limits
            )

            rpy_offsets = (
                self_rate.kp_att_rate * error +
                self_rate.ki_att_rate * self_rate.integral +
                self_rate.kd_att_rate * derivative
            )
            return rpy_offsets

        # Monkey-patch
        self.rate_ctrl.compute_output = safe_compute.__get__(self.rate_ctrl)

    # -------------------------------------------------------
    # Expose rpy and rpy_dot for Crazyflie controllers
    # -------------------------------------------------------
    @property
    def rpy(self):
        return self.ang.copy()

    @property
    def rpy_dot(self):
        return self.rate.copy()

    # -------------------------------------------------------
    # Reset state
    # -------------------------------------------------------
    def reset(self):
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.ang = np.zeros(3)      # roll, pitch, yaw
        self.rate = np.zeros(3)     # p, q, r

    # -------------------------------------------------------
    # Rotation matrices
    # -------------------------------------------------------
    @staticmethod
    def Rx(phi):
        c, s = np.cos(phi), np.sin(phi)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def Ry(theta):
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def Rz(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def R_wb(self, phi, theta, psi):
        return self.Rz(psi) @ self.Ry(theta) @ self.Rx(phi)

    # -------------------------------------------------------
    # Full rigid-body dynamics
    # -------------------------------------------------------
    def _dynamics(self, s, F):
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = s

        f1, f2, f3, f4 = F
        m = self.m
        g = self.g
        Ix, Iy, Iz = self.Ix, self.Iy, self.Iz
        l, b, d = self.l, self.b, self.d

        # Thrust + torques
        U1 = f1 + f2 + f3 + f4
        tau_phi = l * (f4 - f2)
        tau_theta = l * (f3 - f1)
        tau_psi = (d / b) * (f1 - f2 + f3 - f4)

        # Linear acceleration
        zb = self.R_wb(phi, theta, psi) @ np.array([0, 0, 1])
        acc = (U1 / m) * zb - np.array([0, 0, g])
        ax, ay, az = acc

        # Angular acceleration
        p_dot = ((Iy - Iz) * q * r + tau_phi) / Ix
        q_dot = ((Iz - Ix) * p * r + tau_theta) / Iy
        r_dot = ((Ix - Iy) * p * q + tau_psi) / Iz

        # Euler angle rates
        T = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        ang_dot = T @ np.array([p, q, r])

        return np.array([
            vx, vy, vz,
            ax, ay, az,
            *ang_dot,
            p_dot, q_dot, r_dot
        ])

    # -------------------------------------------------------
    # Main controller + physics step
    # -------------------------------------------------------
    def step(self, pos_ref, vel_ref, psi_ref=0.0):
        # ---------------------------
        # Position PD (outer XY)
        # ---------------------------
        exy = pos_ref[:2] - self.x[:2]
        evxy = vel_ref[:2] - self.v[:2]
        a_des = self.Kp_xy * exy + self.Kd_xy * evxy

        # ---------------------------
        # Altitude PD (outer Z)
        # ---------------------------
        ez = pos_ref[2] - self.x[2]
        evz = vel_ref[2] - self.v[2]

        U1 = self.m * (self.g + self.Kp_z * ez + self.Kd_z * evz)
        U1 = float(np.clip(U1, self.U1_min, self.U1_max))

        # Convert thrust → PWM
        thrust_pwm = U1 / (4 * self.k_thrust)
        u0 = np.clip((thrust_pwm - 45000.0) / 10000.0, -1, 1)

        # ---------------------------
        # Desired roll/pitch
        # ---------------------------
        psi = self.ang[2]
        g = self.g

        phi_des = (a_des[0]*np.sin(psi) - a_des[1]*np.cos(psi)) / g
        theta_des = (a_des[0]*np.cos(psi) + a_des[1]*np.sin(psi)) / g

        rpy_des = np.array([phi_des, theta_des, psi_ref])

        # ---------------------------
        # Attitude PID → desired body rates
        # ---------------------------
        rpy_dot_des = self.attitude_ctrl.compute_output(rpy_des)

        # ---------------------------
        # Rate PID → control factors
        # ---------------------------
        rpy_factors = self.rate_ctrl.compute_output(rpy_dot_des)

        # ---------------------------
        # Mixer → PWMs
        # ---------------------------
        PWMs = rpy_control_factors_to_PWM(
            rpy_factors,
            thrust=45000.0 + u0 * 10000.0
        )

        # PWM → forces
        motor_forces = self.k_thrust * PWMs

        # ---------------------------
        # Integrate dynamics (RK4)
        # ---------------------------
        s = np.hstack([self.x, self.v, self.ang, self.rate])
        dt = self.dt

        f = lambda st: self._dynamics(st, motor_forces)
        k1 = f(s)
        k2 = f(s + 0.5*dt*k1)
        k3 = f(s + 0.5*dt*k2)
        k4 = f(s + dt*k3)

        s = s + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        # unpack
        self.x = s[:3]
        self.v = s[3:6]
        self.ang = s[6:9]
        self.rate = s[9:12]

        return {
            "px": self.Kp_xy[0]*exy[0], "dx": self.Kd_xy[0]*evxy[0],
            "py": self.Kp_xy[1]*exy[1], "dy": self.Kd_xy[1]*evxy[1],
            "pz": self.Kp_z*ez,         "dz": self.Kd_z*evz
        }


# =====================================================================
#   SIM WRAPPER
# =====================================================================

class QuadcopterSim:
    def __init__(self, trajectory_fn, dt=0.002):
        self.trajectory_fn = trajectory_fn
        self.drone = QuadcopterPlant(dt=dt)
        self.dt = dt
        self.results = {}

    def simulate(self, t_final=10.0):
        quad = self.drone
        N = int(t_final / quad.dt)
        t = np.arange(N) * quad.dt

        pos_log = []
        ref_log = []
        pid_log = []

        for ti in t:
            pos_ref, vel_ref = self.trajectory_fn(ti)
            pid_terms = quad.step(pos_ref, vel_ref)

            pos_log.append(quad.x.copy())
            ref_log.append(pos_ref.copy())
            pid_log.append(pid_terms)

        self.results = dict(
            t=t,
            pos=np.vstack(pos_log),
            ref=np.vstack(ref_log),
            pid=pid_log
        )
        return self.results

    # ------------------------------------------
    def plot_xyz(self):
        t = self.results["t"]
        pos = self.results["pos"]
        ref = self.results["ref"]

        fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        lbls = ["X", "Y", "Z"]

        for i in range(3):
            axs[i].plot(t, pos[:, i], label=lbls[i])
            axs[i].plot(t, ref[:, i], "--", label=f"{lbls[i]}_ref")
            axs[i].grid(True)
            axs[i].legend()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------
    def plot_pid(self):
        t = self.results["t"]
        pid = self.results["pid"]

        px = np.array([p["px"] for p in pid])
        dx = np.array([p["dx"] for p in pid])
        py = np.array([p["py"] for p in pid])
        dy = np.array([p["dy"] for p in pid])
        pz = np.array([p["pz"] for p in pid])
        dz = np.array([p["dz"] for p in pid])

        fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        axs[0].plot(t, px, label="P_x"); axs[0].plot(t, dx, "--", label="D_x")
        axs[1].plot(t, py, label="P_y"); axs[1].plot(t, dy, "--", label="D_y")
        axs[2].plot(t, pz, label="P_z"); axs[2].plot(t, dz, "--", label="D_z")

        for a in axs:
            a.grid(True)
            a.legend()

        plt.tight_layout()
        plt.show()

    # ------------------------------------------
    def animate(self, speed=1.0):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.animation as animation

        t = self.results["t"]
        pos = self.results["pos"]
        ref = self.results["ref"]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        # reference path
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], "k--", label="reference")

        traj, = ax.plot([], [], [], "b-", label="trajectory")
        point, = ax.plot([], [], [], "ro")

        def update(i):
            traj.set_data(pos[:i, 0], pos[:i, 1])
            traj.set_3d_properties(pos[:i, 2])
            point.set_data([pos[i, 0]], [pos[i, 1]])
            point.set_3d_properties([pos[i, 2]])
            return traj, point

        interval = int((self.dt * 1000) / speed)
        ani = animation.FuncAnimation(fig, update, frames=len(t),
                                      interval=interval, blit=True)

        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.legend()
        plt.show()
