import numpy as np
from phoenix_drone_simulation.envs.base import DroneBaseEnv


class DroneFollowTrajectoryEnv(DroneBaseEnv):

    def __init__(
        self,
        trajectory_fn,
        control_mode="Attitude",
        sim_freq=200,
        observation_frequency=100,
        done_dist_threshold=9999.0,
        gui=True,
        **kwargs,
    ):
        self.trajectory_fn = trajectory_fn
        self.done_dist_threshold = done_dist_threshold
        self.target_pos = np.array([0., 0., 1.], dtype=np.float32)
        self.init_quaternion = np.array([0., 0., 0., 1.], dtype=np.float32)

        init_xyz = np.array([0., 0., 1.], dtype=np.float32)
        init_rpy = np.zeros(3, dtype=np.float32)
        init_xyz_dot = np.zeros(3, dtype=np.float32)
        init_rpy_dot = np.zeros(3, dtype=np.float32)

        super().__init__(
            control_mode=control_mode,
            drone_model="cf21x_bullet",
            physics="PyBulletPhysics",
            init_xyz=init_xyz,
            init_rpy=init_rpy,
            init_xyz_dot=init_xyz_dot,
            init_rpy_dot=init_rpy_dot,
            observation_frequency=observation_frequency,
            sim_freq=sim_freq,
            gui=gui,
            **kwargs,
        )

    # -----------------------------------------------------------
    # Required abstract methods
    # -----------------------------------------------------------
    def _setup_task_specifics(self):
        vis = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1., 0., 0., 0.7],
        )
        self.target_body_id = self.bc.createMultiBody(
            0, -1, vis, self.target_pos
        )

    def compute_info(self):
        return {}

    def compute_potential(self):
        return float(np.linalg.norm(self.drone.xyz - self.target_pos))

    def get_reference_trajectory(self, horizon):
        ref = np.zeros((3, horizon))
        t0 = self.iteration / self.SIM_FREQ
        for i in range(horizon):
            p, _ = self.trajectory_fn(t0 + i * self.TIME_STEP)
            ref[:, i] = p
        return ref

    # -----------------------------------------------------------
    # Controller-friendly state dict
    # -----------------------------------------------------------
    def get_state(self):
        return {
            "pos": self.drone.xyz.copy(),
            "vel": self.drone.xyz_dot.copy(),
            "rpy": self.drone.rpy.copy(),
            "rpy_dot": self.drone.rpy_dot.copy()
        }

    # -----------------------------------------------------------
    # Observation for RL
    # -----------------------------------------------------------
    def compute_observation(self):
        t = self.iteration / self.SIM_FREQ
        pos_ref, vel_ref = self.trajectory_fn(t)
        self.target_pos = pos_ref

        self.bc.resetBasePositionAndOrientation(
            self.target_body_id, pos_ref, (0, 0, 0, 1)
        )

        error_pos = pos_ref - self.drone.xyz
        error_vel = vel_ref - self.drone.xyz_dot

        return np.concatenate([
            self.drone.xyz,
            self.drone.quaternion,
            self.drone.xyz_dot,
            self.drone.rpy_dot,
            error_pos,
            error_vel
        ])

    # -----------------------------------------------------------
    # No episode termination (debug mode)
    # -----------------------------------------------------------
    def compute_done(self):
        return False

    def compute_reward(self, action):
        return -np.linalg.norm(self.drone.xyz - self.target_pos)

    # -----------------------------------------------------------
    # Reset
    # -----------------------------------------------------------
    def task_specific_reset(self):
        pos_ref, vel_ref = self.trajectory_fn(0.0)
        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id, pos_ref, self.init_quaternion
        )
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id, vel_ref, [0, 0, 0]
        )
