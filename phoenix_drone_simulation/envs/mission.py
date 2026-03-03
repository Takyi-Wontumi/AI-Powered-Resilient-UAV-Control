import numpy as np
from phoenix_drone_simulation.envs.utils import deg2rad
from phoenix_drone_simulation.envs.hover import DroneHoverBaseEnv


# =====================================================
# Dropout Manager (STRING MODES)
# =====================================================
class DropoutManager:
    VALID_MODES = {"NONE", "HOV", "RTH", "LAND"}

    def __init__(self, mode="NONE"):
        mode = mode.upper()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid dropout mode: {mode}")

        self.mode = mode
        self.active = False

        # Stored references
        self.hold_pos = None

    def trigger(self, current_pos: np.ndarray):
        self.active = True

        if self.mode == "HOV":
            self.hold_pos = current_pos.copy()

    def clear(self):
        self.active = False
        self.hold_pos = None
        self.mode = "NONE"


# =====================================================
# DroneMissionEnv
# =====================================================
class DroneMissionEnv(DroneHoverBaseEnv):
    """
    Mission-capable environment.

    - Inherits ALL Hover behavior
    - Starts on ground
    - Continuous mission (no episodic resets)
    - Dropout-aware mission logic (HOV / RTH / LAND)
    """

    def __init__(
        self,
        physics,
        control_mode,
        drone_model,
        dropout_mode="NONE",
        observation_noise=1,
        domain_randomization=0.0,
        **kwargs
    ):
        # -----------------------------
        # Mission state
        # -----------------------------
        self.mission_time = 0.0
        self.active_target = None
        self.airborne = False

        # Home position (ground)
        self.home_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Mission parameters
        self.rth_altitude = 0.30          # safe RTH altitude [m]
        self.land_descent_rate = 0.20     # m/s
        self.ground_z = 0.05              # landing threshold

        # Dropout logic
        self.dropout_mgr = DropoutManager(dropout_mode)

        # -----------------------------
        # Initialize HoverEnv
        # -----------------------------
        super().__init__(
            physics=physics,
            control_mode=control_mode,
            drone_model=drone_model,
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            target_pos=self.home_pos,
            **kwargs
        )

        # OVERRIDE hover spawn (ground start)
        self.init_xyz = self.home_pos.copy()

    # =====================================================
    # Mission API
    # =====================================================

    def set_target(self, target: np.ndarray):
        self.active_target = np.asarray(target, dtype=np.float32)

    def trigger_dropout(self):
        self.dropout_mgr.trigger(self.drone.xyz)

    def clear_dropout(self):
        self.dropout_mgr.clear()

    # =====================================================
    # Mission reference logic
    # =====================================================

    def get_reference_trajectory(self):
        """
        Required by DroneBaseEnv.
        Returns the position reference based on mission mode.
        """

        # -------------------------
        # Normal mission
        # -------------------------
        if not self.dropout_mgr.active:
            return self.active_target

        mode = self.dropout_mgr.mode

        # -------------------------
        # Hover in place
        # -------------------------
        if mode == "HOV":
            return self.dropout_mgr.hold_pos

        # -------------------------
        # Return to home (NO descent)
        # -------------------------
        if mode == "RTH":
            return np.array([
                self.home_pos[0],
                self.home_pos[1],
                max(self.drone.xyz[2], self.rth_altitude)
            ])

        # -------------------------
        # Land (vertical descent)
        # -------------------------
        if mode == "LAND":
            z_next = max(
                self.drone.xyz[2] - self.land_descent_rate * self.TIME_STEP,
                0.0
            )
            return np.array([
                self.drone.xyz[0],
                self.drone.xyz[1],
                z_next
            ])

        return self.active_target

    # Friendly alias
    def get_mission_reference(self):
        return self.get_reference_trajectory()

    # =====================================================
    # SAFETY LOGIC
    # =====================================================

    def compute_done(self) -> bool:
        """
        Safety termination logic.

        - Ground contact allowed until airborne
        - LAND mode allows touchdown
        - Other modes treat ground contact as failure
        """
        xyz = self.drone.xyz
        rpy = self.drone.rpy
        rpy_dot = self.drone.rpy_dot

        # Detect liftoff once
        if not self.airborne and xyz[2] > 0.15:
            self.airborne = True

        # LAND completion
        if (
            self.dropout_mgr.active
            and self.dropout_mgr.mode == "LAND"
            and xyz[2] <= self.ground_z
        ):
            print("Landing complete.")
            return True

        # Crash detection (after liftoff, non-LAND)
        if self.airborne and xyz[2] < self.ground_z:
            return True

        if (np.abs(rpy[:2]) > deg2rad(60)).any():
            return True

        if (np.abs(rpy_dot) > deg2rad(300)).any():
            return True

        return False

    # =====================================================
    # Step override (done before reward)
    # =====================================================

    def step(self, action: np.ndarray) -> tuple:
        """Step the simulation once forward with done-before-reward ordering."""
        for _ in range(self.aggregate_phy_steps):
            self.physics.step_forward(action)
            self.compute_observation()
            self.iteration += 1

        next_obs = self.compute_history()
        terminated = self.compute_done()
        reward = self.compute_reward(action)
        info = self.compute_info()
        truncated = False
        self.last_action = action
        return next_obs, reward, terminated, truncated, info

    # =====================================================
    # Reward (diagnostic only)
    # =====================================================

    def compute_reward(self, action) -> float:
        ref = self.get_reference_trajectory()
        if ref is None:
            return 0.0
        return -np.linalg.norm(self.drone.xyz - ref)

    # =====================================================
    # Reset semantics (EMERGENCY ONLY)
    # =====================================================

    def task_specific_reset(self):
        """
        HARD reset.
        NOT a new episode.
        """
        self.mission_time = 0.0
        self.active_target = None
        self.airborne = False
        self.dropout_mgr.clear()

        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=self.home_pos,
            ornObj=self.init_quaternion
        )
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=np.zeros(3),
            angularVelocity=np.zeros(3)
        )
