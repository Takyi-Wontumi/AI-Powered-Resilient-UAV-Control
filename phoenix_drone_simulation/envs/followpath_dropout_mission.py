import numpy as np

from phoenix_drone_simulation.envs.mission import DroneMissionEnv


class DroneFollowPathDropoutMissionEnv(DroneMissionEnv):
    """Mission env that advances a trajectory unless dropout logic overrides it."""

    def __init__(self, trajectory_fn, **kwargs):
        self.trajectory_fn = trajectory_fn
        super().__init__(**kwargs)

    def current_reference(self, t: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        if t is None:
            t = self.mission_time
        pos_ref, vel_ref = self.trajectory_fn(float(t))
        return (
            np.asarray(pos_ref, dtype=np.float32),
            np.asarray(vel_ref, dtype=np.float32),
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        pos_ref, _ = self.current_reference(0.0)
        self.set_target(pos_ref)
        return obs, info

    def step(self, action: np.ndarray) -> tuple:
        self.mission_time += self.TIME_STEP
        if not self.dropout_mgr.active:
            pos_ref, _ = self.current_reference(self.mission_time)
            self.set_target(pos_ref)
        return super().step(action)
