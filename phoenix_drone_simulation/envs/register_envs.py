from gymnasium.envs.registration import register
from AI_UAV_Tests.trajectories_library import Trajectories as T

def register_all_envs():
    # Default (circle)
    register(
        id="DroneFollowPathEnv-v0",
        entry_point="phoenix_drone_simulation.envs.followpath:DroneFollowPathEnv",
        kwargs={"trajectory_fn": T.circle_traj},
    )

    register(
        id="DroneHoverEnv-v0",
        entry_point="phoenix_drone_simulation.envs.followpath:DroneFollowPathEnv",
        kwargs={"trajectory_fn": T.hover_traj},
    )

    register(
        id="DroneSquareEnv-v0",
        entry_point="phoenix_drone_simulation.envs.followpath:DroneFollowPathEnv",
        kwargs={"trajectory_fn": T.square_traj},
    )

    register(
        id="DroneHelixEnv-v0",
        entry_point="phoenix_drone_simulation.envs.followpath:DroneFollowPathEnv",
        kwargs={"trajectory_fn": T.helix_traj},
    )

    register(
        id="DroneSineEnv-v0",
        entry_point="phoenix_drone_simulation.envs.followpath:DroneFollowPathEnv",
        kwargs={"trajectory_fn": T.sine_traj},
    )

    # Eformat for adding new paths
    # register(
    #     id="DroneSpiralEnv-v0",
    #     entry_point="phoenix_drone_simulation.envs.followpath:DroneFollowPathEnv",
    #     kwargs={"trajectory_fn": T.spiral_traj},
    #     )
