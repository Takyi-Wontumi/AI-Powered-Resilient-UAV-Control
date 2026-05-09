from gymnasium.envs.registration import register
from AI_UAV_Tests.core.trajectories_library import Trajectories as T
from AI_UAV_Tests.core.trajectories_library import FlightMission

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

    mission = FlightMission(default_z=1.0, ground_z=0.0)
    mission.add_takeoff(duration=3.0, target_z=1.0)
    mission.add_square(duration=12.0, side=1.0, period=12.0, z=1.0, offset_xy=(-0.5, -0.5))
    mission.add_hover(duration=2.0, z=1.0)
    mission.add_landing(duration=4.0, ground_z=0.055)

    register(
        id="DroneFollowPathDropoutMissionEnv-v0",
        entry_point="phoenix_drone_simulation.envs.followpath_dropout_mission:DroneFollowPathDropoutMissionEnv",
        kwargs={
            "trajectory_fn": mission,
            "physics": "PyBulletPhysics",
            "control_mode": "AttitudeRate",
            "drone_model": "cf21x_bullet",
            "dropout_mode": "NONE",
        },
    )

    # Eformat for adding new paths
    # register(
    #     id="DroneSpiralEnv-v0",
    #     entry_point="phoenix_drone_simulation.envs.followpath:DroneFollowPathEnv",
    #     kwargs={"trajectory_fn": T.spiral_traj},
    #     )
