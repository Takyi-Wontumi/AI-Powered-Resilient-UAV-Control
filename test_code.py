from phoenix_drone_simulation.train import train
from AI_UAV_Tests.trajectories_library import Trajectories

train(task="followpath", algo="ppo", total_timesteps=3_000_000,
      trajectory_fn=Trajectories.helix_traj)