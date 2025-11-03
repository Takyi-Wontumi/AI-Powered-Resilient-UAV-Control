import numpy as np
from quadcopter_simulation import QuadcopterSim
from trajectories_library import Trajectories as path

# =========================================================
# Run Example
# =========================================================
if __name__ == "__main__":
    sim = QuadcopterSim(trajectory_fn=path.square_traj)
    sim.simulate(t_final=20)
    sim.animate(speed=1.0)
    sim.plot_pid()
