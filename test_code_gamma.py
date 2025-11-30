from AI_UAV_Tests.quadcopter_simulation import QuadcopterSim
from AI_UAV_Tests.trajectories_library import Trajectories as path

if __name__ == "__main__":
    sim = QuadcopterSim(trajectory_fn=path.square_traj)
    sim.simulate(t_final=20)
    sim.animate(speed=1.0)
    sim.plot_pid()
