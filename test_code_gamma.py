from AI_UAV_Tests.trajectories_library import Mission_Planner

PATH = r"C:\Users\Lawrence Wontumi\Downloads\AI-Powered-Resilient-UAV-Control\Realworld_Deployment\Mission Planner\ChapelHill_Test2.mission"

TOTAL_TIME = 60.0

traj = Mission_Planner(PATH, total_time=TOTAL_TIME)

traj.add_takeoff(t_start=0.0, duration=3.0, target_z=1.0)
traj.add_hover(t_start=3.0, duration=2.0)
traj.add_landing(t_start=55.0, duration=5.0, ground_z=0.0)

for t_sec in range(0, int(TOTAL_TIME)):
    pos, vel = traj(float(t_sec))
    print(f"t={t_sec:2d}s  pos={pos}")

print(traj.summary())
