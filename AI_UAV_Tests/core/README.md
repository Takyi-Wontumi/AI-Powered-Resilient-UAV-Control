# AI_UAV_Tests Core

This folder contains the reusable building blocks for controller logic, dynamics,
simulation, and trajectory generation.

## Modules

- `quadcopter_env.py`
  - PID controller, shared control helpers, and environment-facing utilities.
- `quadcopter_env_isaac.py`
  - Isaac-specific environment/control integration.
- `quadcopter_dynamics.py`
  - Core dynamics calculations.
- `quadcopter_simulation.py`
  - Standalone simulation support.
- `quadcopter_trajectory_trail.py`
  - Trajectory trail visualization helpers.
- `trajectories_library.py`
  - `FlightMission`, `Trajectories`, and path-generation utilities.
- `flight_tester.py`
  - Flight-test helpers and quick experimentation logic.

## Common imports

```python
from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.core.trajectories_library import FlightMission, Trajectories
```

## Where these modules are used

These core modules are used by:

- `examples/ekf/`
- `examples/rl/`
- `EKF_With_LSTM/`
- `Realworld_Deployment/`

## Typical ways to run code that depends on this folder

Examples:

```powershell
python .\examples\ekf\follow_path_dropout_ekf_mission.py --render
python .\examples\train_circle_with_trail.py --help
python .\examples\train_mission_trajectory.py --help
```

## Expected outputs

Because this folder mainly contains modules rather than standalone reports, the outputs
depend on the caller:

- mission scripts generate plots and terminal metrics
- trajectory scripts generate reference paths
- RL scripts generate checkpoints and logs

## Notes

- Import from the organized subpackages directly to avoid ambiguity.
