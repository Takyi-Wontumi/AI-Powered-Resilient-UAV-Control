# This is a sub folder test

This repository is a fork of [SvenGronauer/phoenix-drone-simulation](https://github.com/SvenGronauer/phoenix-drone-simulation).  
It extends the original project with trajectory-tracking environments, custom control experiments, and robustness-focused testing workflows.

## Author

- Lawrence Wontumi (fork development and project extensions)
- Original upstream project by Sven Gronauer

## 1. Installation

### Clone this fork (not upstream)

```bash
git clone https://github.com/Takyi-Wontumi/AI-Powered-Resilient-UAV-Control.git
cd AI-Powered-Resilient-UAV-Control
```

### Create a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Or use the prebuilt environment (optional)

```powershell
cd drone_sim_env
.\Scripts\Activate.ps1
```

### Install dependencies

```bash
pip install -e .
```

## 2. Sanity Check (run this first)

Run a basic environment before doing anything else:

```python
import gymnasium as gym
import phoenix_drone_simulation

env = gym.make("DroneHoverBulletEnv-v0")
obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

Expected result:

- The drone moves
- No crashes or NaNs
- The simulation runs smoothly

If this fails, stop and fix your setup before training.

## 3. Register Custom Environments

```python
from phoenix_drone_simulation.envs.register_envs import register_all_envs

register_all_envs()
```

## 4. Test Trajectory Environments

```python
import gymnasium as gym
import phoenix_drone_simulation
from phoenix_drone_simulation.envs.register_envs import register_all_envs

register_all_envs()

env = gym.make("DroneSquareEnv-v0")
obs, _ = env.reset()
```

## 5. Run PID / Baseline Control First

Before RL, verify that baseline control is stable.

Run:

```bash
python examples/no_dropout/follow_path_test.py
```

or:

```bash
python examples/training/train_takeoff_hover.py
```

Expected behavior:

- Smooth takeoff
- Stable hover (around 1 meter)
- Reasonable trajectory tracking

If PID is unstable, fix control or physics first. RL will not solve a broken baseline.

## 6. Debug Tracking Performance

The environment logs tracking error. Use:

```python
env.plot_error()
```

Look for:

- Error decreasing over time
- No unstable oscillations
- No sustained drift above 0.3 m

## 7. Train RL (after PID is working)

### Basic hover training

```bash
python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnv-v0
```

### Trajectory tracking training

```bash
python -m phoenix_drone_simulation.train --alg ppo --env DroneFollowPathEnv-v0 --no-mpi
```

If training fails, check:

- Observation values and ranges
- Reward scale
- Control authority (whether the drone can physically follow commands)

## 8. Playback Trained Policy

```bash
python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT
```

Random policy test:

```bash
python -m phoenix_drone_simulation.play --env DroneSquareEnv-v0 --random
```

## 9. Key Scripts

| Script                                    | Purpose                 |
| ----------------------------------------- | ----------------------- |
| `examples/follow_path_test.py`            | PID trajectory tracking |
| `examples/train_takeoff_hover.py`         | Basic stabilization     |
| `examples/train_mission_trajectory.py`    | RL trajectory training  |
| `examples/follow_path_dropout_mission.py` | Dropout experiments     |

## 10. Key Concepts

### Observation space includes:

- Position
- Velocity
- Orientation
- Angular velocity
- Tracking error

### Reward function:

- Penalizes distance from trajectory
- Penalizes excessive control effort
- Penalizes instability

## 11. Recommended Workflow

1. Install
2. Run a random simulation
3. Verify trajectory environments
4. Run PID baseline control
5. Check tracking error
6. Train RL
7. Playback trained policies
8. Run dropout and robustness experiments

## Upstream Publication

Sven Gronauer, Matthias Kissel, Luca Sacchetto, Mathias Korte, Klaus Diepold.  
Using Simulation Optimization to Improve Zero-shot Policy Transfer of Quadrotors.  
https://arxiv.org/abs/2201.01369
