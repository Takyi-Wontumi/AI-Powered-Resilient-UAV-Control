# Phoenix-Drone-Simulation (Fork)

This repository is a fork of [SvenGronauer/phoenix-drone-simulation](https://github.com/SvenGronauer/phoenix-drone-simulation), extended for trajectory tracking, new environment configurations, and control experiments.

It provides Gymnasium-compatible quadrotor simulation environments based on PyBullet, with dynamics modeled around the Bitcraze Crazyflie 2.1 nano quadrotor.

Circle Task | TakeOff
--- | ---
![Circle](./docs/readme/circle3.gif) | ![TakeOff](./docs/readme/takeoff.gif)

## What This Fork Adds

- Additional trajectory-following environments (circle, square, helix, sine, hover path variants).
- Custom mission and control experiments in `AI_UAV_Tests/`.
- Integration-oriented scripts for PID + RL workflows.
- Expanded experiment scripts for dropout/noise and trajectory tracking scenarios.

## Available Environments

### Core environments (registered on `import phoenix_drone_simulation`)

| Environment ID | Task | Physics |
|---|---|---|
| `DroneHoverSimpleEnv-v0` | Hover | Simple |
| `DroneHoverBulletEnv-v0` | Hover | PyBullet |
| `DroneCircleSimpleEnv-v0` | Circle | Simple |
| `DroneCircleBulletEnv-v0` | Circle | PyBullet |
| `DroneTakeOffSimpleEnv-v0` | Take-off | Simple |
| `DroneTakeOffBulletEnv-v0` | Take-off | PyBullet |

### Fork trajectory environments (registered via `register_all_envs()`)

| Environment ID | Trajectory |
|---|---|
| `DroneFollowPathEnv-v0` | Circle default |
| `DroneHoverEnv-v0` | Hover |
| `DroneSquareEnv-v0` | Square |
| `DroneHelixEnv-v0` | Helix |
| `DroneSineEnv-v0` | Sine |

## Installation

### 1. Clone this fork (not upstream)

```bash
git clone https://github.com/Takyi-Wontumi/AI-Powered-Resilient-UAV-Control.git
cd AI-Powered-Resilient-UAV-Control
```

### 2. Create and activate a virtual environment (recommended)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install package dependencies

```bash
pip install -e .
```

## Quick Start

### Create and step a basic environment

```python
import gymnasium as gym
import phoenix_drone_simulation

env = gym.make("DroneHoverBulletEnv-v0")
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

### Use fork-specific trajectory environments

```python
import gymnasium as gym
import phoenix_drone_simulation
from phoenix_drone_simulation.envs.register_envs import register_all_envs

register_all_envs()
env = gym.make("DroneSquareEnv-v0")
```

## Training

Train PPO on a standard hover task:

```bash
python -m phoenix_drone_simulation.train --alg ppo --env DroneHoverBulletEnv-v0
```

Train PPO on this fork's path-following environment:

```bash
python -m phoenix_drone_simulation.train --alg ppo --env DroneFollowPathEnv-v0 --no-mpi
```

Notes:

- Supported algorithms in this codebase include `ppo`, `trpo`, `npg`, and `iwpg`.
- Use `--no-mpi` if MPI is not configured on your machine.

## Playback

Run a saved checkpoint:

```bash
python -m phoenix_drone_simulation.play --ckpt PATH_TO_CKPT
```

Run a random policy in an environment:

```bash
python -m phoenix_drone_simulation.play --env DroneSquareEnv-v0 --random
```

## Useful Example Scripts

- `examples/train_drone_hover.py`
- `examples/train_takeoff_hover.py`
- `examples/train_mission_trajectory.py`
- `examples/follow_path_test.py`
- `examples/follow_path_dropout_mission.py`
- `examples/generate_trajectories.py`

## Requirements

The package currently declares (via `setup.py`):

- Python `>=3.8`
- `gymnasium>=0.29.1`
- `pybullet`
- `torch`
- `numpy==1.24.4`
- `scipy`, `matplotlib`, `pandas`, `tensorboard`, `mpi4py`, `joblib`, `psutil`

## Upstream Publication

Sven Gronauer, Matthias Kissel, Luca Sacchetto, Mathias Korte, Klaus Diepold.  
Using Simulation Optimization to Improve Zero-shot Policy Transfer of Quadrotors.  
https://arxiv.org/abs/2201.01369

## Acknowledgements

- Upstream project by Sven Gronauer and contributors.
- Gym-PyBullet-Drones contributors for foundational simulation work.
- Bitcraze ecosystem and Crazyflie community for hardware and modeling references.
