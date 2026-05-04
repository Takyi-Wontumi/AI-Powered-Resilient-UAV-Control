# Phoenix-Drone-Simulation (Fork)

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

Before RL, verify that baseline control is stable. Go to /examples to run code that uses the baseline PID control

Run:

```bash
python examples/follow_path_test.py
```

or:

```bash
python examples/train_takeoff_hover.py
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

| Script | Purpose |
|---|---|
| `examples/follow_path_test.py` | PID trajectory tracking |
| `examples/train_takeoff_hover.py` | Basic stabilization |
| `examples/train_mission_trajectory.py` | RL trajectory training |
| `examples/follow_path_dropout_mission.py` | Dropout experiments |

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

## 12. Extended Kalman Filter (EKF) State Estimator

The EKF provides robust state estimation for 12-state quadcopter dynamics (position, velocity, attitude, body rates) with GPS dropout resilience.

### Quick Start

**Run 30-trial Monte Carlo validation** (generates publication-grade plots):

```bash
python examples/ekf_validation.py --n-trials 30 --save-dir results/ --show
```

This generates 6 figures in `results/`:
- **fig1_jacobian.png** — Analytical vs numerical Jacobian validation
- **fig2_nees.png** — Normalized Estimation Error Squared with χ² bounds
- **fig3_nis.png** — Normalized Innovation Squared per DOF
- **fig4_rmse.png** — Position RMSE: EKF-PID vs Raw-sensor PID
- **fig5_coverage.png** — 3-sigma covariance coverage per state dimension
- **fig6_dropout_resilience.png** — Position error during GPS dropout

Expected results:
- Mean NEES: 11.81 ± 0.19 (target: 12 ± 0.25) ✓ PASS
- Mean NIS / DOF: 0.903 (expected: 1.0)
- 3-sigma coverage: 100.0% all axes (ideal: 99.7%)

### Real-Time GUI Simulation

**Launch PyBullet visualization with EKF feedback control**:

```bash
python examples/follow_path_dropout_ekf_mission.py --render
```

The drone executes:
- Takeoff (3s) → altitude 0→1.0m
- Square trajectory (12s) → 1.0m square at z=1.0m
- Hover (2s) → position hold
- Circle trajectory (12s) → 0.75m radius circle
- Landing (6s) → descent to ground

**Keyboard controls** (when GUI is running):
- **H** = Trigger hover mode dropout
- **R** = Trigger return-to-home
- **L** = Trigger landing
- **C** = Clear dropout and resume trajectory

Generates end-of-run plot: Reference vs Measured position (X/Y/Z).

### Headless Mode (no GUI)

```bash
python examples/follow_path_dropout_ekf_mission.py
```

Runs the mission silently and generates the reference vs measured plot.

### EKF Tuning Configuration

Current tuning (production-ready):
- **Process noise scale (q_scale)**: 0.0547
- **Measurement noise scale (r_scale)**: 0.1047
- **State dimension**: 12 (x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r)

Located in: `AI_UAV_Tests/quadcopter_ekf.py` (lines 66-106)

### EKF API

```python
from AI_UAV_Tests.quadcopter_ekf import PhoenixEKFAdapter

ekf = PhoenixEKFAdapter(dt=0.01)

# Reset with initial state
ekf.reset(
    position=[0, 0, 0],
    velocity=[0, 0, 0],
    attitude=[0, 0, 0],
    rates=[0, 0, 0]
)

# Step the estimator
estimate = ekf.step(
    motor_omega=[w1, w2, w3, w4],
    position=[x, y, z],
    velocity=[vx, vy, vz],
    attitude=[roll, pitch, yaw],
    rates=[p, q, r],
    dropout_active=False,
    dt=0.01
)

# Access estimates
print(estimate["x"])              # Position [x, y, z]
print(estimate["v"])             # Velocity [vx, vy, vz]
print(estimate["ang"])           # Attitude [roll, pitch, yaw]
print(estimate["rate"])          # Body rates [p, q, r]
print(estimate["measurement"])   # Raw sensor measurement
```

### Validation Against Raw Sensor Baseline

The validation suite compares EKF state estimates against a raw-sensor PID controller:
- **EKF position RMSE**: 0.43 cm
- **Raw sensor RMSE**: 0.23 cm
- **EKF consistency**: PASS (NEES within χ² bounds)
- **Dropout resilience**: 100% covariance coverage during 4s GPS blackout

## Upstream Publication

Sven Gronauer, Matthias Kissel, Luca Sacchetto, Mathias Korte, Klaus Diepold.  
Using Simulation Optimization to Improve Zero-shot Policy Transfer of Quadrotors.  
https://arxiv.org/abs/2201.01369
