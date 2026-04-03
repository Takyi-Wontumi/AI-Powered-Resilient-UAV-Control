# Examples Guide

This file shows how to run scripts in `examples/` from the project root and what you can customize.

## 1. Environment Setup

From repository root:

```powershell
# Option A: activate the prebuilt environment
.\drone_sim_env\Scripts\Activate.ps1

# Option B: create your own venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Deactivate when done:

```powershell
deactivate
```

## 2. Quick Run Pattern

Most scripts are started as:

```powershell
python examples\<script_name>.py
```

For scripts with CLI flags:

```powershell
python examples\<script_name>.py --help
```

## 3. Mission Preflight and Dashboard Controls

Primary script:

```powershell
python examples\follow_path_dropout_mission_3.py
```

Useful preflight customizations:

```powershell
# Disable dashboard window but keep preflight checks
python examples\follow_path_dropout_mission_3.py --no-preflight-dashboard

# Disable preflight completely
python examples\follow_path_dropout_mission_3.py --no-preflight

# Preflight only, then exit
python examples\follow_path_dropout_mission_3.py --preflight-only

# Kinematic preflight at 3x speed without dashboard
python examples\follow_path_dropout_mission_3.py --preflight-mode kinematic --preflight-speed 3 --no-preflight-dashboard

# Physics preflight with custom sampling and speed limits
python examples\follow_path_dropout_mission_3.py --preflight-mode physics --preflight-hz 60 --xy-speed-limit 0.8 --min-z-ref 0.3
```

Mission runtime keyboard controls in this family of scripts:

- `H`: hover at current position
- `R`: return to home
- `L`: land (where supported)
- `C`: clear dropout / resume mission
- `N`: toggle measurement noise (noise scripts)

## 4. Core Follow-Path / Mission Scripts

```powershell
python examples\follow_path_test.py
python examples\follow_path_test_2.py
python examples\follow_path_dropout.py
python examples\follow_path_dropout_2.py
python examples\follow_path_dropout_mission.py
python examples\follow_path_dropout_mission_2.py
python examples\follow_path_dropout_mission_3.py
python examples\follow_path_dropout_mission_noise_log.py
python examples\follow_path_dropout_noise.py
```

Notes:

- `follow_path_dropout_mission_noise_log.py` writes CSV logs for analysis.
- Some behavior is configured directly inside script mission definitions.

## 5. Training Scripts

### Mission trajectory training

```powershell
# Basic mission training
python examples\train_mission_trajectory.py --alg ppo --traj flight_mission --control-mode PWM --epochs 50 --steps-per-epoch 4000

# Circle trajectory with AttitudeRate control and rendering
python examples\train_mission_trajectory.py --traj circle --control-mode AttitudeRate --traj-radius 1.2 --traj-period 14 --render

# Replay from checkpoint
python examples\train_mission_trajectory.py --ckpt "runs\Mission_flight_mission_PWM\2026-03-25__14-25-41\seed_00000" --traj flight_mission --control-mode PWM --play --episodes 5 --print-done-reason
```

Common customizations:

- `--traj` (`circle`, `square`, `helix`, `sine`, `hover`, `flight_mission`)
- `--control-mode` (`PWM`, `Attitude`, `AttitudeRate`)
- `--observation-noise`, `--domain-randomization`, `--motor-thrust-noise`
- `--takeoff-seconds`, `--takeoff-z`
- `--entropy-coef`, `--no-exploration-anneal`, `--no-lr-decay`

### Takeoff + hover training

```powershell
python examples\train_takeoff_hover.py --alg ppo --play
python examples\train_takeoff_hover.py --target-z 1.2 --epochs 30 --until-stable --target-success-rate 0.8
python examples\train_takeoff_hover.py --ckpt "<path_to_seed_dir>" --play --play-episodes 5
```

Common customizations:

- `--target-z`, `--max-episode-s`, hover tolerance flags
- `--observation-noise`, `--domain-randomization`, `--motor-thrust-noise`
- `--until-stable`, `--max-rounds`, `--target-success-rate`

### Circle training with trail

```powershell
python examples\train_circle_with_trail.py --alg ppo --trail --play
python examples\train_circle_with_trail.py --circle-radius 1.0 --circle-period 12 --target-z 1.0 --trail --trail-width 2.2 --trail-color 0.1 0.9 0.9
python examples\train_circle_with_trail.py --ckpt "<path_to_seed_dir>" --trail --play --play-episodes 5
```

Common customizations:

- `--circle-radius`, `--circle-period`, `--takeoff-s`
- `--max-xy-ref-delta`, `--max-z-ref-delta`, `--max-yaw-ref-deg`
- `--trail`, `--trail-color`, `--trail-width`, `--trail-max-points`

## 6. Other Training and Utility Scripts

```powershell
# Single-core hover example
python examples\train_drone_hover.py

# Multi-core PPO example
python examples\train_with_multi_cores.py

# Trajectory data generation (batch mode)
python examples\generate_trajectories.py --env DroneCircleBulletEnv-v0

# Visualize policy trajectories
python examples\generate_trajectories.py --env DroneCircleBulletEnv-v0 --play
```

## 7. SimOpt Scripts

```powershell
python examples\run_simulation_optimization_adam.py --cores 4 --seed 1234
python examples\run_simulation_optimization_SGD.py --cores 4
python examples\run_simulation_optimization_cma_es.py --cores 1
```

Notes:

- These can be compute-heavy, especially CMA-ES settings in the script.
- Use lower core counts first to validate your setup.

## 8. Controller Demo Scripts

```powershell
python examples\hover_with_attitude_PD.py
python examples\hover_with_attitude_PID.py
python examples\takeoff_with_attitude_rate_PID.py
```

## 9. Troubleshooting

- If import errors appear, activate environment and run `pip install -e .` again.
- If PyBullet GUI does not open, verify graphics drivers and avoid remote/headless sessions.
- If training crashes early, first run a baseline script (`follow_path_test.py`) to verify controller stability.
