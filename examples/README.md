# Examples Guide

The `examples/` directory is now organized by behavior so it is easier to find the right entry point.

## Folder map

- `examples/no_dropout/`
  - nominal flight and controller demos without the EKF dropout workflow
- `examples/dropout_ekf/`
  - dropout mission scripts and EKF/dropout mission demos
- `examples/ekf/`
  - EKF validation, EKF diagnostics, and the tuned EKF dropout mission
- `examples/lstm/`
  - older example-based EKF+LSTM flow
- `examples/rl/`
  - RL training and RL evaluation scripts
- `examples/training/`
  - general non-dropout training scripts
- `examples/utilities/`
  - utilities like GIF recording, trajectory generation, and sim optimization

## Quick start by task

### 1. Nominal no-dropout flight

```powershell
python .\examples\no_dropout\follow_path_test.py
python .\examples\no_dropout\fly_with_obstacle.py --trajectory square --render
```

### 2. EKF dropout mission

```powershell
python .\examples\dropout_ekf\follow_path_dropout_ekf_mission.py --render
python .\examples\dropout_ekf\follow_path_dropout_ekf_mission.py --no-dropout --no-plot
```

### 3. EKF Monte Carlo validation

```powershell
python .\examples\ekf\ekf_validation.py --n-trials 5
python .\examples\ekf\ekf_validation.py --n-trials 5 --save-dir .\results\ekf_validation
```

### 4. Older example-based LSTM flow

```powershell
python .\examples\lstm\collect_quadcopter_ekf_drift_data.py
python .\examples\lstm\train_quadcopter_ekf_drift_lstm.py
python .\examples\lstm\demo_quadcopter_ekf_lstm_precomp.py --render
```

### 5. RL training

```powershell
python .\examples\rl\train_dropout_rl_policy.py --help
python .\examples\rl\train_dropout_rl_lstm.py --help
```

### 6. General training scripts

```powershell
python .\examples\training\train_mission_trajectory.py --help
python .\examples\training\train_takeoff_hover.py --help
python .\examples\training\train_circle_with_trail.py --help
```

### 7. Utilities

```powershell
python .\examples\utilities\record_simulation_gif.py --help
python .\examples\utilities\generate_trajectories.py --help
python .\examples\utilities\run_simulation_optimization_adam.py --help
```

## Environment setup

From repository root:

```powershell
.\drone_sim_env\Scripts\Activate.ps1
```

If you are building a fresh environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Where to find detailed usage

Each behavior folder has its own README with:

- exact run commands
- argument descriptions
- expected plots
- expected result files
- what a successful run looks like

Read these next:

- [No-Dropout README](./no_dropout/README.md)
- [Dropout EKF README](./dropout_ekf/README.md)
- [EKF README](./ekf/README.md)
- [LSTM README](./lstm/README.md)
- [RL README](./rl/README.md)
- [Training README](./training/README.md)
- [Utilities README](./utilities/README.md)
