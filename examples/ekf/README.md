# EKF Examples

This folder contains the main estimator-validation and estimator-demo entry points.
Use these scripts when you want to validate the tuned 15-state EKF, inspect dropout
behavior, or generate presentation-ready plots and consistency metrics.

## Scripts

- `follow_path_dropout_ekf_mission.py`
  - Runs the square mission with scheduled dropout windows.
  - Prints nominal vs dropout error metrics.
  - Produces plots for `X/Y/Z`, velocity, and consistency when plotting is enabled.
- `ekf_validation.py`
  - Monte Carlo validation suite.
  - Produces NEES, NIS, RMSE, coverage, and dropout-resilience figures.
- `ekf_diagnostics.py`
  - Single-run and sweep-style diagnostics for process noise and covariance health.
- `imu_noise_analysis.py`
  - Inspects IMU/sensor noise behavior and consistency.
- `analyze_hover_motor_forces.py`
  - Hover/motor-force sanity check for thrust consistency.

## Prerequisites

- Activate the project environment first.
- Run commands from the repository root.
- PyBullet is used by the mission and validation scripts.

## Recommended commands

### 1. EKF mission run with dropout

```powershell
python .\examples\ekf\follow_path_dropout_ekf_mission.py --render
```

Headless:

```powershell
python .\examples\ekf\follow_path_dropout_ekf_mission.py --no-plot
```

No dropout:

```powershell
python .\examples\ekf\follow_path_dropout_ekf_mission.py --no-dropout --no-plot
```

### 2. Monte Carlo EKF validation

```powershell
python .\examples\ekf\ekf_validation.py --n-trials 5
```

Save validation figures:

```powershell
python .\examples\ekf\ekf_validation.py --n-trials 5 --save-dir .\results\ekf_validation
```

### 3. EKF diagnostics

```powershell
python .\examples\ekf\ekf_diagnostics.py
```

Specific step:

```powershell
python .\examples\ekf\ekf_diagnostics.py --step 4 --save-dir .\results\ekf_diag
```

## Important arguments

### `follow_path_dropout_ekf_mission.py`

- `--render`
  - opens the PyBullet GUI
- `--no-plot`
  - suppresses the end-of-run comparison plots
- `--no-dropout`
  - disables the scheduled dropout windows so you can compare against nominal flight

### `ekf_validation.py`

- `--n-trials`
  - number of Monte Carlo trials
- `--no-dropout`
  - run nominal validation with no GPS dropout
- `--trajectory`
  - validation mission type, such as `circle` or `hover`
- `--dropout-start`
  - dropout start time in seconds
- `--dropout-duration`
  - dropout duration in seconds
- `--save-dir`
  - where PNG figures are saved
- `--show`
  - display figures interactively after generation

### `ekf_diagnostics.py`

- `--step`
  - run only one diagnostic stage
- `--save-dir`
  - save generated figures and reports
- `--show`
  - display figures interactively
- `--scales`
  - custom Q-scale values for sweep-style diagnostics

### `imu_noise_analysis.py`

- `--samples`
  - number of stationary IMU samples
- `--dt`
  - sampling period
- `--save-dir`
  - output directory for plots and report files
- `--show`
  - display figures interactively

### `analyze_hover_motor_forces.py`

- `--target-z`
  - target hover altitude
- `--sim-time`
  - simulation duration
- `--dt`
  - controller timestep

## Expected terminal output

### `follow_path_dropout_ekf_mission.py`

Look for:

- `Mean consistency metrics`
- `Nominal only`
- `Dropout only`
- `Nominal position errors`
- `Dropout position errors`
- `Nominal lateral/altitude errors`
- `Dropout lateral/altitude errors`

Typical useful numbers:

- no-dropout lateral drift in cm
- dropout lateral drift in cm
- no-dropout altitude drift in cm
- dropout altitude drift in cm

### `ekf_validation.py`

Look for:

- `Mean NEES`
- `Mean NIS / DOF`
- `EKF position RMSE`
- `3sigma coverage`

## Expected figures and results

### `follow_path_dropout_ekf_mission.py`

When plotting is enabled, expect:

- `Reference, EKF Estimate, and Noisy Measurement`
- `Velocity Reference, Truth, EKF Estimate, and Noisy Measurement`
- `EKF Consistency Metrics`

These are useful for:

- showing lateral vs altitude dropout behavior
- showing Z oscillation vs XY drift
- comparing nominal and dropout consistency visually

### `ekf_validation.py`

When `--save-dir` is provided, expect PNG outputs for:

- Jacobian cross-check
- NEES
- NIS
- RMSE comparison
- covariance coverage
- dropout resilience

## What a successful run looks like

### Mission script

- no-dropout drift near zero
- dropout drift bounded and recoverable
- dropout altitude error staying much smaller than lateral error

### Validation suite

- NEES near the expected dimension
- NIS / DOF near 1 on the navigation-side channels
- EKF RMSE clearly better than raw or unguided behavior

## Imports

These scripts rely on:

```python
from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.ekf.quadcopter_ekf import QuadcopterEKF, PhoenixEKFAdapter
from AI_UAV_Tests.ekf.sensors_ekf import EKFSensorNoise
from AI_UAV_Tests.core.trajectories_library import FlightMission
```
