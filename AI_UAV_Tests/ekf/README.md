# AI_UAV_Tests EKF

This folder contains the estimator itself, sensor models, diagnostics, and EKF-related
helpers used by the example scripts and the newer LSTM pipeline.

## Modules

- `quadcopter_ekf.py`
  - Main 15-state EKF and adapter code.
- `sensors_ekf.py`
  - IMU, gyro-bias, barometer, and other measurement-noise models.
- `ekf_diagnostics.py`
  - EKF-specific diagnostic helpers.
- `ekf_lstm_precomp.py`
  - EKF/LSTM feature-building and pre-compensation helpers for the older path.
- `ekf_tuner_agent.py`
  - Tuning support for EKF experiments.

## Common imports

```python
from AI_UAV_Tests.ekf.quadcopter_ekf import QuadcopterEKF, PhoenixEKFAdapter
from AI_UAV_Tests.ekf.sensors_ekf import EKFSensorNoise
```

## Recommended validation commands

### Single-mission dropout evaluation

```powershell
python .\examples\ekf\follow_path_dropout_ekf_mission.py --no-plot
```

Expected terminal outputs:

- `Mean consistency metrics`
- `Nominal only`
- `Dropout only`
- `Nominal lateral/altitude errors`
- `Dropout lateral/altitude errors`

### Monte Carlo validation

```powershell
python .\examples\ekf\ekf_validation.py --n-trials 5
```

Expected outputs:

- NEES and NIS summaries
- RMSE values
- optional saved validation figures with `--save-dir`

### Diagnostics

```powershell
python .\examples\ekf\ekf_diagnostics.py --step 4
```

Expected outputs:

- process-noise or covariance checks
- NIS or covariance-growth plots, depending on step

## Typical plots and results

Workflows using this folder generate:

- `Reference / EKF / truth` position plots
- velocity plots
- NEES and NIS consistency plots
- Monte Carlo RMSE and coverage figures

## Notes

- Import from the organized `AI_UAV_Tests.ekf` package paths directly.
- This folder is the estimator backbone used by the tuned 15-state EKF mission and validation scripts.
