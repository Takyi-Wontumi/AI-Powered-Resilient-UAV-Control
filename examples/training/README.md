# Training Examples

This folder contains the general non-dropout training scripts that do not belong
specifically to the dropout-EKF, LSTM, or RL buckets.

## Scripts

- `train_circle_with_trail.py`
- `train_drone_hover.py`
- `train_mission_trajectory.py`
- `train_takeoff_hover.py`
- `train_with_multi_cores.py`

## Typical commands

```powershell
python .\examples\training\train_mission_trajectory.py --help
python .\examples\training\train_takeoff_hover.py --help
python .\examples\training\train_circle_with_trail.py --help
python .\examples\training\train_with_multi_cores.py
```

## Expected outputs

- training logs
- checkpoints under `runs/`
- optional GUI playback when `--play` or render flags are used
