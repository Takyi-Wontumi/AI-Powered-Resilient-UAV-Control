# Dropout EKF Examples

This folder contains the dropout-focused mission scripts, including manual dropout,
mission-level dropout handling, and the main EKF-backed dropout mission entry point.

## Scripts

- `follow_path_dropout.py`
  - basic dropout trajectory-following example
- `follow_path_dropout_2.py`
  - alternate dropout baseline
- `follow_path_dropout_mission.py`
  - mission-style dropout example
- `follow_path_dropout_mission_2.py`
  - second mission-style dropout example
- `follow_path_dropout_mission_3.py`
  - most feature-rich mission script with preflight, recording, and keyboard controls
- `follow_path_dropout_mission_noise_log.py`
  - dropout mission that records noise/log data
- `follow_path_dropout_noise.py`
  - dropout mission with noise toggling/logging
- `follow_path_dropout_ekf_mission.py`
  - tuned EKF-backed dropout mission alias inside this dropout-focused folder

## Recommended commands

### Main tuned EKF mission

```powershell
python .\examples\dropout_ekf\follow_path_dropout_ekf_mission.py --render
```

Headless:

```powershell
python .\examples\dropout_ekf\follow_path_dropout_ekf_mission.py --no-plot
```

No scheduled dropout:

```powershell
python .\examples\dropout_ekf\follow_path_dropout_ekf_mission.py --no-dropout --no-plot
```

### Feature-rich mission with preflight

```powershell
python .\examples\dropout_ekf\follow_path_dropout_mission_3.py
```

### Noise/log mission

```powershell
python .\examples\dropout_ekf\follow_path_dropout_mission_noise_log.py
```

## Important arguments

### `follow_path_dropout_ekf_mission.py`

- `--render`
  - opens the PyBullet GUI
- `--no-plot`
  - suppresses the end-of-run comparison plots
- `--no-dropout`
  - disables the scheduled dropout windows so you can run a nominal comparison

Expected outputs:

- terminal consistency metrics
- nominal vs dropout position errors
- nominal vs dropout lateral and altitude RMSE
- optional plots:
  - position reference / estimate / measurement
  - velocity reference / estimate / measurement
  - consistency metrics

### `follow_path_dropout_mission_3.py`

- `--no-preflight`
  - skip preflight preview checks
- `--preflight-only`
  - run only the preflight preview, then exit
- `--preflight-mode`
  - choose preflight simulation style
- `--preflight-speed`
  - speed multiplier for kinematic preview
- `--no-preflight-dashboard`
  - suppress the dashboard while keeping preflight
- `--preflight-hz`
  - preflight sampling frequency
- `--xy-speed-limit`
  - clamp reference XY speed
- `--record-mp4`
  - save an MP4 from PyBullet
- `--record-gif`
  - save a GIF
- `--gif-fps`
  - GIF playback frame rate
- `--gif-frame-skip`
  - capture every N simulation steps
- `--gif-width`, `--gif-height`
  - GIF resolution
- `--gif-camera-distance`, `--gif-camera-yaw`, `--gif-camera-pitch`, `--gif-fov`
  - GIF camera settings

Expected outputs:

- PyBullet GUI mission playback
- optional preflight preview
- optional MP4 or GIF files
- mission keyboard controls:
  - `H` hover
  - `R` return-to-home
  - `L` land
  - `C` clear dropout / resume
  - `N` toggle noise where supported

## What a successful run looks like

- stable takeoff and path following before dropout
- bounded position drift during dropout
- clean recovery after GPS returns
- if using the EKF mission, dropout altitude error should stay much smaller than lateral error
