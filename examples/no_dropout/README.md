# No-Dropout Examples

This folder contains trajectory-following and controller demos that do not depend on
the GPS-dropout / EKF-dropout workflow. Use these scripts to verify nominal flight
behavior, obstacle avoidance setups, and low-level controller behavior.

## Scripts

- `figure_eight_example.py`
  - nominal figure-eight reference example
- `fly_with_obstacle.py`
  - mission with a spawned obstacle and configurable trajectory
- `follow_path_test.py`
  - baseline follow-path test
- `follow_path_test_2.py`
  - alternate follow-path test
- `hover_with_attitude_PD.py`
  - hover demo using attitude PD
- `hover_with_attitude_PID.py`
  - hover demo using attitude PID
- `takeoff_with_attitude_rate_PID.py`
  - takeoff demo using attitude-rate PID

## Recommended commands

### Simple baseline run

```powershell
python .\examples\no_dropout\follow_path_test.py
```

### Obstacle mission

```powershell
python .\examples\no_dropout\fly_with_obstacle.py --trajectory square --render
```

### Hover / takeoff controller demos

```powershell
python .\examples\no_dropout\hover_with_attitude_PD.py
python .\examples\no_dropout\hover_with_attitude_PID.py
python .\examples\no_dropout\takeoff_with_attitude_rate_PID.py
```

## Important arguments

### `fly_with_obstacle.py`

- `--duration`
  - total simulation time in seconds
- `--speedup`
  - runs the mission faster or slower than real time
- `--trajectory`
  - path shape, for example `circle`, `square`, or `point`
- `--radius`
  - circle radius or reference offset
- `--side`
  - square side length
- `--period`
  - path period in seconds
- `--flight-z`
  - main reference altitude
- `--takeoff-z`
  - initial takeoff target altitude
- `--takeoff-trigger-z`
  - altitude threshold for enabling path following
- `--xy-speed-limit`
  - clamps reference XY speed
- `--obstacle-x`, `--obstacle-y`, `--obstacle-z`
  - obstacle position
- `--obstacle-size-x`, `--obstacle-size-y`, `--obstacle-size-z`
  - obstacle box dimensions
- `--obstacle-color`
  - obstacle RGBA color
- `--trail`
  - draw the drone trail
- `--debug-every`
  - print status every N steps
- `--render`
  - open or suppress the PyBullet GUI

## Expected outputs

Depending on the script, expect:

- PyBullet GUI motion
- printed debug status in terminal
- reference tracking behavior without scheduled GPS dropout

These scripts usually do not generate formal EKF validation figures. They are mainly
for checking nominal stability and controller/path behavior.
