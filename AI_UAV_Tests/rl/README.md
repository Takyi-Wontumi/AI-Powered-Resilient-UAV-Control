# AI_UAV_Tests RL

This folder contains RL-specific environment modules used for dropout-aware training and evaluation.

## Modules

- `rl_dropout_policy.py`
  - Main dropout-aware RL environment and policy-facing integration.

## Common import

```python
from AI_UAV_Tests.rl.rl_dropout_policy import DroneDropoutRLEnv
```

## Typical commands

### Train a policy

```powershell
python .\examples\rl\train_dropout_rl_policy.py --help
```

### Evaluate a policy

```powershell
python .\examples\rl\evaluate_dropout_rl_policy.py --help
```

### Run a residual policy

```powershell
python .\examples\rl\run_residual_policy.py
```

## Expected outputs

These workflows typically produce:

- RL checkpoints
- evaluation logs
- terminal summaries of success/tracking/dropout performance

If GUI rendering is enabled in the calling script, expect:

- PyBullet flight visualization
- trajectory-following behavior during dropout windows

## Notes

- The environment here depends on the core control and EKF layers under `AI_UAV_Tests/core` and `AI_UAV_Tests/ekf`.
