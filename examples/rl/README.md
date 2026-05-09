# RL Examples

This folder contains the main RL entry points related to dropout-aware control and residual policies.

## Scripts

- `train_dropout_rl_policy.py`
- `train_dropout_rl_lstm.py`
- `evaluate_dropout_rl_policy.py`
- `run_residual_policy.py`

## Recommended commands

### Baseline residual PPO training

```powershell
python .\examples\rl\train_dropout_rl_policy.py --task hover --steps 200000 --steps-per-epoch 4096 --n-envs 4 --save-dir .\models\ekf_rl_hover\
```

### LSTM-augmented RL training

```powershell
python .\examples\rl\train_dropout_rl_lstm.py --steps 300000 --n-envs 4 --save-dir .\models\dropout_rl_lstm\
```

### Policy evaluation

```powershell
python .\examples\rl\evaluate_dropout_rl_policy.py --render
```

### Residual-policy run

```powershell
python .\examples\rl\run_residual_policy.py
```

## Important arguments

### `train_dropout_rl_policy.py`

- `--task`
  - RL task or scenario, such as `hover`
- `--steps`
  - total training steps
- `--steps-per-epoch`
  - rollout/update frequency
- `--n-envs`
  - number of vectorized environments
- `--eval-freq`
  - evaluation interval
- `--learning-rate`
  - optimizer learning rate
- `--residual-alpha`
  - strength of residual correction
- `--target-kl`
  - PPO KL target
- `--ent-coef`
  - entropy coefficient
- `--save-dir`
  - output model directory
- `--no-render`
  - suppress GUI during training

Expected output:

- PPO progress in terminal
- saved checkpoints under `--save-dir`

### `train_dropout_rl_lstm.py`

- `--steps`
  - total RL training steps
- `--steps-per-epoch`
  - PPO rollout/update cadence
- `--n-envs`
  - number of vectorized environments
- `--lstm-hidden-size`
  - recurrent hidden dimension
- `--lstm-layers`
  - number of recurrent layers
- `--save-dir`
  - output model directory
- `--eval-freq`
  - evaluation interval
- `--learning-rate`
  - optimizer learning rate
- `--residual-alpha`
  - strength of residual correction
- `--target-kl`
  - PPO KL target
- `--ent-coef`
  - entropy coefficient
- `--no-render`
  - suppress GUI during training

### `evaluate_dropout_rl_policy.py`

Run `--help` to inspect the available checkpoint, rendering, and evaluation flags.

Expected output:

- evaluation summary
- success/failure and tracking behavior metrics
- optional GUI run

## Notes

- These scripts depend on `AI_UAV_Tests.rl.rl_dropout_policy`.
- If you are using the newer precomputed LSTM path, that still lives under `EKF_With_LSTM/`.
