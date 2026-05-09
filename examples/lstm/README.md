# LSTM Examples

This folder contains the older EKF+LSTM example flow that lives under `examples/`.
Use it when you want the older example-based data collection, training, demo, and evaluation path.

If you are using the newer large-data precomputed pipeline, use the scripts under `EKF_With_LSTM/`.

## Scripts

- `collect_quadcopter_ekf_drift_data.py`
- `train_quadcopter_ekf_drift_lstm.py`
- `demo_quadcopter_ekf_lstm_precomp.py`
- `evaluate_quadcopter_ekf_lstm_precomp.py`

## Recommended commands

### Collect data

```powershell
python .\examples\lstm\collect_quadcopter_ekf_drift_data.py --missions 40 --steps-per-mission 3500
```

### Train

```powershell
python .\examples\lstm\train_quadcopter_ekf_drift_lstm.py --epochs 12 --batch-size 256
```

### GUI demo

```powershell
python .\examples\lstm\demo_quadcopter_ekf_lstm_precomp.py --render
```

### Evaluation

```powershell
python .\examples\lstm\evaluate_quadcopter_ekf_lstm_precomp.py --n-trials 5 --save-dir .\results\lstm_example_eval
```

## Important arguments

### `collect_quadcopter_ekf_drift_data.py`

- `--missions`
  - number of missions to collect
- `--steps-per-mission`
  - maximum simulation steps per mission
- `--save-dir`
  - where the collected dataset is written
- `--min-steps`
  - minimum accepted mission length
- `--seed`
  - repeatable random seed

Expected output:

- dataset files written under the chosen data directory
- terminal progress for each mission

### `train_quadcopter_ekf_drift_lstm.py`

- `--data-dir`
  - directory containing training data
- `--save-path`
  - output weights file
- `--seq-len`
  - LSTM input sequence length
- `--stride`
  - dataset window stride
- `--epochs`
  - number of epochs
- `--batch-size`
  - mini-batch size
- `--learning-rate`
  - optimizer learning rate
- `--hidden-size`
  - LSTM hidden dimension
- `--early-stop-patience`
  - epochs without improvement before early stopping

Expected output:

- terminal training/validation loss
- saved model weights

### `demo_quadcopter_ekf_lstm_precomp.py`

- `--render`
  - open GUI
- `--no-plot`
  - suppress end-of-run figures
- `--model-path`
  - choose which trained model to load
- `--dropout-duration`
  - dropout window length
- `--dropout-margin`
  - buffer away from mission edges
- `--playback-slowdown`
  - slow down or speed up GUI playback

Expected output:

- PyBullet flight visualization
- optional EKF vs LSTM comparison plots

### `evaluate_quadcopter_ekf_lstm_precomp.py`

- `--n-trials`
  - number of evaluation runs
- `--model-path`
  - weights to evaluate
- `--dropout-duration`
  - dropout length
- `--dropout-margin`
  - distance from mission edges
- `--save-dir`
  - output directory for evaluation plots/results
- `--render`
  - enable GUI
- `--playback-slowdown`
  - GUI speed factor
- `--target-error-cm`
  - target error threshold
- `--state-gain-xy`, `--state-gain-z`
  - state correction gains
- `--state-max-xy`, `--state-max-z`
  - correction caps
- `--state-lead-time`
  - compensation lead time
- `--state-warmup`
  - warmup before compensation
- `--use-position-prior`
  - enable position prior if supported

Expected output:

- EKF vs LSTM evaluation summary
- saved plots in `--save-dir`
