# RL Policy for GPS Dropout Handling

## Quick Start

### 1. Train the Policy (Approach C: RL handles dropout only)

```bash
python examples/train_dropout_rl_policy.py \
    --steps 100000 \
    --steps-per-epoch 2048 \
    --n-envs 1 \
    --save-dir models/dropout_rl/ \
    --learning-rate 3e-4
```

This trains a PPO policy with:
- **State**: EKF estimates (pos, vel, att, rate) + covariance diagonals (uncertainty) + dropout flag
- **Action**: [thrust, roll, pitch, yaw_rate] (normalized to [-1, 1])
- **Training**: Domain randomization of dropout timing/duration (5-15s start, 1-8s duration)
- **Reward**: -position_error - 0.01*cov_trace + recovery_bonus
- **Duration**: ~30 minutes for 100k steps on 1 GPU

### 2. Evaluate Against Baseline

```bash
python examples/evaluate_dropout_rl_policy.py \
    --model models/dropout_rl/best_model.zip \
    --n-trials 10 \
    --save-dir results/
```

Compares:
- **Baseline**: PID + frozen GPS position (current approach)
- **RL Policy**: Learned controller with covariance awareness

Outputs:
- Position RMSE during dropout
- Crash rates
- Comparison plots → `results/dropout_rl_comparison.png`

---

## Architecture Details

### State Space (28 dimensions)
```
[pos_x, pos_y, pos_z,                              # 3
 vel_x, vel_y, vel_z,                              # 3
 roll, pitch, yaw,                                 # 3
 p_rate, q_rate, r_rate,                           # 3
 σ_pos_x, σ_pos_y, σ_pos_z,                        # 3 (uncertainty)
 σ_vel_x, σ_vel_y, σ_vel_z,                        # 3
 σ_roll, σ_pitch, σ_yaw,                           # 3
 σ_p, σ_q, σ_r,                                    # 3
 dropout_active,                                    # 1
 ref_pos_x, ref_pos_y, ref_pos_z,                  # 3
 ref_vel_x, ref_vel_y, ref_vel_z]                  # 3
```

### Action Space (4 dimensions)
```
[thrust_normalized,      # [-1, 1] → thrust ∈ [0.5mg, 1.3mg]
 roll_normalized,        # [-1, 1] → roll ∈ [-20°, 20°]
 pitch_normalized,       # [-1, 1] → pitch ∈ [-20°, 20°]
 yaw_rate_normalized]    # [-1, 1] → yaw_rate ∈ [-500°/s, 500°/s]
```

### Reward Function
```python
reward = (
    -10.0 * pos_error                           # minimize distance from reference
    - 0.01 * trace(P)                           # penalize high uncertainty
    - 0.001 * ||rates_des||                     # smooth control
    + 5.0 * (just_recovered_from_dropout)       # recovery bonus
)
```
Note: Reward is only non-zero during dropout phase. During normal flight, agent is idle.

---

## Advanced: Using RNN/LSTM Policies (TODO)

The current implementation uses feedforward MLPPolicy. For better dropout handling, consider:

### Why RNNs?
- **Temporal Memory**: LSTM can remember trajectory from 500ms ago
- **Dead Reckoning**: Policy can use momentum/heading to estimate motion during dropout
- **Uncertainty Tracking**: Memory of covariance growth helps predict divergence

### Implementation
```python
model = PPO(
    "MlpLstmPolicy",  # or "CnnLstmPolicy" with image input
    env,
    learning_rate=3e-4,
    policy_kwargs=dict(
        net_arch=[256, 256],
        lstm_hidden_size=256,
        n_lstm_layers=2,
    ),
    ...
)
```

### Expected Improvements
- Better handling of long dropouts (>5s)
- Reduced position drift by leveraging trajectory history
- Potential for smooth trajectory continuation under dropout

---

## Domain Randomization Details

Each episode randomizes:
- **Dropout start time**: Uniform(5s, 15s)
- **Dropout duration**: Uniform(1s, 8s)
- **Reference trajectory**: Same mission, but variations in execution timing

This forces the policy to learn robust control rather than memorizing a specific dropout scenario.

---

## Expected Results (Approach C Baseline)

From N=30 validation:
- **Baseline (PID + frozen)**: 0.43 cm RMSE during dropout, 0% crash rate
- **RL Expected**: 0.3-0.4 cm RMSE, 0% crash rate, smoother recovery

The RL agent will likely learn to:
1. Reduce control effort during high uncertainty
2. Increase altitude slightly to create safety margin
3. Maintain heading more smoothly
4. Recover faster upon GPS re-acquisition

---

## Next Steps: Approach A & B

**Approach A**: Replace PID entirely with RL
- Pros: End-to-end learning, potentially better normal-flight control
- Cons: Longer training, harder to debug, less interpretable

**Approach B**: RL augments PID
- Pros: RL learns corrections on top of stable baseline
- Cons: More complex reward design

Start with **Approach C** to validate the approach, then iterate.

---

## Troubleshooting

### Training is slow
- Increase `--n-envs` to 4-8 for parallel training (more GPU memory needed)
- Reduce `--steps` for quick validation run (e.g., 10000)

### Model not converging
- Check reward function scale (-10.0 * error might be too large)
- Increase learning rate (`--learning-rate 1e-3`)
- Visualize training in TensorBoard: `tensorboard --logdir models/dropout_rl/logs`

### Evaluation fails
- Ensure `--model` path is correct (ends in `.zip`)
- Check that model was saved (look for `best_model.zip` in save directory)

---

## Code Files
- `AI_UAV_Tests/rl_dropout_policy.py` - Gym environment wrapper
- `examples/train_dropout_rl_policy.py` - PPO training script
- `examples/evaluate_dropout_rl_policy.py` - Baseline comparison + metrics
