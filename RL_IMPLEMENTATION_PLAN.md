# RL Implementation Plan: GPS Dropout Handling

## ✓ Completed (Approach C: RL handles dropout only)

### Core Implementation
- [x] `AI_UAV_Tests/rl_dropout_policy.py` — Gym environment with EKF state + covariance as input
- [x] `examples/train_dropout_rl_policy.py` — PPO training with domain randomization
- [x] `examples/evaluate_dropout_rl_policy.py` — Baseline comparison + metrics
- [x] `examples/train_dropout_rl_lstm.py` — LSTM variant for dead reckoning
- [x] Documentation:
  - `RL_DROPOUT_QUICKSTART.md` — Quick start guide
  - `LSTM_DEAD_RECKONING.md` — Deep dive on temporal memory + dead reckoning

### Technology Stack
- [x] stable-baselines3 (PPO implementation)
- [x] gymnasium (gym environment)
- [x] Integration with existing EKF + PID controller

---

## → Next: Quick Validation Run (Approach C)

### Step 1: Train Feedforward Policy (5-10 minutes)
```bash
cd ~/Downloads/AI-Powered-Resilient-UAV-Control

# Quick test run (10k steps, should show convergence trend)
python examples/train_dropout_rl_policy.py \
    --steps 10000 \
    --save-dir models/dropout_rl_test/

# Full run (100k steps, ~30 minutes)
python examples/train_dropout_rl_policy.py \
    --steps 100000 \
    --save-dir models/dropout_rl_full/
```

**What to expect:**
- Episode reward should increase over time (moving average)
- Mean episode return visible in terminal output
- Model checkpoints saved every 10k steps
- TensorBoard logs in `models/dropout_rl_full/logs/`

### Step 2: Evaluate Against Baseline
```bash
python examples/evaluate_dropout_rl_policy.py \
    --model models/dropout_rl_full/best_model.zip \
    --n-trials 10 \
    --save-dir results/dropout_rl_eval/
```

**Output:**
- Position RMSE comparison (RL vs Baseline)
- Crash rates
- Comparison plot: `results/dropout_rl_eval/dropout_rl_comparison.png`

**Expected results:**
- RL RMSE: 0.35-0.45 cm (comparable to baseline 0.43 cm)
- Crash rate: 0% (both should be safe)
- Policy will likely learn to:
  - Reduce control effort during high uncertainty
  - Maintain altitude more smoothly
  - Recover faster upon GPS re-acquisition

---

## Later: LSTM Policy (Better Dead Reckoning)

### Step 3: Train LSTM Policy (1-2 hours)
```bash
python examples/train_dropout_rl_lstm.py \
    --steps 150000 \
    --lstm-layers 2 \
    --lstm-hidden-size 256 \
    --save-dir models/dropout_rl_lstm/
```

**Expected improvements:**
- RMSE during dropout: 0.2-0.3 cm (better dead reckoning)
- Smoother recovery when GPS returns
- Better generalization to variable dropout durations

---

## Future Work: Approach A & B

### Approach A: End-to-End RL (Replace PID entirely)
```
State:     [EKF state + covariance + reference trajectory]
Action:    [Motor speeds] (4-dim)
Agent:     RL learns entire control loop (not just dropout handling)
Training:  200k+ steps (much longer, more complex)
Reward:    -tracking_error - control_effort + stability_bonus
```

**Pros:**
- Single unified policy
- Can optimize normal flight behavior too
- Fully learned recovery from dropout

**Cons:**
- Longer training
- Harder to debug failures
- Need better reward shaping
- May not generalize to new trajectories

### Approach B: RL Augments PID
```
State:     [PID error + EKF covariance + dropout flag]
Action:    [thrust_correction, attitude_correction] (adjustments to PID)
Agent:     RL learns to improve upon PID baseline
Training:  50k steps (faster, PID does heavy lifting)
Reward:    -correction_magnitude - error_residual
```

**Pros:**
- Faster training (PID baseline is stable)
- Interpretable: Can see what RL is correcting
- Safety: PID fallback if RL fails

**Cons:**
- Limited flexibility (can only correct PID)
- May not discover fundamentally different strategies

---

## Success Criteria

### Approach C (Current)
- [ ] Training runs without errors
- [ ] Episode return increases monotonically
- [ ] RL policy achieves comparable RMSE to baseline (<0.5 cm)
- [ ] 0% crash rate during evaluation
- [ ] Policy handles random dropout scenarios (domain randomization working)

### Approach A/B (Later)
- [ ] Improvement over baseline (lower RMSE or faster recovery)
- [ ] Smooth trajectories (no jerky control)
- [ ] Generalization to unseen trajectories
- [ ] Stable training (no divergence)

---

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir models/dropout_rl_full/logs/
# Visit http://localhost:6006
```

Watch for:
- `rollout/ep_rew_mean` — should increase over time
- `train/policy_loss` — should decrease
- `train/value_loss` — should stabilize

### Training Logs
Terminal output shows:
- Episode returns (should trend upward)
- Policy gradient magnitude
- Value function loss

Healthy training:
```
| rollout/ | ep_rew_mean | -1234.5  |
| rollout/ | ep_len_mean | 1200     |
| train/   | policy_loss | -0.15    |
| train/   | value_loss  | 45.3     |
```

---

## File Structure After Training

```
models/
├── dropout_rl_full/
│   ├── best_model.zip              # Best policy (use this for evaluation)
│   ├── rl_dropout_policy_final.zip  # Final policy
│   ├── rl_dropout_policy_10000_steps.zip
│   ├── rl_dropout_policy_20000_steps.zip
│   └── logs/
│       └── dropout_rl_training_1/   # TensorBoard events

results/
└── dropout_rl_eval/
    ├── dropout_rl_comparison.png    # Comparison plot
    └── metrics.txt                  # Summary statistics
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Training is very slow | Computation overhead | Reduce steps, disable visualization |
| Reward not increasing | Reward scale too large | Adjust `-10.0 * pos_error` term |
| Model crashes | Bad initialization | Train longer, increase n_epochs to 20 |
| Evaluation fails | Model path wrong | Use `best_model.zip` not final |
| OOM (out of memory) | Batch size too large | Reduce `--steps-per-epoch` |

---

## Timeline Estimate

| Phase | Duration | Status |
|-------|----------|--------|
| Approach C (Feedforward) | 30-45 min | Ready to start |
| Validation | 15-20 min | Ready to start |
| Approach C LSTM | 1-2 hours | Ready to start |
| Approach A (E2E) | 2-3 hours | Design ready |
| Approach B (Augment) | 1-2 hours | Design ready |

**Total to all three approaches:** 6-10 hours of training (spread across runs)

---

## What You'll Learn

1. **RL for control** — Training agents on continuous control tasks
2. **Domain randomization** — Robust policies via simulation variation
3. **Covariance-aware RL** — Using uncertainty as input signal
4. **LSTM for temporal** — Recurrent policies for dead reckoning
5. **Safety in RL** — Keeping crash rate at 0% through reward design

This is a complete pipeline for learning dropout-resilient controllers!
