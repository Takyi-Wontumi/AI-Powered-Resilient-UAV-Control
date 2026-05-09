# LSTM-Based Dead Reckoning for GPS Dropout Missions

## The Problem with Feedforward Networks

**Feedforward (MLP) Policy:**
```
Input: [pos, vel, att, rate, σ_pos, σ_vel, ..., dropout_flag] → Hidden → Output action
```
- **Stateless**: Each decision is independent; no memory of past trajectory
- **Vulnerable to drift**: If EKF drifts 1cm during dropout, policy sees isolated state and might over-correct
- **Loss of intent**: After re-acquisition, policy doesn't remember what heading/altitude it was maintaining

**Result**: Policy must learn everything from scratch each dropout episode, leading to:
- Slower training convergence
- Suboptimal behavior during long dropouts
- Abrupt transitions when GPS re-appears

---

## The LSTM Solution: Temporal Memory

**LSTM (Recurrent) Policy:**
```
Input: [pos, vel, att, rate, σ, dropout_flag] → LSTM(hidden_state) → Hidden → Output action
       ↑ Current state at t
                        ↑ Memory of states at t-1, t-2, ...
```

### How LSTM Enables Dead Reckoning

**Phase 1: Before Dropout (GPS Available)**
```
LSTM learns (remembers) in time steps t-1, t-2, t-3:
  - Velocity vector direction: "We're moving North-East"
  - Altitude trend: "Climbing toward 1.5m"
  - Covariance trend: P is small and stable
  
Hidden state h_t contains this trajectory context.
```

**Phase 2: During Dropout (GPS Denied)**
```
Input: [pos_drifting, vel_estimated, att_from_imu, σ_large, dropout=true]
       ↓
LSTM hidden state h_t contains:
  - "We were moving North-East" (from t-1, t-2)
  - "Altitude was stable at 1.5m"
  - "Covariance was growing during dropout"
       ↓
Output: Control command that:
  - Maintains North-East heading (dead reckoning)
  - Keeps altitude near 1.5m (inertial memory)
  - Compensates for observed drift
```

**Phase 3: GPS Re-acquisition (Recovery)**
```
Input: [pos_recovered, vel_corrected, att_updated, σ_small, dropout=false]
       ↓
LSTM h_t remembers the mission trajectory before dropout
       ↓
Smooth transition: "Resume the North-East flight at 1.5m"
     vs.  Abrupt jerk: "Where am I? Let me recalculate everything"
```

---

## Mathematical Insight

**Feedforward (at time t during dropout):**
```
a_t = f(x_t)

where x_t = [pos_t, vel_t, att_t, rate_t, σ_t, dropout_t]
```
Loss of history: x_t doesn't contain information about x_{t-1}, x_{t-2}, ...

**LSTM (at time t):**
```
h_t = LSTM(x_t, h_{t-1})       # h_t = f(x_t, h_{t-1}, W_lstm)
a_t = policy(h_t)              # a_t = g(h_t)

where h_t summarizes x_0, x_1, ..., x_t
```
Keeps trajectory history: h_t encodes the past, enabling:
- Velocity trend estimation: d(v)/dt ≈ (v_t - v_{t-k})/k
- Altitude trend: "Are we climbing or descending?"
- Covariance growth rate: dP/dt tells us "GPS has been out this long"

---

## Expected Improvements (LSTM vs Feedforward)

### Scenario: 5-second GPS dropout

**Feedforward MLP:**
- Initial position error: 0 (starts with EKF estimate)
- Error growth: ~quadratic with time (drifts faster as policy struggles)
- Peak error during dropout: 3-5 cm
- Recovery time after re-acquisition: 0.5-1.0s (jarring)
- Trajectory smoothness: Lower

**LSTM:**
- Initial position error: 0 (starts with EKF estimate)
- Error growth: ~linear (policy uses velocity trend to predict motion)
- Peak error during dropout: 1-2 cm (better dead reckoning)
- Recovery time after re-acquisition: 0.1-0.2s (smooth)
- Trajectory smoothness: Higher (uses momentum memory)

### Quantitative Prediction
With a 256-dim hidden state on a 2-layer LSTM:
- **Information capacity**: Can encode ~500ms of trajectory history
- **Dead reckoning window**: Up to ~4s dropout duration
- **Position drift during 4s dropout**: 0.5-1.5 cm (vs 3-5 cm for MLP)

---

## When LSTM Fails: Physical Limits

Even LSTM has hard limits, as you correctly identified:

### 1. Drift Accumulation (Long Dropouts)
```
Position error = ∫∫ (bias_accel) dt²

After 10 seconds without GPS:
  - IMU bias drift: ~1 cm/s² residual
  - Accumulated error: 0.5 * 1e-2 * (10)² ≈ 0.5 m

LSTM can slow this but not eliminate it. Eventually:
  error > mission_tolerance
```

**Solution**: Combine LSTM with other sensors
- Visual odometry (monocular/stereo camera)
- LiDAR scan matching
- Inertial Measurement Unit (IMU) bias estimation

### 2. External Disturbances (Wind, Gusts)
```
Wind gust during dropout:
  - Wind force: F = 0.5 * rho * v² * C_d * A
  - Unobservable by EKF (no GPS to measure error)
  - LSTM can't learn what it can't see
  
Result: Drift in wind direction regardless of policy
```

**Solution**: Anemometer or wind estimation from motor commands
- Motor speed changes reveal wind compensation needed
- LSTM can learn: "When motors are at [ω1, ω2, ω3, ω4], I'm in [wind_x, wind_y] m/s"

### 3. Gimbal Lock / Extreme Attitudes
```
If drone tilts >45° during dropout and power is cut:
  - IMU can't measure gravity vector
  - Attitude estimate diverges
  - LSTM can't recover without external reference
```

**Solution**: Constrain controller to avoid extreme attitudes
- Add attitude penalty to reward: -10 * (|roll| + |pitch|)
- LSTMlead learns: "Stay level to maintain dead reckoning accuracy"

---

## Implementation: LSTM vs MLP

### Feedforward (Current)
```python
model = PPO("MlpPolicy", env, ...)
```
Training: 100k steps, ~30 minutes
Memory: ~100 MB
Performance: 0.43 cm RMSE during dropout

### LSTM (Recommended)
```python
policy_kwargs = {
    "net_arch": [256, 256],
    "lstm_hidden_size": 256,
    "n_lstm_layers": 2,
}
model = PPO("MlpLstmPolicy", env, policy_kwargs=policy_kwargs, ...)
```
Training: 100k steps, ~1-2 hours (slower due to recurrence)
Memory: ~150 MB
Performance: 0.2-0.3 cm RMSE during dropout (estimated)

---

## Training with LSTM

```bash
python examples/train_dropout_rl_lstm.py \
    --steps 150000 \
    --lstm-layers 2 \
    --lstm-hidden-size 256 \
    --save-dir models/dropout_rl_lstm/
```

Key differences:
- Takes longer to train (recurrent computation)
- More stable convergence (temporal structure helps)
- Better generalization to unseen dropout durations

---

## Evaluation Strategy

Compare three policies:
1. **Baseline**: PID + frozen position (current solution)
2. **RL MLP**: Feedforward RL policy (simple, fast)
3. **RL LSTM**: Recurrent RL policy (complex, better dead reckoning)

Test on:
- Standard dropout (5-9s at t=5s)
- Long dropout (2-4s at random times)
- Very long dropout (10s, beyond LSTM horizon)
- Wind disturbance (if available)

---

## Summary: Your "Pilot in Clouds" Analogy

You're exactly right:

| Factor | Pilot in Clouds | RL LSTM in GPS Dropout |
|--------|-----------------|------------------------|
| **Instruments** | Altimeter, airspeed, compass | EKF state + covariance |
| **Memory** | Remembers heading before entering clouds | LSTM h_t encodes trajectory |
| **Dead reckoning** | Maintains heading + altitude from memory | Uses velocity trends from history |
| **Recovery** | Smooth transition when exiting clouds | No abrupt jerks when GPS returns |
| **Limits** | Long clouds = fuel concerns, wind shear | Long dropouts = drift accumulation, wind |

The LSTM policy becomes exactly this: an agent that flies on instruments, trusting the EKF and its own trajectory memory until GPS returns.
