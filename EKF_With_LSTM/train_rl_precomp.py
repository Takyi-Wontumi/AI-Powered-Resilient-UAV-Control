#!/usr/bin/env python3
"""
RL Training — PPO residual correction policy over frozen LSTM pre-compensation
===============================================================================
Trains a small MLP policy that outputs [α_x, α_y] ∈ [0.0, 2.0]² — per-axis
confidence multipliers on the LSTM drift prediction each step during GPS dropout.

V11 changes from V10:
  - Action space upper bound widened from 1.5 → 2.0 (fixes ceiling-clamping)
  - Waypoint displacement hard-capped at MAX_WAYPOINT_SHIFT=0.25 m (fixes straying)

The LSTM is fully frozen throughout.  The RL policy learns when to trust each
axis of the LSTM drift prediction more or less, using 3-waypoint lookahead
geometry to identify turns, 180° reversals, straight legs, and arcs.

Architecture:
  Observation (42-dim) → MLP [128, 128] → action [α_x, α_y] ∈ [0.0, 2.0]²

Output: experiments/rl_precomp_policy.zip
        experiments/best_model.zip  (best checkpoint by eval reward)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from uav_rl_env import UAVPrecompEnv

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS  = 1_000_000   # increase to 2M+ for full convergence
N_ENVS           = 1           # set to 2+ if CPU can handle multiple PyBullet instances
CHECKPOINT_FREQ  = 50_000      # save checkpoint every N environment steps
N_EVAL_EPISODES  = 3           # episodes per evaluation pass

SAVE_DIR         = Path('experiments')
LOG_DIR          = Path('experiments/rl_logs')
POLICY_SAVE_PATH = SAVE_DIR / 'rl_precomp_policy'
CHECKPOINT_DIR   = SAVE_DIR / 'rl_checkpoints'

SAVE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

# ── Banner ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RL TRAINING — PPO over frozen LSTM pre-compensation")
print("=" * 70)
print(f"  Total timesteps  : {TOTAL_TIMESTEPS:,}")
print(f"  Parallel envs    : {N_ENVS}")
print(f"  Checkpoint every : {CHECKPOINT_FREQ:,} steps")
print(f"  Policy output    : {POLICY_SAVE_PATH}.zip")
print(f"  Best model       : {SAVE_DIR / 'best_model.zip'}")
print(f"  Tensorboard      : tensorboard --logdir {LOG_DIR}\n")

# ── Environments ──────────────────────────────────────────────────────────────
# Training environment(s) — headless, CPU LSTM inference for speed
train_env = DummyVecEnv([
    lambda: UAVPrecompEnv(render_mode=None, lstm_device='cpu')
    for _ in range(N_ENVS)
])

# Separate evaluation environment — never used for gradient updates
eval_env = DummyVecEnv([
    lambda: UAVPrecompEnv(render_mode=None, lstm_device='cpu')
])

# ── PPO Model ─────────────────────────────────────────────────────────────────
model = PPO(
    policy          = 'MlpPolicy',
    env             = train_env,
    n_steps         = 2048,           # steps per env per rollout
    batch_size      = 64,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.0,            # no entropy bonus (V10: action space [0.0, 1.5])
    learning_rate   = 3e-4,
    verbose         = 1,
    device          = 'cpu',          # MLP policy is faster on CPU than GPU
    tensorboard_log = str(LOG_DIR),
    policy_kwargs   = dict(net_arch=[128, 128]),
)

n_params = sum(p.numel() for p in model.policy.parameters())
print(f"Policy: MLP [128, 128]  ({n_params:,} parameters)")
print(f"Observation: {model.observation_space.shape[0]}-dim")
print(f"Action:      {model.action_space.shape[0]}-dim ([alpha_x, alpha_y] ∈ [0.0, 1.5]²)\n")

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
    save_path   = str(CHECKPOINT_DIR),
    name_prefix = 'rl_precomp',
    verbose     = 1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path = str(SAVE_DIR),
    log_path             = str(LOG_DIR),
    eval_freq            = max(50_000 // N_ENVS, 1),
    n_eval_episodes      = N_EVAL_EPISODES,
    deterministic        = True,
    verbose              = 1,
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Starting training — press Ctrl+C at any time to save and exit.\n")
try:
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, eval_cb],
    )
    print("\n[OK] Training complete.")
except KeyboardInterrupt:
    print("\n[!] Training interrupted — saving current model...")

model.save(str(POLICY_SAVE_PATH))
print(f"[OK] Policy saved  →  {POLICY_SAVE_PATH}.zip")

train_env.close()
eval_env.close()
