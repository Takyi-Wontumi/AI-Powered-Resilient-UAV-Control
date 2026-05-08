#!/usr/bin/env python3
"""
Train Position-Drift LSTM
=========================
Trains a multi-horizon LSTM to predict EKF dead-reckoning position error.

  Input  (300 steps × 6): [vx, vy, vz, wx, wy, wz]
                           velocity + gyro (both GPS-independent)
                           300 steps = 1.5 s context window

  Output (5 horizons × 3): position drift = dead_reckoning_pos - true_pos
                            at t+50, t+100, t+150, t+200, t+250 steps ahead
                            = +0.25 s, +0.5 s, +0.75 s, +1.0 s, +1.25 s

The LSTM learns the *pattern* of drift accumulation from velocity and
rotation-rate history, allowing the EKF to apply a learned correction
on top of dead-reckoning during GPS dropout.

Architecture: identical to v3_fast so it is a drop-in replacement.
  - Input projection:  6 → 256
  - LSTM:              512 hidden, 2 layers, inter-layer dropout=0.1
  - 5 output heads:    256 → 128 → ReLU → Dropout → 3

Weights saved to: experiments/lstm_position_drift.pt
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import json
import pickle

print("\n" + "="*100)
print("TRAIN POSITION-DRIFT LSTM")
print("="*100 + "\n")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# =========================================================
# Model  (same architecture as v3_fast — drop-in replacement)
# =========================================================

class PositionDriftLSTM(nn.Module):
    """
    Multi-horizon LSTM for predicting GPS dead-reckoning position drift.

    Input:  [vx, vy, vz, wx, wy, wz, t_norm]  (7 features)
            velocity + gyro + time-since-dropout (normalised 0->1)
    Output: position_drift[h] = DR_pos - true_pos  at horizon h=1..7
            horizons at +0.5s, +1.0s, +1.5s, +2.0s, +2.5s, +3.0s, +3.5s
    """
    def __init__(self, horizon_steps=7, hidden_size=256):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = 1
        self.horizon_steps = horizon_steps
        self.proj = nn.Linear(7, hidden_size)   # 7 inputs: vel(3) + gyro(3) + t_norm(1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1,
                            batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 3)
            )
            for _ in range(horizon_steps)
        ])

    def forward(self, x):
        x = self.proj(x)
        lstm_out, _ = self.lstm(x)
        final = lstm_out[:, -1, :]
        return torch.stack([head(final) for head in self.heads], dim=1)


# =========================================================
# Dataset
# =========================================================

class PositionDriftDataset(Dataset):
    """
    Each sample is:
      input   : (seq_len, 7) -- window of [vel, gyro, t_norm]
      targets : (7, 3)       -- position drift at each of 7 future horizons

    Horizons: +0.5s, +1.0s, +1.5s, +2.0s, +2.5s, +3.0s, +3.5s
    (100 steps x 0.005s = 0.5s per stride)
    """

    # Must match SHADOW_RESET_INTERVAL in collect_pybullet_position_drift_data_precomp.py
    SHADOW_RESET_INTERVAL = 2500

    # Spacing between prediction horizons (steps).  100 steps x 0.005s = 0.5s.
    HORIZON_STRIDE = 100
    NUM_HORIZONS   = 7

    def __init__(self, missions, seq_len=300, stride=10):
        self.sequences = []
        self.seq_len   = seq_len

        for m in missions:
            vel    = m['vel_meas']    # (T, 3)
            gyro   = m['gyro_meas']  # (T, 3)
            t_norm = m['t_norm_meas'].reshape(-1, 1)  # (T, 1)
            drift  = m['dr_pos'] - m['true_pos']      # (T, 3)

            features = np.concatenate([vel, gyro, t_norm], axis=1).astype(np.float32)  # (T, 7)
            T = len(features)

            RI = self.SHADOW_RESET_INTERVAL
            HS = self.HORIZON_STRIDE
            NH = self.NUM_HORIZONS

            for start in range(0, T, stride):
                end = start + seq_len
                horizon_idxs = [end + HS * (h + 1) for h in range(NH)]
                if horizon_idxs[-1] >= T:
                    break

                if end // RI != start // RI:
                    continue

                if any(hi // RI != (end - 1) // RI for hi in horizon_idxs):
                    continue

                seq  = features[start:end]                                               # (seq_len, 7)
                tgts = np.array([drift[hi] for hi in horizon_idxs], dtype=np.float32)  # (7, 3)
                self.sequences.append({
                    'input':   seq,
                    'targets': tgts,
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        return (torch.from_numpy(s['input']).float(),
                torch.from_numpy(s['targets']).float())


# =========================================================
# Load dataset
# =========================================================

print("[1/5] Loading dataset...\n")

data_dir  = Path('data/position_drift')
pkl_files = sorted(data_dir.glob('position_drift_precomp_*.pkl'))

if not pkl_files:
    print("  ERROR: Dataset not found!")
    print("  Run: python generate_position_drift_data.py\n")
    sys.exit(1)

# Load ONLY PyBullet-collected PKL files.
# Synthetic data uses fake gyro values (max ~0.5 rad/s) which don't match real
# quadrotor dynamics (5-15 rad/s on corners), so we exclude it entirely.
pybullet_files = [f for f in pkl_files if 'precomp' in f.name]
if not pybullet_files:
    print("  ERROR: No precomp dataset found!")
    print("  Run: python collect_pybullet_position_drift_data_precomp.py\n")
    sys.exit(1)

missions = []
for pkl_file in pybullet_files:
    with open(pkl_file, 'rb') as f:
        m = pickle.load(f)
    print(f"  Loading (pybullet): {pkl_file.name}  -> {len(m)} missions")
    missions.extend(m)
print(f"\n  Total missions: {len(missions)}\n")

# =========================================================
# Prepare sequences
# =========================================================

print("[2/5] Building sequences...\n")
dataset = PositionDriftDataset(missions, seq_len=300, stride=10)
print(f"  Total sequences: {len(dataset):,}\n")

train_size = int(0.70 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_ds, val_ds, test_ds = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=0, pin_memory=(device=='cuda'))
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0, pin_memory=(device=='cuda'))
test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=0, pin_memory=(device=='cuda'))

print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}\n")

# =========================================================
# Build model
# =========================================================

print("[3/5] Building model...\n")
model = PositionDriftLSTM(horizon_steps=7, hidden_size=256).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  Architecture: {model.hidden_size} hidden, {model.num_layers} LSTM layers\n")

# =========================================================
# Training
# =========================================================

print("[4/5] Training...\n")

SAVE_PATH = Path('experiments/lstm_position_drift_precomp.pt')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-3)   # scaled for batch_size=256
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10)

# Horizon weights: nearer horizons matter more (7 horizons, +0.5s to +3.5s)
h_weights = torch.tensor([0.25, 0.20, 0.17, 0.14, 0.11, 0.08, 0.05]).to(device)

best_val_loss   = float('inf')
patience        = 60
patience_counter = 0
history         = []
num_epochs      = 600

for epoch in range(num_epochs):
    # ---- Train ----
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)               # (B, 7, 3)
        loss  = sum(h_weights[h] * criterion(preds[:, h, :], y_batch[:, h, :])
                    for h in range(7))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # ---- Validate ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            preds   = model(x_batch)
            val_loss += sum(h_weights[h] * criterion(preds[:, h, :], y_batch[:, h, :])
                            for h in range(7)).item()
    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss   = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
    else:
        patience_counter += 1

    history.append({'epoch': epoch + 1, 'train': train_loss, 'val': val_loss})

    lr_now = optimizer.param_groups[0]['lr']
    improved = '*' if patience_counter == 0 else ' '
    print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
          f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | Best: {best_val_loss:.5f} | "
          f"LR: {lr_now:.2e} {improved}", flush=True)

    if patience_counter >= patience:
        print(f"\n  Early stop at epoch {epoch + 1}")
        break

print()

# =========================================================
# Evaluate on test set
# =========================================================

print("[5/5] Evaluating on test set...\n")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

test_loss = 0.0
horizon_errors = np.zeros(7)    # per-horizon MAE (metres)
horizon_counts = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        preds   = model(x_batch)

        wl = sum(h_weights[h] * criterion(preds[:, h, :], y_batch[:, h, :])
                 for h in range(7))
        test_loss += wl.item()

        # Per-horizon positional MAE (cm)
        for h in range(7):
            err = torch.norm(preds[:, h, :] - y_batch[:, h, :], dim=1).mean().item()
            horizon_errors[h] += err
        horizon_counts += 1

test_loss /= len(test_loader)
horizon_errors /= horizon_counts

print(f"  Test Loss (weighted MSE): {test_loss:.5f}")
print(f"\n  Per-horizon position error (mean L2, metres):")
HORIZON_MS = [500, 1000, 1500, 2000, 2500, 3000, 3500]
for h in range(7):
    print(f"    Horizon {h+1} (+{HORIZON_MS[h]}ms): {horizon_errors[h]*100:.2f} cm")

# =========================================================
# Save results
# =========================================================

results = {
    'timestamp':   datetime.now().isoformat(),
    'model':       'lstm_position_drift',
    'description': 'Predicts EKF dead-reckoning drift from velocity+gyro history',
    'architecture': {
        'input_features': ['vx', 'vy', 'vz', 'wx', 'wy', 'wz', 't_norm'],
        'seq_len':        300,
        'hidden_size':    model.hidden_size,
        'num_layers':     model.num_layers,
        'bidirectional':  False,
        'horizon_steps':  model.horizon_steps,
        'horizon_ms':     [500, 1000, 1500, 2000, 2500, 3000, 3500],
        'total_params':   int(total_params),
    },
    'dataset': {
        'total_sequences': len(dataset),
        'missions':        len(missions),
    },
    'results': {
        'best_val_loss': float(best_val_loss),
        'test_loss':     float(test_loss),
        'horizon_mae_cm': [float(e * 100) for e in horizon_errors],
    },
}

result_file = Path('experiments/lstm_position_drift_precomp_results.json')
with open(result_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved: {result_file}")

print("\n" + "="*100)
print("TRAINING COMPLETE")
print("="*100)
print(f"""
Position-Drift LSTM:

  Architecture : {model.hidden_size} hidden, {model.num_layers} LSTM layer(s), {model.horizon_steps} horizons
  Input        : velocity + gyro (GPS-independent)
  Target       : dead_reckoning_pos - true_pos  (metres)
  Parameters   : {total_params:,}

  Best val loss : {best_val_loss:.5f}
  Test loss     : {test_loss:.5f}

  Weights: {SAVE_PATH}

Next: the demo will automatically use this model if it exists.
      (It is a drop-in replacement for multiHorizon_drift_model_v3_fast.pt)
""")
