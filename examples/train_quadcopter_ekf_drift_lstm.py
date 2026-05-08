#!/usr/bin/env python3
"""Train a multi-horizon LSTM on QuadcopterEKF shadow-drift data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from AI_UAV_Tests.ekf_lstm_precomp import (
    BUFFER_LEN,
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    HORIZON_STRIDE,
    NUM_HORIZONS,
    QuadcopterDriftDataset,
    QuadcopterPositionDriftLSTM,
    SHADOW_RESET_INTERVAL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM drift predictor for the 12-state QuadcopterEKF."
    )
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument(
        "--trajectory-filter",
        type=str,
        default="all",
        choices=["all", "hover", "circle", "square", "helix", "sine", "figure_eight"],
        help="Train only on one trajectory type instead of the full mixed dataset.",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Train from every compatible dataset in the data directory instead of only the newest compatible dataset.",
    )
    return parser.parse_args()


def load_missions(data_dir: Path, *, latest_only: bool) -> list[dict]:
    pkl_files = sorted(data_dir.glob("quadcopter_ekf_drift_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(
            f"No quadcopter EKF drift dataset found in {data_dir}. "
            "Run collect_quadcopter_ekf_drift_data.py first."
        )
    if latest_only:
        pkl_files = [pkl_files[-1]]

    missions: list[dict] = []
    skipped_legacy = 0
    for pkl_file in pkl_files:
        with pkl_file.open("rb") as f:
            data = pickle.load(f)
        filtered = [
            mission
            for mission in data
            if mission.get("collector_version") == "runtime_consistent_v2"
        ]
        skipped_legacy += len(data) - len(filtered)
        print(
            f"Loading {pkl_file.name:50s} -> "
            f"{len(filtered)} compatible / {len(data)} total missions"
        )
        missions.extend(filtered)
    if skipped_legacy > 0:
        print(f"Skipped legacy missions: {skipped_legacy}")
    if not missions:
        raise RuntimeError(
            "No compatible runtime_consistent_v2 missions found. "
            "Re-run collect_quadcopter_ekf_drift_data.py with the fixed collector."
        )
    return missions


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_dir)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 76}")
    print("Train QuadcopterEKF Drift LSTM")
    print(f"{'=' * 76}")
    print(f"Device             : {device}")
    print(f"Data dir           : {data_dir}")
    print(f"Save path          : {save_path}")
    print(f"Latest only        : {not args.all_datasets}")
    print(f"{'=' * 76}\n")

    missions = load_missions(data_dir, latest_only=not args.all_datasets)
    if args.trajectory_filter != "all":
        missions = [
            mission for mission in missions
            if str(mission.get("mission_type")) == str(args.trajectory_filter)
        ]
        if not missions:
            raise RuntimeError(
                f"No missions found for trajectory_filter={args.trajectory_filter!r}."
            )
        print(f"Filtered missions for {args.trajectory_filter}: {len(missions)}")
    print(f"\nTotal missions: {len(missions)}")

    dataset = QuadcopterDriftDataset(
        missions,
        seq_len=args.seq_len,
        stride=args.stride,
        horizon_stride=HORIZON_STRIDE,
        num_horizons=NUM_HORIZONS,
        shadow_reset_interval=SHADOW_RESET_INTERVAL,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after sequence extraction.")
    print(f"Extracted sequences : {len(dataset)}")

    val_size = max(1, int(round(0.1 * len(dataset))))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = QuadcopterPositionDriftLSTM(
        horizon_steps=NUM_HORIZONS,
        hidden_size=args.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for seq, tgts in train_loader:
            seq = seq.to(device)
            tgts = tgts.to(device)
            preds = model(seq)
            loss = F.smooth_l1_loss(preds, tgts)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = int(seq.shape[0])
            train_loss_sum += float(loss.item()) * batch_size
            train_count += batch_size

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for seq, tgts in val_loader:
                seq = seq.to(device)
                tgts = tgts.to(device)
                preds = model(seq)
                loss = F.smooth_l1_loss(preds, tgts)
                batch_size = int(seq.shape[0])
                val_loss_sum += float(loss.item()) * batch_size
                val_count += batch_size

        train_loss = train_loss_sum / max(1, train_count)
        val_loss = val_loss_sum / max(1, val_count)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= max(1, int(args.early_stop_patience)):
                print(
                    f"Early stopping at epoch {epoch:02d}: "
                    f"no validation improvement for {epochs_without_improvement} epoch(s)."
                )
                break

    meta = {
        "created_at": datetime.now().isoformat(),
        "device": device,
        "num_missions": len(missions),
        "num_sequences": len(dataset),
        "train_sequences": train_size,
        "val_sequences": val_size,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "horizon_stride": HORIZON_STRIDE,
        "num_horizons": NUM_HORIZONS,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "epochs_trained": len(history),
        "early_stop_patience": args.early_stop_patience,
        "trajectory_filter": args.trajectory_filter,
        "latest_only": bool(not args.all_datasets),
        "best_val_loss": best_val,
        "history": history,
    }
    meta_path = save_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nBest model saved to  : {save_path}")
    print(f"Training metadata    : {meta_path}")


if __name__ == "__main__":
    main()
