#!/usr/bin/env python3
"""Train a calibration-parameter prediction MLP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset import CalibrationDataset, find_samples
from network import CalibrationMLP, torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibration prediction model")
    parser.add_argument("--data-dir", required=True, help="Directory containing .npz calibration samples")
    parser.add_argument("--output", default="model/checkpoints/calibration_mlp.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--glob", default="*.npz", help="Sample filename pattern inside --data-dir")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save epoch checkpoints every N epochs; 0 disables history saves")
    args = parser.parse_args()

    if torch is None:
        raise SystemExit("PyTorch is required for training")

    samples = sorted(Path(args.data_dir).glob(args.glob))
    if not samples:
        raise SystemExit(f"No .npz samples found in {args.data_dir}")

    dataset = CalibrationDataset(samples)
    feature_mean = np.mean(dataset.features, axis=0)
    feature_std = np.std(dataset.features, axis=0) + 1e-6
    x_np = (dataset.features - feature_mean) / feature_std

    x = torch.as_tensor(x_np, dtype=torch.float32)
    y = torch.as_tensor(dataset.targets, dtype=torch.float32)

    model = CalibrationMLP(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint_payload(epoch: int, loss_value: float) -> dict:
        return {
            "model_state": model.state_dict(),
            "input_dim": int(x.shape[1]),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "epoch": int(epoch),
            "loss": float(loss_value),
            "sample_paths": [str(p) for p in samples],
        }

    n = len(dataset)
    best_loss = float("inf")
    best_path = output.with_name(f"{output.stem}_best{output.suffix}")
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            pred = model(x[idx])
            loss = loss_fn(pred, y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        epoch_loss = total / n
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint_payload(epoch, epoch_loss), best_path)
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            epoch_path = output.with_name(f"{output.stem}_epoch{epoch:04d}{output.suffix}")
            torch.save(checkpoint_payload(epoch, epoch_loss), epoch_path)
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} loss={epoch_loss:.6f}")

    torch.save(checkpoint_payload(args.epochs, epoch_loss), output)
    latest_path = output.with_name(f"{output.stem}_latest{output.suffix}")
    torch.save(checkpoint_payload(args.epochs, epoch_loss), latest_path)
    print(f"saved {output}")
    print(f"saved latest {latest_path}")
    print(f"saved best {best_path} (loss={best_loss:.6f})")


if __name__ == "__main__":
    main()
