#!/usr/bin/env python3

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np


def main():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset, Subset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is not installed. Install torch first, then rerun this script."
        ) from exc

    parser = argparse.ArgumentParser(description="Train a lightweight LOS Transformer predictor.")
    parser.add_argument("--dataset", default="~/drone_ws/datasets/los_transformer_grouped.npz")
    parser.add_argument("--output-dir", default="~/drone_ws/models/los_transformer")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--risk-loss-weight", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(Path(args.dataset).expanduser(), allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    risk = data["risk"].astype(np.float32)
    sample_weight = data["sample_weight"].astype(np.float32)
    feature_mean = data["feature_mean"].astype(np.float32)
    feature_std = data["feature_std"].astype(np.float32)
    target_mean = data["target_mean"].astype(np.float32)
    target_std = data["target_std"].astype(np.float32)

    x = (x - feature_mean) / feature_std
    y_norm = (y - target_mean) / target_std

    dataset = LosDataset(x, y_norm, risk, sample_weight, torch)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - args.val_ratio)))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False)

    model = create_model(
        input_dim=x.shape[-1],
        output_dim=y.shape[-1],
        history_len=x.shape[1],
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        nn=nn,
        torch=torch,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    reg_loss_fn = nn.SmoothL1Loss(reduction="none")
    risk_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val = math.inf
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            reg_loss_fn,
            risk_loss_fn,
            args.risk_loss_weight,
            torch,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            reg_loss_fn,
            risk_loss_fn,
            args.risk_loss_weight,
            torch,
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "features": data["features"].tolist(),
                    "targets": data["targets"].tolist(),
                    "feature_mean": feature_mean,
                    "feature_std": feature_std,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "history_len": int(data["history_len"]),
                    "horizon": int(data["horizon"]),
                    "fx": float(data["fx"]),
                    "fy": float(data["fy"]),
                    "config": vars(args),
                },
                output_dir / "best.pt",
            )

        print(
            "epoch {:03d} train_loss={:.5f} val_loss={:.5f} val_reg={:.5f} val_risk={:.5f}".format(
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["reg_loss"],
                val_metrics["risk_loss"],
            )
        )

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"wrote {output_dir / 'best.pt'}")
    print(f"wrote {output_dir / 'history.json'}")


class LosDataset:
    def __init__(self, x, y, risk, sample_weight, torch):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.risk = torch.from_numpy(risk[:, None])
        self.sample_weight = torch.from_numpy(sample_weight[:, None])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.risk[index], self.sample_weight[index]


def create_model(input_dim, output_dim, history_len, d_model, heads, layers, dropout, nn, torch):
    class LosTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(input_dim, d_model)
            self.pos = nn.Parameter(torch.zeros(1, history_len, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.reg = nn.Linear(d_model, output_dim)
            self.risk = nn.Linear(d_model, 1)

        def forward(self, x):
            h = self.input(x) + self.pos[:, : x.shape[1], :]
            h = self.encoder(h)
            pooled = h[:, -1, :]
            z = self.head(pooled)
            return self.reg(z), self.risk(z)

    return LosTransformer()


def run_epoch(model, loader, optimizer, reg_loss_fn, risk_loss_fn, risk_loss_weight, torch):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total = 0.0
    total_reg = 0.0
    total_risk = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y, risk, weight in loader:
            pred, risk_logit = model(x)
            reg_loss = reg_loss_fn(pred, y).mean(dim=1, keepdim=True)
            risk_loss = risk_loss_fn(risk_logit, risk)
            loss = (weight * (reg_loss + risk_loss_weight * risk_loss)).mean()

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch = x.shape[0]
            total += float(loss.detach()) * batch
            total_reg += float((weight * reg_loss).mean().detach()) * batch
            total_risk += float((weight * risk_loss).mean().detach()) * batch
            count += batch

    denom = max(count, 1)
    return {
        "loss": total / denom,
        "reg_loss": total_reg / denom,
        "risk_loss": total_risk / denom,
    }


if __name__ == "__main__":
    main()
