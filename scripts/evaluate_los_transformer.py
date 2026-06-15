#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from train_los_transformer import create_model


def main():
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is not installed. Install torch first.") from exc

    parser = argparse.ArgumentParser(description="Evaluate a trained LOS Transformer model.")
    parser.add_argument("--dataset", default="~/drone_ws/datasets/los_transformer_grouped.npz")
    parser.add_argument("--model", default="~/drone_ws/models/los_transformer_v1/best.pt")
    parser.add_argument("--history", default="~/drone_ws/models/los_transformer_v1/history.json")
    parser.add_argument("--output-dir", default="~/drone_ws/plots/los_transformer_v1")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser()
    model_path = Path(args.model).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path, allow_pickle=True)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})

    x_raw = data["x"].astype(np.float32)
    y_true = data["y"].astype(np.float32)
    risk_true = data["risk"].astype(np.float32)
    groups = data["group"]
    sources = data["source"]
    feature_mean = checkpoint["feature_mean"].astype(np.float32)
    feature_std = checkpoint["feature_std"].astype(np.float32)
    target_mean = checkpoint["target_mean"].astype(np.float32)
    target_std = checkpoint["target_std"].astype(np.float32)
    target_names = checkpoint.get("targets", data.get("targets", np.array([f"target_{i}" for i in range(y_true.shape[1])])))
    x = (x_raw - feature_mean) / feature_std

    model = create_model(
        input_dim=x.shape[-1],
        output_dim=y_true.shape[-1],
        history_len=x.shape[1],
        d_model=int(config.get("d_model", 64)),
        heads=int(config.get("heads", 4)),
        layers=int(config.get("layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
        nn=nn,
        torch=torch,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_pred_norm = []
    risk_logits = []
    with torch.no_grad():
        for start in range(0, x.shape[0], args.batch_size):
            batch = torch.from_numpy(x[start : start + args.batch_size])
            pred, risk_logit = model(batch)
            y_pred_norm.append(pred.numpy())
            risk_logits.append(risk_logit.numpy()[:, 0])

    y_pred = np.concatenate(y_pred_norm, axis=0) * target_std + target_mean
    risk_logit = np.concatenate(risk_logits, axis=0)
    risk_prob = sigmoid(risk_logit)

    metrics = {
        "model": str(model_path),
        "dataset": str(dataset_path),
        "samples": int(x.shape[0]),
        "overall": regression_metrics(y_true, y_pred, target_names),
        "risk": risk_metrics(risk_true, risk_prob, args.risk_threshold),
        "groups": {},
    }
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        metrics["groups"][group] = {
            "samples": int(np.sum(mask)),
            "regression": regression_metrics(y_true[mask], y_pred[mask], target_names),
            "risk": risk_metrics(risk_true[mask], risk_prob[mask], args.risk_threshold),
        }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    plot_training_history(Path(args.history).expanduser(), output_dir / "training_history.png")
    plot_prediction_scatter(y_true, y_pred, output_dir / "prediction_scatter.png", target_names)
    plot_risk_histogram(risk_true, risk_prob, output_dir / "risk_probability.png")
    plot_source_examples(y_true, y_pred, risk_true, risk_prob, groups, sources, output_dir, target_names)

    print(f"wrote {metrics_path}")
    print(f"wrote plots to {output_dir}")


def regression_metrics(y_true, y_pred, names):
    if y_true.size == 0:
        return {}
    err = y_pred - y_true
    result = {}
    names = [str(name) for name in names]
    for index, name in enumerate(names):
        result[name] = {
            "mae": float(np.mean(np.abs(err[:, index]))),
            "rmse": float(np.sqrt(np.mean(err[:, index] ** 2))),
        }
    if err.shape[1] >= 2:
        result["los_angle_norm_mae"] = float(np.mean(np.linalg.norm(err[:, :2], axis=1)))
    return result


def risk_metrics(y_true, prob, threshold):
    if y_true.size == 0:
        return {}
    pred = prob >= threshold
    truth = y_true >= 0.5
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    tn = int(np.sum(~pred & ~truth))
    fn = int(np.sum(~pred & truth))
    accuracy = (tp + tn) / max(len(truth), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "positive_rate": float(np.mean(truth)),
        "prob_mean": float(np.mean(prob)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def plot_training_history(path, output):
    if not path.exists():
        return
    history = json.loads(path.read_text())
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train"]["loss"] for row in history]
    val_loss = [row["val"]["loss"] for row in history]
    val_reg = [row["val"]["reg_loss"] for row in history]
    val_risk = [row["val"]["risk_loss"] for row in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.plot(epochs, val_reg, label="val regression")
    plt.plot(epochs, val_risk, label="val risk")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_prediction_scatter(y_true, y_pred, output, names):
    names = [str(name) for name in names]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.7))
    axes = np.atleast_1d(axes)
    for index, ax in enumerate(axes):
        ax.scatter(y_true[:, index], y_pred[:, index], s=3, alpha=0.25)
        lo = min(float(np.min(y_true[:, index])), float(np.min(y_pred[:, index])))
        hi = max(float(np.max(y_true[:, index])), float(np.max(y_pred[:, index])))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_title(names[index])
        ax.set_xlabel("true")
        ax.set_ylabel("pred")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_risk_histogram(y_true, prob, output):
    plt.figure(figsize=(7, 4.2))
    plt.hist(prob[y_true < 0.5], bins=40, alpha=0.65, label="safe")
    plt.hist(prob[y_true >= 0.5], bins=40, alpha=0.65, label="risk")
    plt.xlabel("predicted risk probability")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def plot_source_examples(y_true, y_pred, risk_true, risk_prob, groups, sources, output_dir, target_names):
    target_names = [str(name) for name in target_names]
    selected = []
    for group in ("clean_success", "boundary_success", "failure_risk"):
        candidates = [src for src in sorted(set(sources.tolist())) if np.any((sources == src) & (groups == group))]
        if candidates:
            selected.append((group, candidates[0]))

    for group, src in selected:
        mask = sources == src
        true_part = y_true[mask]
        pred_part = y_pred[mask]
        risk_part = risk_true[mask]
        prob_part = risk_prob[mask]
        if true_part.shape[0] == 0:
            continue
        limit = min(true_part.shape[0], 260)
        idx = np.arange(limit)
        fig, axes = plt.subplots(len(target_names) + 1, 1, figsize=(9, 2.2 * (len(target_names) + 1)), sharex=True)
        axes = np.atleast_1d(axes)
        for axis_index, ax in enumerate(axes[: len(target_names)]):
            ax.plot(idx, true_part[:limit, axis_index], label="true", linewidth=1.2)
            ax.plot(idx, pred_part[:limit, axis_index], label="pred", linewidth=1.0)
            ax.set_ylabel(target_names[axis_index])
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
        risk_axis = axes[len(target_names)]
        risk_axis.plot(idx, risk_part[:limit], label="risk label", linewidth=1.2)
        risk_axis.plot(idx, prob_part[:limit], label="risk prob", linewidth=1.0)
        risk_axis.set_ylabel("risk")
        risk_axis.set_xlabel("sample index within CSV")
        risk_axis.grid(True, alpha=0.3)
        risk_axis.legend(loc="upper right")
        fig.suptitle(f"{group}: {Path(src).name}", fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / f"example_{group}.png", dpi=160)
        plt.close(fig)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    main()
