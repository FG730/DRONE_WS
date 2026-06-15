#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_FEATURES = [
    "error_x_px",
    "error_y_px",
    "area_px",
    "virtual_error_x_px",
    "virtual_error_y_px",
    "cmd_vx",
    "cmd_vy",
    "cmd_vz",
    "cmd_yaw_rate",
]

DEFAULT_TARGETS = [
    "error_x_px",
    "error_y_px",
    "area_px",
]


def main():
    parser = argparse.ArgumentParser(
        description="Build a Transformer time-series dataset from visual servoing CSV logs."
    )
    parser.add_argument("csv_paths", nargs="+", help="Input experiment CSV files or directories.")
    parser.add_argument("--output", default="~/drone_ws/datasets/transformer_visual_servo.npz")
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--include-captured", action="store_true", help="Keep rows after virtual capture.")
    parser.add_argument("--allow-lost", action="store_true", help="Allow windows that include LOST rows.")
    parser.add_argument("--min-area", type=float, default=1.0)
    args = parser.parse_args()

    features = split_names(args.features)
    targets = split_names(args.targets)
    csv_files = expand_csv_paths(args.csv_paths)
    if not csv_files:
        raise SystemExit("No CSV files found.")

    xs = []
    ys = []
    sources = []
    for csv_path in csv_files:
        rows = read_rows(csv_path)
        if not rows:
            continue
        x_part, y_part = build_samples(
            rows,
            features,
            targets,
            args.history_len,
            args.horizon,
            args.include_captured,
            args.allow_lost,
            args.min_area,
        )
        if x_part.size == 0:
            print(f"skip {csv_path}: no valid windows")
            continue
        xs.append(x_part)
        ys.append(y_part)
        sources.extend([str(csv_path)] * x_part.shape[0])
        print(f"{csv_path}: {x_part.shape[0]} samples")

    if not xs:
        raise SystemExit("No valid samples generated.")

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    std = x.reshape(-1, x.shape[-1]).std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        features=np.array(features),
        targets=np.array(targets),
        sources=np.array(sources),
        history_len=np.array(args.history_len),
        horizon=np.array(args.horizon),
    )

    meta_path = output.with_suffix(".json")
    meta = {
        "output": str(output),
        "num_samples": int(x.shape[0]),
        "history_len": args.history_len,
        "horizon": args.horizon,
        "features": features,
        "targets": targets,
        "csv_files": [str(path) for path in csv_files],
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {output}")
    print(f"wrote {meta_path}")
    print(f"x shape={x.shape}, y shape={y.shape}")


def split_names(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def expand_csv_paths(paths):
    csv_files = []
    for item in paths:
        path = Path(item).expanduser()
        if path.is_dir():
            csv_files.extend(sorted(path.glob("*.csv")))
        elif path.exists():
            csv_files.append(path)
    return csv_files


def read_rows(csv_path):
    with open(csv_path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_samples(rows, features, targets, history_len, horizon, include_captured, allow_lost, min_area):
    feature_matrix = rows_to_matrix(rows, features)
    target_matrix = rows_to_matrix(rows, targets)
    valid = valid_rows(rows, feature_matrix, target_matrix, include_captured, allow_lost, min_area)
    xs = []
    ys = []
    last_start = len(rows) - history_len - horizon + 1
    for start in range(max(last_start, 0)):
        hist = slice(start, start + history_len)
        label = start + history_len + horizon - 1
        if not np.all(valid[hist]) or not valid[label]:
            continue
        xs.append(feature_matrix[hist])
        ys.append(target_matrix[label])
    if not xs:
        return np.empty((0, history_len, len(features))), np.empty((0, len(targets)))
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def rows_to_matrix(rows, names):
    matrix = np.empty((len(rows), len(names)), dtype=float)
    for row_index, row in enumerate(rows):
        for col_index, name in enumerate(names):
            matrix[row_index, col_index] = to_float(row.get(name, ""))
    return matrix


def valid_rows(rows, feature_matrix, target_matrix, include_captured, allow_lost, min_area):
    finite = np.all(np.isfinite(feature_matrix), axis=1) & np.all(np.isfinite(target_matrix), axis=1)
    area_ok = feature_matrix[:, 2] >= min_area if feature_matrix.shape[1] >= 3 else True
    state_ok = np.ones(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        state = row.get("state", "")
        if not allow_lost and state in ("UNKNOWN", "SEARCH", "LOST"):
            state_ok[index] = False
        if not include_captured and row.get("captured", "0") == "1":
            state_ok[index] = False
    return finite & area_ok & state_ok


def to_float(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


if __name__ == "__main__":
    main()
