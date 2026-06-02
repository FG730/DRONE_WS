#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG_DIR = Path.home() / "drone_ws" / "logs"
DEFAULT_OUTPUT_DIR = Path.home() / "drone_ws" / "plots"


def main():
    parser = argparse.ArgumentParser(description="Plot visual tracking experiment CSV data.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Experiment CSV path. If omitted, use the newest CSV in ~/drone_ws/logs.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--capture-radius",
        type=float,
        default=None,
        help="Capture radius line to draw on distance plot. Defaults to first captured distance if available.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser() if args.csv_path else newest_csv(DEFAULT_LOG_DIR)
    if csv_path is None:
        raise SystemExit(f"No CSV files found in {DEFAULT_LOG_DIR}")
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_columns(csv_path)
    stem = csv_path.stem

    capture_time = first_capture_time(df)
    capture_radius = args.capture_radius
    if capture_radius is None:
        capture_radius = first_captured_distance(df)

    print(f"Reading: {csv_path}")
    print(f"Writing plots to: {output_dir}")
    print_summary(df, capture_time, capture_radius)

    plot_distance(df, output_dir / f"{stem}_distance.png", capture_time, capture_radius)
    plot_pixel_errors(df, output_dir / f"{stem}_pixel_error.png", capture_time)
    plot_commands(df, output_dir / f"{stem}_commands.png", capture_time)
    plot_xy_trajectory(df, output_dir / f"{stem}_xy_trajectory.png", capture_time)
    plot_3d_trajectory(df, output_dir / f"{stem}_3d_trajectory.png")
    plot_capture_state(df, output_dir / f"{stem}_capture_state.png", capture_time)

    print("Done.")


def newest_csv(log_dir):
    files = sorted(log_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def read_csv_columns(csv_path):
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"CSV file has no rows: {csv_path}")

    columns = {name: [] for name in reader.fieldnames}
    for row in rows:
        for name in columns:
            value = row.get(name, "")
            if name == "state":
                columns[name].append(value)
            else:
                columns[name].append(to_float(value))

    for name, values in columns.items():
        if name != "state":
            columns[name] = np.array(values, dtype=float)
    return columns


def to_float(value):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def first_capture_time(df):
    if "captured" not in df:
        return None
    indices = np.where(df["captured"] == 1)[0]
    if indices.size == 0:
        return None
    idx = indices[0]
    if "capture_time_sec" in df and np.isfinite(df["capture_time_sec"][idx]):
        return float(df["capture_time_sec"][idx])
    return float(df["time_sec"][idx])


def first_captured_distance(df):
    if "captured" not in df or "distance" not in df:
        return None
    indices = np.where((df["captured"] == 1) & np.isfinite(df["distance"]))[0]
    if indices.size == 0:
        return None
    return float(df["distance"][indices[0]])


def print_summary(df, capture_time, capture_radius):
    valid_distance = finite_values(df.get("distance", np.array([])))
    valid_error = finite_values(df.get("error_norm_px", np.array([])))
    valid_virtual_error = finite_values(df.get("virtual_error_norm_px", np.array([])))

    if valid_distance.size:
        print(f"Distance min/last: {np.min(valid_distance):.3f} m / {valid_distance[-1]:.3f} m")
    if valid_error.size:
        print(f"Pixel error mean/max: {np.mean(valid_error):.1f} px / {np.max(valid_error):.1f} px")
    if valid_virtual_error.size:
        print(
            "Virtual pixel error mean/max: "
            f"{np.mean(valid_virtual_error):.1f} px / {np.max(valid_virtual_error):.1f} px"
        )
    if capture_time is not None:
        print(f"Capture time: {capture_time:.3f} s")
    if capture_radius is not None:
        print(f"Capture radius/reference: {capture_radius:.3f} m")


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def plot_distance(df, path, capture_time, capture_radius):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_sec"], df["distance"], linewidth=2, label="distance")
    if capture_radius is not None:
        ax.axhline(capture_radius, linestyle="--", color="tab:red", label="capture radius")
    add_capture_line(ax, capture_time)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance to Target")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, path)


def plot_pixel_errors(df, path, capture_time):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time_sec"], df["error_x_px"], label="error_x")
    ax.plot(df["time_sec"], df["error_y_px"], label="error_y")
    ax.plot(df["time_sec"], df["error_norm_px"], label="error_norm", linewidth=2)
    if "virtual_error_norm_px" in df:
        ax.plot(
            df["time_sec"],
            df["virtual_error_norm_px"],
            label="virtual_error_norm",
            linewidth=2,
            linestyle="--",
        )
        ax.plot(
            df["time_sec"],
            df["virtual_error_x_px"],
            label="virtual_error_x",
            alpha=0.75,
            linestyle=":",
        )
        ax.plot(
            df["time_sec"],
            df["virtual_error_y_px"],
            label="virtual_error_y",
            alpha=0.75,
            linestyle=":",
        )
    add_capture_line(ax, capture_time)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pixel error (px)")
    ax.set_title("Image-Plane Tracking Error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, path)


def plot_commands(df, path, capture_time):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    command_cols = [
        ("cmd_vx", "Forward vx (m/s)"),
        ("cmd_vy", "Lateral vy (m/s)"),
        ("cmd_vz", "Vertical vz (m/s)"),
        ("cmd_yaw_rate", "Yaw rate (rad/s)"),
    ]
    for ax, (col, label) in zip(axes, command_cols):
        ax.plot(df["time_sec"], df[col], linewidth=1.6)
        add_capture_line(ax, capture_time)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Visual Servo Commands")
    save(fig, path)


def plot_xy_trajectory(df, path, capture_time):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(df["target_x"], df["target_y"], label="target", linewidth=2)
    ax.plot(df["drone_x"], df["drone_y"], label="drone", linewidth=2)
    mark_start_end(ax, df["target_x"], df["target_y"], "target")
    mark_start_end(ax, df["drone_x"], df["drone_y"], "drone")
    mark_capture_xy(ax, df, capture_time)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, path)


def plot_3d_trajectory(df, path):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(df["target_x"], df["target_y"], df["target_z"], label="target", linewidth=2)
    ax.plot(df["drone_x"], df["drone_y"], df["drone_z"], label="drone", linewidth=2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory")
    ax.legend()
    save(fig, path)


def plot_capture_state(df, path, capture_time):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.step(df["time_sec"], df["captured"], where="post", linewidth=2)
    add_capture_line(ax, capture_time)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Captured")
    ax.set_title("Virtual Capture State")
    ax.grid(True, alpha=0.3)
    save(fig, path)


def add_capture_line(ax, capture_time):
    if capture_time is not None:
        ax.axvline(capture_time, linestyle="--", color="tab:red", alpha=0.8, label="capture time")


def mark_start_end(ax, xs, ys, label):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    valid = np.where(np.isfinite(xs) & np.isfinite(ys))[0]
    if valid.size == 0:
        return
    first = valid[0]
    last = valid[-1]
    ax.scatter(xs[first], ys[first], marker="o", s=45)
    ax.scatter(xs[last], ys[last], marker="x", s=65)
    ax.annotate(f"{label} start", (xs[first], ys[first]))


def mark_capture_xy(ax, df, capture_time):
    if capture_time is None:
        return
    idx = int(np.nanargmin(np.abs(df["time_sec"] - capture_time)))
    if not np.isfinite(df["drone_x"][idx]) or not np.isfinite(df["drone_y"][idx]):
        return
    ax.scatter(df["drone_x"][idx], df["drone_y"][idx], marker="*", s=120, color="tab:red", label="capture")


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    main()
