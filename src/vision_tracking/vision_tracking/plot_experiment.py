#!/usr/bin/env python3

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOG_DIR = Path.home() / "drone_ws" / "logs"
DEFAULT_OUTPUT_DIR = Path.home() / "drone_ws" / "plots"
MPS_TO_KMPH = 3.6


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
    parser.add_argument(
        "--speed-unit",
        choices=["kmh", "mps"],
        default="kmh",
        help="Display linear command speeds in km/h or m/s. CSV values remain SI units.",
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
    full_df = df
    df = truncate_at_first_capture(df)
    stem = csv_path.stem

    capture_time = first_capture_time(full_df)
    capture_radius = args.capture_radius
    if capture_radius is None:
        capture_radius = first_captured_distance(full_df)

    print(f"Reading: {csv_path}")
    print(f"Writing plots to: {output_dir}")
    print(f"Rows plotted: {len(df['time_sec'])} / {len(full_df['time_sec'])} (truncated at first capture)")
    lost_report = lost_before_capture(df)
    if lost_report["lost"]:
        print(
            "WARNING: target was LOST before capture; "
            f"lost_rows={lost_report['rows']}, max_lost_duration={lost_report['max_duration']:.3f}s"
        )
    print_summary(df, capture_time, capture_radius)

    plot_distance(df, output_dir / f"{stem}_distance.png", capture_time, capture_radius)
    plot_pixel_errors(df, output_dir / f"{stem}_pixel_error.png", capture_time)
    plot_commands(df, output_dir / f"{stem}_commands.png", capture_time, args.speed_unit)
    plot_xy_trajectory(df, output_dir / f"{stem}_xy_trajectory.png", capture_time)
    plot_3d_trajectory(df, output_dir / f"{stem}_3d_trajectory.png", capture_time)
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


def truncate_at_first_capture(df):
    if "captured" not in df:
        return df

    indices = np.where(df["captured"] == 1)[0]
    if indices.size == 0:
        return df

    end = int(indices[0]) + 1
    truncated = {}
    for name, values in df.items():
        if name == "state":
            truncated[name] = values[:end]
        else:
            truncated[name] = values[:end]
    return truncated


def first_captured_distance(df):
    if "captured" not in df or "distance" not in df:
        return None
    indices = np.where((df["captured"] == 1) & np.isfinite(df["distance"]))[0]
    if indices.size == 0:
        return None
    return float(df["distance"][indices[0]])


def lost_before_capture(df):
    states = df.get("state", [])
    first_seen = first_seen_index(df)
    if first_seen is None:
        return {"lost": False, "rows": 0, "max_duration": 0.0}

    lost_rows = 0
    max_duration = 0.0
    segment_start = None
    segment_last = None

    for index in range(first_seen, len(states)):
        row_time = float(df["time_sec"][index]) if "time_sec" in df and np.isfinite(df["time_sec"][index]) else np.nan
        if states[index] == "LOST":
            lost_rows += 1
            if segment_start is None:
                segment_start = row_time
            segment_last = row_time
            continue

        if segment_start is not None:
            max_duration = max(max_duration, segment_duration(segment_start, segment_last))
            segment_start = None
            segment_last = None

    if segment_start is not None:
        max_duration = max(max_duration, segment_duration(segment_start, segment_last))

    return {"lost": lost_rows > 0, "rows": lost_rows, "max_duration": max_duration}


def first_seen_index(df):
    states = df.get("state", [])
    error_x = df.get("error_x_px", np.full(len(states), np.nan))
    area = df.get("area_px", np.full(len(states), np.nan))
    for index, state in enumerate(states):
        if state in ("UNKNOWN", "SEARCH", "LOST", "CAPTURED"):
            continue
        if np.isfinite(error_x[index]) or np.isfinite(area[index]):
            return index
    return None


def segment_duration(start, end):
    if np.isfinite(start) and np.isfinite(end):
        return max(0.0, end - start)
    return 0.0


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


def plot_commands(df, path, capture_time, speed_unit):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    speed_scale = MPS_TO_KMPH if speed_unit == "kmh" else 1.0
    speed_label = "km/h" if speed_unit == "kmh" else "m/s"
    command_cols = [
        ("cmd_vx", f"Forward vx ({speed_label})", speed_scale),
        ("cmd_vy", f"Lateral vy ({speed_label})", speed_scale),
        ("cmd_vz", f"Vertical vz ({speed_label})", speed_scale),
        ("cmd_yaw_rate", "Yaw rate (rad/s)", 1.0),
    ]
    for ax, (col, label, scale) in zip(axes, command_cols):
        ax.plot(df["time_sec"], df[col] * scale, linewidth=1.6)
        add_capture_line(ax, capture_time)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Visual Servo Commands")
    save(fig, path)


def plot_xy_trajectory(df, path, capture_time):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(df["target_x"], df["target_y"], label="target truth", linewidth=2.2, color="tab:red")
    ax.plot(df["drone_x"], df["drone_y"], label="drone actual", linewidth=2.2, color="tab:blue")
    mark_start_end(ax, df["target_x"], df["target_y"], "target")
    mark_start_end(ax, df["drone_x"], df["drone_y"], "drone")
    mark_capture_xy(ax, df, capture_time)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory: Target and Drone")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, path)


def plot_3d_trajectory(df, path, capture_time):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        df["target_x"],
        df["target_y"],
        df["target_z"],
        label="target truth",
        linewidth=2.2,
        color="tab:red",
    )
    ax.plot(
        df["drone_x"],
        df["drone_y"],
        df["drone_z"],
        label="drone actual",
        linewidth=2.2,
        color="tab:blue",
    )
    mark_start_end_3d(ax, df["target_x"], df["target_y"], df["target_z"], "target")
    mark_start_end_3d(ax, df["drone_x"], df["drone_y"], df["drone_z"], "drone")
    mark_capture_3d(ax, df, capture_time)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectory: Target and Drone")
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


def mark_start_end_3d(ax, xs, ys, zs, label):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    valid = np.where(np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs))[0]
    if valid.size == 0:
        return
    first = valid[0]
    last = valid[-1]
    ax.scatter(xs[first], ys[first], zs[first], marker="o", s=45)
    ax.scatter(xs[last], ys[last], zs[last], marker="x", s=65)
    ax.text(xs[first], ys[first], zs[first], f"{label} start")


def mark_capture_3d(ax, df, capture_time):
    if capture_time is None:
        return
    idx = int(np.nanargmin(np.abs(df["time_sec"] - capture_time)))
    if not (
        np.isfinite(df["drone_x"][idx])
        and np.isfinite(df["drone_y"][idx])
        and np.isfinite(df["drone_z"][idx])
    ):
        return
    ax.scatter(
        df["drone_x"][idx],
        df["drone_y"][idx],
        df["drone_z"][idx],
        marker="*",
        s=120,
        color="tab:red",
        label="capture",
    )


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    main()
