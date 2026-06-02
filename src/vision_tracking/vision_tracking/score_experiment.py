#!/usr/bin/env python3

import csv
import glob
import math
import os
from pathlib import Path


DEFAULT_LOG_DIR = Path.home() / "drone_ws" / "logs"


def main():
    path = newest_csv()
    if path is None:
        raise SystemExit(f"No CSV files found in {DEFAULT_LOG_DIR}")

    rows = read_rows(path)
    print(f"CSV: {path}")
    print(f"rows: {len(rows)}")

    captured_rows = [row for row in rows if row.get("captured") == "1"]
    active_end = rows.index(captured_rows[0]) + 1 if captured_rows else len(rows)
    active_rows = [
        row
        for row in rows[:active_end]
        if row.get("state") not in ("UNKNOWN", "SEARCH", "LOST", "CAPTURED")
    ]

    print(f"captured: {bool(captured_rows)}")
    if captured_rows:
        first = captured_rows[0]
        print(f"capture_time_sec: {first.get('capture_time_sec', '')}")
        print(f"capture_target_time_sec: {first.get('capture_target_time_sec', '')}")
        print(f"capture_distance: {first.get('distance', '')}")

    for name in (
        "distance",
        "virtual_error_norm_px",
        "virtual_error_y_px",
        "cmd_vx",
        "cmd_vy",
        "cmd_vz",
        "cmd_yaw_rate",
        "drone_z",
        "area_px",
    ):
        values = finite_values(active_rows, name)
        if not values:
            continue
        print(
            f"{name}: mean={mean(values):.3f} min={min(values):.3f} "
            f"max={max(values):.3f} last={values[-1]:.3f}"
        )

    distance = finite_values(active_rows, "distance")
    if distance:
        print(f"min_distance: {min(distance):.3f}")
        print(f"distance_pullaway_after_min: {pullaway_after_min(distance):.3f}")

    cmd_vz = finite_values(active_rows, "cmd_vz")
    if cmd_vz:
        print(f"cmd_vz_sign_changes: {sign_changes(cmd_vz)}")
        print(f"cmd_vz_mean_abs_delta: {mean_abs_delta(cmd_vz):.3f}")

    drone_z = finite_values(active_rows, "drone_z")
    if drone_z:
        print(f"drone_z_range: {max(drone_z) - min(drone_z):.3f}")

    print(f"score_lower_is_better: {score(active_rows, captured_rows):.3f}")


def newest_csv():
    files = glob.glob(str(DEFAULT_LOG_DIR / "*.csv"))
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


def read_rows(path):
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def finite_values(rows, name):
    values = []
    for row in rows:
        value = to_float(row.get(name, ""))
        if math.isfinite(value):
            values.append(value)
    return values


def to_float(value):
    try:
        return float(value) if value not in ("", None) else math.nan
    except ValueError:
        return math.nan


def mean(values):
    return sum(values) / len(values)


def sign_changes(values):
    return sum(1 for prev, cur in zip(values, values[1:]) if prev * cur < 0.0)


def mean_abs_delta(values):
    if len(values) < 2:
        return 0.0
    return mean([abs(cur - prev) for prev, cur in zip(values, values[1:])])


def pullaway_after_min(distance):
    if not distance:
        return math.nan
    min_index = min(range(len(distance)), key=distance.__getitem__)
    return max(distance[min_index:]) - distance[min_index]


def score(active_rows, captured_rows):
    distance = finite_values(active_rows, "distance")
    cmd_vz = finite_values(active_rows, "cmd_vz")
    drone_z = finite_values(active_rows, "drone_z")

    min_distance = min(distance) if distance else 99.0
    pullaway = pullaway_after_min(distance) if distance else 99.0
    z_range = max(drone_z) - min(drone_z) if drone_z else 0.0
    vz_chatter = mean_abs_delta(cmd_vz) if cmd_vz else 0.0

    if captured_rows:
        capture_time = to_float(captured_rows[0].get("capture_target_time_sec", ""))
        capture_term = capture_time if math.isfinite(capture_time) else 50.0
        miss_penalty = 0.0
    else:
        capture_term = 50.0
        miss_penalty = 100.0

    return capture_term + miss_penalty + 10.0 * min_distance + 3.0 * pullaway + 5.0 * z_range + 20.0 * vz_chatter


if __name__ == "__main__":
    main()
