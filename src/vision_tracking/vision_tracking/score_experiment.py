#!/usr/bin/env python3

import csv
import glob
import math
import os
import argparse
from pathlib import Path


DEFAULT_LOG_DIR = Path.home() / "drone_ws" / "logs"


def main():
    parser = argparse.ArgumentParser(description="Score a visual tracking experiment CSV.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        help="Experiment CSV path. If omitted, use the newest CSV in ~/drone_ws/logs.",
    )
    args = parser.parse_args()

    path = Path(args.csv_path).expanduser() if args.csv_path else newest_csv()
    if path is None:
        raise SystemExit(f"No CSV files found in {DEFAULT_LOG_DIR}")
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")

    rows = read_rows(path)
    print(f"CSV: {path}")
    print(f"rows: {len(rows)}")

    captured_rows = [row for row in rows if row.get("captured") == "1"]
    active_end = rows.index(captured_rows[0]) + 1 if captured_rows else len(rows)
    first_seen = first_seen_index(rows[:active_end])
    lost_report = lost_before_capture(rows[:active_end], first_seen)
    active_rows = [
        row
        for row in rows[:active_end]
        if row.get("state") not in ("UNKNOWN", "SEARCH", "LOST", "CAPTURED")
    ]

    print(f"captured: {bool(captured_rows)}")
    print(f"valid_continuous_capture: {bool(captured_rows) and not lost_report['lost']}")
    print(f"first_seen_time_sec: {time_at(rows, first_seen)}")
    print(f"lost_before_capture: {lost_report['lost']}")
    print(f"lost_rows_before_capture: {lost_report['rows']}")
    print(f"max_lost_duration_before_capture: {lost_report['max_duration']:.3f}")
    print(f"active_rows_before_capture: {len(active_rows)}")
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

    print(f"score_lower_is_better: {score(active_rows, captured_rows, lost_report):.3f}")


def newest_csv():
    files = glob.glob(str(DEFAULT_LOG_DIR / "*.csv"))
    if not files:
        return None
    return Path(max(files, key=os.path.getmtime))


def read_rows(path):
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def first_seen_index(rows):
    for index, row in enumerate(rows):
        if target_is_seen(row):
            return index
    return None


def target_is_seen(row):
    state = row.get("state", "")
    if state in ("UNKNOWN", "SEARCH", "LOST", "CAPTURED"):
        return False
    return math.isfinite(to_float(row.get("error_x_px", ""))) or math.isfinite(to_float(row.get("area_px", "")))


def time_at(rows, index):
    if index is None or index >= len(rows):
        return ""
    value = to_float(rows[index].get("time_sec", ""))
    return f"{value:.3f}" if math.isfinite(value) else ""


def lost_before_capture(rows, first_seen):
    if first_seen is None:
        return {"lost": False, "rows": 0, "max_duration": 0.0}

    lost_rows = 0
    max_duration = 0.0
    segment_start = None
    segment_last = None

    for row in rows[first_seen:]:
        state = row.get("state", "")
        row_time = to_float(row.get("time_sec", ""))
        if state == "LOST":
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


def segment_duration(start, end):
    if math.isfinite(start) and math.isfinite(end):
        return max(0.0, end - start)
    return 0.0


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


def score(active_rows, captured_rows, lost_report=None):
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

    lost_penalty = 0.0
    if lost_report and lost_report["lost"]:
        lost_penalty = 200.0 + 20.0 * lost_report["max_duration"] + 0.5 * lost_report["rows"]

    return (
        capture_term
        + miss_penalty
        + lost_penalty
        + 10.0 * min_distance
        + 3.0 * pullaway
        + 5.0 * z_range
        + 20.0 * vz_chatter
    )


if __name__ == "__main__":
    main()
