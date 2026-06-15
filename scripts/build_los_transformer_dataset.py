#!/usr/bin/env python3

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


LOS_FEATURES = [
    "theta_x",
    "theta_y",
    "theta_dot_x",
    "theta_dot_y",
    "area_log",
    "area_dot",
    "virtual_theta_x",
    "virtual_theta_y",
    "cmd_vx",
    "cmd_vy",
    "cmd_vz",
    "cmd_yaw_rate",
    "drone_vx",
    "drone_vy",
    "drone_vz",
    "drone_z",
    "drone_roll",
    "drone_pitch",
    "drone_yaw",
]

LOS_TARGETS = [
    "delta_theta_x",
    "delta_theta_y",
    "delta_area_log",
]

GROUP_WEIGHTS = {
    "clean_success": 1.0,
    "boundary_success": 0.4,
    "failure_risk": 0.15,
}

BAD_STATES = {"UNKNOWN", "SEARCH", "LOST"}


def main():
    parser = argparse.ArgumentParser(
        description="Build a grouped LOS Transformer dataset with sample weights and risk labels."
    )
    parser.add_argument("csv_paths", nargs="+", help="Input experiment CSV files or directories.")
    parser.add_argument("--output", default="~/drone_ws/datasets/los_transformer_grouped.npz")
    parser.add_argument("--manifest", default="~/drone_ws/datasets/los_transformer_manifest.csv")
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--risk-lookahead", type=int, default=10)
    parser.add_argument("--image-width", type=float, default=1280.0)
    parser.add_argument("--image-height", type=float, default=960.0)
    parser.add_argument("--horizontal-fov-rad", type=float, default=1.74)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--min-file-samples", type=int, default=80)
    parser.add_argument("--include-excluded", action="store_true")
    args = parser.parse_args()

    csv_files = expand_csv_paths(args.csv_paths)
    if not csv_files:
        raise SystemExit("No CSV files found.")

    fx, fy = camera_focal_lengths(args.image_width, args.image_height, args.horizontal_fov_rad)
    manifest_rows = []
    all_x = []
    all_y = []
    all_risk = []
    all_weight = []
    all_group = []
    all_source = []

    for csv_path in csv_files:
        rows = read_rows(csv_path)
        analysis = analyze_csv(rows, csv_path, fx, fy, args)
        group, group_weight, reasons = classify_file(analysis, args)
        analysis["group"] = group
        analysis["weight"] = group_weight
        analysis["reasons"] = ";".join(reasons)
        manifest_rows.append(analysis)

        if group == "exclude" and not args.include_excluded:
            print(f"exclude {csv_path}: {analysis['reasons']}")
            continue

        x_part, y_part, risk_part, weight_part = build_samples(rows, fx, fy, group_weight, args)
        if x_part.size == 0:
            print(f"skip {csv_path}: no valid windows")
            continue

        all_x.append(x_part)
        all_y.append(y_part)
        all_risk.append(risk_part)
        all_weight.append(weight_part)
        all_group.extend([group] * x_part.shape[0])
        all_source.extend([str(csv_path)] * x_part.shape[0])
        print(f"{csv_path}: group={group} samples={x_part.shape[0]} weight={group_weight:.2f}")

    write_manifest(manifest_rows, args.manifest)

    if not all_x:
        raise SystemExit("No valid samples generated.")

    x = np.concatenate(all_x, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    risk = np.concatenate(all_risk, axis=0).astype(np.float32)
    weight = np.concatenate(all_weight, axis=0).astype(np.float32)
    groups = np.array(all_group)
    sources = np.array(all_source)

    feature_mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    feature_std = x.reshape(-1, x.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

    target_mean = y.mean(axis=0)
    target_std = y.std(axis=0)
    target_std = np.where(target_std < 1e-6, 1.0, target_std)

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        risk=risk,
        sample_weight=weight,
        group=groups,
        source=sources,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        features=np.array(LOS_FEATURES),
        targets=np.array(LOS_TARGETS),
        history_len=np.array(args.history_len),
        horizon=np.array(args.horizon),
        fx=np.array(fx, dtype=np.float32),
        fy=np.array(fy, dtype=np.float32),
    )

    summary = dataset_summary(groups, risk, weight)
    meta = {
        "output": str(output),
        "manifest": str(Path(args.manifest).expanduser()),
        "num_samples": int(x.shape[0]),
        "history_len": args.history_len,
        "horizon": args.horizon,
        "risk_lookahead": args.risk_lookahead,
        "features": LOS_FEATURES,
        "targets": LOS_TARGETS,
        "fx": fx,
        "fy": fy,
        "group_weights": GROUP_WEIGHTS,
        "summary": summary,
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
    }
    meta_path = output.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"wrote {output}")
    print(f"wrote {meta_path}")
    print(f"wrote {Path(args.manifest).expanduser()}")
    print(f"x shape={x.shape}, y shape={y.shape}, risk shape={risk.shape}")
    print(json.dumps(summary, indent=2))


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


def camera_focal_lengths(width, height, horizontal_fov):
    fx = width / (2.0 * math.tan(horizontal_fov / 2.0))
    vertical_fov = 2.0 * math.atan(math.tan(horizontal_fov / 2.0) * height / width)
    fy = height / (2.0 * math.tan(vertical_fov / 2.0))
    return fx, fy


def analyze_csv(rows, csv_path, fx, fy, args):
    valid_visual = []
    target_points = []
    states = {}
    captured = False
    distances = []
    areas = []

    for row in rows:
        state = row.get("state", "")
        states[state] = states.get(state, 0) + 1
        captured = captured or row.get("captured", "0") == "1"
        area = to_float(row.get("area_px", ""))
        if math.isfinite(area):
            areas.append(area)
        distance = to_float(row.get("distance", ""))
        if math.isfinite(distance):
            distances.append(distance)
        point = target_point(row)
        if point is not None:
            target_points.append(point)

    visual_matrix = make_feature_rows(rows, fx, fy)
    for index, feature_row in enumerate(visual_matrix):
        if row_is_valid(rows[index], feature_row, args.min_area):
            valid_visual.append(index)

    speeds, jumps, repeats = target_motion_stats(target_points)
    rough_samples = max(0, len(valid_visual) - args.history_len - args.horizon + 1)
    idx_match = re.search(r"random_(\d+)_seed_(\d+)", csv_path.name)

    return {
        "file": str(csv_path),
        "index": int(idx_match.group(1)) if idx_match else -1,
        "seed": int(idx_match.group(2)) if idx_match else -1,
        "rows": len(rows),
        "valid_visual_rows": len(valid_visual),
        "rough_samples": rough_samples,
        "captured": int(captured),
        "lost_rows": states.get("LOST", 0),
        "dist_start": distances[0] if distances else math.nan,
        "dist_min": min(distances) if distances else math.nan,
        "dist_end": distances[-1] if distances else math.nan,
        "area_max": max(areas) if areas else math.nan,
        "speed_median": median(speeds),
        "speed_max": max(speeds) if speeds else math.nan,
        "target_jumps": jumps,
        "target_repeats": repeats,
    }


def classify_file(analysis, args):
    reasons = []
    if analysis["rough_samples"] < args.min_file_samples:
        reasons.append("low_samples")
    if not math.isfinite(analysis["area_max"]):
        reasons.append("no_area")
    if analysis["target_jumps"] > 0 or analysis["target_repeats"] > 5:
        reasons.append("target_motion")

    if any(reason in reasons for reason in ("low_samples", "no_area", "target_motion")):
        return "exclude", 0.0, reasons

    captured = bool(analysis["captured"])
    lost = analysis["lost_rows"]
    dist_min = analysis["dist_min"]
    speed_max = analysis["speed_max"]

    if (
        captured
        and lost == 0
        and math.isfinite(dist_min)
        and dist_min <= 1.0
        and (not math.isfinite(speed_max) or speed_max <= 9.0)
    ):
        return "clean_success", GROUP_WEIGHTS["clean_success"], reasons

    if captured or (math.isfinite(dist_min) and dist_min <= 2.0 and lost <= 10):
        if math.isfinite(speed_max) and speed_max > 9.0:
            reasons.append("speed_spike")
        return "boundary_success", GROUP_WEIGHTS["boundary_success"], reasons

    reasons.append("failure_or_far")
    return "failure_risk", GROUP_WEIGHTS["failure_risk"], reasons


def build_samples(rows, fx, fy, group_weight, args):
    features = make_feature_rows(rows, fx, fy)
    absolute_targets = features[:, [0, 1, 4]]
    valid = np.array([row_is_valid(row, feature, args.min_area) for row, feature in zip(rows, features)])

    xs = []
    ys = []
    risks = []
    weights = []
    last_start = len(rows) - args.history_len - args.horizon + 1
    for start in range(max(last_start, 0)):
        hist = slice(start, start + args.history_len)
        label = start + args.history_len + args.horizon - 1
        if not np.all(valid[hist]) or not valid[label]:
            continue

        current = absolute_targets[start + args.history_len - 1]
        future = absolute_targets[label]
        xs.append(features[hist])
        ys.append(future - current)
        risks.append(risk_label(rows, features, start + args.history_len, label, args.risk_lookahead))
        weights.append(group_weight)

    if not xs:
        return (
            np.empty((0, args.history_len, len(LOS_FEATURES))),
            np.empty((0, len(LOS_TARGETS))),
            np.empty((0,)),
            np.empty((0,)),
        )
    return np.stack(xs), np.stack(ys), np.array(risks), np.array(weights)


def make_feature_rows(rows, fx, fy):
    values = []
    prev_t = None
    prev_theta_x = None
    prev_theta_y = None
    prev_area_log = None
    prev_drone_t = None
    prev_drone_pos = None

    for row in rows:
        t = to_float(row.get("time_sec", ""))
        error_x = to_float(row.get("error_x_px", ""))
        error_y = to_float(row.get("error_y_px", ""))
        area = to_float(row.get("area_px", ""))
        virtual_x = to_float(row.get("virtual_error_x_px", ""))
        virtual_y = to_float(row.get("virtual_error_y_px", ""))

        theta_x = math.atan2(error_x, fx) if math.isfinite(error_x) else math.nan
        theta_y = math.atan2(error_y, fy) if math.isfinite(error_y) else math.nan
        virtual_theta_x = math.atan2(virtual_x, fx) if math.isfinite(virtual_x) else math.nan
        virtual_theta_y = math.atan2(virtual_y, fy) if math.isfinite(virtual_y) else math.nan
        area_log = math.log(max(area, 0.0) + 1.0) if math.isfinite(area) else math.nan

        theta_dot_x = 0.0
        theta_dot_y = 0.0
        area_dot = 0.0
        if prev_t is not None and math.isfinite(t):
            dt = max(t - prev_t, 1e-3)
            if all(math.isfinite(v) for v in (theta_x, prev_theta_x)):
                theta_dot_x = (theta_x - prev_theta_x) / dt
            if all(math.isfinite(v) for v in (theta_y, prev_theta_y)):
                theta_dot_y = (theta_y - prev_theta_y) / dt
            if all(math.isfinite(v) for v in (area_log, prev_area_log)):
                area_dot = (area_log - prev_area_log) / dt

        drone_x = to_float(row.get("drone_x", ""))
        drone_y = to_float(row.get("drone_y", ""))
        drone_z = to_float(row.get("drone_z", ""))
        drone_vx = to_float(row.get("drone_vx", ""))
        drone_vy = to_float(row.get("drone_vy", ""))
        drone_vz = to_float(row.get("drone_vz", ""))
        if not all(math.isfinite(v) for v in (drone_vx, drone_vy, drone_vz)):
            drone_vx, drone_vy, drone_vz = derive_velocity(t, (drone_x, drone_y, drone_z), prev_drone_t, prev_drone_pos)

        drone_roll = finite_or_default(to_float(row.get("drone_roll", "")), 0.0)
        drone_pitch = finite_or_default(to_float(row.get("drone_pitch", "")), 0.0)
        drone_yaw = finite_or_default(to_float(row.get("drone_yaw", "")), 0.0)

        values.append(
            [
                theta_x,
                theta_y,
                theta_dot_x,
                theta_dot_y,
                area_log,
                area_dot,
                virtual_theta_x,
                virtual_theta_y,
                to_float(row.get("cmd_vx", "")),
                to_float(row.get("cmd_vy", "")),
                to_float(row.get("cmd_vz", "")),
                to_float(row.get("cmd_yaw_rate", "")),
                drone_vx,
                drone_vy,
                drone_vz,
                drone_z,
                drone_roll,
                drone_pitch,
                drone_yaw,
            ]
        )

        if math.isfinite(t):
            prev_t = t
            prev_theta_x = theta_x
            prev_theta_y = theta_y
            prev_area_log = area_log
            if all(math.isfinite(v) for v in (drone_x, drone_y, drone_z)):
                prev_drone_t = t
                prev_drone_pos = (drone_x, drone_y, drone_z)

    return np.array(values, dtype=float)


def derive_velocity(t, pos, prev_t, prev_pos):
    if (
        prev_t is None
        or prev_pos is None
        or not math.isfinite(t)
        or not all(math.isfinite(v) for v in pos)
        or not all(math.isfinite(v) for v in prev_pos)
    ):
        return 0.0, 0.0, 0.0
    dt = max(t - prev_t, 1e-3)
    return (
        (pos[0] - prev_pos[0]) / dt,
        (pos[1] - prev_pos[1]) / dt,
        (pos[2] - prev_pos[2]) / dt,
    )


def finite_or_default(value, default):
    return value if math.isfinite(value) else default


def row_is_valid(row, feature, min_area):
    if row.get("state", "") in BAD_STATES:
        return False
    if row.get("captured", "0") == "1":
        return False
    if not np.all(np.isfinite(feature)):
        return False
    return math.exp(feature[4]) - 1.0 >= min_area


def risk_label(rows, features, start, label, lookahead):
    end = min(len(rows), label + lookahead + 1)
    for index in range(label, end):
        state = rows[index].get("state", "")
        if state in BAD_STATES:
            return 1.0
        theta_x = abs(features[index, 0])
        theta_y = abs(features[index, 1])
        theta_rate = math.hypot(features[index, 2], features[index, 3])
        if theta_x > 0.50 or theta_y > 0.42 or theta_rate > 1.2:
            return 1.0

    area_now = features[start - 1, 4]
    area_future = features[label, 4]
    if math.isfinite(area_now) and math.isfinite(area_future) and area_future < area_now - 0.65:
        return 1.0
    return 0.0


def target_point(row):
    t = to_float(row.get("target_time_sec", ""))
    x = to_float(row.get("target_x", ""))
    y = to_float(row.get("target_y", ""))
    z = to_float(row.get("target_z", ""))
    if not all(math.isfinite(v) for v in (t, x, y, z)):
        return None
    return t, x, y, z


def target_motion_stats(points):
    speeds = []
    jumps = 0
    repeats = 0
    for prev, cur in zip(points, points[1:]):
        dt = cur[0] - prev[0]
        dist = math.dist(prev[1:], cur[1:])
        if dist < 1e-6:
            repeats += 1
        if dt > 1e-6:
            speed = dist / dt
            speeds.append(speed)
            if dist > 0.5:
                jumps += 1
    return speeds, jumps, repeats


def write_manifest(rows, path):
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "index",
        "seed",
        "group",
        "weight",
        "reasons",
        "rows",
        "valid_visual_rows",
        "rough_samples",
        "captured",
        "lost_rows",
        "dist_start",
        "dist_min",
        "dist_end",
        "area_max",
        "speed_median",
        "speed_max",
        "target_jumps",
        "target_repeats",
    ]
    with open(output, "w", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def dataset_summary(groups, risk, weight):
    summary = {
        "samples": int(len(groups)),
        "risk_positive": int(np.sum(risk > 0.5)),
        "weighted_samples": float(np.sum(weight)),
        "groups": {},
    }
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        summary["groups"][group] = {
            "samples": int(np.sum(mask)),
            "risk_positive": int(np.sum(risk[mask] > 0.5)),
            "weighted_samples": float(np.sum(weight[mask])),
        }
    return summary


def to_float(value):
    if value in (None, ""):
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def median(values):
    if not values:
        return math.nan
    values = sorted(values)
    return values[len(values) // 2]


if __name__ == "__main__":
    main()
