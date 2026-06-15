#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path


BAD_STATES = {"UNKNOWN", "SEARCH", "LOST"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate online LOS Transformer predictions recorded in an experiment CSV.")
    parser.add_argument("csv_path", help="Experiment CSV with pred_theta_x/pred_theta_y/risk_probability columns.")
    parser.add_argument("--horizon-sec", type=float, default=0.30)
    parser.add_argument("--image-width", type=float, default=1280.0)
    parser.add_argument("--image-height", type=float, default=960.0)
    parser.add_argument("--horizontal-fov-rad", type=float, default=1.74)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--edge-theta-x-rad", type=float, default=0.50)
    parser.add_argument("--edge-theta-y-rad", type=float, default=0.42)
    parser.add_argument(
        "--include-after-capture",
        action="store_true",
        help="Include rows after virtual capture. Default evaluates only pre-capture guidance behavior.",
    )
    args = parser.parse_args()

    rows = read_rows(Path(args.csv_path))
    if not rows:
        raise SystemExit("No rows found.")

    fx, fy = camera_focal_lengths(args.image_width, args.image_height, args.horizontal_fov_rad)
    samples = build_samples(rows, args.horizon_sec, fx, fy, include_after_capture=args.include_after_capture)
    if not samples:
        raise SystemExit("No valid Transformer prediction samples found in this CSV.")

    theta_x_errors = [abs(sample["pred_theta_x"] - sample["future_theta_x"]) for sample in samples]
    theta_y_errors = [abs(sample["pred_theta_y"] - sample["future_theta_y"]) for sample in samples]
    los_errors = [math.hypot(sample["pred_theta_x"] - sample["future_theta_x"], sample["pred_theta_y"] - sample["future_theta_y"]) for sample in samples]
    hold_los_errors = [
        math.hypot(sample["theta_x"] - sample["future_theta_x"], sample["theta_y"] - sample["future_theta_y"])
        for sample in samples
    ]
    rate_los_errors = [
        math.hypot(
            sample["rate_pred_theta_x"] - sample["future_theta_x"],
            sample["rate_pred_theta_y"] - sample["future_theta_y"],
        )
        for sample in samples
    ]
    area_errors = [abs(sample["pred_area_log"] - sample["future_area_log"]) for sample in samples if math.isfinite(sample["pred_area_log"]) and math.isfinite(sample["future_area_log"])]
    risk_labels = [future_is_risky(sample, args.edge_theta_x_rad, args.edge_theta_y_rad) for sample in samples]
    risk_predictions = [sample["risk_probability"] >= args.risk_threshold for sample in samples]

    print(f"file: {Path(args.csv_path).name}")
    print(f"samples: {len(samples)}")
    print(f"horizon_sec: {args.horizon_sec:.2f}")
    print(f"theta_x_mae_rad: {mean(theta_x_errors):.4f}")
    print(f"theta_y_mae_rad: {mean(theta_y_errors):.4f}")
    print(f"los_norm_mae_rad: {mean(los_errors):.4f}")
    print(f"hold_baseline_los_norm_mae_rad: {mean(hold_los_errors):.4f}")
    print(f"rate_baseline_los_norm_mae_rad: {mean(rate_los_errors):.4f}")
    print(f"better_than_hold_ratio: {better_ratio(los_errors, hold_los_errors):.3f}")
    print(f"better_than_rate_ratio: {better_ratio(los_errors, rate_los_errors):.3f}")
    if area_errors:
        print(f"area_log_mae: {mean(area_errors):.4f}")
    print_risk_metrics(risk_labels, risk_predictions)

    high_risk = [sample for sample in samples if sample["risk_probability"] >= args.risk_threshold]
    if high_risk:
        first = high_risk[0]
        print(
            "first_high_risk: "
            f"t={first['time_sec']:.2f}s risk={first['risk_probability']:.2f} "
            f"future_state={first['future_state']}"
        )


def read_rows(path):
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def camera_focal_lengths(width, height, horizontal_fov):
    fx = width / (2.0 * math.tan(horizontal_fov / 2.0))
    vertical_fov = 2.0 * math.atan(math.tan(horizontal_fov / 2.0) * height / width)
    fy = height / (2.0 * math.tan(vertical_fov / 2.0))
    return fx, fy


def build_samples(rows, horizon_sec, fx, fy, include_after_capture=False):
    parsed = add_rates([parse_row(row, fx, fy) for row in rows])
    samples = []
    future_index = 0
    for index, row in enumerate(parsed):
        if not include_after_capture and row["captured"]:
            break
        if not prediction_is_valid(row):
            continue
        target_time = row["time_sec"] + horizon_sec
        future_index = max(future_index, index)
        while future_index + 1 < len(parsed) and parsed[future_index]["time_sec"] < target_time:
            future_index += 1
        future = parsed[future_index]
        if not actual_is_valid(future):
            continue
        sample = dict(row)
        sample["future_theta_x"] = future["theta_x"]
        sample["future_theta_y"] = future["theta_y"]
        sample["future_area_log"] = future["area_log"]
        sample["future_state"] = future["state"]
        sample["rate_pred_theta_x"] = row["theta_x"] + row["theta_dot_x"] * horizon_sec
        sample["rate_pred_theta_y"] = row["theta_y"] + row["theta_dot_y"] * horizon_sec
        samples.append(sample)
    return samples


def parse_row(row, fx, fy):
    error_x = to_float(row.get("error_x_px"))
    error_y = to_float(row.get("error_y_px"))
    area = to_float(row.get("area_px"))
    return {
        "time_sec": to_float(row.get("time_sec")),
        "state": row.get("state", ""),
        "captured": row.get("captured", "0") == "1",
        "theta_x": math.atan2(error_x, fx) if math.isfinite(error_x) else math.nan,
        "theta_y": math.atan2(error_y, fy) if math.isfinite(error_y) else math.nan,
        "area_log": math.log(max(area, 0.0) + 1.0) if math.isfinite(area) else math.nan,
        "pred_theta_x": to_float(row.get("pred_theta_x")),
        "pred_theta_y": to_float(row.get("pred_theta_y")),
        "pred_area_log": to_float(row.get("pred_area_log")),
        "risk_probability": to_float(row.get("risk_probability")),
    }


def add_rates(rows):
    previous = None
    for row in rows:
        row["theta_dot_x"] = 0.0
        row["theta_dot_y"] = 0.0
        row["area_dot"] = 0.0
        if previous is not None and math.isfinite(row["time_sec"]) and math.isfinite(previous["time_sec"]):
            dt = max(row["time_sec"] - previous["time_sec"], 1e-3)
            if math.isfinite(row["theta_x"]) and math.isfinite(previous["theta_x"]):
                row["theta_dot_x"] = (row["theta_x"] - previous["theta_x"]) / dt
            if math.isfinite(row["theta_y"]) and math.isfinite(previous["theta_y"]):
                row["theta_dot_y"] = (row["theta_y"] - previous["theta_y"]) / dt
            if math.isfinite(row["area_log"]) and math.isfinite(previous["area_log"]):
                row["area_dot"] = (row["area_log"] - previous["area_log"]) / dt
        if actual_is_valid(row):
            previous = row
    return rows


def prediction_is_valid(row):
    return (
        math.isfinite(row["time_sec"])
        and math.isfinite(row["pred_theta_x"])
        and math.isfinite(row["pred_theta_y"])
        and math.isfinite(row["risk_probability"])
    )


def actual_is_valid(row):
    return math.isfinite(row["theta_x"]) and math.isfinite(row["theta_y"]) and math.isfinite(row["area_log"])


def future_is_risky(sample, edge_x, edge_y):
    if sample["future_state"] in BAD_STATES:
        return True
    return abs(sample["future_theta_x"]) >= edge_x or abs(sample["future_theta_y"]) >= edge_y


def print_risk_metrics(labels, predictions):
    total = len(labels)
    tp = sum(1 for label, pred in zip(labels, predictions) if label and pred)
    tn = sum(1 for label, pred in zip(labels, predictions) if not label and not pred)
    fp = sum(1 for label, pred in zip(labels, predictions) if not label and pred)
    fn = sum(1 for label, pred in zip(labels, predictions) if label and not pred)
    accuracy = (tp + tn) / total if total else math.nan
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    print(f"risk_positive_rows: {sum(labels)}")
    print(f"risk_accuracy: {accuracy:.3f}")
    print(f"risk_precision: {precision:.3f}" if math.isfinite(precision) else "risk_precision: nan")
    print(f"risk_recall: {recall:.3f}" if math.isfinite(recall) else "risk_recall: nan")


def mean(values):
    return sum(values) / len(values) if values else math.nan


def better_ratio(errors, baseline_errors):
    if not errors:
        return math.nan
    return sum(1 for error, baseline in zip(errors, baseline_errors) if error < baseline) / len(errors)


def to_float(value):
    try:
        return float(value) if value not in ("", None) else math.nan
    except ValueError:
        return math.nan


if __name__ == "__main__":
    main()
