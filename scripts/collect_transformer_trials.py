#!/usr/bin/env python3

import argparse
import re
import random
import subprocess
from pathlib import Path


RUNNER = Path.home() / "drone_ws" / "scripts" / "run_tuning_trial.py"
LOG_DIR = Path.home() / "drone_ws" / "logs"
TRAIN_DATA_DIR = Path.home() / "drone_ws" / "data" / "transformer_train_random_v2"
NODE_LOG_DIR = Path.home() / "drone_ws" / "tuning_node_logs"


def main():
    parser = argparse.ArgumentParser(description="Collect randomized CSV trials for Transformer training.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Trial index to start from. Defaults to the next unused index for the name prefix.",
    )
    parser.add_argument("--name-prefix", default="transformer_random")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--capture-radius", type=float, default=0.8)
    parser.add_argument("--controller-profile", default="los_rate_predictive")
    parser.add_argument("--start-x", type=float, default=8.0)
    parser.add_argument("--start-y", type=float, default=1.0)
    parser.add_argument("--start-z", type=float, default=4.0)
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--target-dt", type=float, default=0.02)
    parser.add_argument("--speed-min", type=float, default=2.0)
    parser.add_argument("--speed-max", type=float, default=5.0)
    parser.add_argument("--lateral-amp-min", type=float, default=1.0)
    parser.add_argument("--lateral-amp-max", type=float, default=3.0)
    parser.add_argument("--z-amp-min", type=float, default=0.2)
    parser.add_argument("--z-amp-max", type=float, default=0.8)
    parser.add_argument("--turn-amp-min-deg", type=float, default=0.0)
    parser.add_argument("--turn-amp-max-deg", type=float, default=0.0)
    parser.add_argument("--period-min", type=float, default=3.0)
    parser.add_argument("--period-max", type=float, default=9.0)
    parser.add_argument("--random-terms", type=int, default=4)
    parser.add_argument("--max-start-xy", type=float, default=30.0)
    parser.add_argument(
        "--with-transformer-predictor",
        action="store_true",
        help="Start the online LOS Transformer predictor for every collected trial.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--wait-for-enter",
        action="store_true",
        help="Pause before every trial so the operator can reset/take off/hold the drone.",
    )
    args = parser.parse_args()

    start_index = args.start_index
    if start_index is None:
        start_index = next_trial_index(args.name_prefix)
        print(f"Auto start index: {start_index:03d}")

    for offset in range(args.count):
        index = start_index + offset
        seed = args.seed_start + index
        rng = random.Random(seed)
        command = build_command(args, index, seed, rng)
        print(" ".join(command))
        if args.dry_run:
            continue
        if args.wait_for_enter:
            input(f"Ready for trial {index + 1}/{args.count} seed={seed}. Press Enter to start...")
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(f"trial failed with code {result.returncode}: seed={seed}")


def next_trial_index(name_prefix):
    pattern = re.compile(rf"tune_{re.escape(name_prefix)}_(\d+)_seed_\d+_")
    max_index = -1
    for directory in (LOG_DIR, TRAIN_DATA_DIR):
        if not directory.exists():
            continue
        for path in directory.glob("*.csv"):
            match = pattern.search(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def build_command(args, index, seed, rng):
    speed = rng.uniform(args.speed_min, args.speed_max)
    lateral_amp = rng.uniform(args.lateral_amp_min, args.lateral_amp_max)
    z_amp = rng.uniform(args.z_amp_min, args.z_amp_max)
    turn_amp = rng.uniform(args.turn_amp_min_deg, args.turn_amp_max_deg)

    command = [
        str(RUNNER),
        "--name",
        f"{args.name_prefix}_{index:03d}_seed_{seed}",
        "--timeout",
        f"{args.timeout:.1f}",
        "--capture-radius",
        f"{args.capture_radius:.3f}",
        "--controller-profile",
        args.controller_profile,
        "--target-mode",
        "random_maneuver",
        "--target-speed",
        f"{speed:.3f}",
        "--start-x",
        f"{args.start_x:.3f}",
        "--start-y",
        f"{args.start_y:.3f}",
        "--start-z",
        f"{args.start_z:.3f}",
        "--heading-deg",
        f"{args.heading_deg:.3f}",
        "--target-dt",
        f"{args.target_dt:.3f}",
        "--seed",
        str(seed),
        "--random-speed-amp",
        f"{max(0.2, speed * 0.25):.3f}",
        "--random-lateral-amp",
        f"{lateral_amp:.3f}",
        "--random-z-amp",
        f"{z_amp:.3f}",
        "--random-turn-amp-deg",
        f"{turn_amp:.3f}",
        "--random-min-period",
        f"{args.period_min:.3f}",
        "--random-max-period",
        f"{args.period_max:.3f}",
        "--random-terms",
        str(args.random_terms),
        "--max-start-xy",
        f"{args.max_start_xy:.3f}",
    ]
    if args.with_transformer_predictor:
        command.append("--with-transformer-predictor")
    return command


if __name__ == "__main__":
    main()
