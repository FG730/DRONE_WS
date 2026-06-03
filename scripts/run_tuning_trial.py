#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import signal
import subprocess
import time
from pathlib import Path


ROS_SETUP = "source /opt/ros/humble/setup.bash && source ~/drone_ws/install/setup.bash"
CAMERA_TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"
LOG_DIR = Path.home() / "drone_ws" / "logs"
NODE_LOG_DIR = Path.home() / "drone_ws" / "tuning_node_logs"


CONTROLLER_PARAMS = [
    "-p base_forward_speed:=5.2",
    "-p max_forward_speed:=6.0",
    "-p min_forward_scale:=0.30",
    "-p kp_lateral:=0.0070",
    "-p kd_lateral:=0.0015",
    "-p kp_vertical:=-0.0100",
    "-p kd_vertical:=-0.0010",
    "-p max_lateral_speed:=1.6",
    "-p max_vertical_speed:=1.8",
    "-p max_yaw_rate:=0.25",
    "-p pitch_pixel_gain:=1.0",
    "-p roll_pixel_gain:=0.70",
    "-p pitch_sign:=1.0",
    "-p roll_sign:=1.0",
    "-p vertical_error_alpha:=0.45",
    "-p lateral_error_alpha:=0.25",
    "-p vertical_deadband_px:=3.0",
    "-p lateral_deadband_px:=8.0",
    "-p max_vertical_accel:=2.2",
    "-p max_lateral_accel:=1.2",
    "-p enable_vertical_area_schedule:=true",
    "-p far_area_px:=2500.0",
    "-p near_area_px:=18000.0",
    "-p near_vertical_gain_scale:=0.45",
    "-p near_vertical_speed_scale:=0.45",
    "-p near_vertical_accel_scale:=0.45",
    "-p enable_terminal_dash:=true",
    "-p terminal_area_px:=12000.0",
    "-p terminal_min_forward_scale:=0.90",
    "-p terminal_lateral_scale:=0.15",
    "-p terminal_vertical_scale:=0.12",
    "-p terminal_yaw_scale:=0.15",
    "-p max_virtual_offset_px:=260.0",
]


ARC_FAST_CONTROLLER_PARAMS = [
    "-p base_forward_speed:=7.2",
    "-p max_forward_speed:=8.0",
    "-p min_forward_scale:=0.72",
    "-p kp_lateral:=0.0100",
    "-p kd_lateral:=0.0022",
    "-p kp_vertical:=-0.0100",
    "-p kd_vertical:=-0.0010",
    "-p max_lateral_speed:=2.4",
    "-p max_vertical_speed:=1.8",
    "-p max_yaw_rate:=0.35",
    "-p center_x_px:=45.0",
    "-p center_y_px:=70.0",
    "-p slow_x_px:=520.0",
    "-p slow_y_px:=320.0",
    "-p rate_slow_px_s:=1800.0",
    "-p pitch_pixel_gain:=1.0",
    "-p roll_pixel_gain:=0.70",
    "-p pitch_sign:=1.0",
    "-p roll_sign:=1.0",
    "-p vertical_error_alpha:=0.45",
    "-p lateral_error_alpha:=0.35",
    "-p vertical_deadband_px:=3.0",
    "-p lateral_deadband_px:=6.0",
    "-p max_vertical_accel:=2.2",
    "-p max_lateral_accel:=2.2",
    "-p enable_vertical_area_schedule:=true",
    "-p far_area_px:=2500.0",
    "-p near_area_px:=18000.0",
    "-p near_vertical_gain_scale:=0.45",
    "-p near_vertical_speed_scale:=0.45",
    "-p near_vertical_accel_scale:=0.45",
    "-p enable_terminal_dash:=true",
    "-p terminal_area_px:=10000.0",
    "-p terminal_min_forward_scale:=0.95",
    "-p terminal_lateral_scale:=0.20",
    "-p terminal_vertical_scale:=0.12",
    "-p terminal_yaw_scale:=0.20",
    "-p max_virtual_offset_px:=260.0",
]


def main():
    parser = argparse.ArgumentParser(description="Run one automated visual tracking tuning trial.")
    parser.add_argument("--name", default=time.strftime("trial_%H%M%S"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--pre-target-delay", type=float, default=5.0)
    parser.add_argument("--capture-radius", type=float, default=1.2)
    parser.add_argument("--target-mode", choices=["straight", "arc", "s_curve"], default="straight")
    parser.add_argument("--controller-profile", choices=["trial7", "arc_fast"], default="trial7")
    parser.add_argument("--target-speed", type=float, default=3.0)
    parser.add_argument("--start-x", type=float, default=8.0)
    parser.add_argument("--start-y", type=float, default=2.0)
    parser.add_argument("--start-z", type=float, default=4.0)
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--target-dt", type=float, default=0.02)
    parser.add_argument("--arc-radius", type=float, default=60.0)
    parser.add_argument("--arc-direction", choices=["left", "right"], default="left")
    parser.add_argument("--s-amp-y", type=float, default=1.5)
    parser.add_argument("--s-period", type=float, default=6.0)
    parser.add_argument("--z-amp", type=float, default=0.25)
    parser.add_argument("--z-period", type=float, default=9.0)
    parser.add_argument("--max-start-z", type=float, default=4.5)
    parser.add_argument("--min-start-z", type=float, default=2.0)
    parser.add_argument("--max-start-xy", type=float, default=30.0)
    parser.add_argument("--no-bridge", action="store_true")
    parser.add_argument("--no-color-tracker", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    NODE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"tune_{args.name}"
    processes = []

    try:
        if not start_pose_is_reasonable(args.max_start_xy, args.min_start_z, args.max_start_z):
            print("invalid trial: vehicle is not near the expected start pose. Reset PX4/Gazebo or reposition before tuning.")
            return

        if not args.no_bridge:
            processes.append(
                start_process(
                    "bridge",
                    f"{ROS_SETUP} && ros2 run ros_gz_bridge parameter_bridge "
                    f"{CAMERA_TOPIC}@sensor_msgs/msg/Image@gz.msgs.Image",
                    prefix,
                )
            )
            time.sleep(1.0)

        if not args.no_color_tracker:
            processes.append(
                start_process(
                    "color_tracker",
                    f"{ROS_SETUP} && ros2 run vision_tracking color_tracker",
                    prefix,
                )
            )
            time.sleep(1.0)

        processes.append(
            start_process(
                "logger",
                f"{ROS_SETUP} && ros2 run vision_tracking experiment_logger --ros-args "
                f"-p capture_radius:={args.capture_radius} "
                "-p capture_hold_sec:=0.0 "
                "-p require_fresh_target_pose:=true "
                "-p target_pose_stale_sec:=1.0 "
                "-p swap_px4_xy_to_gazebo:=true "
                f"-p file_prefix:={prefix}",
                prefix,
            )
        )
        time.sleep(1.0)

        processes.append(
            start_process(
                "controller",
                f"{ROS_SETUP} && ros2 run vision_tracking attitude_pn_bearing_servo --ros-args "
                + " ".join(controller_params(args.controller_profile)),
                prefix,
            )
        )
        time.sleep(1.0)

        processes.append(
            start_process(
                "offboard",
                f"{ROS_SETUP} && ros2 run vision_tracking px4_visual_offboard --ros-args "
                f"-p max_forward_speed:={offboard_limit(args.controller_profile, 'forward')} "
                f"-p max_lateral_speed:={offboard_limit(args.controller_profile, 'lateral')} "
                "-p max_vertical_speed:=1.4 "
                f"-p max_yaw_rate:={offboard_limit(args.controller_profile, 'yaw')} "
                "-p forward_sign:=1.0 "
                "-p lateral_sign:=1.0 "
                "-p mode_retry_sec:=1.0",
                prefix,
            )
        )

        reset_pose_file()
        time.sleep(args.pre_target_delay)

        processes.append(
            start_process(
                "target",
                f"{ROS_SETUP} && ~/drone_ws/scripts/move_target_realistic.py "
                f"--mode {args.target_mode} "
                f"--speed {args.target_speed} "
                f"--start-x {args.start_x} "
                f"--start-y {args.start_y} "
                f"--start-z {args.start_z} "
                f"--heading-deg {args.heading_deg} "
                f"--dt {args.target_dt} "
                f"--arc-radius {args.arc_radius} "
                f"--arc-direction {args.arc_direction} "
                f"--s-amp-y {args.s_amp_y} "
                f"--s-period {args.s_period} "
                f"--z-amp {args.z_amp} "
                f"--z-period {args.z_period}",
                prefix,
            )
        )

        if not wait_for_fresh_target_pose(timeout=5.0):
            print("invalid trial: target pose file was not freshly written")
            return

        csv_path = wait_for_csv(prefix, timeout=10.0)
        print(f"CSV: {csv_path}")
        result = wait_for_trial_result(csv_path, args.timeout)
        print(result)

    finally:
        stop_processes(processes)

    score_latest()


def start_process(name, command, prefix):
    log_path = NODE_LOG_DIR / f"{prefix}_{name}.log"
    log_file = open(log_path, "w")
    print(f"starting {name}, log={log_path}")
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", command],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return name, proc, log_file


def controller_params(profile):
    if profile == "arc_fast":
        return ARC_FAST_CONTROLLER_PARAMS
    return CONTROLLER_PARAMS


def offboard_limit(profile, axis):
    if profile == "arc_fast":
        return {
            "forward": 8.0,
            "lateral": 2.4,
            "yaw": 0.35,
        }[axis]
    return {
        "forward": 6.0,
        "lateral": 1.6,
        "yaw": 0.25,
    }[axis]


def stop_processes(processes):
    for name, proc, log_file in reversed(processes):
        if proc.poll() is None:
            print(f"stopping {name}")
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    deadline = time.time() + 5.0
    for name, proc, log_file in reversed(processes):
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            print(f"killing {name}")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        log_file.close()


def wait_for_csv(prefix, timeout):
    deadline = time.time() + timeout
    pattern = str(LOG_DIR / f"{prefix}_*.csv")
    while time.time() < deadline:
        matches = glob.glob(pattern)
        if matches:
            return Path(max(matches, key=os.path.getmtime))
        time.sleep(0.2)
    raise RuntimeError(f"No CSV created for prefix {prefix}")


def reset_pose_file():
    try:
        os.unlink("/tmp/red_target_pose.csv")
    except FileNotFoundError:
        pass


def wait_for_fresh_target_pose(timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open("/tmp/red_target_pose.csv") as pose_file:
                line = pose_file.readline().strip()
        except OSError:
            time.sleep(0.1)
            continue

        parts = line.split(",")
        if len(parts) == 4:
            try:
                pose_time = float(parts[0])
            except ValueError:
                pose_time = 0.0
            if time.time() - pose_time < 1.0:
                return True
        time.sleep(0.1)
    return False


def wait_for_trial_result(csv_path, timeout):
    start = time.time()
    last_target_time = 0.0
    while time.time() - start < timeout:
        rows = read_rows(csv_path)
        if rows:
            captured = [row for row in rows if row.get("captured") == "1"]
            if captured:
                return "captured"
            target_times = [to_float(row.get("target_time_sec", "")) for row in rows]
            target_times = [value for value in target_times if value == value]
            if target_times:
                last_target_time = max(target_times)
        time.sleep(0.5)
    return f"timeout target_time={last_target_time:.1f}s"


def read_rows(csv_path):
    try:
        with open(csv_path, newline="") as csv_file:
            return list(csv.DictReader(csv_file))
    except OSError:
        return []


def to_float(value):
    try:
        return float(value) if value not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def score_latest():
    command = f"{ROS_SETUP} && ros2 run vision_tracking score_experiment"
    subprocess.run(["/bin/bash", "-lc", command], check=False)


def start_pose_is_reasonable(max_xy, min_z, max_z):
    values = read_local_position("/fmu/out/vehicle_local_position_v1")
    if not values:
        values = read_local_position("/fmu/out/vehicle_local_position")
    if not values:
        print("invalid trial: could not read vehicle_local_position")
        return False

    if values.get("xy_valid") is False or values.get("z_valid") is False:
        print(f"invalid trial: local position invalid: {values}")
        return False

    px4_x = float(values.get("x", 9999.0))
    px4_y = float(values.get("y", 9999.0))
    gazebo_z = -float(values.get("z", -9999.0))
    horizontal = (px4_x * px4_x + px4_y * px4_y) ** 0.5
    print(f"start pose check: px4_xy=({px4_x:.2f},{px4_y:.2f}), gazebo_z={gazebo_z:.2f}, horizontal={horizontal:.2f}")
    return min_z <= gazebo_z <= max_z and horizontal <= max_xy


def read_local_position(topic):
    command = f"{ROS_SETUP} && timeout 5 ros2 topic echo --once {topic}"
    result = subprocess.run(
        ["/bin/bash", "-lc", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return parse_echo_mapping(result.stdout)


def parse_echo_mapping(text):
    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value.lower() == "true":
            values[key] = True
        elif raw_value.lower() == "false":
            values[key] = False
        else:
            try:
                values[key] = float(raw_value)
            except ValueError:
                pass
    return values


if __name__ == "__main__":
    main()
