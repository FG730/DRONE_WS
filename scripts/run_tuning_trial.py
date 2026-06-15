#!/usr/bin/env python3

import argparse
import csv
import glob
import math
import os
import signal
import subprocess
import time
from pathlib import Path


ROS_SETUP = "source /opt/ros/humble/setup.bash && source /home/wsy/drone_ws/install/setup.bash"
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


ARC_BALANCED_CONTROLLER_PARAMS = [
    "-p base_forward_speed:=7.0",
    "-p max_forward_speed:=8.0",
    "-p min_forward_scale:=0.48",
    "-p kp_lateral:=0.0120",
    "-p kd_lateral:=0.0025",
    "-p kp_vertical:=-0.0130",
    "-p kd_vertical:=-0.0013",
    "-p max_lateral_speed:=2.8",
    "-p max_vertical_speed:=2.3",
    "-p max_yaw_rate:=0.42",
    "-p center_x_px:=40.0",
    "-p center_y_px:=55.0",
    "-p slow_x_px:=360.0",
    "-p slow_y_px:=220.0",
    "-p rate_slow_px_s:=1500.0",
    "-p pitch_pixel_gain:=0.85",
    "-p roll_pixel_gain:=0.80",
    "-p pitch_sign:=1.0",
    "-p roll_sign:=1.0",
    "-p vertical_error_alpha:=0.55",
    "-p lateral_error_alpha:=0.45",
    "-p vertical_deadband_px:=2.0",
    "-p lateral_deadband_px:=5.0",
    "-p max_vertical_accel:=3.0",
    "-p max_lateral_accel:=3.0",
    "-p enable_vertical_area_schedule:=true",
    "-p far_area_px:=2500.0",
    "-p near_area_px:=22000.0",
    "-p near_vertical_gain_scale:=0.65",
    "-p near_vertical_speed_scale:=0.65",
    "-p near_vertical_accel_scale:=0.65",
    "-p enable_terminal_dash:=true",
    "-p terminal_area_px:=18000.0",
    "-p terminal_min_forward_scale:=0.72",
    "-p terminal_lateral_scale:=0.35",
    "-p terminal_vertical_scale:=0.35",
    "-p terminal_yaw_scale:=0.30",
    "-p max_virtual_offset_px:=220.0",
]


LOS_RATE_CONTROLLER_PARAMS = [
    "-p fixed_forward_speed:=9.0",
    "-p max_forward_speed:=10.0",
    "-p min_forward_scale:=0.35",
    "-p angle_slow_x_rad:=0.75",
    "-p angle_slow_y_rad:=0.65",
    "-p los_rate_slow_rad_s:=2.0",
    "-p angle_gain_lateral:=5.0",
    "-p angle_gain_vertical:=3.4",
    "-p navigation_gain_lateral:=1.2",
    "-p navigation_gain_vertical:=0.9",
    "-p yaw_angle_gain:=0.85",
    "-p yaw_rate_gain:=0.16",
    "-p max_lateral_speed:=3.2",
    "-p max_vertical_speed:=1.3",
    "-p max_yaw_rate:=0.45",
    "-p max_lateral_accel:=5.0",
    "-p max_vertical_accel:=1.6",
    "-p max_yaw_accel:=1.2",
    "-p angle_alpha:=0.55",
    "-p los_rate_alpha:=0.30",
    "-p angle_deadband_x_rad:=0.006",
    "-p angle_deadband_y_rad:=0.006",
    "-p enable_terminal_dash:=true",
    "-p close_area_px:=12000.0",
    "-p close_center_x_rad:=0.26",
    "-p close_center_y_rad:=0.22",
    "-p close_min_forward_scale:=0.45",
    "-p close_max_forward_scale:=0.85",
    "-p close_correction_scale:=1.05",
    "-p terminal_area_px:=22000.0",
    "-p terminal_center_x_rad:=0.16",
    "-p terminal_center_y_rad:=0.12",
    "-p terminal_min_forward_scale:=0.95",
    "-p terminal_offcenter_forward_scale:=0.22",
    "-p terminal_correction_scale:=0.80",
    "-p enable_keep_in_view:=true",
    "-p keep_in_view_begin_ratio:=0.55",
    "-p keep_in_view_full_ratio:=0.95",
    "-p edge_angle_x_rad:=0.56",
    "-p edge_angle_y_rad:=0.46",
    "-p edge_los_rate_rad_s:=1.15",
    "-p edge_forward_scale:=0.10",
    "-p edge_correction_scale:=1.55",
    "-p pitch_pixel_gain:=0.85",
    "-p roll_pixel_gain:=0.80",
    "-p pitch_sign:=1.0",
    "-p roll_sign:=1.0",
    "-p max_virtual_offset_px:=220.0",
]


LOS_RATE_PREDICTIVE_CONTROLLER_PARAMS = LOS_RATE_CONTROLLER_PARAMS + [
    "-p enable_predictive_terminal:=true",
    "-p predict_time_sec:=0.25",
    "-p predict_start_area_px:=15000.0",
    "-p predict_full_area_px:=36000.0",
    "-p predict_rate_trigger_rad_s:=0.75",
    "-p predict_rate_full_rad_s:=1.80",
    "-p predict_center_x_rad:=0.20",
    "-p predict_center_y_rad:=0.16",
    "-p predict_max_angle_rad:=0.78",
    "-p predict_forward_scale:=0.20",
    "-p predict_lateral_boost:=1.45",
    "-p predict_vertical_boost:=1.10",
    "-p predict_yaw_scale:=1.00",
    "-p edge_forward_scale:=0.25",
    "-p edge_correction_scale:=1.65",
    "-p max_lateral_speed:=4.5",
    "-p max_yaw_rate:=0.70",
    "-p max_lateral_accel:=7.0",
    "-p max_yaw_accel:=2.0",
]


def main():
    parser = argparse.ArgumentParser(description="Run one automated visual tracking tuning trial.")
    parser.add_argument("--name", default=time.strftime("trial_%H%M%S"))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--pre-target-delay", type=float, default=5.0)
    parser.add_argument("--post-capture-delay", type=float, default=5.0)
    parser.add_argument("--capture-radius", type=float, default=1.2)
    parser.add_argument("--target-mode", choices=["straight", "arc", "s_curve", "random_maneuver"], default="straight")
    parser.add_argument(
        "--controller-profile",
        choices=["trial7", "arc_fast", "arc_balanced", "los_rate", "los_rate_predictive"],
        default="trial7",
    )
    parser.add_argument("--target-speed", type=float, default=3.0)
    parser.add_argument("--start-x", type=float, default=8.0)
    parser.add_argument("--start-y", type=float, default=2.0)
    parser.add_argument("--start-z", type=float, default=4.0)
    parser.add_argument("--heading-deg", type=float, default=0.0)
    parser.add_argument("--target-dt", type=float, default=0.02)
    parser.add_argument("--target-start-hold-sec", type=float, default=0.0)
    parser.add_argument("--arc-radius", type=float, default=60.0)
    parser.add_argument("--arc-direction", choices=["left", "right"], default="left")
    parser.add_argument("--s-amp-y", type=float, default=1.5)
    parser.add_argument("--s-period", type=float, default=6.0)
    parser.add_argument("--z-amp", type=float, default=0.25)
    parser.add_argument("--z-period", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-speed-amp", type=float, default=0.8)
    parser.add_argument("--random-lateral-amp", type=float, default=2.0)
    parser.add_argument("--random-z-amp", type=float, default=0.6)
    parser.add_argument("--random-turn-amp-deg", type=float, default=18.0)
    parser.add_argument("--random-min-period", type=float, default=3.0)
    parser.add_argument("--random-max-period", type=float, default=9.0)
    parser.add_argument("--random-terms", type=int, default=4)
    parser.add_argument("--max-start-z", type=float, default=4.5)
    parser.add_argument("--min-start-z", type=float, default=2.0)
    parser.add_argument("--max-start-xy", type=float, default=30.0)
    parser.add_argument("--return-to-origin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return-altitude", type=float, default=3.0)
    parser.add_argument("--return-timeout", type=float, default=45.0)
    parser.add_argument("--no-bridge", action="store_true")
    parser.add_argument("--no-color-tracker", action="store_true")
    parser.add_argument(
        "--with-transformer-predictor",
        action="store_true",
        help="Start the trained LOS Transformer predictor during the trial. It runs in parallel and does not control the UAV.",
    )
    parser.add_argument(
        "--enable-transformer-guidance",
        action="store_true",
        help="Fuse Transformer-predicted LOS into the LOS-rate controller with a small feed-forward weight.",
    )
    parser.add_argument("--transformer-alpha", type=float, default=0.20)
    parser.add_argument("--transformer-alpha-x", type=float, default=0.0)
    parser.add_argument("--transformer-alpha-y", type=float, default=0.12)
    parser.add_argument("--transformer-start-area-px", type=float, default=8000.0)
    parser.add_argument("--transformer-stale-sec", type=float, default=0.30)
    parser.add_argument("--transformer-max-delta-rad", type=float, default=0.20)
    parser.add_argument("--transformer-guidance-mode", choices=["angle", "velocity"], default="angle")
    parser.add_argument("--transformer-horizon-sec", type=float, default=0.30)
    parser.add_argument("--transformer-velocity-gain-lateral", type=float, default=1.0)
    parser.add_argument("--transformer-velocity-gain-vertical", type=float, default=1.0)
    parser.add_argument("--no-transformer-consistency-gate", action="store_true")
    parser.add_argument("--predict-forward-scale", type=float, default=None)
    parser.add_argument("--predict-lateral-boost", type=float, default=None)
    parser.add_argument("--predict-vertical-boost", type=float, default=None)
    parser.add_argument("--predict-yaw-scale", type=float, default=None)
    parser.add_argument("--max-lateral-speed-override", type=float, default=None)
    parser.add_argument("--max-yaw-rate-override", type=float, default=None)
    parser.add_argument(
        "--transformer-model-path",
        default="~/drone_ws/models/los_delta_transformer_v2/best.pt",
        help="Model checkpoint used when --with-transformer-predictor is enabled.",
    )
    parser.add_argument(
        "--allow-lost-before-capture",
        action="store_true",
        help="Allow a trial to count even if the target is lost after first visual acquisition.",
    )
    args = parser.parse_args()
    if args.enable_transformer_guidance:
        args.with_transformer_predictor = True

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    NODE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"tune_{args.name}"
    processes = []
    ran_trial = False
    initial_heading = 0.0
    trial_start_wall_time = time.time()

    try:
        if not start_pose_is_reasonable(args.max_start_xy, args.min_start_z, args.max_start_z):
            print("invalid trial: vehicle is not near the expected start pose. Reset PX4/Gazebo or reposition before tuning.")
            return
        initial_heading = read_initial_heading()
        print(f"initial heading for return: {initial_heading:.3f} rad")

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
                    f"{ROS_SETUP} && ros2 run vision_tracking color_tracker --ros-args -p display:=false",
                    prefix,
                )
            )
            time.sleep(1.0)

        processes.append(
            start_process(
                "controller",
                f"{ROS_SETUP} && ros2 run vision_tracking {controller_node(args.controller_profile)} --ros-args "
                + " ".join(controller_params(args.controller_profile, args)),
                prefix,
            )
        )
        time.sleep(1.0)

        if args.with_transformer_predictor:
            processes.append(
                start_process(
                    "transformer_predictor",
                    f"{ROS_SETUP} && ros2 run vision_tracking los_transformer_predictor --ros-args "
                    f"-p model_path:={args.transformer_model_path}",
                    prefix,
                )
            )
            time.sleep(1.0)

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
                f"--start-hold-sec {args.target_start_hold_sec} "
                f"--arc-radius {args.arc_radius} "
                f"--arc-direction {args.arc_direction} "
                f"--s-amp-y {args.s_amp_y} "
                f"--s-period {args.s_period} "
                f"--z-amp {args.z_amp} "
                f"--z-period {args.z_period} "
                f"--seed {args.seed} "
                f"--random-speed-amp {args.random_speed_amp} "
                f"--random-lateral-amp {args.random_lateral_amp} "
                f"--random-z-amp {args.random_z_amp} "
                f"--random-turn-amp-deg {args.random_turn_amp_deg} "
                f"--random-min-period {args.random_min_period} "
                f"--random-max-period {args.random_max_period} "
                f"--random-terms {args.random_terms}",
                prefix,
            )
        )

        if not wait_for_fresh_target_pose(timeout=12.0):
            print("invalid trial: target pose file was not freshly written")
            return

        processes.append(
            start_process(
                "logger",
                f"{ROS_SETUP} && ros2 run vision_tracking experiment_logger --ros-args "
                f"-p capture_radius:={args.capture_radius} "
                "-p capture_hold_sec:=0.0 "
                "-p require_fresh_target_pose:=true "
                "-p target_pose_stale_sec:=2.5 "
                "-p swap_px4_xy_to_gazebo:=true "
                f"-p file_prefix:={prefix}",
                prefix,
            )
        )
        time.sleep(0.3)

        processes.append(
            start_process(
                "offboard",
                f"{ROS_SETUP} && ros2 run vision_tracking px4_visual_offboard --ros-args "
                f"-p max_forward_speed:={offboard_limit(args.controller_profile, 'forward')} "
                f"-p max_lateral_speed:={offboard_lateral_limit(args)} "
                f"-p max_vertical_speed:={offboard_limit(args.controller_profile, 'vertical')} "
                f"-p max_yaw_rate:={offboard_yaw_limit(args)} "
                "-p forward_sign:=1.0 "
                "-p lateral_sign:=1.0 "
                "-p mode_retry_sec:=1.0 "
                "-p hold_after_capture:=true "
                "-p capture_hold_mode_delay_sec:=2.0",
                prefix,
            )
        )

        csv_path = wait_for_csv(prefix, timeout=10.0, newer_than=trial_start_wall_time)
        ran_trial = True
        print(f"CSV: {csv_path}")
        result = wait_for_trial_result(
            csv_path,
            args.timeout,
            strict_no_lost=not args.allow_lost_before_capture,
        )
        print(result)
        if result == "captured" and args.post_capture_delay > 0.0:
            print(f"post-capture zero-velocity hold for {args.post_capture_delay:.1f}s")
            time.sleep(args.post_capture_delay)

    finally:
        stop_processes(processes)
        cleanup_target()
        if ran_trial and args.return_to_origin:
            return_to_origin(args, prefix, initial_heading)

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


def controller_params(profile, args=None):
    if profile == "los_rate_predictive":
        params = list(LOS_RATE_PREDICTIVE_CONTROLLER_PARAMS)
    elif profile == "los_rate":
        params = list(LOS_RATE_CONTROLLER_PARAMS)
    elif profile == "arc_balanced":
        params = list(ARC_BALANCED_CONTROLLER_PARAMS)
    elif profile == "arc_fast":
        params = list(ARC_FAST_CONTROLLER_PARAMS)
    else:
        params = list(CONTROLLER_PARAMS)

    if args is not None and args.enable_transformer_guidance and profile in ("los_rate", "los_rate_predictive"):
        params.extend(
            [
                "-p enable_transformer_guidance:=true",
                f"-p transformer_alpha:={args.transformer_alpha}",
                f"-p transformer_alpha_x:={args.transformer_alpha_x}",
                f"-p transformer_alpha_y:={args.transformer_alpha_y}",
                f"-p transformer_start_area_px:={args.transformer_start_area_px}",
                f"-p transformer_stale_sec:={args.transformer_stale_sec}",
                f"-p transformer_max_delta_rad:={args.transformer_max_delta_rad}",
                f"-p transformer_guidance_mode:={args.transformer_guidance_mode}",
                f"-p transformer_horizon_sec:={args.transformer_horizon_sec}",
                f"-p transformer_velocity_gain_lateral:={args.transformer_velocity_gain_lateral}",
                f"-p transformer_velocity_gain_vertical:={args.transformer_velocity_gain_vertical}",
                f"-p transformer_consistency_gate:={str(not args.no_transformer_consistency_gate).lower()}",
            ]
        )
    if args is not None and profile in ("los_rate", "los_rate_predictive"):
        if args.predict_forward_scale is not None:
            params.append(f"-p predict_forward_scale:={args.predict_forward_scale}")
        if args.predict_lateral_boost is not None:
            params.append(f"-p predict_lateral_boost:={args.predict_lateral_boost}")
        if args.predict_vertical_boost is not None:
            params.append(f"-p predict_vertical_boost:={args.predict_vertical_boost}")
        if args.predict_yaw_scale is not None:
            params.append(f"-p predict_yaw_scale:={args.predict_yaw_scale}")
        if args.max_lateral_speed_override is not None:
            params.append(f"-p max_lateral_speed:={args.max_lateral_speed_override}")
        if args.max_yaw_rate_override is not None:
            params.append(f"-p max_yaw_rate:={args.max_yaw_rate_override}")
    return params


def controller_node(profile):
    if profile in ("los_rate", "los_rate_predictive"):
        return "los_rate_bearing_servo"
    return "attitude_pn_bearing_servo"


def offboard_limit(profile, axis):
    if profile in ("los_rate", "los_rate_predictive"):
        limits = {
            "forward": 10.0,
            "lateral": 3.2,
            "vertical": 2.4,
            "yaw": 0.45,
        }
        if profile == "los_rate_predictive":
            limits["lateral"] = 4.5
            limits["yaw"] = 0.70
        return limits[axis]
    if profile == "arc_fast":
        return {
            "forward": 8.0,
            "lateral": 2.4,
            "vertical": 1.4,
            "yaw": 0.35,
        }[axis]
    if profile == "arc_balanced":
        return {
            "forward": 8.0,
            "lateral": 2.8,
            "vertical": 2.2,
            "yaw": 0.42,
        }[axis]
    return {
        "forward": 6.0,
        "lateral": 1.6,
        "vertical": 1.4,
        "yaw": 0.25,
    }[axis]


def offboard_lateral_limit(args):
    if args.max_lateral_speed_override is not None:
        return args.max_lateral_speed_override
    return offboard_limit(args.controller_profile, "lateral")


def offboard_yaw_limit(args):
    if args.max_yaw_rate_override is not None:
        return args.max_yaw_rate_override
    return offboard_limit(args.controller_profile, "yaw")


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


def wait_for_csv(prefix, timeout, newer_than=0.0):
    deadline = time.time() + timeout
    pattern = str(LOG_DIR / f"{prefix}_*.csv")
    while time.time() < deadline:
        matches = [
            path
            for path in glob.glob(pattern)
            if os.path.getmtime(path) >= newer_than
        ]
        if matches:
            return Path(max(matches, key=os.path.getmtime))
        time.sleep(0.2)
    raise RuntimeError(f"No CSV created for prefix {prefix}")


def reset_pose_file():
    try:
        os.unlink("/tmp/red_target_pose.csv")
    except FileNotFoundError:
        pass


def cleanup_target():
    remove_command = [
        "gz",
        "service",
        "-s",
        "/world/default/remove",
        "--reqtype",
        "gz.msgs.Entity",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "2000",
        "--req",
        'name: "red_target" type: MODEL',
    ]
    subprocess.run(remove_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    reset_pose_file()


def return_to_origin(args, prefix, initial_heading):
    log_path = NODE_LOG_DIR / f"{prefix}_return_to_origin.log"
    command = (
        f"{ROS_SETUP} && ros2 run vision_tracking px4_return_to_origin --ros-args "
        "-p target_x:=0.0 "
        "-p target_y:=0.0 "
        f"-p target_altitude:={args.return_altitude} "
        f"-p target_yaw:={initial_heading} "
        f"-p timeout_sec:={args.return_timeout}"
    )
    print(f"returning to origin, log={log_path}")
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            ["/bin/bash", "-lc", command],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(f"return_to_origin exited with code {result.returncode}")


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


def wait_for_trial_result(csv_path, timeout, strict_no_lost=True):
    start = time.time()
    last_target_time = 0.0
    target_seen = False
    while time.time() - start < timeout:
        rows = read_rows(csv_path)
        if rows:
            for row in rows:
                if target_is_seen(row):
                    target_seen = True
                elif strict_no_lost and target_seen and row.get("state") == "LOST":
                    row_time = to_float(row.get("time_sec", ""))
                    return f"invalid_lost_before_capture time_sec={row_time:.3f}"

            captured = [row for row in rows if row.get("captured") == "1"]
            if captured:
                if strict_no_lost:
                    lost_report = lost_before_capture(rows[: rows.index(captured[0]) + 1])
                    if lost_report["lost"]:
                        return (
                            "invalid_lost_before_capture "
                            f"lost_rows={lost_report['rows']} "
                            f"max_lost_duration={lost_report['max_duration']:.3f}s"
                        )
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


def target_is_seen(row):
    state = row.get("state", "")
    if state in ("UNKNOWN", "SEARCH", "LOST", "CAPTURED"):
        return False
    return math.isfinite(to_float(row.get("error_x_px", ""))) or math.isfinite(to_float(row.get("area_px", "")))


def lost_before_capture(rows):
    first_seen = None
    for index, row in enumerate(rows):
        if target_is_seen(row):
            first_seen = index
            break

    if first_seen is None:
        return {"lost": False, "rows": 0, "max_duration": 0.0}

    lost_rows = 0
    max_duration = 0.0
    segment_start = None
    segment_last = None
    for row in rows[first_seen:]:
        row_time = to_float(row.get("time_sec", ""))
        if row.get("state") == "LOST":
            lost_rows += 1
            if segment_start is None:
                segment_start = row_time
            segment_last = row_time
            continue

        if segment_start is not None:
            if segment_start == segment_start and segment_last == segment_last:
                max_duration = max(max_duration, max(0.0, segment_last - segment_start))
            segment_start = None
            segment_last = None

    if segment_start is not None and segment_start == segment_start and segment_last == segment_last:
        max_duration = max(max_duration, max(0.0, segment_last - segment_start))

    return {"lost": lost_rows > 0, "rows": lost_rows, "max_duration": max_duration}


def score_latest():
    command = f"{ROS_SETUP} && ros2 run vision_tracking score_experiment"
    subprocess.run(["/bin/bash", "-lc", command], check=False)


def start_pose_is_reasonable(max_xy, min_z, max_z):
    values = {}
    for _ in range(3):
        values = read_local_position("/fmu/out/vehicle_local_position_v1")
        if values:
            break
        time.sleep(1.0)
    if not values:
        for _ in range(3):
            values = read_local_position("/fmu/out/vehicle_local_position")
            if values:
                break
            time.sleep(1.0)
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


def read_initial_heading():
    values = read_local_position("/fmu/out/vehicle_local_position_v1")
    if not values:
        values = read_local_position("/fmu/out/vehicle_local_position")
    if not values:
        return 0.0
    if values.get("heading_good_for_control") is False:
        return 0.0
    try:
        return float(values.get("heading", 0.0))
    except (TypeError, ValueError):
        return 0.0


def read_local_position(topic):
    command = f"{ROS_SETUP} && timeout 8 ros2 topic echo --once --qos-reliability best_effort {topic}"
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
