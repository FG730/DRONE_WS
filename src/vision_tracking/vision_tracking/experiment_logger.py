#!/usr/bin/env python3

import csv
import math
import os
import time
from datetime import datetime

import rclpy
from geometry_msgs.msg import Point, Twist
from px4_msgs.msg import VehicleLocalPosition
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_msgs.msg import String


class ExperimentLogger(Node):
    def __init__(self):
        super().__init__("experiment_logger")

        self.declare_parameter("target_pose_file", "/tmp/red_target_pose.csv")
        self.declare_parameter("drone_position_topic", "/fmu/out/vehicle_local_position_v1")
        self.declare_parameter("legacy_drone_position_topic", "/fmu/out/vehicle_local_position")
        self.declare_parameter("swap_px4_xy_to_gazebo", True)
        self.declare_parameter("capture_radius", 1.0)
        self.declare_parameter("capture_hold_sec", 0.0)
        self.declare_parameter("require_fresh_target_pose", True)
        self.declare_parameter("target_pose_stale_sec", 1.0)
        self.declare_parameter("log_rate_hz", 20.0)
        self.declare_parameter("output_dir", os.path.expanduser("~/drone_ws/logs"))
        self.declare_parameter("file_prefix", "experiment")

        self.target_pose_file = os.path.expanduser(self.get_parameter("target_pose_file").value)
        self.drone_position_topic = self.get_parameter("drone_position_topic").value
        self.legacy_drone_position_topic = self.get_parameter("legacy_drone_position_topic").value
        self.swap_px4_xy_to_gazebo = bool(self.get_parameter("swap_px4_xy_to_gazebo").value)
        self.capture_radius = float(self.get_parameter("capture_radius").value)
        self.capture_hold_sec = float(self.get_parameter("capture_hold_sec").value)
        self.require_fresh_target_pose = bool(self.get_parameter("require_fresh_target_pose").value)
        self.target_pose_stale_sec = float(self.get_parameter("target_pose_stale_sec").value)
        self.log_rate_hz = float(self.get_parameter("log_rate_hz").value)
        output_dir = os.path.expanduser(self.get_parameter("output_dir").value)
        file_prefix = self.get_parameter("file_prefix").value

        os.makedirs(output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(output_dir, f"{file_prefix}_{stamp}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "time_sec",
                "state",
                "error_x_px",
                "error_y_px",
                "error_norm_px",
                "virtual_error_x_px",
                "virtual_error_y_px",
                "virtual_error_norm_px",
                "area_px",
                "cmd_vx",
                "cmd_vy",
                "cmd_vz",
                "cmd_yaw_rate",
                "drone_x",
                "drone_y",
                "drone_z",
                "target_x",
                "target_y",
                "target_z",
                "distance",
                "captured",
                "capture_time_sec",
                "target_time_sec",
                "capture_target_time_sec",
            ],
        )
        self.writer.writeheader()

        self.latest_error = None
        self.latest_virtual_error = None
        self.latest_cmd = None
        self.latest_state = "UNKNOWN"
        self.drone_pos = None
        self.target_pos = None
        self.target_first_seen_time = None
        self.start_time = self.get_clock().now()
        self.wall_start_time = time.time()
        self.capture_start_time = None
        self.captured = False
        self.capture_time_sec = math.nan
        self.capture_target_time_sec = math.nan
        self.target_pose_file_warned = False

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Point, "/vision/target_error", self.error_callback, 10)
        self.create_subscription(Point, "/vision/virtual_target_error", self.virtual_error_callback, 10)
        self.create_subscription(Twist, "/vision/cmd_velocity", self.cmd_callback, 10)
        self.create_subscription(String, "/vision/tracking_state", self.state_callback, 10)
        self.create_subscription(
            VehicleLocalPosition,
            self.drone_position_topic,
            self.local_position_callback,
            px4_qos,
        )
        self.create_subscription(
            VehicleLocalPosition,
            self.legacy_drone_position_topic,
            self.local_position_callback,
            px4_qos,
        )
        self.captured_pub = self.create_publisher(Bool, "/vision/captured", 10)
        self.timer = self.create_timer(1.0 / self.log_rate_hz, self.timer_callback)

        self.get_logger().info(f"Logging experiment CSV to {self.csv_path}")
        self.get_logger().info(
            f"Virtual capture: distance <= {self.capture_radius:.2f} m for {self.capture_hold_sec:.2f} s"
        )
        self.get_logger().info(
            f"Reading drone PX4 local position from {self.drone_position_topic} and {self.legacy_drone_position_topic}"
        )
        self.get_logger().info(
            f"PX4 local to Gazebo transform: swap_xy={self.swap_px4_xy_to_gazebo}, z=-px4_z"
        )
        self.get_logger().info(f"Reading target truth pose from {self.target_pose_file}")
        self.get_logger().info(
            f"Target pose freshness: require_fresh={self.require_fresh_target_pose}, "
            f"stale_sec={self.target_pose_stale_sec:.2f}"
        )

    def error_callback(self, msg):
        self.latest_error = msg

    def virtual_error_callback(self, msg):
        self.latest_virtual_error = msg

    def cmd_callback(self, msg):
        self.latest_cmd = msg

    def state_callback(self, msg):
        self.latest_state = msg.data

    def local_position_callback(self, msg):
        if not msg.xy_valid or not msg.z_valid:
            return

        px4_x = float(msg.x)
        px4_y = float(msg.y)
        gazebo_z = float(-msg.z)

        if self.swap_px4_xy_to_gazebo:
            self.drone_pos = (px4_y, px4_x, gazebo_z)
        else:
            self.drone_pos = (px4_x, px4_y, gazebo_z)

    def timer_callback(self):
        now = self.get_clock().now()
        time_sec = (now - self.start_time).nanoseconds / 1e9
        self.read_target_pose_file(now)
        distance = self.compute_distance()
        target_time_sec = self.compute_target_time(now)

        error_x = self.latest_error.x if self.latest_error else math.nan
        error_y = self.latest_error.y if self.latest_error else math.nan
        area = self.latest_error.z if self.latest_error else math.nan
        error_norm = math.hypot(error_x, error_y) if self.latest_error else math.nan
        virtual_error_x = self.latest_virtual_error.x if self.latest_virtual_error else math.nan
        virtual_error_y = self.latest_virtual_error.y if self.latest_virtual_error else math.nan
        virtual_error_norm = (
            math.hypot(virtual_error_x, virtual_error_y)
            if self.latest_virtual_error
            else math.nan
        )

        self.update_capture(now, time_sec, distance)
        self.publish_captured()

        cmd_vx = self.latest_cmd.linear.x if self.latest_cmd else math.nan
        cmd_vy = self.latest_cmd.linear.y if self.latest_cmd else math.nan
        cmd_vz = self.latest_cmd.linear.z if self.latest_cmd else math.nan
        cmd_yaw = self.latest_cmd.angular.z if self.latest_cmd else math.nan

        drone_x, drone_y, drone_z = self.drone_pos if self.drone_pos else (math.nan, math.nan, math.nan)
        target_x, target_y, target_z = self.target_pos if self.target_pos else (math.nan, math.nan, math.nan)

        self.writer.writerow(
            {
                "time_sec": f"{time_sec:.3f}",
                "state": self.latest_state,
                "error_x_px": self.format_float(error_x),
                "error_y_px": self.format_float(error_y),
                "error_norm_px": self.format_float(error_norm),
                "virtual_error_x_px": self.format_float(virtual_error_x),
                "virtual_error_y_px": self.format_float(virtual_error_y),
                "virtual_error_norm_px": self.format_float(virtual_error_norm),
                "area_px": self.format_float(area),
                "cmd_vx": self.format_float(cmd_vx),
                "cmd_vy": self.format_float(cmd_vy),
                "cmd_vz": self.format_float(cmd_vz),
                "cmd_yaw_rate": self.format_float(cmd_yaw),
                "drone_x": self.format_float(drone_x),
                "drone_y": self.format_float(drone_y),
                "drone_z": self.format_float(drone_z),
                "target_x": self.format_float(target_x),
                "target_y": self.format_float(target_y),
                "target_z": self.format_float(target_z),
                "distance": self.format_float(distance),
                "captured": int(self.captured),
                "capture_time_sec": self.format_float(self.capture_time_sec),
                "target_time_sec": self.format_float(target_time_sec),
                "capture_target_time_sec": self.format_float(self.capture_target_time_sec),
            }
        )
        self.csv_file.flush()

        if self.captured:
            self.get_logger().info(
                f"CAPTURED distance={distance:.2f} m capture_time={self.capture_time_sec:.2f} s",
                throttle_duration_sec=1.0,
            )
        else:
            if math.isnan(distance):
                missing = []
                if self.drone_pos is None:
                    missing.append("drone PX4 local position")
                if self.target_pos is None:
                    missing.append("target pose file")
                self.get_logger().warn(
                    "No truth distance yet; missing: {}. Capture will NOT trigger.".format(
                        ", ".join(missing) if missing else "unknown"
                    ),
                    throttle_duration_sec=5.0,
                )
            self.get_logger().info(
                f"state={self.latest_state} distance={self.format_float(distance)} m captured=0",
                throttle_duration_sec=1.0,
            )

    def compute_distance(self):
        if self.drone_pos is None or self.target_pos is None:
            return math.nan
        dx = self.drone_pos[0] - self.target_pos[0]
        dy = self.drone_pos[1] - self.target_pos[1]
        dz = self.drone_pos[2] - self.target_pos[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def read_target_pose_file(self, now):
        self.target_pos = self.read_pose_file(
            self.target_pose_file,
            self.target_pos,
            "target",
            now,
        )

    def read_pose_file(self, path, previous_pos, label, now):
        try:
            with open(path, "r") as pose_file:
                line = pose_file.readline().strip()
        except OSError:
            if label == "target" and not self.target_pose_file_warned:
                self.get_logger().warn(f"Target pose file not found yet: {path}")
                self.target_pose_file_warned = True
            return previous_pos

        parts = line.split(",")
        if len(parts) != 4:
            return previous_pos

        try:
            pose_wall_time = float(parts[0])
            if self.require_fresh_target_pose:
                if pose_wall_time < self.wall_start_time:
                    return None
                logger_wall_time = self.wall_start_time + (now - self.start_time).nanoseconds / 1e9
                if logger_wall_time - pose_wall_time > self.target_pose_stale_sec:
                    return None
            if label == "target" and self.target_first_seen_time is None:
                self.target_first_seen_time = now
            return (float(parts[1]), float(parts[2]), float(parts[3]))
        except ValueError:
            return previous_pos

    def compute_target_time(self, now):
        if self.target_first_seen_time is None:
            return math.nan
        return (now - self.target_first_seen_time).nanoseconds / 1e9

    def update_capture(self, now, time_sec, distance):
        if self.captured or math.isnan(distance):
            return

        if distance <= self.capture_radius:
            if self.capture_hold_sec <= 0.0:
                self.captured = True
                self.capture_time_sec = time_sec
                self.capture_target_time_sec = self.compute_target_time(now)
                return

            if self.capture_start_time is None:
                self.capture_start_time = now
                return

            hold_sec = (now - self.capture_start_time).nanoseconds / 1e9
            if hold_sec >= self.capture_hold_sec:
                self.captured = True
                self.capture_time_sec = time_sec
                self.capture_target_time_sec = self.compute_target_time(now)
        else:
            self.capture_start_time = None

    def publish_captured(self):
        msg = Bool()
        msg.data = self.captured
        self.captured_pub.publish(msg)

    @staticmethod
    def format_float(value):
        if value is None or math.isnan(value):
            return ""
        return f"{value:.6f}"

    def destroy_node(self):
        self.csv_file.flush()
        self.csv_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = ExperimentLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
