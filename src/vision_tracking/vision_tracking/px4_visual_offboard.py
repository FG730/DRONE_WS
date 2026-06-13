#!/usr/bin/env python3

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleAttitude, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


CMD_TOPIC = "/vision/cmd_velocity"


class Px4VisualOffboard(Node):
    def __init__(self):
        super().__init__("px4_visual_offboard")

        self.declare_parameter("auto_offboard", True)
        self.declare_parameter("auto_arm", False)
        self.declare_parameter("start_delay_sec", 2.0)
        self.declare_parameter("mode_retry_sec", 1.0)
        self.declare_parameter("cmd_timeout_sec", 0.5)
        self.declare_parameter("max_forward_speed", 0.30)
        self.declare_parameter("max_lateral_speed", 0.25)
        self.declare_parameter("max_vertical_speed", 0.25)
        self.declare_parameter("max_yaw_rate", 0.30)
        self.declare_parameter("forward_sign", 1.0)
        self.declare_parameter("lateral_sign", 1.0)
        self.declare_parameter("forward_yaw_offset_deg", 0.0)
        self.declare_parameter("allow_reverse", False)
        self.declare_parameter("hold_after_capture", True)
        self.declare_parameter("capture_hold_mode_delay_sec", 2.0)
        self.declare_parameter("yaw_only_velocity_transform", True)

        self.auto_offboard = bool(self.get_parameter("auto_offboard").value)
        self.auto_arm = bool(self.get_parameter("auto_arm").value)
        self.start_delay_sec = float(self.get_parameter("start_delay_sec").value)
        self.mode_retry_sec = float(self.get_parameter("mode_retry_sec").value)
        self.cmd_timeout_sec = float(self.get_parameter("cmd_timeout_sec").value)
        self.max_forward_speed = float(self.get_parameter("max_forward_speed").value)
        self.max_lateral_speed = float(self.get_parameter("max_lateral_speed").value)
        self.max_vertical_speed = float(self.get_parameter("max_vertical_speed").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.forward_sign = float(self.get_parameter("forward_sign").value)
        self.lateral_sign = float(self.get_parameter("lateral_sign").value)
        self.forward_yaw_offset = math.radians(float(self.get_parameter("forward_yaw_offset_deg").value))
        self.allow_reverse = bool(self.get_parameter("allow_reverse").value)
        self.hold_after_capture = bool(self.get_parameter("hold_after_capture").value)
        self.capture_hold_mode_delay_sec = float(self.get_parameter("capture_hold_mode_delay_sec").value)
        self.yaw_only_velocity_transform = bool(self.get_parameter("yaw_only_velocity_transform").value)

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            px4_qos,
        )
        self.trajectory_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            px4_qos,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            px4_qos,
        )

        self.cmd_sub = self.create_subscription(Twist, CMD_TOPIC, self.cmd_callback, 10)
        self.captured_sub = self.create_subscription(Bool, "/vision/captured", self.captured_callback, 10)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v4",
            self.status_callback,
            px4_qos,
        )
        self.legacy_status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status",
            self.status_callback,
            px4_qos,
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.local_position_callback,
            px4_qos,
        )
        self.legacy_local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.local_position_callback,
            px4_qos,
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_callback,
            px4_qos,
        )

        self.latest_cmd = Twist()
        self.last_cmd_time = None
        self.heading = 0.0
        self.attitude_q = [1.0, 0.0, 0.0, 0.0]
        self.nav_state = None
        self.arming_state = None
        self.start_time = time.monotonic()
        self.last_mode_request_time = 0.0
        self.arm_sent = False
        self.captured = False
        self.capture_time = None

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(f"Subscribed to {CMD_TOPIC}")
        self.get_logger().info("Publishing low-speed PX4 Offboard velocity setpoints")

    def cmd_callback(self, msg):
        if self.captured and self.hold_after_capture:
            self.latest_cmd = Twist()
            self.last_cmd_time = time.monotonic()
            return
        self.latest_cmd = msg
        self.last_cmd_time = time.monotonic()

    def captured_callback(self, msg):
        if bool(msg.data) and not self.captured:
            self.get_logger().info("Capture received; commanding zero velocity before Hold/Loiter")
            self.captured = True
            self.capture_time = time.monotonic()
            self.latest_cmd = Twist()
            self.last_cmd_time = time.monotonic()

    def status_callback(self, msg):
        self.nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def local_position_callback(self, msg):
        if msg.heading_good_for_control:
            self.heading = msg.heading

    def attitude_callback(self, msg):
        self.attitude_q = [float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])]

    def timer_callback(self):
        now_us = self.get_clock().now().nanoseconds // 1000
        self.publish_offboard_control_mode(now_us)
        self.publish_trajectory_setpoint(now_us)

        if time.monotonic() - self.start_time < self.start_delay_sec:
            return

        if self.should_request_hold_after_capture():
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=4.0,
                param3=3.0,
            )
            self.last_mode_request_time = time.monotonic()
            self.get_logger().info(
                f"Requested AUTO.LOITER/Hold after capture (nav_state={self.nav_state}, arming_state={self.arming_state})"
            )
            return

        if self.auto_offboard and self.should_request_offboard():
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0,
            )
            self.last_mode_request_time = time.monotonic()
            self.get_logger().info(
                f"Requested OFFBOARD mode (nav_state={self.nav_state}, arming_state={self.arming_state})"
            )

        if self.auto_arm and not self.arm_sent:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=1.0,
            )
            self.arm_sent = True
            self.get_logger().info("Requested arm")

    def should_request_offboard(self):
        if self.captured and self.hold_after_capture:
            return False

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            return False

        if time.monotonic() - self.last_mode_request_time < self.mode_retry_sec:
            return False

        return True

    def should_request_hold_after_capture(self):
        if not self.captured or not self.hold_after_capture:
            return False

        if self.capture_time is None:
            return False

        if time.monotonic() - self.capture_time < self.capture_hold_mode_delay_sec:
            return False

        if time.monotonic() - self.last_mode_request_time < self.mode_retry_sec:
            return False

        return True

    def publish_offboard_control_mode(self, timestamp):
        msg = OffboardControlMode()
        msg.timestamp = timestamp
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self, timestamp):
        body_forward, body_right, vertical_up, yaw_rate = self.get_safe_body_command()
        body_forward *= self.forward_sign
        body_right *= self.lateral_sign

        if self.yaw_only_velocity_transform:
            north, east, down = self.rotate_body_yaw_to_ned(body_forward, body_right, vertical_up)
        else:
            # PX4 VehicleAttitude.q rotates FRD body vectors into NED.
            # The visual command uses forward/right/up, so convert up to FRD down first.
            north, east, down = self.rotate_body_frd_to_ned(
                body_forward,
                body_right,
                -vertical_up,
            )

        msg = TrajectorySetpoint()
        msg.timestamp = timestamp
        msg.position = [math.nan, math.nan, math.nan]
        msg.velocity = [float(north), float(east), float(down)]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = math.nan
        msg.yawspeed = float(-yaw_rate)
        self.trajectory_pub.publish(msg)

        self.get_logger().info(
            "body_v=({:.2f}, {:.2f}, {:.2f}) -> ned_v=({:.2f}, {:.2f}, {:.2f}) yaw_rate={:.2f}".format(
                body_forward,
                body_right,
                vertical_up,
                north,
                east,
                down,
                yaw_rate,
            ),
            throttle_duration_sec=0.5,
        )

    def get_safe_body_command(self):
        if self.captured and self.hold_after_capture:
            return 0.0, 0.0, 0.0, 0.0

        if self.last_cmd_time is None:
            return 0.0, 0.0, 0.0, 0.0

        if time.monotonic() - self.last_cmd_time > self.cmd_timeout_sec:
            return 0.0, 0.0, 0.0, 0.0

        min_forward_speed = -self.max_forward_speed if self.allow_reverse else 0.0
        forward = self.clamp(self.latest_cmd.linear.x, min_forward_speed, self.max_forward_speed)
        body_right = self.clamp(self.latest_cmd.linear.y, -self.max_lateral_speed, self.max_lateral_speed)
        vertical_up = self.clamp(self.latest_cmd.linear.z, -self.max_vertical_speed, self.max_vertical_speed)
        yaw_rate = self.clamp(self.latest_cmd.angular.z, -self.max_yaw_rate, self.max_yaw_rate)
        return forward, body_right, vertical_up, yaw_rate

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0, param4=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.command_pub.publish(msg)

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def rotate_body_frd_to_ned(self, x, y, z):
        w, qx, qy, qz = self.attitude_q

        # Rotate vector by Hamilton quaternion: q * v * q_conjugate.
        tx = 2.0 * (qy * z - qz * y)
        ty = 2.0 * (qz * x - qx * z)
        tz = 2.0 * (qx * y - qy * x)

        rx = x + w * tx + (qy * tz - qz * ty)
        ry = y + w * ty + (qz * tx - qx * tz)
        rz = z + w * tz + (qx * ty - qy * tx)
        return rx, ry, rz

    def rotate_body_yaw_to_ned(self, forward, right, vertical_up):
        heading = self.heading + self.forward_yaw_offset
        cos_yaw = math.cos(heading)
        sin_yaw = math.sin(heading)
        north = forward * cos_yaw - right * sin_yaw
        east = forward * sin_yaw + right * cos_yaw
        down = -vertical_up
        return north, east, down


def main():
    rclpy.init()
    node = Px4VisualOffboard()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
