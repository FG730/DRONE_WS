#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String


ERROR_TOPIC = "/vision/target_error"
CMD_TOPIC = "/vision/cmd_velocity"


class BearingServo(Node):
    def __init__(self):
        super().__init__("bearing_servo")

        self.declare_parameter("fixed_forward_speed", 0.15)
        self.declare_parameter("kp_yaw", 0.0020)
        self.declare_parameter("kp_z", -0.0020)
        self.declare_parameter("kp_lateral", 0.0010)
        self.declare_parameter("max_yaw_rate", 0.5)
        self.declare_parameter("max_lateral_speed", 0.15)
        self.declare_parameter("max_vertical_speed", 0.3)
        self.declare_parameter("enable_lateral", True)
        self.declare_parameter("enable_vertical", False)
        self.declare_parameter("center_x_px", 60.0)
        self.declare_parameter("center_y_px", 80.0)
        self.declare_parameter("slow_x_px", 220.0)
        self.declare_parameter("slow_y_px", 180.0)
        self.declare_parameter("min_area", 100.0)
        self.declare_parameter("enable_approach_guard", True)
        self.declare_parameter("area_drop_epsilon", 20.0)
        self.declare_parameter("area_drop_frames", 4)
        self.declare_parameter("guard_hold_frames", 20)
        self.declare_parameter("target_timeout_sec", 0.5)

        self.fixed_forward_speed = float(self.get_parameter("fixed_forward_speed").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.kp_z = float(self.get_parameter("kp_z").value)
        self.kp_lateral = float(self.get_parameter("kp_lateral").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.max_lateral_speed = float(self.get_parameter("max_lateral_speed").value)
        self.max_vertical_speed = float(self.get_parameter("max_vertical_speed").value)
        self.enable_lateral = bool(self.get_parameter("enable_lateral").value)
        self.enable_vertical = bool(self.get_parameter("enable_vertical").value)
        self.center_x_px = float(self.get_parameter("center_x_px").value)
        self.center_y_px = float(self.get_parameter("center_y_px").value)
        self.slow_x_px = float(self.get_parameter("slow_x_px").value)
        self.slow_y_px = float(self.get_parameter("slow_y_px").value)
        self.min_area = float(self.get_parameter("min_area").value)
        self.enable_approach_guard = bool(self.get_parameter("enable_approach_guard").value)
        self.area_drop_epsilon = float(self.get_parameter("area_drop_epsilon").value)
        self.area_drop_frames = int(self.get_parameter("area_drop_frames").value)
        self.guard_hold_frames = int(self.get_parameter("guard_hold_frames").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)

        self.last_area = None
        self.area_drop_count = 0
        self.guard_hold_count = 0
        self.last_target_time = None
        self.state = "SEARCH"
        self.captured = False

        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.state_pub = self.create_publisher(String, "/vision/tracking_state", 10)
        self.error_sub = self.create_subscription(Point, ERROR_TOPIC, self.error_callback, 10)
        self.captured_sub = self.create_subscription(Bool, "/vision/captured", self.captured_callback, 10)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(f"Subscribed to {ERROR_TOPIC}")
        self.get_logger().info(f"Publishing bearing-only fixed-speed commands to {CMD_TOPIC}")

    def error_callback(self, error):
        if self.captured:
            self.publish_stop("CAPTURED")
            return

        cmd = Twist()
        self.last_target_time = self.get_clock().now()

        yaw_rate = -self.kp_yaw * error.x
        cmd.angular.z = self._clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        if self.enable_vertical:
            vertical_speed = self.kp_z * error.y
            cmd.linear.z = self._clamp(vertical_speed, -self.max_vertical_speed, self.max_vertical_speed)
        else:
            cmd.linear.z = 0.0

        scale = self.forward_scale(error)
        guarded = self.update_approach_guard(error.z)
        if guarded:
            scale = 0.0

        self.state = self.select_state(error, scale, guarded)
        self.publish_state()
        cmd.linear.x = self.fixed_forward_speed * scale
        if self.enable_lateral:
            lateral_speed = self.kp_lateral * error.x
            cmd.linear.y = self._clamp(lateral_speed, -self.max_lateral_speed, self.max_lateral_speed)
        else:
            cmd.linear.y = 0.0
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            "state={} err(px)=({:.1f}, {:.1f}) area={:.1f} -> scale={:.2f} guard={} cmd vx={:.2f} vy={:.2f} vz={:.2f} yaw_rate={:.2f}".format(
                self.state,
                error.x,
                error.y,
                error.z,
                scale,
                guarded,
                cmd.linear.x,
                cmd.linear.y,
                cmd.linear.z,
                cmd.angular.z,
            ),
            throttle_duration_sec=0.5,
        )

    def watchdog_callback(self):
        if self.captured:
            self.publish_stop("CAPTURED")
            return

        if self.last_target_time is None:
            self.publish_stop("SEARCH")
            return

        age_sec = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
        if age_sec > self.target_timeout_sec:
            self.last_area = None
            self.area_drop_count = 0
            self.guard_hold_count = 0
            self.publish_stop("LOST")

    def captured_callback(self, msg):
        self.captured = bool(msg.data)
        if self.captured:
            self.publish_stop("CAPTURED")

    def publish_stop(self, state):
        self.state = state
        self.publish_state()
        self.cmd_pub.publish(Twist())
        self.get_logger().info(f"state={self.state} -> cmd stop", throttle_duration_sec=0.5)

    def publish_state(self):
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

    def forward_scale(self, error):
        if error.z < self.min_area:
            return 0.0

        x_abs = abs(error.x)
        y_abs = abs(error.y)

        if not self.enable_vertical:
            y_abs = 0.0

        if x_abs <= self.center_x_px and y_abs <= self.center_y_px:
            return 1.0

        x_scale = x_abs / max(self.slow_x_px, 1.0)
        y_scale = y_abs / max(self.slow_y_px, 1.0)
        error_scale = max(x_scale, y_scale)
        return self._clamp(1.0 - error_scale, 0.0, 1.0)

    def select_state(self, error, scale, guarded):
        if guarded:
            return "GUARD"

        if error.z < self.min_area:
            return "SEARCH"

        x_abs = abs(error.x)
        y_abs = abs(error.y) if self.enable_vertical else 0.0

        if x_abs <= self.center_x_px and y_abs <= self.center_y_px:
            return "APPROACH"

        if scale > 0.0:
            return "TRACK"

        return "ALIGN"

    def update_approach_guard(self, area):
        if not self.enable_approach_guard:
            self.last_area = area
            return False

        if self.guard_hold_count > 0:
            self.guard_hold_count -= 1
            self.last_area = area
            return True

        if self.last_area is None:
            self.last_area = area
            return False

        if self.last_area - area > self.area_drop_epsilon:
            self.area_drop_count += 1
        else:
            self.area_drop_count = 0

        self.last_area = area

        if self.area_drop_count >= self.area_drop_frames:
            self.area_drop_count = 0
            self.guard_hold_count = self.guard_hold_frames
            return True

        return False

    @staticmethod
    def _clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))


def main():
    rclpy.init()
    node = BearingServo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
