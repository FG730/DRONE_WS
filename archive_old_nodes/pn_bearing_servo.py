#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import String


ERROR_TOPIC = "/vision/target_error"
CMD_TOPIC = "/vision/cmd_velocity"


class PnBearingServo(Node):
    def __init__(self):
        super().__init__("pn_bearing_servo")

        self.declare_parameter("base_forward_speed", 1.5)
        self.declare_parameter("max_forward_speed", 5.0)
        self.declare_parameter("min_forward_scale", 0.15)
        self.declare_parameter("kp_lateral", 0.0060)
        self.declare_parameter("kd_lateral", 0.0015)
        self.declare_parameter("kp_vertical", -0.0060)
        self.declare_parameter("kd_vertical", -0.0012)
        self.declare_parameter("kp_yaw", 0.0010)
        self.declare_parameter("kd_yaw", 0.0004)
        self.declare_parameter("max_lateral_speed", 2.0)
        self.declare_parameter("max_vertical_speed", 1.2)
        self.declare_parameter("max_yaw_rate", 0.35)
        self.declare_parameter("center_x_px", 25.0)
        self.declare_parameter("center_y_px", 45.0)
        self.declare_parameter("slow_x_px", 260.0)
        self.declare_parameter("slow_y_px", 220.0)
        self.declare_parameter("rate_slow_px_s", 900.0)
        self.declare_parameter("min_area", 100.0)
        self.declare_parameter("derivative_alpha", 0.35)
        self.declare_parameter("target_timeout_sec", 0.35)
        self.declare_parameter("enable_approach_guard", True)
        self.declare_parameter("area_drop_epsilon", 40.0)
        self.declare_parameter("area_drop_frames", 3)
        self.declare_parameter("guard_hold_frames", 8)

        self.base_forward_speed = float(self.get_parameter("base_forward_speed").value)
        self.max_forward_speed = float(self.get_parameter("max_forward_speed").value)
        self.min_forward_scale = float(self.get_parameter("min_forward_scale").value)
        self.kp_lateral = float(self.get_parameter("kp_lateral").value)
        self.kd_lateral = float(self.get_parameter("kd_lateral").value)
        self.kp_vertical = float(self.get_parameter("kp_vertical").value)
        self.kd_vertical = float(self.get_parameter("kd_vertical").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.kd_yaw = float(self.get_parameter("kd_yaw").value)
        self.max_lateral_speed = float(self.get_parameter("max_lateral_speed").value)
        self.max_vertical_speed = float(self.get_parameter("max_vertical_speed").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.center_x_px = float(self.get_parameter("center_x_px").value)
        self.center_y_px = float(self.get_parameter("center_y_px").value)
        self.slow_x_px = float(self.get_parameter("slow_x_px").value)
        self.slow_y_px = float(self.get_parameter("slow_y_px").value)
        self.rate_slow_px_s = float(self.get_parameter("rate_slow_px_s").value)
        self.min_area = float(self.get_parameter("min_area").value)
        self.derivative_alpha = float(self.get_parameter("derivative_alpha").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.enable_approach_guard = bool(self.get_parameter("enable_approach_guard").value)
        self.area_drop_epsilon = float(self.get_parameter("area_drop_epsilon").value)
        self.area_drop_frames = int(self.get_parameter("area_drop_frames").value)
        self.guard_hold_frames = int(self.get_parameter("guard_hold_frames").value)

        self.last_error = None
        self.last_time = None
        self.filtered_x_dot = 0.0
        self.filtered_y_dot = 0.0
        self.last_area = None
        self.area_drop_count = 0
        self.guard_hold_count = 0
        self.last_target_time = None
        self.captured = False
        self.state = "SEARCH"

        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.state_pub = self.create_publisher(String, "/vision/tracking_state", 10)
        self.create_subscription(Point, ERROR_TOPIC, self.error_callback, 10)
        self.create_subscription(Bool, "/vision/captured", self.captured_callback, 10)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(f"Subscribed to {ERROR_TOPIC}")
        self.get_logger().info("Publishing PN-inspired bearing-rate visual guidance commands")

    def error_callback(self, error):
        if self.captured:
            self.publish_stop("CAPTURED")
            return

        now = self.get_clock().now()
        self.last_target_time = now
        x_dot, y_dot = self.estimate_error_rate(error, now)

        guarded = self.update_approach_guard(error.z)
        forward_scale = self.forward_scale(error, x_dot, y_dot)
        if guarded:
            forward_scale = 0.0

        cmd = Twist()
        cmd.linear.x = self._clamp(
            self.base_forward_speed * forward_scale,
            0.0,
            self.max_forward_speed,
        )

        lateral = self.kp_lateral * error.x + self.kd_lateral * x_dot
        vertical = self.kp_vertical * error.y + self.kd_vertical * y_dot
        yaw_rate = -(self.kp_yaw * error.x + self.kd_yaw * x_dot)

        cmd.linear.y = self._clamp(lateral, -self.max_lateral_speed, self.max_lateral_speed)
        cmd.linear.z = self._clamp(vertical, -self.max_vertical_speed, self.max_vertical_speed)
        cmd.angular.z = self._clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        self.state = self.select_state(error, forward_scale, guarded)
        self.publish_state()
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            "state={} err=({:.1f},{:.1f}) rate=({:.1f},{:.1f}) area={:.0f} scale={:.2f} guard={} cmd=({:.2f},{:.2f},{:.2f},{:.2f})".format(
                self.state,
                error.x,
                error.y,
                x_dot,
                y_dot,
                error.z,
                forward_scale,
                guarded,
                cmd.linear.x,
                cmd.linear.y,
                cmd.linear.z,
                cmd.angular.z,
            ),
            throttle_duration_sec=0.5,
        )

    def estimate_error_rate(self, error, now):
        if self.last_error is None or self.last_time is None:
            self.last_error = error
            self.last_time = now
            return 0.0, 0.0

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 1e-3:
            return self.filtered_x_dot, self.filtered_y_dot

        raw_x_dot = (error.x - self.last_error.x) / dt
        raw_y_dot = (error.y - self.last_error.y) / dt
        alpha = self._clamp(self.derivative_alpha, 0.0, 1.0)
        self.filtered_x_dot = alpha * raw_x_dot + (1.0 - alpha) * self.filtered_x_dot
        self.filtered_y_dot = alpha * raw_y_dot + (1.0 - alpha) * self.filtered_y_dot

        self.last_error = error
        self.last_time = now
        return self.filtered_x_dot, self.filtered_y_dot

    def forward_scale(self, error, x_dot, y_dot):
        if error.z < self.min_area:
            return 0.0

        x_scale = abs(error.x) / max(self.slow_x_px, 1.0)
        y_scale = abs(error.y) / max(self.slow_y_px, 1.0)
        rate_scale = math.hypot(x_dot, y_dot) / max(self.rate_slow_px_s, 1.0)
        penalty = max(x_scale, y_scale, rate_scale)

        if abs(error.x) <= self.center_x_px and abs(error.y) <= self.center_y_px:
            return self._clamp(1.0 - 0.5 * rate_scale, self.min_forward_scale, 1.0)

        return self._clamp(1.0 - penalty, self.min_forward_scale, 1.0)

    def select_state(self, error, forward_scale, guarded):
        if guarded:
            return "GUARD"
        if error.z < self.min_area:
            return "SEARCH"
        if abs(error.x) <= self.center_x_px and abs(error.y) <= self.center_y_px:
            return "PN_APPROACH"
        if forward_scale > self.min_forward_scale:
            return "PN_TRACK"
        return "PN_ALIGN"

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

    def watchdog_callback(self):
        if self.captured:
            self.publish_stop("CAPTURED")
            return

        if self.last_target_time is None:
            self.publish_stop("SEARCH")
            return

        age_sec = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
        if age_sec > self.target_timeout_sec:
            self.last_error = None
            self.last_time = None
            self.filtered_x_dot = 0.0
            self.filtered_y_dot = 0.0
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

    @staticmethod
    def _clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))


def main():
    rclpy.init()
    node = PnBearingServo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
