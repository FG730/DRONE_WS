#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point, Twist
from px4_msgs.msg import VehicleAttitude
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from std_msgs.msg import String


ERROR_TOPIC = "/vision/target_error"
VIRTUAL_ERROR_TOPIC = "/vision/virtual_target_error"
CMD_TOPIC = "/vision/cmd_velocity"


class AttitudePnBearingServo(Node):
    def __init__(self):
        super().__init__("attitude_pn_bearing_servo")

        self.declare_parameter("base_forward_speed", 3.0)
        self.declare_parameter("max_forward_speed", 5.0)
        self.declare_parameter("min_forward_scale", 0.20)
        self.declare_parameter("kp_lateral", 0.0080)
        self.declare_parameter("kd_lateral", 0.0020)
        self.declare_parameter("kp_vertical", -0.0060)
        self.declare_parameter("kd_vertical", -0.0015)
        self.declare_parameter("kp_yaw", 0.0010)
        self.declare_parameter("kd_yaw", 0.0004)
        self.declare_parameter("max_lateral_speed", 2.0)
        self.declare_parameter("max_vertical_speed", 1.2)
        self.declare_parameter("max_yaw_rate", 0.35)
        self.declare_parameter("center_x_px", 35.0)
        self.declare_parameter("center_y_px", 65.0)
        self.declare_parameter("slow_x_px", 300.0)
        self.declare_parameter("slow_y_px", 260.0)
        self.declare_parameter("rate_slow_px_s", 1000.0)
        self.declare_parameter("min_area", 100.0)
        self.declare_parameter("derivative_alpha", 0.35)
        self.declare_parameter("vertical_error_alpha", 0.18)
        self.declare_parameter("lateral_error_alpha", 0.30)
        self.declare_parameter("vertical_deadband_px", 18.0)
        self.declare_parameter("lateral_deadband_px", 6.0)
        self.declare_parameter("max_vertical_accel", 0.8)
        self.declare_parameter("max_lateral_accel", 1.5)
        self.declare_parameter("enable_vertical_area_schedule", True)
        self.declare_parameter("near_area_px", 9000.0)
        self.declare_parameter("far_area_px", 2500.0)
        self.declare_parameter("near_vertical_gain_scale", 0.35)
        self.declare_parameter("near_vertical_speed_scale", 0.35)
        self.declare_parameter("near_vertical_accel_scale", 0.35)
        self.declare_parameter("enable_terminal_dash", True)
        self.declare_parameter("terminal_area_px", 45000.0)
        self.declare_parameter("terminal_min_forward_scale", 0.85)
        self.declare_parameter("terminal_lateral_scale", 0.20)
        self.declare_parameter("terminal_vertical_scale", 0.15)
        self.declare_parameter("terminal_yaw_scale", 0.20)
        self.declare_parameter("target_timeout_sec", 0.35)
        self.declare_parameter("enable_approach_guard", False)
        self.declare_parameter("area_drop_epsilon", 40.0)
        self.declare_parameter("area_drop_frames", 3)
        self.declare_parameter("guard_hold_frames", 8)

        self.declare_parameter("image_width", 1280.0)
        self.declare_parameter("image_height", 960.0)
        self.declare_parameter("horizontal_fov_rad", 1.74)
        self.declare_parameter("enable_attitude_compensation", True)
        self.declare_parameter("pitch_pixel_gain", 0.70)
        self.declare_parameter("roll_pixel_gain", 0.70)
        self.declare_parameter("pitch_sign", 1.0)
        self.declare_parameter("roll_sign", 1.0)
        self.declare_parameter("max_virtual_offset_px", 260.0)

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
        self.vertical_error_alpha = float(self.get_parameter("vertical_error_alpha").value)
        self.lateral_error_alpha = float(self.get_parameter("lateral_error_alpha").value)
        self.vertical_deadband_px = float(self.get_parameter("vertical_deadband_px").value)
        self.lateral_deadband_px = float(self.get_parameter("lateral_deadband_px").value)
        self.max_vertical_accel = float(self.get_parameter("max_vertical_accel").value)
        self.max_lateral_accel = float(self.get_parameter("max_lateral_accel").value)
        self.enable_vertical_area_schedule = bool(self.get_parameter("enable_vertical_area_schedule").value)
        self.near_area_px = float(self.get_parameter("near_area_px").value)
        self.far_area_px = float(self.get_parameter("far_area_px").value)
        self.near_vertical_gain_scale = float(self.get_parameter("near_vertical_gain_scale").value)
        self.near_vertical_speed_scale = float(self.get_parameter("near_vertical_speed_scale").value)
        self.near_vertical_accel_scale = float(self.get_parameter("near_vertical_accel_scale").value)
        self.enable_terminal_dash = bool(self.get_parameter("enable_terminal_dash").value)
        self.terminal_area_px = float(self.get_parameter("terminal_area_px").value)
        self.terminal_min_forward_scale = float(self.get_parameter("terminal_min_forward_scale").value)
        self.terminal_lateral_scale = float(self.get_parameter("terminal_lateral_scale").value)
        self.terminal_vertical_scale = float(self.get_parameter("terminal_vertical_scale").value)
        self.terminal_yaw_scale = float(self.get_parameter("terminal_yaw_scale").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.enable_approach_guard = bool(self.get_parameter("enable_approach_guard").value)
        self.area_drop_epsilon = float(self.get_parameter("area_drop_epsilon").value)
        self.area_drop_frames = int(self.get_parameter("area_drop_frames").value)
        self.guard_hold_frames = int(self.get_parameter("guard_hold_frames").value)

        self.image_width = float(self.get_parameter("image_width").value)
        self.image_height = float(self.get_parameter("image_height").value)
        self.horizontal_fov_rad = float(self.get_parameter("horizontal_fov_rad").value)
        self.enable_attitude_compensation = bool(self.get_parameter("enable_attitude_compensation").value)
        self.pitch_pixel_gain = float(self.get_parameter("pitch_pixel_gain").value)
        self.roll_pixel_gain = float(self.get_parameter("roll_pixel_gain").value)
        self.pitch_sign = float(self.get_parameter("pitch_sign").value)
        self.roll_sign = float(self.get_parameter("roll_sign").value)
        self.max_virtual_offset_px = float(self.get_parameter("max_virtual_offset_px").value)

        self.fx = self.image_width / (2.0 * math.tan(self.horizontal_fov_rad / 2.0))
        vertical_fov = 2.0 * math.atan(math.tan(self.horizontal_fov_rad / 2.0) * self.image_height / self.image_width)
        self.fy = self.image_height / (2.0 * math.tan(vertical_fov / 2.0))

        self.last_virtual_error = None
        self.last_time = None
        self.filtered_error_x = 0.0
        self.filtered_error_y = 0.0
        self.have_filtered_error = False
        self.filtered_x_dot = 0.0
        self.filtered_y_dot = 0.0
        self.last_cmd_time = None
        self.last_cmd_y = 0.0
        self.last_cmd_z = 0.0
        self.last_area = None
        self.area_drop_count = 0
        self.guard_hold_count = 0
        self.last_target_time = None
        self.captured = False
        self.state = "SEARCH"
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.have_attitude = False

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.state_pub = self.create_publisher(String, "/vision/tracking_state", 10)
        self.virtual_error_pub = self.create_publisher(Point, VIRTUAL_ERROR_TOPIC, 10)
        self.create_subscription(Point, ERROR_TOPIC, self.error_callback, 10)
        self.create_subscription(Bool, "/vision/captured", self.captured_callback, 10)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_callback, px4_qos)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(f"Subscribed to {ERROR_TOPIC} and /fmu/out/vehicle_attitude")
        self.get_logger().info(
            "Publishing attitude-compensated virtual bearing guidance commands "
            f"(fx={self.fx:.1f}, fy={self.fy:.1f})"
        )

    def attitude_callback(self, msg):
        self.roll, self.pitch, self.yaw = self.quaternion_to_euler(
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        )
        self.have_attitude = True

    def error_callback(self, raw_error):
        if self.captured:
            self.publish_stop("CAPTURED")
            return

        now = self.get_clock().now()
        self.last_target_time = now

        virtual_error, offset_x, offset_y = self.compensate_error(raw_error)
        control_error = self.filter_control_error(virtual_error)
        self.virtual_error_pub.publish(virtual_error)
        x_dot, y_dot = self.estimate_error_rate(control_error, now)

        guarded = self.update_approach_guard(raw_error.z)
        forward_scale = self.forward_scale(control_error, x_dot, y_dot)
        terminal = self.is_terminal_dash(raw_error.z)
        if terminal:
            forward_scale = max(forward_scale, self.terminal_min_forward_scale)
        if guarded:
            forward_scale = 0.0

        cmd = Twist()
        cmd.linear.x = self._clamp(
            self.base_forward_speed * forward_scale,
            0.0,
            self.max_forward_speed,
        )

        lateral = self.kp_lateral * control_error.x + self.kd_lateral * x_dot
        vertical_schedule = self.vertical_schedule_scale(raw_error.z)
        vertical = vertical_schedule["gain"] * (self.kp_vertical * control_error.y + self.kd_vertical * y_dot)
        yaw_rate = -(self.kp_yaw * control_error.x + self.kd_yaw * x_dot)

        if terminal:
            lateral *= self.terminal_lateral_scale
            vertical *= self.terminal_vertical_scale
            yaw_rate *= self.terminal_yaw_scale

        lateral = self._clamp(lateral, -self.max_lateral_speed, self.max_lateral_speed)
        vertical_limit = self.max_vertical_speed * vertical_schedule["speed"]
        vertical = self._clamp(vertical, -vertical_limit, vertical_limit)
        cmd.linear.y, cmd.linear.z = self.rate_limit_lateral_vertical(
            lateral,
            vertical,
            now,
            vertical_schedule["accel"],
        )
        cmd.angular.z = self._clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)

        self.state = self.select_state(control_error, forward_scale, guarded, terminal)
        self.publish_state()
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            "state={} raw=({:.1f},{:.1f}) virt=({:.1f},{:.1f}) ctrl=({:.1f},{:.1f}) off=({:.1f},{:.1f}) "
            "att(r,p)=({:.1f},{:.1f}) rate=({:.1f},{:.1f}) area={:.0f} terminal={} vz_scale=({:.2f},{:.2f},{:.2f}) scale={:.2f} cmd=({:.2f},{:.2f},{:.2f},{:.2f})".format(
                self.state,
                raw_error.x,
                raw_error.y,
                virtual_error.x,
                virtual_error.y,
                control_error.x,
                control_error.y,
                offset_x,
                offset_y,
                math.degrees(self.roll),
                math.degrees(self.pitch),
                x_dot,
                y_dot,
                raw_error.z,
                terminal,
                vertical_schedule["gain"],
                vertical_schedule["speed"],
                vertical_schedule["accel"],
                forward_scale,
                cmd.linear.x,
                cmd.linear.y,
                cmd.linear.z,
                cmd.angular.z,
            ),
            throttle_duration_sec=0.5,
        )

    def compensate_error(self, raw_error):
        offset_x = 0.0
        offset_y = 0.0

        if self.enable_attitude_compensation and self.have_attitude:
            offset_x = self.roll_sign * self.roll_pixel_gain * self.fx * math.tan(self.roll)
            offset_y = self.pitch_sign * self.pitch_pixel_gain * self.fy * math.tan(self.pitch)
            offset_x = self._clamp(offset_x, -self.max_virtual_offset_px, self.max_virtual_offset_px)
            offset_y = self._clamp(offset_y, -self.max_virtual_offset_px, self.max_virtual_offset_px)

        virtual_error = Point()
        virtual_error.x = float(raw_error.x - offset_x)
        virtual_error.y = float(raw_error.y - offset_y)
        virtual_error.z = float(raw_error.z)
        return virtual_error, offset_x, offset_y

    def is_terminal_dash(self, area):
        return self.enable_terminal_dash and area >= self.terminal_area_px

    def filter_control_error(self, virtual_error):
        if not self.have_filtered_error:
            self.filtered_error_x = virtual_error.x
            self.filtered_error_y = virtual_error.y
            self.have_filtered_error = True
        else:
            ax = self._clamp(self.lateral_error_alpha, 0.0, 1.0)
            ay = self._clamp(self.vertical_error_alpha, 0.0, 1.0)
            self.filtered_error_x = ax * virtual_error.x + (1.0 - ax) * self.filtered_error_x
            self.filtered_error_y = ay * virtual_error.y + (1.0 - ay) * self.filtered_error_y

        control_error = Point()
        control_error.x = self.apply_deadband(self.filtered_error_x, self.lateral_deadband_px)
        control_error.y = self.apply_deadband(self.filtered_error_y, self.vertical_deadband_px)
        control_error.z = virtual_error.z
        return control_error

    def vertical_schedule_scale(self, area):
        if not self.enable_vertical_area_schedule:
            return {"gain": 1.0, "speed": 1.0, "accel": 1.0}

        if self.near_area_px <= self.far_area_px:
            closeness = 0.0
        else:
            closeness = (area - self.far_area_px) / (self.near_area_px - self.far_area_px)
            closeness = self._clamp(closeness, 0.0, 1.0)

        return {
            "gain": self.lerp(1.0, self.near_vertical_gain_scale, closeness),
            "speed": self.lerp(1.0, self.near_vertical_speed_scale, closeness),
            "accel": self.lerp(1.0, self.near_vertical_accel_scale, closeness),
        }

    def rate_limit_lateral_vertical(self, lateral, vertical, now, vertical_accel_scale):
        if self.last_cmd_time is None:
            self.last_cmd_time = now
            self.last_cmd_y = lateral
            self.last_cmd_z = vertical
            return lateral, vertical

        dt = max((now - self.last_cmd_time).nanoseconds / 1e9, 1e-3)
        max_dy = max(self.max_lateral_accel, 0.0) * dt
        max_dz = max(self.max_vertical_accel * vertical_accel_scale, 0.0) * dt
        limited_y = self.last_cmd_y + self._clamp(lateral - self.last_cmd_y, -max_dy, max_dy)
        limited_z = self.last_cmd_z + self._clamp(vertical - self.last_cmd_z, -max_dz, max_dz)

        self.last_cmd_time = now
        self.last_cmd_y = limited_y
        self.last_cmd_z = limited_z
        return limited_y, limited_z

    def estimate_error_rate(self, error, now):
        if self.last_virtual_error is None or self.last_time is None:
            self.last_virtual_error = error
            self.last_time = now
            return 0.0, 0.0

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 1e-3:
            return self.filtered_x_dot, self.filtered_y_dot

        raw_x_dot = (error.x - self.last_virtual_error.x) / dt
        raw_y_dot = (error.y - self.last_virtual_error.y) / dt
        alpha = self._clamp(self.derivative_alpha, 0.0, 1.0)
        self.filtered_x_dot = alpha * raw_x_dot + (1.0 - alpha) * self.filtered_x_dot
        self.filtered_y_dot = alpha * raw_y_dot + (1.0 - alpha) * self.filtered_y_dot

        self.last_virtual_error = error
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
            return self._clamp(1.0 - 0.4 * rate_scale, self.min_forward_scale, 1.0)

        return self._clamp(1.0 - penalty, self.min_forward_scale, 1.0)

    def select_state(self, error, forward_scale, guarded, terminal=False):
        if guarded:
            return "GUARD"
        if error.z < self.min_area:
            return "SEARCH"
        if terminal:
            return "TERMINAL_DASH"
        if abs(error.x) <= self.center_x_px and abs(error.y) <= self.center_y_px:
            return "VIRTUAL_APPROACH"
        if forward_scale > self.min_forward_scale:
            return "VIRTUAL_TRACK"
        return "VIRTUAL_ALIGN"

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
            self.last_virtual_error = None
            self.last_time = None
            self.have_filtered_error = False
            self.filtered_x_dot = 0.0
            self.filtered_y_dot = 0.0
            self.last_cmd_time = None
            self.last_cmd_y = 0.0
            self.last_cmd_z = 0.0
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
    def quaternion_to_euler(w, x, y, z):
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))

    @staticmethod
    def apply_deadband(value, deadband):
        if abs(value) <= deadband:
            return 0.0
        return math.copysign(abs(value) - deadband, value)

    @staticmethod
    def lerp(far_value, near_value, closeness):
        return far_value + (near_value - far_value) * closeness


def main():
    rclpy.init()
    node = AttitudePnBearingServo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
