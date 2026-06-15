#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point, Twist
from px4_msgs.msg import VehicleAttitude
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


ERROR_TOPIC = "/vision/target_error"
VIRTUAL_ERROR_TOPIC = "/vision/virtual_target_error"
CMD_TOPIC = "/vision/cmd_velocity"


class LosRateBearingServo(Node):
    def __init__(self):
        super().__init__("los_rate_bearing_servo")

        self.declare_parameter("fixed_forward_speed", 6.0)
        self.declare_parameter("max_forward_speed", 8.0)
        self.declare_parameter("min_forward_scale", 0.45)
        self.declare_parameter("angle_slow_x_rad", 0.42)
        self.declare_parameter("angle_slow_y_rad", 0.34)
        self.declare_parameter("los_rate_slow_rad_s", 1.0)
        self.declare_parameter("angle_gain_lateral", 3.8)
        self.declare_parameter("angle_gain_vertical", 3.2)
        self.declare_parameter("navigation_gain_lateral", 2.8)
        self.declare_parameter("navigation_gain_vertical", 2.2)
        self.declare_parameter("yaw_angle_gain", 0.85)
        self.declare_parameter("yaw_rate_gain", 0.16)
        self.declare_parameter("max_lateral_speed", 3.0)
        self.declare_parameter("max_vertical_speed", 2.2)
        self.declare_parameter("max_yaw_rate", 0.45)
        self.declare_parameter("max_lateral_accel", 4.0)
        self.declare_parameter("max_vertical_accel", 3.0)
        self.declare_parameter("max_yaw_accel", 1.2)
        self.declare_parameter("angle_alpha", 0.55)
        self.declare_parameter("los_rate_alpha", 0.30)
        self.declare_parameter("angle_deadband_x_rad", 0.006)
        self.declare_parameter("angle_deadband_y_rad", 0.006)
        self.declare_parameter("min_area", 100.0)
        self.declare_parameter("target_timeout_sec", 0.35)
        self.declare_parameter("enable_terminal_dash", True)
        self.declare_parameter("close_area_px", 4500.0)
        self.declare_parameter("close_center_x_rad", 0.26)
        self.declare_parameter("close_center_y_rad", 0.22)
        self.declare_parameter("close_min_forward_scale", 0.22)
        self.declare_parameter("close_max_forward_scale", 0.62)
        self.declare_parameter("close_correction_scale", 1.15)
        self.declare_parameter("terminal_area_px", 18000.0)
        self.declare_parameter("terminal_center_x_rad", 0.16)
        self.declare_parameter("terminal_center_y_rad", 0.12)
        self.declare_parameter("terminal_min_forward_scale", 0.80)
        self.declare_parameter("terminal_offcenter_forward_scale", 0.45)
        self.declare_parameter("terminal_correction_scale", 0.55)
        self.declare_parameter("enable_keep_in_view", True)
        self.declare_parameter("keep_in_view_begin_ratio", 0.65)
        self.declare_parameter("keep_in_view_full_ratio", 1.0)
        self.declare_parameter("edge_angle_x_rad", 0.58)
        self.declare_parameter("edge_angle_y_rad", 0.48)
        self.declare_parameter("edge_los_rate_rad_s", 1.35)
        self.declare_parameter("edge_forward_scale", 0.12)
        self.declare_parameter("edge_correction_scale", 1.45)
        self.declare_parameter("enable_predictive_terminal", False)
        self.declare_parameter("predict_time_sec", 0.35)
        self.declare_parameter("predict_start_area_px", 12000.0)
        self.declare_parameter("predict_full_area_px", 36000.0)
        self.declare_parameter("predict_rate_trigger_rad_s", 0.75)
        self.declare_parameter("predict_rate_full_rad_s", 1.80)
        self.declare_parameter("predict_center_x_rad", 0.20)
        self.declare_parameter("predict_center_y_rad", 0.16)
        self.declare_parameter("predict_max_angle_rad", 0.78)
        self.declare_parameter("predict_forward_scale", 0.08)
        self.declare_parameter("predict_lateral_boost", 1.55)
        self.declare_parameter("predict_vertical_boost", 1.10)
        self.declare_parameter("predict_yaw_scale", 0.65)
        self.declare_parameter("enable_transformer_guidance", False)
        self.declare_parameter("transformer_alpha", 0.20)
        self.declare_parameter("transformer_alpha_x", 0.0)
        self.declare_parameter("transformer_alpha_y", 0.12)
        self.declare_parameter("transformer_start_area_px", 8000.0)
        self.declare_parameter("transformer_stale_sec", 0.30)
        self.declare_parameter("transformer_max_delta_rad", 0.20)
        self.declare_parameter("transformer_consistency_gate", True)
        self.declare_parameter("transformer_rate_deadband_rad_s", 0.03)
        self.declare_parameter("transformer_guidance_mode", "angle")
        self.declare_parameter("transformer_horizon_sec", 0.30)
        self.declare_parameter("transformer_velocity_gain_lateral", 1.0)
        self.declare_parameter("transformer_velocity_gain_vertical", 1.0)

        self.declare_parameter("image_width", 1280.0)
        self.declare_parameter("image_height", 960.0)
        self.declare_parameter("horizontal_fov_rad", 1.74)
        self.declare_parameter("enable_attitude_compensation", True)
        self.declare_parameter("pitch_pixel_gain", 0.85)
        self.declare_parameter("roll_pixel_gain", 0.80)
        self.declare_parameter("pitch_sign", 1.0)
        self.declare_parameter("roll_sign", 1.0)
        self.declare_parameter("max_virtual_offset_px", 220.0)

        self.fixed_forward_speed = float(self.get_parameter("fixed_forward_speed").value)
        self.max_forward_speed = float(self.get_parameter("max_forward_speed").value)
        self.min_forward_scale = float(self.get_parameter("min_forward_scale").value)
        self.angle_slow_x_rad = float(self.get_parameter("angle_slow_x_rad").value)
        self.angle_slow_y_rad = float(self.get_parameter("angle_slow_y_rad").value)
        self.los_rate_slow_rad_s = float(self.get_parameter("los_rate_slow_rad_s").value)
        self.angle_gain_lateral = float(self.get_parameter("angle_gain_lateral").value)
        self.angle_gain_vertical = float(self.get_parameter("angle_gain_vertical").value)
        self.navigation_gain_lateral = float(self.get_parameter("navigation_gain_lateral").value)
        self.navigation_gain_vertical = float(self.get_parameter("navigation_gain_vertical").value)
        self.yaw_angle_gain = float(self.get_parameter("yaw_angle_gain").value)
        self.yaw_rate_gain = float(self.get_parameter("yaw_rate_gain").value)
        self.max_lateral_speed = float(self.get_parameter("max_lateral_speed").value)
        self.max_vertical_speed = float(self.get_parameter("max_vertical_speed").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.max_lateral_accel = float(self.get_parameter("max_lateral_accel").value)
        self.max_vertical_accel = float(self.get_parameter("max_vertical_accel").value)
        self.max_yaw_accel = float(self.get_parameter("max_yaw_accel").value)
        self.angle_alpha = float(self.get_parameter("angle_alpha").value)
        self.los_rate_alpha = float(self.get_parameter("los_rate_alpha").value)
        self.angle_deadband_x_rad = float(self.get_parameter("angle_deadband_x_rad").value)
        self.angle_deadband_y_rad = float(self.get_parameter("angle_deadband_y_rad").value)
        self.min_area = float(self.get_parameter("min_area").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.enable_terminal_dash = bool(self.get_parameter("enable_terminal_dash").value)
        self.close_area_px = float(self.get_parameter("close_area_px").value)
        self.close_center_x_rad = float(self.get_parameter("close_center_x_rad").value)
        self.close_center_y_rad = float(self.get_parameter("close_center_y_rad").value)
        self.close_min_forward_scale = float(self.get_parameter("close_min_forward_scale").value)
        self.close_max_forward_scale = float(self.get_parameter("close_max_forward_scale").value)
        self.close_correction_scale = float(self.get_parameter("close_correction_scale").value)
        self.terminal_area_px = float(self.get_parameter("terminal_area_px").value)
        self.terminal_center_x_rad = float(self.get_parameter("terminal_center_x_rad").value)
        self.terminal_center_y_rad = float(self.get_parameter("terminal_center_y_rad").value)
        self.terminal_min_forward_scale = float(self.get_parameter("terminal_min_forward_scale").value)
        self.terminal_offcenter_forward_scale = float(self.get_parameter("terminal_offcenter_forward_scale").value)
        self.terminal_correction_scale = float(self.get_parameter("terminal_correction_scale").value)
        self.enable_keep_in_view = bool(self.get_parameter("enable_keep_in_view").value)
        self.keep_in_view_begin_ratio = float(self.get_parameter("keep_in_view_begin_ratio").value)
        self.keep_in_view_full_ratio = float(self.get_parameter("keep_in_view_full_ratio").value)
        self.edge_angle_x_rad = float(self.get_parameter("edge_angle_x_rad").value)
        self.edge_angle_y_rad = float(self.get_parameter("edge_angle_y_rad").value)
        self.edge_los_rate_rad_s = float(self.get_parameter("edge_los_rate_rad_s").value)
        self.edge_forward_scale = float(self.get_parameter("edge_forward_scale").value)
        self.edge_correction_scale = float(self.get_parameter("edge_correction_scale").value)
        self.enable_predictive_terminal = bool(self.get_parameter("enable_predictive_terminal").value)
        self.predict_time_sec = float(self.get_parameter("predict_time_sec").value)
        self.predict_start_area_px = float(self.get_parameter("predict_start_area_px").value)
        self.predict_full_area_px = float(self.get_parameter("predict_full_area_px").value)
        self.predict_rate_trigger_rad_s = float(self.get_parameter("predict_rate_trigger_rad_s").value)
        self.predict_rate_full_rad_s = float(self.get_parameter("predict_rate_full_rad_s").value)
        self.predict_center_x_rad = float(self.get_parameter("predict_center_x_rad").value)
        self.predict_center_y_rad = float(self.get_parameter("predict_center_y_rad").value)
        self.predict_max_angle_rad = float(self.get_parameter("predict_max_angle_rad").value)
        self.predict_forward_scale = float(self.get_parameter("predict_forward_scale").value)
        self.predict_lateral_boost = float(self.get_parameter("predict_lateral_boost").value)
        self.predict_vertical_boost = float(self.get_parameter("predict_vertical_boost").value)
        self.predict_yaw_scale = float(self.get_parameter("predict_yaw_scale").value)
        self.enable_transformer_guidance = bool(self.get_parameter("enable_transformer_guidance").value)
        self.transformer_alpha = float(self.get_parameter("transformer_alpha").value)
        self.transformer_alpha_x = float(self.get_parameter("transformer_alpha_x").value)
        self.transformer_alpha_y = float(self.get_parameter("transformer_alpha_y").value)
        self.transformer_start_area_px = float(self.get_parameter("transformer_start_area_px").value)
        self.transformer_stale_sec = float(self.get_parameter("transformer_stale_sec").value)
        self.transformer_max_delta_rad = float(self.get_parameter("transformer_max_delta_rad").value)
        self.transformer_consistency_gate = bool(self.get_parameter("transformer_consistency_gate").value)
        self.transformer_rate_deadband_rad_s = float(self.get_parameter("transformer_rate_deadband_rad_s").value)
        self.transformer_guidance_mode = str(self.get_parameter("transformer_guidance_mode").value)
        self.transformer_horizon_sec = float(self.get_parameter("transformer_horizon_sec").value)
        self.transformer_velocity_gain_lateral = float(self.get_parameter("transformer_velocity_gain_lateral").value)
        self.transformer_velocity_gain_vertical = float(self.get_parameter("transformer_velocity_gain_vertical").value)

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

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.have_attitude = False
        self.have_angle = False
        self.filtered_los_x = 0.0
        self.filtered_los_y = 0.0
        self.filtered_los_rate_x = 0.0
        self.filtered_los_rate_y = 0.0
        self.last_los_x = None
        self.last_los_y = None
        self.last_time = None
        self.last_cmd_time = None
        self.last_cmd_y = 0.0
        self.last_cmd_z = 0.0
        self.last_cmd_yaw = 0.0
        self.last_target_time = None
        self.latest_transformer_los = None
        self.latest_transformer_time = None
        self.captured = False
        self.state = "SEARCH"

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
        self.create_subscription(Point, "/vision/predicted_los", self.predicted_los_callback, 10)
        self.create_subscription(Bool, "/vision/captured", self.captured_callback, 10)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_callback, px4_qos)
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info(f"Subscribed to {ERROR_TOPIC} and /fmu/out/vehicle_attitude")
        self.get_logger().info(
            "Publishing LOS-rate bearing guidance commands "
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

    def predicted_los_callback(self, msg):
        self.latest_transformer_los = msg
        self.latest_transformer_time = self.get_clock().now()

    def error_callback(self, raw_error):
        if self.captured:
            self.publish_stop("CAPTURED")
            return

        now = self.get_clock().now()
        self.last_target_time = now

        virtual_error, offset_x, offset_y = self.compensate_error(raw_error)
        self.virtual_error_pub.publish(virtual_error)

        los_x, los_y = self.pixel_error_to_los(virtual_error)
        los_x, los_y = self.filter_los_angle(los_x, los_y)
        los_rate_x, los_rate_y = self.estimate_los_rate(los_x, los_y, now)
        los_x = self.apply_deadband(los_x, self.angle_deadband_x_rad)
        los_y = self.apply_deadband(los_y, self.angle_deadband_y_rad)

        close_area = self.enable_terminal_dash and raw_error.z >= self.close_area_px
        terminal_area = self.enable_terminal_dash and raw_error.z >= self.terminal_area_px
        close_alignment = self.alignment_ratio(los_x, los_y, self.close_center_x_rad, self.close_center_y_rad)
        terminal_alignment = self.alignment_ratio(los_x, los_y, self.terminal_center_x_rad, self.terminal_center_y_rad)
        pred_los_x, pred_los_y = self.predict_los(los_x, los_y, los_rate_x, los_rate_y)
        predict_weight, predict_risk = self.predictive_terminal_weight(
            raw_error.z,
            pred_los_x,
            pred_los_y,
            los_rate_x,
            los_rate_y,
        )
        guide_los_x = self.lerp(los_x, pred_los_x, predict_weight)
        guide_los_y = self.lerp(los_y, pred_los_y, predict_weight)
        transformer_weight_x, transformer_weight_y, transformer_delta_x, transformer_delta_y = self.transformer_guidance_delta(
            los_x,
            los_y,
            los_rate_x,
            los_rate_y,
            raw_error.z,
            now,
        )
        if self.transformer_guidance_mode == "angle":
            guide_los_x += transformer_weight_x * transformer_delta_x
            guide_los_y += transformer_weight_y * transformer_delta_y
        guide_los_x = self._clamp(guide_los_x, -self.predict_max_angle_rad, self.predict_max_angle_rad)
        guide_los_y = self._clamp(guide_los_y, -self.predict_max_angle_rad, self.predict_max_angle_rad)
        close_ready = close_area and close_alignment <= 1.0
        terminal_ready = (
            terminal_area
            and terminal_alignment <= 1.0
        )
        forward_scale = self.forward_scale(los_x, los_y, los_rate_x, los_rate_y, raw_error.z)
        if terminal_ready:
            forward_scale = max(forward_scale, self.terminal_min_forward_scale)
        elif terminal_area:
            offcenter_scale = self.close_forward_scale(terminal_alignment)
            forward_scale = min(forward_scale, self.terminal_offcenter_forward_scale, offcenter_scale)
        elif close_area:
            forward_scale = min(forward_scale, self.close_forward_scale(close_alignment))

        keep_ratio = self.keep_in_view_ratio(los_x, los_y, los_rate_x, los_rate_y)
        keep_weight = self.keep_in_view_weight(keep_ratio)
        if self.enable_keep_in_view and keep_weight > 0.0:
            keep_forward_scale = self.lerp(1.0, self.edge_forward_scale, keep_weight)
            forward_scale = min(forward_scale, keep_forward_scale)

        if self.enable_predictive_terminal and predict_weight > 0.0:
            predict_forward = self.lerp(1.0, self.predict_forward_scale, predict_risk)
            forward_scale = min(forward_scale, predict_forward)

        forward = self._clamp(
            self.fixed_forward_speed * forward_scale,
            0.0,
            self.max_forward_speed,
        )

        correction_scale = self.correction_scale(close_area, terminal_ready, keep_weight)
        predictive_lateral_boost = self.lerp(1.0, self.predict_lateral_boost, predict_weight)
        predictive_vertical_boost = self.lerp(1.0, self.predict_vertical_boost, predict_weight)
        predictive_yaw_scale = self.lerp(1.0, self.predict_yaw_scale, predict_weight)
        lateral = correction_scale * predictive_lateral_boost * (
            self.angle_gain_lateral * guide_los_x
            + self.navigation_gain_lateral * forward * los_rate_x
        )
        vertical = -correction_scale * predictive_vertical_boost * (
            self.angle_gain_vertical * guide_los_y
            + self.navigation_gain_vertical * forward * los_rate_y
        )
        if self.transformer_guidance_mode == "velocity":
            lead_rate_x, lead_rate_y = self.transformer_lead_rates(transformer_delta_x, transformer_delta_y)
            lateral += (
                correction_scale
                * transformer_weight_x
                * self.transformer_velocity_gain_lateral
                * self.navigation_gain_lateral
                * forward
                * lead_rate_x
            )
            vertical -= (
                correction_scale
                * transformer_weight_y
                * self.transformer_velocity_gain_vertical
                * self.navigation_gain_vertical
                * forward
                * lead_rate_y
            )
        yaw_rate = -correction_scale * predictive_yaw_scale * (
            self.yaw_angle_gain * guide_los_x
            + self.yaw_rate_gain * los_rate_x
        )

        lateral = self._clamp(lateral, -self.max_lateral_speed, self.max_lateral_speed)
        vertical = self._clamp(vertical, -self.max_vertical_speed, self.max_vertical_speed)
        yaw_rate = self._clamp(yaw_rate, -self.max_yaw_rate, self.max_yaw_rate)
        lateral, vertical, yaw_rate = self.rate_limit_commands(lateral, vertical, yaw_rate, now)

        cmd = Twist()
        cmd.linear.x = forward
        cmd.linear.y = lateral
        cmd.linear.z = vertical
        cmd.angular.z = yaw_rate

        self.state = self.select_state(
            los_x,
            los_y,
            los_rate_x,
            los_rate_y,
            close_area,
            close_ready,
            terminal_area,
            terminal_ready,
            keep_weight,
            predict_weight,
            raw_error.z,
        )
        self.publish_state()
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            "state={} raw=({:.1f},{:.1f}) virt=({:.1f},{:.1f}) off=({:.1f},{:.1f}) "
            "los=({:.3f},{:.3f}) pred=({:.3f},{:.3f}) los_rate=({:.3f},{:.3f}) "
            "area={:.0f} close={} terminal={} align=({:.2f},{:.2f}) "
            "keep=({:.2f},{:.2f}) predict=({:.2f},{:.2f}) transformer=({:.2f},{:.2f},{:.3f},{:.3f}) "
            "scale={:.2f} corr={:.2f} "
            "cmd=({:.2f},{:.2f},{:.2f},{:.2f})".format(
                self.state,
                raw_error.x,
                raw_error.y,
                virtual_error.x,
                virtual_error.y,
                offset_x,
                offset_y,
                los_x,
                los_y,
                pred_los_x,
                pred_los_y,
                los_rate_x,
                los_rate_y,
                raw_error.z,
                close_area,
                terminal_ready,
                close_alignment,
                terminal_alignment,
                keep_ratio,
                keep_weight,
                predict_weight,
                predict_risk,
                transformer_weight_x,
                transformer_weight_y,
                transformer_delta_x,
                transformer_delta_y,
                forward_scale,
                correction_scale,
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

    def transformer_guidance_delta(self, los_x, los_y, los_rate_x, los_rate_y, area, now):
        if not self.enable_transformer_guidance:
            return 0.0, 0.0, 0.0, 0.0
        if area < self.transformer_start_area_px:
            return 0.0, 0.0, 0.0, 0.0
        if self.latest_transformer_los is None or self.latest_transformer_time is None:
            return 0.0, 0.0, 0.0, 0.0

        age_sec = (now - self.latest_transformer_time).nanoseconds / 1e9
        if age_sec > self.transformer_stale_sec:
            return 0.0, 0.0, 0.0, 0.0

        pred_x = float(self.latest_transformer_los.x)
        pred_y = float(self.latest_transformer_los.y)
        if not math.isfinite(pred_x) or not math.isfinite(pred_y):
            return 0.0, 0.0, 0.0, 0.0

        max_delta = max(self.transformer_max_delta_rad, 1e-3)
        delta_x = self._clamp(pred_x - los_x, -max_delta, max_delta)
        delta_y = self._clamp(pred_y - los_y, -max_delta, max_delta)
        alpha_x = self._clamp(self.transformer_alpha_x, 0.0, 1.0)
        alpha_y = self._clamp(self.transformer_alpha_y, 0.0, 1.0)

        if self.transformer_consistency_gate:
            rate_deadband = max(self.transformer_rate_deadband_rad_s, 0.0)
            if abs(los_rate_x) > rate_deadband and delta_x * los_rate_x < 0.0:
                alpha_x = 0.0
            if abs(los_rate_y) > rate_deadband and delta_y * los_rate_y < 0.0:
                alpha_y = 0.0

        return alpha_x, alpha_y, delta_x, delta_y

    def transformer_lead_rates(self, delta_x, delta_y):
        horizon = max(self.transformer_horizon_sec, 1e-3)
        return delta_x / horizon, delta_y / horizon

    def pixel_error_to_los(self, error):
        return math.atan2(error.x, self.fx), math.atan2(error.y, self.fy)

    def filter_los_angle(self, los_x, los_y):
        alpha = self._clamp(self.angle_alpha, 0.0, 1.0)
        if not self.have_angle:
            self.filtered_los_x = los_x
            self.filtered_los_y = los_y
            self.have_angle = True
        else:
            self.filtered_los_x = alpha * los_x + (1.0 - alpha) * self.filtered_los_x
            self.filtered_los_y = alpha * los_y + (1.0 - alpha) * self.filtered_los_y
        return self.filtered_los_x, self.filtered_los_y

    def estimate_los_rate(self, los_x, los_y, now):
        if self.last_los_x is None or self.last_los_y is None or self.last_time is None:
            self.last_los_x = los_x
            self.last_los_y = los_y
            self.last_time = now
            return 0.0, 0.0

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt <= 1e-3:
            return self.filtered_los_rate_x, self.filtered_los_rate_y

        raw_rate_x = (los_x - self.last_los_x) / dt
        raw_rate_y = (los_y - self.last_los_y) / dt
        alpha = self._clamp(self.los_rate_alpha, 0.0, 1.0)
        self.filtered_los_rate_x = alpha * raw_rate_x + (1.0 - alpha) * self.filtered_los_rate_x
        self.filtered_los_rate_y = alpha * raw_rate_y + (1.0 - alpha) * self.filtered_los_rate_y
        self.last_los_x = los_x
        self.last_los_y = los_y
        self.last_time = now
        return self.filtered_los_rate_x, self.filtered_los_rate_y

    def forward_scale(self, los_x, los_y, los_rate_x, los_rate_y, area):
        if area < self.min_area:
            return 0.0

        x_penalty = abs(los_x) / max(self.angle_slow_x_rad, 1e-3)
        y_penalty = abs(los_y) / max(self.angle_slow_y_rad, 1e-3)
        rate_penalty = math.hypot(los_rate_x, los_rate_y) / max(self.los_rate_slow_rad_s, 1e-3)
        penalty = max(x_penalty, y_penalty, 0.55 * rate_penalty)
        return self._clamp(1.0 - 0.55 * penalty, self.min_forward_scale, 1.0)

    def alignment_ratio(self, los_x, los_y, center_x, center_y):
        x_ratio = abs(los_x) / max(center_x, 1e-3)
        y_ratio = abs(los_y) / max(center_y, 1e-3)
        return max(x_ratio, y_ratio)

    def close_forward_scale(self, alignment):
        if alignment <= 1.0:
            return self.close_max_forward_scale

        penalty = self._clamp((alignment - 1.0) / 2.0, 0.0, 1.0)
        return self.lerp(self.close_max_forward_scale, self.close_min_forward_scale, penalty)

    def predict_los(self, los_x, los_y, los_rate_x, los_rate_y):
        predict_time = max(self.predict_time_sec, 0.0)
        max_angle = max(self.predict_max_angle_rad, 1e-3)
        pred_x = self._clamp(los_x + los_rate_x * predict_time, -max_angle, max_angle)
        pred_y = self._clamp(los_y + los_rate_y * predict_time, -max_angle, max_angle)
        return pred_x, pred_y

    def predictive_terminal_weight(self, area, pred_los_x, pred_los_y, los_rate_x, los_rate_y):
        if not self.enable_predictive_terminal:
            return 0.0, 0.0

        area_weight = self.smoothstep(
            (area - self.predict_start_area_px)
            / max(self.predict_full_area_px - self.predict_start_area_px, 1e-3)
        )
        rate_norm = math.hypot(los_rate_x, los_rate_y)
        rate_weight = self.smoothstep(
            (rate_norm - self.predict_rate_trigger_rad_s)
            / max(self.predict_rate_full_rad_s - self.predict_rate_trigger_rad_s, 1e-3)
        )
        pred_alignment = self.alignment_ratio(
            pred_los_x,
            pred_los_y,
            self.predict_center_x_rad,
            self.predict_center_y_rad,
        )
        alignment_weight = self.smoothstep((pred_alignment - 0.75) / 1.25)

        weight = max(area_weight, min(rate_weight, alignment_weight))
        risk = max(area_weight, rate_weight, alignment_weight)
        return self._clamp(weight, 0.0, 1.0), self._clamp(risk, 0.0, 1.0)

    def keep_in_view_ratio(self, los_x, los_y, los_rate_x, los_rate_y):
        x_ratio = abs(los_x) / max(self.edge_angle_x_rad, 1e-3)
        y_ratio = abs(los_y) / max(self.edge_angle_y_rad, 1e-3)
        rate_ratio = math.hypot(los_rate_x, los_rate_y) / max(self.edge_los_rate_rad_s, 1e-3)
        return max(x_ratio, y_ratio, rate_ratio)

    def keep_in_view_weight(self, keep_ratio):
        if not self.enable_keep_in_view:
            return 0.0
        begin = self.keep_in_view_begin_ratio
        full = max(self.keep_in_view_full_ratio, begin + 1e-3)
        return self._clamp((keep_ratio - begin) / (full - begin), 0.0, 1.0)

    def correction_scale(self, close_area, terminal_ready, keep_weight):
        if terminal_ready:
            base_scale = self.terminal_correction_scale
        elif close_area:
            base_scale = self.close_correction_scale
        else:
            base_scale = 1.0

        if self.enable_keep_in_view and keep_weight > 0.0:
            base_scale *= self.lerp(1.0, self.edge_correction_scale, keep_weight)
        return base_scale

    def rate_limit_commands(self, lateral, vertical, yaw_rate, now):
        if self.last_cmd_time is None:
            self.last_cmd_time = now
            self.last_cmd_y = lateral
            self.last_cmd_z = vertical
            self.last_cmd_yaw = yaw_rate
            return lateral, vertical, yaw_rate

        dt = max((now - self.last_cmd_time).nanoseconds / 1e9, 1e-3)
        limited_y = self.last_cmd_y + self._clamp(
            lateral - self.last_cmd_y,
            -self.max_lateral_accel * dt,
            self.max_lateral_accel * dt,
        )
        limited_z = self.last_cmd_z + self._clamp(
            vertical - self.last_cmd_z,
            -self.max_vertical_accel * dt,
            self.max_vertical_accel * dt,
        )
        limited_yaw = self.last_cmd_yaw + self._clamp(
            yaw_rate - self.last_cmd_yaw,
            -self.max_yaw_accel * dt,
            self.max_yaw_accel * dt,
        )

        self.last_cmd_time = now
        self.last_cmd_y = limited_y
        self.last_cmd_z = limited_z
        self.last_cmd_yaw = limited_yaw
        return limited_y, limited_z, limited_yaw

    def select_state(
        self,
        los_x,
        los_y,
        los_rate_x,
        los_rate_y,
        close_area,
        close_ready,
        terminal_area,
        terminal_ready,
        keep_weight,
        predict_weight,
        area,
    ):
        if area < self.min_area:
            return "SEARCH"
        if predict_weight >= 0.35:
            return "LOS_PREDICT_TERMINAL"
        if keep_weight >= 0.95:
            return "LOS_KEEP_IN_VIEW"
        if terminal_ready:
            return "LOS_TERMINAL"
        if terminal_area:
            return "LOS_NEAR_ALIGN"
        if close_ready:
            return "LOS_CLOSE_PURSUE"
        if close_area:
            return "LOS_CLOSE_ALIGN"
        centered = abs(los_x) <= 0.035 and abs(los_y) <= 0.045
        stable = math.hypot(los_rate_x, los_rate_y) <= 0.18
        if centered and stable:
            return "LOS_INTERCEPT"
        if centered:
            return "LOS_PURSUE"
        return "LOS_ALIGN"

    def watchdog_callback(self):
        if self.captured:
            self.publish_stop("CAPTURED")
            return

        if self.last_target_time is None:
            self.publish_stop("SEARCH")
            return

        age_sec = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
        if age_sec > self.target_timeout_sec:
            self.reset_tracking_memory()
            self.publish_stop("LOST")

    def reset_tracking_memory(self):
        self.have_angle = False
        self.filtered_los_rate_x = 0.0
        self.filtered_los_rate_y = 0.0
        self.last_los_x = None
        self.last_los_y = None
        self.last_time = None
        self.last_cmd_time = None
        self.last_cmd_y = 0.0
        self.last_cmd_z = 0.0
        self.last_cmd_yaw = 0.0

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
    def lerp(start, end, ratio):
        return start + (end - start) * ratio

    @classmethod
    def smoothstep(cls, value):
        value = cls._clamp(value, 0.0, 1.0)
        return value * value * (3.0 - 2.0 * value)


def main():
    rclpy.init()
    node = LosRateBearingServo()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
