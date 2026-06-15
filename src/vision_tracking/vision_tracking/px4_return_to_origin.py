#!/usr/bin/env python3

import math
import time

import rclpy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class Px4ReturnToOrigin(Node):
    def __init__(self):
        super().__init__("px4_return_to_origin")

        self.declare_parameter("target_x", 0.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_altitude", 3.0)
        self.declare_parameter("xy_radius", 0.6)
        self.declare_parameter("z_radius", 0.35)
        self.declare_parameter("max_xy_speed", 3.0)
        self.declare_parameter("max_z_speed", 1.0)
        self.declare_parameter("target_yaw", 0.0)
        self.declare_parameter("yaw_radius", 0.15)
        self.declare_parameter("max_yaw_rate", 0.6)
        self.declare_parameter("kp_yaw", 1.2)
        self.declare_parameter("kp_xy", 0.8)
        self.declare_parameter("kp_z", 0.7)
        self.declare_parameter("hold_sec", 1.0)
        self.declare_parameter("timeout_sec", 45.0)
        self.declare_parameter("mode_retry_sec", 1.0)

        self.target_x = float(self.get_parameter("target_x").value)
        self.target_y = float(self.get_parameter("target_y").value)
        self.target_z = -float(self.get_parameter("target_altitude").value)
        self.xy_radius = float(self.get_parameter("xy_radius").value)
        self.z_radius = float(self.get_parameter("z_radius").value)
        self.max_xy_speed = float(self.get_parameter("max_xy_speed").value)
        self.max_z_speed = float(self.get_parameter("max_z_speed").value)
        self.target_yaw = float(self.get_parameter("target_yaw").value)
        self.yaw_radius = float(self.get_parameter("yaw_radius").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.kp_yaw = float(self.get_parameter("kp_yaw").value)
        self.kp_xy = float(self.get_parameter("kp_xy").value)
        self.kp_z = float(self.get_parameter("kp_z").value)
        self.hold_sec = float(self.get_parameter("hold_sec").value)
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.mode_retry_sec = float(self.get_parameter("mode_retry_sec").value)

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.offboard_pub = self.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", px4_qos)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", px4_qos)
        self.command_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", px4_qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.position_callback, px4_qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.position_callback, px4_qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v4", self.status_callback, px4_qos)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status", self.status_callback, px4_qos)

        self.position = None
        self.heading = None
        self.nav_state = None
        self.start_time = time.monotonic()
        self.last_mode_request_time = 0.0
        self.reached_since = None
        self.finished = False

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(
            "Returning to PX4 local origin: x={:.1f}, y={:.1f}, altitude={:.1f} m, yaw={:.2f} rad".format(
                self.target_x,
                self.target_y,
                -self.target_z,
                self.target_yaw,
            )
        )

    def position_callback(self, msg):
        if not msg.xy_valid or not msg.z_valid:
            return
        self.position = (float(msg.x), float(msg.y), float(msg.z))
        if msg.heading_good_for_control:
            self.heading = float(msg.heading)

    def status_callback(self, msg):
        self.nav_state = msg.nav_state

    def timer_callback(self):
        now_us = self.get_clock().now().nanoseconds // 1000
        self.publish_offboard_control_mode(now_us)

        if self.position is None:
            self.publish_velocity_setpoint(now_us, 0.0, 0.0, 0.0, 0.0)
            if time.monotonic() - self.start_time > self.timeout_sec:
                self.get_logger().warn("Return timeout before local position became available")
                self.finished = True
            return

        vx, vy, vz, yaw_rate, xy_error, z_error, yaw_error = self.return_velocity()
        reached = (
            xy_error <= self.xy_radius
            and abs(z_error) <= self.z_radius
            and abs(yaw_error) <= self.yaw_radius
        )

        if reached:
            self.publish_velocity_setpoint(now_us, 0.0, 0.0, 0.0, 0.0)
            if self.reached_since is None:
                self.reached_since = time.monotonic()
                self.get_logger().info(
                    "Return target reached: xy_error={:.2f} m, z_error={:.2f} m, yaw_error={:.2f} rad".format(
                        xy_error,
                        z_error,
                        yaw_error,
                    )
                )
            if time.monotonic() - self.reached_since >= self.hold_sec:
                self.request_hold()
                self.finished = True
            return

        self.reached_since = None
        self.publish_velocity_setpoint(now_us, vx, vy, vz, yaw_rate)
        self.request_offboard_if_needed()

        if time.monotonic() - self.start_time > self.timeout_sec:
            self.get_logger().warn(
                "Return timeout: xy_error={:.2f} m, z_error={:.2f} m, yaw_error={:.2f} rad".format(
                    xy_error,
                    z_error,
                    yaw_error,
                )
            )
            self.request_hold()
            self.finished = True

        self.get_logger().info(
            "return pos=({:.2f},{:.2f},{:.2f}) err_xy={:.2f} err_z={:.2f} err_yaw={:.2f} vel=({:.2f},{:.2f},{:.2f}) yaw_rate={:.2f}".format(
                self.position[0],
                self.position[1],
                self.position[2],
                xy_error,
                z_error,
                yaw_error,
                vx,
                vy,
                vz,
                yaw_rate,
            ),
            throttle_duration_sec=1.0,
        )

    def return_velocity(self):
        x, y, z = self.position
        dx = self.target_x - x
        dy = self.target_y - y
        dz = self.target_z - z
        xy_error = math.hypot(dx, dy)

        vx = self.kp_xy * dx
        vy = self.kp_xy * dy
        xy_speed = math.hypot(vx, vy)
        if xy_speed > self.max_xy_speed and xy_speed > 1e-6:
            scale = self.max_xy_speed / xy_speed
            vx *= scale
            vy *= scale

        vz = self.clamp(self.kp_z * dz, -self.max_z_speed, self.max_z_speed)
        yaw_error = 0.0
        if self.heading is not None:
            yaw_error = self.wrap_pi(self.target_yaw - self.heading)
        yaw_rate = self.clamp(self.kp_yaw * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)
        return vx, vy, vz, yaw_rate, xy_error, dz, yaw_error

    def request_offboard_if_needed(self):
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            return
        if time.monotonic() - self.last_mode_request_time < self.mode_retry_sec:
            return
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.last_mode_request_time = time.monotonic()
        self.get_logger().info(f"Requested OFFBOARD for return (nav_state={self.nav_state})")

    def request_hold(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=4.0,
            param3=3.0,
        )
        self.get_logger().info("Requested AUTO.LOITER/Hold after return")

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

    def publish_velocity_setpoint(self, timestamp, vx, vy, vz, yaw_rate):
        msg = TrajectorySetpoint()
        msg.timestamp = timestamp
        msg.position = [math.nan, math.nan, math.nan]
        msg.velocity = [float(vx), float(vy), float(vz)]
        msg.acceleration = [math.nan, math.nan, math.nan]
        msg.jerk = [math.nan, math.nan, math.nan]
        msg.yaw = float(self.target_yaw)
        msg.yawspeed = float(yaw_rate)
        self.trajectory_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0):
        msg = VehicleCommand()
        msg.timestamp = self.get_clock().now().nanoseconds // 1000
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
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

    @staticmethod
    def wrap_pi(value):
        return math.atan2(math.sin(value), math.cos(value))


def main():
    rclpy.init()
    node = Px4ReturnToOrigin()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
