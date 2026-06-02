#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node


ERROR_TOPIC = "/vision/target_error"
CMD_TOPIC = "/vision/cmd_velocity"


class VisualServo(Node):
    def __init__(self):
        super().__init__("visual_servo")

        self.declare_parameter("kp_yaw", 0.0020)
        self.declare_parameter("kp_z", -0.0020)
        self.declare_parameter("kp_x", 0.00008)
        self.declare_parameter("desired_area", 16000.0)
        self.declare_parameter("max_yaw_rate", 0.6)
        self.declare_parameter("max_vertical_speed", 0.6)
        self.declare_parameter("max_forward_speed", 1.0)

        self.kp_yaw = self.get_parameter("kp_yaw").value
        self.kp_z = self.get_parameter("kp_z").value
        self.kp_x = self.get_parameter("kp_x").value
        self.desired_area = self.get_parameter("desired_area").value
        self.max_yaw_rate = self.get_parameter("max_yaw_rate").value
        self.max_vertical_speed = self.get_parameter("max_vertical_speed").value
        self.max_forward_speed = self.get_parameter("max_forward_speed").value

        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.error_sub = self.create_subscription(Point, ERROR_TOPIC, self.error_callback, 10)

        self.get_logger().info(f"Subscribed to {ERROR_TOPIC}")
        self.get_logger().info(f"Publishing simulated velocity commands to {CMD_TOPIC}")

    def error_callback(self, error):
        cmd = Twist()

        area_error = self.desired_area - error.z
        cmd.linear.x = self._clamp(self.kp_x * area_error, -self.max_forward_speed, self.max_forward_speed)
        cmd.linear.z = self._clamp(self.kp_z * error.y, -self.max_vertical_speed, self.max_vertical_speed)
        cmd.angular.z = self._clamp(-self.kp_yaw * error.x, -self.max_yaw_rate, self.max_yaw_rate)

        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            "err(px)=({:.1f}, {:.1f}) area={:.1f} -> cmd vx={:.2f} vz={:.2f} yaw_rate={:.2f}".format(
                error.x,
                error.y,
                error.z,
                cmd.linear.x,
                cmd.linear.z,
                cmd.angular.z,
            ),
            throttle_duration_sec=0.5,
        )

    @staticmethod
    def _clamp(value, min_value, max_value):
        return max(min_value, min(max_value, value))


def main():
    rclpy.init()
    node = VisualServo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
