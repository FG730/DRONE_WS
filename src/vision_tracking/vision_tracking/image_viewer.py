#!/usr/bin/env python3

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


IMAGE_TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"


class ImageViewer(Node):
    def __init__(self):
        super().__init__("image_viewer")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10,
        )
        self.get_logger().info(f"Subscribed to {IMAGE_TOPIC}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("x500_mono_cam", frame)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ImageViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()