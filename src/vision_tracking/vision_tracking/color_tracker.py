#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image


IMAGE_TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"


class ColorTracker(Node):
    def __init__(self):
        super().__init__("color_tracker")
        self.bridge = CvBridge()
        self.target_pub = self.create_publisher(Point, "/vision/target_center", 10)
        self.error_pub = self.create_publisher(Point, "/vision/target_error", 10)
        self.image_sub = self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.get_logger().info(f"Tracking red target from {IMAGE_TOPIC}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        image_height, image_width = frame.shape[:2]
        image_center_x = image_width // 2
        image_center_y = image_height // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red_1 = np.array([0, 100, 80])
        upper_red_1 = np.array([10, 255, 255])
        lower_red_2 = np.array([170, 100, 80])
        upper_red_2 = np.array([180, 255, 255])

        mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        mask = cv2.bitwise_or(mask_1, mask_2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 100:
                moments = cv2.moments(largest)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    x, y, w, h = cv2.boundingRect(largest)

                    target = Point()
                    target.x = float(cx)
                    target.y = float(cy)
                    target.z = float(area)
                    self.target_pub.publish(target)

                    error_x = cx - image_center_x
                    error_y = cy - image_center_y
                    error = Point()
                    error.x = float(error_x)
                    error.y = float(error_y)
                    error.z = float(area)
                    self.error_pub.publish(error)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.line(frame, (image_center_x, image_center_y), (cx, cy), (0, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f"target ({cx}, {cy}) err=({error_x}, {error_y}) area={area:.0f}",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        cv2.drawMarker(
            frame,
            (image_center_x, image_center_y),
            (255, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=28,
            thickness=2,
        )
        cv2.imshow("color_tracker", frame)
        cv2.imshow("red_mask", mask)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ColorTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
