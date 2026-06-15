#!/usr/bin/env python3

import math
import os
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from px4_msgs.msg import VehicleAttitude, VehicleLocalPosition
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String


ERROR_TOPIC = "/vision/target_error"
VIRTUAL_ERROR_TOPIC = "/vision/virtual_target_error"
CMD_TOPIC = "/vision/cmd_velocity"


class LosTransformerPredictor(Node):
    def __init__(self):
        super().__init__("los_transformer_predictor")

        self.declare_parameter("model_path", os.path.expanduser("~/drone_ws/models/los_transformer_v1/best.pt"))
        self.declare_parameter("image_width", 1280.0)
        self.declare_parameter("image_height", 960.0)
        self.declare_parameter("horizontal_fov_rad", 1.74)
        self.declare_parameter("min_area", 1.0)
        self.declare_parameter("virtual_stale_sec", 0.20)
        self.declare_parameter("cmd_stale_sec", 0.50)
        self.declare_parameter("risk_threshold", 0.5)

        self.model_path = os.path.expanduser(self.get_parameter("model_path").value)
        self.image_width = float(self.get_parameter("image_width").value)
        self.image_height = float(self.get_parameter("image_height").value)
        self.horizontal_fov_rad = float(self.get_parameter("horizontal_fov_rad").value)
        self.min_area = float(self.get_parameter("min_area").value)
        self.virtual_stale_sec = float(self.get_parameter("virtual_stale_sec").value)
        self.cmd_stale_sec = float(self.get_parameter("cmd_stale_sec").value)
        self.risk_threshold = float(self.get_parameter("risk_threshold").value)

        self.fx = self.image_width / (2.0 * math.tan(self.horizontal_fov_rad / 2.0))
        vertical_fov = 2.0 * math.atan(math.tan(self.horizontal_fov_rad / 2.0) * self.image_height / self.image_width)
        self.fy = self.image_height / (2.0 * math.tan(vertical_fov / 2.0))

        self.torch = None
        self.model = None
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None
        self.targets = []
        self.predicts_delta = False
        self.history_len = 20
        self.load_model()

        self.history = deque(maxlen=self.history_len)
        self.latest_virtual_error = None
        self.latest_virtual_time = None
        self.latest_cmd = Twist()
        self.latest_cmd_time = None
        self.last_time = None
        self.last_theta_x = None
        self.last_theta_y = None
        self.last_area_log = None
        self.drone_vx = 0.0
        self.drone_vy = 0.0
        self.drone_vz = 0.0
        self.drone_z = 0.0
        self.drone_roll = 0.0
        self.drone_pitch = 0.0
        self.drone_yaw = 0.0

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.predicted_los_pub = self.create_publisher(Point, "/vision/predicted_los", 10)
        self.predicted_error_pub = self.create_publisher(Point, "/vision/predicted_target_error", 10)
        self.risk_pub = self.create_publisher(Float32, "/vision/risk_probability", 10)
        self.state_pub = self.create_publisher(String, "/vision/transformer_state", 10)

        self.create_subscription(Point, ERROR_TOPIC, self.error_callback, 10)
        self.create_subscription(Point, VIRTUAL_ERROR_TOPIC, self.virtual_error_callback, 10)
        self.create_subscription(Twist, CMD_TOPIC, self.cmd_callback, 10)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.local_position_callback, px4_qos)
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position", self.local_position_callback, px4_qos)
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.attitude_callback, px4_qos)

        self.get_logger().info(
            f"Loaded LOS Transformer predictor from {self.model_path}, history_len={self.history_len}, "
            f"targets={self.targets}, predicts_delta={self.predicts_delta}, fx={self.fx:.1f}, fy={self.fy:.1f}"
        )

    def load_model(self):
        try:
            import torch
            import torch.nn as nn
        except ModuleNotFoundError as exc:
            raise SystemExit("PyTorch is not installed. Install torch before running los_transformer_predictor.") from exc

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        self.target_mean = np.asarray(checkpoint["target_mean"], dtype=np.float32)
        self.target_std = np.asarray(checkpoint["target_std"], dtype=np.float32)
        self.targets = [str(name) for name in checkpoint.get("targets", [])]
        self.predicts_delta = any(name.startswith("delta_") for name in self.targets)
        self.history_len = int(checkpoint["history_len"])
        config = checkpoint.get("config", {})

        self.model = self.create_model(
            input_dim=int(self.feature_mean.shape[0]),
            output_dim=int(self.target_mean.shape[0]),
            history_len=self.history_len,
            d_model=int(config.get("d_model", 64)),
            heads=int(config.get("heads", 4)),
            layers=int(config.get("layers", 2)),
            dropout=float(config.get("dropout", 0.1)),
            nn=nn,
            torch=torch,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @staticmethod
    def create_model(input_dim, output_dim, history_len, d_model, heads, layers, dropout, nn, torch):
        class LosTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input = nn.Linear(input_dim, d_model)
                self.pos = nn.Parameter(torch.zeros(1, history_len, d_model))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.head = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.reg = nn.Linear(d_model, output_dim)
                self.risk = nn.Linear(d_model, 1)

            def forward(self, x):
                h = self.input(x) + self.pos[:, : x.shape[1], :]
                h = self.encoder(h)
                pooled = h[:, -1, :]
                z = self.head(pooled)
                return self.reg(z), self.risk(z)

        return LosTransformer()

    def virtual_error_callback(self, msg):
        self.latest_virtual_error = msg
        self.latest_virtual_time = self.get_clock().now()

    def cmd_callback(self, msg):
        self.latest_cmd = msg
        self.latest_cmd_time = self.get_clock().now()

    def error_callback(self, raw_error):
        if raw_error.z < self.min_area:
            self.publish_state("NO_TARGET")
            return

        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9
        error_for_virtual = self.select_virtual_error(raw_error, now)
        cmd = self.safe_cmd(now)
        feature = self.make_feature(raw_error, error_for_virtual, cmd, now_sec)
        if feature is None:
            self.publish_state("WAIT_FEATURE")
            return

        self.history.append(feature)
        if len(self.history) < self.history_len:
            self.publish_state(f"WARMUP {len(self.history)}/{self.history_len}")
            return

        x = np.asarray(self.history, dtype=np.float32)
        x_norm = (x - self.feature_mean) / self.feature_std
        with self.torch.no_grad():
            batch = self.torch.from_numpy(x_norm[None, :, :])
            pred_norm, risk_logit = self.model(batch)
            pred = pred_norm.numpy()[0] * self.target_std + self.target_mean
            risk = float(self.torch.sigmoid(risk_logit)[0, 0].numpy())

        self.publish_prediction(pred, risk, feature)

    def local_position_callback(self, msg):
        if not msg.z_valid:
            return
        self.drone_vx = float(msg.vy)
        self.drone_vy = float(msg.vx)
        self.drone_vz = float(-msg.vz)
        self.drone_z = float(-msg.z)

    def attitude_callback(self, msg):
        self.drone_roll, self.drone_pitch, self.drone_yaw = self.quaternion_to_euler(
            float(msg.q[0]),
            float(msg.q[1]),
            float(msg.q[2]),
            float(msg.q[3]),
        )

    def select_virtual_error(self, raw_error, now):
        if self.latest_virtual_error is None or self.latest_virtual_time is None:
            return raw_error
        age = (now - self.latest_virtual_time).nanoseconds / 1e9
        if age > self.virtual_stale_sec:
            return raw_error
        return self.latest_virtual_error

    def safe_cmd(self, now):
        if self.latest_cmd_time is None:
            return Twist()
        age = (now - self.latest_cmd_time).nanoseconds / 1e9
        if age > self.cmd_stale_sec:
            return Twist()
        return self.latest_cmd

    def make_feature(self, raw_error, virtual_error, cmd, now_sec):
        theta_x = math.atan2(float(raw_error.x), self.fx)
        theta_y = math.atan2(float(raw_error.y), self.fy)
        area_log = math.log(max(float(raw_error.z), 0.0) + 1.0)
        virtual_theta_x = math.atan2(float(virtual_error.x), self.fx)
        virtual_theta_y = math.atan2(float(virtual_error.y), self.fy)

        theta_dot_x = 0.0
        theta_dot_y = 0.0
        area_dot = 0.0
        if self.last_time is not None:
            dt = max(now_sec - self.last_time, 1e-3)
            theta_dot_x = (theta_x - self.last_theta_x) / dt
            theta_dot_y = (theta_y - self.last_theta_y) / dt
            area_dot = (area_log - self.last_area_log) / dt

        self.last_time = now_sec
        self.last_theta_x = theta_x
        self.last_theta_y = theta_y
        self.last_area_log = area_log

        feature = np.asarray(
            [
                theta_x,
                theta_y,
                theta_dot_x,
                theta_dot_y,
                area_log,
                area_dot,
                virtual_theta_x,
                virtual_theta_y,
                float(cmd.linear.x),
                float(cmd.linear.y),
                float(cmd.linear.z),
                float(cmd.angular.z),
                self.drone_vx,
                self.drone_vy,
                self.drone_vz,
                self.drone_z,
                self.drone_roll,
                self.drone_pitch,
                self.drone_yaw,
            ],
            dtype=np.float32,
        )
        feature = feature[: self.feature_mean.shape[0]]
        if not np.all(np.isfinite(feature)):
            return None
        return feature

    def publish_prediction(self, pred, risk, current_feature):
        if self.predicts_delta:
            theta_x = float(current_feature[0] + pred[0])
            theta_y = float(current_feature[1] + pred[1])
            area_log = float(current_feature[4] + pred[2])
        else:
            theta_x = float(pred[0])
            theta_y = float(pred[1])
            area_log = float(pred[2])

        los_msg = Point()
        los_msg.x = theta_x
        los_msg.y = theta_y
        los_msg.z = area_log
        self.predicted_los_pub.publish(los_msg)

        error_msg = Point()
        error_msg.x = math.tan(theta_x) * self.fx
        error_msg.y = math.tan(theta_y) * self.fy
        error_msg.z = max(math.exp(area_log) - 1.0, 0.0)
        self.predicted_error_pub.publish(error_msg)

        risk_msg = Float32()
        risk_msg.data = risk
        self.risk_pub.publish(risk_msg)

        state = "RISK" if risk >= self.risk_threshold else "OK"
        self.publish_state(
            f"{state} theta=({theta_x:.3f},{theta_y:.3f}) area_log={area_log:.2f} risk={risk:.2f}"
        )

    def publish_state(self, text):
        msg = String()
        msg.data = text
        self.state_pub.publish(msg)
        self.get_logger().info(text, throttle_duration_sec=1.0)

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


def main():
    rclpy.init()
    node = LosTransformerPredictor()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
