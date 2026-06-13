# 无人机视觉追击仿真系统架构

```mermaid
flowchart LR
    subgraph SIM["Gazebo / PX4 SITL 仿真环境"]
        GZ["Gazebo 仿真世界<br/>default.sdf"]
        DRONE["我方无人机模型<br/>x500_mono_cam"]
        CAM["单目相机<br/>/world/.../camera/image"]
        TARGET["红色目标球<br/>直线 / 平面弧线 / 三维弧线 / S 型"]
        TRUTH["目标真值位姿<br/>/tmp/red_target_pose.csv"]

        GZ --> DRONE
        GZ --> TARGET
        DRONE --> CAM
        TARGET --> TRUTH
    end

    subgraph PX4["PX4 飞控 / SITL"]
        PX4CORE["PX4 飞控核心<br/>EKF2 / Commander / Offboard"]
        PX4POSE["PX4 状态话题<br/>vehicle_local_position_v1<br/>vehicle_attitude<br/>vehicle_status_v4"]
        PX4IN["PX4 控制输入话题<br/>trajectory_setpoint<br/>offboard_control_mode<br/>vehicle_command"]
    end

    subgraph ROS["ROS 2 工作空间：drone_ws"]
        BRIDGE["ros_gz_bridge<br/>Gazebo 图像 -> ROS 图像"]
        TRACKER["color_tracker.py<br/>HSV 红色目标分割<br/>质心 + 面积"]
        GUIDANCE["los_rate_bearing_servo.py<br/>视线角 + 视线角速度制导<br/>视野保持 + 末端逻辑"]
        OFFBOARD["px4_visual_offboard.py<br/>仅航向角速度坐标变换<br/>机体系指令 -> PX4 NED 速度设定"]
        LOGGER["experiment_logger.py<br/>CSV 数据记录 + 虚拟捕获半径"]
        PLOT["plot_experiment.py<br/>距离 / 误差 / 指令 / 三维轨迹绘图"]
    end

    subgraph GCS["地面站 / 用户交互"]
        QGC["QGroundControl<br/>解锁 / 起飞 / Hold / 状态监控"]
        USER["操作者 / 研究人员<br/>启动实验 + 分析结果"]
    end

    subgraph AUTO["实验自动化"]
        RUNNER["run_tuning_trial.py<br/>启动桥接 / 识别 / 记录<br/>制导控制 / Offboard / 目标运动"]
        CONFIG["参数配置文件<br/>pixel_error_ibvs_arc3d_baseline.yaml<br/>los_rate_predictive_terminal.yaml"]
    end

    CAM -- "gz.msgs.Image" --> BRIDGE
    BRIDGE -- "sensor_msgs/Image" --> TRACKER
    TRACKER -- "/vision/target_error<br/>像素误差 error_x/error_y + 面积 area" --> GUIDANCE
    PX4POSE -- "姿态补偿<br/>航向角反馈" --> GUIDANCE
    GUIDANCE -- "/vision/cmd_velocity<br/>前向/侧向/垂向速度 + 偏航角速度" --> OFFBOARD
    OFFBOARD -- "/fmu/in/*" --> PX4IN
    PX4IN --> PX4CORE
    PX4CORE --> DRONE
    PX4CORE -- "/fmu/out/*" --> PX4POSE

    TRACKER --> LOGGER
    GUIDANCE --> LOGGER
    PX4POSE --> LOGGER
    TRUTH --> LOGGER
    LOGGER -- "实验数据 CSV" --> PLOT

    QGC <-- "MAVLink UDP" --> PX4CORE
    USER --> QGC
    USER --> RUNNER
    CONFIG --> RUNNER
    RUNNER --> BRIDGE
    RUNNER --> TRACKER
    RUNNER --> LOGGER
    RUNNER --> GUIDANCE
    RUNNER --> OFFBOARD
    RUNNER --> TARGET
```

## 核心数据流

1. Gazebo 负责仿真我方无人机、单目相机和运动红色目标球。
2. `ros_gz_bridge` 将 Gazebo 相机图像转换为 ROS 2 图像话题。
3. `color_tracker.py` 通过 HSV 颜色阈值分割红色目标，输出目标质心、像素误差和面积。
4. `los_rate_bearing_servo.py` 将像素误差转换为视线角和视线角速度，生成前向、侧向、垂向和偏航角速度指令。
5. `px4_visual_offboard.py` 将视觉制导输出的机体系速度指令转换为 PX4 NED 速度设定值。
6. PX4 在 Offboard 模式下执行速度设定值，并驱动 Gazebo 中的无人机运动。
7. `experiment_logger.py` 同步记录图像误差、控制指令、PX4 状态、目标真值、相对距离和捕获状态。

## 核心算法

- 视觉识别：HSV 红色分割、轮廓筛选、质心提取、面积估计。
- 基线制导：`attitude_pn_bearing_servo.py` 中的像素误差 / IBVS 控制方法。
- 新版制导：`los_rate_bearing_servo.py` 中的视线角 + 视线角速度制导方法。
- 视野保持：目标接近图像边缘时降低前进速度，并增强横向/垂向修正。
- 末端逻辑：根据目标面积、视线偏差和捕获半径判断末端追击状态。
- Offboard 接口：仅使用航向角进行机体系到 NED 速度变换，避免机体俯仰导致前进速度耦合到高度方向。

## 通讯链路

- Gazebo 到 ROS 2：`ros_gz_bridge`
- ROS 2 到 PX4：`/fmu/in/trajectory_setpoint`、`/fmu/in/offboard_control_mode`、`/fmu/in/vehicle_command`
- PX4 到 ROS 2：`/fmu/out/vehicle_local_position_v1`、`/fmu/out/vehicle_attitude`、`/fmu/out/vehicle_status_v4`
- QGC 到 PX4：MAVLink UDP
- 目标真值到记录器：`/tmp/red_target_pose.csv`
