# DRONE_WS

PX4 + Gazebo + ROS 2 visual tracking workspace for simulated drone pursuit.

## Main Package

- `src/vision_tracking`: ROS 2 vision tracking, control, logging, plotting, and scoring nodes.
- `scripts`: Gazebo target spawning/motion scripts and tuning helpers.
- `archive_old_nodes`: older controller prototypes kept for comparison.

## External Dependency

This workspace expects `px4_msgs` in `src/px4_msgs`.

```bash
cd ~/drone_ws/src
git clone https://github.com/PX4/px4_msgs.git
```

## Build

```bash
cd ~/drone_ws
colcon build --packages-select vision_tracking
source /opt/ros/humble/setup.bash
source ~/drone_ws/install/setup.bash
```
