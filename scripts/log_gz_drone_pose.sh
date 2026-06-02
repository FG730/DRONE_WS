#!/usr/bin/env bash
set -euo pipefail

WORLD="${WORLD:-default}"
NAME="${NAME:-x500_mono_cam_0}"
POSE_FILE="${POSE_FILE:-/tmp/x500_pose.csv}"

if ! command -v gz >/dev/null 2>&1; then
  echo "gz command not found. Start Gazebo/PX4 first."
  exit 1
fi

echo "Reading Gazebo pose for ${NAME} from /world/${WORLD}/dynamic_pose/info"
echo "Writing drone truth pose to ${POSE_FILE}"

gz topic -e -t "/world/${WORLD}/dynamic_pose/info" | awk -v target_name="${NAME}" -v pose_file="${POSE_FILE}" '
  $1 == "pose" && $2 == "{" {
    in_pose = 1
    matched = 0
    in_position = 0
    x = 0
    y = 0
    z = 0
  }

  in_pose && $1 == "name:" {
    name = $2
    gsub(/"/, "", name)
    if (name == target_name) {
      matched = 1
    }
  }

  in_pose && matched && $1 == "position" && $2 == "{" {
    in_position = 1
  }

  in_position && $1 == "x:" { x = $2 }
  in_position && $1 == "y:" { y = $2 }
  in_position && $1 == "z:" { z = $2 }

  in_position && $1 == "}" {
    in_position = 0
    cmd = "date +%s.%N"
    cmd | getline ts
    close(cmd)
    print ts "," x "," y "," z > pose_file
    fflush(pose_file)

    count += 1
    if (count % 50 == 0) {
      print "drone_pose," x "," y "," z > "/dev/stderr"
      fflush("/dev/stderr")
    }
  }
'
