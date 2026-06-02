#!/usr/bin/env bash
set -euo pipefail

WORLD="${WORLD:-default}"
NAME="${NAME:-red_target}"
CENTER_X="${CENTER_X:-5.0}"
CENTER_Y="${CENTER_Y:-0.0}"
CENTER_Z="${CENTER_Z:-3.0}"
AMP_X="${AMP_X:-0.8}"
AMP_Y="${AMP_Y:-0.6}"
AMP_Z="${AMP_Z:-0.2}"
PERIOD="${PERIOD:-16.0}"
DT="${DT:-0.1}"
SPAWN_FIRST="${SPAWN_FIRST:-1}"
POSE_FILE="${POSE_FILE:-/tmp/red_target_pose.csv}"

if ! command -v gz >/dev/null 2>&1; then
  echo "gz command not found. Start Gazebo/PX4 first."
  exit 1
fi

if [[ "${SPAWN_FIRST}" == "1" ]]; then
  X="${CENTER_X}" Y="${CENTER_Y}" Z="${CENTER_Z}" NAME="${NAME}" \
    "$(dirname "$0")/spawn_red_target.sh"
fi

echo "Moving ${NAME} around (${CENTER_X}, ${CENTER_Y}, ${CENTER_Z})"
echo "amplitude=(${AMP_X}, ${AMP_Y}, ${AMP_Z}), period=${PERIOD}s, dt=${DT}s"
echo "Writing target truth pose to ${POSE_FILE}"

t=0
while true; do
  x=$(awk -v c="${CENTER_X}" -v a="${AMP_X}" -v t="${t}" -v p="${PERIOD}" 'BEGIN {print c + a * sin(2 * 3.141592653589793 * t / p)}')
  y=$(awk -v c="${CENTER_Y}" -v a="${AMP_Y}" -v t="${t}" -v p="${PERIOD}" 'BEGIN {print c + a * sin(2 * 3.141592653589793 * t / (p * 1.3))}')
  z=$(awk -v c="${CENTER_Z}" -v a="${AMP_Z}" -v t="${t}" -v p="${PERIOD}" 'BEGIN {print c + a * sin(2 * 3.141592653589793 * t / (p * 1.7))}')

  gz service -s "/world/${WORLD}/set_pose" \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 1000 \
    --req "name: \"${NAME}\" position {x: ${x} y: ${y} z: ${z}} orientation {w: 1}" >/dev/null || true

  printf "%s,%s,%s,%s\n" "$(date +%s.%N)" "${x}" "${y}" "${z}" > "${POSE_FILE}"

  sleep "${DT}"
  t=$(awk -v t="${t}" -v dt="${DT}" 'BEGIN {print t + dt}')
done
