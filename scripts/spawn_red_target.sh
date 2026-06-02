#!/usr/bin/env bash
set -euo pipefail

WORLD="${WORLD:-default}"
NAME="${NAME:-red_target}"
X="${X:-3}"
Y="${Y:-0}"
Z="${Z:-1.5}"
RADIUS="${RADIUS:-0.35}"

if ! command -v gz >/dev/null 2>&1; then
  echo "gz command not found. Start Gazebo/PX4 environment first."
  exit 1
fi

echo "Spawning ${NAME} in world ${WORLD} at (${X}, ${Y}, ${Z}), radius=${RADIUS}"

gz service -s "/world/${WORLD}/remove" \
  --reqtype gz.msgs.Entity \
  --reptype gz.msgs.Boolean \
  --timeout 1000 \
  --req "name: \"${NAME}\" type: MODEL" >/dev/null 2>&1 || true

gz service -s "/world/${WORLD}/create" \
  --reqtype gz.msgs.EntityFactory \
  --reptype gz.msgs.Boolean \
  --timeout 3000 \
  --req "sdf: \"<?xml version=\\\"1.0\\\" ?><sdf version=\\\"1.9\\\"><model name=\\\"${NAME}\\\"><static>true</static><pose>${X} ${Y} ${Z} 0 0 0</pose><link name=\\\"link\\\"><visual name=\\\"visual\\\"><geometry><sphere><radius>${RADIUS}</radius></sphere></geometry><material><ambient>1 0 0 1</ambient><diffuse>1 0 0 1</diffuse><specular>0.2 0 0 1</specular></material></visual><collision name=\\\"collision\\\"><geometry><sphere><radius>${RADIUS}</radius></sphere></geometry></collision></link></model></sdf>\""

echo "Done."
