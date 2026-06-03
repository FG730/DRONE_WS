#!/usr/bin/env python3

import argparse
import math
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Move a Gazebo red target with speed-limited trajectories.")
    parser.add_argument("--world", default="default")
    parser.add_argument("--name", default="red_target")
    parser.add_argument("--mode", choices=["straight", "arc", "s_curve"], default="s_curve")
    parser.add_argument("--speed", type=float, default=3.0, help="Nominal target forward speed in m/s.")
    parser.add_argument("--start-x", type=float, default=12.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=4.0)
    parser.add_argument("--heading-deg", type=float, default=180.0, help="Target travel direction in Gazebo XY plane.")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--duration", type=float, default=0.0, help="Move duration in seconds. Use 0 to run until Ctrl+C.")
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--arc-radius", type=float, default=60.0, help="Arc radius in meters for arc mode.")
    parser.add_argument("--arc-direction", choices=["left", "right"], default="left")
    parser.add_argument("--s-amp-y", type=float, default=1.5)
    parser.add_argument("--s-period", type=float, default=6.0)
    parser.add_argument("--z-amp", type=float, default=0.25)
    parser.add_argument("--z-period", type=float, default=9.0)
    parser.add_argument("--pose-file", default="/tmp/red_target_pose.csv")
    args = parser.parse_args()

    check_gz()
    spawn_target(args)

    pose_file = Path(args.pose_file).expanduser()
    heading = math.radians(args.heading_deg)
    forward = (math.cos(heading), math.sin(heading))
    lateral = (-math.sin(heading), math.cos(heading))

    print(
        f"Moving {args.name}: mode={args.mode}, speed={args.speed:.2f} m/s, "
        f"start=({args.start_x:.1f}, {args.start_y:.1f}, {args.start_z:.1f}), "
        f"heading={args.heading_deg:.1f} deg"
    )
    print(f"Writing target truth pose to {pose_file}")

    start_time = time.monotonic()
    next_time = start_time

    try:
        while True:
            now = time.monotonic()
            t = now - start_time
            if args.duration > 0.0 and t > args.duration:
                print(f"Target motion duration reached at t={t:.1f}s; target remains at last pose.")
                break

            x, y, z = target_pose(args, t, forward, lateral)
            set_pose(args.world, args.name, x, y, z)
            pose_file.write_text(f"{time.time():.9f},{x:.6f},{y:.6f},{z:.6f}\n")

            next_time += args.dt
            sleep_time = max(0.0, next_time - time.monotonic())
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Target motion stopped by Ctrl+C; target remains at last pose.")


def check_gz():
    result = subprocess.run(["which", "gz"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0:
        raise SystemExit("gz command not found. Start Gazebo/PX4 first.")


def spawn_target(args):
    sdf = (
        '<?xml version="1.0" ?>'
        '<sdf version="1.9">'
        f'<model name="{args.name}">'
        '<static>true</static>'
        f'<pose>{args.start_x} {args.start_y} {args.start_z} 0 0 0</pose>'
        '<link name="link">'
        '<visual name="visual">'
        '<geometry><sphere>'
        f'<radius>{args.radius}</radius>'
        '</sphere></geometry>'
        '<material>'
        '<ambient>1 0 0 1</ambient>'
        '<diffuse>1 0 0 1</diffuse>'
        '<specular>0.2 0 0 1</specular>'
        '</material>'
        '</visual>'
        '<collision name="collision">'
        '<geometry><sphere>'
        f'<radius>{args.radius}</radius>'
        '</sphere></geometry>'
        '</collision>'
        '</link>'
        '</model>'
        '</sdf>'
    )

    subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{args.world}/remove",
            "--reqtype",
            "gz.msgs.Entity",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "1000",
            "--req",
            f'name: "{args.name}" type: MODEL',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    run_gz_service(
        args.world,
        "create",
        "gz.msgs.EntityFactory",
        f'sdf: "{escape_sdf(sdf)}"',
        timeout="8000",
    )


def target_pose(args, t, forward, lateral):
    along = args.speed * t
    side = 0.0
    z_offset = 0.0

    if args.mode == "arc":
        arc_radius = max(args.arc_radius, 0.1)
        direction = 1.0 if args.arc_direction == "left" else -1.0
        theta = along / arc_radius
        along = arc_radius * math.sin(theta)
        side = direction * arc_radius * (1.0 - math.cos(theta))

    if args.mode == "s_curve":
        side = args.s_amp_y * math.sin(2.0 * math.pi * t / max(args.s_period, 0.1))
        z_offset = args.z_amp * math.sin(2.0 * math.pi * t / max(args.z_period, 0.1))

    x = args.start_x + forward[0] * along + lateral[0] * side
    y = args.start_y + forward[1] * along + lateral[1] * side
    z = args.start_z + z_offset
    return x, y, z


def set_pose(world, name, x, y, z):
    req = f'name: "{name}" position {{x: {x:.6f} y: {y:.6f} z: {z:.6f}}} orientation {{w: 1}}'
    run_gz_service(world, "set_pose", "gz.msgs.Pose", req, timeout="2000", quiet=True)


def run_gz_service(world, service, req_type, req, timeout="1000", quiet=False):
    cmd = [
        "gz",
        "service",
        "-s",
        f"/world/{world}/{service}",
        "--reqtype",
        req_type,
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        timeout,
        "--req",
        req,
    ]
    stdout = subprocess.DEVNULL if quiet else None
    result = subprocess.run(cmd, stdout=stdout)
    if result.returncode != 0 and not quiet:
        raise SystemExit(f"gz service failed: {' '.join(cmd)}")


def escape_sdf(sdf):
    return sdf.replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    main()
